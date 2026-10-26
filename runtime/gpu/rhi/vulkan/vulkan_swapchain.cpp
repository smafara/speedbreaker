// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
// The window's surface and the swapchain (gpu/rhi/swapchain.h).
#include "vulkan_internal.h"

namespace gpu::rhi::vulkan
{
    namespace impl
    {
        namespace
        {
            // ---- the window (swapchain.h) -----------------------------------------------

            class VkSurfaceObject final : public Surface
            {
            public:
                VkSurfaceObject(Context& c, VkSurfaceKHR surface, std::function<Extent3D()> windowPixels)
                    : m_c(c), m_surface(surface), m_windowPixels(std::move(windowPixels)) {}
                Extent3D CurrentExtent() override
                {
                    VkSurfaceCapabilitiesKHR caps{};
                    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_c.physical, m_surface, &caps) != VK_SUCCESS)
                        return { 0, 0, 1 };
                    if (caps.currentExtent.width != UINT32_MAX)
                        return { caps.currentExtent.width, caps.currentExtent.height, 1 };
                    return m_windowPixels ? m_windowPixels() : Extent3D{ 0, 0, 1 };
                }
                VkSurfaceKHR Handle() const { return m_surface; }

            private:
                Context& m_c;
                VkSurfaceKHR m_surface;
                std::function<Extent3D()> m_windowPixels;
            };

    #if defined(__APPLE__) && !TARGET_OS_IOS
            // Metal's host time is mach_absolute_time's (CLOCK_UPTIME_RAW); the
            // steady clock goes on through sleep. Steady = host + this.
            int64_t HostToSteadyNs()
            {
                return NowNs() - int64_t(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
            }
    #endif

            // The swapchain, the presents in flight, and (when the caller wants
            // present times) the thread that waits for each to reach the screen.
            // The thread is the swapchain's: it is joined before the swapchain is
            // destroyed, which replaces video/swapchain_use.h's generations.
            class VkSwapchainObject final : public Swapchain
            {
            public:
                VkSwapchainObject(Context& c, VkSurfaceObject& surface, const SwapchainDesc& desc) : m_c(c), m_surface(surface), m_desc(desc) {}

                ~VkSwapchainObject() override
                {
                    if (m_waiter.joinable())
                    {
                        {
                            std::lock_guard lock(m_mutex);
                            m_stop = true;
                        }
                        m_cv.notify_all();
                        m_waiter.join();
                    }
                    {
                        std::lock_guard lock(m_c.queueMutex);
                        vkDeviceWaitIdle(m_c.device);
                    }
                    for (VkSemaphore s : m_acquire)
                        vkDestroySemaphore(m_c.device, s, nullptr);
                    for (VkSemaphore s : m_renderDone)
                        vkDestroySemaphore(m_c.device, s, nullptr);
                    m_images.clear();
                    if (m_swapchain)
                        vkDestroySwapchainKHR(m_c.device, m_swapchain, nullptr);
                }

                bool Create()
                {
                    VkPhysicalDevice physical = m_c.physical;
                    VkSurfaceKHR surface = m_surface.Handle();
                    VkSurfaceCapabilitiesKHR caps;
                    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps);
                    uint32_t count = 0;
                    vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, nullptr);
                    std::vector<VkSurfaceFormatKHR> formats(count);
                    vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, formats.data());
                    // The preferred format in plain sRGB: the picture is sRGB-encoded.
                    // MoltenVK lists each format in every color space it knows (asked
                    // for VK_EXT_swapchain_colorspace or not), and the last
                    // B8G8R8A8_UNORM entry is Display P3 *linear*: the layer showed
                    // the picture's values as linear light, washed out and pale
                    // (captures of the swapchain looked right: its values were).
                    auto pick = [&](Format format) {
                        return std::find_if(formats.begin(), formats.end(), [&](const VkSurfaceFormatKHR& f) {
                            return (format == Format::Undefined ? FromVk(f.format) != Format::Undefined : FromVk(f.format) == format) &&
                                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
                        });
                    };
                    auto chosen = pick(m_desc.preferredFormat);
                    if (chosen == formats.end())
                        chosen = pick(Format::Undefined);
                    if (chosen == formats.end())
                        chosen = formats.begin();
                    m_format = chosen->format;
                    static bool formatLogged = false;
                    if (!std::exchange(formatLogged, true))
                        fprintf(stderr, "[video] swapchain format %d, color space %d (%zu offered)\n", int(m_format), int(chosen->colorSpace),
                            formats.size());

                    uint32_t modeCount = 0;
                    vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &modeCount, nullptr);
                    std::vector<VkPresentModeKHR> modes(modeCount);
                    vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &modeCount, modes.data());
                    // V-Sync off: mailbox (no tearing, newest frame) where there is
                    // one, else immediate.
                    m_mode = VK_PRESENT_MODE_FIFO_KHR;
                    if (!m_desc.vsync)
                        for (VkPresentModeKHR wanted : { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR })
                            if (std::find(modes.begin(), modes.end(), wanted) != modes.end())
                            {
                                m_mode = wanted;
                                break;
                            }

                    VkSwapchainCreateInfoKHR info{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
                    info.surface = surface;
                    info.minImageCount = std::max(caps.minImageCount, m_desc.minImages);
                    if (caps.maxImageCount)
                        info.minImageCount = std::min(info.minImageCount, caps.maxImageCount);
                    info.imageFormat = m_format;
                    info.imageColorSpace = chosen->colorSpace;
                    info.imageExtent = { m_desc.extent.width, m_desc.extent.height };
                    info.imageArrayLayers = 1;
                    info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
                    m_readable = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
                    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
                    info.preTransform = caps.currentTransform;
                    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
                    info.presentMode = m_mode;
                    info.clipped = VK_TRUE;
                    if (VkResult r = vkCreateSwapchainKHR(m_c.device, &info, nullptr, &m_swapchain); r != VK_SUCCESS)
                    {
                        fprintf(stderr, "[rhi/vulkan] vkCreateSwapchainKHR failed (%d)\n", int(r));
                        return false;
                    }
                    m_extent = m_desc.extent;

                    vkGetSwapchainImagesKHR(m_c.device, m_swapchain, &count, nullptr);
                    std::vector<VkImage> images(count);
                    vkGetSwapchainImagesKHR(m_c.device, m_swapchain, &count, images.data());
                    ImageDesc id;
                    id.format = FromVk(m_format);
                    id.extent = m_extent;
                    id.usage = kImageRenderTarget | kImageCopyDst | (m_readable ? kImageCopySrc : 0);
                    id.name = "swapchain";
                    VkSemaphoreCreateInfo si{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
                    for (VkImage image : images)
                    {
                        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
                        vci.image = image;
                        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                        vci.format = m_format;
                        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                        VkImageView view = VK_NULL_HANDLE;
                        vkCreateImageView(m_c.device, &vci, nullptr, &view);
                        m_images.push_back(std::make_unique<VkImageObject>(m_c, id, image, VK_NULL_HANDLE, view, 0));
                        VkSemaphore done, acquire;
                        vkCreateSemaphore(m_c.device, &si, nullptr, &done);
                        vkCreateSemaphore(m_c.device, &si, nullptr, &acquire);
                        m_renderDone.push_back(done);
                        m_acquire.push_back(acquire);
                    }

                    {
                        std::string offered;
                        for (VkPresentModeKHR m : modes)
                            offered += std::string(offered.empty() ? "" : ", ") + (m == VK_PRESENT_MODE_IMMEDIATE_KHR ? "IMMEDIATE"
                                : m == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" : m == VK_PRESENT_MODE_FIFO_KHR ? "FIFO"
                                : m == VK_PRESENT_MODE_FIFO_RELAXED_KHR ? "FIFO_RELAXED" : std::to_string(int(m)));
                        m_describe = std::string(PresentModeName()) + ", " + std::to_string(count) + " images (the surface offers " + offered +
                            ", and " + (caps.maxImageCount ? std::to_string(caps.minImageCount) + " to " + std::to_string(caps.maxImageCount)
                                                           : std::to_string(caps.minImageCount) + " or more") + " images)";
                    }
                    if (m_desc.onShown && m_c.waitForPresent)
                        m_waiter = std::thread([this] { Waiter(); });
                    return true;
                }

                Extent3D Extent() const override { return m_extent; }
                Format GetFormat() const override { return FromVk(m_format); }
                VkFormat VkFormatOf() const { return m_format; }
                uint32_t ImageCount() const override { return uint32_t(m_images.size()); }
                bool Readable() const override { return m_readable; }
                const char* PresentModeName() const override
                {
                    return m_mode == VK_PRESENT_MODE_FIFO_KHR ? "FIFO" : m_mode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" : "IMMEDIATE";
                }
                std::string Describe() const override { return m_describe; }

                SurfaceStatus Acquire(Image*& image) override
                {
                    image = nullptr;
                    VkSemaphore semaphore = m_acquire[m_acquireNext];
                    VkResult r = vkAcquireNextImageKHR(m_c.device, m_swapchain, UINT64_MAX, semaphore, VK_NULL_HANDLE, &m_current);
                    if (r == VK_ERROR_OUT_OF_DATE_KHR)
                        return SurfaceStatus::OutOfDate;
                    m_acquireNext = (m_acquireNext + 1) % m_acquire.size();
                    m_waitSemaphore = semaphore;
                    image = m_images[m_current].get();
                    return r == VK_SUBOPTIMAL_KHR ? SurfaceStatus::Suboptimal : SurfaceStatus::Ok;
                }

                // What the submission that draws the acquired image waits for and signals.
                void SubmitSemaphores(VkSemaphore& wait, VkSemaphore& signal) const
                {
                    wait = m_waitSemaphore;
                    signal = m_renderDone[m_current];
                }

                SurfaceStatus Present(const PresentRequest& request) override
                {
                    VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
                    pi.waitSemaphoreCount = 1;
                    pi.pWaitSemaphores = &m_renderDone[m_current];
                    pi.swapchainCount = 1;
                    pi.pSwapchains = &m_swapchain;
                    pi.pImageIndices = &m_current;
                    const int64_t presented = request.presented ? request.presented : NowNs();
                    const bool timed = m_waiter.joinable() && request.id;
                    VkPresentIdKHR presentId{ VK_STRUCTURE_TYPE_PRESENT_ID_KHR };
                    uint64_t id = request.id;
                    if (timed)
                    {
                        presentId.swapchainCount = 1;
                        presentId.pPresentIds = &id;
                        pi.pNext = &presentId;
                    }
    #if defined(__APPLE__) && !TARGET_OS_IOS
                    // Only a frame shown at a time carries the driver's present
                    // times (and, for the diagnostics, every present): the rest are
                    // presented plainly.
                    VkPresentTimeGOOGLE presentTime{};
                    VkPresentTimesInfoGOOGLE presentTimes{ VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE };
                    if (m_c.pastTiming && request.id)
                    {
                        presentTime.presentID = uint32_t(request.id);
                        if (request.notBefore)
                            presentTime.desiredPresentTime = uint64_t(std::max<int64_t>(request.notBefore - HostToSteadyNs(), 0));
                        if (presentTime.desiredPresentTime || m_desc.reportAll)
                        {
                            presentTimes.swapchainCount = 1;
                            presentTimes.pTimes = &presentTime;
                            presentTimes.pNext = pi.pNext;
                            pi.pNext = &presentTimes;
                        }
                    }
    #endif
                    VkResult r;
                    {
                        std::lock_guard lock(m_c.queueMutex);
                        r = vkQueuePresentKHR(m_c.queue, &pi);
                    }
                    if (timed && (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR))
                    {
                        {
                            std::lock_guard lock(m_mutex);
                            m_pending.push_back({ request.id, presented });
                        }
                        m_cv.notify_one();
                    }
                    return r == VK_ERROR_OUT_OF_DATE_KHR ? SurfaceStatus::OutOfDate : r == VK_SUBOPTIMAL_KHR ? SurfaceStatus::Suboptimal
                                                                                                              : SurfaceStatus::Ok;
                }

            private:
                struct Pending { uint64_t id; int64_t presented; };

                // The driver's reports on the presents shown at a time, since the
                // last call (Metal's presentedTime: 0 when the drawable was never shown).
                void ReadReports()
                {
    #if defined(__APPLE__) && !TARGET_OS_IOS
                    if (!m_c.pastTiming || !m_desc.onReport)
                        return;
                    uint32_t count = 0;
                    std::vector<VkPastPresentationTimingGOOGLE> past;
                    if (m_c.pastTiming(m_c.device, m_swapchain, &count, nullptr) == VK_SUCCESS && count)
                    {
                        past.resize(count);
                        if (m_c.pastTiming(m_c.device, m_swapchain, &count, past.data()) < VK_SUCCESS)
                            count = 0;
                        past.resize(count);
                    }
                    const int64_t hostToSteady = HostToSteadyNs();
                    for (const VkPastPresentationTimingGOOGLE& e : past)
                        m_desc.onReport({ e.presentID, e.actualPresentTime ? int64_t(e.actualPresentTime) + hostToSteady : 0 });
    #endif
                }

                void Waiter()
                {
                    if (m_desc.onThreadStart)
                        m_desc.onThreadStart();
                    while (true)
                    {
                        Pending t;
                        {
                            std::unique_lock lock(m_mutex);
                            m_cv.wait(lock, [&] { return m_stop || !m_pending.empty(); });
                            if (m_stop)
                                return;
                            t = m_pending.front();
                        }
                        const int64_t start = NowNs();
                        VkResult r = m_c.waitForPresent(m_c.device, m_swapchain, t.id, 50'000'000);
                        const int64_t now = NowNs();
                        {
                            std::lock_guard lock(m_mutex);
                            if (m_stop)
                                return;
                            // Not on screen yet: wait again. A hidden window may show
                            // nothing for a long time; past 8 waiting, drop the oldest.
                            if (r == VK_TIMEOUT && m_pending.size() <= 8)
                                continue;
                            if (!m_pending.empty() && m_pending.front().id == t.id)
                                m_pending.pop_front();
                        }
                        if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR)
                        {
                            ReadReports();
                            m_desc.onShown({ t.id, t.presented, now, now - start > 50'000 });
                        }
                    }
                }

                Context& m_c;
                VkSurfaceObject& m_surface;
                SwapchainDesc m_desc;
                VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
                VkFormat m_format = VK_FORMAT_UNDEFINED;
                VkPresentModeKHR m_mode = VK_PRESENT_MODE_FIFO_KHR;
                Extent3D m_extent;
                bool m_readable = false;
                std::string m_describe;
                std::vector<std::unique_ptr<VkImageObject>> m_images;
                std::vector<VkSemaphore> m_acquire, m_renderDone;  // one each per image
                size_t m_acquireNext = 0;
                uint32_t m_current = 0;
                VkSemaphore m_waitSemaphore = VK_NULL_HANDLE;
                std::mutex m_mutex;
                std::condition_variable m_cv;
                std::deque<Pending> m_pending;  // presented, not yet seen on screen
                bool m_stop = false;
                std::thread m_waiter;
            };
        }

        std::unique_ptr<Surface> MakeSurface(Context& c, VkSurfaceKHR surface, std::function<Extent3D()> windowPixels)
        {
            return std::make_unique<VkSurfaceObject>(c, surface, std::move(windowPixels));
        }
        std::unique_ptr<Swapchain> MakeSwapchain(Context& c, Surface& surface, const SwapchainDesc& desc)
        {
            auto swapchain = std::make_unique<VkSwapchainObject>(c, static_cast<VkSurfaceObject&>(surface), desc);
            return swapchain->Create() ? std::move(swapchain) : nullptr;
        }
        void SwapchainSubmitSemaphores(Swapchain& swapchain, VkSemaphore& wait, VkSemaphore& signal)
        {
            static_cast<VkSwapchainObject&>(swapchain).SubmitSemaphores(wait, signal);
        }
        VkFormat SwapchainFormat(const Swapchain& swapchain) { return static_cast<const VkSwapchainObject&>(swapchain).VkFormatOf(); }
    }
}
