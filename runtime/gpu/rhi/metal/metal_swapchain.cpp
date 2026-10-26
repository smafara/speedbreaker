// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
// The window's layer and swapchain (gpu/rhi/swapchain.h); see metal_device.cpp.
#include "metal_internal.h"

#include <algorithm>
#include <atomic>
#include <thread>

#include "metal_native.h"

namespace gpu::rhi::metal
{
    namespace impl
    {
        namespace
        {
            // ---- the window ------------------------------------------------------------

            class MetalSurface final : public Surface
            {
            public:
                explicit MetalSurface(CA::MetalLayer* layer) : m_layer(Keep(layer)) {}
                Extent3D CurrentExtent() override
                {
                    const CGSize size = m_layer->drawableSize();
                    return { uint32_t(std::max(size.width, 0.0)), uint32_t(std::max(size.height, 0.0)), 1 };
                }
                CA::MetalLayer* Layer() const { return m_layer.get(); }

            private:
                Ptr<CA::MetalLayer> m_layer;
            };

            // Drawables come from the layer; a present is a command buffer of its own
            // on the queue (after the one that drew, so it waits for it) that shows
            // the drawable, at a time when asked. The presented handler is both
            // callbacks of swapchain.h: it has the display's own time of the frame.
            class MetalSwapchain final : public Swapchain
            {
            public:
                MetalSwapchain(Context& c, MetalSurface& surface, const SwapchainDesc& desc)
                    : m_c(c), m_layer(Keep(surface.Layer())), m_desc(desc), m_extent(desc.extent)
                {
                    m_layer->setDevice(c.device.get());
                    m_layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
                    m_layer->setFramebufferOnly(!desc.readable);
                    m_layer->setDrawableSize(CGSize{ double(desc.extent.width), double(desc.extent.height) });
                    m_layer->setMaximumDrawableCount(std::clamp<uint32_t>(desc.minImages, 2, 3));
                    m_layer->setDisplaySyncEnabled(desc.vsync);
                    m_layer->setAllowsNextDrawableTimeout(false);  // Acquire blocks until one is free
                }

                ~MetalSwapchain() override
                {
                    Pool pool;
                    std::unique_lock lock(m_mutex);
                    if (!m_idle.wait_for(lock, std::chrono::seconds(2), [this] { return m_inFlight == 0; }))
                        fprintf(stderr, "[rhi/metal] %u presents never reported; going on\n", m_inFlight);
                }

                Extent3D Extent() const override { return m_extent; }
                Format GetFormat() const override { return Format::BGRA8; }
                uint32_t ImageCount() const override { return std::clamp<uint32_t>(m_desc.minImages, 2, 3); }
                bool Readable() const override { return m_desc.readable; }
                const char* PresentModeName() const override { return m_desc.vsync ? "displaySync" : "immediate"; }
                std::string Describe() const override
                {
                    return std::string("CAMetalLayer, BGRA8, ") + PresentModeName() + ", " + std::to_string(ImageCount()) + " drawables";
                }

                SurfaceStatus Acquire(Image*& image) override
                {
                    Pool pool;
                    image = nullptr;
                    m_image.reset();
                    m_drawable = Keep(m_layer->nextDrawable());
                    if (!m_drawable)
                        return SurfaceStatus::OutOfDate;
                    MTL::Texture* texture = m_drawable->texture();
                    if (texture->width() != m_extent.width || texture->height() != m_extent.height)
                    {
                        m_drawable.reset();  // the window changed size: the policy builds it again
                        return SurfaceStatus::OutOfDate;
                    }
                    ImageDesc d;
                    d.format = Format::BGRA8;
                    d.extent = m_extent;
                    d.usage = kImageRenderTarget | kImageCopyDst | (m_desc.readable ? kImageCopySrc : 0);
                    d.name = "swapchain image";
                    m_image = std::make_unique<MetalImage>(d, Keep(texture), 0);
                    image = m_image.get();
                    return SurfaceStatus::Ok;
                }

                SurfaceStatus Present(const PresentRequest& request) override
                {
                    Pool pool;
                    if (!m_drawable)
                        return SurfaceStatus::OutOfDate;
                    const TimeNs presented = request.presented ? request.presented : NowNs();
                    const bool timed = request.id && (m_desc.onShown || m_desc.onReport);
                    if (timed)
                    {
                        uint64_t sequence;
                        {
                            std::lock_guard lock(m_mutex);
                            m_inFlight++;
                            sequence = m_nextSequence++;
                        }
                        const uint64_t id = request.id;
                        const bool wantReport = request.notBefore || m_desc.reportAll;
                        m_drawable->addPresentedHandler([this, sequence, id, presented, wantReport](MTL::Drawable* d) {
                            const double t = d->presentedTime();  // seconds on the uptime clock; 0: dropped
                            const TimeNs onScreen = t > 0 ? TimeNs(t * 1e9) + SteadyMinusUptime() : 0;
                            {
                                // The handlers can run on different threads, out of order: the
                                // callbacks are in present order, one at a time.
                                std::lock_guard lock(m_callbackMutex);
                                m_finished[sequence] = Finished{ id, presented, onScreen, wantReport };
                                while (!m_finished.empty() && m_finished.begin()->first == m_deliverNext)
                                {
                                    const Finished f = m_finished.begin()->second;
                                    m_finished.erase(m_finished.begin());
                                    m_deliverNext++;
                                    thread_local const void* started = nullptr;
                                    if (started != this)
                                    {
                                        started = this;
                                        if (m_desc.onThreadStart)
                                            m_desc.onThreadStart();
                                    }
                                    if (f.onScreen && m_desc.onShown)
                                        m_desc.onShown(PresentShown{ f.id, f.presented, f.onScreen, true });
                                    if (f.wantReport && m_desc.onReport)
                                        m_desc.onReport(PresentReport{ f.id, f.onScreen });
                                }
                            }
                            std::lock_guard lock(m_mutex);
                            m_inFlight--;
                            m_idle.notify_all();
                        });
                    }
                    MTL::CommandBuffer* cb = m_c.queue->commandBuffer();
                    if (request.notBefore)
                        cb->presentDrawableAtTime(m_drawable.get(), double(request.notBefore - SteadyMinusUptime()) * 1e-9);
                    else
                        cb->presentDrawable(m_drawable.get());
                    cb->commit();
                    m_drawable.reset();
                    m_image.reset();
                    return SurfaceStatus::Ok;
                }

            private:
                // Metal's times are mach_absolute_time's, which stops while the Mac
                // sleeps; std::chrono::steady_clock here keeps going (it is
                // mach_continuous_time). The difference, now.
                static TimeNs SteadyMinusUptime()
                {
                    return TimeNs(clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)) - TimeNs(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
                }
                static TimeNs NowNs()
                {
                    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                }

                Context& m_c;
                Ptr<CA::MetalLayer> m_layer;
                SwapchainDesc m_desc;
                Extent3D m_extent;
                Ptr<CA::MetalDrawable> m_drawable;
                std::unique_ptr<MetalImage> m_image;
                std::mutex m_mutex;          // m_inFlight
                std::condition_variable m_idle;
                uint32_t m_inFlight = 0;
                std::mutex m_callbackMutex;  // the callbacks come one at a time
                struct Finished { uint64_t id; TimeNs presented, onScreen; bool wantReport; };
                uint64_t m_nextSequence = 0;                // (under m_mutex) given to each timed present
                uint64_t m_deliverNext = 0;                 // (under m_callbackMutex) the next to call back
                std::map<uint64_t, Finished> m_finished;    // (same) done, waiting for its turn
            };
        }

        std::unique_ptr<Swapchain> MakeSwapchain(Context& c, Surface& surface, const SwapchainDesc& desc)
        {
            return std::make_unique<MetalSwapchain>(c, static_cast<MetalSurface&>(surface), desc);
        }
    }

    std::unique_ptr<Surface> CreateSurface(Device& device, void* caMetalLayer)
    {
        (void)device;
        impl::Pool pool;
        return std::make_unique<impl::MetalSurface>(static_cast<CA::MetalLayer*>(caMetalLayer));
    }
}
