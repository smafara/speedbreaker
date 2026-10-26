// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// The Vulkan backend's way in for code that still speaks Vulkan: the game's
// presenter (which chooses the device and its extensions, and owns the
// window's surface) and the renderer, until they have moved onto the RHI.
// Nothing outside gpu/rhi/vulkan and these two should include this.
#pragma once
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <vulkan/vulkan.h>

#include "../command_list.h"
#include "../device.h"

namespace gpu::rhi::vulkan
{
    // What the caller decided, in Vulkan terms: this is the policy half of
    // device creation (which GPU, which extensions and features, per the
    // settings and the platform); the backend does the rest (the queue, the
    // memory properties, the pipeline cache, the capabilities) and owns the
    // device from then on.
    struct DeviceConfig
    {
        VkInstance instance = VK_NULL_HANDLE;   // the backend destroys it with the device
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        uint32_t queueFamily = 0;               // one queue of it, graphics + compute
        std::vector<const char*> extensions;
        const VkPhysicalDeviceFeatures* features = nullptr;
        const void* next = nullptr;             // VkDeviceCreateInfo's pNext chain (feature structs)
    };

    // nullptr (and the Vulkan error logged) if the device can't be made. On
    // failure `instance` stays the caller's.
    std::unique_ptr<Device> CreateDevice(const DeviceConfig& config);

    // The handles of a device made here. `queueMutex` guards every
    // vkQueueSubmit / vkQueuePresentKHR / vkQueueWaitIdle on `queue`.
    struct Native
    {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        uint32_t queueFamily = 0;
        VkQueue queue = VK_NULL_HANDLE;
        std::mutex* queueMutex = nullptr;
    };
    const Native& GetNative(const rhi::Device& device);

    // The window's surface (made by the caller from SDL; the backend neither
    // owns nor destroys it). `windowPixels` answers where the surface leaves
    // its size to the swapchain (Wayland's currentExtent is 0xFFFFFFFF).
    // The Vulkan objects behind an object of the RHI, for the presenter's own
    // recording until its passes are on the RHI (a list's command buffer is
    // recorded into between its Begin and End, mixed with the list's calls).
    VkImage NativeImage(const Image& image);
    VkImageView NativeView(const Image& image);
    VkFormat NativeFormat(const Swapchain& swapchain);
    VkCommandBuffer NativeCommandBuffer(const CommandList& list);
    // An image of someone else's (the renderer's front image, until the
    // renderer is on the RHI) as an RHI image, with a view of its own made
    // here: the image is not destroyed with it, and the caller keeps the
    // layout it is in. Destroy it before the image goes.
    std::unique_ptr<Image> WrapImage(Device& device, VkImage image, Format format, uint32_t width, uint32_t height, uint32_t usage);

    std::unique_ptr<Surface> CreateSurface(Device& device, VkSurfaceKHR surface, std::function<Extent3D()> windowPixels);
}
