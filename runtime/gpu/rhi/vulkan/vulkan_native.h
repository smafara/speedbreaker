// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// The Vulkan backend's way in for code that still speaks Vulkan: the game's
// presenter (which chooses the device and its extensions, and owns the
// window's surface) and the renderer, until they have moved onto the RHI.
// Nothing outside gpu/rhi/vulkan and these two should include this.
#pragma once
#include <memory>
#include <mutex>
#include <vector>

#include <vulkan/vulkan.h>

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
}
