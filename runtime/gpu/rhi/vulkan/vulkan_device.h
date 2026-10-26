// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// The Vulkan implementation of the RHI (gpu/rhi/device.h).
//
// The files: vulkan_device.cpp (the device and its creation), vulkan_commands.cpp
// (the command list), vulkan_swapchain.cpp (the window's surface and the
// swapchain), and vulkan_internal.h, the types they share. What the presenter
// and renderer may reach into is vulkan_native.h.
#pragma once
#include <memory>

#include "../device.h"

namespace gpu::rhi::vulkan
{
    // A device of its own, headless (no surface: nothing here presents): the
    // first discrete GPU, else the first device. nullptr when there is no
    // usable Vulkan device. For tests and tools; the game's presenter creates
    // its device with CreateDevice (vulkan_native.h).
    std::unique_ptr<Device> CreateHeadlessDevice();
}
