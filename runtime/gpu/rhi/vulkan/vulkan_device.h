// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// The Vulkan implementation of the RHI (gpu/rhi/device.h).
//
// STATE (DECISIONS.md, presenter refactor): everything in the draft interface
// is implemented and tested (tests/rhi_vulkan_test.cpp): buffers, imported
// host memory, images, samplers, binding layouts and sets, compute and
// graphics pipelines, command lists, fences, events and queries. Not yet:
// the swapchain (next), and the renderer's own needs beyond the draft (image
// views in other formats, memory block suballocation: see device.h).
#pragma once
#include <memory>

#include "../device.h"

namespace gpu::rhi::vulkan
{
    // A device of its own, headless (no surface: nothing here presents): the
    // first discrete GPU, else the first device. nullptr when there is no
    // usable Vulkan device. For tests and tools; the game's presenter creates
    // its device itself until it moves onto this backend (step 2).
    std::unique_ptr<Device> CreateHeadlessDevice();
}
