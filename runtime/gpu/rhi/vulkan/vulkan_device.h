// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// The Vulkan implementation of the RHI (gpu/rhi/device.h).
//
// STATE (DECISIONS.md, presenter refactor step 1): buffers, imported host
// memory, samplers, binding layouts and sets, compute pipelines, command
// lists (buffer work, compute, state, queries, events), fences and the
// queue are implemented and tested (tests/rhi_vulkan_test.cpp). Images,
// graphics pipelines and rendering passes, which the presenter's post-
// processing needs next, abort with "not implemented yet" until they are
// written; so does anything else the draft interface has and no caller needs.
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
