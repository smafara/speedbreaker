// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// The Metal implementation of the RHI (gpu/rhi/device.h), through metal-cpp
// (DECISIONS.md D13). Apple platforms; built with -DSB_METAL=ON.
#pragma once
#include <memory>

#include "../device.h"

namespace gpu::rhi::metal
{
    // The system's default Metal device (a window is metal_native.h's CreateSurface).
    // nullptr when there is none.
    std::unique_ptr<Device> CreateDevice();
}
