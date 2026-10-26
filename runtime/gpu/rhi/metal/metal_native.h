// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// What the presenter hands the Metal backend that the RHI has no word for: the
// window's layer.
#pragma once
#include <memory>

#include "../swapchain.h"

namespace gpu::rhi::metal
{
    // The surface over a CAMetalLayer (what SDL_Metal_GetLayer returns for a
    // window made with SDL_WINDOW_METAL). The device must be a Metal one.
    std::unique_ptr<Surface> CreateSurface(Device& device, void* caMetalLayer);
}
