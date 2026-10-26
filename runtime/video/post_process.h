// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// The presenter's post-processing on the RHI (gpu/rhi): anti-aliasing at the
// game's resolution (compute), then the picture scaled into the letterboxed
// part of the window (fragment; bilinear, bicubic, or FSR 1's EASU and RCAS),
// through the tone table when Brightness or the 360's display gamma is in use.
// The 360 drew NFSMW with 2x MSAA; the renderer draws single-sampled, so
// edges get an FXAA-style pass (edge-directed, with a sub-pixel blend).
// Scaling is Catmull-Rom bicubic with optional sharpening, clamped to the 2x2
// source texels around each sample so it can't ring.
// NFSMW_AA=0 turns the anti-aliasing off; NFSMW_SCALING=linear (or bicubic, the
// default); NFSMW_SHARPEN=<0..1> (default 0.3). The shaders are GLSL, compiled
// when first needed and handed to the device as SPIR-V (DECISIONS.md D12).
//
// Images the caller gives are in the ShaderRead state. Calls outside a
// rendering pass say so; DrawPicture is inside one, on a target of the
// format Ready() was given.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>

#include <gpu/rhi/command_list.h>
#include <gpu/rhi/device.h>

#include "picture_fit.h"
#include "upscale.h"

namespace video
{
    class PostProcessor
    {
    public:
        // `waitIdle`: waits for every frame in flight (the images and the
        // tone table are replaced only when none can be reading them).
        PostProcessor(gpu::rhi::Device& device, std::function<void()> waitIdle);
        ~PostProcessor();

        // The pipelines for a target of `format` (made again if it changes).
        // False: they can't be made, and the caller blits the frame instead.
        bool Ready(gpu::rhi::Format format);

        // Settings, read every frame (the menu changes them live).
        static bool AntiAliasing();
        static float Sharpen();

        // Anti-aliasing: `source` (w x h) into an image of the module's, which
        // is returned in the ShaderRead state. Outside a pass.
        gpu::rhi::Image& RecordAntiAliasing(gpu::rhi::CommandList& list, gpu::rhi::Image& source, uint32_t w, uint32_t h,
            const FrameBand& band, bool bandOn);

        // Before the scale pass: whether this frame's picture goes through the
        // tone table, with the table brought up to date when its inputs
        // changed. False at Brightness 0 with the gamma off. Outside a pass.
        bool PrepareTone(gpu::rhi::CommandList& list);

        // The pass a frame takes (video/upscale.h), FSR 1 only where its
        // pipelines are ready (made the first time); a line in the log when
        // the pass or the sizes change.
        upscale::Pass ChoosePass(bool tone, uint32_t w, uint32_t h, int32_t dw, int32_t dh);
        // The same for Settings' scaling value `mode`, silently.
        upscale::Pass PassFor(int32_t mode, bool tone, uint32_t w, uint32_t h, int32_t dw, int32_t dh);

        // FSR 1's EASU: the w x h `input` enlarged to dw x dh, for RCAS's reads
        // in DrawPicture. Outside a pass.
        void RecordEasu(gpu::rhi::CommandList& list, gpu::rhi::Image& input, uint32_t w, uint32_t h, int32_t dw, int32_t dh,
            const FrameBand& band, bool bandOn);

        // Inside a pass: `input` at dx, dy, dw x dh by `pass` (FSR 1: RCAS over
        // RecordEasu's image), through the tone table when `tone`.
        void DrawPicture(gpu::rhi::CommandList& list, upscale::Pass pass, bool tone, gpu::rhi::Image& input, uint32_t w, uint32_t h,
            const FrameBand& band, bool bandOn, int32_t dx, int32_t dy, int32_t dw, int32_t dh, float sharpen);

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
