// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// The render hardware interface (DECISIONS.md D3, D8-D10): what the renderer
// and the presenter ask of a graphics API, at the level of commands (buffers,
// images, pipelines, command lists, a queue with fences) and no lower. The
// Xenos logic (PM4, shader translation, texture and render-target caching,
// resolves) stays above it and is shared by every backend.
//
// DRAFT for review: nothing includes these headers yet. The formats, usages
// and states are the ones renderer.cpp and presenter.cpp use today, not a
// general graphics API's; add one when a caller needs it.
#pragma once
#include <cstdint>

namespace gpu::rhi
{
    // The formats the renderer and presenter create images in.
    enum class Format : uint8_t
    {
        Undefined,
        R8, RG8, RGBA8, BGRA8, RGBA8_sRGB, BGRA8_sRGB,
        R16, RG16, RGBA16,              // UNORM
        RG16_SNORM, RGBA16_SNORM,
        R16F, RG16F, RGBA16F,
        R32F, RG32F, RGBA32F, R32_UINT,
        A2B10G10R10,                    // 2_10_10_10 UNORM
        D32F_S8,                        // depth 32 float + stencil 8
        BC1, BC2, BC3, BC5,             // UNORM blocks
    };

    // Where a buffer's memory is, for the CPU's and the GPU's use (memory_plan.h
    // chooses among these today by what the device offers).
    enum class MemoryKind : uint8_t
    {
        Device,       // GPU only; filled by copies
        Upload,       // CPU-mapped, write-combined is fine; the GPU reads (rings, staging)
        UploadCached, // CPU-mapped and cached: the CPU also reads it (shadow copies, readback)
        Stream,       // written by the CPU every frame and read by the GPU (the renderer's ring): where the device likes that best
    };

    enum BufferUsage : uint32_t
    {
        kBufferStorage   = 1 << 0,
        kBufferUniform   = 1 << 1,
        kBufferIndex     = 1 << 2,
        kBufferCopySrc   = 1 << 3,
        kBufferCopyDst   = 1 << 4,
    };

    enum ImageUsage : uint32_t
    {
        kImageSampled      = 1 << 0,
        kImageStorage      = 1 << 1,
        kImageRenderTarget = 1 << 2,
        kImageDepthStencil = 1 << 3,
        kImageCopySrc      = 1 << 4,
        kImageCopyDst      = 1 << 5,
    };

    enum class ImageKind : uint8_t { Image2D, Image3D, Cube };

    // D9: what a resource is about to be used as. The Vulkan backend maps a
    // state to layout + stages + access; Metal tracks hazards itself and
    // needs a fence only where it can't (see CommandList::Transition).
    enum class State : uint8_t
    {
        Undefined,         // contents don't matter (a new image, or one about to be overwritten whole)
        ShaderRead,        // sampled image / read-only buffer
        ShaderWrite,       // storage image / buffer written (and read) by shaders
        RenderTarget,
        DepthRead,
        DepthWrite,
        CopySrc,
        CopyDst,
        IndexRead,         // buffers
        Present,           // swapchain images
        HostRead,          // buffers: the CPU reads it after the GPU's writes
        General,           // images made `general` (ImageDesc): usable as anything; ordered by CommandList::MemoryBarrier
    };

    enum class ShaderStage : uint8_t { Vertex = 1, Fragment = 2, Compute = 4 };
    inline uint8_t operator|(ShaderStage a, ShaderStage b) { return uint8_t(a) | uint8_t(b); }

    enum class Filter : uint8_t { Nearest, Linear };
    enum class AddressMode : uint8_t { Repeat, ClampToEdge, MirrorRepeat, ClampToBorder };
    // A view's component: a channel of the image, or a constant.
    enum class Swizzle : uint8_t { Identity, R, G, B, A, Zero, One };

    // (No triangle fan: the renderer converts fans and polygons to lists, and Metal has no fan.)
    enum class Topology : uint8_t { PointList, LineList, LineStrip, TriangleList, TriangleStrip };
    enum class CullMode : uint8_t { None, Front, Back };
    enum class CompareOp : uint8_t { Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always };
    enum class StencilOp : uint8_t { Keep, Zero, Replace, IncrementClamp, DecrementClamp, Invert, IncrementWrap, DecrementWrap };
    enum class BlendFactor : uint8_t
    {
        Zero, One, SrcColor, OneMinusSrcColor, DstColor, OneMinusDstColor, SrcAlpha, OneMinusSrcAlpha,
        DstAlpha, OneMinusDstAlpha, ConstantColor, OneMinusConstantColor, ConstantAlpha, OneMinusConstantAlpha,
        SrcAlphaSaturate,
    };
    enum class BlendOp : uint8_t { Add, Subtract, ReverseSubtract, Min, Max };
    enum class LoadOp : uint8_t { Load, Clear, DontCare };
    enum class StoreOp : uint8_t { Store, DontCare };

    struct Extent3D { uint32_t width = 1, height = 1, depth = 1; };
    struct Rect2D { int32_t x = 0, y = 0; uint32_t width = 0, height = 0; };
    struct ClearColor { float rgba[4] = {}; };
    struct ClearDepthStencil { float depth = 1.0f; uint8_t stencil = 0; };
}
