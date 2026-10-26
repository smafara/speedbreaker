// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// What the renderer and the presenter record. One list is recorded by one
// thread at a time. The calls are those renderer.cpp and presenter.cpp make
// of vkCmd* today, in their order of frequency (draws, constants, binds,
// barriers, copies), without Vulkan's layouts and stage masks (D9).
#pragma once
#include <cstddef>
#include <cstdint>

#include "device.h"

namespace gpu::rhi
{
    struct ColorAttachment
    {
        Image* image = nullptr;
        LoadOp load = LoadOp::Load;
        StoreOp store = StoreOp::Store;
        ClearColor clear;
    };

    struct DepthAttachment
    {
        Image* image = nullptr;
        LoadOp load = LoadOp::Load;
        StoreOp store = StoreOp::Store;
        ClearDepthStencil clear;
        bool readOnly = false;
    };

    struct RenderingDesc
    {
        ColorAttachment color[4];
        DepthAttachment depth;
        Rect2D area;
    };

    // Image state changes (D9). `before` is the state the resource is in;
    // Undefined drops the contents.
    struct ImageTransition { Image* image = nullptr; State before = State::Undefined, after = State::ShaderRead; };
    struct BufferTransition { Buffer* buffer = nullptr; State before = State::ShaderWrite, after = State::ShaderRead; };

    struct BufferImageCopy
    {
        uint64_t bufferOffset = 0;
        uint32_t bufferRowLength = 0;    // texels; 0: tightly packed
        uint32_t level = 0, layer = 0, layerCount = 1;
        Rect2D region;                   // in the image (x, y, width, height)
        uint32_t z = 0, depth = 1;
    };

    class CommandList
    {
    public:
        virtual ~CommandList() = default;

        // The list may be recorded again after its submission's fence is signalled.
        virtual void Begin() = 0;
        virtual void End() = 0;

        // ---- passes ----
        virtual void BeginRendering(const RenderingDesc& desc) = 0;
        virtual void EndRendering() = 0;
        // Inside a pass: clear part of the attachments (a guest clear).
        virtual void ClearAttachments(const Rect2D& area, const ClearColor* color, uint32_t colorMask,
            const ClearDepthStencil* depthStencil, bool clearDepth, bool clearStencil) = 0;

        // ---- state ----
        virtual void SetPipeline(const Pipeline& pipeline) = 0;
        virtual void SetViewport(float x, float y, float width, float height, float minDepth, float maxDepth) = 0;
        virtual void SetScissor(const Rect2D& rect) = 0;
        virtual void SetStencil(uint8_t reference, uint8_t compareMask, uint8_t writeMask) = 0;
        virtual void SetDepthBias(float constant, float clamp, float slope) = 0;
        virtual void SetBlendConstants(const float rgba[4]) = 0;

        // ---- bindings (D10) ----
        // A persistent table; `dynamicOffsets` for its UniformBufferDynamic slots, in slot order.
        virtual void BindSet(const BindingLayout& layout, uint32_t table, BindingSet& set,
            const uint32_t* dynamicOffsets = nullptr, uint32_t dynamicCount = 0) = 0;
        // A pushed table: these entries, for the next draws.
        virtual void PushBindings(const BindingLayout& layout, uint32_t table, const BindingEntry* entries, size_t count) = 0;
        virtual void PushConstants(const BindingLayout& layout, const void* data, uint32_t size) = 0;

        // ---- work ----
        virtual void Draw(uint32_t vertices, uint32_t instances, uint32_t firstVertex, uint32_t firstInstance) = 0;
        virtual void DrawIndexed(uint32_t indices, uint32_t instances, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) = 0;
        virtual void BindIndexBuffer(Buffer& buffer, uint64_t offset, bool index32) = 0;
        virtual void Dispatch(uint32_t x, uint32_t y, uint32_t z) = 0;

        // ---- transfers ----
        virtual void CopyBuffer(Buffer& src, Buffer& dst, uint64_t srcOffset, uint64_t dstOffset, uint64_t size) = 0;
        virtual void CopyBufferToImage(Buffer& src, Image& dst, const BufferImageCopy* regions, uint32_t count) = 0;
        virtual void CopyImageToBuffer(Image& src, Buffer& dst, const BufferImageCopy* regions, uint32_t count) = 0;
        virtual void BlitImage(Image& src, const Rect2D& srcRect, Image& dst, const Rect2D& dstRect, Filter filter) = 0;
        virtual void FillBuffer(Buffer& dst, uint64_t offset, uint64_t size, uint32_t value) = 0;
        virtual void UpdateBuffer(Buffer& dst, uint64_t offset, const void* data, uint64_t size) = 0;
        virtual void ClearImage(Image& image, const ClearColor& color) = 0;
        virtual void ClearImage(Image& image, const ClearDepthStencil& value) = 0;

        // ---- synchronisation ----
        virtual void Transition(const ImageTransition* images, size_t imageCount,
            const BufferTransition* buffers = nullptr, size_t bufferCount = 0) = 0;
        // Outside a pass: `event` is signalled once everything recorded so far has executed.
        virtual void SignalEvent(Event& event) = 0;

        // ---- queries ----
        virtual void ResetQueries(QueryPool& pool, uint32_t first, uint32_t count) = 0;
        // `end`: after all the work recorded before it (else before any of it
        // starts, for the first of a pair). Metal can only stamp the edges of encoders.
        virtual void WriteTimestamp(QueryPool& pool, uint32_t index, bool end) = 0;
        virtual void BeginQuery(QueryPool& pool, uint32_t index) = 0;
        virtual void EndQuery(QueryPool& pool, uint32_t index) = 0;
    };
}
