// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
// The Metal command list (gpu/rhi/command_list.h); see metal_device.cpp.
#include "metal_internal.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace gpu::rhi::metal
{
    namespace impl
    {
        namespace
        {
            class MetalCommandList final : public CommandList
            {
            public:
                MetalCommandList(Context& c, InternalPipelines& internal) : m_c(c), m_internal(internal) {}
                ~MetalCommandList() override { EndEncoder(); }

                // Ends the recording and commits it; `fence` is signalled when it has run.
                void Commit(MetalFence& fence)
                {
                    EndEncoder();
                    std::shared_ptr<FenceState> state = fence.Arm();
                    m_commands->addCompletedHandler([state](MTL::CommandBuffer* cb) {
                        if (cb->error())
                            fprintf(stderr, "[rhi/metal] a command buffer failed: %s\n", Describe(cb->error()).c_str());
                        std::lock_guard lock(state->mutex);
                        state->signaled = true;
                        state->cv.notify_all();
                    });
                    m_commands->commit();
                }

                void Begin() override
                {
                    Pool pool;
                    EndEncoder();
                    m_temps.clear();
                    m_sets = {};
                    m_pushed = {};
                    m_constantsSize = 0;
                    m_computePipeline = nullptr;
                    m_graphicsPipeline = nullptr;
                    m_rs = {};
                    m_commands = Keep(m_c.queue->commandBuffer());
                }

                void End() override { EndEncoder(); }

                // ---- passes ----
                void BeginRendering(const RenderingDesc& desc) override
                {
                    Pool pool;
                    EndEncoder();
                    auto pass = Own(MTL::RenderPassDescriptor::alloc()->init());
                    m_passFormats = {};
                    m_rs.targetWidth = desc.area.x + desc.area.width;
                    m_rs.targetHeight = desc.area.y + desc.area.height;
                    auto loadAction = [](LoadOp op) {
                        return op == LoadOp::Clear ? MTL::LoadActionClear : op == LoadOp::Load ? MTL::LoadActionLoad : MTL::LoadActionDontCare;
                    };
                    auto storeAction = [](StoreOp op) { return op == StoreOp::Store ? MTL::StoreActionStore : MTL::StoreActionDontCare; };
                    for (int i = 0; i < 4; i++)
                    {
                        const ColorAttachment& a = desc.color[i];
                        if (!a.image)
                            continue;  // (a gap)
                        auto* image = static_cast<MetalImage*>(a.image);
                        MTL::RenderPassColorAttachmentDescriptor* d = pass->colorAttachments()->object(i);
                        d->setTexture(image->Handle());
                        d->setLoadAction(loadAction(a.load));
                        d->setStoreAction(storeAction(a.store));
                        d->setClearColor(MTL::ClearColor::Make(a.clear.rgba[0], a.clear.rgba[1], a.clear.rgba[2], a.clear.rgba[3]));
                        m_passFormats.color[i] = image->Desc().format;
                        m_rs.targetWidth = image->Desc().extent.width;
                        m_rs.targetHeight = image->Desc().extent.height;
                    }
                    if (desc.depth.image)
                    {
                        auto* image = static_cast<MetalImage*>(desc.depth.image);
                        MTL::RenderPassDepthAttachmentDescriptor* d = pass->depthAttachment();
                        d->setTexture(image->Handle());
                        d->setLoadAction(loadAction(desc.depth.load));
                        d->setStoreAction(desc.depth.readOnly ? MTL::StoreActionStore : storeAction(desc.depth.store));
                        d->setClearDepth(desc.depth.clear.depth);
                        MTL::RenderPassStencilAttachmentDescriptor* st = pass->stencilAttachment();
                        st->setTexture(image->Handle());
                        st->setLoadAction(loadAction(desc.depth.load));
                        st->setStoreAction(desc.depth.readOnly ? MTL::StoreActionStore : storeAction(desc.depth.store));
                        st->setClearStencil(desc.depth.clear.stencil);
                        m_passFormats.depth = image->Desc().format;
                        m_rs.targetWidth = image->Desc().extent.width;
                        m_rs.targetHeight = image->Desc().extent.height;
                    }
                    if (m_c.visibility)
                        pass->setVisibilityResultBuffer(m_c.visibility.get());
                    m_render = Keep(m_commands->renderCommandEncoder(pass.get()));
                    ReapplyRender();
                }

                void EndRendering() override { EndEncoder(); }

                void ClearAttachments(const Rect2D& area, const ClearColor* color, uint32_t colorMask, const ClearDepthStencil* depthStencil, bool clearDepth,
                    bool clearStencil) override
                {
                    float constants[20] = {};  // K: float4 c[4]; float d (a block is 16-byte aligned: 80 bytes)
                    for (int i = 0; i < 4; i++)
                        if (color && (colorMask >> i & 1))
                            memcpy(&constants[i * 4], color[i].rgba, 16);
                    const bool depth = clearDepth && depthStencil && m_passFormats.depth != Format::Undefined;
                    if (depth)
                        constants[16] = depthStencil->depth;
                    const bool stencil = clearStencil && depthStencil && m_passFormats.depth != Format::Undefined;
                    MTL::RenderPipelineState* pipeline = m_internal.Clear(m_passFormats.color, colorMask, m_passFormats.depth, depth);
                    m_render->setRenderPipelineState(pipeline);
                    // The depth-stencil state: writes the depth when asked, and the stencil by Replace.
                    {
                        Pool pool;
                        auto d = Own(MTL::DepthStencilDescriptor::alloc()->init());
                        d->setDepthCompareFunction(MTL::CompareFunctionAlways);
                        d->setDepthWriteEnabled(depth);
                        if (stencil)
                        {
                            auto s = Own(MTL::StencilDescriptor::alloc()->init());
                            s->setStencilCompareFunction(MTL::CompareFunctionAlways);
                            s->setStencilFailureOperation(MTL::StencilOperationReplace);
                            s->setDepthFailureOperation(MTL::StencilOperationReplace);
                            s->setDepthStencilPassOperation(MTL::StencilOperationReplace);
                            d->setFrontFaceStencil(s.get());
                            d->setBackFaceStencil(s.get());
                        }
                        auto state = Own(m_c.device->newDepthStencilState(d.get()));
                        m_render->setDepthStencilState(state.get());
                    }
                    if (stencil)
                        m_render->setStencilReferenceValue(depthStencil->stencil);
                    m_render->setCullMode(MTL::CullModeNone);
                    m_render->setDepthBias(0, 0, 0);
                    m_render->setViewport(MTL::Viewport{ 0, 0, double(m_rs.targetWidth), double(m_rs.targetHeight), 0.0, 1.0 });
                    m_render->setScissorRect(Clamp(area));
                    m_render->setVertexBytes(constants, sizeof(constants), 0);
                    m_render->setFragmentBytes(constants, sizeof(constants), 0);
                    m_render->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
                    ReapplyRender();  // what the clear changed
                }

                // ---- state ----
                // As in Vulkan, what was bound stays for the tables the new pipeline's layout shares (the
                // same slots), and the others are gone.
                void DropIncompatibleTables(const MetalBindingLayout& layout)
                {
                    auto same = [&](const MetalBindingLayout* other, uint32_t t) {
                        if (!other || t >= layout.Desc().tables.size() || t >= other->Desc().tables.size())
                            return false;
                        const auto& a = layout.Desc().tables[t].slots;
                        const auto& b = other->Desc().tables[t].slots;
                        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](const BindingSlot& x, const BindingSlot& y) {
                            return x.index == y.index && x.kind == y.kind && x.stages == y.stages;
                        });
                    };
                    for (uint32_t t = 0; t < m_sets.size(); t++)
                    {
                        if (m_sets[t].set && !same(m_sets[t].layout, t))
                            m_sets[t].set = nullptr;
                        if (m_pushed[t].layout && !same(m_pushed[t].layout, t))
                            m_pushed[t].layout = nullptr;
                    }
                }
                void SetPipeline(const Pipeline& pipeline) override
                {
                    const auto* p = &static_cast<const MetalPipeline&>(pipeline);
                    if (p->layout)
                        DropIncompatibleTables(*p->layout);
                    if (p->IsCompute())
                    {
                        m_computePipeline = static_cast<const MetalComputePipeline*>(p);
                        if (m_compute)
                            ApplyComputePipeline();
                    }
                    else
                    {
                        m_graphicsPipeline = static_cast<const MetalGraphicsPipeline*>(p);
                        if (m_render)
                            ApplyGraphicsPipeline();
                    }
                }
                void SetViewport(float x, float y, float width, float height, float minDepth, float maxDepth) override
                {
                    m_rs.viewport = MTL::Viewport{ x, y, width, height, minDepth, maxDepth };
                    m_rs.viewportSet = true;
                    if (m_render)
                        m_render->setViewport(m_rs.viewport);
                }
                void SetScissor(const Rect2D& rect) override
                {
                    m_rs.scissor = rect;
                    m_rs.scissorSet = true;
                    if (m_render)
                        m_render->setScissorRect(Clamp(rect));
                }
                void SetStencil(const StencilState& front, const StencilState& back) override
                {
                    m_rs.stencilFront = front;
                    m_rs.stencilBack = back;
                    if (m_render && m_graphicsPipeline)
                    {
                        m_render->setStencilReferenceValues(front.reference, back.reference);
                        m_render->setDepthStencilState(m_graphicsPipeline->DepthStencil(front, back));
                    }
                }
                void SetDepthBias(float constant, float clamp, float slope) override
                {
                    m_rs.biasConstant = constant;
                    m_rs.biasClamp = clamp;
                    m_rs.biasSlope = slope;
                    if (m_render && m_graphicsPipeline && m_graphicsPipeline->desc.depthBias)
                        m_render->setDepthBias(constant, slope, clamp);
                }
                void SetBlendConstants(const float rgba[4]) override
                {
                    memcpy(m_rs.blend, rgba, 16);
                    if (m_render)
                        m_render->setBlendColor(rgba[0], rgba[1], rgba[2], rgba[3]);
                }

                // ---- bindings ----
                void BindSet(const BindingLayout& layout, uint32_t table, BindingSet& set, const uint32_t* dynamicOffsets, uint32_t dynamicCount) override
                {
                    Bound& b = m_sets.at(table);
                    b.layout = &static_cast<const MetalBindingLayout&>(layout);
                    b.set = &static_cast<MetalBindingSet&>(set);
                    b.dynamic.assign(dynamicOffsets, dynamicOffsets + dynamicCount);
                    m_pushed.at(table).layout = nullptr;
                    ApplyTable(table);
                }
                void PushBindings(const BindingLayout& layout, uint32_t table, const BindingEntry* entries, size_t count) override
                {
                    Pushed& p = m_pushed.at(table);
                    p.layout = &static_cast<const MetalBindingLayout&>(layout);
                    p.entries.assign(entries, entries + count);
                    m_sets.at(table).set = nullptr;
                    ApplyTable(table);
                }
                void PushConstants(const BindingLayout& layout, const void* data, uint32_t size) override
                {
                    if (size > sizeof(m_constants))
                        Fail("push constants beyond 256 bytes");
                    m_constantsLayout = &static_cast<const MetalBindingLayout&>(layout);
                    memcpy(m_constants, data, size);
                    // The shader's block is the layout's size; Vulkan lets a caller give a prefix (what the
                    // shader reads), Metal wants the whole block (the rest is whatever was there).
                    m_constantsSize = (std::max<uint32_t>(size, m_constantsLayout->Desc().pushConstantBytes) + 15) & ~15u;  // (a block is 16-byte aligned)
                    if (m_constantsSize > sizeof(m_constants))
                        Fail("push constants beyond 256 bytes");
                    ApplyConstants();
                }

                // ---- work ----
                void Draw(uint32_t vertices, uint32_t instances, uint32_t firstVertex, uint32_t firstInstance) override
                {
                    m_render->drawPrimitives(m_graphicsPipeline->primitive, firstVertex, vertices, instances, firstInstance);
                }
                void DrawIndexed(uint32_t indices, uint32_t instances, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) override
                {
                    const uint32_t size = m_rs.index32 ? 4 : 2;
                    m_render->drawIndexedPrimitives(m_graphicsPipeline->primitive, indices, m_rs.index32 ? MTL::IndexTypeUInt32 : MTL::IndexTypeUInt16,
                        static_cast<MetalBuffer*>(m_rs.indexBuffer)->Handle(), m_rs.indexOffset + uint64_t(firstIndex) * size, instances, vertexOffset, firstInstance);
                }
                void BindIndexBuffer(Buffer& buffer, uint64_t offset, bool index32) override
                {
                    m_rs.indexBuffer = &buffer;
                    m_rs.indexOffset = offset;
                    m_rs.index32 = index32;
                }
                void Dispatch(uint32_t x, uint32_t y, uint32_t z) override
                {
                    BeginCompute();
                    m_compute->dispatchThreadgroups(MTL::Size::Make(x, y, z), m_computePipeline->Group());
                }

                // ---- transfers ----
                void CopyBuffer(Buffer& src, Buffer& dst, uint64_t srcOffset, uint64_t dstOffset, uint64_t size) override
                {
                    BeginBlit()->copyFromBuffer(static_cast<MetalBuffer&>(src).Handle(), srcOffset, static_cast<MetalBuffer&>(dst).Handle(), dstOffset, size);
                }
                void CopyBufferToImage(Buffer& src, Image& dst, const BufferImageCopy* regions, uint32_t count) override
                {
                    MTL::BlitCommandEncoder* blit = BeginBlit();
                    auto& image = static_cast<MetalImage&>(dst);
                    for (uint32_t i = 0; i < count; i++)
                    {
                        const BufferImageCopy& r = regions[i];
                        const Layout l = LayoutOf(image.Desc().format, r);
                        for (uint32_t layer = 0; layer < r.layerCount; layer++)
                            blit->copyFromBuffer(static_cast<MetalBuffer&>(src).Handle(), r.bufferOffset + uint64_t(layer) * l.perImage * r.depth, l.perRow, l.perImage,
                                MTL::Size::Make(r.region.width, r.region.height, r.depth), image.Handle(), r.layer + layer, r.level,
                                MTL::Origin::Make(r.region.x, r.region.y, r.z));
                    }
                }
                void CopyImageToBuffer(Image& src, Buffer& dst, const BufferImageCopy* regions, uint32_t count) override
                {
                    MTL::BlitCommandEncoder* blit = BeginBlit();
                    auto& image = static_cast<MetalImage&>(src);
                    for (uint32_t i = 0; i < count; i++)
                    {
                        const BufferImageCopy& r = regions[i];
                        const Layout l = LayoutOf(image.Desc().format, r);
                        for (uint32_t layer = 0; layer < r.layerCount; layer++)
                            blit->copyFromTexture(image.Handle(), r.layer + layer, r.level, MTL::Origin::Make(r.region.x, r.region.y, r.z),
                                MTL::Size::Make(r.region.width, r.region.height, r.depth), static_cast<MetalBuffer&>(dst).Handle(),
                                r.bufferOffset + uint64_t(layer) * l.perImage * r.depth, l.perRow, l.perImage);
                    }
                }
                void BlitImage(Image& src, const Rect2D& srcRect, Image& dst, const Rect2D& dstRect, Filter filter) override
                {
                    auto& s = static_cast<MetalImage&>(src);
                    auto& d = static_cast<MetalImage&>(dst);
                    if (srcRect.width == dstRect.width && srcRect.height == dstRect.height && s.Desc().format == d.Desc().format)
                    {
                        BeginBlit()->copyFromTexture(s.Handle(), 0, 0, MTL::Origin::Make(srcRect.x, srcRect.y, 0), MTL::Size::Make(srcRect.width, srcRect.height, 1),
                            d.Handle(), 0, 0, MTL::Origin::Make(dstRect.x, dstRect.y, 0));
                        return;
                    }
                    Pool pool;
                    EndEncoder();
                    auto pass = Own(MTL::RenderPassDescriptor::alloc()->init());
                    MTL::RenderPassColorAttachmentDescriptor* a = pass->colorAttachments()->object(0);
                    a->setTexture(d.Handle());
                    a->setLoadAction(MTL::LoadActionLoad);
                    a->setStoreAction(MTL::StoreActionStore);
                    auto render = Keep(m_commands->renderCommandEncoder(pass.get()));
                    render->setRenderPipelineState(m_internal.Scale(d.Desc().format));
                    render->setViewport(MTL::Viewport{ double(dstRect.x), double(dstRect.y), double(dstRect.width), double(dstRect.height), 0.0, 1.0 });
                    const uint32_t w = d.Desc().extent.width, h = d.Desc().extent.height;
                    const int32_t x0 = std::max(dstRect.x, 0), y0 = std::max(dstRect.y, 0);
                    const int32_t x1 = std::min<int32_t>(dstRect.x + dstRect.width, w), y1 = std::min<int32_t>(dstRect.y + dstRect.height, h);
                    render->setScissorRect(MTL::ScissorRect{ NS::UInteger(x0), NS::UInteger(y0), NS::UInteger(std::max(x1 - x0, 0)), NS::UInteger(std::max(y1 - y0, 0)) });
                    const float sw = float(s.Desc().extent.width), sh = float(s.Desc().extent.height);
                    const float b[4] = { srcRect.x / sw, srcRect.y / sh, srcRect.width / sw, srcRect.height / sh };
                    render->setVertexBytes(b, sizeof(b), 0);
                    render->setFragmentTexture(s.Handle(), 0);
                    render->setFragmentSamplerState(m_c.BlitSampler(filter), 0);
                    render->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
                    render->endEncoding();
                }
                void FillBuffer(Buffer& dst, uint64_t offset, uint64_t size, uint32_t value) override
                {
                    const uint8_t b = uint8_t(value);
                    if (value == b * 0x01010101u)
                    {
                        BeginBlit()->fillBuffer(static_cast<MetalBuffer&>(dst).Handle(), NS::Range::Make(offset, size), b);
                        return;
                    }
                    std::vector<uint32_t> words(size / 4, value);  // (the whole words of a pattern)
                    UpdateBuffer(dst, offset, words.data(), words.size() * 4);
                }
                void UpdateBuffer(Buffer& dst, uint64_t offset, const void* data, uint64_t size) override
                {
                    // Metal has no inline update: a temporary, kept until the list is done with.
                    Ptr<MTL::Buffer> temp = Own(m_c.device->newBuffer(data, size, MTL::ResourceStorageModeShared));
                    BeginBlit()->copyFromBuffer(temp.get(), 0, static_cast<MetalBuffer&>(dst).Handle(), offset, size);
                    m_temps.push_back(std::move(temp));
                }
                void ClearImage(Image& image, const ClearColor& color) override
                {
                    Pool pool;
                    EndEncoder();
                    auto pass = Own(MTL::RenderPassDescriptor::alloc()->init());
                    MTL::RenderPassColorAttachmentDescriptor* a = pass->colorAttachments()->object(0);
                    a->setTexture(static_cast<MetalImage&>(image).Handle());
                    a->setLoadAction(MTL::LoadActionClear);
                    a->setStoreAction(MTL::StoreActionStore);
                    a->setClearColor(MTL::ClearColor::Make(color.rgba[0], color.rgba[1], color.rgba[2], color.rgba[3]));
                    m_commands->renderCommandEncoder(pass.get())->endEncoding();
                }
                void ClearImage(Image& image, const ClearDepthStencil& value) override
                {
                    Pool pool;
                    EndEncoder();
                    auto pass = Own(MTL::RenderPassDescriptor::alloc()->init());
                    MTL::Texture* texture = static_cast<MetalImage&>(image).Handle();
                    MTL::RenderPassDepthAttachmentDescriptor* d = pass->depthAttachment();
                    d->setTexture(texture);
                    d->setLoadAction(MTL::LoadActionClear);
                    d->setStoreAction(MTL::StoreActionStore);
                    d->setClearDepth(value.depth);
                    MTL::RenderPassStencilAttachmentDescriptor* s = pass->stencilAttachment();
                    s->setTexture(texture);
                    s->setLoadAction(MTL::LoadActionClear);
                    s->setStoreAction(MTL::StoreActionStore);
                    s->setClearStencil(value.stencil);
                    m_commands->renderCommandEncoder(pass.get())->endEncoding();
                }

                // ---- synchronisation ----
                void Transition(const ImageTransition*, size_t, const BufferTransition*, size_t) override {}  // Metal tracks the hazards
                // (Within a serial compute encoder a dispatch sees the one before; across encoders Metal tracks
                // the hazards of the resources it knows.)
                void MemoryBarrier(Barrier) override {}
                void SignalEvent(Event& event) override
                {
                    EndEncoder();
                    auto& e = static_cast<MetalEvent&>(event);
                    m_commands->encodeSignalEvent(e.Handle(), e.Next());
                }

                // ---- queries ----
                void ResetQueries(QueryPool& pool, uint32_t first, uint32_t count) override
                {
                    // Timestamps are overwritten by the next stamp; counts are zeroed
                    // where the queue will get to them.
                    auto& p = static_cast<MetalQueryPool&>(pool);
                    if (p.Kind() != QueryKind::Occlusion)
                        return;
                    Pool autorelease;
                    BeginBlit()->fillBuffer(m_c.visibility.get(), NS::Range(p.VisibilityOffset(first), NS::UInteger(count) * sizeof(uint64_t)), 0);
                }

                // Apple GPUs sample counters at the edges of encoders only, so a stamp
                // is an encoder of its own, sampled as it starts (an end sample came back
                // zero): after the encoders recorded before it, `end` or not.
                void WriteTimestamp(QueryPool& pool, uint32_t index, bool) override
                {
                    auto& p = static_cast<MetalQueryPool&>(pool);
                    Pool autorelease;
                    EndEncoder();
                    auto pass = Own(MTL::BlitPassDescriptor::alloc()->init());
                    MTL::BlitPassSampleBufferAttachmentDescriptor* a = pass->sampleBufferAttachments()->object(0);
                    a->setSampleBuffer(p.Samples());
                    a->setStartOfEncoderSampleIndex(index);
                    a->setEndOfEncoderSampleIndex(MTL::CounterDontSample);
                    auto* blit = m_commands->blitCommandEncoder(pass.get());
                    blit->fillBuffer(m_c.scratch.get(), NS::Range(0, 4), 0);  // (an encoder with no work may be dropped)
                    blit->endEncoding();
                }

                void BeginQuery(QueryPool& pool, uint32_t index) override
                {
                    if (!m_render)
                        Fail("BeginQuery outside a render pass");
                    m_render->setVisibilityResultMode(MTL::VisibilityResultModeCounting, static_cast<MetalQueryPool&>(pool).VisibilityOffset(index));
                }
                void EndQuery(QueryPool& pool, uint32_t index) override
                {
                    if (!m_render)
                        Fail("EndQuery outside a render pass");
                    m_render->setVisibilityResultMode(MTL::VisibilityResultModeDisabled, static_cast<MetalQueryPool&>(pool).VisibilityOffset(index));
                }

            private:
                struct Bound
                {
                    const MetalBindingLayout* layout = nullptr;
                    MetalBindingSet* set = nullptr;
                    std::vector<uint32_t> dynamic;
                };
                struct Pushed
                {
                    const MetalBindingLayout* layout = nullptr;
                    std::vector<BindingEntry> entries;
                };
                // Dynamic state that outlives an encoder (as Vulkan's does the render pass).
                struct RenderState
                {
                    bool viewportSet = false, scissorSet = false;
                    MTL::Viewport viewport{};
                    Rect2D scissor;
                    CommandList::StencilState stencilFront, stencilBack;
                    float biasConstant = 0, biasClamp = 0, biasSlope = 0;
                    float blend[4] = {};
                    Buffer* indexBuffer = nullptr;
                    uint64_t indexOffset = 0;
                    bool index32 = false;
                    uint32_t targetWidth = 0, targetHeight = 0;
                };
                struct PassFormats
                {
                    Format color[4] = {};
                    Format depth = Format::Undefined;
                };
                struct Layout
                {
                    uint32_t perRow, perImage;
                };

                // Bytes per row and per image of a buffer region (BC: of blocks).
                static Layout LayoutOf(Format format, const BufferImageCopy& r)
                {
                    const uint32_t width = r.bufferRowLength ? r.bufferRowLength : r.region.width;
                    if (IsBlock(format))
                    {
                        const uint32_t block = format == Format::BC1 ? 8 : 16;
                        const uint32_t perRow = (width + 3) / 4 * block;
                        return { perRow, perRow * ((r.region.height + 3) / 4) };
                    }
                    const uint32_t perRow = width * BytesPerPixel(format);
                    return { perRow, perRow * r.region.height };
                }

                MTL::ScissorRect Clamp(const Rect2D& r) const
                {
                    const int32_t x0 = std::clamp<int32_t>(r.x, 0, int32_t(m_rs.targetWidth)), y0 = std::clamp<int32_t>(r.y, 0, int32_t(m_rs.targetHeight));
                    const int32_t x1 = std::clamp<int32_t>(r.x + int32_t(r.width), 0, int32_t(m_rs.targetWidth));
                    const int32_t y1 = std::clamp<int32_t>(r.y + int32_t(r.height), 0, int32_t(m_rs.targetHeight));
                    return MTL::ScissorRect{ NS::UInteger(x0), NS::UInteger(y0), NS::UInteger(std::max(x1 - x0, 0)), NS::UInteger(std::max(y1 - y0, 0)) };
                }

                void EndEncoder()
                {
                    if (m_compute)
                    {
                        m_compute->endEncoding();
                        m_compute.reset();
                    }
                    if (m_blit)
                    {
                        m_blit->endEncoding();
                        m_blit.reset();
                    }
                    if (m_render)
                    {
                        m_render->endEncoding();
                        m_render.reset();
                    }
                }

                MTL::BlitCommandEncoder* BeginBlit()
                {
                    if (!m_blit)
                    {
                        Pool pool;
                        EndEncoder();
                        m_blit = Keep(m_commands->blitCommandEncoder());
                    }
                    return m_blit.get();
                }

                void BeginCompute()
                {
                    if (m_compute)
                        return;
                    Pool pool;
                    EndEncoder();
                    m_compute = Keep(m_commands->computeCommandEncoder());
                    // What was bound before: Metal's belongs to the encoder.
                    if (m_computePipeline)
                        ApplyComputePipeline();
                    for (uint32_t t = 0; t < m_sets.size(); t++)
                        ApplyTable(t);
                    ApplyConstants();
                }

                void ApplyComputePipeline() { m_compute->setComputePipelineState(m_computePipeline->Handle()); }

                void ApplyGraphicsPipeline()
                {
                    const MetalGraphicsPipeline& p = *m_graphicsPipeline;
                    m_render->setRenderPipelineState(p.state.get());
                    m_render->setDepthStencilState(p.DepthStencil(m_rs.stencilFront, m_rs.stencilBack));
                    m_render->setCullMode(p.cull);
                    m_render->setFrontFacingWinding(p.winding);
                    m_render->setDepthClipMode(p.desc.depthClamp ? MTL::DepthClipModeClamp : MTL::DepthClipModeClip);
                    if (p.desc.depthBias)
                        m_render->setDepthBias(m_rs.biasConstant, m_rs.biasSlope, m_rs.biasClamp);
                    else
                        m_render->setDepthBias(0, 0, 0);
                }

                // A fresh render encoder gets everything the list has set so far.
                // Whether the pipeline last set draws into the formats of this pass (Vulkan leaves a
                // stale one bound across passes unused; Metal refuses it).
                bool PipelineFitsPass() const
                {
                    const GraphicsPipelineDesc& d = m_graphicsPipeline->desc;
                    for (int i = 0; i < 4; i++)
                        if (d.colorFormats[i] != m_passFormats.color[i])
                            return false;
                    return d.depthFormat == m_passFormats.depth;
                }
                void ReapplyRender()
                {
                    if (m_graphicsPipeline && PipelineFitsPass())
                        ApplyGraphicsPipeline();
                    m_render->setStencilReferenceValues(m_rs.stencilFront.reference, m_rs.stencilBack.reference);
                    m_render->setBlendColor(m_rs.blend[0], m_rs.blend[1], m_rs.blend[2], m_rs.blend[3]);
                    m_render->setViewport(m_rs.viewportSet ? m_rs.viewport : MTL::Viewport{ 0, 0, double(m_rs.targetWidth), double(m_rs.targetHeight), 0.0, 1.0 });
                    m_render->setScissorRect(Clamp(m_rs.scissorSet ? m_rs.scissor : Rect2D{ 0, 0, m_rs.targetWidth, m_rs.targetHeight }));
                    for (uint32_t t = 0; t < m_sets.size(); t++)
                        ApplyTable(t);
                    ApplyConstants();
                }

                void ApplyConstants()
                {
                    if (!m_constantsSize)
                        return;
                    const uint32_t index = m_constantsLayout->PushConstantBuffer();
                    if (m_compute)
                        m_compute->setBytes(m_constants, m_constantsSize, index);
                    if (m_render)
                    {
                        m_render->setVertexBytes(m_constants, m_constantsSize, index);
                        m_render->setFragmentBytes(m_constants, m_constantsSize, index);
                    }
                }

                // Table `table` as last bound (a set or pushed entries) on the encoder in use.
                void ApplyTable(uint32_t table)
                {
                    if (!m_compute && !m_render)
                        return;
                    const Bound& b = m_sets[table];
                    const Pushed& p = m_pushed[table];
                    if (b.set)
                    {
                        const std::vector<BindingEntry> entries = b.set->Entries();
                        ApplyEntries(*b.layout, table, entries.data(), entries.size(), b.dynamic);
                    }
                    else if (p.layout)
                    {
                        ApplyEntries(*p.layout, table, p.entries.data(), p.entries.size(), {});
                    }
                }

                void ApplyEntries(const MetalBindingLayout& layout, uint32_t table, const BindingEntry* entries, size_t count, const std::vector<uint32_t>& dynamic)
                {
                    for (size_t i = 0; i < count; i++)
                    {
                        const SlotMap* m = layout.Find(table, entries[i].slot);
                        if (!m)
                            Fail("a binding has no slot in its layout");
                        const uint8_t stages = m->slot.stages;
                        switch (m->slot.kind)
                        {
                        case BindingKind::StorageBuffer:
                        case BindingKind::UniformBufferDynamic:
                        {
                            uint64_t offset = entries[i].buffer.offset;
                            if (m->dynamicOrder >= 0 && size_t(m->dynamicOrder) < dynamic.size())
                                offset += dynamic[m->dynamicOrder];
                            MTL::Buffer* buffer = static_cast<MetalBuffer*>(entries[i].buffer.buffer)->Handle();
                            if (m_compute)
                                m_compute->setBuffer(buffer, offset, m->buffer);
                            if (m_render && (stages & uint8_t(ShaderStage::Vertex)))
                                m_render->setVertexBuffer(buffer, offset, m->buffer);
                            if (m_render && (stages & uint8_t(ShaderStage::Fragment)))
                                m_render->setFragmentBuffer(buffer, offset, m->buffer);
                            break;
                        }
                        case BindingKind::CombinedImageSampler:
                        case BindingKind::SampledImage:
                        case BindingKind::StorageImage:
                        {
                            MTL::Texture* texture = entries[i].image.view ? static_cast<const MetalImageView*>(entries[i].image.view)->Handle()
                                                                          : static_cast<MetalImage*>(entries[i].image.image)->Handle();
                                    MTL::SamplerState* sampler = nullptr;
                            if (m->slot.kind == BindingKind::CombinedImageSampler)
                            {
                                const Sampler* s = m->slot.immutableSampler ? m->slot.immutableSampler : entries[i].image.sampler;
                                sampler = static_cast<const MetalSampler*>(s)->Handle();
                            }
                            if (m_compute)
                            {
                                m_compute->setTexture(texture, m->texture);
                                if (sampler)
                                    m_compute->setSamplerState(sampler, m->sampler);
                            }
                            if (m_render && (stages & uint8_t(ShaderStage::Vertex)))
                            {
                                m_render->setVertexTexture(texture, m->texture);
                                if (sampler)
                                    m_render->setVertexSamplerState(sampler, m->sampler);
                            }
                            if (m_render && (stages & uint8_t(ShaderStage::Fragment)))
                            {
                                m_render->setFragmentTexture(texture, m->texture);
                                if (sampler)
                                    m_render->setFragmentSamplerState(sampler, m->sampler);
                            }
                            break;
                        }
                        }
                    }
                }

                Context& m_c;
                InternalPipelines& m_internal;
                Ptr<MTL::CommandBuffer> m_commands;
                Ptr<MTL::BlitCommandEncoder> m_blit;
                Ptr<MTL::ComputeCommandEncoder> m_compute;
                Ptr<MTL::RenderCommandEncoder> m_render;
                const MetalComputePipeline* m_computePipeline = nullptr;
                const MetalGraphicsPipeline* m_graphicsPipeline = nullptr;
                RenderState m_rs;
                PassFormats m_passFormats;
                std::array<Bound, 4> m_sets;
                std::array<Pushed, 4> m_pushed;
                const MetalBindingLayout* m_constantsLayout = nullptr;
                uint8_t m_constants[256] = {};
                uint32_t m_constantsSize = 0;
                std::vector<Ptr<MTL::Buffer>> m_temps;  // until the list is recorded again
            };
        }

        std::unique_ptr<CommandList> MakeCommandList(Context& c, InternalPipelines& internal)
        {
            return std::make_unique<MetalCommandList>(c, internal);
        }

        void CommitCommandList(CommandList& list, MetalFence& fence)
        {
            static_cast<MetalCommandList&>(list).Commit(fence);
        }
    }
}
