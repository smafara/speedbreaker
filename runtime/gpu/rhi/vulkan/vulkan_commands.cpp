// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
// The Vulkan command list (gpu/rhi/command_list.h).
#include "vulkan_internal.h"

namespace gpu::rhi::vulkan
{
    namespace impl
    {
        namespace
        {
            class VkCommandList final : public CommandList
            {
            public:
                explicit VkCommandList(Context& c) : m_c(c)
                {
                    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
                    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                    pci.queueFamilyIndex = c.family;
                    Check(vkCreateCommandPool(c.device, &pci, nullptr, &m_pool), "vkCreateCommandPool");
                    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
                    ai.commandPool = m_pool;
                    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                    ai.commandBufferCount = 1;
                    Check(vkAllocateCommandBuffers(c.device, &ai, &m_cmd), "vkAllocateCommandBuffers");
                }
                ~VkCommandList() override { vkDestroyCommandPool(m_c.device, m_pool, nullptr); }
                VkCommandBuffer Handle() const { return m_cmd; }

                void Begin() override
                {
                    Check(vkResetCommandBuffer(m_cmd, 0), "vkResetCommandBuffer");
                    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    Check(vkBeginCommandBuffer(m_cmd, &bi), "vkBeginCommandBuffer");
                }
                void End() override { Check(vkEndCommandBuffer(m_cmd), "vkEndCommandBuffer"); }

                void BeginRendering(const RenderingDesc& desc) override
                {
                    auto loadOp = [](LoadOp op) {
                        return op == LoadOp::Clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : op == LoadOp::Load ? VK_ATTACHMENT_LOAD_OP_LOAD
                                                                                                       : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                    };
                    auto storeOp = [](StoreOp op) { return op == StoreOp::Store ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE; };
                    VkRenderingAttachmentInfo colors[4]{};
                    // Always four attachments, as the renderer's pipelines are: a pass
                    // may hold targets a draw in it doesn't write, and the counts of
                    // pipeline and pass must agree. A gap has no image.
                    constexpr uint32_t colorCount = 4;
                    for (uint32_t i = 0; i < 4; i++)
                    {
                        const ColorAttachment& a = desc.color[i];
                        VkRenderingAttachmentInfo& info = colors[i];
                        info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                        if (!a.image)
                            continue;
                        const auto* image = static_cast<VkImageObject*>(a.image);
                        info.imageView = image->View();
                        info.imageLayout = image->Layout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                        info.loadOp = loadOp(a.load);
                        info.storeOp = storeOp(a.store);
                        memcpy(info.clearValue.color.float32, a.clear.rgba, sizeof(a.clear.rgba));
                    }
                    VkRenderingAttachmentInfo depth{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                    if (desc.depth.image)
                    {
                        const auto* image = static_cast<VkImageObject*>(desc.depth.image);
                        depth.imageView = image->View();
                        depth.imageLayout = image->Layout(desc.depth.readOnly ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                                              : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
                        depth.loadOp = loadOp(desc.depth.load);
                        depth.storeOp = storeOp(desc.depth.store);
                        depth.clearValue.depthStencil = { desc.depth.clear.depth, desc.depth.clear.stencil };
                    }
                    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
                    ri.renderArea = { { desc.area.x, desc.area.y }, { desc.area.width, desc.area.height } };
                    ri.layerCount = 1;
                    ri.colorAttachmentCount = colorCount;
                    ri.pColorAttachments = colors;
                    ri.pDepthAttachment = desc.depth.image ? &depth : nullptr;
                    ri.pStencilAttachment = desc.depth.image ? &depth : nullptr;  // (the one depth + stencil format)
                    m_c.beginRendering(m_cmd, &ri);
                }
                void EndRendering() override { m_c.endRendering(m_cmd); }
                void ClearAttachments(const Rect2D& area, const ClearColor* color, uint32_t colorMask, const ClearDepthStencil* depthStencil,
                    bool clearDepth, bool clearStencil) override
                {
                    VkClearAttachment attachments[5]{};
                    uint32_t count = 0;
                    for (uint32_t i = 0; i < 4; i++)
                        if (color && (colorMask & (1u << i)))
                        {
                            attachments[count].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                            attachments[count].colorAttachment = i;
                            memcpy(attachments[count].clearValue.color.float32, color->rgba, sizeof(color->rgba));
                            count++;
                        }
                    if (depthStencil && (clearDepth || clearStencil))
                    {
                        attachments[count].aspectMask = (clearDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (clearStencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
                        attachments[count].clearValue.depthStencil = { depthStencil->depth, depthStencil->stencil };
                        count++;
                    }
                    VkClearRect rect{ { { area.x, area.y }, { area.width, area.height } }, 0, 1 };
                    if (count)
                        vkCmdClearAttachments(m_cmd, count, attachments, 1, &rect);
                }

                void SetPipeline(const Pipeline& pipeline) override
                {
                    const auto& p = static_cast<const VkPipelineObject&>(pipeline);
                    m_point = p.Point();
                    vkCmdBindPipeline(m_cmd, m_point, p.Handle());
                }
                void SetViewport(float x, float y, float width, float height, float minDepth, float maxDepth) override
                {
                    VkViewport v{ x, y, width, height, minDepth, maxDepth };
                    vkCmdSetViewport(m_cmd, 0, 1, &v);
                }
                void SetScissor(const Rect2D& r) override
                {
                    VkRect2D s{ { r.x, r.y }, { r.width, r.height } };
                    vkCmdSetScissor(m_cmd, 0, 1, &s);
                }
                void SetStencil(const StencilState& front, const StencilState& back) override
                {
                    vkCmdSetStencilCompareMask(m_cmd, VK_STENCIL_FACE_FRONT_BIT, front.compareMask);
                    vkCmdSetStencilCompareMask(m_cmd, VK_STENCIL_FACE_BACK_BIT, back.compareMask);
                    vkCmdSetStencilWriteMask(m_cmd, VK_STENCIL_FACE_FRONT_BIT, front.writeMask);
                    vkCmdSetStencilWriteMask(m_cmd, VK_STENCIL_FACE_BACK_BIT, back.writeMask);
                    vkCmdSetStencilReference(m_cmd, VK_STENCIL_FACE_FRONT_BIT, front.reference);
                    vkCmdSetStencilReference(m_cmd, VK_STENCIL_FACE_BACK_BIT, back.reference);
                }
                void SetDepthBias(float constant, float clamp, float slope) override { vkCmdSetDepthBias(m_cmd, constant, clamp, slope); }
                void SetBlendConstants(const float rgba[4]) override { vkCmdSetBlendConstants(m_cmd, rgba); }

                // (The set is bound for the kind of pipeline set last: set the pipeline first.)
                void BindSet(const BindingLayout& layout, uint32_t table, BindingSet& set, const uint32_t* dynamicOffsets,
                    uint32_t dynamicCount) override
                {
                    VkDescriptorSet s = static_cast<VkBindingSet&>(set).Handle();
                    vkCmdBindDescriptorSets(m_cmd, m_point, static_cast<const VkBindingLayout&>(layout).Handle(), table, 1, &s, dynamicCount,
                        dynamicOffsets);
                }
                void PushBindings(const BindingLayout& layout, uint32_t table, const BindingEntry* entries, size_t count) override
                {
                    const auto& l = static_cast<const VkBindingLayout&>(layout);
                    Writes w(l, table, entries, count);
                    m_c.pushDescriptor(m_cmd, m_point, l.Handle(), table, uint32_t(w.writes.size()), w.writes.data());
                }
                void PushConstants(const BindingLayout& layout, const void* data, uint32_t size) override
                {
                    vkCmdPushConstants(m_cmd, static_cast<const VkBindingLayout&>(layout).Handle(),
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
                }

                void Draw(uint32_t vertices, uint32_t instances, uint32_t firstVertex, uint32_t firstInstance) override
                {
                    vkCmdDraw(m_cmd, vertices, instances, firstVertex, firstInstance);
                }
                void DrawIndexed(uint32_t indices, uint32_t instances, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) override
                {
                    vkCmdDrawIndexed(m_cmd, indices, instances, firstIndex, vertexOffset, firstInstance);
                }
                void BindIndexBuffer(Buffer& buffer, uint64_t offset, bool index32) override
                {
                    vkCmdBindIndexBuffer(m_cmd, static_cast<VkBufferObject&>(buffer).Handle(), offset,
                        index32 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
                }
                void Dispatch(uint32_t x, uint32_t y, uint32_t z) override { vkCmdDispatch(m_cmd, x, y, z); }

                void CopyBuffer(Buffer& src, Buffer& dst, uint64_t srcOffset, uint64_t dstOffset, uint64_t size) override
                {
                    VkBufferCopy region{ srcOffset, dstOffset, size };
                    vkCmdCopyBuffer(m_cmd, static_cast<VkBufferObject&>(src).Handle(), static_cast<VkBufferObject&>(dst).Handle(), 1, &region);
                }
                void CopyBufferToImage(Buffer& src, Image& dst, const BufferImageCopy* regions, uint32_t count) override
                {
                    auto& image = static_cast<VkImageObject&>(dst);
                    std::vector<VkBufferImageCopy> vk = CopyRegions(image, regions, count);
                    vkCmdCopyBufferToImage(m_cmd, static_cast<VkBufferObject&>(src).Handle(), image.Handle(),
                        image.Layout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL), uint32_t(vk.size()), vk.data());
                }
                void CopyImageToBuffer(Image& src, Buffer& dst, const BufferImageCopy* regions, uint32_t count) override
                {
                    auto& image = static_cast<VkImageObject&>(src);
                    std::vector<VkBufferImageCopy> vk = CopyRegions(image, regions, count);
                    vkCmdCopyImageToBuffer(m_cmd, image.Handle(), image.Layout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
                        static_cast<VkBufferObject&>(dst).Handle(), uint32_t(vk.size()), vk.data());
                }
                void BlitImage(Image& src, const Rect2D& srcRect, Image& dst, const Rect2D& dstRect, Filter filter) override
                {
                    auto& s = static_cast<VkImageObject&>(src);
                    auto& d = static_cast<VkImageObject&>(dst);
                    VkImageBlit blit{};
                    blit.srcSubresource = { Aspect(s.Desc().format), 0, 0, 1 };
                    blit.dstSubresource = { Aspect(d.Desc().format), 0, 0, 1 };
                    blit.srcOffsets[0] = { srcRect.x, srcRect.y, 0 };
                    blit.srcOffsets[1] = { srcRect.x + int32_t(srcRect.width), srcRect.y + int32_t(srcRect.height), 1 };
                    blit.dstOffsets[0] = { dstRect.x, dstRect.y, 0 };
                    blit.dstOffsets[1] = { dstRect.x + int32_t(dstRect.width), dstRect.y + int32_t(dstRect.height), 1 };
                    vkCmdBlitImage(m_cmd, s.Handle(), s.Layout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL), d.Handle(),
                        d.Layout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL), 1, &blit, ToVk(filter));
                }
                void FillBuffer(Buffer& dst, uint64_t offset, uint64_t size, uint32_t value) override
                {
                    vkCmdFillBuffer(m_cmd, static_cast<VkBufferObject&>(dst).Handle(), offset, size, value);
                }
                void UpdateBuffer(Buffer& dst, uint64_t offset, const void* data, uint64_t size) override
                {
                    vkCmdUpdateBuffer(m_cmd, static_cast<VkBufferObject&>(dst).Handle(), offset, size, data);
                }
                // (An image to clear is in State::CopyDst.)
                void ClearImage(Image& image, const ClearColor& color) override
                {
                    auto& i = static_cast<VkImageObject&>(image);
                    VkClearColorValue value;
                    memcpy(value.float32, color.rgba, sizeof(color.rgba));
                    VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS };
                    vkCmdClearColorImage(m_cmd, i.Handle(), i.Layout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL), &value, 1, &range);
                }
                void ClearImage(Image& image, const ClearDepthStencil& depthStencil) override
                {
                    auto& i = static_cast<VkImageObject&>(image);
                    VkClearDepthStencilValue value{ depthStencil.depth, depthStencil.stencil };
                    VkImageSubresourceRange range{ Aspect(i.Desc().format), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS };
                    vkCmdClearDepthStencilImage(m_cmd, i.Handle(), i.Layout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL), &value, 1, &range);
                }

                void Transition(const ImageTransition* images, size_t imageCount, const BufferTransition* buffers, size_t bufferCount) override
                {
                    VkPipelineStageFlags srcStages = 0, dstStages = 0;
                    bool generalBarrier = false;
                    std::vector<VkBufferMemoryBarrier> bufferBarriers;
                    for (size_t i = 0; i < bufferCount; i++)
                    {
                        VkPipelineStageFlags s, d;
                        VkAccessFlags sa, da;
                        BufferUse(buffers[i].before, s, sa);
                        BufferUse(buffers[i].after, d, da);
                        srcStages |= s;
                        dstStages |= d;
                        VkBufferMemoryBarrier b{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
                        b.srcAccessMask = sa;
                        b.dstAccessMask = da;
                        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        b.buffer = static_cast<VkBufferObject*>(buffers[i].buffer)->Handle();
                        b.size = VK_WHOLE_SIZE;
                        bufferBarriers.push_back(b);
                    }
                    std::vector<VkImageMemoryBarrier> imageBarriers;
                    for (size_t i = 0; i < imageCount; i++)
                    {
                        auto* image = static_cast<VkImageObject*>(images[i].image);
                        if (image->Desc().general && images[i].before != State::Undefined)
                        {
                            // A general image changes no layout; this is the dependency its
                            // users ask for (another submission's writes -> this list's work).
                            srcStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                            dstStages |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                            generalBarrier = true;
                            continue;
                        }
                        VkPipelineStageFlags s, d;
                        VkAccessFlags sa, da;
                        VkImageLayout from, to;
                        ImageUse(images[i].before, from, s, sa);
                        ImageUse(images[i].after, to, d, da);
                        srcStages |= s;
                        dstStages |= d;
                        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
                        b.srcAccessMask = sa;
                        b.dstAccessMask = da;
                        b.oldLayout = from;
                        b.newLayout = to;
                        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        b.image = image->Handle();
                        b.subresourceRange = { Aspect(image->Desc().format), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS };
                        imageBarriers.push_back(b);
                    }
                    VkMemoryBarrier memory{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT,
                        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
                    if (generalBarrier || !bufferBarriers.empty() || !imageBarriers.empty())
                        vkCmdPipelineBarrier(m_cmd, srcStages, dstStages, 0, generalBarrier ? 1 : 0, &memory, uint32_t(bufferBarriers.size()),
                            bufferBarriers.data(), uint32_t(imageBarriers.size()), imageBarriers.data());
                }
                void MemoryBarrier(Barrier kind) override
                {
                    constexpr VkPipelineStageFlags kNonVertex = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
                    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
                    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                    VkPipelineStageFlags src, dst;
                    switch (kind)
                    {
                    case Barrier::Full:
                        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                        src = dst = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                        break;
                    case Barrier::Compute:
                        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        src = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                        dst = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                        break;
                    case Barrier::NarrowAfterPass:
                        // The vertex stage is named in the source although Vulkan
                        // implies it: MoltenVK maps stages one to one, and without
                        // it a resolve overwriting what this pass's vertex shader
                        // read would wait only for its fragment stage.
                        mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                        src = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                        dst = kNonVertex;
                        break;
                    default:  // NarrowAfterCompute
                        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        src = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                        dst = kNonVertex;
                        break;
                    }
                    vkCmdPipelineBarrier(m_cmd, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
                }
                void SignalEvent(Event& event) override
                {
                    vkCmdSetEvent(m_cmd, static_cast<VkEventObject&>(event).Handle(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                }

                void ResetQueries(QueryPool& pool, uint32_t first, uint32_t count) override
                {
                    vkCmdResetQueryPool(m_cmd, static_cast<VkQueryPoolObject&>(pool).Handle(), first, count);
                }
                void WriteTimestamp(QueryPool& pool, uint32_t index, bool end) override
                {
                    vkCmdWriteTimestamp(m_cmd, end ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        static_cast<VkQueryPoolObject&>(pool).Handle(), index);
                }
                void BeginQuery(QueryPool& pool, uint32_t index) override
                {
                    vkCmdBeginQuery(m_cmd, static_cast<VkQueryPoolObject&>(pool).Handle(), index,
                        m_c.preciseOcclusion ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
                }
                void EndQuery(QueryPool& pool, uint32_t index) override
                {
                    vkCmdEndQuery(m_cmd, static_cast<VkQueryPoolObject&>(pool).Handle(), index);
                }

            private:
                static std::vector<VkBufferImageCopy> CopyRegions(const VkImageObject& image, const BufferImageCopy* regions, uint32_t count)
                {
                    std::vector<VkBufferImageCopy> out;
                    for (uint32_t i = 0; i < count; i++)
                    {
                        const BufferImageCopy& r = regions[i];
                        VkBufferImageCopy c{};
                        c.bufferOffset = r.bufferOffset;
                        c.bufferRowLength = r.bufferRowLength;
                        c.imageSubresource = { Aspect(image.Desc().format), r.level, r.layer, r.layerCount };
                        c.imageOffset = { r.region.x, r.region.y, int32_t(r.z) };
                        c.imageExtent = { r.region.width, r.region.height, r.depth };
                        out.push_back(c);
                    }
                    return out;
                }

                Context& m_c;
                VkCommandPool m_pool = VK_NULL_HANDLE;
                VkCommandBuffer m_cmd = VK_NULL_HANDLE;
                VkPipelineBindPoint m_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            };
        }

        std::unique_ptr<CommandList> MakeCommandList(Context& c) { return std::make_unique<VkCommandList>(c); }
        VkCommandBuffer CommandBufferOf(const CommandList& list) { return static_cast<const VkCommandList&>(list).Handle(); }
    }
}
