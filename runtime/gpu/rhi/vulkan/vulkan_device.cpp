// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
#include "vulkan_device.h"
#include "vulkan_native.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "../command_list.h"

namespace gpu::rhi::vulkan
{
    namespace
    {
        [[noreturn]] void NotImplemented(const char* what)
        {
            fprintf(stderr, "[rhi/vulkan] %s: not implemented yet\n", what);
            abort();
        }

        void Check(VkResult r, const char* what)
        {
            if (r != VK_SUCCESS)
            {
                fprintf(stderr, "[rhi/vulkan] %s failed (%d)\n", what, int(r));
                abort();
            }
        }

        // What every object needs of its device. The device owns it: objects
        // must be destroyed before the device, as in Vulkan.
        struct Context
        {
            VkInstance instance = VK_NULL_HANDLE;
            VkPhysicalDevice physical = VK_NULL_HANDLE;
            VkDevice device = VK_NULL_HANDLE;
            uint32_t family = 0;
            VkQueue queue = VK_NULL_HANDLE;
            std::mutex queueMutex;  // every vkQueueSubmit
            VkPhysicalDeviceMemoryProperties memory{};
            VkDeviceSize nonCoherentAtom = 1;
            VkPipelineCache pipelineCache = VK_NULL_HANDLE;
            bool preciseOcclusion = false;
            PFN_vkCmdPushDescriptorSetKHR pushDescriptor = nullptr;
            PFN_vkGetMemoryHostPointerPropertiesEXT hostPointerProperties = nullptr;
            std::string deviceName;
            Capabilities caps;
            Native native;

            uint32_t FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const
            {
                for (uint32_t i = 0; i < memory.memoryTypeCount; i++)
                    if ((typeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & required) == required)
                        return i;
                return UINT32_MAX;
            }
        };

        VkBufferUsageFlags ToVk(BufferUsage usage)
        {
            return VkBufferUsageFlags(
                ((usage & kBufferStorage) ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : 0) |
                ((usage & kBufferUniform) ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : 0) |
                ((usage & kBufferIndex) ? VK_BUFFER_USAGE_INDEX_BUFFER_BIT : 0) |
                ((usage & kBufferCopySrc) ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 0) |
                ((usage & kBufferCopyDst) ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0));
        }

        VkShaderStageFlags ToVkStages(uint8_t stages)
        {
            return VkShaderStageFlags(
                ((stages & uint8_t(ShaderStage::Vertex)) ? VK_SHADER_STAGE_VERTEX_BIT : 0) |
                ((stages & uint8_t(ShaderStage::Fragment)) ? VK_SHADER_STAGE_FRAGMENT_BIT : 0) |
                ((stages & uint8_t(ShaderStage::Compute)) ? VK_SHADER_STAGE_COMPUTE_BIT : 0));
        }

        VkDescriptorType ToVk(BindingKind kind)
        {
            switch (kind)
            {
            case BindingKind::StorageBuffer: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            case BindingKind::UniformBufferDynamic: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            case BindingKind::CombinedImageSampler: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            case BindingKind::StorageImage: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            }
            return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }

        // The stages and accesses a buffer in `state` is used by (D9).
        void BufferUse(State state, VkPipelineStageFlags& stages, VkAccessFlags& access)
        {
            constexpr VkPipelineStageFlags kShaders = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            switch (state)
            {
            case State::ShaderRead: stages = kShaders; access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT; break;
            case State::ShaderWrite: stages = kShaders; access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; break;
            case State::CopySrc: stages = VK_PIPELINE_STAGE_TRANSFER_BIT; access = VK_ACCESS_TRANSFER_READ_BIT; break;
            case State::CopyDst: stages = VK_PIPELINE_STAGE_TRANSFER_BIT; access = VK_ACCESS_TRANSFER_WRITE_BIT; break;
            case State::IndexRead: stages = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT; access = VK_ACCESS_INDEX_READ_BIT; break;
            case State::HostRead: stages = VK_PIPELINE_STAGE_HOST_BIT; access = VK_ACCESS_HOST_READ_BIT; break;
            default: stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; access = 0; break;
            }
        }

        // ---- resources -------------------------------------------------------

        class VkBufferObject final : public Buffer
        {
        public:
            VkBufferObject(Context& c, VkBuffer buffer, VkDeviceMemory memory, uint64_t size, void* mapped, bool coherent)
                : m_c(c), m_buffer(buffer), m_memory(memory), m_size(size), m_mapped(mapped), m_coherent(coherent) {}
            ~VkBufferObject() override
            {
                vkDestroyBuffer(m_c.device, m_buffer, nullptr);
                if (m_mapped && !m_imported)
                    vkUnmapMemory(m_c.device, m_memory);
                vkFreeMemory(m_c.device, m_memory, nullptr);
            }
            uint64_t Size() const override { return m_size; }
            void* Map() override { return m_mapped; }
            void Invalidate(uint64_t offset, uint64_t size) override
            {
                if (m_coherent || !m_mapped)
                    return;
                const VkDeviceSize atom = m_c.nonCoherentAtom;
                VkDeviceSize begin = offset / atom * atom;
                VkDeviceSize end = std::min<VkDeviceSize>((offset + size + atom - 1) / atom * atom, m_size);
                VkMappedMemoryRange range{ VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, nullptr, m_memory, begin, end - begin };
                vkInvalidateMappedMemoryRanges(m_c.device, 1, &range);
            }
            VkBuffer Handle() const { return m_buffer; }
            void MarkImported() { m_imported = true; }

        private:
            Context& m_c;
            VkBuffer m_buffer;
            VkDeviceMemory m_memory;
            uint64_t m_size;
            void* m_mapped;
            bool m_coherent;
            bool m_imported = false;  // m_mapped is the caller's pointer: nothing to unmap
        };

        class VkSamplerObject final : public Sampler
        {
        public:
            VkSamplerObject(Context& c, VkSampler sampler) : m_c(c), m_sampler(sampler) {}
            ~VkSamplerObject() override { vkDestroySampler(m_c.device, m_sampler, nullptr); }
            VkSampler Handle() const { return m_sampler; }

        private:
            Context& m_c;
            VkSampler m_sampler;
        };

        // ---- bindings (D10) ------------------------------------------------------

        class VkBindingLayout final : public BindingLayout
        {
        public:
            VkBindingLayout(Context& c, BindingLayoutDesc desc) : m_c(c), m_desc(std::move(desc))
            {
                for (const BindingTableDesc& table : m_desc.tables)
                {
                    std::vector<VkSampler> immutable(table.slots.size());
                    std::vector<VkDescriptorSetLayoutBinding> bindings;
                    for (size_t i = 0; i < table.slots.size(); i++)
                    {
                        const BindingSlot& slot = table.slots[i];
                        VkDescriptorSetLayoutBinding b{ slot.index, ToVk(slot.kind), 1, ToVkStages(slot.stages), nullptr };
                        if (slot.immutableSampler)
                        {
                            immutable[i] = static_cast<const VkSamplerObject*>(slot.immutableSampler)->Handle();
                            b.pImmutableSamplers = &immutable[i];
                        }
                        bindings.push_back(b);
                    }
                    VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
                    ci.flags = table.pushed ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0;
                    ci.bindingCount = uint32_t(bindings.size());
                    ci.pBindings = bindings.data();
                    VkDescriptorSetLayout set;
                    Check(vkCreateDescriptorSetLayout(c.device, &ci, nullptr, &set), "vkCreateDescriptorSetLayout");
                    m_sets.push_back(set);
                }
                VkPushConstantRange range{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0,
                    m_desc.pushConstantBytes };
                VkPipelineLayoutCreateInfo pl{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
                pl.setLayoutCount = uint32_t(m_sets.size());
                pl.pSetLayouts = m_sets.data();
                pl.pushConstantRangeCount = m_desc.pushConstantBytes ? 1 : 0;
                pl.pPushConstantRanges = &range;
                Check(vkCreatePipelineLayout(c.device, &pl, nullptr, &m_layout), "vkCreatePipelineLayout");
            }
            ~VkBindingLayout() override
            {
                vkDestroyPipelineLayout(m_c.device, m_layout, nullptr);
                for (VkDescriptorSetLayout set : m_sets)
                    vkDestroyDescriptorSetLayout(m_c.device, set, nullptr);
            }
            VkPipelineLayout Handle() const { return m_layout; }
            VkDescriptorSetLayout Set(uint32_t table) const { return m_sets.at(table); }
            const BindingTableDesc& Table(uint32_t table) const { return m_desc.tables.at(table); }
            const BindingSlot* Find(uint32_t table, uint32_t slot) const
            {
                for (const BindingSlot& s : Table(table).slots)
                    if (s.index == slot)
                        return &s;
                return nullptr;
            }

        private:
            Context& m_c;
            BindingLayoutDesc m_desc;
            std::vector<VkDescriptorSetLayout> m_sets;
            VkPipelineLayout m_layout = VK_NULL_HANDLE;
        };

        // Writes `entries` of table `table` as descriptor writes (into a set,
        // or pushed); the infos are kept in the vectors until the call.
        struct Writes
        {
            std::vector<VkWriteDescriptorSet> writes;
            std::vector<VkDescriptorBufferInfo> buffers;

            Writes(const VkBindingLayout& layout, uint32_t table, const BindingEntry* entries, size_t count)
            {
                buffers.reserve(count);  // (the writes point into it)
                for (size_t i = 0; i < count; i++)
                {
                    const BindingSlot* slot = layout.Find(table, entries[i].slot);
                    if (!slot)
                    {
                        fprintf(stderr, "[rhi/vulkan] binding table %u has no slot %u\n", table, entries[i].slot);
                        abort();
                    }
                    if (slot->kind == BindingKind::CombinedImageSampler || slot->kind == BindingKind::StorageImage)
                        NotImplemented("image bindings");
                    const BufferBinding& b = entries[i].buffer;
                    buffers.push_back({ static_cast<VkBufferObject*>(b.buffer)->Handle(), b.offset, b.size == ~0ull ? VK_WHOLE_SIZE : b.size });
                    VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                    w.dstBinding = slot->index;
                    w.descriptorCount = 1;
                    w.descriptorType = ToVk(slot->kind);
                    w.pBufferInfo = &buffers.back();
                    writes.push_back(w);
                }
            }
        };

        class VkBindingSet final : public BindingSet
        {
        public:
            VkBindingSet(Context& c, const VkBindingLayout& layout, uint32_t table) : m_c(c), m_layout(layout), m_table(table)
            {
                std::vector<VkDescriptorPoolSize> sizes;
                for (const BindingSlot& slot : layout.Table(table).slots)
                {
                    VkDescriptorType type = ToVk(slot.kind);
                    auto it = std::find_if(sizes.begin(), sizes.end(), [&](const VkDescriptorPoolSize& s) { return s.type == type; });
                    if (it == sizes.end())
                        sizes.push_back({ type, 1 });
                    else
                        it->descriptorCount++;
                }
                VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
                pci.maxSets = 1;
                pci.poolSizeCount = uint32_t(sizes.size());
                pci.pPoolSizes = sizes.data();
                Check(vkCreateDescriptorPool(c.device, &pci, nullptr, &m_pool), "vkCreateDescriptorPool");
                VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
                ai.descriptorPool = m_pool;
                ai.descriptorSetCount = 1;
                VkDescriptorSetLayout setLayout = layout.Set(table);
                ai.pSetLayouts = &setLayout;
                Check(vkAllocateDescriptorSets(c.device, &ai, &m_set), "vkAllocateDescriptorSets");
            }
            ~VkBindingSet() override { vkDestroyDescriptorPool(m_c.device, m_pool, nullptr); }
            void Update(const BindingEntry* entries, size_t count) override
            {
                Writes w(m_layout, m_table, entries, count);
                for (VkWriteDescriptorSet& write : w.writes)
                    write.dstSet = m_set;
                vkUpdateDescriptorSets(m_c.device, uint32_t(w.writes.size()), w.writes.data(), 0, nullptr);
            }
            VkDescriptorSet Handle() const { return m_set; }

        private:
            Context& m_c;
            const VkBindingLayout& m_layout;
            uint32_t m_table;
            VkDescriptorPool m_pool = VK_NULL_HANDLE;
            VkDescriptorSet m_set = VK_NULL_HANDLE;
        };

        // ---- pipelines -----------------------------------------------------------

        class VkPipelineObject final : public Pipeline
        {
        public:
            VkPipelineObject(Context& c, VkPipeline pipeline, VkPipelineBindPoint point) : m_c(c), m_pipeline(pipeline), m_point(point) {}
            ~VkPipelineObject() override { vkDestroyPipeline(m_c.device, m_pipeline, nullptr); }
            VkPipeline Handle() const { return m_pipeline; }
            VkPipelineBindPoint Point() const { return m_point; }

        private:
            Context& m_c;
            VkPipeline m_pipeline;
            VkPipelineBindPoint m_point;
        };

        // ---- synchronisation -----------------------------------------------------

        class VkFenceObject final : public Fence
        {
        public:
            explicit VkFenceObject(Context& c) : m_c(c)
            {
                VkFenceCreateInfo ci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
                Check(vkCreateFence(c.device, &ci, nullptr, &m_fence), "vkCreateFence");
            }
            ~VkFenceObject() override { vkDestroyFence(m_c.device, m_fence, nullptr); }
            bool Signaled() override { return vkGetFenceStatus(m_c.device, m_fence) == VK_SUCCESS; }
            void Wait() override { Check(vkWaitForFences(m_c.device, 1, &m_fence, VK_TRUE, UINT64_MAX), "vkWaitForFences"); }
            void Reset() override { Check(vkResetFences(m_c.device, 1, &m_fence), "vkResetFences"); }
            VkFence Handle() const { return m_fence; }

        private:
            Context& m_c;
            VkFence m_fence = VK_NULL_HANDLE;
        };

        class VkEventObject final : public Event
        {
        public:
            explicit VkEventObject(Context& c) : m_c(c)
            {
                VkEventCreateInfo ci{ VK_STRUCTURE_TYPE_EVENT_CREATE_INFO };
                Check(vkCreateEvent(c.device, &ci, nullptr, &m_event), "vkCreateEvent");
            }
            ~VkEventObject() override { vkDestroyEvent(m_c.device, m_event, nullptr); }
            bool Signaled() override { return vkGetEventStatus(m_c.device, m_event) == VK_EVENT_SET; }
            void Reset() override { Check(vkResetEvent(m_c.device, m_event), "vkResetEvent"); }
            VkEvent Handle() const { return m_event; }

        private:
            Context& m_c;
            VkEvent m_event = VK_NULL_HANDLE;
        };

        class VkQueryPoolObject final : public QueryPool
        {
        public:
            VkQueryPoolObject(Context& c, QueryKind kind, uint32_t count) : m_c(c), m_kind(kind)
            {
                VkQueryPoolCreateInfo ci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
                ci.queryType = kind == QueryKind::Timestamp ? VK_QUERY_TYPE_TIMESTAMP : VK_QUERY_TYPE_OCCLUSION;
                ci.queryCount = count;
                Check(vkCreateQueryPool(c.device, &ci, nullptr, &m_pool), "vkCreateQueryPool");
            }
            ~VkQueryPoolObject() override { vkDestroyQueryPool(m_c.device, m_pool, nullptr); }
            bool Read(uint32_t first, uint32_t count, uint64_t* out) override
            {
                return vkGetQueryPoolResults(m_c.device, m_pool, first, count, count * sizeof(uint64_t), out, sizeof(uint64_t),
                    VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
            }
            VkQueryPool Handle() const { return m_pool; }
            QueryKind Kind() const { return m_kind; }

        private:
            Context& m_c;
            QueryKind m_kind;
            VkQueryPool m_pool = VK_NULL_HANDLE;
        };

        // ---- command list --------------------------------------------------------

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

            void BeginRendering(const RenderingDesc&) override { NotImplemented("BeginRendering"); }
            void EndRendering() override { NotImplemented("EndRendering"); }
            void ClearAttachments(const Rect2D&, const ClearColor*, uint32_t, const ClearDepthStencil*, bool, bool) override
            {
                NotImplemented("ClearAttachments");
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
            void SetStencil(uint8_t reference, uint8_t compareMask, uint8_t writeMask) override
            {
                vkCmdSetStencilReference(m_cmd, VK_STENCIL_FACE_FRONT_AND_BACK, reference);
                vkCmdSetStencilCompareMask(m_cmd, VK_STENCIL_FACE_FRONT_AND_BACK, compareMask);
                vkCmdSetStencilWriteMask(m_cmd, VK_STENCIL_FACE_FRONT_AND_BACK, writeMask);
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
            void CopyBufferToImage(Buffer&, Image&, const BufferImageCopy*, uint32_t) override { NotImplemented("CopyBufferToImage"); }
            void CopyImageToBuffer(Image&, Buffer&, const BufferImageCopy*, uint32_t) override { NotImplemented("CopyImageToBuffer"); }
            void BlitImage(Image&, const Rect2D&, Image&, const Rect2D&, Filter) override { NotImplemented("BlitImage"); }
            void FillBuffer(Buffer& dst, uint64_t offset, uint64_t size, uint32_t value) override
            {
                vkCmdFillBuffer(m_cmd, static_cast<VkBufferObject&>(dst).Handle(), offset, size, value);
            }
            void UpdateBuffer(Buffer& dst, uint64_t offset, const void* data, uint64_t size) override
            {
                vkCmdUpdateBuffer(m_cmd, static_cast<VkBufferObject&>(dst).Handle(), offset, size, data);
            }
            void ClearImage(Image&, const ClearColor&) override { NotImplemented("ClearImage"); }
            void ClearImage(Image&, const ClearDepthStencil&) override { NotImplemented("ClearImage"); }

            void Transition(const ImageTransition*, size_t imageCount, const BufferTransition* buffers, size_t bufferCount) override
            {
                if (imageCount)
                    NotImplemented("image transitions");
                VkPipelineStageFlags srcStages = 0, dstStages = 0;
                std::vector<VkBufferMemoryBarrier> barriers;
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
                    barriers.push_back(b);
                }
                if (!barriers.empty())
                    vkCmdPipelineBarrier(m_cmd, srcStages, dstStages, 0, 0, nullptr, uint32_t(barriers.size()), barriers.data(), 0, nullptr);
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
            Context& m_c;
            VkCommandPool m_pool = VK_NULL_HANDLE;
            VkCommandBuffer m_cmd = VK_NULL_HANDLE;
            VkPipelineBindPoint m_point = VK_PIPELINE_BIND_POINT_COMPUTE;
        };

        // ---- the device ----------------------------------------------------------

        class VkDeviceObject final : public Device
        {
        public:
            explicit VkDeviceObject(std::unique_ptr<Context> context) : m_c(std::move(context)) {}
            ~VkDeviceObject() override
            {
                vkDeviceWaitIdle(m_c->device);
                vkDestroyPipelineCache(m_c->device, m_c->pipelineCache, nullptr);
                vkDestroyDevice(m_c->device, nullptr);
                vkDestroyInstance(m_c->instance, nullptr);
            }

            const Capabilities& GetCapabilities() const override { return m_c->caps; }
            const vulkan::Native& Native() const { return m_c->native; }

            std::unique_ptr<Buffer> CreateBuffer(const BufferDesc& desc) override
            {
                VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.size = desc.size;
                bci.usage = ToVk(BufferUsage(desc.usage));
                VkBuffer buffer;
                if (vkCreateBuffer(m_c->device, &bci, nullptr, &buffer) != VK_SUCCESS)
                    return nullptr;
                VkMemoryRequirements req;
                vkGetBufferMemoryRequirements(m_c->device, buffer, &req);
                VkMemoryPropertyFlags want = desc.memory == MemoryKind::Device
                    ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                    : desc.memory == MemoryKind::Upload
                        ? VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                        : VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                uint32_t type = m_c->FindMemoryType(req.memoryTypeBits, want);
                VkDeviceMemory memory = VK_NULL_HANDLE;
                if (type != UINT32_MAX)
                {
                    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                    mai.allocationSize = req.size;
                    mai.memoryTypeIndex = type;
                    if (vkAllocateMemory(m_c->device, &mai, nullptr, &memory) != VK_SUCCESS)
                        memory = VK_NULL_HANDLE;
                }
                if (!memory || vkBindBufferMemory(m_c->device, buffer, memory, 0) != VK_SUCCESS)
                {
                    vkDestroyBuffer(m_c->device, buffer, nullptr);
                    if (memory)
                        vkFreeMemory(m_c->device, memory, nullptr);
                    return nullptr;
                }
                void* mapped = nullptr;
                if (desc.memory != MemoryKind::Device)
                    Check(vkMapMemory(m_c->device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
                bool coherent = (m_c->memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
                return std::make_unique<VkBufferObject>(*m_c, buffer, memory, desc.size, mapped, coherent);
            }

            std::unique_ptr<Buffer> ImportHostMemory(void* host, uint64_t size, uint32_t usage, const char*) override
            {
                if (!m_c->hostPointerProperties)
                    return nullptr;
                const uint64_t align = m_c->caps.hostMemoryAlignment;
                if (reinterpret_cast<uintptr_t>(host) % align || size % align)
                {
                    fprintf(stderr, "[rhi/vulkan] host memory import: %p + %llu isn't aligned to %llu\n", host, (unsigned long long)size,
                        (unsigned long long)align);
                    return nullptr;
                }
                VkMemoryHostPointerPropertiesEXT hpp{ VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
                if (m_c->hostPointerProperties(m_c->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &hpp) != VK_SUCCESS)
                    return nullptr;
                VkExternalMemoryBufferCreateInfo embci{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
                embci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
                VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.pNext = &embci;
                bci.size = size;
                bci.usage = ToVk(BufferUsage(usage));
                VkBuffer buffer;
                if (vkCreateBuffer(m_c->device, &bci, nullptr, &buffer) != VK_SUCCESS)
                    return nullptr;
                VkMemoryRequirements req;
                vkGetBufferMemoryRequirements(m_c->device, buffer, &req);
                VkImportMemoryHostPointerInfoEXT import{ VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT };
                import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
                import.pHostPointer = host;
                VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                mai.pNext = &import;
                mai.allocationSize = size;
                mai.memoryTypeIndex = m_c->FindMemoryType(req.memoryTypeBits & hpp.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                VkDeviceMemory memory = VK_NULL_HANDLE;
                if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(m_c->device, &mai, nullptr, &memory) != VK_SUCCESS ||
                    vkBindBufferMemory(m_c->device, buffer, memory, 0) != VK_SUCCESS)
                {
                    vkDestroyBuffer(m_c->device, buffer, nullptr);
                    if (memory)
                        vkFreeMemory(m_c->device, memory, nullptr);
                    return nullptr;
                }
                auto result = std::make_unique<VkBufferObject>(*m_c, buffer, memory, size, host, true);
                result->MarkImported();
                return result;
            }

            std::unique_ptr<Image> CreateImage(const ImageDesc&) override { NotImplemented("CreateImage"); }

            std::unique_ptr<Sampler> CreateSampler(const SamplerDesc& desc) override
            {
                auto filter = [](Filter f) { return f == Filter::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR; };
                auto address = [](AddressMode a) {
                    return a == AddressMode::ClampToEdge ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                        : a == AddressMode::MirrorRepeat ? VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT : VK_SAMPLER_ADDRESS_MODE_REPEAT;
                };
                VkSamplerCreateInfo ci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
                ci.magFilter = filter(desc.mag);
                ci.minFilter = filter(desc.min);
                ci.mipmapMode = desc.mip == Filter::Nearest ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
                ci.addressModeU = address(desc.u);
                ci.addressModeV = address(desc.v);
                ci.addressModeW = address(desc.w);
                ci.anisotropyEnable = desc.maxAnisotropy > 1.0f ? VK_TRUE : VK_FALSE;
                ci.maxAnisotropy = desc.maxAnisotropy;
                ci.maxLod = VK_LOD_CLAMP_NONE;
                VkSampler sampler;
                Check(vkCreateSampler(m_c->device, &ci, nullptr, &sampler), "vkCreateSampler");
                return std::make_unique<VkSamplerObject>(*m_c, sampler);
            }

            std::unique_ptr<BindingLayout> CreateBindingLayout(const BindingLayoutDesc& desc) override
            {
                return std::make_unique<VkBindingLayout>(*m_c, desc);
            }
            std::unique_ptr<BindingSet> CreateBindingSet(const BindingLayout& layout, uint32_t table) override
            {
                return std::make_unique<VkBindingSet>(*m_c, static_cast<const VkBindingLayout&>(layout), table);
            }

            std::unique_ptr<Pipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc&) override { NotImplemented("CreateGraphicsPipeline"); }

            std::unique_ptr<Pipeline> CreateComputePipeline(const ComputePipelineDesc& desc) override
            {
                VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
                smci.codeSize = desc.code.words * 4;
                smci.pCode = desc.code.spirv;
                VkShaderModule module;
                if (vkCreateShaderModule(m_c->device, &smci, nullptr, &module) != VK_SUCCESS)
                    return nullptr;
                VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
                cpci.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr };
                cpci.layout = static_cast<const VkBindingLayout*>(desc.layout)->Handle();
                VkPipeline pipeline = VK_NULL_HANDLE;
                VkResult r = vkCreateComputePipelines(m_c->device, m_c->pipelineCache, 1, &cpci, nullptr, &pipeline);
                vkDestroyShaderModule(m_c->device, module, nullptr);
                if (r != VK_SUCCESS)
                    return nullptr;
                return std::make_unique<VkPipelineObject>(*m_c, pipeline, VK_PIPELINE_BIND_POINT_COMPUTE);
            }

            std::vector<uint8_t> SavePipelineCache() override
            {
                size_t size = 0;
                vkGetPipelineCacheData(m_c->device, m_c->pipelineCache, &size, nullptr);
                std::vector<uint8_t> data(size);
                if (size)
                    vkGetPipelineCacheData(m_c->device, m_c->pipelineCache, &size, data.data());
                data.resize(size);
                return data;
            }
            void LoadPipelineCache(const void* data, size_t size) override
            {
                VkPipelineCacheCreateInfo ci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
                ci.initialDataSize = size;
                ci.pInitialData = data;
                VkPipelineCache cache;
                if (vkCreatePipelineCache(m_c->device, &ci, nullptr, &cache) != VK_SUCCESS)
                    return;  // (the old, empty one stays)
                vkDestroyPipelineCache(m_c->device, m_c->pipelineCache, nullptr);
                m_c->pipelineCache = cache;
            }

            std::unique_ptr<CommandList> CreateCommandList() override { return std::make_unique<VkCommandList>(*m_c); }
            std::unique_ptr<Fence> CreateFence() override { return std::make_unique<VkFenceObject>(*m_c); }
            std::unique_ptr<Event> CreateEvent() override { return std::make_unique<VkEventObject>(*m_c); }
            std::unique_ptr<QueryPool> CreateQueryPool(QueryKind kind, uint32_t count) override
            {
                return std::make_unique<VkQueryPoolObject>(*m_c, kind, count);
            }

            void Submit(CommandList& list, Fence& fence) override
            {
                VkCommandBuffer cmd = static_cast<VkCommandList&>(list).Handle();
                VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                si.commandBufferCount = 1;
                si.pCommandBuffers = &cmd;
                std::lock_guard lock(m_c->queueMutex);
                Check(vkQueueSubmit(m_c->queue, 1, &si, static_cast<VkFenceObject&>(fence).Handle()), "vkQueueSubmit");
            }
            void WaitIdle() override
            {
                std::lock_guard lock(m_c->queueMutex);
                vkDeviceWaitIdle(m_c->device);
            }

        private:
            std::unique_ptr<Context> m_c;
        };

        bool HasExtension(const std::vector<VkExtensionProperties>& list, const char* name)
        {
            return std::any_of(list.begin(), list.end(), [name](const VkExtensionProperties& e) { return strcmp(e.extensionName, name) == 0; });
        }
    }

    std::unique_ptr<Device> CreateHeadlessDevice()
    {
        auto c = std::make_unique<Context>();

        // The instance. On macOS the loader lists MoltenVK only when asked
        // (portability enumeration); iOS and others may not offer it.
        uint32_t count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> instanceExts(count);
        vkEnumerateInstanceExtensionProperties(nullptr, &count, instanceExts.data());
        std::vector<const char*> exts;
        const bool portability = HasExtension(instanceExts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        if (portability)
            exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        if (HasExtension(instanceExts, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
            exts.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
        app.pApplicationName = "SpeedBreaker";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.flags = portability ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0;
        ici.pApplicationInfo = &app;
        ici.enabledExtensionCount = uint32_t(exts.size());
        ici.ppEnabledExtensionNames = exts.data();
        if (vkCreateInstance(&ici, nullptr, &c->instance) != VK_SUCCESS)
            return nullptr;

        // The device: a discrete GPU first, with a graphics + compute queue.
        count = 0;
        vkEnumeratePhysicalDevices(c->instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(c->instance, &count, devices.data());
        VkPhysicalDeviceProperties props{};
        for (int pass = 0; pass < 2 && !c->physical; pass++)
            for (VkPhysicalDevice d : devices)
            {
                VkPhysicalDeviceProperties p;
                vkGetPhysicalDeviceProperties(d, &p);
                uint32_t qc = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(d, &qc, nullptr);
                std::vector<VkQueueFamilyProperties> qs(qc);
                vkGetPhysicalDeviceQueueFamilyProperties(d, &qc, qs.data());
                for (uint32_t f = 0; f < qc; f++)
                    if ((qs[f].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT) &&
                        (pass == 1 || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU))
                    {
                        c->physical = d;
                        c->family = f;
                        props = p;
                        break;
                    }
                if (c->physical)
                    break;
            }
        if (!c->physical)
        {
            vkDestroyInstance(c->instance, nullptr);
            return nullptr;
        }

        count = 0;
        vkEnumerateDeviceExtensionProperties(c->physical, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> avail(count);
        vkEnumerateDeviceExtensionProperties(c->physical, nullptr, &count, avail.data());
        std::vector<const char*> devExts;
        const bool pushDescriptor = HasExtension(avail, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        const bool hostImport = HasExtension(avail, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        if (pushDescriptor)
            devExts.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        if (hostImport)
            devExts.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        if (HasExtension(avail, "VK_KHR_portability_subset"))
            devExts.push_back("VK_KHR_portability_subset");

        VkPhysicalDeviceFeatures supported;
        vkGetPhysicalDeviceFeatures(c->physical, &supported);
        VkPhysicalDeviceFeatures features{};
        features.vertexPipelineStoresAndAtomics = supported.vertexPipelineStoresAndAtomics;
        features.fragmentStoresAndAtomics = supported.fragmentStoresAndAtomics;
        features.independentBlend = supported.independentBlend;
        features.depthClamp = supported.depthClamp;
        features.samplerAnisotropy = supported.samplerAnisotropy;
        features.textureCompressionBC = supported.textureCompressionBC;
        features.occlusionQueryPrecise = supported.occlusionQueryPrecise;

        DeviceConfig config;
        config.instance = c->instance;
        config.physical = c->physical;
        config.queueFamily = c->family;
        config.extensions = devExts;
        config.features = &features;
        std::unique_ptr<Device> device = CreateDevice(config);
        if (!device)
            vkDestroyInstance(c->instance, nullptr);
        return device;
    }

    std::unique_ptr<Device> CreateDevice(const DeviceConfig& config)
    {
        auto c = std::make_unique<Context>();
        c->instance = config.instance;
        c->physical = config.physical;
        c->family = config.queueFamily;
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(c->physical, &props);
        vkGetPhysicalDeviceMemoryProperties(c->physical, &c->memory);
        c->nonCoherentAtom = std::max<VkDeviceSize>(props.limits.nonCoherentAtomSize, 1);
        auto has = [&](const char* name) {
            return std::any_of(config.extensions.begin(), config.extensions.end(), [name](const char* e) { return strcmp(e, name) == 0; });
        };
        const bool pushDescriptor = has(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        const bool hostImport = has(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        VkPhysicalDeviceFeatures supported;
        vkGetPhysicalDeviceFeatures(c->physical, &supported);
        c->preciseOcclusion = config.features && config.features->occlusionQueryPrecise;

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qci.queueFamilyIndex = c->family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        dci.pNext = config.next;
        dci.pEnabledFeatures = config.features;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = uint32_t(config.extensions.size());
        dci.ppEnabledExtensionNames = config.extensions.data();
        if (VkResult r = vkCreateDevice(c->physical, &dci, nullptr, &c->device); r != VK_SUCCESS)
        {
            fprintf(stderr, "[rhi/vulkan] vkCreateDevice failed (%d)\n", int(r));
            return nullptr;
        }
        vkGetDeviceQueue(c->device, c->family, 0, &c->queue);
        c->native = { c->instance, c->physical, c->device, c->family, c->queue, &c->queueMutex };
        c->pushDescriptor = pushDescriptor ? reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
            vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR")) : nullptr;
        c->hostPointerProperties = hostImport ? reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
            vkGetDeviceProcAddr(c->device, "vkGetMemoryHostPointerPropertiesEXT")) : nullptr;
        VkPipelineCacheCreateInfo pcci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
        Check(vkCreatePipelineCache(c->device, &pcci, nullptr, &c->pipelineCache), "vkCreatePipelineCache");

        // What the device offers.
        c->deviceName = props.deviceName;
        Capabilities& caps = c->caps;
        caps.backend = "Vulkan";
        caps.deviceName = c->deviceName.c_str();
        caps.unifiedMemory = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        caps.canImportHostMemory = c->hostPointerProperties != nullptr;
        caps.hostMemoryAlignment = 4096;
        if (caps.canImportHostMemory)
        {
            VkPhysicalDeviceExternalMemoryHostPropertiesEXT host{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT };
            VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &host };
            vkGetPhysicalDeviceProperties2(c->physical, &props2);
            caps.hostMemoryAlignment = std::max<uint64_t>(host.minImportedHostPointerAlignment, 1);
        }
        caps.maxBufferSize = props.limits.maxStorageBufferRange;
        caps.textureCompressionBC = config.features && config.features->textureCompressionBC;
        caps.occlusionQueries = c->preciseOcclusion;
        caps.timestampQueries = props.limits.timestampComputeAndGraphics;
        return std::make_unique<VkDeviceObject>(std::move(c));
    }

    const Native& GetNative(const rhi::Device& device)
    {
        return static_cast<const VkDeviceObject&>(device).Native();
    }
}
