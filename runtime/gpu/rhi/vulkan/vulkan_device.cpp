// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
#include "vulkan_device.h"
#include "vulkan_native.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <thread>
#include <time.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <mutex>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "../../memory_plan.h"
#include "../../pipeline_cache_header.h"
#include "../command_list.h"
#include "vulkan_image_memory.h"

namespace gpu::rhi::vulkan
{
    namespace
    {
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
            std::function<void(std::unique_lock<std::mutex>&)> submitGate;  // Device::SetSubmitGate
            VkPhysicalDeviceMemoryProperties memory{};
            VkDeviceSize nonCoherentAtom = 1;
            VkPipelineCache pipelineCache = VK_NULL_HANDLE;
            bool preciseOcclusion = false;
            PFN_vkResetQueryPool hostResetQueries = nullptr;  // VK_EXT_host_query_reset enabled (QueryPool::Reset)
            VkPhysicalDeviceProperties props{};
            bool memoryBudget = false;                        // VK_EXT_memory_budget
            std::unique_ptr<ImageMemory> imageMemory;         // destroyed before the device (VkDeviceObject)
            // Guest memory and the ring (gpu/memory_plan.h, D14).
            memplan::Memory planMemory;
            bool planned = false;
            memplan::Gpu planGpu = memplan::Gpu::kOther;
            uint32_t guestType = memplan::kNoType;            // the copy's memory type (kNoType: imported)
            std::string guestChoice;                          // for the "memory" line
            bool shadowInVram = false;
            PFN_vkCmdPushDescriptorSetKHR pushDescriptor = nullptr;
            PFN_vkGetMemoryHostPointerPropertiesEXT hostPointerProperties = nullptr;
            PFN_vkCmdBeginRenderingKHR beginRendering = nullptr;
            PFN_vkCmdEndRenderingKHR endRendering = nullptr;
            PFN_vkWaitForPresentKHR waitForPresent = nullptr;                 // present timing
            PFN_vkGetPastPresentationTimingGOOGLE pastTiming = nullptr;       // display timing
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
            case BindingKind::SampledImage: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
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


        VkFormat ToVk(Format f)
        {
            switch (f)
            {
            case Format::R8: return VK_FORMAT_R8_UNORM;
            case Format::RG8: return VK_FORMAT_R8G8_UNORM;
            case Format::RGBA8: return VK_FORMAT_R8G8B8A8_UNORM;
            case Format::BGRA8: return VK_FORMAT_B8G8R8A8_UNORM;
            case Format::RGBA8_sRGB: return VK_FORMAT_R8G8B8A8_SRGB;
            case Format::BGRA8_sRGB: return VK_FORMAT_B8G8R8A8_SRGB;
            case Format::R16: return VK_FORMAT_R16_UNORM;
            case Format::RG16: return VK_FORMAT_R16G16_UNORM;
            case Format::RGBA16: return VK_FORMAT_R16G16B16A16_UNORM;
            case Format::RG16_SNORM: return VK_FORMAT_R16G16_SNORM;
            case Format::RGBA16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
            case Format::R16F: return VK_FORMAT_R16_SFLOAT;
            case Format::RG16F: return VK_FORMAT_R16G16_SFLOAT;
            case Format::RGBA16F: return VK_FORMAT_R16G16B16A16_SFLOAT;
            case Format::R32F: return VK_FORMAT_R32_SFLOAT;
            case Format::RG32F: return VK_FORMAT_R32G32_SFLOAT;
            case Format::RGBA32F: return VK_FORMAT_R32G32B32A32_SFLOAT;
            case Format::R32_UINT: return VK_FORMAT_R32_UINT;
            case Format::A2B10G10R10: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
            case Format::D32F_S8: return VK_FORMAT_D32_SFLOAT_S8_UINT;
            case Format::BC1: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
            case Format::BC2: return VK_FORMAT_BC2_UNORM_BLOCK;
            case Format::BC3: return VK_FORMAT_BC3_UNORM_BLOCK;
            case Format::BC5: return VK_FORMAT_BC5_UNORM_BLOCK;
            case Format::Undefined: break;
            }
            return VK_FORMAT_UNDEFINED;
        }

        Format FromVk(VkFormat f)
        {
            switch (f)
            {
            case VK_FORMAT_R8G8B8A8_UNORM: return Format::RGBA8;
            case VK_FORMAT_B8G8R8A8_UNORM: return Format::BGRA8;
            case VK_FORMAT_R8G8B8A8_SRGB: return Format::RGBA8_sRGB;
            case VK_FORMAT_B8G8R8A8_SRGB: return Format::BGRA8_sRGB;
            case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return Format::A2B10G10R10;
            case VK_FORMAT_R16G16B16A16_SFLOAT: return Format::RGBA16F;
            default: return Format::Undefined;
            }
        }

        int64_t NowNs()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        bool HasDepth(Format f) { return f == Format::D32F_S8; }

        VkImageAspectFlags Aspect(Format f)
        {
            return HasDepth(f) ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VK_IMAGE_ASPECT_COLOR_BIT;
        }

        VkImageUsageFlags ToVk(ImageUsage usage)
        {
            return VkImageUsageFlags(
                ((usage & kImageSampled) ? VK_IMAGE_USAGE_SAMPLED_BIT : 0) |
                ((usage & kImageStorage) ? VK_IMAGE_USAGE_STORAGE_BIT : 0) |
                ((usage & kImageRenderTarget) ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0) |
                ((usage & kImageDepthStencil) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : 0) |
                ((usage & kImageCopySrc) ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0) |
                ((usage & kImageCopyDst) ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0));
        }

        // What an image in `state` is: its layout, and the stages and accesses that use it (D9).
        void ImageUse(State state, VkImageLayout& layout, VkPipelineStageFlags& stages, VkAccessFlags& access)
        {
            constexpr VkPipelineStageFlags kShaders = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            constexpr VkPipelineStageFlags kDepth = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            switch (state)
            {
            case State::ShaderRead:
                layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; stages = kShaders; access = VK_ACCESS_SHADER_READ_BIT; break;
            case State::ShaderWrite:
                layout = VK_IMAGE_LAYOUT_GENERAL; stages = kShaders; access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; break;
            case State::RenderTarget:
                layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; break;
            case State::DepthRead:
                layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL; stages = kDepth | kShaders;
                access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT; break;
            case State::DepthWrite:
                layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; stages = kDepth;
                access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; break;
            case State::CopySrc:
                layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; stages = VK_PIPELINE_STAGE_TRANSFER_BIT; access = VK_ACCESS_TRANSFER_READ_BIT; break;
            case State::CopyDst:
                layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; stages = VK_PIPELINE_STAGE_TRANSFER_BIT; access = VK_ACCESS_TRANSFER_WRITE_BIT; break;
            case State::Present:
                layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; stages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT; access = 0; break;
            case State::General:
                layout = VK_IMAGE_LAYOUT_GENERAL; stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT; break;
            default:
                layout = VK_IMAGE_LAYOUT_UNDEFINED; stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; access = 0; break;
            }
        }

        VkBlendFactor ToVk(BlendFactor f)
        {
            switch (f)
            {
            case BlendFactor::Zero: return VK_BLEND_FACTOR_ZERO;
            case BlendFactor::One: return VK_BLEND_FACTOR_ONE;
            case BlendFactor::SrcColor: return VK_BLEND_FACTOR_SRC_COLOR;
            case BlendFactor::OneMinusSrcColor: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
            case BlendFactor::DstColor: return VK_BLEND_FACTOR_DST_COLOR;
            case BlendFactor::OneMinusDstColor: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
            case BlendFactor::SrcAlpha: return VK_BLEND_FACTOR_SRC_ALPHA;
            case BlendFactor::OneMinusSrcAlpha: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            case BlendFactor::DstAlpha: return VK_BLEND_FACTOR_DST_ALPHA;
            case BlendFactor::OneMinusDstAlpha: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
            case BlendFactor::ConstantColor: return VK_BLEND_FACTOR_CONSTANT_COLOR;
            case BlendFactor::OneMinusConstantColor: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
            case BlendFactor::ConstantAlpha: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
            case BlendFactor::OneMinusConstantAlpha: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
            case BlendFactor::SrcAlphaSaturate: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
            }
            return VK_BLEND_FACTOR_ONE;
        }

        VkBlendOp ToVk(BlendOp op)
        {
            switch (op)
            {
            case BlendOp::Add: return VK_BLEND_OP_ADD;
            case BlendOp::Subtract: return VK_BLEND_OP_SUBTRACT;
            case BlendOp::ReverseSubtract: return VK_BLEND_OP_REVERSE_SUBTRACT;
            case BlendOp::Min: return VK_BLEND_OP_MIN;
            case BlendOp::Max: return VK_BLEND_OP_MAX;
            }
            return VK_BLEND_OP_ADD;
        }

        VkCompareOp ToVk(CompareOp op)
        {
            switch (op)
            {
            case CompareOp::Never: return VK_COMPARE_OP_NEVER;
            case CompareOp::Less: return VK_COMPARE_OP_LESS;
            case CompareOp::Equal: return VK_COMPARE_OP_EQUAL;
            case CompareOp::LessEqual: return VK_COMPARE_OP_LESS_OR_EQUAL;
            case CompareOp::Greater: return VK_COMPARE_OP_GREATER;
            case CompareOp::NotEqual: return VK_COMPARE_OP_NOT_EQUAL;
            case CompareOp::GreaterEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
            case CompareOp::Always: return VK_COMPARE_OP_ALWAYS;
            }
            return VK_COMPARE_OP_ALWAYS;
        }

        VkStencilOp ToVk(StencilOp op)
        {
            switch (op)
            {
            case StencilOp::Keep: return VK_STENCIL_OP_KEEP;
            case StencilOp::Zero: return VK_STENCIL_OP_ZERO;
            case StencilOp::Replace: return VK_STENCIL_OP_REPLACE;
            case StencilOp::IncrementClamp: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
            case StencilOp::DecrementClamp: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
            case StencilOp::Invert: return VK_STENCIL_OP_INVERT;
            case StencilOp::IncrementWrap: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
            case StencilOp::DecrementWrap: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
            }
            return VK_STENCIL_OP_KEEP;
        }

        VkPrimitiveTopology ToVk(Topology t)
        {
            switch (t)
            {
            case Topology::PointList: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
            case Topology::LineList: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            case Topology::LineStrip: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
            case Topology::TriangleList: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            case Topology::TriangleStrip: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
            }
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        }

        VkFilter ToVk(Filter f) { return f == Filter::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR; }

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


        class VkImageObject final : public Image
        {
        public:
            // `memory` VK_NULL_HANDLE: the image is the swapchain's, and goes with it.
            VkImageObject(Context& c, const ImageDesc& desc, VkImage image, VkDeviceMemory memory, VkImageView view, uint64_t bytes)
                : m_c(c), m_desc(desc), m_image(image), m_memory(memory), m_view(view), m_bytes(bytes) {}
            ~VkImageObject() override
            {
                vkDestroyImageView(m_c.device, m_view, nullptr);
                if (!m_memory)
                    return;
                vkDestroyImage(m_c.device, m_image, nullptr);
                if (m_pooled)
                    m_c.imageMemory->Free(m_allocation);
                else
                    vkFreeMemory(m_c.device, m_memory, nullptr);
            }
            const ImageDesc& Desc() const override { return m_desc; }
            uint64_t Bytes() const override { return m_bytes; }
            VkImage Handle() const { return m_image; }
            VkImageView View() const { return m_view; }
            // The layout the image is in for a use that would put it in `layout`: a `general` image stays in GENERAL.
            VkImageLayout Layout(VkImageLayout layout) const { return m_desc.general ? VK_IMAGE_LAYOUT_GENERAL : layout; }
            // Its memory is a range of a block (ImageMemory), freed with the image.
            void SetAllocation(const ImageMemory::Allocation& a) { m_allocation = a; m_pooled = true; }

        private:
            Context& m_c;
            ImageDesc m_desc;
            VkImage m_image;
            VkDeviceMemory m_memory;
            VkImageView m_view;
            uint64_t m_bytes;
            ImageMemory::Allocation m_allocation;
            bool m_pooled = false;
        };

        class VkImageViewObject final : public ImageView
        {
        public:
            VkImageViewObject(Context& c, VkImageView view) : m_c(c), m_view(view) {}
            ~VkImageViewObject() override { vkDestroyImageView(m_c.device, m_view, nullptr); }
            VkImageView Handle() const { return m_view; }

        private:
            Context& m_c;
            VkImageView m_view;
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
            std::vector<VkDescriptorImageInfo> images;

            Writes(const VkBindingLayout& layout, uint32_t table, const BindingEntry* entries, size_t count)
            {
                buffers.reserve(count);  // (the writes point into these)
                images.reserve(count);
                for (size_t i = 0; i < count; i++)
                {
                    const BindingSlot* slot = layout.Find(table, entries[i].slot);
                    if (!slot)
                    {
                        fprintf(stderr, "[rhi/vulkan] binding table %u has no slot %u\n", table, entries[i].slot);
                        abort();
                    }
                    VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                    w.dstBinding = slot->index;
                    w.descriptorCount = 1;
                    w.descriptorType = ToVk(slot->kind);
                    if (slot->kind == BindingKind::CombinedImageSampler || slot->kind == BindingKind::SampledImage ||
                        slot->kind == BindingKind::StorageImage)
                    {
                        const ImageBinding& b = entries[i].image;
                        const auto* image = static_cast<VkImageObject*>(b.image);
                        VkSampler sampler = b.sampler && !slot->immutableSampler ? static_cast<VkSamplerObject*>(b.sampler)->Handle() : VK_NULL_HANDLE;
                        VkImageView view = b.view ? static_cast<const VkImageViewObject*>(b.view)->Handle() : image->View();
                        images.push_back({ sampler, view,
                            image->Layout(slot->kind == BindingKind::StorageImage ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) });
                        w.pImageInfo = &images.back();
                    }
                    else
                    {
                        const BufferBinding& b = entries[i].buffer;
                        buffers.push_back({ static_cast<VkBufferObject*>(b.buffer)->Handle(), b.offset, b.size == ~0ull ? VK_WHOLE_SIZE : b.size });
                        w.pBufferInfo = &buffers.back();
                    }
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
            VkFenceObject(Context& c, bool signaled) : m_c(c)
            {
                VkFenceCreateInfo ci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
                ci.flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0;
                Check(vkCreateFence(c.device, &ci, nullptr, &m_fence), "vkCreateFence");
            }
            ~VkFenceObject() override { vkDestroyFence(m_c.device, m_fence, nullptr); }
            bool Signaled() override { return vkGetFenceStatus(m_c.device, m_fence) == VK_SUCCESS; }
            bool Wait(uint64_t timeoutNs) override
            {
                const VkResult r = vkWaitForFences(m_c.device, 1, &m_fence, VK_TRUE, timeoutNs);
                if (r != VK_TIMEOUT)
                    Check(r, "vkWaitForFences");
                return r == VK_SUCCESS;
            }
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
            bool Reset(uint32_t first, uint32_t count) override
            {
                if (!m_c.hostResetQueries)
                    return false;
                m_c.hostResetQueries(m_c.device, m_pool, first, count);
                return true;
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
                    if (image->Desc().general && images[i].before == State::General)
                        continue;  // (a general image only ever leaves Undefined)
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
                if (!bufferBarriers.empty() || !imageBarriers.empty())
                    vkCmdPipelineBarrier(m_cmd, srcStages, dstStages, 0, 0, nullptr, uint32_t(bufferBarriers.size()), bufferBarriers.data(),
                        uint32_t(imageBarriers.size()), imageBarriers.data());
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

        // ---- the device ----------------------------------------------------------


        // ---- the window (swapchain.h) -----------------------------------------------

        class VkSurfaceObject final : public Surface
        {
        public:
            VkSurfaceObject(Context& c, VkSurfaceKHR surface, std::function<Extent3D()> windowPixels)
                : m_c(c), m_surface(surface), m_windowPixels(std::move(windowPixels)) {}
            Extent3D CurrentExtent() override
            {
                VkSurfaceCapabilitiesKHR caps{};
                if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_c.physical, m_surface, &caps) != VK_SUCCESS)
                    return { 0, 0, 1 };
                if (caps.currentExtent.width != UINT32_MAX)
                    return { caps.currentExtent.width, caps.currentExtent.height, 1 };
                return m_windowPixels ? m_windowPixels() : Extent3D{ 0, 0, 1 };
            }
            VkSurfaceKHR Handle() const { return m_surface; }

        private:
            Context& m_c;
            VkSurfaceKHR m_surface;
            std::function<Extent3D()> m_windowPixels;
        };

#if defined(__APPLE__) && !TARGET_OS_IOS
        // Metal's host time is mach_absolute_time's (CLOCK_UPTIME_RAW); the
        // steady clock goes on through sleep. Steady = host + this.
        int64_t HostToSteadyNs()
        {
            return NowNs() - int64_t(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
        }
#endif

        // The swapchain, the presents in flight, and (when the caller wants
        // present times) the thread that waits for each to reach the screen.
        // The thread is the swapchain's: it is joined before the swapchain is
        // destroyed, which replaces video/swapchain_use.h's generations.
        class VkSwapchainObject final : public Swapchain
        {
        public:
            VkSwapchainObject(Context& c, VkSurfaceObject& surface, const SwapchainDesc& desc) : m_c(c), m_surface(surface), m_desc(desc) {}

            ~VkSwapchainObject() override
            {
                if (m_waiter.joinable())
                {
                    {
                        std::lock_guard lock(m_mutex);
                        m_stop = true;
                    }
                    m_cv.notify_all();
                    m_waiter.join();
                }
                {
                    std::lock_guard lock(m_c.queueMutex);
                    vkDeviceWaitIdle(m_c.device);
                }
                for (VkSemaphore s : m_acquire)
                    vkDestroySemaphore(m_c.device, s, nullptr);
                for (VkSemaphore s : m_renderDone)
                    vkDestroySemaphore(m_c.device, s, nullptr);
                m_images.clear();
                if (m_swapchain)
                    vkDestroySwapchainKHR(m_c.device, m_swapchain, nullptr);
            }

            bool Create()
            {
                VkPhysicalDevice physical = m_c.physical;
                VkSurfaceKHR surface = m_surface.Handle();
                VkSurfaceCapabilitiesKHR caps;
                vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps);
                uint32_t count = 0;
                vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, nullptr);
                std::vector<VkSurfaceFormatKHR> formats(count);
                vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, formats.data());
                // The preferred format in plain sRGB: the picture is sRGB-encoded.
                // MoltenVK lists each format in every color space it knows (asked
                // for VK_EXT_swapchain_colorspace or not), and the last
                // B8G8R8A8_UNORM entry is Display P3 *linear*: the layer showed
                // the picture's values as linear light, washed out and pale
                // (captures of the swapchain looked right: its values were).
                auto pick = [&](Format format) {
                    return std::find_if(formats.begin(), formats.end(), [&](const VkSurfaceFormatKHR& f) {
                        return (format == Format::Undefined ? FromVk(f.format) != Format::Undefined : FromVk(f.format) == format) &&
                            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
                    });
                };
                auto chosen = pick(m_desc.preferredFormat);
                if (chosen == formats.end())
                    chosen = pick(Format::Undefined);
                if (chosen == formats.end())
                    chosen = formats.begin();
                m_format = chosen->format;
                static bool formatLogged = false;
                if (!std::exchange(formatLogged, true))
                    fprintf(stderr, "[video] swapchain format %d, color space %d (%zu offered)\n", int(m_format), int(chosen->colorSpace),
                        formats.size());

                uint32_t modeCount = 0;
                vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &modeCount, nullptr);
                std::vector<VkPresentModeKHR> modes(modeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &modeCount, modes.data());
                // V-Sync off: mailbox (no tearing, newest frame) where there is
                // one, else immediate.
                m_mode = VK_PRESENT_MODE_FIFO_KHR;
                if (!m_desc.vsync)
                    for (VkPresentModeKHR wanted : { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR })
                        if (std::find(modes.begin(), modes.end(), wanted) != modes.end())
                        {
                            m_mode = wanted;
                            break;
                        }

                VkSwapchainCreateInfoKHR info{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
                info.surface = surface;
                info.minImageCount = std::max(caps.minImageCount, m_desc.minImages);
                if (caps.maxImageCount)
                    info.minImageCount = std::min(info.minImageCount, caps.maxImageCount);
                info.imageFormat = m_format;
                info.imageColorSpace = chosen->colorSpace;
                info.imageExtent = { m_desc.extent.width, m_desc.extent.height };
                info.imageArrayLayers = 1;
                info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
                m_readable = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
                info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
                info.preTransform = caps.currentTransform;
                info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
                info.presentMode = m_mode;
                info.clipped = VK_TRUE;
                if (VkResult r = vkCreateSwapchainKHR(m_c.device, &info, nullptr, &m_swapchain); r != VK_SUCCESS)
                {
                    fprintf(stderr, "[rhi/vulkan] vkCreateSwapchainKHR failed (%d)\n", int(r));
                    return false;
                }
                m_extent = m_desc.extent;

                vkGetSwapchainImagesKHR(m_c.device, m_swapchain, &count, nullptr);
                std::vector<VkImage> images(count);
                vkGetSwapchainImagesKHR(m_c.device, m_swapchain, &count, images.data());
                ImageDesc id;
                id.format = FromVk(m_format);
                id.extent = m_extent;
                id.usage = kImageRenderTarget | kImageCopyDst | (m_readable ? kImageCopySrc : 0);
                id.name = "swapchain";
                VkSemaphoreCreateInfo si{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
                for (VkImage image : images)
                {
                    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
                    vci.image = image;
                    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                    vci.format = m_format;
                    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                    VkImageView view = VK_NULL_HANDLE;
                    vkCreateImageView(m_c.device, &vci, nullptr, &view);
                    m_images.push_back(std::make_unique<VkImageObject>(m_c, id, image, VK_NULL_HANDLE, view, 0));
                    VkSemaphore done, acquire;
                    vkCreateSemaphore(m_c.device, &si, nullptr, &done);
                    vkCreateSemaphore(m_c.device, &si, nullptr, &acquire);
                    m_renderDone.push_back(done);
                    m_acquire.push_back(acquire);
                }

                {
                    std::string offered;
                    for (VkPresentModeKHR m : modes)
                        offered += std::string(offered.empty() ? "" : ", ") + (m == VK_PRESENT_MODE_IMMEDIATE_KHR ? "IMMEDIATE"
                            : m == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" : m == VK_PRESENT_MODE_FIFO_KHR ? "FIFO"
                            : m == VK_PRESENT_MODE_FIFO_RELAXED_KHR ? "FIFO_RELAXED" : std::to_string(int(m)));
                    m_describe = std::string(PresentModeName()) + ", " + std::to_string(count) + " images (the surface offers " + offered +
                        ", and " + (caps.maxImageCount ? std::to_string(caps.minImageCount) + " to " + std::to_string(caps.maxImageCount)
                                                       : std::to_string(caps.minImageCount) + " or more") + " images)";
                }
                if (m_desc.onShown && m_c.waitForPresent)
                    m_waiter = std::thread([this] { Waiter(); });
                return true;
            }

            Extent3D Extent() const override { return m_extent; }
            Format GetFormat() const override { return FromVk(m_format); }
            VkFormat VkFormatOf() const { return m_format; }
            uint32_t ImageCount() const override { return uint32_t(m_images.size()); }
            bool Readable() const override { return m_readable; }
            const char* PresentModeName() const override
            {
                return m_mode == VK_PRESENT_MODE_FIFO_KHR ? "FIFO" : m_mode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" : "IMMEDIATE";
            }
            std::string Describe() const override { return m_describe; }

            SurfaceStatus Acquire(Image*& image) override
            {
                image = nullptr;
                VkSemaphore semaphore = m_acquire[m_acquireNext];
                VkResult r = vkAcquireNextImageKHR(m_c.device, m_swapchain, UINT64_MAX, semaphore, VK_NULL_HANDLE, &m_current);
                if (r == VK_ERROR_OUT_OF_DATE_KHR)
                    return SurfaceStatus::OutOfDate;
                m_acquireNext = (m_acquireNext + 1) % m_acquire.size();
                m_waitSemaphore = semaphore;
                image = m_images[m_current].get();
                return r == VK_SUBOPTIMAL_KHR ? SurfaceStatus::Suboptimal : SurfaceStatus::Ok;
            }

            // What the submission that draws the acquired image waits for and signals.
            void SubmitSemaphores(VkSemaphore& wait, VkSemaphore& signal) const
            {
                wait = m_waitSemaphore;
                signal = m_renderDone[m_current];
            }

            SurfaceStatus Present(const PresentRequest& request) override
            {
                VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
                pi.waitSemaphoreCount = 1;
                pi.pWaitSemaphores = &m_renderDone[m_current];
                pi.swapchainCount = 1;
                pi.pSwapchains = &m_swapchain;
                pi.pImageIndices = &m_current;
                const int64_t presented = request.presented ? request.presented : NowNs();
                const bool timed = m_waiter.joinable() && request.id;
                VkPresentIdKHR presentId{ VK_STRUCTURE_TYPE_PRESENT_ID_KHR };
                uint64_t id = request.id;
                if (timed)
                {
                    presentId.swapchainCount = 1;
                    presentId.pPresentIds = &id;
                    pi.pNext = &presentId;
                }
#if defined(__APPLE__) && !TARGET_OS_IOS
                // Only a frame shown at a time carries the driver's present
                // times (and, for the diagnostics, every present): the rest are
                // presented plainly.
                VkPresentTimeGOOGLE presentTime{};
                VkPresentTimesInfoGOOGLE presentTimes{ VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE };
                if (m_c.pastTiming && request.id)
                {
                    presentTime.presentID = uint32_t(request.id);
                    if (request.notBefore)
                        presentTime.desiredPresentTime = uint64_t(std::max<int64_t>(request.notBefore - HostToSteadyNs(), 0));
                    if (presentTime.desiredPresentTime || m_desc.reportAll)
                    {
                        presentTimes.swapchainCount = 1;
                        presentTimes.pTimes = &presentTime;
                        presentTimes.pNext = pi.pNext;
                        pi.pNext = &presentTimes;
                    }
                }
#endif
                VkResult r;
                {
                    std::lock_guard lock(m_c.queueMutex);
                    r = vkQueuePresentKHR(m_c.queue, &pi);
                }
                if (timed && (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR))
                {
                    {
                        std::lock_guard lock(m_mutex);
                        m_pending.push_back({ request.id, presented });
                    }
                    m_cv.notify_one();
                }
                return r == VK_ERROR_OUT_OF_DATE_KHR ? SurfaceStatus::OutOfDate : r == VK_SUBOPTIMAL_KHR ? SurfaceStatus::Suboptimal
                                                                                                          : SurfaceStatus::Ok;
            }

        private:
            struct Pending { uint64_t id; int64_t presented; };

            // The driver's reports on the presents shown at a time, since the
            // last call (Metal's presentedTime: 0 when the drawable was never shown).
            void ReadReports()
            {
#if defined(__APPLE__) && !TARGET_OS_IOS
                if (!m_c.pastTiming || !m_desc.onReport)
                    return;
                uint32_t count = 0;
                std::vector<VkPastPresentationTimingGOOGLE> past;
                if (m_c.pastTiming(m_c.device, m_swapchain, &count, nullptr) == VK_SUCCESS && count)
                {
                    past.resize(count);
                    if (m_c.pastTiming(m_c.device, m_swapchain, &count, past.data()) < VK_SUCCESS)
                        count = 0;
                    past.resize(count);
                }
                const int64_t hostToSteady = HostToSteadyNs();
                for (const VkPastPresentationTimingGOOGLE& e : past)
                    m_desc.onReport({ e.presentID, e.actualPresentTime ? int64_t(e.actualPresentTime) + hostToSteady : 0 });
#endif
            }

            void Waiter()
            {
                if (m_desc.onThreadStart)
                    m_desc.onThreadStart();
                while (true)
                {
                    Pending t;
                    {
                        std::unique_lock lock(m_mutex);
                        m_cv.wait(lock, [&] { return m_stop || !m_pending.empty(); });
                        if (m_stop)
                            return;
                        t = m_pending.front();
                    }
                    const int64_t start = NowNs();
                    VkResult r = m_c.waitForPresent(m_c.device, m_swapchain, t.id, 50'000'000);
                    const int64_t now = NowNs();
                    {
                        std::lock_guard lock(m_mutex);
                        if (m_stop)
                            return;
                        // Not on screen yet: wait again. A hidden window may show
                        // nothing for a long time; past 8 waiting, drop the oldest.
                        if (r == VK_TIMEOUT && m_pending.size() <= 8)
                            continue;
                        if (!m_pending.empty() && m_pending.front().id == t.id)
                            m_pending.pop_front();
                    }
                    if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR)
                    {
                        ReadReports();
                        m_desc.onShown({ t.id, t.presented, now, now - start > 50'000 });
                    }
                }
            }

            Context& m_c;
            VkSurfaceObject& m_surface;
            SwapchainDesc m_desc;
            VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
            VkFormat m_format = VK_FORMAT_UNDEFINED;
            VkPresentModeKHR m_mode = VK_PRESENT_MODE_FIFO_KHR;
            Extent3D m_extent;
            bool m_readable = false;
            std::string m_describe;
            std::vector<std::unique_ptr<VkImageObject>> m_images;
            std::vector<VkSemaphore> m_acquire, m_renderDone;  // one each per image
            size_t m_acquireNext = 0;
            uint32_t m_current = 0;
            VkSemaphore m_waitSemaphore = VK_NULL_HANDLE;
            std::mutex m_mutex;
            std::condition_variable m_cv;
            std::deque<Pending> m_pending;  // presented, not yet seen on screen
            bool m_stop = false;
            std::thread m_waiter;
        };

        class VkDeviceObject final : public Device
        {
        public:
            explicit VkDeviceObject(std::unique_ptr<Context> context) : m_c(std::move(context)) {}
            ~VkDeviceObject() override
            {
                vkDeviceWaitIdle(m_c->device);
                m_c->imageMemory.reset();
                vkDestroyPipelineCache(m_c->device, m_c->pipelineCache, nullptr);
                vkDestroyDevice(m_c->device, nullptr);
                vkDestroyInstance(m_c->instance, nullptr);
            }

            const Capabilities& GetCapabilities() const override { return m_c->caps; }
            Context* Ctx() { return m_c.get(); }
            const vulkan::Native& Native() const { return m_c->native; }

            std::unique_ptr<Buffer> CreateBuffer(const BufferDesc& desc) override
            {
                if (desc.memory == MemoryKind::Stream)
                    return CreateStreamBuffer(desc);
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
                // Cached where there is some (the CPU reads it), else any coherent host-visible type.
                if (type == UINT32_MAX && desc.memory == MemoryKind::UploadCached)
                    type = m_c->FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
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


            // ---- where the big CPU-written allocations go (gpu/memory_plan.h, D14) ----
            //
            // Guest memory and the upload ring: a plan lists the places to try
            // in turn; a place the driver refuses (out of memory, say) is logged
            // and the next one tried; only when none is left does the caller
            // give up. Test hooks for the plan's fallbacks, which the machines
            // we test on never need:
            //   NFSMW_TEST_BAR_HEAP_MB=<MB>: the heaps holding CPU-mapped VRAM
            //     read as that size to the plan, and the copy's and the ring's
            //     allocations from those types fail (out of device memory) once
            //     they would pass it, as on a GPU without resizable BAR;
            //   NFSMW_TEST_ALLOC_FAIL=<places>: those places fail as out of
            //     device memory (comma-separated: import, copy-vram,
            //     copy-system, copy-cached, ring-vram, ring-cached, ring-system);
            //   NFSMW_TEST_GPU_KIND=integrated|discrete|other: the plan takes
            //     the GPU for that kind (the Deck's choice on the Steam Machine).
            struct MemoryTest
            {
                uint64_t barBytes = 0, barUsed = 0;
                std::string fail;  // ",a,b,"
                const char* kind = nullptr;
            };
            MemoryTest m_test = [] {
                MemoryTest t;
                if (const char* v = std::getenv("NFSMW_TEST_BAR_HEAP_MB"); v && std::atoi(v) > 0)
                    t.barBytes = uint64_t(std::atoi(v)) << 20;
                if (const char* v = std::getenv("NFSMW_TEST_ALLOC_FAIL"); v && v[0])
                    t.fail = std::string(",") + v + ",";
                t.kind = std::getenv("NFSMW_TEST_GPU_KIND");
                return t;
            }();

            bool MemoryTestRefuses(memplan::Place place, bool ring)
            {
                const char* name = memplan::Name(place, ring);
                if (m_test.fail.find(std::string(",") + name + ",") == std::string::npos)
                    return false;
                fprintf(stderr, "[rhi/vulkan] NFSMW_TEST_ALLOC_FAIL: %s refused (out of device memory)\n", name);
                return true;
            }

            // The first type among `bits` with `want`, else the first among
            // `bits` (the memory is the driver's to say yes or no to).
            uint32_t FindTypeLoose(uint32_t bits, VkMemoryPropertyFlags want) const
            {
                const uint32_t type = m_c->FindMemoryType(bits, want);
                if (type != UINT32_MAX)
                    return type;
                for (uint32_t i = 0; i < m_c->memory.memoryTypeCount; i++)
                    if (bits & (1u << i))
                        return i;
                return 0;
            }

            // vkAllocateMemory for a planned place, through the test hooks.
            VkResult AllocatePlanned(const memplan::Candidate& c, bool ring, VkDeviceSize size, VkDeviceMemory& memory)
            {
                if (MemoryTestRefuses(c.place, ring))
                    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                const VkMemoryPropertyFlags bar = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                const bool mappedVram = (m_c->memory.memoryTypes[c.type].propertyFlags & bar) == bar;
                if (mappedVram && m_test.barBytes && m_test.barUsed + size > m_test.barBytes)
                {
                    fprintf(stderr, "[rhi/vulkan] NFSMW_TEST_BAR_HEAP_MB: %llu MB of CPU-mapped VRAM refused (%llu of %llu MB taken)\n",
                        (unsigned long long)(size >> 20), (unsigned long long)(m_test.barUsed >> 20), (unsigned long long)(m_test.barBytes >> 20));
                    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
                }
                VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                mai.allocationSize = size;
                mai.memoryTypeIndex = c.type;
                const VkResult r = vkAllocateMemory(m_c->device, &mai, nullptr, &memory);
                if (r == VK_SUCCESS && mappedVram)
                    m_test.barUsed += size;
                return r;
            }

            // The device's memory as the plan sees it (NFSMW_TEST_BAR_HEAP_MB
            // applied), logged once: bug reports then show what a GPU offers.
            void PlanDeviceMemory()
            {
                if (m_c->planned)
                    return;
                m_c->planned = true;
                const VkPhysicalDeviceMemoryProperties& mp = m_c->memory;
                memplan::Memory& m = m_c->planMemory;
                std::string line;
                for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
                {
                    m.heaps.push_back(mp.memoryHeaps[h].size);
                    line += std::format("{}{} {}{}", h ? ", " : "", h, memplan::Size(m.heaps.back()),
                        (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " device-local" : "");
                }
                line += "; types (type:heap/flags):";
                for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
                {
                    m.types.push_back({ mp.memoryTypes[i].propertyFlags, mp.memoryTypes[i].heapIndex });
                    line += std::format(" {}:{}/{:X}", i, m.types.back().heap, m.types.back().flags);
                }
                if (m_test.barBytes)
                {
                    memplan::ShrinkMappedVram(m, m_test.barBytes);
                    line += std::format(" (NFSMW_TEST_BAR_HEAP_MB: CPU-mapped VRAM's heaps read as {} MB)", m_test.barBytes >> 20);
                }
                fprintf(stderr, "[rhi/vulkan] memory heaps: %s\n", line.c_str());
            }

            uint32_t DriverId() const
            {
                if (m_c->props.apiVersion < VK_API_VERSION_1_2)
                    return 0;
                VkPhysicalDeviceDriverProperties driver{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
                VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver };
                vkGetPhysicalDeviceProperties2(m_c->physical, &props2);
                return uint32_t(driver.driverID);
            }

            // Guest physical memory imported without a copy (unified memory, or
            // a discrete GPU reading system memory over PCIe).
            std::unique_ptr<Buffer> ImportGuest(void* host, uint64_t size)
            {
                if (!m_c->hostPointerProperties)  // (no host-memory import)
                    return nullptr;
                if (MemoryTestRefuses(memplan::Place::kImport, false))
                {
                    fprintf(stderr, "[rhi/vulkan] importing guest memory failed (%d)\n", int(VK_ERROR_OUT_OF_DEVICE_MEMORY));
                    return nullptr;
                }
                VkMemoryHostPointerPropertiesEXT hpp{ VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
                if (VkResult r = m_c->hostPointerProperties(m_c->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &hpp);
                    r != VK_SUCCESS)
                {
                    // (A driver can refuse the pointer itself: on Windows guest
                    // memory is a view of a file mapping, which only Intel's
                    // driver is known to import. Logged, so a bug report tells
                    // this from a refused allocation.)
                    fprintf(stderr, "[rhi/vulkan] importing guest memory failed: the driver doesn't take its host pointer (%d)\n", int(r));
                    return nullptr;
                }
                VkExternalMemoryBufferCreateInfo embci{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
                embci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
                VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.pNext = &embci;
                bci.size = size;
                bci.usage = ToVk(BufferUsage(kBufferStorage | kBufferCopySrc | kBufferCopyDst));
                VkBuffer buffer = VK_NULL_HANDLE;
                Check(vkCreateBuffer(m_c->device, &bci, nullptr, &buffer), "vkCreateBuffer(shared)");
                VkMemoryRequirements req;
                vkGetBufferMemoryRequirements(m_c->device, buffer, &req);
                VkImportMemoryHostPointerInfoEXT import{ VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT };
                import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
                import.pHostPointer = host;
                VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                mai.pNext = &import;
                mai.allocationSize = size;
                mai.memoryTypeIndex = FindTypeLoose(req.memoryTypeBits & hpp.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                VkDeviceMemory memory = VK_NULL_HANDLE;
                VkResult r = vkAllocateMemory(m_c->device, &mai, nullptr, &memory);
                if (r == VK_SUCCESS && (r = vkBindBufferMemory(m_c->device, buffer, memory, 0)) != VK_SUCCESS)
                {
                    vkFreeMemory(m_c->device, memory, nullptr);
                    memory = VK_NULL_HANDLE;
                }
                if (r != VK_SUCCESS)
                {
                    fprintf(stderr, "[rhi/vulkan] importing guest memory failed (%d)\n", int(r));
                    vkDestroyBuffer(m_c->device, buffer, nullptr);
                    return nullptr;
                }
                auto result = std::make_unique<VkBufferObject>(*m_c, buffer, memory, size, host, true);
                result->MarkImported();
                return result;
            }

            // The buffer of guest memory's copy (an unbound one: a place that
            // fails before binding leaves it for the next).
            VkBuffer NewCopyBuffer(uint64_t size, VkMemoryRequirements& req)
            {
                VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.size = size;
                bci.usage = ToVk(BufferUsage(kBufferStorage | kBufferCopySrc | kBufferCopyDst));
                VkBuffer buffer = VK_NULL_HANDLE;
                Check(vkCreateBuffer(m_c->device, &bci, nullptr, &buffer), "vkCreateBuffer(shadow)");
                vkGetBufferMemoryRequirements(m_c->device, buffer, &req);
                return buffer;
            }

            // Guest memory's copy for the GPU in `c`'s memory type: on a
            // discrete GPU in VRAM the CPU writes through resizable BAR when
            // it holds the copy, else in system memory; on an integrated one
            // in system memory (memory_plan.h). False, with nothing kept, when
            // the driver refuses it; `buffer` is then still unbound, or replaced.
            std::unique_ptr<Buffer> CreateShadowGuest(uint64_t size, VkBuffer& buffer, const VkMemoryRequirements& req,
                const memplan::Candidate& c, void*& mapped)
            {
                VkDeviceMemory memory = VK_NULL_HANDLE;
                void* p = nullptr;
                const char* what = "vkAllocateMemory(shadow)";
                VkResult r = AllocatePlanned(c, false, req.size, memory);
                if (r == VK_SUCCESS && (r = vkMapMemory(m_c->device, memory, 0, VK_WHOLE_SIZE, 0, &p)) != VK_SUCCESS)
                    what = "vkMapMemory(shadow)";
                else if (r == VK_SUCCESS && (r = vkBindBufferMemory(m_c->device, buffer, memory, 0)) != VK_SUCCESS)
                {
                    what = "vkBindBufferMemory(shadow)";
                    vkDestroyBuffer(m_c->device, buffer, nullptr);  // (a buffer binds once)
                    VkMemoryRequirements again;
                    buffer = NewCopyBuffer(size, again);
                }
                if (r != VK_SUCCESS)
                {
                    fprintf(stderr, "[rhi/vulkan] %s failed: %d (%s, memory type %u): trying the next place\n", what, int(r),
                        memplan::Name(c.place, false), c.type);
                    if (memory)
                        vkFreeMemory(m_c->device, memory, nullptr);
                    return nullptr;
                }
                mapped = p;
                const bool coherent = (m_c->memory.memoryTypes[c.type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
                return std::make_unique<VkBufferObject>(*m_c, buffer, memory, size, p, coherent);
            }

            // `failed`, a list of place names, as the log's ", after ... failed".
            static std::string AfterFailed(const std::string& failed)
            {
                return failed.empty() ? std::string() : " after " + failed + " failed";
            }

            bool CreateGuestMemory(void* host, uint64_t size, GuestMemory& out) override
            {
                const VkPhysicalDeviceMemoryProperties& mp = m_c->memory;
                PlanDeviceMemory();
                memplan::GuestInput in;
                in.gpu = m_c->caps.kind == Capabilities::Kind::Discrete ? memplan::Gpu::kDiscrete
                    : m_c->caps.kind == Capabilities::Kind::Integrated ? memplan::Gpu::kIntegrated : memplan::Gpu::kOther;
                if (const char* kind = m_test.kind)
                {
                    std::string_view k(kind);
                    in.gpu = k == "integrated" ? memplan::Gpu::kIntegrated : k == "discrete" ? memplan::Gpu::kDiscrete
                        : k == "other" ? memplan::Gpu::kOther : in.gpu;
                    fprintf(stderr, "[rhi/vulkan] NFSMW_TEST_GPU_KIND=%s: memory planned as for %s GPU\n", kind,
                        in.gpu == memplan::Gpu::kDiscrete ? "a discrete" : in.gpu == memplan::Gpu::kIntegrated ? "an integrated" : "another");
                }
                in.importable = m_c->hostPointerProperties != nullptr;
                in.importRefusal = memplan::ImportRefusal(DriverId());
                in.overrides = memplan::ParseOverrides(std::getenv("NFSMW_SHARED_MEMORY"), std::getenv("NFSMW_SHADOW_MEMORY"),
                    std::getenv("NFSMW_RING_MEMORY"));
                VkMemoryRequirements req;
                VkBuffer copy = NewCopyBuffer(size, req);
                in.bits = req.memoryTypeBits;
                m_c->planGpu = in.gpu;
                memplan::Plan plan = memplan::PlanGuestMemory(m_c->planMemory, in);
                std::string failed;
                const memplan::Candidate* placed = nullptr;
                for (const memplan::Candidate& c : plan.order)
                {
                    void* mapped = nullptr;
                    if (c.place == memplan::Place::kImport)
                        out.buffer = ImportGuest(host, size);
                    else
                        out.buffer = CreateShadowGuest(size, copy, req, c, mapped);
                    if (out.buffer)
                    {
                        placed = &c;
                        out.shadow = static_cast<uint8_t*>(mapped);
                        break;
                    }
                    failed += std::string(failed.empty() ? "" : ", ") + memplan::Name(c.place, false);
                }
                if (!placed)
                {
                    fprintf(stderr, "[rhi/vulkan] guest memory: no place could hold it (%s failed; %s)\n", failed.c_str(), plan.why.c_str());
                    vkDestroyBuffer(m_c->device, copy, nullptr);
                    return false;
                }
                if (placed->place == memplan::Place::kImport)
                {
                    vkDestroyBuffer(m_c->device, copy, nullptr);
                    fprintf(stderr, "[rhi/vulkan] guest memory: imported (zero-copy)\n");
                    m_c->guestChoice = "imported (zero-copy)";
                }
                else
                {
                    m_c->guestType = placed->type;
                    m_c->shadowInVram = (mp.memoryTypes[placed->type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
                    out.shadowInVram = m_c->shadowInVram;
                    fprintf(stderr, "[rhi/vulkan] guest memory: GPU copy in %s (memory type %u, flags %X), updated from the write watch\n",
                        m_c->shadowInVram ? "VRAM (CPU-mapped)" : "host memory", placed->type, mp.memoryTypes[placed->type].propertyFlags);
                    m_c->guestChoice = std::format("as a GPU copy in {}, memory type {}", placed->place == memplan::Place::kVram ? "CPU-mapped VRAM"
                        : placed->place == memplan::Place::kCached ? "cached host memory" : "host memory", placed->type);
                }
                m_c->guestChoice += " (" + plan.why + ")" + AfterFailed(failed);
                return true;
            }

            // The upload ring: device-local when the host can map it
            // (resizable BAR) and it fits, so the GPU reads constants and
            // indices from VRAM; else cached system memory (memory_plan.h).
            // NFSMW_RING_MEMORY=cached prefers cached system memory anyway: the
            // CPU's ~40 MB of writes a frame then aren't write-combined PCIe
            // stores (an A/B experiment: on the Steam Machine 2% less CPU,
            // twice the GPU time); =vram|system name the other places.
            std::unique_ptr<Buffer> CreateStreamBuffer(const BufferDesc& desc)
            {
                PlanDeviceMemory();
                VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.size = desc.size;
                bci.usage = ToVk(BufferUsage(desc.usage));
                VkBuffer buffer = VK_NULL_HANDLE;
                Check(vkCreateBuffer(m_c->device, &bci, nullptr, &buffer), "vkCreateBuffer(ring)");
                VkMemoryRequirements req;
                vkGetBufferMemoryRequirements(m_c->device, buffer, &req);
                const memplan::Overrides o = memplan::ParseOverrides(nullptr, nullptr, std::getenv("NFSMW_RING_MEMORY"));
                memplan::Plan plan = memplan::PlanRing(m_c->planMemory, req.memoryTypeBits, m_c->guestType, o.ring);
                std::string failed;
                for (const memplan::Candidate& c : plan.order)
                {
                    VkDeviceMemory memory = VK_NULL_HANDLE;
                    void* p = nullptr;
                    const char* what = "vkAllocateMemory(ring)";
                    VkResult r = AllocatePlanned(c, true, req.size, memory);
                    if (r == VK_SUCCESS && (r = vkMapMemory(m_c->device, memory, 0, VK_WHOLE_SIZE, 0, &p)) != VK_SUCCESS)
                        what = "vkMapMemory(ring)";
                    else if (r == VK_SUCCESS && (r = vkBindBufferMemory(m_c->device, buffer, memory, 0)) != VK_SUCCESS)
                    {
                        what = "vkBindBufferMemory(ring)";
                        vkDestroyBuffer(m_c->device, buffer, nullptr);  // (a buffer binds once)
                        Check(vkCreateBuffer(m_c->device, &bci, nullptr, &buffer), "vkCreateBuffer(ring)");
                    }
                    if (r != VK_SUCCESS)
                    {
                        fprintf(stderr, "[rhi/vulkan] %s failed: %d (%s, memory type %u): trying the next place\n", what, int(r),
                            memplan::Name(c.place, true), c.type);
                        if (memory)
                            vkFreeMemory(m_c->device, memory, nullptr);
                        failed += std::string(failed.empty() ? "" : ", ") + memplan::Name(c.place, true);
                        continue;
                    }
                    const VkMemoryPropertyFlags flags = m_c->memory.memoryTypes[c.type].propertyFlags;
                    fprintf(stderr, "[rhi/vulkan] upload ring: memory type %u (flags %X)\n", c.type, flags);
                    // The choice and why, in one line.
                    fprintf(stderr, "[rhi/vulkan] memory: guest memory %s; upload ring in %s, memory type %u (%s)%s\n", m_c->guestChoice.c_str(),
                        c.place == memplan::Place::kVram ? "CPU-mapped VRAM" : c.place == memplan::Place::kCached ? "cached host memory"
                            : "host memory", c.type, plan.why.c_str(), AfterFailed(failed).c_str());
                    // Issue #40: say so when the GPU reads guest memory across PCIe.
                    if (const char* note = memplan::OverPcieNote(m_c->planGpu, m_c->guestType != memplan::kNoType && m_c->shadowInVram,
                            (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0))
                        fprintf(stderr, "[rhi/vulkan] note: %s\n", note);
                    return std::make_unique<VkBufferObject>(*m_c, buffer, memory, desc.size, p,
                        (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0);
                }
                fprintf(stderr, "[rhi/vulkan] upload ring: no place could hold it (%s failed; %s)\n", failed.c_str(), plan.why.c_str());
                vkDestroyBuffer(m_c->device, buffer, nullptr);
                return nullptr;
            }

            std::unique_ptr<Image> CreateImage(const ImageDesc& desc) override
            {
                const auto start = std::chrono::steady_clock::now();
                const VkFormat format = ToVk(desc.format);
                const bool cube = desc.kind == ImageKind::Cube;
                const bool store = desc.storeFormat != Format::Undefined;
                // A store format: views in that format of an image whose own
                // isn't storable, with the usage the view says (the image is
                // sampled through views of its own format).
                const VkFormat formats[2] = { format, store ? ToVk(desc.storeFormat) : format };
                VkImageFormatListCreateInfo list{ VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, nullptr, 2, formats };
                VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
                ici.pNext = store ? &list : nullptr;
                ici.flags = (cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0) |
                    (store ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0);
                ici.imageType = desc.kind == ImageKind::Image3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
                ici.format = format;
                ici.extent = { desc.extent.width, desc.extent.height, desc.extent.depth };
                ici.mipLevels = desc.levels;
                ici.arrayLayers = cube ? 6 : desc.layers;
                ici.samples = VK_SAMPLE_COUNT_1_BIT;
                ici.tiling = VK_IMAGE_TILING_OPTIMAL;  // (image memory blocks hold nothing linear)
                ici.usage = ToVk(ImageUsage(desc.usage));
                ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VkImage image;
                if (vkCreateImage(m_c->device, &ici, nullptr, &image) != VK_SUCCESS)
                    return nullptr;
                ImageMemory::Allocation memory;
                if (!m_c->imageMemory->Bind(image, memory))
                {
                    fprintf(stderr, "[rhi/vulkan] no memory for an image (%ux%u, format %d)\n", desc.extent.width, desc.extent.height, int(format));
                    vkDestroyImage(m_c->device, image, nullptr);
                    return nullptr;
                }
                VkImageViewUsageCreateInfo sampled{ VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, nullptr, VK_IMAGE_USAGE_SAMPLED_BIT };
                VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
                vci.pNext = store ? &sampled : nullptr;
                vci.image = image;
                vci.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : desc.kind == ImageKind::Image3D ? VK_IMAGE_VIEW_TYPE_3D
                    : desc.layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
                vci.format = format;
                vci.subresourceRange = { Aspect(desc.format), 0, desc.levels, 0, cube ? 6 : desc.layers };
                VkImageView view;
                if (vkCreateImageView(m_c->device, &vci, nullptr, &view) != VK_SUCCESS)
                {
                    vkDestroyImage(m_c->device, image, nullptr);
                    m_c->imageMemory->Free(memory);
                    return nullptr;
                }
                auto result = std::make_unique<VkImageObject>(*m_c, desc, image, memory.memory, view, memory.bytes);
                result->SetAllocation(memory);
                m_c->imageMemory->Made(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()));
                return result;
            }

            std::unique_ptr<ImageView> CreateImageView(const Image& image, const ImageViewDesc& desc) override
            {
                const auto& img = static_cast<const VkImageObject&>(image);
                const ImageDesc& d = img.Desc();
                auto component = [](Swizzle s) {
                    switch (s)
                    {
                    case Swizzle::R: return VK_COMPONENT_SWIZZLE_R;
                    case Swizzle::G: return VK_COMPONENT_SWIZZLE_G;
                    case Swizzle::B: return VK_COMPONENT_SWIZZLE_B;
                    case Swizzle::A: return VK_COMPONENT_SWIZZLE_A;
                    case Swizzle::Zero: return VK_COMPONENT_SWIZZLE_ZERO;
                    case Swizzle::One: return VK_COMPONENT_SWIZZLE_ONE;
                    default: return VK_COMPONENT_SWIZZLE_IDENTITY;
                    }
                };
                const Format format = desc.format == Format::Undefined ? d.format : desc.format;
                const bool cube = d.kind == ImageKind::Cube;
                VkImageViewUsageCreateInfo usage{ VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, nullptr, 0 };
                VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
                if (format != d.format)  // (a reinterpretation: the view's own usage is storage)
                {
                    usage.usage = VK_IMAGE_USAGE_STORAGE_BIT;
                    vci.pNext = &usage;
                }
                vci.image = img.Handle();
                vci.format = ToVk(format);
                if (desc.layerCount)
                {
                    vci.viewType = desc.layerCount > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
                    vci.subresourceRange = { Aspect(format), 0, 1, desc.layer, desc.layerCount };
                }
                else
                {
                    vci.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : d.kind == ImageKind::Image3D ? VK_IMAGE_VIEW_TYPE_3D
                        : d.layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
                    vci.subresourceRange = { Aspect(format), 0, d.levels, 0, cube ? 6 : d.layers };
                }
                if (desc.depthOnly)
                    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                vci.components = { component(desc.swizzle[0]), component(desc.swizzle[1]), component(desc.swizzle[2]), component(desc.swizzle[3]) };
                VkImageView view;
                if (vkCreateImageView(m_c->device, &vci, nullptr, &view) != VK_SUCCESS)
                    return nullptr;
                return std::make_unique<VkImageViewObject>(*m_c, view);
            }

            bool SupportsImage(const ImageDesc& desc) const override
            {
                const bool cube = desc.kind == ImageKind::Cube;
                const bool store = desc.storeFormat != Format::Undefined;
                // (The image's format list is Vulkan 1.2.)
                if (store && m_c->props.apiVersion < VK_API_VERSION_1_2)
                    return false;
                if (store)
                {
                    VkFormatProperties fp;
                    vkGetPhysicalDeviceFormatProperties(m_c->physical, ToVk(desc.storeFormat), &fp);
                    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
                        return false;
                }
                VkImageFormatProperties props;
                if (vkGetPhysicalDeviceImageFormatProperties(m_c->physical, ToVk(desc.format), VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                        ToVk(ImageUsage(desc.usage)),
                        (cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0) |
                            (store ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0),
                        &props) != VK_SUCCESS)
                    return false;
                return !cube || props.maxArrayLayers >= 6;
            }

            uint32_t FormatFeatures(Format format) const override
            {
                VkFormatProperties fp;
                vkGetPhysicalDeviceFormatProperties(m_c->physical, ToVk(format), &fp);
                return ((fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) ? kFormatBlend : 0) |
                    ((fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ? kFormatStorage : 0);
            }

            void GetMemoryUsage(uint64_t& used, uint64_t& budget) const override
            {
                used = budget = 0;
                if (!m_c->memoryBudget)
                    return;
                VkPhysicalDeviceMemoryBudgetPropertiesEXT info{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
                VkPhysicalDeviceMemoryProperties2 props{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &info };
                vkGetPhysicalDeviceMemoryProperties2(m_c->physical, &props);
                for (uint32_t h = 0; h < props.memoryProperties.memoryHeapCount; h++)
                    if (props.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    {
                        used += info.heapUsage[h];
                        budget += info.heapBudget[h];
                    }
            }

            ImageMemoryStats GetImageMemoryStats() const override { return m_c->imageMemory->Stats(); }

            std::unique_ptr<Sampler> CreateSampler(const SamplerDesc& desc) override
            {
                auto filter = [](Filter f) { return f == Filter::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR; };
                auto address = [](AddressMode a) {
                    return a == AddressMode::ClampToEdge ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                        : a == AddressMode::ClampToBorder ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
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
                ci.minLod = desc.minLod;
                ci.maxLod = desc.maxLod >= 1000.0f ? VK_LOD_CLAMP_NONE : desc.maxLod;
                ci.borderColor = desc.whiteBorder ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
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

            std::unique_ptr<Pipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc& desc) override
            {
                VkShaderModule modules[2] = {};
                const ShaderCode* codes[2] = { &desc.vertex, &desc.fragment };
                VkPipelineShaderStageCreateInfo stages[2]{};
                for (int i = 0; i < 2; i++)
                {
                    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
                    smci.codeSize = codes[i]->words * 4;
                    smci.pCode = codes[i]->spirv;
                    if (vkCreateShaderModule(m_c->device, &smci, nullptr, &modules[i]) != VK_SUCCESS)
                    {
                        for (VkShaderModule m : modules)
                            if (m)
                                vkDestroyShaderModule(m_c->device, m, nullptr);
                        return nullptr;
                    }
                    stages[i] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                        i == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT, modules[i], "main", nullptr };
                }
                VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };  // (vertices come from buffers in the shader)
                VkPipelineInputAssemblyStateCreateInfo assembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
                assembly.topology = ToVk(desc.topology);
                assembly.primitiveRestartEnable = desc.primitiveRestart ? VK_TRUE : VK_FALSE;
                VkPipelineViewportStateCreateInfo viewport{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
                viewport.viewportCount = viewport.scissorCount = 1;
                VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
                raster.depthClampEnable = desc.depthClamp ? VK_TRUE : VK_FALSE;
                raster.polygonMode = VK_POLYGON_MODE_FILL;
                raster.cullMode = desc.cull == CullMode::Front ? VK_CULL_MODE_FRONT_BIT : desc.cull == CullMode::Back ? VK_CULL_MODE_BACK_BIT
                                                                                                                       : VK_CULL_MODE_NONE;
                raster.frontFace = desc.frontCounterClockwise ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
                raster.depthBiasEnable = desc.depthBias ? VK_TRUE : VK_FALSE;
                raster.lineWidth = 1.0f;
                VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
                multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
                VkPipelineDepthStencilStateCreateInfo depth{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
                depth.depthTestEnable = desc.depthTest ? VK_TRUE : VK_FALSE;
                depth.depthWriteEnable = desc.depthWrite ? VK_TRUE : VK_FALSE;
                depth.depthCompareOp = ToVk(desc.depthCompare);
                depth.stencilTestEnable = desc.stencilTest ? VK_TRUE : VK_FALSE;
                auto face = [](const StencilFace& f) {
                    VkStencilOpState s{};
                    s.failOp = ToVk(f.fail);
                    s.passOp = ToVk(f.pass);
                    s.depthFailOp = ToVk(f.depthFail);
                    s.compareOp = ToVk(f.compare);
                    return s;
                };
                depth.front = face(desc.stencilFront);
                depth.back = face(desc.stencilBack);
                VkPipelineColorBlendAttachmentState attachments[4]{};
                VkFormat colorFormats[4] = {};
                constexpr uint32_t colorCount = 4;  // (see BeginRendering; a gap has no format and writes nothing)
                for (uint32_t i = 0; i < 4; i++)
                {
                    colorFormats[i] = ToVk(desc.colorFormats[i]);
                    if (desc.colorFormats[i] == Format::Undefined)
                        continue;
                    const BlendAttachment& b = desc.blend[i];
                    attachments[i].blendEnable = b.enable ? VK_TRUE : VK_FALSE;
                    attachments[i].srcColorBlendFactor = ToVk(b.srcColor);
                    attachments[i].dstColorBlendFactor = ToVk(b.dstColor);
                    attachments[i].colorBlendOp = ToVk(b.colorOp);
                    attachments[i].srcAlphaBlendFactor = ToVk(b.srcAlpha);
                    attachments[i].dstAlphaBlendFactor = ToVk(b.dstAlpha);
                    attachments[i].alphaBlendOp = ToVk(b.alphaOp);
                    attachments[i].colorWriteMask = b.writeMask;  // (R, G, B, A are the low four bits in both)
                }
                VkPipelineColorBlendStateCreateInfo blend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
                blend.attachmentCount = colorCount;
                blend.pAttachments = attachments;
                const VkDynamicState dynamic[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_DEPTH_BIAS,
                    VK_DYNAMIC_STATE_BLEND_CONSTANTS };
                VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
                dynamicState.dynamicStateCount = uint32_t(sizeof(dynamic) / sizeof(dynamic[0]));
                dynamicState.pDynamicStates = dynamic;
                VkPipelineRenderingCreateInfo rendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
                rendering.colorAttachmentCount = colorCount;
                rendering.pColorAttachmentFormats = colorFormats;
                if (desc.depthFormat != Format::Undefined)
                {
                    rendering.depthAttachmentFormat = ToVk(desc.depthFormat);
                    rendering.stencilAttachmentFormat = HasDepth(desc.depthFormat) ? ToVk(desc.depthFormat) : VK_FORMAT_UNDEFINED;
                }
                // An experiment (NFSMW_SHADING_RATE): every draw shades once per w x h pixels.
                VkPipelineFragmentShadingRateStateCreateInfoKHR rate{ VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR };
                if (desc.shadingRateW)
                {
                    rate.fragmentSize = { desc.shadingRateW, desc.shadingRateH };
                    rate.combinerOps[0] = rate.combinerOps[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
                    rendering.pNext = &rate;
                }
                VkGraphicsPipelineCreateInfo gpci{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
                gpci.pNext = &rendering;
                gpci.stageCount = 2;
                gpci.pStages = stages;
                gpci.pVertexInputState = &vertexInput;
                gpci.pInputAssemblyState = &assembly;
                gpci.pViewportState = &viewport;
                gpci.pRasterizationState = &raster;
                gpci.pMultisampleState = &multisample;
                gpci.pDepthStencilState = &depth;
                gpci.pColorBlendState = &blend;
                gpci.pDynamicState = &dynamicState;
                gpci.layout = static_cast<const VkBindingLayout*>(desc.layout)->Handle();
                VkPipeline pipeline = VK_NULL_HANDLE;
                VkResult r = vkCreateGraphicsPipelines(m_c->device, m_c->pipelineCache, 1, &gpci, nullptr, &pipeline);
                for (VkShaderModule m : modules)
                    vkDestroyShaderModule(m_c->device, m, nullptr);
                if (r != VK_SUCCESS)
                    return nullptr;
                return std::make_unique<VkPipelineObject>(*m_c, pipeline, VK_PIPELINE_BIND_POINT_GRAPHICS);
            }

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
                // A cache another GPU, driver or OS version made is set aside
                // by the driver without a word (MoltenVK's device ID carries
                // the macOS version, so every macOS update does it), and the
                // first drive afterwards stutters while the pipelines compile
                // again. The header says so in advance
                // (gpu/pipeline_cache_header.h): one line for the log, and the
                // stale data isn't handed over. The next save writes this
                // device's header, so it's said once.
                gpu::pipeline_cache::Maker device{ m_c->props.vendorID, m_c->props.deviceID, {} };
                std::memcpy(device.uuid.data(), m_c->props.pipelineCacheUUID, device.uuid.size());
                const auto check = gpu::pipeline_cache::CheckHeader(static_cast<const uint8_t*>(data), size, device);
                if (const std::string line = gpu::pipeline_cache::LogLine(check, device); !line.empty())
                {
                    fprintf(stderr, "%s\n", line.c_str());
                    data = nullptr;
                    size = 0;
                }
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
            std::unique_ptr<Fence> CreateFence(bool signaled) override { return std::make_unique<VkFenceObject>(*m_c, signaled); }
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
                std::unique_lock lock(m_c->queueMutex);
                if (m_c->submitGate)
                    m_c->submitGate(lock);
                Check(vkQueueSubmit(m_c->queue, 1, &si, static_cast<VkFenceObject&>(fence).Handle()), "vkQueueSubmit");
            }
            void SetSubmitGate(std::function<void(std::unique_lock<std::mutex>&)> gate) override { m_c->submitGate = std::move(gate); }
            std::unique_ptr<Swapchain> CreateSwapchain(Surface& surface, const SwapchainDesc& desc) override
            {
                auto swapchain = std::make_unique<VkSwapchainObject>(*m_c, static_cast<VkSurfaceObject&>(surface), desc);
                return swapchain->Create() ? std::move(swapchain) : nullptr;
            }
            void SubmitToSwapchain(CommandList& list, Fence& fence, Swapchain& swapchain) override
            {
                VkCommandBuffer cmd = static_cast<VkCommandList&>(list).Handle();
                VkSemaphore wait, signal;
                static_cast<VkSwapchainObject&>(swapchain).SubmitSemaphores(wait, signal);
                const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                si.waitSemaphoreCount = 1;
                si.pWaitSemaphores = &wait;
                si.pWaitDstStageMask = &waitStage;
                si.commandBufferCount = 1;
                si.pCommandBuffers = &cmd;
                si.signalSemaphoreCount = 1;
                si.pSignalSemaphores = &signal;
                std::unique_lock lock(m_c->queueMutex);
                if (m_c->submitGate)
                    m_c->submitGate(lock);
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
        // Dynamic rendering, and what MoltenVK needs it to bring along.
        for (const char* name : { VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
                 VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, VK_KHR_MULTIVIEW_EXTENSION_NAME, VK_KHR_MAINTENANCE_2_EXTENSION_NAME })
            if (HasExtension(avail, name))
                devExts.push_back(name);
        VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamicRendering{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR };
        dynamicRendering.dynamicRendering = VK_TRUE;

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
        config.next = &dynamicRendering;
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
        bool hostQueryReset = false;
        for (auto* in = static_cast<const VkBaseInStructure*>(config.next); in; in = in->pNext)
            if (in->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES)
                hostQueryReset |= reinterpret_cast<const VkPhysicalDeviceHostQueryResetFeatures*>(in)->hostQueryReset == VK_TRUE;

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
        if (hostQueryReset)
            c->hostResetQueries = reinterpret_cast<PFN_vkResetQueryPool>(vkGetDeviceProcAddr(c->device, "vkResetQueryPool"));
        c->native = { c->instance, c->physical, c->device, c->family, c->queue, &c->queueMutex };
        c->pushDescriptor = pushDescriptor ? reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
            vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR")) : nullptr;
        c->hostPointerProperties = hostImport ? reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
            vkGetDeviceProcAddr(c->device, "vkGetMemoryHostPointerPropertiesEXT")) : nullptr;
        c->beginRendering = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(vkGetDeviceProcAddr(c->device, "vkCmdBeginRenderingKHR"));
        c->endRendering = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(vkGetDeviceProcAddr(c->device, "vkCmdEndRenderingKHR"));
        // Present timing: the caller enabled these (policy: the settings and
        // the platform); the interface reports what it was given.
        if (has(VK_KHR_PRESENT_ID_EXTENSION_NAME) && has(VK_KHR_PRESENT_WAIT_EXTENSION_NAME))
            c->waitForPresent = reinterpret_cast<PFN_vkWaitForPresentKHR>(vkGetDeviceProcAddr(c->device, "vkWaitForPresentKHR"));
        if (has(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME))
            c->pastTiming = reinterpret_cast<PFN_vkGetPastPresentationTimingGOOGLE>(
                vkGetDeviceProcAddr(c->device, "vkGetPastPresentationTimingGOOGLE"));
        VkPipelineCacheCreateInfo pcci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
        Check(vkCreatePipelineCache(c->device, &pcci, nullptr, &c->pipelineCache), "vkCreatePipelineCache");

        // What the device offers.
        c->deviceName = props.deviceName;
        c->props = props;
        {
            uint32_t extensionCount = 0;
            vkEnumerateDeviceExtensionProperties(c->physical, nullptr, &extensionCount, nullptr);
            std::vector<VkExtensionProperties> extensions(extensionCount);
            vkEnumerateDeviceExtensionProperties(c->physical, nullptr, &extensionCount, extensions.data());
            for (const VkExtensionProperties& e : extensions)
                c->memoryBudget |= strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
        }
        Context* raw = c.get();
        c->imageMemory = std::make_unique<ImageMemory>(
            c->device, [raw](uint32_t bits) { return raw->FindMemoryType(bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); }, config.onThread);
        Capabilities& caps = c->caps;
        caps.kind = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? Capabilities::Kind::Discrete
            : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? Capabilities::Kind::Integrated : Capabilities::Kind::Other;
        caps.vendorId = props.vendorID;
        caps.maxStorageBufferRange = props.limits.maxStorageBufferRange;
        caps.maxAnisotropy = supported.samplerAnisotropy ? props.limits.maxSamplerAnisotropy : 1.0f;
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
        caps.timestampPeriodNs = props.limits.timestampPeriod;
        caps.presentTiming = c->waitForPresent != nullptr;
#if defined(__APPLE__) && !TARGET_OS_IOS
        caps.scheduledPresent = c->pastTiming != nullptr;
#endif
        return std::make_unique<VkDeviceObject>(std::move(c));
    }

    std::unique_ptr<Surface> CreateSurface(Device& device, VkSurfaceKHR surface, std::function<Extent3D()> windowPixels)
    {
        return std::make_unique<VkSurfaceObject>(*static_cast<VkDeviceObject&>(device).Ctx(), surface, std::move(windowPixels));
    }

    VkImage NativeImage(const Image& image) { return static_cast<const VkImageObject&>(image).Handle(); }
    VkImageView NativeView(const Image& image) { return static_cast<const VkImageObject&>(image).View(); }
    VkImageView NativeImageView(const ImageView& view) { return static_cast<const VkImageViewObject&>(view).Handle(); }
    VkBuffer NativeBuffer(const Buffer& buffer) { return static_cast<const VkBufferObject&>(buffer).Handle(); }
    VkFormat NativeFormat(const Swapchain& swapchain) { return static_cast<const VkSwapchainObject&>(swapchain).VkFormatOf(); }
    std::unique_ptr<Image> WrapImage(Device& device, VkImage image, Format format, uint32_t width, uint32_t height, uint32_t usage)
    {
        Context& c = *static_cast<VkDeviceObject&>(device).Ctx();
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = ToVk(format);
        vci.subresourceRange = { Aspect(format), 0, 1, 0, 1 };
        VkImageView view = VK_NULL_HANDLE;
        if (vkCreateImageView(c.device, &vci, nullptr, &view) != VK_SUCCESS)
            return nullptr;
        ImageDesc desc;
        desc.format = format;
        desc.extent = { width, height, 1 };
        desc.usage = usage;
        desc.name = "wrapped";
        return std::make_unique<VkImageObject>(c, desc, image, VK_NULL_HANDLE, view, 0);
    }

    VkCommandBuffer NativeCommandBuffer(const CommandList& list) { return static_cast<const VkCommandList&>(list).Handle(); }

    const Native& GetNative(const rhi::Device& device)
    {
        return static_cast<const VkDeviceObject&>(device).Native();
    }
}
