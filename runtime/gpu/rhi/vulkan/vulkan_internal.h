// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// The Vulkan backend's private types, shared by its source files
// (vulkan_device.cpp, vulkan_commands.cpp, vulkan_swapchain.cpp). Nothing
// outside gpu/rhi/vulkan includes this.
#pragma once
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
#include "../swapchain.h"
#include "vulkan_image_memory.h"
#include "vulkan_native.h"

namespace gpu::rhi::vulkan
{
    namespace impl
    {
        inline void Check(VkResult r, const char* what)
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

        inline VkBufferUsageFlags ToVk(BufferUsage usage)
        {
            return VkBufferUsageFlags(
                ((usage & kBufferStorage) ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : 0) |
                ((usage & kBufferUniform) ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : 0) |
                ((usage & kBufferIndex) ? VK_BUFFER_USAGE_INDEX_BUFFER_BIT : 0) |
                ((usage & kBufferCopySrc) ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : 0) |
                ((usage & kBufferCopyDst) ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0));
        }

        inline VkShaderStageFlags ToVkStages(uint8_t stages)
        {
            return VkShaderStageFlags(
                ((stages & uint8_t(ShaderStage::Vertex)) ? VK_SHADER_STAGE_VERTEX_BIT : 0) |
                ((stages & uint8_t(ShaderStage::Fragment)) ? VK_SHADER_STAGE_FRAGMENT_BIT : 0) |
                ((stages & uint8_t(ShaderStage::Compute)) ? VK_SHADER_STAGE_COMPUTE_BIT : 0));
        }

        inline VkDescriptorType ToVk(BindingKind kind)
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
        inline void BufferUse(State state, VkPipelineStageFlags& stages, VkAccessFlags& access)
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


        inline VkFormat ToVk(Format f)
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

        inline Format FromVk(VkFormat f)
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

        inline int64_t NowNs()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        inline VkImageAspectFlags Aspect(Format f)
        {
            return HasDepth(f) ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VK_IMAGE_ASPECT_COLOR_BIT;
        }

        inline VkImageUsageFlags ToVk(ImageUsage usage)
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
        inline void ImageUse(State state, VkImageLayout& layout, VkPipelineStageFlags& stages, VkAccessFlags& access)
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

        inline VkBlendFactor ToVk(BlendFactor f)
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

        inline VkBlendOp ToVk(BlendOp op)
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

        inline VkCompareOp ToVk(CompareOp op)
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

        inline VkStencilOp ToVk(StencilOp op)
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

        inline VkPrimitiveTopology ToVk(Topology t)
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

        inline VkFilter ToVk(Filter f) { return f == Filter::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR; }

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

        // ---- the other source files ----------------------------------------------------

        // vulkan_commands.cpp
        std::unique_ptr<CommandList> MakeCommandList(Context& c);
        VkCommandBuffer CommandBufferOf(const CommandList& list);
        // vulkan_swapchain.cpp
        std::unique_ptr<Surface> MakeSurface(Context& c, VkSurfaceKHR surface, std::function<Extent3D()> windowPixels);
        // nullptr when the swapchain couldn't be made.
        std::unique_ptr<Swapchain> MakeSwapchain(Context& c, Surface& surface, const SwapchainDesc& desc);
        // The semaphores a submit to the swapchain waits for and signals.
        void SwapchainSubmitSemaphores(Swapchain& swapchain, VkSemaphore& wait, VkSemaphore& signal);
        VkFormat SwapchainFormat(const Swapchain& swapchain);
    }
}
