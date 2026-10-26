// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
#include "metal_device.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <spirv_cross/spirv_msl.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../command_list.h"
#include "../swapchain.h"
#include "metal_native.h"

// How the RHI maps onto Metal (DECISIONS.md D13):
//
//  * Memory. Device buffers are private, Upload ones shared and write-combined,
//    UploadCached ones shared; guest memory is imported with newBuffer(pointer,
//    ...), which needs the host's page size (16 KB on Apple silicon) for both
//    pointer and length. Apple silicon is unified, so Invalidate is nothing.
//  * Shaders are SPIR-V (D4): SPIRV-Cross makes MSL of them when a pipeline is
//    created, with every (table, slot) of the binding layout mapped to a fixed
//    Metal buffer / texture / sampler index (SlotMap), the push constants to a
//    buffer after the last of them, and the vertex shaders' Y flipped, so that
//    a picture drawn with Vulkan's clip space comes out the same.
//  * A command list is a MTLCommandBuffer with one encoder at a time: blit,
//    compute or render. Metal's bindings belong to an encoder, Vulkan's to the
//    command buffer, so the list keeps what was bound and applies it again to
//    each new encoder.
//  * Synchronisation. Resources made here are hazard-tracked by Metal, so
//    Transition has nothing to do. A Fence is a flag set by the command
//    buffer's completion handler; an Event is a MTLSharedEvent signalled
//    between encoders.
//  * metal-cpp's objects are reference counted by hand: NS::SharedPtr owns
//    what this file creates, and what Metal returns autoreleased is retained at
//    once, inside a pool of its own (NS::AutoreleasePool), since the game's
//    threads have none.
namespace gpu::rhi::metal
{
    namespace
    {
        template<typename T>
        using Ptr = NS::SharedPtr<T>;

        // A new*/alloc'd object: ours already (+1).
        template<typename T>
        Ptr<T> Own(T* created) { return NS::TransferPtr(created); }
        // An autoreleased one: retained.
        template<typename T>
        Ptr<T> Keep(T* borrowed) { return NS::RetainPtr(borrowed); }

        // An autorelease pool for the scope: the game's threads have none.
        struct Pool
        {
            NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
            ~Pool() { pool->release(); }
        };

        [[noreturn]] void Fail(const char* what)
        {
            fprintf(stderr, "[rhi/metal] %s\n", what);
            abort();
        }

        std::string Describe(const NS::Error* error)
        {
            return error && error->localizedDescription() ? error->localizedDescription()->utf8String() : "unknown error";
        }

        struct Context
        {
            Ptr<MTL::Device> device;
            Ptr<MTL::CommandQueue> queue;
            std::string deviceName;
            Capabilities caps;
            Ptr<MTL::SamplerState> blitSamplers[2];  // nearest, linear: the scaled copies'

            MTL::SamplerState* BlitSampler(Filter f) const { return blitSamplers[f == Filter::Linear].get(); }

            // Occlusion counts. A render pass names one visibility buffer when it
            // begins, before any query is, so every pool is a range of this one.
            static constexpr uint32_t kVisibilitySlots = 16384;
            Ptr<MTL::Buffer> visibility;
            std::mutex visibilityMutex;
            std::vector<std::pair<uint32_t, uint32_t>> visibilityFree{ { 0, kVisibilitySlots } };  // (first, count)
            // The timestamp counter set, when the GPU can sample it at encoder edges.
            Ptr<MTL::CounterSet> timestampCounters;
            Ptr<MTL::Buffer> scratch;  // what the timestamp encoders write

            // First slot of `count` free ones; false when there's no room.
            bool AllocateVisibility(uint32_t count, uint32_t& first)
            {
                std::lock_guard lock(visibilityMutex);
                for (auto it = visibilityFree.begin(); it != visibilityFree.end(); ++it)
                {
                    if (it->second < count)
                        continue;
                    first = it->first;
                    it->first += count;
                    it->second -= count;
                    if (!it->second)
                        visibilityFree.erase(it);
                    return true;
                }
                return false;
            }
            void ReleaseVisibility(uint32_t first, uint32_t count)
            {
                std::lock_guard lock(visibilityMutex);
                visibilityFree.emplace_back(first, count);
            }
        };

        // ---- formats ----------------------------------------------------------

        MTL::PixelFormat ToMtl(Format f)
        {
            switch (f)
            {
            case Format::R8: return MTL::PixelFormatR8Unorm;
            case Format::RG8: return MTL::PixelFormatRG8Unorm;
            case Format::RGBA8: return MTL::PixelFormatRGBA8Unorm;
            case Format::BGRA8: return MTL::PixelFormatBGRA8Unorm;
            case Format::RGBA8_sRGB: return MTL::PixelFormatRGBA8Unorm_sRGB;
            case Format::BGRA8_sRGB: return MTL::PixelFormatBGRA8Unorm_sRGB;
            case Format::R16: return MTL::PixelFormatR16Unorm;
            case Format::RG16: return MTL::PixelFormatRG16Unorm;
            case Format::RGBA16: return MTL::PixelFormatRGBA16Unorm;
            case Format::RG16_SNORM: return MTL::PixelFormatRG16Snorm;
            case Format::RGBA16_SNORM: return MTL::PixelFormatRGBA16Snorm;
            case Format::R16F: return MTL::PixelFormatR16Float;
            case Format::RG16F: return MTL::PixelFormatRG16Float;
            case Format::RGBA16F: return MTL::PixelFormatRGBA16Float;
            case Format::R32F: return MTL::PixelFormatR32Float;
            case Format::RG32F: return MTL::PixelFormatRG32Float;
            case Format::RGBA32F: return MTL::PixelFormatRGBA32Float;
            case Format::R32_UINT: return MTL::PixelFormatR32Uint;
            case Format::A2B10G10R10: return MTL::PixelFormatRGB10A2Unorm;
            case Format::D32F_S8: return MTL::PixelFormatDepth32Float_Stencil8;
            case Format::BC1: return MTL::PixelFormatBC1_RGBA;
            case Format::BC2: return MTL::PixelFormatBC2_RGBA;
            case Format::BC3: return MTL::PixelFormatBC3_RGBA;
            case Format::BC5: return MTL::PixelFormatBC5_RGUnorm;
            case Format::Undefined: break;
            }
            return MTL::PixelFormatInvalid;
        }

        uint32_t BytesPerPixel(Format f)
        {
            switch (f)
            {
            case Format::R8: return 1;
            case Format::RG8: case Format::R16: case Format::R16F: return 2;
            case Format::RGBA8: case Format::BGRA8: case Format::RGBA8_sRGB: case Format::BGRA8_sRGB: case Format::RG16:
            case Format::RG16_SNORM: case Format::RG16F: case Format::R32F: case Format::R32_UINT: case Format::A2B10G10R10: return 4;
            case Format::RGBA16: case Format::RGBA16_SNORM: case Format::RGBA16F: case Format::RG32F: case Format::D32F_S8: return 8;
            case Format::RGBA32F: return 16;
            default: return 0;  // blocks (BC) and Undefined
            }
        }

        bool IsBlock(Format f) { return f == Format::BC1 || f == Format::BC2 || f == Format::BC3 || f == Format::BC5; }
        bool HasDepth(Format f) { return f == Format::D32F_S8; }

        MTL::CompareFunction ToMtl(CompareOp op)
        {
            switch (op)
            {
            case CompareOp::Never: return MTL::CompareFunctionNever;
            case CompareOp::Less: return MTL::CompareFunctionLess;
            case CompareOp::Equal: return MTL::CompareFunctionEqual;
            case CompareOp::LessEqual: return MTL::CompareFunctionLessEqual;
            case CompareOp::Greater: return MTL::CompareFunctionGreater;
            case CompareOp::NotEqual: return MTL::CompareFunctionNotEqual;
            case CompareOp::GreaterEqual: return MTL::CompareFunctionGreaterEqual;
            case CompareOp::Always: break;
            }
            return MTL::CompareFunctionAlways;
        }

        MTL::StencilOperation ToMtl(StencilOp op)
        {
            switch (op)
            {
            case StencilOp::Keep: return MTL::StencilOperationKeep;
            case StencilOp::Zero: return MTL::StencilOperationZero;
            case StencilOp::Replace: return MTL::StencilOperationReplace;
            case StencilOp::IncrementClamp: return MTL::StencilOperationIncrementClamp;
            case StencilOp::DecrementClamp: return MTL::StencilOperationDecrementClamp;
            case StencilOp::Invert: return MTL::StencilOperationInvert;
            case StencilOp::IncrementWrap: return MTL::StencilOperationIncrementWrap;
            case StencilOp::DecrementWrap: return MTL::StencilOperationDecrementWrap;
            }
            return MTL::StencilOperationKeep;
        }

        MTL::BlendFactor ToMtl(BlendFactor f)
        {
            switch (f)
            {
            case BlendFactor::Zero: return MTL::BlendFactorZero;
            case BlendFactor::One: return MTL::BlendFactorOne;
            case BlendFactor::SrcColor: return MTL::BlendFactorSourceColor;
            case BlendFactor::OneMinusSrcColor: return MTL::BlendFactorOneMinusSourceColor;
            case BlendFactor::DstColor: return MTL::BlendFactorDestinationColor;
            case BlendFactor::OneMinusDstColor: return MTL::BlendFactorOneMinusDestinationColor;
            case BlendFactor::SrcAlpha: return MTL::BlendFactorSourceAlpha;
            case BlendFactor::OneMinusSrcAlpha: return MTL::BlendFactorOneMinusSourceAlpha;
            case BlendFactor::DstAlpha: return MTL::BlendFactorDestinationAlpha;
            case BlendFactor::OneMinusDstAlpha: return MTL::BlendFactorOneMinusDestinationAlpha;
            case BlendFactor::ConstantColor: return MTL::BlendFactorBlendColor;
            case BlendFactor::OneMinusConstantColor: return MTL::BlendFactorOneMinusBlendColor;
            case BlendFactor::ConstantAlpha: return MTL::BlendFactorBlendAlpha;
            case BlendFactor::OneMinusConstantAlpha: return MTL::BlendFactorOneMinusBlendAlpha;
            case BlendFactor::SrcAlphaSaturate: return MTL::BlendFactorSourceAlphaSaturated;
            }
            return MTL::BlendFactorOne;
        }

        MTL::BlendOperation ToMtl(BlendOp op)
        {
            switch (op)
            {
            case BlendOp::Add: return MTL::BlendOperationAdd;
            case BlendOp::Subtract: return MTL::BlendOperationSubtract;
            case BlendOp::ReverseSubtract: return MTL::BlendOperationReverseSubtract;
            case BlendOp::Min: return MTL::BlendOperationMin;
            case BlendOp::Max: return MTL::BlendOperationMax;
            }
            return MTL::BlendOperationAdd;
        }

        MTL::PrimitiveType ToMtl(Topology t)
        {
            switch (t)
            {
            case Topology::PointList: return MTL::PrimitiveTypePoint;
            case Topology::LineList: return MTL::PrimitiveTypeLine;
            case Topology::LineStrip: return MTL::PrimitiveTypeLineStrip;
            case Topology::TriangleList: return MTL::PrimitiveTypeTriangle;
            case Topology::TriangleStrip: return MTL::PrimitiveTypeTriangleStrip;
            }
            return MTL::PrimitiveTypeTriangle;
        }

        MTL::PrimitiveTopologyClass TopologyClass(Topology t)
        {
            switch (t)
            {
            case Topology::PointList: return MTL::PrimitiveTopologyClassPoint;
            case Topology::LineList: case Topology::LineStrip: return MTL::PrimitiveTopologyClassLine;
            default: return MTL::PrimitiveTopologyClassTriangle;
            }
        }

        MTL::SamplerMinMagFilter ToMtl(Filter f) { return f == Filter::Nearest ? MTL::SamplerMinMagFilterNearest : MTL::SamplerMinMagFilterLinear; }

        MTL::SamplerAddressMode ToMtl(AddressMode a)
        {
            return a == AddressMode::ClampToEdge ? MTL::SamplerAddressModeClampToEdge
                : a == AddressMode::ClampToBorder ? MTL::SamplerAddressModeClampToBorderColor
                : a == AddressMode::MirrorRepeat ? MTL::SamplerAddressModeMirrorRepeat : MTL::SamplerAddressModeRepeat;
        }

        // ---- resources -------------------------------------------------------

        class MetalBuffer final : public Buffer
        {
        public:
            MetalBuffer(Ptr<MTL::Buffer> buffer, uint64_t size, void* mapped) : m_buffer(std::move(buffer)), m_size(size), m_mapped(mapped) {}
            uint64_t Size() const override { return m_size; }
            void* Map() override { return m_mapped; }
            void Invalidate(uint64_t, uint64_t) override {}  // unified memory: the GPU's writes are the CPU's once the fence is seen
            MTL::Buffer* Handle() const { return m_buffer.get(); }

        private:
            Ptr<MTL::Buffer> m_buffer;
            uint64_t m_size;
            void* m_mapped;
        };

        class MetalSampler final : public Sampler
        {
        public:
            explicit MetalSampler(Ptr<MTL::SamplerState> state) : m_state(std::move(state)) {}
            MTL::SamplerState* Handle() const { return m_state.get(); }

        private:
            Ptr<MTL::SamplerState> m_state;
        };

        class MetalImage final : public Image
        {
        public:
            MetalImage(const ImageDesc& desc, Ptr<MTL::Texture> texture, uint64_t bytes) : m_desc(desc), m_texture(std::move(texture)), m_bytes(bytes) {}
            const ImageDesc& Desc() const override { return m_desc; }
            uint64_t Bytes() const override { return m_bytes; }
            MTL::Texture* Handle() const { return m_texture.get(); }

        private:
            ImageDesc m_desc;
            Ptr<MTL::Texture> m_texture;
            uint64_t m_bytes;
        };

        // A view of an image other than its own (ImageViewDesc).
        class MetalImageView final : public ImageView
        {
        public:
            explicit MetalImageView(Ptr<MTL::Texture> texture) : m_texture(std::move(texture)) {}
            MTL::Texture* Handle() const { return m_texture.get(); }

        private:
            Ptr<MTL::Texture> m_texture;
        };

        // ---- bindings (D10) ----------------------------------------------------

        // Where one slot of a layout lives in Metal's argument tables.
        struct SlotMap
        {
            BindingSlot slot;
            uint32_t table = 0;
            uint32_t buffer = ~0u, texture = ~0u, sampler = ~0u;  // ~0: the slot has none
            int32_t dynamicOrder = -1;                            // among the table's UniformBufferDynamic slots
        };

        class MetalBindingLayout final : public BindingLayout
        {
        public:
            explicit MetalBindingLayout(BindingLayoutDesc desc) : m_desc(std::move(desc))
            {
                uint32_t buffers = 0, textures = 0, samplers = 0;
                for (uint32_t t = 0; t < m_desc.tables.size(); t++)
                {
                    int32_t dynamic = 0;
                    for (const BindingSlot& slot : m_desc.tables[t].slots)
                    {
                        SlotMap m;
                        m.slot = slot;
                        m.table = t;
                        switch (slot.kind)
                        {
                        case BindingKind::StorageBuffer: m.buffer = buffers++; break;
                        case BindingKind::UniformBufferDynamic: m.buffer = buffers++; m.dynamicOrder = dynamic++; break;
                        case BindingKind::CombinedImageSampler: m.texture = textures++; m.sampler = samplers++; break;
                        case BindingKind::SampledImage: m.texture = textures++; break;
                        case BindingKind::StorageImage: m.texture = textures++; break;
                        }
                        m_slots.push_back(m);
                    }
                }
                m_pushConstantBuffer = buffers;
            }

            const BindingLayoutDesc& Desc() const { return m_desc; }
            const std::vector<SlotMap>& Slots() const { return m_slots; }
            uint32_t PushConstantBuffer() const { return m_pushConstantBuffer; }
            const SlotMap* Find(uint32_t table, uint32_t slot) const
            {
                for (const SlotMap& m : m_slots)
                    if (m.table == table && m.slot.index == slot)
                        return &m;
                return nullptr;
            }

        private:
            BindingLayoutDesc m_desc;
            std::vector<SlotMap> m_slots;
            uint32_t m_pushConstantBuffer = 0;
        };

        class MetalBindingSet final : public BindingSet
        {
        public:
            MetalBindingSet(const MetalBindingLayout& layout, uint32_t table) : m_layout(layout), m_table(table) {}
            void Update(const BindingEntry* entries, size_t count) override
            {
                std::lock_guard lock(m_mutex);
                for (size_t i = 0; i < count; i++)
                {
                    const SlotMap* m = m_layout.Find(m_table, entries[i].slot);
                    if (!m)
                        Fail("a binding set has no such slot");
                    auto it = std::find_if(m_entries.begin(), m_entries.end(), [&](const BindingEntry& e) { return e.slot == entries[i].slot; });
                    if (it != m_entries.end())
                        *it = entries[i];
                    else
                        m_entries.push_back(entries[i]);
                }
            }
            // The entries now (copied: another thread may Update meanwhile).
            std::vector<BindingEntry> Entries() const
            {
                std::lock_guard lock(m_mutex);
                return m_entries;
            }
            uint32_t Table() const { return m_table; }

        private:
            const MetalBindingLayout& m_layout;
            uint32_t m_table;
            mutable std::mutex m_mutex;
            std::vector<BindingEntry> m_entries;
        };

        // ---- shaders -------------------------------------------------------------

        // MSL for `code` as stage `model` against `layout`; the entry point's
        // name and (compute) workgroup size come back in `entry`, `size`.
        std::string ToMsl(const ShaderCode& code, spv::ExecutionModel model, const MetalBindingLayout& layout, std::string& entry, MTL::Size* size, bool pointSize = false)
        {
            spirv_cross::CompilerMSL compiler(code.spirv, code.words);
            spirv_cross::CompilerMSL::Options options;
            options.platform = spirv_cross::CompilerMSL::Options::macOS;
            options.set_msl_version(3, 0);
            options.argument_buffers = false;
            options.enable_point_size_builtin = pointSize;  // (Metal rejects a point size unless it draws points)
            options.enable_decoration_binding = false;
            // Vulkan's clip space has Y down and Metal's up: the vertex shaders'
            // gl_Position.y is flipped, so a picture comes out the same way up.
            options.pad_fragment_output_components = true;
            compiler.set_msl_options(options);
            spirv_cross::CompilerGLSL::Options common = compiler.get_common_options();
            common.vertex.flip_vert_y = model == spv::ExecutionModelVertex;
            compiler.set_common_options(common);
            for (const SlotMap& m : layout.Slots())
            {
                spirv_cross::MSLResourceBinding b;
                b.stage = model;
                b.desc_set = m.table;
                b.binding = m.slot.index;
                b.msl_buffer = m.buffer != ~0u ? m.buffer : 0;
                b.msl_texture = m.texture != ~0u ? m.texture : 0;
                b.msl_sampler = m.sampler != ~0u ? m.sampler : 0;
                compiler.add_msl_resource_binding(b);
            }
            if (layout.Desc().pushConstantBytes)
            {
                spirv_cross::MSLResourceBinding b;
                b.stage = model;
                b.desc_set = spirv_cross::kPushConstDescSet;
                b.binding = spirv_cross::kPushConstBinding;
                b.msl_buffer = layout.PushConstantBuffer();
                compiler.add_msl_resource_binding(b);
            }
            std::string msl = compiler.compile();
            entry = compiler.get_cleansed_entry_point_name("main", model);
            if (size)
            {
                const spirv_cross::SPIREntryPoint& ep = compiler.get_entry_point("main", model);
                *size = MTL::Size::Make(ep.workgroup_size.x, ep.workgroup_size.y, ep.workgroup_size.z);
            }
            return msl;
        }

        // A function of `msl` named `entry`, or an empty pointer (logged).
        Ptr<MTL::Function> MakeFunction(MTL::Device& device, const std::string& msl, const std::string& entry)
        {
            Pool pool;
            auto options = Own(MTL::CompileOptions::alloc()->init());
            options->setLanguageVersion(MTL::LanguageVersion3_0);
            NS::Error* error = nullptr;
            NS::String* source = NS::String::string(msl.c_str(), NS::UTF8StringEncoding);
            Ptr<MTL::Library> library = Own(device.newLibrary(source, options.get(), &error));
            if (!library)
            {
                fprintf(stderr, "[rhi/metal] shader didn't compile: %s\n", Describe(error).c_str());
                return {};
            }
            return Own(library->newFunction(NS::String::string(entry.c_str(), NS::UTF8StringEncoding)));
        }

        // ---- pipelines -------------------------------------------------------------

        class MetalPipeline : public Pipeline
        {
        public:
            virtual bool IsCompute() const = 0;
            const MetalBindingLayout* layout = nullptr;  // what it was made against
        };

        class MetalComputePipeline final : public MetalPipeline
        {
        public:
            MetalComputePipeline(Ptr<MTL::ComputePipelineState> state, MTL::Size group) : m_state(std::move(state)), m_group(group) {}
            bool IsCompute() const override { return true; }
            MTL::ComputePipelineState* Handle() const { return m_state.get(); }
            MTL::Size Group() const { return m_group; }

        private:
            Ptr<MTL::ComputePipelineState> m_state;
            MTL::Size m_group;
        };

        class MetalGraphicsPipeline final : public MetalPipeline
        {
        public:
            bool IsCompute() const override { return false; }
            Ptr<MTL::RenderPipelineState> state;
            GraphicsPipelineDesc desc;  // (shader pointers dropped: raster and depth-stencil state only)
            MTL::Device* device = nullptr;
            MTL::PrimitiveType primitive = MTL::PrimitiveTypeTriangle;
            MTL::CullMode cull = MTL::CullModeNone;
            MTL::Winding winding = MTL::WindingClockwise;

            // The depth-stencil state for the stencil masks set now. Metal keeps
            // them in the state object, Vulkan makes them dynamic.
            MTL::DepthStencilState* DepthStencil(const CommandList::StencilState& front, const CommandList::StencilState& back) const
            {
                const uint32_t key = uint32_t(front.compareMask) << 24 | uint32_t(front.writeMask) << 16 | uint32_t(back.compareMask) << 8 | back.writeMask;
                std::lock_guard lock(m_mutex);
                auto it = m_states.find(key);
                if (it != m_states.end())
                    return it->second.get();
                auto d = Own(MTL::DepthStencilDescriptor::alloc()->init());
                d->setDepthCompareFunction(desc.depthTest ? ToMtl(desc.depthCompare) : MTL::CompareFunctionAlways);
                d->setDepthWriteEnabled(desc.depthTest && desc.depthWrite);  // (Vulkan: writes only with the test on)
                if (desc.stencilTest)
                {
                    auto face = [&](const StencilFace& f, const CommandList::StencilState& state) {
                        auto s = Own(MTL::StencilDescriptor::alloc()->init());
                        s->setStencilFailureOperation(ToMtl(f.fail));
                        s->setDepthStencilPassOperation(ToMtl(f.pass));
                        s->setDepthFailureOperation(ToMtl(f.depthFail));
                        s->setStencilCompareFunction(ToMtl(f.compare));
                        s->setReadMask(state.compareMask);
                        s->setWriteMask(state.writeMask);
                        return s;
                    };
                    auto frontFace = face(desc.stencilFront, front), backFace = face(desc.stencilBack, back);
                    d->setFrontFaceStencil(frontFace.get());
                    d->setBackFaceStencil(backFace.get());
                }
                Ptr<MTL::DepthStencilState> state = Own(device->newDepthStencilState(d.get()));
                MTL::DepthStencilState* raw = state.get();
                m_states.emplace(key, std::move(state));
                return raw;
            }

        private:
            mutable std::mutex m_mutex;
            mutable std::map<uint32_t, Ptr<MTL::DepthStencilState>> m_states;
        };

        // ---- synchronisation ---------------------------------------------------------

        // Set by a command buffer's completion handler.
        struct FenceState
        {
            std::mutex mutex;
            std::condition_variable cv;
            bool signaled = false;
        };

        class MetalFence final : public Fence
        {
        public:
            explicit MetalFence(bool signaled) : m_state(std::make_shared<FenceState>()) { m_state->signaled = signaled; }
            bool Signaled() override
            {
                std::lock_guard lock(m_state->mutex);
                return m_state->signaled;
            }
            bool Wait(uint64_t timeoutNs) override
            {
                std::unique_lock lock(m_state->mutex);
                if (timeoutNs == UINT64_MAX)
                {
                    m_state->cv.wait(lock, [&] { return m_state->signaled; });
                    return true;
                }
                return m_state->cv.wait_for(lock, std::chrono::nanoseconds(timeoutNs), [&] { return m_state->signaled; });
            }
            void Reset() override
            {
                std::lock_guard lock(m_state->mutex);
                m_state->signaled = false;
            }
            // Submit: unsignalled until the handler runs.
            std::shared_ptr<FenceState> Arm()
            {
                Reset();
                return m_state;
            }

        private:
            std::shared_ptr<FenceState> m_state;
        };

        class MetalEvent final : public Event
        {
        public:
            explicit MetalEvent(Ptr<MTL::SharedEvent> event) : m_event(std::move(event)) {}
            bool Signaled() override
            {
                const uint64_t expect = m_expect.load(std::memory_order_acquire);
                return expect && m_event->signaledValue() >= expect;
            }
            void Reset() override { m_expect.store(0, std::memory_order_release); }
            // The value the next SignalEvent signals.
            uint64_t Next()
            {
                const uint64_t value = ++m_counter;
                m_expect.store(value, std::memory_order_release);
                return value;
            }
            MTL::SharedEvent* Handle() const { return m_event.get(); }

        private:
            Ptr<MTL::SharedEvent> m_event;
            std::atomic<uint64_t> m_counter{ 0 }, m_expect{ 0 };
        };

        class MetalQueryPool final : public QueryPool
        {
        public:
            MetalQueryPool(Context& c, QueryKind kind, uint32_t count) : m_c(c), m_kind(kind), m_count(count)
            {
                Pool pool;
                if (kind == QueryKind::Occlusion)
                {
                    if (!c.AllocateVisibility(count, m_first))
                        Fail("out of occlusion query slots");
                    return;
                }
                if (!c.timestampCounters)
                    Fail("timestamps are not offered");
                auto d = Own(MTL::CounterSampleBufferDescriptor::alloc()->init());
                d->setCounterSet(c.timestampCounters.get());
                d->setStorageMode(MTL::StorageModeShared);
                d->setSampleCount(count);
                NS::Error* error = nullptr;
                m_samples = Own(c.device->newCounterSampleBuffer(d.get(), &error));
                if (!m_samples)
                    Fail(("the timestamp buffer failed: " + Describe(error)).c_str());
            }
            ~MetalQueryPool() override
            {
                if (m_kind == QueryKind::Occlusion)
                    m_c.ReleaseVisibility(m_first, m_count);
            }

            bool Read(uint32_t first, uint32_t count, uint64_t* out) override
            {
                if (first + count > m_count)
                    return false;
                if (m_kind == QueryKind::Occlusion)
                {
                    const auto* counts = static_cast<const uint64_t*>(m_c.visibility->contents()) + m_first + first;
                    std::copy(counts, counts + count, out);
                    return true;
                }
                Pool pool;
                NS::Data* data = m_samples->resolveCounterRange(NS::Range(first, count));
                if (!data || data->length() < count * sizeof(uint64_t))
                    return false;
                const auto* ticks = static_cast<const uint64_t*>(data->bytes());
                for (uint32_t i = 0; i < count; i++)
                {
                    if (ticks[i] == MTL::CounterErrorValue)  // never written
                        return false;
                    out[i] = ticks[i];
                }
                return true;
            }

            bool Reset(uint32_t first, uint32_t count) override
            {
                if (first + count > m_count)
                    return false;
                if (m_kind == QueryKind::Occlusion)  // (the counts are in a shared buffer)
                    memset(static_cast<uint64_t*>(m_c.visibility->contents()) + m_first + first, 0, size_t(count) * sizeof(uint64_t));
                return true;  // (a timestamp is overwritten by its next sample)
            }

            QueryKind Kind() const { return m_kind; }
            uint32_t Count() const { return m_count; }
            // Occlusion: the byte offset of query `index` in the visibility buffer.
            NS::UInteger VisibilityOffset(uint32_t index) const { return NS::UInteger(m_first + index) * sizeof(uint64_t); }
            MTL::CounterSampleBuffer* Samples() const { return m_samples.get(); }

        private:
            Context& m_c;
            QueryKind m_kind;
            uint32_t m_count;
            uint32_t m_first = 0;
            Ptr<MTL::CounterSampleBuffer> m_samples;
        };

        // ---- command lists -------------------------------------------------------------

        // MSL of our own: a clear of part of the attachments in a pass
        // (ClearAttachments) and a scaled copy of one image into another
        // (BlitImage; Metal's blit encoder can't scale). One pipeline per
        // target format, made on first use.
        class InternalPipelines
        {
        public:
            explicit InternalPipelines(Context& c) : m_c(c) {}

            // The pipeline that fills the attachments `colorMask` selects (of
            // `formats`), the depth if `depth`, with the values in buffer 0.
            MTL::RenderPipelineState* Clear(const Format colorFormats[4], uint32_t colorMask, Format depthFormat, bool depth)
            {
                std::string key = "clear";
                for (int i = 0; i < 4; i++)
                    key += "," + std::to_string(int(colorFormats[i])) + (colorMask >> i & 1 ? "w" : "-");
                key += "," + std::to_string(int(depthFormat)) + (depth ? "d" : "-");
                std::lock_guard lock(m_mutex);
                if (auto it = m_states.find(key); it != m_states.end())
                    return it->second.get();
                std::string src = "#include <metal_stdlib>\nusing namespace metal;\nstruct K { float4 c[4]; float d; };\n"
                                  "vertex float4 vs(uint v [[vertex_id]]) { float2 p = float2(float((v << 1) & 2u), float(v & 2u)); return float4(p * 2.0 - 1.0, 0.0, 1.0); }\n"
                                  "struct O {\n";
                for (int i = 0; i < 4; i++)
                    if (colorFormats[i] != Format::Undefined)
                        src += "  float4 c" + std::to_string(i) + " [[color(" + std::to_string(i) + ")]];\n";
                if (depth)
                    src += "  float d [[depth(any)]];\n";
                src += "};\nfragment O fs(constant K& k [[buffer(0)]]) {\n  O o;\n";
                for (int i = 0; i < 4; i++)
                    if (colorFormats[i] != Format::Undefined)
                        src += "  o.c" + std::to_string(i) + " = k.c[" + std::to_string(i) + "];\n";
                if (depth)
                    src += "  o.d = k.d;\n";
                src += "  return o;\n}\n";
                auto d = Own(MTL::RenderPipelineDescriptor::alloc()->init());
                if (!Functions(src, "vs", "fs", *d.get()))
                    Fail("the internal clear shader didn't compile");
                for (int i = 0; i < 4; i++)
                    if (colorFormats[i] != Format::Undefined)
                    {
                        MTL::RenderPipelineColorAttachmentDescriptor* a = d->colorAttachments()->object(i);
                        a->setPixelFormat(ToMtl(colorFormats[i]));
                        a->setWriteMask(colorMask >> i & 1 ? MTL::ColorWriteMaskAll : MTL::ColorWriteMaskNone);
                    }
                if (depthFormat != Format::Undefined)
                {
                    d->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float_Stencil8);
                    d->setStencilAttachmentPixelFormat(MTL::PixelFormatDepth32Float_Stencil8);
                }
                return Store(key, *d.get());
            }

            // The pipeline that draws a source texture into a `format` target
            // over the viewport; origin and scale of the source in buffer 0.
            MTL::RenderPipelineState* Scale(Format format)
            {
                const std::string key = "scale," + std::to_string(int(format));
                std::lock_guard lock(m_mutex);
                if (auto it = m_states.find(key); it != m_states.end())
                    return it->second.get();
                const std::string src =
                    "#include <metal_stdlib>\nusing namespace metal;\nstruct B { float2 origin; float2 scale; };\n"
                    "struct V { float4 pos [[position]]; float2 uv; };\n"
                    "vertex V vs(uint v [[vertex_id]], constant B& b [[buffer(0)]]) {\n"
                    "  float2 p = float2(float((v << 1) & 2u), float(v & 2u));\n"
                    "  V o; o.pos = float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0); o.uv = b.origin + p * b.scale; return o; }\n"
                    "fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) { return t.sample(s, in.uv); }\n";
                auto d = Own(MTL::RenderPipelineDescriptor::alloc()->init());
                if (!Functions(src, "vs", "fs", *d.get()))
                    Fail("the internal scale shader didn't compile");
                d->colorAttachments()->object(0)->setPixelFormat(ToMtl(format));
                return Store(key, *d.get());
            }

        private:
            bool Functions(const std::string& src, const char* vs, const char* fs, MTL::RenderPipelineDescriptor& d)
            {
                Ptr<MTL::Function> v = MakeFunction(*m_c.device.get(), src, vs), f = MakeFunction(*m_c.device.get(), src, fs);
                if (!v || !f)
                    return false;
                d.setVertexFunction(v.get());
                d.setFragmentFunction(f.get());
                return true;
            }

            MTL::RenderPipelineState* Store(const std::string& key, MTL::RenderPipelineDescriptor& d)
            {
                NS::Error* error = nullptr;
                Ptr<MTL::RenderPipelineState> state = Own(m_c.device->newRenderPipelineState(&d, &error));
                if (!state)
                    Fail(("an internal pipeline failed: " + Describe(error)).c_str());
                MTL::RenderPipelineState* raw = state.get();
                m_states.emplace(key, std::move(state));
                return raw;
            }

            Context& m_c;
            std::mutex m_mutex;
            std::map<std::string, Ptr<MTL::RenderPipelineState>> m_states;
        };

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

        // ---- the device ----------------------------------------------------------------

        // ---- the window ------------------------------------------------------------

        class MetalSurface final : public Surface
        {
        public:
            explicit MetalSurface(CA::MetalLayer* layer) : m_layer(Keep(layer)) {}
            Extent3D CurrentExtent() override
            {
                const CGSize size = m_layer->drawableSize();
                return { uint32_t(std::max(size.width, 0.0)), uint32_t(std::max(size.height, 0.0)), 1 };
            }
            CA::MetalLayer* Layer() const { return m_layer.get(); }

        private:
            Ptr<CA::MetalLayer> m_layer;
        };

        // Drawables come from the layer; a present is a command buffer of its own
        // on the queue (after the one that drew, so it waits for it) that shows
        // the drawable, at a time when asked. The presented handler is both
        // callbacks of swapchain.h: it has the display's own time of the frame.
        class MetalSwapchain final : public Swapchain
        {
        public:
            MetalSwapchain(Context& c, MetalSurface& surface, const SwapchainDesc& desc)
                : m_c(c), m_layer(Keep(surface.Layer())), m_desc(desc), m_extent(desc.extent)
            {
                m_layer->setDevice(c.device.get());
                m_layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
                m_layer->setFramebufferOnly(!desc.readable);
                m_layer->setDrawableSize(CGSize{ double(desc.extent.width), double(desc.extent.height) });
                m_layer->setMaximumDrawableCount(std::clamp<uint32_t>(desc.minImages, 2, 3));
                m_layer->setDisplaySyncEnabled(desc.vsync);
                m_layer->setAllowsNextDrawableTimeout(false);  // Acquire blocks until one is free
            }

            ~MetalSwapchain() override
            {
                Pool pool;
                std::unique_lock lock(m_mutex);
                if (!m_idle.wait_for(lock, std::chrono::seconds(2), [this] { return m_inFlight == 0; }))
                    fprintf(stderr, "[rhi/metal] %u presents never reported; going on\n", m_inFlight);
            }

            Extent3D Extent() const override { return m_extent; }
            Format GetFormat() const override { return Format::BGRA8; }
            uint32_t ImageCount() const override { return std::clamp<uint32_t>(m_desc.minImages, 2, 3); }
            bool Readable() const override { return m_desc.readable; }
            const char* PresentModeName() const override { return m_desc.vsync ? "displaySync" : "immediate"; }
            std::string Describe() const override
            {
                return std::string("CAMetalLayer, BGRA8, ") + PresentModeName() + ", " + std::to_string(ImageCount()) + " drawables";
            }

            SurfaceStatus Acquire(Image*& image) override
            {
                Pool pool;
                image = nullptr;
                m_image.reset();
                m_drawable = Keep(m_layer->nextDrawable());
                if (!m_drawable)
                    return SurfaceStatus::OutOfDate;
                MTL::Texture* texture = m_drawable->texture();
                if (texture->width() != m_extent.width || texture->height() != m_extent.height)
                {
                    m_drawable.reset();  // the window changed size: the policy builds it again
                    return SurfaceStatus::OutOfDate;
                }
                ImageDesc d;
                d.format = Format::BGRA8;
                d.extent = m_extent;
                d.usage = kImageRenderTarget | kImageCopyDst | (m_desc.readable ? kImageCopySrc : 0);
                d.name = "swapchain image";
                m_image = std::make_unique<MetalImage>(d, Keep(texture), 0);
                image = m_image.get();
                return SurfaceStatus::Ok;
            }

            SurfaceStatus Present(const PresentRequest& request) override
            {
                Pool pool;
                if (!m_drawable)
                    return SurfaceStatus::OutOfDate;
                const TimeNs presented = request.presented ? request.presented : NowNs();
                const bool timed = request.id && (m_desc.onShown || m_desc.onReport);
                if (timed)
                {
                    uint64_t sequence;
                    {
                        std::lock_guard lock(m_mutex);
                        m_inFlight++;
                        sequence = m_nextSequence++;
                    }
                    const uint64_t id = request.id;
                    const bool wantReport = request.notBefore || m_desc.reportAll;
                    m_drawable->addPresentedHandler([this, sequence, id, presented, wantReport](MTL::Drawable* d) {
                        const double t = d->presentedTime();  // seconds on the uptime clock; 0: dropped
                        const TimeNs onScreen = t > 0 ? TimeNs(t * 1e9) + SteadyMinusUptime() : 0;
                        {
                            // The handlers can run on different threads, out of order: the
                            // callbacks are in present order, one at a time.
                            std::lock_guard lock(m_callbackMutex);
                            m_finished[sequence] = Finished{ id, presented, onScreen, wantReport };
                            while (!m_finished.empty() && m_finished.begin()->first == m_deliverNext)
                            {
                                const Finished f = m_finished.begin()->second;
                                m_finished.erase(m_finished.begin());
                                m_deliverNext++;
                                thread_local const void* started = nullptr;
                                if (started != this)
                                {
                                    started = this;
                                    if (m_desc.onThreadStart)
                                        m_desc.onThreadStart();
                                }
                                if (f.onScreen && m_desc.onShown)
                                    m_desc.onShown(PresentShown{ f.id, f.presented, f.onScreen, true });
                                if (f.wantReport && m_desc.onReport)
                                    m_desc.onReport(PresentReport{ f.id, f.onScreen });
                            }
                        }
                        std::lock_guard lock(m_mutex);
                        m_inFlight--;
                        m_idle.notify_all();
                    });
                }
                MTL::CommandBuffer* cb = m_c.queue->commandBuffer();
                if (request.notBefore)
                    cb->presentDrawableAtTime(m_drawable.get(), double(request.notBefore - SteadyMinusUptime()) * 1e-9);
                else
                    cb->presentDrawable(m_drawable.get());
                cb->commit();
                m_drawable.reset();
                m_image.reset();
                return SurfaceStatus::Ok;
            }

        private:
            // Metal's times are mach_absolute_time's, which stops while the Mac
            // sleeps; std::chrono::steady_clock here keeps going (it is
            // mach_continuous_time). The difference, now.
            static TimeNs SteadyMinusUptime()
            {
                return TimeNs(clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)) - TimeNs(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
            }
            static TimeNs NowNs()
            {
                return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            }

            Context& m_c;
            Ptr<CA::MetalLayer> m_layer;
            SwapchainDesc m_desc;
            Extent3D m_extent;
            Ptr<CA::MetalDrawable> m_drawable;
            std::unique_ptr<MetalImage> m_image;
            std::mutex m_mutex;          // m_inFlight
            std::condition_variable m_idle;
            uint32_t m_inFlight = 0;
            std::mutex m_callbackMutex;  // the callbacks come one at a time
            struct Finished { uint64_t id; TimeNs presented, onScreen; bool wantReport; };
            uint64_t m_nextSequence = 0;                // (under m_mutex) given to each timed present
            uint64_t m_deliverNext = 0;                 // (under m_callbackMutex) the next to call back
            std::map<uint64_t, Finished> m_finished;    // (same) done, waiting for its turn
        };

        class MetalDevice final : public Device
        {
        public:
            explicit MetalDevice(std::unique_ptr<Context> context) : m_c(std::move(context)), m_internal(*m_c) {}

            const Capabilities& GetCapabilities() const override { return m_c->caps; }

            std::unique_ptr<Buffer> CreateBuffer(const BufferDesc& desc) override
            {
                Pool pool;
                // (Stream: the CPU writes it every frame, the GPU reads it: unified memory, write-combined.)
                MTL::ResourceOptions options = desc.memory == MemoryKind::Device ? MTL::ResourceStorageModePrivate
                    : desc.memory == MemoryKind::Upload || desc.memory == MemoryKind::Stream
                        ? (MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined)
                        : MTL::ResourceStorageModeShared;
                Ptr<MTL::Buffer> buffer = Own(m_c->device->newBuffer(desc.size, options));
                if (!buffer)
                    return nullptr;
                SetLabel(*buffer.get(), desc.name);
                void* mapped = desc.memory == MemoryKind::Device ? nullptr : buffer->contents();
                return std::make_unique<MetalBuffer>(std::move(buffer), desc.size, mapped);
            }

            std::unique_ptr<Buffer> ImportHostMemory(void* host, uint64_t size, uint32_t, const char* name) override
            {
                Pool pool;
                const uint64_t page = m_c->caps.hostMemoryAlignment;
                if (!host || reinterpret_cast<uintptr_t>(host) % page || size % page || size > m_c->caps.maxBufferSize)
                    return nullptr;
                Ptr<MTL::Buffer> buffer = Own(m_c->device->newBuffer(host, size, MTL::ResourceStorageModeShared, nullptr));
                if (!buffer)
                    return nullptr;
                SetLabel(*buffer.get(), name);
                return std::make_unique<MetalBuffer>(std::move(buffer), size, host);
            }

            bool CreateGuestMemory(void* host, uint64_t size, GuestMemory& out) override
            {
                // Imported without a copy where the pointer and size are the pages' (16 KB), else a copy
                // the CPU keeps current (the renderer's shadow mode).
                out.buffer = ImportHostMemory(host, size, 0, "guest memory");
                if (out.buffer)
                {
                    fprintf(stderr, "[rhi/metal] guest memory: imported (zero-copy)\n");
                    return true;
                }
                Pool pool;
                Ptr<MTL::Buffer> buffer = Own(m_c->device->newBuffer(size, MTL::ResourceStorageModeShared));
                if (!buffer)
                    return false;
                SetLabel(*buffer.get(), "guest memory (copy)");
                void* mapped = buffer->contents();
                out.buffer = std::make_unique<MetalBuffer>(std::move(buffer), size, mapped);
                out.shadow = static_cast<uint8_t*>(mapped);
                fprintf(stderr, "[rhi/metal] guest memory: a copy kept current from the write watch (the host memory can't be imported)\n");
                return true;
            }

            std::unique_ptr<ImageView> CreateImageView(const Image& image, const ImageViewDesc& desc) override
            {
                Pool pool;
                const auto& img = static_cast<const MetalImage&>(image);
                const ImageDesc& d = img.Desc();
                const Format format = desc.format == Format::Undefined ? d.format : desc.format;
                const MTL::PixelFormat pixel = ToMtl(format);
                if (pixel == MTL::PixelFormatInvalid)
                    return nullptr;
                auto channel = [](Swizzle s, MTL::TextureSwizzle identity) {
                    switch (s)
                    {
                    case Swizzle::R: return MTL::TextureSwizzleRed;
                    case Swizzle::G: return MTL::TextureSwizzleGreen;
                    case Swizzle::B: return MTL::TextureSwizzleBlue;
                    case Swizzle::A: return MTL::TextureSwizzleAlpha;
                    case Swizzle::Zero: return MTL::TextureSwizzleZero;
                    case Swizzle::One: return MTL::TextureSwizzleOne;
                    default: return identity;
                    }
                };
                const MTL::TextureSwizzleChannels swizzle = MTL::TextureSwizzleChannels::Make(channel(desc.swizzle[0], MTL::TextureSwizzleRed),
                    channel(desc.swizzle[1], MTL::TextureSwizzleGreen), channel(desc.swizzle[2], MTL::TextureSwizzleBlue),
                    channel(desc.swizzle[3], MTL::TextureSwizzleAlpha));
                MTL::TextureType type;
                NS::Range slices(0, 1);
                if (desc.layerCount)
                {
                    type = desc.layerCount > 1 ? MTL::TextureType2DArray : MTL::TextureType2D;
                    slices = NS::Range(desc.layer, desc.layerCount);
                }
                else
                {
                    type = d.kind == ImageKind::Image3D ? MTL::TextureType3D : d.kind == ImageKind::Cube ? MTL::TextureTypeCube
                        : d.layers > 1 ? MTL::TextureType2DArray : MTL::TextureType2D;
                    slices = NS::Range(0, d.kind == ImageKind::Cube ? 6 : d.layers);
                }
                Ptr<MTL::Texture> view = Own(img.Handle()->newTextureView(pixel, type, NS::Range(0, d.levels), slices, swizzle));
                if (!view)
                    return nullptr;
                return std::make_unique<MetalImageView>(std::move(view));
            }

            bool SupportsImage(const ImageDesc& desc) const override
            {
                return ToMtl(desc.format) != MTL::PixelFormatInvalid &&
                    (desc.storeFormat == Format::Undefined || (ToMtl(desc.storeFormat) != MTL::PixelFormatInvalid &&
                        BytesPerPixel(desc.storeFormat) == BytesPerPixel(desc.format)));
            }

            uint32_t FormatFeatures(Format format) const override
            {
                // Apple GPUs blend every colour format but the 32-bit float ones, and store to
                // the ones that aren't packed or compressed.
                const bool wide = format == Format::R32F || format == Format::RG32F || format == Format::RGBA32F;
                return (IsBlock(format) || HasDepth(format) || wide ? 0 : kFormatBlend) |
                    (IsBlock(format) || HasDepth(format) || format == Format::A2B10G10R10 ? 0 : kFormatStorage);
            }

            void GetMemoryUsage(uint64_t& used, uint64_t& budget) const override
            {
                used = m_c->device->currentAllocatedSize();
                budget = m_c->device->recommendedMaxWorkingSetSize();
            }

            ImageMemoryStats GetImageMemoryStats() const override { return {}; }

            std::unique_ptr<Image> CreateImage(const ImageDesc& desc) override
            {
                Pool pool;
                const MTL::PixelFormat format = ToMtl(desc.format);
                if (format == MTL::PixelFormatInvalid)
                    return nullptr;
                auto d = Own(MTL::TextureDescriptor::alloc()->init());
                d->setTextureType(desc.kind == ImageKind::Image3D ? MTL::TextureType3D : desc.kind == ImageKind::Cube ? MTL::TextureTypeCube
                    : desc.layers > 1 ? MTL::TextureType2DArray : MTL::TextureType2D);
                d->setPixelFormat(format);
                d->setWidth(desc.extent.width);
                d->setHeight(desc.extent.height);
                d->setDepth(desc.kind == ImageKind::Image3D ? desc.extent.depth : 1);
                d->setMipmapLevelCount(desc.levels);
                d->setArrayLength(desc.kind == ImageKind::Cube || desc.layers <= 1 ? 1 : desc.layers);
                MTL::TextureUsage usage = 0;
                if (desc.usage & kImageSampled)
                    usage |= MTL::TextureUsageShaderRead;
                if (desc.usage & kImageStorage)
                    usage |= MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite;
                // Clears and scaled copies are render passes here, so what can be copied into can be drawn to.
                if ((desc.usage & (kImageRenderTarget | kImageDepthStencil)) ||
                    ((desc.usage & kImageCopyDst) && !IsBlock(desc.format) && !HasDepth(desc.format)))
                    usage |= MTL::TextureUsageRenderTarget;
                if ((desc.usage & kImageRenderTarget) && (desc.usage & kImageCopySrc))
                    usage |= MTL::TextureUsageShaderRead;  // (a scaled copy from it samples it)
                if ((desc.usage & kImageCopySrc) && !IsBlock(desc.format))
                    usage |= MTL::TextureUsageShaderRead;  // (the same, from any copy source)
                if (desc.storeFormat != Format::Undefined)
                    usage |= MTL::TextureUsagePixelFormatView | MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite;
                d->setUsage(usage);
                d->setStorageMode(MTL::StorageModePrivate);
                Ptr<MTL::Texture> texture = Own(m_c->device->newTexture(d.get()));
                if (!texture)
                    return nullptr;
                if (desc.name && *desc.name)
                    texture->setLabel(NS::String::string(desc.name, NS::UTF8StringEncoding));
                const MTL::SizeAndAlign size = m_c->device->heapTextureSizeAndAlign(d.get());
                return std::make_unique<MetalImage>(desc, std::move(texture), size.size);
            }

            std::unique_ptr<Sampler> CreateSampler(const SamplerDesc& desc) override
            {
                Pool pool;
                auto d = Own(MTL::SamplerDescriptor::alloc()->init());
                d->setMagFilter(ToMtl(desc.mag));
                d->setMinFilter(ToMtl(desc.min));
                d->setMipFilter(desc.mip == Filter::Nearest ? MTL::SamplerMipFilterNearest : MTL::SamplerMipFilterLinear);
                d->setSAddressMode(ToMtl(desc.u));
                d->setTAddressMode(ToMtl(desc.v));
                d->setRAddressMode(ToMtl(desc.w));
                d->setMaxAnisotropy(std::max<NS::UInteger>(1, NS::UInteger(desc.maxAnisotropy)));
                d->setLodMinClamp(desc.minLod);
                d->setLodMaxClamp(desc.maxLod);
                d->setBorderColor(desc.whiteBorder ? MTL::SamplerBorderColorOpaqueWhite : MTL::SamplerBorderColorTransparentBlack);
                return std::make_unique<MetalSampler>(Own(m_c->device->newSamplerState(d.get())));
            }

            std::unique_ptr<BindingLayout> CreateBindingLayout(const BindingLayoutDesc& desc) override
            {
                return std::make_unique<MetalBindingLayout>(desc);
            }
            std::unique_ptr<BindingSet> CreateBindingSet(const BindingLayout& layout, uint32_t table) override
            {
                return std::make_unique<MetalBindingSet>(static_cast<const MetalBindingLayout&>(layout), table);
            }

            std::unique_ptr<Pipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc& desc) override
            {
                Pool pool;
                const auto& layout = *static_cast<const MetalBindingLayout*>(desc.layout);
                std::string vsEntry, fsEntry;
                const std::string vsMsl = ToMsl(desc.vertex, spv::ExecutionModelVertex, layout, vsEntry, nullptr, desc.topology == Topology::PointList);
                const std::string fsMsl = ToMsl(desc.fragment, spv::ExecutionModelFragment, layout, fsEntry, nullptr);
                Ptr<MTL::Function> vs = MakeFunction(*m_c->device.get(), vsMsl, vsEntry), fs = MakeFunction(*m_c->device.get(), fsMsl, fsEntry);
                if (!vs || !fs)
                    return nullptr;
                auto d = Own(MTL::RenderPipelineDescriptor::alloc()->init());
                d->setVertexFunction(vs.get());
                d->setFragmentFunction(fs.get());
                d->setInputPrimitiveTopology(TopologyClass(desc.topology));
                for (int i = 0; i < 4; i++)
                {
                    if (desc.colorFormats[i] == Format::Undefined)
                        continue;
                    const BlendAttachment& b = desc.blend[i];
                    MTL::RenderPipelineColorAttachmentDescriptor* a = d->colorAttachments()->object(i);
                    a->setPixelFormat(ToMtl(desc.colorFormats[i]));
                    a->setBlendingEnabled(b.enable);
                    a->setSourceRGBBlendFactor(ToMtl(b.srcColor));
                    a->setDestinationRGBBlendFactor(ToMtl(b.dstColor));
                    a->setRgbBlendOperation(ToMtl(b.colorOp));
                    a->setSourceAlphaBlendFactor(ToMtl(b.srcAlpha));
                    a->setDestinationAlphaBlendFactor(ToMtl(b.dstAlpha));
                    a->setAlphaBlendOperation(ToMtl(b.alphaOp));
                    // Vulkan's mask is R G B A from the low bit; Metal's is the reverse.
                    MTL::ColorWriteMask mask = 0;
                    if (b.writeMask & 1) mask |= MTL::ColorWriteMaskRed;
                    if (b.writeMask & 2) mask |= MTL::ColorWriteMaskGreen;
                    if (b.writeMask & 4) mask |= MTL::ColorWriteMaskBlue;
                    if (b.writeMask & 8) mask |= MTL::ColorWriteMaskAlpha;
                    a->setWriteMask(mask);
                }
                if (desc.depthFormat != Format::Undefined)
                {
                    d->setDepthAttachmentPixelFormat(ToMtl(desc.depthFormat));
                    if (HasDepth(desc.depthFormat))
                        d->setStencilAttachmentPixelFormat(ToMtl(desc.depthFormat));
                }
                NS::Error* error = nullptr;
                Ptr<MTL::RenderPipelineState> state = Own(m_c->device->newRenderPipelineState(d.get(), &error));
                if (!state)
                {
                    fprintf(stderr, "[rhi/metal] render pipeline failed: %s\n", Describe(error).c_str());
                    return nullptr;
                }
                auto pipeline = std::make_unique<MetalGraphicsPipeline>();
                pipeline->state = std::move(state);
                pipeline->layout = &layout;
                pipeline->desc = desc;
                pipeline->desc.vertex = pipeline->desc.fragment = {};
                pipeline->desc.layout = nullptr;
                pipeline->device = m_c->device.get();
                pipeline->primitive = ToMtl(desc.topology);
                pipeline->cull = desc.cull == CullMode::None ? MTL::CullModeNone : desc.cull == CullMode::Front ? MTL::CullModeFront : MTL::CullModeBack;
                // Vulkan's front face is judged in framebuffer space, y down; with the vertex
                // shaders' Y flipped (ToMsl) that is Metal's, so the sense carries over.
                pipeline->winding = desc.frontCounterClockwise ? MTL::WindingCounterClockwise : MTL::WindingClockwise;
                return pipeline;
            }

            std::unique_ptr<Pipeline> CreateComputePipeline(const ComputePipelineDesc& desc) override
            {
                Pool pool;
                const auto& layout = *static_cast<const MetalBindingLayout*>(desc.layout);
                std::string entry;
                MTL::Size group{};
                const std::string msl = ToMsl(desc.code, spv::ExecutionModelGLCompute, layout, entry, &group);
                Ptr<MTL::Function> function = MakeFunction(*m_c->device.get(), msl, entry);
                if (!function)
                    return nullptr;
                NS::Error* error = nullptr;
                Ptr<MTL::ComputePipelineState> state = Own(m_c->device->newComputePipelineState(function.get(), &error));
                if (!state)
                {
                    fprintf(stderr, "[rhi/metal] compute pipeline failed: %s\n", Describe(error).c_str());
                    return nullptr;
                }
                auto compute = std::make_unique<MetalComputePipeline>(std::move(state), group);
                compute->layout = &layout;
                return compute;
            }

            std::vector<uint8_t> SavePipelineCache() override { return {}; }  // (Metal's own cache keeps what it compiled)
            void LoadPipelineCache(const void*, size_t) override {}

            std::unique_ptr<CommandList> CreateCommandList() override { return std::make_unique<MetalCommandList>(*m_c, m_internal); }
            std::unique_ptr<Fence> CreateFence(bool signaled) override { return std::make_unique<MetalFence>(signaled); }
            std::unique_ptr<Event> CreateEvent() override
            {
                Pool pool;
                return std::make_unique<MetalEvent>(Own(m_c->device->newSharedEvent()));
            }
            std::unique_ptr<QueryPool> CreateQueryPool(QueryKind kind, uint32_t count) override { return std::make_unique<MetalQueryPool>(*m_c, kind, count); }

            std::unique_ptr<Swapchain> CreateSwapchain(Surface& surface, const SwapchainDesc& desc) override
            {
                Pool pool;
                return std::make_unique<MetalSwapchain>(*m_c, static_cast<MetalSurface&>(surface), desc);
            }

            void Submit(CommandList& list, Fence& fence) override
            {
                std::unique_lock lock(m_submitMutex);
                if (m_submitGate)
                    m_submitGate(lock);
                static_cast<MetalCommandList&>(list).Commit(static_cast<MetalFence&>(fence));
            }
            void SetSubmitGate(std::function<void(std::unique_lock<std::mutex>&)> gate) override { m_submitGate = std::move(gate); }
            // The drawable needs nothing from the list: the present that follows waits for it on the queue.
            void SubmitToSwapchain(CommandList& list, Fence& fence, Swapchain&) override { Submit(list, fence); }

            void WaitIdle() override
            {
                Pool pool;
                // Everything committed so far has run when an empty buffer committed after it has.
                MTL::CommandBuffer* cb = m_c->queue->commandBuffer();
                cb->commit();
                cb->waitUntilCompleted();
            }

        private:
            static void SetLabel(MTL::Resource& resource, const char* name)
            {
                if (name && *name)
                    resource.setLabel(NS::String::string(name, NS::UTF8StringEncoding));
            }

            std::unique_ptr<Context> m_c;
            InternalPipelines m_internal;
            std::mutex m_submitMutex;  // commits come one at a time, so a gate can hold the next
            std::function<void(std::unique_lock<std::mutex>&)> m_submitGate;
        };
    }

    std::unique_ptr<Surface> CreateSurface(Device& device, void* caMetalLayer)
    {
        (void)device;
        Pool pool;
        return std::make_unique<MetalSurface>(static_cast<CA::MetalLayer*>(caMetalLayer));
    }

    std::unique_ptr<Device> CreateDevice()
    {
        Pool pool;
        auto c = std::make_unique<Context>();
        c->device = Own(MTL::CreateSystemDefaultDevice());
        if (!c->device)
            return nullptr;
        c->queue = Own(c->device->newCommandQueue());
        if (!c->queue)
            return nullptr;
        for (int i = 0; i < 2; i++)
        {
            auto d = Own(MTL::SamplerDescriptor::alloc()->init());
            const MTL::SamplerMinMagFilter f = i ? MTL::SamplerMinMagFilterLinear : MTL::SamplerMinMagFilterNearest;
            d->setMagFilter(f);
            d->setMinFilter(f);
            d->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
            d->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
            c->blitSamplers[i] = Own(c->device->newSamplerState(d.get()));
        }
        c->deviceName = c->device->name()->utf8String();
        Capabilities& caps = c->caps;
        caps.backend = "Metal";
        caps.deviceName = c->deviceName.c_str();
        caps.unifiedMemory = c->device->hasUnifiedMemory();
        caps.canImportHostMemory = true;
        caps.hostMemoryAlignment = 16384;  // the page size (Apple silicon)
        caps.maxBufferSize = c->device->maxBufferLength();
        caps.textureCompressionBC = c->device->supportsBCTextureCompression();
        c->visibility = Own(c->device->newBuffer(Context::kVisibilitySlots * sizeof(uint64_t), MTL::ResourceStorageModeShared));
        c->scratch = Own(c->device->newBuffer(16, MTL::ResourceStorageModePrivate));
        std::memset(c->visibility->contents(), 0, c->visibility->length());
        caps.occlusionQueries = true;
        if (c->device->supportsCounterSampling(MTL::CounterSamplingPointAtStageBoundary))
        {
            NS::Array* sets = c->device->counterSets();
            for (NS::UInteger i = 0; sets && i < sets->count(); i++)
            {
                auto* set = static_cast<MTL::CounterSet*>(sets->object(i));
                if (set->name()->isEqualToString(MTL::CommonCounterSetTimestamp))
                    c->timestampCounters = Keep(set);
            }
        }
        caps.timestampQueries = bool(c->timestampCounters);
        if (caps.timestampQueries)
        {
            // GPU ticks to nanoseconds: how far the GPU's clock moves against the CPU's.
            MTL::Timestamp cpu0 = 0, gpu0 = 0, cpu1 = 0, gpu1 = 0;
            c->device->sampleTimestamps(&cpu0, &gpu0);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            c->device->sampleTimestamps(&cpu1, &gpu1);
            const double ratio = gpu1 > gpu0 ? double(cpu1 - cpu0) / double(gpu1 - gpu0) : 1.0;
            caps.timestampPeriodNs = ratio;
        }
        caps.fragmentShadingRate = false;
        caps.kind = Capabilities::Kind::Integrated;  // (unified memory)
        caps.vendorId = 0x106B;                      // Apple
        caps.maxStorageBufferRange = c->device->maxBufferLength();
        caps.maxAnisotropy = 16.0f;
        caps.presentTiming = true;      // the drawable's presented handler
        caps.scheduledPresent = true;   // presentDrawable:atTime:
        caps.memoryBudget = c->device->recommendedMaxWorkingSetSize();
        return std::make_unique<MetalDevice>(std::move(c));
    }
}
