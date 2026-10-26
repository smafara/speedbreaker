// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
#include "metal_device.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <spirv_cross/spirv_msl.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "../command_list.h"

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
        std::string ToMsl(const ShaderCode& code, spv::ExecutionModel model, const MetalBindingLayout& layout, std::string& entry, MTL::Size* size)
        {
            spirv_cross::CompilerMSL compiler(code.spirv, code.words);
            spirv_cross::CompilerMSL::Options options;
            options.platform = spirv_cross::CompilerMSL::Options::macOS;
            options.set_msl_version(3, 0);
            options.argument_buffers = false;
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

            // The depth-stencil state for the stencil masks set now. Metal keeps
            // them in the state object, Vulkan makes them dynamic.
            MTL::DepthStencilState* DepthStencil(uint8_t compareMask, uint8_t writeMask)
            {
                const uint16_t key = uint16_t(compareMask) << 8 | writeMask;
                std::lock_guard lock(m_mutex);
                auto it = m_states.find(key);
                if (it != m_states.end())
                    return it->second.get();
                auto d = Own(MTL::DepthStencilDescriptor::alloc()->init());
                d->setDepthCompareFunction(desc.depthTest ? ToMtl(desc.depthCompare) : MTL::CompareFunctionAlways);
                d->setDepthWriteEnabled(desc.depthTest && desc.depthWrite);
                if (!desc.depthTest && desc.depthWrite)
                    d->setDepthWriteEnabled(true);  // (Vulkan: writes need no test; Always passes)
                if (desc.stencilTest)
                {
                    auto face = [&](const StencilFace& f) {
                        auto s = Own(MTL::StencilDescriptor::alloc()->init());
                        s->setStencilFailureOperation(ToMtl(f.fail));
                        s->setDepthStencilPassOperation(ToMtl(f.pass));
                        s->setDepthFailureOperation(ToMtl(f.depthFail));
                        s->setStencilCompareFunction(ToMtl(f.compare));
                        s->setReadMask(compareMask);
                        s->setWriteMask(writeMask);
                        return s;
                    };
                    auto front = face(desc.stencilFront), back = face(desc.stencilBack);
                    d->setFrontFaceStencil(front.get());
                    d->setBackFaceStencil(back.get());
                }
                Ptr<MTL::DepthStencilState> state = Own(device->newDepthStencilState(d.get()));
                MTL::DepthStencilState* raw = state.get();
                m_states.emplace(key, std::move(state));
                return raw;
            }

        private:
            std::mutex m_mutex;
            std::map<uint16_t, Ptr<MTL::DepthStencilState>> m_states;
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
            void Wait() override
            {
                std::unique_lock lock(m_state->mutex);
                m_state->cv.wait(lock, [&] { return m_state->signaled; });
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
            MetalQueryPool(QueryKind kind, uint32_t count) : m_kind(kind), m_count(count) {}
            bool Read(uint32_t, uint32_t, uint64_t*) override { return false; }  // (queries: not yet)

        private:
            QueryKind m_kind;
            uint32_t m_count;
        };

        // ---- command lists -------------------------------------------------------------

        class MetalCommandList final : public CommandList
        {
        public:
            explicit MetalCommandList(Context& c) : m_c(c) {}
            ~MetalCommandList() override
            {
                EndEncoder();
                m_commands.reset();
            }

            // Ends the recording and commits it; `fence` is signalled when it has run.
            void Commit(MetalFence& fence)
            {
                if (m_commands->status() != MTL::CommandBufferStatusNotEnqueued && m_commands->status() != MTL::CommandBufferStatusEnqueued)
                    Fail("a command list was submitted twice");
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
                m_pipeline = nullptr;
                m_commands = Keep(m_c.queue->commandBuffer());
            }

            void End() override { EndEncoder(); }

            // ---- passes ----
            void BeginRendering(const RenderingDesc&) override { Fail("BeginRendering: not yet"); }
            void EndRendering() override { Fail("EndRendering: not yet"); }
            void ClearAttachments(const Rect2D&, const ClearColor*, uint32_t, const ClearDepthStencil*, bool, bool) override { Fail("ClearAttachments: not yet"); }

            // ---- state ----
            void SetPipeline(const Pipeline& pipeline) override
            {
                m_pipeline = &static_cast<const MetalPipeline&>(pipeline);
                if (m_compute && m_pipeline->IsCompute())
                    ApplyPipeline();
            }
            void SetViewport(float, float, float, float, float, float) override { Fail("SetViewport: not yet"); }
            void SetScissor(const Rect2D&) override { Fail("SetScissor: not yet"); }
            void SetStencil(uint8_t, uint8_t, uint8_t) override { Fail("SetStencil: not yet"); }
            void SetDepthBias(float, float, float) override { Fail("SetDepthBias: not yet"); }
            void SetBlendConstants(const float[4]) override { Fail("SetBlendConstants: not yet"); }

            // ---- bindings ----
            void BindSet(const BindingLayout& layout, uint32_t table, BindingSet& set, const uint32_t* dynamicOffsets, uint32_t dynamicCount) override
            {
                Bound& b = m_sets.at(table);
                b.layout = &static_cast<const MetalBindingLayout&>(layout);
                b.set = &static_cast<MetalBindingSet&>(set);
                b.dynamic.assign(dynamicOffsets, dynamicOffsets + dynamicCount);
                m_pushed.at(table).entries.clear();
                m_pushed.at(table).layout = nullptr;
                if (m_compute)
                    ApplyTable(table);
            }
            void PushBindings(const BindingLayout& layout, uint32_t table, const BindingEntry* entries, size_t count) override
            {
                Pushed& p = m_pushed.at(table);
                p.layout = &static_cast<const MetalBindingLayout&>(layout);
                p.entries.assign(entries, entries + count);
                m_sets.at(table).set = nullptr;
                if (m_compute)
                    ApplyTable(table);
            }
            void PushConstants(const BindingLayout& layout, const void* data, uint32_t size) override
            {
                if (size > sizeof(m_constants))
                    Fail("push constants beyond 256 bytes");
                m_constantsLayout = &static_cast<const MetalBindingLayout&>(layout);
                memcpy(m_constants, data, size);
                m_constantsSize = size;
                if (m_compute)
                    m_compute->setBytes(m_constants, m_constantsSize, m_constantsLayout->PushConstantBuffer());
            }

            // ---- work ----
            void Draw(uint32_t, uint32_t, uint32_t, uint32_t) override { Fail("Draw: not yet"); }
            void DrawIndexed(uint32_t, uint32_t, uint32_t, int32_t, uint32_t) override { Fail("DrawIndexed: not yet"); }
            void BindIndexBuffer(Buffer&, uint64_t, bool) override { Fail("BindIndexBuffer: not yet"); }
            void Dispatch(uint32_t x, uint32_t y, uint32_t z) override
            {
                BeginCompute();
                m_compute->dispatchThreadgroups(MTL::Size::Make(x, y, z), static_cast<const MetalComputePipeline*>(m_pipeline)->Group());
            }

            // ---- transfers ----
            void CopyBuffer(Buffer& src, Buffer& dst, uint64_t srcOffset, uint64_t dstOffset, uint64_t size) override
            {
                BeginBlit()->copyFromBuffer(static_cast<MetalBuffer&>(src).Handle(), srcOffset, static_cast<MetalBuffer&>(dst).Handle(), dstOffset, size);
            }
            void CopyBufferToImage(Buffer&, Image&, const BufferImageCopy*, uint32_t) override { Fail("CopyBufferToImage: not yet"); }
            void CopyImageToBuffer(Image&, Buffer&, const BufferImageCopy*, uint32_t) override { Fail("CopyImageToBuffer: not yet"); }
            void BlitImage(Image&, const Rect2D&, Image&, const Rect2D&, Filter) override { Fail("BlitImage: not yet"); }
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
            void ClearImage(Image&, const ClearColor&) override { Fail("ClearImage: not yet"); }
            void ClearImage(Image&, const ClearDepthStencil&) override { Fail("ClearImage: not yet"); }

            // ---- synchronisation ----
            void Transition(const ImageTransition*, size_t, const BufferTransition*, size_t) override {}  // Metal tracks the hazards
            void SignalEvent(Event& event) override
            {
                EndEncoder();
                auto& e = static_cast<MetalEvent&>(event);
                m_commands->encodeSignalEvent(e.Handle(), e.Next());
            }

            // ---- queries ----
            void ResetQueries(QueryPool&, uint32_t, uint32_t) override {}
            void WriteTimestamp(QueryPool&, uint32_t, bool) override {}
            void BeginQuery(QueryPool&, uint32_t) override { Fail("BeginQuery: not yet"); }
            void EndQuery(QueryPool&, uint32_t) override { Fail("EndQuery: not yet"); }

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
                if (m_pipeline && m_pipeline->IsCompute())
                    ApplyPipeline();
                for (uint32_t t = 0; t < m_sets.size(); t++)
                    ApplyTable(t);
                if (m_constantsSize)
                    m_compute->setBytes(m_constants, m_constantsSize, m_constantsLayout->PushConstantBuffer());
            }

            void ApplyPipeline() { m_compute->setComputePipelineState(static_cast<const MetalComputePipeline*>(m_pipeline)->Handle()); }

            // Table `table` as last bound (a set or pushed entries) on the compute encoder.
            void ApplyTable(uint32_t table)
            {
                const Bound& b = m_sets[table];
                const Pushed& p = m_pushed[table];
                if (b.set)
                {
                    ApplyEntries(*b.layout, table, b.set->Entries().data(), b.set->Entries().size(), b.dynamic);
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
                    switch (m->slot.kind)
                    {
                    case BindingKind::StorageBuffer:
                    case BindingKind::UniformBufferDynamic:
                    {
                        uint64_t offset = entries[i].buffer.offset;
                        if (m->dynamicOrder >= 0 && size_t(m->dynamicOrder) < dynamic.size())
                            offset += dynamic[m->dynamicOrder];
                        m_compute->setBuffer(static_cast<MetalBuffer*>(entries[i].buffer.buffer)->Handle(), offset, m->buffer);
                        break;
                    }
                    case BindingKind::CombinedImageSampler:
                    case BindingKind::StorageImage:
                        m_compute->setTexture(static_cast<MetalImage*>(entries[i].image.image)->Handle(), m->texture);
                        if (m->slot.kind == BindingKind::CombinedImageSampler)
                        {
                            const Sampler* sampler = m->slot.immutableSampler ? m->slot.immutableSampler : entries[i].image.sampler;
                            m_compute->setSamplerState(static_cast<const MetalSampler*>(sampler)->Handle(), m->sampler);
                        }
                        break;
                    }
                }
            }

            Context& m_c;
            Ptr<MTL::CommandBuffer> m_commands;
            Ptr<MTL::BlitCommandEncoder> m_blit;
            Ptr<MTL::ComputeCommandEncoder> m_compute;
            const MetalPipeline* m_pipeline = nullptr;
            std::array<Bound, 4> m_sets;
            std::array<Pushed, 4> m_pushed;
            const MetalBindingLayout* m_constantsLayout = nullptr;
            uint8_t m_constants[256] = {};
            uint32_t m_constantsSize = 0;
            std::vector<Ptr<MTL::Buffer>> m_temps;  // until the list is recorded again
        };

        // ---- the device ----------------------------------------------------------------

        class MetalDevice final : public Device
        {
        public:
            explicit MetalDevice(std::unique_ptr<Context> context) : m_c(std::move(context)) {}

            const Capabilities& GetCapabilities() const override { return m_c->caps; }

            std::unique_ptr<Buffer> CreateBuffer(const BufferDesc& desc) override
            {
                Pool pool;
                MTL::ResourceOptions options = desc.memory == MemoryKind::Device ? MTL::ResourceStorageModePrivate
                    : desc.memory == MemoryKind::Upload ? (MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined)
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

            std::unique_ptr<Image> CreateImage(const ImageDesc&) override { return nullptr; }  // (images: not yet)

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

            std::unique_ptr<Pipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc&) override { return nullptr; }  // (graphics: not yet)

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
                return std::make_unique<MetalComputePipeline>(std::move(state), group);
            }

            std::vector<uint8_t> SavePipelineCache() override { return {}; }  // (Metal's own cache keeps what it compiled)
            void LoadPipelineCache(const void*, size_t) override {}

            std::unique_ptr<CommandList> CreateCommandList() override { return std::make_unique<MetalCommandList>(*m_c); }
            std::unique_ptr<Fence> CreateFence(bool signaled) override { return std::make_unique<MetalFence>(signaled); }
            std::unique_ptr<Event> CreateEvent() override
            {
                Pool pool;
                return std::make_unique<MetalEvent>(Own(m_c->device->newSharedEvent()));
            }
            std::unique_ptr<QueryPool> CreateQueryPool(QueryKind kind, uint32_t count) override { return std::make_unique<MetalQueryPool>(kind, count); }

            std::unique_ptr<Swapchain> CreateSwapchain(Surface&, const SwapchainDesc&) override { return nullptr; }  // (the window: not yet)

            void Submit(CommandList& list, Fence& fence) override
            {
                static_cast<MetalCommandList&>(list).Commit(static_cast<MetalFence&>(fence));
            }
            void SubmitToSwapchain(CommandList&, Fence&, Swapchain&) override { Fail("SubmitToSwapchain: not yet"); }

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
        };
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
        c->deviceName = c->device->name()->utf8String();
        Capabilities& caps = c->caps;
        caps.backend = "Metal";
        caps.deviceName = c->deviceName.c_str();
        caps.unifiedMemory = c->device->hasUnifiedMemory();
        caps.canImportHostMemory = true;
        caps.hostMemoryAlignment = 16384;  // the page size (Apple silicon)
        caps.maxBufferSize = c->device->maxBufferLength();
        caps.textureCompressionBC = c->device->supportsBCTextureCompression();
        caps.occlusionQueries = false;
        caps.timestampQueries = false;
        caps.fragmentShadingRate = false;
        caps.presentTiming = false;
        caps.scheduledPresent = false;
        caps.memoryBudget = c->device->recommendedMaxWorkingSetSize();
        return std::make_unique<MetalDevice>(std::move(c));
    }
}
