// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// The Metal backend's private types, shared by its source files
// (metal_device.cpp, metal_commands.cpp, metal_swapchain.cpp,
// metal_shaders.cpp). How the RHI maps onto Metal is in metal_device.cpp.
// Nothing outside gpu/rhi/metal includes this.
#pragma once
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <spirv_cross/spirv.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../command_list.h"
#include "../swapchain.h"

namespace gpu::rhi::metal
{
    namespace impl
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

        [[noreturn]] inline void Fail(const char* what)
        {
            fprintf(stderr, "[rhi/metal] %s\n", what);
            abort();
        }

        inline std::string Describe(const NS::Error* error)
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

        inline MTL::PixelFormat ToMtl(Format f)
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

        inline MTL::CompareFunction ToMtl(CompareOp op)
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

        inline MTL::StencilOperation ToMtl(StencilOp op)
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

        inline MTL::BlendFactor ToMtl(BlendFactor f)
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

        inline MTL::BlendOperation ToMtl(BlendOp op)
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

        inline MTL::PrimitiveType ToMtl(Topology t)
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

        inline MTL::PrimitiveTopologyClass TopologyClass(Topology t)
        {
            switch (t)
            {
            case Topology::PointList: return MTL::PrimitiveTopologyClassPoint;
            case Topology::LineList: case Topology::LineStrip: return MTL::PrimitiveTopologyClassLine;
            default: return MTL::PrimitiveTopologyClassTriangle;
            }
        }

        inline MTL::SamplerMinMagFilter ToMtl(Filter f) { return f == Filter::Nearest ? MTL::SamplerMinMagFilterNearest : MTL::SamplerMinMagFilterLinear; }

        inline MTL::SamplerAddressMode ToMtl(AddressMode a)
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

        // ---- shaders (metal_shaders.cpp) ---------------------------------------------

        // MSL for `code` (SPIR-V) in the binding layout's slot numbering; `entry` gets the entry point's
        // name in it, `size` (compute) the workgroup size.
        std::string ToMsl(const ShaderCode& code, spv::ExecutionModel model, const MetalBindingLayout& layout, std::string& entry,
            MTL::Size* size, bool pointSize = false);
        // A function of `msl` named `entry`, or an empty pointer (logged).
        Ptr<MTL::Function> MakeFunction(MTL::Device& device, const std::string& msl, const std::string& entry);

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

        // ---- the other source files ----------------------------------------------------

        // metal_commands.cpp: a command list, and the submit of one (its command buffer commits with the fence).
        std::unique_ptr<CommandList> MakeCommandList(Context& c, InternalPipelines& internal);
        void CommitCommandList(CommandList& list, MetalFence& fence);
        // metal_swapchain.cpp
        std::unique_ptr<Swapchain> MakeSwapchain(Context& c, Surface& surface, const SwapchainDesc& desc);
    }
}
