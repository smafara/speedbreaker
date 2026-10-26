// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
#include "metal_device.h"

#include "metal_internal.h"
#include "metal_native.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>


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
    using namespace impl;

    namespace
    {
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
                // (Linear filtering: said of the depth formats only, the one thing asked so far.)
                return (IsBlock(format) || HasDepth(format) || wide ? 0 : kFormatBlend) |
                    (IsBlock(format) || HasDepth(format) || format == Format::A2B10G10R10 ? 0 : kFormatStorage) |
                    (HasDepth(format) ? kFormatSampleLinear : 0);
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

            std::unique_ptr<CommandList> CreateCommandList() override { return MakeCommandList(*m_c, m_internal); }
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
                return MakeSwapchain(*m_c, surface, desc);
            }

            void Submit(CommandList& list, Fence& fence) override
            {
                std::unique_lock lock(m_submitMutex);
                if (m_submitGate)
                    m_submitGate(lock);
                CommitCommandList(list, static_cast<MetalFence&>(fence));
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

