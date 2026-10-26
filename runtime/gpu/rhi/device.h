// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// The RHI's objects and the device that makes them. See types.h, and
// command_list.h for what is recorded.
//
// WHERE THE BOUNDARY IS (the questions a reviewer should push on):
//
//  * Memory placement. renderer.cpp today suballocates image memory in 64 MB
//    blocks (BlockRanges) to avoid vkAllocateMemory's cost, and plans where the
//    big buffers go (memory_plan.h). Both are Vulkan-driver problems, so they
//    move INTO the Vulkan backend (CreateImage just returns an image); Metal
//    would use MTLHeap or plain allocation. The plan's choice among memory
//    kinds stays above, as MemoryKind.
//
//  * Guest memory. The renderer reads guest memory through a 512 MB buffer
//    imported without a copy (vertex fetch in the shaders is a storage
//    buffer read, texture loads copy from it). ImportHostMemory is that, with
//    capabilities saying whether the device can and what alignment it needs
//    (Vulkan: external_memory_host, 4 KB; Metal: newBufferWithBytesNoCopy,
//    16 KB page-aligned, the biggest open risk of the Metal backend). When it
//    can't, the renderer's shadow mode (a copy it keeps in sync) applies, as
//    now.
//
//  * Pipelines are built from the shader as SPIR-V (D4): the Metal backend
//    converts it (SPIRV-Cross, MSL, MTLLibrary) inside CreateGraphicsPipeline.
//    Creation is thread-safe and may take tens of ms: the renderer calls it on
//    its worker threads (asynchronous pipelines), and a replay waits for them.
//
//  * Bindings (D10): a pipeline is created against a BindingLayout, which
//    declares tables of slots. A table is either a BindingSet (resources bound
//    once, rebound to the command list rarely: the renderer's "set 0" of guest
//    memory, constants and ring) or pushed per draw (the textures). This is the
//    one place the draft goes a step beyond "set resources at slots": a
//    persistent table as an object, because rewriting 3 bindings per draw
//    would cost on Vulkan and Metal has argument buffers for exactly this.
//
//  * Queue. One queue, shared with the presenter. Submit and Present take the
//    backend's internal lock (VulkanContext::queueMutex goes away) and wait
//    while the app is in the background (presenter's WaitForeground).
//
//  * Fine-grained fences. The renderer signals a VkEvent after the writes of
//    a part of a submission, so the CPU can wait for a texture's resolve
//    without waiting for the whole command buffer. Event/SignalEvent models
//    that; Metal can do it with MTLSharedEvent signalled between encoders at
//    some cost (an encoder break), which is a measurable risk, not a blocker.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "swapchain.h"
#include "types.h"

namespace gpu::rhi
{
    class CommandList;

    // ---- resources ---------------------------------------------------------

    struct BufferDesc
    {
        uint64_t size = 0;
        uint32_t usage = 0;       // BufferUsage
        MemoryKind memory = MemoryKind::Device;
        const char* name = "";    // for debuggers (Xcode, RenderDoc)
    };

    class Buffer
    {
    public:
        virtual ~Buffer() = default;
        virtual uint64_t Size() const = 0;
        // The CPU's pointer for Upload / UploadCached buffers (mapped for life);
        // nullptr for Device ones.
        virtual void* Map() = 0;
        // UploadCached: make the GPU's writes visible to the CPU's reads of
        // [offset, offset + size), after the fence for them.
        virtual void Invalidate(uint64_t offset, uint64_t size) = 0;
    };

    struct ImageDesc
    {
        ImageKind kind = ImageKind::Image2D;
        Format format = Format::RGBA8;
        Extent3D extent;
        uint32_t layers = 1;      // 6 for a cube
        uint32_t levels = 1;
        uint32_t usage = 0;       // ImageUsage
        const char* name = "";
    };

    // An image with its one view over all of it. (Today's CreateImage takes a
    // view type and aspect; the aspect follows the format here. Views in
    // another format, which the renderer uses for a few resolves, are an open
    // item: ImageDesc will gain a list of compatible view formats.)
    class Image
    {
    public:
        virtual ~Image() = default;
        virtual const ImageDesc& Desc() const = 0;
        // Bytes the device reserved for it (the renderer's memory accounting).
        virtual uint64_t Bytes() const = 0;
    };

    struct SamplerDesc
    {
        Filter mag = Filter::Linear, min = Filter::Linear, mip = Filter::Linear;
        AddressMode u = AddressMode::Repeat, v = AddressMode::Repeat, w = AddressMode::Repeat;
        float maxAnisotropy = 1.0f;
    };
    class Sampler { public: virtual ~Sampler() = default; };

    // ---- bindings (D10) ------------------------------------------------------

    enum class BindingKind : uint8_t
    {
        StorageBuffer,
        UniformBufferDynamic,    // a uniform buffer whose offset is given at bind time
        CombinedImageSampler,
        StorageImage,
    };

    struct BindingSlot
    {
        uint32_t index = 0;
        BindingKind kind = BindingKind::StorageBuffer;
        uint8_t stages = 0;      // ShaderStage bits
        // CombinedImageSampler: a sampler fixed in the layout (the renderer
        // uses one for every texture). Metal allows 16 samplers a stage but
        // 128 textures: the backend binds this one sampler once, rather than
        // one per texture the shader's combined image-samplers would become.
        // nullptr: ImageBinding::sampler is given at bind time.
        const Sampler* immutableSampler = nullptr;
    };

    // A table of slots: persistent (a BindingSet) or pushed per draw.
    struct BindingTableDesc
    {
        std::vector<BindingSlot> slots;
        bool pushed = false;
    };

    struct BindingLayoutDesc
    {
        std::vector<BindingTableDesc> tables;   // table i is bound at table index i
        uint32_t pushConstantBytes = 0;         // both graphics stages see them
    };
    class BindingLayout { public: virtual ~BindingLayout() = default; };

    struct BufferBinding { Buffer* buffer = nullptr; uint64_t offset = 0, size = ~0ull; };
    struct ImageBinding { Image* image = nullptr; Sampler* sampler = nullptr; };
    struct BindingEntry
    {
        uint32_t slot = 0;
        BufferBinding buffer;    // StorageBuffer, UniformBufferDynamic
        ImageBinding image;      // CombinedImageSampler, StorageImage
    };

    // The resources of one persistent table.
    class BindingSet
    {
    public:
        virtual ~BindingSet() = default;
        virtual void Update(const BindingEntry* entries, size_t count) = 0;
    };

    // ---- pipelines -----------------------------------------------------------

    // The shader as SPIR-V words (what glslang makes from the translator's or
    // the presenter's GLSL).
    struct ShaderCode { const uint32_t* spirv = nullptr; size_t words = 0; };

    struct BlendAttachment
    {
        bool enable = false;
        BlendFactor srcColor = BlendFactor::One, dstColor = BlendFactor::Zero;
        BlendOp colorOp = BlendOp::Add;
        BlendFactor srcAlpha = BlendFactor::One, dstAlpha = BlendFactor::Zero;
        BlendOp alphaOp = BlendOp::Add;
        uint8_t writeMask = 0xF;
    };

    struct StencilFace
    {
        StencilOp fail = StencilOp::Keep, pass = StencilOp::Keep, depthFail = StencilOp::Keep;
        CompareOp compare = CompareOp::Always;
    };

    struct GraphicsPipelineDesc
    {
        ShaderCode vertex, fragment;
        const BindingLayout* layout = nullptr;
        Topology topology = Topology::TriangleList;
        bool primitiveRestart = false;
        Format colorFormats[4] = {};          // Undefined: not attached
        Format depthFormat = Format::Undefined;
        BlendAttachment blend[4];
        bool depthTest = false, depthWrite = false;
        CompareOp depthCompare = CompareOp::Always;
        bool stencilTest = false;
        StencilFace stencilFront, stencilBack;
        CullMode cull = CullMode::None;
        bool frontCounterClockwise = false;
        bool depthClamp = false;
        bool depthBias = false;
        // Set by CommandList (viewport, scissor, stencil values, depth bias,
        // blend constants): dynamic state, as today.
    };

    struct ComputePipelineDesc
    {
        ShaderCode code;
        const BindingLayout* layout = nullptr;
    };

    class Pipeline { public: virtual ~Pipeline() = default; };

    // ---- synchronisation -----------------------------------------------------

    // A fence for one submission: the CPU waits on it or polls it.
    class Fence
    {
    public:
        virtual ~Fence() = default;
        virtual bool Signaled() = 0;
        virtual void Wait() = 0;
        virtual void Reset() = 0;
    };

    // Signalled from inside a command list (CommandList::SignalEvent), seen
    // by the CPU as soon as the GPU reaches that point, before the submission
    // ends (the renderer's write fences).
    class Event
    {
    public:
        virtual ~Event() = default;
        virtual bool Signaled() = 0;
        virtual void Reset() = 0;
    };

    // Timestamps and occlusion counts.
    enum class QueryKind : uint8_t { Timestamp, Occlusion };
    class QueryPool
    {
    public:
        virtual ~QueryPool() = default;
        // The results of [first, first + count) once their submission is done.
        virtual bool Read(uint32_t first, uint32_t count, uint64_t* out) = 0;
    };

    // ---- the device ----------------------------------------------------------

    struct Capabilities
    {
        const char* backend = "";            // "Vulkan", "Metal"
        const char* deviceName = "";
        bool unifiedMemory = false;
        bool canImportHostMemory = false;    // ImportHostMemory works
        uint64_t hostMemoryAlignment = 4096; // its pointer and size must be a multiple of this
        uint64_t maxBufferSize = 0;
        bool textureCompressionBC = true;
        bool occlusionQueries = false;       // precise counting
        bool timestampQueries = false;
        double timestampPeriodNs = 1.0;      // nanoseconds per timestamp tick
        bool fragmentShadingRate = false;
        bool presentTiming = false;          // PresentShown is delivered (VK_KHR_present_wait; Metal always)
        bool scheduledPresent = false;       // PresentRequest::notBefore is honoured (display timing; Metal)
        uint64_t memoryBudget = 0;           // bytes, 0 unknown
    };

    class Device
    {
    public:
        virtual ~Device() = default;
        virtual const Capabilities& GetCapabilities() const = 0;

        virtual std::unique_ptr<Buffer> CreateBuffer(const BufferDesc& desc) = 0;
        // `host` (aligned as Capabilities says) as a buffer without a copy.
        virtual std::unique_ptr<Buffer> ImportHostMemory(void* host, uint64_t size, uint32_t usage, const char* name) = 0;
        // nullptr when the device turns the image down (the renderer tolerates it for scaled targets).
        virtual std::unique_ptr<Image> CreateImage(const ImageDesc& desc) = 0;
        virtual std::unique_ptr<Sampler> CreateSampler(const SamplerDesc& desc) = 0;

        virtual std::unique_ptr<BindingLayout> CreateBindingLayout(const BindingLayoutDesc& desc) = 0;
        virtual std::unique_ptr<BindingSet> CreateBindingSet(const BindingLayout& layout, uint32_t table) = 0;
        // Thread-safe. nullptr on failure.
        virtual std::unique_ptr<Pipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc& desc) = 0;
        virtual std::unique_ptr<Pipeline> CreateComputePipeline(const ComputePipelineDesc& desc) = 0;
        // The compiled-pipeline cache on disk (today's VkPipelineCache file).
        virtual std::vector<uint8_t> SavePipelineCache() = 0;
        virtual void LoadPipelineCache(const void* data, size_t size) = 0;

        virtual std::unique_ptr<CommandList> CreateCommandList() = 0;
        // `signaled`: starts signalled (a frame slot's first wait returns at once).
        virtual std::unique_ptr<Fence> CreateFence(bool signaled = false) = 0;
        virtual std::unique_ptr<Event> CreateEvent() = 0;
        virtual std::unique_ptr<QueryPool> CreateQueryPool(QueryKind kind, uint32_t count) = 0;

        // The window (swapchain.h). The Surface comes from the backend's
        // native header (a VkSurfaceKHR, a CAMetalLayer). nullptr on failure.
        virtual std::unique_ptr<Swapchain> CreateSwapchain(Surface& surface, const SwapchainDesc& desc) = 0;

        // The queue. `fence` is signalled when the list has finished.
        virtual void Submit(CommandList& list, Fence& fence) = 0;
        // The same for the list that draws the image acquired from
        // `swapchain`: it waits for the image to be free (Vulkan: the acquire
        // semaphore) and what Present shows waits for it (the render-done one).
        virtual void SubmitToSwapchain(CommandList& list, Fence& fence, Swapchain& swapchain) = 0;
        virtual void WaitIdle() = 0;
    };
}
