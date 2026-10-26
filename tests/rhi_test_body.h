// Tests for the RHI's backends (runtime/gpu/rhi/vulkan, runtime/gpu/rhi/metal): the
// same tests on each, real calls on the machine's device, headless. The includer
// (rhi_vulkan_test.cpp, rhi_metal_test.cpp) defines RHI_BACKEND_NAME and
// MakeDevice() before including this file. Each test is one of the concepts
// the renderer will lean on (DECISIONS.md, presenter refactor step 1): copies
// and barriers on buffers, importing host memory without a copy, a binding
// set with a dynamic uniform offset, pushed bindings, push constants, events
// signalled from inside a command list, timestamps, and the graphics half:
// images, rendering with a depth test, blending and a partial clear in a
// pass, a sampled texture through a pushed binding with an immutable
// sampler, and clears and copies of images. No device
// (CI without one) is a skip, not a failure. Build with the tests (-DSB_TESTS=ON).
#include <gpu/rhi/command_list.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include <sys/mman.h>

#include "rhi_test_shaders.h"

using namespace gpu::rhi;

static int g_failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_failures++; printf("  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } } while (0)

namespace
{
    constexpr uint32_t kCount = 1024;  // words in the buffers; 16 groups of 64

    std::unique_ptr<Device> g_device;

    // Records `record` into a list, submits it and waits for its fence.
    template<typename F>
    void Run(F record)
    {
        auto list = g_device->CreateCommandList();
        auto fence = g_device->CreateFence();
        list->Begin();
        record(*list);
        list->End();
        g_device->Submit(*list, *fence);
        fence->Wait();
    }

    std::unique_ptr<Buffer> Make(uint64_t size, MemoryKind memory, uint32_t usage)
    {
        BufferDesc desc;
        desc.size = size;
        desc.memory = memory;
        desc.usage = usage;
        return g_device->CreateBuffer(desc);
    }

    ShaderCode Code(const uint32_t* words, size_t count) { return { words, count }; }

    uint32_t* Words(Buffer& b) { return static_cast<uint32_t*>(b.Map()); }

    // Layout of {0: read-only storage, 1: storage}, a persistent table, and a push constant.
    std::unique_ptr<BindingLayout> InOutLayout()
    {
        BindingLayoutDesc desc;
        desc.tables.push_back({ { { 0, BindingKind::StorageBuffer, uint8_t(ShaderStage::Compute) },
                                  { 1, BindingKind::StorageBuffer, uint8_t(ShaderStage::Compute) } }, false });
        desc.pushConstantBytes = 4;
        return g_device->CreateBindingLayout(desc);
    }

    BindingEntry Buf(uint32_t slot, Buffer& b, uint64_t offset = 0, uint64_t size = ~0ull)
    {
        BindingEntry e;
        e.slot = slot;
        e.buffer = { &b, offset, size };
        return e;
    }

    void Capabilities_()
    {
        const Capabilities& c = g_device->GetCapabilities();
        printf("  %s on %s: unified memory %d, host import %d (alignment %llu), timestamps %d, occlusion %d\n", c.backend, c.deviceName,
            c.unifiedMemory, c.canImportHostMemory, (unsigned long long)c.hostMemoryAlignment, c.timestampQueries, c.occlusionQueries);
        CHECK(std::string(c.backend) == RHI_BACKEND_NAME, "backend name");
    }

    // Copies and barriers: host-visible upload -> device-local -> cached readback.
    void BufferCopies()
    {
        auto up = Make(kCount * 4, MemoryKind::Upload, kBufferCopySrc);
        auto dev = Make(kCount * 4, MemoryKind::Device, kBufferCopySrc | kBufferCopyDst);
        auto back = Make(kCount * 4, MemoryKind::UploadCached, kBufferCopyDst);
        CHECK(up && dev && back, "buffers created");
        CHECK(up->Map() != nullptr && dev->Map() == nullptr && back->Map() != nullptr, "Map: host-visible kinds only");
        for (uint32_t i = 0; i < kCount; i++)
            Words(*up)[i] = i * 2654435761u;
        Run([&](CommandList& cmd) {
            cmd.CopyBuffer(*up, *dev, 0, 0, kCount * 4);
            BufferTransition t[] = { { dev.get(), State::CopyDst, State::CopySrc } };
            cmd.Transition(nullptr, 0, t, 1);
            cmd.CopyBuffer(*dev, *back, 0, 0, kCount * 4);
            BufferTransition h[] = { { back.get(), State::CopyDst, State::HostRead } };
            cmd.Transition(nullptr, 0, h, 1);
        });
        back->Invalidate(0, kCount * 4);
        CHECK(memcmp(Words(*up), Words(*back), kCount * 4) == 0, "the data came back unchanged");
    }

    // Guest memory's way in: host memory used by the GPU in place (no copy).
    void HostImport()
    {
        const Capabilities& caps = g_device->GetCapabilities();
        if (!caps.canImportHostMemory)
        {
            printf("  host import: not offered by this device (skipped)\n");
            return;
        }
        const uint64_t align = std::max<uint64_t>(caps.hostMemoryAlignment, 16384);
        const uint64_t size = align * 4;
        void* host = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        CHECK(host != MAP_FAILED && reinterpret_cast<uintptr_t>(host) % caps.hostMemoryAlignment == 0, "aligned host memory");
        auto* words = static_cast<uint32_t*>(host);
        for (uint32_t i = 0; i < kCount; i++)
            words[i] = i;

        CHECK(g_device->ImportHostMemory(static_cast<char*>(host) + 4, size - 4, kBufferStorage, "misaligned") == nullptr,
            "a misaligned import is refused");
        auto shared = g_device->ImportHostMemory(host, size, kBufferStorage | kBufferCopySrc, "test");
        CHECK(shared != nullptr, "imported");
        if (!shared)
            return;
        CHECK(shared->Map() == host && shared->Size() == size, "the import is the host memory");

        auto out = Make(kCount * 4, MemoryKind::Upload, kBufferStorage);
        auto layout = InOutLayout();
        auto set = g_device->CreateBindingSet(*layout, 0);
        BindingEntry entries[] = { Buf(0, *shared, 0, kCount * 4), Buf(1, *out) };
        set->Update(entries, 2);
        ComputePipelineDesc pd;
        pd.code = Code(k_add, sizeof(k_add) / 4);
        pd.layout = layout.get();
        auto pipeline = g_device->CreateComputePipeline(pd);
        CHECK(pipeline != nullptr, "compute pipeline");
        if (!pipeline)
            return;
        auto dispatch = [&](uint32_t add) {
            Run([&](CommandList& cmd) {
                cmd.SetPipeline(*pipeline);
                cmd.BindSet(*layout, 0, *set);
                cmd.PushConstants(*layout, &add, 4);
                cmd.Dispatch(kCount / 64, 1, 1);
                BufferTransition t[] = { { out.get(), State::ShaderWrite, State::HostRead } };
                cmd.Transition(nullptr, 0, t, 1);
            });
        };
        dispatch(7);
        bool ok = true;
        for (uint32_t i = 0; i < kCount; i++)
            ok &= Words(*out)[i] == i + 7;
        CHECK(ok, "the shader read the host memory (+ the push constant)");
        // No copy: what the CPU writes afterwards is what the GPU reads next.
        for (uint32_t i = 0; i < kCount; i++)
            words[i] = i * 3;
        dispatch(1);
        ok = true;
        for (uint32_t i = 0; i < kCount; i++)
            ok &= Words(*out)[i] == i * 3 + 1;
        CHECK(ok, "the CPU's later writes are seen without a new import");
        shared.reset();
        munmap(host, size);
    }

    // The renderer's "set 0": a persistent table with a dynamic uniform offset.
    void DynamicOffsets()
    {
        BindingLayoutDesc desc;
        desc.tables.push_back({ { { 0, BindingKind::UniformBufferDynamic, uint8_t(ShaderStage::Compute) },
                                  { 1, BindingKind::StorageBuffer, uint8_t(ShaderStage::Compute) } }, false });
        auto layout = g_device->CreateBindingLayout(desc);
        auto uniforms = Make(1024, MemoryKind::Upload, kBufferUniform);
        auto outA = Make(kCount * 4, MemoryKind::Upload, kBufferStorage);
        auto outB = Make(kCount * 4, MemoryKind::Upload, kBufferStorage);
        auto* u = static_cast<uint32_t*>(uniforms->Map());
        u[0] = 100;           // offset 0
        u[256 / 4] = 5000;    // offset 256 (a multiple of any device's alignment)
        auto set = g_device->CreateBindingSet(*layout, 0);
        BindingEntry entries[] = { Buf(0, *uniforms, 0, 16), Buf(1, *outA) };
        set->Update(entries, 2);
        ComputePipelineDesc pd;
        pd.code = Code(k_dyn, sizeof(k_dyn) / 4);
        pd.layout = layout.get();
        auto pipeline = g_device->CreateComputePipeline(pd);
        CHECK(pipeline != nullptr, "compute pipeline");
        if (!pipeline)
            return;
        auto run = [&](uint32_t offset) {
            Run([&](CommandList& cmd) {
                cmd.SetPipeline(*pipeline);
                cmd.BindSet(*layout, 0, *set, &offset, 1);
                cmd.Dispatch(kCount / 64, 1, 1);
                BufferTransition t[] = { { outA.get(), State::ShaderWrite, State::HostRead }, { outB.get(), State::ShaderWrite, State::HostRead } };
                cmd.Transition(nullptr, 0, t, 2);
            });
        };
        run(0);
        bool ok = true;
        for (uint32_t i = 0; i < kCount; i++)
            ok &= Words(*outA)[i] == 100 + i;
        CHECK(ok, "dynamic offset 0 reads the first uniform");
        run(256);
        ok = true;
        for (uint32_t i = 0; i < kCount; i++)
            ok &= Words(*outA)[i] == 5000 + i;
        CHECK(ok, "dynamic offset 256 reads the second, with the same set");
        // The set updated: the same shader now writes the other buffer.
        BindingEntry again[] = { Buf(1, *outB) };
        set->Update(again, 1);
        run(0);
        ok = true;
        for (uint32_t i = 0; i < kCount; i++)
            ok &= Words(*outB)[i] == 100 + i && Words(*outA)[i] == 5000 + i;
        CHECK(ok, "BindingSet::Update retargets one slot and leaves the other buffer alone");
    }

    // A persistent table plus a pushed one (the renderer's set 0 and its per-draw textures).
    void PushedBindings()
    {
        BindingLayoutDesc desc;
        desc.tables.push_back({ { { 0, BindingKind::StorageBuffer, uint8_t(ShaderStage::Compute) } }, false });
        desc.tables.push_back({ { { 0, BindingKind::StorageBuffer, uint8_t(ShaderStage::Compute) } }, true });
        auto layout = g_device->CreateBindingLayout(desc);
        auto in = Make(kCount * 4, MemoryKind::Upload, kBufferStorage);
        auto outA = Make(kCount * 4, MemoryKind::Upload, kBufferStorage);
        auto outB = Make(kCount * 4, MemoryKind::Upload, kBufferStorage);
        for (uint32_t i = 0; i < kCount; i++)
            Words(*in)[i] = i + 1;
        auto set = g_device->CreateBindingSet(*layout, 0);
        BindingEntry e[] = { Buf(0, *in) };
        set->Update(e, 1);
        ComputePipelineDesc pd;
        pd.code = Code(k_push, sizeof(k_push) / 4);
        pd.layout = layout.get();
        auto pipeline = g_device->CreateComputePipeline(pd);
        CHECK(pipeline != nullptr, "compute pipeline");
        if (!pipeline)
            return;
        Run([&](CommandList& cmd) {
            cmd.SetPipeline(*pipeline);
            cmd.BindSet(*layout, 0, *set);
            BindingEntry a[] = { Buf(0, *outA) };
            cmd.PushBindings(*layout, 1, a, 1);
            cmd.Dispatch(kCount / 64, 1, 1);
            BindingEntry b[] = { Buf(0, *outB) };  // pushed again: the next dispatch writes elsewhere
            cmd.PushBindings(*layout, 1, b, 1);
            cmd.Dispatch(kCount / 64, 1, 1);
            BufferTransition t[] = { { outA.get(), State::ShaderWrite, State::HostRead }, { outB.get(), State::ShaderWrite, State::HostRead } };
            cmd.Transition(nullptr, 0, t, 2);
        });
        bool ok = true;
        for (uint32_t i = 0; i < kCount; i++)
            ok &= Words(*outA)[i] == (i + 1) * 2 && Words(*outB)[i] == (i + 1) * 2;
        CHECK(ok, "both pushed targets were written through one persistent input");
    }

    // The renderer's write fences: an event signalled from inside a list,
    // seen by the CPU while the list still runs.
    void Events()
    {
        auto layout = InOutLayout();
        auto in = Make(kCount * 4, MemoryKind::Upload, kBufferStorage);
        auto out = Make(kCount * 4, MemoryKind::Upload, kBufferStorage);
        auto set = g_device->CreateBindingSet(*layout, 0);
        BindingEntry e[] = { Buf(0, *in), Buf(1, *out) };
        set->Update(e, 2);
        for (uint32_t i = 0; i < kCount; i++)
            Words(*in)[i] = i;
        ComputePipelineDesc pd;
        pd.code = Code(k_spin, sizeof(k_spin) / 4);
        pd.layout = layout.get();
        auto pipeline = g_device->CreateComputePipeline(pd);
        CHECK(pipeline != nullptr, "compute pipeline");
        if (!pipeline)
            return;
        auto event = g_device->CreateEvent();
        auto list = g_device->CreateCommandList();
        auto fence = g_device->CreateFence();
        CHECK(!event->Signaled(), "a new event is not signalled");
        list->Begin();
        list->SetPipeline(*pipeline);
        list->BindSet(*layout, 0, *set);
        uint32_t quick = 1, heavy = 1u << 20;
        list->PushConstants(*layout, &quick, 4);
        list->Dispatch(kCount / 64, 1, 1);
        list->SignalEvent(*event);
        BufferTransition t[] = { { out.get(), State::ShaderWrite, State::ShaderRead } };
        list->Transition(nullptr, 0, t, 1);
        list->PushConstants(*layout, &heavy, 4);
        for (int i = 0; i < 4; i++)  // long enough that the fence is well after the event
            list->Dispatch(kCount / 64, 1, 1);
        list->End();
        g_device->Submit(*list, *fence);
        auto start = std::chrono::steady_clock::now();
        while (!event->Signaled() && std::chrono::steady_clock::now() - start < std::chrono::seconds(10))
            std::this_thread::yield();
        const bool seen = event->Signaled();
        const bool fenceDone = fence->Signaled();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        fence->Wait();
        double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        CHECK(seen, "the event was signalled");
        printf("  event seen after %.2f ms; list finished after %.2f ms; the event was seen %s the fence\n", ms, total,
            fenceDone ? "after" : "before");
        event->Reset();
        CHECK(!event->Signaled(), "Reset clears it");
        fence->Reset();
        CHECK(!fence->Signaled(), "a reset fence is unsignalled");
    }

    void Timestamps()
    {
        if (!g_device->GetCapabilities().timestampQueries)
        {
            printf("  timestamps: not offered (skipped)\n");
            return;
        }
        auto pool = g_device->CreateQueryPool(QueryKind::Timestamp, 2);
        auto buf = Make(1024, MemoryKind::Device, kBufferCopyDst);
        Run([&](CommandList& cmd) {
            cmd.ResetQueries(*pool, 0, 2);
            cmd.WriteTimestamp(*pool, 0, false);
            cmd.FillBuffer(*buf, 0, 1024, 0xABCD);
            cmd.WriteTimestamp(*pool, 1, true);
        });
        uint64_t t[2] = {};
        CHECK(pool->Read(0, 2, t), "results ready after the fence");
        CHECK(t[1] >= t[0], "the end is not before the start (%llu, %llu)", (unsigned long long)t[0], (unsigned long long)t[1]);
    }

    // ---- the graphics half ---------------------------------------------------

    struct DrawConstants { float color[4]; float z; };

    std::unique_ptr<Image> MakeImage(Format format, uint32_t w, uint32_t h, uint32_t usage)
    {
        ImageDesc desc;
        desc.format = format;
        desc.extent = { w, h, 1 };
        desc.usage = usage;
        return g_device->CreateImage(desc);
    }

    // Reads an 8x8 RGBA8 image back as bytes (the image is in `state`).
    std::vector<uint8_t> Readback(CommandList& cmd, Image& image, Buffer& buffer, State state, uint32_t w, uint32_t h)
    {
        ImageTransition t[] = { { &image, state, State::CopySrc } };
        cmd.Transition(t, 1);
        BufferImageCopy copy;
        copy.region = { 0, 0, w, h };
        cmd.CopyImageToBuffer(image, buffer, &copy, 1);
        BufferTransition b[] = { { &buffer, State::CopyDst, State::HostRead } };
        cmd.Transition(nullptr, 0, b, 1);
        return {};
    }

    GraphicsPipelineDesc FullscreenDesc(const BindingLayout& layout, Format color, ShaderCode fragment)
    {
        GraphicsPipelineDesc d;
        d.vertex = Code(k_tri_vert, sizeof(k_tri_vert) / 4);
        d.fragment = fragment;
        d.layout = &layout;
        d.colorFormats[0] = color;
        return d;
    }

    void Rendering()
    {
        BindingLayoutDesc ld;
        ld.pushConstantBytes = sizeof(DrawConstants);
        auto layout = g_device->CreateBindingLayout(ld);
        ShaderCode fs = Code(k_color_frag, sizeof(k_color_frag) / 4);

        GraphicsPipelineDesc depthDesc = FullscreenDesc(*layout, Format::RGBA8, fs);
        depthDesc.depthFormat = Format::D32F_S8;
        depthDesc.depthTest = depthDesc.depthWrite = true;
        depthDesc.depthCompare = CompareOp::Less;
        GraphicsPipelineDesc blendDesc = FullscreenDesc(*layout, Format::RGBA8, fs);
        blendDesc.depthFormat = Format::D32F_S8;  // (attached, tested off: the pass has one)
        blendDesc.blend[0] = { true, BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha, BlendOp::Add, BlendFactor::One, BlendFactor::Zero,
            BlendOp::Add, 0xF };
        const auto began = std::chrono::steady_clock::now();
        auto depthPipeline = g_device->CreateGraphicsPipeline(depthDesc);
        auto blendPipeline = g_device->CreateGraphicsPipeline(blendDesc);
        printf("  two graphics pipelines made in %.1f ms (SPIR-V in; on Metal also MSL, library, pipeline state)\n",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count());
        CHECK(depthPipeline && blendPipeline, "graphics pipelines");
        if (!depthPipeline || !blendPipeline)
            return;

        auto target = MakeImage(Format::RGBA8, 8, 8, kImageRenderTarget | kImageCopySrc);
        auto depth = MakeImage(Format::D32F_S8, 8, 8, kImageDepthStencil);
        CHECK(target && depth, "images");
        CHECK(target->Bytes() >= 8 * 8 * 4, "the image's memory is accounted (%llu)", (unsigned long long)target->Bytes());
        auto back = Make(8 * 8 * 4, MemoryKind::UploadCached, kBufferCopyDst);

        Run([&](CommandList& cmd) {
            ImageTransition begin[] = { { target.get(), State::Undefined, State::RenderTarget }, { depth.get(), State::Undefined, State::DepthWrite } };
            cmd.Transition(begin, 2);
            RenderingDesc rd;
            rd.color[0] = { target.get(), LoadOp::Clear, StoreOp::Store, { { 0, 0, 1, 1 } } };
            rd.depth = { depth.get(), LoadOp::Clear, StoreOp::DontCare, { 1.0f, 0 }, false };
            rd.area = { 0, 0, 8, 8 };
            cmd.BeginRendering(rd);
            cmd.SetViewport(0, 0, 8, 8, 0, 1);
            cmd.SetScissor({ 0, 0, 8, 8 });
            cmd.SetPipeline(*depthPipeline);
            auto draw = [&](float r, float g, float b, float a, float z) {
                DrawConstants c = { { r, g, b, a }, z };
                cmd.PushConstants(*layout, &c, sizeof(c));
                cmd.Draw(3, 1, 0, 0);
            };
            draw(1, 1, 0, 1, 0.3f);   // yellow, nearest
            draw(1, 0, 0, 1, 0.5f);   // red, behind it: the depth test rejects it
            cmd.SetPipeline(*blendPipeline);
            draw(1, 0, 0, 0.5f, 0.0f);  // red at half alpha over the yellow
            ClearColor magenta = { { 1, 0, 1, 1 } };
            cmd.ClearAttachments({ 0, 0, 4, 8 }, &magenta, 1, nullptr, false, false);  // the left half, mid-pass
            cmd.EndRendering();
            Readback(cmd, *target, *back, State::RenderTarget, 8, 8);
        });
        back->Invalidate(0, 8 * 8 * 4);
        const uint8_t* px = static_cast<const uint8_t*>(back->Map());
        auto near = [](int a, int b) { return a - b <= 2 && b - a <= 2; };
        bool ok = true;
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
            {
                const uint8_t* p = px + (y * 8 + x) * 4;
                if (x < 4)
                    ok &= p[0] == 255 && p[1] == 0 && p[2] == 255;                 // the cleared half: magenta
                else
                    ok &= near(p[0], 255) && near(p[1], 128) && p[2] == 0;          // yellow blended with red at 0.5: orange
            }
        CHECK(ok, "depth rejection, alpha blending and the mid-pass clear (pixel 0,0 = %u,%u,%u; 7,7 = %u,%u,%u)", px[0], px[1], px[2],
            px[(7 * 8 + 7) * 4], px[(7 * 8 + 7) * 4 + 1], px[(7 * 8 + 7) * 4 + 2]);
    }

    // Which way up a picture comes out, and which face is the front: the
    // vertices of orient.vert run top left, top right, bottom left, as seen on
    // the screen, so the top left half of the target is lit, and the triangle
    // is front-facing for frontCounterClockwise = false (Vulkan's rule: the
    // framebuffer's y runs down). Both backends must agree.
    void Orientation()
    {
        BindingLayoutDesc ld;
        ld.pushConstantBytes = sizeof(DrawConstants);
        auto layout = g_device->CreateBindingLayout(ld);
        auto target = MakeImage(Format::RGBA8, 8, 8, kImageRenderTarget | kImageCopySrc);
        auto back = Make(8 * 8 * 4, MemoryKind::UploadCached, kBufferCopyDst);
        // Cull Back / Front, frontCounterClockwise false / true: lit or not.
        bool lit[2][2];
        bool pictureOk = true;
        for (int cull = 0; cull < 2; cull++)
            for (int ccw = 0; ccw < 2; ccw++)
            {
                GraphicsPipelineDesc d;
                d.vertex = Code(k_orient_vert, sizeof(k_orient_vert) / 4);
                d.fragment = Code(k_color_frag, sizeof(k_color_frag) / 4);
                d.layout = layout.get();
                d.colorFormats[0] = Format::RGBA8;
                d.cull = cull ? CullMode::Front : CullMode::Back;
                d.frontCounterClockwise = ccw;
                auto pipeline = g_device->CreateGraphicsPipeline(d);
                CHECK(pipeline != nullptr, "orientation pipeline");
                if (!pipeline)
                    return;
                Run([&](CommandList& cmd) {
                    ImageTransition begin[] = { { target.get(), State::Undefined, State::RenderTarget } };
                    cmd.Transition(begin, 1);
                    RenderingDesc rd;
                    rd.color[0] = { target.get(), LoadOp::Clear, StoreOp::Store, { { 0, 0, 0, 1 } } };
                    rd.area = { 0, 0, 8, 8 };
                    cmd.BeginRendering(rd);
                    cmd.SetViewport(0, 0, 8, 8, 0, 1);
                    cmd.SetScissor({ 0, 0, 8, 8 });
                    cmd.SetPipeline(*pipeline);
                    DrawConstants c = { { 1, 1, 1, 1 }, 0.0f };
                    cmd.PushConstants(*layout, &c, sizeof(c));
                    cmd.Draw(3, 1, 0, 0);
                    cmd.EndRendering();
                    Readback(cmd, *target, *back, State::RenderTarget, 8, 8);
                });
                back->Invalidate(0, 8 * 8 * 4);
                const uint8_t* px = static_cast<const uint8_t*>(back->Map());
                auto at = [&](int x, int y) { return px[(y * 8 + x) * 4] > 128; };
                lit[cull][ccw] = at(1, 1);
                if (lit[cull][ccw])
                    pictureOk &= at(1, 1) && at(5, 0) && at(0, 5) && !at(1, 7) && !at(7, 1) && !at(6, 6);
            }
        CHECK(pictureOk, "the lit half is the top left one");
        CHECK(lit[0][0] && !lit[0][1] && !lit[1][0] && lit[1][1],
            "the triangle is front-facing with frontCounterClockwise = false: cull Back shows it %d/%d, cull Front %d/%d", lit[0][0], lit[0][1],
            lit[1][0], lit[1][1]);
    }

    void Textures()
    {
        auto sampler = g_device->CreateSampler({ Filter::Nearest, Filter::Nearest, Filter::Nearest, AddressMode::ClampToEdge,
            AddressMode::ClampToEdge, AddressMode::ClampToEdge, 1.0f });
        BindingLayoutDesc ld;
        ld.tables.push_back({ { { 0, BindingKind::CombinedImageSampler, uint8_t(ShaderStage::Fragment), sampler.get() } }, true });
        auto layout = g_device->CreateBindingLayout(ld);
        auto pipeline = g_device->CreateGraphicsPipeline(FullscreenDesc(*layout, Format::RGBA8, Code(k_tex_frag, sizeof(k_tex_frag) / 4)));
        CHECK(pipeline != nullptr, "textured pipeline");
        if (!pipeline)
            return;

        // A 4x4 texture, texel (x, y) = (x * 60, y * 60, 7, 255), uploaded by a copy.
        auto up = Make(4 * 4 * 4, MemoryKind::Upload, kBufferCopySrc);
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
            {
                uint8_t* t = static_cast<uint8_t*>(up->Map()) + (y * 4 + x) * 4;
                t[0] = uint8_t(x * 60);
                t[1] = uint8_t(y * 60);
                t[2] = 7;
                t[3] = 255;
            }
        auto texture = MakeImage(Format::RGBA8, 4, 4, kImageSampled | kImageCopyDst);
        auto target = MakeImage(Format::RGBA8, 8, 8, kImageRenderTarget | kImageCopySrc);
        auto back = Make(8 * 8 * 4, MemoryKind::UploadCached, kBufferCopyDst);
        Run([&](CommandList& cmd) {
            ImageTransition up1[] = { { texture.get(), State::Undefined, State::CopyDst } };
            cmd.Transition(up1, 1);
            BufferImageCopy copy;
            copy.region = { 0, 0, 4, 4 };
            cmd.CopyBufferToImage(*up, *texture, &copy, 1);
            ImageTransition up2[] = { { texture.get(), State::CopyDst, State::ShaderRead }, { target.get(), State::Undefined, State::RenderTarget } };
            cmd.Transition(up2, 2);
            RenderingDesc rd;
            rd.color[0] = { target.get(), LoadOp::DontCare, StoreOp::Store, {} };
            rd.area = { 0, 0, 8, 8 };
            cmd.BeginRendering(rd);
            cmd.SetViewport(0, 0, 8, 8, 0, 1);
            cmd.SetScissor({ 0, 0, 8, 8 });
            cmd.SetPipeline(*pipeline);
            BindingEntry e;
            e.slot = 0;
            e.image = { texture.get(), nullptr };  // (the layout's sampler)
            cmd.PushBindings(*layout, 0, &e, 1);
            cmd.Draw(3, 1, 0, 0);
            cmd.EndRendering();
            Readback(cmd, *target, *back, State::RenderTarget, 8, 8);
        });
        back->Invalidate(0, 8 * 8 * 4);
        const uint8_t* px = static_cast<const uint8_t*>(back->Map());
        bool ok = true;
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
            {
                const uint8_t* p = px + (y * 8 + x) * 4;  // each texel covers 2x2 pixels, nearest
                ok &= p[0] == (x / 2) * 60 && p[1] == (y / 2) * 60 && p[2] == 7 && p[3] == 255;
            }
        CHECK(ok, "the texture was sampled texel for texel (pixel 5,3 = %u,%u,%u)", px[(3 * 8 + 5) * 4], px[(3 * 8 + 5) * 4 + 1],
            px[(3 * 8 + 5) * 4 + 2]);
    }

    void ImageClearsAndBlit()
    {
        auto a = MakeImage(Format::RGBA8, 8, 8, kImageCopySrc | kImageCopyDst);
        auto b = MakeImage(Format::RGBA8, 4, 4, kImageCopySrc | kImageCopyDst);
        auto back = Make(4 * 4 * 4, MemoryKind::UploadCached, kBufferCopyDst);
        Run([&](CommandList& cmd) {
            ImageTransition t1[] = { { a.get(), State::Undefined, State::CopyDst }, { b.get(), State::Undefined, State::CopyDst } };
            cmd.Transition(t1, 2);
            cmd.ClearImage(*a, ClearColor{ { 0.0f, 1.0f, 0.0f, 1.0f } });
            ImageTransition t2[] = { { a.get(), State::CopyDst, State::CopySrc } };
            cmd.Transition(t2, 1);
            cmd.BlitImage(*a, { 0, 0, 8, 8 }, *b, { 0, 0, 4, 4 }, Filter::Linear);  // 8x8 -> 4x4
            Readback(cmd, *b, *back, State::CopyDst, 4, 4);
        });
        back->Invalidate(0, 4 * 4 * 4);
        const uint8_t* px = static_cast<const uint8_t*>(back->Map());
        bool ok = true;
        for (int i = 0; i < 16; i++)
            ok &= px[i * 4] == 0 && px[i * 4 + 1] == 255 && px[i * 4 + 2] == 0 && px[i * 4 + 3] == 255;
        CHECK(ok, "cleared green, blitted down, read back (%u,%u,%u,%u)", px[0], px[1], px[2], px[3]);
    }
}

    // Occlusion counts: samples that passed the depth test inside the query.
    void Occlusion()
    {
        if (!g_device->GetCapabilities().occlusionQueries)
        {
            printf("  occlusion: not offered (skipped)\n");
            return;
        }
        BindingLayoutDesc ld;
        ld.pushConstantBytes = sizeof(DrawConstants);
        auto layout = g_device->CreateBindingLayout(ld);
        GraphicsPipelineDesc d = FullscreenDesc(*layout, Format::RGBA8, Code(k_color_frag, sizeof(k_color_frag) / 4));
        d.depthFormat = Format::D32F_S8;
        d.depthTest = d.depthWrite = true;
        d.depthCompare = CompareOp::Less;
        auto pipeline = g_device->CreateGraphicsPipeline(d);
        CHECK(pipeline != nullptr, "occlusion pipeline");
        if (!pipeline)
            return;
        auto target = MakeImage(Format::RGBA8, 8, 8, kImageRenderTarget);
        auto depth = MakeImage(Format::D32F_S8, 8, 8, kImageDepthStencil);
        auto pool = g_device->CreateQueryPool(QueryKind::Occlusion, 2);
        auto pass = [&](bool resetFirst, bool queryVisible) {
            Run([&](CommandList& cmd) {
                if (resetFirst)
                    cmd.ResetQueries(*pool, 0, 2);
                ImageTransition begin[] = { { target.get(), State::Undefined, State::RenderTarget }, { depth.get(), State::Undefined, State::DepthWrite } };
                cmd.Transition(begin, 2);
                RenderingDesc rd;
                rd.color[0] = { target.get(), LoadOp::Clear, StoreOp::DontCare, { { 0, 0, 0, 1 } } };
                rd.depth = { depth.get(), LoadOp::Clear, StoreOp::DontCare, { 1.0f, 0 }, false };
                rd.area = { 0, 0, 8, 8 };
                cmd.BeginRendering(rd);
                cmd.SetViewport(0, 0, 8, 8, 0, 1);
                cmd.SetScissor({ 0, 0, 8, 8 });
                cmd.SetPipeline(*pipeline);
                auto draw = [&](float z) {
                    DrawConstants c = { { 1, 1, 1, 1 }, z };
                    cmd.PushConstants(*layout, &c, sizeof(c));
                    cmd.Draw(3, 1, 0, 0);
                };
                cmd.BeginQuery(*pool, 0);
                if (queryVisible)
                    draw(0.3f);  // the whole target
                cmd.EndQuery(*pool, 0);
                cmd.BeginQuery(*pool, 1);
                draw(0.5f);  // behind the first (or the first, when it wasn't drawn)
                cmd.EndQuery(*pool, 1);
                cmd.EndRendering();
            });
        };
        pass(true, true);
        uint64_t n[2] = {};
        CHECK(pool->Read(0, 2, n), "occlusion results ready after the fence");
        CHECK(n[0] > 0, "the visible draw counted samples (%llu)", (unsigned long long)n[0]);
        CHECK(n[1] == 0, "the draw behind it counted none (%llu)", (unsigned long long)n[1]);
        pass(true, false);  // the second pass draws one thing only, in query 1
        CHECK(pool->Read(0, 2, n), "results of the second pass");
        CHECK(n[0] == 0, "a reset clears the old count (%llu)", (unsigned long long)n[0]);
        CHECK(n[1] > 0, "the lone draw is in front now (%llu)", (unsigned long long)n[1]);
    }


    // The renderer's per-fetch swizzles: the 4x4 texture of Textures() through a view with red and blue swapped.
    void SwizzledView()
    {
        auto sampler = g_device->CreateSampler({ Filter::Nearest, Filter::Nearest, Filter::Nearest, AddressMode::ClampToEdge,
            AddressMode::ClampToEdge, AddressMode::ClampToEdge, 1.0f });
        BindingLayoutDesc ld;
        ld.tables.push_back({ { { 0, BindingKind::CombinedImageSampler, uint8_t(ShaderStage::Fragment), sampler.get() } }, true });
        auto layout = g_device->CreateBindingLayout(ld);
        auto pipeline = g_device->CreateGraphicsPipeline(FullscreenDesc(*layout, Format::RGBA8, Code(k_tex_frag, sizeof(k_tex_frag) / 4)));
        CHECK(pipeline != nullptr, "textured pipeline");
        auto up = Make(4 * 4 * 4, MemoryKind::Upload, kBufferCopySrc);
        for (int i = 0; i < 16; i++)
        {
            uint8_t* t = static_cast<uint8_t*>(up->Map()) + i * 4;
            t[0] = uint8_t(i * 10);
            t[1] = 100;
            t[2] = 7;
            t[3] = 20;
        }
        auto texture = MakeImage(Format::RGBA8, 4, 4, kImageSampled | kImageCopyDst);
        ImageViewDesc vd;
        vd.swizzle[0] = Swizzle::B;
        vd.swizzle[2] = Swizzle::R;
        vd.swizzle[3] = Swizzle::One;
        auto view = g_device->CreateImageView(*texture, vd);
        CHECK(view != nullptr, "a swizzled view");
        if (!pipeline || !view)
            return;
        auto target = MakeImage(Format::RGBA8, 8, 8, kImageRenderTarget | kImageCopySrc);
        auto back = Make(8 * 8 * 4, MemoryKind::UploadCached, kBufferCopyDst);
        Run([&](CommandList& cmd) {
            ImageTransition up1[] = { { texture.get(), State::Undefined, State::CopyDst } };
            cmd.Transition(up1, 1);
            BufferImageCopy copy;
            copy.region = { 0, 0, 4, 4 };
            cmd.CopyBufferToImage(*up, *texture, &copy, 1);
            ImageTransition up2[] = { { texture.get(), State::CopyDst, State::ShaderRead }, { target.get(), State::Undefined, State::RenderTarget } };
            cmd.Transition(up2, 2);
            RenderingDesc rd;
            rd.color[0] = { target.get(), LoadOp::DontCare, StoreOp::Store, {} };
            rd.area = { 0, 0, 8, 8 };
            cmd.BeginRendering(rd);
            cmd.SetViewport(0, 0, 8, 8, 0, 1);
            cmd.SetScissor({ 0, 0, 8, 8 });
            cmd.SetPipeline(*pipeline);
            BindingEntry e;
            e.slot = 0;
            e.image = { texture.get(), nullptr, view.get() };
            cmd.PushBindings(*layout, 0, &e, 1);
            cmd.Draw(3, 1, 0, 0);
            cmd.EndRendering();
            Readback(cmd, *target, *back, State::RenderTarget, 8, 8);
        });
        back->Invalidate(0, 8 * 8 * 4);
        const uint8_t* px = static_cast<const uint8_t*>(back->Map());
        bool ok = true;
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
            {
                const uint8_t* p = px + (y * 8 + x) * 4;
                ok &= p[0] == 7 && p[1] == 100 && p[2] == ((y / 2) * 4 + x / 2) * 10 && p[3] == 255;
            }
        CHECK(ok, "red and blue were swapped and alpha forced to one (pixel 5,3 = %u,%u,%u,%u)", px[(3 * 8 + 5) * 4], px[(3 * 8 + 5) * 4 + 1],
            px[(3 * 8 + 5) * 4 + 2], px[(3 * 8 + 5) * 4 + 3]);
    }

    // The renderer's images: one layout for life (`general`), work ordered by MemoryBarrier alone.
    void GeneralImage()
    {
        ImageDesc desc;
        desc.format = Format::RGBA8;
        desc.extent = { 8, 8, 1 };
        desc.usage = kImageRenderTarget | kImageCopySrc | kImageCopyDst;
        desc.general = true;
        auto image = g_device->CreateImage(desc);
        CHECK(image != nullptr, "a general image");
        if (!image)
            return;
        auto back = Make(8 * 8 * 4, MemoryKind::UploadCached, kBufferCopyDst);
        Run([&](CommandList& cmd) {
            ImageTransition t[] = { { image.get(), State::Undefined, State::General } };
            cmd.Transition(t, 1);
            RenderingDesc rd;
            rd.color[0] = { image.get(), LoadOp::Clear, StoreOp::Store, { { 1, 0.5f, 0, 1 } } };
            rd.area = { 0, 0, 8, 8 };
            cmd.BeginRendering(rd);
            cmd.EndRendering();
            cmd.MemoryBarrier(CommandList::Barrier::NarrowAfterPass);
            BufferImageCopy copy;
            copy.region = { 0, 0, 8, 8 };
            cmd.CopyImageToBuffer(*image, *back, &copy, 1);
            cmd.MemoryBarrier(CommandList::Barrier::Full);
            BufferTransition h[] = { { back.get(), State::CopyDst, State::HostRead } };
            cmd.Transition(nullptr, 0, h, 1);
        });
        back->Invalidate(0, 8 * 8 * 4);
        const uint8_t* px = static_cast<const uint8_t*>(back->Map());
        CHECK(px[0] == 255 && px[1] >= 127 && px[1] <= 128 && px[2] == 0 && px[3] == 255, "the pass cleared it and the copy read it, in one layout (%u,%u,%u,%u)",
            px[0], px[1], px[2], px[3]);
    }

    // Guest memory (a buffer of the host's pages, imported or copied) and the ring's kind of memory.
    void GuestAndStream()
    {
        const uint64_t size = 16ull << 20;
        void* host = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        CHECK(host != MAP_FAILED, "host memory");
        GuestMemory guest;
        CHECK(g_device->CreateGuestMemory(host, size, guest), "guest memory");
        CHECK(guest.buffer && guest.buffer->Size() == size, "its buffer");
        printf("  guest memory %s\n", guest.shadow ? "copied (shadow)" : "imported");
        auto ring = Make(1 << 20, MemoryKind::Stream, kBufferStorage | kBufferUniform | kBufferIndex);
        CHECK(ring && ring->Map(), "a stream buffer the CPU can write");
        if (ring && ring->Map())
            memset(ring->Map(), 0xA5, 1 << 20);
        guest = {};
        munmap(host, size);
    }

int main()
{
    g_device = MakeDevice();
    if (!g_device)
    {
        printf("no " RHI_BACKEND_NAME " device: skipped\n");
        return 0;
    }
    printf("capabilities\n");
    Capabilities_();
    printf("buffer copies\n");
    BufferCopies();
    printf("host memory import\n");
    HostImport();
    printf("binding set, dynamic offsets\n");
    DynamicOffsets();
    printf("pushed bindings\n");
    PushedBindings();
    printf("events\n");
    Events();
    printf("timestamps\n");
    Timestamps();
    printf("rendering: depth, blend, partial clear\n");
    Rendering();
    printf("orientation, winding\n");
    Orientation();
    printf("occlusion queries\n");
    Occlusion();
    printf("sampled texture, pushed binding, immutable sampler\n");
    Textures();
    printf("image clear, blit, copy\n");
    ImageClearsAndBlit();
    printf("swizzled view\n");
    SwizzledView();
    printf("general image, memory barrier\n");
    GeneralImage();
    printf("guest memory, stream buffer\n");
    GuestAndStream();
    g_device.reset();
    printf(g_failures ? "%d FAILED\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
