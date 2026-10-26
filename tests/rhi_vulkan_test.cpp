// Tests for the RHI's Vulkan backend (runtime/gpu/rhi/vulkan): real calls on
// the machine's Vulkan device, headless. Each test is one of the concepts
// the renderer will lean on (DECISIONS.md, presenter refactor step 1): copies
// and barriers on buffers, importing host memory without a copy, a binding
// set with a dynamic uniform offset, pushed bindings, push constants, events
// signalled from inside a command list, and timestamps. No Vulkan device
// (CI without one) is a skip, not a failure. Build with the tests (-DSB_TESTS=ON).
#include <gpu/rhi/command_list.h>
#include <gpu/rhi/vulkan/vulkan_device.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include <sys/mman.h>

#include "rhi_vulkan_test_shaders.h"

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
        CHECK(std::string(c.backend) == "Vulkan", "backend name");
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
}

int main()
{
    g_device = vulkan::CreateHeadlessDevice();
    if (!g_device)
    {
        printf("no Vulkan device: skipped\n");
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
    g_device.reset();
    printf(g_failures ? "%d FAILED\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
