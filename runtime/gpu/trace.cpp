// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
#include <stdafx.h>
#include "trace.h"
#include "renderer.h"

#include <kernel/write_watch.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gpu::trace
{
    namespace
    {
        constexpr size_t kPhysicalSize = 0x20000000;
        constexpr size_t kPage = 4096;
        constexpr uint32_t kPages = uint32_t(kPhysicalSize / kPage);

        const char* const s_path = std::getenv("NFSMW_CAPTURE");

        uint64_t EnvNumber(const char* name, uint64_t fallback)
        {
            const char* v = std::getenv(name);
            return v && *v ? strtoull(v, nullptr, 0) : fallback;
        }

        enum class Phase { Waiting, Capturing, Done };
        Phase s_phase = Phase::Waiting;
        FILE* s_file = nullptr;
        std::vector<uint8_t> s_shadow;  // guest memory as the file has it
        uint64_t s_startFrame = 0;      // the swap count the capture began at
        uint64_t s_batches = 0, s_deltaPages = 0, s_snapshotPages = 0;
        std::chrono::steady_clock::time_point s_began;

        void Put(const void* data, size_t size)
        {
            if (s_file && fwrite(data, 1, size, s_file) != size)
            {
                fprintf(stderr, "[capture] write to %s failed: stopping\n", s_path);
                fclose(s_file);
                s_file = nullptr;
                s_phase = Phase::Done;
            }
        }

        void PutU32(uint32_t v)
        {
            Put(&v, 4);
        }

        bool ZeroPage(const uint8_t* page)
        {
            static const uint8_t zeros[kPage] = {};
            return memcmp(page, zeros, kPage) == 0;
        }

        // Whether the GPU, not the CPU, last wrote the page: its change is a
        // result of the work, not an input to it.
        bool WrittenByGpu(uint32_t page)
        {
            writewatch::Writer kind = writewatch::LastWriter(page * uint32_t(kPage)).kind;
            return kind == writewatch::Writer::kResolve || kind == writewatch::Writer::kGpuStore;
        }

        void Begin(const CpState& state, uint64_t frame)
        {
            s_file = fopen(s_path, "wb");
            if (!s_file)
            {
                fprintf(stderr, "[capture] can't write %s: not capturing\n", s_path);
                s_phase = Phase::Done;
                return;
            }
            s_phase = Phase::Capturing;
            s_startFrame = frame;
            s_began = std::chrono::steady_clock::now();
            // Nothing in flight, so the memory below is what the GPU work so far left.
            renderer::Flush();

            Put("SBGTRC01", 8);
            PutU32(state.regCount);
            PutU32(state.primary);
            PutU32(state.primarySize);
            PutU32(state.readIndex);
            PutU32(state.writeback);
            PutU32(uint32_t(frame));
            Put(&state.binMask, 8);
            Put(&state.binSelect, 8);
            Put(state.regs, size_t(state.regCount) * 4);

            s_shadow.assign(kPhysicalSize, 0);
            long countAt = ftell(s_file);
            PutU32(0);  // the page count, filled in below
            uint32_t pages = 0;
            for (uint32_t i = 0; i < kPages; i++)
            {
                const uint8_t* live = state.physical + size_t(i) * kPage;
                memcpy(&s_shadow[size_t(i) * kPage], live, kPage);
                const uint8_t* copy = &s_shadow[size_t(i) * kPage];
                if (ZeroPage(copy))
                    continue;
                PutU32(i);
                Put(copy, kPage);
                pages++;
            }
            s_snapshotPages = pages;
            if (s_file)
            {
                long end = ftell(s_file);
                fseek(s_file, countAt, SEEK_SET);
                fwrite(&pages, 4, 1, s_file);
                fseek(s_file, end, SEEK_SET);
            }
            fprintf(stderr, "[capture] started at swap %llu: %u pages (%u MB) of guest memory, ring 0x%08X (%u KB), read index %u\n",
                (unsigned long long)frame, pages, pages * 4 / 1024, state.primary, state.primarySize / 1024, state.readIndex);
        }

        // The pages the CPU changed since the last batch: compared with what
        // the file already holds, which then catches up with all of them.
        void PutDelta(const CpState& state)
        {
            long countAt = ftell(s_file);
            PutU32(0);
            uint32_t pages = 0;
            for (uint32_t i = 0; i < kPages; i++)
            {
                const uint8_t* live = state.physical + size_t(i) * kPage;
                uint8_t* kept = &s_shadow[size_t(i) * kPage];
                if (memcmp(kept, live, kPage) == 0)
                    continue;
                memcpy(kept, live, kPage);
                if (WrittenByGpu(i))
                    continue;
                PutU32(i);
                Put(kept, kPage);
                pages++;
            }
            if (s_file)
            {
                long end = ftell(s_file);
                fseek(s_file, countAt, SEEK_SET);
                fwrite(&pages, 4, 1, s_file);
                fseek(s_file, end, SEEK_SET);
            }
            s_deltaPages += pages;
        }

        void End(uint64_t frame)
        {
            PutU32(2);
            long size = s_file ? ftell(s_file) : 0;
            if (s_file)
                fclose(s_file);
            s_file = nullptr;
            s_phase = Phase::Done;
            s_shadow = std::vector<uint8_t>();
            fprintf(stderr, "[capture] done at swap %llu: %llu batches, %llu delta pages, %.0f MB in %s (%.1f s)\n",
                (unsigned long long)frame, (unsigned long long)s_batches, (unsigned long long)s_deltaPages, double(size) / 1048576.0,
                s_path, std::chrono::duration<double>(std::chrono::steady_clock::now() - s_began).count());
        }
    }

    bool Enabled()
    {
        return s_path != nullptr && s_phase != Phase::Done;
    }

    void BeforeBatch(const CpState& state, uint32_t writeIndex, uint64_t frame)
    {
        static const uint64_t start = EnvNumber("NFSMW_CAPTURE_START", 600);
        static const uint64_t frames = EnvNumber("NFSMW_CAPTURE_FRAMES", 5);
        if (s_phase == Phase::Waiting)
        {
            if (frame < start)
                return;
            Begin(state, frame);
            if (s_phase != Phase::Capturing)
                return;
        }
        else if (s_phase == Phase::Capturing)
        {
            if (frame > s_startFrame + frames)
            {
                End(frame);
                return;
            }
            // The previous batch's work, finished, so that its results are in memory.
            renderer::Flush();
        }
        else
            return;

        PutU32(1);
        PutU32(writeIndex);
        PutU32(uint32_t(frame));
        PutDelta(state);
        s_batches++;
    }
}
