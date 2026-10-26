// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// GPU input capture (DECISIONS.md D7): the command processor's input for a few
// frames, recorded so that the renderer can be fed the same frames again
// without the game (replay), which the game itself can't do (its timing moves
// every run). NFSMW_CAPTURE=<file> turns it on; NFSMW_CAPTURE_START=<swap>
// (default 600) is the first swap count to capture from and
// NFSMW_CAPTURE_FRAMES=<n> (default 5) the frames after the partial one that
// starts it. The game runs much slower while capturing.
//
// File (host byte order): "SBGTRC01", the command processor's state at the
// start (registers, ring, bin mask), a snapshot of guest physical memory (the
// non-zero 4 KB pages), then one record per batch of ring packets: its write
// pointer and the pages the CPU changed since the previous batch. Pages the
// GPU wrote (resolves, the command processor's own stores) are left out: a
// replay computes those itself, which is what it is there to test.
#pragma once
#include <cstdint>

namespace gpu::trace
{
    // The command processor's state a replay starts from.
    struct CpState
    {
        const uint32_t* regs;
        uint32_t regCount;
        const uint8_t* physical;  // guest physical memory (512 MB), host address
        uint32_t primary;         // ring: physical address, bytes, and the read index (dwords)
        uint32_t primarySize;
        uint32_t readIndex;
        uint32_t writeback;       // read pointer write-back address (0: off)
        uint64_t binMask, binSelect;
    };

    // NFSMW_CAPTURE is set.
    bool Enabled();

    // Command processor thread, before it executes the ring up to `writeIndex`
    // (dwords): `frame` is the swaps executed so far. Starts the capture at
    // the configured frame, records the batch while one is running, and ends
    // it after the configured frames.
    void BeforeBatch(const CpState& state, uint32_t writeIndex, uint64_t frame);
}
