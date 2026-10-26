// SpeedBreaker runtime. GPL-3.0-or-later (see COPYING).
//
// GPU input capture (DECISIONS.md D7): the command processor's input for a few
// frames, recorded so that the renderer can be fed the same frames again
// without the game (replay), which the game itself can't do (its timing moves
// every run). NFSMW_CAPTURE=<file> turns it on; NFSMW_CAPTURE_START=<swap>
// (default 600) is the first swap count to capture from and
// NFSMW_CAPTURE_TRIGGER=<file> (instead of the swap count: from when the file
// exists) and NFSMW_CAPTURE_FRAMES=<n> (default 5) the frames after the partial one that
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
#include <vector>

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

    // Replay: NFSMW_REPLAY=<file> runs the renderer from a capture, without
    // the game. The command processor thread loads it (ReplayLoad), then in
    // place of the game's write pointer takes one batch after another
    // (ReplayNext).
    bool ReplayRequested();
    const char* ReplayPath();

    struct ReplayState
    {
        std::vector<uint32_t> regs;
        uint32_t primary = 0, primarySize = 0, readIndex = 0, writeback = 0;
        uint32_t startFrame = 0;  // the swap count the capture began at
        uint64_t binMask = 0, binSelect = 0;
    };
    // Reads the header and loads the snapshot into guest physical memory.
    bool ReplayLoad(const char* path, uint8_t* physical, ReplayState& state);

    // Any thread: the replay has run all its batches (and their GPU work).
    bool ReplayFinished();

    constexpr uint32_t kReplayEnd = ~0u;
    // Command processor thread, with the previous batch consumed: waits for
    // its work to finish, applies the next batch's CPU changes to guest
    // memory and returns its write pointer (dwords), or kReplayEnd.
    uint32_t ReplayNext(uint8_t* physical);
}
