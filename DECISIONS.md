# Metal backend: decisions and progress log

Working log for the `feature/metal-backend-renderer` branch (personal fork,
`smafara/speedbreaker`). It records what was decided, why, and what was done, so a
later session can pick up without the original conversation. Newest entries go at the
bottom of each section. Keep it factual: decisions, rationale, and results, not chatter.

## Goal

Add a direct Metal renderer beside the existing Vulkan one (Vulkan stays, and stays
the default for Linux, Windows and iOS until Metal is proven). This is a curiosity
project; architectural correctness is preferred over speed of prototyping.

Out of scope for now: MetalFX / frame interpolation, KosmicKrisp, direct MSL emission
(see "Later").

## Decisions

### D1. Fork and branch (2026-10-25)
- Fork: https://github.com/smafara/speedbreaker. Remotes: `origin` is upstream
  `SpeedBreakerProject/speedbreaker`, `fork` is the personal fork.
- Branch: `feature/metal-backend-renderer`, tracking `fork`. Never push to `origin`.
- Upstream does not merge PRs directly (releases arrive as single commits), so there
  is no upstream-conventions pressure on commit shape.

### D2. Refactor first, then add Metal (2026-10-25)
Chosen: build a backend interface with Vulkan as the only implementation first, then
add Metal behind it. Rejected: a standalone Metal presenter prototype first.
- Why: correctness of the architecture over early visible results. The interface
  should come from what the renderer really does.
- Accepted risk: the interface is designed against one backend and may leak Vulkan
  concepts. Mitigation: shape it for Metal (see D3) and revisit if it gets cumbersome.

### D3. Interface granularity: command-level RHI (2026-10-25)
Chosen: a thin resource/command layer (buffers, images, pipelines, command lists,
queue/fences, barriers). Xenos emulation logic stays shared and calls it.
Rejected: a pass-level backend (backend implements "resolve this rect", "untile this
texture"), because each backend would duplicate emulation logic.
- Guidance: do not leak Vulkan-isms into the API (explicit image layouts,
  `vkCmdPipelineBarrier`, push descriptors). Use resource-usage transitions that Metal
  can map to hazard tracking or fences.
- Revisit if the layer becomes cumbersome (the user explicitly accepts refactoring it
  later).

### D4. Shaders stay GLSL for now (2026-10-25)
The RHI takes GLSL source plus a stage. Vulkan: glslang to SPIR-V (as today, through
the disk cache). Metal: glslang to SPIR-V, then SPIRV-Cross to MSL.
- Why: the Xenos translator and presenter shaders are proven; this keeps them untouched.
- Intent: emit MSL directly from the translator later (see "Later").

### D6. Toolchain: Xcode's clang, not Homebrew LLVM (2026-10-25)
Chosen: build with Apple clang 21 (Xcode) and skip `brew install llvm lld`.
- Why it should work: `setup_tools.sh` falls back to the system `clang` when
  `/opt/homebrew/opt/llvm` is absent (needs 18+; `LLVM=<bin dir>` overrides);
  `build_deps.sh` already uses Apple's clang; CMake needs no lld on macOS (lld-link is
  Windows only). Homebrew's llvm/lld (llvm-mc, llvm-objdump, ld.lld) are needed only by
  `tests/run_instr_tests.sh` (README, XenonRecomp instruction tests), which we skip.
- Caveat: releases are built with Homebrew LLVM, so codegen of the recompiled PPC code
  may differ from release builds. Fine for our purpose as long as every baseline and
  comparison uses the same compiler. Fallback if Apple clang fails to compile something:
  install Homebrew `llvm` and pass its clang in `CMAKE_C(XX)_COMPILER`.
- Pass `-DCMAKE_C_COMPILER=$(xcrun -f clang) -DCMAKE_CXX_COMPILER=$(xcrun -f clang++)`
  explicitly (runtime/CMakeLists.txt notes CMake otherwise picks `/usr/bin/c++` for
  Objective-C++).

### D5. Commit hygiene (proposed, not yet confirmed by the user)
Small, buildable commits, one behavior-preserving refactor step each, on the fork.
Confirm before relying on it.

### D7. Verification by GPU-input capture and replay (proposed 2026-10-26, not yet agreed)
The game is not deterministic, so frames from two runs can't be diffed. Instead record the
renderer's input and replay it headless (the PCSX2 GS-dump / Dolphin FifoCI / Xenia trace
model). Findings from reading `gpu/command_processor.cpp` (details in "CP findings" below).
A RAM snapshot alone cannot restore the game: recompiled guest code runs as host C++
calls (`void sub_XXXX(PPCContext&, uint8_t* base)`), so thread state lives on host stacks
(this is my reading of the generated signatures, not a traced call site).

CP findings (2026-10-26):
- One command-processor thread (`Worker`, file-static state) reads the ring from guest
  physical memory (`Physical(primary)`), polling `MMIO_BASE + REG_CP_RB_WPTR*4`
  (0x7FC80000 page; the game's register stores land in guest memory, no MMIO traps).
  `ExecuteBuffer(Reader&, ctx)` runs the packets; `INDIRECT_BUFFER` nests.
- Inputs to capture: ring bytes per batch (`s_readIndex*4 .. writeIndex*4`), indirect
  buffers, the guest physical memory the draws/resolves read (vertex/index data, textures
  via fetch constants, `IM_LOAD` shader microcode), and the values behind
  `WAIT_REG_MEM` / `COND_WRITE` that the CPU sets.
- Outputs: renderer commands to the presenter, and guest stores (fences, scratch regs,
  `EVENT_WRITE_SHD`, resolves into guest memory via `GuestStore` / `writewatch`).
- Replay needs: guest-code callbacks stubbed (`DispatchInterrupt` calls guest code;
  `Vsync` thread), `writewatch::BeforeHostWrite` when loading memory, `NFSMW_HEADLESS`
  does not help (it disables the renderer), ring params from `VdInitializeRingBuffer`
  (`gpu/imports_video.cpp`). Hook points: `ExecuteBuffer` entry/exit (capture), the
  `Worker` loop (replay feeds WPTR), `main.cpp` before the guest thread starts.
- Proposed capture format: header + initial physical-memory snapshot (sparse, zero pages
  skipped) + per batch: ring bytes, WPTR, and a delta of the pages the CPU changed since
  the previous batch. Delta by memcmp against a shadow copy, with the GPU drained
  (`renderer::Flush`) and the shadow refreshed after each batch so GPU resolves are not
  recorded (they are recomputed in replay). Capture mode is slow; that is acceptable.
- Cheap first step: hash the ring bytes per frame, run a scene twice, see where the input
  diverges (same hook point).

### D8-D10. RHI shape (2026-10-26, chosen by the user from three options each)
Refines D3. D8, dispatch: abstract classes (`rhi::Device`, `CommandList`, `Buffer`, `Image`,
`Pipeline`, ...) implemented by each backend and chosen at runtime, so Vulkan stays
selectable and both can be compared in one binary; the cost is a virtual call per command
(~7k draws a frame: well under 0.1 ms). Rejected: opaque handles with a function table,
compile-time backend. D9, barriers: resource states (the renderer says what a resource is
about to be used as: RenderTarget, ShaderRead, CopySrc, CopyDst, ...); the Vulkan backend
maps a state to layout + stage + access, Metal needs little (it tracks hazards) and emits
fences only where required. Rejected: Vulkan-style stage/access/layout barriers in the RHI,
automatic barrier inference (maybe later). D10, bindings: a declared binding table (each
pipeline declares its binding slots, the command list sets resources at slots); Vulkan
implements it with push descriptors / descriptor sets, Metal with argument table slots.
Rejected: descriptor sets and pools as RHI objects, bindless (needs shader translator
changes, deferred by D4).

### D11. Swapchain and present timing (2026-10-26; the user said go ahead; implemented for Vulkan)
`runtime/gpu/rhi/swapchain.h` (+ `Device::CreateSwapchain`, `SubmitToSwapchain`). The
interface is not the Vulkan calls: Vulkan learns when a present reached the screen by
waiting for it on a thread (VK_KHR_present_wait) and polls the driver's reports
(VK_GOOGLE_display_timing); Metal hands both over in a presented handler on the drawable.
So the backend delivers **callbacks** (`PresentShown{id, presented, onScreen, timed}`,
`PresentReport{id, onScreen}`), and a present request carries the caller's id and an optional
`notBefore` time (steady clock; the backend converts: Vulkan/Apple host time, Metal
presentAtTime). The waiter thread, its QoS and the `SwapchainUse` generation scheme move
inside the Vulkan backend; Acquire/Submit/Present hide the acquire/render-done semaphores.
Pacing policy (hold before present, desired-time choice and watchdog, refresh fit, repeats,
swapchain rebuild policy, iOS scale, SDL) stays in the presenter. Open: whether images are
exposed as `rhi::Image` (drafted so) and the Metal drawable's `framebufferOnly=NO` cost.

### D12. Presenter shaders: GLSL compiled at run time, SPIR-V into the RHI (2026-10-26, "A'")
The user first chose build-time compilation (A), then asked for a re-analysis once it showed
the case for it was wrong: glslang is linked into the game anyway (the renderer compiles the
translated shaders at run time), the presenter's compiles take 1-3 ms in the log, and a
build-time step needs a glslang tool per host (iOS cross-builds) plus the variants
(`ToneVariant`, `fsr1::EasuGlsl`) enumerated. So: keep the inline GLSL, compile with
`CompileShader`, give the SPIR-V to `Device::Create*Pipeline` (which also puts the presenter's
pipelines in the backend's pipeline cache). Found while re-analysing: (1) the front buffer is
the renderer's own raw `VkImage`, written in GENERAL and read by the presenter in queue
order: until the renderer moves, the presenter wraps it (transitional `WrapImage`, state
`ShaderWrite` = GENERAL, which samples fine) ; (2) `ui::Draw(VkCommandBuffer)` (ImGui's
Vulkan backend) needs the frame list's native command buffer: transitional
`NativeCommandBuffer`; the ImGui Metal backend later needs the encoder instead.

## Findings from the survey (2026-10-25)

Vulkan footprint (approximate counts of `Vk*`/`vk*`/`VK_*` references):
- `runtime/gpu/renderer.cpp`: ~1000 (11k lines; Xenos logic and Vulkan mixed together).
- `runtime/video/presenter.cpp`: ~820 (4.9k lines; swapchain, post passes, FSR1, timing).
- `runtime/thirdparty/imgui/backends/imgui_impl_vulkan.*`: ~700; driven from `ui/ui.cpp`.
  Dear ImGui ships a Metal backend, so the UI is the easy part.
- Small leaks: `video/presenter.h`, `gpu/renderer.h`, `ui/ui.h`, `gpu/memory_plan.h`,
  `video/swapchain_policy.*`, `video/swapchain_use.h`, `gpu/pipeline_cache_header.h`,
  `hid/input.cpp`, `platform/ios_ui.mm`, `CMakeLists.txt`.

Seams:
- `gpu/renderer.h` is already API-agnostic (`Draw`, `Submit`, `Flush`, `QueueWrite`,
  `PresentFrontBuffer`, ...). The Vulkan lives inside `renderer.cpp`.
- `video/presenter.h` leaks Vulkan on purpose: `VulkanContext` (device, queue, queue
  mutex, extension function pointers) is shared with the renderer, and
  `SubmitFrontImage(VkImage, ...)` / `ReleaseFrontImage(VkImage)` pass raw images.
- Shader path: Xenos microcode, `gpu/shader_translator.cpp`, GLSL, glslang
  (`gpu::renderer::CompileShader`), SPIR-V, disk cache. The presenter has inline GLSL
  for AA, scale and FSR1.

Vulkan features the device hard-requires (`presenter.cpp`, `kNeededDeviceExtensions`):
swapchain, dynamic rendering, depth/stencil resolve, renderpass2, multiview,
maintenance2, push descriptor. Optionally: `EXT_external_memory_host`, fragment shading
rate, present_id / present_wait, `GOOGLE_display_timing`, `EXT_memory_budget`.

## Open design questions

- **Guest memory import (biggest Metal risk).** The renderer imports all guest physical
  memory as one storage buffer via `VK_EXT_external_memory_host`. Metal's nearest
  equivalent is `newBufferWithBytesNoCopy`, which needs page-aligned, page-multiple
  memory. Check how the guest memory region is allocated before designing the buffer
  API.
- **Synchronization model.** Vulkan explicit barriers and layouts versus Metal hazard
  tracking / fences. The RHI exposes resource-usage transitions only.
- **Pipeline cache.** Vulkan `VkPipelineCache` blobs (see `gpu/pipeline_cache_header.h`)
  versus Metal binary archives / PSO compile stalls.
- **Present timing.** The vblank lock uses `present_wait` / `GOOGLE_display_timing`;
  Metal would use `CAMetalLayer` presented handlers.
- **Queue sharing.** The renderer records on the command-processor thread while the
  presenter submits on another; both hold `queueMutex`. The RHI needs a queue model
  that both backends can satisfy.

## Plan

1. Build from source and capture a baseline (launch, a fixed scene, `[perf]` log lines)
   so each refactor step can be checked against unchanged Vulkan behavior.
2. Define `gpu/rhi/` core types, Vulkan-only. Migrate the presenter's swapchain and
   post passes first (smaller, self-contained) to prove the API shape.
3. Migrate `renderer.cpp` in slices: buffers and memory plan, then textures, then
   pipelines and passes. Build and compare against the baseline after each slice.
4. Implement the Metal backend behind the RHI (including Dear ImGui's Metal backend).

## Environment notes

- macOS (Darwin), Apple Silicon. Vulkan SDK at `~/Developer/SDKs/Vulkan/1.4.363.0`
  (`current` symlink), which provides `glslangValidator`.
- The Vulkan SDK (2026-10-25 check) already provides, under `macOS/`: glslang 16.6 with
  CMake config (`lib/cmake/glslang`), the Vulkan loader, MoltenVK, KosmicKrisp
  (`lib/KosmicKrisp.framework`, `libvulkan_kosmickrisp.dylib`, ICD json in
  `share/vulkan/icd.d`), SPIRV-Tools, and SPIRV-Cross with an MSL backend
  (`lib/cmake/spirv_cross_msl`, `libspirv-cross-c-shared`), which D4's Metal path needs.
- Still missing for a from-source build: SDL3 development files (headers + CMake config;
  only the dylib inside the installed app exists) and FFmpeg-XMA (project-patched static
  libavcodec/libavutil, `tools/ffmpeg-xma`, built by `scripts/setup_tools.sh` /
  `scripts/build_ffmpeg_xma.sh`).
- Build scope (read from the scripts, 2026-10-25): `scripts/setup_tools.sh` builds only
  host build tools into git-ignored `tools/` (XenonRecomp, extract-xiso, FFmpeg-XMA,
  Xenia instruction tests); it needs LLVM/lld (Homebrew `llvm`, `lld`). Target libraries on
  macOS are built by `scripts/macos/build_deps.sh <prefix>`, which already includes SDL3
  3.4.16, glslang 16.6, Vulkan loader, SPIRV-Tools and FFmpeg-XMA (into
  `<prefix>/ffmpeg-xma`; pass it as `-DFFMPEG_XMA`). It has no subset option, so one run
  fills both gaps left by the SDK; nothing needs adding to either script. The Vulkan SDK
  is still useful for SPIRV-Cross (MSL), MoltenVK and KosmicKrisp. The game itself is recompiled from the
  user's own `default.xex` into git-ignored `ppc/` (README "Building from source" step 2);
  the build needs that output. Never commit `ppc/`, `game/` or `tools/`.
- `cmake` and `ninja` are in `/opt/homebrew/bin`. Homebrew has no sdl3, molten-vk,
  glslang or ffmpeg installed; `scripts/macos/build_deps.sh` builds the bundled set from
  source, but the SDK covers most of it.
- An installed `/Applications/SpeedBreaker.app` exists, with settings/logs/saves under
  `~/Library/Application Support/SpeedBreaker`. Its installed game files are at
  `/Volumes/WD_SN850X/SpeedBreaker/game` (`game-location` points there; contains
  `default.xex`, `Movies/`, `NFS/`). Checked 2026-10-25: `default.xex` is 3,936,256 bytes
  with the supported SHA-256 (`aebdf3c1...c4ba4c`). `config/nfsmw.toml` reads
  `../game/files/default.xex` and writes `../ppc`; `.gitignore` ignores `/game` without a
  trailing slash on purpose so a symlink works. Plan: symlink `game/files` to the
  installed folder rather than copying it.
- A second, untouched clone of upstream exists at
  `/Volumes/WD_SN850X/Developer/DeadSpace/speedbreaker` (nothing built there).
- Tests live in `tests/` (pure-logic, none exercise the renderer).

## Later

- Emit MSL directly from the Xenos translator and port the presenter shaders to MSL
  (replaces glslang to SPIRV-Cross).
- MetalFX / frame interpolation. The main blocker is motion vectors (the game has none).
  Note that `game/hfr.cpp` ("Smooth Motion") already renders engine-interpolated
  in-between frames.
- Try KosmicKrisp as an alternate Vulkan ICD on Apple Silicon (a cheap experiment that
  needs no code change).

## Log

- 2026-10-25: forked, created the branch, surveyed the code, recorded D1 to D4.
- 2026-10-25: symlinked `game/files` to the installed game folder; ran
  `scripts/setup_tools.sh` with Apple clang (D6): XenonRecomp (9a1a466), extract-xiso and
  FFmpeg-XMA 7.1.1 built into `tools/`. Ran XenonRecomp on `default.xex`: `ppc/` generated
  (225 files, 139 MB, 1.8 s), no errors. Git status stays clean apart from this file.
  Next: `scripts/macos/build_deps.sh build/deps`, then cmake + ninja, then the baseline.
- 2026-10-25: `scripts/macos/build_deps.sh build/deps` OK (SDL3 3.4.16, glslang 16.6.0, loader
  1.4.357, SPIRV-Tools, FFmpeg-XMA; macOS 15.0 arm64). Configured and built with Apple clang:
  `cmake -S . -B build/main -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=$(xcrun -f clang) -DCMAKE_CXX_COMPILER=$(xcrun -f clang++)
  -DCMAKE_PREFIX_PATH=$PWD/build/deps -DFFMPEG_XMA=$PWD/build/deps/ffmpeg-xma
  -DSB_MOLTENVK=~/Developer/SDKs/Vulkan/current/macOS/lib/libMoltenVK.dylib` (no Homebrew
  MoltenVK, so the SDK's is used), then `ninja -C build/main SpeedBreaker`: 347 steps, OK,
  3 warnings. Binary: `build/main/runtime/SpeedBreaker`, run from the repo root.
- 2026-10-25: first baseline run (90 s, killed by a perl alarm, exit 142 is expected; macOS
  has no `timeout`). Game starts and renders on MoltenVK. Device: Apple M5 Max (loader lists
  two; it picks 0). Reached the front end/menus only, not a race: steady 30.0 fps (worst
  frame 33.5 ms), ~6 draws, 3 passes, 2 resolves, 3 submits per frame, GPU busy ~4.1 ms,
  CP busy ~0.2 ms/frame, 792 MB device memory, 0 dropped frames. Early frames at start up
  were slower (21-47 fps, shader/pipeline warm up; one 817 ms hitch). Full log:
  `/private/tmp/claude-501/-Volumes-WD-SN850X-Developer-speedbreaker/d3146dd6-aef9-49da-928c-158c6be5f61d/scratchpad/baseline_run.log`
  (session scratch, may be gone). LIMITATION: menus exercise little of the renderer; a
  proper regression baseline needs a race scene played by a human (or a scripted capture),
  plus screenshots. Decide how before refactoring `renderer.cpp`.
- 2026-10-25: in-world baseline (user drove, 120 s, MoltenVK, Apple M5 Max, Apple-clang
  build of this branch at 61365bd). Logs kept (git-ignored) in
  `build/baselines/vulkan-moltenvk-world.log` and `...-menu.log`. Summary of the in-world
  `[perf]` lines (one per ~2.5 s): locked 60.0 fps with worst frame 16.8 ms for most of the
  run; about 3.5k-6.9k draws/frame, 20-21 passes, 25 resolves, GPU busy 5.6-6.8 ms,
  command processor busy 2.1-4.2 ms/frame, device memory 984-1048 MB. Hitches while
  loading into the world: worst frames of 1.7 s and 3.0 s (fps 31 and 23), and 0.8 s at
  start; two shorter dips (167 ms, 117 ms) at the transitions to and from the world. Use
  these as the reference: after each refactor step the steady in-world numbers should be
  unchanged within noise. Still missing: screenshots for visual diffs.
- 2026-10-26: D7 agreed in the form recommended (ring-hash experiment first; replay as a
  mode of the main binary). Added `NFSMW_HASH_STREAM=1` to `gpu/command_processor.cpp`: hashes
  every consumed packet (indirect buffers included) and logs
  `[hash] frame N packets P dwords D stream H` at each XE_SWAP. Off unless set.
  RESULT, two unattended 45 s runs from launch (same build, no input): frames 1-233 are
  bit-identical (same packets, dwords and stream hash, including all boot/loading frames).
  First divergence at frame 234, and the cause is the GAME, not renderer noise: run 1
  opened `MOVIES/psa_ENGLISH_ntsc.wmv`, run 2 `MOVIES/attract_movie_ENGLISH_ntsc.wmv`
  (a different intro movie), so the guest took a different branch (RNG or time driven; not
  yet traced). The streams never re-converge after that. Implication: the GPU input is far
  more deterministic than feared; a deterministic mode (fixed seed/time, scripted input via
  the existing `NFSMW_UI_KEYS` style hooks) may be viable for front-end scenes, and capture
  and replay is still needed for in-world frames (threads, traffic). Also found: the
  existing `NFSMW_DUMP_FRAMES=<dir>` (every 60th front buffer as PPM, CPU path); the
  front buffer is a guest-memory resolve target, so a diff at that level is
  backend-independent.
- 2026-10-26 CORRECTION to the entry above: the two runs were not unattended. The user
  watched the intros in run 1 and skipped them in run 2, so the movie difference at frame
  234 is human input, not an RNG or clock branch in the game. Nothing in frames 1-233 (or
  anywhere yet) shows game-side non-determinism; the real question is still open. Redo with
  hands off (no input at all), then with an identical input sequence.
- 2026-10-26: hands-off retest (no input at all, two 60 s runs from launch, same build; both
  played the same four intro movies in the same order). RESULT: the streams do NOT stay
  identical. Frames 1-19 match except 10-11 (packet counts differ by 2 there, then
  re-converge); from about frame 20 on, only 762 of 3222 frames have even the same packet
  count and 17 of 3222 the same hash (none after frame 500). Frame counts also differ
  (3225 vs 3222 frames in 60 s). The earlier "233 identical frames" pair was therefore the
  boot/loading phase only, before movie playback and its threads start; do not generalize
  it. Cause is timing: with the frame index (swap count) as the alignment, a movie frame
  shown over 1 or 2 swaps (decode thread, vblank) shifts everything after it. Comparing
  runs by swap index is not meaningful in time-driven phases.
  CONCLUSION: running the game again does not reproduce the GPU input even without a human;
  a deterministic mode would have to freeze guest time and thread scheduling, which is
  out of reach here. The capture and replay design (D7) stands as the verification plan.
  `NFSMW_HASH_STREAM` stays as a diagnostic (and will be reused to check that a replay
  reproduces the captured stream exactly).
- 2026-10-26: capture implemented (`gpu/trace.{h,cpp}`, hooked in `Worker` before each batch;
  `kernel/write_watch.cpp` tracks last writers when NFSMW_CAPTURE is set). Env:
  `NFSMW_CAPTURE=<file>`, `NFSMW_CAPTURE_START=<swap>` (default 600),
  `NFSMW_CAPTURE_FRAMES=<n>` (default 5). Format in `gpu/trace.h`. First capture (intro movie
  scene, swaps 600-606, hands off): 89,180 non-zero pages (348 MB) snapshot, 13 batches,
  285 CPU-delta pages in total, 350 MB file, capture took 0.6 s; the game was not slowed
  noticeably. Captures live in `build/captures/` (git-ignored; they contain game memory,
  never commit or share them). The CPU delta leaves out pages whose last writer is the GPU
  (`kResolve` from resolves, `kGpuStore` from the command processor's own stores). Known
  weakness: "last writer" is sticky, so a CPU write to a page the GPU wrote earlier, if it
  never faulted, would be missed. The replay's stream-hash check will not catch this; a
  visual check of replayed frames will. Not yet verified: that a replay reproduces the
  frames. Next: replay (a mode of the main binary, `NFSMW_REPLAY=<file>`).
- 2026-10-26: replay implemented (`NFSMW_REPLAY=<capture>`, `NFSMW_REPLAY_EXIT=1` to quit at the
  end). `main.cpp` starts `gpu::StartReplay` instead of the guest thread (window, presenter and
  renderer start as usual; no guest code runs, no vblank thread). `gpu::trace::ReplayLoad`
  puts the snapshot into guest memory (through `writewatch::BeforeHostWrite`), `StartReplay`
  restores the registers (and their MMIO mirror), ring, bin mask and swap count, and the
  command processor's `Worker` takes its write pointer from `trace::ReplayNext` (flush the GPU,
  apply the next batch's CPU delta, return its wptr) instead of the MMIO page. The interrupt
  callback stays 0, so interrupts are no-ops.
  CPU stand-in: a `WAIT_REG_MEM` still unsatisfied once the GPU is idle waits on a value the
  game's CPU would have set (D3D's fence handshake on `0x06449000/4`, cleared by the CPU
  mid-batch). The capture holds CPU changes only at batch boundaries, so those handshakes
  leave no trace (reading the next batches' deltas ahead did not work: there are fewer records
  than handshakes). The replay writes a value that satisfies the wait (`ref`, else neighbours)
  and logs the first few as `[replay] stood in for the CPU`.
  RESULT on `intro.sbtrace` (swaps 600-606; correction: the scene is the profile-load dialog
  "Loading NAME. Please don't turn off your console", not a movie): 13 batches replay in
  0.1 s; with `NFSMW_HASH_STREAM=1` the stream hash of swaps 602-606 equals the captured run's
  exactly (601 differs: the capture starts mid-frame, 1343 vs 1400 packets, as expected);
  `NFSMW_DUMP_FRAMES=<dir> NFSMW_DUMP_EVERY=1` (new stride knob, default 60) dumps every swap,
  and two separate replays give bit-identical PPMs (6 of 6). The image is right (dialog, text,
  textures). So capture + replay works for this 2D UI scene on the Vulkan backend.
  Not covered yet: a 3D in-world scene (needs a capture while driving), EDRAM carried across
  the capture start (assumes clears), the scene's limited GPU variety. Next: capture
  in-world, then the pass/fail tool (compare dumps against a stored baseline with a
  tolerance, for cross-backend use), then the RHI refactor.
- 2026-10-26: in-world capture (`build/captures/world.sbtrace`, swaps 1675-1681, 395 MB, driving
  scene; started with the new `NFSMW_CAPTURE_TRIGGER=<file>`: the capture begins when the file
  exists, so a person can say "now"). Replay: stream hashes of swaps 1677-1681 equal the
  captured run's (1676 is the partial first frame). FINDING: the first two replays of it gave
  different images (A: HUD only, no 3D world; B: world but no car) because the renderer's
  asynchronous pipeline compilation skips draws whose pipeline is not ready
  (`[pipelines] frame N: 2282 draws skipped, 66 compiling`), so the image depended on compile
  timing. A replay must wait for pipelines in full: the renderer now does so whenever
  `NFSMW_REPLAY` is set (same as Asynchronous Shaders off). After that: 0 skipped draws and two
  replays give bit-identical frames (6 of 6); the last frame is the complete scene (car, street,
  houses, HUD). Lesson for the Metal backend: anything that makes output depend on timing
  (async compiles, frame-budgeted waits) must be switchable off for comparison runs.
  Replay of this scene takes ~6 s (the intro one 0.1 s), dominated by pipeline compiles.
- 2026-10-26: pass/fail check added: `scripts/replay/replay_check.sh [--record] [--tolerance N]
  [--max-bad-percent P] [<capture>...]` replays each `build/captures/<name>.sbtrace` and compares
  its frames with `build/baselines/frames/<name>/` (git-ignored; recorded now from the Vulkan
  backend: `intro` 6 frames, `world` 6 frames); `scripts/replay/compare_frames.py` does the
  diff (a pixel is bad when a channel differs by more than the tolerance; a frame fails above
  the allowed percentage of bad pixels). `SB_REPLAY_ENV` passes extra environment (a backend
  switch later). Vulkan against its own baseline: 0 bad pixels in all 12 frames. Checked that it
  fails: one pixel changed by 16 is reported at tolerance 0 and passes at tolerance 20.
  This is the gate for every refactor step (D5): run it before each commit.
- 2026-10-26: presented-picture check (the presenter refactor needs it: the frame dumps are the
  front buffer in guest memory, before the presenter's post-processing and swapchain).
  `NFSMW_DUMP_PRESENTED=<file>` (replay only; main.cpp) saves, 30 frames after the replay's
  end, `video::CaptureFrame()`: the game picture as the window shows it, without UI;
  `NFSMW_REPLAY_EXIT=1` now ends the run from main.cpp (not the command processor).
  `RunFrame` returned early without drawing when the game stopped presenting and the UI
  needed nothing, so a screenshot request never got an answer: a pending `CaptureFrame` now
  counts as wanting a frame (this also helps the bug report's screenshot while the game is
  idle). `replay_check.sh` passes it as `presented.ppm`, compared like the frame dumps.
  Size on this machine: 5120x2880 (the window), so that baseline depends on this machine's
  window size and settings (AA, scaling): re-record if they change. Two replays of `world`
  give byte-identical presented pictures; a flipped byte in the baseline fails the check.
  The whole check (both captures, 7 pictures each) takes ~2 s with warm pipeline caches.
- 2026-10-26: D8-D10 chosen (see above). Presenter refactor order: (1) device layer
  (`rhi::Device`, replacing `VulkanContext` leakage), (2) `Swapchain` + present timing,
  (3) post passes against the RHI, (4) slim `RunFrame`. Gate for each: `scripts/replay/
  replay_check.sh` (frames + presented picture) before every commit. Next: draft the RHI
  headers (`gpu/rhi/`) from what `renderer.cpp` and `presenter.cpp` actually use, for review
  before implementing.
- 2026-10-26: RHI headers drafted for review: `runtime/gpu/rhi/{types,device,command_list}.h`
  (510 lines, headers only, compile standalone, included by nothing yet). Built from the
  renderer's real use: ~25 formats, buffers (device / upload / cached), images (one view),
  samplers, `BindingLayout` + persistent `BindingSet` + pushed tables (D10), graphics and
  compute pipelines (created from SPIR-V, thread-safe), `CommandList` (rendering passes,
  state, draws, copies, clears, `Transition` with states per D9), `Fence`, `Event`, `QueryPool`,
  `Device::Submit`. Boundary decisions written in the `device.h` header comment: image block
  suballocation and the memory plan's driver parts move INSIDE the Vulkan backend; guest memory
  is `ImportHostMemory` with capability + alignment (Metal: 16 KB, newBufferWithBytesNoCopy);
  one shared queue with an internal lock replaces `VulkanContext::queueMutex`.
  Open items found while drafting: (a) `Event`/`SignalEvent` (the renderer's VkEvent write
  fences) has no cheap Metal equivalent: MTLSharedEvent between encoders costs an encoder
  break; (b) image views in a format other than the image's (a few resolves) are not in the
  draft; (c) the renderer reads vertex data from guest memory as a storage buffer (512 MB
  buffer; Metal's maxBufferLength must allow it, and the "split memory" path exists for
  devices that can't); (d) occlusion counting, fragment shading rate, multiview and depth/
  stencil resolve are not in the draft (the last two: check whether they are used at all);
  (e) pipeline cache blob format is per backend; (f) `Present`/swapchain come with step 2.
- 2026-10-26: draft fixes after the risk review: `Topology::TriangleFan` removed (the renderer
  converts fans/polygons to lists; Metal has none), `BindingSlot::immutableSampler` added (one
  sampler for all textures, bound once on Metal).
  SHADER-PATH SPIKE (throwaway, in the scratchpad, not in the repo): the 115 SPIR-V modules in
  `~/Library/Caches/SpeedBreaker/spirv` (real translated Xenos vertex/pixel shaders and the
  renderer's compute shaders: 46 vertex, 59 fragment, 10 compute) through SPIRV-Cross
  (`--msl --msl-version 30000`): 115 of 115 convert; then compiled with the system Metal
  compiler (`MTLDevice newLibraryWithSource`, Apple M5 Max; the offline `metal` tool is not
  installed: it needs `xcodebuild -downloadComponent MetalToolchain`, not done): 115 of 115
  compile, 3.3 s total, avg 28 ms, worst 267 ms (library only; pipeline-state creation is
  extra and not measured). Resource use: at most 4 buffers, 7 textures and 7 samplers in a
  shader (Metal: 31 / 128 / 16), no buffer-index collisions. Vertex shaders take the ring and
  the 512 MB guest memory as `const device` buffers, the constants as a 10 KB `constant`
  buffer (dynamic offset), the compute resolves guest memory as a device buffer plus a
  `texture2d<uint, access::write>`. So the D4 path (GLSL -> SPIR-V -> MSL) is viable for the
  whole current shader set; it says nothing yet about correct results (that is the replay
  check on a Metal backend) or pipeline-state creation time.
- 2026-10-26: the user installed the Metal Toolchain (`xcrun metal` 32023.921, air64-apple-darwin27.0.0).
  The same 115 MSL files compile with the offline compiler (`xcrun metal -std=metal3.0 -c`):
  115 of 115, 7 s. Agrees with the runtime compiler. Environment note: offline `metal`/`metallib`
  now available for shader debugging and for precompiled shader libraries later.
- 2026-10-26: presenter refactor STEP 1 done: the RHI's Vulkan backend exists for the
  buffer/compute/sync slice. `runtime/gpu/rhi/vulkan/vulkan_device.{h,cpp}` (~700 lines;
  built into the game by the runtime glob, used by nothing yet): `CreateHeadlessDevice()`
  makes an instance + device of its own (portability enumeration, push descriptors, host
  import, first discrete GPU else first device); implemented: buffers (Device / Upload /
  UploadCached), `ImportHostMemory` (alignment checked against the device's), samplers,
  `BindingLayout` / `BindingSet` (persistent, dynamic uniform offsets) / pushed tables,
  compute pipelines, pipeline cache save/load, command lists (buffer copies, fills, updates,
  compute, state, buffer transitions, events, queries), fences, events, query pools, `Submit`
  under an internal queue lock. Not implemented (abort with "not implemented yet", filled in
  when the presenter's post passes need them): images, graphics pipelines, rendering passes,
  clears, blits, image copies and transitions. Draft interface refinement: `WriteTimestamp`
  takes `end`.
  `tests/rhi_vulkan_test.cpp` (+ `rhi_vulkan_test_shaders.h`, four compute shaders as SPIR-V):
  real calls on the machine's device, headless, a skip without one. All pass on the M5 Max
  via MoltenVK: buffer copies and barriers; host-memory import (needs 16 KB alignment on
  Apple; a misaligned import is refused; the CPU's later writes are seen without re-import);
  a persistent set with dynamic uniform offsets (0 and 256) and `Update` retargeting one
  slot; a persistent + pushed table; push constants; an event signalled from inside a list
  is seen by the CPU 0.43 ms in while the list runs 64 ms (so mid-submission write fences
  work as the renderer needs, on MoltenVK at least); timestamps. Run: `ctest -R rhi_vulkan`
  in `build/main` (configured with -DSB_TESTS=ON). Replay check still passes (nothing uses
  the backend yet).
  Next (step 2): move the presenter's instance/device creation and swapchain/present timing
  onto the backend; the presenter then holds an `rhi::Device`.
- 2026-10-26: presenter refactor STEP 2a done: the presenter's device is made by the RHI's
  Vulkan backend. New `gpu/rhi/vulkan/vulkan_native.h`: `DeviceConfig` (instance, physical
  device, queue family, extension list, features, pNext chain: the policy the presenter decides
  from settings, env and the platform) -> `CreateDevice()` (the backend creates the VkDevice,
  gets the queue, reads the memory properties, makes the pipeline cache, fills
  `Capabilities`, owns the queue lock) and `GetNative()` (instance, physical, device,
  queue family, queue, queue mutex) for code that still speaks Vulkan. `CreateHeadlessDevice`
  now goes through the same function. In `presenter.cpp`: the `vkCreateDevice` + queue code is
  gone, `s_queueMutex` became `*s_queueMutexPtr` (a placeholder mutex until the device
  exists, then the backend's; the renderer's and ui's `VulkanContext::queueMutex` is the same
  lock as before in effect), `s_rhi` holds the device. Not changed: the presenter still
  creates the instance and the surface (SDL), chooses the GPU, builds the feature chain,
  and, as before, never destroys the device or instance (the process exits). Gate: replay
  check passes on both captures (frames and presented picture), `rhi_vulkan_test` passes.
  Next (2b): the swapchain, its images and the present timing behind an `rhi::Swapchain`.
- 2026-10-26: backend graphics half done (before the swapchain, which needs it): images
  (`CreateImage`: dedicated device-local allocation each, one view over all of it; the renderer's
  block suballocation comes with its migration), image states -> layout + stage + access
  (D9), image copies, blits, clears, `BeginRendering`/`EndRendering` (dynamic rendering;
  the headless device enables it with the extensions MoltenVK needs), `ClearAttachments`, image
  bindings (sampled with the layout's immutable sampler or one given at bind time; storage),
  `CreateGraphicsPipeline` (blend, depth, stencil, raster state; viewport, scissor, stencil
  values, depth bias and blend constants dynamic, as the renderer has them). Nothing in the
  interface is unimplemented now. `rhi_vulkan_test` gained: depth test rejection, alpha blend,
  a partial `ClearAttachments` in a pass, a sampled 4x4 texture through a pushed binding with an
  immutable sampler (texel for texel), image clear + blit + copy round trip: all pass on the M5
  Max. Replay check unchanged (the game doesn't use these yet).
  Next: `rhi::Swapchain` (images, acquire/present, present timing) and the presenter's
  post passes on top.
- 2026-10-26: swapchain interface drafted (D11) for review; nothing implements it yet.
- 2026-10-26: D11 implemented. `VkSurfaceObject`/`VkSwapchainObject` in the Vulkan backend:
  acquire/present semaphores, the present-wait thread (owned by the swapchain, joined before it
  is destroyed, so `video/swapchain_use.h` and its test are gone), driver reports polled after
  each wait and sent as `PresentReport`, `notBefore` converted to display timing's
  desiredPresentTime. The presenter now builds its swapchain through `Device::CreateSwapchain`,
  acquires/presents through it, and keeps its pacing policy (callbacks `OnShown`, `OnReport`). Its
  command buffers are still raw Vulkan (transitional accessors `NativeImage/NativeView/
  NativeSubmitSemaphores/NativeFormat` in vulkan_native.h, to go when the post passes move to the
  RHI); it takes the queue lock for its submit only, Present takes it inside. Every present now
  has an id. `NFSMW_VBLANK_LOG`'s `.actual.csv` holds steady-clock ns now (was Metal host time).
  Checks: replay check PASSED (incl. the presented picture), ctest 71/71 (quit_game_test fixed:
  it scanned main()'s loop to the first `}`, which the replay exit had broken), live run 50 s:
  presents timed, refresh fit valid (16.668 ms), and with NFSMW_PRESENT_AT=12.5 907 presents
  shown at a time, 823 driver reports, no watchdog trip.
  Next: the post passes against the RHI (AA, FSR1, tone, scale), then slim `RunFrame`.
- 2026-10-26: presenter step 1 of the post-pass port: each frame slot has an `rhi::CommandList`
  and `rhi::Fence` (`CreateFence(signaled)` added); submit is `SubmitToSwapchain`, so the
  presenter's raw `vkQueueSubmit`, queue lock, wait stages and acquire/render-done semaphore
  accessors are gone. Its commands are still recorded raw into the list's native command
  buffer (`NativeCommandBuffer`). Replay check PASSED, rhi/quit tests pass.
- 2026-10-26: post-processing moved onto the RHI (steps 2 and 3 of the port together: the
  scale/tone and AA/EASU passes share one set of state and one render pass, so splitting them
  would only have meant writing the glue twice). New `video/post_process.{h,cpp}`
  (`PostProcessor`): the GLSL stays inline (D12) and is compiled with `CompileShader`; AA,
  EASU, RCAS, scale and tone pipelines, the tone table and the AA/FSR images are RHI objects,
  and the passes are recorded on the frame's `CommandList` (`Transition` replaces the
  hand-written barriers; an old state is given to each transition so a write after last
  frame's read waits for it). The presenter keeps the source image (the renderer's raw front
  image, wrapped with the transitional `vulkan::WrapImage`, moved GENERAL -> SHADER_READ_ONLY
  -> GENERAL around the passes with raw barriers), the swapchain pass, UI, captures and the
  NFSMW_UPSCALE_COMPARE shots (still raw Vulkan, step 4). To check it: `scripts/replay/
  post_check.sh` replays both captures under six settings sets (default, AA off, bilinear,
  FSR 1, tone, FSR 1 + tone), each with its own baselines recorded from the build before the
  port: all 12 presented pictures are bit-identical. Not covered by replay: the CPU-path
  source (`s_source`, no renderer image) and the compare tool. ctest 71/71 (fsr1_test and
  display_tone_test read the new file now); a 30 s live run shows FSR 1 at 60 fps.
  Next: readbacks (screenshots, cover check, region capture, compare shots), then slim RunFrame.

