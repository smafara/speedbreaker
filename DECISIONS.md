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
