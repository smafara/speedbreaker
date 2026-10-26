# metal-cpp (vendored)

Apple's header-only C++ interface to Metal, from https://github.com/apple/metal-cpp at commit
8cdde68ae36a636c84cbb48ebcbc8f3b60251ea0 (2026-09-22; "macOS 27.0.1, iOS 27.0.1"), unmodified:
the `Foundation/`, `Metal/`, `MetalFX/` and `QuartzCore/` folders (its `SingleHeader/` is left
out: it is generated from these), Apache License 2.0 (`LICENSE.txt` as it is; also
`LICENSES/metal-cpp-Apache-2.0.txt`). `UPSTREAM-README.md` is the upstream README
(memory management rules, `NS::SharedPtr`).

Used by the Metal backend of the RHI (`runtime/gpu/rhi/metal/`, DECISIONS.md D13), on Apple
platforms only. Exactly one source file defines `NS_PRIVATE_IMPLEMENTATION`,
`MTL_PRIVATE_IMPLEMENTATION` and `CA_PRIVATE_IMPLEMENTATION` before including them
(`runtime/gpu/rhi/metal/metal_cpp.cpp`).

SHA-256 of the sorted per-file SHA-256 list of the four folders:

    b8f516baba53407098cb27cb914305c0bd6d25e3923d84d338a808a4d43bfe75  Foundation Metal MetalFX QuartzCore
    f4e92c7fe2aa066e294c0da9d08683dc9fb741ee6e6debbfad99a0c96e07c1f2  LICENSE.txt
