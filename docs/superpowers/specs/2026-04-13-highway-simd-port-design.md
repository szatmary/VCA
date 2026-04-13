# Highway SIMD Port — Design Spec

**Date:** 2026-04-13
**Branch:** `highway` (off `stable`)
**Status:** Design approved, pending plan

## Goal

Replace the entire hand-written SIMD layer in `source/lib/analyzer/simd/` (x86 assembly, SSSE3 intrinsics, ARM NEON intrinsics, and scalar fallbacks) with a single portable implementation using [Google Highway](https://github.com/google/highway). Maintain backward compatibility of the public `vcaLib.h` API via deprecated shims.

## Scope Clarifications (found during fact-check)

- **Only DCT kernels are ported.** The native reference has three separate DCT implementations (`dct8_c`, `dct16_c`, `dct32_c`), each with its own butterfly coefficients. The Highway port mirrors this structure with three kernels (`Dct8`, `Dct16`, `Dct32`) — not a single 8×8 building block.
- **No entropy Highway kernel.** Investigation of the existing entropy SIMD (`source/lib/analyzer/simd/entropy.{cpp,h}` and `arm/entropy-neon.cpp`) shows it was never implemented — `entropy_avx2` is a `//TODO` stub returning `0.0`, and its call site in `EntropyCalculation.cpp:49-52` has always been commented out. The ARM NEON entropy file is an explicit placeholder. Entropy runtime has always used the native `entropy_c` / `entropy_lowpass_c` functions from `EntropyNative.cpp`. **This port deletes the dead entropy SIMD scaffolding and leaves entropy running on the native implementation unchanged.** No Highway entropy kernel is written.

## Motivation

- The current layer has three parallel implementations of every kernel (x86 asm, x86 intrinsics, ARM NEON) plus a scalar `noAsmImpl` fallback. ~7,000 lines of code to maintain.
- Adding a new architecture (RISC-V, SVE, WASM) requires writing a fourth implementation.
- Highway provides runtime dispatch across SSE4/AVX2/AVX-512/NEON/SVE/RVV/WASM/scalar from one source, compiled multiple times automatically via `HWY_EXPORT` / `HWY_DYNAMIC_DISPATCH`.
- The project's two SIMD kernels — DCT 8×8 and entropy calculation — are both expressible in Highway's op set.

## Scope

### Deleted from `source/lib/analyzer/simd/`

- All `.asm` files: `dct8.asm`, `cpu-a.asm`, `const-a.asm`, `x86inc.asm`, `x86util.asm`
- `cpu.cpp`, `cpu.h` (replaced by thin `targets.{h,cpp}` wrapper around Highway)
- `dct-ssse3.{cpp,h}`, `dct8.h`
- `entropy.{cpp,h}`
- `noAsmImpl8bit.cpp`, `noAsmImpl10bit.cpp`, `noAsmImpl12bit.cpp` (Highway's `HWY_SCALAR` target replaces these)
- `arm/dct-neon.{cpp,h}`, `arm/entropy-neon.cpp`, `arm/neon-utils.h` (entire `arm/` subdirectory)
- `README.txt` updated to document the Highway-based layer

### Created in `source/lib/analyzer/simd/`

- `dct_hwy.{h,cpp}` — portable DCT 8/16/32 kernels (three independent implementations, mirroring the native reference)
<!-- entropy Highway kernel removed from scope; see "Scope Clarifications" -->
- `targets.{h,cpp}` — thin wrapper over `hwy::SupportedTargets()` / `hwy::TargetName()` for logging and CLI `--asm` control
- Updated `CMakeLists.txt` — no NASM, no arch-specific branches

### Modified outside `source/lib/analyzer/simd/`

- `source/lib/analyzer/DCTTransform.cpp` — drop `#if defined(VCA_ARCH_*)` blocks and `CpuSimd` switch dispatch; call `vca::Dct8x8()` directly (Highway handles target selection internally).
- `source/lib/analyzer/EntropyCalculation.cpp` — remove `#include <analyzer/simd/entropy.h>` and the commented-out `entropy_avx2` block. Keep the native path unchanged.
- `source/lib/vcaLib.h` — `CpuSimd` enum and all related APIs marked `[[deprecated]]`; new non-`CpuSimd` overloads added alongside.
- Root `CMakeLists.txt` — remove `enable_language(ASM_NASM)`, `find_package(NASM)`, `VCA_ARCH_X86` / `VCA_ARCH_ARM` arch detection; add Highway via `FetchContent`.
- `source/lib/test/DCTTestImplementationsIdenticalOutput.cpp` — rewrite to iterate over `hwy::SupportedTargets()` via `hwy::SetSupportedTargetsForTest`, comparing each to `DCTTransformsNative`.
<!-- No new entropy test: entropy has no SIMD implementation to cross-check. -->
- `docs/cli.md` — document updated `--asm` semantics.

## Architecture

### Highway integration (CMake)

```cmake
include(FetchContent)
FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG        <pinned-release-tag>   # resolved at branch creation — use the latest Highway stable release tag from https://github.com/google/highway/releases
)
set(HWY_ENABLE_TESTS    OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB  OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)

target_link_libraries(analyzer PRIVATE hwy)
```

Highway is a private dependency of the analyzer lib; its headers must not appear in any public VCA header. C++17 required (project already on it).

First configure needs network access; subsequent builds use the `_deps/` cache. Documented in `README.txt`.

### DCT kernels (`dct_hwy.cpp`)

The native reference (`DCTTransformsNative.cpp`) has three independent DCT implementations — `dct8_c`, `dct16_c`, `dct32_c` — each using its own `partialButterflyN` (two-pass row/column transform with per-size integer coefficient tables `g_t8`, `g_t16`, `g_t32`). The Highway port mirrors this: three separate kernels, one file.

Uses Highway's `foreach_target.h` mechanism to compile each kernel once per enabled target:

```cpp
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "analyzer/simd/dct_hwy.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace vca::HWY_NAMESPACE {
    namespace hn = hwy::HWY_NAMESPACE;

    void Dct8Impl (const int16_t* src, int16_t* dst, intptr_t srcStride, unsigned bitDepth);
    void Dct16Impl(const int16_t* src, int16_t* dst, intptr_t srcStride, unsigned bitDepth);
    void Dct32Impl(const int16_t* src, int16_t* dst, intptr_t srcStride, unsigned bitDepth);
}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace vca {
    HWY_EXPORT(Dct8Impl);
    HWY_EXPORT(Dct16Impl);
    HWY_EXPORT(Dct32Impl);

    void Dct8 (const int16_t* src, int16_t* dst, intptr_t srcStride, unsigned bitDepth) {
        HWY_DYNAMIC_DISPATCH(Dct8Impl)(src, dst, srcStride, bitDepth);
    }
    void Dct16(const int16_t* src, int16_t* dst, intptr_t srcStride, unsigned bitDepth) {
        HWY_DYNAMIC_DISPATCH(Dct16Impl)(src, dst, srcStride, bitDepth);
    }
    void Dct32(const int16_t* src, int16_t* dst, intptr_t srcStride, unsigned bitDepth) {
        HWY_DYNAMIC_DISPATCH(Dct32Impl)(src, dst, srcStride, bitDepth);
    }
}
#endif
```

- Signatures **exactly** match the native `dct{8,16,32}_c` — same parameter list — so `DCTTransform.cpp` can swap one call for another.
- Lane shape: `hn::CappedTag<int16_t, N>` for the per-row transform, where N = 8/16/32. This gives a fixed logical width regardless of target vector width; Highway handles cases where the target's native width is narrower than N by unrolling.
- Algorithm source of truth: `DCTTransformsNative.cpp` functions `partialButterfly8`, `partialButterfly16`, `partialButterfly32`. The Highway version ports the butterfly structure op-for-op using `hn::Add`, `hn::Sub`, `hn::MulAdd` against broadcast coefficients from `g_t8`/`g_t16`/`g_t32`.
- Output must be **bit-identical** to the native reference (validated by the cross-check test).

### Entropy (no Highway kernel)

Per the "Scope Clarifications" section, the existing entropy SIMD is dead scaffolding and is simply deleted. `EntropyCalculation.cpp` already calls `vca::entropy_c` / `vca::entropy_lowpass_c` from `EntropyNative.cpp`, and this does not change. The only modification to `EntropyCalculation.cpp` is removing the `#include <analyzer/simd/entropy.h>` line and the dead `//if (cpuSimd == CpuSimd::AVX2)` comment block.

### Target control & logging (`targets.{h,cpp}`)

Thin wrapper used only by:
- Startup logging ("Highway target: AVX2").
- CLI `--asm` flag parsing (calls `hwy::SetSupportedTargetsForTest()` *once* at startup to constrain Highway process-globally).

Not referenced by any hot path.

### Backward compatibility (`vcaLib.h`)

**`CpuSimd` enum** — kept but marked deprecated:

```cpp
enum class [[deprecated("CpuSimd is ignored; Highway chooses the runtime target. "
                        "Will be removed in a future version.")]]
CpuSimd { ... };
```

**Public DCT/entropy functions** — overloaded:

```cpp
// New primary overload — no CpuSimd parameter.
void performDCTBlockSize8 (unsigned bitDepth, int16_t* pixel, int16_t* coeff);
void performDCTBlockSize16(unsigned bitDepth, int16_t* pixel, int16_t* coeff);
void performDCTBlockSize32(unsigned bitDepth, int16_t* pixel, int16_t* coeff);

// Deprecated shims forwarding to the new overload.
[[deprecated("CpuSimd parameter is ignored; call the overload without it.")]]
inline void performDCTBlockSize8 (unsigned bitDepth, int16_t* pixel, int16_t* coeff, CpuSimd) {
    performDCTBlockSize8(bitDepth, pixel, coeff);
}
// (and 16, 32)
```

**`isSimdSupported()` / `cpuDetectMaxSimd()`** — these live in the internal `source/lib/analyzer/simd/cpu.h` (not `vcaLib.h`), so they have no external consumers. They are deleted along with `cpu.{h,cpp}`. Internal call sites are rewritten to use Highway directly.

**`vca_emms()` macro** — preserved for preprocessor compatibility, but routed through a deprecated inline function so call sites produce warnings:

```cpp
[[deprecated("vca_emms is a no-op under Highway; remove calls.")]]
inline void vca_emms_deprecated() noexcept {}
#define vca_emms() ::vca::vca_emms_deprecated()
```

Macro semantics preserved; compiler inlines to nothing in release; deprecation warning fires at every call site.

**Internal call sites** (inside VCA itself) are updated to the new signatures, so the VCA build is warning-clean. Only external consumers of `vcaLib.h` see the deprecation warnings — signaling the required migration without breaking them.

**ABI impact**: adding overloads and `[[deprecated]]` attributes preserves ABI. Existing binaries linking against the old symbols continue to resolve.

### CLI `--asm` flag

The existing `--asm` flag lets the user force a SIMD level for testing. Retained with remapped semantics:

| CLI value       | Action                                         |
|-----------------|------------------------------------------------|
| `auto` / unset  | Highway default (best supported)               |
| `none`          | `hwy::SetSupportedTargetsForTest(HWY_SCALAR)`  |
| `sse2`/`ssse3`/`sse4` | `HWY_SSE4`                               |
| `avx2`          | `HWY_AVX2`                                     |
| `neon`/`neon_dotprod` | `HWY_NEON`                               |

Called once at startup in `vca.cpp` before any analyzer work begins. Documented in `docs/cli.md`.

## Testing strategy

### Primary: cross-check against native reference

**DCT** — extend `source/lib/test/DCTTestImplementationsIdenticalOutput.cpp`:

- Iterate over `hwy::SupportedTargets()` using `hwy::SetSupportedTargetsForTest(target)` in a test fixture.
- For each target, feed the same random input block through `DCTTransformsNative` (scalar reference) and `vca::Dct8x8` (Highway via dynamic dispatch).
- `ASSERT_EQ` element-wise on the 8×8 coefficient output.
- Cover bit depths 8, 10, 12; block sizes 8, 16, 32; multiple random seeds.

**Entropy** — new file `source/lib/test/EntropyTestImplementationsIdenticalOutput.cpp` following the same pattern, comparing Highway entropy to `EntropyNative`.

### Secondary: end-to-end smoke test

Before merge, run VCA against a sample video on both `stable` and `highway` branches and diff the CSV output. Any divergence is a regression. Not automated in the test suite — developer runs manually as a final gate.

### Bit-exactness requirement

Highway implementations must produce **bit-identical** output to the scalar native reference. Any difference is a bug. DCT uses integer math throughout; entropy's scalar log/divide parts are computed the same way as the native reference.

## Build system changes

- Remove: `enable_language(ASM_NASM)`, `find_package(NASM)`, `NASM_ENABLED`, `ENABLE_ASSEMBLY`, `VCA_ARCH_X86`, `VCA_ARCH_ARM`, `VCA_DISABLE_SIMD`.
- Remove: any conditional compilation guards (`#if defined(VCA_ARCH_*)`) in non-SIMD files — audit and clean up where they exist.
- Add: Highway via `FetchContent` as shown above.
- `--asm` CLI flag handling moves from arch-conditional code to unconditional Highway target control.

## Out of scope

- Performance tuning. Goal is correctness parity; any measurable perf delta from Highway vs. the old asm is accepted in v1. A follow-up branch can profile and optimize.
- New SIMD kernels beyond DCT 8×8 and entropy.
- SVE / RVV / WASM validation — Highway supports them for free, but CI won't exercise them in this branch.
- Removing the deprecated shims in `vcaLib.h`. That's a future major-version change.

## Risks

- **Bit-exactness in entropy scalar log/divide**: the native reference may have subtle rounding behavior. Mitigation: the Highway version uses the *same* scalar code for those parts — copied, not rewritten.
- **Highway `foreach_target.h` macro discipline**: the `HWY_TARGET_INCLUDE` / `HWY_ONCE` pattern is finicky. Mitigation: follow Highway's own examples verbatim; test builds on all targets early.
- **Deprecation warnings failing CI**: if VCA's CI treats warnings as errors, deprecated shims would break the build. Mitigation: internal call sites are updated to new signatures so no warnings fire in the VCA build itself.
- **NASM removal breaking unrelated build paths**: `NASM_ENABLED` may be referenced in places we haven't audited yet. Mitigation: grep for the macro during implementation, audit all references.

## Open questions (resolved during brainstorm)

- ~~Dependency management~~ → FetchContent
- ~~Dispatch strategy~~ → Dynamic dispatch via `HWY_EXPORT`
- ~~Testing approach~~ → Cross-check against native reference, with E2E smoke test as secondary gate
- ~~Directory naming~~ → Keep `source/lib/analyzer/simd/`
- ~~API compatibility~~ → Keep public API via deprecated shims with compiler warnings
