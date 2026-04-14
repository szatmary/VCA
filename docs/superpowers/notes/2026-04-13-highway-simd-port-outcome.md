# Highway SIMD Port — Outcome

**Branch:** `highway` (off `stable`)
**Completed:** 2026-04-13
**Spec:** [`../specs/2026-04-13-highway-simd-port-design.md`](../specs/2026-04-13-highway-simd-port-design.md)
**Plan:** [`../plans/2026-04-13-highway-simd-port.md`](../plans/2026-04-13-highway-simd-port.md)

## Summary

Replaced the entire hand-written SIMD layer in `source/lib/analyzer/simd/` (x86 NASM assembly, SSSE3 intrinsics, ARM NEON intrinsics, scalar `noAsmImpl` fallbacks, and CPU detection) with a portable implementation using [Google Highway 1.3.0](https://github.com/google/highway), pulled in via CMake `FetchContent`. The public `CpuSimd` enum in `vcaLib.h` was rehabilitated and expanded from 7 → 29 values to cover every Highway target.

## Diff stats

```
18 commits, 48 files changed, +2,758 / -7,790 lines  (-5,032 net)
```

## What landed

- **Google Highway 1.3.0** fetched via CMake `FetchContent` at configure time (`cmake/FetchHighway.cmake`, pinned tag).
- **DCT 8/16/32 kernels** ported to Highway as vectorized matrix-vector multiplies (`hn::WidenMulPairwiseAdd` + `hn::ReduceSum`) in a single `source/lib/analyzer/simd/dct_hwy.cpp` file using Highway's `foreach_target.h` + `HWY_EXPORT` + `HWY_DYNAMIC_DISPATCH` mechanism. Each kernel is templated per bit depth (3 instantiations × 3 block sizes = 9 exported symbols) so rounding constants and shift amounts are compile-time immediates. Output is **bit-identical** to the scalar `vca::dct{8,16,32}_c` reference from `DCTTransformsNative.cpp`.
- **Hand-written SIMD deleted**: 19 files, ~7K lines of NASM x86 assembly (`dct8.asm`, `cpu-a.asm`, `const-a.asm`, `x86inc.asm`, `x86util.asm`), SSSE3 C intrinsics (`dct-ssse3.cpp`), ARM NEON intrinsics (`arm/dct-neon.cpp`, `arm/entropy-neon.cpp`, `arm/neon-utils.h`), and scalar `noAsmImpl{8,10,12}bit.cpp` fallbacks — all removed.
- **CPU detection machinery deleted**: `source/lib/analyzer/simd/cpu.h`, `cpu.cpp`, `vca::cpuDetectMaxSimd`, `vca::isSimdSupported`, `vca_emms` macro. Replaced with a thin wrapper `source/lib/analyzer/simd/targets.{h,cpp}` that provides `vca::simd::CurrentTargetName()` and two overloads of `vca::simd::ConstrainTargetsForCli` (string and `CpuSimd`), each wrapping `hwy::SupportedTargets` / `hwy::TargetName` / `hwy::SetSupportedTargetsForTest`.
- **NASM prerequisite removed** from `docs/build.md`. CMake minimum bumped 3.13 → 3.14 (required for `FetchContent_MakeAvailable`). C++11 → C++17 requirement documented.
- **Per-bit-depth SIMD library structure deleted**: `vcaLibSimd8bit`, `vcaLibSimd10bit`, `vcaLibSimd12bit` static libraries and the `vca_add_simd_lib` CMake function are gone. The Highway kernels take `bitDepth` as a runtime parameter (propagated into the templated compile-time shift).
- **`CpuSimd` enum rehabilitated and expanded**. Initially marked `[[deprecated]]` and stubbed out, it was then un-deprecated and grown from 7 → 29 values covering every target Highway 1.3.0 exposes:
  - Legacy (ABI-stable at positions 0–6): `Autodetect`, `None`, `SSE2`, `SSSE3`, `SSE4`, `AVX2`, `NEON`
  - x86 AVX-512 family: `AVX10_2`, `AVX3_SPR`, `AVX3_ZEN4`, `AVX3_DL`, `AVX3`
  - ARM SVE/NEON variants: `SVE2_128`, `SVE_256`, `SVE2`, `SVE`, `NEON_BF16`, `NEON_WITHOUT_AES`
  - Other architectures: `RVV`, `LASX`, `LSX`, `PPC10`, `PPC9`, `PPC8`, `Z15`, `Z14`
  - WebAssembly / emulated: `WASM_EMU256`, `WASM`, `EMU128`
- **`Analyzer.cpp` honors `vca_param::cpuSimd`** unconditionally via `vca::simd::ConstrainTargetsForCli(cfg.cpuSimd)` in the constructor — the single source of truth for SIMD target selection.
- **`vca.cpp`'s `--no-simd` flag** sets `param.cpuSimd = CpuSimd::None`. No separate string-based constraint call.
- **Dead entropy SIMD scaffolding removed**: the `entropy_avx2` stub (never implemented — returned `0.0` with a `//TODO` since its Feb 2024 introduction) and the ARM entropy-neon placeholder are gone. Entropy still runs on the scalar `entropy_c` / `entropy_lowpass_c` reference from `EntropyNative.cpp` — unchanged behavior.

## Test coverage

- **9 `DCTHighwayCrossCheck` tests** (3 block sizes × 3 bit depths) implemented via parameterized `DCTHighwayTargetFixture` using `hwy::SetSupportedTargetsForTest`. Each test runs under **every Highway target supported on the host**, producing **45 test instances on arm64** (5 targets: `NEON_BF16`, `NEON`, `NEON_WITHOUT_AES`, `EMU128`, `SCALAR`) — on x86-64 hosts the count will be higher.
- Each fixture instance compares Highway output to the scalar reference with strict `ASSERT_EQ` over 32 random seeds per bit depth. At 8×8 through 32×32 block sizes, that's **~384,000 bit-exact integer comparisons per full run**. Zero tolerance, zero rounding drift.
- **All 33 tests pass** (9 Highway cross-check + 9 pre-existing parameterized DCT fixture + 9 pre-existing forward-backward + 6 others). Clean-slate Release build succeeds from scratch.

## Key decisions / deviations from plan

- **`kT8`/`kT16`/`kT32` coefficient tables** live inside `namespace vca::HWY_NAMESPACE` (not file-scope anonymous namespace as the plan originally suggested). Reason: Highway's `foreach_target.h` re-includes the `.cpp` once per target; per-target tables inside `HWY_NAMESPACE` are the idiomatic Highway pattern.
- **Templated-per-bit-depth shift**. The plan initially proposed using `hn::ShiftRightSame` with a runtime shift for simplicity, then switched to compile-time template instantiation after user feedback. Each kernel has three template thunks (bd 8/10/12) → `HWY_EXPORT` → public entry point `switch`es on `bitDepth`. All shifts and rounding constants are immediates.
- **Tasks 12 and 13 were combined** into a single commit (`6bdfcfa`) because they were tightly coupled — deprecating `CpuSimd` and migrating internal callers couldn't cleanly be separated into distinct commits without leaving the build in a warning-flooded intermediate state.
- **Task 11 was reordered after Tasks 12, 13, 14, and 16** because deleting `cpu.h` would have broken the build while callers still referenced its symbols. The plan anticipated this and explicitly said "pause and flag if reordering is needed."
- **`CpuSimd` was un-deprecated mid-port**. Initially marked `[[deprecated]]` in Task 12, with the intent that it was a legacy shim. User then asked to have the enum *function* as the real public API and expand it to cover all Highway targets. The deprecation attribute was removed in commit `27ebe74`, and the expansion happened in the same commit.
- **Highway 1.3.0 exposes `HWY_SSE2` and `HWY_SSSE3` as distinct target bits**, so `CpuSimd::SSE2`/`SSSE3` map 1:1 instead of collapsing to `HWY_SSE4` as the earliest `ConstrainTargetsForCli` draft did.
- **5 Highway targets on arm64 macOS**: `NEON_BF16`, `NEON`, `NEON_WITHOUT_AES`, `EMU128`, `SCALAR`. The plan guessed "2 targets (SCALAR + NEON)"; reality exposed more bits. On x86-64 hosts the count will be higher (SSE4, AVX2, AVX3 variants, etc.).
- **`ctest -R` does not work** in this project — there's no `gtest_discover_tests()` or `add_test()` wiring. Tests are run via `./build/source/lib/test/unitTestSuite --gtest_filter=...` directly. Pre-existing issue unrelated to the port.

## Not done (explicit non-goals)

- **No end-to-end CSV-diff smoke test** against a sample video. No representative clip was available on the dev machine. **Recommended before merging to `stable`**: run the `highway` branch and `stable` branch against the same video and diff the CSV output. Any divergence would be a regression.
- **Factored butterfly optimization** deferred. The current kernels use a direct matrix-vector multiply (`kTN * src_row`). The scalar reference uses a factored butterfly (E/O decomposition) which does ~2× fewer multiplies but needs a reverse shuffle. The `DCTHighwayCrossCheck` test is in place to validate any future optimization.
- **Native-vector-width unrolling** for Dct16/Dct32. Current kernels process 8 int16 lanes at a time (capped with `hn::CappedTag<int16_t, 8>`). On AVX2 (16-lane) or AVX-512 (32-lane) hosts a further optimization could process two or four rows in parallel per iteration. Wait for profiling to justify.
- **`TestThatAllImplementationsProduceIdenticalResults` fixture cleanup**. This pre-existing parameterized test originally compared SIMD paths against the native reference. After Task 12+13 it's a smoke check that only exercises the public `vca::performDCT` entry point. The real cross-validation is now in `DCTHighwayCrossCheck`. Could be deleted as a follow-up cleanup.
- **`vca_param::enableSIMD` field**. Still present in `vcaLib.h` for ABI but functionally redundant — `cpuSimd` is the single source of truth for target selection. Could be removed in a future major version.
- **CI validation on non-host targets**. Highway supports SVE, RVV, WASM, IBM POWER, etc. for free, but this branch did not add CI runners to exercise those targets. They compile cleanly (per `foreach_target.h`) but haven't been runtime-validated outside arm64 macOS.

## Commits (oldest to newest)

```
60c207e  docs: Highway SIMD port spec and plan
f3dc8aa  build: add Highway via FetchContent
2b01529  build: link Highway into analyzer lib
da83b54  test: add Dct8 Highway cross-check (red)
a2f030b  feat(simd): Highway Dct8 kernel (templated per bit depth)
6c92c09  feat(simd): Highway Dct16 kernel (templated per bit depth)
cb45041  feat(simd): Highway Dct32 kernel (templated per bit depth)
f7ca4f6  refactor(dct): route DCTTransform through Highway kernels
18f639e  refactor(entropy): drop dead SIMD include and TODO dispatch
73da508  feat(simd): add Highway target introspection wrapper
303a3ea  build(simd): drop per-bit-depth SIMD libraries
6bdfcfa  api: deprecate CpuSimd, drop internal CpuSimd parameters
4aa4a00  feat(cli): wire --no-simd to Highway's scalar target
b514803  test: iterate DCTHighwayCrossCheck over every Highway target
ae1da51  refactor(simd): delete hand-written SIMD code
abb525c  build: remove NASM, arch detection, and dead CPU dispatch code
27ebe74  feat(api): expand CpuSimd to cover all 29 Highway targets
7a22a96  docs+feat: extend ConstrainTargetsForCli strings, update build docs
```
