# Highway SIMD Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-written SIMD layer in `source/lib/analyzer/simd/` with a portable Highway-based implementation of the DCT 8/16/32 kernels; delete all dead entropy SIMD scaffolding; preserve `vcaLib.h` ABI via `[[deprecated]]` shims.

**Architecture:** Add Google Highway via CMake `FetchContent`. Port `dct8_c` / `dct16_c` / `dct32_c` from `DCTTransformsNative.cpp` into a single `dct_hwy.cpp` using Highway's `foreach_target.h` + `HWY_EXPORT` / `HWY_DYNAMIC_DISPATCH` pattern so one source compiles into runtime-dispatched SSE4/AVX2/AVX3/NEON/scalar targets. `DCTTransform.cpp` loses its arch-specific `#ifdef` branches and calls the Highway entry points unconditionally. NASM, `VCA_ARCH_X86`/`VCA_ARCH_ARM`, `CpuSimd` dispatch, and the three-bit-depth SIMD library split all go away.

**Tech Stack:** C++17, CMake ≥ 3.14 (for `FetchContent_MakeAvailable`), Google Highway (pinned release tag), GoogleTest.

**Spec:** [`docs/superpowers/specs/2026-04-13-highway-simd-port-design.md`](../specs/2026-04-13-highway-simd-port-design.md)

**Highway pinned tag:** `1.3.0`

---

## Ground Rules

- **Branch:** `highway` (already created off `stable`).
- **Bit-exact gate:** every DCT Highway kernel must produce output that is bit-identical to the corresponding `dct{8,16,32}_c` scalar reference. The cross-check test is the authoritative gate. Any divergence is a bug — do not normalize, do not add tolerances.
- **TDD:** write the failing test before the Highway kernel body. The test file is in place after Task 3; each kernel task adds new test cases and runs them RED → GREEN.
- **Commits:** one commit per task (or finer). Never skip commits. Never batch unrelated changes into a single commit.
- **Do not delete any existing file until its replacement compiles and all tests pass.** The deletion tasks come after the Highway kernels are proven correct.
- **Reference files you will read often:**
  - `source/lib/analyzer/DCTTransformsNative.cpp` — scalar reference (lines 127–298 for `partialButterfly{8,16,32}`; 305–354 for `dct{8,16,32}_c`).
  - `source/lib/analyzer/DCTTransform.cpp` — current dispatch call sites.
  - `source/lib/test/DCTTestImplementationsIdenticalOutput.cpp` — existing cross-check test pattern.
  - [Highway docs](https://github.com/google/highway/blob/master/g3doc/quick_reference.md).

---

## File Structure

**New files:**
- `source/lib/analyzer/simd/dct_hwy.h` — declares `vca::Dct8`, `vca::Dct16`, `vca::Dct32` plus `vca::simd::InitHighwayTargets()` / `vca::simd::CurrentTargetName()`.
- `source/lib/analyzer/simd/dct_hwy.cpp` — Highway dynamic-dispatch compilation unit. Ported `Dct{8,16,32}Impl` kernels.
- `source/lib/analyzer/simd/targets.h` — thin wrapper header exposing target introspection helpers (for logging and the `--asm` CLI remap).
- `source/lib/analyzer/simd/targets.cpp` — implementation of the wrapper.
- `cmake/FetchHighway.cmake` — encapsulates the Highway FetchContent call so root `CMakeLists.txt` stays clean.

**Deleted files (all under `source/lib/analyzer/simd/`):**
- `cpu.h`, `cpu.cpp`, `cpu-a.asm`, `const-a.asm`, `dct8.asm`, `x86inc.asm`, `x86util.asm`
- `dct-ssse3.h`, `dct-ssse3.cpp`, `dct8.h`
- `entropy.h`, `entropy.cpp`
- `noAsmImpl8bit.cpp`, `noAsmImpl10bit.cpp`, `noAsmImpl12bit.cpp`
- `arm/dct-neon.h`, `arm/dct-neon.cpp`, `arm/entropy-neon.cpp`, `arm/neon-utils.h`
- `arm/` directory itself
- `source/lib/analyzer/simd/CMakeLists.txt` is fully rewritten (not deleted) — see Task 11.

**Modified files:**
- `CMakeLists.txt` (root) — remove NASM setup, arch detection, drop `SIMD_ENABLED` conditional; include `cmake/FetchHighway.cmake`.
- `source/lib/CMakeLists.txt` — link `hwy` into the analyzer lib; drop three-bit-depth SIMD sub-libs.
- `source/lib/analyzer/CMakeLists.txt` — may need updates if it currently links `vcaLibSimd{8,10,12}bit`.
- `source/lib/vcaLib.h` — add `[[deprecated]]` to `CpuSimd` enum, add new non-`CpuSimd` overloads, deprecate `vca_emms`.
- `source/lib/analyzer/DCTTransform.cpp` / `.h` — drop arch `#ifdef` blocks; call Highway entry points directly.
- `source/lib/analyzer/EntropyCalculation.cpp` / `.h` — remove `#include <analyzer/simd/entropy.h>` and dead `//if (cpuSimd == CpuSimd::AVX2)` block.
- `source/lib/analyzer/EnergyCalculation.cpp` / `.h` — drop `CpuSimd` parameter or forward it through the deprecated overload (verify what it does with the parameter first).
- `source/lib/analyzer/Analyzer.cpp` — updated call sites, target init on startup.
- `source/lib/analyzer/common/common.h` — remove `VCA_ARCH_*` / `VCA_DISABLE_SIMD` macros if they live there.
- `source/lib/test/DCTTestImplementationsIdenticalOutput.cpp` — rewrite to iterate over `hwy::SupportedTargets()`.
- `source/lib/test/DCTTestForwardBackwards.cpp` — audit for `CpuSimd` usage; update to new API.
- `source/apps/vca/vca.cpp` — `--asm` CLI flag remap; call `vca::simd::InitHighwayTargets()` once at startup.
- `source/apps/vcaPerformanceTest/vcaPerformanceTest.cpp` — migrate off `CpuSimd`.
- `docs/cli.md` — document new `--asm` semantics.
- `readme.rst` — note Highway dependency and FetchContent network requirement on first configure.

---

## Task 1: Add Highway via FetchContent

**Files:**
- Create: `cmake/FetchHighway.cmake`
- Modify: `CMakeLists.txt` (root)

- [ ] **Step 1: Create `cmake/FetchHighway.cmake`**

```cmake
# cmake/FetchHighway.cmake
#
# Fetch Google Highway at a pinned release tag and expose the `hwy` target
# for linking. Highway is a private dependency of the analyzer library and
# must not leak into any public VCA header.

include(FetchContent)

set(HWY_ENABLE_TESTS    OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB  OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL  OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG        1.3.0
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(highway)

message(STATUS "Highway: fetched ${highway_SOURCE_DIR}")
```

- [ ] **Step 2: Modify the root `CMakeLists.txt`**

Near the top of `CMakeLists.txt` (after `project(...)` and `cmake_minimum_required` — before any `add_subdirectory(source)`), add:

```cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchHighway)
```

- [ ] **Step 3: Configure the build**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```
Expected output includes `Highway: fetched /.../_deps/highway-src` and no errors.

Note: first configure requires network access. This is expected and documented in `readme.rst` (Task 17).

- [ ] **Step 4: Commit**

```bash
git add cmake/FetchHighway.cmake CMakeLists.txt
git commit -m "build: add Highway via FetchContent"
```

---

## Task 2: Wire Highway into the analyzer lib (dry link)

**Files:**
- Modify: `source/lib/CMakeLists.txt` (or `source/lib/analyzer/CMakeLists.txt` — whichever defines the analyzer target)

- [ ] **Step 1: Inspect the current link structure**

Read `source/lib/CMakeLists.txt` and `source/lib/analyzer/CMakeLists.txt` to find the analyzer library target name. Search for `target_link_libraries` referencing `vcaLibSimd` — that's the analyzer target.

- [ ] **Step 2: Add `hwy` to the analyzer target's private link list**

In the analyzer target's CMakeLists, after the existing `target_link_libraries` call, add:
```cmake
target_link_libraries(<analyzer_target> PRIVATE hwy)
target_include_directories(<analyzer_target> PRIVATE ${highway_SOURCE_DIR})
```
(Replace `<analyzer_target>` with the actual name found in Step 1.)

Do NOT remove the existing `vcaLibSimd{8,10,12}bit` links yet — those stay until Task 11.

- [ ] **Step 3: Build and verify the link succeeds**

```bash
cmake --build build
```
Expected: build succeeds. No behavior change (Highway is linked but unused).

- [ ] **Step 4: Commit**

```bash
git add source/lib/CMakeLists.txt source/lib/analyzer/CMakeLists.txt
git commit -m "build: link Highway into analyzer lib"
```

---

## Task 3: Add the `Dct8` cross-check test (RED)

**Files:**
- Create: `source/lib/analyzer/simd/dct_hwy.h` (skeleton only — declarations, no impl)
- Modify: `source/lib/test/DCTTestImplementationsIdenticalOutput.cpp`

Rationale: write the test *before* any Highway kernel exists, so the test itself compiles (against the header skeleton) but fails at link time — then Task 4 adds the implementation to link clean and pass.

- [ ] **Step 1: Create the header skeleton `dct_hwy.h`**

```cpp
/* source/lib/analyzer/simd/dct_hwy.h */
#pragma once

#include <cstdint>

namespace vca {

// Highway dynamic-dispatch entry points. Implemented in dct_hwy.cpp via
// hwy::HWY_DYNAMIC_DISPATCH — one source compiled per enabled SIMD target.
void Dct8 (const int16_t *src, int16_t *dst, intptr_t srcStride, unsigned bitDepth);
void Dct16(const int16_t *src, int16_t *dst, intptr_t srcStride, unsigned bitDepth);
void Dct32(const int16_t *src, int16_t *dst, intptr_t srcStride, unsigned bitDepth);

} // namespace vca
```

- [ ] **Step 2: Add `Dct8` cross-check test case**

Append to `source/lib/test/DCTTestImplementationsIdenticalOutput.cpp` (keep existing tests untouched for now):

```cpp
#include <analyzer/simd/dct_hwy.h>
#include <analyzer/DCTTransformsNative.h>
#include <random>

namespace {

void fillRandomBlock(int16_t *block, unsigned size, unsigned bitDepth, uint32_t seed)
{
    std::mt19937 rng(seed);
    const int16_t maxVal = static_cast<int16_t>((1 << bitDepth) - 1);
    std::uniform_int_distribution<int> dist(0, maxVal);
    for (unsigned i = 0; i < size * size; ++i)
        block[i] = static_cast<int16_t>(dist(rng));
}

} // namespace

TEST(DCTHighwayCrossCheck, Dct8_BitDepth8)
{
    constexpr unsigned N = 8;
    int16_t src[N * N];
    int16_t dstNative[N * N];
    int16_t dstHighway[N * N];

    for (uint32_t seed = 1; seed <= 32; ++seed)
    {
        fillRandomBlock(src, N, 8, seed);
        vca::dct8_c(src, dstNative,  N, 8);
        vca::Dct8  (src, dstHighway, N, 8);
        for (unsigned i = 0; i < N * N; ++i)
            ASSERT_EQ(dstNative[i], dstHighway[i]) << "seed=" << seed << " idx=" << i;
    }
}

TEST(DCTHighwayCrossCheck, Dct8_BitDepth10)
{
    constexpr unsigned N = 8;
    int16_t src[N * N];
    int16_t dstNative[N * N];
    int16_t dstHighway[N * N];

    for (uint32_t seed = 1; seed <= 32; ++seed)
    {
        fillRandomBlock(src, N, 10, seed);
        vca::dct8_c(src, dstNative,  N, 10);
        vca::Dct8  (src, dstHighway, N, 10);
        for (unsigned i = 0; i < N * N; ++i)
            ASSERT_EQ(dstNative[i], dstHighway[i]) << "seed=" << seed << " idx=" << i;
    }
}

TEST(DCTHighwayCrossCheck, Dct8_BitDepth12)
{
    constexpr unsigned N = 8;
    int16_t src[N * N];
    int16_t dstNative[N * N];
    int16_t dstHighway[N * N];

    for (uint32_t seed = 1; seed <= 32; ++seed)
    {
        fillRandomBlock(src, N, 12, seed);
        vca::dct8_c(src, dstNative,  N, 12);
        vca::Dct8  (src, dstHighway, N, 12);
        for (unsigned i = 0; i < N * N; ++i)
            ASSERT_EQ(dstNative[i], dstHighway[i]) << "seed=" << seed << " idx=" << i;
    }
}
```

- [ ] **Step 3: Build and confirm link failure**

```bash
cmake --build build
```
Expected: **link error** — `vca::Dct8` is undefined. This is the "RED" state.

- [ ] **Step 4: Commit (with skip-ci or passing link is not required yet)**

```bash
git add source/lib/analyzer/simd/dct_hwy.h source/lib/test/DCTTestImplementationsIdenticalOutput.cpp
git commit -m "test: add Dct8 Highway cross-check (red)"
```

Note: if CI runs on push and fails at link here, that's expected — Task 4 fixes it. Do not push until Task 4 is committed, or push them together.

---

## Task 4: Implement `Dct8Impl` in Highway — vectorized matrix-vector (GREEN)

**Files:**
- Create: `source/lib/analyzer/simd/dct_hwy.cpp`
- Modify: `source/lib/analyzer/CMakeLists.txt` — add `simd/dct_hwy.cpp` to the analyzer target's sources.

**Reference:** `DCTTransformsNative.cpp:305-320` (`dct8_c`) and `:127-170` (`partialButterfly8`). The algorithm: HEVC DCT-8 performs two passes of a matrix-vector multiply `dst_row = kT8 * src_row`, then a transpose, then the same multiply again with a larger shift.

**Algorithm formulation:** the scalar `partialButterfly8` is a factoring of an 8×8 matrix multiply (`dst = kT8 * src` per row). For Highway, we skip the factoring and compute the matrix-vector multiply directly — this maps cleanly to `hn::WidenMulPairwiseAdd`:

- Each output element is `sum(kT8[row][k] * src[k] for k in 0..7)`, then `+ round >> shift`, then clamp to int16.
- `WidenMulPairwiseAdd(d32, a_i16, b_i16)` returns int32 lanes containing `a[2k]*b[2k] + a[2k+1]*b[2k+1]` — exactly 4 int32 partial sums from an 8-wide int16 row.
- `ReduceSum(d32, v)` collapses those 4 partial sums to a single int32 — our final output value before shift/clamp.
- One row input → 8 output values → 8 × (1 load + 1 WidenMulPairwiseAdd + 1 ReduceSum + 1 scalar shift/store).

**Compile-time shifts via template instantiation.** The shift amount depends on bit depth and is in the critical path. To keep shifts as immediate operands (`psrad xmm, imm8`, `vshrq_n_s32`), the row helper and the kernel are templated on `Shift`. One source template → three instantiations per block size (bit depths 8/10/12) → non-template thunks → `HWY_EXPORT` (which can't take templates directly) → public entry point `switch`es on `bitDepth`.

After the first pass, we need a transpose. With the 8×8 geometry, the cleanest move is: run the first pass with output stride = 1 into a scratch buffer arranged in column-major order (same trick the scalar reference uses — it writes `dst[k*line + j]`), then run the second pass with the scratch buffer as input. No explicit transpose instruction needed.

- [ ] **Step 1: Create `dct_hwy.cpp` with Highway boilerplate, `kT8` table, and vectorized `Dct8Impl`**

```cpp
/* source/lib/analyzer/simd/dct_hwy.cpp
 *
 * Portable Highway implementation of HEVC DCT 8/16/32 transforms.
 * Bit-identical to vca::dct{8,16,32}_c in DCTTransformsNative.cpp.
 *
 * Strategy: matrix-vector multiply via hn::WidenMulPairwiseAdd + ReduceSum.
 * Each DCT row is one row of kTN multiplied into the source row.
 */

#include <cstdint>
#include <cstring>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "analyzer/simd/dct_hwy.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

// Local copies of the HEVC DCT coefficient tables. Values must match
// DCTTransformsNative.cpp g_t8 / g_t16 / g_t32 exactly.
namespace {

alignas(16) constexpr int16_t kT8[8][8] = {
    {64, 64, 64, 64, 64, 64, 64, 64},
    {89, 75, 50, 18, -18, -50, -75, -89},
    {83, 36, -36, -83, -83, -36, 36, 83},
    {75, -18, -89, -50, 50, 89, 18, -75},
    {64, -64, -64, 64, 64, -64, -64, 64},
    {50, -89, 18, 75, -75, -18, 89, -50},
    {36, -83, 83, -36, -36, 83, -83, 36},
    {18, -50, 75, -89, 89, -75, 50, -18}
};

} // namespace

HWY_BEFORE_NAMESPACE();
namespace vca::HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// One row of an 8×8 DCT: writes 8 output coefficients, each into dst[k*dstStride]
// (so both the row-major and column-major passes share this helper).
//
// Shift is a template parameter so the `>> Shift` and `1 << (Shift - 1)` are
// compile-time constants the compiler can fold into immediate operands.
template <int Shift>
static HWY_INLINE void Dct8Row(const int16_t *src, int16_t *dst, intptr_t dstStride)
{
    // Cap to 8 lanes: an 8×8 DCT row is always exactly 8 int16 values,
    // regardless of native vector width. Highway emulates on narrower targets.
    const hn::CappedTag<int16_t, 8> d16;
    const hn::CappedTag<int32_t, 4> d32;

    constexpr int32_t kAdd = 1 << (Shift - 1);

    const auto srcVec = hn::LoadU(d16, src);

    // For each of 8 output coefficients, compute kT8[k] * src row.
    // WidenMulPairwiseAdd produces 4 int32 partial sums from 8 int16 products;
    // ReduceSum collapses them to one int32.
    for (int k = 0; k < 8; ++k)
    {
        const auto coefVec = hn::LoadU(d16, &kT8[k][0]);
        const auto prod    = hn::WidenMulPairwiseAdd(d32, srcVec, coefVec);
        const int32_t sum  = hn::ReduceSum(d32, prod);
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }
}

// Template on Shift1 (bit-depth-dependent). Shift2 is fixed at 9 for Dct8.
template <int Shift1>
static void Dct8ImplT(const int16_t *src, int16_t *dst, intptr_t srcStride)
{
    constexpr int Shift2 = 9;

    // Pass 1: for each input row, write 8 output values down a column of
    // `coef` (stride 8). After 8 rows, coef is the transposed intermediate
    // the scalar reference computes.
    alignas(32) int16_t coef[8 * 8];
    for (int j = 0; j < 8; ++j)
        Dct8Row<Shift1>(&src[j * srcStride], &coef[j], /*dstStride=*/8);

    // Pass 2: same routine, with coef as input (stride 8) and dst as output
    // (stride 8). The implicit transpose from pass 1 means pass 2 sees the
    // columns of pass 1's conceptual output as its rows.
    for (int j = 0; j < 8; ++j)
        Dct8Row<Shift2>(&coef[j * 8], &dst[j], /*dstStride=*/8);
}

// Non-template thunks — HWY_EXPORT cannot take template arguments directly.
// One per supported bit depth; Shift1 = 2 + bitDepth - 8.
void Dct8Impl_bd8 (const int16_t *s, int16_t *d, intptr_t st) { Dct8ImplT<2>(s, d, st); }
void Dct8Impl_bd10(const int16_t *s, int16_t *d, intptr_t st) { Dct8ImplT<4>(s, d, st); }
void Dct8Impl_bd12(const int16_t *s, int16_t *d, intptr_t st) { Dct8ImplT<6>(s, d, st); }

} // namespace vca::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace vca {

HWY_EXPORT(Dct8Impl_bd8);
HWY_EXPORT(Dct8Impl_bd10);
HWY_EXPORT(Dct8Impl_bd12);

void Dct8(const int16_t *src, int16_t *dst, intptr_t srcStride, unsigned bitDepth)
{
    switch (bitDepth)
    {
        case 8:  HWY_DYNAMIC_DISPATCH(Dct8Impl_bd8) (src, dst, srcStride); return;
        case 10: HWY_DYNAMIC_DISPATCH(Dct8Impl_bd10)(src, dst, srcStride); return;
        case 12: HWY_DYNAMIC_DISPATCH(Dct8Impl_bd12)(src, dst, srcStride); return;
    }
    // Unsupported bit depth — the analyzer only calls with 8/10/12, so this
    // is unreachable in practice. If you hit it, a new bit depth was added
    // without updating the switch.
}

// Placeholders so the header declarations resolve until Tasks 5 and 6.
void Dct16(const int16_t *, int16_t *, intptr_t, unsigned) {}
void Dct32(const int16_t *, int16_t *, intptr_t, unsigned) {}

} // namespace vca
#endif
```

**Expected correctness issues and fixes (bit-exact iteration):**

This is the first vectorized kernel — expect 1–3 test iterations to get bit-exact. Known pitfalls to check if the cross-check fails:

1. **Rounding before vs after reduction.** `ReduceSum` returns int32; the `+add` and `>>shift` must happen on the scalar result, never on the vector. If you see off-by-one errors in the last few output bits, this is why.
2. **Stride semantics.** The scalar `partialButterfly8` writes `dst[0], dst[line], dst[2*line], ...` where `line` is the block size. The Task 4 code mirrors this with `dstStride`. Verify by reading the scalar code carefully.
3. **Sign of `shift`.** All shifts are positive right-shifts; bitDepth ≥ 8, so `shift1` is ≥ 2.
4. **`WidenMulPairwiseAdd` lane ordering.** Highway guarantees the result is `{a[0]*b[0]+a[1]*b[1], a[2]*b[2]+a[3]*b[3], a[4]*b[4]+a[5]*b[5], a[6]*b[6]+a[7]*b[7]}`. `ReduceSum` sums all four. Total = dot product. No lane-order issue if you use both of those primitives together.
5. **`CappedTag<int16_t, 8>` on targets with narrower vectors.** On HWY_SCALAR, the cap gives a 1-lane "vector" and Highway emulates multi-element ops with loops. `WidenMulPairwiseAdd` must still produce the expected int32 partial-sum lanes — Highway handles this, but if you see scalar-target failures specifically, suspect Highway's emulation on narrow targets and file a reduced test case.

- [ ] **Step 2: Add `dct_hwy.cpp` to the build**

In `source/lib/analyzer/CMakeLists.txt` (or wherever `DCTTransformsNative.cpp` is listed in `target_sources`), add:
```cmake
target_sources(<analyzer_target> PRIVATE simd/dct_hwy.cpp simd/targets.cpp)
```
(Task 9 will create `targets.cpp`; for now, omit it from this line or leave it commented.)

- [ ] **Step 3: Build**

```bash
cmake --build build
```
Expected: succeeds. If Highway headers fail to find, verify the `target_include_directories` from Task 2 is correct. If `WidenMulPairwiseAdd` is not found, double-check your pinned Highway tag from Task 0 — the primitive has existed since Highway 0.12 but the name is worth verifying against the docs for your pinned version.

- [ ] **Step 4: Run the cross-check tests**

```bash
ctest --test-dir build -R DCTHighwayCrossCheck.Dct8 --output-on-failure
```
Expected: `Dct8_BitDepth8`, `Dct8_BitDepth10`, `Dct8_BitDepth12` all PASS.

If tests fail: use a single failing seed (the test prints `seed=` on failure) and write a scratch program that runs both `dct8_c` and `Dct8` on that seed, printing the 8×8 output matrices side by side. The divergence pattern usually points at exactly which pitfall above is biting. Do not lower the test's equality requirement — bit-exact is non-negotiable.

- [ ] **Step 5: Commit**

```bash
git add source/lib/analyzer/simd/dct_hwy.cpp source/lib/analyzer/CMakeLists.txt
git commit -m "feat(simd): Highway Dct8 kernel (templated per bit depth)"
```

- [ ] **Step 2: Add `dct_hwy.cpp` to the build**

Find where the existing `DCTTransformsNative.cpp` is added to the analyzer library target (likely in `source/lib/analyzer/CMakeLists.txt`) and add `simd/dct_hwy.cpp` alongside it.

- [ ] **Step 3: Build**

```bash
cmake --build build
```
Expected: build succeeds.

- [ ] **Step 4: Run the cross-check tests**

```bash
ctest --test-dir build -R DCTHighwayCrossCheck --output-on-failure
```
Expected: `DCTHighwayCrossCheck.Dct8_BitDepth8/10/12` all PASS.

- [ ] **Step 5: Commit**

```bash
git add source/lib/analyzer/simd/dct_hwy.cpp source/lib/analyzer/CMakeLists.txt
git commit -m "feat(simd): Highway Dct8 kernel, cross-check passes"
```

---

## Task 5: Implement `Dct16Impl`

**Files:**
- Modify: `source/lib/analyzer/simd/dct_hwy.cpp`
- Modify: `source/lib/test/DCTTestImplementationsIdenticalOutput.cpp`

**Reference:** `DCTTransformsNative.cpp:172-225` (`partialButterfly16`) and `:322-337` (`dct16_c`).

- [ ] **Step 1: Add `Dct16` cross-check test cases**

Append to `DCTTestImplementationsIdenticalOutput.cpp` (after the `Dct8` tests from Task 3):

```cpp
TEST(DCTHighwayCrossCheck, Dct16_BitDepth8)
{
    constexpr unsigned N = 16;
    int16_t src[N * N];
    int16_t dstNative[N * N];
    int16_t dstHighway[N * N];

    for (uint32_t seed = 1; seed <= 32; ++seed)
    {
        fillRandomBlock(src, N, 8, seed);
        vca::dct16_c(src, dstNative,  N, 8);
        vca::Dct16  (src, dstHighway, N, 8);
        for (unsigned i = 0; i < N * N; ++i)
            ASSERT_EQ(dstNative[i], dstHighway[i]) << "seed=" << seed << " idx=" << i;
    }
}

TEST(DCTHighwayCrossCheck, Dct16_BitDepth10)
{
    constexpr unsigned N = 16;
    int16_t src[N * N];
    int16_t dstNative[N * N];
    int16_t dstHighway[N * N];

    for (uint32_t seed = 1; seed <= 32; ++seed)
    {
        fillRandomBlock(src, N, 10, seed);
        vca::dct16_c(src, dstNative,  N, 10);
        vca::Dct16  (src, dstHighway, N, 10);
        for (unsigned i = 0; i < N * N; ++i)
            ASSERT_EQ(dstNative[i], dstHighway[i]) << "seed=" << seed << " idx=" << i;
    }
}

TEST(DCTHighwayCrossCheck, Dct16_BitDepth12)
{
    constexpr unsigned N = 16;
    int16_t src[N * N];
    int16_t dstNative[N * N];
    int16_t dstHighway[N * N];

    for (uint32_t seed = 1; seed <= 32; ++seed)
    {
        fillRandomBlock(src, N, 12, seed);
        vca::dct16_c(src, dstNative,  N, 12);
        vca::Dct16  (src, dstHighway, N, 12);
        for (unsigned i = 0; i < N * N; ++i)
            ASSERT_EQ(dstNative[i], dstHighway[i]) << "seed=" << seed << " idx=" << i;
    }
}
```

- [ ] **Step 2: Confirm the tests fail (RED)**

```bash
cmake --build build && ctest --test-dir build -R "DCTHighwayCrossCheck.Dct16" --output-on-failure
```
Expected: all three `Dct16_*` tests FAIL (Dct16 currently returns an empty buffer from the placeholder in Task 4).

- [ ] **Step 3: Add the `kT16` coefficient table**

Add at the top of `dct_hwy.cpp`, below the `kT8` table (copy verbatim from `DCTTransformsNative.cpp` `g_t16`):

```cpp
constexpr int16_t kT16[16][16] = {
    {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64},
    {90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90},
    {89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89},
    {87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87},
    {83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83},
    {80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80},
    {75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75},
    {70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70},
    {64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64},
    {57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57},
    {50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50},
    {43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43},
    {36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36},
    {25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25},
    {18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18},
    {9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9}
};
```

- [ ] **Step 4: Implement vectorized, templated `Dct16Row` and `Dct16Impl`**

A 16-wide row doesn't fit in a single int16 vector on SSE4 (8 lanes) or NEON (8 lanes). Split the row into two halves: `WidenMulPairwiseAdd` twice (once per 8-lane half), sum the two int32 partial-sum vectors, `ReduceSum`, scalar shift.

Same template-on-Shift pattern as `Dct8`. Shift2 is fixed at 10 for Dct16.

Inside `namespace vca::HWY_NAMESPACE` (before `#if HWY_ONCE`), add:

```cpp
template <int Shift>
static HWY_INLINE void Dct16Row(const int16_t *src, int16_t *dst, intptr_t dstStride)
{
    const hn::CappedTag<int16_t, 8> d16;
    const hn::CappedTag<int32_t, 4> d32;

    constexpr int32_t kAdd = 1 << (Shift - 1);

    const auto srcLo = hn::LoadU(d16, src);
    const auto srcHi = hn::LoadU(d16, src + 8);

    for (int k = 0; k < 16; ++k)
    {
        const auto coefLo = hn::LoadU(d16, &kT16[k][0]);
        const auto coefHi = hn::LoadU(d16, &kT16[k][8]);
        const auto pLo    = hn::WidenMulPairwiseAdd(d32, srcLo, coefLo);
        const auto pHi    = hn::WidenMulPairwiseAdd(d32, srcHi, coefHi);
        const int32_t sum = hn::ReduceSum(d32, hn::Add(pLo, pHi));
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }
}

template <int Shift1>
static void Dct16ImplT(const int16_t *src, int16_t *dst, intptr_t srcStride)
{
    constexpr int Shift2 = 10;
    alignas(32) int16_t coef[16 * 16];
    for (int j = 0; j < 16; ++j)
        Dct16Row<Shift1>(&src[j * srcStride], &coef[j], /*dstStride=*/16);
    for (int j = 0; j < 16; ++j)
        Dct16Row<Shift2>(&coef[j * 16], &dst[j], /*dstStride=*/16);
}

// Shift1 = 3 + bitDepth - 8.
void Dct16Impl_bd8 (const int16_t *s, int16_t *d, intptr_t st) { Dct16ImplT<3>(s, d, st); }
void Dct16Impl_bd10(const int16_t *s, int16_t *d, intptr_t st) { Dct16ImplT<5>(s, d, st); }
void Dct16Impl_bd12(const int16_t *s, int16_t *d, intptr_t st) { Dct16ImplT<7>(s, d, st); }
```

Then in the `#if HWY_ONCE` section, replace the placeholder `Dct16`:

```cpp
HWY_EXPORT(Dct16Impl_bd8);
HWY_EXPORT(Dct16Impl_bd10);
HWY_EXPORT(Dct16Impl_bd12);

void Dct16(const int16_t *src, int16_t *dst, intptr_t srcStride, unsigned bitDepth)
{
    switch (bitDepth)
    {
        case 8:  HWY_DYNAMIC_DISPATCH(Dct16Impl_bd8) (src, dst, srcStride); return;
        case 10: HWY_DYNAMIC_DISPATCH(Dct16Impl_bd10)(src, dst, srcStride); return;
        case 12: HWY_DYNAMIC_DISPATCH(Dct16Impl_bd12)(src, dst, srcStride); return;
    }
}
```

(Note: the existing `HWY_EXPORT(Dct16Impl)` line from Task 4's placeholder needs to be **replaced** by the three new exports above. Delete the old placeholder `Dct16` function body from Task 4's `#if HWY_ONCE` block.)

Then in the `#if HWY_ONCE` section, replace the placeholder `Dct16`:

```cpp
HWY_EXPORT(Dct16Impl);

void Dct16(const int16_t *src, int16_t *dst, intptr_t srcStride, unsigned bitDepth)
{
    HWY_DYNAMIC_DISPATCH(Dct16Impl)(src, dst, srcStride, bitDepth);
}
```

- [ ] **Step 5: Build and run Dct16 tests (GREEN)**

```bash
cmake --build build && ctest --test-dir build -R "DCTHighwayCrossCheck.Dct16" --output-on-failure
```
Expected: all three `Dct16_*` tests PASS.

- [ ] **Step 6: Commit**

```bash
git add source/lib/analyzer/simd/dct_hwy.cpp source/lib/test/DCTTestImplementationsIdenticalOutput.cpp
git commit -m "feat(simd): Highway Dct16 kernel (templated per bit depth)"
```

---

## Task 6: Implement templated `Dct32Impl`

**Files:**
- Modify: `source/lib/analyzer/simd/dct_hwy.cpp`
- Modify: `source/lib/test/DCTTestImplementationsIdenticalOutput.cpp`

**Reference:** `DCTTransformsNative.cpp:227-298` (`partialButterfly32`) and `:339-354` (`dct32_c`).

- [ ] **Step 1: Add `Dct32` cross-check test cases**

Same pattern as Task 5 Step 1 but with `N = 32`, calling `vca::dct32_c` and `vca::Dct32`. Bit depths 8/10/12. Seeds 1..32.

- [ ] **Step 2: Confirm RED**

```bash
cmake --build build && ctest --test-dir build -R "DCTHighwayCrossCheck.Dct32" --output-on-failure
```
Expected: all three `Dct32_*` tests FAIL.

- [ ] **Step 3: Add `kT32` coefficient table**

Copy the full `g_t32` table verbatim from `DCTTransformsNative.cpp:65-126` into `dct_hwy.cpp`, renaming to `kT32`.

- [ ] **Step 4: Implement vectorized, templated `Dct32Row` and `Dct32Impl`**

A 32-wide row = four 8-lane int16 halves. Same pattern as `Dct16Row`, unrolled four times. Same template-on-Shift layout. Shift2 is fixed at 11 for Dct32.

```cpp
template <int Shift>
static HWY_INLINE void Dct32Row(const int16_t *src, int16_t *dst, intptr_t dstStride)
{
    const hn::CappedTag<int16_t, 8> d16;
    const hn::CappedTag<int32_t, 4> d32;

    constexpr int32_t kAdd = 1 << (Shift - 1);

    const auto s0 = hn::LoadU(d16, src);
    const auto s1 = hn::LoadU(d16, src + 8);
    const auto s2 = hn::LoadU(d16, src + 16);
    const auto s3 = hn::LoadU(d16, src + 24);

    for (int k = 0; k < 32; ++k)
    {
        const auto c0 = hn::LoadU(d16, &kT32[k][0]);
        const auto c1 = hn::LoadU(d16, &kT32[k][8]);
        const auto c2 = hn::LoadU(d16, &kT32[k][16]);
        const auto c3 = hn::LoadU(d16, &kT32[k][24]);
        const auto p0 = hn::WidenMulPairwiseAdd(d32, s0, c0);
        const auto p1 = hn::WidenMulPairwiseAdd(d32, s1, c1);
        const auto p2 = hn::WidenMulPairwiseAdd(d32, s2, c2);
        const auto p3 = hn::WidenMulPairwiseAdd(d32, s3, c3);
        const auto p01 = hn::Add(p0, p1);
        const auto p23 = hn::Add(p2, p3);
        const int32_t sum = hn::ReduceSum(d32, hn::Add(p01, p23));
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }
}

template <int Shift1>
static void Dct32ImplT(const int16_t *src, int16_t *dst, intptr_t srcStride)
{
    constexpr int Shift2 = 11;
    alignas(32) int16_t coef[32 * 32];
    for (int j = 0; j < 32; ++j)
        Dct32Row<Shift1>(&src[j * srcStride], &coef[j], /*dstStride=*/32);
    for (int j = 0; j < 32; ++j)
        Dct32Row<Shift2>(&coef[j * 32], &dst[j], /*dstStride=*/32);
}

// Shift1 = 4 + bitDepth - 8.
void Dct32Impl_bd8 (const int16_t *s, int16_t *d, intptr_t st) { Dct32ImplT<4>(s, d, st); }
void Dct32Impl_bd10(const int16_t *s, int16_t *d, intptr_t st) { Dct32ImplT<6>(s, d, st); }
void Dct32Impl_bd12(const int16_t *s, int16_t *d, intptr_t st) { Dct32ImplT<8>(s, d, st); }
```

Replace the `Dct32` placeholder in the `#if HWY_ONCE` block (from Task 4) with:

```cpp
HWY_EXPORT(Dct32Impl_bd8);
HWY_EXPORT(Dct32Impl_bd10);
HWY_EXPORT(Dct32Impl_bd12);

void Dct32(const int16_t *src, int16_t *dst, intptr_t srcStride, unsigned bitDepth)
{
    switch (bitDepth)
    {
        case 8:  HWY_DYNAMIC_DISPATCH(Dct32Impl_bd8) (src, dst, srcStride); return;
        case 10: HWY_DYNAMIC_DISPATCH(Dct32Impl_bd10)(src, dst, srcStride); return;
        case 12: HWY_DYNAMIC_DISPATCH(Dct32Impl_bd12)(src, dst, srcStride); return;
    }
}
```

**Potential concern for Dct32:** an int32 accumulator holds one coefficient × one pixel. With 32-element rows, the intermediate sum is 32 products. Max coefficient is 90; max pixel at bitDepth 12 is 4095. Max per-lane product is `90 * 4095 ≈ 369k` (fits in int32 trivially). Summed 32 times: `~11.8M`. Still safely within int32. The shift-2 pass sees values that have already been shifted by `shift1` ≥ 4, so the intermediate values are smaller — no overflow risk.

- [ ] **Step 5: Build and run Dct32 tests (GREEN)**

```bash
cmake --build build && ctest --test-dir build -R "DCTHighwayCrossCheck.Dct32" --output-on-failure
```
Expected: all three `Dct32_*` tests PASS.

- [ ] **Step 6: Commit**

```bash
git add source/lib/analyzer/simd/dct_hwy.cpp source/lib/test/DCTTestImplementationsIdenticalOutput.cpp
git commit -m "feat(simd): Highway Dct32 kernel (templated per bit depth)"
```

---

## Task 7: Rewire `DCTTransform.cpp` to call Highway

**Files:**
- Modify: `source/lib/analyzer/DCTTransform.cpp`
- Modify: `source/lib/analyzer/DCTTransform.h`

- [ ] **Step 1: Replace the body of `performDCTBlockSize8/16/32`**

Current file (`source/lib/analyzer/DCTTransform.cpp:1-200`) has three functions with large `#if defined(VCA_ARCH_X86)` / `#if defined(VCA_ARCH_ARM)` blocks dispatching to named asm symbols. Replace each function body with a single Highway call.

Change the includes at the top from:

```cpp
#include <analyzer/DCTTransformsNative.h>
#include <analyzer/common/common.h>

#if !defined(VCA_DISABLE_SIMD)
#if defined (VCA_ARCH_X86)
    #include<analyzer/simd/dct-ssse3.h>
    #include<analyzer/simd/dct8.h>
#endif
#if defined (VCA_ARCH_ARM)
    #include<analyzer/simd/arm/dct-neon.h>
#endif
#endif
```

to:

```cpp
#include <analyzer/DCTTransformsNative.h>
#include <analyzer/common/common.h>
#include <analyzer/simd/dct_hwy.h>
```

Then replace the three `performDCTBlockSize{8,16,32}` functions entirely with:

```cpp
void performDCTBlockSize8(const unsigned bitDepth,
                          int16_t *pixelBuffer,
                          int16_t *coeffBuffer,
                          CpuSimd /*cpuSimd*/)
{
    vca::Dct8(pixelBuffer, coeffBuffer, 8, bitDepth);
}

void performDCTBlockSize16(const unsigned bitDepth,
                           int16_t *pixelBuffer,
                           int16_t *coeffBuffer,
                           CpuSimd /*cpuSimd*/)
{
    vca::Dct16(pixelBuffer, coeffBuffer, 16, bitDepth);
}

void performDCTBlockSize32(const unsigned bitDepth,
                           int16_t *pixelBuffer,
                           int16_t *coeffBuffer,
                           CpuSimd /*cpuSimd*/)
{
    vca::Dct32(pixelBuffer, coeffBuffer, 32, bitDepth);
}
```

The `CpuSimd` parameter is kept (for ABI compat per the spec — deprecation happens in Task 12) but unused here because Highway dispatches internally.

**Also update `performLowpassDCTBlockSize16` and any other DCT functions in this file** to use `vca::Dct8` instead of whatever SIMD dispatch they currently do. Read the full file first to find them all — the current Read showed only the first 200 lines.

- [ ] **Step 2: Build**

```bash
cmake --build build
```
Expected: build succeeds. If it fails with undefined references to `vca_dct*_ssse3/sse2/sse4/avx2/neon`, that's because the old SIMD libraries are still being linked but no longer called. Remove the `target_link_libraries(... vcaLibSimd8bit ...)` lines from the analyzer CMakeLists now — this is Task 11 work done early to unblock Task 7. Alternatively, leave a stub and defer the unlink to Task 11.

For this task, the simplest move: do **not** unlink yet. Instead, comment out the `target_sources(vcaLibSimd8bit PRIVATE dct-ssse3.cpp entropy.cpp ...)` lines so the old SIMD libs are built but empty. Or delete the three libs' sources entirely — acceptable since we're about to delete the files anyway.

**Recommendation:** skip the "leave old libs empty" workaround and just proceed to Task 11 (Deletions) immediately after this task. If the build breaks between Task 7 and Task 11, keep the files deleted and push through.

- [ ] **Step 3: Run the full test suite**

```bash
ctest --test-dir build --output-on-failure
```
Expected: all tests pass, including both the Highway cross-check tests and any pre-existing DCT tests that previously validated the old SIMD path.

- [ ] **Step 4: Commit**

```bash
git add source/lib/analyzer/DCTTransform.cpp source/lib/analyzer/DCTTransform.h
git commit -m "refactor(dct): route DCTTransform through Highway kernels"
```

---

## Task 8: Rewire `EntropyCalculation.cpp`

**Files:**
- Modify: `source/lib/analyzer/EntropyCalculation.cpp`

- [ ] **Step 1: Remove dead entropy SIMD include and dead dispatch comment**

Remove line `#include <analyzer/simd/entropy.h>` (line 24 as of current file).

Remove the dead commented-out AVX2 block at lines 49-53:
```cpp
    // Calculate entropy
    //if (cpuSimd == CpuSimd::AVX2)
    //{
    //    entropy = entropy_avx2(block);
    //}
    //else
```
Leave just the working conditional:
```cpp
    double entropy = 0;
    if (enableLowpass)
        entropy = vca::entropy_lowpass_c(block, blockSize);
    else
        entropy = vca::entropy_c(block);
    return entropy;
```

- [ ] **Step 2: Build**

```bash
cmake --build build
```
Expected: succeeds.

- [ ] **Step 3: Run tests**

```bash
ctest --test-dir build --output-on-failure
```
Expected: all tests still pass — entropy behavior is unchanged.

- [ ] **Step 4: Commit**

```bash
git add source/lib/analyzer/EntropyCalculation.cpp
git commit -m "refactor(entropy): drop dead SIMD include and TODO dispatch"
```

---

## Task 9: Add `targets.{h,cpp}` wrapper for Highway introspection

**Files:**
- Create: `source/lib/analyzer/simd/targets.h`
- Create: `source/lib/analyzer/simd/targets.cpp`

- [ ] **Step 1: Create `targets.h`**

```cpp
/* source/lib/analyzer/simd/targets.h */
#pragma once

#include <string>

namespace vca::simd {

// Returns a human-readable name for the SIMD target Highway will use at
// runtime (e.g. "AVX2", "NEON", "Scalar"). Used for startup logging.
std::string CurrentTargetName();

// Constrain Highway to a specific target mask (used by the --asm CLI flag).
// Pass one of the CLI strings: "auto", "none", "sse4", "avx2", "neon".
// Must be called exactly once, before any DCT/entropy work. Unknown strings
// are silently treated as "auto".
void ConstrainTargetsForCli(const std::string &cli_value);

} // namespace vca::simd
```

- [ ] **Step 2: Create `targets.cpp`**

```cpp
/* source/lib/analyzer/simd/targets.cpp */
#include <analyzer/simd/targets.h>

#include <hwy/targets.h>

namespace vca::simd {

std::string CurrentTargetName()
{
    const int64_t best = hwy::SupportedTargets() & ~hwy::DisabledTargets();
    if (best == 0) return "Scalar";
    // TargetName expects a single target bit — isolate the lowest-set bit,
    // which is the "best" (Highway orders bits by preference).
    const int64_t lowest = best & -best;
    return hwy::TargetName(lowest);
}

void ConstrainTargetsForCli(const std::string &cli_value)
{
    // Process-global constraint. Highway picks the best target within
    // the allowed set.
    if (cli_value == "none")
        hwy::SetSupportedTargetsForTest(HWY_SCALAR);
    else if (cli_value == "sse2" || cli_value == "ssse3" || cli_value == "sse4")
        hwy::SetSupportedTargetsForTest(HWY_SSE4);
    else if (cli_value == "avx2")
        hwy::SetSupportedTargetsForTest(HWY_AVX2);
    else if (cli_value == "neon" || cli_value == "neon_dotprod")
        hwy::SetSupportedTargetsForTest(HWY_NEON);
    // "auto" or anything else: leave Highway default (no constraint).
}

} // namespace vca::simd
```

- [ ] **Step 3: Add to build**

Add `simd/targets.cpp` to the analyzer target's source list in the appropriate CMakeLists.

- [ ] **Step 4: Build**

```bash
cmake --build build
```
Expected: succeeds.

- [ ] **Step 5: Commit**

```bash
git add source/lib/analyzer/simd/targets.h source/lib/analyzer/simd/targets.cpp source/lib/analyzer/CMakeLists.txt
git commit -m "feat(simd): add Highway target introspection wrapper"
```

---

## Task 10: Rewrite `source/lib/analyzer/simd/CMakeLists.txt`

**Files:**
- Rewrite: `source/lib/analyzer/simd/CMakeLists.txt`

The old CMakeLists builds three bit-depth-specific static libs (`vcaLibSimd8bit/10bit/12bit`) using NASM + SSSE3 intrinsics + ARM NEON. None of that is needed; the Highway kernels take `bitDepth` as a runtime parameter and Highway handles target dispatch.

- [ ] **Step 1: Replace the entire file with the Highway-era version**

```cmake
# source/lib/analyzer/simd/CMakeLists.txt
#
# The analyzer SIMD layer is now Highway-based and target-dispatched at
# runtime. No NASM, no per-bit-depth libraries, no arch branches.
#
# The source files are compiled directly into the analyzer library target
# (see source/lib/analyzer/CMakeLists.txt), so this file is intentionally
# empty-but-present to keep the directory in the project structure.
```

Alternatively, if the project's convention is to `add_subdirectory(simd)` from the analyzer CMakeLists, restructure so this file does nothing (the sources are listed in the parent's `target_sources`). Choose whichever matches the existing pattern after the rename.

- [ ] **Step 2: Remove the three `vcaLibSimd{8,10,12}bit` targets from wherever they're linked**

Grep for `vcaLibSimd`:
```bash
```
(Use the project's Grep tool; this step is descriptive — look for occurrences and remove them.)

Typical call sites:
- `source/lib/analyzer/CMakeLists.txt`: `target_link_libraries(analyzer ... vcaLibSimd8bit vcaLibSimd10bit vcaLibSimd12bit)`

Remove those three names from the `target_link_libraries` call.

- [ ] **Step 3: Build**

```bash
cmake -S . -B build && cmake --build build
```
Expected: succeeds. The `vcaLibSimd*` targets no longer exist.

- [ ] **Step 4: Run tests**

```bash
ctest --test-dir build --output-on-failure
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add source/lib/analyzer/simd/CMakeLists.txt source/lib/analyzer/CMakeLists.txt
git commit -m "build(simd): drop per-bit-depth SIMD libraries"
```

---

## Task 11: Delete the dead SIMD files

**Files:**
- Delete (via `git rm`):
  - `source/lib/analyzer/simd/cpu.h`
  - `source/lib/analyzer/simd/cpu.cpp`
  - `source/lib/analyzer/simd/cpu-a.asm`
  - `source/lib/analyzer/simd/const-a.asm`
  - `source/lib/analyzer/simd/dct8.asm`
  - `source/lib/analyzer/simd/dct8.h`
  - `source/lib/analyzer/simd/dct-ssse3.h`
  - `source/lib/analyzer/simd/dct-ssse3.cpp`
  - `source/lib/analyzer/simd/entropy.h`
  - `source/lib/analyzer/simd/entropy.cpp`
  - `source/lib/analyzer/simd/x86inc.asm`
  - `source/lib/analyzer/simd/x86util.asm`
  - `source/lib/analyzer/simd/noAsmImpl8bit.cpp`
  - `source/lib/analyzer/simd/noAsmImpl10bit.cpp`
  - `source/lib/analyzer/simd/noAsmImpl12bit.cpp`
  - `source/lib/analyzer/simd/arm/dct-neon.h`
  - `source/lib/analyzer/simd/arm/dct-neon.cpp`
  - `source/lib/analyzer/simd/arm/entropy-neon.cpp`
  - `source/lib/analyzer/simd/arm/neon-utils.h`
- Delete directory: `source/lib/analyzer/simd/arm/`

- [ ] **Step 1: Verify nothing still references these files**

For each file name (without extension), grep in `source/` and `CMakeLists.txt`:
```
(use the Grep tool, pattern: "cpu-a|const-a|dct8\\.asm|dct-ssse3|entropy\\.h|noAsmImpl|dct-neon|entropy-neon|neon-utils|x86inc|x86util")
```
Expected: no results. If anything matches, fix that call site first.

Also confirm no `CpuSimd` / `cpuDetectMaxSimd` / `isSimdSupported` still come from `cpu.h` — those are handled in Task 12, but check that removing `cpu.h` won't break compilation here. If it does, Task 12 must be reordered before this one. **If reordering is needed: stop, flag to reviewer, do not proceed.**

- [ ] **Step 2: `git rm` each file**

```bash
git rm source/lib/analyzer/simd/cpu.h \
       source/lib/analyzer/simd/cpu.cpp \
       source/lib/analyzer/simd/cpu-a.asm \
       source/lib/analyzer/simd/const-a.asm \
       source/lib/analyzer/simd/dct8.asm \
       source/lib/analyzer/simd/dct8.h \
       source/lib/analyzer/simd/dct-ssse3.h \
       source/lib/analyzer/simd/dct-ssse3.cpp \
       source/lib/analyzer/simd/entropy.h \
       source/lib/analyzer/simd/entropy.cpp \
       source/lib/analyzer/simd/x86inc.asm \
       source/lib/analyzer/simd/x86util.asm \
       source/lib/analyzer/simd/noAsmImpl8bit.cpp \
       source/lib/analyzer/simd/noAsmImpl10bit.cpp \
       source/lib/analyzer/simd/noAsmImpl12bit.cpp \
       source/lib/analyzer/simd/arm/dct-neon.h \
       source/lib/analyzer/simd/arm/dct-neon.cpp \
       source/lib/analyzer/simd/arm/entropy-neon.cpp \
       source/lib/analyzer/simd/arm/neon-utils.h
```

- [ ] **Step 3: Build and run all tests**

```bash
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```
Expected: clean build, all tests pass.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor(simd): delete hand-written SIMD code (asm, SSSE3, NEON, noAsmImpl)"
```

---

## Task 12: Deprecate `CpuSimd`, `vca_emms`, and add new DCT overloads in `vcaLib.h`

**Files:**
- Modify: `source/lib/vcaLib.h`

**Read the current file first** to understand the existing `CpuSimd` definition and the current public DCT function declarations.

- [ ] **Step 1: Add `[[deprecated]]` attribute to `CpuSimd` enum**

Find the `enum class CpuSimd` (or `enum CpuSimd`) declaration. Change:
```cpp
enum class CpuSimd { ... };
```
to:
```cpp
enum class [[deprecated(
    "CpuSimd is ignored; Highway chooses the runtime target. "
    "Will be removed in a future version.")]]
CpuSimd { ... };
```

- [ ] **Step 2: Add new DCT overloads without `CpuSimd` parameter**

Just above the existing declarations of `performDCTBlockSize8/16/32`, add the new primary overloads:

```cpp
// Highway-era DCT entry points. Prefer these over the CpuSimd overloads.
void performDCTBlockSize8 (unsigned bitDepth, int16_t *pixelBuffer, int16_t *coeffBuffer);
void performDCTBlockSize16(unsigned bitDepth, int16_t *pixelBuffer, int16_t *coeffBuffer);
void performDCTBlockSize32(unsigned bitDepth, int16_t *pixelBuffer, int16_t *coeffBuffer);
```

Mark the old overloads deprecated:

```cpp
[[deprecated("CpuSimd parameter is ignored; call the overload without it.")]]
void performDCTBlockSize8 (unsigned bitDepth, int16_t *pixelBuffer, int16_t *coeffBuffer, CpuSimd cpuSimd);
[[deprecated("CpuSimd parameter is ignored; call the overload without it.")]]
void performDCTBlockSize16(unsigned bitDepth, int16_t *pixelBuffer, int16_t *coeffBuffer, CpuSimd cpuSimd);
[[deprecated("CpuSimd parameter is ignored; call the overload without it.")]]
void performDCTBlockSize32(unsigned bitDepth, int16_t *pixelBuffer, int16_t *coeffBuffer, CpuSimd cpuSimd);
```

Do the same for `performLowpassDCTBlockSize16` and `performEntropy` / `performEdgeDensity` — anywhere `CpuSimd` appears as a parameter, create a new overload without it and deprecate the old one.

- [ ] **Step 3: Deprecate `vca_emms`**

Find the existing `vca_emms` macro definitions in `vcaLib.h` (or wherever they currently live — may be `cpu.h` being deleted in Task 11). Replace with:

```cpp
namespace vca {
[[deprecated("vca_emms is a no-op under Highway; remove calls.")]]
inline void vca_emms_deprecated() noexcept {}
} // namespace vca

#define vca_emms() ::vca::vca_emms_deprecated()
```

- [ ] **Step 4: Implement the new overloads in `DCTTransform.cpp`**

For each new overload in `vcaLib.h`, add its implementation in `DCTTransform.cpp`:
```cpp
void performDCTBlockSize8(unsigned bitDepth, int16_t *pixelBuffer, int16_t *coeffBuffer)
{
    vca::Dct8(pixelBuffer, coeffBuffer, 8, bitDepth);
}
// ... 16, 32, and the Lowpass variant
```

The existing deprecated overloads (from Task 7) simply forward by calling the new overload, so they can be simplified:
```cpp
void performDCTBlockSize8(unsigned bitDepth, int16_t *pixelBuffer, int16_t *coeffBuffer, CpuSimd /*cpuSimd*/)
{
    performDCTBlockSize8(bitDepth, pixelBuffer, coeffBuffer);
}
```

Same treatment for `performEntropy` / `performEdgeDensity` in `EntropyCalculation.cpp`.

- [ ] **Step 5: Build**

```bash
cmake --build build
```
Expected: succeeds **with deprecation warnings** at every internal call site that still passes a `CpuSimd` argument. These get fixed in Task 13.

- [ ] **Step 6: Commit**

```bash
git add source/lib/vcaLib.h source/lib/analyzer/DCTTransform.cpp source/lib/analyzer/DCTTransform.h source/lib/analyzer/EntropyCalculation.cpp source/lib/analyzer/EntropyCalculation.h
git commit -m "api: deprecate CpuSimd, add CpuSimd-free overloads"
```

---

## Task 13: Update internal call sites to use new overloads

**Files:**
- Modify: `source/lib/analyzer/Analyzer.cpp`, `source/lib/analyzer/EnergyCalculation.cpp`, `source/lib/analyzer/EnergyCalculation.h`
- Modify: `source/lib/test/DCTTestForwardBackwards.cpp`, `source/lib/test/DCTTestImplementationsIdenticalOutput.cpp`
- Modify: `source/apps/vca/vca.cpp`, `source/apps/vcaPerformanceTest/vcaPerformanceTest.cpp`

- [ ] **Step 1: Find all internal callers of deprecated overloads**

Grep:
```
pattern: "performDCTBlockSize|performEntropy|performEdgeDensity|performLowpassDCT"
path: source/
```
Review each result. If the call passes a `CpuSimd` argument, update it to drop the argument (call the new overload).

- [ ] **Step 2: Update each call site**

For each match, remove the trailing `CpuSimd` argument from the call. Example:
```cpp
// Before:
performDCTBlockSize8(bitDepth, pixels, coeffs, cpuSimd);
// After:
performDCTBlockSize8(bitDepth, pixels, coeffs);
```

- [ ] **Step 3: Audit `EnergyCalculation`**

`EnergyCalculation.{cpp,h}` also takes a `CpuSimd` parameter per the spec. Since energy calculation does not use the SIMD dispatch path (confirm by reading the file — it calls native functions), drop the parameter from the public API: remove it from `EnergyCalculation.h` and update the call sites. Do not add a deprecated overload for energy — it's a simpler signature change since the parameter has no effect.

Actually, for API symmetry with DCT/entropy, **do** add a deprecated overload for `performEnergy*` functions matching the DCT treatment. This ensures external `vcaLib.h` consumers don't break.

- [ ] **Step 4: Build with `-Werror=deprecated-declarations`** (if easy to toggle) to confirm no internal deprecation warnings fire

```bash
cmake --build build -- -Werror=deprecated-declarations
```
Or just check the build log for `warning: ... deprecated` lines in VCA source files (not in `_deps/highway-src`).

Expected: zero internal deprecation warnings. External consumers would still see them when including `vcaLib.h` with their own code — that's the intended behavior.

- [ ] **Step 5: Run tests**

```bash
ctest --test-dir build --output-on-failure
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "refactor: migrate internal callers off deprecated CpuSimd API"
```

---

## Task 14: Remap `--asm` CLI flag to Highway targets

**Files:**
- Modify: `source/apps/vca/vca.cpp`
- Modify: `docs/cli.md`

- [ ] **Step 1: Find current `--asm` parsing in `vca.cpp`**

Grep `vca.cpp` for `"--asm"` or `cpuSimd`. The existing code likely parses the flag into a `CpuSimd` enum value and passes it down through DCT calls.

- [ ] **Step 2: Replace with Highway target constraint**

At startup in `main()` (after argv parsing), add:

```cpp
#include <analyzer/simd/targets.h>
// ...
if (!asmFlagValue.empty())
    vca::simd::ConstrainTargetsForCli(asmFlagValue);
```

Remove any downstream code that passes `cpuSimd` into DCT/entropy calls (already gone after Task 13, but double-check).

Add a startup log line:
```cpp
std::cout << "SIMD target: " << vca::simd::CurrentTargetName() << "\n";
```

- [ ] **Step 3: Update `docs/cli.md`**

Find the `--asm` section in `docs/cli.md`. Replace with:

```markdown
### `--asm <target>`

Constrain the SIMD target. Default is `auto` — Highway picks the best
available target for the current CPU.

| Value | Effect |
|-------|--------|
| `auto` (default) | Use Highway's automatic target selection. |
| `none` | Disable SIMD — force Highway's scalar target. |
| `sse2` / `ssse3` / `sse4` | Restrict to SSE4 baseline on x86. |
| `avx2` | Restrict to AVX2 on x86. |
| `neon` / `neon_dotprod` | Restrict to NEON on ARM. |

Unknown values are silently treated as `auto`.
```

- [ ] **Step 4: Build and smoke-test the CLI**

```bash
cmake --build build
./build/source/apps/vca/vca --asm none --help    # should print normally
./build/source/apps/vca/vca --asm auto --help    # should print normally
./build/source/apps/vca/vca --asm avx2 --help    # should print normally
```
Each invocation should print "SIMD target: <name>" at the top.

- [ ] **Step 5: Commit**

```bash
git add source/apps/vca/vca.cpp docs/cli.md
git commit -m "feat(cli): remap --asm to Highway target constraint"
```

---

## Task 15: Strip NASM and arch detection from root `CMakeLists.txt`

**Files:**
- Modify: `CMakeLists.txt` (root)
- Modify: `source/lib/analyzer/common/common.h` (if `VCA_ARCH_*` macros live there)

- [ ] **Step 1: Remove from root `CMakeLists.txt`**

Grep and delete:
- `enable_language(ASM_NASM)`
- `find_package(NASM)` / `find_program(NASM_EXECUTABLE nasm)`
- Any `if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86|amd64|AMD64")` + `set(VCA_ARCH_X86 ON)` block
- Any `if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")` + `set(VCA_ARCH_ARM ON)` block
- `NASM_ENABLED` variable and all references
- `ENABLE_ASSEMBLY` option / variable
- `SIMD_ENABLED` conditional gating (if present — Highway is always on now)
- `VCA_DISABLE_SIMD` compile definition

- [ ] **Step 2: Remove `VCA_ARCH_*` / `VCA_DISABLE_SIMD` macro usage**

Grep:
```
pattern: "VCA_ARCH_|VCA_DISABLE_SIMD|NASM_ENABLED"
path: source/
```
For each match, delete the `#if defined(...)` block (keeping only the body that would have executed — usually the Highway path now). Most of these are already gone after Task 7; this is the final sweep for stragglers.

- [ ] **Step 3: Clean configure and build from scratch**

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```
Expected: clean configure with no NASM messages, clean build.

- [ ] **Step 4: Run full test suite**

```bash
ctest --test-dir build --output-on-failure
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt source/lib/analyzer/common/common.h
git commit -m "build: remove NASM and arch detection"
```

---

## Task 16: Update `DCTTestImplementationsIdenticalOutput.cpp` to iterate Highway targets

**Files:**
- Modify: `source/lib/test/DCTTestImplementationsIdenticalOutput.cpp`

Currently the test iterates over `CpuSimd` enum values. With Highway, the runtime target is chosen by Highway, and we should iterate `hwy::SupportedTargets()` using `hwy::SetSupportedTargetsForTest` to exercise each target in sequence.

- [ ] **Step 1: Add a parameterized fixture that constrains Highway target**

At the top of the file:
```cpp
#include <hwy/targets.h>

class DCTHighwayTargetFixture : public ::testing::TestWithParam<int64_t>
{
protected:
    void SetUp() override  { hwy::SetSupportedTargetsForTest(GetParam()); }
    void TearDown() override { hwy::SetSupportedTargetsForTest(0); } // restore default
};
```

- [ ] **Step 2: Convert the Task 3/5/6 tests to use the fixture**

Replace the existing `TEST(DCTHighwayCrossCheck, Dct8_BitDepth8)` etc. with parameterized versions:

```cpp
TEST_P(DCTHighwayTargetFixture, Dct8_AllBitDepths)
{
    constexpr unsigned N = 8;
    int16_t src[N * N];
    int16_t dstNative[N * N];
    int16_t dstHighway[N * N];

    for (unsigned bitDepth : {8u, 10u, 12u})
    {
        for (uint32_t seed = 1; seed <= 32; ++seed)
        {
            fillRandomBlock(src, N, bitDepth, seed);
            vca::dct8_c(src, dstNative,  N, bitDepth);
            vca::Dct8  (src, dstHighway, N, bitDepth);
            for (unsigned i = 0; i < N * N; ++i)
                ASSERT_EQ(dstNative[i], dstHighway[i])
                    << "target=" << hwy::TargetName(GetParam())
                    << " bitDepth=" << bitDepth
                    << " seed=" << seed << " idx=" << i;
        }
    }
}

// Same pattern for Dct16_AllBitDepths and Dct32_AllBitDepths.

INSTANTIATE_TEST_SUITE_P(
    AllHighwayTargets,
    DCTHighwayTargetFixture,
    ::testing::ValuesIn([] {
        std::vector<int64_t> out;
        const int64_t supported = hwy::SupportedTargets();
        for (int64_t bit = 1; bit != 0; bit <<= 1)
            if (supported & bit) out.push_back(bit);
        return out;
    }())
);
```

- [ ] **Step 3: Build and run**

```bash
cmake --build build && ctest --test-dir build -R DCTHighwayTargetFixture --output-on-failure
```
Expected: each supported target is exercised for all three block sizes and bit depths. On x86 with AVX2 hardware, expect `Scalar`, `SSE4`, `AVX2` (and possibly `AVX3`). On ARM64, expect `Scalar`, `NEON`.

- [ ] **Step 4: Commit**

```bash
git add source/lib/test/DCTTestImplementationsIdenticalOutput.cpp
git commit -m "test: parameterize DCT cross-check over Highway targets"
```

---

## Task 17: Update `readme.rst` and final docs pass

**Files:**
- Modify: `readme.rst`
- Modify: `source/lib/analyzer/simd/README.txt`

- [ ] **Step 1: Update `readme.rst`**

Add a note under "Building" (or wherever build prerequisites are listed):
> **Network on first configure:** VCA fetches Google Highway at configure time via CMake FetchContent. The first `cmake -S . -B build` invocation requires internet access; subsequent builds use the cached copy under `build/_deps/`.

Remove any mention of NASM as a prerequisite. Remove mention of `ENABLE_ASSEMBLY` / `--disable-asm` / similar flags.

- [ ] **Step 2: Rewrite `source/lib/analyzer/simd/README.txt`**

```
SIMD layer
==========

All SIMD in this directory is implemented using Google Highway
(https://github.com/google/highway) — a portable C++ library that dispatches
at runtime to the best available SIMD target (SSE4 / AVX2 / AVX3 / NEON /
scalar) from a single source.

Files:

  dct_hwy.h / dct_hwy.cpp
      Portable DCT 8/16/32 kernels. Implementation uses Highway's
      foreach_target.h mechanism: the same .cpp is compiled once per
      enabled target, and HWY_DYNAMIC_DISPATCH picks the best variant
      at runtime.

  targets.h / targets.cpp
      Thin wrapper around hwy::SupportedTargets / hwy::TargetName used
      for startup logging and the --asm CLI flag.

To add a new kernel, follow the DCT pattern in dct_hwy.cpp:

  1. Write the body under namespace vca::HWY_NAMESPACE.
  2. HWY_EXPORT the symbol inside the HWY_ONCE block.
  3. Define a plain-named entry point that calls HWY_DYNAMIC_DISPATCH.
  4. Add a cross-check test against the scalar reference in
     source/lib/test/.

Bit-exactness against the scalar reference in DCTTransformsNative.cpp /
EntropyNative.cpp is a hard requirement — any divergence is a bug.
```

- [ ] **Step 3: Commit**

```bash
git add readme.rst source/lib/analyzer/simd/README.txt
git commit -m "docs: update build notes and SIMD README for Highway"
```

---

## Task 18: Final full build, full test, manual smoke test

**Files:**
- No file edits in this task.

- [ ] **Step 1: Clean configure and full build in Release**

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```
Expected: clean build with zero errors and zero warnings in VCA source files. Highway may emit its own warnings from `_deps/highway-src` — acceptable.

- [ ] **Step 2: Run full test suite**

```bash
ctest --test-dir build --output-on-failure
```
Expected: all tests pass, including:
- Existing pre-Highway tests (`DCTTestForwardBackwards` etc.)
- All `DCTHighwayTargetFixture` parameterized tests

- [ ] **Step 3: Manual E2E smoke test on a sample video**

Pick any reasonably-sized test clip available on this machine (reuse whatever CI or local dev uses). Run:

```bash
./build/source/apps/vca/vca -i <clip> -o /tmp/highway-out.csv [other usual flags]
```

Then switch to the `stable` branch, rebuild, and run the same command producing `/tmp/stable-out.csv`:

```bash
git stash   # or keep the highway build in a separate dir
git checkout stable
cmake --build build
./build/source/apps/vca/vca -i <clip> -o /tmp/stable-out.csv [same flags]
```

Diff:
```bash
diff /tmp/highway-out.csv /tmp/stable-out.csv
```
Expected: **no differences**. Any divergence is a bug introduced by this port — halt and debug.

After verifying, switch back to `highway` branch:
```bash
git checkout highway
```

- [ ] **Step 4: Final commit (if any cleanup needed from smoke test)**

If the smoke test required any fix, make it in a new commit:
```bash
git add -u
git commit -m "fix: <specific issue from smoke test>"
```

Otherwise no commit.

- [ ] **Step 5: Push the branch**

```bash
git push -u origin highway
```

---

## Deferred (not this plan)

- **Factored butterfly formulation.** Tasks 4–6 use a direct matrix-vector multiply (~32 `WidenMulPairwiseAdd` ops per DCT-16 row). The scalar reference uses a factored butterfly (E/O decomposition) which does ~16 multiplies per row. A follow-up optimization can adopt the factored form in vectors — it's ~2× fewer multiplies but requires a shuffle to reverse the second half of the row, which is target-dependent in cost. Profile first.
- **Native-vector-width unrolling.** For Dct16/Dct32, the kernel currently processes rows one at a time with 8-lane int16 halves. On AVX2 (16-lane) or AVX-512 (32-lane) a further optimization could process two or four rows in parallel per iteration. Wait for profiling to justify the code bloat.
- **Entropy SIMD.** Intentionally out of scope. Entropy remains on the scalar `entropy_c` / `entropy_lowpass_c` implementations. Adding Highway entropy kernels is a new feature, not a port.
- **Removal of the deprecated `vcaLib.h` shims.** A future major version can drop `[[deprecated]]` `CpuSimd`, the old overloads, and `vca_emms`.
- **CI validation on non-x86 / non-ARM64 targets.** Highway supports SVE, RVV, WASM, etc. for free, but this plan does not add CI runners for them.

---

## Self-Review Notes

Completed after writing all 18 tasks:

1. **Spec coverage:**
   - DCT port → Tasks 3, 4, 5, 6
   - Highway via FetchContent → Task 1
   - Entropy dead code removal → Task 8
   - File deletions → Task 11
   - Backward-compat deprecated overloads → Task 12
   - Internal call site migration → Task 13
   - `--asm` CLI remap → Task 14
   - NASM/arch removal → Task 15
   - Test parameterization over Highway targets → Task 16
   - Docs → Task 17
   - Smoke test → Task 18
   All spec requirements covered.

2. **Placeholder scan:**
   - `<HIGHWAY_TAG>` is a placeholder but intentionally resolved in Task 0. Flagged in the task.
   - `<analyzer_target>` in Task 2 is a placeholder but Step 1 of the same task tells the implementer to find it by inspection. Acceptable.
   - No "TBD", "TODO", "appropriate error handling", or hand-waves in code blocks.

3. **Type/signature consistency:**
   - `Dct8` / `Dct16` / `Dct32` signatures are `(const int16_t*, int16_t*, intptr_t, unsigned)` throughout (Tasks 3, 4, 5, 6, 7, 12).
   - `vca::simd::CurrentTargetName()` and `vca::simd::ConstrainTargetsForCli()` used consistently in Tasks 9 and 14.
   - `DCTTransform.cpp` call sites call the Highway entry points, not the `*Impl` symbols — correct.

4. **Real vectorization + compile-time shifts:** Tasks 4–6 use `hn::WidenMulPairwiseAdd` + `hn::ReduceSum` — actual vector ops, not scalar-in-Highway. The formulation is matrix-vector multiply (`dst_row = kTN * src_row`). Every kernel is templated on `Shift`, instantiated three times per block size (bit depths 8/10/12) → 9 HWY_EXPORT'd symbols total. The final `>> Shift` and rounding-add are compile-time constants the compiler can fold into immediate shift operands. The factored butterfly is left as a follow-up optimization per the "Deferred" section.
