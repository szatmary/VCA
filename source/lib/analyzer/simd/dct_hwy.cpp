/* source/lib/analyzer/simd/dct_hwy.cpp
 *
 * Portable Highway implementation of HEVC DCT 8/16/32 transforms.
 * Bit-identical to vca::dct{8,16,32}_c in DCTTransformsNative.cpp.
 *
 * Strategy: factored butterfly (E/O decomposition) matching
 * partialButterfly{8,16,32} in DCTTransformsNative.cpp. The factored
 * form reduces per-row multiply count by ~3-4x versus the naive
 * matrix-vector formulation.
 *
 * Butterfly arithmetic runs on int32-promoted lanes. An int16-lane
 * butterfly would overflow on pass 2 of the DCT, because pass-1
 * coefficients exceed int16 range when E[k] = src[k] + src[N-1-k] is
 * formed. Loading int16 source, promoting to int32 via hn::PromoteTo,
 * and running every butterfly/dot-product stage in int32 matches the
 * scalar reference bit-for-bit while keeping full SIMD throughput.
 * Short dot products (length-2 on EEE/EEEE, length-4 on EO/EEO) stay
 * scalar — the vector setup cost exceeds the gain at that size.
 *
 * HWY_SCALAR has HWY_LANES == 1 (no multi-lane vectors, no UpperHalf),
 * so the vector forms collapse to per-element work. The scalar target
 * keeps the plain scalar butterfly for that reason. Every real SIMD
 * target (NEON, NEON_BF16, EMU128, SSE4, AVX2, ...) runs the
 * vectorized path.
 */

#include <cstdint>
#include <cstring>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "analyzer/simd/dct_hwy.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace vca::HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Local copies of the HEVC DCT coefficient tables. Values must match
// DCTTransformsNative.cpp g_t8 / g_t16 / g_t32 exactly.
// Defined inside the per-target namespace so each Highway target gets its
// own copy when foreach_target.h re-includes this file.
alignas(16) static constexpr int16_t kT8[8][8] = {
    {64, 64, 64, 64, 64, 64, 64, 64},
    {89, 75, 50, 18, -18, -50, -75, -89},
    {83, 36, -36, -83, -83, -36, 36, 83},
    {75, -18, -89, -50, 50, 89, 18, -75},
    {64, -64, -64, 64, 64, -64, -64, 64},
    {50, -89, 18, 75, -75, -18, 89, -50},
    {36, -83, 83, -36, -36, 83, -83, 36},
    {18, -50, 75, -89, 89, -75, 50, -18}
};

alignas(16) static constexpr int16_t kT16[16][16] = {
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

alignas(16) static constexpr int16_t kT32[32][32] = {
    {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
     64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64},
    {90, 90,  88,  85,  82,  78,  73,  67,  61,  54,  46,  38,  31,  22,  13,  4,
     -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90},
    {90,  87,  80,  70,  57,  43,  25,  9,  -9, -25, -43, -57, -70, -80, -87, -90,
     -90, -87, -80, -70, -57, -43, -25, -9, 9,  25,  43,  57,  70,  80,  87,  90},
    {90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13,
     13, 38, 61, 78, 88, 90, 85,  73,  54,  31,  4,   -22, -46, -67, -82, -90},
    {89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89,
     89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89},
    {88,  67,  31,  -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85,  61,  22,
     -22, -61, -85, -90, -73, -38, 4,   46,  78,  90, 82, 54, 13, -31, -67, -88},
    {87,  57,  9,  -43, -80, -90, -70, -25, 25,  70,  90,  80,  43,  -9, -57, -87,
     -87, -57, -9, 43,  80,  90,  70,  25,  -25, -70, -90, -80, -43, 9,  57,  87},
    {85, 46, -13, -67, -90, -73, -22, 38,  82,  88, 54, -4, -61, -90, -78, -31,
     31, 78, 90,  61,  4,   -54, -88, -82, -38, 22, 73, 90, 67,  13,  -46, -85},
    {83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83,
     83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83},
    {82,  22,  -54, -90, -61, 13, 78, 85,  31,  -46, -90, -67, 4,  73, 88,  38,
     -38, -88, -73, -4,  67,  90, 46, -31, -85, -78, -13, 61,  90, 54, -22, -82},
    {80,  9,  -70, -87, -25, 57,  90,  43,  -43, -90, -57, 25,  87,  70,  -9, -80,
     -80, -9, 70,  87,  25,  -57, -90, -43, 43,  90,  57,  -25, -87, -70, 9,  80},
    {78, -4, -82, -73, 13,  85,  67, -22, -88, -61, 31,  90,  54, -38, -90, -46,
     46, 90, 38,  -54, -90, -31, 61, 88,  22,  -67, -85, -13, 73, 82,  4,   -78},
    {75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75,
     75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75},
    {73,  -31, -90, -22, 78, 67,  -38, -90, -13, 82, 61,  -46, -88, -4, 85, 54,
     -54, -85, 4,   88,  46, -61, -82, 13,  90,  38, -67, -78, 22,  90, 31, -73},
    {70,  -43, -87, 9,  90,  25,  -80, -57, 57,  80,  -25, -90, -9, 87,  43,  -70,
     -70, 43,  87,  -9, -90, -25, 80,  57,  -57, -80, 25,  90,  9,  -87, -43, 70},
    {67, -54, -78, 38,  85, -22, -90, 4,   90, 13, -88, -31, 82,  46, -73, -61,
     61, 73,  -46, -82, 31, 88,  -13, -90, -4, 90, 22,  -85, -38, 78, 54,  -67},
    {64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64,
     64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64},
    {61,  -73, -46, 82, 31,  -88, -13, 90, -4,  -90, 22, 85,  -38, -78, 54, 67,
     -67, -54, 78,  38, -85, -22, 90,  4,  -90, 13,  88, -31, -82, 46,  73, -61},
    {57,  -80, -25, 90,  -9, -87, 43,  70,  -70, -43, 87,  9,  -90, 25,  80,  -57,
     -57, 80,  25,  -90, 9,  87,  -43, -70, 70,  43,  -87, -9, 90,  -25, -80, 57},
    {54, -85, -4,  88, -46, -61, 82,  13, -90, 38,  67, -78, -22, 90, -31, -73,
     73, 31,  -90, 22, 78,  -67, -38, 90, -13, -82, 61, 46,  -88, 4,  85,  -54},
    {50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50,
     50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50},
    {46,  -90, 38, 54,  -90, 31, 61,  -88, 22, 67,  -85, 13, 73,  -82, 4,  78,
     -78, -4,  82, -73, -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46},
    {43,  -90, 57,  25,  -87, 70,  9,  -80, 80,  -9, -70, 87,  -25, -57, 90,  -43,
     -43, 90,  -57, -25, 87,  -70, -9, 80,  -80, 9,  70,  -87, 25,  57,  -90, 43},
    {38, -88, 73,  -4, -67, 90,  -46, -31, 85, -78, 13,  61, -90, 54,  22, -82,
     82, -22, -54, 90, -61, -13, 78,  -85, 31, 46,  -90, 67, 4,   -73, 88, -38},
    {36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36,
     36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36},
    {31,  -78, 90, -61, 4,  54,  -88, 82, -38, -22, 73,  -90, 67, -13, -46, 85,
     -85, 46,  13, -67, 90, -73, 22,  38, -82, 88,  -54, -4,  61, -90, 78,  -31},
    {25,  -70, 90,  -80, 43,  9,  -57, 87,  -87, 57,  -9, -43, 80,  -90, 70,  -25,
     -25, 70,  -90, 80,  -43, -9, 57,  -87, 87,  -57, 9,  43,  -80, 90,  -70, 25},
    {22, -61, 85, -90, 73,  -38, -4,  46, -78, 90, -82, 54,  -13, -31, 67, -88,
     88, -67, 31, 13,  -54, 82,  -90, 78, -46, 4,  38,  -73, 90,  -85, 61, -22},
    {18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18,
     18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18},
    {13,  -38, 61,  -78, 88,  -90, 85, -73, 54, -31, 4,  22,  -46, 67,  -82, 90,
     -90, 82,  -67, 46,  -22, -4,  31, -54, 73, -85, 90, -88, 78,  -61, 38,  -13},
    {9,  -25, 43,  -57, 70,  -80, 87,  -90, 90,  -87, 80,  -70, 57,  -43, 25,  -9,
     -9, 25,  -43, 57,  -70, 80,  -87, 90,  -90, 87,  -80, 70,  -57, 43,  -25, 9},
    {4,  -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90,
     90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4}
};

// One row of an 8x8 DCT: writes 8 output coefficients, each into dst[k*dstStride]
// (so both the row-major and column-major passes share this helper).
//
// Factored butterfly form mirroring partialButterfly8 in
// DCTTransformsNative.cpp. Butterfly runs on 4-lane int32 vectors; the
// length-4 dot products on O stay scalar because the vector-setup cost
// exceeds the gain at that size.
template <int Shift>
static HWY_INLINE void Dct8Row(const int16_t *src, int16_t *dst, intptr_t dstStride)
{
    constexpr int32_t kAdd = 1 << (Shift - 1);

#if HWY_TARGET != HWY_SCALAR
    const hn::CappedTag<int32_t, 4> d32;
    const hn::Rebind<int16_t, decltype(d32)> d16_half;  // 4-lane int16

    // Load src[0..3] and src[4..7] as 4-lane int16, promote to int32.
    const auto lo32 = hn::PromoteTo(d32, hn::LoadU(d16_half, src));
    const auto hi32 = hn::PromoteTo(d32, hn::LoadU(d16_half, src + 4));
    const auto hiRev32 = hn::Reverse(d32, hi32);  // [s7, s6, s5, s4]

    // E[0..3] = src[0..3] + src[7..4],  O[0..3] = src[0..3] - src[7..4]
    const auto Ev = hn::Add(lo32, hiRev32);
    const auto Ov = hn::Sub(lo32, hiRev32);

    alignas(16) int32_t E[4];
    alignas(16) int32_t O[4];
    hn::StoreU(Ev, d32, E);
    hn::StoreU(Ov, d32, O);
#else
    int32_t E[4];
    int32_t O[4];
    for (int k = 0; k < 4; ++k)
    {
        E[k] = src[k] + src[7 - k];
        O[k] = src[k] - src[7 - k];
    }
#endif

    // EE[0..1] and EO[0..1] — scalar, only 2 elements each.
    const int32_t EE0 = E[0] + E[3];
    const int32_t EE1 = E[1] + E[2];
    const int32_t EO0 = E[0] - E[3];
    const int32_t EO1 = E[1] - E[2];

    // Even-indexed outputs: length-2 dot products on EE / EO.
    dst[0 * dstStride] = static_cast<int16_t>(
        (kT8[0][0] * EE0 + kT8[0][1] * EE1 + kAdd) >> Shift);
    dst[4 * dstStride] = static_cast<int16_t>(
        (kT8[4][0] * EE0 + kT8[4][1] * EE1 + kAdd) >> Shift);
    dst[2 * dstStride] = static_cast<int16_t>(
        (kT8[2][0] * EO0 + kT8[2][1] * EO1 + kAdd) >> Shift);
    dst[6 * dstStride] = static_cast<int16_t>(
        (kT8[6][0] * EO0 + kT8[6][1] * EO1 + kAdd) >> Shift);

    // Odd-indexed outputs: length-4 dot products on O. Scalar is fine
    // here — 4 outputs × 4 terms is too small to benefit from SIMD.
    dst[1 * dstStride] = static_cast<int16_t>(
        (kT8[1][0] * O[0] + kT8[1][1] * O[1] + kT8[1][2] * O[2] + kT8[1][3] * O[3] + kAdd) >> Shift);
    dst[3 * dstStride] = static_cast<int16_t>(
        (kT8[3][0] * O[0] + kT8[3][1] * O[1] + kT8[3][2] * O[2] + kT8[3][3] * O[3] + kAdd) >> Shift);
    dst[5 * dstStride] = static_cast<int16_t>(
        (kT8[5][0] * O[0] + kT8[5][1] * O[1] + kT8[5][2] * O[2] + kT8[5][3] * O[3] + kAdd) >> Shift);
    dst[7 * dstStride] = static_cast<int16_t>(
        (kT8[7][0] * O[0] + kT8[7][1] * O[1] + kT8[7][2] * O[2] + kT8[7][3] * O[3] + kAdd) >> Shift);
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

// Non-template thunks - HWY_EXPORT cannot take template arguments directly.
// One per supported bit depth; Shift1 = 2 + bitDepth - 8.
void Dct8Impl_bd8 (const int16_t *s, int16_t *d, intptr_t st) { Dct8ImplT<2>(s, d, st); }
void Dct8Impl_bd10(const int16_t *s, int16_t *d, intptr_t st) { Dct8ImplT<4>(s, d, st); }
void Dct8Impl_bd12(const int16_t *s, int16_t *d, intptr_t st) { Dct8ImplT<6>(s, d, st); }

// One row of a 16x16 DCT. Factored butterfly mirroring partialButterfly16.
// Butterfly runs on 4-lane int32 vectors. For pass 1 (Shift <= 8) the
// length-8 dot products on O use an int16 fast path via
// hn::WidenMulPairwiseAdd — this maps to smull/smlal on NEON, roughly
// double the throughput of the int32 Mul path. Pass 2 (Shift >= 9) keeps
// the int32 form because pass-1 coefficients exceed int16 range. Length-4
// dot products on EO stay scalar (too small to win on SIMD).
template <int Shift>
static HWY_INLINE void Dct16Row(const int16_t *src, int16_t *dst, intptr_t dstStride)
{
    constexpr int32_t kAdd = 1 << (Shift - 1);

#if HWY_TARGET != HWY_SCALAR
    const hn::CappedTag<int32_t, 4> d32;
    const hn::Rebind<int16_t, decltype(d32)> d16_half;  // 4-lane int16

    // Load src[0..3], src[4..7], src[8..11], src[12..15] and promote to int32.
    const auto s0 = hn::PromoteTo(d32, hn::LoadU(d16_half, src));       // src[0..3]
    const auto s1 = hn::PromoteTo(d32, hn::LoadU(d16_half, src + 4));   // src[4..7]
    const auto s2 = hn::PromoteTo(d32, hn::LoadU(d16_half, src + 8));   // src[8..11]
    const auto s3 = hn::PromoteTo(d32, hn::LoadU(d16_half, src + 12));  // src[12..15]

    // E[k] = src[k] + src[15-k], O[k] = src[k] - src[15-k] for k = 0..7.
    // E[0..3] = s0 + Reverse(s3),  E[4..7] = s1 + Reverse(s2)
    const auto s3_rev = hn::Reverse(d32, s3);  // [src[15], src[14], src[13], src[12]]
    const auto s2_rev = hn::Reverse(d32, s2);  // [src[11], src[10], src[9], src[8]]

    const auto E_03 = hn::Add(s0, s3_rev);  // E[0..3]
    const auto E_47 = hn::Add(s1, s2_rev);  // E[4..7]
    const auto O_03 = hn::Sub(s0, s3_rev);  // O[0..3]
    const auto O_47 = hn::Sub(s1, s2_rev);  // O[4..7]

    // EE[k] = E[k] + E[7-k], EO[k] = E[k] - E[7-k] for k = 0..3.
    const auto E_47_rev = hn::Reverse(d32, E_47);  // [E[7], E[6], E[5], E[4]]
    const auto EEv = hn::Add(E_03, E_47_rev);      // EE[0..3]
    const auto EOv = hn::Sub(E_03, E_47_rev);      // EO[0..3]

    alignas(16) int32_t EE[4];
    alignas(16) int32_t EO[4];
    hn::StoreU(EEv, d32, EE);
    hn::StoreU(EOv, d32, EO);
#else
    int32_t E[8];
    int32_t O_arr_scalar[8];
    for (int k = 0; k < 8; ++k)
    {
        E[k] = src[k] + src[15 - k];
        O_arr_scalar[k] = src[k] - src[15 - k];
    }
    int32_t EE[4];
    int32_t EO[4];
    for (int k = 0; k < 4; ++k)
    {
        EE[k] = E[k] + E[7 - k];
        EO[k] = E[k] - E[7 - k];
    }
#endif

    // EEE and EEO — length-2 each, scalar.
    const int32_t EEE0 = EE[0] + EE[3];
    const int32_t EEE1 = EE[1] + EE[2];
    const int32_t EEO0 = EE[0] - EE[3];
    const int32_t EEO1 = EE[1] - EE[2];

    // dst[0], dst[8*stride]: length-2 dot product on EEE.
    dst[0 * dstStride] = static_cast<int16_t>(
        (kT16[0][0] * EEE0 + kT16[0][1] * EEE1 + kAdd) >> Shift);
    dst[8 * dstStride] = static_cast<int16_t>(
        (kT16[8][0] * EEE0 + kT16[8][1] * EEE1 + kAdd) >> Shift);

    // dst[4*stride], dst[12*stride]: length-2 dot product on EEO.
    dst[4 * dstStride] = static_cast<int16_t>(
        (kT16[4][0] * EEO0 + kT16[4][1] * EEO1 + kAdd) >> Shift);
    dst[12 * dstStride] = static_cast<int16_t>(
        (kT16[12][0] * EEO0 + kT16[12][1] * EEO1 + kAdd) >> Shift);

    // dst[{2,6,10,14}*stride]: length-4 dot products on EO. Scalar.
    for (int k = 2; k < 16; k += 4)
    {
        const int32_t sum = kT16[k][0] * EO[0] + kT16[k][1] * EO[1]
                          + kT16[k][2] * EO[2] + kT16[k][3] * EO[3];
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }

    // dst[odd*stride]: length-8 dot products on O.
#if HWY_TARGET != HWY_SCALAR
    if constexpr (Shift <= 8)
    {
        // Pass 1 fast path: O[k] = src[k] - src[15-k] fits in int16
        // because src is bit-depth-bounded (max ~4095 for bd12). Compute
        // O as an 8-lane int16 vector and use WidenMulPairwiseAdd, which
        // maps to smull/smlal on NEON — ~2x the throughput of int32 Mul.
        const hn::CappedTag<int16_t, 8> d16_full;
        const hn::Repartition<int32_t, decltype(d16_full)> d32_wide;

        const auto lo16 = hn::LoadU(d16_full, src);       // src[0..7]
        const auto hi16 = hn::LoadU(d16_full, src + 8);   // src[8..15]
        const auto hi16_rev = hn::Reverse(d16_full, hi16);
        const auto O16 = hn::Sub(lo16, hi16_rev);         // O[0..7] in int16

        for (int k = 1; k < 16; k += 2)
        {
            const auto coef16 = hn::LoadU(d16_full, &kT16[k][0]);
            const auto prod = hn::WidenMulPairwiseAdd(d32_wide, O16, coef16);
            const int32_t sum = hn::ReduceSum(d32_wide, prod);
            dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
        }
    }
    else
    {
        // Pass 2: O can exceed int16 range, so stay in int32.
        // O_03 and O_47 are already loaded as int32 vectors.
        for (int k = 1; k < 16; k += 2)
        {
            const auto c_03 = hn::PromoteTo(d32, hn::LoadU(d16_half, &kT16[k][0]));
            const auto c_47 = hn::PromoteTo(d32, hn::LoadU(d16_half, &kT16[k][4]));
            const auto prod = hn::Add(hn::Mul(O_03, c_03), hn::Mul(O_47, c_47));
            const int32_t sum = hn::ReduceSum(d32, prod);
            dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
        }
    }
#else
    for (int k = 1; k < 16; k += 2)
    {
        const int32_t sum = kT16[k][0] * O_arr_scalar[0] + kT16[k][1] * O_arr_scalar[1]
                          + kT16[k][2] * O_arr_scalar[2] + kT16[k][3] * O_arr_scalar[3]
                          + kT16[k][4] * O_arr_scalar[4] + kT16[k][5] * O_arr_scalar[5]
                          + kT16[k][6] * O_arr_scalar[6] + kT16[k][7] * O_arr_scalar[7];
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }
#endif
}

// Template on Shift1 (bit-depth-dependent). Shift2 is fixed at 10 for Dct16.
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

// One row of a 32x32 DCT. Factored butterfly mirroring partialButterfly32.
// Butterfly runs on 4-lane int32 vectors. Length-16 dot products on O
// run as four vector Mul + three Add + ReduceSum each. Length-8 dot
// products on EO run as two vector Mul + Add + ReduceSum. Shorter
// products (length-2 on EEEE/EEEO, length-4 on EEO) stay scalar.
template <int Shift>
static HWY_INLINE void Dct32Row(const int16_t *src, int16_t *dst, intptr_t dstStride)
{
    constexpr int32_t kAdd = 1 << (Shift - 1);

#if HWY_TARGET != HWY_SCALAR
    const hn::CappedTag<int32_t, 4> d32;
    const hn::Rebind<int16_t, decltype(d32)> d16_half;  // 4-lane int16

    // Load src[0..31] as eight 4-lane int32 vectors.
    const auto s0 = hn::PromoteTo(d32, hn::LoadU(d16_half, src + 0));
    const auto s1 = hn::PromoteTo(d32, hn::LoadU(d16_half, src + 4));
    const auto s2 = hn::PromoteTo(d32, hn::LoadU(d16_half, src + 8));
    const auto s3 = hn::PromoteTo(d32, hn::LoadU(d16_half, src + 12));
    const auto s4 = hn::PromoteTo(d32, hn::LoadU(d16_half, src + 16));
    const auto s5 = hn::PromoteTo(d32, hn::LoadU(d16_half, src + 20));
    const auto s6 = hn::PromoteTo(d32, hn::LoadU(d16_half, src + 24));
    const auto s7 = hn::PromoteTo(d32, hn::LoadU(d16_half, src + 28));

    // Stage 1: E[0..15] = src[0..15] + src[31..16],
    //          O[0..15] = src[0..15] - src[31..16].
    //   E[0..3]   <- s0 + Reverse(s7)
    //   E[4..7]   <- s1 + Reverse(s6)
    //   E[8..11]  <- s2 + Reverse(s5)
    //   E[12..15] <- s3 + Reverse(s4)
    const auto s7r = hn::Reverse(d32, s7);
    const auto s6r = hn::Reverse(d32, s6);
    const auto s5r = hn::Reverse(d32, s5);
    const auto s4r = hn::Reverse(d32, s4);

    const auto E_03   = hn::Add(s0, s7r);
    const auto E_47   = hn::Add(s1, s6r);
    const auto E_811  = hn::Add(s2, s5r);
    const auto E_1215 = hn::Add(s3, s4r);
    const auto O_a    = hn::Sub(s0, s7r);  // O[0..3]
    const auto O_b    = hn::Sub(s1, s6r);  // O[4..7]
    const auto O_c    = hn::Sub(s2, s5r);  // O[8..11]
    const auto O_d    = hn::Sub(s3, s4r);  // O[12..15]

    // Stage 2: EE[k] = E[k] + E[15-k], EO[k] = E[k] - E[15-k] for k=0..7.
    //   EE[0..3] <- E_03   + Reverse(E_1215)
    //   EE[4..7] <- E_47   + Reverse(E_811)
    const auto E_1215r = hn::Reverse(d32, E_1215);
    const auto E_811r  = hn::Reverse(d32, E_811);

    const auto EE_03 = hn::Add(E_03, E_1215r);  // EE[0..3]
    const auto EE_47 = hn::Add(E_47, E_811r);   // EE[4..7]
    const auto EO_03 = hn::Sub(E_03, E_1215r);  // EO[0..3]
    const auto EO_47 = hn::Sub(E_47, E_811r);   // EO[4..7]

    // Stage 3: EEE[k] = EE[k] + EE[7-k], EEO[k] = EE[k] - EE[7-k] for k=0..3.
    const auto EE_47r = hn::Reverse(d32, EE_47);
    const auto EEEv = hn::Add(EE_03, EE_47r);   // EEE[0..3]
    const auto EEOv = hn::Sub(EE_03, EE_47r);   // EEO[0..3]

    alignas(16) int32_t EEE[4];
    alignas(16) int32_t EEO[4];
    alignas(16) int32_t EO[8];
    hn::StoreU(EEEv, d32, EEE);
    hn::StoreU(EEOv, d32, EEO);
    hn::StoreU(EO_03, d32, EO);
    hn::StoreU(EO_47, d32, EO + 4);
#else
    int32_t E[16];
    int32_t O_scalar[16];
    for (int k = 0; k < 16; ++k)
    {
        E[k] = src[k] + src[31 - k];
        O_scalar[k] = src[k] - src[31 - k];
    }
    int32_t EE[8];
    int32_t EO[8];
    for (int k = 0; k < 8; ++k)
    {
        EE[k] = E[k] + E[15 - k];
        EO[k] = E[k] - E[15 - k];
    }
    int32_t EEE[4];
    int32_t EEO[4];
    for (int k = 0; k < 4; ++k)
    {
        EEE[k] = EE[k] + EE[7 - k];
        EEO[k] = EE[k] - EE[7 - k];
    }
#endif

    // Stage 4: EEEE / EEEO — length-2 each, scalar.
    const int32_t EEEE0 = EEE[0] + EEE[3];
    const int32_t EEEE1 = EEE[1] + EEE[2];
    const int32_t EEEO0 = EEE[0] - EEE[3];
    const int32_t EEEO1 = EEE[1] - EEE[2];

    // dst[0], dst[16*stride]: length-2 dot product on EEEE.
    dst[0 * dstStride] = static_cast<int16_t>(
        (kT32[0][0] * EEEE0 + kT32[0][1] * EEEE1 + kAdd) >> Shift);
    dst[16 * dstStride] = static_cast<int16_t>(
        (kT32[16][0] * EEEE0 + kT32[16][1] * EEEE1 + kAdd) >> Shift);

    // dst[8*stride], dst[24*stride]: length-2 dot product on EEEO.
    dst[8 * dstStride] = static_cast<int16_t>(
        (kT32[8][0] * EEEO0 + kT32[8][1] * EEEO1 + kAdd) >> Shift);
    dst[24 * dstStride] = static_cast<int16_t>(
        (kT32[24][0] * EEEO0 + kT32[24][1] * EEEO1 + kAdd) >> Shift);

    // dst[{4,12,20,28}*stride]: length-4 dot products on EEO. Scalar.
    for (int k = 4; k < 32; k += 8)
    {
        const int32_t sum = kT32[k][0] * EEO[0] + kT32[k][1] * EEO[1]
                          + kT32[k][2] * EEO[2] + kT32[k][3] * EEO[3];
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }

#if HWY_TARGET != HWY_SCALAR
    // dst[{2,6,10,14,18,22,26,30}*stride]: length-8 dot products on EO.
    // Vectorize via two Mul + Add + ReduceSum per output.
    for (int k = 2; k < 32; k += 4)
    {
        const auto c_03 = hn::PromoteTo(d32, hn::LoadU(d16_half, &kT32[k][0]));
        const auto c_47 = hn::PromoteTo(d32, hn::LoadU(d16_half, &kT32[k][4]));
        const auto prod = hn::Add(hn::Mul(EO_03, c_03), hn::Mul(EO_47, c_47));
        const int32_t sum = hn::ReduceSum(d32, prod);
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }

    // dst[odd*stride]: length-16 dot products on O.
    if constexpr (Shift <= 8)
    {
        // Pass 1 fast path: O[k] = src[k] - src[31-k] fits in int16
        // because src is bit-depth-bounded. Compute O into an aligned
        // 16-element int16 buffer, then run the length-16 dot product
        // with a cap-16 int16 tag. On AVX2 (native 16-lane int16) this
        // issues one vpmaddwd per output; on NEON (native 8-lane) the
        // inner loop runs twice, matching the previous behavior.
        // This is the single biggest hotspot on x86 — 16 outputs × 16.
        const hn::CappedTag<int16_t, 8> d16_half_i16;

        const auto a16 = hn::LoadU(d16_half_i16, src);        // src[0..7]
        const auto b16 = hn::LoadU(d16_half_i16, src + 8);    // src[8..15]
        const auto c16 = hn::LoadU(d16_half_i16, src + 16);   // src[16..23]
        const auto e16 = hn::LoadU(d16_half_i16, src + 24);   // src[24..31]

        const auto e16_rev = hn::Reverse(d16_half_i16, e16);  // [src[31..24]]
        const auto c16_rev = hn::Reverse(d16_half_i16, c16);  // [src[23..16]]

        // O[0..7]  = src[0..7]  - src[31..24]
        // O[8..15] = src[8..15] - src[23..16]
        alignas(32) int16_t O_arr[16];
        hn::StoreU(hn::Sub(a16, e16_rev), d16_half_i16, O_arr);
        hn::StoreU(hn::Sub(b16, c16_rev), d16_half_i16, O_arr + 8);

        // Cap-16 int16 tag: 16 lanes on AVX2+, 8 lanes on NEON.
        const hn::CappedTag<int16_t, 16> d16_cap16;
        const hn::Repartition<int32_t, decltype(d16_cap16)> d32_cap;
        using VecD32Cap = hn::Vec<decltype(d32_cap)>;
        const size_t kLanes16 = hn::Lanes(d16_cap16);

        for (int k = 1; k < 32; k += 2)
        {
            VecD32Cap accum = hn::Zero(d32_cap);
            for (size_t i = 0; i < 16; i += kLanes16)
            {
                const auto o_slice    = hn::LoadU(d16_cap16, &O_arr[i]);
                const auto coef_slice = hn::LoadU(d16_cap16, &kT32[k][i]);
                accum = hn::Add(accum,
                                hn::WidenMulPairwiseAdd(d32_cap, o_slice, coef_slice));
            }
            const int32_t sum = hn::ReduceSum(d32_cap, accum);
            dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
        }
    }
    else
    {
        // Pass 2: int32 form. Four Mul + three Add + ReduceSum per output.
        for (int k = 1; k < 32; k += 2)
        {
            const auto c_a = hn::PromoteTo(d32, hn::LoadU(d16_half, &kT32[k][0]));
            const auto c_b = hn::PromoteTo(d32, hn::LoadU(d16_half, &kT32[k][4]));
            const auto c_c = hn::PromoteTo(d32, hn::LoadU(d16_half, &kT32[k][8]));
            const auto c_d = hn::PromoteTo(d32, hn::LoadU(d16_half, &kT32[k][12]));

            const auto p_ab = hn::Add(hn::Mul(O_a, c_a), hn::Mul(O_b, c_b));
            const auto p_cd = hn::Add(hn::Mul(O_c, c_c), hn::Mul(O_d, c_d));
            const int32_t sum = hn::ReduceSum(d32, hn::Add(p_ab, p_cd));
            dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
        }
    }
#else
    // SCALAR fallback for length-8 and length-16 dot products.
    for (int k = 2; k < 32; k += 4)
    {
        const int32_t sum = kT32[k][0] * EO[0] + kT32[k][1] * EO[1]
                          + kT32[k][2] * EO[2] + kT32[k][3] * EO[3]
                          + kT32[k][4] * EO[4] + kT32[k][5] * EO[5]
                          + kT32[k][6] * EO[6] + kT32[k][7] * EO[7];
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }
    for (int k = 1; k < 32; k += 2)
    {
        const int32_t sum = kT32[k][0]  * O_scalar[0]  + kT32[k][1]  * O_scalar[1]
                          + kT32[k][2]  * O_scalar[2]  + kT32[k][3]  * O_scalar[3]
                          + kT32[k][4]  * O_scalar[4]  + kT32[k][5]  * O_scalar[5]
                          + kT32[k][6]  * O_scalar[6]  + kT32[k][7]  * O_scalar[7]
                          + kT32[k][8]  * O_scalar[8]  + kT32[k][9]  * O_scalar[9]
                          + kT32[k][10] * O_scalar[10] + kT32[k][11] * O_scalar[11]
                          + kT32[k][12] * O_scalar[12] + kT32[k][13] * O_scalar[13]
                          + kT32[k][14] * O_scalar[14] + kT32[k][15] * O_scalar[15];
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }
#endif
}

// Template on Shift1 (bit-depth-dependent). Shift2 is fixed at 11 for Dct32.
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

} // namespace vca::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace vca {

HWY_EXPORT(Dct8Impl_bd8);
HWY_EXPORT(Dct8Impl_bd10);
HWY_EXPORT(Dct8Impl_bd12);
HWY_EXPORT(Dct16Impl_bd8);
HWY_EXPORT(Dct16Impl_bd10);
HWY_EXPORT(Dct16Impl_bd12);
HWY_EXPORT(Dct32Impl_bd8);
HWY_EXPORT(Dct32Impl_bd10);
HWY_EXPORT(Dct32Impl_bd12);

void Dct8(const int16_t *src, int16_t *dst, intptr_t srcStride, unsigned bitDepth)
{
    switch (bitDepth)
    {
        case 8:  HWY_DYNAMIC_DISPATCH(Dct8Impl_bd8) (src, dst, srcStride); return;
        case 10: HWY_DYNAMIC_DISPATCH(Dct8Impl_bd10)(src, dst, srcStride); return;
        case 12: HWY_DYNAMIC_DISPATCH(Dct8Impl_bd12)(src, dst, srcStride); return;
    }
    // Unsupported bit depth - the analyzer only calls with 8/10/12, so this
    // is unreachable in practice.
}

void Dct16(const int16_t *src, int16_t *dst, intptr_t srcStride, unsigned bitDepth)
{
    switch (bitDepth)
    {
        case 8:  HWY_DYNAMIC_DISPATCH(Dct16Impl_bd8) (src, dst, srcStride); return;
        case 10: HWY_DYNAMIC_DISPATCH(Dct16Impl_bd10)(src, dst, srcStride); return;
        case 12: HWY_DYNAMIC_DISPATCH(Dct16Impl_bd12)(src, dst, srcStride); return;
    }
}

void Dct32(const int16_t *src, int16_t *dst, intptr_t srcStride, unsigned bitDepth)
{
    switch (bitDepth)
    {
        case 8:  HWY_DYNAMIC_DISPATCH(Dct32Impl_bd8) (src, dst, srcStride); return;
        case 10: HWY_DYNAMIC_DISPATCH(Dct32Impl_bd10)(src, dst, srcStride); return;
        case 12: HWY_DYNAMIC_DISPATCH(Dct32Impl_bd12)(src, dst, srcStride); return;
    }
}

} // namespace vca
#endif
