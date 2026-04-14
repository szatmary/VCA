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
 * All butterfly arithmetic and dot products run in int32 scalar math.
 * An int16-lane butterfly (e.g. via hn::Reverse + Add/Sub) cannot stay
 * bit-exact with the scalar reference: the second DCT pass operates on
 * the first pass's int16 coefficients, and E[k] = src[k] + src[N-1-k]
 * on those inputs can overflow int16. The scalar reference uses `int`
 * for the E/O/EE/... temporaries, so we match that precision here.
 * The speedup over the prior matrix-vector kernels comes entirely from
 * the reduced multiply count, not from SIMD lane throughput on the
 * dot products themselves. The file still lives in the Highway per-
 * target translation unit so HWY_EXPORT / HWY_DYNAMIC_DISPATCH dispatch
 * continues to work the same way as the rest of the analyzer kernels.
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
// Factored butterfly form, mirroring partialButterfly8 in
// DCTTransformsNative.cpp. All arithmetic is scalar int32 — see the
// file header for why an int16-lane butterfly would break bit-exactness.
template <int Shift>
static HWY_INLINE void Dct8Row(const int16_t *src, int16_t *dst, intptr_t dstStride)
{
    constexpr int32_t kAdd = 1 << (Shift - 1);

    // E[k] = src[k] + src[7-k], O[k] = src[k] - src[7-k] for k = 0..3.
    // Intermediate values are computed in int32 because the second DCT pass
    // can produce pass-1 coefficients whose sum exceeds int16 range.
    const int E0 = src[0] + src[7];
    const int E1 = src[1] + src[6];
    const int E2 = src[2] + src[5];
    const int E3 = src[3] + src[4];
    const int O0 = src[0] - src[7];
    const int O1 = src[1] - src[6];
    const int O2 = src[2] - src[5];
    const int O3 = src[3] - src[4];

    // EE[0..1] and EO[0..1] — scalar, only 2 elements each.
    const int EE0 = E0 + E3;
    const int EE1 = E1 + E2;
    const int EO0 = E0 - E3;
    const int EO1 = E1 - E2;

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
        (kT8[1][0] * O0 + kT8[1][1] * O1 + kT8[1][2] * O2 + kT8[1][3] * O3 + kAdd) >> Shift);
    dst[3 * dstStride] = static_cast<int16_t>(
        (kT8[3][0] * O0 + kT8[3][1] * O1 + kT8[3][2] * O2 + kT8[3][3] * O3 + kAdd) >> Shift);
    dst[5 * dstStride] = static_cast<int16_t>(
        (kT8[5][0] * O0 + kT8[5][1] * O1 + kT8[5][2] * O2 + kT8[5][3] * O3 + kAdd) >> Shift);
    dst[7 * dstStride] = static_cast<int16_t>(
        (kT8[7][0] * O0 + kT8[7][1] * O1 + kT8[7][2] * O2 + kT8[7][3] * O3 + kAdd) >> Shift);
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
// All butterfly arithmetic runs in int32 to match the scalar reference
// bit-for-bit: pass-2 inputs (pass-1 coefficients) can exceed int16 range
// when summed/differenced, so the butterfly cannot stay in int16 lanes.
template <int Shift>
static HWY_INLINE void Dct16Row(const int16_t *src, int16_t *dst, intptr_t dstStride)
{
    constexpr int32_t kAdd = 1 << (Shift - 1);

    // E[k] = src[k] + src[15-k], O[k] = src[k] - src[15-k] for k = 0..7.
    int E[8];
    int O[8];
    for (int k = 0; k < 8; ++k)
    {
        E[k] = src[k] + src[15 - k];
        O[k] = src[k] - src[15 - k];
    }

    // EE[k] = E[k] + E[7-k], EO[k] = E[k] - E[7-k] for k = 0..3.
    const int EE0 = E[0] + E[7];
    const int EE1 = E[1] + E[6];
    const int EE2 = E[2] + E[5];
    const int EE3 = E[3] + E[4];
    const int EO0 = E[0] - E[7];
    const int EO1 = E[1] - E[6];
    const int EO2 = E[2] - E[5];
    const int EO3 = E[3] - E[4];

    // EEE and EEO — length-2 each.
    const int EEE0 = EE0 + EE3;
    const int EEE1 = EE1 + EE2;
    const int EEO0 = EE0 - EE3;
    const int EEO1 = EE1 - EE2;

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

    // dst[{2,6,10,14}*stride]: length-4 dot products on EO.
    for (int k = 2; k < 16; k += 4)
    {
        const int sum = kT16[k][0] * EO0 + kT16[k][1] * EO1
                      + kT16[k][2] * EO2 + kT16[k][3] * EO3;
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }

    // dst[odd*stride]: length-8 dot products on O.
    for (int k = 1; k < 16; k += 2)
    {
        const int sum = kT16[k][0] * O[0] + kT16[k][1] * O[1]
                      + kT16[k][2] * O[2] + kT16[k][3] * O[3]
                      + kT16[k][4] * O[4] + kT16[k][5] * O[5]
                      + kT16[k][6] * O[6] + kT16[k][7] * O[7];
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }
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
// All butterfly arithmetic runs in int32: pass-2 inputs (pass-1 coefficients)
// can exceed int16 range when summed/differenced.
template <int Shift>
static HWY_INLINE void Dct32Row(const int16_t *src, int16_t *dst, intptr_t dstStride)
{
    constexpr int32_t kAdd = 1 << (Shift - 1);

    // E[k] = src[k] + src[31-k], O[k] = src[k] - src[31-k] for k = 0..15.
    int E[16];
    int O[16];
    for (int k = 0; k < 16; ++k)
    {
        E[k] = src[k] + src[31 - k];
        O[k] = src[k] - src[31 - k];
    }

    // EE[k] = E[k] + E[15-k], EO[k] = E[k] - E[15-k] for k = 0..7.
    int EE[8];
    int EO[8];
    for (int k = 0; k < 8; ++k)
    {
        EE[k] = E[k] + E[15 - k];
        EO[k] = E[k] - E[15 - k];
    }

    // EEE[k] = EE[k] + EE[7-k], EEO[k] = EE[k] - EE[7-k] for k = 0..3.
    const int EEE0 = EE[0] + EE[7];
    const int EEE1 = EE[1] + EE[6];
    const int EEE2 = EE[2] + EE[5];
    const int EEE3 = EE[3] + EE[4];
    const int EEO0 = EE[0] - EE[7];
    const int EEO1 = EE[1] - EE[6];
    const int EEO2 = EE[2] - EE[5];
    const int EEO3 = EE[3] - EE[4];

    // EEEE / EEEO — length-2 each.
    const int EEEE0 = EEE0 + EEE3;
    const int EEEE1 = EEE1 + EEE2;
    const int EEEO0 = EEE0 - EEE3;
    const int EEEO1 = EEE1 - EEE2;

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

    // dst[{4,12,20,28}*stride]: length-4 dot products on EEO.
    for (int k = 4; k < 32; k += 8)
    {
        const int sum = kT32[k][0] * EEO0 + kT32[k][1] * EEO1
                      + kT32[k][2] * EEO2 + kT32[k][3] * EEO3;
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }

    // dst[{2,6,10,14,18,22,26,30}*stride]: length-8 dot products on EO.
    for (int k = 2; k < 32; k += 4)
    {
        const int sum = kT32[k][0] * EO[0] + kT32[k][1] * EO[1]
                      + kT32[k][2] * EO[2] + kT32[k][3] * EO[3]
                      + kT32[k][4] * EO[4] + kT32[k][5] * EO[5]
                      + kT32[k][6] * EO[6] + kT32[k][7] * EO[7];
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }

    // dst[odd*stride]: length-16 dot products on O.
    for (int k = 1; k < 32; k += 2)
    {
        const int sum = kT32[k][0] * O[0]   + kT32[k][1] * O[1]
                      + kT32[k][2] * O[2]   + kT32[k][3] * O[3]
                      + kT32[k][4] * O[4]   + kT32[k][5] * O[5]
                      + kT32[k][6] * O[6]   + kT32[k][7] * O[7]
                      + kT32[k][8] * O[8]   + kT32[k][9] * O[9]
                      + kT32[k][10] * O[10] + kT32[k][11] * O[11]
                      + kT32[k][12] * O[12] + kT32[k][13] * O[13]
                      + kT32[k][14] * O[14] + kT32[k][15] * O[15];
        dst[k * dstStride] = static_cast<int16_t>((sum + kAdd) >> Shift);
    }
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
