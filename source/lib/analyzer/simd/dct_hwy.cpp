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

// One row of an 8x8 DCT: writes 8 output coefficients, each into dst[k*dstStride]
// (so both the row-major and column-major passes share this helper).
//
// Shift is a template parameter so the `>> Shift` and `1 << (Shift - 1)` are
// compile-time constants the compiler can fold into immediate operands.
template <int Shift>
static HWY_INLINE void Dct8Row(const int16_t *src, int16_t *dst, intptr_t dstStride)
{
    // Cap to 8 lanes: an 8x8 DCT row is always exactly 8 int16 values,
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

// Non-template thunks - HWY_EXPORT cannot take template arguments directly.
// One per supported bit depth; Shift1 = 2 + bitDepth - 8.
void Dct8Impl_bd8 (const int16_t *s, int16_t *d, intptr_t st) { Dct8ImplT<2>(s, d, st); }
void Dct8Impl_bd10(const int16_t *s, int16_t *d, intptr_t st) { Dct8ImplT<4>(s, d, st); }
void Dct8Impl_bd12(const int16_t *s, int16_t *d, intptr_t st) { Dct8ImplT<6>(s, d, st); }

// One row of a 16x16 DCT. A 16-wide row doesn't fit in a single int16 vector
// on SSE4/NEON (8 lanes), so we split the row into two 8-lane halves, run
// WidenMulPairwiseAdd on each half, sum the int32 partial vectors, then
// ReduceSum to scalar.
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

// Placeholder so the header declarations resolve until Task 6.
void Dct32(const int16_t *, int16_t *, intptr_t, unsigned) {}

} // namespace vca
#endif
