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
