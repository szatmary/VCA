/* source/lib/analyzer/simd/targets.cpp */
#include <analyzer/simd/targets.h>

#include <hwy/targets.h>

namespace vca::simd {

std::string CurrentTargetName()
{
    // Highway's SupportedTargets() returns the bitmask of targets available
    // to the runtime dispatcher, already honoring any DisableTargets() or
    // SetSupportedTargetsForTest() calls. The lowest-set bit is the "best"
    // target (Highway orders bits by preference). TargetName expects a
    // single target bit.
    const int64_t supported = hwy::SupportedTargets();
    if (supported == 0)
        return "SCALAR";
    const int64_t lowest = supported & -supported;
    return hwy::TargetName(lowest);
}

void ConstrainTargetsForCli(const std::string &cli_value)
{
    // Process-global constraint. Highway picks the best target within the
    // allowed set. Called once at startup before any DCT/entropy work.
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

void ConstrainTargetsForCli(CpuSimd hint)
{
    switch (hint)
    {
        case CpuSimd::Autodetect:
            // No constraint: Highway uses its full supported-target set.
            break;
        case CpuSimd::None:
            hwy::SetSupportedTargetsForTest(HWY_SCALAR);
            break;
        case CpuSimd::SSE2:
            hwy::SetSupportedTargetsForTest(HWY_SSE2);
            break;
        case CpuSimd::SSSE3:
            hwy::SetSupportedTargetsForTest(HWY_SSSE3);
            break;
        case CpuSimd::SSE4:
            hwy::SetSupportedTargetsForTest(HWY_SSE4);
            break;
        case CpuSimd::AVX2:
            hwy::SetSupportedTargetsForTest(HWY_AVX2);
            break;
        case CpuSimd::NEON:
            hwy::SetSupportedTargetsForTest(HWY_NEON);
            break;
        case CpuSimd::AVX10_2:
            hwy::SetSupportedTargetsForTest(HWY_AVX10_2);
            break;
        case CpuSimd::AVX3_SPR:
            hwy::SetSupportedTargetsForTest(HWY_AVX3_SPR);
            break;
        case CpuSimd::AVX3_ZEN4:
            hwy::SetSupportedTargetsForTest(HWY_AVX3_ZEN4);
            break;
        case CpuSimd::AVX3_DL:
            hwy::SetSupportedTargetsForTest(HWY_AVX3_DL);
            break;
        case CpuSimd::AVX3:
            hwy::SetSupportedTargetsForTest(HWY_AVX3);
            break;
        case CpuSimd::SVE2_128:
            hwy::SetSupportedTargetsForTest(HWY_SVE2_128);
            break;
        case CpuSimd::SVE_256:
            hwy::SetSupportedTargetsForTest(HWY_SVE_256);
            break;
        case CpuSimd::SVE2:
            hwy::SetSupportedTargetsForTest(HWY_SVE2);
            break;
        case CpuSimd::SVE:
            hwy::SetSupportedTargetsForTest(HWY_SVE);
            break;
        case CpuSimd::NEON_BF16:
            hwy::SetSupportedTargetsForTest(HWY_NEON_BF16);
            break;
        case CpuSimd::NEON_WITHOUT_AES:
            hwy::SetSupportedTargetsForTest(HWY_NEON_WITHOUT_AES);
            break;
        case CpuSimd::RVV:
            hwy::SetSupportedTargetsForTest(HWY_RVV);
            break;
        case CpuSimd::LASX:
            hwy::SetSupportedTargetsForTest(HWY_LASX);
            break;
        case CpuSimd::LSX:
            hwy::SetSupportedTargetsForTest(HWY_LSX);
            break;
        case CpuSimd::PPC10:
            hwy::SetSupportedTargetsForTest(HWY_PPC10);
            break;
        case CpuSimd::PPC9:
            hwy::SetSupportedTargetsForTest(HWY_PPC9);
            break;
        case CpuSimd::PPC8:
            hwy::SetSupportedTargetsForTest(HWY_PPC8);
            break;
        case CpuSimd::Z15:
            hwy::SetSupportedTargetsForTest(HWY_Z15);
            break;
        case CpuSimd::Z14:
            hwy::SetSupportedTargetsForTest(HWY_Z14);
            break;
        case CpuSimd::WASM_EMU256:
            hwy::SetSupportedTargetsForTest(HWY_WASM_EMU256);
            break;
        case CpuSimd::WASM:
            hwy::SetSupportedTargetsForTest(HWY_WASM);
            break;
        case CpuSimd::EMU128:
            hwy::SetSupportedTargetsForTest(HWY_EMU128);
            break;
    }
}

} // namespace vca::simd
