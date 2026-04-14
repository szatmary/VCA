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
    // allowed set. Called once at startup before any DCT work. Unknown
    // strings (including "auto") leave Highway's default in place.
    //
    // Supports every Highway 1.3.0 target name, plus the legacy short
    // strings from the original VCA --asm flag (sse2/ssse3 were collapsed
    // into HWY_SSE4 at first; they now map 1:1).

    // --- Special / legacy ---
    if (cli_value == "none" || cli_value == "scalar")
        hwy::SetSupportedTargetsForTest(HWY_SCALAR);
    else if (cli_value == "emu128")
        hwy::SetSupportedTargetsForTest(HWY_EMU128);

    // --- x86 ---
    else if (cli_value == "sse2")
        hwy::SetSupportedTargetsForTest(HWY_SSE2);
    else if (cli_value == "ssse3")
        hwy::SetSupportedTargetsForTest(HWY_SSSE3);
    else if (cli_value == "sse4")
        hwy::SetSupportedTargetsForTest(HWY_SSE4);
    else if (cli_value == "avx2")
        hwy::SetSupportedTargetsForTest(HWY_AVX2);
    else if (cli_value == "avx3")
        hwy::SetSupportedTargetsForTest(HWY_AVX3);
    else if (cli_value == "avx3_dl")
        hwy::SetSupportedTargetsForTest(HWY_AVX3_DL);
    else if (cli_value == "avx3_zen4")
        hwy::SetSupportedTargetsForTest(HWY_AVX3_ZEN4);
    else if (cli_value == "avx3_spr")
        hwy::SetSupportedTargetsForTest(HWY_AVX3_SPR);
    else if (cli_value == "avx10_2")
        hwy::SetSupportedTargetsForTest(HWY_AVX10_2);

    // --- ARM ---
    else if (cli_value == "neon" || cli_value == "neon_dotprod")
        hwy::SetSupportedTargetsForTest(HWY_NEON);
    else if (cli_value == "neon_bf16")
        hwy::SetSupportedTargetsForTest(HWY_NEON_BF16);
    else if (cli_value == "neon_without_aes")
        hwy::SetSupportedTargetsForTest(HWY_NEON_WITHOUT_AES);
    else if (cli_value == "sve")
        hwy::SetSupportedTargetsForTest(HWY_SVE);
    else if (cli_value == "sve2")
        hwy::SetSupportedTargetsForTest(HWY_SVE2);
    else if (cli_value == "sve_256")
        hwy::SetSupportedTargetsForTest(HWY_SVE_256);
    else if (cli_value == "sve2_128")
        hwy::SetSupportedTargetsForTest(HWY_SVE2_128);

    // --- RISC-V / LoongArch ---
    else if (cli_value == "rvv")
        hwy::SetSupportedTargetsForTest(HWY_RVV);
    else if (cli_value == "lsx")
        hwy::SetSupportedTargetsForTest(HWY_LSX);
    else if (cli_value == "lasx")
        hwy::SetSupportedTargetsForTest(HWY_LASX);

    // --- IBM POWER / Z ---
    else if (cli_value == "ppc8")
        hwy::SetSupportedTargetsForTest(HWY_PPC8);
    else if (cli_value == "ppc9")
        hwy::SetSupportedTargetsForTest(HWY_PPC9);
    else if (cli_value == "ppc10")
        hwy::SetSupportedTargetsForTest(HWY_PPC10);
    else if (cli_value == "z14")
        hwy::SetSupportedTargetsForTest(HWY_Z14);
    else if (cli_value == "z15")
        hwy::SetSupportedTargetsForTest(HWY_Z15);

    // --- WebAssembly ---
    else if (cli_value == "wasm")
        hwy::SetSupportedTargetsForTest(HWY_WASM);
    else if (cli_value == "wasm_emu256")
        hwy::SetSupportedTargetsForTest(HWY_WASM_EMU256);

    // "auto" or anything unrecognized: leave Highway default (no constraint).
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
