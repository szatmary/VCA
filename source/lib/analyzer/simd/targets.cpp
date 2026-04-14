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

} // namespace vca::simd
