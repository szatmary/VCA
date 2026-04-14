/* source/lib/analyzer/simd/targets.h */
#pragma once

#include <string>

namespace vca::simd {

// Returns a human-readable name for the SIMD target Highway will use at
// runtime (e.g. "AVX2", "NEON", "Scalar"). Used for startup logging.
std::string CurrentTargetName();

// Constrain Highway to a specific target mask (used by the --asm CLI flag).
// Pass one of the CLI strings: "auto", "none", "sse2", "ssse3", "sse4",
// "avx2", "neon", "neon_dotprod". Must be called exactly once, before any
// DCT/entropy work. Unknown strings are silently treated as "auto".
void ConstrainTargetsForCli(const std::string &cli_value);

} // namespace vca::simd
