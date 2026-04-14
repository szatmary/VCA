/* source/lib/analyzer/simd/targets.h */
#pragma once

#include <string>
#include <vcaLib.h>

namespace vca::simd {

// Returns a human-readable name for the SIMD target Highway will use at
// runtime (e.g. "AVX2", "NEON", "Scalar"). Used for startup logging.
std::string CurrentTargetName();

// Constrain Highway to a specific target mask, interpreted from a CLI
// string ("auto", "none", "sse2", "ssse3", "sse4", "avx2", "neon",
// "neon_dotprod"). Must be called once at startup before any DCT work.
// Unknown strings are silently treated as "auto".
void ConstrainTargetsForCli(const std::string &cli_value);

// Constrain Highway from a vca_param::cpuSimd hint. Maps each CpuSimd
// enum value 1:1 onto the corresponding Highway target bit and calls
// hwy::SetSupportedTargetsForTest(). CpuSimd::Autodetect is a no-op and
// leaves Highway's default supported-target set in place.
void ConstrainTargetsForCli(CpuSimd hint);

} // namespace vca::simd
