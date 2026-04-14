# source/lib/analyzer/simd/CMakeLists.txt
#
# The analyzer SIMD layer is Highway-based and runtime-dispatched.
# No NASM, no per-bit-depth libraries, no arch branches.
#
# The Highway-based source files (dct_hwy.cpp, targets.cpp) are compiled
# directly into the vcaInternal analyzer library. They are listed in
# source/lib/analyzer/CMakeLists.txt alongside the native analyzer sources,
# not built here as a sub-library. This file is intentionally a stub — it
# exists so CMake's add_subdirectory(simd) from the parent (if any) still
# resolves without error.
