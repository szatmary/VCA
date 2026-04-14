cmake_minimum_required(VERSION 3.13)

# Library sources (scalar + shared headers)
add_library(vcaInternal STATIC
    common/common.h
    common/EnumMapper.h
    Analyzer.h
    Analyzer.cpp
    DCTTransform.h
    DCTTransform.cpp
    DCTTransformsNative.h
    DCTTransformsNative.cpp
    EnergyCalculation.h
    EnergyCalculation.cpp
    EntropyNative.h
    EntropyNative.cpp
    EntropyCalculation.h
    EntropyCalculation.cpp
    MultiThreadQueue.h
    MultiThreadQueue.cpp
    ProcessingThread.h
    ProcessingThread.cpp
    ShotDetection.h
    ShotDetection.cpp
    simd/dct_hwy.cpp
    simd/targets.h
    simd/targets.cpp
)

target_include_directories(vcaInternal PRIVATE ${LIB_SOURCE_DIR})

target_link_libraries(vcaInternal PRIVATE Threads::Threads)

# Highway is the sole SIMD backend. Runtime dispatch is handled inside
# dct_hwy.cpp / targets.cpp via HWY_DYNAMIC_DISPATCH; there is no
# configure-time arch detection or NASM path anymore.
target_link_libraries(vcaInternal PRIVATE hwy)
target_include_directories(vcaInternal SYSTEM PRIVATE ${highway_SOURCE_DIR})
