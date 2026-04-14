# cmake/FetchHighway.cmake
#
# Fetch Google Highway at a pinned release tag and expose the `hwy` target
# for linking. Highway is a private dependency of the analyzer library and
# must not leak into any public VCA header.

include(FetchContent)

set(HWY_ENABLE_TESTS    OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB  OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL  OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG        1.3.0
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(highway)

message(STATUS "Highway: fetched ${highway_SOURCE_DIR}")
