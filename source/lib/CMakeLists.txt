cmake_minimum_required(VERSION 3.13)

# Version header
configure_file("${PROJECT_SOURCE_DIR}/source/lib/vca_config.h.in"
    "${PROJECT_BINARY_DIR}/vca_config.h")

find_package(Git QUIET) # present in 2.8.8

# Defaults
set(VCA_VERSION "unknown")
set(VCA_BASE_TAG "0.0.0")
set(VCA_TAG_DISTANCE "0")
set(VCA_REVISION_ID "")
set(VCA_DIRTY "")

set(LIB_SOURCE_DIR ${CMAKE_SOURCE_DIR}/source/lib)

if(GIT_EXECUTABLE AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # nearest tag reachable from HEAD
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE VCA_BASE_TAG
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # distance from tag to HEAD
    if(NOT VCA_BASE_TAG STREQUAL "")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-list ${VCA_BASE_TAG}..HEAD --count --first-parent
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            OUTPUT_VARIABLE VCA_TAG_DISTANCE
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()

    # revision id (short)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE VCA_REVISION_ID
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # dirty flag (optional)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} diff --quiet
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        RESULT_VARIABLE GIT_DIFF_EXIT
        ERROR_QUIET
    )
    if(GIT_DIFF_EXIT EQUAL 1)
        set(VCA_DIRTY "-dirty")
    endif()
endif()

# Compose version string
if(VCA_BASE_TAG STREQUAL "" OR VCA_BASE_TAG STREQUAL "unknown")
    set(VCA_VERSION "unknown")
else()
    if(VCA_TAG_DISTANCE STREQUAL "0")
        # exact tag
        set(VCA_VERSION "${VCA_BASE_TAG}${VCA_DIRTY}")
    else()
        # tag+N-g<sha>[-dirty]
        set(VCA_VERSION "${VCA_BASE_TAG}+${VCA_TAG_DISTANCE}-g${VCA_REVISION_ID}${VCA_DIRTY}")
    endif()
endif()

message(STATUS "VCA version: ${VCA_VERSION}")
list(APPEND VFLAG "-DVCA_VERSION=${VCA_VERSION}")
set_source_files_properties(vcaLib.cpp PROPERTIES COMPILE_FLAGS ${VFLAG})

find_package(Threads REQUIRED)

# SIMD is handled entirely by Google Highway at runtime. No NASM, no
# arch detection, no configure-time dispatch. See cmake/FetchHighway.cmake.

# Libraries in this directory
add_library(vcaLib "")
target_sources(vcaLib
    PRIVATE
        vcaLib.cpp
    PUBLIC
        vcaLib.h
)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(GCC 1)
endif()

# Add analyzer (Highway SIMD is compiled directly into vcaInternal)
add_subdirectory(analyzer)

# Tests
if (ENABLE_TEST)
    message(STATUS "Enable building of unit tests")
    add_subdirectory(test)
else()
    message(STATUS "Not building unit tests")
endif()

# Includes & compile definitions
target_include_directories(vcaInternal PRIVATE ${LIB_SOURCE_DIR})
target_include_directories(vcaLib PRIVATE ${LIB_SOURCE_DIR})

target_link_libraries(vcaLib
    vcaInternal
)

set_target_properties(vcaLib PROPERTIES OUTPUT_NAME vcaLib)
set_target_properties(vcaLib PROPERTIES PUBLIC_HEADER "vcaLib.h")

install(
    TARGETS vcaLib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    COMPONENT libraries
    PUBLIC_HEADER DESTINATION include
)
