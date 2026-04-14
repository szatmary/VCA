# Building instructions

The software is tested mostly in Linux and Windows OS. It requires some pre-requisite software to be installed before compiling. The steps to build the project in Linux and Windows are explained below.

## Prerequisites

 1. [CMake](https://cmake.org) version 3.14 or higher (required for `FetchContent_MakeAvailable`).
 2. [Git](https://git-scm.com/).
 3. C++ compiler with C++17 support.
 4. Network access at configure time (CMake fetches [Google Highway](https://github.com/google/highway) for SIMD kernels via `FetchContent`; subsequent builds use the cached copy under `build/_deps/`).

The following C++17 compilers have been known to work:

 * Visual Studio 2019 or later
 * GCC 7 or later
 * Clang 5 or later

NASM is **not** required — the previous hand-written x86 assembly layer was replaced with a portable implementation using Google Highway. The Highway dependency is fetched automatically at CMake configure time.

## Execute Build

The following commands will checkout the project source code and create a directory called 'build' where the compiler output will be placed. CMake is then used for generating build files and compiling the VCA binaries.

    $ git clone https://github.com/cd-athena/VCA.git
    $ cd VCA
    $ mkdir build
    $ cd build
    $ cmake ../
    $ cmake --build .

This will create VCA binaries in the VCA/build/source/apps/ folder.

## Docker Build

VCA can also be used via a Docker container that is build with the `Dockerfile` found in the root directory.

Simply execute the command `docker build --tag vca .` to build the container.

Note that the videos that should be analysed already need to be inside the `videos` directory, since they will be copied into the container.

Afterwards, enter the Docker container in an interactive session via `docker run --rm -it vca`.
The VCA binary and the videos that are to be analysed are found in the directory that is opened with the before mentioned command.
