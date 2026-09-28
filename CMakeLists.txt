cmake_minimum_required(VERSION 3.16)
project(mini_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Build types.
#
# Default to Release. Debug and Tsan are the two extra configurations
# required by SPEC.md; note that neither adds -march=native or a blanket
# -mavx2 — SIMD kernels opt in per function via
# __attribute__((target("avx2,fma"))) / target("f16c") in their own
# translation units, with __builtin_cpu_supports dispatch at call sites, so
# the scalar fallback path is compiled without any AVX flags and is a real
# fallback, not just untested dead code.
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "Tsan")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
# Tsan: -O1 (not -O0) so TSan's own overhead stays tolerable while keeping
# line-accurate reports; -g for symbolized traces.
set(CMAKE_CXX_FLAGS_TSAN "-O1 -g -fsanitize=thread -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS_TSAN "-fsanitize=thread")

add_compile_options(-Wall -Wextra)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Core library: everything under src/ except main.cpp, so tests can link
# against it without pulling in the CLI entry point.
# ---------------------------------------------------------------------------
add_library(mini_infer_core STATIC
  src/tensor.cpp
  src/context.cpp
  src/gguf.cpp
  src/graph.cpp
  src/gallocr.cpp
  src/ops_cpu.cpp
  src/threadpool.cpp
  src/quant.cpp
  src/quant_avx2.cpp
  src/model_qwen2.cpp
)
target_include_directories(mini_infer_core PUBLIC src)
target_link_libraries(mini_infer_core PUBLIC Threads::Threads)

add_executable(mini_infer src/main.cpp)
target_link_libraries(mini_infer PRIVATE mini_infer_core)

enable_testing()
add_subdirectory(tests)
add_subdirectory(tools)
