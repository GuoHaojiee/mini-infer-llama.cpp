add_executable(mi_dump dump.cpp)
target_link_libraries(mi_dump PRIVATE mini_infer_core)

# Stage 6 step 0 measurement tools (see docs/stage6.md): standalone, no
# GGUF/model loading, so they isolate kernel-level throughput from every
# other per-decode-step cost.
add_executable(bench_matvec bench_matvec.cpp)
target_link_libraries(bench_matvec PRIVATE mini_infer_core)

add_executable(bench_memcpy bench_memcpy.cpp)
target_link_libraries(bench_memcpy PRIVATE Threads::Threads)

add_executable(bench_memread bench_memread.cpp)
target_link_libraries(bench_memread PRIVATE Threads::Threads)

# tools/llama_logits.cpp: the only target in this project allowed to link
# third_party/llama.cpp (SPEC.md: reference/对拍-only, never copied from).
# mini-infer has no third_party/ of its own -- it reuses the sibling
# mini-llama.cpp project's already-built checkout (per the user's decision
# at project kickoff: "复用，直接引用现有文件"), located via MI_LLAMA_CPP_ROOT
# (a cache variable so it can be overridden with
# -DMI_LLAMA_CPP_ROOT=/path/to/llama.cpp on a machine without that sibling
# checkout). Building directly against .../build/bin rather than
# find_package(llama): that build directory isn't laid out like an
# installed package (no lib/ subdir), so llama-config.cmake's assumptions
# don't apply -- same approach and same reasoning as mini-llama.cpp's own
# tools/ref_dump/CMakeLists.txt.
set(MI_LLAMA_CPP_ROOT
    "${CMAKE_SOURCE_DIR}/../mini-llama.cpp/third_party/llama.cpp"
    CACHE PATH "Path to a built llama.cpp checkout, for tools/llama_logits")
set(MI_LLAMA_CPP_BIN "${MI_LLAMA_CPP_ROOT}/build/bin")

if(EXISTS "${MI_LLAMA_CPP_ROOT}/include/llama.h" AND EXISTS "${MI_LLAMA_CPP_BIN}")
  add_executable(llama_logits llama_logits.cpp)
  target_include_directories(llama_logits PRIVATE
      ${MI_LLAMA_CPP_ROOT}/include
      ${MI_LLAMA_CPP_ROOT}/ggml/include
  )
  target_link_directories(llama_logits PRIVATE ${MI_LLAMA_CPP_BIN})
  target_link_libraries(llama_logits PRIVATE llama ggml ggml-base ggml-cpu Threads::Threads)
  set_target_properties(llama_logits PROPERTIES
      BUILD_RPATH ${MI_LLAMA_CPP_BIN}
      INSTALL_RPATH ${MI_LLAMA_CPP_BIN})
else()
  message(STATUS "tools/llama_logits: no llama.cpp checkout found at "
                  "${MI_LLAMA_CPP_ROOT} (set -DMI_LLAMA_CPP_ROOT=... to build it) -- skipping")
endif()
