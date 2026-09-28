# One executable per module, per SPEC.md ("每个模块一个测试可执行文件").
# Each is a placeholder until its stage lands (see comments in each file);
# CTest wiring itself is real from this commit on.
set(MI_TEST_MODULES
  tensor
  context
  gguf
  graph
  gallocr
  ops_cpu
  threadpool
  quant
  model_qwen2
)

# On some kernels/containers, ThreadSanitizer's fixed shadow-memory layout
# conflicts with a PIE binary's randomized load address ("FATAL:
# ThreadSanitizer: unexpected memory mapping"), independent of anything in
# this project. The documented workaround is to disable ASLR for the test
# process; `setarch -R` does that without needing root or a global sysctl
# change, so we apply it only for the Tsan build type.
find_program(MI_SETARCH_EXE setarch)
if(CMAKE_BUILD_TYPE STREQUAL "Tsan" AND MI_SETARCH_EXE)
  set(MI_TEST_RUNNER_PREFIX ${MI_SETARCH_EXE} ${CMAKE_SYSTEM_PROCESSOR} -R)
else()
  set(MI_TEST_RUNNER_PREFIX "")
endif()

foreach(mod ${MI_TEST_MODULES})
  add_executable(test_${mod} test_${mod}.cpp)
  target_link_libraries(test_${mod} PRIVATE mini_infer_core)
  add_test(NAME ${mod} COMMAND ${MI_TEST_RUNNER_PREFIX} $<TARGET_FILE:test_${mod}>)
endforeach()

# test_gguf and test_model_qwen2 read fixtures under tests/fixtures/; give
# them an absolute path so they don't depend on ctest's working directory.
target_compile_definitions(test_gguf PRIVATE MI_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
target_compile_definitions(test_model_qwen2 PRIVATE MI_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
target_compile_definitions(test_graph PRIVATE MI_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

# Stage 2b's "运行期零动态分配" check: a standalone target (not part of the
# per-module loop above) because it needs -Wl,--wrap=malloc,--wrap=free,
# which would break every *other* target if applied globally (they don't
# define __wrap_malloc/__wrap_free, so the linker would report those as
# undefined). ThreadSanitizer's own allocator interposition doesn't mix
# meaningfully with --wrap's, and this test is about allocation counting,
# not thread-safety, so it only runs in non-Tsan configurations.
if(NOT CMAKE_BUILD_TYPE STREQUAL "Tsan")
  add_executable(test_zero_alloc test_zero_alloc.cpp)
  target_link_libraries(test_zero_alloc PRIVATE mini_infer_core)
  target_compile_definitions(test_zero_alloc PRIVATE
    MI_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
  target_link_options(test_zero_alloc PRIVATE -Wl,--wrap=malloc -Wl,--wrap=free)
  add_test(NAME zero_alloc COMMAND $<TARGET_FILE:test_zero_alloc>)
endif()
