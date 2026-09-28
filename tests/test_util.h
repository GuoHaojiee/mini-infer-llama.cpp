// Minimal homemade test harness (no third-party test framework, per
// SPEC.md's "no third-party libraries" constraint). Each tests/test_*.cpp
// is its own executable, registered with CTest by tests/CMakeLists.txt;
// non-zero exit means at least one MI_CHECK failed.
#pragma once

#include <cstdio>

namespace mi_test {
inline int g_failures = 0;
inline int exit_code() { return g_failures == 0 ? 0 : 1; }
}  // namespace mi_test

#define MI_CHECK(cond)                                                        \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond, __FILE__,     \
                    __LINE__);                                                \
      ::mi_test::g_failures++;                                                \
    }                                                                         \
  } while (0)
