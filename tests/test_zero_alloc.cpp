// Stage 2b's actual promise -- "运行期零动态分配" -- checked two ways at
// once, per PROMPTS.md: (1) a global operator new/delete override, and
// (2) `-Wl,--wrap=malloc,--wrap=free` (see tests/CMakeLists.txt) counting
// every call made by *anything* linked into this binary, including libc
// internals. After mi_kv_cache_init() (which runs reserve() -- expected to
// allocate freely) and one warm-up decode step (glibc's math functions
// occasionally do one-time lazy setup on first use, e.g. loading precision
// tables for pow/exp; without a warm-up that cost would land in the
// counted window and look like a false-positive "leak" that isn't actually
// about mini-infer's own behavior), 64 consecutive decode steps must make
// zero calls to new/delete/malloc/free.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

#include "model_qwen2.h"
#include "test_util.h"

namespace {
long g_new_count = 0;
long g_delete_count = 0;
bool g_counting = false;
}  // namespace

void* operator new(std::size_t sz) {
  if (g_counting) g_new_count++;
  void* p = std::malloc(sz ? sz : 1);
  if (!p) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t sz) { return ::operator new(sz); }
void operator delete(void* p) noexcept {
  if (g_counting) g_delete_count++;
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

extern "C" void* __real_malloc(std::size_t);
extern "C" void __real_free(void*);

namespace {
long g_malloc_count = 0;
long g_free_count = 0;
}  // namespace

extern "C" void* __wrap_malloc(std::size_t sz) {
  if (g_counting) g_malloc_count++;
  return __real_malloc(sz);
}
extern "C" void __wrap_free(void* p) {
  if (g_counting) g_free_count++;
  __real_free(p);
}

namespace {
std::string fixture_path() { return std::string(MI_FIXTURE_DIR) + "/tiny_qwen2.gguf"; }
}  // namespace

int main() {
  std::string err;
  mi_qwen2_model* model = mi_qwen2_model_load(fixture_path().c_str(), &err);
  MI_CHECK(model != nullptr);
  if (!model) {
    std::fprintf(stderr, "failed to load fixture: %s\n", err.c_str());
    return ::mi_test::exit_code();
  }

  const int n_ctx = 128;
  mi_kv_cache* kv = mi_kv_cache_init(model->hparams, model->weights, n_ctx);  // reserve(): allocates freely
  int n_vocab = model->hparams.n_vocab;
  std::vector<float> logits((size_t)n_vocab);

  // Warm-up: one decode step outside the counted window (see file header).
  int32_t tok = 0;
  {
    mi_qwen2_forward_params fp{};
    fp.tokens = &tok;
    fp.n_tokens = 1;
    fp.n_past = 0;
    mi_qwen2_forward(model, kv, fp, logits.data());
    tok = mi_qwen2_sample_greedy(logits.data(), n_vocab);
  }

  g_new_count = g_delete_count = g_malloc_count = g_free_count = 0;
  g_counting = true;

  const int n_steps = 64;
  for (int step = 1; step <= n_steps; step++) {
    mi_qwen2_forward_params fp{};
    fp.tokens = &tok;
    fp.n_tokens = 1;
    fp.n_past = step;
    mi_qwen2_forward(model, kv, fp, logits.data());
    tok = mi_qwen2_sample_greedy(logits.data(), n_vocab);
  }

  g_counting = false;

  std::fprintf(stderr,
                "after reserve()+warmup, %d decode steps: new=%ld delete=%ld malloc=%ld free=%ld\n",
                n_steps, g_new_count, g_delete_count, g_malloc_count, g_free_count);
  MI_CHECK(g_new_count == 0);
  MI_CHECK(g_delete_count == 0);
  MI_CHECK(g_malloc_count == 0);
  MI_CHECK(g_free_count == 0);

  mi_kv_cache_free(kv);
  mi_qwen2_model_free(model);
  return ::mi_test::exit_code();
}
