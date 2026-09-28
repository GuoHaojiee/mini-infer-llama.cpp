// Stage 3: barrier correctness + mul_mat's three parallel modes, checked
// two ways per PROMPTS.md:
//   1. Bit-exactness: every (mode, n_threads) combination must reproduce
//      the n_threads == 1 result byte-for-byte. This holds unconditionally
//      here (see ops_cpu.cpp's mi_compute_mul_mat comment) because every
//      output row is written by exactly one thread with the same inner
//      k-loop order no matter which thread that is -- so a bit-mismatch
//      would mean the *wrong set of rows* got computed or written twice,
//      not just "different float rounding".
//   2. Running under ThreadSanitizer (see tests/CMakeLists.txt: this
//      binary is built and run in all three configs, Tsan included) with
//      small matrices and a small multi-op graph -- PROMPTS.md explicitly
//      asks for this ("不要在整模型上跑 TSan") since a full model forward
//      would make Tsan's instrumentation overhead impractical.
#include <cstring>
#include <random>
#include <vector>

#include "context.h"
#include "graph.h"
#include "ops_cpu.h"
#include "test_util.h"
#include "threadpool.h"

namespace {

mi_context* new_eager_ctx(size_t tensor_capacity, size_t data_capacity) {
  mi_context_params p{};
  p.tensor_capacity = tensor_capacity;
  p.data_capacity = data_capacity;
  p.no_alloc = false;
  return mi_context_new(p);
}

std::vector<float> random_vec(size_t n, std::mt19937* rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n);
  for (float& x : v) x = dist(*rng);
  return v;
}

// Runs dst = mul_mat(a, b) through a fresh threadpool configured with
// (mode, n_threads, chunk) and returns dst's data as a vector. Every call
// gets its own context/pool so different configurations can't leak state
// into each other (mi_mulmat_set_mode is process-global -- see
// ops_cpu.cpp -- so a stale pool from a previous case must never still be
// alive when this one runs).
std::vector<float> run_mul_mat(mi_mulmat_mode mode, int n_threads, int chunk, int64_t k, int64_t n,
                                int64_t m, const std::vector<float>& a_data,
                                const std::vector<float>& b_data) {
  mi_context* ctx = new_eager_ctx(16, (size_t)(k * n + k * m + n * m) * sizeof(float) + 4096);
  mi_tensor* a = mi_new_tensor_2d(ctx, MI_TYPE_F32, k, n);
  std::memcpy(a->data, a_data.data(), a_data.size() * sizeof(float));
  mi_tensor* b = mi_new_tensor_2d(ctx, MI_TYPE_F32, k, m);
  std::memcpy(b->data, b_data.data(), b_data.size() * sizeof(float));
  mi_tensor* dst = mi_graph_mul_mat(ctx, a, b);

  mi_cgraph* g = mi_graph_new(ctx, 16);
  mi_graph_build_forward(g, dst);

  mi_threadpool_params tp{n_threads, /*pin_threads=*/false, mode, chunk};
  mi_threadpool* pool = mi_threadpool_new(tp);
  mi_graph_compute(g, pool);
  mi_threadpool_free(pool);

  std::vector<float> out((float*)dst->data, (float*)dst->data + n * m);
  mi_context_free(ctx);
  return out;
}

// A 5-node chain (rms_norm -> mul (weight) -> mul_mat -> add (bias) ->
// silu) exercising every node type's row-splitting, with a barrier between
// each: if the barrier failed to make one node's writes visible before the
// next node reads them, this would show up as a bit mismatch against the
// n_threads == 1 reference (most likely NaN/garbage from an unwritten
// row, or a race caught directly by TSan).
std::vector<float> run_chain(mi_mulmat_mode mode, int n_threads, int chunk, int64_t d, int64_t r,
                              int64_t d_out, const std::vector<float>& x_data,
                              const std::vector<float>& norm_w_data,
                              const std::vector<float>& proj_w_data,
                              const std::vector<float>& bias_data) {
  mi_context* ctx = new_eager_ctx(
      32, (size_t)(d * r + d + d * d_out + d_out + d * r + d_out * r + d_out * r) * sizeof(float) +
              8192);
  mi_tensor* x = mi_new_tensor_2d(ctx, MI_TYPE_F32, d, r);
  std::memcpy(x->data, x_data.data(), x_data.size() * sizeof(float));
  mi_tensor* norm_w = mi_new_tensor_1d(ctx, MI_TYPE_F32, d);
  std::memcpy(norm_w->data, norm_w_data.data(), norm_w_data.size() * sizeof(float));
  mi_tensor* proj_w = mi_new_tensor_2d(ctx, MI_TYPE_F32, d, d_out);
  std::memcpy(proj_w->data, proj_w_data.data(), proj_w_data.size() * sizeof(float));
  mi_tensor* bias = mi_new_tensor_1d(ctx, MI_TYPE_F32, d_out);
  std::memcpy(bias->data, bias_data.data(), bias_data.size() * sizeof(float));

  mi_tensor* normed = mi_graph_rms_norm(ctx, x, 1e-6f);
  mi_tensor* scaled = mi_graph_mul(ctx, normed, norm_w);
  mi_tensor* proj = mi_graph_mul_mat(ctx, proj_w, scaled);
  mi_tensor* biased = mi_graph_add(ctx, proj, bias);
  mi_tensor* out = mi_graph_silu(ctx, biased);

  mi_cgraph* g = mi_graph_new(ctx, 16);
  mi_graph_build_forward(g, out);

  mi_threadpool_params tp{n_threads, /*pin_threads=*/false, mode, chunk};
  mi_threadpool* pool = mi_threadpool_new(tp);
  mi_graph_compute(g, pool);
  mi_threadpool_free(pool);

  std::vector<float> result((float*)out->data, (float*)out->data + d_out * r);
  mi_context_free(ctx);
  return result;
}

}  // namespace

static void test_enum_distinct() {
  MI_CHECK(MI_MULMAT_SINGLE != MI_MULMAT_STATIC);
  MI_CHECK(MI_MULMAT_STATIC != MI_MULMAT_DYNAMIC);
}

// PROMPTS.md: "三种模式、线程数 1/2/4/8 下,输出与单线程逐位一致".
// k/n/m deliberately not multiples of any tested thread count, so
// mi_split_range's remainder-to-the-front handling and DYNAMIC's chunk
// boundaries both get exercised against an uneven split.
static void test_mul_mat_bit_exact_across_modes_and_threads() {
  const int64_t k = 48, n = 37, m = 5;
  std::mt19937 rng(12345);
  std::vector<float> a_data = random_vec((size_t)(k * n), &rng);
  std::vector<float> b_data = random_vec((size_t)(k * m), &rng);

  std::vector<float> reference = run_mul_mat(MI_MULMAT_STATIC, 1, 8, k, n, m, a_data, b_data);
  MI_CHECK(reference.size() == (size_t)(n * m));

  const int thread_counts[] = {1, 2, 4, 8};
  for (int nt : thread_counts) {
    {
      auto out = run_mul_mat(MI_MULMAT_SINGLE, nt, 8, k, n, m, a_data, b_data);
      MI_CHECK(std::memcmp(out.data(), reference.data(), out.size() * sizeof(float)) == 0);
    }
    {
      auto out = run_mul_mat(MI_MULMAT_STATIC, nt, 8, k, n, m, a_data, b_data);
      MI_CHECK(std::memcmp(out.data(), reference.data(), out.size() * sizeof(float)) == 0);
    }
    for (int chunk : {1, 3, 16}) {
      auto out = run_mul_mat(MI_MULMAT_DYNAMIC, nt, chunk, k, n, m, a_data, b_data);
      MI_CHECK(std::memcmp(out.data(), reference.data(), out.size() * sizeof(float)) == 0);
    }
  }
}

// Same bit-exactness property, but across a 5-node chain with a barrier
// between every node -- this is what actually exercises "does the barrier
// make one node's output visible before the next node starts", since a
// single-node test can't detect a broken barrier (there's nothing after it
// to read stale data).
static void test_chain_bit_exact_across_modes_and_threads() {
  const int64_t d = 16, r = 9, d_out = 11;
  std::mt19937 rng(999);
  std::vector<float> x_data = random_vec((size_t)(d * r), &rng);
  std::vector<float> norm_w_data = random_vec((size_t)d, &rng);
  std::vector<float> proj_w_data = random_vec((size_t)(d * d_out), &rng);
  std::vector<float> bias_data = random_vec((size_t)d_out, &rng);

  std::vector<float> reference =
      run_chain(MI_MULMAT_STATIC, 1, 4, d, r, d_out, x_data, norm_w_data, proj_w_data, bias_data);
  MI_CHECK(reference.size() == (size_t)(d_out * r));

  const int thread_counts[] = {1, 2, 4, 8};
  const mi_mulmat_mode modes[] = {MI_MULMAT_SINGLE, MI_MULMAT_STATIC, MI_MULMAT_DYNAMIC};
  for (int nt : thread_counts) {
    for (mi_mulmat_mode mode : modes) {
      auto out = run_chain(mode, nt, 4, d, r, d_out, x_data, norm_w_data, proj_w_data, bias_data);
      MI_CHECK(std::memcmp(out.data(), reference.data(), out.size() * sizeof(float)) == 0);
    }
  }
}

// Runs the same tiny chain repeatedly at max thread count -- cheap even
// under TSan (per PROMPTS.md's "小矩阵...很小的图"), but repeating raises
// the chance any latent race in the barrier/dynamic-counter shows up as
// either a wrong answer (checked here) or a TSan report (checked by the
// Tsan CTest configuration actually running this binary).
static void test_repeated_runs_stay_consistent() {
  const int64_t d = 8, r = 5, d_out = 6;
  std::mt19937 rng(7);
  std::vector<float> x_data = random_vec((size_t)(d * r), &rng);
  std::vector<float> norm_w_data = random_vec((size_t)d, &rng);
  std::vector<float> proj_w_data = random_vec((size_t)(d * d_out), &rng);
  std::vector<float> bias_data = random_vec((size_t)d_out, &rng);

  std::vector<float> reference =
      run_chain(MI_MULMAT_STATIC, 1, 2, d, r, d_out, x_data, norm_w_data, proj_w_data, bias_data);

  for (int iter = 0; iter < 20; iter++) {
    auto out = run_chain(MI_MULMAT_DYNAMIC, 8, 2, d, r, d_out, x_data, norm_w_data, proj_w_data,
                          bias_data);
    MI_CHECK(std::memcmp(out.data(), reference.data(), out.size() * sizeof(float)) == 0);
  }
}

int main() {
  test_enum_distinct();
  test_mul_mat_bit_exact_across_modes_and_threads();
  test_chain_bit_exact_across_modes_and_threads();
  test_repeated_runs_stay_consistent();
  return ::mi_test::exit_code();
}
