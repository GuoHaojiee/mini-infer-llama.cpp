// tools/bench_matvec: stage 6 raw-kernel microbenchmark. Isolates
// mi_vec_dot_q8_0_wact's own throughput from every other per-decode-step
// cost (graph build, gallocr, threadpool/barrier) by repeatedly running the
// exact matrix-vector shape decode's Q8_0 mul_mat actually hits -- default
// k=4864, n=896 is blk.N.ffn_down.weight's shape in Qwen2.5-0.5B (see
// docs/stage6.md), a decode-time matrix-vector product (m=1).
//
// Two modes, both reported:
//   - "L3-resident" (default n_matrices=1): the same ~4.6 MB weight matrix
//     (fits inside this machine's 16 MB L3) is read every iteration, so
//     after the first pass it's served from L3, not DRAM -- this measures
//     the kernel's compute throughput with cache effects mostly factored
//     out, but is NOT a DRAM-bandwidth measurement (real decode reads a
//     >500 MB model that never fits in any cache).
//   - "DRAM" (n_matrices > 1, chosen so n_matrices * matrix_bytes >= a
//     target total, default 256 MiB): iterations round-robin through many
//     independent matrices, each only touched once every n_matrices
//     iterations, well past cache retention -- this is the number
//     comparable to a real streaming decode's GB/s.
//
// MI_QUANT_FORCE_SCALAR=1 selects the scalar dispatch path, same env var as
// the rest of the project (see quant.cpp).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "quant.h"

namespace {

// One n x k Q8_0 weight matrix plus a matching quantized activation vector,
// built from random data with a fixed seed so every matrix (L3-resident or
// one of the DRAM round-robin set) is generated the same way.
struct Matrix {
  std::vector<block_q8_0> weight;  // n rows x nb blocks
  int64_t nb;
};

Matrix make_matrix(int64_t k, int64_t n, int64_t nb, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Matrix m;
  m.nb = nb;
  m.weight.resize((size_t)n * nb);
  std::vector<float> row_f32((size_t)k);
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < k; j++) row_f32[j] = dist(rng);
    mi_quantize_row_q8_0(row_f32.data(), m.weight.data() + i * nb, k);
  }
  return m;
}

// Runs `iters` iterations of "dot every row of one matrix against `act`",
// cycling through `matrices` round-robin (matrices.size()==1 for the
// L3-resident mode), and reports elapsed seconds + a sink value.
double run(const std::vector<Matrix>& matrices, const block_q8_0_act* act, int64_t k, int64_t n,
            int iters, float* out_sink) {
  std::vector<float> out((size_t)n);
  size_t n_matrices = matrices.size();

  // Untimed warmup: touch every matrix once so first-touch page faults
  // don't land inside the timed loop.
  for (size_t mi = 0; mi < n_matrices; mi++) {
    const int64_t nb = matrices[mi].nb;
    for (int64_t i = 0; i < n; i++) {
      out[i] = mi_vec_dot_q8_0_wact(matrices[mi].weight.data() + (size_t)i * nb, act, k);
    }
  }

  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; it++) {
    const Matrix& m = matrices[(size_t)it % n_matrices];
    for (int64_t i = 0; i < n; i++) {
      out[i] = mi_vec_dot_q8_0_wact(m.weight.data() + (size_t)i * m.nb, act, k);
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  *out_sink = out[0];
  return std::chrono::duration<double>(t1 - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  int64_t k = argc > 1 ? std::atoll(argv[1]) : 4864;
  int64_t n = argc > 2 ? std::atoll(argv[2]) : 896;
  int iters = argc > 3 ? std::atoi(argv[3]) : 1000;
  // DRAM mode's target total footprint across all round-robin matrices.
  size_t dram_target_bytes = argc > 4 ? (size_t)std::atoll(argv[4]) : (size_t)256 * 1024 * 1024;

  if (k % MI_QK8_0 != 0) {
    std::fprintf(stderr, "k must be a multiple of %d\n", MI_QK8_0);
    return 1;
  }
  int64_t nb = k / MI_QK8_0;
  size_t matrix_bytes = (size_t)n * (size_t)nb * sizeof(block_q8_0);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> act_f32((size_t)k);
  for (int64_t j = 0; j < k; j++) act_f32[j] = dist(rng);
  std::vector<block_q8_0_act> act(nb);
  mi_quantize_row_q8_0_act(act_f32.data(), act.data(), k);

  // Mode 1: L3-resident -- one matrix, reused every iteration.
  {
    std::mt19937 mrng(1);
    std::vector<Matrix> matrices;
    matrices.push_back(make_matrix(k, n, nb, mrng));
    float sink = 0.0f;
    double secs = run(matrices, act.data(), k, n, iters, &sink);
    double total_bytes = (double)matrix_bytes * (double)iters;
    double gbps = total_bytes / secs / 1e9;
    double gflops = (double)n * (double)k * 2.0 * (double)iters / secs / 1e9;
    std::printf(
        "mode=L3-resident k=%lld n=%lld iters=%d n_matrices=1 matrix_bytes=%zu secs=%.6f "
        "effective_GBps=%.4f effective_GFLOPs=%.4f out0=%f\n",
        (long long)k, (long long)n, iters, matrix_bytes, secs, gbps, gflops, sink);
  }

  // Mode 2: DRAM -- enough independent matrices that round-robining through
  // them exceeds this machine's L3 (16 MiB), well past dram_target_bytes.
  {
    size_t n_matrices = (dram_target_bytes + matrix_bytes - 1) / matrix_bytes;
    if (n_matrices < 1) n_matrices = 1;
    std::mt19937 mrng(2);
    std::vector<Matrix> matrices;
    matrices.reserve(n_matrices);
    for (size_t i = 0; i < n_matrices; i++) matrices.push_back(make_matrix(k, n, nb, mrng));
    float sink = 0.0f;
    double secs = run(matrices, act.data(), k, n, iters, &sink);
    double total_bytes = (double)matrix_bytes * (double)iters;
    double gbps = total_bytes / secs / 1e9;
    double gflops = (double)n * (double)k * 2.0 * (double)iters / secs / 1e9;
    std::printf(
        "mode=DRAM k=%lld n=%lld iters=%d n_matrices=%zu matrix_bytes=%zu total_footprint_bytes=%zu "
        "secs=%.6f effective_GBps=%.4f effective_GFLOPs=%.4f out0=%f\n",
        (long long)k, (long long)n, iters, n_matrices, matrix_bytes, n_matrices * matrix_bytes, secs,
        gbps, gflops, sink);
  }

  return 0;
}
