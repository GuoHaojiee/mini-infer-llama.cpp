// tools/bench_memread: stage 6's read-only bandwidth ceiling. memcpy
// (tools/bench_memcpy.cpp) both reads and writes the full buffer size every
// iteration, so it understates the bandwidth available to a workload that's
// almost pure reads -- like Q8_0 decode, which streams the weight matrix
// and writes back a handful of output floats. This tool instead does a
// sequential AVX2 load-and-accumulate pass (no writes to the source buffer
// at all, and the destination is a single reduced sum per thread) over an
// array of at least 1 GiB, once per thread on an independent, non-
// overlapping slice.
//
// The accumulate (not just touching every cache line) matters: without it,
// the compiler or hardware prefetcher could in principle let a "read but
// discard" loop skip real memory traffic; summing forces every loaded value
// to actually reach a register.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include <immintrin.h>

namespace {

__attribute__((target("avx2"))) float sum_avx2(const float* p, size_t n_floats) {
  __m256 acc = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= n_floats; i += 8) {
    acc = _mm256_add_ps(acc, _mm256_loadu_ps(p + i));
  }
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 sum4 = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(sum4);
  __m128 sums = _mm_add_ps(sum4, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  float result = _mm_cvtss_f32(sums);
  for (; i < n_floats; i++) result += p[i];
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  size_t total_bytes = argc > 1 ? (size_t)std::atoll(argv[1]) : (size_t)1024 * 1024 * 1024;
  int n_threads = argc > 2 ? std::atoi(argv[2]) : 1;
  int iters = argc > 3 ? std::atoi(argv[3]) : 20;

  size_t n_floats = total_bytes / sizeof(float);
  std::vector<float> buf(n_floats, 1.0f);
  size_t chunk_floats = n_floats / (size_t)n_threads;

  std::vector<float> partial((size_t)n_threads, 0.0f);

  auto run_once = [&]() {
    std::vector<std::thread> threads;
    threads.reserve((size_t)n_threads);
    for (int t = 0; t < n_threads; t++) {
      threads.emplace_back([&, t]() {
        size_t off = (size_t)t * chunk_floats;
        size_t len = (t == n_threads - 1) ? (n_floats - off) : chunk_floats;
        partial[(size_t)t] = sum_avx2(buf.data() + off, len);
      });
    }
    for (auto& th : threads) th.join();
  };

  run_once();  // warmup: fault in buf's pages

  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; it++) run_once();
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  double gbps = (double)total_bytes * (double)iters / secs / 1e9;

  float total = 0.0f;
  for (float v : partial) total += v;

  std::printf(
      "total_bytes=%zu n_threads=%d iters=%d secs=%.6f effective_GBps=%.4f checksum=%f\n",
      total_bytes, n_threads, iters, secs, gbps, total);
  return 0;
}
