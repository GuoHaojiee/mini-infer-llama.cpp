// tools/bench_memcpy: stage 6 step 0's "achievable bandwidth ceiling"
// measurement (docs/stage6.md). Each of n_threads threads memcpy's an
// independent, non-overlapping slice of a large src/dst buffer pair (no
// false sharing across slices) -- this is the number mini-infer's Q8_0
// decode GB/s should be judged against, not a theoretical peak.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  size_t total_bytes = argc > 1 ? (size_t)std::atoll(argv[1]) : (size_t)512 * 1024 * 1024;
  int n_threads = argc > 2 ? std::atoi(argv[2]) : 1;
  int iters = argc > 3 ? std::atoi(argv[3]) : 20;

  std::vector<char> src(total_bytes, (char)1);
  std::vector<char> dst(total_bytes, (char)0);
  size_t chunk = total_bytes / (size_t)n_threads;

  auto run_once = [&]() {
    std::vector<std::thread> threads;
    threads.reserve((size_t)n_threads);
    for (int t = 0; t < n_threads; t++) {
      threads.emplace_back([&, t]() {
        size_t off = (size_t)t * chunk;
        size_t len = (t == n_threads - 1) ? (total_bytes - off) : chunk;
        std::memcpy(dst.data() + off, src.data() + off, len);
      });
    }
    for (auto& th : threads) th.join();
  };

  run_once();  // warmup: fault in both buffers' pages

  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; it++) run_once();
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  double gbps = (double)total_bytes * (double)iters / secs / 1e9;

  std::printf("total_bytes=%zu n_threads=%d iters=%d secs=%.6f effective_GBps=%.4f\n", total_bytes,
              n_threads, iters, secs, gbps);
  return 0;
}
