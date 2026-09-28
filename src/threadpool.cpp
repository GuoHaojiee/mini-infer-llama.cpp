// mini-infer: persistent thread pool implementation (stage 3).
//
// Two independent synchronization mechanisms:
//
// 1. Outer "new graph published" wakeup (impl->run_id): mi_threadpool_run_graph
//    (called by the main thread, which acts as worker 0) stores the new
//    mi_cgraph* and then does `run_id.fetch_add(1, release)`. Idle workers
//    spin on `run_id.load(acquire)` waiting for it to change; once it does,
//    the acquire/release pair on run_id guarantees the just-stored graph
//    pointer (and, transitively, everything the main thread did before
//    that store -- e.g. resetting mul_mat's dynamic counter for node 0, see
//    below) is visible before the worker reads it.
//
// 2. Inner per-node barrier (pool->generation / pool->arrived): a classic
//    sense-reversing barrier. Every one of pool->n_threads participants
//    calls `arrived.fetch_add(1, acq_rel)` after finishing its slice of the
//    current node. fetch_add on the same atomic totally orders all
//    participants' RMWs, and acq_rel means each one both acquires the
//    previous participant's release and releases for the next -- so the
//    thread whose fetch_add makes the count reach n_threads has, by then,
//    observed every other thread's writes to the finished node's dst->data
//    (each such write precedes its author's fetch_add in program order,
//    and visibility propagates through the acq_rel chain). That thread is
//    therefore safe to do bookkeeping for the *next* node -- specifically,
//    if the next node is a MUL_MAT, resetting the dynamic-claim counter --
//    strictly before `generation.fetch_add(1, release)`. Every other
//    thread is spin-waiting on `generation.load(acquire)`; the release/
//    acquire pair makes both the finished node's writes and the dynamic-
//    counter reset visible before any thread starts the next node. This is
//    what makes a node's KV-cache cpy, or one layer's output, safely
//    readable by the next node without any per-tensor synchronization.
//
// Both mechanisms are self-implemented (no std::barrier) per SPEC.md, and
// use only <atomic>/<thread>/pthread (for CPU affinity), no third-party
// library.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "threadpool.h"

#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "graph.h"
#include "ops_cpu.h"

namespace {

bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v && v[0] == '1' && v[1] == '\0';
}

// Stage 6 step 3c: MI_BARRIER_SPIN_ONLY=1 disables the yield fallback below
// (never falls back to sched_yield(), spins on cpu_relax() indefinitely
// instead) -- a runtime toggle so both strategies can be measured on the
// same binary, same machine, same run methodology as every other env-var
// toggle in this project (MI_QUANT_FORCE_SCALAR, GALLOCR_NO_INPLACE, ...).
// Read once, cached: safe because nothing mutates it after the threadpool's
// worker threads are spawned (mi_threadpool_new's happens-before argument,
// same as mi_mulmat_set_mode's).
bool spin_only() {
  static const bool v = env_flag("MI_BARRIER_SPIN_ONLY");
  return v;
}

// Stage 6 step 3c: MI_BARRIER_STATS=1 turns on the chrono-based wait-time
// accumulation in barrier_wait below. Off by default so normal runs (and
// every benchmark that isn't specifically measuring barrier wait time)
// never pay a steady_clock::now() call per barrier -- see docs/stage6.md's
// step-0/1 numbers, all measured with this off.
bool stats_enabled() {
  static const bool v = env_flag("MI_BARRIER_STATS");
  return v;
}

std::atomic<uint64_t> g_barrier_wait_ns{0};
std::atomic<uint64_t> g_barrier_wait_count{0};

// Busy-spin for a while (cheap; the other threads are usually only
// nanoseconds behind on a node that's microseconds of work), then fall
// back to sched_yield() -- "自旋后短暂 yield" per SPEC.md, rather than a
// mutex/condvar, which would cost a syscall on every node of every step.
// Stage 6 step 3c: spin_only() can disable the yield fallback -- see above.
constexpr int kSpinLimit = 4000;

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}

void spin_backoff(int* spins) {
  if (spin_only() || *spins < kSpinLimit) {
    ++*spins;
    cpu_relax();
  } else {
    std::this_thread::yield();
  }
}

// See file header for the full memory-order argument. `next_node` is the
// node about to be entered by every participant once this call returns;
// only the winning (last-arriving) thread inspects it, strictly before the
// release store, so the reset is guaranteed visible in time.
void barrier_wait(mi_threadpool* pool, const mi_tensor* next_node) {
  uint32_t my_gen = pool->generation.load(std::memory_order_relaxed);
  int prev_arrived = pool->arrived.fetch_add(1, std::memory_order_acq_rel);
  if (prev_arrived + 1 == pool->n_threads) {
    if (next_node && next_node->op == MI_OP_MUL_MAT) {
      mi_mulmat_reset_dynamic_counter();
    }
    pool->arrived.store(0, std::memory_order_relaxed);
    pool->generation.fetch_add(1, std::memory_order_release);
  } else if (stats_enabled()) {
    auto t0 = std::chrono::steady_clock::now();
    int spins = 0;
    while (pool->generation.load(std::memory_order_acquire) == my_gen) {
      spin_backoff(&spins);
    }
    auto t1 = std::chrono::steady_clock::now();
    uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    g_barrier_wait_ns.fetch_add(ns, std::memory_order_relaxed);
    g_barrier_wait_count.fetch_add(1, std::memory_order_relaxed);
  } else {
    int spins = 0;
    while (pool->generation.load(std::memory_order_acquire) == my_gen) {
      spin_backoff(&spins);
    }
  }
}

struct mi_threadpool_impl {
  std::vector<std::thread> workers;  // n_threads - 1; main thread is worker 0
  std::atomic<uint32_t> run_id{0};
  std::atomic<mi_cgraph*> graph{nullptr};
  std::atomic<bool> stop{false};
  bool pin_threads;
};

void pin_this_thread(int ith) {
  unsigned hc = std::thread::hardware_concurrency();
  if (hc == 0) hc = 1;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(ith % (int)hc, &set);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (rc != 0) {
    std::fprintf(stderr, "mini-infer: warning: pthread_setaffinity_np failed for thread %d (rc=%d)\n",
                 ith, rc);
  }
}

// Shared by the main thread and every worker: run every node of the
// currently-published graph, in order, with a barrier between nodes.
//
// Stage 6: iterates g->exec_nodes, not g->nodes -- exec_nodes already
// excludes every RESHAPE/VIEW/PERMUTE/TRANSPOSE node (see graph.h/.cpp),
// which move no data and whose mi_compute_forward dispatch was already a
// no-op, so skipping them here means skipping their barrier_wait too (the
// actual cost: docs/stage6.md's step-0 measurement found 40.9% of a decode
// step's barrier calls were on exactly this class of node). g->nodes
// itself is untouched, in the exact order gallocr's allocation walk (a
// separate, single-threaded pass that already ran before this) needs.
void run_current_graph(mi_threadpool* pool, mi_threadpool_impl* impl, int ith) {
  mi_cgraph* g = impl->graph.load(std::memory_order_relaxed);
  int n_nodes = g->n_exec_nodes;
  for (int i = 0; i < n_nodes; i++) {
    mi_tensor* node = g->exec_nodes[i];
    mi_compute_forward(node, ith, pool->n_threads);
    mi_tensor* next_node = (i + 1 < n_nodes) ? g->exec_nodes[i + 1] : nullptr;
    barrier_wait(pool, next_node);
  }
}

void worker_main(mi_threadpool* pool, mi_threadpool_impl* impl, int ith) {
  if (impl->pin_threads) pin_this_thread(ith);
  uint32_t last_run = 0;
  int spins = 0;
  for (;;) {
    uint32_t run;
    while ((run = impl->run_id.load(std::memory_order_acquire)) == last_run) {
      spin_backoff(&spins);
    }
    last_run = run;
    spins = 0;
    if (impl->stop.load(std::memory_order_relaxed)) return;
    run_current_graph(pool, impl, ith);
  }
}

}  // namespace

mi_threadpool* mi_threadpool_new(const mi_threadpool_params& params) {
  auto* pool = new mi_threadpool();
  pool->n_threads = params.n_threads > 0 ? params.n_threads : 1;
  pool->mulmat_mode = params.mulmat_mode;
  pool->dynamic_chunk = params.dynamic_chunk;
  pool->generation.store(0, std::memory_order_relaxed);
  pool->arrived.store(0, std::memory_order_relaxed);

  // Set once, before any worker thread exists: mi_compute_mul_mat reads
  // this global config on every call, but nothing ever mutates it again
  // for this pool's lifetime, so the std::thread constructor's own
  // happens-before edge (parent's writes before thread creation are
  // visible to the new thread) is all the synchronization this needs.
  mi_mulmat_set_mode(params.mulmat_mode, params.dynamic_chunk);

  auto* impl = new mi_threadpool_impl();
  impl->pin_threads = params.pin_threads;
  pool->impl = impl;

  if (params.pin_threads) pin_this_thread(0);  // main thread == worker 0

  for (int ith = 1; ith < pool->n_threads; ith++) {
    impl->workers.emplace_back(worker_main, pool, impl, ith);
  }
  return pool;
}

void mi_threadpool_free(mi_threadpool* pool) {
  if (!pool) return;
  auto* impl = (mi_threadpool_impl*)pool->impl;
  impl->stop.store(true, std::memory_order_relaxed);
  impl->run_id.fetch_add(1, std::memory_order_release);
  for (auto& t : impl->workers) t.join();
  delete impl;
  delete pool;
}

void mi_threadpool_run_graph(mi_threadpool* pool, mi_cgraph* g) {
  auto* impl = (mi_threadpool_impl*)pool->impl;
  // No previous-node barrier to piggyback the counter reset on for node 0
  // of a fresh run -- done here instead, strictly before the run_id
  // release store below, so it's visible to workers by the same argument
  // as the per-node barrier's reset (see file header). Checks exec_nodes[0]
  // (stage 6), not nodes[0]: exec_nodes is what run_current_graph actually
  // executes first.
  if (g->n_exec_nodes > 0 && g->exec_nodes[0]->op == MI_OP_MUL_MAT) {
    mi_mulmat_reset_dynamic_counter();
  }
  impl->graph.store(g, std::memory_order_relaxed);
  impl->run_id.fetch_add(1, std::memory_order_release);
  run_current_graph(pool, impl, 0);
}

void mi_threadpool_reset_barrier_stats() {
  g_barrier_wait_ns.store(0, std::memory_order_relaxed);
  g_barrier_wait_count.store(0, std::memory_order_relaxed);
}

void mi_threadpool_get_barrier_stats(uint64_t* total_wait_ns, uint64_t* wait_count) {
  *total_wait_ns = g_barrier_wait_ns.load(std::memory_order_relaxed);
  *wait_count = g_barrier_wait_count.load(std::memory_order_relaxed);
}
