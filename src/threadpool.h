// mini-infer: persistent thread pool (stage 3).
//
// Threads are started once (mi_threadpool_new) and parked between graph
// executions. For each node, all threads (including the calling/main
// thread, which acts as worker 0) call the node's ops_cpu kernel with their
// ith/nth slice, then synchronize on a barrier before moving to the next
// node — see docs/architecture.md for why a barrier (not a lock-free
// pipeline) is enough here: nodes are executed strictly in topological
// order and every node's inputs must be fully written before it starts.
// See docs/stage3.md for the barrier's memory-order argument in full and
// threadpool.cpp's barrier_wait for the implementation.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "ops_cpu.h"  // mi_mulmat_mode

struct mi_cgraph;

struct mi_threadpool_params {
    int n_threads;
    bool pin_threads;      // pthread_setaffinity_np one thread per core
    mi_mulmat_mode mulmat_mode;
    int dynamic_chunk;     // rows per fetch_add claim in MI_MULMAT_DYNAMIC
};

struct mi_threadpool {
    int n_threads;
    mi_mulmat_mode mulmat_mode;
    int dynamic_chunk;

    // Barrier state: a generation counter plus an arrival count, both
    // atomic. Workers spin on the generation counter (with a short yield
    // backoff) rather than blocking on a mutex, since node execution times
    // are short enough that syscalls would dominate. Exact memory orders
    // are documented next to the implementation in threadpool.cpp.
    std::atomic<uint32_t> generation;
    std::atomic<int> arrived;

    void* impl;  // worker thread handles, per-thread state
};

mi_threadpool* mi_threadpool_new(const mi_threadpool_params& params);
void mi_threadpool_free(mi_threadpool* pool);

// Runs every node of g in order across pool->n_threads workers (worker 0 is
// the calling thread), calling mi_compute_forward(node, ith, nth) on each,
// with a barrier between nodes. mul_mat uses pool->mulmat_mode to choose
// how output rows are split; other elementwise/reduction ops always split
// rows statically across threads regardless of mulmat_mode.
void mi_threadpool_run_graph(mi_threadpool* pool, mi_cgraph* g);

// Stage 6 step 3c: barrier wait-time measurement, active only when the
// MI_BARRIER_STATS=1 env var is set (checked once, cached -- see
// threadpool.cpp) so normal runs never pay the chrono overhead. Accumulates
// wall time spent in barrier_wait's spin/yield loop (every thread that
// *isn't* the last to arrive) across the process's whole lifetime; reset
// lets a caller (e.g. a --bench-style CLI flag) isolate one measured
// section. Not a per-thread breakdown -- the aggregate is what "average
// wait per barrier call" (total_ns / count) needs.
void mi_threadpool_reset_barrier_stats();
void mi_threadpool_get_barrier_stats(uint64_t* total_wait_ns, uint64_t* wait_count);
