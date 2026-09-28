// mini-infer: graph allocator (stage 2b).
//
// Plans buffer offsets for every non-persistent tensor in a graph against
// one preallocated arena, reusing space via a free-list and (optionally)
// in-place rewrites of elementwise ops. Two-phase use, matching SPEC.md's
// "reserve 与规划分离":
//   1. mi_gallocr_reserve(): run once at startup against the worst-case
//      graphs (max prefill batch, and decode at n_past = n_ctx - 1); sizes
//      and allocates the arena to the max of the two.
//   2. mi_gallocr_alloc_graph(): run once per forward step against the
//      freshly rebuilt graph; only computes offsets into the existing
//      arena, no malloc/free.
// See docs/architecture.md and docs/stage2b.md for the rehearsal algorithm
// and the in-place/view rules.
#pragma once

#include <cstddef>
#include <cstdint>

#include "graph.h"
#include "tensor.h"

struct mi_gallocr_stats {
    size_t peak_bytes;       // high-water mark of arena bytes in use
    size_t baseline_bytes;   // sum of each intermediate tensor's own size,
                              // as if independently allocated (no reuse)
    size_t n_inplace_reuse;  // count of elementwise ops rewritten in place
    size_t n_bestfit_reuse;  // count of allocations satisfied by reusing a
                              // freed block rather than growing the arena
};

struct mi_gallocr {
    unsigned char* arena;
    size_t arena_size;

    // Debug mode: when the environment variable GALLOCR_POISON=1 is set at
    // construction time, freeing a block immediately fills it with NaN
    // (float bit pattern 0x7fc00000) so any read-after-free shows up as
    // NaN in the output instead of silently reading stale data.
    bool poison_enabled;

    mi_gallocr_stats stats;

    // Free-list / block-tracking state is intentionally not exposed here;
    // it is an implementation detail of gallocr.cpp (best-fit list of
    // {offset, size} spans, merged with neighbors on free).
    void* impl;
};

mi_gallocr* mi_gallocr_new(void);
void mi_gallocr_free(mi_gallocr* galloc);

// Rehearsal pass: walks g->nodes in order, and for each node's tensor:
//   - counts references (fills t->n_children, crediting view chains to
//     their view_src per docs/architecture.md);
//   - decides, without touching any buffer, the peak arena size needed.
// Does not allocate the arena. Exposed separately from reserve() so tests
// can check the rehearsal's ref-counting and sizing decisions in isolation.
void mi_gallocr_rehearse(mi_gallocr* galloc, mi_cgraph* g);

// Runs mi_gallocr_rehearse() against every graph in `graphs` (used to cover
// both the max-prefill and max-decode worst cases), takes the max required
// size, and allocates the arena exactly once. Must be called before the
// first mi_gallocr_alloc_graph().
void mi_gallocr_reserve(mi_gallocr* galloc, mi_cgraph** graphs, int n_graphs);

// Per-step planning: assigns a data pointer (into the existing arena) to
// every non-persistent tensor in g->nodes, applying in-place reuse and
// best-fit placement. Persistent tensors (mi_tensor_is_persistent) are
// skipped — their data must already be set by the caller. Never grows or
// reallocates the arena; asserts if a graph built for reserve() would
// exceed it.
void mi_gallocr_alloc_graph(mi_gallocr* galloc, mi_cgraph* g);

mi_gallocr_stats mi_gallocr_get_stats(const mi_gallocr* galloc);

// Stage 6 wrap-up: a pure graph-structural analysis, independent of any
// mi_gallocr instance or actual allocation -- the theoretical lower bound
// no allocator (however clever) could beat: at every position i in g's
// topological order, sum mi_nbytes() over every non-persistent, non-view
// tensor whose live range [birth, death] contains i (birth = the position
// where it's computed; death = the last position that reads it, resolved
// through view chains, or its own birth if nothing downstream reads it);
// the return value is the max of that sum over all i. This is strictly a
// liveness bound (what a perfect, unconstrained packer could achieve if
// every live tensor needs disjoint bytes) -- it does NOT additionally
// credit in-place elimination (an op physically overwriting one of its own
// inputs), which is a separate, real optimization gallocr also does and
// can push its actual peak_bytes *below* this number for graphs with many
// in-place-eligible ops. O(n_nodes^2); only meant for offline diagnostics
// (e.g. --gallocr-stats), never called per decode step.
size_t mi_gallocr_theoretical_min_bytes(const mi_cgraph* g);

// Debug/test hooks, read once by mi_gallocr_new() from the environment
// (same idiom as GALLOCR_POISON): GALLOCR_NO_INPLACE=1 disables the
// in-place rewrite path (every allocation goes through best-fit/grow, as
// if no op ever supported in-place), GALLOCR_NO_BESTFIT=1 disables free-
// list reuse (every allocation grows the arena's watermark, never reuses a
// freed block). Used to produce the 4 (in-place x best-fit) combinations
// PROMPTS.md's stage 2b test matrix asks for, all of which must still
// match stage 2a bit-for-bit -- these flags only change *which* bytes a
// tensor lands on, never the arithmetic.
