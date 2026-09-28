// mini-infer: computation graph (stage 2+).
//
// Node tensors are built with the mi_graph_* builder functions below: each
// call allocates a result tensor from a mi_context (metadata only, data
// stays null until gallocr or execute() fills it in) and records op/src so
// the graph can later be topologically sorted and executed. The actual
// math for each op lives in ops_cpu.* and is shared verbatim with stage 1's
// direct-call forward pass — see docs/architecture.md, section "kernel
// reuse across stage 1 and stage 2".
#pragma once

#include <cstddef>

#include "context.h"
#include "tensor.h"

struct mi_threadpool;  // forward decl, stage 3

struct mi_cgraph {
    mi_tensor** nodes;  // topologically sorted compute nodes (op != NONE)
    int n_nodes;

    mi_tensor** leafs;  // referenced leaves (weights/inputs/KV), for reserve/debug
    int n_leafs;

    // Stage 6: strict subsequence of `nodes` (same relative topological
    // order) excluding every node with mi_op_is_view(op) true
    // (RESHAPE/VIEW/PERMUTE/TRANSPOSE) -- those nodes move no data and
    // mi_compute_forward's dispatch for them is already a no-op, but
    // *executing* them still cost a full threadpool barrier_wait per node
    // (docs/stage6.md's step-0 measurement: 40.9% of a decode step's
    // barrier calls were on exactly this class of node). Populated by
    // visit() at graph-*build* time (not by gallocr), so it's correct
    // whether or not this graph ever goes through gallocr at all (several
    // tests build+execute a graph via eager tensor allocation, no gallocr
    // involved). `nodes` itself is untouched and still contains every node
    // including views -- gallocr's allocation walk still needs to see them
    // to resolve each view's data pointer (see gallocr.cpp) and must stay
    // in the original topological order for its refcount-driven free
    // logic; only the threadpool's execution loop reads exec_nodes.
    mi_tensor** exec_nodes;
    int n_exec_nodes;

    int capacity;  // max nodes/leafs/exec_nodes this graph object can hold
};

// Graph objects are metadata-only and, like tensors, come from a context so
// that building a fresh graph every forward step is allocation-free.
mi_cgraph* mi_graph_new(mi_context* ctx, int capacity);

// Walks the dependency DAG from `output` via src[0]/src[1] (post-order DFS,
// dedup by visited-set), appending compute nodes to g->nodes in topological
// order and leaf tensors (op == NONE) to g->leafs. Resets the graph first,
// so this is the *first* root of a forward step.
void mi_graph_build_forward(mi_cgraph* g, mi_tensor* output);

// Same DFS as mi_graph_build_forward, but does not reset g first -- for
// every *additional* root after the first. Needed because a cpy into a
// slice of a persistent/shared buffer (KV-cache append, or one head's
// slice of a multi-head attention output) is never anyone's src[]: nothing
// downstream reads "the cpy tensor", it reads the underlying buffer
// through a *different* view built straight from that buffer's leaf.
// Expanding from every such cpy result before expanding from the model's
// final output guarantees each write lands in g->nodes (in array order,
// which is execution order) before whatever later reads the buffer it
// wrote into -- see model_qwen2.cpp and docs/stage2a.md for why a single
// DFS from just the final output would silently skip every KV-cache write.
void mi_graph_build_forward_expand(mi_cgraph* g, mi_tensor* output);

// Executes every node in g->nodes in order, dispatching to the matching
// ops_cpu kernel. `pool` is null before stage 3 (single-threaded); from
// stage 3 on, execute() hands each node to the threadpool and blocks on a
// barrier between nodes (see docs/architecture.md and threadpool.h for the
// memory-order reasoning).
void mi_graph_compute(mi_cgraph* g, mi_threadpool* pool);

// ---------------------------------------------------------------------------
// Node builders. Each returns a new result tensor with op/src/op_params set
// and data == nullptr (filled later by gallocr's arena or, in stage 2a
// before gallocr exists, by the context's own bump allocator).
// ---------------------------------------------------------------------------
mi_tensor* mi_graph_get_rows(mi_context* ctx, mi_tensor* a, mi_tensor* ids);
mi_tensor* mi_graph_rms_norm(mi_context* ctx, mi_tensor* a, float eps);
mi_tensor* mi_graph_mul(mi_context* ctx, mi_tensor* a, mi_tensor* b);
mi_tensor* mi_graph_add(mi_context* ctx, mi_tensor* a, mi_tensor* b);
// Quantizes `a` (F32) to Q8_0, row by row. Exposed mainly so
// mi_graph_mul_mat can insert it transparently (see below); tests may also
// build it directly to check the graph/gallocr wiring around it in
// isolation from the full model.
mi_tensor* mi_graph_quantize_q8_0(mi_context* ctx, mi_tensor* a);
// If `a` is Q8_0 (a weight loaded from a Q8_0 GGUF), transparently inserts
// a mi_graph_quantize_q8_0(b) node and uses its output as the actual
// second operand -- so every caller (model_qwen2.cpp's linear()) just
// calls mi_graph_mul_mat(ctx, weight, activation) regardless of weight
// type, per SPEC.md's "按每个张量的实际类型分派". The quantized-activation
// tensor is an ordinary graph node from gallocr's point of view: no
// scratch buffer, no special-cased allocation path.
mi_tensor* mi_graph_mul_mat(mi_context* ctx, mi_tensor* a, mi_tensor* b);
// n_past: position of b's first row in the sequence (NEOX pairing uses
// absolute position, computed per-row inside the kernel as n_past + row).
mi_tensor* mi_graph_rope(mi_context* ctx, mi_tensor* a, int n_past, int n_dims,
                          float freq_base);
// Causal softmax: row i (0-based within the batch) may attend to columns
// [0, n_past + i]; scale is applied before the mask. No separate mask
// tensor is materialized. Stage 6: `a` may be 2D [n_kv,n_tokens] or 3D
// [n_kv,n_tokens,n_head] -- every (token,head) pair is an independent row,
// same as every (token) column used to be alone (see ops_cpu.cpp).
mi_tensor* mi_graph_softmax(mi_context* ctx, mi_tensor* a, int n_past, float scale);
mi_tensor* mi_graph_silu(mi_context* ctx, mi_tensor* a);
mi_tensor* mi_graph_scale(mi_context* ctx, mi_tensor* a, float s);
// Writes a into dst_view's storage at dst_view's existing view offset
// (used to append K/V rows into the persistent KV cache tensor).
mi_tensor* mi_graph_cpy(mi_context* ctx, mi_tensor* a, mi_tensor* dst_view);

mi_tensor* mi_graph_reshape(mi_context* ctx, mi_tensor* a, int n_dims, const int64_t* ne);
mi_tensor* mi_graph_view(mi_context* ctx, mi_tensor* a, int n_dims, const int64_t* ne,
                          const size_t* nb, size_t offset);
mi_tensor* mi_graph_permute(mi_context* ctx, mi_tensor* a, int axis0, int axis1,
                             int axis2, int axis3);
mi_tensor* mi_graph_transpose(mi_context* ctx, mi_tensor* a);

// Stage 6 step 3b: batched multi-head attention (see tensor.h's
// MI_OP_ATTN_QK/MI_OP_ATTN_KQV doc comments for the exact per-element
// formulas). cache_k/cache_v are always the full, persistent KV-cache
// tensors -- never per-head views -- the GQA head-sharing mapping
// (kvh = h / n_rep) is applied inside the kernel.
mi_tensor* mi_graph_attn_qk(mi_context* ctx, mi_tensor* q_rot, mi_tensor* cache_k, int n_past,
                             int n_rep);
mi_tensor* mi_graph_attn_kqv(mi_context* ctx, mi_tensor* probs, mi_tensor* cache_v, int n_rep);
