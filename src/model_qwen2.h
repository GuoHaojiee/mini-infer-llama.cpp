// mini-infer: Qwen2(.5) model definition and forward pass.
//
// This header is stable across stages 1-4: stage 1 implements
// mi_qwen2_forward() with direct ops_cpu calls, stage 2 rewrites its body to
// build+execute a graph, stage 3 runs that graph through the threadpool,
// and stage 4 adds F16/Q8_0 weight handling — the signature here does not
// change, so main.cpp and the dump/compare tooling built against it don't
// change either. See docs/architecture.md.
#pragma once

#include <cstdint>
#include <string>

#include "context.h"
#include "gguf.h"
#include "tensor.h"
#include "threadpool.h"

struct mi_gallocr;  // stage 2b, see gallocr.h; kept as a pointer here so this
                     // header doesn't need to pull in gallocr.h's includes
struct mi_cgraph;    // stage 2, see graph.h; same reasoning

// ---------------------------------------------------------------------------
// Hyperparameters — always read from GGUF metadata, never hardcoded
// (SPEC.md 模型事实). n_ctx is a runtime choice (-c), independent of and
// bounded by n_ctx_train.
// ---------------------------------------------------------------------------
struct mi_qwen2_hparams {
    int n_vocab;
    int n_embd;
    int n_layer;
    int n_head;
    int n_head_kv;
    int n_ff;
    int head_dim;        // derived: n_embd / n_head
    float rms_eps;
    float rope_freq_base;
    int n_ctx_train;     // GGUF context_length: an upper bound, not n_ctx
};

bool mi_qwen2_hparams_load(const mi_gguf_context* gguf, mi_qwen2_hparams* out,
                            std::string* err);

// ---------------------------------------------------------------------------
// Weights. Every pointer aliases the GGUF mmap (zero-copy); tensors are
// plain mi_tensor leaves (op = NONE, flags = MI_TENSOR_FLAG_WEIGHT) created
// via mi_context_new_tensor_from_data so kernels can take them directly as
// srcs. Per-tensor dtype (F32/F16/Q8_0) is whatever the GGUF file actually
// stored for that tensor (SPEC.md: don't assume a uniform type).
// ---------------------------------------------------------------------------
struct mi_qwen2_layer_weights {
    mi_tensor* attn_norm;
    mi_tensor* attn_q_w;
    mi_tensor* attn_q_b;
    mi_tensor* attn_k_w;
    mi_tensor* attn_k_b;
    mi_tensor* attn_v_w;
    mi_tensor* attn_v_b;
    mi_tensor* attn_output_w;  // no bias
    mi_tensor* ffn_norm;
    mi_tensor* ffn_gate_w;
    mi_tensor* ffn_up_w;
    mi_tensor* ffn_down_w;     // no bias on any FFN projection
};

struct mi_qwen2_weights {
    mi_tensor* token_embd;
    mi_tensor* output_norm;
    // output.weight if present in the GGUF, else == token_embd (tied
    // embedding/lm_head per SPEC.md).
    mi_tensor* output;
    mi_qwen2_layer_weights* layers;  // hparams.n_layer entries
};

struct mi_qwen2_model {
    mi_gguf_context* gguf;   // owns the mmap; kept alive for the model's lifetime
    mi_vocab* vocab;
    mi_qwen2_hparams hparams;
    mi_context* weight_ctx;  // metadata-only pool for weight tensor structs
    mi_qwen2_weights weights;
};

mi_qwen2_model* mi_qwen2_model_load(const char* gguf_path, std::string* err);
void mi_qwen2_model_free(mi_qwen2_model* model);

// ---------------------------------------------------------------------------
// KV cache. Preallocated once for n_ctx tokens (SPEC.md:
// [n_layer][n_ctx][n_kv_head][head_dim]); realized as one persistent tensor
// pair per layer, each shaped ne = {head_dim, n_head_kv, n_ctx} (ne[0]
// innermost, matching the project-wide GGUF-style dimension convention).
//
// Stage 2b anchors mini-infer's two other reserve-once, reuse-every-step
// resources here, since both are sized from exactly the same (hparams,
// n_ctx) pair this constructor already takes:
//   - galloc: the gallocr arena for every non-persistent intermediate
//     tensor's data, reserved once against the worst-case prefill
//     (n_tokens = n_ctx) and decode (n_past = n_ctx - 1) graphs.
//   - graph_ctx: the per-step graph-building context (tensor metadata,
//     mi_cgraph bookkeeping, and the eagerly-allocated `ids`/`logits`
//     tensors -- see model_qwen2.cpp). mi_qwen2_forward calls
//     mi_context_reset() on it, never mi_context_new/free -- allocating a
//     fresh context every step would itself violate stage 2b's "zero heap
//     allocation after reserve" goal regardless of how good gallocr's own
//     arena reuse is.
// Stage 3 adds a third: pool, the persistent threadpool. It doesn't depend
// on n_ctx the way galloc/graph_ctx do, but its lifetime is the same
// (spawned once, torn down once), so it lives here rather than being
// passed around separately.
// ---------------------------------------------------------------------------
struct mi_kv_cache {
    mi_context* ctx;   // owns the K/V tensor metadata + their backing storage
    mi_tensor** k;      // hparams.n_layer entries
    mi_tensor** v;      // hparams.n_layer entries
    int n_layer;
    int n_ctx;
    int n_past;  // number of tokens currently stored (0 at the start of a sequence)

    mi_gallocr* galloc;
    mi_context* graph_ctx;
    mi_threadpool* pool;
};

// tp_params defaults to a single-thread pool (n_threads=1) so every
// existing call site (tests, tools) that doesn't care about stage 3 keeps
// running exactly one thread, unchanged -- mi_qwen2_forward always routes
// through mi_graph_compute(g, kv->pool) now, and a 1-thread pool is bit-
// identical to the old pool-less path (see docs/stage3.md).
//
// `weights` (stage 4) must be the same model's already-loaded weights:
// reserve_gallocr's worst-case rehearsal graphs are built with dummy
// tensors that copy each real weight's *type* (F32/F16/Q8_0), because
// mi_graph_mul_mat's behavior (whether it inserts a Q8_0-quantize node)
// and every tensor's byte size both depend on that type -- rehearsing
// against the wrong type would size (or shape) the arena for a graph the
// real forward pass never builds. See docs/stage4.md.
mi_kv_cache* mi_kv_cache_init(const mi_qwen2_hparams& hparams, const mi_qwen2_weights& weights,
                               int n_ctx,
                               const mi_threadpool_params& tp_params = {1, true, MI_MULMAT_STATIC, 8});
void mi_kv_cache_free(mi_kv_cache* kv);

// ---------------------------------------------------------------------------
// Forward pass + sampling
// ---------------------------------------------------------------------------

// Optional hook for exporting intermediate tensors (--dump). Checkpoint
// names used by mi_qwen2_forward: "embd" (post-embedding, pre-layer-0),
// "layer_{i}" (output of layer i, 0-based), "output_norm" (post-final-norm
// hidden state — the tensor HF's last output_hidden_states entry
// corresponds to), "logits".
struct mi_dump_sink {
    void (*on_tensor)(void* user_data, const char* name, const mi_tensor* t);
    void* user_data;
};

struct mi_qwen2_forward_params {
    const int32_t* tokens;
    int n_tokens;   // > 1 for prefill, == 1 for decode
    int n_past;     // tokens already resident in kv before this call
    mi_dump_sink* dump;  // may be nullptr
    // Stage 6 measurement hook: if non-null, *out_graph is set to this
    // call's scheduled mi_cgraph* right after it's built (before
    // execution), so a caller (main.cpp's --graph-stats) can inspect
    // g->nodes/n_nodes -- e.g. an op histogram -- without mi_qwen2_forward
    // needing to know what "stats" means. Left null by every existing
    // caller, so this is a pure addition with no effect unless opted in.
    mi_cgraph** out_graph = nullptr;
    // Stage 6 wrap-up: when true, logits_out only gets n_vocab floats (the
    // *last* token's logits), not n_tokens * n_vocab -- real generation
    // (and --bench, which measures it) only ever samples from the last
    // prefill position, so computing the lm_head projection (an
    // [n_embd,n_vocab] matmul, this model's largest by far: n_vocab is
    // 151936) for every one of n_tokens positions during prefill wastes
    // O(n_tokens) of that cost on results nobody reads. Left false by
    // every existing caller (including --dump's teacher-forcing path,
    // which needs every position's logits to compare against HF) and by
    // decode (n_tokens == 1, where this is a no-op either way), so this is
    // a pure opt-in with no effect unless requested.
    bool last_logits_only = false;
};

// Runs one forward step, writes kv's new K/V rows at [n_past, n_past +
// n_tokens), and writes n_tokens * hparams.n_vocab logits (row-major, row i
// = next-token distribution after tokens[0..i]) into logits_out, which the
// caller allocates with that many floats. Does not update kv->n_past; the
// caller does that after deciding the step succeeded.
void mi_qwen2_forward(mi_qwen2_model* model, mi_kv_cache* kv,
                       const mi_qwen2_forward_params& params, float* logits_out);

int32_t mi_qwen2_sample_greedy(const float* logits, int n_vocab);
