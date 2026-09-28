// mini-infer: CPU compute kernels.
//
// Every kernel has the signature `void mi_compute_X(mi_tensor* dst, int ith,
// int nth)`: it reads dst->src[0] (and dst->src[1] where applicable) plus
// dst->op_params, and writes into dst->data, which the caller has already
// allocated. `ith`/`nth` select this thread's slice of dst's output rows.
//
// This single signature style is what lets stage 1 (direct calls, ith=0,
// nth=1), stage 2 (graph execute(), still ith=0, nth=1 until stage 3), and
// stage 3 (threadpool calling every worker with its real ith/nth) all run
// the exact same code path — see docs/architecture.md, "kernel reuse across
// stages", for the full argument.
#pragma once

#include "tensor.h"

// ---------------------------------------------------------------------------
// op_params access. mi_tensor::op_params is a fixed int32_t[] shared by all
// ops; each op defines its own slot layout (documented per-op below) and
// values are memcpy'd in/out so a slot can hold either an int32_t or a
// float's bit pattern.
// ---------------------------------------------------------------------------
void mi_op_params_set_f32(mi_tensor* t, int slot, float v);
float mi_op_params_get_f32(const mi_tensor* t, int slot);
void mi_op_params_set_i32(mi_tensor* t, int slot, int32_t v);
int32_t mi_op_params_get_i32(const mi_tensor* t, int slot);

// ---------------------------------------------------------------------------
// mul_mat parallel-mode control (stage 3). mi_compute_mul_mat keeps the
// fixed (dst, ith, nth) signature; *how* it splits dst's n output rows
// across nth callers is a small piece of global config set once by
// mi_threadpool_new, never touched by stage 1/2's direct or dump-
// interleaved calls (those always run nth == 1, and get the same answer
// under every mode -- see docs/stage3.md, "why mode/thread-count can't
// change bit-exactness": each output row is written by exactly one thread,
// with the same inner k-loop order regardless of which thread that is or
// how rows were assigned, so row-parallelism never reorders any floating-
// point accumulation).
// ---------------------------------------------------------------------------
enum mi_mulmat_mode {
  MI_MULMAT_SINGLE,   // only ith == 0 computes; other callers no-op
  MI_MULMAT_STATIC,   // mi_split_range(n, ith, nth) -- the default, and the
                       // only mode stage 1/2 ever implicitly used (nth == 1)
  MI_MULMAT_DYNAMIC,  // rows claimed via atomic fetch_add in chunks
};

void mi_mulmat_set_mode(mi_mulmat_mode mode, int dynamic_chunk);

// Must be called by exactly one thread, strictly before any of nth callers
// start MI_MULMAT_DYNAMIC's row-claiming loop for the *next* mul_mat node
// dispatched -- see threadpool.cpp's barrier_wait, which calls this from
// whichever thread is last to arrive at the barrier, before releasing the
// others. Harmless to call under SINGLE/STATIC modes.
void mi_mulmat_reset_dynamic_counter();

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// dst = src0[src1[i], :] row gather. src0: [ne0, n_rows], src1: I32 [n_ids].
// dst: [ne0, n_ids]. No op_params.
void mi_compute_get_rows(mi_tensor* dst, int ith, int nth);

// dst = rms_norm(src0) along ne[0]. op_params[0] = eps (f32).
void mi_compute_rms_norm(mi_tensor* dst, int ith, int nth);

// dst = src0 * src1 (elementwise, src1 broadcasts over trailing dims same
// as ggml's row-broadcast rule: src1->ne[0] == src0->ne[0]). No op_params.
void mi_compute_mul(mi_tensor* dst, int ith, int nth);

// dst = src0 + src1, same broadcast rule as mul. No op_params.
void mi_compute_add(mi_tensor* dst, int ith, int nth);

// dst = src0^T @ src1 in the ggml sense: src0 [k, n] (weight, row i is the
// i-th output's weight vector of length k = src0->ne[0]), src1 [k, m]
// (activations), dst [n, m]. dst->ne[0] == src0->ne[1], dst->ne[1] ==
// src1->ne[1]. No op_params; type dispatch (F32/F16/Q8_0 weight) happens
// inside based on src0->type.
void mi_compute_mul_mat(mi_tensor* dst, int ith, int nth);

// NEOX-style RoPE in place shape (dst is a fresh tensor, src0 unmodified).
// Pairs element i with element i + n_dims/2 within each head (not adjacent
// pairs). Row r of src0 (0-based within this call's batch) is rotated at
// absolute position n_past + r.
// op_params[0] = n_past (i32), op_params[1] = n_dims (i32, rotary dims,
// <= head_dim), op_params[2] = freq_base bits (f32).
void mi_compute_rope(mi_tensor* dst, int ith, int nth);

// Causal softmax over ne[0] (key positions) with an implicit mask: output
// row r (0-based within the batch) may attend to columns [0, n_past + r];
// later columns are treated as -inf before the max-subtract/exp/normalize.
// Scale is applied to src0 before the mask.
// op_params[0] = n_past (i32), op_params[1] = scale bits (f32).
// Stage 6: src0 may be 2D [n_kv,n_tokens] or 3D [n_kv,n_tokens,n_head] --
// every (token,head) pair is an independent row (the causal mask only
// depends on the token index, same for every head); n_head defaults to 1
// via mi_tensor's ne[]-padding, so the 2D case is unchanged.
void mi_compute_softmax(mi_tensor* dst, int ith, int nth);

// Stage 6 step 3b: batched multi-head attention QK scores. dst =
// [n_kv_total, n_tokens, n_head]; dst->src[0] = q_rot
// [head_dim, n_head, n_tokens] (contiguous); dst->src[1] = the full
// persistent K cache [head_dim, n_head_kv, n_ctx] (never a per-head view --
// the GQA mapping kvh = h / n_rep is applied inside this kernel).
// op_params[0] = n_past (i32), op_params[1] = n_rep (i32). See tensor.h's
// MI_OP_ATTN_QK doc comment for the exact per-element formula.
void mi_compute_attn_qk(mi_tensor* dst, int ith, int nth);

// Stage 6 step 3b: batched multi-head attention output (scores @ V). dst =
// [head_dim, n_head, n_tokens]; dst->src[0] = probs
// [n_kv_total, n_tokens, n_head] (softmax output); dst->src[1] = the full
// persistent V cache [head_dim, n_head_kv, n_ctx]. op_params[0] = n_rep
// (i32). See tensor.h's MI_OP_ATTN_KQV doc comment for the exact
// per-element formula.
void mi_compute_attn_kqv(mi_tensor* dst, int ith, int nth);

// dst = silu(src0) = src0 * sigmoid(src0). No op_params.
void mi_compute_silu(mi_tensor* dst, int ith, int nth);

// dst = src0 * s. op_params[0] = s (f32).
void mi_compute_scale(mi_tensor* dst, int ith, int nth);

// Copies src0 into dst's storage (dst is normally a view into a persistent
// tensor, e.g. one layer's KV cache slice). Requires same shape; dst may
// have different strides (nb) than a contiguous tensor, e.g. when writing
// one token's row into [n_ctx, n_kv_head, head_dim] KV storage. No op_params.
void mi_compute_cpy(mi_tensor* dst, int ith, int nth);

// dst (Q8_0) = quantize_row_q8_0(src0), row by row. src0 must be F32 and
// contiguous in dim 0 (always true here: src0 is a freshly graph-computed
// activation, never a weight or a strided view -- see mi_graph_mul_mat,
// the only builder that creates this node). No op_params.
void mi_compute_quantize_q8_0(mi_tensor* dst, int ith, int nth);

// Dispatches on dst->op to the matching mi_compute_* above. Used by
// mi_graph_compute(); view ops (RESHAPE/VIEW/PERMUTE/TRANSPOSE) and
// MI_OP_NONE are no-ops here since they carry no data movement.
void mi_compute_forward(mi_tensor* dst, int ith, int nth);
