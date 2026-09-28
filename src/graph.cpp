#include "graph.h"

#include "ops_cpu.h"
#include "threadpool.h"

namespace {

// dst has the same shape as `a` -- the common case for elementwise/
// normalization ops. Copies all 4 dims (trailing ones are 1 for tensors
// with fewer "real" dims), so this works for both 2D activations and 3D
// per-head tensors without the builder needing to know which.
mi_tensor* new_like(mi_context* ctx, const mi_tensor* a) {
  return mi_context_new_tensor(ctx, a->type, 4, a->ne);
}

}  // namespace

mi_cgraph* mi_graph_new(mi_context* ctx, int capacity) {
  auto* g = (mi_cgraph*)mi_context_alloc_bytes(ctx, sizeof(mi_cgraph));
  g->nodes = (mi_tensor**)mi_context_alloc_bytes(ctx, sizeof(mi_tensor*) * (size_t)capacity);
  g->leafs = (mi_tensor**)mi_context_alloc_bytes(ctx, sizeof(mi_tensor*) * (size_t)capacity);
  g->exec_nodes = (mi_tensor**)mi_context_alloc_bytes(ctx, sizeof(mi_tensor*) * (size_t)capacity);
  g->n_nodes = 0;
  g->n_leafs = 0;
  g->n_exec_nodes = 0;
  g->capacity = capacity;
  return g;
}

namespace {

bool already_visited(const mi_cgraph* g, const mi_tensor* t) {
  for (int i = 0; i < g->n_nodes; i++) {
    if (g->nodes[i] == t) return true;
  }
  for (int i = 0; i < g->n_leafs; i++) {
    if (g->leafs[i] == t) return true;
  }
  return false;
}

// Post-order DFS: visits both srcs before appending `t` itself, so
// g->nodes ends up topologically sorted (every node appears after
// everything it reads). already_visited() doubles as the "visited set" --
// once appended to nodes/leafs, a tensor is never traversed again, which
// is what makes a diamond-shaped dependency (e.g. two heads both reading
// the same RoPE'd Q tensor) safe instead of duplicated or infinitely
// recursive.
void visit(mi_cgraph* g, mi_tensor* t) {
  if (!t || already_visited(g, t)) return;
  visit(g, t->src[0]);
  visit(g, t->src[1]);
  if (t->op == MI_OP_NONE) {
    g->leafs[g->n_leafs++] = t;
  } else {
    g->nodes[g->n_nodes++] = t;
    // Stage 6: exec_nodes is a strict subsequence of nodes (same relative
    // topological order), built right here at graph-build time -- see
    // graph.h's doc comment on why this can't wait for gallocr.
    if (!mi_op_is_view(t->op)) g->exec_nodes[g->n_exec_nodes++] = t;
  }
}

}  // namespace

void mi_graph_build_forward(mi_cgraph* g, mi_tensor* output) {
  g->n_nodes = 0;
  g->n_leafs = 0;
  g->n_exec_nodes = 0;
  visit(g, output);
}

void mi_graph_build_forward_expand(mi_cgraph* g, mi_tensor* output) { visit(g, output); }

void mi_graph_compute(mi_cgraph* g, mi_threadpool* pool) {
  if (pool) {
    mi_threadpool_run_graph(pool, g);
    return;
  }
  for (int i = 0; i < g->n_nodes; i++) {
    mi_compute_forward(g->nodes[i], 0, 1);
  }
}

mi_tensor* mi_graph_get_rows(mi_context* ctx, mi_tensor* a, mi_tensor* ids) {
  mi_tensor* dst = mi_new_tensor_2d(ctx, MI_TYPE_F32, a->ne[0], ids->ne[0]);
  dst->op = MI_OP_GET_ROWS;
  dst->src[0] = a;
  dst->src[1] = ids;
  return dst;
}

mi_tensor* mi_graph_rms_norm(mi_context* ctx, mi_tensor* a, float eps) {
  mi_tensor* dst = new_like(ctx, a);
  dst->op = MI_OP_RMS_NORM;
  dst->src[0] = a;
  mi_op_params_set_f32(dst, 0, eps);
  return dst;
}

mi_tensor* mi_graph_mul(mi_context* ctx, mi_tensor* a, mi_tensor* b) {
  mi_tensor* dst = new_like(ctx, a);
  dst->op = MI_OP_MUL;
  dst->src[0] = a;
  dst->src[1] = b;
  return dst;
}

mi_tensor* mi_graph_add(mi_context* ctx, mi_tensor* a, mi_tensor* b) {
  mi_tensor* dst = new_like(ctx, a);
  dst->op = MI_OP_ADD;
  dst->src[0] = a;
  dst->src[1] = b;
  return dst;
}

mi_tensor* mi_graph_quantize_q8_0(mi_context* ctx, mi_tensor* a) {
  // Stage 6: output is MI_TYPE_Q8_0_ACT (float scale), not MI_TYPE_Q8_0
  // (fp16 scale, the real GGUF weight layout) -- this tensor never touches
  // a GGUF file, only mul_mat's Q8_0 branch reads it, so there's no format
  // to match and storing the scale as float avoids a redundant fp16 round
  // trip on data that started out as F32 anyway. See quant.h's
  // block_q8_0_act.
  mi_tensor* dst = mi_context_new_tensor(ctx, MI_TYPE_Q8_0_ACT, 4, a->ne);
  dst->op = MI_OP_QUANTIZE_Q8_0;
  dst->src[0] = a;
  return dst;
}

mi_tensor* mi_graph_mul_mat(mi_context* ctx, mi_tensor* a, mi_tensor* b) {
  // Stage 6: if the caller already quantized b itself (see
  // model_qwen2.cpp's maybe_quantize_shared -- Q/K/V and ffn_gate/ffn_up
  // read the *same* activation tensor with different Q8_0 weights, so
  // quantizing it once and passing the Q8_0_ACT result to every one of
  // those mul_mat calls avoids re-quantizing identical data 2-3 times per
  // layer), don't insert a second quantize node on top of it.
  bool already_quantized = b->type == MI_TYPE_Q8_0_ACT;
  mi_tensor* rhs = (a->type == MI_TYPE_Q8_0 && !already_quantized) ? mi_graph_quantize_q8_0(ctx, b) : b;
  mi_tensor* dst = mi_new_tensor_2d(ctx, MI_TYPE_F32, a->ne[1], b->ne[1]);
  dst->op = MI_OP_MUL_MAT;
  dst->src[0] = a;
  dst->src[1] = rhs;
  return dst;
}

mi_tensor* mi_graph_rope(mi_context* ctx, mi_tensor* a, int n_past, int n_dims, float freq_base) {
  mi_tensor* dst = new_like(ctx, a);
  dst->op = MI_OP_ROPE;
  dst->src[0] = a;
  mi_op_params_set_i32(dst, 0, n_past);
  mi_op_params_set_i32(dst, 1, n_dims);
  mi_op_params_set_f32(dst, 2, freq_base);
  return dst;
}

mi_tensor* mi_graph_softmax(mi_context* ctx, mi_tensor* a, int n_past, float scale) {
  mi_tensor* dst = new_like(ctx, a);
  dst->op = MI_OP_SOFTMAX;
  dst->src[0] = a;
  mi_op_params_set_i32(dst, 0, n_past);
  mi_op_params_set_f32(dst, 1, scale);
  return dst;
}

mi_tensor* mi_graph_silu(mi_context* ctx, mi_tensor* a) {
  mi_tensor* dst = new_like(ctx, a);
  dst->op = MI_OP_SILU;
  dst->src[0] = a;
  return dst;
}

mi_tensor* mi_graph_scale(mi_context* ctx, mi_tensor* a, float s) {
  mi_tensor* dst = new_like(ctx, a);
  dst->op = MI_OP_SCALE;
  dst->src[0] = a;
  mi_op_params_set_f32(dst, 0, s);
  return dst;
}

mi_tensor* mi_graph_cpy(mi_context* ctx, mi_tensor* a, mi_tensor* dst_view) {
  (void)ctx;  // dst_view already owns its (aliased) storage; no allocation here
  dst_view->op = MI_OP_CPY;
  dst_view->src[0] = a;
  return dst_view;
}

mi_tensor* mi_graph_reshape(mi_context* ctx, mi_tensor* a, int n_dims, const int64_t* ne) {
  // Every reshape call in this project targets a fresh, contiguous
  // producer, so the new shape's natural (contiguous) strides are always
  // correct -- reshape never needs to re-derive strides from `a`'s own
  // (possibly different-rank) layout.
  size_t nb[4] = {0, 0, 0, 0};
  size_t block_bytes = mi_type_block_bytes(a->type);
  int64_t block_size = (int64_t)mi_type_block_size(a->type);
  nb[0] = block_bytes;
  if (n_dims > 1) nb[1] = nb[0] * (size_t)(ne[0] / block_size);
  for (int i = 2; i < n_dims; i++) nb[i] = nb[i - 1] * (size_t)ne[i - 1];

  mi_tensor* t = mi_context_view(ctx, a, a->type, n_dims, ne, nb, 0);
  t->op = MI_OP_RESHAPE;
  t->src[0] = a;
  return t;
}

mi_tensor* mi_graph_view(mi_context* ctx, mi_tensor* a, int n_dims, const int64_t* ne,
                          const size_t* nb, size_t offset) {
  mi_tensor* t = mi_context_view(ctx, a, a->type, n_dims, ne, nb, offset);
  t->op = MI_OP_VIEW;
  t->src[0] = a;
  return t;
}

// axisK: output dimension K takes its size/stride from input dimension
// axisK (numpy.transpose(a, axes) convention), always over the full 4 dims.
mi_tensor* mi_graph_permute(mi_context* ctx, mi_tensor* a, int axis0, int axis1, int axis2,
                             int axis3) {
  int perm[4] = {axis0, axis1, axis2, axis3};
  int64_t ne[4];
  size_t nb[4];
  for (int i = 0; i < 4; i++) {
    ne[i] = a->ne[perm[i]];
    nb[i] = a->nb[perm[i]];
  }
  mi_tensor* t = mi_context_view(ctx, a, a->type, 4, ne, nb, 0);
  t->op = MI_OP_PERMUTE;
  t->src[0] = a;
  return t;
}

// Swaps the first two dimensions (the "matrix transpose" case); dims 2/3
// are left alone. Used for the scores@V attention step's V-cache read (see
// model_qwen2.cpp) without ever materializing a physically-transposed copy.
// Stage 6 step 3b: batched multi-head attention QK scores. q_rot:
// [head_dim, n_head, n_tokens] (contiguous, RoPE's output); cache_k: the
// full persistent K cache [head_dim, n_head_kv, n_ctx] -- passed whole
// (never a per-head view) and indexed inside the kernel via the GQA
// mapping kvh = h / n_rep, so no K data is ever copied or re-viewed per
// head. dst: [n_kv_total, n_tokens, n_head]. See tensor.h's MI_OP_ATTN_QK
// doc comment for the exact per-element formula and why it's bit-exact
// with the old per-head mul_mat loop.
mi_tensor* mi_graph_attn_qk(mi_context* ctx, mi_tensor* q_rot, mi_tensor* cache_k, int n_past,
                             int n_rep) {
  int64_t n_head = q_rot->ne[1];
  int64_t n_tokens = q_rot->ne[2];
  int64_t n_kv_total = n_past + n_tokens;
  mi_tensor* dst = mi_new_tensor_3d(ctx, MI_TYPE_F32, n_kv_total, n_tokens, n_head);
  dst->op = MI_OP_ATTN_QK;
  dst->src[0] = q_rot;
  dst->src[1] = cache_k;
  mi_op_params_set_i32(dst, 0, n_past);
  mi_op_params_set_i32(dst, 1, n_rep);
  return dst;
}

// Stage 6 step 3b: batched multi-head attention output (scores @ V).
// probs: [n_kv_total, n_tokens, n_head] (softmax output); cache_v: the full
// persistent V cache [head_dim, n_head_kv, n_ctx], same "pass whole, index
// with GQA mapping inside the kernel" approach as mi_graph_attn_qk. dst:
// [head_dim, n_head, n_tokens] -- deliberately the same "head_dim
// innermost, heads consecutive" contiguous layout reshape_heads uses, so
// the caller can turn this directly into attn_concat via a free
// mi_graph_reshape, no per-head cpy needed.
mi_tensor* mi_graph_attn_kqv(mi_context* ctx, mi_tensor* probs, mi_tensor* cache_v, int n_rep) {
  int64_t n_tokens = probs->ne[1];
  int64_t n_head = probs->ne[2];
  int64_t head_dim = cache_v->ne[0];
  mi_tensor* dst = mi_new_tensor_3d(ctx, MI_TYPE_F32, head_dim, n_head, n_tokens);
  dst->op = MI_OP_ATTN_KQV;
  dst->src[0] = probs;
  dst->src[1] = cache_v;
  mi_op_params_set_i32(dst, 0, n_rep);
  return dst;
}

mi_tensor* mi_graph_transpose(mi_context* ctx, mi_tensor* a) {
  int64_t ne[4] = {a->ne[1], a->ne[0], a->ne[2], a->ne[3]};
  size_t nb[4] = {a->nb[1], a->nb[0], a->nb[2], a->nb[3]};
  mi_tensor* t = mi_context_view(ctx, a, a->type, 4, ne, nb, 0);
  t->op = MI_OP_TRANSPOSE;
  t->src[0] = a;
  return t;
}
