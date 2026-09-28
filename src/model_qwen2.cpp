#include "model_qwen2.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gallocr.h"
#include "graph.h"
#include "ops_cpu.h"

namespace {

bool gguf_need_u32(const mi_gguf_context* gguf, const char* key, uint32_t* v, std::string* err) {
  if (!mi_gguf_get_u32(gguf, key, v)) {
    if (err) *err = std::string("qwen2 hparams: missing or wrong-typed key '") + key + "'";
    return false;
  }
  return true;
}

bool gguf_need_f32(const mi_gguf_context* gguf, const char* key, float* v, std::string* err) {
  if (!mi_gguf_get_f32(gguf, key, v)) {
    if (err) *err = std::string("qwen2 hparams: missing or wrong-typed key '") + key + "'";
    return false;
  }
  return true;
}

mi_tensor* load_weight(mi_context* ctx, const mi_gguf_context* gguf, const std::string& name,
                        std::string* err) {
  const mi_gguf_tensor_info* info = mi_gguf_find_tensor(gguf, name.c_str());
  if (!info) {
    if (err) *err = "missing weight tensor: " + name;
    return nullptr;
  }
  // No type restriction here (stage 4): every kernel that reads a weight
  // (get_rows, mul_mat) dispatches on its actual type (F32/F16/Q8_0).
  // mi_gguf_open already rejected anything else (e.g. Q4_0) at parse time,
  // so any type that reaches this point is one this engine can compute
  // with -- see docs/stage4.md.
  mi_tensor* t =
      mi_context_new_tensor_from_data(ctx, info->type, info->n_dims, info->ne, (void*)info->data);
  t->flags = MI_TENSOR_FLAG_WEIGHT;
  mi_tensor_set_name(t, name.c_str());
  return t;
}

// -- forward-pass helpers (stage 2a: builds graph nodes via mi_graph_*
// instead of calling mi_compute_* directly; nothing here does any
// computation -- see docs/stage2a.md and docs/architecture.md's "kernel
// reuse across stages" for how stage 1's direct calls and this graph both
// end up running the exact same ops_cpu kernels) ------------------------

mi_tensor* rms_norm_and_scale(mi_context* ctx, mi_tensor* x, mi_tensor* weight, float eps) {
  return mi_graph_mul(ctx, mi_graph_rms_norm(ctx, x, eps), weight);
}

// w: [in, out] (GGUF convention, ne[0] = in), x: [in, n_tokens] -> [out, n_tokens].
mi_tensor* linear(mi_context* ctx, mi_tensor* w, mi_tensor* x, mi_tensor* bias) {
  mi_tensor* y = mi_graph_mul_mat(ctx, w, x);
  return bias ? mi_graph_add(ctx, y, bias) : y;
}

// Stage 6: Q/K/V (and separately ffn_gate/ffn_up) all read the *same*
// activation tensor with different Q8_0 weights -- without this,
// mi_graph_mul_mat would insert its own quantize node on every one of
// those calls, redundantly re-quantizing identical F32 data 2-3 times per
// layer (see docs/stage6.md's step-0/3 investigation). Quantizing once
// here and passing the result to every linear() call that shares this `x`
// lets mi_graph_mul_mat recognize (via b->type == MI_TYPE_Q8_0_ACT) that
// its activation operand is already quantized and skip inserting another
// quantize node -- see graph.cpp. `like_weight` is any one of the weights
// that will read this quantized result; assumes (true for how this
// project's models are always quantized -- llama-quantize applies one
// type to every linear-projection weight in a file) that Q/K/V's three
// weights, or ffn_gate/ffn_up's two, share the same type -- if that type
// isn't Q8_0, x is returned unchanged and every caller's mi_graph_mul_mat
// falls back to its own normal per-call handling (F32/F16 weights never
// quantize the activation at all).
mi_tensor* maybe_quantize_shared(mi_context* ctx, mi_tensor* x, const mi_tensor* like_weight) {
  if (like_weight->type != MI_TYPE_Q8_0) return x;
  return mi_graph_quantize_q8_0(ctx, x);
}

// [n_embd, n_tokens] contiguous -> [head_dim, n_head, n_tokens] view. Pure
// metadata: HF's own reshape (view(*, -1, head_dim)) uses the identical
// "head_dim innermost, heads are consecutive blocks" layout, confirmed
// against modeling_qwen2.py. mi_graph_reshape's contiguous-strides formula
// happens to reduce to exactly this because x is always a fresh mul_mat/
// add result here.
mi_tensor* reshape_heads(mi_context* ctx, mi_tensor* x, int64_t head_dim, int64_t n_head) {
  int64_t ne[3] = {head_dim, n_head, x->ne[1]};
  return mi_graph_reshape(ctx, x, 3, ne);
}

// cache: [head_dim, n_head_kv, n_ctx] -> the [head_dim, n_head_kv, n_tokens]
// write target starting at position n_past, for this step's cpy.
mi_tensor* kv_write_view(mi_context* ctx, mi_tensor* cache, int n_past, int64_t n_tokens) {
  int64_t ne[3] = {cache->ne[0], cache->ne[1], n_tokens};
  size_t nb[3] = {cache->nb[0], cache->nb[1], cache->nb[2]};
  return mi_graph_view(ctx, cache, 3, ne, nb, (size_t)n_past * cache->nb[2]);
}

// Append-only list backed by caller-provided (mi_context_alloc_bytes'd)
// storage instead of std::vector: stage 2b's whole point is that a forward
// step never calls malloc/free, and a fresh std::vector's first push_back
// would do exactly that every single call, regardless of how well gallocr
// itself behaves. capacity is sized once by the caller (see
// build_model_graph) and never exceeded, so push() never reallocates.
struct tensor_list {
  mi_tensor** items;
  int count;
  void push(mi_tensor* t) { items[count++] = t; }
};

tensor_list make_tensor_list(mi_context* ctx, size_t capacity) {
  tensor_list l;
  l.items = (mi_tensor**)mi_context_alloc_bytes(ctx, sizeof(mi_tensor*) * capacity);
  l.count = 0;
  return l;
}

// Appends a layer's computation to the graph and returns its output node.
// `side_effects` collects every cpy result (KV-cache writes, per-head
// attn_concat writes): none of them is ever read via src[] by anything
// downstream (the *reader* of a shared buffer is built straight from the
// buffer's leaf tensor, not from the cpy that wrote into part of it), so a
// single top-down DFS from the model's final output would never discover
// them and execute() would silently skip every KV-cache/attn-concat write.
// mi_qwen2_forward expands the graph from each of these first -- see
// docs/stage2a.md.
mi_tensor* forward_layer(mi_context* ctx, const mi_qwen2_hparams& hp,
                          const mi_qwen2_layer_weights& lw, mi_tensor* cache_k, mi_tensor* cache_v,
                          mi_tensor* cur, int n_past, tensor_list* side_effects) {
  int64_t n_tokens = cur->ne[1];
  float scale = 1.0f / std::sqrt((float)hp.head_dim);

  mi_tensor* residual = cur;
  mi_tensor* normed = rms_norm_and_scale(ctx, cur, lw.attn_norm, hp.rms_eps);

  // Stage 6: quantize `normed` once (if Q8_0) instead of once per q/k/v
  // linear() call -- see maybe_quantize_shared's doc comment.
  mi_tensor* normed_q = maybe_quantize_shared(ctx, normed, lw.attn_q_w);
  mi_tensor* q = linear(ctx, lw.attn_q_w, normed_q, lw.attn_q_b);
  mi_tensor* k = linear(ctx, lw.attn_k_w, normed_q, lw.attn_k_b);
  mi_tensor* v = linear(ctx, lw.attn_v_w, normed_q, lw.attn_v_b);

  mi_tensor* qh = reshape_heads(ctx, q, hp.head_dim, hp.n_head);
  mi_tensor* kh = reshape_heads(ctx, k, hp.head_dim, hp.n_head_kv);
  mi_tensor* vh = reshape_heads(ctx, v, hp.head_dim, hp.n_head_kv);

  mi_tensor* qh_rot = mi_graph_rope(ctx, qh, n_past, hp.head_dim, hp.rope_freq_base);
  mi_tensor* kh_rot = mi_graph_rope(ctx, kh, n_past, hp.head_dim, hp.rope_freq_base);

  mi_tensor* k_dst = mi_graph_cpy(ctx, kh_rot, kv_write_view(ctx, cache_k, n_past, n_tokens));
  side_effects->push(k_dst);
  mi_tensor* v_dst = mi_graph_cpy(ctx, vh, kv_write_view(ctx, cache_v, n_past, n_tokens));
  side_effects->push(v_dst);

  // Stage 6 step 3b: batched across all heads in one attn_qk + one softmax
  // + one attn_kqv + one free reshape (see docs/stage6.md), replacing what
  // used to be an 8-node-per-head loop (view/mul_mat/softmax/view/
  // transpose/mul_mat/cpy x n_head). GQA's head-sharing (kvh = h / n_rep)
  // is applied inside mi_compute_attn_qk/mi_compute_attn_kqv, reading
  // cache_k/cache_v whole -- no per-head K/V view or copy at all anymore.
  // Bit-exact with the old per-head loop: see mi_graph_attn_qk/kqv's doc
  // comments in graph.cpp for the per-element accumulation-order argument.
  int n_rep = hp.n_head / hp.n_head_kv;
  mi_tensor* scores = mi_graph_attn_qk(ctx, qh_rot, cache_k, n_past, n_rep);
  mi_tensor* probs = mi_graph_softmax(ctx, scores, n_past, scale);
  mi_tensor* kqv = mi_graph_attn_kqv(ctx, probs, cache_v, n_rep);  // [head_dim, n_head, n_tokens]
  // Free reshape: kqv's "head_dim innermost, heads consecutive" contiguous
  // layout is byte-for-byte the layout attn_output_w's linear() expects
  // (the same layout reshape_heads uses, just in reverse) -- no cpy needed.
  int64_t concat_ne[2] = {hp.n_embd, n_tokens};
  mi_tensor* attn_concat = mi_graph_reshape(ctx, kqv, 2, concat_ne);

  mi_tensor* attn_proj = linear(ctx, lw.attn_output_w, attn_concat, nullptr);
  mi_tensor* after_attn = mi_graph_add(ctx, residual, attn_proj);

  mi_tensor* residual2 = after_attn;
  mi_tensor* normed2 = rms_norm_and_scale(ctx, after_attn, lw.ffn_norm, hp.rms_eps);
  // Stage 6: same dedup as normed_q above, for gate/up's shared input.
  mi_tensor* normed2_q = maybe_quantize_shared(ctx, normed2, lw.ffn_gate_w);
  mi_tensor* gate = linear(ctx, lw.ffn_gate_w, normed2_q, nullptr);
  mi_tensor* up = linear(ctx, lw.ffn_up_w, normed2_q, nullptr);
  mi_tensor* act = mi_graph_mul(ctx, mi_graph_silu(ctx, gate), up);
  mi_tensor* down = linear(ctx, lw.ffn_down_w, act, nullptr);
  return mi_graph_add(ctx, residual2, down);
}

// Every distinct tensor built for one forward step ends up in exactly one
// of: the context's tensor pool, and (since every non-leaf op is a graph
// node, every leaf a graph leaf) mi_cgraph's nodes/leafs arrays. One shared,
// n_tokens-independent (the node *count* never depends on batch size, only
// tensor sizes do) estimate sizes both consistently, at reserve time and
// every step after.
size_t estimate_node_capacity(const mi_qwen2_hparams& hp) {
  // +7 per layer: stage 4's mi_graph_mul_mat inserts one extra
  // MI_OP_QUANTIZE_Q8_0 node per mul_mat call whose weight is Q8_0, and a
  // layer has up to 7 mul_mats (q/k/v/o + ffn gate/up/down). This capacity
  // is a single upper bound shared by F32/F16/Q8_0 models alike (real
  // per-step node counts for the non-Q8_0 cases are simply smaller than
  // this budget, never larger) -- see reserve_gallocr for why the actual
  // *sizing* still has to mirror the real model's per-tensor types.
  size_t tensors_per_layer = 24 + 10 * (size_t)hp.n_head + 7;
  // +1 outside the loop for the final lm_head mul_mat's own possible
  // quantize node.
  return (size_t)hp.n_layer * tensors_per_layer + 65;
}

// One forward step's graph: embedding, every layer, final norm, lm_head.
// Shared by mi_qwen2_forward (real weights/KV) and reserve_gallocr's
// worst-case sizing graphs (dummy weights, same real KV cache tensors --
// see below), so the two are guaranteed to build structurally identical
// graphs; nothing here does any computation, only builds node metadata.
struct built_graph {
  mi_tensor* embd;
  mi_tensor** layer_outputs;  // hp.n_layer entries, from mi_context_alloc_bytes
  mi_tensor* final_normed;
  mi_tensor* logits;
  tensor_list side_effects;
};

// Bytes build_model_graph itself needs from ctx beyond the tensors it
// builds (layer_outputs and side_effects' backing arrays) -- folded into
// estimate_graph_ctx_bytes / reserve_gallocr's planning-context sizing so
// every caller of build_model_graph stays allocation-free.
size_t estimate_build_graph_bookkeeping_bytes(const mi_qwen2_hparams& hp) {
  size_t layer_outputs_bytes = (size_t)hp.n_layer * sizeof(mi_tensor*);
  size_t side_effects_bytes = (size_t)hp.n_layer * (2 + (size_t)hp.n_head) * sizeof(mi_tensor*);
  return layer_outputs_bytes + side_effects_bytes;
}

built_graph build_model_graph(mi_context* ctx, const mi_qwen2_hparams& hp,
                               const mi_qwen2_weights& weights, mi_tensor** cache_k,
                               mi_tensor** cache_v, mi_tensor* ids, int n_past,
                               bool last_logits_only) {
  built_graph bg{};
  bg.layer_outputs =
      (mi_tensor**)mi_context_alloc_bytes(ctx, sizeof(mi_tensor*) * (size_t)hp.n_layer);
  bg.side_effects = make_tensor_list(ctx, (size_t)hp.n_layer * (2 + (size_t)hp.n_head));

  bg.embd = mi_graph_get_rows(ctx, weights.token_embd, ids);
  mi_tensor* cur = bg.embd;
  for (int l = 0; l < hp.n_layer; l++) {
    cur = forward_layer(ctx, hp, weights.layers[l], cache_k[l], cache_v[l], cur, n_past,
                         &bg.side_effects);
    bg.layer_outputs[l] = cur;
  }
  bg.final_normed = rms_norm_and_scale(ctx, cur, weights.output_norm, hp.rms_eps);
  // Stage 6 wrap-up: bg.final_normed itself always stays the full
  // [n_embd,n_tokens] tensor (dump reads the whole thing as the
  // "output_norm" checkpoint) -- only the lm_head matmul's input is
  // restricted to the last column, via a free view, when the caller says
  // it'll never read any other position's logits.
  mi_tensor* logits_input = bg.final_normed;
  if (last_logits_only && bg.final_normed->ne[1] > 1) {
    int64_t last = bg.final_normed->ne[1] - 1;
    int64_t ne[2] = {bg.final_normed->ne[0], 1};
    size_t nb[2] = {bg.final_normed->nb[0], bg.final_normed->nb[1]};
    logits_input = mi_graph_view(ctx, bg.final_normed, 2, ne, nb, (size_t)last * bg.final_normed->nb[1]);
  }
  bg.logits = linear(ctx, weights.output, logits_input, nullptr);
  return bg;
}

// Expands the graph from every side-effecting cpy write before the main
// output -- see forward_layer's comment and docs/stage2a.md/2b.md for why a
// single DFS from `logits` alone would miss them (nothing downstream reads
// a cpy result; readers are built straight from the buffer's leaf tensor
// instead, so the writes are otherwise invisible to a backward walk).
mi_cgraph* schedule_graph(mi_context* ctx, int capacity, const built_graph& bg) {
  mi_cgraph* g = mi_graph_new(ctx, capacity);
  bool first = true;
  for (int i = 0; i < bg.side_effects.count; i++) {
    mi_tensor* se = bg.side_effects.items[i];
    if (first) {
      mi_graph_build_forward(g, se);
      first = false;
    } else {
      mi_graph_build_forward_expand(g, se);
    }
  }
  if (first) {
    mi_graph_build_forward(g, bg.logits);
  } else {
    mi_graph_build_forward_expand(g, bg.logits);
  }
  return g;
}

// graph_ctx's own bookkeeping needs: the mi_cgraph struct and its three
// pointer arrays (mi_graph_new: nodes/leafs/exec_nodes, stage 6 added the
// third), the eagerly-allocated `logits` output sized for the worst-case
// n_tokens = n_ctx prefill (see mi_qwen2_forward: `logits` is always
// persistent/OUTPUT-flagged, never gallocr-managed, per SPEC.md's
// persistent-tensor list), and the small `ids` input buffer.
size_t estimate_graph_ctx_bytes(const mi_qwen2_hparams& hp, int n_ctx) {
  size_t node_capacity = estimate_node_capacity(hp);
  size_t bookkeeping = sizeof(mi_cgraph) + 3 * node_capacity * sizeof(mi_tensor*);
  size_t eager_logits = (size_t)hp.n_vocab * (size_t)n_ctx * sizeof(float);
  size_t eager_ids = (size_t)n_ctx * sizeof(int32_t);
  return bookkeeping + estimate_build_graph_bookkeeping_bytes(hp) + eager_logits + eager_ids +
         (1u << 16);
}

// -- reserve-time worst-case graphs (stage 2b) ---------------------------
// Only tensor *shapes* matter for sizing gallocr's arena; the dummy
// weights below are never dereferenced (reserve() only rehearses -- counts
// references and sizes the arena -- it never executes a kernel).

mi_tensor* make_dummy_weight(mi_context* ctx, mi_type type, int64_t ne0, int64_t ne1) {
  int64_t ne[2] = {ne0, ne1};
  mi_tensor* t = mi_context_new_tensor_from_data(ctx, type, (ne1 > 1 ? 2 : 1), ne, nullptr);
  t->flags = MI_TENSOR_FLAG_WEIGHT;
  return t;
}

// Dummy weights must carry the *real* model's per-tensor types, not a
// hardcoded assumption: reserve_gallocr's rehearsal graphs need to trigger
// exactly the same mi_graph_mul_mat behavior (quantize-node insertion or
// not) and the same mi_nbytes() accounting the real forward pass will use,
// or the arena reserved here could be sized for the wrong graph shape
// entirely -- see docs/stage4.md for the concrete failure this would
// otherwise cause. `real` supplies one real layer's weights purely to read
// ->type off of; its data pointers (real mmap'd weights) are never
// touched, only its types and (already-hparams-derived) shapes.
mi_qwen2_layer_weights make_dummy_layer_weights(mi_context* ctx, const mi_qwen2_hparams& hp,
                                                 const mi_qwen2_layer_weights& real) {
  mi_qwen2_layer_weights lw{};
  int64_t qkv_out = (int64_t)hp.n_head * hp.head_dim;
  int64_t kv_out = (int64_t)hp.n_head_kv * hp.head_dim;
  lw.attn_norm = make_dummy_weight(ctx, real.attn_norm->type, hp.n_embd, 1);
  lw.attn_q_w = make_dummy_weight(ctx, real.attn_q_w->type, hp.n_embd, qkv_out);
  lw.attn_q_b = make_dummy_weight(ctx, real.attn_q_b->type, qkv_out, 1);
  lw.attn_k_w = make_dummy_weight(ctx, real.attn_k_w->type, hp.n_embd, kv_out);
  lw.attn_k_b = make_dummy_weight(ctx, real.attn_k_b->type, kv_out, 1);
  lw.attn_v_w = make_dummy_weight(ctx, real.attn_v_w->type, hp.n_embd, kv_out);
  lw.attn_v_b = make_dummy_weight(ctx, real.attn_v_b->type, kv_out, 1);
  lw.attn_output_w = make_dummy_weight(ctx, real.attn_output_w->type, qkv_out, hp.n_embd);
  lw.ffn_norm = make_dummy_weight(ctx, real.ffn_norm->type, hp.n_embd, 1);
  lw.ffn_gate_w = make_dummy_weight(ctx, real.ffn_gate_w->type, hp.n_embd, hp.n_ff);
  lw.ffn_up_w = make_dummy_weight(ctx, real.ffn_up_w->type, hp.n_embd, hp.n_ff);
  lw.ffn_down_w = make_dummy_weight(ctx, real.ffn_down_w->type, hp.n_ff, hp.n_embd);
  return lw;
}

// Builds the two worst-case graphs SPEC.md's "reserve 与规划分离" calls
// for (prefill at n_tokens = n_ctx, decode at n_past = n_ctx - 1; both
// share n_kv_total = n_ctx, the true worst case for a context of this
// size) against the *real* KV-cache tensors (already allocated, just
// above) but throwaway dummy weights, rehearses both, and reserves
// kv->galloc's arena to the max -- exactly once, at kv_cache_init time.
void reserve_gallocr(mi_kv_cache* kv, const mi_qwen2_hparams& hp,
                      const mi_qwen2_weights& real_weights) {
  int node_capacity = (int)estimate_node_capacity(hp);

  mi_context_params wcp{};
  wcp.tensor_capacity = (size_t)hp.n_layer * 16 + 16;
  wcp.data_capacity = 0;
  wcp.no_alloc = true;
  mi_context* wctx = mi_context_new(wcp);

  mi_qwen2_weights weights{};
  weights.token_embd = make_dummy_weight(wctx, real_weights.token_embd->type, hp.n_embd, hp.n_vocab);
  weights.output_norm = make_dummy_weight(wctx, real_weights.output_norm->type, hp.n_embd, 1);
  // Tied embedding (real_weights.output == real_weights.token_embd) or a
  // real, separately-typed output.weight: mirror whichever the real model
  // actually has, same reasoning as every other dummy weight here.
  weights.output = (real_weights.output == real_weights.token_embd)
                        ? weights.token_embd
                        : make_dummy_weight(wctx, real_weights.output->type, hp.n_embd, hp.n_vocab);
  weights.layers =
      (mi_qwen2_layer_weights*)std::malloc(sizeof(mi_qwen2_layer_weights) * (size_t)hp.n_layer);
  for (int l = 0; l < hp.n_layer; l++) {
    weights.layers[l] = make_dummy_layer_weights(wctx, hp, real_weights.layers[l]);
  }

  auto make_planning_ctx = [&]() {
    mi_context_params pcp{};
    pcp.tensor_capacity = (size_t)node_capacity;
    pcp.data_capacity = sizeof(mi_cgraph) + 3 * (size_t)node_capacity * sizeof(mi_tensor*) +
                         estimate_build_graph_bookkeeping_bytes(hp) + 4096;
    pcp.no_alloc = true;
    return mi_context_new(pcp);
  };
  mi_context* ctx_prefill = make_planning_ctx();
  mi_context* ctx_decode = make_planning_ctx();

  int64_t ne_prefill[1] = {kv->n_ctx};
  mi_tensor* ids_prefill =
      mi_context_new_tensor_from_data(ctx_prefill, MI_TYPE_I32, 1, ne_prefill, nullptr);
  ids_prefill->flags = MI_TENSOR_FLAG_INPUT;
  // last_logits_only=false: reserve()'s job is to size the arena for the
  // worst case, and a real caller may still ask for full per-position
  // logits (e.g. --dump's teacher-forcing path) -- the smaller graph a
  // last_logits_only=true call would build must never be what determines
  // the reserved capacity.
  built_graph bg_prefill = build_model_graph(ctx_prefill, hp, weights, kv->k, kv->v, ids_prefill,
                                              /*n_past=*/0, /*last_logits_only=*/false);
  bg_prefill.logits->flags = MI_TENSOR_FLAG_OUTPUT;  // data stays null: never read during rehearsal
  mi_cgraph* g_prefill = schedule_graph(ctx_prefill, node_capacity, bg_prefill);

  int64_t ne_decode[1] = {1};
  mi_tensor* ids_decode =
      mi_context_new_tensor_from_data(ctx_decode, MI_TYPE_I32, 1, ne_decode, nullptr);
  ids_decode->flags = MI_TENSOR_FLAG_INPUT;
  built_graph bg_decode = build_model_graph(ctx_decode, hp, weights, kv->k, kv->v, ids_decode,
                                             /*n_past=*/kv->n_ctx - 1, /*last_logits_only=*/false);
  bg_decode.logits->flags = MI_TENSOR_FLAG_OUTPUT;
  mi_cgraph* g_decode = schedule_graph(ctx_decode, node_capacity, bg_decode);

  mi_cgraph* graphs[2] = {g_prefill, g_decode};
  mi_gallocr_reserve(kv->galloc, graphs, 2);

  std::free(weights.layers);
  mi_context_free(ctx_prefill);
  mi_context_free(ctx_decode);
  mi_context_free(wctx);
}

}  // namespace

bool mi_qwen2_hparams_load(const mi_gguf_context* gguf, mi_qwen2_hparams* out, std::string* err) {
  uint32_t n_embd = 0, n_layer = 0, n_head = 0, n_head_kv = 0, n_ff = 0, n_ctx_train = 0;
  float rope_freq_base = 0, rms_eps = 0;
  if (!gguf_need_u32(gguf, "qwen2.embedding_length", &n_embd, err)) return false;
  if (!gguf_need_u32(gguf, "qwen2.block_count", &n_layer, err)) return false;
  if (!gguf_need_u32(gguf, "qwen2.attention.head_count", &n_head, err)) return false;
  if (!gguf_need_u32(gguf, "qwen2.attention.head_count_kv", &n_head_kv, err)) return false;
  if (!gguf_need_u32(gguf, "qwen2.feed_forward_length", &n_ff, err)) return false;
  if (!gguf_need_u32(gguf, "qwen2.context_length", &n_ctx_train, err)) return false;
  if (!gguf_need_f32(gguf, "qwen2.rope.freq_base", &rope_freq_base, err)) return false;
  if (!gguf_need_f32(gguf, "qwen2.attention.layer_norm_rms_epsilon", &rms_eps, err)) return false;

  if (n_head == 0 || n_embd % n_head != 0) {
    if (err) *err = "qwen2 hparams: embedding_length is not divisible by head_count";
    return false;
  }
  if (n_head_kv == 0 || n_head % n_head_kv != 0) {
    if (err) *err = "qwen2 hparams: head_count is not a multiple of head_count_kv";
    return false;
  }

  const mi_gguf_tensor_info* tok_embd = mi_gguf_find_tensor(gguf, "token_embd.weight");
  if (!tok_embd) {
    if (err) *err = "qwen2 hparams: missing tensor token_embd.weight";
    return false;
  }

  out->n_embd = (int)n_embd;
  out->n_layer = (int)n_layer;
  out->n_head = (int)n_head;
  out->n_head_kv = (int)n_head_kv;
  out->n_ff = (int)n_ff;
  out->head_dim = (int)(n_embd / n_head);
  out->rms_eps = rms_eps;
  out->rope_freq_base = rope_freq_base;
  out->n_ctx_train = (int)n_ctx_train;
  out->n_vocab = (int)tok_embd->ne[1];
  return true;
}

mi_qwen2_model* mi_qwen2_model_load(const char* gguf_path, std::string* err) {
  std::string local_err;
  mi_gguf_context* gguf = mi_gguf_open(gguf_path, &local_err);
  if (!gguf) {
    if (err) *err = local_err;
    return nullptr;
  }

  mi_qwen2_hparams hparams{};
  if (!mi_qwen2_hparams_load(gguf, &hparams, &local_err)) {
    mi_gguf_close(gguf);
    if (err) *err = local_err;
    return nullptr;
  }

  mi_vocab* vocab = mi_vocab_load(gguf, &local_err);
  if (!vocab) {
    mi_gguf_close(gguf);
    if (err) *err = local_err;
    return nullptr;
  }

  mi_context_params cp{};
  cp.tensor_capacity = (size_t)hparams.n_layer * 16 + 16;
  cp.data_capacity = 0;
  cp.no_alloc = true;
  mi_context* wctx = mi_context_new(cp);

  auto* model = new mi_qwen2_model();
  model->gguf = gguf;
  model->vocab = vocab;
  model->hparams = hparams;
  model->weight_ctx = wctx;
  model->weights.layers =
      (mi_qwen2_layer_weights*)std::calloc((size_t)hparams.n_layer, sizeof(mi_qwen2_layer_weights));

  bool ok = true;
  model->weights.token_embd = load_weight(wctx, gguf, "token_embd.weight", &local_err);
  ok = ok && model->weights.token_embd;
  model->weights.output_norm = load_weight(wctx, gguf, "output_norm.weight", &local_err);
  ok = ok && model->weights.output_norm;
  if (ok) {
    if (mi_gguf_find_tensor(gguf, "output.weight")) {
      model->weights.output = load_weight(wctx, gguf, "output.weight", &local_err);
      ok = ok && model->weights.output;
    } else {
      model->weights.output = model->weights.token_embd;  // tied embedding/lm_head (SPEC.md)
    }
  }

  for (int l = 0; ok && l < hparams.n_layer; l++) {
    char name[128];
    auto ld = [&](const char* suffix) {
      std::snprintf(name, sizeof(name), "blk.%d.%s", l, suffix);
      return load_weight(wctx, gguf, name, &local_err);
    };
    mi_qwen2_layer_weights& lw = model->weights.layers[l];
    lw.attn_norm = ld("attn_norm.weight");
    lw.attn_q_w = ld("attn_q.weight");
    lw.attn_q_b = ld("attn_q.bias");
    lw.attn_k_w = ld("attn_k.weight");
    lw.attn_k_b = ld("attn_k.bias");
    lw.attn_v_w = ld("attn_v.weight");
    lw.attn_v_b = ld("attn_v.bias");
    lw.attn_output_w = ld("attn_output.weight");
    lw.ffn_norm = ld("ffn_norm.weight");
    lw.ffn_gate_w = ld("ffn_gate.weight");
    lw.ffn_up_w = ld("ffn_up.weight");
    lw.ffn_down_w = ld("ffn_down.weight");
    ok = lw.attn_norm && lw.attn_q_w && lw.attn_q_b && lw.attn_k_w && lw.attn_k_b && lw.attn_v_w &&
         lw.attn_v_b && lw.attn_output_w && lw.ffn_norm && lw.ffn_gate_w && lw.ffn_up_w &&
         lw.ffn_down_w;
  }

  if (!ok) {
    if (err) *err = local_err;
    mi_qwen2_model_free(model);
    return nullptr;
  }
  return model;
}

void mi_qwen2_model_free(mi_qwen2_model* model) {
  if (!model) return;
  std::free(model->weights.layers);
  mi_context_free(model->weight_ctx);
  mi_vocab_free(model->vocab);
  mi_gguf_close(model->gguf);
  delete model;
}

mi_kv_cache* mi_kv_cache_init(const mi_qwen2_hparams& hparams, const mi_qwen2_weights& weights,
                               int n_ctx, const mi_threadpool_params& tp_params) {
  auto* kv = new mi_kv_cache();
  kv->n_layer = hparams.n_layer;
  kv->n_ctx = n_ctx;
  kv->n_past = 0;

  size_t per_tensor_bytes = (size_t)hparams.head_dim * hparams.n_head_kv * n_ctx * sizeof(float);
  mi_context_params cp{};
  cp.tensor_capacity = (size_t)hparams.n_layer * 2 + 4;
  cp.data_capacity = per_tensor_bytes * 2 * hparams.n_layer;
  cp.no_alloc = false;
  kv->ctx = mi_context_new(cp);

  kv->k = (mi_tensor**)std::malloc(sizeof(mi_tensor*) * hparams.n_layer);
  kv->v = (mi_tensor**)std::malloc(sizeof(mi_tensor*) * hparams.n_layer);
  for (int l = 0; l < hparams.n_layer; l++) {
    kv->k[l] = mi_new_tensor_3d(kv->ctx, MI_TYPE_F32, hparams.head_dim, hparams.n_head_kv, n_ctx);
    kv->k[l]->flags = MI_TENSOR_FLAG_KV;
    kv->v[l] = mi_new_tensor_3d(kv->ctx, MI_TYPE_F32, hparams.head_dim, hparams.n_head_kv, n_ctx);
    kv->v[l]->flags = MI_TENSOR_FLAG_KV;
  }

  // Persistent, reused-every-step graph-building context -- see
  // model_qwen2.h's mi_kv_cache doc comment for why this can't be
  // malloc'd/freed per forward call the way stage 1/2a's scratch context
  // was.
  mi_context_params gcp{};
  gcp.tensor_capacity = estimate_node_capacity(hparams);
  gcp.data_capacity = estimate_graph_ctx_bytes(hparams, n_ctx);
  gcp.no_alloc = true;
  kv->graph_ctx = mi_context_new(gcp);

  kv->galloc = mi_gallocr_new();
  reserve_gallocr(kv, hparams, weights);

  kv->pool = mi_threadpool_new(tp_params);

  return kv;
}

void mi_kv_cache_free(mi_kv_cache* kv) {
  if (!kv) return;
  mi_threadpool_free(kv->pool);
  mi_gallocr_free(kv->galloc);
  mi_context_free(kv->graph_ctx);
  std::free(kv->k);
  std::free(kv->v);
  mi_context_free(kv->ctx);
  delete kv;
}

void mi_qwen2_forward(mi_qwen2_model* model, mi_kv_cache* kv, const mi_qwen2_forward_params& params,
                       float* logits_out) {
  const mi_qwen2_hparams& hp = model->hparams;
  int64_t n_tokens = params.n_tokens;

  // Reused every step, never malloc'd/freed here -- see mi_kv_cache's doc
  // comment. Resetting just rewinds two bump pointers, no heap traffic.
  mi_context_reset(kv->graph_ctx);
  mi_context* ctx = kv->graph_ctx;

  mi_tensor* ids = mi_context_new_tensor_eager(ctx, MI_TYPE_I32, 1, &n_tokens);
  ids->flags = MI_TENSOR_FLAG_INPUT;  // persistent: gallocr must never touch it
  std::memcpy(ids->data, params.tokens, sizeof(int32_t) * (size_t)n_tokens);

  // Build phase: every mi_graph_* call below only creates tensor metadata
  // (op/src/shape) -- no kernel runs yet, and (stage 2b) no tensor data is
  // assigned yet either except for persistent/eager tensors. n_past/
  // n_tokens (hence every shape and every softmax/RoPE op_param) are baked
  // in fresh here every call, which is why the graph is rebuilt from
  // scratch every forward step rather than reshaped in place (see
  // docs/architecture.md).
  built_graph bg = build_model_graph(ctx, hp, model->weights, kv->k, kv->v, ids, params.n_past,
                                      params.last_logits_only);

  // `logits` is always eagerly allocated and flagged OUTPUT: it's read via
  // the memcpy below on *every* call, dump or not, so it must never be
  // gallocr-managed (SPEC.md's persistent-tensor list explicitly includes
  // "输出"). The other checkpoints (embd/layer_i/output_norm) are only
  // special when dumping, so they stay ordinary gallocr-managed tensors
  // and get read out via the interleaved execution loop below instead of
  // needing the same eager treatment -- see docs/stage2b.md.
  bg.logits->data = mi_context_alloc_bytes(ctx, mi_nbytes(bg.logits));
  bg.logits->flags = MI_TENSOR_FLAG_OUTPUT;

  mi_cgraph* g = schedule_graph(ctx, (int)estimate_node_capacity(hp), bg);
  if (params.out_graph) *params.out_graph = g;
  mi_gallocr_alloc_graph(kv->galloc, g);

  if (params.dump) {
    // Interleaved with execution, not batched after: gallocr is free to
    // reuse embd's/layer_i's buffer later in this same graph (their
    // natural refcount allows it, same as in a non-dump call), so each
    // checkpoint must be copied out by the dump sink immediately after
    // it's computed, before anything downstream can overwrite it.
    for (int i = 0; i < g->n_nodes; i++) {
      mi_tensor* node = g->nodes[i];
      mi_compute_forward(node, 0, 1);
      if (node == bg.embd) {
        params.dump->on_tensor(params.dump->user_data, "embd", node);
      }
      for (int l = 0; l < hp.n_layer; l++) {
        if (node == bg.layer_outputs[(size_t)l]) {
          char name[32];
          std::snprintf(name, sizeof(name), "layer_%d", l);
          params.dump->on_tensor(params.dump->user_data, name, node);
          break;
        }
      }
      if (node == bg.final_normed) {
        params.dump->on_tensor(params.dump->user_data, "output_norm", node);
      }
    }
    params.dump->on_tensor(params.dump->user_data, "logits", bg.logits);
  } else {
    // Stage 3: real threadpool. kv->pool always exists (defaults to
    // n_threads=1 if the caller didn't ask for more), so this is the only
    // dispatch path for the non-dump forward -- see mi_kv_cache's doc
    // comment for why a 1-thread pool reproduces stage 2b's behavior
    // exactly rather than needing a separate pool-less fallback.
    mi_graph_compute(g, kv->pool);
  }

  // bg.logits->ne[1] is n_tokens normally, or 1 when params.last_logits_only
  // restricted the lm_head matmul to the last position -- copy exactly what
  // was actually computed, not what the caller's n_tokens might suggest.
  // Callers that set last_logits_only must size logits_out for n_vocab
  // floats, not n_tokens * n_vocab (see mi_qwen2_forward_params's doc
  // comment).
  std::memcpy(logits_out, bg.logits->data, sizeof(float) * (size_t)hp.n_vocab * (size_t)bg.logits->ne[1]);
}

int32_t mi_qwen2_sample_greedy(const float* logits, int n_vocab) {
  int32_t best = 0;
  float best_val = logits[0];
  for (int i = 1; i < n_vocab; i++) {
    if (logits[i] > best_val) {
      best_val = logits[i];
      best = i;
    }
  }
  return best;
}
