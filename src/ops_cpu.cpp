#include "ops_cpu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

#include "quant.h"

namespace {

// mul_mat parallel-mode config (see ops_cpu.h). Written once at
// mi_threadpool_new time and read on every mi_compute_mul_mat call;
// mi_mulmat_next_row is the only piece that mutates per-node, reset by
// threadpool.cpp's barrier immediately before releasing threads into the
// next mul_mat node -- see that reset call's doc comment for the
// happens-before argument. Plain relaxed ops suffice everywhere here: mode
// and chunk never race with a concurrent mi_compute_mul_mat call (they're
// fixed before any worker thread is spawned), and the counter only needs
// fetch_add's atomicity (any partition of [0, n) across claims is
// correct) -- the *visibility* of the reset is already established by the
// barrier's own release/acquire on a separate atomic.
std::atomic<int> g_mulmat_mode{MI_MULMAT_STATIC};
std::atomic<int> g_mulmat_chunk{8};
std::atomic<int64_t> g_mulmat_next_row{0};

// Every kernel indexes through ne[]/nb[] via this one helper rather than
// assuming any operand is contiguous. That's what lets the exact same
// mul_mat implementation serve both a plain weight matmul (genuinely
// contiguous) and the attention-internal matmuls, where src0 is a strided
// view into Q/K/V or a *transposed* view of the KV cache (see
// model_qwen2.cpp's per-head attention loop) -- there is no separate
// "fast contiguous path" to keep in sync with a "slow strided path" bug for
// bug, because there is only one path. A real cost, paid deliberately in
// stage 1: this is scalar and doesn't vectorize past the compiler's own
// auto-vectorization. Stage 4's SIMD dot products are added as a
// contiguity-checked fast path in front of this, not a replacement for it.
inline float* mi_f32_ptr(const mi_tensor* t, int64_t i0, int64_t i1 = 0, int64_t i2 = 0,
                          int64_t i3 = 0) {
  return (float*)((char*)t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
}

// Static (even, remainder-to-the-front) split of [0, n) into nth pieces;
// piece `ith` is [start, end). With nth == 1 (stage 1, and stage 2 before
// the threadpool exists) this is always the full range -- see ops_cpu.h.
void mi_split_range(int64_t n, int ith, int nth, int64_t* start, int64_t* end) {
  int64_t per = n / nth;
  int64_t rem = n % nth;
  int64_t s = ith * per + (ith < rem ? ith : rem);
  int64_t e = s + per + (ith < rem ? 1 : 0);
  *start = s;
  *end = e;
}

template <typename F>
void ew_binop(mi_tensor* dst, F op, int ith, int nth) {
  const mi_tensor* a = dst->src[0];
  const mi_tensor* b = dst->src[1];
  int64_t ne0 = a->ne[0], ne1 = a->ne[1], ne2 = a->ne[2], ne3 = a->ne[3];
  int64_t n_rows = ne1 * ne2 * ne3;
  int64_t r0, r1;
  mi_split_range(n_rows, ith, nth, &r0, &r1);
  for (int64_t r = r0; r < r1; r++) {
    int64_t i1 = r % ne1;
    int64_t i2 = (r / ne1) % ne2;
    int64_t i3 = r / (ne1 * ne2);
    // b broadcasts along any dim where it has size 1 -- covers both a
    // same-shape elementwise op (b->ne[k] == a->ne[k] for all k) and a
    // "row vector" broadcast (rms_norm's weight, QKV bias: ne0 matches,
    // ne1..ne3 are 1).
    int64_t b1 = (b->ne[1] > 1) ? i1 : 0;
    int64_t b2 = (b->ne[2] > 1) ? i2 : 0;
    int64_t b3 = (b->ne[3] > 1) ? i3 : 0;
    for (int64_t i = 0; i < ne0; i++) {
      float av = *mi_f32_ptr(a, i, i1, i2, i3);
      float bv = *mi_f32_ptr(b, i, b1, b2, b3);
      *mi_f32_ptr(dst, i, i1, i2, i3) = op(av, bv);
    }
  }
}

}  // namespace

void mi_op_params_set_f32(mi_tensor* t, int slot, float v) {
  std::memcpy(&t->op_params[slot], &v, sizeof(float));
}
float mi_op_params_get_f32(const mi_tensor* t, int slot) {
  float v;
  std::memcpy(&v, &t->op_params[slot], sizeof(float));
  return v;
}
void mi_op_params_set_i32(mi_tensor* t, int slot, int32_t v) {
  std::memcpy(&t->op_params[slot], &v, sizeof(int32_t));
}
int32_t mi_op_params_get_i32(const mi_tensor* t, int slot) {
  int32_t v;
  std::memcpy(&v, &t->op_params[slot], sizeof(int32_t));
  return v;
}

void mi_mulmat_set_mode(mi_mulmat_mode mode, int dynamic_chunk) {
  g_mulmat_mode.store((int)mode, std::memory_order_relaxed);
  g_mulmat_chunk.store(dynamic_chunk > 0 ? dynamic_chunk : 1, std::memory_order_relaxed);
}

void mi_mulmat_reset_dynamic_counter() {
  g_mulmat_next_row.store(0, std::memory_order_relaxed);
}

void mi_compute_get_rows(mi_tensor* dst, int ith, int nth) {
  const mi_tensor* src0 = dst->src[0];  // [ne0, n_rows], F32/F16/Q8_0
  const mi_tensor* src1 = dst->src[1];  // I32 [n_ids]
  int64_t ne0 = src0->ne[0];
  int64_t n_ids = src1->ne[0];
  int64_t j0, j1;
  mi_split_range(n_ids, ith, nth, &j0, &j1);
  for (int64_t j = j0; j < j1; j++) {
    int32_t id = *(const int32_t*)((const char*)src1->data + j * src1->nb[0]);
    if (src0->type == MI_TYPE_F32) {
      // Unchanged from stage 1: kept as its own branch (rather than folded
      // into the generic byte-row dispatch below) so this exact code path
      // -- and every bit-exactness test built on it -- never moves.
      for (int64_t i = 0; i < ne0; i++) {
        *mi_f32_ptr(dst, i, j) = *mi_f32_ptr(src0, i, id);
      }
    } else {
      // token_embd loaded from an F16/Q8_0 GGUF: dequantize the looked-up
      // row into dst's F32 storage (SPEC.md: "token_embd 为 Q8_0 时按行
      // 反量化"). src0 is always a plain contiguous weight leaf here, never
      // a view, so byte offset id * nb[1] is exactly row `id`'s start.
      const char* src_row = (const char*)src0->data + id * src0->nb[1];
      float* dst_row = mi_f32_ptr(dst, 0, j);
      if (src0->type == MI_TYPE_F16) {
        const uint16_t* h = (const uint16_t*)src_row;
        for (int64_t i = 0; i < ne0; i++) dst_row[i] = mi_fp16_to_fp32(h[i]);
      } else {  // MI_TYPE_Q8_0
        mi_dequantize_row_q8_0((const block_q8_0*)src_row, dst_row, ne0);
      }
    }
  }
}

void mi_compute_rms_norm(mi_tensor* dst, int ith, int nth) {
  const mi_tensor* src0 = dst->src[0];
  float eps = mi_op_params_get_f32(dst, 0);
  int64_t ne0 = src0->ne[0];
  int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
  int64_t r0, r1;
  mi_split_range(n_rows, ith, nth, &r0, &r1);
  for (int64_t r = r0; r < r1; r++) {
    int64_t i1 = r % src0->ne[1];
    int64_t i2 = (r / src0->ne[1]) % src0->ne[2];
    int64_t i3 = r / (src0->ne[1] * src0->ne[2]);
    float sumsq = 0.0f;
    for (int64_t i = 0; i < ne0; i++) {
      float v = *mi_f32_ptr(src0, i, i1, i2, i3);
      sumsq += v * v;
    }
    float mean = sumsq / (float)ne0;
    float scale = 1.0f / std::sqrt(mean + eps);
    for (int64_t i = 0; i < ne0; i++) {
      *mi_f32_ptr(dst, i, i1, i2, i3) = (*mi_f32_ptr(src0, i, i1, i2, i3)) * scale;
    }
  }
}

void mi_compute_mul(mi_tensor* dst, int ith, int nth) {
  ew_binop(
      dst, [](float x, float y) { return x * y; }, ith, nth);
}

void mi_compute_add(mi_tensor* dst, int ith, int nth) {
  ew_binop(
      dst, [](float x, float y) { return x + y; }, ith, nth);
}

void mi_compute_mul_mat(mi_tensor* dst, int ith, int nth) {
  const mi_tensor* a = dst->src[0];  // [k, n]
  const mi_tensor* b = dst->src[1];  // [k, m]
  int64_t k = a->ne[0];
  int64_t n = a->ne[1];
  int64_t m = b->ne[1];

  // Row i's dot products always run in this exact order regardless of
  // *which* thread computes it or how row assignment was decided -- rows
  // are fully independent (dst[i, :] never reads any other row), so
  // switching mi_mulmat_mode or the thread count can only change which
  // thread writes a given row, never the floating-point operation order
  // within it. That's what makes stage 3's "same output under every mode/
  // thread-count" requirement automatic rather than something to prove per
  // mode -- see docs/stage3.md. Stage 4 adds a second axis of dispatch,
  // on a->type (the weight's actual GGUF type), orthogonal to the mode
  // dispatch below: every mode still just decides *which* rows `i` this
  // thread handles, compute_row decides *how* a row is computed.
  //   - F32: unchanged from stage 1 -- kept byte-for-byte identical so
  //     every prior bit-exactness test keeps passing without modification.
  //   - F16: a is a plain contiguous weight row (never a view); b is F32
  //     and, in this project, always a freshly graph-allocated contiguous
  //     tensor when a is a real weight, so mi_f32_ptr(b, 0, j) is a valid
  //     start-of-row pointer for a straight-line vec_dot.
  //   - Q8_0: b has *already* been quantized to Q8_0 by an upstream
  //     MI_OP_QUANTIZE_Q8_0 node that mi_graph_mul_mat inserts whenever
  //     a->type == MI_TYPE_Q8_0 (see graph.cpp) -- this kernel never
  //     quantizes on the fly itself, so there is no scratch buffer to
  //     manage here at all.
  auto compute_row = [&](int64_t i) {
    switch (a->type) {
      case MI_TYPE_F32: {
        for (int64_t j = 0; j < m; j++) {
          float acc = 0.0f;
          for (int64_t t = 0; t < k; t++) {
            acc += (*mi_f32_ptr(a, t, i)) * (*mi_f32_ptr(b, t, j));
          }
          *mi_f32_ptr(dst, i, j) = acc;
        }
        break;
      }
      case MI_TYPE_F16: {
        const uint16_t* arow = (const uint16_t*)((const char*)a->data + i * a->nb[1]);
        for (int64_t j = 0; j < m; j++) {
          const float* brow = mi_f32_ptr(b, 0, j);
          *mi_f32_ptr(dst, i, j) = mi_vec_dot_f16_f32(arow, brow, k);
        }
        break;
      }
      case MI_TYPE_Q8_0: {
        // b is MI_TYPE_Q8_0_ACT (float scale), not MI_TYPE_Q8_0 -- see
        // graph.cpp's mi_graph_quantize_q8_0 (stage 6) -- so it's read as
        // block_q8_0_act, not block_q8_0, and dotted via
        // mi_vec_dot_q8_0_wact, not mi_vec_dot_q8_0_q8_0.
        const block_q8_0* arow = (const block_q8_0*)((const char*)a->data + i * a->nb[1]);
        for (int64_t j = 0; j < m; j++) {
          const block_q8_0_act* brow = (const block_q8_0_act*)((const char*)b->data + j * b->nb[1]);
          *mi_f32_ptr(dst, i, j) = mi_vec_dot_q8_0_wact(arow, brow, k);
        }
        break;
      }
      default:
        break;  // I32 is never a mul_mat operand
    }
  };

  mi_mulmat_mode mode = (mi_mulmat_mode)g_mulmat_mode.load(std::memory_order_relaxed);
  switch (mode) {
    case MI_MULMAT_SINGLE:
      if (ith == 0) {
        for (int64_t i = 0; i < n; i++) compute_row(i);
      }
      return;
    case MI_MULMAT_DYNAMIC: {
      int64_t chunk = g_mulmat_chunk.load(std::memory_order_relaxed);
      for (;;) {
        int64_t start = g_mulmat_next_row.fetch_add(chunk, std::memory_order_relaxed);
        if (start >= n) break;
        int64_t end = std::min(start + chunk, n);
        for (int64_t i = start; i < end; i++) compute_row(i);
      }
      return;
    }
    case MI_MULMAT_STATIC:
    default: {
      int64_t i0, i1;
      mi_split_range(n, ith, nth, &i0, &i1);
      for (int64_t i = i0; i < i1; i++) compute_row(i);
      return;
    }
  }
}

void mi_compute_rope(mi_tensor* dst, int ith, int nth) {
  const mi_tensor* src0 = dst->src[0];  // [head_dim, n_heads, n_tokens]
  int32_t n_past = mi_op_params_get_i32(dst, 0);
  int32_t n_dims = mi_op_params_get_i32(dst, 1);
  float freq_base = mi_op_params_get_f32(dst, 2);
  int64_t head_dim = src0->ne[0];
  int64_t n_heads = src0->ne[1];
  int64_t half = n_dims / 2;

  int64_t n_rows = n_heads * src0->ne[2];
  int64_t r0, r1;
  mi_split_range(n_rows, ith, nth, &r0, &r1);
  for (int64_t r = r0; r < r1; r++) {
    int64_t i1 = r % n_heads;  // head index
    int64_t i2 = r / n_heads;  // token index within this call's batch
    int64_t pos = n_past + i2;
    for (int64_t p = 0; p < half; p++) {
      // theta_p = pos * base^(-2p/n_dims); NEOX pairs element p with
      // p + n_dims/2 (the two halves of the rotary span), not adjacent
      // elements -- see SPEC.md.
      float freq = std::pow(freq_base, -(float)(2 * p) / (float)n_dims);
      float theta = (float)pos * freq;
      float c = std::cos(theta), s = std::sin(theta);
      float x0 = *mi_f32_ptr(src0, p, i1, i2);
      float x1 = *mi_f32_ptr(src0, p + half, i1, i2);
      *mi_f32_ptr(dst, p, i1, i2) = x0 * c - x1 * s;
      *mi_f32_ptr(dst, p + half, i1, i2) = x0 * s + x1 * c;
    }
    for (int64_t i = n_dims; i < head_dim; i++) {
      *mi_f32_ptr(dst, i, i1, i2) = *mi_f32_ptr(src0, i, i1, i2);
    }
  }
}

void mi_compute_softmax(mi_tensor* dst, int ith, int nth) {
  const mi_tensor* src0 = dst->src[0];  // [n_kv, n_tokens] or [n_kv, n_tokens, n_head]
  int32_t n_past = mi_op_params_get_i32(dst, 0);
  float scale = mi_op_params_get_f32(dst, 1);
  int64_t n_kv = src0->ne[0];
  int64_t n_tokens = src0->ne[1];
  int64_t n_head = src0->ne[2];  // 1 for the plain 2D case (ne[]-padding)
  // Stage 6: every (token, head) pair is an independent row -- the causal
  // mask depends only on the token index j, same for every head, so
  // batching heads in changes nothing about any individual row's
  // computation, only how many rows there are to split across threads.
  int64_t n_rows = n_tokens * n_head;
  int64_t r0, r1;
  mi_split_range(n_rows, ith, nth, &r0, &r1);
  for (int64_t r = r0; r < r1; r++) {
    int64_t j = r % n_tokens;
    int64_t h = r / n_tokens;
    // Row j (0-based in this call's batch) is absolute position n_past+j
    // and may attend to keys [0, n_past+j].
    int64_t allowed = n_past + j + 1;
    if (allowed > n_kv) allowed = n_kv;

    float maxv = -INFINITY;
    for (int64_t i = 0; i < allowed; i++) {
      float v = (*mi_f32_ptr(src0, i, j, h)) * scale;
      if (v > maxv) maxv = v;
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < allowed; i++) {
      float v = (*mi_f32_ptr(src0, i, j, h)) * scale;
      float e = std::exp(v - maxv);
      *mi_f32_ptr(dst, i, j, h) = e;
      sum += e;
    }
    float inv_sum = 1.0f / sum;
    for (int64_t i = 0; i < allowed; i++) {
      *mi_f32_ptr(dst, i, j, h) *= inv_sum;
    }
    for (int64_t i = allowed; i < n_kv; i++) {
      *mi_f32_ptr(dst, i, j, h) = 0.0f;
    }
  }
}

// Stage 6 step 3b: see ops_cpu.h's doc comment. Parallelized over (head,
// kv-row) pairs, same granularity mul_mat's F32 path used per head before
// (one thread per output row), just flattened across all n_head heads into
// one node instead of n_head separate graph nodes -- see docs/stage6.md
// for why this is bit-exact with the old per-head loop (same d-ascending
// accumulation order for every (h,i,j), only *which thread* computes a
// given (h,i) pair changes, which stage 3's mul_mat argument already
// established never affects the result).
void mi_compute_attn_qk(mi_tensor* dst, int ith, int nth) {
  const mi_tensor* q_rot = dst->src[0];    // [head_dim, n_head, n_tokens]
  const mi_tensor* cache_k = dst->src[1];  // [head_dim, n_head_kv, n_ctx]
  int32_t n_rep = mi_op_params_get_i32(dst, 1);
  int64_t head_dim = q_rot->ne[0];
  int64_t n_head = q_rot->ne[1];
  int64_t n_tokens = q_rot->ne[2];
  int64_t n_kv_total = dst->ne[0];

  int64_t n_units = n_head * n_kv_total;
  int64_t u0, u1;
  mi_split_range(n_units, ith, nth, &u0, &u1);
  for (int64_t u = u0; u < u1; u++) {
    int64_t h = u / n_kv_total;
    int64_t i = u % n_kv_total;
    int64_t kvh = h / n_rep;
    for (int64_t j = 0; j < n_tokens; j++) {
      float acc = 0.0f;
      for (int64_t d = 0; d < head_dim; d++) {
        acc += (*mi_f32_ptr(cache_k, d, kvh, i)) * (*mi_f32_ptr(q_rot, d, h, j));
      }
      *mi_f32_ptr(dst, i, j, h) = acc;
    }
  }
}

// Stage 6 step 3b: see ops_cpu.h's doc comment. Parallelized over (head,
// head_dim-row) pairs, same granularity mul_mat's F32 path used per head
// before (one thread per output row, here the output row is a head_dim
// index), flattened across all n_head heads.
void mi_compute_attn_kqv(mi_tensor* dst, int ith, int nth) {
  const mi_tensor* probs = dst->src[0];    // [n_kv_total, n_tokens, n_head]
  const mi_tensor* cache_v = dst->src[1];  // [head_dim, n_head_kv, n_ctx]
  int32_t n_rep = mi_op_params_get_i32(dst, 0);
  int64_t n_kv_total = probs->ne[0];
  int64_t n_tokens = probs->ne[1];
  int64_t n_head = probs->ne[2];
  int64_t head_dim = dst->ne[0];

  int64_t n_units = n_head * head_dim;
  int64_t u0, u1;
  mi_split_range(n_units, ith, nth, &u0, &u1);
  for (int64_t u = u0; u < u1; u++) {
    int64_t h = u / head_dim;
    int64_t o = u % head_dim;
    int64_t kvh = h / n_rep;
    for (int64_t j = 0; j < n_tokens; j++) {
      float acc = 0.0f;
      for (int64_t t = 0; t < n_kv_total; t++) {
        acc += (*mi_f32_ptr(cache_v, o, kvh, t)) * (*mi_f32_ptr(probs, t, j, h));
      }
      *mi_f32_ptr(dst, o, h, j) = acc;
    }
  }
}

void mi_compute_silu(mi_tensor* dst, int ith, int nth) {
  const mi_tensor* src0 = dst->src[0];
  int64_t ne0 = src0->ne[0];
  int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
  int64_t r0, r1;
  mi_split_range(n_rows, ith, nth, &r0, &r1);
  for (int64_t r = r0; r < r1; r++) {
    int64_t i1 = r % src0->ne[1];
    int64_t i2 = (r / src0->ne[1]) % src0->ne[2];
    int64_t i3 = r / (src0->ne[1] * src0->ne[2]);
    for (int64_t i = 0; i < ne0; i++) {
      float x = *mi_f32_ptr(src0, i, i1, i2, i3);
      *mi_f32_ptr(dst, i, i1, i2, i3) = x / (1.0f + std::exp(-x));
    }
  }
}

void mi_compute_scale(mi_tensor* dst, int ith, int nth) {
  const mi_tensor* src0 = dst->src[0];
  float s = mi_op_params_get_f32(dst, 0);
  int64_t ne0 = src0->ne[0];
  int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
  int64_t r0, r1;
  mi_split_range(n_rows, ith, nth, &r0, &r1);
  for (int64_t r = r0; r < r1; r++) {
    int64_t i1 = r % src0->ne[1];
    int64_t i2 = (r / src0->ne[1]) % src0->ne[2];
    int64_t i3 = r / (src0->ne[1] * src0->ne[2]);
    for (int64_t i = 0; i < ne0; i++) {
      *mi_f32_ptr(dst, i, i1, i2, i3) = (*mi_f32_ptr(src0, i, i1, i2, i3)) * s;
    }
  }
}

void mi_compute_cpy(mi_tensor* dst, int ith, int nth) {
  const mi_tensor* src0 = dst->src[0];
  int64_t ne0 = src0->ne[0];
  int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
  int64_t r0, r1;
  mi_split_range(n_rows, ith, nth, &r0, &r1);
  for (int64_t r = r0; r < r1; r++) {
    int64_t i1 = r % src0->ne[1];
    int64_t i2 = (r / src0->ne[1]) % src0->ne[2];
    int64_t i3 = r / (src0->ne[1] * src0->ne[2]);
    for (int64_t i = 0; i < ne0; i++) {
      *mi_f32_ptr(dst, i, i1, i2, i3) = *mi_f32_ptr(src0, i, i1, i2, i3);
    }
  }
}

void mi_compute_quantize_q8_0(mi_tensor* dst, int ith, int nth) {
  const mi_tensor* src0 = dst->src[0];
  int64_t ne0 = src0->ne[0];
  int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
  int64_t r0, r1;
  mi_split_range(n_rows, ith, nth, &r0, &r1);
  for (int64_t r = r0; r < r1; r++) {
    int64_t i1 = r % src0->ne[1];
    int64_t i2 = (r / src0->ne[1]) % src0->ne[2];
    int64_t i3 = r / (src0->ne[1] * src0->ne[2]);
    const float* src_row = mi_f32_ptr(src0, 0, i1, i2, i3);
    // dst is MI_TYPE_Q8_0_ACT (float scale) -- see graph.cpp's
    // mi_graph_quantize_q8_0 (stage 6).
    block_q8_0_act* dst_row =
        (block_q8_0_act*)((char*)dst->data + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
    mi_quantize_row_q8_0_act(src_row, dst_row, ne0);
  }
}

void mi_compute_forward(mi_tensor* dst, int ith, int nth) {
  switch (dst->op) {
    case MI_OP_GET_ROWS:
      mi_compute_get_rows(dst, ith, nth);
      break;
    case MI_OP_RMS_NORM:
      mi_compute_rms_norm(dst, ith, nth);
      break;
    case MI_OP_MUL:
      mi_compute_mul(dst, ith, nth);
      break;
    case MI_OP_ADD:
      mi_compute_add(dst, ith, nth);
      break;
    case MI_OP_MUL_MAT:
      mi_compute_mul_mat(dst, ith, nth);
      break;
    case MI_OP_ROPE:
      mi_compute_rope(dst, ith, nth);
      break;
    case MI_OP_SOFTMAX:
      mi_compute_softmax(dst, ith, nth);
      break;
    case MI_OP_SILU:
      mi_compute_silu(dst, ith, nth);
      break;
    case MI_OP_SCALE:
      mi_compute_scale(dst, ith, nth);
      break;
    case MI_OP_CPY:
      mi_compute_cpy(dst, ith, nth);
      break;
    case MI_OP_QUANTIZE_Q8_0:
      mi_compute_quantize_q8_0(dst, ith, nth);
      break;
    case MI_OP_ATTN_QK:
      mi_compute_attn_qk(dst, ith, nth);
      break;
    case MI_OP_ATTN_KQV:
      mi_compute_attn_kqv(dst, ith, nth);
      break;
    case MI_OP_NONE:
    case MI_OP_RESHAPE:
    case MI_OP_VIEW:
    case MI_OP_PERMUTE:
    case MI_OP_TRANSPOSE:
    case MI_OP_COUNT:
      break;  // leaves and pure metadata reinterpretations move no data
  }
}
