// Hand-verified unit tests for every kernel in ops_cpu.cpp. Shapes are
// small enough that expected values are computed by hand (or via the same
// textbook formula, independently of the kernel's code path) rather than
// by re-running the kernel under test.
#include <cmath>
#include <initializer_list>

#include "context.h"
#include "ops_cpu.h"
#include "test_util.h"

namespace {

mi_context* g_ctx = nullptr;

void set_data(mi_tensor* t, std::initializer_list<float> vals) {
  float* p = (float*)t->data;
  int i = 0;
  for (float v : vals) p[i++] = v;
}

bool close(float a, float b, float tol = 1e-5f) { return std::fabs(a - b) <= tol; }

}  // namespace

static void test_op_params_roundtrip() {
  mi_tensor t{};
  mi_op_params_set_f32(&t, 0, 3.5f);
  mi_op_params_set_i32(&t, 1, -42);
  mi_op_params_set_f32(&t, 2, -1.25f);
  MI_CHECK(mi_op_params_get_f32(&t, 0) == 3.5f);
  MI_CHECK(mi_op_params_get_i32(&t, 1) == -42);
  MI_CHECK(mi_op_params_get_f32(&t, 2) == -1.25f);
}

static void test_get_rows() {
  // table: 3 rows of 2 floats. row0=[1,2] row1=[3,4] row2=[5,6].
  mi_tensor* table = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 3);
  set_data(table, {1, 2, 3, 4, 5, 6});
  mi_tensor* ids = mi_new_tensor_1d(g_ctx, MI_TYPE_I32, 3);
  ((int32_t*)ids->data)[0] = 2;
  ((int32_t*)ids->data)[1] = 0;
  ((int32_t*)ids->data)[2] = 1;
  mi_tensor* dst = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 3);
  dst->op = MI_OP_GET_ROWS;
  dst->src[0] = table;
  dst->src[1] = ids;

  mi_compute_get_rows(dst, 0, 1);
  float* d = (float*)dst->data;
  MI_CHECK(d[0] == 5 && d[1] == 6);  // row 2
  MI_CHECK(d[2] == 1 && d[3] == 2);  // row 0
  MI_CHECK(d[4] == 3 && d[5] == 4);  // row 1

  // Simulate a 2-thread split and check it matches the single-threaded run.
  mi_tensor* dst2 = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 3);
  dst2->op = MI_OP_GET_ROWS;
  dst2->src[0] = table;
  dst2->src[1] = ids;
  mi_compute_get_rows(dst2, 0, 2);
  mi_compute_get_rows(dst2, 1, 2);
  for (int i = 0; i < 6; i++) MI_CHECK(((float*)dst2->data)[i] == d[i]);
}

static void test_rms_norm() {
  mi_tensor* src = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 4, 1);
  set_data(src, {1, 2, 3, 4});
  mi_tensor* dst = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 4, 1);
  dst->op = MI_OP_RMS_NORM;
  dst->src[0] = src;
  float eps = 1e-6f;
  mi_op_params_set_f32(dst, 0, eps);

  mi_compute_rms_norm(dst, 0, 1);

  float mean_sq = (1 + 4 + 9 + 16) / 4.0f;
  float scale = 1.0f / std::sqrt(mean_sq + eps);
  float* d = (float*)dst->data;
  MI_CHECK(close(d[0], 1 * scale));
  MI_CHECK(close(d[1], 2 * scale));
  MI_CHECK(close(d[2], 3 * scale));
  MI_CHECK(close(d[3], 4 * scale));
}

static void test_mul_add_broadcast() {
  // src0: 2 columns of 2 floats. col0=[1,2] col1=[3,4].
  mi_tensor* a = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 2);
  set_data(a, {1, 2, 3, 4});
  // broadcast row vector [10, 100], ne1 == 1.
  mi_tensor* b = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 1);
  set_data(b, {10, 100});

  mi_tensor* mul = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 2);
  mul->op = MI_OP_MUL;
  mul->src[0] = a;
  mul->src[1] = b;
  mi_compute_mul(mul, 0, 1);
  float* m = (float*)mul->data;
  MI_CHECK(m[0] == 10 && m[1] == 200);   // col0 = [1*10, 2*100]
  MI_CHECK(m[2] == 30 && m[3] == 400);   // col1 = [3*10, 4*100]

  mi_tensor* add = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 2);
  add->op = MI_OP_ADD;
  add->src[0] = a;
  add->src[1] = b;
  mi_compute_add(add, 0, 1);
  float* s = (float*)add->data;
  MI_CHECK(s[0] == 11 && s[1] == 102);
  MI_CHECK(s[2] == 13 && s[3] == 104);

  // Same-shape (no broadcast) case.
  mi_tensor* c = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 2);
  set_data(c, {1, 1, 1, 1});
  mi_tensor* add2 = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 2);
  add2->op = MI_OP_ADD;
  add2->src[0] = a;
  add2->src[1] = c;
  mi_compute_add(add2, 0, 1);
  float* s2 = (float*)add2->data;
  MI_CHECK(s2[0] == 2 && s2[1] == 3 && s2[2] == 4 && s2[3] == 5);
}

static void test_mul_mat_plain() {
  // a: [k=2, n=3] rows (as mul_mat "weight rows"): row0=[1,0] row1=[0,1] row2=[1,1]
  mi_tensor* a = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 3);
  set_data(a, {1, 0, 0, 1, 1, 1});
  // b: [k=2, m=1]: [5, 7]
  mi_tensor* b = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 1);
  set_data(b, {5, 7});

  mi_tensor* dst = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 3, 1);
  dst->op = MI_OP_MUL_MAT;
  dst->src[0] = a;
  dst->src[1] = b;
  mi_compute_mul_mat(dst, 0, 1);
  float* d = (float*)dst->data;
  MI_CHECK(d[0] == 5);
  MI_CHECK(d[1] == 7);
  MI_CHECK(d[2] == 12);

  // Simulated 2-thread row split must match.
  mi_tensor* dst2 = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 3, 1);
  dst2->op = MI_OP_MUL_MAT;
  dst2->src[0] = a;
  dst2->src[1] = b;
  mi_compute_mul_mat(dst2, 0, 2);
  mi_compute_mul_mat(dst2, 1, 2);
  for (int i = 0; i < 3; i++) MI_CHECK(((float*)dst2->data)[i] == d[i]);
}

// This is the exact pattern model_qwen2.cpp uses for the attention output
// (scores @ V): V is stored [head_dim, n_kv] (same layout as the K-cache),
// and a *metadata-only* transposed view [n_kv, head_dim] is built via
// mi_context_view so the one general mul_mat kernel can compute it without
// an actual data transpose.
static void test_mul_mat_transposed_view() {
  // V: [head_dim=2, n_kv=3], V(:,0)=[1,2] V(:,1)=[3,4] V(:,2)=[5,6].
  mi_tensor* v = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 3);
  set_data(v, {1, 2, 3, 4, 5, 6});

  // Transposed view: ne=[n_kv=3, head_dim=2], strides swapped, same data.
  int64_t ne[2] = {3, 2};
  size_t nb[2] = {v->nb[1], v->nb[0]};
  mi_tensor* vt = mi_context_view(g_ctx, v, MI_TYPE_F32, 2, ne, nb, 0);
  MI_CHECK(vt->view_src == v);

  // scores (softmax weights): [k=3, m=1] = [0.1, 0.2, 0.7]
  mi_tensor* scores = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 3, 1);
  set_data(scores, {0.1f, 0.2f, 0.7f});

  mi_tensor* dst = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 1);
  dst->op = MI_OP_MUL_MAT;
  dst->src[0] = vt;
  dst->src[1] = scores;
  mi_compute_mul_mat(dst, 0, 1);
  float* d = (float*)dst->data;
  MI_CHECK(close(d[0], 1 * 0.1f + 3 * 0.2f + 5 * 0.7f));  // 4.2
  MI_CHECK(close(d[1], 2 * 0.1f + 4 * 0.2f + 6 * 0.7f));  // 5.2
}

static void test_rope_identity_at_pos_zero() {
  mi_tensor* src = mi_new_tensor_3d(g_ctx, MI_TYPE_F32, 4, 1, 1);
  set_data(src, {1, 2, 3, 4});
  mi_tensor* dst = mi_new_tensor_3d(g_ctx, MI_TYPE_F32, 4, 1, 1);
  dst->op = MI_OP_ROPE;
  dst->src[0] = src;
  mi_op_params_set_i32(dst, 0, 0);       // n_past = 0
  mi_op_params_set_i32(dst, 1, 4);       // n_dims = head_dim
  mi_op_params_set_f32(dst, 2, 10000.0f);

  mi_compute_rope(dst, 0, 1);
  float* d = (float*)dst->data;
  // At absolute position 0, theta = 0 for every pair regardless of
  // frequency, so RoPE must be the identity transform.
  MI_CHECK(close(d[0], 1) && close(d[1], 2) && close(d[2], 3) && close(d[3], 4));
}

static void test_rope_known_rotation() {
  mi_tensor* src = mi_new_tensor_3d(g_ctx, MI_TYPE_F32, 2, 1, 1);
  set_data(src, {1, 0});
  mi_tensor* dst = mi_new_tensor_3d(g_ctx, MI_TYPE_F32, 2, 1, 1);
  dst->op = MI_OP_ROPE;
  dst->src[0] = src;
  mi_op_params_set_i32(dst, 0, 5);    // n_past = 5 -> pos = 5
  mi_op_params_set_i32(dst, 1, 2);    // n_dims = 2 -> one pair, freq = base^0 = 1
  mi_op_params_set_f32(dst, 2, 10000.0f);

  mi_compute_rope(dst, 0, 1);
  float* d = (float*)dst->data;
  float theta = 5.0f;  // pos * freq, freq == 1 for the p=0 pair
  MI_CHECK(close(d[0], std::cos(theta)));
  MI_CHECK(close(d[1], std::sin(theta)));
}

static void test_softmax_causal_mask() {
  // [n_kv=4, n_tokens=2]. token j=0 (pos 0) may only see key 0; token j=1
  // (pos 1) may see keys 0,1. Values at masked-out positions are set to an
  // extreme value to prove they're truly ignored, not just numerically
  // negligible.
  mi_tensor* src = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 4, 2);
  set_data(src, {5, 99, 99, 99,   // column j=0
                 1, 3, 99, 99});  // column j=1
  mi_tensor* dst = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 4, 2);
  dst->op = MI_OP_SOFTMAX;
  dst->src[0] = src;
  mi_op_params_set_i32(dst, 0, 0);      // n_past = 0
  mi_op_params_set_f32(dst, 1, 1.0f);   // scale = 1

  mi_compute_softmax(dst, 0, 1);
  float* d = (float*)dst->data;

  // j=0: single visible key -> softmax is trivially 1, regardless of value.
  MI_CHECK(close(d[0], 1.0f) && d[1] == 0 && d[2] == 0 && d[3] == 0);

  // j=1: visible keys are [1,3] -> softmax([1,3]).
  float e0 = std::exp(1.0f - 3.0f), e1 = std::exp(3.0f - 3.0f);
  float sum = e0 + e1;
  MI_CHECK(close(d[4], e0 / sum));
  MI_CHECK(close(d[5], e1 / sum));
  MI_CHECK(d[6] == 0 && d[7] == 0);  // masked, regardless of the 99s
}

static void test_silu() {
  mi_tensor* src = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 3, 1);
  set_data(src, {-2, 0, 2});
  mi_tensor* dst = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 3, 1);
  dst->op = MI_OP_SILU;
  dst->src[0] = src;
  mi_compute_silu(dst, 0, 1);
  float* d = (float*)dst->data;
  auto silu = [](float x) { return x / (1.0f + std::exp(-x)); };
  MI_CHECK(close(d[0], silu(-2)));
  MI_CHECK(close(d[1], silu(0)));
  MI_CHECK(close(d[2], silu(2)));
}

static void test_scale() {
  mi_tensor* src = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 3, 1);
  set_data(src, {1, 2, 3});
  mi_tensor* dst = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 3, 1);
  dst->op = MI_OP_SCALE;
  dst->src[0] = src;
  mi_op_params_set_f32(dst, 0, 2.5f);
  mi_compute_scale(dst, 0, 1);
  float* d = (float*)dst->data;
  MI_CHECK(d[0] == 2.5f && d[1] == 5.0f && d[2] == 7.5f);
}

static void test_cpy_into_view() {
  // A 5-slot "cache", pre-filled with -1 sentinels; cpy writes only into
  // slot 2, neighbors must be untouched.
  mi_tensor* cache = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 3, 5);
  set_data(cache, {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});
  mi_tensor* src = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 3, 1);
  set_data(src, {7, 8, 9});

  int64_t ne[2] = {3, 1};
  size_t nb[2] = {cache->nb[0], cache->nb[1]};
  mi_tensor* dst_view = mi_context_view(g_ctx, cache, MI_TYPE_F32, 2, ne, nb, 2 * cache->nb[1]);
  dst_view->op = MI_OP_CPY;
  dst_view->src[0] = src;
  mi_compute_cpy(dst_view, 0, 1);

  float* c = (float*)cache->data;
  for (int col = 0; col < 5; col++) {
    if (col == 2) {
      MI_CHECK(c[col * 3 + 0] == 7 && c[col * 3 + 1] == 8 && c[col * 3 + 2] == 9);
    } else {
      MI_CHECK(c[col * 3 + 0] == -1 && c[col * 3 + 1] == -1 && c[col * 3 + 2] == -1);
    }
  }
}

static void test_dispatch() {
  mi_tensor* a = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 1);
  set_data(a, {1, 2});
  mi_tensor* b = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 1);
  set_data(b, {10, 20});
  mi_tensor* dst = mi_new_tensor_2d(g_ctx, MI_TYPE_F32, 2, 1);
  dst->op = MI_OP_ADD;
  dst->src[0] = a;
  dst->src[1] = b;
  mi_compute_forward(dst, 0, 1);
  float* d = (float*)dst->data;
  MI_CHECK(d[0] == 11 && d[1] == 22);
}

int main() {
  mi_context_params p{};
  p.tensor_capacity = 256;
  p.data_capacity = 1 << 16;
  p.no_alloc = false;
  g_ctx = mi_context_new(p);

  test_op_params_roundtrip();
  test_get_rows();
  test_rms_norm();
  test_mul_add_broadcast();
  test_mul_mat_plain();
  test_mul_mat_transposed_view();
  test_rope_identity_at_pos_zero();
  test_rope_known_rotation();
  test_softmax_causal_mask();
  test_silu();
  test_scale();
  test_cpy_into_view();
  test_dispatch();

  mi_context_free(g_ctx);
  return ::mi_test::exit_code();
}
