// Small hand-built graphs covering the scenarios PROMPTS.md's stage 2b
// explicitly asks for: view 链 (view chain), view 后原地 (in-place must
// not fire through a view), 多消费者 (a multi-consumer tensor survives
// until its last reader, even under allocation pressure that would expose
// an early free), 图输出不被复用 (an OUTPUT-flagged tensor is never
// touched by gallocr even when something else in the same graph reads it).
#include <cmath>
#include <cstdio>
#include <initializer_list>

#include "context.h"
#include "gallocr.h"
#include "graph.h"
#include "ops_cpu.h"
#include "test_util.h"

namespace {

mi_context* g_ctx = nullptr;

mi_tensor* leaf_1d(int64_t n, std::initializer_list<float> vals) {
  mi_tensor* t = mi_new_tensor_1d(g_ctx, MI_TYPE_F32, n);
  float* p = (float*)t->data;
  int i = 0;
  for (float v : vals) p[i++] = v;
  return t;
}

// A no_alloc metadata-only context, like the one gallocr manages tensors
// in during a real forward step; t->data starts null for every non-leaf
// node built in it (mi_graph_* builders), unlike g_ctx's eager tensors.
mi_context* new_graph_ctx() {
  mi_context_params p{};
  p.tensor_capacity = 64;
  p.data_capacity = 4096;
  p.no_alloc = true;
  return mi_context_new(p);
}

}  // namespace

static void test_view_chain() {
  mi_context* ctx = new_graph_ctx();
  mi_tensor* x = leaf_1d(1, {4.0f});  // from g_ctx: eager, persistent-ish (WEIGHT not set, but unused as src for compute besides via `a`)
  x->flags = MI_TENSOR_FLAG_WEIGHT;   // simplest way to make it persistent: no allocation needed for it

  mi_tensor* a = mi_graph_silu(ctx, x);  // node, non-persistent, needs allocation

  int64_t ne1[1] = {2};
  size_t nb1[1] = {a->nb[0]};
  mi_tensor* view1 = mi_graph_view(ctx, a, 1, ne1, nb1, 0);  // first 2 elements of a

  int64_t ne2[1] = {1};
  size_t nb2[1] = {view1->nb[0]};
  mi_tensor* view2 = mi_graph_view(ctx, view1, 1, ne2, nb2, view1->nb[0]);  // 2nd element of view1

  mi_tensor* consumer = mi_graph_silu(ctx, view2);

  MI_CHECK(view1->view_src == a);
  MI_CHECK(view2->view_src == a);  // flattened to one hop, not view1

  mi_cgraph* g = mi_graph_new(ctx, 16);
  mi_graph_build_forward(g, consumer);

  mi_gallocr* galloc = mi_gallocr_new();
  mi_cgraph* graphs[1] = {g};
  mi_gallocr_reserve(galloc, graphs, 1);
  mi_gallocr_alloc_graph(galloc, g);

  MI_CHECK(a->data != nullptr);
  MI_CHECK(view1->data == (unsigned char*)a->data + view1->view_offs);
  MI_CHECK(view2->data == (unsigned char*)a->data + view2->view_offs);
  // consumer is the only real (non-view) reader, reached through 2 view
  // hops -- both correctly credited to `a` itself.
  MI_CHECK(a->n_children == 1);

  mi_gallocr_free(galloc);
  mi_context_free(ctx);
}

static void test_inplace_not_through_view() {
  mi_context* ctx = new_graph_ctx();
  mi_tensor* x = leaf_1d(2, {1.0f, 2.0f});
  x->flags = MI_TENSOR_FLAG_WEIGHT;
  mi_tensor* y = leaf_1d(2, {10.0f, 20.0f});
  y->flags = MI_TENSOR_FLAG_WEIGHT;

  mi_tensor* a = mi_graph_silu(ctx, x);  // node, needs allocation

  int64_t ne[1] = {2};
  size_t nb[1] = {a->nb[0]};
  mi_tensor* view_of_a = mi_graph_view(ctx, a, 1, ne, nb, 0);  // full-range view, still a view

  // b reads the VIEW, not `a` directly -- op_supports_inplace(ADD) is
  // true and a's refcount will be exactly 1, so every condition *except*
  // "non-view" would allow reuse; this must still not alias, or writing
  // through b would corrupt `a` if any *other* view of it were still live
  // in a less trivial graph (see docs/stage2b.md for the real bug this
  // class of check catches: attn_concat via mul_mat).
  mi_tensor* b = mi_graph_add(ctx, view_of_a, y);

  mi_cgraph* g = mi_graph_new(ctx, 16);
  mi_graph_build_forward(g, b);

  mi_gallocr* galloc = mi_gallocr_new();
  mi_cgraph* graphs[1] = {g};
  mi_gallocr_reserve(galloc, graphs, 1);
  mi_gallocr_alloc_graph(galloc, g);

  MI_CHECK(a->n_children == 1);
  MI_CHECK(b->data != a->data);
  MI_CHECK(mi_gallocr_get_stats(galloc).n_inplace_reuse == 0);

  mi_context_free(ctx);
  mi_gallocr_free(galloc);
}

static void test_multiple_consumers_survive_to_last_reader() {
  mi_context* ctx = new_graph_ctx();
  mi_tensor* x = leaf_1d(1, {4.0f});
  x->flags = MI_TENSOR_FLAG_WEIGHT;

  mi_tensor* a = mi_graph_silu(ctx, x);        // read by BOTH b and c below
  mi_tensor* b = mi_graph_scale(ctx, a, 2.0f);  // consumer 1
  // A same-size distractor allocated *between* the two reads of `a`: if
  // gallocr incorrectly freed `a` after b's processing (treating refcount
  // as if b were the only consumer), best-fit would hand junk this exact
  // freed slot, corrupting `a` before c gets to read it.
  mi_tensor* junk = mi_graph_silu(ctx, x);
  mi_tensor* c = mi_graph_scale(ctx, a, 3.0f);  // consumer 2
  mi_tensor* d = mi_graph_add(ctx, b, c);
  mi_tensor* e = mi_graph_add(ctx, d, junk);

  mi_cgraph* g = mi_graph_new(ctx, 16);
  mi_graph_build_forward(g, e);

  mi_gallocr* galloc = mi_gallocr_new();
  mi_cgraph* graphs[1] = {g};
  mi_gallocr_reserve(galloc, graphs, 1);
  mi_gallocr_alloc_graph(galloc, g);

  MI_CHECK(a->n_children == 2);
  MI_CHECK(b->data != a->data);  // refcount 2 != 1: in-place must not fire
  MI_CHECK(c->data != a->data);

  mi_graph_compute(g, nullptr);

  float silu4 = 4.0f / (1.0f + std::exp(-4.0f));
  float expected = silu4 * 2.0f + silu4 * 3.0f + silu4;  // b + c + junk
  MI_CHECK(std::fabs(*(float*)e->data - expected) < 1e-4f);

  mi_context_free(ctx);
  mi_gallocr_free(galloc);
}

static void test_graph_output_not_reused() {
  mi_context* ctx = new_graph_ctx();
  mi_tensor* x = leaf_1d(1, {4.0f});
  x->flags = MI_TENSOR_FLAG_WEIGHT;

  mi_tensor* out = mi_graph_silu(ctx, x);
  // Simulates how mi_qwen2_forward treats `logits`: eagerly given real
  // data and flagged OUTPUT *before* gallocr ever sees it, even though
  // (unlike logits) it's also read by something else in this graph.
  float backing = 0.0f;
  out->data = &backing;
  out->flags = MI_TENSOR_FLAG_OUTPUT;

  mi_tensor* y = mi_graph_scale(ctx, out, 2.0f);  // reads `out` as a normal input

  mi_cgraph* g = mi_graph_new(ctx, 16);
  mi_graph_build_forward(g, y);

  mi_gallocr* galloc = mi_gallocr_new();
  mi_cgraph* graphs[1] = {g};
  mi_gallocr_reserve(galloc, graphs, 1);
  mi_gallocr_alloc_graph(galloc, g);

  // Persistent (OUTPUT is one of mi_tensor_is_persistent's bits): never
  // counted as a reference, never allocated/freed by gallocr.
  MI_CHECK(out->n_children == 0);
  MI_CHECK(out->data == &backing);

  // out->data was never populated by a kernel in this test (only `y`'s
  // computation runs), so seed it by hand and confirm `y` reads through
  // the untouched pointer correctly.
  backing = 4.0f / (1.0f + std::exp(-4.0f));  // silu(4)
  mi_compute_forward(y, 0, 1);
  MI_CHECK(std::fabs(*(float*)y->data - backing * 2.0f) < 1e-4f);

  mi_context_free(ctx);
  mi_gallocr_free(galloc);
}

static void test_stats_sane() {
  mi_context* ctx = new_graph_ctx();
  mi_tensor* x = leaf_1d(4, {1.0f, 2.0f, 3.0f, 4.0f});
  x->flags = MI_TENSOR_FLAG_WEIGHT;

  // A chain of in-place-eligible ops: each has exactly one consumer, so
  // every step after the first should reuse its predecessor's buffer.
  mi_tensor* a = mi_graph_silu(ctx, x);
  mi_tensor* b = mi_graph_scale(ctx, a, 2.0f);
  mi_tensor* c = mi_graph_scale(ctx, b, 3.0f);

  mi_cgraph* g = mi_graph_new(ctx, 16);
  mi_graph_build_forward(g, c);

  mi_gallocr* galloc = mi_gallocr_new();
  mi_cgraph* graphs[1] = {g};
  mi_gallocr_reserve(galloc, graphs, 1);
  mi_gallocr_alloc_graph(galloc, g);

  mi_gallocr_stats stats = mi_gallocr_get_stats(galloc);
  MI_CHECK(stats.n_inplace_reuse >= 2);  // b into a's slot, c into b's slot
  MI_CHECK(stats.peak_bytes <= stats.baseline_bytes);
  MI_CHECK(b->data == a->data);
  MI_CHECK(c->data == a->data);

  mi_context_free(ctx);
  mi_gallocr_free(galloc);
}

int main() {
  mi_context_params p{};
  p.tensor_capacity = 256;
  p.data_capacity = 1 << 16;
  p.no_alloc = false;
  g_ctx = mi_context_new(p);

  test_view_chain();
  test_inplace_not_through_view();
  test_multiple_consumers_survive_to_last_reader();
  test_graph_output_not_reused();
  test_stats_sane();

  mi_context_free(g_ctx);
  return ::mi_test::exit_code();
}
