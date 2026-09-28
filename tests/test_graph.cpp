// tests/test_graph.cpp covers two things:
//   1. mi_graph_build_forward's topological sort on small hand-built
//      graphs (dedup on a diamond dependency, leaf vs. node classification,
//      view ops appearing in g->nodes, multi-root expand()), plus a tiny
//      build+execute integration check.
//   2. The stage 1 -> stage 2a migration's acceptance test PROMPTS.md asks
//      for: every dump tensor from the new graph-based
//      mi_qwen2_forward must be bit-identical to stage 1's direct-call
//      implementation. Those golden bytes were captured *before* the
//      rewrite (see tests/fixtures/stage1_golden_{prefill,decode}/ and
//      docs/stage2a.md) using the real tiny_qwen2.gguf fixture, so this
//      check needs no big model file and no golden-generation code path
//      to keep alive in this binary.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "context.h"
#include "graph.h"
#include "model_qwen2.h"
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

// A no-op unary "op" node just for exercising the DFS -- doesn't need a
// real kernel since these tests only check topology, not computed values
// (mi_compute_forward's dispatch is exercised separately by the
// build+execute integration test below, and exhaustively by
// tests/test_ops_cpu.cpp).
mi_tensor* op_node(mi_op op, mi_tensor* src0, mi_tensor* src1 = nullptr) {
  mi_tensor* t = mi_new_tensor_1d(g_ctx, MI_TYPE_F32, 1);
  t->op = op;
  t->src[0] = src0;
  t->src[1] = src1;
  return t;
}
}  // namespace

static void test_chain() {
  mi_tensor* a = leaf_1d(1, {1});
  mi_tensor* b = op_node(MI_OP_SILU, a);
  mi_tensor* c = op_node(MI_OP_SILU, b);

  mi_cgraph* g = mi_graph_new(g_ctx, 16);
  mi_graph_build_forward(g, c);

  MI_CHECK(g->n_leafs == 1 && g->leafs[0] == a);
  MI_CHECK(g->n_nodes == 2);
  MI_CHECK(g->nodes[0] == b);
  MI_CHECK(g->nodes[1] == c);
}

static void test_diamond_dedup() {
  mi_tensor* a = leaf_1d(1, {1});
  mi_tensor* b = op_node(MI_OP_SILU, a);
  mi_tensor* c = op_node(MI_OP_SILU, a);
  mi_tensor* d = op_node(MI_OP_ADD, b, c);

  mi_cgraph* g = mi_graph_new(g_ctx, 16);
  mi_graph_build_forward(g, d);

  // `a` must appear exactly once despite being reachable via both b and c.
  MI_CHECK(g->n_leafs == 1 && g->leafs[0] == a);
  MI_CHECK(g->n_nodes == 3);
  MI_CHECK(g->nodes[0] == b);
  MI_CHECK(g->nodes[1] == c);
  MI_CHECK(g->nodes[2] == d);
}

static void test_view_is_a_node_not_a_leaf() {
  mi_tensor* w = leaf_1d(4, {1, 2, 3, 4});
  int64_t ne[1] = {2};
  size_t nb[1] = {w->nb[0]};
  mi_tensor* v = mi_graph_view(g_ctx, w, 1, ne, nb, 2 * w->nb[0]);  // last 2 elements
  mi_tensor* r = op_node(MI_OP_SILU, v);

  mi_cgraph* g = mi_graph_new(g_ctx, 16);
  mi_graph_build_forward(g, r);

  MI_CHECK(g->n_leafs == 1 && g->leafs[0] == w);
  // v has op == MI_OP_VIEW (not NONE), so it belongs in nodes even though
  // mi_compute_forward does nothing for it at execute time.
  MI_CHECK(g->n_nodes == 2);
  MI_CHECK(g->nodes[0] == v);
  MI_CHECK(g->nodes[1] == r);
}

static void test_multi_root_expand() {
  mi_tensor* x = leaf_1d(1, {1});
  mi_tensor* e = op_node(MI_OP_SILU, x);
  mi_tensor* y = leaf_1d(1, {2});
  mi_tensor* f = op_node(MI_OP_SILU, y);

  mi_cgraph* g = mi_graph_new(g_ctx, 16);
  mi_graph_build_forward(g, e);         // first root: resets, then visits e
  mi_graph_build_forward_expand(g, f);  // second root: does not reset

  MI_CHECK(g->n_leafs == 2);
  MI_CHECK(g->n_nodes == 2);
  MI_CHECK(g->nodes[0] == e);  // e's subtree was appended first
  MI_CHECK(g->nodes[1] == f);
}

static void test_build_and_execute() {
  mi_tensor* a = leaf_1d(3, {1, 2, 3});
  mi_tensor* b = leaf_1d(3, {10, 20, 30});
  mi_tensor* c = mi_graph_add(g_ctx, a, b);

  mi_cgraph* g = mi_graph_new(g_ctx, 16);
  mi_graph_build_forward(g, c);
  MI_CHECK(c->data != nullptr);  // mi_graph_add already allocated c's data at build time
  mi_graph_compute(g, nullptr);

  float* d = (float*)c->data;
  MI_CHECK(d[0] == 11 && d[1] == 22 && d[2] == 33);
}

// ---------------------------------------------------------------------------
// Stage 1 -> stage 2a bit-exactness regression (see file header comment).
// ---------------------------------------------------------------------------
namespace {
std::string fixture_path() { return std::string(MI_FIXTURE_DIR) + "/tiny_qwen2.gguf"; }

struct captured_tensor {
  std::string name;
  std::vector<float> data;
};

void on_capture(void* user_data, const char* name, const mi_tensor* t) {
  auto* out = (std::vector<captured_tensor>*)user_data;
  size_t n = (size_t)mi_nelements(t);
  captured_tensor ct;
  ct.name = name;
  ct.data.assign((const float*)t->data, (const float*)t->data + n);
  out->push_back(std::move(ct));
}

void check_against_golden(const std::vector<captured_tensor>& items, const std::string& dir) {
  MI_CHECK(!items.empty());
  for (const auto& ct : items) {
    std::string path = dir + "/" + ct.name + ".bin";
    std::ifstream in(path, std::ios::binary);
    MI_CHECK(in.good());
    if (!in.good()) continue;
    in.seekg(0, std::ios::end);
    size_t bytes = (size_t)in.tellg();
    in.seekg(0);
    MI_CHECK(bytes == ct.data.size() * sizeof(float));
    std::vector<float> golden(ct.data.size());
    in.read((char*)golden.data(), (std::streamsize)bytes);
    MI_CHECK(golden.size() == ct.data.size());
    bool bit_exact = (golden.size() == ct.data.size());
    for (size_t i = 0; bit_exact && i < golden.size(); i++) {
      if (golden[i] != ct.data[i]) bit_exact = false;
    }
    MI_CHECK(bit_exact);
    if (!bit_exact) {
      std::fprintf(stderr, "golden mismatch for checkpoint '%s' (dir=%s)\n", ct.name.c_str(),
                   dir.c_str());
    }
  }
}
}  // namespace

static void test_golden_prefill() {
  std::string err;
  mi_qwen2_model* model = mi_qwen2_model_load(fixture_path().c_str(), &err);
  MI_CHECK(model != nullptr);
  if (!model) return;

  mi_kv_cache* kv = mi_kv_cache_init(model->hparams, model->weights, 16);
  int32_t tokens[5] = {0, 1, 2, 3, 4};
  int n_vocab = model->hparams.n_vocab;
  std::vector<float> logits(5 * (size_t)n_vocab);

  std::vector<captured_tensor> captured;
  mi_dump_sink sink{&on_capture, &captured};
  mi_qwen2_forward_params fp{};
  fp.tokens = tokens;
  fp.n_tokens = 5;
  fp.n_past = 0;
  fp.dump = &sink;
  mi_qwen2_forward(model, kv, fp, logits.data());

  check_against_golden(captured, std::string(MI_FIXTURE_DIR) + "/stage1_golden_prefill");

  mi_kv_cache_free(kv);
  mi_qwen2_model_free(model);
}

static void test_golden_decode() {
  std::string err;
  mi_qwen2_model* model = mi_qwen2_model_load(fixture_path().c_str(), &err);
  MI_CHECK(model != nullptr);
  if (!model) return;

  mi_kv_cache* kv = mi_kv_cache_init(model->hparams, model->weights, 16);
  int n_vocab = model->hparams.n_vocab;

  // Matches how tests/fixtures/stage1_golden_decode was generated (see
  // docs/stage2a.md): silently prefill ids[0:5], then dump the n_past=5,
  // n_tokens=1 call for id 5.
  int32_t warmup[5] = {0, 1, 2, 3, 4};
  std::vector<float> warmup_logits(5 * (size_t)n_vocab);
  mi_qwen2_forward_params wp{};
  wp.tokens = warmup;
  wp.n_tokens = 5;
  wp.n_past = 0;
  mi_qwen2_forward(model, kv, wp, warmup_logits.data());
  kv->n_past = 5;

  int32_t decode_tok = 5;
  std::vector<float> logits((size_t)n_vocab);
  std::vector<captured_tensor> captured;
  mi_dump_sink sink{&on_capture, &captured};
  mi_qwen2_forward_params fp{};
  fp.tokens = &decode_tok;
  fp.n_tokens = 1;
  fp.n_past = 5;
  fp.dump = &sink;
  mi_qwen2_forward(model, kv, fp, logits.data());

  check_against_golden(captured, std::string(MI_FIXTURE_DIR) + "/stage1_golden_decode");

  mi_kv_cache_free(kv);
  mi_qwen2_model_free(model);
}

int main() {
  mi_context_params p{};
  p.tensor_capacity = 256;
  p.data_capacity = 1 << 16;
  p.no_alloc = false;
  g_ctx = mi_context_new(p);

  test_chain();
  test_diamond_dedup();
  test_view_is_a_node_not_a_leaf();
  test_multi_root_expand();
  test_build_and_execute();

  mi_context_free(g_ctx);

  test_golden_prefill();
  test_golden_decode();

  return ::mi_test::exit_code();
}
