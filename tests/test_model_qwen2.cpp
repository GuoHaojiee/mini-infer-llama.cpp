// Hermetic tests against tests/fixtures/tiny_qwen2.gguf (random weights,
// tiny shapes -- see scripts/make_tiny_qwen2_fixture.py). This does not
// check numerical accuracy against HF (the weights aren't real; that's
// scripts/compare.py's job on the real Qwen2.5-0.5B GGUF). What it does
// check, at CTest speed with no external files: hparams parse correctly,
// the forward pass produces finite output of the right shape, and --
// PROMPTS.md's explicit stage 1 "额外验证" -- batch prefill and
// token-by-token incremental decode produce bit-identical logits at every
// position.
#include <cmath>
#include <cstdio>
#include <vector>

#include "model_qwen2.h"
#include "test_util.h"

namespace {
std::string fixture_path() { return std::string(MI_FIXTURE_DIR) + "/tiny_qwen2.gguf"; }
std::string q8_0_fixture_path() { return std::string(MI_FIXTURE_DIR) + "/tiny_qwen2_q8_0.gguf"; }
}  // namespace

static void test_hparams() {
  std::string err;
  mi_qwen2_model* model = mi_qwen2_model_load(fixture_path().c_str(), &err);
  MI_CHECK(model != nullptr);
  if (!model) {
    std::fprintf(stderr, "failed to load fixture: %s\n", err.c_str());
    return;
  }
  MI_CHECK(model->hparams.n_layer == 2);
  MI_CHECK(model->hparams.n_embd == 8);
  MI_CHECK(model->hparams.n_head == 2);
  MI_CHECK(model->hparams.n_head_kv == 1);
  MI_CHECK(model->hparams.head_dim == 4);
  MI_CHECK(model->hparams.n_ff == 16);
  MI_CHECK(model->hparams.n_vocab == 32);
  MI_CHECK(model->weights.output == model->weights.token_embd);  // tied lm_head
  mi_qwen2_model_free(model);
}

static void test_missing_file() {
  std::string err;
  mi_qwen2_model* model = mi_qwen2_model_load("/nonexistent/model.gguf", &err);
  MI_CHECK(model == nullptr);
  MI_CHECK(!err.empty());
}

static void test_prefill_shape_and_finite() {
  std::string err;
  mi_qwen2_model* model = mi_qwen2_model_load(fixture_path().c_str(), &err);
  MI_CHECK(model != nullptr);
  if (!model) return;

  int n_ctx = 16;
  mi_kv_cache* kv = mi_kv_cache_init(model->hparams, model->weights, n_ctx);

  int32_t tokens[5] = {0, 1, 2, 3, 4};
  int n_vocab = model->hparams.n_vocab;
  std::vector<float> logits(5 * n_vocab);

  mi_qwen2_forward_params fp{};
  fp.tokens = tokens;
  fp.n_tokens = 5;
  fp.n_past = 0;
  fp.dump = nullptr;
  mi_qwen2_forward(model, kv, fp, logits.data());

  bool all_finite = true;
  for (float v : logits) {
    if (!std::isfinite(v)) all_finite = false;
  }
  MI_CHECK(all_finite);

  // Greedy sampling picks a real, in-range vocab id.
  int32_t top = mi_qwen2_sample_greedy(logits.data(), n_vocab);
  MI_CHECK(top >= 0 && top < n_vocab);

  mi_kv_cache_free(kv);
  mi_qwen2_model_free(model);
}

// The internal-consistency check PROMPTS.md's stage 1 explicitly asks for:
// batch prefill and one-token-at-a-time incremental decode must produce
// exactly the same logits at every position. Any bug in RoPE's position
// offset, the causal mask boundary, or the KV-cache write offset would
// break this even if it happened to still look plausible on its own.
static void test_batch_vs_incremental_logits_match() {
  std::string err;
  mi_qwen2_model* model = mi_qwen2_model_load(fixture_path().c_str(), &err);
  MI_CHECK(model != nullptr);
  if (!model) return;

  int n_ctx = 16;
  int n_vocab = model->hparams.n_vocab;
  int32_t tokens[5] = {3, 7, 1, 22, 5};
  int n_tokens = 5;

  std::vector<float> batch_logits((size_t)n_tokens * n_vocab);
  {
    mi_kv_cache* kv = mi_kv_cache_init(model->hparams, model->weights, n_ctx);
    mi_qwen2_forward_params fp{};
    fp.tokens = tokens;
    fp.n_tokens = n_tokens;
    fp.n_past = 0;
    mi_qwen2_forward(model, kv, fp, batch_logits.data());
    mi_kv_cache_free(kv);
  }

  std::vector<float> incr_logits((size_t)n_tokens * n_vocab);
  {
    mi_kv_cache* kv = mi_kv_cache_init(model->hparams, model->weights, n_ctx);
    for (int t = 0; t < n_tokens; t++) {
      mi_qwen2_forward_params fp{};
      fp.tokens = &tokens[t];
      fp.n_tokens = 1;
      fp.n_past = t;
      mi_qwen2_forward(model, kv, fp, incr_logits.data() + (size_t)t * n_vocab);
      kv->n_past = t + 1;
    }
    mi_kv_cache_free(kv);
  }

  int mismatches = 0;
  for (size_t i = 0; i < batch_logits.size(); i++) {
    if (batch_logits[i] != incr_logits[i]) mismatches++;
  }
  MI_CHECK(mismatches == 0);
  if (mismatches != 0) {
    std::fprintf(stderr, "batch vs incremental: %d/%zu logits differ\n", mismatches,
                 batch_logits.size());
  }

  mi_qwen2_model_free(model);
}

// Stage 6 wrap-up: last_logits_only=true must produce exactly the same
// bytes as the last position of a full (last_logits_only=false) call --
// it's the same lm_head matmul, just restricted via a free view to one
// column instead of all of them (see model_qwen2.cpp's build_model_graph),
// so this should be bit-exact, not just close.
static void test_last_logits_only_matches_full() {
  std::string err;
  mi_qwen2_model* model = mi_qwen2_model_load(fixture_path().c_str(), &err);
  MI_CHECK(model != nullptr);
  if (!model) return;

  int n_ctx = 16;
  mi_kv_cache* kv = mi_kv_cache_init(model->hparams, model->weights, n_ctx);

  int32_t tokens[5] = {0, 1, 2, 3, 4};
  int n_vocab = model->hparams.n_vocab;

  std::vector<float> full_logits(5 * n_vocab);
  mi_qwen2_forward_params fp_full{};
  fp_full.tokens = tokens;
  fp_full.n_tokens = 5;
  fp_full.n_past = 0;
  mi_qwen2_forward(model, kv, fp_full, full_logits.data());

  std::vector<float> last_only(n_vocab);
  mi_qwen2_forward_params fp_last{};
  fp_last.tokens = tokens;
  fp_last.n_tokens = 5;
  fp_last.n_past = 0;
  fp_last.last_logits_only = true;
  mi_qwen2_forward(model, kv, fp_last, last_only.data());

  const float* expected = full_logits.data() + 4 * (size_t)n_vocab;
  int mismatches = 0;
  for (int i = 0; i < n_vocab; i++) {
    if (last_only[i] != expected[i]) mismatches++;
  }
  MI_CHECK(mismatches == 0);

  mi_kv_cache_free(kv);
  mi_qwen2_model_free(model);
}

// -- Stage 4: tests/fixtures/tiny_qwen2_q8_0.gguf (all 2D weight matrices
// Q8_0, norm/bias F32 -- see scripts/make_tiny_qwen2_q8_0_fixture.py, built
// with gguf-py's own reference quantizer, not this project's). These are
// the same three properties the F32 fixture checks above, now exercising
// mi_graph_mul_mat's quantize-node insertion, reserve_gallocr's real-type
// mirroring (task from docs/stage4.md), and mi_compute_get_rows's Q8_0
// dequant path end-to-end.
static void test_q8_0_hparams() {
  std::string err;
  mi_qwen2_model* model = mi_qwen2_model_load(q8_0_fixture_path().c_str(), &err);
  MI_CHECK(model != nullptr);
  if (!model) {
    std::fprintf(stderr, "failed to load q8_0 fixture: %s\n", err.c_str());
    return;
  }
  MI_CHECK(model->hparams.n_layer == 2);
  MI_CHECK(model->hparams.n_embd == 32);
  MI_CHECK(model->weights.token_embd->type == MI_TYPE_Q8_0);
  MI_CHECK(model->weights.output == model->weights.token_embd);  // tied lm_head, also Q8_0
  MI_CHECK(model->weights.layers[0].attn_q_w->type == MI_TYPE_Q8_0);
  MI_CHECK(model->weights.layers[0].attn_norm->type == MI_TYPE_F32);  // norms stay F32
  MI_CHECK(model->weights.layers[0].attn_q_b->type == MI_TYPE_F32);   // biases stay F32
  mi_qwen2_model_free(model);
}

static void test_q8_0_prefill_shape_and_finite() {
  std::string err;
  mi_qwen2_model* model = mi_qwen2_model_load(q8_0_fixture_path().c_str(), &err);
  MI_CHECK(model != nullptr);
  if (!model) return;

  int n_ctx = 16;
  mi_kv_cache* kv = mi_kv_cache_init(model->hparams, model->weights, n_ctx);

  int32_t tokens[5] = {0, 1, 2, 3, 4};
  int n_vocab = model->hparams.n_vocab;
  std::vector<float> logits(5 * n_vocab);

  mi_qwen2_forward_params fp{};
  fp.tokens = tokens;
  fp.n_tokens = 5;
  fp.n_past = 0;
  mi_qwen2_forward(model, kv, fp, logits.data());

  bool all_finite = true;
  for (float v : logits) {
    if (!std::isfinite(v)) all_finite = false;
  }
  MI_CHECK(all_finite);

  int32_t top = mi_qwen2_sample_greedy(logits.data(), n_vocab);
  MI_CHECK(top >= 0 && top < n_vocab);

  mi_kv_cache_free(kv);
  mi_qwen2_model_free(model);
}

// Quantizing each activation row happens independently per row (see
// mi_compute_quantize_q8_0), so this property should survive Q8_0 exactly
// as it does for F32 -- a regression here would mean either RoPE/softmax's
// position handling broke, or (more specific to this stage) that
// quantization is accidentally coupling across rows/batches somehow.
static void test_q8_0_batch_vs_incremental_logits_match() {
  std::string err;
  mi_qwen2_model* model = mi_qwen2_model_load(q8_0_fixture_path().c_str(), &err);
  MI_CHECK(model != nullptr);
  if (!model) return;

  int n_ctx = 16;
  int n_vocab = model->hparams.n_vocab;
  int32_t tokens[5] = {3, 7, 1, 22, 5};
  int n_tokens = 5;

  std::vector<float> batch_logits((size_t)n_tokens * n_vocab);
  {
    mi_kv_cache* kv = mi_kv_cache_init(model->hparams, model->weights, n_ctx);
    mi_qwen2_forward_params fp{};
    fp.tokens = tokens;
    fp.n_tokens = n_tokens;
    fp.n_past = 0;
    mi_qwen2_forward(model, kv, fp, batch_logits.data());
    mi_kv_cache_free(kv);
  }

  std::vector<float> incr_logits((size_t)n_tokens * n_vocab);
  {
    mi_kv_cache* kv = mi_kv_cache_init(model->hparams, model->weights, n_ctx);
    for (int t = 0; t < n_tokens; t++) {
      mi_qwen2_forward_params fp{};
      fp.tokens = &tokens[t];
      fp.n_tokens = 1;
      fp.n_past = t;
      mi_qwen2_forward(model, kv, fp, incr_logits.data() + (size_t)t * n_vocab);
      kv->n_past = t + 1;
    }
    mi_kv_cache_free(kv);
  }

  int mismatches = 0;
  for (size_t i = 0; i < batch_logits.size(); i++) {
    if (batch_logits[i] != incr_logits[i]) mismatches++;
  }
  MI_CHECK(mismatches == 0);
  if (mismatches != 0) {
    std::fprintf(stderr, "q8_0 batch vs incremental: %d/%zu logits differ\n", mismatches,
                 batch_logits.size());
  }

  mi_qwen2_model_free(model);
}

int main() {
  test_hparams();
  test_missing_file();
  test_prefill_shape_and_finite();
  test_batch_vs_incremental_logits_match();
  test_last_logits_only_matches_full();
  test_q8_0_hparams();
  test_q8_0_prefill_shape_and_finite();
  test_q8_0_batch_vs_incremental_logits_match();
  return ::mi_test::exit_code();
}
