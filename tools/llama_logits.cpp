// mini-infer: teacher-forcing logits exporter built on llama.cpp's own C
// API -- the only file in this project that links third_party/llama.cpp
// (see tools/CMakeLists.txt). Used for stage 4's Q8_0-vs-llama.cpp
// comparison (SPEC.md "对拍标准"): given the same GGUF and the same token
// id sequence, export every position's logits so scripts/compare_q8_llamacpp.py
// can compute top-1 agreement / cosine similarity / max abs error against
// this project's own --dump output.
//
// Deliberately minimal: llama_decode + llama_get_logits_ith only, no
// common_params/common_init_from_params (that pulls in libllama-common and
// its tokenizer/sampler machinery, none of which this tool needs since the
// prompt is already a token id list, not text -- same reasoning as
// main.cpp's --prompt-ids).
//
// Usage: llama_logits <model.gguf> <ids_file> <out_prefix>
// ids_file: whitespace-separated token ids (same format as mini_infer's
// --prompt-ids). Output: <out_prefix>.bin (n_tokens * n_vocab float32,
// row-major) + <out_prefix>.shape ("n_tokens n_vocab"), matching
// main.cpp's write_dump_tensor so both sides load identically in Python.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "llama.h"

namespace {

bool read_ids_file(const std::string& path, std::vector<llama_token>* ids) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "failed to open ids file: %s\n", path.c_str());
    return false;
  }
  int32_t id;
  while (in >> id) ids->push_back((llama_token)id);
  return !ids->empty();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <model.gguf> <ids_file> <out_prefix>\n", argv[0]);
    return 1;
  }
  const std::string model_path = argv[1];
  const std::string ids_path = argv[2];
  const std::string out_prefix = argv[3];

  std::vector<llama_token> tokens;
  if (!read_ids_file(ids_path, &tokens)) {
    std::fprintf(stderr, "ids file has no token ids: %s\n", ids_path.c_str());
    return 1;
  }
  const int n_tokens = (int)tokens.size();

  llama_backend_init();

  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;  // CPU only, for a fair comparison with mini-infer
  llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
  if (!model) {
    std::fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
    return 1;
  }

  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = (uint32_t)n_tokens;
  cp.n_batch = (uint32_t)n_tokens;
  cp.n_ubatch = (uint32_t)n_tokens;
  int n_threads = (int)std::thread::hardware_concurrency();
  if (n_threads <= 0) n_threads = 1;
  cp.n_threads = n_threads;
  cp.n_threads_batch = n_threads;

  llama_context* ctx = llama_init_from_model(model, cp);
  if (!ctx) {
    std::fprintf(stderr, "failed to create llama_context\n");
    llama_model_free(model);
    return 1;
  }

  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);

  // llama_batch_get_one() only requests the last token's logits; teacher-
  // forcing comparison needs every position, so build the batch by hand
  // with logits[i] = 1 for all i.
  llama_batch batch = llama_batch_init(n_tokens, /*embd=*/0, /*n_seq_max=*/1);
  for (int i = 0; i < n_tokens; i++) {
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = 1;
  }
  batch.n_tokens = n_tokens;

  if (llama_decode(ctx, batch) != 0) {
    std::fprintf(stderr, "llama_decode failed\n");
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 1;
  }

  std::vector<float> logits((size_t)n_tokens * (size_t)n_vocab);
  for (int i = 0; i < n_tokens; i++) {
    const float* row = llama_get_logits_ith(ctx, i);
    std::memcpy(logits.data() + (size_t)i * (size_t)n_vocab, row, sizeof(float) * (size_t)n_vocab);
  }

  std::ofstream bin(out_prefix + ".bin", std::ios::binary);
  bin.write((const char*)logits.data(), (std::streamsize)(logits.size() * sizeof(float)));
  std::ofstream shp(out_prefix + ".shape");
  shp << n_tokens << " " << n_vocab << "\n";

  std::fprintf(stderr, "wrote %d x %d logits to %s.bin\n", n_tokens, n_vocab, out_prefix.c_str());

  llama_batch_free(batch);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
