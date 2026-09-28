// mini-infer CLI: loads a Qwen2 GGUF model, prefills a token-id prompt
// (read from --prompt-ids, since encoding text is HF's job -- see
// scripts/encode.py, and SPEC.md), then either dumps teacher-forcing
// checkpoint tensors (--dump) or greedily decodes -n new tokens.
//
// stdout carries only generated token ids (one decimal integer per line)
// so scripts/compare.py can capture a generation run with a plain
// subprocess pipe; decoded text and all diagnostics go to stderr.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "gallocr.h"
#include "graph.h"
#include "model_qwen2.h"
#include "tensor.h"

namespace {

struct mi_cli_options {
  std::string model_path;
  int n_ctx = 2048;
  int n_threads = 4;
  bool pin_threads = true;
  mi_mulmat_mode mulmat_mode = MI_MULMAT_STATIC;
  int mulmat_chunk = 8;
  int n_predict = 128;
  std::string prompt_ids_path;
  std::string dump_dir;              // empty = no dump
  std::string dump_incremental_dir;  // empty = not requested
  int dump_n_past = 0;  // with --dump/--gallocr-stats: silently prefill
                        // ids[0:n_past] first, then run the *measured*
                        // forward call over ids[n_past:] -- lets --dump
                        // capture a decode-style (n_tokens=1, n_past>0)
                        // call, and lets --gallocr-stats measure gallocr's
                        // stats for a specific (n_tokens, n_past) pair
                        // (see scripts/bench_gallocr.py).
  bool gallocr_stats = false;
  bool graph_stats = false;  // stage 6 measurement: prints a per-op-type
                              // node-count histogram for the measured call
                              // (see docs/stage6.md's barrier-count step).
  bool bench = false;  // see scripts/bench_threads.py: times only the
                        // prefill call and the decode loop's forward
                        // calls (std::chrono, in-process), so per-process
                        // overhead (mmap, gallocr reserve) never pollutes
                        // the reported tok/s -- unlike wrapping the whole
                        // subprocess in Python's wall-clock timer would.
};

void print_usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s <model.gguf> --prompt-ids <ids.txt> [-c N] [-t N] [-n N] "
               "[--dump DIR [--dump-n-past K]] [--dump-incremental DIR] [--gallocr-stats] "
               "[--graph-stats] [--bench] "
               "[--no-pin-threads] [--mulmat-mode single|static|dynamic] [--mulmat-chunk N]\n",
               argv0);
}

bool parse_mulmat_mode(const char* s, mi_mulmat_mode* out) {
  if (std::strcmp(s, "single") == 0) {
    *out = MI_MULMAT_SINGLE;
  } else if (std::strcmp(s, "static") == 0) {
    *out = MI_MULMAT_STATIC;
  } else if (std::strcmp(s, "dynamic") == 0) {
    *out = MI_MULMAT_DYNAMIC;
  } else {
    return false;
  }
  return true;
}

bool parse_args(int argc, char** argv, mi_cli_options* opt) {
  std::vector<std::string> positional;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) return nullptr;
      return argv[++i];
    };
    if (a == "-c" || a == "--ctx") {
      const char* v = next();
      if (!v) return false;
      opt->n_ctx = std::atoi(v);
    } else if (a == "-t") {
      const char* v = next();
      if (!v) return false;
      opt->n_threads = std::atoi(v);
    } else if (a == "-n") {
      const char* v = next();
      if (!v) return false;
      opt->n_predict = std::atoi(v);
    } else if (a == "--prompt-ids") {
      const char* v = next();
      if (!v) return false;
      opt->prompt_ids_path = v;
    } else if (a == "--dump") {
      const char* v = next();
      if (!v) return false;
      opt->dump_dir = v;
    } else if (a == "--dump-n-past") {
      const char* v = next();
      if (!v) return false;
      opt->dump_n_past = std::atoi(v);
    } else if (a == "--dump-incremental") {
      const char* v = next();
      if (!v) return false;
      opt->dump_incremental_dir = v;
    } else if (a == "--gallocr-stats") {
      opt->gallocr_stats = true;
    } else if (a == "--graph-stats") {
      opt->graph_stats = true;
    } else if (a == "--bench") {
      opt->bench = true;
    } else if (a == "--no-pin-threads") {
      opt->pin_threads = false;
    } else if (a == "--pin-threads") {
      opt->pin_threads = true;
    } else if (a == "--mulmat-mode") {
      const char* v = next();
      if (!v || !parse_mulmat_mode(v, &opt->mulmat_mode)) return false;
    } else if (a == "--mulmat-chunk") {
      const char* v = next();
      if (!v) return false;
      opt->mulmat_chunk = std::atoi(v);
    } else if (a == "-h" || a == "--help") {
      return false;
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "unknown option: %s\n", a.c_str());
      return false;
    } else {
      positional.push_back(a);
    }
  }
  if (positional.size() != 1) {
    std::fprintf(stderr, "expected exactly one positional argument (model path), got %zu\n",
                 positional.size());
    return false;
  }
  opt->model_path = positional[0];
  if (opt->prompt_ids_path.empty()) {
    std::fprintf(stderr, "--prompt-ids is required\n");
    return false;
  }
  return true;
}

bool read_ids_file(const std::string& path, std::vector<int32_t>* ids) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "failed to open ids file: %s\n", path.c_str());
    return false;
  }
  int32_t id;
  while (in >> id) ids->push_back(id);
  if (ids->empty()) {
    std::fprintf(stderr, "ids file has no token ids: %s\n", path.c_str());
    return false;
  }
  return true;
}

// dst tensors passed to mi_dump_sink are always freshly allocated
// contiguous 2D [feature_dim, n_tokens] tensors (see model_qwen2.cpp), so a
// raw byte dump plus a "numpy order" (n_tokens, feature_dim) shape line is
// enough for scripts/compare.py to load with plain numpy.fromfile+reshape.
void write_dump_tensor(const std::string& dir, const char* name, const mi_tensor* t) {
  std::string base = dir + "/" + name;
  std::ofstream bin(base + ".bin", std::ios::binary);
  bin.write((const char*)t->data, (std::streamsize)mi_nbytes(t));
  std::ofstream shp(base + ".shape");
  shp << t->ne[1] << " " << t->ne[0] << "\n";
}

struct dump_ctx {
  std::string dir;
};

void on_dump_tensor(void* user_data, const char* name, const mi_tensor* t) {
  write_dump_tensor(((dump_ctx*)user_data)->dir, name, t);
}

struct incremental_dump_ctx {
  std::string dir;
  int step;
};

// Only "logits" is written (one file per step, name includes the step
// index) -- scripts/compare.py checks that feeding the prompt one token at
// a time through a persistent, growing KV cache gives the exact same
// per-position logits as a single batched prefill call (PROMPTS.md's
// "批量 prefill 与逐 token prefill 的最终 logits 一致"). This exercises the
// real model end-to-end; tests/test_model_qwen2.cpp already checks the
// same property hermetically against a tiny synthetic model.
void on_dump_incremental(void* user_data, const char* name, const mi_tensor* t) {
  if (std::strcmp(name, "logits") != 0) return;
  auto* dc = (incremental_dump_ctx*)user_data;
  char fname[64];
  std::snprintf(fname, sizeof(fname), "logits_step_%d", dc->step);
  write_dump_tensor(dc->dir, fname, t);
}

}  // namespace

int main(int argc, char** argv) {
  mi_cli_options opt;
  if (!parse_args(argc, argv, &opt)) {
    print_usage(argv[0]);
    return 2;
  }

  std::vector<int32_t> prompt_ids;
  if (!read_ids_file(opt.prompt_ids_path, &prompt_ids)) return 1;

  std::string err;
  mi_qwen2_model* model = mi_qwen2_model_load(opt.model_path.c_str(), &err);
  if (!model) {
    std::fprintf(stderr, "failed to load model: %s\n", err.c_str());
    return 1;
  }

  int n_ctx = opt.n_ctx;
  if ((int)prompt_ids.size() > n_ctx) {
    std::fprintf(stderr, "prompt (%zu tokens) exceeds -c/--ctx (%d)\n", prompt_ids.size(), n_ctx);
    mi_qwen2_model_free(model);
    return 1;
  }

  mi_threadpool_params tp_params{opt.n_threads, opt.pin_threads, opt.mulmat_mode, opt.mulmat_chunk};
  mi_kv_cache* kv = mi_kv_cache_init(model->hparams, model->weights, n_ctx, tp_params);
  int n_vocab = model->hparams.n_vocab;

  if (!opt.dump_dir.empty()) {
    // Teacher-forcing dump over ids[dump_n_past:], with ids[0:dump_n_past]
    // silently prefilled first (not dumped) to populate real KV-cache
    // history -- dump_n_past=0 (the default) is a from-scratch prefill
    // dump, matching scripts/ref_hf.py's single forward pass with
    // output_hidden_states=True; dump_n_past>0 captures a decode-style
    // (small n_tokens, real history) call instead.
    if (opt.dump_n_past < 0 || opt.dump_n_past >= (int)prompt_ids.size()) {
      std::fprintf(stderr, "--dump-n-past must be in [0, n_prompt_tokens)\n");
      mi_kv_cache_free(kv);
      mi_qwen2_model_free(model);
      return 1;
    }
    if (opt.dump_n_past > 0) {
      std::vector<float> warmup_logits((size_t)opt.dump_n_past * n_vocab);
      mi_qwen2_forward_params wp{};
      wp.tokens = prompt_ids.data();
      wp.n_tokens = opt.dump_n_past;
      wp.n_past = 0;
      mi_qwen2_forward(model, kv, wp, warmup_logits.data());
      kv->n_past = opt.dump_n_past;
    }
    int32_t* rest_tokens = prompt_ids.data() + opt.dump_n_past;
    int n_rest = (int)prompt_ids.size() - opt.dump_n_past;
    std::vector<float> logits((size_t)n_rest * n_vocab);
    dump_ctx dc{opt.dump_dir};
    mi_dump_sink sink{&on_dump_tensor, &dc};
    mi_qwen2_forward_params fp{};
    fp.tokens = rest_tokens;
    fp.n_tokens = n_rest;
    fp.n_past = opt.dump_n_past;
    fp.dump = &sink;
    mi_qwen2_forward(model, kv, fp, logits.data());
    mi_kv_cache_free(kv);
    mi_qwen2_model_free(model);
    return 0;
  }

  if (opt.gallocr_stats) {
    // Same ids[0:dump_n_past] silent-warmup / ids[dump_n_past:] measured-
    // call split as --dump, but just reports mi_gallocr_get_stats() for
    // the measured call instead of dumping tensors -- see
    // scripts/bench_gallocr.py, which drives this across the n_tokens/
    // n_past points bench/results/gallocr.txt reports.
    if (opt.dump_n_past < 0 || opt.dump_n_past >= (int)prompt_ids.size()) {
      std::fprintf(stderr, "--dump-n-past must be in [0, n_prompt_tokens)\n");
      mi_kv_cache_free(kv);
      mi_qwen2_model_free(model);
      return 1;
    }
    if (opt.dump_n_past > 0) {
      std::vector<float> warmup_logits((size_t)opt.dump_n_past * n_vocab);
      mi_qwen2_forward_params wp{};
      wp.tokens = prompt_ids.data();
      wp.n_tokens = opt.dump_n_past;
      wp.n_past = 0;
      mi_qwen2_forward(model, kv, wp, warmup_logits.data());
      kv->n_past = opt.dump_n_past;
    }
    int32_t* rest_tokens = prompt_ids.data() + opt.dump_n_past;
    int n_rest = (int)prompt_ids.size() - opt.dump_n_past;
    std::vector<float> logits((size_t)n_rest * n_vocab);
    mi_qwen2_forward_params fp{};
    fp.tokens = rest_tokens;
    fp.n_tokens = n_rest;
    fp.n_past = opt.dump_n_past;
    mi_cgraph* g = nullptr;
    fp.out_graph = &g;
    mi_qwen2_forward(model, kv, fp, logits.data());
    mi_gallocr_stats stats = mi_gallocr_get_stats(kv->galloc);
    // Stage 6 wrap-up: theoretical_min_bytes is a pure graph-structural
    // analysis (see gallocr.h), computed here purely for reporting -- not
    // part of the real allocation path above.
    size_t theoretical_min = mi_gallocr_theoretical_min_bytes(g);
    std::printf(
        "n_ctx=%d n_tokens=%d n_past=%d peak_bytes=%zu baseline_bytes=%zu theoretical_min_bytes=%zu "
        "n_inplace_reuse=%zu n_bestfit_reuse=%zu\n",
        n_ctx, n_rest, opt.dump_n_past, stats.peak_bytes, stats.baseline_bytes, theoretical_min,
        stats.n_inplace_reuse, stats.n_bestfit_reuse);
    mi_kv_cache_free(kv);
    mi_qwen2_model_free(model);
    return 0;
  }

  if (opt.graph_stats) {
    // Same ids[0:dump_n_past] silent-warmup / ids[dump_n_past:] measured-
    // call split as --gallocr-stats, but reports a per-op-type node-count
    // histogram for the measured call's scheduled graph instead -- stage
    // 6's "how many barrier_wait calls per decode step, and how many of
    // them are on a no-compute (view/reshape/permute/transpose) node"
    // measurement (see docs/stage6.md). One barrier_wait happens per
    // g->nodes entry (threadpool.cpp's run_current_graph), so n_nodes
    // *is* the barrier count for this call.
    if (opt.dump_n_past < 0 || opt.dump_n_past >= (int)prompt_ids.size()) {
      std::fprintf(stderr, "--dump-n-past must be in [0, n_prompt_tokens)\n");
      mi_kv_cache_free(kv);
      mi_qwen2_model_free(model);
      return 1;
    }
    if (opt.dump_n_past > 0) {
      std::vector<float> warmup_logits((size_t)opt.dump_n_past * n_vocab);
      mi_qwen2_forward_params wp{};
      wp.tokens = prompt_ids.data();
      wp.n_tokens = opt.dump_n_past;
      wp.n_past = 0;
      mi_qwen2_forward(model, kv, wp, warmup_logits.data());
      kv->n_past = opt.dump_n_past;
    }
    int32_t* rest_tokens = prompt_ids.data() + opt.dump_n_past;
    int n_rest = (int)prompt_ids.size() - opt.dump_n_past;
    std::vector<float> logits((size_t)n_rest * n_vocab);
    mi_qwen2_forward_params fp{};
    fp.tokens = rest_tokens;
    fp.n_tokens = n_rest;
    fp.n_past = opt.dump_n_past;
    mi_cgraph* g = nullptr;
    fp.out_graph = &g;
    mi_qwen2_forward(model, kv, fp, logits.data());
    int histogram[MI_OP_COUNT] = {0};
    for (int i = 0; i < g->n_nodes; i++) histogram[g->nodes[i]->op]++;
    int n_nop = histogram[MI_OP_RESHAPE] + histogram[MI_OP_VIEW] + histogram[MI_OP_PERMUTE] +
                histogram[MI_OP_TRANSPOSE];
    // g->n_exec_nodes (stage 6) is the number of barrier_wait calls the
    // threadpool actually makes for this call -- n_nodes - n_nop_nodes.
    std::printf("n_ctx=%d n_tokens=%d n_past=%d n_nodes=%d n_nop_nodes=%d n_exec_nodes=%d\n", n_ctx,
                n_rest, opt.dump_n_past, g->n_nodes, n_nop, g->n_exec_nodes);
    for (int op = 0; op < MI_OP_COUNT; op++) {
      if (histogram[op] == 0) continue;
      std::printf("  %-16s %d\n", mi_op_name((mi_op)op), histogram[op]);
    }
    mi_kv_cache_free(kv);
    mi_qwen2_model_free(model);
    return 0;
  }

  if (opt.bench) {
    // One process = one (n_threads, mulmat_mode[, chunk]) measurement:
    // times the single prefill call over the whole prompt, then the
    // decode loop's forward calls (n_predict - 1 of them, since the last
    // sampled token is printed but never fed back through another
    // forward) -- see scripts/bench_threads.py, which launches this with
    // --bench once per warmup/repeat per config and parses the printed
    // line. Suppresses the normal per-token stdout stream so stdout stays
    // exactly one machine-parseable line.
    // Stage 6 wrap-up: last_logits_only=true -- real generation (this is
    // what --bench measures) only ever samples from the last prefill
    // position, so logits only needs n_vocab floats, not
    // prompt_ids.size() * n_vocab (see mi_qwen2_forward_params's doc
    // comment; this is the prefill speed change reported in docs/stage6.md).
    std::vector<float> logits((size_t)n_vocab);
    mi_qwen2_forward_params fp{};
    fp.tokens = prompt_ids.data();
    fp.n_tokens = (int)prompt_ids.size();
    fp.n_past = 0;
    fp.last_logits_only = true;
    auto t0 = std::chrono::steady_clock::now();
    mi_qwen2_forward(model, kv, fp, logits.data());
    auto t1 = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    kv->n_past = (int)prompt_ids.size();

    int32_t next_id = mi_qwen2_sample_greedy(logits.data(), n_vocab);

    double decode_ms = 0.0;
    int decode_steps = 0;
    std::vector<float> step_logits(n_vocab);
    // Stage 6 step 3c: only the *decode* loop's barrier waits are measured
    // (reset right before it starts) -- prefill's barrier behavior isn't
    // this step's target. mi_threadpool_get_barrier_stats reads 0/0 unless
    // MI_BARRIER_STATS=1 was set (see threadpool.cpp), so this is free when
    // not measuring.
    mi_threadpool_reset_barrier_stats();
    for (int step = 0; step < opt.n_predict; step++) {
      if (step + 1 == opt.n_predict) break;
      if (kv->n_past >= n_ctx) break;
      mi_qwen2_forward_params sp{};
      sp.tokens = &next_id;
      sp.n_tokens = 1;
      sp.n_past = kv->n_past;
      auto ds0 = std::chrono::steady_clock::now();
      mi_qwen2_forward(model, kv, sp, step_logits.data());
      auto ds1 = std::chrono::steady_clock::now();
      decode_ms += std::chrono::duration<double, std::milli>(ds1 - ds0).count();
      decode_steps++;
      kv->n_past += 1;
      next_id = mi_qwen2_sample_greedy(step_logits.data(), n_vocab);
    }

    double prefill_tok_per_s = prefill_ms > 0.0 ? (double)prompt_ids.size() / (prefill_ms / 1000.0) : 0.0;
    double decode_tok_per_s = decode_ms > 0.0 ? (double)decode_steps / (decode_ms / 1000.0) : 0.0;
    std::printf(
        "prefill_tokens=%zu prefill_ms=%.3f prefill_tok_per_s=%.3f decode_tokens=%d decode_ms=%.3f "
        "decode_tok_per_s=%.3f",
        prompt_ids.size(), prefill_ms, prefill_tok_per_s, decode_steps, decode_ms, decode_tok_per_s);
    uint64_t barrier_ns = 0, barrier_count = 0;
    mi_threadpool_get_barrier_stats(&barrier_ns, &barrier_count);
    if (barrier_count > 0) {
      std::printf(" barrier_wait_count=%llu avg_barrier_wait_ns=%.1f", (unsigned long long)barrier_count,
                  (double)barrier_ns / (double)barrier_count);
    }
    std::printf("\n");
    mi_kv_cache_free(kv);
    mi_qwen2_model_free(model);
    return 0;
  }

  if (!opt.dump_incremental_dir.empty()) {
    incremental_dump_ctx dc{opt.dump_incremental_dir, 0};
    mi_dump_sink sink{&on_dump_incremental, &dc};
    std::vector<float> step_logits(n_vocab);
    for (size_t i = 0; i < prompt_ids.size(); i++) {
      dc.step = (int)i;
      mi_qwen2_forward_params fp{};
      fp.tokens = &prompt_ids[i];
      fp.n_tokens = 1;
      fp.n_past = (int)i;
      fp.dump = &sink;
      mi_qwen2_forward(model, kv, fp, step_logits.data());
    }
    mi_kv_cache_free(kv);
    mi_qwen2_model_free(model);
    return 0;
  }

  // Stage 6 wrap-up: same last_logits_only optimization as --bench above --
  // greedy generation only ever samples from the last prefill position.
  std::vector<float> logits((size_t)n_vocab);
  mi_qwen2_forward_params fp{};
  fp.tokens = prompt_ids.data();
  fp.n_tokens = (int)prompt_ids.size();
  fp.n_past = 0;
  fp.last_logits_only = true;
  mi_qwen2_forward(model, kv, fp, logits.data());
  kv->n_past = (int)prompt_ids.size();

  int32_t next_id = mi_qwen2_sample_greedy(logits.data(), n_vocab);

  for (int step = 0; step < opt.n_predict; step++) {
    std::printf("%d\n", next_id);
    std::fflush(stdout);
    std::string piece = mi_vocab_token_to_text(model->vocab, next_id);
    std::fwrite(piece.data(), 1, piece.size(), stderr);
    std::fflush(stderr);

    if (step + 1 == opt.n_predict) break;
    if (kv->n_past >= n_ctx) {
      std::fprintf(stderr, "\n[hit -c/--ctx limit, stopping]\n");
      break;
    }

    std::vector<float> step_logits(n_vocab);
    mi_qwen2_forward_params sp{};
    sp.tokens = &next_id;
    sp.n_tokens = 1;
    sp.n_past = kv->n_past;
    mi_qwen2_forward(model, kv, sp, step_logits.data());
    kv->n_past += 1;
    next_id = mi_qwen2_sample_greedy(step_logits.data(), n_vocab);
  }
  std::fprintf(stderr, "\n");

  mi_kv_cache_free(kv);
  mi_qwen2_model_free(model);
  return 0;
}
