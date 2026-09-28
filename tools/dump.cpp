// tools/dump: prints a GGUF file's metadata and tensor table in a stable,
// line-oriented, machine-parseable format so scripts/compare_gguf_dump.py
// can diff it against Python's `gguf` library item-by-item (SPEC.md
// forbids eyeballing this kind of comparison). Also supports decoding a
// file of token-id sequences through the vocab, for
// scripts/compare_decode.py to check against HF's tokenizer.decode().
//
// Output format (metadata):
//   GGUF_VERSION <version>
//   ALIGNMENT <alignment>
//   N_KV <count>
//   KV <index> <key> <type_name> <value>
//   ARR <index> <key> <elem_type_name> <count> <elem0> <elem1> ...
//   N_TENSORS <count>
//   TENSOR <index> <name> <type_name> <n_dims> <ne0> <ne1> <ne2> <ne3> <nbytes>
//
// Every field on a line is tab-separated. String values/elements are
// escaped (backslash, \n, \t, \r) so a value can never introduce a stray
// tab or newline into the line-oriented format; see escape() below.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gguf.h"

namespace {

std::string escape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\r':
        out += "\\r";
        break;
      default:
        out.push_back(c);
    }
  }
  return out;
}

const char* gguf_type_name(mi_gguf_type t) {
  switch (t) {
    case MI_GGUF_TYPE_UINT8:
      return "UINT8";
    case MI_GGUF_TYPE_INT8:
      return "INT8";
    case MI_GGUF_TYPE_UINT16:
      return "UINT16";
    case MI_GGUF_TYPE_INT16:
      return "INT16";
    case MI_GGUF_TYPE_UINT32:
      return "UINT32";
    case MI_GGUF_TYPE_INT32:
      return "INT32";
    case MI_GGUF_TYPE_FLOAT32:
      return "FLOAT32";
    case MI_GGUF_TYPE_BOOL:
      return "BOOL";
    case MI_GGUF_TYPE_STRING:
      return "STRING";
    case MI_GGUF_TYPE_ARRAY:
      return "ARRAY";
    case MI_GGUF_TYPE_UINT64:
      return "UINT64";
    case MI_GGUF_TYPE_INT64:
      return "INT64";
    case MI_GGUF_TYPE_FLOAT64:
      return "FLOAT64";
  }
  return "UNKNOWN";
}

int dump_metadata(const mi_gguf_context* ctx) {
  std::printf("GGUF_VERSION\t%u\n", mi_gguf_version(ctx));
  std::printf("ALIGNMENT\t%zu\n", mi_gguf_alignment(ctx));

  size_t n_kv = mi_gguf_n_kv(ctx);
  std::printf("N_KV\t%zu\n", n_kv);
  for (size_t i = 0; i < n_kv; i++) {
    const char* key = mi_gguf_kv_key(ctx, i);
    mi_gguf_type type = mi_gguf_kv_type(ctx, i);
    if (type == MI_GGUF_TYPE_ARRAY) {
      mi_gguf_type elem_type = mi_gguf_kv_arr_type(ctx, i);
      size_t n = mi_gguf_kv_arr_len(ctx, i);
      std::printf("ARR\t%zu\t%s\t%s\t%zu", i, key, gguf_type_name(elem_type), n);
      for (size_t j = 0; j < n; j++) {
        std::string elem;
        mi_gguf_kv_arr_elem_to_string(ctx, i, j, &elem);
        std::printf("\t%s", escape(elem).c_str());
      }
      std::printf("\n");
    } else {
      std::string val;
      mi_gguf_kv_scalar_to_string(ctx, i, &val);
      std::printf("KV\t%zu\t%s\t%s\t%s\n", i, key, gguf_type_name(type), escape(val).c_str());
    }
  }

  size_t n_tensors = mi_gguf_n_tensors(ctx);
  std::printf("N_TENSORS\t%zu\n", n_tensors);
  for (size_t i = 0; i < n_tensors; i++) {
    const mi_gguf_tensor_info* t = mi_gguf_get_tensor(ctx, i);
    std::printf("TENSOR\t%zu\t%s\t%s\t%d\t%lld\t%lld\t%lld\t%lld\t%zu\n", i, t->name,
                mi_type_name(t->type), t->n_dims, (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3], t->nbytes);
  }
  return 0;
}

// One line of whitespace-separated int32 ids in -> one line of decoded
// UTF-8 text out (concatenated across the whole line, since a multi-byte
// character's pieces must be joined before they're valid UTF-8).
int decode_ids_file(const mi_gguf_context* ctx, const std::string& path) {
  std::string err;
  mi_vocab* vocab = mi_vocab_load(ctx, &err);
  if (!vocab) {
    std::fprintf(stderr, "dump: failed to load vocab: %s\n", err.c_str());
    return 1;
  }
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "dump: failed to open ids file '%s'\n", path.c_str());
    mi_vocab_free(vocab);
    return 1;
  }
  std::string line;
  while (std::getline(in, line)) {
    std::vector<int32_t> ids;
    std::istringstream iss(line);
    int32_t id;
    while (iss >> id) ids.push_back(id);
    std::string text = mi_vocab_decode(vocab, ids.data(), ids.size());
    // Decoded text (e.g. a chat-template sequence) commonly contains
    // embedded newlines/tabs of its own; escape them the same way the
    // metadata dump does so one input line always maps to exactly one
    // output line.
    std::fputs(escape(text).c_str(), stdout);
    std::fputc('\n', stdout);
  }
  mi_vocab_free(vocab);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.gguf> [--decode-ids <ids_file>]\n", argv[0]);
    return 2;
  }
  const char* path = argv[1];
  std::string err;
  mi_gguf_context* ctx = mi_gguf_open(path, &err);
  if (!ctx) {
    std::fprintf(stderr, "dump: %s\n", err.c_str());
    return 1;
  }

  int rc = 0;
  if (argc >= 4 && std::strcmp(argv[2], "--decode-ids") == 0) {
    rc = decode_ids_file(ctx, argv[3]);
  } else {
    rc = dump_metadata(ctx);
  }

  mi_gguf_close(ctx);
  return rc;
}
