#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Byte-cursor reader over the mmap'd file.
// ---------------------------------------------------------------------------
class mi_cursor {
 public:
  mi_cursor(const uint8_t* base, size_t size) : p_(base), end_(base + size) {}

  bool ok() const { return ok_; }
  size_t offset(const uint8_t* base) const { return (size_t)(p_ - base); }

  template <typename T>
  bool read(T* out) {
    if (!ok_ || (size_t)(end_ - p_) < sizeof(T)) {
      ok_ = false;
      return false;
    }
    std::memcpy(out, p_, sizeof(T));
    p_ += sizeof(T);
    return true;
  }

  bool read_bytes(void* out, size_t n) {
    if (!ok_ || (size_t)(end_ - p_) < n) {
      ok_ = false;
      return false;
    }
    std::memcpy(out, p_, n);
    p_ += n;
    return true;
  }

  bool read_string(std::string* out) {
    uint64_t len = 0;
    if (!read(&len) || (size_t)(end_ - p_) < len) {
      ok_ = false;
      return false;
    }
    out->assign(reinterpret_cast<const char*>(p_), (size_t)len);
    p_ += len;
    return true;
  }

 private:
  const uint8_t* p_;
  const uint8_t* end_;
  bool ok_ = true;
};

// ---------------------------------------------------------------------------
// A parsed metadata value: exactly one scalar field or one array vector is
// populated, per `type`/`arr_type`.
// ---------------------------------------------------------------------------
struct mi_gguf_value {
  mi_gguf_type type = MI_GGUF_TYPE_UINT8;
  mi_gguf_type arr_type = MI_GGUF_TYPE_UINT8;  // valid only when type == ARRAY

  uint64_t u = 0;
  int64_t i = 0;
  double f = 0;
  bool b = false;
  std::string s;

  std::vector<int64_t> arr_i;  // all integer array kinds (signed & unsigned)
  std::vector<double> arr_f;
  std::vector<bool> arr_b;
  std::vector<std::string> arr_s;
};

struct mi_gguf_kv_entry {
  std::string key;
  mi_gguf_value value;
};

size_t arr_len(const mi_gguf_value& v) {
  switch (v.arr_type) {
    case MI_GGUF_TYPE_STRING:
      return v.arr_s.size();
    case MI_GGUF_TYPE_BOOL:
      return v.arr_b.size();
    case MI_GGUF_TYPE_FLOAT32:
    case MI_GGUF_TYPE_FLOAT64:
      return v.arr_f.size();
    default:
      return v.arr_i.size();
  }
}

// Reads one value of scalar type `type` (STRING included) into `v`'s
// matching scalar field. Does not handle ARRAY; see mi_read_value.
bool mi_read_scalar(mi_cursor& c, mi_gguf_type type, mi_gguf_value* v) {
  switch (type) {
    case MI_GGUF_TYPE_UINT8: {
      uint8_t x;
      if (!c.read(&x)) return false;
      v->u = x;
      v->i = x;
      return true;
    }
    case MI_GGUF_TYPE_INT8: {
      int8_t x;
      if (!c.read(&x)) return false;
      v->i = x;
      v->u = (uint64_t)(int64_t)x;
      return true;
    }
    case MI_GGUF_TYPE_UINT16: {
      uint16_t x;
      if (!c.read(&x)) return false;
      v->u = x;
      v->i = x;
      return true;
    }
    case MI_GGUF_TYPE_INT16: {
      int16_t x;
      if (!c.read(&x)) return false;
      v->i = x;
      v->u = (uint64_t)(int64_t)x;
      return true;
    }
    case MI_GGUF_TYPE_UINT32: {
      uint32_t x;
      if (!c.read(&x)) return false;
      v->u = x;
      v->i = x;
      return true;
    }
    case MI_GGUF_TYPE_INT32: {
      int32_t x;
      if (!c.read(&x)) return false;
      v->i = x;
      v->u = (uint64_t)(int64_t)x;
      return true;
    }
    case MI_GGUF_TYPE_FLOAT32: {
      float x;
      if (!c.read(&x)) return false;
      v->f = x;
      return true;
    }
    case MI_GGUF_TYPE_BOOL: {
      uint8_t x;
      if (!c.read(&x)) return false;
      v->b = x != 0;
      return true;
    }
    case MI_GGUF_TYPE_UINT64: {
      uint64_t x;
      if (!c.read(&x)) return false;
      v->u = x;
      v->i = (int64_t)x;
      return true;
    }
    case MI_GGUF_TYPE_INT64: {
      int64_t x;
      if (!c.read(&x)) return false;
      v->i = x;
      v->u = (uint64_t)x;
      return true;
    }
    case MI_GGUF_TYPE_FLOAT64: {
      double x;
      if (!c.read(&x)) return false;
      v->f = x;
      return true;
    }
    case MI_GGUF_TYPE_STRING:
      return c.read_string(&v->s);
    default:
      return false;
  }
}

// Reads a full metadata value (scalar or array) whose raw type id was just
// read from the file.
bool mi_read_value(mi_cursor& c, uint32_t raw_type, mi_gguf_value* v) {
  if (raw_type != MI_GGUF_TYPE_ARRAY) {
    v->type = (mi_gguf_type)raw_type;
    return mi_read_scalar(c, v->type, v);
  }
  uint32_t elem_type_raw = 0;
  uint64_t len = 0;
  if (!c.read(&elem_type_raw) || !c.read(&len)) return false;
  // Nested arrays are part of the GGUF value-type space in principle but no
  // real writer (llama.cpp's gguf-py included) ever emits one; treat as
  // unsupported rather than guessing a recursive representation nothing can
  // exercise.
  if (elem_type_raw == MI_GGUF_TYPE_ARRAY) return false;

  v->type = MI_GGUF_TYPE_ARRAY;
  v->arr_type = (mi_gguf_type)elem_type_raw;
  for (uint64_t i = 0; i < len; i++) {
    mi_gguf_value tmp;
    if (!mi_read_scalar(c, v->arr_type, &tmp)) return false;
    switch (v->arr_type) {
      case MI_GGUF_TYPE_STRING:
        v->arr_s.push_back(std::move(tmp.s));
        break;
      case MI_GGUF_TYPE_BOOL:
        v->arr_b.push_back(tmp.b);
        break;
      case MI_GGUF_TYPE_FLOAT32:
      case MI_GGUF_TYPE_FLOAT64:
        v->arr_f.push_back(tmp.f);
        break;
      default:
        v->arr_i.push_back(tmp.i);
        break;
    }
  }
  return true;
}

size_t align_up(size_t x, size_t a) {
  size_t r = x % a;
  return r == 0 ? x : x + (a - r);
}

}  // namespace

// ---------------------------------------------------------------------------
// mi_gguf_context
// ---------------------------------------------------------------------------
struct mi_gguf_context {
  int fd = -1;
  void* map = nullptr;
  size_t map_size = 0;
  uint32_t version = 0;
  size_t alignment = 32;
  std::vector<mi_gguf_kv_entry> kv;
  std::vector<mi_gguf_tensor_info> tensors;
  std::unordered_map<std::string, size_t> tensor_index;
};

namespace {
const mi_gguf_kv_entry* find_kv(const mi_gguf_context* ctx, const char* key) {
  for (const auto& e : ctx->kv) {
    if (e.key == key) return &e;
  }
  return nullptr;
}
}  // namespace

mi_gguf_context* mi_gguf_open(const char* path, std::string* err) {
  auto set_err = [&](const std::string& msg) {
    if (err) *err = msg;
  };

  int fd = ::open(path, O_RDONLY);
  if (fd < 0) {
    set_err(std::string("failed to open '") + path + "': " + std::strerror(errno));
    return nullptr;
  }
  struct stat st{};
  if (fstat(fd, &st) != 0) {
    ::close(fd);
    set_err(std::string("fstat failed: ") + std::strerror(errno));
    return nullptr;
  }
  size_t size = (size_t)st.st_size;
  if (size < 24) {  // magic(4) + version(4) + tensor_count(8) + kv_count(8)
    ::close(fd);
    set_err("file too small to be a GGUF file");
    return nullptr;
  }
  void* map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    ::close(fd);
    set_err(std::string("mmap failed: ") + std::strerror(errno));
    return nullptr;
  }
  auto cleanup = [&]() {
    munmap(map, size);
    ::close(fd);
  };

  const uint8_t* base = static_cast<const uint8_t*>(map);
  mi_cursor c(base, size);

  char magic[4];
  uint32_t version = 0;
  uint64_t tensor_count = 0, kv_count = 0;
  if (!c.read_bytes(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
    cleanup();
    set_err("not a GGUF file (bad magic)");
    return nullptr;
  }
  if (!c.read(&version) || version != 3) {
    cleanup();
    set_err("unsupported GGUF version (mini-infer only implements v3)");
    return nullptr;
  }
  if (!c.read(&tensor_count) || !c.read(&kv_count)) {
    cleanup();
    set_err("truncated GGUF header");
    return nullptr;
  }

  std::vector<mi_gguf_kv_entry> kv;
  kv.reserve((size_t)kv_count);
  for (uint64_t i = 0; i < kv_count; i++) {
    mi_gguf_kv_entry e;
    uint32_t raw_type = 0;
    if (!c.read_string(&e.key) || !c.read(&raw_type) || !mi_read_value(c, raw_type, &e.value)) {
      cleanup();
      set_err("truncated or corrupt metadata table");
      return nullptr;
    }
    kv.push_back(std::move(e));
  }

  size_t alignment = 32;
  for (const auto& e : kv) {
    if (e.key == "general.alignment" && e.value.type == MI_GGUF_TYPE_UINT32) {
      alignment = (size_t)e.value.u;
    }
  }
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    cleanup();
    set_err("invalid general.alignment (must be a non-zero power of two)");
    return nullptr;
  }

  std::vector<mi_gguf_tensor_info> tensors;
  tensors.reserve((size_t)tensor_count);
  for (uint64_t i = 0; i < tensor_count; i++) {
    std::string name;
    uint32_t n_dims = 0;
    if (!c.read_string(&name) || !c.read(&n_dims) || n_dims == 0 || n_dims > MI_MAX_DIMS) {
      cleanup();
      set_err("corrupt tensor info table");
      return nullptr;
    }
    if (name.size() >= sizeof(mi_gguf_tensor_info::name)) {
      cleanup();
      set_err("tensor name too long: " + name);
      return nullptr;
    }

    mi_gguf_tensor_info info{};
    std::strncpy(info.name, name.c_str(), sizeof(info.name) - 1);
    info.n_dims = (int)n_dims;
    for (int d = 0; d < MI_MAX_DIMS; d++) info.ne[d] = 1;

    bool dims_ok = true;
    for (uint32_t d = 0; d < n_dims; d++) {
      uint64_t dim = 0;
      if (!c.read(&dim)) {
        dims_ok = false;
        break;
      }
      info.ne[d] = (int64_t)dim;
    }
    uint32_t raw_type = 0;
    uint64_t off = 0;
    if (!dims_ok || !c.read(&raw_type) || !c.read(&off)) {
      cleanup();
      set_err("corrupt tensor info table");
      return nullptr;
    }

    info.gguf_type_id = raw_type;
    switch (raw_type) {
      case 0:
        info.type = MI_TYPE_F32;
        break;
      case 1:
        info.type = MI_TYPE_F16;
        break;
      case 8:
        info.type = MI_TYPE_Q8_0;
        break;
      default:
        cleanup();
        set_err("unsupported tensor type: name=" + name +
                " type_id=" + std::to_string(raw_type));
        return nullptr;
    }
    info.offset = (size_t)off;
    tensors.push_back(info);
  }

  if (!c.ok()) {
    cleanup();
    set_err("truncated GGUF file while parsing header tables");
    return nullptr;
  }

  size_t data_offset = align_up(c.offset(base), alignment);
  if (data_offset > size) {
    cleanup();
    set_err("data section start exceeds file size");
    return nullptr;
  }

  for (auto& t : tensors) {
    int64_t n_elems = 1;
    for (int d = 0; d < MI_MAX_DIMS; d++) n_elems *= t.ne[d];
    t.nbytes = mi_row_size(t.type, n_elems);
    size_t abs_off = data_offset + t.offset;
    if (abs_off + t.nbytes > size) {
      cleanup();
      set_err(std::string("tensor '") + t.name + "' data extends past end of file");
      return nullptr;
    }
    t.data = base + abs_off;
  }

  auto* ctx = new mi_gguf_context();
  ctx->fd = fd;
  ctx->map = map;
  ctx->map_size = size;
  ctx->version = version;
  ctx->alignment = alignment;
  ctx->kv = std::move(kv);
  ctx->tensors = std::move(tensors);
  for (size_t i = 0; i < ctx->tensors.size(); i++) {
    ctx->tensor_index.emplace(ctx->tensors[i].name, i);
  }
  return ctx;
}

void mi_gguf_close(mi_gguf_context* ctx) {
  if (!ctx) return;
  if (ctx->map) munmap(ctx->map, ctx->map_size);
  if (ctx->fd >= 0) ::close(ctx->fd);
  delete ctx;
}

uint32_t mi_gguf_version(const mi_gguf_context* ctx) { return ctx->version; }
size_t mi_gguf_alignment(const mi_gguf_context* ctx) { return ctx->alignment; }

size_t mi_gguf_n_tensors(const mi_gguf_context* ctx) { return ctx->tensors.size(); }

const mi_gguf_tensor_info* mi_gguf_get_tensor(const mi_gguf_context* ctx, size_t i) {
  return i < ctx->tensors.size() ? &ctx->tensors[i] : nullptr;
}

const mi_gguf_tensor_info* mi_gguf_find_tensor(const mi_gguf_context* ctx, const char* name) {
  auto it = ctx->tensor_index.find(name);
  return it == ctx->tensor_index.end() ? nullptr : &ctx->tensors[it->second];
}

size_t mi_gguf_n_kv(const mi_gguf_context* ctx) { return ctx->kv.size(); }
const char* mi_gguf_kv_key(const mi_gguf_context* ctx, size_t i) { return ctx->kv[i].key.c_str(); }
mi_gguf_type mi_gguf_kv_type(const mi_gguf_context* ctx, size_t i) { return ctx->kv[i].value.type; }
mi_gguf_type mi_gguf_kv_arr_type(const mi_gguf_context* ctx, size_t i) {
  return ctx->kv[i].value.arr_type;
}
size_t mi_gguf_kv_arr_len(const mi_gguf_context* ctx, size_t i) { return arr_len(ctx->kv[i].value); }

bool mi_gguf_get_u32(const mi_gguf_context* ctx, const char* key, uint32_t* out) {
  const auto* e = find_kv(ctx, key);
  if (!e || e->value.type != MI_GGUF_TYPE_UINT32) return false;
  *out = (uint32_t)e->value.u;
  return true;
}

bool mi_gguf_get_i32(const mi_gguf_context* ctx, const char* key, int32_t* out) {
  const auto* e = find_kv(ctx, key);
  if (!e || e->value.type != MI_GGUF_TYPE_INT32) return false;
  *out = (int32_t)e->value.i;
  return true;
}

bool mi_gguf_get_f32(const mi_gguf_context* ctx, const char* key, float* out) {
  const auto* e = find_kv(ctx, key);
  if (!e || e->value.type != MI_GGUF_TYPE_FLOAT32) return false;
  *out = (float)e->value.f;
  return true;
}

bool mi_gguf_get_bool(const mi_gguf_context* ctx, const char* key, bool* out) {
  const auto* e = find_kv(ctx, key);
  if (!e || e->value.type != MI_GGUF_TYPE_BOOL) return false;
  *out = e->value.b;
  return true;
}

bool mi_gguf_get_str(const mi_gguf_context* ctx, const char* key, std::string* out) {
  const auto* e = find_kv(ctx, key);
  if (!e || e->value.type != MI_GGUF_TYPE_STRING) return false;
  *out = e->value.s;
  return true;
}

bool mi_gguf_get_arr_n(const mi_gguf_context* ctx, const char* key, size_t* n) {
  const auto* e = find_kv(ctx, key);
  if (!e || e->value.type != MI_GGUF_TYPE_ARRAY) return false;
  *n = arr_len(e->value);
  return true;
}

bool mi_gguf_get_arr_str(const mi_gguf_context* ctx, const char* key, size_t idx,
                          std::string* out) {
  const auto* e = find_kv(ctx, key);
  if (!e || e->value.type != MI_GGUF_TYPE_ARRAY || e->value.arr_type != MI_GGUF_TYPE_STRING) {
    return false;
  }
  if (idx >= e->value.arr_s.size()) return false;
  *out = e->value.arr_s[idx];
  return true;
}

namespace {
std::string format_scalar(mi_gguf_type type, uint64_t u, int64_t i, double f, bool b,
                           const std::string& s) {
  char buf[64];
  switch (type) {
    case MI_GGUF_TYPE_UINT8:
    case MI_GGUF_TYPE_UINT16:
    case MI_GGUF_TYPE_UINT32:
    case MI_GGUF_TYPE_UINT64:
      std::snprintf(buf, sizeof(buf), "%llu", (unsigned long long)u);
      return buf;
    case MI_GGUF_TYPE_INT8:
    case MI_GGUF_TYPE_INT16:
    case MI_GGUF_TYPE_INT32:
    case MI_GGUF_TYPE_INT64:
      std::snprintf(buf, sizeof(buf), "%lld", (long long)i);
      return buf;
    case MI_GGUF_TYPE_FLOAT32:
      // 9 significant digits round-trips any float32 exactly.
      std::snprintf(buf, sizeof(buf), "%.9g", f);
      return buf;
    case MI_GGUF_TYPE_FLOAT64:
      std::snprintf(buf, sizeof(buf), "%.17g", f);
      return buf;
    case MI_GGUF_TYPE_BOOL:
      return b ? "true" : "false";
    case MI_GGUF_TYPE_STRING:
      return s;
    default:
      return std::string();
  }
}
}  // namespace

bool mi_gguf_kv_scalar_to_string(const mi_gguf_context* ctx, size_t i, std::string* out) {
  if (i >= ctx->kv.size()) return false;
  const auto& v = ctx->kv[i].value;
  if (v.type == MI_GGUF_TYPE_ARRAY) return false;
  *out = format_scalar(v.type, v.u, v.i, v.f, v.b, v.s);
  return true;
}

bool mi_gguf_kv_arr_elem_to_string(const mi_gguf_context* ctx, size_t i, size_t idx,
                                    std::string* out) {
  if (i >= ctx->kv.size()) return false;
  const auto& v = ctx->kv[i].value;
  if (v.type != MI_GGUF_TYPE_ARRAY) return false;
  switch (v.arr_type) {
    case MI_GGUF_TYPE_STRING:
      if (idx >= v.arr_s.size()) return false;
      *out = v.arr_s[idx];
      return true;
    case MI_GGUF_TYPE_BOOL:
      if (idx >= v.arr_b.size()) return false;
      *out = v.arr_b[idx] ? "true" : "false";
      return true;
    case MI_GGUF_TYPE_FLOAT32:
    case MI_GGUF_TYPE_FLOAT64:
      if (idx >= v.arr_f.size()) return false;
      *out = format_scalar(v.arr_type, 0, 0, v.arr_f[idx], false, "");
      return true;
    default:
      if (idx >= v.arr_i.size()) return false;
      *out = format_scalar(v.arr_type, (uint64_t)v.arr_i[idx], v.arr_i[idx], 0, false, "");
      return true;
  }
}

bool mi_gguf_get_arr_i32(const mi_gguf_context* ctx, const char* key, size_t idx, int32_t* out) {
  const auto* e = find_kv(ctx, key);
  if (!e || e->value.type != MI_GGUF_TYPE_ARRAY) return false;
  if (e->value.arr_type == MI_GGUF_TYPE_BOOL) {
    if (idx >= e->value.arr_b.size()) return false;
    *out = e->value.arr_b[idx] ? 1 : 0;
    return true;
  }
  if (e->value.arr_type == MI_GGUF_TYPE_STRING || e->value.arr_type == MI_GGUF_TYPE_FLOAT32 ||
      e->value.arr_type == MI_GGUF_TYPE_FLOAT64) {
    return false;
  }
  if (idx >= e->value.arr_i.size()) return false;
  *out = (int32_t)e->value.arr_i[idx];
  return true;
}

// ---------------------------------------------------------------------------
// Vocab: byte-level BPE id -> text decoding.
// ---------------------------------------------------------------------------
namespace {

// GPT2/byte-level-BPE's bytes_to_unicode(): maps every one of the 256 byte
// values to a distinct Unicode codepoint (printable ASCII/Latin-1 bytes map
// to themselves; everything else — including the common case of space,
// byte 0x20 -> U+0120 'Ġ' — maps to a codepoint starting at 256). GGUF
// stores NORMAL-type BPE tokens already run through this mapping, so
// decoding means inverting it.
struct mi_byte_unicode_tables {
  uint32_t byte_to_cp[256];
  std::unordered_map<uint32_t, uint8_t> cp_to_byte;

  mi_byte_unicode_tables() {
    bool in_bs[256] = {false};
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    for (int b : bs) in_bs[b] = true;
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
      if (!in_bs[b]) {
        bs.push_back(b);
        cs.push_back(256 + n);
        n++;
      }
    }
    for (size_t i = 0; i < bs.size(); i++) byte_to_cp[bs[i]] = (uint32_t)cs[i];
    for (int b = 0; b < 256; b++) cp_to_byte[byte_to_cp[b]] = (uint8_t)b;
  }
};

const mi_byte_unicode_tables& byte_unicode_tables() {
  static mi_byte_unicode_tables t;
  return t;
}

void utf8_append(std::string* out, uint32_t cp) {
  if (cp < 0x80) {
    out->push_back((char)cp);
  } else if (cp < 0x800) {
    out->push_back((char)(0xC0 | (cp >> 6)));
    out->push_back((char)(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out->push_back((char)(0xE0 | (cp >> 12)));
    out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    out->push_back((char)(0xF0 | (cp >> 18)));
    out->push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back((char)(0x80 | (cp & 0x3F)));
  }
}

// Decodes one UTF-8 codepoint at s[i], advancing i past it.
bool utf8_next(const std::string& s, size_t& i, uint32_t* cp) {
  if (i >= s.size()) return false;
  unsigned char c0 = (unsigned char)s[i];
  size_t len;
  uint32_t val;
  if ((c0 & 0x80) == 0) {
    len = 1;
    val = c0;
  } else if ((c0 & 0xE0) == 0xC0) {
    len = 2;
    val = c0 & 0x1F;
  } else if ((c0 & 0xF0) == 0xE0) {
    len = 3;
    val = c0 & 0x0F;
  } else if ((c0 & 0xF8) == 0xF0) {
    len = 4;
    val = c0 & 0x07;
  } else {
    return false;
  }
  if (i + len > s.size()) return false;
  for (size_t k = 1; k < len; k++) {
    unsigned char ck = (unsigned char)s[i + k];
    if ((ck & 0xC0) != 0x80) return false;
    val = (val << 6) | (ck & 0x3F);
  }
  *cp = val;
  i += len;
  return true;
}

std::string decode_byte_level_token(const std::string& s) {
  const auto& tab = byte_unicode_tables();
  std::string out;
  size_t i = 0;
  uint32_t cp;
  while (utf8_next(s, i, &cp)) {
    auto it = tab.cp_to_byte.find(cp);
    if (it != tab.cp_to_byte.end()) {
      out.push_back((char)it->second);
    } else {
      // Not one of the 256 byte-mapped codepoints; shouldn't happen for a
      // real GPT2/Qwen2 NORMAL token, but pass it through as UTF-8 rather
      // than silently dropping data.
      utf8_append(&out, cp);
    }
  }
  return out;
}

enum mi_gguf_token_type {
  MI_GGUF_TOKEN_NORMAL = 1,
  MI_GGUF_TOKEN_UNKNOWN = 2,
  MI_GGUF_TOKEN_CONTROL = 3,
  MI_GGUF_TOKEN_USER_DEFINED = 4,
  MI_GGUF_TOKEN_UNUSED = 5,
  MI_GGUF_TOKEN_BYTE = 6,
};

// "<0xHH>" byte-fallback tokens (used by some SentencePiece vocabs; absent
// from Qwen2's GPT2-style vocab but cheap to support correctly).
bool try_parse_byte_token(const std::string& s, uint8_t* out) {
  if (s.size() != 6 || s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>') return false;
  auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  int hi = hex(s[3]), lo = hex(s[4]);
  if (hi < 0 || lo < 0) return false;
  *out = (uint8_t)((hi << 4) | lo);
  return true;
}

}  // namespace

struct mi_vocab {
  // Precomputed at load time: decoded[id] is the raw byte sequence that id
  // contributes to a decoded sequence. For NORMAL tokens this is already
  // the reversed byte-level BPE mapping; concatenating decoded[] for a
  // sequence of ids reconstructs the original UTF-8 bytes even where a
  // single multi-byte character was split across multiple ids.
  std::vector<std::string> decoded;
};

mi_vocab* mi_vocab_load(const mi_gguf_context* ctx, std::string* err) {
  const mi_gguf_kv_entry* toks_kv = find_kv(ctx, "tokenizer.ggml.tokens");
  if (!toks_kv || toks_kv->value.type != MI_GGUF_TYPE_ARRAY ||
      toks_kv->value.arr_type != MI_GGUF_TYPE_STRING) {
    if (err) *err = "missing or malformed tokenizer.ggml.tokens";
    return nullptr;
  }
  const mi_gguf_kv_entry* types_kv = find_kv(ctx, "tokenizer.ggml.token_type");
  const std::vector<int64_t>* types = nullptr;
  if (types_kv && types_kv->value.type == MI_GGUF_TYPE_ARRAY &&
      types_kv->value.arr_type != MI_GGUF_TYPE_STRING &&
      types_kv->value.arr_type != MI_GGUF_TYPE_BOOL &&
      types_kv->value.arr_type != MI_GGUF_TYPE_FLOAT32 &&
      types_kv->value.arr_type != MI_GGUF_TYPE_FLOAT64) {
    types = &types_kv->value.arr_i;
  }

  auto* vocab = new mi_vocab();
  size_t n = toks_kv->value.arr_s.size();
  vocab->decoded.resize(n);
  for (size_t id = 0; id < n; id++) {
    int type =
        (types && id < types->size()) ? (int)(*types)[id] : (int)MI_GGUF_TOKEN_NORMAL;
    const std::string& s = toks_kv->value.arr_s[id];
    uint8_t single_byte;
    if (type == MI_GGUF_TOKEN_BYTE && try_parse_byte_token(s, &single_byte)) {
      vocab->decoded[id] = std::string(1, (char)single_byte);
    } else if (type == MI_GGUF_TOKEN_NORMAL) {
      vocab->decoded[id] = decode_byte_level_token(s);
    } else {
      vocab->decoded[id] = s;  // control/user-defined/unknown/unused: literal text
    }
  }
  return vocab;
}

void mi_vocab_free(mi_vocab* vocab) { delete vocab; }

size_t mi_vocab_size(const mi_vocab* vocab) { return vocab->decoded.size(); }

std::string mi_vocab_token_to_text(const mi_vocab* vocab, int32_t id) {
  if (id < 0 || (size_t)id >= vocab->decoded.size()) return std::string();
  return vocab->decoded[id];
}

std::string mi_vocab_decode(const mi_vocab* vocab, const int32_t* ids, size_t n_ids) {
  std::string out;
  for (size_t i = 0; i < n_ids; i++) out += mi_vocab_token_to_text(vocab, ids[i]);
  return out;
}
