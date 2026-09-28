// mini-infer: GGUF v3 file parsing (stage 0).
//
// mi_gguf_open() mmaps the file and parses the header, metadata key-value
// table, and tensor info table; tensor data itself is never copied — each
// mi_gguf_tensor_info::data points directly into the mmap at the tensor's
// (alignment-padded) offset. No tokenizer *encoding* lives here: this
// module only reads the stored vocab and decodes ids back to text (see
// mi_vocab_* below). Encoding is done out of process by scripts/encode.py.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "tensor.h"

// ---------------------------------------------------------------------------
// Metadata value types, per the GGUF spec.
// ---------------------------------------------------------------------------
enum mi_gguf_type {
    MI_GGUF_TYPE_UINT8 = 0,
    MI_GGUF_TYPE_INT8 = 1,
    MI_GGUF_TYPE_UINT16 = 2,
    MI_GGUF_TYPE_INT16 = 3,
    MI_GGUF_TYPE_UINT32 = 4,
    MI_GGUF_TYPE_INT32 = 5,
    MI_GGUF_TYPE_FLOAT32 = 6,
    MI_GGUF_TYPE_BOOL = 7,
    MI_GGUF_TYPE_STRING = 8,
    MI_GGUF_TYPE_ARRAY = 9,
    MI_GGUF_TYPE_UINT64 = 10,
    MI_GGUF_TYPE_INT64 = 11,
    MI_GGUF_TYPE_FLOAT64 = 12,
};

struct mi_gguf_tensor_info {
    char name[256];
    int n_dims;
    // GGUF order: ne[0] is the innermost (fastest-varying) dimension.
    int64_t ne[MI_MAX_DIMS];
    mi_type type;    // mapped from the GGUF ggml_type id; see mi_gguf_open
    uint32_t gguf_type_id;  // raw id, kept for error messages / dump tool
    size_t offset;   // byte offset into the data section (before alignment padding)
    const void* data;  // == mmap_base + data_section_start + aligned(offset)
    size_t nbytes;
};

struct mi_gguf_context;  // opaque: owns the mmap and parsed tables

// Returns nullptr on I/O error, bad magic/version, or an unsupported tensor
// type; in the unsupported-type case, writes the offending tensor's name
// and raw type id into *err (caller should print it and exit, per SPEC.md:
// "遇到其他类型报出张量名和类型后退出").
mi_gguf_context* mi_gguf_open(const char* path, std::string* err);
void mi_gguf_close(mi_gguf_context* ctx);

uint32_t mi_gguf_version(const mi_gguf_context* ctx);
size_t mi_gguf_alignment(const mi_gguf_context* ctx);  // general.alignment, default 32

// -- tensors ------------------------------------------------------------
size_t mi_gguf_n_tensors(const mi_gguf_context* ctx);
const mi_gguf_tensor_info* mi_gguf_get_tensor(const mi_gguf_context* ctx, size_t i);
// nullptr if not present.
const mi_gguf_tensor_info* mi_gguf_find_tensor(const mi_gguf_context* ctx, const char* name);

// -- metadata key-value ---------------------------------------------------
size_t mi_gguf_n_kv(const mi_gguf_context* ctx);
const char* mi_gguf_kv_key(const mi_gguf_context* ctx, size_t i);
mi_gguf_type mi_gguf_kv_type(const mi_gguf_context* ctx, size_t i);
// For MI_GGUF_TYPE_ARRAY, the element type and length.
mi_gguf_type mi_gguf_kv_arr_type(const mi_gguf_context* ctx, size_t i);
size_t mi_gguf_kv_arr_len(const mi_gguf_context* ctx, size_t i);

// Typed scalar accessors, looked up by key. Return false if the key is
// absent or its stored type doesn't match.
bool mi_gguf_get_u32(const mi_gguf_context* ctx, const char* key, uint32_t* out);
bool mi_gguf_get_i32(const mi_gguf_context* ctx, const char* key, int32_t* out);
bool mi_gguf_get_f32(const mi_gguf_context* ctx, const char* key, float* out);
bool mi_gguf_get_bool(const mi_gguf_context* ctx, const char* key, bool* out);
bool mi_gguf_get_str(const mi_gguf_context* ctx, const char* key, std::string* out);

// String-array accessors (used for tokenizer.ggml.tokens / .merges).
bool mi_gguf_get_arr_n(const mi_gguf_context* ctx, const char* key, size_t* n);
bool mi_gguf_get_arr_str(const mi_gguf_context* ctx, const char* key, size_t idx,
                          std::string* out);
bool mi_gguf_get_arr_i32(const mi_gguf_context* ctx, const char* key, size_t idx,
                          int32_t* out);

// Generic text rendering, for tools/dump and debugging: covers every
// scalar/array-element type (including the float and 64-bit ones no typed
// getter above exposes) without a combinatorial explosion of accessors.
// Integers render as decimal, floats with enough digits to round-trip,
// bool as "true"/"false", strings verbatim (caller escapes if embedding
// into a line-oriented format). Returns false for i == ARRAY itself (walk
// its elements via mi_gguf_kv_arr_type/_len + mi_gguf_kv_arr_elem_to_string).
bool mi_gguf_kv_scalar_to_string(const mi_gguf_context* ctx, size_t i, std::string* out);
bool mi_gguf_kv_arr_elem_to_string(const mi_gguf_context* ctx, size_t i, size_t idx,
                                    std::string* out);

// ---------------------------------------------------------------------------
// Vocab: id -> text decoding only (no encoding; see scripts/encode.py).
// Reads tokenizer.ggml.tokens (+ token_type/scores where needed for
// special-token handling) and reverses byte-level BPE's byte<->unicode
// mapping so decoded text matches HF tokenizer.decode() exactly.
// ---------------------------------------------------------------------------
struct mi_vocab;

mi_vocab* mi_vocab_load(const mi_gguf_context* ctx, std::string* err);
void mi_vocab_free(mi_vocab* vocab);

size_t mi_vocab_size(const mi_vocab* vocab);
// Raw decoded bytes contributed by a single id. For a NORMAL token this
// reverses the GPT2/byte-level-BPE byte<->unicode mapping (so a multi-byte
// UTF-8 character split across several ids only becomes valid UTF-8 once
// its pieces are concatenated — call mi_vocab_decode for a whole sequence
// rather than treating each id's text as standalone). Control/user-defined
// tokens (e.g. "<|im_start|>") are returned as their literal text, matching
// HF's tokenizer.decode(ids) default of skip_special_tokens=False.
std::string mi_vocab_token_to_text(const mi_vocab* vocab, int32_t id);
// Decodes a whole sequence, concatenating each id's text.
std::string mi_vocab_decode(const mi_vocab* vocab, const int32_t* ids, size_t n_ids);
