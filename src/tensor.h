// mini-infer: core tensor data structure.
//
// A mi_tensor is a plain-old-data description of an N-D array plus enough
// bookkeeping (op, src, view info, refcount, flags) to serve both as a
// "value" (stage 1, direct kernel calls) and as a graph node (stage 2+).
// See docs/architecture.md for the full design rationale.
#pragma once

#include <cstddef>
#include <cstdint>

#define MI_MAX_DIMS 4
#define MI_MAX_SRC 2
#define MI_MAX_OP_PARAMS 8
#define MI_MAX_NAME 64

// ---------------------------------------------------------------------------
// Element types
// ---------------------------------------------------------------------------
enum mi_type {
    MI_TYPE_F32 = 0,
    MI_TYPE_F16 = 1,
    MI_TYPE_Q8_0 = 2,
    MI_TYPE_I32 = 3,  // token ids / positions, never a weight type
    // Stage 6: MI_OP_QUANTIZE_Q8_0's output block layout -- same 32-int8
    // body as MI_TYPE_Q8_0 but a 4-byte float scale instead of a 2-byte
    // fp16 one (see quant.h's block_q8_0_act), so it needs its own type tag
    // for mi_nbytes()/gallocr to size its buffer correctly. Appended at the
    // end (not inserted after MI_TYPE_Q8_0) so no existing type's numeric
    // value changes. Never produced by gguf.cpp -- only ever appears as a
    // graph-internal scratch tensor's type, never a GGUF file's.
    MI_TYPE_Q8_0_ACT = 4,
    MI_TYPE_COUNT,
};

// Bytes-per-block and elements-per-block for each type (Q8_0 is block
// quantized: 32 elements share one scale). Defined in tensor.cpp.
size_t mi_type_block_size(mi_type type);   // elements per block (1 for F32/F16/I32)
size_t mi_type_block_bytes(mi_type type);  // bytes per block
const char* mi_type_name(mi_type type);

// Number of bytes needed to store `ne` contiguous elements of `type`.
// Asserts ne is a multiple of the block size for quantized types.
size_t mi_row_size(mi_type type, int64_t ne);

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------
enum mi_op {
    MI_OP_NONE = 0,  // leaf: weight, input, or KV cache storage

    // compute ops (stage 1 kernels, reused verbatim by the stage 2 graph)
    MI_OP_GET_ROWS,
    MI_OP_RMS_NORM,
    MI_OP_MUL,
    MI_OP_ADD,
    MI_OP_MUL_MAT,
    MI_OP_ROPE,
    MI_OP_SOFTMAX,   // fused scale + causal mask + softmax
    MI_OP_SILU,
    MI_OP_SCALE,
    MI_OP_CPY,       // write src[0] into dst's (view) storage, e.g. KV cache
    // Quantizes src[0] (F32) into dst (Q8_0), row by row. Inserted
    // automatically by mi_graph_mul_mat (stage 4) when its weight operand
    // is Q8_0, so the quantized-activation buffer is an ordinary graph
    // tensor that gallocr plans like any other -- never a hand-managed
    // scratch pointer (SPEC.md: "运行期仍需零分配"). See ops_cpu.h.
    MI_OP_QUANTIZE_Q8_0,

    // view-only ops: never touch data, only reinterpret an existing buffer.
    // dst->view_src is set and dst->data aliases view_src's data + view_offs.
    MI_OP_RESHAPE,
    MI_OP_VIEW,
    MI_OP_PERMUTE,
    MI_OP_TRANSPOSE,

    // Stage 6 step 3b: batched multi-head attention, replacing what used to
    // be a per-head loop of view+mul_mat+softmax+view+transpose+mul_mat+cpy
    // (see docs/stage6.md). Appended at the end (not grouped with the other
    // compute ops above) purely so no existing op's numeric value changes.
    // Both are F32-only (K/V cache and Q/scores are always F32 in this
    // project -- no Q8_0/F16 dispatch needed).
    //   MI_OP_ATTN_QK:  src[0]=Q_rot [head_dim,n_head,n_tokens] (contiguous),
    //     src[1]=the full persistent K cache [head_dim,n_head_kv,n_ctx].
    //     dst=[n_kv_total,n_tokens,n_head]. op_params: n_past (i32), n_rep
    //     (i32, = n_head/n_head_kv, the GQA sharing factor). For each
    //     (h,i,j): dst[i,j,h] = sum_d K[d, h/n_rep, i] * Q_rot[d,h,j] -- same
    //     summation order (d ascending) mul_mat's F32 path already used per
    //     head, just batched into one node.
    //   MI_OP_ATTN_KQV: src[0]=probs [n_kv_total,n_tokens,n_head] (softmax
    //     output), src[1]=the full persistent V cache [head_dim,n_head_kv,n_ctx].
    //     dst=[head_dim,n_head,n_tokens] (contiguous -- same "head_dim
    //     innermost, heads consecutive" layout as reshape_heads, so a plain
    //     reshape turns it directly into attn_concat's [n_embd,n_tokens],
    //     no cpy needed). op_params: n_rep (i32). For each (h,o,j):
    //     dst[o,h,j] = sum_t V[o, h/n_rep, t] * probs[t,j,h] -- same t-
    //     ascending summation order as before.
    MI_OP_ATTN_QK,
    MI_OP_ATTN_KQV,

    MI_OP_COUNT,
};

const char* mi_op_name(mi_op op);
bool mi_op_is_view(mi_op op);  // RESHAPE/VIEW/PERMUTE/TRANSPOSE

// ---------------------------------------------------------------------------
// Tensor flags: what a tensor *is*, independent of how it was produced.
// A tensor with any of WEIGHT/INPUT/OUTPUT/KV set is "persistent" and must
// never be assigned a buffer by gallocr's arena (stage 2b).
// ---------------------------------------------------------------------------
enum mi_tensor_flag : uint32_t {
    MI_TENSOR_FLAG_WEIGHT = 1u << 0,
    MI_TENSOR_FLAG_INPUT = 1u << 1,
    MI_TENSOR_FLAG_OUTPUT = 1u << 2,
    MI_TENSOR_FLAG_KV = 1u << 3,
};

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------
struct mi_tensor {
    mi_type type;

    // ne[0] is the innermost (fastest-varying) dimension, matching GGUF's
    // convention. nb[i] is the stride of dimension i, in bytes.
    int64_t ne[MI_MAX_DIMS];
    size_t nb[MI_MAX_DIMS];

    mi_op op;
    // Small POD parameters for the op (e.g. rms_norm eps, rope freq_base,
    // scale factor, n_past). Values are stored via memcpy of their native
    // type into this array; see mi_op_params_get_f32/i32 in ops_cpu.h.
    int32_t op_params[MI_MAX_OP_PARAMS];

    mi_tensor* src[MI_MAX_SRC];

    void* data;

    // View bookkeeping. If view_src is non-null, this tensor does not own
    // its storage: `data == view_src->data + view_offs`. View chains are
    // flattened at construction time so view_src always points at the
    // actual storage owner (never at another view).
    mi_tensor* view_src;
    size_t view_offs;

    // Reference count: number of graph nodes that read this tensor as a
    // src. Filled by gallocr's rehearsal pass (stage 2b); unused/zero
    // before that. For a tensor with view_src set, consumption is counted
    // against view_src, not against the view itself.
    int32_t n_children;

    uint32_t flags;

    char name[MI_MAX_NAME];

    // Reserved for backend-specific scratch (e.g. per-thread quantized
    // activation buffer). Not used before stage 4.
    void* extra;
};

inline bool mi_tensor_is_view(const mi_tensor* t) { return t->view_src != nullptr; }

// True if flags contain WEIGHT/INPUT/OUTPUT/KV: gallocr must not place this
// tensor's data in the reusable arena.
inline bool mi_tensor_is_persistent(const mi_tensor* t) {
    return (t->flags & (MI_TENSOR_FLAG_WEIGHT | MI_TENSOR_FLAG_INPUT |
                         MI_TENSOR_FLAG_OUTPUT | MI_TENSOR_FLAG_KV)) != 0;
}

int64_t mi_nelements(const mi_tensor* t);
size_t mi_nbytes(const mi_tensor* t);
bool mi_same_shape(const mi_tensor* a, const mi_tensor* b);

void mi_tensor_set_name(mi_tensor* t, const char* name);
