#include "tensor.h"

#include <cassert>
#include <cstring>

namespace {

struct mi_type_traits {
    const char* name;
    int64_t block_size;   // elements per block
    size_t block_bytes;   // bytes per block
};

// Q8_0's block (1 fp16 scale + 32 int8 = 34 bytes) matches llama.cpp's GGUF
// layout; see quant.h's block_q8_0 (stage 4) for the struct itself. The
// size fact is needed as early as stage 0 (computing a GGUF tensor's byte
// size), well before the dequant/vec_dot kernels exist, so it is recorded
// here rather than by including quant.h.
const mi_type_traits& mi_get_type_traits(mi_type type) {
    static const mi_type_traits table[MI_TYPE_COUNT] = {
        /* MI_TYPE_F32      */ {"f32", 1, 4},
        /* MI_TYPE_F16      */ {"f16", 1, 2},
        /* MI_TYPE_Q8_0     */ {"q8_0", 32, 34},
        /* MI_TYPE_I32      */ {"i32", 1, 4},
        /* MI_TYPE_Q8_0_ACT */ {"q8_0_act", 32, 36},  // stage 6: float scale, see tensor.h
    };
    assert(type >= 0 && type < MI_TYPE_COUNT);
    return table[type];
}

}  // namespace

size_t mi_type_block_size(mi_type type) { return (size_t)mi_get_type_traits(type).block_size; }
size_t mi_type_block_bytes(mi_type type) { return mi_get_type_traits(type).block_bytes; }
const char* mi_type_name(mi_type type) { return mi_get_type_traits(type).name; }

size_t mi_row_size(mi_type type, int64_t ne) {
    int64_t bs = mi_get_type_traits(type).block_size;
    assert(ne % bs == 0 && "ne must be a whole number of blocks for this type");
    return (size_t)(ne / bs) * mi_type_block_bytes(type);
}

const char* mi_op_name(mi_op op) {
    static const char* names[MI_OP_COUNT] = {
        "none",     "get_rows", "rms_norm",       "mul",      "add",
        "mul_mat",  "rope",     "softmax",        "silu",     "scale",
        "cpy",      "quantize_q8_0", "reshape",   "view",     "permute",
        "transpose", "attn_qk", "attn_kqv",
    };
    assert(op >= 0 && op < MI_OP_COUNT);
    return names[op];
}

bool mi_op_is_view(mi_op op) {
    return op == MI_OP_RESHAPE || op == MI_OP_VIEW || op == MI_OP_PERMUTE ||
           op == MI_OP_TRANSPOSE;
}

int64_t mi_nelements(const mi_tensor* t) {
    int64_t n = 1;
    for (int i = 0; i < MI_MAX_DIMS; i++) n *= t->ne[i];
    return n;
}

size_t mi_nbytes(const mi_tensor* t) { return mi_row_size(t->type, mi_nelements(t)); }

bool mi_same_shape(const mi_tensor* a, const mi_tensor* b) {
    for (int i = 0; i < MI_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) return false;
    }
    return true;
}

void mi_tensor_set_name(mi_tensor* t, const char* name) {
    std::strncpy(t->name, name, MI_MAX_NAME - 1);
    t->name[MI_MAX_NAME - 1] = '\0';
}
