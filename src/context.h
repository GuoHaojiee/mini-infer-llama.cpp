// mini-infer: context = a preallocated pool of mi_tensor metadata (+ an
// optional bump-allocated data arena). Never calls new/malloc per tensor
// once created, so rebuilding a graph every forward step costs zero heap
// allocations for the metadata itself. See docs/architecture.md.
#pragma once

#include <cstddef>

#include "tensor.h"

struct mi_context {
    // Metadata pool: fixed-capacity array of mi_tensor, handed out by
    // bump pointer. Reset by mi_context_reset() to reuse for the next step.
    mi_tensor* tensors;
    size_t tensor_capacity;
    size_t tensor_count;

    // Data arena. When no_alloc == false, mi_context_new_tensor also
    // bump-allocates mi_nbytes(t) bytes from this arena for t->data. When
    // no_alloc == true, mi_context_new_tensor leaves t->data null (the
    // caller — gallocr, or GGUF mmap loading — assigns it later); the
    // arena itself may still be nonzero-sized and is used directly via
    // mi_context_alloc_bytes / mi_context_new_tensor_eager for graph
    // bookkeeping and persistent/OUTPUT tensors that need data immediately
    // even inside a no_alloc=true (gallocr-managed) context.
    unsigned char* data;
    size_t data_capacity;
    size_t data_used;
    bool no_alloc;
};

struct mi_context_params {
    size_t tensor_capacity;  // max number of mi_tensor metadata slots
    size_t data_capacity;    // bytes for the data arena (may be nonzero even if no_alloc)
    bool no_alloc;           // if true, mi_context_new_tensor leaves data == nullptr
};

// Allocates ctx->tensors and (if !no_alloc) ctx->data as two single
// malloc'd blocks up front; no further heap traffic afterward.
mi_context* mi_context_new(const mi_context_params& params);
void mi_context_free(mi_context* ctx);

// Rewinds tensor_count and data_used to 0 so the pool can be reused by the
// next forward step. Does not release the underlying blocks.
void mi_context_reset(mi_context* ctx);

// Allocates one mi_tensor slot from the pool and zero-initializes it
// (op = MI_OP_NONE, src = {nullptr}, view_src = nullptr, n_children = 0,
// flags = 0). If !ctx->no_alloc, also bump-allocates t->data from the data
// arena. Aborts (not throws) if either pool is exhausted: a capacity miss
// is a sizing bug in reserve(), not a recoverable runtime condition.
mi_tensor* mi_context_new_tensor(mi_context* ctx, mi_type type, int n_dims,
                                  const int64_t* ne);

// Allocates only a metadata slot and points t->data at the caller-owned
// `data` (never touches ctx's data arena). Used for tensors whose storage
// is external and outlives the context: GGUF-mmap'd weights, and KV cache
// buffers allocated once at startup. The caller is responsible for `data`
// staying valid and for setting flags (WEIGHT/KV/...) afterward.
mi_tensor* mi_context_new_tensor_from_data(mi_context* ctx, mi_type type, int n_dims,
                                            const int64_t* ne, void* data);

// Like mi_context_new_tensor, but always bump-allocates data from ctx's own
// arena regardless of ctx->no_alloc. For the handful of tensors that must
// have valid data immediately even inside an otherwise no_alloc=true
// (gallocr-managed) context: SPEC.md's persistent/OUTPUT-flagged tensors
// (stage 2b's `logits`, never arena-managed since gallocr must never
// reuse/free it) and small fixed bookkeeping like the per-step `ids` input.
mi_tensor* mi_context_new_tensor_eager(mi_context* ctx, mi_type type, int n_dims,
                                        const int64_t* ne);

// Allocates a metadata-only view over `src`'s storage with caller-chosen
// shape/strides/offset (never bump-allocates from ctx's data arena for the
// view itself). view_src is resolved to src's own view_src if src is
// itself a view, so chains flatten to one hop (see tensor.h); view_offs is
// computed arithmetically (src's own offset plus this call's `offset`),
// never by pointer subtraction, so it's correct even before view_src has a
// real address. `data` is resolved immediately if view_src->data is
// already non-null (stage 1/2a's eager contexts, weights, KV cache);
// otherwise it stays null until gallocr resolves it once view_src's
// address is known (stage 2b — see gallocr.cpp). This is the low-level
// primitive model_qwen2.cpp uses for per-head slicing, the transposed-V
// trick, and KV-cache write targets, and that mi_graph_view/permute/
// reshape/transpose build on top of (adding op/src for graph dependency
// tracking, which a bare view has no need for outside a graph).
// mi_nbytes()/mi_nelements() on the result assume a contiguous layout and
// so are not meaningful for a genuinely strided view -- only ne/nb/data
// matter for kernels, which index through nb[] directly.
mi_tensor* mi_context_view(mi_context* ctx, mi_tensor* src, mi_type type, int n_dims,
                            const int64_t* ne, const size_t* nb, size_t offset);

// Convenience wrappers for 1D/2D/3D/4D shapes.
mi_tensor* mi_new_tensor_1d(mi_context* ctx, mi_type type, int64_t ne0);
mi_tensor* mi_new_tensor_2d(mi_context* ctx, mi_type type, int64_t ne0, int64_t ne1);
mi_tensor* mi_new_tensor_3d(mi_context* ctx, mi_type type, int64_t ne0, int64_t ne1,
                             int64_t ne2);
mi_tensor* mi_new_tensor_4d(mi_context* ctx, mi_type type, int64_t ne0, int64_t ne1,
                             int64_t ne2, int64_t ne3);

size_t mi_context_used_bytes(const mi_context* ctx);

// Bump-allocates nbytes (max_align_t-aligned) from the same data arena
// mi_context_new_tensor uses, for callers that need raw storage rather
// than a tensor -- graph.h's mi_graph_new uses this for the mi_cgraph
// struct itself and its nodes/leafs pointer arrays, so building a graph
// every forward step stays as allocation-free as building the tensors in
// it. Works regardless of ctx->no_alloc (that flag only controls whether
// mi_context_new_tensor itself eagerly allocates data for the tensors it
// creates); just requires ctx to actually have a data arena
// (data_capacity > 0) with enough room left.
void* mi_context_alloc_bytes(mi_context* ctx, size_t nbytes);
