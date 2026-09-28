#include "context.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>

namespace {

// ne[0] is innermost (SPEC.md / GGUF convention). For block-quantized types
// nb[0] is bytes-per-block (not bytes-per-element) -- ne[0] is required to
// be a whole number of blocks, so nb[1] (bytes per "row") is block_bytes *
// (ne[0]/block_size), matching mi_row_size in tensor.cpp exactly.
void compute_strides(mi_tensor* t) {
  size_t block_size = mi_type_block_size(t->type);
  size_t block_bytes = mi_type_block_bytes(t->type);
  t->nb[0] = block_bytes;
  t->nb[1] = t->nb[0] * (size_t)(t->ne[0] / (int64_t)block_size);
  t->nb[2] = t->nb[1] * (size_t)t->ne[1];
  t->nb[3] = t->nb[2] * (size_t)t->ne[2];
}

mi_tensor* alloc_tensor_slot(mi_context* ctx, mi_type type, int n_dims, const int64_t* ne) {
  assert(ctx->tensor_count < ctx->tensor_capacity && "mi_context tensor pool exhausted");
  mi_tensor* t = &ctx->tensors[ctx->tensor_count++];
  *t = mi_tensor{};
  t->type = type;
  for (int i = 0; i < MI_MAX_DIMS; i++) t->ne[i] = 1;
  for (int i = 0; i < n_dims; i++) t->ne[i] = ne[i];
  compute_strides(t);
  return t;
}

}  // namespace

mi_context* mi_context_new(const mi_context_params& params) {
  auto* ctx = new mi_context();
  ctx->tensor_capacity = params.tensor_capacity;
  ctx->tensor_count = 0;
  ctx->tensors = (mi_tensor*)std::malloc(sizeof(mi_tensor) * params.tensor_capacity);
  ctx->no_alloc = params.no_alloc;
  // no_alloc only controls whether mi_context_new_tensor itself eagerly
  // allocates data for the tensors it creates -- a no_alloc=true context
  // (stage 2b: intermediate tensor data deferred to gallocr) can still
  // have its own data arena, used via mi_context_alloc_bytes /
  // mi_context_new_tensor_eager for graph bookkeeping and the handful of
  // tensors that must be valid immediately (persistent/OUTPUT leaves).
  ctx->data_capacity = params.data_capacity;
  ctx->data = ctx->data_capacity ? (unsigned char*)std::malloc(ctx->data_capacity) : nullptr;
  ctx->data_used = 0;
  return ctx;
}

void mi_context_free(mi_context* ctx) {
  if (!ctx) return;
  std::free(ctx->tensors);
  std::free(ctx->data);
  delete ctx;
}

void mi_context_reset(mi_context* ctx) {
  ctx->tensor_count = 0;
  ctx->data_used = 0;
}

mi_tensor* mi_context_new_tensor(mi_context* ctx, mi_type type, int n_dims, const int64_t* ne) {
  mi_tensor* t = alloc_tensor_slot(ctx, type, n_dims, ne);
  if (!ctx->no_alloc) {
    size_t nbytes = mi_nbytes(t);
    assert(ctx->data_used + nbytes <= ctx->data_capacity && "mi_context data arena exhausted");
    t->data = ctx->data + ctx->data_used;
    ctx->data_used += nbytes;
  }
  return t;
}

mi_tensor* mi_context_new_tensor_from_data(mi_context* ctx, mi_type type, int n_dims,
                                            const int64_t* ne, void* data) {
  mi_tensor* t = alloc_tensor_slot(ctx, type, n_dims, ne);
  t->data = data;
  return t;
}

mi_tensor* mi_context_new_tensor_eager(mi_context* ctx, mi_type type, int n_dims,
                                        const int64_t* ne) {
  mi_tensor* t = alloc_tensor_slot(ctx, type, n_dims, ne);
  t->data = mi_context_alloc_bytes(ctx, mi_nbytes(t));
  return t;
}

mi_tensor* mi_new_tensor_1d(mi_context* ctx, mi_type type, int64_t ne0) {
  int64_t ne[1] = {ne0};
  return mi_context_new_tensor(ctx, type, 1, ne);
}

mi_tensor* mi_new_tensor_2d(mi_context* ctx, mi_type type, int64_t ne0, int64_t ne1) {
  int64_t ne[2] = {ne0, ne1};
  return mi_context_new_tensor(ctx, type, 2, ne);
}

mi_tensor* mi_new_tensor_3d(mi_context* ctx, mi_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
  int64_t ne[3] = {ne0, ne1, ne2};
  return mi_context_new_tensor(ctx, type, 3, ne);
}

mi_tensor* mi_new_tensor_4d(mi_context* ctx, mi_type type, int64_t ne0, int64_t ne1, int64_t ne2,
                             int64_t ne3) {
  int64_t ne[4] = {ne0, ne1, ne2, ne3};
  return mi_context_new_tensor(ctx, type, 4, ne);
}

size_t mi_context_used_bytes(const mi_context* ctx) { return ctx->data_used; }

void* mi_context_alloc_bytes(mi_context* ctx, size_t nbytes) {
  size_t align = alignof(std::max_align_t);
  size_t aligned_used = (ctx->data_used + align - 1) / align * align;
  assert(aligned_used + nbytes <= ctx->data_capacity && "mi_context data arena exhausted");
  void* p = ctx->data + aligned_used;
  ctx->data_used = aligned_used + nbytes;
  return p;
}

mi_tensor* mi_context_view(mi_context* ctx, mi_tensor* src, mi_type type, int n_dims,
                            const int64_t* ne, const size_t* nb, size_t offset) {
  assert(ctx->tensor_count < ctx->tensor_capacity && "mi_context tensor pool exhausted");
  mi_tensor* t = &ctx->tensors[ctx->tensor_count++];
  *t = mi_tensor{};
  t->type = type;
  for (int i = 0; i < MI_MAX_DIMS; i++) t->ne[i] = 1;
  for (int i = 0; i < n_dims; i++) {
    t->ne[i] = ne[i];
    t->nb[i] = nb[i];
  }
  for (int i = n_dims; i < MI_MAX_DIMS; i++) {
    t->nb[i] = (i == 0) ? mi_type_block_bytes(type) : t->nb[i - 1] * (size_t)t->ne[i - 1];
  }
  t->view_src = src->view_src ? src->view_src : src;
  // view_offs is computed arithmetically (src's own offset from the root,
  // plus this new offset), never by pointer subtraction, because src->data
  // may still be null: stage 2b defers intermediate tensors' data to
  // gallocr, which runs its planning pass *after* the whole graph (views
  // included) has already been built. `data` is resolved opportunistically
  // here (immediately, for views of already-allocated roots -- weights, KV
  // cache, anything from stage 1/2a's eager contexts) and otherwise stays
  // null until mi_gallocr_alloc_graph resolves it once view_src has a real
  // address (see gallocr.cpp).
  size_t base_offset = src->view_src ? src->view_offs : 0;
  t->view_offs = base_offset + offset;
  t->data = t->view_src->data ? (unsigned char*)t->view_src->data + t->view_offs : nullptr;
  return t;
}
