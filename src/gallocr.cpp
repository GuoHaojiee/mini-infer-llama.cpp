#include "gallocr.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ops_cpu.h"

namespace {

struct free_block {
  size_t offset;
  size_t size;
};

struct mi_gallocr_impl {
  std::vector<free_block> free_list;
  bool no_inplace = false;
  bool no_bestfit = false;
};

bool env_flag_set(const char* name) {
  const char* v = std::getenv(name);
  return v && std::string(v) == "1";
}

// gallocr's own "has this non-view tensor already been given an arena
// offset during the current planning pass" marker. Every tensor is freshly
// zero-initialized when carved from its context (see context.cpp's
// `*t = mi_tensor{}`), so this bit is always unset on a brand-new tensor;
// gallocr sets it the moment it assigns an offset. It lives in the high bit
// of the public `flags` word (tensor.h only defines bits 0-3) purely as a
// gallocr-internal implementation detail -- nothing outside this file ever
// reads or depends on it.
constexpr uint32_t kAllocatedBit = 1u << 31;
bool is_allocated(const mi_tensor* t) { return (t->flags & kAllocatedBit) != 0; }
void mark_allocated(mi_tensor* t) { t->flags |= kAllocatedBit; }

// Only ops verified (by inspecting ops_cpu.cpp) to read src[i] and write
// dst[i] with no cross-index dependency within a row are eligible for
// in-place rewrite -- matching SPEC.md's "逐元素算子(add、mul、silu、
// scale、rope 等支持原地的算子)". mul_mat and get_rows are deliberately
// excluded even though they might pass the other 5 conditions: mul_mat's
// dst[i,j] depends on *all* of src1[:,j], not just src1[i,j], so aliasing
// dst with src1 corrupts values a later (i,j) still needs to read (see
// docs/stage2b.md for the concrete failure this project actually hit:
// attn_concat, read in full by the O-projection's mul_mat, would otherwise
// look like a valid in-place candidate by every *other* condition). softmax
// happens to be index-local in this implementation too, but is kept off
// the whitelist to stay conservative and match SPEC.md's literal list.
bool op_supports_inplace(mi_op op) {
  switch (op) {
    case MI_OP_ADD:
    case MI_OP_MUL:
    case MI_OP_SILU:
    case MI_OP_SCALE:
    case MI_OP_ROPE:
    case MI_OP_RMS_NORM:
      return true;
    default:
      return false;
  }
}

mi_tensor* resolve_root(mi_tensor* t) { return t->view_src ? t->view_src : t; }

// SPEC.md's 5(+op-whitelist) in-place conditions, checked on the *direct*
// src (never resolved through a view -- see condition 2's rationale in
// docs/stage2b.md: reusing a partial view's region could clobber sibling
// data other consumers still need).
bool is_eligible_inplace_src(const mi_tensor* dst, mi_tensor* src) {
  if (!src) return false;
  if (mi_tensor_is_view(src)) return false;         // condition 2: non-view
  if (mi_tensor_is_persistent(src)) return false;   // condition 4: non-persistent
  if (src->n_children != 1) return false;           // condition 1 (+3, see docs/stage2b.md)
  if (src->type != dst->type) return false;         // condition 6: same type
  if (mi_nbytes(src) != mi_nbytes(dst)) return false;  // condition 6: same size
  return true;
}

// Remaining-uses-not-yet-processed counter, used only during the
// allocation walk to decide *freeing* timing (has the last consumer run
// yet). Deliberately kept separate from n_children (see count_references):
// n_children must stay the *original* total for the whole pass, or "am I
// the sole consumer" (in-place eligibility) and "am I merely the last of
// several" both read as 1 and become indistinguishable -- concretely, in a
// 2-consumer case, checking the live-decrementing count during the second
// consumer's turn would see 1 and wrongly call it sole-consumer eligible.
// Stashed in mi_tensor::extra ("reserved... not used before stage 4" per
// tensor.h) rather than a side map, so no heap allocation is needed.
int remaining(const mi_tensor* t) { return (int)(intptr_t)t->extra; }
void set_remaining(mi_tensor* t, int n) { t->extra = (void*)(intptr_t)n; }

// Rehearsal's reference-counting pass: for every genuine compute/cpy node
// (never a pure view op -- see docs/stage2a.md/2b.md, a view doesn't itself
// "consume" its parent, whoever reads through it does), credit each src to
// its resolved root, skipping persistent roots entirely. n_children ends up
// holding each tensor's *total* reference count for this graph, used as-is
// (never decremented) by is_eligible_inplace_src's "sole consumer" check;
// plan_graph separately seeds `extra` as the mutable remaining-count.
void count_references(mi_cgraph* g) {
  for (int i = 0; i < g->n_nodes; i++) g->nodes[i]->n_children = 0;
  for (int i = 0; i < g->n_leafs; i++) g->leafs[i]->n_children = 0;

  for (int i = 0; i < g->n_nodes; i++) {
    mi_tensor* t = g->nodes[i];
    if (mi_op_is_view(t->op)) continue;
    for (int s = 0; s < MI_MAX_SRC; s++) {
      mi_tensor* src = t->src[s];
      if (!src) continue;
      mi_tensor* root = resolve_root(src);
      if (mi_tensor_is_persistent(root)) continue;
      root->n_children++;
    }
  }

  for (int i = 0; i < g->n_nodes; i++) set_remaining(g->nodes[i], g->nodes[i]->n_children);
  for (int i = 0; i < g->n_leafs; i++) set_remaining(g->leafs[i], g->leafs[i]->n_children);
}

// "As if independently allocated": every non-persistent tensor that owns a
// buffer (a node that isn't a view, or a bare non-persistent leaf) counted
// once at its own size, regardless of reuse.
size_t compute_baseline(const mi_cgraph* g) {
  size_t total = 0;
  for (int i = 0; i < g->n_nodes; i++) {
    const mi_tensor* t = g->nodes[i];
    if (mi_tensor_is_view(t) || mi_tensor_is_persistent(t)) continue;
    total += mi_nbytes(t);
  }
  for (int i = 0; i < g->n_leafs; i++) {
    const mi_tensor* t = g->leafs[i];
    if (mi_tensor_is_persistent(t)) continue;
    total += mi_nbytes(t);
  }
  return total;
}

void poison_range(unsigned char* arena, size_t offset, size_t size) {
  uint32_t nan_bits = 0x7fc00000u;
  float nanf;
  std::memcpy(&nanf, &nan_bits, sizeof(nanf));
  size_t n = size / sizeof(float);
  float* p = (float*)(arena + offset);
  for (size_t i = 0; i < n; i++) p[i] = nanf;
}

// The shared core: rehearsal (arena == nullptr, dry run -- just enough to
// size the arena and count references) and real per-step planning
// (arena != nullptr) run the *identical* best-fit/in-place algorithm, so
// the size reserve() computes is exactly what alloc_graph() will actually
// need -- a rehearsal that used a cruder algorithm would size the arena
// for the wrong (too-small, or wastefully-too-large) peak.
void plan_graph(mi_gallocr* galloc, mi_cgraph* g, unsigned char* arena, size_t arena_size) {
  bool dry_run = (arena == nullptr);
  auto* impl = (mi_gallocr_impl*)galloc->impl;
  impl->free_list.clear();
  size_t watermark = 0;
  galloc->stats.n_inplace_reuse = 0;
  galloc->stats.n_bestfit_reuse = 0;

  count_references(g);

  auto set_offset = [](mi_tensor* t, size_t off) { t->view_offs = off; };
  auto get_offset = [](const mi_tensor* t) { return t->view_offs; };

  auto alloc_bytes = [&](size_t need) -> size_t {
    if (!impl->no_bestfit) {
      int best = -1;
      for (size_t i = 0; i < impl->free_list.size(); i++) {
        if (impl->free_list[i].size >= need &&
            (best < 0 || impl->free_list[i].size < impl->free_list[(size_t)best].size)) {
          best = (int)i;
        }
      }
      if (best >= 0) {
        size_t off = impl->free_list[(size_t)best].offset;
        size_t blk_size = impl->free_list[(size_t)best].size;
        impl->free_list.erase(impl->free_list.begin() + best);
        if (blk_size > need) impl->free_list.push_back({off + need, blk_size - need});
        galloc->stats.n_bestfit_reuse++;
        return off;
      }
    }
    size_t off = watermark;
    watermark += need;
    // A real, always-on check (not assert(), which -DNDEBUG compiles out
    // of the Release build this project measures performance with):
    // exceeding the reserved arena means silently writing past a malloc'd
    // buffer, which is memory corruption, not a recoverable condition --
    // this must be checked in every build configuration.
    if (!dry_run && watermark > arena_size) {
      std::fprintf(stderr,
                    "gallocr: graph needs %zu bytes, more than the %zu reserved -- "
                    "reserve()'s worst-case graphs didn't cover this case\n",
                    watermark, arena_size);
      std::abort();
    }
    return off;
  };

  auto free_bytes = [&](size_t offset, size_t size) {
    if (!dry_run && galloc->poison_enabled) poison_range(arena, offset, size);
    free_block nb{offset, size};
    auto it = impl->free_list.begin();
    while (it != impl->free_list.end()) {
      if (it->offset + it->size == nb.offset) {
        nb.offset = it->offset;
        nb.size += it->size;
        it = impl->free_list.erase(it);
        continue;
      }
      if (nb.offset + nb.size == it->offset) {
        nb.size += it->size;
        it = impl->free_list.erase(it);
        continue;
      }
      ++it;
    }
    impl->free_list.push_back(nb);
  };

  // Lazily allocates a non-persistent bare leaf (e.g. the multi-head
  // attention concat buffer: it has no producer node of its own -- nothing
  // ever computes "into" it as a normal node output, several independent
  // per-head cpy writes each land in a different slice of it instead) the
  // first time anything references it, whether directly or through a view.
  // Doing this lazily (not eagerly for every leaf up front) means e.g. a
  // later layer's concat buffer doesn't occupy arena space while an
  // earlier layer's is still the only one alive.
  auto ensure_root_allocated = [&](mi_tensor* root) {
    if (mi_tensor_is_persistent(root)) return;
    if (is_allocated(root)) return;
    size_t off = alloc_bytes(mi_nbytes(root));
    set_offset(root, off);
    mark_allocated(root);
    if (!dry_run) root->data = arena + off;
  };

  for (int i = 0; i < g->n_nodes; i++) {
    mi_tensor* t = g->nodes[i];

    if (mi_op_is_view(t->op)) {
      // Pure view (RESHAPE/VIEW/PERMUTE/TRANSPOSE): shares view_src's
      // buffer, no allocation, and -- matching count_references -- no
      // refcount effects; whoever reads *through* this view is what
      // actually consumes view_src.
      ensure_root_allocated(t->view_src);
      if (!dry_run) t->data = (unsigned char*)t->view_src->data + t->view_offs;
      continue;
    }

    mi_tensor* inplace_src = nullptr;

    if (mi_tensor_is_view(t)) {
      // cpy: t is itself a view (its destination slice), already carrying
      // view_src/view_offs from the mi_graph_view call mi_graph_cpy
      // mutated into a CPY op -- resolve its data the same way, no
      // allocation. Its src[0] (what's being copied *from*) still needs
      // normal refcount processing below.
      ensure_root_allocated(t->view_src);
      if (!dry_run) t->data = (unsigned char*)t->view_src->data + t->view_offs;
    } else if (mi_tensor_is_persistent(t)) {
      // e.g. `logits`: caller already gave it real data (see
      // model_qwen2.cpp) precisely so gallocr never touches it.
    } else {
      if (!impl->no_inplace && op_supports_inplace(t->op)) {
        for (int s = 0; s < MI_MAX_SRC; s++) {
          if (is_eligible_inplace_src(t, t->src[s])) {
            inplace_src = t->src[s];
            break;
          }
        }
      }
      if (inplace_src) {
        set_offset(t, get_offset(inplace_src));
        mark_allocated(t);
        if (!dry_run) t->data = inplace_src->data;
        galloc->stats.n_inplace_reuse++;
      } else {
        size_t off = alloc_bytes(mi_nbytes(t));
        set_offset(t, off);
        mark_allocated(t);
        if (!dry_run) t->data = arena + off;
      }
    }

    for (int s = 0; s < MI_MAX_SRC; s++) {
      mi_tensor* src = t->src[s];
      if (!src) continue;
      mi_tensor* root = resolve_root(src);
      ensure_root_allocated(root);
      if (mi_tensor_is_persistent(root)) continue;
      int left = remaining(root) - 1;
      assert(left >= 0 && "gallocr: refcount underflow (double free or bad count)");
      set_remaining(root, left);
      if (left == 0 && root != inplace_src) {
        free_bytes(get_offset(root), mi_nbytes(root));
      }
    }
  }

  galloc->stats.peak_bytes = watermark;
  galloc->stats.baseline_bytes = compute_baseline(g);
}

}  // namespace

mi_gallocr* mi_gallocr_new(void) {
  auto* g = new mi_gallocr();
  g->arena = nullptr;
  g->arena_size = 0;
  g->poison_enabled = env_flag_set("GALLOCR_POISON");
  g->stats = mi_gallocr_stats{};
  auto* impl = new mi_gallocr_impl();
  impl->no_inplace = env_flag_set("GALLOCR_NO_INPLACE");
  impl->no_bestfit = env_flag_set("GALLOCR_NO_BESTFIT");
  g->impl = impl;
  return g;
}

void mi_gallocr_free(mi_gallocr* galloc) {
  if (!galloc) return;
  std::free(galloc->arena);
  delete (mi_gallocr_impl*)galloc->impl;
  delete galloc;
}

void mi_gallocr_rehearse(mi_gallocr* galloc, mi_cgraph* g) { plan_graph(galloc, g, nullptr, 0); }

void mi_gallocr_reserve(mi_gallocr* galloc, mi_cgraph** graphs, int n_graphs) {
  size_t max_peak = 0, max_baseline = 0, max_capacity = 0;
  for (int i = 0; i < n_graphs; i++) {
    mi_gallocr_rehearse(galloc, graphs[i]);
    if (galloc->stats.peak_bytes > max_peak) max_peak = galloc->stats.peak_bytes;
    if (galloc->stats.baseline_bytes > max_baseline) max_baseline = galloc->stats.baseline_bytes;
    if ((size_t)graphs[i]->capacity > max_capacity) max_capacity = (size_t)graphs[i]->capacity;
  }

  std::free(galloc->arena);
  galloc->arena = (unsigned char*)std::malloc(max_peak > 0 ? max_peak : 1);
  galloc->arena_size = max_peak;

  // Reserve free-list capacity once, up front, so later alloc_graph() calls
  // (the steady-state, must-be-allocation-free path) never trigger a
  // std::vector growth/reallocation.
  auto* impl = (mi_gallocr_impl*)galloc->impl;
  impl->free_list.reserve(max_capacity + 16);

  galloc->stats.peak_bytes = max_peak;
  galloc->stats.baseline_bytes = max_baseline;
  galloc->stats.n_inplace_reuse = 0;
  galloc->stats.n_bestfit_reuse = 0;
}

void mi_gallocr_alloc_graph(mi_gallocr* galloc, mi_cgraph* g) {
  if (!galloc->arena) {
    // Without this check, plan_graph would silently take the dry_run path
    // (arena == nullptr) and leave every tensor's data null, turning into
    // a null-pointer crash several function calls away inside whatever
    // kernel runs first -- much harder to diagnose than failing right here.
    std::fprintf(stderr,
                  "gallocr: mi_gallocr_alloc_graph called before mi_gallocr_reserve\n");
    std::abort();
  }
  plan_graph(galloc, g, galloc->arena, galloc->arena_size);
}

mi_gallocr_stats mi_gallocr_get_stats(const mi_gallocr* galloc) { return galloc->stats; }

size_t mi_gallocr_theoretical_min_bytes(const mi_cgraph* g) {
  int n = g->n_nodes;
  std::vector<int> death(n, -1);
  // O(n) per lookup (linear scan of g->nodes) is fine here: this whole
  // function is O(n^2) by design, an offline diagnostic, never on any
  // decode-step hot path (see gallocr.h's doc comment).
  auto index_of = [&](const mi_tensor* t) -> int {
    for (int i = 0; i < n; i++) {
      if (g->nodes[i] == t) return i;
    }
    return -1;
  };
  for (int i = 0; i < n; i++) {
    const mi_tensor* t = g->nodes[i];
    if (mi_op_is_view(t->op)) continue;  // a view doesn't itself consume anything
    for (int s = 0; s < MI_MAX_SRC; s++) {
      mi_tensor* src = t->src[s];
      if (!src) continue;
      mi_tensor* root = resolve_root(src);
      if (mi_tensor_is_persistent(root)) continue;
      int ri = index_of(root);
      if (ri >= 0) death[ri] = i;  // increasing i, so the last write wins
    }
  }
  size_t peak = 0;
  for (int i = 0; i < n; i++) {
    size_t live = 0;
    for (int j = 0; j <= i; j++) {
      const mi_tensor* t = g->nodes[j];
      if (mi_op_is_view(t->op) || mi_tensor_is_persistent(t)) continue;
      int end = death[j] >= j ? death[j] : j;  // never read downstream -> alive only at birth
      if (end >= i) live += mi_nbytes(t);
    }
    if (live > peak) peak = live;
  }
  return peak;
}
