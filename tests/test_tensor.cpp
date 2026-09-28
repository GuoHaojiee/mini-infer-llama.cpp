// Real checks against tensor.h's already-defined (inline) logic. The
// non-inline functions (mi_nelements, mi_row_size, mi_op_is_view, ...) are
// implemented starting stage 0/1 and get their real coverage then.
#include "tensor.h"
#include "test_util.h"

int main() {
  mi_tensor t{};
  t.flags = 0;
  MI_CHECK(!mi_tensor_is_persistent(&t));

  t.flags = MI_TENSOR_FLAG_WEIGHT;
  MI_CHECK(mi_tensor_is_persistent(&t));

  t.flags = MI_TENSOR_FLAG_KV;
  MI_CHECK(mi_tensor_is_persistent(&t));

  mi_tensor t2{};
  MI_CHECK(!mi_tensor_is_view(&t2));
  mi_tensor owner{};
  t2.view_src = &owner;
  MI_CHECK(mi_tensor_is_view(&t2));

  // src[2] must be enough for every op this project defines (get_rows,
  // mul_mat, cpy, ... each use at most 2 srcs); see docs/architecture.md.
  static_assert(MI_MAX_SRC == 2, "SPEC.md specifies src[2]");

  return ::mi_test::exit_code();
}
