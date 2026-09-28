// Placeholder: real pool/bump-allocation and zero-heap-traffic tests arrive
// in stage 1 (mi_context_new/new_tensor/reset) per PROMPTS.md.
#include "context.h"
#include "test_util.h"

int main() {
  mi_context_params p{};
  p.tensor_capacity = 16;
  p.data_capacity = 1024;
  p.no_alloc = false;
  MI_CHECK(p.tensor_capacity == 16);
  MI_CHECK(!p.no_alloc);
  return ::mi_test::exit_code();
}
