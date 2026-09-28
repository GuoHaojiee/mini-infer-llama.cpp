// Stage 4: block_q8_0 layout (already checked before stage 4 landed), F16
// <-> F32 conversion correctness, Q8_0 quantize/dequantize round-trip
// accuracy, and scalar-vs-AVX2 equivalence for both vec_dot kernels per
// PROMPTS.md's exact standard: Q8_0's int32 partial sum must match bit-for-
// bit, the final scaled float must match within 1e-5 relative error (both
// for Q8_0 and, "同理", for F16), tested on random data plus the boundary
// values PROMPTS.md calls out (+-127, an all-zero block).
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "quant.h"
#include "test_util.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define MI_TEST_HAVE_X86_INTRIN 1
#endif

namespace {

bool cpu_has_avx2() { return __builtin_cpu_supports("avx2"); }
bool cpu_has_avx2_f16c() {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
}

float rel_err(float got, float want) {
  float denom = std::fabs(want);
  if (denom < 1e-6f) denom = 1e-6f;
  return std::fabs(got - want) / denom;
}

#ifdef MI_TEST_HAVE_X86_INTRIN
// Hardware ground truth for fp16<->fp32, used only to cross-check the
// scalar bit-manipulation implementation in quant.cpp on machines that
// have F16C -- independent of any code in this project.
__attribute__((target("f16c"))) float hw_fp16_to_fp32(uint16_t h) {
  __m128i v = _mm_cvtsi32_si128(h);
  __m128 f = _mm_cvtph_ps(v);
  return _mm_cvtss_f32(f);
}
__attribute__((target("f16c"))) uint16_t hw_fp32_to_fp16(float f) {
  __m128 v = _mm_set_ss(f);
  __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
  return (uint16_t)_mm_cvtsi128_si32(h);
}
#endif

}  // namespace

static void test_block_q8_0_layout() {
  MI_CHECK(MI_QK8_0 == 32);
  MI_CHECK(sizeof(block_q8_0) == 34);
  MI_CHECK(offsetof(block_q8_0, d) == 0);
  MI_CHECK(offsetof(block_q8_0, qs) == 2);
}

static void test_fp16_known_values() {
  struct Case { uint16_t bits; float value; };
  const Case cases[] = {
      {0x0000, 0.0f},   {0x8000, -0.0f},  {0x3C00, 1.0f},  {0xBC00, -1.0f},
      {0x4000, 2.0f},   {0x3800, 0.5f},   {0x4200, 3.0f},  {0x0001, 5.9604645e-8f},
      {0x03FF, 6.09755516e-5f},
  };
  for (const auto& c : cases) {
    float got = mi_fp16_to_fp32(c.bits);
    MI_CHECK(rel_err(got, c.value) < 1e-6f || (c.value == 0.0f && got == 0.0f));
  }
  // Round trip: value -> f16 -> f32 for a few clean values should recover
  // exactly (they're all exactly representable in f16).
  const float roundtrip_values[] = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.5f, 3.0f, 8.0f, -16.0f};
  for (float v : roundtrip_values) {
    uint16_t h = mi_fp32_to_fp16(v);
    float back = mi_fp16_to_fp32(h);
    MI_CHECK(back == v);
  }
  // inf / nan
  MI_CHECK(std::isinf(mi_fp16_to_fp32(0x7C00)));
  MI_CHECK(std::isinf(mi_fp16_to_fp32(0xFC00)));
  MI_CHECK(std::isnan(mi_fp16_to_fp32(0x7E00)));
}

#ifdef MI_TEST_HAVE_X86_INTRIN
static void test_fp16_matches_hardware_f16c() {
  if (!cpu_has_avx2_f16c()) {
    return;  // nothing to cross-check on this machine; scalar path is
             // still covered by test_fp16_known_values above.
  }
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  for (int i = 0; i < 2000; i++) {
    float f = dist(rng);
    uint16_t sw = mi_fp32_to_fp16(f);
    uint16_t hw = hw_fp32_to_fp16(f);
    MI_CHECK(sw == hw);

    uint16_t h = (uint16_t)rng();
    float sw_f = mi_fp16_to_fp32(h);
    float hw_f = hw_fp16_to_fp32(h);
    // Bit-compare via memcpy so NaN payloads (which != themselves) don't
    // spuriously fail; sign/exponent/mantissa should still match exactly.
    if (std::isnan(sw_f) || std::isnan(hw_f)) {
      MI_CHECK(std::isnan(sw_f) && std::isnan(hw_f));
    } else {
      MI_CHECK(sw_f == hw_f);
    }
  }
}
#endif

static void test_quantize_dequantize_round_trip() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  const int64_t n = 32 * 5;  // 5 blocks
  std::vector<float> x(n);
  for (float& v : x) v = dist(rng);

  std::vector<block_q8_0> blocks(n / MI_QK8_0);
  mi_quantize_row_q8_0(x.data(), blocks.data(), n);

  std::vector<float> back(n);
  mi_dequantize_row_q8_0(blocks.data(), back.data(), n);

  for (int64_t i = 0; i < n; i++) {
    int64_t blk = i / MI_QK8_0;
    float d = mi_fp16_to_fp32(blocks[blk].d);
    // Quantization error is bounded by half a quantization step (d), plus
    // a little slack for the f16 scale factor's own rounding.
    MI_CHECK(std::fabs(back[i] - x[i]) <= d * 0.6f + 1e-6f);
  }

  // amax element in each block should quantize to exactly +-127.
  for (int64_t blk = 0; blk < n / MI_QK8_0; blk++) {
    float amax = 0.0f;
    int amax_idx = 0;
    for (int j = 0; j < MI_QK8_0; j++) {
      float av = std::fabs(x[blk * MI_QK8_0 + j]);
      if (av > amax) {
        amax = av;
        amax_idx = j;
      }
    }
    int8_t q = blocks[blk].qs[amax_idx];
    MI_CHECK(q == 127 || q == -127);
  }
}

static void test_quantize_all_zero_block() {
  std::vector<float> x(MI_QK8_0, 0.0f);
  block_q8_0 b;
  mi_quantize_row_q8_0(x.data(), &b, MI_QK8_0);
  MI_CHECK(mi_fp16_to_fp32(b.d) == 0.0f);
  for (int j = 0; j < MI_QK8_0; j++) {
    MI_CHECK(b.qs[j] == 0);
  }
  float back[MI_QK8_0];
  mi_dequantize_row_q8_0(&b, back, MI_QK8_0);
  for (int j = 0; j < MI_QK8_0; j++) {
    MI_CHECK(back[j] == 0.0f);
  }
}

static void test_q8_0_isum_boundary_and_random() {
  if (!cpu_has_avx2()) return;  // AVX2 path untestable on this machine

  auto check_block = [](const int8_t x[MI_QK8_0], const int8_t y[MI_QK8_0]) {
    int32_t s_scalar = mi_q8_0_block_isum_scalar(x, y);
    int32_t s_avx2 = mi_q8_0_block_isum_avx2(x, y);
    MI_CHECK(s_scalar == s_avx2);
  };

  int8_t all_127[MI_QK8_0], all_neg127[MI_QK8_0], zeros[MI_QK8_0], mixed[MI_QK8_0];
  for (int j = 0; j < MI_QK8_0; j++) {
    all_127[j] = 127;
    all_neg127[j] = -127;
    zeros[j] = 0;
    mixed[j] = (j % 2 == 0) ? 127 : -127;
  }
  check_block(all_127, all_127);
  check_block(all_127, all_neg127);
  check_block(all_neg127, all_neg127);
  check_block(zeros, zeros);
  check_block(zeros, all_127);
  check_block(mixed, all_127);
  check_block(mixed, mixed);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-127, 127);
  for (int trial = 0; trial < 200; trial++) {
    int8_t x[MI_QK8_0], y[MI_QK8_0];
    for (int j = 0; j < MI_QK8_0; j++) {
      x[j] = (int8_t)dist(rng);
      y[j] = (int8_t)dist(rng);
    }
    check_block(x, y);
  }
}

static void test_vec_dot_q8_0_scalar_vs_avx2() {
  if (!cpu_has_avx2()) return;

  std::mt19937 rng(321);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  const int64_t n = 32 * 8;
  std::vector<float> xf(n), yf(n);
  for (float& v : xf) v = dist(rng);
  for (float& v : yf) v = dist(rng);

  std::vector<block_q8_0> xb(n / MI_QK8_0), yb(n / MI_QK8_0);
  mi_quantize_row_q8_0(xf.data(), xb.data(), n);
  mi_quantize_row_q8_0(yf.data(), yb.data(), n);

  float r_scalar = mi_vec_dot_q8_0_q8_0_scalar(xb.data(), yb.data(), n);
  float r_avx2 = mi_vec_dot_q8_0_q8_0_avx2(xb.data(), yb.data(), n);
  // Stage 6: the AVX2 path accumulates in 8 SIMD lanes independently
  // (combined via a single hsum at the very end of the row) rather than one
  // sequential scalar accumulator, so this is no longer expected to be
  // bit-identical to the scalar reference -- only within PROMPTS.md's
  // <1e-5 relative-error standard (see docs/stage6.md).
  MI_CHECK(rel_err(r_avx2, r_scalar) < 1e-5f);

  // Stage 6: both backends mi_vec_dot_q8_0_q8_0_avx2 picks between at
  // runtime, exercised directly regardless of what this test machine's CPU
  // actually has (mirrors the isum test's boundary-value philosophy).
  float r_f16c = mi_vec_dot_q8_0_q8_0_avx2_f16c(xb.data(), yb.data(), n);
  float r_lut = mi_vec_dot_q8_0_q8_0_avx2_lut(xb.data(), yb.data(), n);
  MI_CHECK(rel_err(r_f16c, r_scalar) < 1e-5f);
  MI_CHECK(rel_err(r_lut, r_scalar) < 1e-5f);

  float r_dispatch = mi_vec_dot_q8_0_q8_0(xb.data(), yb.data(), n);
  MI_CHECK(r_dispatch == r_avx2);  // dispatcher must pick AVX2 on this machine
}

static void test_block_q8_0_act_layout() {
  MI_CHECK(sizeof(block_q8_0_act) == 36);
  MI_CHECK(offsetof(block_q8_0_act, d) == 0);
  MI_CHECK(offsetof(block_q8_0_act, qs) == 4);
}

static void test_quantize_row_q8_0_act_matches_q8_0_within_float_precision() {
  // block_q8_0_act's quantization decisions (amax, rounding, qs[]) should
  // be identical to block_q8_0's -- only the scale storage differs (float
  // vs fp16-round-tripped) -- so the two should agree on qs[] exactly and
  // on d up to fp16's own precision loss.
  std::mt19937 rng(999);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  const int64_t n = 32 * 6;
  std::vector<float> x(n);
  for (float& v : x) v = dist(rng);

  std::vector<block_q8_0> ref(n / MI_QK8_0);
  std::vector<block_q8_0_act> act(n / MI_QK8_0);
  mi_quantize_row_q8_0(x.data(), ref.data(), n);
  mi_quantize_row_q8_0_act(x.data(), act.data(), n);

  for (size_t i = 0; i < ref.size(); i++) {
    float d_ref = mi_fp16_to_fp32(ref[i].d);
    MI_CHECK(rel_err(act[i].d, d_ref) < 1e-3f);  // fp16 has ~3 decimal digits
    for (int j = 0; j < MI_QK8_0; j++) {
      MI_CHECK(act[i].qs[j] == ref[i].qs[j]);
    }
  }
}

static void test_vec_dot_q8_0_wact_scalar_vs_avx2() {
  if (!cpu_has_avx2()) return;

  std::mt19937 rng(4242);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  const int64_t n = 32 * 8;
  std::vector<float> wf(n), af(n);
  for (float& v : wf) v = dist(rng);
  for (float& v : af) v = dist(rng);

  std::vector<block_q8_0> wb(n / MI_QK8_0);
  std::vector<block_q8_0_act> ab(n / MI_QK8_0);
  mi_quantize_row_q8_0(wf.data(), wb.data(), n);
  mi_quantize_row_q8_0_act(af.data(), ab.data(), n);

  float r_scalar = mi_vec_dot_q8_0_wact_scalar(wb.data(), ab.data(), n);
  float r_f16c = mi_vec_dot_q8_0_wact_avx2_f16c(wb.data(), ab.data(), n);
  float r_lut = mi_vec_dot_q8_0_wact_avx2_lut(wb.data(), ab.data(), n);
  MI_CHECK(rel_err(r_f16c, r_scalar) < 1e-5f);
  MI_CHECK(rel_err(r_lut, r_scalar) < 1e-5f);

  float r_avx2 = mi_vec_dot_q8_0_wact_avx2(wb.data(), ab.data(), n);
  float r_dispatch = mi_vec_dot_q8_0_wact(wb.data(), ab.data(), n);
  MI_CHECK(r_dispatch == r_avx2);  // dispatcher must pick AVX2 on this machine
}

static void test_vec_dot_f16_f32_scalar_vs_avx2() {
  if (!cpu_has_avx2_f16c()) return;

  std::mt19937 rng(555);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  const int64_t n = 8 * 13 + 5;  // deliberately not a multiple of 8: exercises the AVX2 tail loop
  std::vector<uint16_t> a(n);
  std::vector<float> b(n);
  for (int64_t i = 0; i < n; i++) {
    a[i] = mi_fp32_to_fp16(dist(rng));
    b[i] = dist(rng);
  }

  float r_scalar = mi_vec_dot_f16_f32_scalar(a.data(), b.data(), n);
  float r_avx2 = mi_vec_dot_f16_f32_avx2(a.data(), b.data(), n);
  MI_CHECK(rel_err(r_avx2, r_scalar) < 1e-5f);

  float r_dispatch = mi_vec_dot_f16_f32(a.data(), b.data(), n);
  MI_CHECK(r_dispatch == r_avx2);
}

int main() {
  test_block_q8_0_layout();
  test_fp16_known_values();
#ifdef MI_TEST_HAVE_X86_INTRIN
  test_fp16_matches_hardware_f16c();
#endif
  test_quantize_dequantize_round_trip();
  test_quantize_all_zero_block();
  test_q8_0_isum_boundary_and_random();
  test_vec_dot_q8_0_scalar_vs_avx2();
  test_block_q8_0_act_layout();
  test_quantize_row_q8_0_act_matches_q8_0_within_float_precision();
  test_vec_dot_q8_0_wact_scalar_vs_avx2();
  test_vec_dot_f16_f32_scalar_vs_avx2();
  return ::mi_test::exit_code();
}
