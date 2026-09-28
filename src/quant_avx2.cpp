// mini-infer: AVX2/F16C SIMD kernels (stage 4; row-level accumulator +
// inlined F16C scale conversion added in stage 6).
//
// Every function here is individually __attribute__((target(...)))'d, per
// SPEC.md's build rule: this translation unit is NOT compiled with a global
// -mavx2/-march=native (see CMakeLists.txt), so any function *without* a
// target attribute in this file would fail to compile if it used an
// intrinsic, catching the mistake at build time rather than silently
// producing a binary that SIGILLs on an older CPU. Callers reach these only
// through quant.cpp's runtime dispatchers, which check
// __builtin_cpu_supports() first.
//
// Stage 6 rewrite, motivated by docs/stage6.md's step-0 profiling (perf
// record on real decode): the old per-block scale conversion called
// mi_fp16_to_fp32 (defined in quant.cpp, a different translation unit, not
// inlined, no LTO) twice per 32-element block -- that single non-inlined
// call was 46.37% of all decode samples, *more* than the AVX2 integer dot
// product itself (40.85%). Two independent fixes, both below:
//   1. Scale conversion now happens via hardware F16C (_mm_cvtph_ps),
//      defined in *this* file with a matching target attribute so GCC can
//      actually inline it into the caller -- no more call instruction in
//      the per-block loop (verified with objdump, see docs/stage6.md).
//      Falls back to a 65536-entry lookup table (quant.cpp's
//      mi_fp16_to_fp32_lut) on a hypothetical AVX2-without-F16C machine --
//      F16C is a separate CPUID bit from AVX2, so this is a real, if
//      essentially unreachable on real hardware, fallback rather than an
//      assumption (same reasoning stage 4 already applied to
//      mi_vec_dot_f16_f32). mi_vec_dot_q8_0_q8_0_avx2_f16c/_lut and
//      mi_vec_dot_q8_0_wact_avx2_f16c/_lut are exposed in quant.h purely so
//      tests can exercise both regardless of what the test machine has.
//   2. One __m256 accumulator for the *whole row*, not one horizontal sum
//      per block: each block's 32-wide int8 dot product yields 8 int32
//      partial sums (never combined into a single scalar); those get
//      converted to float, multiplied by that block's scalar dx*dy, and
//      FMA-accumulated lane-wise into the row accumulator. Only after all
//      blocks in the row are done does a single hsum_ps combine the 8
//      lanes into the final scalar. This is mathematically the same sum,
//      regrouped (block i, lane j contributes dx_i*dy_i*partial[i][j]
//      either way), so it satisfies PROMPTS.md's <1e-5 relative-error
//      standard, but is no longer forced through one sequential scalar
//      accumulator, so an hsum instruction sequence no longer runs on
//      every single block.
#include <cstdlib>

#include <immintrin.h>

#include "quant.h"

namespace {

bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v && v[0] == '1' && v[1] == '\0';
}

// Horizontal sum of 8 int32 lanes -> scalar. Standard two-step shuffle
// (128-bit halves added, then two rounds of shuffle+add within 128 bits).
// target("avx2") even though this helper is never called on its own from
// outside this file: GCC requires every function that *uses* an AVX2
// intrinsic to carry the attribute itself (it doesn't get inherited from
// the caller), inline or not.
__attribute__((target("avx2"))) inline int32_t hsum_epi32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i sum4 = _mm_add_epi32(lo, hi);
  __m128i shuf = _mm_shuffle_epi32(sum4, _MM_SHUFFLE(1, 0, 3, 2));
  __m128i sum2 = _mm_add_epi32(sum4, shuf);
  shuf = _mm_shuffle_epi32(sum2, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i sum1 = _mm_add_epi32(sum2, shuf);
  return _mm_cvtsi128_si32(sum1);
}

// Horizontal sum of 8 float lanes -> scalar.
__attribute__((target("avx2"))) inline float hsum_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 sum4 = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(sum4);
  __m128 sums = _mm_add_ps(sum4, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

// One 32-wide signed-int8 x signed-int8 dot product, per PROMPTS.md's
// specified technique: _mm256_maddubs_epi16 wants its first operand
// unsigned, so take abs(x) (unsigned-valued, safe since Q8_0 values are
// bounded to [-127, 127] by construction -- see mi_quantize_row_q8_0 --
// so abs() never hits the -128 overflow case) and fold x's sign into y via
// _mm256_sign_epi8 before multiplying; maddubs_epi16 sums adjacent pairs
// into i16, madd_epi16 (against a vector of 1s) widens+sums adjacent pairs
// again into i32. Deliberately NOT horizontally summed here (unlike stage
// 4's version) -- the 8 int32 lanes are the per-block partial sums a row-
// level accumulator combines only once, at the end of the whole row; see
// file header. mi_q8_0_block_isum_avx2 below (kept for
// tests/test_quant.cpp's boundary/random checks) is just this plus hsum_epi32.
__attribute__((target("avx2"))) inline __m256i q8_0_block_dot32x8(const int8_t x[MI_QK8_0],
                                                                    const int8_t y[MI_QK8_0]) {
  __m256i vx = _mm256_loadu_si256((const __m256i*)x);
  __m256i vy = _mm256_loadu_si256((const __m256i*)y);
  __m256i ax = _mm256_sign_epi8(vx, vx);
  __m256i sy = _mm256_sign_epi8(vy, vx);
  __m256i dot16 = _mm256_maddubs_epi16(ax, sy);
  return _mm256_madd_epi16(dot16, _mm256_set1_epi16(1));
}

// Hardware fp16->fp32 for exactly one scale factor, fully inlined into
// whichever row-level function calls it (both are compiled with a target
// string that includes f16c, so GCC can inline across this call -- verified
// with objdump, see docs/stage6.md; this is the direct fix for the 46.37%
// perf finding, which was entirely a non-inlined *function call*, not
// "scalar work being slow").
__attribute__((target("avx2,fma,f16c"))) inline float f16c_scale_to_f32(uint16_t h) {
  __m128i v = _mm_cvtsi32_si128((int)h);
  __m128 f = _mm_cvtph_ps(v);
  return _mm_cvtss_f32(f);
}

}  // namespace

__attribute__((target("avx2"))) int32_t mi_q8_0_block_isum_avx2(const int8_t x[MI_QK8_0],
                                                                   const int8_t y[MI_QK8_0]) {
  return hsum_epi32(q8_0_block_dot32x8(x, y));
}

// ---------------------------------------------------------------------------
// block_q8_0 x block_q8_0 (both fp16-scaled) -- kept for
// tests/test_quant.cpp only, see quant.h's doc comment on why mul_mat no
// longer calls this family.
// ---------------------------------------------------------------------------
__attribute__((target("avx2,fma,f16c"))) float mi_vec_dot_q8_0_q8_0_avx2_f16c(const block_q8_0* x,
                                                                                 const block_q8_0* y,
                                                                                 int64_t n) {
  int64_t nb = n / MI_QK8_0;
  __m256 acc = _mm256_setzero_ps();
  for (int64_t i = 0; i < nb; i++) {
    __m256 dotf = _mm256_cvtepi32_ps(q8_0_block_dot32x8(x[i].qs, y[i].qs));
    float dxdy = f16c_scale_to_f32(x[i].d) * f16c_scale_to_f32(y[i].d);
    acc = _mm256_fmadd_ps(dotf, _mm256_set1_ps(dxdy), acc);
  }
  return hsum_ps(acc);
}

__attribute__((target("avx2,fma"))) float mi_vec_dot_q8_0_q8_0_avx2_lut(const block_q8_0* x,
                                                                          const block_q8_0* y,
                                                                          int64_t n) {
  const float* lut = mi_fp16_to_fp32_lut();
  int64_t nb = n / MI_QK8_0;
  __m256 acc = _mm256_setzero_ps();
  for (int64_t i = 0; i < nb; i++) {
    __m256 dotf = _mm256_cvtepi32_ps(q8_0_block_dot32x8(x[i].qs, y[i].qs));
    float dxdy = lut[x[i].d] * lut[y[i].d];
    acc = _mm256_fmadd_ps(dotf, _mm256_set1_ps(dxdy), acc);
  }
  return hsum_ps(acc);
}

float mi_vec_dot_q8_0_q8_0_avx2(const block_q8_0* x, const block_q8_0* y, int64_t n) {
  static const bool use_f16c = __builtin_cpu_supports("f16c") && !env_flag("MI_QUANT_FORCE_LUT");
  return use_f16c ? mi_vec_dot_q8_0_q8_0_avx2_f16c(x, y, n) : mi_vec_dot_q8_0_q8_0_avx2_lut(x, y, n);
}

// ---------------------------------------------------------------------------
// block_q8_0 (weight, fp16 scale) x block_q8_0_act (activation, float
// scale) -- what mul_mat's Q8_0 path actually calls. Only ONE fp16->fp32
// conversion per block now (the weight's), since the activation's scale is
// already a float -- half the conversions of the block_q8_0 x block_q8_0
// path above, on top of that path's own fixes.
// ---------------------------------------------------------------------------
__attribute__((target("avx2,fma,f16c"))) float mi_vec_dot_q8_0_wact_avx2_f16c(const block_q8_0* w,
                                                                                 const block_q8_0_act* a,
                                                                                 int64_t n) {
  int64_t nb = n / MI_QK8_0;
  __m256 acc = _mm256_setzero_ps();
  for (int64_t i = 0; i < nb; i++) {
    __m256 dotf = _mm256_cvtepi32_ps(q8_0_block_dot32x8(w[i].qs, a[i].qs));
    float dxdy = f16c_scale_to_f32(w[i].d) * a[i].d;
    acc = _mm256_fmadd_ps(dotf, _mm256_set1_ps(dxdy), acc);
  }
  return hsum_ps(acc);
}

__attribute__((target("avx2,fma"))) float mi_vec_dot_q8_0_wact_avx2_lut(const block_q8_0* w,
                                                                          const block_q8_0_act* a,
                                                                          int64_t n) {
  const float* lut = mi_fp16_to_fp32_lut();
  int64_t nb = n / MI_QK8_0;
  __m256 acc = _mm256_setzero_ps();
  for (int64_t i = 0; i < nb; i++) {
    __m256 dotf = _mm256_cvtepi32_ps(q8_0_block_dot32x8(w[i].qs, a[i].qs));
    float dxdy = lut[w[i].d] * a[i].d;
    acc = _mm256_fmadd_ps(dotf, _mm256_set1_ps(dxdy), acc);
  }
  return hsum_ps(acc);
}

float mi_vec_dot_q8_0_wact_avx2(const block_q8_0* w, const block_q8_0_act* a, int64_t n) {
  static const bool use_f16c = __builtin_cpu_supports("f16c") && !env_flag("MI_QUANT_FORCE_LUT");
  return use_f16c ? mi_vec_dot_q8_0_wact_avx2_f16c(w, a, n) : mi_vec_dot_q8_0_wact_avx2_lut(w, a, n);
}

__attribute__((target("avx2,fma,f16c"))) float mi_vec_dot_f16_f32_avx2(const uint16_t* a,
                                                                         const float* b,
                                                                         int64_t n) {
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m128i ha = _mm_loadu_si128((const __m128i*)(a + i));
    __m256 fa = _mm256_cvtph_ps(ha);
    __m256 fb = _mm256_loadu_ps(b + i);
    acc = _mm256_fmadd_ps(fa, fb, acc);
  }
  float result = hsum_ps(acc);
  for (; i < n; i++) {
    result = __builtin_fmaf(mi_fp16_to_fp32(a[i]), b[i], result);
  }
  return result;
}
