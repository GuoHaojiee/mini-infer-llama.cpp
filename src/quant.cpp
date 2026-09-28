// mini-infer: Q8_0 quantization and F16 helpers, scalar reference
// implementations + runtime dispatch (stage 4). The AVX2/F16C versions live
// in quant_avx2.cpp, each individually __attribute__((target(...)))'d --
// this file is compiled with no extra ISA flags, so it doubles as the real
// fallback for a non-AVX2 machine, not just an untested code path.
#include "quant.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v && v[0] == '1' && v[1] == '\0';
}

}  // namespace

// ---------------------------------------------------------------------------
// F16 <-> F32. Standard IEEE 754 binary16 <-> binary32 bit manipulation
// (public numerical format spec, not any particular project's source) --
// handles zero, subnormals, normals, inf and nan on both ends.
// ---------------------------------------------------------------------------
float mi_fp16_to_fp32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp10 = (h >> 10) & 0x1Fu;
  uint32_t mant10 = h & 0x3FFu;
  uint32_t bits;

  if (exp10 == 0) {
    if (mant10 == 0) {
      bits = sign;  // +-0
    } else {
      // Subnormal half: normalize the mantissa by shifting left until its
      // leading bit reaches position 10 (the implicit-1 slot for a normal
      // number), tracking the shift count `s`. The half's value is
      // mant10 * 2^-24; after shifting, (mant10<<s) sits in [0x400, 0x7FF)
      // i.e. 1.frac * 2^(10-s), so the half's value is
      // (1.frac) * 2^(10-s-24) = (1.frac) * 2^(-s-14), giving biased f32
      // exponent 127 + (-s-14) = 113 - s.
      uint32_t mant = mant10;
      int32_t s = 0;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        s++;
      }
      mant &= 0x3FFu;
      uint32_t exp32 = (uint32_t)(113 - s);
      bits = sign | (exp32 << 23) | (mant << 13);
    }
  } else if (exp10 == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant10 << 13);  // inf / nan
  } else {
    uint32_t exp32 = exp10 - 15u + 127u;
    bits = sign | (exp32 << 23) | (mant10 << 13);
  }

  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

const float* mi_fp16_to_fp32_lut() {
  static float table[65536];
  static const bool built = [] {
    for (int i = 0; i < 65536; i++) table[i] = mi_fp16_to_fp32((uint16_t)i);
    return true;
  }();
  (void)built;
  return table;
}

uint16_t mi_fp32_to_fp16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 16) & 0x8000u;
  uint32_t exp8 = (bits >> 23) & 0xFFu;
  uint32_t mant32 = bits & 0x7FFFFFu;

  if (exp8 == 0xFFu) {
    uint32_t mant16 = mant32 ? 0x200u : 0u;  // keep NaN non-zero, inf zero
    return (uint16_t)(sign | 0x7C00u | mant16);
  }

  int32_t exp16 = (int32_t)exp8 - 127 + 15;

  if (exp16 >= 0x1F) {
    return (uint16_t)(sign | 0x7C00u);  // overflow -> inf
  }

  if (exp16 <= 0) {
    if (exp16 < -10) {
      return (uint16_t)sign;  // too small even for a subnormal half -> 0
    }
    // Subnormal half: add the implicit leading 1 back, then shift right by
    // enough bits to land in the 10-bit subnormal mantissa field, rounding
    // to nearest-even on the bits shifted out.
    mant32 |= 0x800000u;
    int32_t shift = 14 - exp16;
    uint32_t mant16 = mant32 >> shift;
    uint32_t remainder = mant32 & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (mant16 & 1u))) {
      mant16 += 1;
    }
    return (uint16_t)(sign | mant16);
  }

  // Normal half: round-to-nearest-even on the dropped 13 mantissa bits.
  uint32_t mant16 = mant32 >> 13;
  uint32_t remainder = mant32 & 0x1FFFu;
  const uint32_t halfway = 0x1000u;
  if (remainder > halfway || (remainder == halfway && (mant16 & 1u))) {
    mant16 += 1;
    if (mant16 == 0x400u) {
      mant16 = 0;
      exp16 += 1;
      if (exp16 >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    }
  }
  return (uint16_t)(sign | ((uint32_t)exp16 << 10) | mant16);
}

// ---------------------------------------------------------------------------
// Q8_0
// ---------------------------------------------------------------------------
void mi_quantize_row_q8_0(const float* x, block_q8_0* y, int64_t n) {
  int64_t nb = n / MI_QK8_0;
  for (int64_t i = 0; i < nb; i++) {
    const float* xb = x + i * MI_QK8_0;
    float amax = 0.0f;
    for (int j = 0; j < MI_QK8_0; j++) {
      float av = std::fabs(xb[j]);
      if (av > amax) amax = av;
    }
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    y[i].d = mi_fp32_to_fp16(d);
    for (int j = 0; j < MI_QK8_0; j++) {
      float scaled = xb[j] * id;
      int32_t q = (int32_t)lrintf(scaled);
      // d = amax/127 guarantees |x/d| <= 127 for every element in this
      // block (see docs/stage4.md), so q should never leave [-127, 127];
      // this clamp only guards against the id==0 (all-zero block) case,
      // where scaled is 0 and q is already 0.
      if (q > 127) q = 127;
      if (q < -127) q = -127;
      y[i].qs[j] = (int8_t)q;
    }
  }
}

void mi_dequantize_row_q8_0(const block_q8_0* x, float* y, int64_t n) {
  int64_t nb = n / MI_QK8_0;
  for (int64_t i = 0; i < nb; i++) {
    float d = mi_fp16_to_fp32(x[i].d);
    for (int j = 0; j < MI_QK8_0; j++) {
      y[i * MI_QK8_0 + j] = (float)x[i].qs[j] * d;
    }
  }
}

int32_t mi_q8_0_block_isum_scalar(const int8_t x[MI_QK8_0], const int8_t y[MI_QK8_0]) {
  int32_t s = 0;
  for (int j = 0; j < MI_QK8_0; j++) {
    s += (int32_t)x[j] * (int32_t)y[j];
  }
  return s;
}

float mi_vec_dot_q8_0_q8_0_scalar(const block_q8_0* x, const block_q8_0* y, int64_t n) {
  int64_t nb = n / MI_QK8_0;
  float acc = 0.0f;
  for (int64_t i = 0; i < nb; i++) {
    int32_t isum = mi_q8_0_block_isum_scalar(x[i].qs, y[i].qs);
    float dxdy = mi_fp16_to_fp32(x[i].d) * mi_fp16_to_fp32(y[i].d);
    acc = std::fma((float)isum, dxdy, acc);
  }
  return acc;
}

float mi_vec_dot_q8_0_q8_0(const block_q8_0* x, const block_q8_0* y, int64_t n) {
  static const bool force_scalar = env_flag("MI_QUANT_FORCE_SCALAR");
  static const bool has_avx2 = __builtin_cpu_supports("avx2");
  if (!force_scalar && has_avx2) {
    return mi_vec_dot_q8_0_q8_0_avx2(x, y, n);
  }
  return mi_vec_dot_q8_0_q8_0_scalar(x, y, n);
}

// ---------------------------------------------------------------------------
// Q8_0 activation block (stage 6) -- see quant.h's block_q8_0_act doc
// comment. mi_vec_dot_q8_0_wact_scalar mirrors mi_vec_dot_q8_0_q8_0_scalar
// exactly except y's scale is read as a plain float (no mi_fp16_to_fp32
// call needed on that side at all).
// ---------------------------------------------------------------------------
void mi_quantize_row_q8_0_act(const float* x, block_q8_0_act* y, int64_t n) {
  int64_t nb = n / MI_QK8_0;
  for (int64_t i = 0; i < nb; i++) {
    const float* xb = x + i * MI_QK8_0;
    float amax = 0.0f;
    for (int j = 0; j < MI_QK8_0; j++) {
      float av = std::fabs(xb[j]);
      if (av > amax) amax = av;
    }
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    y[i].d = d;
    for (int j = 0; j < MI_QK8_0; j++) {
      float scaled = xb[j] * id;
      int32_t q = (int32_t)lrintf(scaled);
      if (q > 127) q = 127;
      if (q < -127) q = -127;
      y[i].qs[j] = (int8_t)q;
    }
  }
}

float mi_vec_dot_q8_0_wact_scalar(const block_q8_0* w, const block_q8_0_act* a, int64_t n) {
  int64_t nb = n / MI_QK8_0;
  float acc = 0.0f;
  for (int64_t i = 0; i < nb; i++) {
    int32_t isum = mi_q8_0_block_isum_scalar(w[i].qs, a[i].qs);
    float dxdy = mi_fp16_to_fp32(w[i].d) * a[i].d;
    acc = std::fma((float)isum, dxdy, acc);
  }
  return acc;
}

float mi_vec_dot_q8_0_wact(const block_q8_0* w, const block_q8_0_act* a, int64_t n) {
  static const bool force_scalar = env_flag("MI_QUANT_FORCE_SCALAR");
  static const bool has_avx2 = __builtin_cpu_supports("avx2");
  if (!force_scalar && has_avx2) {
    return mi_vec_dot_q8_0_wact_avx2(w, a, n);
  }
  return mi_vec_dot_q8_0_wact_scalar(w, a, n);
}

// ---------------------------------------------------------------------------
// F16 weight x F32 activation dot product
// ---------------------------------------------------------------------------
float mi_vec_dot_f16_f32_scalar(const uint16_t* a, const float* b, int64_t n) {
  float acc = 0.0f;
  for (int64_t i = 0; i < n; i++) {
    acc = std::fma(mi_fp16_to_fp32(a[i]), b[i], acc);
  }
  return acc;
}

float mi_vec_dot_f16_f32(const uint16_t* a, const float* b, int64_t n) {
  static const bool force_scalar = env_flag("MI_QUANT_FORCE_SCALAR");
  static const bool has_avx2_f16c =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
  if (!force_scalar && has_avx2_f16c) {
    return mi_vec_dot_f16_f32_avx2(a, b, n);
  }
  return mi_vec_dot_f16_f32_scalar(a, b, n);
}
