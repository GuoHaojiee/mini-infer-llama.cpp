// mini-infer: Q8_0 quantization (stage 4) and F16 helpers.
//
// block_q8_0 layout matches llama.cpp's GGUF Q8_0 exactly (1 fp16 scale + 32
// int8 values = 34 bytes) so files produced by llama-quantize load directly.
// vec_dot has a scalar reference implementation and a separate AVX2 one,
// selected at runtime via __builtin_cpu_supports; there is no "AVX2 or
// nothing" assumption baked into callers. See docs/stage4.md (written in
// stage 4) for the intrinsics reasoning.
//
// MI_QUANT_FORCE_SCALAR=1 (read once, cached) forces both dispatchers below
// to always pick the scalar path even on an AVX2-capable machine -- used
// only by scripts/bench_quant.py to get a genuine "F16/Q8_0 scalar" data
// point on the same hardware as the AVX2 numbers (same idiom as stage 2b's
// GALLOCR_NO_INPLACE/GALLOCR_NO_BESTFIT env toggles).
#pragma once

#include <cstddef>
#include <cstdint>

#define MI_QK8_0 32

#pragma pack(push, 1)
struct block_q8_0 {
    uint16_t d;      // fp16 scale, GGUF field order: d then qs
    int8_t qs[MI_QK8_0];
};
#pragma pack(pop)
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must match GGUF Q8_0 layout");

// ---------------------------------------------------------------------------
// F16 <-> F32 (scalar; F16C intrinsic versions live in a target-attributed
// translation unit added in stage 4, dispatched at runtime).
// ---------------------------------------------------------------------------
float mi_fp16_to_fp32(uint16_t h);
uint16_t mi_fp32_to_fp16(float f);

// Lazily-built 65536-entry fp16->fp32 lookup table (magic-static init, so
// the O(65536) build happens at most once, thread-safely, on first use --
// never inside a decode step's hot path here, only from the AVX2-without-
// F16C fallback row functions in quant_avx2.cpp). F16C (vcvtph2ps) is used
// instead whenever __builtin_cpu_supports("f16c") is true; this table exists
// so that fallback is a real, buildable, exercised code path rather than an
// assumption that every AVX2 CPU also has F16C (see mi_vec_dot_f16_f32's
// comment above -- same "separate CPUID bit" reasoning, now stage 6's Q8_0
// dot products too).
const float* mi_fp16_to_fp32_lut();

// ---------------------------------------------------------------------------
// Q8_0
// ---------------------------------------------------------------------------

// Quantizes n contiguous f32 values (n must be a multiple of MI_QK8_0) into
// n/MI_QK8_0 blocks. Used to quantize activations on the fly for the Q8_0
// mul_mat path.
void mi_quantize_row_q8_0(const float* x, block_q8_0* y, int64_t n);

// Dequantizes n/MI_QK8_0 blocks back to f32. Used for reading a Q8_0
// token_embd row during get_rows.
void mi_dequantize_row_q8_0(const block_q8_0* x, float* y, int64_t n);

// Dot product of two Q8_0-quantized rows of length n (both fp16-scaled,
// GGUF-compatible blocks). Scalar reference implementation; always correct,
// used both as the fallback and as the oracle in tests that compare against
// the AVX2 version below.
//
// Stage 6: mul_mat's Q8_0 path no longer calls this -- it calls
// mi_vec_dot_q8_0_wact below instead (weight x activation, where the
// activation's scale is plain float, not fp16 -- see block_q8_0_act).
// This function and its AVX2/dispatcher siblings are kept, still fully
// optimized (see docs/stage6.md), purely because tests/test_quant.cpp
// exercises the "both operands are real GGUF-layout Q8_0 blocks" case
// directly and SPEC.md forbids narrowing an existing test's pass criteria.
float mi_vec_dot_q8_0_q8_0_scalar(const block_q8_0* x, const block_q8_0* y, int64_t n);

// AVX2 + FMA implementation of the same dot product. Only called after
// __builtin_cpu_supports("avx2") (checked by the runtime dispatcher in
// mi_vec_dot_q8_0_q8_0, not by the caller). Defined in a translation unit
// built with __attribute__((target("avx2,fma"))); never in a globally
// -mavx2 compiled file. Internally picks between an F16C-based and a
// lookup-table-based scale conversion at runtime (see
// mi_vec_dot_q8_0_q8_0_avx2_f16c/_lut below) depending on whether the CPU
// actually has F16C (a separate CPUID bit from AVX2).
float mi_vec_dot_q8_0_q8_0_avx2(const block_q8_0* x, const block_q8_0* y, int64_t n);

// The two backends mi_vec_dot_q8_0_q8_0_avx2 picks between, exposed purely
// so tests can exercise both regardless of what the test machine's CPU
// actually supports (mirrors mi_q8_0_block_isum_scalar/_avx2's existing
// "exposed purely for tests" precedent below). Both use one __m256
// accumulator for the whole row (per-block FMA, a single horizontal sum at
// the end) rather than one horizontal sum per block -- see docs/stage6.md's
// step-0 profiling for why that mattered.
float mi_vec_dot_q8_0_q8_0_avx2_f16c(const block_q8_0* x, const block_q8_0* y, int64_t n);
float mi_vec_dot_q8_0_q8_0_avx2_lut(const block_q8_0* x, const block_q8_0* y, int64_t n);

// Runtime-dispatching entry point; still used by tests/test_quant.cpp
// directly, not by mul_mat (see above).
float mi_vec_dot_q8_0_q8_0(const block_q8_0* x, const block_q8_0* y, int64_t n);

// Raw int32 dot product of one 32-element block, no scaling applied --
// exposed purely so tests can check the AVX2 accumulation matches the
// scalar reference exactly (PROMPTS.md: "int32 部分完全一致") before any
// floating-point scaling enters the picture. vec_dot's scalar/AVX2
// implementations are themselves thin wrappers around these two.
int32_t mi_q8_0_block_isum_scalar(const int8_t x[MI_QK8_0], const int8_t y[MI_QK8_0]);
int32_t mi_q8_0_block_isum_avx2(const int8_t x[MI_QK8_0], const int8_t y[MI_QK8_0]);

// ---------------------------------------------------------------------------
// Q8_0 activation block (stage 6). Same 32-element int8 body as block_q8_0,
// but the scale is a plain float, not a round-tripped-through-fp16 value.
// This block never appears in a GGUF file -- it only ever lives in the
// graph-managed scratch tensor MI_OP_QUANTIZE_Q8_0 produces (see
// tensor.h's MI_TYPE_Q8_0_ACT, graph.cpp's mi_graph_quantize_q8_0), so
// there's no on-disk format to match, and skipping the fp16 round trip
// removes one of the two per-block fp16->fp32 conversions mul_mat's Q8_0
// path used to do -- docs/stage6.md's step-0 profiling found that
// conversion (not the AVX2 integer dot product) was the dominant decode
// cost, and it was already lossy/free-standing precision the activation
// never needed in the first place (the activation was F32 right up until
// this quantize step).
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct block_q8_0_act {
    float d;
    int8_t qs[MI_QK8_0];
};
#pragma pack(pop)
static_assert(sizeof(block_q8_0_act) == 36, "block_q8_0_act must be a 4-byte float scale + 32 int8");

// Quantizes n contiguous f32 values into n/MI_QK8_0 block_q8_0_act blocks.
// Same amax/round-to-nearest quantization as mi_quantize_row_q8_0, only the
// scale storage differs (float instead of fp16).
void mi_quantize_row_q8_0_act(const float* x, block_q8_0_act* y, int64_t n);

// Weight (block_q8_0, fp16 scale, real GGUF layout) x activation
// (block_q8_0_act, float scale) dot product -- what mul_mat's Q8_0 path
// actually calls (see ops_cpu.cpp's mi_compute_mul_mat).
float mi_vec_dot_q8_0_wact_scalar(const block_q8_0* w, const block_q8_0_act* a, int64_t n);
float mi_vec_dot_q8_0_wact_avx2(const block_q8_0* w, const block_q8_0_act* a, int64_t n);
float mi_vec_dot_q8_0_wact_avx2_f16c(const block_q8_0* w, const block_q8_0_act* a, int64_t n);
float mi_vec_dot_q8_0_wact_avx2_lut(const block_q8_0* w, const block_q8_0_act* a, int64_t n);
float mi_vec_dot_q8_0_wact(const block_q8_0* w, const block_q8_0_act* a, int64_t n);

// ---------------------------------------------------------------------------
// F16 weight x F32 activation dot product, used by mul_mat's F16 path.
// Unlike Q8_0, F16 mul_mat never requantizes the activation -- the
// activation is read as-is (F32) and each weight element is converted from
// F16 on the fly, so there is no graph-level "quantize" node for this path,
// only a kernel-level scalar/SIMD split (see ops_cpu.cpp's mi_compute_mul_mat).
// ---------------------------------------------------------------------------
float mi_vec_dot_f16_f32_scalar(const uint16_t* a, const float* b, int64_t n);

// F16C + AVX2 + FMA implementation. Only called after both
// __builtin_cpu_supports("avx2") and __builtin_cpu_supports("f16c") (F16C is
// a separate CPUID bit from AVX2, even though virtually every AVX2-capable
// x86 CPU in practice also has it -- checked explicitly rather than assumed).
float mi_vec_dot_f16_f32_avx2(const uint16_t* a, const float* b, int64_t n);

// Runtime-dispatching entry point mul_mat's F16 path actually calls.
float mi_vec_dot_f16_f32(const uint16_t* a, const float* b, int64_t n);
