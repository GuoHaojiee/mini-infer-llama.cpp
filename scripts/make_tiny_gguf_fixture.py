#!/usr/bin/env python3
"""Generates tests/fixtures/tiny.gguf: a small, hand-picked GGUF file used
by tests/test_gguf.cpp for hermetic, independent-of-the-real-model unit
tests. Built with the reference `gguf` library (gguf-py), not mini-infer's
own writer -- mini-infer never writes GGUF, only reads it, so this fixture
is genuinely an external oracle, not a self-fulfilling round-trip.

Run once; the output binary is committed to the repo. Re-run and re-commit
only if the fixture's contents (see below) need to change.
"""
import numpy as np

import gguf


def bytes_to_unicode() -> dict:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def main():
    out_path = "tests/fixtures/tiny.gguf"
    w = gguf.GGUFWriter(out_path, "mini_infer_test")

    # Custom (non-default) alignment, to exercise that code path -- the
    # real Qwen2 GGUF files never set this key, so they only test the
    # default-32 path.
    w.add_custom_alignment(16)

    # Scalars covering every GGUF value type.
    w.add_uint8("test.u8", 200)
    w.add_int8("test.i8", -100)
    w.add_uint16("test.u16", 60000)
    w.add_int16("test.i16", -30000)
    w.add_uint32("test.u32", 4000000000)
    w.add_int32("test.i32", -2000000000)
    w.add_float32("test.f32", 3.140000104904175)  # exact float32 value
    w.add_bool("test.bool_true", True)
    w.add_bool("test.bool_false", False)
    w.add_string("test.string", "hello \"world\"\nwith newline and \ttab")
    w.add_uint64("test.u64", 123456789012345)
    w.add_int64("test.i64", -123456789012345)
    w.add_float64("test.f64", 2.718281828459045)

    # Arrays covering string/int/float/bool element types.
    w.add_array("test.arr_str", ["a", "bb", "ccc"])
    w.add_array("test.arr_i32", [1, -2, 3, -4])
    w.add_array("test.arr_f32", [1.5, -2.5, 3.5])
    w.add_array("test.arr_bool", [True, False, True])

    # A minimal 2-token vocab exercising both a NORMAL byte-level-BPE token
    # (bytes 0x20 0x41 == " A", stored as "ĠA") and a CONTROL token
    # (literal text, not byte-mapped).
    bu = bytes_to_unicode()
    normal_tok = bu[0x20] + bu[0x41]
    w.add_array("tokenizer.ggml.tokens", [normal_tok, "<|test|>"])
    w.add_array("tokenizer.ggml.token_type", [1, 3])  # NORMAL, CONTROL

    # Tensors covering F32 (1D and 2D), F16, and Q8_0.
    t_f32 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    w.add_tensor("t_f32", t_f32)

    t_f32_2d = np.arange(6, dtype=np.float32).reshape(3, 2)  # ne = [2, 3]
    w.add_tensor("t_f32_2d", t_f32_2d)

    t_f16 = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float16)
    w.add_tensor("t_f16", t_f16)

    q8_src = (np.arange(32, dtype=np.float32) - 16.0)
    q8_data = gguf.quants.quantize(q8_src, gguf.GGMLQuantizationType.Q8_0)
    w.add_tensor("t_q8_0", q8_data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
