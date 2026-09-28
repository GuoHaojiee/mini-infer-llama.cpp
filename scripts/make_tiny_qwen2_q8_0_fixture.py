#!/usr/bin/env python3
"""Generates tests/fixtures/tiny_qwen2_q8_0.gguf: a Q8_0 sibling of
make_tiny_qwen2_fixture.py's tiny_qwen2.gguf, for tests/test_model_qwen2.cpp's
hermetic stage-4 checks (loads, forward runs with no NaN, decode-vs-prefill
agreement -- same properties the F32 tiny fixture checks, on a Q8_0 model).

Dimensions here are NOT the same as tiny_qwen2.gguf's (n_embd=8 etc.):
Q8_0 quantizes each row in blocks of 32 floats, so every weight matrix's
contraction dimension (n_embd, and n_head*head_dim for the O-projection)
must be a multiple of 32. n_embd=32, head_dim=32, n_ff=64 satisfy this while
staying small enough for a fast, hermetic test. 1D norm/bias tensors are
kept F32, matching real llama-quantize --pure output (SPEC.md: "norm、bias
通常仍是 F32").

Uses gguf-py's own reference quantizer (gguf.quants.quantize), the same
library llama-quantize's GGUF writing is built on -- not this project's own
mi_quantize_row_q8_0 (that would make this fixture worthless as an
independent check of this project's quantizer).
"""
import numpy as np

import gguf
from gguf.quants import quantize
from gguf.constants import GGMLQuantizationType

N_LAYER = 2
N_EMBD = 32
N_HEAD = 1
N_HEAD_KV = 1
HEAD_DIM = N_EMBD // N_HEAD
N_FF = 64
N_VOCAB = 64


def main():
    rng = np.random.default_rng(4321)

    def randw(*shape):
        return rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)

    def norm_w(n):
        return (1.0 + rng.uniform(-0.05, 0.05, size=n)).astype(np.float32)

    def q8(arr: np.ndarray) -> np.ndarray:
        return quantize(arr, GGMLQuantizationType.Q8_0)

    out_path = "tests/fixtures/tiny_qwen2_q8_0.gguf"
    w = gguf.GGUFWriter(out_path, "qwen2")

    w.add_uint32("qwen2.block_count", N_LAYER)
    w.add_uint32("qwen2.embedding_length", N_EMBD)
    w.add_uint32("qwen2.attention.head_count", N_HEAD)
    w.add_uint32("qwen2.attention.head_count_kv", N_HEAD_KV)
    w.add_uint32("qwen2.feed_forward_length", N_FF)
    w.add_uint32("qwen2.context_length", 128)
    w.add_float32("qwen2.rope.freq_base", 10000.0)
    w.add_float32("qwen2.attention.layer_norm_rms_epsilon", 1e-6)

    w.add_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(N_VOCAB)])
    w.add_array("tokenizer.ggml.token_type", [1] * N_VOCAB)

    embd = randw(N_VOCAB, N_EMBD)
    w.add_tensor("token_embd.weight", q8(embd), raw_dtype=GGMLQuantizationType.Q8_0)
    w.add_tensor("output_norm.weight", norm_w(N_EMBD))
    # No output.weight: tied embedding, same as the real Qwen2.5 GGUFs.

    for l in range(N_LAYER):
        w.add_tensor(f"blk.{l}.attn_norm.weight", norm_w(N_EMBD))

        qw = randw(N_HEAD * HEAD_DIM, N_EMBD)
        w.add_tensor(f"blk.{l}.attn_q.weight", q8(qw), raw_dtype=GGMLQuantizationType.Q8_0)
        w.add_tensor(f"blk.{l}.attn_q.bias", randw(N_HEAD * HEAD_DIM))

        kw = randw(N_HEAD_KV * HEAD_DIM, N_EMBD)
        w.add_tensor(f"blk.{l}.attn_k.weight", q8(kw), raw_dtype=GGMLQuantizationType.Q8_0)
        w.add_tensor(f"blk.{l}.attn_k.bias", randw(N_HEAD_KV * HEAD_DIM))

        vw = randw(N_HEAD_KV * HEAD_DIM, N_EMBD)
        w.add_tensor(f"blk.{l}.attn_v.weight", q8(vw), raw_dtype=GGMLQuantizationType.Q8_0)
        w.add_tensor(f"blk.{l}.attn_v.bias", randw(N_HEAD_KV * HEAD_DIM))

        ow = randw(N_EMBD, N_HEAD * HEAD_DIM)
        w.add_tensor(f"blk.{l}.attn_output.weight", q8(ow), raw_dtype=GGMLQuantizationType.Q8_0)

        w.add_tensor(f"blk.{l}.ffn_norm.weight", norm_w(N_EMBD))

        gw = randw(N_FF, N_EMBD)
        w.add_tensor(f"blk.{l}.ffn_gate.weight", q8(gw), raw_dtype=GGMLQuantizationType.Q8_0)
        uw = randw(N_FF, N_EMBD)
        w.add_tensor(f"blk.{l}.ffn_up.weight", q8(uw), raw_dtype=GGMLQuantizationType.Q8_0)
        dw = randw(N_EMBD, N_FF)
        w.add_tensor(f"blk.{l}.ffn_down.weight", q8(dw), raw_dtype=GGMLQuantizationType.Q8_0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
