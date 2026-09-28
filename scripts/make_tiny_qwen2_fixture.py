#!/usr/bin/env python3
"""Generates tests/fixtures/tiny_qwen2.gguf: a structurally valid but tiny
(n_layer=2, n_embd=8, n_head=2, n_head_kv=1, n_ff=16, n_vocab=32) Qwen2
model with random (fixed-seed) F32 weights, matching every tensor name and
metadata key SPEC.md lists for a real Qwen2 GGUF.

This is NOT for accuracy testing (the weights are random) -- it's for
tests/test_model_qwen2.cpp's hermetic, CTest-speed structural checks: does
the forward pass produce the right shapes with no NaN, and, crucially, do
batch prefill and token-by-token incremental decode produce bit-identical
logits at every position (PROMPTS.md's stage 1 "额外验证" requirement).
That property doesn't depend on the weights being real, only on the
position/KV-cache bookkeeping being correct, so a tiny random model is a
faster and more targeted test than loading the real 0.5B checkpoint for it.

Built with the reference `gguf` library, same independent-oracle rationale
as scripts/make_tiny_gguf_fixture.py.
"""
import numpy as np

import gguf

N_LAYER = 2
N_EMBD = 8
N_HEAD = 2
N_HEAD_KV = 1
HEAD_DIM = N_EMBD // N_HEAD
N_FF = 16
N_VOCAB = 32


def main():
    rng = np.random.default_rng(1234)

    def randw(*shape):
        return rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)

    def norm_w(n):
        return (1.0 + rng.uniform(-0.05, 0.05, size=n)).astype(np.float32)

    out_path = "tests/fixtures/tiny_qwen2.gguf"
    w = gguf.GGUFWriter(out_path, "qwen2")

    w.add_uint32("qwen2.block_count", N_LAYER)
    w.add_uint32("qwen2.embedding_length", N_EMBD)
    w.add_uint32("qwen2.attention.head_count", N_HEAD)
    w.add_uint32("qwen2.attention.head_count_kv", N_HEAD_KV)
    w.add_uint32("qwen2.feed_forward_length", N_FF)
    w.add_uint32("qwen2.context_length", 128)
    w.add_float32("qwen2.rope.freq_base", 10000.0)
    w.add_float32("qwen2.attention.layer_norm_rms_epsilon", 1e-6)

    # Minimal vocab: mi_qwen2_model_load always calls mi_vocab_load, so this
    # must be present and exactly N_VOCAB entries long, even though this
    # fixture is never used for text decode.
    w.add_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(N_VOCAB)])
    w.add_array("tokenizer.ggml.token_type", [1] * N_VOCAB)

    w.add_tensor("token_embd.weight", randw(N_VOCAB, N_EMBD))
    w.add_tensor("output_norm.weight", norm_w(N_EMBD))
    # No output.weight: exercises the tied-embedding path, same as the real
    # Qwen2.5-0.5B-Instruct GGUF.

    for l in range(N_LAYER):
        w.add_tensor(f"blk.{l}.attn_norm.weight", norm_w(N_EMBD))
        w.add_tensor(f"blk.{l}.attn_q.weight", randw(N_HEAD * HEAD_DIM, N_EMBD))
        w.add_tensor(f"blk.{l}.attn_q.bias", randw(N_HEAD * HEAD_DIM))
        w.add_tensor(f"blk.{l}.attn_k.weight", randw(N_HEAD_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(f"blk.{l}.attn_k.bias", randw(N_HEAD_KV * HEAD_DIM))
        w.add_tensor(f"blk.{l}.attn_v.weight", randw(N_HEAD_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(f"blk.{l}.attn_v.bias", randw(N_HEAD_KV * HEAD_DIM))
        w.add_tensor(f"blk.{l}.attn_output.weight", randw(N_EMBD, N_HEAD * HEAD_DIM))
        w.add_tensor(f"blk.{l}.ffn_norm.weight", norm_w(N_EMBD))
        w.add_tensor(f"blk.{l}.ffn_gate.weight", randw(N_FF, N_EMBD))
        w.add_tensor(f"blk.{l}.ffn_up.weight", randw(N_FF, N_EMBD))
        w.add_tensor(f"blk.{l}.ffn_down.weight", randw(N_EMBD, N_FF))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
