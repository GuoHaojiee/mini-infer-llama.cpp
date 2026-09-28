#!/usr/bin/env python3
"""Stage 4's Q8_0-vs-llama.cpp accuracy comparison, per SPEC.md's "对拍标准"
for this pair: teacher forcing over the same token sequence on the same
Q8_0 GGUF, reporting top-1 agreement, cosine similarity, and max absolute
error (not requiring bit-exactness, unlike the F32-vs-HF standard).

mini-infer's logits come from `mini_infer --dump` (this project's own
graph, Q8_0 mul_mat path). llama.cpp's come from tools/llama_logits (built
only if a sibling llama.cpp checkout exists -- see tools/CMakeLists.txt).

Usage:
    python3 scripts/compare_q8_llamacpp.py --model <q8_0.gguf> \
        --cpp-bin build/mini_infer --llama-logits-bin build/tools/llama_logits \
        [--ids-file <existing ids.txt> | --hf-model Qwen/Qwen2.5-0.5B-Instruct]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

MIN_TOKENS = 256

# Long enough (repeated/varied) filler text to tokenize to >= MIN_TOKENS
# with the real Qwen2.5 BPE tokenizer -- content doesn't matter for this
# accuracy check (it's not evaluating generation quality, just whether two
# engines agree on logits for the same input), only that it's real,
# non-degenerate token diversity, not a repeated single id.
DEFAULT_TEXT = (
    "mini-infer is a from-scratch C++17 CPU inference engine for Qwen2.5, "
    "built around a computation graph, a graph allocator that plans memory "
    "ahead of time, a persistent thread pool with dynamic work-stealing "
    "for mul_mat, and Q8_0 quantization with a hand-written AVX2 dot "
    "product. Each stage was implemented and tested before moving to the "
    "next: GGUF parsing, a naive F32 forward pass checked against "
    "Hugging Face transformers position by position, a graph-based "
    "rewrite that reuses the exact same kernels, a static memory planner "
    "that reduces peak memory by tens to nearly a hundred times compared "
    "to allocating every intermediate tensor independently, multi-threaded "
    "execution with a self-implemented barrier, and finally F16 and Q8_0 "
    "quantized inference with AVX2 acceleration. This paragraph exists "
    "purely to give the tokenizer enough varied text to produce a long, "
    "realistic sequence of token ids for a teacher-forcing accuracy check "
    "between this engine and llama.cpp on the same quantized model file."
)


def build_ids_file(out_path: str, hf_model: str, text: str) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_model)
    ids = tok(text)["input_ids"]
    while len(ids) < MIN_TOKENS:
        ids = ids + tok(text)["input_ids"]
    with open(out_path, "w") as f:
        f.write(" ".join(str(i) for i in ids) + "\n")
    return len(ids)


def load_dump(prefix: str) -> np.ndarray:
    with open(prefix + ".shape") as f:
        n_tokens, n_vocab = map(int, f.read().split())
    return np.fromfile(prefix + ".bin", dtype=np.float32).reshape(n_tokens, n_vocab)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Q8_0 GGUF path")
    ap.add_argument("--cpp-bin", default="build/mini_infer")
    ap.add_argument("--llama-logits-bin", default="build/tools/llama_logits")
    ap.add_argument("--ids-file", default=None, help="existing token-id file; if omitted, one is generated")
    ap.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="bench/results/accuracy_q8_vs_llamacpp.txt")
    args = ap.parse_args()

    if not os.path.exists(args.llama_logits_bin):
        print(f"error: {args.llama_logits_bin} not found -- build it first "
              f"(requires a sibling llama.cpp checkout, see tools/CMakeLists.txt)", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.ids_file:
            ids_path = args.ids_file
            with open(ids_path) as f:
                n_tokens = len(f.read().split())
            source = f"user-provided file: {ids_path}"
        else:
            ids_path = os.path.join(tmpdir, "ids.txt")
            n_tokens = build_ids_file(ids_path, args.hf_model, DEFAULT_TEXT)
            source = (f"tokenized (via {args.hf_model}'s tokenizer, transformers.AutoTokenizer) "
                      f"from a fixed English description of this project, repeated until >= "
                      f"{MIN_TOKENS} tokens")

        if n_tokens < MIN_TOKENS:
            print(f"error: token sequence only has {n_tokens} tokens, need >= {MIN_TOKENS}",
                  file=sys.stderr)
            return 1

        mi_dir = os.path.join(tmpdir, "mi_dump")
        os.makedirs(mi_dir, exist_ok=True)
        subprocess.run(
            [args.cpp_bin, args.model, "--prompt-ids", ids_path, "--dump", mi_dir,
             "-c", str(n_tokens)],
            check=True, capture_output=True, text=True,
        )
        mi_logits = load_dump(os.path.join(mi_dir, "logits"))

        llama_prefix = os.path.join(tmpdir, "llama_out")
        subprocess.run(
            [args.llama_logits_bin, args.model, ids_path, llama_prefix],
            check=True, capture_output=True, text=True,
        )
        llama_logits = load_dump(llama_prefix)

    if mi_logits.shape != llama_logits.shape:
        print(f"error: shape mismatch mi={mi_logits.shape} llama={llama_logits.shape}", file=sys.stderr)
        return 1

    top1_mi = mi_logits.argmax(axis=1)
    top1_llama = llama_logits.argmax(axis=1)
    top1_agreement = float((top1_mi == top1_llama).mean())

    dot = np.sum(mi_logits * llama_logits, axis=1)
    norm_mi = np.linalg.norm(mi_logits, axis=1)
    norm_llama = np.linalg.norm(llama_logits, axis=1)
    cos_per_row = dot / (norm_mi * norm_llama)
    cos_mean = float(cos_per_row.mean())
    cos_min = float(cos_per_row.min())

    abs_err = np.abs(mi_logits - llama_logits)
    max_abs_err = float(abs_err.max())
    mean_abs_err = float(abs_err.mean())

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("# Q8_0 vs llama.cpp accuracy (stage 4)")
    log(f"model: {args.model}")
    log(f"token sequence length: {n_tokens}")
    log(f"token sequence source: {source}")
    log(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    log("Comparison method: teacher forcing, same token sequence fed to both engines "
        "on the same Q8_0 GGUF; logits compared position by position (SPEC.md's "
        "对拍标准 for Q8_0 vs llama.cpp: top-1 agreement rate, cosine similarity, "
        "max absolute error -- bit-exactness is not required).")
    log("")
    log(f"top1_agreement_rate: {top1_agreement:.6f}  ({int(round(top1_agreement * n_tokens))}/{n_tokens} positions)")
    log(f"cosine_similarity_mean: {cos_mean:.6f}")
    log(f"cosine_similarity_min:  {cos_min:.6f}")
    log(f"max_abs_error: {max_abs_err:.6f}")
    log(f"mean_abs_error: {mean_abs_err:.6f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
