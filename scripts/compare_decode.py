#!/usr/bin/env python3
"""Checks tools/dump's --decode-ids mode (id -> text, byte-level BPE
reversal) against HF's tokenizer.decode() for a handful of sequences that
exercise Chinese text, emoji, and special/control tokens -- per PROMPTS.md
stage 0: "对若干 token id 序列(含中文、emoji、特殊 token)decode,与 HF
tokenizer.decode 一致".

Usage:
    python3 scripts/compare_decode.py <model.gguf> [--dump-bin PATH]
"""
import argparse
import subprocess
import sys

from transformers import AutoTokenizer

HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def unescape(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n == "n":
                out.append("\n")
            elif n == "t":
                out.append("\t")
            elif n == "r":
                out.append("\r")
            elif n == "\\":
                out.append("\\")
            else:
                out.append(c)
                out.append(n)
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)

SAMPLES = [
    "Hello, world!",
    "你好，世界！这是一个测试。",
    "emoji test: 😀🚀✨ and mixed 中文emoji混合😊",
    "Line with\ttab and 'quotes' and \"double\"",
]


def build_sequences(tok) -> list:
    seqs = [tok(s)["input_ids"] for s in SAMPLES]
    msgs = [{"role": "user", "content": "你好，请用一句话介绍一下你自己 😊"}]
    seqs.append(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
    # A sequence made entirely of special/control tokens.
    seqs.append([tok.bos_token_id or 151643, 151644, 151645, tok.eos_token_id])
    return seqs


def run_dump_decode(dump_bin: str, model_path: str, seqs: list) -> list:
    ids_text = "\n".join(" ".join(str(i) for i in seq) for seq in seqs)
    proc = subprocess.run(
        [dump_bin, model_path, "--decode-ids", "/dev/stdin"],
        input=ids_text,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = proc.stdout.split("\n")
    # One trailing '\n' after the last line produces one trailing empty
    # element from split; drop it if present.
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return [unescape(line) for line in lines]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a .gguf file (for the vocab)")
    ap.add_argument("--dump-bin", default="build/tools/mi_dump")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    seqs = build_sequences(tok)
    mine = run_dump_decode(args.dump_bin, args.model, seqs)

    if len(mine) != len(seqs):
        print(f"FAIL: expected {len(seqs)} decoded lines, got {len(mine)}")
        return 1

    failures = 0
    for i, seq in enumerate(seqs):
        ref = tok.decode(seq)
        got = mine[i]
        ok = got == ref
        status = "OK" if ok else "FAIL"
        label = SAMPLES[i] if i < len(SAMPLES) else f"seq#{i}"
        print(f"[{status}] {label!r}")
        if not ok:
            failures += 1
            print(f"    hf : {ref!r}")
            print(f"    c++: {got!r}")

    if failures:
        print(f"FAIL: {failures}/{len(seqs)} sequences mismatched")
        return 1
    print(f"PASS: all {len(seqs)} sequences match HF tokenizer.decode()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
