#!/usr/bin/env python3
"""Tokenizes text into an ids file mini-infer's C++ side can consume
(--prompt-ids), and scripts/ref_hf.py's dump/generate subcommands also
read. mini-infer never encodes text itself (SPEC.md) -- this script and
HF's tokenizer are the only place a chat template or BPE merge table gets
applied.

ids file format: a single line of space-separated decimal token ids.

Usage:
    python3 scripts/encode.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --chat "介绍一下你自己" --out ids.txt
    python3 scripts/encode.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --text "Hello, world!" --out ids.txt
"""
import argparse

from transformers import AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--chat", help="user message; wrapped with apply_chat_template")
    group.add_argument("--text", help="raw text, tokenized as-is (no chat template)")
    ap.add_argument("--out", required=True, help="output ids file path")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    if args.chat is not None:
        # Instruct models must go through the chat template (SPEC.md) --
        # feeding raw text to an instruct model produces degenerate output.
        messages = [{"role": "user", "content": args.chat}]
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    else:
        ids = tok(args.text)["input_ids"]

    with open(args.out, "w") as f:
        f.write(" ".join(str(i) for i in ids) + "\n")
    print(f"wrote {len(ids)} ids to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
