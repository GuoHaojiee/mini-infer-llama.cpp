#!/usr/bin/env python3
"""Reference forward pass / generation via HF transformers, fp32,
attn_implementation="eager" (SPEC.md's "对拍标准"). Two subcommands:

  dump      Teacher-forcing single forward pass over a given ids file;
            exports embd / layer_i / output_norm / logits as .bin+.shape,
            in the same format tools/mi_dump's caller (main.cpp --dump)
            writes, for scripts/compare.py to diff.

  generate  Greedy multi-step generation (do_sample=False) for exactly N
            new tokens, written as an ids file -- for comparing against
            mini_infer's own greedy generation loop.

Gotcha (see docs/stage1.md): with this transformers version,
output_hidden_states's last tuple entry is *overwritten* with the
post-final-norm last_hidden_state (check_model_inputs's
tie_last_hidden_states=True), so the raw pre-norm output of the *last*
decoder layer is never exposed through hidden_states. We register our own
forward hook on model.model.layers[-1] to recover it, so all n_layer layer
checkpoints -- not just the first n_layer-1 -- get compared.
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    return model


def read_ids(path: str) -> list:
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def write_tensor(out_dir: str, name: str, t: torch.Tensor) -> None:
    arr = np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32))
    arr.tofile(f"{out_dir}/{name}.bin")
    with open(f"{out_dir}/{name}.shape", "w") as f:
        f.write(" ".join(str(d) for d in arr.shape) + "\n")


def cmd_dump(args) -> int:
    model = load_model()
    n_layer = model.config.num_hidden_layers

    captured_last_raw = {}

    def hook(_module, _inputs, output):
        # Qwen2DecoderLayer returns a tuple; hidden_states is element 0.
        captured_last_raw["value"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.model.layers[-1].register_forward_hook(hook)
    try:
        ids = read_ids(args.ids)
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()

    hidden_states = out.hidden_states  # tuple, length n_layer + 1
    assert len(hidden_states) == n_layer + 1, (
        f"expected {n_layer + 1} hidden_states entries, got {len(hidden_states)}"
    )

    write_tensor(args.out_dir, "embd", hidden_states[0][0])
    for l in range(n_layer - 1):
        write_tensor(args.out_dir, f"layer_{l}", hidden_states[l + 1][0])
    # Last layer's raw (pre-final-norm) output: hidden_states[-1] was
    # overwritten with the post-norm state, so use our own hook instead.
    write_tensor(args.out_dir, f"layer_{n_layer - 1}", captured_last_raw["value"][0])
    write_tensor(args.out_dir, "output_norm", hidden_states[-1][0])
    write_tensor(args.out_dir, "logits", out.logits[0])

    print(f"wrote {n_layer + 3} checkpoint tensors to {args.out_dir} "
          f"(n_tokens={len(ids)}, n_vocab={out.logits.shape[-1]})")
    return 0


def cmd_generate(args) -> int:
    model = load_model()
    ids = read_ids(args.ids)
    input_ids = torch.tensor([ids], dtype=torch.long)

    if args.method == "generate":
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.n,
                min_new_tokens=args.n,  # forces exactly n tokens, no early EOS stop,
                do_sample=False,        # matching mini_infer's unconditional greedy loop
                num_beams=1,
            )
        new_ids = out[0, len(ids):].tolist()
    else:
        # "stepwise": re-run a full, cache-free forward pass over the whole
        # sequence-so-far at every step (no KV cache at all, not even
        # internally). See docs/stage1.md: this exists because
        # model.generate()'s own KV-cache path was found to disagree with a
        # from-scratch forward pass by enough to flip an argmax at a close
        # decision (top1/top2 gap ~0.17) on a real prompt, which is a
        # property of transformers' generate(), not of mini-infer. Stepwise
        # is the fairer, cache-independent baseline for comparison.
        cur = list(ids)
        new_ids = []
        with torch.no_grad():
            for _ in range(args.n):
                step_input = torch.tensor([cur], dtype=torch.long)
                logits = model(input_ids=step_input, use_cache=False, return_dict=True).logits
                next_id = int(logits[0, -1].argmax())
                new_ids.append(next_id)
                cur.append(next_id)

    assert len(new_ids) == args.n, f"expected {args.n} new tokens, got {len(new_ids)}"
    with open(args.out, "w") as f:
        f.write(" ".join(str(i) for i in new_ids) + "\n")
    print(f"wrote {len(new_ids)} generated ids to {args.out} (method={args.method})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dump = sub.add_parser("dump")
    p_dump.add_argument("--ids", required=True)
    p_dump.add_argument("--out-dir", required=True)
    p_dump.set_defaults(func=cmd_dump)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--ids", required=True)
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("-n", type=int, default=32)
    p_gen.add_argument("--method", choices=["generate", "stepwise"], default="generate")
    p_gen.set_defaults(func=cmd_generate)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
