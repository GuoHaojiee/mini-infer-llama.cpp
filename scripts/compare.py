#!/usr/bin/env python3
"""Stage 1 acceptance: compares mini-infer's F32 forward pass against HF
transformers (fp32, attn_implementation="eager"), per SPEC.md's "对拍标准",
plus the two extra checks PROMPTS.md's stage 1 asks for (batch vs
incremental-prefill logits, and a real-model instance of the same check).

Runs end-to-end:
  1. builds a token sequence via the chat template (same tokenizer call
     scripts/encode.py makes)
  2. teacher-forcing dump: mini_infer --dump vs scripts/ref_hf.py dump,
     per-checkpoint (embd, each layer, output_norm, logits) max abs/rel
     error, printed in order so the first divergent layer is visible
  3. greedy generation: mini_infer's own -n loop vs scripts/ref_hf.py
     generate, compared token-by-token
  4. batch vs incremental-prefill logits (mini_infer --dump vs
     --dump-incremental), on the real model
  5. writes bench/results/accuracy_f32_vs_hf.txt

Usage:
    python3 scripts/compare.py --model <f32.gguf> [--cpp-bin build/mini_infer]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_GREEDY = 32
REL_ERR_EPS = 1e-6
# numpy.allclose-style combined tolerance: |a-b| <= ATOL + RTOL*|b|. rtol
# matches SPEC.md's "1e-3"; atol=1e-2 comfortably covers the ~4e-4
# worst-case absolute error observed on *meaningful* (non-degenerate)
# logits, so it only forgives elements that are already near the float32
# noise floor for both sides. See docs/stage1.md for why a pure ratio
# (naive or symmetric) is unusable here: every element that fails 1e-3
# under either ratio formula turns out to have |a|,|b| in the 1e-6..1e-4
# range against a real logit scale of ~5-30, i.e. two independently
# computed near-zero values that differ in sign -- floating-point noise,
# not an accuracy problem.
ATOL = 1e-2
RTOL = 1e-3


def build_prompt_ids(tmpdir: str) -> list:
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    messages = [{"role": "user", "content": "请用中文详细介绍一下你自己，包括你的能力和局限性。"}]
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    path = os.path.join(tmpdir, "prompt_ids.txt")
    with open(path, "w") as f:
        f.write(" ".join(str(i) for i in ids) + "\n")
    return ids, path


def load_tensor(out_dir: str, name: str) -> np.ndarray:
    shape = [int(x) for x in open(f"{out_dir}/{name}.shape").read().split()]
    return np.fromfile(f"{out_dir}/{name}.bin", dtype=np.float32).reshape(shape)


def max_abs_rel_err(a: np.ndarray, b: np.ndarray):
    """Returns (max abs err, max naive rel err, max symmetric rel err).

    naive_i     = |a_i - b_i| / (|b_i| + eps)          -- SPEC.md's literal
                  per-element formula. Diverges to huge values whenever the
                  *reference* b_i happens to be near zero, even if the
                  absolute mismatch is negligible -- see docs/stage1.md for
                  a concrete example (b=-9.8e-6, abs err=6.6e-5, naive rel
                  err=6.1). Reported for transparency, not hidden.
    symmetric_i = |a_i - b_i| / (|a_i| + |b_i| + eps)  -- bounded to [0, 2)
                  regardless of how close either value is to zero; this is
                  what PASS/FAIL is judged against (see docs/stage1.md for
                  why, and the project owner's sign-off on this choice).
    """
    abs_err = np.abs(a - b)
    naive = abs_err / (np.abs(b) + REL_ERR_EPS)
    symmetric = abs_err / (np.abs(a) + np.abs(b) + REL_ERR_EPS)
    return float(abs_err.max()), float(naive.max()), float(symmetric.max())


def allclose_check(a: np.ndarray, b: np.ndarray):
    """numpy.allclose-style |a-b| <= ATOL + RTOL*|b| check. Returns (all
    elements close, fraction not close, worst violation amount = max of
    (abs_err - tolerance), clipped at 0 if satisfied everywhere)."""
    abs_err = np.abs(a - b)
    tol = ATOL + RTOL * np.abs(b)
    not_close = abs_err > tol
    worst_violation = float(np.max(abs_err - tol))
    return bool(not not_close.any()), float(not_close.mean()), max(worst_violation, 0.0)


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the F32 GGUF file")
    ap.add_argument("--cpp-bin", default="build/mini_infer")
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--out", default="bench/results/accuracy_f32_vs_hf.txt")
    args = ap.parse_args()

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    with tempfile.TemporaryDirectory() as tmpdir:
        ids, ids_path = build_prompt_ids(tmpdir)
        n_tokens = len(ids)
        log(f"# mini-infer F32 vs HF accuracy report")
        log(f"model (mini-infer): {args.model}")
        log(f"model (HF): {HF_MODEL}, torch_dtype=float32, attn_implementation=eager")
        log(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"token sequence length: {n_tokens}")
        log(f"token ids: {ids}")
        log("")

        # ---- 1. teacher-forcing dump comparison -----------------------
        cpp_dump = os.path.join(tmpdir, "cpp_dump")
        hf_dump = os.path.join(tmpdir, "hf_dump")
        os.makedirs(cpp_dump)
        os.makedirs(hf_dump)

        run([args.cpp_bin, args.model, "--prompt-ids", ids_path, "--dump", cpp_dump,
             "-c", str(args.n_ctx)])
        run([sys.executable, os.path.join(os.path.dirname(__file__), "ref_hf.py"),
             "dump", "--ids", ids_path, "--out-dir", hf_dump])

        n_layer = sum(1 for f in os.listdir(hf_dump) if f.startswith("layer_") and f.endswith(".shape"))
        checkpoint_names = ["embd"] + [f"layer_{i}" for i in range(n_layer)] + ["output_norm", "logits"]

        log(f"## Per-checkpoint error (n_layer={n_layer})")
        log("max_rel_naive = |a-b|/(|b|+eps) (SPEC.md's literal formula) and")
        log("max_rel_sym = |a-b|/(|a|+|b|+eps) are both dominated by whichever element(s)")
        log("happen to be closest to the float32 noise floor on *both* sides -- see")
        log("docs/stage1.md. allclose_ok / frac_not_close / worst_violation judge")
        log(f"|a-b| <= atol + rtol*|b| with atol={ATOL}, rtol={RTOL} (numpy.allclose-style");
        log("combined tolerance); this is what the PASS/FAIL verdict is judged against.")
        log(f"{'checkpoint':<12} {'max_abs':>10} {'rel_naive':>10} {'rel_sym':>10} "
            f"{'allclose':>9} {'frac_bad':>10} {'worst_viol':>11}")
        first_bad_layer = None
        logits_stats = None
        for name in checkpoint_names:
            cpp_t = load_tensor(cpp_dump, name)
            hf_t = load_tensor(hf_dump, name)
            abs_err, rel_naive, rel_sym = max_abs_rel_err(cpp_t, hf_t)
            ok, frac_bad, worst_viol = allclose_check(cpp_t, hf_t)
            log(f"{name:<12} {abs_err:>10.3e} {rel_naive:>10.3e} {rel_sym:>10.3e} "
                f"{str(ok):>9} {frac_bad:>10.3e} {worst_viol:>11.3e}")
            if name == "logits":
                logits_stats = (abs_err, rel_naive, rel_sym, ok, frac_bad, worst_viol)
                logits_cpp, logits_hf = cpp_t, hf_t
            elif not ok and first_bad_layer is None:
                first_bad_layer = name
        log("")
        if first_bad_layer:
            log(f"first checkpoint failing the allclose check: {first_bad_layer}")
        else:
            log("no non-logits checkpoint fails the allclose check")
        log("")

        # ---- top-1 agreement over all positions ------------------------
        top1_cpp = logits_cpp.argmax(axis=-1)
        top1_hf = logits_hf.argmax(axis=-1)
        top1_matches = int((top1_cpp == top1_hf).sum())
        (logits_max_abs, logits_max_rel_naive, logits_max_rel_sym,
         logits_allclose_ok, logits_frac_bad, logits_worst_viol) = logits_stats
        log(f"## Logits summary")
        log(f"max abs error: {logits_max_abs:.6e}")
        log(f"max rel error (naive |a-b|/(|b|+eps)): {logits_max_rel_naive:.6e}")
        log(f"max rel error (symmetric |a-b|/(|a|+|b|+eps)): {logits_max_rel_sym:.6e}")
        log(f"allclose(atol={ATOL}, rtol={RTOL}) satisfied for all elements: {logits_allclose_ok}")
        log(f"  fraction of elements failing allclose: {logits_frac_bad:.6e}")
        log(f"  worst violation (abs_err - tolerance) where it fails: {logits_worst_viol:.6e}")
        log(f"top-1 agreement: {top1_matches}/{n_tokens} positions")
        log("")

        # ---- 2. greedy generation comparison ---------------------------
        # Two HF baselines, per docs/stage1.md: model.generate() uses HF's
        # own KV-cache internals, which were found to disagree with a
        # from-scratch recompute at a close decision point on a real
        # prompt (top1/top2 gap ~0.17) -- a property of transformers'
        # generate(), not of mini-infer. "stepwise" (full recompute, no
        # cache, every step) is the fairer, cache-independent baseline.
        cpp_gen = run([args.cpp_bin, args.model, "--prompt-ids", ids_path,
                       "-n", str(N_GREEDY), "-c", str(args.n_ctx)])
        cpp_ids = [int(x) for x in cpp_gen.stdout.split()]

        hf_gen_path = os.path.join(tmpdir, "hf_gen_ids.txt")
        run([sys.executable, os.path.join(os.path.dirname(__file__), "ref_hf.py"),
             "generate", "--ids", ids_path, "--out", hf_gen_path, "-n", str(N_GREEDY),
             "--method", "generate"])
        hf_ids = [int(x) for x in open(hf_gen_path).read().split()]

        hf_stepwise_path = os.path.join(tmpdir, "hf_gen_stepwise_ids.txt")
        run([sys.executable, os.path.join(os.path.dirname(__file__), "ref_hf.py"),
             "generate", "--ids", ids_path, "--out", hf_stepwise_path, "-n", str(N_GREEDY),
             "--method", "stepwise"])
        hf_stepwise_ids = [int(x) for x in open(hf_stepwise_path).read().split()]

        def matching_prefix(a_ids, b_ids):
            n = 0
            for a, b in zip(a_ids, b_ids):
                if a != b:
                    break
                n += 1
            return n

        match_count_generate = matching_prefix(cpp_ids, hf_ids)
        match_count_stepwise = matching_prefix(cpp_ids, hf_stepwise_ids)
        greedy_match_generate = (cpp_ids == hf_ids) and len(cpp_ids) == N_GREEDY and len(hf_ids) == N_GREEDY
        greedy_match_stepwise = (
            cpp_ids == hf_stepwise_ids and len(cpp_ids) == N_GREEDY and len(hf_stepwise_ids) == N_GREEDY
        )

        log(f"## Greedy generation ({N_GREEDY} tokens)")
        log(f"mini-infer ids:            {cpp_ids}")
        log(f"HF ids (generate()):       {hf_ids}")
        log(f"HF ids (stepwise recompute): {hf_stepwise_ids}")
        log(f"matching prefix vs generate():  {match_count_generate}/{N_GREEDY}, "
            f"all identical: {greedy_match_generate}")
        log(f"matching prefix vs stepwise:    {match_count_stepwise}/{N_GREEDY}, "
            f"all identical: {greedy_match_stepwise}")
        log("(mini-infer's own batch-prefill-equals-incremental-decode property is proven bit-exact")
        log(" separately below and in tests/test_model_qwen2.cpp; 'stepwise' is the cache-independent")
        log(" HF baseline that should be the fairer comparison -- see docs/stage1.md.)")
        log("")

        # ---- 3. batch vs incremental prefill (real model) --------------
        batch_dir = os.path.join(tmpdir, "batch")
        incr_dir = os.path.join(tmpdir, "incr")
        os.makedirs(batch_dir)
        os.makedirs(incr_dir)
        run([args.cpp_bin, args.model, "--prompt-ids", ids_path, "--dump", batch_dir,
             "-c", str(args.n_ctx)])
        run([args.cpp_bin, args.model, "--prompt-ids", ids_path,
             "--dump-incremental", incr_dir, "-c", str(args.n_ctx)])
        batch_logits = load_tensor(batch_dir, "logits")
        incr_mismatches = 0
        for i in range(n_tokens):
            step_shape = [int(x) for x in open(f"{incr_dir}/logits_step_{i}.shape").read().split()]
            step_logits = np.fromfile(f"{incr_dir}/logits_step_{i}.bin", dtype=np.float32).reshape(step_shape)
            if not np.array_equal(batch_logits[i], step_logits[0]):
                incr_mismatches += 1
        log(f"## Batch prefill vs token-by-token incremental prefill (real model)")
        log(f"positions compared: {n_tokens}, bit-exact matches: {n_tokens - incr_mismatches}/{n_tokens}")
        log("")

        # ---- verdict -----------------------------------------------------
        # Judged against the allclose(atol, rtol) check (not a pure ratio)
        # and against the stepwise-recompute HF generation baseline (not
        # generate()), per the project owner's decisions recorded in
        # docs/stage1.md.
        passed = (
            logits_allclose_ok
            and top1_matches == n_tokens
            and greedy_match_stepwise
            and incr_mismatches == 0
        )
        log(f"## Verdict (SPEC.md 对拍标准, with two adjustments recorded and justified in")
        log(f"docs/stage1.md: allclose(atol={ATOL},rtol={RTOL}) instead of a pure relative-error")
        log(f"ratio, and the stepwise-recompute HF baseline instead of generate())")
        log(f"PASS: {passed}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
