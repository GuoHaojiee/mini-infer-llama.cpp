#!/usr/bin/env python3
"""Stage 2b's gallocr peak-vs-baseline measurement, per PROMPTS.md:
prefill at n_tokens = 128/512/1024 (capped at n_ctx) and decode at
n_past = 0 and n_ctx-1. Every number comes from mini_infer --gallocr-stats
actually running mi_gallocr_get_stats() on the real model -- nothing here
is estimated (SPEC.md: no invented performance numbers).

Usage:
    python3 scripts/bench_gallocr.py --model <f32.gguf> [--cpp-bin build/mini_infer]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

N_CTX = 1024  # smallest value that still fits every required prefill tier;
# decode's n_past=n_ctx-1 case needs an (n_ctx-1)-token warmup prefill, and
# this stage has no SIMD/threading yet (that's stages 3/4), so keeping
# n_ctx no larger than necessary keeps this benchmark's own runtime sane
# without changing what's actually being measured (the worst-case n_kv_total
# for gallocr's reserve() is still n_ctx, whatever value that is).
PREFILL_TIERS = [128, 512, 1024]


def make_ids_file(path: str, n: int, vocab_cap: int = 100) -> None:
    with open(path, "w") as f:
        f.write(" ".join(str(i % vocab_cap) for i in range(n)) + "\n")


def run_stats(cpp_bin: str, model: str, ids_path: str, dump_n_past: int, n_ctx: int,
              no_bestfit: bool = False) -> dict:
    env = os.environ.copy()
    if no_bestfit:
        env["GALLOCR_NO_BESTFIT"] = "1"
    else:
        env.pop("GALLOCR_NO_BESTFIT", None)
    proc = subprocess.run(
        [cpp_bin, model, "--prompt-ids", ids_path, "--gallocr-stats",
         "--dump-n-past", str(dump_n_past), "-c", str(n_ctx)],
        check=True, capture_output=True, text=True, env=env,
    )
    line = proc.stdout.strip()
    fields = dict(kv.split("=") for kv in line.split())
    return {k: int(v) for k, v in fields.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cpp-bin", default="build/mini_infer")
    ap.add_argument("--out", default="bench/results/gallocr.txt")
    args = ap.parse_args()

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("# gallocr peak-vs-baseline (stage 2b; stage 6 added no_bestfit_MB and theoretical_min_MB)")
    log(f"model: {args.model}")
    log(f"n_ctx: {N_CTX}")
    log(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    log("All numbers are from mi_gallocr_get_stats() / mi_gallocr_theoretical_min_bytes() after an")
    log("actual mi_gallocr_alloc_graph() call on the real model (never estimated).")
    log("  peak_MB: gallocr arena bytes actually used for that call (best-fit + in-place, the")
    log("           configuration mini_infer actually runs with).")
    log("  baseline_MB: sum of each intermediate tensor's own size as if independently allocated")
    log("               (no reuse at all) -- i.e. what stage 2a would have used.")
    log("  no_bestfit_MB: GALLOCR_NO_BESTFIT=1 -- refcounted free-after-last-use still happens,")
    log("                 but a freed block is never reused for a *different* tensor (only")
    log("                 in-place elimination, a separate mechanism, still applies).")
    log("  theoretical_min_MB: pure liveness bound (see gallocr.h's mi_gallocr_theoretical_min_bytes")
    log("                      doc comment) -- max over topological positions of the sum of live")
    log("                      non-view tensor sizes at that position. Does NOT credit in-place")
    log("                      elimination, so it can come out *above* peak_MB when many ops are")
    log("                      in-place-eligible (peak_MB benefits from a technique this bound")
    log("                      deliberately excludes) -- not a bug, see the two numbers side by side.")
    log("")
    log(f"{'case':<22} {'n_tokens':>8} {'n_past':>8} {'peak_MB':>10} {'no_bestfit_MB':>14} "
        f"{'theoretical_min_MB':>19} {'baseline_MB':>12} {'n_inplace':>10} {'n_bestfit':>10}")

    with tempfile.TemporaryDirectory() as tmpdir:
        rows = []

        for n_tokens in PREFILL_TIERS:
            if n_tokens > N_CTX:
                continue
            ids_path = os.path.join(tmpdir, f"prefill_{n_tokens}.txt")
            make_ids_file(ids_path, n_tokens)
            stats = run_stats(args.cpp_bin, args.model, ids_path, dump_n_past=0, n_ctx=N_CTX)
            nb_stats = run_stats(args.cpp_bin, args.model, ids_path, dump_n_past=0, n_ctx=N_CTX,
                                  no_bestfit=True)
            rows.append((f"prefill n={n_tokens}", stats, nb_stats))

        # decode at n_past = 0: a single fresh token, no warmup.
        ids_path = os.path.join(tmpdir, "decode_0.txt")
        make_ids_file(ids_path, 1)
        stats = run_stats(args.cpp_bin, args.model, ids_path, dump_n_past=0, n_ctx=N_CTX)
        nb_stats = run_stats(args.cpp_bin, args.model, ids_path, dump_n_past=0, n_ctx=N_CTX,
                              no_bestfit=True)
        rows.append(("decode n_past=0", stats, nb_stats))

        # decode at n_past = n_ctx - 1: warm up n_ctx-1 tokens, measure the
        # last (n_ctx-th) one.
        ids_path = os.path.join(tmpdir, "decode_last.txt")
        make_ids_file(ids_path, N_CTX)
        stats = run_stats(args.cpp_bin, args.model, ids_path, dump_n_past=N_CTX - 1, n_ctx=N_CTX)
        nb_stats = run_stats(args.cpp_bin, args.model, ids_path, dump_n_past=N_CTX - 1, n_ctx=N_CTX,
                              no_bestfit=True)
        rows.append((f"decode n_past={N_CTX - 1}", stats, nb_stats))

        for name, stats, nb_stats in rows:
            peak_mb = stats["peak_bytes"] / 1024 / 1024
            nb_mb = nb_stats["peak_bytes"] / 1024 / 1024
            theo_mb = stats["theoretical_min_bytes"] / 1024 / 1024
            base_mb = stats["baseline_bytes"] / 1024 / 1024
            log(f"{name:<22} {stats['n_tokens']:>8} {stats['n_past']:>8} {peak_mb:>10.3f} "
                f"{nb_mb:>14.3f} {theo_mb:>19.3f} {base_mb:>12.3f} {stats['n_inplace_reuse']:>10} "
                f"{stats['n_bestfit_reuse']:>10}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
