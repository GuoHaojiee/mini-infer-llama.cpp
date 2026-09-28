#!/usr/bin/env python3
"""Stage 4's F32 / F16(scalar) / F16(AVX2+F16C) / Q8_0(scalar) / Q8_0(AVX2)
throughput comparison, per PROMPTS.md: same prompt/decode length, thread
counts 1 and stage 3's fastest decode thread count (6 on this machine --
see docs/stage3.md's real measurement, not re-derived here), method
otherwise identical to scripts/bench_threads.py (mini_infer --bench times
only the prefill call and the decode loop's forward calls, in-process).

"scalar" here means MI_QUANT_FORCE_SCALAR=1 (see quant.cpp): forces the
F16/Q8_0 dot-product dispatcher to always take the portable path even on
this AVX2-capable machine, so the SIMD-vs-scalar comparison is on the same
hardware, not "measured on a different machine". F32 has no such split --
mul_mat's F32 path never goes through quant.cpp's dispatcher at all -- so
it appears once per thread count, not twice.

Usage:
    python3 scripts/bench_quant.py --f32-model <f32.gguf> --f16-model <f16.gguf> \
        --q8-model <q8_0.gguf> [--cpp-bin build/mini_infer]
"""
import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time

N_CTX = 64
PROMPT_LEN = 16
DECODE_LEN = 8  # yields DECODE_LEN - 1 measured decode-forward calls
THREAD_COUNTS = [1, 6]  # 6 = stage 3's fastest decode thread count (docs/stage3.md)
N_WARMUP = 1
N_REPEATS = 5


def make_ids_file(path: str, n: int, vocab_cap: int = 100) -> None:
    with open(path, "w") as f:
        f.write(" ".join(str(i % vocab_cap) for i in range(n)) + "\n")


def run_bench(cpp_bin: str, model: str, ids_path: str, n_threads: int, force_scalar: bool) -> dict:
    env = os.environ.copy()
    if force_scalar:
        env["MI_QUANT_FORCE_SCALAR"] = "1"
    else:
        env.pop("MI_QUANT_FORCE_SCALAR", None)
    proc = subprocess.run(
        [cpp_bin, model, "--prompt-ids", ids_path, "--bench", "-c", str(N_CTX),
         "-n", str(DECODE_LEN), "-t", str(n_threads)],
        check=True, capture_output=True, text=True, env=env,
    )
    line = proc.stdout.strip()
    fields = dict(kv.split("=") for kv in line.split())
    return {k: float(v) for k, v in fields.items()}


def measure_point(cpp_bin: str, model: str, ids_path: str, n_threads: int, force_scalar: bool):
    for _ in range(N_WARMUP):
        run_bench(cpp_bin, model, ids_path, n_threads, force_scalar)
    prefill_vals, decode_vals = [], []
    for _ in range(N_REPEATS):
        r = run_bench(cpp_bin, model, ids_path, n_threads, force_scalar)
        prefill_vals.append(r["prefill_tok_per_s"])
        decode_vals.append(r["decode_tok_per_s"])
    return prefill_vals, decode_vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f32-model", required=True)
    ap.add_argument("--f16-model", required=True)
    ap.add_argument("--q8-model", required=True)
    ap.add_argument("--cpp-bin", default="build/mini_infer")
    ap.add_argument("--out-prefix", default="bench/results/quant")
    args = ap.parse_args()

    configs = [
        ("F32", args.f32_model, False),
        ("F16_scalar", args.f16_model, True),
        ("F16_avx2", args.f16_model, False),
        ("Q8_0_scalar", args.q8_model, True),
        ("Q8_0_avx2", args.q8_model, False),
    ]

    prefill_rows = []
    decode_rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        ids_path = os.path.join(tmpdir, "prompt.txt")
        make_ids_file(ids_path, PROMPT_LEN)

        points = [(name, model, fs, nt) for (name, model, fs) in configs for nt in THREAD_COUNTS]
        for i, (name, model, force_scalar, nt) in enumerate(points):
            print(f"[{i + 1}/{len(points)}] {name} n_threads={nt} ...", file=sys.stderr)
            t0 = time.time()
            prefill_vals, decode_vals = measure_point(args.cpp_bin, model, ids_path, nt, force_scalar)
            print(f"    done in {time.time() - t0:.1f}s", file=sys.stderr)
            prefill_rows.append((name, nt, prefill_vals))
            decode_rows.append((name, nt, decode_vals))

    def write_csv(path: str, rows, phase: str):
        with open(path, "w") as f:
            f.write(f"# {phase} tok/s -- mini-infer stage 4 quant-path comparison\n")
            f.write(f"# f32_model: {args.f32_model}\n")
            f.write(f"# f16_model: {args.f16_model}\n")
            f.write(f"# q8_0_model: {args.q8_model}\n")
            f.write(f"# n_ctx: {N_CTX} prompt_len: {PROMPT_LEN} decode_len: {DECODE_LEN}\n")
            f.write(f"# warmup: {N_WARMUP} repeats: {N_REPEATS}\n")
            f.write(f"# generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("config,n_threads,rep,tok_per_s\n")
            for name, nt, vals in rows:
                for rep, v in enumerate(vals):
                    f.write(f"{name},{nt},{rep},{v:.4f}\n")
        print(f"wrote {path}")

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    write_csv(f"{args.out_prefix}_prefill.csv", prefill_rows, "prefill")
    write_csv(f"{args.out_prefix}_decode.csv", decode_rows, "decode")

    def print_summary(rows, phase: str):
        print(f"\n# {phase} summary (median [min, max] tok/s)")
        print(f"{'config':<14} {'n_threads':>9} {'median':>10} {'min':>10} {'max':>10}")
        for name, nt, vals in rows:
            print(f"{name:<14} {nt:>9} {statistics.median(vals):>10.3f} {min(vals):>10.3f} "
                  f"{max(vals):>10.3f}")

    print_summary(prefill_rows, "prefill")
    print_summary(decode_rows, "decode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
