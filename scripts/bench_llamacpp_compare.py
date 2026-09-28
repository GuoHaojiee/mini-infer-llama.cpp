#!/usr/bin/env python3
"""Stage 5: mini-infer vs llama.cpp's own llama-bench, same Q8_0 GGUF, same
thread counts, same prompt/gen length (PROMPTS.md: "与 llama.cpp 的
llama-bench 对比(同模型、同线程数、同 prompt 长度与生成长度)").

Methodology differs slightly between the two sides and is reported as-is,
not normalized away: mini_infer's numbers are this project's own
--bench (median/min/max of 1 warmup + 5 repeats, see scripts/bench_quant.py).
llama-bench's numbers are its own native output (avg/stddev of 5 repeats,
-r 5, its default) -- we don't recompute llama-bench's internal statistics
from raw samples (it doesn't expose them), so "avg" from llama-bench sits
next to "median" from mini_infer in the same table; this is disclosed in
the output, not hidden.

-ngl 0 forces llama-bench onto CPU only, for a fair comparison with this
CPU-only project (SPEC.md's own scope).

Usage:
    python3 scripts/bench_llamacpp_compare.py --q8-model <q8_0.gguf> \
        --cpp-bin build/mini_infer --llama-bench-bin <path to llama-bench>
"""
import argparse
import csv
import io
import os
import statistics
import subprocess
import sys
import tempfile
import time

PROMPT_LEN = 16
DECODE_LEN = 8
N_CTX = 64
THREAD_COUNTS = [1, 3, 4, 6]  # stage 6: 3-4 is where this machine's pure-read
                               # bandwidth peaks (tools/bench_memread.cpp),
                               # so it's a more informative comparison point
                               # than 6 alone now that decode is closer to
                               # memory-bound (see docs/stage6.md)
N_WARMUP = 1
N_REPEATS = 5


def make_ids_file(path: str, n: int, vocab_cap: int = 100) -> None:
    with open(path, "w") as f:
        f.write(" ".join(str(i % vocab_cap) for i in range(n)) + "\n")


def run_mini_infer(cpp_bin: str, model: str, ids_path: str, n_threads: int) -> dict:
    proc = subprocess.run(
        [cpp_bin, model, "--prompt-ids", ids_path, "--bench", "-c", str(N_CTX),
         "-n", str(DECODE_LEN), "-t", str(n_threads)],
        check=True, capture_output=True, text=True,
    )
    fields = dict(kv.split("=") for kv in proc.stdout.strip().split())
    return {k: float(v) for k, v in fields.items()}


def measure_mini_infer(cpp_bin: str, model: str, ids_path: str, n_threads: int):
    for _ in range(N_WARMUP):
        run_mini_infer(cpp_bin, model, ids_path, n_threads)
    prefill_vals, decode_vals = [], []
    for _ in range(N_REPEATS):
        r = run_mini_infer(cpp_bin, model, ids_path, n_threads)
        prefill_vals.append(r["prefill_tok_per_s"])
        decode_vals.append(r["decode_tok_per_s"])
    return prefill_vals, decode_vals


def run_llama_bench(llama_bench_bin: str, model: str, n_threads: int) -> dict:
    proc = subprocess.run(
        [llama_bench_bin, "-m", model, "-p", str(PROMPT_LEN), "-n", str(DECODE_LEN),
         "-t", str(n_threads), "-ngl", "0", "-r", str(N_REPEATS), "-o", "csv"],
        check=True, capture_output=True, text=True,
    )
    reader = csv.DictReader(io.StringIO(proc.stdout))
    rows = list(reader)
    # llama-bench emits one row per (n_prompt, n_gen) pair actually exercised;
    # with -p 16 -n 8 that's a "prefill-only" row (n_prompt=16, n_gen=0) and
    # a "decode-only" row (n_prompt=0, n_gen=8).
    prefill_row = next(r for r in rows if int(r["n_prompt"]) > 0 and int(r["n_gen"]) == 0)
    decode_row = next(r for r in rows if int(r["n_prompt"]) == 0 and int(r["n_gen"]) > 0)
    return {
        "prefill_avg_ts": float(prefill_row["avg_ts"]),
        "prefill_stddev_ts": float(prefill_row["stddev_ts"]),
        "decode_avg_ts": float(decode_row["avg_ts"]),
        "decode_stddev_ts": float(decode_row["stddev_ts"]),
        "build_commit": prefill_row["build_commit"],
        "cpu_info": prefill_row["cpu_info"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q8-model", required=True)
    ap.add_argument("--cpp-bin", default="build/mini_infer")
    ap.add_argument("--llama-bench-bin", required=True)
    ap.add_argument("--out", default="bench/results/llama_bench_compare.csv")
    args = ap.parse_args()

    if not os.path.exists(args.llama_bench_bin):
        print(f"error: {args.llama_bench_bin} not found", file=sys.stderr)
        return 1

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        ids_path = os.path.join(tmpdir, "prompt.txt")
        make_ids_file(ids_path, PROMPT_LEN)

        for nt in THREAD_COUNTS:
            print(f"mini_infer Q8_0 AVX2, n_threads={nt} ...", file=sys.stderr)
            t0 = time.time()
            prefill_vals, decode_vals = measure_mini_infer(args.cpp_bin, args.q8_model, ids_path, nt)
            print(f"  done in {time.time() - t0:.1f}s", file=sys.stderr)
            rows.append(("mini-infer", "prefill", nt, "median", statistics.median(prefill_vals),
                         min(prefill_vals), max(prefill_vals)))
            rows.append(("mini-infer", "decode", nt, "median", statistics.median(decode_vals),
                         min(decode_vals), max(decode_vals)))

            print(f"llama-bench, n_threads={nt} ...", file=sys.stderr)
            t0 = time.time()
            lb = run_llama_bench(args.llama_bench_bin, args.q8_model, nt)
            print(f"  done in {time.time() - t0:.1f}s", file=sys.stderr)
            rows.append(("llama.cpp", "prefill", nt, "avg", lb["prefill_avg_ts"],
                         lb["prefill_avg_ts"] - lb["prefill_stddev_ts"],
                         lb["prefill_avg_ts"] + lb["prefill_stddev_ts"]))
            rows.append(("llama.cpp", "decode", nt, "avg", lb["decode_avg_ts"],
                         lb["decode_avg_ts"] - lb["decode_stddev_ts"],
                         lb["decode_avg_ts"] + lb["decode_stddev_ts"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# mini-infer vs llama.cpp (llama-bench), same Q8_0 GGUF, same prompt/gen length\n")
        f.write(f"# model: {args.q8_model}\n")
        f.write(f"# n_ctx: {N_CTX} prompt_len: {PROMPT_LEN} decode_len: {DECODE_LEN}\n")
        f.write("# mini-infer: median of 1 warmup + 5 repeats (this project's own --bench)\n")
        f.write("# llama.cpp: llama-bench's own avg of 5 repeats (-r 5); "
                "'center' column holds median for mini-infer and avg for llama.cpp -- not the same\n")
        f.write("# statistic, reported side by side as each tool natively provides it, not normalized\n")
        f.write(f"# generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("engine,phase,n_threads,stat_kind,center_tok_per_s,low,high\n")
        for engine, phase, nt, kind, center, low, high in rows:
            f.write(f"{engine},{phase},{nt},{kind},{center:.4f},{low:.4f},{high:.4f}\n")
    print(f"wrote {args.out}")

    print("\n# summary (center tok/s)")
    print(f"{'engine':<12}{'phase':<10}{'n_threads':>10}{'stat':>8}{'tok/s':>12}")
    for engine, phase, nt, kind, center, _, _ in rows:
        print(f"{engine:<12}{phase:<10}{nt:>10}{kind:>8}{center:>12.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
