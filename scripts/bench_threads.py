#!/usr/bin/env python3
"""Stage 3's thread-scaling measurement, per PROMPTS.md: fixed model (F32),
fixed prompt length, fixed generation length; thread counts 1/2/4/.../
physical-core-count; mul_mat's three modes, with a few chunk sizes swept
for MI_MULMAT_DYNAMIC. Every number comes from mini_infer --bench actually
running the real model (std::chrono around just the prefill call and the
decode loop's forward calls, in-process -- see main.cpp -- so per-process
overhead like mmap/gallocr-reserve never pollutes the reported tok/s).
Follows SPEC.md's "基准约定": prefill and decode tok/s reported in separate
CSVs, each point is 1 warmup + 5 measured repeats (median/min/max), thread
pinning is on (mini_infer's default), and system info (lscpu, SMT/hybrid
topology, turbo status, compiler version) is recorded alongside the data.
Raw numbers only -- no "speedup" column, no conclusions (SPEC.md).

Usage:
    python3 scripts/bench_threads.py --model <f32.gguf> [--cpp-bin build/mini_infer]
"""
import argparse
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

N_CTX = 128  # comfortably fits PROMPT_LEN + DECODE_LEN with margin; gallocr's
# reserve cost at this size is negligible (measured separately, ~0.5s
# process-wide including model load) and --bench doesn't time it anyway.
PROMPT_LEN = 32
DECODE_LEN = 16  # yields DECODE_LEN - 1 measured decode-forward calls (see
# main.cpp's --bench branch: the last sampled token is never fed back).
DYNAMIC_CHUNKS = [1, 8, 32]
N_WARMUP = 1
N_REPEATS = 5


def make_ids_file(path: str, n: int, vocab_cap: int = 100) -> None:
    with open(path, "w") as f:
        f.write(" ".join(str(i % vocab_cap) for i in range(n)) + "\n")


def detect_physical_cores() -> int:
    """Unique (package, core_id) pairs under /sys -- distinguishes physical
    cores from SMT siblings. Falls back to os.cpu_count() (logical) if the
    topology isn't exposed (e.g. some containers/non-Linux)."""
    base = "/sys/devices/system/cpu"
    seen = set()
    try:
        for name in os.listdir(base):
            if not (name.startswith("cpu") and name[3:].isdigit()):
                continue
            core_path = f"{base}/{name}/topology/core_id"
            if not os.path.exists(core_path):
                continue
            with open(core_path) as f:
                core_id = f.read().strip()
            pkg_path = f"{base}/{name}/topology/physical_package_id"
            pkg = "0"
            if os.path.exists(pkg_path):
                with open(pkg_path) as f:
                    pkg = f.read().strip()
            seen.add((pkg, core_id))
    except OSError:
        pass
    return len(seen) if seen else (os.cpu_count() or 1)


def detect_hybrid() -> str:
    """Best-effort big.LITTLE/hybrid detection: distinct 'cpu MHz max' or
    core types under /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq
    would differ across core types on a hybrid part. Reported as a fact for
    the record, not used to change what gets benchmarked (SPEC.md asks
    only that it be recorded)."""
    base = "/sys/devices/system/cpu"
    max_freqs = set()
    try:
        for name in os.listdir(base):
            if not (name.startswith("cpu") and name[3:].isdigit()):
                continue
            p = f"{base}/{name}/cpufreq/scaling_max_freq"
            if os.path.exists(p):
                with open(p) as f:
                    max_freqs.add(f.read().strip())
    except OSError:
        pass
    if len(max_freqs) > 1:
        return f"possibly hybrid (distinct max frequencies: {sorted(max_freqs)})"
    return "uniform (single max frequency across cores, or topology not exposed)"


def detect_turbo_status() -> str:
    candidates = [
        "/sys/devices/system/cpu/intel_pstate/no_turbo",
        "/sys/devices/system/cpu/cpufreq/boost",
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                return f"{p} = {f.read().strip()}"
    return "not exposed in this environment (checked intel_pstate/no_turbo, cpufreq/boost)"


def gather_sysinfo() -> str:
    lines = []
    lines.append(f"platform: {platform.platform()}")
    lines.append(f"python: {sys.version.split()[0]}")
    try:
        cxx_version = subprocess.run(["c++", "--version"], capture_output=True, text=True, check=True)
        lines.append(f"compiler: {cxx_version.stdout.splitlines()[0]}")
    except (OSError, subprocess.CalledProcessError):
        lines.append("compiler: <c++ --version failed>")
    lines.append(f"physical_cores_detected: {detect_physical_cores()}")
    lines.append(f"logical_cpus (os.cpu_count()): {os.cpu_count()}")
    lines.append(f"hybrid_check: {detect_hybrid()}")
    lines.append(f"turbo_status: {detect_turbo_status()}")
    lines.append("")
    lines.append("--- lscpu ---")
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, check=True)
        lines.append(lscpu.stdout)
    except (OSError, subprocess.CalledProcessError) as e:
        lines.append(f"<lscpu failed: {e}>")
    return "\n".join(lines)


def run_bench(cpp_bin: str, model: str, ids_path: str, n_ctx: int, n_threads: int, mode: str,
              chunk: int) -> dict:
    cmd = [
        cpp_bin, model, "--prompt-ids", ids_path, "--bench", "-c", str(n_ctx), "-n", str(DECODE_LEN),
        "-t", str(n_threads), "--mulmat-mode", mode,
    ]
    if mode == "dynamic":
        cmd += ["--mulmat-chunk", str(chunk)]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    line = proc.stdout.strip()
    fields = dict(kv.split("=") for kv in line.split())
    return {k: float(v) for k, v in fields.items()}


def measure_point(cpp_bin: str, model: str, ids_path: str, n_ctx: int, n_threads: int, mode: str,
                   chunk: int):
    for _ in range(N_WARMUP):
        run_bench(cpp_bin, model, ids_path, n_ctx, n_threads, mode, chunk)
    prefill_vals, decode_vals = [], []
    for _ in range(N_REPEATS):
        r = run_bench(cpp_bin, model, ids_path, n_ctx, n_threads, mode, chunk)
        prefill_vals.append(r["prefill_tok_per_s"])
        decode_vals.append(r["decode_tok_per_s"])
    return prefill_vals, decode_vals


def thread_sweep(n_phys: int):
    threads = []
    p = 1
    while p < n_phys:
        threads.append(p)
        p *= 2
    threads.append(n_phys)
    return sorted(set(threads))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cpp-bin", default="build/mini_infer")
    ap.add_argument("--out-prefix", default="bench/results/threads")
    args = ap.parse_args()

    n_phys = detect_physical_cores()
    threads_list = thread_sweep(n_phys)

    sysinfo = gather_sysinfo()
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    with open(f"{args.out_prefix}_sysinfo.txt", "w") as f:
        f.write(sysinfo + "\n")
    print(f"wrote {args.out_prefix}_sysinfo.txt")

    prefill_rows = []
    decode_rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        ids_path = os.path.join(tmpdir, "prompt.txt")
        make_ids_file(ids_path, PROMPT_LEN)

        configs = []
        for nt in threads_list:
            configs.append((nt, "single", 0))
            configs.append((nt, "static", 0))
            for c in DYNAMIC_CHUNKS:
                configs.append((nt, "dynamic", c))

        for i, (nt, mode, chunk) in enumerate(configs):
            label = f"{mode}" + (f"/chunk={chunk}" if mode == "dynamic" else "")
            print(f"[{i + 1}/{len(configs)}] n_threads={nt} mode={label} ...", file=sys.stderr)
            t_start = time.time()
            prefill_vals, decode_vals = measure_point(args.cpp_bin, args.model, ids_path, N_CTX, nt,
                                                        mode, chunk)
            elapsed = time.time() - t_start
            print(f"    done in {elapsed:.1f}s", file=sys.stderr)

            prefill_rows.append((nt, mode, chunk, prefill_vals))
            decode_rows.append((nt, mode, chunk, decode_vals))

    def write_csv(path: str, rows, phase: str):
        with open(path, "w") as f:
            f.write(f"# {phase} tok/s -- mini-infer stage 3 thread-scaling bench\n")
            f.write(f"# model: {args.model}\n")
            f.write("# model format: F32\n")
            f.write(f"# n_ctx: {N_CTX} prompt_len: {PROMPT_LEN} decode_len: {DECODE_LEN}\n")
            f.write(f"# warmup: {N_WARMUP} repeats: {N_REPEATS}\n")
            f.write(f"# generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# physical_cores: {n_phys} (see {args.out_prefix}_sysinfo.txt for full topology)\n")
            f.write("n_threads,mode,dynamic_chunk,rep,tok_per_s\n")
            for nt, mode, chunk, vals in rows:
                for rep, v in enumerate(vals):
                    f.write(f"{nt},{mode},{chunk},{rep},{v:.4f}\n")
        print(f"wrote {path}")

    write_csv(f"{args.out_prefix}_prefill.csv", prefill_rows, "prefill")
    write_csv(f"{args.out_prefix}_decode.csv", decode_rows, "decode")

    def print_summary(rows, phase: str):
        print(f"\n# {phase} summary (median [min, max] tok/s)")
        print(f"{'n_threads':>9} {'mode':<10} {'chunk':>6} {'median':>10} {'min':>10} {'max':>10}")
        for nt, mode, chunk, vals in rows:
            print(f"{nt:>9} {mode:<10} {chunk:>6} {statistics.median(vals):>10.3f} "
                  f"{min(vals):>10.3f} {max(vals):>10.3f}")

    print_summary(prefill_rows, "prefill")
    print_summary(decode_rows, "decode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
