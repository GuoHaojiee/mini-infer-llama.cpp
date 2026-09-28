#!/usr/bin/env python3
"""Stage 5: mechanical computation of the specific derived metrics
PROMPTS.md lists, from bench/results' raw files only -- no new
measurements, no judgment calls, every number annotated with its formula,
source file+line, and measurement conditions (SPEC.md: "这是纯计算,不写
评价;缺数据的项写 TODO"). Writes bench/results/summary.md.

Usage:
    python3 scripts/summarize.py
"""
import csv
import os
import re
import statistics
import sys

RESULTS = "bench/results"


def read_lines(path):
    with open(path) as f:
        return f.readlines()


_CSV_HEADER_FIRST_COLS = {"n_threads", "config", "engine"}


def read_csv_rows(path):
    with open(path) as f:
        return [row for row in csv.reader(f)
                if row and not row[0].startswith("#") and row[0] not in _CSV_HEADER_FIRST_COLS]


class Section:
    def __init__(self, title):
        self.title = title
        self.items = []

    def add(self, label, value, formula, source, conditions):
        self.items.append((label, value, formula, source, conditions))

    def todo(self, label, reason):
        self.items.append((label, "TODO", None, None, reason))

    def render(self):
        out = [f"## {self.title}\n"]
        for label, value, formula, source, conditions in self.items:
            out.append(f"### {label}\n")
            out.append(f"- 数值: {value}\n")
            if formula:
                out.append(f"- 公式: {formula}\n")
            if source:
                out.append(f"- 来源: {source}\n")
            if conditions:
                out.append(f"- 测量条件: {conditions}\n")
            out.append("\n")
        return "".join(out)


def section_gallocr():
    s = Section("gallocr:各 prefill 档位的激活峰值 MB vs 基线 MB")
    path = f"{RESULTS}/gallocr.txt"
    if not os.path.exists(path):
        s.todo("gallocr peak vs baseline", f"{path} 不存在")
        return s
    lines = read_lines(path)
    n_ctx = None
    model = None
    for i, line in enumerate(lines):
        if line.startswith("n_ctx:"):
            n_ctx = line.split(":", 1)[1].strip()
        if line.startswith("model:"):
            model = line.split(":", 1)[1].strip()
        if line.startswith("prefill n="):
            parts = line.split()
            case = parts[0] + " " + parts[1]
            n_tokens, n_past, peak_mb, base_mb, reduction, n_inplace, n_bestfit = parts[2:]
            s.add(
                f"{case} (n_tokens={n_tokens})",
                f"peak={peak_mb} MB, baseline={base_mb} MB, reduction={reduction}",
                "reduction = baseline_bytes / peak_bytes (mi_gallocr_get_stats() 实测字段,"
                "非估算)",
                f"{path}:{i + 1}",
                f"model={model}, n_ctx={n_ctx}, F32",
            )
    return s


def section_parallel():
    s = Section("并行:decode 最快线程数下 dynamic 相对单线程的倍数;"
                "同线程数下 dynamic 相对 static 的百分比差")
    path = f"{RESULTS}/threads_decode.csv"
    if not os.path.exists(path):
        s.todo("parallel speedup", f"{path} 不存在")
        return s
    rows = read_csv_rows(path)  # n_threads,mode,dynamic_chunk,rep,tok_per_s
    by_key = {}
    for r in rows:
        nt, mode, chunk, rep, tok = r
        by_key.setdefault((int(nt), mode), []).append(float(tok))

    static_by_threads = {nt: statistics.median(v) for (nt, mode), v in by_key.items() if mode == "static"}
    if not static_by_threads:
        s.todo("decode 最快线程数", "threads_decode.csv 里没有 static 模式的数据")
        return s
    fastest_t = max(static_by_threads, key=lambda nt: static_by_threads[nt])
    fastest_static_median = static_by_threads[fastest_t]
    single_thread_median = static_by_threads.get(1)

    s.add(
        "decode 最快线程数 t (按 static 模式各线程数中位数取最大值确定)",
        f"t = {fastest_t} (median tok/s = {fastest_static_median:.3f})",
        "t = argmax_{n_threads} median(tok_per_s | mode=static, n_threads)",
        f"{path}",
        "F32, prompt_len=32, decode_len=16 (见该文件表头)",
    )

    # threads_decode.csv has one dynamic row per chunk size; report each.
    by_chunk = {}
    for r in rows:
        nt, mode, chunk, rep, tok = r
        if mode == "dynamic":
            by_chunk.setdefault((int(nt), int(chunk)), []).append(float(tok))
    for (nt, chunk), vals in sorted(by_chunk.items()):
        if nt != fastest_t:
            continue
        med = statistics.median(vals)
        if single_thread_median:
            speedup = med / single_thread_median
            s.add(
                f"dynamic(chunk={chunk})@t={fastest_t} 相对单线程(static@1)的倍数",
                f"{speedup:.3f}x  (dynamic median={med:.3f} tok/s, static@1 median={single_thread_median:.3f} tok/s)",
                "speedup = median(tok_per_s | mode=dynamic, n_threads=t, chunk) / "
                "median(tok_per_s | mode=static, n_threads=1)",
                f"{path}",
                "F32, prompt_len=32, decode_len=16",
            )
        static_t_median = static_by_threads[fastest_t]
        pct_diff = (med - static_t_median) / static_t_median * 100.0
        s.add(
            f"dynamic(chunk={chunk}) 相对 static,同线程数 t={fastest_t} 的百分比差",
            f"{pct_diff:+.2f}%  (dynamic median={med:.3f}, static median={static_t_median:.3f})",
            "pct_diff = (median(dynamic) - median(static)) / median(static) * 100, "
            "both at n_threads=t",
            f"{path}",
            "F32, prompt_len=32, decode_len=16",
        )
    return s


def section_quant():
    s = Section("量化:Q8_0 相对 F16 的文件体积降幅;"
                "同线程数下 Q8_0 AVX2 相对 F16 SIMD 的 decode 提升百分比")
    sizes_path = f"{RESULTS}/model_sizes.txt"
    if os.path.exists(sizes_path):
        lines = read_lines(sizes_path)
        f16_bytes = q8_bytes = None
        f16_line = q8_line = None
        for i, line in enumerate(lines):
            if "f16.gguf" in line:
                f16_bytes = int(re.search(r"(\d+) bytes", line).group(1))
                f16_line = i + 1
            if "q8_0.gguf" in line:
                q8_bytes = int(re.search(r"(\d+) bytes", line).group(1))
                q8_line = i + 1
        if f16_bytes and q8_bytes:
            reduction_pct = (f16_bytes - q8_bytes) / f16_bytes * 100.0
            s.add(
                "Q8_0 相对 F16 的文件体积降幅",
                f"{reduction_pct:.2f}%  (F16={f16_bytes} bytes, Q8_0={q8_bytes} bytes)",
                "reduction_pct = (f16_bytes - q8_0_bytes) / f16_bytes * 100",
                f"{sizes_path}:{f16_line},{q8_line}",
                "real stat() sizes, 与 docs/models.md 阶段 0 记录一致",
            )
    else:
        s.todo("Q8_0 vs F16 文件体积降幅", f"{sizes_path} 不存在")

    decode_path = f"{RESULTS}/quant_decode.csv"
    if os.path.exists(decode_path):
        rows = read_csv_rows(decode_path)  # config,n_threads,rep,tok_per_s
        by_cfg = {}
        for r in rows:
            cfg, nt, rep, tok = r
            by_cfg.setdefault((cfg, int(nt)), []).append(float(tok))
        for nt in sorted({k[1] for k in by_cfg}):
            f16 = by_cfg.get(("F16_avx2", nt))
            q8 = by_cfg.get(("Q8_0_avx2", nt))
            if f16 and q8:
                f16_med = statistics.median(f16)
                q8_med = statistics.median(q8)
                pct = (q8_med - f16_med) / f16_med * 100.0
                s.add(
                    f"Q8_0 AVX2 相对 F16 AVX2/F16C 的 decode 提升百分比,n_threads={nt}",
                    f"{pct:+.2f}%  (Q8_0_avx2 median={q8_med:.3f} tok/s, F16_avx2 median={f16_med:.3f} tok/s)",
                    "pct = (median(Q8_0_avx2) - median(F16_avx2)) / median(F16_avx2) * 100",
                    f"{decode_path}",
                    "prompt_len=16, decode_len=8, n_ctx=64 (见该文件表头)",
                )
    else:
        s.todo("Q8_0 AVX2 vs F16 AVX2 decode 提升百分比", f"{decode_path} 不存在")
    return s


def section_accuracy():
    s = Section("精度:F32 vs HF 的 logits 最大相对误差;Q8_0 vs llama.cpp 的 top-1 一致率")
    f32_path = f"{RESULTS}/accuracy_f32_vs_hf.txt"
    if os.path.exists(f32_path):
        lines = read_lines(f32_path)
        for i, line in enumerate(lines):
            if line.startswith("max rel error (naive"):
                s.add(
                    "F32 vs HF, logits 最大相对误差(naive: |a-b|/(|b|+eps))",
                    line.split(":", 1)[1].strip(),
                    "SPEC.md 字面公式 max_rel_naive = |a-b|/(|b|+eps),逐位置逐 vocab 维取最大值",
                    f"{f32_path}:{i + 1}",
                    "F32 GGUF vs HF transformers fp32 eager, teacher forcing,详见该文件",
                )
            if line.startswith("max rel error (symmetric"):
                s.add(
                    "F32 vs HF, logits 最大相对误差(symmetric: |a-b|/(|a|+|b|+eps))",
                    line.split(":", 1)[1].strip(),
                    "max_rel_sym = |a-b|/(|a|+|b|+eps),用于说明 naive 公式在浮点噪声地板附近"
                    "被单个元素主导的问题(见 docs/stage1.md)",
                    f"{f32_path}:{i + 1}",
                    "同上",
                )
            if line.startswith("allclose(atol"):
                s.add(
                    "F32 vs HF, allclose(atol=0.01, rtol=0.001) 是否全部满足",
                    line.split(":", 1)[1].strip(),
                    "numpy.allclose 风格:|a-b| <= atol + rtol*|b|,阶段 1 验收实际使用的标准"
                    "(见 docs/stage1.md 里记录的、经用户确认的标准调整)",
                    f"{f32_path}:{i + 1}",
                    "同上",
                )
    else:
        s.todo("F32 vs HF 最大相对误差", f"{f32_path} 不存在")

    q8_path = f"{RESULTS}/accuracy_q8_vs_llamacpp.txt"
    if os.path.exists(q8_path):
        lines = read_lines(q8_path)
        for i, line in enumerate(lines):
            if line.startswith("top1_agreement_rate"):
                s.add(
                    "Q8_0 vs llama.cpp, top-1 一致率",
                    line.split(":", 1)[1].strip(),
                    "top1_agreement_rate = (argmax(mi_logits, axis=1) == argmax(llama_logits, axis=1)).mean()",
                    f"{q8_path}:{i + 1}",
                    "同一 Q8_0 GGUF,同一 token 序列,teacher forcing,详见该文件",
                )
    else:
        s.todo("Q8_0 vs llama.cpp top-1 一致率", f"{q8_path} 不存在")
    return s


def section_speed():
    s = Section("速度:t 线程 Q8_0 decode tok/s,以及相同条件下 llama.cpp 的 tok/s 和两者比值")
    path = f"{RESULTS}/llama_bench_compare.csv"
    if not os.path.exists(path):
        s.todo("Q8_0 decode tok/s vs llama.cpp", f"{path} 不存在")
        return s
    rows = read_csv_rows(path)  # engine,phase,n_threads,stat_kind,center_tok_per_s,low,high
    by_key = {}
    for r in rows:
        engine, phase, nt, kind, center, low, high = r
        by_key[(engine, phase, int(nt))] = float(center)
    for nt in sorted({k[2] for k in by_key}):
        mi = by_key.get(("mini-infer", "decode", nt))
        lc = by_key.get(("llama.cpp", "decode", nt))
        if mi is not None and lc is not None:
            ratio = mi / lc
            s.add(
                f"Q8_0 decode tok/s, n_threads={nt}",
                f"mini-infer={mi:.3f} tok/s (median of 5), llama.cpp={lc:.3f} tok/s (avg of 5), "
                f"ratio(mini-infer/llama.cpp)={ratio:.4f}",
                "ratio = mini_infer_decode_tok_per_s / llama_bench_decode_tok_per_s "
                "(两侧统计量不同:median vs avg,原始如实并列,不归一化,见该文件表头注释)",
                f"{path}",
                "同一 Q8_0 GGUF,-ngl 0(纯 CPU),prompt_len=16,decode_len=8",
            )
    return s


def main() -> int:
    sections = [
        section_gallocr(),
        section_parallel(),
        section_quant(),
        section_accuracy(),
        section_speed(),
    ]
    out_path = f"{RESULTS}/summary.md"
    with open(out_path, "w") as f:
        f.write("# bench/results 派生指标汇总(纯机械计算,无评价)\n\n")
        f.write("每一项给出:数值、计算公式、来源文件(与行号,若适用)、测量条件。"
                "本文件由 `scripts/summarize.py` 从 `bench/results/` 下的原始文件生成,"
                "不引入任何未在那些文件中出现过的数字。\n\n")
        for s in sections:
            f.write(s.render())
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
