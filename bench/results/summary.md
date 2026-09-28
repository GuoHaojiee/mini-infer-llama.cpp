# bench/results 派生指标汇总(纯机械计算,无评价)

每一项给出:数值、计算公式、来源文件(与行号,若适用)、测量条件。本文件由 `scripts/summarize.py` 从 `bench/results/` 下的原始文件生成,不引入任何未在那些文件中出现过的数字。

## gallocr:各 prefill 档位的激活峰值 MB vs 基线 MB
### prefill n=128 (n_tokens=128)
- 数值: peak=6.250 MB, baseline=415.312 MB, reduction=66.45x
- 公式: reduction = baseline_bytes / peak_bytes (mi_gallocr_get_stats() 实测字段,非估算)
- 来源: bench/results/gallocr.txt:13
- 测量条件: model=/home/guohaojie/Guo/mini-llama.cpp/models/qwen2.5-0.5b-instruct-f32.gguf, n_ctx=1024, F32

### prefill n=512 (n_tokens=512)
- 数值: peak=26.750 MB, baseline=2165.250 MB, reduction=80.94x
- 公式: reduction = baseline_bytes / peak_bytes (mi_gallocr_get_stats() 实测字段,非估算)
- 来源: bench/results/gallocr.txt:14
- 测量条件: model=/home/guohaojie/Guo/mini-llama.cpp/models/qwen2.5-0.5b-instruct-f32.gguf, n_ctx=1024, F32

### prefill n=1024 (n_tokens=1024)
- 数值: peak=57.500 MB, baseline=5674.500 MB, reduction=98.69x
- 公式: reduction = baseline_bytes / peak_bytes (mi_gallocr_get_stats() 实测字段,非估算)
- 来源: bench/results/gallocr.txt:15
- 测量条件: model=/home/guohaojie/Guo/mini-llama.cpp/models/qwen2.5-0.5b-instruct-f32.gguf, n_ctx=1024, F32

## 并行:decode 最快线程数下 dynamic 相对单线程的倍数;同线程数下 dynamic 相对 static 的百分比差
### decode 最快线程数 t (按 static 模式各线程数中位数取最大值确定)
- 数值: t = 6 (median tok/s = 12.766)
- 公式: t = argmax_{n_threads} median(tok_per_s | mode=static, n_threads)
- 来源: bench/results/threads_decode.csv
- 测量条件: F32, prompt_len=32, decode_len=16 (见该文件表头)

### dynamic(chunk=1)@t=6 相对单线程(static@1)的倍数
- 数值: 3.985x  (dynamic median=11.149 tok/s, static@1 median=2.798 tok/s)
- 公式: speedup = median(tok_per_s | mode=dynamic, n_threads=t, chunk) / median(tok_per_s | mode=static, n_threads=1)
- 来源: bench/results/threads_decode.csv
- 测量条件: F32, prompt_len=32, decode_len=16

### dynamic(chunk=1) 相对 static,同线程数 t=6 的百分比差
- 数值: -12.67%  (dynamic median=11.149, static median=12.766)
- 公式: pct_diff = (median(dynamic) - median(static)) / median(static) * 100, both at n_threads=t
- 来源: bench/results/threads_decode.csv
- 测量条件: F32, prompt_len=32, decode_len=16

### dynamic(chunk=8)@t=6 相对单线程(static@1)的倍数
- 数值: 4.538x  (dynamic median=12.697 tok/s, static@1 median=2.798 tok/s)
- 公式: speedup = median(tok_per_s | mode=dynamic, n_threads=t, chunk) / median(tok_per_s | mode=static, n_threads=1)
- 来源: bench/results/threads_decode.csv
- 测量条件: F32, prompt_len=32, decode_len=16

### dynamic(chunk=8) 相对 static,同线程数 t=6 的百分比差
- 数值: -0.54%  (dynamic median=12.697, static median=12.766)
- 公式: pct_diff = (median(dynamic) - median(static)) / median(static) * 100, both at n_threads=t
- 来源: bench/results/threads_decode.csv
- 测量条件: F32, prompt_len=32, decode_len=16

### dynamic(chunk=32)@t=6 相对单线程(static@1)的倍数
- 数值: 4.488x  (dynamic median=12.558 tok/s, static@1 median=2.798 tok/s)
- 公式: speedup = median(tok_per_s | mode=dynamic, n_threads=t, chunk) / median(tok_per_s | mode=static, n_threads=1)
- 来源: bench/results/threads_decode.csv
- 测量条件: F32, prompt_len=32, decode_len=16

### dynamic(chunk=32) 相对 static,同线程数 t=6 的百分比差
- 数值: -1.63%  (dynamic median=12.558, static median=12.766)
- 公式: pct_diff = (median(dynamic) - median(static)) / median(static) * 100, both at n_threads=t
- 来源: bench/results/threads_decode.csv
- 测量条件: F32, prompt_len=32, decode_len=16

## 量化:Q8_0 相对 F16 的文件体积降幅;同线程数下 Q8_0 AVX2 相对 F16 SIMD 的 decode 提升百分比
### Q8_0 相对 F16 的文件体积降幅
- 数值: 46.58%  (F16=994157056 bytes, Q8_0=531068416 bytes)
- 公式: reduction_pct = (f16_bytes - q8_0_bytes) / f16_bytes * 100
- 来源: bench/results/model_sizes.txt:6,7
- 测量条件: real stat() sizes, 与 docs/models.md 阶段 0 记录一致

### Q8_0 AVX2 相对 F16 AVX2/F16C 的 decode 提升百分比,n_threads=1
- 数值: -10.34%  (Q8_0_avx2 median=15.407 tok/s, F16_avx2 median=17.183 tok/s)
- 公式: pct = (median(Q8_0_avx2) - median(F16_avx2)) / median(F16_avx2) * 100
- 来源: bench/results/quant_decode.csv
- 测量条件: prompt_len=16, decode_len=8, n_ctx=64 (见该文件表头)

### Q8_0 AVX2 相对 F16 AVX2/F16C 的 decode 提升百分比,n_threads=6
- 数值: +8.09%  (Q8_0_avx2 median=33.139 tok/s, F16_avx2 median=30.659 tok/s)
- 公式: pct = (median(Q8_0_avx2) - median(F16_avx2)) / median(F16_avx2) * 100
- 来源: bench/results/quant_decode.csv
- 测量条件: prompt_len=16, decode_len=8, n_ctx=64 (见该文件表头)

## 精度:F32 vs HF 的 logits 最大相对误差;Q8_0 vs llama.cpp 的 top-1 一致率
### F32 vs HF, logits 最大相对误差(naive: |a-b|/(|b|+eps))
- 数值: 6.121701e+00
- 公式: SPEC.md 字面公式 max_rel_naive = |a-b|/(|b|+eps),逐位置逐 vocab 维取最大值
- 来源: bench/results/accuracy_f32_vs_hf.txt:48
- 测量条件: F32 GGUF vs HF transformers fp32 eager, teacher forcing,详见该文件

### F32 vs HF, logits 最大相对误差(symmetric: |a-b|/(|a|+|b|+eps))
- 数值: 9.943926e-01
- 公式: max_rel_sym = |a-b|/(|a|+|b|+eps),用于说明 naive 公式在浮点噪声地板附近被单个元素主导的问题(见 docs/stage1.md)
- 来源: bench/results/accuracy_f32_vs_hf.txt:49
- 测量条件: 同上

### F32 vs HF, allclose(atol=0.01, rtol=0.001) 是否全部满足
- 数值: True
- 公式: numpy.allclose 风格:|a-b| <= atol + rtol*|b|,阶段 1 验收实际使用的标准(见 docs/stage1.md 里记录的、经用户确认的标准调整)
- 来源: bench/results/accuracy_f32_vs_hf.txt:50
- 测量条件: 同上

### Q8_0 vs llama.cpp, top-1 一致率
- 数值: 0.971154  (404/416 positions)
- 公式: top1_agreement_rate = (argmax(mi_logits, axis=1) == argmax(llama_logits, axis=1)).mean()
- 来源: bench/results/accuracy_q8_vs_llamacpp.txt:9
- 测量条件: 同一 Q8_0 GGUF,同一 token 序列,teacher forcing,详见该文件

## 速度:t 线程 Q8_0 decode tok/s,以及相同条件下 llama.cpp 的 tok/s 和两者比值
### Q8_0 decode tok/s, n_threads=1
- 数值: mini-infer=15.434 tok/s (median of 5), llama.cpp=37.791 tok/s (avg of 5), ratio(mini-infer/llama.cpp)=0.4084
- 公式: ratio = mini_infer_decode_tok_per_s / llama_bench_decode_tok_per_s (两侧统计量不同:median vs avg,原始如实并列,不归一化,见该文件表头注释)
- 来源: bench/results/llama_bench_compare.csv
- 测量条件: 同一 Q8_0 GGUF,-ngl 0(纯 CPU),prompt_len=16,decode_len=8

### Q8_0 decode tok/s, n_threads=6
- 数值: mini-infer=33.203 tok/s (median of 5), llama.cpp=75.018 tok/s (avg of 5), ratio(mini-infer/llama.cpp)=0.4426
- 公式: ratio = mini_infer_decode_tok_per_s / llama_bench_decode_tok_per_s (两侧统计量不同:median vs avg,原始如实并列,不归一化,见该文件表头注释)
- 来源: bench/results/llama_bench_compare.csv
- 测量条件: 同一 Q8_0 GGUF,-ngl 0(纯 CPU),prompt_len=16,decode_len=8

