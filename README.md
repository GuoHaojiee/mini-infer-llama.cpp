# mini-infer

从零实现的 C++17 端侧 LLM 推理引擎,纯 CPU,参照 llama.cpp 的架构思路,以
计算图为核心。能加载 GGUF 格式的 Qwen2.5-0.5B-Instruct 并完成自回归生成。
不用第三方 C++ 库,不复制 ggml/llama.cpp 的源码,只参考公开的 GGUF 格式
规范和 Q8_0 块布局。

核心设计:

1. **gallocr**:静态规划 + best-fit 复用 + 原地写入,decode 稳态运行期
   零动态分配(`--wrap=malloc,free` 验证过)。
2. **线程池**:常驻线程 + 自实现 sense-reversing barrier,mul_mat 三种
   并行模式(single/static/dynamic fetch_add)。
3. **Q8_0 + AVX2**:直接加载 `llama-quantize` 产出的 GGUF,块布局与
   llama.cpp 一致;手写 AVX2 整数点积 + F16C 硬件转换,标量回退路径是
   真回退。
4. **批量注意力**:所有 attention head 一次算完(`MI_OP_ATTN_QK`/
   `MI_OP_ATTN_KQV`,GQA 头共享用索引映射,不拷贝 K/V),配合"无计算节点
   跳过 barrier",单次 decode 的节点数从 3461 降到 797。

## 构建

```bash
cmake -S . -B build                       # 默认 Release, -O3
cmake --build build -j$(nproc)

cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j$(nproc)

cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Tsan
cmake --build build-tsan -j$(nproc)
```

`tools/llama_logits` 是唯一链接外部代码的地方(一个 llama.cpp checkout,
只用于 Q8_0 对拍),默认指向 `../mini-llama.cpp/third_party/llama.cpp`
(`-DMI_LLAMA_CPP_ROOT=...` 可改路径);找不到时 CMake 跳过这一个
target,其余不受影响。

## 测试

```bash
ctest --test-dir build
ctest --test-dir build-debug
setarch $(uname -m) -R ctest --test-dir build-tsan   # 关掉 ASLR,避免和 TSan 的固定 shadow memory 布局冲突
```

`test_zero_alloc` 只在 Release/Debug 下构建(TSan 自己的分配器接管和
`--wrap=malloc,free` 没验证过能否混用)。

## 运行

```bash
# 编码 prompt(分词依赖 Python 侧的 transformers)
python3 scripts/encode.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --chat "介绍一下你自己" --out ids.txt

# 贪心生成 64 个 token,4 线程
build/mini_infer model.gguf --prompt-ids ids.txt -c 2048 -t 4 -n 64
```

stdout 只输出生成的 token id(一行一个),解码文本和诊断信息在 stderr。
完整参数见 `src/main.cpp` 或 `--help`。

## 架构

```
GGUF(mmap)→ gguf.cpp 解析 → model_qwen2.cpp 建 mi_tensor 叶子(不拷贝数据)
  → mi_kv_cache_init:分配 KV cache,gallocr 按最坏情况预演两张图申请 arena,
    启动线程池
  → 每步 mi_qwen2_forward:
      build_model_graph 搭图(Q/K/V 权重是 Q8_0 时自动插入量化节点,
      三份共享同一输入的量化去重;注意力批量算完,不按 head 拆节点)
      → gallocr 分配偏移(不产生堆分配)
      → 线程池按拓扑序执行,view/reshape 类节点不参与 barrier
  → logits → 贪心采样 → 下一个 token
```

| 文件 | 作用 |
|---|---|
| `tensor.h/.cpp` | `mi_tensor` 结构、类型元信息(F32/F16/Q8_0/Q8_0_ACT/I32) |
| `context.h/.cpp` | 张量元数据池 + 数据 arena |
| `gguf.h/.cpp` | GGUF v3 解析(mmap,零拷贝) |
| `graph.h/.cpp` | 计算图构建 + 拓扑排序,`exec_nodes`(跳过无计算节点) |
| `gallocr.h/.cpp` | best-fit + 原地复用 + reserve/plan 两阶段 |
| `threadpool.h/.cpp` | 常驻线程池,自实现 barrier |
| `ops_cpu.h/.cpp` | 所有数值 kernel,含批量注意力 `attn_qk`/`attn_kqv` |
| `quant.h/.cpp` + `quant_avx2.cpp` | F16↔F32、Q8_0 量化/点积(标量 + AVX2/F16C) |
| `model_qwen2.h/.cpp` | Qwen2 架构组装、KV cache |
| `main.cpp` | CLI:prefill/decode、`--dump`、`--gallocr-stats`、`--graph-stats`、`--bench` |
| `tools/dump.cpp` | GGUF 元数据检查工具 |
| `tools/llama_logits.cpp` | 对拍用,导出 llama.cpp 的 logits |
| `tools/bench_matvec/bench_memread/bench_memcpy.cpp` | kernel 级微基准 |

## 基准结果

硬件:AMD Ryzen 5 5600H(6C/12T),g++ 11.4.0,`-O3` Release。完整
`lscpu` 见 `bench/results/env_info.txt`。每组 1 次预热 + 5 次重复取
中位数,原始数据在 `bench/results/*.csv`,数字全部来自实际运行。

### decode tok/s(Q8_0,阶段 4 基线到当前)

| 线程数 | 阶段 4/5 | 阶段 6 后 | llama.cpp | 比值 |
|---|---|---|---|---|
| 1 | 15.407 | 36.807 | 36.783 | 1.00x |
| 3 | -- | 58.195 | 67.669 | 0.86x |
| 4 | -- | 64.035 | 72.525 | 0.88x |
| 6 | 33.139 | 62.537 | 70.431 | 0.89x |

单线程从 llama.cpp 的 41% 升到打平,6 线程从 44% 升到 89%。详细过程
(每一步做了什么、为什么、有效带宽 GB/s、barrier 次数)见
`docs/stage6.md`。纯读带宽上限(`tools/bench_memread.cpp`)在这台机器
上 1 线程 27.8 GB/s,3-4 线程见顶(41-43 GB/s)。

### F16 vs Q8_0(阶段 6 后)

| 配置 | 1 线程 | 4 线程 | 文件大小 |
|---|---|---|---|
| F16 AVX2+F16C | 17.770 | 38.952 | 994157056 |
| Q8_0 AVX2 | 36.469 | 63.877 | 531068416 |

### gallocr 内存(Q8_0,n_ctx=1024)

| case | 实际 arena | 无块复用 | 理论下界 | 逐张量独立分配 |
|---|---|---|---|---|
| prefill 1024 | 123.5 MB | 4427.6 MB | 115.5 MB | 5790.6 MB |
| decode | 0.12 MB | 4.32 MB | 0.11 MB | 5.66 MB |

理论下界不计原地复用,所以有时比实际 arena 还大,详见 `docs/stage6.md`。

### 精度

- F32 vs HF(teacher forcing,42 位置):`allclose(atol=0.01, rtol=0.001)`
  全部满足,贪心生成 32 个 token 与 HF 逐位一致。
- Q8_0 vs llama.cpp(416 位置):top-1 一致率 98.3%,余弦相似度均值
  0.9995。

完整派生指标见 `bench/results/summary.md`(`scripts/summarize.py` 机械
计算生成)。

## 已知局限

- 纯 CPU,无 GPU/加速器后端。
- 只支持 F32/F16/Q8_0 权重,不支持 Q4_0/Q4_K 等其他量化格式。
- 只支持 Qwen2.5 架构,没做成通用多架构加载器。
- 单请求,无 batching。
- 只有贪心采样。
- 分词依赖 Python(`scripts/encode.py`),C++ 侧只接受 token id。
- prefill 不是本项目的优化目标(decode 才是),仍是纯标量注意力矩阵乘,
  n_ctx 较大时较慢。

## 目录结构

```
mini-infer/
  CMakeLists.txt      Release/Debug/Tsan 三个配置
  src/                引擎本体
  tests/              每个模块一个测试可执行文件,CTest 管理
  tools/              mi_dump、llama_logits、bench_* 系列微基准
  scripts/            Python 编码/对拍/基准脚本
  bench/results/      基准与精度报告原始输出
  docs/               每个阶段的设计笔记 + architecture.md
```

设计笔记(做了什么、为什么、踩过的坑、常见追问)在 `docs/stageN.md`;
`docs/architecture.md` 是跨阶段的核心数据结构文档。
