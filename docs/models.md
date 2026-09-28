# 模型文件

mini-infer 本身不做 HF -> GGUF 转换或量化(那是 llama.cpp 的工作,SPEC.md 也要求
用 llama.cpp 的 `convert_hf_to_gguf.py` 与 `llama-quantize` 生成对拍用的模型)。
本阶段直接复用了同机上另一个项目 `/home/guohaojie/Guo/mini-llama.cpp` 已经生成
好的三个文件(经用户确认可以复用,见阶段 0 开工前的确认记录),没有重新下载/
转换/编译,以节省时间;但生成这些文件所用的命令、llama.cpp 版本、转换参数都
如实记录在下面,均来自该项目自己的 `docs/ENV.md` 和 `scripts/prepare_models.sh`
(不是本项目凭空编造的)。

## 模型来源

- HuggingFace:`Qwen/Qwen2.5-0.5B-Instruct`
- 本地缓存:`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct`
  以及 `mini-llama.cpp/models/Qwen2.5-0.5B-Instruct-hf`(通过 `hf download` 落盘)

## llama.cpp 版本

- 仓库:`https://github.com/ggml-org/llama.cpp`(浅克隆,`--depth 1`)
- commit:`85ca3b52c33f5477f75985a251543f5f73010e8a`
- 编译:
  ```
  cmake -S third_party/llama.cpp -B third_party/llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON
  cmake --build third_party/llama.cpp/build -j12
  ```
  (`GGML_NATIVE=ON` 在这台机器上等效于开启 AVX2/FMA/F16C,详见
  `mini-llama.cpp/docs/ENV.md`)

## 生成命令

来自 `mini-llama.cpp/scripts/prepare_models.sh`:

```bash
HF_DIR=models/Qwen2.5-0.5B-Instruct-hf
LLAMA_CPP=third_party/llama.cpp
QUANTIZE=$LLAMA_CPP/build/bin/llama-quantize

hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir "$HF_DIR"

# F32:阶段 1 对拍 HF 用
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$HF_DIR" \
  --outfile models/qwen2.5-0.5b-instruct-f32.gguf --outtype f32

# F16:阶段 4 量化对比的公平基线
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$HF_DIR" \
  --outfile models/qwen2.5-0.5b-instruct-f16.gguf --outtype f16

# Q8_0:阶段 4 量化实现的目标格式
# --pure:保证所有权重矩阵统一量化为同一类型,1D 的 norm/bias 张量仍保留 F32
"$QUANTIZE" --pure models/qwen2.5-0.5b-instruct-f16.gguf \
  models/qwen2.5-0.5b-instruct-q8_0.gguf Q8_0
```

（该脚本还顺带生成了一个 Q4_0 文件,mini-infer 不支持 Q4_0,不使用它。)

## 三个文件的实测统计

路径:`/home/guohaojie/Guo/mini-llama.cpp/models/`。文件大小为 `ls -la` 实测字节数
(非估算);张量类型统计为本项目 `tools/mi_dump` 的实际输出(逐张量类型计数,
非估算/非转述)。

| 文件 | 用途 | 实测大小(字节) | 实测大小 |
|---|---|---|---|
| `qwen2.5-0.5b-instruct-f32.gguf` | 阶段 1 对拍 HF | 1982079488 | 1.85 GiB |
| `qwen2.5-0.5b-instruct-f16.gguf` | 阶段 4 基线 | 994157056 | 948 MiB |
| `qwen2.5-0.5b-instruct-q8_0.gguf` | 阶段 4 量化目标 | 531068416 | 506 MiB |

张量类型统计(`mi_dump <file> | awk -F'\t' '$1=="TENSOR"{c[$4]++} END{for(t in c) print t,c[t]}'`,
共 290 个张量):

| 文件 | f32 | f16 | q8_0 |
|---|---|---|---|
| F32 | 290 | 0 | 0 |
| F16 | 121 | 169 | 0 |
| Q8_0 | 121 | 0 | 169 |

F16/Q8_0 文件里那 121 个仍是 F32 的张量,是每层的 `attn_norm`、`ffn_norm`、
`attn_q/k/v.bias`(24 层 × 5 个 = 120)加上最外层的 `output_norm.weight`
(1 个)——都是 1D 的 norm/bias 张量,`llama-quantize` 本来就不对它们量化,
这是预期行为,不是 `--pure` 没生效(与 mini-llama.cpp 自己核对过的结论一致,
本项目用 `mi_dump` 重新验证了一遍,数字吻合)。169 个量化张量 = 24 层 ×
(attn_q/k/v/output + ffn_gate/up/down = 7 个) = 168,再加上共享的
`token_embd.weight` = 169。

## 关键超参数(供后续阶段核对,来自 GGUF 元数据,非硬编码)

用 `tools/mi_dump` 读出(三个文件一致):

| key | 值 |
|---|---|
| `qwen2.block_count` | 24 |
| `qwen2.embedding_length` | 896 |
| `qwen2.attention.head_count` | 14 |
| `qwen2.attention.head_count_kv` | 2 |
| `qwen2.feed_forward_length` | 4864 |
| `qwen2.rope.freq_base` | 1000000 |
| `qwen2.attention.layer_norm_rms_epsilon` | 1e-06 |
| `qwen2.context_length`(上限,不直接当 n_ctx 用) | 32768 |

head_dim = 896 / 14 = 64。`output.weight` 不存在于任何一个文件中,三个模型都
是 tied embedding,lm_head 用 `token_embd.weight`(阶段 1 需要处理这一点)。
