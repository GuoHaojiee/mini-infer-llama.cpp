# 阶段 1:朴素 F32 前向 + 对拍 HF

## 做了什么

- `src/context.cpp`:tensor 元数据池(bump 分配)+ 一个额外的低级原语
  `mi_context_view`(自定义 ne/nb/offset 的裸 view 构造器)。阶段 1 没有
  计算图,这个函数是 model_qwen2.cpp 手写前向时构造"per-head 切片"
  "KV cache 写入目标""转置读取视图"的唯一入口;阶段 2 的
  `mi_graph_view/permute/reshape/transpose` 会在这个原语上面加 op/src
  做图依赖记录。
- `src/ops_cpu.cpp`:get_rows、rms_norm、mul、add(带广播)、mul_mat、
  rope(NEOX)、softmax(因果 mask + scale)、silu、scale、cpy 十个 kernel,
  全部通过统一的 `mi_f32_ptr(t,i0,i1,i2,i3)` 按 ne/nb 寻址,不假设任何
  操作数是连续的。
- `src/model_qwen2.cpp`:直接函数调用串起的前向 —— embedding、每层
  RMSNorm+权重缩放、Q/K/V(带 bias)投影、RoPE、按 head 手工循环的 GQA
  因果注意力(细节见下)、O 投影、残差、RMSNorm、SwiGLU FFN、残差、
  最终 RMSNorm、tied lm_head。
- `src/main.cpp`:`-c/--ctx`、`-t`(阶段 3 前先占位)、`-n`、`--prompt-ids`、
  `--dump`(teacher forcing 单次 prefill 导出 checkpoint)、
  `--dump-incremental`(额外加的模式,逐 token 导出 logits,用于验证
  batch/incremental 一致性,见下)。stdout 只输出生成的 token id(一行
  一个,给脚本用),解码文本和诊断信息都在 stderr。
- `scripts/encode.py`:HF tokenizer + `apply_chat_template` -> ids 文件
  (一行、空格分隔)。C++ 侧不做任何编码。
- `scripts/ref_hf.py`:HF transformers,fp32、`attn_implementation="eager"`。
  `dump` 子命令做 teacher forcing 单次前向,导出 embd/每层/output_norm/
  logits;`generate` 子命令支持两种基线(`--method generate` 用
  `model.generate`,`--method stepwise` 每步重新完整前向,见下)。
- `scripts/compare.py`:跑通整条对拍链路,写
  `bench/results/accuracy_f32_vs_hf.txt`。
- `scripts/make_tiny_qwen2_fixture.py` + `tests/test_model_qwen2.cpp`:
  一个随机权重的迷你 Qwen2(2 层、n_embd=8)GGUF,CTest 里做不依赖 HF/
  真实模型的结构性测试,包括 PROMPTS.md 要求的"批量 prefill 与逐 token
  prefill 最终 logits 一致"验证(逐 bit 相等)。

## 关键设计决策

### mul_mat 设计成完全按 ne/nb 寻址,不假设连续

这是整个注意力实现能这么简单的原因。标准做法(llama.cpp/ggml)里,
`scores @ V` 这一步要求 V 在 KV cache 里按"反过来"的布局存储(ne0 是
position 而不是 head_dim),这样 mul_mat 的快速路径(假设连续行)才能用。
本项目的 mul_mat 从第一天就写成"两个操作数都按 nb[] 寻址,不管 ne0 是不是
连续",所以完全不需要那个技巧:V-cache 和 K-cache 用**同样**的布局
(`[head_dim, n_head_kv, n_ctx]`),`scores @ V` 时对 V 的某个 head 切片
现场构造一个**纯元数据**的转置 view(交换 ne 和 nb,数据指针不变),直接
喂给同一个 mul_mat kernel。`tests/test_ops_cpu.cpp` 里的
`test_mul_mat_transposed_view` 专门测了这个 view 的正确性。
代价是显式付出的:这条路径在阶段 4 加 SIMD 点积时不能直接套用(SIMD 点积
需要连续行),那时候会在 mul_mat 里加一条"检测到两个操作数都连续 ->
走快速路径"的分支,不连续的场景(全部的注意力内部矩阵乘)继续走这条
scalar 通用路径 —— 现在的实现已经是那条"通用回退路径"本身。

### 每次 forward 都新建一个 scratch context,用完即删

阶段 1 没有 gallocr,`mi_qwen2_forward` 每次调用都新建一个足够大(公式
估出来再乘 2 加 1MB 余量)的 `mi_context`,跑完整个前向后释放。这是
故意先不优化的:阶段 2b 的职责就是把这里变成零分配,阶段 1 的目标是
"跑对",不是"跑快"。

### KV cache 与 dump checkpoint 的对应关系

`mi_qwen2_forward_params::dump` 在这几个点被调用:embedding 查表之后
(`"embd"`)、每层残差相加之后(`"layer_{i}"`)、最终 RMSNorm 之后
(`"output_norm"`)、lm_head 之后(`"logits"`)。这个约定在
`model_qwen2.h` 里写死,阶段 2 改成图之后这几个调用点不变。

## 调试笔记(按 PROMPTS.md 建议的顺序)

- **RoPE 配对方式**:NEOX 风格是把 `[0, n_dims/2)` 和
  `[n_dims/2, n_dims)` 两半对应位置配对(`p` 与 `p+half`),不是相邻两个
  元素配对。频率公式 `freq_p = base^(-2p/n_dims)`、旋转角
  `theta = pos * freq_p`,和 HF `modeling_qwen2.py` 里
  `rotate_half` + 两份拼接的 cos/sin 表展开后的公式完全一致(见
  `tests/test_ops_cpu.cpp::test_rope_known_rotation` 的手算验证)。
- **GQA 的 head 映射**:`kv_head = h / (n_head / n_head_kv)`,即 query
  head 按组连续映射到同一个 kv head(head 0..6 -> kv 0,7..13 -> kv 1),
  对应 HF `repeat_kv` 的 `repeat_interleave` 语义,不是"隔一个取一个"。
- **QKV bias**:Q/K/V 三个投影都带 bias,O 投影和 FFN 三个投影都不带 ——
  直接对着 HF `Qwen2Attention.__init__` 里几个 `nn.Linear(..., bias=?)`
  核对过。
- **RMSNorm 的 eps**:从 GGUF 的
  `qwen2.attention.layer_norm_rms_epsilon` 读,不硬编码;算出来是
  ~1e-6,和 HF config 的 `rms_norm_eps` 一致。
- **KV cache 位置偏移**:RoPE 和 softmax 的因果 mask 都用同一个
  `n_past`,行 r(这次调用批次内第 r 个 token)的绝对位置是
  `n_past + r`——预填充(prefill,n_past=0)和续写(decode,n_past>0)
  是同一套公式,唯一的输入差异是 `n_past` 的值。
- **mask**:softmax kernel 内部按 `n_past + r + 1` 算允许看到的 key
  数量,不物化 mask 矩阵。`tests/test_ops_cpu.cpp::test_softmax_causal_mask`
  专门用极端值(99)验证被 mask 掉的位置不管数值多大都不会泄漏进结果。

以上六处都是先写单元测试(用手算的小规模例子)钉住行为,再接到真实模型
上跑通的 —— 真正花时间的不是写公式,是每次跑出不对的结果后回去对着
HF 源码一行行核对(比如 Q8_0 类型 id 那种"想当然"的错误在阶段 0 就吃过
一次亏,这次特意每一步都去读 `modeling_qwen2.py` 源码而不是凭记忆)。

## 两处停下来问过的地方(对拍标准的执行细节)

严格按 SPEC.md 第 5 条("不得放宽标准,除非我同意"),这两处不是我自己
决定放宽的,是发现字面标准有问题后问过、拿到明确答复才这样做的。

### 1. "最大相对误差 < 1e-3" 用什么公式

最朴素的逐元素公式 `|a-b|/(|b|+eps)`,在 logits(151936 维)上跑出来
最大值是 6.12,远超 1e-3。追查发现:所有超标的元素(占比 0.024%)对应
的 HF 参考值本身就在 1e-5 量级(而真实 logit 中位数是 5.6),纯比值公式
被这种"两边都是浮点噪声底、只是符号不同的机器零点"的元素拉爆,和实现
质量无关。跟对称公式 `|a-b|/(|a|+|b|+eps)` 一样的问题依然存在(最大值
0.994)——凡是"取比值再取 max"的公式,只要测试集里有一个这样的元素就会
失效。最终改用 `numpy.allclose` 风格的绝对+相对组合容差
`|a-b| <= atol + rtol*|b|`(`atol=1e-2`,`rtol=1e-3`,对应 SPEC 的
"1e-3"),这是数值计算里处理"两个值都可能接近零"场景的标准做法。三种
公式的数值 `bench/results/accuracy_f32_vs_hf.txt` 里都如实报告,不隐藏
朴素公式算出来的"6.12"。

### 2. "贪心生成 32 个 token 与 HF 完全一致"用 HF 的什么调用

用 `model.generate(do_sample=False)` 跑出来前 22 个 token 完全一致,
第 23 个开始分叉。追查:把这个位置的完整上下文(42+22=64 个 token)单独
做一次 `model(..., use_cache=False)` 前向,HF 自己算出来的 argmax 和
mini-infer 一致;但 `model.generate()`(不管 `use_cache=True` 还是显式
传 `False`)在这一步给出的是另一个 token。也就是说 **HF 的
`generate()` 路径本身和它自己的一次性前向路径在这个位置对不上**,而
这个位置 top1/top2 的 logit 差距只有 0.165(比较接近)。这更像是
`generate()` 内部实现(不管是 KV cache 还是别的什么机制)和整段重算之间
存在的一点数值差异,不是 mini-infer 的 bug。最终 `scripts/ref_hf.py`
的 `generate` 子命令加了 `--method stepwise`(每一步都对完整历史重新做
一次不带 cache 的前向,不调用 `generate()`),这个基线下 mini-infer 的
32 个 token **完全一致**;`generate()` 基线的结果也如实保留在报告里
(22/32),说明分叉的来源是 HF 自己的路径差异,不是被隐藏了。

## 测试结果

```
$ ctest --test-dir build          # Release
100% tests passed, 0 tests failed out of 9
```
(Debug、Tsan 两种构建配置下同样 9/9 通过。)

`tests/test_ops_cpu.cpp`(13 组手算断言,覆盖全部 10 个 kernel,包括
mul_mat 的转置 view 场景、RoPE 的已知旋转角、softmax 的因果边界)和
`tests/test_model_qwen2.cpp`(迷你随机权重模型:hparams 解析、prefill
输出有限且形状正确、**批量 prefill 与逐 token 增量 decode 的 logits 逐
bit 相等**)全部通过,细节见对应源文件顶部注释。

`bench/results/accuracy_f32_vs_hf.txt`(真实 Qwen2.5-0.5B-Instruct F32
模型,42 token 的中文 instruct 提示词):

- 各层 checkpoint(embd + 24 层 + output_norm + logits)的最大绝对误差
  全部在 1e-3 量级以内(embd 精确为 0,因为只是查表)。
- logits:最大绝对误差 4.36e-4;`allclose(atol=1e-2, rtol=1e-3)` 全部
  151936×42 个元素都满足;**所有 42 个位置 top-1 完全一致**。
- 贪心生成 32 个 token:对 `stepwise`(无 cache 逐步重算)基线**完全
  一致**;对 `generate()` 基线前 22 个一致,第 23 个起分叉(原因见上)。
- 批量 prefill 与逐 token 增量 decode 的 logits:42/42 位置逐 bit 相等
  (真实模型上验证,`tests/test_model_qwen2.cpp` 里用迷你模型也验证过
  一次)。

## 面试官可能追问的 3 个问题

**Q1:你说 mul_mat 完全通用(不假设连续),这样会不会让阶段 4 的 SIMD
优化变得很别扭?**

不会变得别扭,只是不能"就地"加速这条路径。阶段 4 的 AVX2/F16C 点积面向
的是权重矩阵乘(mul_mat 的两个操作数都是连续的行),会在 mul_mat 里加一条
"两边都连续 -> 走快速点积"的分支,不满足条件(全部的注意力内部矩阵乘:
QK^T、scores@V)继续走现在这条 scalar 通用循环。也就是说现在写的这份
实现本身就会成为阶段 4 的"回退路径",不需要重写,只是多一条判断。真正的
代价是注意力内部这部分矩阵乘永远吃不到 SIMD —— 但对 0.5B 这个规模、
head_dim=64 这样的小矩阵,收益本来就有限,不值得为了它牺牲现在的简洁性。

**Q2:批量 prefill 和逐 token decode 的 logits 逐 bit 相等,这个"逐
bit"是巧合还是设计出来的?**

是设计出来的,原因是每个位置的计算路径本身不依赖"这次调用里还有哪些
其它位置"。RoPE 用的是 `n_past + 行号` 算绝对位置,softmax 的因果
mask 用同一个 `n_past` 算允许看到的 key 数量,KV cache 写入用的是
`n_past` 决定的绝对偏移——这三处都只依赖"当前位置的绝对序号"和"KV
cache 里已经有的历史",不依赖"这次 forward 调用一共处理了几个新
token"。所以对固定位置 i 而言,不管是在一次处理 42 个 token 的批次里
算出来的,还是在处理到第 i 个 token 时单独算出来的,输入(同样的历史
KV、同样的 Q)和计算路径(同样的 kernel、同样的操作顺序)完全相同,
IEEE 754 浮点运算是确定性的,自然逐 bit 相等。如果这里不是逐 bit
相等,那说明某处引入了"依赖批次大小"的分支(比如某个循环边界写错、
KV cache 偏移算错),这个测试就是用来专门捕获这类 bug 的。

**Q3:HF 的 `generate()` 和它自己的一次性前向对不上,你怎么确定这不是
mini-infer 通过某种方式"凑巧对上"了一次性前向、但其实两边都有各自的
bug 只是没有互相暴露?**

因为验证链条不是只有这一个点。第一,42 个位置的 teacher forcing 对拍
(每层误差、最终 logits 误差、全部位置 top-1)是独立于生成过程的,直接
比较的是"给定完全相同的历史,这一步算出来的分布对不对",这一项全部
通过。第二,32 个 token 的 `stepwise` 基线本身就是"每一步都重新做一次
完整的一次性前向"(不依赖 mini-infer 的实现方式),它和 mini-infer
逐 token 生成的结果完全一致,说明两边在**每一步**、不仅仅是第一步,都
算出了相同的下一个 token。第三,batch-vs-incremental 的自洽性检查证明
了 mini-infer 自己内部在"重新计算 vs 增量计算"之间没有分歧。三项加在
一起,能排除"两边凑巧在某个特定输入上一致、但各自独立地错"这种情况——
这种巧合需要在 42 个 teacher-forcing 位置和 32 个连续生成步骤上都恰好
发生,概率上不可能。
