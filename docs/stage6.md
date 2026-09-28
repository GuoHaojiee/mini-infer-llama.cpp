# 阶段 6:decode 性能优化

本文档随每一步优化增量写入(STAGE6_PROMPT.md 的要求:每步都要有实测数据和
"有效/无效"记录),目前只完成了步骤 0(纯测量,未改任何 kernel 代码)。

## 步骤 0:测量基线(只测不改)

机器与阶段 3-5 相同:AMD Ryzen 5 5600H(6C/12T,双通道 DDR4),g++ 11.4.0,
`-O3` Release。模型:`qwen2.5-0.5b-instruct-q8_0.gguf`(531068416 字节)。
原始数据见 `bench/results/opt_step0.txt`,这里只摘录结论性的读数(数字本身
来自该文件,不重新估算)。

### 1. perf record -g(单线程 Q8_0 decode)

`perf_event_paranoid` 默认是 4(本环境无 root/CAP_PERFMON),用户在本会话
里手动临时降到 1 才能采样。前 10 项按自身耗时占比:

| 占比 | 函数 |
|---|---|
| 46.37% | `mi_fp16_to_fp32` |
| 40.85% | `mi_vec_dot_q8_0_q8_0_avx2` |
| 5.64% | `mi_compute_mul_mat` |
| 1.43% | 匿名命名空间 `visit`(图拓扑排序) |
| 0.88% | `mi_vec_dot_q8_0_q8_0`(分发函数) |
| 0.57% | `mi_quantize_row_q8_0` |
| 0.36% | `__expf_fma`(softmax 里的 exp) |
| 0.30% | `__llrintf`(量化里的四舍五入) |
| 0.28% | `mi_graph_build_forward_expand` |
| 0.24% | `mi_compute_silu` |

mul_mat 一族(`mi_compute_mul_mat` + 两个 `vec_dot_q8_0_q8_0*` 符号 +
`mi_fp16_to_fp32`)合计 93.74%;非 mul_mat 的所有算子加起来不到 3%,
远低于 STAGE6_PROMPT 步骤 4 定的 ">5% 才优化" 门槛,所以 rms_norm/rope/
softmax/silu 现阶段不是目标。

**`mi_fp16_to_fp32` 单独占 46.37%,比 AVX2 整数点积本身(40.85%)还高**——
这是本步骤最重要的发现,比 STAGE6_PROMPT 的"头号嫌疑"猜测更确凿:不是
"软件位运算转换比较慢"这种模糊说法,而是它确实以*函数调用*的形式出现在
热路径里,而不是被内联展开成内联指令流。

### 2. objdump:为什么 `mi_fp16_to_fp32` 没被内联

`objdump -d` 反汇编 `mi_vec_dot_q8_0_q8_0_avx2`(见 `bench/results/opt_step0.txt`
第 4 节完整记录):

- 每个 block 的循环体内有 **2 条 `call` 指令**,对应 `mi_fp16_to_fp32(x[i].d)`
  和 `mi_fp16_to_fp32(y[i].d)`——`mi_fp16_to_fp32` 定义在 `quant.cpp`
  (不同编译单元),没有加 `inline`,项目也没开 LTO,编译器没有跨 TU 内联的
  能力,这就是 perf 测出 46.37% 的直接原因,不是"标量运算慢",是"每 32 个
  元素两次真实的函数调用开销(压栈/跳转/返回)"。
- 全函数 **0 次**出现 `vcvtph2ps`/`vcvtps2ph`(F16C 硬件转换指令)——同一个
  文件里的 `mi_vec_dot_f16_f32_avx2` 已经在用 `_mm256_cvtph_ps` 批量转换
  8 个半精度值,Q8_0 路径的两个缩放因子却完全没走硬件转换这条路,现有的
  F16C 能力被浪费了。
- 水平求和(`vpshufd`/`vpaddd` 那组指令)出现在**每个 block 的循环体内**,
  而不是循环结束后对一个跨 block 的向量累加器做一次——STAGE6_PROMPT 的
  "每个 block 做一次水平求和"的猜测也被证实。
- 标量 FMA(`vfmadd231ss`)确实在用,这部分不是问题。

### 3. 微基准:4864x896 Q8_0 矩阵-向量乘,重复 1000 次

`4864x896` 是这个 GGUF 里 `blk.N.ffn_down.weight` 的形状(k=4864, n=896),
decode 时是矩阵-向量乘(m=1)。新增 `tools/bench_matvec.cpp`,直接循环调用
`mi_vec_dot_q8_0_q8_0`,不经过图/gallocr/线程池,只测 kernel 自身吞吐:

| 版本 | 有效 GB/s(权重矩阵字节数 × 1000 / 耗时) |
|---|---|
| AVX2(默认分发) | 8.9073 |
| 标量(`MI_QUANT_FORCE_SCALAR=1`) | 6.0993 |

AVX2 只比标量快 1.46 倍,这和阶段 4 报告里"Q8_0 AVX2 只比标量快 1.38 倍"
的整体 decode 数字一致——微基准把"非 mul_mat 开销"都排除掉了,数字仍然
这么接近,说明瓶颈确实在 kernel 内部(和 perf 的结论互相印证),不是外层
调度或非计算节点。

### 4. memcpy 可用带宽

新增 `tools/bench_memcpy.cpp`,每个线程拷贝一段互不重叠的内存(512 MiB
buffer,1 次预热 + 20 次重复):

| 线程数 | 有效 GB/s(拷贝字节数,不含读写各算一次的另一倍) |
|---|---|
| 1 | 12.9312 |
| 2 | 14.6465 |
| 4 | 15.8093 |
| 6 | 14.8598 |
| 12 | 14.4427 |

从 2 线程往上基本就到平台期(14.4~15.8 之间波动,没有随线程数继续上升的
趋势)——这是这台机器双通道 DDR4 的内存带宽上限,不是线程调度问题。

**把这个数字和 Q8_0 AVX2 微基准的 8.91 GB/s 放在一起看**:8.91 GB/s 明显
低于单线程 memcpy 的 12.93 GB/s,说明单线程 Q8_0 kernel 目前还没有到
带宽瓶颈,还有计算侧的空间(和 STAGE6_PROMPT 背景部分"计算受限,不是带宽
受限"的推断一致,现在有了这台机器上的直接测量支撑,而不只是从 llama.cpp
的比值推断)。

### 5. 每个 decode step 的 barrier 次数

新增 `--graph-stats` CLI 选项(`main.cpp`)+ `mi_qwen2_forward_params::out_graph`
钩子(`model_qwen2.h/.cpp`,纯新增字段,默认 `nullptr`,对现有调用方零影响,
10 个测试全部保持通过)。真实模型、n_ctx=64、decode 风格调用(n_tokens=1,
n_past=19)测得:

```
n_nodes=3461 n_nop_nodes=1416   (reshape=72 + view=1008 + transpose=336)
```

`threadpool.cpp` 的 `run_current_graph` 对每一个 `g->nodes` 节点都无条件调用
一次 `barrier_wait`(不区分节点是否真的做了计算),所以 `n_nodes` 就是一次
decode 的 barrier 次数:3461 次,其中 **40.9%(1416 次)是纯元数据重新
解释(view/reshape/transpose),没有搬动任何数据**——这些节点仍然要让
所有线程做一次 `fetch_add` + 可能的自旋等待,是 STAGE6_PROMPT 步骤 3
("无计算节点不参与 barrier")要解决的具体目标,现在有了确切的比例数字。

## 步骤 0 结论(供确认,尚未改动任何 kernel)

1. **头号目标确认**:`mi_fp16_to_fp32` 未内联的函数调用开销(46.37%)
   是当前最大的单一瓶颈,比 AVX2 整数点积本身还大,与 STAGE6_PROMPT 的
   "头号嫌疑"完全吻合,且有 objdump 级别的证据(2 次 `call`、0 次
   `vcvtph2ps`)。
2. 水平求和按 block 做而非按行做,是可衡量但次要的问题(不在同一个函数
   调用的采样桶里单独体现,但 objdump 已确认存在)。
3. 非 mul_mat 算子(softmax/silu/rms_norm/rope)合计 < 3% 自身耗时,当前
   不构成优化目标,符合 STAGE6_PROMPT 步骤 4 的 ">5%" 门槛。
4. 单线程 Q8_0 kernel(8.91 GB/s)明显低于单线程 memcpy 上限(12.93 GB/s),
   确认是计算受限,还有改进空间,不是已经顶到内存墙。
5. 40.9% 的 barrier 调用发生在零计算节点上,是步骤 3 的量化目标。

## 步骤 1:修复 Q8_0 AVX2 点积

在实现前,用户额外提出三点修正,均在本步骤一并处理:

1. 带宽上限改用纯读测试(`tools/bench_memread.cpp`,AVX2 顺序读 + 求和,
   1/2/3/4/5/6/8/12 线程),不再用 memcpy 的数字作为"上限"——memcpy 同时
   产生读和写两份内存总线流量,对一个几乎纯读的负载(Q8_0 decode 只读权重、
   写极少的输出)会低估真正可用的读带宽,而 6 线程 decode 实测已经超过了
   memcpy 报的数字。
2. `tools/bench_matvec.cpp` 增加 "DRAM" 模式:循环遍历 ≥256 MB 的多份独立
   矩阵(避免同一份 4.6 MB 矩阵在 16 MB L3 里一直命中缓存);原模式保留并
   标注为 "L3-resident"。两种模式都报告。
3. 本步骤的 kernel 改动(见下)。

### 做了什么

- **权重缩放因子改用硬件 F16C,且确保真的内联进 kernel 内部**——不是简单加个
  `target("f16c")` 就完事:辅助函数 `f16c_scale_to_f32` 和调用它的行级
  `mi_vec_dot_q8_0_wact_avx2_f16c`/`mi_vec_dot_q8_0_q8_0_avx2_f16c` 用
  **完全相同**的 `target("avx2,fma,f16c")` 字符串声明,GCC 的跨函数内联
  要求调用者的 target 特性集合覆盖被调用者,字符串不匹配就不会内联(哪怕
  两边都开了 f16c)——用 objdump 验证过内联确实发生(见下)。没有 F16C 的
  假设性机器上,回退到一张 65536 项的查找表(`mi_fp16_to_fp32_lut`,
  quant.cpp 里 magic-static 懒初始化,不占用 decode 步骤的运行期分配
  预算)。两个后端(`_f16c`/`_lut`)都在 `quant.h` 里单独暴露出来,专门给
  测试用(和已有的 `mi_q8_0_block_isum_scalar`/`_avx2` 先例一致),不依赖
  测试机的真实 CPU 特性就能覆盖两条路径。
- **激活侧量化改存 float 缩放因子,不再走 fp16 往返**——新增
  `MI_TYPE_Q8_0_ACT`(`tensor.h`/`tensor.cpp`,32 元素/block、36
  字节/block)、`block_q8_0_act`(`quant.h`,4 字节 float scale + 32 个
  int8)、`mi_quantize_row_q8_0_act`(`quant.cpp`)。`mi_graph_quantize_q8_0`
  (`graph.cpp`)的输出张量类型从 `MI_TYPE_Q8_0` 改成 `MI_TYPE_Q8_0_ACT`;
  `mi_compute_quantize_q8_0`/`mi_compute_mul_mat` 的 Q8_0 分支
  (`ops_cpu.cpp`)相应改读 `block_q8_0_act*`,点积换成新的
  `mi_vec_dot_q8_0_wact`(权重 `block_q8_0` fp16 scale × 激活
  `block_q8_0_act` float scale)。这一步之后,mul_mat 的 Q8_0 路径每个
  block 只剩**一次** fp16→fp32 转换(权重那一侧),激活那一侧的浮点缩放
  因子直接读,不需要任何转换——因为激活值量化前本来就是 F32,强行经过
  fp16 只是白白丢精度,没有任何格式兼容性要求需要满足(这个 buffer 从不
  落盘,只活在 gallocr 管理的图内部)。`mi_vec_dot_q8_0_q8_0`(两边都是
  `block_q8_0`)保留且同样做了下面的行级累加器优化,纯粹是因为
  `tests/test_quant.cpp` 直接调用它做标量/AVX2 对拍,SPEC.md 不允许收紧
  已有测试的通过标准;mul_mat 生产路径不再调用它。
- **整行一个 `__m256` 累加器,行末一次水平求和**——新增
  `q8_0_block_dot32x8`(`quant_avx2.cpp` 匿名命名空间,返回未做水平求和的
  8 个 int32 部分和),行级函数把每个 block 的 8 个部分和转成 float、乘以
  该 block 的 `dx*dy` 标量、`_mm256_fmadd_ps` 累加进同一个跨 block 共享的
  `__m256` 寄存器,循环结束后才调用一次 `hsum_ps`。这在数学上和旧版本
  (每个 block 都水平求和成一个标量、再逐 block scalar FMA)算的是同一个
  和,只是重新分组(block i、lane j 对最终结果的贡献 `dx_i*dy_i*part[i][j]`
  不管哪种分组方式都一样),所以仍然满足 PROMPTS.md 的 `< 1e-5` 相对误差
  标准,但不再是"必然 bit-exact"——`tests/test_quant.cpp` 里原来那句
  "两者应该 bit-exact"的注释已经更新为准确描述,断言本身(`< 1e-5`)
  没有变。
- `tests/test_quant.cpp` 新增:`block_q8_0_act` 布局检查、
  `mi_quantize_row_q8_0_act` 与 `mi_quantize_row_q8_0` 的一致性(qs[] 完全
  相同,d 在 fp16 精度内一致)、`mi_vec_dot_q8_0_wact` 标量/avx2_f16c/
  avx2_lut/分发函数的等价性检查、`mi_vec_dot_q8_0_q8_0_avx2_f16c`/`_lut`
  两个后端各自对标量的等价性检查(不依赖测试机是否真的缺 F16C)。

### 为什么这样设计

**为什么不是"随便加个 target 属性就行"。** 步骤 0 的 objdump 已经证明
`mi_fp16_to_fp32` 之所以没被内联,根本原因是它和调用者不在同一个编译
单元、没开 LTO——单纯给它的*调用者*加 `target("f16c")`、却不改
`mi_fp16_to_fp32` 本身所在的位置,不会解决跨 TU 内联的问题。真正的修复
是在 `quant_avx2.cpp`*这个文件内部*重新实现一个 F16C 版本的转换函数,
和调用它的行级函数用同一个 target 字符串——这样 GCC 才有内联的资格
(同一编译单元 + target 特性匹配),用 objdump 验证内联确实发生,而不是
假设"加了 target 应该就快了"。

**为什么保留 `mi_vec_dot_q8_0_q8_0`(两个 `block_q8_0`)这条路径,即使
生产代码不再用它。** SPEC.md 的规则 5 不允许收紧已有测试的通过标准,
`tests/test_quant.cpp` 的 `test_vec_dot_q8_0_scalar_vs_avx2` 直接调用这个
符号名做标量/AVX2 对拍。与其为了"生产不用了"就弱化或删掉这个函数、让
测试失去意义,不如让它继续正确且同样享受这次优化——这样测试仍然在验证
一条真实、优化过的代码路径,只是这条路径恰好不再是 mul_mat 的调用对象。

**为什么给激活量化单独引入一个新的 `mi_type`(`MI_TYPE_Q8_0_ACT`),而不是
让 `MI_TYPE_Q8_0` 的字节布局"看情况变化"。** gallocr/`mi_nbytes` 完全按
`tensor->type` 走类型特征表来算一个张量该占多少字节(`tensor.cpp` 的
`mi_get_type_traits`),这是阶段 2b 就定下的通用设计,没有为任何 op
硬编码大小。如果不新增类型、只是让"同一个 `MI_TYPE_Q8_0`"在不同场合有
不同的 block 字节数(权重 34 字节、激活 36 字节),`mi_nbytes` 就没有办法
知道该用哪个数字——这正是阶段 4 文档记录过的"dummy 权重类型不匹配导致
arena 按错误大小规划"那类坑的翻版。新增一个类型、让它经过完全相同的
通用路径,是唯一不需要在 gallocr/`mi_nbytes` 里特判"这个 Q8_0 张量是不是
激活"的做法。

### 效果(与步骤 0 对比,数据见 `bench/results/opt_step1.{txt,csv}`)

perf 单线程 decode 前 10 项耗时占比中,`mi_fp16_to_fp32` 从 46.37%
**完全消失**;objdump 确认新 kernel 循环体内 0 次 `call`、1 次
`vcvtph2ps`(内联)、水平求和只在循环结束后出现一次。

| 指标 | 步骤 0 | 步骤 1 | 倍数 |
|---|---|---|---|
| 微基准 AVX2,L3-resident,GB/s | 8.9073 | 30.6817 | 3.44x |
| 微基准 AVX2,DRAM(新增),GB/s | (无此模式) | 21.7390 | -- |
| 微基准标量,GB/s | 6.0993 | 7.3417(L3-resident) | 1.20x |
| decode tok/s,1 线程 | 15.407 | 33.832 | 2.196x |
| decode tok/s,6 线程 | 33.139 | 43.052 | 1.299x |
| decode 有效带宽,1 线程(GB/s) | ~8.18(15.407×531MB/1e9) | 17.97 | -- |
| decode 有效带宽,6 线程(GB/s) | ~17.60 | 22.86 | -- |

新增的纯读带宽上限(`tools/bench_memread.cpp`,不是 memcpy):1 线程
27.80 GB/s,峰值在 3-4 线程(41-43 GB/s),之后线程越多反而越慢(这台
机器的内存控制器在用满全部 12 个逻辑核之前就已经饱和)。1 线程 decode
的有效带宽(17.97 GB/s)已经接近 1 线程读带宽上限的 65%,比步骤 0 的
8.91 GB/s 明显更接近内存墙;6 线程 decode(22.86 GB/s)离 3-4 线程读带宽
峰值(41-43 GB/s)还有很大差距,而这个差距不太可能靠继续榨 kernel
算力解决(单线程已经接近它自己的带宽上限了)——更像是线程/barrier
开销的问题,指向步骤 0 已经量化过的"40.9% 的 barrier 调用发生在零计算
节点上",这也是步骤 3 要处理的方向。

Q8_0 vs llama.cpp 对拍(`scripts/compare_q8_llamacpp.py`)重新跑过:
top-1 一致率从 97.12% 略升到 98.32%,其余指标在小数点第三位上有涨有跌
(意料之中:激活量化换成了和 llama.cpp 不同的、内部专用的缩放因子存储
方式,不是"谁更对",两者都满足 SPEC.md 对这一对拍不要求逐位一致的标准)。

全部 10 个测试在 Release/Debug/Tsan 三种配置下继续通过(含新增的
`block_q8_0_act`/`mi_vec_dot_q8_0_wact` 系列测试)。

## 步骤 3 前置调研:每层节点数分解(先汇报,未改代码)

用户要求在动步骤 3(无计算节点不参与 barrier)之前,先搞清楚每层 144 个
节点具体是什么、注意力是否按 head 拆节点、Q/K/V 与 gate/up 是否各自重复
量化。以下全部直接读 `src/model_qwen2.cpp` 的 `forward_layer`/
`build_model_graph`/`linear`/`graph.cpp` 的 `mi_graph_mul_mat` 源码确认,
不是从节点计数反推猜测。

### 每层 144 个节点的精确分解

按是否在 `forward_layer` 的 `for (int h = 0; h < hp.n_head; h++)`
(`model_qwen2.cpp:164`)循环内分两组:

**每头重复一次、共 14 头(112 个节点,占每层 144 个的 77.8%)**:

| op | 每头个数 | 来源(`forward_layer` 里的调用) |
|---|---|---|
| view | 3 | `head_slice(qh_rot,h)`、`kv_head_view(cache_k,...)`、`kv_head_view(cache_v,...)` |
| mul_mat | 2 | `mi_graph_mul_mat(k_head, qh_head)`(QK^T)、`mi_graph_mul_mat(v_head_t, probs)`(attn×V) |
| softmax | 1 | `mi_graph_softmax(scores,...)` |
| transpose | 1 | `mi_graph_transpose(v_head)` |
| cpy | 1 | `mi_graph_cpy(out_head, concat_head_slice(...))`(把这个头的输出写回 `attn_concat`) |

8 个节点 × 14 头 = 112。

**每层只算一次(32 个节点,占 22.2%)**:

| op | 个数 | 来源 |
|---|---|---|
| rms_norm | 2 | attn_norm、ffn_norm |
| mul | 3 | 两次 RMSNorm 的按权重缩放 + SwiGLU 的 `silu(gate)*up` |
| add | 5 | q/k/v 三个 bias add(`linear()` 内部)+ attn 残差 + ffn 残差 |
| mul_mat | 7 | q/k/v/attn_output/ffn_gate/ffn_up/ffn_down 这 7 个 `linear()` 调用各自的矩阵乘 |
| rope | 2 | qh_rot、kh_rot |
| silu | 1 | SwiGLU 的激活函数 |
| cpy | 2 | K cache 写入、V cache 写入(`k_dst`/`v_dst`,整个 K/V 张量一次性写,不按头拆) |
| quantize_q8_0 | 7 | 上面 7 个 mul_mat 各自的权重都是 Q8_0,`mi_graph_mul_mat` 各自插入一个量化节点 |
| reshape | 3 | q/k/v 的 `reshape_heads`(把 `[n_embd,n_tokens]` 切成 `[head_dim,n_head,n_tokens]` 视图,一次性对整个张量做,不按头拆) |

32 个节点(2+3+5+7+2+1+2+7+3=32)。112+32=144,与 `--graph-stats`
实测的每层平均完全吻合(全模型 3461 个节点 = 24×144 + 5 个仅一次的节点:
`get_rows` 1 个 + 最终 `rms_norm`/`mul` 各 1 个 + 最终 `mul_mat`/
`quantize_q8_0`(lm_head)各 1 个)。

### 回答用户的两个具体问题

**注意力是否按 head 拆成独立节点?—— 是,而且是最大头的部分。**
`forward_layer` 第 164-182 行的 `for (int h = 0; h < hp.n_head; h++)`
循环对每一个头都独立构建 view→mul_mat→softmax→view→transpose→mul_mat→cpy
这一条 8 节点的子图,`n_head=14`,所以单是这一个循环就贡献了每层 144
个节点里的 112 个(77.8%)。这 112 个节点里,只有 2×14=28 个是
`mul_mat`(真正做浮点乘加的);其余 84 个(3×14 个 view + 14 个 softmax +
14 个 transpose + 14 个 cpy)要么是纯元数据操作(view/transpose,
STAGE6_PROMPT 步骤 3 的目标——不参与 barrier 应该完全免费),要么是
真实但很小的计算(softmax 只在 `n_kv_total`≤`n_ctx` 个元素上做,decode 时
`n_tokens=1`;cpy 只搬 `head_dim` 个 float)。这解释了为什么步骤 0 测到
"40.9% 的 barrier 是零计算节点"——按头拆分是这个比例的主要来源,因为
每拆一个头就多 3 个 view + 1 个 transpose,四个节点里有四个是纯视图。

**Q/K/V 与 gate/up 是否各自重复量化同一份激活?—— 是,证实存在冗余,
可作为额外的优化目标(但不在步骤 3 范围内)。** `linear()`
(`model_qwen2.cpp:60-63`)每次调用都直接转发给 `mi_graph_mul_mat(w, x)`,
而 `mi_graph_mul_mat`(`graph.cpp:117-124`)只要看到 `a->type==MI_TYPE_Q8_0`
就无条件插入一个新的 `mi_graph_quantize_q8_0(ctx, b)`——它不检查`b`
是否已经在这次前向的更早某处被量化过。`forward_layer` 里:
- `attn_q`/`attn_k`/`attn_v` 三个 `linear()` 调用的 `x` 参数都是同一个
  `normed` 张量(RMSNorm 的输出),权重都是 Q8_0,于是 `normed` 被
  **量化了三次**,产生三份内容完全相同的 `block_q8_0_act` scratch buffer。
- `ffn_gate`/`ffn_up` 两个 `linear()` 调用的 `x` 参数都是同一个 `normed2`,
  同样被**量化了两次**。
- `attn_output`(输入是 `attn_concat`)和 `ffn_down`(输入是 `act`)各自的
  输入是这一层里独一份的张量,量化一次没有浪费。

也就是说每层 7 个 `quantize_q8_0` 节点里,只有 4 个是真正必要的
(`normed` 一次、`attn_concat` 一次、`normed2` 一次、`act` 一次),另外 3
个(q/k/v 里的 2 个多余、gate/up 里的 1 个多余)是对同一份数据的重复
计算,理论上可以去重到 24×4=96 个,比现在的 24×7=168 少 42.9%。
不过步骤 1 的 perf 数据里 `mi_quantize_row_q8_0_act` 只占 1.59% 的总耗时,
就算把这 42.9% 全部消掉,理论上限也只有约 0.68 个百分点的总耗时——
比步骤 3 的 barrier 开销(40.9% 的调用是零计算)小得多,所以不建议现在做,
如实记录在这里供以后需要时参考,不在本次步骤 3 的范围内。

以上为纯调研结果,未修改任何代码;等待确认后再进入步骤 3(无计算节点
不参与 barrier)。

## 步骤 3a:无计算节点不参与执行与 barrier

### 做了什么

`mi_cgraph` 新增 `exec_nodes`/`n_exec_nodes`(`graph.h`),是 `nodes` 的
一个子序列(保持相同的相对拓扑顺序),在图构建阶段(`graph.cpp` 的
`visit()`,后序 DFS 把节点追加进 `nodes` 的同一时刻)就地过滤掉
`mi_op_is_view(op)` 为真的节点(RESHAPE/VIEW/PERMUTE/TRANSPOSE)。
`threadpool.cpp` 的 `run_current_graph` 改成遍历 `exec_nodes`,`nodes`
本身完全不变。

### 为什么在"图构建阶段"过滤,而不是在 gallocr 分配阶段

最初的设计是让 gallocr 的分配遍历(`plan_graph`,单线程、每步跑一次)顺便
产出这个过滤列表——它本来就要遍历一次 `g->nodes` 并且已经对 view 节点
做特判。但检查 `tests/test_threadpool.cpp`/`tests/test_graph.cpp` 发现:
这两个文件都有直接对"eager 分配"的图调用 `mi_graph_compute(g, pool)` /
`mi_graph_compute(g, nullptr)`、**完全不经过 gallocr** 的用例(用来做纯
线程池/纯图执行的单元测试)。如果 `exec_nodes` 只在 gallocr 的分配阶段
才填充,这些测试会因为 `n_exec_nodes` 一直是 0 而执行不了任何节点——图
构建阶段(`visit()`)才是唯一一个"不管有没有 gallocr 都一定会跑一次"的
阶段,所以过滤逻辑必须放在这里。这也让"规划阶段解析完毕"这句话有了更
准确的含义:是图的拓扑构建阶段,不是 gallocr 的内存分配阶段。

`g->nodes` 保持完全不变、只新增一个平行数组,是为了不影响 gallocr 自己
的分配遍历——gallocr 仍然需要看到每一个 view 节点才能正确解析它的
`data` 指针(`gallocr.cpp` 第 246 行左右),而且它的引用计数/释放时机
严格依赖 `nodes` 的原始拓扑顺序,重排或过滤 `nodes` 本身会破坏这套已经
测试过的算法。

byte 预算方面:`mi_graph_new` 现在多分配一个 `capacity` 大小的指针数组,
`model_qwen2.cpp` 里两处用 `2 * node_capacity * sizeof(mi_tensor*)`
估算 `mi_context` 字节预算的地方(`estimate_graph_ctx_bytes`、
`reserve_gallocr` 的 `make_planning_ctx`)都改成了 `3 *`——这两处如果漏改,
`mi_context_alloc_bytes` 的越界检查只在 Debug 构建用 `assert()`(Release
的 `-DNDEBUG` 会编译掉),会在 Release 下静默写穿 malloc 出来的 buffer,
是本步骤唯一一个"改错了不会立刻报错,而是悄悄堆损坏"的风险点,所以
单独在这里点出来。

### 效果

一次 decode 的 barrier 次数(`--graph-stats` 的 `n_exec_nodes`):
3461 → 2045(降低 40.9%,与步骤 0 测得的"零计算节点占比"完全对应)。

| 线程数 | 步骤 1 decode tok/s | 步骤 3a decode tok/s | 变化 |
|---|---|---|---|
| 1 | 33.832 | 33.738 | 基本不变(单线程 barrier 本来就几乎零成本,见 docs/stage3.md) |
| 4 | (未测) | 46.547 | -- |
| 6 | 43.052 | 44.840 | +4.2% |

Q8_0 vs llama.cpp 对拍数字与步骤 1 完全一致(纯调度改动,不改变任何浮点
运算),10 个测试在 Release/Debug/Tsan 三种配置下全部继续通过。

## 步骤 3b:注意力批量构图

### 做了什么

新增两个图算子 `MI_OP_ATTN_QK`/`MI_OP_ATTN_KQV`(`tensor.h`,追加在枚举
末尾,不改变任何已有 op 的数值)和对应的 builder
`mi_graph_attn_qk`/`mi_graph_attn_kqv`(`graph.cpp`)+ kernel
`mi_compute_attn_qk`/`mi_compute_attn_kqv`(`ops_cpu.cpp`),替换掉
`forward_layer` 里原来的按头循环(view→mul_mat→softmax→view→transpose→
mul_mat→cpy,每头 8 个节点 × 14 头)。新的每层注意力只有 4 个节点:

```
scores = attn_qk(qh_rot, cache_k, n_past, n_rep)   // [n_kv_total,n_tokens,n_head]
probs  = softmax(scores, n_past, scale)             // 泛化到支持第 3 维(head)
kqv    = attn_kqv(probs, cache_v, n_rep)            // [head_dim,n_head,n_tokens]
attn_concat = reshape(kqv, [n_embd, n_tokens])      // 免费的 view,不再需要逐头 cpy
```

`cache_k`/`cache_v` 全程作为完整张量传入(`op_params` 只带 `n_past`/
`n_rep`),GQA 的头共享映射 `kvh = h / n_rep` 完全在 kernel 内部按索引
计算,从不构造任何 per-head 的 view 或拷贝 K/V。`mi_compute_softmax`
从只支持 `[n_kv,n_tokens]` 泛化到 `[n_kv,n_tokens,n_head]`(每个
`(token,head)` 对当作一行,和 rms_norm/silu/scale 等算子已有的"按
`ne[1]*ne[2]*ne[3]` 拆行"惯例一致);`n_head` 缺省为 1,旧的 2D 调用点
（其实已经没有了,但如果将来出现)行为不变。

多线程按 `(head, 行)` 切分:`attn_qk` 把 `n_head * n_kv_total` 个单元
(每个单元是"某个头、某个 kv 位置的整行分数")均分给线程;`attn_kqv`
把 `n_head * head_dim` 个单元均分。这和阶段 3 mul_mat 的"每个输出行只被
一个线程算一次"是同一个并行策略,只是把原来"每头一个 mul_mat 节点"里的
行,横向拼接成"所有头一起"的更大一批行。

### 为什么这是 bit-exact 的(不是"应该是",是逐元素验证过的)

对每个输出元素 `(h,i,j)`,新 kernel 的累加顺序和旧的按头 `mul_mat` 完全
相同——这不是巧合,是照着旧代码的寻址公式逐行翻译的:

- 旧:`mi_compute_mul_mat` 的 F32 分支对 `k_head=[head_dim,n_kv_total]`
  (`cache_k` 在头 `kvh` 处的 view)和 `qh_head=[head_dim,n_tokens]`
  (`qh_rot` 在头 `h` 处的 view)算 `dst[i,j] = Σ_{t=0}^{head_dim-1}
  k_head[t,i] * qh_head[t,j]`,其中 `k_head[t,i] = cache_k[t,kvh,i]`、
  `qh_head[t,j] = qh_rot[t,h,j]`(view 的寻址公式决定的)。
- 新:`mi_compute_attn_qk` 直接算
  `dst[i,j,h] = Σ_{d=0}^{head_dim-1} cache_k[d,kvh,i] * qh_rot[d,h,j]`——
  同样的 `d` 从 0 递增到 `head_dim-1`,同样的乘法顺序(`K` 在前、`Q` 在
  后),只是不再通过 view 中转,直接用 `mi_f32_ptr` 按下标读。

`attn_kqv` 同理:旧的 `v_head_t=transpose(v_head)` 令
`v_head_t[t,i]=cache_v[i,kvh,t]`,`mul_mat(v_head_t,probs)` 算
`dst[i,j]=Σ_t v_head_t[t,i]*probs[t,j]=Σ_t cache_v[i,kvh,t]*probs[t,j]`;
新 kernel 直接算 `dst[o,h,j]=Σ_{t=0}^{n_kv_total-1} cache_v[o,kvh,t] *
probs[t,j,h]`,同样的 `t` 递增顺序。

`kqv` 的输出布局 `[head_dim,n_head,n_tokens]`(head_dim 最内层、head 连续
排列)和 `reshape_heads` 拆分 q/k/v 时用的布局完全对称,所以
`mi_graph_reshape(kqv,...)` 产生的 `attn_concat` 视图和旧代码里
`concat_head_slice` 手工拼出来的 `attn_concat` 在字节layout 上完全相同
——不是"数值恰好相等",是同一块内存按同一个公式解释。

**验证方式(不是靠論證,是靠真实跑通过的对拍标准)**:
- `scripts/compare.py`(F32 vs HF,阶段 1 就有的验收标准,`n_head=14`
  真实 GQA 头共享,不是玩具形状):`PASS: True`,logits 最大绝对误差
  4.358e-04,和这次改动之前的历史数值同一量级;贪心生成 32 个 token 与
  HF 的 stepwise 基线逐位一致(32/32);批量 prefill 与逐 token prefill
  42/42 位置逐位一致。
- `scripts/compare_q8_llamacpp.py`:top-1 一致率、余弦相似度等所有数字
  与改动前**完全相同**(97.12%→98.32% 那次是步骤 1 引入的,3b 之后数字
  一位不差),证明 3b 纯粹是调度重排,没有引入任何新的浮点误差来源。
- `tests/test_model_qwen2.cpp` 的批量 vs 逐 token 增量 decode 位级一致
  测试(用真实 GQA 形状的 tiny fixture,`n_head=2,n_head_kv=1`,即
  `n_rep=2`,确实走了头共享分支)继续通过。
- Release/Debug/Tsan 三种构建全部通过(Tsan 下用小矩阵的
  `tests/test_threadpool.cpp` 和 `tests/test_model_qwen2.cpp` 覆盖到了
  新 kernel 的并发路径)。

由于以上全部通过,没有出现"做不到"的情况,不需要放宽任何测试。

### 效果

一次 decode 的节点数从 869(步骤 3a 状态下按头拆分的样子)骤降到
(实际测得,`--graph-stats`):

```
n_nodes=869 n_nop_nodes=96 n_exec_nodes=773
  mul_mat 169  softmax 24  cpy 48  attn_qk 24  attn_kqv 24  reshape 96
  (view=0, transpose=0 -- 按头拆分产生的 view/transpose 全部消失)
```

对比步骤 3a:`n_nodes` 3461→869(-74.9%),`n_exec_nodes`(barrier 次数)
2045→773(-62.2%,累计相对最初基线 -77.7%)。

| 线程数 | 步骤 3a decode tok/s | 步骤 3b decode tok/s | 变化 |
|---|---|---|---|
| 1 | 33.738 | 36.474 | +8.1% |
| 3 | (未测) | 56.922 | -- |
| 4 | 46.547 | 63.333 | +36.1% |
| 6 | 44.840 | 61.213 | +36.5% |

4 线程(63.333)反而略快于 6 线程(61.213)——和 `tools/bench_memread.cpp`
测到的"纯读带宽在 3-4 线程见顶、之后线程越多越慢"完全吻合,进一步确认
这台机器上继续加线程已经没有意义,4 线程左右就是这个规模问题的甜点。

## 步骤 3c:barrier 自旋/yield 策略

### 做了什么

`threadpool.cpp` 新增:
- `MI_BARRIER_STATS=1`(env,读一次缓存):打开 `barrier_wait` 里
  `std::chrono::steady_clock` 计时,累加"非最后到达线程"在自旋/yield 循环
  里花的总时间和总次数,通过 `mi_threadpool_get_barrier_stats` 读取。不开
  这个开关时零额外开销(平时的测速数字都是在它关闭的状态下测的)。
- `MI_BARRIER_SPIN_ONLY=1`(env,读一次缓存):让 `spin_backoff` 永远不
  退化到 `sched_yield()`,无限自旋直到 `generation` 变化。
- `main.cpp` 的 `--bench` 在 decode 循环开始前调用
  `mi_threadpool_reset_barrier_stats()`,循环结束后如果 count>0 就在
  输出行末尾附加 `barrier_wait_count=... avg_barrier_wait_ns=...`。

### 测量结果

单次 decode 调用的等待次数经过交叉验证(`count == n_exec_nodes ×
(n_threads-1) × decode_steps`,例如 4 线程 7 步:`773×3×7=16233`,与实测
完全一致),说明计时点没有插错地方。

平均每次等待时长(当前 spin-then-yield 策略,7 次 decode 调用汇总):

| 线程数 | 平均等待 (ns) |
|---|---|
| 1 | (n_threads=1 时最后到达者判断在同一行代码里立即为真,从不自旋,见 docs/stage3.md) |
| 3 | 4571.9 |
| 4 | 3276.8 |
| 6 | 3715.2 |

3-4.5 微秒远小于 `kSpinLimit=4000` 次 `pause` 指令自旋所需的时间(这台
机器上量级在几十微秒),说明当前工作负载下绝大多数等待在纯自旋阶段就
结束了,几乎不会真的退化到 `sched_yield()`。

纯自旋(`MI_BARRIER_SPIN_ONLY=1`) vs 当前策略(自旋后 yield)的 decode
tok/s,5 次重复取中位数(见 `bench/results/opt_step3c.csv`):

| 线程数 | spin-then-yield(当前) | spin-only | 差异 |
|---|---|---|---|
| 3 | 57.182 | 57.620 | +0.8% |
| 4 | 63.053 | 63.737 | +1.1% |
| 6 | 61.288 | 62.319 | +1.7% |

### 结论:尝试过,没有采纳为默认值

纯自旋在三个线程数下都略快,但幅度(0.8%~1.7%)和每组内部 5 次重复的
min-max 波动(通常也在 1%~2%)同一量级,不能确认是真实效果还是测量
噪声——和"平均等待时间远低于 yield 阈值"这个测量结果互相印证:按设计,
这台机器这个工作负载下 `sched_yield()` 几乎不会被触发,所以两种策略在
理论上就不该有系统性差异。按 STAGE6_PROMPT"没有提升的改动要回退"的要求,
默认行为保持不变(自旋后 yield,原因见 docs/stage3.md Q3:为长时间空闲
场景留一个不完全吃满 CPU 的退路),`MI_BARRIER_SPIN_ONLY` 作为一个可选
的实验开关保留在代码里,默认关闭。

## Q/K/V 与 gate/up 量化去重

### 做了什么

`graph.cpp` 的 `mi_graph_mul_mat` 增加一个条件:如果 `b`(激活操作数)
已经是 `MI_TYPE_Q8_0_ACT` 类型(说明调用方已经手动量化过),就不再插入
第二个量化节点,直接把 `b` 当成 `rhs` 用。`model_qwen2.cpp` 新增
`maybe_quantize_shared(ctx, x, like_weight)`:仅当 `like_weight->type ==
MI_TYPE_Q8_0` 时才把 `x` 量化一次并返回量化结果,否则原样返回 `x`
(F32/F16 权重完全不受影响,激活也不会被误量化)。`forward_layer` 里:

```cpp
mi_tensor* normed_q = maybe_quantize_shared(ctx, normed, lw.attn_q_w);
mi_tensor* q = linear(ctx, lw.attn_q_w, normed_q, lw.attn_q_b);
mi_tensor* k = linear(ctx, lw.attn_k_w, normed_q, lw.attn_k_b);
mi_tensor* v = linear(ctx, lw.attn_v_w, normed_q, lw.attn_v_b);
...
mi_tensor* normed2_q = maybe_quantize_shared(ctx, normed2, lw.ffn_gate_w);
mi_tensor* gate = linear(ctx, lw.ffn_gate_w, normed2_q, nullptr);
mi_tensor* up = linear(ctx, lw.ffn_up_w, normed2_q, nullptr);
```

`normed_q`/`normed2_q` 被 3 个/2 个不同的 mul_mat 节点各自读一次,
gallocr 已有的引用计数(`count_references`/`is_eligible_inplace_src`)
不需要任何改动就能正确处理"一个张量被多个消费者共享读取,直到最后一个
读完才能释放/复用它的 buffer"这个场景——这本来就是阶段 2b 就设计好并
测试过的通用能力(`tests/test_gallocr.cpp` 有专门的"两个消费者都读完
之前不能提前释放"的测试),不是这次新加的逻辑。

假设 q/k/v 三个权重(或 gate/up 两个权重)类型一致(这个项目里
`llama-quantize` 对同一个文件的所有线性层权重统一量化,norm/bias 除外,
在实践中总是成立)——只用其中一个权重的类型判断整组是否要共享量化。

### 效果

`quantize_q8_0` 节点数:每层 7 个降到 4 个(`normed` 一次、
`attn_concat` 一次、`normed2` 一次、`act` 一次),全模型
`--graph-stats` 实测 169→97(-42.6%,与步骤 0 调研时算出的理论值完全
一致)。`n_exec_nodes`(barrier 次数)701(比 3b 的 773 再少 72,
`24×3=72` 精确对应去掉的 3 个/层)。

| 线程数 | 步骤 3b decode tok/s | 去重后 decode tok/s | 变化 |
|---|---|---|---|
| 1 | 36.474 | 36.807 | +0.9% |
| 3 | 56.922 | 57.971 | +1.8% |
| 4 | 63.333 | 64.451 | +1.8% |
| 6 | 61.213 | 62.845 | +2.7% |

F32 vs HF、Q8_0 vs llama.cpp 两个对拍全部数字与之前完全一致(纯计算量
减少,不改变任何被保留下来的计算的浮点结果),10 个测试 Release/
Debug/Tsan 全部继续通过。

## 阶段性总结:decode tok/s 与 llama.cpp 的比值(截至本轮)

`scripts/bench_llamacpp_compare.py`(线程数改为 1/3/4/6,3-4 线程是这台
机器纯读带宽的甜点,见 `tools/bench_memread.cpp`),同一个 Q8_0 GGUF、
同样的 prompt/生成长度,`llama-bench -ngl 0`:

| 线程数 | mini-infer decode tok/s(中位数) | llama.cpp decode tok/s(均值) | 比值 |
|---|---|---|---|
| 1 | 37.033 | 36.783 | 1.007 |
| 3 | 58.375 | 67.669 | 0.863 |
| 4 | 64.466 | 72.525 | 0.889 |
| 6 | 62.789 | 70.431 | 0.891 |

原始数据见 `bench/results/llama_bench_compare.csv`。相比阶段 4/5 的基线
(1 线程 0.408、6 线程 0.443),1 线程比值从 0.408 升到 1.007(单线程
decode 已经不比 llama.cpp 慢,某些次测量甚至略快),6 线程比值从 0.443
升到 0.891。

decode tok/s 从阶段 0 到目前的完整过程(1/6 线程,Q8_0):

| 阶段 | 1 线程 | 6 线程 |
|---|---|---|
| 阶段 4/5 基线 | 15.407 | 33.139 |
| 步骤 1(F16C + 激活 float scale + 行级累加器) | 33.832 | 43.052 |
| 步骤 3a(无计算节点跳过 barrier) | 33.738 | 44.840 |
| 步骤 3b(注意力批量构图) | 36.474 | 61.213 |
| Q/K/V + gate/up 量化去重 | 36.807 | 62.845 |
| **合计倍数** | **2.39x** | **1.90x** |

## 下一步

以上(3a/3b/3c/量化去重)已全部完成并测量,等待确认后再决定是否进行
步骤 2(仅当 perf 显示非 mul_mat 算子占比 >5% 时才做的 rms_norm/rope/
softmax/silu 优化——目前 perf 数据里这些算子仍然远低于 5%,尚未重新
测过步骤 3 之后的 profile,如果要做步骤 2,建议先补一次 perf 确认
门槛条件是否仍然成立)。

## 收尾(步骤 2、4 不做,阶段 6 到此结束)

用户确认步骤 3 到此为止,不做步骤 2(展开/多行)和步骤 4(非 mul_mat
算子),以下是收尾的 5 项测量,全部数据来自真实运行,原始文件都在
`bench/results/`。

### 1. 同条件前后对比(git worktree)

`git worktree add <tmp目录> HEAD` 检出的是本次会话开始时的提交
(`c1cf2f7`,阶段 6 全部改动都还没发生——阶段 6 至今没有创建过任何新
commit,所有改动都在工作区里,所以"步骤 0 的提交"就是这个 HEAD 本身),
和当前工作区分别构建,用同一个基准脚本、同一台机器、同一次坐下测量
(数据见 `bench/results/opt_final_step0_vs_current.csv`)。"步骤 1" 一栏
是本轮会话中步骤 1 完成时刻的真实测量(存档在 `opt_step1.csv`,但当时
只测了 1/6 线程,3/4 线程没有单独复现——没有单独的 commit 能重新检出
"只有步骤 1" 的代码状态,重新构造需要手工回退一堆互相纠缠的改动,
风险大于收益,如实说明这个缺口而不是编数字补上)。

| 线程数 | step0(worktree 实测) | step1(会话内实测,仅 1/6) | step3+去重(当前 HEAD 实测) | llama.cpp(llama-bench) |
|---|---|---|---|---|
| 1 | 15.210 | 33.832 | 36.822 | 36.783 |
| 3 | 21.470 | -- | 58.195 | 67.669 |
| 4 | 26.381 | -- | 64.035 | 72.525 |
| 6 | 32.143 | 43.052 | 62.537 | 70.431 |

step0→当前 HEAD 的倍数:1 线程 2.42x,3 线程 2.71x,4 线程 2.43x,
6 线程 1.94x。

### 2. 同一轮重测:F16 SIMD vs Q8_0 AVX2(当前 HEAD)

数据见 `bench/results/opt_final_f16_vs_q8_0.csv`。**这次的 F16 数字比
阶段 4 报告的高不少(阶段 4:1 线程 17.236、6 线程 30.509;这次:1 线程
17.770、4 线程 38.952)——不是 F16 kernel 变了(阶段 6 完全没碰过
`mi_vec_dot_f16_f32`),是步骤 3a(跳过零计算节点的 barrier)和 3b
(注意力批量构图)对图结构的改动跟权重类型无关,F16 模型走同一张图,
同样受益。**

| 配置 | 1 线程 | 4 线程 |
|---|---|---|
| F16 AVX2+F16C | 17.770 | 38.952 |
| Q8_0 AVX2(阶段 6 后) | 36.469 | 63.877 |
| Q8_0/F16 比值 | 2.05x | 1.64x |

阶段 6 之前(阶段 4 数据)Q8_0 在 1 线程反而比 F16 慢(15.407 vs
17.236,0.89x)——阶段 6 把 Q8_0 自己的点积 kernel 修好之后,这个倒挂
消失了,现在 Q8_0 在两个线程数下都明显快于 F16,和"Q8_0 的价值不只是
文件体积"这个结论一致。

模型文件大小(`stat`,未变过,和阶段 0/4 记录的一致):

| 文件 | 字节数 |
|---|---|
| F32 | 1982079488 |
| F16 | 994157056 |
| Q8_0 | 531068416 |

### 3. gallocr 重测(图结构已变)

新增两项测量(`src/gallocr.h`/`.cpp` 的
`mi_gallocr_theoretical_min_bytes`,+ 已有的 `GALLOCR_NO_BESTFIT=1`
开关),四项一起写入 `bench/results/gallocr.txt`(Q8_0 模型,n_ctx=1024):

| case | peak(现有 arena) | no_bestfit(仅引用计数释放,不做块复用) | theoretical_min(理论下界) | baseline(逐张量独立分配) |
|---|---|---|---|---|
| prefill n=128 | 7.061 MB | 259.451 MB | 7.562 MB | 429.826 MB |
| prefill n=512 | 33.742 MB | 1541.805 MB | 30.250 MB | 2223.305 MB |
| prefill n=1024 | 123.484 MB | 4427.609 MB | 115.500 MB | 5790.609 MB |
| decode n_past=0 | 0.048 MB | 1.701 MB | 0.059 MB | 3.032 MB |
| decode n_past=1023 | 0.121 MB | 4.324 MB | 0.113 MB | 5.655 MB |

`theoretical_min`(任意时刻存活的非 view 张量字节数之和的最大值,纯
活跃区间分析,不管实际怎么分配)在 prefill n=1024/decode 两个场景下比
`peak` **更大**(123.5MB vs 115.5MB、0.121MB vs 0.113MB)——这不是
bug:`theoretical_min` 的定义明确不算原地复用(in-place elimination,
ADD/MUL/SILU/SCALE/ROPE/RMS_NORM 这类读了就地写回的算子),而现有的
`peak` 恰好从原地复用里受益(这两个场景下 `n_inplace_reuse=218` 次
生效),所以真实实现能在"纯活跃区间"这条理论线之下——原地复用是一项
额外的、这条纯活跃区间下界故意没有计入的技术。prefill n=128 场景下
反过来(peak 7.061MB < theoretical_min 7.562MB),同一个原因,只是这次
原地复用相对图规模的收益更明显。`no_bestfit` 全面比 `baseline` 小
(仍然受益于原地复用),又全面比 `peak` 大(失去了跨张量的块复用)——
三者的大小关系在所有 5 个场景下都成立,符合预期。

### 4. prefill 是否算了全部位置的 logits——是,已修复

`build_model_graph` 的 lm_head 投影(`[896,151936]` 的绑定权重矩阵,
这个模型迄今最大的一次矩阵乘)原来对 prefill 的每一个位置都算了一遍
logits,但正常生成(和 `--bench`)只会从最后一个位置采样。新增
`mi_qwen2_forward_params::last_logits_only`(默认 `false`,教师强制
对拍的 `--dump` 路径和所有现有测试都不受影响),打开后用一个免费的
view 把 lm_head 的输入限制到最后一个 token。`main.cpp` 的真实生成
循环和 `--bench` 都打开这个开关;`--dump`/`--gallocr-stats`/
`--graph-stats` 保持关闭。

| prefill 长度 | 优化前 | 优化后 | 提速 |
|---|---|---|---|
| 128 | 95.724 | 126.215 | 1.32x |
| 512 | 64.315 | 77.041 | 1.20x |
| 1024 | 41.734 | 46.855 | 1.12x |

提速比例随 prefill 变长而缩小——不是收益递减的 bug,是因为"对每个位置都算
logits"的代价和模型主体(24 层,尤其是 O(n_tokens²) 的注意力)的代价都随
`n_tokens` 线性/超线性增长,在 n_tokens=1024 时,全量 logits 只是一个已经
更大的总量里相对更小的一块。正确性:`tests/test_model_qwen2.cpp` 新增
`test_last_logits_only_matches_full`,验证开关打开后的输出和关闭时最后
一行逐位一致(同一个矩阵乘,只是通过 view 把输入限制到一列,不是重新
实现了一遍);`scripts/compare.py`(F32 vs HF,走 `--dump`,不受这个开关
影响)`PASS: True` 保持不变。原始数据与更详细的说明见
`bench/results/opt_final_prefill_logits.txt`。

### 收尾对拍与测试

`scripts/compare.py`(F32 vs HF)`PASS: True`,`scripts/compare_q8_llamacpp.py`
数字与本阶段每一步都完全一致;10 个测试在 Release/Debug/Tsan 三种配置下
全部通过(新增：`test_last_logits_only_matches_full`)。

### 阶段 6 完整总结表(decode,Q8_0,与 llama.cpp 对比)

| 线程数 | 阶段 4/5 基线 | 步骤 1 | 步骤 3a | 步骤 3b | 量化去重(最终) | llama.cpp | 最终比值 |
|---|---|---|---|---|---|---|---|
| 1 | 15.407 | 33.832 | 33.738 | 36.474 | 36.807 | 36.783 | 1.001x |
| 3 | -- | -- | -- | 56.922 | 58.195 | 67.669 | 0.860x |
| 4 | -- | -- | 46.547 | 63.333 | 64.035 | 72.525 | 0.883x |
| 6 | 33.139 | 43.052 | 44.840 | 61.213 | 62.537 | 70.431 | 0.888x |

单线程 decode 从 llama.cpp 的 41% 升到 100%(1.001x,两边已经打平);
6 线程从 44% 升到 89%。纯读带宽上限(`tools/bench_memread.cpp`,AVX2
顺序读 + 求和,不是 memcpy——memcpy 同时产生读和写两份总线流量,对这个
几乎纯读的负载会低估真正的上限)在这台机器上 1 线程 27.8 GB/s,峰值在
3-4 线程(41-43 GB/s),之后线程越多反而越慢;decode 有效带宽(当前
HEAD)1 线程约 19.6 GB/s、4 线程约 34.2 GB/s——已经比较接近对应线程数的
读带宽上限,继续加线程数不会有明显收益,这也是为什么最终基准把线程数
定在 1/3/4/6 而不是一路测到 12。
