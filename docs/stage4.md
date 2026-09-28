# 阶段 4:F16 + Q8_0 + AVX2

## 做了什么

- `src/quant.h/.cpp`:F16↔F32 标量转换(标准 IEEE 754 binary16 位操作,
  在 `tests/test_quant.cpp` 里和硬件 F16C(`_mm_cvtph_ps`/`_mm_cvtps_ph`)
  交叉验证过);`block_q8_0`(1 个 fp16 缩放因子 + 32 个 int8,34 字节,
  字段顺序与 llama.cpp 一致);`mi_quantize_row_q8_0`/
  `mi_dequantize_row_q8_0`;`mi_vec_dot_q8_0_q8_0`/`mi_vec_dot_f16_f32`
  的标量参考版 + 运行时分发(`__builtin_cpu_supports`,`MI_QUANT_FORCE_SCALAR`
  环境变量可强制走标量,供 `scripts/bench_quant.py` 在同一台机器上量出
  "标量版" 数据点)。
- `src/quant_avx2.cpp`(新文件,每个函数单独 `__attribute__((target(...)))`,
  这个编译单元本身不开任何全局 SIMD flag):Q8_0 点积按 PROMPTS.md 指定的
  技术实现(`_mm256_sign_epi8` 处理符号、`_mm256_maddubs_epi16` +
  `_mm256_madd_epi16` 累加到 int32、每个 block 的浮点部分用 FMA 累加);
  F16 点积用 F16C 的 `_mm256_cvtph_ps` 批量转换 8 个半精度值,配合
  `_mm256_fmadd_ps` 做向量化 FMA。
- `src/tensor.h`:新增 `MI_OP_QUANTIZE_Q8_0`。`src/ops_cpu.h/.cpp`:新增
  `mi_compute_quantize_q8_0` kernel;`mi_compute_mul_mat` 按
  `src[0]->type`(F32/F16/Q8_0)三路分派,F32 分支与阶段 1-3 逐字节不变;
  `mi_compute_get_rows` 按 `src0->type` 增加 F16/Q8_0 反量化分支,F32 分支
  同样不变。
- `src/graph.h/.cpp`:新增 `mi_graph_quantize_q8_0` builder;
  `mi_graph_mul_mat` 在 `a->type == MI_TYPE_Q8_0` 时自动为 `b` 插入一个
  量化节点——量化后的激活值就是一个普通的图节点,gallocr 像管理其他中间
  张量一样管理它的 buffer,调用方(`model_qwen2.cpp` 的 `linear()`)完全
  不用关心权重是什么类型。
- `src/model_qwen2.cpp`:`reserve_gallocr` 的 dummy 权重改为镜像真实模型
  每个张量的实际类型(而不是硬编码 F32)——原因和踩过的坑见下文;
  `estimate_node_capacity` 按每层最多 7 个 mul_mat 各带一个量化节点的
  最坏情况上调;`mi_kv_cache_init` 签名新增 `const mi_qwen2_weights&`
  参数。`load_weight` 去掉了阶段 1 遗留的 "只认 F32" 检查。
- `tests/test_quant.cpp`:block_q8_0 布局、F16 已知值 + 硬件 F16C 交叉验证、
  量化/反量化往返精度、Q8_0 int32 部分标量与 AVX2 完全一致(随机数据 +
  ±127 边界值 + 全零 block)、两个 vec_dot 的最终浮点值标量与 AVX2 相对
  误差 < 1e-5。
- `tests/fixtures/tiny_qwen2_q8_0.gguf`(`scripts/make_tiny_qwen2_q8_0_fixture.py`,
  用 `gguf-py` 自带的参考量化器而不是本项目自己的量化代码生成,
  维度取 n_embd=32/head_dim=32/n_ff=64 以满足 Q8_0 的 32 元素分块对齐)+
  `tests/test_model_qwen2.cpp` 的三个新测试(hparams 类型检查、prefill 有限性、
  批量 vs 逐 token 增量 decode 逐位一致)。
- `tools/llama_logits.cpp` + `tools/CMakeLists.txt`:见下方"构建方式"。
- `scripts/compare_q8_llamacpp.py`:真实 Q8_0 GGUF 上,416 个 token 位置的
  teacher forcing 对拍,结果见 `bench/results/accuracy_q8_vs_llamacpp.txt`。
- `scripts/bench_quant.py`:F32 / F16 标量 / F16 AVX2+F16C / Q8_0 标量 /
  Q8_0 AVX2 五组,线程数 1 和 6(阶段 3 测得的 decode 最快线程数),结果见
  `bench/results/quant_{prefill,decode}.csv`。
- `bench/results/model_sizes.txt`:F32/F16/Q8_0 三个文件的真实 `stat()`
  大小(与 `docs/models.md` 阶段 0 记录的数字一致,文件未变过)。

## 为什么这样设计

**量化激活是图节点,不是 kernel 内部的临时 buffer。** `tensor.h` 的
`extra` 字段注释里早先写过"预留给量化激活的 per-thread buffer",但真正
写到这一步才发现那条路走不通:量化后 buffer 的大小随 `n_tokens`(prefill
批量、decode 单 token)每步变化,如果放在 `mi_compute_mul_mat` 内部临时
分配,要么违反"运行期零分配"(阶段 2b 的 `test_zero_alloc` 会失败),
要么需要自己再实现一套"按最坏情况预分配 + 复用"的逻辑——而这套逻辑
gallocr 已经有了。把量化做成一个真正的 `MI_OP_QUANTIZE_Q8_0` 图节点,
插在 `mi_graph_mul_mat` 内部,直接复用阶段 2b 的 reserve/plan 机制:
量化输出该多大、该不该被后面的节点原地复用、什么时候可以释放,gallocr
按处理其他任何张量的方式处理它,不需要新写一行分配逻辑。这是本阶段
"没有专门代码就自动工作"的部分,也是当初把 gallocr 设计得足够通用
(按 `t->op` 和 `mi_tensor_is_persistent` 泛化处理,不针对具体算子硬编码)
换来的红利。

**F16 走 kernel 级分支,Q8_0 走图级插入——这不是随意的不一致,是两者
本质不同决定的。** F16 mul_mat 从不重新量化激活值:权重是 F16,激活值
本来就是 F32,点积内部把权重的每个 F16 元素转成 F32 现算,没有中间格式
需要管理,所以纯粹是"kernel 内部换一种读权重的方式",不涉及图结构。
Q8_0 mul_mat 则要求*两个*操作数都是 Q8_0(AVX2 的整数点积指令要求两边
都是 int8),激活值必须先被转换成一种新的、带自己的存储和生命周期的
张量,这就是图结构层面的事情了。

**为什么 mul_mat 的 F32 分支要保持逐字节不变,而不是把三种类型合并成
一个更"优雅"的模板/循环。** 阶段 1-3 建立起来的所有 bit-exact 测试
(HF 对拍、图版本与直接调用版对拍、gallocr 四种复用组合对拍、线程池
多线程数对拍)全部基于 F32 路径当时的精确浮点运算顺序。合并实现哪怕
在数学上等价,只要循环展开方式、临时变量的引入方式发生变化,编译器
生成的浮点运算顺序就可能变(尤其是 `-O3` 下),之前的 bit-exact 断言
就可能失败又找不到"真正原因"。所以这里选择了看起来"重复"的写法:
F32 分支原封不动地摘录过来,F16/Q8_0 是全新增加的分支,而不是让三者
共享一段"更通用"的代码。

## 踩过的坑

**`reserve_gallocr` 的 dummy 权重最初硬编码成 F32,而真实模型可能是
Q8_0——这会让 arena 按错误的图形状来定。** `mi_gallocr_reserve` 的
"最坏情况"图是用一批不会被解引用的 dummy 权重张量搭出来的(阶段 2b
就是这样设计的,目的是不用等真实权重加载完就能规划内存)。阶段 4 之前
dummy 权重永远是 F32,这在阶段 4 之前没问题,因为当时只有 F32 一种情况。
但阶段 4 让 `mi_graph_mul_mat` 的行为本身依赖权重类型(Q8_0 时会多插入
一个量化节点),如果 dummy 权重继续硬编码成 F32、而真实模型其实是
Q8_0,`reserve_gallocr` 排练出来的图会比真实每一步实际要跑的图*少*
一些节点、也漏算了每个量化节点自己的 buffer 大小——arena 会按一个从未
真正发生过的、更小的图来定容量。这不会在小测试里必然崩溃(gallocr 的
"运行时超出预留容量就 abort" 检查会在某一步真的用到 Q8_0 时抓到它,但
那已经是运行期的 `abort()`,而不是加载模型时就能发现的问题)。修复
方式是让 `make_dummy_weight`/`reserve_gallocr` 从真实的
`mi_qwen2_weights`(已经加载好、类型已知,数据指针不解引用)读取每个
张量的实际类型,dummy 权重只复制类型和形状,不复制数据——这样排练出的
图形状和真实前向传播会构建的图完全一致,不管模型是 F32、F16 还是 Q8_0。
这个 bug 是在写这段代码时通过重新审视 stage 2b 的假设主动发现并修掉的,
不是运行时崩溃后倒推出来的——但如果没有想到这一层,最可能的症状会是
"模型第一次 decode 到某个刚好触发 Q8_0 路径的 step 时莫名其妙 abort",
非常难定位。

**阶段 1 遗留的 "只支持 F32 GGUF" 检查,忘了在阶段 4 删掉,第一次真实
模型冒烟测试直接被拦下。** `load_weight()` 里有一条从阶段 1 就在的检查:
非 F32 类型的权重直接报错退出。这条检查在阶段 1-3 是对的(那时候确实
只支持 F32),但阶段 4 已经让所有 kernel 支持 F16/Q8_0 分派之后,这条
检查变成了一个"忘记移除的旧约束",导致第一次拿真实 Q8_0 GGUF 冒烟测试
时,还没跑到任何新写的 kernel 代码,模型加载阶段就直接报错退出。这是
个提醒:引入"阶段限制"注释/检查的时候,最好在检查旁边写清楚"这条限制
在哪个阶段解除",而不是只写"当前阶段不支持"——否则很容易在真正解除
限制的那个阶段忘记删除对应的检查代码。

## tools/llama_logits 构建方式

`tools/llama_logits.cpp` 是本项目里唯一允许链接 `third_party/llama.cpp`
的地方(和同机 `mini-llama.cpp` 项目的 `tools/ref_dump` 同一条约定)。
本项目自己没有 `third_party/`,直接复用 `mini-llama.cpp` 已经构建好的
llama.cpp checkout(用户在项目开工前已确认可以这样复用)。

`tools/CMakeLists.txt` 里:

```cmake
set(MI_LLAMA_CPP_ROOT
    "${CMAKE_SOURCE_DIR}/../mini-llama.cpp/third_party/llama.cpp"
    CACHE PATH "...")
```

正常执行 `cmake -S . -B build && cmake --build build` 就会自动探测到
这个路径(本机确实存在该 checkout)并构建出 `build/tools/llama_logits`;
如果在没有这个 sibling checkout 的机器上构建,`tools/CMakeLists.txt`
会打印一条 `message(STATUS ...)` 说明跳过原因,其余所有 target 正常
构建,不受影响。要在别的机器上指到别的 llama.cpp 构建目录,传
`-DMI_LLAMA_CPP_ROOT=/path/to/llama.cpp`(要求该目录下有
`include/llama.h` 和 `build/bin/*.so`,即先用
`cmake -DGGML_NATIVE=ON && cmake --build` 构建过一遍)。

用法:`llama_logits <model.gguf> <ids_file> <out_prefix>`,只用
`llama_model_load_from_file` + `llama_init_from_model` +
`llama_batch_init`(手动把每个位置的 `logits[i]` 设成 1,不用
`llama_batch_get_one`,后者默认只要最后一个位置的 logits)+
`llama_decode` + `llama_get_logits_ith`,不链接 `common`(不需要它的
tokenizer/sampler,因为输入已经是 token id 而不是文本,和
`mini_infer --prompt-ids` 同样的理由)。输出格式
(`<prefix>.bin` + `<prefix>.shape`)和 `main.cpp` 的 `write_dump_tensor`
完全一致,`scripts/compare_q8_llamacpp.py` 两边用同一个加载函数读。

## 对拍结果(scripts/compare_q8_llamacpp.py)

见 `bench/results/accuracy_q8_vs_llamacpp.txt`。摘要:

| 指标 | 数值 |
|---|---|
| token 序列长度 | 416(Qwen2.5 tokenizer 编码一段固定英文描述并重复到超过 256) |
| top-1 一致率 | 97.12%(404/416) |
| 余弦相似度均值 | 0.999511 |
| 余弦相似度最小值 | 0.998201 |
| 最大绝对误差 | 0.847039 |
| 平均绝对误差 | 0.079325 |

不要求逐位一致(SPEC.md 对 Q8_0 vs llama.cpp 的标准本来就不是 bit-exact),
top-1 不是 100% 属于预期:Q8_0 量化噪声偶尔会让两个概率非常接近的候选
词互换名次,这不代表实现有 bug。

## 模型文件大小(bench/results/model_sizes.txt)

| 文件 | 大小(字节) |
|---|---|
| F32 | 1982079488 |
| F16 | 994157056 |
| Q8_0 | 531068416 |

## 基准结果(scripts/bench_quant.py)

真实机器同阶段 3(AMD Ryzen 5 5600H,6 物理核)。测试参数:prompt 长度
16,decode 长度 8(7 次实际 decode forward 调用),n_ctx=64,每组 1 次
预热 + 5 次重复。完整原始数据见 `bench/results/quant_{prefill,decode}.csv`;
下表是中位数(min–max 全部在中位数 ±2% 以内):

**prefill tok/s**

| config | n_threads=1 | n_threads=6 |
|---|---|---|
| F32 | 2.788 | 14.320 |
| F16_scalar | 0.693 | 2.463 |
| F16_avx2 | 19.404 | 77.237 |
| Q8_0_scalar | 11.356 | 31.103 |
| Q8_0_avx2 | 16.113 | 44.740 |

**decode tok/s**

| config | n_threads=1 | n_threads=6 |
|---|---|---|
| F32 | 2.795 | 12.781 |
| F16_scalar | 0.691 | 2.417 |
| F16_avx2 | 17.236 | 30.509 |
| Q8_0_scalar | 11.051 | 24.886 |
| Q8_0_avx2 | 15.418 | 32.984 |

两个直接从数据本身能读出、不需要额外假设的现象:

- **F16_scalar 比 F32 还慢(单线程 prefill 0.69 vs 2.79 tok/s)。**
  `mi_vec_dot_f16_f32_scalar` 每个元素都要调用一次
  `mi_fp16_to_fp32`(位操作 + 可能的 subnormal 归一化循环 + 函数调用
  开销),而 F32 路径每个元素只有一次乘加;F32 分支从阶段 1 起就没有
  为任何一种量化格式付出额外的逐元素转换成本,所以"F16 标量版"在这台
  机器上比 F32 直接算还慢,这符合"F16 标量版本身没有性能意义、只是
  SIMD 版本的正确性对照和公平基线"这一定位(PROMPTS.md 把它称为
  "作为量化的公平基线",不是要证明它比 F32 快)。
- **F16_avx2 在两个线程数下都比 Q8_0_avx2 快(单线程 prefill 19.4 vs
  16.1 tok/s)。** 这台机器上 F16C 的 `_mm256_cvtph_ps`
  是一条单指令把 8 个半精度值转成 8 个 float,之后就是一次普通的
  `_mm256_fmadd_ps`;Q8_0 的 AVX2 点积除了 `_mm256_maddubs_epi16` +
  `_mm256_madd_epi16` 这两条整数指令外,每 32 个元素(一个 block)还要
  多读一次两个 fp16 缩放因子、做一次标量乘法和一次标量 FMA——这些额外的
  逐 block 标量开销在小到中等矩阵规模下没有被 block 内的整数计算摊薄
  掉。这不代表 Q8_0 在这个项目里"不值得用"(Q8_0 的价值主要在文件体积
  和内存带宽,806MB→531MB,而不是这台机器上这个矩阵规模下的纯计算
  吞吐),只是如实记录这台机器、这个实现在这个规模下测出来的数字。

## 面试官可能追问的 3 个问题

**Q1:为什么 Q8_0 的激活量化必须做成图节点,而不能像很多推理引擎那样
用一个线程本地的临时 buffer,用完就地复用?**

因为这个项目从阶段 2b 起就有一条硬性验收标准:decode 步的稳态运行期
零堆分配(`tests/test_zero_alloc.cpp` 用 `--wrap=malloc,free` 直接检查)。
一个"线程本地临时 buffer"要么在第一次用到时 `malloc` 一次(之后复用没问题,
但第一次分配已经违反"运行期"零分配的定义,除非能确保它发生在 reserve
阶段而不是真正的 decode 循环里),要么每步都要按当前 `n_tokens` 重新
分配/判断是否够大——这两条路都是在给 gallocr 已经解决过的问题重新发明
一遍。做成图节点后,这个 buffer 的生命周期规划完全交给 reserve 阶段的
两个最坏情况图(prefill n_tokens=n_ctx、decode n_past=n_ctx-1)处理,
和其他任何中间张量没有区别。

**Q2:Q8_0 的 AVX2 点积你选择让标量和 AVX2 版本在"最终浮点结果"上也
尽量一致(而不只是官方要求的 < 1e-5 相对误差),这是怎么做到的,为什么
F16 的点积做不到同样的事?**

Q8_0 的每个 32 元素 block 只贡献*一个*标量浮点数(`isum * dx * dy`,
`isum` 是整数、不受累加顺序影响,两个 fp16 缩放因子的乘法也是逐 block
独立算的),而块与块之间的累加,标量版和 AVX2 版都选择了同一种方式——
逐 block 调用 `std::fma`/`__builtin_fmaf`,顺序也相同——所以理论上除了
"标量路径调用的是 C++ 标准库的 fma、AVX2 路径调用的是编译器内建的
`__builtin_fmaf`"这类实现细节差异外,没有引入任何新的舍入路径,实测下来
两者确实是 bit-exact(见 `tests/test_quant.cpp`)。F16 做不到同样的事,
是因为 F16 点积的"整数部分"不存在——它从一开始就是浮点乘加,AVX2 版本
一次处理 8 个元素、用向量寄存器内部的树状规约做累加,标量版本是严格的
逐元素顺序累加,这是两种本质不同的浮点求和顺序,IEEE 754 加法不满足
结合律,所以只能退回到 PROMPTS.md 本来就允许的 "< 1e-5 相对误差"
这个更宽松的标准。

**Q3:你在 `mi_compute_mul_mat` 里给 F32/F16/Q8_0 写了三个几乎独立的
分支,而不是抽象出一个"读一个权重元素"的通用接口套进同一套循环——
这不会增加维护成本吗?**

会增加一点样板代码,但这是有意的取舍:F32 分支必须和阶段 1 起产生的
浮点运算顺序逐字节相同(全项目 bit-exact 测试链条的基础),任何"抽象出
通用读取接口"的重构,哪怕数学上等价,只要改变了循环嵌套方式或编译器
看到的表达式形状,就可能悄悄改变 `-O3` 下的浮点运算顺序,让一批历史
测试失败、还很难定位到底是不是这次重构引入的真实 bug。三个分支各自
独立、其中两个(F16/Q8_0)是全新代码、一个(F32)是原样保留,是用"少量
重复"换"每次改动的影响范围可控"——这和 stage 4 开头决定"F32 分支必须
保持不变"是同一个设计立场的两次体现。
