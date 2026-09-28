# 阶段 2b:gallocr(核心)

分工说明:PROMPTS.md 里阶段 2b 默认的分工是分配逻辑由用户自己实现,我只负责
接口/预演/测试/poison。这一次我明确回复"由你全部实现",所以接口、预演、
分配逻辑(best-fit、原地复用、view 共享)、测试、poison、统计接口全部由我
一人完成。

## 做了什么

- `src/gallocr.cpp`:一块 `malloc` 一次的 arena + 空闲块链表(`std::vector`,
  在 `reserve()` 时就把容量 `reserve()` 好,之后的 `alloc_graph()` 绝不
  触发它扩容)、best-fit 选块、释放时与相邻空闲块合并、原地复用(6 个条件
  全部检查,含一个只有实测才发现的隐藏条件,见下)、poison 调试模式、
  `GALLOCR_NO_INPLACE`/`GALLOCR_NO_BESTFIT` 两个调试开关。
- `src/context.cpp` 新增 `mi_context_new_tensor_eager`,并且把
  `mi_context_view` 改成"view_offs 用算术计算、data 指针延迟解析"——这是
  阶段 2b 必须做的改动,原因见下。
- `src/model_qwen2.cpp`:`mi_kv_cache` 新增 `galloc`(gallocr 实例)和
  `graph_ctx`(每步复用、只 reset 不重新 `mi_context_new` 的图构建
  context)。`mi_kv_cache_init` 里做一次性的 `reserve_gallocr`:用假权重
  (只要形状对,数值从不会被读到,因为 reserve 只做预演,从不执行 kernel)
  搭出 prefill(`n_tokens=n_ctx`)和 decode(`n_past=n_ctx-1`)两个最坏情况
  的图,喂给 `mi_gallocr_reserve`。`mi_qwen2_forward` 从直接 new/free 一个
  scratch context,改成 `mi_context_reset(kv->graph_ctx)` + 调用
  `mi_gallocr_alloc_graph`。
- `logits` 从"普通中间张量"改成"eager 分配、打上 `MI_TENSOR_FLAG_OUTPUT`
  标记的持久张量"——原因和别的 checkpoint(`embd`/`layer_i`/`output_norm`)
  不一样,见下。
- 测试:`tests/test_gallocr.cpp`(view 链、view 后原地、多消费者、图输出
  不被复用四个必测场景,外加一个 stats 合理性检查)、
  `tests/test_zero_alloc.cpp`(全局 `operator new/delete` 重载 +
  `-Wl,--wrap=malloc,--wrap=free`,reserve 之后 64 个 decode step 计数
  为 0)、`tests/test_graph.cpp`/`tests/test_model_qwen2.cpp` 复用阶段 2a
  的 golden dump 在 `GALLOCR_NO_INPLACE`/`GALLOCR_NO_BESTFIT` 四种组合 +
  `GALLOCR_POISON=1` 下全部逐位一致。
- `bench/results/gallocr.txt`(见 `scripts/bench_gallocr.py`):真实
  Qwen2.5-0.5B F32 模型上,prefill n_tokens=128/512/1024、decode
  n_past=0 和 n_ctx-1 五个点的峰值 vs 基线实测数据。

## 实现过程中发现的三个非显而易见的问题

这三个问题都不是"看 SPEC 就能想到"的,是写测试、跑测试的过程中暴露出来的,
记录下来是因为它们比"代码怎么写"更有价值。

### 1. mul_mat 不能进原地复用白名单——即使它满足另外 5 个条件

SPEC.md 第 3 条列的 6 个条件里,"引用计数为 1"和"非视图"等都是对**输入
张量**的检查,唯独没有直接说"哪些算子允许原地"——只在括号里给了个例子
"add、mul、silu、scale、rope 等"。我一开始的实现只检查了那 5 个条件,
没有单独检查算子类型,结果 `attn_proj = mul_mat(attn_output_w, attn_concat)`
这一步:`attn_concat` 非视图、非持久、类型大小都对、而且引用计数确实是
1(只有这一个 mul_mat 读它)——**看起来完全满足条件**,于是被选中原地复用。
但 mul_mat 不是逐元素算子:`dst[i,j]` 依赖 `attn_concat` 整个第 j 列
(所有 t 的 `attn_concat[t,j]`),而不只是 `attn_concat[i,j]`。如果 dst
和 attn_concat 共享同一块内存,算完 `dst[0,j]` 就会覆盖
`attn_concat[0,j]`,而计算 `dst[5,j]` 时还需要读原始的
`attn_concat[0,j]`——数据已经被自己覆盖了。

解决方式:加一张显式白名单(`op_supports_inplace`),只有确认过"每个输出
元素只读输入的同一个位置、算完立刻写回、不依赖别的位置"的算子
(add/mul/silu/scale/rope,以及审查过确实安全的 rms_norm)才允许原地,
mul_mat/get_rows/softmax 一律不进白名单(softmax 在我这份实现里其实也是
安全的,但为了严格对应 SPEC 举的例子,保守地不加进去)。这是
`tests/test_gallocr.cpp` 里 `test_inplace_not_through_view` 间接覆盖的
场景之一,但这个 bug 本身是在**真实模型**跑 golden dump 对拍时,靠
"结果对不上"发现的,不是靠某条单元测试——单元测试是发现问题**之后**为了
钉住它才补的。

### 2. "引用计数为 1"必须是原始总数,不能是递减中的剩余数

第二个更隐蔽:如果一个张量被两个消费者读(比如同一个 RMSNorm 输出被两个
不同的下游算子各用一次),处理第一个消费者时引用计数还是 2(不满足"==1",
正确地不做原地);但处理第二个(也是最后一个)消费者时,如果引用计数已经
在处理第一个消费者时被递减过一次,此刻读到的就是 1——和"从一开始就只有
一个消费者"完全看不出区别,于是会被错误地判定为满足条件、原地复用。

这次是靠 `tests/test_gallocr.cpp` 的 `test_multiple_consumers_survive_to_
last_reader` 直接测出来的(构造两个消费者共享一个输入,断言两次都不能
原地)。修复方式:把"总引用计数"(`n_children`,预演阶段算出来之后不再
改动)和"还剩几个未处理的引用"(借用 `mi_tensor::extra` 这个保留字段,
在分配阶段递减)彻底分成两个字段。原地复用的判断只看前者。

这里还想额外说明一点:仔细分析后,"作为多个消费者里最后一个的时候原地
复用"在**内存安全**上其实并不出错(前面的消费者已经读完了,不会有人再
读旧值)——SPEC.md 写的是"唯一消费者",按字面理解就是"从来只有一个",
不是"恰好是最后一个"。我选择按字面(更保守)的理解实现,而不是自己判断
"反正安全所以放宽",这是执行"不自行放宽标准"的一个具体例子,虽然这次
不是对拍标准而是内部实现细节。

### 3. view 的 data 指针不能在构造时就解析

阶段 1/2a 里,每个张量一造出来就立刻有真实内存(`ctx->no_alloc=false`),
`mi_context_view` 在构造 view 的那一刻用 `src->data + offset` 算出
`data` 完全没问题。阶段 2b 把中间张量的 `data` 交给 gallocr 事后决定
(`ctx->no_alloc=true`),而 view 是在**图搭建阶段**(所有 `mi_graph_*`
调用发生时)构造的,那时候它指向的节点(比如 `qh_rot`)可能还没被 gallocr
分配——`src->data` 还是空指针,`src->data + offset` 直接算出一个无效地址。

解决方式:把 `view_offs` 的计算方式从"指针相减"改成"纯算术"(父级自己的
`view_offs` 加上这次的 `offset`,不管父级 `data` 是不是空),`data` 则是
"能算就算,算不出来就先空着",gallocr 在分配阶段(拓扑序走到这个 view
节点时,它的 `view_src` 一定已经被安排好,因为图是按依赖顺序执行的)
补上这一步。这是我在写 gallocr 分配逻辑、跑第一次全模型测试时,进程直接
崩在空指针解引用上才发现的——修复之后再也没出现过,说明这类"提前解析
尚不存在的地址"的 bug,靠跑一次真实模型比靠读代码更容易发现。

## SPEC.md 第 3 条六个条件,分别举一个违反后出错的例子

- **引用计数为 1**:违反 → 上面"问题 2"的场景,两个消费者共享一个输入,
  如果都按"原地"处理,第二个消费者的原地写会在**第一个消费者还没读到它
  之前**就覆盖数据(如果执行顺序反过来,或者两个消费者被同一个线程池
  节点并行处理——虽然本项目要到阶段 3 才有线程池,但"多消费者不能原地"
  这条本来就是为将来的并行执行做的保险)。
- **非视图**:违反 → 上面"问题 1"的 `attn_concat`/mul_mat 场景,或者更
  直接地:如果允许一个 view(比如某个 head 的切片)被原地复用,写入的
  内容只覆盖了 view 覆盖的那一小块字节,但 gallocr 的释放逻辑是按**整个
  view_src** 的大小记账的——会造成"半头被原地覆盖、另外几个头的切片没被
  覆盖、但 gallocr 却认为整个 view_src 已经被这次原地复用接管"的记账
  错乱,后续对 view_src 别的部分的释放/复用判断全部失真。
- **没有仍存活的视图指向它**:违反 → 假设 `qh_rot` 已经被两个 head 的
  view 引用过一个(另一个还没被处理),如果这时候因为"看起来引用计数够低"
  就把 `qh_rot` 原地复用给别的节点,还没处理到的那个 head 的 view 会读到
  被覆盖的错误数据。本项目的引用计数设计(视图的引用记到 view_src 头上)
  已经把这一条并入"引用计数为 1"里了,不需要单独判断,但如果记账方式
  搞反(比如引用记到视图自己头上而不是 view_src),这条就会独立违反。
- **非持久张量**:违反 → 如果允许复用一个权重张量(比如
  `attn_output_w`)的"内存",下一层甚至下一次 forward 调用读到的权重就
  是别的中间结果,模型直接输出垃圾。
- **非图输出**:违反 → 见下面"为什么 logits 走 eager+OUTPUT"那一段,
  `logits` 如果被当成普通中间张量参与 arena 管理,可能在
  `mi_graph_compute` 跑完、调用方还没来得及 `memcpy` 出去之前,就被同一个
  arena 里另一个不相关的分配覆盖(虽然本项目 `logits` 天然引用计数为 0、
  不会被自动释放,但如果以后哪个改动不小心让某个节点把 `logits` 当成自己
  的输入,这个洞就会立刻暴露)。
- **类型与大小相同**:违反 → 如果允许一个 F32→Q8_0(阶段 4 之后)或者
  形状不同的原地复用,写入的字节数和布局对不上,轻则越界写坏相邻内存,
  重则直接段错误。

## 为什么 `logits` 走 eager 分配 + `OUTPUT` 标记,而 `embd`/`layer_i`/
`output_norm` 不用

`logits` 在**每一次** `mi_qwen2_forward` 调用里都会被 `memcpy` 到调用方
的缓冲区,不管有没有开 `--dump`——所以它必须一直存活到 `mi_graph_compute`
完全跑完之后,不能被 gallocr 当成普通中间张量随意复用/释放。做法是:
构造完 `logits` 节点后,手动给它一块 eager 分配的内存(`mi_context_
alloc_bytes`,不经过 gallocr)并打上 `MI_TENSOR_FLAG_OUTPUT`,这样
`mi_tensor_is_persistent(logits)` 为真,gallocr 的分配循环在处理这个
节点时直接跳过它(既不分配也不参与引用计数/释放),保证它的地址从构造
那一刻起就不会再变。

`embd`/`layer_i`/`output_norm` 只有在 `--dump` 打开时才需要在整张图跑完
之后还能读到——这是调试/对拍专用的需求,不是每次调用都要付出的代价。
如果也把它们都做成 eager+OUTPUT,平时(不 dump 的)推理路径就要为 24+2
个永不释放的缓冲区多付内存,而这些内存在正常推理里本来是可以被后面的层
复用掉的。所以选择了另一种做法:dump 打开时,不再把整张图一次性丢给
`mi_graph_compute`,而是自己在 `mi_qwen2_forward` 里逐节点调用
`mi_compute_forward`,每算完一个 checkpoint 张量就立刻回调 dump sink
把数据拷出去(`main.cpp` 的 `write_dump_tensor` 立刻把字节写进文件)——
gallocr 之后完全可以复用这块内存,不影响正确性,因为需要的数据已经在
被覆盖之前抄走了。这是"两种 checkpoint 需求本质不同(每次都要 vs 仅
调试时需要)所以给两种不同实现"的一个具体例子,而不是为了图省事把所有
checkpoint 都用同一种(更贵的)方式处理。

## 为什么 reserve 和每步规划要分开

如果每一步都重新"预演 + 分配",预演阶段要重新算一遍每个张量的大小、
引用计数、最佳匹配决策——这本身不产生堆分配(都是对已有 `mi_tensor`
字段的读写),但**分配 arena 本身**(`malloc`)如果每步都做一次,就完全
违背了"运行期零动态分配"的目标,decode 阶段每生成一个 token 都要
`malloc`/`free` 一大块内存,不仅慢,也让"是否发生了不该有的堆分配"这件事
没法用一个简单的计数器去验证。

拆成两步之后:
- `reserve()` 只在会话开始时跑一次,用两张**最坏情况**的图(prefill
  取 `n_tokens=n_ctx`,decode 取 `n_past=n_ctx-1`——两者的 `n_kv_total`
  都等于 `n_ctx`,分别对应"批量算完整个上下文"和"上下文快满时再生成一个
  token"这两种真实会发生的最坏内存占用场景)算出来的峰值的较大者,一次性
  `malloc`。
- 之后每一步只做"预演 + 分配"里"分配"的部分,而且是在**同一块**已经
  `malloc` 好的内存上重新计算偏移——`mi_gallocr_alloc_graph` 全程不调用
  `malloc`/`free`。

这也是为什么 `mi_kv_cache` 要多背两个字段(`galloc` 和 `graph_ctx`)而
不是让 `mi_qwen2_forward` 自己管理:reserve 出来的 arena 和 reset 复用的
graph_ctx,生命周期天然和"一次生成会话"(等同于 KV cache 的生命周期)绑在
一起,而不是和"某一次 forward 调用"绑在一起。

## 测试结果

```
$ ctest --test-dir build          # Release,10 个测试(含 zero_alloc)
100% tests passed, 0 tests failed out of 10
```
(Debug 同样 10/10;Tsan 9/9——`zero_alloc` 因为要和 `--wrap=malloc` 的
链接方式打交道,和 TSan 自己的分配器接管机制没有验证过混用是否可靠,
所以只在非 Tsan 配置下构建,详见 `tests/CMakeLists.txt` 的注释。)

`tests/test_gallocr.cpp` 五个场景(view 链、view 后原地、多消费者、
图输出不被复用、stats 合理性)全部通过;`tests/test_zero_alloc.cpp`
(全局 `operator new/delete` 重载 + `-Wl,--wrap=malloc,--wrap=free`,
预热一次后跑 64 个 decode step)四个计数器(new/delete/malloc/free)
全部为 0。

`tests/test_graph.cpp`(阶段 1 golden dump 回归)和
`tests/test_model_qwen2.cpp`(批量 vs 增量 decode 一致性)在
`GALLOCR_NO_INPLACE`×`GALLOCR_NO_BESTFIT` 四种组合、以及
`GALLOCR_POISON=1` 下,都额外手动跑过一遍,全部逐位一致(这四种组合
不是 CTest 里自动跑的,是本阶段验收时手动交叉验证的,记录在这里而不是
让 CI 每次都跑 4 遍是因为这四种组合的结果理论上必须完全相同——它们只是
换了个字节偏移,不改变任何浮点运算顺序)。

`bench/results/gallocr.txt`:真实 Qwen2.5-0.5B-Instruct F32 模型上
(n_ctx=1024),peak vs baseline 的实测数据(prefill n_tokens=
128/512/1024,decode n_past=0 和 n_ctx-1=1023),由
`scripts/bench_gallocr.py` 跑 `mini_infer --gallocr-stats` 得到,
原始数据见该文件,摘录如下:

| case | n_tokens | n_past | peak_MB | baseline_MB | reduction_x | n_inplace | n_bestfit |
|---|---|---|---|---|---|---|---|
| prefill n=128 | 128 | 0 | 6.250 | 415.312 | 66.45x | 218 | 1289 |
| prefill n=512 | 512 | 0 | 26.750 | 2165.250 | 80.94x | 218 | 1288 |
| prefill n=1024 | 1024 | 0 | 57.500 | 5674.500 | 98.69x | 218 | 1288 |
| decode n_past=0 | 1 | 0 | 0.048 | 2.919 | 60.39x | 218 | 1290 |
| decode n_past=1023 | 1 | 1023 | 0.056 | 5.542 | 98.69x | 218 | 1288 |

`n_inplace_reuse` 恒为 218 是符合预期的:这是每层固定的原地复用节点数
(rms_norm/silu/mul/add/rope/scale 这些逐元素算子),和 n_tokens 无关,
不随 batch 变化。`reduction_x` 随 n_tokens 增大而增大,是因为 baseline
里像 KQ 分数矩阵这类 `O(n_tokens^2)` 或 `O(n_tokens * n_ctx)` 的张量
在"独立分配"模型下线性/平方增长,而 gallocr 的 arena 只需要同时存活的
张量总和,大部分这类大张量在图里生命周期很短、算完就被下一个同样大小
的张量复用,所以峰值增长慢得多。

## 面试官可能追问的 3 个问题

**Q1:你说"原地复用"和"best-fit 复用"是两回事,能不能举例说明区别?**

原地复用是"这个节点的输出,直接写进它自己某个输入的buffer里",复用的是
**同一个字节区间**,不产生新的分配决策,由 `op_supports_inplace` +
`is_eligible_inplace_src` 的 6 个条件判断。best-fit 复用是"这个节点需要
一块新内存,但空闲链表里刚好有一块之前释放的、大小合适的块,直接用它,
不用把 arena 的水位线往后推"——复用的是**不同张量之间、生命周期不重叠**
的空间。两者可以同时发生在同一次 forward 里:某些节点原地写自己的输入,
另一些节点从空闲链表捡一块别人用剩下的。`GALLOCR_NO_INPLACE`/
`GALLOCR_NO_BESTFIT` 这两个开关能独立打开/关闭,就是为了在测试里把这两种
复用方式分开验证。

**Q2:你实现里 view 的"data 指针延迟解析"这个改动,如果不做会出现什么
具体的失败现象?第一次遇到的时候你是怎么定位到问题所在的?**

现象是段错误(空指针解引用),发生在 `mi_gallocr_alloc_graph` 内部处理
某个 view 节点、试图执行 `t->data = src->data + offset` 的那一行——因为
`src`(比如 `qh_rot`,一个还没轮到 gallocr 分配的普通中间节点)这时候
`data` 还是空指针,空指针加偏移量本身不会立刻崩,但后面任何试图读写这个
地址的 kernel 调用都会崩。定位过程:先看崩溃点的调用栈确认是哪个 view
节点的 `data` 计算出了问题,然后回想"阶段 1/2a 里这行代码为什么一直没
出过问题"——因为那两个阶段所有张量在**构造的瞬间**就已经有真实内存了,
`src->data` 从来不会是空的;阶段 2b 把这个前提改变了(`data` 推迟到
gallocr 阶段才赋值),但 `mi_context_view` 这个更底层的公共函数没有跟着
一起改,这个不一致直到跑真实模型(有大量此类 view)时才暴露。

**Q3:如果把"非持久张量不参与 arena 管理"这条规则去掉,权重也交给
gallocr 管理会怎样?**

权重的生命周期和"一次 forward 调用"完全不同步——它们要在模型整个生命周期
内保持不变,而 gallocr 的 arena 设计前提是"每次 forward 调用重新规划,
调用结束后这块内存随时可能被下一次调用覆盖"。如果把权重也交给 arena
管理,第一次 forward 调用可能就会因为权重张量的"引用计数"在某一层用完后
归零,而被当成可以释放/复用的空间——第二层还没读到的权重就已经被第一层
算出来的激活值覆盖掉了。这也是为什么"持久"这个概念必须在 gallocr 介入
**之前**就通过 `flags` 位固定下来,而不是让 gallocr 自己去猜"这个张量是不是
应该长期存活"。
