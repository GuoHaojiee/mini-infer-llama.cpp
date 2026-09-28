# 阶段 2a:计算图

## 做了什么

- `src/graph.cpp`:`mi_graph_new`(图对象本身和 nodes/leafs 数组都从
  context 的字节区里切,不用 `new`/`malloc`)、`mi_graph_build_forward` /
  `mi_graph_build_forward_expand`(后序 DFS 拓扑排序,去重)、
  `mi_graph_compute`(顺序调用 `mi_compute_forward` 分派到 ops_cpu 的
  kernel)、以及 `get_rows/rms_norm/mul/add/mul_mat/rope/softmax/silu/
  scale/cpy/reshape/view/permute/transpose` 十四个节点构造函数。
- `src/context.cpp` 新增 `mi_context_alloc_bytes`:从同一块 data 区切
  任意大小的裸字节(不经过 tensor 的 ne/nb 计算),给 `mi_graph_new` 用,
  保证"每步重建计算图不产生堆分配"这条约束延伸到图对象本身,不仅是
  tensor 元数据。
- `src/model_qwen2.cpp`:前向的每个算子调用从"构造张量 + 立刻调用
  `mi_compute_*`"改成"调用 `mi_graph_*` 只构造节点",全部构图完成后统一
  调 `mi_graph_build_forward`/`_expand` 排序、`mi_graph_compute` 一次性
  执行。`mi_qwen2_forward` 的对外签名和行为完全不变。
- `tests/test_graph.cpp`:小规模手搭图的拓扑排序单元测试,以及阶段 1 ->
  阶段 2a 的逐位回归测试(见下)。

## 关键设计问题:cpy 写入在图里"没有读者"

这是阶段 2a 里唯一真正棘手的地方,值得单独说明。

KV cache 的写入(`mi_graph_cpy` 把这一步新算出的 K/V 写进 `cache_k[l]`/
`cache_v[l]` 的某个切片)和多头注意力输出的拼接(每个 head 的结果通过
`cpy` 写进 `attn_concat` 的一个切片),这两类写入在数据流图上是"死胡同"
——**没有任何节点会把这次 cpy 返回的张量当作自己的 `src[]`**。后续真正
读取这块内存的节点(算 attention score 时读 `cache_k[l]` 某个 kv head 的
历史、O 投影读整个 `attn_concat`)是直接从**共享的原始 leaf 张量**
(`cache_k[l]`、`attn_concat` 本身)另起一个 view 构造出来的,而不是从
cpy 的返回值取的。

后果是:如果只从最终的 `logits` 出发做一次反向 DFS(`mi_graph_build_
forward(g, logits)`),遍历路径是
`logits -> ... -> k_head(view of cache_k[l]) -> cache_k[l](leaf)`,
**根本不会经过那次 cpy**——因为 `cache_k[l]` 是叶子,它的 `src[]` 是
空的,DFS 走到它就停了。这意味着 cpy 节点永远不会被加进 `g->nodes`,
`execute()` 会悄悄跳过所有 KV cache 写入,而不会报任何错——注意力算出来
的分数会用到全零(或上一次残留)的 K/V,而不是这一步刚写进去的值。

**这不是我编出来的假想问题**——第一次跑通图版本时,如果不处理这个,
prefill 的结果确实还是对的(因为 `n_past=0` 时 K/V 全部是"这一步新算的",
凑巧不依赖 cache 里的历史),但 decode(`n_past>0`,需要真正读到 cache
里之前几步的历史)会用到未初始化的内存,dump 出来的 checkpoint 一比对
立刻不一致——这正是 SPEC.md"批量 prefill 与逐 token prefill 要一致"
这条验收标准原本设计要抓的那类 bug,只是这次在更早的阶段(图 vs 直接调用
的一致性)就先被抓到了。

**解决方式**:给 `mi_graph_build_forward_expand` 这个"追加根"接口——
每次前向,把**每一个** cpy 的返回值都当作一个独立的"根"先 expand 一遍
(而不是只从 `logits` 一个根出发),最后再 expand 主输出 `logits`。
`execute()` 按 `g->nodes` 数组下标顺序执行,只要 cpy 对应的 expand 调用
发生在"读它的那个 view 对应节点被加入 nodes"之前(程序顺序上必然如此,
因为 KV 写入和拼接写入在 forward_layer 里就是写在读之前),cpy 就会先于
读它的节点出现在 `g->nodes` 里。14 个 head 各自的拼接写入之间不需要相互
排序(互不重叠的内存),KV 的 K 写入和 V 写入之间也不需要相互排序
(不同张量)——只需要"每个写,都在对应的读之前"这一条,`expand()` 按
构造顺序调用就自然满足。

这是 ggml/llama.cpp 实际代码里解决同一个问题用的同一个机制(它们把 KV
cache 的写入显式地多 `ggml_build_forward_expand` 一次),不是本项目
发明的技巧,但确实是这类"图里存在纯副作用节点"场景的标准解法,值得在这里
解释清楚为什么直接从单一输出反向遍历不够用。

## 测试

```
$ ctest --test-dir build
100% tests passed, 0 tests failed out of 9
```
(Debug、Tsan 同样 9/9。)

`tests/test_graph.cpp`:
- 拓扑排序单元测试:链式依赖、菱形依赖去重(共享的叶子只出现一次)、
  view 节点归类到 `nodes` 而不是 `leafs`(因为 `op != NONE`)、
  `mi_graph_build_forward_expand` 追加根、一个真正 build+execute 的小
  加法图(验证图版本确实调用了和阶段 1 相同的 `mi_compute_add`)。
- **阶段 1 vs 阶段 2a 逐位回归**:用 `tests/fixtures/tiny_qwen2.gguf`
  (阶段 1 就已经建好的迷你随机权重模型),对比新的图版本 `mi_qwen2_
  forward` 产出的全部 dump 张量(`embd`、每层、`output_norm`、`logits`),
  和**改造前**用同一个模型跑出来的 golden 文件(`tests/fixtures/
  stage1_golden_{prefill,decode}/`,改造前用当时的 stage 1 二进制生成,
  提交进仓库)逐 float 比较。prefill(`n_tokens=5, n_past=0`)和
  decode(先 warmup 5 个 token,再 `n_tokens=1, n_past=5`)两种情况都
  测,**全部逐位相同**。

另外在真实的 Qwen2.5-0.5B F32 模型上跑了一次手工抽查(prefill 4 个
token、生成 12 个 token),图版本的 dump 文件和生成文本与阶段 1 完全
一致,耗时也基本相当(单次前向多出来的图构造开销可以忽略)。

## 面试官可能追问的 3 个问题

**Q1:你为什么不在 `mi_graph_view`/`mi_context_view` 里直接把这类
"写后读"的依赖关系记录下来,而是要求调用方手动 `expand()` 每一个 cpy?**

因为 view 构造的时候,根本不知道"以后还会不会有人往它指向的这块内存写
东西"。`mi_context_view` 只是在做一件事:根据现有的 `ne`/`nb`/`offset`
算出正确的 `data` 指针,它的输入只有"从哪个张量、按什么形状去看"这一点
信息,不知道调用方之后会不会对同一块内存再发起一次 `cpy`。要在构造 view
的时候自动记录"写后读"关系,要么得让每个持久张量自己维护一份"谁写过我"
的列表(增加了叶子张量的状态,且这份列表本身的生命周期管理又是新问题),
要么退化成每次读之前扫描整个图找出所有写者(等价于我现在做的事,只是
挪了个地方做,还更隐蔽)。让调用方(`model_qwen2.cpp`)在构造 cpy 的
现场就记下它,反而是信息最全、最不容易遗漏的地方——毕竟"这次 forward
一共发起了哪些 cpy"这件事,构造图的代码比图本身更清楚。

**Q2:如果忘了给某个 cpy 调用 `expand()`,会有什么后果?测试能不能
抓到?**

后果是那次写入的数据永远不会被真正写进目标内存(因为它对应的节点没有
进入 `g->nodes`,`execute()` 根本不会碰它),读它的那个 view 会读到
该内存位置**当时恰好是什么**——对 KV cache 而言就是上一次前向残留的
旧值,或者(如果是这次前向里第一次用到这块 scratch 内存)未初始化的
垃圾。`tests/test_graph.cpp` 的 decode 回归测试专门抓这个:decode 场景
下 attention 必须正确读到 warmup 阶段写进 cache 的历史 K/V,如果某个
`expand()` 被漏掉,decode 的 `logits`/`layer_i` dump 会和 golden 文件
对不上,测试直接失败,而且失败的具体是 decode 用例(prefill 用例可能
因为"这一步的 K/V 全是新算的、不依赖历史"而凑巧蒙混过关)——这也是我
们坚持"prefill 和 decode 都要测"而不是只测其中一种的原因。

**Q3:`mi_graph_build_forward_expand` 需要按什么顺序调用才能保证正确?
如果 14 个 head 的拼接写入互相调换 `expand()` 的调用顺序,会不会出
问题?**

不会,因为这 14 次写入各自落在 `attn_concat` **互不重叠**的字节区间
(每个 head 占 `head_dim` 个 float 的一段,offset 不同),彼此之间没有
真实的先后依赖,可以按任何顺序 `expand()`。唯一必须保证的顺序是"每个
写,都要在读它的节点**之前**出现在 `g->nodes` 里"——具体来说,是"所有
side-effect 的 `expand()` 调用,要发生在读取那块共享内存的主输出(这里
是 `logits`)的 `build_forward`/`expand` 调用**之前**"。这是因为
`execute()` 完全按 `g->nodes` 数组下标顺序跑,不做任何基于依赖关系的
重排——顺序完全由"谁先被 DFS 追加进数组"决定,而 DFS 追加的顺序又完全
由我们**调用 `expand()` 的程序顺序**决定。`model_qwen2.cpp` 里的写法
(先收集本层所有 cpy 到 `side_effects`,所有层处理完之后统一按收集顺序
`expand()`,最后才 `expand()` 主输出)天然满足这一点,不需要额外排序
逻辑。
