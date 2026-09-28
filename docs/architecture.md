# mini-infer 架构设计

本文档描述 src/ 下核心数据结构的接口设计,以及阶段 1(直接调用)与阶段 2+
(计算图)如何复用同一批 kernel。对应的头文件已经写出(仅接口,无实现);
本文档解释"为什么这样设计"。

## 1. tensor(src/tensor.h)

```c
struct mi_tensor {
    mi_type type;
    int64_t ne[4];       // ne[0] 最内层维度(GGUF 约定)
    size_t  nb[4];        // 每维步长(字节)
    mi_op   op;
    int32_t op_params[8];
    mi_tensor* src[2];
    void* data;
    mi_tensor* view_src;
    size_t view_offs;
    int32_t n_children;   // 引用计数
    uint32_t flags;       // WEIGHT/INPUT/OUTPUT/KV
    char name[64];
    void* extra;
};
```

**为什么 `src` 只有 2 个槽位。** 本项目要用到的算子(get_rows、rms_norm、mul、
add、mul_mat、rope、softmax、silu、scale、cpy)里,占用 src 最多的是
get_rows(表 + ids)、mul_mat(权重 + 激活)、mul/add(两个操作数)、
cpy(源 + 目标 view),都是 2 个。为了避免每个算子都需要一个"位置张量"或
"mask 张量"作为第三个输入,做了两个设计选择:

- RoPE 的位置信息不用单独的张量表示,而是编码进 `op_params`(`n_past`,
  因为一批内第 r 行的绝对位置就是 `n_past + r`,不需要显式传一个 position
  数组)。
- softmax 的因果 mask 不物化成张量,而是在 kernel 内部根据 `op_params` 里的
  `n_past` 现算:输出第 r 行只允许看到列 `[0, n_past + r]`。

这两个决定我在"不确定之处"里列出来,因为如果以后要支持非因果的自定义
mask,这个设计就要改。

**`view_src` 只指向真正的存储持有者,不形成链。** 对 reshape/permute/view
连续作用产生的张量,构造时立刻把 `view_src` 解析成
`src->view_src ? src->view_src : src`,而不是指向上一个 view。这样
gallocr 判断"这块 buffer 还有没有活着的 view"时只需要看一层,不需要递归
遍历链条。阶段 2b 的"view 链"测试就是验证这个展平行为本身是正确的。

**`n_children` 由 gallocr 的预演阶段(阶段 2b)填,不是 graph 构建阶段填。**
阶段 2a 构图时暂不做内存复用,所以不需要引用计数;把这个字段的"生产者"放在
gallocr 里,是因为引用计数的语义(比如 view 的引用记到 view_src 上)本来
就是分配器的决策,不是图结构本身的属性。

**`flags` 用位掩码而不是一个"是否持久"的 bool。** 权重、输入、输出、
KV cache 这四类张量在"是否参与 arena 管理"这件事上行为一致(都不参与),
但在其他地方需要区分(比如 dump 工具只想导出 OUTPUT,内存统计只想单独看
KV)。`mi_tensor_is_persistent()` 是这四个 bit 的或。

## 2. context(src/context.h)

```c
struct mi_context {
    mi_tensor* tensors; size_t tensor_capacity, tensor_count;
    unsigned char* data; size_t data_capacity, data_used;
    bool no_alloc;
};
```

一次性 malloc 两块内存(元数据数组 + 数据区),之后 `mi_context_new_tensor`
只是从这两块里各切一段,`mi_context_reset` 把两个游标归零即可复用。这满足
SPEC.md 的约束:"tensor 元数据从预分配的 context 池中取,不使用
new / vector 扩容"。

三种使用模式:

| 模式 | no_alloc | data 从哪来 | 用在哪 |
|---|---|---|---|
| 自带数据 | false | ctx 自己的 data 区 bump 分配 | 阶段 1 直接调用、阶段 2a 中间张量(暂不复用) |
| 纯元数据 | true | 由 gallocr 事后赋值 | 阶段 2b 起的图中间张量 |
| 外部数据 | (不适用,用 `mi_context_new_tensor_from_data`) | 调用者传入指针,不占用 ctx 的 data 区 | GGUF mmap 权重、KV cache |

第三种是新增的第三条路径:权重和 KV cache 的数据本来就不该被 arena 管理
(它们是持久张量),也不该被 context 的 data 区 bump 分配(它们的生命周期
和 context 不一致 —— 权重活到模型销毁,KV cache 活到会话结束,而普通
context 每步 reset)。`mi_context_new_tensor_from_data` 只切元数据槽位,
`data` 指针由调用者提供。

## 3. graph(src/graph.h)

```c
struct mi_cgraph { mi_tensor** nodes; int n_nodes; mi_tensor** leafs; int n_leafs; int capacity; };
```

构图分两步:

1. 用 `mi_graph_*`(get_rows/rms_norm/.../cpy/view 等)一路调用,每次返回
   一个新的、`op` 和 `src` 已填好、`data == nullptr` 的结果张量 —— 这一步
   只是在搭一条"表达式"链,不涉及执行顺序。
2. `mi_graph_build_forward(g, output)` 从 `output` 出发,沿 `src[0]/src[1]`
   做后序 DFS(去重),把每个 op 节点按依赖关系产生的顺序放进 `g->nodes`,
   叶子(`op == NONE`,即权重/输入/KV)放进 `g->leafs`。

`mi_graph_compute` 再按 `g->nodes` 顺序,对每个节点调用
`mi_compute_forward`(在 ops_cpu.h 里定义,按 `node->op` 分派到具体 kernel)。

**为什么每步都重建图。** `n_past` 和 `n_tokens` 每步都可能变(prefill 到
decode 的过渡、KV cache 逐步增长),意味着每个算子的输出形状、softmax 的
mask 范围、cpy 写入 KV 的 view 偏移都在变。与其设计一套"可变形状的静态图 +
运行时 patch",不如利用 context 的"元数据零堆分配"特性,每步整体重新搭一遍
图 —— 构图本身几乎不花时间(只是填结构体),真正的开销在 gallocr 的规划
(阶段 2b 会把"规划"和"reserve 出的 arena 大小"分开,规划这部分也不产生堆
分配)。

## 4. gallocr(src/gallocr.h,阶段 2b)

接口已经定义好(见 gallocr.h 内的详细注释),核心分两个阶段:

- `mi_gallocr_reserve(galloc, graphs[], n)`:对一组"最坏情况"的图(prefill
  取最大 batch、decode 取 `n_past = n_ctx - 1`)做预演,取所需峰值的最大值,
  一次性申请 arena。
- `mi_gallocr_alloc_graph(galloc, g)`:每步对新图重新规划偏移,复用同一块
  arena,不再申请内存。

这个二分是为了同时满足两个互斥的目标:内存要按"实际会用到的最坏情况"来
定,但运行时(尤其是 decode 的每一步)不能有任何堆分配。PROMPTS.md 原本的
分工是分配逻辑(best-fit、原地复用、view 共享 buffer)由用户实现,我负责
接口/预演/测试/poison;阶段 2b 实际开始时用户回复"由你全部实现",所以最终
接口、分配逻辑、测试、poison 调试模式全部由我完成——具体的算法设计、
踩过的坑(包括一处直到跑真实模型才暴露的 bug)见 docs/stage2b.md。

## 5. threadpool(src/threadpool.h,阶段 3)

常驻线程,每个节点执行完做一次 barrier 同步再进入下一个节点。kernel 函数
本身(见下一节)已经在阶段 1 就设计成 `(dst, ith, nth)` 的形式,threadpool
只是把 `ith`/`nth` 换成真实值,不需要改 kernel 签名或 graph 执行逻辑。

## 6. ops_cpu 中 kernel 函数的签名风格(src/ops_cpu.h)

```c
void mi_compute_mul_mat(mi_tensor* dst, int ith, int nth);
```

统一约定:kernel 只接受 `dst`(从 `dst->src[0]/src[1]` 读输入、从
`dst->op_params` 读参数、写到 `dst->data`)加上 `ith`/`nth` 两个线程切片
参数。`dst->data` 由调用者预先分配好(阶段 1/2a:context 的 data 区;
阶段 2b 起:gallocr 规划好的 arena 偏移)。

### 阶段 1 直接调用版本 与 阶段 2 图版本如何复用同一批 kernel

这是 PROMPTS.md 第一条消息明确要求说明的点,单独展开:

- 阶段 1(model_qwen2.cpp 直接调用):手写前向逻辑里,每一步显式
  `dst = mi_new_tensor_xxx(ctx, ...); dst->src[0] = a; ...; mi_compute_mul_mat(dst, 0, 1);`
  ——构造张量、填 src、立刻执行,三步紧挨着写在一起。
- 阶段 2a(计算图):把"构造张量 + 填 src"搬到 `mi_graph_mul_mat(ctx, a, b)`
  里(只返回 `dst`,不执行);所有这些调用先跑一遍搭出完整的表达式树,再
  由 `mi_graph_build_forward` 排序、`mi_graph_compute` 统一执行,执行时调用
  的还是 `mi_compute_mul_mat(dst, 0, 1)`——字面上同一个函数、同样的参数。
- 阶段 3 起:`mi_graph_compute` 内部把 `mi_compute_mul_mat(dst, 0, 1)` 换成
  线程池里每个线程调用 `mi_compute_mul_mat(dst, ith, nth)`,函数本身不变。

也就是说,`mi_compute_*` 这一层是唯一"真正做数学"的代码,阶段 1/2a/2b/3/4
的区别只在于**谁调用它、传什么 ith/nth、dst->data 从哪来**,从不重写计算
逻辑本身。这也是阶段 2a 验收标准("图版本与阶段 1 版本 memcmp 逐位一致")
成立的根本原因:两边跑的是同一段浮点代码,顺序完全相同。

op_params 各算子的槽位约定(见 ops_cpu.h 顶部注释,这里汇总一遍):

| op | op_params[0] | op_params[1] | op_params[2] |
|---|---|---|---|
| rms_norm | eps (f32) | - | - |
| rope | n_past (i32) | n_dims (i32) | freq_base (f32) |
| softmax | n_past (i32) | scale (f32) | - |
| scale | s (f32) | - | - |

## 7. quant / gguf / model_qwen2

quant.h、gguf.h、model_qwen2.h 的详细设计见各自头文件顶部的注释;
model_qwen2.h 的 `mi_qwen2_forward()` 签名从阶段 1 到阶段 4 保持不变——内部
实现从"直接调用"演变到"图 + 线程池 + 量化 kernel",但 main.cpp 和对拍/dump
工具从不需要跟着改,这是刻意的设计:让"如何计算"和"计算什么"两件事在接口
层面分开。

## 8. 目录与构建

见根目录 CMakeLists.txt(Release/Debug/Tsan 三个配置)和 tests/CMakeLists.txt
(每模块一个可执行文件,CTest 管理)。SIMD 相关的约定(每函数 target 属性、
运行时 __builtin_cpu_supports 分发、不开全局 -mavx2/-march=native)已经写进
CMakeLists.txt 顶部注释,阶段 4 落地时直接照此执行。
