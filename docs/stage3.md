# 阶段 3:线程池 + 动态任务领取

## 做了什么

- `src/ops_cpu.h/.cpp`:给 `mi_compute_mul_mat` 加了三种并行模式
  (`MI_MULMAT_SINGLE`/`STATIC`/`DYNAMIC`),通过一组进程全局的
  mode/chunk/动态计数器配置(`mi_mulmat_set_mode`/
  `mi_mulmat_reset_dynamic_counter`)控制,kernel 签名本身完全不变,仍是
  `(dst, ith, nth)`。
- `src/threadpool.h/.cpp`(全新实现):常驻线程池,自实现的 sense-reversing
  barrier(不用 `std::barrier`),线程绑核(`pthread_setaffinity_np`,可通过
  `--no-pin-threads` 关闭)。
- `src/graph.cpp`:`mi_graph_compute(g, pool)` 在 `pool != nullptr` 时转发给
  `mi_threadpool_run_graph`,`pool == nullptr` 时保留阶段 2 的单线程循环
  (仍然被 `--dump` 的逐节点交错执行路径使用)。
- `src/model_qwen2.h/.cpp`:`mi_kv_cache` 新增 `pool` 字段,和 `galloc`/
  `graph_ctx` 一样是"reserve 一次、每步复用"的资源;`mi_kv_cache_init` 新增
  一个有默认值的 `mi_threadpool_params` 参数(默认 `{1, true, STATIC, 8}`),
  所以阶段 2b 遗留的所有调用点(测试、`mi_dump`)不用改代码就继续以单线程
  运行,且与"真的过一遍线程池、只是 n_threads=1"完全等价(见下文"为什么
  bit-exact 是自动成立的")。
- `src/main.cpp`:`-t` 从"接受但不生效"变成真正生效;新增
  `--no-pin-threads`/`--pin-threads`、`--mulmat-mode`、`--mulmat-chunk`、
  以及给基准脚本用的 `--bench`(只在进程内用 `std::chrono` 给 prefill 调用
  和 decode 循环的每次 forward 调用计时,不含 mmap/gallocr reserve 的固定
  开销)。
- `tests/test_threadpool.cpp`(从占位符改写):mul_mat 单节点 + 5 节点链
  (rms_norm→mul→mul_mat→add→silu)在三种模式 × 线程数
  {1,2,4,8} × dynamic 的几个 chunk 下,与单线程结果逐位一致;这个二进制
  在 Release/Debug/Tsan 三种配置下都跑(`tests/CMakeLists.txt` 的标准
  per-module 循环,不需要为它加特殊规则),Tsan 下矩阵/图都刻意选得很小
  (k≤48, n≤37, r≤9),满足 PROMPTS.md"不要在整模型上跑 TSan"的要求。
- `scripts/bench_threads.py` + `bench/results/threads_*.csv`、
  `threads_sysinfo.txt`:真实测得的 prefill/decode tok/s,按线程数
  ×模式(dynamic 另扫 chunk)分组,一次预热 + 5 次重复,报告中位数/最小值/
  最大值。

## 为什么这样设计

**mode/chunk 是全局配置,不是 kernel 的参数。** `mi_compute_mul_mat` 的签名
从阶段 1 起就固定成 `(dst, ith, nth)`,阶段 3 不想为了"用哪种并行策略"这个
纯运行时的调度决策去改签名(改了签名,阶段 1/2 的直接调用、`--dump` 的
逐节点循环也都要跟着改)。这个决策"多久变一次"也支持把它放在全局:一次
forward 里所有 mul_mat 节点用的是同一个 mode(由 `mi_threadpool_new` 在
建池时设一次),真正"每个节点都不同"的只有 DYNAMIC 模式下的领取计数器,
这个计数器的生命周期天然和"下一个 mul_mat 节点"绑定,放在 barrier 里重置
（见下文）比塞进 `dst->op_params`(op_params 是构图时就定死的,不该用来
传"这次跑多少线程"这种运行时信息)更合适。

**为什么"三种模式 × 任意线程数"的 bit-exact 是自动成立的,不需要每种模式
单独证明。** `mul_mat` 的输出矩阵按行(`dst` 的 `ne[1]`,也就是权重的
输出特征维)天然无依赖:第 `i` 行的计算只读 `a` 的第 `i` 行和整个 `b`,
从不读其他行的结果,也从不写回 `dst` 以外的地方。三种模式的区别只是
"哪个线程算第几行",行内的 `k` 次乘加顺序(单线程时的顺序)完全不变。
所以只要证明"每行恰好被算一次、写到正确位置",bit-exact 就自动成立,
不用像"多线程规约求和"那样担心浮点结合律。`MI_MULMAT_SINGLE` 的正确性
（其他线程必须真的什么都不做,不能重复算一部分行)和 `MI_MULMAT_DYNAMIC`
的正确性(fetch_add 的原子性保证 `[0, n)` 被恰好分区一次,无论 chunk
大小)都由这条性质保证。stage 2b 定的 `mi_split_range` 从一开始就是"偶数
切分、余数分给前面几个线程",这条切分规则本身也不影响 bit-exact,因为它
只决定"哪个线程拿哪几行",不改变行内运算顺序。

**barrier 的内存序论证(sense-reversing barrier,自实现,不用
`std::barrier`)。** 见 `threadpool.cpp` 文件头注释,这里摘要:
`arrived` 用 `fetch_add(acq_rel)`——同一个原子对象上的所有 RMW 天然有一个
全序,`acq_rel` 保证这个全序里每一次 `fetch_add` 既能看到前一次的写入
（acquire 那一半),又能把自己之前的写入传递给下一次(release 那一半)。
所以"让 `arrived` 达到 `n_threads` 的那次 `fetch_add`"所在的线程,在那一刻
已经间接看到了本节点里*所有*线程写 `dst->data` 的结果(每个线程写数据都
先于它自己的 `fetch_add`,在程序顺序里)。这个"最后到达"的线程接着做两件
事,都严格发生在 `generation.fetch_add(release)` 之前:(1)如果下一个
节点是 `MUL_MAT`,重置动态计数器;(2)把 `arrived` 清零。其他线程在
`generation.load(acquire)` 上自旋,一旦观察到新值,release/acquire 配对
保证"本节点所有写入"和"下一节点的计数器重置"都对它可见。这是把两件事
（"确保上一个节点写完"和"确保下一个节点的调度状态是新的")压缩进同一次
barrier 里完成,不需要为动态计数器的重置单独加一轮同步。

**外层"发布新图"用独立的 `run_id` 计数器,不能用图指针本身判断"是否变了"
——这是本阶段唯一一个真正调试出来前就被我从设计上排除掉的坑,值得单独
展开(见下节)。**

**为什么 `mi_kv_cache_init` 给线程池参数一个默认值而不是要求所有调用点
都传。** 阶段 2b 遗留的测试(`test_zero_alloc`、`test_model_qwen2`、
`test_graph`)和 `mi_dump` 工具都不关心多线程,默认值
`{n_threads=1, pin_threads=true, STATIC, 8}` 配上"bit-exact 自动成立"的
性质,保证它们的行为和阶段 2b 时完全一样,不用碰这些文件。

## 踩过的坑

**"图指针没变"不能当成"没有新工作"的判据——因为这个项目的图控制释放,
指针大概率真的不会变。** 第一次设计"线程怎么知道该处理新图了"这个外层
唤醒机制时,最直觉的写法是让 worker 自旋比较 `graph` 指针:
`while (pool->graph.load() == last_seen_ptr) { spin(); }`。但
`mi_qwen2_forward` 每步只 `mi_context_reset(kv->graph_ctx)`(把 bump
指针清零复用同一块内存),不会 new 一个新 context,所以
`mi_graph_new(ctx, cap)` 在这块复位后的 arena 上分配出的 `mi_cgraph*`
——以及它内部的 `nodes`/`leafs` 数组——在连续两次 forward 调用里*几乎必然
是同一个地址*。如果用指针比较做唤醒判据,第一次调用能跑(初始值是
`nullptr`,第一次赋值必然"看起来变了"),但从第二次 decode 步开始,worker
会永远自旋在一个"从它的角度看从未变化"的值上,整个程序卡死在第二次
`mi_threadpool_run_graph` 调用——而且只有在真正的自回归多步 decode 场景
才会触发,一次性的 prefill-only 冒烟测试完全看不出问题。解决方式是唤醒
判据换成一个单调递增的 `run_id`(`std::atomic<uint64_t>`,每次
`mi_threadpool_run_graph` 调用都 `fetch_add(1, release)`),而不是比较
"数据本身"是否变化——这类"用版本号/序号代替值比较"的模式,在这个项目里
其实和阶段 2b 的 gallocr `remaining()`/`n_children` 分离是同一类教训:
不要用一个容易在正常操作下"看起来没变但其实是全新一轮"的量去做同步判据。

## 面试官可能追问的 3 个问题

**Q1:你说 mul_mat 三种并行模式在任何线程数下都 bit-exact,这个论证对
其他算子(比如 softmax、rms_norm)成立吗?为什么这个项目里它们不需要
三种模式?**

论证的关键前提是"输出的每个划分单元(这里是矩阵的行)之间没有交叉依赖,
划分单元内部的运算顺序不随线程数变化"。rms_norm/softmax/silu/scale 等
逐元素或按行归约的算子同样满足这个前提(每行的归约——比如 softmax 的
max/sum、rms_norm 的平方和——只在那一行内部进行,不跨行),所以它们
换成任何线程数、任何切分方式都一样 bit-exact,这也是为什么 SPEC.md
明确说"其他逐元素/归约算子按行静态切分"就够、不需要像 mul_mat 那样再给
三种模式——静态切分和动态切分对这些算子只有调度开销的区别,没有正确性
或数值差异,所以不值得为它们单独做 DYNAMIC/SINGLE 变体。真正需要多种
模式的原因是 mul_mat 在实际模型里行为差异大(prefill 的 M 维大、decode
退化成矩阵-向量乘,不同问题规模下 STATIC 的均分策略可能因为某些行的
计算量不均衡——比如这个项目里其实每行计算量是均匀的,但更一般的 llama.cpp
场景里可以想象权重的某些行涉及缓存效应不同——而导致负载不均,DYNAMIC
用来兜底这种情况),这是调度策略的取舍,不是正确性要求。

**Q2:如果两个 mul_mat 节点在同一个 forward 里前后相邻(比如 attention 里
Q/K/V 三个投影),DYNAMIC 模式的计数器重置为什么不会有竞态——万一某个
线程还没读完上一个 mul_mat 节点的计数器,barrier 的"最后到达者"就已经把
它重置成 0 了?**

不会,因为 barrier 本身保证了"重置"和"读取"之间的顺序:计数器只在
`barrier_wait` 里被重置,而重置发生在"这一个 barrier 调用让 `arrived`
达到 `n_threads`"的那个时间点——也就是说,*所有* n_threads 个线程都已经
退出了上一个 mul_mat 节点的领取循环(每个线程的领取循环用
`start >= n` 判断退出,退出后才会调用 `barrier_wait`),才可能触发重置。
换句话说,重置这行代码在时间上严格晚于"所有线程都不会再读这一轮计数器"
这件事,不存在"重置发生时还有人在读旧一轮"的窗口。

**Q3:这个线程池的 worker 在没有图可跑的时候会一直忙等(`spin_backoff`
最终退化成 `sched_yield()`,而不是阻塞在条件变量上),这在真实的
serving 场景里(比如两次请求之间可能空闲几秒到几分钟)不会白白吃满
CPU 吗?你为什么不用条件变量?**

会,这个设计明确用"吃 CPU 换延迟"——SPEC.md 原文要求"自旋后短暂
yield",本项目面向的是"模型加载后持续做批量推理"的场景,节点之间的
间隔通常是几微秒到几十微秒(一次 kernel 调用的量级),用互斥锁/条件变量
的系统调用开销(微秒级但对这个量级不算小)反而会成为瓶颈;`yield()`
在真的长时间没有新图时会让出 CPU 给其他进程,不是完全的忙等,但确实
不会把 CPU 完全让出来睡眠。如果要支持"两次请求间隔很长"的 serving 场景,
需要在这个 spin/yield 循环基础上加一层"自旋超过某个阈值后才真正 park 到
条件变量/futex"的分层等待策略——这是一个明确的、当前实现里没有做的
扩展点,原因是本阶段的验收目标(和后续阶段的基准)都是"模型已加载、
持续 decode"的场景,不覆盖长空闲间隔。

## 基准结果

真实机器:AMD Ryzen 5 5600H,6 物理核 / 12 逻辑核(SMT),单一频率
（未检测到大小核),未在本环境暴露睿频开关文件(`intel_pstate/no_turbo`
是 Intel 专属,这台 AMD 机器上不存在;`cpufreq/boost` 在此环境也未暴露),
编译器 g++ 11.4.0,`-O3` Release 构建。完整 `lscpu` 见
`bench/results/threads_sysinfo.txt`。测试参数:F32 Qwen2.5-0.5B-Instruct,
prompt 长度 32,decode 长度 16(15 次实际 decode forward 调用),
每组 1 次预热 + 5 次重复。完整原始数据见
`bench/results/threads_{prefill,decode}.csv`;下表是中位数(min–max 全部
在中位数 ±1% 以内,故只列中位数):

**prefill tok/s**

| n_threads | single | static | dynamic/chunk=1 | dynamic/chunk=8 | dynamic/chunk=32 |
|---|---|---|---|---|---|
| 1 | 2.743 | 2.813 | 2.813 | 2.814 | 2.814 |
| 2 | 1.630 | 5.458 | 5.437 | 5.440 | 5.435 |
| 4 | 1.606 | 10.612 | 10.511 | 10.530 | 10.515 |
| 6 | 1.561 | 14.534 | 14.506 | 14.644 | 14.220 |

**decode tok/s**

| n_threads | single | static | dynamic/chunk=1 | dynamic/chunk=8 | dynamic/chunk=32 |
|---|---|---|---|---|---|
| 1 | 2.726 | 2.801 | 2.792 | 2.796 | 2.799 |
| 2 | 1.627 | 5.259 | 4.583 | 5.146 | 5.221 |
| 4 | 1.602 | 9.803 | 8.263 | 9.549 | 9.676 |
| 6 | 1.560 | 12.759 | 11.132 | 12.679 | 12.555 |

两个可以从数据本身直接读出、不需要额外假设的现象:

- **`single` 模式的吞吐不随线程数增长,反而在 n_threads ≥ 2 时明显低于
  n_threads = 1。** 这是符合预期的:`single` 下永远只有 1 个线程真正计算,
  其余线程仍然要参与每个节点后的 barrier(参与 `fetch_add`、参与自旋),
  n_threads = 1 时barrier 的"最后一个到达者判断"在同一行代码里立即为真、
  从不自旋,而 n_threads ≥ 2 时每个节点都多一轮真实的跨线程同步和自旋
  开销,纯粹是这层同步成本、没有任何计算收益,所以比单线程池还慢。这正是
  `single` 模式作为"三选一"里一个基线/调试选项存在的意义,不是生产配置。
- **`static` 与 `dynamic` 在 prefill 下几乎不受 chunk 大小影响,但在
  decode 下 chunk 越小差距越明显**(例如 6 线程 decode:chunk=1 比
  static 慢约 13%,chunk=8/32 只慢 0.6%~1.6%)。Prefill 的 mul_mat 每次
  要处理的输出行数远大于 decode(decode 是矩阵-向量乘,输出行数由权重
  形状决定,与 batch 无关,行数本身就少),行数越少,`fetch_add` 领取的
  次数相对总工作量的占比越高,原子操作本身的开销才会显现出来——这与
  DYNAMIC 模式设计初衷("用来兜底负载不均")无关,是本项目这套 kernel
  行内计算量均匀时的一个纯调度开销现象。

## 已知局限 / 后续阶段会解决

本阶段的所有数字都是纯标量(无 SIMD)、F32 权重下测得的,阶段 4 引入
AVX2 向量化点积和 Q8_0 量化后,单线程算力本身会大幅提升,届时
`bench_threads.py` 的方法论(prefill/decode 分开、预热+5 次重复、
mode/chunk 扫描)不变,但需要重新在阶段 4 的二进制上跑一遍,不能复用本
阶段这份数字。
