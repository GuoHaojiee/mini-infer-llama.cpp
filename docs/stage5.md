# 阶段 5:基准 + README

## 做了什么

- `scripts/bench_llamacpp_compare.py`(新增):同一 Q8_0 GGUF、同线程数
  (1 和阶段 3 测得的 decode 最快线程数 6)、同 prompt/生成长度,分别跑
  mini-infer 自己的 `--bench` 和 llama.cpp 的 `llama-bench -ngl 0`,
  原始数据写到 `bench/results/llama_bench_compare.csv`。
- `scripts/run_all_bench.sh`:一键串联 `bench_gallocr.py`、
  `bench_threads.py`、`bench_quant.py`、`compare_q8_llamacpp.py`、
  `bench_llamacpp_compare.py`,外加把环境信息(`lscpu`、SMT/大小核检测、
  睿频状态、编译器版本,复用 `bench_threads.py` 已有的 `gather_sysinfo`)
  写到 `bench/results/env_info.txt`。已经完整跑过一遍,见下方"跑出来的
  数据"。
- `scripts/summarize.py`:PROMPTS.md 列的五类派生指标,全部从
  `bench/results/` 已有的原始文件里机械计算出来(每一项都标注数值、公式、
  来源文件(+ 行号,能定位到的情况下)、测量条件),输出
  `bench/results/summary.md`。不做任何新的测量、不写"提升了多少倍"
  之外的评价性文字。
- `README.md`:项目介绍、构建/测试/运行方法、文字版架构图、模块一览表、
  直接引用 `bench/results` 数据的基准表格(每个数字都标了硬件/模型/
  线程数/基线)、已知局限(纯 CPU、只支持 F32/F16/Q8_0、只支持
  Qwen2.5、单请求、贪心采样、tokenizer 依赖 Python)。

## 为什么这样设计

**`summarize.py` 和它读的原始文件严格分开,不在基准脚本里顺手算比值。**
阶段 2b-4 的每个基准脚本(`bench_gallocr.py`/`bench_threads.py`/
`bench_quant.py`)职责都是"跑一次真实调用、写一行原始数字",不做任何
"提升了多少"的计算——这是 SPEC.md"基准约定"里明确要求的("只输出原始
数据表格...不写结论,由我分析"),阶段 5 的 `summarize.py` 才是唯一
被允许做这类计算的地方,而且明确要求写出公式和来源。把这两件事分成
两类脚本,而不是让每个基准脚本自己顺手加一列"倍数",是为了保证
"原始测量"和"派生计算"之间有一条清楚的边界:如果以后要改计算公式
(比如中位数换成均值),只用改 `summarize.py`,不用重新跑一遍所有基准。

**mini-infer 和 llama-bench 的统计量不同(中位数 vs 均值),没有强行
统一。** `mini_infer --bench` 用的是本项目从阶段 3 起就定好的"1 次预热
+ 5 次重复取中位数/最小值/最大值"(SPEC.md 的基准约定);`llama-bench`
自己内部也跑 5 次重复,但只汇报均值和标准差,不暴露原始的 5 个样本,
没法在外部重新算中位数。与其为了"看起来统一"而假装能对齐统计口径
(比如拿 llama-bench 的均值硬套中位数的位置),不如在
`bench_llamacpp_compare.py` 的输出文件表头和 `summarize.py` 的措辞里
明确写清楚"这一列是中位数、那一列是均值",两个数字并排放但不混淆——
如实记录比统计上的"整洁"更重要。

**"与 llama.cpp 比速度"这一节如实写慢多少,没有回避或美化。**
PROMPTS.md 明确要求"不得出现...'超过 llama.cpp'之类的表述;比 llama.cpp
慢就如实写慢多少",而且这本来就是预期结果:llama.cpp 的 Q8_0 AVX2 kernel
经过多年打磨(cache 分块、权重 repack 等),本项目阶段 4 的目标是"先写对
再写一层直接的 SIMD",从没打算、也不在阶段范围内去做那些更深的优化。
README 里这部分直接写了实测的比值(decode 大约是 llama.cpp 的 41%-46%),
不加修饰。

## 踩过的坑

**`gallocr.txt` 里 "prefill n=128" 这一列本身带一个空格,朴素的
`line.split()` 解析会把它拆成两个字段,导致后面所有列对不上号。**
写 `summarize.py` 解析 `bench/results/gallocr.txt` 时,第一版直接
`case, n_tokens, n_past, peak_mb, base_mb, reduction, n_inplace, n_bestfit = line.split()`,
运行时报 "too many values to unpack (expected 8)"——因为
`"prefill n=128"` 本身按空白分割是两个 token(`"prefill"` 和 `"n=128"`),
真正的字段从第三个位置开始。这是一个典型的"表格里某一列的显示格式
(为了人读起来直观,加了个空格)和机器解析用的分隔符(同一个空白字符)
冲突"的坑——`bench_gallocr.py` 当初设计输出格式时是纯粹为了给人看,
没有考虑到阶段 5 会有另一个脚本反过来解析它。修复很简单(先取前两个
token 拼成 case,剩下的按位置对应),但如果 `bench/results/` 下的文件
以后还要新增机器解析的消费者,更稳妥的做法是从一开始就用真正的分隔符
(比如 TSV 或 CSV)而不是"对齐的空格表格"。

**CSV 表头行被当成数据行解析,数字字段读到字符串 `"n_threads"` 直接
`int()` 崩溃。** `threads_decode.csv`/`quant_decode.csv`/
`llama_bench_compare.csv` 都以一行列名开头(`n_threads,mode,...`、
`config,n_threads,...`、`engine,phase,...`),`summarize.py` 最初的
`read_csv_rows` 只过滤掉 `#` 开头的注释行,没有过滤表头本身,导致
`int(nt)` 在读到字面量 `"n_threads"` 时抛异常。修复方式是维护一个"已知
表头首列名"的集合(`n_threads`/`config`/`engine`)一并跳过——这是个
一次性绕过,不是很优雅,但因为这几个 CSV 文件本来就是阶段 2b-4 各自
独立设计的输出格式(没有为了阶段 5 的复用统一过schema),现在把它们
拿来给同一个脚本解析,这种"表头长得不完全一样"的摩擦是预期之内的。

## 跑出来的数据

见 `bench/results/`(`env_info.txt`、`gallocr.txt`、
`threads_{prefill,decode}.csv` + `threads_sysinfo.txt`、
`quant_{prefill,decode}.csv`、`accuracy_f32_vs_hf.txt`、
`accuracy_q8_vs_llamacpp.txt`、`model_sizes.txt`、
`llama_bench_compare.csv`、`summary.md`)和 `README.md` 的"基准结果"
一节(直接引用这些文件,数字与来源逐一对应)。

## 面试官可能追问的 3 个问题

**Q1:`summarize.py` 里"决定 decode 最快线程数"这一步,你是用 static
模式各线程数的中位数取最大值,而不是把 static/dynamic/single 三种模式
混在一起比——这个选择的依据是什么?**

"最快线程数"这个概念本身是要回答"这台机器适合配几个 worker 线程",
这是一个和"用哪种 mul_mat 调度策略"正交的问题——`single` 模式在设计上
就不是生产配置(阶段 3 的实测数据显示它随线程数增加反而变慢,因为其他
线程只是空转陪着做 barrier 同步),把它混进来找"最快线程数"会人为地
压低这个数字的意义;`dynamic` 在小 chunk 时又会因为原子操作开销比
`static` 略慢。`static` 是没有任何调度开销、纯粹反映"这个线程数下这台
机器的硬件能提供多少算力"的模式,用它来定义"最快线程数"这个物理量,
再拿这个线程数去看其他模式的相对表现,逻辑上更干净。

**Q2:你在 README 里说"Q8_0 AVX2 在 n_threads=1 比 F16 AVX2 慢"——这是
不是说明你的 Q8_0 AVX2 kernel 写得不够好,还有优化空间?**

有优化空间,但"不够好"需要分清楚是算法选择的问题还是这个规模下的
客观现象。当前 Q8_0 点积每处理一个 32 元素 block 就要额外做一次
"读两个 fp16 缩放因子、标量乘法、标量 FMA"的工作,这部分是标量的、
不能被那 32 个元素的整数向量计算摊薄;矩阵越大,block 数量越多但每个
block 的边际开销占比不变,所以这不是一个"矩阵变大就会自动缓解"的问题,
而是"点积本身的 block 化开销相对 F16C 的纯直线转换更高"——要真正缩小
差距,需要把多个 block 的缩放因子也批量向量化处理(比如一次性转换 8
个 block 的 d 值),这是一个明确的、当前实现里没做的优化点,属于"阶段
4 定的目标是先写对再写一层直接的 SIMD",没有把 kernel 打磨到极致作为
本阶段验收标准。

**Q3:整个项目从阶段 0 到阶段 5,如果只能保留一条你认为最重要的设计
原则,会是哪一条?**

"kernel 的浮点运算顺序一旦定下来就不能因为重构而改变,所有跨阶段的
正确性验证都建立在这条不变量上"。从阶段 1 的 HF 对拍,到阶段 2a 的
"图版本与直接调用版本逐位一致",到阶段 2b gallocr 的四种复用组合
对拍,到阶段 3 三种并行模式 × 多线程数的 bit-exact,再到阶段 4"F32
分支必须原样保留、不能和 F16/Q8_0 合并成一个更通用的循环"——每一次
新增一层复杂度(计算图、内存复用、多线程、量化),都靠"新增的这一层
不改变已有 kernel 的浮点顺序"这条原则,把新增部分的正确性验证收窄成
一个局部问题,而不用每次都重新对一遍 HF。反过来说,如果这条原则被
破坏过一次(比如为了"代码更整洁"合并了 F32/Q8_0 的循环),后面几个
阶段建立起来的所有 bit-exact 测试链条都会变得不可信——出了 bug 也
分不清是新引入的还是重构带来的。
