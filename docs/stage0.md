# 阶段 0:GGUF 解析

## 做了什么

- `src/gguf.cpp`:GGUF v3 解析器。mmap 整个文件(`PROT_READ, MAP_PRIVATE`),
  用一个带边界检查的字节游标(`mi_cursor`)顺序解析 header、metadata
  key-value 表(13 种值类型全支持,含数组;数组元素类型也覆盖了字符串/
  整数族/浮点族/bool)、tensor info 表,最后按 `general.alignment`
  (默认 32,来自元数据缺省)算出数据区起始偏移,给每个张量算出
  `data = mmap_base + data_offset + tensor.offset`,全程零拷贝。
- 张量类型:支持 F32(id 0)、F16(id 1)、Q8_0(id 8);其他类型在解析
  张量信息表时立即失败,`mi_gguf_open` 返回 `nullptr` 并把张量名和原始
  type id 写进 `err`(用 Q4_0 文件验证过,见下方测试记录)。
- 词表:`mi_vocab_load` 读 `tokenizer.ggml.tokens` + `tokenizer.ggml.token_type`,
  在加载时把每个 token 预先解码成原始字节(GPT2 风格 byte-level BPE 的
  256 项字节-unicode 映射的反向表),NORMAL 类型走字节还原,
  CONTROL/USER_DEFINED/UNKNOWN/UNUSED 类型按字面文本输出(不做
  skip_special_tokens,与 HF `tokenizer.decode()` 的默认行为一致)。
- `tools/dump.cpp`:把全部元数据和张量表打印成 tab 分隔的行格式(字符串
  转义 `\`、`\n`、`\t`、`\r`,保证一行一条记录),以及一个
  `--decode-ids <file>` 模式做 id -> 文本解码。
- `scripts/compare_gguf_dump.py`:独立用 Python `gguf` 库解析同一文件,
  和 dump 输出逐条比对(KV 键、类型、标量值/数组每个元素、张量名/类型/
  形状/字节数),自动判定通过/失败,不依赖人眼。
- `scripts/compare_decode.py`:用 HF `AutoTokenizer` 对含中文、emoji、
  制表符、引号和完整 chat template 的样例文本编码得到 id 序列,让
  `mi_dump --decode-ids` 解码,和 `tokenizer.decode()` 逐条比较。
- `tests/test_gguf.cpp` + `tests/fixtures/tiny.gguf`:一个用 Python `gguf`
  库(不是我们自己的写入器)生成的小文件,覆盖全部 13 种 KV 值类型、非默认
  alignment(16,真实 Qwen2 文件从不设置这个 key,所以没有它就永远测不到
  这条路径)、F32/F16/Q8_0 三种张量类型、以及一个两词条的迷你词表
  (一个 NORMAL、一个 CONTROL),在 CTest 里做不依赖大模型文件的单元测试。
- `docs/models.md`:F32/F16/Q8_0 三个模型文件复用自同机的
  `mini-llama.cpp` 项目(经确认;命令和 llama.cpp 版本如实转记自那个项目
  自己的 `docs/ENV.md` / `scripts/prepare_models.sh`,不是编造的),记录了
  `mi_dump` 实测的张量类型统计。

## 为什么这样设计

**为什么用 mmap 而不是 read() 整个文件进内存。** 模型文件几 GB,`read()`
需要一次性分配等大的堆内存并整体拷贝;mmap 让内核按需换页,`mi_gguf_open`
对 2GB 的 F32 文件只花 60-70ms(header/kv/tensor 表解析,不触碰权重数据),
后续访问权重时才真正产生缺页。这也是"张量数据按对齐后零拷贝访问"的唯一
实现方式——一旦拷贝一份,就不再是零拷贝了。

**为什么 KV 值统一存成"每种标量类别一个最宽字段"而不是按 13 种类型各存
一份。** `mi_gguf_value` 只有 `u`(uint64)、`i`(int64)、`f`(double)、
`b`、`s` 五个标量槽位,13 种 GGUF 类型按族(无符号整数/有符号整数/浮点/
bool/字符串)映射进对应槽位,同时保留原始 `type` 做精确的类型标记。这样
`mi_gguf_kv_scalar_to_string` 之类的通用渲染函数不需要 13 个 `case` 分支
各写一遍取值逻辑,只需要 5 个;新增数组元素类型的支持(`arr_i`/`arr_f`/
`arr_b`/`arr_s`)同理。代价是牺牲了"存储即类型"的静态类型安全,但 KV 表
一次性加载、之后只读,不是热路径,这个代价可以接受。

**为什么专门为 dump 设计两个"generic to_string"函数而不是给每种类型加
专属 getter。** `mi_gguf_get_u32/i32/f32/bool/str` 这些按 key 查询的类型化
接口,是给"程序知道自己要什么类型"的调用者用的(比如阶段 1 读 hparams,
明确知道 `qwen2.attention.head_count` 是 UINT32)。但 dump 工具要遍历
*任意* 一个 KV 条目并打印它,不知道类型,如果继续走"按 key 强类型查询"
就要为 8 种标量类型(还没算数组)各写一次 dump 分支。所以额外加了两个
只服务于"渲染成文本"这一件事的通用函数,两类接口分工明确,互不影响。

**为什么每个 NORMAL token 在加载时就完整解码,而不是在 decode 调用时才
现算字节映射。** 词表只有 15 万条,预解码一次的代价可以忽略;换来的好处
是 `mi_vocab_decode` 变成纯粹的字符串拼接,而且"多字节 UTF-8 字符被拆成
多个 token"这个事实被正确处理——每个 token 只贡献它自己的原始字节,
拼接顺序对了,最终整体按 UTF-8 解释才会正确,单独看某一个 token 的
"解码结果"本身可能不是合法 UTF-8(比如中文的三个字节被分进三个 id),这是
预期行为,已经在函数头注释里写清楚。

**为什么 `tests/fixtures/tiny.gguf` 要用 Python `gguf` 库生成而不是我们
自己写一个 mini-infer 写入器。** 如果拿自己写的写入器生成测试数据、再用
自己写的读取器去读,两边共享同一套(可能错误的)理解,bug 会互相抵消,
测试形同虚设。用独立的参考实现(Python `gguf` 库)生成 fixture,才是真正
独立的 oracle——这和 `scripts/compare_gguf_dump.py` 拿同一个库去对拍大
模型文件是同一个道理。

## 踩过的坑

1. **Q8_0 的 GGUF 类型 id 不是 2,是 8。** 最初想当然地假设类型 id 按
   `F32=0, F16=1, Q8_0=2` 连续排列,直接查了 `gguf-py/constants.py` 才发现
   实际是 `Q8_0=8`(2 是 `Q4_0`)。如果不核对直接写死映射表,加载真实
   Q8_0 文件会在 `case` 分支里直接走进"unsupported type"分支,而且报错信息
   会显示 type_id=8——容易让人怀疑是不是漏看了这个 id,而不是一眼看出
   映射表写错了。解决方法就是老老实实去读 `gguf-py` 的常量表,而不是凭
   记忆里"F32/F16/Q8_0 应该是连续的"这种直觉。
2. **KV 数组值一行打印导致的行边界问题。** 最初 `--decode-ids` 直接把解码
   出来的文本原样输出加一个换行,結果 chat template 这种本身包含多个
   `\n` 的解码结果,会在 stdout 里制造出比输入行数更多的输出行,
   跟 Python 测试脚本"一行输入对一行输出"的假设对不上(`compare_decode.py`
   一开始报 "expected 6 lines got 11")。解决方式是复用 KV 转储时已经写好
   的 `escape()` 函数,把解码结果里的真实 `\n`/`\t` 转义成两字符序列,
   保证一行输入永远对应一行输出,Python 侧再做对应的 unescape。这个坑
   说明"文本类输出格式"从一开始就要考虑内容本身可能包含分隔符,不能假设
   业务数据"看起来不会有换行"。
3. **GGUF 张量的 `shape` 打印顺序 vs `ne[]` 存储顺序。** 一开始担心 Python
   `gguf` 库打印的 shape 是不是经过反转(它内部确实为了 numpy 数组视图做过
   一次 `reversed()`),生怕拿反转后的 shape 去和我们自己 `ne[0]`-最内层的
   约定比较会对不上。追进 `gguf_reader.py` 源码才确认 `ReaderTensor.shape`
   字段用的是反转前的原始 `dims`(反转只用于内部 numpy reshape),和我们
   `ne[]` 的存储顺序完全一致,`compare_gguf_dump.py` 直接比较 `t.shape`
   和 `ne[]` 不需要额外反转。这类"到底谁反转了"的问题,靠猜测很容易两边都
   反转一次或者都不反转从而"凑巧"通过测试,唯一可靠的做法是去读参考实现的
   源码,而不是看打印出来的几个数字长得像不像。

## 测试结果

```
$ ctest --test-dir build
100% tests passed, 0 tests failed out of 9
```
(Debug 与 Tsan 两种构建配置下同样 9/9 通过,记录见本文档最后。)

```
$ python3 scripts/compare_gguf_dump.py <F32/F16/Q8_0 三个文件> --dump-bin build/tools/mi_dump
KV entries checked: 39
Tensors checked: 290
PASS: dump matches python gguf library exactly       # 三个文件都是这个结果

$ python3 scripts/compare_gguf_dump.py tests/fixtures/tiny.gguf --dump-bin build/tools/mi_dump
KV entries checked: 21
Tensors checked: 4
PASS: dump matches python gguf library exactly

$ ./build/tools/mi_dump <q4_0 文件路径>
dump: unsupported tensor type: name=token_embd.weight type_id=2
(退出码 1,符合 SPEC.md"遇到其他类型报出张量名和类型后退出")

$ python3 scripts/compare_decode.py <F32 文件> --dump-bin build/tools/mi_dump
[OK] 'Hello, world!'
[OK] '你好，世界！这是一个测试。'
[OK] 'emoji test: 😀🚀✨ and mixed 中文emoji混合😊'
[OK] 'Line with\ttab and \'quotes\' and "double"'
[OK] 'seq#4'   # 完整 chat template 序列(多个 <|im_start|>/<|im_end|> 及换行)
[OK] 'seq#5'   # 纯特殊 token 序列
PASS: all 6 sequences match HF tokenizer.decode()
```

## 面试官可能追问的 3 个问题

**Q1:为什么权重可以 mmap 零拷贝,但 tokenizer 的词表不行(要解码成
`std::vector<std::string>`)?两者不是都在同一个 GGUF 文件里吗?**

权重张量在文件里的字节布局就是它运行时需要的布局(F32/F16 是连续的浮点
数组,Q8_0 是连续的定长 block),kernel 可以直接对 mmap 出来的指针做数学
运算,不需要转换,所以能零拷贝。词表不同:GGUF 里存的是"byte-level BPE
编码后"的字符串(比如空格被编码成 `Ġ`),而我们要的是解码回原始字节后的
文本——这是一次真正的数据变换,不是同一份字节的不同解释方式,必然要产生
新内存。零拷贝的前提是"文件里的表示"和"使用时的表示"字节级相同,这里不
成立。

**Q2:`general.alignment` 不存在时用默认值 32,如果 32 本身不是 2 的幂
会怎样?你在代码里为什么还要检查它是不是 2 的幂?**

32 当然是 2 的幂,这个检查针对的是文件里*显式写了* `general.alignment`
但写了个非法值(比如 0,或者 24 这种不是 2 的幂的数)的情况——这是在信任
一个来自文件、本质上是外部输入的数字之前的防御性检查。如果不检查直接拿
一个比如说 0 的 alignment 去做 `offs % alignment`,会直接除零崩溃;如果是
非 2 的幂但非零的值,对齐计算出来的偏移不一定是所有张量都遵守的边界,
后续访问可能读到错位的数据而不报错,是更隐蔽的 bug。检查失败时直接返回
`nullptr` 并报错,比静默使用一个可疑的对齐值安全。

**Q3:`mi_vocab_token_to_text(id)` 对单个 id 返回的字节序列,你自己在
注释里说"可能不是合法 UTF-8"。既然如此,为什么不干脆不提供这个单 token
接口,只留 `mi_vocab_decode`(整个序列一起解码)?**

单 token 接口在调试时很有用——比如要打印"prefill 阶段处理到第 N 个
token 是什么"这种诊断信息时,或者以后写 tokenizer 相关的其它工具时,
需要看单个 id 对应的原始片段而不是要求调用方总是攒齐一整个序列才能看到
任何输出。保留这个接口、但在文档里明确说明它的字节可能是某个多字节字符
的一部分、不能单独当作合法 UTF-8 使用,是"给出有用的原语,同时诚实说明
它的限制"的取舍,而不是为了接口"看起来安全"就砍掉一个有诊断价值的能力。
真正要保证得到合法文本的调用方,必须用 `mi_vocab_decode` 而不是自己拼
多次 `token_to_text` 的结果做字符串比较。

## 附:Debug / Tsan 构建下的测试结果

```
$ cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build-debug && ctest --test-dir build-debug
100% tests passed, 0 tests failed out of 9

$ cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Tsan && cmake --build build-tsan && ctest --test-dir build-tsan
100% tests passed, 0 tests failed out of 9
```
