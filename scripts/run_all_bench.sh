#!/usr/bin/env bash
# Stage 5: one-click runner for every benchmark/accuracy script this
# project has (PROMPTS.md: "一键跑完 gallocr 统计、线程扫描、精度与量化
#对比、与 llama.cpp 的 llama-bench 对比"). Every step below is a script
# that was already run individually in stages 2b/3/4/5 -- this just chains
# them with one consistent invocation, and (re-)writes the same
# bench/results/ files those individual runs already produced/refreshed.
#
# Requires: build/mini_infer (Release build), a real F32/F16/Q8_0 GGUF
# triple, and -- for llama_logits / the llama-bench comparison -- the
# sibling mini-llama.cpp project's built third_party/llama.cpp checkout
# (see tools/CMakeLists.txt's MI_LLAMA_CPP_ROOT). If that checkout isn't
# present, this script still runs everything else and reports which two
# steps it skipped.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CPP_BIN="${CPP_BIN:-build/mini_infer}"
F32_MODEL="${F32_MODEL:-/home/guohaojie/Guo/mini-llama.cpp/models/qwen2.5-0.5b-instruct-f32.gguf}"
F16_MODEL="${F16_MODEL:-/home/guohaojie/Guo/mini-llama.cpp/models/qwen2.5-0.5b-instruct-f16.gguf}"
Q8_0_MODEL="${Q8_0_MODEL:-/home/guohaojie/Guo/mini-llama.cpp/models/qwen2.5-0.5b-instruct-q8_0.gguf}"
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$ROOT_DIR/../mini-llama.cpp/third_party/llama.cpp}"
LLAMA_LOGITS_BIN="${LLAMA_LOGITS_BIN:-build/tools/llama_logits}"
LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-$LLAMA_CPP_ROOT/build/bin/llama-bench}"

mkdir -p bench/results

if [ ! -x "$CPP_BIN" ]; then
  echo "error: $CPP_BIN not found -- build the Release configuration first" >&2
  exit 1
fi

echo "== [1/6] environment info (lscpu, hybrid/turbo check, compiler version) =="
python3 -c "
import sys; sys.path.insert(0, 'scripts')
import bench_threads as bt
with open('bench/results/env_info.txt', 'w') as f:
    f.write(bt.gather_sysinfo())
print('wrote bench/results/env_info.txt')
"

echo "== [2/6] gallocr peak-vs-baseline (scripts/bench_gallocr.py) =="
python3 scripts/bench_gallocr.py --model "$F32_MODEL" --cpp-bin "$CPP_BIN"

echo "== [3/6] thread scaling (scripts/bench_threads.py) =="
python3 scripts/bench_threads.py --model "$F32_MODEL" --cpp-bin "$CPP_BIN"

echo "== [4/6] quantization throughput (scripts/bench_quant.py) =="
python3 scripts/bench_quant.py --f32-model "$F32_MODEL" --f16-model "$F16_MODEL" \
  --q8-model "$Q8_0_MODEL" --cpp-bin "$CPP_BIN"

echo "== [5/6] Q8_0 vs llama.cpp accuracy (scripts/compare_q8_llamacpp.py) =="
if [ -x "$LLAMA_LOGITS_BIN" ]; then
  python3 scripts/compare_q8_llamacpp.py --model "$Q8_0_MODEL" --cpp-bin "$CPP_BIN" \
    --llama-logits-bin "$LLAMA_LOGITS_BIN"
else
  echo "skipped: $LLAMA_LOGITS_BIN not found (no sibling llama.cpp checkout at $LLAMA_CPP_ROOT)"
fi

echo "== [6/6] mini-infer vs llama-bench, same model/threads/prompt/gen length =="
if [ -x "$LLAMA_BENCH_BIN" ]; then
  python3 scripts/bench_llamacpp_compare.py --q8-model "$Q8_0_MODEL" --cpp-bin "$CPP_BIN" \
    --llama-bench-bin "$LLAMA_BENCH_BIN"
else
  echo "skipped: $LLAMA_BENCH_BIN not found (no sibling llama.cpp checkout at $LLAMA_CPP_ROOT)"
fi

echo ""
echo "all done -- raw output under bench/results/. Run scripts/summarize.py next."
