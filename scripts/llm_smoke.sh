#!/bin/bash
# Check the LLM and training surface, then run the training demo.
# Compiler: ./bin/souc if this tree has one, otherwise SOUC_BIN.
set -u
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"
souc="${SOUC_BIN:-}"
if [ -z "$souc" ]; then
    if [ -x "$root/bin/souc" ]; then
        souc="$root/bin/souc"
    else
        souc=/workspace/sounio/bin/souc
    fi
fi
export SOUNIO_STDLIB_PATH="$root/stdlib"
files="
examples/tiny_train_demo.sio
examples/tiny_gpt_demo.sio
examples/llama_load_bench.sio
examples/llm_serve.sio
examples/llama_sample_demo.sio
examples/llama_gpu_decode_bench.sio
examples/llama_kv_paged_demo.sio
stdlib/llm/serve.sio
stdlib/learn/adam_f64.sio
"
for f in $files; do
    echo "check $f"
    "$souc" check "$f" >/tmp/llm_smoke_check.out 2>&1 || {
        tail -20 /tmp/llm_smoke_check.out
        echo "SMOKE FAIL $f"
        exit 1
    }
done
echo "run tiny_train_demo"
"$souc" compile examples/tiny_train_demo.sio -o /tmp/tiny_train_smoke >/tmp/llm_smoke_cc.out 2>&1 || {
    tail -20 /tmp/llm_smoke_cc.out
    echo "SMOKE FAIL compile"
    exit 1
}
out=$(/tmp/tiny_train_smoke)
echo "$out"
echo "$out" | grep -q "TRAIN PASS" || { echo "SMOKE FAIL run"; exit 1; }
echo "SMOKE PASS"
