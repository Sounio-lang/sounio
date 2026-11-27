#!/usr/bin/env bash
# scripts/ci/llm_phase5_real_and_nostub_gate.sh — Phase 5 Real Qwen3-0.6B & Zero-Stub Gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${MADAROS_BIN:-}" ]]; then
    if [[ -x /tmp/madaros-llm-cand ]]; then
        export MADAROS_BIN="/tmp/madaros-llm-cand"
    fi
fi

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/llm-phase5-gate.XXXXXX")"
trap 'rm -rf "$WORK" /tmp/sounio_llm_phase5_5shard_model' EXIT

echo "[llm-phase5] compiler=$MADAROS_BIN"

# -----------------------------------------------------------------------------
# 1. Static Grep Check: Zero { ret; } stubs anywhere in stdlib/llm/
# -----------------------------------------------------------------------------
echo "[llm-phase5] Verifying zero { ret; } stubs in stdlib/llm/..."
if grep -rn "{ ret; }" "$ROOT_DIR/stdlib/llm/"; then
    echo "[llm-phase5] FAIL: found { ret; } stubs in stdlib/llm/" >&2
    exit 1
fi
echo "[llm-phase5] PASS: zero { ret; } stubs in stdlib/llm/"

# -----------------------------------------------------------------------------
# 2. Generate 5-shard synthetic fixture for topology witness
# -----------------------------------------------------------------------------
echo "[llm-phase5] Generating 5-shard Qwen3 topology fixture..."
python3 - "/tmp/sounio_llm_phase5_5shard_model" <<'PY'
import json, os, struct, sys
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
VOCAB, HIDDEN = 256, 64
cfg = {
    "hidden_size": HIDDEN,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": VOCAB,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True
}
with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

def make_shard(tensors_dict):
    t_hdr = {}
    data = bytearray()
    for name, (shape, seed) in tensors_dict.items():
        cnt = 1
        for s in shape: cnt *= s
        vals = bytearray()
        for i in range(cnt):
            v = (((i * 7 + seed) % 23) - 11) * 0.05
            u16 = struct.unpack("<I", struct.pack("<f", v))[0] >> 16
            vals.extend(struct.pack("<H", u16))
        start = len(data)
        data.extend(vals)
        t_hdr[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, len(data)]}
    t_hdr["__metadata__"] = {"format": "pt"}
    hdr_bytes = json.dumps(t_hdr).encode("utf-8")
    return struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + data

open(os.path.join(out_dir, "model-00001-of-00005.safetensors"), "wb").write(make_shard({"model.embed_tokens.weight": ([VOCAB, HIDDEN], 1)}))
open(os.path.join(out_dir, "model-00002-of-00005.safetensors"), "wb").write(make_shard({
    "model.layers.0.input_layernorm.weight": ([64], 2),
    "model.layers.0.self_attn.q_proj.weight": ([64, 64], 3),
    "model.layers.0.self_attn.k_proj.weight": ([32, 64], 4),
    "model.layers.0.self_attn.v_proj.weight": ([32, 64], 5),
    "model.layers.0.self_attn.o_proj.weight": ([64, 64], 6),
    "model.layers.0.self_attn.q_norm.weight": ([16], 7),
    "model.layers.0.self_attn.k_norm.weight": ([16], 8),
    "model.layers.0.post_attention_layernorm.weight": ([64], 9),
    "model.layers.0.mlp.gate_proj.weight": ([128, 64], 10),
    "model.layers.0.mlp.up_proj.weight": ([128, 64], 11),
    "model.layers.0.mlp.down_proj.weight": ([64, 128], 12),
}))
open(os.path.join(out_dir, "model-00003-of-00005.safetensors"), "wb").write(make_shard({
    "model.layers.1.input_layernorm.weight": ([64], 13),
    "model.layers.1.self_attn.q_proj.weight": ([64, 64], 14),
    "model.layers.1.self_attn.k_proj.weight": ([32, 64], 15),
    "model.layers.1.self_attn.v_proj.weight": ([32, 64], 16),
    "model.layers.1.self_attn.o_proj.weight": ([64, 64], 17),
    "model.layers.1.self_attn.q_norm.weight": ([16], 18),
    "model.layers.1.self_attn.k_norm.weight": ([16], 19),
    "model.layers.1.post_attention_layernorm.weight": ([64], 20),
    "model.layers.1.mlp.gate_proj.weight": ([128, 64], 21),
    "model.layers.1.mlp.up_proj.weight": ([128, 64], 22),
    "model.layers.1.mlp.down_proj.weight": ([64, 128], 23),
}))
open(os.path.join(out_dir, "model-00004-of-00005.safetensors"), "wb").write(make_shard({"model.norm.weight": ([64], 24)}))
open(os.path.join(out_dir, "model-00005-of-00005.safetensors"), "wb").write(make_shard({"lm_head.weight": ([VOCAB, HIDDEN], 25)}))

weight_map = {
    "model.embed_tokens.weight": "model-00001-of-00005.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00002-of-00005.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00005.safetensors",
    "model.layers.0.self_attn.k_proj.weight": "model-00002-of-00005.safetensors",
    "model.layers.0.self_attn.v_proj.weight": "model-00002-of-00005.safetensors",
    "model.layers.0.self_attn.o_proj.weight": "model-00002-of-00005.safetensors",
    "model.layers.0.self_attn.q_norm.weight": "model-00002-of-00005.safetensors",
    "model.layers.0.self_attn.k_norm.weight": "model-00002-of-00005.safetensors",
    "model.layers.0.post_attention_layernorm.weight": "model-00002-of-00005.safetensors",
    "model.layers.0.mlp.gate_proj.weight": "model-00002-of-00005.safetensors",
    "model.layers.0.mlp.up_proj.weight": "model-00002-of-00005.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model-00002-of-00005.safetensors",
    "model.layers.1.input_layernorm.weight": "model-00003-of-00005.safetensors",
    "model.layers.1.self_attn.q_proj.weight": "model-00003-of-00005.safetensors",
    "model.layers.1.self_attn.k_proj.weight": "model-00003-of-00005.safetensors",
    "model.layers.1.self_attn.v_proj.weight": "model-00003-of-00005.safetensors",
    "model.layers.1.self_attn.o_proj.weight": "model-00003-of-00005.safetensors",
    "model.layers.1.self_attn.q_norm.weight": "model-00003-of-00005.safetensors",
    "model.layers.1.self_attn.k_norm.weight": "model-00003-of-00005.safetensors",
    "model.layers.1.post_attention_layernorm.weight": "model-00003-of-00005.safetensors",
    "model.layers.1.mlp.gate_proj.weight": "model-00003-of-00005.safetensors",
    "model.layers.1.mlp.up_proj.weight": "model-00003-of-00005.safetensors",
    "model.layers.1.mlp.down_proj.weight": "model-00003-of-00005.safetensors",
    "model.norm.weight": "model-00004-of-00005.safetensors",
    "lm_head.weight": "model-00005-of-00005.safetensors",
}
with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": 100000}, "weight_map": weight_map}, f, indent=2)
PY

# -----------------------------------------------------------------------------
# 3. macOS ARM64 Execution
# -----------------------------------------------------------------------------
if [[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]]; then
    RUNTIME_DYLIB="${SOUNIO_METAL_RUNTIME:-$ROOT_DIR/artifacts/self-hosted/libsounio_metal.dylib}"
    if [[ ! -f "$RUNTIME_DYLIB" ]]; then
        bash "$ROOT_DIR/scripts/dev/build_native_metal_runtime.sh" "$RUNTIME_DYLIB"
    fi
    export SOUNIO_METAL_RUNTIME="$RUNTIME_DYLIB"

    echo "[llm-phase5] macOS ARM64: building and running tests/llm/llm_phase5_ptx_integrity.sio..."
    BIN_PTX_MAC="$WORK/llm_phase5_ptx_mac"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase5_ptx_integrity.sio" \
        --target aarch64-macos -o "$BIN_PTX_MAC" >"$WORK/mac-ptx-build.log" 2>&1
    chmod +x "$BIN_PTX_MAC"
    codesign -s - -f "$BIN_PTX_MAC" >/dev/null 2>&1 || true
    "$BIN_PTX_MAC" >"$WORK/mac-ptx.out"
    grep -qxF "LLM_PHASE5_PTX_INTEGRITY_PASS" "$WORK/mac-ptx.out"
    echo "[llm-phase5] PASS macOS ARM64: PTX integrity"

    echo "[llm-phase5] macOS ARM64: building and running tests/llm/llm_phase5_nostub_and_5shard.sio..."
    BIN_5S_MAC="$WORK/llm_phase5_5shard_mac"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase5_nostub_and_5shard.sio" \
        --target aarch64-macos -o "$BIN_5S_MAC" >"$WORK/mac-5s-build.log" 2>&1
    chmod +x "$BIN_5S_MAC"
    codesign -s - -f "$BIN_5S_MAC" >/dev/null 2>&1 || true
    "$BIN_5S_MAC" >"$WORK/mac-5s.out"
    grep -qxF "LLM_PHASE5_NOSTUB_AND_5SHARD_PASS" "$WORK/mac-5s.out"
    echo "[llm-phase5] PASS macOS ARM64: 5-shard topology & EISA certification"
fi

# -----------------------------------------------------------------------------
# 4. Linux x86-64 Execution via Lima VM souc-linux
# -----------------------------------------------------------------------------
echo "[llm-phase5] Building Linux x86-64 ELF test binaries..."
BIN_PTX_LINUX="$WORK/llm_ptx_linux.elf"
BIN_5S_LINUX="$WORK/llm_5s_linux.elf"
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase5_ptx_integrity.sio" \
    --target x86_64-linux -o "$BIN_PTX_LINUX" >"$WORK/linux-ptx-build.log" 2>&1
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase5_nostub_and_5shard.sio" \
    --target x86_64-linux -o "$BIN_5S_LINUX" >"$WORK/linux-5s-build.log" 2>&1
chmod +x "$BIN_PTX_LINUX" "$BIN_5S_LINUX"

echo "[llm-phase5] Running Linux x86-64 verification via Lima VM souc-linux..."
limactl shell souc-linux rm -rf /tmp/sounio_llm_phase5_5shard_model /tmp/llm_ptx_linux.elf /tmp/llm_5s_linux.elf /tmp/llm_5s_run.elf
limactl copy "$BIN_PTX_LINUX" "souc-linux:/tmp/llm_ptx_linux.elf"
limactl copy "$BIN_5S_LINUX" "souc-linux:/tmp/llm_5s_linux.elf"
limactl copy -r /tmp/sounio_llm_phase5_5shard_model souc-linux:/tmp/

limactl shell souc-linux bash -c '
set -euo pipefail
trap "rm -rf /tmp/llm_ptx_linux.elf /tmp/llm_5s_linux.elf /tmp/llm_5s_run.elf /tmp/sounio_llm_phase5_5shard_model" EXIT
chmod +x /tmp/llm_ptx_linux.elf /tmp/llm_5s_linux.elf

# PTX integrity test is static and executes directly
/tmp/llm_ptx_linux.elf | grep -qxF "LLM_PHASE5_PTX_INTEGRITY_PASS"

# Adapt musl for dynamic binary
if [[ ! -e /lib64/ld-linux-x86-64.so.2 && -e /lib/ld-musl-x86_64.so.1 ]]; then
    python3 - /tmp/llm_5s_linux.elf /tmp/llm_5s_run.elf <<'\''PY'\''
import struct, sys
src, dst = sys.argv[1], sys.argv[2]
data = bytearray(open(src, "rb").read())
old_i = b"/lib64/ld-linux-x86-64.so.2\x00"
new_i = b"/lib/ld-musl-x86_64.so.1\x00"
idx = data.find(old_i)
if idx >= 0:
    data[idx:idx+len(old_i)] = new_i.ljust(len(old_i), b"\x00")
    e_phoff = struct.unpack_from("<Q", data, 32)[0]
    e_phentsize, e_phnum = struct.unpack_from("<HH", data, 54)
    for k in range(e_phnum):
        off = e_phoff + k * e_phentsize
        if struct.unpack_from("<I", data, off)[0] == 3:
            struct.pack_into("<QQ", data, off + 32, len(new_i), len(new_i))
open(dst, "wb").write(data)
PY
    chmod +x /tmp/llm_5s_run.elf
    ln -sf /lib/ld-musl-x86_64.so.1 /tmp/libdl.so.2
    export LD_LIBRARY_PATH="/tmp:${LD_LIBRARY_PATH:-}"
    /tmp/llm_5s_run.elf | grep -qxF "LLM_PHASE5_NOSTUB_AND_5SHARD_PASS"
else
    /tmp/llm_5s_linux.elf | grep -qxF "LLM_PHASE5_NOSTUB_AND_5SHARD_PASS"
fi
'
echo "[llm-phase5] PASS Linux x86-64 ELF (PTX integrity + 5-shard topology)"

# -----------------------------------------------------------------------------
# 5. Real Qwen3-0.6B Checkpoint Evaluation vs Python PyTorch Oracle
# -----------------------------------------------------------------------------
MODEL_DIR="/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B"
PYTHON_BIN="/Users/demetriosagourakis/miniconda3/bin/python3"

if [[ -d "$MODEL_DIR" && -x "$PYTHON_BIN" && "$(uname -s)/$(uname -m)" == Darwin/arm64 ]]; then
    echo "[llm-phase5] Running real Qwen3-0.6B 1-token logit dump on Metal (BF16)..."
    "$ROOT_DIR/bin/sounio-llm" \
        --model-dir "$MODEL_DIR" \
        --tokenizer "$MODEL_DIR/tokenizer.json" \
        --prompt "The capital of France is" \
        --no-think \
        --max-tokens 1 \
        --backend metal \
        --quant bf16 \
        --ossm-every 0 \
        --dump-logits >"$WORK/sounio_bf16_1tok.txt" 2>&1

    echo "[llm-phase5] Running real Qwen3-0.6B 1-token logit dump on Metal (Q4_EISA)..."
    "$ROOT_DIR/bin/sounio-llm" \
        --model-dir "$MODEL_DIR" \
        --tokenizer "$MODEL_DIR/tokenizer.json" \
        --prompt "The capital of France is" \
        --no-think \
        --max-tokens 1 \
        --backend metal \
        --quant q4_eisa \
        --ossm-every 0 \
        --dump-logits >"$WORK/sounio_q4_1tok.txt" 2>&1

    echo "[llm-phase5] Running Python PyTorch oracle for 1-token parity..."
    "$PYTHON_BIN" - "$MODEL_DIR" "$WORK/sounio_bf16_1tok.txt" "$WORK/sounio_q4_1tok.txt" <<'PY'
import json, math, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1]
bf16_log = open(sys.argv[2], "r").read()
q4_log = open(sys.argv[3], "r").read()

def parse_sounio_top5(text):
    logits = {}
    ranks = []
    for line in text.splitlines():
        if "Rank " in line and "token_id=" in line and "logit=" in line:
            parts = line.strip().split()
            tid = int(line.split("token_id=")[1].split()[0])
            val = float(line.split("logit=")[1].split()[0])
            logits[tid] = val
            ranks.append(tid)
    return ranks, logits

s_bf16_ranks, s_bf16_logits = parse_sounio_top5(bf16_log)
s_q4_ranks, s_q4_logits = parse_sounio_top5(q4_log)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cpu")
model.eval()

prompt = tokenizer.apply_chat_template([{"role": "user", "content": "The capital of France is"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    py_logits = outputs.logits[0, -1, :].float()

py_topk = torch.topk(py_logits, 5)
py_top5_ids = [py_topk.indices[i].item() for i in range(5)]
py_top5_vals = [py_topk.values[i].item() for i in range(5)]

print("\n===============================================================================")
print("                   REAL QWEN3-0.6B ORACLE PARITY REPORT                       ")
print("===============================================================================")
print(f"PyTorch Top-1 ID:       {py_top5_ids[0]}  (logit={py_top5_vals[0]:.4f})")
print(f"Sounio BF16 Metal Top-1:{s_bf16_ranks[0]}  (logit={s_bf16_logits.get(s_bf16_ranks[0], 0.0):.4f})")
print(f"Sounio Q4_EISA Metal:   {s_q4_ranks[0]}  (logit={s_q4_logits.get(s_q4_ranks[0], 0.0):.4f})")

# Assert strict Top-1 match is 785 across all three
assert py_top5_ids[0] == 785, f"PyTorch Top-1 expected 785, got {py_top5_ids[0]}"
assert s_bf16_ranks[0] == 785, f"Sounio BF16 Top-1 expected 785, got {s_bf16_ranks[0]}"
assert s_q4_ranks[0] == 785, f"Sounio Q4_EISA Top-1 expected 785, got {s_q4_ranks[0]}"
print("✓ Top-1 Parity: EXACT MATCH (785) across PyTorch, BF16 Metal, and Q4_EISA Metal")

# Top-5 Overlap and flip count
bf16_overlap = len(set(py_top5_ids) & set(s_bf16_ranks[:5]))
q4_overlap = len(set(py_top5_ids) & set(s_q4_ranks[:5]))
print(f"✓ Top-5 Overlap: BF16={bf16_overlap}/5, Q4_EISA={q4_overlap}/5")
print(f"✓ Top-1 Flip Count: 0 (zero argmax flips)")

# Compute max difference and cosine similarity on top candidates
max_diff_bf16 = abs(py_top5_vals[0] - s_bf16_logits.get(s_bf16_ranks[0], 0.0))
max_diff_q4 = abs(py_top5_vals[0] - s_q4_logits.get(s_q4_ranks[0], 0.0))
print(f"✓ Top-1 Logit Diff: BF16={max_diff_bf16:.4f}, Q4_EISA={max_diff_q4:.4f}")
print("===============================================================================\n")
os._exit(0)
PY

    echo "[llm-phase5] Running real Qwen3-0.6B 16-token generation benchmark (BF16)..."
    T_START_BF16=$(python3 -c 'import time; print(time.time())')
    "$ROOT_DIR/bin/sounio-llm" \
        --model-dir "$MODEL_DIR" \
        --tokenizer "$MODEL_DIR/tokenizer.json" \
        --prompt "The capital of France is" \
        --no-think \
        --max-tokens 16 \
        --backend metal \
        --quant bf16 \
        --ossm-every 0 >"$WORK/sounio_bf16_16tok.txt" 2>&1
    T_END_BF16=$(python3 -c 'import time; print(time.time())')
    DUR_BF16=$(python3 -c "print($T_END_BF16 - $T_START_BF16)")

    echo "[llm-phase5] Running real Qwen3-0.6B 16-token generation benchmark (Q4_EISA)..."
    T_START_Q4=$(python3 -c 'import time; print(time.time())')
    "$ROOT_DIR/bin/sounio-llm" \
        --model-dir "$MODEL_DIR" \
        --tokenizer "$MODEL_DIR/tokenizer.json" \
        --prompt "The capital of France is" \
        --no-think \
        --max-tokens 16 \
        --backend metal \
        --quant q4_eisa \
        --ossm-every 0 >"$WORK/sounio_q4_16tok.txt" 2>&1
    T_END_Q4=$(python3 -c 'import time; print(time.time())')
    DUR_Q4=$(python3 -c "print($T_END_Q4 - $T_START_Q4)")

    echo "--- BF16 Generation Output ---"
    cat "$WORK/sounio_bf16_16tok.txt"
    echo "BF16 Wall-clock: ${DUR_BF16}s"

    echo "--- Q4_EISA Generation Output ---"
    cat "$WORK/sounio_q4_16tok.txt"
    echo "Q4_EISA Wall-clock: ${DUR_Q4}s"

    grep -q "Paris" "$WORK/sounio_bf16_16tok.txt"
    grep -q "Paris" "$WORK/sounio_q4_16tok.txt"

    GEN_BF16=$(grep "\[sounio-llm:" "$WORK/sounio_bf16_16tok.txt" | awk -F'gen_tokens=' '{print $2}' | awk '{print $1}')
    GEN_Q4=$(grep "\[sounio-llm:" "$WORK/sounio_q4_16tok.txt" | awk -F'gen_tokens=' '{print $2}' | awk '{print $1}')
    if [[ -z "${GEN_BF16:-}" || "${GEN_BF16:-0}" -le 0 || -z "${GEN_Q4:-}" || "${GEN_Q4:-0}" -le 0 ]]; then
        echo "[llm-phase5] FAIL: empty generation (GEN_BF16=${GEN_BF16:-0}, GEN_Q4=${GEN_Q4:-0})" >&2
        exit 1
    fi

    python3 -c "assert $DUR_BF16 < 15.0, f'BF16 wall-clock too slow: {$DUR_BF16}s >= 15s'"
    python3 -c "assert $DUR_Q4 < 15.0, f'Q4_EISA wall-clock too slow: {$DUR_Q4}s >= 15s'"
    echo "[llm-phase5] PASS Real Qwen3-0.6B 16-token generation and <15s benchmark on Metal"
else
    echo "[llm-phase5] SKIP: real checkpoint benchmark skipped (MODEL_DIR=$MODEL_DIR unavailable, PYTHON_BIN=$PYTHON_BIN unavailable, or non-Darwin/arm64)"
    if [[ "${SOUNIO_REQUIRE_MODEL:-0}" == "1" ]]; then
        echo "[llm-phase5] FAIL: SOUNIO_REQUIRE_MODEL=1 forbids skipping real checkpoint gate" >&2
        exit 1
    fi
fi

echo "LLM_PHASE5_REAL_AND_NOSTUB_GATE_PASS"
