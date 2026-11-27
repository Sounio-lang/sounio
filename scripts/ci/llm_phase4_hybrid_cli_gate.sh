#!/usr/bin/env bash
# Phase 4 Hybrid O-SSM & CLI Gate:
#   1. Non-associative Cayley-Dickson octonion state space algebra witness (real=0, oct>0).
#   2. O(1) context state memory footprint verification (36 layers * 4096 hidden * 4 = 589824 bytes).
#   3. Exact CPU vs GPU parity (< 1e-5 on Metal NoCopy & Linux CUDA PTX driver ABI).
#   4. Interleaved Hybrid Qwen3 GQA + O-SSM multi-token chat generation (--think & --no-think).
#   5. Sovereign bin/sounio-llm CLI driver (--version, --help, chat generation with telemetry).
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

WORK="$(mktemp -d "${TMPDIR:-/tmp}/llm-phase4-gate.XXXXXX")"
MODEL_DIR="/tmp/sounio_llm_phase4_model"
TOK_FILE="/tmp/sounio_llm_phase4_tok.json"

trap 'rm -rf "$WORK" "$MODEL_DIR" "$TOK_FILE" /tmp/sounio_llm_cli_* /tmp/sounio-llm-cache' EXIT

echo "[llm-phase4] compiler=$MADAROS_BIN"

create_fixtures() {
    local m_dir="$1"
    local t_file="$2"
    mkdir -p "$m_dir"

    python3 - "$m_dir" "$t_file" <<'PY'
import json, math, os, struct, sys

out_dir, tok_file = sys.argv[1], sys.argv[2]

VOCAB = 151936
HIDDEN = 64
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

def make_embed_bf16(vocab, cols):
    base = []
    for c in range(cols):
        val = (((c * 5) % 23) - 11) * 0.0625
        b = struct.pack("<f", float(val))
        u32 = struct.unpack("<I", b)[0]
        base.append(u32 >> 16)

    base_bytes = bytearray(struct.pack(f"<{cols}H", *base))
    rows_bytes = bytearray()
    for r in range(vocab):
        val0 = -2.0 + (r * 4.0 / vocab)
        b = struct.pack("<f", float(val0))
        u16_0 = struct.unpack("<I", b)[0] >> 16
        base_bytes[0] = u16_0 & 0xFF
        base_bytes[1] = (u16_0 >> 8) & 0xFF
        rows_bytes.extend(base_bytes)
    return rows_bytes

def make_bf16_bytes_varied(rows, cols, seed):
    templates = []
    for r in range(23):
        t = []
        for c in range(cols):
            val = (((c * 5 + r * 7 + seed) % 23) - 11) * 0.0625 + (0.125 if (c % 8) == (r % 8) else 0.0)
            b = struct.pack("<f", float(val))
            u32 = struct.unpack("<I", b)[0]
            t.append(u32 >> 16)
        templates.append(struct.pack(f"<{cols}H", *t))
    full_bytes = bytearray()
    for r in range(rows):
        full_bytes.extend(templates[(r + seed) % 23])
    return full_bytes

s1_tensors = {}
s1_bytes = bytearray()

def add_tensor(s_tensors, s_bytes, name, shape, seed):
    if len(shape) == 1:
        rows, cols = 1, shape[0]
    else:
        rows, cols = shape[0], shape[1]
    payload = make_bf16_bytes_varied(rows, cols, seed)
    start = len(s_bytes)
    end = start + len(payload)
    s_bytes.extend(payload)
    s_tensors[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, end]}

# Embeddings: strictly unique row weights
embed_payload = make_embed_bf16(VOCAB, HIDDEN)
emb_start = len(s1_bytes)
emb_end = emb_start + len(embed_payload)
s1_bytes.extend(embed_payload)
s1_tensors["model.embed_tokens.weight"] = {"dtype": "BF16", "shape": [VOCAB, HIDDEN], "data_offsets": [emb_start, emb_end]}
s1_tensors["lm_head.weight"] = {"dtype": "BF16", "shape": [VOCAB, HIDDEN], "data_offsets": [emb_start, emb_end]}

# Shard 1: layer 0
add_tensor(s1_tensors, s1_bytes, "model.layers.0.input_layernorm.weight", [64], 2)
add_tensor(s1_tensors, s1_bytes, "model.layers.0.self_attn.q_proj.weight", [64, 64], 3)
add_tensor(s1_tensors, s1_bytes, "model.layers.0.self_attn.k_proj.weight", [32, 64], 4)
add_tensor(s1_tensors, s1_bytes, "model.layers.0.self_attn.v_proj.weight", [32, 64], 5)
add_tensor(s1_tensors, s1_bytes, "model.layers.0.self_attn.o_proj.weight", [64, 64], 6)
add_tensor(s1_tensors, s1_bytes, "model.layers.0.self_attn.q_norm.weight", [16], 7)
add_tensor(s1_tensors, s1_bytes, "model.layers.0.self_attn.k_norm.weight", [16], 8)
add_tensor(s1_tensors, s1_bytes, "model.layers.0.post_attention_layernorm.weight", [64], 9)
add_tensor(s1_tensors, s1_bytes, "model.layers.0.mlp.gate_proj.weight", [128, 64], 10)
add_tensor(s1_tensors, s1_bytes, "model.layers.0.mlp.up_proj.weight", [128, 64], 11)
add_tensor(s1_tensors, s1_bytes, "model.layers.0.mlp.down_proj.weight", [64, 128], 12)

s1_tensors["__metadata__"] = {"format": "pt"}
s1_hdr_bytes = json.dumps(s1_tensors).encode("utf-8")
with open(os.path.join(out_dir, "model-00001-of-00002.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(s1_hdr_bytes)))
    f.write(s1_hdr_bytes)
    f.write(s1_bytes)

# Shard 2: layer 1 + model.norm.weight
s2_tensors = {}
s2_bytes = bytearray()
add_tensor(s2_tensors, s2_bytes, "model.layers.1.input_layernorm.weight", [64], 13)
add_tensor(s2_tensors, s2_bytes, "model.layers.1.self_attn.q_proj.weight", [64, 64], 14)
add_tensor(s2_tensors, s2_bytes, "model.layers.1.self_attn.k_proj.weight", [32, 64], 15)
add_tensor(s2_tensors, s2_bytes, "model.layers.1.self_attn.v_proj.weight", [32, 64], 16)
add_tensor(s2_tensors, s2_bytes, "model.layers.1.self_attn.o_proj.weight", [64, 64], 17)
add_tensor(s2_tensors, s2_bytes, "model.layers.1.self_attn.q_norm.weight", [16], 18)
add_tensor(s2_tensors, s2_bytes, "model.layers.1.self_attn.k_norm.weight", [16], 19)
add_tensor(s2_tensors, s2_bytes, "model.layers.1.post_attention_layernorm.weight", [64], 20)
add_tensor(s2_tensors, s2_bytes, "model.layers.1.mlp.gate_proj.weight", [128, 64], 21)
add_tensor(s2_tensors, s2_bytes, "model.layers.1.mlp.up_proj.weight", [128, 64], 22)
add_tensor(s2_tensors, s2_bytes, "model.layers.1.mlp.down_proj.weight", [64, 128], 23)
add_tensor(s2_tensors, s2_bytes, "model.norm.weight", [64], 24)

s2_tensors["__metadata__"] = {"format": "pt"}
s2_hdr_bytes = json.dumps(s2_tensors).encode("utf-8")
with open(os.path.join(out_dir, "model-00002-of-00002.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(s2_hdr_bytes)))
    f.write(s2_hdr_bytes)
    f.write(s2_bytes)

# Index
weight_map = {}
for k in s1_tensors:
    if k != "__metadata__": weight_map[k] = "model-00001-of-00002.safetensors"
for k in s2_tensors:
    if k != "__metadata__": weight_map[k] = "model-00002-of-00002.safetensors"

idx = {"metadata": {"total_size": len(s1_bytes) + len(s2_bytes)}, "weight_map": weight_map}
with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(idx, f, indent=2)

# Tokenizer: use host checkpoint if present, else fallback
host_checkpoint_tok = "/Users/demetriosagourakis/sounio-research/checkpoints/sounio-coder-3b/tokenizer.json"
if os.path.exists(host_checkpoint_tok):
    with open(host_checkpoint_tok, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
else:
    vocab = {}
    for i in range(256):
        vocab[chr(i) if i >= 32 and i <= 126 else f"<0x{i:02X}>"] = i
    vocab["<|im_start|>"] = 151644
    vocab["<|im_end|>"] = 151645
    vocab["<|endoftext|>"] = 151643
    vocab["<think>"] = 151667
    vocab["</think>"] = 151668

    added_tokens = [
        {"content": "<|im_start|>", "id": 151644, "special": True},
        {"content": "<|im_end|>", "id": 151645, "special": True},
        {"content": "<|endoftext|>", "id": 151643, "special": True},
        {"content": "<think>", "id": 151667, "special": True},
        {"content": "</think>", "id": 151668, "special": True},
    ]

    tok_data = {
        "version": "1.0",
        "added_tokens": added_tokens,
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": []
        }
    }

with open(tok_file, "w", encoding="utf-8") as f:
    json.dump(tok_data, f, indent=2)
PY
}

echo "[llm-phase4] Generating model and tokenizer fixtures..."
create_fixtures "$MODEL_DIR" "$TOK_FILE"

# macOS Native ARM64 + Metal NoCopy Verification
if [[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]]; then
    RUNTIME_DYLIB="${SOUNIO_METAL_RUNTIME:-$ROOT_DIR/artifacts/self-hosted/libsounio_metal.dylib}"
    if [[ ! -f "$RUNTIME_DYLIB" ]]; then
        bash "$ROOT_DIR/scripts/dev/build_native_metal_runtime.sh" "$RUNTIME_DYLIB"
    fi
    export SOUNIO_METAL_RUNTIME="$RUNTIME_DYLIB"

    BIN_MAC="$WORK/llm_phase4_hybrid_cli_mac"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase4_hybrid_cli.sio" \
        --target aarch64-macos -o "$BIN_MAC" >"$WORK/mac-build.log" 2>&1
    if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body' "$WORK/mac-build.log"; then
        echo "[llm-phase4] FAIL: macOS build refused or missed lowered body" >&2
        cat "$WORK/mac-build.log" >&2
        exit 1
    fi
    chmod +x "$BIN_MAC"
    codesign -s - -f "$BIN_MAC"
    codesign --verify "$BIN_MAC"

    "$BIN_MAC" >"$WORK/mac-run.stdout" 2>"$WORK/mac-run.stderr"
    grep -qxF "LLM_PHASE4_HYBRID_CLI_PASS" "$WORK/mac-run.stdout"
    echo "[llm-phase4] PASS macOS ARM64 Metal NoCopy (O-SSM + Hybrid Chat Generation)"
fi

# Linux x86-64 ELF Build & Execution
BIN_LINUX="$WORK/llm_phase4_hybrid_cli_linux.elf"
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase4_hybrid_cli.sio" \
    --target x86_64-linux -o "$BIN_LINUX" >"$WORK/linux-build.log" 2>&1
if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body' "$WORK/linux-build.log"; then
    echo "[llm-phase4] FAIL: Linux build refused or missed lowered body" >&2
    cat "$WORK/linux-build.log" >&2
    exit 1
fi
chmod +x "$BIN_LINUX"

run_linux_verification() {
    local runner_prefix="$1"
    local vm_work="/tmp/llm-phase4-vm-$$"
    if [[ "$runner_prefix" == "local" ]]; then
        readelf -d "$BIN_LINUX" >"$WORK/linux-dyn.txt"
        grep -q "libdl.so.2" "$WORK/linux-dyn.txt"
        local needed_cnt
        needed_cnt="$(grep -c "Shared library:" "$WORK/linux-dyn.txt" || true)"
        test "$needed_cnt" -eq 1

        if [[ ! -e /dev/nvidiactl ]]; then
            gcc -shared -fPIC -O2 -nostdlib \
                "$ROOT_DIR/tests/fixtures/llm/cuda_driver_mock_witness.c" \
                -o "$WORK/libcuda_mock.so"
            export SOUNIO_CUDA_DRIVER_LIB="$WORK/libcuda_mock.so"
            echo "[llm-phase4] linux mode=cpu_mock_driver (no /dev/nvidiactl)"
        else
            echo "[llm-phase4] linux mode=real_nvidia_driver"
        fi

        local exec_lin="$BIN_LINUX"
        if [[ ! -e /lib64/ld-linux-x86-64.so.2 && -e /lib/ld-musl-x86_64.so.1 ]]; then
            ln -sf /lib/ld-musl-x86_64.so.1 "$WORK/libdl.so.2"
            export LD_LIBRARY_PATH="$WORK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            python3 - "$BIN_LINUX" "$WORK/run_musl.elf" <<'PY'
import struct, sys
src, dst = sys.argv[1], sys.argv[2]
data = bytearray(open(src, "rb").read())
old_i = b"/lib64/ld-linux-x86-64.so.2\x00"
new_i = b"/lib/ld-musl-x86_64.so.1\x00"
idx = data.find(old_i)
assert idx >= 0, "PT_INTERP string not found"
data[idx:idx+len(old_i)] = new_i.ljust(len(old_i), b"\x00")
e_phoff = struct.unpack_from("<Q", data, 32)[0]
e_phentsize, e_phnum = struct.unpack_from("<HH", data, 54)
for k in range(e_phnum):
    off = e_phoff + k * e_phentsize
    if struct.unpack_from("<I", data, off)[0] == 3:
        struct.pack_into("<QQ", data, off + 32, len(new_i), len(new_i))
open(dst, "wb").write(data)
PY
            chmod +x "$WORK/run_musl.elf"
            exec_lin="$WORK/run_musl.elf"
            echo "[llm-phase4] linux interp=musl_pt_interp_adapter"
        fi
        "$exec_lin" >"$WORK/linux-run.stdout" 2>"$WORK/linux-run.stderr"
        grep -qxF "LLM_PHASE4_HYBRID_CLI_PASS" "$WORK/linux-run.stdout"
    else
        limactl shell souc-linux bash -c "mkdir -p '$vm_work'"
        limactl copy "$BIN_LINUX" "souc-linux:$vm_work/llm_phase4_hybrid_cli_linux.elf"
        limactl copy "$ROOT_DIR/tests/fixtures/llm/cuda_driver_mock_witness.c" "souc-linux:$vm_work/cuda_driver_mock_witness.c"
        limactl shell souc-linux bash -c "mkdir -p /tmp/sounio_llm_phase4_model"
        limactl copy "$MODEL_DIR/config.json" "souc-linux:/tmp/sounio_llm_phase4_model/config.json"
        limactl copy "$MODEL_DIR/model.safetensors.index.json" "souc-linux:/tmp/sounio_llm_phase4_model/model.safetensors.index.json"
        limactl copy "$MODEL_DIR/model-00001-of-00002.safetensors" "souc-linux:/tmp/sounio_llm_phase4_model/model-00001-of-00002.safetensors"
        limactl copy "$MODEL_DIR/model-00002-of-00002.safetensors" "souc-linux:/tmp/sounio_llm_phase4_model/model-00002-of-00002.safetensors"
        limactl copy "$TOK_FILE" "souc-linux:/tmp/sounio_llm_phase4_tok.json"

        limactl shell souc-linux bash -s -- "$vm_work" <<'EOS'
set -euo pipefail
VM_WORK="$1"
trap 'rm -rf "$VM_WORK" /tmp/sounio_llm_phase4_model /tmp/sounio_llm_phase4_tok.json' EXIT

chmod +x "$VM_WORK/llm_phase4_hybrid_cli_linux.elf"
readelf -d "$VM_WORK/llm_phase4_hybrid_cli_linux.elf" > "$VM_WORK/dyn.txt"
grep -q "libdl.so.2" "$VM_WORK/dyn.txt"
NEEDED_CNT="$(grep -c "Shared library:" "$VM_WORK/dyn.txt" || true)"
test "$NEEDED_CNT" -eq 1

if [[ ! -e /dev/nvidiactl ]]; then
    gcc -shared -fPIC -O2 -nostdlib "$VM_WORK/cuda_driver_mock_witness.c" -o "$VM_WORK/libcuda_mock.so"
    export SOUNIO_CUDA_DRIVER_LIB="$VM_WORK/libcuda_mock.so"
    echo "[llm-phase4] linux mode=lima_cpu_mock_driver (no /dev/nvidiactl)"
else
    echo "[llm-phase4] linux mode=lima_real_nvidia_driver"
fi

EXEC_BIN="$VM_WORK/llm_phase4_hybrid_cli_linux.elf"
if [[ ! -e /lib64/ld-linux-x86-64.so.2 && -e /lib/ld-musl-x86_64.so.1 ]]; then
    ln -sf /lib/ld-musl-x86_64.so.1 "$VM_WORK/libdl.so.2"
    export LD_LIBRARY_PATH="$VM_WORK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    python3 - "$VM_WORK/llm_phase4_hybrid_cli_linux.elf" "$VM_WORK/run_musl.elf" <<'PY'
import struct, sys
src, dst = sys.argv[1], sys.argv[2]
data = bytearray(open(src, "rb").read())
old_i = b"/lib64/ld-linux-x86-64.so.2\x00"
new_i = b"/lib/ld-musl-x86_64.so.1\x00"
idx = data.find(old_i)
assert idx >= 0, "PT_INTERP string not found"
data[idx:idx+len(old_i)] = new_i.ljust(len(old_i), b"\x00")
e_phoff = struct.unpack_from("<Q", data, 32)[0]
e_phentsize, e_phnum = struct.unpack_from("<HH", data, 54)
for k in range(e_phnum):
    off = e_phoff + k * e_phentsize
    if struct.unpack_from("<I", data, off)[0] == 3:
        struct.pack_into("<QQ", data, off + 32, len(new_i), len(new_i))
open(dst, "wb").write(data)
PY
    chmod +x "$VM_WORK/run_musl.elf"
    EXEC_BIN="$VM_WORK/run_musl.elf"
    echo "[llm-phase4] linux interp=musl_pt_interp_adapter"
fi

"$EXEC_BIN" >"$VM_WORK/out.txt"
grep -qxF "LLM_PHASE4_HYBRID_CLI_PASS" "$VM_WORK/out.txt"
EOS
    fi
    echo "[llm-phase4] PASS Linux x86-64 ELF + PTX + CUDA Driver ABI (O-SSM + Hybrid Chat Generation)"
}

if [[ "$(uname -s)" == Linux ]]; then
    run_linux_verification "local"
elif command -v limactl >/dev/null 2>&1 && limactl list 2>/dev/null | grep -q "^souc-linux[[:space:]]\+Running"; then
    run_linux_verification "lima"
else
    echo "[llm-phase4] ERROR: neither Linux host nor running Lima VM 'souc-linux' available" >&2
    exit 2
fi

# Sovereign bin/sounio-llm CLI Verification
echo "[llm-phase4] Verifying bin/sounio-llm CLI..."
"$ROOT_DIR/bin/sounio-llm" --version >"$WORK/cli-version.txt"
grep -q "sounio-llm 0.80.0" "$WORK/cli-version.txt"

"$ROOT_DIR/bin/sounio-llm" --help >"$WORK/cli-help.txt"
grep -q "Usage: sounio-llm" "$WORK/cli-help.txt"

echo "[llm-phase4] Running chat generation with thinking enabled..."
"$ROOT_DIR/bin/sounio-llm" \
    --prompt "soma" \
    --think \
    --max-tokens 2 \
    --ossm-every 2 >"$WORK/cli-chat-think.txt" 2>&1
grep -q "\[sounio-llm: prompt_tokens=" "$WORK/cli-chat-think.txt"

echo "[llm-phase4] Running chat generation with thinking disabled..."
"$ROOT_DIR/bin/sounio-llm" \
    --prompt "soma" \
    --no-think \
    --max-tokens 2 \
    --ossm-every 2 >"$WORK/cli-chat-nothink.txt" 2>&1
grep -q "\[sounio-llm: prompt_tokens=" "$WORK/cli-chat-nothink.txt"

echo "LLM_PHASE4_HYBRID_CLI_GATE_PASS"
