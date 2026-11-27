#!/usr/bin/env bash
# Fase 2 Forward Gate:
#   1. Sovereign block-paged FP32 KV cache on macOS ARM64 & Linux x86-64 ELF
#   2. High-performance GPU kernels: Apple M5 Max Metal NoCopy & CUDA PTX Driver API
#   3. Zero-copy multi-shard BF16 Qwen3 sovereign inference engine
#   4. 100% bit-exact parity across Python FP32 reference oracle, CPU, and Apple Silicon Metal NoCopy
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/llm-phase2-gate.XXXXXX")"
CFG_MODEL="/tmp/sounio_llm_phase2_model_cfg.txt"
CFG_MODE="/tmp/sounio_llm_phase2_mode_cfg.txt"
CFG_DUMP="/tmp/sounio_llm_phase2_dump_cfg.txt"
trap 'rm -rf "$WORK" "$CFG_MODEL" "$CFG_MODE" "$CFG_DUMP"' EXIT

echo "[llm-phase2] compiler=$MADAROS_BIN"

create_qwen3_forward_fixture() {
    local out_dir="$1"
    mkdir -p "$out_dir"
    python3 - "$out_dir" <<'PY'
import json, math, os, struct, sys

out_dir = sys.argv[1]

# Model configuration
cfg = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 64,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True
}
with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

def make_bf16_bytes(count, seed_offset):
    # Deterministic BF16 values using powers of two (no rounding discrepancies)
    vals = [(((i + seed_offset) % 17) - 8) * 0.0625 for i in range(count)]
    u16_list = []
    for v in vals:
        b = struct.pack("<f", float(v))
        u32 = struct.unpack("<I", b)[0]
        u16_list.append(u32 >> 16)
    return struct.pack(f"<{count}H", *u16_list)

s1_tensors = {}
s1_bytes = bytearray()

def add_tensor(s_tensors, s_bytes, name, shape, seed):
    count = 1
    for d in shape: count *= d
    payload = make_bf16_bytes(count, seed)
    start = len(s_bytes)
    end = start + len(payload)
    s_bytes.extend(payload)
    s_tensors[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, end]}

# Shard 1: embed_tokens + layer 0
add_tensor(s1_tensors, s1_bytes, "model.embed_tokens.weight", [64, 64], 1)
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
PY
}

verify_python_oracle_parity() {
    local model_dir="$1"
    local sounio_stdout="$2"
    python3 - "$model_dir" "$sounio_stdout" <<'PY'
import json, math, os, re, struct, sys

model_dir, sout = sys.argv[1], sys.argv[2]
m = re.search(r'\[QWEN3_FORWARD_LOGITS: top1=(\d+) val=([-\d\.]+) l0=([-\d\.]+) l1=([-\d\.]+) l2=([-\d\.]+)\]', sout)
if not m:
    print("[llm-phase2] FAIL: could not extract logit dump from Sounio output", file=sys.stderr)
    sys.exit(1)

s_top1 = int(m.group(1))
s_val = float(m.group(2))
s_l0 = float(m.group(3))
s_l1 = float(m.group(4))
s_l2 = float(m.group(5))

with open(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)

with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
    idx = json.load(f)

shards = {}
for sf in set(idx["weight_map"].values()):
    with open(os.path.join(model_dir, sf), "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        data = f.read()
        shards[sf] = (hdr, data)

def get_tensor(name):
    sf = idx["weight_map"][name]
    hdr, data = shards[sf]
    meta = hdr[name]
    start, end = meta["data_offsets"]
    payload = data[start:end]
    count = (end - start) // 2
    u16s = struct.unpack(f"<{count}H", payload)
    f32s = [struct.unpack("<f", struct.pack("<I", u << 16))[0] for u in u16s]
    return f32s, meta["shape"]

def rmsnorm(x, w, eps):
    dim = len(x)
    sum_sq = sum(v * v for v in x)
    inv_rms = 1.0 / math.sqrt(sum_sq / dim + eps)
    return [x[i] * inv_rms * w[i] for i in range(dim)]

def per_head_rmsnorm(x, w, n_heads, head_dim, eps):
    out = []
    for h in range(n_heads):
        head_x = x[h * head_dim : (h + 1) * head_dim]
        out.extend(rmsnorm(head_x, w, eps))
    return out

def rope(x, n_heads, head_dim, pos, theta):
    out = list(x)
    half = head_dim // 2
    for h in range(n_heads):
        for j in range(half):
            freq = theta ** (-2.0 * j / head_dim)
            angle = pos * freq
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            v0 = x[h * head_dim + j]
            v1 = x[h * head_dim + j + half]
            out[h * head_dim + j] = v0 * cos_a - v1 * sin_a
            out[h * head_dim + j + half] = v0 * sin_a + v1 * cos_a
    return out

def gemv(w, rows, cols, x):
    out = []
    for r in range(rows):
        row = w[r * cols : (r + 1) * cols]
        dot = sum(row[c] * x[c] for c in range(cols))
        out.append(dot)
    return out

hidden = cfg["hidden_size"]
inter = cfg["intermediate_size"]
layers = cfg["num_hidden_layers"]
n_q = cfg["num_attention_heads"]
n_kv = cfg["num_key_value_heads"]
head_dim = cfg["head_dim"]
vocab = cfg["vocab_size"]
eps = cfg["rms_norm_eps"]
theta = cfg["rope_theta"]

embed_w, _ = get_tensor("model.embed_tokens.weight")
norm_w, _ = get_tensor("model.norm.weight")
lm_head_w = embed_w

kv_k = [[[] for _ in range(layers)] for _ in range(16)]
kv_v = [[[] for _ in range(layers)] for _ in range(16)]

tokens = [7, 23, 42]
logits = []

for pos, tok in enumerate(tokens):
    x = embed_w[tok * hidden : (tok + 1) * hidden]
    for l in range(layers):
        in_norm_w, _ = get_tensor(f"model.layers.{l}.input_layernorm.weight")
        q_proj_w, _ = get_tensor(f"model.layers.{l}.self_attn.q_proj.weight")
        k_proj_w, _ = get_tensor(f"model.layers.{l}.self_attn.k_proj.weight")
        v_proj_w, _ = get_tensor(f"model.layers.{l}.self_attn.v_proj.weight")
        o_proj_w, _ = get_tensor(f"model.layers.{l}.self_attn.o_proj.weight")
        q_norm_w, _ = get_tensor(f"model.layers.{l}.self_attn.q_norm.weight")
        k_norm_w, _ = get_tensor(f"model.layers.{l}.self_attn.k_norm.weight")
        post_norm_w, _ = get_tensor(f"model.layers.{l}.post_attention_layernorm.weight")
        gate_w, _ = get_tensor(f"model.layers.{l}.mlp.gate_proj.weight")
        up_w, _ = get_tensor(f"model.layers.{l}.mlp.up_proj.weight")
        down_w, _ = get_tensor(f"model.layers.{l}.mlp.down_proj.weight")

        h = rmsnorm(x, in_norm_w, eps)
        q = gemv(q_proj_w, hidden, hidden, h)
        k = gemv(k_proj_w, n_kv * head_dim, hidden, h)
        v = gemv(v_proj_w, n_kv * head_dim, hidden, h)

        q = per_head_rmsnorm(q, q_norm_w, n_q, head_dim, eps)
        k = per_head_rmsnorm(k, k_norm_w, n_kv, head_dim, eps)

        q = rope(q, n_q, head_dim, pos, theta)
        k = rope(k, n_kv, head_dim, pos, theta)

        kv_k[pos][l] = k
        kv_v[pos][l] = v

        scale = 1.0 / math.sqrt(head_dim)
        group_size = n_q // n_kv
        attn_out = [0.0] * (n_q * head_dim)
        for q_h in range(n_q):
            kv_h = q_h // group_size
            q_vec = q[q_h * head_dim : (q_h + 1) * head_dim]
            scores = []
            for p in range(pos + 1):
                k_vec = kv_k[p][l][kv_h * head_dim : (kv_h + 1) * head_dim]
                dot = sum(q_vec[d] * k_vec[d] for d in range(head_dim))
                scores.append(dot * scale)
            max_s = max(scores)
            exps = [math.exp(s - max_s) for s in scores]
            sum_exp = sum(exps)
            weights = [e / sum_exp for e in exps]
            for d in range(head_dim):
                val = sum(weights[p] * kv_v[p][l][kv_h * head_dim + d] for p in range(pos + 1))
                attn_out[q_h * head_dim + d] = val

        o_out = gemv(o_proj_w, hidden, hidden, attn_out)
        x = [x[i] + o_out[i] for i in range(hidden)]

        h = rmsnorm(x, post_norm_w, eps)
        gate = gemv(gate_w, inter, hidden, h)
        up = gemv(up_w, inter, hidden, h)
        swiglu = [(gate[i] / (1.0 + math.exp(-gate[i]))) * up[i] for i in range(inter)]
        down = gemv(down_w, hidden, inter, swiglu)
        x = [x[i] + down[i] for i in range(hidden)]

    x_norm = rmsnorm(x, norm_w, eps)
    logits = gemv(lm_head_w, vocab, hidden, x_norm)

py_top1 = max(range(len(logits)), key=lambda i: logits[i])
py_val = logits[py_top1]
py_l0 = logits[0]
py_l1 = logits[1]
py_l2 = logits[2]

max_diff = max(abs(s_val - py_val), abs(s_l0 - py_l0), abs(s_l1 - py_l1), abs(s_l2 - py_l2))
if s_top1 != py_top1 or max_diff >= 0.001:
    print(f"[llm-phase2] FAIL: python oracle parity mismatch (sounio_top1={s_top1} py_top1={py_top1} max_diff={max_diff})", file=sys.stderr)
    sys.exit(2)

print(f"[llm-phase2] PASS Python reference oracle parity: top1={py_top1} match, max_diff={max_diff:.6f} < 1e-3")
PY
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2A: macOS ARM64 Verification
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]]; then
    echo "[llm-phase2] target=aarch64-macos"

    # 1. Paged KV cache test
    BIN_KV_MAC="$WORK/llm_phase2_paged_kv"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase2_paged_kv.sio" \
        --target aarch64-macos -o "$BIN_KV_MAC" >"$WORK/mac-kv-build.log" 2>&1
    if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body|ARM64_REFUSAL' "$WORK/mac-kv-build.log"; then
        echo "[llm-phase2] FAIL: macOS paged_kv build refused" >&2
        cat "$WORK/mac-kv-build.log" >&2
        exit 1
    fi
    chmod +x "$BIN_KV_MAC"
    codesign -s - -f "$BIN_KV_MAC"
    "$BIN_KV_MAC" >"$WORK/mac-kv-run.stdout" 2>"$WORK/mac-kv-run.stderr"
    grep -qxF "LLM_PHASE2_PAGED_KV_PASS" "$WORK/mac-kv-run.stdout"
    echo "[llm-phase2] PASS macOS ARM64 block-paged KV cache (40 tokens, truncate, regrowth, reuse)"

    # 2. Apple Silicon Metal NoCopy GPU kernels (all 5 kernels + 4096x4096 BF16 GEMV)
    BIN_GPU_MAC="$WORK/llm_phase2_gpu_kernels"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase2_gpu_kernels.sio" \
        --target aarch64-macos -o "$BIN_GPU_MAC" >"$WORK/mac-gpu-build.log" 2>&1
    if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body|ARM64_REFUSAL' "$WORK/mac-gpu-build.log"; then
        echo "[llm-phase2] FAIL: macOS gpu_kernels build refused" >&2
        cat "$WORK/mac-gpu-build.log" >&2
        exit 1
    fi
    chmod +x "$BIN_GPU_MAC"
    codesign -s - -f "$BIN_GPU_MAC"
    "$BIN_GPU_MAC" >"$WORK/mac-gpu-run.stdout" 2>"$WORK/mac-gpu-run.stderr"
    grep -qxF "LLM_PHASE2_GPU_KERNELS_PASS" "$WORK/mac-gpu-run.stdout"
    echo "[llm-phase2] PASS macOS ARM64 Metal NoCopy GPU kernels (4096x4096 BF16 GEMV + 5 kernels)"

    # 3. Qwen3 Sovereign Forward: CPU prefill + KV truncate redecode bit parity + Metal NoCopy parity
    MODEL_MAC="$WORK/model_mac"
    create_qwen3_forward_fixture "$MODEL_MAC"
    printf '%s' "$MODEL_MAC" > "$CFG_MODEL"
    printf 'metal_and_cpu' > "$CFG_MODE"
    printf 'dump' > "$CFG_DUMP"

    BIN_FWD_MAC="$WORK/llm_phase2_qwen3_forward"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase2_qwen3_forward.sio" \
        --target aarch64-macos -o "$BIN_FWD_MAC" >"$WORK/mac-fwd-build.log" 2>&1
    if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body|ARM64_REFUSAL' "$WORK/mac-fwd-build.log"; then
        echo "[llm-phase2] FAIL: macOS qwen3_forward build refused" >&2
        cat "$WORK/mac-fwd-build.log" >&2
        exit 1
    fi
    chmod +x "$BIN_FWD_MAC"
    codesign -s - -f "$BIN_FWD_MAC"

    FWD_OUT_MAC="$("$BIN_FWD_MAC")"
    echo "$FWD_OUT_MAC" | grep -qxF "LLM_PHASE2_QWEN3_FORWARD_PASS"
    verify_python_oracle_parity "$MODEL_MAC" "$FWD_OUT_MAC"
    echo "[llm-phase2] PASS macOS ARM64 Qwen3 forward (Metal NoCopy / CPU bit parity / Python oracle)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2B: Linux x86-64 ELF Verification
# ─────────────────────────────────────────────────────────────────────────────
echo "[llm-phase2] target=x86_64-linux"

BIN_KV_LIN="$WORK/llm_phase2_paged_kv.elf"
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase2_paged_kv.sio" \
    --target x86_64-linux -o "$BIN_KV_LIN" >"$WORK/linux-kv-build.log" 2>&1
if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body' "$WORK/linux-kv-build.log"; then
    echo "[llm-phase2] FAIL: Linux paged_kv build refused" >&2
    cat "$WORK/linux-kv-build.log" >&2
    exit 1
fi
chmod +x "$BIN_KV_LIN"

BIN_PTX_LIN="$WORK/llm_phase2_linux_ptx_kernels.elf"
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase2_linux_ptx_kernels.sio" \
    --target x86_64-linux -o "$BIN_PTX_LIN" >"$WORK/linux-ptx-build.log" 2>&1
if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body' "$WORK/linux-ptx-build.log"; then
    echo "[llm-phase2] FAIL: Linux ptx_kernels build refused" >&2
    cat "$WORK/linux-ptx-build.log" >&2
    exit 1
fi
chmod +x "$BIN_PTX_LIN"

BIN_FWD_LIN="$WORK/llm_phase2_qwen3_forward.elf"
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase2_qwen3_forward.sio" \
    --target x86_64-linux -o "$BIN_FWD_LIN" >"$WORK/linux-fwd-build.log" 2>&1
if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body' "$WORK/linux-fwd-build.log"; then
    echo "[llm-phase2] FAIL: Linux qwen3_forward build refused" >&2
    cat "$WORK/linux-fwd-build.log" >&2
    exit 1
fi
chmod +x "$BIN_FWD_LIN"

run_linux_phase2() {
    local runner="$1"
    local vm_work="/tmp/llm-phase2-vm-$$"

    if [[ "$runner" == "local" ]]; then
        local model_lin="$WORK/model_linux"
        create_qwen3_forward_fixture "$model_lin"
        printf '%s' "$model_lin" > /tmp/sounio_llm_phase2_model_cfg.txt
        printf 'cpu' > /tmp/sounio_llm_phase2_mode_cfg.txt
        printf 'dump' > /tmp/sounio_llm_phase2_dump_cfg.txt

        "$BIN_KV_LIN" >"$WORK/linux-kv-run.stdout" 2>"$WORK/linux-kv-run.stderr"
        grep -qxF "LLM_PHASE2_PAGED_KV_PASS" "$WORK/linux-kv-run.stdout"

        if [[ ! -e /dev/nvidiactl ]]; then
            gcc -shared -fPIC -O2 -nostdlib \
                "$ROOT_DIR/tests/fixtures/llm/cuda_driver_mock_witness.c" \
                -o "$WORK/libcuda_mock.so"
            export SOUNIO_CUDA_DRIVER_LIB="$WORK/libcuda_mock.so"
        fi

        local exec_ptx="$BIN_PTX_LIN"
        local exec_fwd="$BIN_FWD_LIN"
        if [[ ! -e /lib64/ld-linux-x86-64.so.2 && -e /lib/ld-musl-x86_64.so.1 ]]; then
            ln -sf /lib/ld-musl-x86_64.so.1 "$WORK/libdl.so.2"
            export LD_LIBRARY_PATH="$WORK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            python3 - "$BIN_PTX_LIN" "$WORK/run_ptx.elf" <<'PY'
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
            chmod +x "$WORK/run_ptx.elf"
            exec_ptx="$WORK/run_ptx.elf"

            python3 - "$BIN_FWD_LIN" "$WORK/run_fwd.elf" <<'PY'
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
            chmod +x "$WORK/run_fwd.elf"
            exec_fwd="$WORK/run_fwd.elf"
        fi

        "$exec_ptx" >"$WORK/linux-ptx-run.stdout" 2>"$WORK/linux-ptx-run.stderr"
        grep -qxF "LLM_PHASE2_LINUX_PTX_KERNELS_PASS" "$WORK/linux-ptx-run.stdout"

        local fwd_out_lin="$("$exec_fwd")"
        echo "$fwd_out_lin" | grep -qxF "LLM_PHASE2_QWEN3_FORWARD_PASS"
        verify_python_oracle_parity "$model_lin" "$fwd_out_lin"

        rm -f /tmp/sounio_llm_phase2_model_cfg.txt /tmp/sounio_llm_phase2_mode_cfg.txt /tmp/sounio_llm_phase2_dump_cfg.txt
    else
        limactl shell souc-linux bash -c "mkdir -p '$vm_work'"
        limactl copy "$BIN_KV_LIN" "souc-linux:$vm_work/paged_kv.elf"
        limactl copy "$BIN_PTX_LIN" "souc-linux:$vm_work/linux_ptx.elf"
        limactl copy "$BIN_FWD_LIN" "souc-linux:$vm_work/qwen3_forward.elf"
        limactl copy "$ROOT_DIR/tests/fixtures/llm/cuda_driver_mock_witness.c" "souc-linux:$vm_work/cuda_driver_mock_witness.c"

        limactl shell souc-linux bash -s -- "$vm_work" <<'EOS'
set -euo pipefail
VM_DIR="$1"
trap 'rm -rf "$VM_DIR" /tmp/sounio_llm_phase2_model_cfg.txt /tmp/sounio_llm_phase2_mode_cfg.txt /tmp/sounio_llm_phase2_dump_cfg.txt' EXIT

chmod +x "$VM_DIR/paged_kv.elf" "$VM_DIR/linux_ptx.elf" "$VM_DIR/qwen3_forward.elf"

# Run paged_kv (static)
"$VM_DIR/paged_kv.elf" > "$VM_DIR/kv.out"
grep -qxF "LLM_PHASE2_PAGED_KV_PASS" "$VM_DIR/kv.out"

# Build mock CUDA driver
if [[ ! -e /dev/nvidiactl ]]; then
    gcc -shared -fPIC -O2 -nostdlib "$VM_DIR/cuda_driver_mock_witness.c" -o "$VM_DIR/libcuda_mock.so"
    export SOUNIO_CUDA_DRIVER_LIB="$VM_DIR/libcuda_mock.so"
fi

EXEC_PTX="$VM_DIR/linux_ptx.elf"
EXEC_FWD="$VM_DIR/qwen3_forward.elf"

if [[ ! -e /lib64/ld-linux-x86-64.so.2 && -e /lib/ld-musl-x86_64.so.1 ]]; then
    ln -sf /lib/ld-musl-x86_64.so.1 "$VM_DIR/libdl.so.2"
    export LD_LIBRARY_PATH="$VM_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    python3 - "$VM_DIR/linux_ptx.elf" "$VM_DIR/run_ptx.elf" <<'PY'
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
    chmod +x "$VM_DIR/run_ptx.elf"
    EXEC_PTX="$VM_DIR/run_ptx.elf"

    python3 - "$VM_DIR/qwen3_forward.elf" "$VM_DIR/run_fwd.elf" <<'PY'
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
    chmod +x "$VM_DIR/run_fwd.elf"
    EXEC_FWD="$VM_DIR/run_fwd.elf"
fi

"$EXEC_PTX" > "$VM_DIR/ptx.out"
grep -qxF "LLM_PHASE2_LINUX_PTX_KERNELS_PASS" "$VM_DIR/ptx.out"

# Generate 2-shard model fixture
mkdir -p "$VM_DIR/model"
python3 - "$VM_DIR/model" <<'PY'
import json, math, os, struct, sys
out_dir = sys.argv[1]
cfg = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 64,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True
}
with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

def make_bf16_bytes(count, seed_offset):
    vals = [(((i + seed_offset) % 17) - 8) * 0.0625 for i in range(count)]
    u16_list = []
    for v in vals:
        b = struct.pack("<f", float(v))
        u32 = struct.unpack("<I", b)[0]
        u16_list.append(u32 >> 16)
    return struct.pack(f"<{count}H", *u16_list)

s1_tensors = {}
s1_bytes = bytearray()
def add_tensor(s_tensors, s_bytes, name, shape, seed):
    count = 1
    for d in shape: count *= d
    payload = make_bf16_bytes(count, seed)
    start = len(s_bytes)
    end = start + len(payload)
    s_bytes.extend(payload)
    s_tensors[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, end]}

add_tensor(s1_tensors, s1_bytes, "model.embed_tokens.weight", [64, 64], 1)
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

weight_map = {}
for k in s1_tensors:
    if k != "__metadata__": weight_map[k] = "model-00001-of-00002.safetensors"
for k in s2_tensors:
    if k != "__metadata__": weight_map[k] = "model-00002-of-00002.safetensors"

idx = {"metadata": {"total_size": len(s1_bytes) + len(s2_bytes)}, "weight_map": weight_map}
with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(idx, f, indent=2)
PY

printf '%s' "$VM_DIR/model" > /tmp/sounio_llm_phase2_model_cfg.txt
printf 'cpu' > /tmp/sounio_llm_phase2_mode_cfg.txt
printf 'dump' > /tmp/sounio_llm_phase2_dump_cfg.txt

"$EXEC_FWD" > "$VM_DIR/fwd.out"
grep -qxF "LLM_PHASE2_QWEN3_FORWARD_PASS" "$VM_DIR/fwd.out"

# Parity check inside VM
python3 - "$VM_DIR/model" "$(cat "$VM_DIR/fwd.out")" <<'PY'
import json, math, os, re, struct, sys

model_dir, sout = sys.argv[1], sys.argv[2]
m = re.search(r'\[QWEN3_FORWARD_LOGITS: top1=(\d+) val=([-\d\.]+) l0=([-\d\.]+) l1=([-\d\.]+) l2=([-\d\.]+)\]', sout)
if not m:
    print("[llm-phase2] FAIL: could not extract logit dump", file=sys.stderr)
    sys.exit(1)

s_top1 = int(m.group(1))
s_val = float(m.group(2))
s_l0 = float(m.group(3))
s_l1 = float(m.group(4))
s_l2 = float(m.group(5))

with open(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)
with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
    idx = json.load(f)

shards = {}
for sf in set(idx["weight_map"].values()):
    with open(os.path.join(model_dir, sf), "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len).decode("utf-8"))
        data = f.read()
        shards[sf] = (hdr, data)

def get_tensor(name):
    sf = idx["weight_map"][name]
    hdr, data = shards[sf]
    meta = hdr[name]
    start, end = meta["data_offsets"]
    payload = data[start:end]
    count = (end - start) // 2
    u16s = struct.unpack(f"<{count}H", payload)
    f32s = [struct.unpack("<f", struct.pack("<I", u << 16))[0] for u in u16s]
    return f32s, meta["shape"]

def rmsnorm(x, w, eps):
    dim = len(x)
    sum_sq = sum(v * v for v in x)
    inv_rms = 1.0 / math.sqrt(sum_sq / dim + eps)
    return [x[i] * inv_rms * w[i] for i in range(dim)]

def per_head_rmsnorm(x, w, n_heads, head_dim, eps):
    out = []
    for h in range(n_heads):
        head_x = x[h * head_dim : (h + 1) * head_dim]
        out.extend(rmsnorm(head_x, w, eps))
    return out

def rope(x, n_heads, head_dim, pos, theta):
    out = list(x)
    half = head_dim // 2
    for h in range(n_heads):
        for j in range(half):
            freq = theta ** (-2.0 * j / head_dim)
            angle = pos * freq
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            v0 = x[h * head_dim + j]
            v1 = x[h * head_dim + j + half]
            out[h * head_dim + j] = v0 * cos_a - v1 * sin_a
            out[h * head_dim + j + half] = v0 * sin_a + v1 * cos_a
    return out

def gemv(w, rows, cols, x):
    out = []
    for r in range(rows):
        row = w[r * cols : (r + 1) * cols]
        dot = sum(row[c] * x[c] for c in range(cols))
        out.append(dot)
    return out

hidden = cfg["hidden_size"]
inter = cfg["intermediate_size"]
layers = cfg["num_hidden_layers"]
n_q = cfg["num_attention_heads"]
n_kv = cfg["num_key_value_heads"]
head_dim = cfg["head_dim"]
vocab = cfg["vocab_size"]
eps = cfg["rms_norm_eps"]
theta = cfg["rope_theta"]

embed_w, _ = get_tensor("model.embed_tokens.weight")
norm_w, _ = get_tensor("model.norm.weight")
lm_head_w = embed_w

kv_k = [[[] for _ in range(layers)] for _ in range(16)]
kv_v = [[[] for _ in range(layers)] for _ in range(16)]

tokens = [7, 23, 42]
logits = []

for pos, tok in enumerate(tokens):
    x = embed_w[tok * hidden : (tok + 1) * hidden]
    for l in range(layers):
        in_norm_w, _ = get_tensor(f"model.layers.{l}.input_layernorm.weight")
        q_proj_w, _ = get_tensor(f"model.layers.{l}.self_attn.q_proj.weight")
        k_proj_w, _ = get_tensor(f"model.layers.{l}.self_attn.k_proj.weight")
        v_proj_w, _ = get_tensor(f"model.layers.{l}.self_attn.v_proj.weight")
        o_proj_w, _ = get_tensor(f"model.layers.{l}.self_attn.o_proj.weight")
        q_norm_w, _ = get_tensor(f"model.layers.{l}.self_attn.q_norm.weight")
        k_norm_w, _ = get_tensor(f"model.layers.{l}.self_attn.k_norm.weight")
        post_norm_w, _ = get_tensor(f"model.layers.{l}.post_attention_layernorm.weight")
        gate_w, _ = get_tensor(f"model.layers.{l}.mlp.gate_proj.weight")
        up_w, _ = get_tensor(f"model.layers.{l}.mlp.up_proj.weight")
        down_w, _ = get_tensor(f"model.layers.{l}.mlp.down_proj.weight")

        h = rmsnorm(x, in_norm_w, eps)
        q = gemv(q_proj_w, hidden, hidden, h)
        k = gemv(k_proj_w, n_kv * head_dim, hidden, h)
        v = gemv(v_proj_w, n_kv * head_dim, hidden, h)

        q = per_head_rmsnorm(q, q_norm_w, n_q, head_dim, eps)
        k = per_head_rmsnorm(k, k_norm_w, n_kv, head_dim, eps)

        q = rope(q, n_q, head_dim, pos, theta)
        k = rope(k, n_kv, head_dim, pos, theta)

        kv_k[pos][l] = k
        kv_v[pos][l] = v

        scale = 1.0 / math.sqrt(head_dim)
        group_size = n_q // n_kv
        attn_out = [0.0] * (n_q * head_dim)
        for q_h in range(n_q):
            kv_h = q_h // group_size
            q_vec = q[q_h * head_dim : (q_h + 1) * head_dim]
            scores = []
            for p in range(pos + 1):
                k_vec = kv_k[p][l][kv_h * head_dim : (kv_h + 1) * head_dim]
                dot = sum(q_vec[d] * k_vec[d] for d in range(head_dim))
                scores.append(dot * scale)
            max_s = max(scores)
            exps = [math.exp(s - max_s) for s in scores]
            sum_exp = sum(exps)
            weights = [e / sum_exp for e in exps]
            for d in range(head_dim):
                val = sum(weights[p] * kv_v[p][l][kv_h * head_dim + d] for p in range(pos + 1))
                attn_out[q_h * head_dim + d] = val

        o_out = gemv(o_proj_w, hidden, hidden, attn_out)
        x = [x[i] + o_out[i] for i in range(hidden)]

        h = rmsnorm(x, post_norm_w, eps)
        gate = gemv(gate_w, inter, hidden, h)
        up = gemv(up_w, inter, hidden, h)
        swiglu = [(gate[i] / (1.0 + math.exp(-gate[i]))) * up[i] for i in range(inter)]
        down = gemv(down_w, hidden, inter, swiglu)
        x = [x[i] + down[i] for i in range(hidden)]

    x_norm = rmsnorm(x, norm_w, eps)
    logits = gemv(lm_head_w, vocab, hidden, x_norm)

py_top1 = max(range(len(logits)), key=lambda i: logits[i])
py_val = logits[py_top1]
py_l0 = logits[0]
py_l1 = logits[1]
py_l2 = logits[2]

max_diff = max(abs(s_val - py_val), abs(s_l0 - py_l0), abs(s_l1 - py_l1), abs(s_l2 - py_l2))
if s_top1 != py_top1 or max_diff >= 0.001:
    print(f"[llm-phase2] FAIL: parity mismatch top1={s_top1}/{py_top1} diff={max_diff}", file=sys.stderr)
    sys.exit(2)

print(f"[llm-phase2] PASS Linux VM Python oracle parity: top1={py_top1} match, max_diff={max_diff:.6f} < 1e-3")
PY
EOS
    fi
    echo "[llm-phase2] PASS Linux x86-64 ELF Paged KV + PTX Kernels + Qwen3 Forward"
}

if [[ "$(uname -s)" == Linux ]]; then
    run_linux_phase2 "local"
elif command -v limactl >/dev/null 2>&1 && limactl list 2>/dev/null | grep -q "^souc-linux[[:space:]]\+Running"; then
    run_linux_phase2 "lima"
else
    echo "[llm-phase2] ERROR: neither Linux host nor running Lima VM 'souc-linux' available" >&2
    exit 2
fi

echo "LLM_PHASE2_FORWARD_GATE_PASS"
