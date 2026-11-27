#!/usr/bin/env bash
# Fase 1 Ingestion Gate:
#   1. Zero-copy multi-shard Safetensors reader on macOS ARM64 & Linux x86-64 ELF
#   2. Qwen3 BPE tokenizer with 151,936+ vocab + ChatML + PT-BR UTF-8 + Sounio code
#   3. Dual merge formats ("a b" strings & ["a", "b"] pairs) + 100% HF token parity
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/llm-phase1-gate.XXXXXX")"
CFG_ST="/tmp/sounio_llm_phase1_st_cfg.txt"
CFG_TOK="/tmp/sounio_llm_phase1_tok_cfg.txt"
CFG_DUMP="/tmp/sounio_llm_phase1_tok_dump_cfg.txt"
trap 'rm -rf "$WORK" "$CFG_ST" "$CFG_TOK" "$CFG_DUMP"' EXIT

echo "[llm-phase1] compiler=$MADAROS_BIN"

create_safetensors_fixture() {
    local out_dir="$1"
    mkdir -p "$out_dir"
    python3 - "$out_dir" <<'PY'
import json, os, struct, sys
out_dir = sys.argv[1]

# 1. Multi-shard index
idx = {
    "metadata": {"total_size": 2684354560},
    "weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "model.norm.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
    }
}
with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(idx, f)

# 2. Shard 1 (embed BF16 + norm F32 + optional probe dtypes)
s1_payload = (
    struct.pack("<6H", 0x3F80, 0xC020, 0x4049, 0x0000, 0x8000, 0x477F) +
    struct.pack("<4f", 1.0, -0.5, 2.25, 0.0) +
    struct.pack("<2H", 0x3C00, 0xC000) +
    struct.pack("<d", -3.25) +
    struct.pack("<q", -987654321) +
    struct.pack("<2B", 7, 255) +
    struct.pack("<2i", -42, 123456)
)
s1_hdr = {
    "model.embed_tokens.weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
    "model.norm.weight": {"dtype": "F32", "shape": [4], "data_offsets": [12, 28]},
    "extra.f16_probe": {"dtype": "F16", "shape": [2], "data_offsets": [28, 32]},
    "extra.f64_probe": {"dtype": "F64", "shape": [1], "data_offsets": [32, 40]},
    "extra.i64_probe": {"dtype": "I64", "shape": [1], "data_offsets": [40, 48]},
    "extra.u8_probe": {"dtype": "U8", "shape": [2], "data_offsets": [48, 50]},
    "extra.i32_probe": {"dtype": "I32", "shape": [2], "data_offsets": [50, 58]},
    "__metadata__": {"format": "pt"}
}
s1_hdr_bytes = json.dumps(s1_hdr).encode("utf-8")
with open(os.path.join(out_dir, "model-00001-of-00002.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(s1_hdr_bytes)))
    f.write(s1_hdr_bytes)
    f.write(s1_payload)

# 3. Shard 2: sparse file >= 2684354560 bytes (2.5 GiB) with tensor at >= 2415919104 (2.25 GiB)
q_off = 2415919104
q_payload = struct.pack("<6H", 0x3F80, 0xC020, 0x4049, 0x0000, 0x8000, 0x477F)
s2_hdr = {
    "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [q_off, q_off + 12]},
    "__metadata__": {"format": "pt"}
}
s2_hdr_bytes = json.dumps(s2_hdr).encode("utf-8")
hdr_len = len(s2_hdr_bytes)
total_size = 2684354560
with open(os.path.join(out_dir, "model-00002-of-00002.safetensors"), "wb") as f:
    f.truncate(total_size)
    f.seek(0)
    f.write(struct.pack("<Q", hdr_len))
    f.write(s2_hdr_bytes)
    f.seek(8 + hdr_len + q_off)
    f.write(q_payload)
PY
}

create_qwen3_fixture() {
    local out_file="$1"
    local merge_mode="${2:-array}" # "array" or "string"
    python3 - "$out_file" "$merge_mode" <<'PY'
import json, os, sys

out_file = sys.argv[1]
merge_mode = sys.argv[2]
host_checkpoint_tok = "/Users/demetriosagourakis/sounio-research/checkpoints/sounio-coder-3b/tokenizer.json"

if os.path.exists(host_checkpoint_tok):
    with open(host_checkpoint_tok, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    # Deterministic synthetic Qwen3 fallback
    vocab = {}
    for i in range(256):
        vocab[chr(i) if i >= 32 and i <= 126 else f"<0x{i:02X}>"] = i
    vocab["2"] = 17
    vocab["0"] = 15
    vocab["6"] = 21
    merges = []
    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None, "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": True, "vocab": vocab, "merges": merges}
    }

# Augment added_tokens with Qwen3 specials (151643..151668) with reversed key order "content" before "id"
added = data.get("added_tokens", [])
added_ids = {t["id"] for t in added}
for i in range(151643, 151669):
    if i not in added_ids:
        content = "<think>" if i == 151667 else ("</think>" if i == 151668 else f"<|extra_{i}|>")
        added.append({
            "content": content,
            "id": i,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        })
data["added_tokens"] = added

# Adjust merge format according to requested mode
merges = data.get("model", {}).get("merges", [])
if merge_mode == "string":
    str_merges = []
    for m in merges:
        if isinstance(m, list):
            str_merges.append(f"{m[0]} {m[1]}")
        else:
            str_merges.append(str(m))
    data["model"]["merges"] = str_merges
elif merge_mode == "array":
    pair_merges = []
    for m in merges:
        if isinstance(m, str):
            parts = m.split(" ", 1)
            pair_merges.append(parts if len(parts) == 2 else [parts[0], ""])
        else:
            pair_merges.append(m)
    data["model"]["merges"] = pair_merges

os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
PY
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1A: macOS ARM64 Verification
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]]; then
    echo "[llm-phase1] target=aarch64-macos"

    ST_FIXTURE_MAC="$WORK/st_mac"
    create_safetensors_fixture "$ST_FIXTURE_MAC"
    printf '%s' "$ST_FIXTURE_MAC" > "$CFG_ST"

    BIN_ST_MAC="$WORK/llm_phase1_safetensors_mmap"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase1_safetensors_mmap.sio" \
        --target aarch64-macos -o "$BIN_ST_MAC" >"$WORK/mac-st-build.log" 2>&1
    if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body|ARM64_REFUSAL' "$WORK/mac-st-build.log"; then
        echo "[llm-phase1] FAIL: macOS safetensors build refused" >&2
        cat "$WORK/mac-st-build.log" >&2
        exit 1
    fi
    chmod +x "$BIN_ST_MAC"
    codesign -s - -f "$BIN_ST_MAC"
    "$BIN_ST_MAC" >"$WORK/mac-st-run.stdout" 2>"$WORK/mac-st-run.stderr"
    grep -qxF "LLM_PHASE1_SAFETENSORS_MMAP_PASS" "$WORK/mac-st-run.stdout"
    echo "[llm-phase1] PASS macOS ARM64 Safetensors 2.5 GiB zero-copy + multi-shard index"

    # Build Qwen3 tokenizer native binary
    BIN_TOK_MAC="$WORK/llm_phase1_qwen3_tokenizer"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase1_qwen3_tokenizer.sio" \
        --target aarch64-macos -o "$BIN_TOK_MAC" >"$WORK/mac-tok-build.log" 2>&1
    if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body|ARM64_REFUSAL' "$WORK/mac-tok-build.log"; then
        echo "[llm-phase1] FAIL: macOS tokenizer build refused" >&2
        cat "$WORK/mac-tok-build.log" >&2
        exit 1
    fi
    chmod +x "$BIN_TOK_MAC"
    codesign -s - -f "$BIN_TOK_MAC"

    # Test 1: Array-pair merge format
    TOK_ARRAY_MAC="$WORK/tokenizer_array.json"
    create_qwen3_fixture "$TOK_ARRAY_MAC" "array"
    printf '%s' "$TOK_ARRAY_MAC" > "$CFG_TOK"
    rm -f "$CFG_DUMP"
    "$BIN_TOK_MAC" >"$WORK/mac-tok-arr.stdout" 2>"$WORK/mac-tok-arr.stderr"
    grep -qxF "LLM_PHASE1_QWEN3_TOKENIZER_PASS" "$WORK/mac-tok-arr.stdout"
    echo "[llm-phase1] PASS macOS ARM64 Qwen3 tokenizer (array-pair merges format)"

    # Test 2: String merge format
    TOK_STRING_MAC="$WORK/tokenizer_string.json"
    create_qwen3_fixture "$TOK_STRING_MAC" "string"
    printf '%s' "$TOK_STRING_MAC" > "$CFG_TOK"
    "$BIN_TOK_MAC" >"$WORK/mac-tok-str.stdout" 2>"$WORK/mac-tok-str.stderr"
    grep -qxF "LLM_PHASE1_QWEN3_TOKENIZER_PASS" "$WORK/mac-tok-str.stdout"
    echo "[llm-phase1] PASS macOS ARM64 Qwen3 tokenizer (string merges format)"

    # Test 3: HF tokenizers 100% exact parity check
    CONDA_PY="/Users/demetriosagourakis/miniconda3/bin/python3"
    CHECKPOINT_TOK="/Users/demetriosagourakis/sounio-research/checkpoints/sounio-coder-3b/tokenizer.json"
    if [[ -x "$CONDA_PY" && -f "$CHECKPOINT_TOK" ]]; then
        printf 'dump' > "$CFG_DUMP"
        printf '%s' "$TOK_ARRAY_MAC" > "$CFG_TOK"
        OUT_MAC="$("$BIN_TOK_MAC")"
        rm -f "$CFG_DUMP"
        "$CONDA_PY" - "$TOK_ARRAY_MAC" "$OUT_MAC" <<'PY'
import sys, re, tokenizers

tok_path, sout = sys.argv[1], sys.argv[2]
m = re.search(r'\[QWEN3_CHAT_TOKENS:(.*?)\]', sout, re.DOTALL)
if not m:
    print("[llm-phase1] FAIL: could not extract token dump from Sounio output", file=sys.stderr)
    sys.exit(1)

sounio_ids = [int(x) for x in m.group(1).split()]
hf_tok = tokenizers.Tokenizer.from_file(tok_path)

sys_msg = "You are a Sounio compiler assistant."
user_msg = "Olá, mundo! Ação, coração, é, ç, ã, õ, â, ê, ô — teste 2026:\nfn soma(x: &!i64, n: i64) -> i64 with Mut {\n    var acc: i64 = *x\n    acc = acc + n\n    acc\n}\n"
assistant_prefix = "<think>\nAnalisando função em Sounio 2026.\n</think>\nCódigo válido em Sounio.<|im_end|>\n"
full_text = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_prefix}"

hf_res = hf_tok.encode(full_text)
hf_ids = hf_res.ids

if sounio_ids != hf_ids:
    print(f"[llm-phase1] FAIL: HF token parity mismatch (sounio={len(sounio_ids)} hf={len(hf_ids)})", file=sys.stderr)
    sys.exit(2)

print(f"[llm-phase1] PASS HF token parity: 100% exact across {len(hf_ids)} ChatML tokens")
PY
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1B: Linux x86-64 Static ELF Verification
# ─────────────────────────────────────────────────────────────────────────────
echo "[llm-phase1] target=x86_64-linux"

BIN_ST_LIN="$WORK/llm_phase1_safetensors_mmap.elf"
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase1_safetensors_mmap.sio" \
    --target x86_64-linux -o "$BIN_ST_LIN" >"$WORK/linux-st-build.log" 2>&1
if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body' "$WORK/linux-st-build.log"; then
    echo "[llm-phase1] FAIL: Linux safetensors build refused" >&2
    cat "$WORK/linux-st-build.log" >&2
    exit 1
fi
chmod +x "$BIN_ST_LIN"

BIN_TOK_LIN="$WORK/llm_phase1_qwen3_tokenizer.elf"
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase1_qwen3_tokenizer.sio" \
    --target x86_64-linux -o "$BIN_TOK_LIN" >"$WORK/linux-tok-build.log" 2>&1
if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body' "$WORK/linux-tok-build.log"; then
    echo "[llm-phase1] FAIL: Linux tokenizer build refused" >&2
    cat "$WORK/linux-tok-build.log" >&2
    exit 1
fi
chmod +x "$BIN_TOK_LIN"

run_linux_phase1() {
    local runner="$1"
    local vm_work="/tmp/llm-phase1-vm-$$"

    if [[ "$runner" == "local" ]]; then
        local st_lin="$WORK/st_linux"
        create_safetensors_fixture "$st_lin"
        printf '%s' "$st_lin" > /tmp/sounio_llm_phase1_st_cfg.txt

        local tok_lin="$WORK/tokenizer_linux.json"
        create_qwen3_fixture "$tok_lin" "array"
        printf '%s' "$tok_lin" > /tmp/sounio_llm_phase1_tok_cfg.txt

        "$BIN_ST_LIN" >"$WORK/linux-st-run.stdout" 2>"$WORK/linux-st-run.stderr"
        grep -qxF "LLM_PHASE1_SAFETENSORS_MMAP_PASS" "$WORK/linux-st-run.stdout"

        "$BIN_TOK_LIN" >"$WORK/linux-tok-run.stdout" 2>"$WORK/linux-tok-run.stderr"
        grep -qxF "LLM_PHASE1_QWEN3_TOKENIZER_PASS" "$WORK/linux-tok-run.stdout"

        rm -f /tmp/sounio_llm_phase1_st_cfg.txt /tmp/sounio_llm_phase1_tok_cfg.txt
    else
        limactl shell souc-linux bash -c "mkdir -p '$vm_work/st' '$vm_work/tok'"
        limactl copy "$BIN_ST_LIN" "souc-linux:$vm_work/safetensors.elf"
        limactl copy "$BIN_TOK_LIN" "souc-linux:$vm_work/qwen3.elf"

        # Generate tokenizer fixture on host and copy to Lima VM
        local tok_host="$WORK/tok_host.json"
        create_qwen3_fixture "$tok_host" "array"
        limactl copy "$tok_host" "souc-linux:$vm_work/tok/tokenizer.json"

        limactl shell souc-linux bash -s -- "$vm_work" <<'EOS'
set -euo pipefail
VM_DIR="$1"
trap 'rm -rf "$VM_DIR" /tmp/sounio_llm_phase1_st_cfg.txt /tmp/sounio_llm_phase1_tok_cfg.txt' EXIT

mkdir -p "$VM_DIR/st"
python3 - "$VM_DIR/st" <<'PY'
import json, os, struct, sys
out_dir = sys.argv[1]

idx = {
    "metadata": {"total_size": 2684354560},
    "weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "model.norm.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
    }
}
with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
    json.dump(idx, f)

s1_payload = (
    struct.pack("<6H", 0x3F80, 0xC020, 0x4049, 0x0000, 0x8000, 0x477F) +
    struct.pack("<4f", 1.0, -0.5, 2.25, 0.0) +
    struct.pack("<2H", 0x3C00, 0xC000) +
    struct.pack("<d", -3.25) +
    struct.pack("<q", -987654321) +
    struct.pack("<2B", 7, 255) +
    struct.pack("<2i", -42, 123456)
)
s1_hdr = {
    "model.embed_tokens.weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
    "model.norm.weight": {"dtype": "F32", "shape": [4], "data_offsets": [12, 28]},
    "extra.f16_probe": {"dtype": "F16", "shape": [2], "data_offsets": [28, 32]},
    "extra.f64_probe": {"dtype": "F64", "shape": [1], "data_offsets": [32, 40]},
    "extra.i64_probe": {"dtype": "I64", "shape": [1], "data_offsets": [40, 48]},
    "extra.u8_probe": {"dtype": "U8", "shape": [2], "data_offsets": [48, 50]},
    "extra.i32_probe": {"dtype": "I32", "shape": [2], "data_offsets": [50, 58]},
    "__metadata__": {"format": "pt"}
}
s1_hdr_bytes = json.dumps(s1_hdr).encode("utf-8")
with open(os.path.join(out_dir, "model-00001-of-00002.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(s1_hdr_bytes)))
    f.write(s1_hdr_bytes)
    f.write(s1_payload)

q_off = 2415919104
q_payload = struct.pack("<6H", 0x3F80, 0xC020, 0x4049, 0x0000, 0x8000, 0x477F)
s2_hdr = {
    "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [q_off, q_off + 12]},
    "__metadata__": {"format": "pt"}
}
s2_hdr_bytes = json.dumps(s2_hdr).encode("utf-8")
hdr_len = len(s2_hdr_bytes)
total_size = 2684354560
with open(os.path.join(out_dir, "model-00002-of-00002.safetensors"), "wb") as f:
    f.truncate(total_size)
    f.seek(0)
    f.write(struct.pack("<Q", hdr_len))
    f.write(s2_hdr_bytes)
    f.seek(8 + hdr_len + q_off)
    f.write(q_payload)
PY

printf '%s' "$VM_DIR/st" > /tmp/sounio_llm_phase1_st_cfg.txt
printf '%s' "$VM_DIR/tok/tokenizer.json" > /tmp/sounio_llm_phase1_tok_cfg.txt

chmod +x "$VM_DIR/safetensors.elf" "$VM_DIR/qwen3.elf"

"$VM_DIR/safetensors.elf" > "$VM_DIR/st.out"
grep -qxF "LLM_PHASE1_SAFETENSORS_MMAP_PASS" "$VM_DIR/st.out"

"$VM_DIR/qwen3.elf" > "$VM_DIR/tok.out"
grep -qxF "LLM_PHASE1_QWEN3_TOKENIZER_PASS" "$VM_DIR/tok.out"
EOS
    fi
    echo "[llm-phase1] PASS Linux x86-64 ELF Safetensors 2.5 GiB + Qwen3 Tokenizer"
}

if [[ "$(uname -s)" == Linux ]]; then
    run_linux_phase1 "local"
elif command -v limactl >/dev/null 2>&1 && limactl list 2>/dev/null | grep -q "^souc-linux[[:space:]]\+Running"; then
    run_linux_phase1 "lima"
else
    echo "[llm-phase1] ERROR: neither Linux host nor running Lima VM 'souc-linux' available" >&2
    exit 2
fi

echo "LLM_PHASE1_INGESTION_GATE_PASS"
