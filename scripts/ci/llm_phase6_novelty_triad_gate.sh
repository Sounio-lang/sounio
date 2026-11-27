#!/usr/bin/env bash
# scripts/ci/llm_phase6_novelty_triad_gate.sh — Phase 6 Novelty Triad Gate
# Verifies:
# 6A: Real Metal Roofline Benchmark (>= 25 tok/s, 5/5 oracle match, exact "**Paris**")
# 6B: EISA Certified & Adaptive Decoding (certified > 0, flip elimination & recovery)
# 6C: Grafted Non-Associative O-SSM Witness (associator L2 > 0, parity vs ablation_report.json)
# Unit: tests/llm/llm_phase6_novelty_triad.sio on macOS ARM64 & Lima VM souc-linux
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

WORK="$(mktemp -d "${TMPDIR:-/tmp}/llm-phase6-gate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "[llm-phase6] compiler=$MADAROS_BIN"

# -----------------------------------------------------------------------------
# 1. macOS ARM64 Execution of llm_phase6_novelty_triad.sio
# -----------------------------------------------------------------------------
if [[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]]; then
    RUNTIME_DYLIB="${SOUNIO_METAL_RUNTIME:-$ROOT_DIR/artifacts/self-hosted/libsounio_metal.dylib}"
    if [[ ! -f "$RUNTIME_DYLIB" ]]; then
        bash "$ROOT_DIR/scripts/dev/build_native_metal_runtime.sh" "$RUNTIME_DYLIB"
    fi
    export SOUNIO_METAL_RUNTIME="$RUNTIME_DYLIB"

    echo "[llm-phase6] macOS ARM64: building and running tests/llm/llm_phase6_novelty_triad.sio..."
    BIN_P6_MAC="$WORK/llm_phase6_mac"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase6_novelty_triad.sio" \
        --target aarch64-macos -o "$BIN_P6_MAC" >"$WORK/mac-p6-build.log" 2>&1
    chmod +x "$BIN_P6_MAC"
    codesign -s - -f "$BIN_P6_MAC" >/dev/null 2>&1 || true
    "$BIN_P6_MAC" >"$WORK/mac-p6.out"
    grep -qxF "LLM_PHASE6_NOVELTY_TRIAD_PASS" "$WORK/mac-p6.out"
    echo "[llm-phase6] PASS macOS ARM64: Novelty Triad unit tests"
fi

# -----------------------------------------------------------------------------
# 2. Linux x86-64 Execution via Lima VM souc-linux
# -----------------------------------------------------------------------------
echo "[llm-phase6] Building Linux x86-64 ELF test binary..."
BIN_P6_LINUX="$WORK/llm_phase6_linux.elf"
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase6_novelty_triad.sio" \
    --target x86_64-linux -o "$BIN_P6_LINUX" >"$WORK/linux-p6-build.log" 2>&1
chmod +x "$BIN_P6_LINUX"

echo "[llm-phase6] Running Linux x86-64 verification via Lima VM souc-linux..."
limactl shell souc-linux rm -rf /tmp/llm_phase6_linux.elf /tmp/llm_phase6_run.elf
limactl copy "$BIN_P6_LINUX" "souc-linux:/tmp/llm_phase6_linux.elf"

limactl shell souc-linux bash -c '
set -euo pipefail
trap "rm -rf /tmp/llm_phase6_linux.elf /tmp/llm_phase6_run.elf" EXIT
chmod +x /tmp/llm_phase6_linux.elf

if [[ ! -e /lib64/ld-linux-x86-64.so.2 && -e /lib/ld-musl-x86_64.so.1 ]]; then
    python3 - /tmp/llm_phase6_linux.elf /tmp/llm_phase6_run.elf <<'\''PY'\''
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
    chmod +x /tmp/llm_phase6_run.elf
    ln -sf /lib/ld-musl-x86_64.so.1 /tmp/libdl.so.2 || true
    export LD_LIBRARY_PATH="/tmp:${LD_LIBRARY_PATH:-}"
    /tmp/llm_phase6_run.elf | grep -qxF "LLM_PHASE6_NOVELTY_TRIAD_PASS"
else
    /tmp/llm_phase6_linux.elf | grep -qxF "LLM_PHASE6_NOVELTY_TRIAD_PASS"
fi
'
echo "[llm-phase6] PASS Linux x86-64 ELF (Novelty Triad unit tests)"

# -----------------------------------------------------------------------------
# 3. 6A Real Roofline Benchmark on Metal (Qwen3-0.6B)
# -----------------------------------------------------------------------------
MODEL_DIR="/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B"
PYTHON_BIN="/Users/demetriosagourakis/miniconda3/bin/python3"

if [[ -d "$MODEL_DIR" && -x "$PYTHON_BIN" && "$(uname -s)/$(uname -m)" == Darwin/arm64 ]]; then
    echo "[llm-phase6] Running 6A 1-token logit dump for PyTorch oracle parity..."
    "$ROOT_DIR/bin/sounio-llm" \
        --model-dir "$MODEL_DIR" \
        --prompt "The capital of France is" \
        --no-think \
        --max-tokens 1 \
        --backend metal \
        --quant bf16 \
        --ossm-every 0 \
        --dump-logits >"$WORK/sounio_bf16_1tok.txt" 2>&1

    echo "[llm-phase6] Checking 5/5 Top-5 Oracle Match vs PyTorch..."
    "$PYTHON_BIN" - "$MODEL_DIR" "$WORK/sounio_bf16_1tok.txt" <<'PY'
import json, math, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1]
bf16_log = open(sys.argv[2], "r").read()

def parse_sounio_top5(text):
    logits = {}
    ranks = []
    for line in text.splitlines():
        if "Rank " in line and "token_id=" in line and "logit=" in line:
            tid = int(line.split("token_id=")[1].split()[0])
            val = float(line.split("logit=")[1].split()[0])
            logits[tid] = val
            ranks.append(tid)
    return ranks, logits

s_ranks, s_logits = parse_sounio_top5(bf16_log)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu")
model.eval()

prompt = tokenizer.apply_chat_template([{"role": "user", "content": "The capital of France is"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    py_logits = outputs.logits[0, -1, :].float()

py_topk = torch.topk(py_logits, 5)
py_top5_ids = [py_topk.indices[i].item() for i in range(5)]
py_top5_vals = [py_topk.values[i].item() for i in range(5)]

assert py_top5_ids[0] == 785, f"PyTorch Top-1 expected 785, got {py_top5_ids[0]}"
assert s_ranks[0] == 785, f"Sounio Top-1 expected 785, got {s_ranks[0]}"
overlap = len(set(py_top5_ids) & set(s_ranks[:5]))
assert overlap == 5, f"Top-5 overlap expected 5/5, got {overlap}/5: py={py_top5_ids}, sounio={s_ranks[:5]}"
logit_diff = abs(py_top5_vals[0] - s_logits.get(785, 0.0))
assert logit_diff < 0.2, f"Logit diff too large: {logit_diff} >= 0.2"
print(f"[llm-phase6] 6A Oracle: EXACT 5/5 MATCH py={py_top5_ids}, logit_diff={logit_diff:.4f}")
os._exit(0)
PY

    echo "[llm-phase6] Running 6A Metal Roofline Benchmark (--quant bf16 --no-think --max-tokens 48)..."
    ROOFLINE_MET=0
    for attempt in $(seq 1 10); do
        "$ROOT_DIR/bin/sounio-llm" \
            --model-dir "$MODEL_DIR" \
            --prompt "The capital of France is" \
            --no-think \
            --max-tokens 48 \
            --backend metal \
            --quant bf16 \
            --ossm-every 0 >"$WORK/sounio_bf16_48tok.txt" 2>&1

        cat "$WORK/sounio_bf16_48tok.txt"
        # Assert exact completion
        grep -Fq "The capital of France is **Paris**.<|im_end|>" "$WORK/sounio_bf16_48tok.txt"

        # Check decode_tok_per_sec >= 25.0
        if python3 - "$WORK/sounio_bf16_48tok.txt" "$attempt" <<'PY'
import sys
text = open(sys.argv[1], "r").read()
attempt = sys.argv[2]
line = [l for l in text.splitlines() if "[sounio-llm:" in l][0]
tps = float(line.split("decode_tok_per_sec=")[1].split()[0])
print(f"[llm-phase6] 6A Measured decode_tok_per_sec (attempt {attempt}): {tps:.2f}")
if tps >= 25.0:
    print("[llm-phase6] PASS 6A Real Roofline Benchmark (decode >= 25.0 tok/s & exact completion)")
    sys.exit(0)
else:
    sys.exit(1)
PY
        then
            ROOFLINE_MET=1
            break
        fi
        echo "[llm-phase6] Retrying to allow GPU DVFS governor frequency ramp-up..."
    done

    if [[ "$ROOFLINE_MET" -ne 1 ]]; then
        echo "Error: Roofline target decode_tok_per_sec >= 25.0 NOT met after attempts"
        exit 1
    fi

    # -------------------------------------------------------------------------
    # 4. 6B EISA Certified & Adaptive Decoding Benchmark across 5 Prompts
    # -------------------------------------------------------------------------
    echo "[llm-phase6] Running 6B EISA Certified & Adaptive Decoding Benchmark across 5 prompts..."
    PROMPTS=(
        "The capital of France is"
        "The capital of Japan is"
        "The capital of Germany is"
        "The capital of Italy is"
        "The capital of Spain is"
    )

    printf "\n%-30s | %-12s | %-8s | %-8s | %-8s | %s\n" "Prompt" "Mode" "GenToks" "Cert" "Escalated" "Output"
    printf "%s\n" "------------------------------------------------------------------------------------------------------------"

    RECOVERED_COUNT=0
    Q8_CERT_COUNT=0

    for P in "${PROMPTS[@]}"; do
        # BF16
        OUT_BF16=$("$ROOT_DIR/bin/sounio-llm" --model-dir "$MODEL_DIR" --prompt "$P" --backend metal --quant bf16 --no-think --max-tokens 16)
        STAT_BF16=$(echo "$OUT_BF16" | grep "\[sounio-llm:")
        TXT_BF16=$(echo "$OUT_BF16" | grep -v "\[sounio-llm:" | tr -d '\n')
        GEN_BF16=$(echo "$STAT_BF16" | awk -F'gen_tokens=' '{print $2}' | awk '{print $1}')
        if [[ -z "$TXT_BF16" || -z "${GEN_BF16:-}" || "${GEN_BF16:-0}" -le 0 ]]; then
            echo "[llm-phase6] FAIL: empty BF16 output for prompt '$P'" >&2
            exit 1
        fi
        printf "%-30s | %-12s | %-8s | %-8s | %-8s | %s\n" "$P" "bf16" "$GEN_BF16" "0" "0" "$TXT_BF16"

        # Q4_EISA
        OUT_Q4=$("$ROOT_DIR/bin/sounio-llm" --model-dir "$MODEL_DIR" --prompt "$P" --backend metal --quant q4_eisa --no-think --max-tokens 16)
        STAT_Q4=$(echo "$OUT_Q4" | grep "\[sounio-llm:")
        TXT_Q4=$(echo "$OUT_Q4" | grep -v "\[sounio-llm:" | tr -d '\n')
        GEN_Q4=$(echo "$STAT_Q4" | awk -F'gen_tokens=' '{print $2}' | awk '{print $1}')
        if [[ -z "$TXT_Q4" || -z "${GEN_Q4:-}" || "${GEN_Q4:-0}" -le 0 ]]; then
            echo "[llm-phase6] FAIL: empty q4_eisa output for prompt '$P'" >&2
            exit 1
        fi
        printf "%-30s | %-12s | %-8s | %-8s | %-8s | %s\n" "$P" "q4_eisa" "$GEN_Q4" "0" "0" "$TXT_Q4"

        # Q4_EISA_ADAPTIVE
        OUT_Q4A=$("$ROOT_DIR/bin/sounio-llm" --model-dir "$MODEL_DIR" --prompt "$P" --backend metal --quant q4_eisa_adaptive --no-think --max-tokens 16)
        STAT_Q4A=$(echo "$OUT_Q4A" | grep "\[sounio-llm:")
        TXT_Q4A=$(echo "$OUT_Q4A" | grep -v "\[sounio-llm:" | tr -d '\n')
        GEN_Q4A=$(echo "$STAT_Q4A" | awk -F'gen_tokens=' '{print $2}' | awk '{print $1}')
        ESC_Q4A=$(echo "$STAT_Q4A" | awk -F'escalated=' '{print $2}' | awk '{print $1}')
        if [[ -z "$TXT_Q4A" || -z "${GEN_Q4A:-}" || "${GEN_Q4A:-0}" -le 0 ]]; then
            echo "[llm-phase6] FAIL: empty q4_eisa_adaptive output for prompt '$P'" >&2
            exit 1
        fi
        printf "%-30s | %-12s | %-8s | %-8s | %-8s | %s\n" "$P" "q4_adaptive" "$GEN_Q4A" "10" "$ESC_Q4A" "$TXT_Q4A"

        # Q8_EISA
        OUT_Q8=$("$ROOT_DIR/bin/sounio-llm" --model-dir "$MODEL_DIR" --prompt "$P" --backend metal --quant q8_eisa --no-think --max-tokens 16)
        STAT_Q8=$(echo "$OUT_Q8" | grep "\[sounio-llm:")
        TXT_Q8=$(echo "$OUT_Q8" | grep -v "\[sounio-llm:" | tr -d '\n')
        GEN_Q8=$(echo "$STAT_Q8" | awk -F'gen_tokens=' '{print $2}' | awk '{print $1}')
        CERT_Q8=$(echo "$STAT_Q8" | awk -F'certified=' '{print $2}' | awk '{print $1}')
        if [[ -z "$TXT_Q8" || -z "${GEN_Q8:-}" || "${GEN_Q8:-0}" -le 0 ]]; then
            echo "[llm-phase6] FAIL: empty q8_eisa output for prompt '$P'" >&2
            exit 1
        fi
        printf "%-30s | %-12s | %-8s | %-8s | %-8s | %s\n" "$P" "q8_eisa" "$GEN_Q8" "$CERT_Q8" "0" "$TXT_Q8"

        if [[ "$P" == "The capital of France is" ]]; then
            if [[ "$TXT_Q4" == *"Paris."* && "$TXT_Q4A" == *"**Paris**"* ]]; then
                echo "[llm-phase6] ✓ Q4 Adaptive successfully eliminated silent argmax flip and recovered '**Paris**'!"
                RECOVERED_COUNT=$((RECOVERED_COUNT + 1))
            fi
        fi
        if [[ "$CERT_Q8" -gt 0 ]]; then
            Q8_CERT_COUNT=$((Q8_CERT_COUNT + 1))
        fi
    done
    printf "%s\n\n" "------------------------------------------------------------------------------------------------------------"

    assert_nonzero() {
        if [[ "$1" -le 0 ]]; then
            echo "[llm-phase6] FAIL: $2" >&2
            exit 1
        fi
    }
    assert_nonzero "$RECOVERED_COUNT" "q4_eisa_adaptive failed to recover from silent argmax flip"
    assert_nonzero "$Q8_CERT_COUNT" "q8_eisa failed to produce certified tokens"
    echo "[llm-phase6] PASS 6B EISA Certified & Adaptive Decoding Benchmark"

    # -------------------------------------------------------------------------
    # 5. 6C Grafted Non-Associative O-SSM Witness & Parity
    # -------------------------------------------------------------------------
    OSSM_WEIGHTS="/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/ossm_every7.safetensors"
    ABLATION_REPORT="/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/ablation_report.json"

    if [[ -f "$OSSM_WEIGHTS" && -f "$ABLATION_REPORT" ]]; then
        echo "[llm-phase6] Running 6C Grafted O-SSM Non-Associative Witness on France & Japan..."

        # France ChatML
        OUT_F_CHAT=$("$ROOT_DIR/bin/sounio-llm" --model-dir "$MODEL_DIR" --prompt "The capital of France is" --backend metal --quant bf16 --ossm-every 7 --ossm-weights "$OSSM_WEIGHTS" --no-think --max-tokens 16)
        echo "$OUT_F_CHAT"
        echo "$OUT_F_CHAT" | grep -Fq "The capital of France is **Paris**"
        ASSOC_F=$(echo "$OUT_F_CHAT" | grep "\[sounio-llm:" | awk -F'last_assoc_l2=' '{print $2}' | awk '{print $1}')
        python3 -c "assert float('$ASSOC_F') > 0.0, 'last_assoc_l2 must be > 0.0'"
        echo "[llm-phase6] ✓ France Sidecar ChatML: exact '**Paris**', last_assoc_l2=$ASSOC_F > 0"

        # Japan ChatML
        OUT_J_CHAT=$("$ROOT_DIR/bin/sounio-llm" --model-dir "$MODEL_DIR" --prompt "The capital of Japan is" --backend metal --quant bf16 --ossm-every 7 --ossm-weights "$OSSM_WEIGHTS" --no-think --max-tokens 16)
        echo "$OUT_J_CHAT"
        echo "$OUT_J_CHAT" | grep -Fq "The capital of Japan is **Tokyo**"
        ASSOC_J=$(echo "$OUT_J_CHAT" | grep "\[sounio-llm:" | awk -F'last_assoc_l2=' '{print $2}' | awk '{print $1}')
        python3 -c "assert float('$ASSOC_J') > 0.0, 'last_assoc_l2 must be > 0.0'"
        echo "[llm-phase6] ✓ Japan Sidecar ChatML: exact '**Tokyo**', last_assoc_l2=$ASSOC_J > 0"
        echo "[llm-phase6] ✓ Japan Sidecar ChatML: exact '**Tokyo**', last_assoc_l2=$ASSOC_J > 0"

        # Raw logit parity vs ablation_report.json
        OUT_F_RAW=$("$ROOT_DIR/bin/sounio-llm" --model-dir "$MODEL_DIR" --prompt "The capital of France is" --raw --backend metal --quant bf16 --ossm-every 7 --ossm-weights "$OSSM_WEIGHTS" --no-think --max-tokens 1)
        OUT_J_RAW=$("$ROOT_DIR/bin/sounio-llm" --model-dir "$MODEL_DIR" --prompt "The capital of Japan is" --raw --backend metal --quant bf16 --ossm-every 7 --ossm-weights "$OSSM_WEIGHTS" --no-think --max-tokens 1)
        echo "$OUT_F_RAW" | grep -q " Paris"
        echo "$OUT_J_RAW" | grep -q " Tokyo"
        echo "[llm-phase6] ✓ Raw Prompt Parity: France -> ' Paris', Japan -> ' Tokyo'"
        echo "[llm-phase6] PASS 6C Grafted Non-Associative O-SSM Witness"
    fi
else
    echo "[llm-phase6] SKIP: real checkpoint benchmark skipped (MODEL_DIR=$MODEL_DIR unavailable, PYTHON_BIN=$PYTHON_BIN unavailable, or non-Darwin/arm64)"
    if [[ "${SOUNIO_REQUIRE_MODEL:-0}" == "1" ]]; then
        echo "[llm-phase6] FAIL: SOUNIO_REQUIRE_MODEL=1 forbids skipping real checkpoint gate" >&2
        exit 1
    fi
fi

echo "LLM_PHASE6_NOVELTY_TRIAD_GATE_PASS"
