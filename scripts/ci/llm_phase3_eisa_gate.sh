#!/usr/bin/env bash
# Phase 3 EISA Quantization Gate:
#   1. Memory budget verification: qwen3_eisa_q4_bytes(8190735360) == 5119209600.
#   2. Test A: Qwen3-8B width (64 x 4096) Block-32 Q4_EISA and Q8_EISA quantization.
#      - Strict containment: abs(y_bf16 - val) <= err for 100% of rows on CPU and GPU.
#      - CPU vs GPU parity (Metal NoCopy on Darwin/arm64; PTX ABI witness on Linux x86-64 ELF).
#      - Epistemic tightness: mean(err_q8) < mean(err_q4) * 0.1 (> 10x tighter).
#   3. Test B: Constructive Argmax Discrimination & Epistemic Separation.
#      - Close logits where Q4 is uncertified, but Q8 is certified.
#      - Strict separation of err and u: non-zero u_x increases combined_u while err and val remain unchanged.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/llm-phase3-gate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "[llm-phase3] compiler=$MADAROS_BIN"

if [[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]]; then
    RUNTIME_DYLIB="${SOUNIO_METAL_RUNTIME:-$ROOT_DIR/artifacts/self-hosted/libsounio_metal.dylib}"
    if [[ ! -f "$RUNTIME_DYLIB" ]]; then
        bash "$ROOT_DIR/scripts/dev/build_native_metal_runtime.sh" "$RUNTIME_DYLIB"
    fi
    export SOUNIO_METAL_RUNTIME="$RUNTIME_DYLIB"

    BIN_MAC="$WORK/llm_phase3_eisa_quant_mac"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase3_eisa_quant.sio" \
        --target aarch64-macos -o "$BIN_MAC" >"$WORK/mac-build.log" 2>&1
    if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body' "$WORK/mac-build.log"; then
        echo "[llm-phase3] FAIL: macOS build refused or missed lowered body" >&2
        cat "$WORK/mac-build.log" >&2
        exit 1
    fi
    chmod +x "$BIN_MAC"
    codesign -s - -f "$BIN_MAC"
    codesign --verify "$BIN_MAC"

    "$BIN_MAC" >"$WORK/mac-run.stdout" 2>"$WORK/mac-run.stderr"
    grep -qxF "LLM_PHASE3_EISA_QUANT_PASS" "$WORK/mac-run.stdout"
    echo "[llm-phase3] PASS macOS ARM64 Metal NoCopy (Q4/Q8 EISA dual-accumulation GEMV & Argmax Cert)"
fi

BIN_LINUX="$WORK/llm_phase3_eisa_quant_linux.elf"
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase3_eisa_quant.sio" \
    --target x86_64-linux -o "$BIN_LINUX" >"$WORK/linux-build.log" 2>&1
if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body' "$WORK/linux-build.log"; then
    echo "[llm-phase3] FAIL: Linux build refused or missed lowered body" >&2
    cat "$WORK/linux-build.log" >&2
    exit 1
fi
chmod +x "$BIN_LINUX"

run_linux_verification() {
    local runner_prefix="$1"
    local vm_work="/tmp/llm-phase3-vm-$$"
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
            echo "[llm-phase3] linux mode=cpu_mock_driver (no /dev/nvidiactl)"
        else
            echo "[llm-phase3] linux mode=real_nvidia_driver"
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
            echo "[llm-phase3] linux interp=musl_pt_interp_adapter"
        fi
        "$exec_lin" >"$WORK/linux-run.stdout" 2>"$WORK/linux-run.stderr"
        grep -qxF "LLM_PHASE3_EISA_QUANT_PASS" "$WORK/linux-run.stdout"
    else
        limactl shell souc-linux bash -c "mkdir -p '$vm_work'"
        limactl copy "$BIN_LINUX" "souc-linux:$vm_work/llm_phase3_eisa_quant_linux.elf"
        limactl copy "$ROOT_DIR/tests/fixtures/llm/cuda_driver_mock_witness.c" "souc-linux:$vm_work/cuda_driver_mock_witness.c"
        limactl shell souc-linux bash -s -- "$vm_work" <<'EOS'
set -euo pipefail
VM_WORK="$1"
trap 'rm -rf "$VM_WORK"' EXIT

chmod +x "$VM_WORK/llm_phase3_eisa_quant_linux.elf"
readelf -d "$VM_WORK/llm_phase3_eisa_quant_linux.elf" > "$VM_WORK/dyn.txt"
grep -q "libdl.so.2" "$VM_WORK/dyn.txt"
NEEDED_CNT="$(grep -c "Shared library:" "$VM_WORK/dyn.txt" || true)"
test "$NEEDED_CNT" -eq 1

if [[ ! -e /dev/nvidiactl ]]; then
    gcc -shared -fPIC -O2 -nostdlib "$VM_WORK/cuda_driver_mock_witness.c" -o "$VM_WORK/libcuda_mock.so"
    export SOUNIO_CUDA_DRIVER_LIB="$VM_WORK/libcuda_mock.so"
    echo "[llm-phase3] linux mode=lima_cpu_mock_driver (no /dev/nvidiactl)"
else
    echo "[llm-phase3] linux mode=lima_real_nvidia_driver"
fi

EXEC_BIN="$VM_WORK/llm_phase3_eisa_quant_linux.elf"
if [[ ! -e /lib64/ld-linux-x86-64.so.2 && -e /lib/ld-musl-x86_64.so.1 ]]; then
    ln -sf /lib/ld-musl-x86_64.so.1 "$VM_WORK/libdl.so.2"
    export LD_LIBRARY_PATH="$VM_WORK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    python3 - "$VM_WORK/llm_phase3_eisa_quant_linux.elf" "$VM_WORK/run_musl.elf" <<'PY'
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
    echo "[llm-phase3] linux interp=musl_pt_interp_adapter"
fi

"$EXEC_BIN" >"$VM_WORK/out.txt"
grep -qxF "LLM_PHASE3_EISA_QUANT_PASS" "$VM_WORK/out.txt"
EOS
    fi
    echo "[llm-phase3] PASS Linux x86-64 ELF + PTX + CUDA Driver ABI (Q4/Q8 EISA dual-accumulation GEMV & Argmax Cert)"
}

if [[ "$(uname -s)" == Linux ]]; then
    run_linux_verification "local"
elif command -v limactl >/dev/null 2>&1 && limactl list 2>/dev/null | grep -q "^souc-linux[[:space:]]\+Running"; then
    run_linux_verification "lima"
else
    echo "[llm-phase3] ERROR: neither Linux host nor running Lima VM 'souc-linux' available" >&2
    exit 2
fi

echo "LLM_PHASE3_EISA_GATE_PASS"
