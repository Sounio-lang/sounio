#!/usr/bin/env bash
# Fase 0 Foundation Gate:
#   1. macOS ARM64 + real Metal NoCopy over a 2.5 GiB sparse shard (2.25 GiB offset)
#   2. Linux x86-64 ELF + PTX + libcuda.so.1 / SOUNIO_CUDA_DRIVER_LIB ABI witness
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/llm-phase0-gate.XXXXXX")"
MAC_SHARD_CFG="/tmp/sounio_llm_phase0_shard_path.txt"
trap 'rm -rf "$WORK" "$MAC_SHARD_CFG"' EXIT

SHARD_BYTES=2684354560
OFFSET_BYTES=2415919104

create_sparse_shard() {
    local dst="$1"
    python3 - "$dst" "$SHARD_BYTES" "$OFFSET_BYTES" <<'PY'
import os, struct, sys
path, total, off = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, "wb") as f:
    f.truncate(total)
    f.seek(off)
    f.write(struct.pack("<6H", 0x3F80, 0xC020, 0x4049, 0x0000, 0x8000, 0x477F))
PY
}

echo "[llm-phase0] compiler=$MADAROS_BIN"

if [[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]]; then
    RUNTIME_DYLIB="${SOUNIO_METAL_RUNTIME:-$ROOT_DIR/artifacts/self-hosted/libsounio_metal.dylib}"
    if [[ ! -f "$RUNTIME_DYLIB" ]]; then
        bash "$ROOT_DIR/scripts/dev/build_native_metal_runtime.sh" "$RUNTIME_DYLIB"
    fi
    export SOUNIO_METAL_RUNTIME="$RUNTIME_DYLIB"

    SHARD_MAC="$WORK/shard_2_5gib.bin"
    create_sparse_shard "$SHARD_MAC"

    BIN_MAC="$WORK/llm_phase0_metal_nocopy"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase0_metal_nocopy.sio" \
        --target aarch64-macos -o "$BIN_MAC" >"$WORK/mac-build.log" 2>&1
    if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body' "$WORK/mac-build.log"; then
        echo "[llm-phase0] FAIL: macOS build refused or missed lowered body" >&2
        cat "$WORK/mac-build.log" >&2
        exit 1
    fi
    chmod +x "$BIN_MAC"
    codesign -s - -f "$BIN_MAC"
    codesign --verify "$BIN_MAC"

    printf '%s' "$SHARD_MAC" > "$MAC_SHARD_CFG"
    "$BIN_MAC" "$SHARD_MAC" >"$WORK/mac-run.stdout" 2>"$WORK/mac-run.stderr"
    grep -qxF "LLM_PHASE0_METAL_NOCOPY_PASS" "$WORK/mac-run.stdout"
    echo "[llm-phase0] PASS macOS ARM64 Metal NoCopy (2.5 GiB shard, offset 2.25 GiB)"
fi

BIN_LINUX="$WORK/llm_phase0_linux_cuda_ptx.elf"
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase0_linux_cuda_ptx.sio" \
    --target x86_64-linux -o "$BIN_LINUX" >"$WORK/linux-build.log" 2>&1
if grep -Eq 'NATIVE_REFUSAL|missing_lowered_body' "$WORK/linux-build.log"; then
    echo "[llm-phase0] FAIL: Linux build refused or missed lowered body" >&2
    cat "$WORK/linux-build.log" >&2
    exit 1
fi
chmod +x "$BIN_LINUX"

run_linux_verification() {
    local runner_prefix="$1"
    local vm_work="/tmp/llm-phase0-vm-$$"
    if [[ "$runner_prefix" == "local" ]]; then
        readelf -d "$BIN_LINUX" >"$WORK/linux-dyn.txt"
        grep -q "libdl.so.2" "$WORK/linux-dyn.txt"
        local needed_cnt
        needed_cnt="$(grep -c "Shared library:" "$WORK/linux-dyn.txt" || true)"
        test "$needed_cnt" -eq 1

        local shard_lin="$WORK/shard_2_5gib_linux.bin"
        create_sparse_shard "$shard_lin"

        if [[ ! -e /dev/nvidiactl ]]; then
            gcc -shared -fPIC -O2 -nostdlib \
                "$ROOT_DIR/tests/fixtures/llm/cuda_driver_mock_witness.c" \
                -o "$WORK/libcuda_mock.so"
            export SOUNIO_CUDA_DRIVER_LIB="$WORK/libcuda_mock.so"
            echo "[llm-phase0] linux mode=cpu_mock_driver (no /dev/nvidiactl)"
        else
            echo "[llm-phase0] linux mode=real_nvidia_driver"
        fi
        printf '%s' "$shard_lin" > /tmp/sounio_llm_phase0_linux_shard.txt
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
            echo "[llm-phase0] linux interp=musl_pt_interp_adapter"
        fi
        "$exec_lin" "$shard_lin" >"$WORK/linux-run.stdout" 2>"$WORK/linux-run.stderr"
        rm -f /tmp/sounio_llm_phase0_linux_shard.txt
        grep -qxF "LLM_PHASE0_LINUX_CUDA_PTX_PASS" "$WORK/linux-run.stdout"
    else
        limactl shell souc-linux bash -c "mkdir -p '$vm_work'"
        limactl copy "$BIN_LINUX" "souc-linux:$vm_work/llm_phase0_linux_cuda_ptx.elf"
        limactl copy "$ROOT_DIR/tests/fixtures/llm/cuda_driver_mock_witness.c" "souc-linux:$vm_work/cuda_driver_mock_witness.c"
        limactl shell souc-linux bash -s -- "$vm_work" "$SHARD_BYTES" "$OFFSET_BYTES" <<'EOS'
set -euo pipefail
VM_WORK="$1"
SHARD_BYTES="$2"
OFFSET_BYTES="$3"
trap 'rm -rf "$VM_WORK" /tmp/sounio_llm_phase0_linux_shard.txt' EXIT

chmod +x "$VM_WORK/llm_phase0_linux_cuda_ptx.elf"
readelf -d "$VM_WORK/llm_phase0_linux_cuda_ptx.elf" > "$VM_WORK/dyn.txt"
grep -q "libdl.so.2" "$VM_WORK/dyn.txt"
NEEDED_CNT="$(grep -c "Shared library:" "$VM_WORK/dyn.txt" || true)"
test "$NEEDED_CNT" -eq 1

python3 - "$VM_WORK/shard.bin" "$SHARD_BYTES" "$OFFSET_BYTES" <<'PY'
import struct, sys
path, total, off = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, "wb") as f:
    f.truncate(total)
    f.seek(off)
    f.write(struct.pack("<6H", 0x3F80, 0xC020, 0x4049, 0x0000, 0x8000, 0x477F))
PY

if [[ ! -e /dev/nvidiactl ]]; then
    gcc -shared -fPIC -O2 -nostdlib "$VM_WORK/cuda_driver_mock_witness.c" -o "$VM_WORK/libcuda_mock.so"
    export SOUNIO_CUDA_DRIVER_LIB="$VM_WORK/libcuda_mock.so"
    echo "[llm-phase0] linux mode=lima_cpu_mock_driver (no /dev/nvidiactl)"
else
    echo "[llm-phase0] linux mode=lima_real_nvidia_driver"
fi

EXEC_BIN="$VM_WORK/llm_phase0_linux_cuda_ptx.elf"
if [[ ! -e /lib64/ld-linux-x86-64.so.2 && -e /lib/ld-musl-x86_64.so.1 ]]; then
    ln -sf /lib/ld-musl-x86_64.so.1 "$VM_WORK/libdl.so.2"
    export LD_LIBRARY_PATH="$VM_WORK${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    python3 - "$VM_WORK/llm_phase0_linux_cuda_ptx.elf" "$VM_WORK/run_musl.elf" <<'PY'
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
    echo "[llm-phase0] linux interp=musl_pt_interp_adapter"
fi

printf '%s' "$VM_WORK/shard.bin" > /tmp/sounio_llm_phase0_linux_shard.txt
"$EXEC_BIN" "$VM_WORK/shard.bin" >"$VM_WORK/out.txt"
grep -qxF "LLM_PHASE0_LINUX_CUDA_PTX_PASS" "$VM_WORK/out.txt"
EOS
    fi
    echo "[llm-phase0] PASS Linux x86-64 ELF + PTX + CUDA Driver ABI (2.5 GiB shard, offset 2.25 GiB)"
}

if [[ "$(uname -s)" == Linux ]]; then
    run_linux_verification "local"
elif command -v limactl >/dev/null 2>&1 && limactl list 2>/dev/null | grep -q "^souc-linux[[:space:]]\+Running"; then
    run_linux_verification "lima"
else
    echo "[llm-phase0] ERROR: neither Linux host nor running Lima VM 'souc-linux' available" >&2
    exit 2
fi

echo "LLM_PHASE0_FOUNDATION_GATE_PASS"
