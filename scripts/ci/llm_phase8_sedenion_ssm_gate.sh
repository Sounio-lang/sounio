#!/usr/bin/env bash
# scripts/ci/llm_phase8_sedenion_ssm_gate.sh — Phase 8 Sedenion SSM Integration Gate
# Verifies:
# (1) Zero { ret; } stubs in stdlib/llm/ and 0 sounio-lint errors.
# (2) Zero sorry or axiom in formal/lean4/SounioSedenionProjector.lean and clean Lake build.
# (3) tests/llm/llm_phase8_sedenion_ssm.sio on macOS ARM64 & Linux x86-64 ELF via souc-linux.
# (4) Research ablation report validation (parity <= 2%, 6 models x 10 seeds finite, exact p-value keys).
# (5) Output LLM_PHASE8_SEDENION_SSM_GATE_PASS.
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

WORK="$(mktemp -d "${TMPDIR:-/tmp}/llm-phase8-gate.XXXXXX")"
cleanup() {
    rm -rf "$WORK"
}
trap cleanup EXIT

echo "[llm-phase8] compiler=$MADAROS_BIN"

PYTHON_BIN="/Users/demetriosagourakis/miniconda3/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

# -----------------------------------------------------------------------------
# 1. Check zero { ret; } stubs & sounio-lint 0 errors
# -----------------------------------------------------------------------------
echo "[llm-phase8] Checking zero { ret; } stubs in stdlib/llm/..."
if grep -rn "{ ret; }" "$ROOT_DIR/stdlib/llm/" >"$WORK/stubs.txt" 2>&1; then
    echo "[llm-phase8] ERROR: found { ret; } stubs in stdlib/llm/:" >&2
    cat "$WORK/stubs.txt" >&2
    exit 1
fi
echo "[llm-phase8] PASS zero { ret; } stubs"

echo "[llm-phase8] Checking sounio-lint on Phase 8 stdlib and test files..."
"$PYTHON_BIN" "$ROOT_DIR/scripts/dev/sounio-lint.py" --errors-only "$ROOT_DIR/stdlib/llm/sedenion_ssm.sio"
"$PYTHON_BIN" "$ROOT_DIR/scripts/dev/sounio-lint.py" --errors-only "$ROOT_DIR/stdlib/llm/mod.sio"
"$PYTHON_BIN" "$ROOT_DIR/scripts/dev/sounio-lint.py" --errors-only "$ROOT_DIR/tests/llm/llm_phase8_sedenion_ssm.sio"
echo "[llm-phase8] PASS sounio-lint 0 errors"

# -----------------------------------------------------------------------------
# 2. Check Lean 4: zero sorry/axiom and lake build SounioSedenionProjector
# -----------------------------------------------------------------------------
echo "[llm-phase8] Checking zero sorry or axiom in formal/lean4/SounioSedenionProjector.lean..."
if grep -nE "\bsorry\b|\baxiom\b" "$ROOT_DIR/formal/lean4/SounioSedenionProjector.lean" >"$WORK/lean_sorry.txt" 2>&1; then
    echo "[llm-phase8] ERROR: found sorry or axiom in formal/lean4/SounioSedenionProjector.lean:" >&2
    cat "$WORK/lean_sorry.txt" >&2
    exit 1
fi
echo "[llm-phase8] PASS zero sorry or axiom in SounioSedenionProjector.lean"

if ! command -v lake >/dev/null 2>&1; then
    if [[ -d "$HOME/.elan/bin" ]]; then
        export PATH="$HOME/.elan/bin:$PATH"
    fi
fi

if command -v lake >/dev/null 2>&1; then
    echo "[llm-phase8] Building SounioSedenionProjector with Lake..."
    (cd "$ROOT_DIR/formal/lean4" && lake build SounioSedenionProjector)
    echo "[llm-phase8] PASS Lake build SounioSedenionProjector"
else
    echo "[llm-phase8] WARNING: lake not found in PATH or ~/.elan/bin; skipping lake build" >&2
fi

# -----------------------------------------------------------------------------
# 3. Build & run tests/llm/llm_phase8_sedenion_ssm.sio on macOS ARM64 & Linux x86-64
# -----------------------------------------------------------------------------
if [[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]]; then
    echo "[llm-phase8] macOS ARM64: building and running tests/llm/llm_phase8_sedenion_ssm.sio..."
    BIN_P8_MAC="$WORK/llm_phase8_mac"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase8_sedenion_ssm.sio" \
        --target aarch64-macos -o "$BIN_P8_MAC" >"$WORK/mac-p8-build.log" 2>&1
    chmod +x "$BIN_P8_MAC"
    codesign -s - -f "$BIN_P8_MAC" >/dev/null 2>&1 || true
    "$BIN_P8_MAC" >"$WORK/mac-p8.out"
    grep -qxF "LLM_PHASE8_SEDENION_SSM_PASS" "$WORK/mac-p8.out"
    echo "[llm-phase8] PASS macOS ARM64: Phase 8 Sedenion SSM unit tests"
fi

echo "[llm-phase8] Building Linux x86-64 ELF test binary..."
BIN_P8_LINUX="$WORK/llm_phase8_linux.elf"
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase8_sedenion_ssm.sio" \
    --target x86_64-linux -o "$BIN_P8_LINUX" >"$WORK/linux-p8-build.log" 2>&1
chmod +x "$BIN_P8_LINUX"

if command -v limactl >/dev/null 2>&1 && limactl list 2>/dev/null | grep -q "^souc-linux[[:space:]]\+Running"; then
    echo "[llm-phase8] Running Linux x86-64 verification via Lima VM souc-linux..."
    limactl shell souc-linux rm -rf /tmp/llm_phase8_linux.elf /tmp/llm_phase8_run.elf
    limactl copy "$BIN_P8_LINUX" "souc-linux:/tmp/llm_phase8_linux.elf"

    limactl shell souc-linux bash -c '
set -euo pipefail
trap "rm -rf /tmp/llm_phase8_linux.elf /tmp/llm_phase8_run.elf" EXIT
chmod +x /tmp/llm_phase8_linux.elf

if [[ ! -e /lib64/ld-linux-x86-64.so.2 && -e /lib/ld-musl-x86_64.so.1 ]]; then
    python3 - /tmp/llm_phase8_linux.elf /tmp/llm_phase8_run.elf <<'\''PY'\''
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
    chmod +x /tmp/llm_phase8_run.elf
    ln -sf /lib/ld-musl-x86_64.so.1 /tmp/libdl.so.2 || true
    export LD_LIBRARY_PATH="/tmp:${LD_LIBRARY_PATH:-}"
    /tmp/llm_phase8_run.elf | grep -qxF "LLM_PHASE8_SEDENION_SSM_PASS"
else
    /tmp/llm_phase8_linux.elf | grep -qxF "LLM_PHASE8_SEDENION_SSM_PASS"
fi
'
    echo "[llm-phase8] PASS Linux x86-64 ELF via souc-linux (Phase 8 Sedenion SSM unit tests)"
elif [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
    echo "[llm-phase8] Running Linux x86-64 verification on host Linux..."
    "$BIN_P8_LINUX" | grep -qxF "LLM_PHASE8_SEDENION_SSM_PASS"
    echo "[llm-phase8] PASS Linux x86-64 ELF on host (Phase 8 Sedenion SSM unit tests)"
else
    echo "[llm-phase8] Running Linux x86-64 verification via Lima VM souc-linux..."
    limactl shell souc-linux rm -rf /tmp/llm_phase8_linux.elf /tmp/llm_phase8_run.elf
    limactl copy "$BIN_P8_LINUX" "souc-linux:/tmp/llm_phase8_linux.elf"
    limactl shell souc-linux bash -c '
set -euo pipefail
trap "rm -rf /tmp/llm_phase8_linux.elf /tmp/llm_phase8_run.elf" EXIT
chmod +x /tmp/llm_phase8_linux.elf

if [[ ! -e /lib64/ld-linux-x86-64.so.2 && -e /lib/ld-musl-x86_64.so.1 ]]; then
    python3 - /tmp/llm_phase8_linux.elf /tmp/llm_phase8_run.elf <<'\''PY'\''
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
    chmod +x /tmp/llm_phase8_run.elf
    ln -sf /lib/ld-musl-x86_64.so.1 /tmp/libdl.so.2 || true
    export LD_LIBRARY_PATH="/tmp:${LD_LIBRARY_PATH:-}"
    /tmp/llm_phase8_run.elf | grep -qxF "LLM_PHASE8_SEDENION_SSM_PASS"
else
    /tmp/llm_phase8_linux.elf | grep -qxF "LLM_PHASE8_SEDENION_SSM_PASS"
fi
'
    echo "[llm-phase8] PASS Linux x86-64 ELF via souc-linux (Phase 8 Sedenion SSM unit tests)"
fi

# -----------------------------------------------------------------------------
# 4. Python validation of Phase 8 ablation report
# -----------------------------------------------------------------------------
REPORT_P8="/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/phase8_sedenion_ablation_report.json"

echo "[llm-phase8] Validating Phase 8 research ablation report..."
"$PYTHON_BIN" - "$REPORT_P8" <<'PY'
import json, math, sys

report_path = sys.argv[1]
with open(report_path, "r") as fp:
    data = json.load(fp)

# 1. Parity param_count checks
assert data["parity"]["param_count"]["part2"]["within_2pct"] is True, "part2 within_2pct not True"
assert data["parity"]["param_count"]["part3"]["within_2pct"] is True, "part3 within_2pct not True"

# 2. All 6 models x 10 seeds present with finite values
expected_models = [
    "SedenionDeltaSSM",
    "SedenionMulSSM",
    "TwoOctonionSSM",
    "FourQuatBlockSSM",
    "GatedDeltaNet16",
    "Random16BasisSSM"
]
expected_seeds = [42, 137, 271, 314, 577, 777, 1024, 1618, 2026, 4096]

assert data["config"]["seeds"] == expected_seeds, f"Unexpected seeds in config: {data['config']['seeds']}"

# Part 2 raw
raw_p2 = data["part2_kv4d_overwrite_delete"]["raw"]
assert len(raw_p2) == 60, f"part2 raw length expected 60, got {len(raw_p2)}"
p2_combos = set()
for r in raw_p2:
    m = r["model"]
    s = r["seed"]
    assert m in expected_models, f"Unknown model in part2: {m}"
    assert s in expected_seeds, f"Unknown seed in part2: {s}"
    p2_combos.add((m, s))
    for metric in ["test_mse", "active_cos_sim", "deleted_norm"]:
        val = r[metric]
        assert math.isfinite(val), f"Non-finite value {val} for {m} seed {s} {metric}"
assert len(p2_combos) == 60, "Not all 60 (model, seed) combos present in part2"

# Part 3 raw
raw_p3 = data["part3_real_16ch_eeg_and_seizure"]["raw"]
expected_datasets = ["eegmmidb_S001R01", "chb01", "chb03", "chb05", "chb11"]
assert len(raw_p3) == 6 * 10 * 5, f"part3 raw length expected 300, got {len(raw_p3)}"
p3_combos = set()
for r in raw_p3:
    m = r["model"]
    s = r["seed"]
    ds = r["dataset"]
    assert m in expected_models, f"Unknown model in part3: {m}"
    assert s in expected_seeds, f"Unknown seed in part3: {s}"
    assert ds in expected_datasets, f"Unknown dataset in part3: {ds}"
    p3_combos.add((m, s, ds))
    assert math.isfinite(r["mse_test_ii"]), f"Non-finite mse_test_ii for {m} {s} {ds}"
    if ds.startswith("chb"):
        assert math.isfinite(r["mse_ictal"]), f"Non-finite mse_ictal for {m} {s} {ds}"
        assert math.isfinite(r["disruption_ratio"]), f"Non-finite disruption_ratio for {m} {s} {ds}"
assert len(p3_combos) == 300, "Not all 300 combos present in part3"

# 3. Exact p-value keys present
baseline_models = [m for m in expected_models if m != "SedenionDeltaSSM"]

p2_pvals = data["part2_kv4d_overwrite_delete"]
for metric_key in [
    "paired_vs_SedenionDeltaSSM_test_mse",
    "paired_vs_SedenionDeltaSSM_active_cos_sim",
    "paired_vs_SedenionDeltaSSM_deleted_norm"
]:
    assert metric_key in p2_pvals, f"Missing {metric_key} in part2"
    for bm in baseline_models:
        assert bm in p2_pvals[metric_key], f"Missing {bm} in part2 {metric_key}"
        entry = p2_pvals[metric_key][bm]
        for pkey in ["ttest_rel_p", "wilcoxon_p", "mean_diff_sed_minus_base", "sed_wins_n", "n"]:
            assert pkey in entry, f"Missing {pkey} in part2 {metric_key}[{bm}]"
            assert math.isfinite(entry[pkey]), f"Non-finite {pkey} in part2 {metric_key}[{bm}]"

p3_pooled = data["part3_real_16ch_eeg_and_seizure"]["chbmit_pooled_4patients_x_10seeds"]
for metric in ["mse_test_ii", "mse_ictal", "disruption_ratio"]:
    assert metric in p3_pooled, f"Missing {metric} in part3 chbmit_pooled"
    pair_block = p3_pooled[metric]["paired_vs_SedenionDeltaSSM_40pairs"]
    for bm in baseline_models:
        assert bm in pair_block, f"Missing {bm} in part3 pooled {metric}"
        entry = pair_block[bm]
        for pkey in ["ttest_rel_p", "wilcoxon_p", "mean_diff_sed_minus_base", "sed_wins_n", "n"]:
            assert pkey in entry, f"Missing {pkey} in part3 pooled {metric}[{bm}]"
            assert math.isfinite(entry[pkey]), f"Non-finite {pkey} in part3 pooled {metric}[{bm}]"

print("[llm-phase8] ✓ phase8_sedenion_ablation_report.json verified: parity <=2%, 6 models x 10 seeds finite, exact p-value keys present")
PY
echo "[llm-phase8] PASS Research ablation report validation"

# -----------------------------------------------------------------------------
# 5. Output Final Pass Marker
# -----------------------------------------------------------------------------
echo "LLM_PHASE8_SEDENION_SSM_GATE_PASS"
