#!/usr/bin/env bash
# scripts/ci/llm_phase9_rank4_longctx_gate.sh — Phase 9 Sedenion Gated Delta SSM & Long-Context Gate
# Verifies:
# (1) Zero { ret; } stubs in stdlib/llm/ and 0 sounio-lint errors.
# (2) Zero sorry or axiom in formal/lean4/SounioSedenionProjector.lean,
#     presence of zd84_immortal_cols_0_8, zd84_immortal_rows_0_8, zd84_exact_immortal_coords,
#     and clean Lake build.
# (3) Build & run tests/llm/llm_phase9_rank4_longctx.sio on macOS ARM64 Metal
#     (asserting METAL_SEDENION_PARITY_OK and LLM_PHASE9_RANK4_LONGCTX_PASS)
#     AND Linux x86-64 ELF via souc-linux (asserting LLM_PHASE9_RANK4_LONGCTX_PASS).
# (4) Run Phase 8 integration gate (asserting LLM_PHASE8_SEDENION_SSM_GATE_PASS).
# (5) Strict Python validation of Phase 9 research ablation report
#     (parity <= 2%, 120 raw finite entries, immortality < 1e-6, Welch K=4,6,8,12, exact p-value keys).
# (6) Output LLM_PHASE9_RANK4_LONGCTX_GATE_PASS.
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

WORK="$(mktemp -d "${TMPDIR:-/tmp}/llm-phase9-gate.XXXXXX")"
cleanup() {
    rm -rf "$WORK"
}
trap cleanup EXIT

echo "[llm-phase9] compiler=$MADAROS_BIN"

PYTHON_BIN="/Users/demetriosagourakis/miniconda3/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

# -----------------------------------------------------------------------------
# 1. Check zero { ret; } stubs & sounio-lint 0 errors
# -----------------------------------------------------------------------------
echo "[llm-phase9] Checking zero { ret; } stubs in stdlib/llm/..."
if grep -rn "{ ret; }" "$ROOT_DIR/stdlib/llm/" >"$WORK/stubs.txt" 2>&1; then
    echo "[llm-phase9] ERROR: found { ret; } stubs in stdlib/llm/:" >&2
    cat "$WORK/stubs.txt" >&2
    exit 1
fi
echo "[llm-phase9] PASS zero { ret; } stubs"

echo "[llm-phase9] Checking sounio-lint on Phase 9 stdlib and test files..."
"$PYTHON_BIN" "$ROOT_DIR/scripts/dev/sounio-lint.py" --errors-only "$ROOT_DIR/stdlib/llm/sedenion_ssm.sio"
"$PYTHON_BIN" "$ROOT_DIR/scripts/dev/sounio-lint.py" --errors-only "$ROOT_DIR/stdlib/llm/mod.sio"
"$PYTHON_BIN" "$ROOT_DIR/scripts/dev/sounio-lint.py" --errors-only "$ROOT_DIR/tests/llm/llm_phase8_sedenion_ssm.sio"
"$PYTHON_BIN" "$ROOT_DIR/scripts/dev/sounio-lint.py" --errors-only "$ROOT_DIR/tests/llm/llm_phase9_rank4_longctx.sio"
echo "[llm-phase9] PASS sounio-lint 0 errors"

# -----------------------------------------------------------------------------
# 2. Check Lean 4: zero sorry/axiom, immortal theorems, and lake build
# -----------------------------------------------------------------------------
echo "[llm-phase9] Checking zero sorry or axiom in formal/lean4/SounioSedenionProjector.lean..."
if grep -nE "\bsorry\b|\baxiom\b" "$ROOT_DIR/formal/lean4/SounioSedenionProjector.lean" >"$WORK/lean_sorry.txt" 2>&1; then
    echo "[llm-phase9] ERROR: found sorry or axiom in formal/lean4/SounioSedenionProjector.lean:" >&2
    cat "$WORK/lean_sorry.txt" >&2
    exit 1
fi
echo "[llm-phase9] PASS zero sorry or axiom in SounioSedenionProjector.lean"

echo "[llm-phase9] Checking presence of immortal subspace theorems..."
for thm in zd84_immortal_cols_0_8 zd84_immortal_rows_0_8 zd84_exact_immortal_coords; do
    if ! grep -q "\b$thm\b" "$ROOT_DIR/formal/lean4/SounioSedenionProjector.lean"; then
        echo "[llm-phase9] ERROR: missing theorem $thm in SounioSedenionProjector.lean" >&2
        exit 1
    fi
done
echo "[llm-phase9] PASS presence of immortal subspace theorems"

if ! command -v lake >/dev/null 2>&1; then
    if [[ -d "$HOME/.elan/bin" ]]; then
        export PATH="$HOME/.elan/bin:$PATH"
    fi
fi

if command -v lake >/dev/null 2>&1; then
    echo "[llm-phase9] Building SounioSedenionProjector with Lake..."
    (cd "$ROOT_DIR/formal/lean4" && lake build SounioSedenionProjector)
    echo "[llm-phase9] PASS Lake build SounioSedenionProjector"
else
    echo "[llm-phase9] WARNING: lake not found in PATH or ~/.elan/bin; skipping lake build" >&2
fi

# -----------------------------------------------------------------------------
# 3. Build & run tests/llm/llm_phase9_rank4_longctx.sio on macOS ARM64 & Linux x86-64
# -----------------------------------------------------------------------------
if [[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]]; then
    echo "[llm-phase9] macOS ARM64: building and running tests/llm/llm_phase9_rank4_longctx.sio..."
    BIN_P9_MAC="$WORK/llm_phase9_mac"
    "$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase9_rank4_longctx.sio" \
        --target aarch64-macos -o "$BIN_P9_MAC" >"$WORK/mac-p9-build.log" 2>&1
    chmod +x "$BIN_P9_MAC"
    codesign -s - -f "$BIN_P9_MAC" >/dev/null 2>&1 || true

    export SOUNIO_METAL_RUNTIME="${SOUNIO_METAL_RUNTIME:-$ROOT_DIR/artifacts/self-hosted/libsounio_metal.dylib}"
    "$BIN_P9_MAC" >"$WORK/mac-p9.out"
    cat "$WORK/mac-p9.out"
    grep -qxF "METAL_SEDENION_PARITY_OK" "$WORK/mac-p9.out"
    grep -qxF "LLM_PHASE9_RANK4_LONGCTX_PASS" "$WORK/mac-p9.out"
    echo "[llm-phase9] PASS macOS ARM64 Metal: Phase 9 Rank-4 Long-Context unit & parity tests"
fi

echo "[llm-phase9] Building Linux x86-64 ELF test binary..."
BIN_P9_LINUX="$WORK/llm_phase9_linux.elf"
"$MADAROS_BIN" build "$ROOT_DIR/tests/llm/llm_phase9_rank4_longctx.sio" \
    --target x86_64-linux -o "$BIN_P9_LINUX" >"$WORK/linux-p9-build.log" 2>&1
chmod +x "$BIN_P9_LINUX"

if command -v limactl >/dev/null 2>&1 && limactl list 2>/dev/null | grep -q "^souc-linux[[:space:]]\+Running"; then
    echo "[llm-phase9] Running Linux x86-64 verification via Lima VM souc-linux..."
    limactl shell souc-linux rm -rf /tmp/llm_phase9_linux.elf /tmp/llm_phase9_run.elf
    limactl copy "$BIN_P9_LINUX" "souc-linux:/tmp/llm_phase9_linux.elf"

    limactl shell souc-linux bash -c '
set -euo pipefail
trap "rm -rf /tmp/llm_phase9_linux.elf /tmp/llm_phase9_run.elf" EXIT
chmod +x /tmp/llm_phase9_linux.elf

if [[ ! -e /lib64/ld-linux-x86-64.so.2 && -e /lib/ld-musl-x86_64.so.1 ]]; then
    python3 - /tmp/llm_phase9_linux.elf /tmp/llm_phase9_run.elf <<'\''PY'\''
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
    chmod +x /tmp/llm_phase9_run.elf
    ln -sf /lib/ld-musl-x86_64.so.1 /tmp/libdl.so.2 || true
    export LD_LIBRARY_PATH="/tmp:${LD_LIBRARY_PATH:-}"
    /tmp/llm_phase9_run.elf | grep -qxF "LLM_PHASE9_RANK4_LONGCTX_PASS"
else
    /tmp/llm_phase9_linux.elf | grep -qxF "LLM_PHASE9_RANK4_LONGCTX_PASS"
fi
'
    echo "[llm-phase9] PASS Linux x86-64 ELF via souc-linux (Phase 9 Rank-4 Long-Context unit tests)"
elif [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
    echo "[llm-phase9] Running Linux x86-64 verification on host Linux..."
    "$BIN_P9_LINUX" | grep -qxF "LLM_PHASE9_RANK4_LONGCTX_PASS"
    echo "[llm-phase9] PASS Linux x86-64 ELF on host (Phase 9 Rank-4 Long-Context unit tests)"
else
    echo "[llm-phase9] Running Linux x86-64 verification via Lima VM souc-linux..."
    limactl shell souc-linux rm -rf /tmp/llm_phase9_linux.elf /tmp/llm_phase9_run.elf
    limactl copy "$BIN_P9_LINUX" "souc-linux:/tmp/llm_phase9_linux.elf"
    limactl shell souc-linux bash -c '
set -euo pipefail
trap "rm -rf /tmp/llm_phase9_linux.elf /tmp/llm_phase9_run.elf" EXIT
chmod +x /tmp/llm_phase9_linux.elf

if [[ ! -e /lib64/ld-linux-x86-64.so.2 && -e /lib/ld-musl-x86_64.so.1 ]]; then
    python3 - /tmp/llm_phase9_linux.elf /tmp/llm_phase9_run.elf <<'\''PY'\''
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
    chmod +x /tmp/llm_phase9_run.elf
    ln -sf /lib/ld-musl-x86_64.so.1 /tmp/libdl.so.2 || true
    export LD_LIBRARY_PATH="/tmp:${LD_LIBRARY_PATH:-}"
    /tmp/llm_phase9_run.elf | grep -qxF "LLM_PHASE9_RANK4_LONGCTX_PASS"
else
    /tmp/llm_phase9_linux.elf | grep -qxF "LLM_PHASE9_RANK4_LONGCTX_PASS"
fi
'
    echo "[llm-phase9] PASS Linux x86-64 ELF via souc-linux (Phase 9 Rank-4 Long-Context unit tests)"
fi

# -----------------------------------------------------------------------------
# 4. Run Phase 8 Integration Gate
# -----------------------------------------------------------------------------
echo "[llm-phase9] Running Phase 8 Sedenion SSM integration gate..."
bash "$ROOT_DIR/scripts/ci/llm_phase8_sedenion_ssm_gate.sh" | grep -qxF "LLM_PHASE8_SEDENION_SSM_GATE_PASS"
echo "[llm-phase9] PASS Phase 8 integration gate"

# -----------------------------------------------------------------------------
# 5. Strict Python validation of Phase 9 ablation report
# -----------------------------------------------------------------------------
RESEARCH_DIR="/Users/demetriosagourakis/sounio-research"
REPORT_P9="$RESEARCH_DIR/checkpoints/Qwen3-0.6B-ossm-sidecar/phase9_rank4_longctx_report.json"

if [[ -d "$RESEARCH_DIR" ]]; then
    echo "[llm-phase9] Validating Phase 9 research ablation report at $REPORT_P9..."
    if [[ ! -f "$REPORT_P9" ]]; then
        echo "[llm-phase9] ERROR: report file not found at $REPORT_P9" >&2
        exit 1
    fi

    "$PYTHON_BIN" - "$REPORT_P9" <<'PY'
import json, math, sys

report_path = sys.argv[1]
with open(report_path, "r") as fp:
    data = json.load(fp)

# 1. Parity param_count checks
assert data["parity"]["param_count"]["part2"]["within_2pct"] is True, "part2 within_2pct not True"
assert data["parity"]["param_count"]["part3"]["within_2pct"] is True, "part3 within_2pct not True"

# 2. Config checks: T == 128, seeds == [42, 137, 271, 314, 577, 777, 1024, 1618, 2026, 4096]
assert data["config"]["T"] == 128, f"Unexpected config.T: {data['config'].get('T')}"
expected_seeds = [42, 137, 271, 314, 577, 777, 1024, 1618, 2026, 4096]
assert data["config"]["seeds"] == expected_seeds, f"Unexpected config.seeds: {data['config'].get('seeds')}"

# 3. Raw entries check: len(raw) == 120, all 5 metrics finite for all 6 models x 10 seeds x Parts 2 & 3
expected_models = [
    "SedenionGatedDeltaSSM",
    "SedenionDeltaSSM",
    "GatedDeltaNet4x4",
    "DeltaProductRank4",
    "GrassmannQR4SSM",
    "GatedDeltaNet16"
]
metrics = ["test_mse", "active_mse", "deleted_mse", "active_cos_sim", "deleted_norm"]

raw = data["raw"]
assert len(raw) == 120, f"Expected 120 raw entries, got {len(raw)}"

seen_combos = set()
for r in raw:
    part = r["part"]
    m = r["model"]
    s = r["seed"]
    assert part in (2, 3), f"Unexpected part: {part}"
    assert m in expected_models, f"Unexpected model: {m}"
    assert s in expected_seeds, f"Unexpected seed: {s}"
    seen_combos.add((part, m, s))
    for met in metrics:
        val = r[met]
        assert math.isfinite(val), f"Non-finite value {val} for part {part} {m} seed {s} {met}"

assert len(seen_combos) == 120, f"Expected 120 unique (part, model, seed) combos, got {len(seen_combos)}"

# 4. Part 1 checks: immortality < 1e-6 in float64 and float32, Welch K=4,6,8,12 present
p1 = data["part1_welch_immortal_and_speed"]
imm = p1["immortality_V72"]
assert imm["immortal_lt_1e-6_float64"] is True, "immortal_lt_1e-6_float64 is not True"
assert imm["immortal_lt_1e-6_float32"] is True, "immortal_lt_1e-6_float32 is not True"

welch = p1["welch_packing"]
for k in [4, 6, 8, 12]:
    k_key = f"K={k}"
    assert k_key in welch, f"Missing Welch key {k_key} in part1"

# 5. Paired p-value checks for both SedenionGatedDeltaSSM and SedenionDeltaSSM vs all 5 other models
sed_models = ["SedenionGatedDeltaSSM", "SedenionDeltaSSM"]

for part_key in ["part2_discrete_kv4d_T128", "part3_continuous_unseen_kv4d_T128"]:
    assert part_key in data, f"Missing {part_key} in report"
    part_data = data[part_key]
    assert "paired" in part_data, f"Missing paired in {part_key}"
    paired_dict = part_data["paired"]
    for met in metrics:
        assert met in paired_dict, f"Missing metric {met} in {part_key} paired"
        for sd in sed_models:
            assert sd in paired_dict[met], f"Missing {sd} in {part_key} paired[{met}]"
            base_targets = [m for m in expected_models if m != sd]
            for bm in base_targets:
                assert bm in paired_dict[met][sd], f"Missing baseline {bm} in {part_key} paired[{met}][{sd}]"
                entry = paired_dict[met][sd][bm]
                assert "ttest_rel_p" in entry, f"Missing ttest_rel_p in {part_key} paired[{met}][{sd}][{bm}]"
                assert "wilcoxon_p" in entry, f"Missing wilcoxon_p in {part_key} paired[{met}][{sd}][{bm}]"
                assert entry["ttest_rel_p"] is not None and math.isfinite(entry["ttest_rel_p"]), f"Non-finite ttest_rel_p for {part_key} {met} {sd} vs {bm}"
                assert entry["wilcoxon_p"] is not None and math.isfinite(entry["wilcoxon_p"]), f"Non-finite wilcoxon_p for {part_key} {met} {sd} vs {bm}"

print("[llm-phase9] ✓ phase9_rank4_longctx_report.json verified: parity <=2%, 120 finite raw entries, immortality <1e-6, Welch K=4,6,8,12, all paired p-values present")
PY
    echo "[llm-phase9] PASS Research ablation report validation"
else
    echo "[llm-phase9] Skipping local research report validation (remote CI environment without $RESEARCH_DIR)"
fi

# -----------------------------------------------------------------------------
# 6. Output Final Pass Marker
# -----------------------------------------------------------------------------
echo "LLM_PHASE9_RANK4_LONGCTX_GATE_PASS"
