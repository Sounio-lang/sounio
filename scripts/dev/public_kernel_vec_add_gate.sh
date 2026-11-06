#!/usr/bin/env bash
# Run inside an authorized NVIDIA allocation. No CPU fallback or skip.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/lib/resolve_souc.sh
sounio_require_souc
OUT="$(mktemp -d "${SIO31_ARTIFACT_ROOT:-/tmp}/sio31-gate.XXXXXX")"
printf 'Evidence directory: %s\n' "$OUT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
TARGET="cuda-sm80"
"$SOUC_BIN" --version > "$OUT/version.log"
"$SOUC_BIN" info > "$OUT/compiler-info.log"
"$SOUC_BIN" check examples/kernel_vec_add.sio > "$OUT/check.log" 2>&1
"$SOUC_BIN" build examples/kernel_vec_add.sio --backend gpu \
    --gpu-target "$TARGET" -o "$OUT/vec_add.ptx" > "$OUT/build.log" 2>&1
"$SOUC_BIN" build examples/kernel_vec_add.sio --backend gpu \
    -o "$OUT/default.ptx" > "$OUT/default.build.log" 2>&1
cmp "$OUT/vec_add.ptx" "$OUT/default.ptx"
python3 - "$OUT/vec_add.ptx" <<'PY'
from pathlib import Path
import sys
ptx = Path(sys.argv[1]).read_text()
for required in (".target sm_80\n", "ld.global.f64", "st.global.f64", "add.f64"):
    if required not in ptx:
        raise SystemExit("missing public PTX contract: " + required)
PY
python3 tests/gpu/public_kernel_vec_add.py "$OUT/vec_add.ptx" | tee "$OUT/runtime.json"

# Mutate the source, not the emitted PTX: the exact same compiler must produce
# a loadable but numerically wrong kernel when the store is removed.
python3 - "$OUT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
source = Path("examples/kernel_vec_add.sio").read_text()
store = "c[tid as usize] = a[tid as usize] + b[tid as usize]"
if source.count(store) != 1:
    raise SystemExit("negative control requires exactly one store")
(root / "no_store.sio").write_text(source.replace(store, ""))
PY
"$SOUC_BIN" check "$OUT/no_store.sio" > "$OUT/no_store.check.log" 2>&1
"$SOUC_BIN" build "$OUT/no_store.sio" --backend gpu \
    --gpu-target "$TARGET" -o "$OUT/no_store.ptx" > "$OUT/no_store.build.log" 2>&1
if python3 tests/gpu/public_kernel_vec_add.py "$OUT/no_store.ptx" > "$OUT/no_store.runtime.log" 2>&1; then
    printf 'FAIL: missing-store control unexpectedly passed\n' >&2
    exit 1
fi
for CASE in var_local if_value; do
    if "$SOUC_BIN" build "tests/gpu/literals/$CASE.sio" --backend gpu \
        --gpu-target "$TARGET" -o "$OUT/$CASE.ptx" > "$OUT/$CASE.log" 2>&1; then
        printf 'FAIL: %s unexpectedly compiled\n' "$CASE" >&2
        exit 1
    fi
done
python3 - "$OUT" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
root = Path(sys.argv[1])
log = (root / "no_store.runtime.log").read_text()
if "AssertionError: n=1 offset=1: got -4096.0, expected 7.75" not in log:
    raise SystemExit("negative control failed for an unexpected reason:\n" + log)
for case, diagnostic in (("var_local", "GPU codegen refused: a `var` local"),
                         ("if_value", "HLIR_LOWERING_REFUSED")):
    if (root / (case + ".ptx")).exists():
        raise SystemExit(case + ": refusal wrote an artifact")
    if diagnostic not in (root / (case + ".log")).read_text():
        raise SystemExit(case + ": unexpected refusal")
info = (root / "compiler-info.log").read_text().splitlines()
raw = [line.split(":", 1)[1].strip() for line in info if line.startswith("raw_elf:")]
if len(raw) != 1:
    raise SystemExit("compiler info did not identify exactly one raw artifact")
paths = [Path("examples/kernel_vec_add.sio"),
         Path("tests/gpu/public_kernel_vec_add.py"), Path(raw[0]),
         Path("self-hosted/compiler/main.sio"),
         Path("self-hosted/gpu/hlir_to_gpu.sio"),
         Path("self-hosted/gpu/lower_to_ptx.sio"), *sorted(root.glob("*.ptx"))]
receipt = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(root / "hashes.json").write_text(json.dumps(receipt, indent=2) + "\n")
revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
(root / "provenance.json").write_text(json.dumps({
    "source_base": os.environ.get("SIO31_SOURCE_REVISION") or revision.stdout.strip() or "unknown",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "requested_target": "cuda-sm80", "emitted_target": "sm_80",
    "scope": "default PTX mode, one block, f64 vector add",
}, indent=2) + "\n")
print("PASS: sm80, all sizes, inputs/guards, missing-store control, ALLOCA/PHI refusals")
PY
