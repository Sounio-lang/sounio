#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="${SOUNIO_SEMANTIC_ORC_INPUT_DIR:-/workspace/hyperbolic-semantic-networks/data/cpc2026/sounio_input}"
REGIME="${SOUNIO_SEMANTIC_ORC_REGIME:-normative}"
SUPPORT_INDEX="${SOUNIO_SEMANTIC_ORC_SUPPORT_INDEX:-0}"
SUPPORT_STRIDE="${SOUNIO_SEMANTIC_ORC_SUPPORT_STRIDE:-16}"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_KAXI_RUNTIME_DIR:-$(mktemp -d /tmp/sounio-semantic-orc-kaxi-runtime.XXXXXX)}"
SOURCE="$OUT_DIR/swow16_fixture_gate.sio"
MANIFEST="$OUT_DIR/swow16_fixture_manifest.json"
KAXI_PACK="$OUT_DIR/swow16_kaxi_pack.json"
PTX="${SOUNIO_KAXI_PTX:-$OUT_DIR/sinkhorn16.ptx}"
RUNNER="${SOUNIO_KAXI_RUNNER:-$OUT_DIR/kaxi_ptx_runner}"
RUN_OUT="$OUT_DIR/kaxi_runtime.out"
RESULT="$OUT_DIR/swow16_kaxi_runtime_result.json"

mkdir -p "$OUT_DIR"

echo "[semantic-orc-kaxi-runtime] input=$INPUT_DIR"
echo "[semantic-orc-kaxi-runtime] regime=$REGIME"
echo "[semantic-orc-kaxi-runtime] support_index=$SUPPORT_INDEX"
echo "[semantic-orc-kaxi-runtime] out=$OUT_DIR"

SOUNIO_SEMANTIC_ORC_INPUT_DIR="$INPUT_DIR" \
SOUNIO_SEMANTIC_ORC_REGIME="$REGIME" \
SOUNIO_SEMANTIC_ORC_SUPPORT_INDEX="$SUPPORT_INDEX" \
SOUNIO_SEMANTIC_ORC_SUPPORT_STRIDE="$SUPPORT_STRIDE" \
SOUNIO_SEMANTIC_ORC_KAXI_PACK_DIR="$OUT_DIR/pack_gate" \
  bash scripts/ci/semantic_orc_swow16_kaxi_pack_gate.sh >/dev/null
python3 scripts/research/semantic_orc_swow16_fixture.py \
  --input-dir "$INPUT_DIR" \
  --regime "$REGIME" \
  --support-index "$SUPPORT_INDEX" \
  --support-stride "$SUPPORT_STRIDE" \
  --sounio-out "$SOURCE" \
  --manifest "$MANIFEST" \
  --kaxi-pack "$KAXI_PACK"

emit_skipped() {
  local reason="$1"
  python3 - "$RESULT" "$reason" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
reason = sys.argv[2]
path.write_text(json.dumps({
    "schema": "sounio.semantic_orc.swow16_kaxi_runtime_result.v1",
    "status": "skipped",
    "reason": reason,
    "boundaries": [
        "no_gpu_runtime_claim",
        "pack_gate_remains_cpu_authority_for_this_environment",
    ],
}, indent=2, sort_keys=True) + "\n")
PY
  echo "[semantic-orc-kaxi-runtime] SKIPPED ($reason) artifact=$RESULT"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  emit_skipped "nvidia_smi_missing"
  exit 0
fi
if ! nvidia-smi >/dev/null 2>&1; then
  emit_skipped "gpu_inaccessible_or_missing"
  exit 0
fi

if [[ ! -x "$RUNNER" ]]; then
  if ! command -v cc >/dev/null 2>&1; then
    emit_skipped "cc_missing_and_no_prebuilt_runner"
    exit 0
  fi
  if ! cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -lm -o "$RUNNER"; then
    echo "[semantic-orc-kaxi-runtime] FAIL: runner_build_failed" >&2
    exit 1
  fi
fi

if [[ ! -s "$PTX" ]]; then
  if ! ./bin/kretikos kaxi-emit-ptx sinkhorn16 --f32 -o "$PTX" >/dev/null 2>"$OUT_DIR/ptx_emit.err"; then
    echo "[semantic-orc-kaxi-runtime] FAIL: PTX emission failed" >&2
    cat "$OUT_DIR/ptx_emit.err" >&2
    exit 1
  fi
fi

INIT_MEM="$(python3 - "$KAXI_PACK" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["init_mem_csv"])
PY
)"

"$RUNNER" "$PTX" --threads 1 --mem-words 320 --init-mem "$INIT_MEM" --type f32 >"$RUN_OUT"

python3 - "$KAXI_PACK" "$RUN_OUT" "$RESULT" <<'PY'
import json
import math
import pathlib
import re
import sys

pack_path, run_path, result_path = map(pathlib.Path, sys.argv[1:4])
pack = json.loads(pack_path.read_text())
text = run_path.read_text()
status = re.search(r"sounio_kaxi_runtime status=([^ ]+) reason=([^ ]+) stage=([^ ]+) cuda_result=([0-9-]+)", text)
if not status or status.group(1) != "pass":
    raise SystemExit("runtime did not report pass")
mem_line = next((line for line in text.splitlines() if line.startswith("MEM:")), "")
mem = [float(value) for value in mem_line.split()[1:]]
if len(mem) < 320:
    raise SystemExit("runtime MEM output has fewer than 320 values")
u_gpu = mem[288:304]
v_gpu = mem[304:320]
oracle = pack["oracle"]
u_ref = [float(value) for value in oracle["u"]]
v_ref = [float(value) for value in oracle["v"]]
maxdu = max(abs(a - b) for a, b in zip(u_gpu, u_ref))
maxdv = max(abs(a - b) for a, b in zip(v_gpu, v_ref))
ok = maxdu <= 0.005 and maxdv <= 0.005
doc = {
    "schema": "sounio.semantic_orc.swow16_kaxi_runtime_result.v1",
    "status": "pass" if ok else "fail",
    "regime": pack["regime"],
    "support_index": pack["support_index"],
    "support_stride": pack["support_stride"],
    "support_offset": pack["support_offset"],
    "runtime": {
        "status": status.group(1),
        "reason": status.group(2),
        "stage": status.group(3),
        "cuda_result": int(status.group(4)),
    },
    "semantic": {
        "maxdu_vs_pack_oracle": maxdu,
        "maxdv_vs_pack_oracle": maxdv,
        "tolerance": 0.005,
    },
    "boundaries": [
        "gpu_runtime_compared_to_pack_oracle_only",
        "entropic_regularized_transport_only",
        "clinical_claims_not_enforced",
        "no_convergence_theorem",
    ],
}
result_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
if not ok:
    raise SystemExit(f"runtime differs from pack oracle: maxdu={maxdu} maxdv={maxdv}")
PY

echo "[semantic-orc-kaxi-runtime] PASS artifact=$RESULT"
