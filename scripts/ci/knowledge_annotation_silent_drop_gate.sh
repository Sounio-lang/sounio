#!/usr/bin/env bash
# Accusation gate: an unrecognised Knowledge<…> annotation component is
# swallowed with no diagnostic (parser else-arm "Unknown component — skip").
#
# This is NOT a regression gate. It FAILS while the silence exists. When a
# named diagnostic starts refusing the unknown-component witness, this gate
# turns green. Do not add keywords here; that is a language change.
#
# Artifact: status + metrics {total, passed, failed, not_run}
# Receipt: tests/audit/KNOWLEDGE_ANNOTATION_SILENT_DROP_2026-08-19.md
# (docs/audit/ blocked this turn — topic-registry claimed by another lane)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/scripts/lib/resolve_souc.sh"
sounio_require_souc

UNKNOWN="$ROOT/tests/audit/knowledge_annotation_unknown_component.sio"
MEASURED="$ROOT/tests/audit/knowledge_annotation_measured.sio"
UNKNOWN_COERCE="$ROOT/tests/audit/knowledge_annotation_unknown_coerce.sio"
MEASURED_COERCE="$ROOT/tests/audit/knowledge_annotation_measured_coerce.sio"
OUT_DIR="${KNOWLEDGE_SILENT_DROP_OUT:-$ROOT/artifacts/audit/knowledge_annotation_silent_drop}"
mkdir -p "$OUT_DIR"

use_slurm=0
if command -v srun >/dev/null 2>&1 && [[ "${KNOWLEDGE_SILENT_DROP_SLURM:-1}" != "0" ]]; then
  use_slurm=1
fi

run_check() {
  local src="$1"
  local log="$2"
  "$SOUC_BIN" check "$src" >"$log" 2>&1
}

run_all_slurm() {
  tar -czf - -C "$ROOT" bin/madaros-linux-x86_64 bin/souc bin/madaros \
      tests/audit/knowledge_annotation_unknown_component.sio \
      tests/audit/knowledge_annotation_measured.sio \
      tests/audit/knowledge_annotation_unknown_coerce.sio \
      tests/audit/knowledge_annotation_measured_coerce.sio \
    | srun -p "${KNOWLEDGE_SILENT_DROP_PARTITION:-cpu-ops}" -N1 -n1 -c2 \
        --mem=8G --time=00:08:00 --chdir=/tmp --job-name=know-silent-drop \
        bash -c '
          set -euo pipefail
          export TMPDIR=/tmp
          WORKDIR=$(mktemp -d /tmp/know-silent.XXXXXX)
          cd "$WORKDIR"
          tar xzf -
          export MADAROS_STACK_KB=524288
          ulimit -s 1048576 || true
          mkdir -p /tmp/know-silent-out
          for f in tests/audit/knowledge_annotation_*.sio; do
            set +e
            ./bin/souc check "$f" >"/tmp/know-silent-out/$(basename "$f").log" 2>&1
            echo $? >"/tmp/know-silent-out/$(basename "$f").rc"
            set -e
          done
          tar -C /tmp/know-silent-out -czf - .
        ' >"$OUT_DIR/slurm_bundle.tar.gz"
  tar -C "$OUT_DIR" -xzf "$OUT_DIR/slurm_bundle.tar.gz"
}

total=4
passed=0
failed=0
not_run=0
silence="unknown"
unknown_rc=99
measured_rc=99
unknown_coerce_rc=99
measured_coerce_rc=99

if [[ "$use_slurm" -eq 1 && ! -x "$ROOT/bin/madaros-linux-x86_64" ]]; then
  not_run=4
  python3 - "$OUT_DIR/status.json" "$total" "$passed" "$failed" "$not_run" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
total, passed, failed, not_run = map(int, sys.argv[2:])
path.write_text(json.dumps({
    "schema": "sounio.knowledge_annotation_silent_drop.v1",
    "status": "not_run",
    "reason": "madaros_elf_missing",
    "metrics": {"total": total, "passed": passed, "failed": failed, "not_run": not_run},
}, indent=2) + "\n")
PY
  echo "status=not_run"
  echo "metrics {total=$total, passed=$passed, failed=$failed, not_run=$not_run}"
  exit 0
fi

if [[ "$use_slurm" -eq 1 ]]; then
  run_all_slurm
  cp "$OUT_DIR/knowledge_annotation_unknown_component.sio.log" "$OUT_DIR/unknown.log"
  cp "$OUT_DIR/knowledge_annotation_measured.sio.log" "$OUT_DIR/measured.log"
  cp "$OUT_DIR/knowledge_annotation_unknown_coerce.sio.log" "$OUT_DIR/unknown_coerce.log"
  cp "$OUT_DIR/knowledge_annotation_measured_coerce.sio.log" "$OUT_DIR/measured_coerce.log"
  unknown_rc=$(cat "$OUT_DIR/knowledge_annotation_unknown_component.sio.rc")
  measured_rc=$(cat "$OUT_DIR/knowledge_annotation_measured.sio.rc")
  unknown_coerce_rc=$(cat "$OUT_DIR/knowledge_annotation_unknown_coerce.sio.rc")
  measured_coerce_rc=$(cat "$OUT_DIR/knowledge_annotation_measured_coerce.sio.rc")
else
  set +e
  run_check "$UNKNOWN" "$OUT_DIR/unknown.log"
  unknown_rc=$?
  run_check "$MEASURED" "$OUT_DIR/measured.log"
  measured_rc=$?
  run_check "$UNKNOWN_COERCE" "$OUT_DIR/unknown_coerce.log"
  unknown_coerce_rc=$?
  run_check "$MEASURED_COERCE" "$OUT_DIR/measured_coerce.log"
  measured_coerce_rc=$?
  set -e
fi

if [[ "$measured_rc" -ne 0 ]]; then
  echo "NEGATIVE CONTROL FAILED: Knowledge[f64, Measured] did not typecheck." >&2
  echo "The defect is not the silent else-arm described in the dispatch. Stopping." >&2
  cat "$OUT_DIR/measured.log" >&2
  failed=1
  python3 - "$OUT_DIR/status.json" "$total" "$passed" "$failed" "$not_run" "$measured_rc" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
total, passed, failed, not_run, measured_rc = map(int, sys.argv[2:])
path.write_text(json.dumps({
    "schema": "sounio.knowledge_annotation_silent_drop.v1",
    "status": "fail",
    "reason": "negative_control_measured_did_not_compile",
    "measured_rc": measured_rc,
    "metrics": {"total": total, "passed": passed, "failed": failed, "not_run": not_run},
}, indent=2) + "\n")
PY
  echo "status=fail"
  echo "metrics {total=$total, passed=$passed, failed=$failed, not_run=$not_run}"
  exit 2
fi
measured_ok=1
passed=$((passed + 1))

if [[ "$unknown_rc" -eq 0 ]]; then
  silence="present"
  failed=$((failed + 1))
else
  silence="absent"
  passed=$((passed + 1))
fi

if [[ "$unknown_coerce_rc" -eq 0 ]]; then
  passed=$((passed + 1))
else
  failed=$((failed + 1))
fi
if [[ "$measured_coerce_rc" -eq 0 ]]; then
  passed=$((passed + 1))
else
  failed=$((failed + 1))
fi

if [[ "$silence" == "present" ]]; then
  status="fail"
else
  status="pass"
fi

python3 - "$OUT_DIR/status.json" "$status" "$silence" "$total" "$passed" "$failed" "$not_run" \
  "$unknown_rc" "$measured_rc" "$unknown_coerce_rc" "$measured_coerce_rc" "$use_slurm" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
status, silence = sys.argv[2], sys.argv[3]
total, passed, failed, not_run = map(int, sys.argv[4:8])
unknown_rc, measured_rc, unknown_coerce_rc, measured_coerce_rc, use_slurm = map(int, sys.argv[8:13])
path.write_text(json.dumps({
    "schema": "sounio.knowledge_annotation_silent_drop.v1",
    "status": status,
    "silence": silence,
    "harness": "slurm" if use_slurm else "local",
    "unknown_rc": unknown_rc,
    "measured_rc": measured_rc,
    "unknown_coerce_rc": unknown_coerce_rc,
    "measured_coerce_rc": measured_coerce_rc,
    "metrics": {"total": total, "passed": passed, "failed": failed, "not_run": not_run},
    "note": (
        "Accusation: unknown component typechecks with no diagnostic. "
        "TypeEntry also drops recognised Measured provenance (both coerce to bare Knowledge)."
        if silence == "present" else
        "Unknown component is now refused. Silence closed."
    ),
}, indent=2) + "\n")
PY

echo "status=$status"
echo "silence=$silence"
echo "harness=$([[ "$use_slurm" -eq 1 ]] && echo slurm || echo local)"
echo "unknown_rc=$unknown_rc measured_rc=$measured_rc unknown_coerce_rc=$unknown_coerce_rc measured_coerce_rc=$measured_coerce_rc"
echo "metrics {total=$total, passed=$passed, failed=$failed, not_run=$not_run}"

if [[ "$silence" == "present" ]]; then
  echo >&2
  echo "ACCUSATION: unknown Knowledge annotation component is silently dropped." >&2
  echo "  $UNKNOWN typechecked (rc=0). Parser else-arm advances with no diagnostic." >&2
  echo "  This gate fails while that silence exists. It is not a regression gate." >&2
  exit 1
fi
exit 0
