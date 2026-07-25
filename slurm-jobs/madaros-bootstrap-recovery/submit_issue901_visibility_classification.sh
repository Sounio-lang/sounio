#!/usr/bin/env bash
# Classify the #901 visibility frontier without changing the bootstrap gate.
#
# This companion job consumes a provenance-checked compiler artifact retained
# by the recovery gate and collects strict-checker/frontend probes side by side.
# It does not classify language semantics: logs remain evidence for a later
# decision. A bridge artifact is diagnostic only, never an operational Madaros
# generation.

set -euo pipefail

REPO="${REPO:-$(pwd)}"
SOURCE_REF="${SOURCE_REF:-HEAD}"
RECOVERY_RESULT_DIR="${SOUNIO_MADAROS_RECOVERY_RESULT_DIR:-}"
COMPILER_KIND="${SOUNIO_MADAROS_RECOVERY_COMPILER_KIND:-bridge}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"
ACCOUNT="${ACCOUNT:-omics}"
QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-12G}"
JOB_CPUS="${JOB_CPUS:-1}"
JOB_TIME="${JOB_TIME:-00:15:00}"
ORANGEFS_TMP="${ORANGEFS_TMP:-/orangefs/training/tmp}"
RUN_ID="${RUN_ID:-madaros-issue901-visibility-$(date -u +%Y%m%dT%H%M%S)}"

fail() {
  echo "[madaros-issue901-visibility-submit] FAIL: $*" >&2
  exit 1
}

safe_component() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]]
}

safe_orangefs_path() {
  [[ "$1" =~ ^/orangefs/[A-Za-z0-9._/-]+$ ]]
}

case "$COMPILER_KIND" in
  bridge|stage2) ;;
  *) fail 'SOUNIO_MADAROS_RECOVERY_COMPILER_KIND must be bridge or stage2' ;;
esac

[[ -n "$RECOVERY_RESULT_DIR" ]] || fail 'SOUNIO_MADAROS_RECOVERY_RESULT_DIR is required'
safe_orangefs_path "$RECOVERY_RESULT_DIR" || fail 'SOUNIO_MADAROS_RECOVERY_RESULT_DIR must be an absolute OrangeFS path without whitespace'
safe_orangefs_path "$ORANGEFS_TMP" || fail 'ORANGEFS_TMP must be an absolute OrangeFS path without whitespace'
safe_component "$RUN_ID" || fail 'RUN_ID may contain only letters, digits, dot, underscore, and dash'

SOURCE_COMMIT="$(git -C "$REPO" rev-parse "$SOURCE_REF")"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || fail 'source checkout is dirty; commit the probe launcher first'

LOGIN_POD="$($KUBECTL -n "$NS" get pods -l app.kubernetes.io/name=login --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
[[ -n "$LOGIN_POD" ]] || fail 'no running Slurm login pod'

TARBALL="/tmp/$RUN_ID.tgz"
SBATCH="/tmp/$RUN_ID.sbatch"
SNAP="/tmp/$RUN_ID-snapshot"
RESULT_DIR="$ORANGEFS_TMP/$RUN_ID.result"
REMOTE_ARCHIVE="$ORANGEFS_TMP/$RUN_ID.tgz"
REMOTE_SBATCH="$ORANGEFS_TMP/$RUN_ID.sbatch"

rm -rf "$SNAP"
mkdir -p "$SNAP"
git -C "$REPO" archive "$SOURCE_COMMIT" -- \
  self-hosted \
  stdlib \
  tests/multimodule \
  | tar -x -C "$SNAP"
tar -C "$SNAP" -czf "$TARBALL" .
rm -rf "$SNAP"

$KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "mkdir -p '$ORANGEFS_TMP'"
$KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "mkdir '$RESULT_DIR'" || fail "result directory already exists or cannot be created: $RESULT_DIR"
$KUBECTL -n "$NS" cp "$TARBALL" "$LOGIN_POD:/tmp/$RUN_ID.tgz"
$KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "mv '/tmp/$RUN_ID.tgz' '$REMOTE_ARCHIVE'"

cat > "$SBATCH" <<EOF
#!/usr/bin/env bash
#SBATCH -J $RUN_ID
#SBATCH -p $PARTITION
#SBATCH -A $ACCOUNT
#SBATCH --qos=$QOS
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c $JOB_CPUS
#SBATCH --mem=$JOB_MEM
#SBATCH --time=$JOB_TIME
#SBATCH -o $ORANGEFS_TMP/$RUN_ID.slurmout
#SBATCH -e $ORANGEFS_TMP/$RUN_ID.slurmout
set -u -o pipefail

ROOT=/tmp/$RUN_ID-\${SLURM_JOB_ID:-manual}
REPO=\$ROOT/repo
RESULT_DIR=$RESULT_DIR
RECOVERY_RESULT_DIR=$RECOVERY_RESULT_DIR
COMPILER_KIND=$COMPILER_KIND
COMPILER=\$RECOVERY_RESULT_DIR/gate-work/madaros-\$COMPILER_KIND
if [[ "\$COMPILER_KIND" == bridge ]]; then
  PROVENANCE=\$RECOVERY_RESULT_DIR/gate-work/bridge-provenance.tsv
  EXPECTED_SHA_KEY=bridge_madaros_sha256
  ARTIFACT_ROLE=diagnostic_bridge
else
  PROVENANCE=\$RECOVERY_RESULT_DIR/gate-work/bootstrap-recovery-receipt.tsv
  EXPECTED_SHA_KEY=stage2_madaros_sha256
  ARTIFACT_ROLE=operational_fixed_point
fi

rm -rf "\$ROOT"
mkdir -p "\$REPO"
tar -xzf $REMOTE_ARCHIVE -C "\$REPO"
cd "\$REPO"

sha256() {
  sha256sum "\$1" 2>/dev/null | awk '{print \$1}' || shasum -a 256 "\$1" | awk '{print \$1}'
}

tsv_value() {
  awk -F '\\t' -v key="\$1" '\$1 == key { print \$2; exit }' "\$2"
}

block() {
  printf 'classification\\tBLOCKED\\n' > "\$RESULT_DIR/status.tsv"
  printf 'reason\\t%s\\n' "\$1" >> "\$RESULT_DIR/status.tsv"
  echo 'status=BLOCKED' >> "\$RESULT_DIR/environment.tsv"
  exit 0
}

run_probe() {
  local label="\$1"
  local mode="\$2"
  local source="\$3"
  local rc=0
  set +e
  SOUNIO_STDLIB_PATH="\$REPO/stdlib" "\$COMPILER" "\$mode" "\$source" >"\$RESULT_DIR/\$label.log" 2>&1
  rc=\$?
  set -e
  printf '%s_rc\\t%s\\n' "\$label" "\$rc" >> "\$RESULT_DIR/status.tsv"
}

{
  echo 'receipt_version=madaros-issue901-visibility-classification-v1'
  echo 'source_commit=$SOURCE_COMMIT'
  echo "job_id=\${SLURM_JOB_ID:-manual}"
  echo "compiler_kind=\$COMPILER_KIND"
  echo "artifact_role=\$ARTIFACT_ROLE"
  echo "compiler_path=\$COMPILER"
  echo "provenance_receipt=\$PROVENANCE"
} > "\$RESULT_DIR/environment.tsv"

if [[ ! -x "\$COMPILER" || "\$(head -c4 "\$COMPILER" 2>/dev/null)" != \$'\\x7fELF' ]]; then
  block "missing_\${COMPILER_KIND}_artifact"
fi

[[ -r "\$PROVENANCE" ]] || block 'missing_provenance_receipt'
[[ "\$(tsv_value source_commit "\$PROVENANCE")" == "$SOURCE_COMMIT" ]] || block 'provenance_source_commit_mismatch'
[[ "\$(tsv_value artifact_role "\$PROVENANCE")" == "\$ARTIFACT_ROLE" ]] || block 'provenance_artifact_role_mismatch'
if [[ "\$COMPILER_KIND" == bridge ]]; then
  [[ "\$(tsv_value closure_status "\$PROVENANCE")" == complete ]] || block 'bridge_closure_incomplete'
  [[ "\$(tsv_value closure_parse_failed "\$PROVENANCE")" == false ]] || block 'bridge_closure_parse_failed'
fi
if [[ "\$COMPILER_KIND" == stage2 ]]; then
  [[ "\$(tsv_value operational_fixed_point "\$PROVENANCE")" == sha256-stage1-equals-stage2 ]] || block 'missing_operational_fixed_point'
fi
COMPILER_SHA256="\$(sha256 "\$COMPILER")"
[[ "\$(tsv_value "\$EXPECTED_SHA_KEY" "\$PROVENANCE")" == "\$COMPILER_SHA256" ]] || block 'provenance_compiler_sha256_mismatch'
printf 'compiler_sha256\\t%s\\n' "\$COMPILER_SHA256" >> "\$RESULT_DIR/environment.tsv"
printf 'classification\\tCOLLECTED\\n' > "\$RESULT_DIR/status.tsv"

run_probe strict_main --check "\$REPO/self-hosted/compiler/main.sio"
run_probe frontend_main --probe-frontend "\$REPO/self-hosted/compiler/main.sio"
run_probe frontend_private_fn --probe-frontend "\$REPO/tests/multimodule/visibility_fn_private_main.sio"
run_probe frontend_private_struct --probe-frontend "\$REPO/tests/multimodule/visibility_struct_private_main.sio"
run_probe frontend_private_enum --probe-frontend "\$REPO/tests/multimodule/visibility_enum_private_main.sio"

echo 'status=COLLECTED' >> "\$RESULT_DIR/environment.tsv"
EOF

$KUBECTL -n "$NS" cp "$SBATCH" "$LOGIN_POD:$REMOTE_SBATCH"
JOB_SUBMISSION="$($KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "sbatch '$REMOTE_SBATCH'")"
rm -f "$TARBALL" "$SBATCH"

echo "$JOB_SUBMISSION"
echo "SOURCE_COMMIT=$SOURCE_COMMIT"
echo "RECOVERY_RESULT_DIR=$RECOVERY_RESULT_DIR"
echo "COMPILER_KIND=$COMPILER_KIND"
echo "RESULT_DIR=$RESULT_DIR"
