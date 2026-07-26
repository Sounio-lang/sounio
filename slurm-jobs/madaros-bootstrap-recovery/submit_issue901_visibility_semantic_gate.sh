#!/usr/bin/env bash
# Submit the #901 exact-import semantic gate against a recovered Madaros stage2.
#
# The recovery run proves an operational fixed point. This companion job proves
# a narrow language contract only after binding the candidate stage2 ELF to the
# same committed compiler snapshot recorded by that recovery receipt.

set -euo pipefail

REPO="${REPO:-$(pwd)}"
SOURCE_REF="${SOURCE_REF:-}"
RECOVERY_RESULT_DIR="${SOUNIO_MADAROS_RECOVERY_RESULT_DIR:-}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"
ACCOUNT="${ACCOUNT:-omics}"
QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-24G}"
JOB_CPUS="${JOB_CPUS:-2}"
JOB_TIME="${JOB_TIME:-00:20:00}"
ORANGEFS_TMP="${ORANGEFS_TMP:-/orangefs/training/tmp}"
RUN_ID="${RUN_ID:-madaros-issue901-visibility-semantic-$(date -u +%Y%m%dT%H%M%S)}"

fail() {
  echo "[madaros-issue901-visibility-semantic-submit] FAIL: $*" >&2
  exit 1
}

safe_component() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]]
}

safe_orangefs_path() {
  [[ "$1" =~ ^/orangefs/[A-Za-z0-9._/-]+$ ]] && [[ "/$1/" != *'/../'* ]]
}

[[ -n "$SOURCE_REF" ]] || fail 'SOURCE_REF must name the exact compiler snapshot recorded by the recovery receipt'
[[ -n "$RECOVERY_RESULT_DIR" ]] || fail 'SOUNIO_MADAROS_RECOVERY_RESULT_DIR is required'
safe_orangefs_path "$RECOVERY_RESULT_DIR" || fail 'SOUNIO_MADAROS_RECOVERY_RESULT_DIR must be an absolute OrangeFS path without whitespace'
safe_orangefs_path "$ORANGEFS_TMP" || fail 'ORANGEFS_TMP must be an absolute OrangeFS path without whitespace'
safe_component "$RUN_ID" || fail 'RUN_ID may contain only letters, digits, dot, underscore, and dash'

SOURCE_COMMIT="$(git -C "$REPO" rev-parse "$SOURCE_REF")"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || fail 'source checkout is dirty; commit the semantic-gate launcher first'

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
  bin/madaros \
  stdlib \
  scripts/ci/madaros_visibility_context_gate.sh \
  tests/compiler/madaros_visibility_context \
  tests/multimodule/visibility_fn_private_main.sio \
  tests/multimodule/visibility_fn_private_lib.sio \
  tests/multimodule/visibility_struct_private_main.sio \
  tests/multimodule/visibility_struct_private_lib.sio \
  tests/multimodule/visibility_enum_private_main.sio \
  tests/multimodule/visibility_enum_private_lib.sio \
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
set -uo pipefail

ROOT=/tmp/$RUN_ID-\${SLURM_JOB_ID:-manual}
REPO=\$ROOT/repo
RESULT_DIR=$RESULT_DIR
RECOVERY_RESULT_DIR=$RECOVERY_RESULT_DIR
COMPILER=\$RECOVERY_RESULT_DIR/gate-work/madaros-stage2
PROVENANCE=\$RECOVERY_RESULT_DIR/gate-work/bootstrap-recovery-receipt.tsv

rm -rf "\$ROOT"
mkdir -p "\$REPO"
tar -xzf $REMOTE_ARCHIVE -C "\$REPO"
chmod +x "\$REPO/bin/madaros" "\$REPO/scripts/ci/madaros_visibility_context_gate.sh"
cd "\$REPO"

sha256() {
  sha256sum "\$1" 2>/dev/null | awk '{print \$1}' || shasum -a 256 "\$1" | awk '{print \$1}'
}

tsv_value() {
  awk -F '\\t' -v key="\$1" '\$1 == key { print \$2; exit }' "\$2"
}

tsv_count() {
  awk -F '\\t' -v key="\$1" '\$1 == key { count += 1 } END { print count + 0 }' "\$2"
}

block() {
  printf 'classification\\tBLOCKED\\n' > "\$RESULT_DIR/status.tsv"
  printf 'reason\\t%s\\n' "\$1" >> "\$RESULT_DIR/status.tsv"
  echo 'status=BLOCKED' >> "\$RESULT_DIR/environment.tsv"
  exit 0
}

{
  echo 'receipt_version=madaros-issue901-visibility-semantic-v1'
  echo 'source_commit=$SOURCE_COMMIT'
  echo "job_id=\${SLURM_JOB_ID:-manual}"
  echo 'artifact_role=operational_fixed_point'
  echo "compiler_path=\$COMPILER"
  echo "provenance_receipt=\$PROVENANCE"
} > "\$RESULT_DIR/environment.tsv"

if [[ ! -x "\$COMPILER" || "\$(head -c4 "\$COMPILER" 2>/dev/null)" != \$'\\x7fELF' ]]; then
  block 'missing_stage2_artifact'
fi
[[ -r "\$PROVENANCE" ]] || block 'missing_provenance_receipt'
for key in receipt_version source_commit artifact_role operational_fixed_point stage1_madaros_sha256 stage2_madaros_sha256; do
  [[ "\$(tsv_count "\$key" "\$PROVENANCE")" == 1 ]] || block "invalid_receipt_key_\$key"
done
[[ "\$(tsv_value receipt_version "\$PROVENANCE")" == madaros-bootstrap-recovery-901-v1 ]] || block 'provenance_receipt_version_mismatch'
[[ "\$(tsv_value source_commit "\$PROVENANCE")" == "$SOURCE_COMMIT" ]] || block 'provenance_source_commit_mismatch'
[[ "\$(tsv_value artifact_role "\$PROVENANCE")" == operational_fixed_point ]] || block 'provenance_artifact_role_mismatch'
[[ "\$(tsv_value operational_fixed_point "\$PROVENANCE")" == sha256-stage1-equals-stage2 ]] || block 'missing_operational_fixed_point'
[[ "\$(tsv_value stage1_madaros_sha256 "\$PROVENANCE")" == "\$(tsv_value stage2_madaros_sha256 "\$PROVENANCE")" ]] || block 'stage1_stage2_hash_mismatch'
COMPILER_SHA256="\$(sha256 "\$COMPILER")"
[[ "\$(tsv_value stage2_madaros_sha256 "\$PROVENANCE")" == "\$COMPILER_SHA256" ]] || block 'provenance_compiler_sha256_mismatch'
printf 'compiler_sha256\\t%s\\n' "\$COMPILER_SHA256" >> "\$RESULT_DIR/environment.tsv"

set +e
SOUNIO_STDLIB_PATH="\$REPO/stdlib" \\
MADAROS_RAW_BIN="\$COMPILER" \\
SOUNIO_MADAROS_VISIBILITY_CONTEXT_BIN="\$REPO/bin/madaros" \\
SOUNIO_MADAROS_VISIBILITY_CONTEXT_EXPECT=resolved \\
SOUNIO_MADAROS_VISIBILITY_CONTEXT_DIR="\$RESULT_DIR/gate-work" \\
SOUNIO_MADAROS_VISIBILITY_CONTEXT_KEEP=1 \\
  bash "\$REPO/scripts/ci/madaros_visibility_context_gate.sh" > "\$RESULT_DIR/gate.log" 2>&1
RC=\$?
set -e
printf 'exit_code\\t%s\\n' "\$RC" > "\$RESULT_DIR/status.tsv"
if [[ "\$RC" -eq 0 ]]; then
  printf 'classification\\tGREEN\\n' >> "\$RESULT_DIR/status.tsv"
  printf 'semantic_visibility\\tresolved\\n' >> "\$RESULT_DIR/status.tsv"
  echo 'status=PASS' >> "\$RESULT_DIR/environment.tsv"
elif [[ "\$RC" -eq 124 ]]; then
  printf 'classification\\tTIMEOUT\\n' >> "\$RESULT_DIR/status.tsv"
  echo 'status=TIMEOUT' >> "\$RESULT_DIR/environment.tsv"
else
  printf 'classification\\tFIXABLE\\n' >> "\$RESULT_DIR/status.tsv"
  printf 'reason\\tvisibility_context_gate_failed\\n' >> "\$RESULT_DIR/status.tsv"
  echo 'status=FAIL' >> "\$RESULT_DIR/environment.tsv"
fi
exit "\$RC"
EOF

$KUBECTL -n "$NS" cp "$SBATCH" "$LOGIN_POD:$REMOTE_SBATCH"
JOB_SUBMISSION="$($KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "sbatch '$REMOTE_SBATCH'")"
rm -f "$TARBALL" "$SBATCH"

echo "$JOB_SUBMISSION"
echo "SOURCE_COMMIT=$SOURCE_COMMIT"
echo "RECOVERY_RESULT_DIR=$RECOVERY_RESULT_DIR"
echo "RESULT_DIR=$RESULT_DIR"
