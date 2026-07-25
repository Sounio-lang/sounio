#!/usr/bin/env bash
# Submit the quarantined #901 Madaros bootstrap-recovery gate to Slurm.
# The job archives a committed source snapshot and keeps all outputs on OrangeFS.

set -euo pipefail

REPO="${REPO:-$(pwd)}"
SOURCE_REF="${SOURCE_REF:-HEAD}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"
ACCOUNT="${ACCOUNT:-omics}"
QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-24G}"
JOB_CPUS="${JOB_CPUS:-2}"
JOB_TIME="${JOB_TIME:-00:45:00}"
ORANGEFS_TMP="${ORANGEFS_TMP:-/orangefs/training/tmp}"
RUN_ID="${RUN_ID:-madaros-issue901-recovery-$(date -u +%Y%m%dT%H%M%S)}"
CHECK_PREFLIGHT_TRACE="${SOUNIO_CHECK_PREFLIGHT_TRACE:-0}"

fail() {
  echo "[madaros-issue901-recovery-submit] FAIL: $*" >&2
  exit 1
}

SOURCE_COMMIT="$(git -C "$REPO" rev-parse "$SOURCE_REF")"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || fail 'source checkout is dirty; commit the recovery candidate first'
[[ -x "$REPO/bin/souc-linux-x86_64" ]] || fail "missing legacy recovery bootstrap: $REPO/bin/souc-linux-x86_64"
[[ "$CHECK_PREFLIGHT_TRACE" == '0' || "$CHECK_PREFLIGHT_TRACE" == '1' ]] || fail 'SOUNIO_CHECK_PREFLIGHT_TRACE must be 0 or 1'

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
  bin/souc-linux-x86_64 \
  self-hosted \
  stdlib \
  scripts/ci/build_modular_madaros.sh \
  scripts/ci/madaros_imported_runtime_acceptance_gate.sh \
  scripts/ci/madaros_struct_layout_capacity_gate.sh \
  scripts/ci/madaros_scope_contextual_binding_gate.sh \
  scripts/ci/madaros_bootstrap_recovery_901_gate.sh \
  scripts/dev/souc-build-lock.sh \
  scripts/dev/souc_build_lock.py \
  scripts/research/generate_madaros_struct_layout_capacity_fixture.py \
  tests/compiler/madaros_imported_runtime_acceptance \
  tests/compiler/madaros_struct_layout_capacity \
  tests/run-pass/let_scope_binding_name.sio \
  tests/run-pass/let_policy_binding_name.sio \
  tests/run-pass/let_is_binding_name.sio \
  tests/run-pass/let_study_binding_name.sio \
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
rm -rf "\$ROOT"
mkdir -p "\$REPO"
tar -xzf $REMOTE_ARCHIVE -C "\$REPO"
chmod +x "\$REPO/bin/souc-linux-x86_64" "\$REPO/scripts/ci/"*.sh "\$REPO/scripts/dev/souc-build-lock.sh"
cd "\$REPO"

{
  echo "source_commit=$SOURCE_COMMIT"
  echo "job_id=\${SLURM_JOB_ID:-manual}"
  echo "host=\$(hostname)"
  echo "partition=$PARTITION"
  echo "cpus=$JOB_CPUS mem=$JOB_MEM"
  echo "check_preflight_trace=$CHECK_PREFLIGHT_TRACE"
} > "\$RESULT_DIR/environment.tsv"

SOUNIO_MADAROS_RECOVERY_KEEP=1 \
SOUNIO_MADAROS_RECOVERY_DIR="\$RESULT_DIR/gate-work" \
SOUNIO_MADAROS_RECOVERY_SOURCE_COMMIT="$SOURCE_COMMIT" \
SOUNIO_CHECK_PREFLIGHT_TRACE="$CHECK_PREFLIGHT_TRACE" \
  bash scripts/ci/madaros_bootstrap_recovery_901_gate.sh >"\$RESULT_DIR/gate.log" 2>&1
RC=\$?
printf 'exit_code\\t%s\\n' "\$RC" > "\$RESULT_DIR/status.tsv"
if [ "\$RC" -eq 0 ]; then
  echo 'status=PASS' >> "\$RESULT_DIR/status.tsv"
else
  echo 'status=FAIL' >> "\$RESULT_DIR/status.tsv"
fi
exit "\$RC"
EOF

$KUBECTL -n "$NS" cp "$SBATCH" "$LOGIN_POD:$REMOTE_SBATCH"
JOB_SUBMISSION="$($KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "sbatch '$REMOTE_SBATCH'")"
rm -f "$TARBALL" "$SBATCH"

echo "$JOB_SUBMISSION"
echo "SOURCE_COMMIT=$SOURCE_COMMIT"
echo "RESULT_DIR=$RESULT_DIR"
