#!/usr/bin/env bash
# Submit the #901 source-fresh Madaros self-seed proof to Slurm.
#
# The worker receives an exact `git archive` snapshot plus a provenance TSV.
# The gate validates that TSV against the unpacked sources; it never fabricates
# a Git checkout or treats a synthetic commit as current-source evidence.

set -euo pipefail
export LC_ALL=C

REPO="${REPO:-$(pwd)}"
SOURCE_REF="${SOURCE_REF:-HEAD}"
NS="${NS:-slurm-pilot}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
PARTITION="${PARTITION:-cpu-ops}"
ACCOUNT="${ACCOUNT:-omics}"
QOS="${QOS:-cpuops}"
JOB_MEM="${JOB_MEM:-24G}"
JOB_CPUS="${JOB_CPUS:-2}"
JOB_TIME="${JOB_TIME:-00:45:00}"
ORANGEFS_TMP="${ORANGEFS_TMP:-/orangefs/training/tmp}"
RUN_ID="${RUN_ID:-madaros-issue901-scale-source-fresh-$(date -u +%Y%m%dT%H%M%S)}"
PREPARE_ONLY="${SOUNIO_MADAROS_ISSUE901_SCALE_SOURCE_FRESH_PREPARE_ONLY:-0}"

fail() {
  echo "[madaros-issue901-scale-source-fresh-submit] FAIL: $*" >&2
  exit 1
}

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

portable_mode() {
  stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1"
}

require_path() {
  [[ -e "$REPO/$1" ]] || fail "required source path is missing: $1"
}

[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || fail "REPO is not a Git worktree: $REPO"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || fail 'source checkout is dirty; commit the candidate before source-fresh Slurm evidence'
SOURCE_COMMIT="$(git -C "$REPO" rev-parse "${SOURCE_REF}^{commit}")"
SOURCE_TREE="$(git -C "$REPO" rev-parse "${SOURCE_COMMIT}^{tree}")"

for path in \
  bin/souc \
  bin/madaros \
  bin/madaros-linux-x86_64 \
  self-hosted \
  stdlib \
  tools/science_boundary \
  scripts/ci/madaros_native_multimodule_scale_source_fresh_gate.sh \
  scripts/ci/fixtures/madaros_m0_source_compat.patch \
  tests/run-pass/madaros_native_multimodule_scale_prob.sio \
  tests/run-pass/madaros_native_multimodule_scale_prob_textbook.sio \
  tests/run-pass/let_scope_binding_name.sio \
  tests/run-pass/let_policy_binding_name.sio \
  tests/run-pass/let_is_binding_name.sio \
  tests/run-pass/let_study_binding_name.sio \
  tests/stdlib/prob/test_prob_stdlib.sio; do
  require_path "$path"
done

[[ "$PREPARE_ONLY" == '0' || "$PREPARE_ONLY" == '1' ]] || fail 'SOUNIO_MADAROS_ISSUE901_SCALE_SOURCE_FRESH_PREPARE_ONLY must be 0 or 1'

TMP_ROOT="$(mktemp -d /tmp/madaros-issue901-scale-source-fresh-submit.XXXXXX)"
SNAP="$TMP_ROOT/snapshot"
TARBALL="$TMP_ROOT/$RUN_ID.tgz"
SBATCH="$TMP_ROOT/$RUN_ID.sbatch"
RESULT_DIR="$ORANGEFS_TMP/$RUN_ID.result"
REMOTE_ARCHIVE="$ORANGEFS_TMP/$RUN_ID.tgz"
REMOTE_SBATCH="$ORANGEFS_TMP/$RUN_ID.sbatch"

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

mkdir -p "$SNAP"
git -C "$REPO" archive "$SOURCE_COMMIT" -- \
  bin/souc \
  bin/madaros \
  bin/madaros-linux-x86_64 \
  self-hosted \
  stdlib \
  tools/science_boundary \
  scripts/ci/madaros_native_multimodule_scale_source_fresh_gate.sh \
  scripts/ci/fixtures/madaros_m0_source_compat.patch \
  tests/run-pass/madaros_native_multimodule_scale_prob.sio \
  tests/run-pass/madaros_native_multimodule_scale_prob_textbook.sio \
  tests/run-pass/let_scope_binding_name.sio \
  tests/run-pass/let_policy_binding_name.sio \
  tests/run-pass/let_is_binding_name.sio \
  tests/run-pass/let_study_binding_name.sio \
  tests/stdlib/prob/test_prob_stdlib.sio | tar -x -C "$SNAP"

MADAROS_ROOT_BLOB="$(git -C "$REPO" ls-tree "$SOURCE_COMMIT" -- bin/madaros-linux-x86_64 | awk 'NR == 1 {print $3}')"
[[ "$MADAROS_ROOT_BLOB" =~ ^[0-9a-f]{40}$ ]] || fail 'Madaros root is not tracked by the source commit'

MANIFEST="$SNAP/.issue901-scale-source-manifest.tsv"
MANIFEST_PATHS="$TMP_ROOT/proof-input-paths.txt"
find "$SNAP" -type f -print | LC_ALL=C sort >"$MANIFEST_PATHS"
: >"$MANIFEST"
while IFS= read -r absolute; do
  relative="${absolute#"$SNAP"/}"
  printf '%s\t%s\t%s\n' "$(portable_mode "$absolute")" "$relative" "$(portable_sha256 "$absolute")" >>"$MANIFEST"
done <"$MANIFEST_PATHS"
PROOF_INPUT_MANIFEST_SHA256="$(portable_sha256 "$MANIFEST")"

{
  printf 'provenance_version\tissue901-scale-source-fresh-archive-v3\n'
  printf 'source_origin\tgit-archive-exact-commit\n'
  printf 'source_head\t%s\n' "$SOURCE_COMMIT"
  printf 'source_tree\t%s\n' "$SOURCE_TREE"
  printf 'main_sio_sha256\t%s\n' "$(portable_sha256 "$SNAP/self-hosted/compiler/main.sio")"
  printf 'gate_script_sha256\t%s\n' "$(portable_sha256 "$SNAP/scripts/ci/madaros_native_multimodule_scale_source_fresh_gate.sh")"
  printf 'madaros_root_git_blob\t%s\n' "$MADAROS_ROOT_BLOB"
  printf 'madaros_root_sha256\t%s\n' "$(portable_sha256 "$SNAP/bin/madaros-linux-x86_64")"
  printf 'souc_wrapper_sha256\t%s\n' "$(portable_sha256 "$SNAP/bin/souc")"
  printf 'madaros_wrapper_sha256\t%s\n' "$(portable_sha256 "$SNAP/bin/madaros")"
  printf 'acceptance_probe_sha256\t%s\n' "$(portable_sha256 "$SNAP/tests/run-pass/madaros_native_multimodule_scale_prob.sio")"
  printf 'textbook_probe_sha256\t%s\n' "$(portable_sha256 "$SNAP/tests/run-pass/madaros_native_multimodule_scale_prob_textbook.sio")"
  printf 'stdlib_driver_sha256\t%s\n' "$(portable_sha256 "$SNAP/tests/stdlib/prob/test_prob_stdlib.sio")"
  printf 'm0_compat_patch_sha256\t%s\n' "$(portable_sha256 "$SNAP/scripts/ci/fixtures/madaros_m0_source_compat.patch")"
  printf 'proof_input_manifest_sha256\t%s\n' "$PROOF_INPUT_MANIFEST_SHA256"
} >"$SNAP/.issue901-scale-source-provenance.tsv"

bash "$SNAP/scripts/ci/madaros_native_multimodule_scale_source_fresh_gate.sh" --provenance-only

tar -C "$SNAP" -czf "$TARBALL" .
ARCHIVE_SHA256="$(portable_sha256 "$TARBALL")"

if [[ "$PREPARE_ONLY" == '1' ]]; then
  printf 'PREPARED_SOURCE_COMMIT=%s\n' "$SOURCE_COMMIT"
  printf 'PREPARED_SOURCE_TREE=%s\n' "$SOURCE_TREE"
  printf 'PREPARED_ARCHIVE_SHA256=%s\n' "$ARCHIVE_SHA256"
  exit 0
fi

LOGIN_POD="$("$KUBECTL_BIN" -n "$NS" get pods -l app.kubernetes.io/name=login --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
[[ -n "$LOGIN_POD" ]] || fail 'no running Slurm login pod'

"$KUBECTL_BIN" -n "$NS" exec "$LOGIN_POD" -- bash -lc "mkdir -p '$ORANGEFS_TMP'"
"$KUBECTL_BIN" -n "$NS" exec "$LOGIN_POD" -- bash -lc "mkdir '$RESULT_DIR'" || fail "result directory already exists or cannot be created: $RESULT_DIR"
"$KUBECTL_BIN" -n "$NS" cp "$TARBALL" "$LOGIN_POD:/tmp/$RUN_ID.tgz"
"$KUBECTL_BIN" -n "$NS" exec "$LOGIN_POD" -- bash -lc "mv '/tmp/$RUN_ID.tgz' '$REMOTE_ARCHIVE'"

cat >"$SBATCH" <<EOF
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
ARCHIVE=$REMOTE_ARCHIVE
ARCHIVE_SHA256=$ARCHIVE_SHA256
write_status() {
  local code="\$1"
  local status="\$2"
  local class="\$3"
  local reason="\${4:-}"
  {
    printf 'exit_code\\t%s\\n' "\$code"
    printf 'status\\t%s\\n' "\$status"
    printf 'class\\t%s\\n' "\$class"
    [[ -z "\$reason" ]] || printf 'reason\\t%s\\n' "\$reason"
  } >"\$RESULT_DIR/status.tsv" || true
}
setup_fail() {
  write_status 125 FAIL harness-routing "\$1"
  exit 125
}

rm -rf "\$ROOT" || setup_fail worker-root-cleanup-failed
mkdir -p "\$REPO" || setup_fail worker-root-create-failed

actual_sha=""
actual_sha=\$(sha256sum "\$ARCHIVE" | awk '{print \$1}') || setup_fail archive-sha-read-failed
if [[ "\$actual_sha" != "\$ARCHIVE_SHA256" ]]; then
  setup_fail archive-sha-mismatch
fi

tar -xzf "\$ARCHIVE" -C "\$REPO" || setup_fail archive-extract-failed
chmod +x "\$REPO/bin/souc" "\$REPO/bin/madaros" "\$REPO/bin/madaros-linux-x86_64" \\
  "\$REPO/scripts/ci/madaros_native_multimodule_scale_source_fresh_gate.sh" || setup_fail archive-permission-setup-failed

{
  printf 'source_head\\t%s\\n' '$SOURCE_COMMIT'
  printf 'source_tree\\t%s\\n' '$SOURCE_TREE'
  printf 'archive_sha256\\t%s\\n' "\$ARCHIVE_SHA256"
  printf 'job_id\\t%s\\n' "\${SLURM_JOB_ID:-manual}"
  printf 'host\\t%s\\n' "\$(hostname)"
  printf 'partition\\t%s\\n' '$PARTITION'
  printf 'resources\\tcpus=$JOB_CPUS mem=$JOB_MEM time=$JOB_TIME\\n'
} >"\$RESULT_DIR/environment.tsv" || setup_fail environment-receipt-write-failed

set +e
SOUNIO_MADAROS_ISSUE901_SCALE_SOURCE_FRESH_KEEP=1 \\
SOUNIO_MADAROS_ISSUE901_SCALE_SOURCE_FRESH_DIR="\$ROOT/gate-work" \\
  bash "\$REPO/scripts/ci/madaros_native_multimodule_scale_source_fresh_gate.sh" >"\$ROOT/gate.log" 2>&1
RC=\$?
set -e

cp "\$ROOT/gate.log" "\$RESULT_DIR/gate.log" || setup_fail gate-log-copy-failed
if [[ -f "\$ROOT/gate-work/madaros_native_multimodule_scale_901_source_fresh_receipt.tsv" ]]; then
  cp "\$ROOT/gate-work/madaros_native_multimodule_scale_901_source_fresh_receipt.tsv" "\$RESULT_DIR/receipt.tsv" || setup_fail gate-receipt-copy-failed
fi
if ! tar -C "\$ROOT" -czf "\$RESULT_DIR/gate-work.tgz" gate-work; then
  write_status 125 FAIL harness-routing evidence-bundle-write-failed
  rm -rf "\$ROOT"
  exit 125
fi
if [[ "\$RC" -eq 0 ]]; then
  write_status 0 PASS green
else
  write_status "\$RC" FAIL fixable-or-blocked-see-gate-log
fi
rm -rf "\$ROOT"
exit "\$RC"
EOF

"$KUBECTL_BIN" -n "$NS" cp "$SBATCH" "$LOGIN_POD:$REMOTE_SBATCH"
JOB_SUBMISSION="$("$KUBECTL_BIN" -n "$NS" exec "$LOGIN_POD" -- bash -lc "sbatch '$REMOTE_SBATCH'")"

printf '%s\n' "$JOB_SUBMISSION"
printf 'SOURCE_COMMIT=%s\n' "$SOURCE_COMMIT"
printf 'SOURCE_TREE=%s\n' "$SOURCE_TREE"
printf 'ARCHIVE_SHA256=%s\n' "$ARCHIVE_SHA256"
printf 'RESULT_DIR=%s\n' "$RESULT_DIR"
