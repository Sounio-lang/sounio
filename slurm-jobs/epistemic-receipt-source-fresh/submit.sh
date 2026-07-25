#!/usr/bin/env bash
# Submit an exact-commit direct-raw epistemic receipt gate to Slurm.
#
# The payload reconstructs the requested Git tree and installs the original
# commit object locally before running the gate. This lets the gate keep its
# clean-tree/HEAD provenance checks without transferring repository history.

set -euo pipefail

REPO="${REPO:-$(pwd)}"
SOURCE_REF="${SOURCE_REF:-HEAD}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-gpu-orangefs}"
ACCOUNT="${ACCOUNT:-lab}"
QOS="${QOS:-normal}"
JOB_MEM="${JOB_MEM:-64G}"
JOB_CPUS="${JOB_CPUS:-4}"
JOB_TIME="${JOB_TIME:-01:30:00}"
ORANGEFS_ROOT="${ORANGEFS_ROOT:-/orangefs/training/sounio/epistemic-receipt-source-fresh}"
RUN_ID="${RUN_ID:-epistemic-receipt-source-fresh-$(date -u +%Y%m%dT%H%M%S)}"

usage() {
    cat <<'EOF'
Usage:
  SOURCE_REF=<commit-or-ref> bash slurm-jobs/epistemic-receipt-source-fresh/submit.sh

Environment:
  REPO, SOURCE_REF, NS, KUBECTL, PARTITION, ACCOUNT, QOS, JOB_MEM, JOB_CPUS,
  JOB_TIME, ORANGEFS_ROOT, RUN_ID

The source checkout must be clean. The submitted job reconstructs the exact
source tree and runs scripts/ci/epistemic_receipt_source_fresh_gate.sh on a
current-source Madaros ELF. It does not make scientific or clinical claims.
EOF
}

fail() {
    echo "[epistemic-receipt-source-fresh-submit] FAIL: $*" >&2
    exit 1
}

if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
    usage
    exit 0
fi
[[ $# -eq 0 ]] || fail 'usage: submit.sh [--help]'
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe RUN_ID: $RUN_ID"

SOURCE_COMMIT="$(git -C "$REPO" rev-parse "$SOURCE_REF")"
SOURCE_TREE="$(git -C "$REPO" rev-parse "$SOURCE_COMMIT^{tree}")"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || fail 'source checkout is dirty; commit the candidate first'
git -C "$REPO" cat-file -e "$SOURCE_COMMIT:bin/souc-linux-x86_64" || fail 'source commit lacks bootstrap ELF bin/souc-linux-x86_64'
git -C "$REPO" cat-file -e "$SOURCE_COMMIT:scripts/ci/epistemic_receipt_source_fresh_gate.sh" || fail 'source commit lacks epistemic source-fresh gate'

LOGIN_POD="$($KUBECTL -n "$NS" get pods -l app.kubernetes.io/name=login --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
[[ -n "$LOGIN_POD" ]] || fail 'no running Slurm login pod'

LOCAL_ROOT="$(mktemp -d /tmp/epistemic-receipt-source-fresh-submit.XXXXXX)"
ARCHIVE="$LOCAL_ROOT/$RUN_ID.source.tgz"
COMMIT_OBJECT="$LOCAL_ROOT/$RUN_ID.commit"
SBATCH="$LOCAL_ROOT/$RUN_ID.sbatch"
RESULT_DIR="$ORANGEFS_ROOT/$RUN_ID.result"
REMOTE_ARCHIVE="$ORANGEFS_ROOT/$RUN_ID.source.tgz"
REMOTE_COMMIT="$ORANGEFS_ROOT/$RUN_ID.commit"

cleanup() {
    rm -rf "$LOCAL_ROOT"
}
trap cleanup EXIT

# Archive all tracked content: the reconstructed tree must equal SOURCE_TREE.
git -C "$REPO" archive --format=tar "$SOURCE_COMMIT" | gzip -1 >"$ARCHIVE"
git -C "$REPO" cat-file commit "$SOURCE_COMMIT" >"$COMMIT_OBJECT"

$KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "mkdir -p '$ORANGEFS_ROOT'"
$KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "test ! -e '$RESULT_DIR'" || fail "result directory already exists: $RESULT_DIR"
$KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "mkdir '$RESULT_DIR'"
$KUBECTL -n "$NS" cp "$ARCHIVE" "$LOGIN_POD:/tmp/$RUN_ID.source.tgz"
$KUBECTL -n "$NS" cp "$COMMIT_OBJECT" "$LOGIN_POD:/tmp/$RUN_ID.commit"
$KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "mv '/tmp/$RUN_ID.source.tgz' '$REMOTE_ARCHIVE'; mv '/tmp/$RUN_ID.commit' '$REMOTE_COMMIT'"

cat >"$SBATCH" <<EOF
#!/usr/bin/bash
#SBATCH -J $RUN_ID
#SBATCH -p $PARTITION
#SBATCH -A $ACCOUNT
#SBATCH --qos=$QOS
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c $JOB_CPUS
#SBATCH --mem=$JOB_MEM
#SBATCH --time=$JOB_TIME
#SBATCH -o /tmp/$RUN_ID-%j.slurmout
#SBATCH -e /tmp/$RUN_ID-%j.slurmerr
set -u -o pipefail
unset BASH_ENV ENV CDPATH GLOBIGNORE
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES
export PATH=/usr/bin:/bin
export HOME=/tmp

ROOT=/tmp/$RUN_ID-\${SLURM_JOB_ID:-manual}
REPO=\$ROOT/repo
RESULT_DIR=$RESULT_DIR
ARCHIVE=$REMOTE_ARCHIVE
COMMIT_OBJECT=$REMOTE_COMMIT
EXPECTED_COMMIT=$SOURCE_COMMIT
EXPECTED_TREE=$SOURCE_TREE

exec >"\$RESULT_DIR/runner.log" 2>&1

fail() {
  echo "[epistemic-receipt-source-fresh-slurm] FAIL: \$*" >&2
  exit 1
}

rm -rf "\$ROOT"
mkdir -p "\$REPO"
tar -xzf "\$ARCHIVE" -C "\$REPO" || fail 'source archive extraction failed'

# Recreate the source tree object from the full archive, then attach the exact
# original commit object. The replacement removes only unavailable history.
git -C "\$REPO" init -q || fail 'git init failed'
git -C "\$REPO" config user.name source-fresh-gate
git -C "\$REPO" config user.email source-fresh-gate@invalid
git -C "\$REPO" add -f -A || fail 'git add failed'
RECONSTRUCTED_TREE=\$(git -C "\$REPO" write-tree) || fail 'git write-tree failed'
[[ "\$RECONSTRUCTED_TREE" == "\$EXPECTED_TREE" ]] || fail "tree mismatch expected=\$EXPECTED_TREE reconstructed=\$RECONSTRUCTED_TREE"
INSTALLED_COMMIT=\$(git -C "\$REPO" hash-object -t commit -w "\$COMMIT_OBJECT") || fail 'commit object installation failed'
[[ "\$INSTALLED_COMMIT" == "\$EXPECTED_COMMIT" ]] || fail "commit mismatch expected=\$EXPECTED_COMMIT installed=\$INSTALLED_COMMIT"
git -C "\$REPO" update-ref refs/heads/source "\$EXPECTED_COMMIT" || fail 'source ref installation failed'
git -C "\$REPO" symbolic-ref HEAD refs/heads/source || fail 'HEAD installation failed'
git -C "\$REPO" replace --graft "\$EXPECTED_COMMIT" || fail 'history-free replacement installation failed'
[[ "\$(git -C "\$REPO" rev-parse HEAD)" == "\$EXPECTED_COMMIT" ]] || fail 'HEAD does not match requested commit'
[[ "\$(git -C "\$REPO" rev-parse 'HEAD^{tree}')" == "\$EXPECTED_TREE" ]] || fail 'HEAD tree does not match requested tree'
[[ -z "\$(git -C "\$REPO" status --porcelain)" ]] || fail 'reconstructed source tree is not clean'

chmod +x "\$REPO/bin/souc-linux-x86_64" \
  "\$REPO/scripts/ci/build_modular_madaros.sh" \
  "\$REPO/scripts/ci/epistemic_receipt_source_fresh_gate.sh" \
  "\$REPO/scripts/dev/souc-build-lock.sh"

{
  echo "requested_commit=\$EXPECTED_COMMIT"
  echo "requested_tree=\$EXPECTED_TREE"
  echo "reconstructed_tree=\$RECONSTRUCTED_TREE"
  echo "job_id=\${SLURM_JOB_ID:-manual}"
  echo "host=\$(hostname)"
  echo "partition=$PARTITION"
  echo "cpus=$JOB_CPUS mem=$JOB_MEM"
} >"\$RESULT_DIR/environment.tsv"

SOUNIO_EPISTEMIC_RECEIPT_SOURCE_FRESH_KEEP=1 \
SOUNIO_EPISTEMIC_RECEIPT_SOURCE_FRESH_DIR="\$RESULT_DIR/gate-work" \
  bash "\$REPO/scripts/ci/epistemic_receipt_source_fresh_gate.sh" >"\$RESULT_DIR/gate.log" 2>&1
RC=\$?
printf 'exit_code\\t%s\\n' "\$RC" >"\$RESULT_DIR/status.tsv"
if [[ "\$RC" -eq 0 ]]; then
  echo 'status=PASS' >>"\$RESULT_DIR/status.tsv"
else
  echo 'status=FAIL' >>"\$RESULT_DIR/status.tsv"
fi
exit "\$RC"
EOF

$KUBECTL -n "$NS" cp "$SBATCH" "$LOGIN_POD:/tmp/$RUN_ID.sbatch"
$KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "mv '/tmp/$RUN_ID.sbatch' '$ORANGEFS_ROOT/$RUN_ID.sbatch'"
JOB_SUBMISSION="$($KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "sbatch --parsable --hold --export=NIL --chdir='$ORANGEFS_ROOT' '$ORANGEFS_ROOT/$RUN_ID.sbatch'")"
JOB_ID="${JOB_SUBMISSION%%;*}"
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || fail "sbatch did not return a numeric job id: $JOB_SUBMISSION"
$KUBECTL -n "$NS" exec "$LOGIN_POD" -- bash -lc "sha256sum '$REMOTE_ARCHIVE' '$REMOTE_COMMIT' '$ORANGEFS_ROOT/$RUN_ID.sbatch' > '$RESULT_DIR/submission.sha256'; printf 'job_id=%s\\nsource_commit=%s\\nsource_tree=%s\\n' '$JOB_ID' '$SOURCE_COMMIT' '$SOURCE_TREE' > '$RESULT_DIR/submission.tsv'"
$KUBECTL -n "$NS" exec "$LOGIN_POD" -- scontrol release "$JOB_ID"

echo "Submitted batch job $JOB_ID"
echo "SOURCE_COMMIT=$SOURCE_COMMIT"
echo "SOURCE_TREE=$SOURCE_TREE"
echo "RESULT_DIR=$RESULT_DIR"
