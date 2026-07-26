#!/usr/bin/env bash
# Submit the exact-commit direct-raw epistemic receipt gate to a worker-local
# Slurm allocation. This is the no-BeagleCockpit-MCP fallback: the worker
# fetches the published Git commit into its own clone/PVC cache, then builds
# Madaros from that clean source tree. It never reads or writes OrangeFS.

set -euo pipefail

REPO="${REPO:-$(pwd)}"
SOURCE_REF="${SOURCE_REF:-HEAD}"
SOURCE_REMOTE="${SOURCE_REMOTE:-}"
WORKER_GIT_SSL_VERIFY="${WORKER_GIT_SSL_VERIFY:-true}"
WORKER_LOWER_TRACE="${WORKER_LOWER_TRACE:-0}"
WORKER_NV2_IR_TRACE="${WORKER_NV2_IR_TRACE:-0}"
WORKER_PROBE="${WORKER_PROBE:-source-fresh-gate}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
PARTITION="${PARTITION:-gpu-orangefs}"
NODELIST="${NODELIST:-}"
JOB_MEM="${JOB_MEM:-64G}"
JOB_CPUS="${JOB_CPUS:-4}"
JOB_TIME="${JOB_TIME:-01:30:00}"
RUN_ID="${RUN_ID:-epistemic-receipt-source-fresh-$(date -u +%Y%m%dT%H%M%S)}"

usage() {
    cat <<'EOF'
Usage:
  SOURCE_REF=<commit-or-ref> bash slurm-jobs/epistemic-receipt-source-fresh/submit.sh

Environment:
  REPO, SOURCE_REF, SOURCE_REMOTE, WORKER_GIT_SSL_VERIFY, WORKER_LOWER_TRACE,
  WORKER_NV2_IR_TRACE, WORKER_PROBE, NS, KUBECTL, PARTITION, NODELIST, JOB_MEM, JOB_CPUS,
  JOB_TIME, RUN_ID

This is the direct-Slurm fallback for sessions where BeagleCockpit MCP is not
loaded. It streams the worker's gate output back through srun. The worker
clones the exact published source commit into worker-local scratch/PVC cache;
OrangeFS is intentionally not used.

Set WORKER_GIT_SSL_VERIFY=false only for a worker image with a documented
missing CA bundle. That transport exception is reported in the job output;
the requested commit and tree are still checked before the source build.

Set WORKER_LOWER_TRACE=1 only for crash localization. It enables the existing
module-frontend and IR-lowering traces inside the worker's raw ELF.

Set WORKER_NV2_IR_TRACE=1 only for backend crash localization. It enables the
existing native-v2 IR trace inside the worker's raw ELF.

WORKER_PROBE=source-fresh-gate runs the acceptance gate. WORKER_PROBE=block-ladder
builds the same raw ELF and runs twelve generated worker-local programs to locate
the first block shape that reproduces a lowering crash. The probe never changes
the checked-out source tree and is not acceptance evidence.
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
[[ "$JOB_CPUS" =~ ^[1-9][0-9]*$ ]] || fail "invalid JOB_CPUS: $JOB_CPUS"
[[ "$WORKER_GIT_SSL_VERIFY" == 'true' || "$WORKER_GIT_SSL_VERIFY" == 'false' ]] || fail "invalid WORKER_GIT_SSL_VERIFY: $WORKER_GIT_SSL_VERIFY"
[[ "$WORKER_LOWER_TRACE" == '0' || "$WORKER_LOWER_TRACE" == '1' ]] || fail "invalid WORKER_LOWER_TRACE: $WORKER_LOWER_TRACE"
[[ "$WORKER_NV2_IR_TRACE" == '0' || "$WORKER_NV2_IR_TRACE" == '1' ]] || fail "invalid WORKER_NV2_IR_TRACE: $WORKER_NV2_IR_TRACE"
[[ "$WORKER_PROBE" == 'source-fresh-gate' || "$WORKER_PROBE" == 'block-ladder' ]] || fail "invalid WORKER_PROBE: $WORKER_PROBE"

SOURCE_COMMIT="$(git -C "$REPO" rev-parse "$SOURCE_REF")"
SOURCE_TREE="$(git -C "$REPO" rev-parse "$SOURCE_COMMIT^{tree}")"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || fail 'source checkout is dirty; commit the candidate first'
git -C "$REPO" cat-file -e "$SOURCE_COMMIT:bin/souc-linux-x86_64" || fail 'source commit lacks bootstrap ELF bin/souc-linux-x86_64'
git -C "$REPO" cat-file -e "$SOURCE_COMMIT:scripts/ci/epistemic_receipt_source_fresh_gate.sh" || fail 'source commit lacks epistemic source-fresh gate'

if [[ -z "$SOURCE_REMOTE" ]]; then
    SOURCE_REMOTE="$(git -C "$REPO" remote get-url origin)"
fi
[[ "$SOURCE_REMOTE" =~ ^(https://|http://|git@) ]] || fail "unsupported SOURCE_REMOTE: $SOURCE_REMOTE"

LOGIN_POD="$($KUBECTL -n "$NS" get pods -l app.kubernetes.io/name=login --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
[[ -n "$LOGIN_POD" ]] || fail 'no running Slurm login pod'

LOCAL_ROOT="$(mktemp -d /tmp/epistemic-receipt-source-fresh-submit.XXXXXX)"
RUNNER="$LOCAL_ROOT/$RUN_ID.runner.sh"
REMOTE_RUNNER="/tmp/$RUN_ID.runner.sh"

cleanup() {
    rm -rf "$LOCAL_ROOT"
}
trap cleanup EXIT

cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
unset BASH_ENV ENV CDPATH GLOBIGNORE
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES
export PATH=/usr/bin:/bin
export HOME=/tmp

ROOT="\${TMPDIR:-/tmp}/$RUN_ID-\${SLURM_JOB_ID:-manual}"
REPO="\$ROOT/repo"
EXPECTED_COMMIT="$SOURCE_COMMIT"
EXPECTED_TREE="$SOURCE_TREE"
SOURCE_REMOTE="$SOURCE_REMOTE"
WORKER_GIT_SSL_VERIFY="$WORKER_GIT_SSL_VERIFY"
WORKER_LOWER_TRACE="$WORKER_LOWER_TRACE"
WORKER_NV2_IR_TRACE="$WORKER_NV2_IR_TRACE"
WORKER_PROBE="$WORKER_PROBE"

cleanup() {
  rm -rf "\$ROOT"
}
trap cleanup EXIT

fail() {
  echo "[epistemic-receipt-source-fresh-slurm] FAIL: \$*" >&2
  exit 1
}

echo "source_fresh_slurm_job_id=\${SLURM_JOB_ID:-manual}"
echo "source_remote=\$SOURCE_REMOTE"
echo "worker_git_ssl_verify=\$WORKER_GIT_SSL_VERIFY"
echo "worker_lower_trace=\$WORKER_LOWER_TRACE"
echo "worker_nv2_ir_trace=\$WORKER_NV2_IR_TRACE"
echo "worker_probe=\$WORKER_PROBE"
echo "requested_commit=\$EXPECTED_COMMIT"
echo "requested_tree=\$EXPECTED_TREE"
echo "worker_host=\$(hostname)"
echo "worker_root=\$ROOT"

rm -rf "\$ROOT"
mkdir -p "\$REPO"
git -C "\$REPO" init -q || fail 'git init failed'
git -C "\$REPO" remote add origin "\$SOURCE_REMOTE" || fail 'git remote setup failed'
if [[ "\$WORKER_GIT_SSL_VERIFY" == 'false' ]]; then
  echo 'worker_git_tls_transport=unverified-ca-bundle-missing'
  git -C "\$REPO" -c http.sslVerify=false fetch --no-tags --depth=1 origin "\$EXPECTED_COMMIT" || fail 'worker could not fetch requested source commit'
else
  git -C "\$REPO" fetch --no-tags --depth=1 origin "\$EXPECTED_COMMIT" || fail 'worker could not fetch requested source commit'
fi
git -C "\$REPO" checkout -q --detach FETCH_HEAD || fail 'worker could not checkout requested source commit'

[[ "\$(git -C "\$REPO" rev-parse HEAD)" == "\$EXPECTED_COMMIT" ]] || fail 'worker HEAD does not match requested commit'
[[ "\$(git -C "\$REPO" rev-parse 'HEAD^{tree}')" == "\$EXPECTED_TREE" ]] || fail 'worker tree does not match requested tree'
[[ -z "\$(git -C "\$REPO" status --porcelain)" ]] || fail 'worker source tree is not clean'

if [[ "\$WORKER_LOWER_TRACE" == '1' ]]; then
  export SOUNIO_MODULE_FRONTEND_LOWER_TRACE=1
  export SOUNIO_LOWER_SUMMARY_TRACE=1
  export SOUNIO_LOWER_LIVE_TRACE=1
fi

if [[ "\$WORKER_NV2_IR_TRACE" == '1' ]]; then
  export SOUNIO_NV2_IR_TRACE=1
fi

chmod +x "\$REPO/bin/souc-linux-x86_64" \\
  "\$REPO/scripts/ci/build_modular_madaros.sh" \\
  "\$REPO/scripts/ci/epistemic_receipt_source_fresh_gate.sh" \\
  "\$REPO/scripts/dev/souc-build-lock.sh"

cd "\$REPO"
if [[ "\$WORKER_PROBE" == 'block-ladder' ]]; then
  RAW_MADAROS="\$ROOT/probe-madaros"
  PROBE_ROOT="\$ROOT/block-ladder"
  BUILD_LOG="\$ROOT/probe-build.log"
  mkdir -p "\$PROBE_ROOT/cwd"
  if ! bash "\$REPO/scripts/ci/build_modular_madaros.sh" "\$RAW_MADAROS" >"\$BUILD_LOG" 2>&1; then
    tail -n 120 "\$BUILD_LOG" >&2 || true
    fail 'block-ladder current-source build failed'
  fi
  [[ -x "\$RAW_MADAROS" ]] || fail 'block-ladder build did not emit an executable raw ELF'
  [[ -z "\$(git -C "\$REPO" status --porcelain)" ]] || fail 'source tree changed during block-ladder build'

  cat >"\$PROBE_ROOT/empty_main.sio" <<'SIO'
fn main() with IO, Panic {
}
SIO
  cat >"\$PROBE_ROOT/local_assert_helper.sio" <<'SIO'
fn op_require_nonzero(tag: i64) -> i64 with Panic {
    assert(tag != 0)
    tag
}

fn main() with IO, Panic {
}
SIO
  cat >"\$PROBE_ROOT/local_param_identity.sio" <<'SIO'
fn local_param_identity(tag: i64) -> i64 {
    tag
}

fn main() with IO, Panic {
}
SIO
  cat >"\$PROBE_ROOT/local_literal_let.sio" <<'SIO'
fn local_literal_let(tag: i64) -> i64 {
    let marker = 1
    1
}

fn main() with IO, Panic {
}
SIO
  cat >"\$PROBE_ROOT/local_literal_comparison_tail.sio" <<'SIO'
fn local_literal_comparison_tail() -> bool {
    1 != 0
}

fn main() with IO, Panic {
}
SIO
  cat >"\$PROBE_ROOT/local_comparison_tail.sio" <<'SIO'
fn local_comparison_tail(tag: i64) -> bool {
    tag != 0
}

fn main() with IO, Panic {
}
SIO
  cat >"\$PROBE_ROOT/local_comparison_let.sio" <<'SIO'
fn local_comparison_let(tag: i64) -> i64 {
    let predicate = tag != 0
    1
}

fn main() with IO, Panic {
}
SIO
  cat >"\$PROBE_ROOT/local_assert_constant.sio" <<'SIO'
fn local_assert_constant(tag: i64) -> i64 with Panic {
    assert(1 != 0)
    1
}

fn main() with IO, Panic {
}
SIO
  cat >"\$PROBE_ROOT/local_assert_literal.sio" <<'SIO'
fn local_assert_literal(tag: i64) -> i64 with Panic {
    assert(tag != 0)
    1
}

fn main() with IO, Panic {
}
SIO
  cat >"\$PROBE_ROOT/import_empty.sio" <<'SIO'
use epistemic::observation_provenance::*

fn main() with IO, Panic {
}
SIO
  cat >"\$PROBE_ROOT/import_literal_let.sio" <<'SIO'
use epistemic::observation_provenance::*

fn main() with IO, Panic {
    let marker = 1
}
SIO
  cat >"\$PROBE_ROOT/import_first_call.sio" <<'SIO'
use epistemic::observation_provenance::*

fn main() with IO, Panic {
    let opportunity = op_observation_opportunity_i64(10, 11)
}
SIO

  run_probe() {
    local label="\$1"
    local source="\$2"
    echo "block_ladder_begin=\$label"
    (
      cd "\$PROBE_ROOT/cwd"
      exec env \\
        -u MADAROS_RAW_BIN \\
        -u SOUNIO_MADAROS_BIN \\
        SOUNIO_STDLIB_PATH="\$REPO/stdlib" \\
        "\$RAW_MADAROS" run "\$source"
    )
    echo "block_ladder_pass=\$label"
  }

  run_probe empty_main "\$PROBE_ROOT/empty_main.sio"
  run_probe local_param_identity "\$PROBE_ROOT/local_param_identity.sio"
  run_probe local_literal_let "\$PROBE_ROOT/local_literal_let.sio"
  run_probe local_literal_comparison_tail "\$PROBE_ROOT/local_literal_comparison_tail.sio"
  run_probe local_comparison_tail "\$PROBE_ROOT/local_comparison_tail.sio"
  run_probe local_comparison_let "\$PROBE_ROOT/local_comparison_let.sio"
  run_probe local_assert_constant "\$PROBE_ROOT/local_assert_constant.sio"
  run_probe local_assert_literal "\$PROBE_ROOT/local_assert_literal.sio"
  run_probe local_assert_helper "\$PROBE_ROOT/local_assert_helper.sio"
  run_probe import_empty "\$PROBE_ROOT/import_empty.sio"
  run_probe import_literal_let "\$PROBE_ROOT/import_literal_let.sio"
  run_probe import_first_call "\$PROBE_ROOT/import_first_call.sio"
  [[ -z "\$(git -C "\$REPO" status --porcelain)" ]] || fail 'source tree changed during block-ladder probe'
  echo '[epistemic-receipt-source-fresh] PASS: block-ladder completed'
else
  SOUNIO_EPISTEMIC_RECEIPT_SOURCE_FRESH_KEEP=0 \\
    bash "\$REPO/scripts/ci/epistemic_receipt_source_fresh_gate.sh"
fi
EOF
chmod 700 "$RUNNER"

$KUBECTL -n "$NS" exec "$LOGIN_POD" -- /usr/bin/bash -lc '
  set -euo pipefail
  test -S /run/slurm/sack.socket
  scontrol ping >/dev/null
'
$KUBECTL -n "$NS" cp "$RUNNER" "$LOGIN_POD:$REMOTE_RUNNER"

SRUN_ARGS=(
    --label
    --unbuffered
    --kill-on-bad-exit=1
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$JOB_CPUS"
    --mem="$JOB_MEM"
    --time="$JOB_TIME"
    --export=NIL
    --chdir=/tmp
)
if [[ -n "$NODELIST" ]]; then
    SRUN_ARGS+=(--nodelist="$NODELIST")
fi
printf -v SRUN_RENDERED '%q ' "${SRUN_ARGS[@]}"

set +e
$KUBECTL -n "$NS" exec "$LOGIN_POD" -- /usr/bin/bash -lc "
  set -uo pipefail
  base64 -w 0 '$REMOTE_RUNNER' | srun $SRUN_RENDERED /usr/bin/bash -lc 'base64 -d > /tmp/$RUN_ID.runner.sh && chmod 700 /tmp/$RUN_ID.runner.sh && exec /usr/bin/bash /tmp/$RUN_ID.runner.sh'
  rc=\${PIPESTATUS[1]}
  rm -f '$REMOTE_RUNNER'
  exit \$rc
"
RC=$?
set -e

if [[ "$RC" -eq 0 ]]; then
    echo "[epistemic-receipt-source-fresh-submit] PASS: source_commit=$SOURCE_COMMIT source_tree=$SOURCE_TREE transport=worker-local-git-clone"
else
    echo "[epistemic-receipt-source-fresh-submit] FAIL: source_commit=$SOURCE_COMMIT source_tree=$SOURCE_TREE transport=worker-local-git-clone rc=$RC" >&2
fi
exit "$RC"
