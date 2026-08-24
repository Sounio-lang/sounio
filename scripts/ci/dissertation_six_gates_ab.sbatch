#!/usr/bin/env bash
#SBATCH --job-name=diss-gates-ab
#SBATCH --partition=cpu-ops
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/orangefs/training/slurm-diss-gates-ab-%j.out
#SBATCH --error=/orangefs/training/slurm-diss-gates-ab-%j.err

# DISAMBIGUATION RUN.
#
# Job 9904 measured the six dissertation gates on main HEAD with the CHECKED-IN
# bin/souc and got pass=1 fail=3 skip=2, against a June claim of 6/6. That is a
# real measurement but it does not identify a cause, because this project has
# twice been misled by a stale prebuilt binary: issue #1689 was an entire false
# investigation caused by one, and this session fable-1 proved that a Madaros
# built on a stale base produces PHANTOM parse failures in files that are
# actually pristine.
#
# So this job runs the same gates twice on the same tree: once with the
# checked-in binary (arm A, reproducing 9904) and once with a Madaros built
# FROM SOURCE on this node (arm B). Any gate that fails in A and passes in B was
# a stale-instrument artefact. Any gate that fails in BOTH is a real defect in
# the dissertation code, and that is the set that actually matters 37 days
# before the defense.
#
# Cluster correctness (probed, not assumed): compute nodes cannot see
# /workspace, --chdir must not inherit the submitter cwd, TMPDIR must be reset.

set -uo pipefail
export TMPDIR=/tmp

# Pin: default reproduces job 9904/9908 (June-gap measurement).
# Override for current tip: DISS_AB_COMMIT=$(git rev-parse origin/main) sbatch ...
SOURCE_COMMIT="${DISS_AB_COMMIT:-6f2c4e2461cc5da3a5952e1335d46b89e3160a4b}"
BRANCH="${DISS_AB_BRANCH:-main}"
RUN_ID="diss-gates-ab-${SOURCE_COMMIT:0:10}-$(date -u +%Y%m%dT%H%M%SZ)-job${SLURM_JOB_ID}"
STAGE_ROOT="/orangefs/training/${RUN_ID}"
REPO="${STAGE_ROOT}/repo"
OUT="${STAGE_ROOT}/gate-logs"
R="${STAGE_ROOT}/RESULT.txt"

mkdir -p "${STAGE_ROOT}" "${OUT}"
: > "$R"

{
  echo "SCHEMA=sounio.dissertation.six-gate-ab-instrument.v1"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
  echo "NODE=$(hostname)"
  echo "SOURCE_COMMIT=${SOURCE_COMMIT}"
  echo "PRIOR_RUN=job9904 (checked-in binary: pass=1 fail=3 skip=2)"
  echo "STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${STAGE_ROOT}/execution_context.txt"

git -c http.sslVerify=false clone --quiet --branch "${BRANCH}" --single-branch \
  https://github.com/Sounio-lang/sounio.git "${REPO}"
git -C "${REPO}" checkout --quiet --detach "${SOURCE_COMMIT}"
if [ "$(git -C "${REPO}" rev-parse HEAD)" != "${SOURCE_COMMIT}" ]; then
  echo "FATAL: commit pin not honoured" >> "$R"; cat "$R"; exit 2
fi
cd "${REPO}"

# Is node present on THIS node? The two gates that skipped in 9904 need it.
# Record the answer either way rather than silently skipping again.
if command -v node >/dev/null 2>&1; then
  echo "node_on_compute_node=$(node --version 2>&1)" >> "${STAGE_ROOT}/execution_context.txt"
else
  echo "node_on_compute_node=ABSENT (frontend_parity + pbpk28_parity will skip again)" \
    >> "${STAGE_ROOT}/execution_context.txt"
fi

GATES="confidence_gate dossier frontend_parity pbpk28_parity pbpk_hessian pbpk_suite"

run_arm () {
  arm="$1"; label="$2"
  echo "=== ARM ${arm}: ${label} ===" >> "$R"
  # Force Madaros wrapper: suite gate otherwise defaults SOUC_BIN to
  # souc-seq-leansingle.sh (Seq<T> lean_single path) which is NOT the A/B question.
  export SOUC_BIN="${DISS_AB_SOUC_BIN:-$REPO/bin/souc}"
  export SOUC="$SOUC_BIN"
  echo "  SOUC_BIN=${SOUC_BIN}" >> "$R"
  for g in $GATES; do
    # confidence gate file is dissertation_confidence_gate_gate.sh (double _gate)
    if [ "$g" = confidence_gate ]; then
      s="scripts/ci/dissertation_confidence_gate_gate.sh"
    else
      s="scripts/ci/dissertation_${g}_gate.sh"
    fi
    [ -f "$s" ] || { printf '  %-18s MISSING script=%s\n' "$g" "$s" >> "$R"; continue; }
    st=$(date +%s)
    timeout 1500 bash "$s" > "${OUT}/${arm}_${g}.log" 2>&1
    rc=$?
    el=$(( $(date +%s) - st ))
    # SKIP is not PASS: gates often exit 0 after printing SKIP (node absent).
    if grep -qiE "SKIP:|self-skip|SKIPPED \(|SKIP node" "${OUT}/${arm}_${g}.log" 2>/dev/null; then
      v=SKIP
      why=$(grep -iE "SKIP:|SKIPPED|node not available" "${OUT}/${arm}_${g}.log" | head -1 | tr -s " ")
      printf '  %-18s %-5s rc=%-3s %ss  why=%s\n' "$g" "$v" "$rc" "$el" "$why" >> "$R"
    elif [ "$rc" -eq 0 ]; then
      v=PASS
      printf '  %-18s %-5s rc=%-3s %ss\n' "$g" "$v" "$rc" "$el" >> "$R"
    else
      v=FAIL
      printf '  %-18s %-5s rc=%-3s %ss\n' "$g" "$v" "$rc" "$el" >> "$R"
    fi
    eval "${arm}_${g}=\$v"
  done
}

# ---- ARM A: the checked-in binary, reproducing job 9904 ----
{
  echo "arm_A_souc_md5=$(md5sum bin/souc 2>/dev/null | awk '{print $1}')"
  echo "arm_A_version=$(timeout 60 ./bin/souc --version 2>&1 | head -1)"
} >> "${STAGE_ROOT}/instrument.txt"
run_arm A "checked-in bin/souc (as job 9904)"

# ---- Build Madaros FROM SOURCE on this node ----
echo "" >> "$R"
echo "=== BUILDING MADAROS FROM SOURCE ===" >> "$R"
bst=$(date +%s)
timeout 5400 bash scripts/ci/build_modular_madaros.sh "${STAGE_ROOT}/madaros-from-source" \
  > "${OUT}/build.log" 2>&1
brc=$?
bel=$(( $(date +%s) - bst ))
echo "  build rc=${brc} elapsed=${bel}s" >> "$R"

if [ "$brc" -ne 0 ] || [ ! -x "${STAGE_ROOT}/madaros-from-source" ]; then
  {
    echo "  BUILD FAILED — arm B cannot run."
    echo "  That is itself a finding: main HEAD does not self-build here."
    echo "  Last 20 lines of build log:"
    tail -20 "${OUT}/build.log" | sed 's/^/    /'
  } >> "$R"
  cat "$R"; echo "STAGE_ROOT=${STAGE_ROOT}"; exit 0
fi

# Point the wrapper at the freshly built engine, the way the repo expects.
mkdir -p artifacts/self-hosted
cp "${STAGE_ROOT}/madaros-from-source" artifacts/self-hosted/madaros
chmod +x artifacts/self-hosted/madaros
{
  echo "arm_B_madaros_md5=$(md5sum artifacts/self-hosted/madaros | awk '{print $1}')"
  echo "arm_B_version=$(timeout 60 ./bin/souc --version 2>&1 | head -1)"
  echo "arm_B_build_elapsed_s=${bel}"
} >> "${STAGE_ROOT}/instrument.txt"

# ---- ARM B: same gates, from-source engine ----
echo "" >> "$R"
run_arm B "Madaros built from source this run"

# ---- The comparison that answers the question ----
{
  echo ""
  echo "=== VERDICT PER GATE ==="
  for g in $GATES; do
    a=$(eval "echo \${A_${g}:-?}"); b=$(eval "echo \${B_${g}:-?}")
    if [ "$a" = FAIL ] && [ "$b" = PASS ]; then note="STALE-INSTRUMENT ARTEFACT (A failed, B passed)"
    elif [ "$a" = FAIL ] && [ "$b" = FAIL ]; then note="REAL DEFECT (fails on both engines)"
    elif [ "$a" = SKIP ] || [ "$b" = SKIP ]; then note="UNMEASURED (skip != pass) A=$a B=$b"
    elif [ "$a" = "$b" ]; then note="same on both ($a)"
    else note="A=$a B=$b — inspect"; fi
    printf '  %-18s %s\n' "$g" "$note"
  done
  echo ""
  echo "FINISHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "NOTE: skip != pass. Gates needing Node skip when the node lacks it."
} >> "$R"

cat "$R"
cat "${STAGE_ROOT}/instrument.txt"
echo "STAGE_ROOT=${STAGE_ROOT}"
