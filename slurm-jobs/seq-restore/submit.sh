#!/usr/bin/env bash
# slurm-jobs/seq-restore/submit.sh — build-verify job for Seq<T> restoration
# (branch feat/seq-restore, commit 671f4a04e).
#
# WHAT THIS DOES
# --------------
# 1. Stages lean_single.sio + bin/souc-linux-x86_64 + required test/stdlib files
#    onto OrangeFS (/orangefs/training/sounio/seq-restore/<RUN_ID>/).
# 2. Submits a cpu-ops sbatch job that:
#    a. Runs the 3-stage JIT bootstrap (gen1 → gen2 → gen3) on the staged source.
#    b. Asserts gen2.elf md5 == gen3.elf md5 (fixed-point gate).
#    c. Compiles tests/run-pass/seq_kaxi_fuse.sio → seq_kaxi_fuse.elf and runs it;
#       asserts output contains "seq_kaxi_fuse: PASS".
#    d. Compiles tests/run-pass/rapamycin_kaxi_fuse_prior.sio similarly;
#       asserts output contains "PASS".
#    e. Writes SLURM output to /orangefs/training/sounio/seq-restore/<RUN_ID>/result.txt
#       with final PASS/FAIL summary.
# 3. Streams the job ID so you can poll with `squeue -j <JID>` and fetch
#    /orangefs/training/sounio/seq-restore/<RUN_ID>/result.txt when done.
#
# PREREQUISITES
# -------------
#   - Run from inside the /tmp/seq-restore worktree (or any clone with this branch).
#   - The worktree MUST have bin/souc-linux-x86_64 (the existing seed binary).
#   - cpu-ops partition must be up (verify with `sinfo`).
#   - The staging host (where you run this script) must be able to write to OrangeFS.
#     If running from the k8s login pod (slurm-pilot), OrangeFS is at /orangefs/training.
#
# USAGE
# -----
#   cd /tmp/seq-restore
#   bash slurm-jobs/seq-restore/submit.sh [--dry-run]
#
# FILESYSTEM NOTES
# ----------------
# Compute nodes in cpu-ops do NOT mount /workspace. They DO see /orangefs/training
# (OrangeFS/pvfs2). This script stages everything needed under a RUN_ID directory
# on OrangeFS, so the sbatch job is self-contained. The login pod (this script's
# host) ALSO cannot see /orangefs/training directly — use `srun --pty bash` on the
# login node OR copy via `kubectl cp` to the slurm-pilot pod if needed.
#
# If you are submitting from the workspace pod (where /orangefs is not mounted),
# add `--remote-stage` and the script will kubectl-cp the tarball to slurm-pilot
# and run staging from there.

set -euo pipefail

WORKTREE="${WORKTREE:-$(pwd)}"
PART="${PART:-cpu-ops}"
MEM="${MEM:-16G}"
CPUS="${CPUS:-4}"
TIMELIMIT="${TIMELIMIT:-00:30:00}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

RUN_ID="seq-restore-$(date -u +%Y%m%dT%H%M%S)-$$"
STAGE_DIR="/orangefs/training/sounio/seq-restore/${RUN_ID}"
TARBALL="/tmp/${RUN_ID}.tgz"

echo "[seq-restore] RUN_ID = ${RUN_ID}"
echo "[seq-restore] WORKTREE = ${WORKTREE}"
echo "[seq-restore] STAGE_DIR = ${STAGE_DIR}"

# ── Preflight ──────────────────────────────────────────────────────────────────
if [[ ! -f "${WORKTREE}/self-hosted/compiler/lean_single.sio" ]]; then
  echo "ERROR: ${WORKTREE}/self-hosted/compiler/lean_single.sio not found" >&2
  exit 1
fi
if [[ ! -x "${WORKTREE}/bin/souc-linux-x86_64" ]]; then
  echo "ERROR: ${WORKTREE}/bin/souc-linux-x86_64 not found/not executable" >&2
  exit 1
fi
if ! grep -q "seq_hash_base\|SEQ_INNER_TY" "${WORKTREE}/self-hosted/compiler/lean_single.sio"; then
  echo "ERROR: Seq<T> patch not found in lean_single.sio — are you on feat/seq-restore?" >&2
  exit 1
fi

# ── Build staging tarball ──────────────────────────────────────────────────────
# Include only what the sbatch job needs to run fully self-contained on OrangeFS.
echo "[1/4] Building staging tarball..."
REQUIRED_PATHS=(
  "self-hosted/compiler/lean_single.sio"
  "bin/souc-linux-x86_64"
  "tests/run-pass/seq_kaxi_fuse.sio"
  "tests/run-pass/rapamycin_kaxi_fuse_prior.sio"
  "stdlib/epistemic"
  "stdlib/units"
  "stdlib/physics"
)
MISSING=()
for p in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "${WORKTREE}/${p}" ]]; then
    MISSING+=("${p}")
  fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "ERROR: required staging files missing from worktree:" >&2
  printf '  %s\n' "${MISSING[@]}" >&2
  exit 1
fi

tar -czf "${TARBALL}" \
  --transform "s|^|${RUN_ID}/|" \
  -C "${WORKTREE}" \
  "${REQUIRED_PATHS[@]}"

echo "  tarball: ${TARBALL} ($(stat -c%s "${TARBALL}") bytes)"

# ── Stage on OrangeFS ─────────────────────────────────────────────────────────
# This only works if /orangefs/training is mounted on this host.
# If not, kubectl-cp to slurm-pilot pod first; see FILESYSTEM NOTES above.
if [[ -d "/orangefs/training" ]]; then
  echo "[2/4] Staging to OrangeFS..."
  mkdir -p "${STAGE_DIR}"
  tar -xzf "${TARBALL}" -C "/orangefs/training/sounio/seq-restore/"
  echo "  staged: ${STAGE_DIR}"
else
  echo "[2/4] /orangefs/training not mounted on this host."
  echo "  To stage: kubectl cp ${TARBALL} <slurm-pilot-pod>:/tmp/ && kubectl exec <slurm-pilot-pod> -- tar -xzf /tmp/$(basename "${TARBALL}") -C /orangefs/training/sounio/seq-restore/"
  if [[ ${DRY_RUN} -eq 0 ]]; then
    echo "  Re-run with --dry-run to skip staging, or stage manually first."
    exit 1
  fi
fi

# ── Write sbatch job ───────────────────────────────────────────────────────────
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
cat > "${SBATCH_FILE}" <<SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=seq-restore-verify
#SBATCH --partition=${PART}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIMELIMIT}
#SBATCH --output=/orangefs/training/sounio/seq-restore/${RUN_ID}/slurm-%j.log
#SBATCH --chdir=/orangefs/training/sounio/seq-restore/${RUN_ID}

set -euo pipefail
WDIR="/orangefs/training/sounio/seq-restore/${RUN_ID}"
RESULT="\${WDIR}/result.txt"
SOUC="\${WDIR}/bin/souc-linux-x86_64"

echo "=== seq-restore verification job @ \$(hostname) @ \$(date -u) ===" | tee "\${RESULT}"
echo "RUN_ID: ${RUN_ID}" | tee -a "\${RESULT}"

# ── Stage 0: seed binary compiles lean_single → gen1.elf ───────────────────
echo "[stage0] seeding gen1.elf..." | tee -a "\${RESULT}"
t0=\$(date +%s)
"\${SOUC}" "\${WDIR}/self-hosted/compiler/lean_single.sio" "\${WDIR}/gen1.elf"
chmod +x "\${WDIR}/gen1.elf"
echo "  gen1.elf: \$(stat -c%s "\${WDIR}/gen1.elf") bytes in \$((\$(date +%s)-t0))s" | tee -a "\${RESULT}"

# ── Stage 1: gen1 compiles lean_single → gen2.elf ──────────────────────────
echo "[stage1] gen1 → gen2.elf..." | tee -a "\${RESULT}"
t1=\$(date +%s)
"\${WDIR}/gen1.elf" "\${WDIR}/self-hosted/compiler/lean_single.sio" "\${WDIR}/gen2.elf"
chmod +x "\${WDIR}/gen2.elf"
echo "  gen2.elf: \$(stat -c%s "\${WDIR}/gen2.elf") bytes in \$((\$(date +%s)-t1))s" | tee -a "\${RESULT}"

# ── Stage 2: gen2 compiles lean_single → gen3.elf ──────────────────────────
echo "[stage2] gen2 → gen3.elf..." | tee -a "\${RESULT}"
t2=\$(date +%s)
"\${WDIR}/gen2.elf" "\${WDIR}/self-hosted/compiler/lean_single.sio" "\${WDIR}/gen3.elf"
chmod +x "\${WDIR}/gen3.elf"
echo "  gen3.elf: \$(stat -c%s "\${WDIR}/gen3.elf") bytes in \$((\$(date +%s)-t2))s" | tee -a "\${RESULT}"

# ── Fixed-point gate: gen2 == gen3 ─────────────────────────────────────────
MD2=\$(md5sum "\${WDIR}/gen2.elf" | awk '{print \$1}')
MD3=\$(md5sum "\${WDIR}/gen3.elf" | awk '{print \$1}')
echo "[fixed-point] gen2 md5=\${MD2}  gen3 md5=\${MD3}" | tee -a "\${RESULT}"
if [[ "\${MD2}" != "\${MD3}" ]]; then
  echo "FAIL: gen2 != gen3 (fixed-point violated — codegen divergence in lean_single)" | tee -a "\${RESULT}"
  exit 1
fi
echo "  fixed-point: PASS (gen2 == gen3)" | tee -a "\${RESULT}"

# Use gen3 as the verified compiler for all subsequent steps
VERIFIED_SOUC="\${WDIR}/gen3.elf"

# ── seq_kaxi_fuse test ──────────────────────────────────────────────────────
echo "[test1] seq_kaxi_fuse.sio..." | tee -a "\${RESULT}"
"\${VERIFIED_SOUC}" "\${WDIR}/tests/run-pass/seq_kaxi_fuse.sio" "\${WDIR}/seq_kaxi_fuse.elf"
chmod +x "\${WDIR}/seq_kaxi_fuse.elf"
SEQ_OUT=\$("\${WDIR}/seq_kaxi_fuse.elf" 2>&1 || true)
echo "  output: \${SEQ_OUT}" | tee -a "\${RESULT}"
if echo "\${SEQ_OUT}" | grep -q "seq_kaxi_fuse: PASS"; then
  echo "  seq_kaxi_fuse: PASS" | tee -a "\${RESULT}"
else
  echo "FAIL: seq_kaxi_fuse did not output 'seq_kaxi_fuse: PASS'" | tee -a "\${RESULT}"
  exit 1
fi

# ── rapamycin_kaxi_fuse_prior test ─────────────────────────────────────────
# This test uses 'use epistemic::kaxi::*' — the compiler resolves stdlib imports
# relative to the source file path. Stage WDIR/stdlib to satisfy the import.
echo "[test2] rapamycin_kaxi_fuse_prior.sio..." | tee -a "\${RESULT}"
"\${VERIFIED_SOUC}" "\${WDIR}/tests/run-pass/rapamycin_kaxi_fuse_prior.sio" "\${WDIR}/rapa.elf"
chmod +x "\${WDIR}/rapa.elf"
RAPA_OUT=\$("\${WDIR}/rapa.elf" 2>&1 || true)
echo "  output: \${RAPA_OUT}" | tee -a "\${RESULT}"
if echo "\${RAPA_OUT}" | grep -q "PASS"; then
  echo "  rapamycin_kaxi_fuse_prior: PASS" | tee -a "\${RESULT}"
else
  echo "FAIL: rapamycin_kaxi_fuse_prior did not output 'PASS'" | tee -a "\${RESULT}"
  exit 1
fi

echo "" | tee -a "\${RESULT}"
echo "=== ALL GATES PASSED: fixed-point + seq_kaxi_fuse + rapamycin_kaxi_fuse_prior ===" | tee -a "\${RESULT}"
echo "  Verified souc: \${VERIFIED_SOUC}" | tee -a "\${RESULT}"
echo "  Copy gen3.elf as the new bin/souc-linux-x86_64 to update the seed binary." | tee -a "\${RESULT}"
SBATCH_EOF

echo "[3/4] sbatch file: ${SBATCH_FILE}"

# ── Submit or dry-run ─────────────────────────────────────────────────────────
if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "[4/4] DRY RUN — not submitting. sbatch file at ${SBATCH_FILE}"
  echo ""
  echo "  To submit manually:"
  echo "    sbatch ${SBATCH_FILE}"
  echo ""
  echo "  To poll:"
  echo "    squeue -u \$(whoami)"
  echo ""
  echo "  To fetch results:"
  echo "    cat ${STAGE_DIR}/result.txt"
else
  echo "[4/4] Submitting to SLURM partition=${PART}..."
  JOB_ID=$(sbatch "${SBATCH_FILE}" | awk '{print $NF}')
  echo ""
  echo "  Submitted job ID: ${JOB_ID}"
  echo "  Poll: squeue -j ${JOB_ID}"
  echo "  Results: ${STAGE_DIR}/result.txt"
  echo "  Full SLURM log: ${STAGE_DIR}/slurm-${JOB_ID}.log"
fi

echo ""
echo "[seq-restore] DONE. Patch on branch feat/seq-restore @ /tmp/seq-restore"
echo "              Full diff: git -C /tmp/seq-restore diff HEAD~1 HEAD -- self-hosted/compiler/lean_single.sio"
