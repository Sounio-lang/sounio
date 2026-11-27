#!/usr/bin/env bash
#SBATCH --job-name=gen2-gate-tb
#SBATCH --no-requeue
#SBATCH --partition=gpu-orangefs
#SBATCH --nodelist=gpuorangefs-multi-r740-proxmox
#SBATCH --comment=lane:trait-bounds
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/gen2-gate-tb-%j.log
#
# Fixed-point gate (scripts/ci/madaros_fixed_point_gate.sh) for PR #3755
# (trait bounds, E315/E316) @ 985be73a04, rebased on main 5848deb483.
# Inputs key 33c1136b8d991ce04c086b07a208e85769f8550a1764cc577e0f136a2bda40dd,
# self-hosted/ tree c8003e3328d5b58b7eb15940d047219f16ffc893.
#
# Prediction, registered before submission (2026-11-27 ~11:00 UTC): rung fixpoint
# (gen2 == gen3), rc 0, gen1/gen2 IR parity passes. Gate-step MaxRSS within
# 1.15 x 39,038,888 K (job 28684), wall within 1.3 x the #3748/#3750 receipts.
# Model: the change only adds refusals (E315/E316) for violated trait bounds and
# keeps FnDef.type_params; Madaros's own source declares no trait-bounded generic
# function that a call violates, so gen1 compiles it exactly as before in
# behaviour, and the self-compile is deterministic given its inputs (fixpoint
# receipts 28684/28886). gen2 differs from main's gen2 because the compiler
# source differs; that is expected and is not an anomaly. A gen2 != gen3 would be
# a real anomaly (the new checker or specializer path is not a fixed point),
# logged before deciding what is wrong.
# Results as Slurm step names: r-head-tree-*, r-head-key-*, r-head-<c>-gen1-*,
# r-head-gate-rc<rc>-s<sec>, r-head-gen2-*, r-head-gen3-*, r-head-irparity-*.

set -uo pipefail
HEAD_C="985be73a0478f23486adacc643deec7df58d4808"
HEAD_TREE="c8003e3328d5b58b7eb15940d047219f16ffc893"
SELF="$0"
REMOTE="https://github.com/Sounio-lang/sounio.git"
JOB="${SLURM_JOB_ID:-manual}"
WORK="/tmp/gen2-gate-tb-$JOB"
CA="$WORK/ca.crt"
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --job-name="$1" true < /dev/null > /dev/null 2>&1 || true; }
die() { rec "r-abort-$(echo "$*" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-60)"; exit 2; }
sha() { [[ -s "$1" ]] && sha256sum "$1" | cut -c1-64 || echo none; }
mkdir -p "$WORK/home" || die mkdir
export HOME="$WORK/home" XDG_CACHE_HOME="$WORK/home/.cache"
awk 'f { print } /^__CA_BUNDLE__$/ { f = 1 }' "$SELF" > "$CA"
grep -q 'BEGIN CERTIFICATE' "$CA" 2>/dev/null || cp /etc/ssl/certs/ca-certificates.crt "$CA" 2>/dev/null || die "no CA"
ulimit -s unlimited 2>/dev/null || true
unset SOUC_BIN SOUNIO_SOUC_BIN MADAROS_BIN MADAROS_RAW_BIN SOUNIO_MADAROS_CACHE
arm() {
  local n="$1" c="$2" tr="$3" d="$WORK/$1"
  git init -q "$d" || die "init $n"
  git -C "$d" -c http.sslCAInfo="$CA" fetch -q --depth 1 --no-tags "$REMOTE" "$c" > /dev/null 2>&1 || die "fetch $n"
  git -C "$d" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD || die "checkout $n"
  [[ "$(git -C "$d" rev-parse HEAD)" == "$c" ]] || die "HEAD $n"
  local got_tree; got_tree="$(git -C "$d" rev-parse HEAD:self-hosted)"
  rec "r-$n-tree-$got_tree"
  local ik; ik="$( (cd "$d" && bash scripts/ci/gen2_receipt_gate.sh 2>/dev/null) | sed -n "s/^gen2 inputs key at HEAD: //p")"
  rec "r-$n-key-${ik:-none}"
  [[ "$got_tree" == "$tr" ]] || die "tree $n"
  ( cd "$d" && SOUNIO_BUILD_LOCK="$WORK/$n.lock" bash scripts/ci/build_modular_madaros.sh "$WORK/$n.gen1" > "$WORK/$n.build.log" 2>&1 )
  [[ -x "$WORK/$n.gen1" ]] || die "build $n"
  rec "r-$n-${c:0:10}-gen1-$(sha "$WORK/$n.gen1")"
  local t0=$SECONDS rc=0
  ( cd "$d" && srun --ntasks=1 --cpus-per-task=4 --mem=0 --job-name="$n-gate" --export=ALL \
      env MADAROS_BIN="$WORK/$n.gen1" SOUNIO_MADAROS_FP_DIR="$WORK/$n.fp" SOUNIO_MADAROS_FP_MIN_INTO_ACC_DONE=122 \
      bash scripts/ci/madaros_fixed_point_gate.sh < /dev/null > "$WORK/$n.gate.log" 2>&1 ) || rc=$?
  rec "r-$n-gate-rc$rc-s$((SECONDS - t0))"
  rec "r-$n-gen2-$(sha "$WORK/$n.fp/madaros.gen2")"
  rec "r-$n-gen3-$(sha "$WORK/$n.fp/madaros.gen3")"
  grep -m 2 -iE "ir parity|parity" "$WORK/$n.gate.log" | while read -r l; do
    rec "r-$n-irparity-$(echo "$l" | tr -c 'A-Za-z0-9_.:=-' '_' | cut -c1-100)"; done
  { grep -m 6 -E "reached|rung|FAIL|error|Error|PASS|_OK" "$WORK/$n.gate.log"; tail -n 3 "$WORK/$n.gate.log"; } | while read -r l; do
    rec "r-$n-log-$(echo "$l" | tr -c 'A-Za-z0-9_.:=-' '_' | cut -c1-100)"; done
}
rec "r-node-$(hostname | cut -c1-30)"
arm head "$HEAD_C" "$HEAD_TREE"

rec "r-done"
exit 0
__CA_BUNDLE__
