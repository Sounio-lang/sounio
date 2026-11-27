#!/usr/bin/env bash
#SBATCH --job-name=gen2-gate-3743e
#SBATCH --no-requeue
#SBATCH --partition=gpu-orangefs
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/gen2-gate-3743e-%j.log
#
# Fixed-point gate (scripts/ci/madaros_fixed_point_gate.sh, workflow env) for
# #3743 @ 94d93f125c: 0073f4a5ba (main 5848deb483 merged in) plus the exporter-aware
# unit-shadowing fix (ast_type_reaches, pfi_note_type_reach). self-hosted/ tree
# 662dd3e593f9c7dec858bd0c75e6f5f88d2f3a76, inputs key b2bc081f2c39cc3bacfc658d84c66d4e385b06a40f9833c66e38807d2dd0fe43.
# Replaces 29669 (0073f4a5ba, key 463192c0), cancelled because this fix changes the key.
#
# Prediction, registered before submission (2026-11-27 ~11:15 UTC): rung fixpoint
# (gen2 == gen3), rc 0, gen1/gen2 IR parity passes, gate-step MaxRSS within
# 1.15 x 39,038,888 K (job 28684).
# Model: the change adds one 4x16384 i64 table and a bounded edge walk that runs
# only when the checker asks whether a unit-named user type is visible; Madaros
# itself declares no user type named like a unit beside a glob import, so its
# own typing does not change, and the self-compile reaches the same fixed point
# shape as 28886 and 29568.
#
# Results, all recorded as Slurm step names:
#   r-head-tree-<40 hex>, r-head-key-<64 hex inputs key>
#   r-head-<commit>-gen1-<64 hex>
#   r-head-gate-rc<rc>-s<sec>
#   r-head-gen2-<64 hex>
#   r-head-gen3-<64 hex>
#   r-head-irparity-<line>
#   r-head-log-...
# MaxRSS per step comes from sacct. Full sha256 digests, because the gen2
# receipt gate (#3750 review) requires 64 hex.

set -uo pipefail
HEAD_C="94d93f125c921a110a9a959b5944395ae6db2376"
HEAD_TREE="662dd3e593f9c7dec858bd0c75e6f5f88d2f3a76"
SELF="$0"
REMOTE="https://github.com/Sounio-lang/sounio.git"
JOB="${SLURM_JOB_ID:-manual}"
WORK="/tmp/gen2-gate-3743e-$JOB"
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
arm() {  # arm <name> <commit> <tree>
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
