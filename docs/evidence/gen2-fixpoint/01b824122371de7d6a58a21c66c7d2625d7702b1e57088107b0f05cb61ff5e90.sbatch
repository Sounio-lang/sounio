#!/usr/bin/env bash
#SBATCH --job-name=gen2-gate-l3-literal
#SBATCH --no-requeue
#SBATCH --partition=gpu-orangefs
#SBATCH --nodelist=gpuorangefs-r770-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/gen2-gate-l3-literal-%j.log
#
# Fixed-point gate for PR #3759 (generic struct literal: type arguments from the literal, E317/E318,
# in-scope same-name parameter) at 3006336295 on top of main 7d174f6c2e (#3755).
# Compiler input key 01b824122371de7d6a58a21c66c7d2625d7702b1e57088107b0f05cb61ff5e90,
# self-hosted tree 8485999cb7dff608eedcf9e48d1e5d5da2e3369c.
#
# Prediction, registered before submission (2026-11-28):
#   rung fixpoint reached (gen2 == gen3), gate rc 0, gen1/gen2 IR parity passes.
#   Gate-step MaxRSS <= 1.15 x 39,484,072 K (job 31701, #3755 gate), so it fits the R770's 60 GiB.
#   Wall <= 1.3 x 10,767 s on the R740 (R770 speed unknown; a longer run is not a failure by itself).
# Model: the patch touches parse_struct_literal (turbofish kept in Expr.type_args) and the checker's
#   struct-literal path. The compiler source check on the earlier round showed 0 E317 and 0 E318 on
#   self-hosted/compiler/main.sio; the final patch adds the in-scope same-name rule, which only turns a
#   would-be E317 into the type the specializer already used (heap_vec 7/7, heap_realloc 2/2 verified).
#   So gen1 compiles the patched source without new errors and gen2 reproduces gen1 byte for byte.
# Falsifier: gen2 != gen3, gate rc != 0, OOM on 60 GiB, an E317/E318 on the compiler's own source,
#   or a build failure of gen1. An OOM is a receipt too (rerun on the R740, 512 GiB).
# Results as Slurm step names: r-head-tree-<40hex>, r-head-key-<64hex>, r-head-<c>-gen1-<64hex>,
#   r-head-gate-rc<rc>-s<sec>, r-head-gen2-<64hex>, r-head-gen3-<64hex>, r-head-irparity-..., r-head-log-...
set -uo pipefail
HEAD_C="3006336295"
HEAD_FULL="30063362956e7ab1a1605f1e3936581d619ad60b"
HEAD_TREE="8485999cb7dff608eedcf9e48d1e5d5da2e3369c"
SELF="$0"
REMOTE="https://github.com/Sounio-lang/sounio.git"
JOB="${SLURM_JOB_ID:-manual}"
WORK="/tmp/gen2-gate-l3-literal-$JOB"
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
  local n="$1" c="$2" d="$WORK/$1"
  git init -q "$d" || die "init $n"
  git -C "$d" -c http.sslCAInfo="$CA" fetch -q --depth 1 --no-tags "$REMOTE" "$c" > /dev/null 2>&1 || die "fetch $n"
  git -C "$d" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD || die "checkout $n"
  [[ "$(git -C "$d" rev-parse HEAD)" == "$c" ]] || die "HEAD $n"
  local got_tree; got_tree="$(git -C "$d" rev-parse HEAD:self-hosted)"
  [[ "$got_tree" == "$HEAD_TREE" ]] || die "tree $n"
  rec "r-$n-tree-$got_tree"
  local ik; ik="$( (cd "$d" && bash scripts/ci/gen2_receipt_gate.sh 2>/dev/null) | sed -n "s/^gen2 inputs key at HEAD: //p")"
  rec "r-$n-key-${ik:-none}"
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
arm head "$HEAD_FULL"
rec "r-done"
exit 0
__CA_BUNDLE__
