#!/usr/bin/env bash
# REVIEW TEMPLATE. Only root may instantiate/authorize/execute it. No sbatch call.
set -euo pipefail
ACTION="${1:?use --review|--stage|--preflight|--build}"
PLAN="${2:?supplied actual plan JSON required}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HELPER="$REPO/tests/fixtures/wave4_root06_recipe/metadata.mjs"
NODE_BOOTSTRAP="${ROOT06_NODE_BIN:?explicit pinned Node bootstrap required}"
printf '%s  %s\n' "${ROOT06_NODE_SHA256:?qualified Node hash required}" "$NODE_BOOTSTRAP" | sha256sum -c - >&2
if [[ "$ACTION" == --review ]];then
 exec "$NODE_BOOTSTRAP" "$HELPER" verify "$PLAN" review
fi
: "${SLURM_JOB_ID:?root-authorized execution requires Slurm}"
"$NODE_BOOTSTRAP" "$HELPER" authorize "$PLAN"
# Safe NUL field assignment; no eval, shell interpolation or argument splitting.
ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/root06-env.XXXXXXXX")"
"$NODE_BOOTSTRAP" "$HELPER" env "$PLAN" > "$ENV_FILE"
while IFS= read -r -d '' key && IFS= read -r -d '' value;do
 case "$key" in
 R|S_GIT|S|D|PREFLIGHT|CACHE|LOCK|OLOCK|SEED|NODE|BASH_BIN|GIT|TAR|TIME|TIMEOUT|SDK_ENV|SDK_ENV_R|ARCHIVE|CAPACITY|WORKER_PATCH|PREFLIGHT_RECIPE|PRODUCER|BUILD_SECONDS|PREFLIGHT_SECONDS|CXX_PIN|MAGIC_DB)
  printf -v "$key" '%s' "$value";;
 *) exit 64;;
 esac
done < "$ENV_FILE"
[[ "$NODE" == "$NODE_BOOTSTRAP" ]] || exit 64
case "$ACTION" in
 --stage) [[ ! -e "$R" && ! -L "$R" ]]; mkdir "$R";PHASE=stage;OUT="$R/staging";;
 --preflight) [[ "$(cat "$R/STAGING_STATUS")" == 0 && ! -e "$PREFLIGHT" ]];PHASE=preflight;OUT="$PREFLIGHT";;
 --build) [[ -s "$R/PREFLIGHT_READY" && "$(cat "$PREFLIGHT/STATUS")" == 0 && ! -e "$D" && ! -e "$R/RAW_READY" ]];PHASE=build;OUT="$D";;
 *) exit 64;;
esac
mkdir "$OUT"
# Scope is only this private run root. Compiler lock is distinct from recipe.lock.
STATUS_FILE="$OUT/STATUS"
[[ "$PHASE" != build ]] || STATUS_FILE="$D/BUILD_STATUS"
[[ "$PHASE" != stage ]] || STATUS_FILE="$R/STAGING_STATUS"
finish(){
 rc=$?
 trap - EXIT INT TERM
 set +e
 "$NODE" "$HELPER" verify "$PLAN" final > "$R/final-$PHASE.json" 2> "$OUT/FINAL_GUARD.stderr"
 guard_rc=$?
 if [[ "$guard_rc" != 0 ]];then rc=1;fi
 if [[ "$PHASE" == build && "$rc" == 0 ]];then
  [[ -s "$D/madaros" && "$(od -An -tx1 -N4 "$D/madaros" | tr -d ' \n')" == 7f454c46 ]] || rc=1
  [[ "$rc" != 0 ]] || sha256sum "$D/madaros" > "$D/COMPILER.sha256" || rc=1
 fi
 printf '%s\n' "$rc" > "$STATUS_FILE" || rc=1
 date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/ENDED_UTC" || rc=1
 # Readiness is the LAST operation, only after terminal EXIT guards/status.
 # On failure retain all artifacts; never publish or rename an ELF as Ready.
 if [[ "$rc" == 0 && "$PHASE" != stage ]];then
  "$NODE" "$HELPER" publish "$PLAN" "$PHASE" > "$OUT/PUBLICATION.stdout" 2> "$OUT/PUBLICATION.stderr" || rc=1
 fi
 if [[ "$rc" != 0 ]];then printf '%s\n' "$rc" > "$STATUS_FILE";fi
 exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cp "$ENV_FILE" "$OUT/PLAN.env0"
exec 9>"$OLOCK"
flock -n 9
# Caller-supplied env is qualified, pinned and captured through complete SDK manifests.
ORIGINAL_R="$R";R="$SDK_ENV_R";source "$SDK_ENV";R="$ORIGINAL_R"
export CXX="$CXX_PIN" MAGIC="$MAGIC_DB"
ulimit -s 524288
ulimit -c 0
CHECK_PHASE="$PHASE"
[[ "$PHASE" != stage ]] || CHECK_PHASE=review
"$NODE" "$HELPER" verify "$PLAN" "$CHECK_PHASE" > "$OUT/PLAN.before.json"
if [[ "$PHASE" == stage ]];then
 mkdir "$S_GIT"
 [[ "$("$GIT" get-tar-commit-id < "$ARCHIVE")" == "$("$NODE" -p 'JSON.parse(require("fs").readFileSync(process.argv[1])).head' "$PLAN")" ]]
 "$TAR" -xf "$ARCHIVE" -C "$S_GIT"
 "$NODE" "$HELPER" verify "$PLAN" pristine > "$OUT/PRISTINE.json"
 cp -a "$S_GIT" "$S"
 "$NODE" "$HELPER" verify "$PLAN" copy > "$OUT/COPY_FIDELITY.json"
 mkdir "$R/seed-private" "$CACHE"
 cp -p "$PRODUCER" "$SEED"
 [[ -x "$SEED" ]]
 (
  cd "$S"
  "$GIT" apply --check "$CAPACITY"
  "$GIT" apply "$CAPACITY"
  "$GIT" apply --check "$WORKER_PATCH"
  "$GIT" apply "$WORKER_PATCH"
 ) > "$OUT/OVERLAYS.stdout" 2> "$OUT/OVERLAYS.stderr"
 "$NODE" "$HELPER" verify "$PLAN" overlay > "$OUT/OVERLAYS.json"
 exit 0
fi
"$NODE" "$HELPER" verify "$PLAN" "$PHASE" > "$OUT/SNAPSHOT.before.json"
export SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_BUILD_LOCK="$LOCK"
export SOUNIO_MADAROS_CACHE="$CACHE" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export SOUNIO_LOOP_REGIONS=0 SOUC_BIN="$SEED" SOUNIO_SOUC_BIN="$SEED"
export ROOT06_RUN_ROOT="$R" ROOT06_SOURCE="$S" ROOT06_PREFLIGHT_OUTPUT="$PREFLIGHT"
set +e
if [[ "$PHASE" == preflight ]];then
 "$TIME" -v "$TIMEOUT" -k 30 "$PREFLIGHT_SECONDS" "$BASH_BIN" "$PREFLIGHT_RECIPE" > "$OUT/work.stdout" 2> "$OUT/work.stderr"
 work_rc=$?
else
 (
  cd "$S"
  "$TIME" -v "$TIMEOUT" -k 30 "$BUILD_SECONDS" "$BASH_BIN" scripts/ci/build_modular_madaros.sh "$D/madaros"
 ) > "$OUT/work.stdout" 2> "$OUT/work.stderr"
 work_rc=$?
fi
set -e
printf '%s\n' "$work_rc" > "$OUT/WORK_STATUS"
[[ "$work_rc" == 0 ]]
set +e
grep -Ei 'error\[E|(^|[[:space:]])error:|parse error|undefined identifier|unknown variable|field initializer mismatch|would have dropped|cannot compile an assignment|native.*refus|emission.*refus|closure.*refus|AST closure incomplete|IR_INVALID_REG|missing_lowered_body' "$OUT/work.stdout" "$OUT/work.stderr" > "$OUT/EXPLICIT_ERRORS"
diagnostic_rc=$?
set -e
[[ "$diagnostic_rc" == 1 ]] # grep0=diagnostic; grep2/read error is also refusal.
if [[ "$PHASE" == preflight ]];then "$NODE" "$HELPER" preflight-result "$PLAN";fi
# EXIT trap alone publishes readiness after all whole-graph final guards.
