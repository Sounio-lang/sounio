#!/usr/bin/env bash
# Paired diagnostic only; no compiler source edits/bootstrap.
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}" "${CXX:?}" "${NODE_BIN:?}"
: "${SEED9:?}" "${SEEDCC:?}" "${RAW04:?}" "${SDK_STDLIB:?}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
F="$ROOT/tests/fixtures/wave4_concrete_tuple_lifetime"
OUT="${1:?fresh results path}"
mkdir "$OUT"
printf '%s  %s\n' \
 9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a "$SEED9" \
 ccfaec48d1bae176091ef1f3401c2d863c1f05284390f3c50ca9d66e94c7114f "$SEEDCC" \
 a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333 "$RAW04" > "$OUT/COMPILERS.sha256"
sha256sum -c "$OUT/COMPILERS.sha256" > "$OUT/COMPILERS.before.verify"
find "$F" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCES.sha256"
sha256sum "$CXX" "$NODE_BIN" "$ROOT/tests/fixtures/wave4_variance_shadow/admission.mjs" > "$OUT/TOOLS.sha256"
"$NODE_BIN" "$F/guards.mjs" seal "$SDK_STDLIB" "$OUT/SDK.json"
sha256sum "$OUT/SDK.json" > "$OUT/SDK_MANIFEST.sha256"
finish(){
 rc=$?;trap - EXIT;set +e
 sha256sum -c "$OUT/COMPILERS.sha256" > "$OUT/COMPILERS.after.verify" 2>&1 || rc=1
 sha256sum -c "$OUT/SOURCES.sha256" > "$OUT/SOURCES.after.verify" 2>&1 || rc=1
 sha256sum -c "$OUT/TOOLS.sha256" > "$OUT/TOOLS.after.verify" 2>&1 || rc=1
 sha256sum -c "$OUT/SDK_MANIFEST.sha256" > "$OUT/SDK_MANIFEST.after.verify" 2>&1 || rc=1
 "$NODE_BIN" "$F/guards.mjs" verify "$SDK_STDLIB" "$OUT/SDK.json" > "$OUT/SDK.after.verify" 2>&1 || rc=1
 printf '%s\n' "$rc" > "$OUT/STATUS"
 exit "$rc"
}
trap finish EXIT
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -fno-omit-frame-pointer \
 "$F/reference.cpp" -o "$OUT/reference" > "$OUT/reference.build.stdout" 2> "$OUT/reference.build.stderr"
"$OUT/reference" --controls > "$OUT/reference.controls.stdout" 2> "$OUT/reference.controls.stderr"
"$OUT/reference" --expected > "$OUT/expected.stdout"
record(){
 local prefix=$1;seconds=$2;shift 2
 local rc=0
 timeout --kill-after=5 "$seconds" "$@" > "$prefix.stdout" 2> "$prefix.stderr" || rc=$?
 printf '%s\n' "$rc" > "$prefix.STATUS"
}
diagnostic(){
 local prefix=$1 rc=0
 grep -Ei 'error\[E|(^|[[:space:]])error:|parse error|undefined identifier|unknown variable|field initializer mismatch|would have dropped|cannot compile an assignment|native.*refus|AST closure incomplete|IR_INVALID_REG' "$prefix.stdout" "$prefix.stderr" > "$prefix.EXPLICIT_ERRORS" || rc=$?
 if [[ "$rc" == 0 ]];then return 1;fi
 [[ "$rc" == 1 ]]
}
export SOUNIO_STDLIB_PATH="$SDK_STDLIB" SOUNIO_LOOP_CHECKPOINT_ELISION=0
for tag in seed9 seedcc raw0 raw1;do
 mkdir "$OUT/$tag"
 compiler="$SEED9";mode=0
 case "$tag" in seedcc) compiler="$SEEDCC";;raw0) compiler="$RAW04";;raw1) compiler="$RAW04";mode=1;;esac
 for name in flat nested assertion_control;do
  prefix="$OUT/$tag/$name"
  file="$F/$name.sio"
  if [[ "$tag" == raw* ]];then
   record "$prefix.check" 120 env SOUNIO_LOOP_REGIONS="$mode" "$compiler" check "$file"
   record "$prefix.compile" 120 env SOUNIO_LOOP_REGIONS="$mode" "$compiler" compile "$file" -o "$prefix.elf"
  else
   printf '%s\n' 'parser/typecheck integrated into producer compile; no separate check CLI' > "$prefix.CHECK_MODE"
   record "$prefix.compile" 120 "$compiler" "$file" "$prefix.elf"
  fi
  admissible=yes
  [[ "$(cat "$prefix.compile.STATUS")" == 0 ]] || admissible=no
  diagnostic "$prefix.compile" || admissible=no
  if [[ "$tag" == raw* ]];then [[ "$(cat "$prefix.check.STATUS")" == 0 ]] && diagnostic "$prefix.check" || admissible=no;fi
  if [[ "$admissible" == yes && -s "$prefix.elf" && "$(od -An -tx1 -N4 "$prefix.elf" | tr -d ' \n')" == 7f454c46 ]];then
   chmod u+x "$prefix.elf"
   sha256sum "$prefix.elf" > "$prefix.ELF.before.sha256"
   record "$prefix.native" 30 "$prefix.elf"
   sha256sum -c "$prefix.ELF.before.sha256" > "$prefix.ELF.after.verify"
   record "$prefix.machine" 10 "$OUT/reference" --scan "$prefix.elf"
   if [[ "$name" != assertion_control ]];then
    record "$prefix.oracle" 10 "$OUT/reference" --check "$prefix.native.stdout"
   fi
  else
   printf '%s\n' 'native not run: compile/check/diagnostic/ELF admission failed' > "$prefix.NATIVE_NOT_RUN"
  fi
 done
done
printf '%s\n' 'AUDIT_EXECUTED_NOT_NUMERICAL_ACCEPTANCE: inspect each compile/native/oracle status' > "$OUT/AUDIT_SCOPE"
