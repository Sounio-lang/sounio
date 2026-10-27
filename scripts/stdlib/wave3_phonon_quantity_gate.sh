#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
[[ $# == 1 && "$1" == /* && "$1" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo "fresh absolute output required" >&2;exit 2; }
OUT="$(realpath -m -- "$1")"
[[ ! -e "$OUT" ]] || { echo "refuse existing output" >&2;exit 2; }
if [[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]];then AUTHORITY="slurm:$SLURM_JOB_ID"
elif [[ "${GITHUB_ACTIONS:-}" == true && "${GITHUB_RUN_ID:-}" =~ ^[1-9][0-9]*$ && "${GITHUB_JOB:-}" =~ ^[A-Za-z0-9_.-]+$ ]];then AUTHORITY="github-actions:$GITHUB_RUN_ID:$GITHUB_JOB"
else echo "scheduled Slurm or identified official CI required" >&2;exit 2;fi
: "${MADAROS_RAW_BIN:?qualified raw compiler required}"
[[ "$MADAROS_RAW_BIN" == /* ]] || { echo "raw compiler must be absolute" >&2;exit 2; }
RAW="$(realpath -e "$MADAROS_RAW_BIN")"
[[ "$RAW" == /* && -x "$RAW" && "$(od -An -tx1 -N4 "$RAW"|tr -d ' \n')" == 7f454c46 ]] || exit 2
NODE="${NODE:-node}"
CXX="${CXX:-g++}"
WRAPPER="$ROOT/scripts/ci/souc-native-wrapper.sh"
BOUNDER="$ROOT/scripts/stdlib/bounded_command.mjs"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_SOUC_BIN="$RAW" SOUC_NATIVE_BIN="$RAW" SOUNIO_SOUC_RAW_MODE=modular
ulimit -s 524288
ulimit -c 0
finish(){
 local r=$? changed=0
 trap - EXIT
 for m in SOURCES TOOLS EXECUTABLES ARTIFACTS;do
  if [[ -s "$OUT/$m.sha256" ]];then sha256sum -c "$OUT/$m.sha256" > "$OUT/$m.final.verify" 2>&1 || changed=1;fi
 done
 [[ "$changed" == 0 ]] || r=1
 printf '%s\n' "$r" > "$OUT/STATUS"
 if [[ "$r" == 0 ]];then echo WAVE3_PHONON_QUANTITY_GATE_OK;fi
 exit "$r"
}
trap finish EXIT
printf '%s\n' "$AUTHORITY" > "$OUT/EXECUTION_AUTHORITY"
cat > "$OUT/PLAN.tsv" <<'PLAN'
tests/stdlib/physics/test_phonon_quantity.sio	-
tests/stdlib/physics/test_pbpk_phonon_q.sio	-
tests/stdlib/physics/test_phonon_quantity_parity.sio	PHONON_SI_PARITY_PASS
tests/stdlib/wave3-phonon-quantity/checked.sio	PHONON_QUANTITY_CHECKED_OK
PLAN
(
 cd "$ROOT"
 find stdlib tests/stdlib/wave3-phonon-quantity -type f -print0
 while IFS=$'\t' read -r f marker;do printf '%s\0' "$f";done < "$OUT/PLAN.tsv"
 printf '%s\0' scripts/stdlib/wave3_phonon_quantity_gate.sh scripts/stdlib/bounded_command.mjs scripts/ci/souc-native-wrapper.sh tests/reference/wave3_phonon_quantity_reference.cpp
) | sort -zu | while IFS= read -r -d '' f;do sha256sum "$ROOT/$f";done > "$OUT/SOURCES.sha256"
sha256sum "$RAW" "$(command -v "$NODE")" "$(command -v "$CXX")" > "$OUT/TOOLS.sha256"
"$CXX" --version > "$OUT/CXX_VERSION"
printf 'item\tphase\tstatus\n' > "$OUT/RESULTS.tsv"
: > "$OUT/EXECUTABLES.sha256"
phase(){
 local name="$1" kind="$2" seconds="$3" bytes="$4" r=0
 shift 4
 "$NODE" "$BOUNDER" "$seconds" "$bytes" -- "$@" > "$OUT/$name.$kind.log" 2>&1 || r=$?
 printf '%s\t%s\t%s\n' "$name" "$kind" "$r" >> "$OUT/RESULTS.tsv"
 return "$r"
}
failed=0
while IFS=$'\t' read -r f marker;do
 name="$(basename "$f" .sio)"
 if ! phase "$name" check 300 4194304 bash "$WRAPPER" check "$ROOT/$f";then failed=1;continue;fi
 if ! grep -Fxq 'check: OK' "$OUT/$name.check.log" || grep -Fq 'warning[W044]' "$OUT/$name.check.log";then failed=1;continue;fi
 if ! phase "$name" compile 300 4194304 bash "$WRAPPER" compile "$ROOT/$f" -o "$OUT/$name.elf";then failed=1;continue;fi
 if grep -Fq 'warning[W044]' "$OUT/$name.compile.log";then failed=1;continue;fi
 if [[ ! -s "$OUT/$name.elf" || "$(od -An -tx1 -N4 "$OUT/$name.elf"|tr -d ' \n')" != 7f454c46 ]];then failed=1;continue;fi
 chmod +x "$OUT/$name.elf"
 sha256sum "$OUT/$name.elf" >> "$OUT/EXECUTABLES.sha256"
 if ! phase "$name" native 600 1048576 "$OUT/$name.elf";then failed=1;continue;fi
 if [[ "$marker" != - && "$(grep -Fxc "$marker" "$OUT/$name.native.log"||true)" != 1 ]];then failed=1;fi
done < "$OUT/PLAN.tsv"
mkdir "$OUT/private"
for f in test_phonon_quantity test_pbpk_phonon_q;do
 original="$ROOT/tests/stdlib/wave3-phonon-quantity/legacy_callers/$f.sio.in"
 derived="$OUT/private/legacy_$f.sio"
 cp "$original" "$derived";cmp "$original" "$derived"
sha256sum "$derived" >> "$OUT/SOURCES.sha256"
 r=0;phase "legacy_$f" check-expected-1 300 4194304 bash "$WRAPPER" check "$derived" || r=$?
 if [[ "$r" != 1 ]] || ! grep -Fq 'error[E137]' "$OUT/legacy_$f.check-expected-1.log";then failed=1;fi
done
original="$ROOT/tests/stdlib/wave3-phonon-quantity/linear_owner_reuse.sio.in"
derived="$OUT/private/linear_owner_reuse.sio"
cp "$original" "$derived";cmp "$original" "$derived"
sha256sum "$derived" >> "$OUT/SOURCES.sha256"
for kind in check compile;do
 r=0
 if [[ "$kind" == check ]];then args=(check "$derived");else args=(compile "$derived" -o "$OUT/private/forbidden.elf");fi
 phase owner-reuse "$kind-expected-1" 300 4194304 bash "$WRAPPER" "${args[@]}" || r=$?
 if [[ "$r" != 1 ]] || ! grep -Fq 'error[E039]' "$OUT/owner-reuse.$kind-expected-1.log";then failed=1;fi
done
if phase reference compile 180 1048576 "$CXX" -std=c++23 -O2 -ffp-contract=off -Wall -Wextra -Werror "$ROOT/tests/reference/wave3_phonon_quantity_reference.cpp" -o "$OUT/reference";then
 sha256sum "$OUT/reference" >> "$OUT/EXECUTABLES.sha256"
 if ! phase reference native 30 1048576 "$OUT/reference" "$OUT/checked.native.log";then failed=1
 elif ! grep -Fxq 'PHONON_QUANTITY_CPP_CONCORDANCE_OK rows=16 negatives=20 relative=1e-11' "$OUT/reference.native.log";then failed=1;fi
 if ! phase oracle mutate 30 1048576 "$OUT/reference" --wrong-diagonal "$OUT/checked.native.log" "$OUT/wrong-diagonal.log";then failed=1
 else
  r=0;phase oracle reject-wrong-diagonal-expected-1 30 1048576 "$OUT/reference" "$OUT/wrong-diagonal.log" || r=$?
  if [[ "$r" != 1 ]] || ! grep -Fq 'phonon Quantity numerical mismatch' "$OUT/oracle.reject-wrong-diagonal-expected-1.log";then failed=1;fi
 fi
else failed=1;fi
find "$OUT" -maxdepth 1 -name '*.log' -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/ARTIFACTS.sha256"
sha256sum "$OUT/PLAN.tsv" "$OUT/RESULTS.tsv" "$OUT/EXECUTION_AUTHORITY" "$OUT/CXX_VERSION" >> "$OUT/ARTIFACTS.sha256"
[[ "$failed" == 0 ]]
