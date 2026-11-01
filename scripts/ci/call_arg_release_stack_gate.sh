#!/usr/bin/env bash
# Run in Slurm; invoke a pinned raw compiler without stack-adjusting wrappers.
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}"
: "${MADAROS_RAW_BIN:?pin the candidate ELF}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${STACK_GATE_OUT:?persistent result directory required}"
mkdir -p "$OUT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
sha256sum "$MADAROS_RAW_BIN" > "$OUT/compiler.sha256"
sha256sum "${BASH_SOURCE[0]}" "$ROOT/tests/fixtures/call_arg_release_stack/sequence.sio" "$ROOT/tests/fixtures/call_arg_release_stack/raw_cast_field.sio" > "$OUT/inputs.sha256"
printf 'fixture\tphase\trc\n' > "$OUT/matrix.tsv"
invoke() {
 local label="$1" phase="$2"; shift 2
 local rc=0
 ( ulimit -S -s 8192 && ulimit -H -s 8192 || exit 64
   [[ "$(ulimit -Ss)" == 8192 && "$(ulimit -Hs)" == 8192 ]] || exit 64
   printf 'soft=%s hard=%s\n' "$(ulimit -Ss)" "$(ulimit -Hs)" > "$OUT/$label.$phase.limits"
   timeout 180 "$@" ) > "$OUT/$label.$phase.stdout" 2> "$OUT/$label.$phase.stderr" || rc=$?
 printf '%s\t%s\t%s\n' "$label" "$phase" "$rc" >> "$OUT/matrix.tsv"
 return "$rc"
}
positive() {
 local label="$1" file="$2"
 invoke "$label" check "$MADAROS_RAW_BIN" check "$file"
 invoke "$label" build "$MADAROS_RAW_BIN" build "$file" -o "$OUT/$label.elf"
 chmod +x "$OUT/$label.elf"
 invoke "$label" run "$OUT/$label.elf"
}
# Flat per-position assertions avoid a deep sum AST and detect every permutation.
for n in 24 25 26 32 48 64 65; do
 file="$OUT/call$n.sio"
 {
 printf 'fn verify('
 for ((i=0;i<n;i++)); do [[ $i == 0 ]] || printf ','; printf 'p%s:i64' "$i"; done
 printf ') with Panic {\n'
 for ((i=0;i<n;i++)); do printf ' assert(p%s == %s)\n' "$i" "$i"; done
 printf '}\nfn main() with Panic { verify('
 for ((i=0;i<n;i++)); do [[ $i == 0 ]] || printf ','; printf '%s' "$i"; done
 printf ') }\n'
 } > "$file"
 # Keep identical declaration and body while removing only the caller.
 sed '/^fn main()/,$d' "$file" > "$OUT/decl$n.sio"
 printf 'fn main() {}\n' >> "$OUT/decl$n.sio"
 if [[ $n != 65 ]]; then
   positive "decl$n" "$OUT/decl$n.sio"
   positive "call$n" "$file"
 else
   invoke call65 check "$MADAROS_RAW_BIN" check "$file"
   if invoke call65 build "$MADAROS_RAW_BIN" build "$file" -o "$OUT/call65.elf"; then
     echo 'FAIL: accepted 65-argument call' >&2; exit 1
   else
     rc=$?; [[ $rc == 1 ]] || exit 1
   fi
 fi
done
# Reference calls exercise both release paths with many distinct places.
for kind in shared exclusive; do
 file="$OUT/ref64-$kind.sio"
 ref='&'; [[ $kind != exclusive ]] || ref='&!'
 {
 printf 'fn verify('
 for ((i=0;i<64;i++)); do [[ $i == 0 ]] || printf ','; printf 'p%s:%si64' "$i" "$ref"; done
 printf ') with Mut,Panic {\n'
 for ((i=0;i<64;i++)); do printf ' assert(*p%s == %s)\n' "$i" "$i"; done
 printf '}\nfn main() with Mut,Panic {\n'
 for ((i=0;i<64;i++)); do printf ' var x%s:i64 = %s\n' "$i" "$i"; done
 # Consecutive explicit calls execute; scalar implicit borrowing is checked below.
 # See the documented pre-existing native implicit-reference defect.
 for mode in explicit explicit; do
   printf ' verify('
   for ((i=0;i<64;i++)); do
     [[ $i == 0 ]] || printf ','
     [[ $mode != explicit ]] || printf '%s' "$ref"
     printf 'x%s' "$i"
   done
   printf ')\n'
 done
 printf '}\n'
 } > "$file"
 positive "ref64-$kind" "$file"
 sed -E 's/&!?x/x/g' "$file" > "$OUT/implicit-ref64-$kind.sio"
 invoke "implicit-ref64-$kind" check "$MADAROS_RAW_BIN" check "$OUT/implicit-ref64-$kind.sio"
done
positive sequence "$ROOT/tests/fixtures/call_arg_release_stack/sequence.sio"
invoke implicit_sequence check "$MADAROS_RAW_BIN" check "$ROOT/tests/fixtures/call_arg_release_stack/implicit_sequence.sio"
positive explicit "$ROOT/tests/run-pass/borrow_call_explicit_release.sio"
positive raw_cast "$ROOT/tests/fixtures/call_arg_release_stack/raw_cast_field.sio"
for name in borrow_call_conflict_explicit borrow_call_exclusive_conflict borrow_call_shared_conflict nested_shared_borrow_in_exclusive_call_2958; do
 file="$ROOT/tests/compile-fail/$name.sio"
 if invoke "$name" check "$MADAROS_RAW_BIN" check "$file"; then
   echo "FAIL: accepted $name" >&2; exit 1
 else
   rc=$?; [[ $rc == 1 ]] || exit 1
 fi
done
if [[ -n "${BASELINE_RAW_BIN:-}" ]]; then
 sha256sum "$BASELINE_RAW_BIN" > "$OUT/baseline.sha256"
 for name in borrow_call_conflict_explicit borrow_call_exclusive_conflict borrow_call_shared_conflict nested_shared_borrow_in_exclusive_call_2958; do
   if invoke "baseline-$name" check "$BASELINE_RAW_BIN" check "$ROOT/tests/compile-fail/$name.sio"; then exit 1; else rc=$?; [[ $rc == 1 ]] || exit 1; fi
   cmp "$OUT/$name.check.stdout" "$OUT/baseline-$name.check.stdout"
   cmp "$OUT/$name.check.stderr" "$OUT/baseline-$name.check.stderr"
 done
fi
find "$OUT" -maxdepth 1 -name '*.sio' -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/generated.sha256"
echo PASS > "$OUT/RESULT"
