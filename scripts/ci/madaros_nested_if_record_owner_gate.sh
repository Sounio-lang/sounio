#!/usr/bin/env bash
# Compile and execute typed-field witnesses with the selected current-source ELF.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
raw=${MADAROS_RAW_BIN:?set MADAROS_RAW_BIN to the current-source compiler}
[[ -x "$raw" ]] || { echo 'compiler is not executable' >&2; exit 65; }
if [[ -n ${SOUNIO_MADAROS_RECORD_OWNER_GATE_DIR:-} ]]; then
 work=$SOUNIO_MADAROS_RECORD_OWNER_GATE_DIR
 [[ ! -e "$work" ]] || { echo "refusing existing output: $work" >&2; exit 65; }
 mkdir -p "$work"
else
 work=$(mktemp -d /tmp/sounio-record-owner.XXXXXX)
fi
trap 'gate_rc=$?; printf "%s\n" "$gate_rc" > "$work/STATUS"' EXIT
ulimit -s 524288
export SOUNIO_STDLIB_PATH="$repo/stdlib"
cd "$repo"
sources=(tests/run-pass/nested_if_record_owner.sio tests/run-pass/nested_if_tuple_record_owner.sio)
sha256sum "$raw" > "$work/COMPILER.sha256"
negative=tests/compile-fail/nested_if_incompatible_record_owner.sio
sha256sum "${sources[@]}" "$negative" > "$work/INPUTS.sha256"
printf 'kind\tsource\texit_status\n' > "$work/RESULTS.tsv"
for src in "${sources[@]}"; do
 name=$(basename "$src" .sio)
 expected=$(sed -nE 's|^//@ expect-stdout(-contains)?: ||p' "$src")
 [[ -n "$expected" ]] || { echo "missing completion marker: $src" >&2; exit 65; }
 rc=0
 timeout 240 "$raw" build "$src" --target native -o "$work/$name.elf" > "$work/$name.compile.stdout" 2> "$work/$name.compile.stderr" || rc=$?
 printf 'compile\t%s\t%s\n' "$src" "$rc" >> "$work/RESULTS.tsv"
 [[ $rc == 0 ]] || { cat "$work/$name.compile.stdout" "$work/$name.compile.stderr" >&2; exit 1; }
 chmod +x "$work/$name.elf"
 sha256sum "$work/$name.elf" > "$work/$name.ELF.sha256"
 rc=0
 /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$work/$name.resource" timeout 60 "$work/$name.elf" > "$work/$name.stdout" 2> "$work/$name.stderr" || rc=$?
 printf 'native\t%s\t%s\n' "$src" "$rc" >> "$work/RESULTS.tsv"
 [[ $rc == 0 ]] || { cat "$work/$name.stdout" "$work/$name.stderr" >&2; exit 1; }
 while IFS= read -r marker; do
  [[ -z "$marker" ]] || grep -Fq -- "$marker" "$work/$name.stdout" || { echo "missing marker: $marker" >&2; exit 1; }
 done <<< "$expected"
 printf 'stdout-marker\t%s\t0\n' "$src" >> "$work/RESULTS.tsv"
done
rc=0
timeout 120 "$raw" check "$negative" > "$work/incompatible.check.stdout" 2> "$work/incompatible.check.stderr" || rc=$?
if [[ $rc == 1 ]] && grep -Fqi 'incompatible types' "$work/incompatible.check.stdout" "$work/incompatible.check.stderr"; then
 printf 'negative\t%s:expected=1:actual=1\t0\n' "$negative" >> "$work/RESULTS.tsv"
else
 printf 'negative\t%s:expected=1:actual=%s\t1\n' "$negative" "$rc" >> "$work/RESULTS.tsv"
 exit 1
fi
sha256sum -c "$work/INPUTS.sha256" > "$work/INPUTS.verify.txt"
sha256sum -c "$work/COMPILER.sha256" > "$work/COMPILER.verify.txt"
printf '0\n' > "$work/STATUS"
echo '[madaros-record-owner] PASS: nested record and tuple owners preserve typed field projection'
