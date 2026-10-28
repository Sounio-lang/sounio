#!/usr/bin/env bash
# Run only inside Slurm against a prepared, immutable candidate source tree.
# Explicit raw ELF arguments preserve compiler custody; no compiler bootstrap.
set -euo pipefail
if test "$#" != 5;then
 echo "usage: runtime_bytes_gate.sh RAW SOURCE ORACLE PLATFORM_LIBRARY FRESH_OUTPUT" >&2
 exit 64
fi
if test -z "${SLURM_JOB_ID:-}";then echo "Slurm job required" >&2;exit 64;fi
raw=$(realpath "$1")
source_root=$(realpath "$2")
oracle=$(realpath "$3")
platform=$(realpath "$4")
out=$(realpath -m "$5")
test -x "$raw";test -x "$oracle";test -s "$platform";test ! -e "$out"
mkdir -p "$out"
export SOUNIO_STDLIB_PATH="$source_root/stdlib"
export LD_LIBRARY_PATH="$(dirname "$platform"):${LD_LIBRARY_PATH:-}"
ulimit -s524288
ulimit -c0
cd "$source_root"
sha256sum "$raw" "$oracle" "$platform" > "$out/ARTIFACTS.sha256"
find stdlib tests/stdlib/runtime_regression tests/fixtures/runtime_bytes_contract -type f -print0|sort -z|xargs -0 sha256sum > "$out/SOURCE.sha256"
sha256sum tests/fixtures/fmri/tiny_real_slice_nifti1.nii tests/fixtures/fmri/tiny_real_motion.tsv > "$out/ORIGINAL_DATA.sha256"
bad=0
last_status=0
printf 'phase\texpected\tactual\n' > "$out/RESULTS.tsv"
phase(){
 local name="$1" expected="$2";shift 2
 last_status=0
 /usr/bin/time -v "$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || last_status=$?
 printf '%s\t%s\t%s\n' "$name" "$expected" "$last_status" >> "$out/RESULTS.tsv"
 if test "$last_status" != "$expected";then bad=1;fi
}
phase generate-fixtures 0 "$oracle" --generate "$out/fixtures"
test "$last_status" = 0
find "$out/fixtures" -type f -print0|sort -z|xargs -0 sha256sum > "$out/GENERATED_DATA.sha256"
for family in binary literal text dynamic_slice;do
 name="runtime_${family}_as_bytes"
 if test "$family" = dynamic_slice;then name=runtime_dynamic_slice;fi
 original="tests/fixtures/runtime_bytes_contract/original/$name.sio"
 phase "original-$family-check" 1 timeout 120 "$raw" check "$original"
 phase "original-$family-compile" 1 timeout 180 "$raw" compile "$original" -o "$out/original-$family.elf"
 test ! -e "$out/original-$family.elf" || bad=1
done
for library in utf8 utf8_bytes mod lib;do
 phase "library-$library-check" 0 timeout 180 "$raw" check "stdlib/text/$library.sio"
done
declare -A ready
for family in binary literal text dynamic_slice owned last_use;do
 name="tests/stdlib/runtime_regression/runtime_${family}_as_bytes.sio"
 if test "$family" = dynamic_slice;then name=tests/stdlib/runtime_regression/runtime_dynamic_slice.sio;fi
 if test "$family" = owned;then name=tests/fixtures/runtime_bytes_contract/utf8_bytes_owned.sio;fi
 if test "$family" = last_use;then name=tests/fixtures/runtime_bytes_contract/borrow_after_last_use.sio;fi
 phase "$family-check" 0 timeout 180 "$raw" check "$name"
 phase "$family-compile" 0 timeout 300 "$raw" compile "$name" -o "$out/$family.elf"
 ready[$family]=0
 if test "$last_status" = 0;then
  chmod +x "$out/$family.elf"
  sha256sum "$out/$family.elf" > "$out/$family-ELF.sha256"
  ready[$family]=1
 fi
done
run_bytes(){
 local label="$1" family="$2" kind="$3" expected="$4";shift 4
 if test "${ready[$family]}" != 1;then bad=1;return;fi
 phase "$label-native" 0 timeout 30 "$out/$family.elf" "$@"
 phase "$label-oracle" 0 "$oracle" --kind "$kind" --source "$expected" --stdout "$out/$label-native.stdout"
}
run_bytes binary-default binary binary "$source_root/tests/fixtures/fmri/tiny_real_slice_nifti1.nii"
run_bytes literal-default literal literal -
run_bytes text-default text text "$source_root/tests/fixtures/fmri/tiny_real_motion.tsv"
run_bytes slice-default dynamic_slice slice "$source_root/tests/fixtures/fmri/tiny_real_motion.tsv"
run_bytes binary-all-bytes binary binary "$out/fixtures/all-bytes.bin" "$out/fixtures/all-bytes.bin"
run_bytes binary-nul binary binary "$out/fixtures/utf8-nul.bin" "$out/fixtures/utf8-nul.bin"
run_bytes binary-empty binary binary "$out/fixtures/empty.bin" "$out/fixtures/empty.bin"
run_bytes text-nul text text "$out/fixtures/utf8-nul.bin" "$out/fixtures/utf8-nul.bin"
run_bytes text-scalar text text "$out/fixtures/split-scalar.bin" "$out/fixtures/split-scalar.bin"
run_bytes text-empty text text "$out/fixtures/empty.bin" "$out/fixtures/empty.bin"
run_bytes slice-all-bytes dynamic_slice slice "$out/fixtures/all-bytes.bin" "$out/fixtures/all-bytes.bin"
run_bytes slice-inside-scalar dynamic_slice slice "$out/fixtures/split-scalar.bin" "$out/fixtures/split-scalar.bin"
run_bytes slice-empty dynamic_slice slice "$out/fixtures/empty.bin" "$out/fixtures/empty.bin"
run_bytes owned-native owned utf8-owned -
if test "${ready[last_use]}" = 1;then
 phase last-use-native 0 timeout 30 "$out/last_use.elf"
 grep -Fx RUNTIME_BYTES_LAST_USE_OK "$out/last-use-native.stdout" > "$out/LAST_USE.marker" || bad=1
fi
for family in binary literal text dynamic_slice;do
 if test "${ready[$family]}" = 1;then
  phase "$family-extra-argc" 64 timeout 30 "$out/$family.elf" extra argument
 fi
done
if test "${ready[text]}" = 1;then
 phase text-invalid-utf8 1 timeout 30 "$out/text.elf" "$out/fixtures/invalid-utf8.bin"
 grep -Fx 'RUNTIME_PROBE_TEXT_ERROR 5' "$out/text-invalid-utf8.stdout" > "$out/INVALID_UTF8.marker" || bad=1
 if grep -Fq RUNTIME_PROBE_TEXT_AFTER "$out/text-invalid-utf8.stdout";then bad=1;fi
fi
if test "${ready[binary]}" = 1;then
 phase binary-missing-file 1 timeout 30 "$out/binary.elf" "$out/fixtures/missing-file.bin"
 grep -Fx 'RUNTIME_PROBE_ERROR 2' "$out/binary-missing-file.stdout" > "$out/MISSING_FILE.marker" || bad=1
 if grep -Fq RUNTIME_PROBE_BINARY_AFTER "$out/binary-missing-file.stdout";then bad=1;fi
fi
for name in borrow_close borrow_mutate borrow_forward_close;do
 path="tests/fixtures/runtime_bytes_contract/$name.sio"
 phase "$name-check" 1 timeout 180 "$raw" check "$path"
 grep -F -e 'error[E038]' -e 'error[E091]' -e 'error[E259]' "$out/$name-check.stdout" > "$out/$name-DIAGNOSTIC" || bad=1
 phase "$name-compile" 1 timeout 300 "$raw" compile "$path" -o "$out/$name.elf"
 test ! -e "$out/$name.elf" || bad=1
 if grep -Fq 'error[E264]' "$out/$name-check.stdout";then bad=1;fi
done
phase oracle-missing-argc 64 "$oracle"
phase oracle-unknown-kind 64 "$oracle" --kind unknown --source - --stdout "$out/binary-default-native.stdout"
phase oracle-wrong-kind 1 "$oracle" --kind text --source "$source_root/tests/fixtures/fmri/tiny_real_slice_nifti1.nii" --stdout "$out/binary-default-native.stdout"
for change in wrong-byte wrong-index missing-row duplicate-row wrong-length missing-after;do
 phase "mutate-$change" 0 "$oracle" --mutate "$change" "$out/binary-default-native.stdout" "$out/control-$change.stdout"
 phase "oracle-$change" 1 "$oracle" --kind binary --source "$source_root/tests/fixtures/fmri/tiny_real_slice_nifti1.nii" --stdout "$out/control-$change.stdout"
done
sha256sum -c "$out/ARTIFACTS.sha256" > "$out/ARTIFACTS.final.verify"
sha256sum -c "$out/SOURCE.sha256" > "$out/SOURCE.final.verify"
sha256sum -c "$out/ORIGINAL_DATA.sha256" > "$out/ORIGINAL_DATA.final.verify"
sha256sum -c "$out/GENERATED_DATA.sha256" > "$out/GENERATED_DATA.final.verify"
for family in binary literal text dynamic_slice owned last_use;do
 if test -f "$out/$family-ELF.sha256";then sha256sum -c "$out/$family-ELF.sha256" >> "$out/ELFS.final.verify";fi
done
printf '%s\n' "$bad" > "$out/GATE_STATUS"
test "$bad" = 0
printf '%s\n' RUNTIME_BYTES_CPP23_EXACT_ONLY_OK > "$out/STATUS"
cat "$out/STATUS"
