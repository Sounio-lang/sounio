#!/usr/bin/env bash
# Private ChangedTests protocol: one raw-modular compile, then that retained ELF.
set -euo pipefail
[[ $# == 5 ]] || { echo 'NATIVE_PHASE_REFUSE invalid_arguments' >&2;exit 64; }
raw=$1;expected_sha=$2;src=$3;native_timeout=$4;out=$5
[[ $out == /* && -d $out && ! -L $out && -z $(find "$out" -mindepth 1 -maxdepth 1 -print -quit) ]] || exit 64
phase=preflight
trap 'rc=$?;printf "%s\n" "$rc" > "$out/RESULT.status";printf "%s\n" "$phase" > "$out/RESULT.phase"' EXIT
[[ $raw == /* && -f $raw && -r $raw && -x $raw && $expected_sha =~ ^[0-9a-f]{64}$ ]] || exit 65
[[ $native_timeout =~ ^[1-9][0-9]*$ && "$SOUNIO_SOUC_RAW_MODE" == modular ]] || exit 65
[[ -f $src && ! -L $src ]] || exit 65
raw=$(readlink -f "$raw")
[[ "$(head -c 4 "$raw")" == $'\177ELF' ]] || { echo 'NATIVE_PHASE_REFUSE compiler_not_elf' >&2;exit 65; }
src=$(readlink -f "$src")
elf=$out/program.elf
[[ ! -e $elf && ! -L $elf ]]
printf '%s  %s\n' "$expected_sha" "$raw" > "$out/COMPILER.sha256"
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.before.verify"
sha256sum "$src" > "$out/SOURCE.sha256"
printf 'cwd=%q\nstdlib=%q\nraw_mode=%q\ncompile_budget_seconds=30\nnative_budget_seconds=%q\n' \
 "$PWD" "$SOUNIO_STDLIB_PATH" "$SOUNIO_SOUC_RAW_MODE" "$native_timeout" > "$out/CONTEXT.txt"
printf 'compile: timeout 30 %q %q -o %q\nnative: timeout %q %q\n' \
 "$raw" "$src" "$elf" "$native_timeout" "$elf" > "$out/COMMANDS.txt"
phase=compile
printf '%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$out/compile.started"
rc=0
timeout --kill-after=5 30 "$raw" "$src" -o "$elf" > "$out/compile.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$out/compile.status"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/compile.finished"
if [[ $rc != 0 ]];then
 cat "$out/compile.log"
 echo "NATIVE_PHASE_FAILURE phase=compile rc=$rc" >&2
 exit "$rc"
fi
# The legacy wrapper's modular branch can mask raw rc!=0. This private path
# captures the real rc above and also refuses an error diagnostic with rc0.
if grep -Eq '^[[:space:]]*error(\[[^]]+\])?:|typecheck: failed|^[[:space:]]*fatal:' "$out/compile.log";then
 cat "$out/compile.log"
 echo 'NATIVE_PHASE_FAILURE phase=compile diagnostic_with_exit_zero' >&2
 exit 1
fi
[[ -f $elf && ! -L $elf && -s $elf ]] || { echo 'NATIVE_PHASE_FAILURE phase=compile missing_regular_elf' >&2;exit 1; }
readelf -h "$elf" > "$out/ELF.header" 2> "$out/ELF.header.stderr" || { echo 'NATIVE_PHASE_FAILURE phase=compile invalid_elf' >&2;exit 1; }
grep -q 'Machine:.*X86-64' "$out/ELF.header" || { echo 'NATIVE_PHASE_FAILURE phase=compile unsupported_elf_machine' >&2;exit 1; }
chmod +x "$elf"
[[ -x $elf ]]
phase=integrity-before-native
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.pre-native.verify"
sha256sum -c "$out/SOURCE.sha256" > "$out/SOURCE.pre-native.verify"
sha256sum "$elf" > "$out/ELF.sha256"
stat -c '%d:%i:%s:%f' "$elf" > "$out/ELF.identity.before"
phase=native
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/native.started"
rc=0
timeout --kill-after=5 "$native_timeout" "$elf" > "$out/native.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$out/native.status"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$out/native.finished"
phase=integrity-after-native
sha256sum -c "$out/ELF.sha256" > "$out/ELF.after.verify"
stat -c '%d:%i:%s:%f' "$elf" > "$out/ELF.identity.after"
cmp "$out/ELF.identity.before" "$out/ELF.identity.after"
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.after.verify"
sha256sum -c "$out/SOURCE.sha256" > "$out/SOURCE.after.verify"
# Match wrapper run's compile-log concatenation; keep native program output
# unadorned on success so the caller retains its stdout/marker/xfail semantics.
printf '%s' "$(cat "$out/compile.log")"
cat "$out/native.log"
phase=native
if [[ $rc != 0 ]];then echo "NATIVE_PHASE_FAILURE phase=native rc=$rc" >&2;fi
exit "$rc"
