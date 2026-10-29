#!/bin/bash
# Run only from a Slurm allocation. SOURCE is an immutable compiler snapshot.
set -euo pipefail
if test "$#" != 4; then printf 'usage: %s SOURCE RAW SEED RESULTS\n' "$0" >&2; exit 64; fi
test -n "${SLURM_JOB_ID:-}" || { printf 'Slurm allocation required\n' >&2; exit 65; }
S=$1; RAW=$2; SEED=$3; R=$4
mkdir -p "$R"
cd "$S"
export SOUNIO_STDLIB_PATH="$S/self-hosted"
ulimit -s 524288
ulimit -c 0
printf '8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932  %s\n' "$RAW" > "$R/RAW.sha256"
printf '9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a  %s\n' "$SEED" > "$R/SEED.sha256"
sha256sum -c "$R/RAW.sha256" > "$R/RAW.before.verify"
sha256sum -c "$R/SEED.sha256" > "$R/SEED.before.verify"
sha256sum self-hosted/ir/loop_checkpoint_elision.sio tests/fixtures/wave4_loop_checkpoint_elision/component.sio > "$R/COMPONENT.sha256"
status=0
/usr/bin/time -v timeout 300 "$RAW" check tests/fixtures/wave4_loop_checkpoint_elision/component.sio > "$R/check.stdout" 2> "$R/check.stderr" || status=$?
printf '%s\n' "$status" > "$R/CHECK_STATUS"
test "$status" = 0
status=0
/usr/bin/time -v timeout 300 "$SEED" tests/fixtures/wave4_loop_checkpoint_elision/component.sio "$R/component.elf" > "$R/compile.stdout" 2> "$R/compile.stderr" || status=$?
printf '%s\n' "$status" > "$R/COMPILE_STATUS"
test "$status" = 0
chmod +x "$R/component.elf"
sha256sum "$R/component.elf" > "$R/ELF.sha256"
status=0
/usr/bin/time -v timeout 60 "$R/component.elf" > "$R/native.stdout" 2> "$R/native.stderr" || status=$?
printf '%s\n' "$status" > "$R/NATIVE_STATUS"
test "$status" = 0
test "$(grep -c '^LCE_CASE ' "$R/native.stdout")" = 22
test "$(grep -c '^LCE_ATTACK ' "$R/native.stdout")" = 66
test "$(grep -c '^LCE_CONTROL_FLAGS ' "$R/native.stdout")" = 4
test "$(grep -c '^LCE_FLOAT_FLAGS ' "$R/native.stdout")" = 3
test "$(grep -cx 'LCE_ALIAS_ARGUMENTS_OK' "$R/native.stdout")" = 1
test "$(grep -cx 'LCE_MODULE_WRAPPER_OK' "$R/native.stdout")" = 1
test "$(grep -cx 'LOOP_CHECKPOINT_COMPONENT_ONLY_OK' "$R/native.stdout")" = 1
test "$(wc -l < "$R/native.stdout")" = 98
sha256sum -c "$R/COMPONENT.sha256" > "$R/COMPONENT.final.verify"
sha256sum -c "$R/RAW.sha256" > "$R/RAW.final.verify"
sha256sum -c "$R/SEED.sha256" > "$R/SEED.final.verify"
printf 'LOOP_CHECKPOINT_COMPONENT_GATE_OK\n'
