#!/bin/bash
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
root=/scratch/sounio-stdlib-advanced-20261023
run="$root/pilot_replay_hot01"
src="$root/check06/source"
before="$root/qualification01/pilot_B31/n16_alpha.25/batch0"
cd "$src"
export SOUNIO_STDLIB_PATH="$src/stdlib" SOUNIO_SOUC_ENGINE=madaros MADAROS_RAW_BIN="$root/compiler/artifacts/madaros.elf" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 TMPDIR="$run/tmp"
ulimit -s 524288
sha256sum "$before/driver.sio" "$MADAROS_RAW_BIN" stdlib/stats/regression/dynamic_count_glm.sio stdlib/linalg/least_squares_scale.sio > "$run/INPUTS.sha256"
bin/souc compile "$before/driver.sio" -o "$run/driver.elf" > "$run/compile.stdout" 2> "$run/compile.stderr"
sha256sum "$run/driver.elf" > "$run/ELF.sha256"
rc=0
/usr/bin/time -v -o "$run/native.resource" "$run/driver.elf" > "$run/native.stdout" 2> "$run/native.stderr" || rc=$?
printf '%s\n' "$rc" > "$run/native.status"
[[ "$rc" == 0 ]]
grep -qx 'COUNT PROFILE CALIBRATION BATCH COMPLETE' "$run/native.stdout"
n=$(wc -l < "$before/native.stdout")
head -n "$n" "$run/native.stdout" > "$run/shared-prefix.stdout"
cmp "$before/native.stdout" "$run/shared-prefix.stdout"
"$root/qualification01/pilot_B31/tools/count_profile_calibration" --summarize 31 16 "$run/native.stdout" > "$run/summary.tsv"
sha256sum -c "$run/INPUTS.sha256" > "$run/INPUTS-after.verify"
printf '0\n' > "$run/STATUS"
