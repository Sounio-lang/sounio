#!/usr/bin/env bash
# Execute ONLY inside a Slurm allocation. Frozen source/compiler identities are
# required inputs; original source tree is read-only. No statistical computation.
# Usage: profile_normalization_into.sh FROZEN_SOURCE BEFORE_MODULE COMPILER OUTPUT
# Both variants call the unchanged legacy20000x512 workload, isolating the leaf
# result-allocation change. New strided API correctness has a separate oracle.
set -euo pipefail
[[ ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || { echo 'Slurm allocation required' >&2; exit 64; }
[[ $# == 4 ]] || exit 64
source_root=$(realpath "$1")
before=$(realpath "$2")
compiler=$(realpath "$3")
out=$4
[[ -d "$source_root/stdlib" && -f "$before" && -x "$compiler" && ! -e "$out" ]] || exit 65
mkdir -p "$out"
out=$(realpath "$out")
case "$out/" in "$source_root/"*) exit 65;; esac
ulimit -s 524288
printf 'job=%s\nstack_kib=%s\npurpose=paired-whole-program-normalization\n' "$SLURM_JOB_ID" "$(ulimit -s)" > "$out/RUN.txt"
sha256sum "$compiler" > "$out/COMPILER.sha256"
sha256sum "$before" "$source_root/stdlib/genomics/dynamic_normalization.sio" "$source_root/tests/run-pass/stdlib_deep_normalization.sio" > "$out/SOURCES.sha256"
printf 'variant\ttrial\tphase\twall_seconds\tuser_seconds\tsystem_seconds\tmax_rss_kib\texit_status\taccepted\n' > "$out/PHASES.tsv"
for variant in before after; do
 mkdir -p "$out/$variant"/{source,tmp,cache,artifacts}
 cp -a "$source_root/stdlib" "$out/$variant/source/"
 if [[ $variant == before ]]; then cp "$before" "$out/$variant/source/stdlib/genomics/dynamic_normalization.sio"; fi
 cp "$source_root/tests/run-pass/stdlib_deep_normalization.sio" "$out/$variant/source/workload.sio"
 find "$out/$variant/source" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/$variant/INPUTS.sha256"
 # Canonical wrapper, exact committed/attested ELF override; no decoder/build.
 rc=0
 env SOUNIO_STDLIB_PATH="$out/$variant/source/stdlib" MADAROS_RAW_BIN="$compiler" SOUNIO_SOUC_ENGINE=madaros SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_MADAROS_CACHE="$out/$variant/cache" TMPDIR="$out/$variant/tmp" \
 /usr/bin/time -f '%e\t%U\t%S\t%M\t%x' -o "$out/$variant/compile.time" timeout 240 "$source_root/bin/souc" compile "$out/$variant/source/workload.sio" -o "$out/$variant/artifacts/workload.elf" > "$out/$variant/compile.stdout" 2> "$out/$variant/compile.stderr" || rc=$?
 accepted=0; [[ $rc == 0 ]] && accepted=1
 printf '%s\t0\tcompile\t%s\t%s\n' "$variant" "$(tail -n1 "$out/$variant/compile.time")" "$accepted" >> "$out/PHASES.tsv"
 [[ $rc == 0 ]] || exit "$rc"
 chmod +x "$out/$variant/artifacts/workload.elf"
 sha256sum "$out/$variant/artifacts/workload.elf" > "$out/$variant/ELF.sha256"
done
for trial in 1 2 3; do
 if [[ $trial == 2 ]]; then variants='after before'; else variants='before after'; fi
 for variant in $variants; do
  rc=0
  /usr/bin/time -f '%e\t%U\t%S\t%M\t%x' -o "$out/$variant/run-$trial.time" timeout 180 "$out/$variant/artifacts/workload.elf" > "$out/$variant/run-$trial.stdout" 2> "$out/$variant/run-$trial.stderr" || rc=$?
  accepted=0
  if [[ $rc == 0 ]] && grep -Fxq STDLIB_DEEP_NORMALIZATION_PASS "$out/$variant/run-$trial.stdout"; then accepted=1; fi
  printf '%s\t%s\trun\t%s\t%s\n' "$variant" "$trial" "$(tail -n1 "$out/$variant/run-$trial.time")" "$accepted" >> "$out/PHASES.tsv"
  [[ $accepted == 1 ]] || exit 1
  cmp "$out/before/run-1.stdout" "$out/$variant/run-$trial.stdout"
 done
done
for variant in before after; do
 sha256sum -c "$out/$variant/INPUTS.sha256" > "$out/$variant/INPUTS.after.verify"
 sha256sum -c "$out/$variant/ELF.sha256" > "$out/$variant/ELF.after.verify"
done
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.after.verify"
sha256sum -c "$out/SOURCES.sha256" > "$out/SOURCES.after.verify"
find "$out" -type f ! -name OUTPUTS.sha256 -print0 | sort -z | xargs -0 sha256sum > "$out/OUTPUTS.sha256"
printf 'All paired whole-program trials accepted; descriptive three-pair sample, not isolated kernel timing.\n' > "$out/READY"
