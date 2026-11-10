#!/usr/bin/env bash
# Foundry-only compile/run phase measurements; not a corpus or benchmark verdict.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
die() { printf 'phase-profile: %s\n' "$*" >&2; exit 78; }
# The runner emits fixed, one-field-per-line JSON. Read only its scalar fields;
# reject missing/duplicate fields rather than evaluating or sourcing JSON.
json_string() {
 local key="$1" file="$2"
 local values=()
 mapfile -t values < <(sed -nE 's/^[[:space:]]*"'"$key"'":[[:space:]]*"([^"]*)",?[[:space:]]*$/\1/p' "$file")
 [[ ${#values[@]} == 1 ]] || return 1
 printf '%s' "${values[0]}"
}
json_integer() {
 local key="$1" file="$2"
 local values=()
 mapfile -t values < <(sed -nE 's/^[[:space:]]*"'"$key"'":[[:space:]]*([0-9]+),?[[:space:]]*$/\1/p' "$file")
 [[ ${#values[@]} == 1 ]] || return 1
 printf '%s' "${values[0]}"
}
guard() {
 [[ "${SOUNIO_FOUNDRY_ENV_POLICY:-}" == clean-v1 ]] || die "Foundry clean-v1 required"
 [[ "${SOUNIO_PROFILE:-}" == corpus-smoke ]] || die "Foundry corpus-smoke dispatch required"
 [[ "${SOUNIO_RUN_ID:-}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "missing run identity"
 [[ -n "${SOUNIO_SOURCE:-}" && "$(realpath -e "$SOUNIO_SOURCE")" == "$root" ]] || die "source root mismatch"
 [[ -n "${SOUNIO_ARTIFACTS:-}" && -d "$SOUNIO_ARTIFACTS" ]] || die "missing artifact root"
 artifacts="$(realpath -e "$SOUNIO_ARTIFACTS")"
 case "$artifacts/" in "$root/"*) die "artifacts must be outside source";; esac
 receipt="$artifacts/runtime-receipt.json"
 [[ -f "$receipt" ]] || die "missing preexisting Foundry runtime receipt"
 [[ "$(json_string schema "$receipt")" == sounio.compiler_foundry.runtime_receipt.v1 ]] || die "receipt schema"
 [[ "$(json_string environment_policy "$receipt")" == clean-v1 ]] || die "receipt environment policy"
 [[ "$(json_integer actual_uid "$receipt")" == "$(id -u)" ]] || die "receipt uid"
 [[ "$(json_integer actual_gid "$receipt")" == "$(id -g)" ]] || die "receipt gid"
 job="$(json_string slurm_job_id "$receipt")"
 [[ "$job" =~ ^[1-9][0-9]*$ ]] || die "positive Slurm job receipt required"
}
guard
# Internal batch subprocess: no compiler, hashing, or per-execution timers here.
# Its one shell startup and guard are included in the batch wall measurement.
if [[ "${1:-}" == --batch-run ]]; then
 [[ $# == 2 ]] || die "internal batch requires one entry directory"
 batch_dir="$(realpath -e "$2")"
 case "$batch_dir/" in "$artifacts/phase-profile/"*) ;; *) die "batch escapes profile";; esac
 [[ -f "$batch_dir/INITIAL_RUN_ACCEPTED" && -x "$batch_dir/program.elf" ]] || die "batch requires accepted initial ELF run"
 batch_rc=0
 for i in {1..20}; do
  "$batch_dir/program.elf" > "$batch_dir/batch.$i.stdout" 2> "$batch_dir/batch.$i.stderr" || batch_rc=$?
 done
 exit "$batch_rc"
fi
[[ $# == 0 ]] || die "usage: profile_stats_viz_phases.sh (inside Foundry package dispatch)"
[[ -f "$root/PACKAGE_READY" ]] || die "prepared runtime package required"
[[ -x /usr/bin/time ]] || die "GNU /usr/bin/time required"
out="$artifacts/phase-profile"
mkdir -m 700 -- "$out"
completed=0
finish() {
 local code=$?
 if [[ $completed != 1 ]]; then printf 'exit_code=%s\n' "$code" > "$out/PROFILE_FAILED"; fi
}
trap finish EXIT
printf 'schema=stats-viz-phase-profile-v1\npurpose=paired-compile-and-direct-ELF-run-measurement\ncorpus_acceptance=false\ntrials=3\nbatch_executions=20\npolicy=committed-Madaros-required;canonical-launcher-including-integrity-rechecks;initial-materialization-outside-timing\nmax_rss=GNU-time-process-descendant-maximum-not-simultaneous-aggregate\nbatch_overhead=one-shell-startup-and-Foundry-guard-included\ntime_resolution_s=0.01\nstack_ceiling_kib=524288-not-resident-memory-reservation\n' > "$out/provenance.txt"
export SOUNIO_SOUC_ENGINE=madaros SOUNIO_STDLIB_PATH="$root/stdlib"
# Initial materialization is outside timing; each canonical compile includes
# its normal strict committed-byte integrity rechecks and launcher startup.
export SOUNIO_REQUIRE_COMMITTED_MADAROS=1
(
 cd "$root"
 bash scripts/lib/materialize_madaros_prebuilt.sh --verify
 sha256sum --check bin/madaros-linux-x86_64.sha256
) > "$out/compiler-preparation.log" 2>&1
verified_compiler_sha="$(sha256sum -- "$root/bin/madaros-linux-x86_64" | cut -c1-64)"
printf 'variable\tvalue\n' > "$out/inherited-compiler.tsv"
for key in SOUNIO_SOUC_BIN SOUNIO_MADAROS_BIN MADAROS_RAW_BIN MADAROS_BIN SOUC_BIN SOUNIO_TEST_SOUC_BIN; do
 inherited="${!key:-}"
 if [[ -n "$inherited" ]]; then
  case "$key" in
   SOUC_BIN|SOUNIO_TEST_SOUC_BIN)
    case "$inherited" in
     "$root/bin/souc"|"$root/scripts/ci/souc-native-wrapper.sh"|"$root/bin/madaros-linux-x86_64") ;;
     *) die "external inherited compiler path: $key";;
    esac;;
   *)
    [[ "$inherited" == "$root/bin/madaros-linux-x86_64" ]] || die "external raw compiler override: $key";;
  esac
  [[ -x "$inherited" && ! -L "$inherited" ]] || die "invalid inherited compiler file: $key"
  if [[ "$inherited" == "$root/bin/madaros-linux-x86_64" ]]; then
   [[ "$(sha256sum -- "$inherited" | cut -c1-64)" == "$verified_compiler_sha" ]] || die "inherited compiler checksum: $key"
  fi
 fi
 printf '%s\t%s\n' "$key" "$inherited" >> "$out/inherited-compiler.tsv"
done
unset MADAROS_RAW_BIN SOUNIO_MADAROS_BIN SOUNIO_SOUC_BIN MADAROS_BIN SOUNIO_TEST_SOUC_BIN
export MADAROS_STACK_KB=524288
ulimit -s 524288 || die "cannot establish 524288 KiB parent stack ceiling"
[[ "$(ulimit -s)" == 524288 ]] || die "parent stack ceiling mismatch"
unset SOUC_BIN
source "$root/scripts/lib/resolve_souc.sh"
sounio_require_souc
[[ "$(realpath -e "$SOUC_BIN")" == "$root/bin/souc" ]] || die "canonical root/bin/souc resolution required"
sha() { sha256sum -- "$1" | cut -c1-64; }
compiler_sha="$(sha "$root/bin/madaros-linux-x86_64")"
receipt_sha="$(sha "$receipt")"
inputs() {
 (
  cd "$root"
  [[ -z "$(find bin scripts stdlib tests -type l -print -quit)" ]] || die "symlinked package inputs unsupported"
  find bin scripts stdlib tests -type f ! -path bin/.madaros-linux-x86_64.verified -print0 |
   LC_ALL=C sort -z | xargs -0 sha256sum
 )
}
inputs > "$out/inputs.before.sha256"
inputs_sha="$(sha "$out/inputs.before.sha256")"
cp -- "$receipt" "$out/runtime-receipt.json"
printf 'kind\ttrial\tphase\twall_s\tuser_s\tsystem_s\tmax_rss_kib\trc\taccepted\tentry_bytes\telf_bytes\tsource_sha256\tcompiler_sha256\tinputs_sha256\truntime_receipt_sha256\tslurm_job_id\telf_sha256\tstdout_sha256\tstderr_sha256\n' > "$out/phases.tsv"
failed=0
phase_rc=0
timed() {
 local phase="$1"; shift
 if /usr/bin/time -q -o "$work/$phase.time" -f '%e\t%U\t%S\t%M\t%x' timeout --kill-after=5s 300s "$@" > "$work/$phase.stdout" 2> "$work/$phase.stderr"; then phase_rc=0; else phase_rc=$?; fi
}
record() {
 local phase="$1" accepted="$2" stdout="$3" stderr="$4"
 local wall=NA user=NA sys=NA rss=NA timer_rc=NA
 if [[ -f "$work/$phase.time" ]]; then IFS=$'\t' read -r wall user sys rss timer_rc < "$work/$phase.time" || true; fi
 [[ "$timer_rc" == "$phase_rc" ]] || accepted=0
 printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$kind" "$trial" "$phase" "$wall" "$user" "$sys" "$rss" "$phase_rc" "$accepted" "$entry_bytes" "$elf_bytes" "$source_sha" "$compiler_sha" "$inputs_sha" "$receipt_sha" "$job" "$elf_sha" "$(sha "$stdout")" "$(sha "$stderr")" >> "$out/phases.tsv"
 if [[ $accepted != 1 ]]; then failed=1; fi
}
skip() {
 local phase="$1"
 : > "$work/$phase.stdout"; : > "$work/$phase.stderr"
 phase_rc=not_run
 record "$phase" 0 "$work/$phase.stdout" "$work/$phase.stderr"
}
run_entry() {
 kind="$1"; trial="$2"; local entry="$3"
 local source="$root/$entry"
 [[ -f "$source" ]] || die "missing fixed source: $entry"
 work="$out/$kind-$trial"
 mkdir -m 700 -- "$work"
 printf '%s\n' "$entry" > "$work/entry.txt"
 source_sha="$(sha "$source")"; entry_bytes="$(stat -c %s "$source")"
 elf_sha=NA; elf_bytes=0
 timed compile "$SOUC_BIN" compile "$entry" -o "$work/program.elf"
 local accepted=0
 if [[ $phase_rc == 0 && -s "$work/program.elf" && -x "$work/program.elf" ]] &&
    [[ "$(head -c4 "$work/program.elf" | od -An -tx1 | tr -d ' \n')" == 7f454c46 ]]; then
  accepted=1; elf_sha="$(sha "$work/program.elf")"; elf_bytes="$(stat -c %s "$work/program.elf")"
 fi
 record compile "$accepted" "$work/compile.stdout" "$work/compile.stderr"
 if [[ $accepted != 1 ]]; then skip run; skip run_batch20; return; fi
 timed run "$work/program.elf"
 accepted=1
 [[ $phase_rc == 0 && "$(sha "$work/program.elf")" == "$elf_sha" ]] || accepted=0
 local markers=0 line pattern
 while IFS= read -r line; do
  case "$line" in
   '//@ expect-stdout: '*) pattern="${line#//@ expect-stdout: }";;
   '//@ expect-stdout-contains: '*) pattern="${line#//@ expect-stdout-contains: }";;
   *) continue;;
  esac
  markers=$((markers+1))
  grep -Fq -- "$pattern" "$work/run.stdout" || accepted=0
 done < "$source"
 [[ $markers -gt 0 ]] || accepted=0
 record run "$accepted" "$work/run.stdout" "$work/run.stderr"
 if [[ $accepted != 1 ]]; then skip run_batch20; return; fi
 touch "$work/INITIAL_RUN_ACCEPTED"
 timed run_batch20 bash "$root/scripts/dev/profile_stats_viz_phases.sh" --batch-run "$work"
 accepted=1
 [[ $phase_rc == 0 && "$(sha "$work/program.elf")" == "$elf_sha" ]] || accepted=0
 for i in {1..20}; do
  [[ -f "$work/batch.$i.stdout" && -f "$work/batch.$i.stderr" ]] || { accepted=0; continue; }
  cmp -s -- "$work/run.stdout" "$work/batch.$i.stdout" || accepted=0
 done
 # Batch stdout/stderr columns identify manifests of the twenty retained files.
 (
  cd "$work"
  for i in {1..20}; do sha256sum -- "batch.$i.stdout"; done
 ) > "$work/batch-stdout.sha256" || accepted=0
 (
  cd "$work"
  for i in {1..20}; do sha256sum -- "batch.$i.stderr"; done
 ) > "$work/batch-stderr.sha256" || accepted=0
 record run_batch20 "$accepted" "$work/batch-stdout.sha256" "$work/batch-stderr.sha256"
}
cd "$root"
for trial in 1 2 3; do
 if [[ $trial == 2 ]]; then
  run_entry after "$trial" tests/run-pass/stats_poisson_reference.sio
  run_entry before "$trial" tests/run-pass/phase_before_poisson_reference.sio
 else
  run_entry before "$trial" tests/run-pass/phase_before_poisson_reference.sio
  run_entry after "$trial" tests/run-pass/stats_poisson_reference.sio
 fi
done
run_entry viz 1 tests/run-pass/viz_poisson_offsets.sio
inputs > "$out/inputs.after.sha256"
cmp -s "$out/inputs.before.sha256" "$out/inputs.after.sha256" || failed=1
[[ "$(sha "$root/bin/madaros-linux-x86_64")" == "$compiler_sha" && "$(sha "$receipt")" == "$receipt_sha" ]] || failed=1
printf 'accepted=%s\n' "$((1-failed))" > "$out/summary.txt"
if [[ $failed != 0 ]]; then exit 1; fi
printf 'purpose=phase-profile-not-corpus\n' > "$out/PROFILE_READY"
completed=1
printf 'Phase profile ready: %s\n' "$out"
