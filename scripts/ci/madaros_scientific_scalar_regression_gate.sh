#!/usr/bin/env bash
# Strict native scientific scalar regressions; reuse an explicit built compiler.
# Usage: bash scripts/ci/madaros_scientific_scalar_regression_gate.sh COMPILER OUTPUT_DIR
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "usage: $0 COMPILER OUTPUT_DIR" >&2
  exit 2
fi
caller_dir=$(pwd -P)
SCRIPT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")
ROOT=$(cd "$(dirname "$SCRIPT")/../.." && pwd -P)
compiler_arg=$1
output_arg=$2
[[ $compiler_arg == /* ]] || compiler_arg="$caller_dir/$compiler_arg"
[[ $output_arg == /* ]] || output_arg="$caller_dir/$output_arg"
mkdir -p "$output_arg"
OUTPUT=$(cd "$output_arg" && pwd -P)
RUN=$(mktemp -d "$OUTPUT/scientific-scalar.XXXXXX")
mkdir -p "$RUN/expected" "$RUN/cases"
printf 'result_dir=%s\n' "$RUN"
trap 'rc=$?; printf "exit_code=%s\n" "$rc" > "$RUN/gate.status"' EXIT

for tool in timeout cmp sha256sum od; do
  command -v "$tool" >/dev/null || { echo "FAIL: required tool $tool missing" >&2; exit 2; }
done
[[ -f $compiler_arg && -x $compiler_arg ]] || { echo "FAIL: compiler is not an executable file: $compiler_arg" >&2; exit 2; }
COMPILER=$(cd "$(dirname "$compiler_arg")" && pwd -P)/$(basename "$compiler_arg")
magic=$(od -An -tx1 -N4 "$COMPILER")
[[ ${magic//[[:space:]]/} == 7f454c46 ]] || { echo "FAIL: explicit compiler must be a built ELF, not a wrapper" >&2; exit 2; }
cd "$ROOT"
# The source and stdlib always come from the same repository as this gate.
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset MADAROS_RAW_BIN MADAROS_BIN SOUNIO_SOUC_BIN SOUC_BIN SOUNIO_MADAROS_BIN SOUNIO_TEST_SOUC_BIN SOUNIO_SOUC_ENGINE
# Diagnostic peel bypasses must not turn the -O regression into a skipped pass.
for key in ${!SOUNIO_OCP_SKIP_@}; do unset "$key"; done
ulimit -S -s 65536

fixtures=(
  tests/run-pass/unit_parameter_nominal_issue2874.sio
  tests/run-pass/unit_parameter_range_issue2874.sio
  tests/run-pass/unit_parameter_representation_issue2874.sio
  tests/run-pass/knowledge_direct_call_variance_issue2827.sio
  tests/multimodule/unit_representation_2874/main.sio
  tests/run-pass/optimized_print_f64_name_id.sio
)
labels=(unit_nominal unit_range unit_representation knowledge_direct_call unit_imported optimized_print_guard)
for source in "${fixtures[@]}" tests/multimodule/unit_representation_2874/defs.sio; do
  [[ -f $source ]] || { echo "FAIL: missing real fixture $source" >&2; exit 2; }
done
[[ -d $ROOT/stdlib ]] || { echo "FAIL: matching repository stdlib missing" >&2; exit 2; }

printf '0.070000\n' > "$RUN/expected/unit_nominal.stdout"
printf '0.070000\nRESULTADO_CORRETO\n' > "$RUN/expected/unit_range.stdout"
printf 'UNIT_PARAMETER_PASS\n' > "$RUN/expected/unit_representation.stdout"
printf 'CALL\n0.002500\nCALL\n0.002500\n0.000000\n' > "$RUN/expected/knowledge_direct_call.stdout"
printf 'UNIT_IMPORTED_METADATA_PASS\n' > "$RUN/expected/unit_imported.stdout"
printf '0.070000\n' > "$RUN/expected/optimized_print_guard.stdout"

{
  printf 'compiler=%s\nrepository=%s\nstdlib=%s\ncompile_timeout_seconds=180\nruntime_timeout_seconds=30\nstack_kib=65536\n' "$COMPILER" "$ROOT" "$SOUNIO_STDLIB_PATH"
  printf 'job_id=%s\n' "${SLURM_JOB_ID:-not_slurm}"
} > "$RUN/provenance.txt"
if [[ -e $ROOT/.git ]]; then
  [[ $(git rev-parse --show-toplevel) == "$ROOT" ]] || { echo "FAIL: Git root does not match gate source root" >&2; exit 2; }
  {
    printf 'provenance_kind=git_checkout\nsource_head='; git rev-parse HEAD
    printf 'stdlib_tree='; git rev-parse HEAD:stdlib
  } >> "$RUN/provenance.txt"
  git status --porcelain > "$RUN/worktree-status.txt"
else
  [[ ${SOUNIO_SCI_SOURCE_COMMIT:-} =~ ^[0-9a-fA-F]{40}$ ]] || {
    echo "FAIL: no .git; require SOUNIO_SCI_SOURCE_COMMIT from the caller-verified snapshot manifest (40 hex characters)" >&2
    exit 2
  }
  printf 'provenance_kind=snapshot_manifest_asserted\nsource_commit=%s\nsource_head=unavailable\nstdlib_tree=unavailable\n' "$SOUNIO_SCI_SOURCE_COMMIT" >> "$RUN/provenance.txt"
  printf 'unavailable: snapshot has no .git\n' > "$RUN/worktree-status.txt"
fi
sha256sum "$COMPILER" > "$RUN/compiler.sha256"
sha256sum "$SCRIPT" > "$RUN/gate-script.sha256"
sha256sum "${fixtures[@]}" tests/multimodule/unit_representation_2874/defs.sio > "$RUN/fixtures.sha256"
sha256sum "$ROOT/stdlib/epistemic/knowledge.sio" > "$RUN/knowledge-source.sha256"
sha256sum self-hosted/ir/lower.sio self-hosted/ir/opt_cleanup.sio > "$RUN/compiler-source.sha256"
printf 'mode\tfixture\tcompile_rc\trun_rc\tstdout_cmp_rc\tverdict\n' > "$RUN/results.tsv"
failed=0
for mode in default optimized; do
  flags=()
  [[ $mode != optimized ]] || flags=(-O)
  for idx in "${!fixtures[@]}"; do
    source=${fixtures[$idx]}
    label=${labels[$idx]}
    case_dir="$RUN/cases/$mode-$label"
    mkdir -p "$case_dir"
    elf="$case_dir/program.elf"
    printf '%q ' timeout 180 "$COMPILER" "$source" -o "$elf" "${flags[@]}" > "$case_dir/compile.command"
    printf '\n' >> "$case_dir/compile.command"
    compile_rc=0
    timeout 180 "$COMPILER" "$source" -o "$elf" "${flags[@]}" > "$case_dir/compile.stdout" 2> "$case_dir/compile.stderr" || compile_rc=$?
    run_rc=not_run
    cmp_rc=not_run
    verdict=FAIL
    if [[ $compile_rc == 0 && -s $elf ]]; then
      chmod +x "$elf"
      sha256sum "$elf" > "$case_dir/program.sha256"
      run_rc=0
      timeout 30 "$elf" > "$case_dir/runtime.stdout" 2> "$case_dir/runtime.stderr" || run_rc=$?
      cmp_rc=0
      cmp "$RUN/expected/$label.stdout" "$case_dir/runtime.stdout" > "$case_dir/stdout.compare" 2>&1 || cmp_rc=$?
      # Exit status exercises each fixture's internal numerical guards. The
      # nominal case is display-only; representation/range cover its value.
      if [[ $run_rc == 0 && $cmp_rc == 0 ]]; then verdict=PASS; fi
    elif [[ $compile_rc == 0 ]]; then
      printf 'compiler exited zero without a nonempty native artifact\n' > "$case_dir/artifact.failure"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$label" "$compile_rc" "$run_rc" "$cmp_rc" "$verdict" >> "$RUN/results.tsv"
    printf '%s %s compile=%s run=%s cmp=%s %s\n' "$mode" "$label" "$compile_rc" "$run_rc" "$cmp_rc" "$verdict"
    [[ $verdict == PASS ]] || failed=1
  done
done
if [[ $failed == 0 ]]; then
  printf 'SCIENTIFIC_SCALAR_REGRESSION_PASS\n' | tee "$RUN/verdict.txt"
else
  printf 'SCIENTIFIC_SCALAR_REGRESSION_FAIL\n' | tee "$RUN/verdict.txt"
fi
exit "$failed"
