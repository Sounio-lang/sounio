#!/usr/bin/env bash
# Current-run compiler transfer; hashes bind the tracked input bytes, not a
# complete-consumption proof. Never execute a downloaded compiler before verify.
# `workflow` checks that ci.yml has one Witness Gate producer and that every
# current-source consumer depends on it, verifies the receipt, and does not
# build another ELF.
set -euo pipefail
export LC_ALL=C
fail() { printf 'STDLIB_DEEP_RECEIPT_FAIL: %s\n' "$*" >&2; exit 65; }
script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
root=$(cd "$(dirname "$script")/../.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/stdlib-deep-receipt.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
check_revision() {
 local repo=$1 expected=$2 actual
 [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || fail 'expected Git SHA must be a full SHA1'
 actual=$(git -C "$repo" rev-parse HEAD) || fail 'cannot read checkout HEAD'
 [[ "$actual" == "$expected" ]] || fail 'checkout HEAD differs from expected Git SHA'
}
paths() {
 local repo=$1 output=$2 fixture
 git -C "$repo" ls-files -z -- bin scripts self-hosted stdlib > "$work/paths.nul" || fail 'cannot enumerate tracked inputs'
 : > "$output"
 while IFS= read -r -d '' fixture; do
  case "$fixture" in *$'\n'*|*$'\r'*|*\\*) fail 'unsupported manifest filename' ;; esac
  [[ -f "$repo/$fixture" ]] || fail "tracked input missing or not a file: $fixture"
  printf '%s\n' "$fixture" >> "$output"
 done < "$work/paths.nul"
 sort -o "$output" "$output"
 [[ -s "$output" ]] || fail 'empty tracked input list'
}
prepare() {
 local repo=$1 compiler=$2 output=$3 expected=$4 fixture
 check_revision "$repo" "$expected"
 [[ -f "$compiler" && -x "$compiler" ]] || fail 'fresh compiler is missing or not executable'
 [[ "$output" == /* ]] || fail 'artifact directory must be absolute'
 mkdir -p "$output"
 [[ -z "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]] || fail 'artifact directory is not empty'
 printf '1\n' > "$output/RECEIPT_VERSION"
 printf '%s\n' "$expected" > "$output/SOURCE_GIT_SHA"
 paths "$repo" "$output/TRACKED_PATHS.txt"
 : > "$output/INPUTS.sha256"
 while IFS= read -r fixture; do
  (cd "$repo" && sha256sum -- "$fixture") >> "$output/INPUTS.sha256"
 done < "$output/TRACKED_PATHS.txt"
 cp -- "$compiler" "$output/madaros.elf"
 chmod 755 "$output/madaros.elf"
 (cd "$output" && sha256sum madaros.elf > COMPILER.sha256)
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_PREPARED %s\n' "$expected"
}
verify() {
 local repo=$1 artifact=$2 expected=$3 required line
 check_revision "$repo" "$expected"
 for required in RECEIPT_VERSION SOURCE_GIT_SHA TRACKED_PATHS.txt INPUTS.sha256 COMPILER.sha256 madaros.elf; do
  [[ -f "$artifact/$required" && ! -L "$artifact/$required" ]] || fail "missing or symlinked artifact file: $required"
 done
 printf '1\n' | cmp -s - "$artifact/RECEIPT_VERSION" || fail 'unsupported receipt version'
 printf '%s\n' "$expected" | cmp -s - "$artifact/SOURCE_GIT_SHA" || fail 'artifact source Git SHA mismatch'
 [[ "$(wc -l < "$artifact/COMPILER.sha256" | tr -d ' ')" == 1 ]] || fail 'compiler manifest must contain one entry'
 grep -Eq '^[0-9a-f]{64}  madaros\.elf$' "$artifact/COMPILER.sha256" || fail 'invalid compiler manifest'
 (cd "$artifact" && sha256sum -c COMPILER.sha256) || fail 'compiler hash mismatch'
 paths "$repo" "$work/current-paths.txt"
 cmp -s "$work/current-paths.txt" "$artifact/TRACKED_PATHS.txt" || fail 'tracked input set differs'
 : > "$work/manifest-paths.txt"
 while IFS= read -r line; do
  [[ "$line" =~ ^[0-9a-f]{64}'  '(.+)$ ]] || fail 'invalid source manifest line'
  printf '%s\n' "${BASH_REMATCH[1]}" >> "$work/manifest-paths.txt"
 done < "$artifact/INPUTS.sha256"
 cmp -s "$work/manifest-paths.txt" "$artifact/TRACKED_PATHS.txt" || fail 'source manifest set/order differs'
 (cd "$repo" && sha256sum -c "$artifact/INPUTS.sha256") || fail 'tracked source hash mismatch'
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_VERIFIED %s\n' "$expected"
}
selftest() {
 # An isolated disposable Git fixture; no project index/ref is changed.
 local repo="$work/fixture" artifact="$work/artifact" sha control rc
 mkdir -p "$repo"/{bin,scripts,self-hosted,stdlib}
 printf 'seed\n' > "$repo/bin/seed"
 printf 'script\n' > "$repo/scripts/build.sh"
 printf 'source\n' > "$repo/self-hosted/main.sio"
 printf 'stdlib\n' > "$repo/stdlib/mod.sio"
 git -c init.defaultBranch=main init -q "$repo"
 git -C "$repo" add bin scripts self-hosted stdlib
 git -C "$repo" -c commit.gpgsign=false -c core.hooksPath=/dev/null -c user.name=ReceiptFixture -c user.email=receipt-fixture@example.invalid commit -qm fixture
 sha=$(git -C "$repo" rev-parse HEAD)
 printf 'NONEXECUTED_TEST_COMPILER_BYTES\n' > "$work/compiler"
 chmod 755 "$work/compiler"
 "$script" prepare "$repo" "$work/compiler" "$artifact" "$sha" > "$work/prepare.log"
 "$script" verify "$repo" "$artifact" "$sha" > "$work/verify.log"
 # The compiler is version-bound, not attempt-bound: a failed-only CI rerun
 # may reuse this run's earlier successful build at the identical source SHA.
 GITHUB_RUN_ATTEMPT=2 "$script" verify "$repo" "$artifact" "$sha" > "$work/same-sha-rerun.log"
 printf 'RECEIPT_SAME_SHA_REUSE_ACCEPTED\n'
 rc=0
 GITHUB_RUN_ATTEMPT=2 "$script" verify "$repo" "$artifact" 0000000000000000000000000000000000000000 > "$work/different-sha-rerun.log" 2>&1 || rc=$?
 [[ "$rc" == 65 ]] || fail 'different-SHA reuse did not produce controlled exit65'
 grep -q '^STDLIB_DEEP_RECEIPT_FAIL:' "$work/different-sha-rerun.log" || fail 'different-SHA reuse lacked refusal diagnostic'
 printf 'RECEIPT_DIFFERENT_SHA_REUSE_REJECTED\n'
 for control in source-git-sha compiler-bit source-file missing-manifest; do
  cp -a "$artifact" "$work/$control"
  case "$control" in
   source-git-sha) printf '%040d\n' 0 > "$work/$control/SOURCE_GIT_SHA" ;;
   compiler-bit) printf X >> "$work/$control/madaros.elf" ;;
   source-file) cp "$repo/stdlib/mod.sio" "$work/original-source"; printf X >> "$repo/stdlib/mod.sio" ;;
   missing-manifest) rm "$work/$control/INPUTS.sha256" ;;
  esac
  rc=0
  "$script" verify "$repo" "$work/$control" "$sha" > "$work/$control.log" 2>&1 || rc=$?
  [[ "$rc" == 65 ]] || fail "tamper did not produce controlled exit65: $control ($rc)"
  grep -q '^STDLIB_DEEP_RECEIPT_FAIL:' "$work/$control.log" || fail "tamper lacked refusal diagnostic: $control"
  if [[ "$control" == source-file ]]; then cp "$work/original-source" "$repo/stdlib/mod.sio"; fi
  printf 'RECEIPT_TAMPER_REJECTED %s\n' "$control"
 done
 "$script" verify "$repo" "$artifact" "$sha" > "$work/final-verify.log"
 run_workflow "$root/.github/workflows/ci.yml"
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_SELFTEST_PASS\n'
}
artifact_name_re='^[[:space:]]+name:[[:space:]]+current-source-madaros-\$\{\{[[:space:]]*github\.run_id[[:space:]]*\}\}-\$\{\{[[:space:]]*github\.sha[[:space:]]*\}\}[[:space:]]*$'
producer_job=madaros-witness-gate
consumer_jobs=(madaros-current-source-deref-f64 stdlib-deep-science gate-wave-0)
# A shell comment is `#` at the start of the line, or whitespace then `#`.
# `true # bash ... verify` does not run verify. A scan of the raw line still
# sees the command text, and the copy proceeds.
strip_shell_comment() {
 local line=$1
 [[ "$line" =~ ^[[:space:]]*# ]] && return 0
 printf '%s\n' "${line%% #*}"
}
decomment_blob() {
 local blob=$1 line
 while IFS= read -r line; do
  strip_shell_comment "$line"
 done <<<"$blob"
}
scan_job_body() {
 local blob=$1 mode= line uploads=0 downloads=0 upload_at=0 download_at=0 verify_at=0 prepare_at=0 build_runs=0 unattached=0 n=0
 while IFS= read -r line; do
  n=$((n + 1))
  line=$(strip_shell_comment "$line")
  [[ -n "$line" ]] || continue
  if [[ "$line" =~ ^[[:space:]]{6}-[[:space:]] ]]; then
   mode=
   [[ "$line" == *"actions/upload-artifact@"* ]] && mode=upload
   [[ "$line" == *"actions/download-artifact@"* ]] && mode=download
  fi
  [[ "$line" =~ ^[[:space:]]+uses:[[:space:]]+actions/upload-artifact@ ]] && mode=upload
  [[ "$line" =~ ^[[:space:]]+uses:[[:space:]]+actions/download-artifact@ ]] && mode=download
  if [[ "$line" =~ $artifact_name_re ]]; then
   case "$mode" in
    upload) uploads=$((uploads + 1)); upload_at=$n ;;
    download) downloads=$((downloads + 1)); download_at=$n ;;
    *) unattached=1 ;;
   esac
   mode=
  fi
  [[ "$verify_at" == 0 && "$line" == *"stdlib_deep_compiler_receipt.sh verify"* ]] && verify_at=$n
  [[ "$prepare_at" == 0 && "$line" == *"stdlib_deep_compiler_receipt.sh prepare"* ]] && prepare_at=$n
  [[ "$line" == *"build_modular_madaros.sh"* && "$line" != *"hashFiles("* ]] && build_runs=$((build_runs + 1))
 done <<<"$blob"
 printf '%s %s %s %s %s %s %s %s\n' "$uploads" "$downloads" "$upload_at" "$download_at" "$verify_at" "$prepare_at" "$build_runs" "$unattached"
}
# A verify command the step can skip or ignore is not a consumer. A YAML `if:`,
# `continue-on-error`, a shell `if`, `set +e`, `&&` / `||`, a shell `!`, or a
# pipeline lets receipt failure pass. Without pipefail the pipeline status is
# the last command, so `verify | tee` still copies the ELF. Process
# substitution drops verify's status even when pipefail is on. A trailing `&`
# or `coproc` runs the copy before verify finishes, and verify's failure does
# not fail the step. An ERR trap runs the copy after verify fails, before the
# step exits. Placement is then skipped, or a rejected ELF is copied and run,
# and the f64 gate can build another ELF while this check stays green.
# `set -e` does not save a command in an AND/OR list, a shell conditional, a
# negation, a pipe, or a background job.
verify_step_problems() {
 local blob=$1 job=$2 line in_step=0 step_if=0 step_swallow=0 step_verify=0 bang amp
 flush_verify_step() {
  if [[ "$in_step" == 1 && "$step_verify" == 1 && ( "$step_if" == 1 || "$step_swallow" == 1 ) ]]; then
   printf 'verify-skippable %s\n' "$job"
  fi
 }
 while IFS= read -r line; do
  line=$(strip_shell_comment "$line")
  [[ -n "$line" ]] || continue
  if [[ "$line" =~ ^[[:space:]]{6}-[[:space:]] ]]; then
   flush_verify_step
   in_step=1
   step_if=0
   step_swallow=0
   step_verify=0
   # `- if: false` is a step condition on the opening line, not an eight-space key.
   [[ "$line" =~ ^[[:space:]]{6}-[[:space:]]+if: ]] && step_if=1
   [[ "$line" =~ ^[[:space:]]{6}-[[:space:]]+continue-on-error: ]] && step_swallow=1
  fi
  [[ "$in_step" == 1 ]] || continue
  # Step keys sit at eight spaces. A shell `if` inside `run: |` is indented further.
  [[ "$line" =~ ^[[:space:]]{8}if: ]] && step_if=1
  [[ "$line" =~ ^[[:space:]]{8}continue-on-error: ]] && step_swallow=1
  [[ "$line" == *'||'* || "$line" == *'&&'* ]] && step_swallow=1
  [[ "$line" =~ (^|[[:space:]])(if|elif|while|until)[[:space:]] ]] && step_swallow=1
  [[ "$line" =~ set[[:space:]]+\+([^[:space:]]*e|o[[:space:]]+[\'\"]?errexit) ]] && step_swallow=1
  # `!=` is a comparison. Any other `!` in this step negates a command.
  bang=${line//!=/}
  [[ "$bang" == *'!'* ]] && step_swallow=1
  # YAML `run: |` starts a block. Any other `|` is a shell pipeline.
  if [[ "$line" == *'|'* ]]; then
   [[ "$line" =~ ^[[:space:]]*(-\ )?run:[[:space:]]*\|[+-]?[[:space:]]*$ ]] || step_swallow=1
  fi
  [[ "$line" == *'<('* || "$line" == *'>('* ]] && step_swallow=1
  [[ "$line" =~ set[[:space:]]+\+o[[:space:]]+[\'\"]?pipefail ]] && step_swallow=1
  # `&&`, `&>`, `>&`, and `|&` are not background jobs. A remaining `&` is.
  amp=$line
  amp=${amp//&&/}
  amp=${amp//&>/}
  amp=${amp//>&/}
  amp=${amp//|&/}
  [[ "$amp" == *'&'* ]] && step_swallow=1
  [[ "$line" =~ (^|[[:space:]])coproc([[:space:]]|$) ]] && step_swallow=1
  [[ "$line" =~ (^|[[:space:]])trap[[:space:]] ]] && step_swallow=1
  # `echo "$(verify)"` and backticks succeed when verify fails. `set -e` does
  # not apply inside the substitution. An assignment of the substitution does
  # fail the step; nesting it in another command does not. A plain `echo` or
  # `printf` of the verify command is not a run either: the text matches, and
  # the copy still proceeds.
  [[ "$line" == *'$('* || "$line" == *'`'* ]] && step_swallow=1
  [[ "$line" =~ (^|[[:space:]])(echo|printf)([[:space:]]|$) ]] && step_swallow=1
  # `true` and `:` ignore their arguments and return 0, so the verify text is
  # not executed. A heredoc holds that text without running it. `bash +e`
  # turns errexit off in that shell, so a failed receipt still reaches `cp`.
  # `run:` and `name:` are YAML keys, not the `:` builtin.
  [[ "$line" =~ (^|[[:space:]])true([[:space:]]|$) || "$line" == *'/true'* ]] && step_swallow=1
  [[ "$line" =~ ^[[:space:]]{10,}:[[:space:]] || "$line" =~ run:[[:space:]]+:[[:space:]] ]] && step_swallow=1
  [[ "$line" == *'<<'* ]] && step_swallow=1
  [[ "$line" =~ (^|[[:space:]])\+e([^[:alnum:]_]|$) || "$line" =~ \+o[[:space:]]+[\'\"]?errexit ]] && step_swallow=1
  # `exit 0` and `exec /bin/true` end the step successfully before verify runs.
  # The text is still in the file, and the next step copies or builds another ELF.
  [[ "$line" =~ (^|[[:space:]])(exit|exec)([[:space:]]|$) ]] && step_swallow=1
  if [[ "$line" =~ ^[[:space:]]*(-\ )?shell: ]]; then
   [[ ( "$line" == *'-e'* || "$line" == *'errexit'* ) && "$line" == *'pipefail'* ]] || step_swallow=1
  fi
  [[ "$line" == *"stdlib_deep_compiler_receipt.sh verify"* ]] && step_verify=1
 done <<<"$blob"
 flush_verify_step
}
# Copying or executing the downloaded ELF before `verify` runs a rejected
# receipt, or runs the f64 gate while the verified compiler is still absent
# so that gate builds another ELF. Line order inside a step counts.
compiler_use_before_verify() {
 local blob=$1 line verify_at=0 token head use_at vhead
 local -a tokens=(madaros.elf madaros-wave0.elf deep_science_gate.sh madaros_current_source_f64_lowering_gate.sh sounio-madaros-current-source-f64-lowering/madaros)
 while IFS= read -r line; do
  line=$(strip_shell_comment "$line")
  [[ -n "$line" ]] || continue
  vhead=
  [[ "$line" == *"stdlib_deep_compiler_receipt.sh verify"* ]] && vhead=${line%%stdlib_deep_compiler_receipt.sh verify*}
  for token in "${tokens[@]}"; do
   [[ "$line" == *"$token"* ]] || continue
   [[ "$token" == 'madaros_current_source_f64_lowering_gate.sh' && "$line" == *'hashFiles('* ]] && continue
   [[ "$verify_at" == 1 ]] && continue
   head=${line%%"$token"*}
   if [[ -z "$vhead" || ${#head} -lt ${#vhead} ]]; then
    printf 'use-before-verify\n'
    return
   fi
  done
  [[ -n "$vhead" || "$line" == *"stdlib_deep_compiler_receipt.sh verify"* ]] && verify_at=1
 done <<<"$blob"
}
# GitHub's default shell is `bash -eo pipefail`. A workflow or job `shell:`
# replaces that for every `run:` step that does not set its own. Without
# errexit, a failed receipt is ignored and the next line copies the ELF.
shell_is_strict() {
 local line=$1
 [[ -z "$line" ]] && return 0
 [[ ( "$line" == *'-e'* || "$line" == *'errexit'* ) && "$line" == *'pipefail'* ]]
}
job_default_shell() {
 local blob=$1 line found=
 while IFS= read -r line; do
  line=$(strip_shell_comment "$line")
  [[ -n "$line" ]] || continue
  [[ "$line" =~ ^[[:space:]]{4}steps: ]] && break
  [[ "$line" =~ shell: ]] && found=$line
 done <<<"$blob"
 printf '%s\n' "$found"
}
# The shell key on the verify step, when that step sets one. It overrides a
# workflow or job default for that step.
verify_step_shell() {
 local blob=$1 line in_step=0 step_verify=0 shell_line=
 while IFS= read -r line; do
  line=$(strip_shell_comment "$line")
  [[ -n "$line" ]] || continue
  if [[ "$line" =~ ^[[:space:]]{6}-[[:space:]] ]]; then
   if [[ "$in_step" == 1 && "$step_verify" == 1 ]]; then
    printf '%s\n' "$shell_line"
    return 0
   fi
   in_step=1
   step_verify=0
   shell_line=
  fi
  [[ "$in_step" == 1 ]] || continue
  [[ "$line" == *"stdlib_deep_compiler_receipt.sh verify"* ]] && step_verify=1
  [[ "$line" =~ shell: ]] && shell_line=$line
 done <<<"$blob"
 if [[ "$in_step" == 1 && "$step_verify" == 1 ]]; then
  printf '%s\n' "$shell_line"
 fi
}
needs_has() {
 local text=$1 id=$2
 [[ "$text" =~ (^|[^A-Za-z0-9_-])${id}([^A-Za-z0-9_-]|$) ]]
}
# Producer and consumer job conditions are an OR of `flag == 'true'` terms.
# Replacing || with && keeps the same flag names and skips the producer while
# a consumer still wants to run.
or_true_flags() {
 local expr=$1 rest piece term_re="^needs\\.impact\\.outputs\\.[a-z0-9_]+ == 'true'$" out=''
 [[ "$expr" == *'&&'* || "$expr" == *'!'* || "$expr" == *'('* || "$expr" == *')'* ]] && return 1
 rest="$expr"
 while [[ -n "$rest" ]]; do
  if [[ "$rest" == *'||'* ]]; then
   piece="${rest%%||*}"
   rest="${rest#*||}"
  else
   piece="$rest"
   rest=
  fi
  piece="${piece#"${piece%%[![:space:]]*}"}"
  piece="${piece%"${piece##*[![:space:]]}"}"
  [[ "$piece" =~ $term_re ]] || return 1
  printf -v out '%s%s\n' "$out" "$piece"
 done
 [[ -n "$out" ]] || return 1
 printf '%s' "$out"
}
workflow_problems() {
 local file=$1 line job= in_jobs=0 in_needs=0 id rest flag consumer
 local workflow_shell='' job_shell=''
 local -a problems=()
 local -A body=() needs_text=() if_text=()
 [[ -f "$file" ]] || { printf 'workflow-missing %s\n' "$file"; return 0; }
 while IFS= read -r line || [[ -n "$line" ]]; do
  if [[ "$line" =~ ^[^[:space:]#] ]]; then
   if [[ "$line" == "jobs:" ]]; then in_jobs=1; job=; continue; fi
   in_jobs=0
   job=
   continue
  fi
  if [[ "$in_jobs" != 1 ]]; then
   [[ "$line" =~ shell: ]] && workflow_shell=$line
   continue
  fi
  if [[ "$line" =~ ^\ \ ([A-Za-z0-9_-]+):[[:space:]]*$ ]]; then
   job=${BASH_REMATCH[1]}
   body[$job]=
   needs_text[$job]=
   if_text[$job]=
   in_needs=0
   continue
  fi
  [[ -n "$job" ]] || continue
  body[$job]+=$line$'\n'
  if [[ "$line" =~ ^\ \ \ \ needs:[[:space:]]*(.*)$ ]]; then
   rest=${BASH_REMATCH[1]}
   # A name after `#` is not a dependency. `needs: [impact] # madaros-witness-gate`
   # does not wait for the producer.
   rest=${rest%% #*}
   [[ "$rest" == \#* ]] && rest=
   rest=${rest#"${rest%%[![:space:]]*}"}
   rest=${rest%"${rest##*[![:space:]]}"}
   needs_text[$job]=$rest
   if [[ -z "$rest" ]]; then in_needs=1; else in_needs=0; fi
   continue
  fi
  if [[ "$in_needs" == 1 ]]; then
   [[ "$line" =~ ^[[:space:]]*# ]] && continue
   rest=${line%% #*}
   if [[ "$rest" =~ ^\ \ \ \ \ \ -\ ([A-Za-z0-9_-]+)[[:space:]]*$ ]]; then
    needs_text[$job]+=" ${BASH_REMATCH[1]}"
    continue
   fi
   in_needs=0
  fi
  if [[ -z "${if_text[$job]}" && "$line" =~ ^\ \ \ \ if:[[:space:]]*(.*)$ ]]; then
   if_text[$job]=${BASH_REMATCH[1]}
  fi
 done < "$file"
 [[ -n "${body[$producer_job]:-}" ]] || problems+=("missing-producer $producer_job")
 local uploads downloads upload_at download_at verify_at prepare_at build_runs unattached blob known=0 code=''
 local producer_flags='' consumer_flags artifact_path='path: ${{ runner.temp }}/stdlib-deep-compiler'
 if [[ -n "${body[$producer_job]:-}" ]]; then
  blob=${body[$producer_job]}
  read -r uploads downloads upload_at download_at verify_at prepare_at build_runs unattached < <(scan_job_body "$blob")
  [[ "$uploads" == 1 && "$downloads" == 0 && "$unattached" == 0 ]] || problems+=("upload-count $producer_job uploads=$uploads downloads=$downloads unattached=$unattached")
  [[ "$unattached" == 0 ]] || problems+=("artifact-unattached $producer_job")
  [[ "$build_runs" -ge 1 ]] || problems+=("producer-build $producer_job")
  [[ "$prepare_at" -gt 0 && "$upload_at" -gt "$prepare_at" ]] || problems+=("prepare-before-upload $producer_job")
  [[ "$blob" == *"$artifact_path"* ]] || problems+=("download-path $producer_job")
  [[ -n "${if_text[$producer_job]:-}" && "${if_text[$producer_job]}" != *"always()"* ]] || problems+=("producer-if $producer_job")
  producer_flags=$(or_true_flags "${if_text[$producer_job]:-}" | sed 's/ == .*$//' | sort -u) || problems+=("impact-or $producer_job")
  [[ -n "$producer_flags" ]] || problems+=("producer-if $producer_job")
 fi
 for consumer in "${consumer_jobs[@]}"; do
  [[ -n "${body[$consumer]:-}" ]] || { problems+=("missing-consumer $consumer"); continue; }
  needs_has "${needs_text[$consumer]:-}" "$producer_job" || problems+=("needs-producer $consumer")
  blob=${body[$consumer]}
  read -r uploads downloads upload_at download_at verify_at prepare_at build_runs unattached < <(scan_job_body "$blob")
  [[ "$uploads" == 0 && "$downloads" == 1 && "$unattached" == 0 ]] || problems+=("download-count $consumer uploads=$uploads downloads=$downloads unattached=$unattached")
  [[ "$unattached" == 0 ]] || problems+=("artifact-unattached $consumer")
  [[ "$build_runs" == 0 ]] || problems+=("consumer-rebuilds $consumer")
  [[ "$download_at" -gt 0 && "$verify_at" -gt "$download_at" ]] || problems+=("verify-after-download $consumer")
  while IFS= read -r flag; do
   [[ -n "$flag" ]] || continue
   problems+=("$flag")
  done < <(verify_step_problems "$blob" "$consumer")
  while IFS= read -r flag; do
   [[ -n "$flag" ]] || continue
   problems+=("$flag $consumer")
  done < <(compiler_use_before_verify "$blob")
  job_shell=$(job_default_shell "$blob")
  [[ -z "$job_shell" ]] && job_shell=$workflow_shell
  # A shell key on the verify step overrides the default and is checked there.
  if [[ -z "$(verify_step_shell "$blob")" ]]; then
   shell_is_strict "$job_shell" || problems+=("shell-default $consumer")
  fi
  code=$(decomment_blob "$blob")
  [[ "$code" =~ stdlib_deep_compiler_receipt\.sh[[:space:]]+verify[[:space:]]*\\?[[:space:]]*\"\$GITHUB_WORKSPACE\"[[:space:]]*\"\$RUNNER_TEMP/stdlib-deep-compiler\"[[:space:]]*\"\$GITHUB_SHA\" ]] || problems+=("verify-arguments $consumer")
  [[ "$blob" == *"$artifact_path"* ]] || problems+=("download-path $consumer")
  [[ -n "${if_text[$consumer]:-}" ]] || problems+=("consumer-if $consumer")
  [[ "${if_text[$consumer]:-}" != *"always()"* ]] || problems+=("consumer-always $consumer")
  consumer_flags=$(or_true_flags "${if_text[$consumer]:-}" | sed 's/ == .*$//' | sort -u) || problems+=("impact-or $consumer")
  [[ -n "$consumer_flags" ]] || problems+=("consumer-if $consumer")
  if [[ "$code" == *'SOUNIO_MADAROS_F64_LOWERING_GATE_DIR'* ]]; then
   [[ "$code" == *'mkdir "$SOUNIO_MADAROS_F64_LOWERING_GATE_DIR"'* && "$code" != *'mkdir -p "$SOUNIO_MADAROS_F64_LOWERING_GATE_DIR"'* ]] || problems+=("fresh-gate-dir $consumer")
  fi
  while IFS= read -r flag; do
   [[ -n "$flag" ]] || continue
   grep -Fxq "$flag" <<<"$producer_flags" || problems+=("impact-subset $consumer $flag")
  done <<<"$consumer_flags"
 done
 for id in "${!body[@]}"; do
  known=0
  [[ "$id" == "$producer_job" ]] && known=1
  for consumer in "${consumer_jobs[@]}"; do [[ "$id" == "$consumer" ]] && known=1; done
  [[ "$known" == 1 ]] && continue
  read -r uploads downloads upload_at download_at verify_at prepare_at build_runs unattached < <(scan_job_body "${body[$id]}")
  [[ "$uploads" == 0 && "$downloads" == 0 && "$unattached" == 0 ]] || problems+=("unexpected-transfer $id uploads=$uploads downloads=$downloads unattached=$unattached")
 done
 if grep -q 'SOUNIO_MADAROS_F64_LOWERING_BUILD' "$file"; then problems+=("builder-override $file"); fi
 if ((${#problems[@]} > 0)); then printf '%s\n' "${problems[@]}"; fi
}
expect_workflow_problem() {
 local file=$1 token=$2 problems
 problems=$(workflow_problems "$file")
 grep -q "$token" <<<"$problems" || fail "workflow control accepted $token: ${problems:-<none>}"
 printf 'RECEIPT_WORKFLOW_CONTROL_REJECTED %s\n' "$token"
}
workflow_controls() {
 local base="$work/wf-good.yml" bad
 mkdir -p "$work"
 cat >"$base" <<'EOF'
name: fixture
on: push
jobs:
  madaros-witness-gate:
    needs: impact
    if: needs.impact.outputs.compiler == 'true' || needs.impact.outputs.runtime == 'true' || needs.impact.outputs.stdlib == 'true' || needs.impact.outputs.tests == 'true' || needs.impact.outputs.full == 'true'
    steps:
      - name: Build
        run: bash scripts/ci/build_modular_madaros.sh /tmp/madaros-ci.elf
      - name: Bind
        run: bash scripts/ci/stdlib_deep_compiler_receipt.sh prepare \
            "$GITHUB_WORKSPACE" /tmp/madaros-ci.elf "$RUNNER_TEMP/stdlib-deep-compiler" "$GITHUB_SHA"
      - name: Transfer
        uses: actions/upload-artifact@v4
        with:
          name: current-source-madaros-${{ github.run_id }}-${{ github.sha }}
          path: ${{ runner.temp }}/stdlib-deep-compiler/
  madaros-current-source-deref-f64:
    needs: [impact, madaros-witness-gate]
    if: needs.impact.outputs.compiler == 'true' || needs.impact.outputs.stdlib == 'true'
    steps:
      - name: Receive
        uses: actions/download-artifact@v4
        with:
          name: current-source-madaros-${{ github.run_id }}-${{ github.sha }}
          path: ${{ runner.temp }}/stdlib-deep-compiler
      - name: Verify
        run: bash scripts/ci/stdlib_deep_compiler_receipt.sh verify \
            "$GITHUB_WORKSPACE" "$RUNNER_TEMP/stdlib-deep-compiler" "$GITHUB_SHA"
  stdlib-deep-science:
    needs: [impact, madaros-witness-gate]
    if: needs.impact.outputs.compiler == 'true' || needs.impact.outputs.runtime == 'true'
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: current-source-madaros-${{ github.run_id }}-${{ github.sha }}
          path: ${{ runner.temp }}/stdlib-deep-compiler
      - name: Verify
        run: bash scripts/ci/stdlib_deep_compiler_receipt.sh verify "$GITHUB_WORKSPACE" "$RUNNER_TEMP/stdlib-deep-compiler" "$GITHUB_SHA"
  gate-wave-0:
    needs: [impact, madaros-witness-gate]
    if: needs.impact.outputs.full == 'true'
    steps:
      - name: Receive
        uses: actions/download-artifact@v4
        with:
          name: current-source-madaros-${{ github.run_id }}-${{ github.sha }}
          path: ${{ runner.temp }}/stdlib-deep-compiler
      - name: Verify
        run: |
          bash scripts/ci/stdlib_deep_compiler_receipt.sh verify \
            "$GITHUB_WORKSPACE" "$RUNNER_TEMP/stdlib-deep-compiler" "$GITHUB_SHA"
  unrelated:
    needs: impact
    if: needs.impact.outputs.website == 'true'
    steps:
      - name: Build something else
        run: bash scripts/ci/build_modular_madaros.sh /tmp/other.elf
EOF
 local good
 good=$(workflow_problems "$base")
 [[ -z "$good" ]] || fail "workflow control rejected the good fixture: $good"
 printf 'RECEIPT_WORKFLOW_CONTROL_ACCEPTED good\n'
 bad="$work/wf-needs.yml"
 sed 's/needs: \[impact, madaros-witness-gate\]/needs: impact/' "$base" >"$bad"
 # The sed above rewrites every consumer. One missing edge is enough.
 expect_workflow_problem "$bad" needs-producer
 bad="$work/wf-needs-comment.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /^    needs: \[impact, madaros-witness-gate\]/{print "    needs: [impact] # madaros-witness-gate"; arm=0; next} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" needs-producer
 bad="$work/wf-needs-comment-list.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /^    needs: \[impact, madaros-witness-gate\]/{print "    needs:"; print "      - impact"; print "      # - madaros-witness-gate"; arm=0; next} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" needs-producer
 bad="$work/wf-verify-echo.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{sub(/bash/, "echo bash"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-trailing-comment.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{sub(/bash/, "true # bash"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-after-download
 bad="$work/wf-verify-true.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{sub(/bash/, "true bash"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-colon.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{sub(/bash/, ": bash"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-heredoc.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{print "          cat <<'\''EOF'\''"; arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-bash-plus-e.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{sub(/bash/, "bash +e"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-exit.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{print "          exit 0"; arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-exec-true.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{print "          exec /bin/true"; arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-quoted-errexit.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{print "          set +o '\''errexit'\''"; arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify.yml"
 awk 'BEGIN{skip=0} /madaros-current-source-deref-f64/{skip=1} skip && /stdlib_deep_compiler_receipt.sh verify/{skip=0; next} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-after-download
 bad="$work/wf-rebuild.yml"
 awk 'BEGIN{arm=0} /^  gate-wave-0:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{print; print "          bash scripts/ci/build_modular_madaros.sh /tmp/madaros-wave0.elf"; arm=0; next} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" consumer-rebuilds
 bad="$work/wf-second-upload.yml"
 awk 'BEGIN{arm=0} /^  unrelated:/{arm=1} arm && /build_modular_madaros.sh/{print; print "      - uses: actions/upload-artifact@v4"; print "        with:"; print "          name: current-source-madaros-${{ github.run_id }}-${{ github.sha }}"; print "          path: ${{ runner.temp }}/stdlib-deep-compiler/"; arm=0; next} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" unexpected-transfer
 bad="$work/wf-subset.yml"
 sed "s/if: needs.impact.outputs.full == 'true'/if: needs.impact.outputs.website == 'true'/" "$base" >"$bad"
 expect_workflow_problem "$bad" impact-subset
 bad="$work/wf-and.yml"
 sed '/needs.impact.outputs.runtime/s/ || / \&\& /g' "$base" >"$bad"
 expect_workflow_problem "$bad" impact-or
 bad="$work/wf-always.yml"
 sed 's/if: needs.impact.outputs.full/if: always() \&\& needs.impact.outputs.full/' "$base" >"$bad"
 expect_workflow_problem "$bad" consumer-always
 bad="$work/wf-unattached.yml"
 sed 's/uses: actions\/upload-artifact@v4/uses: actions\/checkout@v4/' "$base" >"$bad"
 expect_workflow_problem "$bad" artifact-unattached
 bad="$work/wf-verify-if.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /name: Verify/{print; print "        if: false"; arm=0; next} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-if-open.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /name: Verify/{print "      - if: false"; arm=0; next} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-or.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /\$GITHUB_SHA"/{sub(/"$/, "\" || true"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-and.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /\$GITHUB_SHA"/{sub(/"$/, "\" \&\& true"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-continue.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /name: Verify/{print; print "        continue-on-error: true"; arm=0; next} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-shell-if.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{sub(/bash/, "if bash"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-set-plus-e.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{print "          set +e"; arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-bang.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{sub(/bash/, "! bash"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-pipe.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /\$GITHUB_SHA"/{sub(/"$/, "\" | tee log"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-procsub.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{sub(/bash/, "cat <(bash"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-shell-nopipefail.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /name: Verify/{print; print "        shell: bash -e {0}"; arm=0; next} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-background.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /\$GITHUB_SHA"/{sub(/"$/, "\" \&"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-coproc.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{sub(/bash/, "coproc VERIFY bash"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-trap.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{print "          trap true ERR"; arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-subst.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{sub(/bash/, "echo \"$(bash"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-verify-backtick.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{sub(/bash/, "echo `bash"); arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" verify-skippable
 bad="$work/wf-job-shell.yml"
 awk '/^  madaros-current-source-deref-f64:/{print; print "    defaults:"; print "      run:"; print "        shell: bash {0}"; next} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" shell-default
 bad="$work/wf-workflow-shell.yml"
 awk '/^jobs:/{print "defaults:"; print "  run:"; print "    shell: bash {0}"} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" shell-default
 bad="$work/wf-strict-shell.yml"
 awk '/^  madaros-current-source-deref-f64:/{print; print "    defaults:"; print "      run:"; print "        shell: bash -eo pipefail {0}"; next} {print}' "$base" >"$bad"
 good=$(workflow_problems "$bad")
 [[ -z "$good" ]] || fail "strict job shell was rejected: $good"
 printf 'RECEIPT_WORKFLOW_CONTROL_ACCEPTED strict-shell\n'
 bad="$work/wf-use-before-verify.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /name: Verify/{print "      - name: Run early"; print "        run: bash scripts/ci/madaros_current_source_f64_lowering_gate.sh"; arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" use-before-verify
 bad="$work/wf-copy-before-verify.yml"
 awk 'BEGIN{arm=0} /^  madaros-current-source-deref-f64:/{arm=1} arm && /stdlib_deep_compiler_receipt.sh verify/{print "          cp madaros.elf /tmp/out"; arm=0} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" use-before-verify
 bad="$work/wf-missing.yml"
 awk 'BEGIN{drop=0} /^  gate-wave-0:/{drop=1} drop && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ && !/^  gate-wave-0:/{drop=0} drop{next} {print}' "$base" >"$bad"
 expect_workflow_problem "$bad" missing-consumer
}
run_workflow() {
 local target=${1:-$root/.github/workflows/ci.yml} problems sabotaged
 bash "$root/scripts/ci/madaros_current_source_f64_lowering_gate.sh" --resolve-compiler-selftest \
  || fail 'preplaced compiler contract failed'
 workflow_controls
 problems=$(workflow_problems "$target")
 [[ -z "$problems" ]] || fail "workflow wiring mismatch: $problems"
 sabotaged="$work/ci-sabotaged.yml"
 sed '/^  gate-wave-0:$/,/^  [A-Za-z0-9_-]*:/ s/needs: \[impact, madaros-witness-gate\]/needs: impact/' "$target" >"$sabotaged"
 problems=$(workflow_problems "$sabotaged")
 grep -q 'needs-producer gate-wave-0' <<<"$problems" || fail "real workflow sabotage was not rejected: ${problems:-<none>}"
 printf 'RECEIPT_WORKFLOW_REAL_SABOTAGE_REJECTED needs-producer\n'
 sabotaged="$work/ci-mkdir-p.yml"
 sed 's/mkdir "\$SOUNIO_MADAROS_F64_LOWERING_GATE_DIR"/mkdir -p "\$SOUNIO_MADAROS_F64_LOWERING_GATE_DIR"/' "$target" >"$sabotaged"
 problems=$(workflow_problems "$sabotaged")
 grep -q 'fresh-gate-dir madaros-current-source-deref-f64' <<<"$problems" || fail "mkdir -p sabotage was not rejected: ${problems:-<none>}"
 printf 'RECEIPT_WORKFLOW_REAL_SABOTAGE_REJECTED fresh-gate-dir\n'
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_WORKFLOW_PASS\n'
}
case ${1:-} in
 prepare) [[ $# == 5 ]] || fail 'prepare REPO COMPILER ABSOLUTE_ARTIFACT_DIR EXPECTED_SHA'; prepare "$2" "$3" "$4" "$5" ;;
 verify) [[ $# == 4 ]] || fail 'verify REPO ARTIFACT_DIR EXPECTED_SHA'; verify "$2" "$3" "$4" ;;
 selftest) selftest ;;
 workflow) [[ $# == 1 || $# == 2 ]] || fail 'workflow [WORKFLOW_YML]'; run_workflow "${2:-$root/.github/workflows/ci.yml}" ;;
 *) fail 'expected prepare, verify, selftest or workflow' ;;
esac
