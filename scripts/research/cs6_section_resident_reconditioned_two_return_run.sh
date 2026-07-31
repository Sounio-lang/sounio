#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/scripts/research/cs6_section_resident_reconditioned_two_return_probe.cpp"
verifier_file="$repo_root/scripts/research/cs6_section_resident_reconditioned_two_return_verify.py"
runner_file="$repo_root/scripts/research/cs6_section_resident_reconditioned_two_return_run.sh"
flattened_receipt_file="$repo_root/scripts/research/cs6_section_resident_two_return_receipt_v1.txt"
flattened_provenance_file="$repo_root/scripts/research/cs6_section_resident_two_return_provenance_v1.txt"
flattened_verifier_file="$repo_root/scripts/research/cs6_section_resident_two_return_verify.py"

expected_flattened_receipt_sha256="14315dd35ada83d13bddaa1c653e0dea86a9da91379559e7f64d69b314077dba"
expected_flattened_physical_sha256="536dea89d9f841e0afedaaeb9ef116f5237fb7dd96f7774340850833b5f4b0b1"

capd_config=""
run_dir=""
challenge=""
cxx="${CXX:-g++}"

usage() {
  printf '%s\n' \
    "usage: $0 --capd-config PATH --run-dir DIR --challenge SHA256 [--cxx PATH]"
}

while (($#)); do
  case "$1" in
    --capd-config) capd_config="${2:-}"; shift 2 ;;
    --run-dir) run_dir="${2:-}"; shift 2 ;;
    --challenge) challenge="${2:-}"; shift 2 ;;
    --cxx) cxx="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

[[ -n "$capd_config" && -n "$run_dir" && -n "$challenge" ]] || {
  usage >&2
  exit 64
}
[[ "$challenge" =~ ^[0-9a-f]{64}$ ]] || {
  printf 'challenge must be lowercase SHA-256\n' >&2
  exit 64
}
for path in "$source_file" "$verifier_file" "$runner_file" \
  "$flattened_receipt_file" "$flattened_provenance_file" \
  "$flattened_verifier_file"; do
  [[ -f "$path" ]] || {
    printf 'section-resident reconditioned source is incomplete: %s\n' \
      "$path" >&2
    exit 66
  }
done

capd_config="$(realpath "$capd_config")"
[[ -x "$capd_config" ]] || { printf 'invalid capd-config\n' >&2; exit 66; }
cxx_path="$(realpath "$(command -v "$cxx")")"
python_path="$(realpath "$(command -v python3)")"
run_dir="$(realpath -m "$run_dir")"
[[ ! -e "$run_dir" ]] || { printf 'run directory already exists\n' >&2; exit 73; }
cd "$repo_root"
mkdir -p "$(dirname "$run_dir")"
work="$(mktemp -d "$(dirname "$run_dir")/.cs6-section-resident-reconditioned.XXXXXX")"
trap 'rm -rf "$work"' EXIT

cp "$source_file" "$work/probe-source.cpp"
cp "$verifier_file" "$work/verifier.py"
cp "$runner_file" "$work/runner.sh"
cp "$capd_config" "$work/capd-config-retained"
cp "$flattened_receipt_file" "$work/flattened-baseline-receipt.txt"
cp "$flattened_provenance_file" "$work/flattened-baseline-provenance.txt"
cp "$flattened_verifier_file" "$work/flattened-baseline-verifier.py"

{
  printf 'INPUT_SCHEMA=sounio.cs6.section-resident-reconditioned-two-return-input.v1\n'
  printf 'SOURCE=N0\n'
  printf 'U_INDEX=20000\n'
  printf 'S_INDEX=15000\n'
  printf 'U_TILES=40000\n'
  printf 'S_TILES=30000\n'
  printf 'ORDER=8\n'
  printf 'RETURN_COUNT=2\n'
  printf 'SECTION=COORDINATE_W_EQUALS_ZERO\n'
  printf 'CROSSING_DIRECTION=MINUS_PLUS\n'
  printf 'VECTOR_FIELD=CS6_FROZEN_22.3274637391\n'
  printf 'TANGENT_GAUGES=IDENTITY,MIDPOINT_M,ORIENTED_QR\n'
} > "$work/input.txt"

"$capd_config" --cflags > "$work/capd-cflags.txt"
"$capd_config" --libs > "$work/capd-libs.txt"
"$capd_config" --modversion > "$work/capd-version.txt"
"$cxx_path" --version > "$work/compiler-version.txt"
"$python_path" --version > "$work/python-version.txt" 2>&1
printf '%s\n' "$cxx_path" > "$work/compiler-path.txt"
printf '%s\n' "$python_path" > "$work/python-path.txt"
git -C "$repo_root" rev-parse HEAD > "$work/git-head.txt"
git -C "$repo_root" status --short --untracked-files=all > "$work/git-status.txt"
sed -i 's/[[:space:]]\+$//' "$work/capd-cflags.txt" "$work/capd-libs.txt"

grep -Fxq '5.3.0' "$work/capd-version.txt" || {
  printf 'CAPD version is not 5.3.0\n' >&2
  exit 65
}
grep -Eq '(^| )-D__USE_FILIB__( |$)' "$work/capd-cflags.txt" || {
  printf 'CAPD config does not select FILIB\n' >&2
  exit 65
}
grep -Eq '(^| )-frounding-math( |$)' "$work/capd-cflags.txt" || {
  printf 'CAPD config omits -frounding-math\n' >&2
  exit 65
}
! grep -Eq '(^| )(-U__USE_FILIB__|-fno-rounding-math)( |$)' \
  "$work/capd-cflags.txt" || {
  printf 'CAPD config contains a contradictory effective flag\n' >&2
  exit 65
}

read -r -a cflags < "$work/capd-cflags.txt"
read -r -a libs < "$work/capd-libs.txt"
"$cxx_path" "${cflags[@]}" -dM -E -x c++ /dev/null \
  > "$work/preprocessor-macros.txt"
sed -i 's/[[:space:]]\+$//' "$work/preprocessor-macros.txt"
[[ "$(grep -Ec '^#define __USE_FILIB__( 1)?$' "$work/preprocessor-macros.txt")" == 1 ]] || {
  printf 'effective preprocessor state does not select FILIB\n' >&2
  exit 65
}
"$cxx_path" "${cflags[@]}" -O0 -Q --help=optimizers -c -x c++ /dev/null \
  -o "$work/effective-options.o" > "$work/effective-options.txt" \
  2> "$work/effective-options-stderr.txt"
rm -f "$work/effective-options.o"
sed -i 's/[[:space:]]\+$//; /^$/d' "$work/effective-options.txt"
grep -Eq '^[[:space:]]*-frounding-math[[:space:]]+\[enabled\]' \
  "$work/effective-options.txt" || {
  printf 'effective compiler state disables -frounding-math\n' >&2
  exit 65
}

source_hash="$(sha256sum "$work/probe-source.cpp" | awk '{print $1}')"
verifier_hash="$(sha256sum "$work/verifier.py" | awk '{print $1}')"
runner_hash="$(sha256sum "$work/runner.sh" | awk '{print $1}')"
input_hash="$(sha256sum "$work/input.txt" | awk '{print $1}')"
flattened_receipt_hash="$(sha256sum "$work/flattened-baseline-receipt.txt" | awk '{print $1}')"
flattened_provenance_hash="$(sha256sum "$work/flattened-baseline-provenance.txt" | awk '{print $1}')"
flattened_verifier_hash="$(sha256sum "$work/flattened-baseline-verifier.py" | awk '{print $1}')"

[[ "$flattened_receipt_hash" == "$expected_flattened_receipt_sha256" ]] || {
  printf 'flattened baseline receipt hash mismatch\n' >&2
  exit 65
}
[[ "$(grep -Fxc "RECEIPT_SHA256=$expected_flattened_receipt_sha256" \
  "$work/flattened-baseline-provenance.txt")" == 1 ]] || {
  printf 'flattened baseline provenance does not bind its receipt\n' >&2
  exit 65
}
[[ "$(grep -Fxc "VERIFIER_SHA256=$flattened_verifier_hash" \
  "$work/flattened-baseline-provenance.txt")" == 1 ]] || {
  printf 'flattened baseline provenance does not bind its verifier\n' >&2
  exit 65
}
[[ "$(grep -Fxc "PHYSICAL_CHAIN_SHA256=$expected_flattened_physical_sha256" \
  "$work/flattened-baseline-provenance.txt")" == 1 ]] || {
  printf 'flattened baseline physical digest mismatch\n' >&2
  exit 65
}

compile_args=("$cxx_path" -std=c++17 "${cflags[@]}" -O0 \
  "-DCS6_WORKER_SOURCE_SHA256=\"$source_hash\"" \
  "-DCS6_INPUT_SHA256=\"$input_hash\"" \
  "-DCS6_RUN_CHALLENGE=\"$challenge\"" \
  "$work/probe-source.cpp" -MD \
  -MF "$work/dependencies.d" -o "$work/probe-binary.tmp" "${libs[@]}")
{
  printf '%q' "${compile_args[0]}"
  printf ' %q' "${compile_args[@]:1}"
  printf '\n'
} > "$work/compile-command.txt"

"${compile_args[@]}" 2> "$work/compile-stderr.txt"
mv "$work/probe-binary.tmp" "$work/probe-binary"

ldd "$work/probe-binary" > "$work/runtime-linkage.txt"
sed ':a;N;$!ba;s/\\\n/ /g' "$work/dependencies.d" | \
  sed 's/^[^:]*:[[:space:]]*//' | tr ' ' '\n' | sed '/^$/d' | sort -u \
  > "$work/dependency-paths.txt"
: > "$work/dependencies-before-run.sha256"
while IFS= read -r dependency; do
  [[ -f "$dependency" ]] && sha256sum "$dependency"
done < "$work/dependency-paths.txt" >> "$work/dependencies-before-run.sha256"
sort -u "$work/dependencies-before-run.sha256" \
  -o "$work/dependencies-before-run.sha256"
[[ -s "$work/dependencies-before-run.sha256" ]] || {
  printf 'compiler emitted no hashable dependencies\n' >&2
  exit 65
}

: > "$work/link-inputs-before-run.sha256"
for argument in "${libs[@]}"; do
  [[ -f "$argument" ]] && sha256sum "$argument" \
    >> "$work/link-inputs-before-run.sha256"
done
[[ -s "$work/link-inputs-before-run.sha256" ]] || {
  printf 'CAPD link flags contain no hashable files\n' >&2
  exit 65
}

awk '/=> \// {print $3} /^\// {print $1}' "$work/runtime-linkage.txt" | sort -u \
  > "$work/runtime-library-paths.txt"
: > "$work/runtime-libraries-before-run.sha256"
while IFS= read -r library; do
  [[ -f "$library" ]] && sha256sum "$library"
done < "$work/runtime-library-paths.txt" \
  >> "$work/runtime-libraries-before-run.sha256"

set +e
timeout 180 "$work/probe-binary" > "$work/ledger.txt" \
  2> "$work/probe-stderr.txt"
probe_rc=$?
set -e
[[ "$probe_rc" -eq 0 ]] || {
  printf 'probe failed with exit %s\n' "$probe_rc" >&2
  exit 74
}
[[ "$(grep -Fxc "WORKER_SOURCE_SHA256=$source_hash" "$work/ledger.txt")" == 1 ]] || {
  printf 'ledger source binding is absent or duplicated\n' >&2
  exit 74
}
[[ "$(grep -Fxc "INPUT_SHA256=$input_hash" "$work/ledger.txt")" == 1 ]] || {
  printf 'ledger input binding is absent or duplicated\n' >&2
  exit 74
}
[[ "$(grep -Fxc "RUN_CHALLENGE=$challenge" "$work/ledger.txt")" == 1 ]] || {
  printf 'ledger challenge binding is absent or duplicated\n' >&2
  exit 74
}
[[ "$(grep -Fxc "FLATTENED_BASELINE_RECEIPT_SHA256=$expected_flattened_receipt_sha256" \
  "$work/ledger.txt")" == 1 ]] || {
  printf 'ledger flattened receipt binding is absent or duplicated\n' >&2
  exit 74
}
[[ "$(grep -Fxc "FLATTENED_BASELINE_PHYSICAL_CHAIN_SHA256=$expected_flattened_physical_sha256" \
  "$work/ledger.txt")" == 1 ]] || {
  printf 'ledger flattened physical binding is absent or duplicated\n' >&2
  exit 74
}

receipt_hash="$(sha256sum "$work/ledger.txt" | awk '{print $1}')"
"$python_path" "$work/verifier.py" "$work/ledger.txt" \
  --expected-source-sha256 "$source_hash" \
  --expected-input-sha256 "$input_hash" \
  --expected-run-challenge "$challenge" \
  --expected-receipt-sha256 "$receipt_hash" \
  --expected-baseline-receipt-sha256 "$expected_flattened_receipt_sha256" \
  --expected-flattened-physical-sha256 "$expected_flattened_physical_sha256" \
  > "$work/verification.txt"

physical_chain_hash="$(awk -F= '$1 == "PHYSICAL_CHAIN_SHA256" {print $2}' \
  "$work/verification.txt")"
comparison_chain_hash="$(awk -F= '$1 == "COMPARISON_CHAIN_SHA256" {print $2}' \
  "$work/verification.txt")"
flattened_physical_hash="$(awk -F= '$1 == "FLATTENED_PHYSICAL_CHAIN_SHA256" {print $2}' \
  "$work/verification.txt")"
[[ "$physical_chain_hash" =~ ^[0-9a-f]{64}$ ]] || {
  printf 'verifier emitted no canonical physical digest\n' >&2
  exit 74
}
[[ "$comparison_chain_hash" =~ ^[0-9a-f]{64}$ ]] || {
  printf 'verifier emitted no canonical comparison digest\n' >&2
  exit 74
}
[[ "$flattened_physical_hash" == "$expected_flattened_physical_sha256" ]] || {
  printf 'verifier flattened physical digest mismatch\n' >&2
  exit 74
}
[[ "$(grep -Fxc "PHYSICAL_CHAIN_SHA256=$physical_chain_hash" \
  "$work/verification.txt")" == 1 ]] || {
  printf 'verifier physical digest is absent or duplicated\n' >&2
  exit 74
}
[[ "$(grep -Fxc "COMPARISON_CHAIN_SHA256=$comparison_chain_hash" \
  "$work/verification.txt")" == 1 ]] || {
  printf 'verifier comparison digest is absent or duplicated\n' >&2
  exit 74
}
[[ "$(grep -Fxc "FLATTENED_PHYSICAL_CHAIN_SHA256=$flattened_physical_hash" \
  "$work/verification.txt")" == 1 ]] || {
  printf 'verifier flattened physical digest is absent or duplicated\n' >&2
  exit 74
}

: > "$work/dependencies-after-run.sha256"
while IFS= read -r dependency; do
  [[ -f "$dependency" ]] && sha256sum "$dependency"
done < "$work/dependency-paths.txt" >> "$work/dependencies-after-run.sha256"
sort -u "$work/dependencies-after-run.sha256" \
  -o "$work/dependencies-after-run.sha256"
cmp -s "$work/dependencies-before-run.sha256" \
  "$work/dependencies-after-run.sha256" || {
  printf 'compiler dependencies changed during run\n' >&2
  exit 74
}
sed "s#  $work/#  BUNDLE/#" "$work/dependencies-after-run.sha256" \
  > "$work/dependencies-retained.sha256"

: > "$work/link-inputs-after-run.sha256"
for argument in "${libs[@]}"; do
  [[ -f "$argument" ]] && sha256sum "$argument" \
    >> "$work/link-inputs-after-run.sha256"
done
cmp -s "$work/link-inputs-before-run.sha256" \
  "$work/link-inputs-after-run.sha256" || {
  printf 'link inputs changed during run\n' >&2
  exit 74
}

: > "$work/runtime-libraries-after-run.sha256"
while IFS= read -r library; do
  [[ -f "$library" ]] && sha256sum "$library"
done < "$work/runtime-library-paths.txt" \
  >> "$work/runtime-libraries-after-run.sha256"
cmp -s "$work/runtime-libraries-before-run.sha256" \
  "$work/runtime-libraries-after-run.sha256" || {
  printf 'runtime libraries changed during run\n' >&2
  exit 74
}

for binding in \
  "$source_file:$source_hash:source" \
  "$verifier_file:$verifier_hash:verifier" \
  "$runner_file:$runner_hash:runner" \
  "$flattened_receipt_file:$flattened_receipt_hash:flattened baseline receipt" \
  "$flattened_provenance_file:$flattened_provenance_hash:flattened baseline provenance" \
  "$flattened_verifier_file:$flattened_verifier_hash:flattened baseline verifier"; do
  path="${binding%%:*}"
  remainder="${binding#*:}"
  expected_hash="${remainder%%:*}"
  label="${remainder#*:}"
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected_hash" ]] || {
    printf '%s changed during run\n' "$label" >&2
    exit 74
  }
done

{
  printf 'MANIFEST_KIND=CS6_SECTION_RESIDENT_RECONDITIONED_TWO_RETURN_V1\n'
  printf 'RUN_COMPLETE=true\n'
  printf 'WORKER_EXIT=0\n'
  printf 'SOURCE_SHA256=%s\n' "$source_hash"
  printf 'VERIFIER_SHA256=%s\n' "$verifier_hash"
  printf 'RUNNER_SHA256=%s\n' "$runner_hash"
  printf 'INPUT_SHA256=%s\n' "$input_hash"
  printf 'RUN_CHALLENGE=%s\n' "$challenge"
  printf 'EXECUTABLE_SHA256=%s\n' "$(sha256sum "$work/probe-binary" | awk '{print $1}')"
  printf 'RECEIPT_SHA256=%s\n' "$receipt_hash"
  printf 'VERIFICATION_SHA256=%s\n' "$(sha256sum "$work/verification.txt" | awk '{print $1}')"
  printf 'PHYSICAL_CHAIN_SHA256=%s\n' "$physical_chain_hash"
  printf 'COMPARISON_CHAIN_SHA256=%s\n' "$comparison_chain_hash"
  printf 'FLATTENED_BASELINE_RECEIPT_SHA256=%s\n' "$flattened_receipt_hash"
  printf 'FLATTENED_BASELINE_PROVENANCE_SHA256=%s\n' "$flattened_provenance_hash"
  printf 'FLATTENED_BASELINE_VERIFIER_SHA256=%s\n' "$flattened_verifier_hash"
  printf 'FLATTENED_BASELINE_PHYSICAL_SHA256=%s\n' "$expected_flattened_physical_sha256"
  printf 'CAPD_CONFIG_SHA256=%s\n' "$(sha256sum "$work/capd-config-retained" | awk '{print $1}')"
  printf 'CAPD_CFLAGS_SHA256=%s\n' "$(sha256sum "$work/capd-cflags.txt" | awk '{print $1}')"
  printf 'CAPD_LIBS_SHA256=%s\n' "$(sha256sum "$work/capd-libs.txt" | awk '{print $1}')"
  printf 'CAPD_VERSION_SHA256=%s\n' "$(sha256sum "$work/capd-version.txt" | awk '{print $1}')"
  printf 'PREPROCESSOR_MACROS_SHA256=%s\n' "$(sha256sum "$work/preprocessor-macros.txt" | awk '{print $1}')"
  printf 'EFFECTIVE_OPTIONS_SHA256=%s\n' "$(sha256sum "$work/effective-options.txt" | awk '{print $1}')"
  printf 'EFFECTIVE_OPTIONS_STDERR_SHA256=%s\n' "$(sha256sum "$work/effective-options-stderr.txt" | awk '{print $1}')"
  printf 'DEPENDENCY_PATHS_SHA256=%s\n' "$(sha256sum "$work/dependency-paths.txt" | awk '{print $1}')"
  printf 'DEPENDENCIES_SHA256=%s\n' "$(sha256sum "$work/dependencies-retained.sha256" | awk '{print $1}')"
  printf 'LINK_INPUTS_SHA256=%s\n' "$(sha256sum "$work/link-inputs-after-run.sha256" | awk '{print $1}')"
  printf 'RUNTIME_LIBRARIES_SHA256=%s\n' "$(sha256sum "$work/runtime-libraries-after-run.sha256" | awk '{print $1}')"
  printf 'COMPILER_SHA256=%s\n' "$(sha256sum "$cxx_path" | awk '{print $1}')"
  printf 'COMPILER_VERSION_SHA256=%s\n' "$(sha256sum "$work/compiler-version.txt" | awk '{print $1}')"
  printf 'PYTHON_SHA256=%s\n' "$(sha256sum "$python_path" | awk '{print $1}')"
  printf 'PYTHON_VERSION_SHA256=%s\n' "$(sha256sum "$work/python-version.txt" | awk '{print $1}')"
  printf 'RUNTIME_LINKAGE_SHA256=%s\n' "$(sha256sum "$work/runtime-linkage.txt" | awk '{print $1}')"
  printf 'COMPILE_COMMAND_SHA256=%s\n' "$(sha256sum "$work/compile-command.txt" | awk '{print $1}')"
  printf 'COMPILE_STDERR_SHA256=%s\n' "$(sha256sum "$work/compile-stderr.txt" | awk '{print $1}')"
  printf 'PROBE_STDERR_SHA256=%s\n' "$(sha256sum "$work/probe-stderr.txt" | awk '{print $1}')"
  printf 'GIT_HEAD=%s\n' "$(cat "$work/git-head.txt")"
  printf 'GIT_STATUS_CLEAN=%s\n' "$([[ ! -s "$work/git-status.txt" ]] && printf true || printf false)"
  printf 'CAPD_VERSION=5.3.0\n'
  printf 'INTERVAL_BACKEND=FILIB\n'
  printf 'OPTIMIZATION_LEVEL=O0\n'
  printf 'ROUNDING_MATH_EFFECTIVE=true\n'
  printf 'DEPENDENCIES_STABLE_DURING_RUN=true\n'
  printf 'FLATTENED_BASELINE_STABLE_DURING_RUN=true\n'
  printf 'DEPENDENCY_CONTENT_HASHES_COMPLETE=false\n'
  printf 'EXECUTION_PROVENANCE_ATTESTED=false\n'
  printf 'INDEPENDENT_REPLAY_REQUIRED=true\n'
  printf 'PROMOTION_ELIGIBLE=false\n'
} > "$work/run-manifest.txt"

find "$work" -maxdepth 1 -type f ! -name bundle-index.sha256 -print0 \
  | sort -z | xargs -0 sha256sum | sed "s#  $work/#  #" \
  > "$work/bundle-index.sha256"
mv "$work" "$run_dir"
trap - EXIT
printf 'RUN_DIR=%s\n' "$run_dir"
printf 'RECEIPT_SHA256=%s\n' "$receipt_hash"
printf 'PHYSICAL_CHAIN_SHA256=%s\n' "$physical_chain_hash"
printf 'COMPARISON_CHAIN_SHA256=%s\n' "$comparison_chain_hash"
printf 'VERIFY_PASS=true\n'
