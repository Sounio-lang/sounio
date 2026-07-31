#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/scripts/research/cs6_c1_dependency_probe.cpp"
verifier_file="$repo_root/scripts/research/cs6_c1_dependency_verify.py"
runner_file="$repo_root/scripts/research/cs6_c1_dependency_run.sh"
baseline_receipt="$repo_root/scripts/research/cs6_section_resident_reconditioned_two_return_receipt_v1.txt"
baseline_provenance="$repo_root/scripts/research/cs6_section_resident_reconditioned_two_return_provenance_v1.txt"

expected_baseline_receipt="3d17e9b8ad09c9b253c56b181a4eab90c0390eb5582e3ca542ccb3dcc44f6956"
expected_baseline_provenance="22fad25dfa795b63f361d45cc9de1d10177b3f7cd812a75252d3f47b1438344d"
expected_baseline_physical="8b5073b5261708991597af9d784b2b1ad998f5355f92a659925ec3f3882b4e3e"

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
  "$baseline_receipt" "$baseline_provenance"; do
  [[ -f "$path" ]] || { printf 'missing runner input: %s\n' "$path" >&2; exit 66; }
done

[[ "$(sha256sum "$baseline_receipt" | awk '{print $1}')" == \
  "$expected_baseline_receipt" ]] || {
  printf 'baseline receipt hash mismatch\n' >&2
  exit 65
}
[[ "$(sha256sum "$baseline_provenance" | awk '{print $1}')" == \
  "$expected_baseline_provenance" ]] || {
  printf 'baseline provenance hash mismatch\n' >&2
  exit 65
}
grep -Fxq "PHYSICAL_CHAIN_SHA256=$expected_baseline_physical" \
  "$baseline_provenance" || {
  printf 'baseline physical digest mismatch\n' >&2
  exit 65
}

capd_config="$(realpath "$capd_config")"
[[ -x "$capd_config" ]] || { printf 'invalid capd-config\n' >&2; exit 66; }
cxx_path="$(realpath "$(command -v "$cxx")")"
python_path="$(realpath "$(command -v python3)")"
run_dir="$(realpath -m "$run_dir")"
[[ ! -e "$run_dir" ]] || { printf 'run directory already exists\n' >&2; exit 73; }
mkdir -p "$(dirname "$run_dir")"
work="$(mktemp -d "$(dirname "$run_dir")/.cs6-c1-dependency.XXXXXX")"
trap 'rm -rf "$work"' EXIT

cp "$source_file" "$work/probe-source.cpp"
cp "$verifier_file" "$work/verifier.py"
cp "$runner_file" "$work/runner.sh"
cp "$baseline_receipt" "$work/baseline-receipt.txt"
cp "$baseline_provenance" "$work/baseline-provenance.txt"

{
  printf 'INPUT_SCHEMA=sounio.cs6.c1-dependency-input.v1\n'
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
  printf 'ROUTE_A=C2_AFFINE_JACOBIAN_CARRIER\n'
  printf 'ROUTE_B=FINAL_COLUMN_PROJECTIVE_SLOPE_CONTROL\n'
  printf 'BASELINE_RECEIPT_SHA256=%s\n' "$expected_baseline_receipt"
  printf 'BASELINE_PHYSICAL_SHA256=%s\n' "$expected_baseline_physical"
} > "$work/input.txt"

"$capd_config" --cflags > "$work/capd-cflags.txt"
"$capd_config" --libs > "$work/capd-libs.txt"
"$capd_config" --modversion > "$work/capd-version.txt"
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
  printf 'CAPD config contains contradictory interval flags\n' >&2
  exit 65
}

read -r -a cflags < "$work/capd-cflags.txt"
read -r -a libs < "$work/capd-libs.txt"
"$cxx_path" "${cflags[@]}" -dM -E -x c++ /dev/null \
  > "$work/preprocessor-macros.txt"
sed -i 's/[[:space:]]\+$//' "$work/preprocessor-macros.txt"
grep -Eq '^#define __USE_FILIB__( 1)?$' "$work/preprocessor-macros.txt" || {
  printf 'effective preprocessor state does not select FILIB\n' >&2
  exit 65
}
"$cxx_path" "${cflags[@]}" -O0 -Q --help=optimizers -c -x c++ /dev/null \
  -o "$work/effective-options.o" > "$work/effective-options.txt" \
  2> "$work/effective-options-stderr.txt"
rm -f "$work/effective-options.o"
sed -i 's/[[:space:]]\+$//' "$work/effective-options.txt"
while [[ -s "$work/effective-options.txt" ]] &&
      [[ -z "$(tail -n 1 "$work/effective-options.txt")" ]]; do
  sed -i '$d' "$work/effective-options.txt"
done
grep -Eq '^[[:space:]]*-frounding-math[[:space:]]+\[enabled\]' \
  "$work/effective-options.txt" || {
  printf 'effective compiler state disables -frounding-math\n' >&2
  exit 65
}

"$cxx_path" --version > "$work/compiler-version.txt"
"$python_path" --version > "$work/python-version.txt" 2>&1
printf '%s\n' "$cxx_path" > "$work/compiler-path.txt"
printf '%s\n' "$python_path" > "$work/python-path.txt"
git -C "$repo_root" rev-parse HEAD > "$work/git-head.txt"
git -C "$repo_root" status --short --untracked-files=all > "$work/git-status.txt"

source_hash="$(sha256sum "$work/probe-source.cpp" | awk '{print $1}')"
verifier_hash="$(sha256sum "$work/verifier.py" | awk '{print $1}')"
runner_hash="$(sha256sum "$work/runner.sh" | awk '{print $1}')"
input_hash="$(sha256sum "$work/input.txt" | awk '{print $1}')"

compile_args=("$cxx_path" -std=c++17 "${cflags[@]}" -O0 \
  "-DCS6_WORKER_SOURCE_SHA256=\"$source_hash\"" \
  "-DCS6_INPUT_SHA256=\"$input_hash\"" \
  "-DCS6_RUN_CHALLENGE=\"$challenge\"" \
  "$work/probe-source.cpp" -MD -MF "$work/dependencies.d" \
  -o "$work/probe-binary.tmp" "${libs[@]}")
{
  printf '%q' "${compile_args[0]}"
  printf ' %q' "${compile_args[@]:1}"
  printf '\n'
} > "$work/compile-command.txt"
"${compile_args[@]}" 2> "$work/compile-stderr.txt"
mv "$work/probe-binary.tmp" "$work/probe-binary"

sed ':a;N;$!ba;s/\\\n/ /g' "$work/dependencies.d" | \
  sed 's/^[^:]*:[[:space:]]*//' | tr ' ' '\n' | sed '/^$/d' | sort -u \
  > "$work/dependency-paths.txt"
: > "$work/dependencies-before.sha256"
while IFS= read -r dependency; do
  if [[ "$dependency" == "$work/probe-source.cpp" ]]; then
    printf '%s  BUNDLE/probe-source.cpp\n' \
      "$(sha256sum "$dependency" | awk '{print $1}')"
  elif [[ -f "$dependency" ]]; then
    sha256sum "$dependency"
  fi
done < "$work/dependency-paths.txt" >> "$work/dependencies-before.sha256"
sort -u "$work/dependencies-before.sha256" -o "$work/dependencies-before.sha256"
[[ -s "$work/dependencies-before.sha256" ]] || {
  printf 'compiler emitted no hashable dependencies\n' >&2
  exit 65
}

: > "$work/link-inputs.sha256"
for argument in "${libs[@]}"; do
  [[ -f "$argument" ]] && sha256sum "$argument" >> "$work/link-inputs.sha256"
done
[[ -s "$work/link-inputs.sha256" ]] || {
  printf 'CAPD link flags contain no hashable inputs\n' >&2
  exit 65
}

ldd "$work/probe-binary" > "$work/runtime-linkage.txt"
awk '/=> \// {print $3} /^\// {print $1}' "$work/runtime-linkage.txt" | \
  sort -u > "$work/runtime-library-paths.txt"
: > "$work/runtime-libraries.sha256"
while IFS= read -r library; do
  [[ -f "$library" ]] && sha256sum "$library"
done < "$work/runtime-library-paths.txt" >> "$work/runtime-libraries.sha256"

set +e
timeout 180 "$work/probe-binary" > "$work/ledger.txt" \
  2> "$work/probe-stderr.txt"
probe_rc=$?
set -e
[[ "$probe_rc" -eq 0 ]] || {
  printf 'probe failed with exit %s\n' "$probe_rc" >&2
  exit 74
}
[[ ! -s "$work/probe-stderr.txt" ]] || {
  printf 'probe emitted stderr\n' >&2
  exit 74
}

"$python_path" "$work/verifier.py" "$work/ledger.txt" \
  --source-sha "$source_hash" --input-sha "$input_hash" \
  --challenge "$challenge" --self-test-mutations \
  > "$work/verification.txt" 2> "$work/verification-stderr.txt"
[[ ! -s "$work/verification-stderr.txt" ]] || {
  printf 'verifier emitted stderr\n' >&2
  exit 65
}
grep -Fxq 'MUTATION_TESTS=39' "$work/verification.txt"
grep -Fxq 'MUTATIONS_REJECTED=39' "$work/verification.txt"
grep -Fxq 'CERTIFICATE_PASS=true' "$work/verification.txt"

: > "$work/dependencies-after.sha256"
while IFS= read -r dependency; do
  if [[ "$dependency" == "$work/probe-source.cpp" ]]; then
    printf '%s  BUNDLE/probe-source.cpp\n' \
      "$(sha256sum "$dependency" | awk '{print $1}')"
  elif [[ -f "$dependency" ]]; then
    sha256sum "$dependency"
  fi
done < "$work/dependency-paths.txt" >> "$work/dependencies-after.sha256"
sort -u "$work/dependencies-after.sha256" -o "$work/dependencies-after.sha256"
cmp -s "$work/dependencies-before.sha256" "$work/dependencies-after.sha256" || {
  printf 'compile dependency changed during execution\n' >&2
  exit 65
}
mv "$work/dependencies-after.sha256" "$work/dependencies.sha256"
rm "$work/dependencies-before.sha256"

receipt_hash="$(sha256sum "$work/ledger.txt" | awk '{print $1}')"
verification_hash="$(sha256sum "$work/verification.txt" | awk '{print $1}')"
physical_hash="$(awk -F= '$1=="PHYSICAL_SHA256" {print $2}' "$work/verification.txt")"
executable_hash="$(sha256sum "$work/probe-binary" | awk '{print $1}')"
dependencies_hash="$(sha256sum "$work/dependencies.sha256" | awk '{print $1}')"
link_inputs_hash="$(sha256sum "$work/link-inputs.sha256" | awk '{print $1}')"
runtime_libraries_hash="$(sha256sum "$work/runtime-libraries.sha256" | awk '{print $1}')"
dependency_count="$(wc -l < "$work/dependencies.sha256" | tr -d ' ')"
link_input_count="$(wc -l < "$work/link-inputs.sha256" | tr -d ' ')"
runtime_library_count="$(wc -l < "$work/runtime-libraries.sha256" | tr -d ' ')"

{
  printf 'MANIFEST_SCHEMA=sounio.cs6.c1-dependency-run-manifest.v1\n'
  printf 'RUN_COMPLETE=true\n'
  printf 'SOURCE_SHA256=%s\n' "$source_hash"
  printf 'VERIFIER_SHA256=%s\n' "$verifier_hash"
  printf 'RUNNER_SHA256=%s\n' "$runner_hash"
  printf 'INPUT_SHA256=%s\n' "$input_hash"
  printf 'RUN_CHALLENGE=%s\n' "$challenge"
  printf 'EXECUTABLE_SHA256=%s\n' "$executable_hash"
  printf 'RECEIPT_SHA256=%s\n' "$receipt_hash"
  printf 'VERIFICATION_SHA256=%s\n' "$verification_hash"
  printf 'PHYSICAL_SHA256=%s\n' "$physical_hash"
  printf 'DEPENDENCIES_SHA256=%s\n' "$dependencies_hash"
  printf 'DEPENDENCY_COUNT=%s\n' "$dependency_count"
  printf 'LINK_INPUTS_SHA256=%s\n' "$link_inputs_hash"
  printf 'LINK_INPUT_COUNT=%s\n' "$link_input_count"
  printf 'RUNTIME_LIBRARIES_SHA256=%s\n' "$runtime_libraries_hash"
  printf 'RUNTIME_LIBRARY_COUNT=%s\n' "$runtime_library_count"
  printf 'BASELINE_RECEIPT_SHA256=%s\n' "$expected_baseline_receipt"
  printf 'BASELINE_PROVENANCE_SHA256=%s\n' "$expected_baseline_provenance"
  printf 'BASELINE_PHYSICAL_SHA256=%s\n' "$expected_baseline_physical"
  printf 'CAPD_VERSION=5.3.0\n'
  printf 'INTERVAL_BACKEND=FILIB\n'
  printf 'OPTIMIZATION_LEVEL=O0\n'
  printf 'EXECUTION_TRUST_MODEL=LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION\n'
  printf 'REMOTE_ATTESTATION_PRESENT=false\n'
  printf 'INDEPENDENT_REPLAY_REQUIRED=true\n'
  printf 'PROMOTION_ELIGIBLE=false\n'
} > "$work/manifest.txt"

mv "$work" "$run_dir"
trap - EXIT
printf 'run_dir=%s\n' "$run_dir"
printf 'receipt_sha256=%s\n' "$receipt_hash"
printf 'physical_sha256=%s\n' "$physical_hash"
