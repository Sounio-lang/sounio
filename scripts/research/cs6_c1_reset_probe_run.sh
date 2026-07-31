#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/scripts/research/cs6_c1_reset_probe.cpp"
verifier="$repo_root/scripts/research/cs6_c1_reset_verify.py"
runner_file="$repo_root/scripts/research/cs6_c1_reset_probe_run.sh"

capd_config=""
run_dir=""
cxx="${CXX:-g++}"
source_set="N0"
target_set="N0"
u_index="20000"
s_index="15000"
u_tiles="40000"
s_tiles="30000"
order="8"
expect_rebox_worse=false
expect_c0_nontransversal_failure=false

usage() {
  printf '%s\n' \
    "usage: $0 --capd-config PATH --run-dir DIR [options]" \
    "  --cxx PATH" \
    "  --source N0|N1 --target N0|N1" \
    "  --u-index N --s-index N --u-tiles N --s-tiles N --order N" \
    "  --expect-rebox-worse" \
    "  --expect-c0-nontransversal-failure"
}

while (($#)); do
  case "$1" in
    --capd-config) capd_config="${2:-}"; shift 2 ;;
    --run-dir) run_dir="${2:-}"; shift 2 ;;
    --cxx) cxx="${2:-}"; shift 2 ;;
    --source) source_set="${2:-}"; shift 2 ;;
    --target) target_set="${2:-}"; shift 2 ;;
    --u-index) u_index="${2:-}"; shift 2 ;;
    --s-index) s_index="${2:-}"; shift 2 ;;
    --u-tiles) u_tiles="${2:-}"; shift 2 ;;
    --s-tiles) s_tiles="${2:-}"; shift 2 ;;
    --order) order="${2:-}"; shift 2 ;;
    --expect-rebox-worse) expect_rebox_worse=true; shift ;;
    --expect-c0-nontransversal-failure) expect_c0_nontransversal_failure=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
done

[[ -n "$capd_config" && -n "$run_dir" ]] || { usage >&2; exit 64; }
[[ "$expect_rebox_worse" != true || "$expect_c0_nontransversal_failure" != true ]] || {
  printf 'expected-outcome flags are mutually exclusive\n' >&2
  exit 64
}
[[ "$source_set" == N0 || "$source_set" == N1 ]] || { printf 'invalid source\n' >&2; exit 64; }
[[ "$target_set" == N0 || "$target_set" == N1 ]] || { printf 'invalid target\n' >&2; exit 64; }
[[ "$source_set" != N1 || "$target_set" == N0 ]] || { printf 'edge outside frozen adjacency\n' >&2; exit 64; }
for value in "$u_index" "$s_index" "$u_tiles" "$s_tiles" "$order"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { printf 'non-integer probe coordinate: %s\n' "$value" >&2; exit 64; }
done
((u_tiles > 0 && s_tiles > 0 && order > 0 && u_index < u_tiles && s_index < s_tiles)) || {
  printf 'invalid tile or order\n' >&2
  exit 64
}

capd_config="$(realpath "$capd_config")"
[[ -x "$capd_config" ]] || { printf 'CAPD config is not executable: %s\n' "$capd_config" >&2; exit 66; }
[[ -f "$source_file" && -f "$verifier" ]] || { printf 'probe sources missing\n' >&2; exit 66; }
cxx_path="$(command -v "$cxx")"
cxx_path="$(realpath "$cxx_path")"
python_path="$(realpath "$(command -v python3)")"
run_dir="$(realpath -m "$run_dir")"
[[ ! -e "$run_dir" ]] || { printf 'run directory already exists: %s\n' "$run_dir" >&2; exit 73; }
parent="$(dirname "$run_dir")"
mkdir -p "$parent"
work="$(mktemp -d "$parent/.cs6-c1-reset.XXXXXX")"
trap 'rm -rf "$work"' EXIT

cp "$source_file" "$work/probe-source.cpp"
cp "$verifier" "$work/verifier.py"
cp "$runner_file" "$work/runner.sh"
cp "$capd_config" "$work/capd-config-retained"
"$capd_config" --cflags > "$work/capd-cflags.txt"
"$capd_config" --libs > "$work/capd-libs.txt"
"$capd_config" --modversion > "$work/capd-version.txt"
cp "$cxx_path" "$work/compiler-driver-retained"
"$cxx_path" --version > "$work/compiler-version.txt"
cp "$python_path" "$work/python-driver-retained"
"$python_path" --version > "$work/python-version.txt" 2>&1
git -C "$repo_root" rev-parse HEAD > "$work/git-head.txt"
git -C "$repo_root" status --short --untracked-files=all > "$work/git-status.txt"

grep -Fxq '5.3.0' "$work/capd-version.txt" || {
  printf 'CAPD version is not the frozen 5.3.0\n' >&2
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

read -r -a cflags < "$work/capd-cflags.txt"
read -r -a libs < "$work/capd-libs.txt"
"$cxx_path" "${cflags[@]}" -dM -E -x c++ /dev/null \
  > "$work/capd-preprocessor-macros.txt"
[[ "$(grep -Ec '^#define __USE_FILIB__( 1)?$' "$work/capd-preprocessor-macros.txt")" == 1 ]] || {
  printf 'effective compiler state does not select FILIB\n' >&2
  exit 65
}
"$cxx_path" "${cflags[@]}" -Q --help=optimizers -c -x c++ /dev/null \
  -o "$work/effective-options.o" > "$work/cxx-effective-options.txt" \
  2> "$work/cxx-effective-options-stderr.txt"
rm -f "$work/effective-options.o"
grep -Eq '^[[:space:]]*-frounding-math[[:space:]]+\[enabled\]' \
  "$work/cxx-effective-options.txt" || {
  printf 'effective compiler state disables -frounding-math\n' >&2
  exit 65
}
snapshot_hash="$(sha256sum "$work/probe-source.cpp" | awk '{print $1}')"
"$cxx_path" -std=c++17 -O2 \
  "-DCS6_WORKER_SOURCE_SHA256=\"$snapshot_hash\"" \
  "${cflags[@]}" "$work/probe-source.cpp" \
  -o "$work/probe-binary.tmp" "${libs[@]}" 2> "$work/compile-stderr.txt"
mv "$work/probe-binary.tmp" "$work/probe-binary"
ldd "$work/probe-binary" > "$work/runtime-linkage.txt"

: > "$work/capd-libraries.sha256"
for argument in "${libs[@]}"; do
  if [[ -f "$argument" ]]; then
    sha256sum "$argument" >> "$work/capd-libraries.sha256"
  fi
done
[[ -s "$work/capd-libraries.sha256" ]] || {
  printf 'CAPD library flags contain no hashable files\n' >&2
  exit 65
}
awk '/=> \// {print $3} /^\// {print $1}' "$work/runtime-linkage.txt" | \
  while IFS= read -r library; do
    [[ -f "$library" ]] && sha256sum "$library"
  done > "$work/runtime-libraries.sha256"

: > "$work/capd-headers.sha256"
for argument in "${cflags[@]}"; do
  if [[ "$argument" == -I* && -d "${argument#-I}" ]]; then
    find "${argument#-I}" -type f -print0 | sort -z | xargs -0 -r sha256sum \
      >> "$work/capd-headers.sha256"
  fi
done
sort -u "$work/capd-headers.sha256" -o "$work/capd-headers.sha256"
[[ -s "$work/capd-headers.sha256" ]] || {
  printf 'CAPD include flags contain no hashable headers\n' >&2
  exit 65
}

set +e
"$work/probe-binary" probe "$source_set" "$target_set" \
  "$u_index" "$s_index" "$u_tiles" "$s_tiles" "$order" \
  > "$work/ledger.txt" 2> "$work/probe-stderr.txt"
probe_rc=$?
set -e
source_binding_count="$(grep -Fxc "WORKER_SOURCE_SHA256=$snapshot_hash" "$work/ledger.txt" || true)"
[[ "$source_binding_count" == 1 ]] || {
  printf 'worker ledger is not bound to the compiled source snapshot\n' >&2
  exit 74
}

if [[ "$expect_c0_nontransversal_failure" == true ]]; then
  [[ "$probe_rc" -eq 2 ]] || {
    printf 'expected worker exit 2 for C0 nontransversality, got %s\n' "$probe_rc" >&2
    exit 74
  }
  if "$python_path" "$work/verifier.py" "$work/ledger.txt" > /dev/null 2>&1; then
    printf 'ordinary verifier accepted expected-failure ledger\n' >&2
    exit 74
  fi
  "$python_path" "$work/verifier.py" "$work/ledger.txt" \
    --expect-c0-nontransversal-failure > "$work/verification.txt"
  printf 'WORKER_EXIT=2\n' >> "$work/verification.txt"
  expected_outcome="C0_NONTRANSVERSAL_FAILURE"
else
  [[ "$probe_rc" -eq 0 ]] || {
    printf 'worker failed unexpectedly with exit %s\n' "$probe_rc" >&2
    exit 74
  }
  verify_args=("$work/verifier.py" "$work/ledger.txt")
  if [[ "$expect_rebox_worse" == true ]]; then
    verify_args+=(--expect-rebox-worse)
  fi
  "$python_path" "${verify_args[@]}" > "$work/verification.txt"
  expected_outcome="PASS"
fi

source_hash="$(sha256sum "$source_file" | awk '{print $1}')"
verifier_hash="$(sha256sum "$verifier" | awk '{print $1}')"
verifier_snapshot_hash="$(sha256sum "$work/verifier.py" | awk '{print $1}')"
runner_hash="$(sha256sum "$runner_file" | awk '{print $1}')"
runner_snapshot_hash="$(sha256sum "$work/runner.sh" | awk '{print $1}')"
[[ "$source_hash" == "$snapshot_hash" ]] || { printf 'source changed during run\n' >&2; exit 74; }
[[ "$verifier_hash" == "$verifier_snapshot_hash" ]] || { printf 'verifier changed during run\n' >&2; exit 74; }
[[ "$runner_hash" == "$runner_snapshot_hash" ]] || { printf 'runner changed during run\n' >&2; exit 74; }
[[ "$(sha256sum "$capd_config" | awk '{print $1}')" == "$(sha256sum "$work/capd-config-retained" | awk '{print $1}')" ]] || {
  printf 'CAPD config changed during run\n' >&2
  exit 74
}
[[ "$(sha256sum "$cxx_path" | awk '{print $1}')" == "$(sha256sum "$work/compiler-driver-retained" | awk '{print $1}')" ]] || {
  printf 'compiler changed during run\n' >&2
  exit 74
}
[[ "$(sha256sum "$python_path" | awk '{print $1}')" == "$(sha256sum "$work/python-driver-retained" | awk '{print $1}')" ]] || {
  printf 'Python changed during run\n' >&2
  exit 74
}

{
  printf 'MANIFEST_KIND=CS6_C1_RESET_BOUNDED_RUN_V1\n'
  printf 'RUN_COMPLETE=true\n'
  printf 'SOURCE_SHA256=%s\n' "$snapshot_hash"
  printf 'VERIFIER_SHA256=%s\n' "$verifier_snapshot_hash"
  printf 'RUNNER_SHA256=%s\n' "$runner_snapshot_hash"
  printf 'EXECUTABLE_SHA256=%s\n' "$(sha256sum "$work/probe-binary" | awk '{print $1}')"
  printf 'LEDGER_SHA256=%s\n' "$(sha256sum "$work/ledger.txt" | awk '{print $1}')"
  printf 'VERIFICATION_SHA256=%s\n' "$(sha256sum "$work/verification.txt" | awk '{print $1}')"
  printf 'CAPD_CONFIG_SHA256=%s\n' "$(sha256sum "$work/capd-config-retained" | awk '{print $1}')"
  printf 'CAPD_CFLAGS_SHA256=%s\n' "$(sha256sum "$work/capd-cflags.txt" | awk '{print $1}')"
  printf 'CAPD_LIBS_SHA256=%s\n' "$(sha256sum "$work/capd-libs.txt" | awk '{print $1}')"
  printf 'CAPD_VERSION_SHA256=%s\n' "$(sha256sum "$work/capd-version.txt" | awk '{print $1}')"
  printf 'CAPD_PREPROCESSOR_MACROS_SHA256=%s\n' "$(sha256sum "$work/capd-preprocessor-macros.txt" | awk '{print $1}')"
  printf 'CXX_EFFECTIVE_OPTIONS_SHA256=%s\n' "$(sha256sum "$work/cxx-effective-options.txt" | awk '{print $1}')"
  printf 'CXX_EFFECTIVE_OPTIONS_STDERR_SHA256=%s\n' "$(sha256sum "$work/cxx-effective-options-stderr.txt" | awk '{print $1}')"
  printf 'CXX_DRIVER_SHA256=%s\n' "$(sha256sum "$work/compiler-driver-retained" | awk '{print $1}')"
  printf 'CXX_VERSION_SHA256=%s\n' "$(sha256sum "$work/compiler-version.txt" | awk '{print $1}')"
  printf 'RUNTIME_LINKAGE_SHA256=%s\n' "$(sha256sum "$work/runtime-linkage.txt" | awk '{print $1}')"
  printf 'CAPD_LIBRARY_MANIFEST_SHA256=%s\n' "$(sha256sum "$work/capd-libraries.sha256" | awk '{print $1}')"
  printf 'CAPD_HEADER_MANIFEST_SHA256=%s\n' "$(sha256sum "$work/capd-headers.sha256" | awk '{print $1}')"
  printf 'RUNTIME_LIBRARY_MANIFEST_SHA256=%s\n' "$(sha256sum "$work/runtime-libraries.sha256" | awk '{print $1}')"
  printf 'COMPILE_STDERR_SHA256=%s\n' "$(sha256sum "$work/compile-stderr.txt" | awk '{print $1}')"
  printf 'PROBE_STDERR_SHA256=%s\n' "$(sha256sum "$work/probe-stderr.txt" | awk '{print $1}')"
  printf 'PYTHON_DRIVER_SHA256=%s\n' "$(sha256sum "$work/python-driver-retained" | awk '{print $1}')"
  printf 'PYTHON_VERSION_SHA256=%s\n' "$(sha256sum "$work/python-version.txt" | awk '{print $1}')"
  printf 'GIT_HEAD_SHA256=%s\n' "$(sha256sum "$work/git-head.txt" | awk '{print $1}')"
  printf 'GIT_STATUS_SHA256=%s\n' "$(sha256sum "$work/git-status.txt" | awk '{print $1}')"
  printf 'CAPD_CONFIG_PATH=%s\n' "$capd_config"
  printf 'CXX_PATH=%s\n' "$cxx_path"
  printf 'PYTHON_PATH=%s\n' "$python_path"
  printf 'SOURCE=%s\nTARGET=%s\n' "$source_set" "$target_set"
  printf 'U_INDEX=%s\nS_INDEX=%s\nU_TILES=%s\nS_TILES=%s\nORDER=%s\n' \
    "$u_index" "$s_index" "$u_tiles" "$s_tiles" "$order"
  printf 'PROBE_EXIT=%s\n' "$probe_rc"
  printf 'EXPECTED_OUTCOME=%s\n' "$expected_outcome"
  printf 'EXPECT_REBOX_WORSE=%s\n' "$expect_rebox_worse"
  printf 'DEPENDENCY_CONTENT_HASHES_COMPLETE=false\n'
  printf 'EXECUTION_TRUST_MODEL=LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION\n'
  printf 'REMOTE_ATTESTATION_PRESENT=false\n'
  printf 'INDEPENDENT_REPLAY_REQUIRED=true\n'
  printf 'PROMOTION_ELIGIBLE=false\n'
} > "$work/run-manifest.txt"

mv "$work" "$run_dir"
trap - EXIT
printf 'RUN_DIR=%s\n' "$run_dir"
cat "$run_dir/verification.txt"
