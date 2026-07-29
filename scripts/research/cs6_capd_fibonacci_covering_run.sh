#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/scripts/research/cs6_capd_fibonacci_covering.cpp"

run_dir=""
shards=12
jobs=12
capd_config="${CS6_CAPD_CONFIG:-capd-config}"
cxx="${CXX:-c++}"

usage() {
  cat <<'EOF'
Usage: cs6_capd_fibonacci_covering_run.sh --run-dir DIR [options]

Run inside an authorised Foundry/Slurm CPU allocation. This script does not
submit a job.

Options:
  --run-dir DIR       New artifact directory (required; must not exist)
  --shards N          Complete partition count (default: 12)
  --jobs N            Concurrent local workers inside allocation (default: 12)
  --capd-config PATH  Pinned capd-config executable
  --cxx PATH          C++ compiler (default: $CXX or c++)
EOF
}

positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      run_dir="${2:?--run-dir requires a value}"
      shift 2
      ;;
    --shards)
      shards="${2:?--shards requires a value}"
      shift 2
      ;;
    --jobs)
      jobs="${2:?--jobs requires a value}"
      shift 2
      ;;
    --capd-config)
      capd_config="${2:?--capd-config requires a value}"
      shift 2
      ;;
    --cxx)
      cxx="${2:?--cxx requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$run_dir" ]] || { usage >&2; exit 2; }
positive_int "$shards" || { echo "invalid --shards: $shards" >&2; exit 2; }
positive_int "$jobs" || { echo "invalid --jobs: $jobs" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "refusing proof run outside a Slurm allocation" >&2
  exit 3
}
command -v "$capd_config" >/dev/null 2>&1 || {
  echo "capd-config unavailable: $capd_config" >&2
  exit 3
}
command -v "$cxx" >/dev/null 2>&1 || {
  echo "C++ compiler unavailable: $cxx" >&2
  exit 3
}
[[ ! -e "$run_dir" ]] || {
  echo "refusing existing run directory: $run_dir" >&2
  exit 2
}

mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
snapshot="$run_dir/proof-source.cpp"
binary="$run_dir/proof-binary"
manifest="$run_dir/run-manifest.txt"
config_snapshot="$run_dir/capd-config-retained"
compiler_snapshot="$run_dir/compiler-driver-retained"
cflags_file="$run_dir/capd-cflags.txt"
libs_file="$run_dir/capd-libs.txt"
compiler_version_file="$run_dir/compiler-version.txt"
cp "$source_file" "$snapshot"

capd_config_real="$(readlink -f "$(command -v "$capd_config")")"
cxx_real="$(readlink -f "$(command -v "$cxx")")"
capd_cflags="$($capd_config --cflags)"
capd_libs="$($capd_config --libs)"
cp "$capd_config_real" "$config_snapshot"
cp "$cxx_real" "$compiler_snapshot"
printf '%s' "$capd_cflags" > "$cflags_file"
printf '%s' "$capd_libs" > "$libs_file"

# capd-config intentionally emits compiler and linker arguments.
# shellcheck disable=SC2086
"$cxx" -std=c++17 -O2 "$snapshot" $capd_cflags $capd_libs -o "$binary"

source_sha="$(sha256sum "$snapshot" | awk '{print $1}')"
binary_sha="$(sha256sum "$binary" | awk '{print $1}')"
config_sha="$(sha256sum "$capd_config_real" | awk '{print $1}')"
cflags_sha="$(sha256sum "$cflags_file" | awk '{print $1}')"
libs_sha="$(sha256sum "$libs_file" | awk '{print $1}')"
cxx_sha="$(sha256sum "$compiler_snapshot" | awk '{print $1}')"
cxx_version="$($cxx --version | head -n 1)"
printf '%s' "$cxx_version" > "$compiler_version_file"
cxx_version_sha="$(sha256sum "$compiler_version_file" | awk '{print $1}')"

write_manifest() {
  local complete="$1"
  cat > "$manifest" <<EOF
MANIFEST_KIND=CS6_CAPD_FIBONACCI_RUN_V1
RUN_COMPLETE=$complete
SOURCE_SHA256=$source_sha
EXECUTABLE_SHA256=$binary_sha
CAPD_CONFIG_SHA256=$config_sha
CAPD_CFLAGS_SHA256=$cflags_sha
CAPD_LIBS_SHA256=$libs_sha
CXX_DRIVER_SHA256=$cxx_sha
CXX_VERSION_SHA256=$cxx_version_sha
CAPD_CONFIG_PATH=$capd_config_real
CXX_PATH=$cxx_real
CXX_VERSION=$cxx_version
SLURM_JOB_ID=$SLURM_JOB_ID
EXECUTION_TRUST_MODEL=AUTHORIZED_FOUNDRY_SLURM_CPU_TCB_NO_ATTESTATION
REMOTE_ATTESTATION_PRESENT=false
INDEPENDENT_REPLAY_REQUIRED=true
GRID=N0_U:200,N1_U:75,SUPPORT_S:75,EXIT_S:1200
ORDER=8
SHARDS=$shards
EOF
}

write_manifest false
export binary run_dir shards
seq 1 "$shards" | xargs -P "$jobs" -n 1 sh -c '
  ordinal="$1"
  "$binary" 200 75 75 1200 8 "$ordinal" "$shards" \
    "$run_dir/ledger-$ordinal.txt" > "$run_dir/shard-$ordinal.txt"
' _

test "$(sha256sum "$snapshot" | awk '{print $1}')" = "$source_sha"
test "$(sha256sum "$binary" | awk '{print $1}')" = "$binary_sha"
for ordinal in $(seq 1 "$shards"); do
  test -s "$run_dir/shard-$ordinal.txt"
  test -s "$run_dir/ledger-$ordinal.txt"
done
write_manifest true
echo "CS6_CAPD_FIBONACCI_RUN PASS run_dir=$run_dir shards=$shards"
