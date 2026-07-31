#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_file="$repo_root/scripts/research/cs6_capd_c1_cone.cpp"
c0_source_file="$repo_root/scripts/research/cs6_capd_fibonacci_covering.cpp"
c0_aggregate_file="$repo_root/scripts/research/cs6_capd_fibonacci_covering_aggregate.py"

run_dir=""
c0_certificate=""
c0_run_dir=""
n0_u=""
n1_u=""
s_tiles=""
order=""
shards=12
jobs=12
capd_config="${CS6_CAPD_CONFIG:-capd-config}"
cxx="${CXX:-c++}"

usage() {
  cat <<'EOF'
Usage: cs6_capd_c1_cone_run.sh --run-dir DIR --c0-certificate FILE \
  --c0-run-dir DIR \
  --n0-u N --n1-u N --s-tiles N --order N [options]

Run a complete regular-grid C1 enclosure when the environment and Slurm control
plane agree on an active same-UID allocation containing the execution node.
Submission through the authorised Foundry route remains a separate operational
requirement; this script cannot attest submission origin or process membership.
It does not submit a job. Grid and order are deliberately explicit because the
inherited C0 grid is not a valid C1 default.

Options:
  --shards N          Complete partition count (default: 12)
  --jobs N            Concurrent workers in the allocation (default: 12)
  --capd-config PATH  Pinned capd-config executable
  --cxx PATH          C++ compiler (default: $CXX or c++)
EOF
}

positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) run_dir="${2:?missing --run-dir value}"; shift 2 ;;
    --c0-certificate) c0_certificate="${2:?missing --c0-certificate value}"; shift 2 ;;
    --c0-run-dir) c0_run_dir="${2:?missing --c0-run-dir value}"; shift 2 ;;
    --n0-u) n0_u="${2:?missing --n0-u value}"; shift 2 ;;
    --n1-u) n1_u="${2:?missing --n1-u value}"; shift 2 ;;
    --s-tiles) s_tiles="${2:?missing --s-tiles value}"; shift 2 ;;
    --order) order="${2:?missing --order value}"; shift 2 ;;
    --shards) shards="${2:?missing --shards value}"; shift 2 ;;
    --jobs) jobs="${2:?missing --jobs value}"; shift 2 ;;
    --capd-config) capd_config="${2:?missing --capd-config value}"; shift 2 ;;
    --cxx) cxx="${2:?missing --cxx value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$run_dir" && -n "$c0_certificate" && -n "$c0_run_dir" && -n "$n0_u" &&
   -n "$n1_u" && -n "$s_tiles" && -n "$order" ]] || {
  usage >&2
  exit 2
}
for value in "$n0_u" "$n1_u" "$s_tiles" "$order" "$shards" "$jobs"; do
  positive_int "$value" || { echo "expected positive integer: $value" >&2; exit 2; }
done
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "refusing exhaustive C1 run outside a Slurm allocation" >&2
  exit 3
}
command -v scontrol >/dev/null 2>&1 || {
  echo "cannot verify Slurm allocation: scontrol unavailable" >&2
  exit 3
}
if ! slurm_job_record="$(scontrol -o show job "$SLURM_JOB_ID" 2>/dev/null)"; then
  echo "cannot verify SLURM_JOB_ID against the Slurm control plane" >&2
  exit 3
fi
grep -Eq "(^| )JobId=${SLURM_JOB_ID}( |$)" <<<"$slurm_job_record" || {
  echo "Slurm control-plane record does not match SLURM_JOB_ID" >&2
  exit 3
}
grep -Eq '(^| )JobState=(RUNNING|COMPLETING)( |$)' <<<"$slurm_job_record" || {
  echo "Slurm job is not active" >&2
  exit 3
}
execution_uid="$(id -u)"
grep -Eq "(^| )UserId=[^ ]+\\(${execution_uid}\\)( |$)" \
  <<<"$slurm_job_record" || {
  echo "Slurm allocation UID does not match this process" >&2
  exit 3
}
slurm_nodelist="$(awk '{for (i=1; i<=NF; ++i) if ($i ~ /^NodeList=/) {sub(/^NodeList=/, "", $i); print $i}}' <<<"$slurm_job_record")"
[[ -n "$slurm_nodelist" && -n "${SLURMD_NODENAME:-}" ]] || {
  echo "Slurm allocation node metadata is unavailable" >&2
  exit 3
}
execution_node="$(hostname -s)"
[[ "$execution_node" == "$SLURMD_NODENAME" ]] || {
  echo "current host does not match SLURMD_NODENAME" >&2
  exit 3
}
slurm_hostnames="$(scontrol show hostnames "$slurm_nodelist")"
grep -Fxq "$execution_node" <<<"$slurm_hostnames" || {
  echo "current host is not in the Slurm allocation" >&2
  exit 3
}
slurm_version="$(scontrol --version)"
[[ -f "$c0_certificate" ]] || {
  echo "missing C0 aggregate certificate: $c0_certificate" >&2
  exit 3
}
[[ -d "$c0_run_dir" && -f "$c0_aggregate_file" ]] || {
  echo "missing C0 raw evidence bundle or aggregator" >&2
  exit 3
}
[[ -z "$(find "$c0_run_dir" -type l -print -quit)" ]] || {
  echo "C0 raw evidence bundle contains a symlink" >&2
  exit 3
}
grep -Fxq 'CERTIFICATE_KIND=CAPD_RIGOROUS_COVERING_AGGREGATE_V1' "$c0_certificate"
grep -Fxq 'CAPD_SOURCE_TREE_DECLARED=capd-5.3.0' "$c0_certificate"
grep -Fxq 'INTERVAL_BACKEND_DECLARED=FILIB' "$c0_certificate"
grep -Fxq 'MAP=P^6' "$c0_certificate"
grep -Fxq 'SECTION_ORIENTATION=MinusPlus' "$c0_certificate"
grep -Fxq 'GRID=N0_U:200,N1_U:75,SUPPORT_S:75,EXIT_S:1200' "$c0_certificate"
grep -Fxq 'ADJACENCY_MATRIX=[[1,1],[1,0]]' "$c0_certificate"
grep -Fxq 'FIBONACCI_COVERINGS_PROVED=true' "$c0_certificate"
grep -Fxq 'POSITIVE_ENTROPY_PROVED=true' "$c0_certificate"
[[ -f "$c0_source_file" ]]
c0_source_sha="$(sha256sum "$c0_source_file" | awk '{print $1}')"
grep -Fxq "SOURCE_SHA256=$c0_source_sha" "$c0_certificate"
c0_shards="$(sed -n 's/^SHARDS=//p' "$c0_certificate")"
positive_int "$c0_shards" || {
  echo "invalid C0 shard count" >&2
  exit 3
}
c0_check="$(mktemp -d)"
trap 'rm -rf "$c0_check"' EXIT
python3 "$c0_aggregate_file" \
  --run-dir "$c0_run_dir" \
  --shards "$c0_shards" \
  --source "$c0_source_file" \
  --ledger-output "$c0_check/ledger.txt" \
  --certificate-output "$c0_check/certificate.txt" >/dev/null
cmp -s "$c0_certificate" "$c0_check/certificate.txt" || {
  echo "C0 certificate does not match reaggregated raw evidence" >&2
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
c0_snapshot="$run_dir/c0-certificate.txt"
config_snapshot="$run_dir/capd-config-retained"
compiler_snapshot="$run_dir/compiler-driver-retained"
cflags_file="$run_dir/capd-cflags.txt"
libs_file="$run_dir/capd-libs.txt"
compiler_version_file="$run_dir/compiler-version.txt"
capd_version_file="$run_dir/capd-version.txt"
capd_pc_file="$run_dir/capd.pc"
capd_library_manifest="$run_dir/capd-libraries.sha256"
capd_header_manifest="$run_dir/capd-headers.sha256"
runtime_linkage_file="$run_dir/runtime-linkage.txt"
runtime_library_manifest="$run_dir/runtime-libraries.sha256"
slurm_job_file="$run_dir/slurm-job.txt"
slurm_version_file="$run_dir/slurm-version.txt"
slurm_hostnames_file="$run_dir/slurm-hostnames.txt"
c0_aggregator_snapshot="$run_dir/c0-aggregator.py"
c0_run_snapshot="$run_dir/c0-run"

cp "$source_file" "$snapshot"
cp "$c0_certificate" "$c0_snapshot"
cp "$c0_aggregate_file" "$c0_aggregator_snapshot"
mkdir "$c0_run_snapshot"
cp -a "$c0_run_dir/." "$c0_run_snapshot/"
[[ -z "$(find "$c0_run_snapshot" -type l -print -quit)" ]]
capd_config_real="$(readlink -f "$(command -v "$capd_config")")"
cxx_real="$(readlink -f "$(command -v "$cxx")")"
capd_cflags="$($capd_config --cflags)"
capd_libs="$($capd_config --libs)"
capd_version="$($capd_config --modversion)"
[[ "$capd_version" == "5.3.0" ]] || {
  echo "expected CAPD 5.3.0, got: $capd_version" >&2
  exit 3
}
[[ " $capd_cflags " == *" -D__USE_FILIB__ "* ]] || {
  echo "CAPD configuration does not select FILIB" >&2
  exit 3
}
capd_pc_real="$(readlink -f "$($capd_config --path)")"
[[ -f "$capd_pc_real" ]] || {
  echo "CAPD pkg-config contract unavailable" >&2
  exit 3
}
cp "$capd_config_real" "$config_snapshot"
cp "$cxx_real" "$compiler_snapshot"
printf '%s' "$capd_cflags" > "$cflags_file"
printf '%s' "$capd_libs" > "$libs_file"
printf '%s' "$($cxx --version | head -n 1)" > "$compiler_version_file"
printf '%s\n' "$capd_version" > "$capd_version_file"
cp "$capd_pc_real" "$capd_pc_file"
printf '%s\n' "$slurm_job_record" > "$slurm_job_file"
printf '%s\n' "$slurm_version" > "$slurm_version_file"
printf '%s\n' "$slurm_hostnames" > "$slurm_hostnames_file"

: > "$capd_library_manifest"
for library in $capd_libs; do
  [[ -f "$library" ]] && \
    sha256sum "$(readlink -f "$library")" >> "$capd_library_manifest"
done
test -s "$capd_library_manifest"
{
  for flag in $capd_cflags; do
    if [[ "$flag" == -I* ]]; then
      include_root="$(readlink -f "${flag#-I}")"
      [[ -d "$include_root" ]] || exit 3
      find "$include_root" -type f -print0
    fi
  done
} | sort -zu | xargs -0 -r sha256sum > "$capd_header_manifest"
test -s "$capd_header_manifest"

source_sha="$(sha256sum "$snapshot" | awk '{print $1}')"
c0_sha="$(sha256sum "$c0_snapshot" | awk '{print $1}')"
config_sha="$(sha256sum "$config_snapshot" | awk '{print $1}')"
cflags_sha="$(sha256sum "$cflags_file" | awk '{print $1}')"
libs_sha="$(sha256sum "$libs_file" | awk '{print $1}')"
cxx_sha="$(sha256sum "$compiler_snapshot" | awk '{print $1}')"
cxx_version_sha="$(sha256sum "$compiler_version_file" | awk '{print $1}')"
capd_version_sha="$(sha256sum "$capd_version_file" | awk '{print $1}')"
capd_pc_sha="$(sha256sum "$capd_pc_file" | awk '{print $1}')"
capd_library_manifest_sha="$(sha256sum "$capd_library_manifest" | awk '{print $1}')"
capd_header_manifest_sha="$(sha256sum "$capd_header_manifest" | awk '{print $1}')"
c0_aggregator_sha="$(sha256sum "$c0_aggregator_snapshot" | awk '{print $1}')"

# capd-config intentionally emits compiler and linker arguments. Inputs above
# are hashed before compilation and rechecked immediately afterwards.
test "$(sha256sum "$cxx_real" | awk '{print $1}')" = "$cxx_sha"
# shellcheck disable=SC2086
"$cxx_real" -std=c++17 -O2 "$snapshot" $capd_cflags $capd_libs -o "$binary"
test "$(sha256sum "$snapshot" | awk '{print $1}')" = "$source_sha"
test "$(sha256sum "$cxx_real" | awk '{print $1}')" = "$cxx_sha"
test "$(sha256sum "$compiler_snapshot" | awk '{print $1}')" = "$cxx_sha"
sha256sum --check --status "$capd_library_manifest"
sha256sum --check --status "$capd_header_manifest"

ldd "$binary" > "$runtime_linkage_file"
awk '/=> \// {print $3} /^[[:space:]]*\// {print $1}' "$runtime_linkage_file" | \
  sort -u | xargs -r sha256sum > "$runtime_library_manifest"
test -s "$runtime_library_manifest"

binary_sha="$(sha256sum "$binary" | awk '{print $1}')"
runtime_linkage_sha="$(sha256sum "$runtime_linkage_file" | awk '{print $1}')"
runtime_library_manifest_sha="$(sha256sum "$runtime_library_manifest" | awk '{print $1}')"
slurm_job_sha="$(sha256sum "$slurm_job_file" | awk '{print $1}')"
slurm_version_sha="$(sha256sum "$slurm_version_file" | awk '{print $1}')"
slurm_hostnames_sha="$(sha256sum "$slurm_hostnames_file" | awk '{print $1}')"
expected_raw=$(((n0_u + n1_u) * s_tiles))
expected_records=$(((2 * n0_u + n1_u) * s_tiles))

write_manifest() {
  local complete="$1"
  cat > "$manifest" <<EOF
MANIFEST_KIND=CS6_CAPD_C1_CONE_RUN_V1
RUN_COMPLETE=$complete
SOURCE_SHA256=$source_sha
EXECUTABLE_SHA256=$binary_sha
C0_CERTIFICATE_SHA256=$c0_sha
CAPD_CONFIG_SHA256=$config_sha
CAPD_CFLAGS_SHA256=$cflags_sha
CAPD_LIBS_SHA256=$libs_sha
CXX_DRIVER_SHA256=$cxx_sha
CXX_VERSION_SHA256=$cxx_version_sha
CAPD_VERSION_SHA256=$capd_version_sha
CAPD_PC_SHA256=$capd_pc_sha
CAPD_LIBRARY_MANIFEST_SHA256=$capd_library_manifest_sha
CAPD_HEADER_MANIFEST_SHA256=$capd_header_manifest_sha
RUNTIME_LINKAGE_SHA256=$runtime_linkage_sha
RUNTIME_LIBRARY_MANIFEST_SHA256=$runtime_library_manifest_sha
SLURM_JOB_RECORD_SHA256=$slurm_job_sha
SLURM_VERSION_SHA256=$slurm_version_sha
SLURM_HOSTNAMES_SHA256=$slurm_hostnames_sha
C0_AGGREGATOR_SHA256=$c0_aggregator_sha
CAPD_CONFIG_PATH=$capd_config_real
CXX_PATH=$cxx_real
CXX_VERSION=$(cat "$compiler_version_file")
SLURM_JOB_ID=$SLURM_JOB_ID
SLURM_NODELIST=$slurm_nodelist
EXECUTION_NODE=$execution_node
EXECUTION_UID=$execution_uid
EXECUTION_TRUST_MODEL=SAME_UID_ACTIVE_SLURM_ALLOCATION_INCLUDES_EXECUTION_NODE_NO_REMOTE_ATTESTATION
REMOTE_ATTESTATION_PRESENT=false
INDEPENDENT_REPLAY_REQUIRED=true
GRID=N0_U:$n0_u,N1_U:$n1_u,S:$s_tiles
ORDER=$order
C1_SET=C1Rect2Set
C1_INITIAL_DERIVATIVE=B*R_SOURCE_TANGENT_ZERO_NORMAL
RAW_TILES=$expected_raw
EDGE_RECORDS=$expected_records
SHARDS=$shards
EOF
}

write_manifest false
export binary run_dir n0_u n1_u s_tiles order shards
seq 1 "$shards" | xargs -P "$jobs" -n 1 sh -c '
  ordinal="$1"
  "$binary" proof "$n0_u" "$n1_u" "$s_tiles" "$order" \
    "$ordinal" "$shards" "$run_dir/ledger-$ordinal.txt" \
    > "$run_dir/shard-$ordinal.txt"
' _

test "$(sha256sum "$snapshot" | awk '{print $1}')" = "$source_sha"
test "$(sha256sum "$binary" | awk '{print $1}')" = "$binary_sha"
test "$(sha256sum "$c0_snapshot" | awk '{print $1}')" = "$c0_sha"
sha256sum --check --status "$capd_library_manifest"
sha256sum --check --status "$capd_header_manifest"
sha256sum --check --status "$runtime_library_manifest"
for ordinal in $(seq 1 "$shards"); do
  test -s "$run_dir/shard-$ordinal.txt"
  test -f "$run_dir/ledger-$ordinal.txt"
  grep -Fxq 'SHARD_PASS=true' "$run_dir/shard-$ordinal.txt"
done
write_manifest true
echo "CS6_CAPD_C1_CONE_RUN PASS run_dir=$run_dir shards=$shards"
