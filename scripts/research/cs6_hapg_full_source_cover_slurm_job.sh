#!/usr/bin/env bash
#SBATCH --job-name=cs6-hapg-cover
#SBATCH --partition=gpu-orangefs
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --nodelist=gpuorangefs-r770-proxmox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --export=NIL

set -euo pipefail

fail() {
  echo "H-APG Slurm job error: $*" >&2
  exit 1
}

[[ $# -eq 2 ]] || fail "usage: $0 CONFIG EXPECTED_CONFIG_SHA256"
config_source=$1
expected_config_sha=$2
[[ $expected_config_sha =~ ^[0-9a-f]{64}$ ]] || fail "invalid configuration digest"
[[ -f $config_source && ! -L $config_source ]] || \
  fail "configuration must be a regular file"
config_source=$(realpath -e -- "$config_source")
[[ $config_source == /orangefs/training/* ]] || \
  fail "configuration must be staged on OrangeFS"
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail "missing numeric SLURM_JOB_ID"
python_bin=$(command -v python3) || fail "python3 is unavailable"
python_bin=$(realpath -e -- "$python_bin")
[[ $python_bin == /* && -x $python_bin ]] || fail "python3 does not resolve to an executable absolute path"
[[ $("$python_bin" -B -c 'import os, sys; print(os.path.realpath(sys.executable))') == "$python_bin" ]] || \
  fail "python3 does not report the resolved absolute executable path"

work=$(mktemp -d "/tmp/cs6-hapg-cover-${SLURM_JOB_ID}.XXXXXXXX")
cleanup() {
  chmod -R u+w "$work" 2>/dev/null || true
  rm -rf -- "$work"
}
trap cleanup EXIT
config="$work/transport-config.txt"
cp --no-preserve=mode,ownership,timestamps "$config_source" "$config"
[[ $(sha256sum "$config" | awk '{print $1}') == "$expected_config_sha" ]] || \
  fail "configuration digest mismatch"

declare -A cfg=()
while IFS= read -r line || [[ -n $line ]]; do
  [[ -n $line && $line != *$'\r'* && $line == *=* ]] || fail "malformed configuration"
  equals=${line//[^=]/}
  [[ $equals == = ]] || fail "configuration rows must contain exactly one equals sign"
  key=${line%%=*}
  value=${line#*=}
  [[ $key =~ ^[A-Z0-9_]+$ && -n $value && -z ${cfg[$key]+present} ]] || \
    fail "duplicate, empty, or unsafe configuration field"
  cfg[$key]=$value
done < "$config"

common_required=(
  SCHEMA MODE BASE_REPO_BUNDLE_PATH BASE_REPO_BUNDLE_SHA256 BASE_GIT_HEAD
  REPO_DELTA_BUNDLE_PATH REPO_DELTA_BUNDLE_SHA256 PREBUILT_ARCHIVE_PATH
  PREBUILT_ARCHIVE_SHA256 EXPECTED_GIT_HEAD EXPECTED_CONTRACT_SHA256 OUTPUT_DIRECTORY
)
for key in "${common_required[@]}"; do
  [[ -n ${cfg[$key]+present} ]] || fail "missing configuration field: $key"
done
[[ ${cfg[SCHEMA]} == sounio.cs6.hapg-full-source-cover-slurm-config.v3 ]] || \
  fail "configuration schema mismatch"
[[ ${cfg[MODE]} == kat || ${cfg[MODE]} == adaptive ]] || fail "invalid mode"
expected_job_name="cs6-hapg-${cfg[MODE]}-v6-${expected_config_sha:0:12}"
[[ ${SLURM_JOB_NAME:-} == "$expected_job_name" ]] || \
  fail "Slurm job name must be exactly $expected_job_name"
export TZ=UTC
export SLURM_TIME_FORMAT=standard
required=("${common_required[@]}")
if [[ ${cfg[MODE]} == adaptive ]]; then
  required+=(KAT_ARCHIVE_PATH KAT_ARCHIVE_SHA256 KAT_JOB_ID)
fi
[[ ${#cfg[@]} -eq ${#required[@]} ]] || fail "configuration field count mismatch"
for key in "${required[@]}"; do
  [[ -n ${cfg[$key]+present} ]] || fail "missing configuration field: $key"
done
[[ ${cfg[BASE_REPO_BUNDLE_SHA256]} =~ ^[0-9a-f]{64}$ ]] || fail "invalid base Git bundle digest"
[[ ${cfg[REPO_DELTA_BUNDLE_SHA256]} =~ ^[0-9a-f]{64}$ ]] || fail "invalid delta Git bundle digest"
[[ ${cfg[PREBUILT_ARCHIVE_SHA256]} =~ ^[0-9a-f]{64}$ ]] || fail "invalid prebuilt digest"
[[ ${cfg[EXPECTED_CONTRACT_SHA256]} =~ ^[0-9a-f]{64}$ ]] || fail "invalid contract digest"
[[ ${cfg[BASE_GIT_HEAD]} =~ ^[0-9a-f]{40}$ ]] || fail "invalid base Git head"
[[ ${cfg[EXPECTED_GIT_HEAD]} =~ ^[0-9a-f]{40}$ ]] || fail "invalid Git head"
if [[ ${cfg[MODE]} == adaptive ]]; then
  [[ ${cfg[KAT_ARCHIVE_SHA256]} =~ ^[0-9a-f]{64}$ ]] || fail "invalid KAT archive digest"
  [[ ${cfg[KAT_JOB_ID]} =~ ^[1-9][0-9]*$ ]] || fail "invalid KAT Slurm job id"
fi

staged_file_keys=(BASE_REPO_BUNDLE_PATH REPO_DELTA_BUNDLE_PATH PREBUILT_ARCHIVE_PATH)
if [[ ${cfg[MODE]} == adaptive ]]; then
  staged_file_keys+=(KAT_ARCHIVE_PATH)
fi
for key in "${staged_file_keys[@]}"; do
  path=${cfg[$key]}
  [[ $path == /orangefs/training/* && -f $path && ! -L $path ]] || \
    fail "$key must be a regular staged OrangeFS file"
  cfg[$key]=$(realpath -e -- "$path")
  [[ ${cfg[$key]} == /orangefs/training/* ]] || fail "$key escapes OrangeFS"
done
output_dir=${cfg[OUTPUT_DIRECTORY]}
[[ $output_dir == /orangefs/training/* && -d $output_dir && ! -L $output_dir ]] || \
  fail "OUTPUT_DIRECTORY must be an existing OrangeFS directory"
output_dir=$(realpath -e -- "$output_dir")
[[ $output_dir == /orangefs/training/* ]] || fail "OUTPUT_DIRECTORY escapes OrangeFS"

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

base_repo_bundle="$work/repo-base.bundle"
repo_delta_bundle="$work/repo-delta.bundle"
prebuilt_archive="$work/prebuilt.tar"
kat_archive=""
cp --no-preserve=mode,ownership,timestamps --reflink=never \
  "${cfg[BASE_REPO_BUNDLE_PATH]}" "$base_repo_bundle"
cp --no-preserve=mode,ownership,timestamps --reflink=never \
  "${cfg[REPO_DELTA_BUNDLE_PATH]}" "$repo_delta_bundle"
cp --no-preserve=mode,ownership,timestamps --reflink=never \
  "${cfg[PREBUILT_ARCHIVE_PATH]}" "$prebuilt_archive"
[[ $(sha256_file "$base_repo_bundle") == "${cfg[BASE_REPO_BUNDLE_SHA256]}" ]] || \
  fail "staged base Git bundle digest mismatch"
[[ $(sha256_file "$repo_delta_bundle") == "${cfg[REPO_DELTA_BUNDLE_SHA256]}" ]] || \
  fail "staged delta Git bundle digest mismatch"
[[ $(sha256_file "$prebuilt_archive") == "${cfg[PREBUILT_ARCHIVE_SHA256]}" ]] || \
  fail "staged prebuilt archive digest mismatch"
if [[ ${cfg[MODE]} == adaptive ]]; then
  kat_input="$work/kat-input"
  mkdir -- "$kat_input"
  canonical_kat_basename="cs6-hapg-kat-job${cfg[KAT_JOB_ID]}-${cfg[EXPECTED_GIT_HEAD]}.tar"
  [[ $(basename "${cfg[KAT_ARCHIVE_PATH]}") == "$canonical_kat_basename" ]] || \
    fail "KAT archive source basename is noncanonical"
  kat_archive="$kat_input/$canonical_kat_basename"
  kat_archive_sidecar="$kat_archive.sha256"
  [[ ! -e $kat_archive && ! -L $kat_archive ]] || fail "canonical KAT archive staging collision"
  [[ ! -e $kat_archive_sidecar && ! -L $kat_archive_sidecar ]] || \
    fail "canonical KAT sidecar staging collision"
  cp --no-preserve=mode,ownership,timestamps --reflink=never \
    "${cfg[KAT_ARCHIVE_PATH]}" "$kat_archive"
  [[ $(sha256_file "$kat_archive") == "${cfg[KAT_ARCHIVE_SHA256]}" ]] || \
    fail "staged KAT archive digest mismatch"
  source_kat_sidecar="${cfg[KAT_ARCHIVE_PATH]}.sha256"
  [[ -f $source_kat_sidecar && ! -L $source_kat_sidecar ]] || \
    fail "KAT archive sidecar is missing or unsafe"
  expected_source_kat_sidecar_sha=$(printf '%s  %s\n' "${cfg[KAT_ARCHIVE_SHA256]}" \
    "$(basename "${cfg[KAT_ARCHIVE_PATH]}")" | sha256sum | awk '{print $1}')
  [[ $(sha256_file "$source_kat_sidecar") == "$expected_source_kat_sidecar_sha" ]] || \
    fail "KAT archive sidecar mismatch"
  printf '%s  %s\n' "${cfg[KAT_ARCHIVE_SHA256]}" "$(basename "$kat_archive")" \
    > "$kat_archive_sidecar"
  expected_kat_sidecar_sha=$(printf '%s  %s\n' "${cfg[KAT_ARCHIVE_SHA256]}" \
    "$(basename "$kat_archive")" | sha256sum | awk '{print $1}')
  [[ $(sha256_file "$kat_archive_sidecar") == "$expected_kat_sidecar_sha" ]] || \
    fail "staged KAT archive sidecar mismatch"
fi
[[ $(git bundle list-heads "$base_repo_bundle") == "${cfg[BASE_GIT_HEAD]} HEAD" ]] || \
  fail "base Git bundle must expose only the configured HEAD"
[[ $(git bundle list-heads "$repo_delta_bundle") == "${cfg[EXPECTED_GIT_HEAD]} HEAD" ]] || \
  fail "delta Git bundle must expose only the expected HEAD"
mapfile -t delta_header < <(awk '1; /^$/ {exit}' "$repo_delta_bundle")
[[ ${#delta_header[@]} -eq 4 ]] || fail "delta Git bundle header field count mismatch"
[[ ${delta_header[0]} == '# v2 git bundle' ]] || fail "delta Git bundle must use format v2"
[[ ${delta_header[1]} == -"${cfg[BASE_GIT_HEAD]} "* ]] || \
  fail "delta Git bundle must declare the frozen base as its only prerequisite"
[[ ${delta_header[2]} == "${cfg[EXPECTED_GIT_HEAD]} HEAD" && -z ${delta_header[3]} ]] || \
  fail "delta Git bundle header ref mismatch"

git clone --quiet --no-checkout "$base_repo_bundle" "$work/repo"
[[ $(git -C "$work/repo" rev-parse HEAD) == "${cfg[BASE_GIT_HEAD]}" ]] || \
  fail "base Git bundle head mismatch"
git -C "$work/repo" bundle verify "$repo_delta_bundle" >/dev/null
git -C "$work/repo" fetch --quiet "$repo_delta_bundle" HEAD
[[ $(git -C "$work/repo" rev-parse FETCH_HEAD) == "${cfg[EXPECTED_GIT_HEAD]}" ]] || \
  fail "delta Git bundle head mismatch"
git -C "$work/repo" checkout --quiet --detach "${cfg[EXPECTED_GIT_HEAD]}"
[[ $(git -C "$work/repo" rev-parse HEAD) == "${cfg[EXPECTED_GIT_HEAD]}" ]] || \
  fail "checked-out Git head mismatch"
[[ -z $(git -C "$work/repo" status --short --untracked-files=all) ]] || \
  fail "checked-out Git bundle is dirty"

contract="$work/repo/scripts/research/cs6_hapg_full_source_cover_contract_v6.txt"
runner="$work/repo/scripts/research/cs6_hapg_full_source_cover_run.py"
aggregator="$work/repo/scripts/research/cs6_hapg_full_source_cover_aggregate.py"
kat_anchor="$work/repo/scripts/research/cs6_hapg_full_source_cover_kat_anchor.py"
job_script="$work/repo/scripts/research/cs6_hapg_full_source_cover_slurm_job.sh"
[[ $(sha256_file "$contract") == "${cfg[EXPECTED_CONTRACT_SHA256]}" ]] || \
  fail "frozen contract digest mismatch"
contract_value() {
  local key=$1
  local value
  value=$(awk -F= -v key="$key" '$1 == key {print $2}' "$contract")
  [[ -n $value ]] || fail "missing frozen contract field: $key"
  printf '%s' "$value"
}
[[ ${cfg[BASE_GIT_HEAD]} == "$(contract_value BASE_REPO_BUNDLE_GIT_HEAD)" ]] || \
  fail "base Git head differs from the frozen contract"
[[ ${cfg[BASE_REPO_BUNDLE_SHA256]} == "$(contract_value BASE_REPO_BUNDLE_SHA256)" ]] || \
  fail "base Git bundle digest differs from the frozen contract"
executed_job_script_sha=$(sha256_file "$0")
[[ $executed_job_script_sha == "$(contract_value SLURM_JOB_SCRIPT_SHA256)" ]] || \
  fail "executed Slurm job script differs from the frozen contract"
[[ $(sha256_file "$job_script") == "$executed_job_script_sha" ]] || \
  fail "cloned and executing Slurm job scripts differ"
[[ $(sha256_file "$kat_anchor") == "$(contract_value KAT_ANCHOR_SHA256)" ]] || \
  fail "cloned KAT anchor validator differs from the frozen contract"

"$python_bin" -B - "$prebuilt_archive" "$work" <<'PY'
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

archive = Path(sys.argv[1])
root = Path(sys.argv[2])
with tarfile.open(archive, "r:*") as source:
    members = source.getmembers()
    if not members:
        raise SystemExit("empty prebuilt archive")
    names: set[str] = set()
    for member in members:
        pure = PurePosixPath(member.name)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
            or pure.parts[0] != "prebuilt"
            or pure.as_posix() != member.name.rstrip("/")
            or pure.as_posix() in names
            or not (member.isdir() or member.isfile())
        ):
            raise SystemExit("unsafe prebuilt archive member")
        names.add(pure.as_posix())
    for member in members:
        target = root.joinpath(*PurePosixPath(member.name).parts)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            target.chmod(0o755)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        incoming = source.extractfile(member)
        if incoming is None:
            raise SystemExit("missing prebuilt member payload")
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            member.mode & 0o777,
        )
        with incoming, os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(incoming, output)
PY

prebuilt="$work/prebuilt"
result="$work/result"
common=(
  "$python_bin" -B "$runner" --mode "${cfg[MODE]}" --prebuilt-dir "$prebuilt"
  --run-dir "$result" --root-challenge "$(contract_value KAT_ROOT_CHALLENGE)"
  --jobs "$(contract_value BOUNDED_PILOT_JOBS)"
  --timeout-seconds "$(contract_value BOUNDED_PILOT_LEAF_TIMEOUT_SECONDS)"
  --self-test-mutations --enforce-frozen-contract
)
if [[ ${cfg[MODE]} == adaptive ]]; then
  common=(
    "$python_bin" -B "$runner" --mode adaptive --prebuilt-dir "$prebuilt"
    --run-dir "$result"
    --root-challenge "$(contract_value BOUNDED_PILOT_ROOT_CHALLENGE)"
    --replay-root-challenge "$(contract_value BOUNDED_PILOT_REPLAY_ROOT_CHALLENGE)"
    --jobs "$(contract_value BOUNDED_PILOT_JOBS)"
    --timeout-seconds "$(contract_value BOUNDED_PILOT_LEAF_TIMEOUT_SECONDS)"
    --max-nodes "$(contract_value BOUNDED_PILOT_MAX_NODES)"
    --max-waves "$(contract_value BOUNDED_PILOT_MAX_WAVES)"
    --max-u-depth "$(contract_value BOUNDED_PILOT_MAX_U_DEPTH)"
    --max-s-depth "$(contract_value BOUNDED_PILOT_MAX_S_DEPTH)"
    --kat-archive "$kat_archive"
    --kat-archive-sha256 "${cfg[KAT_ARCHIVE_SHA256]}"
    --kat-job-id "${cfg[KAT_JOB_ID]}"
    --transport-repo-delta-sha256 "${cfg[REPO_DELTA_BUNDLE_SHA256]}"
    --transport-prebuilt-archive-sha256 "${cfg[PREBUILT_ARCHIVE_SHA256]}"
    --self-test-mutations --enforce-frozen-contract
  )
fi
"${common[@]}"

scontrol -o show job "$SLURM_JOB_ID" > "$work/transport-slurm-job-record.txt"
aggregation_sha=$(printf '%064d' 0)
aggregation_rc=0
post_run_gate_pass=true
failure_stage=NONE
if [[ ${cfg[MODE]} == adaptive ]]; then
  if "$python_bin" -B "$aggregator" "$result" \
      --expected-contract-sha "${cfg[EXPECTED_CONTRACT_SHA256]}" \
      --expected-git-head "${cfg[EXPECTED_GIT_HEAD]}" \
      --kat-archive "$kat_archive" \
      --kat-archive-sha256 "${cfg[KAT_ARCHIVE_SHA256]}" \
      --kat-job-id "${cfg[KAT_JOB_ID]}" \
      --transport-repo-delta-sha256 "${cfg[REPO_DELTA_BUNDLE_SHA256]}" \
      --transport-prebuilt-archive-sha256 "${cfg[PREBUILT_ARCHIVE_SHA256]}" \
      --output "$work/aggregation.txt" --self-test-mutations \
      > "$work/aggregator.stdout" 2> "$work/aggregator.stderr"; then
    if [[ -f $work/aggregation.txt && ! -L $work/aggregation.txt ]]; then
      aggregation_sha=$(sha256_file "$work/aggregation.txt")
    else
      aggregation_rc=1
      post_run_gate_pass=false
      failure_stage=POST_RUN_INDEPENDENT_AGGREGATION_OUTPUT_MISSING
    fi
  else
    aggregation_rc=$?
    post_run_gate_pass=false
    failure_stage=POST_RUN_INDEPENDENT_AGGREGATION
  fi
  if [[ $aggregation_rc -ne 0 ]]; then
    cat > "$work/post-run-failure.txt" <<EOF
SCHEMA=sounio.cs6.hapg-full-source-cover-post-run-failure.v1
MODE=adaptive
SLURM_JOB_ID=$SLURM_JOB_ID
FAILURE_STAGE=$failure_stage
FAILURE_RC=$aggregation_rc
RESULT_RUN_MANIFEST_SHA256=$(if [[ -f $result/run-manifest.txt && ! -L $result/run-manifest.txt ]]; then sha256_file "$result/run-manifest.txt"; else printf '%064d' 0; fi)
RESULT_FILES_INDEX_SHA256=$(if [[ -f $result/files.sha256 && ! -L $result/files.sha256 ]]; then sha256_file "$result/files.sha256"; else printf '%064d' 0; fi)
AGGREGATOR_STDOUT_SHA256=$(sha256_file "$work/aggregator.stdout")
AGGREGATOR_STDERR_SHA256=$(sha256_file "$work/aggregator.stderr")
SCIENTIFIC_RESULT_AVAILABLE=false
EXECUTION_PROVENANCE_ATTESTED=false
PROMOTION_ELIGIBLE=false
EOF
  fi
fi

kat_required=false
kat_job_id=0
kat_archive_sha=$(printf '%064d' 0)
kat_certificate_sha=$(printf '%064d' 0)
if [[ ${cfg[MODE]} == adaptive ]]; then
  kat_required=true
  kat_job_id=${cfg[KAT_JOB_ID]}
  kat_archive_sha=${cfg[KAT_ARCHIVE_SHA256]}
  if [[ -f $result/kat-prerequisite-certificate.txt && ! -L $result/kat-prerequisite-certificate.txt ]]; then
    kat_certificate_sha=$(sha256_file "$result/kat-prerequisite-certificate.txt")
  fi
fi
cat > "$work/transport-manifest.txt" <<EOF
SCHEMA=sounio.cs6.hapg-full-source-cover-transport.v3
MODE=${cfg[MODE]}
SLURM_JOB_ID=$SLURM_JOB_ID
SLURM_NODE=${SLURMD_NODENAME:-UNKNOWN}
EXPECTED_GIT_HEAD=${cfg[EXPECTED_GIT_HEAD]}
EXPECTED_CONTRACT_SHA256=${cfg[EXPECTED_CONTRACT_SHA256]}
SLURM_JOB_SCRIPT_SHA256=$executed_job_script_sha
CONFIG_SHA256=$(sha256_file "$config")
PYTHON_EXECUTABLE_REALPATH=$python_bin
BASE_REPO_BUNDLE_SHA256=${cfg[BASE_REPO_BUNDLE_SHA256]}
BASE_GIT_HEAD=${cfg[BASE_GIT_HEAD]}
REPO_DELTA_BUNDLE_SHA256=${cfg[REPO_DELTA_BUNDLE_SHA256]}
PREBUILT_ARCHIVE_SHA256=${cfg[PREBUILT_ARCHIVE_SHA256]}
KAT_PREREQUISITE_REQUIRED=$kat_required
KAT_JOB_ID=$kat_job_id
KAT_ARCHIVE_SHA256=$kat_archive_sha
KAT_CERTIFICATE_SHA256=$kat_certificate_sha
RESULT_RUN_MANIFEST_SHA256=$(if [[ -f $result/run-manifest.txt && ! -L $result/run-manifest.txt ]]; then sha256_file "$result/run-manifest.txt"; else printf '%064d' 0; fi)
RESULT_FILES_INDEX_SHA256=$(if [[ -f $result/files.sha256 && ! -L $result/files.sha256 ]]; then sha256_file "$result/files.sha256"; else printf '%064d' 0; fi)
AGGREGATION_SHA256=$aggregation_sha
POST_RUN_GATE_PASS=$post_run_gate_pass
FAILURE_STAGE=$failure_stage
FAILURE_RC=$aggregation_rc
EXECUTION_PROVENANCE_ATTESTED=false
PROMOTION_ELIGIBLE=false
EOF

archive_inputs=(result transport-config.txt transport-manifest.txt transport-slurm-job-record.txt)
if [[ ${cfg[MODE]} == adaptive ]]; then
  archive_inputs+=(aggregator.stdout aggregator.stderr)
fi
if [[ $aggregation_rc -ne 0 ]]; then
  archive="$work/cs6-hapg-${cfg[MODE]}-diagnostic-job${SLURM_JOB_ID}-${cfg[EXPECTED_GIT_HEAD]}.tar"
  archive_inputs+=(post-run-failure.txt)
else
  archive="$work/cs6-hapg-${cfg[MODE]}-job${SLURM_JOB_ID}-${cfg[EXPECTED_GIT_HEAD]}.tar"
fi
if [[ ${cfg[MODE]} == adaptive && $aggregation_rc -eq 0 ]]; then
  archive_inputs+=(aggregation.txt)
fi
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
  --format=posix --pax-option=delete=atime,delete=ctime \
  -C "$work" -cf "$archive" "${archive_inputs[@]}"
archive_sha=$(sha256_file "$archive")
destination="$output_dir/$(basename "$archive")"
sidecar="$destination.sha256"
"$python_bin" -B - "$archive" "$destination" "$archive_sha" "$sidecar" <<'PY'
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
expected = sys.argv[3]
sidecar = Path(sys.argv[4])
archive_identity: tuple[int, int] | None = None


def fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EROFS}:
                raise
    finally:
        os.close(descriptor)


def unlink_owned(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        current = path.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) == identity:
            path.unlink()
    except FileNotFoundError:
        pass


def publish_file(path: Path, incoming, expected_sha: str) -> tuple[int, int]:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    identity = (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as output:
            while True:
                block = incoming.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected_sha:
            raise RuntimeError("published byte stream digest mismatch")
        reopened_digest = hashlib.sha256()
        with path.open("rb") as reopened:
            while True:
                block = reopened.read(1024 * 1024)
                if not block:
                    break
                reopened_digest.update(block)
        actual = reopened_digest.hexdigest()
        if actual != expected_sha:
            raise RuntimeError("reopened published file digest mismatch")
        fsync_parent(path)
        return identity
    except Exception:
        unlink_owned(path, identity)
        raise


try:
    with source.open("rb") as incoming:
        archive_identity = publish_file(destination, incoming, expected)
    sidecar_raw = f"{expected}  {destination.name}\n".encode("ascii")
    from io import BytesIO

    publish_file(sidecar, BytesIO(sidecar_raw), hashlib.sha256(sidecar_raw).hexdigest())
except Exception:
    unlink_owned(destination, archive_identity)
    raise
PY
[[ $(sha256_file "$destination") == "$archive_sha" ]] || \
  fail "transported archive digest mismatch"
[[ $(cat "$sidecar") == "$archive_sha  $(basename "$destination")" ]] || \
  fail "transport sidecar mismatch"

echo "HAPG_TRANSPORT_ARCHIVE=$destination"
echo "HAPG_TRANSPORT_ARCHIVE_SHA256=$archive_sha"
echo "HAPG_TRANSPORT_SIDECAR=$sidecar"
echo "PROMOTION_ELIGIBLE=false"
if [[ $aggregation_rc -ne 0 ]]; then
  echo "HAPG_POST_RUN_DIAGNOSTIC_ARCHIVE_PUBLISHED=true" >&2
  exit "$aggregation_rc"
fi
