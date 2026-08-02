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

work=$(mktemp -d "/tmp/cs6-hapg-cover-${SLURM_JOB_ID}.XXXXXXXX")
cleanup() {
  chmod -R u+w "$work" 2>/dev/null || true
  rm -rf -- "$work"
}
trap cleanup EXIT
config="$work/config.txt"
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

required=(
  SCHEMA MODE REPO_BUNDLE_PATH REPO_BUNDLE_SHA256 PREBUILT_ARCHIVE_PATH
  PREBUILT_ARCHIVE_SHA256 EXPECTED_GIT_HEAD EXPECTED_CONTRACT_SHA256
  OUTPUT_DIRECTORY
)
[[ ${#cfg[@]} -eq ${#required[@]} ]] || fail "configuration field count mismatch"
for key in "${required[@]}"; do
  [[ -n ${cfg[$key]+present} ]] || fail "missing configuration field: $key"
done
[[ ${cfg[SCHEMA]} == sounio.cs6.hapg-full-source-cover-slurm-config.v1 ]] || \
  fail "configuration schema mismatch"
[[ ${cfg[MODE]} == kat || ${cfg[MODE]} == adaptive ]] || fail "invalid mode"
[[ ${cfg[REPO_BUNDLE_SHA256]} =~ ^[0-9a-f]{64}$ ]] || fail "invalid Git bundle digest"
[[ ${cfg[PREBUILT_ARCHIVE_SHA256]} =~ ^[0-9a-f]{64}$ ]] || fail "invalid prebuilt digest"
[[ ${cfg[EXPECTED_CONTRACT_SHA256]} =~ ^[0-9a-f]{64}$ ]] || fail "invalid contract digest"
[[ ${cfg[EXPECTED_GIT_HEAD]} =~ ^[0-9a-f]{40}$ ]] || fail "invalid Git head"

for key in REPO_BUNDLE_PATH PREBUILT_ARCHIVE_PATH; do
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

repo_bundle="$work/repo.bundle"
prebuilt_archive="$work/prebuilt.tar"
cp --no-preserve=mode,ownership,timestamps --reflink=never \
  "${cfg[REPO_BUNDLE_PATH]}" "$repo_bundle"
cp --no-preserve=mode,ownership,timestamps --reflink=never \
  "${cfg[PREBUILT_ARCHIVE_PATH]}" "$prebuilt_archive"
[[ $(sha256_file "$repo_bundle") == "${cfg[REPO_BUNDLE_SHA256]}" ]] || \
  fail "staged Git bundle digest mismatch"
[[ $(sha256_file "$prebuilt_archive") == "${cfg[PREBUILT_ARCHIVE_SHA256]}" ]] || \
  fail "staged prebuilt archive digest mismatch"

git clone --quiet --no-checkout "$repo_bundle" "$work/repo"
git -C "$work/repo" checkout --quiet --detach "${cfg[EXPECTED_GIT_HEAD]}"
[[ $(git -C "$work/repo" rev-parse HEAD) == "${cfg[EXPECTED_GIT_HEAD]}" ]] || \
  fail "checked-out Git head mismatch"
[[ -z $(git -C "$work/repo" status --short --untracked-files=all) ]] || \
  fail "checked-out Git bundle is dirty"

contract="$work/repo/scripts/research/cs6_hapg_full_source_cover_contract_v2.txt"
runner="$work/repo/scripts/research/cs6_hapg_full_source_cover_run.py"
aggregator="$work/repo/scripts/research/cs6_hapg_full_source_cover_aggregate.py"
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
executed_job_script_sha=$(sha256_file "$0")
[[ $executed_job_script_sha == "$(contract_value SLURM_JOB_SCRIPT_SHA256)" ]] || \
  fail "executed Slurm job script differs from the frozen contract"
[[ $(sha256_file "$job_script") == "$executed_job_script_sha" ]] || \
  fail "cloned and executing Slurm job scripts differ"

python3 -B - "$prebuilt_archive" "$work" <<'PY'
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
  python3 -B "$runner" --mode "${cfg[MODE]}" --prebuilt-dir "$prebuilt"
  --run-dir "$result" --root-challenge "$(contract_value KAT_ROOT_CHALLENGE)"
  --jobs "$(contract_value BOUNDED_PILOT_JOBS)"
  --timeout-seconds "$(contract_value BOUNDED_PILOT_LEAF_TIMEOUT_SECONDS)"
  --self-test-mutations --enforce-frozen-contract
)
if [[ ${cfg[MODE]} == adaptive ]]; then
  common=(
    python3 -B "$runner" --mode adaptive --prebuilt-dir "$prebuilt"
    --run-dir "$result"
    --root-challenge "$(contract_value BOUNDED_PILOT_ROOT_CHALLENGE)"
    --replay-root-challenge "$(contract_value BOUNDED_PILOT_REPLAY_ROOT_CHALLENGE)"
    --jobs "$(contract_value BOUNDED_PILOT_JOBS)"
    --timeout-seconds "$(contract_value BOUNDED_PILOT_LEAF_TIMEOUT_SECONDS)"
    --max-nodes "$(contract_value BOUNDED_PILOT_MAX_NODES)"
    --max-waves "$(contract_value BOUNDED_PILOT_MAX_WAVES)"
    --max-u-depth "$(contract_value BOUNDED_PILOT_MAX_U_DEPTH)"
    --max-s-depth "$(contract_value BOUNDED_PILOT_MAX_S_DEPTH)"
    --self-test-mutations --enforce-frozen-contract
  )
fi
"${common[@]}"

aggregation_sha=$(printf '%064d' 0)
if [[ ${cfg[MODE]} == adaptive ]]; then
  python3 -B "$aggregator" "$result" \
    --expected-contract-sha "${cfg[EXPECTED_CONTRACT_SHA256]}" \
    --expected-git-head "${cfg[EXPECTED_GIT_HEAD]}" \
    --output "$work/aggregation.txt" --self-test-mutations
  aggregation_sha=$(sha256_file "$work/aggregation.txt")
fi

scontrol -o show job "$SLURM_JOB_ID" > "$work/transport-slurm-job-record.txt"
cat > "$work/transport-manifest.txt" <<EOF
SCHEMA=sounio.cs6.hapg-full-source-cover-transport.v1
MODE=${cfg[MODE]}
SLURM_JOB_ID=$SLURM_JOB_ID
SLURM_NODE=${SLURMD_NODENAME:-UNKNOWN}
EXPECTED_GIT_HEAD=${cfg[EXPECTED_GIT_HEAD]}
EXPECTED_CONTRACT_SHA256=${cfg[EXPECTED_CONTRACT_SHA256]}
SLURM_JOB_SCRIPT_SHA256=$executed_job_script_sha
CONFIG_SHA256=$(sha256_file "$config")
REPO_BUNDLE_SHA256=${cfg[REPO_BUNDLE_SHA256]}
PREBUILT_ARCHIVE_SHA256=${cfg[PREBUILT_ARCHIVE_SHA256]}
RESULT_RUN_MANIFEST_SHA256=$(sha256_file "$result/run-manifest.txt")
RESULT_FILES_INDEX_SHA256=$(sha256_file "$result/files.sha256")
AGGREGATION_SHA256=$aggregation_sha
EXECUTION_PROVENANCE_ATTESTED=false
PROMOTION_ELIGIBLE=false
EOF

archive="$work/cs6-hapg-${cfg[MODE]}-job${SLURM_JOB_ID}-${cfg[EXPECTED_GIT_HEAD]}.tar"
archive_inputs=(result transport-manifest.txt transport-slurm-job-record.txt)
if [[ ${cfg[MODE]} == adaptive ]]; then
  archive_inputs+=(aggregation.txt)
fi
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
  --format=posix --pax-option=delete=atime,delete=ctime \
  -C "$work" -cf "$archive" "${archive_inputs[@]}"
archive_sha=$(sha256_file "$archive")
destination="$output_dir/$(basename "$archive")"
sidecar="$destination.sha256"
python3 -B - "$archive" "$destination" "$archive_sha" "$sidecar" <<'PY'
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
