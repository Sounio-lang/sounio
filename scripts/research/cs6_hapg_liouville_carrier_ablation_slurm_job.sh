#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG" >&2
  exit 64
fi

config=$1
[[ -f $config && ! -L $config ]] || {
  echo "V7-A Slurm error: config is not a regular file" >&2
  exit 1
}

declare -A fields=()
while IFS='=' read -r key value; do
  [[ -n $key && -n $value && $key != *' '* && -z ${fields[$key]+x} ]] || {
    echo "V7-A Slurm error: malformed or duplicate config field" >&2
    exit 1
  }
  fields[$key]=$value
done < "$config"

required=(
  SCHEMA REPO_BUNDLE REPO_BUNDLE_SHA256 GIT_HEAD PREBUILT_ARCHIVE
  PREBUILT_ARCHIVE_SHA256 OUTPUT_ARCHIVE ROOT_CHALLENGE FROZEN_CONTRACT_SHA256
  JOBS TIMEOUT_SECONDS
)
for key in "${required[@]}"; do
  [[ -n ${fields[$key]:-} ]] || {
    echo "V7-A Slurm error: missing config field $key" >&2
    exit 1
  }
done
[[ ${fields[SCHEMA]} == sounio.cs6.hapg-liouville-carrier-ablation-slurm-config.v1 ]] || {
  echo "V7-A Slurm error: config schema mismatch" >&2
  exit 1
}
[[ ${fields[FROZEN_CONTRACT_SHA256]} == decf9089e1dc9aae513f48c48a00e1c815a585b6ba7e9cd1c09b0b514fd58481 ]] || {
  echo "V7-A Slurm error: frozen contract digest mismatch" >&2
  exit 1
}
sha_re='^[0-9a-f]{64}$'
for key in REPO_BUNDLE_SHA256 PREBUILT_ARCHIVE_SHA256 ROOT_CHALLENGE FROZEN_CONTRACT_SHA256; do
  [[ ${fields[$key]} =~ $sha_re ]] || {
    echo "V7-A Slurm error: malformed digest $key" >&2
    exit 1
  }
done
[[ ${fields[GIT_HEAD]} =~ ^[0-9a-f]{40}$ ]] || {
  echo "V7-A Slurm error: malformed Git commit" >&2
  exit 1
}
[[ ${fields[JOBS]} =~ ^([1-9]|[1-9][0-9]|1[01][0-9]|120)$ ]] || {
  echo "V7-A Slurm error: JOBS is outside 1..120" >&2
  exit 1
}
[[ ${fields[TIMEOUT_SECONDS]} =~ ^[1-9][0-9]*$ ]] || {
  echo "V7-A Slurm error: timeout is not canonical" >&2
  exit 1
}
for key in REPO_BUNDLE PREBUILT_ARCHIVE; do
  [[ -f ${fields[$key]} && ! -L ${fields[$key]} ]] || {
    echo "V7-A Slurm error: input artifact is not a regular file: $key" >&2
    exit 1
  }
done
[[ ! -e ${fields[OUTPUT_ARCHIVE]} && ! -L ${fields[OUTPUT_ARCHIVE]} ]] || {
  echo "V7-A Slurm error: output archive already exists" >&2
  exit 1
}
[[ ! -e ${fields[OUTPUT_ARCHIVE]}.sha256 && ! -L ${fields[OUTPUT_ARCHIVE]}.sha256 ]] || {
  echo "V7-A Slurm error: output sidecar already exists" >&2
  exit 1
}
[[ $(sha256sum "${fields[REPO_BUNDLE]}" | awk '{print $1}') == ${fields[REPO_BUNDLE_SHA256]} ]] || {
  echo "V7-A Slurm error: repository bundle digest mismatch" >&2
  exit 1
}
[[ $(sha256sum "${fields[PREBUILT_ARCHIVE]}" | awk '{print $1}') == ${fields[PREBUILT_ARCHIVE_SHA256]} ]] || {
  echo "V7-A Slurm error: prebuilt archive digest mismatch" >&2
  exit 1
}

scratch=${TMPDIR:-/tmp}/cs6-v7-carrier-${SLURM_JOB_ID:-manual}-$$
mkdir -m 700 "$scratch"
mkdir -m 700 "$scratch/transport"
cleanup() {
  rm -rf "$scratch"
}
trap cleanup EXIT

git clone --quiet "${fields[REPO_BUNDLE]}" "$scratch/repo"
git -C "$scratch/repo" checkout --quiet --detach "${fields[GIT_HEAD]}"
[[ $(git -C "$scratch/repo" rev-parse HEAD) == ${fields[GIT_HEAD]} ]] || {
  echo "V7-A Slurm error: checked-out commit mismatch" >&2
  exit 1
}
[[ -z $(git -C "$scratch/repo" status --short --untracked-files=all) ]] || {
  echo "V7-A Slurm error: staged repository is dirty" >&2
  exit 1
}

mkdir "$scratch/prebuilt"
python3 -B - "${fields[PREBUILT_ARCHIVE]}" "$scratch/prebuilt" <<'PY'
from pathlib import Path, PurePosixPath
import sys
import tarfile

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
with tarfile.open(archive, "r:") as handle:
    members = handle.getmembers()
    if not members or len(members) > 64:
        raise SystemExit("V7-A Slurm error: invalid prebuilt member count")
    if sum(member.size for member in members) > 512 * 1024 * 1024:
        raise SystemExit("V7-A Slurm error: prebuilt archive exceeds total bound")
    for member in members:
        token = PurePosixPath(member.name)
        if token.is_absolute() or ".." in token.parts or len(token.parts) != 1 or not member.isfile():
            raise SystemExit("V7-A Slurm error: unsafe prebuilt archive member")
        if len(member.name.encode("utf-8")) > 255 or member.size > 256 * 1024 * 1024:
            raise SystemExit("V7-A Slurm error: prebuilt member exceeds bounds")
    for member in members:
        source = handle.extractfile(member)
        if source is None:
            raise SystemExit("V7-A Slurm error: unreadable prebuilt member")
        target = destination / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source, target.open("xb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
PY
chmod 700 "$scratch/prebuilt/worker-binary"

[[ ${SLURM_CPUS_PER_TASK:-} =~ ^[1-9][0-9]*$ ]] || {
  echo "V7-A Slurm error: SLURM_CPUS_PER_TASK is unavailable" >&2
  exit 1
}
(( fields[JOBS] <= SLURM_CPUS_PER_TASK )) || {
  echo "V7-A Slurm error: JOBS exceeds allocated CPUs per task" >&2
  exit 1
}

cp "$config" "$scratch/transport/config.txt"
sha256sum "$0" > "$scratch/transport/job-script.sha256"
sha256sum "$config" > "$scratch/transport/config.sha256"
printf '%s\n' "$0" > "$scratch/transport/job-script-path.txt"
printf '%s\n' "$config" > "$scratch/transport/config-path.txt"
scontrol show job "${SLURM_JOB_ID:-0}" > "$scratch/transport/scontrol-job.txt" 2>&1 || true
uname -a > "$scratch/prebuilt/node-uname.txt"
lscpu > "$scratch/prebuilt/node-lscpu.txt"
ldd "$scratch/prebuilt/worker-binary" > "$scratch/prebuilt/node-runtime-linkage.txt"
! grep -Fq '=> not found' "$scratch/prebuilt/node-runtime-linkage.txt" || {
  echo "V7-A Slurm error: worker has an unresolved node runtime library" >&2
  exit 1
}
python3 -B - "$scratch/prebuilt/node-runtime-linkage.txt" \
  "$scratch/prebuilt/node-runtime-libraries.sha256" <<'PY'
from pathlib import Path
import hashlib
import sys

rows = set()
for line in Path(sys.argv[1]).read_text(encoding="ascii").splitlines():
    fields = line.split()
    candidate = None
    if "=>" in fields and fields.index("=>") + 1 < len(fields):
        candidate = fields[fields.index("=>") + 1]
    elif fields and fields[0].startswith("/"):
        candidate = fields[0]
    if candidate and candidate.startswith("/") and Path(candidate).is_file():
        path = Path(candidate)
        rows.add(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")
if not rows:
    raise SystemExit("V7-A Slurm error: no hashable node runtime libraries")
Path(sys.argv[2]).write_text("\n".join(sorted(rows)) + "\n", encoding="ascii")
PY

{
  printf 'SCHEMA=sounio.cs6.hapg-liouville-carrier-ablation-slurm-context.v1\n'
  printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-NONE}"
  printf 'SLURM_JOB_NODELIST=%s\n' "${SLURM_JOB_NODELIST:-NONE}"
  printf 'SLURM_CPUS_ON_NODE=%s\n' "${SLURM_CPUS_ON_NODE:-UNKNOWN}"
  printf 'SLURM_CPUS_PER_TASK=%s\n' "${SLURM_CPUS_PER_TASK:-UNKNOWN}"
} > "$scratch/prebuilt/slurm-context.txt"

set +e
python3 -B "$scratch/repo/scripts/research/cs6_hapg_liouville_carrier_ablation_run.py" \
  --repo "$scratch/repo" \
  --worker "$scratch/prebuilt/worker-binary" \
  --provenance-dir "$scratch/prebuilt" \
  --run-dir "$scratch/result" \
  --root-challenge "${fields[ROOT_CHALLENGE]}" \
  --jobs "${fields[JOBS]}" \
  --timeout-seconds "${fields[TIMEOUT_SECONDS]}" \
  --self-test-mutations \
  --keep-failed \
  > "$scratch/transport/runner.stdout.txt" \
  2> "$scratch/transport/runner.stderr.txt"
runner_rc=$?
set -e
printf '%s\n' "$runner_rc" > "$scratch/transport/runner-rc.txt"
audit_rc=125
if [[ $runner_rc -ne 0 && $runner_rc -ne 2 ]]; then
  printf 'false\n' > "$scratch/transport/runner-result-present.txt"
  failed_work=$(sed -n 's/^FAILED_WORK_DIR=//p' "$scratch/transport/runner.stderr.txt" | tail -n 1)
  if [[ -n $failed_work && $failed_work == "$scratch"/* && -d $failed_work && ! -L $failed_work ]]; then
    mv "$failed_work" "$scratch/failed-result"
  fi
else
  [[ -d $scratch/result && ! -L $scratch/result ]] || {
    echo "V7-A Slurm error: runner did not publish its result directory" >&2
    exit 1
  }
  printf 'true\n' > "$scratch/transport/runner-result-present.txt"
  set +e
  python3 -B "$scratch/repo/scripts/research/cs6_hapg_liouville_carrier_ablation_retained_verify.py" \
    --repo "$scratch/repo" \
    --worker "$scratch/prebuilt/worker-binary" \
    "$scratch/result" \
    > "$scratch/transport/retained-audit.stdout.txt" \
    2> "$scratch/transport/retained-audit.stderr.txt"
  audit_rc=$?
  set -e
fi
printf '%s\n' "$audit_rc" > "$scratch/transport/retained-audit-rc.txt"
printf '%s\n' "$([[ $audit_rc -eq 0 ]] && echo true || echo false)" \
  > "$scratch/transport/retained-audit-pass.txt"

publication_kind=RESULT
publication_target=${fields[OUTPUT_ARCHIVE]}
publication_rc=$runner_rc
if [[ $runner_rc -eq 0 || $runner_rc -eq 2 ]]; then
  if [[ $audit_rc -ne 0 ]]; then
    publication_kind=QUARANTINE
    publication_target="${fields[OUTPUT_ARCHIVE]}.quarantine-audit-${SLURM_JOB_ID:-manual}"
    publication_rc=70
  fi
else
  publication_kind=QUARANTINE
  publication_target="${fields[OUTPUT_ARCHIVE]}.quarantine-runner-${SLURM_JOB_ID:-manual}"
fi

archive_tmp=$scratch/result.tar.tmp
archive_members=(transport)
[[ -d $scratch/result ]] && archive_members+=(result)
[[ -d $scratch/failed-result ]] && archive_members+=(failed-result)
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -C "$scratch" -cf "$archive_tmp" "${archive_members[@]}"
archive_sha=$(sha256sum "$archive_tmp" | awk '{print $1}')
mkdir -p "$(dirname "$publication_target")"
python3 -B - "$archive_tmp" "$publication_target" <<'PY'
import os
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
descriptor = os.open(target, flags, 0o600)
try:
    with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
        while block := input_handle.read(1024 * 1024):
            output_handle.write(block)
        output_handle.flush()
        os.fsync(output_handle.fileno())
except BaseException:
    target.unlink(missing_ok=True)
    raise
PY

[[ $(sha256sum "$publication_target" | awk '{print $1}') == "$archive_sha" ]] || {
  echo "V7-A Slurm error: published archive digest mismatch" >&2
  exit 1
}
python3 -B - "${publication_target}.sha256" "$archive_sha" \
  "$(basename "$publication_target")" <<'PY'
import os
from pathlib import Path
import sys

target = Path(sys.argv[1])
raw = f"{sys.argv[2]}  {sys.argv[3]}\n".encode("ascii")
descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(raw)
    handle.flush()
    os.fsync(handle.fileno())
PY

printf '%s_ARCHIVE=%s\n' "$publication_kind" "$publication_target"
printf '%s_ARCHIVE_SHA256=%s\n' "$publication_kind" "$archive_sha"
printf '%s_SIDECAR=%s.sha256\n' "$publication_kind" "$publication_target"
exit "$publication_rc"
