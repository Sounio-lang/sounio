#!/bin/bash
set -euo pipefail
umask 077

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG" >&2
  exit 64
fi
[[ -f $0 && ! -L $0 ]] || {
  echo "V7-A.1 Slurm error: job script is not a regular file" >&2
  exit 1
}

config=$1
[[ -f $config && ! -L $config ]] || {
  echo "V7-A.1 Slurm error: config is not a regular file" >&2
  exit 1
}

declare -A fields=()
while IFS='=' read -r key value; do
  [[ -n $key && -n $value && $key != *' '* && -z ${fields[$key]+x} ]] || {
    echo "V7-A.1 Slurm error: malformed or duplicate config field" >&2
    exit 1
  }
  fields[$key]=$value
done < "$config"

required=(
  SCHEMA EXECUTION_CLASS REPO_ARCHIVE REPO_ARCHIVE_SHA256 GIT_HEAD PREBUILT_ARCHIVE
  PREBUILT_ARCHIVE_SHA256 OUTPUT_ARCHIVE ROOT_CHALLENGE FROZEN_CONTRACT_SHA256
  COORDINATE_MANIFEST_SHA256 JOB_SCRIPT_SHA256 JOBS TIMEOUT_SECONDS
  SUBMITTED_JOB_SCRIPT
  SLURM_PARTITION SLURM_ACCOUNT SLURM_QOS SLURM_NODELIST SLURM_NODES
  SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_JOB_NAME SLURM_TIME_LIMIT
  SLURM_MIN_MEMORY_NODE SLURM_EXCLUSIVE
)
for key in "${required[@]}"; do
  [[ -n ${fields[$key]:-} ]] || {
    echo "V7-A.1 Slurm error: missing config field $key" >&2
    exit 1
  }
done
[[ ${#fields[@]} -eq ${#required[@]} ]] || {
  echo "V7-A.1 Slurm error: config has undeclared fields" >&2
  exit 1
}
[[ ${fields[SCHEMA]} == sounio.cs6.hapg-liouville-checkpoint-slurm-config.v2 ]] || {
  echo "V7-A.1 Slurm error: config schema mismatch" >&2
  exit 1
}
[[ ${fields[FROZEN_CONTRACT_SHA256]} == 3afc0475847ad8054234a2ddfa108b768cfd81991d0be71fc21c991f363631ce ]] || {
  echo "V7-A.1 Slurm error: frozen contract digest mismatch" >&2
  exit 1
}
[[ ${fields[COORDINATE_MANIFEST_SHA256]} == 527afc7c205fcf09b15a0bff91df6935f19ed2b7e7926895916ac5da33a992a7 ]] || {
  echo "V7-A.1 Slurm error: coordinate manifest digest mismatch" >&2
  exit 1
}
[[ ${fields[ROOT_CHALLENGE]} == ad536f25d02182c8b2add12ef1a7e8a8a18b4eb0d12e68535ea94ddb3eff0bdf ]] || {
  echo "V7-A.1 Slurm error: root challenge mismatch" >&2
  exit 1
}
sha_re='^[0-9a-f]{64}$'
for key in REPO_ARCHIVE_SHA256 PREBUILT_ARCHIVE_SHA256 ROOT_CHALLENGE \
  FROZEN_CONTRACT_SHA256 COORDINATE_MANIFEST_SHA256 JOB_SCRIPT_SHA256; do
  [[ ${fields[$key]} =~ $sha_re ]] || {
    echo "V7-A.1 Slurm error: malformed digest $key" >&2
    exit 1
  }
done
[[ ${fields[GIT_HEAD]} =~ ^[0-9a-f]{40}$ ]] || {
  echo "V7-A.1 Slurm error: malformed Git commit" >&2
  exit 1
}
[[ ${fields[JOBS]} =~ ^[1-9]$ ]] || {
  echo "V7-A.1 Slurm error: JOBS is outside 1..9" >&2
  exit 1
}
[[ ${fields[TIMEOUT_SECONDS]} =~ ^[1-9][0-9]*$ ]] || {
  echo "V7-A.1 Slurm error: timeout is not canonical" >&2
  exit 1
}
case ${fields[EXECUTION_CLASS]} in
  AUTHORITATIVE_SLURM)
    [[ ${fields[SLURM_PARTITION]} == gpu-orangefs &&
       ${fields[SLURM_ACCOUNT]} == lab &&
       ${fields[SLURM_QOS]} == normal &&
       ${fields[SLURM_NODELIST]} == gpuorangefs-r770-proxmox &&
       ${fields[SLURM_NODES]} == 1 &&
       ${fields[SLURM_NTASKS]} == 1 &&
       ${fields[SLURM_CPUS_PER_TASK]} == 9 &&
       ${fields[SLURM_JOB_NAME]} == cs6-v7a1-checkpoint &&
       ${fields[SLURM_TIME_LIMIT]} == 00:20:00 &&
       ${fields[SLURM_MIN_MEMORY_NODE]} == 8G &&
       ${fields[SLURM_EXCLUSIVE]} == NODE ]] || {
      echo "V7-A.1 Slurm error: authoritative allocation config differs" >&2
      exit 1
    }
    [[ -z ${CS6_V7A1_SCONTROL_BIN:-} && ${CS6_V7A1_SYNTHETIC_GATE:-0} == 0 ]] || {
      echo "V7-A.1 Slurm error: authoritative execution forbids scheduler overrides" >&2
      exit 1
    }
    export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export LC_ALL=C TZ=UTC TMPDIR=/tmp
    unset PYTHONPATH PYTHONHOME BASH_ENV ENV CDPATH GLOBIGNORE
    hash -r
    unset -f scontrol hostname python3 grep id git sha256sum tar awk ldd \
      lscpu uname cp chmod mkdir rm mv sed tail 2>/dev/null || true
    scontrol_bin=/usr/bin/scontrol
    python_bin=/usr/bin/python3
    hostname_bin=/usr/bin/hostname
    grep_bin=/usr/bin/grep
    id_bin=/usr/bin/id
    ;;
  SYNTHETIC_GATE)
    [[ ${CS6_V7A1_SYNTHETIC_GATE:-0} == 1 &&
       -x ${CS6_V7A1_SCONTROL_BIN:-} ]] || {
      echo "V7-A.1 Slurm error: synthetic gate is not explicitly enabled" >&2
      exit 1
    }
    scontrol_bin=${CS6_V7A1_SCONTROL_BIN}
    python_bin=$(command -v python3)
    hostname_bin=$(command -v hostname)
    grep_bin=$(command -v grep)
    id_bin=$(command -v id)
    ;;
  *)
    echo "V7-A.1 Slurm error: unknown execution class" >&2
    exit 1
    ;;
esac
[[ -x $scontrol_bin ]] || {
  echo "V7-A.1 Slurm error: scontrol is unavailable" >&2
  exit 1
}
[[ ${fields[SUBMITTED_JOB_SCRIPT]} == /* &&
   -f ${fields[SUBMITTED_JOB_SCRIPT]} &&
   ! -L ${fields[SUBMITTED_JOB_SCRIPT]} ]] || {
  echo "V7-A.1 Slurm error: submitted job script path is invalid" >&2
  exit 1
}
[[ $(sha256sum "${fields[SUBMITTED_JOB_SCRIPT]}" | awk '{print $1}') == ${fields[JOB_SCRIPT_SHA256]} &&
   $(sha256sum "$0" | awk '{print $1}') == ${fields[JOB_SCRIPT_SHA256]} ]] || {
  echo "V7-A.1 Slurm error: submitted and spooled job-script bytes differ" >&2
  exit 1
}
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ &&
   ${SLURM_JOB_NODELIST:-} == ${fields[SLURM_NODELIST]} &&
   ${SLURMD_NODENAME:-} == ${fields[SLURM_NODELIST]} &&
   ${SLURM_JOB_NUM_NODES:-} == ${fields[SLURM_NODES]} &&
   ${SLURM_NTASKS:-} == ${fields[SLURM_NTASKS]} &&
   ${SLURM_CPUS_PER_TASK:-} == ${fields[SLURM_CPUS_PER_TASK]} &&
   ${SLURM_EXPORT_ENV:-} == NIL &&
   ${SLURM_RESTART_COUNT:-0} == 0 &&
   ${SLURM_CPUS_ON_NODE:-} =~ ^[1-9][0-9]*$ ]] || {
  echo "V7-A.1 Slurm error: allocation environment differs from config" >&2
  exit 1
}
(( fields[JOBS] <= SLURM_CPUS_PER_TASK && fields[JOBS] <= SLURM_CPUS_ON_NODE )) || {
  echo "V7-A.1 Slurm error: attempt concurrency exceeds allocated CPUs" >&2
  exit 1
}
execution_host=$($hostname_bin -s)
[[ $execution_host == ${fields[SLURM_NODELIST]} ]] || {
  echo "V7-A.1 Slurm error: execution host differs from allocation" >&2
  exit 1
}
scontrol_record=$($scontrol_bin -o show job "$SLURM_JOB_ID") || {
  echo "V7-A.1 Slurm error: cannot read the active allocation" >&2
  exit 1
}
[[ -n $scontrol_record && $scontrol_record != *$'\n'* ]] || {
  echo "V7-A.1 Slurm error: noncanonical scontrol record" >&2
  exit 1
}
$scontrol_bin show hostnames "${fields[SLURM_NODELIST]}" | $grep_bin -Fxq "$execution_host" || {
  echo "V7-A.1 Slurm error: execution host is outside the allocation" >&2
  exit 1
}
scontrol_version=$($scontrol_bin --version)
$python_bin -I -B - "$scontrol_record" "$SLURM_JOB_ID" "$($id_bin -u)" \
  "${fields[SLURM_PARTITION]}" "${fields[SLURM_ACCOUNT]}" \
  "${fields[SLURM_QOS]}" "${fields[SLURM_NODELIST]}" \
  "${fields[SLURM_NODES]}" "${fields[SLURM_NTASKS]}" \
  "${fields[SLURM_CPUS_PER_TASK]}" "${fields[JOBS]}" \
  "$SLURM_CPUS_ON_NODE" \
  "${fields[SUBMITTED_JOB_SCRIPT]}" "${fields[SLURM_JOB_NAME]}" \
  "${fields[SLURM_TIME_LIMIT]}" "${fields[SLURM_MIN_MEMORY_NODE]}" \
  "${fields[SLURM_EXCLUSIVE]}" <<'PY'
import re
import shlex
import sys

(record, job_id, uid, partition, account, qos, node, nodes, tasks, cpus, jobs,
 cpus_on_node,
 command, job_name, time_limit, memory, exclusive) = sys.argv[1:]
fields = {}
for token in shlex.split(record):
    if "=" not in token:
        continue
    key, value = token.split("=", 1)
    if not key or not value or key in fields:
        raise SystemExit("V7-A.1 Slurm error: malformed scontrol fields")
    fields[key] = value
expected = {
    "JobId": job_id,
    "JobState": "RUNNING",
    "Partition": partition,
    "Account": account,
    "QOS": qos,
    "NodeList": node,
    "BatchHost": node,
    "NumNodes": nodes,
    "NumTasks": tasks,
    "CPUs/Task": cpus,
    "Requeue": "0",
    "Restarts": "0",
    "Command": command,
    "JobName": job_name,
    "TimeLimit": time_limit,
    "MinMemoryNode": memory,
    "OverSubscribe": "NO",
}
user = re.fullmatch(r"[^()]+\(([0-9]+)\)", fields.get("UserId", ""))
if (
    any(fields.get(key) != value for key, value in expected.items())
    or user is None
    or user.group(1) != uid
    or fields.get("NumCPUs") != cpus_on_node
):
    raise SystemExit("V7-A.1 Slurm error: control-plane allocation differs")
PY
for key in REPO_ARCHIVE PREBUILT_ARCHIVE; do
  [[ -f ${fields[$key]} && ! -L ${fields[$key]} ]] || {
    echo "V7-A.1 Slurm error: input artifact is not regular: $key" >&2
    exit 1
  }
done
[[ ! -e ${fields[OUTPUT_ARCHIVE]} && ! -L ${fields[OUTPUT_ARCHIVE]} ]] || {
  echo "V7-A.1 Slurm error: output archive already exists" >&2
  exit 1
}
[[ ! -e ${fields[OUTPUT_ARCHIVE]}.sha256 && ! -L ${fields[OUTPUT_ARCHIVE]}.sha256 ]] || {
  echo "V7-A.1 Slurm error: output sidecar already exists" >&2
  exit 1
}
[[ $(sha256sum "${fields[REPO_ARCHIVE]}" | awk '{print $1}') == ${fields[REPO_ARCHIVE_SHA256]} ]] || {
  echo "V7-A.1 Slurm error: repository archive digest mismatch" >&2
  exit 1
}
[[ $(sha256sum "${fields[PREBUILT_ARCHIVE]}" | awk '{print $1}') == ${fields[PREBUILT_ARCHIVE_SHA256]} ]] || {
  echo "V7-A.1 Slurm error: prebuilt archive digest mismatch" >&2
  exit 1
}
[[ $(sha256sum "$0" | awk '{print $1}') == ${fields[JOB_SCRIPT_SHA256]} ]] || {
  echo "V7-A.1 Slurm error: job script digest mismatch" >&2
  exit 1
}

scratch=${TMPDIR:-/tmp}/cs6-v7a1-checkpoint-${SLURM_JOB_ID:-manual}-$$
mkdir -m 700 "$scratch"
mkdir -m 700 "$scratch/transport"
cleanup() {
  rm -rf "$scratch"
}
trap cleanup EXIT

mkdir "$scratch/repo"
python3 -I -B - "${fields[REPO_ARCHIVE]}" "$scratch/repo" "${fields[GIT_HEAD]}" <<'PY'
from pathlib import Path, PurePosixPath
import sys
import tarfile

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
expected_head = sys.argv[3]
expected_files = {
    "scripts/research/cs6_hapg_liouville_checkpoint_contract_v1.txt",
    "scripts/research/cs6_hapg_liouville_checkpoint_coordinates_v1.tsv",
    "scripts/research/cs6_hapg_liouville_checkpoint_probe.cpp",
    "scripts/research/cs6_hapg_liouville_checkpoint_verify.py",
    "scripts/research/cs6_hapg_liouville_checkpoint_run.py",
    "scripts/research/cs6_hapg_liouville_checkpoint_retained_verify.py",
    "scripts/research/cs6_hapg_liouville_checkpoint_slurm_job.sh",
    "scripts/research/cs6_plucker_cocycle_verify.py",
}
allowed_directories = {
    prefix
    for name in expected_files
    for prefix in ("scripts", "scripts/research")
    if name.startswith(prefix + "/")
}
with tarfile.open(archive, "r:") as handle:
    if handle.pax_headers.get("comment") != expected_head:
        raise SystemExit("V7-A.1 Slurm error: repository archive commit mismatch")
    members = handle.getmembers()
    seen = set()
    for member in members:
        token = PurePosixPath(member.name)
        if token.is_absolute() or ".." in token.parts:
            raise SystemExit("V7-A.1 Slurm error: unsafe repository archive path")
        if member.isdir() and member.name.rstrip("/") in allowed_directories:
            continue
        if not member.isfile() or member.name not in expected_files or member.name in seen:
            raise SystemExit("V7-A.1 Slurm error: repository archive member differs")
        if member.size > 4 * 1024 * 1024:
            raise SystemExit("V7-A.1 Slurm error: repository archive member exceeds bound")
        seen.add(member.name)
    if seen != expected_files:
        raise SystemExit("V7-A.1 Slurm error: repository archive is incomplete")
    for member in members:
        if not member.isfile():
            continue
        source = handle.extractfile(member)
        if source is None:
            raise SystemExit("V7-A.1 Slurm error: unreadable repository archive member")
        target = destination / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source, target.open("xb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
PY
cmp "$0" "$scratch/repo/scripts/research/cs6_hapg_liouville_checkpoint_slurm_job.sh" || {
  echo "V7-A.1 Slurm error: executed and cloned job scripts differ" >&2
  exit 1
}

mkdir "$scratch/prebuilt"
python3 -I -B - "${fields[PREBUILT_ARCHIVE]}" "$scratch/prebuilt" <<'PY'
from pathlib import Path, PurePosixPath
import sys
import tarfile

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
with tarfile.open(archive, "r:") as handle:
    members = handle.getmembers()
    if not members or len(members) > 64:
        raise SystemExit("V7-A.1 Slurm error: invalid prebuilt member count")
    if sum(member.size for member in members) > 512 * 1024 * 1024:
        raise SystemExit("V7-A.1 Slurm error: prebuilt archive exceeds total bound")
    for member in members:
        token = PurePosixPath(member.name)
        if token.is_absolute() or ".." in token.parts or len(token.parts) != 1 or not member.isfile():
            raise SystemExit("V7-A.1 Slurm error: unsafe prebuilt archive member")
        if len(member.name.encode("utf-8")) > 255 or member.size > 256 * 1024 * 1024:
            raise SystemExit("V7-A.1 Slurm error: prebuilt member exceeds bounds")
    for member in members:
        source = handle.extractfile(member)
        if source is None:
            raise SystemExit("V7-A.1 Slurm error: unreadable prebuilt member")
        target = destination / member.name
        with source, target.open("xb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
PY
chmod 700 "$scratch/prebuilt/worker-binary"

cp "$config" "$scratch/transport/config.txt"
printf '%s\n' "$(sha256sum "$0" | awk '{print $1}')" > "$scratch/transport/job-script.sha256"
printf '%s\n' "$(sha256sum "$config" | awk '{print $1}')" > "$scratch/transport/config.sha256"
printf '%s\n' "$0" > "$scratch/transport/job-script-path.txt"
printf '%s\n' "${fields[SUBMITTED_JOB_SCRIPT]}" \
  > "$scratch/transport/submitted-job-script-path.txt"
printf '%s\n' "$config" > "$scratch/transport/config-path.txt"
printf '%s\n' "$scontrol_record" > "$scratch/transport/scontrol-job.txt"
cp "$0" "$scratch/prebuilt/slurm-job-script.sh"
cp "$config" "$scratch/prebuilt/slurm-config.txt"
cp "${fields[REPO_ARCHIVE]}" "$scratch/prebuilt/repo-source.tar"
printf '%s\n' "$(sha256sum "$scratch/prebuilt/slurm-job-script.sh" | awk '{print $1}')" \
  > "$scratch/prebuilt/slurm-job-script.sha256"
printf '%s\n' "$(sha256sum "$scratch/prebuilt/slurm-config.txt" | awk '{print $1}')" \
  > "$scratch/prebuilt/slurm-config.sha256"
printf '%s\n' "$(sha256sum "$scratch/prebuilt/repo-source.tar" | awk '{print $1}')" \
  > "$scratch/prebuilt/repo-source.sha256"
printf '%s\n' "$scontrol_record" > "$scratch/prebuilt/slurm-control-plane.txt"
printf '%s\n' "$(sha256sum "$scratch/prebuilt/slurm-control-plane.txt" | awk '{print $1}')" \
  > "$scratch/prebuilt/slurm-control-plane.sha256"
uname -a > "$scratch/prebuilt/node-uname.txt"
lscpu > "$scratch/prebuilt/node-lscpu.txt"
ldd "$scratch/prebuilt/worker-binary" > "$scratch/prebuilt/node-runtime-linkage.txt"
! grep -Fq '=> not found' "$scratch/prebuilt/node-runtime-linkage.txt" || {
  echo "V7-A.1 Slurm error: worker has an unresolved node runtime library" >&2
  exit 1
}
python3 -I -B - "$scratch/prebuilt/node-runtime-linkage.txt" \
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
    raise SystemExit("V7-A.1 Slurm error: no hashable node runtime libraries")
Path(sys.argv[2]).write_text("\n".join(sorted(rows)) + "\n", encoding="ascii")
PY

{
  printf 'SCHEMA=sounio.cs6.hapg-liouville-checkpoint-slurm-context.v2\n'
  printf 'EXECUTION_CLASS=%s\n' "${fields[EXECUTION_CLASS]}"
  printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
  printf 'SLURM_JOB_NODELIST=%s\n' "$SLURM_JOB_NODELIST"
  printf 'SLURMD_NODENAME=%s\n' "$SLURMD_NODENAME"
  printf 'EXECUTION_HOST=%s\n' "$execution_host"
  printf 'EXECUTION_UID=%s\n' "$($id_bin -u)"
  printf 'SLURM_JOB_NUM_NODES=%s\n' "$SLURM_JOB_NUM_NODES"
  printf 'SLURM_NTASKS=%s\n' "$SLURM_NTASKS"
  printf 'SLURM_CPUS_ON_NODE=%s\n' "$SLURM_CPUS_ON_NODE"
  printf 'SLURM_CPUS_PER_TASK=%s\n' "$SLURM_CPUS_PER_TASK"
  printf 'SLURM_RESTART_COUNT=%s\n' "${SLURM_RESTART_COUNT:-0}"
  printf 'SLURM_EXPORT_ENV=%s\n' "$SLURM_EXPORT_ENV"
  printf 'SLURM_PARTITION=%s\n' "${fields[SLURM_PARTITION]}"
  printf 'SLURM_ACCOUNT=%s\n' "${fields[SLURM_ACCOUNT]}"
  printf 'SLURM_QOS=%s\n' "${fields[SLURM_QOS]}"
  printf 'SLURM_JOB_NAME=%s\n' "${fields[SLURM_JOB_NAME]}"
  printf 'SLURM_TIME_LIMIT=%s\n' "${fields[SLURM_TIME_LIMIT]}"
  printf 'SLURM_MIN_MEMORY_NODE=%s\n' "${fields[SLURM_MIN_MEMORY_NODE]}"
  printf 'SLURM_EXCLUSIVE=%s\n' "${fields[SLURM_EXCLUSIVE]}"
  printf 'SLURM_COMMAND=%s\n' "${fields[SUBMITTED_JOB_SCRIPT]}"
  printf 'SCONTROL_PATH=%s\n' "$scontrol_bin"
  printf 'SCONTROL_VERSION=%s\n' "$scontrol_version"
} > "$scratch/prebuilt/slurm-context.txt"

runner_extra=()
audit_extra=(--allow-transport-archive)
if [[ ${fields[EXECUTION_CLASS]} == SYNTHETIC_GATE ]]; then
  runner_extra+=(--allow-synthetic-gate)
  audit_extra+=(--allow-synthetic-gate)
fi
set +e
python3 -I -B "$scratch/repo/scripts/research/cs6_hapg_liouville_checkpoint_run.py" \
  --repo "$scratch/repo" \
  --worker "$scratch/prebuilt/worker-binary" \
  --provenance-dir "$scratch/prebuilt" \
  --run-dir "$scratch/result" \
  --root-challenge "${fields[ROOT_CHALLENGE]}" \
  --jobs "${fields[JOBS]}" \
  --timeout-seconds "${fields[TIMEOUT_SECONDS]}" \
  --self-test-mutations \
  --keep-failed \
  "${runner_extra[@]}" \
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
    echo "V7-A.1 Slurm error: runner did not publish its result directory" >&2
    exit 1
  }
  printf 'true\n' > "$scratch/transport/runner-result-present.txt"
  set +e
  python3 -I -B "$scratch/repo/scripts/research/cs6_hapg_liouville_checkpoint_retained_verify.py" \
    --repo "$scratch/repo" \
    "${audit_extra[@]}" \
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
tar --format=ustar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -C "$scratch" -cf "$archive_tmp" "${archive_members[@]}"
archive_sha=$(sha256sum "$archive_tmp" | awk '{print $1}')
mkdir -p "$(dirname "$publication_target")"
python3 -I -B - "$archive_tmp" "$publication_target" <<'PY'
import os
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
stage = target.with_name(f".{target.name}.stage-{os.getpid()}")
descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
        while block := input_handle.read(1024 * 1024):
            output_handle.write(block)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.link(stage, target)
except BaseException:
    stage.unlink(missing_ok=True)
    raise
stage.unlink()
PY
[[ $(sha256sum "$publication_target" | awk '{print $1}') == "$archive_sha" ]] || {
  echo "V7-A.1 Slurm error: published archive digest mismatch" >&2
  exit 1
}
if ! python3 -I -B - "${publication_target}.sha256" "$archive_sha" \
  "$(basename "$publication_target")" <<'PY'
import os
from pathlib import Path
import sys

target = Path(sys.argv[1])
raw = f"{sys.argv[2]}  {sys.argv[3]}\n".encode("ascii")
stage = target.with_name(f".{target.name}.stage-{os.getpid()}")
descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(stage, target)
except BaseException:
    stage.unlink(missing_ok=True)
    raise
stage.unlink()
PY
then
  echo "V7-A.1 Slurm error: sidecar publication failed" >&2
  exit 1
fi

printf '%s_ARCHIVE=%s\n' "$publication_kind" "$publication_target"
printf '%s_ARCHIVE_SHA256=%s\n' "$publication_kind" "$archive_sha"
printf '%s_SIDECAR=%s.sha256\n' "$publication_kind" "$publication_target"
exit "$publication_rc"
