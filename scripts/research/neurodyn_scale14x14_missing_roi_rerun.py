#!/usr/bin/env python3
"""Rerun the ds006731 14x14 missing ROI subjects on the r740 worker.

This is orchestration plumbing only.  It keeps the scientific preprocessing
profile used by the existing P3 receipts, but fixes the durability gap by
copying each successful ROI artifact to CephFS before moving to the next
subject.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_NAMESPACE = "slurm-pilot"
DEFAULT_POD = "slurm-pilot-worker-gpuorangefs-multi-mpttj"
DEFAULT_CONTAINER = "slurmd"
DEFAULT_REMOTE_ROOT = "/tmp/neurodyn-ds006731-r740"
DEFAULT_DURABLE_BASE = "/mnt/cephfs/neurodyn/derivatives/ds006731"
DEFAULT_SELECTED_TSV = (
    "artifacts/research/neurodyn/p3_candidates/"
    "ds006731_p3_contract_20260705T102500Z/"
    "balanced_8x8_worker_20260705T113600Z/selected_subjects.tsv"
)
DEFAULT_SUBJECTS = ["HVS7", "T011", "T015", "T016", "T017", "T018", "T019"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--pod", default=DEFAULT_POD)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--durable-base", default=DEFAULT_DURABLE_BASE)
    parser.add_argument("--durable-dir", default="")
    parser.add_argument("--selected-tsv", default=DEFAULT_SELECTED_TSV)
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-fmriprep", action="store_true")
    return parser.parse_args()


def now_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def normalize_subject(subject: str) -> str:
    subject = subject.strip()
    if not subject:
        raise ValueError("empty subject")
    return subject if subject.startswith("sub-") else f"sub-{subject}"


def task_from_bold_path(path: str) -> str:
    match = re.search(r"_task-([^_]+)_bold\.nii\.gz$", path)
    if not match:
        raise ValueError(f"cannot infer task from BOLD path: {path}")
    return match.group(1)


def read_subject_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = {}
        for row in reader:
            subject = normalize_subject(row["subject_id"])
            row["subject_id"] = subject
            row["task"] = task_from_bold_path(row["bold_path"])
            rows[subject] = row
        return rows


def run(cmd: list[str], *, dry_run: bool, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def kubectl(args: argparse.Namespace, remote_command: str, *, dry_run: bool, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(
        [
            "kubectl",
            "-n",
            args.namespace,
            "exec",
            args.pod,
            "-c",
            args.container,
            "--",
            "sh",
            "-lc",
            remote_command,
        ],
        dry_run=dry_run,
        check=check,
    )


def remote_python_download_script(root: str, row: dict[str, str]) -> str:
    files = []
    for key in ("bold_path", "anat_path"):
        rel = row[key]
        files.append(rel)
        if rel.endswith(".nii.gz"):
            files.append(rel.removesuffix(".nii.gz") + ".json")
    payload = json.dumps({"root": root, "files": files})
    return rf"""python3 - <<'PY'
import json
import os
import urllib.request
from pathlib import Path

payload = json.loads({payload!r})
root = Path(payload["root"])
base = "https://s3.amazonaws.com/openneuro.org/ds006731"
for rel in payload["files"]:
    target = root / "bids" / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"download_skip existing {{target}}")
        continue
    url = f"{{base}}/{{rel}}"
    tmp = target.with_suffix(target.suffix + ".tmp")
    req = urllib.request.Request(url, headers={{"User-Agent": "sounio-neurodyn-scale14x14/1.0"}})
    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(tmp, target)
    print(f"downloaded {{target}} bytes={{target.stat().st_size}}")
PY"""


def remote_participants_script(root: str, rows: list[dict[str, str]]) -> str:
    participants = [
        {
            "participant_id": row["subject_id"],
            "group": "MDD" if row["label"] == "1" else "HV",
            "site": "ds006731",
        }
        for row in rows
    ]
    payload = json.dumps({"root": root, "participants": participants})
    return rf"""python3 - <<'PY'
import csv
import json
from pathlib import Path

payload = json.loads({payload!r})
path = Path(payload["root"]) / "bids" / "participants.tsv"
path.parent.mkdir(parents=True, exist_ok=True)
existing = {{}}
if path.exists():
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            existing[row["participant_id"]] = row
for row in payload["participants"]:
    existing[row["participant_id"]] = row
with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["participant_id", "group", "site"], delimiter="\t")
    writer.writeheader()
    for subject in sorted(existing):
        row = existing[subject]
        writer.writerow({{
            "participant_id": subject,
            "group": row.get("group", ""),
            "site": row.get("site", "") or "ds006731",
        }})
print(f"participants_tsv={{path}} rows={{len(existing)}}")
PY"""


def fmriprep_command(root: str, subject: str) -> str:
    label = subject.removeprefix("sub-")
    log = f"{root}/fmriprep_{subject}_full_sloppy.log"
    rc = f"{root}/fmriprep_{subject}_full_sloppy.rc"
    return " ".join(
        [
            "set -u;",
            f"cd {shlex.quote(root)};",
            "mkdir -p out work singularity-cache;",
            "export SINGULARITY_CACHEDIR=./singularity-cache;",
            "export TEMPLATEFLOW_HOME=/templateflow;",
            "singularity run --cleanenv",
            f"-B {shlex.quote(root + '/bids')}:/data:ro",
            f"-B {shlex.quote(root + '/out')}:/out",
            f"-B {shlex.quote(root + '/work')}:/work",
            f"-B {shlex.quote(root + '/license')}:/license:ro",
            f"-B {shlex.quote(root + '/templateflow')}:/templateflow",
            f"{shlex.quote(root + '/fmriprep-24.1.1.sif')}",
            "/data /out participant",
            f"--participant-label {shlex.quote(label)}",
            "--skip-bids-validation",
            "--fs-license-file /license/license.txt",
            "--fs-no-reconall --sloppy",
            "--nthreads 12 --omp-nthreads 4 --mem-mb 48000",
            "--notrack",
            "--output-spaces MNI152NLin2009cAsym:res-2",
            "-w /work",
            f"> {shlex.quote(log)} 2>&1;",
            "run_rc=$?;",
            f"echo $run_rc > {shlex.quote(rc)};",
            "exit $run_rc",
        ]
    )


def roi_extract_command(root: str, durable_dir: str, row: dict[str, str]) -> str:
    subject = row["subject_id"]
    task = row["task"]
    group = "MDD" if row["label"] == "1" else "HV"
    participant_file = f"{root}/participants_{subject}.tsv"
    return " ".join(
        [
            "set -u;",
            f"printf 'participant_id\\tgroup\\tsite\\n{subject}\\t{group}\\tds006731\\n' > {shlex.quote(participant_file)};",
            f"singularity exec -B {shlex.quote(root + '/out')}:/fmriprep:ro",
            f"-B {shlex.quote(root + '/atlas')}:/atlas:ro",
            f"-B {shlex.quote(root)}:/pilot",
            f"-B {shlex.quote(root + '/roi')}:/roi",
            f"{shlex.quote(root + '/fmriprep-24.1.1.sif')}",
            "python /pilot/extract_fmriprep_roi_timeseries.py",
            "--fmriprep-dir /fmriprep",
            f"--participants-tsv /pilot/{Path(participant_file).name}",
            "--atlas-image /atlas/schaefer_2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
            "--output-roi-dir /roi",
            f"--phenotypic-output /pilot/roi_{subject}_phenotypic.csv",
            f"--subject-id {shlex.quote(subject)}",
            f"--task {shlex.quote(task)}",
            "--confounds-mode motion",
            "--standardize",
            "--detrend",
            "--fail-on-empty",
            f"> {shlex.quote(root + f'/roi_extract_{subject}.log')} 2>&1;",
            "roi_rc=$?;",
            f"echo $roi_rc > {shlex.quote(root + f'/roi_extract_{subject}.rc')};",
            "test \"$roi_rc\" = 0;",
            f"test \"$(cat {shlex.quote(root + f'/roi_extract_{subject}.rc')})\" = 0;",
            f"mkdir -p {shlex.quote(durable_dir + '/roi')} {shlex.quote(durable_dir + '/logs')} {shlex.quote(durable_dir + '/rc')};",
            f"cp {shlex.quote(root + f'/roi/{subject}.1D')} {shlex.quote(durable_dir + f'/roi/{subject}.1D')};",
            f"cp {shlex.quote(root + f'/roi_{subject}_phenotypic.csv')} {shlex.quote(durable_dir + '/')};",
            f"cp {shlex.quote(root + f'/roi_extract_{subject}.rc')} {shlex.quote(durable_dir + f'/rc/')};",
            f"cp {shlex.quote(root + f'/roi_extract_{subject}.log')} {shlex.quote(durable_dir + f'/logs/')};",
            f"cp {shlex.quote(root + f'/fmriprep_{subject}_full_sloppy.rc')} {shlex.quote(durable_dir + f'/rc/')} 2>/dev/null || true;",
            f"cp {shlex.quote(root + f'/fmriprep_{subject}_full_sloppy.log')} {shlex.quote(durable_dir + f'/logs/')} 2>/dev/null || true;",
            f"sha256sum {shlex.quote(durable_dir + f'/roi/{subject}.1D')} > {shlex.quote(durable_dir + f'/roi/{subject}.1D.sha256')};",
        ]
    )


def main() -> int:
    args = parse_args()
    selected = read_subject_rows(Path(args.selected_tsv))
    subjects = [normalize_subject(item) for item in (args.subject or DEFAULT_SUBJECTS)]
    rows = []
    for subject in subjects:
        if subject not in selected:
            raise SystemExit(f"subject not found in selected TSV: {subject}")
        rows.append(selected[subject])
    durable_dir = args.durable_dir or f"{args.durable_base}/scale14x14_missing_{now_stamp()}"

    manifest = {
        "schema": "brain_ossm.neurodyn.scale14x14_missing_roi_rerun.v1",
        "created_at_utc": now_stamp(),
        "namespace": args.namespace,
        "pod": args.pod,
        "container": args.container,
        "remote_root": args.remote_root,
        "durable_dir": durable_dir,
        "subjects": [{"subject_id": row["subject_id"], "label": row["label"], "task": row["task"]} for row in rows],
        "claim_boundary": "ROI preprocessing only. No clinical, mechanistic, or replicated O-SSM claim.",
    }
    print(json.dumps(manifest, indent=2), flush=True)

    kubectl(args, f"mkdir -p {shlex.quote(durable_dir)}", dry_run=args.dry_run)
    kubectl(args, remote_participants_script(args.remote_root, rows), dry_run=args.dry_run)

    for row in rows:
        subject = row["subject_id"]
        print(f"== subject {subject} task={row['task']} ==", flush=True)
        kubectl(args, remote_python_download_script(args.remote_root, row), dry_run=args.dry_run)
        if not args.skip_fmriprep:
            kubectl(args, f"test -s {shlex.quote(args.remote_root + f'/out/{subject}.html')} && echo fmriprep_skip_existing || ({fmriprep_command(args.remote_root, subject)})", dry_run=args.dry_run)
        kubectl(args, roi_extract_command(args.remote_root, durable_dir, row), dry_run=args.dry_run)

    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    kubectl(
        args,
        f"cat > {shlex.quote(durable_dir + '/rerun_manifest.json')} <<'JSON'\n{manifest_json}\nJSON\n"
        f"find {shlex.quote(durable_dir)} -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > {shlex.quote(durable_dir + '/SHA256SUMS')}",
        dry_run=args.dry_run,
    )
    print(f"durable_dir={durable_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
