#!/usr/bin/env python3
"""Validate the fresh v5 KAT envelope and emit its adaptive authorization."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path("/tmp/cs6-hapg-stage-v5-7e09b89b")
HEAD = "7e09b89b94a773c6f5609dbc4b98f16dc22a9d5f"
CONTRACT = "5fe8436fe3384663d9046e754f1a27b4ac537e5a51ccc1d90469bd96f7ba3dab"
BASE = "cacd77ffa07966499f4614d3f84e03132bf01d765ca4fabc727c0701a9480389"
DELTA = "f18eca876a0dca56f3ba01ac32cac04f54e04dcc8df243cd49ef0f598f98de52"
PREBUILT = "3f78b0d2e94534e7da4dd483c922aed54fbe2a3cb598a57b0ad862e7625c1688"
JOB_SCRIPT = "906dca0fba614060046495f8651bf095f37d924b9ecb23442ecdd558c856e598"
CONFIG = "0cf799ebf131502cfd30347bc50060bc51a4c5853380ca58baf7dcbcdb1423f0"
ARCHIVE = "cs6-hapg-kat-job8458-7e09b89b94a773c6f5609dbc4b98f16dc22a9d5f.tar"
ARCHIVE_SHA = "f288918070c73636458a61687ae1abf0199038b99801e1392ec4774b26ae7c7f"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n"), path
    for line in raw.decode("ascii").splitlines():
        assert line.count("=") == 1, (path, line)
        key, value = line.split("=", 1)
        assert key and value and key not in result, (path, key)
        result[key] = value
    return result


def require(actual: dict[str, str], expected: dict[str, str]) -> None:
    for key, value in expected.items():
        assert actual.get(key) == value, (key, actual.get(key), value)


archive = ROOT / ARCHIVE
sidecar = ROOT / f"{ARCHIVE}.sha256"
assert sha(archive) == ARCHIVE_SHA
assert sidecar.read_text(encoding="ascii").split() == [ARCHIVE_SHA, ARCHIVE]

with tarfile.open(archive, "r:") as handle:
    names: set[str] = set()
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        assert member.name not in names
        names.add(member.name)
        assert not path.is_absolute() and ".." not in path.parts
        assert member.isdir() or member.isfile()
    assert len(names) == 525

result = ROOT / "kat-extracted" / "result"
transport_path = ROOT / "kat-extracted" / "transport-manifest.txt"
transport_job_path = ROOT / "kat-extracted" / "transport-slurm-job-record.txt"
manifest_path = result / "run-manifest.txt"
contract_path = result / "run-contract.txt"
summary_path = result / "summary.txt"
index_path = result / "files.sha256"
job_record_path = result / "slurm-job-record.txt"

transport = rows(transport_path)
manifest = rows(manifest_path)
contract = rows(contract_path)
summary = rows(summary_path)
sacct = rows(ROOT / "kat-8458.sacct.txt")

require(transport, {
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-transport.v2",
    "MODE": "kat", "SLURM_JOB_ID": "8458",
    "SLURM_NODE": "gpuorangefs-r770-proxmox",
    "EXPECTED_GIT_HEAD": HEAD, "EXPECTED_CONTRACT_SHA256": CONTRACT,
    "SLURM_JOB_SCRIPT_SHA256": JOB_SCRIPT, "CONFIG_SHA256": CONFIG,
    "BASE_REPO_BUNDLE_SHA256": BASE, "REPO_DELTA_BUNDLE_SHA256": DELTA,
    "PREBUILT_ARCHIVE_SHA256": PREBUILT,
    "EXECUTION_PROVENANCE_ATTESTED": "false", "PROMOTION_ELIGIBLE": "false",
})
require(manifest, {
    "RUN_COMPLETE": "true", "MODE": "kat", "FILE_COUNT": "501",
    "EVALUATED_NODE_COUNT": "53", "WAVE_COUNT": "1",
    "EXECUTION_PROVENANCE_ATTESTED": "false", "PROMOTION_ELIGIBLE": "false",
})
require(contract, {
    "FROZEN_CONTRACT_SHA256": CONTRACT, "MODE": "kat",
    "BUILD_MODE": "VERIFIED_PREBUILT_BUNDLE", "SLURM_JOB_ID": "8458",
    "EXECUTION_NODE": "gpuorangefs-r770-proxmox",
    "SLURM_JOB_VERIFIED": "true", "JOBS": "32",
    "EXECUTION_PROVENANCE_ATTESTED": "false", "PROMOTION_ELIGIBLE": "false",
})
require(summary, {
    "MODE": "kat", "BOUNDED_RUN_COMPLETE": "true", "INFRASTRUCTURE_VALID": "true",
    "EVALUATED_NODE_COUNT": "53", "WAVE_COUNT": "1", "HPG_SIGNED_CHART_COUNT": "52",
    "HAPG_ATTEMPTED_COUNT": "52", "HAPG_CERTIFIED_COUNT": "48", "HAPG_RESCUE_COUNT": "20",
    "HPG_MUTATION_TESTS": "4108", "HPG_MUTATIONS_REJECTED": "4108",
    "HAPG_MUTATION_TESTS": "5824", "HAPG_MUTATIONS_REJECTED": "5824",
    "HAPG_FULL_SOURCE_COVER_CANDIDATE": "false", "OPEN_PROBLEM_SOLVED": "false",
    "PROMOTION_ELIGIBLE": "false",
})
require(sacct, {
    "JOB_ID": "8458", "STATE": "COMPLETED", "EXIT_CODE": "0:0",
    "SUBMIT_UTC": "2026-08-02T06:28:07", "START_UTC": "2026-08-02T06:28:07",
    "END_UTC": "2026-08-02T06:29:38", "ELAPSED_SECONDS": "91",
    "NODE": "gpuorangefs-r770-proxmox", "ALLOC_CPUS": "120", "REQ_CPUS": "32",
})

assert (result / "git-head.txt").read_text(encoding="ascii") == f"{HEAD}\n"
assert (result / "git-status.txt").read_bytes() == b""
assert sha(manifest_path) == transport["RESULT_RUN_MANIFEST_SHA256"]
assert sha(index_path) == transport["RESULT_FILES_INDEX_SHA256"]
assert sha(contract_path) == manifest["RUN_CONTRACT_SHA256"]
assert sha(index_path) == manifest["FILES_INDEX_SHA256"]
assert sha(job_record_path) == contract["SLURM_JOB_RECORD_SHA256"]

indexed: dict[str, str] = {}
for line in index_path.read_text(encoding="ascii").splitlines():
    digest, rel = line.split("  ", 1)
    path = PurePosixPath(rel)
    assert len(digest) == 64 and rel not in indexed
    assert not path.is_absolute() and ".." not in path.parts
    indexed[rel] = digest
actual = {p.relative_to(result).as_posix() for p in result.rglob("*") if p.is_file()}
assert set(indexed) == actual - {"files.sha256", "run-manifest.txt"}
for rel, digest in indexed.items():
    assert sha(result / rel) == digest, rel

authorization = {
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-kat-authorization.v1",
    "AUTHORIZATION_SCOPE": "ONE_V5_ADAPTIVE_SLURM_SUBMISSION",
    "EXECUTION_GIT_HEAD": HEAD,
    "FROZEN_CONTRACT_SHA256": CONTRACT,
    "BASE_REPO_BUNDLE_SHA256": BASE,
    "REPO_DELTA_BUNDLE_SHA256": DELTA,
    "PREBUILT_ARCHIVE_SHA256": PREBUILT,
    "SLURM_JOB_SCRIPT_SHA256": JOB_SCRIPT,
    "KAT_JOB_ID": "8458",
    "KAT_STATE": "COMPLETED",
    "KAT_EXIT_CODE": "0:0",
    "KAT_SUBMIT_UTC": sacct["SUBMIT_UTC"],
    "KAT_START_UTC": sacct["START_UTC"],
    "KAT_END_UTC": sacct["END_UTC"],
    "KAT_NODE": sacct["NODE"],
    "KAT_ALLOC_CPUS": sacct["ALLOC_CPUS"],
    "KAT_CONFIG_SHA256": CONFIG,
    "KAT_ARCHIVE_BASENAME": ARCHIVE,
    "KAT_ARCHIVE_SHA256": ARCHIVE_SHA,
    "KAT_ARCHIVE_SIDECAR_SHA256": sha(sidecar),
    "KAT_TRANSPORT_MANIFEST_SHA256": sha(transport_path),
    "KAT_TRANSPORT_JOB_RECORD_SHA256": sha(transport_job_path),
    "KAT_RESULT_RUN_MANIFEST_SHA256": sha(manifest_path),
    "KAT_RESULT_FILES_INDEX_SHA256": sha(index_path),
    "KAT_RESULT_RUN_CONTRACT_SHA256": sha(contract_path),
    "KAT_RESULT_SUMMARY_SHA256": sha(summary_path),
    "KAT_RESULT_JOB_RECORD_SHA256": sha(job_record_path),
    "KAT_SACCT_RECORD_SHA256": sha(ROOT / "kat-8458.sacct.txt"),
    "KAT_STDOUT_SHA256": sha(ROOT / "kat-8458.out"),
    "KAT_STDERR_SHA256": sha(ROOT / "kat-8458.err"),
    "KAT_GATE_PASS": "true",
}
for key, value in authorization.items():
    print(f"{key}={value}")
