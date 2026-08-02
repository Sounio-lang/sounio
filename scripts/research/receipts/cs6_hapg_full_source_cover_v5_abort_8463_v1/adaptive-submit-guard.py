#!/usr/bin/env python3
"""Fail closed unless the fresh KAT authorizes this exact adaptive submission."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


ROOT = Path("/orangefs/training/cs6-hapg-cover/7e09b89b94a773c6")
BASE = Path("/orangefs/training/cs6-hapg-cover/6ca2515af28d58d0/repo.bundle")
HEAD = "7e09b89b94a773c6f5609dbc4b98f16dc22a9d5f"
CONTRACT = "5fe8436fe3384663d9046e754f1a27b4ac537e5a51ccc1d90469bd96f7ba3dab"
AUTH_SHA = "7073311d87709fc0d583ca2b037280b374308371c8e2301c4fe0a1d4fe1fb61d"
ADAPTIVE_CONFIG_SHA = "2154027e1fcea4e5e21375427355fce045a20da4975008f9f710291c310f94ce"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_bytes(data: bytes, label: str) -> dict[str, str]:
    assert b"\r" not in data and data.endswith(b"\n"), label
    result: dict[str, str] = {}
    for line in data.decode("ascii").splitlines():
        assert line.count("=") == 1, (label, line)
        key, value = line.split("=", 1)
        assert key and value and key not in result, (label, key)
        result[key] = value
    return result


def rows(path: Path) -> dict[str, str]:
    return rows_bytes(path.read_bytes(), str(path))


def require(actual: dict[str, str], expected: dict[str, str]) -> None:
    for key, value in expected.items():
        assert actual.get(key) == value, (key, actual.get(key), value)


auth_path = ROOT / "kat-authorization.txt"
assert sha(auth_path) == AUTH_SHA
auth = rows(auth_path)
require(auth, {
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-kat-authorization.v1",
    "AUTHORIZATION_SCOPE": "ONE_V5_ADAPTIVE_SLURM_SUBMISSION",
    "EXECUTION_GIT_HEAD": HEAD,
    "FROZEN_CONTRACT_SHA256": CONTRACT,
    "KAT_JOB_ID": "8458", "KAT_STATE": "COMPLETED", "KAT_EXIT_CODE": "0:0",
    "KAT_SUBMIT_UTC": "2026-08-02T06:28:07",
    "KAT_START_UTC": "2026-08-02T06:28:07",
    "KAT_END_UTC": "2026-08-02T06:29:38",
    "KAT_NODE": "gpuorangefs-r770-proxmox", "KAT_ALLOC_CPUS": "120",
    "KAT_GATE_PASS": "true",
})

expected_files = {
    BASE: auth["BASE_REPO_BUNDLE_SHA256"],
    ROOT / "repo-v5.delta": auth["REPO_DELTA_BUNDLE_SHA256"],
    ROOT / "prebuilt.tar": auth["PREBUILT_ARCHIVE_SHA256"],
    ROOT / "hapg-job.sh": auth["SLURM_JOB_SCRIPT_SHA256"],
    ROOT / "kat-config.txt": auth["KAT_CONFIG_SHA256"],
    ROOT / "adaptive-config.txt": ADAPTIVE_CONFIG_SHA,
}
for path, digest in expected_files.items():
    assert path.is_file() and not path.is_symlink(), path
    assert sha(path) == digest, path

adaptive = rows(ROOT / "adaptive-config.txt")
require(adaptive, {
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-slurm-config.v2",
    "MODE": "adaptive", "BASE_REPO_BUNDLE_SHA256": auth["BASE_REPO_BUNDLE_SHA256"],
    "BASE_GIT_HEAD": "6ca2515af28d58d025097f94c73025c0f5bc266d",
    "REPO_DELTA_BUNDLE_SHA256": auth["REPO_DELTA_BUNDLE_SHA256"],
    "PREBUILT_ARCHIVE_SHA256": auth["PREBUILT_ARCHIVE_SHA256"],
    "EXPECTED_GIT_HEAD": HEAD, "EXPECTED_CONTRACT_SHA256": CONTRACT,
})
assert len(adaptive) == 12

archive = ROOT / "results" / auth["KAT_ARCHIVE_BASENAME"]
sidecar = Path(f"{archive}.sha256")
assert sha(archive) == auth["KAT_ARCHIVE_SHA256"]
assert sha(sidecar) == auth["KAT_ARCHIVE_SIDECAR_SHA256"]
assert sidecar.read_text(encoding="ascii").split() == [auth["KAT_ARCHIVE_SHA256"], archive.name]

with tarfile.open(archive, "r:") as handle:
    members = handle.getmembers()
    member_map = {member.name: member for member in members}
    assert len(members) == len(member_map) == 525
    for member in members:
        path = PurePosixPath(member.name)
        assert not path.is_absolute() and ".." not in path.parts
        assert member.isdir() or member.isfile()

    def payload(name: str) -> bytes:
        member = member_map[name]
        assert member.isfile()
        extracted = handle.extractfile(member)
        assert extracted is not None
        return extracted.read()

    transport_data = payload("transport-manifest.txt")
    transport_job_data = payload("transport-slurm-job-record.txt")
    manifest_data = payload("result/run-manifest.txt")
    contract_data = payload("result/run-contract.txt")
    summary_data = payload("result/summary.txt")
    index_data = payload("result/files.sha256")
    job_record_data = payload("result/slurm-job-record.txt")
    assert sha_bytes(transport_data) == auth["KAT_TRANSPORT_MANIFEST_SHA256"]
    assert sha_bytes(transport_job_data) == auth["KAT_TRANSPORT_JOB_RECORD_SHA256"]
    assert sha_bytes(manifest_data) == auth["KAT_RESULT_RUN_MANIFEST_SHA256"]
    assert sha_bytes(index_data) == auth["KAT_RESULT_FILES_INDEX_SHA256"]
    assert sha_bytes(contract_data) == auth["KAT_RESULT_RUN_CONTRACT_SHA256"]
    assert sha_bytes(summary_data) == auth["KAT_RESULT_SUMMARY_SHA256"]
    assert sha_bytes(job_record_data) == auth["KAT_RESULT_JOB_RECORD_SHA256"]

    transport = rows_bytes(transport_data, "transport-manifest.txt")
    manifest = rows_bytes(manifest_data, "result/run-manifest.txt")
    contract = rows_bytes(contract_data, "result/run-contract.txt")
    summary = rows_bytes(summary_data, "result/summary.txt")
    require(transport, {
        "MODE": "kat", "SLURM_JOB_ID": "8458", "EXPECTED_GIT_HEAD": HEAD,
        "EXPECTED_CONTRACT_SHA256": CONTRACT,
        "CONFIG_SHA256": auth["KAT_CONFIG_SHA256"],
        "BASE_REPO_BUNDLE_SHA256": auth["BASE_REPO_BUNDLE_SHA256"],
        "REPO_DELTA_BUNDLE_SHA256": auth["REPO_DELTA_BUNDLE_SHA256"],
        "PREBUILT_ARCHIVE_SHA256": auth["PREBUILT_ARCHIVE_SHA256"],
        "RESULT_RUN_MANIFEST_SHA256": auth["KAT_RESULT_RUN_MANIFEST_SHA256"],
        "RESULT_FILES_INDEX_SHA256": auth["KAT_RESULT_FILES_INDEX_SHA256"],
        "PROMOTION_ELIGIBLE": "false",
    })
    require(manifest, {
        "RUN_COMPLETE": "true", "MODE": "kat", "FILE_COUNT": "501",
        "EVALUATED_NODE_COUNT": "53", "WAVE_COUNT": "1",
        "RUN_CONTRACT_SHA256": auth["KAT_RESULT_RUN_CONTRACT_SHA256"],
        "FILES_INDEX_SHA256": auth["KAT_RESULT_FILES_INDEX_SHA256"],
    })
    require(contract, {
        "FROZEN_CONTRACT_SHA256": CONTRACT, "MODE": "kat",
        "SLURM_JOB_ID": "8458", "SLURM_JOB_VERIFIED": "true",
        "SLURM_JOB_RECORD_SHA256": auth["KAT_RESULT_JOB_RECORD_SHA256"],
    })
    require(summary, {
        "BOUNDED_RUN_COMPLETE": "true", "INFRASTRUCTURE_VALID": "true",
        "EVALUATED_NODE_COUNT": "53", "HPG_SIGNED_CHART_COUNT": "52",
        "HAPG_ATTEMPTED_COUNT": "52", "HAPG_CERTIFIED_COUNT": "48",
        "HAPG_RESCUE_COUNT": "20", "HPG_MUTATION_TESTS": "4108",
        "HPG_MUTATIONS_REJECTED": "4108", "HAPG_MUTATION_TESTS": "5824",
        "HAPG_MUTATIONS_REJECTED": "5824", "OPEN_PROBLEM_SOLVED": "false",
    })

    indexed: dict[str, str] = {}
    for line in index_data.decode("ascii").splitlines():
        digest, rel = line.split("  ", 1)
        path = PurePosixPath(rel)
        assert rel not in indexed and not path.is_absolute() and ".." not in path.parts
        indexed[rel] = digest
    assert len(indexed) == 501
    for rel, digest in indexed.items():
        assert sha_bytes(payload(f"result/{rel}")) == digest, rel

sacct = subprocess.check_output([
    "sacct", "-X", "-j", "8458", "--starttime", "2026-08-02",
    "--parsable2", "--noheader",
    "-o", "JobIDRaw,JobName,Partition,State,ExitCode,Submit,Start,End,ElapsedRaw,NodeList,AllocCPUS,ReqCPUS",
], text=True).strip().splitlines()
assert sacct == [
    "8458|cs6-hapg-kat-v5|gpu-orangefs|COMPLETED|0:0|2026-08-02T06:28:07|"
    "2026-08-02T06:28:07|2026-08-02T06:29:38|91|gpuorangefs-r770-proxmox|120|32"
]
assert not (ROOT / "adaptive-submission.txt").exists()
assert not list((ROOT / "results").glob(f"cs6-hapg-adaptive-job*-{HEAD}.tar*"))
print(f"HAPG_V5_KAT_AUTH_SHA256={AUTH_SHA}")
print("HAPG_V5_ADAPTIVE_SUBMISSION_AUTHORIZED=true")
