"""Read-only terminal packet audit for gen2 CPU envelope v1.

This does not establish Slurm completion, banner execution, CI qualification,
or a causal comparison. Run only after result.json has been written.
"""
import argparse
import hashlib
import json
import pathlib
import tarfile

EXPECTED = {
    "protocol_sha256": "3e09ae0d861c2a72b226c3e3f82f46750b9f2031b2a8b53912715c2338cd4c9c",
    "runner_sha256": "74ade8a0f872fb9f2818452df1a6c0e73c891657724b8d605475ebd12f6472ea",
    "job": "11991",
    "boot_id": "c8e3d8c4-8c29-4caf-bbd5-95c99af8a003",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    require(path.is_file() and not path.is_symlink(), "missing or linked file: " + str(path))
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit(root, expected=EXPECTED):
    root = pathlib.Path(root)
    protocol_hash = sha(root / "protocol.json")
    require(protocol_hash == expected["protocol_sha256"], "protocol mismatch")
    protocol = json.loads((root / "protocol.json").read_text())
    require(sha(root / "measure.py") == expected["runner_sha256"], "runner mismatch")
    for name, pin in protocol["files_sha256"].items():
        require(sha(root / name) == pin, "input mismatch: " + name)
    entry = json.loads((root / "attempt-entered.json").read_text())
    result = json.loads((root / "result.json").read_text())
    for key, value in expected.items():
        require(entry[key] == value, "attempt identity mismatch: " + key)
    require(entry["hostname"] == protocol["node"], "hostname mismatch")
    require(result["job"] == expected["job"], "result job mismatch")
    allocation = entry["allocation"]
    require(allocation["SLURM_JOB_ID"] == expected["job"], "allocation job mismatch")
    require(int(allocation["SLURM_CPUS_PER_TASK"]) == protocol["cpus"], "CPU allocation mismatch")
    require(int(allocation["SLURM_MEM_PER_NODE"]) == protocol["slurm_memory_mib"], "memory allocation mismatch")
    for key in ("boot_unchanged", "source_files_unchanged", "compiler_unchanged"):
        require(result[key] is True, "runner custody failed: " + key)
    for key in ("ci_qualified", "inkling_qualified", "causal_claim"):
        require(result[key] is False, "unsupported promotion: " + key)

    # Independently check the current source tree against the pinned archive,
    # including added files, which the runner's initial file map cannot detect.
    source_names = set()
    with tarfile.open(root / "source.tar") as archive:
        for member in archive:
            path = pathlib.PurePosixPath(member.name)
            require(not path.is_absolute() and ".." not in path.parts, "unsafe archive path")
            require(member.isdir() or member.isfile(), "unsupported archive member")
            if member.isfile():
                name = str(path)
                require(name not in source_names, "duplicate source member")
                source_names.add(name)
                with archive.extractfile(member) as handle:
                    pin = hashlib.file_digest(handle, "sha256").hexdigest()
                require(sha(root / "tree" / name) == pin, "source mismatch: " + name)
    actual = set()
    for path in (root / "tree").rglob("*"):
        require(not path.is_symlink(), "linked source path")
        if not path.is_dir():
            require(path.is_file(), "unsupported source path")
            actual.add(str(path.relative_to(root / "tree")))
    require(actual == source_names, "source inventory mismatch")

    artifact = root / "madaros.gen2"
    exists = artifact.is_file() and not artifact.is_symlink()
    require(result["artifact_exists"] is exists, "artifact presence mismatch")
    artifact_sha = sha(artifact) if exists else None
    require(result["artifact_sha256"] == artifact_sha, "artifact hash mismatch")
    require(type(result["compiler_rc"]) is int, "invalid compiler exit code")
    require(type(result["timed_out"]) is bool, "invalid timeout flag")
    complete = result["compiler_rc"] == 0 and not result["timed_out"] and exists and artifact.stat().st_size > 0
    require(result["compile_complete"] is complete, "inconsistent completion claim")
    require(result["compiler_rc"] != 0 or complete, "exit zero without completed artifact")
    receipt_files = ["protocol.json", "measure.py", "source.tar", "madaros",
                     "attempt-entered.json", "result.json", "compile.log", "samples.jsonl"]
    if exists:
        receipt_files.append("madaros.gen2")
    return {
        "schema": "pireus-gen2-cpu-terminal-packet-audit-v1",
        "job": expected["job"], "source": protocol["source"],
        "packet_integrity_verified": True, "compile_complete": complete,
        "compiler_rc": result["compiler_rc"], "timed_out": result["timed_out"],
        "source_files_verified": len(source_names),
        "files_sha256": {name: sha(root / name) for name in receipt_files},
        "scheduler_terminal_verified": False, "banner_verified": False,
        "ci_qualified": False, "inkling_qualified": False, "causal_claim": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.packet), indent=2))


if __name__ == "__main__":
    main()
