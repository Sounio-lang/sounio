#!/usr/bin/env python3
"""Validate the frozen V7-B bridge ledger against V7-A.1 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


REPO_REL = {
    "contract": Path("scripts/research/cs6_v7b_full_hpg_bridge_contract_v1.txt"),
    "ledger": Path("scripts/research/cs6_v7b_full_hpg_bridge_ledger_v1.tsv"),
    "parent_summary": Path(
        "scripts/research/receipts/cs6_hapg_liouville_checkpoint_job_8496_v1/result/summary.txt"
    ),
    "parent_results": Path(
        "scripts/research/receipts/cs6_hapg_liouville_checkpoint_job_8496_v1/result/results.tsv"
    ),
    "parent_decisions": Path(
        "scripts/research/receipts/cs6_hapg_liouville_checkpoint_job_8496_v1/result/decisions.tsv"
    ),
    "parent_attempt_contract": Path(
        "scripts/research/receipts/cs6_hapg_liouville_checkpoint_job_8496_v1/result/attempt-contract.tsv"
    ),
    "parent_report": Path("docs/research/cs6_hapg_liouville_checkpoint_2026-08-02.md"),
}

EXPECTED_DIGESTS = {
    "parent_report": "4f31c72ab42e992ba761981cd0608a6bf43f4167c4ba252cf33da5e41e3d8ad8",
    "parent_summary": "6324f38df8370e7716a1cf88c94f1ba6366415c27c98e4974c37367708e4f554",
    "parent_results": "dad954fc00086da081409a9cb6ba94cfb3e7aa799b7d6526588410ccaeec6aaa",
    "parent_decisions": "e11c568ca78e311640dd5f5ea5e501d9cdc128c2ef9ca68ab5be3d4c999efa5f",
    "parent_attempt_contract": "972801088390565d63df153316efeebf71511dc9ac01f718d58871213296310c",
}

EXPECTED_PARENT_SUMMARY = {
    "SCHEMA": "sounio.cs6.hapg-liouville-checkpoint-summary.v1",
    "RUN_COMPLETE": "true",
    "RUN_VALID": "true",
    "ATTEMPTS_COMPLETED": "9",
    "VERIFIED_CHECKPOINTS": "6",
    "BOUND_RQ_NAN": "3",
    "OUTCOME": "BOTH_ALTERNATIVES_PASS",
    "MUTATION_TESTS": "276",
    "MUTATIONS_REJECTED": "276",
    "C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED": "false",
    "FULL_HPG_PIPELINE_EVALUATED": "false",
    "V7_B_ELIGIBILITY": "false",
    "V7_B_WINNER": "NONE",
    "PROMOTION_ELIGIBLE": "false",
    "OPEN_PROBLEM_SOLVED": "false",
    "FPGA_EXECUTION": "false",
}

EXPECTED_CONTRACT = {
    "SCHEMA": "sounio.cs6.v7b-full-hpg-bridge-contract.v1",
    "CONTRACT_STATE": "PRE_EXECUTION_FROZEN",
    "BASE_COMMIT": "77c985ae24803ee7f4d1499f8de7983a3e895696",
    "PARENT_V7A1_SLURM_JOB": "8496",
    "PARENT_V7A1_CONTRACT_SHA256": "3afc0475847ad8054234a2ddfa108b768cfd81991d0be71fc21c991f363631ce",
    "PARENT_V7A1_COORDINATES_SHA256": "527afc7c205fcf09b15a0bff91df6935f19ed2b7e7926895916ac5da33a992a7",
    "CELL_COUNT": "3",
    "CANDIDATE_CARRIERS": "C0HORect2Set,C0Rect2Set",
    "BRIDGE_CELL_CARRIER_COUNT": "6",
    "BRIDGE_BOUNDARY_ORDER": "LIOUVILLE_CHECKPOINT,C1_BOUNDARY,C2_BOUNDARY,SECTION_RESIDENT_CROSSING",
    "LEDGER_ROW_COUNT": "24",
    "PARENT_LIOUVILLE_ROWS_SATISFIED": "6",
    "PROSPECTIVE_ROWS_REQUIRED": "18",
    "MAXIMUM_PROSPECTIVE_EVALUATIONS": "18",
    "V7B_FREEZE_ONLY": "true",
    "V7B_EXECUTED": "false",
    "C1_EXECUTED": "false",
    "C2_EXECUTED": "false",
    "DOWNSTREAM_SECTION_RESIDENT_EXECUTED": "false",
    "V7_B_ELIGIBILITY": "false",
    "V7_B_WINNER": "NONE",
    "PROMOTION_ELIGIBLE": "false",
    "OPEN_PROBLEM_SOLVED": "false",
    "FPGA_EXECUTION": "false",
}

EXPECTED_BOUNDARIES = (
    "LIOUVILLE_CHECKPOINT",
    "C1_BOUNDARY",
    "C2_BOUNDARY",
    "SECTION_RESIDENT_CROSSING",
)
EXPECTED_CARRIERS = ("C0HORect2Set", "C0Rect2Set")
EXPECTED_PARENT_ORDINALS = ("22", "23", "24")
SATISFIED = "SATISFIED_BY_V7A1"
REQUIRED = "REQUIRED_UNRUN"


def fail(message: str) -> None:
    raise SystemExit(f"V7-B bridge gate error: {message}")


def canonical(path: Path) -> bytes:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"noncanonical text bytes: {path}")
    if not raw.isascii():
        fail(f"non-ASCII text: {path}")
    return raw


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in canonical(path).decode("ascii").splitlines():
        if line.count("=") != 1:
            fail(f"malformed KV line in {path}: {line}")
        key, value = line.split("=", 1)
        if not key or not value or key in result:
            fail(f"invalid or duplicate KV key in {path}: {key}")
        result[key] = value
    return result


def parse_prefaced_tsv(path: Path, header_prefix: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    lines = canonical(path).decode("ascii").splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.startswith(header_prefix)), -1)
    if header_index < 0:
        fail(f"table header missing in {path}")
    meta: dict[str, str] = {}
    for line in lines[:header_index]:
        if line.count("=") != 1:
            fail(f"malformed metadata line in {path}: {line}")
        key, value = line.split("=", 1)
        if not key or not value or key in meta:
            fail(f"invalid or duplicate metadata key in {path}: {key}")
        meta[key] = value
    rows = list(csv.DictReader(lines[header_index:], delimiter="\t"))
    return meta, rows


def write_kv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}={value}\n" for key, value in rows), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", type=Path)
    args = parser.parse_args()

    root = Path.cwd()
    paths = {name: root / rel for name, rel in REPO_REL.items()}
    for name, path in paths.items():
        if not path.exists():
            fail(f"missing required file {name}: {path}")

    for name, expected in EXPECTED_DIGESTS.items():
        actual = digest(paths[name])
        if actual != expected:
            fail(f"{name} digest drift: {actual}")

    contract = parse_kv(paths["contract"])
    for key, value in EXPECTED_CONTRACT.items():
        if contract.get(key) != value:
            fail(f"contract field mismatch {key}: {contract.get(key)!r}")

    parent_summary = parse_kv(paths["parent_summary"])
    for key, value in EXPECTED_PARENT_SUMMARY.items():
        if parent_summary.get(key) != value:
            fail(f"parent summary field mismatch {key}: {parent_summary.get(key)!r}")

    _, parent_rows = parse_prefaced_tsv(paths["parent_results"], "ATTEMPT_INDEX\t")
    parent_by_attempt = {row["ATTEMPT_INDEX"]: row for row in parent_rows}
    if len(parent_rows) != 9:
        fail("parent V7-A.1 result matrix is not exactly nine rows")

    decision_meta, decisions = parse_prefaced_tsv(paths["parent_decisions"], "LIOUVILLE_CARRIER\t")
    if decision_meta:
        fail("unexpected decision metadata")
    if {row["LIOUVILLE_CARRIER"]: row["DECISION"] for row in decisions} != {
        "C0HOTripletonSet": "BASELINE_VALID",
        "C0HORect2Set": "MASKED_CHECKPOINT_PASS",
        "C0Rect2Set": "MASKED_CHECKPOINT_PASS",
    }:
        fail("parent carrier decisions drifted")

    ledger_meta, ledger_rows = parse_prefaced_tsv(paths["ledger"], "ROW_ID\t")
    if ledger_meta.get("SCHEMA") != contract["LEDGER_SCHEMA"]:
        fail("ledger schema mismatch")
    if ledger_meta.get("ROW_COUNT") != contract["LEDGER_ROW_COUNT"]:
        fail("ledger row-count metadata mismatch")
    if len(ledger_rows) != 24:
        fail("ledger is not exactly 24 rows")

    row_ids = [row["ROW_ID"] for row in ledger_rows]
    if row_ids != [f"{index:03d}" for index in range(1, 25)]:
        fail("ledger row IDs are not contiguous")

    satisfied = [row for row in ledger_rows if row["STATUS"] == SATISFIED]
    required = [row for row in ledger_rows if row["STATUS"] == REQUIRED]
    if len(satisfied) != 6 or len(required) != 18:
        fail("satisfied/required ledger counts drifted")

    seen: set[tuple[str, str, str]] = set()
    for row in ledger_rows:
        key = (row["PARENT_V7_ORDINAL"], row["CARRIER"], row["BOUNDARY"])
        if key in seen:
            fail(f"duplicate ledger key: {key}")
        seen.add(key)
        if row["PARENT_V7_ORDINAL"] not in EXPECTED_PARENT_ORDINALS:
            fail(f"unexpected parent ordinal: {row['PARENT_V7_ORDINAL']}")
        if row["CARRIER"] not in EXPECTED_CARRIERS:
            fail(f"unexpected bridge carrier: {row['CARRIER']}")
        if row["BOUNDARY"] not in EXPECTED_BOUNDARIES:
            fail(f"unexpected boundary: {row['BOUNDARY']}")
        if row["STATUS"] == SATISFIED:
            parent = parent_by_attempt.get(row["PARENT_ATTEMPT_INDEX"])
            if parent is None:
                fail(f"satisfied row lacks parent attempt: {row['ROW_ID']}")
            checks = {
                "PARENT_V7_ORDINAL": row["PARENT_V7_ORDINAL"],
                "CHECKPOINT_ROLE": row["CHECKPOINT_ROLE"],
                "NODE_ID": row["NODE_ID"],
                "LIOUVILLE_CARRIER": row["CARRIER"],
                "STATUS": "VERIFIED_CHECKPOINT",
                "ATTEMPT_BINDING": row["PARENT_ATTEMPT_BINDING"],
                "LIOUVILLE_DET": row["PARENT_DET_INTERVAL"],
                "MUTATION_TESTS": "46",
                "MUTATIONS_REJECTED": "46",
                "CHECKPOINT_PASS": "true",
            }
            for column, expected in checks.items():
                if parent.get(column) != expected:
                    fail(f"parent evidence mismatch row {row['ROW_ID']} column {column}")
            if row["BOUNDARY"] != "LIOUVILLE_CHECKPOINT":
                fail("only Liouville rows may be parent-satisfied")
        elif row["STATUS"] == REQUIRED:
            if row["BOUNDARY"] == "LIOUVILLE_CHECKPOINT":
                fail("Liouville checkpoint cannot be required-unrun in this bridge")
            if row["PARENT_ATTEMPT_INDEX"] != "-" or row["PARENT_ATTEMPT_BINDING"] != "-":
                fail(f"required row contains parent binding: {row['ROW_ID']}")
        else:
            fail(f"unexpected row status: {row['STATUS']}")

    for ordinal in EXPECTED_PARENT_ORDINALS:
        for carrier in EXPECTED_CARRIERS:
            actual = [row["BOUNDARY"] for row in ledger_rows if row["PARENT_V7_ORDINAL"] == ordinal and row["CARRIER"] == carrier]
            if tuple(actual) != EXPECTED_BOUNDARIES:
                fail(f"boundary order drift for ordinal={ordinal} carrier={carrier}")

    result_rows = [
        ("SCHEMA", "sounio.cs6.v7b-full-hpg-bridge-freeze-result.v1"),
        ("BRIDGE_LEDGER_VALID", "true"),
        ("PARENT_V7A1_EVIDENCE_BOUND", "true"),
        ("LEDGER_ROWS", "24"),
        ("SATISFIED_BY_V7A1", "6"),
        ("REQUIRED_UNRUN", "18"),
        ("CANDIDATE_CARRIERS", "2"),
        ("CELL_COUNT", "3"),
        ("BRIDGE_BOUNDARIES", "4"),
        ("PROSPECTIVE_EVALUATIONS_REQUIRED", "18"),
        ("C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED", "false"),
        ("FULL_HPG_PIPELINE_EVALUATED", "false"),
        ("V7_B_ELIGIBILITY", "false"),
        ("V7_B_WINNER", "NONE"),
        ("PROMOTION_ELIGIBLE", "false"),
        ("OPEN_PROBLEM_SOLVED", "false"),
        ("FPGA_EXECUTION", "false"),
    ]

    if args.receipt_dir:
        receipt = args.receipt_dir
        receipt.mkdir(parents=True, exist_ok=True)
        write_kv(receipt / "summary.txt", result_rows)
        (receipt / "ledger.sha256").write_text(f"{digest(paths['ledger'])}  {REPO_REL['ledger']}\n", encoding="ascii")
        (receipt / "contract.sha256").write_text(f"{digest(paths['contract'])}  {REPO_REL['contract']}\n", encoding="ascii")
        files = sorted((key, digest(path), str(path.relative_to(root))) for key, path in paths.items())
        (receipt / "files.sha256").write_text(
            "".join(f"{file_digest}  {rel}\n" for _, file_digest, rel in files),
            encoding="ascii",
        )
        (receipt / "blocker.txt").write_text(
            "\n".join(
                [
                    "Blocker-ID: BLK-20260802-cs6-v7b-full-hpg-bridge-execution",
                    "Status: classified",
                    "Severity: B3",
                    "Class: evidence-gap",
                    "Owner: codex-root",
                    "Lane: cs6-v7b-full-hpg-bridge-20260802",
                    f"Worktree: {root}",
                    "Branch: research/cs6-v7b-full-hpg-bridge-20260802",
                    "Files-Owned: docs/research/cs6_v7b_full_hpg_bridge_2026-08-02.md, scripts/ci/cs6_v7b_full_hpg_bridge_gate.sh, scripts/research/cs6_v7b_full_hpg_bridge_contract_v1.txt, scripts/research/cs6_v7b_full_hpg_bridge_ledger_v1.tsv, scripts/research/cs6_v7b_full_hpg_bridge_gate.py, scripts/research/receipts/cs6_v7b_full_hpg_bridge_freeze_v1/**",
                    "Files-Read-Only: V7-A.1 report, contract, result receipts, and retained archive sidecar",
                    "Do-Not-Touch: frozen V7-A and V7-A.1 contracts and result artifacts",
                    "Repro: bash scripts/ci/cs6_v7b_full_hpg_bridge_gate.sh",
                    "Observed: bridge ledger is frozen and parent Liouville evidence is bound; 18 downstream C1/C2/section-resident rows remain unrun",
                    "Expected: prospective Slurm bridge run supplies verifier-passed C1, C2, and section-resident evidence for all six candidate cell-carrier pairs",
                    "Acceptance-Gate: a future frozen execution matrix passes in-job and clean-checkout retained audits",
                    "Evidence-Level: E3",
                    "Evidence: scripts/research/receipts/cs6_v7b_full_hpg_bridge_freeze_v1/summary.txt",
                    "Fallback-Path: none",
                    "Legacy-Kept: yes",
                    "LLM-Offload: logged:.claude/llm_offload_log.md",
                    "Next-Action: implement and run the 18-row prospective Slurm bridge worker without changing frozen V7-A.1 evidence",
                    "",
                ]
            ),
            encoding="ascii",
        )

    for key, value in result_rows:
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
