#!/usr/bin/env python3
"""Generate and replay a tiny SMT/Farkas QF_LIA certificate.

The fixture format is deliberately small and integer-scaled:

- instance CSV rows: row_id, coeff_x, bound for inequalities coeff_x * x <= bound
- certificate CSV rows: row_id, multiplier

The replay accepts a certificate when every multiplier is nonnegative, referenced
rows exist, all variable coefficients cancel to zero, and the combined bound is
strictly negative. This is a file-level Farkas smoke replay, not Alethe, LFSC,
Carcara, or a general SMT proof checker.
"""

from __future__ import annotations

import csv
import hashlib
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Row:
    row_id: str
    coeff_x: int
    bound: int


@dataclass
class ReplayResult:
    name: str
    instance_name: str
    expected_accept: int
    accepted: int
    combined_coeff_x: int
    combined_bound: int
    reason: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_instance(path: Path) -> dict[str, Row]:
    rows: dict[str, Row] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = Row(raw["row_id"], int(raw["coeff_x"]), int(raw["bound"]))
            if row.row_id in rows:
                raise ValueError(f"duplicate row_id {row.row_id}")
            rows[row.row_id] = row
    if not rows:
        raise ValueError("empty instance")
    return rows


def replay(instance_path: Path, cert_path: Path, expected_accept: int) -> ReplayResult:
    rows = read_instance(instance_path)
    combined_coeff = 0
    combined_bound = 0
    instance_name = instance_path.stem

    with cert_path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row_id = raw["row_id"]
            multiplier = int(raw["multiplier"])
            if multiplier < 0:
                return ReplayResult(cert_path.stem, instance_name, expected_accept, 0, combined_coeff, combined_bound, f"negative_multiplier_{row_id}")
            row = rows.get(row_id)
            if row is None:
                return ReplayResult(cert_path.stem, instance_name, expected_accept, 0, combined_coeff, combined_bound, f"missing_row_{row_id}")
            combined_coeff += multiplier * row.coeff_x
            combined_bound += multiplier * row.bound

    if combined_coeff != 0:
        return ReplayResult(cert_path.stem, instance_name, expected_accept, 0, combined_coeff, combined_bound, "nonzero_combined_coeff")
    if combined_bound >= 0:
        return ReplayResult(cert_path.stem, instance_name, expected_accept, 0, combined_coeff, combined_bound, "nonnegative_combined_bound")
    return ReplayResult(cert_path.stem, instance_name, expected_accept, 1, combined_coeff, combined_bound, "accepted")


def write_cert(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "multiplier"])
        writer.writeheader()
        for row_id, multiplier in rows:
            writer.writerow({"row_id": row_id, "multiplier": multiplier})


@dataclass(frozen=True)
class Fixture:
    name: str
    instance: Path
    cert: Path
    expected_accept: int


def write_instance(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "coeff_x", "bound"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"row_id": row.row_id, "coeff_x": row.coeff_x, "bound": row.bound})


def write_fixtures(out_dir: Path) -> list[Fixture]:
    fixtures = out_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)

    unsat_instance = fixtures / "qflia_x_le_0_and_x_ge_1.csv"
    write_instance(unsat_instance, [Row("r1", 1, 0), Row("r2", -1, -1)])

    tight_sat_instance = fixtures / "qflia_x_le_0_and_x_ge_0.csv"
    write_instance(tight_sat_instance, [Row("r1", 1, 0), Row("r2", -1, 0)])

    certs: dict[str, Path] = {}
    certs["farkas_x_le_0_x_ge_1"] = fixtures / "farkas_x_le_0_x_ge_1.csv"
    write_cert(certs["farkas_x_le_0_x_ge_1"], [("r1", 1), ("r2", 1)])

    certs["farkas_nonzero_coeff"] = fixtures / "farkas_nonzero_coeff.csv"
    write_cert(certs["farkas_nonzero_coeff"], [("r1", 1), ("r2", 0)])

    certs["farkas_nonnegative_bound"] = fixtures / "farkas_nonnegative_bound.csv"
    write_cert(certs["farkas_nonnegative_bound"], [("r1", 1), ("r2", 1)])

    certs["farkas_negative_multiplier"] = fixtures / "farkas_negative_multiplier.csv"
    write_cert(certs["farkas_negative_multiplier"], [("r1", 1), ("r2", -1)])

    certs["farkas_missing_row"] = fixtures / "farkas_missing_row.csv"
    write_cert(certs["farkas_missing_row"], [("r1", 1), ("r3", 1)])

    return [
        Fixture("farkas_x_le_0_x_ge_1", unsat_instance, certs["farkas_x_le_0_x_ge_1"], 1),
        Fixture("farkas_nonzero_coeff", unsat_instance, certs["farkas_nonzero_coeff"], 0),
        Fixture("farkas_nonnegative_bound", tight_sat_instance, certs["farkas_nonnegative_bound"], 0),
        Fixture("farkas_negative_multiplier", unsat_instance, certs["farkas_negative_multiplier"], 0),
        Fixture("farkas_missing_row", unsat_instance, certs["farkas_missing_row"], 0),
    ]


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(os.environ.get("SOLVER_SMT_FARKAS_REPLAY_OUT_DIR", f"/tmp/sounio-solver-smt-farkas-replay-tiny-{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    fixtures = write_fixtures(out_dir)
    results = [replay(fixture.instance, fixture.cert, fixture.expected_accept) for fixture in fixtures]

    result_path = out_dir / "replay_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "instance_name",
                "expected_accept",
                "accepted",
                "combined_coeff_x",
                "combined_bound",
                "reason",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    all_expected = all(result.expected_accept == result.accepted for result in results)
    positive = next(result for result in results if result.name == "farkas_x_le_0_x_ge_1")
    manifest_path = out_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("schema=sounio.solver.smt_farkas_replay_tiny.run.v1\n")
        f.write(f"timestamp_utc={stamp}\n")
        f.write(f"repo={repo}\n")
        f.write("format=integer_scaled_farkas_csv_subset\n")
        f.write(f"accepted_instance={fixtures[0].instance}\n")
        f.write(f"accepted_instance_sha256={sha256(fixtures[0].instance)}\n")
        f.write(f"accepted_certificate={fixtures[0].cert}\n")
        f.write(f"accepted_certificate_sha256={sha256(fixtures[0].cert)}\n")
        f.write(f"replay_results={result_path}\n")
        f.write(f"accepted_combined_coeff_x={positive.combined_coeff_x}\n")
        f.write(f"accepted_combined_bound={positive.combined_bound}\n")
        f.write(f"negative_controls={sum(1 for result in results if result.expected_accept == 0)}\n")
        f.write(f"all_expected={1 if all_expected else 0}\n")
        f.write("note=Tiny file-level Farkas replay over integer-scaled QF_LIA rows. Not Alethe, LFSC, Carcara, or a general SMT proof checker.\n")

    print(manifest_path.read_text(encoding="utf-8"), end="")
    return 0 if all_expected else 1


if __name__ == "__main__":
    sys.exit(main())
