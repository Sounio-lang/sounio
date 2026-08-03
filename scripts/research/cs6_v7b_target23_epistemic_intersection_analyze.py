#!/usr/bin/env python3
"""Audit exact determinant intersections in retained V7-B depth-5 receipts."""

from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


ATTEMPT_RE = re.compile(r"^\./attempts/([^/]+)/stdout\.txt$")
ATTEMPT_NAME_RE = re.compile(
    r"^A\d{4}_(P\d+_DU\d+_DS\d+)_(C0HORect2Set|C0Rect2Set)$"
)
INTERVAL_RE = re.compile(r"(?:^| )([A-Z0-9_]+)=\[([^,]+),([^\]]+)\]")
BOOL_RE = re.compile(r"(?:^| )([A-Z0-9_]+)=(true|false)(?= |$)")

LINE_KEYS = {
    "C1_P2_CONTROL": ("c1", "DET"),
    "C2_FULL_P2": ("c2", "HULL_DET"),
    "AFFINE_CARRIER": ("affine", "DET"),
    "HOMOGENEOUS_LOCAL_P2": ("resident", "RECON_DET"),
    "PLUCKER_COCYCLE": ("homogeneous", "DET"),
    "LIOUVILLE": ("liouville", "DET"),
}


@dataclass(frozen=True)
class Endpoint:
    value: Fraction
    token: str
    source: str


@dataclass(frozen=True)
class Interval:
    lower: Endpoint
    upper: Endpoint


def exact_hex(token: str) -> Fraction:
    value = float.fromhex(token)
    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def parse_interval(line: str, label: str, source: str) -> Interval:
    fields = {match.group(1): match.groups()[1:] for match in INTERVAL_RE.finditer(line)}
    if label not in fields:
        raise ValueError(f"{source}: missing {label}")
    lower_token, upper_token = fields[label]
    lower = Endpoint(exact_hex(lower_token), lower_token, source)
    upper = Endpoint(exact_hex(upper_token), upper_token, source)
    if lower.value > upper.value:
        raise ValueError(f"{source}: inverted interval {label}")
    return Interval(lower, upper)


def intersection(intervals: dict[str, Interval], names: tuple[str, ...]) -> Interval | None:
    lower = max((intervals[name].lower for name in names), key=lambda item: item.value)
    upper = min((intervals[name].upper for name in names), key=lambda item: item.value)
    return Interval(lower, upper) if lower.value <= upper.value else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_attempt(text: str, attempt: str) -> tuple[dict[str, Interval], dict[str, bool]]:
    intervals: dict[str, Interval] = {}
    flags: dict[str, bool] = {}
    for line in text.splitlines():
        head = line.split(" ", 1)[0]
        if head in LINE_KEYS:
            name, label = LINE_KEYS[head]
            if name in intervals:
                raise ValueError(f"{attempt}: duplicate {head}")
            intervals[name] = parse_interval(line, label, f"{attempt}:{name}")
        if head in {"SUMMARY", "LEAF_RESULT"}:
            flags.update(
                (match.group(1), match.group(2) == "true")
                for match in BOOL_RE.finditer(line)
            )
    expected = set(name for name, _ in LINE_KEYS.values())
    if set(intervals) != expected:
        raise ValueError(f"{attempt}: determinant set {sorted(intervals)} != {sorted(expected)}")
    return intervals, flags


def analyze(archive: Path, output_dir: Path) -> None:
    rows: list[str] = []
    counts = {
        "attempts": 0,
        "liouville_strict_negative": 0,
        "legacy_certificate_false": 0,
        "structural_pass": 0,
        "homogeneous_computation_valid": 0,
        "probe_pass": 0,
        "current_four_nonempty": 0,
        "contract_four_nonempty": 0,
        "all_six_nonempty": 0,
        "all_six_intersection_strict_negative": 0,
        "all_six_intersection_equals_liouville": 0,
    }
    lower_sources: dict[str, int] = {}
    upper_sources: dict[str, int] = {}
    zero_containment: dict[str, int] = {}
    carrier_counts: dict[str, int] = {}
    coordinate_carriers: dict[str, set[str]] = {}
    names_current = ("homogeneous", "resident", "affine", "liouville")
    names_contract = ("c1", "c2", "resident", "liouville")
    names_all = ("c1", "c2", "affine", "resident", "homogeneous", "liouville")

    with tarfile.open(archive, "r:gz") as bundle:
        members = sorted(
            (member for member in bundle.getmembers() if ATTEMPT_RE.match(member.name)),
            key=lambda member: ATTEMPT_RE.match(member.name).group(1),
        )
        for member in members:
            attempt = ATTEMPT_RE.match(member.name).group(1)
            attempt_match = ATTEMPT_NAME_RE.fullmatch(attempt)
            if attempt_match is None:
                raise ValueError(f"noncanonical attempt name: {attempt}")
            coordinate, carrier = attempt_match.groups()
            carrier_counts[carrier] = carrier_counts.get(carrier, 0) + 1
            coordinate_carriers.setdefault(coordinate, set()).add(carrier)
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read {member.name}")
            intervals, flags = parse_attempt(extracted.read().decode("utf-8"), attempt)
            current = intersection(intervals, names_current)
            contract = intersection(intervals, names_contract)
            all_six = intersection(intervals, names_all)
            counts["attempts"] += 1
            counts["liouville_strict_negative"] += intervals["liouville"].upper.value < 0
            counts["legacy_certificate_false"] += not flags.get("TERMINAL_CERTIFIED", True)
            counts["structural_pass"] += flags.get("STRUCTURAL_PASS", False)
            counts["homogeneous_computation_valid"] += flags.get(
                "HOMOGENEOUS_COMPUTATION_VALID", False
            )
            counts["probe_pass"] += flags.get("PROBE_PASS", False)
            counts["current_four_nonempty"] += current is not None
            counts["contract_four_nonempty"] += contract is not None
            counts["all_six_nonempty"] += all_six is not None
            strict_negative = all_six is not None and all_six.upper.value < 0
            counts["all_six_intersection_strict_negative"] += strict_negative
            counts["all_six_intersection_equals_liouville"] += (
                all_six is not None
                and all_six.lower.value == intervals["liouville"].lower.value
                and all_six.upper.value == intervals["liouville"].upper.value
            )
            for name, value in intervals.items():
                zero_containment[name] = zero_containment.get(name, 0) + (
                    value.lower.value <= 0 <= value.upper.value
                )
            if all_six is not None:
                lower_sources[all_six.lower.source.split(":")[-1]] = (
                    lower_sources.get(all_six.lower.source.split(":")[-1], 0) + 1
                )
                upper_sources[all_six.upper.source.split(":")[-1]] = (
                    upper_sources.get(all_six.upper.source.split(":")[-1], 0) + 1
                )
            rows.append(
                "\t".join(
                    (
                        attempt,
                        str(flags.get("TERMINAL_CERTIFIED", "missing")).lower(),
                        str(current is not None).lower(),
                        str(contract is not None).lower(),
                        str(all_six is not None).lower(),
                        str(strict_negative).lower(),
                        all_six.lower.token if all_six else "NA",
                        all_six.lower.source.split(":")[-1] if all_six else "NA",
                        all_six.upper.token if all_six else "NA",
                        all_six.upper.source.split(":")[-1] if all_six else "NA",
                    )
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "attempt\tlegacy_terminal_certified\tcurrent_four_nonempty\t"
        "contract_four_nonempty\tall_six_nonempty\tall_six_strict_negative\t"
        "intersection_lower\tlower_source\tintersection_upper\tupper_source\n"
    )
    (output_dir / "attempt_intersections.tsv").write_text(
        header + "\n".join(rows) + "\n", encoding="ascii"
    )
    summary_lines = [
        "CS6_V7B_EPISTEMIC_INTERSECTION_VERSION=1",
        f"INPUT_ARCHIVE_SHA256={sha256(archive)}",
        "ARITHMETIC=EXACT_BINARY64_ENDPOINTS_AS_RATIONALS",
        "CURRENT_FOUR=HOMOGENEOUS,RESIDENT_RECONSTRUCTED,AFFINE,LIOUVILLE",
        "CONTRACT_FOUR=C1,C2,RESIDENT_RECONSTRUCTED,LIOUVILLE",
        "ALL_SIX=C1,C2,AFFINE,RESIDENT_RECONSTRUCTED,HOMOGENEOUS,LIOUVILLE",
    ]
    summary_lines.extend(f"{key.upper()}={value}" for key, value in counts.items())
    summary_lines.append(f"COORDINATE_PAIRS={len(coordinate_carriers)}")
    summary_lines.extend(
        f"CARRIER_{key.upper()}={value}" for key, value in sorted(carrier_counts.items())
    )
    summary_lines.extend(
        f"ALL_SIX_LOWER_SOURCE_{key.upper()}={value}" for key, value in sorted(lower_sources.items())
    )
    summary_lines.extend(
        f"ALL_SIX_UPPER_SOURCE_{key.upper()}={value}" for key, value in sorted(upper_sources.items())
    )
    summary_lines.extend(
        f"CONTAINS_ZERO_{key.upper()}={value}" for key, value in sorted(zero_containment.items())
    )
    bounded_pass = (
        counts["attempts"] == 200
        and len(coordinate_carriers) == 100
        and all(
            carriers == {"C0HORect2Set", "C0Rect2Set"}
            for carriers in coordinate_carriers.values()
        )
        and carrier_counts == {"C0HORect2Set": 100, "C0Rect2Set": 100}
        and counts["liouville_strict_negative"] == 200
        and counts["legacy_certificate_false"] == 200
        and counts["structural_pass"] == 200
        and counts["homogeneous_computation_valid"] == 200
        and counts["probe_pass"] == 200
        and counts["current_four_nonempty"] == 200
        and counts["contract_four_nonempty"] == 200
        and counts["all_six_nonempty"] == 200
        and counts["all_six_intersection_strict_negative"] == 200
        and counts["all_six_intersection_equals_liouville"] == 200
    )
    summary_lines.extend(
        (
            f"BOUNDED_RECEIPT_INTERSECTION_PASS={str(bounded_pass).lower()}",
            "LEGACY_CERTIFICATE_RECLASSIFIED=false",
            "GLOBAL_HPG_CERTIFICATE=false",
            "V7_B_ELIGIBILITY=false",
            "PROMOTION_ELIGIBLE=false",
            "OPEN_PROBLEM_SOLVED=false",
            "NOVELTY_OR_PRIORITY_CLAIMED=false",
            "FPGA_EXECUTION=false",
        )
    )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="ascii")
    if not bounded_pass:
        raise SystemExit("bounded receipt intersection audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    analyze(args.archive, args.output_dir)


if __name__ == "__main__":
    main()
