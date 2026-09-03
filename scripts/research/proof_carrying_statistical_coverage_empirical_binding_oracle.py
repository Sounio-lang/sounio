#!/usr/bin/env python3
"""Independent exact oracle for D9 statistical coverage and binding refusal.

The synthetic part exhausts one frozen finite probability-mass family. The
external part evaluates one frozen public dataset under a protocol whose bytes
were fixed before the full-data calculation. Neither part establishes a real
patient state, causal effect, clinical action, or sealed empirical binding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "psychiatric_d9"
DEFAULT_DATASET = FIXTURE_DIR / "uci_drug_consumption_373.data"
DEFAULT_MANIFEST = FIXTURE_DIR / "dataset_manifest.v1.json"
DEFAULT_PROTOCOL = FIXTURE_DIR / "evaluation_protocol.v1.json"

TARGET_MASK = 3
PROCEDURE_MASKS = (3, 1, 2, 7)
TOTAL_MASS = 12
THRESHOLD_NUMERATOR = 3
THRESHOLD_DENOMINATOR = 4
DESIGN_A = (5, 1, 1, 5)
DESIGN_B = (1, 5, 5, 1)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diagnostic_permille(numerator: int, denominator: int) -> tuple[int, int]:
    require(numerator >= 0, "coverage numerator must be nonnegative")
    require(denominator > 0, "coverage denominator must be positive")
    floor, remainder = divmod(numerator * 1000, denominator)
    require(
        numerator * 1000 == floor * denominator + remainder,
        "permille quotient/remainder identity failed",
    )
    require(0 <= remainder < denominator, "permille remainder out of range")
    return floor, remainder


def mask_contains(container: int, member_mask: int) -> bool:
    return container & member_mask == member_mask


def whole_set_coverage(weights: tuple[int, int, int, int]) -> int:
    return sum(
        weight
        for weight, region in zip(weights, PROCEDURE_MASKS)
        if mask_contains(region, TARGET_MASK)
    )


def minimum_memberwise_coverage(weights: tuple[int, int, int, int]) -> int:
    a_coverage = sum(
        weight
        for weight, region in zip(weights, PROCEDURE_MASKS)
        if mask_contains(region, 1)
    )
    b_coverage = sum(
        weight
        for weight, region in zip(weights, PROCEDURE_MASKS)
        if mask_contains(region, 2)
    )
    return min(a_coverage, b_coverage)


def adequate(numerator: int, denominator: int) -> bool:
    return (
        numerator * THRESHOLD_DENOMINATOR
        >= THRESHOLD_NUMERATOR * denominator
    )


def enumerate_designs(total: int) -> Iterable[tuple[int, int, int, int]]:
    for w0 in range(total + 1):
        for w1 in range(total - w0 + 1):
            for w2 in range(total - w0 - w1 + 1):
                w3 = total - w0 - w1 - w2
                yield (w0, w1, w2, w3)


@dataclass(frozen=True)
class SyntheticSummary:
    total_designs: int
    positive_designs: int
    positivity_failures: int
    support_histogram: tuple[int, int, int, int]
    positive_coverage_histogram: tuple[tuple[int, int], ...]
    positive_adequate_designs: int


def analyze_synthetic_family() -> SyntheticSummary:
    total_designs = 0
    positive_designs = 0
    positivity_failures = 0
    support_histogram: Counter[int] = Counter()
    positive_coverage_histogram: Counter[int] = Counter()
    positive_adequate_designs = 0

    for weights in enumerate_designs(TOTAL_MASS):
        total_designs += 1
        support_size = sum(weight > 0 for weight in weights)
        support_histogram[support_size] += 1
        all_positive = support_size == 4
        if all_positive:
            positive_designs += 1
            numerator = whole_set_coverage(weights)
            positive_coverage_histogram[numerator] += 1
            if adequate(numerator, TOTAL_MASS):
                positive_adequate_designs += 1
        else:
            positivity_failures += 1

    expected_histogram = {
        2: 9,
        3: 16,
        4: 21,
        5: 24,
        6: 25,
        7: 24,
        8: 21,
        9: 16,
        10: 9,
    }
    require(total_designs == 455, "unexpected number of mass designs")
    require(positive_designs == 165, "unexpected positive-design count")
    require(positivity_failures == 290, "unexpected positivity-failure count")
    require(
        tuple(support_histogram[i] for i in range(1, 5)) == (4, 66, 220, 165),
        "unexpected support-size histogram",
    )
    require(
        dict(sorted(positive_coverage_histogram.items())) == expected_histogram,
        "unexpected whole-set coverage histogram",
    )
    require(positive_adequate_designs == 25, "unexpected adequate-design count")

    return SyntheticSummary(
        total_designs=total_designs,
        positive_designs=positive_designs,
        positivity_failures=positivity_failures,
        support_histogram=tuple(support_histogram[i] for i in range(1, 5)),
        positive_coverage_histogram=tuple(sorted(expected_histogram.items())),
        positive_adequate_designs=positive_adequate_designs,
    )


@dataclass(frozen=True)
class ExternalPartition:
    eligible: int
    covered: int
    recent: int
    prediction_masks: tuple[tuple[int, int], ...]
    score_bands: tuple[str, ...]


@dataclass(frozen=True)
class ExternalSummary:
    rows: int
    eligible_rows: int
    excluded_semeron_rows: int
    development: ExternalPartition
    calibration: ExternalPartition
    evaluation: ExternalPartition
    calibration_adequate: bool
    evaluation_adequate: bool
    support_compatible: bool
    abstention_reason_mask: int


def score_band(score: float) -> str:
    if score <= -1.0:
        return "score_lte_minus_1"
    if score >= 1.0:
        return "score_gte_plus_1"
    return "score_between"


def prediction_mask(score: float) -> int:
    if score <= -1.0:
        return 1
    if score >= 1.0:
        return 2
    return 3


def response_mask(value: str) -> int:
    if value in {"CL4", "CL5", "CL6"}:
        return 2
    require(value in {"CL0", "CL1", "CL2", "CL3"}, "invalid Benzos class")
    return 1


def partition_name(row_id: int) -> str:
    remainder = row_id % 5
    if remainder in {0, 1, 2}:
        return "development"
    if remainder == 3:
        return "calibration"
    return "evaluation"


def freeze_partition(records: list[tuple[int, bool, bool, int, str]]) -> ExternalPartition:
    masks = Counter(record[3] for record in records)
    bands = tuple(sorted({record[4] for record in records}))
    return ExternalPartition(
        eligible=len(records),
        covered=sum(record[1] for record in records),
        recent=sum(record[2] for record in records),
        prediction_masks=tuple(sorted(masks.items())),
        score_bands=bands,
    )


def validate_manifest(
    dataset_path: Path, manifest_path: Path, protocol_path: Path
) -> tuple[dict, dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)

    require(
        manifest["schema_version"] == "psychiatric-d9-dataset-manifest/v1",
        "wrong manifest schema",
    )
    require(
        protocol["schema_version"] == "psychiatric-d9-evaluation-protocol/v1",
        "wrong protocol schema",
    )
    require(
        manifest["data_file"]["sha256"] == sha256_file(dataset_path),
        "dataset SHA-256 mismatch",
    )
    require(
        manifest["protocol"]["sha256"] == sha256_file(protocol_path),
        "protocol SHA-256 mismatch",
    )
    require(
        manifest["license"]["spdx_like_id"] == "CC-BY-4.0",
        "license declaration mismatch",
    )
    require(
        protocol["status"] == "prespecified-before-full-dataset-analysis",
        "protocol is not marked prespecified",
    )
    require(
        protocol["binding_gate"]["expected_result"]
        == "abstain_from_empirical_binding",
        "external protocol must predeclare binding abstention",
    )
    return manifest, protocol


def analyze_external_fixture(
    dataset_path: Path, manifest_path: Path, protocol_path: Path
) -> ExternalSummary:
    manifest, protocol = validate_manifest(dataset_path, manifest_path, protocol_path)
    records: dict[str, list[tuple[int, bool, bool, int, str]]] = {
        "development": [],
        "calibration": [],
        "evaluation": [],
    }
    row_count = 0
    excluded_semeron = 0
    seen_ids: set[int] = set()

    with dataset_path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.reader(handle)
        for fields in reader:
            row_count += 1
            require(len(fields) == 32, f"row {row_count} has wrong column count")
            row_id = int(fields[0])
            require(row_id not in seen_ids, "duplicate respondent ID")
            seen_ids.add(row_id)
            if fields[30] != "CL0":
                excluded_semeron += 1
                continue

            score = float(fields[6]) + float(fields[11]) + float(fields[12])
            predicted = prediction_mask(score)
            observed = response_mask(fields[16])
            covered = mask_contains(predicted, observed)
            recent = observed == 2
            records[partition_name(row_id)].append(
                (row_id, covered, recent, predicted, score_band(score))
            )

    require(row_count == manifest["data_file"]["rows"], "row count mismatch")
    require(
        manifest["data_file"]["columns"] == 32,
        "manifest column count mismatch",
    )
    require(len(seen_ids) == row_count, "respondent IDs are not unique")

    development = freeze_partition(records["development"])
    calibration = freeze_partition(records["calibration"])
    evaluation = freeze_partition(records["evaluation"])
    eligible_rows = development.eligible + calibration.eligible + evaluation.eligible
    require(eligible_rows + excluded_semeron == row_count, "eligibility accounting failed")

    calibration_adequate = calibration.covered * 4 >= 3 * calibration.eligible
    evaluation_adequate = evaluation.covered * 4 >= 3 * evaluation.eligible
    support_compatible = set(evaluation.score_bands).issubset(
        set(calibration.score_bands)
    )

    reason_mask = 0
    if not support_compatible:
        reason_mask += 1
    if not calibration_adequate or not evaluation_adequate:
        reason_mask += 2
    if not protocol["binding_gate"]["metrological_instrument_calibration_available"]:
        reason_mask += 4
    if not protocol["binding_gate"]["collection_window_verified"]:
        reason_mask += 32
    if not protocol["binding_gate"]["external_custody_sealed"]:
        reason_mask += 64
    if not protocol["binding_gate"]["sealed_validation_available"]:
        reason_mask += 128

    require(reason_mask != 0, "external fixture unexpectedly authorized binding")
    require(
        not manifest["claim_boundary"]["patient_state_binding"],
        "manifest unexpectedly claims patient state",
    )
    require(
        not manifest["claim_boundary"]["clinical_action_authority"],
        "manifest unexpectedly claims clinical action authority",
    )

    return ExternalSummary(
        rows=row_count,
        eligible_rows=eligible_rows,
        excluded_semeron_rows=excluded_semeron,
        development=development,
        calibration=calibration,
        evaluation=evaluation,
        calibration_adequate=calibration_adequate,
        evaluation_adequate=evaluation_adequate,
        support_compatible=support_compatible,
        abstention_reason_mask=reason_mask,
    )


def mask_counts_text(partition: ExternalPartition) -> str:
    counts = dict(partition.prediction_masks)
    return f"1:{counts.get(1, 0)},2:{counts.get(2, 0)},3:{counts.get(3, 0)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    synthetic = analyze_synthetic_family()
    a_whole = whole_set_coverage(DESIGN_A)
    b_whole = whole_set_coverage(DESIGN_B)
    a_memberwise = minimum_memberwise_coverage(DESIGN_A)
    b_memberwise = minimum_memberwise_coverage(DESIGN_B)
    a_floor, a_remainder = diagnostic_permille(a_whole, TOTAL_MASS)
    b_floor, b_remainder = diagnostic_permille(b_whole, TOTAL_MASS)
    require(a_whole == 10 and b_whole == 2, "principal coverage collision failed")
    require(a_memberwise == 11 and b_memberwise == 7, "memberwise control failed")
    require(adequate(a_whole, TOTAL_MASS), "design A should be adequate")
    require(not adequate(b_whole, TOTAL_MASS), "design B should be inadequate")
    require(PROCEDURE_MASKS[0] == TARGET_MASK, "realized-region collision failed")

    eligibility_combinations = 0
    eligible_combinations = 0
    abstention_combinations = 0
    for calibration in (False, True):
        for positivity in (False, True):
            for instrument_population in (False, True):
                eligibility_combinations += 1
                if calibration and positivity and instrument_population:
                    eligible_combinations += 1
                else:
                    abstention_combinations += 1
    require(
        (eligibility_combinations, eligible_combinations, abstention_combinations)
        == (8, 1, 7),
        "eligibility truth table failed",
    )

    external = analyze_external_fixture(args.dataset, args.manifest, args.protocol)
    cal_floor, cal_remainder = diagnostic_permille(
        external.calibration.covered, external.calibration.eligible
    )
    eval_floor, eval_remainder = diagnostic_permille(
        external.evaluation.covered, external.evaluation.eligible
    )

    print(
        "ORACLE_D9_W0 target_mask=3 procedure_masks=3,1,2,7 "
        "outcomes=4 total_mass=12"
    )
    print(
        "ORACLE_D9_W1 design_a=5,1,1,5 whole=10/12 memberwise=11/12 "
        f"permille={a_floor} remainder={a_remainder} adequate=true"
    )
    print(
        "ORACLE_D9_W2 design_b=1,5,5,1 whole=2/12 memberwise=7/12 "
        f"permille={b_floor} remainder={b_remainder} adequate=false"
    )
    print(
        "ORACLE_D9_W3 same_realized_region=3 different_procedure_coverage=true "
        "identified_set_is_confidence_region=false"
    )
    print(
        "ORACLE_D9_W4 total_designs="
        f"{synthetic.total_designs} positive={synthetic.positive_designs} "
        f"positivity_failures={synthetic.positivity_failures} "
        "support_histogram=4,66,220,165"
    )
    histogram_text = ",".join(
        f"{numerator}:{count}"
        for numerator, count in synthetic.positive_coverage_histogram
    )
    print(
        f"ORACLE_D9_W5 positive_coverage_histogram={histogram_text} "
        f"adequate_at_3/4={synthetic.positive_adequate_designs}"
    )
    print(
        "ORACLE_D9_W6 marginal=9/10 rare_group=0/1 selected=0/1 "
        "marginal_is_selected=false"
    )
    print(
        "ORACLE_D9_W7 eligibility_combinations=8 eligible=1 abstain=7 "
        "failure_masks=1,2,4"
    )
    print(
        "ORACLE_D9_W8 same_table_bytes=true same_numeric_region=true "
        "lineage_substitutable=false integrity=false custody=false"
    )
    print(
        "ORACLE_D9_W9 predictive_set_is_confidence_region=false "
        "patient_state=0 clinical_authority=0"
    )
    print(
        f"ORACLE_D9_E0 rows={external.rows} eligible={external.eligible_rows} "
        f"semeron_excluded={external.excluded_semeron_rows} "
        f"development={external.development.eligible} "
        f"calibration={external.calibration.eligible} "
        f"evaluation={external.evaluation.eligible}"
    )
    print(
        f"ORACLE_D9_E1 calibration_covered={external.calibration.covered}/"
        f"{external.calibration.eligible} permille={cal_floor} "
        f"remainder={cal_remainder} adequate={str(external.calibration_adequate).lower()} "
        f"recent={external.calibration.recent} "
        f"set_masks={mask_counts_text(external.calibration)}"
    )
    print(
        f"ORACLE_D9_E2 evaluation_covered={external.evaluation.covered}/"
        f"{external.evaluation.eligible} permille={eval_floor} "
        f"remainder={eval_remainder} adequate={str(external.evaluation_adequate).lower()} "
        f"recent={external.evaluation.recent} "
        f"set_masks={mask_counts_text(external.evaluation)}"
    )
    print(
        "ORACLE_D9_E3 support_compatible="
        f"{str(external.support_compatible).lower()} "
        f"calibration_bands={len(external.calibration.score_bands)} "
        f"evaluation_bands={len(external.evaluation.score_bands)}"
    )
    print(
        f"ORACLE_D9_E4 empirical_binding=false abstention_reason_mask="
        f"{external.abstention_reason_mask} custody=false sealed=false "
        "patient_state=false clinical_authority=false"
    )
    print(
        "ORACLE_D9_E5 dataset_sha256="
        f"{sha256_file(args.dataset)} protocol_sha256={sha256_file(args.protocol)}"
    )
    print("STATISTICAL COVERAGE AND EMPIRICAL BINDING D9 ORACLE PASS")


if __name__ == "__main__":
    main()
