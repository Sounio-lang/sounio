#!/usr/bin/env python3
"""Quantify the QR carrier experiment and certify the anchored C2 alternative."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tarfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-c2-anchored-local-hset-covering.v1"
LEAF_ID = "U08-0000000223_S09-0000000325"
ATTEMPT_MEMBER = "./attempts/A0199_P243_DU1_DS1_C0HORect2Set/stdout.txt"
CAPD_ARCHIVE_SHA256 = "fb4cf8732a8e153a480dc907eb87c38de52d9c2cd8fea95c61b2f3448947084c"
CAPD_STDOUT_SHA256 = "752e232191c20d11ddd4557865e846c3fd21f6a16168b1b255eff9810fd3a25e"
CAPD_WORKER_SOURCE_SHA256 = "1b0cee7fdd4df70487af3c9ec516471298c3ae9e5f8e291cee1e8d1adc6f97fa"
CAPD_WORKER_BINARY_SHA256 = "4968f5534df3472915357c49c4fadb6db3c409888703cfb67f49a67466a21647"
OLD_HSET_AGGREGATE_SHA256 = "5f6f52f406bb6fb6cedfc7f7c3169cbcd31cd856fb34874b2b5df374cb9ebe0e"
TRIPLETON_RECEIPT_SHA256 = "eece06dce709b7a5ce69bed619aadf81137f49ebb2f647d4c72ed9b74f9cbc2b"
TARGET_U_X = Fraction(-4644852547588741, 6250000000000000)
TARGET_U_Y = Fraction(13381910583555019, 20000000000000000)
DELTA0_RADIUS = Fraction(1, 256)
DELTA1_RADIUS = Fraction(1, 512)


def fail(message: str) -> None:
    raise SystemExit(f"covector QR/C2 analysis error: {message}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


@dataclass(frozen=True)
class Interval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            fail("reversed interval")

    @classmethod
    def point(cls, value: Fraction | int) -> "Interval":
        return cls(Fraction(value), Fraction(value))

    def __add__(self, other: "Interval | Fraction | int") -> "Interval":
        other = as_interval(other)
        return Interval(self.lower + other.lower, self.upper + other.upper)

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval(-self.upper, -self.lower)

    def __sub__(self, other: "Interval | Fraction | int") -> "Interval":
        return self + (-as_interval(other))

    def __rsub__(self, other: "Interval | Fraction | int") -> "Interval":
        return as_interval(other) - self

    def __mul__(self, other: "Interval | Fraction | int") -> "Interval":
        other = as_interval(other)
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return Interval(min(products), max(products))

    __rmul__ = __mul__

    def square(self) -> "Interval":
        lower = 0 if self.lower <= 0 <= self.upper else min(self.lower**2, self.upper**2)
        return Interval(lower, max(self.lower**2, self.upper**2))

    def max_abs(self) -> Fraction:
        return max(abs(self.lower), abs(self.upper))


def as_interval(value: Interval | Fraction | int) -> Interval:
    return value if isinstance(value, Interval) else Interval.point(value)


def parse_hex(value: str) -> Fraction:
    try:
        return Fraction(*float.fromhex(value).as_integer_ratio())
    except ValueError as error:
        fail(f"invalid hexadecimal endpoint {value!r}: {error}")


INTERVAL_PATTERN = re.compile(r"([A-Z0-9_]+)=\[([^,]+),([^\]]+)\]")


def parse_interval_line(line: str) -> dict[str, Interval]:
    result: dict[str, Interval] = {}
    for name, lower, upper in INTERVAL_PATTERN.findall(line):
        if name in result:
            fail(f"duplicate interval field {name}")
        result[name] = Interval(parse_hex(lower), parse_hex(upper))
    return result


def parse_fields(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not raw or "=" not in raw:
            fail(f"invalid field line {number} in {path}")
        key, value = raw.split("=", 1)
        if key in result:
            fail(f"duplicate field {key} in {path}")
        result[key] = value
    return result


def require(mapping: dict[str, str], key: str, expected: str) -> None:
    if mapping.get(key) != expected:
        fail(f"{key}: expected {expected!r}, got {mapping.get(key)!r}")


def emit_interval(prefix: str, value: Interval) -> None:
    print(f"{prefix}_LOWER_Q={value.lower}")
    print(f"{prefix}_UPPER_Q={value.upper}")


def read_capd_attempt(repo: Path) -> tuple[dict[str, Interval], dict[str, Interval], dict[str, Interval]]:
    receipt = repo / "scripts/research/receipts/cs6_v7b_target23_depth5_boundary_refine_v1"
    archive = receipt / "full-result.tar.gz"
    if sha256(archive) != CAPD_ARCHIVE_SHA256:
        fail("CAPD archive hash mismatch")
    with tarfile.open(archive, "r:gz") as bundle:
        matches = [member for member in bundle.getmembers() if member.name == ATTEMPT_MEMBER]
        if len(matches) != 1 or not matches[0].isfile():
            fail("CAPD attempt member is missing, duplicated, or not regular")
        handle = bundle.extractfile(matches[0])
        if handle is None:
            fail("CAPD attempt member cannot be read")
        stdout_bytes = handle.read()
    if sha256_bytes(stdout_bytes) != CAPD_STDOUT_SHA256:
        fail("CAPD stdout hash mismatch")
    stdout = stdout_bytes.decode("ascii")

    with (receipt / "results.tsv").open(encoding="ascii", newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row["ATTEMPT_INDEX"] == "199"]
    if len(rows) != 1:
        fail("results.tsv does not contain exactly one attempt 199")
    row = rows[0]
    expected_row = {
        "NODE_ID": LEAF_ID,
        "CARRIER": "C0HORect2Set",
        "STATUS": "DESCENDANT_PROBE_PASS",
        "WORKER_RC": "0",
        "STDOUT_SHA256": CAPD_STDOUT_SHA256,
        "C1_ORIENTATION_UNRESOLVED": "true",
        "C2_HULL_ORIENTATION_UNRESOLVED": "true",
        "EVENT1_CHARTS_CERTIFIED": "true",
        "EVENT2_CHARTS_CERTIFIED": "true",
        "HOMOGENEOUS_COMPUTATION_VALID": "true",
        "CERTIFICATE_PASS": "false",
        "PROBE_PASS": "true",
    }
    for key, expected in expected_row.items():
        if row.get(key) != expected:
            fail(f"attempt 199 {key} mismatch")

    scalar: dict[str, str] = {}
    selected: dict[str, dict[str, Interval]] = {}
    summary_tokens: dict[str, str] = {}
    binding_tokens: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line and " " not in line:
            key, value = line.split("=", 1)
            scalar[key] = value
        for prefix in ("SOURCE_TILE", "C1_P2_CONTROL", "C2_FULL_P2", "C2_CENTER_P2"):
            if line.startswith(prefix + " "):
                if prefix in selected:
                    fail(f"duplicate {prefix} line")
                selected[prefix] = parse_interval_line(line)
        if line.startswith("SUMMARY "):
            summary_tokens = dict(token.split("=", 1) for token in line.split()[1:])
        if line.startswith("V7_BINDING "):
            binding_tokens = {
                token.split("=", 1)[0]: token.split("=", 1)[1]
                for token in line.split()[1:]
                if "=" in token and not token.startswith("INITIAL")
            }
    required_scalars = {
        "SCHEMA": "sounio.cs6.plucker-cocycle-leaf.v1",
        "WORKER_SOURCE_SHA256": CAPD_WORKER_SOURCE_SHA256,
        "SOURCE": "N0",
        "U_DEPTH": "8",
        "U_INDEX": "223",
        "S_DEPTH": "9",
        "S_INDEX": "325",
        "ORDER": "8",
        "RETURN_COUNT": "2",
        "SECTION": "COORDINATE_W_EQUALS_ZERO",
        "CROSSING_DIRECTION": "MINUS_PLUS",
        "CAPD_SOURCE_TREE_DECLARED": "capd-5.3.0",
        "INTERVAL_BACKEND_DECLARED": "FILIB",
        "DIAGONAL_TAYLOR_TO_DERIVATIVE_FACTOR": "2",
        "OFFDIAGONAL_TAYLOR_TO_DERIVATIVE_FACTOR": "1",
    }
    for key, expected in required_scalars.items():
        require(scalar, key, expected)
    if set(selected) != {"SOURCE_TILE", "C1_P2_CONTROL", "C2_FULL_P2", "C2_CENTER_P2"}:
        fail("missing CAPD C1/C2 line")
    for scalar_key, row_key in (
        ("INPUT_SHA256", "INPUT_SHA256"),
        ("RUN_CHALLENGE", "RUN_CHALLENGE"),
    ):
        if scalar.get(scalar_key) != row.get(row_key):
            fail(f"CAPD stdout/results binding mismatch for {scalar_key}")
    if binding_tokens.get("ATTEMPT_BINDING") != row.get("ATTEMPT_BINDING"):
        fail("CAPD attempt binding mismatch")
    source_tile = selected["SOURCE_TILE"]
    if source_tile["DELTA0"].lower > -DELTA0_RADIUS or source_tile["DELTA0"].upper < DELTA0_RADIUS:
        fail("CAPD DELTA0 does not enclose the exact xi scale")
    if source_tile["DELTA1"].lower > -DELTA1_RADIUS or source_tile["DELTA1"].upper < DELTA1_RADIUS:
        fail("CAPD DELTA1 does not enclose the exact eta scale")
    for key in (
        "ALL_FINITE",
        "C1_C2_DP_OVERLAP",
        "EVENT_ORDER_CERTIFIED",
        "EVENT_TRANSVERSALITY_CERTIFIED",
        "IMPACT_TIME_CROSSCHECK",
        "LIOUVILLE_ORIENTATION_CERTIFIED",
        "STRUCTURAL_PASS",
        "PROBE_PASS",
    ):
        require(summary_tokens, key, "true")
    require(summary_tokens, "CERTIFICATE_PASS", "false")

    source = repo / "scripts/research/cs6_hapg_liouville_carrier_ablation_probe.cpp"
    if sha256(source) != CAPD_WORKER_SOURCE_SHA256:
        fail("retained CAPD worker source hash mismatch")
    worker_binary_line = (receipt / "provenance/worker.sha256").read_text(encoding="ascii").strip()
    if worker_binary_line != f"{CAPD_WORKER_BINARY_SHA256}  worker-binary":
        fail("CAPD worker binary provenance mismatch")
    run_summary = parse_fields(receipt / "summary.txt")
    require(run_summary, "RUN_COMPLETE", "true")
    require(run_summary, "RUN_VALID", "true")
    require(run_summary, "REFINED_REGION_PROBE_COVER_PASS", "true")
    return selected["C1_P2_CONTROL"], selected["C2_FULL_P2"], selected["C2_CENTER_P2"]


def rational_json_interval(value: object) -> Interval:
    if not isinstance(value, list) or len(value) != 2:
        fail("invalid JSON interval")
    return Interval(Fraction(value[0]), Fraction(value[1]))


def directional_generator_radius(path: Path) -> tuple[Fraction, Fraction, Fraction]:
    raw = json.loads(path.read_text(encoding="ascii"))
    carriers = raw.get("carriers")
    if not isinstance(carriers, list) or len(carriers) != 1:
        fail(f"{path.name} must contain exactly one carrier")
    components = carriers[0].get("components")
    if not isinstance(components, list) or len(components) != 4:
        fail(f"{path.name} carrier must contain four components")
    x_entries = {tuple(item["monomial"]): rational_json_interval(item["interval"]) for item in components[0]["coefficients"]}
    y_entries = {tuple(item["monomial"]): rational_json_interval(item["interval"]) for item in components[1]["coefficients"]}
    eta_radius = Fraction(0)
    residual_radius = Fraction(0)
    for monomial in set(x_entries) | set(y_entries):
        coefficient = TARGET_U_X * x_entries.get(monomial, Interval.point(0)) + TARGET_U_Y * y_entries.get(monomial, Interval.point(0))
        if monomial[1] and not any(monomial[2:]):
            eta_radius += coefficient.max_abs()
        if any(monomial[2:]):
            residual_radius += coefficient.max_abs()
    residual_radius += (TARGET_U_X * rational_json_interval(components[0]["remainder"])).max_abs()
    residual_radius += (TARGET_U_Y * rational_json_interval(components[1]["remainder"])).max_abs()
    return eta_radius, residual_radius, eta_radius + residual_radius


def main() -> None:
    script = Path(__file__).resolve()
    repo = script.parents[2]
    receipts = repo / "scripts/research/receipts"
    qr_receipt_dir = receipts / "cs6_v7b_target23_arb_tm2r_covector_qr_v1"
    old_hset_dir = receipts / "cs6_v7b_target23_arb_tm2r_hset_covering_v1"
    old_aggregate_path = old_hset_dir / "aggregate.txt"
    tripleton_path = qr_receipt_dir / "face_LEFT_XLEH_ROOT.json"
    baseline_face_path = old_hset_dir / "face_LEFT_XLEH_ROOT.json"
    if sha256(old_aggregate_path) != OLD_HSET_AGGREGATE_SHA256:
        fail("old h-set aggregate hash mismatch")
    if sha256(tripleton_path) != TRIPLETON_RECEIPT_SHA256:
        fail("tripleton receipt hash mismatch")
    old = parse_fields(old_aggregate_path)
    tripleton = json.loads(tripleton_path.read_text(encoding="ascii"))
    if tripleton.get("schema") != "sounio.cs6.v7b-target23-arb-tm2r-covector-qr-exit-face.v1":
        fail("wrong tripleton schema")
    if tripleton.get("carrier_kind") != "DYNAMIC_TRIPLETON":
        fail("wrong tripleton carrier kind")
    if tripleton.get("selected_source_face_chain_certificate") is not True:
        fail("tripleton lacks the selected face chain certificate")
    if tripleton.get("point_fallback_used") is not False:
        fail("tripleton used point fallback")
    if tripleton.get("worker_source_sha256") != sha256(repo / "scripts/research/cs6_v7b_target23_arb_tm2r_covector_qr_face_worker.py"):
        fail("tripleton worker source hash mismatch")

    c1, c2_full, c2_center = read_capd_attempt(repo)
    c1_direct = TARGET_U_X * c1["DP00"] + TARGET_U_Y * c1["DP10"]
    center_g0 = TARGET_U_X * c2_center["DP00"] + TARGET_U_Y * c2_center["DP10"]
    center_g1 = TARGET_U_X * c2_center["DP01"] + TARGET_U_Y * c2_center["DP11"]
    h00 = 2 * (TARGET_U_X * c2_full["D2P000"] + TARGET_U_Y * c2_full["D2P100"])
    h01 = TARGET_U_X * c2_full["D2P001"] + TARGET_U_Y * c2_full["D2P101"]
    h11 = 2 * (TARGET_U_X * c2_full["D2P011"] + TARGET_U_Y * c2_full["D2P111"])
    delta0 = Interval(-DELTA0_RADIUS, DELTA0_RADIUS)
    delta1 = Interval(-DELTA1_RADIUS, DELTA1_RADIUS)
    derivative_global = center_g0 + h00 * delta0 + h01 * delta1
    derivative_xi = Interval(derivative_global.lower / 256, derivative_global.upper / 256)
    if c1_direct.lower >= 0 or c1_direct.upper <= 0:
        fail("direct C1 derivative unexpectedly excludes zero")
    if derivative_xi.lower <= 0:
        fail("C2 mean-value derivative is not strictly positive")

    center_u = TARGET_U_X * c2_center["X0"] + TARGET_U_Y * c2_center["X1"]

    def face_value(sign: int) -> Interval:
        d0 = Interval.point(Fraction(sign, 256))
        return (
            center_u
            + center_g0 * d0
            + center_g1 * delta1
            + Fraction(1, 2) * h00 * d0.square()
            + h01 * d0 * delta1
            + Fraction(1, 2) * h11 * delta1.square()
        )

    left_face = face_value(-1)
    right_face_direct = face_value(1)
    right_lower_anchored = left_face.lower + 2 * derivative_xi.lower
    anchored_gap = right_lower_anchored - left_face.upper
    if anchored_gap <= 0 or right_face_direct.lower < right_lower_anchored:
        fail("anchored monotonicity does not separate the exit faces")
    target_u_center = (left_face.upper + right_lower_anchored) / 2
    target_u_radius = anchored_gap / 4
    if target_u_radius <= 0:
        fail("constructed target unstable radius is non-positive")

    stable_image = Interval(Fraction(old["SUPPORT_STABLE_IMAGE_LOWER_Q"]), Fraction(old["SUPPORT_STABLE_IMAGE_UPPER_Q"]))
    entry_margin = min(stable_image.lower + 1, 1 - stable_image.upper)
    if entry_margin <= 0:
        fail("retained stable support meets the entry boundary")
    left_normalized = Interval(
        (left_face.lower - target_u_center) / target_u_radius,
        (left_face.upper - target_u_center) / target_u_radius,
    )
    right_normalized_lower = (right_lower_anchored - target_u_center) / target_u_radius
    exit_margin = min(-1 - left_normalized.upper, right_normalized_lower - 1)
    if exit_margin <= 0:
        fail("constructed target chart does not separate exit faces")

    target_s_radius = Fraction(old["TARGET_S_RADIUS_Q"])
    target_linear_det = Interval(
        Fraction(old["TARGET_LINEAR_DETERMINANT_LOWER_Q"]),
        Fraction(old["TARGET_LINEAR_DETERMINANT_UPPER_Q"]),
    )
    target_chart_det = target_linear_det * (target_u_radius * target_s_radius)
    physical_det = Interval(
        Fraction(old["PHYSICAL_RETURN_DETERMINANT_LOWER_Q"]),
        Fraction(old["PHYSICAL_RETURN_DETERMINANT_UPPER_Q"]),
    )
    normalized_det = physical_det * target_linear_det * (
        1 / (target_u_radius * target_s_radius)
    )
    if target_chart_det.lower <= 0 or normalized_det.upper >= 0:
        fail("target chart or normalized return determinant has wrong sign")

    baseline_eta, baseline_rho, baseline_total = directional_generator_radius(baseline_face_path)
    triple_eta, triple_rho, triple_total = directional_generator_radius(tripleton_path)
    radius_ratio = triple_total / baseline_total
    improvement = baseline_total / triple_total
    if radius_ratio <= 1 or triple_total <= baseline_total / 18:
        fail("tripleton unexpectedly satisfies the compression target")

    print(f"SCHEMA={SCHEMA}")
    print(f"ANALYZER_SOURCE_SHA256={sha256(script)}")
    print(f"LEAF_ID={LEAF_ID}")
    print("MAP=P2")
    print("SECTION=w=0")
    print("SOURCE_COORDINATES=FROZEN_UPO_FRAME_NORMALIZED_XI_ETA")
    print("TARGET_UNSTABLE_SELECTION=FROZEN_COVECTOR")
    print(f"TARGET_UNSTABLE_ROW_X_Q={TARGET_U_X}")
    print(f"TARGET_UNSTABLE_ROW_Y_Q={TARGET_U_Y}")
    print(f"CAPD_ARCHIVE_SHA256={CAPD_ARCHIVE_SHA256}")
    print(f"CAPD_ATTEMPT_MEMBER={ATTEMPT_MEMBER}")
    print(f"CAPD_ATTEMPT_STDOUT_SHA256={CAPD_STDOUT_SHA256}")
    print(f"CAPD_WORKER_SOURCE_SHA256={CAPD_WORKER_SOURCE_SHA256}")
    print(f"CAPD_WORKER_BINARY_SHA256={CAPD_WORKER_BINARY_SHA256}")
    print(f"OLD_HSET_AGGREGATE_SHA256={OLD_HSET_AGGREGATE_SHA256}")
    print(f"TRIPLETON_RECEIPT_SHA256={TRIPLETON_RECEIPT_SHA256}")
    print(f"DOUBLETON_STDERR_SHA256={sha256(qr_receipt_dir / 'doubleton_LEFT_XLEH_ROOT.stderr.txt')}")
    print(f"DOUBLETON_INCOMPLETE_SHA256={sha256(qr_receipt_dir / 'doubleton_LEFT_XLEH_ROOT.incomplete.json')}")
    print("CAPD_VERSION=5.3.0")
    print("CAPD_INTERVAL_BACKEND=FILIB")
    print("CAPD_RETURN_COUNT=2")
    print("CAPD_HESSIAN_DIAGONAL_FACTOR=2")
    print("CAPD_HESSIAN_OFFDIAGONAL_FACTOR=1")
    print(f"CAPD_DELTA0_RADIUS_Q={DELTA0_RADIUS}")
    print(f"CAPD_DELTA1_RADIUS_Q={DELTA1_RADIUS}")
    emit_interval("C1_DIRECT_DU_DGLOBAL0", c1_direct)
    emit_interval("C2_CENTER_DU_DGLOBAL0", center_g0)
    emit_interval("C2_ACTUAL_HESSIAN_00", h00)
    emit_interval("C2_ACTUAL_HESSIAN_01", h01)
    emit_interval("C2_ACTUAL_HESSIAN_11", h11)
    emit_interval("C2_MEAN_VALUE_DU_DGLOBAL0", derivative_global)
    emit_interval("C2_MEAN_VALUE_DU_DXI", derivative_xi)
    emit_interval("ANCHORED_LEFT_FACE_U_RAW", left_face)
    emit_interval("DIRECT_RIGHT_FACE_U_RAW", right_face_direct)
    print(f"ANCHORED_RIGHT_FACE_LOWER_Q={right_lower_anchored}")
    print(f"ANCHORED_EXIT_FACE_GAP_Q={anchored_gap}")
    print(f"TARGET_U_CENTER_Q={target_u_center}")
    print(f"TARGET_U_RADIUS_Q={target_u_radius}")
    print(f"TARGET_S_CENTER_Q={old['TARGET_S_CENTER_Q']}")
    print(f"TARGET_S_RADIUS_Q={target_s_radius}")
    emit_interval("NORMALIZED_LEFT_EXIT_IMAGE", left_normalized)
    print(f"NORMALIZED_RIGHT_EXIT_LOWER_Q={right_normalized_lower}")
    emit_interval("SUPPORT_STABLE_IMAGE", stable_image)
    print(f"ENTRY_MARGIN_Q={entry_margin}")
    print(f"EXIT_MARGIN_Q={exit_margin}")
    emit_interval("PHYSICAL_RETURN_DETERMINANT", physical_det)
    emit_interval("TARGET_LINEAR_DETERMINANT", target_linear_det)
    emit_interval("TARGET_CHART_DETERMINANT", target_chart_det)
    emit_interval("NORMALIZED_RETURN_DETERMINANT", normalized_det)
    print(f"BASELINE_DIRECTIONAL_ETA_RADIUS_Q={baseline_eta}")
    print(f"BASELINE_DIRECTIONAL_RESIDUAL_RADIUS_Q={baseline_rho}")
    print(f"BASELINE_DIRECTIONAL_TOTAL_RADIUS_Q={baseline_total}")
    print(f"TRIPLETON_DIRECTIONAL_ETA_RADIUS_Q={triple_eta}")
    print(f"TRIPLETON_DIRECTIONAL_RESIDUAL_RADIUS_Q={triple_rho}")
    print(f"TRIPLETON_DIRECTIONAL_TOTAL_RADIUS_Q={triple_total}")
    print(f"TRIPLETON_TO_BASELINE_RADIUS_RATIO_Q={radius_ratio}")
    print(f"TRIPLETON_IMPROVEMENT_FACTOR_Q={improvement}")
    print("QR_TARGET_IMPROVEMENT_FACTOR=18")
    print("FIXED_COVECTOR_QR_CERTIFICATE=false")
    print("FIXED_COVECTOR_QR_STATUS=NO_COMPLETED_RECEIPT")
    print("DYNAMIC_DOUBLETON_SECOND_SECTION_CERTIFICATE=false")
    print("DYNAMIC_DOUBLETON_STATUS=INTERRUPTED_AFTER_FIRST_EVENT")
    print("DYNAMIC_TRIPLETON_SECOND_SECTION_CERTIFICATE=true")
    print("DYNAMIC_TRIPLETON_DIRECTIONAL_IMPROVEMENT_CERTIFICATE=false")
    print("DYNAMIC_TRIPLETON_DIRECTIONAL_RADIUS_RATIO_GT_ONE=true")
    print("C1_DIRECT_MONOTONICITY_CERTIFICATE=false")
    print("C2_MEAN_VALUE_MONOTONICITY_CERTIFICATE=true")
    print("ANCHORED_EXIT_FACE_INEQUALITIES_CERTIFICATE=true")
    print("HSET_COORDINATES_CERTIFICATE=true")
    print("ENTRY_BOUNDARY_AVOIDANCE_CERTIFICATE=true")
    print("COVERING_DEGREE=1")
    print("COVERING_DEGREE_CERTIFICATE=true")
    print("RETURN_MAP_DETERMINANT_CERTIFICATE=true")
    print("LOCAL_HSET_COVERING_RELATION_CERTIFICATE=true")
    for key in (
        "RECURRENT_COVERING_GRAPH_CERTIFICATE",
        "FIBONACCI_COVERING_CERTIFICATE",
        "GLOBAL_HPG_CERTIFICATE",
        "CHAOS_PROVED",
        "CHAOTIC_ATTRACTOR_PROVED",
        "OPEN_PROBLEM_SOLVED",
        "NOVELTY_OR_PRIORITY_CLAIMED",
        "POINT_FALLBACK_USED",
    ):
        print(f"{key}=false")
    print("CAPD_USED=true")


if __name__ == "__main__":
    main()
