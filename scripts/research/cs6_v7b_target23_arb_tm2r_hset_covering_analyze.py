#!/usr/bin/env python3
"""Construct and certify a local h-set covering from retained second-section carriers."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from flint import arb

import cs6_v7b_target23_arb_tm2r_first_return_worker as base


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-local-hset-covering.v1"
CARRIER_SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-hset-carrier.v1"
FACE_SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-hset-exit-face.v1"
EXPECTED_TILES = ("XLEL", "XLEH", "XHEL", "XHEH")
EXPECTED_FACE_TILES = {
    "LEFT": {"XLEL", "XLEH"},
    "RIGHT": {"XHEL", "XHEH"},
}
TARGET_UNSTABLE_X_Q = Fraction("-0.74317640761419856")
TARGET_UNSTABLE_Y_Q = Fraction("0.66909552917775095")


@dataclass
class Carrier:
    carrier_id: str
    tile_id: str
    xi_low: Fraction
    xi_high: Fraction
    eta_low: Fraction
    eta_high: Fraction
    components: list[base.TM2R]


def fail(message: str) -> None:
    raise SystemExit(f"h-set covering analysis error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ball(value: Fraction) -> arb:
    return base.rational_ball(value)


def interval(pair: object) -> arb:
    if not isinstance(pair, list) or len(pair) != 2:
        fail("invalid serialized interval")
    lower, upper = Fraction(pair[0]), Fraction(pair[1])
    if lower > upper:
        fail("reversed serialized interval")
    result = arb(ball((lower + upper) / 2), ball((upper - lower) / 2))
    if result.lower() > ball(lower) or result.upper() < ball(upper):
        fail("outward interval reconstruction failed")
    return result


def tm2r(raw: object) -> base.TM2R:
    if not isinstance(raw, dict):
        fail("invalid TM2R object")
    coefficients: dict[tuple[int, ...], arb] = {}
    entries = raw.get("coefficients")
    if not isinstance(entries, list):
        fail("missing TM2R coefficients")
    for entry in entries:
        if not isinstance(entry, dict):
            fail("invalid coefficient entry")
        monomial = tuple(int(value) for value in entry["monomial"])
        if len(monomial) != base.VARIABLES or any(value < 0 for value in monomial):
            fail("invalid monomial")
        if monomial in coefficients:
            fail("duplicate monomial")
        coefficients[monomial] = interval(entry["interval"])
    return base.TM2R(coefficients, interval(raw.get("remainder")))


def variable_interval(lower: Fraction, upper: Fraction) -> arb:
    return arb(ball((lower + upper) / 2), ball((upper - lower) / 2))


def range_on_box(
    model: base.TM2R,
    source_values: dict[int, Fraction] | None = None,
) -> arb:
    source_values = source_values or {}
    variables = [base.UNIT for _ in range(base.VARIABLES)]
    for variable, value in source_values.items():
        variables[variable] = ball(value)
    result = arb(0)
    for monomial, coefficient in model.coefficients.items():
        term = coefficient
        for variable, exponent in enumerate(monomial):
            for _ in range(exponent):
                term *= variables[variable]
        result += term
    return result + model.remainder


def hull(values: list[arb]) -> arb:
    if not values:
        fail("empty hull")
    result = values[0]
    for value in values[1:]:
        result = result.union(value)
    return result


def exact_pair(value: arb) -> tuple[Fraction, Fraction]:
    return Fraction(base.lower_fraction(value)), Fraction(base.upper_fraction(value))


def emit_interval(prefix: str, value: arb) -> None:
    print(f"{prefix}_LOWER_Q={base.lower_fraction(value)}")
    print(f"{prefix}_UPPER_Q={base.upper_fraction(value)}")


def local_models(carrier: Carrier) -> tuple[base.TM2R, base.TM2R]:
    x, y = carrier.components[0], carrier.components[1]
    determinant = base.UNSTABLE_X * base.STABLE_Y - base.STABLE_X * base.UNSTABLE_Y
    u = (
        base.STABLE_Y * (x - base.ORIGIN_X)
        - base.STABLE_X * (y - base.ORIGIN_Y)
    ) / determinant
    s = (
        -base.UNSTABLE_Y * (x - base.ORIGIN_X)
        + base.UNSTABLE_X * (y - base.ORIGIN_Y)
    ) / determinant
    return u, s


def candidate_target_models(carrier: Carrier) -> tuple[base.TM2R, base.TM2R]:
    x, y = carrier.components[0], carrier.components[1]
    nx = ball(TARGET_UNSTABLE_X_Q)
    ny = ball(TARGET_UNSTABLE_Y_Q)
    return nx * x + ny * y, -ny * x + nx * y


def face_value(carrier: Carrier, face: Fraction) -> Fraction | None:
    if face < carrier.xi_low or face > carrier.xi_high:
        return None
    center = (carrier.xi_low + carrier.xi_high) / 2
    radius = (carrier.xi_high - carrier.xi_low) / 2
    if radius <= 0:
        fail("degenerate carrier source domain")
    local = (face - center) / radius
    if local < -1 or local > 1:
        fail("global face mapped outside local source coordinate")
    return local


def load_carriers(paths: list[Path]) -> tuple[list[Carrier], dict[str, str], str]:
    if len(paths) != len(EXPECTED_TILES):
        fail("exactly four tile receipts are required")
    carriers: list[Carrier] = []
    receipt_hashes: dict[str, str] = {}
    worker_hash = ""
    seen_tiles: set[str] = set()
    for path in paths:
        raw = json.loads(path.read_text(encoding="ascii"))
        if raw.get("schema") != CARRIER_SCHEMA:
            fail(f"wrong carrier schema in {path}")
        tile_id = raw.get("tile_id")
        if tile_id not in EXPECTED_TILES or tile_id in seen_tiles:
            fail("missing, duplicate, or unknown source tile")
        seen_tiles.add(tile_id)
        if raw.get("leaf_id") != base.LEAF_ID:
            fail("wrong leaf id")
        if raw.get("selected_source_chain_certificate") is not True:
            fail("source tile lacks the second-return chain certificate")
        if raw.get("point_fallback_used") is not False:
            fail("point fallback is forbidden")
        current_worker_hash = str(raw.get("worker_source_sha256"))
        if worker_hash and current_worker_hash != worker_hash:
            fail("carrier worker hash mismatch across tiles")
        worker_hash = current_worker_hash
        receipt_hashes[tile_id] = sha256(path)
        raw_carriers = raw.get("carriers")
        if not isinstance(raw_carriers, list) or not raw_carriers:
            fail("tile contains no final carrier")
        for item in raw_carriers:
            domain = item["source_domain"]
            xi_low, xi_high = map(Fraction, domain["xi"])
            eta_low, eta_high = map(Fraction, domain["eta"])
            components = [tm2r(component) for component in item["components"]]
            if len(components) != 4:
                fail("carrier does not have four TM2R components")
            if exact_pair(range_on_box(components[2])) != (Fraction(0), Fraction(0)):
                fail("final carrier is not exactly on w=0")
            normal = interval(item["event_normal"])
            if normal.lower() <= 0:
                fail("final carrier lacks positive transversality")
            carriers.append(
                Carrier(
                    str(item["carrier_id"]),
                    tile_id,
                    xi_low,
                    xi_high,
                    eta_low,
                    eta_high,
                    components,
                )
            )
    if seen_tiles != set(EXPECTED_TILES):
        fail("source tile cover is incomplete")
    return carriers, receipt_hashes, worker_hash


def load_face_carriers(
    paths: list[Path], support_worker_hash: str
) -> tuple[dict[str, list[Carrier]], dict[str, str], str]:
    if len(paths) < 4:
        fail("at least four exit-face receipts are required")
    result = {"LEFT": [], "RIGHT": []}
    receipt_hashes: dict[str, str] = {}
    face_worker_hash = ""
    seen = {"LEFT": set(), "RIGHT": set()}
    cover_intervals = {"LEFT": [], "RIGHT": []}
    for path in paths:
        raw = json.loads(path.read_text(encoding="ascii"))
        if raw.get("schema") != FACE_SCHEMA:
            fail(f"wrong exit-face schema in {path}")
        face = raw.get("source_face")
        tile_id = raw.get("tile_id")
        refinement = raw.get("eta_refinement")
        if face not in EXPECTED_FACE_TILES or tile_id not in EXPECTED_FACE_TILES[face]:
            fail("unknown face/tile pair")
        if refinement not in {"ROOT", "L", "H"}:
            fail("unknown eta face refinement")
        receipt_id = f"{tile_id}_{refinement}"
        if receipt_id in seen[face]:
            fail("duplicate face/tile/refinement receipt")
        seen[face].add(receipt_id)
        if raw.get("support_worker_source_sha256") != support_worker_hash:
            fail("face receipt does not bind the retained support worker")
        current_hash = str(raw.get("worker_source_sha256"))
        if face_worker_hash and current_hash != face_worker_hash:
            fail("face worker hash mismatch")
        face_worker_hash = current_hash
        if raw.get("selected_source_face_chain_certificate") is not True:
            fail("source face lacks the second-return chain certificate")
        if raw.get("point_fallback_used") is not False:
            fail("point fallback is forbidden on exit faces")
        receipt_hashes[f"{face}_{receipt_id}"] = sha256(path)
        source_domain = raw.get("source_domain")
        eta_low, eta_high = map(Fraction, source_domain["eta"])
        cover_intervals[face].append((eta_low, eta_high))
        for item in raw.get("carriers", []):
            domain = item["source_domain"]
            xi_low, xi_high = map(Fraction, domain["xi"])
            eta_low, eta_high = map(Fraction, domain["eta"])
            expected_xi = Fraction(-1) if face == "LEFT" else Fraction(1)
            if xi_low != expected_xi or xi_high != expected_xi:
                fail("exit-face carrier does not have exact global xi")
            components = [tm2r(component) for component in item["components"]]
            if len(components) != 4:
                fail("exit-face carrier does not have four components")
            if exact_pair(range_on_box(components[2])) != (Fraction(0), Fraction(0)):
                fail("exit-face carrier is not exactly on w=0")
            if interval(item["event_normal"]).lower() <= 0:
                fail("exit-face carrier lacks positive transversality")
            result[face].append(
                Carrier(
                    str(item["carrier_id"]),
                    str(tile_id),
                    xi_low,
                    xi_high,
                    eta_low,
                    eta_high,
                    components,
                )
            )
    for face, expected_tiles in EXPECTED_FACE_TILES.items():
        observed_tiles = {item.split("_", 1)[0] for item in seen[face]}
        if observed_tiles != expected_tiles or not result[face]:
            fail(f"incomplete {face} exit-face cover")
        ordered = sorted(cover_intervals[face])
        if ordered[0][0] != -1 or ordered[-1][1] != 1:
            fail(f"{face} eta cover does not span [-1,1]")
        for current, following in zip(ordered, ordered[1:], strict=False):
            if current[1] != following[0]:
                fail(f"{face} eta cover has a gap or overlap")
    return result, receipt_hashes, face_worker_hash


def main() -> None:
    paths = [Path(argument) for argument in sys.argv[1:]]
    support_paths = []
    face_paths = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="ascii"))
        if raw.get("schema") == CARRIER_SCHEMA:
            support_paths.append(path)
        elif raw.get("schema") == FACE_SCHEMA:
            face_paths.append(path)
        else:
            fail(f"unknown receipt schema in {path}")
    carriers, receipt_hashes, worker_hash = load_carriers(support_paths)
    face_carriers, face_receipt_hashes, face_worker_hash = load_face_carriers(
        face_paths, worker_hash
    )
    if len(carriers) != 5:
        fail(f"expected five final carriers, found {len(carriers)}")

    target_models = [
        (carrier, *candidate_target_models(carrier)) for carrier in carriers
    ]
    stable_support = hull(
        [range_on_box(s_model) for _carrier, _u_model, s_model in target_models]
    )
    left_images = [
        range_on_box(candidate_target_models(carrier)[0])
        for carrier in face_carriers["LEFT"]
    ]
    right_images = [
        range_on_box(candidate_target_models(carrier)[0])
        for carrier in face_carriers["RIGHT"]
    ]
    left_image = hull(left_images)
    right_image = hull(right_images)
    left_lower, left_upper = exact_pair(left_image)
    right_lower, right_upper = exact_pair(right_image)

    left_center = (left_lower + left_upper) / 2
    right_center = (right_lower + right_upper) / 2
    if left_center < right_center:
        degree_candidate = 1
        center_gap = right_center - left_center
        signed_face_gap = right_lower - left_upper
    elif right_center < left_center:
        degree_candidate = -1
        center_gap = left_center - right_center
        signed_face_gap = left_lower - right_upper
    else:
        fail("exit-face center ordering is unresolved")
    target_u_center = (left_center + right_center) / 2
    target_u_radius = center_gap / 4
    stable_lower, stable_upper = exact_pair(stable_support)
    target_s_center = (stable_lower + stable_upper) / 2
    target_s_radius = stable_upper - stable_lower
    if target_u_radius <= 0 or target_s_radius <= 0:
        fail("constructed target h-set has a non-positive radius")

    normalized_stable = (
        stable_support - ball(target_s_center)
    ) / ball(target_s_radius)
    normalized_left = (left_image - ball(target_u_center)) / ball(target_u_radius)
    normalized_right = (right_image - ball(target_u_center)) / ball(target_u_radius)
    if normalized_stable.lower() <= -1 or normalized_stable.upper() >= 1:
        fail("support image meets the target entry boundary")
    if degree_candidate == 1:
        exit_margin = min(
            Fraction(-1) - Fraction(base.upper_fraction(normalized_left)),
            Fraction(base.lower_fraction(normalized_right)) - Fraction(1),
        )
    else:
        exit_margin = min(
            Fraction(base.lower_fraction(normalized_left)) - Fraction(1),
            Fraction(-1) - Fraction(base.upper_fraction(normalized_right)),
        )
    entry_margin = min(
        Fraction(base.lower_fraction(normalized_stable)) + 1,
        1 - Fraction(base.upper_fraction(normalized_stable)),
    )
    if entry_margin <= 0:
        fail("the target entry-boundary margin is non-positive")
    exit_certificate = exit_margin > 0 and signed_face_gap > 0
    face_overlap = max(Fraction(0), -signed_face_gap)

    initial, u_interval, s_interval = base.initial_leaf()
    initial_normal = initial[0].range() * initial[1].range() - base.ZS
    if initial_normal.lower() <= 0:
        fail("source h-set lacks positive initial normal velocity")
    frame_determinant = (
        base.UNSTABLE_X * base.STABLE_Y - base.STABLE_X * base.UNSTABLE_Y
    )
    source_u_lower, source_u_upper = exact_pair(u_interval)
    source_s_lower, source_s_upper = exact_pair(s_interval)
    source_u_center = (source_u_lower + source_u_upper) / 2
    source_s_center = (source_s_lower + source_s_upper) / 2
    source_u_radius = (source_u_upper - source_u_lower) / 2
    source_s_radius = (source_s_upper - source_s_lower) / 2
    source_chart_determinant = (
        frame_determinant * ball(source_u_radius) * ball(source_s_radius)
    )
    target_linear_determinant = (
        ball(TARGET_UNSTABLE_X_Q) ** 2
        + ball(TARGET_UNSTABLE_Y_Q) ** 2
    )
    target_chart_determinant = (
        ball(target_u_radius)
        * ball(target_s_radius)
        / target_linear_determinant
    )
    physical_determinants: list[arb] = []
    normalized_determinants: list[arb] = []
    for carrier in carriers:
        x, y, _w, ell = carrier.components
        final_normal_model = x * y - base.ZS
        final_normal = range_on_box(final_normal_model)
        if final_normal.lower() <= 0:
            fail(f"final normal sign unresolved for {carrier.carrier_id}")
        exp_ell = range_on_box(ell).exp()
        physical = exp_ell * initial_normal / final_normal * source_chart_determinant
        normalized = physical / target_chart_determinant
        if physical.upper() >= 0:
            fail(f"physical determinant sign unresolved for {carrier.carrier_id}")
        if normalized.upper() >= 0:
            fail(f"normalized determinant sign unresolved for {carrier.carrier_id}")
        physical_determinants.append(physical)
        normalized_determinants.append(normalized)

    print(f"SCHEMA={SCHEMA}")
    print(f"ANALYZER_SOURCE_SHA256={sha256(Path(__file__))}")
    print(f"CARRIER_WORKER_SOURCE_SHA256={worker_hash}")
    print(f"FACE_WORKER_SOURCE_SHA256={face_worker_hash}")
    print(f"LEAF_ID={base.LEAF_ID}")
    print("MAP=P2")
    print("SECTION=w=0")
    print("SOURCE_COORDINATES=FROZEN_UPO_FRAME_NORMALIZED_XI_ETA")
    print("TARGET_COORDINATES=FROZEN_RATIONAL_AFFINE_CANDIDATE")
    print("TARGET_GEOMETRY_SELECTION=BEST_DISCOVERED_LINEAR_FACE_DIRECTION_AND_SUPPORT_HULL")
    print(f"TARGET_UNSTABLE_ROW_X_Q={TARGET_UNSTABLE_X_Q}")
    print(f"TARGET_UNSTABLE_ROW_Y_Q={TARGET_UNSTABLE_Y_Q}")
    print(f"TARGET_STABLE_ROW_X_Q={-TARGET_UNSTABLE_Y_Q}")
    print(f"TARGET_STABLE_ROW_Y_Q={TARGET_UNSTABLE_X_Q}")
    print("SOURCE_TILES=" + ",".join(EXPECTED_TILES))
    print(f"SOURCE_TILE_COUNT={len(EXPECTED_TILES)}")
    print(f"SECOND_EVENT_CARRIER_COUNT={len(carriers)}")
    for tile_id in EXPECTED_TILES:
        print(f"CARRIER_RECEIPT_SHA256_{tile_id}={receipt_hashes[tile_id]}")
    print(f"FACE_RECEIPT_COUNT={len(face_receipt_hashes)}")
    for index, receipt_id in enumerate(sorted(face_receipt_hashes), start=1):
        print(f"FACE_RECEIPT_ID_{index}={receipt_id}")
        print(f"FACE_RECEIPT_SHA256_{index}={face_receipt_hashes[receipt_id]}")
    print(f"SOURCE_U_CENTER_Q={source_u_center}")
    print(f"SOURCE_U_RADIUS_Q={source_u_radius}")
    print(f"SOURCE_S_CENTER_Q={source_s_center}")
    print(f"SOURCE_S_RADIUS_Q={source_s_radius}")
    print(f"TARGET_U_CENTER_Q={target_u_center}")
    print(f"TARGET_U_RADIUS_Q={target_u_radius}")
    print(f"TARGET_S_CENTER_Q={target_s_center}")
    print(f"TARGET_S_RADIUS_Q={target_s_radius}")
    emit_interval("FRAME_DETERMINANT", frame_determinant)
    emit_interval("TARGET_LINEAR_DETERMINANT", target_linear_determinant)
    emit_interval("SOURCE_CHART_DETERMINANT", source_chart_determinant)
    emit_interval("TARGET_CHART_DETERMINANT", target_chart_determinant)
    emit_interval("SUPPORT_STABLE_IMAGE", normalized_stable)
    emit_interval("LEFT_EXIT_UNSTABLE_IMAGE", normalized_left)
    emit_interval("RIGHT_EXIT_UNSTABLE_IMAGE", normalized_right)
    print(f"ENTRY_MARGIN_Q={entry_margin}")
    print(f"EXIT_MARGIN_Q={exit_margin}")
    print(f"SIGNED_EXIT_FACE_GAP_Q={signed_face_gap}")
    print(f"EXIT_FACE_OVERLAP_Q={face_overlap}")
    print(f"COVERING_DEGREE_CANDIDATE={degree_candidate}")
    emit_interval("INITIAL_NORMAL", initial_normal)
    emit_interval("PHYSICAL_RETURN_DETERMINANT", hull(physical_determinants))
    emit_interval("NORMALIZED_RETURN_DETERMINANT", hull(normalized_determinants))
    print("HSET_COORDINATES_CERTIFICATE=true")
    print("ENTRY_BOUNDARY_AVOIDANCE_CERTIFICATE=true")
    print(f"EXIT_FACE_INEQUALITIES_CERTIFICATE={str(exit_certificate).lower()}")
    print("COVERING_DEGREE_CERTIFICATE=false")
    print("RETURN_MAP_DETERMINANT_CERTIFICATE=true")
    print("LOCAL_HSET_COVERING_RELATION_CERTIFICATE=false")
    print("COVERING_CANDIDATE_FALSIFIED_BY=EXIT_FACE_OVERLAP")
    print("RECURRENT_COVERING_GRAPH_CERTIFICATE=false")
    print("FIBONACCI_COVERING_CERTIFICATE=false")
    print("GLOBAL_HPG_CERTIFICATE=false")
    print("CHAOS_PROVED=false")
    print("CHAOTIC_ATTRACTOR_PROVED=false")
    print("OPEN_PROBLEM_SOLVED=false")
    print("NOVELTY_OR_PRIORITY_CLAIMED=false")
    print("CAPD_USED=false")
    print("POINT_FALLBACK_USED=false")


if __name__ == "__main__":
    main()
