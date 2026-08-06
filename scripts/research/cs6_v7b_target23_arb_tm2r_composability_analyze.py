#!/usr/bin/env python3
"""Derive h-set C and certify the local covering B -> C from four receipts."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from flint import arb

import cs6_v7b_target23_arb_tm2r_composability_carrier_worker as worker
import cs6_v7b_target23_arb_tm2r_first_return_worker as base


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-composability-covering.v1"
CARRIER_SCHEMA = worker.SCHEMA
EXPECTED_TILES = worker.EXPECTED_TILES
VARIABLE_NAMES = ("XI", "ETA", "RHO0", "RHO1", "RHO2", "RHO3")
EXPECTED_ROOTS = {
    "XLEL": ((-1, 0), (-1, 0)),
    "XLEH": ((-1, 0), (0, 1)),
    "XHEL": ((0, 1), (-1, 0)),
    "XHEH": ((0, 1), (0, 1)),
}


@dataclass(frozen=True)
class Domain:
    bounds: tuple[tuple[Fraction, Fraction], ...]
    lineage: tuple[str, ...]


@dataclass(frozen=True)
class Carrier:
    carrier_id: str
    tile_id: str
    domain: Domain
    components: tuple[base.TM2R, ...]


def fail(message: str) -> None:
    raise SystemExit(f"composability analysis error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ball(value: Fraction) -> arb:
    return base.rational_ball(value)


def interval(raw: object) -> arb:
    if not isinstance(raw, list) or len(raw) != 2:
        fail("invalid serialized interval")
    lower, upper = map(Fraction, raw)
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
            fail("invalid TM2R coefficient")
        monomial = tuple(int(value) for value in entry["monomial"])
        if len(monomial) != base.VARIABLES or any(value < 0 for value in monomial):
            fail("invalid TM2R monomial")
        if monomial in coefficients:
            fail("duplicate TM2R monomial")
        coefficients[monomial] = interval(entry["interval"])
    return base.TM2R(coefficients, interval(raw.get("remainder")))


def range_on_box(
    model: base.TM2R, fixed: dict[int, Fraction] | None = None
) -> arb:
    variables = [base.UNIT for _ in range(base.VARIABLES)]
    for variable, value in (fixed or {}).items():
        variables[variable] = ball(value)
    result = arb(0)
    for monomial, coefficient in model.coefficients.items():
        term = coefficient
        for variable, exponent in enumerate(monomial):
            for _ in range(exponent):
                term *= variables[variable]
        result += term
    return result + model.remainder


def exact_pair(value: arb) -> tuple[Fraction, Fraction]:
    return Fraction(base.lower_fraction(value)), Fraction(base.upper_fraction(value))


def hull(values: list[arb]) -> arb:
    if not values:
        fail("cannot hull an empty family")
    result = values[0]
    for value in values[1:]:
        result = result.union(value)
    return result


def emit_interval(prefix: str, value: arb) -> None:
    print(f"{prefix}_LOWER_Q={base.lower_fraction(value)}")
    print(f"{prefix}_UPPER_Q={base.upper_fraction(value)}")


def parse_bounds(raw: object) -> tuple[tuple[Fraction, Fraction], ...]:
    if not isinstance(raw, dict):
        fail("domain bounds are missing")
    result = []
    for name in VARIABLE_NAMES:
        pair = raw.get(name.lower())
        if not isinstance(pair, list) or len(pair) != 2:
            fail(f"invalid domain bound for {name}")
        lower, upper = map(Fraction, pair)
        if lower > upper:
            fail(f"reversed domain bound for {name}")
        result.append((lower, upper))
    return tuple(result)


def parse_domain(raw: object) -> Domain:
    if not isinstance(raw, dict):
        fail("invalid symbolic domain")
    bounds = parse_bounds(raw.get("bounds"))
    lineage_raw = raw.get("split_lineage")
    trace = raw.get("split_trace")
    if not isinstance(lineage_raw, list) or not isinstance(trace, list):
        fail("domain lineage or trace is missing")
    lineage = tuple(str(item) for item in lineage_raw)
    if len(lineage) != len(trace):
        fail("domain lineage and trace lengths differ")

    reconstructed = [(Fraction(-1), Fraction(1)) for _ in VARIABLE_NAMES]
    expected_lineage: list[str] = []
    for item in trace:
        if not isinstance(item, dict):
            fail("invalid domain split trace entry")
        name = str(item.get("variable"))
        if name not in VARIABLE_NAMES:
            fail("unknown split variable")
        variable = VARIABLE_NAMES.index(name)
        side = str(item.get("side"))
        if side not in {"LEFT", "RIGHT"}:
            fail("unknown split side")
        parent = tuple(map(Fraction, item.get("parent", [])))
        child = tuple(map(Fraction, item.get("child", [])))
        if len(parent) != 2 or len(child) != 2 or parent != reconstructed[variable]:
            fail("split trace parent does not reconstruct")
        cut = Fraction(str(item.get("cut")))
        if cut != (parent[0] + parent[1]) / 2:
            fail("split trace is not an exact rational bisection")
        expected_child = (parent[0], cut) if side == "LEFT" else (cut, parent[1])
        if child != expected_child:
            fail("split trace child does not partition its parent")
        expected_center = Fraction(-1, 2) if side == "LEFT" else Fraction(1, 2)
        if Fraction(str(item.get("tm2r_substitution_center"))) != expected_center:
            fail("wrong TM2R split substitution center")
        if Fraction(str(item.get("tm2r_substitution_radius"))) != Fraction(1, 2):
            fail("wrong TM2R split substitution radius")
        reconstructed[variable] = expected_child
        expected_lineage.append(name + ("L" if side == "LEFT" else "H"))
    if tuple(expected_lineage) != lineage or tuple(reconstructed) != bounds:
        fail("serialized symbolic domain does not reconstruct exactly")
    return Domain(bounds, lineage)


def certify_terminal_cover(root: Domain, leaves: list[Domain]) -> int:
    terminal = {leaf.lineage for leaf in leaves}
    if len(terminal) != len(leaves) or not terminal:
        fail("terminal domain lineages are empty or duplicated")
    if any(path[: len(root.lineage)] != root.lineage for path in terminal):
        fail("terminal domain escaped its source tile")
    checks = 0

    def covers(prefix: tuple[str, ...]) -> bool:
        nonlocal checks
        if prefix in terminal:
            checks += 1
            return not any(
                path != prefix and path[: len(prefix)] == prefix for path in terminal
            )
        next_tokens = {
            path[len(prefix)]
            for path in terminal
            if len(path) > len(prefix) and path[: len(prefix)] == prefix
        }
        variables = {token[:-1] for token in next_tokens if token[-1:] in {"L", "H"}}
        if len(variables) != 1:
            return False
        variable = next(iter(variables))
        if next_tokens != {variable + "L", variable + "H"}:
            return False
        checks += 1
        return covers(prefix + (variable + "L",)) and covers(
            prefix + (variable + "H",)
        )

    if not covers(root.lineage):
        fail("terminal domains do not form an exact binary cover")
    return checks


def certify_boundary_cover(root: Domain, leaves: list[Domain], face: Fraction) -> int:
    """Recompute the exact binary cover induced on one global xi face."""
    terminal = {leaf.lineage: leaf for leaf in leaves}
    checks = 0

    def covers(prefix: tuple[str, ...]) -> bool:
        nonlocal checks
        if prefix in terminal:
            lower, upper = terminal[prefix].bounds[0]
            checks += 1
            return lower == face if face == -1 else upper == face
        next_tokens = {
            path[len(prefix)]
            for path in terminal
            if len(path) > len(prefix) and path[: len(prefix)] == prefix
        }
        variables = {token[:-1] for token in next_tokens if token[-1:] in {"L", "H"}}
        if len(variables) != 1:
            return False
        variable = next(iter(variables))
        checks += 1
        if variable == "XI":
            suffix = "L" if face == -1 else "H"
            token = variable + suffix
            return token in next_tokens and covers(prefix + (token,))
        if next_tokens != {variable + "L", variable + "H"}:
            return False
        return covers(prefix + (variable + "L",)) and covers(
            prefix + (variable + "H",)
        )

    if not covers(root.lineage):
        fail(f"terminal domains do not cover the induced xi={face} face")
    return checks


def load_receipts(
    paths: list[Path],
) -> tuple[
    list[Carrier],
    dict[str, str],
    str,
    dict[str, int],
    dict[str, Domain],
    dict[str, tuple[str, str]],
]:
    if len(paths) != len(EXPECTED_TILES):
        fail("exactly four full-support receipts are required")
    carriers: list[Carrier] = []
    receipt_hashes: dict[str, str] = {}
    worker_hash = ""
    cover_checks: dict[str, int] = {}
    roots: dict[str, Domain] = {}
    execution_profiles: dict[str, tuple[str, str]] = {}
    seen: set[str] = set()
    for path in paths:
        raw = json.loads(path.read_text(encoding="ascii"))
        if raw.get("schema") != CARRIER_SCHEMA:
            fail(f"wrong carrier schema in {path}")
        tile_id = str(raw.get("tile_id"))
        if tile_id not in EXPECTED_TILES or tile_id in seen:
            fail("missing, duplicate, or unknown source tile")
        seen.add(tile_id)
        if raw.get("source_hset") != "B" or raw.get("map") != "P^2":
            fail("receipt transports the wrong source h-set or map")
        if raw.get("leaf_id", base.LEAF_ID) != base.LEAF_ID:
            fail("receipt has the wrong leaf id")
        required_true = (
            "terminal_domain_cover_certified",
            "selected_source_chain_certificate",
            "deterministic_no_rng",
        )
        if any(raw.get(field) is not True for field in required_true):
            fail(f"tile {tile_id} lacks a required positive certificate")
        if raw.get("point_fallback_used") is not False:
            fail(f"tile {tile_id} used point fallback")
        if raw.get("box_flattening_used") is not False:
            fail(f"tile {tile_id} flattened its Taylor-model remainder")
        current_hash = str(raw.get("worker_source_sha256"))
        if worker_hash and current_hash != worker_hash:
            fail("worker hash differs across tile receipts")
        worker_hash = current_hash
        receipt_hashes[tile_id] = sha256(path)
        profile = str(raw.get("execution_profile", "BASE_SPLIT_BUDGET_V1"))
        if profile == "BASE_SPLIT_BUDGET_V1":
            if raw.get("execution_wrapper_source_sha256") is not None:
                fail("base execution unexpectedly declares a wrapper hash")
            wrapper_hash = "NONE"
        elif profile == "EXTENDED_SPLIT_BUDGET_V1":
            retry_worker = Path(__file__).with_name(
                "cs6_v7b_target23_arb_tm2r_composability_retry_worker.py"
            )
            wrapper_hash = str(raw.get("execution_wrapper_source_sha256"))
            if wrapper_hash != sha256(retry_worker):
                fail("extended retry wrapper hash mismatch")
            if raw.get("max_event_split_depth") != 12:
                fail("extended retry has the wrong split-depth budget")
            if raw.get("max_event_split_nodes_per_tile") != 255:
                fail("extended retry has the wrong split-node budget")
        else:
            fail(f"unknown execution profile {profile}")
        execution_profiles[tile_id] = (profile, wrapper_hash)

        root = parse_domain(raw.get("source_domain"))
        expected_xi, expected_eta = EXPECTED_ROOTS[tile_id]
        expected_bounds = (
            tuple(map(Fraction, expected_xi)),
            tuple(map(Fraction, expected_eta)),
            *((Fraction(-1), Fraction(1)) for _ in range(4)),
        )
        if root.bounds != expected_bounds:
            fail(f"tile {tile_id} has the wrong exact source domain")
        roots[tile_id] = root
        raw_carriers = raw.get("carriers")
        if not isinstance(raw_carriers, list) or not raw_carriers:
            fail(f"tile {tile_id} contains no terminal carrier")
        tile_carriers: list[Carrier] = []
        for item in raw_carriers:
            if not isinstance(item, dict):
                fail("invalid terminal carrier")
            domain = parse_domain(item.get("source_domain"))
            components_raw = item.get("components")
            if not isinstance(components_raw, list) or len(components_raw) != 4:
                fail("terminal carrier does not have four TM2R components")
            components = tuple(tm2r(component) for component in components_raw)
            if exact_pair(range_on_box(components[2])) != (Fraction(0), Fraction(0)):
                fail("terminal carrier is not exactly on w=0")
            if interval(item.get("event_normal")).lower() <= 0:
                fail("terminal carrier lacks strict positive transversality")
            tile_carriers.append(
                Carrier(str(item.get("carrier_id")), tile_id, domain, components)
            )
        cover_checks[tile_id] = certify_terminal_cover(
            root, [carrier.domain for carrier in tile_carriers]
        )
        carriers.extend(tile_carriers)
    if seen != set(EXPECTED_TILES):
        fail("the four source tiles do not cover B")
    return (
        carriers,
        receipt_hashes,
        worker_hash,
        cover_checks,
        roots,
        execution_profiles,
    )


def boundary_carriers(carriers: list[Carrier], face: Fraction) -> list[Carrier]:
    result = []
    for carrier in carriers:
        lower, upper = carrier.domain.bounds[0]
        if face == -1 and lower == face:
            result.append(carrier)
        elif face == 1 and upper == face:
            result.append(carrier)
    if not result:
        fail(f"no terminal carriers meet source face xi={face}")
    expected_tiles = {"XLEL", "XLEH"} if face == -1 else {"XHEL", "XHEH"}
    if {carrier.tile_id for carrier in result} != expected_tiles:
        fail(f"the derived xi={face} face misses a source tile")
    return result


def projected_models(
    carrier: Carrier, nx: Fraction, ny: Fraction
) -> tuple[base.TM2R, base.TM2R]:
    x, y = carrier.components[:2]
    return ball(nx) * x + ball(ny) * y, -ball(ny) * x + ball(nx) * y


def main() -> None:
    paths = [Path(argument) for argument in sys.argv[1:]]
    (
        carriers,
        receipt_hashes,
        worker_hash,
        cover_checks,
        roots,
        execution_profiles,
    ) = load_receipts(paths)
    left = boundary_carriers(carriers, Fraction(-1))
    right = boundary_carriers(carriers, Fraction(1))
    boundary_cover_checks = {"LEFT": 0, "RIGHT": 0}
    for tile_id in ("XLEL", "XLEH"):
        boundary_cover_checks["LEFT"] += certify_boundary_cover(
            roots[tile_id],
            [carrier.domain for carrier in carriers if carrier.tile_id == tile_id],
            Fraction(-1),
        )
    for tile_id in ("XHEL", "XHEH"):
        boundary_cover_checks["RIGHT"] += certify_boundary_cover(
            roots[tile_id],
            [carrier.domain for carrier in carriers if carrier.tile_id == tile_id],
            Fraction(1),
        )

    left_x = hull([range_on_box(carrier.components[0], {0: Fraction(-1)}) for carrier in left])
    left_y = hull([range_on_box(carrier.components[1], {0: Fraction(-1)}) for carrier in left])
    right_x = hull([range_on_box(carrier.components[0], {0: Fraction(1)}) for carrier in right])
    right_y = hull([range_on_box(carrier.components[1], {0: Fraction(1)}) for carrier in right])
    left_x_pair, left_y_pair = exact_pair(left_x), exact_pair(left_y)
    right_x_pair, right_y_pair = exact_pair(right_x), exact_pair(right_y)
    nx = sum(right_x_pair) / 2 - sum(left_x_pair) / 2
    ny = sum(right_y_pair) / 2 - sum(left_y_pair) / 2
    if nx == 0 and ny == 0:
        fail("the rational boundary-hull centers coincide")

    def face_images(row_x: Fraction, row_y: Fraction) -> tuple[arb, arb]:
        left_values = [
            range_on_box(projected_models(carrier, row_x, row_y)[0], {0: Fraction(-1)})
            for carrier in left
        ]
        right_values = [
            range_on_box(projected_models(carrier, row_x, row_y)[0], {0: Fraction(1)})
            for carrier in right
        ]
        return hull(left_values), hull(right_values)

    left_u, right_u = face_images(nx, ny)
    left_u_pair, right_u_pair = exact_pair(left_u), exact_pair(right_u)
    left_center = sum(left_u_pair) / 2
    right_center = sum(right_u_pair) / 2
    if right_center < left_center:
        nx, ny = -nx, -ny
        left_u, right_u = face_images(nx, ny)
        left_u_pair, right_u_pair = exact_pair(left_u), exact_pair(right_u)
        left_center = sum(left_u_pair) / 2
        right_center = sum(right_u_pair) / 2
    if right_center <= left_center:
        fail("the derived unstable row does not order the exit-face centers")
    signed_face_gap = right_u_pair[0] - left_u_pair[1]
    if signed_face_gap <= 0:
        fail(f"exit-face images overlap by at least {-signed_face_gap}")

    target_u_center = (left_u_pair[1] + right_u_pair[0]) / 2
    target_u_radius = signed_face_gap / 4
    target_models = [projected_models(carrier, nx, ny) for carrier in carriers]
    stable_support = hull([range_on_box(stable) for _unstable, stable in target_models])
    stable_pair = exact_pair(stable_support)
    target_s_center = sum(stable_pair) / 2
    target_s_radius = stable_pair[1] - stable_pair[0]
    if target_s_radius <= 0:
        fail("the derived target stable radius is not positive")

    normalized_left = (left_u - ball(target_u_center)) / ball(target_u_radius)
    normalized_right = (right_u - ball(target_u_center)) / ball(target_u_radius)
    normalized_stable = (
        stable_support - ball(target_s_center)
    ) / ball(target_s_radius)
    if normalized_left.upper() >= -1:
        fail("the left exit face meets C")
    if normalized_right.lower() <= 1:
        fail("the right exit face meets C")
    if normalized_stable.lower() <= -1 or normalized_stable.upper() >= 1:
        fail("the support image meets an entry boundary of C")
    exit_margin = min(
        Fraction(-1) - Fraction(base.upper_fraction(normalized_left)),
        Fraction(base.lower_fraction(normalized_right)) - Fraction(1),
    )
    entry_margin = min(
        Fraction(base.lower_fraction(normalized_stable)) + 1,
        Fraction(1) - Fraction(base.upper_fraction(normalized_stable)),
    )
    if exit_margin <= 0 or entry_margin <= 0:
        fail("a normalized covering margin is not strictly positive")

    source_state = worker.target_hset()
    initial_normal = range_on_box(source_state[0] * source_state[1] - base.ZS)
    if initial_normal.lower() <= 0:
        fail("source h-set B lacks strict positive normal velocity")
    source_linear_determinant = ball(
        worker.ROW_U_X * worker.ROW_U_X + worker.ROW_U_Y * worker.ROW_U_Y
    )
    source_chart_determinant = (
        ball(worker.U_RADIUS) * ball(worker.S_RADIUS) / source_linear_determinant
    )
    target_linear_determinant = ball(nx * nx + ny * ny)
    target_chart_determinant = (
        ball(target_u_radius) * ball(target_s_radius) / target_linear_determinant
    )
    if source_chart_determinant.lower() <= 0 or target_chart_determinant.lower() <= 0:
        fail("a source or target h-set chart is singular")

    physical_determinants = []
    normalized_determinants = []
    final_normals = []
    for carrier in carriers:
        x, y, _w, ell = carrier.components
        final_normal = range_on_box(x * y - base.ZS)
        if final_normal.lower() <= 0:
            fail(f"final normal sign unresolved for {carrier.carrier_id}")
        physical = (
            range_on_box(ell).exp()
            * initial_normal
            / final_normal
            * source_chart_determinant
        )
        normalized = physical / target_chart_determinant
        if physical.lower() <= 0 or normalized.lower() <= 0:
            fail(f"return determinant sign unresolved for {carrier.carrier_id}")
        final_normals.append(final_normal)
        physical_determinants.append(physical)
        normalized_determinants.append(normalized)

    print(f"SCHEMA={SCHEMA}")
    print(f"ANALYZER_SOURCE_SHA256={sha256(Path(__file__))}")
    print(f"CARRIER_WORKER_SOURCE_SHA256={worker_hash}")
    print(f"LEAF_ID={base.LEAF_ID}")
    print("SOURCE_HSET=B")
    print("TARGET_HSET=C")
    print("MAP=P2")
    print("SECTION=w=0")
    print("TARGET_ROW_DERIVATION=RIGHT_MINUS_LEFT_RIGOROUS_BOUNDARY_HULL_CENTER")
    print("EXIT_FACE_DERIVATION=FULL_SUPPORT_BOUNDARY_RESTRICTION_WITH_XI_FIXED")
    print("SOURCE_CHART_CONVENTION=INVERSE_OF_ORTHOGONAL_ROW_COORDINATES")
    print("TARGET_CHART_CONVENTION=INVERSE_OF_ORTHOGONAL_ROW_COORDINATES")
    print("INITIAL_NORMAL_DEFINITION=x*y-ZS=dw/dt_ON_SECTION")
    print("POINCARE_DETERMINANT_FORMULA=exp(ell)*initial_normal/final_normal")
    print("SYMBOLIC_VARIABLES_PRESERVED=xi,eta,rho0,rho1,rho2,rho3")
    print("HSET_UNSTABLE_DIMENSION=1")
    print("HSET_STABLE_DIMENSION=1")
    print("TM_RESIDUAL_VARIABLES_ARE_ENCLOSURE_PARAMETERS=true")
    print("COVERING_DEGREE_OBJECT=UNSTABLE_TERMINAL_MAP_A")
    print("SOURCE_TILES=" + ",".join(EXPECTED_TILES))
    print(f"SOURCE_TILE_COUNT={len(EXPECTED_TILES)}")
    print(f"SECOND_EVENT_CARRIER_COUNT={len(carriers)}")
    print(f"LEFT_EXIT_CARRIER_COUNT={len(left)}")
    print(f"RIGHT_EXIT_CARRIER_COUNT={len(right)}")
    print(f"LEFT_EXIT_COVER_CHECKS={boundary_cover_checks['LEFT']}")
    print(f"RIGHT_EXIT_COVER_CHECKS={boundary_cover_checks['RIGHT']}")
    for tile_id in EXPECTED_TILES:
        print(f"CARRIER_RECEIPT_SHA256_{tile_id}={receipt_hashes[tile_id]}")
        print(f"TERMINAL_COVER_CHECKS_{tile_id}={cover_checks[tile_id]}")
        profile, wrapper_hash = execution_profiles[tile_id]
        print(f"EXECUTION_PROFILE_{tile_id}={profile}")
        print(f"EXECUTION_WRAPPER_SOURCE_SHA256_{tile_id}={wrapper_hash}")
    print(f"SOURCE_U_CENTER_Q={worker.U_CENTER}")
    print(f"SOURCE_U_RADIUS_Q={worker.U_RADIUS}")
    print(f"SOURCE_S_CENTER_Q={worker.S_CENTER}")
    print(f"SOURCE_S_RADIUS_Q={worker.S_RADIUS}")
    print(f"TARGET_UNSTABLE_ROW_X_Q={nx}")
    print(f"TARGET_UNSTABLE_ROW_Y_Q={ny}")
    print(f"TARGET_STABLE_ROW_X_Q={-ny}")
    print(f"TARGET_STABLE_ROW_Y_Q={nx}")
    print(f"TARGET_U_CENTER_Q={target_u_center}")
    print(f"TARGET_U_RADIUS_Q={target_u_radius}")
    print(f"TARGET_S_CENTER_Q={target_s_center}")
    print(f"TARGET_S_RADIUS_Q={target_s_radius}")
    emit_interval("SOURCE_LINEAR_DETERMINANT", source_linear_determinant)
    emit_interval("TARGET_LINEAR_DETERMINANT", target_linear_determinant)
    emit_interval("SOURCE_CHART_DETERMINANT", source_chart_determinant)
    emit_interval("TARGET_CHART_DETERMINANT", target_chart_determinant)
    emit_interval("SUPPORT_STABLE_RAW", stable_support)
    emit_interval("LEFT_EXIT_UNSTABLE_RAW", left_u)
    emit_interval("RIGHT_EXIT_UNSTABLE_RAW", right_u)
    emit_interval("SUPPORT_STABLE_IMAGE", normalized_stable)
    emit_interval("LEFT_EXIT_UNSTABLE_IMAGE", normalized_left)
    emit_interval("RIGHT_EXIT_UNSTABLE_IMAGE", normalized_right)
    print(f"ENTRY_MARGIN_Q={entry_margin}")
    print(f"EXIT_MARGIN_Q={exit_margin}")
    print(f"SIGNED_EXIT_FACE_GAP_Q={signed_face_gap}")
    print("COVERING_DEGREE=1")
    print("DEGREE_ARGUMENT=ONE_DIMENSIONAL_BOUNDARY_SIGN_HOMOTOPY")
    emit_interval("INITIAL_NORMAL", initial_normal)
    emit_interval("FINAL_NORMAL", hull(final_normals))
    emit_interval("PHYSICAL_RETURN_DETERMINANT", hull(physical_determinants))
    emit_interval("NORMALIZED_RETURN_DETERMINANT", hull(normalized_determinants))
    print("FULL_SUPPORT_CERTIFICATE=true")
    print("DERIVED_EXIT_FACE_COVER_CERTIFICATE=true")
    print("HSET_C_COORDINATES_CERTIFICATE=true")
    print("ENTRY_BOUNDARY_AVOIDANCE_CERTIFICATE=true")
    print("EXIT_FACE_INEQUALITIES_CERTIFICATE=true")
    print("COVERING_DEGREE_CERTIFICATE=true")
    print("RETURN_MAP_DETERMINANT_CERTIFICATE=true")
    print("LOCAL_HSET_COVERING_RELATION_B_TO_C_CERTIFICATE=true")
    print("RECURRENT_COVERING_GRAPH_CERTIFICATE=false")
    print("CHAOS_PROVED=false")
    print("OPEN_PROBLEM_SOLVED=false")
    print("NOVELTY_OR_PRIORITY_CLAIMED=false")
    print("CAPD_USED=false")
    print("POINT_FALLBACK_USED=false")
    print("BOX_FLATTENING_USED=false")


if __name__ == "__main__":
    main()
