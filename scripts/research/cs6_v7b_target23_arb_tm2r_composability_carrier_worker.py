#!/usr/bin/env python3
"""Transport the certified first target h-set through one more rigorous P^2."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import flint
from flint import arb

import cs6_v7b_target23_arb_tm2r_first_return_worker as base
import cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker as adaptive
import cs6_v7b_target23_arb_tm2r_hset_covering_carrier_worker as transport


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-composability-carrier.v1"
EXPECTED_TILES = transport.EXPECTED_TILES
MAX_FIRST_RETURN_STEPS = 2600
MAX_EVENT_SLAB_RADIUS = Fraction(1, 128)

# This is the target chart certified by covector_qr_v1.  Its normalized square
# is the source h-set B for the composability experiment.
ROW_U_X = Fraction(-4644852547588741, 6250000000000000)
ROW_U_Y = Fraction(13381910583555019, 20000000000000000)
U_CENTER = Fraction(
    -30739344851635465032436654293358422308448919,
    7737125245533626718119526400000000000000000,
)
U_RADIUS = Fraction(
    86903166009589810455173320337792871,
    618970019642690137449562112000000000000000,
)
S_CENTER = Fraction(
    -45691635639503762713334489694132757594034167712210370369237054484233348401655,
    3618502788666131106986593281521497120414687020801267626233049500247285301248,
)
S_RADIUS = Fraction(592923315, 274877906944)


@dataclass(frozen=True)
class DomainSplit:
    variable: int
    side: int
    parent_low: Fraction
    cut: Fraction
    parent_high: Fraction
    child_low: Fraction
    child_high: Fraction

    def as_json(self) -> dict[str, object]:
        return {
            "variable": adaptive.VARIABLE_NAMES[self.variable],
            "side": "LEFT" if self.side < 0 else "RIGHT",
            "parent": [str(self.parent_low), str(self.parent_high)],
            "cut": str(self.cut),
            "child": [str(self.child_low), str(self.child_high)],
            "tm2r_substitution_center": str(Fraction(self.side, 2)),
            "tm2r_substitution_radius": "1/2",
        }


@dataclass(frozen=True)
class SymbolicDomain:
    centers: tuple[Fraction, ...]
    radii: tuple[Fraction, ...]
    lineage: tuple[str, ...] = ()
    split_trace: tuple[DomainSplit, ...] = ()

    @classmethod
    def unit(cls) -> "SymbolicDomain":
        return cls(
            tuple(Fraction(0) for _ in range(base.VARIABLES)),
            tuple(Fraction(1) for _ in range(base.VARIABLES)),
        )

    def split(self, variable: int, side: int) -> "SymbolicDomain":
        if not 0 <= variable < base.VARIABLES or side not in (-1, 1):
            raise base.Refusal("INVALID_DOMAIN_SPLIT", "invalid variable or side")
        centers = list(self.centers)
        radii = list(self.radii)
        parent_center = centers[variable]
        parent_radius = radii[variable]
        centers[variable] = parent_center + side * parent_radius / 2
        radii[variable] = parent_radius / 2
        suffix = "L" if side < 0 else "H"
        parent_low = parent_center - parent_radius
        parent_high = parent_center + parent_radius
        child_low = centers[variable] - radii[variable]
        child_high = centers[variable] + radii[variable]
        cut = parent_center
        split = DomainSplit(
            variable,
            side,
            parent_low,
            cut,
            parent_high,
            child_low,
            child_high,
        )
        child = SymbolicDomain(
            tuple(centers),
            tuple(radii),
            self.lineage + (adaptive.VARIABLE_NAMES[variable] + suffix,),
            self.split_trace + (split,),
        )
        expected = (parent_low, cut) if side < 0 else (cut, parent_high)
        if (child_low, child_high) != expected:
            raise base.Refusal(
                "DOMAIN_PARTITION_FAILED", "child bounds do not partition parent"
            )
        return child

    def as_json(self) -> dict[str, object]:
        bounds: dict[str, list[str]] = {}
        for index, name in enumerate(adaptive.VARIABLE_NAMES):
            bounds[name.lower()] = [
                str(self.centers[index] - self.radii[index]),
                str(self.centers[index] + self.radii[index]),
            ]
        return {
            "bounds": bounds,
            "split_lineage": list(self.lineage),
            "split_trace": [split.as_json() for split in self.split_trace],
        }


def split_domain_pair(
    domain: SymbolicDomain, variable: int
) -> tuple[SymbolicDomain, SymbolicDomain]:
    left = domain.split(variable, -1)
    right = domain.split(variable, 1)
    for index in range(base.VARIABLES):
        parent_bounds = (
            domain.centers[index] - domain.radii[index],
            domain.centers[index] + domain.radii[index],
        )
        left_bounds = (
            left.centers[index] - left.radii[index],
            left.centers[index] + left.radii[index],
        )
        right_bounds = (
            right.centers[index] - right.radii[index],
            right.centers[index] + right.radii[index],
        )
        if index == variable:
            if (
                left_bounds[0] != parent_bounds[0]
                or left_bounds[1] != right_bounds[0]
                or right_bounds[1] != parent_bounds[1]
            ):
                raise base.Refusal(
                    "DOMAIN_PARTITION_FAILED", "children do not exactly cover parent"
                )
        elif left_bounds != parent_bounds or right_bounds != parent_bounds:
            raise base.Refusal(
                "DOMAIN_PARTITION_FAILED", "split changed an orthogonal variable"
            )
    return left, right


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_hset() -> list[base.TM2R]:
    determinant = ROW_U_X * ROW_U_X + ROW_U_Y * ROW_U_Y
    if determinant <= 0:
        raise base.Refusal("DEGENERATE_SOURCE_CHART", "target chart determinant is nonpositive")

    xi = [0] * base.VARIABLES
    xi[0] = 1
    eta = [0] * base.VARIABLES
    eta[1] = 1
    xi_monomial = tuple(xi)
    eta_monomial = tuple(eta)

    x_center = (ROW_U_X * U_CENTER - ROW_U_Y * S_CENTER) / determinant
    y_center = (ROW_U_Y * U_CENTER + ROW_U_X * S_CENTER) / determinant
    x = base.TM2R(
        {
            base.ZERO_MONOMIAL: base.rational_ball(x_center),
            xi_monomial: base.rational_ball(ROW_U_X * U_RADIUS / determinant),
            eta_monomial: base.rational_ball(-ROW_U_Y * S_RADIUS / determinant),
        }
    )
    y = base.TM2R(
        {
            base.ZERO_MONOMIAL: base.rational_ball(y_center),
            xi_monomial: base.rational_ball(ROW_U_Y * U_RADIUS / determinant),
            eta_monomial: base.rational_ball(ROW_U_X * S_RADIUS / determinant),
        }
    )
    return [x, y, base.TM2R.constant(0), base.TM2R.constant(0)]


def source_tiles() -> tuple[dict[str, tuple[list[base.TM2R], SymbolicDomain]], int]:
    initial = target_hset()
    xi_left, xi_right, checks = adaptive.split_state(initial, 0)
    unit_domain = SymbolicDomain.unit()
    xi_left_domain, xi_right_domain = split_domain_pair(unit_domain, 0)
    result: dict[str, tuple[list[base.TM2R], SymbolicDomain]] = {}
    for xi_id, xi_state, xi_domain in (
        ("XL", xi_left, xi_left_domain),
        ("XH", xi_right, xi_right_domain),
    ):
        eta_left, eta_right, eta_checks = adaptive.split_state(xi_state, 1)
        checks += eta_checks
        eta_left_domain, eta_right_domain = split_domain_pair(xi_domain, 1)
        result[xi_id + "EL"] = (
            eta_left,
            eta_left_domain,
        )
        result[xi_id + "EH"] = (
            eta_right,
            eta_right_domain,
        )
    return result, checks


def hull(values: list[arb]) -> arb:
    if not values:
        raise base.Refusal("EMPTY_EVENT_HULL", "cannot hull an empty event family")
    result = values[0]
    for value in values[1:]:
        result = result.union(value)
    return result


def certify_terminal_domain_cover(
    root: SymbolicDomain, leaves: list[SymbolicDomain]
) -> int:
    """Verify that terminal binary split lineages exactly cover the root domain."""
    terminal = {leaf.lineage for leaf in leaves}
    if not terminal:
        raise base.Refusal("EMPTY_DOMAIN_COVER", "no terminal symbolic domains")
    if any(path[: len(root.lineage)] != root.lineage for path in terminal):
        raise base.Refusal("DOMAIN_COVER_ESCAPED_ROOT", "terminal lineage escaped root")
    checks = 0

    def covers(prefix: tuple[str, ...]) -> bool:
        nonlocal checks
        if prefix in terminal:
            if any(path != prefix and path[: len(prefix)] == prefix for path in terminal):
                return False
            checks += 1
            return True
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
        raise base.Refusal(
            "DOMAIN_COVER_INCOMPLETE",
            "terminal symbolic split lineages do not form a binary cover",
        )
    return checks


def find_event_slab_extended(
    state: list[base.TM2R], orientation: int
) -> tuple[Fraction, list[arb], arb, base.TM2R, arb]:
    """Run the existing interval-Newton test with one additional slab scale."""
    if orientation not in (-1, 1):
        raise ValueError("event orientation must be -1 or 1")
    initial_ranges = [component.range() for component in state]
    diagnostics: list[str] = []
    for power in range(18, 6, -1):
        radius = Fraction(1, 2**power)
        try:
            backward_box, _iterations, _contraction = (
                transport.event.signed_picard_box(
                    initial_ranges, base.rational_ball(-radius)
                )
            )
            forward_box, _iterations, _contraction = (
                transport.event.signed_picard_box(
                    initial_ranges, base.rational_ball(radius)
                )
            )
        except base.Refusal as refusal:
            diagnostics.append(f"2^-{power}:{refusal.failure_class}")
            continue
        tube = [
            backward_component.union(forward_component)
            for backward_component, forward_component in zip(
                backward_box, forward_box, strict=True
            )
        ]
        derivative = tube[0] * tube[1] - tube[2] - base.ZS
        derivative_has_sign = (
            derivative.upper() < 0
            if orientation < 0
            else derivative.lower() > 0
        )
        if not derivative_has_sign:
            diagnostics.append(f"2^-{power}:DERIVATIVE_SIGN_UNRESOLVED")
            continue
        predictor = -state[2] / derivative.mid()
        radius_ball = base.rational_ball(radius)
        predictor_range = predictor.range()
        if (
            predictor_range.lower() <= -radius_ball
            or predictor_range.upper() >= radius_ball
        ):
            diagnostics.append(f"2^-{power}:PREDICTOR_ESCAPED")
            continue
        predicted_state = transport.chain.variable_time_flow(state, predictor, tube)
        correction = -predicted_state[2].range() / derivative
        event_time_model = predictor.with_remainder(correction)
        event_time_range = event_time_model.range()
        if (
            event_time_range.lower() > -radius_ball
            and event_time_range.upper() < radius_ball
        ):
            return radius, tube, derivative, event_time_model, correction
        diagnostics.append(f"2^-{power}:NEWTON_ESCAPED")
    raise base.Refusal(
        "EVENT_SLAB_UNRESOLVED",
        "extended local Picard/Newton slab search failed; " + ",".join(diagnostics),
    )


def run_adaptive_tile(
    tile_id: str,
    tile_state: list[base.TM2R],
    domain: SymbolicDomain,
) -> tuple[
    transport.event.PositiveReturn,
    transport.chain.DownwardReturnPhase,
    list[transport.TaggedProjection],
    list[transport.chain.SectionProjection],
    list[transport.chain.UpwardReturn],
    int,
    int,
    int,
]:
    first = transport.event.integrate_positive_return(tile_state)
    first_projection = transport.event.interval_newton_project(first)
    print(
        f"first-event-projection tile={tile_id} end_step={first.end_step}",
        file=sys.stderr,
        flush=True,
    )
    approach = transport.chain.integrate_downward_return(first_projection.carrier)

    pending = [transport.TaggedBranch(approach.endpoint, 0, domain, ())]
    final_carriers: list[transport.TaggedProjection] = []
    downward_events: list[transport.chain.SectionProjection] = []
    upward_events: list[transport.chain.UpwardReturn] = []
    downward_split_nodes = 0
    downward_split_reconstructions = 0
    stabilization_checks = 0
    while pending:
        branch = pending.pop()
        try:
            downward = transport.chain.project_downward_event(
                branch.state, approach.reference_time
            )
        except base.Refusal as refusal:
            if (
                branch.depth >= transport.chain.MAX_EVENT_SPLIT_DEPTH
                or downward_split_nodes
                >= transport.chain.MAX_EVENT_SPLIT_NODES_PER_TILE
            ):
                raise base.Refusal(
                    "DOWNWARD_EVENT_COVER_UNRESOLVED",
                    f"branch={branch.path} refusal={refusal.failure_class}",
                ) from refusal
            variable, _weight = adaptive.dominant_variable(branch.state)
            left, right, checks = adaptive.split_state(branch.state, variable)
            left_domain, right_domain = split_domain_pair(branch.domain, variable)
            downward_split_nodes += 1
            downward_split_reconstructions += checks
            child_depth = branch.depth + 1
            name = adaptive.VARIABLE_NAMES[variable]
            pending.extend(
                (
                    transport.TaggedBranch(
                        left,
                        child_depth,
                        left_domain,
                        branch.path + ("DOWN_" + name + "L",),
                    ),
                    transport.TaggedBranch(
                        right,
                        child_depth,
                        right_domain,
                        branch.path + ("DOWN_" + name + "H",),
                    ),
                )
            )
            print(
                f"downward-event-split tile={tile_id} depth={child_depth} "
                f"variable={name} reason={refusal.failure_class}",
                file=sys.stderr,
                flush=True,
            )
            continue

        upward_initial, checks = transport.chain.outward_stabilize_carrier(
            downward.carrier
        )
        stabilization_checks += checks
        tagged: list[transport.TaggedProjection] = []
        original_cover = transport.chain.project_upward_cover

        def capture(
            state: list[base.TM2R], reference_time: Fraction
        ) -> tuple[list[transport.chain.SectionProjection], int, int]:
            nonlocal tagged
            tagged, nodes, reconstructions = transport.tagged_project_upward_cover(
                state, reference_time, branch.domain
            )
            return [item.projection for item in tagged], nodes, reconstructions

        transport.chain.project_upward_cover = capture
        try:
            try:
                upward = transport.chain.seek_upward_return(upward_initial)
            except base.Refusal as refusal:
                if (
                    branch.depth >= transport.chain.MAX_EVENT_SPLIT_DEPTH
                    or downward_split_nodes
                    >= transport.chain.MAX_EVENT_SPLIT_NODES_PER_TILE
                ):
                    raise base.Refusal(
                        "UPWARD_EVENT_COVER_UNRESOLVED",
                        f"branch={branch.path} refusal={refusal.failure_class}",
                    ) from refusal
                variable, _weight = adaptive.dominant_variable(upward_initial)
                left, right, checks = adaptive.split_state(branch.state, variable)
                left_domain, right_domain = split_domain_pair(
                    branch.domain, variable
                )
                downward_split_nodes += 1
                downward_split_reconstructions += checks
                child_depth = branch.depth + 1
                name = adaptive.VARIABLE_NAMES[variable]
                pending.extend(
                    (
                        transport.TaggedBranch(
                            left,
                            child_depth,
                            left_domain,
                            branch.path + ("UP_" + name + "L",),
                        ),
                        transport.TaggedBranch(
                            right,
                            child_depth,
                            right_domain,
                            branch.path + ("UP_" + name + "H",),
                        ),
                    )
                )
                print(
                    f"upward-event-split tile={tile_id} depth={child_depth} "
                    f"variable={name} reason={refusal.failure_class}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
        finally:
            transport.chain.project_upward_cover = original_cover
        if len(tagged) != len(upward.carriers):
            raise base.Refusal(
                "TAGGED_CARRIER_COUNT_MISMATCH",
                f"tagged={len(tagged)} upward={len(upward.carriers)}",
            )
        for item in tagged:
            final_carriers.append(
                transport.TaggedProjection(
                    item.projection,
                    item.domain,
                    branch.path + item.path,
                )
            )
        downward_events.append(downward)
        upward_events.append(upward)
        print(
            f"second-event-projection tile={tile_id} "
            f"downward_path={'_'.join(branch.path) or 'ROOT'} "
            f"carriers={len(tagged)}",
            file=sys.stderr,
            flush=True,
        )

    return (
        first,
        approach,
        final_carriers,
        downward_events,
        upward_events,
        downward_split_nodes,
        downward_split_reconstructions,
        stabilization_checks,
    )


def main() -> None:
    tile_id = os.environ.get("CS6_SOURCE_TILE", "")
    if tile_id not in EXPECTED_TILES:
        raise SystemExit(
            "CS6_SOURCE_TILE must be exactly one of " + ",".join(EXPECTED_TILES)
        )

    base.SOURCE_DEGREE = 2
    base.TIME_TAYLOR_ORDER = 12
    base.recondition = adaptive.point_coefficient_recondition
    transport.event.MAX_PHASE_STEPS = MAX_FIRST_RETURN_STEPS
    transport.chain.find_event_slab = find_event_slab_extended
    tiles, source_split_checks = source_tiles()
    tile_state, domain = tiles[tile_id]
    (
        first,
        approach,
        tagged,
        downward_events,
        upward_events,
        downward_split_nodes,
        downward_split_reconstructions,
        stabilization_checks,
    ) = run_adaptive_tile(tile_id, tile_state, domain)
    domain_cover_checks = certify_terminal_domain_cover(
        domain, [item.domain for item in tagged]
    )

    carriers = []
    for index, item in enumerate(tagged):
        suffix = "_".join(item.path) if item.path else "ROOT"
        carriers.append(
            {
                "carrier_id": f"{tile_id}:{index}:{suffix}",
                "source_domain": item.domain.as_json(),
                "event_split_path": list(item.path),
                "components": [
                    transport.tm2r_json(component)
                    for component in item.projection.carrier
                ],
                "event_time": transport.interval_json(item.projection.event_time),
                "event_derivative": transport.interval_json(item.projection.derivative),
                "event_normal": transport.interval_json(item.projection.normal),
            }
        )

    source_path = Path(__file__)
    payload = {
        "schema": SCHEMA,
        "worker_source_sha256": sha256(source_path),
        "transport_source_sha256": sha256(Path(transport.__file__)),
        "event_chain_source_sha256": sha256(Path(transport.chain.__file__)),
        "adaptive_source_sha256": sha256(Path(adaptive.__file__)),
        "event_projection_source_sha256": sha256(Path(transport.event.__file__)),
        "base_source_sha256": sha256(Path(base.__file__)),
        "python_version": platform.python_version(),
        "python_flint_version": flint.__version__,
        "arb_precision_bits": base.PRECISION_BITS,
        "source_hset": "B",
        "map": "P^2",
        "max_first_return_steps": MAX_FIRST_RETURN_STEPS,
        "max_event_slab_radius": str(MAX_EVENT_SLAB_RADIUS),
        "source_chart": {
            "unstable_row": [str(ROW_U_X), str(ROW_U_Y)],
            "stable_row": [str(-ROW_U_Y), str(ROW_U_X)],
            "determinant": str(ROW_U_X * ROW_U_X + ROW_U_Y * ROW_U_Y),
            "unstable_center": str(U_CENTER),
            "unstable_radius": str(U_RADIUS),
            "stable_center": str(S_CENTER),
            "stable_radius": str(S_RADIUS),
        },
        "tile_id": tile_id,
        "source_domain": domain.as_json(),
        "symbolic_variables": list(adaptive.VARIABLE_NAMES),
        "terminal_domain_cover_certified": True,
        "terminal_domain_cover_checks": domain_cover_checks,
        "source_split_reconstructions": source_split_checks,
        "first_return_end_step": first.end_step,
        "first_event_projected": True,
        "downward_event_time": transport.interval_json(
            hull([item.event_time for item in downward_events])
        ),
        "downward_event_normal": transport.interval_json(
            hull([item.normal for item in downward_events])
        ),
        "downward_event_split_nodes": downward_split_nodes,
        "downward_event_split_reconstructions": downward_split_reconstructions,
        "second_event_time": transport.interval_json(
            hull([item.event_time for item in upward_events])
        ),
        "second_event_derivative": transport.interval_json(
            hull([item.derivative for item in upward_events])
        ),
        "second_event_normal": transport.interval_json(
            hull([item.normal for item in upward_events])
        ),
        "second_event_split_nodes": sum(
            item.split_nodes for item in upward_events
        ),
        "second_event_split_reconstructions": sum(
            item.split_reconstructions for item in upward_events
        ),
        "outward_stabilization_checks": stabilization_checks,
        "selected_source_chain_certificate": True,
        "point_fallback_used": False,
        "box_flattening_used": False,
        "deterministic_no_rng": True,
        "poincare_section": "w=0, strictly positive return orientation",
        "time_step": "1/256",
        "event_slab_acceptance": (
            "signed Picard closure; strict derivative sign; predictor strictly "
            "inside; interval-Newton event time strictly inside"
        ),
        "carriers": carriers,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
