#!/usr/bin/env python3
"""Rigorous up/down/up TM2R event-chain experiment for CS6 leaf 331."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from flint import arb

import cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker as adaptive


event = adaptive.event
base = adaptive.base
MAX_TIME_REFINEMENT_DEPTH = 10
MAX_UPWARD_TIME = Fraction(4)
MAX_EVENT_SPLIT_DEPTH = 8
MAX_EVENT_SPLIT_NODES_PER_TILE = 63


@dataclass
class SectionProjection:
    carrier: list[base.TM2R]
    event_time: arb
    delta: arb
    derivative: arb
    normal: arb
    reference_time: Fraction
    slab_radius: Fraction
    event_time_pure_source_monomials: int
    pure_source_monomials: int
    projected_width: arb


@dataclass
class UpwardReturn:
    carriers: list[list[base.TM2R]]
    event_time: arb
    time_lower: Fraction
    time_upper: Fraction
    derivative: arb
    normal: arb
    initial_negative_departure_tubes: int
    zero_free_prior_tubes: int
    accepted_substeps: int
    time_bisections: int
    minimum_step_power: int
    pure_source_monomials: int
    projected_width: arb
    projected_leaves: int
    split_nodes: int
    split_reconstructions: int


@dataclass
class DownwardReturnPhase:
    endpoint: list[base.TM2R]
    event_tube: list[arb]
    reference_time: Fraction
    derivative: arb
    downward_section_tubes: int
    accepted_substeps: int
    time_bisections: int


@dataclass
class EventBranch:
    branch_id: str
    root_id: str
    state: list[base.TM2R]
    depth: int
    reference_time: Fraction


@dataclass
class EventChainResult:
    branch_id: str
    downward: SectionProjection
    upward: UpwardReturn


def integrate_downward_return(initial: list[base.TM2R]) -> DownwardReturnPhase:
    state = initial
    elapsed = Fraction(0)
    accepted = 0
    bisections = 0
    seen_strict_positive = False
    downward_section_tubes = 0
    pending: list[tuple[Fraction, int]] = [(Fraction(1, 2**8), 0)]
    while elapsed < Fraction(4):
        step_fraction, depth = pending.pop()
        before = [component.range() for component in state]
        before_sign = base.strict_sign(before[2])
        try:
            next_state, tube = adaptive.advance_with_endpoint_intersection(
                state, base.rational_ball(step_fraction)
            )
        except base.Refusal as refusal:
            if (
                refusal.failure_class
                in {"PICARD_NO_CLOSURE", "PICARD_NONCONTRACTION"}
                and depth < MAX_TIME_REFINEMENT_DEPTH
            ):
                half = step_fraction / 2
                pending.extend(((half, depth + 1), (half, depth + 1)))
                bisections += 1
                continue
            raise
        after = [component.range() for component in next_state]
        after_sign = base.strict_sign(after[2])
        if after_sign > 0:
            seen_strict_positive = True
        contains_section = tube[2].lower() <= 0 <= tube[2].upper()
        derivative = tube[0] * tube[1] - tube[2] - base.ZS
        if elapsed == 0:
            if before_sign != 0 or after_sign <= 0 or derivative.lower() <= 0:
                raise base.Refusal(
                    "INITIAL_DEPARTURE_UNRESOLVED",
                    "first projected carrier did not leave upward strictly",
                )
        elif contains_section:
            if derivative.upper() < 0:
                downward_section_tubes += 1
            elif depth < MAX_TIME_REFINEMENT_DEPTH:
                half = step_fraction / 2
                pending.extend(((half, depth + 1), (half, depth + 1)))
                bisections += 1
                continue
            else:
                raise base.Refusal(
                    "DOWNWARD_ORIENTATION_UNRESOLVED",
                    "a downward section tube lacked strictly negative derivative",
                )
        if seen_strict_positive and downward_section_tubes and after_sign < 0:
            return DownwardReturnPhase(
                endpoint=next_state,
                event_tube=tube,
                reference_time=elapsed + step_fraction,
                derivative=derivative,
                downward_section_tubes=downward_section_tubes,
                accepted_substeps=accepted + 1,
                time_bisections=bisections,
            )
        state = next_state
        elapsed += step_fraction
        accepted += 1
        if not pending:
            pending.append((Fraction(1, 2**8), 0))
    raise base.Refusal(
        "DOWNWARD_EVENT_COUNT_UNRESOLVED",
        "no strictly downward section passage within four time units",
    )


def find_event_slab(
    state: list[base.TM2R], orientation: int
) -> tuple[Fraction, list[arb], arb, base.TM2R, arb]:
    if orientation not in (-1, 1):
        raise ValueError("event orientation must be -1 or 1")
    initial_ranges = [component.range() for component in state]
    diagnostics: list[str] = []
    for power in range(18, 7, -1):
        radius = Fraction(1, 2**power)
        try:
            backward_box, _iterations, _contraction = event.signed_picard_box(
                initial_ranges, base.rational_ball(-radius)
            )
            forward_box, _iterations, _contraction = event.signed_picard_box(
                initial_ranges, base.rational_ball(radius)
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
            diagnostics.append(f"2^-{power}:DERIVATIVE_NOT_NEGATIVE")
            continue
        derivative_center = derivative.mid()
        predictor = -state[2] / derivative_center
        radius_ball = base.rational_ball(radius)
        predictor_range = predictor.range()
        if (
            predictor_range.lower() <= -radius_ball
            or predictor_range.upper() >= radius_ball
        ):
            diagnostics.append(
                f"2^-{power}:PREDICTOR_ESCAPED:"
                f"{base.lower_fraction(predictor_range)}:"
                f"{base.upper_fraction(predictor_range)}"
            )
            continue
        predicted_state = variable_time_flow(state, predictor, tube)
        correction = -predicted_state[2].range() / derivative
        event_time_model = predictor.with_remainder(correction)
        event_time_range = event_time_model.range()
        if (
            event_time_range.lower() > -radius_ball
            and event_time_range.upper() < radius_ball
        ):
            return radius, tube, derivative, event_time_model, correction
        diagnostics.append(
            f"2^-{power}:NEWTON_ESCAPED:"
            f"{base.lower_fraction(event_time_range)}:"
            f"{base.upper_fraction(event_time_range)}"
        )
    raise base.Refusal(
        "DOWNWARD_EVENT_SLAB_UNRESOLVED",
        "no local Picard slab strictly contained its TM2 parametric Newton image; "
        + ",".join(diagnostics),
    )


def variable_time_flow(
    initial: list[base.TM2R], event_time: base.TM2R, tube: list[arb]
) -> list[base.TM2R]:
    coefficients = base.tm_flow_coefficients(initial, base.TIME_TAYLOR_ORDER)
    powers = [base.TM2R.constant(1)]
    for _power in range(base.TIME_TAYLOR_ORDER):
        powers.append(powers[-1] * event_time)
    polynomial = [
        sum(
            (
                coefficients[row][power] * powers[power]
                for power in range(base.TIME_TAYLOR_ORDER + 1)
            ),
            base.TM2R.constant(0),
        )
        for row in range(4)
    ]
    remainder_coefficients = base.interval_flow_coefficients(
        tube, base.TIME_TAYLOR_ORDER + 1
    )
    time_radius = base.upper_abs(event_time.range())
    return [
        component.with_remainder(
            arb(
                0,
                base.upper_abs(
                    remainder_coefficients[row][base.TIME_TAYLOR_ORDER + 1]
                )
                * time_radius ** (base.TIME_TAYLOR_ORDER + 1),
            )
        )
        for row, component in enumerate(polynomial)
    ]


def project_section_event(
    state: list[base.TM2R], reference_time: Fraction, orientation: int
) -> SectionProjection:
    radius, tube, derivative, event_time_model, _correction = find_event_slab(
        state, orientation
    )

    event_state = variable_time_flow(state, event_time_model, tube)
    raw_projection = [
        component if row != 2 else base.TM2R.constant(0)
        for row, component in enumerate(event_state)
    ]
    projected = base.recondition(raw_projection)
    projected_ranges = [component.range() for component in projected]
    if projected_ranges[2].lower() != 0 or projected_ranges[2].upper() != 0:
        raise base.Refusal(
            "DOWNWARD_SECTION_PROJECTION_DRIFT",
            "downward projected carrier did not have exact w=0",
        )
    projected_normal = projected_ranges[0] * projected_ranges[1] - base.ZS
    normal_has_sign = (
        projected_normal.upper() < 0
        if orientation < 0
        else projected_normal.lower() > 0
    )
    if not normal_has_sign:
        raise base.Refusal(
            "PROJECTED_TRANSVERSALITY_UNRESOLVED",
            "projected section normal did not have the requested strict sign",
        )
    retained = sum(
        1
        for row in (0, 1, 3)
        for monomial, coefficient in projected[row].coefficients.items()
        if event.pure_source_nonconstant(monomial)
        and (coefficient.lower() != 0 or coefficient.upper() != 0)
    )
    if retained == 0:
        raise base.Refusal(
            "DOWNWARD_SOURCE_DEPENDENCE_LOST",
            "downward event projection retained no pure source monomial",
        )
    event_time_range = event_time_model.range()
    event_time = base.rational_ball(reference_time) + event_time_range
    event_time_retained = sum(
        1
        for monomial, coefficient in event_time_model.coefficients.items()
        if event.pure_source_nonconstant(monomial)
        and (coefficient.lower() != 0 or coefficient.upper() != 0)
    )
    return SectionProjection(
        carrier=projected,
        event_time=event_time,
        delta=event_time_range,
        derivative=derivative,
        normal=projected_normal,
        reference_time=reference_time,
        slab_radius=radius,
        event_time_pure_source_monomials=event_time_retained,
        pure_source_monomials=retained,
        projected_width=base.max_upper(
            [base.width(component) for component in projected_ranges]
        ),
    )


def project_downward_event(
    state: list[base.TM2R], reference_time: Fraction
) -> SectionProjection:
    return project_section_event(state, reference_time, -1)


def project_upward_event(
    state: list[base.TM2R], reference_time: Fraction
) -> SectionProjection:
    return project_section_event(state, reference_time, 1)


def project_upward_cover(
    state: list[base.TM2R], reference_time: Fraction
) -> tuple[list[SectionProjection], int, int]:
    pending: list[tuple[list[base.TM2R], int]] = [(state, 0)]
    projections: list[SectionProjection] = []
    unresolved: list[str] = []
    split_nodes = 0
    split_reconstructions = 0
    while pending:
        branch_state, depth = pending.pop()
        derivative_model = (
            branch_state[0] * branch_state[1]
            - branch_state[2]
            - base.ZS
        )
        derivative_range = derivative_model.range()
        cheap_candidate = False
        refusal_class = "UPWARD_PREFILTER_UNRESOLVED"
        if derivative_range.lower() > 0:
            correction = -branch_state[2].range() / derivative_range
            radius = base.rational_ball(Fraction(1, 2**8))
            cheap_candidate = (
                correction.lower() > -radius and correction.upper() < radius
            )
        if cheap_candidate:
            try:
                projections.append(
                    project_upward_event(branch_state, reference_time)
                )
                continue
            except base.Refusal as refusal:
                refusal_class = refusal.failure_class
        if depth >= MAX_EVENT_SPLIT_DEPTH or split_nodes >= 255:
            unresolved.append(refusal_class)
            continue
        variable, _weight = adaptive.dominant_variable(
            [branch_state[2], derivative_model]
        )
        left, right, checks = adaptive.split_state(branch_state, variable)
        split_nodes += 1
        split_reconstructions += checks
        pending.extend(((left, depth + 1), (right, depth + 1)))
    if unresolved:
        raise base.Refusal(
            "SECOND_EVENT_COVER_UNRESOLVED",
            f"upward Newton cover left {len(unresolved)} unresolved leaves; "
            f"first={unresolved[0]}; split_nodes={split_nodes}",
        )
    return projections, split_nodes, split_reconstructions


def assemble_upward_return(
    projections: list[SectionProjection],
    initial_departures: int,
    zero_free: int,
    accepted: int,
    bisections: int,
    minimum_power: int,
    split_nodes: int,
    split_reconstructions: int,
) -> UpwardReturn:
    event_time = hull([projection.event_time for projection in projections])
    return UpwardReturn(
        carriers=[projection.carrier for projection in projections],
        event_time=event_time,
        time_lower=Fraction(base.lower_fraction(event_time)),
        time_upper=Fraction(base.upper_fraction(event_time)),
        derivative=hull([projection.derivative for projection in projections]),
        normal=hull([projection.normal for projection in projections]),
        initial_negative_departure_tubes=initial_departures,
        zero_free_prior_tubes=zero_free,
        accepted_substeps=accepted,
        time_bisections=bisections,
        minimum_step_power=minimum_power,
        pure_source_monomials=min(
            projection.pure_source_monomials for projection in projections
        ),
        projected_width=base.max_upper(
            [projection.projected_width for projection in projections]
        ),
        projected_leaves=len(projections),
        split_nodes=split_nodes,
        split_reconstructions=split_reconstructions,
    )


def seek_upward_return(initial: list[base.TM2R]) -> UpwardReturn:
    state = initial
    elapsed = Fraction(0)
    accepted = 0
    bisections = 0
    minimum_power = -8
    initial_departures = 0
    zero_free = 0
    seen_strict_negative = False
    pending: list[tuple[Fraction, int]] = [(Fraction(1, 2**8), 0)]
    while elapsed < MAX_UPWARD_TIME:
        step_fraction, depth = pending.pop()
        before = [component.range() for component in state]
        before_sign = base.strict_sign(before[2])
        try:
            next_state, tube = adaptive.advance_with_endpoint_intersection(
                state, base.rational_ball(step_fraction)
            )
        except base.Refusal as refusal:
            if (
                refusal.failure_class
                in {"PICARD_NO_CLOSURE", "PICARD_NONCONTRACTION"}
                and depth < MAX_TIME_REFINEMENT_DEPTH
            ):
                half = step_fraction / 2
                pending.extend(((half, depth + 1), (half, depth + 1)))
                bisections += 1
                minimum_power = min(minimum_power, -8 - depth - 1)
                continue
            raise
        after = [component.range() for component in next_state]
        after_sign = base.strict_sign(after[2])
        if after_sign < 0:
            seen_strict_negative = True
        contains_section = tube[2].lower() <= 0 <= tube[2].upper()
        derivative = tube[0] * tube[1] - tube[2] - base.ZS
        if not contains_section:
            zero_free += 1
        elif elapsed == 0:
            if before_sign == 0 and after_sign < 0 and derivative.upper() < 0:
                initial_departures += 1
            elif depth < MAX_TIME_REFINEMENT_DEPTH:
                half = step_fraction / 2
                pending.extend(((half, depth + 1), (half, depth + 1)))
                bisections += 1
                minimum_power = min(minimum_power, -8 - depth - 1)
                continue
            else:
                raise base.Refusal(
                    "DOWNWARD_DEPARTURE_UNRESOLVED",
                    "projected carrier did not leave the downward section strictly",
                )
        elif seen_strict_negative and before_sign < 0 and after_sign > 0:
            projections, split_nodes, split_reconstructions = project_upward_cover(
                next_state, elapsed + step_fraction
            )
            return assemble_upward_return(
                projections,
                initial_departures,
                zero_free,
                accepted + 1,
                bisections,
                minimum_power,
                split_nodes,
                split_reconstructions,
            )
        else:
            if seen_strict_negative:
                candidates = [
                    (state, elapsed),
                    (next_state, elapsed + step_fraction),
                ]
                candidates.sort(
                    key=lambda item: float(base.width(item[0][2].range()).upper())
                )
                for candidate_state, reference_time in candidates:
                    try:
                        projections, split_nodes, split_reconstructions = (
                            project_upward_cover(
                                candidate_state, reference_time
                            )
                        )
                    except base.Refusal:
                        continue
                    return assemble_upward_return(
                        projections,
                        initial_departures,
                        zero_free,
                        accepted + 1,
                        bisections,
                        minimum_power,
                        split_nodes,
                        split_reconstructions,
                    )
            if depth < MAX_TIME_REFINEMENT_DEPTH:
                half = step_fraction / 2
                pending.extend(((half, depth + 1), (half, depth + 1)))
                bisections += 1
                minimum_power = min(minimum_power, -8 - depth - 1)
                continue
            raise base.Refusal(
                "SECOND_PRIOR_ORIENTATION_UNRESOLVED",
                "a pre-target tube remained section-ambiguous",
            )
        state = next_state
        elapsed += step_fraction
        accepted += 1
        if not pending:
            pending.append((Fraction(1, 2**8), 0))
    raise base.Refusal(
        "SECOND_EVENT_COUNT_UNRESOLVED",
        f"no upward return within {MAX_UPWARD_TIME} time after downward projection; "
        f"final_w={state[2].range()}",
    )


def emit_interval(prefix: str, value: arb) -> None:
    print(f"{prefix}_LOWER_Q={base.lower_fraction(value)}")
    print(f"{prefix}_UPPER_Q={base.upper_fraction(value)}")


def hull(values: list[arb]) -> arb:
    result = values[0]
    for value in values[1:]:
        result = result.union(value)
    return result


def outward_stabilize_carrier(
    state: list[base.TM2R],
) -> tuple[list[base.TM2R], int]:
    checks = 0

    def stabilize(value: arb) -> arb:
        nonlocal checks
        lower = Fraction(base.lower_fraction(value))
        upper = Fraction(base.upper_fraction(value))
        result = arb(
            base.rational_ball((lower + upper) / 2),
            base.rational_ball((upper - lower) / 2),
        )
        if not result.contains(value):
            raise base.Refusal(
                "OUTWARD_STABILIZATION_FAILED",
                "rational endpoint reconstruction did not contain its Arb input",
            )
        checks += 1
        return result

    return [
        base.TM2R(
            {
                monomial: stabilize(coefficient)
                for monomial, coefficient in component.coefficients.items()
            },
            stabilize(component.remainder),
        )
        for component in state
    ], checks


def main() -> None:
    base.SOURCE_DEGREE = 2
    base.TIME_TAYLOR_ORDER = 12
    base.recondition = adaptive.point_coefficient_recondition
    initial, _u, _s = base.initial_leaf()
    xi_left, xi_right, xi_checks = adaptive.split_state(initial, 0)
    source_tiles: list[tuple[str, list[base.TM2R]]] = []
    source_split_reconstructions = xi_checks
    for xi_id, xi_state in (("XL", xi_left), ("XH", xi_right)):
        eta_left, eta_right, eta_checks = adaptive.split_state(xi_state, 1)
        source_split_reconstructions += eta_checks
        source_tiles.extend(
            ((xi_id + "EL", eta_left), (xi_id + "EH", eta_right))
        )
    source_tile_filter = os.environ.get("CS6_SOURCE_TILE")
    if source_tile_filter:
        source_tiles = [
            item for item in source_tiles if item[0] == source_tile_filter
        ]
        if not source_tiles:
            raise ValueError(f"unknown CS6_SOURCE_TILE={source_tile_filter}")

    pending: list[EventBranch] = []
    results: list[EventChainResult] = []
    unresolved: list[tuple[str, str]] = []
    first_return_end_steps: list[int] = []
    first_projected_tiles = 0
    for tile_id, tile_state in source_tiles:
        try:
            first = event.integrate_positive_return(tile_state)
            first_projection = event.interval_newton_project(first)
        except base.Refusal as refusal:
            unresolved.append((tile_id, refusal.failure_class))
            print(
                f"first-event-refusal tile={tile_id} "
                f"class={refusal.failure_class}",
                file=sys.stderr,
                flush=True,
            )
            continue
        first_return_end_steps.append(first.end_step)
        first_projected_tiles += 1
        print(
            f"first-event-projection tile={tile_id} end_step={first.end_step}",
            file=sys.stderr,
            flush=True,
        )
        try:
            approach = integrate_downward_return(first_projection.carrier)
        except base.Refusal as refusal:
            unresolved.append((tile_id, refusal.failure_class))
            print(
                f"downward-approach-refusal tile={tile_id} "
                f"class={refusal.failure_class}",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(
            f"downward-window tile={tile_id} t={approach.reference_time} "
            f"section_tubes={approach.downward_section_tubes}",
            file=sys.stderr,
            flush=True,
        )
        pending.append(
            EventBranch(
                branch_id=tile_id,
                root_id=tile_id,
                state=approach.endpoint,
                depth=0,
                reference_time=approach.reference_time,
            )
        )

    split_nodes = 0
    split_nodes_by_tile = {tile_id: 0 for tile_id, _state in source_tiles}
    split_reconstructions = 0
    outward_stabilization_checks = 0
    split_counts = [0 for _ in range(base.VARIABLES)]
    while pending:
        branch = pending.pop()
        try:
            downward = project_downward_event(branch.state, branch.reference_time)
        except base.Refusal as refusal:
            if (
                branch.depth >= MAX_EVENT_SPLIT_DEPTH
                or split_nodes_by_tile[branch.root_id]
                >= MAX_EVENT_SPLIT_NODES_PER_TILE
            ):
                unresolved.append((branch.branch_id, refusal.failure_class))
                continue
            variable, _weight = adaptive.dominant_variable(branch.state)
            left, right, checks = adaptive.split_state(branch.state, variable)
            split_nodes += 1
            split_nodes_by_tile[branch.root_id] += 1
            split_reconstructions += checks
            split_counts[variable] += 1
            child_depth = branch.depth + 1
            pending.extend(
                reversed(
                    [
                        EventBranch(
                            branch.branch_id + "L",
                            branch.root_id,
                            left,
                            child_depth,
                            branch.reference_time,
                        ),
                        EventBranch(
                            branch.branch_id + "H",
                            branch.root_id,
                            right,
                            child_depth,
                            branch.reference_time,
                        ),
                    ]
                )
            )
            print(
                f"downward-event-split branch={branch.branch_id} "
                f"depth={child_depth} variable={adaptive.VARIABLE_NAMES[variable]} "
                f"reason={refusal.failure_class}",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(
            f"downward-projection branch={branch.branch_id} "
            f"radius={downward.slab_radius}",
            file=sys.stderr,
            flush=True,
        )
        upward_initial, stabilization_checks = outward_stabilize_carrier(
            downward.carrier
        )
        outward_stabilization_checks += stabilization_checks
        try:
            upward = seek_upward_return(upward_initial)
        except base.Refusal as refusal:
            unresolved.append((branch.branch_id, refusal.failure_class))
            continue
        results.append(EventChainResult(branch.branch_id, downward, upward))
        print(
            f"second-return branch={branch.branch_id} "
            f"time={upward.time_lower}:{upward.time_upper}",
            file=sys.stderr,
            flush=True,
        )

    selected_certificate = bool(results) and not unresolved
    full_certificate = selected_certificate and source_tile_filter is None
    downward_times = [result.downward.event_time for result in results]
    second_times = [
        result.downward.event_time + result.upward.event_time
        for result in results
    ]
    source_path = Path(__file__)
    adaptive_path = Path(adaptive.__file__)
    event_path = Path(event.__file__)
    print("SCHEMA=sounio.cs6.v7b-target23-arb-tm2r-event-chain-second-return-worker.v7")
    print(f"WORKER_SOURCE_SHA256={hashlib.sha256(source_path.read_bytes()).hexdigest()}")
    print(f"ADAPTIVE_DEPENDENCY_SHA256={hashlib.sha256(adaptive_path.read_bytes()).hexdigest()}")
    print(f"EVENT_PROJECTION_DEPENDENCY_SHA256={hashlib.sha256(event_path.read_bytes()).hexdigest()}")
    print(f"PYTHON_VERSION={platform.python_version()}")
    print(f"PYTHON_FLINT_VERSION={base.flint.__version__}")
    print(f"LEAF_ID={base.LEAF_ID}")
    print("ARB_PRECISION_BITS=256")
    print("SOURCE_DEGREE=2")
    print("SOURCE_VARIABLES=2")
    print("RESIDUAL_VARIABLES=4")
    print("TIME_TAYLOR_ORDER=12")
    print("RECONDITIONING=QR_POINT_COEFFICIENT_CARRIER")
    print("EVENT_CHAIN=UPWARD_PROJECT_DOWNWARD_PROJECT_UPWARD_RETURN")
    print(f"INITIAL_SOURCE_TILES={len(source_tiles)}")
    print(
        "INITIAL_SOURCE_SPLIT_RECONSTRUCTIONS="
        f"{source_split_reconstructions}"
    )
    print(f"SOURCE_TILE_FILTER={source_tile_filter or 'NONE'}")
    print(f"INITIAL_SOURCE_COVERAGE={str(source_tile_filter is None).lower()}")
    print(f"FIRST_EVENT_PROJECTED_TILES={first_projected_tiles}")
    if first_return_end_steps:
        print(f"FIRST_RETURN_END_STEP_MIN={min(first_return_end_steps)}")
        print(f"FIRST_RETURN_END_STEP_MAX={max(first_return_end_steps)}")
    else:
        print("FIRST_RETURN_END_STEP_MIN=NONE")
        print("FIRST_RETURN_END_STEP_MAX=NONE")
    print(f"DOWNWARD_EVENT_SPLIT_NODES={split_nodes}")
    for tile_id, count in split_nodes_by_tile.items():
        print(f"DOWNWARD_EVENT_SPLIT_NODES_{tile_id}={count}")
    print(f"DOWNWARD_EVENT_SPLIT_RECONSTRUCTIONS={split_reconstructions}")
    for variable, name in enumerate(adaptive.VARIABLE_NAMES):
        print(f"DOWNWARD_EVENT_SPLITS_{name}={split_counts[variable]}")
    print(f"DOWNWARD_PROJECTED_LEAVES={len(results)}")
    print(f"UNRESOLVED_EVENT_LEAVES={len(unresolved)}")
    if results:
        emit_interval("DOWNWARD_EVENT_TIME_HULL", hull(downward_times))
        emit_interval("FULL_SECOND_RETURN_TIME_HULL", hull(second_times))
        emit_interval(
            "SECOND_RETURN_DERIVATIVE_HULL",
            hull([result.upward.derivative for result in results]),
        )
        emit_interval(
            "SECOND_RETURN_NORMAL_HULL",
            hull([result.upward.normal for result in results]),
        )
        print(
            "DOWNWARD_PURE_SOURCE_MONOMIALS_MIN="
            f"{min(result.downward.pure_source_monomials for result in results)}"
        )
        print(
            "DOWNWARD_EVENT_TIME_PURE_SOURCE_MONOMIALS_MIN="
            f"{min(result.downward.event_time_pure_source_monomials for result in results)}"
        )
        print(
            "DOWNWARD_PROJECTED_CARRIER_MAX_WIDTH_UPPER_Q="
            f"{base.upper_fraction(base.max_upper([result.downward.projected_width for result in results]))}"
        )
        print("DOWNWARD_PROJECTED_W_EXACTLY_ZERO=true")
        print(
            "SECOND_EVENT_PURE_SOURCE_MONOMIALS_MIN="
            f"{min(result.upward.pure_source_monomials for result in results)}"
        )
        print(
            "SECOND_EVENT_PROJECTED_CARRIER_MAX_WIDTH_UPPER_Q="
            f"{base.upper_fraction(base.max_upper([result.upward.projected_width for result in results]))}"
        )
        print(
            "SECOND_EVENT_PROJECTED_LEAVES_TOTAL="
            f"{sum(result.upward.projected_leaves for result in results)}"
        )
        print(
            "SECOND_EVENT_SPLIT_NODES_TOTAL="
            f"{sum(result.upward.split_nodes for result in results)}"
        )
        print(
            "SECOND_EVENT_SPLIT_RECONSTRUCTIONS_TOTAL="
            f"{sum(result.upward.split_reconstructions for result in results)}"
        )
        print("SECOND_EVENT_PROJECTED_W_EXACTLY_ZERO=true")
        print(
            "DOWNWARD_INITIAL_DEPARTURE_TUBES_MIN="
            f"{min(result.upward.initial_negative_departure_tubes for result in results)}"
        )
        print(
            "UPWARD_ZERO_FREE_PRIOR_TUBES_MIN="
            f"{min(result.upward.zero_free_prior_tubes for result in results)}"
        )
        print(
            "UPWARD_ACCEPTED_SUBSTEPS_MAX="
            f"{max(result.upward.accepted_substeps for result in results)}"
        )
        print(
            "UPWARD_TIME_BISECTIONS_MAX="
            f"{max(result.upward.time_bisections for result in results)}"
        )
        print(
            "UPWARD_MINIMUM_TIME_STEP_POWER="
            f"{min(result.upward.minimum_step_power for result in results)}"
        )
    else:
        print("DOWNWARD_PROJECTED_W_EXACTLY_ZERO=false")
        print("SECOND_EVENT_PROJECTED_W_EXACTLY_ZERO=false")
    if unresolved:
        print(f"FIRST_UNRESOLVED_BRANCH={unresolved[0][0]}")
        print(f"FIRST_UNRESOLVED_FAILURE_CLASS={unresolved[0][1]}")
    else:
        print("FIRST_UNRESOLVED_BRANCH=NONE")
        print("FIRST_UNRESOLVED_FAILURE_CLASS=NONE")
    print(f"TOTAL_ATTEMPTED_STEPS={base.STATS.attempted_steps}")
    print(f"TOTAL_COMPLETED_STEPS={base.STATS.completed_steps}")
    print(f"TOTAL_PICARD_CONTAINMENTS={base.STATS.picard_containments}")
    print(f"TOTAL_RECONDITIONINGS={base.STATS.reconditionings}")
    print(
        "POINT_COEFFICIENT_RECONDITIONINGS="
        f"{adaptive.INTERSECTION_STATS.point_coefficient_reconditionings}"
    )
    print(
        "COEFFICIENT_UNCERTAINTY_GENERATORS="
        f"{adaptive.INTERSECTION_STATS.coefficient_uncertainty_generators}"
    )
    print(f"TOTAL_GENERATOR_RECONSTRUCTIONS={base.STATS.generator_reconstructions}")
    print(f"OUTWARD_STABILIZATION_CHECKS={outward_stabilization_checks}")
    print("BOUNDED_METHOD_RESULT=true")
    selected_first_projection = first_projected_tiles == len(source_tiles)
    full_first_projection = selected_first_projection and source_tile_filter is None
    print(
        "SELECTED_SOURCE_FIRST_EVENT_PROJECTION_CERTIFICATE="
        f"{str(selected_first_projection).lower()}"
    )
    print(f"SELECTED_SOURCE_CHAIN_CERTIFICATE={str(selected_certificate).lower()}")
    print(f"FULL_LEAF_FIRST_RETURN_CERTIFICATE={str(full_first_projection).lower()}")
    print(
        "FIRST_INTERVAL_NEWTON_EVENT_PROJECTION_CERTIFICATE="
        f"{str(full_first_projection).lower()}"
    )
    print(f"FULL_LEAF_DOWNWARD_EVENT_PROJECTION_CERTIFICATE={str(full_certificate).lower()}")
    print(f"FULL_LEAF_SECOND_RETURN_CERTIFICATE={str(full_certificate).lower()}")
    print("RETURN_MAP_DETERMINANT_CERTIFICATE=false")
    print("COVERING_RELATION_CERTIFICATE=false")
    print("GLOBAL_HPG_CERTIFICATE=false")
    print("V7_B_ELIGIBILITY=false")
    print("CHAOS_PROVED=false")
    print("CHAOTIC_ATTRACTOR_PROVED=false")
    print("OPEN_PROBLEM_SOLVED=false")
    print("NOVELTY_OR_PRIORITY_CLAIMED=false")
    print("CAPD_USED_BY_WORKER=false")
    print("POINT_FALLBACK_USED=false")
    print("FPGA_EXECUTION=false")


if __name__ == "__main__":
    main()
