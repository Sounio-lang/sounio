#!/usr/bin/env python3
"""Adaptive validated TM2R subdivision after the first-return projection."""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from pathlib import Path

from flint import arb

import cs6_v7b_target23_arb_tm2r_second_return_worker as event


base = event.base
MAX_PHASE_STEPS = 2000
MAX_SPLIT_DEPTH = 8
MAX_SPLIT_NODES = 63
MAX_TIME_REFINEMENT_DEPTH = 8
VARIABLE_NAMES = ("XI", "ETA", "RHO0", "RHO1", "RHO2", "RHO3")


@dataclass
class PhaseContext:
    elapsed_time: Fraction = Fraction(0)
    accepted_substeps: int = 0
    seen_strict_negative: bool = False
    initial_departure_tubes: int = 0
    prior_downward_tubes: int = 0
    zero_free_prior_tubes: int = 0

    def clone(self) -> "PhaseContext":
        return PhaseContext(
            elapsed_time=self.elapsed_time,
            accepted_substeps=self.accepted_substeps,
            seen_strict_negative=self.seen_strict_negative,
            initial_departure_tubes=self.initial_departure_tubes,
            prior_downward_tubes=self.prior_downward_tubes,
            zero_free_prior_tubes=self.zero_free_prior_tubes,
        )


@dataclass
class BranchFailure:
    state: list[base.TM2R]
    context: PhaseContext
    failure_class: str
    failure_detail: str


@dataclass
class BranchReturn:
    context: PhaseContext
    before_w: arb
    after_w: arb
    derivative: arb
    normal: arb
    time_lower: Fraction
    time_upper: Fraction


@dataclass
class Branch:
    branch_id: str
    state: list[base.TM2R]
    context: PhaseContext
    depth: int


@dataclass
class IntersectionStatistics:
    strict_endpoint_containments: int = 0
    intersection_only_endpoints: int = 0
    endpoint_intersection_components: int = 0
    taylor_picard_tube_intersection_components: int = 0
    time_bisections: int = 0
    minimum_step_power: int = -8
    orientation_refinement_attempts: int = 0
    point_coefficient_reconditionings: int = 0
    coefficient_uncertainty_generators: int = 0


INTERSECTION_STATS = IntersectionStatistics()


def point_coefficient_recondition(state: list[base.TM2R]) -> list[base.TM2R]:
    """Compress a carrier after moving every Arb coefficient radius to generators."""
    base.STATS.reconditionings += 1
    INTERSECTION_STATS.point_coefficient_reconditionings += 1
    source_coefficients: list[dict[tuple[int, ...], arb]] = [
        {} for _ in range(4)
    ]
    residual_monomials: set[tuple[int, ...]] = set()
    generators: list[list[arb]] = []

    for row, component in enumerate(state):
        for monomial, coefficient in component.coefficients.items():
            midpoint = coefficient.mid()
            radius = coefficient.rad()
            if any(monomial[base.SOURCE_VARIABLES :]):
                residual_monomials.add(monomial)
            else:
                source_coefficients[row][monomial] = midpoint
            if radius.upper() > 0:
                generator = [arb(0) for _ in range(4)]
                generator[row] = radius
                generators.append(generator)
                INTERSECTION_STATS.coefficient_uncertainty_generators += 1

    for monomial in sorted(residual_monomials):
        generator = [
            component.coefficients.get(monomial, arb(0)).mid()
            for component in state
        ]
        if not base.vector_nonzero(generator):
            continue
        if all(exponent % 2 == 0 for exponent in monomial):
            half = [value / 2 for value in generator]
            for row in range(4):
                source_coefficients[row][base.ZERO_MONOMIAL] = (
                    source_coefficients[row].get(base.ZERO_MONOMIAL, arb(0))
                    + half[row]
                )
            generators.append(half)
        else:
            generators.append(generator)

    for row, component in enumerate(state):
        midpoint = component.remainder.mid()
        radius = component.remainder.rad()
        source_coefficients[row][base.ZERO_MONOMIAL] = (
            source_coefficients[row].get(base.ZERO_MONOMIAL, arb(0)) + midpoint
        )
        if radius.upper() > 0:
            generator = [arb(0) for _ in range(4)]
            generator[row] = radius
            generators.append(generator)

    basis, inverse = base.qr_derived_basis(generators)
    radii = [arb(0) for _ in range(4)]
    for generator in generators:
        coordinates = [
            sum(
                (
                    base.rational_ball(inverse[coordinate][row]) * generator[row]
                    for row in range(4)
                ),
                arb(0),
            )
            for coordinate in range(4)
        ]
        for coordinate, projected in enumerate(coordinates):
            radii[coordinate] += base.upper_abs(projected)
        reconstructed = [
            sum(
                (
                    base.rational_ball(basis[row][coordinate])
                    * coordinates[coordinate]
                    for coordinate in range(4)
                ),
                arb(0),
            )
            for row in range(4)
        ]
        if not all(
            enclosure.contains(component)
            for enclosure, component in zip(
                reconstructed, generator, strict=True
            )
        ):
            raise base.Refusal(
                "GENERATOR_RECONSTRUCTION_FAILED",
                "point-coefficient Q-times-Q-inverse failed to enclose a generator",
            )
        base.STATS.generator_reconstructions += 1

    result: list[base.TM2R] = []
    for row in range(4):
        coefficients = dict(source_coefficients[row])
        for coordinate in range(4):
            monomial = [0] * base.VARIABLES
            monomial[base.SOURCE_VARIABLES + coordinate] = 1
            coefficients[tuple(monomial)] = (
                base.rational_ball(basis[row][coordinate]) * radii[coordinate]
            )
        result.append(base.TM2R(coefficients, arb(0)))

    conditioned_ranges = [component.range() for component in result]
    base.STATS.max_reconditioned_width = base.max_upper(
        [
            base.STATS.max_reconditioned_width,
            *[base.width(value) for value in conditioned_ranges],
        ]
    )
    return result


def reparameterize_component(
    component: base.TM2R,
    variable: int,
    center: Fraction,
    radius: Fraction,
) -> base.TM2R:
    coefficients: dict[tuple[int, ...], arb] = {}
    center_ball = base.rational_ball(center)
    radius_ball = base.rational_ball(radius)
    for monomial, coefficient in component.coefficients.items():
        exponent = monomial[variable]
        for retained_exponent in range(exponent + 1):
            transformed = list(monomial)
            transformed[variable] = retained_exponent
            scale = (
                comb(exponent, retained_exponent)
                * center_ball ** (exponent - retained_exponent)
                * radius_ball ** retained_exponent
            )
            key = tuple(transformed)
            coefficients[key] = coefficients.get(key, arb(0)) + coefficient * scale
    return base.TM2R(coefficients, component.remainder)


def split_state(
    state: list[base.TM2R], variable: int
) -> tuple[list[base.TM2R], list[base.TM2R], int]:
    left = [
        reparameterize_component(component, variable, Fraction(-1, 2), Fraction(1, 2))
        for component in state
    ]
    right = [
        reparameterize_component(component, variable, Fraction(1, 2), Fraction(1, 2))
        for component in state
    ]
    inverse_left = [
        reparameterize_component(component, variable, Fraction(1), Fraction(2))
        for component in left
    ]
    inverse_right = [
        reparameterize_component(component, variable, Fraction(-1), Fraction(2))
        for component in right
    ]
    checks = 0
    for parent, recovered_left, recovered_right in zip(
        state, inverse_left, inverse_right, strict=True
    ):
        for recovered in (recovered_left, recovered_right):
            if not all(
                recovered.coefficients.get(monomial, arb(0)).contains(coefficient)
                for monomial, coefficient in parent.coefficients.items()
            ) or not recovered.remainder.contains(parent.remainder):
                raise base.Refusal(
                    "SPLIT_ALGEBRAIC_RECONSTRUCTION_FAILED",
                    "inverse child reparameterization did not enclose its parent TM",
                )
            checks += 1
    return left, right, checks


def variable_weight(state: list[base.TM2R], variable: int) -> arb:
    weight = arb(0)
    for component in state:
        for monomial, coefficient in component.coefficients.items():
            if monomial[variable]:
                weight += base.upper_abs(coefficient) * monomial[variable]
    return weight


def dominant_variable(state: list[base.TM2R]) -> tuple[int, arb]:
    weights = [variable_weight(state, variable) for variable in range(base.VARIABLES)]
    winner = 0
    for variable in range(1, base.VARIABLES):
        if weights[variable].upper() > weights[winner].upper():
            winner = variable
    if weights[winner].upper() <= 0:
        raise base.Refusal("NO_SPLIT_DIRECTION", "all normalized carrier directions vanished")
    return winner, weights[winner]


def advance_with_endpoint_intersection(
    initial: list[base.TM2R], step: arb
) -> tuple[list[base.TM2R], list[arb]]:
    """Advance when Taylor endpoint and Picard tube merely intersect.

    The Picard self-map encloses every true trajectory over the step. The
    Taylor-Lagrange construction independently encloses its endpoint. Strict
    containment of the latter enclosure in the former tube is useful but not
    required by either argument; disjointness would contradict soundness and
    is therefore refused.
    """
    base.STATS.attempted_steps += 1
    initial_range = [component.range() for component in initial]
    box = base.picard_box(initial_range, step)
    coefficients = base.tm_flow_coefficients(initial, base.TIME_TAYLOR_ORDER)
    polynomial = [
        sum(
            (
                coefficients[row][power] * step**power
                for power in range(base.TIME_TAYLOR_ORDER + 1)
            ),
            base.TM2R.constant(0),
        )
        for row in range(4)
    ]
    remainder_coefficients = base.interval_flow_coefficients(
        box, base.TIME_TAYLOR_ORDER + 1
    )
    time_interval = arb(step / 2, step / 2)
    refined_tube: list[arb] = []
    for row in range(4):
        temporal_polynomial = sum(
            (
                coefficients[row][power] * time_interval**power
                for power in range(base.TIME_TAYLOR_ORDER + 1)
            ),
            base.TM2R.constant(0),
        )
        temporal_remainder = (
            base.upper_abs(
                remainder_coefficients[row][base.TIME_TAYLOR_ORDER + 1]
            )
            * step ** (base.TIME_TAYLOR_ORDER + 1)
        )
        temporal_range = temporal_polynomial.with_remainder(
            arb(0, temporal_remainder)
        ).range()
        if not temporal_range.overlaps(box[row]):
            raise base.Refusal(
                "TEMPORAL_TAYLOR_PICARD_TUBES_DISJOINT",
                "temporal Taylor tube was disjoint from its Picard tube",
            )
        refined_tube.append(temporal_range.intersection(box[row]))
        INTERSECTION_STATS.taylor_picard_tube_intersection_components += 1

    raw: list[base.TM2R] = []
    strict = True
    for row, component in enumerate(polynomial):
        time_remainder = (
            base.upper_abs(
                remainder_coefficients[row][base.TIME_TAYLOR_ORDER + 1]
            )
            * step ** (base.TIME_TAYLOR_ORDER + 1)
        )
        candidate = component.with_remainder(arb(0, time_remainder))
        candidate_range = candidate.range()
        if not candidate_range.overlaps(box[row]):
            raise base.Refusal(
                "ENDPOINT_ENCLOSURES_DISJOINT",
                "Taylor endpoint enclosure was disjoint from its Picard tube",
            )
        INTERSECTION_STATS.endpoint_intersection_components += 1
        if not box[row].contains(candidate_range):
            strict = False
        raw.append(candidate)
    if strict:
        INTERSECTION_STATS.strict_endpoint_containments += 1
        base.STATS.endpoint_picard_containments += 1
    else:
        INTERSECTION_STATS.intersection_only_endpoints += 1
    base.STATS.max_raw_width = base.max_upper(
        [base.STATS.max_raw_width, *[base.width(component.range()) for component in raw]]
    )
    result = base.recondition(raw)
    base.STATS.completed_steps += 1
    return result, refined_tube


def continue_branch(
    initial: list[base.TM2R], context: PhaseContext
) -> BranchFailure | BranchReturn:
    state = initial
    current = context.clone()
    pending_steps: list[tuple[Fraction, int]] = [(Fraction(1, 2**8), 0)]
    while current.elapsed_time < Fraction(MAX_PHASE_STEPS, 2**8):
        step_fraction, time_depth = pending_steps.pop()
        step = base.rational_ball(step_fraction)
        before_range = [component.range() for component in state]
        before_sign = base.strict_sign(before_range[2])
        try:
            next_state, tube = advance_with_endpoint_intersection(state, step)
        except base.Refusal as refusal:
            if (
                refusal.failure_class
                in {"PICARD_NO_CLOSURE", "PICARD_NONCONTRACTION"}
                and time_depth < MAX_TIME_REFINEMENT_DEPTH
            ):
                half = step_fraction / 2
                pending_steps.extend(((half, time_depth + 1), (half, time_depth + 1)))
                INTERSECTION_STATS.time_bisections += 1
                INTERSECTION_STATS.minimum_step_power = min(
                    INTERSECTION_STATS.minimum_step_power, -8 - time_depth - 1
                )
                print(
                    f"adaptive-time-bisect t={current.elapsed_time} "
                    f"step={step_fraction} depth={time_depth + 1}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            return BranchFailure(
                state=state,
                context=current,
                failure_class=refusal.failure_class,
                failure_detail=refusal.detail,
            )
        after_range = [component.range() for component in next_state]
        after_sign = base.strict_sign(after_range[2])
        if after_sign < 0:
            current.seen_strict_negative = True
        contains_section = tube[2].lower() <= 0 <= tube[2].upper()
        derivative = tube[0] * tube[1] - tube[2] - base.ZS
        if not contains_section:
            current.zero_free_prior_tubes += 1
        elif current.elapsed_time == 0:
            if before_sign != 0 or after_sign <= 0 or derivative.lower() <= 0:
                return BranchFailure(
                    state=state,
                    context=current,
                    failure_class="INITIAL_DEPARTURE_UNRESOLVED",
                    failure_detail="subdivided phase did not leave the section upward",
                )
            current.initial_departure_tubes += 1
        elif current.seen_strict_negative and before_sign < 0 and after_sign > 0:
            normal = tube[0] * tube[1] - base.ZS
            if normal.lower() <= 0 or derivative.lower() <= 0:
                return BranchFailure(
                    state=state,
                    context=current,
                    failure_class="EVENT_TRANSVERSALITY_UNRESOLVED",
                    failure_detail="subdivided target tube lacked positive transversality",
                )
            time_lower = current.elapsed_time
            current.elapsed_time += step_fraction
            current.accepted_substeps += 1
            return BranchReturn(
                context=current,
                before_w=before_range[2],
                after_w=after_range[2],
                derivative=derivative,
                normal=normal,
                time_lower=time_lower,
                time_upper=current.elapsed_time,
            )
        elif derivative.upper() < 0:
            current.prior_downward_tubes += 1
        else:
            if time_depth < MAX_TIME_REFINEMENT_DEPTH:
                half = step_fraction / 2
                pending_steps.extend(((half, time_depth + 1), (half, time_depth + 1)))
                INTERSECTION_STATS.time_bisections += 1
                INTERSECTION_STATS.orientation_refinement_attempts += 1
                INTERSECTION_STATS.minimum_step_power = min(
                    INTERSECTION_STATS.minimum_step_power, -8 - time_depth - 1
                )
                print(
                    f"adaptive-orientation-bisect t={current.elapsed_time} "
                    f"step={step_fraction} depth={time_depth + 1}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            return BranchFailure(
                state=state,
                context=current,
                failure_class="PRIOR_ORIENTATION_UNRESOLVED",
                failure_detail="subdivided prior section tube lacked downward orientation",
            )
        state = next_state
        current.elapsed_time += step_fraction
        current.accepted_substeps += 1
        if not pending_steps:
            pending_steps.append((Fraction(1, 2**8), 0))
    return BranchFailure(
        state=state,
        context=current,
        failure_class="EVENT_COUNT_UNRESOLVED",
        failure_detail=f"no positive return within {MAX_PHASE_STEPS}/256 post-event time",
    )


def min_lower(values: list[arb]) -> arb:
    result = values[0].lower()
    for value in values[1:]:
        if value.lower() < result:
            result = value.lower()
    return arb(result)


def main() -> None:
    base.SOURCE_DEGREE = 2
    base.TIME_TAYLOR_ORDER = 12
    base.recondition = point_coefficient_recondition
    initial, _u_interval, _s_interval = base.initial_leaf()
    first = event.integrate_positive_return(initial)
    projection = event.interval_newton_project(first)

    transcript: list[str] = []
    split_counts = [0 for _ in range(base.VARIABLES)]
    split_nodes = 0
    coverage_checks = 0
    max_depth = 0
    returns: list[BranchReturn] = []
    unresolved: list[tuple[str, BranchFailure]] = []

    first_continuation = continue_branch(projection.carrier, PhaseContext())
    if isinstance(first_continuation, BranchReturn):
        returns.append(first_continuation)
        transcript.append(f"RETURN:R:{first_continuation.time_upper}")
    else:
        print(
            f"adaptive-prefix refusal t={first_continuation.context.elapsed_time} "
            f"class={first_continuation.failure_class}",
            file=sys.stderr,
            flush=True,
        )
        pending = [
            Branch(
                branch_id="R",
                state=first_continuation.state,
                context=first_continuation.context,
                depth=0,
            )
        ]
        transcript.append(
            f"FAIL:R:{first_continuation.context.elapsed_time}:"
            f"{first_continuation.failure_class}"
        )
        while pending:
            branch = pending.pop()
            if branch.depth >= MAX_SPLIT_DEPTH or split_nodes >= MAX_SPLIT_NODES:
                unresolved.append(
                    (
                        branch.branch_id,
                        BranchFailure(
                            state=branch.state,
                            context=branch.context,
                            failure_class="SUBDIVISION_BUDGET_EXHAUSTED",
                            failure_detail="adaptive binary subdivision budget was exhausted",
                        ),
                    )
                )
                continue
            variable, weight = dominant_variable(branch.state)
            left, right, checks = split_state(branch.state, variable)
            coverage_checks += checks
            split_nodes += 1
            split_counts[variable] += 1
            child_depth = branch.depth + 1
            max_depth = max(max_depth, child_depth)
            transcript.append(
                f"SPLIT:{branch.branch_id}:{VARIABLE_NAMES[variable]}:"
                f"{base.upper_fraction(weight)}"
            )
            print(
                f"adaptive-split branch={branch.branch_id} t={branch.context.elapsed_time} "
                f"depth={child_depth} variable={VARIABLE_NAMES[variable]}",
                file=sys.stderr,
                flush=True,
            )
            child_records: list[Branch] = []
            for suffix, child_state in (("L", left), ("H", right)):
                child_id = branch.branch_id + suffix
                result = continue_branch(child_state, branch.context)
                if isinstance(result, BranchReturn):
                    returns.append(result)
                    transcript.append(f"RETURN:{child_id}:{result.time_upper}")
                    print(
                        f"adaptive-return branch={child_id} "
                        f"time={result.time_upper}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    transcript.append(
                        f"FAIL:{child_id}:{result.context.elapsed_time}:"
                        f"{result.failure_class}"
                    )
                    child_records.append(
                        Branch(child_id, result.state, result.context, child_depth)
                    )
                    print(
                        f"adaptive-refusal branch={child_id} "
                        f"time={result.context.elapsed_time} "
                        f"class={result.failure_class}",
                        file=sys.stderr,
                        flush=True,
                    )
            pending.extend(reversed(child_records))

    full_certificate = bool(returns) and not unresolved
    source_path = Path(__file__)
    event_path = Path(event.__file__)
    base_path = Path(base.__file__)
    print("SCHEMA=sounio.cs6.v7b-target23-arb-tm2r-subdivided-second-return-worker.v1")
    print(f"WORKER_SOURCE_SHA256={hashlib.sha256(source_path.read_bytes()).hexdigest()}")
    print(f"EVENT_PROJECTION_DEPENDENCY_SHA256={hashlib.sha256(event_path.read_bytes()).hexdigest()}")
    print(f"TM2R_DEPENDENCY_SHA256={hashlib.sha256(base_path.read_bytes()).hexdigest()}")
    print(f"TRANSCRIPT_SHA256={hashlib.sha256(chr(10).join(transcript).encode('ascii')).hexdigest()}")
    print(f"PYTHON_VERSION={platform.python_version()}")
    print(f"PYTHON_FLINT_VERSION={base.flint.__version__}")
    print(f"LEAF_ID={base.LEAF_ID}")
    print("ARB_PRECISION_BITS=256")
    print("ARB_THREADS=1")
    print("SOURCE_DEGREE=2")
    print("SOURCE_VARIABLES=2")
    print("RESIDUAL_VARIABLES=4")
    print("TIME_TAYLOR_ORDER=12")
    print("TIME_STEP_POWER=-8")
    print("SUBDIVISION=ADAPTIVE_BINARY_DOMINANT_NORMALIZED_TM_DIRECTION_AT_REFUSAL")
    print(f"MAX_SPLIT_DEPTH={MAX_SPLIT_DEPTH}")
    print(f"MAX_SPLIT_NODES={MAX_SPLIT_NODES}")
    print(f"MAX_TIME_REFINEMENT_DEPTH={MAX_TIME_REFINEMENT_DEPTH}")
    print(f"FIRST_RETURN_END_STEP={first.end_step}")
    print("PROJECTED_W_EXACTLY_ZERO=true")
    print(f"PURE_SOURCE_MONOMIALS_RETAINED={projection.pure_source_monomials_retained}")
    print(f"SPLIT_NODES={split_nodes}")
    print(f"SPLIT_ALGEBRAIC_RECONSTRUCTIONS={coverage_checks}")
    print(f"MAX_REACHED_SPLIT_DEPTH={max_depth}")
    for variable, name in enumerate(VARIABLE_NAMES):
        print(f"SPLITS_{name}={split_counts[variable]}")
    print(f"TERMINAL_RETURN_LEAVES={len(returns)}")
    print(f"UNRESOLVED_LEAVES={len(unresolved)}")
    print(f"TIME_BISECTIONS={INTERSECTION_STATS.time_bisections}")
    print(f"ORIENTATION_REFINEMENT_ATTEMPTS={INTERSECTION_STATS.orientation_refinement_attempts}")
    print(f"MINIMUM_TIME_STEP_POWER={INTERSECTION_STATS.minimum_step_power}")
    if returns:
        lower_times = [result.time_lower for result in returns]
        upper_times = [result.time_upper for result in returns]
        substeps = [result.context.accepted_substeps for result in returns]
        print(f"SECOND_RETURN_MIN_ACCEPTED_SUBSTEPS={min(substeps)}")
        print(f"SECOND_RETURN_MAX_ACCEPTED_SUBSTEPS={max(substeps)}")
        print(f"SECOND_RETURN_TIME_LOWER_Q={min(lower_times)}")
        print(f"SECOND_RETURN_TIME_UPPER_Q={max(upper_times)}")
        print(
            "SECOND_RETURN_DERIVATIVE_MIN_LOWER_Q="
            f"{base.lower_fraction(min_lower([result.derivative for result in returns]))}"
        )
        print(
            "SECOND_RETURN_NORMAL_MIN_LOWER_Q="
            f"{base.lower_fraction(min_lower([result.normal for result in returns]))}"
        )
        print(
            "SECOND_RETURN_INITIAL_DEPARTURE_MIN="
            f"{min(result.context.initial_departure_tubes for result in returns)}"
        )
        print(
            "SECOND_RETURN_PRIOR_DOWNWARD_MIN="
            f"{min(result.context.prior_downward_tubes for result in returns)}"
        )
    else:
        print("SECOND_RETURN_MIN_ACCEPTED_SUBSTEPS=-1")
        print("SECOND_RETURN_MAX_ACCEPTED_SUBSTEPS=-1")
    if unresolved:
        print(f"FIRST_UNRESOLVED_BRANCH={unresolved[0][0]}")
        print(f"FIRST_UNRESOLVED_FAILURE_CLASS={unresolved[0][1].failure_class}")
        print(f"FIRST_UNRESOLVED_TIME_Q={unresolved[0][1].context.elapsed_time}")
    else:
        print("FIRST_UNRESOLVED_BRANCH=NONE")
        print("FIRST_UNRESOLVED_FAILURE_CLASS=NONE")
        print("FIRST_UNRESOLVED_TIME_Q=-1")
    print(f"TOTAL_ATTEMPTED_STEPS={base.STATS.attempted_steps}")
    print(f"TOTAL_COMPLETED_STEPS={base.STATS.completed_steps}")
    print(f"TOTAL_PICARD_CONTAINMENTS={base.STATS.picard_containments}")
    print(f"TOTAL_ENDPOINT_PICARD_CONTAINMENTS={base.STATS.endpoint_picard_containments}")
    print(f"SECOND_PHASE_STRICT_ENDPOINT_CONTAINMENTS={INTERSECTION_STATS.strict_endpoint_containments}")
    print(f"SECOND_PHASE_INTERSECTION_ONLY_ENDPOINTS={INTERSECTION_STATS.intersection_only_endpoints}")
    print(f"SECOND_PHASE_ENDPOINT_INTERSECTION_COMPONENTS={INTERSECTION_STATS.endpoint_intersection_components}")
    print(f"SECOND_PHASE_TAYLOR_PICARD_TUBE_INTERSECTION_COMPONENTS={INTERSECTION_STATS.taylor_picard_tube_intersection_components}")
    print(f"TOTAL_RECONDITIONINGS={base.STATS.reconditionings}")
    print(f"TOTAL_GENERATOR_RECONSTRUCTIONS={base.STATS.generator_reconstructions}")
    print("BOUNDED_METHOD_RESULT=true")
    print("FULL_LEAF_FIRST_RETURN_CERTIFICATE=true")
    print("INTERVAL_NEWTON_EVENT_PROJECTION_CERTIFICATE=true")
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
