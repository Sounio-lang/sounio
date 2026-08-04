#!/usr/bin/env python3
"""Arb TM2R interval-Newton event projection and second-return experiment."""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from flint import arb

import cs6_v7b_target23_arb_tm2r_first_return_worker as base


MAX_PHASE_STEPS = 2000
FIRST_PHASE_TIME_TAYLOR_ORDER = 12
SECOND_PHASE_TIME_TAYLOR_ORDER = 12
FIRST_PHASE_SOURCE_DEGREE = 2
SECOND_PHASE_SOURCE_DEGREE = 2


@dataclass
class ReturnPhase:
    endpoint: list[base.TM2R]
    event_tube: list[arb]
    before_w: arb
    after_w: arb
    w_derivative: arb
    normal: arb
    end_step: int
    initial_departure_tubes: int
    prior_downward_tubes: int
    zero_free_prior_tubes: int


def integrate_positive_return(initial: list[base.TM2R]) -> ReturnPhase:
    state = initial
    seen_strict_negative = False
    initial_departure_tubes = 0
    prior_downward_tubes = 0
    zero_free_prior_tubes = 0
    for step_index in range(MAX_PHASE_STEPS):
        before_range = [component.range() for component in state]
        before_sign = base.strict_sign(before_range[2])
        next_state, tube = base.advance(state, base.STEP)
        after_range = [component.range() for component in next_state]
        after_sign = base.strict_sign(after_range[2])
        if after_sign < 0:
            seen_strict_negative = True
        contains_section = tube[2].lower() <= 0 <= tube[2].upper()
        derivative = tube[0] * tube[1] - tube[2] - base.ZS
        if not contains_section:
            zero_free_prior_tubes += 1
        elif step_index == 0:
            if before_sign != 0 or after_sign <= 0 or derivative.lower() <= 0:
                raise base.Refusal(
                    "INITIAL_DEPARTURE_UNRESOLVED",
                    "phase did not leave its initial section strictly upward",
                )
            initial_departure_tubes += 1
        elif seen_strict_negative and before_sign < 0 and after_sign > 0:
            normal = tube[0] * tube[1] - base.ZS
            if normal.lower() <= 0 or derivative.lower() <= 0:
                raise base.Refusal(
                    "EVENT_TRANSVERSALITY_UNRESOLVED",
                    "target tube lacked a strictly positive w derivative",
                )
            return ReturnPhase(
                endpoint=next_state,
                event_tube=tube,
                before_w=before_range[2],
                after_w=after_range[2],
                w_derivative=derivative,
                normal=normal,
                end_step=step_index + 1,
                initial_departure_tubes=initial_departure_tubes,
                prior_downward_tubes=prior_downward_tubes,
                zero_free_prior_tubes=zero_free_prior_tubes,
            )
        elif derivative.upper() < 0:
            prior_downward_tubes += 1
        else:
            raise base.Refusal(
                "PRIOR_ORIENTATION_UNRESOLVED",
                "a prior section-touching tube lacked strict downward orientation",
            )
        state = next_state
    raise base.Refusal(
        "EVENT_COUNT_UNRESOLVED",
        f"no positive return was validated within {MAX_PHASE_STEPS} phase steps",
    )


def pure_source_nonconstant(monomial: tuple[int, ...]) -> bool:
    return (
        any(monomial[:base.SOURCE_VARIABLES])
        and not any(monomial[base.SOURCE_VARIABLES:])
    )


@dataclass
class Projection:
    carrier: list[base.TM2R]
    delta: arb
    fixed_time_shift: arb
    residual_time_shift: arb
    derivative: arb
    pure_source_monomials_retained: int
    projected_width: arb
    picard_iterations: int
    picard_contraction: arb
    slab_picard_iterations: int
    slab_picard_contraction: arb


def signed_picard_box(initial: list[arb], step: arb) -> tuple[list[arb], int, arb]:
    absolute_step = abs(step)
    time_interval = arb(step / 2, absolute_step / 2)
    box = [
        base.inflate(component.union(component + time_interval * derivative))
        for component, derivative in zip(
            initial, base.field_interval(initial), strict=True
        )
    ]
    for iteration in range(1, 51):
        image = [
            component + time_interval * derivative
            for component, derivative in zip(
                initial, base.field_interval(box), strict=True
            )
        ]
        if all(
            container.contains(candidate)
            for container, candidate in zip(box, image, strict=True)
        ):
            contraction = base.lipschitz_bound(box) * absolute_step
            if contraction.upper() >= 1:
                raise base.Refusal(
                    "PROJECTION_PICARD_NONCONTRACTION",
                    "fixed-time backward projection was not contractive",
                )
            return box, iteration, contraction
        box = [
            base.inflate(container.union(candidate))
            for container, candidate in zip(box, image, strict=True)
        ]
    raise base.Refusal(
        "PROJECTION_PICARD_NO_CLOSURE",
        "fixed-time backward projection Picard tube did not close",
    )


def fixed_time_flow(
    initial: list[base.TM2R], step: arb
) -> tuple[list[base.TM2R], int, arb]:
    initial_range = [component.range() for component in initial]
    box, iterations, contraction = signed_picard_box(initial_range, step)
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
    raw: list[base.TM2R] = []
    for row, component in enumerate(polynomial):
        remainder = (
            base.upper_abs(
                remainder_coefficients[row][base.TIME_TAYLOR_ORDER + 1]
            )
            * abs(step) ** (base.TIME_TAYLOR_ORDER + 1)
        )
        candidate = component.with_remainder(arb(0, remainder))
        if not box[row].contains(candidate.range()):
            raise base.Refusal(
                "PROJECTION_ENDPOINT_ESCAPES_PICARD",
                "fixed-time backward TM endpoint escaped its Picard tube",
            )
        raw.append(candidate)
    return base.recondition(raw), iterations, contraction


def interval_newton_project(phase: ReturnPhase) -> Projection:
    end_ranges = [component.range() for component in phase.endpoint]
    w_end = end_ranges[2]
    derivative = (
        phase.event_tube[0] * phase.event_tube[1]
        - phase.event_tube[2]
        - base.ZS
    )
    if w_end.lower() <= 0 or derivative.lower() <= 0:
        raise base.Refusal(
            "NEWTON_DENOMINATOR_UNRESOLVED",
            "event endpoint or Newton denominator was not strictly positive",
        )
    delta = -w_end / derivative
    if delta.lower() < -base.STEP or delta.upper() > 0:
        raise base.Refusal(
            "NEWTON_TIME_SLAB_ESCAPED",
            "interval-Newton time correction escaped the validated event step",
        )

    # Re-enclose the whole backward time slab from the endpoint carrier and
    # require its position tube to lie in the already validated event tube.
    full_slab_step = arb(delta.lower())
    slab_box, slab_iterations, slab_contraction = signed_picard_box(
        end_ranges, full_slab_step
    )
    if not all(
        event_component.contains(slab_component)
        for event_component, slab_component in zip(
            phase.event_tube, slab_box, strict=True
        )
    ):
        raise base.Refusal(
            "PROJECTION_SLAB_ESCAPES_EVENT_TUBE",
            "backward event-time position slab escaped the validated event tube",
        )

    fixed_shift = delta.mid()
    residual_shift = delta - fixed_shift
    fixed_state, picard_iterations, picard_contraction = fixed_time_flow(
        phase.endpoint, fixed_shift
    )
    vector_field = base.field_interval(phase.event_tube)
    raw_projection: list[base.TM2R] = []
    for row, component in enumerate(fixed_state):
        if row == 2:
            raw_projection.append(base.TM2R.constant(0))
        else:
            raw_projection.append(
                component.with_remainder(vector_field[row] * residual_shift)
            )
    projected = base.recondition(raw_projection)

    retained = 0
    for row in (0, 1, 3):
        retained += sum(
            1
            for monomial, coefficient in projected[row].coefficients.items()
            if pure_source_nonconstant(monomial)
            and (coefficient.lower() != 0 or coefficient.upper() != 0)
        )
    if retained == 0:
        raise base.Refusal(
            "SOURCE_DEPENDENCE_LOST",
            "event projection retained no pure nonconstant source monomial",
        )
    projected_ranges = [component.range() for component in projected]
    if projected_ranges[2].lower() != 0 or projected_ranges[2].upper() != 0:
        raise base.Refusal(
            "SECTION_PROJECTION_DRIFT",
            "projected carrier did not have exact w=0",
        )
    return Projection(
        carrier=projected,
        delta=delta,
        fixed_time_shift=fixed_shift,
        residual_time_shift=residual_shift,
        derivative=derivative,
        pure_source_monomials_retained=retained,
        projected_width=base.max_upper(
            [base.width(component) for component in projected_ranges]
        ),
        picard_iterations=picard_iterations,
        picard_contraction=picard_contraction,
        slab_picard_iterations=slab_iterations,
        slab_picard_contraction=slab_contraction,
    )


def emit_interval(prefix: str, value: arb) -> None:
    print(f"{prefix}_LOWER_Q={base.lower_fraction(value)}")
    print(f"{prefix}_UPPER_Q={base.upper_fraction(value)}")


def main() -> None:
    failure_class = "NONE"
    failure_detail = "NONE"
    failure_phase = "NONE"
    active_phase = "FIRST_RETURN"
    first: ReturnPhase | None = None
    projection: Projection | None = None
    second: ReturnPhase | None = None
    initial, u_interval, s_interval = base.initial_leaf()
    initial_ranges = [component.range() for component in initial]
    try:
        base.SOURCE_DEGREE = FIRST_PHASE_SOURCE_DEGREE
        base.TIME_TAYLOR_ORDER = FIRST_PHASE_TIME_TAYLOR_ORDER
        first = integrate_positive_return(initial)
        active_phase = "EVENT_PROJECTION"
        projection = interval_newton_project(first)
        active_phase = "SECOND_RETURN"
        base.SOURCE_DEGREE = SECOND_PHASE_SOURCE_DEGREE
        base.TIME_TAYLOR_ORDER = SECOND_PHASE_TIME_TAYLOR_ORDER
        second = integrate_positive_return(projection.carrier)
    except base.Refusal as refusal:
        failure_phase = active_phase
        failure_class = refusal.failure_class
        failure_detail = refusal.detail.replace(" ", "_")

    first_certificate = first is not None
    projection_certificate = projection is not None
    second_certificate = second is not None and failure_class == "NONE"

    source_path = Path(__file__)
    dependency_path = Path(base.__file__)
    print("SCHEMA=sounio.cs6.v7b-target23-arb-tm2r-second-return-worker.v1")
    print(f"WORKER_SOURCE_SHA256={hashlib.sha256(source_path.read_bytes()).hexdigest()}")
    print(f"FIRST_RETURN_DEPENDENCY_SHA256={hashlib.sha256(dependency_path.read_bytes()).hexdigest()}")
    print(f"PYTHON_VERSION={platform.python_version()}")
    print(f"PYTHON_FLINT_VERSION={base.flint.__version__}")
    print(f"LEAF_ID={base.LEAF_ID}")
    print("ARB_PRECISION_BITS=256")
    print("ARB_THREADS=1")
    print("SOURCE_VARIABLES=2")
    print("RESIDUAL_VARIABLES=4")
    print(f"FIRST_PHASE_TIME_TAYLOR_ORDER={FIRST_PHASE_TIME_TAYLOR_ORDER}")
    print(f"SECOND_PHASE_TIME_TAYLOR_ORDER={SECOND_PHASE_TIME_TAYLOR_ORDER}")
    print(f"FIRST_PHASE_SOURCE_DEGREE={FIRST_PHASE_SOURCE_DEGREE}")
    print(f"SECOND_PHASE_SOURCE_DEGREE={SECOND_PHASE_SOURCE_DEGREE}")
    print("TIME_STEP_POWER=-8")
    print("EVENT_PROJECTION=INTERVAL_NEWTON_ENDPOINT_SLAB_WITH_PURE_SOURCE_COEFFICIENT_RETENTION")
    emit_interval("LEAF_U", u_interval)
    emit_interval("LEAF_S", s_interval)
    emit_interval("INITIAL_X", initial_ranges[0])
    emit_interval("INITIAL_Y", initial_ranges[1])
    print(f"FAILURE_CLASS={failure_class}")
    print(f"FAILURE_DETAIL={failure_detail}")
    print(f"FAILURE_PHASE={failure_phase}")
    print(f"TOTAL_ATTEMPTED_STEPS={base.STATS.attempted_steps}")
    print(f"TOTAL_COMPLETED_STEPS={base.STATS.completed_steps}")
    print(f"TOTAL_PICARD_CONTAINMENTS={base.STATS.picard_containments}")
    print(f"TOTAL_ENDPOINT_PICARD_CONTAINMENTS={base.STATS.endpoint_picard_containments}")
    print(f"TOTAL_RECONDITIONINGS={base.STATS.reconditionings}")
    print(f"TOTAL_GENERATOR_RECONSTRUCTIONS={base.STATS.generator_reconstructions}")

    if first is not None:
        print(f"FIRST_RETURN_END_STEP={first.end_step}")
        print(f"FIRST_INITIAL_DEPARTURE_TUBES={first.initial_departure_tubes}")
        print(f"FIRST_PRIOR_DOWNWARD_TUBES={first.prior_downward_tubes}")
        print(f"FIRST_ZERO_FREE_PRIOR_TUBES={first.zero_free_prior_tubes}")
        emit_interval("FIRST_RETURN_W_BEFORE", first.before_w)
        emit_interval("FIRST_RETURN_W_AFTER", first.after_w)
        emit_interval("FIRST_RETURN_W_DERIVATIVE", first.w_derivative)
    else:
        print("FIRST_RETURN_END_STEP=-1")

    if projection is not None:
        emit_interval("NEWTON_TIME_CORRECTION", projection.delta)
        emit_interval("NEWTON_FIXED_TIME_SHIFT", projection.fixed_time_shift)
        emit_interval("NEWTON_RESIDUAL_TIME_SHIFT", projection.residual_time_shift)
        emit_interval("NEWTON_DENOMINATOR", projection.derivative)
        print(f"PURE_SOURCE_MONOMIALS_RETAINED={projection.pure_source_monomials_retained}")
        print(f"PROJECTION_PICARD_ITERATIONS={projection.picard_iterations}")
        print(f"PROJECTION_PICARD_CONTRACTION_UPPER_Q={base.upper_fraction(projection.picard_contraction)}")
        print(f"PROJECTION_SLAB_PICARD_ITERATIONS={projection.slab_picard_iterations}")
        print(f"PROJECTION_SLAB_PICARD_CONTRACTION_UPPER_Q={base.upper_fraction(projection.slab_picard_contraction)}")
        print("PROJECTION_SLAB_CONTAINED_IN_EVENT_TUBE=true")
        print(f"PROJECTED_CARRIER_MAX_WIDTH_UPPER_Q={base.upper_fraction(projection.projected_width)}")
        print("PROJECTED_W_EXACTLY_ZERO=true")
    else:
        print("PURE_SOURCE_MONOMIALS_RETAINED=0")
        print("PROJECTION_SLAB_CONTAINED_IN_EVENT_TUBE=false")
        print("PROJECTED_W_EXACTLY_ZERO=false")

    if second is not None:
        print(f"SECOND_RETURN_ELAPSED_END_STEP={second.end_step}")
        print(f"SECOND_RETURN_ELAPSED_TIME_LOWER_Q={Fraction(second.end_step - 1, 2**8)}")
        print(f"SECOND_RETURN_ELAPSED_TIME_UPPER_Q={Fraction(second.end_step, 2**8)}")
        print(f"SECOND_INITIAL_DEPARTURE_TUBES={second.initial_departure_tubes}")
        print(f"SECOND_PRIOR_DOWNWARD_TUBES={second.prior_downward_tubes}")
        print(f"SECOND_ZERO_FREE_PRIOR_TUBES={second.zero_free_prior_tubes}")
        emit_interval("SECOND_RETURN_W_BEFORE", second.before_w)
        emit_interval("SECOND_RETURN_W_AFTER", second.after_w)
        emit_interval("SECOND_RETURN_W_DERIVATIVE", second.w_derivative)
        emit_interval("SECOND_RETURN_NORMAL", second.normal)
    else:
        print("SECOND_RETURN_ELAPSED_END_STEP=-1")

    first_steps = first.end_step if first is not None else 0
    second_attempted = max(0, base.STATS.attempted_steps - first_steps)
    second_completed = max(0, base.STATS.completed_steps - first_steps)
    print(f"SECOND_PHASE_ATTEMPTED_STEPS={second_attempted}")
    print(f"SECOND_PHASE_COMPLETED_STEPS={second_completed}")
    print(f"SECOND_PHASE_COMPLETED_TIME_Q={Fraction(second_completed, 2**8)}")

    print("BOUNDED_METHOD_RESULT=true")
    print(f"FULL_LEAF_FIRST_RETURN_CERTIFICATE={str(first_certificate).lower()}")
    print(f"INTERVAL_NEWTON_EVENT_PROJECTION_CERTIFICATE={str(projection_certificate).lower()}")
    print(f"FULL_LEAF_SECOND_RETURN_CERTIFICATE={str(second_certificate).lower()}")
    print("RETURN_MAP_DETERMINANT_CERTIFICATE=false")
    print("CAPD_USED_BY_WORKER=false")
    print("POINT_FALLBACK_USED=false")
    print("GLOBAL_HPG_CERTIFICATE=false")
    print("V7_B_ELIGIBILITY=false")
    print("CHAOS_PROVED=false")
    print("CHAOTIC_ATTRACTOR_PROVED=false")
    print("OPEN_PROBLEM_SOLVED=false")
    print("NOVELTY_OR_PRIORITY_CLAIMED=false")
    print("FPGA_EXECUTION=false")


if __name__ == "__main__":
    main()
