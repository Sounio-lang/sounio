#!/usr/bin/env python3
"""Transport one CS6 exit face with a target-covector-locked QR carrier."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from fractions import Fraction
from pathlib import Path

import flint
from flint import arb

import cs6_v7b_target23_arb_tm2r_first_return_worker as base
import cs6_v7b_target23_arb_tm2r_hset_covering_carrier_worker as support
import cs6_v7b_target23_arb_tm2r_hset_covering_face_worker as face_base
import cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker as adaptive


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-covector-qr-exit-face.v1"
TARGET_U_X = Fraction(-4644852547588741, 6250000000000000)
TARGET_U_Y = Fraction(13381910583555019, 20000000000000000)
TARGET_U_NORM_SQUARED = TARGET_U_X * TARGET_U_X + TARGET_U_Y * TARGET_U_Y
DYNAMIC_SEEDS = 0
DYNAMIC_TRANSPORTS = 0
SECOND_DIRECTION_SEEDS = 0
SECOND_DIRECTION_TRANSPORTS = 0
ACTIVE_QR_MODE = "DYNAMIC_TRIPLETON"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def covector_locked_basis(
    _generators: list[list[arb]],
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    """Return a rational basis with three columns in ker(TARGET_U)."""
    basis = [
        [TARGET_U_X, -TARGET_U_Y, Fraction(0), Fraction(0)],
        [TARGET_U_Y, TARGET_U_X, Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(0), Fraction(1)],
    ]
    inverse = base.fraction_inverse(basis)
    base.STATS.max_basis_inverse_row_sum = max(
        base.STATS.max_basis_inverse_row_sum,
        max(sum(abs(value) for value in row) for row in inverse),
    )
    if TARGET_U_X * basis[0][1] + TARGET_U_Y * basis[1][1] != 0:
        raise base.Refusal(
            "COVECTOR_KERNEL_BASIS_FAILED",
            "second QR column is not exactly in the target covector kernel",
        )
    return basis, inverse


def qr_basis_with_prefix(
    prefix: list[list[arb]], generators: list[list[arb]]
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    candidates = [
        [float(value.mid()) for value in generator] for generator in prefix
    ]
    remainder = [[float(value.mid()) for value in generator] for generator in generators]
    remainder = [
        vector for vector in remainder if all(math.isfinite(value) for value in vector)
    ]
    remainder.sort(key=lambda vector: sum(value * value for value in vector), reverse=True)
    candidates.extend(remainder)
    candidates.extend([[float(i == j) for i in range(4)] for j in range(4)])
    columns: list[list[float]] = []
    for candidate in candidates:
        vector = list(candidate)
        for column in columns:
            projection = sum(vector[i] * column[i] for i in range(4))
            vector = [vector[i] - projection * column[i] for i in range(4)]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 1e-12:
            columns.append([value / norm for value in vector])
        if len(columns) == 4:
            break
    if len(columns) != 4:
        raise base.Refusal(
            "DYNAMIC_QR_BASIS_INCOMPLETE",
            "could not complete the privileged residual direction",
        )
    basis = [
        [Fraction(format(columns[column][row], ".17g")) for column in range(4)]
        for row in range(4)
    ]
    inverse = base.fraction_inverse(basis)
    base.STATS.max_basis_inverse_row_sum = max(
        base.STATS.max_basis_inverse_row_sum,
        max(sum(abs(value) for value in row) for row in inverse),
    )
    return basis, inverse


def initial_privileged_generators(
    generators: list[list[arb]], count: int
) -> list[list[arb]]:
    def score(generator: list[arb]) -> float:
        x = float(generator[0].mid())
        y = float(generator[1].mid())
        return abs(float(TARGET_U_X) * x + float(TARGET_U_Y) * y)

    candidates = [generator for generator in generators if base.vector_nonzero(generator)]
    if not candidates:
        raise base.Refusal(
            "DYNAMIC_QR_NO_SEED",
            "no nonzero remainder generator was available for rho0",
        )
    first = max(candidates, key=score)
    if count == 1:
        return [first]
    first_vector = [float(value.mid()) for value in first]
    first_norm_squared = sum(value * value for value in first_vector)

    def transverse_score(generator: list[arb]) -> float:
        vector = [float(value.mid()) for value in generator]
        if first_norm_squared <= 0:
            return sum(value * value for value in vector)
        projection = sum(
            vector[index] * first_vector[index] for index in range(4)
        ) / first_norm_squared
        residual = [
            vector[index] - projection * first_vector[index]
            for index in range(4)
        ]
        return sum(value * value for value in residual)

    second = max(candidates, key=transverse_score)
    if transverse_score(second) <= 1e-30:
        return [first]
    return [first, second]


def dynamic_doubleton_recondition(state: list[base.TM2R]) -> list[base.TM2R]:
    """Transport one or two privileged rho directions and re-QR the complement."""
    global DYNAMIC_SEEDS, DYNAMIC_TRANSPORTS
    global SECOND_DIRECTION_SEEDS, SECOND_DIRECTION_TRANSPORTS
    base.STATS.reconditionings += 1
    adaptive.INTERSECTION_STATS.point_coefficient_reconditionings += 1
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
                adaptive.INTERSECTION_STATS.coefficient_uncertainty_generators += 1

    rho0_monomial = [0] * base.VARIABLES
    rho0_monomial[base.SOURCE_VARIABLES] = 1
    rho0_key = tuple(rho0_monomial)
    rho1_monomial = [0] * base.VARIABLES
    rho1_monomial[base.SOURCE_VARIABLES + 1] = 1
    rho1_key = tuple(rho1_monomial)
    privileged0: list[arb] | None = None
    privileged1: list[arb] | None = None
    for monomial in sorted(residual_monomials):
        generator = [
            component.coefficients.get(monomial, arb(0)).mid()
            for component in state
        ]
        if not base.vector_nonzero(generator):
            continue
        if monomial == rho0_key:
            privileged0 = generator
        elif monomial == rho1_key:
            privileged1 = generator
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

    privileged_count = 2 if ACTIVE_QR_MODE == "DYNAMIC_TRIPLETON" else 1
    if privileged0 is None or not base.vector_nonzero(privileged0):
        seeds = initial_privileged_generators(generators, privileged_count)
        privileged0 = seeds[0]
        privileged1 = seeds[1] if len(seeds) == 2 else None
        DYNAMIC_SEEDS += 1
        if privileged1 is not None:
            SECOND_DIRECTION_SEEDS += 1
    else:
        DYNAMIC_TRANSPORTS += 1
        if privileged_count == 2:
            if privileged1 is None or not base.vector_nonzero(privileged1):
                seeds = initial_privileged_generators(generators, 2)
                if len(seeds) == 2:
                    privileged1 = seeds[1]
                    SECOND_DIRECTION_SEEDS += 1
            else:
                SECOND_DIRECTION_TRANSPORTS += 1
    prefix = [privileged0]
    if privileged_count == 2 and privileged1 is not None:
        prefix.append(privileged1)
    basis, inverse = qr_basis_with_prefix(prefix, generators)

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
            for enclosure, component in zip(reconstructed, generator, strict=True)
        ):
            raise base.Refusal(
                "DYNAMIC_GENERATOR_RECONSTRUCTION_FAILED",
                "dynamic Q-times-Q-inverse failed to enclose a generator",
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


def main() -> None:
    face = os.environ.get("CS6_SOURCE_FACE", "")
    tile_id = os.environ.get("CS6_SOURCE_TILE", "")
    if face not in face_base.FACE_TILES:
        raise SystemExit("CS6_SOURCE_FACE must be LEFT or RIGHT")
    allowed_a, allowed_b, global_xi = face_base.FACE_TILES[face]
    if tile_id not in (allowed_a, allowed_b):
        raise SystemExit(f"tile {tile_id!r} does not meet source face {face}")
    eta_refinement = os.environ.get("CS6_FACE_ETA_REFINEMENT", "ROOT")
    if eta_refinement not in {"ROOT", "L", "H"}:
        raise SystemExit("CS6_FACE_ETA_REFINEMENT must be ROOT, L, or H")

    global ACTIVE_QR_MODE
    qr_mode = os.environ.get("CS6_QR_MODE", "DYNAMIC_TRIPLETON")
    if qr_mode not in {
        "DYNAMIC_DOUBLETON",
        "DYNAMIC_TRIPLETON",
        "FIXED_COVECTOR",
    }:
        raise SystemExit(
            "CS6_QR_MODE must be DYNAMIC_DOUBLETON, DYNAMIC_TRIPLETON, "
            "or FIXED_COVECTOR"
        )
    ACTIVE_QR_MODE = qr_mode
    base.SOURCE_DEGREE = 2
    base.TIME_TAYLOR_ORDER = 12
    if qr_mode == "FIXED_COVECTOR":
        base.qr_derived_basis = covector_locked_basis
        base.recondition = adaptive.point_coefficient_recondition
    else:
        base.recondition = dynamic_doubleton_recondition

    tiles, source_split_checks = support.source_tiles()
    tile_state, tile_domain = tiles[tile_id]
    local_xi = Fraction(-1) if face == "LEFT" else Fraction(1)
    face_state = [
        face_base.restrict_component(component, 0, local_xi)
        for component in tile_state
    ]
    domain = support.SourceDomain(
        global_xi,
        Fraction(0),
        tile_domain.eta_center,
        tile_domain.eta_radius,
    )
    if eta_refinement != "ROOT":
        left, right, _checks = adaptive.split_state(face_state, 1)
        if eta_refinement == "L":
            face_state = left
            domain = domain.split(1, -1)
        else:
            face_state = right
            domain = domain.split(1, 1)

    (
        first,
        _first_projection,
        _approach,
        downward,
        upward,
        tagged,
        stabilization_checks,
    ) = support.run_tile(tile_id, face_state, domain)

    carriers = []
    for index, item in enumerate(tagged):
        suffix = "_".join(item.path) if item.path else "ROOT"
        carriers.append(
            {
                "carrier_id": (
                    f"{face}:{tile_id}:{eta_refinement}:{index}:{suffix}"
                ),
                "source_domain": item.domain.as_json(),
                "event_split_path": list(item.path),
                "components": [
                    support.tm2r_json(component)
                    for component in item.projection.carrier
                ],
                "event_time": support.interval_json(item.projection.event_time),
                "event_derivative": support.interval_json(
                    item.projection.derivative
                ),
                "event_normal": support.interval_json(item.projection.normal),
            }
        )

    payload = {
        "schema": SCHEMA,
        "worker_source_sha256": sha256(Path(__file__)),
        "support_worker_source_sha256": sha256(Path(support.__file__)),
        "face_base_worker_source_sha256": sha256(Path(face_base.__file__)),
        "base_source_sha256": sha256(Path(base.__file__)),
        "adaptive_source_sha256": sha256(Path(adaptive.__file__)),
        "python_version": platform.python_version(),
        "python_flint_version": flint.__version__,
        "arb_precision_bits": base.PRECISION_BITS,
        "leaf_id": base.LEAF_ID,
        "source_face": face,
        "tile_id": tile_id,
        "eta_refinement": eta_refinement,
        "source_domain": domain.as_json(),
        "source_split_reconstructions": source_split_checks,
        "first_return_end_step": first.end_step,
        "first_event_projected": True,
        "downward_event_time": support.interval_json(downward.event_time),
        "downward_event_normal": support.interval_json(downward.normal),
        "second_event_time": support.interval_json(upward.event_time),
        "second_event_derivative": support.interval_json(upward.derivative),
        "second_event_normal": support.interval_json(upward.normal),
        "second_event_split_nodes": upward.split_nodes,
        "second_event_split_reconstructions": upward.split_reconstructions,
        "outward_stabilization_checks": stabilization_checks,
        "selected_source_face_chain_certificate": True,
        "point_fallback_used": False,
        "carrier_kind": qr_mode,
        "target_unstable_row": [str(TARGET_U_X), str(TARGET_U_Y)],
        "target_unstable_norm_squared": str(TARGET_U_NORM_SQUARED),
        "target_kernel_orthogonality_exact": qr_mode == "FIXED_COVECTOR",
        "dynamic_direction_seed_count": DYNAMIC_SEEDS,
        "dynamic_direction_transport_count": DYNAMIC_TRANSPORTS,
        "second_direction_seed_count": SECOND_DIRECTION_SEEDS,
        "second_direction_transport_count": SECOND_DIRECTION_TRANSPORTS,
        "reconditionings": base.STATS.reconditionings,
        "generator_reconstructions": base.STATS.generator_reconstructions,
        "maximum_basis_inverse_row_sum": str(
            base.STATS.max_basis_inverse_row_sum
        ),
        "carriers": carriers,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
