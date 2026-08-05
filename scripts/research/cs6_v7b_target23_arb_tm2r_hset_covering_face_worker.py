#!/usr/bin/env python3
"""Transport one exact source exit face to the second CS6 section."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from fractions import Fraction
from pathlib import Path

import flint
from flint import arb

import cs6_v7b_target23_arb_tm2r_first_return_worker as base
import cs6_v7b_target23_arb_tm2r_hset_covering_carrier_worker as support
import cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker as adaptive


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-hset-exit-face.v1"
FACE_TILES = {
    "LEFT": ("XLEL", "XLEH", Fraction(-1)),
    "RIGHT": ("XHEL", "XHEH", Fraction(1)),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restrict_component(
    component: base.TM2R, variable: int, value: Fraction
) -> base.TM2R:
    coefficients: dict[tuple[int, ...], arb] = {}
    value_ball = base.rational_ball(value)
    for monomial, coefficient in component.coefficients.items():
        exponent = monomial[variable]
        restricted = list(monomial)
        restricted[variable] = 0
        key = tuple(restricted)
        coefficients[key] = (
            coefficients.get(key, arb(0))
            + coefficient * value_ball**exponent
        )
    return base.TM2R(coefficients, component.remainder)


def main() -> None:
    face = os.environ.get("CS6_SOURCE_FACE", "")
    tile_id = os.environ.get("CS6_SOURCE_TILE", "")
    if face not in FACE_TILES:
        raise SystemExit("CS6_SOURCE_FACE must be LEFT or RIGHT")
    allowed_a, allowed_b, global_xi = FACE_TILES[face]
    if tile_id not in (allowed_a, allowed_b):
        raise SystemExit(f"tile {tile_id!r} does not meet source face {face}")
    eta_refinement = os.environ.get("CS6_FACE_ETA_REFINEMENT", "ROOT")
    if eta_refinement not in {"ROOT", "L", "H"}:
        raise SystemExit("CS6_FACE_ETA_REFINEMENT must be ROOT, L, or H")

    base.SOURCE_DEGREE = 2
    base.TIME_TAYLOR_ORDER = 12
    base.recondition = adaptive.point_coefficient_recondition
    tiles, source_split_checks = support.source_tiles()
    tile_state, tile_domain = tiles[tile_id]
    local_xi = Fraction(-1) if face == "LEFT" else Fraction(1)
    face_state = [
        restrict_component(component, 0, local_xi)
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
                "event_derivative": support.interval_json(item.projection.derivative),
                "event_normal": support.interval_json(item.projection.normal),
            }
        )

    payload = {
        "schema": SCHEMA,
        "worker_source_sha256": sha256(Path(__file__)),
        "support_worker_source_sha256": sha256(Path(support.__file__)),
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
        "carriers": carriers,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
