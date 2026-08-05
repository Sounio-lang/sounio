#!/usr/bin/env python3
"""Export the five rigorous second-section TM2R carriers, one source tile at a time."""

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

import cs6_v7b_target23_arb_tm2r_event_chain_second_return_worker as chain
import cs6_v7b_target23_arb_tm2r_first_return_worker as base
import cs6_v7b_target23_arb_tm2r_second_return_worker as event
import cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker as adaptive


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-hset-carrier.v1"
EXPECTED_TILES = ("XLEL", "XLEH", "XHEL", "XHEH")


@dataclass(frozen=True)
class SourceDomain:
    xi_center: Fraction
    xi_radius: Fraction
    eta_center: Fraction
    eta_radius: Fraction

    def split(self, variable: int, side: int) -> "SourceDomain":
        if variable not in (0, 1):
            return self
        center = self.xi_center if variable == 0 else self.eta_center
        radius = self.xi_radius if variable == 0 else self.eta_radius
        child_center = center + side * radius / 2
        child_radius = radius / 2
        if variable == 0:
            return SourceDomain(
                child_center,
                child_radius,
                self.eta_center,
                self.eta_radius,
            )
        return SourceDomain(
            self.xi_center,
            self.xi_radius,
            child_center,
            child_radius,
        )

    def as_json(self) -> dict[str, list[str]]:
        return {
            "xi": [
                str(self.xi_center - self.xi_radius),
                str(self.xi_center + self.xi_radius),
            ],
            "eta": [
                str(self.eta_center - self.eta_radius),
                str(self.eta_center + self.eta_radius),
            ],
        }


@dataclass
class TaggedBranch:
    state: list[base.TM2R]
    depth: int
    domain: SourceDomain
    path: tuple[str, ...]


@dataclass
class TaggedProjection:
    projection: chain.SectionProjection
    domain: SourceDomain
    path: tuple[str, ...]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_json(value: arb) -> list[str]:
    return [base.lower_fraction(value), base.upper_fraction(value)]


def tm2r_json(component: base.TM2R) -> dict[str, object]:
    coefficients = []
    for monomial, coefficient in sorted(component.coefficients.items()):
        coefficients.append(
            {
                "monomial": list(monomial),
                "interval": interval_json(coefficient),
            }
        )
    return {
        "coefficients": coefficients,
        "remainder": interval_json(component.remainder),
    }


def source_tiles() -> tuple[dict[str, tuple[list[base.TM2R], SourceDomain]], int]:
    initial, _u, _s = base.initial_leaf()
    xi_left, xi_right, checks = adaptive.split_state(initial, 0)
    result: dict[str, tuple[list[base.TM2R], SourceDomain]] = {}
    for xi_id, xi_state, xi_center in (
        ("XL", xi_left, Fraction(-1, 2)),
        ("XH", xi_right, Fraction(1, 2)),
    ):
        eta_left, eta_right, eta_checks = adaptive.split_state(xi_state, 1)
        checks += eta_checks
        result[xi_id + "EL"] = (
            eta_left,
            SourceDomain(xi_center, Fraction(1, 2), Fraction(-1, 2), Fraction(1, 2)),
        )
        result[xi_id + "EH"] = (
            eta_right,
            SourceDomain(xi_center, Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
        )
    return result, checks


def tagged_project_upward_cover(
    state: list[base.TM2R],
    reference_time: Fraction,
    root_domain: SourceDomain,
) -> tuple[list[TaggedProjection], int, int]:
    pending = [TaggedBranch(state, 0, root_domain, ())]
    projections: list[TaggedProjection] = []
    unresolved: list[str] = []
    split_nodes = 0
    split_reconstructions = 0
    while pending:
        branch = pending.pop()
        derivative_model = (
            branch.state[0] * branch.state[1]
            - branch.state[2]
            - base.ZS
        )
        derivative_range = derivative_model.range()
        cheap_candidate = False
        refusal_class = "UPWARD_PREFILTER_UNRESOLVED"
        if derivative_range.lower() > 0:
            correction = -branch.state[2].range() / derivative_range
            radius = base.rational_ball(Fraction(1, 2**8))
            cheap_candidate = (
                correction.lower() > -radius and correction.upper() < radius
            )
        if cheap_candidate:
            try:
                projection = chain.project_upward_event(
                    branch.state, reference_time
                )
                projections.append(
                    TaggedProjection(projection, branch.domain, branch.path)
                )
                continue
            except base.Refusal as refusal:
                refusal_class = refusal.failure_class
        if branch.depth >= chain.MAX_EVENT_SPLIT_DEPTH or split_nodes >= 255:
            unresolved.append(refusal_class)
            continue
        variable, _weight = adaptive.dominant_variable(
            [branch.state[2], derivative_model]
        )
        left, right, checks = adaptive.split_state(branch.state, variable)
        split_nodes += 1
        split_reconstructions += checks
        name = adaptive.VARIABLE_NAMES[variable]
        pending.extend(
            (
                TaggedBranch(
                    left,
                    branch.depth + 1,
                    branch.domain.split(variable, -1),
                    branch.path + (name + "L",),
                ),
                TaggedBranch(
                    right,
                    branch.depth + 1,
                    branch.domain.split(variable, 1),
                    branch.path + (name + "H",),
                ),
            )
        )
    if unresolved:
        raise base.Refusal(
            "SECOND_EVENT_COVER_UNRESOLVED",
            f"tagged cover left {len(unresolved)} unresolved branches; "
            f"first={unresolved[0]}",
        )
    return projections, split_nodes, split_reconstructions


def run_tile(
    tile_id: str,
    tile_state: list[base.TM2R],
    domain: SourceDomain,
) -> tuple[
    event.PositiveReturn,
    chain.SectionProjection,
    chain.DownwardReturnPhase,
    chain.SectionProjection,
    chain.UpwardReturn,
    list[TaggedProjection],
    int,
]:
    first = event.integrate_positive_return(tile_state)
    first_projection = event.interval_newton_project(first)
    print(
        f"first-event-projection tile={tile_id} end_step={first.end_step}",
        file=sys.stderr,
        flush=True,
    )
    approach = chain.integrate_downward_return(first_projection.carrier)
    downward = chain.project_downward_event(
        approach.endpoint, approach.reference_time
    )
    upward_initial, stabilization_checks = chain.outward_stabilize_carrier(
        downward.carrier
    )

    tagged: list[TaggedProjection] = []
    original_cover = chain.project_upward_cover

    def capture(
        state: list[base.TM2R], reference_time: Fraction
    ) -> tuple[list[chain.SectionProjection], int, int]:
        nonlocal tagged
        candidate, nodes, checks = tagged_project_upward_cover(
            state, reference_time, domain
        )
        tagged = candidate
        return [item.projection for item in candidate], nodes, checks

    chain.project_upward_cover = capture
    try:
        upward = chain.seek_upward_return(upward_initial)
    finally:
        chain.project_upward_cover = original_cover
    if len(tagged) != len(upward.carriers):
        raise base.Refusal(
            "TAGGED_CARRIER_COUNT_MISMATCH",
            f"tagged={len(tagged)} upward={len(upward.carriers)}",
        )
    return (
        first,
        first_projection,
        approach,
        downward,
        upward,
        tagged,
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
    tiles, source_split_checks = source_tiles()
    tile_state, domain = tiles[tile_id]
    (
        first,
        first_projection,
        approach,
        downward,
        upward,
        tagged,
        stabilization_checks,
    ) = run_tile(tile_id, tile_state, domain)

    carriers = []
    for index, item in enumerate(tagged):
        suffix = "_".join(item.path) if item.path else "ROOT"
        carriers.append(
            {
                "carrier_id": f"{tile_id}:{index}:{suffix}",
                "source_domain": item.domain.as_json(),
                "event_split_path": list(item.path),
                "components": [tm2r_json(component) for component in item.projection.carrier],
                "event_time": interval_json(item.projection.event_time),
                "event_derivative": interval_json(item.projection.derivative),
                "event_normal": interval_json(item.projection.normal),
            }
        )

    source_path = Path(__file__)
    payload = {
        "schema": SCHEMA,
        "worker_source_sha256": sha256(source_path),
        "event_chain_source_sha256": sha256(Path(chain.__file__)),
        "adaptive_source_sha256": sha256(Path(adaptive.__file__)),
        "event_projection_source_sha256": sha256(Path(event.__file__)),
        "base_source_sha256": sha256(Path(base.__file__)),
        "python_version": platform.python_version(),
        "python_flint_version": flint.__version__,
        "arb_precision_bits": base.PRECISION_BITS,
        "leaf_id": base.LEAF_ID,
        "tile_id": tile_id,
        "source_domain": domain.as_json(),
        "source_split_reconstructions": source_split_checks,
        "first_return_end_step": first.end_step,
        "first_event_projected": True,
        "downward_event_time": interval_json(downward.event_time),
        "downward_event_normal": interval_json(downward.normal),
        "second_event_time": interval_json(upward.event_time),
        "second_event_derivative": interval_json(upward.derivative),
        "second_event_normal": interval_json(upward.normal),
        "second_event_split_nodes": upward.split_nodes,
        "second_event_split_reconstructions": upward.split_reconstructions,
        "outward_stabilization_checks": stabilization_checks,
        "selected_source_chain_certificate": True,
        "point_fallback_used": False,
        "carriers": carriers,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
