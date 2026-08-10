#!/usr/bin/env python3
"""Transport the accepted raw pre-QR XLEL carrier to the next section."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import flint
from flint import arb

import cs6_v7b_target23_arb_tm2r_composability_carrier_worker as composability
import cs6_v7b_target23_arb_tm2r_event_prerecond_worker as prerecond


centered = prerecond.centered
prior = prerecond.prior
base = prerecond.base
adaptive = prerecond.adaptive
chain = prerecond.chain
event = prerecond.event
transport = composability.transport

SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-prerecond-transport.v1"
TILE_ID = prerecond.TILE_ID
CRITICAL_PATH = prerecond.CRITICAL_PATH
SECTION_ROWS = prerecond.SECTION_ROWS
MAX_SPLIT_DEPTH = 8
MAX_SPLIT_NODES = 255
STOP_AFTER_FIRST_UNRESOLVED = True
EXPECTED_PRERECOND_RECEIPT_SHA256 = (
    "4b615c5632ba9537d639d4fe831c924aff1586a0d4a9db1f2f4efd9c1f1daa3a"
)


@dataclass
class TransportBranch:
    state: list[base.TM2R]
    domain: composability.SymbolicDomain
    depth: int
    path: tuple[str, ...]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_json(value: arb) -> list[str]:
    return [base.lower_fraction(value), base.upper_fraction(value)]


def weights_json(values: list[arb]) -> list[list[str]]:
    return [interval_json(value) for value in values]


def bool_check(checks: list[dict[str, object]], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})


def same_interval(left: arb, right: arb) -> bool:
    return left.contains(right) and right.contains(left)


def same_component(left: base.TM2R, right: base.TM2R) -> bool:
    monomials = left.coefficients.keys() | right.coefficients.keys()
    return all(
        same_interval(
            left.coefficients.get(monomial, arb(0)),
            right.coefficients.get(monomial, arb(0)),
        )
        for monomial in monomials
    ) and same_interval(left.remainder, right.remainder)


def lineage_preserving_recondition(
    state: list[base.TM2R],
) -> list[base.TM2R]:
    """Copy a TM2R carrier without changing any symbolic coordinate."""
    return [
        base.TM2R(dict(component.coefficients), arb(component.remainder))
        for component in state
    ]


def critical_domain() -> composability.SymbolicDomain:
    tiles, _checks = composability.source_tiles()
    _state, domain = tiles[TILE_ID]
    for token in CRITICAL_PATH:
        body = token.removeprefix("DOWN_")
        name, side = body[:-1], body[-1]
        variable = adaptive.VARIABLE_NAMES.index(name)
        left, right = composability.split_domain_pair(domain, variable)
        domain = left if side == "L" else right
    return domain


def capture_accepted_raw_projection(
    state: list[base.TM2R], center: Fraction
) -> tuple[list[base.TM2R], dict[str, object], int]:
    captured: list[list[base.TM2R]] = []
    original = adaptive.point_coefficient_recondition

    def capture(candidate: list[base.TM2R]) -> list[base.TM2R]:
        ranges = [component.range() for component in candidate]
        if ranges[2].lower() == 0 and ranges[2].upper() == 0:
            weights = centered.variable_weights(candidate, rows=SECTION_ROWS)
            if all(value.upper() > 0 for value in weights):
                captured.append(candidate)
        return original(candidate)

    base.recondition = capture
    try:
        chart = prerecond.prerecond_event_chart(state, center)
    finally:
        base.recondition = original
    if chart.get("accepted") is not True:
        raise base.Refusal(
            "PRERECOND_EVENT_REPLAY_REFUSED",
            f"status={chart.get('status')}",
        )
    if len(captured) != 1:
        raise base.Refusal(
            "RAW_PROJECTION_CAPTURE_AMBIGUOUS",
            f"captured={len(captured)}",
        )
    return captured[0], chart, len(captured)


def rational_hull_json(values: list[arb]) -> list[str]:
    if not values:
        raise base.Refusal("EMPTY_TRANSPORT_HULL", "cannot hull an empty family")
    lowers = [Fraction(base.lower_fraction(value)) for value in values]
    uppers = [Fraction(base.upper_fraction(value)) for value in values]
    return [str(min(lowers)), str(max(uppers))]


def transport_next_return(
    raw_projection: list[base.TM2R],
    root_domain: composability.SymbolicDomain,
) -> dict[str, object]:
    result: dict[str, object] = {
        "complete": False,
        "reconditioner": (
            f"{lineage_preserving_recondition.__module__}."
            f"{lineage_preserving_recondition.__qualname__}"
        ),
        "split_depth_limit": MAX_SPLIT_DEPTH,
        "split_node_limit": MAX_SPLIT_NODES,
        "stop_after_first_unresolved": STOP_AFTER_FIRST_UNRESOLVED,
    }
    stabilized, stabilization_checks = chain.outward_stabilize_carrier(
        raw_projection
    )
    pending = [TransportBranch(stabilized, root_domain, 0, ())]
    final: list[transport.TaggedProjection] = []
    unresolved: list[dict[str, object]] = []
    split_nodes = 0
    split_reconstructions = 0
    split_counts = [0 for _ in range(base.VARIABLES)]

    original_recondition = base.recondition
    base.recondition = lineage_preserving_recondition
    try:
        while pending:
            branch = pending.pop()
            tagged: list[transport.TaggedProjection] = []
            original_cover = chain.project_upward_cover

            def capture_cover(
                state: list[base.TM2R], reference_time: Fraction
            ) -> tuple[list[chain.SectionProjection], int, int]:
                nonlocal tagged
                tagged, nodes, reconstructions = (
                    transport.tagged_project_upward_cover(
                        state, reference_time, branch.domain
                    )
                )
                return [item.projection for item in tagged], nodes, reconstructions

            chain.project_upward_cover = capture_cover
            try:
                try:
                    upward = chain.seek_upward_return(branch.state)
                except base.Refusal as refusal:
                    if branch.depth >= MAX_SPLIT_DEPTH or split_nodes >= MAX_SPLIT_NODES:
                        unresolved.append(
                            {
                                "path": list(branch.path),
                                "depth": branch.depth,
                                "failure_class": refusal.failure_class,
                                "detail": refusal.detail,
                                "domain": branch.domain.as_json(),
                            }
                        )
                        if STOP_AFTER_FIRST_UNRESOLVED:
                            pending.clear()
                        continue
                    variable, _weight = adaptive.dominant_variable(branch.state)
                    left, right, checks = adaptive.split_state(
                        branch.state, variable
                    )
                    left_domain, right_domain = composability.split_domain_pair(
                        branch.domain, variable
                    )
                    name = adaptive.VARIABLE_NAMES[variable]
                    child_depth = branch.depth + 1
                    split_nodes += 1
                    split_reconstructions += checks
                    split_counts[variable] += 1
                    pending.extend(
                        (
                            TransportBranch(
                                right,
                                right_domain,
                                child_depth,
                                branch.path + (name + "H",),
                            ),
                            TransportBranch(
                                left,
                                left_domain,
                                child_depth,
                                branch.path + (name + "L",),
                            ),
                        )
                    )
                    print(
                        "prerecond-transport-split "
                        f"depth={child_depth} variable={name} "
                        f"reason={refusal.failure_class}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
            finally:
                chain.project_upward_cover = original_cover

            if len(tagged) != len(upward.carriers):
                raise base.Refusal(
                    "TAGGED_CARRIER_COUNT_MISMATCH",
                    f"tagged={len(tagged)} upward={len(upward.carriers)}",
                )
            for item in tagged:
                final.append(
                    transport.TaggedProjection(
                        item.projection,
                        item.domain,
                        branch.path + item.path,
                    )
                )
            print(
                "prerecond-next-return "
                f"path={'_'.join(branch.path) or 'ROOT'} "
                f"carriers={len(tagged)}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        base.recondition = original_recondition

    cover_certified = False
    cover_checks = 0
    if not unresolved and final:
        cover_checks = composability.certify_terminal_domain_cover(
            root_domain, [item.domain for item in final]
        )
        cover_certified = True

    carriers: list[dict[str, object]] = []
    all_variables_preserved = True
    for index, item in enumerate(final):
        weights = centered.variable_weights(
            item.projection.carrier, rows=SECTION_ROWS
        )
        preserved = all(value.upper() > 0 for value in weights)
        all_variables_preserved = all_variables_preserved and preserved
        carriers.append(
            {
                "carrier_id": index,
                "transport_path": list(item.path),
                "domain": item.domain.as_json(),
                "components": [
                    transport.tm2r_json(component)
                    for component in item.projection.carrier
                ],
                "event_time": interval_json(item.projection.event_time),
                "event_derivative": interval_json(item.projection.derivative),
                "event_normal": interval_json(item.projection.normal),
                "variable_weights": weights_json(weights),
                "all_six_variables_preserved": preserved,
            }
        )

    complete = (
        not unresolved
        and bool(carriers)
        and cover_certified
        and all_variables_preserved
    )
    result.update(
        complete=complete,
        status=(
            "COMPLETE"
            if complete
            else (
                "FINAL_SYMBOLIC_DEPENDENCE_LOST"
                if not unresolved and carriers and not all_variables_preserved
                else "TRANSPORT_REFUSED"
            )
        ),
        outward_stabilization_checks=stabilization_checks,
        split_nodes=split_nodes,
        split_reconstructions=split_reconstructions,
        split_counts={
            adaptive.VARIABLE_NAMES[index]: count
            for index, count in enumerate(split_counts)
        },
        unresolved=unresolved,
        terminal_domain_cover_certified=cover_certified,
        terminal_domain_cover_checks=cover_checks,
        all_six_variables_preserved=all_variables_preserved,
        carriers=carriers,
    )
    # Partial-branch hulls are not global enclosures of an unresolved cover.
    if final and not unresolved:
        result.update(
            event_time=rational_hull_json(
                [item.projection.event_time for item in final]
            ),
            event_derivative=rational_hull_json(
                [item.projection.derivative for item in final]
            ),
            event_normal=rational_hull_json(
                [item.projection.normal for item in final]
            ),
        )
    return result


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("pre-QR transport requires Python >= 3.10")
    base.SOURCE_DEGREE = 2
    base.TIME_TAYLOR_ORDER = 12
    base.recondition = adaptive.point_coefficient_recondition
    event.MAX_PHASE_STEPS = composability.MAX_FIRST_RETURN_STEPS

    source_path = Path(__file__)
    research = source_path.parent
    prerecond_receipt = (
        research
        / "receipts"
        / "cs6_v7b_target23_arb_tm2r_event_prerecond_v1"
        / "event_prerecond.json"
    )
    if not prerecond_receipt.is_file():
        raise SystemExit(f"frozen pre-QR receipt is missing: {prerecond_receipt}")
    prior_payload = json.loads(prerecond_receipt.read_text(encoding="ascii"))

    checks: list[dict[str, object]] = []
    bool_check(
        checks,
        "prior_prerecond_receipt_hash_matches",
        sha256(prerecond_receipt) == EXPECTED_PRERECOND_RECEIPT_SHA256,
    )
    bool_check(
        checks,
        "prior_prerecond_receipt_is_accepted",
        prior_payload.get("classification")
        == "PREDICTOR_CENTERED_PRERECOND_EVENT_ACCEPTED"
        and prior_payload.get("predictor_centered_prerecond_event_accepted") is True,
    )

    state, approach, first_end_step, source_checks, critical_checks = (
        centered.critical_state(checks)
    )
    _predictor, _predictor_range, center, _tube, _derivative, _anchor = (
        centered.frozen_predictor(state, checks)
    )
    raw_projection, chart, captured_count = capture_accepted_raw_projection(
        state, center
    )
    accepted_scale = chart["scales"][-1]
    raw_weights = centered.variable_weights(raw_projection, rows=SECTION_ROWS)
    bool_check(checks, "raw_projection_capture_is_unique", captured_count == 1)
    bool_check(checks, "replayed_prerecond_power_is_12", chart["accepted_power"] == 12)
    bool_check(
        checks,
        "raw_projection_preserves_all_six_variables",
        all(value.upper() > 0 for value in raw_weights),
    )
    bool_check(
        checks,
        "raw_projection_matches_replayed_scale_ranges",
        prior.state_metrics(raw_projection)["ranges"]
        == accepted_scale["raw_projection_state"]["ranges"],
    )
    identity_probe = lineage_preserving_recondition(raw_projection)
    bool_check(
        checks,
        "lineage_reconditioner_is_exact_identity",
        all(
            same_component(left, right)
            for left, right in zip(raw_projection, identity_probe, strict=True)
        ),
    )

    domain = critical_domain()
    transport_result = transport_next_return(raw_projection, domain)
    if transport_result["complete"]:
        bool_check(
            checks,
            "terminal_domain_cover_is_complete",
            bool(transport_result["terminal_domain_cover_certified"]),
        )
        bool_check(
            checks,
            "final_carriers_preserve_all_six_variables",
            bool(transport_result["all_six_variables_preserved"]),
        )

    implementation_ok = all(bool(item["passed"]) for item in checks)
    complete = implementation_ok and transport_result["complete"] is True
    classification = (
        "IMPLEMENTATION_INCONSISTENCY"
        if not implementation_ok
        else (
            "PRERECOND_NEXT_RETURN_COMPLETE"
            if complete
            else (
                "PRERECOND_FINAL_SYMBOLIC_DEPENDENCE_LOST"
                if transport_result["status"]
                == "FINAL_SYMBOLIC_DEPENDENCE_LOST"
                else "PRERECOND_NEXT_RETURN_REFUSED"
            )
        )
    )
    payload = {
        "schema": SCHEMA,
        "worker_source_sha256": sha256(source_path),
        "prerecond_worker_source_sha256": sha256(Path(prerecond.__file__)),
        "centered_worker_source_sha256": sha256(Path(centered.__file__)),
        "composability_source_sha256": sha256(Path(composability.__file__)),
        "transport_source_sha256": sha256(Path(transport.__file__)),
        "chain_source_sha256": sha256(Path(chain.__file__)),
        "adaptive_source_sha256": sha256(Path(adaptive.__file__)),
        "event_source_sha256": sha256(Path(event.__file__)),
        "base_source_sha256": sha256(Path(base.__file__)),
        "prior_prerecond_receipt_sha256": sha256(prerecond_receipt),
        "python_version": platform.python_version(),
        "python_flint_version": flint.__version__,
        "arb_precision_bits": base.PRECISION_BITS,
        "source_degree": base.SOURCE_DEGREE,
        "time_taylor_order": base.TIME_TAYLOR_ORDER,
        "tile_id": TILE_ID,
        "critical_path": list(CRITICAL_PATH),
        "critical_depth": len(CRITICAL_PATH),
        "critical_domain": domain.as_json(),
        "first_return_end_step": first_end_step,
        "downward_reference_time_q": str(approach.reference_time),
        "source_split_reconstruction_checks": source_checks,
        "critical_split_reconstruction_checks": critical_checks,
        "predictor_center_q": str(center),
        "prerecond_accepted_power": chart["accepted_power"],
        "raw_projection_state": prior.state_metrics(raw_projection),
        "raw_projection_components": [
            transport.tm2r_json(component) for component in raw_projection
        ],
        "raw_projection_variable_weights": weights_json(raw_weights),
        "symbolic_transport_policy": "original_six_variables_no_qr_renumbering",
        "transport": transport_result,
        "implementation_checks": checks,
        "implementation_checks_passed": implementation_ok,
        "classification": classification,
        "next_return_complete": complete,
        "diagnostic_complete": True,
        "point_fallback_used": False,
        "box_flattening_used": False,
        "full_transport_attempted": True,
        "covering_relation_certified": False,
        "recurrent_graph_certified": False,
        "chaos_certified": False,
        "open_problem_solved": False,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
