#!/usr/bin/env python3
"""Independent exhaustive oracle for the frozen D6 partial composition."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations


NO_PROBE = 0
PENDING = 1
WITHHELD = 2
COMMITTED = 3


@dataclass(frozen=True)
class Carrier:
    name: str
    first: int
    last: int
    leaves: int
    mask: int
    flat: int
    tree: int
    policy: bool
    boundary: bool
    probe: bool
    status: int
    survivor: int
    burden: int
    evidence_count: int
    evidence_fp: int
    last_provenance: int
    committed_without_policy: bool
    withheld_with_policy: bool
    operator_id: int = 9601
    family_id: int = 520
    protocol_id: int = 8400
    budget_capacity: int = 7


def flat_factor(count: int) -> int:
    return 101**count


def evidence_factor(count: int) -> int:
    return 31**count


def carriers_composable(left: Carrier, right: Carrier) -> bool:
    return (
        left.operator_id == right.operator_id
        and left.family_id == right.family_id
        and left.protocol_id == right.protocol_id
        and left.budget_capacity == right.budget_capacity
        and left.last + 1 == right.first
        and left.mask & right.mask == 0
    )


def compose(left: Carrier, right: Carrier) -> Carrier:
    assert carriers_composable(left, right)
    boundary_applies = left.boundary and right.status == PENDING
    execute_now = boundary_applies and not left.policy
    withhold_now = boundary_applies and left.policy
    committed_before = left.status == COMMITTED or right.status == COMMITTED
    withheld_before = left.status == WITHHELD or right.status == WITHHELD
    has_probe = left.probe or right.probe
    if committed_before:
        status = COMMITTED
    elif withheld_before:
        status = WITHHELD
    elif execute_now:
        status = COMMITTED
    elif withhold_now:
        status = WITHHELD
    elif has_probe:
        status = PENDING
    else:
        status = NO_PROBE
    before_count = left.evidence_count + right.evidence_count
    before_fp = left.evidence_fp * evidence_factor(right.evidence_count) + right.evidence_fp
    result = Carrier(
        name=f"({left.name}*{right.name})",
        first=left.first,
        last=right.last,
        leaves=left.leaves + right.leaves,
        mask=left.mask | right.mask,
        flat=left.flat * flat_factor(right.leaves) + right.flat,
        tree=left.tree * 31 + right.tree,
        policy=left.policy or right.policy,
        boundary=left.boundary or right.boundary,
        probe=has_probe,
        status=status,
        survivor=2 if status == COMMITTED else 3,
        burden=left.burden + right.burden + (4 if execute_now else 0),
        evidence_count=before_count + int(execute_now),
        evidence_fp=before_fp * 31 + 8103 if execute_now else before_fp,
        last_provenance=(
            8103
            if execute_now
            else right.last_provenance
            if right.evidence_count
            else left.last_provenance
        ),
        committed_without_policy=(
            left.committed_without_policy
            or right.committed_without_policy
            or execute_now
        ),
        withheld_with_policy=(
            left.withheld_with_policy or right.withheld_with_policy or withhold_now
        ),
    )
    if left.status == COMMITTED or right.status == COMMITTED:
        assert result.status == COMMITTED, "composition erased committed evidence"
    return result


ATOMS = (
    Carrier("a", 1, 1, 1, 1, 9101, 9101, True, False, False, 0, 3, 3, 1, 8101, 8101, False, False),
    Carrier("b", 2, 2, 1, 2, 9102, 9102, False, True, False, 0, 3, 0, 0, 0, 0, False, False),
    Carrier("c", 3, 3, 1, 4, 9103, 9103, False, False, True, PENDING, 3, 0, 0, 0, 0, False, False),
)


def all_interval_outcomes(lo: int, hi: int, applications: list[tuple[Carrier, Carrier, Carrier]]) -> tuple[Carrier, ...]:
    if hi - lo == 1:
        return (ATOMS[lo],)
    outcomes: list[Carrier] = []
    for cut in range(lo + 1, hi):
        for left in all_interval_outcomes(lo, cut, applications):
            for right in all_interval_outcomes(cut, hi, applications):
                result = compose(left, right)
                applications.append((left, right, result))
                outcomes.append(result)
    return tuple(outcomes)


def main() -> None:
    applications: list[tuple[Carrier, Carrier, Carrier]] = []
    outcomes = all_interval_outcomes(0, 3, applications)
    unique_apps = {(left.name, right.name, result.name) for left, right, result in applications}
    assert len(outcomes) == 2
    assert len(unique_apps) == 4
    by_name = {outcome.name: outcome for outcome in outcomes}
    assert set(by_name) == {"((a*b)*c)", "(a*(b*c))"}
    left = by_name["((a*b)*c)"]
    right = by_name["(a*(b*c))"]
    assert left.flat == right.flat == 93_767_706
    assert left.tree == 9_037_326
    assert right.tree == 573_396
    assert (left.status, right.status) == (WITHHELD, COMMITTED)
    assert (left.survivor, right.survivor) == (3, 2)
    assert (left.burden, right.burden) == (3, 7)
    assert (left.evidence_count, right.evidence_count) == (1, 2)
    assert (left.evidence_fp, right.evidence_fp) == (8_101, 259_234)
    delta = (
        int(left.status != right.status)
        + 2 * int(left.survivor != right.survivor)
        + 4 * int(left.burden != right.burden)
        + 8 * int(left.evidence_count != right.evidence_count)
    )
    assert delta == 15

    a, b, c = ATOMS
    invalid_pairs = (
        (a, c),
        (c, b),
        (a, replace(b, mask=1)),
        (a, replace(b, operator_id=9602)),
        (a, replace(b, protocol_id=8401)),
    )
    assert all(not carriers_composable(x, y) for x, y in invalid_pairs)

    # The flat trace is an associative base-101 concatenation control.
    flat_left = (a.flat * 101 + b.flat) * 101 + c.flat
    flat_right = a.flat * (101**2) + (b.flat * 101 + c.flat)
    assert flat_left == flat_right == 93_767_706
    label_order_checksums = {
        order: order[0] * (101**2) + order[1] * 101 + order[2]
        for order in permutations((9_101, 9_102, 9_103))
    }
    assert len(label_order_checksums) == 6
    assert len(set(label_order_checksums.values())) == 6
    assert label_order_checksums[(9_101, 9_102, 9_103)] == 93_767_706
    assert max(label_order_checksums.values()) < 2**63

    committed_inputs = 0
    for left_input, right_input, result in applications:
        if left_input.status == COMMITTED or right_input.status == COMMITTED:
            committed_inputs += 1
            assert result.status == COMMITTED
            assert result.last_provenance == 8103
    assert committed_inputs == 1
    assert max(outcome.tree for outcome in outcomes) < 2**63
    assert max(outcome.flat for outcome in outcomes) < 2**63

    print("ORACLE_D6_W0 trees=2 applications=4 invalid_pairs=5 flat_fp=93767706")
    print("ORACLE_D6_W1 left=status2,mask3,burden3,fp8101 right=status3,mask2,burden7,fp259234")
    print("ORACLE_D6_W2 difference_bitset=15 committed_input_cases=1 erasures=0")
    print("ORACLE_D6_W3 flat_concat=associative label_orders=6 checksum_collisions=0 grouping_trees=9037326,573396")
    print("PROOF-CARRYING POLICY-OBSERVATION ASSOCIATOR D6 ORACLE PASS")


if __name__ == "__main__":
    main()
