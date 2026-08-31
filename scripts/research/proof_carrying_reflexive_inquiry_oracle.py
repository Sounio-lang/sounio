#!/usr/bin/env python3
"""Independent exhaustive oracle for the bounded D3 reflexive inquiry fixture."""

from __future__ import annotations

from itertools import permutations


MODES = (0, 1, 2)  # passive, relational write, instrument write
ACTIONS = (0, 1, 2)  # P: mode-specific affine update; Q/R: add one/two
MODE_NAMES = {0: "passive", 1: "relational", 2: "instrument"}
ACTION_NAMES = {0: "P", 1: "Q", 2: "R"}


def step(mode: int, state: tuple[int, int], action: int) -> tuple[int, int]:
    relational, instrument = state
    if mode == 1:
        if action == 0:
            relational *= 2
        elif action == 1:
            relational += 1
        else:
            relational += 2
    elif mode == 2:
        if action == 0:
            instrument = instrument * 2 + 1
        elif action == 1:
            instrument += 1
        else:
            instrument += 2
    return relational, instrument


def execute(mode: int, schedule: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, int]]:
    state = (1, 0)
    observations: list[int] = []
    for action in schedule:
        state = step(mode, state, action)
        observations.append(sum(state))
    return tuple(observations), state


def footprint(mode: int) -> frozenset[str]:
    if mode == 1:
        return frozenset({"relational"})
    if mode == 2:
        return frozenset({"instrument"})
    return frozenset()


def survivors_for_trace(observed: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        mode
        for mode in MODES
        if execute(mode, (0, 1))[0] + execute(mode, (1, 0))[0] == observed
    )


def survivor_ids_for_trace(
    family: tuple[tuple[int, int], ...],
    observed: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        hypothesis_id
        for hypothesis_id, mode in family
        if execute(mode, (0, 1))[0] + execute(mode, (1, 0))[0] == observed
    )


def layer_after_pq(mode: int, layer: int) -> int:
    _, state = execute(mode, (0, 1))
    return state[layer]


def audit_survivors(prior: tuple[int, ...], layer: int, value: int) -> tuple[int, ...]:
    return tuple(mode for mode in prior if layer_after_pq(mode, layer) == value)


def main() -> None:
    schedules = tuple(permutations(ACTIONS, 2))
    assert len(schedules) == 6

    traces = {
        mode: {
            schedule: execute(mode, schedule)
            for schedule in schedules
        }
        for mode in MODES
    }

    passive_pq = traces[0][(0, 1)]
    passive_qp = traces[0][(1, 0)]
    assert passive_pq == ((1, 1), (1, 0))
    assert passive_qp == ((1, 1), (1, 0))

    relation_pq = traces[1][(0, 1)]
    relation_qp = traces[1][(1, 0)]
    instrument_pq = traces[2][(0, 1)]
    instrument_qp = traces[2][(1, 0)]
    assert relation_pq == ((2, 3), (3, 0))
    assert relation_qp == ((2, 4), (4, 0))
    assert instrument_pq == ((2, 3), (1, 2))
    assert instrument_qp == ((2, 4), (1, 3))
    assert relation_pq[0] == instrument_pq[0]
    assert relation_qp[0] == instrument_qp[0]
    assert sum(relation_pq[1]) - sum(relation_qp[1]) == -1
    assert sum(instrument_pq[1]) - sum(instrument_qp[1]) == -1

    # Exhaust all ordered, distinct action pairs. P conflicts with both
    # additive actions; Q and R commute on state but not on emitted trace.
    for mode in (1, 2):
        assert footprint(mode)
        for left, right in ((0, 1), (0, 2)):
            forward = traces[mode][(left, right)]
            reverse = traces[mode][(right, left)]
            assert forward[1] != reverse[1]
        additive_forward = traces[mode][(1, 2)]
        additive_reverse = traces[mode][(2, 1)]
        assert additive_forward[1] == additive_reverse[1]
        assert additive_forward[0] != additive_reverse[0]

    for left, right in ((0, 1), (0, 2), (1, 2)):
        assert traces[0][(left, right)] == traces[0][(right, left)]

    observed_order = (2, 3, 2, 4)
    survivors = survivors_for_trace(observed_order)
    assert survivors == (1, 2)
    assert audit_survivors(survivors, 0, 3) == (1,)
    assert audit_survivors(survivors, 1, 2) == (2,)

    unique_declared_traces = {
        execute(mode, (0, 1))[0] + execute(mode, (1, 0))[0]
        for mode in MODES
    }
    assert unique_declared_traces == {(1, 1, 1, 1), (2, 3, 2, 4)}
    assert all(len(survivors_for_trace(trace)) in (1, 2) for trace in unique_declared_traces)

    # All 3! identifier relabelings preserve the semantic partitions because
    # prediction depends only on mode.
    original_ids = (210, 211, 212)
    relabel_count = 0
    for relabeled_ids in permutations((410, 411, 412)):
        assert all(a != b for a, b in zip(original_ids, relabeled_ids))
        relabeled_family = tuple(zip(relabeled_ids, MODES))
        surviving_ids = survivor_ids_for_trace(relabeled_family, observed_order)
        assert surviving_ids == (relabeled_ids[1], relabeled_ids[2])
        surviving_modes = tuple(
            mode
            for hypothesis_id, mode in relabeled_family
            if hypothesis_id in surviving_ids
        )
        assert surviving_modes == survivors
        relabel_count += 1
    assert relabel_count == 6

    initial_mask = 7
    missing_mask = initial_mask
    unaudited_mask = initial_mask
    assert missing_mask == 7 and unaudited_mask == 7
    fingerprint = 6101 * 31 + 6102
    assert fingerprint == 195233
    maximum_two_link_fingerprint = 1_000_000 * 31 + 1_000_000
    assert maximum_two_link_fingerprint == 32_000_000

    print("ORACLE_D3_W0 schedules=6 passive_all_pairs=layers_and_values_commute")
    print("ORACLE_D3_W1 P,Q relation=-1 instrument=-1 projected_traces_identical")
    print("ORACLE_D3_W2 Q,R overlap=true final_layers_commute emitted_values_commute=false")
    print("ORACLE_D3_W3 order_trace=2,3,2,4 survivors=relational,instrument")
    print("ORACLE_D3_W4 relational_audit=3 survivor=relational instrument_audit=2 survivor=instrument")
    print("ORACLE_D3_W5 relabelings=6 partitions=invariant missing=7 unaudited=7")
    print("ORACLE_D3_W6 fingerprint=195233 fingerprint_max2=32000000 i64_safe=true")
    print("PROOF-CARRYING REFLEXIVE INQUIRY D3 ORACLE PASS")


if __name__ == "__main__":
    main()
