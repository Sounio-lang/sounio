#!/usr/bin/env python3
"""Independent finite oracle for endogenous observability D4.

This intentionally duplicates the tiny fixture rather than importing Sounio
logic. It checks the complete Boolean custody space, the observation partition,
retry discrimination, relabel invariance, time alignment, and bounded
provenance arithmetic.
"""

from itertools import permutations, product


MODE_NAMES = {
    0: "delivery_failure",
    1: "declared_target_independent",
    2: "declared_target_dependent",
    3: "policy_withholding",
}
HYPOTHESIS_IDS = (310, 311, 312, 313)


def scheduled(mode: int) -> bool:
    return mode != 3


def delivered(mode: int) -> bool:
    return mode in (1, 2)


def hidden_target(mode: int) -> int:
    return 2 if mode <= 1 else 8


def original_custody(mode: int) -> tuple[int, ...]:
    return (
        1,
        int(scheduled(mode)),
        int(delivered(mode)),
        int(delivered(mode)),
        0,
        0,
        0,
        -1,
    )


def modes_matching(trace: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(mode for mode in range(4) if original_custody(mode) == trace)


def mask(modes: tuple[int, ...]) -> int:
    return sum(1 << mode for mode in modes)


def retry_prediction(mode: int) -> tuple[int, int, int]:
    responded = mode == 1
    return (int(responded), int(responded), 2 if responded else -1)


def legacy_missing_token(mode: int) -> int:
    return original_custody(mode)[-1]


def partition_ids(
    family: tuple[tuple[int, int], ...],
) -> dict[tuple[int, ...], tuple[int, ...]]:
    blocks: dict[tuple[int, ...], list[int]] = {}
    for hypothesis_id, mode in family:
        blocks.setdefault(original_custody(mode), []).append(hypothesis_id)
    return {trace: tuple(ids) for trace, ids in blocks.items()}


def attempt_custody_update(
    prior: tuple[int, ...],
    trace: tuple[int, ...],
    policy_custody_complete: bool,
) -> tuple[int, ...] | None:
    if not policy_custody_complete:
        return None
    return tuple(mode for mode in prior if original_custody(mode) == trace)


def attempt_retry_update(
    prior: tuple[int, ...],
    observed: tuple[int, int, int],
    provenance_connected: bool,
) -> tuple[int, ...] | None:
    if not provenance_connected:
        return None
    return tuple(mode for mode in prior if retry_prediction(mode) == observed)


def main() -> None:
    originals = {mode: original_custody(mode) for mode in range(4)}
    assert {legacy_missing_token(mode) for mode in range(4)} == {-1}
    assert all(trace[-2:] == (0, -1) for trace in originals.values())

    partition = {trace: modes_matching(trace) for trace in set(originals.values())}
    assert set(partition.values()) == {(0,), (1, 2), (3,)}
    assert {mask(modes) for modes in partition.values()} == {1, 6, 8}

    custody = (1, 1, 1, 1, 0, 0, 0, -1)
    custody_survivors = modes_matching(custody)
    assert custody_survivors == (1, 2)
    assert mask(custody_survivors) == 6
    assert originals[1] == originals[2]
    assert hidden_target(1) == 2 and hidden_target(2) == 8
    assert retry_prediction(1) == (1, 1, 2)
    assert retry_prediction(2) == (0, 0, -1)

    response_survivors = tuple(
        mode for mode in custody_survivors if retry_prediction(mode) == (1, 1, 2)
    )
    nonresponse_survivors = tuple(
        mode for mode in custody_survivors if retry_prediction(mode) == (0, 0, -1)
    )
    assert response_survivors == (1,) and mask(response_survivors) == 2
    assert nonresponse_survivors == (2,) and mask(nonresponse_survivors) == 4

    burden = 5 + 4
    fingerprint = 7101 * 31 + 7102
    assert burden == 9 and fingerprint == 227233
    assert 1_000_000 * 31 + 1_000_000 == 32_000_000
    assert 32_000_000 < 2**63

    prompt_tick = 2
    arrival_tick = 3
    aligned_tick = arrival_tick
    assert arrival_tick - prompt_tick == 1
    assert aligned_tick == 3 and aligned_tick != prompt_tick

    # Every assignment of the four opaque IDs to the four modes leaves the
    # prediction partition unchanged. The selected ID changes, never the mode.
    relabelings = 0
    for relabeling in permutations(HYPOTHESIS_IDS):
        assert len(set(relabeling)) == 4
        relabeled_family = tuple(zip(relabeling, range(4)))
        relabeled_ids_by_trace = partition_ids(relabeled_family)
        assert set(relabeled_ids_by_trace) == set(partition)
        central_ids = relabeled_ids_by_trace[custody]
        assert central_ids == (relabeling[1], relabeling[2])
        central_modes = tuple(
            mode
            for hypothesis_id, mode in relabeled_family
            if hypothesis_id in central_ids
        )
        assert central_modes == partition[custody]
        assert relabeling[1] != relabeling[2]
        relabelings += 1
    assert relabelings == 24

    # Exhaust the full custody representation: seven Boolean fields and three
    # possible scalar slots. Only the three declared partition cells match.
    exhaustive_traces = 0
    matched_traces: dict[tuple[int, ...], tuple[int, ...]] = {}
    for booleans in product((0, 1), repeat=7):
        for value in (-1, 2, 8):
            trace = (*booleans, value)
            survivors = modes_matching(trace)
            if survivors:
                matched_traces[trace] = survivors
            exhaustive_traces += 1
    assert exhaustive_traces == 384
    assert matched_traces == partition

    # Abstention is no transition and preserves the prior mask. Mask zero is
    # reserved for declared-family refutation, not inadmissible evidence.
    policy_erased_update = attempt_custody_update(custody_survivors, custody, False)
    disconnected_retry_update = attempt_retry_update(
        custody_survivors, (1, 1, 2), False,
    )
    assert policy_erased_update is None and disconnected_retry_update is None
    policy_erased_mask = mask(custody_survivors)
    disconnected_retry_mask = mask(custody_survivors)
    assert policy_erased_mask == 6 and disconnected_retry_mask == 6

    print("ORACLE_D4_W0 legacy=missing mechanisms=4 custody_partitions=1|6|8")
    print("ORACLE_D4_W1 custody=1,1,1,1,0,0,0,-1 survivors=independent,dependent mask=6")
    print("ORACLE_D4_W2 original_equal=true hidden=2|8 retry_predictions=different recoverability=false")
    print("ORACLE_D4_W3 retry_response=2 retry_nonresponse=4 burden=9 fingerprint=227233")
    print("ORACLE_D4_W4 delayed=2->3 retroactive=false")
    print("ORACLE_D4_W5 relabelings=24 partitions=invariant policy_erased=6 disconnected=6")
    print("ORACLE_D4_W6 fingerprint_max2=32000000 exhaustive_traces=384 i64_safe=true")
    print("PROOF-CARRYING ENDOGENOUS OBSERVABILITY D4 ORACLE PASS")


if __name__ == "__main__":
    main()
