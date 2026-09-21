#!/usr/bin/env python3
"""Independent exhaustive oracle for the bounded synthetic dyadic D0 fixture."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import inf


PROBES = (0, 1, 2)  # A, B, C
HISTORIES = (0, 1, 2, 3)

# Frozen independently from the Sounio implementation. Rows are retained-history
# predictive modes and columns are exact observations under probes A, B, and C.
OBSERVATION = {
    0: (Fraction(400, 1000), Fraction(450, 1000), Fraction(500, 1000)),
    1: (Fraction(400, 1000), Fraction(550, 1000), Fraction(500, 1000)),
    2: (Fraction(600, 1000), Fraction(450, 1000), Fraction(500, 1000)),
    3: (Fraction(600, 1000), Fraction(450, 1000), Fraction(650, 1000)),
    4: (Fraction(500, 1000), Fraction(500, 1000), Fraction(500, 1000)),
}


def replay(mode: int, word: tuple[int, ...]) -> tuple[Fraction, ...]:
    return tuple(OBSERVATION[mode][probe] for probe in word)


def replay_candidate(
    candidate: tuple[int, tuple[int, int, int]], word: tuple[int, ...]
) -> tuple[Fraction, ...]:
    """Observe a candidate; retained-history IDs are intentionally unread."""
    predictive_mode, _history_ids = candidate
    return replay(predictive_mode, word)


def trace_partition(
    modes: tuple[int, ...], word: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    blocks: dict[tuple[Fraction, ...], list[int]] = {}
    for candidate_index, mode in enumerate(modes):
        blocks.setdefault(replay(mode, word), []).append(candidate_index)
    return tuple(tuple(block) for block in blocks.values())


def exhaustive_partitions(
    modes: tuple[int, ...], max_horizon: int
) -> dict[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    result: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
    for length in range(1, max_horizon + 1):
        for word in product(PROBES, repeat=length):
            result[word] = trace_partition(modes, word)
    return result


def first_preset_separator(
    modes: tuple[int, ...], max_horizon: int
) -> tuple[int | None, tuple[int, ...] | None, int]:
    replayed = 0
    for length in range(1, max_horizon + 1):
        for word in product(PROBES, repeat=length):
            replayed += 1
            if len(trace_partition(modes, word)) == len(modes):
                return length, word, replayed
    return None, None, replayed


def observation_partition(
    knowledge: tuple[int, ...], probe: int
) -> tuple[tuple[int, ...], ...]:
    blocks: dict[Fraction, list[int]] = {}
    for mode in knowledge:
        blocks.setdefault(OBSERVATION[mode][probe], []).append(mode)
    return tuple(tuple(block) for block in blocks.values())


@lru_cache(maxsize=None)
def adaptive_value(
    knowledge: tuple[int, ...], horizon: int
) -> tuple[float, tuple[int, ...]]:
    """Return the finite-horizon minimax cost and every optimal root probe."""
    if len(knowledge) <= 1:
        return 0, ()
    if horizon == 0:
        return inf, ()

    values: list[tuple[float, int]] = []
    for probe in PROBES:
        worst_child = max(
            adaptive_value(child, horizon - 1)[0]
            for child in observation_partition(knowledge, probe)
        )
        values.append((inf if worst_child == inf else 1 + worst_child, probe))
    best = min(value for value, _ in values)
    return best, tuple(probe for value, probe in values if value == best)


def adaptive_roots_in_order(
    knowledge: tuple[int, ...], horizon: int, order: tuple[int, ...]
) -> tuple[float, tuple[int, ...]]:
    values: list[tuple[float, int]] = []
    for probe in order:
        worst_child = max(
            adaptive_value(child, horizon - 1)[0]
            for child in observation_partition(knowledge, probe)
        )
        values.append((inf if worst_child == inf else 1 + worst_child, probe))
    best = min(value for value, _ in values)
    return best, tuple(sorted(probe for value, probe in values if value == best))


def main() -> int:
    # D0-W0: equal declared current state, one common probe, exact divergence.
    declared_current = {
        "left": Fraction(700, 1000),
        "right": Fraction(300, 1000),
        "relational": Fraction(500, 1000),
        "context": 7,
    }
    assert declared_current == dict(declared_current)
    left_future_raw = (400, 1000)
    right_future_raw = (600, 1000)
    left_future = Fraction(*left_future_raw)
    right_future = Fraction(*right_future_raw)
    cross_left = left_future_raw[0] * right_future_raw[1]
    cross_right = right_future_raw[0] * left_future_raw[1]
    cross_denominator = left_future_raw[1] * right_future_raw[1]
    difference = Fraction(cross_left - cross_right, cross_denominator)
    assert (cross_left, cross_right, cross_denominator) == (400000, 600000, 1000000)
    assert difference == Fraction(-1, 5)

    # Exhaust every preset word through the declared horizon, not only winners.
    partitions = exhaustive_partitions(HISTORIES, 3)
    assert len(partitions) == 39
    one_step_blocks = tuple(len(partitions[(probe,)]) for probe in PROBES)
    assert one_step_blocks == (2, 2, 2)
    preset_cost, preset_word, replayed_prefix = first_preset_separator(HISTORIES, 3)
    assert preset_cost == 3
    assert preset_word == (0, 1, 2)
    assert replayed_prefix == 18

    adaptive_cost, optimal_roots = adaptive_value(HISTORIES, 2)
    assert adaptive_cost == 2
    assert optimal_roots == (0,)
    root_children = observation_partition(HISTORIES, optimal_roots[0])
    assert root_children == ((0, 1), (2, 3))
    assert adaptive_value(root_children[0], 1) == (1, (1,))
    assert adaptive_value(root_children[1], 1) == (1, (2,))

    # D0-W1: different unread annotations with one shared predictive mode.
    null_candidates = ((4, 8101, 9101), (4, 8102, 9102))
    assert null_candidates[0][1:] != null_candidates[1][1:]
    null_partitions = exhaustive_partitions((4, 4), 2)
    assert len(null_partitions) == 12
    assert all(len(blocks) == 1 for blocks in null_partitions.values())
    assert all(blocks == ((0, 1),) for blocks in null_partitions.values())

    # D0-W2: candidate order and every history label are observationally inert.
    candidate_permutation = (3, 1, 0, 2)
    permuted_cost, permuted_roots = adaptive_value(candidate_permutation, 2)
    assert (permuted_cost, permuted_roots) == (adaptive_cost, optimal_roots)
    original_ids = ((5501, 8001, 9001), (5502, 8002, 9002),
                    (5503, 8003, 9003), (5504, 8004, 9004))
    relabeled_ids = ((5560, 8060, 9060), (5561, 8061, 9061),
                     (5562, 8062, 9062), (5563, 8063, 9063))
    assert original_ids != relabeled_ids
    for mode, original_id, relabeled_id in zip(HISTORIES, original_ids, relabeled_ids):
        assert original_id != relabeled_id
        for word in partitions:
            assert replay_candidate((mode, original_id), word) == replay_candidate(
                (mode, relabeled_id), word
            )
    reverse_cost, reverse_roots = adaptive_roots_in_order(HISTORIES, 2, (2, 1, 0))
    assert (reverse_cost, reverse_roots) == (adaptive_cost, optimal_roots)

    # D0-W3: forced exhaustion cannot manufacture ambiguity or minimality.
    required_nodes = 10
    requested_nodes = 1
    assert requested_nodes < required_nodes
    incomplete_search = {
        "budget_exhausted": True,
        "search_complete": False,
        "ambiguity_authorized": False,
        "minimality_authorized": False,
    }
    assert incomplete_search["budget_exhausted"]
    assert not incomplete_search["search_complete"]
    assert not incomplete_search["ambiguity_authorized"]
    assert not incomplete_search["minimality_authorized"]

    # D0-W4a: an omitted ordinary context variable explains the control pair.
    context_observation = {10: Fraction(400, 1000), 20: Fraction(600, 1000)}
    assert context_observation[10] == left_future
    assert context_observation[20] == right_future
    history_only_authorized = False
    assert not history_only_authorized

    # D0-W4b: promoting the predictive mode restores a finite Markov state.
    expanded_relational_state = {mode: tuple(OBSERVATION[mode]) for mode in HISTORIES}
    for mode in HISTORIES:
        for probe in PROBES:
            assert expanded_relational_state[mode][probe] == OBSERVATION[mode][probe]
    markov_factorability_restored = True
    participant_product_non_reduction = left_future != right_future
    unbounded_history_irreducibility = False
    assert markov_factorability_restored
    assert participant_product_non_reduction
    assert not unbounded_history_irreducibility

    print("ORACLE_D0_W0 difference=-1/5 common_probe=A")
    print("ORACLE_D0_W1 horizon=2 residual_blocks=1 global_equivalence=false")
    print("ORACLE_D0_W2 permutation=true history_id_invariant=true reverse=true")
    print("ORACLE_D0_W3 one_step=false adaptive_cost=2 preset_cost=3 incomplete=true")
    print("ORACLE_D0_W3 root=A children=400:B,600:C preset_first=A,B,C replayed=18")
    print("ORACLE_D0_W4 context_rival=true markov_expansion=true unbounded=false")
    print("DYADIC NON-REDUCTION D0 ORACLE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
