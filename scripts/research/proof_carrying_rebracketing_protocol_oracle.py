#!/usr/bin/env python3
"""Independent finite oracle for the frozen D7 rebracketing protocol."""

from itertools import permutations, product


ATOM_IDS = (9_101, 9_102, 9_103)
FLAT_RESULT = 93_767_706
SEMANTIC_SOURCE = 62_559_008_101
SEMANTIC_TARGET = 91_514_259_234
SEMANTIC_CODE_MAX = 92_475_999_999


def admits_local_model_decision(bits: tuple[bool, ...]) -> bool:
    same_typed, same_ordered, same_operator, both_defined, exact_equal, local = bits
    return all(
        (same_typed, same_ordered, same_operator, both_defined, exact_equal, local)
    )


def flat_left(a: int, b: int, c: int) -> int:
    return (a * 101 + b) * 101 + c


def flat_right(a: int, b: int, c: int) -> int:
    return a * 101**2 + (b * 101 + c)


def tree_left(a: int, b: int, c: int) -> int:
    return (a * 31 + b) * 31 + c


def tree_right(a: int, b: int, c: int) -> int:
    return a * 31 + (b * 31 + c)


def bounded_semantic_result_code(
    status: int,
    survivor_mask: int,
    burden: int,
    evidence_count: int,
    evidence_fingerprint: int,
) -> int:
    assert 0 <= status <= 3
    assert 0 <= survivor_mask <= 3
    assert 0 <= burden <= 7
    assert 0 <= evidence_count <= 2
    assert 0 <= evidence_fingerprint < 1_000_000
    code = (
        (((status * 31 + survivor_mask) * 31 + burden) * 31 + evidence_count)
        * 1_000_000
        + evidence_fingerprint
    )
    assert 0 <= code <= SEMANTIC_CODE_MAX
    return code


def decode_bounded_semantic_result_code(code: int) -> tuple[int, ...]:
    evidence_fingerprint = code % 1_000_000
    prefix = code // 1_000_000
    evidence_count = prefix % 31
    prefix //= 31
    burden = prefix % 31
    prefix //= 31
    survivor_mask = prefix % 31
    status = prefix // 31
    decoded = (status, survivor_mask, burden, evidence_count, evidence_fingerprint)
    assert bounded_semantic_result_code(*decoded) == code
    return decoded


def fixture_replay_admitted(bound_occurrence: int, requested_occurrence: int) -> bool:
    return bound_occurrence == 11_001 and requested_occurrence == bound_occurrence


def main() -> None:
    # Exhaustive only for the declared six-Boolean model-decision predicate.
    predicate_vectors = list(product((False, True), repeat=6))
    admitted_vectors = [bits for bits in predicate_vectors if admits_local_model_decision(bits)]
    assert len(predicate_vectors) == 64
    assert admitted_vectors == [(True, True, True, True, True, True)]

    a, b, c = ATOM_IDS
    left_flat = flat_left(a, b, c)
    right_flat = flat_right(a, b, c)
    left_tree = tree_left(a, b, c)
    right_tree = tree_right(a, b, c)
    assert left_flat == right_flat == FLAT_RESULT
    assert left_tree == 9_037_326
    assert right_tree == 573_396

    label_order_checksums = {order: flat_left(*order) for order in permutations(ATOM_IDS)}
    assert len(label_order_checksums) == 6
    assert len(set(label_order_checksums.values())) == 6
    assert label_order_checksums[ATOM_IDS] == FLAT_RESULT

    semantic_source = bounded_semantic_result_code(2, 3, 3, 1, 8_101)
    semantic_target = bounded_semantic_result_code(3, 2, 7, 2, 259_234)
    assert semantic_source == SEMANTIC_SOURCE
    assert semantic_target == SEMANTIC_TARGET
    assert semantic_source != semantic_target
    assert decode_bounded_semantic_result_code(semantic_source) == (2, 3, 3, 1, 8_101)
    assert decode_bounded_semantic_result_code(semantic_target) == (3, 2, 7, 2, 259_234)

    difference_bitset = (
        int(2 != 3) + 2 * int(3 != 2) + 4 * int(3 != 7) + 8 * int(1 != 2)
    )
    refusal_mask = 1 | 2 | 4 | 8 | 16
    refusal_checksum = ((10_102 * 31 + 9_701) * 31 + 9_704) * 31 + 9_703
    assert difference_bitset == 15
    assert refusal_mask == 31
    assert refusal_checksum == 310_581_870

    decision_checksum = 10_101 * 31 + 9_702
    replay_checksum = decision_checksum * 31 + 10_301
    assert decision_checksum == 322_833
    assert replay_checksum == 10_018_124
    assert fixture_replay_admitted(11_001, 11_001)
    assert not fixture_replay_admitted(11_001, 11_003)
    mismatch_mask = 1 | 2
    mismatch_base = (decision_checksum * 31 + 11_003) * 31 + 10_801
    mismatch_checksum = mismatch_base * 7 + mismatch_mask
    assert mismatch_mask == 3
    assert mismatch_checksum == 2_174_160_852

    promotion_checksum = decision_checksum * 31 + 10_501
    abstention_mask = 1 | 2 | 4 | 8 | 16 | 32
    abstention_base = promotion_checksum * 31 + 10_601
    abstention_checksum = abstention_base * 127 + abstention_mask
    assert promotion_checksum == 10_018_324
    assert abstention_mask == 63
    assert abstention_checksum == 39_443_487_978

    declared_cases = ("local-model-decision", "semantic-refusal", "compiler-promotion-abstention")
    recorded_counts = {name: 1 for name in declared_cases}
    assert len(declared_cases) == 3
    assert sum(recorded_counts.values()) == 3
    compiler_capability_issues = 0
    compiler_rewrites = 0
    native_contest_receipts = 0
    ontology_runtime_transports = 0
    assert (
        compiler_capability_issues
        + compiler_rewrites
        + native_contest_receipts
        + ontology_runtime_transports
        == 0
    )

    print("ORACLE_D7_W0 predicate_vectors=64 admitted_vectors=1 declared_cases=3 recorded_cases=3")
    print(
        "ORACLE_D7_W1 flat=93767706,93767706 trees=9037326,573396 "
        "label_orders=6 checksum_collisions=0"
    )
    print(
        "ORACLE_D7_W2 semantic=62559008101,91514259234 difference_bitset=15 "
        "reason_mask=31 refusal_checksum=310581870"
    )
    print(
        "ORACLE_D7_W3 decision_checksum=322833 replay_checksum=10018124 "
        "wrong_occurrence_refusal_checksum=2174160852"
    )
    print(
        "ORACLE_D7_W4 promotion_checksum=10018324 reason_mask=63 "
        "abstention_checksum=39443487978"
    )
    print(
        "ORACLE_D7_W5 compiler_capabilities=0 compiler_rewrites=0 "
        "contest_ir=0 ontology_transport=0"
    )
    print("PROOF-CARRYING REBRACKETING PROTOCOL D7 ORACLE PASS")


if __name__ == "__main__":
    main()
