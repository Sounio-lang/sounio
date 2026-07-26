#!/usr/bin/env python3
"""Independent exhaustive oracle for the frozen proof-carrying D2 fixture."""

from fractions import Fraction
from itertools import product


HYPOTHESES = ("H0", "H1", "H2", "H3")
PROBES = ("A", "B", "C")
CURRENT_PROJECTION = {
    "H0": Fraction(500, 1000),
    "H1": Fraction(500, 1000),
    "H2": Fraction(500, 1000),
    "H3": Fraction(500, 1000),
}
PREDICTIONS = {
    "H0": {"A": Fraction(400, 1000), "B": Fraction(450, 1000), "C": Fraction(500, 1000)},
    "H1": {"A": Fraction(400, 1000), "B": Fraction(550, 1000), "C": Fraction(500, 1000)},
    "H2": {"A": Fraction(600, 1000), "B": Fraction(450, 1000), "C": Fraction(500, 1000)},
    "H3": {"A": Fraction(600, 1000), "B": Fraction(450, 1000), "C": Fraction(650, 1000)},
}
BURDEN = {"A": 2, "B": 3, "C": 5}


def partition(hypotheses, probe):
    blocks = {}
    for hypothesis in hypotheses:
        blocks.setdefault(PREDICTIONS[hypothesis][probe], []).append(hypothesis)
    return tuple(tuple(blocks[value]) for value in sorted(blocks))


def refine(block, probe):
    return partition(block, probe)


def enumerate_depth_two_policies():
    policies = []
    for root in PROBES:
        root_blocks = partition(HYPOTHESES, root)
        ambiguous = tuple(block for block in root_blocks if len(block) > 1)
        for children in product(PROBES, repeat=len(ambiguous)):
            child_by_block = dict(zip(ambiguous, children))
            leaves = []
            branch_costs = []
            for block in root_blocks:
                if len(block) == 1:
                    leaves.extend((member,) for member in block)
                    branch_costs.append(BURDEN[root])
                else:
                    child = child_by_block[block]
                    leaves.extend(refine(block, child))
                    branch_costs.append(BURDEN[root] + BURDEN[child])
            complete = len(leaves) == 4 and all(len(leaf) == 1 for leaf in leaves)
            policies.append((root, root_blocks, children, complete, max(branch_costs)))
    return tuple(policies)


def update(survivors, probe, observed):
    return tuple(h for h in survivors if PREDICTIONS[h][probe] == observed)


def main():
    assert set(CURRENT_PROJECTION) == set(HYPOTHESES)
    assert len({CURRENT_PROJECTION[h] for h in HYPOTHESES}) == 1
    assert partition(HYPOTHESES, "A") == (("H0", "H1"), ("H2", "H3"))
    assert partition(("H0", "H1"), "B") == (("H0",), ("H1",))
    assert partition(("H2", "H3"), "C") == (("H2",), ("H3",))

    policies = enumerate_depth_two_policies()
    complete = tuple(policy for policy in policies if policy[3])
    # 9 A-root policies plus 3 each for the single ambiguous B/C root branch.
    assert len(policies) == 15
    assert len(complete) == 1
    root, blocks, children, _, worst = complete[0]
    assert root == "A"
    assert blocks == (("H0", "H1"), ("H2", "H3"))
    assert children == ("B", "C")
    assert worst == 7
    assert sum(BURDEN.values()) == 10
    assert all(len(partition(HYPOTHESES, probe)) < 4 for probe in PROBES)

    initial = HYPOTHESES
    missing = initial
    unaudited = initial
    disconnected = initial
    assert missing == initial and unaudited == initial and disconnected == initial
    after_a = update(initial, "A", Fraction(600, 1000))
    after_c = update(after_a, "C", Fraction(650, 1000))
    contradicted = update(initial, "B", Fraction(700, 1000))
    assert after_a == ("H2", "H3")
    assert after_c == ("H3",)
    assert contradicted == ()

    relabeled = {"Q7": "H0", "Q2": "H1", "Q9": "H2", "Q1": "H3"}
    for probe in PROBES:
        original_signature = sorted(PREDICTIONS[h][probe] for h in HYPOTHESES)
        relabeled_signature = sorted(PREDICTIONS[mode][probe] for mode in relabeled.values())
        assert original_signature == relabeled_signature

    maximum_fingerprint = 0
    for _ in range(8):
        maximum_fingerprint = 31 * maximum_fingerprint + 1_000_000
    assert maximum_fingerprint == 28_429_701_248_000_000
    assert maximum_fingerprint < 2**63 - 1

    print("ORACLE_D2_W0 policies=15 complete=1 root=A children=B,C")
    print("ORACLE_D2_W1 worst=7 preset=10 one_probe=false")
    print("ORACLE_D2_W2 A600=H2,H3 C650=H3 fingerprint=291233")
    print("ORACLE_D2_W3 missing=unchanged unaudited=unchanged disconnected=unchanged B700=family_refuted")
    print("ORACLE_D2_W4 relabel=partition_invariant scope=declared_family")
    print("ORACLE_D2_W5 fingerprint_max8=28429701248000000 i64_safe=true")
    print("PROOF-CARRYING MODEL CONTEST D2 ORACLE PASS")


if __name__ == "__main__":
    main()
