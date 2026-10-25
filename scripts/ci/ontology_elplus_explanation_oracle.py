#!/usr/bin/env python3
"""Independently validate reconstructed EL+ derivation DAGs.

The input is the stdout of tests/run-pass/ontology_elplus_explanation.sio.
Every premise must occur earlier in the trace, and every rule's witness must
reconstruct exactly the cell it justifies. No Sounio code is imported.
"""

from __future__ import annotations

import re
import sys

MAXC = 64
ckind = {8: 1, 9: 2, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 2}
carg1 = {9: 2, 10: 1, 11: 2, 12: 1, 13: 0, 14: 0, 15: 0, 16: 1}
carg2 = {9: 0, 10: 9, 11: 3, 12: 2, 13: 9, 14: 3, 15: 2, 16: 0}
exid = {(carg1[c], carg2[c]): c for c in (10, 11, 12, 13, 14, 15)}
stated = {(4, 0), (0, 1), (2, 3), (4, 10), (2, 11), (7, 1), (7, 5), (6, 0)}
role_closed = {(0, 0), (1, 0), (1, 1), (2, 2)}
chains = [(1, 2, 0)]
step_re = re.compile(r"^STEP (\d+) (\d+) (\d+) (\d+)$")


def validate_trace(name: str, steps: list[tuple[int, int, int, int]]) -> None:
    seen: set[tuple[int, int, int]] = set()
    for kind, cell, rule, witness in steps:
        if kind == 0:
            a, b = divmod(cell, MAXC)
            if rule == 0:
                assert (a, b) in stated, (name, "bad stated leaf", cell)
            elif rule == 1:
                assert a == b, (name, "bad reflexive leaf", cell)
            elif rule == 2:
                assert b == 8, (name, "bad top leaf", cell)
            elif rule == 3:
                assert (0, a, witness) in seen and (0, witness, b) in seen, (name, "transitivity", cell)
            elif rule in (4, 5):
                assert ckind.get(witness) == 2, (name, "conjunction shape", cell)
                expected = carg1[witness] if rule == 4 else carg2[witness]
                assert b == expected and (0, a, witness) in seen, (name, "conjunction elimination", cell)
            elif rule == 6:
                assert b == witness and ckind.get(witness) == 2, (name, "conjunction introduction shape", cell)
                assert (0, a, carg1[witness]) in seen and (0, a, carg2[witness]) in seen, (name, "conjunction introduction", cell)
            elif rule == 11:
                role_filler = divmod(witness, MAXC)
                assert (1, *role_filler) in seen, (name, "RtoS premise", cell)
                role, source = divmod(role_filler[0], MAXC)
                assert exid.get((role, role_filler[1])) == b, (name, "RtoS target", cell)
            else:
                raise AssertionError((name, "unknown subsumption rule", rule))
        else:
            a, filler = divmod(cell, MAXC)
            role, source = divmod(a, MAXC)
            if rule == 7:
                assert ckind.get(witness) == 3, (name, "stoR shape", cell)
                assert carg1[witness] == role and carg2[witness] == filler, (name, "stoR target", cell)
                assert (0, source, witness) in seen, (name, "stoR premise", cell)
            elif rule == 8:
                old_filler, new_filler = divmod(witness, MAXC)
                assert filler == new_filler, (name, "Rmono target", cell)
                assert (1, a, old_filler) in seen and (0, old_filler, new_filler) in seen, (name, "Rmono premises", cell)
            elif rule == 9:
                assert (witness, role) in role_closed, (name, "roleSub hierarchy", cell)
                assert (1, witness * MAXC + source, filler) in seen, (name, "roleSub premise", cell)
            elif rule == 10:
                chain_index, mid = divmod(witness, MAXC)
                r1, r2, r3 = chains[chain_index]
                assert role == r3, (name, "roleComp target", cell)
                assert (1, r1 * MAXC + source, mid) in seen, (name, "roleComp left", cell)
                assert (1, r2 * MAXC + mid, filler) in seen, (name, "roleComp right", cell)
            else:
                raise AssertionError((name, "unknown role rule", rule))
        seen.add((kind, cell // MAXC, cell % MAXC))


def main() -> None:
    text = sys.stdin.read()
    traces: dict[str, list[tuple[int, int, int, int]]] = {}
    for line in text.splitlines():
        begin = re.match(r"^(TRACE_[A-Z_]+)_BEGIN$", line)
        if begin:
            current = begin.group(1)
            assert current not in traces
            traces[current] = []
            continue
        end = re.match(r"^(TRACE_[A-Z_]+)_END$", line)
        if end:
            current = None
            continue
        match = step_re.match(line)
        if match:
            traces[current].append(tuple(map(int, match.groups())))

    expected = {
        "TRACE_TRANSITIVE": (0, 4 * MAXC + 1, 3),
        "TRACE_ROLESUB": (0, 4 * MAXC + 13, 11),
        "TRACE_COMPOSE": (0, 4 * MAXC + 14, 11),
        "TRACE_CONJ": (0, 4 * MAXC + 16, 6),
        "TRACE_EDGE": (1, (0 * MAXC + 4) * MAXC + 3, 10),
        "TRACE_CONJ_ELIM": (0, 9 * MAXC + 0, 5),
        "TRACE_TOP": (0, 4 * MAXC + 8, 2),
    }
    assert set(traces) == set(expected), (set(traces), set(expected))
    all_rules = set()
    total_steps = 0
    for name, steps in traces.items():
        assert steps, name
        validate_trace(name, steps)
        kind, cell, rule = expected[name]
        assert steps[-1] == (kind, cell, rule, steps[-1][3]), (name, steps[-1])
        all_rules.update(step[2] for step in steps)
        total_steps += len(steps)
    assert all_rules == set(range(12)), f"incomplete rule coverage: {sorted(all_rules)}"
    assert "ONTOLOGY_ELPLUS_EXPLANATION_OK" in text
    print(f"ONTOLOGY_ELPLUS_PROOF_ORACLE_OK traces={len(traces)} steps={total_steps} rules=0..11")


if __name__ == "__main__":
    main()
