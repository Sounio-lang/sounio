"""Canonical CYP450 ↔ Fano membership for FAERS / 168-theorem work.

MUST match stdlib/medical/cyp450_fano.sio (and the medical dissertation encoding):

  e1=CYP1A2, e2=CYP2C9, e3=CYP2C8, e4=CYP2B6, e5=CYP2C19, e6=CYP2D6, e7=CYP3A4
  Fano lines: (1,2,4) (2,3,5) (3,4,6) (4,5,7) (5,6,1) (6,7,2) (7,1,3)

Do NOT use alternate Cayley–Dickson labelings from other modules without an
explicit isomorphism map — they are disjoint as unordered triples.
"""

from __future__ import annotations

from typing import Iterable

CYP_NAMES = {
    1: "CYP1A2",
    2: "CYP2C9",
    3: "CYP2C8",
    4: "CYP2B6",
    5: "CYP2C19",
    6: "CYP2D6",
    7: "CYP3A4",
}

CYP_INDEX = {v: k for k, v in CYP_NAMES.items()}

# Unordered lines — membership is order-invariant.
FANO_LINES: frozenset[frozenset[int]] = frozenset(
    frozenset(line)
    for line in (
        (1, 2, 4),
        (2, 3, 5),
        (3, 4, 6),
        (4, 5, 7),
        (5, 6, 1),
        (6, 7, 2),
        (7, 1, 3),
    )
)

# Sorted triples for exact ordered-tuple checks when (i,j,k) with i<j<k.
FANO_TRIPLES_SORTED: frozenset[tuple[int, int, int]] = frozenset(
    tuple(sorted(line)) for line in FANO_LINES  # type: ignore[arg-type]
)


def is_fano_ids(a: int, b: int, c: int) -> bool:
    return frozenset((a, b, c)) in FANO_LINES


def is_fano_names(cyp_a: str, cyp_b: str, cyp_c: str) -> bool:
    return is_fano_ids(CYP_INDEX[cyp_a], CYP_INDEX[cyp_b], CYP_INDEX[cyp_c])


def fano_flag(a: int, b: int, c: int) -> str:
    return "True" if is_fano_ids(a, b, c) else "False"


def relabel_csv_rows(rows: Iterable[dict], *, a="cyp_a", b="cyp_b", c="cyp_c", fano="fano") -> list[dict]:
    out = []
    flips = 0
    for r in rows:
        true = is_fano_names(r[a], r[b], r[c])
        old = str(r.get(fano, "")).lower() in ("true", "1", "yes")
        if old != true:
            flips += 1
        nr = dict(r)
        nr[fano] = "True" if true else "False"
        out.append(nr)
    return out, flips
