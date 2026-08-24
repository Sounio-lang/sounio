#!/usr/bin/env python3
"""
C2/C3 — Frozen corruption reference implementation (Rfam OctTree confirmatory).

Task A: flip corruption — exact port of the exploratory procedure
        (rfam_octtree_experiment.py, preserved at lane commit 1ceff5ab19).
Task B: balance-preserving corruption — top-level subsegment swap with
        mirror-complement fallback. The multiset of tokens is invariant by
        construction AND asserted at run time (fail closed).
Negative control: random Dyck words with matched length distribution.

Golden vectors: fixed seed 424242, fixed inputs, frozen outputs. The
confirmatory runner calls verify_golden() before any training and aborts on
mismatch — this is the executable half of the C2 task freeze.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

GOLDEN_SEED = 424242
GOLDEN_PATH = Path(__file__).parent / "golden_corruptions.json"

OPEN, CLOSE, DOT = 1, 2, 0


# ---------------------------------------------------------------
# Task A — flip corruption (exact exploratory port)
# ---------------------------------------------------------------

def flip_corrupt(bracket: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exact port of the exploratory corruption. Mutates a copy."""
    bracket = bracket.copy()
    n_flip = max(1, len(bracket) // 8)
    for _ in range(n_flip):
        r = rng.random()
        if r < 0.33 and (bracket == 1).any():
            pos = rng.choice(np.where(bracket == 1)[0])
            bracket[pos] = 2
        elif r < 0.66 and (bracket == 2).any():
            pos = rng.choice(np.where(bracket == 2)[0])
            bracket[pos] = 1
        elif (bracket == 0).any():
            pos = rng.choice(np.where(bracket == 0)[0])
            bracket[pos] = rng.choice([1, 2])
    return bracket


# ---------------------------------------------------------------
# Task B — balance-preserving corruption
# ---------------------------------------------------------------

def _matched_pairs(tokens: np.ndarray):
    """Stack-parse dot-bracket tokens. Returns (pairs dict, top-level segments)."""
    stack = []
    pairs = {}
    segments = []
    seg_start = None
    for i, t in enumerate(tokens):
        if t == OPEN:
            if not stack:
                seg_start = i
            stack.append(i)
        elif t == CLOSE and stack:
            j = stack.pop()
            pairs[j] = i
            if not stack:
                segments.append((seg_start, i))
    return pairs, segments


def swap_corrupt(tokens: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Balance-preserving corruption (C2 Task B).

    Primary: swap two distinct top-level substructures with length ratio in
    [0.5, 2] (candidate list in scan order, index drawn from rng).
    Fallback (fewer than 2 substructures): mirror-complement the longest
    internal matched substructure (reverse, then swap OPEN<->CLOSE).
    Asserts the token multiset is invariant; raises on violation.
    """
    out = tokens.copy()
    pairs, segments = _matched_pairs(tokens)

    if len(segments) >= 2:
        cands = []
        for a in range(len(segments)):
            for b in range(a + 1, len(segments)):
                la = segments[a][1] - segments[a][0] + 1
                lb = segments[b][1] - segments[b][0] + 1
                ratio = la / lb
                if 0.5 <= ratio <= 2.0:
                    cands.append((a, b))
        if cands:
            a, b = cands[rng.integers(len(cands))]
            (s1, e1), (s2, e2) = segments[a], segments[b]
            out = np.concatenate(
                [tokens[:s1], tokens[s2 : e2 + 1], tokens[e1 + 1 : s2],
                 tokens[s1 : e1 + 1], tokens[e2 + 1 :]]
            )
        else:
            out = _mirror_fallback(tokens, pairs, rng)
    else:
        out = _mirror_fallback(tokens, pairs, rng)

    _assert_count_invariant(tokens, out)
    return out


def _mirror_fallback(tokens: np.ndarray, pairs: dict, rng: np.random.Generator) -> np.ndarray:
    if not pairs:
        # No brackets at all: nothing structure-preserving to corrupt;
        # permutation of dots is a no-op, return unchanged copy.
        return tokens.copy()
    # Longest matched substructure; ties broken by lowest start (deterministic).
    opens = sorted(pairs.keys(), key=lambda j: (-(pairs[j] - j), j))
    j = opens[0]
    i = pairs[j]
    seg = tokens[j : i + 1][::-1].copy()
    seg = np.where(seg == OPEN, CLOSE, np.where(seg == CLOSE, OPEN, seg))
    out = np.concatenate([tokens[:j], seg, tokens[i + 1 :]])
    return out


def _assert_count_invariant(before: np.ndarray, after: np.ndarray) -> None:
    for sym in (OPEN, CLOSE, DOT):
        if int((before == sym).sum()) != int((after == sym).sum()):
            raise ValueError(
                f"Task B count-invariance violated for symbol {sym}: "
                f"{int((before == sym).sum())} -> {int((after == sym).sum())}"
            )
    if len(before) != len(after):
        raise ValueError("Task B length changed")


# ---------------------------------------------------------------
# Negative-control arm — random Dyck words
# ---------------------------------------------------------------

def gen_dyck_word(n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """UNIFORM random Dyck word of semilength n_pairs (cycle lemma).

    Amendment 2026-08-09 (pre-results): replaces the ( A ) B recursive
    sampler, whose distribution is NOT invariant under the Task B
    corruptions (mirror fallback reverses its size bias; the swap perturbs
    its order statistics), which made the NEG arm separable above chance
    (smoke seed 2026080900 / L=64: GRU 0.604). With a uniform sampler and
    both NEG classes conditioned on admitting a swap (rejection, see
    sample_arm), the two NEG classes are exactly identical in distribution
    by construction: uniform-Dyck is preserved by the segment swap
    (involution on words x qualifying pairs, qualifying-pair count
    invariant), so any measured NEG separation is an implementation
    artifact, as the freeze intends.

    Method: uniform arrangement of n+1 OPEN / n CLOSE; rotate to start
    after the LAST minimum of the partial-sum walk; drop the leading OPEN
    (Dvoretzky-Motzkin cycle lemma; each Dyck word has exactly 2n+1
    preimages, hence uniform).
    """
    if n_pairs < 1:
        return np.array([], dtype=np.int64)
    w = np.array([OPEN] * (n_pairs + 1) + [CLOSE] * n_pairs, dtype=np.int64)
    rng.shuffle(w)
    steps = np.where(w == OPEN, 1, -1)
    s = np.concatenate([[0], np.cumsum(steps)])  # S_0 .. S_{2n+1}
    m = int(np.max(np.nonzero(s == s.min())[0]))  # last minimizer
    w2 = np.concatenate([w[m:], w[:m]])
    if w2[0] != OPEN:  # cycle-lemma invariant; fail closed
        raise ValueError("uniform Dyck sampler: rotation does not start OPEN")
    out = w2[1:]
    bal = np.cumsum(np.where(out == OPEN, 1, -1))
    if not ((bal >= 0).all() and bal[-1] == 0):  # fail closed
        raise ValueError("uniform Dyck sampler produced a non-Dyck word")
    return out


def has_qualifying_pair(tokens: np.ndarray) -> bool:
    """True iff swap_corrupt would take the swap path (>= 2 top-level
    substructures with a length-ratio pair in [0.5, 2])."""
    _, segments = _matched_pairs(tokens)
    for a in range(len(segments)):
        for b in range(a + 1, len(segments)):
            la = segments[a][1] - segments[a][0] + 1
            lb = segments[b][1] - segments[b][0] + 1
            if 0.5 <= la / lb <= 2.0:
                return True
    return False


def verify_uniform_dyck(n_check: int = 4000) -> bool:
    """Self-test of the uniform sampler. Fail closed.

    n_pairs = 3 has C_3 = 5 Dyck words; a uniform sampler must hit each
    with frequency 1/5. Chi-square against uniformity at a deterministic
    seed; also checks Dyck validity (already asserted per-call).
    """
    from collections import Counter

    rng = np.random.default_rng(20260809)
    counts = Counter(_encode(gen_dyck_word(3, rng)) for _ in range(n_check))
    expected = {"((()))", "(()())", "(())()", "()(())", "()()()"}
    if set(counts) != expected:
        raise ValueError(f"uniform Dyck sampler: wrong support {set(counts)}")
    chi2 = sum((counts[w] - n_check / 5) ** 2 / (n_check / 5) for w in expected)
    if chi2 > 9.488:  # chi2(df=4) 95% — deterministic seed, deterministic pass
        raise ValueError(f"uniform Dyck sampler fails uniformity: chi2={chi2:.2f}")
    return True


# ---------------------------------------------------------------
# Golden vectors — the executable C2 freeze
# ---------------------------------------------------------------

_GOLDEN_INPUTS = [
    "()",
    "(())",
    "()(())",
    "(())((()))",
    "((.))((..))",
    "((((....))))",
    "()(())(()())(())",
    "..((..))..(())..",
]


def _encode(tokens: np.ndarray) -> str:
    return "".join({OPEN: "(", CLOSE: ")", DOT: "."}[int(t)] for t in tokens)


def _decode(s: str) -> np.ndarray:
    return np.array([{"(": OPEN, ")": CLOSE, ".": DOT}[c] for c in s], dtype=np.int64)


def generate_golden() -> dict:
    rng = np.random.default_rng(GOLDEN_SEED)
    vectors = []
    for s in _GOLDEN_INPUTS:
        tokens = _decode(s)
        a = flip_corrupt(tokens, rng)
        b = swap_corrupt(tokens, rng)
        vectors.append(
            {
                "input": s,
                "task_a_output": _encode(a),
                "task_b_output": _encode(b),
                "task_b_counts_invariant": all(
                    int((tokens == sym).sum()) == int((b == sym).sum())
                    for sym in (OPEN, CLOSE, DOT)
                ),
            }
        )
    raw = json.dumps(vectors, indent=2, sort_keys=True)
    return {
        "freeze_version": "1.0.0",
        "golden_seed": GOLDEN_SEED,
        "vectors": vectors,
        "vectors_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }


def verify_golden() -> bool:
    """Regenerate and compare against the frozen file. Fail closed."""
    if not GOLDEN_PATH.is_file():
        raise FileNotFoundError(f"golden vectors missing: {GOLDEN_PATH}")
    frozen = json.loads(GOLDEN_PATH.read_text())
    fresh = generate_golden()
    if frozen["vectors"] != fresh["vectors"]:
        raise ValueError("golden corruption vectors diverged from the C2 freeze")
    return True


def main():
    doc = generate_golden()
    GOLDEN_PATH.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"golden vectors written: {GOLDEN_PATH}")
    print(f"vectors_sha256={doc['vectors_sha256']}")
    verify_golden()
    print("verify_golden OK")


if __name__ == "__main__":
    main()
