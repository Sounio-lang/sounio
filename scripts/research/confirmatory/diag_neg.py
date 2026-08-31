#!/usr/bin/env python3
"""NEG-arm leak diagnostic (pre-results, gate 6).

Hypothesis: swap_corrupt conditions negatives on the EXISTENCE of a
qualifying swappable pair (ratio in [0.5,2]) or sends them to the mirror
fallback, while positives are unconditional random Dyck words. The feature
"has qualifying pair" then separates classes above chance.

Measures, per (L, seed):
  positives (random Dyck):  P(>=2 segments), P(has qualifying pair)
  negatives (corrupted):    frac swap-path / fallback, P(has qualifying pair
                            in the PRE-corruption word)
  rule accuracy: predict negative iff has_qualifying_pair (the leak ceiling)
  Rfam Task B positives:    P(has qualifying pair) on real structures
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corruptions import (  # noqa: E402
    OPEN, CLOSE, gen_dyck_word, swap_corrupt, _matched_pairs,
)


def n_segments_and_qual(tokens):
    pairs, segments = _matched_pairs(tokens)
    qual = False
    for a in range(len(segments)):
        for b in range(a + 1, len(segments)):
            la = segments[a][1] - segments[a][0] + 1
            lb = segments[b][1] - segments[b][0] + 1
            r = la / lb
            if 0.5 <= r <= 2.0:
                qual = True
                break
        if qual:
            break
    return len(segments), qual


def diag_random_dyck(L, seed, n=20000):
    rng = np.random.default_rng(seed)
    pos_seg1, pos_qual = 0, 0
    neg_swap, neg_fallback, neg_pre_qual = 0, 0, 0
    neg_post_qual = 0
    for _ in range(n):
        n_pairs = max(1, L // 2)
        w = gen_dyck_word(n_pairs, rng)
        ns, q = n_segments_and_qual(w)
        # positive arm
        pos_seg1 += int(ns < 2)
        pos_qual += int(q)
        # negative arm: same distribution pre-corruption
        w2 = gen_dyck_word(n_pairs, rng)
        ns2, q2 = n_segments_and_qual(w2)
        neg_pre_qual += int(q2)
        c = swap_corrupt(w2, rng)
        took_fallback = not (ns2 >= 2 and q2)
        neg_fallback += int(took_fallback)
        neg_swap += int(not took_fallback)
        _, qpost = n_segments_and_qual(c)
        neg_post_qual += int(qpost)
    p_pos = pos_qual / n
    p_neg = neg_post_qual / n
    rule_acc = 0.5 * p_neg + 0.5 * (1 - p_pos)
    return dict(
        L=L, seed=seed, n=n,
        pos_p_lt2seg=pos_seg1 / n,
        pos_p_qual=p_pos,
        neg_frac_fallback=neg_fallback / n,
        neg_pre_p_qual=neg_pre_qual / n,
        neg_post_p_qual=p_neg,
        qualrule_acc=rule_acc,
    )


def diag_rfam(L, seed, n=20000):
    """P(qualifying pair) on real Rfam structures from the frozen TEST pool."""
    import gzip
    here = Path(__file__).resolve().parent
    recs = here / "freeze" / "records.tsv.gz"
    rng = np.random.default_rng(seed)
    lens, structs = [], []
    with gzip.open(recs, "rt") as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            # columns: id, family, clan_group, split, len, structure (C1 layout)
            if len(parts) < 6:
                continue
            rid, fam, clan, split, ln, st = parts[:6]
            if split != "test":
                continue
            ln = int(ln)
            if L // 4 <= ln <= L:
                structs.append(st)
    if not structs:
        return dict(L=L, rfam_pool=0)
    take = min(n, len(structs))
    idx = rng.choice(len(structs), size=take, replace=len(structs) < n)
    qual = 0
    seg1 = 0
    for i in idx:
        t = np.array([{ "(": OPEN, ")": CLOSE }.get(c, 0) for c in structs[i]],
                     dtype=np.int64)
        ns, q = n_segments_and_qual(t)
        qual += int(q)
        seg1 += int(ns < 2)
    return dict(L=L, rfam_pool=len(structs), rfam_p_qual=qual / take,
                rfam_p_lt2seg=seg1 / take)


def main():
    print("== random Dyck (NEG arm population) ==")
    for L in (64, 128, 256, 512):
        for seed in (2026080900, 2026080901, 2026080902):
            d = diag_random_dyck(L, seed, n=10000)
            print(f"L={L:4d} s={d['seed']} pos_qual={d['pos_p_qual']:.3f} "
                  f"pos_<2seg={d['pos_p_lt2seg']:.3f} "
                  f"neg_fallback={d['neg_frac_fallback']:.3f} "
                  f"neg_post_qual={d['neg_post_p_qual']:.3f} "
                  f"=> qual-rule acc ~ {d['qualrule_acc']:.3f}")
    print("== Rfam test pool (Task B positives) ==")
    for L in (64, 128, 256, 512):
        d = diag_rfam(L, 2026080900)
        print(d)


if __name__ == "__main__":
    main()
