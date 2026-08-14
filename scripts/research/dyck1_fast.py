#!/usr/bin/env python3
"""
Fast vectorized Dyck-1 generator — corrected and non-trivially-separable.

Corrects the audit bugs:
  - Valid: proper Dyck words (balanced, never negative, depth ends at 0)
  - Invalid: valid word with one (↔) transposition that breaks validity
  - Both classes have IDENTICAL ( count and P(token0 = ()

Key insight: swapping (→) changes the total count, making count classifier perfect.
Instead, swap a ( at position i with a ) at position j.  Count is preserved,
order is broken.
"""

import numpy as np


def gen_valid_dyck_fast(length, n, rng):
    """Generate n valid Dyck-1 words of given length (vectorized)."""
    assert length % 2 == 0
    tokens = np.zeros((n, length), dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)

    for t in range(length):
        remaining = length - t - 1
        must_open = (depth == 0)
        must_close = (depth >= remaining) & (~must_open)
        rand_open = rng.random(n) < 0.5
        do_open = must_open | (~must_close & rand_open)
        tokens[:, t] = np.where(do_open, 1, 2)
        depth += np.where(do_open, 1, -1)

    return tokens


def _check_validity_batch(tokens):
    """Check Dyck validity for each row. Returns bool array (True = valid)."""
    n, L = tokens.shape
    depth = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    for t in range(L):
        depth += np.where(tokens[:, t] == 1, 1, -1)
        valid &= (depth >= 0)
    valid &= (depth == 0)
    return valid


def _corrupt_one(seq, rng, length):
    """Corrupt a valid Dyck word by transposing a ( at i with a ) at j.
    
    Preserves total count of ( and ).  Tries multiple pairs until one
    breaks validity.  Returns the corrupted seq (or None if all attempts fail).
    """
    open_pos = np.where(seq == 1)[0]
    close_pos = np.where(seq == 2)[0]
    # Exclude position 0 from opens to keep P(token0='(') = 1
    open_pos = open_pos[open_pos > 0]

    if len(open_pos) == 0 or len(close_pos) == 0:
        return None

    for _ in range(30):
        i = rng.choice(open_pos)
        j = rng.choice(close_pos)
        if i == j:
            continue
        candidate = seq.copy()
        candidate[i] = 2  # was (
        candidate[j] = 1  # was )
        # Quick validity check
        d = 0; ok = True
        for t in range(length):
            d += 1 if candidate[t] == 1 else -1
            if d < 0:
                ok = False; break
        if d != 0:
            ok = False
        if not ok:
            return candidate
    return None


def gen_dyck1_fast(length, batch, rng):
    """Generate non-trivially-separable Dyck-1 dataset.

    Valid: proper Dyck words.
    Invalid: transpose ( at i ↔ ) at j in a valid word, breaking validity.
    Both classes have IDENTICAL count of ( — count classifier is at chance.
    """
    if length % 2 != 0:
        length += 1

    n_valid = batch // 2
    n_invalid = batch - n_valid

    valid = gen_valid_dyck_fast(length, n_valid, rng)

    invalid = valid[:n_invalid].copy()
    for i in range(n_invalid):
        corrupted = _corrupt_one(invalid[i], rng, length)
        if corrupted is not None:
            invalid[i] = corrupted

    tokens = np.vstack([valid, invalid])
    labels = np.concatenate([np.ones(n_valid), np.zeros(n_invalid)])
    perm = rng.permutation(batch)
    return tokens[perm], labels[perm]


def verify_generator(rng, length=64, batch=4000):
    """Verify the corrected generator produces non-trivially-separable classes."""
    tokens, labels = gen_dyck1_fast(length, batch, rng)

    valid_mask = labels == 1
    invalid_mask = labels == 0
    valid_tokens = tokens[valid_mask]
    invalid_tokens = tokens[invalid_mask]

    v_valid = _check_validity_batch(valid_tokens)
    i_valid = _check_validity_batch(invalid_tokens)

    valid_frac_open = np.mean(valid_tokens == 1)
    invalid_frac_open = np.mean(invalid_tokens == 1)
    valid_first_open = np.mean(valid_tokens[:, 0] == 1)
    invalid_first_open = np.mean(invalid_tokens[:, 0] == 1)

    # Count classifier
    opens = (tokens == 1).sum(axis=1)
    threshold = np.median(opens)
    pred = (opens >= threshold).astype(int)
    count_acc = max(np.mean(pred == labels), 1 - np.mean(pred == labels))

    n_distinct = len(np.unique(tokens, axis=0))

    print(f"Dyck-1 generator verification (L={length}, n={batch}):")
    print(f"  Valid samples that are genuinely Dyck:  {v_valid.sum()}/{len(v_valid)}")
    print(f"  Invalid samples accidentally valid:     {i_valid.sum()}/{len(i_valid)}")
    print(f"  Fraction of '(' in valid:   {valid_frac_open:.4f}")
    print(f"  Fraction of '(' in invalid: {invalid_frac_open:.4f}")
    print(f"  P(token0='(') in valid:   {valid_first_open:.4f}")
    print(f"  P(token0='(') in invalid: {invalid_first_open:.4f}")
    print(f"  Count classifier accuracy: {count_acc:.4f} (should be ~0.5)")
    print(f"  Distinct sequences: {n_distinct}/{batch}")

    ok = (v_valid.sum() == len(v_valid) and
          i_valid.sum() < len(i_valid) * 0.05 and
          abs(valid_frac_open - invalid_frac_open) < 0.01 and
          abs(valid_first_open - invalid_first_open) < 0.1 and
          count_acc < 0.55 and
          n_distinct > batch * 0.9)
    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == '__main__':
    rng = np.random.default_rng(20260806)
    for L in [32, 64, 128]:
        verify_generator(rng, L, 4000)
        print()
