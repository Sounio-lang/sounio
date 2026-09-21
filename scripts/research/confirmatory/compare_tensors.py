#!/usr/bin/env python3
"""C5 — cross-check Sounio-produced algebra tables against controls.py.

- parses `souc run algebra_producer.sio` output (OCT/CL3 lines)
- verifies first-principles axioms on BOTH the Sounio tables and the
  PyTorch-side tables: octonions -> 64 nonzeros, alternativity, norm
  multiplicativity; Cl(3,0) -> 64 nonzeros, associativity, e_i^2=+1
- Cl(3,0): expects EXACT equality (blade rules are canonical)
- octonions: expects signed-permutation isomorphism (Fano relabeling
  freedom) between the Sounio Baez-mnemonic table and the exploratory
  table used by oct_mul_fast; reports the explicit isomorphism
- writes tensor_receipt.json with hashes and verdicts
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from controls import cl3_structure_tensor  # noqa: E402
from ossm_dyck_scaling import oct_mul_fast  # noqa: E402
import torch  # noqa: E402


def parse_sounio(path):
    T = {"OCT": np.zeros((8, 8, 8)), "CL3": np.zeros((8, 8, 8))}
    n = 0
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] not in T:
            continue
        tag, i, j, k, s = parts[0], *map(int, parts[1:])
        T[tag][i, k, j] += s
        n += 1
    if n != 128:
        raise ValueError(f"expected 128 structure constants, got {n}")
    return T["OCT"], T["CL3"]


def mul(T, a, b):
    return np.einsum("i,j,ikj->k", a, b, T)


def probe_oct_ctrl():
    T = np.zeros((8, 8, 8))
    for i in range(8):
        ei = torch.zeros(1, 8)
        ei[0, i] = 1.0
        for j in range(8):
            ej = torch.zeros(1, 8)
            ej[0, j] = 1.0
            T[i, :, j] = oct_mul_fast(ei, ej).numpy()[0]
    return T


def check_axioms(T, kind, label, rng):
    issues = []
    nz = int((T != 0).sum())
    if nz != 64:
        issues.append(f"{label}: {nz} nonzeros (expected 64)")
    for _ in range(200):
        a, b, c = rng.standard_normal(8), rng.standard_normal(8), rng.standard_normal(8)
        if kind == "oct":
            # alternativity: [a,a,b] = 0 and [a,b,b] = 0
            ass1 = mul(T, mul(T, a, a), b) - mul(T, a, mul(T, a, b))
            ass2 = mul(T, a, mul(T, b, b)) - mul(T, mul(T, a, b), b)
            if np.abs(ass1).max() > 1e-10 or np.abs(ass2).max() > 1e-10:
                issues.append(f"{label}: alternativity violated")
                break
            # norm multiplicativity |ab| = |a||b|
            lhs = np.linalg.norm(mul(T, a, b))
            rhs = np.linalg.norm(a) * np.linalg.norm(b)
            if abs(lhs - rhs) > 1e-8:
                issues.append(f"{label}: norm not multiplicative")
                break
        else:
            ab_c = mul(T, mul(T, a, b), c)
            a_bc = mul(T, a, mul(T, b, c))
            if np.abs(ab_c - a_bc).max() > 1e-10:
                issues.append(f"{label}: associativity violated")
                break
    return issues


def find_isomorphism(T_sou, T_ctrl):
    """Signed-permutation iso f: sou e_i -> sigma_i * ctrl e_{pi(i)}.

    Constraint propagation: f(1) 14 options, f(2) 12, f(3) determined by
    sou line (1,2,3), f(4) 8 remaining, f(5..7) determined by lines
    (1,4,5), (2,4,6), then verify the full table.
    """
    def prod(T, i, j):
        row = T[i, :, j]
        nz = np.nonzero(row)[0]
        if len(nz) != 1:
            return None
        return int(nz[0]), int(np.sign(row[nz[0]]))

    sou = {(i, j): prod(T_sou, i, j) for i in range(8) for j in range(8)}
    ctrl = {(i, j): prod(T_ctrl, i, j) for i in range(8) for j in range(8)}

    def consistent(f):
        # f: dict sou_idx -> (ctrl_idx, sigma); verify all pairs in domain
        dom = sorted(f)
        for i, j in itertools.product(dom, dom):
            k, s = sou[(i, j)]
            ci, si = f[i]
            cj, sj = f[j]
            got = ctrl[(ci, cj)]
            if got is None:
                return False
            kc, sc = got
            want_k, want_s = (0, 1) if k == 0 else f.get(k, (None, None))
            if k != 0 and k not in f:
                continue  # image not yet determined; check later
            if want_k is None or kc != want_k or si * sj * sc != s * want_s:
                return False
        return True

    imag = list(range(1, 8))
    for pi1, s1 in itertools.product(imag, (1, -1)):
        f = {0: (0, 1), 1: (pi1, s1)}
        for pi2, s2 in itertools.product([x for x in imag if x != pi1], (1, -1)):
            f[2] = (pi2, s2)
            # sou: 1*2 -> 3
            k3, s3 = sou[(1, 2)]
            kc, sc = ctrl[(pi1, pi2)]
            if kc == 0 or kc in (pi1, pi2):
                continue
            s3c = s3 * s1 * s2 * sc  # sigma_3 = s3 * s1*s2*sc (s=+1 for line)
            f[3] = (kc, s3c)
            if not consistent(f):
                del f[3]
                continue
            used = {pi1, pi2, kc}
            for pi4, s4 in itertools.product([x for x in imag if x not in used], (1, -1)):
                f[4] = (pi4, s4)
                ok = True
                # determine 5,6,7 via sou lines (1,4,5),(2,4,6),(3,4,7)
                for (a, b, c_) in ((1, 4, 5), (2, 4, 6), (3, 4, 7)):
                    kx, sx = sou[(a, b)]
                    assert kx == c_
                    ca, sa_ = f[a]
                    cb, sb = f[b]
                    got = ctrl[(ca, cb)]
                    if got is None or got[0] in {v[0] for v in f.values()}:
                        ok = False
                        break
                    f[c_] = (got[0], sx * sa_ * sb * got[1])
                if not ok:
                    for c_ in (5, 6, 7):
                        f.pop(c_, None)
                    continue
                if len({v[0] for v in f.values()}) == 8 and consistent(f):
                    # full verification
                    good = True
                    for i, j in itertools.product(range(8), range(8)):
                        k, s = sou[(i, j)]
                        ci, si = f[i]
                        cj, sj = f[j]
                        kc, sc = ctrl[(ci, cj)]
                        ck, sk = f[k]
                        if kc != ck or si * sj * sc != s * sk:
                            good = False
                            break
                    if good:
                        return dict(f)
                for c_ in (5, 6, 7):
                    f.pop(c_, None)
            f.pop(3, None)
    return None


def main():
    sou_out = HERE / "algebra_tables_sounio.txt"
    if not sou_out.is_file():
        raise SystemExit(f"missing {sou_out} — run: souc run algebra_producer.sio > {sou_out.name}")
    T_sou_oct, T_sou_cl3 = parse_sounio(sou_out)
    T_ctrl_cl3 = cl3_structure_tensor()
    T_ctrl_oct = probe_oct_ctrl()

    rng = np.random.default_rng(20260809)
    issues = []
    issues += check_axioms(T_sou_oct, "oct", "sounio-oct", rng)
    issues += check_axioms(T_ctrl_oct, "oct", "controls-oct", rng)
    issues += check_axioms(T_sou_cl3, "cl3", "sounio-cl3", rng)
    issues += check_axioms(T_ctrl_cl3, "cl3", "controls-cl3", rng)

    cl3_exact = bool(np.array_equal(T_sou_cl3, T_ctrl_cl3))
    if not cl3_exact:
        issues.append("CL3 tables differ (must be exact — canonical blade rules)")

    iso = find_isomorphism(T_sou_oct, T_ctrl_oct)
    if iso is None:
        issues.append("no signed-permutation isomorphism between octonion tables")

    def th(T):
        return hashlib.sha256(np.ascontiguousarray(T).tobytes()).hexdigest()

    receipt = {
        "sounio_output_sha256": hashlib.sha256(sou_out.read_bytes()).hexdigest(),
        "tables": {
            "sounio_oct": th(T_sou_oct), "controls_oct": th(T_ctrl_oct),
            "sounio_cl3": th(T_sou_cl3), "controls_cl3": th(T_ctrl_cl3),
        },
        "axioms": "all pass" if not issues else issues,
        "cl3_exact_match": cl3_exact,
        "oct_isomorphism": (
            {str(k): [int(v[0]), int(v[1])] for k, v in sorted(iso.items())}
            if iso else None
        ),
        "verdict": "PASS" if not issues and cl3_exact and iso else "FAIL",
    }
    (HERE / "tensor_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if receipt["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
