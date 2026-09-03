#!/usr/bin/env python3
"""linalg parity vs mpmath (dps=30). Bit-exact matrix wire on stdin:
`<op> <case> <role> <i> <j> <val_bits>`. No scipy."""
import sys, struct, mpmath as mp
mp.mp.dps = 30
def b2f(t): return struct.unpack('<d', struct.pack('<q', int(t)))[0]
def _bits(x): return struct.unpack('<q', struct.pack('<d', x))[0]

def parse(stream):
    toks = [t for t in stream.split() if not t.startswith("#")]
    G = {}; k = 0
    while k + 6 <= len(toks):
        op, cs, role, i, j, vb = toks[k:k+6]; k += 6
        key = (op, int(cs)); G.setdefault(key, {}).setdefault(role, {})
        G[key][role][(int(i), int(j))] = int(vb) if role == "P" else b2f(vb)
    return G

def as_mat(rm):
    if not rm: return None
    R = max(i for i, _ in rm) + 1; C = max(j for _, j in rm) + 1
    return mp.matrix([[mp.mpf(rm.get((i, j), 0.0)) for j in range(C)] for i in range(R)])

def relerr(g, r):
    # Mixed abs/rel tolerance: floor the denominator at 1e-9 (not 1e-300) so that
    # legitimate O(1e-16) roundoff noise against a mathematically-exact-zero
    # reference entry (e.g. QtQ off-diagonal, or a permuted-zero in P·A) doesn't
    # blow up into a spurious ~1e283 "error". A genuine bug (>=1e-2 abs error on
    # a zero reference) still fails: 1e-2 / 1e-9 = 1e7, far past THR.
    return abs(g - r) / max(abs(r), mp.mpf('1e-9'))
def matmax_rel(G, R):
    m = mp.mpf(0)
    for i in range(R.rows):
        for j in range(R.cols):
            m = max(m, relerr(G[i, j], R[i, j]))
    return float(m)

def chk_det(d):   return float(relerr(mp.mpf(d["R"][(0,0)]), mp.det(as_mat(d["A"]))))
def chk_trace(d):
    A = as_mat(d["A"]); return float(relerr(mp.mpf(d["R"][(0,0)]), sum(A[i,i] for i in range(A.rows))))
def chk_fro(d):
    A = as_mat(d["A"]); s = mp.sqrt(sum(A[i,j]**2 for i in range(A.rows) for j in range(A.cols)))
    return float(relerr(mp.mpf(d["R"][(0,0)]), s))
def chk_transpose(d): return matmax_rel(as_mat(d["R"]), as_mat(d["A"]).T)
def chk_mul(d):   return matmax_rel(as_mat(d["R"]), as_mat(d["A"]) * as_mat(d["B"]))
def chk_inv(d):   return matmax_rel(as_mat(d["R"]), as_mat(d["A"])**-1)
def chk_solve(d): return matmax_rel(as_mat(d["A"]) * as_mat(d["R"]), as_mat(d["B"]))

def apply_piv(A, pivrole):
    # Phase-0 CONFIRMED: piv is a full permutation vector; P[k,piv[k]]=1
    #  ⟹ (P·A) row k = A row piv[k], and L·U == P·A.
    n = A.rows
    piv = [int(pivrole[(k, 0)]) for k in range(n)]
    return mp.matrix([[A[piv[k], j] for j in range(A.cols)] for k in range(n)])
def chk_lu(d):
    A = as_mat(d["A"]); L = as_mat(d["L"]); U = as_mat(d["U"])
    PA = apply_piv(A, d.get("P", {}))
    return matmax_rel(L * U, PA)
def chk_qr(d):
    A = as_mat(d["A"]); Q = as_mat(d["Q"]); R = as_mat(d["RR"])
    rec = matmax_rel(Q * R, A)
    QtQ = Q.T * Q; I = mp.eye(Q.cols)
    return max(rec, matmax_rel(QtQ, I))

def chk_eig(d):
    A = as_mat(d["A"]); n = A.rows
    got = sorted(mp.mpf(d["EVAL"][(i,0)]) for i in range(n))
    E, Q = mp.eigsy(A)   # symmetric: real eigenvalues ascending + orthonormal Q
    ref = sorted(mp.mpf(E[i]) for i in range(n))
    ev_err = max(float(relerr(g, r)) for g, r in zip(got, ref))
    V = as_mat(d["EVEC"]); res = 0.0
    for k in range(n):
        vk = mp.matrix([V[i, k] for i in range(n)])   # column k = eigenvector k (Phase-0)
        lam = mp.mpf(d["EVAL"][(k,0)])
        nrm = mp.norm(vk) or mp.mpf(1)
        res = max(res, float(mp.norm(A*vk - lam*vk) / nrm))
    return max(ev_err, res)

CHECKS = {"det":chk_det, "trace":chk_trace, "norm_fro":chk_fro, "transpose":chk_transpose,
          "mul":chk_mul, "inv":chk_inv, "solve":chk_solve, "lu":chk_lu, "qr":chk_qr,
          "eig":chk_eig}
THR = {op: 1e-2 for op in CHECKS}

def run(require_all=False):
    G = parse(sys.stdin.read()); per = {}
    for (op, cs), d in G.items():
        if op not in CHECKS: continue
        try: per.setdefault(op, []).append(CHECKS[op](d))
        except Exception: per.setdefault(op, []).append(float('inf'))
    fail = 0
    print(f"{'op':<12}{'cases':>7}{'max_rel_err':>16}  verdict")
    for op in CHECKS:
        errs = per.get(op, [])
        if not errs:
            print(f"{op:<12}{0:>7}{'NO DATA':>16}  {'FAIL(no-data)' if require_all else 'SKIP'}")
            if require_all: fail = 1
            continue
        mre = max(errs); thr = THR.get(op, 1e-2); ok = mre <= thr
        print(f"{op:<12}{len(errs):>7}{mre:>16.3e}  {'PASS' if ok else 'FAIL(thr=%.0e)'%thr}")
        if not ok: fail = 1
    print("LINALG_PARITY_OK" if not fail else "LINALG_PARITY_FAIL")
    return fail

def selftest():
    A=[[1,2],[3,4]]; B=[[5,6],[7,8]]; R=[[19,22],[43,50]]  # A·B
    lines=[]
    for role,M in (("A",A),("B",B),("R",R)):
        for i in range(2):
            for j in range(2):
                lines.append(f"mul 0 {role} {i} {j} {_bits(float(M[i][j]))}")
    import io; sys.stdin = io.StringIO("\n".join(lines)+"\n")
    assert run() == 0, "selftest mul should pass"
    print("LINALG_REF_SELFTEST_OK")

if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: sys.exit(run(require_all=("--require-all" in sys.argv)))
