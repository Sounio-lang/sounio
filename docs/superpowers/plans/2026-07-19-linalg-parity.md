<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-19-linalg-parity
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-19-linalg-parity
-->

# linalg↔mpmath Parity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Per-operation accuracy map of `stdlib/linalg` `matnm` ops + `eigen_symmetric` vs mpmath (dps=30), behind a gate, fixing root-caused defects the map surfaces. Decompositions checked by reconstruction invariant.

**Architecture:** Extends the special/stats bit-exact parity harness to matrices. A Sounio emitter constructs test `MatNM`s, runs each op, and prints every input/output/factor matrix element as `<op> <case> <role> <i> <j> <val_bits>`. A Python comparator reconstructs the matrices, computes the mpmath reference, and checks element-wise (well-defined ops) or via reconstruction (LU: L·U=P·A; QR: Q·R=A; eig: eigenvalues-as-set + residual).

**Tech Stack:** Sounio (`stdlib/linalg`, lean_single), Python 3 + mpmath, bash.

**Spec:** `docs/superpowers/specs/2026-07-19-linalg-parity-design.md`

---

## Confirmed interfaces (read during planning)

- **MatNM** `{ data: [f64;4096], rows: i64, cols: i64 }`; `matnm_get(m,i,j) = m.data[i*64+j]` (**row-major, storage stride 64**). `matnm_set(m,i,j,v)->MatNM` (functional). `matnm_new/zeros(r,c)`, `matnm_identity(n)`.
- Ops: `matnm_mul/add/sub(a,b)->MatNM`, `matnm_scale(m,s)`, `matnm_transpose(m)`, `matnm_norm_fro(m)->f64`, `matnm_trace(m)->f64`, `matnm_det(m)->f64`, `matnm_inv(m)->MatNM`, `matnm_solve(a,b)->MatNM`, `matnm_lu(m)->LUResult{l,u,piv:[i64;64],n,sign}`, `matnm_qr(a)->QRResult_NM{q,r,rows,cols}`.
- **eigen_symmetric(mat: &![f64;65536], n: i32, eigenvalues: &![f64;256], eigenvectors: &![f64;65536]) -> bool** — flat arrays, stride n (`mat[i*n+j]`), module-level `&!` out-params (see `tests/stdlib/linalg/test_eigen_e2e.sio`).

## Wire format

`<op> <case> <role> <i> <j> <val_bits>` — `val_bits` = `f64_to_bits(value)` (signed i64), EXCEPT role `P` where it's the integer pivot value printed directly. Roles: `A`,`B` (inputs), `R` (result / scalar at 0,0), `L`,`U`,`P` (LU), `Q`,`RR` (QR), `EVAL`,`EVEC` (eigen). Self-describing (inputs emitted too). Reuses `f64_to_bits`/`print_int` builtins; `print_int` appends a newline → comparator tokenizes the whole stream.

## File structure

- Create `stdlib/parity/emit_mat.sio` — `emit_mat(op,case,role,m)` + `emit_scalar(op,case,val)`.
- Create `scripts/parity/linalg_parity_ref.py` — comparator with reconstruction invariants.
- Create `scripts/linalg_parity_gate.sh` — orchestrator.
- Create `tests/parity/linalg_parity_matnm.sio` — tier 1+2 (+ tier 3 lu/qr).
- Create `tests/parity/linalg_parity_eigen.sio` — eigen_symmetric.
- Create `docs/research/2026-07-19-linalg-parity.md` — report.
- Reuse `stdlib/parity/emit.sio` unchanged.

---

## Task 0: Phase-0 confirmations

**Files:** Create `tests/parity/linalg_probe.sio`.

- [ ] **Step 1 — matrix bridge + row-major + by-value param.** Probe:
```
//@ run-pass
use linalg::matnm::*
fn main() -> i32 with IO, Mut, Div, Panic {
    var m = matnm_zeros(2, 2)
    m = matnm_set(m, 0, 0, 1.5)
    m = matnm_set(m, 0, 1, 2.5)
    m = matnm_set(m, 1, 0, 3.5)
    m = matnm_set(m, 1, 1, 4.5)
    // read back row-major; bit each on a local
    var i = 0
    while i < 2 {
        var j = 0
        while j < 2 {
            let v = matnm_get(m, i, j)
            let vb = f64_to_bits(v)
            print("E "); print_int(i); print(" "); print_int(j); print(" "); print_int(vb); print("\n")
            j = j + 1
        }
        i = i + 1
    }
    return 0
}
```
Run under `SOUNIO_SOUC_ENGINE=lean_single`. Decode: E(0,0)=1.5, E(0,1)=2.5, E(1,0)=3.5, E(1,1)=4.5 (confirms row-major + set/get + bridge). **If it segfaults or a MatNM-by-value call fails**, note it — the `emit_mat` helper may need to take the matrix differently (or the loop inlined in main). Record.

- [ ] **Step 2 — matnm_lu piv encoding.** Compute `matnm_lu` of a 3×3 matrix that forces a pivot (e.g. `[[0,1,1],[1,0,1],[1,1,0]]` — zero pivot at (0,0) forces a row swap), emit `l`,`u` elements and the `piv` array (as ints). In Python, decode piv and figure out whether it's (a) a sequence of row-swap partners `piv[k]`=row swapped with k at step k, or (b) a full permutation vector. Determine which makes `L·U == P·A`. **Record the convention** — the comparator depends on it. (Fallback if ambiguous: check `L·U` equals A with rows permuted to SOME permutation — i.e. row-multiset equality — and note piv wasn't decoded.)

- [ ] **Step 3 — eigen_symmetric eigenvector layout.** Run `eigen_symmetric` on the exact 2×2 `[[2,1],[1,2]]` (eigenvalues 1 and 3; eigenvectors ∝ [1,-1]/√2 and [1,1]/√2). Emit `EVAL[i]` and the full `EVEC` flat array. In Python determine whether eigenvector k is **row k** or **column k** of the n×n EVEC block (stride n). **Record the layout** — the #1 convention risk.

- [ ] **Step 4 — commit** the probe:
```bash
git add tests/parity/linalg_probe.sio
git commit -m "test(parity): linalg Phase-0 probe (matrix bridge, row-major, piv encoding, eigenvector layout)"
```
Report all four findings precisely (they parameterize the comparator).

---

## Task 1: emit helper + comparator core (scalar + element-wise)

**Files:** Create `stdlib/parity/emit_mat.sio`, `scripts/parity/linalg_parity_ref.py`.

- [ ] **Step 1 — `stdlib/parity/emit_mat.sio`** (bit values on LOCALS; `print_int` is a builtin):
```
// Matrix parity emit. line: <op> <case> <role> <i> <j> <val_bits>
use linalg::matnm::*

pub fn emit_scalar(op: string, cs: i64, val: f64) with IO, Mut, Div, Panic {
    let vb = f64_to_bits(val)
    print(op); print(" "); print_int(cs); print(" R 0 0 "); print_int(vb); print("\n")
}

pub fn emit_mat(op: string, cs: i64, role: string, m: MatNM) with IO, Mut, Div, Panic {
    var i = 0
    while i < m.rows {
        var j = 0
        while j < m.cols {
            let v = matnm_get(m, i, j)
            let vb = f64_to_bits(v)
            print(op); print(" "); print_int(cs); print(" "); print(role); print(" ")
            print_int(i); print(" "); print_int(j); print(" "); print_int(vb); print("\n")
            j = j + 1
        }
        i = i + 1
    }
}
```
If `emit_mat` taking `MatNM` by value fails to compile/run (Phase-0 Step 1), fall back to inlining the emit loop per matrix in the emitter's `main` (report the change).

- [ ] **Step 2 — comparator `scripts/parity/linalg_parity_ref.py`** (scalar + element-wise ops first; LU/QR/eig added in Tasks 3/4). Whole-stream tokenizer, 6 tokens/line:
```python
#!/usr/bin/env python3
"""linalg parity vs mpmath (dps=30). Bit-exact matrix wire on stdin:
`<op> <case> <role> <i> <j> <val_bits>`. No scipy."""
import sys, struct, mpmath as mp
mp.mp.dps = 30
def b2f(t): return struct.unpack('<d', struct.pack('<q', int(t)))[0]
def _bits(x): return struct.unpack('<q', struct.pack('<d', x))[0]

def parse(stream):
    # -> {(op,case): {role: {(i,j): value_or_int}}}
    toks = [t for t in stream.split() if not t.startswith("#")]
    G = {}
    k = 0
    while k + 6 <= len(toks):
        op, cs, role, i, j, vb = toks[k:k+6]; k += 6
        key = (op, int(cs)); G.setdefault(key, {}).setdefault(role, {})
        # role P carries an integer pivot, not f64 bits
        G[key][role][(int(i), int(j))] = int(vb) if role == "P" else b2f(vb)
    return G

def as_mat(role_map):
    if not role_map: return None
    R = max(i for i, _ in role_map) + 1
    C = max(j for _, j in role_map) + 1
    return mp.matrix([[mp.mpf(role_map.get((i, j), 0.0)) for j in range(C)] for i in range(R)])

def relerr(got, ref):
    return abs(got - ref) / max(abs(ref), mp.mpf('1e-300'))

def matmax_rel(GOT, REF):
    m = mp.mpf(0)
    for i in range(REF.rows):
        for j in range(REF.cols):
            m = max(m, relerr(GOT[i, j], REF[i, j]))
    return float(m)

# op -> reference check: returns max_rel_err given the role dict of one case.
def chk_det(d):   return float(relerr(mp.mpf(d["R"][(0,0)]), mp.det(as_mat(d["A"]))))
def chk_trace(d):
    A = as_mat(d["A"]); return float(relerr(mp.mpf(d["R"][(0,0)]), sum(A[i,i] for i in range(A.rows))))
def chk_fro(d):
    A = as_mat(d["A"]); s = mp.sqrt(sum(A[i,j]**2 for i in range(A.rows) for j in range(A.cols)))
    return float(relerr(mp.mpf(d["R"][(0,0)]), s))
def chk_transpose(d):
    A = as_mat(d["A"]); return matmax_rel(as_mat(d["R"]), A.T)
def chk_mul(d):   return matmax_rel(as_mat(d["R"]), as_mat(d["A"]) * as_mat(d["B"]))
def chk_inv(d):   return matmax_rel(as_mat(d["R"]), as_mat(d["A"])**-1)
def chk_solve(d):
    A = as_mat(d["A"]); x = as_mat(d["R"]); b = as_mat(d["B"])
    return matmax_rel(A * x, b)   # residual A·x = b

CHECKS = {"det":chk_det, "trace":chk_trace, "norm_fro":chk_fro,
          "transpose":chk_transpose, "mul":chk_mul, "inv":chk_inv, "solve":chk_solve}
THR = {op: 1e-2 for op in CHECKS}   # LU/QR/eig thresholds added in later tasks

def run(require_all=False, ops=None):
    G = parse(sys.stdin.read())
    ops = ops or list(CHECKS)
    per = {}   # op -> list of max_rel_err across cases
    for (op, cs), d in G.items():
        if op not in CHECKS: continue
        try: per.setdefault(op, []).append(CHECKS[op](d))
        except Exception as e: per.setdefault(op, []).append(float('inf'))
    fail = 0
    print(f"{'op':<12}{'cases':>7}{'max_rel_err':>16}  verdict")
    for op in ops:
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
    # 2x2 mul: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    lines = []
    A = [[1,2],[3,4]]; B = [[5,6],[7,8]]; R = [[19,22],[43,50]]
    for role, M in (("A",A),("B",B),("R",R)):
        for i in range(2):
            for j in range(2):
                lines.append(f"mul 0 {role} {i} {j} {_bits(float(M[i][j]))}")
    import io; sys.stdin = io.StringIO("\n".join(lines) + "\n")
    assert run() == 0, "selftest mul should pass"
    print("LINALG_REF_SELFTEST_OK")

if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: sys.exit(run(require_all=("--require-all" in sys.argv)))
```

- [ ] **Step 3 — selftest:** `python3 scripts/parity/linalg_parity_ref.py --selftest` → `LINALG_REF_SELFTEST_OK`.
- [ ] **Step 4 — negative:** feed a wrong `mul` R element → `LINALG_PARITY_FAIL` (exit 1).
- [ ] **Step 5 — commit** `feat(parity): linalg emit_mat helper + comparator core (scalar/element-wise)`.

---

## Task 2: matnm tier 1+2 emitter + gate (proving slice)

**Files:** Create `tests/parity/linalg_parity_matnm.sio`, `scripts/linalg_parity_gate.sh`.

- [ ] **Step 1 — emitter** (tier 1 scalar + tier 2 well-defined). For each test case build A (and B / b as needed), run the op, `emit_mat`/`emit_scalar` the inputs and result. Use ≥3 cases: a general invertible 3×3, a 2×2, and an SPD 3×3. Example structure (fill all ops):
```
//@ run-pass
use linalg::matnm::*
use parity::emit_mat::*

fn set4(m0: MatNM, a: f64, b: f64, c: f64, d: f64) -> MatNM with Mut, Div, Panic {
    var m = m0
    m = matnm_set(m, 0, 0, a); m = matnm_set(m, 0, 1, b)
    m = matnm_set(m, 1, 0, c); m = matnm_set(m, 1, 1, d)
    return m
}

fn main() -> i32 with IO, Mut, Div, Panic {
    // Case 0: 2x2 A=[[4,3],[6,3]], B=[[1,2],[3,4]]
    var a = set4(matnm_zeros(2,2), 4.0, 3.0, 6.0, 3.0)
    var b = set4(matnm_zeros(2,2), 1.0, 2.0, 3.0, 4.0)
    emit_mat("det", 0, "A", a);       emit_scalar("det", 0, matnm_det(a))
    emit_mat("trace", 0, "A", a);     emit_scalar("trace", 0, matnm_trace(a))
    emit_mat("norm_fro", 0, "A", a);  emit_scalar("norm_fro", 0, matnm_norm_fro(a))
    emit_mat("transpose", 0, "A", a); emit_mat("transpose", 0, "R", matnm_transpose(a))
    emit_mat("mul", 0, "A", a); emit_mat("mul", 0, "B", b); emit_mat("mul", 0, "R", matnm_mul(a, b))
    emit_mat("inv", 0, "A", a); emit_mat("inv", 0, "R", matnm_inv(a))
    // solve: A x = bvec (2x1)
    var bv = matnm_zeros(2,1); bv = matnm_set(bv,0,0,5.0); bv = matnm_set(bv,1,0,6.0)
    emit_mat("solve", 0, "A", a); emit_mat("solve", 0, "B", bv); emit_mat("solve", 0, "R", matnm_solve(a, bv))
    // ... add Case 1 (3x3 general), Case 2 (3x3 SPD) similarly (use a 3x3 setter or inline matnm_set calls)
    return 0
}
```
Add a 3×3 setter helper and ≥2 more cases covering all 7 ops. Names/signatures per `git grep -nE '^\s*(pub )?fn matnm_' stdlib/linalg/matnm.sio`.

- [ ] **Step 2 — gate `scripts/linalg_parity_gate.sh`** — clone the stats gate, changing: `REF=scripts/parity/linalg_parity_ref.py`, source map `tests/parity/linalg_parity_<fam>.sio`, default families `matnm eigen`, sentinel `LINALG_PARITY_OK`/`LINALG_PARITY_GATE_OK`, keep `SOUNIO_SOUC_ENGINE=lean_single`, `--require-all` unless `REQUIRE_ALL=0`, mpmath SKIP guard. `chmod +x`.

- [ ] **Step 3 — run** `PARITY_FAMILIES="matnm" REQUIRE_ALL=0 bash scripts/linalg_parity_gate.sh`. Paste the table.

- [ ] **Step 4 — FIX-AS-FOUND** (inv/solve/det are the most error-prone). Any op > 1e-2 → root-cause in `stdlib/linalg/matnm.sio`; fix ≈1-line bugs (re-verify + no regression in `tests/stdlib/linalg/**`); else flag with the number. Never loosen a threshold to hide a bug. Separate `fix(linalg): …` commit.

- [ ] **Step 5 — commit** `feat(parity): linalg matnm scalar+element-wise ops + gate`.

---

## Task 3: matnm decompositions (LU, QR) — reconstruction invariants

**Files:** modify `scripts/parity/linalg_parity_ref.py` (+ LU/QR checks), extend `tests/parity/linalg_parity_matnm.sio`.

- [ ] **Step 1 — add LU/QR checks to the comparator:**
```python
def apply_piv(A, pivrole):
    # Phase-0 CONFIRMED: piv is a full PERMUTATION VECTOR (not swap partners):
    # P[k, piv[k]] = 1  ⟹  (P·A) row k = A row piv[k], and L·U == P·A.
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
    n = Q.cols; QtQ = Q.T * Q; I = mp.eye(n)
    orth = matmax_rel(QtQ, I)
    return max(rec, orth)
```
Register `"lu":chk_lu, "qr":chk_qr` in `CHECKS`, `THR`. **Match `apply_piv` to the piv convention recorded in Phase-0 Step 2** (adjust if it's a full permutation vector instead of swap partners).

- [ ] **Step 2 — extend the matnm emitter** with lu/qr per case:
```
    let lu = matnm_lu(a)
    emit_mat("lu", 0, "A", a)
    emit_mat("lu", 0, "L", lu.l)
    emit_mat("lu", 0, "U", lu.u)
    // piv: emit as integers, role P, one per row (j=0)
    var pi = 0
    while pi < a.rows {
        print("lu 0 P "); print_int(pi); print(" 0 "); print_int(lu.piv[pi]); print("\n")
        pi = pi + 1
    }
    let qr = matnm_qr(a)
    emit_mat("qr", 0, "A", a); emit_mat("qr", 0, "Q", qr.q); emit_mat("qr", 0, "RR", qr.r)
```
(Confirm the LUResult/QRResult field access `lu.l`/`lu.u`/`lu.piv`/`qr.q`/`qr.r` compiles; adapt names if different.)

- [ ] **Step 3 — run** `PARITY_FAMILIES="matnm" REQUIRE_ALL=0 …`; paste table (now incl. lu/qr). Fix-as-found (a decomposition failing reconstruction is a real bug). Commit `feat(parity): linalg LU/QR reconstruction invariants` (+ any `fix(linalg):`).

---

## Task 4: eigen_symmetric — eigenvalue set + eigenvector residual

**Files:** Create `tests/parity/linalg_parity_eigen.sio`; add `chk_eig` to the comparator.

- [ ] **Step 1 — eigen emitter** (flat-array interface, module-level `&!` out-params, per `test_eigen_e2e.sio`). Build symmetric A in a flat `[f64;65536]` (`A[i*n+j]`), call `eigen_symmetric(&!WORK, n, &!EVALS, &!EVECS)`, emit `A` (role A, from the flat array), `EVAL` (role EVAL, i-th value at (i,0)), `EVEC` (role EVEC, per the layout from Phase-0 Step 3 — emit so that Python reads eigenvector k as COLUMN k of the emitted EVEC matrix; transpose in the emitter if the native layout is row-major). Use ≥2 symmetric cases (a 2×2 with exact eigenpairs, a 3×3 SPD).

- [ ] **Step 2 — add `chk_eig`:**
```python
def chk_eig(d):
    A = as_mat(d["A"]); n = A.rows
    got = sorted(mp.mpf(d["EVAL"][(i,0)]) for i in range(n))
    E, Q = mp.eigsy(A)   # symmetric: real eigenvalues (ascending) + orthonormal Q
    ref = sorted(mp.mpf(E[i]) for i in range(n))
    ev_err = max(float(relerr(g, r)) for g, r in zip(got, ref))
    # eigenvector residual: EVEC column k pairs with the emitted EVAL[k]
    V = as_mat(d["EVEC"]); res = 0.0
    for k in range(n):
        vk = mp.matrix([V[i, k] for i in range(n)])
        lam = mp.mpf(d["EVAL"][(k,0)])
        nrm = mp.norm(vk) or mp.mpf(1)
        res = max(res, float(mp.norm(A*vk - lam*vk) / nrm))
    return max(ev_err, res)
```
Register `"eig":chk_eig`. (Eigenvalue matching by sorted order assumes no gross degeneracy in the test cases — pick well-separated spectra.)

- [ ] **Step 3 — run** the FULL gate `bash scripts/linalg_parity_gate.sh` (matnm + eigen, require-all). Paste table. Fix-as-found. Commit `feat(parity): linalg symmetric eigen (values-set + eigenvector residual)`.

---

## Task 5: report + final verification

**Files:** Create `docs/research/2026-07-19-linalg-parity.md`.

- [ ] **Step 1 — full gate green + coverage** (`--require-all`, all ops present).
- [ ] **Step 2 — determinism** (two runs identical).
- [ ] **Step 3 — anchors** tight: `trace` exact, `transpose` exact, `mul` by identity exact (rel < 1e-12 where the op is exact-arithmetic).
- [ ] **Step 4 — calibrate** thresholds for any genuine conditioning-limited op (tie to the case's condition number; never hide a bug).
- [ ] **Step 5 — write the report** (mirror special/stats): method (matrix bit-exact bridge, reconstruction invariants), the per-op accuracy map, the **convention findings** (row-major, piv encoding, eigenvector layout), defects found+fixed, reproduce command. Register governance (`node scripts/docs/sync_governance_metadata.mjs`).
- [ ] **Step 6 — commit** report + governance + calibrations.

---

## Notes for the implementer

- `SOUNIO_SOUC_ENGINE=lean_single` for every compile. No `f64 as string`.
- Bit f64 on LOCALS (f64-param-cast-bug guard). Passing `MatNM` by value to a helper may be a Madaros risk — if it fails, inline the emit loop.
- Decomposition checks reconstruct in PYTHON from emitted factors — never trust Sounio matmul for the invariant.
- Never loosen a threshold to hide a bug. A convention detail (piv encoding, eigenvector layout) is a COMPARATOR fix, not a stdlib fix — confirmed in Phase-0.
- Fix-as-found: root-caused ≈1-line stdlib defects → separate `fix(linalg):` commit, re-verified, no regression.
