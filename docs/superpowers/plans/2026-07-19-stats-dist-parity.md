<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-19-stats-dist-parity
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-19-stats-dist-parity
-->

# stats↔mpmath Distribution Parity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-function accuracy map of `stdlib/stats` distribution functions vs mpmath (dps=30), behind a gate, fixing any root-caused defects the map surfaces.

**Architecture:** Reuses the special-function parity harness merged in #1210 — same bit-exact `f64_to_bits` bridge, same `stdlib/parity/emit.sio` (emit1..emit4), same lean_single-locked gate/comparator pattern. A parallel comparator (`stats_parity_ref.py`) holds the distribution reference formulas; a parallel gate (`stats_dist_parity_gate.sh`) drives the stats emitters.

**Tech Stack:** Sounio (`stdlib/stats`, `bin/souc` under `SOUNIO_SOUC_ENGINE=lean_single`), Python 3 + mpmath, bash.

**Spec:** `docs/superpowers/specs/2026-07-19-stats-dist-parity-design.md`

---

## Confirmed signatures & conventions (from `densities.sio`, read during planning)

All distribution fns are `pub` in `stdlib/stats/densities.sio`; discrete ones take
`k` (and `n`) as **i64**. Two conventions differ from the naïve guess — bake these
into the reference:

- **gamma is RATE-parameterized:** `gamma_pdf/cdf(x, shape, rate)` computes
  `shape·ln(rate) − rate·x` → cdf = `P(shape, rate·x)`, pdf = `rate^shape · x^{shape−1} · e^{−rate·x} / Γ(shape)`.
- **geometric is FROM-0:** `geometric_pmf(k, p) = (1−p)^k · p`, support `k = 0,1,2,…`.

Scalars used (avoid the struct-based `distributions.sio` variants):
`normal_pdf(x,mu,sigma)`, `normal_cdf_at(x,mu,sigma)`, `standard_normal_cdf(z)`,
`inverse_standard_normal_cdf(p)`, `exponential_pdf/cdf(x,lambda)`,
`gamma_pdf/cdf(x,shape,rate)`, `beta_pdf/cdf(x,a,b)`, `lognormal_pdf/cdf(x,mu,sigma)`,
`uniform_pdf/cdf(x,a,b)`, `poisson_pmf/cdf(k,lambda)`, `binomial_pmf/cdf(k,n,p)`,
`geometric_pmf(k,p)`.

`standard_normal_cdf` and `inverse_standard_normal_cdf` are in `distributions.sio`;
the rest are in `densities.sio`.

## Wire format & bridge (reused, unchanged from #1210)

`<fn> <nargs> <arg_bits...> <val_bits>` — each field the signed-i64 IEEE-754 bit
pattern of the f64 (`f64_to_bits`, a builtin). Discrete `k`/`n` are emitted as
f64 (`f64_to_bits(3.0)`); Python rounds them back to int for the formula. Bit f64
on LOCALS in the emitter's `main`; pass i64 to `emit1..emit4`. Everything runs
under `SOUNIO_SOUC_ENGINE=lean_single`. `print_int` appends a newline → the
comparator tokenizes the whole stream (whitespace-agnostic).

## File structure

- Create `scripts/parity/stats_parity_ref.py` — mpmath comparator (distribution REF).
- Create `scripts/stats_dist_parity_gate.sh` — orchestrator.
- Create `tests/parity/stats_parity_continuous1.sio` — normal/exponential/standard-normal/quantile.
- Create `tests/parity/stats_parity_continuous2.sio` — gamma/beta.
- Create `tests/parity/stats_parity_continuous3.sio` — lognormal/uniform.
- Create `tests/parity/stats_parity_discrete.sio` — poisson/binomial/geometric.
- Create `docs/research/2026-07-19-stats-dist-parity.md` — the report.
- Reuse (do NOT modify) `stdlib/parity/emit.sio`.

---

## Task 0: Phase-0 sanity + convention confirmation

**Files:** Create `tests/parity/stats_probe.sio`.

- [ ] **Step 1: Confirm the reused bridge still works under lean_single.**

`tests/parity/stats_probe.sio`:
```
//@ run-pass
use stats::densities::*
use parity::emit::*
fn main() -> i32 with IO, Mut, Div, Panic {
    let v = normal_cdf_at(1.0, 0.0, 1.0)   // ≈ 0.8413447
    emit3("normal_cdf_at", f64_to_bits(1.0), f64_to_bits(0.0), f64_to_bits(1.0), f64_to_bits(v))
    return 0
}
```
Run: `export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"; SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile tests/parity/stats_probe.sio -o /tmp/sp.elf && chmod +x /tmp/sp.elf && /tmp/sp.elf`
Expected: one line `normal_cdf_at 3 <bits> <bits> <bits> <bits>`. If it won't
compile/run under lean_single, STOP and report (the reused path is broken).

- [ ] **Step 2: Confirm the two tricky conventions empirically.**

Emit `gamma_cdf(2.0, 2.0, 1.0)` and `geometric_pmf(2, 0.3)` from a scratch program;
run; decode the bits in Python and compare:
- gamma_cdf(x=2, shape=2, rate=1): rate form → `mp.gammainc(2,0,1·2,regularized=True)` ≈ 0.5940; scale form would be `gammainc(2,0,2/1)` — same here, so also test `gamma_cdf(2.0, 2.0, 2.0)`: rate → `gammainc(2,0,4)` ≈ 0.9084 vs scale `gammainc(2,0,1)` ≈ 0.2642. Confirm Sounio matches the RATE value.
- geometric_pmf(k=2, p=0.3): from-0 → `0.7²·0.3 = 0.147`; from-1 → `0.7¹·0.3 = 0.21`. Confirm Sounio = 0.147.

Record both. (Planning already read the source: gamma=rate, geometric=from-0 — this step verifies at runtime.)

- [ ] **Step 3: Commit the probe.**
```bash
git add tests/parity/stats_probe.sio
git commit -m "test(parity): stats Phase-0 probe (bridge reuse + gamma-rate/geometric-from-0 conventions)"
```

---

## Task 1: mpmath distribution comparator

**Files:** Create `scripts/parity/stats_parity_ref.py`.

- [ ] **Step 1: Write the comparator** (mirrors `scripts/parity/special_parity_ref.py`: whole-stream tokenizer, `bits_to_f64`, `--require-all`, `--selftest`). The distribution `REF`:

```python
#!/usr/bin/env python3
"""stats distribution parity vs mpmath (dps=30). Bit-exact wire on stdin:
`<fn> <nargs> <arg_bits...> <val_bits>` (signed-i64 IEEE-754). No scipy."""
import sys, struct, math, mpmath as mp
mp.mp.dps = 30
def bits_to_f64(f): return struct.unpack('<d', struct.pack('<q', int(f)))[0]
def _bits(x): return struct.unpack('<q', struct.pack('<d', x))[0]
SQRT2PI = mp.sqrt(2*mp.pi)

def _norm_pdf(x, mu, s):  z=(x-mu)/s; return mp.e**(-z*z/2)/(s*SQRT2PI)
def _lognorm_pdf(x, mu, s):
    if x <= 0: return mp.mpf(0)
    z=(mp.log(x)-mu)/s; return mp.e**(-z*z/2)/(x*s*SQRT2PI)
def _gamma_pdf(x, sh, rate):
    if x <= 0: return mp.mpf(0)
    return rate**sh * x**(sh-1) * mp.e**(-rate*x) / mp.gamma(sh)
def _beta_pdf(x, a, b):
    if x <= 0 or x >= 1: return mp.mpf(0)
    return x**(a-1)*(1-x)**(b-1)/mp.beta(a,b)
def _uniform_pdf(x, a, b): return mp.mpf(1)/(b-a) if a <= x <= b else mp.mpf(0)
def _uniform_cdf(x, a, b): return mp.mpf(0) if x < a else (mp.mpf(1) if x > b else (x-a)/(b-a))
def _pois_pmf(k, lam): k=int(round(k)); return lam**k * mp.e**(-lam) / mp.factorial(k)
def _pois_cdf(k, lam): k=int(round(k)); return mp.gammainc(k+1, lam, mp.inf, regularized=True)
def _binom_pmf(k, n, p):
    k=int(round(k)); n=int(round(n)); return mp.binomial(n,k)*mp.mpf(p)**k*(1-mp.mpf(p))**(n-k)
def _binom_cdf(k, n, p):
    k=int(round(k)); n=int(round(n))
    if k >= n: return mp.mpf(1)
    return mp.betainc(n-k, k+1, 0, 1-mp.mpf(p), regularized=True)
def _geom_pmf(k, p): k=int(round(k)); return (1-mp.mpf(p))**k * p    # FROM-0

REF = {
    "normal_pdf":    (lambda x,mu,s: _norm_pdf(x,mu,s), 1e-2),
    "normal_cdf_at": (lambda x,mu,s: mp.ncdf((x-mu)/s), 1e-2),
    "standard_normal_cdf":         (lambda z: mp.ncdf(z), 1e-2),
    "inverse_standard_normal_cdf": (lambda p: mp.sqrt(2)*mp.erfinv(2*p-1), 1e-2),
    "exponential_pdf": (lambda x,l: l*mp.e**(-l*x) if x>=0 else mp.mpf(0), 1e-2),
    "exponential_cdf": (lambda x,l: 1-mp.e**(-l*x) if x>=0 else mp.mpf(0), 1e-2),
    "gamma_pdf": (lambda x,sh,r: _gamma_pdf(x,sh,r), 1e-2),
    "gamma_cdf": (lambda x,sh,r: mp.gammainc(sh,0,r*x,regularized=True) if x>0 else mp.mpf(0), 1e-2),
    "beta_pdf":  (lambda x,a,b: _beta_pdf(x,a,b), 1e-2),
    "beta_cdf":  (lambda x,a,b: mp.betainc(a,b,0,x,regularized=True), 1e-2),
    "lognormal_pdf": (lambda x,mu,s: _lognorm_pdf(x,mu,s), 1e-2),
    "lognormal_cdf": (lambda x,mu,s: mp.ncdf((mp.log(x)-mu)/s) if x>0 else mp.mpf(0), 1e-2),
    "uniform_pdf": (lambda x,a,b: _uniform_pdf(x,a,b), 1e-2),
    "uniform_cdf": (lambda x,a,b: _uniform_cdf(x,a,b), 1e-2),
    "poisson_pmf": (lambda k,l: _pois_pmf(k,l), 1e-2),
    "poisson_cdf": (lambda k,l: _pois_cdf(k,l), 1e-2),
    "binomial_pmf": (lambda k,n,p: _binom_pmf(k,n,p), 1e-2),
    "binomial_cdf": (lambda k,n,p: _binom_cdf(k,n,p), 1e-2),
    "geometric_pmf": (lambda k,p: _geom_pmf(k,p), 1e-2),
}

def main(require_all=False):
    rows = {}
    tokens = [t for t in sys.stdin.read().split() if not t.startswith("#")]
    i, n = 0, 0; N = len(tokens)
    while i < N:
        fn = tokens[i]
        if fn not in REF: i += 1; continue
        nargs = int(tokens[i+1])
        args = [bits_to_f64(tokens[i+2+j]) for j in range(nargs)]
        value = bits_to_f64(tokens[i+2+nargs]); i += 2+nargs+1
        ref = float(REF[fn][0](*[mp.mpf(a) for a in args]))
        rel = abs(value-ref)/max(abs(ref),1e-300)
        rows.setdefault(fn, []).append((args, value, ref, rel))
    fail = 0
    print(f"{'function':<26}{'points':>7}{'max_rel_err':>16}  verdict")
    for fn in REF:
        pts = rows.get(fn, [])
        if not pts:
            print(f"{fn:<26}{0:>7}{'NO DATA':>16}  {'FAIL(no-data)' if require_all else 'SKIP'}")
            if require_all: fail = 1
            continue
        worst = max(pts, key=lambda r: r[3]); mre = worst[3]; thr = REF[fn][1]
        ok = mre <= thr
        print(f"{fn:<26}{len(pts):>7}{mre:>16.3e}  {'PASS' if ok else 'FAIL(thr=%.0e)'%thr}")
        if not ok: fail = 1
    print("STATS_DIST_PARITY_OK" if not fail else "STATS_DIST_PARITY_FAIL")
    return fail

def selftest():
    v = float(mp.ncdf(1.0))
    line = f"standard_normal_cdf 1 {_bits(1.0)} {_bits(v)}\n"
    import io; sys.stdin = io.StringIO(line)
    assert main() == 0, "selftest: standard_normal_cdf(1) should pass"
    print("STATS_REF_SELFTEST_OK")

if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: sys.exit(main(require_all=("--require-all" in sys.argv)))
```

- [ ] **Step 2: Positive self-test.** `python3 scripts/parity/stats_parity_ref.py --selftest` → `STATS_REF_SELFTEST_OK`.
- [ ] **Step 3: Negative test.** `python3 -c "import struct;print('standard_normal_cdf 1', struct.unpack('<q',struct.pack('<d',1.0))[0], struct.unpack('<q',struct.pack('<d',0.5))[0])" | python3 scripts/parity/stats_parity_ref.py` → shows a large err and `STATS_DIST_PARITY_FAIL` (exit 1).
- [ ] **Step 4: Commit.**
```bash
git add scripts/parity/stats_parity_ref.py
git commit -m "feat(parity): mpmath distribution comparator (stats_parity_ref.py)"
```

---

## Task 2: continuous group 1 + gate (the proving slice)

**Files:** Create `tests/parity/stats_parity_continuous1.sio`, `scripts/stats_dist_parity_gate.sh`.

- [ ] **Step 1: Emitter** `tests/parity/stats_parity_continuous1.sio` — normal_pdf/normal_cdf_at (from densities), standard_normal_cdf/inverse_standard_normal_cdf (from distributions), exponential_pdf/cdf. Bit LOCALS in main:
```
//@ run-pass
use stats::densities::*
use stats::distributions::*
use parity::emit::*

fn main() -> i32 with IO, Mut, Div, Panic {
    // standard normal cdf + quantile (1-arg)
    let zs = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.96, 2.0]
    var i = 0
    while i < 8 {
        let z = zs[i]
        let c = standard_normal_cdf(z)
        emit1("standard_normal_cdf", f64_to_bits(z), f64_to_bits(c))
        i = i + 1
    }
    let ps = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    var k = 0
    while k < 7 {
        let p = ps[k]
        let q = inverse_standard_normal_cdf(p)
        emit1("inverse_standard_normal_cdf", f64_to_bits(p), f64_to_bits(q))
        k = k + 1
    }
    // normal pdf/cdf with (mu,sigma) and exponential with lambda
    let xs = [-1.0, 0.0, 0.5, 1.0, 2.0]
    var j = 0
    while j < 5 {
        let x = xs[j]
        let np = normal_pdf(x, 0.0, 1.0)
        let nc = normal_cdf_at(x, 0.0, 1.0)
        let np2 = normal_pdf(x, 1.0, 2.0)
        emit3("normal_pdf", f64_to_bits(x), f64_to_bits(0.0), f64_to_bits(1.0), f64_to_bits(np))
        emit3("normal_cdf_at", f64_to_bits(x), f64_to_bits(0.0), f64_to_bits(1.0), f64_to_bits(nc))
        emit3("normal_pdf", f64_to_bits(x), f64_to_bits(1.0), f64_to_bits(2.0), f64_to_bits(np2))
        j = j + 1
    }
    let exs = [0.0, 0.5, 1.0, 2.0, 5.0]
    var m = 0
    while m < 5 {
        let x = exs[m]
        let ep = exponential_pdf(x, 1.5)
        let ec = exponential_cdf(x, 1.5)
        emit2("exponential_pdf", f64_to_bits(x), f64_to_bits(1.5), f64_to_bits(ep))
        emit2("exponential_cdf", f64_to_bits(x), f64_to_bits(1.5), f64_to_bits(ec))
        m = m + 1
    }
    return 0
}
```

- [ ] **Step 2: Gate** `scripts/stats_dist_parity_gate.sh` — clone `scripts/special_scipy_parity_gate.sh`, changing: `REF=scripts/parity/stats_parity_ref.py`, family token → source file map `tests/parity/stats_parity_<fam>.sio`, default families `continuous1 continuous2 continuous3 discrete`, sentinel `STATS_DIST_PARITY_OK`/`STATS_DIST_PARITY_GATE_OK`, `--require-all` unless `REQUIRE_ALL=0`. Keep `export SOUNIO_SOUC_ENGINE=lean_single` and the mpmath SKIP guard. `chmod +x`.

- [ ] **Step 3: Run** `PARITY_FAMILIES="continuous1" REQUIRE_ALL=0 bash scripts/stats_dist_parity_gate.sh`. Paste the table. Expect the 6 functions PASS. **FIX-AS-FOUND:** any fn > 1e-2 → root-cause in the relevant `stdlib/stats/*.sio`; fix if a clear ≈1-line bug (re-verify + no regression in `tests/stdlib/stats/**`), else flag. Convention mismatch → fix the REF, not stdlib. Never loosen a threshold to hide a bug.

- [ ] **Step 4: Commit.**
```bash
git add tests/parity/stats_parity_continuous1.sio scripts/stats_dist_parity_gate.sh
git commit -m "feat(parity): stats continuous group 1 (normal/quantile/exponential) + gate"
```

---

## Task 3: continuous group 2 — gamma (RATE) + beta

**Files:** Create `tests/parity/stats_parity_continuous2.sio`.

- [ ] **Emitter:** `gamma_pdf/cdf(x, shape, rate)` at (shape,rate) ∈ {(2,1),(2,2),(0.5,1),(5,1)}, x ∈ [0.5,1,2,5] → emit3. `beta_pdf/cdf(x, a, b)` at (a,b) ∈ {(2,3),(0.5,0.5),(5,2)}, x ∈ [0.1,0.25,0.5,0.75,0.9] → emit3. (The REF already encodes gamma-as-rate.)
- [ ] Run `PARITY_FAMILIES="continuous1 continuous2" REQUIRE_ALL=0 bash scripts/stats_dist_parity_gate.sh`; paste table; fix-as-found; commit `feat(parity): stats gamma(rate)+beta`.

## Task 4: continuous group 3 — lognormal + uniform

**Files:** Create `tests/parity/stats_parity_continuous3.sio`.

- [ ] **Emitter:** `lognormal_pdf/cdf(x, mu, sigma)` at (mu,sigma) ∈ {(0,1),(0,0.5),(1,1)}, x ∈ [0.5,1,2,5] → emit3. `uniform_pdf/cdf(x, a, b)` at (a,b)=(0,1) and (2,5), x spanning below/inside/above support (e.g. [-1,0.5,1.5,3,6]) to exercise the 0/1 clamps → emit3.
- [ ] Run adding `continuous3`; paste table; fix-as-found; commit `feat(parity): stats lognormal+uniform`.

## Task 5: discrete — poisson + binomial + geometric

**Files:** Create `tests/parity/stats_parity_discrete.sio`.

- [ ] **Emitter (k/n are i64 — call with int literals, emit their f64 form):**
  - `poisson_pmf/cdf(k, lambda)` at lambda ∈ {1.0, 4.0}, k ∈ {0,1,2,4,8} → `emit2("poisson_pmf", f64_to_bits(0.0), f64_to_bits(lambda), f64_to_bits(poisson_pmf(0, lambda)))` (one call per literal k).
  - `binomial_pmf/cdf(k, n, p)` at (n,p) ∈ {(10,0.3),(20,0.5)}, k ∈ {0,3,5,10} → emit3 with `f64_to_bits(k_as_float)`, `f64_to_bits(n_as_float)`, `f64_to_bits(p)`.
  - `geometric_pmf(k, p)` at p ∈ {0.3, 0.5}, k ∈ {0,1,2,5} → emit2. (REF is FROM-0.)
  Anchors to include: `poisson_pmf(0,λ)=e^{−λ}`, `binomial_pmf(0,n,p)=(1−p)^n`, `geometric_pmf(0,p)=p`.
- [ ] Run the FULL default gate `bash scripts/stats_dist_parity_gate.sh` (require-all, all 4 groups); paste table; fix-as-found; commit `feat(parity): stats discrete (poisson/binomial/geometric)`.

---

## Task 6: report + final verification

**Files:** Create `docs/research/2026-07-19-stats-dist-parity.md`.

- [ ] **Step 1: Full gate green + coverage.** `bash scripts/stats_dist_parity_gate.sh` → `STATS_DIST_PARITY_GATE_OK`, all ~20 functions present (require-all).
- [ ] **Step 2: Calibrate thresholds** for any genuine-approximation function above a tight bound (never for a bug — those get fixed or flagged).
- [ ] **Step 3: Determinism** — two runs identical.
- [ ] **Step 4: Anchors** — normal_cdf(0)=0.5, geometric_pmf(0,p)=p, poisson_pmf(0,λ)=e^{−λ} at rel_err < 1e-12.
- [ ] **Step 5: Write the report** (mirror the special report): method (reused bit-exact bridge, mpmath formulas), the full per-function accuracy map, the **convention findings** (gamma=rate, geometric=from-0, and any others), defects found+fixed, reproduce command. Register governance: `node scripts/docs/sync_governance_metadata.mjs`.
- [ ] **Step 6: Commit** report + governance + any calibrated thresholds.

---

## Notes for the implementer

- `SOUNIO_SOUC_ENGINE=lean_single` for every compile (the gate sets it).
- No `f64 as string`. Discrete `k`/`n` cross the wire as f64 bits; Python rounds them.
- Never loosen a threshold to hide a bug. A convention mismatch is a REF fix, not a stdlib fix.
- Fix-as-found: root-caused ≈1-line stdlib defects fixed in separate `fix(stats):` commits, re-verified, no regression in existing `tests/stdlib/stats/**`.
