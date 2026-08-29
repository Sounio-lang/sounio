<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-19-special-scipy-parity
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-19-special-scipy-parity
-->

# SciPy↔Sounio Special-Function Parity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an honest, reproducible per-function accuracy map of `stdlib/special` against an arbitrary-precision mpmath reference, behind a gate.

**Architecture:** A Sounio emitter compiles the special functions, evaluates each at a set of "nice" arguments, and prints one **self-describing** integer line per evaluation. A Python comparator mirrors the arguments from that output, computes mpmath ground truth at `dps=30`, and reports `max_rel_err` per function. A bash gate orchestrates and emits the verdict. Nothing ships into the prebuilt compiler.

**Tech Stack:** Sounio (`stdlib/special`, `bin/souc`), Python 3 + mpmath 1.3.0, bash.

**Spec:** `docs/superpowers/specs/2026-07-19-special-scipy-parity-design.md`

---

## Phase-0 outcome (Task 0, commit `ffa93c5f1`) — DECISIONS THAT BIND THE REST

- **`print_int(n: i64)` is a compiler builtin** — bare call, NO `use`.
- **Bridge = BIT-EXACT** via the `f64_to_bits(x: f64) -> i64` builtin (validated:
  `f64_to_bits(1.0)` = `4607182418800017408`). Lossless across all magnitudes; no
  scaling, no "nice value" constraint on arguments. Fixed-point is only a
  documented fallback and is NOT used.
- **Engine = `lean_single` for EVERYTHING.** Glob-import into non-`pub`
  `special::*` families fails on default Madaros (`E175 private`); `f64_to_bits`
  segfaults on default. Every compile+run in this vertical MUST set
  `SOUNIO_SOUC_ENGINE=lean_single`.
- **f64-cast-bug guard:** keep every `f64_to_bits`/cast in the SAME function scope
  as where the f64 is produced — never pass an f64 across a call boundary and
  bitcast it after (per memory `madaros-f64-param-cast-bug`). The emit helpers
  below therefore take the already-computed f64 and bit it immediately.

---

## The wire format (bit-exact, single source of truth, no drift)

Each emitter line:

```
<fn> <nargs> <arg1_bits> [<arg2_bits> ...] <val_bits>
```

- `fn` — unique function name (`erf`, `bessel_j0`, `lgamma`, …).
- `nargs` — count of argument columns that follow.
- `argK_bits` / `val_bits` — the **signed i64 IEEE-754 bit pattern** of the f64,
  from `f64_to_bits(x)`. Any f64 round-trips exactly (no precision loss, no range
  limit, no shared scale config).

Python reconstructs each f64 exactly via
`struct.unpack('<d', struct.pack('<q', bits))[0]`, computes `ref = mpmath_fn(*args)`,
and records `rel_err = |value - ref| / max(|ref|, 1e-300)`. Because args are also
bit-exact, mpmath evaluates at *exactly* the double Sounio used.

---

## File structure

- Create `tests/parity/probe_fixedpoint.sio` — Phase-0 P0.1 probe.
- Create `tests/parity/probe_globimport.sio` — Phase-0 P0.2 probe.
- Create `tests/parity/probe_bitcast.sio` — Phase-0 P0.3 stretch probe.
- Create `stdlib/parity/emit.sio` — shared Sounio emit helper (`emit1`, `emit2`, `emit3`).
- Create `tests/parity/special_parity_erf.sio` … one emitter per family.
- Create `scripts/parity/special_parity_ref.py` — mpmath reference + comparator.
- Create `scripts/special_scipy_parity_gate.sh` — orchestrator.
- Create `docs/research/2026-07-19-special-scipy-parity.md` — the report.

Per-family emitters (not one giant file) keep each `use special::<mod>::*` import
set isolated and let a family compile/run under its own engine (P0.4).

---

## Task 0: Phase-0 capability probes

**Files:**
- Create: `tests/parity/probe_fixedpoint.sio`
- Create: `tests/parity/probe_globimport.sio`
- Create: `tests/parity/probe_bitcast.sio`

- [ ] **Step 1: Confirm the integer-print primitive name**

Run: `git grep -nE 'pub fn (print_int|print_i64|puti|print_i)\b' stdlib/`
Expected: find the canonical i64 printer (likely `print_int` in `stdlib/io` or
core). Record its exact `use` path. If none prints a bare i64 with newline, note
the closest (`print` + manual int→string). Call the chosen printer `PUTI` below;
substitute the real name in every emitter.

- [ ] **Step 2: Write the fixed-point round-trip probe**

`tests/parity/probe_fixedpoint.sio`:

```
//@ run-pass
//@ expect-stdout: FP_PROBE 842700792949714
use io::*   // adjust to the real print_int path from Step 1

fn main() -> i32 with IO, Mut, Div, Panic {
    let x = 0.842700792949714       // ~erf(1) truncated to 15 digits
    let scaled = x * 1000000000000000.0   // 1e15
    let r = if scaled >= 0.0 { scaled + 0.5 } else { scaled - 0.5 }
    let i = r as i64
    print("FP_PROBE ")
    print_int(i)     // -> PUTI
    print("\n")
    return 0
}
```

- [ ] **Step 3: Run the probe on default Madaros, then lean_single**

Run:
```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
./bin/souc compile tests/parity/probe_fixedpoint.sio -o /tmp/fp.elf && /tmp/fp.elf
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile tests/parity/probe_fixedpoint.sio -o /tmp/fp2.elf && /tmp/fp2.elf
```
Expected: both print `FP_PROBE 842700792949714` (±1 in the last digit is OK).
Record which engine(s) work. If default fails, the whole vertical uses
`lean_single`. **If neither prints the integer, STOP** — the fixed-point bridge
is unavailable; report and reconsider (do not proceed to emitters).

- [ ] **Step 4: Write the glob-import reach probe**

`tests/parity/probe_globimport.sio`:

```
//@ run-pass
//@ expect-stdout: GLOB_OK
use special::bessel::*

fn main() -> i32 with IO, Mut, Div, Panic {
    let j = bessel_j0(0.0)        // non-pub fn reached via glob import
    if j > 0.999 { print("GLOB_OK\n") } else { print("GLOB_FAIL\n") }
    return 0
}
```

- [ ] **Step 5: Run the glob-import probe (both engines)**

Run: same compile+run as Step 3 for `probe_globimport.sio`.
Expected: `GLOB_OK`. Record the engine. If it fails on both, non-pub families
(bessel/airy/zeta/elliptic/hypergeometric/orthopoly) need their functions made
`pub` first — note this as a scoped prerequisite for those families only; the
pub families (erf/gamma/beta/igamma) proceed regardless.

- [ ] **Step 6: Write the bit-exact reinterpret stretch probe**

`tests/parity/probe_bitcast.sio`:

```
//@ run-pass
use io::*

fn main() -> i32 with IO, Mut, Div, Panic {
    // 1.0 has IEEE-754 bits 0x3FF0000000000000 = 4607182418800017408
    let x = 1.0
    let bits = f64_to_bits(x)   // use the stdlib reinterpret if one exists; else a syscall-free transmute helper
    print("BITS ")
    print_int(bits)
    print("\n")
    return 0
}
```

- [ ] **Step 7: Resolve the bit-exact primitive, run, and DECIDE the bridge**

Run: `git grep -nE 'f64_to_bits|to_bits|f64_bits|transmute|reinterpret' stdlib/`.
If a correct reinterpret exists and the probe prints `BITS 4607182418800017408`,
set **BRIDGE = bit-exact** (emit raw bits; Python uses `struct.unpack('<d', struct.pack('<q', bits))`).
Otherwise set **BRIDGE = fixed-point** (the default; the rest of this plan
assumes fixed-point). Record the decision at the top of the report doc.

- [ ] **Step 8: Commit the probe results**

```bash
git add tests/parity/probe_*.sio
git commit -m "test(parity): Phase-0 capability probes (fixed-point/glob/bitcast + engine map)"
```

---

## Task 1: Python mpmath reference + comparator core

**Files:**
- Create: `scripts/parity/special_parity_ref.py`
- Test: inline `--selftest` mode (no pytest in this repo lane)

- [ ] **Step 1: Write the comparator with a self-test**

`scripts/parity/special_parity_ref.py`:

```python
#!/usr/bin/env python3
"""Compare Sounio special-function emitter output against mpmath (dps=30).
Bit-exact wire on stdin: `<fn> <nargs> <arg_bits...> <val_bits>`, each field a
signed-i64 IEEE-754 bit pattern (Sounio f64_to_bits). No scipy dependency."""
import sys, struct, mpmath as mp
mp.mp.dps = 30

def bits_to_f64(field):
    return struct.unpack('<d', struct.pack('<q', int(field)))[0]

# fn -> (callable(*args)->mpf, gross_threshold). Thresholds are calibrated per
# family in later tasks; default gross bar is 1e-2 (fail loudly only on that).
REF = {
    "erf":            (lambda x: mp.erf(x), 1e-2),
    "erfc":           (lambda x: mp.erfc(x), 1e-2),
    "erfinv":         (lambda x: mp.erfinv(x), 1e-2),
    "normal_cdf":     (lambda x: mp.ncdf(x), 1e-2),
    "normal_quantile":(lambda p: mp.sqrt(2)*mp.erfinv(2*p-1), 1e-2),
}

def main():
    rows = {}   # fn -> list[(args, value, ref, rel)]
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        if not parts or parts[0] not in REF: continue
        fn = parts[0]; nargs = int(parts[1])
        args  = [bits_to_f64(parts[2+i]) for i in range(nargs)]
        value = bits_to_f64(parts[2+nargs])
        ref = float(REF[fn][0](*[mp.mpf(a) for a in args]))
        rel = abs(value - ref) / max(abs(ref), 1e-300)
        rows.setdefault(fn, []).append((args, value, ref, rel))
    fail = 0
    print(f"{'function':<18}{'points':>7}{'max_rel_err':>16}  verdict")
    for fn in REF:
        pts = rows.get(fn, [])
        if not pts:
            print(f"{fn:<18}{0:>7}{'NO DATA':>16}  FAIL"); fail = 1; continue
        worst = max(pts, key=lambda r: r[3]); mre = worst[3]; thr = REF[fn][1]
        ok = mre <= thr
        print(f"{fn:<18}{len(pts):>7}{mre:>16.3e}  {'PASS' if ok else 'FAIL(thr=%.0e)'%thr}")
        if not ok: fail = 1
    print("SPECIAL_SCIPY_PARITY_OK" if not fail else "SPECIAL_SCIPY_PARITY_FAIL")
    return fail

def _bits(x):  # float64 -> signed i64 bit pattern (mirror of Sounio f64_to_bits)
    return struct.unpack('<q', struct.pack('<d', x))[0]

def selftest():
    # Synthetic exact line for erf(0.5): args + value emitted as bit patterns.
    val = float(mp.erf(mp.mpf('0.5')))
    line = f"erf 1 {_bits(0.5)} {_bits(val)}\n"
    import io
    sys.stdin = io.StringIO(line)
    rc = main()
    assert rc == 0, "selftest: erf(0.5) should pass"
    print("REF_SELFTEST_OK")

if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: sys.exit(main())
```

- [ ] **Step 2: Run the self-test (verify it passes)**

Run: `python3 scripts/parity/special_parity_ref.py --selftest`
Expected: prints `REF_SELFTEST_OK` (exit 0).

- [ ] **Step 3: Negative self-test (verify it can fail)**

Run: `printf 'erf 1 500000 999999999999999 15\n' | python3 scripts/parity/special_parity_ref.py`
Expected: erf row shows a large `max_rel_err` and `SPECIAL_SCIPY_PARITY_FAIL`.

- [ ] **Step 4: Commit**

```bash
git add scripts/parity/special_parity_ref.py
git commit -m "feat(parity): mpmath reference + comparator core (erf family)"
```

---

## Task 2: erf family end-to-end (the reference slice)

**Files:**
- Create: `stdlib/parity/emit.sio`
- Create: `tests/parity/special_parity_erf.sio`
- Create: `scripts/special_scipy_parity_gate.sh`

- [ ] **Step 1: Write the shared emit helper**

`stdlib/parity/emit.sio` — bit-exact line printers. **Callers pass ALREADY-BITTED
i64** (`f64_to_bits` applied to LOCALS in the caller's scope), so no f64 crosses a
call boundary to be bitcast after (the f64-param-cast-bug guard). `print_int` and
`f64_to_bits` are compiler builtins — no `use`.

```
// Bit-exact parity emit. line: <fn> <nargs> <arg_bits...> <val_bits>
pub fn emit1(name: string, a_bits: i64, val_bits: i64) with IO, Mut, Panic {
    print(name); print(" 1 ")
    print_int(a_bits); print(" ")
    print_int(val_bits); print("\n")
}

pub fn emit2(name: string, a_bits: i64, b_bits: i64, val_bits: i64) with IO, Mut, Panic {
    print(name); print(" 2 ")
    print_int(a_bits); print(" ")
    print_int(b_bits); print(" ")
    print_int(val_bits); print("\n")
}

pub fn emit3(name: string, a_bits: i64, b_bits: i64, c_bits: i64, val_bits: i64) with IO, Mut, Panic {
    print(name); print(" 3 ")
    print_int(a_bits); print(" ")
    print_int(b_bits); print(" ")
    print_int(c_bits); print(" ")
    print_int(val_bits); print("\n")
}
```

- [ ] **Step 2: Write the erf emitter**

`tests/parity/special_parity_erf.sio` — evaluate each erf-family fn at test
points; **bit the arg and result as LOCALS in `main`** (f64-param-cast-bug guard),
pass only i64 to the helper:

```
//@ run-pass
use special::erf::*
use parity::emit::*

fn main() -> i32 with IO, Mut, Div, Panic {
    let xs = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0, -2.0]
    var i = 0
    while i < 10 {
        let x = xs[i]
        let e = erf(x)
        let ec = erfc(x)
        let nc = normal_cdf(x)
        emit1("erf", f64_to_bits(x), f64_to_bits(e))
        emit1("erfc", f64_to_bits(x), f64_to_bits(ec))
        emit1("normal_cdf", f64_to_bits(x), f64_to_bits(nc))
        i = i + 1
    }
    let ps = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    var k = 0
    while k < 7 {
        let p = ps[k]
        let ei = erfinv(p)
        let nq = normal_quantile(p)
        emit1("erfinv", f64_to_bits(p), f64_to_bits(ei))
        emit1("normal_quantile", f64_to_bits(p), f64_to_bits(nq))
        k = k + 1
    }
    return 0
}
```

- [ ] **Step 3: Write the gate orchestrator**

`scripts/special_scipy_parity_gate.sh`:

```bash
#!/usr/bin/env bash
# SciPy↔Sounio special-function parity gate. Reference = mpmath (dps=30).
# Requires mpmath (python3 -c 'import mpmath'). Dev-tier; not wired into ci.yml.
# The vertical is lean_single-LOCKED (Phase-0): glob-import into non-pub
# special::* families and f64_to_bits both require lean_single. Every emitter is
# compiled with SOUNIO_SOUC_ENGINE=lean_single.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single   # mandatory for this vertical
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
REF="scripts/parity/special_parity_ref.py"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
python3 -c 'import mpmath' 2>/dev/null || { echo "SKIP: mpmath not installed"; exit 0; }

emit_all() { : > "$OUT/emit.txt"
  for fam in "$@"; do
    src="tests/parity/special_parity_${fam}.sio"
    if ! "$SOUC" compile "$src" -o "$OUT/$fam.elf" >/dev/null 2>&1; then
      echo "FAIL compile $src"; return 1; fi
    chmod +x "$OUT/$fam.elf"; "$OUT/$fam.elf" >> "$OUT/emit.txt" 2>/dev/null || { echo "FAIL run $src"; return 1; }
  done
}

FAMILIES="${PARITY_FAMILIES:-erf}"
emit_all $FAMILIES || exit 1
python3 "$REF" < "$OUT/emit.txt" | tee "$OUT/report.txt"
grep -q "SPECIAL_SCIPY_PARITY_OK" "$OUT/report.txt" \
  && echo "SPECIAL_SCIPY_PARITY_GATE_OK" || { echo "GATE FAILED"; exit 1; }
```

- [ ] **Step 4: Run the erf gate**

Run: `chmod +x scripts/special_scipy_parity_gate.sh && bash scripts/special_scipy_parity_gate.sh`
Expected: a per-function table for erf/erfc/erfinv/normal_cdf/normal_quantile
with small `max_rel_err`, then `SPECIAL_SCIPY_PARITY_GATE_OK`. If a function's
`max_rel_err` exceeds the 1e-2 gross bar, that is a real finding — record it in
the report (do not silently loosen). Calibrate that fn's threshold in `REF` to
its genuine accuracy only after confirming it's an approximation, not a bug.

- [ ] **Step 5: Commit**

```bash
git add stdlib/parity/emit.sio tests/parity/special_parity_erf.sio scripts/special_scipy_parity_gate.sh
git commit -m "feat(parity): erf family end-to-end (emit helper + gate + mpmath compare)"
```

---

## Tasks 3–6: remaining pub families (names known)

> **Phase-0 supersede:** the bridge is **bit-exact**, so IGNORE every `exp=…` /
> `SCALE` mention in the per-family bullets below — there is no scale/exp. Emit
> each arg and result via `f64_to_bits` of a LOCAL (exactly the Task 2 erf
> emitter pattern) and pass i64 to `emit1/emit2/emit3`. Points may be any f64
> (bit-exact round-trip) — the "nice value" constraint no longer applies. All
> emitters compile under `SOUNIO_SOUC_ENGINE=lean_single` (the gate sets it).
> The per-family bullets' real content is the **point selection + mpmath ref
> mapping + convention checks**; keep those.

Each task follows Task 2's pattern: add the fn to `REF` in
`special_parity_ref.py`, write `tests/parity/special_parity_<fam>.sio`, extend
`PARITY_FAMILIES`, run the gate, commit. Concrete details per family:

### Task 3: gamma (`gamma`, `lgamma`, `digamma`)

- [ ] **Add to `REF`:** `"gamma":(lambda x: mp.gamma(x),1e-2)`, `"lgamma":(lambda x: mp.loggamma(x),1e-2)`, `"digamma":(lambda x: mp.digamma(x),1e-2)`.
- [ ] **Emitter** `special_parity_gamma.sio` (`use special::gamma::*`):
  - `gamma` at small positive args `[0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0]`, `exp=9` (Γ(6)=120).
  - `lgamma` at large args `[10.0,20.0,50.0,100.0,170.0]`, `exp=9` (log Γ is O(hundreds)).
  - `digamma` at `[0.5,1.0,1.5,2.0,3.0,5.0,10.0]`, `exp=12`.
- [ ] Run gate with `PARITY_FAMILIES="erf gamma"`; expect PASS. Commit.

### Task 4: beta (`beta`, `lbeta`, `ibeta`, `ibeta_inv`)

- [ ] **Add to `REF`:** `"beta":(lambda a,b: mp.beta(a,b),1e-2)`, `"lbeta":(lambda a,b: mp.log(mp.beta(a,b)),1e-2)`, `"ibeta":(lambda a,b,x: mp.betainc(a,b,0,x,regularized=True),1e-2)`, `"ibeta_inv":(lambda a,b,p: _betaincinv(a,b,p),1e-2)` where `_betaincinv` is a bisection on `mp.betainc(...,regularized=True)` (add the helper).
- [ ] **Confirm `ibeta` normalization:** check whether Sounio `ibeta` is the *regularized* incomplete beta (I_x) or the raw one; set `regularized` in the ref to match. Add a one-line note in the report.
- [ ] **Emitter** `special_parity_beta.sio` (`use special::beta::*`):
  - `beta`/`lbeta` at `(a,b)` pairs `[(0.5,0.5),(1,1),(2,3),(2.5,1.5),(5,2)]`, `exp=9`.
  - `ibeta` at `(2,3,x)` for `x in [0.1,0.25,0.5,0.75,0.9]`, `exp=15` (∈[0,1]).
  - `ibeta_inv` at `(2,3,p)` for `p in [0.1,0.5,0.9]`, `exp=15`.
- [ ] Run gate `PARITY_FAMILIES="erf gamma beta"`; expect PASS. Commit.

### Task 5: igamma (`igamma_lower`, `igamma_upper`, `chi2_cdf`)

- [ ] **Confirm normalization** (regularized P/Q vs raw γ/Γ) by reading `igamma.sio`; set the mpmath ref to match: regularized `mp.gammainc(a,0,x,regularized=True)` (lower P) / `mp.gammainc(a,x,inf,regularized=True)` (upper Q).
- [ ] **Add to `REF`** accordingly, `1e-2`; `"chi2_cdf":(lambda x,k: mp.gammainc(k/2,0,x/2,regularized=True),1e-2)`.
- [ ] **Emitter** `special_parity_igamma.sio`: `igamma_lower/upper` at `(a,x)` for `a in [0.5,1,2,5]`, `x in [0.5,1,2,5,10]`; `chi2_cdf` at `(x,k)` for `k in [1,2,5]`, `x in [0.5,2,5,10]`; all `exp=15` (∈[0,1]). Commit.

### Task 6: bessel (`bessel_j0/j1/jn`, `bessel_y0/y1/yn`, `bessel_i0/i1/in`, `bessel_k0/k1/kn`)

- [ ] **Verify exact names** with `git grep -nE '^\s*(pub )?fn bessel_' stdlib/special/bessel.sio` (jn/yn/in/kn take `(n:i32, x:f64)`).
- [ ] **Add to `REF`:** `mp.besselj(0,x)`, `mp.besselj(1,x)`, `mp.besselj(n,x)`, `mp.bessely(...)`, `mp.besseli(...)`, `mp.besselk(...)`; `1e-2`. For the `*n` variants Python reads `n` from the first arg column (emitted as `n*1e6` → divide → round).
- [ ] **Emitter** `special_parity_bessel.sio` (`use special::bessel::*`): j0/j1/y0/y1/i0/i1/k0/k1 at `x in [0.5,1,2,5,10]` (oscillatory + growth), `exp=12`; jn/yn/in/kn at `n in [2,3]`, same `x`. Note: `Y`/`K` diverge at 0 — start `x` at 0.5. For `i0/i1` at large x use `exp=6` (grows) or restrict to `x<=5`. Commit.

---

## Task 7: non-pub families (airy, zeta, elliptic, hypergeometric, orthopoly)

These need P0.2 green (glob import reaches non-pub fns). If P0.2 failed, first
make the target functions `pub` in each file (a mechanical, isolated change) and
re-run P0.2.

For **each** of `airy`, `zeta`, `elliptic`, `hypergeometric`, `orthopoly`:

- [ ] **Enumerate the real API:** `git grep -nE '^\s*(pub )?fn ' stdlib/special/<file>.sio`. Record exact names + signatures (there is no way around reading the file; the names are not guessable).
- [ ] **Map each to mpmath:** airy→`mp.airyai/airybi`; zeta→`mp.zeta`; elliptic→`mp.ellipk/ellipe` (watch the m vs k convention — SciPy uses parameter m=k²; match whichever the Sounio fn takes, and note it); hypergeometric→`mp.hyp2f1/hyp1f1`; orthopoly→`mp.legendre/chebyt/hermite/laguerre` (these take `(n, x)`).
- [ ] **Add to `REF`** with `1e-2` gross bar.
- [ ] **Write `special_parity_<file>.sio`** following Task 2: pick ~10–20 nice in-domain points (avoid poles/branch cuts; document any excluded point), choose `exp` so values are representable, emit.
- [ ] **Run gate** adding the family to `PARITY_FAMILIES`; if a family needs `lean_single`, set its engine in the gate's per-family map (P0.4). Record achieved `max_rel_err`. Commit per family.

**Convention pitfalls to check and note in the report:** elliptic m-vs-k;
regularized-vs-raw for any incomplete function; orthopoly normalization
(physicists' vs probabilists' Hermite — `mp.hermite` is physicists' H_n).

---

## Task 8: report doc, threshold calibration, final gate

**Files:**
- Create: `docs/research/2026-07-19-special-scipy-parity.md`
- Modify: `scripts/parity/special_parity_ref.py` (calibrated thresholds)

- [ ] **Step 1: Calibrate per-function thresholds**

For each function whose `max_rel_err` exceeds a tight bound but is a genuine
approximation (not a bug), set its `REF` threshold to just above its achieved
error (e.g. a fn at 3e-6 gets threshold `1e-5`). Anchors (erf(0), Γ(5)=24,
J0(0)=1) keep an implicit tight check via their point being near-exact. Any fn
above the **1e-2 gross bar** is flagged as a likely bug, not calibrated away —
list it in the report's "weak/suspect" section.

> **Coverage check (Task 1 left NO-DATA as SKIP, not FAIL).** The comparator
> skips functions with no emitted data so partial/self-test runs work. The FULL
> gate must therefore separately assert that EVERY function in `REF` appeared in
> the emitter output — else a missing/broken emitter silently SKIPs and hides a
> gap. Add to the gate (or the comparator via a `--require-all` flag): after
> scoring, fail if any `REF` key has 0 data rows in a full run.

- [ ] **Step 2: Run the full gate over every family**

Run: `PARITY_FAMILIES="erf gamma beta igamma bessel airy zeta elliptic hypergeometric orthopoly" bash scripts/special_scipy_parity_gate.sh | tee /tmp/parity_report.txt`
Expected: `SPECIAL_SCIPY_PARITY_GATE_OK` with the full per-function table.

- [ ] **Step 3: Write the report doc**

`docs/research/2026-07-19-special-scipy-parity.md`: the BRIDGE decision (Task 0),
the engine map, the full per-function table (function, #points, achieved
`max_rel_err`, threshold, engine), the convention notes, and a "weak/suspect
functions" section listing anything above 1e-6 with a one-line hypothesis. Frame
the reference as mpmath (dps=30), the ground truth scipy.special is validated
against. Register governance: `node scripts/docs/sync_governance_metadata.mjs`.

- [ ] **Step 4: Commit**

```bash
git add scripts/parity/special_parity_ref.py docs/research/2026-07-19-special-scipy-parity.md docs/governance/
git commit -m "docs(parity): special-function parity report + calibrated thresholds"
```

- [ ] **Step 5: Final verification (spec §7)**

- Gate emits `SPECIAL_SCIPY_PARITY_GATE_OK`.
- Anchors pass tight: erf(0)=0, Γ(5)=24, J0(0)=1 rows show `rel_err < 1e-12`.
- Re-run the gate twice; per-function `max_rel_err` is identical (deterministic).

---

## Notes for the implementer

- **Madaros f64:** if a family segfaults/miscompiles on default `bin/souc`, retry
  with `SOUNIO_SOUC_ENGINE=lean_single` and record it (P0.4). A red default
  engine is usually a compiler artifact, not a math failure.
- **No `f64 as string`, ever.** All numbers cross the boundary as integers.
- **Never loosen a threshold to hide a bug.** The vertical's value is the honest
  map; a wrong function must show up as FAIL, not as a widened tolerance.
- **`PUTI`** is a placeholder for the real i64 printer resolved in Task 0 Step 1;
  substitute it everywhere before running.
