<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-19-special-scipy-parity-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-19-special-scipy-parity-design
-->

# SciPy↔Sounio parity vertical — `stdlib/special`

**Status:** design (approved 2026-07-19)
**Owner:** Data & Science verticals
**Depends on:** nothing new shipping-side; runs against a current-source or shipped Madaros.

## 1. Goal

Produce an honest, reproducible **parity map** of Sounio's special-function
library (`stdlib/special`) against an arbitrary-precision reference. For each
covered function we measure the *actual* achieved accuracy vs ground truth over a
representative point set, and gate on it. The deliverable is not "everything
matches SciPy to 1e-15" — it is a calibrated, per-function accuracy report that
tells the truth about where Sounio's series/approximations stand.

This mirrors the PBPK Sounio↔Node parity pattern (same inputs → both sides →
compare to tolerance → report) and the CSV-reader vertical (a real capability
surfaced behind a byte-exact gate).

## 2. Reference: mpmath, not scipy

`scipy` is **not installed** in the build/gate environment; `mpmath` 1.3.0 **is**.
mpmath is the arbitrary-precision reference that `scipy.special` is itself
validated against, so it is a *stronger* ground truth than scipy's own doubles.
The reference runs at `mp.dps = 30` (≈30 significant digits), far beyond f64, so
reference rounding never limits the measurement. The report is framed as
"Sounio vs the arbitrary-precision reference scipy.special is validated against."

If a scipy-doubles cross-check is ever wanted, it is an optional add-on and does
not block this vertical.

## 3. Scope — the catalog

All `stdlib/special` functions that have a clean mpmath reference:

| Family | File | Functions | mpmath reference |
|---|---|---|---|
| erf | `erf.sio` | `erf`, `erfc`, `erfinv`, `normal_cdf`, `normal_quantile` | `erf`, `erfc`, `erfinv`, `ncdf`, `inverse ncdf` |
| gamma | `gamma.sio` | `gamma`, `lgamma`, `digamma` | `gamma`, `loggamma`, `digamma` |
| beta | `beta.sio` | `beta`, `lbeta`, `ibeta`, `ibeta_inv` | `beta`, `log`, `betainc(regularized)`, inverse |
| igamma | `igamma.sio` | `igamma_lower`, `igamma_upper`, `chi2_cdf` | `gammainc` (regularized lower/upper), χ² cdf |
| bessel | `bessel.sio` | `bessel_j0/j1/jn`, `bessel_y0/y1/yn`, `bessel_i0/i1/in`, `bessel_k0/k1/kn` | `besselj`, `bessely`, `besseli`, `besselk` |
| airy | `airy.sio` | `airy_ai`, `airy_bi` (+ derivatives if present) | `airyai`, `airybi` |
| zeta | `zeta.sio` | Riemann ζ (+ Hurwitz if present) | `zeta` |
| elliptic | `elliptic.sio` | complete `K`, `E` (+ incomplete if present) | `ellipk`, `ellipe` |
| hypergeometric | `hypergeometric.sio` | `2F1` / `1F1` (whatever is implemented) | `hyp2f1`, `hyp1f1` |
| orthopoly | `orthopoly.sio` | Legendre / Chebyshev / Hermite / Laguerre `P_n(x)` | `legendre`, `chebyt`, `hermite`, `laguerre` |

Exact names are enumerated per file during implementation (some functions are
declared `fn` not `pub fn`, but the existing `tests/stdlib/special/*_e2e.sio`
call them via `use special::<mod>::*`, so a glob-import harness reaches them).

**Out of scope:** `caputo.sio`, `mittag_leffler*` (fractional calculus — no
standard mpmath/scipy reference), and `lib.sio` re-exports.

Target: **~40–50 functions.** Being deliberately broad.

## 4. Architecture — five pieces

### 4.1 Phase 0 — capability probes (before any harness)

Do not assume; probe each risky capability on the target engine first (the
CSV-reader lesson):

- **P0.1 fixed-point emit:** compile+run a tiny program that computes a known f64
  (e.g. `0.842700792949714 * 1e15`), converts `as i64`, and prints it via the
  integer path. Confirm the printed int round-trips to the value (validates
  `f64 → i64` numeric conversion + integer printing on the chosen engine).
- **P0.2 glob-import reach:** compile+run a program that `use special::bessel::*`
  and calls a non-`pub` `bessel_j0` — confirm it links (mirrors the e2e tests).
- **P0.3 bit-exact reinterpret (stretch):** probe whether `f64 → i64` *bit
  reinterpret* (not numeric convert) is available and correct (guard against the
  known Madaros f64-cast bug). If clean, upgrade the bridge to bit-exact.
- **P0.4 engine map:** for each family, record whether it compiles+runs under
  default Madaros or requires `SOUNIO_SOUC_ENGINE=lean_single`. The gate uses the
  cheapest engine that works per family; record it in the report.

Phase 0 output: a probe report (`PROBE_OK` sentinels + engine map) that fixes the
bridge choice and per-family engine before the emitter is written.

### 4.2 Number bridge

Primary: **fixed-point.** The Sounio harness emits, per evaluation, a line:

```
<family> <fn> <point_idx> <scaled_int>
```

where `scaled_int = round(value * SCALE)` printed as `i64`. Per-function choice:

- **O(1)-range functions** (erf, erfc, digamma, normal_cdf, regularized ibeta /
  igamma, bessel of moderate arg, orthopoly on [-1,1]): `SCALE = 1e15` →
  ~15 significant digits ≈ full double precision. i64 holds |value·1e15| for
  |value| ≲ 9000, which covers these.
- **Large-magnitude functions** (gamma, beta): compare the **log form**
  (`lgamma`, `lbeta`) which is O(hundreds); `SCALE = 1e9` keeps ~9–10 digits.
  Report notes gamma parity is measured via lgamma.
- **Unbounded/near-pole points** are chosen to keep outputs representable, or the
  point is dropped with a documented reason (no silent truncation).

Stretch (if P0.3 passes): **bit-exact** — emit the raw IEEE-754 bits of the f64
as i64; Python reconstructs the exact double via `struct.unpack`. Zero precision
loss, full range, still integer-only printing. Chosen automatically when the
probe confirms it.

### 4.3 Sounio emitter harness

`tests/parity/special_parity_emit.sio` (or per-family emitters if one module
can't hold all imports cleanly). Reads nothing at runtime — the point set is
compiled in from a committed table (§4.5). For every (function, point) it prints
one bridge line. Uses the sanctioned integer print path only; **no `f64 as
string`**.

### 4.4 Python reference + comparator

`scripts/parity/special_parity_ref.py`:

1. Loads the same committed point table.
2. Computes the mpmath ground truth at each point (`mp.dps = 30`), applying the
   same per-function form (log where the emitter used log) and SCALE.
3. Parses the Sounio emitter output.
4. For each function: `max_abs_err`, `max_rel_err`, worst point, and PASS/FAIL vs
   the per-family threshold (§4.6).
5. Emits a per-function table + an overall verdict line.

No scipy import; mpmath only. Deterministic (no RNG).

### 4.5 Point sets

A committed table `tests/parity/special_points.tsv` (or inlined constants),
per function, choosing representative domains:

- interior typical values,
- near-zero and near-singularity (documented margin),
- negative arguments where defined (erf odd, digamma poles avoided),
- moderately large arguments (bessel oscillatory region, gamma growth),
- known exact anchors (erf(0)=0, Γ(n) integers, J0(0)=1) as sanity rows.

~20–40 points per function.

### 4.6 Tolerance philosophy — the honest map

The point is measurement, not a forced pass:

- Each family gets a **calibrated threshold** set to its genuine achievable
  accuracy (some Sounio functions are truncated series — the current erf test
  already tolerates 1e-5).
- The gate **fails loudly** only on *gross* errors: wrong function, NaN/Inf where
  finite expected, or rel error > **1e-2**.
- The report records the **actual** achieved `max_rel_err` per function, so the
  artifact is a true accuracy map. Tightening a function's implementation later
  is a follow-up that tightens its threshold.
- Anchors (exact identities) are held to 1e-12.

## 5. Deliverables

- `scripts/special_scipy_parity_gate.sh` — orchestrator: Phase-0 probes → compile
  + run emitter (per-family engine) → pipe to comparator → emit
  `SPECIAL_SCIPY_PARITY_GATE_OK` + the per-function table. Dev-tier (needs
  mpmath; not wired into ci.yml unless mpmath is guaranteed on the runner).
- `tests/parity/special_parity_emit*.sio` — Sounio emitter harness(es).
- `scripts/parity/special_parity_ref.py` — mpmath reference + comparator.
- `tests/parity/special_points.tsv` — committed point set (or inlined).
- `docs/research/2026-07-19-special-scipy-parity.md` — the parity report:
  methodology, mpmath-reference rationale, per-function achieved rel error,
  engine map, and honest notes on weak functions.

## 6. Non-goals

- Not fixing/tightening any Sounio special-function implementation (that is a
  follow-up per function the map flags).
- Not installing scipy (mpmath is the reference).
- Not covering fractional-calculus functions (no standard reference).
- Not shipping anything into the prebuilt compiler (this is tests + scripts +
  docs only).

## 7. Verification of the vertical itself

- Phase-0 probes green (bridge + engine map fixed).
- The gate runs end-to-end and emits `SPECIAL_SCIPY_PARITY_GATE_OK`.
- Spot-check ≥3 anchor identities pass at 1e-12 (erf(0), Γ(5)=24, J0(0)=1).
- The report's numbers are reproducible: re-running the gate yields the same
  per-function `max_rel_err` (deterministic).
