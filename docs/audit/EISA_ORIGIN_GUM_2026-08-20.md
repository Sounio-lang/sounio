<!-- docs:meta
topic_id: repo.docs.audit.eisa-origin-gum-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-08-20
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.eisa-origin-gum-2026-08-20
-->

# R7 — `EReg.origin` so first-order GUM stops assuming independence

**Date:** 2026-08-20  
**Lane:** `lane/grok-cli5/eisa-origin-20260820`  
**Worktree:** `/workspace/.wt/eisa-origin`  
**Base:** `origin/main` `67aa2aec12`  
**Dispatch:** founder 2026-08-20, first of three EISA improvements (the only one that **corrects a number**).  
**Pair:** grok-cli4 R6 `Correlated` (type layer). This lane is the machine.

## Defect

`EReg { val, err, u }` propagated `u` under JCGM 100:2008 first-order GUM
**assuming independence**. At the ISA there was no way to know two registers
shared a measurement, so `x + x` produced `u' = u·√2` instead of `u' = 2u`.

Wrong on the metal, exactly as it was wrong in the type.

## Form

```
pub struct EReg { val: f64, err: Dd64, u: f64, origin: i64 }
```

Integer counter per measurement. Not a bit mask. Not slot identity.

## Fusion rule (must stay in the file and in the PR)

- Ids start at **1**.
- **0** is mixed/unknown — a sentinel no legitimate origin may occupy
  (contrast `SOUNIO-NO-VERSUS-UNKNOWN` and the colliding `-1` effect payload).
- `origin(op(a,b)) = origin(a)` if `origin(a) == origin(b)` and both ≠ 0;
  otherwise **0**.
- Correlation fires **only** when both operands have the same non-zero origin.

Consequence: **this never reports false correlation, and fails silent on
mixed.** `(a+b)+a` does not see the shared `a`, because `a+b` is already 0.
Conservative on purpose.

## Arithmetic

| Op | Same non-zero origin | Else |
|---|---|---|
| `eadd` | `u' = u_a + u_b` (ρ=1 envelope) | `√(u_a²+u_b²)` |
| `esub` | `u' = u_a + u_b` (same envelope, **not** \|u_a−u_b\|) | RSS |
| `emul` | **DUE** — RSS kept | RSS |
| `ediv` | **DUE** — RSS kept | RSS |
| `esqrt` | unary; origin preserved; `u' = u/(2z)` | n/a |

`esub` does **not** cancel. Signed ρ=+1 on `f=x−y` would give `|u_a−u_b|`
and `x−x` would report 0. The founder form is `u_a+u_b`, which is the
linear bound `|c1|u1+|c2|u2` and never underestimates. Documented in
`esub`. Conservative.

### `emul` / `ediv` — what I did and why

Product correlation is not the same correction as a sum. I can derive the
JCGM §5.2.2 formulae and I am **not shipping them**:

- `f=xy`, ρ=1: `u' = |y u_x + x u_y|`. Linear bound `|y|u_x + |x|u_y`
  (for `x*x` this is `2|x|u`, the derivative of `x²`).
- `f=x/y`, ρ=1: `u_c² = (u_x² + (q u_y)² − 2 q u_x u_y)/y²`. Signed form
  depends on the sign of `q`; linear bound `(u_x + |q| u_y)/|y|`.

Acceptance is `eadd`/`esub`. I will not ship a product formula I have not
pinned against an integer oracle. Fusion still runs on `emul`/`ediv`
(mixed product becomes origin 0). Marked **DUE** in `core.sio`.

## Constructors

| Constructor | `origin` | Why |
|---|---|---|
| `ereg_exact(v)` | **0** | A constant has `u=0`, so it cannot correlate. It is not a measurement. Origin 0 also stops `(c+x)+x` from looking like `x+x`. |
| `ereg_measured(v, σ)` | **0** | Legacy. Unknown origin. Two such values never fire, including `x+x` of a register built this way. Conservative residual until callers migrate. W2 non-regression depends on this. |
| `ereg_origin(v, σ, id)` | `id` if `id ≥ 1`, else **0** | Measurement constructor. Negative ids (the `-1` collision) are stored as 0. |

There is no process-wide counter. Auto-assigning unique ids needs `Mut`
module state this stdlib module does not have. Callers who need
correlation must pass an id ≥ 1.

## Acceptance

| Control | Result |
|---|---|
| `x+x` same origin 1, `u=3` → `u'=6`, origin 1 | **PASS** Madaros + lean_single. Integer oracle `3+3=6`. IEEE `3.0+3.0` is exact. |
| Distinct origins 1 and 2, `u=3,4` → `u'=5`, origin 0, **does not fire** | **PASS** both engines. Integer oracle `3²+4²=5²=25`. |
| Force `eisa_origins_correlated` always-equal, rebuild independent case | **PASS.** Independent then prints `u=7.000000` and exits 1. Proves the comparison is reached. |
| `(a+b)+a` origin 0, silent | **PASS** |
| `esub` same origin envelope 6, not 0 | **PASS** |
| Integer oracle `docs/audit/repro/eisa_origin/oracle.py` — no floating point | **PASS** (`ORACLE_OK`) |

Mandatory negative: if distinct origins had fired, this would not ship.
They did not.

## Non-regression (egate `k` untouched)

`egate` body and `k` are unchanged (`let _keep_k = k`). Call sites that
reconstruct `EReg` now pass `origin: 0` (machine register file has no
origin lane — residual below).

| Test | Result |
|---|---|
| `tests/stdlib/eisa/test_eisa_core.sio` W1–W5 | ALL PASS (Madaros + lean_single). W2 now also asserts `origin==0`. |
| `tests/stdlib/eisa/test_eisa_isa.sio` P1–P5 | ALL PASS (includes `egate`) |
| `tests/stdlib/eisa/test_eisa_evm.sio` V1–V5 | ALL PASS |
| `tests/stdlib/eisa/test_eisa_backend.sio` B1–B6 | ALL PASS |
| `test_eisa_h_zd`, `test_eisa_bridge`, `test_eisa_evm_v1`, `test_eisa_backend_v1`, `test_eisa_e5_kernel`, `test_eisa_v1e_showcase`, `test_eisa_evm_v2`, `test_eisax_v1_format`, `test_eisa_bridge_v1` | ALL PASS (Madaros) |
| `tests/stdlib/eisa/test_eisax_format.sio` | Madaros `run` rc=1, no stdout verdict (does **not** import `eisa::core`). lean_single `ALL PASS: eisax F1..F7`. File header: `validated_lane: lean_single`. Pre-existing engine split; not this lane. |

Gate: `scripts/ci/eisa_origin_gate.sh`.

## Coordination with R6 (`Correlated`)

Message sent to grok-cli4 lane `r6-correlated-20260820`.

| Type layer (R6) | Machine (R7) |
|---|---|
| Same `ExprIdent` (`m+m`) on Knowledge → require `Correlated` | Copied `EReg` keeps `origin`; `eadd(x,x)` fires ρ=1 |
| Distinct bindings `m1+m2` → no effect (negative must stay green) | Distinct ids, or both 0 → no fire |
| No provenance-string equality (false positives on shared labels) | Integer id, not a string |
| Not full dataflow (`f(m)+f(m)` out) | Mixed fusion to 0; `(a+b)+a` silent |
| `SOUNIO_FORCE_CORRELATED=1` positive control | Patch `eisa_origins_correlated` → always 1; independent fails with `u=7` |

They agree on conservative-certain. They do **not** yet share a wire:
Knowledge slot identity does not automatically stamp `EReg.origin`. That
bridge is a later lane. Until then, type warns and machine corrects only
when the caller uses `ereg_origin`.

## Residuals (honest)

1. **Machine register file has no origin lane.** `EMachine` / `EvmMachine`
   still store `val/ehi/elo/u`. `reg_to_ereg` / `evm_reg_to_ereg` reconstitute
   `origin: 0`. ISA-level `x+x` through the interpreter still uses RSS.
   Adding a lane would touch backend/bridge sizes and was out of the
   "correct a number in `EReg` ops" scope. Named, not hidden.
2. **`EReg2` / `core_v2.sio` untouched.** v2 `u` lane was copied verbatim
   from v0; it still assumes independence. Same defect, parallel type.
3. **`ereg_measured` origin 0.** `x+x` of a legacy measured register still
   uses RSS. The number is corrected when the caller uses `ereg_origin`.
4. **`emul`/`ediv` ρ=1 DUE.** Formulae derived in comments; not applied.
5. **Instruction quota and `egate` `k` not touched** (the other two EISA
   improvements).
6. **No auto-incrementing measurement counter.**

## Commands

```bash
cd /workspace/.wt/eisa-origin
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
env -u SOUC_BIN -u SOUNIO_SOUC_BIN
python3 docs/audit/repro/eisa_origin/oracle.py
./bin/souc run tests/run-pass/eisa_origin_correlated_add.sio
./bin/souc run tests/run-pass/eisa_origin_independent_add.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/run-pass/eisa_origin_correlated_add.sio
bash scripts/ci/eisa_origin_gate.sh
```

Force-equal (the gate does this with a trap that restores `core.sio`):
patch `eisa_origins_correlated` to `return 1`, rerun independent, observe
`u=7.000000` and rc=1.

## Not done

- Did not merge.
- Did not change branch during a remote build (none running on this lane).
- Did not edit `core_v2`, `egate` `k`, or instruction quota.
