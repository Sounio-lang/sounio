<!-- docs:meta
topic_id: repo.docs.audit.cd-exact-rational-archaeology-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.cd-exact-rational-archaeology-2026-08-19
-->

# Archaeology — Cayley–Dickson exactness over ℚ was already unblocked

> **Status**: measured 2026-08-19 on prebuilt Madaros | **Last validated**: 2026-08-19

The founder confirmed: wide integers in Cayley–Dickson are for **exactness**, not
precision. Precision preserves digits. Exactness forbids rounding. `i512` is a
declared seed, not a gap.

Three layers of lost information sat in the same pair of stdlib files. This
note re-ran the evidence against today's Madaros and corrected the headers
that were still telling the next person not to try.

**Protocol v2 (2026-08-19).** `RAT-ZD PROVED` does not promote TypeKind
`Sedenion` or `Octonion`. Those names exist only as `HlirTypeKind`; the
checker does not name them (regra 3, both directions are debt). This lane's
witnesses are `[i64;16]` / `[Rational;16]`. Exactness is still not a
Concept-ID. See
[`PROTOCOL_V2_REEVAL_GROK_CLI3_2026-08-19.md`](PROTOCOL_V2_REEVAL_GROK_CLI3_2026-08-19.md).

**Claims-Forbidden:**

- “F = Rational is blocked by generics”
- “F = Rational is blocked by a `[struct;N]` codegen bug”
- “the generic skeleton does not compile yet”
- “exactness is a kind of precision”
- “raising the handle table would finish this”
- “the engines agree”
- “#651 is still open as filed”

No handle-table raise. No E230. No compiler patch. Header honesty plus a
named residual.

---

## Semantic declaration

Written before any source edit.

```text
Semantic-Lane-ID:     CD-EXACT-RATIONAL-ARCHAEOLOGY-20260819
Owner:                grok-cli3
Concept-IDs:          SOUNIO-PRECISION-PRESERVATION
                      SOUNIO-HYPERCOMPLEX-ZD-EVIDENCE
                      SOUNIO-EXACTNESS (proposed name only; not registered)
Intent-Preserved:     Exact Cayley–Dickson arithmetic forbids rounding.
                      `ab == 0` is a decidable equality in an exact ring,
                      not a tolerance-gated float measurement. Precision
                      (more digits) is a different contract.
Transformation:       none to types, effects, IR, or science. Recover
                      lost status: the concrete ℚ product already runs;
                      the generic `<Rational>` path fails a *different*
                      way than the headers claim.
Types-Changed:        none
Effects-Changed:      none
IR-Changed:           none
Claims-Introduced:    Concrete [Rational;16] CD multiply on today's
                      Madaros prints the exact value `c0=1/1`. The
                      hand-monomorphized sedenion product over ℚ prints
                      RAT-ZD PROVED. Generic F=i64 still proves ZD.
                      Generic F=Rational is blocked by E011 on trait
                      methods of an *imported* struct, rc=1 — not 139,
                      not 182, not 12.
Claims-Forbidden:     see box above
Assumptions:          Prebuilt artifacts/self-hosted/madaros (mtime
                      2026-08-17, 99964767 B) is today's Madaros. No
                      source rebuild. lean_single is not the oracle.
Write-Set:            stdlib/algebra/cayley_dickson_exact_i64.sio
                      (header only)
                      stdlib/algebra/cayley_dickson_exact.sio
                      (STATUS / impl warning only)
                      docs/audit/CD_EXACT_RATIONAL_ARCHAEOLOGY_2026-08-19.md
Read-Set:             docs/handoff/repros/d8_generic_struct_F_mul_segv.sio
                      docs/handoff/repros/cd_exact_rational_pending651.sio
                      docs/handoff/repros/cd_exact_rational_concrete_madaros_ok.sio
                      tests/run-pass/cd_exact_generic_i64.sio
                      docs/internal/concepts/precision-preservation.md
                      docs/internal/concepts/hypercomplex-zero-divisor-evidence.md
                      docs/internal/concepts/SEMANTIC_LANE_CONTRACT.md
                      self-hosted/native/gc.sio
                      docs/audit/HYPERCOMPLEX_651_ROOTCAUSE_2026-07-14.md
                      docs/handoff/compiler_651_defects_codex_dispatch_2026-07-15.md
Positive-Witness:     d8 ELF rc=0 stdout `c0=1/1`
                      cd_exact_rational_concrete_madaros_ok `RAT-ZD PROVED`
                      cd_exact_generic_i64 `ZD PROVED` + 16× `COMP i 0`
                      cd_basis_exact::<Rational> prints `c1=1/1`
                      local-struct ExactRing generic add prints `sum=2/1`
Negative-Witness:     cd_exact_rational_pending651 8× E011, check rc=1
                      /tmp/er_rational_min.sio E011 on add_er::<Rational>
Acceptance-Gate:      headers no longer name #650 or #651-as-filed as
                      live blockers; residual named as E011 imported-
                      struct trait dispatch
Integration-Target:   origin/main (header honesty). Do not merge the
                      still-open PR #816 without re-running it.
Authoritative-Only-If: the commands in §3 are re-run on the same ELF
```

### Concept gap — exactness is not in the registry

`SOUNIO-PRECISION-PRESERVATION` is about `f128` / `f256` / `dd64` / `qd128`:
narrowing must be explicit; more digits must not silently fall back to
`f64`. That is **precision**.

`SOUNIO-HYPERCOMPLEX-ZD-EVIDENCE` is about tokens versus receipts, ordered
pairs, and the `i64` ±1 envelope. It uses exact product equality, but the
concept is **evidence**, not the ring.

Neither Concept-ID is named exactness. The founder distinction — *precision
preserves digits; exactness forbids rounding* — is not a row. This lane
does **not** write `docs/internal/concepts/registry.tsv`. A neighbouring
worktree was on `concept/exactness-20260819` at measurement time; if that
lane registers `SOUNIO-EXACTNESS`, it is the right owner.

Draft contract, not registered:

```text
Proposed-ID:          SOUNIO-EXACTNESS
Intent:               An exact ring computation may not round. Equality
                      is decidable in the ring. A wider integer is a
                      capacity for unrounded coefficients, not a claim
                      of more decimal digits.
Must-not-collapse-to: SOUNIO-PRECISION-PRESERVATION
                      SOUNIO-HYPERCOMPLEX-ZD-EVIDENCE
```

---

## Instrument

```text
worktree:   /workspace/.wt/grok-cli3
branch:     lane/grok-cli3/effect-set-as-data-20260819 @ 354ad1ce4c
compiler:   artifacts/self-hosted/madaros
            99964767 B, mtime 2026-08-17 17:01
identity:   Madaros v0.80.0
stdlib:     $PWD/stdlib
not-used:   bin/souc wrapper (resolves to /workspace/sounio)
            lean_single
            any source rebuild of Madaros
handle table in this source: native_v2_handle_table_capacity_default = 4194304 (2^22)
                             raised from 2^20 on 2026-07-26; comment in
                             self-hosted/native/gc.sio:44–64
```

---

## Layer 1 — the generic skeleton is not blocked on #650

`stdlib/algebra/cayley_dickson_exact_i64.sio` said the generic sibling
“does not compile yet — blocked on compiler features: generic-struct-return,
`impl Trait for Type`, trait-bounded dispatch.”

Those features landed **2026-07-06**, commit `2adb8f061`, PR #650, on main.
The comment was six weeks stale.

Re-measured: `tests/run-pass/cd_exact_generic_i64.sio` against today's
Madaros.

```text
rc=0
ZD PROVED
SQ PASS
NONZERO PASS
COMP 0 0
… through …
COMP 15 0
```

The generic engine over `F = i64` is live. The concrete `*_i64` file is a
zero-trait fallback, not the only working path.

---

## Layer 2 / 3 — #651 as filed is gone; the d8 repro prints 1/1

`cayley_dickson_exact.sio` still said `F = Rational / BigInt` was blocked
by a `[Rational;N]` multiply-accumulate miscompile (garbage at N=16,
SIGSEGV at N=2048), reproduced with no generics, filed as #651, repro
`docs/handoff/repros/d8_generic_struct_F_mul_segv.sio`.

The 2026-07-14 forensic note already re-diagnosed this: not `[struct;N]`,
but native handle-table wrap at 2^20 plus multimodule thin-link rc=12
(`docs/audit/HYPERCOMPLEX_651_ROOTCAUSE_2026-07-14.md`,
`docs/handoff/compiler_651_defects_codex_dispatch_2026-07-15.md`). Issue
**#651 is CLOSED** (2026-07-15T11:35:34Z). The handle table in this
source is 2^22. Nobody put that fact back into the stdlib headers.

### Command 1 — the d8 repro, today

```text
artifacts/self-hosted/madaros check  docs/handoff/repros/d8_generic_struct_F_mul_segv.sio
  CHECK_RC=0

artifacts/self-hosted/madaros compile … -o /tmp/d8_repro.elf
  COMP_RC=0
  chmod +x /tmp/d8_repro.elf && /tmp/d8_repro.elf
  ELF_RC=0
  stdout: c0=1/1
```

Historical symptom at N=16 was garbage (`c[0] ≈ 4.2e6/1`). Today the
exact value is `1/1`. Not 139. Not 182. Not 12. **Pass.**

### Command 2 — concrete sedenion product over ℚ

`tests/run-pass/cd_exact_rational_concrete.sio` is **not on main**. It
lives on `origin/pr/816` / `work/sr651-madaros-witness` (`8d98d1b095`).
PR **#816 is still OPEN**, never merged. That is why the “proof landed”
sentence in the July audit is not executable from `origin/main`.

The same science is in
`docs/handoff/repros/cd_exact_rational_concrete_madaros_ok.sio`:

```text
rc=0
RAT-ZD PROVED
RAT-SQ PASS
sq c0=-1/1
```

Exact annihilation of `(e3+e10)(e6−e15)` over ℚ, on today's Madaros,
with no generics. The science the headers still call blocked **runs**.

---

## The residual is not #651 — it is E011 on imported-struct trait methods

### Command 3 — generic skeleton over Rational

`docs/handoff/repros/cd_exact_rational_pending651.sio` instantiates
`cd_mul_exact::<Rational>`. Its own header says it “COMPILES cleanly now
(the defect is runtime)”.

Today:

```text
check rc=1
8 × error[E011] … no method named for this type
  cd_add_exact, cd_sub_exact, cd_mul_exact (×5), cd_is_zero_exact
```

rc=1, diagnostic E011. **Not 139. Not 182. Not 12.** The July runtime
story is stale. This is a new (or re-emerged) checker residual and
deserves its own dispatch. Do not reopen #651.

Construction without a trait-method call on `F` is fine:

```text
cd_basis_exact::<Rational>(4, 1, rat_zero(), rat_one())
rc=0
stdout: c1=1/1
```

So the exact.sio line “WORKS for construction/dispatch” is still true.
The call `a.er_add(b)` when `F = Rational` is what dies.

### Isolating E011 — not CD, not any struct, imported struct

| Probe | Instantiation | rc | stdout / diagnostic |
|---|---|---:|---|
| `/tmp/er_i64_min.sio` | `add_er::<i64>` | 0 | `sum=2` |
| `/tmp/er_local_struct.sio` | `add_er::<Frac>` (same-file struct) | 0 | `sum=2/1` |
| `/tmp/er_rational_min.sio` | `add_er::<Rational>` (imported struct) | 1 | E011 `no method named` |
| `cd_basis_exact::<Rational>` | construction only | 0 | `c1=1/1` |
| `cd_mul_exact::<Rational>` | pending651 | 1 | 8× E011 |

`souc check stdlib/algebra/cayley_dickson_exact.sio` also reports E011.
That is a category error: it is a generic library checked with unbound
`F`. Existence is a caller. The i64 caller exists and is green.

**Proposed dispatch title (not opened here):**

> Madaros E011: trait-bounded method on an *imported* struct `F`
> (`impl ExactRing for Rational`) — local struct and `i64` monomorphise.

---

## What was not done

- No handle-table raise (already 2^22; the wall moved, it was not removed).
- No compiler patch, no E230, no E231 in this lane.
- No promotion of `pending651` into `tests/run-pass/` (it is red, E011).
- No merge of PR #816.
- No write to `docs/internal/concepts/registry.tsv` or
  `docs/governance/topic-registry.v1.json`.
- No return to the effect-set-as-data rebuild in this note.

---

## Header corrections in this lane

1. `cayley_dickson_exact_i64.sio` — stop naming #650 features as blockers.
2. `cayley_dickson_exact.sio` STATUS — #651-as-filed is closed; residual
   is E011 on imported-struct trait dispatch.

The next person who reads those files should try the concrete ℚ path,
and should not file another `[struct;N]` issue.
