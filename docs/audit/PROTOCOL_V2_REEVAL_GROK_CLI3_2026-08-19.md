<!-- docs:meta
topic_id: repo.docs.audit.protocol-v2-reeval-grok-cli3-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.protocol-v2-reeval-grok-cli3-2026-08-19
-->

# Protocol v2 re-evaluation — what grok-cli3 actually classified

> **Status**: v3 converted — positions are no longer authority | **sha_main**: `a4d44ec22c11` | **v3 blocker**: `msg-1787111556-3882032-13231`

**Protocol v3.** A handwritten position rots. This document's table is a
snapshot. Authority is `tests/kind-census/index.tsv` plus
`scripts/ci/kind_census_fixtures_gate.sh`. The index stores kind, two
paths, expected diagnostic, deepest named layer. It does **not** store a
position. The gate derives the table.

The 16 founder-named effects (IO…ZD) are owned by codex-1 at
`tests/effects/archaeology/`. f128/f256 TypeKinds are owned by grok-cli4
at `tests/typekind/`. This lane converts the rest of what it classified:
the extra seven ladder names, Foo, and the CD/ExactRing surfaces.

> **Status (v2, superseded)**: re-ran that turn | **blocker**: `msg-1787110825-3708985-4281`

Read the blocker first. The defect was the protocol, not the measurements. This
note applies the three v2 rules to **this lane's** prior positions. It does
not re-census the 99 TypeKinds. Families F+H are owned by grok-cli4
(`TYPEKIND_ARCHAEOLOGY_FH_V2_2026-08-19.md`); B by cursor-2; G by grok-cli5.

```text
REGRA 1  escada monótona: Claim-ready ⇒ Executable ⇒ Hypothesis ⇒ Garden
         sem programa que construa o kind e PASSE, máximo = Hypothesis
REGRA 2  Reserva (fora da escada): recusa ACTIVA de todo o uso com
         diagnóstico nomeado; nenhum uso passa. Não é Hypothesis
         (lá o compilador cala) nem Claim-ready (recusar tudo não
         é discriminar). Vale mais que Hypothesis.
REGRA 3  todo o tipo deve existir em todas as camadas. A linha
         regista a camada mais profunda que ainda o NOMEIA.
TESTE    dois programas. Ambos falham = Reserva. Certo passa e
         errado falha = Claim-ready. Sem os dois, ≤ Hypothesis.
```

Instrument: prebuilt `artifacts/self-hosted/madaros` (99 964 767 B, 17 Ago).
`check` only. No rebuild. lean_single is not the oracle.

Table: [`PROTOCOL_V2_REEVAL_GROK_CLI3_2026-08-19.tsv`](PROTOCOL_V2_REEVAL_GROK_CLI3_2026-08-19.tsv)

---

## What I had classified (inventory)

| Surface | v1 position I actually wrote | v2 |
|---|---|---|
| 23 ladder effect names | "names exist in the ASCII ladder"; positive witness file with **uncalled** `fn w_*` | **Claim-ready** as *E035 members* — two-program test this turn |
| unknown `Foo` | SILENCE; "the system accepts any spelling" | **Hypothesis** (etiqueta). Both programs pass. Not Reserva: no named diagnostic |
| f128 / f256 via E218 (one-Sounio `CONTRACT`) | "reservation doing its job"; seed is `LEAN_DEFECT` | **Reserva**. Both construct and arithmetic fail E218. Not Claim-ready. grok-cli4 already has the same correction on F+H |
| `#651` / concrete ℚ CD | science unblocked (`c0=1/1`, `RAT-ZD PROVED`) | not a TypeKind. Bug status, not a garden rung. Stands |
| `CDElementExact<Rational>` | E011, rc=1 | residual compiler defect, not a TypeKind position |
| TypeKind `Sedenion` / `Octonion` | I never classified them. CD work used `[i64;16]` / `[Rational;16]` | **must not** inherit Executable from RAT-ZD PROVED. Regra 3: those names exist **only in HLIR** (`HlirTypeOctonion`, `HlirTypeSedenion`). Checker does not name them. Debt both ways |
| `SOUNIO-EXACTNESS` | proposed, not registered | Garden seed. No TypeKind. No two-program test. ≤ Hypothesis as a concept, not a kind |

I did **not** mark F128 Claim-ready. The one-Sounio census called E218 a
`CONTRACT`. Under v2 that word is too close to Claim-ready. It is Reserva.

---

## Two-program receipts (this turn)

Shape for an effect `E`:

```sounio
// certo — must pass
fn f() -> i64 with E { 0 }
fn main() -> i64 with E { f() }

// errado — must fail
fn f() -> i64 with E { 0 }
fn main() -> i64 { f() }
```

| kind | posição | camada_mais_profunda | certo | errado | sha_main |
|---|---|---|---|---|---|
| IO | Claim-ready | checker | rc=0 | E035 missing IO | a4d44ec22c11 |
| Mut Alloc Panic Div GPU Async Prob Observe | Claim-ready | checker | rc=0 | E035 | a4d44ec22c11 |
| NonAssoc ZD Witness Audit Chaotic Temporal Learn | Claim-ready | checker | rc=0 | E035 | a4d44ec22c11 |
| Epistemic Causal Network Sensor Render Hypothesis MultiTest | Claim-ready | checker | rc=0 | E035 | a4d44ec22c11 |
| Foo | Hypothesis | checker | rc=0 | rc=0, no diagnostic | a4d44ec22c11 |
| f128 | Reserva | parser (E218) | E218 | E218 | a4d44ec22c11 |
| f256 | Reserva | parser (E218) | E218 | E218 | a4d44ec22c11 |

Chaotic's E035 prints `missing: Effect#22` — `print_effect_name` still has no
arm. That is diagnostic debt, not a position change. The pair still
discriminates.

### What Claim-ready means here, and what it does not

For the 23 names, the checker **keeps** the name and **refuses** a caller that
lacks it. That is the two-program test. It is **not**:

- GPU requiring a GPU
- ZD detecting a zero-divisor
- "the effect system supports 23 effects"

The last sentence remains forbidden. `Foo` is silence. A set that accepts an
unregistered name is not closed. Claim-ready-on-E035 and open-complement are
both true. Do not collapse them.

The file `tests/run-pass/effect_set_positive_witness.sio` is **not** a
two-program test. Its `fn w_zd() -> i64 with ZD { 0 }` is never called. Under
v2 that file alone cannot lift anyone above Hypothesis. The receipts above
replace it.

E231 (named unknown-effect) is still unbuilt on the prebuilt ELF. It would
move `Foo` from Hypothesis toward Reserva **only if** every use of `Foo`
then failed with E231 and no use passed. Until that ELF exists, Foo stays
Hypothesis.

---

## One-Sounio E218 — correction

v1: `CONTRACT` / `LEAN_DEFECT` — "Madaros implements a reservation the seed
lacks."

v2: **Reserva** for the *kinds* f128 and f256. The seed still accepting
arithmetic is a `LEAN_DEFECT`. Madaros refusing every construct **and** every
wrong use with the same E218 is Reserva, not Claim-ready. Refusing the
program that should pass is the definition.

This agrees with grok-cli4's F+H v2 table. I do not re-own those rows.

---

## CD archaeology — regra 3, not a promotion

`RAT-ZD PROVED` is a 16-component product over `Rational` / `i64`. The
coefficients are `TyI64` or a stdlib struct. They are not
`HlirTypeSedenion`. Promoting TypeKind Sedenion because a `[T;16]` loop
annihilates would be the same class of error as marking F128 Claim-ready
because E218 fires.

Layer debt named by the blocker, relevant to this lane's subject:

- checker → HLIR: most TypeKinds die before HLIR (19/99 named through)
- HLIR → checker: `Octonion Sedenion Quat* Dual Vec* Mat*` exist only
  as `HlirTypeKind`. The language cannot construct them. Both directions
  are debt.

Exactness (`ab == 0` in a ring, no rounding) is still not a Concept-ID.
That gap is unchanged. Not a TypeKind.

---

## Counts (this lane only)

| posição | n | what |
|---|---:|---|
| Claim-ready | 23 | every ladder effect name, E035 pair |
| Reserva | 2 | f128, f256 (E218 both sides) |
| Hypothesis | 1 | Foo (silence both sides) |
| Garden | 0 | |
| TypeKinds censused | 0 | not this lane |

---

## Not done

- No TypeKind family census. No write to `registry.tsv`.
- No Madaros rebuild. E231 still source-only.
- No handle-table raise.
- Seed not ported.
- Effect names were not traced into HLIR/IR/codegen as *named kinds*.
  Deepest measured layer is checker. That is already debt under regra 3
  if an effect is a kind that must exist in every layer.
