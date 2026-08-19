<!-- docs:meta
topic_id: repo.docs.audit.provenance-keyword-coverage-dispatch-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.provenance-keyword-coverage-dispatch-2026-08-19
-->

# Dispatch: provenance keyword coverage (unwritable enum cases)

**Date:** 2026-08-19  
**sha measured:** `92fade0be1` (`origin/main`)  
**Kind:** forensic dispatch — **no `self-hosted/` change in this PR**  
**Related ruling (blocked):** founder 2026-08-19 withdraw example `label` vocabulary; map **`asserted → Input`** (`docs/spec/S08_EPISTEMIC_VALUES.md` §8.5(c)). That ruling is **not executable** until surface syntax exists for `AstProvInput`.

---

## Semantic lane declaration

```text
Semantic-Lane-ID: prov-keyword-coverage-dispatch-20260819
Owner: grok-cli4
Concept-IDs: SOUNIO-PROVENANCE (surface syntax debt); references S08 / label withdrawal
Intent-Preserved: provenance kinds remain meaningful; this dispatch only costs making them writable
Transformation: none to compiler — document only
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced:
  - three of six AstProvenanceKind variants are unwritable on current main
  - silent skip of unknown Knowledge annotation components (two parser sites)
  - validity annotations are not subject to the same hole (3/3 reachable)
  - hole is origin-of-design (same commit as six-case enum), not a later regression
Claims-Forbidden:
  - that this PR implements the fix
  - that asserted→Input is executable today
  - that PascalCase Input/Source/Literature collide with versioned user code at scale
    (measured: Input 0, Literature 0, Source 2 files outside archive)
Assumptions: Knowledge provenance components are written as PascalCase keywords
  matching existing Derived/Computed/Measured (lexer tables.sio)
Write-Set: docs/audit/PROVENANCE_KEYWORD_COVERAGE_DISPATCH_2026-08-19.md only
Read-Set: self-hosted/parser/{ast,types}.sio, lexer/{token,tables}.sio, S08, git history
Positive-Witness: TokenKind::Derived/Computed/Measured branches construct AST
Negative-Witness: AstProvSource|Literature|Input occur only at enum declaration
Acceptance-Gate: founder chooses enum-complete keywords vs enum shrink; optional E-code
Integration-Target: future parser/lexer PR after ruling
Authoritative-Only-If: line numbers re-verified on stated sha
```

---

## 1. Evidence (re-verified)

### 1.1 Enum declares six

`self-hosted/parser/ast.sio:452-459`:

```text
pub enum AstProvenanceKind {
    AstProvDerived,
    AstProvSource,
    AstProvComputed,
    AstProvLiterature,
    AstProvMeasured,
    AstProvInput,
}
```

### 1.2 Parser reaches three

`self-hosted/parser/types.sio` — Knowledge component loop (angle-bracket form), **lines 1107–1119**:

| token | constructs |
|---|---|
| `TokenKind::Derived` | `AstProvDerived` |
| `TokenKind::Computed` | `AstProvComputed` |
| `TokenKind::Measured` | `AstProvMeasured` |
| else | **silent skip** (`// Unknown component — skip` + `p.advance()`) |

Second path (bracket form), **lines 1379–1390**: same three arms; else advances with **no comment** — same silent discard.

### 1.3 Unwritable variants (occurrence counts in `self-hosted/parser/`)

| variant | occurrences | sites |
|---:|---:|---|
| `AstProvDerived` | 3 | enum + 2 construct sites |
| `AstProvComputed` | 3 | enum + 2 construct sites |
| `AstProvMeasured` | 3 | enum + 2 construct sites |
| **`AstProvSource`** | **1** | **declaration only** |
| **`AstProvLiterature`** | **1** | **declaration only** |
| **`AstProvInput`** | **1** | **declaration only** |

Dispatch claim of “one each for Source/Literature/Input” is **confirmed**. Derived/Computed/Measured are **3 each**, not “only declaration” — the orchestrator’s “three token kinds construct three” is correct.

### 1.4 Lexer has three provenance keywords only

`self-hosted/lexer/token.sio:151-153` — `Derived`, `Computed`, `Measured` only.  
No `TokenKind::Input`, `::Source`, `::Literature`.

`self-hosted/lexer/tables.sio` `keyword_lookup` (PascalCase byte match):

- length 7: `"Derived"` → `TokenKind::Derived` (~line 124)
- length 8: `"Computed"` → `Computed` (~135), `"Measured"` → `Measured` (~137)

There is nothing to parse for Input/Source/Literature even if the parser arms existed.

### 1.5 Control: validity is complete

`AstValidityKind` (`ast.sio:441-445`): three cases.  
Parser (`types.sio:1074-1106`): `Valid` / `ValidUntil` / `ValidWhile` → all three.  
**Hole is provenance-specific**, not a general annotation-parser pattern.

### 1.6 Origin commit — not a regression

| fact | evidence |
|---|---|
| Six-case enum present | `f9da2142f4` (2026-02-27) `parser/ast.sio` — rename to `AstProvenanceKind` already has six variants |
| Three parser arms + silent else | same commit `parser/types.sio` (~511–521 in that tree): Derived/Computed/Measured + `// Unknown component — skip` |

**The parser was never six-wide.** Enum and three-arm parser were introduced together at Stage 1 bootstrap. This is **design lag (enum ahead of surface)**, not a later deletion of keywords.

---

## 2. Proposed fix A — make kinds writable (detail for cost, not implementation)

### 2.1 Minimum for founder ruling `asserted → Input`

| step | where | estimate |
|---|---|---|
| Add `TokenKind::Input` | `lexer/token.sio` near Derived/Computed/Measured; update `tk_is_keyword` | ~2–4 lines + any exhaustiveness dumps in `main.sio` / bootstrap mirrors |
| Recognise `"Input"` (length 5) | `lexer/tables.sio` `keyword_lookup` | ~1 branch (5-byte compare), same style as Derived |
| Parser arms | `parser/types.sio` **both** Knowledge component loops (~1117 and ~1389) | ~4 lines each site: `else if ck == TokenKind::Input { … AstProvInput }` |
| Tests | e.g. extend `self-hosted/test_knowledge.sio` style lex/parse of `Knowledge[T, Input]` | new/extended cases |
| Bootstrap mirrors | `bootstrap/bootstrap_v0.sio` / stage1 copies if still required to match | must not drift from tables |

**Scope if only Input:** ~15–40 lines across lexer+parser+tests, **plus** bootstrap parity if those trees remain authoritative for seed.

### 2.2 If enum is the design (all six surface keywords)

Also add `TokenKind::Source`, `TokenKind::Literature` and matching `keyword_lookup` arms (`"Source"` len 6, `"Literature"` len 10) and parser arms → `AstProvSource` / `AstProvLiterature`.

Same dual-site parser update pattern. Roughly **3×** the Input-only lexer/parser surface (three keywords instead of one).

### 2.3 Casing

Existing provenance keywords are **PascalCase** (`Derived`, not `derived`). Proposed surface: **`Input` / `Source` / `Literature`**, not snake_case, to match the lexer.

Lowercase `input` as an ordinary identifier would **remain** `TokenKind::Ident` under the current keyword_lookup design (exact byte match including case). Collision risk is therefore about **PascalCase** names, not the common `input` parameter name.

### 2.4 Soft/contextual keyword option

`token.sio` already documents contextual keywords (~613+). An alternative is to accept `Input` only inside Knowledge annotation components (parser-driven), keeping it out of the global keyword table. That reduces any residual clash with `Source` in packages but is a **different** design (more parser special-casing, less “keyword like Derived”). Cost/complexity higher than three table rows; founder choice.

---

## 3. Proposed fix B — silent `else` → named diagnostic (independent)

### 3.1 Behaviour today

Unknown component in `Knowledge<…>` / `Knowledge[…]` annotation: **advanced past, no error, provenance left unchanged** (or prior component kept). A typo `Meassured` or a not-yet-keyword `Input` is **invisible**.

### 3.2 Proposed behaviour

On unrecognised component token in that loop: emit a **named** diagnostic and fail closed (do not silently drop).

### 3.3 Error code

Used E2xx on this tree include: 200–207, 213, 216, 218, 219.  
**Free nearby:** among others **E208–E212, E214, E215, E217, E220+**.

**Recommendation:** **`E220`** (free; sits after E218/E219 epistemic/parser-adjacent family) or **`E217`** if a tighter cluster with E218 is preferred.

Suggested message shape (English):

```text
error[E220]: unknown Knowledge type annotation component
note: expected epsilon bound, Valid/ValidUntil/ValidWhile, or provenance
      Derived | Computed | Measured [| Input | Source | Literature when keywords land]
note: unrecognised components are no longer ignored
```

Until keywords land, the note should list only the three writable kinds so `Input` is not advertised early—or list Input as “not yet a keyword” if the diagnostic lands first.

### 3.4 Sites

Both loops in `types.sio` (~1117 and ~1389). One shared helper preferred to avoid a third silent path later.

### 3.5 Independence

Fix B does **not** require Fix A. Landing B first makes the founder’s `asserted → Input` attempt **fail loudly** instead of silently no-oping—better than today’s false parse.

---

## 4. Compatibility risk (measured)

### 4.1 PascalCase keyword collision (relevant if matching Derived style)

| proposed keyword | versioned `.sio` hits (excl. `archive/`) | notes |
|---|---:|---|
| **Input** | **0 files / 0 lines** | clean |
| **Literature** | **0 / 0** | clean |
| **Source** | **2 files / 10 lines** | only `packages/epistemic-core/src/lib.sio` (`struct Source`, helpers) |
| Derived (existing) | 12 / 28 | mostly bootstrap/compiler mirrors |
| Computed (existing) | 8 / 17 | same |
| Measured (existing) | 8 / 17 | same |

**Source** is the only new PascalCase collision surface of note; it is confined to `packages/epistemic-core`. Options: rename package type, or contextual keyword, or accept break in that package.

### 4.2 Lowercase identifiers (not keywords under current lexer)

| word | files | lines | remark |
|---|---:|---:|---|
| `input` | 88 | 2182 | common parameter name — **safe** if keyword is `Input` only |
| `source` | 89 | 1395 | same |
| `literature` | 1 | 3 | `stdlib/graphics/view.sio` counters |

Do **not** introduce lowercase `input` as a hard keyword without a migration plan—that would be a large break. The existing provenance keywords set the precedent: **PascalCase only**.

---

## 5. What the founder must decide

### 5.1 Enum complete vs enum shrink

| option | reading of evidence | work |
|---|---|---|
| **A — Enum is design; parser/lexer lag** | Six cases at birth with three arms → surface unfinished | Add keywords for all missing (or at least **Input** for the ruling) + parser arms |
| **B — Enum over-grew** | Three arms + three tokens are the real design | Delete `AstProvSource`, `AstProvLiterature`, `AstProvInput` (and any dead matches) |

**Evidence leans A:** single commit introduced six-case enum **and** three-arm parser; no later commit removed Input/Source/Literature keywords. S08 §8.2.3 and §8.5(c) already treat Input as **owed**, not surplus.

**Evidence for B would look like:** docs or checker only ever mentioning three kinds, or commits deleting keyword arms. **Not found** on this measurement. Examples’ `asserted`/`constant` labels are a **third vocabulary**, not proof the enum should shrink to three.

### 5.2 Input-only vs full six

Founder named **`asserted → Input`**. Minimum executable path: **Input keyword + arms + Fix B**.  
Source/Literature can follow or ship together; S08 still calls Source vs Input the “only real decision” for asserted mapping—if Source stays unwritable, that decision stays half-stuck.

### 5.3 Diagnostic-only first?

Recommended sequencing if risk-averse:

1. **Fix B** (E220 silent-component) — no new keywords, behaviour change: typos become errors.  
2. **Fix A Input** — unblocks asserted→Input withdrawal.  
3. **Source/Literature** — if enum-complete chosen.

### 5.4 Not in this dispatch

- Implementing any of the above.  
- Changing `examples/` labels (blocked on language).  
- PROV↔ontology reasoner bridge (separate Phase 1 audit).

---

## 6. Acceptance sketch (for the future implementer)

| gate | pass condition |
|---|---|
| Parse `Knowledge[f64, Measured]` | still works |
| Parse `Knowledge[f64, Input]` | constructs `AstProvInput` (after Fix A) |
| Parse `Knowledge[f64, NoSuchProv]` | **error[E220]** (after Fix B), not silent success |
| `souc check` on `packages/epistemic-core` | green or documented Source rename |
| Bootstrap keyword tables | bit-match lexer tables if still concatenated |

---

## Reproduce

```bash
# Enum
sed -n '452,459p' self-hosted/parser/ast.sio

# Parser arms + silent else
sed -n '1107,1119p' self-hosted/parser/types.sio
sed -n '1379,1390p' self-hosted/parser/types.sio

# Occurrence counts
rg -c 'AstProvSource|AstProvLiterature|AstProvInput' self-hosted/parser

# Birth
git show f9da2142f4:self-hosted/parser/types.sio | rg -n 'Unknown component|AstProv'
```
