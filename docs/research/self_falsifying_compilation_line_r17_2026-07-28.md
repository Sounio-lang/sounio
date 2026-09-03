<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r17-2026-07-28
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r17-2026-07-28
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R17 — witness binding, in the compiler

**Date:** 2026-07-28
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `WITNESS_BINDING_IMPLEMENTED__REFUSES_ON_PRESERVED_PROPOSITION`
**Parents:** `self_falsifying_compilation_line_r15_2026-07-28.md` (the limit measured and the repair proposed), `self_falsifying_compilation_line_r16_2026-07-28.md` (the invariance group identified), `self_falsifying_compilation_line_r2_2026-07-26.md` (verdict-token binding — the mechanism this extends)
**Harness:** `scripts/research/self_falsifying_compilation_line_r17_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r17_gate.sh`

---

## 1. Result

> **The compiler now refuses to emit code when a claim's gate exits 0 and emits
> exactly the declared verdict token, but the evidence establishing it has been
> replaced.**

Verdict: `SELF_FALSIFYING_R17_VERDICT WITNESS_BINDING_IMPLEMENTED__REFUSES_ON_PRESERVED_PROPOSITION`.

**First compiler change since R2, ten rungs ago.** The R6–R16 arc was Python
throughout — a fact R11 §4 stated plainly and R12 diagnosed as the line's
central failure. This closes it, and it closes it on the repair the line's own
measurements identified rather than on a guess.

### 1.1 Observed behaviour

Built compiler: `artifacts/self-hosted/madaros-witness-binding`.

| probe | gate | rc | ELF | outcome |
|---|---|---:|---|---|
| W1 | token ✓ witness ✓ | 0 | yes | `CLAIM_PASS` |
| **W2** | **exit 0, token ✓, witness ✗** | **1** | **no** | **`CLAIM_WITNESS_MISMATCH`** |
| W3 | token ✓, no witness emitted | 1 | no | `CLAIM_WITNESS_ABSENT` |
| W4 | witness-changing gate, claim declares **no** witness | 0 | yes | `CLAIM_PASS` |

```
CLAIM_WITNESS_MISMATCH sfc_witness_drifts declared=abc123def456
  emitted=999999999999 (the proposition still holds; its grounds changed)
VERIFY_CLAIMS_FALSIFIED fail=1
```

**W2 is the rung.** Its gate exits 0, so exit-code gating (`build.rs`, and R0)
passes it. It emits exactly the declared token, so verdict-token binding (R2, the
line's contribution) passes it. The build is refused anyway. Nothing below this
mechanism can tell that case from a good build.

**W4 is the safety property.** A claim that declares no `witness` behaves exactly
as before, even against a gate whose witness moved. The mechanism is opt-in.

### 1.2 Regressions — shared code was modified

The token extractor was generalised rather than copied, so R2's path changed and
had to be re-verified, not assumed:

| | rc | ELF | outcome |
|---|---:|---|---|
| R2 token match | 0 | yes | `CLAIM_PASS` |
| R2 token drift | 1 | no | `CLAIM_TOKEN_MISMATCH` |
| R2 token absent | 1 | no | `CLAIM_TOKEN_ABSENT` |
| R0/R1 exit-code gating | 0 | yes | `CLAIM_PASS` + `CLAIM_SKIP` |
| R0/R1 no claims | 0 | yes | `VERIFY_CLAIMS_NOOP` |

### 1.3 The motivating case, bound (R18)

`zd_fiber_spectra_count_law_holds` is now in the R1 bound-claims manifest — the
first claim in this repository to declare a witness. Proposition: the ZD-fiber
adjacency spectra number 3·2^(n−5) for n = 5, 6, 7. Witness: a sha256 over those
spectra, sorted.

| | rc | ELF | |
|---|---:|---|---|
| real gate | 0 | yes | `VERIFY_CLAIMS_OK pass=16`, witness `705d0afd…` |
| perturbed twin | 1 | **no** | `CLAIM_WITNESS_MISMATCH`, emitted `e9f935cb…` |

The twin applies the count-preserving flip R15 measured and R16 explained. It
exits 0, reports 3/6/12 distinct spectra — the count law holds of the perturbed
algebra exactly as of the real tower — and emits the **same** verdict token.
Every spectrum differs, and only the witness records it.

The bound proposition covers n = 5, 6, 7 and says so: the n = 8 computation the
anomaly was first found on takes 86 s and would hit the executor's 30 s cap.

---

## 2. What was implemented

Confined to `self-hosted/compiler/claim_executor.sio`. **The parser needed no
change** — claim field names are not allowlisted, so `witness = "…"` parses
already (the 16-field cap is the only limit).

- A claim may declare `witness = "<fingerprint>"`.
- The gate's captured stdout+stderr is read for `<PREFIX>_WITNESS
  <fingerprint>`, taking the last occurrence, exactly as the token is read.
- `ce_witness_outcome` returns 6 (`MISMATCH`) or 7 (`ABSENT`); both refuse
  codegen. It runs **after** the token decision, so a claim declaring both must
  satisfy what it asserts *and* the grounds it asserts it on.

### 2.1 One derivation, two readers

`ce_extract_verdict_token` and `ce_extract_witness` both delegate to a single
`ce_extract_after(out, needle, len)`. Writing the scan twice would be precisely
the failure R6 measures — one derivation in two shirts, agreeing because it is
the same code rather than because two routes concur — committed inside the arc
that measures it. `X2` checks there is exactly one scan body, rather than
trusting the intent.

### 2.2 The R2 codegen hazard, respected

R2 recorded that assigning back into a variable its enclosing condition reads
does not stick on this path, and that the diagnosis cost four builds. The chain
is therefore `outcome` → `decided` (token) → `settled` (witness), each stage a
**fresh** variable, nothing written back. `X1` checks `var settled = decided` is
present, so a future refactor that reintroduces the hazard fails the clause.

---

## 3. What this is NOT

- **Not a solution to shared misinterpretation.** R0 §3 stands. A witness binds
  *which* evidence was used; it cannot tell whether that evidence is
  well-founded. If claim and check are wrong together, both agree on a witness.
- **Not automatic.** A claim must declare a witness and its gate must emit one;
  nothing here computes a fingerprint for anybody. **One production claim now
  declares a witness** (`zd_fiber_spectra_count_law_holds`, added in R18 —
  §1.3); the rest of the corpus is still unbound in R1's sense, 1 of ~295.
- **The real case is now bound — see §1.3.** As shipped, this rung's probes were
  fixtures with invented fingerprints, and the ZD-fiber contract that motivated
  it was not bound. R18 closed that; the concession is kept here rather than
  deleted, because what it conceded was true of this rung.
- **Not validated against concurrent compiles.** The capture path is still the
  fixed one R2 chose deliberately (a per-process path SIGSEGV'd the compiler);
  two simultaneous `--verify-claims` runs in one container would still collide.
- **Not a claim that witness binding catches more of this repo's history.** R4
  measured the historical arms at zero and nothing here re-runs that.

---

## 4. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r17_contract.py
# expect: X1 8/8 surface, X2 one scan body, X3 receipt bound to the executor sha,
#         SELF_FALSIFYING_R17_VERDICT
#           WITNESS_BINDING_IMPLEMENTED__REFUSES_ON_PRESERVED_PROPOSITION

bash scripts/ci/self_falsifying_compilation_line_r17_gate.sh
```

The contract checks **source surface plus a receipt**; it does not build a
compiler. To regenerate the receipt:

```bash
bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros-witness-binding
SFCL_R17_RUN_COMPILE=1 bash scripts/ci/self_falsifying_compilation_line_r17_gate.sh
```

The build is CPU-heavy and serialises on a global lock shared with other agents
in this workspace; budget tens of minutes. `X3` is bound to the executor's
sha256, so editing the executor invalidates the receipt and the clause says so
instead of certifying stale behaviour — the R2 lesson that *source surface is not
behaviour*.

---

## 5. AI disclosure

Executor change, fixtures, contract, gate and spec drafted under human direction
(2026-07-28). The W1–W4 and regression rows in
`artifacts/self_falsifying_r17_receipt.txt` are transcribed from an actual run of
a compiler built from this executor source, and the receipt is hashed to it. No
clinical content. GAIDeT-ICMJE 2025.
