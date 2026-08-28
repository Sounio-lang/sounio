<!-- docs:meta
topic_id: repo.docs.audit.epistemic-self-application-audit-2026-06-03
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-self-application-audit-2026-06-03
-->

# Forensic audit: §6 "epistemic gradual compilation" self-application claim

**Date:** 2026-06-03. **Scope:** the paper
`docs/research/delta_epistemic_gradual_compilation_paper.md` §6 — the registered
"novel SOTA core" (a compiler applying its epistemic type system to its own
source). **Method:** read-only source forensics on
`self-hosted/compiler/lean_single.sio` + a live gate probe. **Posture:**
adversarial; ready for a negative result.

## What is REAL (the mechanism)

The epistemic pass and its gate machinery genuinely exist and are not stubs:
- `GATE_THRESHOLD = 950` (line 403); per-expression `EXPR_CONF`/`EXPR_GATE`
  arrays; gate set by `EXPR_GATE[p] = (EXPR_CONF[p] ≥ 950 ? 0 : 1)` (22167–72,
  22529–32) — genuine threshold logic.
- The `66 90` NOP "epistemic marker" is actually emitted at guarded call sites
  (`em(0x66); em(0x90)`, line 12602), counted (`EPIST_GATES_GUARDED/DIRECT`),
  and printed every compile as `gates[direct=N guarded=M]` (25328–29).
- Confidence is genuinely computed, not defaulted: `measured` fns → 990,
  `asserted` → 970 (21257–58); serial composition is multiplicative
  `op_conf = CONF[a]·CONF[b]/1000` (22148); `min`-propagation (22294, 22378); a
  `with Epistemic(N)` floor annotation parsed into `FN_EPISTEMIC_MIN` (24507).
  Serial composition **can** gate (`970·970/1000 = 940 < 950`).

So the pass is a real, substantive mechanism. **It is not degenerate.**

## What is OVERSTATED (the self-application claim) — the finding

§6.1 states: *"`lean_single.sio` … Its own source is annotated with
`Knowledge<T>` where appropriate — for example, variance estimates on
parser-rule transition probabilities, confidence values on type-inference
unifications."* **This is not borne out by the source.**

| Searched in the 27 kLoC compiler source | Count |
|---|---|
| `measure(` calls on compiler data | **0** (2 hits are comments about a slot array) |
| `var/let … : Knowledge<…>` declarations | **0** |
| compiler fns declared `measured` / `asserted` | **0** |
| `fn … with Epistemic` declarations on compiler fns | **0** (12 `with Epistemic` hits are all error-strings / comments / checker logic) |
| real `Knowledge<…> = measure(…)` anywhere in `self-hosted/` | **0** (the one hit is an **LSP snippet template string**, `lsp/templates.sio:272`, i.e. a template *for users*) |

The 45 `Knowledge<` occurrences are **type-checker infrastructure** — the `ETY_*`
table names, E170/E171 error strings, codegen `print("Knowledge<")` — i.e. the
machinery that checks **users'** Knowledge types. The compiler **checks**
`Knowledge<T>`; it is not itself **annotated** with it.

### Consequence: "100% certain / 0 guarded / 0 bytes" is vacuous for the compiler

Because the rules make `EXPR_CONF = 1000` for literals and multiplicative over
1000s, and the compiler's source injects **no** sub-1000 confidence (no
`measure`/`Knowledge`/`measured`/`asserted` on its own data), every expression
in the compiler is trivially confidence-1000 → `EXPR_GATE = 0` everywhere → 0
guarded. The headline "the compiler is 100% epistemically certain of itself" is
true only because **the compiler's source carries no uncertainty to be uncertain
about.** Cross-function propagation did not "discharge real guards"; there were
essentially none to discharge.

### The convergence table conflates two different metrics

"26% → 100% certain across 8 generations" measures **pass construct-coverage**
(which syntactic forms the epistemic pass can analyze; gen0 = literals only =
26%), *not* the epistemic certainty of an uncertain source. The §6.6 census
(`EPIST_GATES_DIRECT/GUARDED`) counts **call sites** and is a different,
expression-vs-callsite metric. The paper slides between "% certain expressions"
(113,931) and the call-site census; they are not the same measurement.

### Live-gate probe — the gate was never observed firing

Eight programs were compiled (committed binary `6374e52f`) and their
`gates[direct=N guarded=M]` read: `epistemic_propagation.sio` (7×`measure()`),
`pbpk_simple.sio`, `ekan_knowledge.sio`, `knowledge_associator.sio`,
`epistemic_quantum_vqe.sio`, `routon_projective_measurement.sio`, and a
purpose-built chained-`asserted` probe designed to drive the serial product
below threshold. **All eight reported `guarded=0`.** Across every tested
program — including measurement-heavy and adversarially-chained ones — **no
`66 90` marker was observed to be emitted.** The gate *can* fire by the
arithmetic of the rules (`970·970/1000 = 940 < 950`), but we did not witness it
firing on any program; its practical reachability is **unverified**, and even
the paper's historical "gen6 = 8.4% guarded" could not be reproduced here.

## Confidence tiers (what is proven vs. inferred vs. open)

- **Proven (assert plainly):** the compiler source has **zero** epistemic
  self-annotation (exhaustive grep — a presence/absence fact). The `Knowledge<`
  hits are type-checker machinery, not self-annotation. Self-application **as
  §6.1 describes it is not implemented.**
- **Observed:** across 8 tested programs the gate never fired (`guarded=0`
  uniformly); the `66 90` marker was not witnessed emitted.
- **Inferred (by inspection of the rules, not by observing a runtime dump):**
  `EXPR_CONF = 1000` throughout the compiler's own HIR (no sub-1000 injector is
  present in the source, and literals/ops are 1000-valued) ⇒ the "100% certain /
  0 guarded" self-result is vacuous. *Method note: read at the assignment sites;
  not confirmed against a dumped `EXPR_CONF` array.*
- **Inferred interpretation (not reproduced):** "26% → 100% certain" measures
  pass **construct-coverage** (which syntactic forms the pass can analyze), not
  source uncertainty; the gen0–gen8 table was **not** rebuilt.

## Honest verdict

- **Mechanism:** real and substantive (gate logic, marker, confidence algebra). ✔
- **Self-application as stated in §6.1:** **overstated / not implemented.** The
  compiler is the *type-checker for* `Knowledge<T>`, not a *user of* it; its own
  source is not epistemically annotated. The "100% / 0 guarded / 0 bytes" result
  is vacuous, and the convergence/census narrative conflates pass coverage with
  epistemic certainty.
- **The genuine contribution that survives:** Sounio implements a working
  `Knowledge<T>` epistemic type system (E170/E171, the gate/marker mechanism,
  the GUM-quadrature confidence algebra) that **users** can apply — demonstrated
  on clinical PBPK (§7). The *self*-application headline needs either (a) honest
  reframing ("the epistemic pass runs over the compiler's own HIR; the compiler
  source is not yet epistemically annotated, so its self-certainty is trivial"),
  or (b) actually annotating the compiler — real future work.

## Recommended paper actions (pending owner decision)

1. Correct §6.1: drop the "annotated with `Knowledge<T>` … parser-rule
   probabilities / inference unifications" claim or mark it as aspirational.
2. Reframe §6.4/§6.6: state that "% certain" is pass construct-coverage and the
   census is call-site count; do not present "100% / 0 guarded" as a deep
   epistemic property of the compiler.
3. Re-pin or retire the convergence table (the gen2==gen3 `54327028` md5 is
   likely stale; the current bootstrap fixed-point is a different hash chain —
   see memory `reference_souc_local_cap_e007`).
4. Keep the genuine, defensible claim: a working epistemic type system + gate
   mechanism for **user** code, validated on PBPK.
