<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r1-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r1-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R1 — binding the corpus: 16 real gates bound, and the module-closure wall measured

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `BOUND_16__MODULE_CLOSURE_PASSES` *(was `BOUND_16__MODULE_CLOSURE_BLOCKS` when measured 2026-07-26; the module-closure half was superseded by R29 on 2026-08-01 — see the notice below. The token follows the measurement, which is the only thing that ever moves it.)*
**Parents:** `self_falsifying_compilation_line_2026-07-26.md` (R0 audit: substrate live, corpus unbound), `self_falsifying_compiler_spec_2026-07-25.md` (the mechanism)
**Harness:** `scripts/research/self_falsifying_compilation_line_r1_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r1_gate.sh`

---

> **Supersession notice (2026-08-01).** The module-closure half of this rung no
> longer describes the compiler. R29 gave `claim_executor_verify` a walk over the
> module closure, and the same probe that produced `MODULE_CLOSURE_BLOCKS` here
> now produces `MODULE_CLOSURE_PASSES`: the imported false claim executes,
> `VERIFY_CLAIMS_FALSIFIED fail=1`, no ELF. R1's reading was correct on
> 2026-07-26 — it is superseded by a change to the mechanism, not retracted as an
> error, and nothing below has been rewritten to hide the flip. The binding half
> (16 claims bound to real gates) stands. See
> `self_falsifying_compilation_line_r29_2026-08-01.md`.
>
> This is the line doing to itself what it was built to do to everything else: a
> recorded rung was falsified by later work, and the gate that guarded it went
> red on its own before anyone went looking.


## 0. What this rung did

R0 measured that the mechanism guarded nothing: **0** native claims outside
tests and CI fixtures. R1 attaches real gates to real claims, and walks into the
module-closure limitation on purpose to see how hard the wall is.

> **16 real CI gates are now verified before codegen — and the guard cannot be
> put where the science lives.**

- **Binding works.** `examples/epistemic/rupture_claims_verified.sio` carries
  **16 native claims bound to 16 real CI gates** (no fixtures). Compiled with
  `--verify-claims`, every gate runs before codegen: `VERIFY_CLAIMS_OK pass=16`,
  ELF emitted, ~30 s wall-clock. Swapping one bound gate for an always-failing
  fixture gives `VERIFY_CLAIMS_FALSIFIED`, non-zero exit, **no ELF** — verified,
  not assumed.
- **The module-closure wall is real and measured.**
  `scripts/ci/fixtures/self_falsifying_modclosure_main.sio` imports a module
  whose claim's gate **always fails**. It compiles cleanly:
  `VERIFY_CLAIMS_OK pass=1`, ELF emitted, program runs. The imported false claim
  is **never executed**. Verdict: `MODULE_CLOSURE_BLOCKS`.

The second point is the one that matters. A library whose scientific premise has
been refuted passes silently into every dependent build. Claims can only do work
in a **main source file**, so binding today means hoisting claims out of the
libraries they describe and into a manifest that CI compiles. That manifest is
this rung's deliverable *and* the evidence of the limitation's cost.

---

> **COUNT MOVED, 15 → 16, AND HOW IT WAS FOUND.** R18 added
> `zd_fiber_spectra_count_law_holds` to this manifest — the first claim in the
> repository to declare a `witness`. The bound-claim count is a *measured*
> figure, so the token moved with it and this spec's headline went stale the
> moment that claim landed. Nobody noticed, because **R1's gate had never been
> run by CI** — the very condition `W4_CI_WIRING` has carried since R5. It
> surfaced only when the gates were finally wired, which is the argument for
> wiring in one sentence.

## 1. Results

| Clause | Result | Status |
|---|---|---|
| `B1_MANIFEST_BOUND` | 16 claims bound to real CI gates in a non-test, non-fixture source | binding achieved. |
| `B2_GATES_EXIST` | all 16 bound gate paths exist and are executable | no dangling bindings. |
| `B3_MODULE_CLOSURE` | probe is decisive by construction; recorded outcome `MODULE_CLOSURE_BLOCKS` | wall confirmed. |
| `B4_TIMEOUT_BUDGET` | 5 gates known to exceed the executor budget, none bound | exclusions explicit. |
| `B5_HERMETIC` | 1 gate known to mutate the working tree, not bound; static scan of the 15 clean | compiles do not dirty the repo. |

Compile-arm clauses, re-measurable with `SFCL_R1_RUN_COMPILE=1`:

| Clause | Result |
|---|---|
| `C1_MANIFEST_VERIFIES` | `VERIFY_CLAIMS_OK pass=15`, ELF emitted |
| `C2_RED_GATE_BLOCKS` | `VERIFY_CLAIMS_FALSIFIED`, non-zero exit, no ELF |
| `C3_MODULE_CLOSURE` | `VERIFY_CLAIMS_OK pass=1` while importing a false claim |

Verdict: `SELF_FALSIFYING_R1_VERDICT BOUND_15__MODULE_CLOSURE_BLOCKS`.

**On the verdict form.** R0's §5 fixed this rung's token as
`BOUND_N_OF_294__MODULE_CLOSURE_{BLOCKS,PASSES}`. That form embeds the gate-population
denominator, which moves whenever any gate is added — the token would drift
without the claim changing, which is precisely the **sub-token failure mode**
this line documents. The form is corrected here to carry the bound count only,
and R0's §5 is amended to match. Recording the correction rather than silently
substituting it is the point.

---

## 2. The module-closure experiment

Designed so that either outcome is informative, and so that a clean compile can
have only one explanation.

- `self_falsifying_modclosure_lib.sio` — an imported module carrying
  `mcl_library_claim_that_is_false`, bound to a gate that **always exits
  non-zero**.
- `self_falsifying_modclosure_main.sio` — imports it and calls a function from
  it (so the dependency is real, not decorative), and carries one claim of its
  own bound to an always-passing gate (so a null result cannot be mistaken for
  "verification never ran").

Measured 2026-07-26:

```
CLAIM_PASS mcl_main_claim_that_holds gate=scripts/ci/fixtures/self_falsifying_claim_gate_pass.sh
VERIFY_CLAIMS_OK pass=1
SELF_FALSIFYING_MODCLOSURE_MAIN_OK        <- program ran
exit=0, ELF emitted
```

`pass=1`, not `pass=2`, and no `CLAIM_FAIL`. The importer's own claim ran; the
imported claim did not. **`MODULE_CLOSURE_BLOCKS`.**

This confirms the mechanism spec's stated §6 limitation, but as an *observation*
rather than a note — and it converts it from a scoping caveat into the line's
main engineering obstacle. Independent verification of another agent's stated
limitation was the point of running it at all.

---

## 3. What binding costs, measured

A sample of 20 CI gates was timed with a 45 s probe (2026-07-26, this machine):

| Outcome | Count | Gates |
|---|---:|---|
| green, ≤ 12 s | 15 | `founder_intent_contract` 18 ms · `float_slot_capacity_coherence` 39 ms · `generated_ontology_manifest` 79 ms · `irfunction_instr_capacity_coherence` 104 ms · `g2_zd_fibers` 259 ms · `cd_tower_nullity_histogram_law` 376 ms · `ade_wildgen_mckay` 517 ms · `e_series_semantic_germ` 812 ms · `journal_submission` 867 ms · `self_falsifying_compilation_line` 1 182 ms · `associator_gum_variance` 1 564 ms · `functor_f_g2_covariance` 2 995 ms · `chingon_zd` 4 307 ms · `trigintaduonion_zd` 4 647 ms · `routon_zd` 11 948 ms |
| exceeded 45 s | 5 | `compiler_lane_status` · `heuristic_firewall` · `knowledge_context_static` · `zd_qec_prediction` · `falsification_ledger` |

A second probe ran every candidate gate and diffed `git status --porcelain`
before and after, to see which mutate the tree:

| Outcome | Count | Gates |
|---|---:|---|
| hermetic | 15 | (all others tested) |
| **mutates the working tree** | 1 | `associator_gum_variance` — rewrites `results/associator_gum_variance/{RUNLOG.txt,receipt.v1.json}` with the current timestamp and git SHA on every run |

Four constraints fall out, all of them load-bearing for later rungs:

1. **A quarter of the sample cannot be bound at all.** The executor's per-gate
   budget is 30 s (`CLAIM_GATE_TIMEOUT_MS`); 5 of 20 exceed even a 45 s probe.
   Note the two names in that list worth pausing on: `zd_qec_prediction` and
   `falsification_ledger` — **the falsification ledger's own gate is too slow to
   be a claim gate.** (These gates are not asserted to *fail*; only to cost more
   than the executor will wait.)
2. **Verification is serial.** One subprocess at a time, so the cost of a build
   is the *sum* of its claims' gates. 15 gates ≈ 30 s. Binding all 295 at
   observed rates would put a build in the tens of minutes — which is what makes
   the recorded-witness design of RQ2 a necessity rather than an optimisation.
3. **Gates are not necessarily hermetic — and one of the first 16 was not.**
   `associator_gum_variance_gate.sh` is green and fast, yet rewrites a receipt
   under `results/` stamped with the current time and git SHA. Binding it made
   every compile dirty the working tree and the build non-idempotent — noticed
   only because `git status` showed the damage after a routine run. It has been
   **unbound**, and `B5_HERMETIC` now refuses to let a known-non-hermetic gate
   back in. **Hermeticity is a bindability criterion, not just speed and
   colour**, and it is the sharpest argument yet for RQ2: if running a claim's
   check can change the tree, then "compile, then verify, then compile again"
   is not guaranteed to converge.
4. **Most specs cannot be token-bound yet — and the honest denominator is the
   wide one.** Of **269** documents in `docs/research/`, only **79** carry a
   `**Status:**` line at all, and only **24 (8.9 % of all specs)** declare a
   machine-parseable verdict token; the rest use prose
   (`` `HYPOTHESIS` → `EXECUTABLE` (target) ``). Quoted against the specs that
   already follow the Status convention it is 24/79 ≈ 30 %, which is the
   flattering framing — **R2 should plan against 8.9 %**, because a spec with no
   Status line needs the convention introduced before a token can be bound to
   it. R2's drift guard needs a token on both sides, so the overwhelming
   majority of the corpus needs a convention change before it can be guarded at
   all. Re-derive:

   ```bash
   ls docs/research/*.md | wc -l                                   # all specs
   grep -l -E '^\*\*Status:\*\*' docs/research/*.md | wc -l        # with Status
   grep -lE '^\*\*Status:\*\* `[^`]*` — `[A-Za-z0-9_]+`' docs/research/*.md | wc -l
   ```

---

## 4. Honest accounting of what is still unbound

**15 of 295 gates is 5.1 %.** The corpus is no longer at zero, and it is not
bound. Specifically not done:

- The remaining green gates from the sample (`routon_zd`, `zero_event`, and
  others) were left out to keep the manifest's build under a minute; that
  exclusion is a budget choice, not a technical limit.
- `associator_gum_variance` was bound, then **unbound** when it was caught
  mutating the tree (constraint 3 above). The removal is recorded in the
  manifest itself rather than erased.
- The 16 comment-form claims in `stdlib/epistemic/` were **not** converted.
  An earlier draft of this section justified that by saying they live in library
  modules and so could never execute — **that was wrong, and checking it was the
  point of checking.** `git grep` shows nothing imports
  `rupture_claims.sio`, `zero_provenance_claims.sio` or
  `zero_encounter_pipeline_claim.sio`; they declare no `module`, define no
  functions, and are comment-only files. They could therefore be converted to
  native syntax, given a `fn main`, and compiled as main sources exactly like
  the manifest. The honest reason they were not converted is that it is work
  this rung did not do — not a limitation of the mechanism.
- **The claims assert only what their gates check** — each hypothesis says "this
  contract's clauses hold", never restating the underlying mathematics. A claim
  bound to a gate may assert no more than the gate establishes; anything more
  would manufacture exactly the overclaim this line exists to study.

---

## 5. What this is NOT

- **Not corpus coverage.** 5.4 %, by design, on a hand-picked green-and-fast
  sample.
- **Not evidence the guard is useful.** R0 established that this class of gate
  would not have caught the corpus's historical errors. R1 shows it can be
  *attached*, not that attaching it prevents anything.
- **Not a fix for module closure.** The wall is measured, not moved.
- **Not a compiler change.** This rung adds a manifest, two probe fixtures, a
  spec, a harness and a gate; it touches no compiler source.

---

## 6. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r1_contract.py
# expect: B1..B4 PASS, SELF_FALSIFYING_R1_VERDICT BOUND_15__MODULE_CLOSURE_BLOCKS

bash scripts/ci/self_falsifying_compilation_line_r1_gate.sh
# expect: SELF_FALSIFYING_COMPILATION_LINE_R1_GATE_OK

# re-measure the compile-time facts (needs the claim-aware Madaros; ~1 min):
SFCL_R1_RUN_COMPILE=1 bash scripts/ci/self_falsifying_compilation_line_r1_gate.sh
# expect: C1_MANIFEST_VERIFIES, C2_RED_GATE_BLOCKS, C3_MODULE_CLOSURE all PASS
```

Directly, without the gate:

```bash
export SOUNIO_STDLIB_PATH=$PWD/stdlib && ulimit -s unlimited
RAW=artifacts/self-hosted/madaros-self-falsifying

# binding: 16 real gates run before codegen
$RAW run examples/epistemic/rupture_claims_verified.sio -o /tmp/m.elf --verify-claims

# the wall: a false claim in an imported module is invisible
$RAW run scripts/ci/fixtures/self_falsifying_modclosure_main.sio -o /tmp/mc.elf --verify-claims
```

Gate timings are machine-dependent; re-measure rather than quoting §3.

---

## 7. AI disclosure

Spec, manifest, probe fixtures, harness and gate drafted under human direction
(2026-07-26). All compile-time results were produced by real runs of the
claim-aware compiler and are re-measurable via the gate's compile arm. No
clinical content. GAIDeT-ICMJE 2025.
