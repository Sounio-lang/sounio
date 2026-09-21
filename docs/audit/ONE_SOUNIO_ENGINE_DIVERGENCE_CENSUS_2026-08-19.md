<!-- docs:meta
topic_id: repo.docs.audit.one-sounio-engine-divergence-census-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.one-sounio-engine-divergence-census-2026-08-19
-->

# One Sounio — complete engine-divergence census

> **Status**: census + semantic declaration | **Last validated**: 2026-08-19 | **Source**: overlay of measured corpora + live reconfirm of the three named cases. No engine was patched.

This workstream decides whether a specification exists. The founder’s directive is that **Madaros is the language**. `lean_single` is the bootstrap seed. It has no semantic authority. Where the two engines diverge, the divergence is a **lean_single defect** until there is documented proof that Madaros is wrong.

The machines do not agree. The seed is not an oracle. #1927 already proved the second sentence on three GUM pins.

Companion table: [`ONE_SOUNIO_ENGINE_DIVERGENCE_CENSUS_2026-08-19.tsv`](ONE_SOUNIO_ENGINE_DIVERGENCE_CENSUS_2026-08-19.tsv).

---

## Semantic declaration

Written before any new measurement command and before any file was classified.

```text
Semantic-Lane-ID:     WS-B-ONE-SOUNIO-20260819
Owner:                grok-cli3
Concept-IDs:          SOUNIO-CANONICAL-ENGINE (proposed; draft below).
                      Does not write docs/internal/concepts/registry.tsv
                      (held by grok-cli5 fo-import-port).
Intent-Preserved:     There is one Sounio. The default compiler defines it.
                      A second engine may exist as a bootstrap seed. It may
                      not define values, acceptance, or claims.
Transformation:       Authority inversion. Dual-engine disagreement stops
                      being a count and becomes a defect of the seed, except
                      where a documented proof names Madaros as wrong.
Types-Changed:        none
Effects-Changed:      none
IR-Changed:           none
Claims-Introduced:    Madaros is the language. A specification is declared.
                      The classified gap list is what would close it.
Claims-Forbidden:     "the engines agree";
                      "lean_single is the oracle";
                      "918/1759 is a defect rate";
                      "296 SPAN_DIFF is real span disagreement";
                      "835 RUNTIME_DIFF is numeric drift";
                      "CI green on the seed is Madaros green";
                      "Madaros var=0 is science when JCGM says otherwise".
Assumptions:          Founder directive of 2026-08-19 is binding.
                      Prebuilt Madaros ELF (2026-08-17 17:01, 99964767 B)
                      is the fleet surface. Not an E230 source-build.
                      Aug-17 check census and Aug-02 parity baseline remain
                      the measured populations; this lane does not re-sweep
                      1759×2 on the login pod.
Write-Set:            docs/audit/ONE_SOUNIO_ENGINE_DIVERGENCE_CENSUS_2026-08-19.md
                      docs/audit/ONE_SOUNIO_ENGINE_DIVERGENCE_CENSUS_2026-08-19.tsv
Read-Set:             docs/audit/CROSS_ENGINE_RUNPASS_CENSUS_2026-08-17.md
                      artifacts/research/cross_engine_runpass_census/
                      docs/audit/ENGINE_PARITY_ADJUDICATION_2026-08-02.md
                      tests/engine_parity_baseline.txt
                      tests/run-pass/lean_is_not_oracle_*.sio
                      tests/run-pass/bitwise_not_bootstrap_regression.sio
                      docs/audit/FO_VARIANCE_ACROSS_FN_INDEPENDENT_VERIFY_2026-08-18.md
                      docs/audit/KNOWN_FAILURE_NEEDS_AN_ENGINE_2026-08-19.md
                      docs/spec/LANGUAGE_SPECIFICATION.md §2.5
                      self-hosted/native/codegen_plan.sio (OpBitNot comment)
                      docs/EXACT_CORE.md
                      AGENTS.md claim-oracle paragraph
Positive-Witness:     #1927 three pins: Madaros prints the JCGM number,
                      lean_single prints the dropped-term number.
Negative-Witness:     Any sentence that treats lean_single PASS as the
                      language, or that treats engine disagreement as
                      "they mostly agree".
Acceptance-Gate:      This document classifies every measured family and
                      every named case. No compiler patch. No test retag.
Integration-Target:   origin/main documentation only.
Authoritative-Only-If: founder directive stands; live #1927 still fires;
                      Aug-17 / Aug-02 instruments remain the cited corpora.
```

### Proposed concept (draft; not registered this lane)

`SOUNIO-CANONICAL-ENGINE`

- **Intent.** One language. The user-facing compiler is the spec.
- **Canonical surface.** Default `bin/souc` → Madaros (`artifacts/self-hosted/madaros`).
- **Seed.** `lean_single` / `bin/souc-lean-single-x86_64` is the bootstrap fixed-point ELF. It is not a claim oracle.
- **Default owner of a divergence.** `LEAN_DEFECT`.
- **Exception.** `MADAROS_DEFECT` only with a documented proof that is not “the seed printed something else”. Allowed proofs: JCGM / GUM identity, IEEE bit class, a file’s own `//@ expect-stdout` that Madaros misses, a compiler comment that names a missing node.
- **Forbidden.** Dual-oracle wording. “CI pinned to lean, therefore lean is right.”

`docs/decisions/adr-008-claim-oracle-semantic-clock.md` is cited by `AGENTS.md` and `docs/architecture/F128_F256_LADDER.md`. **The file is absent from `docs/decisions/`.** The clock is practised (#1927) and now declared here. It is not a closed ADR.

---

## Does a specification exist?

**Declared. Not closed.**

| Question | Answer |
|---|---|
| Is there one Sounio? | Yes, by directive. Madaros. |
| Is there a closed written spec that both engines implement? | No. `docs/spec/LANGUAGE_SPECIFICATION.md` still lists `~` among the bitwise operators. Madaros maps `~` to `UnaryOp::OpNot` and rejects `~` on `i64` with E005. |
| Is there a closed executable spec (one engine, one CI oracle, classified exceptions)? | No. Check-phase disagreement on run-pass is **918/1759**. CI still has lean-shaped pins. |
| What would close it? | Every row in the companion TSV either converges or is filed as a documented exception. CI claim jobs run Madaros. The missing ADR is written. The seed is not consulted for values. |

A specification that needs two compilers to explain a program is not a specification. It is a census. This file is that census.

---

## What this is, and is not

**Is.** Classification of every *measured* divergence family, and of every named case the repository already knows, under the new authority. Live reconfirm of the three cases the founder named (`~`, accept/reject, variance) plus the #1927 pins.

**Is not.** A 1759×2 re-sweep on this pod. A compiler patch. A retag of tests. A claim that the engines agree. A claim that the seed is the oracle.

The Aug-17 instrument already said “this is not a defect rate; it is a disagreement census.” That sentence is still true of the *count*. It is no longer true of the *owner*. Disagreement is now a lean defect by default.

---

## Instruments

| Instrument | What it measured | Date | Use here |
|---|---|---|---|
| `scripts/research/cross_engine_runpass_census.sh` | 1759 run-pass `souc check` | 2026-08-17 | check-phase families |
| same, `--run` on 835 AGREE_ACCEPT | dual `souc run` | 2026-08-17 | rc_diff=83; banners are instrument |
| `tests/engine_parity_baseline.txt` + Aug-02 adjudication | 1007-row non-agree table: 70 DIVERGE, 358 MADAROS-ONLY, 268 LEAN-ONLY, 311 NEITHER | 2026-08-02 | runtime both-ran + capability |
| `scripts/ci/engine_parity_gate.sh` | produces the baseline | live | vocabulary only |
| #1927 `lean_is_not_oracle_{scale2,product,add}.sio` | GUM pins | merged 2026-08-19 | live reconfirm |
| `bitwise_not_bootstrap_regression.sio` | `~` | live today | tilde |
| `f256_v0b_arithmetic_rejected.sio` | E218 | live today | accept/reject |
| `docs/audit/FO_VARIANCE_ACROSS_FN_INDEPENDENT_VERIFY_2026-08-18.md` | FO arity / import | 2026-08-18 | MADAROS_DEFECT exception |
| `docs/audit/KNOWN_FAILURE_NEEDS_AN_ENGINE_2026-08-19.md` | 19 XFAIL-lean / XPAS-Madaros | 2026-08-19 | E218/E010/uninit/Vec |

No new full sweep. Fleet constraint: do not launch a 32-job census on the login pod.

### Binaries used for live reconfirm (2026-08-19)

| Surface | Path | bytes | mtime (UTC) |
|---|---|---:|---|
| Madaros | `artifacts/self-hosted/madaros` | 99964767 | 2026-08-17 17:01 |
| lean_single | `bin/souc-lean-single-x86_64` | 2555805 | 2026-08-17 13:57 |

Invocation: `$MAD compile <src> -o <elf>` then execute. Lean: `$LEAN <src> <elf>` then execute. Never bare `souc`.

---

## Owner vocabulary

| Owner | Meaning |
|---|---|
| `LEAN_DEFECT` | Default. Madaros defines the program. The seed is wrong or incomplete. |
| `MADAROS_DEFECT` | Documented proof that Madaros is wrong. The proof is never “lean printed something else”. |
| `INSTRUMENT` | The comparison measured banners, spans, or a missing file. Not a language fact. |
| `DECISION` | Same value, different print form. Neither engine computed a different number. |
| `CONTRACT` | Madaros implements a reservation or guard the seed lacks. Subclass of `LEAN_DEFECT`. |
| `RESERVA` | Protocol v2 (2026-08-19): compiler actively refuses **all** use of a kind with a named diagnostic, and no program that constructs the kind passes. Honest fail-closed slot. **Not** Claim-ready (refusing the right program is not typing) and **not** Hypothesis (the compiler is not silent). E218 f128/f256 is Reserva. |
| `SPEC_STALE` | Written spec still describes the seed. Flag on a `LEAN_DEFECT`, not a second language. |
| `HISTORICAL_LEAN_PIN` | A past closeout used the seed as authority. Recorded, not re-opened. |

---

## The three named cases, live today

### 1. `~` is OpNot on Madaros and bitwise on the seed

Witness: `tests/run-pass/bitwise_not_bootstrap_regression.sio`.

| Engine | Today |
|---|---|
| Madaros | `compile` rc=1, `error[E005] this unary operation is not defined for the type` on `~caller_eff` (`i64`) |
| lean_single | `compile` rc=0, ELF produced |

Madaros has no `OpBitNot`. The parser maps `~` onto `UnaryOp::OpNot`, the same node as `!`. On `i64` that is E005. On `bool` it is silent logical not. The seed implements two’s-complement complement (`~0 == -1`), which is what the run-pass file asserts.

`docs/spec/LANGUAGE_SPECIFICATION.md` §2.5 still lists `~` under **Bitwise**. That line is seed-era. It is not a documented proof that Madaros is wrong. The compiler already writes bitwise not as `0 - 1 - x` and says so in `self-hosted/native/codegen_plan.sio`:

> `MADAROS HAS NO BITWISE NOT`. Adding `OpBitNot` “is a language change, not a fix for this file.”

The same comment still calls `lean_single` “the oracle”. That sentence is now forbidden. It is evidence of the old authority, not a reason to keep it.

| Field | Value |
|---|---|
| Owner | `LEAN_DEFECT` + `SPEC_STALE` |
| Proof that Madaros is wrong | none that survives the directive |
| Convergence | Rewrite §2.5: `~` is `OpNot`, same node as `!`; i64 complement is `0 - 1 - x`. Retag the bootstrap file as compile-fail E005, or `known-failure-on: lean_single`. Do **not** add `OpBitNot` in this workstream. Adding it is a language change and needs a founder pin. |

### 2. Accept / reject

Live control: `tests/compile-fail/f256_v0b_arithmetic_rejected.sio`.

| Engine | Today |
|---|---|
| Madaros | rc=1, `error[E218]` V0-A reservation, three spans |
| lean_single | rc=0, ELF produced |

The seed accepts f256 arithmetic that the language has reserved. That is not “Madaros being strict”. It is the seed failing to implement V0-A.

Corpus (Aug-17 check, 1759 run-pass files):

| Bucket | N | Owner under this directive |
|---|---:|---|
| AGREE_ACCEPT | 835 | not a divergence |
| AGREE_REJECT | 6 | not a divergence |
| MADAROS_ONLY_REJECT | 230 | `LEAN_DEFECT` (seed too loose), unless a row has a Madaros-wrong proof |
| LEAN_ONLY_REJECT | 377 | `LEAN_DEFECT` (seed cannot accept the language) |
| DIAG_DIFF raw / strict | 15 / 6 | classify by code; default `LEAN_DEFECT` |
| SPAN_DIFF raw | 296 | `INSTRUMENT` — extractor matched bare `N:N`. Strict recount collapses almost all into agree-reject |

Accept-versus-reject **607** is the number to cite. 918/1759 is the raw diverge bucket including the instrument-inflated spans.

Top Madaros-only codes among the 230, all `LEAN_DEFECT` until a proof says otherwise:

| N | Code | Convergence |
|--:|---|---|
| 91 | NONE (no `error[E…]` wire) | give the seed the diagnostic wire, or stop claiming it typechecks these files |
| 42 | E137 unresolved name | seed is missing the name/import surface Madaros refuses |
| 13 | E009 type / generic | follow Madaros |
| 13 | E175 import / multi-module | follow Madaros |
| 10 | E004 parse | follow Madaros |
| 10 | E035 type | follow Madaros |
| 7 | E012 async | follow Madaros |
| 7 | E036 epistemic / measure | follow Madaros |

The 19 XFAIL-lean / XPAS-Madaros suite files (2026-08-19 receipt) sit inside this family:

| n | Files | Madaros | lean | Owner |
|--:|---|---|---|---|
| 16 | every E218 f128/f256 reservation in the 19 | E218 fires | 15 accept; 1 (`f128_v0b_implicit_conversion_rejected`) fails as “tail type mismatch”, still no E218 | `RESERVA` (kinds f128/f256) + `LEAN_DEFECT` (seed accepts). Not Claim-ready: both the construct program and the arithmetic program fail E218 on Madaros. |
| 1 | `turbofish_type_arg_arity` | E010 | accepts `first::<i64>(42, 99)` | `CONTRACT` / `LEAN_DEFECT` |
| 1 | `uninit_fixed_array_zero_init` | REAL_RUN, assertions hold | never typechecks (`unknown identifier s`) | `LEAN_DEFECT` |
| 1 | `vec_new_nonexistent_type_eval_zero` | prints `0` and `DOCUMENTS_FABRICATION` | `unknown identifier Vec` | split: seed does not know `Vec` (`LEAN_DEFECT`); Madaros fabricating `0` for a missing type is a **separate** `MADAROS_DEFECT` (W44 never fires) |

E219 (non-allowlisted `extern "C"`) is the same shape on compile-fail / FFI: Madaros refuses, the seed does not have the surface. `LEAN_DEFECT` / `CONTRACT`.

### 3. Variance — two sites, two owners. Do not collapse them.

**Site A — Madaros is right. This is #1927.**

Live today, same ELF pair:

| Pin | JCGM / GUM | Madaros | lean_single |
|---|---:|---:|---:|
| `lean_is_not_oracle_scale2` | `var(2x) = 4·var(x)` → 0.010000 | **0.010000** rc=0 | 0.002500 rc=1 |
| `lean_is_not_oracle_product` | `Var(a·b) = b²Va + a²Vb` → 0.032500 | **0.032500** rc=0 | 0.002500 rc=1 |
| `lean_is_not_oracle_add` | §5.1.2 independent sum → 5.000000 | **5.000000** rc=0 | 4.000000 rc=1 |

The seed drops a term and still exits a compiler. That is why “lean is the oracle” is a forbidden claim. Same-class not-yet-pinned files: `madaros_gum_fo_deep_poly`, `madaros_gum_multichannel_fo`, `madaros_gum_fo_div_if`, `madaros_gum_fo_interproc`, `madaros_gum_independent_product`.

Owner: `LEAN_DEFECT`. Convergence: port first-order transfer into the seed, or stop running these files on the seed. Do not “fix” Madaros to 0.002500 — #1927 is written to fail if anyone does.

**Site B — Madaros is wrong. Documented proof is JCGM, not the seed.**

| Witness | Expected | Madaros | lean_single | Proof |
|---|---:|---:|---:|---|
| same-file `add3` / `add4` (`docs/audit/repro/fo_var_samefile.sio`) | 14.0 / 14.25 | **0.000000** | 14.0 / 14.25 | JCGM 100:2008 §5.1.2; FO `>2 params: skip` in `fo_register_pure_fn_transfer` |
| imported 1-arg identity (`fo_var_import.sio`) | 4.0 | **0.000000** | 4.0 | transfer table never left the defining module |
| #1792 dissertation surfaces (`rapamycin_epistemic_adaptive`, door1 tail, ep28) | live variance | `var=0.000000` / bit-pattern confidence | ~1e-5 / ~1e-9 | fabrication detect gate; `docs/audit/EPISTEMIC_FABRICATION_DETECT_2026-08-17.md` |

Owner: `MADAROS_DEFECT`. The seed happening to print the JCGM number here does not make the seed the oracle. It makes this the documented exception the directive allows.

Convergence: close the arity>2 skip and the import-boundary transfer on Madaros. Do not pin the thesis cells to the seed. The independent-verify receipt already forbids the unconditional reading “every function boundary kills variance”.

**Formatter-adjacent tails** (door1 `0.000000` vs `2.756059e-17`; several dissertation PARITY lines): `DECISION` when both are the same number at six decimals; `MADAROS_DEFECT` when Madaros is the exact zero and the science requires a live residual. Do not pick a tolerance to make them match.

---

## Check-phase families (1759 run-pass, Aug-17)

Every explicit-diverge file falls in one family below. File-level TSV of all 918 is not reproduced here: the Aug-17 OrangeFS stage held it; this lane does not re-emit 918 rows from memory. Family coverage is complete. Individual-file re-adjudication of the 607 accept-versus-reject rows is the next lane, on Slurm.

| Family | N | Owner | Convergence |
|---|---:|---|---|
| AGREE_ACCEPT | 835 | none | keep as the dual-run population |
| AGREE_REJECT | 6 | none | — |
| SPAN_DIFF (raw) | 296 | `INSTRUMENT` | throw the raw extractor away; cite 607 + strict DIAG_DIFF |
| MADAROS_ONLY_REJECT / E218 in run-pass | 2 + compile-fail 16 | `CONTRACT` | seed must refuse V0-A or lose the files |
| MADAROS_ONLY_REJECT / other codes | 228 | `LEAN_DEFECT` | seed accepts what the language rejects; implement the refuse or drop the claim |
| LEAN_ONLY_REJECT | 377 | `LEAN_DEFECT` | seed cannot compile the language; port or stop citing seed coverage |
| DIAG_DIFF strict | 6 | `LEAN_DEFECT` default | read the six codes; Madaros code wins unless a proof says otherwise |
| DIAG_DIFF raw extra | 9 | mixed / one-sided E-code | treat as accept-versus-reject, not as two valid diagnostics |

Historical closeout **#1798** (Madaros accepted forward `inverse_of`, seed E158): closed by aligning Madaros to declaration-order. That used the **old** authority (seed as pin). Independent Sounio justification still holds — helpers are defined before callers; names do not exist forward. Record as `HISTORICAL_LEAN_PIN`, do not reopen.

---

## Runtime, both engines ran (70 DIVERGE, Aug-02 baseline)

These are the programs that **both** engines built and executed, then printed different bytes. Aug-02 already split them. Re-owned below. The committed baseline has **no AGREE rows**; it is the non-agree table (1007 lines).

### Madaros right — `LEAN_DEFECT`

| File | Why Madaros is the language |
|---|---|
| `lean_is_not_oracle_scale2` / `product` / `add` | #1927, live today |
| `madaros_gum_fo_interproc` | scale2=0.010000 vs lean 0.002500 |
| `madaros_gum_independent_product` | 0.032500 vs lean 0.002500 |
| `madaros_gum_fo_deep_poly` | first-order polynomial; expect-stdout names Madaros |
| `madaros_gum_fo_div_if` | same |
| `madaros_gum_multichannel_fo` | same |
| `epistemic_var_accumulator_slots` | expect-stdout `VAR_ACCUM_OK` |
| `global_array_element_list_init` | expect-stdout `10 20 30` / `LIST_OK` |
| `global_array_element_list_ident` and the six siblings (`call`, `call_args`, `call_args_multistmt`, `call_multistmt`, `cast`, `constfold`) | source is `[X, 20, Y]` with `X=7`, `Y=-3`. Truth `7 20 -3`. Seed prints `7 7 7` |
| `global_array_element_list_nonconst_failclosed` | same family |
| `global_array_i8_signed_element_list` | same family |
| `madaros_array_repeat_aggregate_distinct` | expect-stdout `ARRAY_REPEAT_DISTINCT OK` |
| `gpu_kernel_lane_loop` | #1512; GPU is a Madaros surface; seed has no GPU CLI |
| `print_f64_negative` | IEEE negative zero; `printf("%f", -0.0)` is `-0.000000` |

### Madaros wrong — `MADAROS_DEFECT` (documented)

| File | Proof |
|---|---|
| `closure_arity_2` | expect-stdout `PASS`; Madaros prints nothing. #1542 |
| `closure_returned` | same signature |
| `type_hash_3level_nesting` | expect-stdout `type-hash 3-level PASS`; prints nothing |
| `closure_escape` | expect-stdout `PASS`; DIVERGE on a print-nothing-shaped file |
| `correlated_eq_identity` | expect-stdout `ALL PASS`. July-29 doc says CLOSED; Aug-02 still DIVERGE. Residual large-struct copy aliasing is a named Madaros defect |
| `rapamycin_epistemic_adaptive` | #1792; expect-stdout includes `FAMILY_A_VAR_LIVE`; Madaros prints `var=0` |
| `rapamycin_rk4_budget` | same family |
| `vec_new_nonexistent_type_eval_zero` | Madaros prints fabricated `0` (not in the 70; listed so it is not lost) |

### Both satisfy the marker; leftover bytes differ — `DECISION` or inspect

`closure_effect_transparent_hof`, `door1_dense1024_epistemic`, `epistemic_mcts_full`. Door1’s uncertainty tail is the #1792 / formatter border: Madaros `-> 0.000000`, seed `-> 2.756059e-17`.

### Formatter, not value — `DECISION`

Aug-02 counted **19** of the 45 numeric DIVERGE as scientific-notation versus six-decimal fixed. Same number. Convergence: pick one print form. Cheap, not a science fix.

### Science / numeric remainder — default `LEAN_DEFECT`, escalate to `MADAROS_DEFECT` if the Madaros number is an exact fabricated zero

`dissertation_*` (7), remaining `gum_*` (compliance, correlated, euler, h1, iso, supplement1), `pbpk_*`, `rapamycin_epistemic_pbpk`, `rapamycin_iso_budget`, `epistemic_hessian_8inputs`, `epistemic_ode_14comp`, `imported_f64_lognormal_science`, `f2_conjugation_swda`, `m5_held_out_replication`, `graphics_epistemic_demo`, `interpolation`, `optimization_nelder_mead`, `sparse_matrix`, `test_integral_eq`, `text_interpolate`, `knightian_dark_matter`, `knowledge_octonion_structure`, `generic_knowledge`, `let_var_binding_name`, `plot3d_test`, `sret_forwarding_tuple_aggregate`, `madaros_root2_multimodule_method_chain`.

Aug-17 inspected samples: dissertation PARITY scientific tails and door1 are the ones already known to be zero-versus-tiny. Those escalate. The rest stay `LEAN_DEFECT` until a JCGM or expect-stdout proof names Madaros.

### FFI / stdlib OS — `CONTRACT` / `LEAN_DEFECT`

`ffi_integer_return`, `stdlib_mem_alloc`, `stdlib_os_process`, `stdlib_time_basic`. Madaros has an allowlisted real emitter and E219 outside it. The seed has `strip_extern_blocks()` stubs. Different stdout is not a second language. Convergence: seed follows the allowlist + E219, or these files stop running on the seed.

---

## Capability split (parity baseline, not check)

`MADAROS-ONLY` = only Madaros produced a running binary. `LEAN-ONLY` = only the seed did. `NEITHER` = neither did. These are not value disagreements.

| Class | N | Dominant prefixes | Owner |
|---|---:|---|---|
| MADAROS-ONLY | 358 | lorenz 172, solver 142, cardinality/lrat/sat | `LEAN_DEFECT` — the language runs these; the seed does not |
| LEAN-ONLY | 268 | graphics 24, viz 21, lorenz 21, solver 20, async 11, seq 11, ontology 10, heap 6 | default `LEAN_DEFECT` — extra-linguistic until a proof says the program is Sounio and Madaros is missing it |
| NEITHER | 311 | — | not a two-engine language fact |

LEAN-ONLY includes `bitwise_not_bootstrap_regression.sio` (case 1) and a graphics/viz/async cluster the seed still executes and Madaros does not. Under the directive those programs are **not Sounio** until Madaros accepts them, unless someone files a Madaros-wrong proof. That is the inversion. Previously this table was read as “Madaros cannot compile 268 files”. Now it is “268 files are seed-only until proven otherwise”.

Special LEAN-ONLY that already have a proof the other way: none on the tilde file (see case 1). FO import probes are not in this baseline.

---

## Dual-run rc_diff (Aug-17, 835 AGREE_ACCEPT)

| Reclass | N | Owner |
|---|---:|---|
| rc_diff | 83 | default `LEAN_DEFECT`; escalate any exact-zero science print to `MADAROS_DEFECT` |
| rc_same, lean stdout empty | 299 | `INSTRUMENT` / print-path |
| rc_same, both nonempty, hash differs | 453 | mix of Madaros `Compilation successful!` banners and real numeric drift; not a number to cite |

Cite **83**, not 835.

---

## What would close the specification

1. **Authority in CI.** Claim jobs run default Madaros. Seed jobs, if they remain, cannot fail a claim.
2. **Write the missing ADR** that `AGENTS.md` already cites (`adr-008`).
3. **Converge the `LEAN_DEFECT` rows**, seed toward Madaros, starting with: E218 refuse, `~`/§2.5 rewrite, global-array ident `7 7 7`, #1927-class FO terms, E219 allowlist.
4. **Fix the `MADAROS_DEFECT` rows** on Madaros: print-nothing closures (#1542), FO arity>2 and import transfer, #1792 fabricated zero, `Vec::new` evaluation of a missing type.
5. **Kill the instrument lies.** SPAN_DIFF extractor. `souc run` banners in any equality hash. known-failure tags without an engine (#1934).
6. **Stop new seed-era spec sentences.**

Until those land, the honest public sentence is: *Sounio is Madaros; the seed disagrees in the ways this census lists; some of those ways are Madaros bugs with proofs that are not the seed.*

---

## Claims this file refuses

- The engines agree.
- lean_single is the oracle. #1927 is the citation.
- 52.2% is a defect rate.
- 296 is a span disagreement.
- 835 is a runtime disagreement.
- Aligning Madaros to the seed is the default convergence.
- `var=0.000000` under Madaros is a measurement.

---

## Not done

- No compiler patch. No test retag. No `OpBitNot`. No FO transfer fix. No #1792 ABI fix.
- No live 1759×2 re-sweep.
- No file-level TSV of all 918 check divergences (family-complete; file-complete needs Slurm).
- `docs/internal/concepts/registry.tsv` not written (claimed by grok-cli5).
- Docs-registry sync of this topic is blocked on the same claim (`topic-registry.v1.json`, `DOCS_AUTHORITY_MATRIX.md`). Request sent.
- CAP / token table / handle table / E230 / pbpk_suite science not touched.
- #1930 and #1934 not babysat.

---

## Semantic outcome

```text
Semantic-Outcome:          specification declared; census classified; nothing patched
Concept-Status-Before:     dual-engine disagreement treated as a count;
                           seed still used as CI pin and, in comments, as oracle
Concept-Status-After:      one language (Madaros); seed is seed;
                           exception list is the MADAROS_DEFECT rows
Distinctions-Added:        LEAN_DEFECT vs MADAROS_DEFECT vs INSTRUMENT vs DECISION
                           vs CONTRACT vs SPEC_STALE vs HISTORICAL_LEAN_PIN;
                           variance site A (#1927) vs site B (#1792 / FO arity)
Distinctions-Preserved:    compile success != runtime parity;
                           uncertainty != ignorance;
                           IEEE zero != zero provenance
Distinctions-Erased:       none
Evidence-Run:              live compile+run of bitwise_not, E218 control,
                           lean_is_not_oracle_{scale2,product,add};
                           overlay of Aug-17 metrics.json and Aug-02 baseline
Fallback-Path:             none — seed is not a fallback language
Legacy-Kept:               lean_single as bootstrap seed / fixed-point ELF
Conflicting-Lanes:         grok-cli5 fo-import-port holds registry + authority matrix;
                           grok-cli5 lean-not-oracle already shipped #1927 (do not steal)
Next-Semantic-Interface:   file-level Slurm re-adjudication of the 607;
                           ADR-008 written; CI oracle flipped
```
