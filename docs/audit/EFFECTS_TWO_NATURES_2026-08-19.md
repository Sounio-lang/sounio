<!-- docs:meta
topic_id: repo.docs.audit.effects-two-natures-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: cursor-3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.effects-two-natures-2026-08-19
-->

# Are Sounio effects one system or two? — measurement

**Date:** 2026-08-19  
**SHA scanned:** `e64b2cc495e31adf7505db4f8b39c1216d80edd1` (`origin/main` at census time; later `357ac9117d` is docs-only). Vocabulary lock: #1953.  
**Instrument:** `scripts/dev/effect_two_natures_census.py`  
**Compute:** Slurm `srun` via `scripts/dev/slurm_srun_minimal.sh` (`--partition=cpu-ops`, host `cpuops-t560-proxmox`). The node cannot see `/workspace`; the job received a `git archive` of `origin/main` `*.sio` on stdin and wrote JSON on stdout.  
**This is a measurement.** It does not rename an effect, propose a split, or edit `self-hosted/`.

**Hypothesis under test (not a conclusion):** the `with` vocabulary mixes an *observable* group (`IO`, `GPU`, `Prob`, `Observe`, `Async`) that a handler could intercept, and an *implementation* group (`Mut`, `Div`, `Panic`, `Alloc`) that records totality or memory. Motivating shape: `stdlib/math/pure.sio` `fn sin(x: f64) -> f64 with Mut, Div, Panic` — a pure `f64 -> f64` that declares implementation effects because of the loop and the divisions, not because it observes the world.

## Vocabulary (not invented)

`effect_name_to_id` in `self-hosted/check/effects.sio` names **23** effects (IDs in parentheses):

`IO`(0) `Mut`(1) `Alloc`(2) `Panic`(3) `Div`(4) `GPU`(5) `Async`(6) `Prob`(7) `Epistemic`(8) `Causal`(9) `Network`(10) `Sensor`(11) `Render`(12) `Observe`(13) `NonAssoc`(14) `Audit`(15) `Hypothesis`(16) `MultiTest`(17) `ZD`(18) `Witness`(19) `Temporal`(20) `Learn`(21) `Chaotic`(22)

The two-nature buckets above are the dispatch's hypothesis. They cover 9 of 23 names. The other 14 are reported as **ungrouped**; they were not forced into either bucket.

## Controls

**Negative (constructed; must stay zero).** After stripping `//`, `/* */`, and string bodies:

| case | signatures | known effects |
|---|---:|---|
| `// fn ghost() with Mut, Panic, Div { }` | 0 | — |
| `/* fn ghost() with Alloc { } */` | 0 | — |
| `fn main() with IO { println("with Mut, Panic, Div") }` | 1 | `IO` only (`Mut` from the string is absent) |
| comment `works with a map and with the compiler` | 0 | — |

`archive/`, `bootstrap/`, and `*.sio.old` are dropped before scan. A `.sio.old` body `fn ghost() with IO, GPU, Observe` *would* match if scanned (1 signature); it is not scanned.

**Positive.** Every effect with count > 0 has an exhibitable `with` clause (table + TSV). High count without a witness would have been a prose-pattern bug; `missing_witness_bug` is empty.

## 1. Census

Scanned **7662** versioned `.sio` files (git tree at the SHA, minus `archive/` / `bootstrap/` / `*.sio.old`). Found **57752** signatures that declare at least one vocabulary effect.

| effect | total | stdlib | self-hosted | tests | examples | other | witness |
|---|---:|---:|---:|---:|---:|---:|---|
| Mut | 51507 | 10615 | 21827 | 12289 | 5596 | 1180 | `stdlib/algebra/associator_field.sio:26` `with Mut, Div, Panic` |
| Panic | 48384 | 10201 | 21755 | 11517 | 3973 | 938 | same |
| Div | 46808 | 9501 | 20271 | 11382 | 4649 | 1005 | same |
| IO | 10956 | 1722 | 4554 | 2628 | 1520 | 532 | `stdlib/algebra/ladder.sio:143` `with IO, Mut, Panic, Div` |
| Alloc | 5597 | 1129 | 4136 | 225 | 55 | 52 | `stdlib/collections/heap_map.sio:53` `with Alloc, Mut` |
| Epistemic | 405 | 187 | 0 | 173 | 34 | 11 | `stdlib/darwin_pbpk/cumulants.sio:246` `with Mut, Div, Panic, Epistemic` |
| GPU | 193 | 5 | 14 | 123 | 51 | 0 | `stdlib/gpu/clifford_kernel.sio:96` `with GPU, Mut, Div, Panic` |
| NonAssoc | 86 | 18 | 0 | 30 | 35 | 3 | `stdlib/algebra/associator_field.sio:66` `with Mut, Div, Panic, NonAssoc` |
| ZD | 84 | 62 | 0 | 7 | 11 | 4 | `stdlib/epistemic/audited.sio:45` `with Mut, ZD` |
| Prob | 52 | 34 | 0 | 17 | 1 | 0 | `stdlib/chemistry/kinetics.sio:1030` `with Mut, Div, Panic, Prob, Observe, IO` |
| Observe | 44 | 20 | 0 | 22 | 2 | 0 | `stdlib/chemistry/kinetics.sio:1028` `with Observe` |
| Learn | 20 | 12 | 0 | 5 | 1 | 2 | `stdlib/learn/surgical_adam.sio:30` `with Mut, ZD, Learn` |
| Async | 20 | 0 | 0 | 20 | 0 | 0 | `tests/effects/archaeology/async_pass.sio:1` `with Async` |
| Causal | 18 | 3 | 0 | 15 | 0 | 0 | `stdlib/epistemic/composed_effects.sio:276` `with Causal` |
| Witness | 15 | 10 | 0 | 5 | 0 | 0 | `stdlib/regulatory/eu_aiact.sio:33` `with Mut, ZD, Witness` |
| Audit | 11 | 0 | 0 | 9 | 2 | 0 | `tests/effects/archaeology/audit_pass.sio:1` `with Audit` |
| Temporal | 11 | 6 | 0 | 5 | 0 | 0 | `stdlib/epistemic/revivable.sio:51` `with Mut, ZD, Temporal` |
| Hypothesis | 8 | 0 | 0 | 5 | 3 | 0 | `tests/run-pass/hypothesis_registered.sio:28` `with Hypothesis` |
| Chaotic | 7 | 0 | 0 | 6 | 1 | 0 | `tests/effects/archaeology/chaotic_pass.sio:1` `with Chaotic` |
| MultiTest | 4 | 0 | 0 | 4 | 0 | 0 | `tests/compile-fail/multitest_no_correction.sio:13` `with MultiTest` |
| Network | 0 | 0 | 0 | 0 | 0 | 0 | *(named in `effect_name_to_id`; no `with Network`)* |
| Sensor | 0 | 0 | 0 | 0 | 0 | 0 | *(same)* |
| Render | 0 | 0 | 0 | 0 | 0 | 0 | *(same)* |

Machine table: [`EFFECTS_TWO_NATURES_2026-08-19.tsv`](EFFECTS_TWO_NATURES_2026-08-19.tsv).

## 2. Co-occurrence

| class | n | % of 57752 |
|---|---:|---:|
| **implementation only** (`Mut`/`Div`/`Panic`/`Alloc`, and nothing else from the 23) | 46219 | **80.03** |
| at least one hypothesis-observable (`IO`/`GPU`/`Prob`/`Observe`/`Async`) | 11197 | 19.39 |
| of those, also at least one implementation effect | 9169 | 81.89% of the observable row |
| at least one ungrouped vocabulary effect | 635 | 1.10 |
| ungrouped only (no impl, no hypothesis-observable) | 100 | 0.17 |

Most common pairs: `Mut+Panic` 45536, `Div+Mut` 44447, `Div+Panic` 43879, then `IO+Mut` 8691.

The 80% implementation-only mass is the number the hypothesis asked for. It is **usage**, not a design proof.

## 3. Ceiling `[i64; 8]`

`extract_effects` stores at most 8 names (`effects.sio`).

| declared arity | signatures |
|---:|---:|
| 1 | 7971 |
| 2 | 6282 |
| 3 | 32506 |
| 4 | 8803 |
| 5 | 2175 |
| 6 | 15 |
| 7 | 0 |
| 8 | **0** |

Maximum observed: **6**, e.g. `stdlib/chemistry/kinetics.sio:1030` `with Mut, Div, Panic, Prob, Observe, IO`.

If `Div`/`Mut`/`Panic`/`Alloc` are removed from each signature, the leftover arity histogram is `{0: 46219, 1: …, 4: 2}`. **None** would meet or exceed 8. The cap is not binding on `origin/main` either with or without the implementation names.

## 4. Handlers

After the same comment/string strip, **`handle<Name>` is absent** for every vocabulary name, including both hypothesis buckets.

What exists is prose: parser comments (`handle<Effect> { body } with { handlers }`), `print("handle<")` in the printer (inside a string, not counted), and `tests/run-pass/handler_discharge.sio` saying `handle<IO>` is not supported. That is **usage of the comment form**, not a live handler program.

So: nobody wrote `handle<Div>` **and** nobody wrote `handle<IO>` on this SHA. The handler question does **not** separate the two natures. It says the handler surface is not a live corpus yet.

## Surface names outside `effect_name_to_id`

The scanner keeps comma-lists that start with an unknown token so `with Approx, Mut, Div, Panic` still counts `Mut`/`Div`/`Panic`. Unknown tokens are **not** added to the 23-name census. Most frequent leftovers: `Mod` 2813 (solver-portfolio fingerprints, `with Mod {`), `NaturalityG2` 35, `Approx` 31, `NonUnitary` 27. `Mod` and `Approx` are written as effects and are not in `effect_name_to_id`. That is vocabulary drift, recorded so it is not mistaken for `Mut`/`Div` inflation.

## Verdict for the founder

| question | result |
|---|---|
| Do most signatures declare only implementation effects? | **Yes.** 46219 / 57752 = 80.03%. |
| Is the `[i64; 8]` cap full? | **No.** Max 6; zero signatures at 8. Dropping impl names does not create an overflow either. |
| Did anyone `handle` an implementation effect? | **No live `handle<…>` at all** on this SHA (either bucket). |
| Two systems, as a language fact? | **Not proven.** The counts match the suspicion. The handler test cannot confirm it. A split remains a founder decision. |

The motivating `sin` row is real: `stdlib/math/pure.sio:245` `fn sin(x: f64) -> f64 with Mut, Div, Panic`.
