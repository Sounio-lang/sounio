<!-- docs:meta
topic_id: repo.docs.audit.engine-parity-adjudication-2026-08-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.engine-parity-adjudication-2026-08-02
-->

# Engine-parity adjudication — the 70 remaining divergences

**Date:** 2026-08-02
**Baseline:** `tests/engine_parity_baseline.txt` after the seed refresh (#1606)
**Scope:** every `DIVERGE` entry, i.e. programs both engines build and run while
printing different bytes.

## Why there are 70 and not 212

Before refreshing the committed `lean_single` seed, the parity gate compared
against a reference built 2026-07-25 that lagged the source by two commits — one
of them titled *"indexed-element store family — five silent miscompiles"*. The
reference carried defects the source had already fixed, and every divergence
they caused was attributed to Madaros.

Refreshing it collapsed 142 divergences (`agree` 538 → 681). **The 70 that
remain are the ones worth reading**; the 212 were not adjudicable because most
were an artefact of a stale reference.

## Method

Three passes, cheapest first:

1. **Nature of the difference** — compare stdout with whitespace stripped, with
   line counts, and with non-digits stripped, to separate formatting from value.
2. **Objective verdict where the file declares one** — 17 of the 70 carry
   `//@ expect-stdout:`. Whichever engine prints the marker is right; no reading
   required.
3. **Source reading** for the rest.

## Result

| class | n |
|---|---:|
| numeric difference | 45 |
| line-count difference | 11 |
| text difference, digits identical | 14 |

Of the 45 numeric, **19 are the formatter, not the value**: lean prints small
magnitudes in scientific notation where Madaros prints fixed 6-decimal. A value
of `3.075134e-7` *is* `0.000000` at six decimals — both engines computed the
same number. That leaves 26 real numeric differences.

### Adjudicated by declared expectation (17 files)

**Madaros right, lean wrong — 8.** All are features Madaros implements and the
frozen seed does not:

- `madaros_gum_fo_deep_poly`, `madaros_gum_fo_div_if`, `madaros_gum_fo_interproc`,
  `madaros_gum_multichannel_fo` — first-order GUM variance
- `epistemic_var_accumulator_slots`
- `global_array_element_list_init`
- `madaros_array_repeat_aggregate_distinct`
- `gpu_kernel_lane_loop` — the per-lane loop from #1512

**Lean right, Madaros wrong — 4.** These are real Madaros defects:

- `closure_arity_2` — already filed as #1542
- `closure_returned` — same signature: compiles, runs, prints **nothing**, while
  lean prints `PASS`
- `type_hash_3level_nesting` — same signature, different subject
- `correlated_eq_identity` — Madaros prints `T1: FAIL`, `T3a: FAIL`, `T5cov: FAIL`,
  `SOME FAIL`; lean prints `ALL PASS`. Correlation/covariance tracking

**Both satisfy the marker — 5.** The divergence is in output the test does not
assert: `closure_effect_transparent_hof`, `door1_dense1024_epistemic`,
`epistemic_mcts_full`, `madaros_gum_independent_product`, `print_f64_negative`.

### Adjudicated by source (samples, each verified against the file's own stated truth)

`global_array_element_list_ident` — source is `var A: [i64; 3] = [X, 20, Y]` with
`X = 7`, `Y = -3`. Truth is `7 20 -3`. Madaros prints exactly that; **lean prints
`7 7 7`**, replicating the first element. Seven files share this family.

`madaros_gum_independent_product` — the source states the rule it is testing,
`Var(a*b) = b²·Va + a²·Vb`, with `a = 2.0`, `b = 3.0`, `u = 0.05`, and declares
`let want_prod = 0.0325`. Arithmetic: `9(0.0025) + 4(0.0025) = 0.0325`. Madaros
prints `0.032500`; **lean prints `0.002500`**.

`print_f64_negative` — `neg0 = 0.0 * (0.0 - 1.0)` is IEEE **negative zero**.
`printf("%f", -0.0)` prints `-0.000000`, which is what Madaros prints; lean drops
the sign. The file also computes `1.0 / neg0`, whose sign depends on this.

## What this changes

The headline is not the count. It is that **on every case adjudicable without
ambiguity, the split is 8 Madaros-right to 4 lean-right**, and the four are a
recognisable cluster: three of them compile, run, exit 0 and print *nothing*.
That is the same empty-green class this repository keeps rediscovering — a
program that produces no output scoring the same as one that produces the right
answer.

## Open work, in priority order

1. **The print-nothing cluster** — `closure_arity_2` (#1542), `closure_returned`,
   `type_hash_3level_nesting`. One symptom, probably one cause. Highest value:
   it is a silent wrong answer, not a loud one.
2. **`correlated_eq_identity`** — Madaros's correlation tracking disagrees with
   lean's on six of twelve assertions. Related to #1535, which records that lean
   divides correlated variance by use count and that both engines lose it in
   loops.
3. **The 19 formatter divergences** — decide which notation is canonical for
   small magnitudes and make both engines agree. Purely a decision; neither is
   computing anything wrong.
4. **The 53 without `//@ expect-stdout`** — they cannot be adjudicated
   objectively. Adding a marker to each is the cheapest way to make the parity
   gate self-adjudicating for the next reader, and it is mechanical work.

## Reproduce

```bash
# nature of each difference
grep '^DIVERGE' tests/engine_parity_baseline.txt | cut -f2   # the 70

# objective verdict for the 17 that declare one
sed -n 's|^//@ expect-stdout:[[:space:]]*||p' <file>          # the marker
./artifacts/self-hosted/madaros compile <file> -o /tmp/m.elf && /tmp/m.elf
./bin/souc-lean-single-x86_64 <file> /tmp/l.elf && /tmp/l.elf
```
