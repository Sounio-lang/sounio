<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-14-units-vertical-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-14-units-vertical-design
-->

# Design — Harden the units / dimensional-analysis vertical

**Status:** approved design, pre-implementation
**Date:** 2026-07-14
**Constraint:** No compiler changes (Madaros owned by CODEX-2). Work in `stdlib/`, `examples/`, `tests/`, `scripts/`.
**Orthography:** EN-UK.

## 1. Why

Continues the proven playbook that landed the GUM vertical (PR #860): take a module that lives entirely
inside the current compiler's working surface (stdout + pure compute, no file/heap/argv), make it
**importable → run-proven against first-principles values → gated → math-reviewed → merged**.

`units` is the dimensional-analysis sibling of GUM: a `Quantity` is *value + standard uncertainty +
dimension*, and `quantity_add/sub/mul/div` propagate **both** the GUM uncertainty and the dimensions.
This is dissertation-aligned (doses, concentrations, rates) and high identity value.

## 2. Verified starting state

- `stdlib/units/lib.sio` (260 lines) is green and **imports + runs** as native ELF (verified: 3kg+2kg=5,
  mass↔length mismatch detected without panic, 2kg·9.8m/s²=19.6 with the force dimension confirmed).
- API: `UnitDim` (mass/length/time/temperature/amount/current/luminosity exponents), `Quantity`
  (`value`, `uncertainty`, `dim`); base dims (`dim_mass`…), derived dims (`dim_velocity/acceleration/
  force/energy/power/pressure`); `dim_eq`, `quantity_is_compatible`; `quantity_new/add/sub/mul/div/scale`;
  conversions (`convert_*`). `dim_add_exp`/`dim_sub_exp` combine dimensions in mul/div.
- Uncertainty propagation is GUM-correct: add/sub → √(u_a²+u_b²) (dim-checked via `assert`); mul →
  √((b·u_a)²+(a·u_b)²); div → √((u_a/b)²+(a·u_b/b²)²).
- The 7 satellite files (`astronomical`, `atomic`, `cgs`, `imperial`, `natural`, `pharmacological`,
  `qudt`) fail `check` (old idioms) — **left untouched** (additive; scope is `lib.sio`).
- **Gap:** there is **no way to display a `Quantity` with its unit** — no dimension→symbol renderer. You
  can compute 19.6 N but cannot print "19.6 N". This is the real-world-usability hole this work closes.

## 3. Goal

A program can `use` the units module, do dimension-safe arithmetic with uncertainty, detect dimension
mismatches, and **print a quantity with its unit symbol** — proven by compile-and-run against
first-principles values, gated, and math-reviewed.

## 4. Scope

### In
1. **Verify + document** the import/print idiom in the module header (no code change for importability).
2. **`quantity_show` + dimension renderer** — a print-based function (like `gum_report`) that prints
   `label = value ± u <unit>`, where `<unit>` is a recognised derived symbol (N, J, W, Pa, m/s, m/s²) or,
   failing that, the base-dimension form (`kg`, `m`, `s`, with integer exponents, zero exponents omitted,
   dimensionless → empty). Lives in `lib.sio` (importable surface). Print-based — no string assembly.
3. **Run-proof driver** — asserts (first-principles): add/sub/mul/div/scale values + dimensions +
   uncertainty propagation; mismatch detection via `quantity_is_compatible`; conversions.
4. **Runnable gate** + **math-review** (units carry GUM uncertainty → §10 math-review applies).

### Out
- No fix to the 7 satellite files (additive; separate work).
- No file/stdin/argv I/O (compiler-blocked). Output is stdout only.
- No new physics; expose/harden what exists. Renderer covers SI base + the six derived dims already in
  `lib.sio`; a full unit-symbol algebra is future work.
- No compiler edits.

## 5. Design

### 5.1 Import idiom (item 1)
`units::lib` already imports via `use units::lib::*` and runs. Add a header usage note (wildcard import;
`print`/`println` not `print_f64`; inline logic in importing mains — the three known Madaros multi-module
quirks, `docs/audit/MADAROS_MULTIMODULE_PRINT_IMPORT_BUGS_2026-07-13.md`). No signature changes.

### 5.2 Renderer (item 2)
- `dim_show(d: UnitDim) with IO` — prints the unit symbol. Recognition order: match `d` against the six
  named derived dims (force→`N`, energy→`J`, power→`W`, pressure→`Pa`, velocity→`m/s`, acceleration→`m/s^2`)
  via `dim_eq`; else print base symbols with exponents (`kg`,`m`,`s`,`K`,`mol`,`A`,`cd`), printing
  `sym` for exp 1 and `sym^n` for exp≠1, positives then negatives, omitting zero exponents; dimensionless
  → print nothing.
- `quantity_show(label: string, q: Quantity) with IO, Mut, Div, Panic` — prints
  `label = value ± uncertainty ` then `dim_show(q.dim)` then newline. Print-based (`print`, `println`).

### 5.3 Run-proof (items 3, 4)
Driver `tests/stdlib/units/test_units_stdlib.sio`, all inline in `main`, asserts on raw fields via
`.value`/`.uncertainty` and `dim_eq`:
- add: (3±0.4 kg)+(2±0.3 kg) = 5 kg, u=√(0.16+0.09)=0.5, dim=mass.
- sub: mass − mass dim-preserved; u=0.5.
- mul: (2±0.1 kg)·(9.8±0 m/s²) = 19.6 N, u=9.8·0.1=0.98, dim=force.
- div: (10±0 m)/(2±0 s) = 5 m/s, dim=velocity.
- scale: 2·(3 kg)=6 kg.
- mismatch: `quantity_is_compatible(mass, length)` = false.
- conversions: `convert_kg_to_g(2.0)=2000`, `convert_celsius_to_kelvin(0.0)=273.15`.
Prints a `quantity_show` line for a couple of results, then `UNITS_STDLIB_OK`.

## 6. Module layout
```
stdlib/units/lib.sio                        (modify: header note + dim_show + quantity_show)
tests/stdlib/units/test_units_stdlib.sio    (new: run-proof driver)
examples/units/dimensional_report.sio       (new: consumer example using quantity_show)
scripts/units_gate.sh                        (new: compile+run gate)
```

## 7. Verification
- `souc check stdlib/units/lib.sio` green (unchanged public signatures).
- `souc compile … && ./elf` for the driver + example (compile-and-run, never `check` alone).
- `scripts/units_gate.sh` → `UNITS_GATE_OK`.
- Mandatory `bin/llm-offload -t math-review -p xai` on the uncertainty/dimension arithmetic; logged.

## 8. Success criteria
1. A standalone program `use`s `units::lib`, runs as ELF, does dimension-safe arithmetic with uncertainty,
   and prints quantities with unit symbols.
2. Run-proof asserts first-principles values (dims + uncertainty) and passes.
3. Gate green; math-review PASS logged.
4. No compiler files touched; satellites untouched.

## 9. Risks
| Risk | Mitigation |
|---|---|
| Renderer string logic trips a multi-module quirk | Print-based (no `++` assembly); inline; `print`/`println` only. |
| Derived-unit recognition ambiguous (e.g. torque vs energy share dims) | Document that recognition is by SI dimension only; N·m vs J is a known GUM/SI ambiguity, out of scope. |
| `assert` in add/sub aborts on mismatch | Run-proof tests mismatch via `quantity_is_compatible` (bool), never by triggering the assert. |
