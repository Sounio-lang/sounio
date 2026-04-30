<!-- docs:meta
topic_id: repo.docs.guide.egc-depth-guide
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.egc-depth-guide
-->

# EGC Depth Levels D1–D6

**Epistemic Gradual Compilation** — how Sounio tracks what the compiler knows about every expression, and how certain that knowledge is.

All confidence values are integers in `[0, 1000]`.  
`GATE_THRESHOLD = 950`. `PLATINUM = 1000`.

---

## The Core Idea

Every expression token gets a confidence score stored in `EXPR_CONF[p]`.  
The compiler's epistemic inference pass (`epistemic_infer_pass()`) propagates these scores.  
At self-compile, every token reaches `conf=1000` — the compiler knows its own source completely.

```
BRONZE  [0,   699]   unreliable
SILVER  [700, 949]   uncertain
GOLD    [950, 999]   practically certain (above gate)
PLATINUM [1000]      fully verified
```

---

## D1 — Knightian Uncertainty Intervals

**What:** Every expression has a `[BELIEF, PLAUS]` interval around its confidence.  
The gap `PLAUS - BELIEF` is the Dempster-Shafer "unknown unknowns" width.

**Where:** `EXPR_BELIEF[524288]`, `EXPR_PLAUS[524288]` — declared near line 274.  
Phase B population: `measured()` sites get belief=985, plaus=995 (gap=10).  
`asserted()` sites get belief=960, plaus=980 (gap=20). Certain tokens: gap=0.

**Output:** After compilation, `knightian: N sites, gap_sum=G` is printed.

**Invariant (machine-checked):** `BELIEF ≤ CONF ≤ PLAUS` for all tokens.  
See `formal/lean4/SounioMeasConf.lean` → `belief_le_conf_le_plaus`.

---

## D2 — Parameterized `EpistemicComplete` Effect

**What:** `with Epistemic(N)` enforces that every expression in the function body  
has `EXPR_CONF ≥ N`. Default `with Epistemic` uses `N = 950`.

**Usage:**
```sio
fn precise(x: f64) -> f64 with Epistemic(970), Div {
    return x / 2.0   // compiler checks: all tokens here must have conf ≥ 970
}
```

**Where:** Parser at ~line 14197 reads the `(N)` argument into `FN_EPISTEMIC_MIN[fi]`.  
Body scan at ~line 12673 uses `FN_EPISTEMIC_MIN[fi]` instead of the hard-coded threshold.

**Note:** Four locations in the compiler skip `with EffectName, ...` clauses during  
parsing; all four were updated to handle the `(N)` form so it doesn't leak into codegen.

---

## D3 — `Knowledge<T>` as First-Class ETY Kind

**What:** When a function returns a `Knowledge<T>` type, the expression's ETY entry  
gets `kind=7` (new in Gen 17; kinds 0–6 were already taken).  
Epistemic subtyping: `Knowledge<T, c=1000>` is a subtype of `Knowledge<T, c=950>`.

**Where:**  
- `ety_mk_knowledge(inner_kind, conf)` — new function after `ety_mk_struct` (~line 11435)  
- Phase B tags `measured()`, `asserted()`, `constant()` call sites as kind=7  
- `ety_unify` has a kind=7 case: takes the minimum confidence  
- `ety_knowledge_subtype_ok()` checks `src.conf ≥ dst.conf`

**Output:** `knowledge_subtype: N sites, V violations` printed after compilation.

**Tracked:** `EPIST_KNOW_SUBTYPE_SITES`, `EPIST_KNOW_SUBTYPE_VIOLATIONS`.

---

## D4 — R15 Bayesian Runtime Update

**What:** Running with `--r15-monitor`, programs can call `update_conf(gate_id, new_conf)`  
and `read_conf(gate_id)` to write/read the R15 side table at runtime.  
When confidence drops below 950, the next call through that gate fires `UD2`.

**Usage:**
```sio
// compile with: gen17.elf prog.sio out.elf --r15-monitor
let c = read_conf(0)       // → 1000 at startup
update_conf(0, 900)        // degrade gate 0 below threshold
```

**How it works:** The compiler recognizes `update_conf` / `read_conf` by name hash  
(11213716 / 9144208) and emits inline x86:
```
update_conf: mov [r15 + rdi*8], rsi   ; 0x49 0x89 0x34 0xFF
read_conf:   mov rax, [r15 + rdi*8]   ; 0x49 0x8B 0x04 0xFF
```
Only active when `CLI_R15_MONITOR != 0`.

**Example:** `examples/science/epistemic_adaptive.sio`

---

## D5 — Temporal Confidence Decay

**What:** Measurements get stale. An `EpistemicTimed` value carries a `birth_tick`.  
`conf_at_tick(e, current_tick, half_life)` computes decayed confidence via binary halving:

```
conf(age) = 990 >> (age / half_life)
```

At multiples of `half_life`: `990 → 495 → 247 → 123 → ...`  
Below 950: stale. Below 500: unreliable.

**This is a pure example** — no compiler changes. The decay logic lives entirely  
in `examples/science/epistemic_decay.sio`.

**Integration with D4:** `update_conf(gate_id, conf_at_tick(sensor, tick(), 100))`  
drives the R15 side table from sensor age, causing gates to trap when data is stale.

---

## D6 — Curry-Howard Bridge / Graded Modal Logic

**What:** `--emit-proof-obligations` generates a Lake-buildable Lean 4 module  
where every PLATINUM token in the source produces an `example` goal.  
The CI job (`lean-proofs`) verifies these via `lake build`.

**The key theorem** (`formal/lean4/SounioGradedModal.lean`):
```lean
theorem egc_graded_soundness :
    (∀ c, gradedSubtype c c) ∧                    -- reflexive
    (∀ c1 c2 c3, ...) ∧                           -- transitive
    (∀ c, c ≤ 1000 → gradedSubtype PLATINUM c) ∧  -- PLATINUM is top
    (∀ c1 c2 c1' c2', ... → gradedSubtype          -- conf_product monotone
      (conf_product c1 c2) (conf_product c1' c2'))
```

**Patient Zero — modal form:** If the compiler achieves `□_1000` (PLATINUM) on all  
its own tokens, it is epistemically complete. The fixed-point  
`sha256(gen17.elf) = sha256(gen17b.elf)` witnesses this.

**Usage:**
```bash
gen17.elf src.sio out.elf --emit-proof-obligations \
  > formal/lean4/SounioProofObligation.lean
make proof-check   # runs lake build on the generated module
```

**CLI flag:** `CLI_EMIT_PROOFS` (~line 336). Output format: valid Lean 4 with  
`import SounioGradedModal` header so it drops directly into the Lake package.

---

## Key Files at a Glance

| Feature | Code location | Example |
|---------|--------------|---------|
| EXPR_CONF/BELIEF/PLAUS arrays | `lean_single.sio` ~line 264–280 | — |
| `FN_EPISTEMIC_MIN` | `lean_single.sio` ~line 328 | — |
| Phase B epistemic inference | `lean_single.sio` ~line 12240 | — |
| D2 effect parser | `lean_single.sio` ~line 14197 | — |
| D3 `ety_mk_knowledge` | `lean_single.sio` ~line 11435 | — |
| D4 R15 builtins | `lean_single.sio` ~line 8246 | `epistemic_adaptive.sio` |
| D5 decay | — | `epistemic_decay.sio` |
| D6 Lean proofs | `formal/lean4/SounioGradedModal.lean` | `make proof-check` |

---

## Gen 19 — Scientific Computing Readiness

Three features added for honest, reproducible science:

### Unit System (multi-standard)

Seven-dimension packed encoding: `bits[3:0]=mass, [7:4]=length, [11:8]=time, [15:12]=temp, [19:16]=amount, [23:20]=current, [27:24]=lum`.  
4-bit 2's-complement per field; `0` = dimensionless.

**Preloaded unit systems:** SI (40+ units), CGS (erg, dyn, Gauss), Imperial (lb, ft, psi, BTU), Astronomical (AU, pc, ly, Msun), Pharmacological (mmHg, mEq, IU, mcg), Atomic (eV, MeV, GeV, u, Bohr).

**Syntax:**
```sio
unit mg;                      // declare or re-declare a unit name
unit furlong = 201.168 * m;   // user-defined with scale from base unit
let dose: mg = 500.0          // type annotation → dimension tracked
let speed = 9.8_m / 1.0_s    // _suffix literals; result has dim length/time
let bad = 500_mg + 2.0_m      // E: unit dimension mismatch (mass ≠ length)
```

**Dimension algebra:**
- `a + b`, `a - b` — require same dimension vector (or either dimensionless)
- `a * b` — result dim = dim_add(a_dim, b_dim)
- `a / b` — result dim = dim_sub(a_dim, b_dim)

**Code locations:**  
`UNIT_HASH_TABLE/UNIT_DIM_TABLE` globals, `init_unit_table()`, `dim_pack/dim_add/dim_sub`, `unit_register_h`, `unit_lookup_dim` — all near `gl_name_hash`.  
`_suffix` lexing — numeric literal block in `tokenize()`.  
Dimension propagation — `compile_multiplicative()`, `compile_additive()`.  
`unit NAME;` parsing — Pass 0a in `compile_all()`.

**Stdlib:** `stdlib/units/imperial.sio`, `cgs.sio`, `astronomical.sio`, `pharmacological.sio`, `atomic.sio`, `natural.sio`  
**Examples:** `examples/units/basics.sio`, `examples/units/pharma_dose.sio`

---

### I/O: `read_line()`

```sio
fn main() -> i32 with IO {
    let line = read_line()   // reads one line from stdin (fd=0)
    println(line)
    return 0
}
```

Emits `mmap(NULL, 4096, RW, ANON)` + `sys_read(0, buf, 4095)`, strips trailing `\n`, null-terminates.  
Requires `with IO` effect — same as `read_file`.  
**Code location:** `emit_read_line_x86()` near `emit_read_file()`.

---

### `Deterministic` Effect

A function declared `with Deterministic` is statically guaranteed to never perform I/O.  
Attempting to combine `Deterministic` and `IO` is a compile error (E080).

```sio
fn pure_compute(x: f64) -> f64 with Deterministic {
    return x * 2.0   // OK — no I/O
}

fn bad(x: f64) -> f64 with Deterministic, IO {
    println("x")     // E080: Deterministic function declares IO effect
    return x
}
```

Effect bitmask: `Deterministic = 4096` (bit 12).  
**Code location:** effect parsing in Pass 1b (`compile_all()` ~line 14430), enforcement check ~line 14452.

---

## Self-Compile Numbers (Gen 17)

```
tokens:   78,291  (PLATINUM = 100%)
sha256:   da843a52...
binary:   889 KB
knightian: 0 sites on self-compile (all tokens certain)
```

## Self-Compile Numbers (Gen 19)

Gen 19 verified the bootstrap fixed-point after integrating ETY_UNIT_DIM (Knowledge<T>
entries now carry packed 7-field dimensional vectors) and dim_add/dim_sub propagation
through `*` and `/` operators.

```
tokens:   143689
sha256:   f22cccbb2ef303ed9eee6b2c60312c417cd3bf7f27ae6981146ab648b2fff734
binary:   854592 bytes
knightian: 0 sites on self-compile (all tokens certain)
unit_table: 102+ units preloaded across SI/CGS/Imperial/Astro/Pharma/Atomic/Natural
```

Verification:

```bash
./artifacts/self-hosted/souc-self-hosted-x86_64 self-hosted/compiler/lean_single.sio gen19.elf
./gen19.elf self-hosted/compiler/lean_single.sio gen19b.elf
sha256sum gen19.elf gen19b.elf
# f22cccbb2ef303ed9eee6b2c60312c417cd3bf7f27ae6981146ab648b2fff734  gen19.elf
# f22cccbb2ef303ed9eee6b2c60312c417cd3bf7f27ae6981146ab648b2fff734  gen19b.elf
```

The `knowledge_units: N sites` line now appears after every `knowledge_subtype:` line
in the epistemic pass output, tracking how many `Knowledge<T>` bindings carry
dimensional information.

---

## Self-Compile Numbers (Gen 20)

Gen 20 verified the bootstrap fixed-point after integrating Door 2 (GPU kernel
codegen), Door 3 (O-SSM + E-KAN Knowledge<T> examples), and the sedenion 168
verification example. This is the first generation where the self-hosted compiler
reports **100% PLATINUM** epistemic certainty when compiling itself (Gen0→Gen1
shows 98% due to the older stage-0 artifact's analysis pass; Gen1→Gen2 and
Gen2→Gen3 are both 100% PLATINUM).

```
source:    self-hosted/compiler/lean_single.sio
lines:     20,565
tokens:    152,303
sha256:    4d3698974bc3d3fe1d19521e84c18eda4c181e9d366d965540b6ae11b593659f
binary:    908,334 bytes
knightian: 0 sites on self-compile (all expressions certain)
platinum:  100% (Gen1→Gen2, Gen2→Gen3)
warnings:  4 pre-existing (effect not declared, lines 772/778/779/780)
fns:       476  code: 904,238 bytes  patches: 9,189
```

Verification (three-generation chain):

```bash
./artifacts/self-hosted/souc-self-hosted-x86_64 self-hosted/compiler/lean_single.sio /tmp/gen1.elf
/tmp/gen1.elf self-hosted/compiler/lean_single.sio /tmp/gen2.elf
/tmp/gen2.elf self-hosted/compiler/lean_single.sio /tmp/gen3.elf
sha256sum /tmp/gen2.elf /tmp/gen3.elf
# 4d3698974bc3d3fe1d19521e84c18eda4c181e9d366d965540b6ae11b593659f  /tmp/gen2.elf
# 4d3698974bc3d3fe1d19521e84c18eda4c181e9d366d965540b6ae11b593659f  /tmp/gen3.elf
# MATCH — fixed point confirmed
```
