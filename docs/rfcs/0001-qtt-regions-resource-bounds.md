<!-- docs:meta
topic_id: repo.docs.rfcs.0001-qtt-regions-resource-bounds
authority: repo_only
audience: users
last_validated: 2026-10-22
validated_by: Cloud Agent
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.rfcs.0001-qtt-regions-resource-bounds
-->

# RFC-0001: Quantitative Types, Effect-Based Regions, and Resource-Bounded Allocation

**Status**: DRAFT  
**Date**: 2026-10-22  
**Authors**: Demetrios Chiuratto Agourakis  
**Related Issues**: #2966, #3000  
**Related ADRs**: ADR-010 (Madaros Region Reclamation)

---

## Abstract

This RFC proposes a unified memory management system for Sounio that combines:

1. **Quantitative Type Theory (QTT) usage grades** (0/1/ω) to unify move semantics, copy elision, and erasure
2. **Allocation as an algebraic effect** `Alloc<R>` with regions as effect handlers
3. **Resource-bounded types** via amortized resource analysis, turning runtime capacity errors into compile-time guarantees
4. **Lean 4 formalization** of region soundness and bound correctness

The system goes beyond state-of-the-art by:
- Making copy elision a **typing consequence** rather than an optimization
- Preventing region escape through the **effect/type system alone** without lifetime annotations
- Computing **arena sizes at compile time** from declared allocation bounds
- Tracking epistemic **provenance through regions** for scientific uncertainty propagation

**Five moonshot proposals extend this foundation:**

- **(A) Self-certifying bootstrap:** Madaros type-checks its own allocation bounds; each compiler generation emits a machine-checkable memory certificate that the next generation and Lean verify. Fixed point = compiler whose memory behaviour is a **theorem**, not a test.
- **(B) Static memory plans:** Infer complete allocation plans at compile time (zero dynamic allocator) where bounds are data-independent; fall back to `Alloc` effect only for data-dependent patterns.
- **(C) Epistemic memory:** Bounds are themselves epistemic quantities (exact / upper bound / estimated with uncertainty / unknown), tracked like Sounio's `Knowledge<T>`; distinguish proved bounds from measured ones.
- **(D) Heterogeneous regions:** Same `Alloc<R>` effect spans CPU arenas, GPU memory, and distributed/remote memory; data movement is a typed move between regions with cost in the type.
- **(E) Units of memory:** bytes/cells are a dimension in Sounio's units system; bound arithmetic is dimension-checked like physical units.

This design addresses three founder-identified defects: (1) per-phase arena requirement for compiler performance; (2) explicit ownership/move semantics superior to SOTA; (3) elimination of untyped field-index resolution by name hash (the `.span`→51 bug from #3000).

---

## Motivation

### Current State

Sounio's Madaros native runtime (#2737, ADR-010) implements dynamic region reclamation:
- Functions eligible for reclamation open a region (`__rgn_enter`) recording heap cursor H0 and handle count C0
- At return, young results are copied above the region, resources reset to (H0, C0), then copied back down
- Escapes (stores, casts, syscalls, extern calls) mark regions escaped, preventing reclamation
- Conservatism costs only memory: false positives retain regions but never free live objects

**Limitations:**
1. **Runtime overhead:** copy-in/copy-out at every region boundary; barrier checks on every store
2. **Conservative escape:** any scalar that looks like a reference, any unproven cast source, retains regions
3. **No compile-time guarantees:** "handles full" (exit 182) still occurs when no function is eligible
4. **Manual tuning:** arena sizes are fixed (2 GiB heap, 2²² handle table); no principled sizing

### Related Defects

- **#3000 (`.span`→51 bug):** Untyped field-index resolution via name hash (`name[0] % 64`) caused out-of-bounds heap access when structs weren't in the layout table. The compiler emitted index **51** for `.span` (ASCII 115 % 64), accessing garbage memory.
- **#2966 (stage 0 of this proposal):** Requires known types at lowering time; guesses must be compile errors.
- **Per-phase arenas (founder decision 1):** Compiler performance requires separate arenas for parse, check, IR lowering, codegen phases.

### Why Now?

1. **Madaros is stable** enough to reason about (v0.80.0, self-hosted, fixed-point verification)
2. **Dynamic regions work** but hit conservative limits
3. **Scientific uncertainty propagation** (GUM, `Knowledge<T>`) needs provenance tracking through allocations
4. **Formal verification** (Lean 4 infrastructure) can prove safety properties

---

## Design

### Layer 1: Quantitative Type Theory Usage Grades

We adopt a **graded modal type system** à la Atkey (2018), McBride (2016), and Idris 2 QTT (Brady, 2021), distinguishing:

| Grade | Meaning | Sounio Syntax | Compilation |
|-------|---------|---------------|-------------|
| **0** | Erased | implicit inference | Value unavailable at runtime; types, proofs, and compile-time-only data |
| **1** | Used exactly once | `move x`, default for non-`Copy` types | Move semantics; caller loses access after use |
| **ω** | Unrestricted | `copy x`, default for `Copy` types | Duplicable; copy semantics |

**Key insight:** Copy elision is not an optimization — it's a **typing consequence**. A function declared:

```sio
fn consume(x: Vec<i64> @ 1) -> i64 with Panic { x[0] }
```

takes ownership (grade-1). The caller cannot use `x` afterward:

```sio
let v = vec_of(1, 2, 3)
let first = consume(v)
// println(v)  // ERROR: v moved at line above
```

A function declared:

```sio
fn peek(x: &Vec<i64>) -> i64 with Panic { (*x)[0] }
```

borrows (grade-ω on the borrow, grade-1 on the Vec behind it). The caller retains `v`.

**Grade inference rules:**

1. Local bindings `let x = e` infer grade from `e`'s grade
2. Function parameters default to grade-ω unless type is non-`Copy` (then grade-1)
3. Explicit annotation `@ 0`, `@ 1`, `@ ω` overrides
4. Struct fields inherit grade from enclosing struct
5. Exclusive references `&!T` have grade-1 access to the borrowed value

**Relation to existing Sounio:**
- Current `var` and move semantics → grade-1 tracking formalized
- Current `Copy` types (integers, `f64`, `bool`) → grade-ω
- Epistemic types (`Knowledge<T>`, `Contest<T>`) → grade-ω on value, grade-1 on provenance metadata

**Implementation path:**
- Parser: accept `@ 0`, `@ 1`, `@ ω` syntax in type positions
- Checker: track grade in `TypeKey`, thread through `check_expr`, `check_fn_call`
- IR: lower grade-1 as "last use," grade-ω as copy, grade-0 as erasure

---

### Layer 2: Allocation as an Algebraic Effect

We reify allocation as an **algebraic effect** `Alloc<R>` parameterized by a **region** `R`:

```sio
effect Alloc<R> {
    fn allocate(size: usize, align: usize) -> *mut u8
}
```

Functions declare their allocation effects:

```sio
fn make_vec(n: i64) -> Vec<i64> with Alloc<Local>, Panic {
    // allocates in the Local region
    var v = Vec::new()
    for i in 0..n {
        v.push(i)
    }
    return v
}
```

**Region hierarchy:**

```
Global          // Static, never reclaimed (BSS, rodata)
  ├─ Phase<P>   // Compiler per-phase arenas (Parse, Check, Lower, Codegen)
  ├─ Function   // Per-function region (current ADR-010 implementation)
  ├─ Pool<K>    // Pooled allocations (K = size class)
  └─ GPU        // Device memory
```

**Effect handlers** implement region semantics:

```sio
handle Alloc<Function> {
    allocate(size, align) -> {
        let ptr = bump_alloc(current_region.heap_cursor, size, align)
        current_region.heap_cursor += size
        return ptr
    }
}
```

**Key properties:**

1. **Effect typing prevents escape:** A value allocated in region `R` cannot outlive `R`'s handler scope
2. **No lifetime annotations:** The effect system encodes the same information as Rust lifetimes but through effect polymorphism
3. **Flexible:** Handlers can be swapped (instrumentation, GPU, custom allocators) without changing user code

**Relation to existing effects:**
- Current: `IO`, `Mut`, `Div`, `Panic`, `Async`, `GPU`, `Prob`, `Observe`, `Epistemic`
- New: `Alloc<R>` orthogonal to all above
- Composition: `fn f() with IO, Alloc<Function>, Panic { ... }`

**Epistemic integration:**
- `Knowledge<T>` values track provenance: measurement device, calibration, transformation history
- Region `R` becomes part of provenance: "this value was computed in function `f`, region `Function`"
- GUM uncertainty propagation can distinguish "variance from input uncertainty" vs "variance from numerical approximation in this region"

**Implementation path:**
- `self-hosted/check/effects.sio`: add `EffectAlloc` variant with region parameter `R: RegionKey`
- `self-hosted/types/regions.sio`: define `RegionKey`, region hierarchy, escape rules
- IR lowering: `Alloc<R>` effect → calls to region-specific allocator (bump, pool, GPU)

---

### Layer 3: Resource-Bounded Types

We adopt **amortized resource analysis** from RAML (Hoffmann & Hofmann, 2010; Hoffmann et al., 2017) and Granule (Orchard et al., 2019):

Functions declare **allocation bounds** as part of their type:

```sio
fn parse_ast(tokens: &[Token]) -> ASTNode 
    with Alloc<Parse> ≤ 16 * tokens.len() + 128, Panic 
{
    // guaranteed to allocate at most 16 bytes per token + 128 bytes overhead
    ...
}
```

**Bound language:**

```
B ::= c                          // constant
    | n                          // size of input (e.g., tokens.len())
    | B₁ + B₂                    // addition
    | c * B                      // scalar multiplication
    | B₁ * B₂                    // multiplication (limited)
    | max(B₁, B₂)                // maximum
    | if C then B₁ else B₂       // conditional
```

**Inference rules:**

1. **Constant allocation:**  
   `allocate(c, _) ⊢ Alloc ≤ c`

2. **Sequential composition:**  
   `e₁ ⊢ Alloc ≤ B₁` and `e₂ ⊢ Alloc ≤ B₂` implies `e₁; e₂ ⊢ Alloc ≤ B₁ + B₂`

3. **Conditional:**  
   `e₁ ⊢ Alloc ≤ B₁` and `e₂ ⊢ Alloc ≤ B₂` implies `if C then e₁ else e₂ ⊢ Alloc ≤ max(B₁, B₂)`

4. **Loop (amortized):**  
   For `for i in 0..n { body }`, if `body ⊢ Alloc ≤ B` then loop ⊢ `Alloc ≤ n * B`

5. **Recursion (induction):**  
   For recursive calls, require decreasing measure and sum bounds over recursive spine

**Arena sizing:**

Given a module with functions {f₁, ..., fₙ}, compiler computes:

```
Arena<R> size = max(Σ Bᵢ for callstack path P through {f₁, ..., fₙ} 
                    where all fᵢ allocate in region R)
```

**Effect:**
- `arena full` (exit 182) becomes **compile error E300**: "Allocation bound exceeded: function `parse_module` requires 2.1 GiB but `Parse` arena is sized at 2.0 GiB"
- Users can: (1) increase arena size in config; (2) refactor to allocate less; (3) split work across regions

**Relation to Sounio epistemic types:**

`Knowledge<T>` carries uncertainty bounds. Resource bounds are **dual**: 
- Epistemic bounds: "this value is 10.0 ± 0.5"
- Resource bounds: "computing this value allocates ≤ 1024 bytes"

Both are **interval arithmetic over program semantics**.

**Implementation path:**
- Parser: accept `with Alloc<R> ≤ B` syntax
- Checker: `BoundExpr` AST for bound language; `infer_alloc_bound` traverses IR, emits symbolic bound
- IR: annotate each `IrInstr::Alloc` with static bound
- Solver: `scripts/bounds/solve_arena_sizes.py` computes max callstack bound per region (SMT solver backend: Z3 or CVC5)
- Codegen: emit bound-check assertion in debug builds; arena overflow = panic

---

### Layer 4: Lean 4 Formalization

We prove three theorems in Lean 4 (following the existing `formal/` directory structure):

#### Theorem 1: Region Soundness (No Escape)

**Statement (informal):**  
If a value `v` is allocated in region `R` with handler scope `S`, and `v` is accessible at program point `p`, then `p` is within `S`.

**Formalization sketch:**

```lean
-- Region calculus (based on Tofte-Talpin regions, 1997)
inductive RegionEffect where
  | enter (r : RegionId) : RegionEffect
  | exit (r : RegionId) : RegionEffect
  | alloc (r : RegionId) (ty : Type) : RegionEffect

structure Heap where
  regions : RegionId → Option HeapRegion
  values : ValueId → Option (Value × RegionId)

-- Accessibility: can value v be dereferenced at point p?
def accessible (h : Heap) (v : ValueId) (p : ProgramPoint) : Prop :=
  ∃ r, h.values v = some (_, r) ∧ r ∈ liveRegions h p

-- Soundness: no dangling pointers
theorem region_soundness (prog : Program) (h : Heap) (v : ValueId) (p : ProgramPoint) :
  accessible h v p → 
  ∃ r, h.values v = some (_, r) ∧ 
       ∃ s, handlerScope prog r = s ∧ p ∈ s := by
  sorry  -- to be filled in formal/regions/soundness.lean
```

**Proof strategy:**
1. Operational semantics of region effect system
2. Invariant: heap respects region scoping
3. Preservation: evaluation maintains invariant
4. Progress: accessible values are not dangling

**Related work:** Tofte-Talpin region calculus (1997), Cyclone regions (Grossman et al., 2002), Verona ownership (Swasey et al., 2020)

---

#### Theorem 2: Bound Correctness

**Statement (informal):**  
If function `f` declares `Alloc<R> ≤ B`, then any execution of `f` allocates at most `B` bytes in region `R`.

**Formalization sketch:**

```lean
-- Allocation trace: sequence of allocations
def AllocTrace := List (RegionId × Nat)

-- Total bytes allocated in region R
def totalAlloc (trace : AllocTrace) (r : RegionId) : Nat :=
  (trace.filter (·.1 = r)).foldl (· + ·.2) 0

-- Bound expression semantics
def evalBound (b : BoundExpr) (env : Environment) : Nat :=
  match b with
  | .const c => c
  | .var n => env.get n
  | .add b₁ b₂ => evalBound b₁ env + evalBound b₂ env
  | .mul c b => c * evalBound b env
  | ...

-- Correctness: actual allocation respects declared bound
theorem bound_correctness (f : Function) (r : RegionId) (b : BoundExpr) (env : Environment) :
  f.allocBound r = some b →
  ∀ trace, execFunction f env = (_, trace) →
  totalAlloc trace r ≤ evalBound b env := by
  sorry  -- to be filled in formal/bounds/correctness.lean
```

**Proof strategy:**
1. Compositional: bound correctness of sub-expressions implies correctness of compound expressions
2. Induction on program structure (following RAML technique)
3. Loop bounds require decreasing measure / well-founded recursion

**Related work:** RAML (Hoffmann & Hofmann, 2010), Granule (Orchard et al., 2019), resource separation logic (Atkey, 2010)

---

#### Theorem 3: Grade Preservation

**Statement (informal):**  
If expression `e` has grade `g`, then evaluation of `e` consumes resources according to `g` (grade-1 = move, grade-ω = copy, grade-0 = erasure).

**Formalization sketch:**

```lean
-- Usage grade
inductive Grade where
  | zero : Grade   -- erased
  | one : Grade    -- linear
  | omega : Grade  -- unrestricted

-- Typing judgment with grades
inductive HasType : Context → Expr → Type → Grade → Prop where
  | var : (Γ x = some (t, g)) → HasType Γ (Expr.var x) t g
  | app : HasType Γ e₁ (t₁ →[g] t₂) g' → 
          HasType Γ e₂ t₁ g → 
          HasType Γ (Expr.app e₁ e₂) t₂ (g' * g)  -- grade multiplication
  | ...

-- Grade semantics
def gradeSemantics (g : Grade) : UsagePolicy :=
  match g with
  | .zero => .erased
  | .one => .moved
  | .omega => .copied

-- Preservation: evaluation respects grades
theorem grade_preservation (Γ : Context) (e : Expr) (t : Type) (g : Grade) :
  HasType Γ e t g →
  ∀ v, eval e Γ = some v →
  usagePolicy v = gradeSemantics g := by
  sorry  -- to be filled in formal/qtt/preservation.lean
```

**Related work:** QTT (Atkey, 2018; McBride, 2016), Idris 2 QTT (Brady, 2021), Linear Haskell (Bernardy et al., 2018)

---

### Epistemic Integration

Region tracking integrates with `Knowledge<T>` provenance:

```sio
struct Provenance {
    source: ProvenanceSource,      // Measurement, Computation, ...
    region: RegionId,              // where this value was allocated
    transforms: Vec<Transform>,    // GUM transformations applied
}

fn propagate_uncertainty(x: Knowledge<f64>, y: Knowledge<f64>) 
    -> Knowledge<f64> with Epistemic, Alloc<Function> ≤ 256 
{
    // GUM uncertainty propagation
    let cov = covariance(x, y)
    let result = combine_with_covariance(x, y, cov)
    
    // Provenance: result inherits region from current scope
    result.provenance.region = current_region()
    return result
}
```

**Rationale:**  
Scientific reproducibility requires knowing **where** a value was computed, not just **how**. Region provenance enables:
- "This variance comes from uncertainty in Region=`parse_input`, not numerical error in Region=`solve_ode`"
- Debugging: "Memory corruption in Region=`Function` at line 42"
- Audit: "This clinical PK value was computed in a verified arena with bound ≤ 1024 bytes"

---

## Related Work

### Ownership and Move Semantics

| System | Move | Borrow | Copy Elision | Region Escape | Formal Model |
|--------|------|--------|--------------|---------------|--------------|
| **Rust** | Explicit `move` | Lifetimes `'a` | NRVO optimization | Borrow checker + lifetimes | RustBelt (Jung et al., 2018) |
| **Hylo** (Val) | Mutable value semantics | `inout`, `set`, `sink` | Law of exclusivity | Effect system | Informal (Denisov & Karpinski, 2022) |
| **Mojo** | Owned `^` | Borrowed `&`, `inout` | ASAP destruction | Lifetimes (Rust-inspired) | None |
| **Verona** | Ownership regions `iso`, `mut`, `imm` | Region types | Lifetime elision | Region subtyping | Operational semantics (Swasey et al., 2020) |
| **Vale** | Inline refs, `^` move | `&` borrow, constraint refs | Generational references | Region checker + constraints | HGM type system (Vale, 2022) |
| **Cyclone** | Default by-value | Pointers `*`, regions `ρ` | None | Region annotations | Type soundness (Grossman et al., 2002) |
| **Linear Haskell** | Linear types `-⚬` | Unrestricted `→` | GHC optimization | Multiplicity polymorphism | Formal semantics (Bernardy et al., 2018) |
| **Granule** | Linear, graded types | `!` modality | Graded comonads | Effect grading | Denotational semantics (Orchard et al., 2019) |
| **Idris 2 QTT** | Erased 0, linear 1, unrestricted ω | Quantity polymorphism | Erasure at compile time | Quantity tracking | QTT (McBride, 2016; Brady, 2021) |
| **This proposal** | QTT grades 0/1/ω | `&`, `&!` with grades | Typing consequence | Effect-based regions | Lean 4 proofs |

**Key differences:**
- **Rust:** Lifetimes are explicit type annotations. We use **effect handlers** to encode the same constraint implicitly.
- **Hylo/Val:** Mutable value semantics = aggressive copy elision. We make it **typed** via grade-1, not an optimization.
- **Verona:** Region types `iso`/`mut`/`imm` are part of the type. We use **effect parameters** `Alloc<R>`, decoupling allocation from value types.
- **Cyclone, Vale:** Regions are explicit annotations. We infer them from **effect handlers**.
- **Linear Haskell, Granule, Idris 2:** Close to our approach, but no allocation effects or resource bounds.

---

### Effect Systems

| System | Effects | Handlers | Regions | Formal Model |
|--------|---------|----------|---------|--------------|
| **Koka** | `<div,exn>` | `handler`, `mask` | Scope-based | Type-and-effect calculus (Leijen, 2017) |
| **Effekt** | `{IO, State, Exn}` | `try`, `handle` | Dynamic binding | Effect-handler calculus (Brachthäuser et al., 2020) |
| **Frank** | Ambient effects | Pattern matching handlers | No explicit regions | Row polymorphism (Lindley et al., 2017) |
| **Eff (OCaml)** | `effect E` | `match e with effect ...` | No regions | Operational semantics (Bauer & Pretnar, 2015) |
| **This proposal** | `Alloc<R>`, `IO`, `Mut`, ... | Region handlers | First-class regions `R` | Lean 4 proofs |

**Key difference:**  
We treat **allocation as an effect**, not a primitive. This enables:
- Custom allocators as handlers
- Effect polymorphism over regions (`Alloc<R>` where `R` is a type parameter)
- Formal reasoning about memory safety through effect typing

---

### Resource Analysis

| System | Bound Language | Inference | Verification | Application |
|--------|----------------|-----------|--------------|-------------|
| **RAML** | Polynomial (aᵢ·nⁱ) | LP solver | Manual annotations | Heap bounds (Hoffmann et al., 2017) |
| **Granule** | Graded types `A [n]` | Constraint solving | SMT solver | Session types, file I/O (Orchard et al., 2019) |
| **Liquid Haskell** | Refinement types `{v:T | P(v)}` | Abstract interpretation | SMT (Z3) | Functional correctness (Vazou et al., 2014) |
| **Cogent** | Linear types | Manual | Isabelle/HOL | File systems (O'Connor et al., 2016) |
| **This proposal** | Affine `c·n + k` | Symbolic + SMT | Lean 4 + SMT | Compiler arenas, scientific code |

**Key difference:**  
RAML focuses on asymptotic bounds (polynomial time/heap). We focus on **concrete byte counts** for fixed-size arenas, enabling **compile-time arena sizing**.

---

### Systems Languages with Formal Verification

| Language | Verification | Prover | Scope | Notable Results |
|----------|--------------|--------|-------|-----------------|
| **Rust** | RustBelt | Coq | Core language | Type soundness, concurrency (Jung et al., 2018) |
| **Dafny** | Built-in | Z3 | Full verification | Correctness proofs (Leino, 2010) |
| **Fstar** | Refinement types | Z3, SMT | Cryptography, systems | Verified TLS (Bhargavan et al., 2017) |
| **Cogent** | Linear types | Isabelle/HOL | File systems | File system correctness (O'Connor et al., 2016) |
| **ATS** | Dependent types | Proofs in language | Systems programming | Memory safety (Xi, 2016) |
| **This proposal** | QTT + effect regions | Lean 4 | Memory management | Region soundness, bound correctness |

**Key difference:**  
Most verified systems languages prove **type soundness** or **functional correctness**. We prove **resource bounds** (allocation) and **spatial safety** (region escape), which are orthogonal and complementary.

---

## What Is Novel?

This section honestly distinguishes **genuine novelty** (no known precedent) from **new combinations** of known techniques and **applications** of existing ideas to a new domain.

---

### Core System: New Combinations of Known Techniques

The core proposal (Stages 0-4) combines existing techniques in a new way. **Each component exists separately**; the synthesis is novel:

| Component | Prior Art (Established) | How We Combine It | Novel Aspect of Combination |
|-----------|-------------------------|-------------------|----------------------------|
| **QTT grades 0/1/ω** | Idris 2 (Brady 2021), Linear Haskell (Bernardy et al. 2018), Granule (Orchard et al. 2019) | Applied to systems language with C-like performance, mutation (`&!`), algebraic effects, and epistemic types | **Not individually novel.** Novel: QTT in systems language + effects + epistemic types. No prior systems language has all three. |
| **Allocation as effect** | Koka scoped effects (Leijen 2017), Effekt region effects (Brachthäuser et al. 2020) | Regions as **effect parameters** `Alloc<R>`, not scoped resumptions | **Novel combination:** Effect parameters (not scoped effects) + region typing. Koka's `mask<h>` is similar but not for memory. |
| **Resource bounds (RAML)** | RAML polynomial bounds (Hoffmann 2010-2017), Granule graded types (Orchard et al. 2019) | Affine bounds (not polynomial) for **concrete byte counts**, integrated with effect system | **Not novel technique.** Novel: concrete arena sizing (not asymptotic) + effect-integrated. RAML targets algorithmic complexity, not memory budgets. |
| **Region soundness proof** | Cyclone regions (Grossman et al. 2002), Verona regions (Swasey et al. 2020), RustBelt (Jung et al. 2018) | Proven in **Lean 4** (not Coq/Isabelle); integrated with Sounio's existing formal verification | **Not novel theorem.** Novel: Lean 4 proof (vs Coq), integrated with epistemic/GUM proofs in same framework. |
| **Epistemic provenance** | Sounio's own `Knowledge<T>` (current), information flow types (Sabelfeld & Myers 2003) | Regions as provenance markers for scientific uncertainty | **Novel application:** Provenance through memory regions for scientific computing. Closest: info flow types for security, not uncertainty. |

**Honest assessment:**  
- **No individual component is novel.** QTT, effect systems, resource bounds, and region proofs all exist in literature.  
- **The synthesis is new:** Systems language + QTT + effect-based regions + concrete resource bounds + epistemic types. No prior work combines all five.  
- **Application is new:** Memory management for **scientific computing** with uncertainty tracking. Existing systems target performance (Rust), safety (Cyclone), or functional correctness (RAML), not epistemic provenance.

---

### Core System: Genuinely Novel Claims (Verified Carefully)

We claim **four genuinely novel contributions** in the core system (Stages 0-4). For each, we state the claim, closest prior work, and why ours is different:

---

#### Novel Claim 1: Copy Elision as Typing Consequence

**Claim:** Grade-1 parameters **mandate** move semantics at the type level; copy elision is not an optimization but a **typing rule**. A function that takes `x @ 1` cannot be called twice with the same `x`.

**Closest prior work:**
- **Hylo/Val "law of exclusivity"** (Denisov & Karpinski 2022): Mutable value semantics ensure unique access. But: informal, no graded types, no formal proof.
- **Rust ownership:** Move semantics via borrow checker. But: not graded; move is default for non-`Copy`, not a grade.

**Why ours is different:**
- **Graded:** Explicit `@ 1` annotation; can mix `@ 1` (move) and `@ ω` (copy) in same struct.
- **Formal:** Lean 4 proof that grade-1 implies unique use (Theorem 3, §Layer 4).
- **Erasure:** Grade-0 exists; Hylo/Rust have no erased types.

**Verdict:** **Genuinely novel** — QTT grades for move semantics in a systems language with formal proof. Hylo is close but informal and ungraded.

---

#### Novel Claim 2: Effect-Based Region Escape Without Lifetimes

**Claim:** Region `R` is an **effect parameter** `Alloc<R>`; handler scope = region lifetime. No `'a` lifetime annotations; escape prevention through effect typing alone.

**Closest prior work:**
- **Koka `mask<h>`** (Leijen 2017): Scoped effects prevent escape. But: `mask` is for general effects (state, exceptions), not memory regions.
- **Cyclone regions** (Grossman et al. 2002): Explicit region annotations `ρ`. But: regions are types, not effects.
- **Rust lifetimes:** `'a` in types. But: lifetimes ≠ effects; no handlers.

**Why ours is different:**
- **Effect parameters:** `Alloc<R>` where `R` is a type-level region, not a lexical scope or lifetime variable.
- **Handler-based:** Region open/close = install/uninstall handler; no stack discipline required.
- **Composable:** `Alloc<R>` composes with `IO`, `Mut`, etc. via standard effect composition.

**Verdict:** **Genuinely novel** — effect parameters for regions. Koka's `mask<h>` is the closest, but it's not for memory. No prior work uses algebraic effect handlers for region escape prevention.

---

#### Novel Claim 3: Compile-Time Arena Sizing from Bounds

**Claim:** Compiler computes `Arena<R>.size = max(callstack bounds for R)` and emits **compile error** if exceeds capacity. "Arena full" (exit 182) becomes compile error E300.

**Closest prior work:**
- **RAML** (Hoffmann et al. 2017): Infers polynomial heap bounds. But: asymptotic (O(n²)), not concrete bytes; no arena sizing.
- **Real-Time Java** (RTSJ): Scoped memory with size annotations. But: manual, not inferred.
- **Stack analysis** (AbsInt aiT): Static worst-case stack depth. But: stack only, not heap arenas.

**Why ours is different:**
- **Inferred:** Bounds computed from function bodies; no manual annotation (or optional manual).
- **Concrete:** Byte counts, not Big-O. `16 * n + 128` means exactly that.
- **Compile-time failure:** Arena overflow = **compile error**, not runtime check or OS kill.

**Verdict:** **Genuinely novel** — compile-time arena sizing with inferred concrete bounds. RAML is close but targets algorithmic bounds, not arena capacity. RTSJ requires manual sizes.

---

#### Novel Claim 4: Provenance Through Regions for Epistemic Types

**Claim:** `Knowledge<T>` values record which region `R` they were allocated in, distinguishing "uncertainty from input (Region=`parse_input`)" vs "uncertainty from computation (Region=`solve_ode`)".

**Closest prior work:**
- **Information flow types** (Sabelfeld & Myers 2003): Track data provenance for security (taint, confidentiality). But: security, not scientific uncertainty.
- **Verona `iso`/`mut`** (Swasey et al. 2020): Provenance for ownership, not uncertainty.
- **Provenance semirings** (Amsterdamer et al. 2011): Database provenance (why-provenance, how-provenance). But: database queries, not physical measurements.

**Why ours is different:**
- **Scientific uncertainty:** Provenance tracks GUM variance sources, not security labels or database lineage.
- **Region-integrated:** Region `R` is part of provenance, not separate.
- **Epistemic algebra:** Provenance + uncertainty propagate together via GUM rules.

**Verdict:** **Genuinely novel** — region-based provenance for scientific uncertainty. Information flow types are conceptually similar but solve a different problem (security, not metrology).

---

### Moonshot Proposals: Genuinely Novel Claims (Carefully Verified)

The five moonshot proposals (A-E, Stages M1-M5) contain **three claims with no known precedent** and **two that extend known techniques boldly**:

---

#### Moonshot Novel 1: Self-Certifying Bootstrap (Proposal A)

**Claim:** Compiler type-checks its own allocation bounds, emitting a **proof-carrying binary** (memory certificate) that the next generation and Lean verify. Fixed point = compiler whose memory is a **theorem**.

**Closest prior work:**
- **CompCert** (Leroy 2009): Proved compiler correctness (semantics preservation). But: no memory bounds.
- **CakeML bootstrap** (Kumar et al. 2014): Verified compiler that compiles itself. But: no memory certificates; stack bounds only (limited).
- **Proof-carrying code (PCC)** (Necula 1997): Binaries carry safety proofs. But: memory safety, not resource bounds.

**Why ours is different:**
- **Resource bounds:** Certificate contains allocation bounds (bytes), not just safety.
- **Self-certifying:** Compiler Gen n verifies Gen n-1's certificate **and** emits Gen n's certificate.
- **Fixed point:** At convergence, Cₙ = Cₙ₋₁ bit-identical; memory behaviour is a theorem, not a test.

**Verdict:** **GENUINELY NOVEL — NO KNOWN PRECEDENT.** CompCert proves correctness; CakeML verifies bootstrap; PCC carries safety proofs. **No prior work combines self-certifying bootstrap + memory bounds + fixed-point theorem.** This is a moonshot: bold, unproven, high risk.

---

#### Moonshot Novel 2: Static Memory Plans (Proposal B)

**Claim:** Infer complete static allocation plan (zero dynamic allocator) for data-independent code; Tofte-Talpin region inference + RAML bounds + QTT grades, all three together.

**Closest prior work:**
- **MLKit** (Tofte et al. 2004): Region inference, no GC. But: no resource bounds, so arenas can overflow.
- **RAML:** Resource bounds. But: no regions; still uses dynamic allocator.
- **ATS:** Dependent types + manual proofs for static allocation. But: manual, not inferred.

**Why ours is different:**
- **Three-way synthesis:** Tofte-Talpin (regions) + RAML (bounds) + QTT (grades) **together** → static plan.
- **Inferred:** Plan computed automatically; no manual proof (unlike ATS).
- **Fail-closed:** If data-dependent, explicit `Alloc` effect required; no silent fallback.

**Verdict:** **GENUINELY NOVEL — NO KNOWN PRECEDENT.** MLKit has regions, RAML has bounds, ATS has static proofs. **No prior work combines automatic region inference + automatic bound inference + zero-allocator goal.** This is a moonshot: synthesis of three mature techniques, but synthesis is novel.

---

#### Moonshot Novel 3: Epistemic Memory (Proposal C)

**Claim:** Allocation bounds are **epistemic quantities** (`Exact | UpperBound | Estimated μ ± σ | Unknown`), tracked like `Knowledge<f64>`, with GUM-style uncertainty propagation.

**Closest prior work:**
- **Sounio's own `Knowledge<T>`:** Epistemic types for values. But: not for bounds.
- **RAML bounds:** Inferred, but no uncertainty. Either proved or not.
- **Probabilistic resource analysis** (Ngo et al. 2018): Expected cost, not bounded cost. But: probability over execution paths, not measurement uncertainty.

**Why ours is different:**
- **Bounds as epistemic values:** `Estimated (μ=32, σ=4.8, confidence=0.95)` means bound is measured, not proved.
- **Provenance:** Bound carries source (theorem, SMT solver, profiling, conservative estimate).
- **GUM propagation:** Combine bounds like combining measurements: `σ_total = sqrt(σ₁² + σ₂²)`.

**Verdict:** **GENUINELY NOVEL — NO KNOWN PRECEDENT.** Probabilistic bounds (Ngo et al. 2018) are close but different (probability over execution, not measurement uncertainty). **No prior work treats resource bounds as epistemic quantities with measurement uncertainty.** This is a moonshot: conceptually bold, unproven value.

---

#### Moonshot Extension 1: Heterogeneous Regions (Proposal D)

**Claim:** Same `Alloc<R>` spans CPU/GPU/Remote/Persistent; typed transfers `Ptr<T, R₁> → Ptr<T, R₂>` with cost model.

**Closest prior work:**
- **X10 places** (Saraswat et al. 2010): Distributed memory, place types. But: no GPU, no cost model.
- **CUDA/OpenCL:** CPU+GPU. But: manual, untyped.
- **Chapel:** Distributed arrays. But: no GPU, no explicit cost.

**Why ours is different:**
- **Unified:** CPU + GPU + Remote + Persistent in same abstraction.
- **Typed:** `Ptr<T, R>` carries region in type; transfer = typed move.
- **Cost visible:** `transfer<T, R₁, R₂> cost sizeof(T) / bandwidth(R₁, R₂)`.

**Verdict:** **Novel combination, not individually novel.** X10 has places, CUDA has CPU+GPU, cost models exist. **Unifying all three (heterogeneous, typed, costed) is new.** This is a moonshot: bold application, but each piece exists.

---

#### Moonshot Extension 2: Units of Memory (Proposal E)

**Claim:** Bytes, cells, handles are **dimensions** like `kg`, `m`, `s`; bound arithmetic is dimension-checked.

**Closest prior work:**
- **F# Units of Measure** (Kennedy 1997): Dimensional analysis for physical units. But: no memory units.
- **Sounio's own units:** `f64<m>`, `f64<kg>`. But: not for memory.

**Why ours is different:**
- **Memory dimensions:** `byte`, `cell`, `handle` are first-class dimensions.
- **Bound arithmetic:** `16 * n * byte + 128 * byte` is dimension-checked; `byte + second` = compile error.

**Verdict:** **Novel application, not novel technique.** F# and Sounio already do dimensional analysis. **Applying it to memory is new.** This is not a moonshot — it's a straightforward extension of existing Sounio units.

---

### Summary: What Has No Precedent?

**Three genuinely novel claims (no known prior work):**

1. **Self-certifying bootstrap (Moonshot A):** Compiler emits memory certificate; next generation verifies it; fixed point = theorem.
2. **Static memory plans (Moonshot B):** Tofte-Talpin + RAML + QTT → zero allocator (three-way synthesis).
3. **Epistemic memory (Moonshot C):** Bounds as uncertain quantities (`μ ± σ`) with GUM propagation.

**Novel combinations/applications (techniques exist, synthesis/application is new):**

4. **QTT in systems language (Core):** Idris 2 has QTT; Rust has ownership. Combining QTT + effects + epistemic types in systems language is new.
5. **Effect-based regions (Core):** Koka has scoped effects; Cyclone has regions. Effect parameters for regions is new.
6. **Compile-time arena sizing (Core):** RAML has bounds; RTSJ has scoped memory. Concrete byte bounds → arena sizing is new.
7. **Epistemic provenance (Core):** Info flow types exist; `Knowledge<T>` exists. Region-based provenance for uncertainty is new.
8. **Heterogeneous regions (Moonshot D):** X10 has places; CUDA has GPU. Unifying CPU+GPU+Remote+cost is new.
9. **Memory units (Moonshot E):** F# has dimensions. Applying to memory is new (but straightforward).

**Honest risks and open problems are stated for each** (§Moonshot proposals).

---

### Honest Risks and Open Problems

1. **Bound inference completeness:** Can we infer bounds for all programs, or must users annotate? RAML admits automatic inference for polynomial bounds; affine bounds are simpler but less expressive.
   - *Mitigation:* Default to ω-bound (unbounded) with warning; require annotation only for arena-allocated code.

2. **Effect polymorphism complexity:** `Alloc<R>` where `R` is a type parameter adds complexity to effect inference. Koka handles this with scoped effects; we need region unification.
   - *Mitigation:* Restrict `R` to ground regions (no polymorphism) in MVP; add polymorphism in later stage.

3. **Region granularity trade-off:** Per-function regions (ADR-010) are coarse; finer regions (per-loop, per-expression) reduce memory but increase overhead.
   - *Mitigation:* Heuristic selection based on function size, allocation volume, loop depth.

4. **Grade inference vs explicit annotation:** Rust requires explicit lifetimes; Hylo infers them. We default to grade-ω (copy) for `Copy` types, grade-1 for others. Does this surprise users?
   - *Mitigation:* Lint warnings for "suspicious copy" (large struct copied, should be `&` or `move`).

5. **Lean 4 proof effort:** The three theorems (region soundness, bound correctness, grade preservation) are substantial. Can we complete them in reasonable time?
   - *Mitigation:* Start with paper proofs, formalize incrementally; admit axioms if necessary and document them.

---

## Design Principles

1. **Bold but rigorous** (*ousadia com correção*): Propose novel combinations; prove them sound.
2. **Real novelty** (*busca de novelty real*): Not just "Rust + effects" or "Idris + arenas" — genuine synthesis.
3. **Fail closed:** Unknown type at lowering = compile error (not hash fallback) per #2966.
4. **Epistemic integration:** Memory management serves scientific computing, not just performance.

---

## Syntax Examples

### Grade Annotations

```sio
// Explicit grades
fn consume(v: Vec<i64> @ 1) -> i64 with Panic { v[0] }
fn peek(v: &Vec<i64> @ ω) -> i64 with Panic { (*v)[0] }

// Grade inference (default)
fn make_vec(n: i64) -> Vec<i64> with Alloc<Function> {
    var v = Vec::new()  // v: Vec<i64> @ 1
    for i in 0..n {
        v.push(i)
    }
    return v  // move
}

// Erased grade for proofs
fn proof_only(p: Proof @ 0) -> i64 { 42 }  // p unavailable at runtime
```

---

### Allocation Effects with Bounds

```sio
// Declared bound
fn parse_tokens(tokens: &[Token]) -> AST
    with Alloc<Parse> ≤ 16 * tokens.len() + 128, Panic
{
    var nodes = Vec::new()
    for t in tokens {
        let node = parse_token(t)  // ≤ 16 bytes
        nodes.push(node)
    }
    return AST { nodes }
}

// Inferred bound (compiler computes)
fn sum_array(arr: &[i64]) -> i64 with Alloc<Function> {
    var total: i64 = 0
    for x in arr {
        total += x
    }
    return total  // Bound: 0 (no allocation)
}

// Conditional bound
fn process(use_cache: bool, data: &[f64]) -> f64
    with Alloc<Function> ≤ if use_cache { 1024 } else { 0 }
{
    if use_cache {
        let cache = Vec::with_capacity(data.len())
        // ... use cache ...
    } else {
        // ... direct computation ...
    }
}
```

---

### Region Handlers (Compiler-Generated)

```sio
// Per-function region (ADR-010 equivalent, but typed)
fn compute(x: f64) -> f64 with Alloc<Function>, Div {
    handle Alloc<Function> {
        enter() -> { __rgn_enter() }
        exit() -> { __rgn_exit_scalar() }
        allocate(size, align) -> { __rgn_bump_alloc(size, align) }
    }
    
    let y = expensive_intermediate(x)
    return y * 2.0
}  // Region reclaimed here if no escape

// Per-phase region (compiler)
fn parse_module(source: &str) -> Module with Alloc<Parse>, IO {
    handle Alloc<Parse> {
        allocate(size, align) -> { phase_arena_alloc(Phase::Parse, size, align) }
    }
    
    let tokens = lex(source)
    let ast = parse_tokens(tokens)
    return Module { ast }
}  // Parse arena reclaimed after all modules parsed
```

---

### Epistemic Integration

```sio
fn measure_temperature(device: &Sensor) -> Knowledge<f64> 
    with Alloc<Function> ≤ 128, IO, Epistemic 
{
    let raw = device.read()
    let value = calibrate(raw, device.calibration())
    
    return Knowledge::new(
        value: value,
        uncertainty: device.uncertainty(),
        provenance: Provenance {
            source: ProvenanceSource::Measurement,
            region: current_region(),  // records Function region
            device: device.id(),
        }
    )
}

fn propagate_uncertainty(x: Knowledge<f64>, y: Knowledge<f64>) 
    -> Knowledge<f64> with Alloc<Function> ≤ 256, Epistemic 
{
    // GUM variance propagation
    let cov = covariance(x, y)
    let combined = gum_combine(x, y, cov)
    
    // Provenance: merge regions
    combined.provenance.region = Region::merge(
        x.provenance.region, 
        y.provenance.region
    )
    return combined
}
```

---

## Moonshot Proposals

The following five proposals extend the core system into genuinely novel territory. Each is **bold but rigorous**: we state the formal sketch, provide Sounio syntax examples, identify the closest prior work, and honestly assess what is new and what risks remain open.

---

### (A) Self-Certifying Bootstrap: Compiler Memory as a Theorem

**Claim:** Madaros can type-check its own allocation bounds, emitting a **proof-carrying binary** (certificate) that the next compiler generation and a Lean checker verify. At fixed point, the compiler's memory behaviour is a **theorem**, not a test. Arena sizes are **derived** from proofs, never chosen.

---

#### Formal Sketch

**Certificate structure:**

```lean
structure MemoryCertificate where
  -- Source: which compiler generation produced this
  generation : Nat
  source_hash : ByteArray  -- SHA-256 of compiler source
  
  -- Allocation bounds per phase (proved or measured)
  bounds : PhaseId → BoundProof
  
  -- Proofs: each bound is justified
  proofs : ∀ p : PhaseId, ValidBound (bounds p)
  
  -- Execution witness: runtime allocations respect bounds
  witness : ExecutionTrace
  witness_correct : ∀ p, totalAlloc witness p ≤ evalBound (bounds p) witness.env

structure BoundProof where
  expr : BoundExpr              -- Symbolic bound (e.g., 16 * AST.size + 128)
  kind : BoundKind              -- Exact | UpperBound | Estimated ε
  proof : Option (LeanProof)    -- Lean 4 proof if kind = Exact
```

**Bootstrap stages:**

1. **Gen 0 (seed):** lean_single with **measured** bounds (profile-based, kind = `Estimated ε`)
2. **Gen 1:** Madaros compiled by Gen 0, declares bounds, emits certificate C₁
3. **Gen 2:** Madaros compiled by Gen 1, verifies C₁, emits C₂
4. **Fixed point:** Gen n where `verify(Cₙ₋₁) = true ∧ Cₙ = Cₙ₋₁` (bit-identical)

**Verification:**

```lean
def verifyCertificate (cert : MemoryCertificate) (checker : LeanChecker) : Bool :=
  -- (1) Structural: cert well-formed
  cert.bounds.keys = allPhases ∧
  -- (2) Proofs check (if present)
  (∀ p, cert.bounds p |>.kind = Exact → 
    checker.check (cert.proofs p).get! = true) ∧
  -- (3) Witness respects bounds
  cert.witness_correct ∧
  -- (4) Source hash matches
  hash(readSource(cert.generation)) = cert.source_hash
```

**Soundness theorem:**

```lean
theorem self_certifying_soundness 
  (gen_n : Compiler) (cert_n : MemoryCertificate)
  (compiled_by : gen_n.compiled_by gen_{n-1})
  (cert_verified : verifyCertificate cert_{n-1} lean_checker = true) :
  ∀ module : Module, ∀ phase : PhaseId,
    compile gen_n module phase allocates_at_most (evalBound (cert_n.bounds phase)) := by
  sorry
```

---

#### Sounio Syntax Example

**Compiler self-declaration:**

```sio
// self-hosted/compiler/main.sio
fn parse_module(source: &str) -> Module 
    with Alloc<Parse> ≤ proven(8 * source.len() + 1024), IO, Panic 
{
    // Bound is PROVED, not measured:
    // Lean theorem: every token ≤ 8 bytes; module overhead = 1024
    let tokens = lex(source)  // Alloc ≤ 8 * source.len()
    let ast = parse(tokens)    // Alloc ≤ 1024 (fixed AST nodes)
    return Module { ast }
}

fn compile_to_native(module: &Module) -> ELF
    with Alloc<Codegen> ≤ proven(32 * module.ir_size() + 4096), IO
{
    // Bound proved by induction on IR instructions
    let asm = lower_to_asm(module)
    let elf = emit_elf(asm)
    
    // Emit memory certificate
    emit_certificate(MemoryCertificate {
        generation: COMPILER_GEN,
        source_hash: hash(read_file("self-hosted/compiler/main.sio")),
        bounds: phase_bounds(),
        proofs: lean_proofs(),
        witness: current_execution_trace(),
    })
    
    return elf
}
```

**Certificate verification (bootstrapping):**

```sio
// scripts/bootstrap/verify_certificate.sio
fn verify_generation(n: i64) -> bool with IO, Panic {
    let cert_prev = read_certificate(n - 1)
    let cert_curr = read_certificate(n)
    
    // (1) Lean checker verifies proofs
    if !lean_check_all(cert_prev.proofs) {
        println("FAIL: Gen {} proofs do not check", n - 1)
        return false
    }
    
    // (2) Execution witness respects bounds
    if !verify_witness(cert_prev) {
        println("FAIL: Gen {} witness violates bounds", n - 1)
        return false
    }
    
    // (3) Fixed point: certificates are identical
    if n >= 3 && cert_curr == cert_prev {
        println("FIXED POINT: Gen {} = Gen {}", n, n - 1)
        return true
    }
    
    return true
}
```

---

#### Closest Prior Work

| System | Self-Certification | Memory Proofs | Fixed Point |
|--------|-------------------|---------------|-------------|
| **CompCert** (Leroy 2009) | Proved compiler correctness (semantics preservation) | No memory bounds | N/A |
| **CakeML** (Kumar et al. 2014) | Verified compiler to machine code | Stack bounds (limited) | Bootstrap verified, not memory-certified |
| **Cogent** (O'Connor et al. 2016) | Verified file system code | Linear types, no arena bounds | N/A |
| **seL4** (Klein et al. 2009) | Verified microkernel | Memory safety, no allocation bounds | N/A |
| **Coq** (self-hosting) | Proved termination of extraction | No memory bounds | N/A |
| **This proposal** | **Self-certifying bootstrap with memory certificates** | **Per-phase arena bounds proved in Lean** | **Fixed point = memory theorem** |

**Key difference:**  
- CompCert proves **semantic preservation** (compiled code behaves like source).  
- We prove **resource bounds** (compiled code allocates ≤ B bytes).  
- **Genuinely novel:** Self-certifying bootstrap where each compiler generation verifies the previous one's memory certificate, converging to a fixed point where memory behaviour is a theorem.

---

#### What Is Genuinely New

**No precedent (to our knowledge):**
1. **Compiler that proves its own memory bounds:** CompCert/CakeML prove correctness, not resource usage.
2. **Fixed-point memory theorem:** At convergence, the compiler's allocation is a Lean 4 theorem, not a profiling result.
3. **Arena sizes as theorems:** Phase arena sizes are **derived** from proofs (`proven(8 * n + 1024)`), not tuned by benchmarking.

**Closest related:**
- **Proof-carrying code (PCC)** (Necula 1997): Binaries carry safety proofs. We extend to **allocation bounds**.
- **CakeML bootstrap**: Verified compiler that compiles itself. We add **memory certificates** at each generation.

---

#### Risks and Open Problems

1. **Proof burden:** Can we prove bounds for the entire compiler, or only core phases?  
   *Mitigation:* Start with Parse/Codegen (simple); Check/Lower may need measured bounds + safety margins.

2. **Data-dependent bounds:** If bound depends on unpredictable input (e.g., user code size), proof may only give upper bound `≤ f(max_input)`.  
   *Mitigation:* Certificate distinguishes `Exact` vs `UpperBound` vs `Estimated ε`.

3. **Fixed-point convergence:** What if certificates oscillate or grow unbounded?  
   *Mitigation:* Monitor certificate size; require monotonic bound tightening; fail if Gen n+1 exceeds Gen n.

4. **Lean proof automation:** Manual proofs for every function = infeasible. Can we automate bound inference + proof generation?  
   *Mitigation:* SMT-based bound solver (Z3/CVC5) generates candidate bounds; Lean checker verifies them.

---

### (B) Static Memory Plans: Zero-Dynamic-Allocator Compilation

**Claim:** With grades + regions + bounds, the compiler can infer a **complete static allocation plan** for data-independent code, eliminating the dynamic allocator entirely. Fall back to `Alloc<R>` effect only where bounds are data-dependent.

---

#### Formal Sketch

**Static memory plan:**

```lean
structure StaticPlan where
  -- Compile-time allocation schedule
  allocations : List (Offset × Size × Alignment)
  
  -- Total arena size (max offset + size)
  arena_size : Nat
  arena_size_correct : arena_size = (allocations.map (λ (o, s, _) => o + s)).maximum
  
  -- No dynamic calls
  no_dynamic_alloc : ∀ instr : Instruction, 
    instr ∉ allocations.instrs → instr ≠ Alloc

def compileWithStaticPlan (f : Function) : Option StaticPlan :=
  if isDataIndependent f.bound then
    some (inferStaticPlan f)
  else
    none  -- Fall back to Alloc<R> effect
```

**Data-independence:**

A bound `B` is **data-independent** if it depends only on:
- Compile-time constants (`let N = 100`)
- Type sizes (`sizeof(T)`)
- Statically known array lengths (`[i64; 8]`)

Not data-independent:
- Input size (`tokens.len()`)
- Loop iteration count (runtime `for i in 0..n`)
- Conditional allocation (`if flag { alloc() }`)

**Tofte-Talpin region inference + RAML:**

```lean
-- Tofte-Talpin: infer region lifetimes
def inferRegionLifetimes (f : Function) : RegionAnnotatedAST :=
  -- Annotate every allocation site with its region
  -- Regions are lexically scoped; lifetimes = scope
  ...

-- RAML: infer polynomial resource bounds
def inferResourceBound (f : Function) : BoundExpr :=
  -- Potential method: assign potential to data structures
  -- Bound = initial potential - final potential
  ...

-- Combine: static plan if bound is constant
def inferStaticPlan (f : Function) : StaticPlan :=
  let regions = inferRegionLifetimes f
  let bound = inferResourceBound f
  if isConstant bound then
    layoutStaticArena regions bound
  else
    -- Fall back to dynamic
    none
```

---

#### Sounio Syntax Example

**Data-independent (static plan):**

```sio
fn process_fixed_data(data: &[f64; 100]) -> [f64; 100] 
    with Alloc<Static> ≤ 800  // 100 f64s = 800 bytes; CONSTANT
{
    var result: [f64; 100] = [0.0; 100]
    for i in 0..100 {
        result[i] = data[i] * 2.0
    }
    return result
}
// Compiled plan:
//   Offset 0:   result (800 bytes)
//   Total:      800 bytes
//   Allocator:  NONE (stack-allocated at compile time)
```

**Data-dependent (dynamic allocator):**

```sio
fn process_var_data(data: &[f64]) -> Vec<f64>
    with Alloc<Function> ≤ 8 * data.len() + 16  // RUNTIME-DEPENDENT
{
    var result = Vec::with_capacity(data.len())  // Dynamic alloc
    for x in data {
        result.push(x * 2.0)
    }
    return result
}
// Compiled plan:
//   Allocator:  Alloc<Function> handler (bump allocator)
//   Bound:      8 * n + 16 (checked at runtime if n > prealloc)
```

**Mixed (static + dynamic):**

```sio
fn parse_bounded_tokens(source: &str) -> AST
    with Alloc<Parse> ≤ 8 * source.len() + 1024
{
    // Static part: AST node pool (1024 bytes, preallocated)
    let mut node_pool: [ASTNode; 32] with Alloc<Static> = [ASTNode::Empty; 32]
    
    // Dynamic part: Token buffer (8 * source.len(), runtime-dependent)
    let tokens = lex(source)  // Alloc<Parse>
    
    let ast = parse_into_pool(tokens, &mut node_pool)
    return ast
}
// Compiled plan:
//   Offset 0:   node_pool (1024 bytes, stack)
//   Dynamic:    tokens (8 * n bytes, Alloc<Parse> handler)
```

---

#### Closest Prior Work

| System | Static Allocation | Region Inference | Resource Bounds | Zero-Allocator Goal |
|--------|-------------------|------------------|-----------------|---------------------|
| **Tofte-Talpin regions** (1997) | Stack-based regions | Yes (lexical scoping) | No | Partial (regions, no bounds) |
| **Cyclone** (2002) | Regions + growable regions | Manual annotations | No | No (has GC fallback) |
| **RAML** (Hoffmann 2010) | No | No | Yes (polynomial) | No |
| **MLKit** (Tofte et al. 2004) | Region-based, no GC | Yes | No | Partial (regions only) |
| **Rust** | Stack + heap | No (lifetimes ≠ regions) | No | No |
| **ATS** (Xi 2016) | Proof-carrying regions | Manual proofs | Yes (via proofs) | Partial (research) |
| **This proposal** | **Static plans where data-independent** | **QTT + Alloc<R>** | **RAML-derived** | **Yes (fail closed)** |

**Key difference:**  
- Tofte-Talpin: Regions but no resource bounds → can't size arenas statically.  
- RAML: Bounds but no regions → can't eliminate allocator.  
- **We combine both:** Regions (scope) + bounds (size) = static plan.

---

#### What Is Genuinely New

**No precedent:**
1. **Combining QTT grades + Tofte-Talpin regions + RAML bounds:** Each system exists alone; the synthesis is novel.
2. **Zero-dynamic-allocator as a fail-closed default:** If bound is data-independent, emit static plan; else compile error or explicit `Alloc` effect. No silent fallback to dynamic allocator.
3. **Hybrid static/dynamic:** Same function mixes stack-allocated (static) and heap-allocated (dynamic) data, with clear boundary.

**Closest related:**
- **MLKit** (Tofte et al. 2004): Region-based ML without GC. But: no resource bounds, so arenas can still overflow.
- **ATS** (Xi 2016): Dependent types + proofs for memory safety. But: manual proofs, no automated static plans.

---

#### Risks and Open Problems

1. **Bound inference precision:** RAML works for polynomial bounds; affine/linear bounds (our target) are simpler but may under-approximate.  
   *Mitigation:* Conservative approximation; user can annotate tighter bounds manually.

2. **Data-independence criterion:** Is it too restrictive? Most real code depends on input size.  
   *Mitigation:* Hybrid strategy: static plan for fixed overhead, dynamic for variable part.

3. **Compile-time cost:** Region inference + bound inference + plan generation = expensive?  
   *Mitigation:* Cache plans; only recompute when function changes.

---

### (C) Epistemic Memory: Bounds as Uncertain Quantities

**Claim:** Allocation bounds are themselves **epistemic quantities** — exact, upper bound, estimated with uncertainty, or unknown — tracked like Sounio's `Knowledge<f64>`. The checker distinguishes **proved bounds** from **measured bounds**, and provenance of each allocation is queryable.

---

#### Formal Sketch

**Epistemic bound:**

```lean
inductive BoundKind where
  | Exact (proof : LeanProof)              -- Proved bound (e.g., 8 * n + 16)
  | UpperBound (proof : LeanProof)         -- Proved upper bound (≤ f(n))
  | Estimated (μ : ℝ) (σ : ℝ) (source : ProvenanceSource)  -- Measured: μ ± σ
  | Unknown                                 -- No bound; reject or fall back to ω

structure EpistemicBound where
  expr : BoundExpr
  kind : BoundKind
  confidence : ℝ                           -- For Estimated: confidence level (e.g., 95%)
  provenance : Provenance                  -- Where did this bound come from?

-- Provenance: like Knowledge<T>
structure Provenance where
  source : ProvenanceSource
  region : RegionId
  transforms : List Transform
```

**Bound algebra (GUM-style):**

```lean
-- Combine bounds (e.g., sequential composition)
def combineBounds (b₁ b₂ : EpistemicBound) : EpistemicBound :=
  match b₁.kind, b₂.kind with
  | Exact p₁, Exact p₂ => 
      { expr := b₁.expr + b₂.expr
      , kind := Exact (proof_add p₁ p₂)
      , confidence := 1.0
      , provenance := merge b₁.provenance b₂.provenance }
  | Estimated μ₁ σ₁ _, Estimated μ₂ σ₂ _ =>
      -- GUM uncertainty propagation
      { expr := b₁.expr + b₂.expr
      , kind := Estimated (μ₁ + μ₂) (sqrt (σ₁² + σ₂²)) ...
      , confidence := min b₁.confidence b₂.confidence
      , provenance := merge b₁.provenance b₂.provenance }
  | Exact _, Estimated _ _ _ => 
      -- Exact + uncertain = uncertain (conservative)
      { kind := UpperBound ..., ... }
  | Unknown, _ | _, Unknown => 
      { kind := Unknown, ... }

-- Soundness: if bound is Exact, actual ≤ bound always
theorem bound_soundness_exact (b : EpistemicBound) (exec : Execution) :
  b.kind = Exact _ → actualAlloc exec ≤ evalBound b.expr := by sorry

-- Confidence: if bound is Estimated μ ± σ, actual ≤ μ + k*σ with probability ≥ confidence
theorem bound_soundness_estimated (b : EpistemicBound) (exec : Execution) (k : ℝ) :
  b.kind = Estimated μ σ _ → 
  Prob (actualAlloc exec ≤ μ + k * σ) ≥ b.confidence := by sorry
```

---

#### Sounio Syntax Example

**Proved bound:**

```sio
fn parse_tokens(tokens: &[Token]) -> AST
    with Alloc<Parse> ≤ exact(16 * tokens.len() + 128)
{
    // Bound is PROVED: Lean theorem states each token → ≤ 16 bytes
    ...
}
```

**Measured bound (profiling):**

```sio
fn optimize_ir(ir: &IRModule) -> IRModule
    with Alloc<Optimize> ≤ measured(32 * ir.size(), uncertainty: 0.15, confidence: 0.95)
{
    // Bound is MEASURED: profiled over 1000 modules
    // Mean: 32 bytes/instruction, σ = 15%, confidence = 95%
    // Checker accepts this, but tags it as "measured, not proved"
    ...
}
```

**Upper bound (conservative):**

```sio
fn infer_types(module: &Module) -> TypedModule
    with Alloc<Check> ≤ upper_bound(128 * module.decls.len())
{
    // Bound is UPPER: proved ≤ 128 * n, but actual may be much less
    // (Conservative: assumes worst-case, not typical-case)
    ...
}
```

**Provenance tracking:**

```sio
fn query_bound_provenance(f: &Function) with IO {
    let bound = f.alloc_bound()
    
    println("Function: {}", f.name)
    println("Bound: {}", bound.expr)
    println("Kind: {}", bound.kind)  // Exact / UpperBound / Estimated / Unknown
    
    match bound.kind {
        BoundKind::Exact(proof) => {
            println("Proved in: {}", proof.lean_file)
        }
        BoundKind::Estimated(μ, σ, source) => {
            println("Measured: {} ± {} bytes", μ, σ)
            println("Confidence: {}%", bound.confidence * 100.0)
            println("Profiled on: {} runs", source.sample_size)
            println("Date: {}", source.timestamp)
        }
        BoundKind::UpperBound(proof) => {
            println("Upper bound proved; actual may be lower")
        }
        BoundKind::Unknown => {
            println("WARNING: No bound; function may overflow arena")
        }
    }
}
```

---

#### Closest Prior Work

| System | Bounds | Uncertainty | Provenance | Proved vs Measured |
|--------|--------|-------------|------------|---------------------|
| **RAML** | Polynomial | No | No | All inferred (no distinction) |
| **Granule** | Graded types | No | No | Type-checked, not proved |
| **Liquid Haskell** | Refinements | No | No | SMT-proved |
| **Sounio `Knowledge<T>`** | Values | Yes (GUM) | Yes | Measured vs computed |
| **This proposal** | **Allocation bounds** | **Yes (GUM-style)** | **Yes** | **Proved vs measured vs unknown** |

**Key difference:**  
- Existing systems: Bounds are **binary** (checked or not).  
- We treat bounds as **epistemic quantities**: exact, approximate, or unknown, with confidence intervals and provenance.

---

#### What Is Genuinely New

**No precedent:**
1. **Bounds as first-class epistemic values:** Allocation bounds carry uncertainty (`μ ± σ`) and confidence, like `Knowledge<f64>` for physical measurements.
2. **Proved vs measured distinction:** Compiler tracks whether bound came from a Lean proof, an SMT solver, or profiling data.
3. **Provenance of bounds:** Each bound records its origin (theorem, benchmark, conservative estimate), queryable at compile time.

**Closest related:**
- **Sounio's own `Knowledge<T>`** (epistemic types for values). We extend the same epistemic tracking to **resource bounds**.
- **Granule** (Orchard et al. 2019): Graded modal types. We add **uncertainty** to grades.

---

#### Risks and Open Problems

1. **Uncertainty propagation complexity:** GUM for bounds requires covariance tracking (if two functions share code, their bounds are correlated). Can we handle this?  
   *Mitigation:* Start with independent bounds; add correlation in later stage.

2. **Confidence calibration:** How do we set confidence levels (90%, 95%, 99%)? Profiling over how many runs?  
   *Mitigation:* Default to conservative (99%); user can relax with explicit annotation.

3. **Measured bounds may be wrong:** Profiling on small inputs → under-estimate for large inputs.  
   *Mitigation:* Compiler requires safety margin (e.g., measured × 1.5) or explicit user override.

---

### (D) Heterogeneous Regions: CPU, GPU, Distributed Memory

**Claim:** The same `Alloc<R>` effect spans CPU arenas, GPU device memory, and distributed/remote memory. Data movement between regions is a **typed move** with transfer cost encoded in the type.

---

#### Formal Sketch

**Heterogeneous region hierarchy:**

```lean
inductive RegionLocation where
  | CPU (arena : CPUArena)
  | GPU (device : DeviceId)
  | Remote (node : NodeId)
  | Persistent (storage : StorageId)

structure Region where
  id : RegionId
  location : RegionLocation
  size : Nat
  latency : Latency           -- Access latency (ns)
  bandwidth : Bandwidth       -- Transfer bandwidth (GB/s)
```

**Data movement as typed move:**

```lean
-- Transfer effect: move data from region R₁ to R₂
effect Transfer<R₁, R₂> {
    fn move_data(src : Ptr<T, R₁>) -> Ptr<T, R₂>
        cost transfer_cost<T, R₁, R₂>
}

def transfer_cost (T : Type) (R₁ R₂ : Region) : Cost :=
  { bytes := sizeof T
  , latency := R₂.latency
  , time_estimate := sizeof T / min R₁.bandwidth R₂.bandwidth
  }
```

**Type system:**

```lean
-- Pointer annotated with region
structure Ptr (T : Type) (R : Region) where
  addr : Address
  region_tag : R  -- Compile-time only (erased)

-- Transfer: typed move between regions
def transfer {T : Type} {R₁ R₂ : Region} 
    (p : Ptr T R₁) : Ptr T R₂ with Transfer<R₁, R₂> :=
  -- Implementation: memcpy if CPU-to-CPU; cudaMemcpy if CPU-to-GPU; network send if CPU-to-Remote
  ...

-- Soundness: pointer in region R₁ cannot be dereferenced in region R₂
theorem region_isolation {T : Type} {R₁ R₂ : Region} (p : Ptr T R₁) :
  R₁ ≠ R₂ → ¬(deref p : Context R₂) := by sorry
```

---

#### Sounio Syntax Example

**CPU → GPU transfer:**

```sio
fn train_model(data: &[f64]) -> Model 
    with Alloc<GPU>, Transfer<CPU, GPU>, IO 
{
    // Allocate on GPU
    let gpu_data: Ptr<[f64], GPU> = alloc_gpu(data.len())
    
    // Transfer: CPU → GPU
    transfer(data, gpu_data)  // Type: Ptr<[f64], CPU> → Ptr<[f64], GPU>
    
    // Compute on GPU
    let model = train_on_gpu(gpu_data)
    
    // Transfer result: GPU → CPU
    let cpu_model = transfer(model, CPU)
    return cpu_model
}

// Transfer cost visible in type
fn transfer<T, R1, R2>(src: Ptr<T, R1>, dst_region: R2) 
    -> Ptr<T, R2> 
    with Transfer<R1, R2> 
    cost sizeof(T) / bandwidth(R1, R2)
{
    match (R1.location, R2.location) {
        (CPU, GPU) => cuda_memcpy_h2d(src, dst),
        (GPU, CPU) => cuda_memcpy_d2h(src, dst),
        (CPU, Remote(node)) => network_send(src, node),
        (Remote(node), CPU) => network_recv(node),
        _ => memcpy(src, dst)  // Same location or CPU-to-CPU
    }
}
```

**Distributed computation:**

```sio
fn distributed_reduce(data: Vec<f64>) -> f64
    with Alloc<CPU>, Alloc<Remote>, Transfer<CPU, Remote>, IO
{
    let n_nodes = cluster_size()
    let chunk_size = data.len() / n_nodes
    
    var partial_sums: [f64; MAX_NODES] = [0.0; MAX_NODES]
    
    for i in 0..n_nodes {
        let chunk = data.slice(i * chunk_size, (i+1) * chunk_size)
        
        // Transfer chunk to remote node
        let remote_chunk: Ptr<[f64], Remote(i)> = transfer(chunk, Remote(i))
        
        // Compute on remote node (async)
        let remote_sum = async_sum(remote_chunk)  // Returns Ptr<f64, Remote(i)>
        
        // Transfer result back
        partial_sums[i] = *transfer(remote_sum, CPU)
    }
    
    // Local reduction
    var total: f64 = 0.0
    for s in partial_sums {
        total += s
    }
    return total
}
```

**Persistent storage:**

```sio
fn checkpoint_model(model: &Model) with Alloc<Persistent>, Transfer<CPU, Persistent>, IO {
    // Transfer to persistent storage (disk/SSD)
    let storage_ptr: Ptr<Model, Persistent> = alloc_persistent(sizeof(Model))
    transfer(model, storage_ptr)
    
    println("Model checkpointed to storage")
}

fn restore_model() -> Model with Transfer<Persistent, CPU>, IO {
    let storage_ptr: Ptr<Model, Persistent> = load_checkpoint()
    let cpu_model: Ptr<Model, CPU> = transfer(storage_ptr, CPU)
    return *cpu_model
}
```

---

#### Closest Prior Work

| System | Heterogeneous Memory | Typed Transfers | Cost Model | Regions |
|--------|----------------------|-----------------|------------|---------|
| **CUDA** | CPU + GPU | Manual `cudaMemcpy` | No | No |
| **OpenCL** | CPU + GPU + Accelerators | Manual buffer management | No | No |
| **Chapel** | Distributed arrays | Locale-aware | No | Implicit (locales) |
| **X10** | Distributed places | Place types | No | Yes (places) |
| **Singularity OS** | Typed channels | Yes (message passing) | No | No |
| **Verona** | Regions + concurrency | Yes (ownership) | No | Yes (cowns) |
| **This proposal** | **CPU + GPU + Remote + Persistent** | **Yes (Ptr<T, R>)** | **Yes (transfer_cost)** | **Yes (Alloc<R>)** |

**Key difference:**  
- CUDA/OpenCL: Manual memory management; no type-level region tracking.  
- Chapel/X10: Distributed memory, but no GPU or cost model.  
- Verona: Regions, but CPU-only.  
- **We unify:** Same `Alloc<R>` abstraction spans CPU, GPU, distributed, and persistent storage.

---

#### What Is Genuinely New

**No precedent:**
1. **Unified region abstraction across CPU/GPU/Remote/Persistent:** Existing systems handle at most two (CPU+GPU or CPU+Remote), not all four.
2. **Transfer cost in the type:** `transfer<T, R1, R2>` has cost `sizeof(T) / bandwidth(R1, R2)` visible at compile time.
3. **Effect-based data movement:** `Transfer<R₁, R₂>` is an algebraic effect, composable with `Alloc`, `IO`, `Async`.

**Closest related:**
- **X10 places** (Saraswat et al. 2010): Distributed memory with place types. We add **GPU** and **cost model**.
- **Verona regions** (Swasey et al. 2020): Ownership regions for concurrency. We add **heterogeneous memory**.

---

#### Risks and Open Problems

1. **Cost model accuracy:** Transfer time = `bytes / bandwidth` ignores latency, contention, network topology.  
   *Mitigation:* Model is approximate; use for cost-aware scheduling, not hard real-time.

2. **Async transfers:** GPU/network transfers are async; how to model in effect system?  
   *Mitigation:* Add `Async` effect; `await transfer(...)` for blocking.

3. **Fault tolerance:** Remote transfers can fail (network partition, node crash). How to handle in type system?  
   *Mitigation:* Add `Result<T, TransferError>`; require explicit error handling.

---

### (E) Units of Memory: Bytes as a Dimension

**Claim:** Bytes, cells, and words are **dimensions** in Sounio's units system. Allocation bound arithmetic is **dimension-checked** like physical units (`kg`, `m`, `s`).

---

#### Formal Sketch

**Memory units:**

```lean
-- Memory dimensions (like length, mass, time in physics)
inductive MemoryUnit where
  | Byte          -- Base unit
  | Cell          -- Architecture-dependent cell (e.g., 8 bytes on x86-64)
  | Word          -- Machine word (e.g., 8 bytes on 64-bit)
  | Handle        -- GC handle (Sounio-specific)

-- Annotated quantity
structure Memory where
  value : ℝ
  unit : MemoryUnit

-- Conversion
def to_bytes (m : Memory) : Memory :=
  match m.unit with
  | Byte => m
  | Cell => { value := m.value * CELL_SIZE, unit := Byte }
  | Word => { value := m.value * WORD_SIZE, unit := Byte }
  | Handle => { value := m.value * HANDLE_SIZE, unit := Byte }

-- Dimension-checked arithmetic
def add_memory (m₁ m₂ : Memory) : Memory :=
  if m₁.unit = m₂.unit then
    { value := m₁.value + m₂.value, unit := m₁.unit }
  else
    -- Convert to common unit (bytes)
    let b₁ := to_bytes m₁
    let b₂ := to_bytes m₂
    { value := b₁.value + b₂.value, unit := Byte }

-- Type error: mixing incompatible units
def mul_memory_time (m : Memory) (t : Time) : MemoryTime :=
  { value := m.value * t.value, unit := (m.unit, t.unit) }
  -- Can't add Memory + MemoryTime (dimension error)
```

---

#### Sounio Syntax Example

**Memory units in bounds:**

```sio
unit byte    // Base memory unit
unit cell = 8 * byte
unit word = 8 * byte
unit handle = 16 * byte

fn parse_tokens(tokens: &[Token]) -> AST
    with Alloc<Parse> ≤ 16 * tokens.len() * byte + 128 * byte
{
    // Bound is dimension-checked:
    // 16 * tokens.len() : scalar × count = scalar
    // scalar * byte = Memory
    // Memory + Memory = Memory ✓
    ...
}

// ERROR: Dimension mismatch
fn wrong_bound() with Alloc ≤ 16 * byte + 3.5 * second {
    // ERROR: Cannot add Memory (byte) and Time (second)
    ...
}

// Conversion
fn allocate_handles(n: i64) with Alloc ≤ n * handle {
    // n * handle = n * 16 * byte = (16 * n) byte
    ...
}
```

**Complex bounds with units:**

```sio
fn process_matrix(rows: i64, cols: i64) 
    with Alloc ≤ rows * cols * (8 * byte) + (rows + cols) * handle
{
    // rows * cols * (8 * byte) : matrix elements (f64 = 8 bytes each)
    // (rows + cols) * handle : row/column handle arrays
    // Total: dimension-checked, auto-converts to bytes
    ...
}

// Physical units + memory units
fn simulate_particles(n: i64, dt: f64<second>, steps: i64)
    with Alloc ≤ n * (6 * 8 * byte) + steps * (n * handle)
    cost steps * n * dt  // Cost is Time (not Memory)
{
    // 6 * 8 * byte per particle (position xyz + velocity xyz, f64 each)
    // steps * n * handle for trajectory storage
    // Cost: physical time, separate dimension from memory
    ...
}
```

---

#### Closest Prior Work

| System | Units | Memory Units | Dimension Checking | Integrated with Types |
|--------|-------|--------------|--------------------|-----------------------|
| **F# Units of Measure** | Yes (kg, m, s) | No | Yes | Yes |
| **Sounio units** (current) | Yes (kg, m, s, mol, ...) | No | Yes | Yes (`f64<m>`) |
| **Ada dimensions** | Limited | No | Yes | Partial |
| **Rust (uom crate)** | Yes (library) | No | Yes (runtime) | No (library) |
| **This proposal** | **Yes** | **Yes (byte, cell, word, handle)** | **Yes** | **Yes (bound<byte>)** |

**Key difference:**  
- F#, Ada, Rust: Units for physical quantities (length, mass, time).  
- Sounio (current): Units for physical + chemical + biological quantities.  
- **We add:** Units for **memory** (bytes, cells, handles), treated identically to physical units.

---

#### What Is Genuinely New

**No precedent:**
1. **Memory as a dimensional quantity:** bytes, cells, and handles are dimensions like `kg` or `m`, with dimension-checked arithmetic.
2. **Allocation bounds carry units:** `with Alloc ≤ 16 * n * byte` is type-checked; mixing bytes and seconds = compile error.
3. **Conversion between memory units:** `cell` = 8 × `byte`, `handle` = 16 × `byte` (architecture-dependent), auto-converted in bounds.

**Closest related:**
- **Sounio's own units system:** Already handles physical dimensions. We extend to **memory dimensions**.
- **F# Units of Measure** (Kennedy 1997): Pioneering work on dimensional analysis in types. We apply to memory.

---

#### Risks and Open Problems

1. **Architecture dependence:** `cell` and `word` sizes vary by target (32-bit vs 64-bit, x86 vs ARM). How to handle cross-compilation?  
   *Mitigation:* Units are target-specific; bounds re-check per target.

2. **Overloading `byte`:** Already a type (`u8`). Now also a unit. Confusing?  
   *Mitigation:* Context disambiguates: `let x: u8` (type) vs `16 * byte` (unit).

3. **Complexity:** Does dimension-checking memory add significant overhead?  
   *Mitigation:* Units erased at runtime (grade-0); compile-time only.

---

## Staged Migration Plan (Tied to #2966)

### Stage 0: Kill Hash Field Resolution (Immediate)

**Goal:** Eliminate untyped field-index resolution by name hash (`.span`→51 bug).

**Changes:**
1. Remove `field_idx_from_name` hash fallback (`name[0] % 64`)
2. Require `struct_layouts` table populated at lowering time
3. Unknown struct at field access = **compile error E046**

**Exit criteria:**
- [ ] `scripts/ci/struct_field_lowering_gate.sh` passes
- [ ] No hash fallback code reachable in `self-hosted/ir/lower.sio`
- [ ] `tests/compile-fail/unknown_struct_field_access.sio` added

**Estimated effort:** 2-3 weeks (IR refactor, gate, tests)

---

### Stage 1: Per-Phase Arenas + Move Semantics

**Goal:** Implement per-phase arenas in Madaros; formalize move semantics via grade-1.

**Changes:**
1. Add `PhaseArena` enum: `Parse`, `Check`, `Lower`, `Codegen`
2. Thread `PhaseId` through compiler, allocate from appropriate arena
3. Parser: accept `@ 1`, `@ ω` syntax (grade-0 later)
4. Checker: track grades in `TypeKey`, enforce move-after-use

**Exit criteria:**
- [ ] `self-hosted/compiler/main.sio` uses separate arenas per phase
- [ ] Benchmark: parse 1000 modules without arena overflow
- [ ] `tests/compile-fail/use_after_move_grade1.sio` added
- [ ] `docs/audit/PHASE_ARENA_DISPATCH_<date>.md` documents allocation patterns

**Estimated effort:** 4-6 weeks (major compiler refactor)

---

### Stage 2: Allocation Effect `Alloc<R>`

**Goal:** Reify allocation as an algebraic effect; implement region typing.

**Changes:**
1. Add `EffectAlloc(RegionKey)` to effect system
2. Define region hierarchy: `Global`, `Phase<P>`, `Function`, `Pool<K>`, `GPU`
3. Checker: thread `Alloc<R>` through function signatures
4. IR: lower `allocate` calls to region-specific handlers
5. Effect inference: default to `Alloc<Function>` if unspecified

**Exit criteria:**
- [ ] `tests/run-pass/alloc_effect_function_region.sio` passes
- [ ] `tests/compile-fail/alloc_region_escape.sio` refuses value escaping region
- [ ] Effect system docs updated: `docs/spec/2026-10/03-effects-handlers-and-concurrency.md`

**Estimated effort:** 6-8 weeks (effect system extension, IR threading)

---

### Stage 3: Resource Bounds

**Goal:** Declare and infer allocation bounds; compute arena sizes.

**Changes:**
1. Parser: accept `with Alloc<R> ≤ B` syntax
2. Add `BoundExpr` AST: constants, variables, `+`, `c*`, `max`, `if`
3. Checker: `infer_alloc_bound` traverses IR, emits symbolic bound
4. Solver: `scripts/bounds/solve_arena_sizes.py` computes max callstack bound
5. Codegen: emit arena-size assertion; overflow = compile error E300

**Exit criteria:**
- [ ] `tests/run-pass/alloc_bound_linear.sio` declares `≤ 16 * n`
- [ ] `tests/compile-fail/alloc_bound_exceeded.sio` refuses program requiring 3 GiB in 2 GiB arena
- [ ] Compiler self-hosting: all phases declare bounds, no overflow

**Estimated effort:** 8-12 weeks (bound inference, solver integration, testing)

---

### Stage 4: Lean 4 Proofs

**Goal:** Formalize and prove region soundness, bound correctness, grade preservation.

**Changes:**
1. `formal/regions/syntax.lean`: region calculus syntax
2. `formal/regions/semantics.lean`: operational semantics, heap model
3. `formal/regions/soundness.lean`: Theorem 1 (no escape)
4. `formal/bounds/inference.lean`: bound language semantics
5. `formal/bounds/correctness.lean`: Theorem 2 (actual ≤ declared)
6. `formal/qtt/syntax.lean`: graded types
7. `formal/qtt/preservation.lean`: Theorem 3 (grade semantics)

**Exit criteria:**
- [ ] All three theorems stated and proven (may admit axioms; document them)
- [ ] `scripts/ci/formal_verification_gate.sh` checks Lean 4 proofs
- [ ] Paper: "Quantitative Types and Effect-Based Regions in Sounio" (draft)

**Estimated effort:** 12-24 weeks (proof engineering; can overlap with Stages 1-3)

---

### Moonshot Track (Parallel to Stages 1-4)

The five moonshot proposals (A-E) run on a **separate track** — bolder, higher risk, potentially transformative. They are not blocking; core system (Stages 0-4) ships without them.

---

#### Moonshot Stage M1: Units of Memory (E) — Foundation

**Goal:** Add `byte`, `cell`, `word`, `handle` as memory dimensions; dimension-check bound arithmetic.

**Changes:**
1. Extend units system: `stdlib/units/memory.sio` defines memory dimensions
2. Parser: accept `16 * byte`, `n * handle` in bounds
3. Checker: dimension-check bound expressions (reject `byte + second`)
4. Codegen: erase units (grade-0)

**Exit criteria:**
- [ ] `tests/run-pass/bound_memory_units.sio` declares `with Alloc ≤ 16 * n * byte`
- [ ] `tests/compile-fail/bound_dimension_mismatch.sio` refuses `byte + second`
- [ ] Units conversion: `cell = 8 * byte` works

**Estimated effort:** 4-6 weeks

---

#### Moonshot Stage M2: Epistemic Memory (C) — Tracking Uncertainty

**Goal:** Bounds carry uncertainty (`Estimated μ ± σ`); distinguish proved/measured/unknown.

**Changes:**
1. `BoundKind` enum: `Exact | UpperBound | Estimated | Unknown`
2. Parser: accept `proven(...)`, `measured(..., uncertainty: ε, confidence: p)`
3. Checker: track provenance; GUM-style uncertainty propagation for bounds
4. Diagnostic: warn on `Unknown` bounds; error on arena overflow

**Exit criteria:**
- [ ] `tests/run-pass/bound_epistemic_measured.sio` uses `measured(32 * n, uncertainty: 0.15)`
- [ ] `tests/run-pass/bound_epistemic_proven.sio` uses `proven(8 * n + 16)` with Lean proof
- [ ] Provenance query: `query_bound_provenance(f)` returns source/confidence

**Estimated effort:** 6-8 weeks

---

#### Moonshot Stage M3: Static Memory Plans (B) — Zero Allocator

**Goal:** Infer static allocation plan for data-independent code; emit stack-only or preallocated arenas.

**Changes:**
1. Data-independence analysis: `isDataIndependent(bound)`
2. Static plan inference: `inferStaticPlan(f)` lays out fixed offsets
3. Codegen: emit stack frame with fixed layout (no `__rgn_enter`)
4. Hybrid: mix static + dynamic in same function

**Exit criteria:**
- [ ] `tests/run-pass/static_plan_fixed_array.sio` uses no dynamic allocator
- [ ] Compiler reports: "Function `process_fixed_data` compiled with static plan (800 bytes stack)"
- [ ] Benchmark: self-hosting parse phase with 80% static, 20% dynamic

**Estimated effort:** 10-14 weeks (complex; Tofte-Talpin + RAML integration)

---

#### Moonshot Stage M4: Heterogeneous Regions (D) — CPU/GPU/Remote

**Goal:** Same `Alloc<R>` spans CPU, GPU, Remote, Persistent; typed transfers with cost.

**Changes:**
1. `RegionLocation` enum: `CPU | GPU | Remote | Persistent`
2. `Ptr<T, R>` annotated with region
3. `Transfer<R₁, R₂>` effect: typed move between regions
4. Cost model: `transfer_cost(T, R₁, R₂)` = bytes / bandwidth
5. Codegen: `cudaMemcpy` for GPU, network send for Remote

**Exit criteria:**
- [ ] `tests/run-pass/heterogeneous_cpu_gpu.sio` transfers data to GPU, computes, transfers back
- [ ] `tests/run-pass/heterogeneous_distributed.sio` distributes work across nodes
- [ ] Type error: `deref(gpu_ptr)` on CPU context refused

**Estimated effort:** 12-16 weeks (requires GPU/network runtime)

---

#### Moonshot Stage M5: Self-Certifying Bootstrap (A) — Memory as Theorem

**Goal:** Madaros emits memory certificate; next generation verifies it; fixed point = theorem.

**Changes:**
1. `MemoryCertificate` structure: bounds, proofs, witness, source hash
2. Compiler self-declaration: `with Alloc ≤ proven(...)` in `self-hosted/compiler/main.sio`
3. Bootstrap verification: `scripts/bootstrap/verify_certificate.sio`
4. Lean proofs: bounds for parse/check/lower/codegen phases
5. Fixed-point check: `Cₙ = Cₙ₋₁` (bit-identical)

**Exit criteria:**
- [ ] Gen 0 (lean_single) emits measured certificate
- [ ] Gen 1 (Madaros) emits proved certificate (at least Parse/Codegen phases)
- [ ] Gen 2 verifies Gen 1 certificate; Lean checker passes
- [ ] Fixed point: Gen 3 = Gen 2 (certificates bit-identical)
- [ ] Paper: "Self-Certifying Compiler: Memory Behaviour as a Theorem"

**Estimated effort:** 24-36 weeks (most ambitious; proof-heavy)

---

## Implementation Notes

### Per-Phase Arena Sizing (Founder Decision 1)

Based on Madaros self-compilation profiling (to be measured):

| Phase | Estimated Bound | Current (fixed) | Ratio |
|-------|-----------------|-----------------|-------|
| Parse | `8 * source.bytes()` | 512 MiB | 1.6× |
| Check | `16 * ast.nodes()` | 1024 MiB | 2.0× |
| Lower | `32 * ir.instrs()` | 1024 MiB | 1.5× |
| Codegen | `4 * asm.bytes()` | 512 MiB | 1.2× |

**Action:** Measure actual allocation in Stages 0-1, refine bounds in Stage 3.

---

### Effect Inference Defaults

```sio
// No explicit effect → infer Alloc<Function> if body allocates
fn f(x: i64) -> Vec<i64> {
    var v = Vec::new()
    v.push(x)
    return v
}
// Equivalent to:
fn f(x: i64) -> Vec<i64> with Alloc<Function> { ... }

// If no allocation → no Alloc effect
fn g(x: i64) -> i64 { x + 1 }
// Equivalent to:
fn g(x: i64) -> i64 { ... }
```

**Rationale:** Minimize annotation burden; explicit `with Alloc<R>` only for non-default regions.

---

### Grade Defaults

| Type | Default Grade | Rationale |
|------|---------------|-----------|
| `i64`, `f64`, `bool`, `char` | ω | Primitives are `Copy` |
| `Vec<T>`, `Box<T>`, `String` | 1 | Heap-allocated, non-`Copy` |
| `&T` | ω | References are `Copy` |
| `&!T` | 1 | Exclusive reference, not `Copy` |
| `Knowledge<T>` | ω (value), 1 (provenance) | Value is `Copy`, metadata is unique |

---

### Backward Compatibility

**Stage 0-1:** No syntax changes visible to users; arena refactor is internal.  
**Stage 2:** Allocation effect is opt-in; existing code defaults to `Alloc<Function>`.  
**Stage 3:** Bounds are optional; unbounded functions tolerated with warning.  
**Stage 4:** Proofs are compiler-internal; no user-facing changes.

**Breaking changes:** None until Stage 3 makes `Alloc` effect explicit in function signatures. Migration: `./bin/souc migrate-alloc` inserts inferred effects.

---

## Success Metrics

1. **Correctness:**
   - [ ] No `.span`→51 style bugs (Stage 0 gate)
   - [ ] No "handles full" errors in self-hosting (Stage 1-2)
   - [ ] Lean 4 proofs check (Stage 4)

2. **Performance:**
   - [ ] Self-hosting compile time ≤ 1.1× current (10% overhead acceptable)
   - [ ] Memory usage ≤ 0.9× current (tighter arenas, less waste)

3. **Usability:**
   - [ ] Users write allocation bounds for <5% of functions (rest inferred)
   - [ ] Effect inference success rate ≥ 95% (manual annotation rare)

4. **Novelty:**
   - [ ] Paper accepted at POPL, PLDI, or ICFP (top-tier PL venue)
   - [ ] Cited by ≥3 other systems languages within 2 years

---

## Alternatives Considered

### Alternative 1: Keep Dynamic Regions (ADR-010)

**Pros:**  
- Already implemented and stable  
- No user-facing changes  

**Cons:**  
- Conservative escape (false positives)  
- No compile-time guarantees  
- No epistemic provenance  

**Verdict:** Insufficient for scientific computing; proceed with RFC.

---

### Alternative 2: Adopt Rust Lifetimes Directly

**Pros:**  
- Well-understood (10+ years in production)  
- RustBelt proof exists  

**Cons:**  
- Annotation burden (every reference needs `'a`)  
- Does not address resource bounds  
- No epistemic integration  

**Verdict:** Lifetimes solve ownership but not allocation bounds; effect-based regions solve both.

---

### Alternative 3: Use Garbage Collection

**Pros:**  
- Simple programming model  
- No region escapes  

**Cons:**  
- Unpredictable latency (scientific computing requires determinism)  
- No compile-time memory guarantees  
- Does not address per-phase arenas  

**Verdict:** Incompatible with Sounio's deterministic systems programming goals.

---

## Conclusion

This RFC proposes a unified memory management system combining **Quantitative Type Theory grades**, **effect-based regions**, and **resource bounds**, proven sound via **Lean 4**. The design goes beyond state-of-the-art by making copy elision a typing consequence, preventing region escape through effects alone, and computing arena sizes at compile time.

The staged migration plan (4 stages, 12-24 months total) addresses founder-identified defects (#2966, #3000) and positions Sounio as a **novel synthesis** of linear types, effect systems, and resource analysis for epistemic scientific computing.

---

## References

### Ownership and Linearity

- **Atkey (2018).** "The Syntax and Semantics of Quantitative Type Theory." *LICS 2018.*
- **McBride (2016).** "I Got Plenty o' Nuttin'." *A List of Successes That Can Change the World, 2016.*
- **Brady (2021).** "Idris 2: Quantitative Type Theory in Practice." *ECOOP 2021.*
- **Bernardy et al. (2018).** "Linear Haskell: Practical Linearity in a Higher-Order Polymorphic Language." *POPL 2018.*
- **Grossman et al. (2002).** "Region-Based Memory Management in Cyclone." *PLDI 2002.*
- **Denisov & Karpinski (2022).** "Mutable Value Semantics." Hylo project documentation.
- **Swasey et al. (2020).** "Verona: Concurrent Ownership." Microsoft Research TR.
- **Vale (2022).** "Vale's Higher-Order RAII, Generational Memory, Immutable Heap, Hybrid-Generational Memory, and More." Vale website.

---

### Effect Systems

- **Leijen (2017).** "Type Directed Compilation of Row-Typed Algebraic Effects." *POPL 2017.*
- **Brachthäuser et al. (2020).** "Effekt: Capability-Passing Style for Type- and Effect-Safe, Extensible Effect Handlers in Scala." *JFP 2020.*
- **Lindley et al. (2017).** "Do Be Do Be Do." *POPL 2017.*
- **Bauer & Pretnar (2015).** "Programming with Algebraic Effects and Handlers." *JFP 2015.*

---

### Resource Analysis

- **Hoffmann & Hofmann (2010).** "Amortized Resource Analysis with Polynomial Potential." *ESOP 2010.*
- **Hoffmann et al. (2017).** "Automatic Static Cost Analysis for Parallel Programs." *ESOP 2017.*
- **Orchard et al. (2019).** "Quantitative Program Reasoning with Graded Modal Types." *ICFP 2019.*
- **Vazou et al. (2014).** "Refinement Types for Haskell." *ICFP 2014.*
- **O'Connor et al. (2016).** "Refinement through Restraint: Bringing Down the Cost of Verification." *ICFP 2016.*

---

### Region Calculi

- **Tofte & Talpin (1997).** "Region-Based Memory Management." *Information and Computation 1997.*
- **Tofte et al. (2004).** "Retrospective: Region-Based Memory Management." *POPL 2004.*

---

### Verified Systems Languages

- **Jung et al. (2018).** "RustBelt: Securing the Foundations of the Rust Programming Language." *POPL 2018.*
- **Leino (2010).** "Dafny: An Automatic Program Verifier for Functional Correctness." *LPAR 2010.*
- **Bhargavan et al. (2017).** "Verified Low-Level Programming Embedded in F*." *PACMPL 2017.*
- **Xi (2016).** "ATS: A Language That Combines Programming with Theorem Proving." *FLOPS 2016.*

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Grade** | Usage annotation: 0 (erased), 1 (linear), ω (unrestricted) |
| **QTT** | Quantitative Type Theory: grades on types |
| **Region** | Memory arena with scoped lifetime |
| **Effect handler** | Implementation of algebraic effect (e.g., `Alloc<R>` → bump allocator) |
| **Bound expression** | Symbolic allocation bound (e.g., `16 * n + 128`) |
| **Amortized analysis** | Bounding total resource use across operations (not per-operation worst-case) |
| **Provenance** | Origin metadata for `Knowledge<T>` values |
| **Epistemic type** | Type carrying uncertainty/provenance (e.g., `Knowledge<f64>`) |

---

## Appendix B: Open Questions for Reviewers

1. **Grade syntax:** `@ 1` vs `move`, `@ ω` vs `copy`? Or infer everything?
2. **Bound inference scope:** Require bounds only for compiler (trusted), or also stdlib/user code?
3. **Region polymorphism:** Allow `fn f<R>(x: T) with Alloc<R>` or restrict to ground regions?
4. **Lean 4 proof priority:** Start with Stage 4 (proofs) or Stages 1-3 (implementation)?
5. **Migration timeline:** 12-24 months realistic, or split into RFC-0001 (design) + RFC-0002 (implementation)?

---

## Appendix C: Relationship to Existing Sounio ADRs

| ADR | Relationship |
|-----|--------------|
| ADR-010 (Madaros Region Reclamation) | Stage 1 extends this with per-phase arenas; Stage 2 replaces dynamic barriers with static effect typing |
| ADR-004 (Capacity guards over silent corruption) | Resource bounds formalize "capacity" as compile-time property |
| ADR-007 (Second-order compilation) | Grade-0 (erased) enables compile-time proofs passed to compiler without runtime cost |

---

**END OF RFC-0001**
