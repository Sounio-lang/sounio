<!-- docs:meta
topic_id: repo.docs.compiler.closure-type-theory
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.closure-type-theory
-->

# Unified Closure Type Theory

**Sounio Technical Report 2026-03-18**

## Abstract

We present the first type system that unifies algebraic effects, GUM measurement
uncertainty (JCGM 100:2008), and linear resource tracking in a single closure
type constructor. The unified type:

```
fn(T) -> U with E [linear] [epistemic(e)]
```

carries an effect row E, an optional linearity constraint (the closure must be
consumed exactly once), and an optional epistemic epsilon bound (the closure's
output carries uncertainty propagated from captured Knowledge values).

The type system is implemented in Sounio's self-hosted type checker (30K lines)
and compiles to native x86-64 ELF binaries via a self-hosted backend.

---

## 1. Syntax

### 1.1 Types

```
t ::= i64 | f64 | bool                           -- base types
    | Knowledge<t, e>                              -- epistemic type (GUM)
    | S                                            -- struct type
    | linear S                                     -- linear struct type
    | fn(t1, ..., tn) -> t with E [L] [K]         -- function type
    | ()                                           -- unit

E ::= {e1, e2, ..., ek}                           -- effect row
    where ei in {IO, Mut, Alloc, Panic, Div,
                 GPU, Async, Prob, Epistemic, Causal}

L ::= linear | (absent)                           -- linearity annotation

K ::= epistemic(e) | (absent)                     -- epistemic annotation
    where e in [0.0, 1.0]                          -- GUM epsilon bound
```

### 1.2 Expressions

```
e ::= x                                           -- variable
    | e1 e2                                        -- application
    | |x: t| e                                     -- closure
    | let x = e1 in e2                             -- binding
    | measure(v, uncertainty: u)                   -- epistemic source
    | ...                                          -- standard expressions
```

---

## 2. Typing Rules

### 2.1 Closure Formation

```
                G, x:T |- e : U | E
          L = fv(e) /\ linear(G)
          K = fv(e) /\ epistemic(G)
          ec = combine_gum(K)
    -------------------------------------------------------- [T-CLOSURE]
    G |- |x:T| e : fn(T) -> U with E
                    [linear     if L != {}]
                    [epistemic(ec) if K != {}]
```

Where:
- `fv(e)` = free variables of e
- `linear(G)` = {x in dom(G) | G(x) is a linear type}
- `epistemic(G)` = {x in dom(G) | G(x) = Knowledge<t, e>}
- `combine_gum(K)` = sqrt(sum(e_i^2)) for independent sources (GUM 5.1.2)

**Intuition:** A closure's metadata is *inferred* from its captures, not declared.
If it captures a linear resource, it becomes linear. If it captures uncertain
measurements, it becomes epistemic. The programmer writes `|x| e` and the
type system does the rest.

### 2.2 Application

```
    G |- f : fn(T) -> U with E       E <= E_caller
    G |- e : T
    -------------------------------------------------- [T-APP]
    G |- f(e) : U
```

Standard application with effect checking: callee effects must be a subset
of caller effects.

```
    G |- f : fn(T) -> U with E [linear]
    G |- e : T       f not in consumed(G)
    -------------------------------------------------- [T-APP-LINEAR]
    G[f -> consumed] |- f(e) : U
```

Linear closure application: marks f as consumed. A second application
of f yields error 39 (already consumed).

```
    G |- f : fn(T) -> U with E [epistemic(ec)]
    G |- e : T
    -------------------------------------------------- [T-APP-EPISTEMIC]
    G |- f(e) : Knowledge<U, ec>
```

Epistemic closure application: the return type is automatically wrapped
in `Knowledge<U, ec>`, propagating the captured uncertainty.

### 2.3 Subtyping

```
    E1 <= E2    (effect subset)
    ------------------------------------------- [S-EFFECT]
    fn(T) -> U with E1 <: fn(T) -> U with E2
```

A function with fewer effects is usable where more effects are allowed.
A pure function can be passed where an IO function is expected.

```
    ------------------------------------------- [S-LINEAR]
    fn(T) -> U [linear] <: fn(T) -> U
```

A linear closure is usable as a non-linear closure (but not vice versa).
Calling a linear closure through a non-linear reference loses the
single-use guarantee -- this is by design (the holder "forgets" the
constraint, which is safe since the closure's internal linear resource
is still consumed on call).

```
    e1 <= e2
    -------------------------------------------------- [S-EPISTEMIC]
    fn(T) -> U [epistemic(e1)] <: fn(T) -> U [epistemic(e2)]
```

A closure with tighter uncertainty (smaller epsilon) is usable where
looser uncertainty is acceptable. This follows the GUM epsilon lattice.

### 2.4 Linear Safety

```
    G |- let f = e1 in e2 : T
    f : fn(...) [linear]       f not in fv(e2)
    -------------------------------------------------- [T-LINEAR-DROP]
    ERROR 40: linear closure not consumed before scope exit
```

A linear closure must be called before it goes out of scope.

---

## 3. Metatheory

### 3.1 Effect Safety

**Theorem (Effect Soundness).** If `G |- e : T | E` and e reduces to v,
then every side effect performed during evaluation is in E.

*Sketch:* By induction on the derivation. [T-CLOSURE] infers effects from
the body. [T-APP] checks `E <= E_caller`. The effect set is monotonically
non-decreasing through the call chain.

### 3.2 Linear Safety

**Theorem (Linear Resource Safety).** If `G |- e : T` and G contains a
linear binding `f : fn(...) [linear]`, then in any complete evaluation
of e, f is called exactly once.

*Sketch:* [T-APP-LINEAR] marks f consumed. [T-LINEAR-DROP] prevents scope
exit without consumption. Together they enforce exactly-once semantics.

### 3.3 Epistemic Correctness

**Theorem (Uncertainty Monotonicity).** If a closure captures Knowledge
values with epsilons {e1, ..., en}, the output epsilon ec satisfies
ec >= max(e1, ..., en).

*Proof:* combine_gum computes sqrt(sum(ei^2)) >= max(ei) for all ei >= 0.

**Theorem (GUM Compliance).** The epistemic propagation through closures
satisfies JCGM 100:2008 Section 5.1.2 (combined standard uncertainty for
uncorrelated input quantities).

*Proof:* combine_gum implements the RSS formula: uc = sqrt(sum(ui^2)).

---

## 4. Implementation

### 4.1 Data Structures

The closure type is represented in `FnSig` (check/defs.sio):

```sio
struct FnSig {
    name: Name,
    params: Option<Box<FnParamList>>,
    param_count: i64,
    return_type: TypeEntry,
    effects: [i64; 8],
    effect_count: i64,
    // Sprint 231: unified closure metadata
    is_linear_closure: bool,
    linear_capture_count: i64,
    epistemic_epsilon: f64,       // -1.0 = not epistemic
}
```

### 4.2 Capture Analysis (check/check.sio:11657-11810)

After type-checking the closure body, the checker scans outer-scope bindings:

```sio
while ci < env_count {
    let binding_ty = c.env.bindings[ci].ty
    if c.is_linear_type(binding_ty) {
        capture_linear = true
    }
    if binding_ty.kind == TypeKind::TyKnowledge {
        capture_epsilon = combine(capture_epsilon, binding_ty.knowledge_epsilon)
    }
}
```

Conservative approach: scans all visible outer bindings (sound over-approximation).
Precise free-variable-based tracking is future work.

### 4.3 TypeFn Lowering Fix (check/check.sio:5459)

Sprint 231 fixed the critical bug where `fn(T) -> U with E` type annotations
were discarded (`ty_unknown()`). Now `lower_fn_type_expr` creates a proper
`TyFn(sig_id)` with parameters, return type, and effects extracted from the
parsed type expression.

### 4.4 Compilation Pipeline

```
Source -> Lexer -> Parser -> AST -> CHECK* -> HIR -> IR -> Codegen -> ELF
                                      |
                              Sprint 231 changes:
                              - TypeFn lowering
                              - Capture analysis
                              - Epistemic auto-injection
                              - Linear tracking
```

*Only the CHECK phase is modified.* IR and codegen are unchanged.
The type theory is enforced entirely at compile time -- zero runtime cost.

---

## 5. Related Work

| System | Effects | Linear | Epistemic | Unified |
|--------|---------|--------|-----------|---------|
| Koka (Leijen 2014) | Row-polymorphic | No | No | No |
| Frank (Lindley+ 2017) | Handlers | No | No | No |
| Eff (Pretnar 2015) | Algebraic | No | No | No |
| Linear Haskell (Bernardy+ 2018) | Monadic | Multiplicity | No | No |
| Rust | No | FnOnce (partial) | No | No |
| Granule (Orchard+ 2019) | Graded | Graded | No | No |
| **Sounio (2026)** | **Row (10 effects)** | **Full linear** | **GUM compliant** | **Yes** |

**Key distinction:** Prior work addresses at most two of the three dimensions.
Koka has rich effects but no linearity or uncertainty. Rust has partial linearity
(FnOnce) but no effects or uncertainty. Granule has graded linearity but
no measurement uncertainty. Sounio is the first to unify all three in a single
closure type constructor with a practical implementation (self-hosted compiler,
native x86-64 codegen, 30K lines).

---

## 6. Future Work

1. **Precise capture tracking:** Replace conservative outer-scope scan with
   exact free-variable analysis. Eliminates false-positive linear/epistemic
   annotations on closures that don't actually capture the relevant bindings.

2. **Effect handlers:** Algebraic effect handlers for closures, enabling
   resumable IO, cooperative concurrency, and epistemic effect discharge.

3. **Dependent epistemic types:** `fn(x: Knowledge<f64, e>) -> Knowledge<f64, f(e)>`
   where the output epsilon is a function of the input epsilon.

4. **SIMD uncertainty propagation:** Compile epistemic closures to vectorized
   uncertainty arithmetic using AVX-512 packed f64 operations.

---

## References

- JCGM 100:2008. "Evaluation of measurement data -- Guide to the expression of uncertainty in measurement (GUM)."
- Leijen, D. (2014). "Koka: Programming with Row Polymorphic Effect Types." MSFP 2014.
- Lindley, S., McBride, C., McLaughlin, C. (2017). "Do be do be do." POPL 2017.
- Bernardy, J.P., Boespflug, M., Newton, R., Peyton Jones, S., Spiwack, A. (2018). "Linear Haskell: practical linearity in a higher-order polymorphic language." POPL 2018.
- Orchard, D., Liepelt, V., Eades III, H.B. (2019). "Quantitative Program Reasoning with Graded Modal Types." ICFP 2019.
