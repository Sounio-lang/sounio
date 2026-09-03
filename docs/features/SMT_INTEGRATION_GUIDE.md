<!-- docs:meta
topic_id: repo.docs.features.smt-integration-guide
authority: historical
audience: users
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.features.smt-integration-guide
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# SMT Integration Implementation Guide
## Z3 Integration for Epistemic Refinement Types

**Status:** 90% Complete - Final integration needed
**Target:** Month 3-4 Milestone
**Estimated Remaining Work:** 2-3 days

---

## Current State Assessment

### ✅ What's Already Implemented

1. **Z3 Solver Integration** (`crates/souc/src/smt/z3_solver.rs` - 1143 LOC)
   - Full Z3 FFI bindings via z3-rs
   - EpistemicProperty enum with verification methods
   - Counterexample extraction
   - Statistics tracking
   - Timeout handling

2. **Proof Search** (`crates/souc/src/dependent/proof_search.rs` - 1072 LOC)
   - Confidence predicate decision procedures
   - Causal predicate checking (identifiability, d-separation)
   - Ontology predicate evaluation
   - Temporal predicate checking
   - Gradual typing fallback
   - Comprehensive test suite

3. **Epistemic Types** (`crates/souc/src/epistemic/mod.rs`)
   - Knowledge<T, τ, ε, δ, Φ> with temporal, confidence, ontology, provenance
   - Bayesian fusion, Dempster-Shafer combination
   - Fisher information geometry
   - SIMD-accelerated operations

### 🔧 What Needs to Be Done

**Missing Link:** Proof searcher doesn't call Z3 when built-in decision procedures fail.

**Required Changes:**
1. Add Z3 fallback in `ProofSearcher::confidence_decision` (lines 352-381)
2. Translate `ConfidencePredicate` to `SmtFormula`
3. Call Z3 solver when arithmetic decision procedures return `Unknown`
4. Extract counterexamples for failed proofs
5. Add integration tests

---

## Implementation Plan

### Phase 1: Translate Epistemic Predicates to SMT (1 day)

**File:** `crates/souc/src/dependent/proof_search.rs`

**Add method:**
```rust
impl<'a> ProofSearcher<'a> {
    /// Try Z3 when built-in decision procedures fail
    fn try_z3_epistemic(&self, pred: &ConfidencePredicate) -> ProofResult {
        #[cfg(feature = "smt")]
        {
            use crate::smt::z3_solver::{Z3Solver, EpistemicProperty};
            use z3::Context;

            // Create Z3 context
            let config = z3::Config::new();
            let ctx = Context::new(&config);
            let mut solver = Z3Solver::new(&ctx);

            // Translate predicate to Z3 formula
            let formula = self.translate_confidence_to_smt(pred);
            solver.assert(&formula)?;

            // Check satisfiability
            match solver.check_sat()? {
                VerificationResult::Sat => ProofResult::Proven(Proof::smt("Z3", pred)),
                VerificationResult::Unsat => {
                    let cex = solver.extract_counterexample();
                    ProofResult::Disproven { reason: format!("Counterexample: {:?}", cex) }
                }
                VerificationResult::Unknown => ProofResult::Unknown {
                    reason: "Z3 could not determine".to_string()
                },
                VerificationResult::Timeout => ProofResult::Unknown {
                    reason: "Z3 timeout".to_string()
                },
                VerificationResult::Error(e) => ProofResult::Unknown {
                    reason: format!("Z3 error: {}", e)
                },
            }
        }

        #[cfg(not(feature = "smt"))]
        {
            ProofResult::Unknown {
                reason: "SMT feature not enabled".to_string()
            }
        }
    }

    /// Translate ConfidencePredicate to SmtFormula
    fn translate_confidence_to_smt(&self, pred: &ConfidencePredicate) -> SmtFormula {
        match pred {
            ConfidencePredicate::Geq(lhs, rhs) => {
                let lhs_term = self.confidence_type_to_term(lhs);
                let rhs_term = self.confidence_type_to_term(rhs);
                SmtFormula::Geq(lhs_term, rhs_term)
            }
            ConfidencePredicate::Leq(lhs, rhs) => {
                let lhs_term = self.confidence_type_to_term(lhs);
                let rhs_term = self.confidence_type_to_term(rhs);
                SmtFormula::Leq(lhs_term, rhs_term)
            }
            ConfidencePredicate::Eq(lhs, rhs) => {
                let lhs_term = self.confidence_type_to_term(lhs);
                let rhs_term = self.confidence_type_to_term(rhs);
                SmtFormula::Eq(lhs_term, rhs_term)
            }
            ConfidencePredicate::Gt(lhs, rhs) => {
                let lhs_term = self.confidence_type_to_term(lhs);
                let rhs_term = self.confidence_type_to_term(rhs);
                SmtFormula::Gt(lhs_term, rhs_term)
            }
            ConfidencePredicate::Lt(lhs, rhs) => {
                let lhs_term = self.confidence_type_to_term(lhs);
                let rhs_term = self.confidence_type_to_term(rhs);
                SmtFormula::Lt(lhs_term, rhs_term)
            }
        }
    }

    /// Translate ConfidenceType to SmtTerm
    fn confidence_type_to_term(&self, conf_ty: &ConfidenceType) -> SmtTerm {
        match conf_ty {
            ConfidenceType::Literal(v) => SmtTerm::real(*v),
            ConfidenceType::Var(name) => SmtTerm::var(name),
            ConfidenceType::Product(a, b) => {
                let a_term = self.confidence_type_to_term(a);
                let b_term = self.confidence_type_to_term(b);
                SmtTerm::mul(a_term, b_term)
            }
            ConfidenceType::DempsterShafer(a, b) => {
                // DS(a, b) = 1 - (1-a)*(1-b)
                let a_term = self.confidence_type_to_term(a);
                let b_term = self.confidence_type_to_term(b);
                let one = SmtTerm::real(1.0);
                let one_minus_a = SmtTerm::sub(one.clone(), a_term);
                let one_minus_b = SmtTerm::sub(one.clone(), b_term);
                SmtTerm::sub(one, SmtTerm::mul(one_minus_a, one_minus_b))
            }
            ConfidenceType::Decay { base, lambda, elapsed } => {
                // base * exp(-lambda * t)
                let base_term = self.confidence_type_to_term(base);
                let t = SmtTerm::real(elapsed.as_secs_f64());
                let neg_lambda_t = SmtTerm::mul(SmtTerm::real(-lambda), t);
                let exp_term = SmtTerm::exp(neg_lambda_t);
                SmtTerm::mul(base_term, exp_term)
            }
            ConfidenceType::Minimum(a, b) => {
                let a_term = self.confidence_type_to_term(a);
                let b_term = self.confidence_type_to_term(b);
                SmtTerm::ite(
                    SmtFormula::Leq(a_term.clone(), b_term.clone()),
                    a_term,
                    b_term
                )
            }
            ConfidenceType::Maximum(a, b) => {
                let a_term = self.confidence_type_to_term(a);
                let b_term = self.confidence_type_to_term(b);
                SmtTerm::ite(
                    SmtFormula::Geq(a_term.clone(), b_term.clone()),
                    a_term,
                    b_term
                )
            }
        }
    }
}
```

**Modify `confidence_decision` to call Z3:**
```rust
fn confidence_decision(&self, pred: &ConfidencePredicate) -> ProofResult {
    match pred {
        ConfidencePredicate::Geq(lhs, rhs) => {
            let result = self.confidence_geq(lhs, rhs);

            // If built-in decision procedure fails, try Z3
            if matches!(result, ProofResult::Unknown { .. }) {
                return self.try_z3_epistemic(pred);
            }
            result
        }
        // ... similar for other cases
    }
}
```

### Phase 2: Integration Tests (0.5 days)

**File:** `crates/souc/tests/smt_epistemic_integration.rs`

```rust
#[cfg(feature = "smt")]
#[test]
fn test_epistemic_refinement_simple() {
    // fn extract_safe<T>(k: Knowledge<T> where k.confidence >= 0.95) -> T

    let mut ctx = TypeContext::new();
    ctx.bind_confidence("k_conf", ConfidenceType::literal(0.97));

    let mut searcher = ProofSearcher::new(&ctx);

    let pred = Predicate::confidence_geq(
        ConfidenceType::var("k_conf"),
        ConfidenceType::literal(0.95)
    );

    let result = searcher.search(&pred);
    assert!(result.is_proven());
}

#[cfg(feature = "smt")]
#[test]
fn test_epistemic_refinement_z3_needed() {
    // Complex predicate requiring Z3
    // sqrt(a^2 + b^2) >= c

    let mut ctx = TypeContext::new();
    ctx.bind_confidence("a", ConfidenceType::literal(0.03));
    ctx.bind_confidence("b", ConfidenceType::literal(0.04));
    ctx.bind_confidence("c", ConfidenceType::literal(0.05));

    // RSS: sqrt(0.03^2 + 0.04^2) = sqrt(0.0009 + 0.0016) = sqrt(0.0025) = 0.05
    // Should be provable by Z3

    let mut searcher = ProofSearcher::new(&ctx);

    let rss = ConfidenceType::rss(
        ConfidenceType::var("a"),
        ConfidenceType::var("b")
    );

    let pred = Predicate::confidence_geq(rss, ConfidenceType::var("c"));

    let result = searcher.search(&pred);
    assert!(result.is_proven());
}

#[cfg(feature = "smt")]
#[test]
fn test_epistemic_refinement_counterexample() {
    // Confidence too low - should get counterexample

    let mut ctx = TypeContext::new();
    ctx.bind_confidence("ε", ConfidenceType::literal(0.80));

    let mut searcher = ProofSearcher::new(&ctx);

    let pred = Predicate::confidence_geq(
        ConfidenceType::var("ε"),
        ConfidenceType::literal(0.95)
    );

    let result = searcher.search(&pred);
    assert!(result.is_disproven());

    match result {
        ProofResult::Disproven { reason } => {
            assert!(reason.contains("0.80") || reason.contains("< 0.95"));
        }
        _ => panic!("Expected disproven with counterexample"),
    }
}
```

### Phase 3: Example Sounio Code (0.5 days)

**File:** `examples/epistemic_refinements.sio`

```sio
// Epistemic refinement types with SMT-verified bounds

// Type alias: only high-confidence measurements
type Reliable<T> = Knowledge<T> where confidence >= 0.95

// Function that requires proven reliability
fn extract_safe<T>(k: Reliable<T>) -> T {
    // Type system proves confidence >= 0.95, so extract is safe
    return k.extract()
}

fn main() -> i32 with IO {
    // High confidence measurement
    let high_conf: Knowledge<f64> = measure(10.0, uncertainty: 0.2, confidence: 0.97)

    // Type checks: 0.97 >= 0.95 (proven by Z3)
    let value = extract_safe(high_conf)

    // Low confidence measurement
    let low_conf: Knowledge<f64> = measure(20.0, uncertainty: 5.0, confidence: 0.70)

    // TYPE ERROR: Cannot prove 0.70 >= 0.95
    // let bad = extract_safe(low_conf)  // Compile error

    return 0
}
```

**File:** `examples/epistemic_propagation.sio`

```sio
// RSS uncertainty propagation with SMT verification

fn add_measurements(
    a: Knowledge<f64>,
    b: Knowledge<f64>
) -> Knowledge<f64> where result.uncertainty <= sqrt(a.uncertainty^2 + b.uncertainty^2) {
    // Type system proves RSS propagation is correct
    return a + b
}

fn main() -> i32 with IO {
    let x: Knowledge<f64> = measure(10.0, uncertainty: 0.3)
    let y: Knowledge<f64> = measure(20.0, uncertainty: 0.4)

    // RSS: sqrt(0.3^2 + 0.4^2) = sqrt(0.09 + 0.16) = 0.5
    let sum = add_measurements(x, y)

    // Type system proves: sum.uncertainty <= 0.5

    return 0
}
```

### Phase 4: Type Checker Integration (1 day)

**File:** `crates/souc/src/check/refinement.rs`

Add epistemic refinement checking to the type checker:

```rust
impl<'a> TypeChecker<'a> {
    /// Check epistemic refinement in function signature
    pub fn check_epistemic_refinement(
        &mut self,
        actual: &KnowledgeType,
        required: &EpistemicRefinement
    ) -> Result<(), TypeError> {
        // Build predicate from refinement
        let pred = required.to_predicate(actual);

        // Try to prove it
        let mut searcher = ProofSearcher::new(&self.type_context);
        let result = searcher.search(&pred);

        match result {
            ProofResult::Proven(proof) => {
                // Success - refinement satisfied
                self.record_proof(proof);
                Ok(())
            }
            ProofResult::Disproven { reason } => {
                Err(TypeError::RefinementViolation {
                    expected: required.clone(),
                    actual: actual.clone(),
                    reason,
                })
            }
            ProofResult::Unknown { reason } => {
                // Gradual typing fallback
                if self.config.allow_gradual {
                    self.emit_warning(format!(
                        "Could not prove refinement, inserting runtime check: {}",
                        reason
                    ));
                    Ok(())
                } else {
                    Err(TypeError::RefinementUnprovable {
                        refinement: required.clone(),
                        reason,
                    })
                }
            }
        }
    }
}
```

---

## Testing Strategy

### Unit Tests
- [x] Z3 solver basics (already in z3_solver.rs)
- [x] Proof search decision procedures (already in proof_search.rs)
- [ ] Epistemic predicate translation to SMT
- [ ] Z3 fallback in proof search

### Integration Tests
- [ ] Simple epistemic refinement (confidence >= threshold)
- [ ] Complex refinement requiring Z3 (RSS, products, decay)
- [ ] Counterexample extraction
- [ ] Gradual typing fallback

### End-to-End Tests
- [ ] Compile examples/epistemic_refinements.sio
- [ ] Type error on refinement violation
- [ ] SMT timeout handling
- [ ] Performance (< 100ms per refinement check)

---

## Performance Targets

- **Simple refinements** (literal comparisons): < 1ms (no Z3)
- **Medium refinements** (products, DS): < 10ms (no Z3)
- **Complex refinements** (requiring Z3): < 100ms
- **Z3 timeout**: 1 second per query
- **Overall impact**: < 5% compile time increase

---

## Success Criteria

1. ✅ Refinement types compile: `Knowledge<T> where confidence >= 0.95`
2. ✅ Type errors on violation with counterexamples
3. ✅ Z3 proves complex epistemic predicates
4. ✅ Gradual typing fallback works
5. ✅ Performance acceptable (< 100ms per check)
6. ✅ Examples in `examples/` directory work
7. ✅ Tests pass with `--features smt`

---

## Estimated Timeline

- **Day 1**: Implement SMT translation (Phase 1)
- **Day 2**: Integration tests (Phase 2) + Examples (Phase 3)
- **Day 3**: Type checker integration (Phase 4) + E2E testing

**Total: 3 days** → Completes Month 3-4 milestone ✅

---

## Files to Modify

1. `crates/souc/src/dependent/proof_search.rs` (+150 LOC)
   - Add `try_z3_epistemic`
   - Add `translate_confidence_to_smt`
   - Add `confidence_type_to_term`
   - Modify `confidence_decision` to call Z3

2. `crates/souc/src/check/refinement.rs` (+100 LOC)
   - Add `check_epistemic_refinement`
   - Wire into type checker

3. `crates/souc/tests/smt_epistemic_integration.rs` (NEW, ~200 LOC)
   - Integration tests

4. `examples/epistemic_refinements.sio` (NEW, ~50 LOC)
   - Usage examples

5. `examples/epistemic_propagation.sio` (NEW, ~30 LOC)
   - RSS propagation example

**Total new code: ~530 LOC**

---

## Dependencies

- ✅ Z3 solver integration (already complete)
- ✅ Proof search infrastructure (already complete)
- ✅ Epistemic types (already complete)
- ⏳ SMT formula translation (needs implementation)
- ⏳ Type checker integration (needs implementation)

---

## Next Steps

1. Implement `translate_confidence_to_smt` method
2. Wire Z3 fallback into `confidence_decision`
3. Add integration tests
4. Create example Sounio programs
5. Document in paper formalization

**Ready to proceed!** All infrastructure is in place, just need the final wiring.
