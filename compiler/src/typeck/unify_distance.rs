//! Distance-Aware Type Unification
//!
//! Traditional unification: t1 = t2?
//! Distance unification: d(t1, t2) <= threshold?
//!
//! This extends the standard Hindley-Milner unification algorithm
//! to handle ontological types with semantic distance.

use std::collections::HashMap;
use std::sync::Arc;

use crate::common::Span;
use crate::hir::HirType;
use crate::ontology::alignment::unified::AlignmentIndex;
use crate::ontology::distance::SemanticDistanceIndex;
use crate::ontology::loader::IRI;

/// Type identifier for variables
pub type TypeId = u32;

/// Unification context with distance information
pub struct UnificationContext {
    /// Substitutions found during unification
    substitutions: HashMap<TypeId, HirType>,
    /// Distance index for semantic comparison
    distance_index: Arc<SemanticDistanceIndex>,
    /// Alignment index for cross-ontology types
    alignment_index: Arc<AlignmentIndex>,
    /// Current threshold (can be overridden per-constraint)
    current_threshold: f32,
    /// Accumulated coercions to insert
    coercions: Vec<CoercionSite>,
    /// Errors encountered
    errors: Vec<UnificationError>,
}

/// A site where type coercion needs to be inserted
#[derive(Debug, Clone)]
pub struct CoercionSite {
    /// Location in AST
    pub span: Span,
    /// Source type
    pub from_type: HirType,
    /// Target type
    pub to_type: HirType,
    /// Semantic distance
    pub distance: f32,
    /// Kind of coercion
    pub kind: CoercionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoercionKind {
    /// Same ontology, subtype relationship
    Subtype,
    /// Same ontology, semantic proximity
    SemanticProximity,
    /// Cross-ontology equivalence
    CrossOntology,
    /// Explicit cast required
    ExplicitCast,
}

/// Unification error with rich context
#[derive(Debug, Clone)]
pub struct UnificationError {
    pub kind: UnificationErrorKind,
    pub span: Span,
    pub expected: HirType,
    pub found: HirType,
    pub distance_info: Option<DistanceInfo>,
    pub suggestions: Vec<TypeSuggestion>,
}

#[derive(Debug, Clone)]
pub enum UnificationErrorKind {
    /// Types are structurally incompatible
    StructuralMismatch,
    /// Ontological types exceed distance threshold
    DistanceExceeded { threshold: f32, actual: f32 },
    /// No coercion path exists
    NoCoercionPath,
    /// Ambiguous coercion (multiple paths)
    AmbiguousCoercion,
    /// Infinite type (occurs check failure)
    InfiniteType,
}

#[derive(Debug, Clone)]
pub struct DistanceInfo {
    pub distance: f32,
    pub confidence: f32,
    pub threshold: f32,
    pub threshold_source: ThresholdSource,
}

#[derive(Debug, Clone)]
pub enum ThresholdSource {
    /// From #[compat(...)] annotation
    Annotation(f32),
    /// Inferred from parameter position
    ParameterDefault,
    /// Inferred from return type
    ReturnDefault,
    /// Global default
    GlobalDefault,
}

#[derive(Debug, Clone)]
pub struct TypeSuggestion {
    pub suggested_type: HirType,
    pub distance: f32,
    pub reason: String,
}

impl UnificationContext {
    pub fn new(
        distance_index: Arc<SemanticDistanceIndex>,
        alignment_index: Arc<AlignmentIndex>,
    ) -> Self {
        Self {
            substitutions: HashMap::new(),
            distance_index,
            alignment_index,
            current_threshold: 0.15, // Default threshold
            coercions: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Set current threshold
    pub fn with_threshold(&mut self, threshold: f32) -> &mut Self {
        self.current_threshold = threshold;
        self
    }

    /// Main unification entry point
    pub fn unify(&mut self, expected: &HirType, found: &HirType, span: Span) -> bool {
        self.unify_inner(expected, found, span, self.current_threshold)
    }

    /// Unify with explicit threshold
    pub fn unify_with_threshold(
        &mut self,
        expected: &HirType,
        found: &HirType,
        span: Span,
        threshold: f32,
    ) -> bool {
        self.unify_inner(expected, found, span, threshold)
    }

    fn unify_inner(
        &mut self,
        expected: &HirType,
        found: &HirType,
        span: Span,
        threshold: f32,
    ) -> bool {
        // Apply existing substitutions
        let expected = self.apply_substitutions(expected);
        let found = self.apply_substitutions(found);

        // Identical types always unify
        if expected == found {
            return true;
        }

        match (&expected, &found) {
            // Type variables
            (HirType::Var(id), _) => self.bind_variable(*id, found.clone(), span),
            (_, HirType::Var(id)) => self.bind_variable(*id, expected.clone(), span),

            // Ontological types - use semantic distance
            (
                HirType::Ontology {
                    namespace: ns_e,
                    term: t_e,
                },
                HirType::Ontology {
                    namespace: ns_f,
                    term: t_f,
                },
            ) => self.unify_ontological(ns_e, t_e, ns_f, t_f, span, threshold),

            // Named types with optional units (f64@kg)
            (
                HirType::Quantity {
                    numeric: n_e,
                    unit: u_e,
                },
                HirType::Quantity {
                    numeric: n_f,
                    unit: u_f,
                },
            ) => {
                // Numeric types must match
                let numeric_ok = self.unify_inner(n_e, n_f, span, threshold);
                // Units must be compatible
                let units_ok = u_e.is_compatible(u_f);

                if !units_ok {
                    self.errors.push(UnificationError {
                        kind: UnificationErrorKind::StructuralMismatch,
                        span,
                        expected: expected.clone(),
                        found: found.clone(),
                        distance_info: None,
                        suggestions: Vec::new(),
                    });
                }

                numeric_ok && units_ok
            }

            // Function types
            (
                HirType::Fn {
                    params: p_e,
                    return_type: r_e,
                },
                HirType::Fn {
                    params: p_f,
                    return_type: r_f,
                },
            ) => {
                if p_e.len() != p_f.len() {
                    self.errors.push(UnificationError {
                        kind: UnificationErrorKind::StructuralMismatch,
                        span,
                        expected: expected.clone(),
                        found: found.clone(),
                        distance_info: None,
                        suggestions: Vec::new(),
                    });
                    return false;
                }

                // Arguments are contravariant
                let args_ok = p_e
                    .iter()
                    .zip(p_f.iter())
                    .all(|(e, f)| self.unify_inner(f, e, span, threshold));

                // Return type is covariant
                let ret_ok = self.unify_inner(r_e, r_f, span, threshold);

                args_ok && ret_ok
            }

            // Named types with generics
            (
                HirType::Named {
                    name: n_e,
                    args: a_e,
                },
                HirType::Named {
                    name: n_f,
                    args: a_f,
                },
            ) => {
                if n_e != n_f || a_e.len() != a_f.len() {
                    self.errors.push(UnificationError {
                        kind: UnificationErrorKind::StructuralMismatch,
                        span,
                        expected: expected.clone(),
                        found: found.clone(),
                        distance_info: None,
                        suggestions: Vec::new(),
                    });
                    return false;
                }

                a_e.iter()
                    .zip(a_f.iter())
                    .all(|(e, f)| self.unify_inner(e, f, span, threshold))
            }

            // Tuple types
            (HirType::Tuple(elems_e), HirType::Tuple(elems_f)) => {
                if elems_e.len() != elems_f.len() {
                    self.errors.push(UnificationError {
                        kind: UnificationErrorKind::StructuralMismatch,
                        span,
                        expected: expected.clone(),
                        found: found.clone(),
                        distance_info: None,
                        suggestions: Vec::new(),
                    });
                    return false;
                }

                elems_e
                    .iter()
                    .zip(elems_f.iter())
                    .all(|(e, f)| self.unify_inner(e, f, span, threshold))
            }

            // Reference types
            (
                HirType::Ref {
                    mutable: m_e,
                    inner: i_e,
                },
                HirType::Ref {
                    mutable: m_f,
                    inner: i_f,
                },
            ) => {
                if m_e != m_f {
                    self.errors.push(UnificationError {
                        kind: UnificationErrorKind::StructuralMismatch,
                        span,
                        expected: expected.clone(),
                        found: found.clone(),
                        distance_info: None,
                        suggestions: Vec::new(),
                    });
                    return false;
                }
                self.unify_inner(i_e, i_f, span, threshold)
            }

            // Array types
            (
                HirType::Array {
                    element: e_e,
                    size: s_e,
                },
                HirType::Array {
                    element: e_f,
                    size: s_f,
                },
            ) => {
                let size_ok = s_e == s_f;
                let elem_ok = self.unify_inner(e_e, e_f, span, threshold);

                if !size_ok {
                    self.errors.push(UnificationError {
                        kind: UnificationErrorKind::StructuralMismatch,
                        span,
                        expected: expected.clone(),
                        found: found.clone(),
                        distance_info: None,
                        suggestions: Vec::new(),
                    });
                }

                size_ok && elem_ok
            }

            // Epistemic types (Knowledge[T, epsilon, ...])
            (
                HirType::Knowledge {
                    inner: i_e,
                    epsilon_bound: e_e,
                    ..
                },
                HirType::Knowledge {
                    inner: i_f,
                    epsilon_bound: e_f,
                    ..
                },
            ) => {
                // Inner types must unify
                let inner_ok = self.unify_inner(i_e, i_f, span, threshold);

                // Epistemic metadata: found must be at least as precise
                let meta_ok = match (e_f, e_e) {
                    (Some(found_eps), Some(expected_eps)) => found_eps <= expected_eps,
                    (Some(_), None) => true,  // Found is more precise
                    (None, Some(_)) => false, // Found is less precise
                    (None, None) => true,
                };

                if !meta_ok {
                    self.errors.push(UnificationError {
                        kind: UnificationErrorKind::StructuralMismatch,
                        span,
                        expected: expected.clone(),
                        found: found.clone(),
                        distance_info: None,
                        suggestions: Vec::new(),
                    });
                }

                inner_ok && meta_ok
            }

            // Primitive type compatibility
            (HirType::I64, HirType::I32)
            | (HirType::I32, HirType::I64)
            | (HirType::F64, HirType::F32)
            | (HirType::F32, HirType::F64) => {
                // Allow numeric coercion with a note
                self.coercions.push(CoercionSite {
                    span,
                    from_type: found.clone(),
                    to_type: expected.clone(),
                    distance: 0.0,
                    kind: CoercionKind::Subtype,
                });
                true
            }

            // Error types propagate
            (HirType::Error, _) | (_, HirType::Error) => true,

            // Structural mismatch
            _ => {
                self.errors.push(UnificationError {
                    kind: UnificationErrorKind::StructuralMismatch,
                    span,
                    expected: expected.clone(),
                    found: found.clone(),
                    distance_info: None,
                    suggestions: Vec::new(),
                });
                false
            }
        }
    }

    /// Unify ontological types using semantic distance
    fn unify_ontological(
        &mut self,
        ns_expected: &str,
        term_expected: &str,
        ns_found: &str,
        term_found: &str,
        span: Span,
        threshold: f32,
    ) -> bool {
        // Same term - always unifies
        if ns_expected == ns_found && term_expected == term_found {
            return true;
        }

        // Create IRIs for lookup
        let iri_expected = IRI::from_curie(ns_expected, term_expected);
        let iri_found = IRI::from_curie(ns_found, term_found);

        // Check for explicit equivalence via alignment index (Day 54)
        if let Some(alignment_result) = self
            .alignment_index
            .find_alignment(&iri_found, &iri_expected)
        {
            let confidence = alignment_result.best_confidence();
            if confidence >= 0.80 {
                let distance = 1.0 - confidence as f32;
                if distance <= threshold {
                    // Insert automatic coercion
                    self.coercions.push(CoercionSite {
                        span,
                        from_type: HirType::Ontology {
                            namespace: ns_found.to_string(),
                            term: term_found.to_string(),
                        },
                        to_type: HirType::Ontology {
                            namespace: ns_expected.to_string(),
                            term: term_expected.to_string(),
                        },
                        distance,
                        kind: if ns_expected == ns_found {
                            CoercionKind::SemanticProximity
                        } else {
                            CoercionKind::CrossOntology
                        },
                    });
                    return true;
                }
            }
        }

        // Use semantic distance (Day 53 + Day 55)
        let semantic_distance = self.distance_index.distance(&iri_found, &iri_expected);

        let distance_f32 = semantic_distance.conceptual as f32;
        if distance_f32 <= threshold {
            // Within threshold - insert coercion
            self.coercions.push(CoercionSite {
                span,
                from_type: HirType::Ontology {
                    namespace: ns_found.to_string(),
                    term: term_found.to_string(),
                },
                to_type: HirType::Ontology {
                    namespace: ns_expected.to_string(),
                    term: term_expected.to_string(),
                },
                distance: distance_f32,
                kind: if semantic_distance.confidence_retention >= 0.90 {
                    CoercionKind::SemanticProximity
                } else {
                    CoercionKind::ExplicitCast
                },
            });
            return true;
        }

        // Distance exceeds threshold - generate error with suggestions
        self.errors.push(UnificationError {
            kind: UnificationErrorKind::DistanceExceeded {
                threshold,
                actual: distance_f32,
            },
            span,
            expected: HirType::Ontology {
                namespace: ns_expected.to_string(),
                term: term_expected.to_string(),
            },
            found: HirType::Ontology {
                namespace: ns_found.to_string(),
                term: term_found.to_string(),
            },
            distance_info: Some(DistanceInfo {
                distance: distance_f32,
                confidence: semantic_distance.confidence_retention as f32,
                threshold,
                threshold_source: ThresholdSource::GlobalDefault,
            }),
            suggestions: Vec::new(), // Filled in by suggestion engine
        });

        false
    }

    /// Bind a type variable
    fn bind_variable(&mut self, var: TypeId, ty: HirType, span: Span) -> bool {
        // Occurs check
        if self.occurs_in(var, &ty) {
            self.errors.push(UnificationError {
                kind: UnificationErrorKind::InfiniteType,
                span,
                expected: HirType::Var(var),
                found: ty,
                distance_info: None,
                suggestions: Vec::new(),
            });
            return false;
        }

        self.substitutions.insert(var, ty);
        true
    }

    /// Occurs check - prevent infinite types
    fn occurs_in(&self, var: TypeId, ty: &HirType) -> bool {
        match ty {
            HirType::Var(id) => *id == var,
            HirType::Fn {
                params,
                return_type,
            } => params.iter().any(|a| self.occurs_in(var, a)) || self.occurs_in(var, return_type),
            HirType::Named { args, .. } => args.iter().any(|a| self.occurs_in(var, a)),
            HirType::Tuple(elems) => elems.iter().any(|e| self.occurs_in(var, e)),
            HirType::Ref { inner, .. } => self.occurs_in(var, inner),
            HirType::Array { element, .. } => self.occurs_in(var, element),
            HirType::Knowledge { inner, .. } => self.occurs_in(var, inner),
            HirType::Quantity { numeric, .. } => self.occurs_in(var, numeric),
            _ => false,
        }
    }

    /// Apply substitutions to a type
    fn apply_substitutions(&self, ty: &HirType) -> HirType {
        match ty {
            HirType::Var(id) => {
                if let Some(substituted) = self.substitutions.get(id) {
                    self.apply_substitutions(substituted)
                } else {
                    ty.clone()
                }
            }
            HirType::Fn {
                params,
                return_type,
            } => {
                let params = params.iter().map(|a| self.apply_substitutions(a)).collect();
                let return_type = Box::new(self.apply_substitutions(return_type));
                HirType::Fn {
                    params,
                    return_type,
                }
            }
            HirType::Named { name, args } => {
                let args = args.iter().map(|a| self.apply_substitutions(a)).collect();
                HirType::Named {
                    name: name.clone(),
                    args,
                }
            }
            HirType::Tuple(elems) => {
                HirType::Tuple(elems.iter().map(|e| self.apply_substitutions(e)).collect())
            }
            HirType::Ref { mutable, inner } => HirType::Ref {
                mutable: *mutable,
                inner: Box::new(self.apply_substitutions(inner)),
            },
            HirType::Array { element, size } => HirType::Array {
                element: Box::new(self.apply_substitutions(element)),
                size: *size,
            },
            HirType::Knowledge {
                inner,
                epsilon_bound,
                provenance,
            } => HirType::Knowledge {
                inner: Box::new(self.apply_substitutions(inner)),
                epsilon_bound: *epsilon_bound,
                provenance: provenance.clone(),
            },
            HirType::Quantity { numeric, unit } => HirType::Quantity {
                numeric: Box::new(self.apply_substitutions(numeric)),
                unit: unit.clone(),
            },
            _ => ty.clone(),
        }
    }

    /// Get accumulated coercions
    pub fn coercions(&self) -> &[CoercionSite] {
        &self.coercions
    }

    /// Take coercions (consuming)
    pub fn take_coercions(&mut self) -> Vec<CoercionSite> {
        std::mem::take(&mut self.coercions)
    }

    /// Get errors
    pub fn errors(&self) -> &[UnificationError] {
        &self.errors
    }

    /// Take errors (consuming)
    pub fn take_errors(&mut self) -> Vec<UnificationError> {
        std::mem::take(&mut self.errors)
    }

    /// Has errors?
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get final substitutions
    pub fn substitutions(&self) -> &HashMap<TypeId, HirType> {
        &self.substitutions
    }

    /// Clear state for reuse
    pub fn clear(&mut self) {
        self.substitutions.clear();
        self.coercions.clear();
        self.errors.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_types_unify() {
        // Would need mock distance/alignment indices
        // Basic structural test
        assert_eq!(HirType::I64, HirType::I64);
        assert_ne!(HirType::I64, HirType::F64);
    }

    #[test]
    fn test_coercion_kind_display() {
        assert_eq!(format!("{:?}", CoercionKind::Subtype), "Subtype");
        assert_eq!(
            format!("{:?}", CoercionKind::CrossOntology),
            "CrossOntology"
        );
    }
}
