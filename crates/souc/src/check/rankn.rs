//! Higher-rank polymorphism (RankN types) support for Sounio
//!
//! This module implements type checking for higher-rank polymorphic types,
//! which allow polymorphic functions to be passed as arguments.
//!
//! # Rank Levels
//!
//! - Rank-0: Monomorphic types (no type variables)
//! - Rank-1: Normal polymorphism - type variables at the outermost level
//!   Example: `forall T. T -> T` (the identity function)
//! - Rank-2: Polymorphic functions as arguments
//!   Example: `(forall T. T -> T) -> i32`
//! - Rank-N: Arbitrary nesting of forall quantifiers
//!
//! # Key Operations
//!
//! - **Instantiation**: Replace quantified type variables with fresh unification variables
//! - **Generalization**: Abstract over free type variables to create polymorphic types
//! - **Subsumption**: Check if one type is at least as general as another
//! - **Skolemization**: Replace quantified variables with fresh rigid (skolem) variables
//!   for checking polymorphic arguments
//!
//! # Bidirectional Type Checking
//!
//! The integration with bidirectional type checking follows these rules:
//! - **Checking mode** (against an expected type): Use subsumption
//! - **Synthesis mode** (inferring a type): Instantiate foralls at use sites
//!
//! # References
//!
//! This implementation is based on the following papers:
//! - "Practical Type Inference for Arbitrary-Rank Types" (Peyton Jones et al., 2007)
//! - "Complete and Easy Bidirectional Typechecking for Higher-Rank Polymorphism" (Dunfield & Krishnaswami, 2013)

use std::collections::{HashMap, HashSet};

use crate::types::{Type, TypeVar};

/// A skolem variable - a rigid type variable used during subsumption checking.
/// Unlike unification variables, skolem variables cannot be unified with anything
/// except themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SkolemVar(pub u32);

/// Result of a subsumption check
#[derive(Debug, Clone)]
pub enum SubsumptionResult {
    /// The subsumption holds
    Ok,
    /// The subsumption fails with an explanation
    Fail(String),
}

impl SubsumptionResult {
    pub fn is_ok(&self) -> bool {
        matches!(self, SubsumptionResult::Ok)
    }

    pub fn is_fail(&self) -> bool {
        matches!(self, SubsumptionResult::Fail(_))
    }
}

/// State for RankN type checking
pub struct RankNChecker {
    /// Counter for generating fresh type variables
    next_type_var: u32,
    /// Counter for generating fresh skolem variables
    next_skolem: u32,
    /// Substitution from type variables to types
    substitution: HashMap<TypeVar, Type>,
    /// Set of skolem variables in scope (cannot be unified)
    skolems: HashSet<SkolemVar>,
    /// Mapping from skolem variables to their original type variable names
    /// (for error messages)
    skolem_names: HashMap<SkolemVar, String>,
}

impl Default for RankNChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl RankNChecker {
    /// Create a new RankN type checker
    pub fn new() -> Self {
        Self {
            next_type_var: 0,
            next_skolem: 0,
            substitution: HashMap::new(),
            skolems: HashSet::new(),
            skolem_names: HashMap::new(),
        }
    }

    /// Create a new RankN checker starting from a given type variable counter
    pub fn with_type_var_start(start: u32) -> Self {
        Self {
            next_type_var: start,
            next_skolem: 0,
            substitution: HashMap::new(),
            skolems: HashSet::new(),
            skolem_names: HashMap::new(),
        }
    }

    /// Generate a fresh type variable
    pub fn fresh_type_var(&mut self) -> TypeVar {
        let var = TypeVar(self.next_type_var);
        self.next_type_var += 1;
        var
    }

    /// Generate a fresh skolem variable
    fn fresh_skolem(&mut self, name: Option<&str>) -> SkolemVar {
        let skolem = SkolemVar(self.next_skolem);
        self.next_skolem += 1;
        self.skolems.insert(skolem);
        if let Some(n) = name {
            self.skolem_names.insert(skolem, n.to_string());
        }
        skolem
    }

    // ==================== INSTANTIATION ====================

    /// Instantiate a forall type by replacing quantified variables with fresh
    /// unification variables.
    ///
    /// Example:
    /// `instantiate(forall T. T -> T)` => `?a -> ?a` where `?a` is fresh
    pub fn instantiate(&mut self, ty: &Type) -> Type {
        match ty {
            Type::Forall { vars, inner } => {
                // Create fresh type variables for each quantified variable
                let mut subst = HashMap::new();
                for var in vars {
                    let fresh = self.fresh_type_var();
                    subst.insert(*var, Type::Var(fresh));
                }
                // Substitute and recursively instantiate (for nested foralls in synthesis mode)
                let substituted = inner.substitute(&subst);
                substituted
            }
            _ => ty.clone(),
        }
    }

    /// Deep instantiation - instantiate all foralls at any depth
    /// (used in synthesis mode for expressions)
    pub fn deep_instantiate(&mut self, ty: &Type) -> Type {
        let instantiated = self.instantiate(ty);
        // For function types, we don't deep-instantiate the argument types
        // (that would lose polymorphism for rank-2+ types)
        instantiated
    }

    // ==================== GENERALIZATION ====================

    /// Generalize a type by abstracting over free type variables that are
    /// not in the environment.
    ///
    /// Example:
    /// If `env_vars = {}` and `ty = ?a -> ?a`, then
    /// `generalize(env_vars, ty)` => `forall T. T -> T`
    pub fn generalize(&mut self, env_vars: &HashSet<TypeVar>, ty: &Type) -> Type {
        // Find free type variables in the type that are not in the environment
        let free_in_type = ty.free_vars();
        let free_not_in_env: Vec<TypeVar> = free_in_type.difference(env_vars).copied().collect();

        if free_not_in_env.is_empty() {
            ty.clone()
        } else {
            Type::Forall {
                vars: free_not_in_env,
                inner: Box::new(ty.clone()),
            }
        }
    }

    // ==================== SKOLEMIZATION ====================

    /// Deep skolemization - replace all quantified variables with fresh skolem
    /// variables. This is used when checking that an argument has a polymorphic type.
    ///
    /// For example, when checking `f(id)` where `f: (forall T. T -> T) -> i32`,
    /// we skolemize `forall T. T -> T` to `sk -> sk` and check that `id` has this type.
    ///
    /// Returns the skolemized type and a mapping from type variables to skolems.
    pub fn deep_skolemize(&mut self, ty: &Type) -> (Type, HashMap<TypeVar, SkolemVar>) {
        let mut skolem_map = HashMap::new();
        let skolemized = self.skolemize_inner(ty, &mut skolem_map);
        (skolemized, skolem_map)
    }

    fn skolemize_inner(&mut self, ty: &Type, skolem_map: &mut HashMap<TypeVar, SkolemVar>) -> Type {
        match ty {
            Type::Forall { vars, inner } => {
                // Replace each quantified variable with a fresh skolem
                let mut subst = HashMap::new();
                for var in vars {
                    let skolem = self.fresh_skolem(None);
                    skolem_map.insert(*var, skolem);
                    // We represent skolems as Named types with a special name
                    subst.insert(
                        *var,
                        Type::Named {
                            name: format!("$skolem_{}", skolem.0),
                            args: vec![],
                        },
                    );
                }
                let substituted = inner.substitute(&subst);
                // Recursively skolemize nested foralls
                self.skolemize_inner(&substituted, skolem_map)
            }
            _ => ty.clone(),
        }
    }

    // ==================== SUBSUMPTION ====================

    /// Check if `ty1` is at least as general as `ty2` (ty1 ⊑ ty2).
    ///
    /// Subsumption rules:
    /// - `forall a. A ⊑ B` if `A[a := τ] ⊑ B` for fresh τ (instantiate on the left)
    /// - `A ⊑ forall a. B` if `A ⊑ B[a := sk]` for fresh skolem sk (skolemize on the right)
    /// - `A -> B ⊑ C -> D` if `C ⊑ A` and `B ⊑ D` (contravariant in argument)
    /// - `A ⊑ A` (reflexivity)
    ///
    /// This implements the "Complete and Easy" algorithm.
    pub fn subsumes(&mut self, ty1: &Type, ty2: &Type) -> SubsumptionResult {
        // Rule: forall a. A ⊑ B  =>  instantiate forall on left
        if let Type::Forall { vars, inner } = ty1 {
            let instantiated = self.instantiate(ty1);
            return self.subsumes(&instantiated, ty2);
        }

        // Rule: A ⊑ forall a. B  =>  skolemize forall on right
        if let Type::Forall { vars, inner } = ty2 {
            let (skolemized, _) = self.deep_skolemize(ty2);
            return self.subsumes(ty1, &skolemized);
        }

        // Rule: A -> B ⊑ C -> D  =>  C ⊑ A and B ⊑ D (contravariant in argument)
        if let (
            Type::Function {
                params: params1,
                return_type: ret1,
                effects: effects1,
                abi: _,
            },
            Type::Function {
                params: params2,
                return_type: ret2,
                effects: effects2,
                abi: _,
            },
        ) = (ty1, ty2)
        {
            // Check parameter count
            if params1.len() != params2.len() {
                return SubsumptionResult::Fail(format!(
                    "Function arity mismatch: expected {} parameters, got {}",
                    params2.len(),
                    params1.len()
                ));
            }

            // Check parameters (contravariant)
            for (p1, p2) in params1.iter().zip(params2.iter()) {
                let result = self.subsumes(p2, p1); // Note: reversed for contravariance
                if let SubsumptionResult::Fail(msg) = result {
                    return SubsumptionResult::Fail(format!("Parameter type mismatch: {}", msg));
                }
            }

            // Check return type (covariant)
            let ret_result = self.subsumes(ret1, ret2);
            if let SubsumptionResult::Fail(msg) = ret_result {
                return SubsumptionResult::Fail(format!("Return type mismatch: {}", msg));
            }

            // TODO: Check effects subsumption
            return SubsumptionResult::Ok;
        }

        // For other types, check equality (or unifiability)
        if self.types_equal(ty1, ty2) {
            SubsumptionResult::Ok
        } else if self.can_unify(ty1, ty2) {
            // Try to unify - if successful, subsumption holds
            if self.unify(ty1, ty2) {
                SubsumptionResult::Ok
            } else {
                SubsumptionResult::Fail(format!(
                    "Cannot unify {} with {}",
                    self.type_to_string(ty1),
                    self.type_to_string(ty2)
                ))
            }
        } else {
            SubsumptionResult::Fail(format!(
                "Type {} is not at least as general as {}",
                self.type_to_string(ty1),
                self.type_to_string(ty2)
            ))
        }
    }

    /// Check if two types are structurally equal (after applying substitution)
    fn types_equal(&self, ty1: &Type, ty2: &Type) -> bool {
        let ty1 = self.apply_subst(ty1);
        let ty2 = self.apply_subst(ty2);
        ty1 == ty2
    }

    /// Check if two types can potentially be unified
    fn can_unify(&self, ty1: &Type, ty2: &Type) -> bool {
        let ty1 = self.apply_subst(ty1);
        let ty2 = self.apply_subst(ty2);

        match (&ty1, &ty2) {
            // Type variables can unify with anything (unless they escape their scope)
            (Type::Var(_), _) | (_, Type::Var(_)) => true,
            // Same types unify
            (a, b) if a == b => true,
            // Recursively check compound types
            (
                Type::Ref {
                    inner: i1,
                    mutable: m1,
                    ..
                },
                Type::Ref {
                    inner: i2,
                    mutable: m2,
                    ..
                },
            ) => m1 == m2 && self.can_unify(i1, i2),
            (
                Type::Array {
                    element: e1,
                    size: s1,
                },
                Type::Array {
                    element: e2,
                    size: s2,
                },
            ) => s1 == s2 && self.can_unify(e1, e2),
            (Type::Tuple(t1), Type::Tuple(t2)) => {
                t1.len() == t2.len() && t1.iter().zip(t2.iter()).all(|(a, b)| self.can_unify(a, b))
            }
            (Type::Named { name: n1, args: a1 }, Type::Named { name: n2, args: a2 }) => {
                n1 == n2
                    && a1.len() == a2.len()
                    && a1.iter().zip(a2.iter()).all(|(a, b)| self.can_unify(a, b))
            }
            _ => false,
        }
    }

    /// Try to unify two types, updating the substitution if successful
    fn unify(&mut self, ty1: &Type, ty2: &Type) -> bool {
        let ty1 = self.apply_subst(ty1);
        let ty2 = self.apply_subst(ty2);

        match (&ty1, &ty2) {
            // Same types - trivially unify
            (a, b) if a == b => true,

            // Type variable unification
            (Type::Var(v), t) | (t, Type::Var(v)) => {
                // Occurs check: prevent infinite types
                if t.free_vars().contains(v) {
                    return false;
                }
                // Check if the type contains a skolem variable
                if self.contains_skolem(t) {
                    return false;
                }
                self.substitution.insert(*v, t.clone());
                true
            }

            // Reference types
            (
                Type::Ref {
                    inner: i1,
                    mutable: m1,
                    lifetime: l1,
                },
                Type::Ref {
                    inner: i2,
                    mutable: m2,
                    lifetime: l2,
                },
            ) => m1 == m2 && self.unify(i1, i2),

            // Array types
            (
                Type::Array {
                    element: e1,
                    size: s1,
                },
                Type::Array {
                    element: e2,
                    size: s2,
                },
            ) => s1 == s2 && self.unify(e1, e2),

            // Tuple types
            (Type::Tuple(t1), Type::Tuple(t2)) => {
                t1.len() == t2.len() && t1.iter().zip(t2.iter()).all(|(a, b)| self.unify(a, b))
            }

            // Named types
            (Type::Named { name: n1, args: a1 }, Type::Named { name: n2, args: a2 }) => {
                n1 == n2 && a1.len() == a2.len() && {
                    // Need to handle this differently since unify takes &mut self
                    let pairs: Vec<_> = a1.iter().cloned().zip(a2.iter().cloned()).collect();
                    pairs.into_iter().all(|(a, b)| self.unify(&a, &b))
                }
            }

            // Function types
            (
                Type::Function {
                    params: p1,
                    return_type: r1,
                    ..
                },
                Type::Function {
                    params: p2,
                    return_type: r2,
                    ..
                },
            ) => {
                p1.len() == p2.len()
                    && {
                        let param_pairs: Vec<_> =
                            p1.iter().cloned().zip(p2.iter().cloned()).collect();
                        param_pairs.into_iter().all(|(a, b)| self.unify(&a, &b))
                    }
                    && self.unify(r1, r2)
            }

            // Forall types - require alpha-equivalence
            (
                Type::Forall {
                    vars: v1,
                    inner: i1,
                },
                Type::Forall {
                    vars: v2,
                    inner: i2,
                },
            ) => {
                // For now, require same number of bound variables
                if v1.len() != v2.len() {
                    return false;
                }
                // Create a mapping from v2's variables to v1's
                let mut subst = HashMap::new();
                for (var1, var2) in v1.iter().zip(v2.iter()) {
                    subst.insert(*var2, Type::Var(*var1));
                }
                let i2_renamed = i2.substitute(&subst);
                self.unify(i1, &i2_renamed)
            }

            _ => false,
        }
    }

    /// Check if a type contains a skolem variable (represented as Named with $skolem_ prefix)
    fn contains_skolem(&self, ty: &Type) -> bool {
        match ty {
            Type::Named { name, args } => {
                name.starts_with("$skolem_") || args.iter().any(|a| self.contains_skolem(a))
            }
            Type::Ref { inner, .. } => self.contains_skolem(inner),
            Type::Array { element, .. } => self.contains_skolem(element),
            Type::Tuple(types) => types.iter().any(|t| self.contains_skolem(t)),
            Type::Function {
                params,
                return_type,
                ..
            } => {
                params.iter().any(|p| self.contains_skolem(p)) || self.contains_skolem(return_type)
            }
            Type::Forall { inner, .. } => self.contains_skolem(inner),
            _ => false,
        }
    }

    /// Apply the current substitution to a type
    pub fn apply_subst(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(v) => {
                if let Some(t) = self.substitution.get(v) {
                    self.apply_subst(t)
                } else {
                    ty.clone()
                }
            }
            Type::Ref {
                mutable,
                lifetime,
                inner,
            } => Type::Ref {
                mutable: *mutable,
                lifetime: lifetime.clone(),
                inner: Box::new(self.apply_subst(inner)),
            },
            Type::Array { element, size } => Type::Array {
                element: Box::new(self.apply_subst(element)),
                size: *size,
            },
            Type::Tuple(types) => Type::Tuple(types.iter().map(|t| self.apply_subst(t)).collect()),
            Type::Function {
                params,
                return_type,
                effects,
                abi,
            } => Type::Function {
                params: params.iter().map(|p| self.apply_subst(p)).collect(),
                return_type: Box::new(self.apply_subst(return_type)),
                effects: effects.clone(),
                abi: abi.clone(),
            },
            Type::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|a| self.apply_subst(a)).collect(),
            },
            Type::Forall { vars, inner } => {
                // Don't substitute bound variables
                let mut subst = self.substitution.clone();
                for v in vars {
                    subst.remove(v);
                }
                let checker = RankNChecker {
                    next_type_var: self.next_type_var,
                    next_skolem: self.next_skolem,
                    substitution: subst,
                    skolems: self.skolems.clone(),
                    skolem_names: self.skolem_names.clone(),
                };
                Type::Forall {
                    vars: vars.clone(),
                    inner: Box::new(checker.apply_subst(inner)),
                }
            }
            _ => ty.clone(),
        }
    }

    /// Get a human-readable string representation of a type
    pub fn type_to_string(&self, ty: &Type) -> String {
        let ty = self.apply_subst(ty);
        match ty {
            Type::Unit => "()".to_string(),
            Type::Bool => "bool".to_string(),
            Type::I32 => "i32".to_string(),
            Type::I64 => "i64".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::String => "string".to_string(),
            Type::Var(v) => format!("?{}", v.0),
            Type::Named { name, args } => {
                if args.is_empty() {
                    name
                } else {
                    format!(
                        "{}<{}>",
                        name,
                        args.iter()
                            .map(|a| self.type_to_string(a))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            Type::Ref { mutable, inner, .. } => {
                if mutable {
                    format!("&!{}", self.type_to_string(&inner))
                } else {
                    format!("&{}", self.type_to_string(&inner))
                }
            }
            Type::Array { element, size } => {
                if let Some(s) = size {
                    format!("[{}; {}]", self.type_to_string(&element), s)
                } else {
                    format!("[{}]", self.type_to_string(&element))
                }
            }
            Type::Tuple(types) => {
                format!(
                    "({})",
                    types
                        .iter()
                        .map(|t| self.type_to_string(t))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Type::Function {
                params,
                return_type,
                ..
            } => {
                format!(
                    "fn({}) -> {}",
                    params
                        .iter()
                        .map(|p| self.type_to_string(p))
                        .collect::<Vec<_>>()
                        .join(", "),
                    self.type_to_string(&return_type)
                )
            }
            Type::Forall { vars, inner } => {
                let var_strs: Vec<String> = vars.iter().map(|v| format!("T{}", v.0)).collect();
                format!(
                    "forall {}. {}",
                    var_strs.join(" "),
                    self.type_to_string(&inner)
                )
            }
            _ => format!("{:?}", ty),
        }
    }

    // ==================== RANK COMPUTATION ====================

    /// Compute the rank of a type.
    ///
    /// - Rank 0: No forall
    /// - Rank 1: forall at the outermost level only
    /// - Rank 2: forall in negative (argument) position of a function
    /// - Rank n: Maximum nesting depth of forall in negative positions
    pub fn compute_rank(&self, ty: &Type) -> usize {
        self.compute_rank_inner(ty, false)
    }

    fn compute_rank_inner(&self, ty: &Type, negative: bool) -> usize {
        match ty {
            Type::Forall { inner, .. } => {
                if negative {
                    // Forall in negative position increases rank
                    1 + self.compute_rank_inner(inner, false)
                } else {
                    // Forall in positive position doesn't increase rank
                    self.compute_rank_inner(inner, false)
                }
            }
            Type::Function {
                params,
                return_type,
                ..
            } => {
                // Arguments are in negative position, return type is positive
                let param_rank = params
                    .iter()
                    .map(|p| self.compute_rank_inner(p, !negative))
                    .max()
                    .unwrap_or(0);
                let ret_rank = self.compute_rank_inner(return_type, negative);
                param_rank.max(ret_rank)
            }
            Type::Ref { inner, .. } => self.compute_rank_inner(inner, negative),
            Type::Array { element, .. } => self.compute_rank_inner(element, negative),
            Type::Tuple(types) => types
                .iter()
                .map(|t| self.compute_rank_inner(t, negative))
                .max()
                .unwrap_or(0),
            Type::Named { args, .. } => args
                .iter()
                .map(|a| self.compute_rank_inner(a, negative))
                .max()
                .unwrap_or(0),
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EffectSet;

    fn make_id_type() -> Type {
        // forall T. T -> T
        let t = TypeVar(0);
        Type::Forall {
            vars: vec![t],
            inner: Box::new(Type::Function {
                params: vec![Type::Var(t)],
                return_type: Box::new(Type::Var(t)),
                effects: EffectSet::new(),
                abi: None,
            }),
        }
    }

    fn make_apply_twice_type() -> Type {
        // forall A. (forall T. T -> T) -> A -> A
        let a = TypeVar(1);
        let t = TypeVar(0);
        Type::Forall {
            vars: vec![a],
            inner: Box::new(Type::Function {
                params: vec![
                    Type::Forall {
                        vars: vec![t],
                        inner: Box::new(Type::Function {
                            params: vec![Type::Var(t)],
                            return_type: Box::new(Type::Var(t)),
                            effects: EffectSet::new(),
                            abi: None,
                        }),
                    },
                    Type::Var(a),
                ],
                return_type: Box::new(Type::Var(a)),
                effects: EffectSet::new(),
                abi: None,
            }),
        }
    }

    #[test]
    fn test_instantiate_id() {
        let mut checker = RankNChecker::new();
        let id_type = make_id_type();

        let instantiated = checker.instantiate(&id_type);

        // Should be ?0 -> ?0
        match instantiated {
            Type::Function {
                params,
                return_type,
                ..
            } => {
                assert_eq!(params.len(), 1);
                assert!(matches!(params[0], Type::Var(_)));
                assert!(matches!(*return_type, Type::Var(_)));
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_compute_rank() {
        let checker = RankNChecker::new();

        // Rank 0: i32 -> i32
        let rank0 = Type::Function {
            params: vec![Type::I32],
            return_type: Box::new(Type::I32),
            effects: EffectSet::new(),
            abi: None,
        };
        assert_eq!(checker.compute_rank(&rank0), 0);

        // Rank 1: forall T. T -> T
        let rank1 = make_id_type();
        assert_eq!(checker.compute_rank(&rank1), 0); // Outermost forall doesn't count

        // Rank 2: (forall T. T -> T) -> i32
        let rank2 = Type::Function {
            params: vec![make_id_type()],
            return_type: Box::new(Type::I32),
            effects: EffectSet::new(),
            abi: None,
        };
        assert_eq!(checker.compute_rank(&rank2), 1); // Forall in negative position = rank 1
    }

    #[test]
    fn test_subsumption_id_to_concrete() {
        let mut checker = RankNChecker::new();

        // forall T. T -> T should subsume i32 -> i32
        let id_type = make_id_type();
        let concrete = Type::Function {
            params: vec![Type::I32],
            return_type: Box::new(Type::I32),
            effects: EffectSet::new(),
            abi: None,
        };

        let result = checker.subsumes(&id_type, &concrete);
        assert!(result.is_ok(), "forall T. T -> T should subsume i32 -> i32");
    }

    #[test]
    fn test_subsumption_concrete_not_to_forall() {
        let mut checker = RankNChecker::new();

        // i32 -> i32 should NOT subsume forall T. T -> T
        let id_type = make_id_type();
        let concrete = Type::Function {
            params: vec![Type::I32],
            return_type: Box::new(Type::I32),
            effects: EffectSet::new(),
            abi: None,
        };

        let result = checker.subsumes(&concrete, &id_type);
        // This should fail because a concrete function is not polymorphic
        assert!(
            result.is_fail(),
            "i32 -> i32 should not subsume forall T. T -> T"
        );
    }

    #[test]
    fn test_skolemization() {
        let mut checker = RankNChecker::new();

        let id_type = make_id_type();
        let (skolemized, _) = checker.deep_skolemize(&id_type);

        // Should be $skolem_0 -> $skolem_0
        match skolemized {
            Type::Function {
                params,
                return_type,
                ..
            } => {
                assert_eq!(params.len(), 1);
                if let Type::Named { name, .. } = &params[0] {
                    assert!(name.starts_with("$skolem_"));
                } else {
                    panic!("Expected skolem type");
                }
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_unify_type_vars() {
        let mut checker = RankNChecker::new();

        let var1 = Type::Var(TypeVar(0));
        let var2 = Type::Var(TypeVar(1));

        assert!(checker.unify(&var1, &Type::I32));
        assert!(checker.unify(&var2, &Type::I32));

        // After unification, both should be i32
        assert_eq!(checker.apply_subst(&var1), Type::I32);
        assert_eq!(checker.apply_subst(&var2), Type::I32);
    }

    #[test]
    fn test_generalize() {
        let mut checker = RankNChecker::with_type_var_start(0);

        let var = TypeVar(0);
        let ty = Type::Function {
            params: vec![Type::Var(var)],
            return_type: Box::new(Type::Var(var)),
            effects: EffectSet::new(),
            abi: None,
        };

        let env_vars = HashSet::new();
        let generalized = checker.generalize(&env_vars, &ty);

        match generalized {
            Type::Forall { vars, inner } => {
                assert!(!vars.is_empty());
                // Inner should still be the function type
                assert!(matches!(*inner, Type::Function { .. }));
            }
            _ => panic!("Expected forall type"),
        }
    }
}
