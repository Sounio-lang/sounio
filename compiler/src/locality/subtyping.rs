//! Locality Subtyping: The subtype lattice for memory hierarchy.
//!
//! Key insight: Faster localities are subtypes of slower ones.
//! A value in L1 cache can be used where Local memory is expected,
//! but not vice versa. This is covariant subtyping based on speed.
//!
//! The lattice: Register <: L1 <: L2 <: L3 <: Local <: Remote <: Persistent <: Network

use super::types::{Locality, LocalityBound, LocalityConstraint, LocalityParam};
use std::collections::HashMap;

/// Result of a subtype check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubtypeResult {
    /// The subtype relation holds
    Subtype,
    /// The types are equal
    Equal,
    /// The relation is inverted (supertype)
    Supertype,
    /// No subtype relation exists
    Incompatible,
}

impl SubtypeResult {
    /// Check if this result indicates a valid subtype or equal relation.
    pub fn is_valid(&self) -> bool {
        matches!(self, SubtypeResult::Subtype | SubtypeResult::Equal)
    }

    /// Combine two results (for conjunctions).
    pub fn and(&self, other: &SubtypeResult) -> SubtypeResult {
        match (self, other) {
            (SubtypeResult::Equal, r) | (r, SubtypeResult::Equal) => r.clone(),
            (SubtypeResult::Subtype, SubtypeResult::Subtype) => SubtypeResult::Subtype,
            (SubtypeResult::Supertype, SubtypeResult::Supertype) => SubtypeResult::Supertype,
            _ => SubtypeResult::Incompatible,
        }
    }
}

/// The locality lattice for subtyping checks.
pub struct LocalityLattice {
    /// Cache of computed subtype relations
    cache: HashMap<(Locality, Locality), SubtypeResult>,
}

impl LocalityLattice {
    /// Create a new locality lattice.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Check the subtype relation between two localities.
    /// Returns Subtype if `sub` is a subtype of `sup`.
    pub fn check(&mut self, sub: Locality, sup: Locality) -> SubtypeResult {
        if let Some(result) = self.cache.get(&(sub, sup)) {
            return result.clone();
        }

        let result = if sub == sup {
            SubtypeResult::Equal
        } else if sub < sup {
            // Faster is subtype of slower
            SubtypeResult::Subtype
        } else {
            // Slower is supertype of faster
            SubtypeResult::Supertype
        };

        self.cache.insert((sub, sup), result.clone());
        result
    }

    /// Check if `sub` is a subtype of `sup`.
    pub fn is_subtype(&mut self, sub: Locality, sup: Locality) -> bool {
        self.check(sub, sup).is_valid()
    }

    /// Find the join (least upper bound) of two localities.
    /// The join is the slowest of the two.
    pub fn join(&self, a: Locality, b: Locality) -> Locality {
        std::cmp::max(a, b)
    }

    /// Find the meet (greatest lower bound) of two localities.
    /// The meet is the fastest of the two.
    pub fn meet(&self, a: Locality, b: Locality) -> Locality {
        std::cmp::min(a, b)
    }

    /// Check if a locality satisfies a bound.
    pub fn satisfies_bound(&self, locality: Locality, bound: &LocalityBound) -> bool {
        bound.satisfied_by(locality)
    }

    /// Check if bound `sub` is a subbound of `sup`.
    /// A bound is a subbound if it's more restrictive (narrower range toward fast).
    pub fn is_subbound(&self, sub: &LocalityBound, sup: &LocalityBound) -> bool {
        // sub is a subbound if its range is contained in sup's range
        // AND its fastest is at least as fast as sup's fastest
        sub.fastest <= sup.fastest && sub.slowest <= sup.slowest
    }

    /// Solve locality constraints, finding a valid substitution if one exists.
    pub fn solve_constraints(
        &mut self,
        params: &[LocalityParam],
        constraints: &[LocalityConstraint],
    ) -> Option<HashMap<String, Locality>> {
        let mut solution = HashMap::new();

        // Initialize with the fastest allowed by each param's bound
        for param in params {
            solution.insert(param.name.clone(), param.bound.fastest);
        }

        // Iteratively refine until fixed point or failure
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            for constraint in constraints {
                match constraint {
                    LocalityConstraint::FasterOrEqual(l1, l2) => {
                        // l1 <= l2, so if l1 is slower, we need to slow down l2
                        if let (Some(&loc1), Some(&loc2)) = (solution.get(l1), solution.get(l2))
                            && loc1 > loc2
                        {
                            // Need to make l2 at least as slow as l1
                            let new_loc2 = self.join(loc1, loc2);
                            if let Some(param) = params.iter().find(|p| &p.name == l2) {
                                if param.bound.satisfied_by(new_loc2) {
                                    solution.insert(l2.clone(), new_loc2);
                                    changed = true;
                                } else {
                                    return None; // Constraint unsatisfiable
                                }
                            }
                        }
                    }
                    LocalityConstraint::Faster(l1, l2) => {
                        // l1 < l2 strictly
                        if let (Some(&loc1), Some(&loc2)) = (solution.get(l1), solution.get(l2))
                            && loc1 >= loc2
                        {
                            // Need to make l2 strictly slower
                            if let Some(slower) = loc1.slower() {
                                let new_loc2 = self.join(slower, loc2);
                                if let Some(param) = params.iter().find(|p| &p.name == l2) {
                                    if param.bound.satisfied_by(new_loc2) {
                                        solution.insert(l2.clone(), new_loc2);
                                        changed = true;
                                    } else {
                                        return None;
                                    }
                                }
                            } else {
                                return None; // Can't go slower than Network
                            }
                        }
                    }
                    LocalityConstraint::Same(l1, l2) => {
                        if let (Some(&loc1), Some(&loc2)) = (solution.get(l1), solution.get(l2))
                            && loc1 != loc2
                        {
                            // Unify to the slower one
                            let unified = self.join(loc1, loc2);
                            let p1 = params.iter().find(|p| &p.name == l1);
                            let p2 = params.iter().find(|p| &p.name == l2);

                            if let (Some(p1), Some(p2)) = (p1, p2) {
                                if p1.bound.satisfied_by(unified) && p2.bound.satisfied_by(unified)
                                {
                                    solution.insert(l1.clone(), unified);
                                    solution.insert(l2.clone(), unified);
                                    changed = true;
                                } else {
                                    return None;
                                }
                            }
                        }
                    }
                    LocalityConstraint::Bound(l, bound) => {
                        if let Some(&loc) = solution.get(l)
                            && !bound.satisfied_by(loc)
                        {
                            // Try to find a valid locality
                            if loc < bound.fastest {
                                solution.insert(l.clone(), bound.fastest);
                                changed = true;
                            } else if loc > bound.slowest {
                                return None; // Already too slow
                            }
                        }
                    }
                }
            }
        }

        // Verify all constraints are satisfied
        for constraint in constraints {
            if !constraint.satisfied_by(&solution) {
                return None;
            }
        }

        Some(solution)
    }

    /// Infer the locality of an expression based on its components.
    pub fn infer_locality(&self, components: &[Locality]) -> Locality {
        // The locality of a composite is the slowest of its parts
        components
            .iter()
            .copied()
            .max()
            .unwrap_or(Locality::Register)
    }

    /// Compute the "distance" between two localities.
    /// This is useful for optimization decisions.
    pub fn distance(&self, from: Locality, to: Locality) -> i32 {
        (to as i32) - (from as i32)
    }
}

impl Default for LocalityLattice {
    fn default() -> Self {
        Self::new()
    }
}

/// Variance of locality in a type position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variance {
    /// Covariant: faster <: slower (default for values)
    Covariant,
    /// Contravariant: slower <: faster (for function inputs)
    Contravariant,
    /// Invariant: must be equal
    Invariant,
    /// Bivariant: any relation is valid
    Bivariant,
}

impl Variance {
    /// Combine two variances (for nested positions).
    pub fn combine(&self, inner: Variance) -> Variance {
        match (*self, inner) {
            (Variance::Bivariant, _) | (_, Variance::Bivariant) => Variance::Bivariant,
            (Variance::Invariant, _) | (_, Variance::Invariant) => Variance::Invariant,
            (Variance::Covariant, v) | (v, Variance::Covariant) => v,
            (Variance::Contravariant, Variance::Contravariant) => Variance::Covariant,
        }
    }

    /// Check subtyping with this variance.
    pub fn check_subtype(
        &self,
        lattice: &mut LocalityLattice,
        sub: Locality,
        sup: Locality,
    ) -> bool {
        match self {
            Variance::Covariant => lattice.is_subtype(sub, sup),
            Variance::Contravariant => lattice.is_subtype(sup, sub),
            Variance::Invariant => sub == sup,
            Variance::Bivariant => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtype_check() {
        let mut lattice = LocalityLattice::new();

        assert_eq!(
            lattice.check(Locality::L1, Locality::L1),
            SubtypeResult::Equal
        );
        assert_eq!(
            lattice.check(Locality::L1, Locality::Local),
            SubtypeResult::Subtype
        );
        assert_eq!(
            lattice.check(Locality::Local, Locality::L1),
            SubtypeResult::Supertype
        );
    }

    #[test]
    fn test_is_subtype() {
        let mut lattice = LocalityLattice::new();

        assert!(lattice.is_subtype(Locality::Register, Locality::Network));
        assert!(lattice.is_subtype(Locality::L1, Locality::L1));
        assert!(!lattice.is_subtype(Locality::Local, Locality::L2));
    }

    #[test]
    fn test_join_meet() {
        let lattice = LocalityLattice::new();

        assert_eq!(lattice.join(Locality::L1, Locality::L3), Locality::L3);
        assert_eq!(lattice.meet(Locality::L1, Locality::L3), Locality::L1);
        assert_eq!(
            lattice.join(Locality::Register, Locality::Network),
            Locality::Network
        );
        assert_eq!(
            lattice.meet(Locality::Register, Locality::Network),
            Locality::Register
        );
    }

    #[test]
    fn test_solve_simple_constraints() {
        let mut lattice = LocalityLattice::new();

        let params = vec![
            LocalityParam::new("L1", LocalityBound::any()),
            LocalityParam::new("L2", LocalityBound::any()),
        ];

        let constraints = vec![LocalityConstraint::FasterOrEqual(
            "L1".to_string(),
            "L2".to_string(),
        )];

        let solution = lattice.solve_constraints(&params, &constraints);
        assert!(solution.is_some());

        let sol = solution.unwrap();
        assert!(sol[&"L1".to_string()] <= sol[&"L2".to_string()]);
    }

    #[test]
    fn test_solve_strict_constraint() {
        let mut lattice = LocalityLattice::new();

        let params = vec![
            LocalityParam::new("L1", LocalityBound::exact(Locality::L1)),
            LocalityParam::new("L2", LocalityBound::any()),
        ];

        let constraints = vec![LocalityConstraint::Faster(
            "L1".to_string(),
            "L2".to_string(),
        )];

        let solution = lattice.solve_constraints(&params, &constraints);
        assert!(solution.is_some());

        let sol = solution.unwrap();
        assert!(sol[&"L1".to_string()] < sol[&"L2".to_string()]);
    }

    #[test]
    fn test_solve_unsatisfiable() {
        let mut lattice = LocalityLattice::new();

        let params = vec![
            LocalityParam::new("L1", LocalityBound::exact(Locality::Local)),
            LocalityParam::new("L2", LocalityBound::exact(Locality::L1)),
        ];

        // L1 (Local) must be faster than L2 (L1) - impossible!
        let constraints = vec![LocalityConstraint::Faster(
            "L1".to_string(),
            "L2".to_string(),
        )];

        let solution = lattice.solve_constraints(&params, &constraints);
        assert!(solution.is_none());
    }

    #[test]
    fn test_solve_same_constraint() {
        let mut lattice = LocalityLattice::new();

        let params = vec![
            LocalityParam::new("L1", LocalityBound::any()),
            LocalityParam::new("L2", LocalityBound::any()),
        ];

        let constraints = vec![LocalityConstraint::Same("L1".to_string(), "L2".to_string())];

        let solution = lattice.solve_constraints(&params, &constraints);
        assert!(solution.is_some());

        let sol = solution.unwrap();
        assert_eq!(sol[&"L1".to_string()], sol[&"L2".to_string()]);
    }

    #[test]
    fn test_infer_locality() {
        let lattice = LocalityLattice::new();

        assert_eq!(
            lattice.infer_locality(&[Locality::L1, Locality::L2, Locality::L3]),
            Locality::L3
        );

        assert_eq!(
            lattice.infer_locality(&[Locality::Register, Locality::Local]),
            Locality::Local
        );

        assert_eq!(lattice.infer_locality(&[]), Locality::Register);
    }

    #[test]
    fn test_distance() {
        let lattice = LocalityLattice::new();

        assert!(lattice.distance(Locality::L1, Locality::Local) > 0);
        assert!(lattice.distance(Locality::Local, Locality::L1) < 0);
        assert_eq!(lattice.distance(Locality::L2, Locality::L2), 0);
    }

    #[test]
    fn test_variance() {
        let mut lattice = LocalityLattice::new();

        // Covariant: L1 <: Local
        assert!(Variance::Covariant.check_subtype(&mut lattice, Locality::L1, Locality::Local));

        // Contravariant: Local <: L1 (reversed)
        assert!(Variance::Contravariant.check_subtype(&mut lattice, Locality::Local, Locality::L1));

        // Invariant: must be equal
        assert!(Variance::Invariant.check_subtype(&mut lattice, Locality::L2, Locality::L2));
        assert!(!Variance::Invariant.check_subtype(&mut lattice, Locality::L1, Locality::L2));

        // Bivariant: always true
        assert!(Variance::Bivariant.check_subtype(
            &mut lattice,
            Locality::Network,
            Locality::Register
        ));
    }

    #[test]
    fn test_variance_combine() {
        assert_eq!(
            Variance::Covariant.combine(Variance::Covariant),
            Variance::Covariant
        );
        assert_eq!(
            Variance::Covariant.combine(Variance::Contravariant),
            Variance::Contravariant
        );
        assert_eq!(
            Variance::Contravariant.combine(Variance::Contravariant),
            Variance::Covariant
        );
        assert_eq!(
            Variance::Invariant.combine(Variance::Covariant),
            Variance::Invariant
        );
        assert_eq!(
            Variance::Bivariant.combine(Variance::Invariant),
            Variance::Bivariant
        );
    }

    #[test]
    fn test_subbound() {
        let lattice = LocalityLattice::new();

        let hot = LocalityBound::hot();
        let any = LocalityBound::any();

        // hot (Register..L2) is a subbound of any (Register..Network)
        assert!(lattice.is_subbound(&hot, &any));

        // any is not a subbound of hot
        assert!(!lattice.is_subbound(&any, &hot));
    }
}
