//! Qualifier templates for liquid type inference
//!
//! Qualifiers are predicate templates used for automatic inference of refinement types.
//! The key insight from Liquid Types is that instead of inferring arbitrary predicates,
//! we only consider conjunctions of "qualifiers" from a predefined set.
//!
//! # How It Works
//!
//! 1. Define a set of qualifier templates: `Q = {q1, q2, ...}`
//! 2. For each program point, the refinement is some subset of Q
//! 3. Inference finds the largest valid subset using constraint solving
//!
//! # Example Qualifiers
//!
//! - `v > 0` - positive
//! - `v >= 0` - non-negative
//! - `v < x` - less than some variable x
//! - `v = x + y` - sum of two variables
//!
//! # References
//!
//! - Rondon, P. M., Kawaguchi, M., & Jhala, R. (2008). Liquid types.
//! - Rondon, P. M., et al. (2010). Low-level liquid types.

use super::predicate::*;
use std::collections::HashSet;

/// A qualifier template
///
/// A qualifier is a predicate template with free variables that can be
/// instantiated with concrete terms from the program.
#[derive(Debug, Clone)]
pub struct Qualifier {
    /// Template name (for debugging)
    pub name: String,

    /// Free variables in the template
    /// The first variable is always the "value" variable (v)
    pub params: Vec<String>,

    /// The predicate template
    pub predicate: Predicate,

    /// Category for filtering
    pub category: QualifierCategory,
}

/// Category of qualifier for filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualifierCategory {
    /// Basic comparisons (v > 0, v = x)
    Basic,
    /// Arithmetic relationships (v = x + y)
    Arithmetic,
    /// Array/bounds related
    Bounds,
    /// Medical/domain-specific
    Medical,
    /// Custom user-defined
    Custom,
}

impl Qualifier {
    /// Create a new qualifier
    pub fn new(
        name: impl Into<String>,
        params: Vec<&str>,
        pred: Predicate,
        category: QualifierCategory,
    ) -> Self {
        Self {
            name: name.into(),
            params: params.into_iter().map(|s| s.to_string()).collect(),
            predicate: pred,
            category,
        }
    }

    /// Create a basic qualifier (for comparisons with constants)
    pub fn basic(name: impl Into<String>, params: Vec<&str>, pred: Predicate) -> Self {
        Self::new(name, params, pred, QualifierCategory::Basic)
    }

    /// Create an arithmetic qualifier
    pub fn arithmetic(name: impl Into<String>, params: Vec<&str>, pred: Predicate) -> Self {
        Self::new(name, params, pred, QualifierCategory::Arithmetic)
    }

    /// Create a bounds qualifier
    pub fn bounds(name: impl Into<String>, params: Vec<&str>, pred: Predicate) -> Self {
        Self::new(name, params, pred, QualifierCategory::Bounds)
    }

    /// Create a medical qualifier
    pub fn medical(name: impl Into<String>, params: Vec<&str>, pred: Predicate) -> Self {
        Self::new(name, params, pred, QualifierCategory::Medical)
    }

    /// Instantiate the qualifier with concrete terms
    ///
    /// Returns None if the number of arguments doesn't match.
    pub fn instantiate(&self, args: &[Term]) -> Option<Predicate> {
        if args.len() != self.params.len() {
            return None;
        }

        let mut pred = self.predicate.clone();
        for (param, arg) in self.params.iter().zip(args.iter()) {
            pred = pred.substitute(param, arg);
        }

        Some(pred)
    }

    /// Get all possible instantiations given a set of available variables
    ///
    /// This generates all ways to map the qualifier's parameters to the
    /// available variables (excluding the first parameter which is always 'v').
    pub fn all_instantiations(&self, v: &Term, vars: &[Term]) -> Vec<Predicate> {
        if self.params.is_empty() {
            return vec![self.predicate.clone()];
        }

        if self.params.len() == 1 {
            // Single parameter (the value variable)
            return self.instantiate(&[v.clone()]).into_iter().collect();
        }

        // Generate all combinations for remaining parameters
        let remaining_params = self.params.len() - 1;
        let combinations = generate_combinations(vars, remaining_params);

        combinations
            .into_iter()
            .filter_map(|combo| {
                let mut args = vec![v.clone()];
                args.extend(combo);
                self.instantiate(&args)
            })
            .collect()
    }

    /// Check if this qualifier applies to a given type
    pub fn applies_to(&self, _base_type: &crate::types::Type) -> bool {
        // Most qualifiers apply to numeric types
        // Could be refined based on type
        true
    }
}

/// Generate all k-combinations of elements
fn generate_combinations(elements: &[Term], k: usize) -> Vec<Vec<Term>> {
    if k == 0 {
        return vec![vec![]];
    }

    if elements.is_empty() {
        return vec![];
    }

    let mut result = Vec::new();

    // Generate combinations with repetition
    fn helper(
        elements: &[Term],
        k: usize,
        start: usize,
        current: &mut Vec<Term>,
        result: &mut Vec<Vec<Term>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }

        for i in start..elements.len() {
            current.push(elements[i].clone());
            helper(elements, k, i, current, result);
            current.pop();
        }
    }

    let mut current = Vec::new();
    helper(elements, k, 0, &mut current, &mut result);

    // Also allow using same variable multiple times
    fn helper_with_rep(
        elements: &[Term],
        k: usize,
        current: &mut Vec<Term>,
        result: &mut Vec<Vec<Term>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }

        for elem in elements {
            current.push(elem.clone());
            helper_with_rep(elements, k, current, result);
            current.pop();
        }
    }

    let mut current = Vec::new();
    helper_with_rep(elements, k, &mut current, &mut result);

    // Remove duplicates
    let mut seen = HashSet::new();
    result
        .into_iter()
        .filter(|combo| {
            let key = format!("{:?}", combo);
            seen.insert(key)
        })
        .collect()
}

/// Standard qualifiers for liquid type inference
///
/// These are the core qualifiers used for inferring refinements.
pub fn standard_qualifiers() -> Vec<Qualifier> {
    vec![
        // ======== Basic comparisons ========

        // v = 0
        Qualifier::basic(
            "Zero",
            vec!["v"],
            Predicate::eq(Term::var("v"), Term::int(0)),
        ),
        // v > 0
        Qualifier::basic(
            "Pos",
            vec!["v"],
            Predicate::gt(Term::var("v"), Term::int(0)),
        ),
        // v >= 0
        Qualifier::basic(
            "NonNeg",
            vec!["v"],
            Predicate::ge(Term::var("v"), Term::int(0)),
        ),
        // v < 0
        Qualifier::basic(
            "Neg",
            vec!["v"],
            Predicate::lt(Term::var("v"), Term::int(0)),
        ),
        // v <= 0
        Qualifier::basic(
            "NonPos",
            vec!["v"],
            Predicate::le(Term::var("v"), Term::int(0)),
        ),
        // v != 0
        Qualifier::basic(
            "NonZero",
            vec!["v"],
            Predicate::ne(Term::var("v"), Term::int(0)),
        ),
        // ======== Variable comparisons ========

        // v = x
        Qualifier::basic(
            "EqVar",
            vec!["v", "x"],
            Predicate::eq(Term::var("v"), Term::var("x")),
        ),
        // v != x
        Qualifier::basic(
            "NeVar",
            vec!["v", "x"],
            Predicate::ne(Term::var("v"), Term::var("x")),
        ),
        // v < x
        Qualifier::basic(
            "LtVar",
            vec!["v", "x"],
            Predicate::lt(Term::var("v"), Term::var("x")),
        ),
        // v <= x
        Qualifier::basic(
            "LeVar",
            vec!["v", "x"],
            Predicate::le(Term::var("v"), Term::var("x")),
        ),
        // v > x
        Qualifier::basic(
            "GtVar",
            vec!["v", "x"],
            Predicate::gt(Term::var("v"), Term::var("x")),
        ),
        // v >= x
        Qualifier::basic(
            "GeVar",
            vec!["v", "x"],
            Predicate::ge(Term::var("v"), Term::var("x")),
        ),
        // ======== Arithmetic ========

        // v = x + y
        Qualifier::arithmetic(
            "Sum",
            vec!["v", "x", "y"],
            Predicate::eq(Term::var("v"), Term::add(Term::var("x"), Term::var("y"))),
        ),
        // v = x - y
        Qualifier::arithmetic(
            "Diff",
            vec!["v", "x", "y"],
            Predicate::eq(Term::var("v"), Term::sub(Term::var("x"), Term::var("y"))),
        ),
        // v = x * y
        Qualifier::arithmetic(
            "Prod",
            vec!["v", "x", "y"],
            Predicate::eq(Term::var("v"), Term::mul(Term::var("x"), Term::var("y"))),
        ),
        // v = x + 1
        Qualifier::arithmetic(
            "Succ",
            vec!["v", "x"],
            Predicate::eq(Term::var("v"), Term::add(Term::var("x"), Term::int(1))),
        ),
        // v = x - 1
        Qualifier::arithmetic(
            "Pred",
            vec!["v", "x"],
            Predicate::eq(Term::var("v"), Term::sub(Term::var("x"), Term::int(1))),
        ),
        // ======== Bounds ========

        // 0 <= v < x (array bounds)
        Qualifier::bounds(
            "InBounds",
            vec!["v", "x"],
            Predicate::and([
                Predicate::ge(Term::var("v"), Term::int(0)),
                Predicate::lt(Term::var("v"), Term::var("x")),
            ]),
        ),
        // v = len(x)
        Qualifier::bounds(
            "IsLen",
            vec!["v", "x"],
            Predicate::eq(Term::var("v"), Term::Len(Box::new(Term::var("x")))),
        ),
        // v < len(x)
        Qualifier::bounds(
            "LtLen",
            vec!["v", "x"],
            Predicate::lt(Term::var("v"), Term::Len(Box::new(Term::var("x")))),
        ),
        // v >= len(x)
        Qualifier::bounds(
            "GeLen",
            vec!["v", "x"],
            Predicate::ge(Term::var("v"), Term::Len(Box::new(Term::var("x")))),
        ),
    ]
}

/// Medical-specific qualifiers for pharmacological computing
///
/// These qualifiers encode domain-specific constraints for medical applications.
pub fn medical_qualifiers() -> Vec<Qualifier> {
    vec![
        // ======== Dose constraints ========

        // Valid dose range: 0 < dose <= max
        Qualifier::medical(
            "SafeDose",
            vec!["dose", "max"],
            Predicate::and([
                Predicate::gt(Term::var("dose"), Term::float(0.0)),
                Predicate::le(Term::var("dose"), Term::var("max")),
            ]),
        ),
        // Positive dose: dose > 0
        Qualifier::medical(
            "PosDose",
            vec!["dose"],
            Predicate::gt(Term::var("dose"), Term::float(0.0)),
        ),
        // ======== Physiological constraints ========

        // Valid concentration: conc >= 0
        Qualifier::medical(
            "ValidConc",
            vec!["conc"],
            Predicate::ge(Term::var("conc"), Term::float(0.0)),
        ),
        // Therapeutic range: min <= conc <= max
        Qualifier::medical(
            "TherapeuticRange",
            vec!["conc", "min", "max"],
            Predicate::and([
                Predicate::ge(Term::var("conc"), Term::var("min")),
                Predicate::le(Term::var("conc"), Term::var("max")),
            ]),
        ),
        // Valid clearance: 0 < crcl < 200
        Qualifier::medical(
            "ValidCrCl",
            vec!["crcl"],
            Predicate::and([
                Predicate::gt(Term::var("crcl"), Term::float(0.0)),
                Predicate::lt(Term::var("crcl"), Term::float(200.0)),
            ]),
        ),
        // Valid age: 0 <= age <= 150
        Qualifier::medical(
            "ValidAge",
            vec!["age"],
            Predicate::and([
                Predicate::ge(Term::var("age"), Term::float(0.0)),
                Predicate::le(Term::var("age"), Term::float(150.0)),
            ]),
        ),
        // Positive weight: weight > 0
        Qualifier::medical(
            "PosWeight",
            vec!["weight"],
            Predicate::gt(Term::var("weight"), Term::float(0.0)),
        ),
        // Valid weight: 0 < weight <= 500
        Qualifier::medical(
            "ValidWeight",
            vec!["weight"],
            Predicate::and([
                Predicate::gt(Term::var("weight"), Term::float(0.0)),
                Predicate::le(Term::var("weight"), Term::float(500.0)),
            ]),
        ),
        // Valid serum creatinine: 0.1 <= scr <= 20
        Qualifier::medical(
            "ValidSCr",
            vec!["scr"],
            Predicate::and([
                Predicate::ge(Term::var("scr"), Term::float(0.1)),
                Predicate::le(Term::var("scr"), Term::float(20.0)),
            ]),
        ),
        // ======== Adjustment factors ========

        // Dose adjustment factor: 0 < factor <= 1
        Qualifier::medical(
            "AdjustFactor",
            vec!["factor"],
            Predicate::and([
                Predicate::gt(Term::var("factor"), Term::float(0.0)),
                Predicate::le(Term::var("factor"), Term::float(1.0)),
            ]),
        ),
        // Probability: 0 <= p <= 1
        Qualifier::medical(
            "Probability",
            vec!["p"],
            Predicate::and([
                Predicate::ge(Term::var("p"), Term::float(0.0)),
                Predicate::le(Term::var("p"), Term::float(1.0)),
            ]),
        ),
        // ======== Vital signs ========

        // Valid heart rate: 20 <= hr <= 300
        Qualifier::medical(
            "ValidHR",
            vec!["hr"],
            Predicate::and([
                Predicate::ge(Term::var("hr"), Term::float(20.0)),
                Predicate::le(Term::var("hr"), Term::float(300.0)),
            ]),
        ),
        // Valid systolic BP: 40 <= bp <= 300
        Qualifier::medical(
            "ValidSysBP",
            vec!["bp"],
            Predicate::and([
                Predicate::ge(Term::var("bp"), Term::float(40.0)),
                Predicate::le(Term::var("bp"), Term::float(300.0)),
            ]),
        ),
        // Valid diastolic BP: 20 <= bp <= 200
        Qualifier::medical(
            "ValidDiaBP",
            vec!["bp"],
            Predicate::and([
                Predicate::ge(Term::var("bp"), Term::float(20.0)),
                Predicate::le(Term::var("bp"), Term::float(200.0)),
            ]),
        ),
        // Valid temperature: 25 <= temp <= 45 (Celsius)
        Qualifier::medical(
            "ValidTemp",
            vec!["temp"],
            Predicate::and([
                Predicate::ge(Term::var("temp"), Term::float(25.0)),
                Predicate::le(Term::var("temp"), Term::float(45.0)),
            ]),
        ),
    ]
}

/// A qualifier set for inference
///
/// Manages a collection of qualifiers and their instantiations.
#[derive(Debug, Clone)]
pub struct QualifierSet {
    qualifiers: Vec<Qualifier>,
    categories: HashSet<QualifierCategory>,
}

impl QualifierSet {
    /// Create an empty qualifier set
    pub fn new() -> Self {
        Self {
            qualifiers: Vec::new(),
            categories: HashSet::new(),
        }
    }

    /// Create a qualifier set with standard qualifiers
    pub fn standard() -> Self {
        let mut set = Self::new();
        set.add_all(standard_qualifiers());
        set
    }

    /// Create a qualifier set with standard and medical qualifiers
    pub fn with_medical() -> Self {
        let mut set = Self::standard();
        set.add_all(medical_qualifiers());
        set
    }

    /// Add a qualifier to the set
    pub fn add(&mut self, qualifier: Qualifier) {
        self.categories.insert(qualifier.category);
        self.qualifiers.push(qualifier);
    }

    /// Add multiple qualifiers
    pub fn add_all(&mut self, qualifiers: impl IntoIterator<Item = Qualifier>) {
        for q in qualifiers {
            self.add(q);
        }
    }

    /// Get all qualifiers
    pub fn qualifiers(&self) -> &[Qualifier] {
        &self.qualifiers
    }

    /// Get qualifiers by category
    pub fn by_category(&self, category: QualifierCategory) -> Vec<&Qualifier> {
        self.qualifiers
            .iter()
            .filter(|q| q.category == category)
            .collect()
    }

    /// Get all possible instantiations for a value variable given available terms
    pub fn instantiate_all(&self, v: &Term, vars: &[Term]) -> Vec<Predicate> {
        self.qualifiers
            .iter()
            .flat_map(|q| q.all_instantiations(v, vars))
            .collect()
    }

    /// Get the number of qualifiers
    pub fn len(&self) -> usize {
        self.qualifiers.len()
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.qualifiers.is_empty()
    }
}

impl Default for QualifierSet {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qualifier_instantiation() {
        let q = Qualifier::basic(
            "LtVar",
            vec!["v", "x"],
            Predicate::lt(Term::var("v"), Term::var("x")),
        );

        let result = q.instantiate(&[Term::var("y"), Term::var("z")]);
        assert!(result.is_some());

        let pred = result.unwrap();
        assert_eq!(pred, Predicate::lt(Term::var("y"), Term::var("z")));
    }

    #[test]
    fn test_qualifier_wrong_arity() {
        let q = Qualifier::basic(
            "LtVar",
            vec!["v", "x"],
            Predicate::lt(Term::var("v"), Term::var("x")),
        );

        // Wrong number of arguments
        let result = q.instantiate(&[Term::var("y")]);
        assert!(result.is_none());
    }

    #[test]
    fn test_standard_qualifiers_count() {
        let qualifiers = standard_qualifiers();

        // Should have a reasonable number of qualifiers
        assert!(qualifiers.len() >= 15);

        // Check categories
        let categories: HashSet<_> = qualifiers.iter().map(|q| q.category).collect();
        assert!(categories.contains(&QualifierCategory::Basic));
        assert!(categories.contains(&QualifierCategory::Arithmetic));
        assert!(categories.contains(&QualifierCategory::Bounds));
    }

    #[test]
    fn test_medical_qualifiers_count() {
        let qualifiers = medical_qualifiers();

        // Should have medical qualifiers
        assert!(qualifiers.len() >= 10);

        // All should be medical category
        for q in &qualifiers {
            assert_eq!(q.category, QualifierCategory::Medical);
        }
    }

    #[test]
    fn test_qualifier_set_standard() {
        let set = QualifierSet::standard();

        assert!(!set.is_empty());
        assert!(set.len() >= 15);
    }

    #[test]
    fn test_qualifier_set_with_medical() {
        let set = QualifierSet::with_medical();

        let standard = QualifierSet::standard();
        assert!(set.len() > standard.len());
    }

    #[test]
    fn test_qualifier_set_by_category() {
        let set = QualifierSet::with_medical();

        let basic = set.by_category(QualifierCategory::Basic);
        let medical = set.by_category(QualifierCategory::Medical);

        assert!(!basic.is_empty());
        assert!(!medical.is_empty());
    }

    #[test]
    fn test_all_instantiations() {
        let q = Qualifier::basic(
            "LtVar",
            vec!["v", "x"],
            Predicate::lt(Term::var("v"), Term::var("x")),
        );

        let v = Term::var("v");
        let vars = vec![Term::var("a"), Term::var("b")];

        let instantiations = q.all_instantiations(&v, &vars);

        // Should have instantiations for each var
        assert!(instantiations.len() >= 2);
    }

    #[test]
    fn test_single_param_qualifier() {
        let q = Qualifier::basic(
            "Pos",
            vec!["v"],
            Predicate::gt(Term::var("v"), Term::int(0)),
        );

        let v = Term::var("x");
        let vars = vec![Term::var("a"), Term::var("b")];

        let instantiations = q.all_instantiations(&v, &vars);

        // Should have exactly one instantiation
        assert_eq!(instantiations.len(), 1);
        assert_eq!(
            instantiations[0],
            Predicate::gt(Term::var("x"), Term::int(0))
        );
    }

    #[test]
    fn test_therapeutic_range_qualifier() {
        let qualifiers = medical_qualifiers();

        let therapeutic = qualifiers
            .iter()
            .find(|q| q.name == "TherapeuticRange")
            .expect("Should have TherapeuticRange qualifier");

        assert_eq!(therapeutic.params.len(), 3);

        // Instantiate with concrete values
        let result =
            therapeutic.instantiate(&[Term::var("conc"), Term::float(10.0), Term::float(20.0)]);

        assert!(result.is_some());
    }

    #[test]
    fn test_generate_combinations() {
        let elements = vec![Term::var("a"), Term::var("b")];

        let combos_1 = generate_combinations(&elements, 1);
        assert!(combos_1.len() >= 2); // a, b

        let combos_2 = generate_combinations(&elements, 2);
        assert!(combos_2.len() >= 3); // (a,a), (a,b), (b,b)
    }
}
