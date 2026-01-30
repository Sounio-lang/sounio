//! SMT Formula Representation
//!
//! This module defines the core data structures for SMT formulas,
//! independent of any specific solver backend.

use std::collections::HashMap;
use std::fmt;

/// SMT sort (type) for variables and expressions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SmtSort {
    /// Boolean type
    Bool,
    /// Integer type (arbitrary precision)
    Int,
    /// Real number type
    Real,
    /// Bitvector with specified width
    BitVec(u32),
    /// Array from index sort to element sort
    Array(Box<SmtSort>, Box<SmtSort>),
    /// Uninterpreted sort (for abstract types)
    Uninterpreted(String),
    /// Epsilon type for uncertainty tracking
    Epsilon,
    /// Provenance type for data lineage
    Provenance,
}

impl fmt::Display for SmtSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtSort::Bool => write!(f, "Bool"),
            SmtSort::Int => write!(f, "Int"),
            SmtSort::Real => write!(f, "Real"),
            SmtSort::BitVec(w) => write!(f, "(_ BitVec {})", w),
            SmtSort::Array(idx, elem) => write!(f, "(Array {} {})", idx, elem),
            SmtSort::Uninterpreted(name) => write!(f, "{}", name),
            SmtSort::Epsilon => write!(f, "Epsilon"),
            SmtSort::Provenance => write!(f, "Provenance"),
        }
    }
}

/// SMT formula (logical constraint)
#[derive(Debug, Clone, PartialEq)]
pub enum SmtFormula {
    /// True constant
    True,
    /// False constant
    False,
    /// Equality: lhs = rhs
    Eq(Box<SmtTerm>, Box<SmtTerm>),
    /// Less than: lhs < rhs
    Lt(Box<SmtTerm>, Box<SmtTerm>),
    /// Less than or equal: lhs <= rhs
    Le(Box<SmtTerm>, Box<SmtTerm>),
    /// Greater than: lhs > rhs
    Gt(Box<SmtTerm>, Box<SmtTerm>),
    /// Greater than or equal: lhs >= rhs
    Ge(Box<SmtTerm>, Box<SmtTerm>),
    /// Negation: ¬φ
    Not(Box<SmtFormula>),
    /// Conjunction: φ₁ ∧ ... ∧ φₙ
    And(Vec<SmtFormula>),
    /// Disjunction: φ₁ ∨ ... ∨ φₙ
    Or(Vec<SmtFormula>),
    /// Implication: φ ⟹ ψ
    Implies(Box<SmtFormula>, Box<SmtFormula>),
    /// If-then-else: if φ then ψ₁ else ψ₂
    Ite(Box<SmtFormula>, Box<SmtFormula>, Box<SmtFormula>),
    /// Universal quantification: ∀x:τ. φ
    Forall(String, SmtSort, Box<SmtFormula>),
    /// Existential quantification: ∃x:τ. φ
    Exists(String, SmtSort, Box<SmtFormula>),
    /// Predicate application: p(t₁, ..., tₙ)
    App(String, Vec<SmtTerm>),
    /// Term as formula (for boolean terms)
    Term(SmtTerm),
}

impl SmtFormula {
    /// Create a negation, simplifying double negation
    pub fn not(self) -> Self {
        match self {
            SmtFormula::True => SmtFormula::False,
            SmtFormula::False => SmtFormula::True,
            SmtFormula::Not(inner) => *inner,
            other => SmtFormula::Not(Box::new(other)),
        }
    }

    /// Create a conjunction, flattening nested conjunctions
    pub fn and(formulas: impl IntoIterator<Item = SmtFormula>) -> Self {
        let mut result = Vec::new();
        for f in formulas {
            match f {
                SmtFormula::True => continue,
                SmtFormula::False => return SmtFormula::False,
                SmtFormula::And(inner) => result.extend(inner),
                other => result.push(other),
            }
        }
        match result.len() {
            0 => SmtFormula::True,
            1 => result.into_iter().next().unwrap(),
            _ => SmtFormula::And(result),
        }
    }

    /// Create a disjunction, flattening nested disjunctions
    pub fn or(formulas: impl IntoIterator<Item = SmtFormula>) -> Self {
        let mut result = Vec::new();
        for f in formulas {
            match f {
                SmtFormula::False => continue,
                SmtFormula::True => return SmtFormula::True,
                SmtFormula::Or(inner) => result.extend(inner),
                other => result.push(other),
            }
        }
        match result.len() {
            0 => SmtFormula::False,
            1 => result.into_iter().next().unwrap(),
            _ => SmtFormula::Or(result),
        }
    }

    /// Create an implication
    pub fn implies(antecedent: SmtFormula, consequent: SmtFormula) -> Self {
        match (&antecedent, &consequent) {
            (SmtFormula::False, _) => SmtFormula::True,
            (_, SmtFormula::True) => SmtFormula::True,
            (SmtFormula::True, _) => consequent,
            _ => SmtFormula::Implies(Box::new(antecedent), Box::new(consequent)),
        }
    }

    /// Convert to SMT-LIB2 format string
    pub fn to_smtlib(&self) -> String {
        match self {
            SmtFormula::True => "true".to_string(),
            SmtFormula::False => "false".to_string(),
            SmtFormula::Eq(l, r) => format!("(= {} {})", l.to_smtlib(), r.to_smtlib()),
            SmtFormula::Lt(l, r) => format!("(< {} {})", l.to_smtlib(), r.to_smtlib()),
            SmtFormula::Le(l, r) => format!("(<= {} {})", l.to_smtlib(), r.to_smtlib()),
            SmtFormula::Gt(l, r) => format!("(> {} {})", l.to_smtlib(), r.to_smtlib()),
            SmtFormula::Ge(l, r) => format!("(>= {} {})", l.to_smtlib(), r.to_smtlib()),
            SmtFormula::Not(f) => format!("(not {})", f.to_smtlib()),
            SmtFormula::And(fs) => {
                if fs.is_empty() {
                    "true".to_string()
                } else {
                    let inner: Vec<_> = fs.iter().map(|f| f.to_smtlib()).collect();
                    format!("(and {})", inner.join(" "))
                }
            }
            SmtFormula::Or(fs) => {
                if fs.is_empty() {
                    "false".to_string()
                } else {
                    let inner: Vec<_> = fs.iter().map(|f| f.to_smtlib()).collect();
                    format!("(or {})", inner.join(" "))
                }
            }
            SmtFormula::Implies(p, q) => {
                format!("(=> {} {})", p.to_smtlib(), q.to_smtlib())
            }
            SmtFormula::Ite(c, t, e) => {
                format!(
                    "(ite {} {} {})",
                    c.to_smtlib(),
                    t.to_smtlib(),
                    e.to_smtlib()
                )
            }
            SmtFormula::Forall(var, sort, body) => {
                format!("(forall (({} {})) {})", var, sort, body.to_smtlib())
            }
            SmtFormula::Exists(var, sort, body) => {
                format!("(exists (({} {})) {})", var, sort, body.to_smtlib())
            }
            SmtFormula::App(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let arg_strs: Vec<_> = args.iter().map(|a| a.to_smtlib()).collect();
                    format!("({} {})", name, arg_strs.join(" "))
                }
            }
            SmtFormula::Term(t) => t.to_smtlib(),
        }
    }
}

impl fmt::Display for SmtFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_smtlib())
    }
}

/// SMT term (expression)
#[derive(Debug, Clone, PartialEq)]
pub enum SmtTerm {
    /// Variable reference
    Var(String),
    /// Boolean constant
    Bool(bool),
    /// Integer constant
    Int(i64),
    /// Real constant
    Real(f64),
    /// Addition: l + r
    Add(Box<SmtTerm>, Box<SmtTerm>),
    /// Subtraction: l - r
    Sub(Box<SmtTerm>, Box<SmtTerm>),
    /// Multiplication: l * r
    Mul(Box<SmtTerm>, Box<SmtTerm>),
    /// Division: l / r
    Div(Box<SmtTerm>, Box<SmtTerm>),
    /// Modulo: l % r
    Mod(Box<SmtTerm>, Box<SmtTerm>),
    /// Negation: -t
    Neg(Box<SmtTerm>),
    /// Logical not: !t
    Not(Box<SmtTerm>),
    /// Absolute value: |t|
    Abs(Box<SmtTerm>),
    /// Function application: f(t₁, ..., tₙ)
    App(String, Vec<SmtTerm>),
    /// Field access: t.field
    Field(Box<SmtTerm>, String),
    /// Length: len(t)
    Len(Box<SmtTerm>),
    /// If-then-else: if c then t else e
    Ite(Box<SmtTerm>, Box<SmtTerm>, Box<SmtTerm>),
    /// Epsilon value (uncertainty)
    Epsilon(f64),
    /// Provenance marker
    Provenance(String),
}

impl SmtTerm {
    /// Create a variable term
    pub fn var(name: impl Into<String>) -> Self {
        SmtTerm::Var(name.into())
    }

    /// Create an integer constant
    pub fn int(n: i64) -> Self {
        SmtTerm::Int(n)
    }

    /// Create a real constant
    pub fn real(n: f64) -> Self {
        SmtTerm::Real(n)
    }

    /// Convert to SMT-LIB2 format string
    pub fn to_smtlib(&self) -> String {
        match self {
            SmtTerm::Var(name) => name.clone(),
            SmtTerm::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            SmtTerm::Int(n) => {
                if *n < 0 {
                    format!("(- {})", -n)
                } else {
                    n.to_string()
                }
            }
            SmtTerm::Real(f) => {
                if *f < 0.0 {
                    format!("(- {})", -f)
                } else {
                    format!("{}", f)
                }
            }
            SmtTerm::Add(l, r) => format!("(+ {} {})", l.to_smtlib(), r.to_smtlib()),
            SmtTerm::Sub(l, r) => format!("(- {} {})", l.to_smtlib(), r.to_smtlib()),
            SmtTerm::Mul(l, r) => format!("(* {} {})", l.to_smtlib(), r.to_smtlib()),
            SmtTerm::Div(l, r) => format!("(/ {} {})", l.to_smtlib(), r.to_smtlib()),
            SmtTerm::Mod(l, r) => format!("(mod {} {})", l.to_smtlib(), r.to_smtlib()),
            SmtTerm::Neg(t) => format!("(- {})", t.to_smtlib()),
            SmtTerm::Not(t) => format!("(not {})", t.to_smtlib()),
            SmtTerm::Abs(t) => format!("(abs {})", t.to_smtlib()),
            SmtTerm::App(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let arg_strs: Vec<_> = args.iter().map(|a| a.to_smtlib()).collect();
                    format!("({} {})", name, arg_strs.join(" "))
                }
            }
            SmtTerm::Field(base, field) => {
                format!("(field {} {})", base.to_smtlib(), field)
            }
            SmtTerm::Len(t) => format!("(len {})", t.to_smtlib()),
            SmtTerm::Ite(c, t, e) => {
                format!(
                    "(ite {} {} {})",
                    c.to_smtlib(),
                    t.to_smtlib(),
                    e.to_smtlib()
                )
            }
            SmtTerm::Epsilon(e) => format!("(epsilon {})", e),
            SmtTerm::Provenance(p) => format!("(provenance {})", p),
        }
    }
}

impl fmt::Display for SmtTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_smtlib())
    }
}

/// SMT context for managing declarations and assertions
#[derive(Debug, Clone)]
pub struct SmtContext {
    /// Variable declarations: name → sort
    variables: HashMap<String, SmtSort>,
    /// Function declarations: name → (argument sorts, return sort)
    functions: HashMap<String, (Vec<SmtSort>, SmtSort)>,
    /// Uninterpreted sort declarations
    sorts: Vec<String>,
    /// Assertions (constraints to check)
    assertions: Vec<SmtFormula>,
}

impl SmtContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            sorts: Vec::new(),
            assertions: Vec::new(),
        }
    }

    /// Declare a variable with its sort
    pub fn declare_var(&mut self, name: String, sort: SmtSort) {
        self.variables.insert(name, sort);
    }

    /// Check if a variable is declared
    pub fn has_var(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Get the sort of a declared variable
    pub fn var_sort(&self, name: &str) -> Option<&SmtSort> {
        self.variables.get(name)
    }

    /// Declare a function with its signature
    pub fn declare_fun(&mut self, name: String, args: Vec<SmtSort>, ret: SmtSort) {
        self.functions.insert(name, (args, ret));
    }

    /// Declare an uninterpreted sort
    pub fn declare_sort(&mut self, name: String) {
        if !self.sorts.contains(&name) {
            self.sorts.push(name);
        }
    }

    /// Add an assertion to the context
    pub fn assert(&mut self, formula: SmtFormula) {
        self.assertions.push(formula);
    }

    /// Get all assertions
    pub fn assertions(&self) -> &[SmtFormula] {
        &self.assertions
    }

    /// Clear all assertions
    pub fn clear_assertions(&mut self) {
        self.assertions.clear();
    }

    /// Generate SMT-LIB2 script for this context
    pub fn to_smtlib(&self) -> String {
        let mut lines = Vec::new();

        // Logic declaration
        lines.push("(set-logic QF_NRA)".to_string());

        // Sort declarations
        for sort in &self.sorts {
            lines.push(format!("(declare-sort {} 0)", sort));
        }

        // Epsilon sort (for epistemic types)
        lines.push("(declare-sort Epsilon 0)".to_string());
        lines.push("(declare-sort Provenance 0)".to_string());

        // Variable declarations
        for (name, sort) in &self.variables {
            lines.push(format!("(declare-const {} {})", name, sort));
        }

        // Function declarations
        for (name, (args, ret)) in &self.functions {
            let arg_str = if args.is_empty() {
                "()".to_string()
            } else {
                let arg_strs: Vec<_> = args.iter().map(|s| format!("{}", s)).collect();
                format!("({})", arg_strs.join(" "))
            };
            lines.push(format!("(declare-fun {} {} {})", name, arg_str, ret));
        }

        // Assertions
        for assertion in &self.assertions {
            lines.push(format!("(assert {})", assertion.to_smtlib()));
        }

        // Check satisfiability
        lines.push("(check-sat)".to_string());
        lines.push("(get-model)".to_string());

        lines.join("\n")
    }
}

impl Default for SmtContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formula_simplification() {
        // True ∧ P = P
        let p = SmtFormula::Gt(Box::new(SmtTerm::var("x")), Box::new(SmtTerm::int(0)));
        let result = SmtFormula::and([SmtFormula::True, p.clone()]);
        assert_eq!(result, p);

        // False ∨ P = P
        let result = SmtFormula::or([SmtFormula::False, p.clone()]);
        assert_eq!(result, p);
    }

    #[test]
    fn test_smtlib_generation() {
        let formula = SmtFormula::And(vec![
            SmtFormula::Gt(Box::new(SmtTerm::var("x")), Box::new(SmtTerm::int(0))),
            SmtFormula::Lt(Box::new(SmtTerm::var("x")), Box::new(SmtTerm::int(100))),
        ]);

        let smtlib = formula.to_smtlib();
        assert!(smtlib.contains("and"));
        assert!(smtlib.contains("> x 0"));
        assert!(smtlib.contains("< x 100"));
    }

    #[test]
    fn test_context_declarations() {
        let mut ctx = SmtContext::new();
        ctx.declare_var("x".to_string(), SmtSort::Real);
        ctx.declare_var("y".to_string(), SmtSort::Int);

        assert!(ctx.has_var("x"));
        assert!(ctx.has_var("y"));
        assert!(!ctx.has_var("z"));

        assert_eq!(ctx.var_sort("x"), Some(&SmtSort::Real));
        assert_eq!(ctx.var_sort("y"), Some(&SmtSort::Int));
    }

    #[test]
    fn test_full_smtlib_script() {
        let mut ctx = SmtContext::new();
        ctx.declare_var("x".to_string(), SmtSort::Real);
        ctx.assert(SmtFormula::Gt(
            Box::new(SmtTerm::var("x")),
            Box::new(SmtTerm::int(0)),
        ));

        let script = ctx.to_smtlib();
        assert!(script.contains("declare-const x Real"));
        assert!(script.contains("assert"));
        assert!(script.contains("check-sat"));
    }
}
