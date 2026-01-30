//! Predicate language for refinement types
//!
//! Predicates are logical formulas over program values. They form the
//! core of the refinement type system, allowing us to express properties
//! like bounds, relationships between values, and domain-specific constraints.

use crate::types::Type;
use std::collections::HashSet;
use std::fmt;

/// A refinement type: `{ v: T | P }`
///
/// Represents a base type `T` refined by a predicate `P` over the
/// refinement variable `v`.
#[derive(Debug, Clone)]
pub struct RefinementType {
    /// The base type
    pub base: Type,

    /// The refinement variable name
    pub var: String,

    /// The predicate constraining the value
    pub predicate: Predicate,
}

impl RefinementType {
    /// Create a trivially refined type (predicate = true)
    pub fn trivial(base: Type) -> Self {
        Self {
            base,
            var: "v".to_string(),
            predicate: Predicate::True,
        }
    }

    /// Create a refined type with a specific predicate
    pub fn refined(base: Type, var: impl Into<String>, predicate: Predicate) -> Self {
        Self {
            base,
            var: var.into(),
            predicate,
        }
    }

    /// Create a positive number refinement: `{ v | v > 0 }`
    pub fn positive(base: Type) -> Self {
        Self::refined(
            base,
            "v",
            Predicate::Atom(Atom::new(CompareOp::Gt, Term::var("v"), Term::int(0))),
        )
    }

    /// Create a non-negative refinement: `{ v | v >= 0 }`
    pub fn non_negative(base: Type) -> Self {
        Self::refined(
            base,
            "v",
            Predicate::Atom(Atom::new(CompareOp::Ge, Term::var("v"), Term::int(0))),
        )
    }

    /// Create a bounded range refinement: `{ v | lo <= v <= hi }`
    pub fn bounded(base: Type, lo: f64, hi: f64) -> Self {
        Self::refined(
            base,
            "v",
            Predicate::and([
                Predicate::Atom(Atom::new(CompareOp::Ge, Term::var("v"), Term::float(lo))),
                Predicate::Atom(Atom::new(CompareOp::Le, Term::var("v"), Term::float(hi))),
            ]),
        )
    }

    /// Substitute a variable in the predicate
    pub fn substitute(&self, from: &str, to: &Term) -> Self {
        Self {
            base: self.base.clone(),
            var: self.var.clone(),
            predicate: self.predicate.substitute(from, to),
        }
    }

    /// Get free variables in the predicate (excluding the bound variable)
    pub fn free_vars(&self) -> HashSet<String> {
        let mut vars = self.predicate.free_vars();
        vars.remove(&self.var);
        vars
    }

    /// Check if this type has a non-trivial refinement
    pub fn is_refined(&self) -> bool {
        self.predicate != Predicate::True
    }
}

impl fmt::Display for RefinementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.predicate == Predicate::True {
            write!(f, "{:?}", self.base)
        } else {
            write!(
                f,
                "{{ {}: {:?} | {} }}",
                self.var, self.base, self.predicate
            )
        }
    }
}

/// Predicate (logical formula)
///
/// Represents logical formulas that can be checked by the SMT solver.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    /// Always true
    True,

    /// Always false
    False,

    /// Atomic comparison: `lhs op rhs`
    Atom(Atom),

    /// Negation: `¬P`
    Not(Box<Predicate>),

    /// Conjunction: `P₁ ∧ P₂ ∧ ... ∧ Pₙ`
    And(Vec<Predicate>),

    /// Disjunction: `P₁ ∨ P₂ ∨ ... ∨ Pₙ`
    Or(Vec<Predicate>),

    /// Implication: `P ⟹ Q`
    Implies(Box<Predicate>, Box<Predicate>),

    /// Universal quantification: `∀x: T. P`
    Forall(String, Type, Box<Predicate>),

    /// Existential quantification: `∃x: T. P`
    Exists(String, Type, Box<Predicate>),

    /// Predicate application (for abstract refinements): `p(t₁, ..., tₙ)`
    App(String, Vec<Term>),

    /// If-then-else: `if C then P else Q`
    Ite(Box<Predicate>, Box<Predicate>, Box<Predicate>),
}

impl Predicate {
    /// Create a conjunction, simplifying trivial cases
    pub fn and(preds: impl IntoIterator<Item = Predicate>) -> Self {
        let preds: Vec<_> = preds
            .into_iter()
            .filter(|p| *p != Predicate::True)
            .collect();

        if preds.is_empty() {
            Predicate::True
        } else if preds.len() == 1 {
            preds.into_iter().next().unwrap()
        } else if preds.contains(&Predicate::False) {
            Predicate::False
        } else {
            Predicate::And(preds)
        }
    }

    /// Create a disjunction, simplifying trivial cases
    pub fn or(preds: impl IntoIterator<Item = Predicate>) -> Self {
        let preds: Vec<_> = preds
            .into_iter()
            .filter(|p| *p != Predicate::False)
            .collect();

        if preds.is_empty() {
            Predicate::False
        } else if preds.len() == 1 {
            preds.into_iter().next().unwrap()
        } else if preds.contains(&Predicate::True) {
            Predicate::True
        } else {
            Predicate::Or(preds)
        }
    }

    /// Create an implication, simplifying trivial cases
    pub fn implies(p: Predicate, q: Predicate) -> Self {
        match (&p, &q) {
            (Predicate::True, _) => q,
            (_, Predicate::True) => Predicate::True,
            (Predicate::False, _) => Predicate::True,
            _ => Predicate::Implies(Box::new(p), Box::new(q)),
        }
    }

    /// Create a negation, simplifying double negation
    pub fn not(p: Predicate) -> Self {
        match p {
            Predicate::True => Predicate::False,
            Predicate::False => Predicate::True,
            Predicate::Not(inner) => *inner,
            other => Predicate::Not(Box::new(other)),
        }
    }

    /// Negate this predicate
    pub fn negate(self) -> Self {
        Self::not(self)
    }

    /// Create an equality comparison: `lhs = rhs`
    pub fn eq(lhs: Term, rhs: Term) -> Self {
        Predicate::Atom(Atom::new(CompareOp::Eq, lhs, rhs))
    }

    /// Create a less-than comparison: `lhs < rhs`
    pub fn lt(lhs: Term, rhs: Term) -> Self {
        Predicate::Atom(Atom::new(CompareOp::Lt, lhs, rhs))
    }

    /// Create a less-than-or-equal comparison: `lhs <= rhs`
    pub fn le(lhs: Term, rhs: Term) -> Self {
        Predicate::Atom(Atom::new(CompareOp::Le, lhs, rhs))
    }

    /// Create a greater-than comparison: `lhs > rhs`
    pub fn gt(lhs: Term, rhs: Term) -> Self {
        Predicate::Atom(Atom::new(CompareOp::Gt, lhs, rhs))
    }

    /// Create a greater-than-or-equal comparison: `lhs >= rhs`
    pub fn ge(lhs: Term, rhs: Term) -> Self {
        Predicate::Atom(Atom::new(CompareOp::Ge, lhs, rhs))
    }

    /// Create a not-equal comparison: `lhs != rhs`
    pub fn ne(lhs: Term, rhs: Term) -> Self {
        Predicate::Atom(Atom::new(CompareOp::Ne, lhs, rhs))
    }

    /// Substitute a variable with a term throughout the predicate
    pub fn substitute(&self, from: &str, to: &Term) -> Self {
        match self {
            Predicate::True => Predicate::True,
            Predicate::False => Predicate::False,

            Predicate::Atom(atom) => Predicate::Atom(atom.substitute(from, to)),

            Predicate::Not(p) => Predicate::Not(Box::new(p.substitute(from, to))),

            Predicate::And(ps) => {
                Predicate::And(ps.iter().map(|p| p.substitute(from, to)).collect())
            }

            Predicate::Or(ps) => Predicate::Or(ps.iter().map(|p| p.substitute(from, to)).collect()),

            Predicate::Implies(p, q) => Predicate::Implies(
                Box::new(p.substitute(from, to)),
                Box::new(q.substitute(from, to)),
            ),

            Predicate::Forall(x, ty, p) if x != from => {
                Predicate::Forall(x.clone(), ty.clone(), Box::new(p.substitute(from, to)))
            }

            Predicate::Exists(x, ty, p) if x != from => {
                Predicate::Exists(x.clone(), ty.clone(), Box::new(p.substitute(from, to)))
            }

            Predicate::App(name, args) => Predicate::App(
                name.clone(),
                args.iter().map(|a| a.substitute(from, to)).collect(),
            ),

            Predicate::Ite(c, t, e) => Predicate::Ite(
                Box::new(c.substitute(from, to)),
                Box::new(t.substitute(from, to)),
                Box::new(e.substitute(from, to)),
            ),

            // Variable is bound, don't substitute
            other => other.clone(),
        }
    }

    /// Get all free variables in this predicate
    pub fn free_vars(&self) -> HashSet<String> {
        match self {
            Predicate::True | Predicate::False => HashSet::new(),

            Predicate::Atom(atom) => atom.free_vars(),

            Predicate::Not(p) => p.free_vars(),

            Predicate::And(ps) | Predicate::Or(ps) => {
                ps.iter().flat_map(|p| p.free_vars()).collect()
            }

            Predicate::Implies(p, q) => {
                let mut vars = p.free_vars();
                vars.extend(q.free_vars());
                vars
            }

            Predicate::Forall(x, _, p) | Predicate::Exists(x, _, p) => {
                let mut vars = p.free_vars();
                vars.remove(x);
                vars
            }

            Predicate::App(_, args) => args.iter().flat_map(|a| a.free_vars()).collect(),

            Predicate::Ite(c, t, e) => {
                let mut vars = c.free_vars();
                vars.extend(t.free_vars());
                vars.extend(e.free_vars());
                vars
            }
        }
    }

    /// Check if this predicate is trivially true
    pub fn is_trivially_true(&self) -> bool {
        matches!(self, Predicate::True)
    }

    /// Check if this predicate is trivially false
    pub fn is_trivially_false(&self) -> bool {
        matches!(self, Predicate::False)
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Predicate::True => write!(f, "true"),
            Predicate::False => write!(f, "false"),
            Predicate::Atom(a) => write!(f, "{}", a),
            Predicate::Not(p) => write!(f, "not({})", p),
            Predicate::And(ps) => {
                let s: Vec<_> = ps.iter().map(|p| format!("{}", p)).collect();
                write!(f, "({})", s.join(" && "))
            }
            Predicate::Or(ps) => {
                let s: Vec<_> = ps.iter().map(|p| format!("{}", p)).collect();
                write!(f, "({})", s.join(" || "))
            }
            Predicate::Implies(p, q) => write!(f, "({} => {})", p, q),
            Predicate::Forall(x, ty, p) => write!(f, "forall {}: {:?}. {}", x, ty, p),
            Predicate::Exists(x, ty, p) => write!(f, "exists {}: {:?}. {}", x, ty, p),
            Predicate::App(name, args) => {
                let args_str: Vec<_> = args.iter().map(|a| format!("{}", a)).collect();
                write!(f, "{}({})", name, args_str.join(", "))
            }
            Predicate::Ite(c, t, e) => write!(f, "if {} then {} else {}", c, t, e),
        }
    }
}

/// Atomic predicate (comparison between terms)
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    /// Comparison operator
    pub op: CompareOp,
    /// Left-hand side term
    pub lhs: Term,
    /// Right-hand side term
    pub rhs: Term,
}

impl Atom {
    /// Create a new atomic predicate
    pub fn new(op: CompareOp, lhs: Term, rhs: Term) -> Self {
        Self { op, lhs, rhs }
    }

    /// Substitute a variable with a term
    pub fn substitute(&self, from: &str, to: &Term) -> Self {
        Self {
            op: self.op,
            lhs: self.lhs.substitute(from, to),
            rhs: self.rhs.substitute(from, to),
        }
    }

    /// Get free variables
    pub fn free_vars(&self) -> HashSet<String> {
        let mut vars = self.lhs.free_vars();
        vars.extend(self.rhs.free_vars());
        vars
    }

    /// Negate this atom (flip the comparison)
    pub fn negate(&self) -> Self {
        Self {
            op: self.op.negate(),
            lhs: self.lhs.clone(),
            rhs: self.rhs.clone(),
        }
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.lhs, self.op, self.rhs)
    }
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    /// Equal: `=`
    Eq,
    /// Not equal: `!=`
    Ne,
    /// Less than: `<`
    Lt,
    /// Less than or equal: `<=`
    Le,
    /// Greater than: `>`
    Gt,
    /// Greater than or equal: `>=`
    Ge,
}

impl CompareOp {
    /// Negate this comparison operator
    pub fn negate(self) -> Self {
        match self {
            CompareOp::Eq => CompareOp::Ne,
            CompareOp::Ne => CompareOp::Eq,
            CompareOp::Lt => CompareOp::Ge,
            CompareOp::Le => CompareOp::Gt,
            CompareOp::Gt => CompareOp::Le,
            CompareOp::Ge => CompareOp::Lt,
        }
    }

    /// Flip this comparison (swap operands)
    pub fn flip(self) -> Self {
        match self {
            CompareOp::Eq => CompareOp::Eq,
            CompareOp::Ne => CompareOp::Ne,
            CompareOp::Lt => CompareOp::Gt,
            CompareOp::Le => CompareOp::Ge,
            CompareOp::Gt => CompareOp::Lt,
            CompareOp::Ge => CompareOp::Le,
        }
    }
}

impl fmt::Display for CompareOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompareOp::Eq => write!(f, "="),
            CompareOp::Ne => write!(f, "!="),
            CompareOp::Lt => write!(f, "<"),
            CompareOp::Le => write!(f, "<="),
            CompareOp::Gt => write!(f, ">"),
            CompareOp::Ge => write!(f, ">="),
        }
    }
}

/// Term (expression in predicates)
///
/// Terms represent values that can appear in predicates.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// Variable reference
    Var(String),

    /// Integer constant
    Int(i64),

    /// Floating-point constant
    Float(f64),

    /// Boolean constant
    Bool(bool),

    /// Binary arithmetic operation
    BinOp(BinOp, Box<Term>, Box<Term>),

    /// Unary operation
    UnOp(UnOp, Box<Term>),

    /// Function/predicate application
    App(String, Vec<Term>),

    /// Field access: `base.field`
    Field(Box<Term>, String),

    /// Array/sequence length: `len(t)`
    Len(Box<Term>),

    /// Conditional expression: `if c then t else e`
    Ite(Box<Term>, Box<Term>, Box<Term>),
}

impl Term {
    /// Create a variable term
    pub fn var(name: impl Into<String>) -> Self {
        Term::Var(name.into())
    }

    /// Create an integer constant
    pub fn int(n: i64) -> Self {
        Term::Int(n)
    }

    /// Create a floating-point constant
    pub fn float(n: f64) -> Self {
        Term::Float(n)
    }

    /// Create a boolean constant
    pub fn bool(b: bool) -> Self {
        Term::Bool(b)
    }

    /// Create an addition: `lhs + rhs`
    pub fn add(lhs: Term, rhs: Term) -> Self {
        Term::BinOp(BinOp::Add, Box::new(lhs), Box::new(rhs))
    }

    /// Create a subtraction: `lhs - rhs`
    pub fn sub(lhs: Term, rhs: Term) -> Self {
        Term::BinOp(BinOp::Sub, Box::new(lhs), Box::new(rhs))
    }

    /// Create a multiplication: `lhs * rhs`
    pub fn mul(lhs: Term, rhs: Term) -> Self {
        Term::BinOp(BinOp::Mul, Box::new(lhs), Box::new(rhs))
    }

    /// Create a division: `lhs / rhs`
    pub fn div(lhs: Term, rhs: Term) -> Self {
        Term::BinOp(BinOp::Div, Box::new(lhs), Box::new(rhs))
    }

    /// Create a modulo: `lhs % rhs`
    pub fn modulo(lhs: Term, rhs: Term) -> Self {
        Term::BinOp(BinOp::Mod, Box::new(lhs), Box::new(rhs))
    }

    /// Create a negation: `-t`
    pub fn neg(t: Term) -> Self {
        Term::UnOp(UnOp::Neg, Box::new(t))
    }

    /// Create an absolute value: `abs(t)`
    pub fn abs(t: Term) -> Self {
        Term::UnOp(UnOp::Abs, Box::new(t))
    }

    /// Create a length expression: `len(t)`
    pub fn len(t: Term) -> Self {
        Term::Len(Box::new(t))
    }

    /// Create a field access: `base.field`
    pub fn field(base: Term, field: impl Into<String>) -> Self {
        Term::Field(Box::new(base), field.into())
    }

    /// Substitute a variable with a term
    pub fn substitute(&self, from: &str, to: &Term) -> Self {
        match self {
            Term::Var(x) if x == from => to.clone(),
            Term::Var(_) | Term::Int(_) | Term::Float(_) | Term::Bool(_) => self.clone(),

            Term::BinOp(op, l, r) => Term::BinOp(
                *op,
                Box::new(l.substitute(from, to)),
                Box::new(r.substitute(from, to)),
            ),

            Term::UnOp(op, t) => Term::UnOp(*op, Box::new(t.substitute(from, to))),

            Term::App(name, args) => Term::App(
                name.clone(),
                args.iter().map(|a| a.substitute(from, to)).collect(),
            ),

            Term::Field(base, field) => {
                Term::Field(Box::new(base.substitute(from, to)), field.clone())
            }

            Term::Len(t) => Term::Len(Box::new(t.substitute(from, to))),

            Term::Ite(c, t, e) => Term::Ite(
                Box::new(c.substitute(from, to)),
                Box::new(t.substitute(from, to)),
                Box::new(e.substitute(from, to)),
            ),
        }
    }

    /// Get all free variables in this term
    pub fn free_vars(&self) -> HashSet<String> {
        match self {
            Term::Var(x) => {
                let mut set = HashSet::new();
                set.insert(x.clone());
                set
            }
            Term::Int(_) | Term::Float(_) | Term::Bool(_) => HashSet::new(),
            Term::BinOp(_, l, r) => {
                let mut vars = l.free_vars();
                vars.extend(r.free_vars());
                vars
            }
            Term::UnOp(_, t) => t.free_vars(),
            Term::App(_, args) => args.iter().flat_map(|a| a.free_vars()).collect(),
            Term::Field(base, _) => base.free_vars(),
            Term::Len(t) => t.free_vars(),
            Term::Ite(c, t, e) => {
                let mut vars = c.free_vars();
                vars.extend(t.free_vars());
                vars.extend(e.free_vars());
                vars
            }
        }
    }

    /// Check if this term is a constant (no variables)
    pub fn is_constant(&self) -> bool {
        self.free_vars().is_empty()
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Var(x) => write!(f, "{}", x),
            Term::Int(n) => write!(f, "{}", n),
            Term::Float(n) => write!(f, "{:.6}", n),
            Term::Bool(b) => write!(f, "{}", b),
            Term::BinOp(op, l, r) => write!(f, "({} {} {})", l, op, r),
            Term::UnOp(op, t) => write!(f, "{}({})", op, t),
            Term::App(name, args) => {
                let args_str: Vec<_> = args.iter().map(|a| format!("{}", a)).collect();
                write!(f, "{}({})", name, args_str.join(", "))
            }
            Term::Field(base, field) => write!(f, "{}.{}", base, field),
            Term::Len(t) => write!(f, "len({})", t),
            Term::Ite(c, t, e) => write!(f, "(if {} then {} else {})", c, t, e),
        }
    }
}

/// Binary operators for terms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Modulo
    Mod,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
        }
    }
}

/// Unary operators for terms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    /// Negation: `-x`
    Neg,
    /// Logical not: `!x`
    Not,
    /// Absolute value: `abs(x)`
    Abs,
}

impl fmt::Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnOp::Neg => write!(f, "-"),
            UnOp::Not => write!(f, "!"),
            UnOp::Abs => write!(f, "abs"),
        }
    }
}

/// Built-in refinement types for the medical domain
pub mod medical {
    use super::*;

    /// Positive number: `{ v | v > 0 }`
    pub fn positive(base: Type) -> RefinementType {
        RefinementType::positive(base)
    }

    /// Non-negative number: `{ v | v >= 0 }`
    pub fn non_negative(base: Type) -> RefinementType {
        RefinementType::non_negative(base)
    }

    /// Bounded range: `{ v | lo <= v <= hi }`
    pub fn bounded(base: Type, lo: f64, hi: f64) -> RefinementType {
        RefinementType::bounded(base, lo, hi)
    }

    /// Probability: `{ v | 0 <= v <= 1 }`
    pub fn probability(base: Type) -> RefinementType {
        bounded(base, 0.0, 1.0)
    }

    /// Safe dose: `{ dose | 0 < dose <= max }`
    pub fn safe_dose(base: Type, max: f64) -> RefinementType {
        RefinementType::refined(
            base,
            "dose",
            Predicate::and([
                Predicate::Atom(Atom::new(
                    CompareOp::Gt,
                    Term::var("dose"),
                    Term::float(0.0),
                )),
                Predicate::Atom(Atom::new(
                    CompareOp::Le,
                    Term::var("dose"),
                    Term::float(max),
                )),
            ]),
        )
    }

    /// Valid creatinine clearance: `{ crcl | 0 < crcl < 200 mL/min }`
    pub fn valid_crcl(base: Type) -> RefinementType {
        RefinementType::refined(
            base,
            "crcl",
            Predicate::and([
                Predicate::Atom(Atom::new(
                    CompareOp::Gt,
                    Term::var("crcl"),
                    Term::float(0.0),
                )),
                Predicate::Atom(Atom::new(
                    CompareOp::Lt,
                    Term::var("crcl"),
                    Term::float(200.0),
                )),
            ]),
        )
    }

    /// Valid age: `{ age | 0 <= age <= 150 }`
    pub fn valid_age(base: Type) -> RefinementType {
        bounded(base, 0.0, 150.0)
    }

    /// Valid weight: `{ weight | 0 < weight <= 500 kg }`
    pub fn valid_weight(base: Type) -> RefinementType {
        RefinementType::refined(
            base,
            "weight",
            Predicate::and([
                Predicate::Atom(Atom::new(
                    CompareOp::Gt,
                    Term::var("weight"),
                    Term::float(0.0),
                )),
                Predicate::Atom(Atom::new(
                    CompareOp::Le,
                    Term::var("weight"),
                    Term::float(500.0),
                )),
            ]),
        )
    }

    /// Valid serum creatinine: `{ scr | 0.1 <= scr <= 20 mg/dL }`
    pub fn valid_serum_creatinine(base: Type) -> RefinementType {
        bounded(base, 0.1, 20.0)
    }

    /// Therapeutic range: `{ conc | min <= conc <= max }`
    pub fn therapeutic_range(base: Type, min: f64, max: f64) -> RefinementType {
        bounded(base, min, max)
    }

    /// Valid heart rate: `{ hr | 20 <= hr <= 300 bpm }`
    pub fn valid_heart_rate(base: Type) -> RefinementType {
        bounded(base, 20.0, 300.0)
    }

    /// Valid blood pressure (systolic): `{ bp | 40 <= bp <= 300 mmHg }`
    pub fn valid_systolic_bp(base: Type) -> RefinementType {
        bounded(base, 40.0, 300.0)
    }

    /// Valid blood pressure (diastolic): `{ bp | 20 <= bp <= 200 mmHg }`
    pub fn valid_diastolic_bp(base: Type) -> RefinementType {
        bounded(base, 20.0, 200.0)
    }

    /// Valid temperature: `{ temp | 25 <= temp <= 45 C }`
    pub fn valid_temperature(base: Type) -> RefinementType {
        bounded(base, 25.0, 45.0)
    }

    /// Dose adjustment factor: `{ factor | 0 < factor <= 1 }`
    pub fn adjustment_factor(base: Type) -> RefinementType {
        RefinementType::refined(
            base,
            "factor",
            Predicate::and([
                Predicate::Atom(Atom::new(
                    CompareOp::Gt,
                    Term::var("factor"),
                    Term::float(0.0),
                )),
                Predicate::Atom(Atom::new(
                    CompareOp::Le,
                    Term::var("factor"),
                    Term::float(1.0),
                )),
            ]),
        )
    }
}

/// Built-in refinement types for arrays
pub mod array {
    use super::*;

    /// Non-empty array predicate: `len(arr) > 0`
    pub fn non_empty() -> Predicate {
        Predicate::gt(
            Term::App("len".to_string(), vec![Term::var("arr")]),
            Term::int(0),
        )
    }

    /// Array with minimum length: `len(arr) >= min`
    pub fn min_length(min: i64) -> Predicate {
        Predicate::ge(
            Term::App("len".to_string(), vec![Term::var("arr")]),
            Term::int(min),
        )
    }

    /// Array with exact length: `len(arr) = len`
    pub fn exact_length(len: i64) -> Predicate {
        Predicate::eq(
            Term::App("len".to_string(), vec![Term::var("arr")]),
            Term::int(len),
        )
    }

    /// Valid index for array: `0 <= idx < len(arr)`
    pub fn valid_index(arr_var: &str, idx_var: &str) -> Predicate {
        Predicate::and([
            Predicate::ge(Term::var(idx_var), Term::int(0)),
            Predicate::lt(
                Term::var(idx_var),
                Term::App("len".to_string(), vec![Term::var(arr_var)]),
            ),
        ])
    }

    /// Bounded index: `0 <= idx < bound`
    pub fn bounded_index(idx_var: &str, bound: i64) -> Predicate {
        Predicate::and([
            Predicate::ge(Term::var(idx_var), Term::int(0)),
            Predicate::lt(Term::var(idx_var), Term::int(bound)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_refinement() {
        let ty = RefinementType::trivial(Type::I64);
        assert!(!ty.is_refined());
        assert_eq!(ty.predicate, Predicate::True);
    }

    #[test]
    fn test_positive_refinement() {
        let ty = RefinementType::positive(Type::I64);
        assert!(ty.is_refined());
        assert!(ty.free_vars().is_empty());
    }

    #[test]
    fn test_predicate_and_simplification() {
        // true && P = P
        let p = Predicate::gt(Term::var("x"), Term::int(0));
        let result = Predicate::and([Predicate::True, p.clone()]);
        assert_eq!(result, p);

        // P && false = false
        let result = Predicate::and([p.clone(), Predicate::False]);
        assert_eq!(result, Predicate::False);

        // P && true && Q = P && Q
        let q = Predicate::lt(Term::var("x"), Term::int(100));
        let result = Predicate::and([p.clone(), Predicate::True, q.clone()]);
        assert!(matches!(result, Predicate::And(_)));
    }

    #[test]
    fn test_predicate_or_simplification() {
        // false || P = P
        let p = Predicate::gt(Term::var("x"), Term::int(0));
        let result = Predicate::or([Predicate::False, p.clone()]);
        assert_eq!(result, p);

        // P || true = true
        let result = Predicate::or([p.clone(), Predicate::True]);
        assert_eq!(result, Predicate::True);
    }

    #[test]
    fn test_term_substitution() {
        let term = Term::add(Term::var("x"), Term::int(1));
        let result = term.substitute("x", &Term::int(5));
        assert_eq!(result, Term::add(Term::int(5), Term::int(1)));
    }

    #[test]
    fn test_predicate_substitution() {
        let pred = Predicate::gt(Term::var("x"), Term::int(0));
        let result = pred.substitute("x", &Term::var("y"));

        let expected = Predicate::gt(Term::var("y"), Term::int(0));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_free_vars() {
        let pred = Predicate::and([
            Predicate::gt(Term::var("x"), Term::int(0)),
            Predicate::lt(Term::var("y"), Term::var("z")),
        ]);

        let vars = pred.free_vars();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));
        assert_eq!(vars.len(), 3);
    }

    #[test]
    fn test_compare_op_negate() {
        assert_eq!(CompareOp::Eq.negate(), CompareOp::Ne);
        assert_eq!(CompareOp::Lt.negate(), CompareOp::Ge);
        assert_eq!(CompareOp::Le.negate(), CompareOp::Gt);
        assert_eq!(CompareOp::Gt.negate(), CompareOp::Le);
        assert_eq!(CompareOp::Ge.negate(), CompareOp::Lt);
    }

    #[test]
    fn test_medical_refinements() {
        let dose_ty = medical::safe_dose(Type::F64, 1000.0);
        assert!(dose_ty.is_refined());
        assert_eq!(dose_ty.var, "dose");

        let crcl_ty = medical::valid_crcl(Type::F64);
        assert!(crcl_ty.is_refined());
        assert_eq!(crcl_ty.var, "crcl");
    }

    #[test]
    fn test_array_predicates() {
        let non_empty = array::non_empty();
        assert!(non_empty.free_vars().contains("arr"));

        let valid_idx = array::valid_index("arr", "i");
        let vars = valid_idx.free_vars();
        assert!(vars.contains("arr"));
        assert!(vars.contains("i"));
    }

    #[test]
    fn test_refinement_type_display() {
        let ty = RefinementType::positive(Type::I64);
        let display = format!("{}", ty);
        assert!(display.contains("v"));
        assert!(display.contains(">"));
        assert!(display.contains("0"));
    }
}
