//! Symbolic Computation Runtime
//!
//! This module provides symbolic mathematics for Sounio, enabling
//! computer algebra operations as first-class primitives.
//!
//! # Core Operations
//!
//! - `symbolic(name)` - Create symbolic variable
//! - `simplify(expr)` - Simplify expression
//! - `expand(expr)` - Expand expression
//! - `factor(expr)` - Factor expression
//! - `diff(expr, var)` - Symbolic differentiation
//! - `integrate(expr, var)` - Symbolic integration
//! - `solve_symbolic(equation, var)` - Solve equation symbolically
//! - `compile(expr)` - Convert to numerical function
//!
//! # Example
//!
//! ```d
//! let x = symbolic("x");
//! let expr = x^2 + 2*x + 1;
//! let simplified = simplify(expr);  // (x + 1)^2
//! let deriv = diff(expr, x);        // 2*x + 2
//! let solutions = solve_symbolic(expr == 0, x);  // [-1]
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// Symbolic expression
#[derive(Clone, PartialEq)]
pub enum Expr {
    /// Numeric constant
    Const(f64),
    /// Symbolic variable
    Symbol(String),
    /// Addition
    Add(Rc<Expr>, Rc<Expr>),
    /// Subtraction
    Sub(Rc<Expr>, Rc<Expr>),
    /// Multiplication
    Mul(Rc<Expr>, Rc<Expr>),
    /// Division
    Div(Rc<Expr>, Rc<Expr>),
    /// Power
    Pow(Rc<Expr>, Rc<Expr>),
    /// Negation
    Neg(Rc<Expr>),
    /// Function application
    Fn(String, Vec<Rc<Expr>>),
    /// Derivative (unevaluated)
    Derivative(Rc<Expr>, String),
    /// Integral (unevaluated)
    Integral(Rc<Expr>, String),
}

impl Expr {
    /// Create a constant
    pub fn constant(value: f64) -> Self {
        Expr::Const(value)
    }

    /// Create a symbol
    pub fn symbol(name: &str) -> Self {
        Expr::Symbol(name.to_string())
    }

    /// Create zero
    pub fn zero() -> Self {
        Expr::Const(0.0)
    }

    /// Create one
    pub fn one() -> Self {
        Expr::Const(1.0)
    }

    /// Check if expression is zero
    pub fn is_zero(&self) -> bool {
        matches!(self, Expr::Const(x) if x.abs() < 1e-15)
    }

    /// Check if expression is one
    pub fn is_one(&self) -> bool {
        matches!(self, Expr::Const(x) if (x - 1.0).abs() < 1e-15)
    }

    /// Check if expression is a constant
    pub fn is_const(&self) -> bool {
        matches!(self, Expr::Const(_))
    }

    /// Get constant value if this is a constant
    pub fn as_const(&self) -> Option<f64> {
        match self {
            Expr::Const(x) => Some(*x),
            _ => None,
        }
    }

    /// Check if expression contains a variable
    pub fn contains_var(&self, var: &str) -> bool {
        match self {
            Expr::Const(_) => false,
            Expr::Symbol(s) => s == var,
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => a.contains_var(var) || b.contains_var(var),
            Expr::Neg(a) => a.contains_var(var),
            Expr::Fn(_, args) => args.iter().any(|a| a.contains_var(var)),
            Expr::Derivative(e, _) | Expr::Integral(e, _) => e.contains_var(var),
        }
    }

    /// Get all free variables in the expression
    pub fn free_vars(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut HashSet<String>) {
        match self {
            Expr::Const(_) => {}
            Expr::Symbol(s) => {
                vars.insert(s.clone());
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }
            Expr::Neg(a) => a.collect_vars(vars),
            Expr::Fn(_, args) => {
                for arg in args {
                    arg.collect_vars(vars);
                }
            }
            Expr::Derivative(e, _) | Expr::Integral(e, _) => e.collect_vars(vars),
        }
    }

    /// Power operation
    pub fn pow(self, exp: Expr) -> Expr {
        Expr::Pow(Rc::new(self), Rc::new(exp))
    }

    /// Square root
    pub fn sqrt(self) -> Expr {
        Expr::Pow(Rc::new(self), Rc::new(Expr::Const(0.5)))
    }

    /// Sine
    pub fn sin(self) -> Expr {
        Expr::Fn("sin".to_string(), vec![Rc::new(self)])
    }

    /// Cosine
    pub fn cos(self) -> Expr {
        Expr::Fn("cos".to_string(), vec![Rc::new(self)])
    }

    /// Tangent
    pub fn tan(self) -> Expr {
        Expr::Fn("tan".to_string(), vec![Rc::new(self)])
    }

    /// Exponential
    pub fn exp(self) -> Expr {
        Expr::Fn("exp".to_string(), vec![Rc::new(self)])
    }

    /// Natural logarithm
    pub fn ln(self) -> Expr {
        Expr::Fn("ln".to_string(), vec![Rc::new(self)])
    }

    /// Absolute value
    pub fn abs(self) -> Expr {
        Expr::Fn("abs".to_string(), vec![Rc::new(self)])
    }
}

// Operator overloading for Expr
impl Add for Expr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Expr::Add(Rc::new(self), Rc::new(rhs))
    }
}

impl Sub for Expr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Expr::Sub(Rc::new(self), Rc::new(rhs))
    }
}

impl Mul for Expr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Expr::Mul(Rc::new(self), Rc::new(rhs))
    }
}

impl Div for Expr {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Expr::Div(Rc::new(self), Rc::new(rhs))
    }
}

impl Neg for Expr {
    type Output = Self;
    fn neg(self) -> Self {
        Expr::Neg(Rc::new(self))
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(x) => {
                if x.fract().abs() < 1e-10 {
                    write!(f, "{}", *x as i64)
                } else {
                    write!(f, "{}", x)
                }
            }
            Expr::Symbol(s) => write!(f, "{}", s),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Pow(a, b) => write!(f, "({})^({})", a, b),
            Expr::Neg(a) => write!(f, "(-{})", a),
            Expr::Fn(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Expr::Derivative(e, var) => write!(f, "d/d{}[{}]", var, e),
            Expr::Integral(e, var) => write!(f, "∫{}d{}", e, var),
        }
    }
}

/// Simplify an expression
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::Const(_) | Expr::Symbol(_) => expr.clone(),

        // Addition simplifications
        Expr::Add(a, b) => {
            let a = simplify(a);
            let b = simplify(b);

            // 0 + x = x
            if a.is_zero() {
                return b;
            }
            // x + 0 = x
            if b.is_zero() {
                return a;
            }
            // const + const
            if let (Some(ca), Some(cb)) = (a.as_const(), b.as_const()) {
                return Expr::Const(ca + cb);
            }
            // x + x = 2x
            if a == b {
                return simplify(&(Expr::Const(2.0) * a));
            }
            Expr::Add(Rc::new(a), Rc::new(b))
        }

        // Subtraction simplifications
        Expr::Sub(a, b) => {
            let a = simplify(a);
            let b = simplify(b);

            // x - 0 = x
            if b.is_zero() {
                return a;
            }
            // 0 - x = -x
            if a.is_zero() {
                return simplify(&(-b));
            }
            // x - x = 0
            if a == b {
                return Expr::zero();
            }
            // const - const
            if let (Some(ca), Some(cb)) = (a.as_const(), b.as_const()) {
                return Expr::Const(ca - cb);
            }
            Expr::Sub(Rc::new(a), Rc::new(b))
        }

        // Multiplication simplifications
        Expr::Mul(a, b) => {
            let a = simplify(a);
            let b = simplify(b);

            // 0 * x = 0
            if a.is_zero() || b.is_zero() {
                return Expr::zero();
            }
            // 1 * x = x
            if a.is_one() {
                return b;
            }
            // x * 1 = x
            if b.is_one() {
                return a;
            }
            // const * const
            if let (Some(ca), Some(cb)) = (a.as_const(), b.as_const()) {
                return Expr::Const(ca * cb);
            }
            // x * x = x^2
            if a == b {
                return Expr::Pow(Rc::new(a), Rc::new(Expr::Const(2.0)));
            }
            Expr::Mul(Rc::new(a), Rc::new(b))
        }

        // Division simplifications
        Expr::Div(a, b) => {
            let a = simplify(a);
            let b = simplify(b);

            // 0 / x = 0 (assuming x != 0)
            if a.is_zero() {
                return Expr::zero();
            }
            // x / 1 = x
            if b.is_one() {
                return a;
            }
            // x / x = 1 (assuming x != 0)
            if a == b {
                return Expr::one();
            }
            // const / const
            if let (Some(ca), Some(cb)) = (a.as_const(), b.as_const())
                && cb.abs() > 1e-15
            {
                return Expr::Const(ca / cb);
            }
            Expr::Div(Rc::new(a), Rc::new(b))
        }

        // Power simplifications
        Expr::Pow(a, b) => {
            let a = simplify(a);
            let b = simplify(b);

            // x^0 = 1
            if b.is_zero() {
                return Expr::one();
            }
            // x^1 = x
            if b.is_one() {
                return a;
            }
            // 0^x = 0 (for x > 0)
            if a.is_zero() {
                return Expr::zero();
            }
            // 1^x = 1
            if a.is_one() {
                return Expr::one();
            }
            // const^const
            if let (Some(ca), Some(cb)) = (a.as_const(), b.as_const()) {
                return Expr::Const(ca.powf(cb));
            }
            Expr::Pow(Rc::new(a), Rc::new(b))
        }

        // Negation simplifications
        Expr::Neg(a) => {
            let a = simplify(a);
            // -(-x) = x
            if let Expr::Neg(inner) = &a {
                return (**inner).clone();
            }
            // -const
            if let Some(c) = a.as_const() {
                return Expr::Const(-c);
            }
            Expr::Neg(Rc::new(a))
        }

        // Function simplifications
        Expr::Fn(name, args) => {
            let args: Vec<Rc<Expr>> = args.iter().map(|a| Rc::new(simplify(a))).collect();

            // Evaluate constant functions
            if args.len() == 1
                && let Some(x) = args[0].as_const()
            {
                match name.as_str() {
                    "sin" => return Expr::Const(x.sin()),
                    "cos" => return Expr::Const(x.cos()),
                    "tan" => return Expr::Const(x.tan()),
                    "exp" => return Expr::Const(x.exp()),
                    "ln" => return Expr::Const(x.ln()),
                    "abs" => return Expr::Const(x.abs()),
                    "sqrt" => return Expr::Const(x.sqrt()),
                    _ => {}
                }
            }

            Expr::Fn(name.clone(), args)
        }

        Expr::Derivative(e, var) => {
            // Evaluate the derivative
            differentiate(&simplify(e), var)
        }

        Expr::Integral(e, var) => {
            // Try to evaluate the integral
            if let Some(result) = integrate(&simplify(e), var) {
                result
            } else {
                Expr::Integral(Rc::new(simplify(e)), var.clone())
            }
        }
    }
}

/// Expand an expression (distribute multiplication over addition)
pub fn expand(expr: &Expr) -> Expr {
    match expr {
        Expr::Const(_) | Expr::Symbol(_) => expr.clone(),

        Expr::Add(a, b) => Expr::Add(Rc::new(expand(a)), Rc::new(expand(b))),

        Expr::Sub(a, b) => Expr::Sub(Rc::new(expand(a)), Rc::new(expand(b))),

        // (a + b) * c = a*c + b*c
        Expr::Mul(a, b) => {
            let a_exp = expand(a);
            let b_exp = expand(b);

            match (&a_exp, &b_exp) {
                (Expr::Add(a1, a2), _) => {
                    expand(&((**a1).clone() * b_exp.clone() + (**a2).clone() * b_exp))
                }
                (_, Expr::Add(b1, b2)) => {
                    expand(&(a_exp.clone() * (**b1).clone() + a_exp * (**b2).clone()))
                }
                (Expr::Sub(a1, a2), _) => {
                    expand(&((**a1).clone() * b_exp.clone() - (**a2).clone() * b_exp))
                }
                (_, Expr::Sub(b1, b2)) => {
                    expand(&(a_exp.clone() * (**b1).clone() - a_exp * (**b2).clone()))
                }
                _ => Expr::Mul(Rc::new(a_exp), Rc::new(b_exp)),
            }
        }

        // (a + b)^n expansion (binomial for small integer n)
        Expr::Pow(a, b) => {
            let a_exp = expand(a);
            if let Some(n) = b.as_const()
                && n > 0.0
                && n == n.floor()
                && n <= 5.0
            {
                let n = n as usize;
                if let Expr::Add(x, y) = &a_exp {
                    // Binomial expansion
                    let mut result = Expr::zero();
                    for k in 0..=n {
                        let coef = binomial(n, k) as f64;
                        let x_pow = if n - k == 0 {
                            Expr::one()
                        } else {
                            Expr::Pow(x.clone(), Rc::new(Expr::Const((n - k) as f64)))
                        };
                        let y_pow = if k == 0 {
                            Expr::one()
                        } else {
                            Expr::Pow(y.clone(), Rc::new(Expr::Const(k as f64)))
                        };
                        let term = Expr::Const(coef) * x_pow * y_pow;
                        result = result + term;
                    }
                    return simplify(&result);
                }
            }
            Expr::Pow(Rc::new(a_exp), Rc::new(expand(b)))
        }

        Expr::Div(a, b) => Expr::Div(Rc::new(expand(a)), Rc::new(expand(b))),

        Expr::Neg(a) => Expr::Neg(Rc::new(expand(a))),

        Expr::Fn(name, args) => Expr::Fn(
            name.clone(),
            args.iter().map(|a| Rc::new(expand(a))).collect(),
        ),

        _ => expr.clone(),
    }
}

/// Binomial coefficient
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Symbolic differentiation
pub fn differentiate(expr: &Expr, var: &str) -> Expr {
    let result = match expr {
        // d/dx(c) = 0
        Expr::Const(_) => Expr::zero(),

        // d/dx(x) = 1, d/dx(y) = 0
        Expr::Symbol(s) => {
            if s == var {
                Expr::one()
            } else {
                Expr::zero()
            }
        }

        // d/dx(a + b) = d/dx(a) + d/dx(b)
        Expr::Add(a, b) => differentiate(a, var) + differentiate(b, var),

        // d/dx(a - b) = d/dx(a) - d/dx(b)
        Expr::Sub(a, b) => differentiate(a, var) - differentiate(b, var),

        // Product rule: d/dx(a * b) = a' * b + a * b'
        Expr::Mul(a, b) => {
            let da = differentiate(a, var);
            let db = differentiate(b, var);
            da * (**b).clone() + (**a).clone() * db
        }

        // Quotient rule: d/dx(a / b) = (a' * b - a * b') / b²
        Expr::Div(a, b) => {
            let da = differentiate(a, var);
            let db = differentiate(b, var);
            (da * (**b).clone() - (**a).clone() * db) / ((**b).clone() * (**b).clone())
        }

        // Power rule + chain rule
        Expr::Pow(base, exp) => {
            let base_has_var = base.contains_var(var);
            let exp_has_var = exp.contains_var(var);

            match (base_has_var, exp_has_var) {
                // d/dx(c^n) = 0
                (false, false) => Expr::zero(),

                // d/dx(x^n) = n * x^(n-1) * x'
                (true, false) => {
                    let n = (**exp).clone();
                    let base_deriv = differentiate(base, var);
                    n.clone() * (**base).clone().pow(n - Expr::one()) * base_deriv
                }

                // d/dx(a^x) = a^x * ln(a) * x'
                (false, true) => {
                    let exp_deriv = differentiate(exp, var);
                    expr.clone() * (**base).clone().ln() * exp_deriv
                }

                // d/dx(f^g) = f^g * (g' * ln(f) + g * f'/f)
                (true, true) => {
                    let base_deriv = differentiate(base, var);
                    let exp_deriv = differentiate(exp, var);
                    expr.clone()
                        * (exp_deriv * (**base).clone().ln()
                            + (**exp).clone() * base_deriv / (**base).clone())
                }
            }
        }

        // d/dx(-a) = -d/dx(a)
        Expr::Neg(a) => -differentiate(a, var),

        // Chain rule for functions
        Expr::Fn(name, args) => {
            if args.len() == 1 {
                let arg = &args[0];
                let inner_deriv = differentiate(arg, var);

                let outer_deriv = match name.as_str() {
                    // d/dx(sin(u)) = cos(u) * u'
                    "sin" => (**arg).clone().cos(),
                    // d/dx(cos(u)) = -sin(u) * u'
                    "cos" => -(**arg).clone().sin(),
                    // d/dx(tan(u)) = sec²(u) * u' = (1/cos²(u)) * u'
                    "tan" => {
                        let cos_arg = (**arg).clone().cos();
                        Expr::one() / (cos_arg.clone() * cos_arg)
                    }
                    // d/dx(exp(u)) = exp(u) * u'
                    "exp" => (**arg).clone().exp(),
                    // d/dx(ln(u)) = u'/u
                    "ln" => Expr::one() / (**arg).clone(),
                    // d/dx(sqrt(u)) = u' / (2*sqrt(u))
                    "sqrt" => Expr::one() / (Expr::Const(2.0) * (**arg).clone().sqrt()),
                    // d/dx(abs(u)) = sign(u) * u' (undefined at 0)
                    "abs" => Expr::Fn("sign".to_string(), vec![arg.clone()]),
                    _ => {
                        // Unknown function - leave as derivative
                        return Expr::Derivative(Rc::new(expr.clone()), var.to_string());
                    }
                };

                outer_deriv * inner_deriv
            } else {
                // Multi-argument functions need partial derivatives
                Expr::Derivative(Rc::new(expr.clone()), var.to_string())
            }
        }

        _ => Expr::Derivative(Rc::new(expr.clone()), var.to_string()),
    };

    simplify(&result)
}

/// Symbolic integration (returns None if cannot integrate)
pub fn integrate(expr: &Expr, var: &str) -> Option<Expr> {
    let result = match expr {
        // ∫c dx = c*x
        Expr::Const(c) => Expr::Const(*c) * Expr::Symbol(var.to_string()),

        // ∫x dx = x²/2
        Expr::Symbol(s) if s == var => {
            Expr::Symbol(var.to_string()).pow(Expr::Const(2.0)) / Expr::Const(2.0)
        }

        // ∫y dx = y*x (y is constant w.r.t. x)
        Expr::Symbol(s) if s != var => Expr::Symbol(s.clone()) * Expr::Symbol(var.to_string()),

        // ∫(a + b) dx = ∫a dx + ∫b dx
        Expr::Add(a, b) => {
            let int_a = integrate(a, var)?;
            let int_b = integrate(b, var)?;
            int_a + int_b
        }

        // ∫(a - b) dx = ∫a dx - ∫b dx
        Expr::Sub(a, b) => {
            let int_a = integrate(a, var)?;
            let int_b = integrate(b, var)?;
            int_a - int_b
        }

        // ∫c*f dx = c * ∫f dx (if c is constant w.r.t. var)
        Expr::Mul(a, b) => {
            let a_has_var = a.contains_var(var);
            let b_has_var = b.contains_var(var);

            match (a_has_var, b_has_var) {
                (false, true) => {
                    let int_b = integrate(b, var)?;
                    (**a).clone() * int_b
                }
                (true, false) => {
                    let int_a = integrate(a, var)?;
                    int_a * (**b).clone()
                }
                (false, false) => {
                    // Both constant
                    (**a).clone() * (**b).clone() * Expr::Symbol(var.to_string())
                }
                _ => return None, // Product of two variable expressions - need integration by parts
            }
        }

        // ∫x^n dx = x^(n+1)/(n+1) for n != -1
        Expr::Pow(base, exp) => {
            // Only handle x^n where n is constant
            if let Expr::Symbol(s) = base.as_ref()
                && s == var
                && !exp.contains_var(var)
                && let Some(n) = exp.as_const()
            {
                if (n + 1.0).abs() > 1e-10 {
                    return Some(
                        Expr::Symbol(var.to_string()).pow(Expr::Const(n + 1.0))
                            / Expr::Const(n + 1.0),
                    );
                } else {
                    // ∫x^(-1) dx = ln|x|
                    return Some(Expr::Symbol(var.to_string()).abs().ln());
                }
            }
            return None;
        }

        // ∫-f dx = -∫f dx
        Expr::Neg(a) => {
            let int_a = integrate(a, var)?;
            -int_a
        }

        // Standard integrals
        Expr::Fn(name, args) if args.len() == 1 => {
            let arg = &args[0];
            // Only handle simple case where arg == var
            if let Expr::Symbol(s) = arg.as_ref()
                && s == var
            {
                match name.as_str() {
                    // ∫sin(x) dx = -cos(x)
                    "sin" => return Some(-Expr::Symbol(var.to_string()).cos()),
                    // ∫cos(x) dx = sin(x)
                    "cos" => return Some(Expr::Symbol(var.to_string()).sin()),
                    // ∫exp(x) dx = exp(x)
                    "exp" => return Some(Expr::Symbol(var.to_string()).exp()),
                    // ∫1/x dx = ln|x| (but 1/x is represented differently)
                    _ => return None,
                }
            }
            return None;
        }

        _ => return None,
    };

    Some(simplify(&result))
}

/// Substitute a variable with an expression
pub fn substitute(expr: &Expr, var: &str, replacement: &Expr) -> Expr {
    match expr {
        Expr::Const(_) => expr.clone(),
        Expr::Symbol(s) => {
            if s == var {
                replacement.clone()
            } else {
                expr.clone()
            }
        }
        Expr::Add(a, b) => substitute(a, var, replacement) + substitute(b, var, replacement),
        Expr::Sub(a, b) => substitute(a, var, replacement) - substitute(b, var, replacement),
        Expr::Mul(a, b) => substitute(a, var, replacement) * substitute(b, var, replacement),
        Expr::Div(a, b) => substitute(a, var, replacement) / substitute(b, var, replacement),
        Expr::Pow(a, b) => substitute(a, var, replacement).pow(substitute(b, var, replacement)),
        Expr::Neg(a) => -substitute(a, var, replacement),
        Expr::Fn(name, args) => {
            let new_args: Vec<Rc<Expr>> = args
                .iter()
                .map(|a| Rc::new(substitute(a, var, replacement)))
                .collect();
            Expr::Fn(name.clone(), new_args)
        }
        _ => expr.clone(),
    }
}

/// Evaluate expression numerically
pub fn evaluate(expr: &Expr, vars: &HashMap<String, f64>) -> Option<f64> {
    match expr {
        Expr::Const(c) => Some(*c),
        Expr::Symbol(s) => vars.get(s).copied(),
        Expr::Add(a, b) => Some(evaluate(a, vars)? + evaluate(b, vars)?),
        Expr::Sub(a, b) => Some(evaluate(a, vars)? - evaluate(b, vars)?),
        Expr::Mul(a, b) => Some(evaluate(a, vars)? * evaluate(b, vars)?),
        Expr::Div(a, b) => {
            let denom = evaluate(b, vars)?;
            if denom.abs() < 1e-15 {
                None
            } else {
                Some(evaluate(a, vars)? / denom)
            }
        }
        Expr::Pow(a, b) => Some(evaluate(a, vars)?.powf(evaluate(b, vars)?)),
        Expr::Neg(a) => Some(-evaluate(a, vars)?),
        Expr::Fn(name, args) => {
            if args.len() == 1 {
                let x = evaluate(&args[0], vars)?;
                match name.as_str() {
                    "sin" => Some(x.sin()),
                    "cos" => Some(x.cos()),
                    "tan" => Some(x.tan()),
                    "exp" => Some(x.exp()),
                    "ln" => Some(x.ln()),
                    "sqrt" => Some(x.sqrt()),
                    "abs" => Some(x.abs()),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Solve polynomial equation for roots (simple cases)
pub fn solve_polynomial(expr: &Expr, var: &str) -> Vec<f64> {
    // Expand and simplify
    let expr = simplify(&expand(expr));

    // Try to extract polynomial coefficients
    if let Some(coeffs) = extract_polynomial_coeffs(&expr, var) {
        match coeffs.len() {
            // Constant: no solutions unless zero
            1 => {
                if coeffs[0].abs() < 1e-10 {
                    vec![] // Infinite solutions
                } else {
                    vec![]
                }
            }
            // Linear: ax + b = 0 => x = -b/a
            2 => {
                let a = coeffs[1];
                let b = coeffs[0];
                if a.abs() > 1e-10 {
                    vec![-b / a]
                } else {
                    vec![]
                }
            }
            // Quadratic: ax² + bx + c = 0
            3 => {
                let a = coeffs[2];
                let b = coeffs[1];
                let c = coeffs[0];

                if a.abs() < 1e-10 {
                    // Actually linear
                    if b.abs() > 1e-10 {
                        return vec![-c / b];
                    } else {
                        return vec![];
                    }
                }

                let disc = b * b - 4.0 * a * c;
                if disc < 0.0 {
                    vec![] // Complex roots
                } else if disc < 1e-10 {
                    vec![-b / (2.0 * a)]
                } else {
                    let sqrt_disc = disc.sqrt();
                    vec![(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]
                }
            }
            _ => vec![], // Higher degree - not implemented
        }
    } else {
        vec![]
    }
}

/// Extract polynomial coefficients [c0, c1, c2, ...] for c0 + c1*x + c2*x² + ...
fn extract_polynomial_coeffs(expr: &Expr, var: &str) -> Option<Vec<f64>> {
    let mut coeffs = HashMap::new();

    fn collect_terms(expr: &Expr, var: &str, sign: f64, coeffs: &mut HashMap<usize, f64>) -> bool {
        match expr {
            Expr::Const(c) => {
                *coeffs.entry(0).or_insert(0.0) += sign * c;
                true
            }
            Expr::Symbol(s) => {
                if s == var {
                    *coeffs.entry(1).or_insert(0.0) += sign;
                } else {
                    return false; // Other variables
                }
                true
            }
            Expr::Add(a, b) => {
                collect_terms(a, var, sign, coeffs) && collect_terms(b, var, sign, coeffs)
            }
            Expr::Sub(a, b) => {
                collect_terms(a, var, sign, coeffs) && collect_terms(b, var, -sign, coeffs)
            }
            Expr::Neg(a) => collect_terms(a, var, -sign, coeffs),
            Expr::Mul(a, b) => {
                // Handle c * x^n
                if let Some(c) = a.as_const() {
                    if let Expr::Symbol(s) = b.as_ref()
                        && s == var
                    {
                        *coeffs.entry(1).or_insert(0.0) += sign * c;
                        return true;
                    }
                    if let Expr::Pow(base, exp) = b.as_ref()
                        && let (Expr::Symbol(s), Some(n)) = (base.as_ref(), exp.as_const())
                        && s == var
                        && n >= 0.0
                        && n == n.floor()
                    {
                        *coeffs.entry(n as usize).or_insert(0.0) += sign * c;
                        return true;
                    }
                }
                if let Some(c) = b.as_const()
                    && let Expr::Symbol(s) = a.as_ref()
                    && s == var
                {
                    *coeffs.entry(1).or_insert(0.0) += sign * c;
                    return true;
                }
                false
            }
            Expr::Pow(base, exp) => {
                if let (Expr::Symbol(s), Some(n)) = (base.as_ref(), exp.as_const())
                    && s == var
                    && n >= 0.0
                    && n == n.floor()
                {
                    *coeffs.entry(n as usize).or_insert(0.0) += sign;
                    return true;
                }
                false
            }
            _ => false,
        }
    }

    if collect_terms(expr, var, 1.0, &mut coeffs) {
        let max_deg = coeffs.keys().max().copied().unwrap_or(0);
        let mut result = vec![0.0; max_deg + 1];
        for (deg, coef) in coeffs {
            result[deg] = coef;
        }
        Some(result)
    } else {
        None
    }
}

/// Compile expression to a numerical function
pub fn compile_expr(expr: &Expr) -> Box<dyn Fn(&[f64]) -> f64> {
    let vars: Vec<String> = expr.free_vars().into_iter().collect();
    let expr_clone = expr.clone();

    Box::new(move |args: &[f64]| {
        let mut env = HashMap::new();
        for (i, var) in vars.iter().enumerate() {
            if i < args.len() {
                env.insert(var.clone(), args[i]);
            }
        }
        evaluate(&expr_clone, &env).unwrap_or(f64::NAN)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_add() {
        let x = Expr::symbol("x");
        let zero = Expr::zero();

        let expr = x.clone() + zero;
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);
    }

    #[test]
    fn test_simplify_mul() {
        let x = Expr::symbol("x");
        let one = Expr::one();

        let expr = x.clone() * one;
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        let zero = Expr::zero();
        let expr = x.clone() * zero;
        let simplified = simplify(&expr);
        assert!(simplified.is_zero());
    }

    #[test]
    fn test_differentiate_const() {
        let c = Expr::constant(5.0);
        let dc = differentiate(&c, "x");
        assert!(dc.is_zero());
    }

    #[test]
    fn test_differentiate_var() {
        let x = Expr::symbol("x");
        let dx = differentiate(&x, "x");
        assert!(dx.is_one());

        let y = Expr::symbol("y");
        let dy = differentiate(&y, "x");
        assert!(dy.is_zero());
    }

    #[test]
    fn test_differentiate_power() {
        // d/dx(x^2) = 2x
        let x = Expr::symbol("x");
        let x2 = x.clone().pow(Expr::constant(2.0));
        let dx2 = differentiate(&x2, "x");

        let expected = Expr::constant(2.0) * Expr::symbol("x");
        assert_eq!(simplify(&dx2), simplify(&expected));
    }

    #[test]
    fn test_differentiate_product() {
        // d/dx(x * x) = 2x
        let x = Expr::symbol("x");
        let xx = x.clone() * x.clone();
        let dxx = differentiate(&xx, "x");

        // Result should simplify to 2x
        let simplified = simplify(&dxx);
        // x * 1 + 1 * x = 2x
        let vars = HashMap::from([("x".to_string(), 3.0)]);
        assert_eq!(evaluate(&simplified, &vars), Some(6.0));
    }

    #[test]
    fn test_differentiate_sin() {
        // d/dx(sin(x)) = cos(x)
        let x = Expr::symbol("x");
        let sin_x = x.sin();
        let d_sin = differentiate(&sin_x, "x");

        let expected = Expr::symbol("x").cos();
        assert_eq!(simplify(&d_sin), simplify(&expected));
    }

    #[test]
    fn test_integrate_const() {
        // ∫5 dx = 5x
        let c = Expr::constant(5.0);
        let int_c = integrate(&c, "x").unwrap();

        let vars = HashMap::from([("x".to_string(), 2.0)]);
        assert_eq!(evaluate(&int_c, &vars), Some(10.0));
    }

    #[test]
    fn test_integrate_power() {
        // ∫x^2 dx = x^3/3
        let x = Expr::symbol("x");
        let x2 = x.pow(Expr::constant(2.0));
        let int_x2 = integrate(&x2, "x").unwrap();

        let vars = HashMap::from([("x".to_string(), 3.0)]);
        assert_eq!(evaluate(&int_x2, &vars), Some(9.0)); // 27/3 = 9
    }

    #[test]
    fn test_solve_linear() {
        // 2x + 4 = 0 => x = -2
        let x = Expr::symbol("x");
        let expr = Expr::constant(2.0) * x + Expr::constant(4.0);
        let roots = solve_polynomial(&expr, "x");

        assert_eq!(roots.len(), 1);
        assert!((roots[0] + 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_quadratic() {
        // x^2 - 4 = 0 => x = ±2
        let x = Expr::symbol("x");
        let expr = x.pow(Expr::constant(2.0)) - Expr::constant(4.0);
        let mut roots = solve_polynomial(&expr, "x");
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(roots.len(), 2);
        assert!((roots[0] + 2.0).abs() < 1e-10);
        assert!((roots[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_expand_binomial() {
        // (x + 1)^2 = x^2 + 2x + 1
        let x = Expr::symbol("x");
        let expr = (x.clone() + Expr::one()).pow(Expr::constant(2.0));
        let expanded = expand(&expr);

        let vars = HashMap::from([("x".to_string(), 3.0)]);
        // (3 + 1)^2 = 16
        assert_eq!(evaluate(&expanded, &vars), Some(16.0));
    }

    #[test]
    fn test_substitute() {
        // Substitute x with 2 in x^2 + x
        let x = Expr::symbol("x");
        let expr = x.clone().pow(Expr::constant(2.0)) + x;
        let substituted = substitute(&expr, "x", &Expr::constant(2.0));
        let result = simplify(&substituted);

        // 2^2 + 2 = 6
        assert_eq!(result.as_const(), Some(6.0));
    }

    #[test]
    fn test_compile_expr() {
        let x = Expr::symbol("x");
        let expr = x.clone().pow(Expr::constant(2.0)) + Expr::constant(1.0);
        let f = compile_expr(&expr);

        // f(3) = 3^2 + 1 = 10
        assert_eq!(f(&[3.0]), 10.0);
    }

    #[test]
    fn test_display() {
        let x = Expr::symbol("x");
        let expr = x.clone().pow(Expr::constant(2.0)) + Expr::constant(2.0) * x;
        let s = format!("{}", expr);
        assert!(s.contains("x") && s.contains("2"));
    }
}
