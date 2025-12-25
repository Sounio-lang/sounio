//! Symbolic mathematics
//!
//! Implements symbolic expression manipulation, differentiation, and evaluation.
//! Enables symbolic computation and simplification as language features.

use std::collections::HashMap;
use std::fmt;

use miette::Result;

/// A symbolic mathematical expression
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// Constant value (e.g., 3.14, 2)
    Const(f64),

    /// Variable (e.g., x, y, theta)
    Var(String),

    /// Addition: a + b
    Add(Box<Expr>, Box<Expr>),

    /// Subtraction: a - b
    Sub(Box<Expr>, Box<Expr>),

    /// Multiplication: a * b
    Mul(Box<Expr>, Box<Expr>),

    /// Division: a / b
    Div(Box<Expr>, Box<Expr>),

    /// Power: base ^ exponent
    Pow(Box<Expr>, Box<Expr>),

    /// Sine: sin(x)
    Sin(Box<Expr>),

    /// Cosine: cos(x)
    Cos(Box<Expr>),

    /// Exponential: exp(x)
    Exp(Box<Expr>),

    /// Natural logarithm: ln(x)
    Ln(Box<Expr>),

    /// Square root: sqrt(x)
    Sqrt(Box<Expr>),

    /// Absolute value: abs(x)
    Abs(Box<Expr>),

    /// Negation: -x
    Neg(Box<Expr>),
}

// ============================================================================
// Token types for the parser
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    Eof,
}

// ============================================================================
// Tokenizer (Lexer)
// ============================================================================

struct Tokenizer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(input: &'a str) -> Self {
        Tokenizer { input, pos: 0 }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) {
        if let Some(c) = self.peek_char() {
            self.pos += c.len_utf8();
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_number(&mut self) -> f64 {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() || c == '.' {
                self.advance();
            } else {
                break;
            }
        }
        self.input[start..self.pos].parse().unwrap_or(0.0)
    }

    fn read_ident(&mut self) -> String {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }

    fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        match self.peek_char() {
            None => Token::Eof,
            Some(c) => {
                match c {
                    '+' => {
                        self.advance();
                        Token::Plus
                    }
                    '-' => {
                        self.advance();
                        Token::Minus
                    }
                    '*' => {
                        self.advance();
                        Token::Star
                    }
                    '/' => {
                        self.advance();
                        Token::Slash
                    }
                    '^' => {
                        self.advance();
                        Token::Caret
                    }
                    '(' => {
                        self.advance();
                        Token::LParen
                    }
                    ')' => {
                        self.advance();
                        Token::RParen
                    }
                    ',' => {
                        self.advance();
                        Token::Comma
                    }
                    _ if c.is_ascii_digit() || c == '.' => Token::Number(self.read_number()),
                    _ if c.is_alphabetic() || c == '_' => Token::Ident(self.read_ident()),
                    _ => {
                        self.advance();
                        self.next_token()
                    } // Skip unknown chars
                }
            }
        }
    }
}

// ============================================================================
// Recursive Descent Parser with Operator Precedence
// ============================================================================

struct Parser<'a> {
    tokenizer: Tokenizer<'a>,
    current: Token,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        let mut tokenizer = Tokenizer::new(input);
        let current = tokenizer.next_token();
        Parser { tokenizer, current }
    }

    fn advance(&mut self) {
        self.current = self.tokenizer.next_token();
    }

    fn expect(&mut self, expected: Token) -> Result<()> {
        if std::mem::discriminant(&self.current) == std::mem::discriminant(&expected) {
            self.advance();
            Ok(())
        } else {
            Err(miette::miette!(
                "Expected {:?}, got {:?}",
                expected,
                self.current
            ))
        }
    }

    /// Parse entry point
    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_additive()
    }

    /// Additive: handles + and - (lowest precedence for binary ops)
    fn parse_additive(&mut self) -> Result<Expr> {
        let mut left = self.parse_multiplicative()?;

        loop {
            match &self.current {
                Token::Plus => {
                    self.advance();
                    let right = self.parse_multiplicative()?;
                    left = Expr::Add(Box::new(left), Box::new(right));
                }
                Token::Minus => {
                    self.advance();
                    let right = self.parse_multiplicative()?;
                    left = Expr::Sub(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    /// Multiplicative: handles * and /
    fn parse_multiplicative(&mut self) -> Result<Expr> {
        let mut left = self.parse_power()?;

        loop {
            match &self.current {
                Token::Star => {
                    self.advance();
                    let right = self.parse_power()?;
                    left = Expr::Mul(Box::new(left), Box::new(right));
                }
                Token::Slash => {
                    self.advance();
                    let right = self.parse_power()?;
                    left = Expr::Div(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    /// Power: handles ^ (right-associative, highest precedence for binary ops)
    fn parse_power(&mut self) -> Result<Expr> {
        let base = self.parse_unary()?;

        if matches!(self.current, Token::Caret) {
            self.advance();
            let exp = self.parse_power()?; // Right-associative recursion
            Ok(Expr::Pow(Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    /// Unary: handles unary minus
    fn parse_unary(&mut self) -> Result<Expr> {
        if matches!(self.current, Token::Minus) {
            self.advance();
            let expr = self.parse_unary()?;
            Ok(Expr::Neg(Box::new(expr)))
        } else {
            self.parse_primary()
        }
    }

    /// Primary: numbers, variables, function calls, parenthesized expressions
    fn parse_primary(&mut self) -> Result<Expr> {
        match self.current.clone() {
            Token::Number(n) => {
                self.advance();
                Ok(Expr::Const(n))
            }
            Token::Ident(name) => {
                self.advance();
                // Check for function call
                if matches!(self.current, Token::LParen) {
                    self.advance(); // consume '('
                    let arg = self.parse_expr()?;
                    self.expect(Token::RParen)?;

                    match name.as_str() {
                        "sin" => Ok(Expr::Sin(Box::new(arg))),
                        "cos" => Ok(Expr::Cos(Box::new(arg))),
                        "tan" => {
                            // tan(x) = sin(x) / cos(x)
                            Ok(Expr::Div(
                                Box::new(Expr::Sin(Box::new(arg.clone()))),
                                Box::new(Expr::Cos(Box::new(arg))),
                            ))
                        }
                        "exp" => Ok(Expr::Exp(Box::new(arg))),
                        "ln" | "log" => Ok(Expr::Ln(Box::new(arg))),
                        "sqrt" => Ok(Expr::Sqrt(Box::new(arg))),
                        "abs" => Ok(Expr::Abs(Box::new(arg))),
                        _ => Err(miette::miette!("Unknown function: {}", name)),
                    }
                } else {
                    // Variable
                    Ok(Expr::Var(name))
                }
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            Token::Eof => Err(miette::miette!("Unexpected end of expression")),
            other => Err(miette::miette!("Unexpected token: {:?}", other)),
        }
    }
}

impl Expr {
    /// Create a constant
    pub fn constant(value: f64) -> Self {
        Expr::Const(value)
    }

    /// Create a variable
    pub fn var(name: &str) -> Self {
        Expr::Var(name.to_string())
    }

    /// Parse an expression from a string using recursive descent parser
    ///
    /// Supports:
    /// - Numbers: 3.14, 42, -5
    /// - Variables: x, y, theta
    /// - Binary operators: +, -, *, /, ^ (with proper precedence)
    /// - Parentheses: (x + 1) * 2
    /// - Functions: sin(x), cos(x), tan(x), exp(x), ln(x), sqrt(x), abs(x)
    /// - Unary minus: -x, -(x + 1)
    ///
    /// Operator precedence (highest to lowest):
    /// 1. ^ (power, right-associative)
    /// 2. *, / (multiplicative, left-associative)
    /// 3. +, - (additive, left-associative)
    pub fn parse(input: &str) -> Result<Self> {
        let mut parser = Parser::new(input);
        let expr = parser.parse_expr()?;

        // Ensure we consumed all input
        if !matches!(parser.current, Token::Eof) {
            return Err(miette::miette!(
                "Unexpected token after expression: {:?}",
                parser.current
            ));
        }

        Ok(expr)
    }

    /// Differentiate with respect to a variable
    pub fn differentiate(&self, var: &str) -> Expr {
        match self {
            // d/dx(c) = 0
            Expr::Const(_) => Expr::Const(0.0),

            // d/dx(x) = 1, d/dx(y) = 0
            Expr::Var(v) => {
                if v == var {
                    Expr::Const(1.0)
                } else {
                    Expr::Const(0.0)
                }
            }

            // d/dx(f + g) = df/dx + dg/dx
            Expr::Add(f, g) => {
                let df = f.differentiate(var);
                let dg = g.differentiate(var);
                Expr::Add(Box::new(df), Box::new(dg))
            }

            // d/dx(f - g) = df/dx - dg/dx
            Expr::Sub(f, g) => {
                let df = f.differentiate(var);
                let dg = g.differentiate(var);
                Expr::Sub(Box::new(df), Box::new(dg))
            }

            // d/dx(f * g) = f * dg/dx + g * df/dx (product rule)
            Expr::Mul(f, g) => {
                let df = f.differentiate(var);
                let dg = g.differentiate(var);
                let term1 = Expr::Mul(f.clone(), Box::new(dg));
                let term2 = Expr::Mul(g.clone(), Box::new(df));
                Expr::Add(Box::new(term1), Box::new(term2))
            }

            // d/dx(f / g) = (g * df/dx - f * dg/dx) / g^2 (quotient rule)
            Expr::Div(f, g) => {
                let df = f.differentiate(var);
                let dg = g.differentiate(var);
                let numerator = Expr::Sub(
                    Box::new(Expr::Mul(g.clone(), Box::new(df))),
                    Box::new(Expr::Mul(f.clone(), Box::new(dg))),
                );
                let denominator = Expr::Pow(g.clone(), Box::new(Expr::Const(2.0)));
                Expr::Div(Box::new(numerator), Box::new(denominator))
            }

            // d/dx(f^n) = n * f^(n-1) * df/dx (power rule + chain rule)
            Expr::Pow(f, n) => {
                let df = f.differentiate(var);
                let n_minus_1 = Expr::Sub(n.clone(), Box::new(Expr::Const(1.0)));
                let power_part = Expr::Pow(f.clone(), Box::new(n_minus_1));
                let result = Expr::Mul(n.clone(), Box::new(power_part));
                Expr::Mul(Box::new(result), Box::new(df))
            }

            // d/dx(sin(f)) = cos(f) * df/dx (chain rule)
            Expr::Sin(f) => {
                let df = f.differentiate(var);
                let cos_f = Expr::Cos(f.clone());
                Expr::Mul(Box::new(cos_f), Box::new(df))
            }

            // d/dx(cos(f)) = -sin(f) * df/dx (chain rule)
            Expr::Cos(f) => {
                let df = f.differentiate(var);
                let sin_f = Expr::Sin(f.clone());
                let neg_sin_f = Expr::Neg(Box::new(sin_f));
                Expr::Mul(Box::new(neg_sin_f), Box::new(df))
            }

            // d/dx(exp(f)) = exp(f) * df/dx (chain rule)
            Expr::Exp(f) => {
                let df = f.differentiate(var);
                let exp_f = Expr::Exp(f.clone());
                Expr::Mul(Box::new(exp_f), Box::new(df))
            }

            // d/dx(ln(f)) = (1/f) * df/dx (chain rule)
            Expr::Ln(f) => {
                let df = f.differentiate(var);
                let inv_f = Expr::Div(Box::new(Expr::Const(1.0)), f.clone());
                Expr::Mul(Box::new(inv_f), Box::new(df))
            }

            // d/dx(sqrt(f)) = (1/(2*sqrt(f))) * df/dx (chain rule)
            Expr::Sqrt(f) => {
                let df = f.differentiate(var);
                let two_sqrt_f =
                    Expr::Mul(Box::new(Expr::Const(2.0)), Box::new(Expr::Sqrt(f.clone())));
                let inv = Expr::Div(Box::new(Expr::Const(1.0)), Box::new(two_sqrt_f));
                Expr::Mul(Box::new(inv), Box::new(df))
            }

            // d/dx(abs(f)) = sign(f) * df/dx (requires sign function)
            Expr::Abs(f) => {
                let df = f.differentiate(var);
                // Approximate: sign(f) ≈ f / abs(f)
                let sign_f = Expr::Div(f.clone(), Box::new(Expr::Abs(f.clone())));
                Expr::Mul(Box::new(sign_f), Box::new(df))
            }

            // d/dx(-f) = -df/dx
            Expr::Neg(f) => {
                let df = f.differentiate(var);
                Expr::Neg(Box::new(df))
            }
        }
    }

    /// Evaluate the expression with given variable values
    pub fn evaluate(&self, vars: &HashMap<String, f64>) -> Result<f64> {
        match self {
            Expr::Const(c) => Ok(*c),
            Expr::Var(v) => vars
                .get(v)
                .copied()
                .ok_or_else(|| miette::miette!("Unknown variable: {}", v)),
            Expr::Add(a, b) => {
                let av = a.evaluate(vars)?;
                let bv = b.evaluate(vars)?;
                Ok(av + bv)
            }
            Expr::Sub(a, b) => {
                let av = a.evaluate(vars)?;
                let bv = b.evaluate(vars)?;
                Ok(av - bv)
            }
            Expr::Mul(a, b) => {
                let av = a.evaluate(vars)?;
                let bv = b.evaluate(vars)?;
                Ok(av * bv)
            }
            Expr::Div(a, b) => {
                let av = a.evaluate(vars)?;
                let bv = b.evaluate(vars)?;
                if bv == 0.0 {
                    Err(miette::miette!("Division by zero"))
                } else {
                    Ok(av / bv)
                }
            }
            Expr::Pow(base, exp) => {
                let bv = base.evaluate(vars)?;
                let ev = exp.evaluate(vars)?;
                Ok(bv.powf(ev))
            }
            Expr::Sin(a) => {
                let av = a.evaluate(vars)?;
                Ok(av.sin())
            }
            Expr::Cos(a) => {
                let av = a.evaluate(vars)?;
                Ok(av.cos())
            }
            Expr::Exp(a) => {
                let av = a.evaluate(vars)?;
                Ok(av.exp())
            }
            Expr::Ln(a) => {
                let av = a.evaluate(vars)?;
                if av <= 0.0 {
                    Err(miette::miette!("ln of non-positive number"))
                } else {
                    Ok(av.ln())
                }
            }
            Expr::Sqrt(a) => {
                let av = a.evaluate(vars)?;
                if av < 0.0 {
                    Err(miette::miette!("sqrt of negative number"))
                } else {
                    Ok(av.sqrt())
                }
            }
            Expr::Abs(a) => {
                let av = a.evaluate(vars)?;
                Ok(av.abs())
            }
            Expr::Neg(a) => {
                let av = a.evaluate(vars)?;
                Ok(-av)
            }
        }
    }

    /// Simple algebraic simplification
    pub fn simplify(&self) -> Expr {
        match self {
            // 0 + x = x, x + 0 = x
            Expr::Add(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Expr::Const(0.0), _) => sb,
                    (_, Expr::Const(0.0)) => sa,
                    (Expr::Const(x), Expr::Const(y)) => Expr::Const(x + y),
                    _ => Expr::Add(Box::new(sa), Box::new(sb)),
                }
            }

            // x - 0 = x, 0 - x = -x
            Expr::Sub(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (_, Expr::Const(0.0)) => sa,
                    (Expr::Const(0.0), _) => Expr::Neg(Box::new(sb)),
                    (Expr::Const(x), Expr::Const(y)) => Expr::Const(x - y),
                    _ => Expr::Sub(Box::new(sa), Box::new(sb)),
                }
            }

            // 0 * x = 0, x * 0 = 0, 1 * x = x, x * 1 = x
            Expr::Mul(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Expr::Const(0.0),
                    (Expr::Const(1.0), _) => sb,
                    (_, Expr::Const(1.0)) => sa,
                    (Expr::Const(x), Expr::Const(y)) => Expr::Const(x * y),
                    _ => Expr::Mul(Box::new(sa), Box::new(sb)),
                }
            }

            // x / 1 = x, x / x = 1
            Expr::Div(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (_, Expr::Const(1.0)) => sa,
                    (x, y) if x == y => Expr::Const(1.0),
                    (Expr::Const(x), Expr::Const(y)) if *y != 0.0 => Expr::Const(x / y),
                    _ => Expr::Div(Box::new(sa), Box::new(sb)),
                }
            }

            // x^0 = 1, x^1 = x, 0^x = 0, 1^x = 1
            Expr::Pow(base, exp) => {
                let sb = base.simplify();
                let se = exp.simplify();
                match (&sb, &se) {
                    (_, Expr::Const(0.0)) => Expr::Const(1.0),
                    (_, Expr::Const(1.0)) => sb,
                    (Expr::Const(0.0), _) => Expr::Const(0.0),
                    (Expr::Const(1.0), _) => Expr::Const(1.0),
                    (Expr::Const(x), Expr::Const(y)) => Expr::Const(x.powf(*y)),
                    _ => Expr::Pow(Box::new(sb), Box::new(se)),
                }
            }

            // Recursively simplify children
            Expr::Sin(a) => Expr::Sin(Box::new(a.simplify())),
            Expr::Cos(a) => Expr::Cos(Box::new(a.simplify())),
            Expr::Exp(a) => Expr::Exp(Box::new(a.simplify())),
            Expr::Ln(a) => Expr::Ln(Box::new(a.simplify())),
            Expr::Sqrt(a) => Expr::Sqrt(Box::new(a.simplify())),
            Expr::Abs(a) => Expr::Abs(Box::new(a.simplify())),
            Expr::Neg(a) => {
                let sa = a.simplify();
                match sa {
                    Expr::Const(c) => Expr::Const(-c),
                    Expr::Neg(b) => *b, // Double negation
                    _ => Expr::Neg(Box::new(sa)),
                }
            }

            // Constants and variables are already simplified
            other => other.clone(),
        }
    }

    /// Get all variables in the expression
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_variables(&self, vars: &mut Vec<String>) {
        match self {
            Expr::Const(_) => {}
            Expr::Var(v) => vars.push(v.clone()),
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                a.collect_variables(vars);
                b.collect_variables(vars);
            }
            Expr::Sin(a)
            | Expr::Cos(a)
            | Expr::Exp(a)
            | Expr::Ln(a)
            | Expr::Sqrt(a)
            | Expr::Abs(a)
            | Expr::Neg(a) => a.collect_variables(vars),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(c) => write!(f, "{}", c),
            Expr::Var(v) => write!(f, "{}", v),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Pow(a, b) => write!(f, "({}^{})", a, b),
            Expr::Sin(a) => write!(f, "sin({})", a),
            Expr::Cos(a) => write!(f, "cos({})", a),
            Expr::Exp(a) => write!(f, "exp({})", a),
            Expr::Ln(a) => write!(f, "ln({})", a),
            Expr::Sqrt(a) => write!(f, "sqrt({})", a),
            Expr::Abs(a) => write!(f, "abs({})", a),
            Expr::Neg(a) => write!(f, "-({})", a),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_differentiation() {
        // f(x) = x^2
        let expr = Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );

        // df/dx = 2x
        let deriv = expr.differentiate("x");
        let simplified = deriv.simplify();

        // Evaluate at x=3: should be 6
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let result = simplified.evaluate(&vars).unwrap();
        assert!((result - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_evaluation() {
        // f(x, y) = x^2 + y
        let expr = Expr::Add(
            Box::new(Expr::Pow(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Var("y".to_string())),
        );

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        vars.insert("y".to_string(), 3.0);

        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 7.0); // 4 + 3
    }

    #[test]
    fn test_simplification() {
        // 0 + x should simplify to x
        let expr = Expr::Add(
            Box::new(Expr::Const(0.0)),
            Box::new(Expr::Var("x".to_string())),
        );

        let simplified = expr.simplify();
        match simplified {
            Expr::Var(v) => assert_eq!(v, "x"),
            _ => panic!("Should simplify to just x"),
        }
    }

    // ========================================================================
    // Parser tests
    // ========================================================================

    #[test]
    fn test_parse_simple_number() {
        let expr = Expr::parse("42").unwrap();
        assert_eq!(expr, Expr::Const(42.0));
    }

    #[test]
    fn test_parse_simple_variable() {
        let expr = Expr::parse("x").unwrap();
        assert_eq!(expr, Expr::Var("x".to_string()));
    }

    #[test]
    fn test_parse_addition() {
        let expr = Expr::parse("x + 1").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_parse_subtraction() {
        let expr = Expr::parse("x - 3").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 10.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 7.0);
    }

    #[test]
    fn test_parse_multiplication() {
        let expr = Expr::parse("x * 2").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 7.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 14.0);
    }

    #[test]
    fn test_parse_division() {
        let expr = Expr::parse("x / 4").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 20.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_parse_power() {
        let expr = Expr::parse("x^2").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 9.0);
    }

    #[test]
    fn test_parse_parentheses() {
        // (x + 1) * 2 with x=3 should be 8
        let expr = Expr::parse("(x + 1) * 2").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 8.0);
    }

    #[test]
    fn test_parse_precedence_mul_over_add() {
        // x + 2 * 3 should be x + 6, not (x + 2) * 3
        let expr = Expr::parse("x + 2 * 3").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 1.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 7.0); // 1 + 6 = 7
    }

    #[test]
    fn test_parse_precedence_pow_over_mul() {
        // 2 * x^3 with x=2 should be 2 * 8 = 16
        let expr = Expr::parse("2 * x^3").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 16.0);
    }

    #[test]
    fn test_parse_power_right_associative() {
        // 2^3^2 should be 2^(3^2) = 2^9 = 512
        let expr = Expr::parse("2^3^2").unwrap();
        let vars = HashMap::new();
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 512.0);
    }

    #[test]
    fn test_parse_unary_minus() {
        let expr = Expr::parse("-x").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 5.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, -5.0);
    }

    #[test]
    fn test_parse_unary_minus_with_parentheses() {
        let expr = Expr::parse("-(x + 1)").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, -4.0);
    }

    #[test]
    fn test_parse_sin() {
        let expr = Expr::parse("sin(x)").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let result = expr.evaluate(&vars).unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_cos() {
        let expr = Expr::parse("cos(x)").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let result = expr.evaluate(&vars).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_exp() {
        let expr = Expr::parse("exp(x)").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let result = expr.evaluate(&vars).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_ln() {
        let expr = Expr::parse("ln(x)").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), std::f64::consts::E);
        let result = expr.evaluate(&vars).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_sqrt() {
        let expr = Expr::parse("sqrt(x)").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 16.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_parse_abs() {
        let expr = Expr::parse("abs(x)").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), -7.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 7.0);
    }

    #[test]
    fn test_parse_complex_expression() {
        // (x^2 + 2*x + 1) / (x + 1) = (x+1)^2 / (x+1) = x+1
        // With x=3: (9 + 6 + 1) / 4 = 16/4 = 4
        let expr = Expr::parse("(x^2 + 2*x + 1) / (x + 1)").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_parse_nested_functions() {
        // sin(cos(0)) = sin(1) ≈ 0.8414
        let expr = Expr::parse("sin(cos(x))").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 0.0);
        let result = expr.evaluate(&vars).unwrap();
        assert!((result - 0.8414709848).abs() < 0.0001);
    }

    #[test]
    fn test_parse_polynomial() {
        // 3*x^2 + 2*x + 1 with x=2: 12 + 4 + 1 = 17
        let expr = Expr::parse("3*x^2 + 2*x + 1").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 17.0);
    }

    #[test]
    fn test_parse_differentiate() {
        // f(x) = x^2, f'(x) = 2x, f'(3) = 6
        let expr = Expr::parse("x^2").unwrap();
        let deriv = expr.differentiate("x");
        let simplified = deriv.simplify();

        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let result = simplified.evaluate(&vars).unwrap();
        assert!((result - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_multivariate() {
        // x*y + z with x=2, y=3, z=4: 6 + 4 = 10
        let expr = Expr::parse("x*y + z").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        vars.insert("y".to_string(), 3.0);
        vars.insert("z".to_string(), 4.0);
        let result = expr.evaluate(&vars).unwrap();
        assert_eq!(result, 10.0);
    }
}
