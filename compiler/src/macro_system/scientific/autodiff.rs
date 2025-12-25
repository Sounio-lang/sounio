//! Automatic differentiation macro library
//!
//! Provides compile-time generation of derivative code using:
//! - Forward-mode AD (for few inputs, many outputs)
//! - Reverse-mode AD (for many inputs, few outputs)

use std::collections::HashMap;

use crate::lexer::TokenKind;
use crate::macro_system::proc_macro::*;
use crate::macro_system::token_tree::*;

/// Differentiation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffMode {
    Forward,
    Reverse,
}

/// A symbolic expression for differentiation
#[derive(Debug, Clone)]
pub enum SymExpr {
    Const(f64),
    Var(String),
    Binary(Box<SymExpr>, BinOp, Box<SymExpr>),
    Unary(UnOp, Box<SymExpr>),
    Call(String, Vec<SymExpr>),
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Abs,
}

impl SymExpr {
    /// Differentiate with respect to a variable (forward mode)
    pub fn diff(&self, var: &str) -> SymExpr {
        match self {
            SymExpr::Const(_) => SymExpr::Const(0.0),

            SymExpr::Var(name) => {
                if name == var {
                    SymExpr::Const(1.0)
                } else {
                    SymExpr::Const(0.0)
                }
            }

            SymExpr::Binary(left, op, right) => {
                let dl = left.diff(var);
                let dr = right.diff(var);

                match op {
                    BinOp::Add => SymExpr::Binary(Box::new(dl), BinOp::Add, Box::new(dr)),
                    BinOp::Sub => SymExpr::Binary(Box::new(dl), BinOp::Sub, Box::new(dr)),
                    BinOp::Mul => {
                        let term1 = SymExpr::Binary(left.clone(), BinOp::Mul, Box::new(dr));
                        let term2 = SymExpr::Binary(Box::new(dl), BinOp::Mul, right.clone());
                        SymExpr::Binary(Box::new(term1), BinOp::Add, Box::new(term2))
                    }
                    BinOp::Div => {
                        let num1 = SymExpr::Binary(Box::new(dl), BinOp::Mul, right.clone());
                        let num2 = SymExpr::Binary(left.clone(), BinOp::Mul, Box::new(dr));
                        let num = SymExpr::Binary(Box::new(num1), BinOp::Sub, Box::new(num2));
                        let denom = SymExpr::Binary(right.clone(), BinOp::Mul, right.clone());
                        SymExpr::Binary(Box::new(num), BinOp::Div, Box::new(denom))
                    }
                    BinOp::Pow => {
                        if let SymExpr::Const(n) = **right {
                            let coeff = SymExpr::Binary(
                                Box::new(SymExpr::Const(n)),
                                BinOp::Mul,
                                Box::new(SymExpr::Binary(
                                    left.clone(),
                                    BinOp::Pow,
                                    Box::new(SymExpr::Const(n - 1.0)),
                                )),
                            );
                            SymExpr::Binary(Box::new(coeff), BinOp::Mul, Box::new(dl))
                        } else {
                            SymExpr::Call("d_pow".to_string(), vec![*left.clone(), *right.clone()])
                        }
                    }
                }
            }

            SymExpr::Unary(op, inner) => {
                let di = inner.diff(var);

                let chain = match op {
                    UnOp::Neg => {
                        return SymExpr::Unary(UnOp::Neg, Box::new(di));
                    }
                    UnOp::Sqrt => {
                        let two_sqrt = SymExpr::Binary(
                            Box::new(SymExpr::Const(2.0)),
                            BinOp::Mul,
                            Box::new(SymExpr::Unary(UnOp::Sqrt, inner.clone())),
                        );
                        SymExpr::Binary(
                            Box::new(SymExpr::Const(1.0)),
                            BinOp::Div,
                            Box::new(two_sqrt),
                        )
                    }
                    UnOp::Exp => SymExpr::Unary(UnOp::Exp, inner.clone()),
                    UnOp::Log => {
                        SymExpr::Binary(Box::new(SymExpr::Const(1.0)), BinOp::Div, inner.clone())
                    }
                    UnOp::Sin => SymExpr::Unary(UnOp::Cos, inner.clone()),
                    UnOp::Cos => SymExpr::Unary(
                        UnOp::Neg,
                        Box::new(SymExpr::Unary(UnOp::Sin, inner.clone())),
                    ),
                    UnOp::Tan => {
                        let tan_f = SymExpr::Unary(UnOp::Tan, inner.clone());
                        let tan_sq =
                            SymExpr::Binary(Box::new(tan_f.clone()), BinOp::Mul, Box::new(tan_f));
                        SymExpr::Binary(Box::new(SymExpr::Const(1.0)), BinOp::Add, Box::new(tan_sq))
                    }
                    UnOp::Sinh => SymExpr::Unary(UnOp::Cosh, inner.clone()),
                    UnOp::Cosh => SymExpr::Unary(UnOp::Sinh, inner.clone()),
                    UnOp::Tanh => {
                        let tanh_f = SymExpr::Unary(UnOp::Tanh, inner.clone());
                        let tanh_sq =
                            SymExpr::Binary(Box::new(tanh_f.clone()), BinOp::Mul, Box::new(tanh_f));
                        SymExpr::Binary(
                            Box::new(SymExpr::Const(1.0)),
                            BinOp::Sub,
                            Box::new(tanh_sq),
                        )
                    }
                    UnOp::Abs => SymExpr::Call("sign".to_string(), vec![*inner.clone()]),
                };

                SymExpr::Binary(Box::new(chain), BinOp::Mul, Box::new(di))
            }

            SymExpr::Call(name, args) => match name.as_str() {
                "sqrt" if args.len() == 1 => {
                    SymExpr::Unary(UnOp::Sqrt, Box::new(args[0].clone())).diff(var)
                }
                "exp" if args.len() == 1 => {
                    SymExpr::Unary(UnOp::Exp, Box::new(args[0].clone())).diff(var)
                }
                "log" | "ln" if args.len() == 1 => {
                    SymExpr::Unary(UnOp::Log, Box::new(args[0].clone())).diff(var)
                }
                "sin" if args.len() == 1 => {
                    SymExpr::Unary(UnOp::Sin, Box::new(args[0].clone())).diff(var)
                }
                "cos" if args.len() == 1 => {
                    SymExpr::Unary(UnOp::Cos, Box::new(args[0].clone())).diff(var)
                }
                _ => SymExpr::Call(format!("d_{}", name), args.clone()),
            },
        }
    }

    /// Simplify expression
    pub fn simplify(&self) -> SymExpr {
        match self {
            SymExpr::Binary(left, op, right) => {
                let l = left.simplify();
                let r = right.simplify();

                match (op, &l, &r) {
                    (BinOp::Add, SymExpr::Const(a), SymExpr::Const(b)) => SymExpr::Const(a + b),
                    (BinOp::Sub, SymExpr::Const(a), SymExpr::Const(b)) => SymExpr::Const(a - b),
                    (BinOp::Mul, SymExpr::Const(a), SymExpr::Const(b)) => SymExpr::Const(a * b),
                    (BinOp::Div, SymExpr::Const(a), SymExpr::Const(b)) if *b != 0.0 => {
                        SymExpr::Const(a / b)
                    }
                    (BinOp::Pow, SymExpr::Const(a), SymExpr::Const(b)) => {
                        SymExpr::Const(a.powf(*b))
                    }

                    (BinOp::Add, SymExpr::Const(0.0), _) => r,
                    (BinOp::Add, _, SymExpr::Const(0.0)) => l,
                    (BinOp::Sub, _, SymExpr::Const(0.0)) => l,
                    (BinOp::Mul, SymExpr::Const(0.0), _) => SymExpr::Const(0.0),
                    (BinOp::Mul, _, SymExpr::Const(0.0)) => SymExpr::Const(0.0),
                    (BinOp::Mul, SymExpr::Const(1.0), _) => r,
                    (BinOp::Mul, _, SymExpr::Const(1.0)) => l,
                    (BinOp::Div, _, SymExpr::Const(1.0)) => l,
                    (BinOp::Pow, _, SymExpr::Const(0.0)) => SymExpr::Const(1.0),
                    (BinOp::Pow, _, SymExpr::Const(1.0)) => l,

                    _ => SymExpr::Binary(Box::new(l), *op, Box::new(r)),
                }
            }

            SymExpr::Unary(op, inner) => {
                let i = inner.simplify();

                match (op, &i) {
                    (UnOp::Neg, SymExpr::Const(c)) => SymExpr::Const(-c),
                    (UnOp::Neg, SymExpr::Unary(UnOp::Neg, inner2)) => *inner2.clone(),
                    _ => SymExpr::Unary(*op, Box::new(i)),
                }
            }

            other => other.clone(),
        }
    }

    /// Convert to token stream (code generation)
    pub fn to_tokens(&self) -> TokenStream {
        let mut tokens = TokenStream::new();
        self.to_tokens_inner(&mut tokens);
        tokens
    }

    fn to_tokens_inner(&self, tokens: &mut TokenStream) {
        match self {
            SymExpr::Const(c) => {
                let text = if c.fract() == 0.0 {
                    format!("{:.1}", c)
                } else {
                    format!("{}", c)
                };
                tokens.push(make_token(TokenKind::FloatLit, &text));
            }

            SymExpr::Var(name) => {
                tokens.push(make_token(TokenKind::Ident, name));
            }

            SymExpr::Binary(left, op, right) => {
                tokens.push(make_token(TokenKind::LParen, "("));
                left.to_tokens_inner(tokens);

                let op_text = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::Div => "/",
                    BinOp::Pow => ".pow",
                };

                if matches!(op, BinOp::Pow) {
                    tokens.push(make_token(TokenKind::Dot, "."));
                    tokens.push(make_token(TokenKind::Ident, "pow"));
                    tokens.push(make_token(TokenKind::LParen, "("));
                    right.to_tokens_inner(tokens);
                    tokens.push(make_token(TokenKind::RParen, ")"));
                } else {
                    let op_kind = match op {
                        BinOp::Add => TokenKind::Plus,
                        BinOp::Sub => TokenKind::Minus,
                        BinOp::Mul => TokenKind::Star,
                        BinOp::Div => TokenKind::Slash,
                        BinOp::Pow => unreachable!(),
                    };
                    tokens.push(make_token(op_kind, op_text));
                    right.to_tokens_inner(tokens);
                }

                tokens.push(make_token(TokenKind::RParen, ")"));
            }

            SymExpr::Unary(op, inner) => {
                let (prefix, func) = match op {
                    UnOp::Neg => (Some("-"), None),
                    UnOp::Sqrt => (None, Some("sqrt")),
                    UnOp::Exp => (None, Some("exp")),
                    UnOp::Log => (None, Some("ln")),
                    UnOp::Sin => (None, Some("sin")),
                    UnOp::Cos => (None, Some("cos")),
                    UnOp::Tan => (None, Some("tan")),
                    UnOp::Sinh => (None, Some("sinh")),
                    UnOp::Cosh => (None, Some("cosh")),
                    UnOp::Tanh => (None, Some("tanh")),
                    UnOp::Abs => (None, Some("abs")),
                };

                if let Some(p) = prefix {
                    tokens.push(make_token(TokenKind::Minus, p));
                    inner.to_tokens_inner(tokens);
                } else if let Some(f) = func {
                    tokens.push(make_token(TokenKind::Ident, f));
                    tokens.push(make_token(TokenKind::LParen, "("));
                    inner.to_tokens_inner(tokens);
                    tokens.push(make_token(TokenKind::RParen, ")"));
                }
            }

            SymExpr::Call(name, args) => {
                tokens.push(make_token(TokenKind::Ident, name));
                tokens.push(make_token(TokenKind::LParen, "("));
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        tokens.push(make_token(TokenKind::Comma, ","));
                    }
                    arg.to_tokens_inner(tokens);
                }
                tokens.push(make_token(TokenKind::RParen, ")"));
            }
        }
    }
}

fn make_token(kind: TokenKind, text: &str) -> TokenTree {
    TokenTree::Token(TokenWithCtx::new(crate::lexer::Token {
        kind,
        text: text.to_string(),
        span: crate::common::Span::default(),
    }))
}

/// Gradient computation result
#[derive(Debug, Clone)]
pub struct Gradient {
    pub partials: HashMap<String, SymExpr>,
}

impl Gradient {
    pub fn compute(expr: &SymExpr, vars: &[String]) -> Self {
        let partials = vars
            .iter()
            .map(|v| (v.clone(), expr.diff(v).simplify()))
            .collect();
        Gradient { partials }
    }
}
