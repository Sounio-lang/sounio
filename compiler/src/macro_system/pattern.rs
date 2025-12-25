//! Macro pattern matching with repetition support

use std::collections::HashMap;

use super::token_tree::*;
use crate::lexer::TokenKind;

/// A macro pattern
#[derive(Debug, Clone)]
pub enum Pattern {
    Token(TokenKind),
    Literal(String),
    MetaVar {
        name: String,
        fragment: FragmentSpecifier,
    },
    Group {
        delimiter: Delimiter,
        patterns: Vec<Pattern>,
    },
    Repeat {
        patterns: Vec<Pattern>,
        separator: Option<Box<Pattern>>,
        kind: RepeatKind,
    },
    Wildcard,
}

/// Fragment specifiers for metavariables
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentSpecifier {
    Ident,
    Ty,
    Expr,
    Stmt,
    Pat,
    Block,
    Item,
    Lifetime,
    Literal,
    Path,
    Tt,
    Vis,
    TokenTree,
    Effect,
    Unit,
}

/// Repetition kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatKind {
    ZeroOrMore,
    OneOrMore,
    Optional,
}

/// Captured bindings from pattern matching
#[derive(Debug, Clone, Default)]
pub struct Bindings {
    pub singles: HashMap<String, Capture>,
    pub repeats: HashMap<String, Vec<Bindings>>,
}

/// A single capture
#[derive(Debug, Clone)]
pub struct Capture {
    pub trees: Vec<TokenTree>,
    pub fragment: FragmentSpecifier,
}

impl Bindings {
    pub fn new() -> Self {
        Bindings::default()
    }

    pub fn insert_single(&mut self, name: String, capture: Capture) {
        self.singles.insert(name, capture);
    }

    pub fn insert_repeat(&mut self, name: String, bindings: Vec<Bindings>) {
        self.repeats.insert(name, bindings);
    }

    pub fn get_single(&self, name: &str) -> Option<&Capture> {
        self.singles.get(name)
    }

    pub fn get_repeat(&self, name: &str) -> Option<&[Bindings]> {
        self.repeats.get(name).map(|v| v.as_slice())
    }

    pub fn merge(&mut self, other: Bindings) {
        self.singles.extend(other.singles);
        for (name, bindings) in other.repeats {
            self.repeats.entry(name).or_default().extend(bindings);
        }
    }
}

/// Pattern matcher
pub struct PatternMatcher {
    max_depth: usize,
    depth: usize,
}

impl PatternMatcher {
    pub fn new() -> Self {
        PatternMatcher {
            max_depth: 128,
            depth: 0,
        }
    }

    pub fn match_pattern(
        &mut self,
        pattern: &Pattern,
        input: &[TokenTree],
    ) -> Result<(Bindings, usize), MacroError> {
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(MacroError::RecursionLimit {
                depth: self.depth,
                span: input.first().map(|t| t.span()).unwrap_or_default(),
            });
        }

        let result = self.match_pattern_inner(pattern, input);
        self.depth -= 1;
        result
    }

    fn match_pattern_inner(
        &mut self,
        pattern: &Pattern,
        input: &[TokenTree],
    ) -> Result<(Bindings, usize), MacroError> {
        match pattern {
            Pattern::Token(kind) => self.match_token(*kind, input),
            Pattern::Literal(text) => self.match_literal(text, input),
            Pattern::MetaVar { name, fragment } => self.match_metavar(name, *fragment, input),
            Pattern::Group {
                delimiter,
                patterns,
            } => self.match_group(*delimiter, patterns, input),
            Pattern::Repeat {
                patterns,
                separator,
                kind,
            } => self.match_repeat(patterns, separator.as_deref(), *kind, input),
            Pattern::Wildcard => {
                if input.is_empty() {
                    Err(MacroError::PatternMismatch {
                        expected: "_".to_string(),
                        found: "end of input".to_string(),
                        span: Default::default(),
                    })
                } else {
                    Ok((Bindings::new(), 1))
                }
            }
        }
    }

    fn match_token(
        &self,
        kind: TokenKind,
        input: &[TokenTree],
    ) -> Result<(Bindings, usize), MacroError> {
        match input.first() {
            Some(TokenTree::Token(t)) if t.token.kind == kind => Ok((Bindings::new(), 1)),
            Some(tree) => Err(MacroError::PatternMismatch {
                expected: format!("{:?}", kind),
                found: format!("{:?}", tree),
                span: tree.span(),
            }),
            None => Err(MacroError::PatternMismatch {
                expected: format!("{:?}", kind),
                found: "end of input".to_string(),
                span: Default::default(),
            }),
        }
    }

    fn match_literal(
        &self,
        text: &str,
        input: &[TokenTree],
    ) -> Result<(Bindings, usize), MacroError> {
        match input.first() {
            Some(TokenTree::Token(t)) if t.token.text == text => Ok((Bindings::new(), 1)),
            Some(tree) => Err(MacroError::PatternMismatch {
                expected: text.to_string(),
                found: format!("{:?}", tree),
                span: tree.span(),
            }),
            None => Err(MacroError::PatternMismatch {
                expected: text.to_string(),
                found: "end of input".to_string(),
                span: Default::default(),
            }),
        }
    }

    fn match_metavar(
        &mut self,
        name: &str,
        fragment: FragmentSpecifier,
        input: &[TokenTree],
    ) -> Result<(Bindings, usize), MacroError> {
        let (trees, consumed) = self.parse_fragment(fragment, input)?;

        let mut bindings = Bindings::new();
        bindings.insert_single(name.to_string(), Capture { trees, fragment });

        Ok((bindings, consumed))
    }

    fn parse_fragment(
        &mut self,
        fragment: FragmentSpecifier,
        input: &[TokenTree],
    ) -> Result<(Vec<TokenTree>, usize), MacroError> {
        match fragment {
            FragmentSpecifier::Ident => match input.first() {
                Some(TokenTree::Token(t)) if t.token.kind == TokenKind::Ident => {
                    Ok((vec![input[0].clone()], 1))
                }
                _ => Err(MacroError::PatternMismatch {
                    expected: "identifier".to_string(),
                    found: format!("{:?}", input.first()),
                    span: input.first().map(|t| t.span()).unwrap_or_default(),
                }),
            },

            FragmentSpecifier::Literal => match input.first() {
                Some(TokenTree::Token(t))
                    if matches!(
                        t.token.kind,
                        TokenKind::IntLit
                            | TokenKind::FloatLit
                            | TokenKind::StringLit
                            | TokenKind::CharLit
                    ) =>
                {
                    Ok((vec![input[0].clone()], 1))
                }
                _ => Err(MacroError::PatternMismatch {
                    expected: "literal".to_string(),
                    found: format!("{:?}", input.first()),
                    span: input.first().map(|t| t.span()).unwrap_or_default(),
                }),
            },

            FragmentSpecifier::Block => match input.first() {
                Some(TokenTree::Delimited(Delimiter::Brace, _, _)) => {
                    Ok((vec![input[0].clone()], 1))
                }
                _ => Err(MacroError::PatternMismatch {
                    expected: "block".to_string(),
                    found: format!("{:?}", input.first()),
                    span: input.first().map(|t| t.span()).unwrap_or_default(),
                }),
            },

            FragmentSpecifier::TokenTree | FragmentSpecifier::Tt => {
                if input.is_empty() {
                    Err(MacroError::PatternMismatch {
                        expected: "token tree".to_string(),
                        found: "end of input".to_string(),
                        span: Default::default(),
                    })
                } else {
                    Ok((vec![input[0].clone()], 1))
                }
            }

            _ => {
                if input.is_empty() {
                    Err(MacroError::PatternMismatch {
                        expected: format!("{:?}", fragment),
                        found: "end of input".to_string(),
                        span: Default::default(),
                    })
                } else {
                    Ok((vec![input[0].clone()], 1))
                }
            }
        }
    }

    fn match_group(
        &mut self,
        delimiter: Delimiter,
        patterns: &[Pattern],
        input: &[TokenTree],
    ) -> Result<(Bindings, usize), MacroError> {
        match input.first() {
            Some(TokenTree::Delimited(d, inner, _)) if *d == delimiter => {
                let (bindings, consumed) = self.match_sequence(patterns, inner)?;
                if consumed != inner.len() {
                    return Err(MacroError::PatternMismatch {
                        expected: "end of group".to_string(),
                        found: format!("{} remaining tokens", inner.len() - consumed),
                        span: inner.get(consumed).map(|t| t.span()).unwrap_or_default(),
                    });
                }
                Ok((bindings, 1))
            }
            _ => Err(MacroError::PatternMismatch {
                expected: format!("{:?} group", delimiter),
                found: format!("{:?}", input.first()),
                span: input.first().map(|t| t.span()).unwrap_or_default(),
            }),
        }
    }

    pub fn match_sequence(
        &mut self,
        patterns: &[Pattern],
        input: &[TokenTree],
    ) -> Result<(Bindings, usize), MacroError> {
        let mut bindings = Bindings::new();
        let mut pos = 0;

        for pattern in patterns {
            let (pattern_bindings, consumed) = self.match_pattern(pattern, &input[pos..])?;
            bindings.merge(pattern_bindings);
            pos += consumed;
        }

        Ok((bindings, pos))
    }

    fn match_repeat(
        &mut self,
        patterns: &[Pattern],
        separator: Option<&Pattern>,
        kind: RepeatKind,
        input: &[TokenTree],
    ) -> Result<(Bindings, usize), MacroError> {
        let mut all_bindings: Vec<Bindings> = Vec::new();
        let mut pos = 0;

        loop {
            match self.match_sequence(patterns, &input[pos..]) {
                Ok((bindings, consumed)) => {
                    all_bindings.push(bindings);
                    pos += consumed;

                    if let Some(sep) = separator {
                        match self.match_pattern(sep, &input[pos..]) {
                            Ok((_, sep_consumed)) => {
                                pos += sep_consumed;
                            }
                            Err(_) => break,
                        }
                    }
                }
                Err(_) => break,
            }
        }

        match kind {
            RepeatKind::ZeroOrMore => {}
            RepeatKind::OneOrMore => {
                if all_bindings.is_empty() {
                    return Err(MacroError::PatternMismatch {
                        expected: "at least one match".to_string(),
                        found: "zero matches".to_string(),
                        span: input.first().map(|t| t.span()).unwrap_or_default(),
                    });
                }
            }
            RepeatKind::Optional => {
                if all_bindings.len() > 1 {
                    return Err(MacroError::PatternMismatch {
                        expected: "at most one match".to_string(),
                        found: format!("{} matches", all_bindings.len()),
                        span: input.first().map(|t| t.span()).unwrap_or_default(),
                    });
                }
            }
        }

        let names = collect_metavar_names(patterns);
        let mut result = Bindings::new();
        for name in names {
            result.insert_repeat(name, all_bindings.clone());
        }

        Ok((result, pos))
    }
}

fn collect_metavar_names(patterns: &[Pattern]) -> Vec<String> {
    let mut names = Vec::new();
    for pattern in patterns {
        collect_names_from_pattern(pattern, &mut names);
    }
    names
}

fn collect_names_from_pattern(pattern: &Pattern, names: &mut Vec<String>) {
    match pattern {
        Pattern::MetaVar { name, .. } => names.push(name.clone()),
        Pattern::Group { patterns, .. } => {
            for p in patterns {
                collect_names_from_pattern(p, names);
            }
        }
        Pattern::Repeat { patterns, .. } => {
            for p in patterns {
                collect_names_from_pattern(p, names);
            }
        }
        _ => {}
    }
}

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}
