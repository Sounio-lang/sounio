//! Declarative macro definitions and expansion

use std::collections::HashMap;

use super::pattern::*;
use super::token_tree::*;
use crate::common::Span;
use crate::lexer::TokenKind;

/// A declarative macro definition
#[derive(Debug, Clone)]
pub struct MacroDef {
    pub name: String,
    pub arms: Vec<MacroArm>,
    pub is_pub: bool,
    pub is_exported: bool,
    pub doc: Option<String>,
    pub span: Span,
}

/// A single arm of a macro
#[derive(Debug, Clone)]
pub struct MacroArm {
    pub pattern: Vec<Pattern>,
    pub template: Vec<TemplateTree>,
    pub guard: Option<TemplateTree>,
}

/// A template tree for expansion
#[derive(Debug, Clone)]
pub enum TemplateTree {
    Token(TokenKind, String),
    MetaVar(String),
    Group {
        delimiter: Delimiter,
        templates: Vec<TemplateTree>,
    },
    Repeat {
        templates: Vec<TemplateTree>,
        separator: Option<Box<TemplateTree>>,
        kind: RepeatKind,
    },
}

/// Macro expander
pub struct MacroExpander {
    pub macros: HashMap<String, MacroDef>,
    depth: usize,
    max_depth: usize,
    current_ctx: SyntaxContext,
    marks: MarkSet,
    expansion_count: u64,
}

impl MacroExpander {
    pub fn new() -> Self {
        MacroExpander {
            macros: HashMap::new(),
            depth: 0,
            max_depth: 128,
            current_ctx: SyntaxContext::ROOT,
            marks: MarkSet::new(),
            expansion_count: 0,
        }
    }

    pub fn define(&mut self, macro_def: MacroDef) {
        self.macros.insert(macro_def.name.clone(), macro_def);
    }

    pub fn expand(
        &mut self,
        name: &str,
        input: Vec<TokenTree>,
    ) -> Result<Vec<TokenTree>, MacroError> {
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(MacroError::RecursionLimit {
                depth: self.depth,
                span: input.first().map(|t| t.span()).unwrap_or_default(),
            });
        }

        let result = self.expand_inner(name, input);
        self.depth -= 1;
        result
    }

    fn expand_inner(
        &mut self,
        name: &str,
        input: Vec<TokenTree>,
    ) -> Result<Vec<TokenTree>, MacroError> {
        let macro_def =
            self.macros
                .get(name)
                .cloned()
                .ok_or_else(|| MacroError::PatternMismatch {
                    expected: format!("macro '{}'", name),
                    found: "undefined".to_string(),
                    span: input.first().map(|t| t.span()).unwrap_or_default(),
                })?;

        let expansion_ctx = SyntaxContext::fresh();
        self.expansion_count += 1;

        let mark = Mark {
            macro_id: self.expansion_count,
            phase: self.depth as u32,
        };
        self.marks.add_mark(expansion_ctx, mark);

        let mut matcher = PatternMatcher::new();

        for arm in &macro_def.arms {
            match self.try_arm(&mut matcher, arm, &input, expansion_ctx) {
                Ok(result) => return Ok(result),
                Err(_) => continue,
            }
        }

        Err(MacroError::PatternMismatch {
            expected: format!("matching arm for '{}'", name),
            found: "no matching pattern".to_string(),
            span: input.first().map(|t| t.span()).unwrap_or_default(),
        })
    }

    fn try_arm(
        &mut self,
        matcher: &mut PatternMatcher,
        arm: &MacroArm,
        input: &[TokenTree],
        ctx: SyntaxContext,
    ) -> Result<Vec<TokenTree>, MacroError> {
        let (bindings, consumed) = matcher.match_sequence(&arm.pattern, input)?;

        if consumed != input.len() {
            return Err(MacroError::PatternMismatch {
                expected: "end of input".to_string(),
                found: format!("{} remaining tokens", input.len() - consumed),
                span: input.get(consumed).map(|t| t.span()).unwrap_or_default(),
            });
        }

        let mut result = Vec::new();
        for template in &arm.template {
            result.extend(self.transcribe_tree(template, &bindings, ctx)?);
        }

        self.expand_nested(result)
    }

    fn transcribe_tree(
        &self,
        template: &TemplateTree,
        bindings: &Bindings,
        ctx: SyntaxContext,
    ) -> Result<Vec<TokenTree>, MacroError> {
        match template {
            TemplateTree::Token(kind, text) => {
                let token = crate::lexer::Token {
                    kind: *kind,
                    text: text.clone(),
                    span: Span::default(),
                };
                Ok(vec![TokenTree::Token(TokenWithCtx::with_context(
                    token, ctx,
                ))])
            }

            TemplateTree::MetaVar(name) => {
                let capture =
                    bindings
                        .get_single(name)
                        .ok_or_else(|| MacroError::UndefinedMetaVariable {
                            name: name.clone(),
                            span: Span::default(),
                        })?;

                Ok(capture
                    .trees
                    .iter()
                    .cloned()
                    .map(|t| t.with_context(ctx))
                    .collect())
            }

            TemplateTree::Group {
                delimiter,
                templates,
            } => {
                let mut inner = Vec::new();
                for t in templates {
                    inner.extend(self.transcribe_tree(t, bindings, ctx)?);
                }
                Ok(vec![TokenTree::Delimited(
                    *delimiter,
                    inner,
                    Span::default(),
                )])
            }

            TemplateTree::Repeat {
                templates,
                separator,
                kind,
            } => self.transcribe_repeat(templates, separator.as_deref(), *kind, bindings, ctx),
        }
    }

    fn transcribe_repeat(
        &self,
        templates: &[TemplateTree],
        separator: Option<&TemplateTree>,
        _kind: RepeatKind,
        bindings: &Bindings,
        ctx: SyntaxContext,
    ) -> Result<Vec<TokenTree>, MacroError> {
        let repeat_count = self.find_repeat_count(templates, bindings)?;
        let mut result = Vec::new();

        for i in 0..repeat_count {
            let iter_bindings = self.index_bindings(bindings, i)?;

            if i > 0
                && let Some(sep) = separator
            {
                result.extend(self.transcribe_tree(sep, &iter_bindings, ctx)?);
            }

            for template in templates {
                result.extend(self.transcribe_tree(template, &iter_bindings, ctx)?);
            }
        }

        Ok(result)
    }

    fn find_repeat_count(
        &self,
        templates: &[TemplateTree],
        bindings: &Bindings,
    ) -> Result<usize, MacroError> {
        for template in templates {
            if let TemplateTree::MetaVar(name) = template
                && let Some(repeated) = bindings.get_repeat(name)
            {
                return Ok(repeated.len());
            }
        }

        Err(MacroError::InvalidRepetition {
            span: Span::default(),
        })
    }

    fn index_bindings(&self, bindings: &Bindings, index: usize) -> Result<Bindings, MacroError> {
        let mut result = Bindings::new();

        for (name, capture) in &bindings.singles {
            result.insert_single(name.clone(), capture.clone());
        }

        for (name, repeated) in &bindings.repeats {
            if let Some(iter_bindings) = repeated.get(index)
                && let Some(capture) = iter_bindings.get_single(name)
            {
                result.insert_single(name.clone(), capture.clone());
            }
        }

        Ok(result)
    }

    fn expand_nested(&mut self, trees: Vec<TokenTree>) -> Result<Vec<TokenTree>, MacroError> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < trees.len() {
            if i + 1 < trees.len()
                && let (TokenTree::Token(name_tok), TokenTree::Token(bang_tok)) =
                    (&trees[i], &trees[i + 1])
                && name_tok.token.kind == TokenKind::Ident
                && bang_tok.token.kind == TokenKind::Bang
            {
                let macro_name = &name_tok.token.text;

                if self.macros.contains_key(macro_name)
                    && i + 2 < trees.len()
                    && let TokenTree::Delimited(_, inner, _) = &trees[i + 2]
                {
                    let expanded = self.expand(macro_name, inner.clone())?;
                    result.extend(expanded);
                    i += 3;
                    continue;
                }
            }

            match &trees[i] {
                TokenTree::Delimited(d, inner, span) => {
                    let expanded = self.expand_nested(inner.clone())?;
                    result.push(TokenTree::Delimited(*d, expanded, *span));
                }
                other => {
                    result.push(other.clone());
                }
            }

            i += 1;
        }

        Ok(result)
    }
}

impl Default for MacroExpander {
    fn default() -> Self {
        Self::new()
    }
}
