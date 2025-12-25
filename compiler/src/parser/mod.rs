//! Parser for the Sounio language
//!
//! A recursive descent parser that produces an AST from a token stream.

pub mod recovery;

#[cfg(test)]
mod tests;

use crate::ast::*;
use crate::common::{IdGenerator, NodeId, Span};
use crate::lexer::{Token, TokenKind};
use miette::Result;

/// Parse a token stream into an AST
pub fn parse(tokens: &[Token], source: &str) -> Result<Ast> {
    let mut parser = Parser::with_source(tokens, source);
    parser.parse_program()
}

/// Parse a token stream into an AST with a custom NodeId start.
/// Returns the AST and the next available NodeId value.
pub fn parse_with_id_start(tokens: &[Token], source: &str, start_id: u32) -> Result<(Ast, u32)> {
    let mut parser = Parser::with_source_and_id_start(tokens, source, start_id);
    let ast = parser.parse_program()?;
    Ok((ast, parser.next_id_value()))
}

/// Parser state
pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    id_gen: IdGenerator,
    /// When false, don't parse `Ident { ... }` as a struct literal
    /// This is needed to resolve ambiguity in contexts like `match x { ... }`
    allow_struct_literals: bool,
    /// Mapping from NodeId to source spans
    node_spans: std::collections::HashMap<NodeId, Span>,
    /// Pending `>` from splitting a `>>` token (for nested generics like `Option<Box<T>>`)
    pending_gt: bool,
    /// Source text for newline detection
    source: &'a str,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token]) -> Self {
        Self::with_source(tokens, "")
    }

    pub fn with_source(tokens: &'a [Token], source: &'a str) -> Self {
        Self {
            tokens,
            pos: 0,
            id_gen: IdGenerator::new(),
            allow_struct_literals: true,
            node_spans: std::collections::HashMap::new(),
            pending_gt: false,
            source,
        }
    }

    pub fn with_id_start(tokens: &'a [Token], start_id: u32) -> Self {
        Self::with_source_and_id_start(tokens, "", start_id)
    }

    pub fn with_source_and_id_start(tokens: &'a [Token], source: &'a str, start_id: u32) -> Self {
        Self {
            tokens,
            pos: 0,
            id_gen: IdGenerator::with_start(start_id),
            allow_struct_literals: true,
            node_spans: std::collections::HashMap::new(),
            pending_gt: false,
            source,
        }
    }

    pub fn next_id_value(&self) -> u32 {
        self.id_gen.next_value()
    }

    fn next_id(&mut self) -> NodeId {
        self.id_gen.next()
    }

    /// Check if there's a newline between the previous token and the current token.
    /// This is used to prevent parsing `(...)` as a call expression when it's on a new line.
    fn had_newline_before_current(&self) -> bool {
        if self.pos == 0 || self.source.is_empty() {
            return false;
        }
        // Get the span between previous token's end and current token's start
        let prev_end = self
            .tokens
            .get(self.pos - 1)
            .map(|t| t.span.end)
            .unwrap_or(0);
        let curr_start = self.current().span.start;
        // Check if there's a newline in that range
        if curr_start > prev_end && curr_start <= self.source.len() && prev_end <= self.source.len()
        {
            self.source[prev_end..curr_start].contains('\n')
        } else {
            false
        }
    }

    /// Check if current token can be used as a macro name (identifier or keyword)
    fn can_be_macro_name(&self) -> bool {
        matches!(self.peek(), TokenKind::Ident) || self.peek().is_keyword()
    }

    /// Get the text of the current token as a macro name
    fn get_macro_name(&self) -> String {
        self.current().text.clone()
    }

    /// Record the span of a node for error reporting
    fn record_span(&mut self, id: NodeId, span: Span) {
        self.node_spans.insert(id, span);
    }

    /// Parse an expression and record its span
    fn parse_expr_with_span(&mut self) -> Result<Expr> {
        let start = self.current().span.start;
        let expr = self.parse_expr()?;
        let end = self
            .tokens
            .get(self.pos.saturating_sub(1))
            .map(|t| t.span.end)
            .unwrap_or(start);

        let id = match &expr {
            Expr::Literal { id, .. } => *id,
            Expr::Path { id, .. } => *id,
            Expr::Call { id, .. } => *id,
            Expr::OntologyTerm { id, .. } => *id,
            // For complex expressions, record the span
            _ => {
                return Ok(expr);
            }
        };

        // Record span for the expression
        self.record_span(id, Span::new(start, end));
        Ok(expr)
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or_else(|| {
            self.tokens
                .last()
                .expect("token stream should have at least EOF")
        })
    }

    fn peek(&self) -> TokenKind {
        self.current().kind
    }

    fn peek_n(&self, n: usize) -> TokenKind {
        self.tokens
            .get(self.pos + n)
            .map(|t| t.kind)
            .unwrap_or(TokenKind::Eof)
    }

    fn at(&self, kind: TokenKind) -> bool {
        self.peek() == kind
    }

    fn at_any(&self, kinds: &[TokenKind]) -> bool {
        kinds.contains(&self.peek())
    }

    fn advance(&mut self) -> &Token {
        let tok = self.current();
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        // Return the token that was at the previous position
        &self.tokens[self.pos.saturating_sub(1)]
    }

    fn expect(&mut self, kind: TokenKind) -> Result<&Token> {
        if self.at(kind) {
            Ok(self.advance())
        } else {
            Err(miette::miette!(
                "Expected {:?}, found {:?} at position {}",
                kind,
                self.peek(),
                self.current().span.start
            ))
        }
    }

    fn span(&self) -> Span {
        self.current().span
    }

    /// Check if we're at a `>` token (either real or from a pending split `>>`)
    fn at_gt(&self) -> bool {
        self.pending_gt || self.at(TokenKind::Gt)
    }

    /// Consume a `>` token in type context, splitting `>>` if necessary
    fn expect_gt(&mut self) -> Result<()> {
        if self.pending_gt {
            // We already consumed half of a `>>`, just clear the pending flag
            self.pending_gt = false;
            Ok(())
        } else if self.at(TokenKind::Gt) {
            self.advance();
            Ok(())
        } else if self.at(TokenKind::Shr) {
            // Split `>>` into two `>` tokens - consume the first `>`, leave second pending
            self.advance();
            self.pending_gt = true;
            Ok(())
        } else {
            Err(miette::miette!(
                "Expected '>', found {:?} at position {}",
                self.peek(),
                self.current().span.start
            ))
        }
    }

    /// Check if we're at `>` or `>>` (for lookahead in type parsing)
    fn at_gt_or_shr(&self) -> bool {
        self.pending_gt || self.at(TokenKind::Gt) || self.at(TokenKind::Shr)
    }

    // ==================== PROGRAM ====================

    fn parse_program(&mut self) -> Result<Ast> {
        let mut items = Vec::new();

        // Skip file-level doc comments before module declaration
        while self.at(TokenKind::DocCommentOuter) || self.at(TokenKind::DocCommentInner) {
            self.advance();
        }

        // Optional module declaration
        let module_name = if self.at(TokenKind::Module) {
            self.advance();
            let name = self.parse_path()?;
            // Accept optional semicolon after module declaration
            if self.at(TokenKind::Semi) {
                self.advance();
            }
            Some(name)
        } else {
            None
        };

        // Parse items
        while !self.at(TokenKind::Eof) {
            items.push(self.parse_item()?);
        }

        Ok(Ast {
            module_name,
            items,
            node_spans: self.node_spans.clone(),
        })
    }

    // ==================== ITEMS ====================

    pub fn parse_item(&mut self) -> Result<Item> {
        // Skip doc comments (they are attached to following items)
        while self.at(TokenKind::DocCommentOuter) || self.at(TokenKind::DocCommentInner) {
            self.advance();
        }

        // Check for macro invocation at item level (identifier or keyword followed by !)
        if self.can_be_macro_name() && self.peek_n(1) == TokenKind::Bang {
            let macro_inv = self.parse_macro_invocation()?;
            // Consume optional semicolon after macro invocation
            if self.at(TokenKind::Semi) {
                self.advance();
            }
            return Ok(Item::MacroInvocation(macro_inv));
        }

        // Parse attributes (e.g., #[compat(threshold = 0.2)])
        let attributes = self.parse_item_attributes()?;

        // Parse visibility
        let visibility = self.parse_visibility();

        // Parse modifiers
        let modifiers = self.parse_modifiers();

        match self.peek() {
            TokenKind::Fn | TokenKind::Kernel => self.parse_fn(visibility, modifiers, attributes),
            TokenKind::Let | TokenKind::Const => self.parse_global(visibility, modifiers),
            TokenKind::Struct => self.parse_struct(visibility, modifiers),
            TokenKind::Enum => self.parse_enum(visibility, modifiers),
            TokenKind::Trait => self.parse_trait(visibility, modifiers),
            TokenKind::Impl => self.parse_impl(),
            TokenKind::Type => self.parse_type_alias(visibility),
            TokenKind::Effect => self.parse_effect(visibility),
            TokenKind::Handler => self.parse_handler(visibility),
            TokenKind::Import | TokenKind::Use => self.parse_import_with_visibility(visibility),
            TokenKind::Export => self.parse_export(),
            TokenKind::Extern => self.parse_extern_with_visibility(visibility),
            TokenKind::Ontology => self.parse_ontology_import(),
            TokenKind::Align => self.parse_align_decl(),
            TokenKind::Ode => self.parse_ode_def(visibility),
            TokenKind::Pde => self.parse_pde_def(visibility),
            TokenKind::Causal => self.parse_causal_model_def(visibility),
            TokenKind::Module => self.parse_module_decl(visibility),
            _ => Err(miette::miette!(
                "Unexpected token {:?} at start of item",
                self.peek()
            )),
        }
    }

    /// Parse item-level attributes: #[attr], #[attr(args)], etc.
    fn parse_item_attributes(&mut self) -> Result<Vec<Attribute>> {
        let mut attrs = Vec::new();

        while self.at(TokenKind::Hash) {
            self.advance(); // consume #
            self.expect(TokenKind::LBracket)?;

            let start = self.span();
            // Attribute name can be a keyword like 'compat'
            let name = if self.at(TokenKind::Compat) {
                self.advance();
                "compat".to_string()
            } else {
                self.parse_ident()?
            };

            // Parse attribute arguments if present
            let args = if self.at(TokenKind::LParen) {
                self.advance();
                let args = self.parse_attribute_args()?;
                self.expect(TokenKind::RParen)?;
                args
            } else {
                AttributeArgs::Empty
            };

            let end = self.span();
            self.expect(TokenKind::RBracket)?;

            attrs.push(Attribute {
                id: self.next_id(),
                name,
                args,
                span: start.merge(end),
            });
        }

        Ok(attrs)
    }

    /// Check if current token can be used as an identifier in attribute context
    fn is_attr_ident(&self) -> bool {
        matches!(
            self.peek(),
            TokenKind::Ident
                | TokenKind::Threshold
                | TokenKind::Compat
                | TokenKind::Distance
                | TokenKind::From
                | TokenKind::Align
        )
    }

    /// Parse identifier in attribute context (allows some keywords)
    fn parse_attr_ident(&mut self) -> Result<String> {
        match self.peek() {
            TokenKind::Ident => Ok(self.advance().text.clone()),
            TokenKind::Threshold => {
                self.advance();
                Ok("threshold".to_string())
            }
            TokenKind::Compat => {
                self.advance();
                Ok("compat".to_string())
            }
            TokenKind::Distance => {
                self.advance();
                Ok("distance".to_string())
            }
            TokenKind::From => {
                self.advance();
                Ok("from".to_string())
            }
            TokenKind::Align => {
                self.advance();
                Ok("align".to_string())
            }
            _ => Err(miette::miette!(
                "Expected identifier in attribute, found {:?}",
                self.peek()
            )),
        }
    }

    /// Parse attribute arguments inside parentheses
    fn parse_attribute_args(&mut self) -> Result<AttributeArgs> {
        if self.at(TokenKind::RParen) {
            return Ok(AttributeArgs::Empty);
        }

        // Check if this is a named argument (ident = value)
        if self.is_attr_ident() && self.peek_n(1) == TokenKind::Eq {
            let mut named = Vec::new();
            loop {
                let key = self.parse_attr_ident()?;
                self.expect(TokenKind::Eq)?;
                let value = self.parse_attribute_value()?;
                named.push((key, value));

                if !self.at(TokenKind::Comma) {
                    break;
                }
                self.advance();
                if self.at(TokenKind::RParen) {
                    break;
                }
            }
            Ok(AttributeArgs::Named(named))
        } else {
            // Single value or list
            let first = self.parse_attribute_value()?;
            if self.at(TokenKind::Comma) {
                let mut list = vec![first];
                while self.at(TokenKind::Comma) {
                    self.advance();
                    if self.at(TokenKind::RParen) {
                        break;
                    }
                    list.push(self.parse_attribute_value()?);
                }
                Ok(AttributeArgs::List(list))
            } else {
                Ok(AttributeArgs::Value(first))
            }
        }
    }

    /// Parse a single attribute value
    fn parse_attribute_value(&mut self) -> Result<AttributeValue> {
        match self.peek() {
            TokenKind::StringLit => {
                let s = self.advance().text.clone();
                Ok(AttributeValue::String(s[1..s.len() - 1].to_string()))
            }
            TokenKind::IntLit => {
                let s = self.advance().text.clone();
                Ok(AttributeValue::Int(s.parse().unwrap_or(0)))
            }
            TokenKind::FloatLit => {
                let s = self.advance().text.clone();
                Ok(AttributeValue::Float(s.parse().unwrap_or(0.0)))
            }
            TokenKind::True => {
                self.advance();
                Ok(AttributeValue::Bool(true))
            }
            TokenKind::False => {
                self.advance();
                Ok(AttributeValue::Bool(false))
            }
            TokenKind::Ident => {
                let path = self.parse_path()?;
                // Check if this is a nested attribute like cfg(all(...))
                if self.at(TokenKind::LParen) {
                    self.advance();
                    let nested = self.parse_attribute_args()?;
                    self.expect(TokenKind::RParen)?;
                    Ok(AttributeValue::Nested(
                        path.segments.join("::"),
                        Box::new(nested),
                    ))
                } else {
                    Ok(AttributeValue::Path(path))
                }
            }
            _ => Err(miette::miette!(
                "Expected attribute value, found {:?}",
                self.peek()
            )),
        }
    }

    fn parse_visibility(&mut self) -> Visibility {
        if self.at(TokenKind::Pub) {
            self.advance();
            Visibility::Public
        } else {
            Visibility::Private
        }
    }

    fn parse_modifiers(&mut self) -> Modifiers {
        let mut mods = Modifiers::default();

        loop {
            match self.peek() {
                TokenKind::Linear => {
                    self.advance();
                    mods.linear = true;
                }
                TokenKind::Affine => {
                    self.advance();
                    mods.affine = true;
                }
                TokenKind::Async => {
                    self.advance();
                    mods.is_async = true;
                }
                TokenKind::Unsafe => {
                    self.advance();
                    mods.is_unsafe = true;
                }
                _ => break,
            }
        }

        mods
    }

    // ==================== FUNCTIONS ====================

    fn parse_fn(
        &mut self,
        visibility: Visibility,
        modifiers: Modifiers,
        attributes: Vec<Attribute>,
    ) -> Result<Item> {
        let start = self.span();

        // Check for extern "ABI" before fn
        let abi = if self.at(TokenKind::Extern) {
            self.advance();
            // Parse optional ABI string (e.g., "C", "system")
            if self.at(TokenKind::StringLit) {
                let s = self.advance().text.clone();
                // Remove quotes from string literal
                let abi_str = s[1..s.len() - 1].to_string();
                Some(crate::ast::Abi::from_str(&abi_str))
            } else {
                // Default to C ABI if no string specified
                Some(crate::ast::Abi::C)
            }
        } else {
            None
        };

        let is_kernel = if self.at(TokenKind::Kernel) {
            self.advance();
            true
        } else {
            false
        };

        self.expect(TokenKind::Fn)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        let params = self.parse_params()?;
        let return_type = self.parse_return_type()?;
        let effects = self.parse_effect_clause()?;
        let where_clause = self.parse_where_clause()?;
        let body = self.parse_block()?;

        let end = self.span();

        Ok(Item::Function(FnDef {
            id: self.next_id(),
            visibility,
            modifiers: FnModifiers {
                is_async: modifiers.is_async,
                is_unsafe: modifiers.is_unsafe,
                is_kernel,
                abi,
            },
            attributes,
            name,
            generics,
            params,
            return_type,
            effects,
            where_clause,
            body,
            span: start.merge(end),
        }))
    }

    fn parse_params(&mut self) -> Result<Vec<Param>> {
        self.expect(TokenKind::LParen)?;
        let mut params = Vec::new();

        while !self.at(TokenKind::RParen) {
            params.push(self.parse_param()?);
            if !self.at(TokenKind::RParen) {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::RParen)?;
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<Param> {
        let is_mut = if self.at(TokenKind::Mut) {
            self.advance();
            true
        } else {
            false
        };

        // Handle special `self` parameter which doesn't require a type annotation
        if self.at(TokenKind::SelfLower) {
            self.advance();
            return Ok(Param {
                id: self.next_id(),
                is_mut,
                pattern: Pattern::Binding {
                    name: "self".to_string(),
                    mutable: false,
                },
                ty: TypeExpr::SelfType, // Special self type
                attributes: Vec::new(),
            });
        }

        // Handle &self and &mut self
        if self.at(TokenKind::Amp) {
            self.advance();
            let is_ref_mut = if self.at(TokenKind::Mut) {
                self.advance();
                true
            } else {
                false
            };
            if self.at(TokenKind::SelfLower) {
                self.advance();
                return Ok(Param {
                    id: self.next_id(),
                    is_mut: is_ref_mut,
                    pattern: Pattern::Binding {
                        name: "self".to_string(),
                        mutable: is_ref_mut,
                    },
                    ty: TypeExpr::Reference {
                        mutable: is_ref_mut,
                        inner: Box::new(TypeExpr::SelfType),
                    },
                    attributes: Vec::new(),
                });
            }
            // Not &self, backtrack is tricky - for now just error
            return Err(miette::miette!("Expected 'self' after '&' in parameter"));
        }

        let pattern = self.parse_pattern()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type()?;

        Ok(Param {
            id: self.next_id(),
            is_mut,
            pattern,
            ty,
            attributes: Vec::new(),
        })
    }

    fn parse_return_type(&mut self) -> Result<Option<TypeExpr>> {
        if self.at(TokenKind::Arrow) {
            self.advance();
            Ok(Some(self.parse_type()?))
        } else {
            Ok(None)
        }
    }

    fn parse_effect_clause(&mut self) -> Result<Vec<EffectRef>> {
        if self.at(TokenKind::With) {
            self.advance();
            let mut effects = vec![self.parse_effect_ref()?];
            while self.at(TokenKind::Comma) {
                self.advance();
                effects.push(self.parse_effect_ref()?);
            }
            Ok(effects)
        } else {
            Ok(Vec::new())
        }
    }

    fn parse_effect_ref(&mut self) -> Result<EffectRef> {
        let name = self.parse_path()?;
        let args = if self.at(TokenKind::Lt) {
            self.parse_type_args()?
        } else {
            Vec::new()
        };
        Ok(EffectRef {
            id: self.next_id(),
            name,
            args,
        })
    }

    // ==================== STRUCTS ====================

    fn parse_struct(&mut self, visibility: Visibility, modifiers: Modifiers) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Struct)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        let where_clause = self.parse_where_clause()?;

        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) {
            fields.push(self.parse_field()?);
            if !self.at(TokenKind::RBrace) {
                // Allow optional comma
                if self.at(TokenKind::Comma) {
                    self.advance();
                }
            }
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Struct(StructDef {
            id: self.next_id(),
            visibility,
            modifiers: TypeModifiers {
                linear: modifiers.linear,
                affine: modifiers.affine,
            },
            attributes: Vec::new(),
            name,
            generics,
            where_clause,
            fields,
            span: start.merge(end),
        }))
    }

    fn parse_field(&mut self) -> Result<FieldDef> {
        let visibility = self.parse_visibility();
        let name = self.parse_ident()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type()?;

        Ok(FieldDef {
            id: self.next_id(),
            visibility,
            attributes: Vec::new(),
            name,
            ty,
        })
    }

    // ==================== ENUMS ====================

    fn parse_enum(&mut self, visibility: Visibility, modifiers: Modifiers) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Enum)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        let where_clause = self.parse_where_clause()?;

        self.expect(TokenKind::LBrace)?;
        let mut variants = Vec::new();
        while !self.at(TokenKind::RBrace) {
            variants.push(self.parse_variant()?);
            if !self.at(TokenKind::RBrace) {
                if self.at(TokenKind::Comma) {
                    self.advance();
                }
            }
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Enum(EnumDef {
            id: self.next_id(),
            visibility,
            modifiers: TypeModifiers {
                linear: modifiers.linear,
                affine: modifiers.affine,
            },
            name,
            generics,
            where_clause,
            variants,
            span: start.merge(end),
        }))
    }

    fn parse_variant(&mut self) -> Result<VariantDef> {
        let name = self.parse_ident()?;
        let data = if self.at(TokenKind::LParen) {
            self.advance();
            let mut types = Vec::new();
            while !self.at(TokenKind::RParen) {
                types.push(self.parse_type()?);
                if !self.at(TokenKind::RParen) {
                    self.expect(TokenKind::Comma)?;
                }
            }
            self.expect(TokenKind::RParen)?;
            VariantData::Tuple(types)
        } else if self.at(TokenKind::LBrace) {
            self.advance();
            let mut fields = Vec::new();
            while !self.at(TokenKind::RBrace) {
                fields.push(self.parse_field()?);
                if !self.at(TokenKind::RBrace) {
                    if self.at(TokenKind::Comma) {
                        self.advance();
                    }
                }
            }
            self.expect(TokenKind::RBrace)?;
            VariantData::Struct(fields)
        } else {
            VariantData::Unit
        };

        Ok(VariantDef {
            id: self.next_id(),
            name,
            data,
        })
    }

    // ==================== TRAITS & IMPL ====================

    fn parse_trait(&mut self, visibility: Visibility, _modifiers: Modifiers) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Trait)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        let supertraits = if self.at(TokenKind::Colon) {
            self.advance();
            let mut traits = vec![self.parse_path()?];
            while self.at(TokenKind::Plus) {
                self.advance();
                traits.push(self.parse_path()?);
            }
            traits
        } else {
            Vec::new()
        };
        let where_clause = self.parse_where_clause()?;

        self.expect(TokenKind::LBrace)?;
        let mut items = Vec::new();
        while !self.at(TokenKind::RBrace) {
            items.push(self.parse_trait_item()?);
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Trait(TraitDef {
            id: self.next_id(),
            visibility,
            name,
            generics,
            supertraits,
            where_clause,
            items,
            span: start.merge(end),
        }))
    }

    fn parse_trait_item(&mut self) -> Result<TraitItem> {
        let visibility = self.parse_visibility();
        let modifiers = self.parse_modifiers();

        match self.peek() {
            TokenKind::Fn => {
                self.advance();
                let name = self.parse_ident()?;
                let generics = self.parse_generics()?;
                let params = self.parse_params()?;
                let return_type = self.parse_return_type()?;
                let effects = self.parse_effect_clause()?;
                let where_clause = self.parse_where_clause()?;

                let default_body = if self.at(TokenKind::LBrace) {
                    Some(self.parse_block()?)
                } else {
                    self.expect(TokenKind::Semi)?;
                    None
                };

                Ok(TraitItem::Fn(TraitFnDef {
                    id: self.next_id(),
                    name,
                    generics,
                    params,
                    return_type,
                    effects,
                    where_clause,
                    default_body,
                }))
            }
            TokenKind::Type => {
                self.advance();
                let name = self.parse_ident()?;
                let bounds = if self.at(TokenKind::Colon) {
                    self.advance();
                    let mut bounds = vec![self.parse_path()?];
                    while self.at(TokenKind::Plus) {
                        self.advance();
                        bounds.push(self.parse_path()?);
                    }
                    bounds
                } else {
                    Vec::new()
                };
                let default = if self.at(TokenKind::Eq) {
                    self.advance();
                    Some(self.parse_type()?)
                } else {
                    None
                };
                self.expect(TokenKind::Semi)?;

                Ok(TraitItem::Type(TraitTypeDef {
                    id: self.next_id(),
                    name,
                    bounds,
                    default,
                }))
            }
            _ => Err(miette::miette!(
                "Expected trait item, found {:?}",
                self.peek()
            )),
        }
    }

    fn parse_impl(&mut self) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Impl)?;

        let generics = self.parse_generics()?;

        // Check if this is a trait impl
        let (trait_ref, target_type) = if self.peek_n(1) == TokenKind::For {
            let trait_path = self.parse_path()?;
            self.expect(TokenKind::For)?;
            let ty = self.parse_type()?;
            (Some(trait_path), ty)
        } else {
            (None, self.parse_type()?)
        };

        let where_clause = self.parse_where_clause()?;

        self.expect(TokenKind::LBrace)?;
        let mut items = Vec::new();
        while !self.at(TokenKind::RBrace) {
            items.push(self.parse_impl_item()?);
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Impl(ImplDef {
            id: self.next_id(),
            generics,
            trait_ref,
            target_type,
            where_clause,
            items,
            span: start.merge(end),
        }))
    }

    fn parse_impl_item(&mut self) -> Result<ImplItem> {
        // Skip doc comments before impl items
        while self.at(TokenKind::DocCommentOuter) || self.at(TokenKind::DocCommentInner) {
            self.advance();
        }

        let attributes = self.parse_item_attributes()?;
        let visibility = self.parse_visibility();
        let modifiers = self.parse_modifiers();

        match self.peek() {
            TokenKind::Fn | TokenKind::Kernel => {
                let item = self.parse_fn(visibility, modifiers, attributes)?;
                if let Item::Function(f) = item {
                    Ok(ImplItem::Fn(f))
                } else {
                    unreachable!()
                }
            }
            TokenKind::Type => {
                self.advance();
                let name = self.parse_ident()?;
                self.expect(TokenKind::Eq)?;
                let ty = self.parse_type()?;
                self.expect(TokenKind::Semi)?;
                Ok(ImplItem::Type(ImplTypeDef {
                    id: self.next_id(),
                    name,
                    ty,
                }))
            }
            _ => Err(miette::miette!(
                "Expected impl item, found {:?}",
                self.peek()
            )),
        }
    }

    // ==================== TYPE ALIASES ====================

    fn parse_type_alias(&mut self, visibility: Visibility) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Type)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        self.expect(TokenKind::Eq)?;
        // Capture span of the type expression itself
        let ty_start = self.span();
        let ty = self.parse_type()?;
        let ty_end = self.span();
        self.expect(TokenKind::Semi)?;

        let _end = self.span();

        Ok(Item::TypeAlias(TypeAliasDef {
            id: self.next_id(),
            visibility,
            name,
            generics,
            ty,
            span: ty_start.merge(ty_end), // Use type expression span, not whole declaration
        }))
    }

    // ==================== EFFECTS ====================

    fn parse_effect(&mut self, visibility: Visibility) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Effect)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;

        self.expect(TokenKind::LBrace)?;
        let mut operations = Vec::new();
        while !self.at(TokenKind::RBrace) {
            operations.push(self.parse_effect_op()?);
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Effect(EffectDef {
            id: self.next_id(),
            visibility,
            name,
            generics,
            operations,
            span: start.merge(end),
        }))
    }

    fn parse_effect_op(&mut self) -> Result<EffectOpDef> {
        self.expect(TokenKind::Fn)?;
        let name = self.parse_ident()?;
        let params = self.parse_params()?;
        let return_type = self.parse_return_type()?;
        self.expect(TokenKind::Semi)?;

        Ok(EffectOpDef {
            id: self.next_id(),
            name,
            params,
            return_type,
        })
    }

    fn parse_handler(&mut self, visibility: Visibility) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Handler)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        self.expect(TokenKind::For)?;
        let effect = self.parse_path()?;

        self.expect(TokenKind::LBrace)?;
        let mut cases = Vec::new();
        while !self.at(TokenKind::RBrace) {
            cases.push(self.parse_handler_case()?);
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Handler(HandlerDef {
            id: self.next_id(),
            visibility,
            name,
            generics,
            effect,
            cases,
            span: start.merge(end),
        }))
    }

    fn parse_handler_case(&mut self) -> Result<HandlerCase> {
        let name = self.parse_ident()?;
        let params = self.parse_params()?;
        self.expect(TokenKind::FatArrow)?;
        let body = self.parse_expr()?;
        if self.at(TokenKind::Comma) {
            self.advance();
        }

        Ok(HandlerCase {
            id: self.next_id(),
            name,
            params,
            body,
        })
    }

    // ==================== MODULES ====================

    /// Parse module declaration: `pub module foo { ... }` or `mod foo;`
    fn parse_module_decl(&mut self, visibility: Visibility) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Module)?;
        let name = self.parse_ident()?;

        if self.at(TokenKind::LBrace) {
            // Inline module: `module foo { ... }`
            self.advance();
            let mut items = Vec::new();
            while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                items.push(self.parse_item()?);
            }
            self.expect(TokenKind::RBrace)?;
            let end = self.span();

            Ok(Item::Module(ModuleDef {
                id: self.next_id(),
                visibility,
                name,
                items: Some(items),
                span: start.merge(end),
            }))
        } else {
            // File module: `mod foo;`
            self.expect(TokenKind::Semi)?;
            let end = self.span();

            Ok(Item::Module(ModuleDef {
                id: self.next_id(),
                visibility,
                name,
                items: None,
                span: start.merge(end),
            }))
        }
    }

    // ==================== IMPORTS & EXTERN ====================

    /// Parse import with visibility for `pub use` re-exports
    fn parse_import_with_visibility(&mut self, visibility: Visibility) -> Result<Item> {
        let is_reexport = matches!(visibility, Visibility::Public);
        self.parse_import_inner(is_reexport)
    }

    /// Legacy parse_import for backward compatibility
    fn parse_import(&mut self) -> Result<Item> {
        self.parse_import_inner(false)
    }

    /// Parse import/use statement with full syntax support
    /// Supports:
    /// - `import path;` or `use path;` - import entire module
    /// - `use path::{A, B, C};` - selective imports (Rust-style)
    /// - `use path::*;` - glob import
    /// - `use path as alias;` - renamed import
    /// - `import { A, B } from path;` - selective imports (Darwin Atlas style)
    /// - `pub use path::Item;` - re-export
    fn parse_import_inner(&mut self, is_reexport: bool) -> Result<Item> {
        let start = self.span();

        // Accept both 'import' and 'use' keywords (Darwin Atlas compatibility)
        if self.at(TokenKind::Import) {
            self.advance();
        } else if self.at(TokenKind::Use) {
            self.advance();
        } else {
            return Err(miette::miette!("Expected 'import' or 'use'"));
        }

        // Check for `import { items } from path;` syntax (Darwin Atlas style)
        if self.at(TokenKind::LBrace) {
            return self.parse_import_from_syntax(start, is_reexport);
        }

        // Parse the base path
        let path = self.parse_path()?;

        // Check for `use path::{items}` Rust-style syntax
        if self.at(TokenKind::ColonColon) {
            let next = self.peek_n(1);
            if next == TokenKind::LBrace {
                self.advance(); // consume ::
                let items = self.parse_import_items()?;
                self.expect(TokenKind::Semi)?;
                let end = self.span();
                return Ok(Item::Import(ImportDef {
                    id: self.next_id(),
                    path,
                    items: Some(items),
                    is_reexport,
                    span: start.merge(end),
                }));
            } else if next == TokenKind::Star {
                // Glob import: `use path::*`
                self.advance(); // consume ::
                self.advance(); // consume *
                self.expect(TokenKind::Semi)?;
                let end = self.span();
                return Ok(Item::Import(ImportDef {
                    id: self.next_id(),
                    path,
                    items: Some(vec![ImportItem {
                        name: "*".to_string(),
                        alias: None,
                        is_glob: true,
                    }]),
                    is_reexport,
                    span: start.merge(end),
                }));
            }
        }

        // Check for rename: `use path as alias`
        if self.at(TokenKind::As) {
            self.advance();
            let alias = self.parse_ident()?;
            self.expect(TokenKind::Semi)?;
            let end = self.span();

            // The last segment of the path is the item being renamed
            let item_name = path.segments.last().cloned().unwrap_or_default();
            let module_path = if path.segments.len() > 1 {
                Path {
                    segments: path.segments[..path.segments.len() - 1].to_vec(),
                    source_module: path.source_module.clone(),
                    resolved_module: None,
                }
            } else {
                path.clone()
            };

            return Ok(Item::Import(ImportDef {
                id: self.next_id(),
                path: module_path,
                items: Some(vec![ImportItem {
                    name: item_name,
                    alias: Some(alias),
                    is_glob: false,
                }]),
                is_reexport,
                span: start.merge(end),
            }));
        }

        // Simple `import path;` syntax
        self.expect(TokenKind::Semi)?;
        let end = self.span();

        Ok(Item::Import(ImportDef {
            id: self.next_id(),
            path,
            items: None,
            is_reexport,
            span: start.merge(end),
        }))
    }

    /// Parse import items: `{ A, B as C, * }`
    fn parse_import_items(&mut self) -> Result<Vec<ImportItem>> {
        self.expect(TokenKind::LBrace)?;
        let mut items = Vec::new();

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            // Check for glob: `*`
            if self.at(TokenKind::Star) {
                self.advance();
                items.push(ImportItem {
                    name: "*".to_string(),
                    alias: None,
                    is_glob: true,
                });
            } else {
                let name = self.parse_ident()?;
                let alias = if self.at(TokenKind::As) {
                    self.advance();
                    Some(self.parse_ident()?)
                } else {
                    None
                };
                items.push(ImportItem {
                    name,
                    alias,
                    is_glob: false,
                });
            }

            if self.at(TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(items)
    }

    /// Parse `import { A, B as C } from path;` syntax (Darwin Atlas style)
    fn parse_import_from_syntax(&mut self, start: Span, is_reexport: bool) -> Result<Item> {
        let items = self.parse_import_items()?;
        self.expect(TokenKind::From)?;
        let path = self.parse_path()?;
        self.expect(TokenKind::Semi)?;
        let end = self.span();

        Ok(Item::Import(ImportDef {
            id: self.next_id(),
            path,
            items: Some(items),
            is_reexport,
            span: start.merge(end),
        }))
    }

    /// Parse export block: `export { Item1, Item2, ... };`
    fn parse_export(&mut self) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Export)?;

        let mut names = Vec::new();

        if self.at(TokenKind::LBrace) {
            self.advance(); // consume {
            while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                names.push(self.parse_ident()?);
                if self.at(TokenKind::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(TokenKind::RBrace)?;
        } else {
            // Single export: `export foo;`
            names.push(self.parse_ident()?);
        }

        self.expect(TokenKind::Semi)?;
        let end = self.span();

        // For now, exports are parsed but not fully processed
        // We return a placeholder Import item (exports affect visibility, not AST structure)
        Ok(Item::Export(ExportDef {
            id: self.next_id(),
            names,
            span: start.merge(end),
        }))
    }

    /// Parse ontology import: `ontology chebi from "https://...";`
    fn parse_ontology_import(&mut self) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Ontology)?;

        let name = self.parse_ident()?;
        self.expect(TokenKind::From)?;

        // Parse the source URL/path as a string literal
        let source = if self.at(TokenKind::StringLit) {
            let s = self.advance().text.clone();
            // Remove quotes
            s[1..s.len() - 1].to_string()
        } else {
            return Err(miette::miette!(
                "Expected string literal for ontology source"
            ));
        };

        self.expect(TokenKind::Semi)?;
        let end = self.span();

        Ok(Item::OntologyImport(OntologyImportDef {
            id: self.next_id(),
            prefix: name,
            source,
            alias: None,
            span: start.merge(end),
        }))
    }

    /// Parse alignment declaration: `align chebi:drug ~ drugbank:drug with distance 0.1;`
    fn parse_align_decl(&mut self) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Align)?;

        // Parse first term (ontology:term)
        let term1 = self.parse_ontology_term_ref()?;

        // Expect ~
        self.expect(TokenKind::Tilde)?;

        // Parse second term
        let term2 = self.parse_ontology_term_ref()?;

        // Expect "with distance"
        self.expect(TokenKind::With)?;
        self.expect(TokenKind::Distance)?;

        // Parse distance value
        let distance = if self.at(TokenKind::FloatLit) {
            let s = self.advance().text.clone();
            s.parse::<f64>()
                .map_err(|_| miette::miette!("Invalid distance value"))?
        } else if self.at(TokenKind::IntLit) {
            let s = self.advance().text.clone();
            s.parse::<f64>()
                .map_err(|_| miette::miette!("Invalid distance value"))?
        } else {
            return Err(miette::miette!("Expected numeric distance value"));
        };

        self.expect(TokenKind::Semi)?;
        let end = self.span();

        Ok(Item::AlignDecl(AlignDef {
            id: self.next_id(),
            type1: term1,
            type2: term2,
            distance,
            span: start.merge(end),
        }))
    }

    // ==================== ODE/PDE PARSING ====================

    /// Parse ODE definition:
    /// ```d
    /// ode LotkaVolterra {
    ///     params: { alpha: f64, beta: f64 }
    ///     state: { prey: f64, predator: f64 }
    ///     d(prey)/dt = alpha * prey - beta * prey * predator
    ///     d(predator)/dt = delta * prey * predator - gamma * predator
    /// }
    /// ```
    fn parse_ode_def(&mut self, visibility: Visibility) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Ode)?;

        let name = self.parse_ident()?;
        self.expect(TokenKind::LBrace)?;

        let mut params = Vec::new();
        let mut state = Vec::new();
        let mut equations = Vec::new();

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            if self.at(TokenKind::Params) {
                self.advance();
                self.expect(TokenKind::Colon)?;
                params = self.parse_ode_params_block()?;
            } else if self.at(TokenKind::State) {
                self.advance();
                self.expect(TokenKind::Colon)?;
                state = self.parse_ode_state_block()?;
            } else if self.at(TokenKind::Ident) && self.current().text == "d" {
                equations.push(self.parse_ode_equation()?);
            } else {
                // Skip unknown tokens within the block
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;
        let end = self.span();

        Ok(Item::OdeDef(OdeDef {
            id: self.next_id(),
            visibility,
            name,
            params,
            state,
            equations,
            span: start.merge(end),
        }))
    }

    /// Parse params block: `{ alpha: f64, beta: f64 = 1.0 }`
    fn parse_ode_params_block(&mut self) -> Result<Vec<OdeParam>> {
        self.expect(TokenKind::LBrace)?;
        let mut params = Vec::new();

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let start = self.span();
            let name = self.parse_ident()?;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;

            let default = if self.at(TokenKind::Eq) {
                self.advance();
                Some(self.parse_expr()?)
            } else {
                None
            };

            params.push(OdeParam {
                id: self.next_id(),
                name,
                ty,
                default,
                span: start.merge(self.span()),
            });

            if !self.at(TokenKind::RBrace) {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(params)
    }

    /// Parse state block: `{ prey: f64, predator: f64 }`
    fn parse_ode_state_block(&mut self) -> Result<Vec<OdeStateVar>> {
        self.expect(TokenKind::LBrace)?;
        let mut vars = Vec::new();

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let start = self.span();
            let name = self.parse_ident()?;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;

            vars.push(OdeStateVar {
                id: self.next_id(),
                name,
                ty,
                span: start.merge(self.span()),
            });

            if !self.at(TokenKind::RBrace) {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(vars)
    }

    /// Parse ODE equation: `d(prey)/dt = alpha * prey - beta * prey * predator`
    fn parse_ode_equation(&mut self) -> Result<OdeEquation> {
        let start = self.span();

        // Expect 'd' identifier
        let d = self.parse_ident()?;
        if d != "d" {
            return Err(miette::miette!(
                "Expected 'd' for derivative, found '{}'",
                d
            ));
        }

        // Parse (variable)
        self.expect(TokenKind::LParen)?;
        let variable = self.parse_ident()?;
        self.expect(TokenKind::RParen)?;

        // Parse /dt
        self.expect(TokenKind::Slash)?;
        let dt = self.parse_ident()?;
        if dt != "dt" {
            return Err(miette::miette!("Expected 'dt', found '{}'", dt));
        }

        // Parse = expr
        self.expect(TokenKind::Eq)?;
        let rhs = self.parse_expr()?;

        Ok(OdeEquation {
            id: self.next_id(),
            variable,
            rhs,
            span: start.merge(self.span()),
        })
    }

    /// Parse PDE definition:
    /// ```d
    /// pde HeatEquation {
    ///     params: { alpha: f64 }
    ///     domain: [0, 1] x [0, 1]
    ///     equation: du/dt = alpha * laplacian(u)
    ///     boundary: { x=0: u=0, x=1: u=0 }
    /// }
    /// ```
    fn parse_pde_def(&mut self, visibility: Visibility) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Pde)?;

        let name = self.parse_ident()?;
        self.expect(TokenKind::LBrace)?;

        let mut params = Vec::new();
        let mut domain = None;
        let mut equation = None;
        let mut boundary_conditions = Vec::new();
        let mut initial_condition = None;

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            if self.at(TokenKind::Params) {
                self.advance();
                self.expect(TokenKind::Colon)?;
                params = self.parse_ode_params_block()?;
            } else if self.at(TokenKind::Domain) {
                self.advance();
                self.expect(TokenKind::Colon)?;
                domain = Some(self.parse_pde_domain()?);
            } else if self.at(TokenKind::Ident) && self.current().text == "equation" {
                self.advance();
                self.expect(TokenKind::Colon)?;
                equation = Some(self.parse_pde_equation()?);
            } else if self.at(TokenKind::Boundary) {
                self.advance();
                self.expect(TokenKind::Colon)?;
                boundary_conditions = self.parse_boundary_conditions()?;
            } else if self.at(TokenKind::Initial) {
                self.advance();
                self.expect(TokenKind::Colon)?;
                initial_condition = Some(self.parse_expr()?);
            } else {
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;
        let end = self.span();

        // Create default domain if not specified
        let domain = domain.unwrap_or_else(|| PdeDomain {
            id: self.next_id(),
            dimensions: vec![],
            span: start.merge(end),
        });

        // Create default equation if not specified
        let equation = equation.unwrap_or_else(|| PdeEquation {
            id: self.next_id(),
            variable: "u".to_string(),
            time_order: 1,
            rhs: Expr::Literal {
                id: self.next_id(),
                value: Literal::Int(0),
            },
            span: start.merge(end),
        });

        Ok(Item::PdeDef(PdeDef {
            id: self.next_id(),
            visibility,
            name,
            params,
            domain,
            equation,
            boundary_conditions,
            initial_condition,
            span: start.merge(end),
        }))
    }

    /// Parse PDE domain: `[0, 1] x [0, 1]` or `[0, L]`
    fn parse_pde_domain(&mut self) -> Result<PdeDomain> {
        let start = self.span();
        let mut dimensions = Vec::new();

        // Parse first dimension [min, max]
        dimensions.push(self.parse_pde_dimension("x")?);

        // Parse additional dimensions separated by 'x'
        while self.at(TokenKind::Ident) && self.current().text == "x" {
            self.advance();
            let dim_name = if dimensions.len() == 1 { "y" } else { "z" };
            dimensions.push(self.parse_pde_dimension(dim_name)?);
        }

        Ok(PdeDomain {
            id: self.next_id(),
            dimensions,
            span: start.merge(self.span()),
        })
    }

    /// Parse single dimension: `[min, max]`
    fn parse_pde_dimension(&mut self, name: &str) -> Result<PdeDimension> {
        self.expect(TokenKind::LBracket)?;
        let min = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let max = self.parse_expr()?;
        self.expect(TokenKind::RBracket)?;

        Ok(PdeDimension {
            name: name.to_string(),
            min,
            max,
        })
    }

    /// Parse PDE equation: `du/dt = alpha * laplacian(u)`
    fn parse_pde_equation(&mut self) -> Result<PdeEquation> {
        let start = self.span();

        // Parse du/dt or d2u/dt2
        let d = self.parse_ident()?;
        let (variable, time_order) = if d.starts_with("d2") || d == "d2" {
            // Second order: d2u/dt2
            let var = if d.len() > 2 {
                d[2..].to_string()
            } else {
                self.parse_ident()?
            };
            (var, 2)
        } else if d.starts_with('d') {
            // First order: du/dt
            let var = if d.len() > 1 {
                d[1..].to_string()
            } else {
                self.parse_ident()?
            };
            (var, 1)
        } else {
            return Err(miette::miette!("Expected derivative like 'du/dt'"));
        };

        self.expect(TokenKind::Slash)?;
        let _dt = self.parse_ident()?; // dt or dt2

        self.expect(TokenKind::Eq)?;
        let rhs = self.parse_expr()?;

        Ok(PdeEquation {
            id: self.next_id(),
            variable,
            time_order,
            rhs,
            span: start.merge(self.span()),
        })
    }

    /// Parse boundary conditions: `{ x=0: u=0, x=1: u=0 }`
    fn parse_boundary_conditions(&mut self) -> Result<Vec<BoundaryConditionDef>> {
        self.expect(TokenKind::LBrace)?;
        let mut conditions = Vec::new();

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let start = self.span();

            // Parse boundary spec: x=0 or y=1
            let variable = self.parse_ident()?;
            self.expect(TokenKind::Eq)?;
            let value = self.parse_expr()?;

            self.expect(TokenKind::Colon)?;

            // Parse condition type
            let condition = if self.at(TokenKind::Ident) && self.current().text == "periodic" {
                self.advance();
                BoundaryConditionType::Periodic
            } else {
                // Parse u = expr or du/dn = expr
                let lhs = self.parse_ident()?;
                if lhs.starts_with('d') {
                    // Neumann: du/dn = value
                    self.expect(TokenKind::Slash)?;
                    let _dn = self.parse_ident()?;
                    self.expect(TokenKind::Eq)?;
                    let val = self.parse_expr()?;
                    BoundaryConditionType::Neumann(val)
                } else {
                    // Dirichlet: u = value
                    self.expect(TokenKind::Eq)?;
                    let val = self.parse_expr()?;
                    BoundaryConditionType::Dirichlet(val)
                }
            };

            conditions.push(BoundaryConditionDef {
                id: self.next_id(),
                boundary: BoundarySpec { variable, value },
                condition,
                span: start.merge(self.span()),
            });

            if !self.at(TokenKind::RBrace) {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(conditions)
    }

    // ==================== CAUSAL MODEL PARSING ====================

    /// Parse causal model definition:
    /// ```d
    /// causal model SmokingCancer {
    ///     nodes: [Smoking, Tar, Cancer, Genetics]
    ///
    ///     Genetics -> Smoking
    ///     Genetics -> Cancer
    ///     Smoking -> Tar
    ///     Tar -> Cancer
    ///
    ///     equations: {
    ///         Smoking = 0.5 * Genetics + noise,
    ///         Tar = 0.8 * Smoking + noise,
    ///         Cancer = 0.6 * Tar + 0.3 * Genetics + noise
    ///     }
    /// }
    /// ```
    fn parse_causal_model_def(&mut self, visibility: Visibility) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Causal)?;

        // Optional 'model' keyword (causal model X { } or causal X { })
        if self.at(TokenKind::Ident) && self.current().text == "model" {
            self.advance();
        }

        let name = self.parse_ident()?;
        self.expect(TokenKind::LBrace)?;

        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut equations = Vec::new();

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            if self.at(TokenKind::Nodes) {
                // nodes: [A, B, C]
                self.advance();
                self.expect(TokenKind::Colon)?;
                nodes = self.parse_causal_nodes_list()?;
            } else if self.at(TokenKind::Equations) {
                // equations: { X = expr, ... }
                self.advance();
                self.expect(TokenKind::Colon)?;
                equations = self.parse_causal_equations_block()?;
            } else if self.at(TokenKind::Edges) {
                // edges: [A -> B, C -> D]
                self.advance();
                self.expect(TokenKind::Colon)?;
                edges.extend(self.parse_causal_edges_list()?);
            } else if self.at(TokenKind::Ident) {
                // Could be an edge: A -> B
                if self.peek_n(1) == TokenKind::Arrow {
                    edges.push(self.parse_causal_edge()?);
                } else {
                    // Skip unknown token
                    self.advance();
                }
            } else {
                // Skip unknown tokens within the block
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;
        let end = self.span();

        Ok(Item::CausalModel(CausalModelDef {
            id: self.next_id(),
            visibility,
            name,
            nodes,
            edges,
            equations,
            span: start.merge(end),
        }))
    }

    /// Parse causal nodes list: `[A, B, C]` or `[A: f64, B: f64]`
    fn parse_causal_nodes_list(&mut self) -> Result<Vec<CausalNode>> {
        self.expect(TokenKind::LBracket)?;
        let mut nodes = Vec::new();

        while !self.at(TokenKind::RBracket) && !self.at(TokenKind::Eof) {
            let start = self.span();
            let name = self.parse_ident()?;

            // Optional type annotation
            let ty = if self.at(TokenKind::Colon) {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };

            nodes.push(CausalNode {
                id: self.next_id(),
                name,
                ty,
                span: start.merge(self.span()),
            });

            if !self.at(TokenKind::RBracket) {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::RBracket)?;
        Ok(nodes)
    }

    /// Parse causal edges list: `[A -> B, C -> D]`
    fn parse_causal_edges_list(&mut self) -> Result<Vec<CausalEdge>> {
        self.expect(TokenKind::LBracket)?;
        let mut edges = Vec::new();

        while !self.at(TokenKind::RBracket) && !self.at(TokenKind::Eof) {
            edges.push(self.parse_causal_edge()?);

            if !self.at(TokenKind::RBracket) {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::RBracket)?;
        Ok(edges)
    }

    /// Parse a single causal edge: `A -> B`
    fn parse_causal_edge(&mut self) -> Result<CausalEdge> {
        let start = self.span();
        let from = self.parse_ident()?;
        self.expect(TokenKind::Arrow)?;
        let to = self.parse_ident()?;

        Ok(CausalEdge {
            id: self.next_id(),
            from,
            to,
            span: start.merge(self.span()),
        })
    }

    /// Parse causal equations block: `{ X = expr, Y = expr }`
    fn parse_causal_equations_block(&mut self) -> Result<Vec<CausalEquation>> {
        self.expect(TokenKind::LBrace)?;
        let mut equations = Vec::new();

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let start = self.span();
            let variable = self.parse_ident()?;
            self.expect(TokenKind::Eq)?;
            let rhs = self.parse_expr()?;

            equations.push(CausalEquation {
                id: self.next_id(),
                variable,
                rhs,
                span: start.merge(self.span()),
            });

            // Optional comma or semicolon separator
            if self.at(TokenKind::Comma) || self.at(TokenKind::Semi) {
                self.advance();
            }
        }

        self.expect(TokenKind::RBrace)?;
        Ok(equations)
    }

    /// Parse ontology term reference: `chebi:drug` or `SNOMED:12345`
    fn parse_ontology_term_ref(&mut self) -> Result<OntologyTermRef> {
        let start = self.span();

        let ontology = self.parse_ident()?;
        self.expect(TokenKind::Colon)?;

        // Term can be an identifier or a number
        let term = if self.at(TokenKind::Ident) {
            self.advance().text.clone()
        } else if self.at(TokenKind::IntLit) {
            self.advance().text.clone()
        } else {
            return Err(miette::miette!("Expected term identifier or number"));
        };

        let end = self.span();

        Ok(OntologyTermRef {
            id: self.next_id(),
            prefix: ontology,
            term,
            span: start.merge(end),
        })
    }

    fn parse_extern(&mut self) -> Result<Item> {
        self.parse_extern_with_visibility(Visibility::Private)
    }

    fn parse_extern_with_visibility(&mut self, visibility: Visibility) -> Result<Item> {
        let start = self.span();
        self.expect(TokenKind::Extern)?;

        let abi = if self.at(TokenKind::StringLit) {
            let s = self.advance().text.clone();
            // Remove quotes
            let abi_str = &s[1..s.len() - 1];
            Abi::from_str(abi_str)
        } else {
            Abi::C
        };

        // Check if this is `extern fn` (function with C ABI) or `extern { }` (extern block)
        if self.at(TokenKind::Fn) || self.at(TokenKind::Kernel) {
            // extern fn - parse as a regular function with the specified ABI
            return self.parse_extern_fn_def(visibility, abi, start);
        }

        // extern { } block
        self.expect(TokenKind::LBrace)?;
        let mut items = Vec::new();
        while !self.at(TokenKind::RBrace) {
            items.push(self.parse_extern_item()?);
        }
        self.expect(TokenKind::RBrace)?;

        let end = self.span();

        Ok(Item::Extern(ExternBlock {
            id: self.next_id(),
            abi,
            items,
            span: start.merge(end),
        }))
    }

    /// Parse `extern "C" fn name(...) { ... }` - a function definition with explicit ABI
    fn parse_extern_fn_def(
        &mut self,
        visibility: Visibility,
        abi: Abi,
        start: Span,
    ) -> Result<Item> {
        let is_kernel = if self.at(TokenKind::Kernel) {
            self.advance();
            true
        } else {
            false
        };

        self.expect(TokenKind::Fn)?;

        let name = self.parse_ident()?;
        let generics = self.parse_generics()?;
        let params = self.parse_params()?;
        let return_type = self.parse_return_type()?;
        let effects = self.parse_effect_clause()?;
        let where_clause = self.parse_where_clause()?;
        let body = self.parse_block()?;

        let end = self.span();

        Ok(Item::Function(FnDef {
            id: self.next_id(),
            visibility,
            modifiers: FnModifiers {
                is_async: false,
                is_unsafe: false,
                is_kernel,
                abi: Some(abi),
            },
            attributes: vec![],
            name,
            generics,
            params,
            return_type,
            effects,
            where_clause,
            body,
            span: start.merge(end),
        }))
    }

    fn parse_extern_item(&mut self) -> Result<ExternItem> {
        // Check for #[link_name = "..."] attribute
        let link_name = self.parse_link_name_attr()?;

        match self.peek() {
            TokenKind::Fn => {
                let func = self.parse_extern_fn(link_name)?;
                Ok(ExternItem::Fn(func))
            }
            TokenKind::Static => {
                let static_item = self.parse_extern_static(link_name)?;
                Ok(ExternItem::Static(static_item))
            }
            TokenKind::Type => {
                let type_item = self.parse_extern_type()?;
                Ok(ExternItem::Type(type_item))
            }
            _ => {
                let tok = self.current().clone();
                Err(miette::miette!(
                    labels = vec![miette::LabeledSpan::at(
                        tok.span.start..tok.span.end,
                        "expected fn, static, or type"
                    )],
                    "expected extern item, found {:?}",
                    tok.kind
                ))
            }
        }
    }

    fn parse_link_name_attr(&mut self) -> Result<Option<String>> {
        if !self.at(TokenKind::Hash) {
            return Ok(None);
        }
        self.advance(); // #
        self.expect(TokenKind::LBracket)?;

        let ident = self.parse_ident()?;
        if ident != "link_name" {
            // Skip unknown attribute
            while !self.at(TokenKind::RBracket) && !self.at(TokenKind::Eof) {
                self.advance();
            }
            self.expect(TokenKind::RBracket)?;
            return Ok(None);
        }

        self.expect(TokenKind::Eq)?;
        let link_name = if self.at(TokenKind::StringLit) {
            let s = self.advance().text.clone();
            s[1..s.len() - 1].to_string()
        } else {
            return Err(miette::miette!("expected string literal for link_name"));
        };
        self.expect(TokenKind::RBracket)?;
        Ok(Some(link_name))
    }

    fn parse_extern_fn(&mut self, link_name: Option<String>) -> Result<ExternFn> {
        let start = self.span();
        self.expect(TokenKind::Fn)?;
        let name = self.parse_ident()?;

        // Parse parameters, handling variadic
        self.expect(TokenKind::LParen)?;
        let mut params = Vec::new();
        let mut is_variadic = false;

        while !self.at(TokenKind::RParen) {
            // Check for variadic ...
            if self.at(TokenKind::DotDotDot) {
                self.advance();
                is_variadic = true;
                break;
            }

            params.push(self.parse_param()?);

            if !self.at(TokenKind::RParen) {
                self.expect(TokenKind::Comma)?;
                // Allow trailing comma before ...
                if self.at(TokenKind::DotDotDot) {
                    self.advance();
                    is_variadic = true;
                    break;
                }
            }
        }
        self.expect(TokenKind::RParen)?;

        let return_type = self.parse_return_type()?;
        self.expect(TokenKind::Semi)?;

        let end = self.span();

        Ok(ExternFn {
            id: self.next_id(),
            name,
            params,
            return_type,
            is_variadic,
            link_name,
            span: start.merge(end),
        })
    }

    fn parse_extern_static(&mut self, link_name: Option<String>) -> Result<ExternStatic> {
        let start = self.span();
        self.expect(TokenKind::Static)?;

        let is_mut = if self.at(TokenKind::Mut) {
            self.advance();
            true
        } else {
            false
        };

        let name = self.parse_ident()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type()?;
        self.expect(TokenKind::Semi)?;

        let end = self.span();

        Ok(ExternStatic {
            id: self.next_id(),
            name,
            ty,
            is_mut,
            link_name,
            span: start.merge(end),
        })
    }

    fn parse_extern_type(&mut self) -> Result<ExternType> {
        let start = self.span();
        self.expect(TokenKind::Type)?;
        let name = self.parse_ident()?;
        self.expect(TokenKind::Semi)?;
        let end = self.span();

        Ok(ExternType {
            id: self.next_id(),
            name,
            span: start.merge(end),
        })
    }

    // ==================== GLOBALS ====================

    fn parse_global(&mut self, visibility: Visibility, modifiers: Modifiers) -> Result<Item> {
        let start = self.span();
        let is_const = self.at(TokenKind::Const);
        self.advance(); // let or const

        let is_mut = if self.at(TokenKind::Mut) && !is_const {
            self.advance();
            true
        } else {
            false
        };

        let pattern = self.parse_pattern()?;
        let ty = if self.at(TokenKind::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(TokenKind::Eq)?;
        let value = self.parse_expr()?;

        let end = self.span();

        Ok(Item::Global(GlobalDef {
            id: self.next_id(),
            visibility,
            is_const,
            is_mut,
            pattern,
            ty,
            value,
            span: start.merge(end),
        }))
    }

    // ==================== GENERICS ====================

    fn parse_generics(&mut self) -> Result<Generics> {
        if !self.at(TokenKind::Lt) {
            return Ok(Generics { params: Vec::new() });
        }

        self.advance();
        let mut params = Vec::new();

        // Use at_gt_or_shr for consistency with type arg parsing
        while !self.at_gt_or_shr() {
            params.push(self.parse_generic_param()?);
            if !self.at_gt_or_shr() {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect_gt()?;

        Ok(Generics { params })
    }

    fn parse_generic_param(&mut self) -> Result<GenericParam> {
        // Check for const generic
        if self.at(TokenKind::Const) {
            self.advance();
            let name = self.parse_ident()?;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;
            return Ok(GenericParam::Const { name, ty });
        }

        // Type parameter
        let name = self.parse_ident()?;
        let bounds = if self.at(TokenKind::Colon) {
            self.advance();
            let mut bounds = vec![self.parse_path()?];
            while self.at(TokenKind::Plus) {
                self.advance();
                bounds.push(self.parse_path()?);
            }
            bounds
        } else {
            Vec::new()
        };
        let default = if self.at(TokenKind::Eq) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        Ok(GenericParam::Type {
            name,
            bounds,
            default,
        })
    }

    fn parse_type_args(&mut self) -> Result<Vec<TypeExpr>> {
        self.expect(TokenKind::Lt)?;
        let mut args = Vec::new();

        // Use at_gt_or_shr to handle nested generics like Option<Box<T>>
        while !self.at_gt_or_shr() {
            args.push(self.parse_type()?);
            if !self.at_gt_or_shr() {
                self.expect(TokenKind::Comma)?;
            }
        }

        // Use expect_gt to handle >> splitting for nested generics
        self.expect_gt()?;
        Ok(args)
    }

    fn parse_where_clause(&mut self) -> Result<Vec<WherePredicate>> {
        if !self.at(TokenKind::Where) {
            return Ok(Vec::new());
        }

        self.advance();
        let mut predicates = Vec::new();

        loop {
            let ty = self.parse_type()?;
            self.expect(TokenKind::Colon)?;
            let mut bounds = vec![self.parse_path()?];
            while self.at(TokenKind::Plus) {
                self.advance();
                bounds.push(self.parse_path()?);
            }
            predicates.push(WherePredicate { ty, bounds });

            if self.at(TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(predicates)
    }

    // ==================== TYPES ====================

    pub fn parse_type(&mut self) -> Result<TypeExpr> {
        self.parse_type_with_precedence(0)
    }

    fn parse_type_with_precedence(&mut self, min_prec: u8) -> Result<TypeExpr> {
        let mut left = self.parse_type_primary()?;

        // Handle function types: A -> B
        while self.at(TokenKind::Arrow) && min_prec == 0 {
            self.advance();
            let ret = self.parse_type_with_precedence(1)?;
            left = TypeExpr::Function {
                params: vec![left],
                return_type: Box::new(ret),
                effects: Vec::new(),
            };
        }

        Ok(left)
    }

    fn parse_type_primary(&mut self) -> Result<TypeExpr> {
        match self.peek() {
            // Raw pointer types: *const T or *mut T (for FFI)
            TokenKind::Star => {
                self.advance();
                let is_mut = if self.at(TokenKind::Mut) {
                    self.advance();
                    true
                } else if self.at(TokenKind::Const) {
                    self.advance();
                    false
                } else {
                    // Default to const if neither specified
                    false
                };
                let inner = self.parse_type_primary()?;
                Ok(TypeExpr::RawPointer {
                    mutable: is_mut,
                    inner: Box::new(inner),
                })
            }

            // Reference types
            TokenKind::Amp => {
                self.advance();
                let is_mut = if self.at(TokenKind::Mut) {
                    self.advance();
                    true
                } else {
                    false
                };
                let inner = self.parse_type_primary()?;
                Ok(TypeExpr::Reference {
                    mutable: is_mut,
                    inner: Box::new(inner),
                })
            }

            // Array/slice types
            TokenKind::LBracket => {
                self.advance();
                let element = self.parse_type()?;

                if self.at(TokenKind::Semi) {
                    // Fixed-size array: [T; N]
                    self.advance();
                    let size = self.parse_expr()?;
                    self.expect(TokenKind::RBracket)?;
                    Ok(TypeExpr::Array {
                        element: Box::new(element),
                        size: Some(Box::new(size)),
                    })
                } else {
                    // Slice: [T]
                    self.expect(TokenKind::RBracket)?;
                    Ok(TypeExpr::Array {
                        element: Box::new(element),
                        size: None,
                    })
                }
            }

            // Tuple types
            TokenKind::LParen => {
                self.advance();
                if self.at(TokenKind::RParen) {
                    self.advance();
                    return Ok(TypeExpr::Unit);
                }

                let mut elements = vec![self.parse_type()?];
                while self.at(TokenKind::Comma) {
                    self.advance();
                    if self.at(TokenKind::RParen) {
                        break;
                    }
                    elements.push(self.parse_type()?);
                }
                self.expect(TokenKind::RParen)?;

                if elements.len() == 1 {
                    // Single element with trailing comma is a tuple
                    Ok(TypeExpr::Tuple(elements))
                } else {
                    Ok(TypeExpr::Tuple(elements))
                }
            }

            // Named type or ontology term reference (prefix:term)
            TokenKind::Ident => {
                // Check if this is an ontology term reference: prefix:term (single colon)
                // vs a path: module::Type (double colon)
                // Term can be an identifier (chebi:drug) or a number (chebi:15365)
                if self.peek_n(1) == TokenKind::Colon
                    && matches!(self.peek_n(2), TokenKind::Ident | TokenKind::IntLit)
                {
                    // This is an ontology term reference like chebi:drug or chebi:15365
                    let prefix = self.parse_ident()?;
                    self.expect(TokenKind::Colon)?;
                    let term = if self.at(TokenKind::IntLit) {
                        self.advance().text.clone()
                    } else {
                        self.parse_ident()?
                    };
                    return Ok(TypeExpr::Ontology {
                        ontology: prefix,
                        term: Some(term),
                    });
                }

                // Use parse_type_path to support both :: and . as separators
                // This enables Darwin Atlas compatibility (e.g., &operators.Sequence)
                let path = self.parse_type_path()?;
                let args = if self.at(TokenKind::Lt) {
                    self.parse_type_args()?
                } else {
                    Vec::new()
                };

                // Check for compound unit as standalone type: mL/min, kg*m/s2
                // This is shorthand for f64@mL/min
                if args.is_empty()
                    && path.segments.len() == 1
                    && (self.at(TokenKind::Slash) || self.at(TokenKind::Star))
                {
                    let mut unit_str = path.segments[0].clone();
                    // Parse the rest of the compound unit
                    while self.at(TokenKind::Slash) || self.at(TokenKind::Star) {
                        let op = self.advance().text.clone();
                        unit_str.push_str(&op);
                        if self.at(TokenKind::Ident) || self.at(TokenKind::IntLit) {
                            unit_str.push_str(&self.advance().text);
                        }
                    }
                    // Return as f64 with unit annotation (shorthand expansion)
                    return Ok(TypeExpr::Named {
                        path: Path::simple("f64"),
                        args: Vec::new(),
                        unit: Some(unit_str),
                    });
                }

                // Check for unit annotation: Type@unit (e.g., f64@kg, i32@m/s)
                let unit = if self.at(TokenKind::At) {
                    self.advance();
                    // Parse unit identifier (can include / for compound units)
                    if self.at(TokenKind::Ident) {
                        let mut unit_str = self.advance().text.clone();
                        // Handle compound units like m/s, kg*m/s2
                        while self.at(TokenKind::Slash) || self.at(TokenKind::Star) {
                            let op = self.advance().text.clone();
                            unit_str.push_str(&op);
                            if self.at(TokenKind::Ident) || self.at(TokenKind::IntLit) {
                                unit_str.push_str(&self.advance().text);
                            }
                        }
                        Some(unit_str)
                    } else {
                        None
                    }
                } else {
                    None
                };

                Ok(TypeExpr::Named { path, args, unit })
            }

            // Infer type
            TokenKind::Underscore => {
                self.advance();
                Ok(TypeExpr::Infer)
            }

            // Linear algebra primitives
            TokenKind::Vec2 => {
                self.advance();
                Ok(TypeExpr::Named {
                    path: Path::simple("vec2"),
                    args: Vec::new(),
                    unit: None,
                })
            }
            TokenKind::Vec3 => {
                self.advance();
                Ok(TypeExpr::Named {
                    path: Path::simple("vec3"),
                    args: Vec::new(),
                    unit: None,
                })
            }
            TokenKind::Vec4 => {
                self.advance();
                Ok(TypeExpr::Named {
                    path: Path::simple("vec4"),
                    args: Vec::new(),
                    unit: None,
                })
            }
            TokenKind::Mat2 => {
                self.advance();
                Ok(TypeExpr::Named {
                    path: Path::simple("mat2"),
                    args: Vec::new(),
                    unit: None,
                })
            }
            TokenKind::Mat3 => {
                self.advance();
                Ok(TypeExpr::Named {
                    path: Path::simple("mat3"),
                    args: Vec::new(),
                    unit: None,
                })
            }
            TokenKind::Mat4 => {
                self.advance();
                Ok(TypeExpr::Named {
                    path: Path::simple("mat4"),
                    args: Vec::new(),
                    unit: None,
                })
            }
            TokenKind::Quat => {
                self.advance();
                Ok(TypeExpr::Named {
                    path: Path::simple("quat"),
                    args: Vec::new(),
                    unit: None,
                })
            }
            TokenKind::Dual => {
                self.advance();
                Ok(TypeExpr::Named {
                    path: Path::simple("dual"),
                    args: Vec::new(),
                    unit: None,
                })
            }

            // Knowledge type: Knowledge[T, ε < 0.05, Valid(duration), Derived]
            TokenKind::Knowledge => self.parse_knowledge_type(),

            // Quantity type: Quantity[f64, meters]
            TokenKind::Quantity => self.parse_quantity_type(),

            // Tensor type: Tensor[f32, (batch, channels, height, width)]
            TokenKind::Tensor => self.parse_tensor_type(),

            // Tile type: tile<f16, 16, 16>
            TokenKind::Tile => self.parse_tile_type(),

            // Ontology type: OntologyTerm[SNOMED:12345]
            TokenKind::OntologyTerm => self.parse_ontology_type(),

            _ => Err(miette::miette!("Expected type, found {:?}", self.peek())),
        }
    }

    // ==================== DEMETRIOS EPISTEMIC TYPE PARSING ====================

    /// Parse Knowledge[T, ε < 0.05, Valid(duration), Derived]
    fn parse_knowledge_type(&mut self) -> Result<TypeExpr> {
        self.expect(TokenKind::Knowledge)?;
        self.expect(TokenKind::LBracket)?;

        // Parse the value type
        let value_type = Box::new(self.parse_type()?);

        let mut epsilon = None;
        let mut validity = None;
        let mut provenance = None;

        // Parse optional epistemic parameters
        while self.at(TokenKind::Comma) {
            self.advance();
            if self.at(TokenKind::RBracket) {
                break;
            }

            // Check what kind of parameter this is
            match self.peek() {
                // Epsilon bound: ε < 0.05 or epsilon < 0.05
                TokenKind::Ident
                    if self.current().text == "ε" || self.current().text == "epsilon" =>
                {
                    epsilon = Some(self.parse_epsilon_bound()?);
                }
                // Validity conditions
                TokenKind::Valid | TokenKind::ValidUntil | TokenKind::ValidWhile => {
                    validity = Some(self.parse_validity_condition()?);
                }
                // Provenance markers
                TokenKind::Derived
                | TokenKind::SourceProv
                | TokenKind::Computed
                | TokenKind::Literature
                | TokenKind::Measured
                | TokenKind::InputProv => {
                    provenance = Some(self.parse_provenance_marker()?);
                }
                _ => {
                    // Skip unknown parameter
                    self.advance();
                }
            }
        }

        self.expect(TokenKind::RBracket)?;

        Ok(TypeExpr::Knowledge {
            value_type,
            epsilon,
            validity,
            provenance,
        })
    }

    /// Parse ε < 0.05 or epsilon <= value
    fn parse_epsilon_bound(&mut self) -> Result<EpsilonBound> {
        self.advance(); // skip ε or epsilon

        let operator = match self.peek() {
            TokenKind::Lt => {
                self.advance();
                ComparisonOp::Lt
            }
            TokenKind::Le => {
                self.advance();
                ComparisonOp::Le
            }
            TokenKind::Gt => {
                self.advance();
                ComparisonOp::Gt
            }
            TokenKind::Ge => {
                self.advance();
                ComparisonOp::Ge
            }
            TokenKind::Eq => {
                self.advance();
                ComparisonOp::Eq
            }
            TokenKind::EqEq => {
                self.advance();
                ComparisonOp::Eq
            }
            _ => return Err(miette::miette!("Expected comparison operator after ε")),
        };

        let value = Box::new(self.parse_expr()?);

        Ok(EpsilonBound { operator, value })
    }

    /// Parse Valid(duration), ValidUntil(date), ValidWhile(condition)
    fn parse_validity_condition(&mut self) -> Result<ValidityCondition> {
        let kind = match self.peek() {
            TokenKind::Valid => {
                self.advance();
                ValidityKind::Valid
            }
            TokenKind::ValidUntil => {
                self.advance();
                ValidityKind::ValidUntil
            }
            TokenKind::ValidWhile => {
                self.advance();
                ValidityKind::ValidWhile
            }
            _ => return Err(miette::miette!("Expected validity keyword")),
        };

        self.expect(TokenKind::LParen)?;
        let condition = Box::new(self.parse_expr()?);
        self.expect(TokenKind::RParen)?;

        Ok(ValidityCondition { kind, condition })
    }

    /// Parse Derived, Source(name), Computed, Literature(citation)
    fn parse_provenance_marker(&mut self) -> Result<ProvenanceMarker> {
        let kind = match self.peek() {
            TokenKind::Derived => {
                self.advance();
                ProvenanceKind::Derived
            }
            TokenKind::SourceProv => {
                self.advance();
                ProvenanceKind::Source
            }
            TokenKind::Computed => {
                self.advance();
                ProvenanceKind::Computed
            }
            TokenKind::Literature => {
                self.advance();
                ProvenanceKind::Literature
            }
            TokenKind::Measured => {
                self.advance();
                ProvenanceKind::Measured
            }
            TokenKind::InputProv => {
                self.advance();
                ProvenanceKind::Input
            }
            _ => return Err(miette::miette!("Expected provenance keyword")),
        };

        let source = if self.at(TokenKind::LParen) {
            self.advance();
            let expr = self.parse_expr()?;
            self.expect(TokenKind::RParen)?;
            Some(Box::new(expr))
        } else {
            None
        };

        Ok(ProvenanceMarker { kind, source })
    }

    /// Parse Quantity[f64, meters]
    fn parse_quantity_type(&mut self) -> Result<TypeExpr> {
        self.expect(TokenKind::Quantity)?;
        self.expect(TokenKind::LBracket)?;

        let numeric_type = Box::new(self.parse_type()?);
        self.expect(TokenKind::Comma)?;
        let unit = self.parse_unit_expr()?;

        self.expect(TokenKind::RBracket)?;

        Ok(TypeExpr::Quantity { numeric_type, unit })
    }

    /// Parse unit expression: meters, kg*m/s^2
    fn parse_unit_expr(&mut self) -> Result<UnitExpr> {
        let mut base_units = Vec::new();

        // Parse first unit
        let name = self.parse_ident()?;
        let mut exp = 1i32;
        if self.at(TokenKind::Caret) {
            self.advance();
            let neg = if self.at(TokenKind::Minus) {
                self.advance();
                true
            } else {
                false
            };
            if self.at(TokenKind::IntLit) {
                exp = self.advance().text.parse().unwrap_or(1);
                if neg {
                    exp = -exp;
                }
            }
        }
        base_units.push((name, exp));

        // Parse more units with * or /
        while self.at(TokenKind::Star) || self.at(TokenKind::Slash) {
            let is_div = self.at(TokenKind::Slash);
            self.advance();

            let name = self.parse_ident()?;
            let mut exp = 1i32;
            if self.at(TokenKind::Caret) {
                self.advance();
                let neg = if self.at(TokenKind::Minus) {
                    self.advance();
                    true
                } else {
                    false
                };
                if self.at(TokenKind::IntLit) {
                    exp = self.advance().text.parse().unwrap_or(1);
                    if neg {
                        exp = -exp;
                    }
                }
            }
            if is_div {
                exp = -exp;
            }
            base_units.push((name, exp));
        }

        Ok(UnitExpr { base_units })
    }

    /// Parse Tensor[f32, (batch, channels, height, width)]
    fn parse_tensor_type(&mut self) -> Result<TypeExpr> {
        self.expect(TokenKind::Tensor)?;
        self.expect(TokenKind::LBracket)?;

        let element_type = Box::new(self.parse_type()?);
        self.expect(TokenKind::Comma)?;

        // Parse shape tuple
        self.expect(TokenKind::LParen)?;
        let mut shape = Vec::new();
        while !self.at(TokenKind::RParen) {
            let dim = if self.at(TokenKind::Ident) {
                TensorDim::Named(self.advance().text.clone())
            } else if self.at(TokenKind::IntLit) {
                let size: usize = self.advance().text.parse().unwrap_or(0);
                TensorDim::Fixed(size)
            } else if self.at(TokenKind::Underscore) {
                self.advance();
                TensorDim::Dynamic
            } else {
                TensorDim::Expr(Box::new(self.parse_expr()?))
            };
            shape.push(dim);

            if !self.at(TokenKind::RParen) {
                self.expect(TokenKind::Comma)?;
            }
        }
        self.expect(TokenKind::RParen)?;

        self.expect(TokenKind::RBracket)?;

        Ok(TypeExpr::Tensor {
            element_type,
            shape,
        })
    }

    /// Parse tile type: tile<f16, 16, 16> or tile<bf16, 32, 32, "col_major">
    fn parse_tile_type(&mut self) -> Result<TypeExpr> {
        self.expect(TokenKind::Tile)?;
        self.expect(TokenKind::Lt)?;

        // Parse element type (must be scalar: f32, f16, bf16, f8, f4)
        let element_type = Box::new(self.parse_type()?);
        self.expect(TokenKind::Comma)?;

        // Parse tile_m (must be literal integer)
        let tile_m = match self.peek() {
            TokenKind::IntLit => {
                let val: u32 = self
                    .advance()
                    .text
                    .parse()
                    .map_err(|_| miette::miette!("Invalid tile dimension"))?;
                val
            }
            _ => return Err(miette::miette!("Expected integer literal for tile_m")),
        };
        self.expect(TokenKind::Comma)?;

        // Parse tile_n
        let tile_n = match self.peek() {
            TokenKind::IntLit => {
                let val: u32 = self
                    .advance()
                    .text
                    .parse()
                    .map_err(|_| miette::miette!("Invalid tile dimension"))?;
                val
            }
            _ => return Err(miette::miette!("Expected integer literal for tile_n")),
        };

        // Optional layout specifier
        let layout = if self.at(TokenKind::Comma) {
            self.advance();
            match self.peek() {
                TokenKind::StringLit => {
                    let layout_str = self.advance().text.clone();
                    Some(layout_str)
                }
                _ => return Err(miette::miette!("Expected string literal for layout")),
            }
        } else {
            None
        };

        self.expect(TokenKind::Gt)?;

        // Validate tile dimensions
        if !tile_m.is_power_of_two() || !tile_n.is_power_of_two() {
            return Err(miette::miette!(
                "Tile dimensions must be powers of 2, got {}x{}",
                tile_m,
                tile_n
            ));
        }
        if tile_m > 64 || tile_n > 64 {
            return Err(miette::miette!(
                "Tile dimensions must be ≤64, got {}x{}",
                tile_m,
                tile_n
            ));
        }

        Ok(TypeExpr::Tile {
            element_type,
            tile_m,
            tile_n,
            layout,
        })
    }

    /// Parse OntologyTerm[SNOMED:12345]
    fn parse_ontology_type(&mut self) -> Result<TypeExpr> {
        self.expect(TokenKind::OntologyTerm)?;
        self.expect(TokenKind::LBracket)?;

        let ontology = self.parse_ident()?;
        let term = if self.at(TokenKind::Colon) {
            self.advance();
            // Term can be an identifier or a number (like ICD10:E11 or SNOMED:12345)
            if self.at(TokenKind::Ident) {
                Some(self.advance().text.clone())
            } else if self.at(TokenKind::IntLit) {
                Some(self.advance().text.clone())
            } else {
                None
            }
        } else {
            None
        };

        self.expect(TokenKind::RBracket)?;

        Ok(TypeExpr::Ontology { ontology, term })
    }

    // ==================== EXPRESSIONS ====================

    pub fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_expr_with_precedence(0)
    }

    /// Parse an expression without allowing struct literals
    /// Used in contexts like match scrutinee where `x { ... }` is ambiguous
    fn parse_expr_no_struct(&mut self) -> Result<Expr> {
        let old = self.allow_struct_literals;
        self.allow_struct_literals = false;
        let result = self.parse_expr();
        self.allow_struct_literals = old;
        result
    }

    fn parse_expr_with_precedence(&mut self, min_prec: u8) -> Result<Expr> {
        let mut left = self.parse_unary()?;

        loop {
            // Check for range operators (lowest precedence, 0)
            if min_prec == 0 && (self.at(TokenKind::DotDot) || self.at(TokenKind::DotDotEq)) {
                let inclusive = self.at(TokenKind::DotDotEq);
                self.advance();

                // Parse end expression if present (not at end of expression context)
                let end = if self.at_expr_start() {
                    Some(Box::new(self.parse_expr_with_precedence(1)?))
                } else {
                    None
                };

                left = Expr::Range {
                    id: self.next_id(),
                    start: Some(Box::new(left)),
                    end,
                    inclusive,
                };
                continue;
            }

            // Check for binary operators
            if let Some((op, prec, assoc)) = self.binary_op_info() {
                if prec < min_prec {
                    break;
                }

                self.advance();
                let next_min = if assoc == Assoc::Left { prec + 1 } else { prec };
                let right = self.parse_expr_with_precedence(next_min)?;

                left = Expr::Binary {
                    id: self.next_id(),
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }

        Ok(left)
    }

    /// Check if current token can start an expression
    fn at_expr_start(&self) -> bool {
        matches!(
            self.peek(),
            TokenKind::Ident
                | TokenKind::IntLit
                | TokenKind::FloatLit
                | TokenKind::StringLit
                | TokenKind::CharLit
                | TokenKind::True
                | TokenKind::False
                | TokenKind::LParen
                | TokenKind::LBracket
                | TokenKind::LBrace
                | TokenKind::Minus
                | TokenKind::Bang
                | TokenKind::Amp
                | TokenKind::Star
                | TokenKind::If
                | TokenKind::Match
                | TokenKind::Loop
                | TokenKind::While
                | TokenKind::For
                | TokenKind::Return
                | TokenKind::Break
                | TokenKind::Continue
        )
    }

    fn binary_op_info(&self) -> Option<(BinaryOp, u8, Assoc)> {
        let (op, prec, assoc) = match self.peek() {
            TokenKind::PipePipe => (BinaryOp::Or, 1, Assoc::Left),
            TokenKind::AmpAmp => (BinaryOp::And, 2, Assoc::Left),
            TokenKind::EqEq => (BinaryOp::Eq, 3, Assoc::Left),
            TokenKind::Ne => (BinaryOp::Ne, 3, Assoc::Left),
            TokenKind::Lt => (BinaryOp::Lt, 4, Assoc::Left),
            TokenKind::Le => (BinaryOp::Le, 4, Assoc::Left),
            TokenKind::Gt => (BinaryOp::Gt, 4, Assoc::Left),
            TokenKind::Ge => (BinaryOp::Ge, 4, Assoc::Left),
            TokenKind::Pipe => (BinaryOp::BitOr, 5, Assoc::Left),
            TokenKind::Caret => (BinaryOp::BitXor, 6, Assoc::Left),
            TokenKind::Amp => (BinaryOp::BitAnd, 7, Assoc::Left),
            TokenKind::Shl => (BinaryOp::Shl, 8, Assoc::Left),
            TokenKind::Shr => (BinaryOp::Shr, 8, Assoc::Left),
            TokenKind::Plus => (BinaryOp::Add, 9, Assoc::Left),
            TokenKind::Minus => (BinaryOp::Sub, 9, Assoc::Left),
            TokenKind::PlusMinus => (BinaryOp::PlusMinus, 9, Assoc::Left),
            TokenKind::PlusPlus => (BinaryOp::Concat, 9, Assoc::Left),
            TokenKind::Star => (BinaryOp::Mul, 10, Assoc::Left),
            TokenKind::Slash => (BinaryOp::Div, 10, Assoc::Left),
            TokenKind::Percent => (BinaryOp::Rem, 10, Assoc::Left),
            _ => return None,
        };
        Some((op, prec, assoc))
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        match self.peek() {
            TokenKind::Minus => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Unary {
                    id: self.next_id(),
                    op: UnaryOp::Neg,
                    expr: Box::new(expr),
                })
            }
            TokenKind::Bang => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Unary {
                    id: self.next_id(),
                    op: UnaryOp::Not,
                    expr: Box::new(expr),
                })
            }
            TokenKind::Amp => {
                self.advance();
                let is_mut = if self.at(TokenKind::Mut) {
                    self.advance();
                    true
                } else {
                    false
                };
                let expr = self.parse_unary()?;
                Ok(Expr::Unary {
                    id: self.next_id(),
                    op: if is_mut {
                        UnaryOp::RefMut
                    } else {
                        UnaryOp::Ref
                    },
                    expr: Box::new(expr),
                })
            }
            TokenKind::Star => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Unary {
                    id: self.next_id(),
                    op: UnaryOp::Deref,
                    expr: Box::new(expr),
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.peek() {
                TokenKind::LParen => {
                    // Don't parse as call if there's a newline before the '('
                    // This prevents `let x = 3\n(a, b)` from being parsed as `let x = 3(a, b)`
                    if self.had_newline_before_current() {
                        break;
                    }
                    let lparen_start = self.current().span.start;
                    self.advance();
                    let mut args = Vec::new();
                    while !self.at(TokenKind::RParen) {
                        args.push(self.parse_expr_with_span()?);
                        if !self.at(TokenKind::RParen) {
                            self.expect(TokenKind::Comma)?;
                        }
                    }
                    let rparen_end = self.current().span.end;
                    self.expect(TokenKind::RParen)?;
                    let id = self.next_id();
                    self.record_span(id, Span::new(lparen_start, rparen_end));
                    expr = Expr::Call {
                        id,
                        callee: Box::new(expr),
                        args,
                    };
                }
                TokenKind::LBracket => {
                    self.advance();

                    // Check for slice syntax: [..], [..end], [start..], [start..end]
                    let index = if self.at(TokenKind::DotDot) || self.at(TokenKind::DotDotEq) {
                        // [..] or [..end] - range without start
                        let inclusive = self.at(TokenKind::DotDotEq);
                        self.advance();
                        let end = if !self.at(TokenKind::RBracket) {
                            Some(Box::new(self.parse_expr()?))
                        } else {
                            None
                        };
                        Expr::Range {
                            id: self.next_id(),
                            start: None,
                            end,
                            inclusive,
                        }
                    } else {
                        // Regular index or [start..] or [start..end]
                        self.parse_expr()?
                    };

                    self.expect(TokenKind::RBracket)?;
                    expr = Expr::Index {
                        id: self.next_id(),
                        base: Box::new(expr),
                        index: Box::new(index),
                    };
                }
                TokenKind::Dot => {
                    self.advance();
                    if self.at(TokenKind::IntLit) {
                        // Tuple field access
                        let index: usize = self.advance().text.parse().unwrap_or(0);
                        expr = Expr::TupleField {
                            id: self.next_id(),
                            base: Box::new(expr),
                            index,
                        };
                    } else if self.at(TokenKind::Await) {
                        // Postfix await: expr.await
                        self.advance();
                        expr = Expr::Await {
                            id: self.next_id(),
                            expr: Box::new(expr),
                        };
                    } else {
                        let field = self.parse_ident()?;

                        // Check if this is a method call: expr.method(args)
                        if self.at(TokenKind::LParen) {
                            self.advance();
                            let mut args = Vec::new();
                            while !self.at(TokenKind::RParen) {
                                args.push(self.parse_expr_with_span()?);
                                if !self.at(TokenKind::RParen) {
                                    self.expect(TokenKind::Comma)?;
                                }
                            }
                            self.expect(TokenKind::RParen)?;
                            expr = Expr::MethodCall {
                                id: self.next_id(),
                                receiver: Box::new(expr),
                                method: field,
                                args,
                            };
                        } else {
                            expr = Expr::Field {
                                id: self.next_id(),
                                base: Box::new(expr),
                                field,
                            };
                        }
                    }
                }
                TokenKind::Question => {
                    self.advance();
                    expr = Expr::Try {
                        id: self.next_id(),
                        expr: Box::new(expr),
                    };
                }
                TokenKind::As => {
                    self.advance();
                    let ty = self.parse_type()?;
                    expr = Expr::Cast {
                        id: self.next_id(),
                        expr: Box::new(expr),
                        ty,
                    };
                }
                TokenKind::At => {
                    // Type ascription with unit: 84.0@L_per_h means the value has that unit type
                    self.advance();
                    let ty = self.parse_type()?;
                    expr = Expr::Cast {
                        id: self.next_id(),
                        expr: Box::new(expr),
                        ty,
                    };
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.peek() {
            // Literals
            TokenKind::IntLit => {
                let text = self.advance().text.clone();
                let value: i64 = text.replace('_', "").parse().unwrap_or(0);
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Int(value),
                })
            }
            TokenKind::BinLit => {
                let text = self.advance().text.clone();
                // Remove 0b prefix and underscores
                let clean = text.trim_start_matches("0b").replace('_', "");
                let value = i64::from_str_radix(&clean, 2).unwrap_or(0);
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Int(value),
                })
            }
            TokenKind::OctLit => {
                let text = self.advance().text.clone();
                // Remove 0o prefix and underscores
                let clean = text.trim_start_matches("0o").replace('_', "");
                let value = i64::from_str_radix(&clean, 8).unwrap_or(0);
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Int(value),
                })
            }
            TokenKind::HexLit => {
                let text = self.advance().text.clone();
                // Remove 0x prefix and underscores
                let clean = text
                    .trim_start_matches("0x")
                    .trim_start_matches("0X")
                    .replace('_', "");
                let value = i64::from_str_radix(&clean, 16).unwrap_or(0);
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Int(value),
                })
            }
            TokenKind::FloatLit => {
                let text = self.advance().text.clone();
                let value: f64 = text.replace('_', "").parse().unwrap_or(0.0);
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Float(value),
                })
            }
            // Unit literals: 500_mg, 10.5_mL
            TokenKind::IntUnitLit => {
                let text = self.advance().text.clone();
                // Split at the underscore separating number from unit
                // Format: 123_unit or 123_456_unit (underscores in number)
                // Find the last underscore followed by a letter
                let (num_part, unit_part) = split_unit_literal(&text);
                let value: i64 = num_part.replace('_', "").parse().unwrap_or(0);
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::IntUnit(value, unit_part.to_string()),
                })
            }
            TokenKind::FloatUnitLit => {
                let text = self.advance().text.clone();
                // Split at the underscore separating number from unit
                let (num_part, unit_part) = split_unit_literal(&text);
                let value: f64 = num_part.replace('_', "").parse().unwrap_or(0.0);
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::FloatUnit(value, unit_part.to_string()),
                })
            }
            TokenKind::StringLit => {
                let span = self.current().span;
                let text = self.advance().text.clone();
                // Remove quotes
                let value = text[1..text.len() - 1].to_string();
                let id = self.next_id();
                self.record_span(id, span);
                Ok(Expr::Literal {
                    id,
                    value: Literal::String(value),
                })
            }
            TokenKind::CharLit => {
                let text = self.advance().text.clone();
                let value = text.chars().nth(1).unwrap_or('\0');
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Char(value),
                })
            }
            TokenKind::True => {
                self.advance();
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Bool(true),
                })
            }
            TokenKind::False => {
                self.advance();
                Ok(Expr::Literal {
                    id: self.next_id(),
                    value: Literal::Bool(false),
                })
            }

            // Linear algebra constructors: vec2(x, y), vec3(x, y, z), etc.
            // Autodiff constructors: dual(value, derivative)
            TokenKind::Vec2
            | TokenKind::Vec3
            | TokenKind::Vec4
            | TokenKind::Mat2
            | TokenKind::Mat3
            | TokenKind::Mat4
            | TokenKind::Quat
            | TokenKind::Dual => {
                let type_name = self.advance().text.clone();
                self.expect(TokenKind::LParen)?;
                let mut args = Vec::new();
                if !self.at(TokenKind::RParen) {
                    args.push(self.parse_expr()?);
                    while self.at(TokenKind::Comma) {
                        self.advance();
                        if self.at(TokenKind::RParen) {
                            break;
                        }
                        args.push(self.parse_expr()?);
                    }
                }
                self.expect(TokenKind::RParen)?;
                Ok(Expr::Call {
                    id: self.next_id(),
                    callee: Box::new(Expr::Path {
                        id: self.next_id(),
                        path: Path::simple(&type_name),
                    }),
                    args,
                })
            }

            // Identifiers and paths (including contextual keywords used as identifiers)
            // Note: contextual keywords with special syntax (Sample, Query, Observe, etc.)
            // are handled AFTER this case when followed by their special syntax patterns
            _ if self.at(TokenKind::Ident)
                || self.at(TokenKind::SelfLower)
                || (self.is_contextual_keyword() && !self.is_dsl_keyword_with_parens()) =>
            {
                // Check for macro invocation (identifier followed by !)
                if self.peek_n(1) == TokenKind::Bang {
                    let macro_inv = self.parse_macro_invocation()?;
                    return Ok(Expr::MacroInvocation(macro_inv));
                }

                // Check for ontology term literal: prefix:term (e.g., drugbank:DB00945)
                if self.peek_n(1) == TokenKind::Colon
                    && matches!(self.peek_n(2), TokenKind::Ident | TokenKind::IntLit)
                {
                    let prefix = self.parse_ident()?;
                    self.expect(TokenKind::Colon)?;
                    let term = if self.at(TokenKind::Ident) {
                        self.advance().text.clone()
                    } else {
                        self.advance().text.clone() // IntLit
                    };
                    return Ok(Expr::OntologyTerm {
                        id: self.next_id(),
                        ontology: prefix,
                        term,
                    });
                }

                let path = self.parse_path()?;

                // Check for struct literal (only if allowed in this context)
                if self.allow_struct_literals
                    && self.at(TokenKind::LBrace)
                    && !path.segments.is_empty()
                {
                    return self.parse_struct_literal(path);
                }

                Ok(Expr::Path {
                    id: self.next_id(),
                    path,
                })
            }

            // Grouped expression or tuple
            TokenKind::LParen => {
                self.advance();
                if self.at(TokenKind::RParen) {
                    self.advance();
                    return Ok(Expr::Literal {
                        id: self.next_id(),
                        value: Literal::Unit,
                    });
                }

                let expr = self.parse_expr()?;

                if self.at(TokenKind::Comma) {
                    // Tuple
                    let mut elements = vec![expr];
                    while self.at(TokenKind::Comma) {
                        self.advance();
                        if self.at(TokenKind::RParen) {
                            break;
                        }
                        elements.push(self.parse_expr()?);
                    }
                    self.expect(TokenKind::RParen)?;
                    Ok(Expr::Tuple {
                        id: self.next_id(),
                        elements,
                    })
                } else {
                    self.expect(TokenKind::RParen)?;
                    Ok(expr)
                }
            }

            // Array literal: [a, b, c] or repeat syntax [value; count]
            TokenKind::LBracket => {
                self.advance();
                let mut elements = Vec::new();

                // Check for empty array
                if self.at(TokenKind::RBracket) {
                    self.advance();
                    return Ok(Expr::Array {
                        id: self.next_id(),
                        elements,
                    });
                }

                // Parse first element
                let first = self.parse_expr()?;

                // Check for repeat syntax: [value; count]
                if self.at(TokenKind::Semi) {
                    self.advance();
                    let count_expr = self.parse_expr()?;
                    self.expect(TokenKind::RBracket)?;

                    // For now, expand into repeated elements if count is a literal
                    // In a full implementation, this would be a separate AST node
                    if let Expr::Literal {
                        value: Literal::Int(count),
                        ..
                    } = &count_expr
                    {
                        let count = *count as usize;
                        elements.push(first.clone());
                        for _ in 1..count {
                            elements.push(first.clone());
                        }
                    } else {
                        // If count is not a literal, just use first element
                        // TODO: Add proper ArrayRepeat AST node
                        elements.push(first);
                    }
                    return Ok(Expr::Array {
                        id: self.next_id(),
                        elements,
                    });
                }

                // Regular array: [a, b, c]
                elements.push(first);
                while self.at(TokenKind::Comma) {
                    self.advance();
                    if self.at(TokenKind::RBracket) {
                        break; // trailing comma
                    }
                    elements.push(self.parse_expr()?);
                }
                self.expect(TokenKind::RBracket)?;
                Ok(Expr::Array {
                    id: self.next_id(),
                    elements,
                })
            }

            // Block expression
            TokenKind::LBrace => {
                let block = self.parse_block()?;
                Ok(Expr::Block {
                    id: self.next_id(),
                    block,
                })
            }

            // If expression
            TokenKind::If => self.parse_if(),

            // Match expression
            TokenKind::Match => self.parse_match(),

            // Loop expressions
            TokenKind::Loop => self.parse_loop(),
            TokenKind::While => self.parse_while(),
            TokenKind::For => self.parse_for(),

            // Return
            TokenKind::Return => {
                self.advance();
                let value = if self.at_any(&[TokenKind::RBrace, TokenKind::Semi, TokenKind::Eof]) {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                Ok(Expr::Return {
                    id: self.next_id(),
                    value,
                })
            }

            // Break
            TokenKind::Break => {
                self.advance();
                let value = if self.at_any(&[TokenKind::RBrace, TokenKind::Semi, TokenKind::Eof]) {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                Ok(Expr::Break {
                    id: self.next_id(),
                    value,
                })
            }

            // Continue
            TokenKind::Continue => {
                self.advance();
                Ok(Expr::Continue { id: self.next_id() })
            }

            // Closure
            TokenKind::Pipe => self.parse_closure(),

            // Effect operations
            TokenKind::Perform => {
                self.advance();
                let effect = self.parse_path()?;
                self.expect(TokenKind::Dot)?;
                let op = self.parse_ident()?;
                let args = if self.at(TokenKind::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    while !self.at(TokenKind::RParen) {
                        args.push(self.parse_expr()?);
                        if !self.at(TokenKind::RParen) {
                            self.expect(TokenKind::Comma)?;
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    args
                } else {
                    Vec::new()
                };
                Ok(Expr::Perform {
                    id: self.next_id(),
                    effect,
                    op,
                    args,
                })
            }

            TokenKind::Handle => {
                self.advance();
                let expr = Box::new(self.parse_expr()?);
                self.expect(TokenKind::With)?;
                let handler = self.parse_path()?;
                Ok(Expr::Handle {
                    id: self.next_id(),
                    expr,
                    handler,
                })
            }

            // Probabilistic operations
            TokenKind::Sample => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let dist = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                Ok(Expr::Sample {
                    id: self.next_id(),
                    distribution: Box::new(dist),
                })
            }

            // Async block: async { ... }
            TokenKind::Async => {
                self.advance();
                if self.at(TokenKind::Pipe) {
                    // Async closure: async |x| { ... }
                    self.parse_async_closure()
                } else if self.at(TokenKind::LBrace) {
                    // Async block: async { ... }
                    let block = self.parse_block()?;
                    Ok(Expr::AsyncBlock {
                        id: self.next_id(),
                        block,
                    })
                } else {
                    Err(miette::miette!(
                        "Expected '{{' or '|' after 'async', found {:?}",
                        self.peek()
                    ))
                }
            }

            // Spawn: spawn { ... } or spawn expr
            TokenKind::Spawn => {
                self.advance();
                let expr = if self.at(TokenKind::LBrace) {
                    let block = self.parse_block()?;
                    Expr::AsyncBlock {
                        id: self.next_id(),
                        block,
                    }
                } else {
                    self.parse_expr()?
                };
                Ok(Expr::Spawn {
                    id: self.next_id(),
                    expr: Box::new(expr),
                })
            }

            // Await keyword (standalone): await expr
            TokenKind::Await => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::Await {
                    id: self.next_id(),
                    expr: Box::new(expr),
                })
            }

            // ==================== DEMETRIOS EPISTEMIC EXPRESSIONS ====================

            // do(X = 1) - Pearl's causal intervention
            TokenKind::Do => self.parse_do_expr(),

            // counterfactual { factual; do(X=1); outcome }
            TokenKind::Counterfactual => self.parse_counterfactual_expr(),

            // query P(Y | X, do(Z))
            TokenKind::Query => self.parse_query_expr(),

            // observe(data ~ distribution)
            TokenKind::Observe => self.parse_observe_expr(),

            // Knowledge type constructor: Knowledge { value, epsilon, validity, provenance }
            TokenKind::Knowledge => self.parse_knowledge_expr(),

            // Handle keywords followed by ! as macro invocations (e.g., assert!(x))
            _ if self.peek().is_keyword() && self.peek_n(1) == TokenKind::Bang => {
                let macro_inv = self.parse_macro_invocation()?;
                Ok(Expr::MacroInvocation(macro_inv))
            }

            _ => Err(miette::miette!(
                "Unexpected token {:?} in expression",
                self.peek()
            )),
        }
    }

    // ==================== DEMETRIOS EPISTEMIC EXPRESSION PARSING ====================

    /// Parse do(X = value, Y = value, ...) - causal intervention expression
    fn parse_do_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Do)?;
        self.expect(TokenKind::LParen)?;

        let mut interventions = Vec::new();
        while !self.at(TokenKind::RParen) {
            let var_name = self.parse_ident()?;
            self.expect(TokenKind::Eq)?;
            let value = Box::new(self.parse_expr()?);
            interventions.push((var_name, value));

            if !self.at(TokenKind::RParen) {
                self.expect(TokenKind::Comma)?;
            }
        }
        self.expect(TokenKind::RParen)?;

        Ok(Expr::Do {
            id: self.next_id(),
            interventions,
        })
    }

    /// Parse counterfactual { factual; intervention; outcome }
    fn parse_counterfactual_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Counterfactual)?;
        self.expect(TokenKind::LBrace)?;

        // Parse factual observation
        let factual = Box::new(self.parse_expr()?);
        self.expect(TokenKind::Semi)?;

        // Parse intervention (typically a do expression)
        let intervention = Box::new(self.parse_expr()?);
        self.expect(TokenKind::Semi)?;

        // Parse outcome query
        let outcome = Box::new(self.parse_expr()?);

        // Optional trailing semicolon
        if self.at(TokenKind::Semi) {
            self.advance();
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Expr::Counterfactual {
            id: self.next_id(),
            factual,
            intervention,
            outcome,
        })
    }

    /// Parse query P(Y | X, do(Z)) - probabilistic query with conditioning and intervention
    fn parse_query_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Query)?;

        // Check for P( syntax for probability query
        let target = if self.at(TokenKind::Ident) && self.current().text == "P" {
            self.advance();
            self.expect(TokenKind::LParen)?;
            // Parse target as primary only (so | is not consumed as binary OR)
            let t = Box::new(self.parse_primary()?);

            let mut given = Vec::new();
            let mut interventions = Vec::new();

            // Parse conditioning: | X, Y, do(Z)
            if self.at(TokenKind::Pipe) {
                self.advance();
                while !self.at(TokenKind::RParen) {
                    // Check for do() intervention
                    if self.at(TokenKind::Do) {
                        self.advance();
                        self.expect(TokenKind::LParen)?;
                        while !self.at(TokenKind::RParen) {
                            let var_name = self.parse_ident()?;
                            self.expect(TokenKind::Eq)?;
                            let value = Box::new(self.parse_primary()?);
                            interventions.push((var_name, value));
                            if !self.at(TokenKind::RParen) {
                                self.expect(TokenKind::Comma)?;
                            }
                        }
                        self.expect(TokenKind::RParen)?;
                    } else {
                        // Parse given variable as primary only
                        given.push(self.parse_primary()?);
                    }

                    if !self.at(TokenKind::RParen) {
                        self.expect(TokenKind::Comma)?;
                    }
                }
            }
            self.expect(TokenKind::RParen)?;

            return Ok(Expr::Query {
                id: self.next_id(),
                target: t,
                given,
                interventions,
            });
        } else {
            // Simple query expression
            Box::new(self.parse_expr()?)
        };

        Ok(Expr::Query {
            id: self.next_id(),
            target,
            given: Vec::new(),
            interventions: Vec::new(),
        })
    }

    /// Parse observe(data ~ distribution) - probabilistic observation
    fn parse_observe_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Observe)?;
        self.expect(TokenKind::LParen)?;

        let data = Box::new(self.parse_expr()?);

        // Expect ~ for distribution relationship
        if self.at(TokenKind::Tilde) {
            self.advance();
        } else {
            // Allow comma as alternative syntax: observe(data, distribution)
            self.expect(TokenKind::Comma)?;
        }

        let distribution = Box::new(self.parse_expr()?);
        self.expect(TokenKind::RParen)?;

        Ok(Expr::Observe {
            id: self.next_id(),
            data,
            distribution,
        })
    }

    /// Parse Knowledge expression: Knowledge { value: x, epsilon: 0.05, ... }
    /// or Knowledge::new(value, epsilon, validity, provenance)
    fn parse_knowledge_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Knowledge)?;

        // Check for struct-like syntax: Knowledge { ... }
        if self.at(TokenKind::LBrace) {
            self.advance();
            let mut value = None;
            let mut epsilon = None;
            let mut validity = None;
            let mut provenance = None;

            while !self.at(TokenKind::RBrace) {
                let field_name = self.parse_ident()?;
                self.expect(TokenKind::Colon)?;
                let field_value = self.parse_expr()?;

                match field_name.as_str() {
                    "value" => value = Some(Box::new(field_value)),
                    "epsilon" => epsilon = Some(Box::new(field_value)),
                    "validity" => validity = Some(Box::new(field_value)),
                    "provenance" => provenance = Some(Box::new(field_value)),
                    _ => return Err(miette::miette!("Unknown Knowledge field: {}", field_name)),
                }

                if !self.at(TokenKind::RBrace) {
                    self.expect(TokenKind::Comma)?;
                }
            }
            self.expect(TokenKind::RBrace)?;

            let value =
                value.ok_or_else(|| miette::miette!("Knowledge requires a 'value' field"))?;

            return Ok(Expr::KnowledgeExpr {
                id: self.next_id(),
                value,
                epsilon,
                validity,
                provenance,
            });
        }

        // Check for constructor syntax: Knowledge::new(...)
        if self.at(TokenKind::ColonColon) {
            self.advance();
            let method = self.parse_ident()?;
            if method != "new" {
                return Err(miette::miette!("Expected 'new' after Knowledge::"));
            }

            self.expect(TokenKind::LParen)?;
            let value = Box::new(self.parse_expr()?);

            let epsilon = if self.at(TokenKind::Comma) {
                self.advance();
                if !self.at(TokenKind::RParen) {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                }
            } else {
                None
            };

            let validity = if self.at(TokenKind::Comma) {
                self.advance();
                if !self.at(TokenKind::RParen) {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                }
            } else {
                None
            };

            let provenance = if self.at(TokenKind::Comma) {
                self.advance();
                if !self.at(TokenKind::RParen) {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                }
            } else {
                None
            };

            self.expect(TokenKind::RParen)?;

            return Ok(Expr::KnowledgeExpr {
                id: self.next_id(),
                value,
                epsilon,
                validity,
                provenance,
            });
        }

        Err(miette::miette!(
            "Expected '{{' or '::' after Knowledge, found {:?}",
            self.peek()
        ))
    }

    fn parse_struct_literal(&mut self, path: Path) -> Result<Expr> {
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();

        while !self.at(TokenKind::RBrace) {
            let name = self.parse_ident()?;
            let value = if self.at(TokenKind::Colon) {
                self.advance();
                self.parse_expr()?
            } else {
                // Shorthand: name without : means name: name
                Expr::Path {
                    id: self.next_id(),
                    path: Path::simple(&name),
                }
            };
            fields.push((name, value));

            if !self.at(TokenKind::RBrace) {
                self.expect(TokenKind::Comma)?;
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Expr::StructLit {
            id: self.next_id(),
            path,
            fields,
        })
    }

    fn parse_if(&mut self) -> Result<Expr> {
        self.expect(TokenKind::If)?;

        // Check for `if let` pattern matching
        if self.at(TokenKind::Let) {
            return self.parse_if_let();
        }

        // Use parse_expr_no_struct to avoid ambiguity with `if x { ... }`
        // being parsed as struct literal `x { ... }`
        let condition = Box::new(self.parse_expr_no_struct()?);
        let then_branch = self.parse_block()?;
        let else_branch = if self.at(TokenKind::Else) {
            self.advance();
            if self.at(TokenKind::If) {
                Some(Box::new(self.parse_if()?))
            } else {
                Some(Box::new(Expr::Block {
                    id: self.next_id(),
                    block: self.parse_block()?,
                }))
            }
        } else {
            None
        };

        Ok(Expr::If {
            id: self.next_id(),
            condition,
            then_branch,
            else_branch,
        })
    }

    /// Parse `if let PATTERN = EXPR { THEN } else { ELSE }`
    fn parse_if_let(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Let)?;
        let pattern = self.parse_pattern()?;
        self.expect(TokenKind::Eq)?;
        let scrutinee = Box::new(self.parse_expr_no_struct()?);
        let then_branch = self.parse_block()?;

        let else_branch = if self.at(TokenKind::Else) {
            self.advance();
            if self.at(TokenKind::If) {
                Some(Box::new(self.parse_if()?))
            } else {
                Some(Box::new(Expr::Block {
                    id: self.next_id(),
                    block: self.parse_block()?,
                }))
            }
        } else {
            None
        };

        // Convert if let to a Match expression with a single arm
        // This is a common desugaring approach
        let match_arm = MatchArm {
            pattern,
            guard: None,
            body: Expr::Block {
                id: self.next_id(),
                block: then_branch,
            },
        };

        // Create a wildcard arm for the else branch
        let else_arm = MatchArm {
            pattern: Pattern::Wildcard,
            guard: None,
            body: else_branch.map(|e| *e).unwrap_or(Expr::Literal {
                id: self.next_id(),
                value: Literal::Unit,
            }),
        };

        Ok(Expr::Match {
            id: self.next_id(),
            scrutinee,
            arms: vec![match_arm, else_arm],
        })
    }

    fn parse_match(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Match)?;
        // Use parse_expr_no_struct to avoid ambiguity with `match x { ... }`
        // being parsed as struct literal `x { ... }`
        let scrutinee = Box::new(self.parse_expr_no_struct()?);
        self.expect(TokenKind::LBrace)?;

        let mut arms = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let pattern = self.parse_pattern()?;
            let guard = if self.at(TokenKind::If) {
                self.advance();
                Some(Box::new(self.parse_expr()?))
            } else {
                None
            };
            self.expect(TokenKind::FatArrow)?;
            let body = self.parse_expr()?;
            if self.at(TokenKind::Comma) {
                self.advance();
            }
            arms.push(MatchArm {
                pattern,
                guard,
                body,
            });
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Expr::Match {
            id: self.next_id(),
            scrutinee,
            arms,
        })
    }

    fn parse_loop(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Loop)?;
        let body = self.parse_block()?;
        Ok(Expr::Loop {
            id: self.next_id(),
            body,
        })
    }

    fn parse_while(&mut self) -> Result<Expr> {
        self.expect(TokenKind::While)?;
        // Use parse_expr_no_struct to avoid ambiguity with `while x { ... }`
        // being parsed as struct literal `x { ... }`
        let condition = Box::new(self.parse_expr_no_struct()?);
        let body = self.parse_block()?;
        Ok(Expr::While {
            id: self.next_id(),
            condition,
            body,
        })
    }

    fn parse_for(&mut self) -> Result<Expr> {
        self.expect(TokenKind::For)?;
        let pattern = self.parse_pattern()?;
        self.expect(TokenKind::In)?;
        // Use parse_expr_no_struct to avoid ambiguity with `for x in items { ... }`
        // being parsed as struct literal `items { ... }`
        let iter = Box::new(self.parse_expr_no_struct()?);
        let body = self.parse_block()?;
        Ok(Expr::For {
            id: self.next_id(),
            pattern,
            iter,
            body,
        })
    }

    fn parse_closure(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Pipe)?;
        let mut params = Vec::new();
        while !self.at(TokenKind::Pipe) {
            let name = self.parse_ident()?;
            let ty = if self.at(TokenKind::Colon) {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };
            params.push((name, ty));
            if !self.at(TokenKind::Pipe) {
                self.expect(TokenKind::Comma)?;
            }
        }
        self.expect(TokenKind::Pipe)?;

        let return_type = if self.at(TokenKind::Arrow) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = if self.at(TokenKind::LBrace) {
            let block = self.parse_block()?;
            Box::new(Expr::Block {
                id: self.next_id(),
                block,
            })
        } else {
            Box::new(self.parse_expr()?)
        };

        Ok(Expr::Closure {
            id: self.next_id(),
            params,
            return_type,
            body,
        })
    }

    fn parse_async_closure(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Pipe)?;
        let mut params = Vec::new();
        while !self.at(TokenKind::Pipe) {
            let name = self.parse_ident()?;
            let ty = if self.at(TokenKind::Colon) {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };
            params.push((name, ty));
            if !self.at(TokenKind::Pipe) {
                self.expect(TokenKind::Comma)?;
            }
        }
        self.expect(TokenKind::Pipe)?;

        let return_type = if self.at(TokenKind::Arrow) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = if self.at(TokenKind::LBrace) {
            let block = self.parse_block()?;
            Box::new(Expr::Block {
                id: self.next_id(),
                block,
            })
        } else {
            Box::new(self.parse_expr()?)
        };

        Ok(Expr::AsyncClosure {
            id: self.next_id(),
            params,
            return_type,
            body,
        })
    }

    // ==================== STATEMENTS ====================

    fn parse_block(&mut self) -> Result<Block> {
        self.expect(TokenKind::LBrace)?;
        let mut stmts = Vec::new();

        while !self.at(TokenKind::RBrace) {
            stmts.push(self.parse_stmt()?);
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Block { stmts })
    }

    pub fn parse_stmt(&mut self) -> Result<Stmt> {
        // Skip doc comments
        while self.at(TokenKind::DocCommentOuter) || self.at(TokenKind::DocCommentInner) {
            self.advance();
        }

        match self.peek() {
            TokenKind::Let => self.parse_let_stmt(),
            TokenKind::Var => self.parse_var_stmt(),
            TokenKind::Semi => {
                self.advance();
                Ok(Stmt::Empty)
            }
            _ => {
                let expr = self.parse_expr()?;

                // Check for assignment
                if let Some(op) = self.assignment_op() {
                    self.advance();
                    let rhs = self.parse_expr()?;
                    if self.at(TokenKind::Semi) {
                        self.advance();
                    }
                    return Ok(Stmt::Assign {
                        target: expr,
                        op,
                        value: rhs,
                    });
                }

                let has_semi = if self.at(TokenKind::Semi) {
                    self.advance();
                    true
                } else {
                    false
                };

                Ok(Stmt::Expr { expr, has_semi })
            }
        }
    }

    fn assignment_op(&self) -> Option<AssignOp> {
        match self.peek() {
            TokenKind::Eq => Some(AssignOp::Assign),
            TokenKind::PlusEq => Some(AssignOp::AddAssign),
            TokenKind::MinusEq => Some(AssignOp::SubAssign),
            TokenKind::StarEq => Some(AssignOp::MulAssign),
            TokenKind::SlashEq => Some(AssignOp::DivAssign),
            TokenKind::PercentEq => Some(AssignOp::RemAssign),
            TokenKind::AmpEq => Some(AssignOp::BitAndAssign),
            TokenKind::PipeEq => Some(AssignOp::BitOrAssign),
            TokenKind::CaretEq => Some(AssignOp::BitXorAssign),
            TokenKind::ShlEq => Some(AssignOp::ShlAssign),
            TokenKind::ShrEq => Some(AssignOp::ShrAssign),
            _ => None,
        }
    }

    fn parse_let_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenKind::Let)?;
        let is_mut = if self.at(TokenKind::Mut) {
            self.advance();
            true
        } else {
            false
        };

        let pattern = self.parse_pattern()?;
        let ty = if self.at(TokenKind::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        let value = if self.at(TokenKind::Eq) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        if self.at(TokenKind::Semi) {
            self.advance();
        }

        Ok(Stmt::Let {
            is_mut,
            pattern,
            ty,
            value,
        })
    }

    /// Parse a var statement (mutable binding): var x: T = value;
    /// Equivalent to `let mut x: T = value;`
    fn parse_var_stmt(&mut self) -> Result<Stmt> {
        self.expect(TokenKind::Var)?;

        let pattern = self.parse_pattern()?;
        let ty = if self.at(TokenKind::Colon) {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        let value = if self.at(TokenKind::Eq) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        if self.at(TokenKind::Semi) {
            self.advance();
        }

        // var is syntactic sugar for let mut
        Ok(Stmt::Let {
            is_mut: true,
            pattern,
            ty,
            value,
        })
    }

    // ==================== PATTERNS ====================

    fn parse_pattern(&mut self) -> Result<Pattern> {
        match self.peek() {
            TokenKind::Underscore => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            TokenKind::IntLit => {
                let text = self.advance().text.clone();
                let value: i64 = text.replace('_', "").parse().unwrap_or(0);
                Ok(Pattern::Literal(Literal::Int(value)))
            }
            TokenKind::True => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(true)))
            }
            TokenKind::False => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(false)))
            }
            TokenKind::StringLit => {
                let text = self.advance().text.clone();
                let value = text[1..text.len() - 1].to_string();
                Ok(Pattern::Literal(Literal::String(value)))
            }
            TokenKind::LParen => {
                self.advance();
                if self.at(TokenKind::RParen) {
                    self.advance();
                    return Ok(Pattern::Literal(Literal::Unit));
                }
                let mut elements = vec![self.parse_pattern()?];
                while self.at(TokenKind::Comma) {
                    self.advance();
                    if self.at(TokenKind::RParen) {
                        break;
                    }
                    elements.push(self.parse_pattern()?);
                }
                self.expect(TokenKind::RParen)?;
                Ok(Pattern::Tuple(elements))
            }
            // Accept Ident, self, or contextual keywords as pattern bindings
            _ if self.at(TokenKind::Ident)
                || self.at(TokenKind::SelfLower)
                || self.is_contextual_keyword() =>
            {
                let path = self.parse_path()?;
                if self.at(TokenKind::LParen) {
                    // Enum variant with tuple data
                    self.advance();
                    let mut patterns = Vec::new();
                    while !self.at(TokenKind::RParen) {
                        patterns.push(self.parse_pattern()?);
                        if !self.at(TokenKind::RParen) {
                            self.expect(TokenKind::Comma)?;
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    Ok(Pattern::Enum {
                        path,
                        patterns: Some(patterns),
                    })
                } else if self.at(TokenKind::LBrace) {
                    // Struct pattern
                    self.advance();
                    let mut fields = Vec::new();
                    while !self.at(TokenKind::RBrace) {
                        let name = self.parse_ident()?;
                        let pattern = if self.at(TokenKind::Colon) {
                            self.advance();
                            self.parse_pattern()?
                        } else {
                            Pattern::Binding {
                                name: name.clone(),
                                mutable: false,
                            }
                        };
                        fields.push((name, pattern));
                        if !self.at(TokenKind::RBrace) {
                            self.expect(TokenKind::Comma)?;
                        }
                    }
                    self.expect(TokenKind::RBrace)?;
                    Ok(Pattern::Struct { path, fields })
                } else if path.segments.len() == 1 {
                    // Simple binding
                    Ok(Pattern::Binding {
                        name: path.segments.into_iter().next().unwrap(),
                        mutable: false,
                    })
                } else {
                    // Path pattern (unit variant)
                    Ok(Pattern::Enum {
                        path,
                        patterns: None,
                    })
                }
            }
            TokenKind::Mut => {
                self.advance();
                let name = self.parse_ident()?;
                Ok(Pattern::Binding {
                    name,
                    mutable: true,
                })
            }
            _ => Err(miette::miette!("Expected pattern, found {:?}", self.peek())),
        }
    }

    // ==================== HELPERS ====================

    fn parse_ident(&mut self) -> Result<String> {
        if self.at(TokenKind::Ident) {
            Ok(self.advance().text.clone())
        } else if self.at(TokenKind::SelfLower) {
            Ok(self.advance().text.clone())
        } else if self.is_contextual_keyword() {
            // Allow certain keywords to be used as identifiers in specific contexts
            // (like field names, variable names, etc.)
            Ok(self.advance().text.clone())
        } else {
            Err(miette::miette!(
                "Expected identifier, found {:?}",
                self.peek()
            ))
        }
    }

    /// Check if current token is a keyword that can be used as an identifier in certain contexts
    /// These are "soft keywords" that have special meaning in specific syntactic positions
    /// but can be used as variable/parameter names when the context is unambiguous.
    ///
    /// NOTE: Keywords with special expression syntax (Sample, Query, Observe, Infer, Do,
    /// Counterfactual) are NOT included here because they have unambiguous syntactic
    /// positions and shouldn't be used as variable names.
    fn is_contextual_keyword(&self) -> bool {
        matches!(
            self.peek(),
            // Effect system keywords (can be identifiers outside effect declarations)
            TokenKind::Effect
                | TokenKind::Handler
                | TokenKind::Handle
                // ODE/PDE DSL keywords (can be identifiers in normal expressions)
                | TokenKind::State
                | TokenKind::Nodes
                | TokenKind::Edges
                | TokenKind::Domain
                | TokenKind::Boundary
                | TokenKind::Initial
                | TokenKind::Params
                | TokenKind::Var // Can be used as identifier (e.g., "var" field name)
                // Ontology keywords
                | TokenKind::Align
                | TokenKind::Ontology
                | TokenKind::From
                | TokenKind::Distance
                | TokenKind::Threshold
                // Other contextual keywords
                | TokenKind::Type
                | TokenKind::Module
        )
    }

    /// Check if the current token is a DSL keyword that has special syntax with parentheses or braces.
    /// These should NOT be treated as identifiers when followed by their special syntax.
    fn is_dsl_keyword_with_parens(&self) -> bool {
        let next = self.peek_n(1);
        match self.peek() {
            // Keywords with (...) syntax
            TokenKind::Sample
            | TokenKind::Query
            | TokenKind::Observe
            | TokenKind::Infer
            | TokenKind::Do => next == TokenKind::LParen,
            // Keywords with {...} syntax
            TokenKind::Counterfactual => next == TokenKind::LBrace,
            _ => false,
        }
    }

    fn parse_path(&mut self) -> Result<Path> {
        let mut segments = vec![self.parse_ident()?];

        // Only use :: as path separator for module-qualified paths
        // Note: Previously accepted . for Darwin Atlas compatibility, but this broke
        // struct field access (s.field was parsed as path ["s", "field"] instead of
        // Expr::Field). Field access is now handled in parse_postfix via TokenKind::Dot.
        while self.at(TokenKind::ColonColon) {
            // Stop if next token after :: is { or * (for import syntax like `use foo::{a, b}` or `use foo::*`)
            let next = self.peek_n(1);
            if next == TokenKind::LBrace || next == TokenKind::Star {
                break;
            }
            self.advance();
            segments.push(self.parse_ident()?);
        }

        Ok(Path {
            segments,
            source_module: None,
            resolved_module: None,
        })
    }

    /// Parse a path in type context - accepts both :: and . as separators
    /// This enables Darwin Atlas compatibility where module.Type syntax is used
    /// in type positions (e.g., &operators.Sequence)
    fn parse_type_path(&mut self) -> Result<Path> {
        let mut segments = vec![self.parse_ident()?];

        // Accept both :: and . as path separators in type context
        // This is safe because in type context there's no ambiguity with field access
        while self.at(TokenKind::ColonColon) || self.at(TokenKind::Dot) {
            self.advance();
            segments.push(self.parse_ident()?);
        }

        Ok(Path {
            segments,
            source_module: None,
            resolved_module: None,
        })
    }

    // ==================== MACROS ====================

    /// Parse a macro invocation (e.g., vec![1, 2, 3] or assert!(x))
    fn parse_macro_invocation(&mut self) -> Result<MacroInvocation> {
        let start_span = self.span();
        // Macro name can be an identifier or a keyword (e.g., assert!, vec!)
        let name = if self.at(TokenKind::Ident) {
            self.parse_ident()?
        } else if self.peek().is_keyword() {
            let text = self.current().text.clone();
            self.advance();
            text
        } else {
            return Err(miette::miette!(
                "Expected macro name, found {:?}",
                self.peek()
            ));
        };
        self.expect(TokenKind::Bang)?;
        let args = self.parse_macro_args()?;
        let end_span = self.span();

        Ok(MacroInvocation {
            id: self.next_id(),
            name,
            args,
            span: start_span.merge(end_span),
        })
    }

    /// Parse macro arguments (token tree inside delimiters)
    fn parse_macro_args(&mut self) -> Result<Vec<crate::macro_system::token_tree::TokenTree>> {
        use crate::macro_system::token_tree::Delimiter;

        match self.peek() {
            TokenKind::LParen => self.parse_delimited_macro_args(Delimiter::Parenthesis),
            TokenKind::LBracket => self.parse_delimited_macro_args(Delimiter::Bracket),
            TokenKind::LBrace => self.parse_delimited_macro_args(Delimiter::Brace),
            _ => Err(miette::miette!(
                "Expected macro arguments (parentheses, brackets, or braces), found {:?}",
                self.peek()
            )),
        }
    }

    /// Parse delimited macro arguments
    fn parse_delimited_macro_args(
        &mut self,
        delim: crate::macro_system::token_tree::Delimiter,
    ) -> Result<Vec<crate::macro_system::token_tree::TokenTree>> {
        use crate::macro_system::token_tree::Delimiter;

        let open_delim = match delim {
            Delimiter::Parenthesis => TokenKind::LParen,
            Delimiter::Bracket => TokenKind::LBracket,
            Delimiter::Brace => TokenKind::LBrace,
            Delimiter::None => return Err(miette::miette!("Invalid delimiter")),
        };

        let close_delim = match delim {
            Delimiter::Parenthesis => TokenKind::RParen,
            Delimiter::Bracket => TokenKind::RBracket,
            Delimiter::Brace => TokenKind::RBrace,
            Delimiter::None => return Err(miette::miette!("Invalid delimiter")),
        };

        self.expect(open_delim)?;
        let mut args = Vec::new();

        while !self.at(close_delim) && !self.at(TokenKind::Eof) {
            args.push(self.parse_token_tree()?);
        }

        self.expect(close_delim)?;
        Ok(args)
    }

    /// Parse a single token tree (token or delimited sequence)
    fn parse_token_tree(&mut self) -> Result<crate::macro_system::token_tree::TokenTree> {
        use crate::macro_system::token_tree::{Delimiter, TokenTree, TokenWithCtx};

        match self.peek() {
            TokenKind::LParen => {
                let span = self.span();
                let args = self.parse_delimited_macro_args(Delimiter::Parenthesis)?;
                Ok(TokenTree::Delimited(Delimiter::Parenthesis, args, span))
            }
            TokenKind::LBracket => {
                let span = self.span();
                let args = self.parse_delimited_macro_args(Delimiter::Bracket)?;
                Ok(TokenTree::Delimited(Delimiter::Bracket, args, span))
            }
            TokenKind::LBrace => {
                let span = self.span();
                let args = self.parse_delimited_macro_args(Delimiter::Brace)?;
                Ok(TokenTree::Delimited(Delimiter::Brace, args, span))
            }
            _ => {
                let token = self.current().clone();
                self.advance();
                Ok(TokenTree::Token(TokenWithCtx::new(token)))
            }
        }
    }
}

/// Associativity for binary operators
#[derive(Clone, Copy, PartialEq, Eq)]
enum Assoc {
    Left,
    Right,
}

/// Split a unit literal into its numeric and unit parts.
/// For example: "500_mg" -> ("500", "mg"), "1_000_kg" -> ("1_000", "kg")
fn split_unit_literal(text: &str) -> (&str, &str) {
    // Find the last underscore that is followed by a letter (start of unit)
    // We scan backwards to handle cases like "1_000_kg" where the number has underscores
    let bytes = text.as_bytes();
    for i in (0..bytes.len()).rev() {
        if bytes[i] == b'_' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_alphabetic() {
            return (&text[..i], &text[i + 1..]);
        }
    }
    // Fallback: shouldn't happen with valid unit literals
    (text, "")
}
