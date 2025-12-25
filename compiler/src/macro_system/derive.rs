//! Derive macro framework

use super::proc_macro::*;
use super::token_tree::*;
use crate::lexer::TokenKind;

/// Parsed derive input
#[derive(Debug, Clone)]
pub struct DeriveInput {
    pub ident: String,
    pub generics: Generics,
    pub data: Data,
    pub attrs: Vec<Attribute>,
    pub tokens: TokenStream,
}

/// Generics information
#[derive(Debug, Clone, Default)]
pub struct Generics {
    pub params: Vec<GenericParam>,
    pub where_predicates: Vec<WherePredicate>,
}

/// A generic parameter
#[derive(Debug, Clone)]
pub enum GenericParam {
    Type {
        name: String,
        bounds: Vec<TypeBound>,
        default: Option<TokenStream>,
    },
    Lifetime {
        name: String,
        bounds: Vec<String>,
    },
    Const {
        name: String,
        ty: TokenStream,
        default: Option<TokenStream>,
    },
}

/// A type bound
#[derive(Debug, Clone)]
pub struct TypeBound {
    pub path: String,
    pub generics: Vec<TokenStream>,
}

/// A where clause predicate
#[derive(Debug, Clone)]
pub struct WherePredicate {
    pub bounded_ty: TokenStream,
    pub bounds: Vec<TypeBound>,
}

/// The data of a type (struct or enum)
#[derive(Debug, Clone)]
pub enum Data {
    Struct(DataStruct),
    Enum(DataEnum),
}

/// Struct data
#[derive(Debug, Clone)]
pub struct DataStruct {
    pub kind: StructKind,
    pub fields: Fields,
}

/// Struct kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructKind {
    Named,
    Tuple,
    Unit,
}

/// Enum data
#[derive(Debug, Clone)]
pub struct DataEnum {
    pub variants: Vec<Variant>,
}

/// An enum variant
#[derive(Debug, Clone)]
pub struct Variant {
    pub ident: String,
    pub fields: Fields,
    pub discriminant: Option<TokenStream>,
    pub attrs: Vec<Attribute>,
}

/// Fields of a struct or variant
#[derive(Debug, Clone)]
pub struct Fields {
    pub named: bool,
    pub fields: Vec<Field>,
}

impl Fields {
    pub fn empty() -> Self {
        Fields {
            named: false,
            fields: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Field> {
        self.fields.iter()
    }
}

/// A field
#[derive(Debug, Clone)]
pub struct Field {
    pub ident: Option<String>,
    pub ty: TokenStream,
    pub vis: Visibility,
    pub attrs: Vec<Attribute>,
}

/// Visibility
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
    Crate,
    Restricted,
}

/// An attribute
#[derive(Debug, Clone)]
pub struct Attribute {
    pub path: String,
    pub tokens: TokenStream,
    pub style: AttrStyle,
}

/// Attribute style
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttrStyle {
    Outer,
    Inner,
}

/// Parse a derive input from tokens
pub fn parse_derive_input(input: TokenStream) -> Result<DeriveInput, ProcMacroError> {
    let trees = input.into_trees();
    let mut parser = DeriveParser::new(trees);
    parser.parse()
}

struct DeriveParser {
    trees: Vec<TokenTree>,
    pos: usize,
}

impl DeriveParser {
    fn new(trees: Vec<TokenTree>) -> Self {
        DeriveParser { trees, pos: 0 }
    }

    fn parse(&mut self) -> Result<DeriveInput, ProcMacroError> {
        let attrs = self.parse_attributes()?;
        self.parse_visibility()?;
        let data_kind = self.parse_data()?;
        let ident = self.expect_ident()?;
        let generics = self.parse_generics()?;

        let data = match data_kind.as_str() {
            "struct" => Data::Struct(self.parse_struct_body()?),
            "enum" => Data::Enum(self.parse_enum_body()?),
            _ => {
                return Err(ProcMacroError::new(format!(
                    "expected struct or enum, found {}",
                    data_kind
                )));
            }
        };

        Ok(DeriveInput {
            ident,
            generics,
            data,
            attrs,
            tokens: TokenStream::from_iter(self.trees.clone()),
        })
    }

    fn parse_attributes(&mut self) -> Result<Vec<Attribute>, ProcMacroError> {
        let mut attrs = Vec::new();

        while self.check_token(TokenKind::Hash) {
            self.advance();

            let style = if self.check_token(TokenKind::Bang) {
                self.advance();
                AttrStyle::Inner
            } else {
                AttrStyle::Outer
            };

            match self.current() {
                Some(TokenTree::Delimited(Delimiter::Bracket, inner, _)) => {
                    let path = if let Some(TokenTree::Token(t)) = inner.first() {
                        t.token.text.clone()
                    } else {
                        String::new()
                    };

                    attrs.push(Attribute {
                        path,
                        tokens: TokenStream::from_iter(inner.iter().cloned()),
                        style,
                    });
                    self.advance();
                }
                _ => return Err(ProcMacroError::new("expected attribute brackets")),
            }
        }

        Ok(attrs)
    }

    fn parse_visibility(&mut self) -> Result<Visibility, ProcMacroError> {
        if self.check_ident("pub") {
            self.advance();
            Ok(Visibility::Public)
        } else {
            Ok(Visibility::Private)
        }
    }

    fn parse_data(&mut self) -> Result<String, ProcMacroError> {
        if self.check_ident("struct") {
            self.advance();
            Ok("struct".to_string())
        } else if self.check_ident("enum") {
            self.advance();
            Ok("enum".to_string())
        } else {
            Err(ProcMacroError::new("expected struct or enum"))
        }
    }

    fn parse_generics(&mut self) -> Result<Generics, ProcMacroError> {
        if !self.check_token(TokenKind::Lt) {
            return Ok(Generics::default());
        }

        self.advance();
        let mut params = Vec::new();

        while !self.check_token(TokenKind::Gt) {
            if self.check_token(TokenKind::Comma) {
                self.advance();
                continue;
            }

            let name = self.expect_ident()?;
            let bounds = if self.check_token(TokenKind::Colon) {
                self.advance();
                self.parse_bounds()?
            } else {
                Vec::new()
            };

            params.push(GenericParam::Type {
                name,
                bounds,
                default: None,
            });
        }

        self.advance();

        let where_predicates = if self.check_ident("where") {
            self.advance();
            Vec::new() // Simplified
        } else {
            Vec::new()
        };

        Ok(Generics {
            params,
            where_predicates,
        })
    }

    fn parse_bounds(&mut self) -> Result<Vec<TypeBound>, ProcMacroError> {
        let mut bounds = Vec::new();

        loop {
            let path = self.expect_ident()?;
            bounds.push(TypeBound {
                path,
                generics: Vec::new(),
            });

            if !self.check_token(TokenKind::Plus) {
                break;
            }
            self.advance();
        }

        Ok(bounds)
    }

    fn parse_struct_body(&mut self) -> Result<DataStruct, ProcMacroError> {
        match self.current().cloned() {
            Some(TokenTree::Delimited(Delimiter::Brace, inner, _)) => {
                self.advance();
                let fields = self.parse_named_fields(&inner)?;
                Ok(DataStruct {
                    kind: StructKind::Named,
                    fields,
                })
            }
            Some(TokenTree::Delimited(Delimiter::Parenthesis, inner, _)) => {
                self.advance();
                let fields = self.parse_tuple_fields(&inner)?;
                self.expect_token(TokenKind::Semi)?;
                Ok(DataStruct {
                    kind: StructKind::Tuple,
                    fields,
                })
            }
            Some(TokenTree::Token(t)) if t.token.kind == TokenKind::Semi => {
                self.advance();
                Ok(DataStruct {
                    kind: StructKind::Unit,
                    fields: Fields::empty(),
                })
            }
            _ => Err(ProcMacroError::new("expected struct body")),
        }
    }

    fn parse_named_fields(&self, inner: &[TokenTree]) -> Result<Fields, ProcMacroError> {
        let mut fields = Vec::new();
        let mut i = 0;

        while i < inner.len() {
            let vis = if matches!(inner.get(i), Some(TokenTree::Token(t)) if t.token.text == "pub")
            {
                i += 1;
                Visibility::Public
            } else {
                Visibility::Private
            };

            let ident = match inner.get(i) {
                Some(TokenTree::Token(t)) if t.token.kind == TokenKind::Ident => {
                    i += 1;
                    Some(t.token.text.clone())
                }
                _ => return Err(ProcMacroError::new("expected field name")),
            };

            match inner.get(i) {
                Some(TokenTree::Token(t)) if t.token.kind == TokenKind::Colon => {
                    i += 1;
                }
                _ => return Err(ProcMacroError::new("expected ':'")),
            }

            let mut ty_tokens = Vec::new();
            while i < inner.len() {
                match &inner[i] {
                    TokenTree::Token(t) if t.token.kind == TokenKind::Comma => {
                        i += 1;
                        break;
                    }
                    tree => {
                        ty_tokens.push(tree.clone());
                        i += 1;
                    }
                }
            }

            fields.push(Field {
                ident,
                ty: TokenStream::from_iter(ty_tokens),
                vis,
                attrs: Vec::new(),
            });
        }

        Ok(Fields {
            named: true,
            fields,
        })
    }

    fn parse_tuple_fields(&self, inner: &[TokenTree]) -> Result<Fields, ProcMacroError> {
        let mut fields = Vec::new();
        let mut i = 0;

        while i < inner.len() {
            let vis = if matches!(inner.get(i), Some(TokenTree::Token(t)) if t.token.text == "pub")
            {
                i += 1;
                Visibility::Public
            } else {
                Visibility::Private
            };

            let mut ty_tokens = Vec::new();
            while i < inner.len() {
                match &inner[i] {
                    TokenTree::Token(t) if t.token.kind == TokenKind::Comma => {
                        i += 1;
                        break;
                    }
                    tree => {
                        ty_tokens.push(tree.clone());
                        i += 1;
                    }
                }
            }

            if !ty_tokens.is_empty() {
                fields.push(Field {
                    ident: None,
                    ty: TokenStream::from_iter(ty_tokens),
                    vis,
                    attrs: Vec::new(),
                });
            }
        }

        Ok(Fields {
            named: false,
            fields,
        })
    }

    fn parse_enum_body(&mut self) -> Result<DataEnum, ProcMacroError> {
        match self.current().cloned() {
            Some(TokenTree::Delimited(Delimiter::Brace, inner, _)) => {
                self.advance();
                let variants = self.parse_variants(&inner)?;
                Ok(DataEnum { variants })
            }
            _ => Err(ProcMacroError::new("expected enum body")),
        }
    }

    fn parse_variants(&self, inner: &[TokenTree]) -> Result<Vec<Variant>, ProcMacroError> {
        let mut variants = Vec::new();
        let mut i = 0;

        while i < inner.len() {
            let ident = match inner.get(i) {
                Some(TokenTree::Token(t)) if t.token.kind == TokenKind::Ident => {
                    i += 1;
                    t.token.text.clone()
                }
                Some(TokenTree::Token(t)) if t.token.kind == TokenKind::Comma => {
                    i += 1;
                    continue;
                }
                _ => break,
            };

            let fields = match inner.get(i) {
                Some(TokenTree::Delimited(Delimiter::Brace, fields_inner, _)) => {
                    i += 1;
                    self.parse_named_fields(fields_inner)?
                }
                Some(TokenTree::Delimited(Delimiter::Parenthesis, fields_inner, _)) => {
                    i += 1;
                    self.parse_tuple_fields(fields_inner)?
                }
                _ => Fields::empty(),
            };

            let discriminant = if matches!(inner.get(i), Some(TokenTree::Token(t)) if t.token.kind == TokenKind::Eq)
            {
                i += 1;
                let mut disc_tokens = Vec::new();
                while i < inner.len() {
                    match &inner[i] {
                        TokenTree::Token(t) if t.token.kind == TokenKind::Comma => break,
                        tree => {
                            disc_tokens.push(tree.clone());
                            i += 1;
                        }
                    }
                }
                Some(TokenStream::from_iter(disc_tokens))
            } else {
                None
            };

            variants.push(Variant {
                ident,
                fields,
                discriminant,
                attrs: Vec::new(),
            });

            if matches!(inner.get(i), Some(TokenTree::Token(t)) if t.token.kind == TokenKind::Comma)
            {
                i += 1;
            }
        }

        Ok(variants)
    }

    fn current(&self) -> Option<&TokenTree> {
        self.trees.get(self.pos)
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn check_token(&self, kind: TokenKind) -> bool {
        matches!(self.current(), Some(TokenTree::Token(t)) if t.token.kind == kind)
    }

    fn check_ident(&self, name: &str) -> bool {
        matches!(self.current(), Some(TokenTree::Token(t))
            if t.token.kind == TokenKind::Ident && t.token.text == name)
    }

    fn expect_ident(&mut self) -> Result<String, ProcMacroError> {
        match self.current() {
            Some(TokenTree::Token(t)) if t.token.kind == TokenKind::Ident => {
                let name = t.token.text.clone();
                self.advance();
                Ok(name)
            }
            _ => Err(ProcMacroError::new("expected identifier")),
        }
    }

    fn expect_token(&mut self, kind: TokenKind) -> Result<(), ProcMacroError> {
        if self.check_token(kind) {
            self.advance();
            Ok(())
        } else {
            Err(ProcMacroError::new(format!("expected {:?}", kind)))
        }
    }
}
