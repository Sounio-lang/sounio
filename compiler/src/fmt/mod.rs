//! Code formatter for Sounio
//!
//! This module provides AST-based code formatting with:
//! - Configurable style rules
//! - Comment preservation
//! - Incremental formatting
//! - Diff mode

pub mod config;
pub mod ir;
pub mod printer;

pub use config::FormatConfig;
pub use ir::Doc;
pub use printer::Printer;

use crate::ast::{
    Ast, BinaryOp, Block, EffectDef, EnumDef, Expr, FnDef, HandlerDef, ImplDef, Item, Param,
    Pattern, Stmt, StructDef, TraitDef, TypeAliasDef, TypeExpr, UnaryOp, Visibility,
};
use crate::common::Span;

/// Code formatter
pub struct Formatter {
    /// Configuration
    config: FormatConfig,

    /// Source text (for preserving comments)
    source: String,

    /// Comments indexed by position
    comments: Vec<Comment>,
}

/// A comment in source
#[derive(Debug, Clone)]
pub struct Comment {
    pub kind: CommentKind,
    pub content: String,
    pub span: Span,
    pub is_doc: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentKind {
    /// // comment
    Line,
    /// /* comment */
    Block,
    /// /// doc comment
    DocLine,
    /// /** doc comment */
    DocBlock,
}

/// Format error
#[derive(Debug, Clone)]
pub enum FormatError {
    ParseError(String),
    IoError(String),
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::ParseError(s) => write!(f, "Parse error: {}", s),
            FormatError::IoError(s) => write!(f, "IO error: {}", s),
        }
    }
}

impl std::error::Error for FormatError {}

impl Formatter {
    /// Create a new formatter
    pub fn new(config: FormatConfig) -> Self {
        Formatter {
            config,
            source: String::new(),
            comments: Vec::new(),
        }
    }

    /// Create formatter with default config
    pub fn default_formatter() -> Self {
        Self::new(FormatConfig::default())
    }

    /// Format source code
    pub fn format(&mut self, source: &str) -> Result<String, FormatError> {
        self.source = source.to_string();

        // Extract comments before parsing
        self.comments = self.extract_comments(source);

        // Parse the source
        let tokens =
            crate::lexer::lex(source).map_err(|e| FormatError::ParseError(e.to_string()))?;
        let ast = crate::parser::parse(&tokens, source)
            .map_err(|e| FormatError::ParseError(e.to_string()))?;

        // Convert to format IR
        let doc = self.ast_to_doc(&ast);

        // Print
        let printer = Printer::new(self.config.clone());
        Ok(printer.print(&doc))
    }

    /// Format with diff output
    pub fn format_diff(&mut self, source: &str) -> Result<String, FormatError> {
        let formatted = self.format(source)?;

        if formatted == source {
            return Ok(String::new());
        }

        // Generate unified diff
        let mut diff = String::new();
        diff.push_str("--- original\n");
        diff.push_str("+++ formatted\n");

        let original_lines: Vec<&str> = source.lines().collect();
        let formatted_lines: Vec<&str> = formatted.lines().collect();

        let mut i = 0;
        let mut j = 0;

        while i < original_lines.len() || j < formatted_lines.len() {
            if i < original_lines.len() && j < formatted_lines.len() {
                if original_lines[i] == formatted_lines[j] {
                    diff.push_str(&format!(" {}\n", original_lines[i]));
                    i += 1;
                    j += 1;
                } else {
                    diff.push_str(&format!("-{}\n", original_lines[i]));
                    diff.push_str(&format!("+{}\n", formatted_lines[j]));
                    i += 1;
                    j += 1;
                }
            } else if i < original_lines.len() {
                diff.push_str(&format!("-{}\n", original_lines[i]));
                i += 1;
            } else {
                diff.push_str(&format!("+{}\n", formatted_lines[j]));
                j += 1;
            }
        }

        Ok(diff)
    }

    /// Extract comments from source
    fn extract_comments(&self, source: &str) -> Vec<Comment> {
        let mut comments = Vec::new();
        let mut chars = source.char_indices().peekable();

        while let Some((i, c)) = chars.next() {
            if c == '/' {
                if let Some(&(_, next)) = chars.peek() {
                    if next == '/' {
                        // Line comment
                        chars.next();
                        let start = i;
                        let mut content = String::new();
                        let is_doc = chars.peek().map(|&(_, c)| c == '/').unwrap_or(false);

                        if is_doc {
                            chars.next();
                        }

                        while let Some(&(_, c)) = chars.peek() {
                            if c == '\n' {
                                break;
                            }
                            content.push(c);
                            chars.next();
                        }

                        let content_len = content.len();
                        comments.push(Comment {
                            kind: if is_doc {
                                CommentKind::DocLine
                            } else {
                                CommentKind::Line
                            },
                            content,
                            span: Span {
                                start,
                                end: start + content_len + if is_doc { 3 } else { 2 },
                            },
                            is_doc,
                        });
                    } else if next == '*' {
                        // Block comment
                        chars.next();
                        let start = i;
                        let mut content = String::new();
                        let is_doc = chars.peek().map(|&(_, c)| c == '*').unwrap_or(false);

                        if is_doc {
                            chars.next();
                        }

                        let mut prev = ' ';
                        while let Some((_, c)) = chars.next() {
                            if prev == '*' && c == '/' {
                                content.pop(); // Remove trailing *
                                break;
                            }
                            content.push(c);
                            prev = c;
                        }

                        let content_len = content.len();
                        comments.push(Comment {
                            kind: if is_doc {
                                CommentKind::DocBlock
                            } else {
                                CommentKind::Block
                            },
                            content,
                            span: Span {
                                start,
                                end: start + content_len + if is_doc { 5 } else { 4 },
                            },
                            is_doc,
                        });
                    }
                }
            }
        }

        comments
    }

    /// Convert AST to document IR
    fn ast_to_doc(&self, ast: &Ast) -> Doc {
        let mut parts = Vec::new();

        for (i, item) in ast.items.iter().enumerate() {
            if i > 0 {
                parts.push(Doc::Hardline);
                parts.push(Doc::Hardline);
            }
            parts.push(self.item_to_doc(item));
        }

        Doc::Concat(parts)
    }

    /// Format an item
    fn item_to_doc(&self, item: &Item) -> Doc {
        match item {
            Item::Function(f) => self.function_to_doc(f),
            Item::Struct(s) => self.struct_to_doc(s),
            Item::Enum(e) => self.enum_to_doc(e),
            Item::Trait(t) => self.trait_to_doc(t),
            Item::Impl(i) => self.impl_to_doc(i),
            Item::TypeAlias(t) => self.type_alias_to_doc(t),
            // Note: There's no Const item in the AST - constants are GlobalDef with specific flags
            Item::Import(u) => self.import_to_doc(u),
            Item::Effect(e) => self.effect_to_doc(e),
            Item::Handler(h) => self.handler_to_doc(h),
            Item::Global(g) => self.global_to_doc(g),
            Item::Extern(e) => self.extern_to_doc(e),
            Item::MacroInvocation(m) => Doc::Text(format!("{}!(...)", m.name)),
            Item::OntologyImport(o) => {
                Doc::Text(format!("ontology {} from \"{}\";", o.prefix, o.source))
            }
            Item::AlignDecl(a) => Doc::Text(format!(
                "align {}:{} ~ {}:{} with distance {};",
                a.type1.prefix, a.type1.term, a.type2.prefix, a.type2.term, a.distance
            )),
            Item::Export(e) => {
                if e.names.len() == 1 {
                    Doc::Text(format!("export {};", e.names[0]))
                } else {
                    Doc::Text(format!("export {{ {} }};", e.names.join(", ")))
                }
            }
            Item::OdeDef(o) => Doc::Text(format!("ode {} {{ ... }}", o.name)),
            Item::PdeDef(p) => Doc::Text(format!("pde {} {{ ... }}", p.name)),
            Item::CausalModel(c) => Doc::Text(format!("causal model {} {{ ... }}", c.name)),
            Item::Module(m) => {
                let vis = if matches!(m.visibility, Visibility::Public) {
                    "pub "
                } else {
                    ""
                };
                if m.items.is_some() {
                    Doc::Text(format!("{}module {} {{ ... }}", vis, m.name))
                } else {
                    Doc::Text(format!("{}mod {};", vis, m.name))
                }
            }
        }
    }

    /// Format a function
    fn function_to_doc(&self, func: &FnDef) -> Doc {
        let mut parts = Vec::new();

        // Visibility
        if matches!(func.visibility, Visibility::Public) {
            parts.push(Doc::Text("pub ".to_string()));
        }

        // Modifiers
        if func.modifiers.is_async {
            parts.push(Doc::Text("async ".to_string()));
        }
        if func.modifiers.is_unsafe {
            parts.push(Doc::Text("unsafe ".to_string()));
        }
        if func.modifiers.is_kernel {
            parts.push(Doc::Text("kernel ".to_string()));
        }

        // fn keyword and name
        parts.push(Doc::Text(format!("fn {}", func.name)));

        // Generic parameters
        if !func.generics.params.is_empty() {
            parts.push(Doc::Text("<".to_string()));
            let params: Vec<Doc> = func
                .generics
                .params
                .iter()
                .map(|p| self.generic_param_to_doc(p))
                .collect();
            parts.push(Doc::join(params, Doc::Text(", ".to_string())));
            parts.push(Doc::Text(">".to_string()));
        }

        // Parameters
        parts.push(Doc::Text("(".to_string()));

        if func.params.is_empty() {
            parts.push(Doc::Text(")".to_string()));
        } else if func.params.len() <= 3 {
            // Single line parameters
            let params: Vec<Doc> = func.params.iter().map(|p| self.param_to_doc(p)).collect();
            parts.push(Doc::join(params, Doc::Text(", ".to_string())));
            parts.push(Doc::Text(")".to_string()));
        } else {
            // Multi-line parameters
            let params: Vec<Doc> = func
                .params
                .iter()
                .map(|p| {
                    Doc::Concat(vec![
                        Doc::Hardline,
                        Doc::Indent(Box::new(Doc::Concat(vec![
                            self.param_to_doc(p),
                            Doc::Text(",".to_string()),
                        ]))),
                    ])
                })
                .collect();
            parts.push(Doc::Concat(params));
            parts.push(Doc::Hardline);
            parts.push(Doc::Text(")".to_string()));
        }

        // Return type
        if let Some(ret) = &func.return_type {
            parts.push(Doc::Text(" -> ".to_string()));
            parts.push(self.type_to_doc(ret));
        }

        // Effects
        if !func.effects.is_empty() {
            parts.push(Doc::Text(" with ".to_string()));
            let effects: Vec<Doc> = func
                .effects
                .iter()
                .map(|e| Doc::Text(e.name.segments.join("::")))
                .collect();
            parts.push(Doc::join(effects, Doc::Text(", ".to_string())));
        }

        // Body
        parts.push(Doc::Text(" ".to_string()));
        parts.push(self.block_to_doc(&func.body));

        Doc::Concat(parts)
    }

    /// Format a struct
    fn struct_to_doc(&self, s: &StructDef) -> Doc {
        let mut parts = Vec::new();

        // Visibility
        if matches!(s.visibility, Visibility::Public) {
            parts.push(Doc::Text("pub ".to_string()));
        }

        // Linearity
        if s.modifiers.linear {
            parts.push(Doc::Text("linear ".to_string()));
        } else if s.modifiers.affine {
            parts.push(Doc::Text("affine ".to_string()));
        }

        parts.push(Doc::Text(format!("struct {}", s.name)));

        // Generics
        if !s.generics.params.is_empty() {
            parts.push(Doc::Text("<".to_string()));
            let params: Vec<Doc> = s
                .generics
                .params
                .iter()
                .map(|p| self.generic_param_to_doc(p))
                .collect();
            parts.push(Doc::join(params, Doc::Text(", ".to_string())));
            parts.push(Doc::Text(">".to_string()));
        }

        // Fields
        if s.fields.is_empty() {
            parts.push(Doc::Text(" {}".to_string()));
        } else {
            parts.push(Doc::Text(" {".to_string()));
            for field in &s.fields {
                parts.push(Doc::Indent(Box::new(Doc::Concat(vec![
                    Doc::Hardline,
                    Doc::Text(format!("{}: ", field.name)),
                    self.type_to_doc(&field.ty),
                    Doc::Text(",".to_string()),
                ]))));
            }
            parts.push(Doc::Hardline);
            parts.push(Doc::Text("}".to_string()));
        }

        Doc::Concat(parts)
    }

    /// Format an enum
    fn enum_to_doc(&self, e: &EnumDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(e.visibility, Visibility::Public) {
            parts.push(Doc::Text("pub ".to_string()));
        }

        parts.push(Doc::Text(format!("enum {}", e.name)));

        if !e.generics.params.is_empty() {
            parts.push(Doc::Text("<".to_string()));
            let params: Vec<Doc> = e
                .generics
                .params
                .iter()
                .map(|p| self.generic_param_to_doc(p))
                .collect();
            parts.push(Doc::join(params, Doc::Text(", ".to_string())));
            parts.push(Doc::Text(">".to_string()));
        }

        parts.push(Doc::Text(" {".to_string()));

        for variant in &e.variants {
            parts.push(Doc::Indent(Box::new(Doc::Concat(vec![
                Doc::Hardline,
                Doc::Text(format!("{},", variant.name)),
            ]))));
        }

        parts.push(Doc::Hardline);
        parts.push(Doc::Text("}".to_string()));

        Doc::Concat(parts)
    }

    /// Format a trait
    fn trait_to_doc(&self, t: &TraitDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(t.visibility, Visibility::Public) {
            parts.push(Doc::Text("pub ".to_string()));
        }

        parts.push(Doc::Text(format!("trait {}", t.name)));

        if !t.generics.params.is_empty() {
            parts.push(Doc::Text("<".to_string()));
            let params: Vec<Doc> = t
                .generics
                .params
                .iter()
                .map(|p| self.generic_param_to_doc(p))
                .collect();
            parts.push(Doc::join(params, Doc::Text(", ".to_string())));
            parts.push(Doc::Text(">".to_string()));
        }

        parts.push(Doc::Text(" {".to_string()));
        parts.push(Doc::Hardline);
        parts.push(Doc::Text("}".to_string()));

        Doc::Concat(parts)
    }

    /// Format an impl block
    fn impl_to_doc(&self, i: &ImplDef) -> Doc {
        let mut parts = Vec::new();

        parts.push(Doc::Text("impl".to_string()));

        if !i.generics.params.is_empty() {
            parts.push(Doc::Text("<".to_string()));
            let params: Vec<Doc> = i
                .generics
                .params
                .iter()
                .map(|p| self.generic_param_to_doc(p))
                .collect();
            parts.push(Doc::join(params, Doc::Text(", ".to_string())));
            parts.push(Doc::Text(">".to_string()));
        }

        if let Some(trait_ref) = &i.trait_ref {
            parts.push(Doc::Text(format!(" {} for", trait_ref.segments.join("::"))));
        }

        parts.push(Doc::Text(" ".to_string()));
        parts.push(self.type_to_doc(&i.target_type));
        parts.push(Doc::Text(" {".to_string()));
        parts.push(Doc::Hardline);
        parts.push(Doc::Text("}".to_string()));

        Doc::Concat(parts)
    }

    /// Format a type alias
    fn type_alias_to_doc(&self, t: &TypeAliasDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(t.visibility, Visibility::Public) {
            parts.push(Doc::Text("pub ".to_string()));
        }

        parts.push(Doc::Text(format!("type {} = ", t.name)));
        parts.push(self.type_to_doc(&t.ty));
        parts.push(Doc::Text(";".to_string()));

        Doc::Concat(parts)
    }

    /// Format an import
    fn import_to_doc(&self, u: &crate::ast::ImportDef) -> Doc {
        Doc::Text(format!("use {};", u.path.segments.join("::")))
    }

    /// Format a global
    fn global_to_doc(&self, g: &crate::ast::GlobalDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(g.visibility, Visibility::Public) {
            parts.push(Doc::Text("pub ".to_string()));
        }

        if g.is_const {
            parts.push(Doc::Text("const ".to_string()));
        } else if g.is_mut {
            parts.push(Doc::Text("static mut ".to_string()));
        } else {
            parts.push(Doc::Text("static ".to_string()));
        }

        parts.push(self.pattern_to_doc(&g.pattern));

        if let Some(ty) = &g.ty {
            parts.push(Doc::Text(": ".to_string()));
            parts.push(self.type_to_doc(ty));
        }

        parts.push(Doc::Text(" = ".to_string()));
        parts.push(self.expr_to_doc(&g.value));

        parts.push(Doc::Text(";".to_string()));
        Doc::Concat(parts)
    }

    /// Format an extern block
    fn extern_to_doc(&self, _e: &crate::ast::ExternBlock) -> Doc {
        Doc::Text("extern { ... }".to_string())
    }

    /// Format an effect
    fn effect_to_doc(&self, e: &EffectDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(e.visibility, Visibility::Public) {
            parts.push(Doc::Text("pub ".to_string()));
        }

        parts.push(Doc::Text(format!("effect {} {{", e.name)));
        parts.push(Doc::Hardline);
        parts.push(Doc::Text("}".to_string()));

        Doc::Concat(parts)
    }

    /// Format a handler
    fn handler_to_doc(&self, h: &HandlerDef) -> Doc {
        let mut parts = Vec::new();

        parts.push(Doc::Text(format!(
            "handler {} for {} {{",
            h.name,
            h.effect.segments.join("::")
        )));
        parts.push(Doc::Hardline);
        parts.push(Doc::Text("}".to_string()));

        Doc::Concat(parts)
    }

    /// Format a generic parameter
    fn generic_param_to_doc(&self, param: &crate::ast::GenericParam) -> Doc {
        match param {
            crate::ast::GenericParam::Type { name, bounds, .. } => {
                if bounds.is_empty() {
                    Doc::Text(name.clone())
                } else {
                    let bounds_str = bounds
                        .iter()
                        .map(|p| p.segments.join("::"))
                        .collect::<Vec<_>>()
                        .join(" + ");
                    Doc::Text(format!("{}: {}", name, bounds_str))
                }
            }
            crate::ast::GenericParam::Const { name, ty, .. } => Doc::Concat(vec![
                Doc::Text(format!("const {}: ", name)),
                self.type_to_doc(ty),
            ]),
        }
    }

    /// Format a parameter
    fn param_to_doc(&self, param: &Param) -> Doc {
        let mut parts = Vec::new();

        parts.push(self.pattern_to_doc(&param.pattern));
        parts.push(Doc::Text(": ".to_string()));
        parts.push(self.type_to_doc(&param.ty));

        Doc::Concat(parts)
    }

    /// Format a type
    fn type_to_doc(&self, ty: &TypeExpr) -> Doc {
        match ty {
            TypeExpr::Unit => Doc::Text("()".to_string()),
            TypeExpr::SelfType => Doc::Text("Self".to_string()),
            TypeExpr::Named { path, args, unit } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text(path.segments.join("::")));

                if !args.is_empty() {
                    parts.push(Doc::Text("<".to_string()));
                    let args: Vec<Doc> = args.iter().map(|a| self.type_to_doc(a)).collect();
                    parts.push(Doc::join(args, Doc::Text(", ".to_string())));
                    parts.push(Doc::Text(">".to_string()));
                }

                if let Some(u) = unit {
                    parts.push(Doc::Text(format!("<{}>", u)));
                }

                Doc::Concat(parts)
            }
            TypeExpr::Reference { mutable, inner } => {
                if *mutable {
                    Doc::Concat(vec![Doc::Text("&!".to_string()), self.type_to_doc(inner)])
                } else {
                    Doc::Concat(vec![Doc::Text("&".to_string()), self.type_to_doc(inner)])
                }
            }
            TypeExpr::RawPointer { mutable, inner } => {
                if *mutable {
                    Doc::Concat(vec![
                        Doc::Text("*mut ".to_string()),
                        self.type_to_doc(inner),
                    ])
                } else {
                    Doc::Concat(vec![
                        Doc::Text("*const ".to_string()),
                        self.type_to_doc(inner),
                    ])
                }
            }
            TypeExpr::Array { element, size } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text("[".to_string()));
                parts.push(self.type_to_doc(element));
                if let Some(s) = size {
                    parts.push(Doc::Text("; ".to_string()));
                    parts.push(self.expr_to_doc(s));
                }
                parts.push(Doc::Text("]".to_string()));
                Doc::Concat(parts)
            }
            TypeExpr::Tuple(elements) => {
                let mut parts = Vec::new();
                parts.push(Doc::Text("(".to_string()));
                let elems: Vec<Doc> = elements.iter().map(|e| self.type_to_doc(e)).collect();
                parts.push(Doc::join(elems, Doc::Text(", ".to_string())));
                parts.push(Doc::Text(")".to_string()));
                Doc::Concat(parts)
            }
            TypeExpr::Function {
                params,
                return_type,
                effects,
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text("fn(".to_string()));
                let ps: Vec<Doc> = params.iter().map(|p| self.type_to_doc(p)).collect();
                parts.push(Doc::join(ps, Doc::Text(", ".to_string())));
                parts.push(Doc::Text(") -> ".to_string()));
                parts.push(self.type_to_doc(return_type));
                if !effects.is_empty() {
                    parts.push(Doc::Text(" with ".to_string()));
                    let effs: Vec<Doc> = effects
                        .iter()
                        .map(|e| Doc::Text(e.name.segments.join("::")))
                        .collect();
                    parts.push(Doc::join(effs, Doc::Text(", ".to_string())));
                }
                Doc::Concat(parts)
            }
            TypeExpr::Infer => Doc::Text("_".to_string()),

            // Epistemic types
            TypeExpr::Knowledge { value_type, .. } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text("Knowledge[".to_string()));
                parts.push(self.type_to_doc(value_type));
                parts.push(Doc::Text("]".to_string()));
                Doc::Concat(parts)
            }
            TypeExpr::Quantity { numeric_type, unit } => {
                let unit_str = unit
                    .base_units
                    .iter()
                    .map(|(name, exp)| {
                        if *exp == 1 {
                            name.clone()
                        } else {
                            format!("{}^{}", name, exp)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("*");
                let mut parts = Vec::new();
                parts.push(Doc::Text("Quantity[".to_string()));
                parts.push(self.type_to_doc(numeric_type));
                parts.push(Doc::Text(format!(", {}]", unit_str)));
                Doc::Concat(parts)
            }
            TypeExpr::Tensor {
                element_type,
                shape,
            } => {
                let shape_str = shape
                    .iter()
                    .map(|d| match d {
                        crate::ast::TensorDim::Named(n) => n.clone(),
                        crate::ast::TensorDim::Fixed(s) => s.to_string(),
                        crate::ast::TensorDim::Dynamic => "_".to_string(),
                        crate::ast::TensorDim::Expr(_) => "<expr>".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut parts = Vec::new();
                parts.push(Doc::Text("Tensor[".to_string()));
                parts.push(self.type_to_doc(element_type));
                parts.push(Doc::Text(format!(", ({})]", shape_str)));
                Doc::Concat(parts)
            }
            TypeExpr::Ontology { ontology, term } => {
                if let Some(t) = term {
                    Doc::Text(format!("OntologyTerm[{}:{}]", ontology, t))
                } else {
                    Doc::Text(format!("OntologyTerm[{}]", ontology))
                }
            }
            TypeExpr::Linear { inner, linearity } => {
                let kind = match linearity {
                    crate::ast::LinearityKind::Linear => "linear",
                    crate::ast::LinearityKind::Affine => "affine",
                    crate::ast::LinearityKind::Relevant => "relevant",
                    crate::ast::LinearityKind::Unrestricted => "",
                };
                if kind.is_empty() {
                    self.type_to_doc(inner)
                } else {
                    Doc::Concat(vec![
                        self.type_to_doc(inner),
                        Doc::Text(format!(" @ {}", kind)),
                    ])
                }
            }
            TypeExpr::Effected { inner, effects } => {
                let effects_str = effects.effects.join(", ");
                Doc::Concat(vec![
                    self.type_to_doc(inner),
                    Doc::Text(format!(" ! {{{}}}", effects_str)),
                ])
            }
            TypeExpr::Tile {
                element_type,
                tile_m,
                tile_n,
                layout,
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text("tile<".to_string()));
                parts.push(self.type_to_doc(element_type));
                parts.push(Doc::Text(format!(", {}, {}", tile_m, tile_n)));
                if let Some(l) = layout {
                    parts.push(Doc::Text(format!(", \"{}\"", l)));
                }
                parts.push(Doc::Text(">".to_string()));
                Doc::Concat(parts)
            }
            TypeExpr::Refinement {
                var,
                base_type,
                predicate,
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text(format!("{{ {}: ", var)));
                parts.push(self.type_to_doc(base_type));
                parts.push(Doc::Text(" | ".to_string()));
                parts.push(self.expr_to_doc(predicate));
                parts.push(Doc::Text(" }".to_string()));
                Doc::Concat(parts)
            }
        }
    }

    /// Format a block
    fn block_to_doc(&self, block: &Block) -> Doc {
        let mut parts = Vec::new();

        parts.push(Doc::Text("{".to_string()));

        if block.stmts.is_empty() {
            parts.push(Doc::Text("}".to_string()));
        } else {
            for stmt in &block.stmts {
                parts.push(Doc::Indent(Box::new(Doc::Concat(vec![
                    Doc::Hardline,
                    self.stmt_to_doc(stmt),
                ]))));
            }
            parts.push(Doc::Hardline);
            parts.push(Doc::Text("}".to_string()));
        }

        Doc::Concat(parts)
    }

    /// Format a statement
    fn stmt_to_doc(&self, stmt: &Stmt) -> Doc {
        match stmt {
            Stmt::Let {
                pattern, ty, value, ..
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text("let ".to_string()));
                parts.push(self.pattern_to_doc(pattern));
                if let Some(t) = ty {
                    parts.push(Doc::Text(": ".to_string()));
                    parts.push(self.type_to_doc(t));
                }
                if let Some(v) = value {
                    parts.push(Doc::Text(" = ".to_string()));
                    parts.push(self.expr_to_doc(v));
                }
                parts.push(Doc::Text(";".to_string()));
                Doc::Concat(parts)
            }
            Stmt::Expr { expr, has_semi, .. } => {
                if *has_semi {
                    Doc::Concat(vec![self.expr_to_doc(expr), Doc::Text(";".to_string())])
                } else {
                    self.expr_to_doc(expr)
                }
            }
            Stmt::Assign { target, value, .. } => Doc::Concat(vec![
                self.expr_to_doc(target),
                Doc::Text(" = ".to_string()),
                self.expr_to_doc(value),
                Doc::Text(";".to_string()),
            ]),
            Stmt::Empty => Doc::Text("".to_string()),
            Stmt::MacroInvocation(m) => Doc::Text(format!("{}!(...);", m.name)),
        }
    }

    /// Format a pattern
    fn pattern_to_doc(&self, pattern: &Pattern) -> Doc {
        match pattern {
            Pattern::Wildcard => Doc::Text("_".to_string()),
            Pattern::Literal(lit) => Doc::Text(format!("{:?}", lit)),
            Pattern::Binding { name, mutable } => {
                if *mutable {
                    Doc::Text(format!("mut {}", name))
                } else {
                    Doc::Text(name.clone())
                }
            }
            Pattern::Tuple(patterns) => {
                let mut parts = Vec::new();
                parts.push(Doc::Text("(".to_string()));
                let ps: Vec<Doc> = patterns.iter().map(|p| self.pattern_to_doc(p)).collect();
                parts.push(Doc::join(ps, Doc::Text(", ".to_string())));
                parts.push(Doc::Text(")".to_string()));
                Doc::Concat(parts)
            }
            Pattern::Struct { path, fields } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text(path.segments.join("::")));
                parts.push(Doc::Text(" { ".to_string()));
                let fs: Vec<Doc> = fields
                    .iter()
                    .map(|(name, pat)| {
                        Doc::Concat(vec![
                            Doc::Text(format!("{}: ", name)),
                            self.pattern_to_doc(pat),
                        ])
                    })
                    .collect();
                parts.push(Doc::join(fs, Doc::Text(", ".to_string())));
                parts.push(Doc::Text(" }".to_string()));
                Doc::Concat(parts)
            }
            Pattern::Enum { path, patterns } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text(path.segments.join("::")));
                if let Some(ps) = patterns {
                    parts.push(Doc::Text("(".to_string()));
                    let pats: Vec<Doc> = ps.iter().map(|p| self.pattern_to_doc(p)).collect();
                    parts.push(Doc::join(pats, Doc::Text(", ".to_string())));
                    parts.push(Doc::Text(")".to_string()));
                }
                Doc::Concat(parts)
            }
            Pattern::Or(patterns) => {
                let ps: Vec<Doc> = patterns.iter().map(|p| self.pattern_to_doc(p)).collect();
                Doc::join(ps, Doc::Text(" | ".to_string()))
            }
        }
    }

    /// Format an expression
    fn expr_to_doc(&self, expr: &Expr) -> Doc {
        match expr {
            Expr::Literal { value, .. } => Doc::Text(format!("{:?}", value)),
            Expr::Path { path, .. } => Doc::Text(path.segments.join("::")),
            Expr::Binary {
                op, left, right, ..
            } => Doc::Concat(vec![
                self.expr_to_doc(left),
                Doc::Text(format!(" {} ", self.binop_to_string(op))),
                self.expr_to_doc(right),
            ]),
            Expr::Unary { op, expr, .. } => Doc::Concat(vec![
                Doc::Text(self.unop_to_string(op).to_string()),
                self.expr_to_doc(expr),
            ]),
            Expr::Call { callee, args, .. } => {
                let mut parts = Vec::new();
                parts.push(self.expr_to_doc(callee));
                parts.push(Doc::Text("(".to_string()));
                let args: Vec<Doc> = args.iter().map(|a| self.expr_to_doc(a)).collect();
                parts.push(Doc::join(args, Doc::Text(", ".to_string())));
                parts.push(Doc::Text(")".to_string()));
                Doc::Concat(parts)
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                let mut parts = Vec::new();
                parts.push(self.expr_to_doc(receiver));
                parts.push(Doc::Text(format!(".{}(", method)));
                let args: Vec<Doc> = args.iter().map(|a| self.expr_to_doc(a)).collect();
                parts.push(Doc::join(args, Doc::Text(", ".to_string())));
                parts.push(Doc::Text(")".to_string()));
                Doc::Concat(parts)
            }
            Expr::Field { base, field, .. } => Doc::Concat(vec![
                self.expr_to_doc(base),
                Doc::Text(format!(".{}", field)),
            ]),
            Expr::Index { base, index, .. } => Doc::Concat(vec![
                self.expr_to_doc(base),
                Doc::Text("[".to_string()),
                self.expr_to_doc(index),
                Doc::Text("]".to_string()),
            ]),
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text("if ".to_string()));
                parts.push(self.expr_to_doc(condition));
                parts.push(Doc::Text(" ".to_string()));
                parts.push(self.block_to_doc(then_branch));
                if let Some(else_b) = else_branch {
                    parts.push(Doc::Text(" else ".to_string()));
                    parts.push(self.expr_to_doc(else_b));
                }
                Doc::Concat(parts)
            }
            Expr::Match {
                scrutinee, arms, ..
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text("match ".to_string()));
                parts.push(self.expr_to_doc(scrutinee));
                parts.push(Doc::Text(" {".to_string()));
                for arm in arms {
                    parts.push(Doc::Indent(Box::new(Doc::Concat(vec![
                        Doc::Hardline,
                        self.pattern_to_doc(&arm.pattern),
                        Doc::Text(" => ".to_string()),
                        self.expr_to_doc(&arm.body),
                        Doc::Text(",".to_string()),
                    ]))));
                }
                parts.push(Doc::Hardline);
                parts.push(Doc::Text("}".to_string()));
                Doc::Concat(parts)
            }
            Expr::Block { block, .. } => self.block_to_doc(block),
            Expr::Loop { body, .. } => Doc::Concat(vec![
                Doc::Text("loop ".to_string()),
                self.block_to_doc(body),
            ]),
            Expr::While {
                condition, body, ..
            } => Doc::Concat(vec![
                Doc::Text("while ".to_string()),
                self.expr_to_doc(condition),
                Doc::Text(" ".to_string()),
                self.block_to_doc(body),
            ]),
            Expr::For {
                pattern,
                iter,
                body,
                ..
            } => Doc::Concat(vec![
                Doc::Text("for ".to_string()),
                self.pattern_to_doc(pattern),
                Doc::Text(" in ".to_string()),
                self.expr_to_doc(iter),
                Doc::Text(" ".to_string()),
                self.block_to_doc(body),
            ]),
            Expr::Return { value, .. } => {
                if let Some(v) = value {
                    Doc::Concat(vec![Doc::Text("return ".to_string()), self.expr_to_doc(v)])
                } else {
                    Doc::Text("return".to_string())
                }
            }
            Expr::Break { value, .. } => {
                if let Some(v) = value {
                    Doc::Concat(vec![Doc::Text("break ".to_string()), self.expr_to_doc(v)])
                } else {
                    Doc::Text("break".to_string())
                }
            }
            Expr::Continue { .. } => Doc::Text("continue".to_string()),
            Expr::Array { elements, .. } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text("[".to_string()));
                let elems: Vec<Doc> = elements.iter().map(|e| self.expr_to_doc(e)).collect();
                parts.push(Doc::join(elems, Doc::Text(", ".to_string())));
                parts.push(Doc::Text("]".to_string()));
                Doc::Concat(parts)
            }
            Expr::Tuple { elements, .. } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text("(".to_string()));
                let elems: Vec<Doc> = elements.iter().map(|e| self.expr_to_doc(e)).collect();
                parts.push(Doc::join(elems, Doc::Text(", ".to_string())));
                parts.push(Doc::Text(")".to_string()));
                Doc::Concat(parts)
            }
            Expr::StructLit { path, fields, .. } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text(path.segments.join("::")));
                parts.push(Doc::Text(" { ".to_string()));
                let fs: Vec<Doc> = fields
                    .iter()
                    .map(|(name, expr)| {
                        Doc::Concat(vec![
                            Doc::Text(format!("{}: ", name)),
                            self.expr_to_doc(expr),
                        ])
                    })
                    .collect();
                parts.push(Doc::join(fs, Doc::Text(", ".to_string())));
                parts.push(Doc::Text(" }".to_string()));
                Doc::Concat(parts)
            }
            Expr::Closure { params, body, .. } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text("|".to_string()));
                let ps: Vec<Doc> = params
                    .iter()
                    .map(|(name, ty)| {
                        if let Some(t) = ty {
                            Doc::Concat(vec![Doc::Text(format!("{}: ", name)), self.type_to_doc(t)])
                        } else {
                            Doc::Text(name.clone())
                        }
                    })
                    .collect();
                parts.push(Doc::join(ps, Doc::Text(", ".to_string())));
                parts.push(Doc::Text("| ".to_string()));
                parts.push(self.expr_to_doc(body));
                Doc::Concat(parts)
            }
            Expr::Cast { expr, ty, .. } => Doc::Concat(vec![
                self.expr_to_doc(expr),
                Doc::Text(" as ".to_string()),
                self.type_to_doc(ty),
            ]),
            Expr::Try { expr, .. } => {
                Doc::Concat(vec![self.expr_to_doc(expr), Doc::Text("?".to_string())])
            }
            Expr::Await { expr, .. } => Doc::Concat(vec![
                self.expr_to_doc(expr),
                Doc::Text(".await".to_string()),
            ]),
            Expr::Perform { effect, args, .. } => {
                let mut parts = Vec::new();
                parts.push(Doc::Text(format!("perform {}(", effect)));
                let args: Vec<Doc> = args.iter().map(|a| self.expr_to_doc(a)).collect();
                parts.push(Doc::join(args, Doc::Text(", ".to_string())));
                parts.push(Doc::Text(")".to_string()));
                Doc::Concat(parts)
            }
            Expr::Handle { expr, .. } => Doc::Concat(vec![
                Doc::Text("handle ".to_string()),
                self.expr_to_doc(expr),
                Doc::Text(" { ... }".to_string()),
            ]),
            // Handle remaining expression types with a generic format
            _ => Doc::Text("<expr>".to_string()),
        }
    }

    /// Convert binary operator to string
    fn binop_to_string(&self, op: &BinaryOp) -> &'static str {
        match op {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Rem => "%",
            BinaryOp::And => "&&",
            BinaryOp::Or => "||",
            BinaryOp::BitAnd => "&",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::PlusMinus => "±",
            BinaryOp::Concat => "++",
        }
    }

    /// Convert unary operator to string
    fn unop_to_string(&self, op: &UnaryOp) -> &'static str {
        match op {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "!",
            UnaryOp::Ref => "&",
            UnaryOp::RefMut => "&!",
            UnaryOp::Deref => "*",
        }
    }
}

impl Default for Formatter {
    fn default() -> Self {
        Self::default_formatter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formatter_creation() {
        let formatter = Formatter::default();
        assert_eq!(formatter.config.max_width, 100);
    }
}
