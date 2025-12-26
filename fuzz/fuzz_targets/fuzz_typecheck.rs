//! Fuzz target for the Sounio type checker
//!
//! Tests that the type checker handles arbitrary ASTs without panicking.
//! Uses structured fuzzing to generate syntactically valid programs.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use sounio::{check, lexer, parser};

/// Structured input for generating type-checkable Sounio programs
#[derive(Debug, Arbitrary)]
struct FuzzProgram {
    /// Top-level items
    items: Vec<TopLevelItem>,
}

#[derive(Debug, Arbitrary)]
enum TopLevelItem {
    Function(FuzzFunction),
    Struct(FuzzStruct),
    TypeAlias(FuzzTypeAlias),
}

#[derive(Debug, Arbitrary)]
struct FuzzFunction {
    name: SmallIdent,
    params: Vec<FuzzParam>,
    return_type: Option<FuzzType>,
    body: Vec<FuzzStatement>,
    effects: Vec<FuzzEffect>,
}

#[derive(Debug, Arbitrary)]
struct FuzzParam {
    name: SmallIdent,
    ty: FuzzType,
}

#[derive(Debug, Arbitrary)]
struct FuzzStruct {
    name: SmallIdent,
    fields: Vec<FuzzField>,
    is_linear: bool,
}

#[derive(Debug, Arbitrary)]
struct FuzzField {
    name: SmallIdent,
    ty: FuzzType,
}

#[derive(Debug, Arbitrary)]
struct FuzzTypeAlias {
    name: SmallIdent,
    ty: FuzzType,
}

#[derive(Debug, Arbitrary)]
enum FuzzType {
    /// Primitive types: i32, i64, f32, f64, bool, string
    Primitive(PrimitiveType),
    /// Reference type: &T or &!T
    Reference { inner: Box<FuzzType>, mutable: bool },
    /// Array type: [T; N]
    Array { inner: Box<FuzzType>, size: u8 },
    /// Slice type: [T]
    Slice { inner: Box<FuzzType> },
    /// Option type: Option<T>
    Option { inner: Box<FuzzType> },
    /// Result type: Result<T, E>
    Result { ok: Box<FuzzType>, err: Box<FuzzType> },
    /// Named type (identifier)
    Named(SmallIdent),
    /// Unit type: ()
    Unit,
}

#[derive(Debug, Arbitrary)]
enum PrimitiveType {
    I32,
    I64,
    F32,
    F64,
    Bool,
    String,
    U8,
    U32,
    U64,
}

#[derive(Debug, Arbitrary)]
enum FuzzEffect {
    IO,
    Mut,
    Alloc,
    Panic,
    Async,
    Prob,
}

#[derive(Debug, Arbitrary)]
enum FuzzStatement {
    /// let x = expr
    Let { name: SmallIdent, ty: Option<FuzzType>, expr: FuzzExpr },
    /// var x = expr
    Var { name: SmallIdent, ty: Option<FuzzType>, expr: FuzzExpr },
    /// expr;
    Expr(FuzzExpr),
    /// if cond { then } else { else }
    If { cond: FuzzExpr, then_block: Vec<FuzzStatement>, else_block: Option<Vec<FuzzStatement>> },
    /// return expr
    Return(Option<FuzzExpr>),
}

#[derive(Debug, Arbitrary)]
enum FuzzExpr {
    /// Integer literal
    IntLit(i32),
    /// Float literal
    FloatLit { mantissa: u16 },
    /// Boolean literal
    BoolLit(bool),
    /// String literal
    StringLit(SmallString),
    /// Variable reference
    Var(SmallIdent),
    /// Binary operation
    Binary { op: BinaryOp, left: Box<FuzzExpr>, right: Box<FuzzExpr> },
    /// Unary operation
    Unary { op: UnaryOp, expr: Box<FuzzExpr> },
    /// Function call
    Call { func: SmallIdent, args: Vec<FuzzExpr> },
    /// Field access
    Field { expr: Box<FuzzExpr>, field: SmallIdent },
    /// Array literal
    Array { elements: Vec<FuzzExpr> },
    /// Struct literal
    StructLit { name: SmallIdent, fields: Vec<(SmallIdent, FuzzExpr)> },
}

#[derive(Debug, Arbitrary)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Debug, Arbitrary)]
enum UnaryOp {
    Neg,
    Not,
    Ref,
    RefMut,
    Deref,
}

#[derive(Debug, Arbitrary)]
struct SmallIdent {
    first: u8,
    rest: [u8; 4],
    len: u8,
}

#[derive(Debug, Arbitrary)]
struct SmallString {
    chars: [u8; 8],
    len: u8,
}

impl FuzzProgram {
    fn to_source(&self) -> String {
        let mut source = String::new();
        for item in &self.items {
            source.push_str(&item.to_source());
            source.push('\n');
        }
        source
    }
}

impl TopLevelItem {
    fn to_source(&self) -> String {
        match self {
            TopLevelItem::Function(f) => f.to_source(),
            TopLevelItem::Struct(s) => s.to_source(),
            TopLevelItem::TypeAlias(t) => t.to_source(),
        }
    }
}

impl FuzzFunction {
    fn to_source(&self) -> String {
        let params: Vec<String> = self.params.iter().map(|p| p.to_source()).collect();
        let return_type = self.return_type.as_ref()
            .map(|t| format!(" -> {}", t.to_source()))
            .unwrap_or_default();
        let effects = if self.effects.is_empty() {
            String::new()
        } else {
            let eff_names: Vec<String> = self.effects.iter().map(|e| e.to_source()).collect();
            format!(" with {}", eff_names.join(", "))
        };
        let body: Vec<String> = self.body.iter().map(|s| format!("    {}", s.to_source())).collect();

        format!(
            "fn {}({}){}{} {{\n{}\n}}",
            self.name.to_source(),
            params.join(", "),
            return_type,
            effects,
            body.join("\n")
        )
    }
}

impl FuzzParam {
    fn to_source(&self) -> String {
        format!("{}: {}", self.name.to_source(), self.ty.to_source())
    }
}

impl FuzzStruct {
    fn to_source(&self) -> String {
        let linear = if self.is_linear { "linear " } else { "" };
        let fields: Vec<String> = self.fields.iter()
            .map(|f| format!("    {}", f.to_source()))
            .collect();
        format!(
            "{}struct {} {{\n{}\n}}",
            linear,
            self.name.to_source(),
            fields.join(",\n")
        )
    }
}

impl FuzzField {
    fn to_source(&self) -> String {
        format!("{}: {}", self.name.to_source(), self.ty.to_source())
    }
}

impl FuzzTypeAlias {
    fn to_source(&self) -> String {
        format!("type {} = {};", self.name.to_source(), self.ty.to_source())
    }
}

impl FuzzType {
    fn to_source(&self) -> String {
        match self {
            FuzzType::Primitive(p) => p.to_source(),
            FuzzType::Reference { inner, mutable } => {
                if *mutable {
                    format!("&!{}", inner.to_source())
                } else {
                    format!("&{}", inner.to_source())
                }
            }
            FuzzType::Array { inner, size } => {
                format!("[{}; {}]", inner.to_source(), size % 16 + 1)
            }
            FuzzType::Slice { inner } => format!("[{}]", inner.to_source()),
            FuzzType::Option { inner } => format!("Option<{}>", inner.to_source()),
            FuzzType::Result { ok, err } => {
                format!("Result<{}, {}>", ok.to_source(), err.to_source())
            }
            FuzzType::Named(name) => name.to_source(),
            FuzzType::Unit => "()".to_string(),
        }
    }
}

impl PrimitiveType {
    fn to_source(&self) -> String {
        match self {
            PrimitiveType::I32 => "i32".to_string(),
            PrimitiveType::I64 => "i64".to_string(),
            PrimitiveType::F32 => "f32".to_string(),
            PrimitiveType::F64 => "f64".to_string(),
            PrimitiveType::Bool => "bool".to_string(),
            PrimitiveType::String => "string".to_string(),
            PrimitiveType::U8 => "u8".to_string(),
            PrimitiveType::U32 => "u32".to_string(),
            PrimitiveType::U64 => "u64".to_string(),
        }
    }
}

impl FuzzEffect {
    fn to_source(&self) -> String {
        match self {
            FuzzEffect::IO => "IO".to_string(),
            FuzzEffect::Mut => "Mut".to_string(),
            FuzzEffect::Alloc => "Alloc".to_string(),
            FuzzEffect::Panic => "Panic".to_string(),
            FuzzEffect::Async => "Async".to_string(),
            FuzzEffect::Prob => "Prob".to_string(),
        }
    }
}

impl FuzzStatement {
    fn to_source(&self) -> String {
        match self {
            FuzzStatement::Let { name, ty, expr } => {
                let ty_annot = ty.as_ref()
                    .map(|t| format!(": {}", t.to_source()))
                    .unwrap_or_default();
                format!("let {}{} = {};", name.to_source(), ty_annot, expr.to_source())
            }
            FuzzStatement::Var { name, ty, expr } => {
                let ty_annot = ty.as_ref()
                    .map(|t| format!(": {}", t.to_source()))
                    .unwrap_or_default();
                format!("var {}{} = {};", name.to_source(), ty_annot, expr.to_source())
            }
            FuzzStatement::Expr(expr) => format!("{};", expr.to_source()),
            FuzzStatement::If { cond, then_block, else_block } => {
                let then_stmts: Vec<String> = then_block.iter()
                    .map(|s| format!("        {}", s.to_source()))
                    .collect();
                let else_part = else_block.as_ref().map(|block| {
                    let stmts: Vec<String> = block.iter()
                        .map(|s| format!("        {}", s.to_source()))
                        .collect();
                    format!(" else {{\n{}\n    }}", stmts.join("\n"))
                }).unwrap_or_default();
                format!("if {} {{\n{}\n    }}{}", cond.to_source(), then_stmts.join("\n"), else_part)
            }
            FuzzStatement::Return(expr) => {
                match expr {
                    Some(e) => format!("return {};", e.to_source()),
                    None => "return;".to_string(),
                }
            }
        }
    }
}

impl FuzzExpr {
    fn to_source(&self) -> String {
        match self {
            FuzzExpr::IntLit(n) => n.to_string(),
            FuzzExpr::FloatLit { mantissa } => format!("{}.0", mantissa),
            FuzzExpr::BoolLit(b) => if *b { "true" } else { "false" }.to_string(),
            FuzzExpr::StringLit(s) => format!("\"{}\"", s.to_source()),
            FuzzExpr::Var(name) => name.to_source(),
            FuzzExpr::Binary { op, left, right } => {
                format!("({} {} {})", left.to_source(), op.to_source(), right.to_source())
            }
            FuzzExpr::Unary { op, expr } => {
                format!("({}{})", op.to_source(), expr.to_source())
            }
            FuzzExpr::Call { func, args } => {
                let arg_strs: Vec<String> = args.iter().map(|a| a.to_source()).collect();
                format!("{}({})", func.to_source(), arg_strs.join(", "))
            }
            FuzzExpr::Field { expr, field } => {
                format!("{}.{}", expr.to_source(), field.to_source())
            }
            FuzzExpr::Array { elements } => {
                let elem_strs: Vec<String> = elements.iter().map(|e| e.to_source()).collect();
                format!("[{}]", elem_strs.join(", "))
            }
            FuzzExpr::StructLit { name, fields } => {
                let field_strs: Vec<String> = fields.iter()
                    .map(|(n, e)| format!("{}: {}", n.to_source(), e.to_source()))
                    .collect();
                format!("{} {{ {} }}", name.to_source(), field_strs.join(", "))
            }
        }
    }
}

impl BinaryOp {
    fn to_source(&self) -> String {
        match self {
            BinaryOp::Add => "+".to_string(),
            BinaryOp::Sub => "-".to_string(),
            BinaryOp::Mul => "*".to_string(),
            BinaryOp::Div => "/".to_string(),
            BinaryOp::Mod => "%".to_string(),
            BinaryOp::Eq => "==".to_string(),
            BinaryOp::Ne => "!=".to_string(),
            BinaryOp::Lt => "<".to_string(),
            BinaryOp::Le => "<=".to_string(),
            BinaryOp::Gt => ">".to_string(),
            BinaryOp::Ge => ">=".to_string(),
            BinaryOp::And => "&&".to_string(),
            BinaryOp::Or => "||".to_string(),
        }
    }
}

impl UnaryOp {
    fn to_source(&self) -> String {
        match self {
            UnaryOp::Neg => "-".to_string(),
            UnaryOp::Not => "!".to_string(),
            UnaryOp::Ref => "&".to_string(),
            UnaryOp::RefMut => "&!".to_string(),
            UnaryOp::Deref => "*".to_string(),
        }
    }
}

impl SmallIdent {
    fn to_source(&self) -> String {
        let mut s = String::new();
        // First char: letter or underscore (avoid reserved words)
        let first = match self.first % 53 {
            0..=25 => (b'a' + self.first % 26) as char,
            26..=51 => (b'A' + (self.first - 26) % 26) as char,
            _ => '_',
        };
        s.push(first);

        // Add a prefix to avoid colliding with keywords
        s.push_str("_fuzz");

        // Rest: alphanumeric or underscore
        let len = (self.len % 5) as usize;
        for i in 0..len {
            let c = self.rest[i];
            let ch = match c % 63 {
                0..=25 => (b'a' + c % 26) as char,
                26..=51 => (b'A' + (c - 26) % 26) as char,
                52..=61 => (b'0' + (c - 52) % 10) as char,
                _ => '_',
            };
            s.push(ch);
        }
        s
    }
}

impl SmallString {
    fn to_source(&self) -> String {
        let len = (self.len % 9) as usize;
        self.chars[..len]
            .iter()
            .filter_map(|&c| {
                // Only allow safe printable ASCII characters
                if c >= 32 && c < 127 && c != b'"' && c != b'\\' {
                    Some(c as char)
                } else {
                    None
                }
            })
            .collect()
    }
}

fuzz_target!(|program: FuzzProgram| {
    let source = program.to_source();

    // Skip empty programs
    if source.trim().is_empty() {
        return;
    }

    // First, lex the source
    if let Ok(tokens) = lexer::lex(&source) {
        // Then parse
        if let Ok(ast) = parser::parse(&tokens, &source) {
            // Finally, type check - should not panic even on invalid programs
            let _ = check::check(&ast);
        }
    }
});
