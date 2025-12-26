//! Abstract Syntax Tree for the Sounio language
//!
//! This module defines the AST types produced by the parser.

use crate::common::{NodeId, Span};
use serde::{Deserialize, Serialize};

// ==================== ATTRIBUTES ====================

/// Attribute applied to items, expressions, or types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attribute {
    pub id: NodeId,
    /// Attribute name (e.g., "compat", "derive", "cfg")
    pub name: String,
    /// Attribute arguments
    pub args: AttributeArgs,
    /// Attribute span
    pub span: Span,
}

/// Attribute arguments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeArgs {
    /// No arguments: #[inline]
    Empty,
    /// Single value: #[compat(0.15)]
    Value(AttributeValue),
    /// Named arguments: #[cfg(target_os = "linux")]
    Named(Vec<(String, AttributeValue)>),
    /// List of values: #[derive(Debug, Clone)]
    List(Vec<AttributeValue>),
}

/// Attribute value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    /// String literal: "value"
    String(String),
    /// Integer: 42
    Int(i64),
    /// Float: 0.15
    Float(f64),
    /// Boolean: true/false
    Bool(bool),
    /// Path/identifier: Debug, std::fmt::Display
    Path(Path),
    /// Nested attribute: cfg(all(unix, not(target_os = "macos")))
    Nested(String, Box<AttributeArgs>),
}

impl Attribute {
    /// Create a simple attribute with no arguments
    pub fn simple(name: &str) -> Self {
        Self {
            id: NodeId::dummy(),
            name: name.to_string(),
            args: AttributeArgs::Empty,
            span: Span::dummy(),
        }
    }

    /// Create a compat attribute with a threshold
    pub fn compat(threshold: f64) -> Self {
        Self {
            id: NodeId::dummy(),
            name: "compat".to_string(),
            args: AttributeArgs::Value(AttributeValue::Float(threshold)),
            span: Span::dummy(),
        }
    }

    /// Create a compat attribute with a named level
    pub fn compat_level(level: &str) -> Self {
        Self {
            id: NodeId::dummy(),
            name: "compat".to_string(),
            args: AttributeArgs::Value(AttributeValue::Path(Path::simple(level))),
            span: Span::dummy(),
        }
    }

    /// Check if this is a compat attribute
    pub fn is_compat(&self) -> bool {
        self.name == "compat"
    }

    /// Get compat threshold if this is a compat attribute
    pub fn compat_threshold(&self) -> Option<f64> {
        if !self.is_compat() {
            return None;
        }

        match &self.args {
            AttributeArgs::Value(AttributeValue::Float(f)) => Some(*f),
            AttributeArgs::Value(AttributeValue::Path(p)) => {
                // Parse named levels
                match p.segments.first().map(|s| s.as_str()) {
                    Some("exact") => Some(0.0),
                    Some("strict") => Some(0.05),
                    Some("default") => Some(0.15),
                    Some("loose") => Some(0.25),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// Top-level AST
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Ast {
    pub module_name: Option<Path>,
    pub items: Vec<Item>,
    /// Mapping from NodeId to source span for error reporting
    #[serde(skip)]
    pub node_spans: std::collections::HashMap<crate::common::NodeId, crate::common::Span>,
}

/// Item visibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    Private,
}

/// Common modifiers
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Modifiers {
    pub linear: bool,
    pub affine: bool,
    pub is_async: bool,
    pub is_unsafe: bool,
}

/// Type modifiers (linear/affine)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeModifiers {
    pub linear: bool,
    pub affine: bool,
}

/// Function modifiers
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FnModifiers {
    pub is_async: bool,
    pub is_unsafe: bool,
    pub is_kernel: bool,
    /// ABI for extern functions (e.g., extern "C" fn)
    pub abi: Option<Abi>,
}

/// Top-level item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Item {
    Function(FnDef),
    Struct(StructDef),
    Enum(EnumDef),
    Trait(TraitDef),
    Impl(ImplDef),
    TypeAlias(TypeAliasDef),
    Effect(EffectDef),
    Handler(HandlerDef),
    Import(ImportDef),
    Export(ExportDef),
    Extern(ExternBlock),
    Global(GlobalDef),
    MacroInvocation(MacroInvocation),
    /// Ontology import: `ontology chebi from "https://...";`
    OntologyImport(OntologyImportDef),
    /// Alignment declaration: `align chebi:drug ~ drugbank:drug with distance 0.1;`
    AlignDecl(AlignDef),
    /// ODE definition: `ode LotkaVolterra { ... }`
    OdeDef(OdeDef),
    /// PDE definition: `pde HeatEquation { ... }`
    PdeDef(PdeDef),
    /// Causal model definition: `causal model SmokingCancer { ... }`
    CausalModel(CausalModelDef),
    /// Module definition: `pub module foo { ... }` or `mod foo;`
    Module(ModuleDef),
}

// ==================== MODULES ====================

/// Module definition
/// Supports both inline modules and file-based modules:
/// - Inline: `pub module foo { fn bar() {} }`
/// - File-based: `mod foo;` (loads from foo.d or foo/mod.d)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDef {
    pub id: NodeId,
    pub visibility: Visibility,
    /// Module name (single identifier, not a path)
    pub name: String,
    /// Module items: Some(items) for inline, None for file-based (`mod foo;`)
    pub items: Option<Vec<Item>>,
    pub span: Span,
}

// ==================== FUNCTIONS ====================

/// Function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub modifiers: FnModifiers,
    /// Attributes (e.g., #[compat(strict)], #[inline])
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub generics: Generics,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub effects: Vec<EffectRef>,
    pub where_clause: Vec<WherePredicate>,
    pub body: Block,
    pub span: Span,
}

/// Function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub id: NodeId,
    pub is_mut: bool,
    pub pattern: Pattern,
    pub ty: TypeExpr,
    /// Attributes (e.g., #[compat(0.15)] for parameter-level compatibility)
    pub attributes: Vec<Attribute>,
}

// ==================== STRUCTS ====================

/// Struct definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub modifiers: TypeModifiers,
    /// Attributes (e.g., #[derive(Debug)], #[repr(C)])
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub generics: Generics,
    pub where_clause: Vec<WherePredicate>,
    pub fields: Vec<FieldDef>,
    pub span: Span,
}

/// Field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    pub id: NodeId,
    pub visibility: Visibility,
    /// Attributes (e.g., #[compat(strict)] for field-level compatibility)
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub ty: TypeExpr,
}

// ==================== ENUMS ====================

/// Enum definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub modifiers: TypeModifiers,
    pub name: String,
    pub generics: Generics,
    pub where_clause: Vec<WherePredicate>,
    pub variants: Vec<VariantDef>,
    pub span: Span,
}

/// Enum variant definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantDef {
    pub id: NodeId,
    pub name: String,
    pub data: VariantData,
}

/// Variant data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariantData {
    Unit,
    Tuple(Vec<TypeExpr>),
    Struct(Vec<FieldDef>),
}

// ==================== TRAITS ====================

/// Trait definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    pub generics: Generics,
    pub supertraits: Vec<Path>,
    pub where_clause: Vec<WherePredicate>,
    pub items: Vec<TraitItem>,
    pub span: Span,
}

/// Trait item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraitItem {
    Fn(TraitFnDef),
    Type(TraitTypeDef),
}

/// Trait function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitFnDef {
    pub id: NodeId,
    pub name: String,
    pub generics: Generics,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub effects: Vec<EffectRef>,
    pub where_clause: Vec<WherePredicate>,
    pub default_body: Option<Block>,
}

/// Trait associated type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitTypeDef {
    pub id: NodeId,
    pub name: String,
    pub bounds: Vec<Path>,
    pub default: Option<TypeExpr>,
}

// ==================== IMPL ====================

/// Impl block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplDef {
    pub id: NodeId,
    pub generics: Generics,
    pub trait_ref: Option<Path>,
    pub target_type: TypeExpr,
    pub where_clause: Vec<WherePredicate>,
    pub items: Vec<ImplItem>,
    pub span: Span,
}

/// Impl item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplItem {
    Fn(FnDef),
    Type(ImplTypeDef),
}

/// Impl associated type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplTypeDef {
    pub id: NodeId,
    pub name: String,
    pub ty: TypeExpr,
}

// ==================== TYPE ALIAS ====================

/// Type alias definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeAliasDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    pub generics: Generics,
    pub ty: TypeExpr,
    pub span: Span,
}

// ==================== EFFECTS ====================

/// Effect definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    pub generics: Generics,
    pub operations: Vec<EffectOpDef>,
    pub span: Span,
}

/// Effect operation definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectOpDef {
    pub id: NodeId,
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
}

/// Effect reference in function signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectRef {
    pub id: NodeId,
    pub name: Path,
    pub args: Vec<TypeExpr>,
}

/// Handler definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    pub generics: Generics,
    pub effect: Path,
    pub cases: Vec<HandlerCase>,
    pub span: Span,
}

/// Handler case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerCase {
    pub id: NodeId,
    pub name: String,
    pub params: Vec<Param>,
    pub body: Expr,
}

// ==================== IMPORTS & EXTERN ====================

/// Import definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportDef {
    pub id: NodeId,
    /// The module path being imported (e.g., `std::collections`)
    pub path: Path,
    /// Selective imports: None = import entire module, Some([]) = import nothing (invalid)
    /// Examples: `use foo::{A, B}` -> Some([A, B]), `use foo::*` -> Some([*])
    pub items: Option<Vec<ImportItem>>,
    /// Whether this is a re-export (`pub use`)
    pub is_reexport: bool,
    pub span: Span,
}

/// A single import item with optional rename
/// Examples: `A`, `A as B`, `*`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportItem {
    /// The name being imported (or "*" for glob)
    pub name: String,
    /// Optional alias: `use foo::Bar as Baz`
    pub alias: Option<String>,
    /// Whether this is a glob import (`*`)
    pub is_glob: bool,
}

/// Export definition
/// Syntax: `export { Name1, Name2, ... };` or `export Name;`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportDef {
    pub id: NodeId,
    pub names: Vec<String>,
    pub span: Span,
}

// ==================== ODE/PDE DEFINITIONS ====================

/// ODE (Ordinary Differential Equation) definition
/// Syntax:
/// ```d
/// ode LotkaVolterra {
///     params: { alpha: f64, beta: f64, gamma: f64, delta: f64 }
///     state: { prey: f64, predator: f64 }
///
///     d(prey)/dt = alpha * prey - beta * prey * predator
///     d(predator)/dt = delta * prey * predator - gamma * predator
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OdeDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    /// Parameters block: `params: { name: type, ... }`
    pub params: Vec<OdeParam>,
    /// State variables block: `state: { name: type, ... }`
    pub state: Vec<OdeStateVar>,
    /// Differential equations: `d(var)/dt = expr`
    pub equations: Vec<OdeEquation>,
    pub span: Span,
}

/// ODE parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OdeParam {
    pub id: NodeId,
    pub name: String,
    pub ty: TypeExpr,
    pub default: Option<Expr>,
    pub span: Span,
}

/// ODE state variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OdeStateVar {
    pub id: NodeId,
    pub name: String,
    pub ty: TypeExpr,
    pub span: Span,
}

/// ODE differential equation: `d(var)/dt = expr`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OdeEquation {
    pub id: NodeId,
    /// The state variable being differentiated
    pub variable: String,
    /// The right-hand side expression
    pub rhs: Expr,
    pub span: Span,
}

/// PDE (Partial Differential Equation) definition
/// Syntax:
/// ```d
/// pde HeatEquation {
///     params: { alpha: f64 }
///     domain: [0, 1] x [0, 1]
///
///     ∂u/∂t = alpha * (∂²u/∂x² + ∂²u/∂y²)
///
///     boundary: {
///         x = 0: u = 0,
///         x = 1: u = 0,
///         y = 0: ∂u/∂n = 0,
///         y = 1: u = sin(pi * x)
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdeDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    /// Parameters block
    pub params: Vec<OdeParam>,
    /// Domain specification
    pub domain: PdeDomain,
    /// The PDE equation
    pub equation: PdeEquation,
    /// Boundary conditions
    pub boundary_conditions: Vec<BoundaryConditionDef>,
    /// Optional initial condition
    pub initial_condition: Option<Expr>,
    pub span: Span,
}

/// PDE domain specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdeDomain {
    pub id: NodeId,
    /// Spatial dimensions: [(name, min, max), ...]
    pub dimensions: Vec<PdeDimension>,
    pub span: Span,
}

/// Single dimension in PDE domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdeDimension {
    pub name: String,
    pub min: Expr,
    pub max: Expr,
}

/// PDE equation: `∂u/∂t = ...` or `∂²u/∂t² = ...`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdeEquation {
    pub id: NodeId,
    /// Variable name (usually "u")
    pub variable: String,
    /// Time derivative order (1 for parabolic, 2 for hyperbolic)
    pub time_order: u32,
    /// Right-hand side expression
    pub rhs: Expr,
    pub span: Span,
}

/// Boundary condition definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryConditionDef {
    pub id: NodeId,
    /// Which boundary (e.g., "x = 0", "y = 1")
    pub boundary: BoundarySpec,
    /// The condition type and value
    pub condition: BoundaryConditionType,
    pub span: Span,
}

/// Boundary specification: which edge/face
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundarySpec {
    /// Variable name (x, y, z, etc.)
    pub variable: String,
    /// Boundary value (0 or 1 for min/max)
    pub value: Expr,
}

/// Type of boundary condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryConditionType {
    /// Dirichlet: u = value
    Dirichlet(Expr),
    /// Neumann: ∂u/∂n = value
    Neumann(Expr),
    /// Robin: a*u + b*∂u/∂n = value
    Robin { a: Expr, b: Expr, value: Expr },
    /// Periodic boundary
    Periodic,
}

// ==================== CAUSAL MODEL ====================

/// Causal model definition for causal inference
/// Syntax:
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalModelDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    /// Node declarations: `nodes: [A, B, C]`
    pub nodes: Vec<CausalNode>,
    /// Edge declarations: `A -> B`
    pub edges: Vec<CausalEdge>,
    /// Optional structural equations: `equations: { X = expr, ... }`
    pub equations: Vec<CausalEquation>,
    pub span: Span,
}

/// Causal DAG node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalNode {
    pub id: NodeId,
    pub name: String,
    /// Optional type annotation
    pub ty: Option<TypeExpr>,
    pub span: Span,
}

/// Causal DAG edge: `A -> B`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    pub id: NodeId,
    /// Source node name
    pub from: String,
    /// Target node name
    pub to: String,
    pub span: Span,
}

/// Structural causal equation: `X = expr`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEquation {
    pub id: NodeId,
    /// Variable being defined
    pub variable: String,
    /// Right-hand side expression
    pub rhs: Expr,
    pub span: Span,
}

// ==================== ONTOLOGY ====================

/// Ontology import definition
/// Syntax: `ontology <prefix> from "<url>";`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyImportDef {
    pub id: NodeId,
    /// The prefix/namespace to use (e.g., "chebi", "go", "hp")
    pub prefix: String,
    /// The source URL or path
    pub source: String,
    /// Optional alias
    pub alias: Option<String>,
    pub span: Span,
}

/// Ontology alignment declaration
/// Syntax: `align <type1> ~ <type2> with distance <value>;`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignDef {
    pub id: NodeId,
    /// First type in the alignment
    pub type1: OntologyTermRef,
    /// Second type in the alignment
    pub type2: OntologyTermRef,
    /// The semantic distance between the types
    pub distance: f64,
    pub span: Span,
}

/// Reference to an ontology term
/// Syntax: `prefix:term` (e.g., `chebi:15365`, `chebi:drug`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyTermRef {
    pub id: NodeId,
    /// Ontology prefix (e.g., "chebi", "go")
    pub prefix: String,
    /// Term identifier or name (e.g., "15365", "drug")
    pub term: String,
    pub span: Span,
}

impl OntologyTermRef {
    /// Create a new ontology term reference
    pub fn new(prefix: String, term: String, span: Span) -> Self {
        Self {
            id: NodeId::dummy(),
            prefix,
            term,
            span,
        }
    }

    /// Get the full IRI-style representation
    pub fn to_iri(&self) -> String {
        format!("{}:{}", self.prefix, self.term)
    }
}

/// ABI specification for FFI
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Abi {
    /// C ABI (default for extern)
    C,
    /// C ABI with unwind support
    CUnwind,
    /// Rust ABI (default for normal functions)
    Rust,
    /// System ABI (stdcall on Windows, C elsewhere)
    System,
    /// System ABI with unwind support
    SystemUnwind,
    /// x86 stdcall
    Stdcall,
    /// x86 stdcall with unwind support
    StdcallUnwind,
    /// x86 fastcall
    Fastcall,
    /// x86 fastcall with unwind support
    FastcallUnwind,
    /// x86 cdecl
    Cdecl,
    /// Arm AAPCS
    Aapcs,
    /// Win64 ABI
    Win64,
    /// SysV64 ABI
    SysV64,
    /// Platform intrinsic
    PlatformIntrinsic,
    /// Unknown ABI (for forward compatibility)
    Unknown(String),
}

impl Default for Abi {
    fn default() -> Self {
        Abi::Rust
    }
}

impl Abi {
    /// Parse an ABI string
    pub fn from_str(s: &str) -> Self {
        match s {
            "C" => Abi::C,
            "C-unwind" => Abi::CUnwind,
            "Rust" => Abi::Rust,
            "system" => Abi::System,
            "system-unwind" => Abi::SystemUnwind,
            "stdcall" => Abi::Stdcall,
            "stdcall-unwind" => Abi::StdcallUnwind,
            "fastcall" => Abi::Fastcall,
            "fastcall-unwind" => Abi::FastcallUnwind,
            "cdecl" => Abi::Cdecl,
            "aapcs" => Abi::Aapcs,
            "win64" => Abi::Win64,
            "sysv64" => Abi::SysV64,
            "platform-intrinsic" => Abi::PlatformIntrinsic,
            other => Abi::Unknown(other.to_string()),
        }
    }

    /// Get the ABI as a string
    pub fn as_str(&self) -> &str {
        match self {
            Abi::C => "C",
            Abi::CUnwind => "C-unwind",
            Abi::Rust => "Rust",
            Abi::System => "system",
            Abi::SystemUnwind => "system-unwind",
            Abi::Stdcall => "stdcall",
            Abi::StdcallUnwind => "stdcall-unwind",
            Abi::Fastcall => "fastcall",
            Abi::FastcallUnwind => "fastcall-unwind",
            Abi::Cdecl => "cdecl",
            Abi::Aapcs => "aapcs",
            Abi::Win64 => "win64",
            Abi::SysV64 => "sysv64",
            Abi::PlatformIntrinsic => "platform-intrinsic",
            Abi::Unknown(s) => s,
        }
    }

    /// Check if this ABI supports unwinding
    pub fn supports_unwind(&self) -> bool {
        matches!(
            self,
            Abi::CUnwind | Abi::SystemUnwind | Abi::StdcallUnwind | Abi::FastcallUnwind | Abi::Rust
        )
    }
}

impl std::fmt::Display for Abi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Extern block containing foreign declarations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternBlock {
    pub id: NodeId,
    pub abi: Abi,
    pub items: Vec<ExternItem>,
    pub span: Span,
}

/// Item in an extern block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExternItem {
    /// Foreign function declaration
    Fn(ExternFn),
    /// Foreign static variable
    Static(ExternStatic),
    /// Foreign type (opaque)
    Type(ExternType),
}

/// Extern function declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternFn {
    pub id: NodeId,
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub is_variadic: bool,
    pub link_name: Option<String>,
    pub span: Span,
}

/// Extern static variable declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternStatic {
    pub id: NodeId,
    pub name: String,
    pub ty: TypeExpr,
    pub is_mut: bool,
    pub link_name: Option<String>,
    pub span: Span,
}

/// Extern opaque type declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternType {
    pub id: NodeId,
    pub name: String,
    pub span: Span,
}

/// Representation attribute for FFI types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Repr {
    /// Default D representation
    D,
    /// C-compatible representation
    C,
    /// Transparent (single-field newtype)
    Transparent,
    /// Packed representation (no padding)
    Packed,
    /// Specific alignment
    Align(usize),
    /// Integer representation for enums
    Int(IntRepr),
}

/// Integer representation for enums
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntRepr {
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
}

/// Calling convention for function pointers
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallingConvention {
    /// Default D calling convention
    D,
    /// C calling convention
    C,
    /// System calling convention
    System,
    /// stdcall (Windows)
    Stdcall,
    /// fastcall (Windows)
    Fastcall,
    /// cdecl
    Cdecl,
}

// ==================== GLOBALS ====================

/// Global variable/constant definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub is_const: bool,
    pub is_mut: bool,
    pub pattern: Pattern,
    pub ty: Option<TypeExpr>,
    pub value: Expr,
    pub span: Span,
}

// ==================== GENERICS ====================

/// Generic parameters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Generics {
    pub params: Vec<GenericParam>,
}

/// Generic parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenericParam {
    Type {
        name: String,
        bounds: Vec<Path>,
        default: Option<TypeExpr>,
    },
    Const {
        name: String,
        ty: TypeExpr,
    },
}

/// Where predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WherePredicate {
    pub ty: TypeExpr,
    pub bounds: Vec<Path>,
}

// ==================== TYPES ====================

/// Type expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeExpr {
    /// Unit type ()
    Unit,
    /// Self type (in traits and impls)
    SelfType,
    /// Named type: Path<Args>
    Named {
        path: Path,
        args: Vec<TypeExpr>,
        unit: Option<String>,
    },
    /// Reference type: &T or &!T (mutable)
    Reference { mutable: bool, inner: Box<TypeExpr> },
    /// Raw pointer type: *const T or *mut T (for FFI)
    RawPointer { mutable: bool, inner: Box<TypeExpr> },
    /// Array type: [T] or [T; N]
    Array {
        element: Box<TypeExpr>,
        size: Option<Box<Expr>>,
    },
    /// Tuple type: (T1, T2, ...)
    Tuple(Vec<TypeExpr>),
    /// Function type: Fn(A) -> B
    Function {
        params: Vec<TypeExpr>,
        return_type: Box<TypeExpr>,
        effects: Vec<EffectRef>,
    },
    /// Infer type: _
    Infer,

    // ==================== SOUNIO EPISTEMIC TYPES ====================
    /// Knowledge type: Knowledge[T, ε < 0.05, Valid(duration), Derived]
    /// The core epistemic type tracking uncertainty, validity, and provenance
    Knowledge {
        /// The underlying value type (τ)
        value_type: Box<TypeExpr>,
        /// Uncertainty bound (ε)
        epsilon: Option<EpsilonBound>,
        /// Temporal validity condition (δ)
        validity: Option<ValidityCondition>,
        /// Data provenance marker (Φ)
        provenance: Option<ProvenanceMarker>,
    },

    /// Quantity type: Quantity[f64, meters] or f64@kg
    /// Value with physical units for dimensional analysis
    Quantity {
        /// The numeric type (f32, f64, i32, etc.)
        numeric_type: Box<TypeExpr>,
        /// The physical unit expression
        unit: UnitExpr,
    },

    /// Tensor type: Tensor[f32, (batch, channels, height, width)]
    /// Multi-dimensional array with named dimensions
    Tensor {
        /// Element type
        element_type: Box<TypeExpr>,
        /// Shape dimensions (can be expressions or named dimensions)
        shape: Vec<TensorDim>,
    },

    /// Tile type for GPU tile programming: tile<f16, 16, 16>
    /// Cooperative thread group view of a matrix tile in shared memory
    Tile {
        /// Element type (f32, f16, bf16, f8e4m3, f8e5m2, f4)
        element_type: Box<TypeExpr>,
        /// Tile height (static, power of 2, ≤64)
        tile_m: u32,
        /// Tile width (static, power of 2, ≤64)
        tile_n: u32,
        /// Optional memory layout ("row_major", "col_major", "swizzled")
        layout: Option<String>,
    },

    /// Ontology type: OntologyTerm[SNOMED:12345]
    /// Reference to an ontology term for semantic interoperability
    Ontology {
        /// Ontology namespace (SNOMED, ICD10, NIDM, etc.)
        ontology: String,
        /// Optional specific term within the ontology
        term: Option<String>,
    },

    /// Linear/affine type annotation: T @ linear
    /// For GPU memory safety and resource tracking
    Linear {
        inner: Box<TypeExpr>,
        linearity: LinearityKind,
    },

    /// Effect row type: T ! {IO, GPU, Random}
    /// Type annotated with computational effects
    Effected {
        inner: Box<TypeExpr>,
        effects: EffectRow,
    },

    /// Refinement type: { x: T | predicate }
    /// A base type refined by a logical predicate
    /// Example: `{ x: i32 | x > 0 }` for positive integers
    Refinement {
        /// The refinement variable name (e.g., "x")
        var: String,
        /// The base type (e.g., i32)
        base_type: Box<TypeExpr>,
        /// The predicate expression constraining the value
        predicate: Box<Expr>,
    },
}

// ==================== EPISTEMIC TYPE COMPONENTS ====================

/// Uncertainty bound: ε < 0.05, ε = σ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpsilonBound {
    /// Comparison operator: "<", "<=", "=", ">", ">="
    pub operator: ComparisonOp,
    /// The bound value (can be a literal or expression)
    pub value: Box<Expr>,
}

/// Comparison operators for epsilon bounds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    Lt,
    Le,
    Eq,
    Ge,
    Gt,
}

/// Validity condition: Valid(duration), ValidUntil(date), ValidWhile(condition)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidityCondition {
    /// Kind of validity: "Valid", "ValidUntil", "ValidWhile"
    pub kind: ValidityKind,
    /// The condition expression
    pub condition: Box<Expr>,
}

/// Kinds of temporal validity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidityKind {
    /// Valid for a duration
    Valid,
    /// Valid until a specific time
    ValidUntil,
    /// Valid while a condition holds
    ValidWhile,
}

/// Provenance marker: Derived, Source(name), Computed, Literature(citation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceMarker {
    /// Kind of provenance
    pub kind: ProvenanceKind,
    /// Optional source reference
    pub source: Option<Box<Expr>>,
}

/// Kinds of data provenance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceKind {
    /// Derived from other data
    Derived,
    /// Primary source data
    Source,
    /// Computed/calculated value
    Computed,
    /// From published literature
    Literature,
    /// Experimentally measured
    Measured,
    /// User-provided input
    Input,
}

/// Physical unit expression: meters, kg*m/s^2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitExpr {
    /// Base units with exponents: [(unit_name, exponent), ...]
    /// e.g., kg*m/s^2 = [("kg", 1), ("m", 1), ("s", -2)]
    pub base_units: Vec<(String, i32)>,
}

impl UnitExpr {
    /// Create a simple unit
    pub fn simple(name: &str) -> Self {
        Self {
            base_units: vec![(name.to_string(), 1)],
        }
    }

    /// Create a dimensionless unit
    pub fn dimensionless() -> Self {
        Self { base_units: vec![] }
    }
}

/// Tensor dimension (can be named or sized)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorDim {
    /// Named dimension: batch, channels, height, width
    Named(String),
    /// Fixed size dimension
    Fixed(usize),
    /// Dynamic/inferred dimension
    Dynamic,
    /// Expression-based dimension
    Expr(Box<Expr>),
}

/// Linearity kind for resource types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearityKind {
    /// Normal unrestricted type
    Unrestricted,
    /// Must be used exactly once
    Linear,
    /// Must be used at most once
    Affine,
    /// Must be used at least once
    Relevant,
}

/// Effect row: {IO, GPU, Random, ...}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectRow {
    /// List of effects
    pub effects: Vec<String>,
    /// Whether this is an open row (can have more effects)
    pub is_open: bool,
}

// ==================== EXPRESSIONS ====================

/// Expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    /// Literal value
    Literal { id: NodeId, value: Literal },
    /// Path reference
    Path { id: NodeId, path: Path },
    /// Binary operation
    Binary {
        id: NodeId,
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Unary operation
    Unary {
        id: NodeId,
        op: UnaryOp,
        expr: Box<Expr>,
    },
    /// Function call
    Call {
        id: NodeId,
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    /// Method call
    MethodCall {
        id: NodeId,
        receiver: Box<Expr>,
        method: String,
        args: Vec<Expr>,
    },
    /// Field access
    Field {
        id: NodeId,
        base: Box<Expr>,
        field: String,
    },
    /// Tuple field access
    TupleField {
        id: NodeId,
        base: Box<Expr>,
        index: usize,
    },
    /// Index operation
    Index {
        id: NodeId,
        base: Box<Expr>,
        index: Box<Expr>,
    },
    /// Type cast
    Cast {
        id: NodeId,
        expr: Box<Expr>,
        ty: TypeExpr,
    },
    /// Block expression
    Block { id: NodeId, block: Block },
    /// If expression
    If {
        id: NodeId,
        condition: Box<Expr>,
        then_branch: Block,
        else_branch: Option<Box<Expr>>,
    },
    /// Match expression
    Match {
        id: NodeId,
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    /// Loop expression
    Loop { id: NodeId, body: Block },
    /// While loop
    While {
        id: NodeId,
        condition: Box<Expr>,
        body: Block,
    },
    /// For loop
    For {
        id: NodeId,
        pattern: Pattern,
        iter: Box<Expr>,
        body: Block,
    },
    /// Return expression
    Return {
        id: NodeId,
        value: Option<Box<Expr>>,
    },
    /// Break expression
    Break {
        id: NodeId,
        value: Option<Box<Expr>>,
    },
    /// Continue expression
    Continue { id: NodeId },
    /// Closure expression
    Closure {
        id: NodeId,
        params: Vec<(String, Option<TypeExpr>)>,
        return_type: Option<TypeExpr>,
        body: Box<Expr>,
    },
    /// Tuple expression
    Tuple { id: NodeId, elements: Vec<Expr> },
    /// Array expression
    Array { id: NodeId, elements: Vec<Expr> },
    /// Range expression (start..end or start..=end)
    Range {
        id: NodeId,
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        inclusive: bool,
    },
    /// Struct literal
    StructLit {
        id: NodeId,
        path: Path,
        fields: Vec<(String, Expr)>,
    },
    /// Try expression (?)
    Try { id: NodeId, expr: Box<Expr> },
    /// Perform effect operation
    Perform {
        id: NodeId,
        effect: Path,
        op: String,
        args: Vec<Expr>,
    },
    /// Handle effect
    Handle {
        id: NodeId,
        expr: Box<Expr>,
        handler: Path,
    },
    /// Sample from distribution
    Sample { id: NodeId, distribution: Box<Expr> },
    /// Await async expression
    Await { id: NodeId, expr: Box<Expr> },
    /// Async block: async { ... }
    AsyncBlock { id: NodeId, block: Block },
    /// Async closure: async |x| { ... }
    AsyncClosure {
        id: NodeId,
        params: Vec<(String, Option<TypeExpr>)>,
        return_type: Option<TypeExpr>,
        body: Box<Expr>,
    },
    /// Spawn async task: spawn { ... }
    Spawn { id: NodeId, expr: Box<Expr> },
    /// Select expression for waiting on multiple futures
    Select { id: NodeId, arms: Vec<SelectArm> },
    /// Join expression for concurrent execution
    Join { id: NodeId, futures: Vec<Expr> },
    /// Macro invocation
    MacroInvocation(MacroInvocation),

    // ==================== SOUNIO ONTOLOGY EXPRESSIONS ====================
    /// Ontology term literal: prefix:term (e.g., drugbank:DB00945, chebi:15365)
    OntologyTerm {
        id: NodeId,
        /// Ontology prefix (e.g., "drugbank", "chebi")
        ontology: String,
        /// Term identifier (e.g., "DB00945", "15365")
        term: String,
    },

    // ==================== SOUNIO EPISTEMIC EXPRESSIONS ====================
    /// Causal do expression: do(X = 1)
    /// Represents intervention in causal inference (Pearl's do-calculus)
    Do {
        id: NodeId,
        /// List of interventions: [(variable, value), ...]
        interventions: Vec<(String, Box<Expr>)>,
    },

    /// Counterfactual expression: counterfactual { factual; do(X=1); outcome }
    /// Three-step counterfactual computation (abduction, action, prediction)
    Counterfactual {
        id: NodeId,
        /// The factual observation
        factual: Box<Expr>,
        /// The intervention to apply
        intervention: Box<Expr>,
        /// The outcome query
        outcome: Box<Expr>,
    },

    /// Knowledge construction: Knowledge::new(value, epsilon, validity, provenance)
    KnowledgeExpr {
        id: NodeId,
        /// The underlying value
        value: Box<Expr>,
        /// Optional uncertainty bound
        epsilon: Option<Box<Expr>>,
        /// Optional validity condition
        validity: Option<Box<Expr>>,
        /// Optional provenance marker
        provenance: Option<Box<Expr>>,
    },

    /// Uncertainty propagation: x ± σ or x.with_uncertainty(σ)
    Uncertain {
        id: NodeId,
        /// The central value
        value: Box<Expr>,
        /// The uncertainty/standard deviation
        uncertainty: Box<Expr>,
    },

    /// GPU-annotated expression: expr @ gpu.epistemic
    GpuAnnotated {
        id: NodeId,
        /// The inner expression
        expr: Box<Expr>,
        /// GPU annotation kind
        annotation: GpuAnnotation,
    },

    /// Observe expression for probabilistic programming: observe(data ~ distribution)
    Observe {
        id: NodeId,
        /// The observed data
        data: Box<Expr>,
        /// The distribution it's drawn from
        distribution: Box<Expr>,
    },

    /// Query expression: P(Y | X, do(Z))
    /// Probabilistic query with optional conditioning and intervention
    Query {
        id: NodeId,
        /// The target variable/expression
        target: Box<Expr>,
        /// Conditioning variables
        given: Vec<Expr>,
        /// Interventions (do-expressions)
        interventions: Vec<(String, Box<Expr>)>,
    },
}

// ==================== GPU ANNOTATIONS ====================

/// GPU annotation: @gpu, @gpu.epistemic, @gpu.reduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAnnotation {
    /// Annotation kind
    pub kind: GpuAnnotationKind,
    /// Optional parameters
    pub params: Vec<(String, Expr)>,
}

/// Kinds of GPU annotations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuAnnotationKind {
    /// Basic GPU execution
    Gpu,
    /// Epistemic-aware GPU (uncertainty propagation)
    GpuEpistemic,
    /// GPU reduction operation
    GpuReduction,
    /// GPU parallel execution
    GpuParallel,
    /// GPU shared memory
    GpuShared,
    /// GPU device memory
    GpuDevice,
}

/// Literal values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Literal {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    Char(char),
    String(String),
    /// Integer with unit of measure (e.g., 500_mg)
    IntUnit(i64, String),
    /// Float with unit of measure (e.g., 10.5_mL)
    FloatUnit(f64, String),
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Logical
    And,
    Or,
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    // Scientific
    /// Plus-minus operator for uncertain values: `x +- 0.1`
    PlusMinus,
    // Collection operations
    /// Concatenation operator for arrays/slices: `a ++ b`
    Concat,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
    Ref,
    RefMut,
    Deref,
}

/// Match arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Box<Expr>>,
    pub body: Expr,
}

/// Select arm for async select expressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectArm {
    /// The future expression to wait on
    pub future: Expr,
    /// Pattern to bind the result
    pub pattern: Pattern,
    /// Optional guard condition
    pub guard: Option<Box<Expr>>,
    /// Body expression to execute when this arm matches
    pub body: Expr,
}

// ==================== STATEMENTS ====================

/// Statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stmt {
    /// Let binding
    Let {
        is_mut: bool,
        pattern: Pattern,
        ty: Option<TypeExpr>,
        value: Option<Expr>,
    },
    /// Expression statement
    Expr { expr: Expr, has_semi: bool },
    /// Assignment
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },
    /// Empty statement (;)
    Empty,
    /// Macro invocation
    MacroInvocation(MacroInvocation),
}

/// Assignment operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    ShlAssign,
    ShrAssign,
}

/// Block of statements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

// ==================== PATTERNS ====================

/// Pattern for matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Pattern {
    /// Wildcard pattern: _
    Wildcard,
    /// Literal pattern
    Literal(Literal),
    /// Variable binding
    Binding { name: String, mutable: bool },
    /// Tuple pattern: (p1, p2, ...)
    Tuple(Vec<Pattern>),
    /// Struct pattern: S { field: pattern, ... }
    Struct {
        path: Path,
        fields: Vec<(String, Pattern)>,
    },
    /// Enum variant pattern: E::V(p1, p2, ...)
    Enum {
        path: Path,
        patterns: Option<Vec<Pattern>>,
    },
    /// Or pattern: p1 | p2
    Or(Vec<Pattern>),
}

// ==================== MACROS ====================

/// Macro invocation (e.g., vec![1, 2, 3])
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroInvocation {
    pub id: NodeId,
    pub name: String,
    pub args: Vec<crate::macro_system::token_tree::TokenTree>,
    pub span: Span,
}

// ==================== PATHS ====================

/// Module identifier for tracking where definitions originate
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct ModuleId {
    /// The module path (e.g., ["std", "math"] for std::math)
    pub path: Vec<String>,
}

impl ModuleId {
    pub fn new(path: Vec<String>) -> Self {
        Self { path }
    }

    pub fn root() -> Self {
        Self { path: vec![] }
    }

    pub fn from_file_path(file_path: &std::path::Path) -> Self {
        let stem = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        Self {
            path: vec![stem.to_string()],
        }
    }

    pub fn is_root(&self) -> bool {
        self.path.is_empty()
    }

    pub fn join(&self, name: &str) -> Self {
        let mut path = self.path.clone();
        path.push(name.to_string());
        Self { path }
    }
}

impl std::fmt::Display for ModuleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.path.is_empty() {
            write!(f, "<root>")
        } else {
            write!(f, "{}", self.path.join("::"))
        }
    }
}

/// Path (e.g., std::io::Write)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Path {
    /// The path segments (e.g., ["std", "io", "Write"])
    pub segments: Vec<String>,
    /// The module where this path was written (for diagnostics)
    #[serde(default)]
    pub source_module: Option<ModuleId>,
    /// The module this path resolves to (set during name resolution)
    #[serde(default)]
    pub resolved_module: Option<ModuleId>,
}

impl Path {
    pub fn simple(name: &str) -> Self {
        Path {
            segments: vec![name.to_string()],
            source_module: None,
            resolved_module: None,
        }
    }

    pub fn qualified(module: Vec<String>, name: &str) -> Self {
        let mut segments = module;
        segments.push(name.to_string());
        Path {
            segments,
            source_module: None,
            resolved_module: None,
        }
    }

    pub fn with_source_module(mut self, module: ModuleId) -> Self {
        self.source_module = Some(module);
        self
    }

    pub fn with_resolved_module(mut self, module: ModuleId) -> Self {
        self.resolved_module = Some(module);
        self
    }

    pub fn is_simple(&self) -> bool {
        self.segments.len() == 1
    }

    pub fn is_qualified(&self) -> bool {
        self.segments.len() > 1
    }

    pub fn name(&self) -> Option<&str> {
        self.segments.last().map(|s| s.as_str())
    }

    /// Get the module prefix (all segments except the last)
    pub fn module_prefix(&self) -> Option<Vec<String>> {
        if self.segments.len() > 1 {
            Some(self.segments[..self.segments.len() - 1].to_vec())
        } else {
            None
        }
    }

    /// Check if this path starts with a given module prefix
    pub fn starts_with_module(&self, prefix: &[String]) -> bool {
        if prefix.len() >= self.segments.len() {
            return false;
        }
        self.segments[..prefix.len()] == *prefix
    }
}

impl Default for Path {
    fn default() -> Self {
        Path {
            segments: vec![],
            source_module: None,
            resolved_module: None,
        }
    }
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.segments.join("::"))
    }
}
