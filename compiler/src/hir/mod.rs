//! High-level Intermediate Representation (HIR)
//!
//! HIR is the typed AST produced by the type checker. It contains:
//! - Resolved types for all expressions
//! - Resolved names (no more path resolution needed)
//! - Desugared constructs
//! - Ownership and borrowing information

pub mod async_transform;

pub use async_transform::{
    AsyncStateKind, AsyncStateMachine, AsyncStateNode, AsyncTransformer, AwaitExpr, CapturedLocal,
    StateTransition, is_async_function, transform_async_functions,
};

use crate::ast::Abi;
use crate::common::NodeId;

/// HIR root
#[derive(Debug, Clone)]
pub struct Hir {
    pub items: Vec<HirItem>,
    /// Extern blocks (FFI imports) collected from the AST.
    ///
    /// Kept separate from `items` so most compiler passes can ignore FFI unless needed.
    pub externs: Vec<HirExternBlock>,
}

/// HIR item
#[derive(Debug, Clone)]
pub enum HirItem {
    Function(HirFn),
    Struct(HirStruct),
    Enum(HirEnum),
    Trait(HirTrait),
    Impl(HirImpl),
    TypeAlias(HirTypeAlias),
    Effect(HirEffect),
    Handler(HirHandler),
    Global(HirGlobal),
}

// ==================== EXTERN (FFI IMPORTS) ====================

/// HIR extern block (e.g., `extern "C" { fn ...; }`)
#[derive(Debug, Clone)]
pub struct HirExternBlock {
    pub id: NodeId,
    pub abi: Abi,
    pub functions: Vec<HirExternFn>,
}

/// HIR extern function declaration (no body)
#[derive(Debug, Clone)]
pub struct HirExternFn {
    pub id: NodeId,
    pub name: String,
    pub params: Vec<HirParam>,
    pub return_type: HirType,
    pub is_variadic: bool,
    pub link_name: Option<String>,
}

// ==================== FUNCTIONS ====================

/// HIR function
#[derive(Debug, Clone)]
pub struct HirFn {
    pub id: NodeId,
    pub name: String,
    pub ty: HirFnType,
    pub body: HirBlock,
    /// ABI for FFI functions (C, System, etc.)
    pub abi: Abi,
    /// Whether this function is exported (pub extern)
    pub is_exported: bool,
}

/// Function type in HIR
#[derive(Debug, Clone)]
pub struct HirFnType {
    pub params: Vec<HirParam>,
    pub return_type: Box<HirType>,
    pub effects: Vec<HirEffect>,
}

/// HIR parameter
#[derive(Debug, Clone)]
pub struct HirParam {
    pub id: NodeId,
    pub name: String,
    pub ty: HirType,
    pub is_mut: bool,
}

// ==================== TYPES ====================

/// HIR struct
#[derive(Debug, Clone)]
pub struct HirStruct {
    pub id: NodeId,
    pub name: String,
    pub fields: Vec<HirField>,
    pub is_linear: bool,
    pub is_affine: bool,
}

/// HIR field
#[derive(Debug, Clone)]
pub struct HirField {
    pub id: NodeId,
    pub name: String,
    pub ty: HirType,
}

/// HIR enum
#[derive(Debug, Clone)]
pub struct HirEnum {
    pub id: NodeId,
    pub name: String,
    pub variants: Vec<HirVariant>,
    pub is_linear: bool,
    pub is_affine: bool,
}

/// HIR variant
#[derive(Debug, Clone)]
pub struct HirVariant {
    pub id: NodeId,
    pub name: String,
    pub fields: Vec<HirType>,
}

/// HIR trait
#[derive(Debug, Clone)]
pub struct HirTrait {
    pub id: NodeId,
    pub name: String,
    pub methods: Vec<HirTraitMethod>,
}

/// HIR trait method
#[derive(Debug, Clone)]
pub struct HirTraitMethod {
    pub id: NodeId,
    pub name: String,
    pub ty: HirFnType,
    pub has_default: bool,
}

/// HIR impl
#[derive(Debug, Clone)]
pub struct HirImpl {
    pub id: NodeId,
    pub trait_ref: Option<String>,
    pub self_ty: HirType,
    pub methods: Vec<HirFn>,
}

/// HIR type alias
#[derive(Debug, Clone)]
pub struct HirTypeAlias {
    pub id: NodeId,
    pub name: String,
    pub ty: HirType,
}

/// HIR global
#[derive(Debug, Clone)]
pub struct HirGlobal {
    pub id: NodeId,
    pub name: String,
    pub ty: HirType,
    pub value: HirExpr,
    pub is_const: bool,
}

// ==================== EFFECTS ====================

/// HIR effect
#[derive(Debug, Clone)]
pub struct HirEffect {
    pub id: NodeId,
    pub name: String,
    pub operations: Vec<HirEffectOp>,
}

/// HIR effect operation
#[derive(Debug, Clone)]
pub struct HirEffectOp {
    pub id: NodeId,
    pub name: String,
    pub params: Vec<HirType>,
    pub return_type: HirType,
}

/// HIR handler
#[derive(Debug, Clone)]
pub struct HirHandler {
    pub id: NodeId,
    pub name: String,
    pub effect: String,
    pub cases: Vec<HirHandlerCase>,
}

/// HIR handler case
#[derive(Debug, Clone)]
pub struct HirHandlerCase {
    pub id: NodeId,
    pub op_name: String,
    pub params: Vec<String>,
    pub body: HirExpr,
}

// ==================== TYPES ====================

/// HIR type (fully resolved)
#[derive(Debug, Clone, PartialEq)]
pub enum HirType {
    /// Unit type
    Unit,
    /// Boolean
    Bool,
    /// Signed integers
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    /// Unsigned integers
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    /// Floating point
    F32,
    F64,
    /// Character
    Char,
    /// String (owned)
    String,

    // Linear algebra primitives
    /// 2D vector (2x f32)
    Vec2,
    /// 3D vector (3x f32)
    Vec3,
    /// 4D vector (4x f32)
    Vec4,
    /// 2x2 matrix (4x f32, column-major)
    Mat2,
    /// 3x3 matrix (9x f32, column-major)
    Mat3,
    /// 4x4 matrix (16x f32, column-major)
    Mat4,
    /// Quaternion (4x f32: x, y, z, w)
    Quat,

    // Automatic differentiation types
    /// Dual number for forward-mode autodiff (value: f64, derivative: f64)
    Dual,

    /// Reference
    Ref {
        mutable: bool,
        inner: Box<HirType>,
    },
    /// Raw pointer (for FFI)
    RawPointer {
        mutable: bool,
        inner: Box<HirType>,
    },
    /// Array
    Array {
        element: Box<HirType>,
        size: Option<usize>,
    },
    /// Tuple
    Tuple(Vec<HirType>),
    /// Named type (struct/enum/type alias)
    Named {
        name: String,
        args: Vec<HirType>,
    },
    /// Function type
    Fn {
        params: Vec<HirType>,
        return_type: Box<HirType>,
    },
    /// Type variable (for generics)
    Var(u32),
    /// Never type (for diverging expressions)
    Never,
    /// Error type (for error recovery)
    Error,

    // ==================== EPISTEMIC TYPES ====================
    /// Knowledge type: Knowledge[T, ε >= bound]
    Knowledge {
        inner: Box<HirType>,
        epsilon_bound: Option<f64>,
        provenance: Option<HirProvenanceConstraint>,
    },
    /// Physical quantity with unit: Quantity[f64, kg*m/s^2]
    Quantity {
        numeric: Box<HirType>,
        unit: HirUnit,
    },
    /// Tensor with named dimensions: Tensor[f32, (batch, channels, height, width)]
    Tensor {
        element: Box<HirType>,
        dims: Vec<HirTensorDim>,
    },
    /// Ontology term: OntologyTerm[SNOMED:12345]
    Ontology {
        namespace: String,
        term: String,
    },

    // ==================== ASYNC TYPES ====================
    /// Future type: Future<T> represents an async computation that will produce T
    Future {
        /// The output type of the future
        output: Box<HirType>,
    },
}

/// HIR unit expression for physical quantities
#[derive(Debug, Clone, PartialEq)]
pub struct HirUnit {
    /// Numerator units with exponents: [(kg, 1), (m, 1)]
    pub numerator: Vec<(String, i32)>,
    /// Denominator units with exponents: [(s, 2)]
    pub denominator: Vec<(String, i32)>,
}

impl HirUnit {
    pub fn dimensionless() -> Self {
        Self {
            numerator: Vec::new(),
            denominator: Vec::new(),
        }
    }

    pub fn simple(unit: &str) -> Self {
        Self {
            numerator: vec![(unit.to_string(), 1)],
            denominator: Vec::new(),
        }
    }

    /// Check if two units are compatible (same dimensions) for addition/subtraction
    pub fn is_compatible(&self, other: &HirUnit) -> bool {
        // Normalize both units to maps for comparison
        let self_dims = self.to_dimension_map();
        let other_dims = other.to_dimension_map();
        self_dims == other_dims
    }

    /// Convert to a dimension map (unit -> exponent)
    fn to_dimension_map(&self) -> std::collections::HashMap<String, i32> {
        let mut dims = std::collections::HashMap::new();
        for (unit, exp) in &self.numerator {
            *dims.entry(unit.clone()).or_insert(0) += exp;
        }
        for (unit, exp) in &self.denominator {
            *dims.entry(unit.clone()).or_insert(0) -= exp;
        }
        // Remove zero exponents
        dims.retain(|_, v| *v != 0);
        dims
    }

    /// Multiply two units (for multiplication operations)
    pub fn multiply(&self, other: &HirUnit) -> HirUnit {
        let mut dims = self.to_dimension_map();
        for (unit, exp) in other.to_dimension_map() {
            *dims.entry(unit).or_insert(0) += exp;
        }
        dims.retain(|_, v| *v != 0);
        Self::from_dimension_map(dims)
    }

    /// Divide two units (for division operations)
    pub fn divide(&self, other: &HirUnit) -> HirUnit {
        let mut dims = self.to_dimension_map();
        for (unit, exp) in other.to_dimension_map() {
            *dims.entry(unit).or_insert(0) -= exp;
        }
        dims.retain(|_, v| *v != 0);
        Self::from_dimension_map(dims)
    }

    /// Create from dimension map
    fn from_dimension_map(dims: std::collections::HashMap<String, i32>) -> HirUnit {
        let mut numerator = Vec::new();
        let mut denominator = Vec::new();
        for (unit, exp) in dims {
            if exp > 0 {
                numerator.push((unit, exp));
            } else if exp < 0 {
                denominator.push((unit, -exp));
            }
        }
        // Sort for consistent ordering
        numerator.sort_by(|a, b| a.0.cmp(&b.0));
        denominator.sort_by(|a, b| a.0.cmp(&b.0));
        HirUnit {
            numerator,
            denominator,
        }
    }

    /// Check if this is a dimensionless unit
    pub fn is_dimensionless(&self) -> bool {
        self.numerator.is_empty() && self.denominator.is_empty()
    }

    /// Format unit for display
    pub fn format(&self) -> String {
        if self.is_dimensionless() {
            return "1".to_string();
        }
        let format_parts = |parts: &[(String, i32)]| -> String {
            parts
                .iter()
                .map(|(u, e)| {
                    if *e == 1 {
                        u.clone()
                    } else {
                        format!("{}^{}", u, e)
                    }
                })
                .collect::<Vec<_>>()
                .join("*")
        };
        let num = format_parts(&self.numerator);
        let den = format_parts(&self.denominator);
        if self.denominator.is_empty() {
            num
        } else if self.numerator.is_empty() {
            format!("1/{}", den)
        } else {
            format!("{}/{}", num, den)
        }
    }
}

/// HIR tensor dimension
#[derive(Debug, Clone, PartialEq)]
pub enum HirTensorDim {
    /// Named dimension (e.g., "batch", "channels")
    Named(String),
    /// Fixed size dimension
    Fixed(usize),
    /// Dynamic dimension (size known at runtime)
    Dynamic,
}

/// HIR provenance constraint
#[derive(Debug, Clone, PartialEq)]
pub enum HirProvenanceConstraint {
    /// Must come from specific source
    FromSource(String),
    /// Must be derived from listed sources
    DerivedFrom(Vec<String>),
    /// Must be user input
    UserInput,
    /// Must be peer reviewed
    PeerReviewed,
    /// Must comply with regulatory standard
    RegulatoryCompliant(String),
}

impl HirType {
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            HirType::Unit
                | HirType::Bool
                | HirType::I8
                | HirType::I16
                | HirType::I32
                | HirType::I64
                | HirType::I128
                | HirType::Isize
                | HirType::U8
                | HirType::U16
                | HirType::U32
                | HirType::U64
                | HirType::U128
                | HirType::Usize
                | HirType::F32
                | HirType::F64
                | HirType::Char
        )
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            HirType::I8
                | HirType::I16
                | HirType::I32
                | HirType::I64
                | HirType::I128
                | HirType::Isize
                | HirType::U8
                | HirType::U16
                | HirType::U32
                | HirType::U64
                | HirType::U128
                | HirType::Usize
                | HirType::F32
                | HirType::F64
        )
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            HirType::I8
                | HirType::I16
                | HirType::I32
                | HirType::I64
                | HirType::I128
                | HirType::Isize
                | HirType::U8
                | HirType::U16
                | HirType::U32
                | HirType::U64
                | HirType::U128
                | HirType::Usize
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, HirType::F32 | HirType::F64)
    }
}

// ==================== EXPRESSIONS ====================

/// HIR expression (with type information)
#[derive(Debug, Clone)]
pub struct HirExpr {
    pub id: NodeId,
    pub kind: HirExprKind,
    pub ty: HirType,
}

/// HIR expression kind
#[derive(Debug, Clone)]
pub enum HirExprKind {
    /// Literal value
    Literal(HirLiteral),
    /// Local variable
    Local(String),
    /// Global variable
    Global(String),
    /// Binary operation
    Binary {
        op: HirBinaryOp,
        left: Box<HirExpr>,
        right: Box<HirExpr>,
    },
    /// Unary operation
    Unary { op: HirUnaryOp, expr: Box<HirExpr> },
    /// Function call
    Call {
        func: Box<HirExpr>,
        args: Vec<HirExpr>,
    },
    /// Method call
    MethodCall {
        receiver: Box<HirExpr>,
        method: String,
        args: Vec<HirExpr>,
    },
    /// Field access
    Field { base: Box<HirExpr>, field: String },
    /// Tuple field access
    TupleField { base: Box<HirExpr>, index: usize },
    /// Index operation
    Index {
        base: Box<HirExpr>,
        index: Box<HirExpr>,
    },
    /// Type cast
    Cast { expr: Box<HirExpr>, target: HirType },
    /// Block
    Block(HirBlock),
    /// If expression
    If {
        condition: Box<HirExpr>,
        then_branch: HirBlock,
        else_branch: Option<Box<HirExpr>>,
    },
    /// Match expression
    Match {
        scrutinee: Box<HirExpr>,
        arms: Vec<HirMatchArm>,
    },
    /// Infinite loop
    Loop(HirBlock),
    /// While loop - condition re-evaluated each iteration
    While {
        condition: Box<HirExpr>,
        body: HirBlock,
    },
    /// Return
    Return(Option<Box<HirExpr>>),
    /// Break
    Break(Option<Box<HirExpr>>),
    /// Continue
    Continue,
    /// Closure
    Closure {
        params: Vec<HirParam>,
        body: Box<HirExpr>,
    },
    /// Tuple
    Tuple(Vec<HirExpr>),
    /// Array
    Array(Vec<HirExpr>),
    /// Range expression (start..end or start..=end)
    Range {
        start: Option<Box<HirExpr>>,
        end: Option<Box<HirExpr>>,
        inclusive: bool,
    },
    /// Struct literal
    Struct {
        name: String,
        fields: Vec<(String, HirExpr)>,
    },
    /// Enum variant constructor
    Variant {
        enum_name: String,
        variant: String,
        fields: Vec<HirExpr>,
    },
    /// Reference
    Ref { mutable: bool, expr: Box<HirExpr> },
    /// Dereference
    Deref(Box<HirExpr>),
    /// Effect operation
    Perform {
        effect: String,
        op: String,
        args: Vec<HirExpr>,
    },
    /// Handle effect
    Handle { expr: Box<HirExpr>, handler: String },
    /// Sample from distribution
    Sample(Box<HirExpr>),

    // ==================== EPISTEMIC EXPRESSIONS ====================
    /// Knowledge construction: Knowledge::new(value, epsilon, validity, provenance)
    Knowledge {
        value: Box<HirExpr>,
        epsilon: Box<HirExpr>,
        validity: Option<Box<HirExpr>>,
        provenance: Option<HirProvenance>,
    },
    /// Do intervention: do(X = value)
    Do {
        variable: String,
        value: Box<HirExpr>,
    },
    /// Counterfactual expression: counterfactual { factual; intervention; outcome }
    Counterfactual {
        factual: Box<HirExpr>,
        intervention: Box<HirExpr>,
        outcome: Box<HirExpr>,
    },
    /// Query expression: P(target | given, do(interventions))
    Query {
        target: Box<HirExpr>,
        given: Vec<HirExpr>,
        interventions: Vec<HirExpr>,
    },
    /// Observe expression: observe variable = value
    Observe {
        variable: String,
        value: Box<HirExpr>,
    },
    /// Extract epsilon (confidence) from Knowledge value
    EpsilonOf(Box<HirExpr>),
    /// Extract provenance from Knowledge value
    ProvenanceOf(Box<HirExpr>),
    /// Extract validity from Knowledge value
    ValidityOf(Box<HirExpr>),
    /// Unwrap Knowledge to get inner value
    Unwrap(Box<HirExpr>),

    // ==================== ONTOLOGY EXPRESSIONS ====================
    /// Ontology term literal: prefix:term (e.g., chebi:aspirin, drugbank:DB00945)
    OntologyTerm { namespace: String, term: String },

    // ==================== ASYNC EXPRESSIONS ====================
    /// Await expression: future.await
    Await {
        /// The future expression being awaited
        future: Box<HirExpr>,
    },
    /// Spawn expression: spawn { expr }
    Spawn {
        /// The expression to spawn as a new task
        expr: Box<HirExpr>,
    },
    /// Async block: async { ... }
    AsyncBlock {
        /// The block of statements to execute asynchronously
        body: HirBlock,
    },
    /// Join expression: join(future1, future2, ...)
    Join {
        /// The futures to wait on concurrently
        futures: Vec<HirExpr>,
    },
    /// Select expression: select { arm1, arm2, ... }
    Select {
        /// The select arms
        arms: Vec<HirSelectArm>,
    },
}

/// HIR provenance marker
#[derive(Debug, Clone, PartialEq)]
pub enum HirProvenance {
    /// Measured from sensor/instrument
    Measured { source: String },
    /// Derived from computation
    Derived { sources: Vec<String> },
    /// User input
    UserInput,
    /// From peer-reviewed source
    PeerReviewed { citation: String },
    /// Regulatory compliant
    RegulatoryCompliant { standard: String },
}

/// HIR literal
#[derive(Debug, Clone)]
pub enum HirLiteral {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    Char(char),
    String(String),
}

/// HIR binary operator
#[derive(Debug, Clone, Copy)]
pub enum HirBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    PlusMinus,
    Concat,
}

impl HirBinaryOp {
    /// Returns true if this is a comparison operator (returns bool)
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            HirBinaryOp::Eq
                | HirBinaryOp::Ne
                | HirBinaryOp::Lt
                | HirBinaryOp::Le
                | HirBinaryOp::Gt
                | HirBinaryOp::Ge
        )
    }
}

/// HIR unary operator
#[derive(Debug, Clone, Copy)]
pub enum HirUnaryOp {
    Neg,
    Not,
    Ref,
    RefMut,
    Deref,
}

/// HIR match arm
#[derive(Debug, Clone)]
pub struct HirMatchArm {
    pub pattern: HirPattern,
    pub guard: Option<Box<HirExpr>>,
    pub body: HirExpr,
}

/// HIR select arm (for async select expressions)
#[derive(Debug, Clone)]
pub struct HirSelectArm {
    /// The future expression to wait on
    pub future: HirExpr,
    /// Pattern to bind the result
    pub pattern: HirPattern,
    /// Optional guard condition
    pub guard: Option<Box<HirExpr>>,
    /// Body expression to execute when this arm matches
    pub body: HirExpr,
}

// ==================== PATTERNS ====================

/// HIR pattern
#[derive(Debug, Clone)]
pub enum HirPattern {
    Wildcard,
    Literal(HirLiteral),
    Binding {
        name: String,
        mutable: bool,
    },
    Tuple(Vec<HirPattern>),
    Struct {
        name: String,
        fields: Vec<(String, HirPattern)>,
    },
    Variant {
        enum_name: String,
        variant: String,
        patterns: Vec<HirPattern>,
    },
    Or(Vec<HirPattern>),
}

// ==================== BLOCKS & STATEMENTS ====================

/// HIR block
#[derive(Debug, Clone)]
pub struct HirBlock {
    pub stmts: Vec<HirStmt>,
    pub ty: HirType,
}

/// HIR statement
#[derive(Debug, Clone)]
pub enum HirStmt {
    /// Let binding
    Let {
        name: String,
        ty: HirType,
        value: Option<HirExpr>,
        is_mut: bool,
        /// Layout hint from semantic analysis (Day 38)
        layout_hint: Option<LayoutHint>,
    },
    /// Expression
    Expr(HirExpr),
    /// Assignment
    Assign { target: HirExpr, value: HirExpr },
}

/// Layout hint for memory allocation decisions
///
/// Determined by semantic clustering of Knowledge types (Day 38).
/// Concepts that are semantically close in the ontology should be
/// physically close in memory for better cache performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutHint {
    /// Hot data: frequently accessed, allocate on stack (L1/L2 friendly)
    Stack,
    /// Warm data: moderate access, allocate in arena (L2/L3)
    Arena,
    /// Cold data: rare access, allocate on heap (RAM)
    Heap,
}
