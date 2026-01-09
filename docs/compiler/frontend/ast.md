# Abstract Syntax Tree (AST)

The AST is the tree representation of Sounio source code produced by the parser. It is located in `compiler/src/ast/`.

## Overview

The AST captures the syntactic structure of Sounio programs before type checking. It represents the program as the user wrote it, including all syntactic details needed for error reporting and IDE features.

## Top-Level Structure

```rust
/// Top-level AST
pub struct Ast {
    /// Optional module declaration: `module foo::bar;`
    pub module_name: Option<Path>,

    /// All top-level items
    pub items: Vec<Item>,

    /// Mapping from NodeId to source span for error reporting
    pub node_spans: HashMap<NodeId, Span>,
}
```

## Item Types

Items are top-level declarations. The `Item` enum captures all possible item types:

```rust
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
    OntologyImport(OntologyImportDef),
    AlignDecl(AlignDef),
    OdeDef(OdeDef),
    PdeDef(PdeDef),
    CausalModel(CausalModelDef),
    Module(ModuleDef),
}
```

## Function Definitions

```rust
pub struct FnDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub modifiers: FnModifiers,
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

pub struct FnModifiers {
    pub is_async: bool,
    pub is_unsafe: bool,
    pub is_kernel: bool,  // GPU kernel function
    pub abi: Option<Abi>, // For extern functions
}

pub struct Param {
    pub id: NodeId,
    pub is_mut: bool,
    pub pattern: Pattern,
    pub ty: TypeExpr,
    pub attributes: Vec<Attribute>,
}
```

## Type Definitions

### Structs

```rust
pub struct StructDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub modifiers: TypeModifiers,
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub generics: Generics,
    pub where_clause: Vec<WherePredicate>,
    pub fields: Vec<FieldDef>,
    pub span: Span,
}

pub struct TypeModifiers {
    pub linear: bool,   // Must be used exactly once
    pub affine: bool,   // Must be used at most once
}

pub struct FieldDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub ty: TypeExpr,
}
```

### Enums

```rust
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

pub struct VariantDef {
    pub id: NodeId,
    pub name: String,
    pub data: VariantData,
}

pub enum VariantData {
    Unit,                    // E::Variant
    Tuple(Vec<TypeExpr>),    // E::Variant(T1, T2)
    Struct(Vec<FieldDef>),   // E::Variant { x: T1, y: T2 }
}
```

## Type Expressions

The `TypeExpr` enum represents all type syntax:

```rust
pub enum TypeExpr {
    /// Unit type: ()
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

    /// Function type: Fn(A) -> B with Effects
    Function {
        params: Vec<TypeExpr>,
        return_type: Box<TypeExpr>,
        effects: Vec<EffectRef>,
    },

    /// Infer type: _
    Infer,

    // ========== Sounio Epistemic Types ==========

    /// Knowledge type: Knowledge[T, eps < 0.05, Valid(duration), Derived]
    Knowledge {
        value_type: Box<TypeExpr>,
        epsilon: Option<EpsilonBound>,
        validity: Option<ValidityCondition>,
        provenance: Option<ProvenanceMarker>,
    },

    /// Quantity type: Quantity[f64, meters]
    Quantity {
        numeric_type: Box<TypeExpr>,
        unit: UnitExpr,
    },

    /// Tensor type: Tensor[f32, (batch, channels, height, width)]
    Tensor {
        element_type: Box<TypeExpr>,
        shape: Vec<TensorDim>,
    },

    /// Tile type for GPU: tile<f16, 16, 16>
    Tile {
        element_type: Box<TypeExpr>,
        tile_m: u32,
        tile_n: u32,
        layout: Option<String>,
    },

    /// Ontology type: OntologyTerm[SNOMED:12345]
    Ontology {
        ontology: String,
        term: Option<String>,
    },

    /// Linear/affine type annotation: T @ linear
    Linear {
        inner: Box<TypeExpr>,
        linearity: LinearityKind,
    },

    /// Effect row type: T ! {IO, GPU, Random}
    Effected {
        inner: Box<TypeExpr>,
        effects: EffectRow,
    },

    /// Refinement type: { x: T | predicate }
    Refinement {
        var: String,
        base_type: Box<TypeExpr>,
        predicate: Box<Expr>,
    },
}
```

### Epistemic Type Components

```rust
/// Uncertainty bound: eps < 0.05
pub struct EpsilonBound {
    pub operator: ComparisonOp,  // <, <=, =, >=, >
    pub value: Box<Expr>,
}

/// Validity condition: Valid(duration), ValidUntil(date), ValidWhile(condition)
pub struct ValidityCondition {
    pub kind: ValidityKind,
    pub condition: Box<Expr>,
}

pub enum ValidityKind {
    Valid,
    ValidUntil,
    ValidWhile,
}

/// Provenance marker: Derived, Source(name), Computed, Literature(citation)
pub struct ProvenanceMarker {
    pub kind: ProvenanceKind,
    pub source: Option<Box<Expr>>,
}

pub enum ProvenanceKind {
    Derived,
    Source,
    Computed,
    Literature,
    Measured,
    Input,
}
```

## Expressions

The `Expr` enum captures all expression forms:

```rust
pub enum Expr {
    /// Literal value
    Literal { id: NodeId, value: Literal },

    /// Path reference: foo, std::io::Write
    Path { id: NodeId, path: Path },

    /// Binary operation: a + b
    Binary {
        id: NodeId,
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Unary operation: -x, !b, &x, *p
    Unary {
        id: NodeId,
        op: UnaryOp,
        expr: Box<Expr>,
    },

    /// Function call: f(x, y)
    Call {
        id: NodeId,
        callee: Box<Expr>,
        args: Vec<Expr>,
    },

    /// Method call: x.method(args)
    MethodCall {
        id: NodeId,
        receiver: Box<Expr>,
        method: String,
        args: Vec<Expr>,
    },

    /// Field access: x.field
    Field {
        id: NodeId,
        base: Box<Expr>,
        field: String,
    },

    /// Index operation: arr[i]
    Index {
        id: NodeId,
        base: Box<Expr>,
        index: Box<Expr>,
    },

    /// Type cast: x as T
    Cast {
        id: NodeId,
        expr: Box<Expr>,
        ty: TypeExpr,
    },

    /// Block expression: { stmts; expr }
    Block { id: NodeId, block: Block },

    /// If expression: if cond { } else { }
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

    /// Loop expressions
    Loop { id: NodeId, body: Block },
    While { id: NodeId, condition: Box<Expr>, body: Block },
    For { id: NodeId, pattern: Pattern, iter: Box<Expr>, body: Block },

    /// Control flow
    Return { id: NodeId, value: Option<Box<Expr>> },
    Break { id: NodeId, value: Option<Box<Expr>> },
    Continue { id: NodeId },

    /// Closure: |x, y| expr
    Closure {
        id: NodeId,
        params: Vec<(String, Option<TypeExpr>)>,
        return_type: Option<TypeExpr>,
        body: Box<Expr>,
    },

    /// Collections
    Tuple { id: NodeId, elements: Vec<Expr> },
    Array { id: NodeId, elements: Vec<Expr> },
    Range { id: NodeId, start: Option<Box<Expr>>, end: Option<Box<Expr>>, inclusive: bool },

    /// Struct literal: S { field: value, ... }
    StructLit {
        id: NodeId,
        path: Path,
        fields: Vec<(String, Expr)>,
    },

    /// Try expression: expr?
    Try { id: NodeId, expr: Box<Expr> },

    /// Effect operations
    Perform { id: NodeId, effect: Path, op: String, args: Vec<Expr> },
    Handle { id: NodeId, expr: Box<Expr>, handler: Path },
    Sample { id: NodeId, distribution: Box<Expr> },

    /// Async operations
    Await { id: NodeId, expr: Box<Expr> },
    AsyncBlock { id: NodeId, block: Block },
    Spawn { id: NodeId, expr: Box<Expr> },
    Select { id: NodeId, arms: Vec<SelectArm> },
    Join { id: NodeId, futures: Vec<Expr> },

    // ========== Sounio-Specific Expressions ==========

    /// Ontology term: chebi:15365, drugbank:DB00945
    OntologyTerm {
        id: NodeId,
        ontology: String,
        term: String,
    },

    /// Causal do expression: do(X = 1)
    Do {
        id: NodeId,
        interventions: Vec<(String, Box<Expr>)>,
    },

    /// Counterfactual: counterfactual { factual; do(X=1); outcome }
    Counterfactual {
        id: NodeId,
        factual: Box<Expr>,
        intervention: Box<Expr>,
        outcome: Box<Expr>,
    },

    /// Knowledge construction
    KnowledgeExpr {
        id: NodeId,
        value: Box<Expr>,
        epsilon: Option<Box<Expr>>,
        validity: Option<Box<Expr>>,
        provenance: Option<Box<Expr>>,
    },

    /// Uncertainty: x +- sigma
    Uncertain {
        id: NodeId,
        value: Box<Expr>,
        uncertainty: Box<Expr>,
    },

    /// GPU annotation: expr @ gpu.epistemic
    GpuAnnotated {
        id: NodeId,
        expr: Box<Expr>,
        annotation: GpuAnnotation,
    },

    /// Observe for probabilistic programming
    Observe {
        id: NodeId,
        data: Box<Expr>,
        distribution: Box<Expr>,
    },

    /// Probabilistic query: P(Y | X, do(Z))
    Query {
        id: NodeId,
        target: Box<Expr>,
        given: Vec<Expr>,
        interventions: Vec<(String, Box<Expr>)>,
    },
}
```

## Operators

### Binary Operators

```rust
pub enum BinaryOp {
    // Arithmetic
    Add, Sub, Mul, Div, Rem,

    // Comparison
    Eq, Ne, Lt, Le, Gt, Ge,

    // Logical
    And, Or,

    // Bitwise
    BitAnd, BitOr, BitXor, Shl, Shr,

    // Scientific
    PlusMinus,  // x +- 0.1 (uncertainty)

    // Collection
    Concat,  // a ++ b (array concatenation)
}
```

### Unary Operators

```rust
pub enum UnaryOp {
    Neg,     // -x
    Not,     // !b
    Ref,     // &x
    RefMut,  // &!x
    Deref,   // *p
}
```

## Patterns

```rust
pub enum Pattern {
    /// Wildcard: _
    Wildcard,

    /// Literal: 42, "hello", true
    Literal(Literal),

    /// Variable binding: x, mut y
    Binding { name: String, mutable: bool },

    /// Tuple pattern: (a, b, c)
    Tuple(Vec<Pattern>),

    /// Struct pattern: S { field: pattern, ... }
    Struct {
        path: Path,
        fields: Vec<(String, Pattern)>,
    },

    /// Enum variant pattern: E::V(p1, p2)
    Enum {
        path: Path,
        patterns: Option<Vec<Pattern>>,
    },

    /// Or pattern: p1 | p2
    Or(Vec<Pattern>),
}
```

## Statements

```rust
pub enum Stmt {
    /// Let binding: let x = expr; or var x = expr;
    Let {
        is_mut: bool,
        pattern: Pattern,
        ty: Option<TypeExpr>,
        value: Option<Expr>,
    },

    /// Expression statement
    Expr { expr: Expr, has_semi: bool },

    /// Assignment: x = value; or x += value;
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },

    /// Empty statement: ;
    Empty,

    /// Macro invocation
    MacroInvocation(MacroInvocation),
}

pub enum AssignOp {
    Assign,      // =
    AddAssign,   // +=
    SubAssign,   // -=
    MulAssign,   // *=
    DivAssign,   // /=
    RemAssign,   // %=
    BitAndAssign, BitOrAssign, BitXorAssign,
    ShlAssign, ShrAssign,
}
```

## Effects

```rust
/// Effect reference in function signature
pub struct EffectRef {
    pub id: NodeId,
    pub name: Path,
    pub args: Vec<TypeExpr>,
}

/// Effect row: {IO, GPU, ...} or E (effect variable)
pub struct EffectRow {
    pub effects: Vec<String>,
    pub row_var: Option<String>,  // For effect polymorphism
    pub is_open: bool,
}
```

## Generics

```rust
pub struct Generics {
    pub params: Vec<GenericParam>,
}

pub enum GenericParam {
    /// Type parameter: T, T: Bound, T = Default
    Type {
        name: String,
        bounds: Vec<Path>,
        default: Option<TypeExpr>,
    },

    /// Const parameter: const N: usize
    Const {
        name: String,
        ty: TypeExpr,
    },

    /// Effect parameter for row polymorphism: effect E
    Effect {
        name: String,
    },
}

/// Where predicate: T: Bound
pub struct WherePredicate {
    pub ty: TypeExpr,
    pub bounds: Vec<Path>,
}
```

## Paths

```rust
pub struct Path {
    /// The path segments: ["std", "io", "Write"]
    pub segments: Vec<String>,

    /// Source module for diagnostics
    pub source_module: Option<ModuleId>,

    /// Resolved module (set during name resolution)
    pub resolved_module: Option<ModuleId>,
}
```

## Attributes

```rust
pub struct Attribute {
    pub id: NodeId,
    pub name: String,
    pub args: AttributeArgs,
    pub span: Span,
}

pub enum AttributeArgs {
    Empty,                              // #[inline]
    Value(AttributeValue),              // #[compat(0.15)]
    Named(Vec<(String, AttributeValue)>), // #[cfg(target_os = "linux")]
    List(Vec<AttributeValue>),          // #[derive(Debug, Clone)]
}
```

## Files

- `compiler/src/ast/mod.rs` - All AST type definitions
