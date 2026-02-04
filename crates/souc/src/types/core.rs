//! Core type definitions

use std::collections::HashSet;

/// Type variable for polymorphism
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVar(pub u32);

/// Effect variable for effect polymorphism (row polymorphism)
///
/// Effect variables represent unknown effect rows in polymorphic functions.
/// For example, in `fn map<T, U, E>(f: fn(T) -> U with E, xs: [T]) -> [U] with E`,
/// `E` is an effect variable that gets instantiated at call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectVar(pub u32);

/// Dimension size representation for parametric scientific types
///
/// Used for Array<T, N>, Matrix<T, M, N>, and Tensor shapes.
/// Supports compile-time constants, symbolic names, and dynamic sizes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimSize {
    /// Compile-time constant dimension (e.g., 3, 128)
    Const(usize),
    /// Symbolic dimension variable (e.g., N, M, K)
    /// Used for const generics and shape polymorphism
    Symbolic(String),
    /// Type variable representing an unknown dimension
    Var(DimVar),
    /// Dynamic dimension determined at runtime
    Dynamic,
    /// Binary operation on dimensions (for inferred shapes)
    BinOp {
        op: DimOp,
        left: Box<DimSize>,
        right: Box<DimSize>,
    },
}

/// Dimension variable for dimension inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DimVar(pub u32);

impl DimVar {
    /// Create a new dimension variable with the given id
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Binary operations on dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DimOp {
    /// Addition: N + M
    Add,
    /// Subtraction: N - M (e.g., for slicing)
    Sub,
    /// Multiplication: N * M (e.g., for flattening)
    Mul,
    /// Division: N / M (e.g., for reshaping)
    Div,
    /// Maximum: max(N, M) (e.g., for broadcasting)
    Max,
    /// Minimum: min(N, M)
    Min,
}

impl DimSize {
    /// Check if this is a compile-time constant
    pub fn is_const(&self) -> bool {
        matches!(self, DimSize::Const(_))
    }

    /// Check if this is a symbolic dimension
    pub fn is_symbolic(&self) -> bool {
        matches!(self, DimSize::Symbolic(_))
    }

    /// Check if this is dynamic (runtime-determined)
    pub fn is_dynamic(&self) -> bool {
        matches!(self, DimSize::Dynamic)
    }

    /// Get the constant value if this is a constant dimension
    pub fn as_const(&self) -> Option<usize> {
        match self {
            DimSize::Const(n) => Some(*n),
            _ => None,
        }
    }

    /// Get the symbolic name if this is a symbolic dimension
    pub fn as_symbolic(&self) -> Option<&str> {
        match self {
            DimSize::Symbolic(name) => Some(name),
            _ => None,
        }
    }

    /// Try to evaluate to a constant value (recursively for BinOp)
    pub fn try_eval(&self) -> Option<usize> {
        match self {
            DimSize::Const(n) => Some(*n),
            DimSize::BinOp { op, left, right } => {
                let l = left.try_eval()?;
                let r = right.try_eval()?;
                match op {
                    DimOp::Add => l.checked_add(r),
                    DimOp::Sub => l.checked_sub(r),
                    DimOp::Mul => l.checked_mul(r),
                    DimOp::Div => {
                        if r == 0 {
                            None
                        } else {
                            Some(l / r)
                        }
                    }
                    DimOp::Max => Some(l.max(r)),
                    DimOp::Min => Some(l.min(r)),
                }
            }
            _ => None,
        }
    }

    /// Collect all free dimension variables in this dimension
    pub fn free_dim_vars(&self) -> HashSet<DimVar> {
        let mut vars = HashSet::new();
        self.collect_dim_vars(&mut vars);
        vars
    }

    fn collect_dim_vars(&self, vars: &mut HashSet<DimVar>) {
        match self {
            DimSize::Var(v) => {
                vars.insert(*v);
            }
            DimSize::BinOp { left, right, .. } => {
                left.collect_dim_vars(vars);
                right.collect_dim_vars(vars);
            }
            _ => {}
        }
    }
}

impl std::fmt::Display for DimSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DimSize::Const(n) => write!(f, "{}", n),
            DimSize::Symbolic(name) => write!(f, "{}", name),
            DimSize::Var(v) => write!(f, "?D{}", v.0),
            DimSize::Dynamic => write!(f, "?"),
            DimSize::BinOp { op, left, right } => {
                let op_str = match op {
                    DimOp::Add => "+",
                    DimOp::Sub => "-",
                    DimOp::Mul => "*",
                    DimOp::Div => "/",
                    DimOp::Max => "max",
                    DimOp::Min => "min",
                };
                if matches!(op, DimOp::Max | DimOp::Min) {
                    write!(f, "{}({}, {})", op_str, left, right)
                } else {
                    write!(f, "({} {} {})", left, op_str, right)
                }
            }
        }
    }
}

/// Tensor shape representation for the type system
///
/// Shapes can be:
/// - Static: known at compile time (e.g., `[3, 4]`)
/// - Dynamic: determined at runtime (e.g., `[?, ?]`)
/// - Symbolic: named dimensions for shape polymorphism (e.g., `[N, M]`)
/// - Parametric: uses DimSize for full dimension expression support
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TensorShape {
    /// Static shape with known dimensions
    Static(Vec<usize>),
    /// Dynamic shape with known rank but unknown dimensions
    Dynamic(usize),
    /// Symbolic shape with named dimensions (for generics)
    Symbolic(Vec<String>),
    /// Parametric shape using DimSize (new, more general form)
    Parametric(Vec<DimSize>),
}

impl TensorShape {
    /// Number of dimensions (rank)
    pub fn ndim(&self) -> usize {
        match self {
            TensorShape::Static(dims) => dims.len(),
            TensorShape::Dynamic(n) => *n,
            TensorShape::Symbolic(names) => names.len(),
            TensorShape::Parametric(dims) => dims.len(),
        }
    }

    /// Check if shape is a scalar (0-dimensional)
    pub fn is_scalar(&self) -> bool {
        self.ndim() == 0
    }

    /// Check if shape is fully static
    pub fn is_static(&self) -> bool {
        match self {
            TensorShape::Static(_) => true,
            TensorShape::Parametric(dims) => dims.iter().all(|d| d.is_const()),
            _ => false,
        }
    }

    /// Get static dimensions if available
    pub fn static_dims(&self) -> Option<&[usize]> {
        match self {
            TensorShape::Static(dims) => Some(dims),
            _ => None,
        }
    }

    /// Try to evaluate all dimensions to static values
    pub fn try_static_dims(&self) -> Option<Vec<usize>> {
        match self {
            TensorShape::Static(dims) => Some(dims.clone()),
            TensorShape::Parametric(dims) => dims.iter().map(|d| d.try_eval()).collect(),
            _ => None,
        }
    }

    /// Get parametric dimensions if available
    pub fn parametric_dims(&self) -> Option<&[DimSize]> {
        match self {
            TensorShape::Parametric(dims) => Some(dims),
            _ => None,
        }
    }

    /// Convert to parametric form (normalizes all shape representations)
    pub fn to_parametric(&self) -> Vec<DimSize> {
        match self {
            TensorShape::Static(dims) => dims.iter().map(|&n| DimSize::Const(n)).collect(),
            TensorShape::Dynamic(n) => vec![DimSize::Dynamic; *n],
            TensorShape::Symbolic(names) => {
                names.iter().map(|n| DimSize::Symbolic(n.clone())).collect()
            }
            TensorShape::Parametric(dims) => dims.clone(),
        }
    }

    /// Create a parametric shape from DimSize values
    pub fn from_dims(dims: Vec<DimSize>) -> Self {
        // Check if all are constant - use Static form for efficiency
        if dims.iter().all(|d| d.is_const()) {
            TensorShape::Static(dims.iter().filter_map(|d| d.as_const()).collect())
        } else {
            TensorShape::Parametric(dims)
        }
    }

    /// Total number of elements (if static)
    pub fn numel(&self) -> Option<usize> {
        self.try_static_dims().map(|d| d.iter().product())
    }

    /// Check if two shapes are compatible for broadcasting
    pub fn broadcast_compatible(&self, other: &TensorShape) -> bool {
        // Convert both to parametric form for uniform handling
        let d1 = self.to_parametric();
        let d2 = other.to_parametric();

        let max_len = d1.len().max(d2.len());
        for i in 0..max_len {
            let dim1 = d1.get(d1.len().saturating_sub(i + 1));
            let dim2 = d2.get(d2.len().saturating_sub(i + 1));

            match (dim1, dim2) {
                (Some(DimSize::Const(n1)), Some(DimSize::Const(n2))) => {
                    if *n1 != *n2 && *n1 != 1 && *n2 != 1 {
                        return false;
                    }
                }
                (Some(DimSize::Symbolic(s1)), Some(DimSize::Symbolic(s2))) => {
                    // Symbolic dims must match exactly for broadcasting
                    if s1 != s2 {
                        return false;
                    }
                }
                (Some(DimSize::Dynamic), _) | (_, Some(DimSize::Dynamic)) => {
                    // Dynamic is always compatible (checked at runtime)
                }
                (None, _) | (_, None) => {
                    // Missing dimension is implicitly 1, always compatible
                }
                _ => {
                    // Mixed symbolic/const - conservatively allow if const is 1
                    if let Some(DimSize::Const(1)) = dim1 {
                        continue;
                    }
                    if let Some(DimSize::Const(1)) = dim2 {
                        continue;
                    }
                    // Otherwise, compatibility unknown at compile time
                    // Return true and let runtime check
                }
            }
        }
        true
    }
}

impl EffectVar {
    /// Create a new effect variable with the given id
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Core type representation
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // Primitives
    Unit,
    Bool,
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
    F32,
    F64,
    Char,
    Str,
    String,

    // Compound types
    /// Reference: &T or &mut T
    Ref {
        mutable: bool,
        lifetime: Option<Lifetime>,
        inner: Box<Type>,
    },
    /// Raw pointer: *const T or *mut T (for FFI)
    RawPointer {
        mutable: bool,
        inner: Box<Type>,
    },
    /// Array: [T; N] or slice [T]
    /// Also supports Array<T, N> with const generic dimension
    Array {
        element: Box<Type>,
        /// Size can be None (slice), Some static usize, or use dim for parametric
        size: Option<usize>,
    },
    /// Scientific Array: Array<T, N> with const generic dimension
    /// Supports symbolic dimensions for shape polymorphism
    ScientificArray {
        element: Box<Type>,
        /// Parametric dimension (const, symbolic, or inferred)
        dim: DimSize,
    },
    /// Scientific Matrix: Matrix<T, M, N> with const generic dimensions
    /// Row-major storage, supports symbolic dimensions
    Matrix {
        element: Box<Type>,
        /// Number of rows (M)
        rows: DimSize,
        /// Number of columns (N)
        cols: DimSize,
    },
    /// Tuple: (T1, T2, ...)
    Tuple(Vec<Type>),
    /// Function type: fn(A, B) -> C or extern "C" fn(A, B) -> C
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
        effects: EffectSet,
        /// Optional ABI for function pointers (None = Sounio ABI, Some("C") = C ABI, etc.)
        abi: Option<String>,
    },
    /// Named type (struct, enum, type alias)
    Named {
        name: String,
        args: Vec<Type>,
    },
    /// Physical quantity with unit: f64@kg, i32@m/s
    Quantity {
        numeric: Box<Type>,
        unit: String,
    },

    // Polymorphism
    /// Type variable
    Var(TypeVar),
    /// Forall quantifier: forall a. T
    Forall {
        vars: Vec<TypeVar>,
        inner: Box<Type>,
    },

    // Semantic types
    /// Ontology term type: chebi:drug, SNOMED:12345
    Ontology {
        /// Ontology namespace/prefix (e.g., "chebi", "SNOMED")
        namespace: String,
        /// Term identifier (e.g., "drug", "12345")
        term: String,
    },

    // Linear algebra primitives
    /// 2D vector: vec2 (2x f32)
    Vec2,
    /// 3D vector: vec3 (3x f32)
    Vec3,
    /// 4D vector: vec4 (4x f32)
    Vec4,
    /// 2x2 matrix: mat2 (4x f32, column-major)
    Mat2,
    /// 3x3 matrix: mat3 (9x f32, column-major)
    Mat3,
    /// 4x4 matrix: mat4 (16x f32, column-major)
    Mat4,
    /// Quaternion: quat (4x f32: x, y, z, w)
    Quat,

    // ==================== HYPERCOMPLEX TYPES ====================
    /// Octonion: oct (8x f32: basis 1, i, j, k, l, il, jl, kl)
    /// Octonions form a division algebra over R with 8 dimensions
    /// Used for exceptional Lie groups, string theory, and 8x parameter efficiency in ML
    Octonion,

    /// Sedenion: sed (16x f32: basis e₀=1, e₁, e₂, ..., e₁₅)
    /// Sedenions are 16-dimensional hypercomplex numbers via Cayley-Dickson construction
    /// IMPORTANT: Sedenions have ZERO DIVISORS (non-zero elements whose product is zero)
    /// Used for: 16-compartment PBPK models, multi-modal imaging fusion, EEG/MEG analysis
    /// Medical applications: unified multi-organ pharmacokinetics, protein conformation
    Sedenion,

    // ==================== QUATERNIONIC NEURAL NETWORK TYPES ====================
    /// Quaternionic Linear Layer weights: Quat[input_features, output_features]
    /// Each weight is a quaternion (4x f32), giving 4x parameter efficiency
    QuatLinear {
        input_features: usize,
        output_features: usize,
    },
    /// Quaternionic 2D Convolution kernel: Quat[channels_in, channels_out, kH, kW]
    QuatConv2d {
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
    },
    /// Quaternionic recurrent cell state
    QuatRnnState {
        hidden_size: usize,
    },
    /// Quaternionic Gate (for LSTM/GRU gates)
    QuatGate {
        input_size: usize,
        hidden_size: usize,
    },

    // Tensor types
    /// Generic tensor type: Tensor<T, Shape>
    /// Shape can be static, dynamic, or symbolic
    Tensor {
        element: Box<Type>,
        shape: TensorShape,
    },
    /// Double-precision 2D vector: vec2d (2x f64)
    Vec2d,
    /// Double-precision 3D vector: vec3d (3x f64)
    Vec3d,
    /// Double-precision 4D vector: vec4d (4x f64)
    Vec4d,

    // Automatic differentiation types
    /// Dual number for forward-mode autodiff: dual (value: f64, derivative: f64)
    Dual,

    // Causal inference types
    /// Causal graph reference: Causal<GraphName>
    /// Used in effect signatures: `fn f() with Causal<MyGraph>`
    /// The graph_name references a declared `causal model` or `causal graph`
    CausalGraph {
        /// Name of the declared causal model/graph
        graph_name: String,
    },

    // Special types
    /// Never type (!)
    Never,
    /// Unknown type (for inference)
    Unknown,
    /// Error type (for error recovery)
    Error,
    /// Self type (within impl blocks)
    SelfType,
}

impl Type {
    /// Check if this type is a primitive
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            Type::Unit
                | Type::Bool
                | Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::I128
                | Type::Isize
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
                | Type::U128
                | Type::Usize
                | Type::F32
                | Type::F64
                | Type::Char
                | Type::Vec2
                | Type::Vec3
                | Type::Vec4
                | Type::Mat2
                | Type::Mat3
                | Type::Mat4
                | Type::Quat
                | Type::Octonion
                | Type::Sedenion
                | Type::Dual
        )
    }

    /// Check if this type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::I128
                | Type::Isize
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
                | Type::U128
                | Type::Usize
                | Type::F32
                | Type::F64
        )
    }

    /// Check if this type is an integer
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::I128
                | Type::Isize
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
                | Type::U128
                | Type::Usize
        )
    }

    /// Check if this type is a floating point
    pub fn is_float(&self) -> bool {
        matches!(self, Type::F32 | Type::F64)
    }

    /// Check if this type is signed
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128 | Type::Isize
        )
    }

    /// Check if this type is a vector type
    pub fn is_vector(&self) -> bool {
        matches!(
            self,
            Type::Vec2 | Type::Vec3 | Type::Vec4 | Type::Vec2d | Type::Vec3d | Type::Vec4d
        )
    }

    /// Check if this type is a tensor type
    pub fn is_tensor(&self) -> bool {
        matches!(self, Type::Tensor { .. })
    }

    /// Check if this type is f64-based vector
    pub fn is_f64_vector(&self) -> bool {
        matches!(self, Type::Vec2d | Type::Vec3d | Type::Vec4d)
    }

    /// Get tensor shape if this is a tensor type
    pub fn tensor_shape(&self) -> Option<&TensorShape> {
        match self {
            Type::Tensor { shape, .. } => Some(shape),
            _ => None,
        }
    }

    /// Get tensor element type if this is a tensor type
    pub fn tensor_element(&self) -> Option<&Type> {
        match self {
            Type::Tensor { element, .. } => Some(element.as_ref()),
            _ => None,
        }
    }

    /// Check if this type is a matrix type (fixed or parametric)
    pub fn is_matrix(&self) -> bool {
        matches!(
            self,
            Type::Mat2 | Type::Mat3 | Type::Mat4 | Type::Matrix { .. }
        )
    }

    /// Check if this type is a scientific array (Array<T, N>)
    pub fn is_scientific_array(&self) -> bool {
        matches!(self, Type::ScientificArray { .. })
    }

    /// Check if this type is a scientific matrix (Matrix<T, M, N>)
    pub fn is_scientific_matrix(&self) -> bool {
        matches!(self, Type::Matrix { .. })
    }

    /// Get the element type of an array or matrix type
    pub fn array_element(&self) -> Option<&Type> {
        match self {
            Type::Array { element, .. } => Some(element.as_ref()),
            Type::ScientificArray { element, .. } => Some(element.as_ref()),
            Type::Matrix { element, .. } => Some(element.as_ref()),
            _ => None,
        }
    }

    /// Get the dimension of a scientific array if available
    pub fn scientific_array_dim(&self) -> Option<&DimSize> {
        match self {
            Type::ScientificArray { dim, .. } => Some(dim),
            _ => None,
        }
    }

    /// Get the dimensions of a matrix type
    pub fn matrix_dims(&self) -> Option<(&DimSize, &DimSize)> {
        match self {
            Type::Matrix { rows, cols, .. } => Some((rows, cols)),
            // Fixed matrices have known dimensions
            Type::Mat2 => None, // Could return (Const(2), Const(2)) but different type
            Type::Mat3 => None,
            Type::Mat4 => None,
            _ => None,
        }
    }

    /// Check if this type is a quaternion
    pub fn is_quaternion(&self) -> bool {
        matches!(self, Type::Quat)
    }

    /// Check if this type is an octonion
    pub fn is_octonion(&self) -> bool {
        matches!(self, Type::Octonion)
    }

    /// Check if this type is a sedenion
    pub fn is_sedenion(&self) -> bool {
        matches!(self, Type::Sedenion)
    }

    /// Check if this type is a hypercomplex number (quaternion, octonion, or sedenion)
    /// These form the Cayley-Dickson sequence: ℝ → ℂ → ℍ → 𝕆 → 𝕊
    pub fn is_hypercomplex(&self) -> bool {
        matches!(self, Type::Quat | Type::Octonion | Type::Sedenion)
    }

    /// Get the dimension of a hypercomplex type (4, 8, or 16)
    pub fn hypercomplex_dimension(&self) -> Option<usize> {
        match self {
            Type::Quat => Some(4),
            Type::Octonion => Some(8),
            Type::Sedenion => Some(16),
            _ => None,
        }
    }

    /// Check if this hypercomplex type has zero divisors
    /// Only sedenions (and higher Cayley-Dickson constructions) have zero divisors
    pub fn has_zero_divisors(&self) -> bool {
        matches!(self, Type::Sedenion)
    }

    /// Check if this type is a linear algebra primitive
    pub fn is_linear_algebra(&self) -> bool {
        self.is_vector() || self.is_matrix() || self.is_hypercomplex()
    }

    /// Check if this type is a dual number (for autodiff)
    pub fn is_dual(&self) -> bool {
        matches!(self, Type::Dual)
    }

    /// Check if this type supports automatic differentiation
    pub fn is_differentiable(&self) -> bool {
        self.is_float() || self.is_dual()
    }

    /// Get the dimension of a vector type
    pub fn vector_dimension(&self) -> Option<usize> {
        match self {
            Type::Vec2 => Some(2),
            Type::Vec3 => Some(3),
            Type::Vec4 => Some(4),
            _ => None,
        }
    }

    /// Get the dimension of a matrix type (returns NxN)
    pub fn matrix_dimension(&self) -> Option<usize> {
        match self {
            Type::Mat2 => Some(2),
            Type::Mat3 => Some(3),
            Type::Mat4 => Some(4),
            _ => None,
        }
    }

    /// Get all free type variables in this type
    pub fn free_vars(&self) -> HashSet<TypeVar> {
        let mut vars = HashSet::new();
        self.collect_free_vars(&mut vars);
        vars
    }

    fn collect_free_vars(&self, vars: &mut HashSet<TypeVar>) {
        match self {
            Type::Var(v) => {
                vars.insert(*v);
            }
            Type::Ref { inner, .. } => inner.collect_free_vars(vars),
            Type::RawPointer { inner, .. } => inner.collect_free_vars(vars),
            Type::Array { element, .. } => element.collect_free_vars(vars),
            Type::ScientificArray { element, .. } => element.collect_free_vars(vars),
            Type::Matrix { element, .. } => element.collect_free_vars(vars),
            Type::Tensor { element, .. } => element.collect_free_vars(vars),
            Type::Quantity { numeric, .. } => numeric.collect_free_vars(vars),
            Type::Tuple(elems) => {
                for elem in elems {
                    elem.collect_free_vars(vars);
                }
            }
            Type::Function {
                params,
                return_type,
                ..
            } => {
                for param in params {
                    param.collect_free_vars(vars);
                }
                return_type.collect_free_vars(vars);
            }
            Type::Named { args, .. } => {
                for arg in args {
                    arg.collect_free_vars(vars);
                }
            }
            Type::Forall { vars: bound, inner } => {
                let mut inner_vars = HashSet::new();
                inner.collect_free_vars(&mut inner_vars);
                for v in inner_vars {
                    if !bound.contains(&v) {
                        vars.insert(v);
                    }
                }
            }
            _ => {}
        }
    }

    /// Substitute type variables
    pub fn substitute(&self, subst: &std::collections::HashMap<TypeVar, Type>) -> Type {
        match self {
            Type::Var(v) => subst.get(v).cloned().unwrap_or_else(|| self.clone()),
            Type::Ref {
                mutable,
                lifetime,
                inner,
            } => Type::Ref {
                mutable: *mutable,
                lifetime: lifetime.clone(),
                inner: Box::new(inner.substitute(subst)),
            },
            Type::RawPointer { mutable, inner } => Type::RawPointer {
                mutable: *mutable,
                inner: Box::new(inner.substitute(subst)),
            },
            Type::Array { element, size } => Type::Array {
                element: Box::new(element.substitute(subst)),
                size: *size,
            },
            Type::ScientificArray { element, dim } => Type::ScientificArray {
                element: Box::new(element.substitute(subst)),
                dim: dim.clone(),
            },
            Type::Matrix {
                element,
                rows,
                cols,
            } => Type::Matrix {
                element: Box::new(element.substitute(subst)),
                rows: rows.clone(),
                cols: cols.clone(),
            },
            Type::Tensor { element, shape } => Type::Tensor {
                element: Box::new(element.substitute(subst)),
                shape: shape.clone(),
            },
            Type::Quantity { numeric, unit } => Type::Quantity {
                numeric: Box::new(numeric.substitute(subst)),
                unit: unit.clone(),
            },
            Type::Tuple(elems) => Type::Tuple(elems.iter().map(|e| e.substitute(subst)).collect()),
            Type::Function {
                params,
                return_type,
                effects,
                abi,
            } => Type::Function {
                params: params.iter().map(|p| p.substitute(subst)).collect(),
                return_type: Box::new(return_type.substitute(subst)),
                effects: effects.clone(),
                abi: abi.clone(),
            },
            Type::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|a| a.substitute(subst)).collect(),
            },
            Type::Forall { vars, inner } => {
                // Avoid capturing bound variables
                let mut new_subst = subst.clone();
                for v in vars {
                    new_subst.remove(v);
                }
                Type::Forall {
                    vars: vars.clone(),
                    inner: Box::new(inner.substitute(&new_subst)),
                }
            }
            _ => self.clone(),
        }
    }
}

/// Lifetime for references
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lifetime {
    pub name: String,
}

impl Lifetime {
    pub fn static_lifetime() -> Self {
        Self {
            name: "'static".to_string(),
        }
    }

    pub fn anonymous() -> Self {
        Self {
            name: "'_".to_string(),
        }
    }
}

/// Effect in a type signature
#[derive(Debug, Clone, PartialEq)]
pub struct Effect {
    pub name: String,
    pub args: Vec<Type>,
}

impl Effect {
    pub fn io() -> Self {
        Self {
            name: "IO".to_string(),
            args: Vec::new(),
        }
    }

    pub fn mut_effect() -> Self {
        Self {
            name: "Mut".to_string(),
            args: Vec::new(),
        }
    }

    pub fn alloc() -> Self {
        Self {
            name: "Alloc".to_string(),
            args: Vec::new(),
        }
    }

    pub fn prob() -> Self {
        Self {
            name: "Prob".to_string(),
            args: Vec::new(),
        }
    }

    pub fn gpu() -> Self {
        Self {
            name: "GPU".to_string(),
            args: Vec::new(),
        }
    }

    /// Epistemic effect - operations that affect confidence/provenance
    ///
    /// The Epistemic effect tracks operations that:
    /// - Degrade or modify confidence values
    /// - Add to provenance chains
    /// - Cross epistemic firewall boundaries
    /// - Perform uncertainty model operations
    ///
    /// This effect enables:
    /// - Compile-time tracking of epistemic operations
    /// - Effect handlers for confidence boundaries (firewalls)
    /// - Safe composition of epistemic computations
    pub fn epistemic() -> Self {
        Self {
            name: "Epistemic".to_string(),
            args: Vec::new(),
        }
    }

    /// Epistemic effect with confidence bound parameter
    ///
    /// `Epistemic[min_confidence]` indicates the minimum confidence
    /// that will be maintained after the operation.
    pub fn epistemic_bounded(min_confidence: f64) -> Self {
        Self {
            name: "Epistemic".to_string(),
            args: vec![Type::F64], // Would carry the bound as type-level value
        }
    }

    /// Div effect - operations that may divide by zero
    pub fn div() -> Self {
        Self {
            name: "Div".to_string(),
            args: Vec::new(),
        }
    }

    /// Exn effect - operations that may throw exceptions
    pub fn exn() -> Self {
        Self {
            name: "Exn".to_string(),
            args: Vec::new(),
        }
    }

    /// Async effect - asynchronous operations
    pub fn async_effect() -> Self {
        Self {
            name: "Async".to_string(),
            args: Vec::new(),
        }
    }

    /// FFI effect - foreign function interface operations
    ///
    /// The FFI effect tracks operations that interact with foreign code:
    /// - Calling extern "C" functions
    /// - Dereferencing raw pointers
    /// - Memory operations on foreign data
    ///
    /// Functions that perform FFI operations must declare this effect:
    /// ```sounio
    /// fn call_libc() with FFI {
    ///     extern "C" { fn puts(s: *const i8) -> i32 }
    ///     puts(c"hello")
    /// }
    /// ```
    pub fn ffi() -> Self {
        Self {
            name: "FFI".to_string(),
            args: Vec::new(),
        }
    }
}

/// Set of effects with support for effect polymorphism
///
/// An EffectSet represents a row of effects that a function may perform.
/// It contains:
/// - `effects`: Known concrete effects like IO, Mut, Alloc
/// - `effect_vars`: Effect variables for polymorphism (e.g., E in `fn map<E>`)
///
/// Effect variables enable row polymorphism where a function can be generic
/// over the effects it performs, allowing code like:
/// ```sio
/// fn map<T, U, E>(f: fn(T) -> U with E, xs: [T]) -> [U] with E
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EffectSet {
    /// Concrete effects (IO, Mut, Alloc, etc.)
    pub effects: HashSet<String>,
    /// Effect variables for row polymorphism
    pub effect_vars: HashSet<EffectVar>,
    /// Legacy: type variables used as effect variables (for backwards compatibility)
    #[deprecated(note = "Use effect_vars instead")]
    pub vars: HashSet<TypeVar>,
}

impl EffectSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn pure() -> Self {
        Self::new()
    }

    pub fn single(effect: Effect) -> Self {
        let mut set = Self::new();
        set.effects.insert(effect.name);
        set
    }

    /// Create an EffectSet with a single effect variable
    pub fn with_var(var: EffectVar) -> Self {
        let mut set = Self::new();
        set.effect_vars.insert(var);
        set
    }

    pub fn add(&mut self, effect: Effect) {
        self.effects.insert(effect.name);
    }

    /// Add an effect variable to this set
    pub fn add_var(&mut self, var: EffectVar) {
        self.effect_vars.insert(var);
    }

    pub fn union(&self, other: &EffectSet) -> EffectSet {
        #[allow(deprecated)]
        EffectSet {
            effects: self.effects.union(&other.effects).cloned().collect(),
            effect_vars: self
                .effect_vars
                .union(&other.effect_vars)
                .cloned()
                .collect(),
            vars: self.vars.union(&other.vars).cloned().collect(),
        }
    }

    pub fn is_pure(&self) -> bool {
        #[allow(deprecated)]
        {
            self.effects.is_empty() && self.effect_vars.is_empty() && self.vars.is_empty()
        }
    }

    /// Check if this effect set has any unresolved effect variables
    pub fn has_effect_vars(&self) -> bool {
        !self.effect_vars.is_empty()
    }

    pub fn contains(&self, effect: &str) -> bool {
        self.effects.contains(effect)
    }

    /// Check if this effect set contains a specific effect variable
    pub fn contains_var(&self, var: EffectVar) -> bool {
        self.effect_vars.contains(&var)
    }

    /// Subtract handled effects from this effect set.
    ///
    /// Returns a new EffectSet with the specified effects removed.
    /// This is used for effect masking when a handler handles certain effects,
    /// allowing functions to be pure even if they use impure operations internally.
    ///
    /// # Example
    /// ```ignore
    /// let effects = EffectSet::from_effects(&["IO", "Mut", "Alloc"]);
    /// let residual = effects.subtract(&["IO"]);
    /// // residual contains only Mut and Alloc
    /// ```
    pub fn subtract(&self, handled: &[String]) -> EffectSet {
        let mut result = self.clone();
        for h in handled {
            result.effects.remove(h);
        }
        result
    }

    /// Create an EffectSet from a slice of effect names.
    pub fn from_effects(names: &[&str]) -> EffectSet {
        let mut set = EffectSet::new();
        for name in names {
            set.effects.insert((*name).to_string());
        }
        set
    }

    /// Substitute effect variables according to a substitution map
    ///
    /// This is used during instantiation to replace effect variables with
    /// their concrete effect sets.
    pub fn substitute(&self, subst: &std::collections::HashMap<EffectVar, EffectSet>) -> EffectSet {
        let mut result = EffectSet::new();
        result.effects = self.effects.clone();

        for var in &self.effect_vars {
            if let Some(replacement) = subst.get(var) {
                // Replace this variable with the concrete effects
                result.effects.extend(replacement.effects.iter().cloned());
                result
                    .effect_vars
                    .extend(replacement.effect_vars.iter().cloned());
            } else {
                // Variable not in substitution, keep it
                result.effect_vars.insert(*var);
            }
        }

        result
    }

    /// Get all effect variable ids in this set
    pub fn effect_var_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.effect_vars.iter().map(|v| v.0)
    }
}

/// Type scheme (polymorphic type)
#[derive(Debug, Clone)]
pub struct TypeScheme {
    pub vars: Vec<TypeVar>,
    pub ty: Type,
}

impl TypeScheme {
    pub fn mono(ty: Type) -> Self {
        Self {
            vars: Vec::new(),
            ty,
        }
    }

    pub fn instantiate(&self, fresh_vars: &[Type]) -> Type {
        let mut subst = std::collections::HashMap::new();
        for (var, ty) in self.vars.iter().zip(fresh_vars.iter()) {
            subst.insert(*var, ty.clone());
        }
        self.ty.substitute(&subst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_is_numeric() {
        assert!(Type::I32.is_numeric());
        assert!(Type::F64.is_numeric());
        assert!(!Type::Bool.is_numeric());
        assert!(!Type::String.is_numeric());
    }

    #[test]
    fn test_free_vars() {
        let v1 = TypeVar(1);
        let v2 = TypeVar(2);
        let ty = Type::Function {
            params: vec![Type::Var(v1)],
            return_type: Box::new(Type::Var(v2)),
            effects: EffectSet::new(),
            abi: None,
        };
        let vars = ty.free_vars();
        assert!(vars.contains(&v1));
        assert!(vars.contains(&v2));
    }

    #[test]
    fn test_effect_set() {
        let mut effects = EffectSet::new();
        assert!(effects.is_pure());

        effects.add(Effect::io());
        assert!(!effects.is_pure());
        assert!(effects.contains("IO"));
    }

    #[test]
    fn test_effect_set_subtract() {
        // Create an effect set with multiple effects
        let effects = EffectSet::from_effects(&["IO", "Mut", "Alloc"]);
        assert!(effects.contains("IO"));
        assert!(effects.contains("Mut"));
        assert!(effects.contains("Alloc"));

        // Subtract IO effect
        let residual = effects.subtract(&["IO".to_string()]);
        assert!(!residual.contains("IO"));
        assert!(residual.contains("Mut"));
        assert!(residual.contains("Alloc"));

        // Subtract multiple effects
        let residual2 = effects.subtract(&["IO".to_string(), "Mut".to_string()]);
        assert!(!residual2.contains("IO"));
        assert!(!residual2.contains("Mut"));
        assert!(residual2.contains("Alloc"));

        // Subtract all effects
        let residual3 =
            effects.subtract(&["IO".to_string(), "Mut".to_string(), "Alloc".to_string()]);
        assert!(residual3.is_pure());
    }

    #[test]
    fn test_effect_set_from_effects() {
        let effects = EffectSet::from_effects(&["IO", "Mut"]);
        assert!(effects.contains("IO"));
        assert!(effects.contains("Mut"));
        assert!(!effects.contains("Alloc"));
        assert!(!effects.is_pure());
    }

    #[test]
    fn test_effect_masking_pattern() {
        // Test the pattern used in effect masking:
        // A function uses IO and Mut, handles IO internally, residual is just Mut
        let inferred_effects = EffectSet::from_effects(&["IO", "Mut"]);
        let handled_effects = EffectSet::from_effects(&["IO"]);

        // Compute residual effects
        let handled_names: Vec<String> = handled_effects.effects.iter().cloned().collect();
        let residual = inferred_effects.subtract(&handled_names);

        // Only Mut should remain
        assert!(!residual.contains("IO"));
        assert!(residual.contains("Mut"));
    }

    #[test]
    fn test_effect_var() {
        // Test EffectVar creation and equality
        let v1 = EffectVar::new(0);
        let v2 = EffectVar::new(1);
        let v1_dup = EffectVar::new(0);

        assert_eq!(v1, v1_dup);
        assert_ne!(v1, v2);
        assert_eq!(v1.0, 0);
        assert_eq!(v2.0, 1);
    }

    #[test]
    fn test_effect_set_with_effect_vars() {
        // Test EffectSet with effect variables for row polymorphism
        let mut effects = EffectSet::new();
        let e_var = EffectVar::new(42);

        // Add an effect variable
        effects.add_var(e_var);
        assert!(!effects.is_pure()); // Not pure - has an effect variable
        assert!(effects.has_effect_vars());
        assert!(effects.contains_var(e_var));

        // Add a concrete effect
        effects.add(Effect::io());
        assert!(effects.contains("IO"));
        assert!(effects.has_effect_vars());
    }

    #[test]
    fn test_effect_set_substitution() {
        // Test substituting effect variables with concrete effects
        let e_var = EffectVar::new(0);
        let mut effects = EffectSet::with_var(e_var);
        effects.add(Effect::mut_effect()); // effects = {Mut, E}

        // Create substitution: E -> {IO, Alloc}
        let mut replacement = EffectSet::new();
        replacement.add(Effect::io());
        replacement.add(Effect::alloc());

        let mut subst = std::collections::HashMap::new();
        subst.insert(e_var, replacement);

        let result = effects.substitute(&subst);
        // Result should be {Mut, IO, Alloc} with no effect variables
        assert!(result.contains("Mut"));
        assert!(result.contains("IO"));
        assert!(result.contains("Alloc"));
        assert!(!result.has_effect_vars());
    }

    #[test]
    fn test_effect_set_union_with_vars() {
        let e1 = EffectVar::new(1);
        let e2 = EffectVar::new(2);

        let mut set1 = EffectSet::new();
        set1.add(Effect::io());
        set1.add_var(e1);

        let mut set2 = EffectSet::new();
        set2.add(Effect::mut_effect());
        set2.add_var(e2);

        let union = set1.union(&set2);
        // Union should contain both effects and both effect variables
        assert!(union.contains("IO"));
        assert!(union.contains("Mut"));
        assert!(union.contains_var(e1));
        assert!(union.contains_var(e2));
    }
}
