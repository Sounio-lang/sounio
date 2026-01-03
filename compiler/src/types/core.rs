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
    Array {
        element: Box<Type>,
        size: Option<usize>,
    },
    /// Tuple: (T1, T2, ...)
    Tuple(Vec<Type>),
    /// Function type: fn(A, B) -> C
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
        effects: EffectSet,
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

    // Automatic differentiation types
    /// Dual number for forward-mode autodiff: dual (value: f64, derivative: f64)
    Dual,

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
        matches!(self, Type::Vec2 | Type::Vec3 | Type::Vec4)
    }

    /// Check if this type is a matrix type
    pub fn is_matrix(&self) -> bool {
        matches!(self, Type::Mat2 | Type::Mat3 | Type::Mat4)
    }

    /// Check if this type is a quaternion
    pub fn is_quaternion(&self) -> bool {
        matches!(self, Type::Quat)
    }

    /// Check if this type is a linear algebra primitive
    pub fn is_linear_algebra(&self) -> bool {
        self.is_vector() || self.is_matrix() || self.is_quaternion()
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
            Type::Array { element, .. } => element.collect_free_vars(vars),
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
            Type::Array { element, size } => Type::Array {
                element: Box::new(element.substitute(subst)),
                size: *size,
            },
            Type::Tuple(elems) => Type::Tuple(elems.iter().map(|e| e.substitute(subst)).collect()),
            Type::Function {
                params,
                return_type,
                effects,
            } => Type::Function {
                params: params.iter().map(|p| p.substitute(subst)).collect(),
                return_type: Box::new(return_type.substitute(subst)),
                effects: effects.clone(),
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
            effect_vars: self.effect_vars.union(&other.effect_vars).cloned().collect(),
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
                result.effect_vars.extend(replacement.effect_vars.iter().cloned());
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
}
