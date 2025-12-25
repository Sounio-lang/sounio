//! Symbol table implementation

use crate::ast::{ImportItem, ModuleId, Visibility};
use crate::common::{NodeId, Span};
use std::collections::HashMap;

/// Unique definition ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(pub u32);

impl DefId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Kind of definition
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefKind {
    /// Function definition
    Function,
    /// Variable (let binding)
    Variable { mutable: bool },
    /// Function parameter
    Parameter { mutable: bool },
    /// Struct type
    Struct { is_linear: bool, is_affine: bool },
    /// Enum type
    Enum { is_linear: bool, is_affine: bool },
    /// Enum variant
    Variant,
    /// Type alias
    TypeAlias,
    /// Type parameter (generic)
    TypeParam,
    /// Effect definition
    Effect,
    /// Effect operation
    EffectOp,
    /// Constant
    Const,
    /// Module
    Module,
    /// Trait
    Trait,
    /// Field (struct field)
    Field,
    /// Kernel function
    Kernel,
    /// Built-in type
    BuiltinType,
    /// Built-in function (print, println, etc.)
    BuiltinFunction,
}

/// Symbol information
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Unique ID
    pub def_id: DefId,
    /// Name as string
    pub name: String,
    /// Kind of definition
    pub kind: DefKind,
    /// Original AST node
    pub node_id: NodeId,
    /// Span in source
    pub span: Span,
    /// Parent scope (for nested items)
    pub parent: Option<DefId>,
}

/// Scope level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    /// Module/file level
    Module,
    /// Function body
    Function,
    /// Block (if, loop, etc.)
    Block,
    /// Struct/enum definition
    TypeDef,
    /// Impl block
    Impl,
}

/// A single scope
#[derive(Debug)]
pub struct Scope {
    pub kind: ScopeKind,
    /// Names defined in this scope (values: variables, functions, etc.)
    pub names: HashMap<String, DefId>,
    /// Type names (separate namespace)
    pub types: HashMap<String, DefId>,
    /// Parent scope DefId (for functions/methods)
    pub parent_def: Option<DefId>,
}

impl Scope {
    pub fn new(kind: ScopeKind, parent_def: Option<DefId>) -> Self {
        Self {
            kind,
            names: HashMap::new(),
            types: HashMap::new(),
            parent_def,
        }
    }
}

// ==================== MODULE SYSTEM ====================

/// A module with its own namespace and visibility control
#[derive(Debug, Clone)]
pub struct ModuleScope {
    /// Module identity (e.g., ["std", "collections"])
    pub id: ModuleId,
    /// Module name (last segment of path)
    pub name: String,
    /// Private items (visible only within this module)
    pub private_names: HashMap<String, DefId>,
    pub private_types: HashMap<String, DefId>,
    /// Public items (visible from outside)
    pub public_names: HashMap<String, DefId>,
    pub public_types: HashMap<String, DefId>,
    /// Re-exports: local_name -> (source_module, original_name)
    pub reexports: HashMap<String, (ModuleId, String)>,
    /// Child modules
    pub children: HashMap<String, ModuleId>,
    /// Parent module (None for root)
    pub parent: Option<ModuleId>,
    /// Imports in this module: local_name -> (source_module, original_name)
    pub imports: HashMap<String, (ModuleId, String)>,
}

impl ModuleScope {
    pub fn new(id: ModuleId, parent: Option<ModuleId>) -> Self {
        let name = id.path.last().cloned().unwrap_or_default();
        Self {
            id,
            name,
            private_names: HashMap::new(),
            private_types: HashMap::new(),
            public_names: HashMap::new(),
            public_types: HashMap::new(),
            reexports: HashMap::new(),
            children: HashMap::new(),
            parent,
            imports: HashMap::new(),
        }
    }

    /// Create root module
    pub fn root() -> Self {
        Self::new(ModuleId::root(), None)
    }

    /// Define a value with visibility
    pub fn define(&mut self, name: String, def_id: DefId, visibility: Visibility) {
        if matches!(visibility, Visibility::Public) {
            self.public_names.insert(name.clone(), def_id);
        }
        self.private_names.insert(name, def_id);
    }

    /// Define a type with visibility
    pub fn define_type(&mut self, name: String, def_id: DefId, visibility: Visibility) {
        if matches!(visibility, Visibility::Public) {
            self.public_types.insert(name.clone(), def_id);
        }
        self.private_types.insert(name, def_id);
    }

    /// Look up a value, respecting visibility
    pub fn lookup(&self, name: &str, from_same_module: bool) -> Option<DefId> {
        // First check imports
        if let Some((source_mod, orig_name)) = self.imports.get(name) {
            // Import resolution handled by caller
            return None; // Signal to look in source module
        }

        if from_same_module {
            self.private_names.get(name).copied()
        } else {
            self.public_names.get(name).copied()
        }
    }

    /// Look up a type, respecting visibility
    pub fn lookup_type(&self, name: &str, from_same_module: bool) -> Option<DefId> {
        if from_same_module {
            self.private_types.get(name).copied()
        } else {
            self.public_types.get(name).copied()
        }
    }

    /// Add a child module
    pub fn add_child(&mut self, name: String, child_id: ModuleId) {
        self.children.insert(name, child_id);
    }

    /// Register an import
    pub fn add_import(
        &mut self,
        local_name: String,
        source_module: ModuleId,
        original_name: String,
    ) {
        self.imports
            .insert(local_name, (source_module, original_name));
    }

    /// Register a re-export (pub use)
    pub fn add_reexport(
        &mut self,
        local_name: String,
        source_module: ModuleId,
        original_name: String,
        def_id: DefId,
    ) {
        self.reexports
            .insert(local_name.clone(), (source_module, original_name));
        // Re-exports are also public
        self.public_names.insert(local_name, def_id);
    }

    /// Get all public items (for glob imports)
    pub fn public_items(&self) -> impl Iterator<Item = (&String, &DefId)> {
        self.public_names.iter().chain(self.public_types.iter())
    }
}

/// Symbol table with scoped lookups
#[derive(Debug)]
pub struct SymbolTable {
    /// All symbols by DefId
    symbols: HashMap<DefId, Symbol>,
    /// Scope stack
    scopes: Vec<Scope>,
    /// Next DefId
    next_id: u32,
    /// NodeId -> DefId mapping (for definitions)
    node_to_def: HashMap<NodeId, DefId>,
    /// NodeId -> DefId mapping (for references)
    node_to_ref: HashMap<NodeId, DefId>,
    /// All modules by their ID
    modules: HashMap<ModuleId, ModuleScope>,
    /// Current module context for resolution
    current_module: ModuleId,
}

impl SymbolTable {
    pub fn new() -> Self {
        let root_module = ModuleId::root();
        let mut modules = HashMap::new();
        modules.insert(root_module.clone(), ModuleScope::root());

        let mut table = Self {
            symbols: HashMap::new(),
            scopes: Vec::new(),
            next_id: 0,
            node_to_def: HashMap::new(),
            node_to_ref: HashMap::new(),
            modules,
            current_module: root_module,
        };
        // Start with module scope
        table.push_scope(ScopeKind::Module, None);
        // Register built-in types
        table.register_builtins();
        table
    }

    fn register_builtins(&mut self) {
        // Built-in types
        let builtins = [
            "i8",
            "i16",
            "i32",
            "i64",
            "i128",
            "isize",
            "u8",
            "u16",
            "u32",
            "u64",
            "u128",
            "usize",
            "f32",
            "f64",
            "bool",
            "char",
            "String",
            "str",
            // Collection types
            "Vec",
            "HashMap",
            "HashSet",
            "Option",
            "Result",
            "Box",
            "Rc",
            "Arc",
            // Other common types
            "PhantomData",
            "Range",
            // Linear algebra primitives
            "vec2",
            "vec3",
            "vec4",
            "mat2",
            "mat3",
            "mat4",
            "quat",
            "dual",      // Dual number for automatic differentiation
            "uncertain", // Uncertain value with error propagation
            "Tensor",    // Tensor with compile-time shape verification
        ];

        for name in builtins {
            let def_id = self.fresh_def_id();
            let _ = self.define_type(name.to_string(), def_id);
            self.symbols.insert(
                def_id,
                Symbol {
                    def_id,
                    name: name.to_string(),
                    kind: DefKind::BuiltinType,
                    node_id: NodeId(0),
                    span: Span::default(),
                    parent: None,
                },
            );
        }

        // Built-in effects
        let builtin_effects = [
            "IO",    // File, network, console I/O
            "Mut",   // Mutable state
            "Alloc", // Heap allocation
            "Panic", // Recoverable failure
            "Async", // Asynchronous operations
            "GPU",   // GPU kernel launch, device memory
            "Prob",  // Probabilistic computation
            "Div",   // Potential divergence
        ];

        for name in builtin_effects {
            let def_id = self.fresh_def_id();
            let _ = self.define_type(name.to_string(), def_id);
            self.symbols.insert(
                def_id,
                Symbol {
                    def_id,
                    name: name.to_string(),
                    kind: DefKind::Effect,
                    node_id: NodeId(0),
                    span: Span::default(),
                    parent: None,
                },
            );
        }

        // Built-in functions
        let builtin_functions = [
            "print",     // Print without newline
            "println",   // Print with newline
            "assert",    // Runtime assertion
            "assert_eq", // Assert equality
            "panic",     // Abort with message
            "dbg",       // Debug print
            "format",    // String formatting
            // Utility functions
            "len",         // Get length of array/string/tuple
            "type_of",     // Get type name as string
            "read_line",   // Read line from stdin
            "parse_int",   // Parse string to int
            "parse_float", // Parse string to float
            "to_string",   // Convert to string
            // Math functions
            "sqrt",  // Square root
            "abs",   // Absolute value
            "sin",   // Sine
            "cos",   // Cosine
            "tan",   // Tangent
            "exp",   // e^x
            "log",   // Natural logarithm
            "pow",   // Power
            "floor", // Floor
            "ceil",  // Ceiling
            "round", // Round
            "min",   // Minimum
            "max",   // Maximum
            // Linear algebra constructors
            "vec2", // vec2(x, y)
            "vec3", // vec3(x, y, z)
            "vec4", // vec4(x, y, z, w)
            "mat2", // mat2(col0...) - 2x2 matrix
            "mat3", // mat3(col0...) - 3x3 matrix
            "mat4", // mat4(col0...) - 4x4 matrix
            "quat", // quat(x, y, z, w) - quaternion
            // Vector operations
            "dot",
            "cross",
            "normalize",
            "length",
            "length_squared",
            // Quaternion operations
            "quat_mul",
            "quat_conj",
            "quat_inv",
            "quat_normalize",
            "quat_identity",
            // Matrix operations
            "mat_mul",
            "transpose",
            "inverse",
            "determinant",
            // Interpolation
            "lerp",
            "slerp",
            // Conversions
            "quat_to_euler",
            "euler_to_quat",
            "quat_to_mat3",
            "quat_to_mat4",
            "mat3_to_quat",
            // Quaternion Embeddings (Knowledge Graph) - arXiv:1904.10281
            "hamilton_product",     // Hamilton product for quaternion embeddings
            "quat_rotate_vec",      // Rotate vector by quaternion
            "quat_score",           // Score triple (head, relation, tail)
            "quat_embed_init",      // Initialize quaternion embedding
            "quat_normalize_embed", // Normalize to unit quaternion
            "quat_inner_product",   // Inner product of two quaternion embeddings
            // Automatic Differentiation (Forward-mode via dual numbers)
            "dual",       // Constructor: dual(value, derivative) -> dual
            "dual_value", // Extract value component: dual_value(d) -> f64
            "dual_deriv", // Extract derivative component: dual_deriv(d) -> f64
            "grad",       // Compute gradient: grad(f, x) -> f64
            "jacobian",   // Compute Jacobian matrix: jacobian(f, x) -> [[f64]]
            "hessian",    // Compute Hessian matrix: hessian(f, x) -> [[f64]]
            // Uncertainty propagation
            "uncertain",       // Constructor: uncertain(value, error) -> uncertain<f64>
            "uncertain_value", // Extract value: uncertain_value(u) -> f64
            "uncertain_error", // Extract error: uncertain_error(u) -> f64
            "weighted_mean",   // Weighted mean of uncertain values
            // ODE Solvers
            "ode_solve", // Solve ODE: ode_solve(f, y0, t_span, method) -> ODESolution
            "ode_euler", // Euler solver
            "ode_rk4",   // RK4 solver
            "ode_rk45",  // Dormand-Prince adaptive solver
            // Probabilistic programming (Prob effect)
            "sample",  // Sample from distribution: sample(dist) -> f64
            "observe", // Condition on observation: observe(dist, value)
            "infer",   // Run inference: infer(model, method) -> Trace
            // Probability distributions
            "Normal",      // Normal(mean, std)
            "Uniform",     // Uniform(low, high)
            "Beta",        // Beta(alpha, beta)
            "Gamma",       // Gamma(shape, rate)
            "Exponential", // Exponential(rate)
            "Bernoulli",   // Bernoulli(p)
            "Poisson",     // Poisson(lambda)
            // Tensor operations
            "tensor",    // Create tensor from data
            "zeros",     // Zero-filled tensor
            "ones",      // One-filled tensor
            "eye",       // Identity matrix
            "linspace",  // Linearly spaced values
            "arange",    // Range of values
            "reshape",   // Reshape tensor
            "transpose", // Transpose tensor
            "matmul",    // Matrix multiplication
            // Symbolic computation
            "symbolic",       // Create symbolic variable
            "simplify",       // Simplify expression
            "expand",         // Expand expression
            "factor",         // Factor expression
            "diff",           // Symbolic differentiation
            "integrate_sym",  // Symbolic integration
            "solve_symbolic", // Solve equation symbolically
            "compile_expr",   // Compile expression to function
            "substitute",     // Substitute variable with value
            // Causal inference (Causal effect)
            "do_intervention", // Intervention: do(model, var, value)
            "counterfactual",  // Counterfactual query
            "intervene",       // Apply intervention
            "identify",        // Identify causal effect
            "ate",             // Average treatment effect
            "backdoor_adjust", // Backdoor adjustment
            // Model discovery (SINDy)
            "discover",          // Discover model from data
            "sindy",             // SINDy algorithm
            "build_library",     // Build function library
            "poly_library",      // Polynomial library
            "sparse_regression", // Sparse regression
            // FFI / Raw pointer operations
            "null_ptr",          // Create null const pointer
            "null_mut",          // Create null mut pointer
            "is_null",           // Check if pointer is null
            "ptr_eq",            // Compare two pointers
            "ptr_addr",          // Get address as integer
            "ptr_from_addr",     // Create const pointer from address
            "ptr_from_addr_mut", // Create mut pointer from address
            "ptr_offset",        // Offset pointer by bytes
            "ptr_add",           // Add elements to pointer
            "ptr_sub",           // Subtract elements from pointer
            "ptr_diff",          // Difference between pointers
            "as_const",          // Cast *mut to *const
            "as_mut",            // Cast *const to *mut (unsafe)
            "size_of",           // Get size of type
            "align_of",          // Get alignment of type
        ];

        for name in builtin_functions {
            let def_id = self.fresh_def_id();
            let _ = self.define(name.to_string(), def_id);
            self.symbols.insert(
                def_id,
                Symbol {
                    def_id,
                    name: name.to_string(),
                    kind: DefKind::BuiltinFunction,
                    node_id: NodeId(0),
                    span: Span::default(),
                    parent: None,
                },
            );
        }

        // Built-in enum variants (Option::None, Option::Some, Result::Ok, Result::Err)
        let builtin_variants = ["None", "Some", "Ok", "Err"];

        for name in builtin_variants {
            let def_id = self.fresh_def_id();
            let _ = self.define(name.to_string(), def_id);
            self.symbols.insert(
                def_id,
                Symbol {
                    def_id,
                    name: name.to_string(),
                    kind: DefKind::Variant,
                    node_id: NodeId(0),
                    span: Span::default(),
                    parent: None,
                },
            );
        }
    }

    /// Generate fresh DefId
    pub fn fresh_def_id(&mut self) -> DefId {
        let id = DefId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Push new scope
    pub fn push_scope(&mut self, kind: ScopeKind, parent_def: Option<DefId>) {
        self.scopes.push(Scope::new(kind, parent_def));
    }

    /// Pop scope
    pub fn pop_scope(&mut self) -> Option<Scope> {
        self.scopes.pop()
    }

    /// Define a value name in current scope
    pub fn define(&mut self, name: String, def_id: DefId) -> Result<(), String> {
        if let Some(scope) = self.scopes.last_mut() {
            if scope.names.contains_key(&name) {
                return Err(format!("Duplicate definition: {}", name));
            }
            scope.names.insert(name, def_id);
            Ok(())
        } else {
            Err("No scope".to_string())
        }
    }

    /// Define a type name in current scope
    pub fn define_type(&mut self, name: String, def_id: DefId) -> Result<(), String> {
        if let Some(scope) = self.scopes.last_mut() {
            if scope.types.contains_key(&name) {
                return Err(format!("Duplicate type: {}", name));
            }
            scope.types.insert(name, def_id);
            Ok(())
        } else {
            Err("No scope".to_string())
        }
    }

    /// Look up a value name
    pub fn lookup(&self, name: &str) -> Option<DefId> {
        for scope in self.scopes.iter().rev() {
            if let Some(&def_id) = scope.names.get(name) {
                return Some(def_id);
            }
        }
        None
    }

    /// Look up a type name
    pub fn lookup_type(&self, name: &str) -> Option<DefId> {
        for scope in self.scopes.iter().rev() {
            if let Some(&def_id) = scope.types.get(name) {
                return Some(def_id);
            }
        }
        None
    }

    /// Get symbol by DefId
    pub fn get(&self, def_id: DefId) -> Option<&Symbol> {
        self.symbols.get(&def_id)
    }

    /// Insert symbol
    pub fn insert(&mut self, symbol: Symbol) {
        let def_id = symbol.def_id;
        let node_id = symbol.node_id;
        self.node_to_def.insert(node_id, def_id);
        self.symbols.insert(def_id, symbol);
    }

    /// Record a reference from NodeId to DefId
    pub fn record_ref(&mut self, node_id: NodeId, def_id: DefId) {
        self.node_to_ref.insert(node_id, def_id);
    }

    /// Get DefId for a definition node
    pub fn def_for_node(&self, node_id: NodeId) -> Option<DefId> {
        self.node_to_def.get(&node_id).copied()
    }

    /// Get DefId for a reference node
    pub fn ref_for_node(&self, node_id: NodeId) -> Option<DefId> {
        self.node_to_ref.get(&node_id).copied()
    }

    /// Current scope depth
    pub fn depth(&self) -> usize {
        self.scopes.len()
    }

    /// Check if we're at module level
    pub fn at_module_level(&self) -> bool {
        self.scopes.len() == 1
    }

    /// Get all symbols
    pub fn all_symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.symbols.values()
    }

    /// Get all type names from all scopes
    pub fn all_type_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for scope in &self.scopes {
            for name in scope.types.keys() {
                if !names.contains(name) {
                    names.push(name.clone());
                }
            }
        }
        names
    }

    // ==================== MODULE METHODS ====================

    /// Enter a module scope
    pub fn enter_module(&mut self, id: ModuleId) {
        if !self.modules.contains_key(&id) {
            let parent = if id.is_root() {
                None
            } else {
                Some(self.current_module.clone())
            };
            self.modules
                .insert(id.clone(), ModuleScope::new(id.clone(), parent));

            // Register as child of current module
            if let Some(parent_mod) = self.modules.get_mut(&self.current_module) {
                if let Some(name) = id.path.last() {
                    parent_mod.add_child(name.clone(), id.clone());
                }
            }
        }
        self.current_module = id;
    }

    /// Exit current module, return to parent
    pub fn exit_module(&mut self) {
        if let Some(scope) = self.modules.get(&self.current_module) {
            if let Some(parent) = &scope.parent {
                self.current_module = parent.clone();
            }
        }
    }

    /// Get current module ID
    pub fn current_module(&self) -> &ModuleId {
        &self.current_module
    }

    /// Get module by ID
    pub fn get_module(&self, id: &ModuleId) -> Option<&ModuleScope> {
        self.modules.get(id)
    }

    /// Get mutable module by ID
    pub fn get_module_mut(&mut self, id: &ModuleId) -> Option<&mut ModuleScope> {
        self.modules.get_mut(id)
    }

    /// Define a value in the current module with visibility
    pub fn define_in_module(&mut self, name: String, def_id: DefId, visibility: Visibility) {
        if let Some(module) = self.modules.get_mut(&self.current_module) {
            module.define(name, def_id, visibility);
        }
    }

    /// Define a type in the current module with visibility
    pub fn define_type_in_module(&mut self, name: String, def_id: DefId, visibility: Visibility) {
        if let Some(module) = self.modules.get_mut(&self.current_module) {
            module.define_type(name, def_id, visibility);
        }
    }

    /// Resolve a qualified path (e.g., ["std", "collections", "HashMap"])
    /// Returns the DefId if found, respecting visibility from the requesting module
    pub fn resolve_path(&self, path: &[String], from_module: &ModuleId) -> Option<DefId> {
        if path.is_empty() {
            return None;
        }

        if path.len() == 1 {
            // Unqualified name - search current module, then imports, then builtins
            return self.resolve_unqualified(&path[0], from_module);
        }

        // Qualified path - walk the module tree
        let mut current = ModuleId::root();

        for (i, segment) in path.iter().enumerate() {
            if i == path.len() - 1 {
                // Last segment is the item name
                let module = self.modules.get(&current)?;
                let same_module = &current == from_module;
                return module.lookup(segment, same_module);
            } else {
                // Intermediate segment is a module name
                let module = self.modules.get(&current)?;
                current = module.children.get(segment)?.clone();
            }
        }

        None
    }

    /// Resolve a qualified type path
    pub fn resolve_type_path(&self, path: &[String], from_module: &ModuleId) -> Option<DefId> {
        if path.is_empty() {
            return None;
        }

        if path.len() == 1 {
            // Check current module first
            if let Some(module) = self.modules.get(from_module) {
                if let Some(def_id) = module.lookup_type(&path[0], true) {
                    return Some(def_id);
                }
            }
            // Fall back to lexical scope lookup
            return self.lookup_type(&path[0]);
        }

        // Qualified path
        let mut current = ModuleId::root();

        for (i, segment) in path.iter().enumerate() {
            if i == path.len() - 1 {
                let module = self.modules.get(&current)?;
                let same_module = &current == from_module;
                return module.lookup_type(segment, same_module);
            } else {
                let module = self.modules.get(&current)?;
                current = module.children.get(segment)?.clone();
            }
        }

        None
    }

    /// Resolve an unqualified name in a module context
    fn resolve_unqualified(&self, name: &str, from_module: &ModuleId) -> Option<DefId> {
        // 1. Check current module's definitions
        if let Some(module) = self.modules.get(from_module) {
            if let Some(def_id) = module.lookup(name, true) {
                return Some(def_id);
            }

            // 2. Check imports
            if let Some((source_mod, orig_name)) = module.imports.get(name) {
                // Resolve from source module
                if let Some(source_module) = self.modules.get(source_mod) {
                    return source_module.lookup(orig_name, false);
                }
            }
        }

        // 3. Fall back to lexical scope (for builtins, etc.)
        self.lookup(name)
    }

    /// Process an import into the current module
    pub fn process_import(
        &mut self,
        source_path: &[String],
        items: Option<&[ImportItem]>,
        is_reexport: bool,
    ) -> Result<(), String> {
        let source_module_id = ModuleId::new(source_path.to_vec());
        let current = self.current_module.clone();

        // Get source module
        let source_module = self
            .modules
            .get(&source_module_id)
            .ok_or_else(|| format!("Module not found: {}", source_path.join("::")))?
            .clone();

        match items {
            None => {
                // Import entire module - make it accessible by its name
                let module_name = source_path.last().cloned().unwrap_or_default();
                if let Some(current_mod) = self.modules.get_mut(&current) {
                    current_mod.add_child(module_name, source_module_id);
                }
            }
            Some(items) => {
                for item in items {
                    if item.is_glob {
                        // Glob import: import all public items
                        let public_items: Vec<_> = source_module
                            .public_items()
                            .map(|(n, d)| (n.clone(), *d))
                            .collect();

                        if let Some(current_mod) = self.modules.get_mut(&current) {
                            for (name, def_id) in public_items {
                                if is_reexport {
                                    current_mod.add_reexport(
                                        name.clone(),
                                        source_module_id.clone(),
                                        name,
                                        def_id,
                                    );
                                } else {
                                    current_mod.add_import(
                                        name.clone(),
                                        source_module_id.clone(),
                                        name,
                                    );
                                }
                            }
                        }
                    } else {
                        // Selective import
                        let local_name = item.alias.clone().unwrap_or_else(|| item.name.clone());
                        let orig_name = &item.name;

                        // Verify the item exists and is public
                        let def_id = source_module
                            .lookup(orig_name, false)
                            .or_else(|| source_module.lookup_type(orig_name, false))
                            .ok_or_else(|| {
                                format!(
                                    "Item `{}` not found or not public in module `{}`",
                                    orig_name,
                                    source_path.join("::")
                                )
                            })?;

                        if let Some(current_mod) = self.modules.get_mut(&current) {
                            if is_reexport {
                                current_mod.add_reexport(
                                    local_name,
                                    source_module_id.clone(),
                                    orig_name.clone(),
                                    def_id,
                                );
                            } else {
                                current_mod.add_import(
                                    local_name,
                                    source_module_id.clone(),
                                    orig_name.clone(),
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get all module IDs
    pub fn all_modules(&self) -> impl Iterator<Item = &ModuleId> {
        self.modules.keys()
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_nesting() {
        let mut table = SymbolTable::new();

        let def1 = table.fresh_def_id();
        table.define("x".into(), def1).unwrap();

        table.push_scope(ScopeKind::Block, None);
        let def2 = table.fresh_def_id();
        table.define("y".into(), def2).unwrap();

        // Both visible
        assert!(table.lookup("x").is_some());
        assert!(table.lookup("y").is_some());

        table.pop_scope();

        // Only x visible
        assert!(table.lookup("x").is_some());
        assert!(table.lookup("y").is_none());
    }

    #[test]
    fn test_shadowing() {
        let mut table = SymbolTable::new();

        let def1 = table.fresh_def_id();
        table.define("x".into(), def1).unwrap();

        table.push_scope(ScopeKind::Block, None);
        let def2 = table.fresh_def_id();
        table.define("x".into(), def2).unwrap(); // Shadow

        assert_eq!(table.lookup("x"), Some(def2));

        table.pop_scope();
        assert_eq!(table.lookup("x"), Some(def1));
    }

    #[test]
    fn test_builtin_types() {
        let table = SymbolTable::new();

        // Built-in types should be available
        assert!(table.lookup_type("i32").is_some());
        assert!(table.lookup_type("bool").is_some());
        assert!(table.lookup_type("String").is_some());

        // Unknown type should not exist
        assert!(table.lookup_type("FooBar").is_none());
    }
}
