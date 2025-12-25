//! Lexical environment for variable bindings

use std::collections::HashMap;

use super::Value;

/// Lexical scope containing variable bindings
#[derive(Debug, Clone)]
struct Scope {
    bindings: HashMap<String, Value>,
}

impl Scope {
    fn new() -> Self {
        Scope {
            bindings: HashMap::new(),
        }
    }
}

/// Environment for interpreter execution
/// Manages lexical scopes and variable bindings
#[derive(Debug, Clone)]
pub struct Environment {
    /// Stack of scopes (innermost last)
    scopes: Vec<Scope>,
}

impl Environment {
    /// Create a new environment with a global scope
    pub fn new() -> Self {
        Environment {
            scopes: vec![Scope::new()],
        }
    }

    /// Push a new scope
    pub fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    /// Pop the innermost scope
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Define a variable in the current scope
    pub fn define(&mut self, name: String, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.bindings.insert(name, value);
        }
    }

    /// Look up a variable by name
    pub fn get(&self, name: &str) -> Option<Value> {
        // Search from innermost to outermost scope
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.bindings.get(name) {
                return Some(value.clone());
            }
        }
        None
    }

    /// Assign to an existing variable
    pub fn assign(&mut self, name: &str, value: Value) -> bool {
        // Search from innermost to outermost scope
        for scope in self.scopes.iter_mut().rev() {
            if scope.bindings.contains_key(name) {
                scope.bindings.insert(name.to_string(), value);
                return true;
            }
        }
        false
    }

    /// Get all bindings (for capturing closures)
    pub fn capture_all(&self) -> HashMap<String, Value> {
        let mut captured = HashMap::new();
        for scope in &self.scopes {
            for (name, value) in &scope.bindings {
                captured.insert(name.clone(), value.clone());
            }
        }
        captured
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}
