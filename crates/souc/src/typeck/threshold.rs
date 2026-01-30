//! Threshold Resolution for Semantic Distance
//!
//! Determines appropriate thresholds based on:
//! - Explicit #[compat(...)] annotations
//! - Context (parameter position, return type, etc.)
//! - Module-level defaults
//! - Global defaults

use std::collections::HashMap;

use crate::common::Span;

/// Named threshold levels
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ThresholdLevel {
    /// Exact match required (0.0)
    Exact,
    /// Very strict (0.05)
    Strict,
    /// Default tolerance (0.15)
    #[default]
    Default,
    /// Relaxed matching (0.25)
    Loose,
    /// Very permissive (0.40)
    Permissive,
    /// Custom value
    Custom(f32),
}

impl ThresholdLevel {
    /// Convert to numeric threshold
    pub fn as_f32(&self) -> f32 {
        match self {
            ThresholdLevel::Exact => 0.0,
            ThresholdLevel::Strict => 0.05,
            ThresholdLevel::Default => 0.15,
            ThresholdLevel::Loose => 0.25,
            ThresholdLevel::Permissive => 0.40,
            ThresholdLevel::Custom(v) => *v,
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "exact" => Some(ThresholdLevel::Exact),
            "strict" => Some(ThresholdLevel::Strict),
            "default" => Some(ThresholdLevel::Default),
            "loose" => Some(ThresholdLevel::Loose),
            "permissive" => Some(ThresholdLevel::Permissive),
            _ => {
                // Try to parse as float
                s.parse::<f32>().ok().map(ThresholdLevel::Custom)
            }
        }
    }

    /// Check if a distance is within this threshold
    pub fn allows(&self, distance: f32) -> bool {
        distance <= self.as_f32()
    }
}

/// Resolved threshold with provenance
#[derive(Debug, Clone)]
pub struct ResolvedThreshold {
    /// The threshold value
    pub level: ThresholdLevel,
    /// Where this threshold came from
    pub source: ThresholdSource,
    /// Original span if from annotation
    pub annotation_span: Option<Span>,
}

impl ResolvedThreshold {
    pub fn as_f32(&self) -> f32 {
        self.level.as_f32()
    }

    pub fn allows(&self, distance: f32) -> bool {
        self.level.allows(distance)
    }
}

/// Source of a threshold value
#[derive(Debug, Clone, PartialEq)]
pub enum ThresholdSource {
    /// From #[compat(...)] annotation on item
    ItemAnnotation,
    /// From #[compat(...)] annotation on parameter
    ParameterAnnotation,
    /// From #[compat(...)] annotation on type parameter
    TypeParameterAnnotation,
    /// From module-level default
    ModuleDefault,
    /// From global configuration
    GlobalDefault,
    /// Inferred from context
    Inferred(ThresholdContext),
}

/// Context for threshold inference
#[derive(Debug, Clone, PartialEq)]
pub enum ThresholdContext {
    /// Function parameter (stricter by default)
    FunctionParameter,
    /// Return type (moderate)
    ReturnType,
    /// Local variable assignment (looser)
    LocalAssignment,
    /// Field access (strict)
    FieldAccess,
    /// Method call receiver (strict)
    MethodReceiver,
    /// Generic type argument (varies)
    GenericArgument,
    /// Match pattern (exact)
    MatchPattern,
}

impl ThresholdContext {
    /// Get inferred threshold for this context
    pub fn default_threshold(&self) -> ThresholdLevel {
        match self {
            ThresholdContext::FunctionParameter => ThresholdLevel::Default,
            ThresholdContext::ReturnType => ThresholdLevel::Strict,
            ThresholdContext::LocalAssignment => ThresholdLevel::Loose,
            ThresholdContext::FieldAccess => ThresholdLevel::Strict,
            ThresholdContext::MethodReceiver => ThresholdLevel::Strict,
            ThresholdContext::GenericArgument => ThresholdLevel::Default,
            ThresholdContext::MatchPattern => ThresholdLevel::Exact,
        }
    }
}

/// Resolves thresholds for type checking
pub struct ThresholdResolver {
    /// Module-level defaults (module path -> threshold)
    module_defaults: HashMap<String, ThresholdLevel>,
    /// Item-level overrides (item path -> threshold)
    item_overrides: HashMap<String, ThresholdLevel>,
    /// Parameter-level overrides (item path + param index -> threshold)
    param_overrides: HashMap<(String, usize), ThresholdLevel>,
    /// Global default threshold
    global_default: ThresholdLevel,
}

impl ThresholdResolver {
    pub fn new() -> Self {
        Self {
            module_defaults: HashMap::new(),
            item_overrides: HashMap::new(),
            param_overrides: HashMap::new(),
            global_default: ThresholdLevel::Default,
        }
    }

    /// Set global default threshold
    pub fn with_global_default(mut self, level: ThresholdLevel) -> Self {
        self.global_default = level;
        self
    }

    /// Register module-level default
    pub fn register_module_default(&mut self, module_path: &str, level: ThresholdLevel) {
        self.module_defaults.insert(module_path.to_string(), level);
    }

    /// Register item-level override
    pub fn register_item_override(&mut self, item_path: &str, level: ThresholdLevel) {
        self.item_overrides.insert(item_path.to_string(), level);
    }

    /// Register parameter-level override
    pub fn register_param_override(
        &mut self,
        item_path: &str,
        param_index: usize,
        level: ThresholdLevel,
    ) {
        self.param_overrides
            .insert((item_path.to_string(), param_index), level);
    }

    /// Register threshold from a compat attribute string
    /// Formats supported: "0.10", "strict", "loose", etc.
    pub fn register_from_compat_attr(
        &mut self,
        item_path: &str,
        attr_value: &str,
    ) -> Option<ThresholdLevel> {
        if let Some(level) = ThresholdLevel::from_str(attr_value) {
            self.register_item_override(item_path, level);
            Some(level)
        } else {
            None
        }
    }

    /// Resolve threshold for a specific location
    pub fn resolve(
        &self,
        module_path: &str,
        item_path: Option<&str>,
        param_index: Option<usize>,
        context: Option<ThresholdContext>,
    ) -> ResolvedThreshold {
        // Check parameter-level override first
        if let (Some(item), Some(idx)) = (item_path, param_index)
            && let Some(level) = self.param_overrides.get(&(item.to_string(), idx))
        {
            return ResolvedThreshold {
                level: *level,
                source: ThresholdSource::ParameterAnnotation,
                annotation_span: None,
            };
        }

        // Check item-level override
        if let Some(item) = item_path
            && let Some(level) = self.item_overrides.get(item)
        {
            return ResolvedThreshold {
                level: *level,
                source: ThresholdSource::ItemAnnotation,
                annotation_span: None,
            };
        }

        // Check module-level default
        if let Some(level) = self.module_defaults.get(module_path) {
            return ResolvedThreshold {
                level: *level,
                source: ThresholdSource::ModuleDefault,
                annotation_span: None,
            };
        }

        // Use context-based inference
        if let Some(ctx) = context {
            return ResolvedThreshold {
                level: ctx.default_threshold(),
                source: ThresholdSource::Inferred(ctx),
                annotation_span: None,
            };
        }

        // Fall back to global default
        ResolvedThreshold {
            level: self.global_default,
            source: ThresholdSource::GlobalDefault,
            annotation_span: None,
        }
    }

    /// Resolve threshold for function parameter
    pub fn resolve_for_parameter(
        &self,
        module_path: &str,
        function_path: &str,
        param_index: usize,
    ) -> ResolvedThreshold {
        self.resolve(
            module_path,
            Some(function_path),
            Some(param_index),
            Some(ThresholdContext::FunctionParameter),
        )
    }

    /// Resolve threshold for return type
    pub fn resolve_for_return(&self, module_path: &str, function_path: &str) -> ResolvedThreshold {
        self.resolve(
            module_path,
            Some(function_path),
            None,
            Some(ThresholdContext::ReturnType),
        )
    }

    /// Resolve threshold for local assignment
    pub fn resolve_for_local(&self, module_path: &str) -> ResolvedThreshold {
        self.resolve(
            module_path,
            None,
            None,
            Some(ThresholdContext::LocalAssignment),
        )
    }

    /// Resolve threshold for match pattern
    pub fn resolve_for_match(&self, module_path: &str) -> ResolvedThreshold {
        self.resolve(
            module_path,
            None,
            None,
            Some(ThresholdContext::MatchPattern),
        )
    }
}

impl Default for ThresholdResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for threshold configuration
pub struct ThresholdConfig {
    resolver: ThresholdResolver,
}

impl ThresholdConfig {
    pub fn new() -> Self {
        Self {
            resolver: ThresholdResolver::new(),
        }
    }

    /// Set global default
    pub fn global_default(mut self, level: ThresholdLevel) -> Self {
        self.resolver.global_default = level;
        self
    }

    /// Add module default
    pub fn module_default(mut self, module: &str, level: ThresholdLevel) -> Self {
        self.resolver.register_module_default(module, level);
        self
    }

    /// Build the resolver
    pub fn build(self) -> ThresholdResolver {
        self.resolver
    }
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_levels() {
        assert_eq!(ThresholdLevel::Exact.as_f32(), 0.0);
        assert_eq!(ThresholdLevel::Strict.as_f32(), 0.05);
        assert_eq!(ThresholdLevel::Default.as_f32(), 0.15);
        assert_eq!(ThresholdLevel::Loose.as_f32(), 0.25);
        assert_eq!(ThresholdLevel::Permissive.as_f32(), 0.40);
        assert_eq!(ThresholdLevel::Custom(0.33).as_f32(), 0.33);
    }

    #[test]
    fn test_threshold_parsing() {
        assert_eq!(
            ThresholdLevel::from_str("exact"),
            Some(ThresholdLevel::Exact)
        );
        assert_eq!(
            ThresholdLevel::from_str("STRICT"),
            Some(ThresholdLevel::Strict)
        );
        assert_eq!(
            ThresholdLevel::from_str("0.20"),
            Some(ThresholdLevel::Custom(0.20))
        );
        assert_eq!(ThresholdLevel::from_str("invalid"), None);
    }

    #[test]
    fn test_threshold_allows() {
        assert!(ThresholdLevel::Default.allows(0.10));
        assert!(ThresholdLevel::Default.allows(0.15));
        assert!(!ThresholdLevel::Default.allows(0.20));
        assert!(ThresholdLevel::Exact.allows(0.0));
        assert!(!ThresholdLevel::Exact.allows(0.01));
    }

    #[test]
    fn test_resolver_hierarchy() {
        let mut resolver = ThresholdResolver::new();
        resolver.register_module_default("mymod", ThresholdLevel::Loose);
        resolver.register_item_override("mymod::func", ThresholdLevel::Strict);
        resolver.register_param_override("mymod::func", 0, ThresholdLevel::Exact);

        // Global fallback
        let global = resolver.resolve("other", None, None, None);
        assert_eq!(global.level, ThresholdLevel::Default);

        // Module default
        let module = resolver.resolve("mymod", None, None, None);
        assert_eq!(module.level, ThresholdLevel::Loose);

        // Item override
        let item = resolver.resolve("mymod", Some("mymod::func"), None, None);
        assert_eq!(item.level, ThresholdLevel::Strict);

        // Parameter override
        let param = resolver.resolve("mymod", Some("mymod::func"), Some(0), None);
        assert_eq!(param.level, ThresholdLevel::Exact);
    }

    #[test]
    fn test_context_inference() {
        let resolver = ThresholdResolver::new();

        let ret = resolver.resolve(
            "mod",
            Some("mod::f"),
            None,
            Some(ThresholdContext::ReturnType),
        );
        assert_eq!(ret.level, ThresholdLevel::Strict);

        let local = resolver.resolve("mod", None, None, Some(ThresholdContext::LocalAssignment));
        assert_eq!(local.level, ThresholdLevel::Loose);

        let pattern = resolver.resolve("mod", None, None, Some(ThresholdContext::MatchPattern));
        assert_eq!(pattern.level, ThresholdLevel::Exact);
    }
}
