//! Handler Registry and Factory
//!
//! This module provides centralized management of effect handlers:
//!
//! - `HandlerRegistry`: Stores and retrieves handlers by effect name
//! - `HandlerFactory`: Trait for creating handlers (enables custom factories)
//! - `DefaultHandlerFactory`: Creates all built-in handlers
//!
//! # Example
//!
//! ```ignore
//! use sounio::effects::handlers::{HandlerRegistry, DefaultHandlerFactory};
//!
//! // Create registry with all default handlers
//! let registry = HandlerRegistry::with_defaults();
//!
//! // Look up a handler
//! if let Some(handler) = registry.get("IO") {
//!     println!("Found handler: {}", handler.handler_name());
//! }
//!
//! // List all registered effects
//! for effect in registry.effect_names() {
//!     println!("Registered: {}", effect);
//! }
//! ```

use super::{
    AllocHandler, AsyncHandler, DivHandler, EpistemicHandler, ExnHandler, GpuHandler, IOHandler,
    MutHandler, NetworkHandler, PanicHandler, ProbHandler, SensorHandler,
};
use crate::effects::handler_capability::HandlerCapability;
use std::collections::HashMap;
use std::sync::Arc;

/// A registry that stores effect handlers by effect name.
///
/// The registry provides O(1) lookup of handlers by effect name and supports
/// dynamic registration of custom handlers.
#[derive(Default)]
pub struct HandlerRegistry {
    handlers: HashMap<String, Arc<dyn HandlerCapability + Send + Sync>>,
}

impl HandlerRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Create a registry populated with all default handlers.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        DefaultHandlerFactory.register_all(&mut registry);
        registry
    }

    /// Create a registry from a custom factory.
    pub fn from_factory<F: HandlerFactory>(factory: &F) -> Self {
        let mut registry = Self::new();
        factory.register_all(&mut registry);
        registry
    }

    /// Register a handler for an effect.
    ///
    /// If a handler for this effect already exists, it is replaced.
    pub fn register<H>(&mut self, handler: H)
    where
        H: HandlerCapability + Send + Sync + 'static,
    {
        let effect_name = handler.effect_name().to_string();
        self.handlers.insert(effect_name, Arc::new(handler));
    }

    /// Register a handler wrapped in Arc.
    pub fn register_arc(&mut self, handler: Arc<dyn HandlerCapability + Send + Sync>) {
        let effect_name = handler.effect_name().to_string();
        self.handlers.insert(effect_name, handler);
    }

    /// Get a handler by effect name.
    pub fn get(&self, effect_name: &str) -> Option<&Arc<dyn HandlerCapability + Send + Sync>> {
        self.handlers.get(effect_name)
    }

    /// Check if a handler is registered for the given effect.
    pub fn has(&self, effect_name: &str) -> bool {
        self.handlers.contains_key(effect_name)
    }

    /// Remove a handler by effect name.
    ///
    /// Returns the removed handler if it existed.
    pub fn remove(
        &mut self,
        effect_name: &str,
    ) -> Option<Arc<dyn HandlerCapability + Send + Sync>> {
        self.handlers.remove(effect_name)
    }

    /// Get an iterator over all registered effect names.
    pub fn effect_names(&self) -> impl Iterator<Item = &str> {
        self.handlers.keys().map(|s| s.as_str())
    }

    /// Get an iterator over all registered handlers.
    pub fn handlers(&self) -> impl Iterator<Item = &Arc<dyn HandlerCapability + Send + Sync>> {
        self.handlers.values()
    }

    /// Get the number of registered handlers.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Clear all registered handlers.
    pub fn clear(&mut self) {
        self.handlers.clear();
    }

    /// Merge another registry into this one.
    ///
    /// Handlers from `other` will overwrite existing handlers with the same effect name.
    pub fn merge(&mut self, other: HandlerRegistry) {
        self.handlers.extend(other.handlers);
    }
}

impl std::fmt::Debug for HandlerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandlerRegistry")
            .field("effects", &self.handlers.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Trait for creating and registering effect handlers.
///
/// Implement this trait to provide custom handler factories, e.g., for testing
/// with mock handlers or for different runtime configurations.
pub trait HandlerFactory {
    /// Register all handlers from this factory into the registry.
    fn register_all(&self, registry: &mut HandlerRegistry);

    /// Create a handler for a specific effect, if supported.
    fn create(&self, effect_name: &str) -> Option<Box<dyn HandlerCapability + Send + Sync>>;

    /// List all effect names this factory can create handlers for.
    fn supported_effects(&self) -> Vec<&'static str>;
}

/// The default factory that creates all built-in handlers.
///
/// This factory creates handlers for all 12 standard effects:
/// - Alloc, Async, Div, Epistemic, Exn, GPU
/// - IO, Mut, Network, Panic, Prob, Sensor
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultHandlerFactory;

impl DefaultHandlerFactory {
    /// All effect names supported by the default factory.
    pub const EFFECTS: &'static [&'static str] = &[
        "Alloc",
        "Async",
        "Div",
        "Epistemic",
        "Exn",
        "GPU",
        "IO",
        "Mut",
        "Network",
        "Panic",
        "Prob",
        "Sensor",
    ];
}

impl HandlerFactory for DefaultHandlerFactory {
    fn register_all(&self, registry: &mut HandlerRegistry) {
        registry.register(AllocHandler::new());
        registry.register(AsyncHandler::new());
        registry.register(DivHandler::new());
        registry.register(EpistemicHandler::new());
        registry.register(ExnHandler::new());
        registry.register(GpuHandler::new());
        registry.register(IOHandler::new());
        registry.register(MutHandler::new());
        registry.register(NetworkHandler::new());
        registry.register(PanicHandler::new());
        registry.register(ProbHandler::new());
        registry.register(SensorHandler::new());
    }

    fn create(&self, effect_name: &str) -> Option<Box<dyn HandlerCapability + Send + Sync>> {
        match effect_name {
            "Alloc" => Some(Box::new(AllocHandler::new())),
            "Async" => Some(Box::new(AsyncHandler::new())),
            "Div" => Some(Box::new(DivHandler::new())),
            "Epistemic" => Some(Box::new(EpistemicHandler::new())),
            "Exn" => Some(Box::new(ExnHandler::new())),
            "GPU" => Some(Box::new(GpuHandler::new())),
            "IO" => Some(Box::new(IOHandler::new())),
            "Mut" => Some(Box::new(MutHandler::new())),
            "Network" => Some(Box::new(NetworkHandler::new())),
            "Panic" => Some(Box::new(PanicHandler::new())),
            "Prob" => Some(Box::new(ProbHandler::new())),
            "Sensor" => Some(Box::new(SensorHandler::new())),
            _ => None,
        }
    }

    fn supported_effects(&self) -> Vec<&'static str> {
        Self::EFFECTS.to_vec()
    }
}

/// Builder for creating a custom handler registry.
///
/// Provides a fluent API for configuring which handlers to include.
///
/// # Example
///
/// ```ignore
/// let registry = HandlerRegistryBuilder::new()
///     .with_io()
///     .with_mut()
///     .with_custom(MyCustomHandler::new())
///     .build();
/// ```
#[derive(Default)]
pub struct HandlerRegistryBuilder {
    registry: HandlerRegistry,
}

impl HandlerRegistryBuilder {
    /// Create a new builder with an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start with all default handlers.
    pub fn with_defaults(mut self) -> Self {
        DefaultHandlerFactory.register_all(&mut self.registry);
        self
    }

    /// Add the Alloc handler.
    pub fn with_alloc(mut self) -> Self {
        self.registry.register(AllocHandler::new());
        self
    }

    /// Add the Async handler.
    pub fn with_async(mut self) -> Self {
        self.registry.register(AsyncHandler::new());
        self
    }

    /// Add the Div handler.
    pub fn with_div(mut self) -> Self {
        self.registry.register(DivHandler::new());
        self
    }

    /// Add the Epistemic handler.
    pub fn with_epistemic(mut self) -> Self {
        self.registry.register(EpistemicHandler::new());
        self
    }

    /// Add the Exn handler.
    pub fn with_exn(mut self) -> Self {
        self.registry.register(ExnHandler::new());
        self
    }

    /// Add the Gpu handler.
    pub fn with_gpu(mut self) -> Self {
        self.registry.register(GpuHandler::new());
        self
    }

    /// Add the IO handler.
    pub fn with_io(mut self) -> Self {
        self.registry.register(IOHandler::new());
        self
    }

    /// Add the Mut handler.
    pub fn with_mut(mut self) -> Self {
        self.registry.register(MutHandler::new());
        self
    }

    /// Add the Network handler.
    pub fn with_network(mut self) -> Self {
        self.registry.register(NetworkHandler::new());
        self
    }

    /// Add the Panic handler.
    pub fn with_panic(mut self) -> Self {
        self.registry.register(PanicHandler::new());
        self
    }

    /// Add the Prob handler.
    pub fn with_prob(mut self) -> Self {
        self.registry.register(ProbHandler::new());
        self
    }

    /// Add the Sensor handler.
    pub fn with_sensor(mut self) -> Self {
        self.registry.register(SensorHandler::new());
        self
    }

    /// Add a custom handler.
    pub fn with_custom<H>(mut self, handler: H) -> Self
    where
        H: HandlerCapability + Send + Sync + 'static,
    {
        self.registry.register(handler);
        self
    }

    /// Build the registry.
    pub fn build(self) -> HandlerRegistry {
        self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::handler_capability::{Continuation, HandlerState};

    #[test]
    fn test_empty_registry() {
        let registry = HandlerRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(!registry.has("IO"));
    }

    #[test]
    fn test_with_defaults() {
        let registry = HandlerRegistry::with_defaults();
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 12);

        // Check all effects are registered
        for effect in DefaultHandlerFactory::EFFECTS {
            assert!(registry.has(effect), "Missing handler for {}", effect);
        }
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = HandlerRegistry::new();
        registry.register(IOHandler::new());

        assert!(registry.has("IO"));
        let handler = registry.get("IO").expect("IO handler should exist");
        assert_eq!(handler.effect_name(), "IO");
        assert_eq!(handler.handler_name(), "IOHandler");
    }

    #[test]
    fn test_remove() {
        let mut registry = HandlerRegistry::with_defaults();
        assert!(registry.has("IO"));

        let removed = registry.remove("IO");
        assert!(removed.is_some());
        assert!(!registry.has("IO"));
        assert_eq!(registry.len(), 11);
    }

    #[test]
    fn test_effect_names() {
        let registry = HandlerRegistry::with_defaults();
        let names: Vec<&str> = registry.effect_names().collect();
        assert_eq!(names.len(), 12);
        assert!(names.contains(&"IO"));
        assert!(names.contains(&"Mut"));
        assert!(names.contains(&"Sensor"));
    }

    #[test]
    fn test_clear() {
        let mut registry = HandlerRegistry::with_defaults();
        assert!(!registry.is_empty());

        registry.clear();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_merge() {
        let mut registry1 = HandlerRegistry::new();
        registry1.register(IOHandler::new());

        let mut registry2 = HandlerRegistry::new();
        registry2.register(MutHandler::new());

        registry1.merge(registry2);
        assert!(registry1.has("IO"));
        assert!(registry1.has("Mut"));
        assert_eq!(registry1.len(), 2);
    }

    #[test]
    fn test_default_factory_create() {
        let factory = DefaultHandlerFactory;

        for effect in DefaultHandlerFactory::EFFECTS {
            let handler = factory.create(effect);
            assert!(
                handler.is_some(),
                "Factory should create handler for {}",
                effect
            );
            assert_eq!(handler.unwrap().effect_name(), *effect);
        }

        assert!(factory.create("Unknown").is_none());
    }

    #[test]
    fn test_default_factory_supported_effects() {
        let factory = DefaultHandlerFactory;
        let effects = factory.supported_effects();
        assert_eq!(effects.len(), 12);
    }

    #[test]
    fn test_builder_empty() {
        let registry = HandlerRegistryBuilder::new().build();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_builder_with_defaults() {
        let registry = HandlerRegistryBuilder::new().with_defaults().build();
        assert_eq!(registry.len(), 12);
    }

    #[test]
    fn test_builder_selective() {
        let registry = HandlerRegistryBuilder::new()
            .with_io()
            .with_mut()
            .with_panic()
            .build();

        assert_eq!(registry.len(), 3);
        assert!(registry.has("IO"));
        assert!(registry.has("Mut"));
        assert!(registry.has("Panic"));
        assert!(!registry.has("Alloc"));
    }

    #[test]
    fn test_handler_invocation_through_registry() {
        let registry = HandlerRegistry::with_defaults();

        let io_handler = registry.get("IO").expect("IO handler");
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        // Test that handler can be invoked through registry
        let result = io_handler.handle(
            "print",
            &[crate::interp::Value::String("test".into())],
            cont,
            &mut state,
        );

        // Should succeed (Resume with Unit)
        match result {
            crate::effects::handler_capability::HandlerResult::Resume(_) => {}
            _ => panic!("Expected Resume result"),
        }
    }

    #[test]
    fn test_debug_impl() {
        let registry = HandlerRegistry::with_defaults();
        let debug_str = format!("{:?}", registry);
        assert!(debug_str.contains("HandlerRegistry"));
        assert!(debug_str.contains("effects"));
    }

    #[test]
    fn test_replace_handler() {
        let mut registry = HandlerRegistry::new();
        registry.register(IOHandler::new());

        // Register again - should replace
        registry.register(IOHandler::new());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_from_factory() {
        let registry = HandlerRegistry::from_factory(&DefaultHandlerFactory);
        assert_eq!(registry.len(), 12);
    }
}
