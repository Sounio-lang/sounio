//! Linearity types for effect handlers
//!
//! Defines linearity constraints for continuations:
//! - **Linear (ExactlyOnce)**: Continuation must be resumed exactly once
//! - **Affine (OnceOrNever)**: Continuation may be resumed at most once
//! - **Unrestricted (MultiShot)**: Continuation may be resumed multiple times
//!
//! These constraints ensure safe resource management and prevent double-free
//! errors in effect handlers.
//!
//! # Research Background
//!
//! Based on:
//! - "Soundly Handling Linearity" (POPL 2024) - Linearity in effect systems
//! - "Retrofitting Effect Handlers onto OCaml" (PLDI 2021) - One-shot vs multi-shot
//! - "Affect: An OCaml Library for Algebraic Effects" (POPL 2025) - Effect linearity

/// Linearity constraint for continuations
///
/// Determines how many times a continuation can be resumed:
///
/// - `ExactlyOnce`: Linear - must be resumed exactly once (checked at runtime)
/// - `OnceOrNever`: Affine - may be resumed zero or one times
/// - `MultiShot`: Unrestricted - may be resumed multiple times
///
/// # Examples
///
/// ```ignore
/// // Linear (ExactlyOnce): File handle must be closed
/// effect File {
///     fn close(handle: FileHandle) with Linearity::ExactlyOnce;
/// }
///
/// // Affine (OnceOrNever): Optional cleanup
/// effect Optional {
///     fn maybe_cleanup() with Linearity::OnceOrNever;
/// }
///
/// // Multi-shot (Unrestricted): Non-determinism
/// effect Amb {
///     fn choose(a: T, b: T) -> T with Linearity::MultiShot;
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Linearity {
    /// Affine: may be resumed zero or one times
    ///
    /// Used for optional cleanup or error handling where resumption is not guaranteed.
    OnceOrNever,

    /// Linear: must be resumed exactly once
    ///
    /// Used for resource management (files, locks) where cleanup is mandatory.
    /// Runtime checks ensure the continuation is resumed exactly once.
    ExactlyOnce,

    /// Unrestricted: may be resumed multiple times
    ///
    /// Used for non-determinism, backtracking, or probabilistic effects.
    /// The continuation can be cloned and resumed multiple times with different values.
    MultiShot,
}

impl Linearity {
    /// Check if this linearity allows multiple resumptions
    pub fn is_multi_shot(&self) -> bool {
        matches!(self, Linearity::MultiShot)
    }

    /// Check if this linearity is affine (at most once)
    pub fn is_affine(&self) -> bool {
        matches!(self, Linearity::OnceOrNever)
    }

    /// Check if this linearity is linear (exactly once)
    pub fn is_linear(&self) -> bool {
        matches!(self, Linearity::ExactlyOnce)
    }

    /// Check if resumption is required
    pub fn requires_resumption(&self) -> bool {
        matches!(self, Linearity::ExactlyOnce)
    }

    /// Check if this linearity is compatible with another
    ///
    /// Returns `true` if a continuation with `self` linearity can be used
    /// where `other` linearity is expected.
    pub fn is_compatible_with(&self, other: Linearity) -> bool {
        match (self, other) {
            // Exact match is always compatible
            (a, b) if a == &b => true,
            // Multi-shot can simulate single-shot by using only once
            (Linearity::MultiShot, Linearity::ExactlyOnce) => true,
            (Linearity::MultiShot, Linearity::OnceOrNever) => true,
            // Linear can be used where affine is expected (stronger guarantee)
            (Linearity::ExactlyOnce, Linearity::OnceOrNever) => true,
            // Everything else is incompatible
            _ => false,
        }
    }
}

impl Default for Linearity {
    fn default() -> Self {
        // Default to linear (one-shot) for safety
        Linearity::ExactlyOnce
    }
}

impl std::fmt::Display for Linearity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Linearity::OnceOrNever => write!(f, "affine"),
            Linearity::ExactlyOnce => write!(f, "linear"),
            Linearity::MultiShot => write!(f, "unrestricted"),
        }
    }
}

/// Information about an effect operation's linearity
#[derive(Debug, Clone)]
pub struct EffectOpInfo {
    /// Effect name (e.g., "IO", "State", "Exn")
    pub effect: String,
    /// Operation name (e.g., "read_file", "get", "throw")
    pub operation: String,
    /// Linearity constraint for this operation
    pub linearity: Linearity,
}

impl EffectOpInfo {
    /// Create a new effect operation info
    pub fn new(
        effect: impl Into<String>,
        operation: impl Into<String>,
        linearity: Linearity,
    ) -> Self {
        Self {
            effect: effect.into(),
            operation: operation.into(),
            linearity,
        }
    }

    /// Create an operation with linear (exactly-once) linearity
    pub fn linear(effect: impl Into<String>, operation: impl Into<String>) -> Self {
        Self::new(effect, operation, Linearity::ExactlyOnce)
    }

    /// Create an operation with affine (once-or-never) linearity
    pub fn affine(effect: impl Into<String>, operation: impl Into<String>) -> Self {
        Self::new(effect, operation, Linearity::OnceOrNever)
    }

    /// Create an operation with multi-shot linearity
    pub fn multi_shot(effect: impl Into<String>, operation: impl Into<String>) -> Self {
        Self::new(effect, operation, Linearity::MultiShot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linearity_predicates() {
        assert!(Linearity::MultiShot.is_multi_shot());
        assert!(!Linearity::ExactlyOnce.is_multi_shot());
        assert!(!Linearity::OnceOrNever.is_multi_shot());

        assert!(Linearity::OnceOrNever.is_affine());
        assert!(!Linearity::ExactlyOnce.is_affine());

        assert!(Linearity::ExactlyOnce.is_linear());
        assert!(!Linearity::OnceOrNever.is_linear());

        assert!(Linearity::ExactlyOnce.requires_resumption());
        assert!(!Linearity::OnceOrNever.requires_resumption());
        assert!(!Linearity::MultiShot.requires_resumption());
    }

    #[test]
    fn test_linearity_compatibility() {
        // Exact match
        assert!(Linearity::ExactlyOnce.is_compatible_with(Linearity::ExactlyOnce));
        assert!(Linearity::OnceOrNever.is_compatible_with(Linearity::OnceOrNever));
        assert!(Linearity::MultiShot.is_compatible_with(Linearity::MultiShot));

        // Multi-shot can simulate one-shot
        assert!(Linearity::MultiShot.is_compatible_with(Linearity::ExactlyOnce));
        assert!(Linearity::MultiShot.is_compatible_with(Linearity::OnceOrNever));

        // Linear can be used where affine is expected
        assert!(Linearity::ExactlyOnce.is_compatible_with(Linearity::OnceOrNever));

        // Incompatible
        assert!(!Linearity::OnceOrNever.is_compatible_with(Linearity::ExactlyOnce));
        assert!(!Linearity::ExactlyOnce.is_compatible_with(Linearity::MultiShot));
        assert!(!Linearity::OnceOrNever.is_compatible_with(Linearity::MultiShot));
    }

    #[test]
    fn test_linearity_display() {
        assert_eq!(Linearity::OnceOrNever.to_string(), "affine");
        assert_eq!(Linearity::ExactlyOnce.to_string(), "linear");
        assert_eq!(Linearity::MultiShot.to_string(), "unrestricted");
    }

    #[test]
    fn test_effect_op_info() {
        let op = EffectOpInfo::linear("IO", "read_file");
        assert_eq!(op.effect, "IO");
        assert_eq!(op.operation, "read_file");
        assert_eq!(op.linearity, Linearity::ExactlyOnce);

        let op = EffectOpInfo::affine("Exn", "throw");
        assert_eq!(op.linearity, Linearity::OnceOrNever);

        let op = EffectOpInfo::multi_shot("Amb", "choose");
        assert_eq!(op.linearity, Linearity::MultiShot);
    }
}
