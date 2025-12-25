//! Linear typing context
//!
//! Tracks variable bindings with their modalities and usage counts.
//! Supports context splitting for tensor introduction and merging for
//! additive elimination.

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::LinearError;
use super::linear_types::LinearType;
use super::modality::Modality;

/// Usage count for tracking how many times a variable has been used
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum UsageCount {
    /// Not yet used
    #[default]
    Zero,
    /// Used exactly once
    One,
    /// Used more than once
    Many,
}

impl UsageCount {
    /// Increment the usage count
    pub fn increment(self) -> Self {
        match self {
            UsageCount::Zero => UsageCount::One,
            UsageCount::One | UsageCount::Many => UsageCount::Many,
        }
    }

    /// Check if used at least once
    pub fn is_used(self) -> bool {
        !matches!(self, UsageCount::Zero)
    }

    /// Check if used more than once
    pub fn is_overused(self) -> bool {
        matches!(self, UsageCount::Many)
    }

    /// Add two usage counts
    pub fn add(self, other: UsageCount) -> UsageCount {
        match (self, other) {
            (UsageCount::Zero, x) | (x, UsageCount::Zero) => x,
            _ => UsageCount::Many,
        }
    }
}

impl fmt::Display for UsageCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UsageCount::Zero => write!(f, "0"),
            UsageCount::One => write!(f, "1"),
            UsageCount::Many => write!(f, "ω"),
        }
    }
}

/// A binding in the linear context
#[derive(Clone, Debug)]
pub struct LinearBinding {
    /// Variable name
    pub name: String,
    /// Type of the variable
    pub typ: LinearType,
    /// Usage modality
    pub modality: Modality,
    /// How many times this variable has been used
    pub usage: UsageCount,
    /// Source location (for error messages)
    pub span: Option<(usize, usize)>,
}

impl LinearBinding {
    /// Create a new binding
    pub fn new(name: impl Into<String>, typ: LinearType, modality: Modality) -> Self {
        Self {
            name: name.into(),
            typ,
            modality,
            usage: UsageCount::Zero,
            span: None,
        }
    }

    /// Create with span
    pub fn with_span(mut self, start: usize, end: usize) -> Self {
        self.span = Some((start, end));
        self
    }

    /// Check if usage is valid for the modality
    pub fn check_usage(&self) -> Result<(), LinearError> {
        match (self.modality, self.usage) {
            // Linear: must use exactly once
            (Modality::Linear, UsageCount::Zero) => {
                Err(LinearError::UnusedLinear(self.name.clone()))
            }
            (Modality::Linear, UsageCount::Many) => {
                Err(LinearError::OverusedLinear(self.name.clone()))
            }

            // Affine: can use 0 or 1 time
            (Modality::Affine, UsageCount::Many) => {
                Err(LinearError::OverusedAffine(self.name.clone()))
            }

            // Relevant: must use at least once
            (Modality::Relevant, UsageCount::Zero) => {
                Err(LinearError::UnusedRelevant(self.name.clone()))
            }

            // Unrestricted: any usage is fine
            _ => Ok(()),
        }
    }

    /// Mark as used (increment usage count)
    pub fn mark_used(&mut self) {
        self.usage = self.usage.increment();
    }

    /// Check if can be used again
    pub fn can_use(&self) -> bool {
        match (self.modality, self.usage) {
            (Modality::Linear, UsageCount::Zero) => true,
            (Modality::Linear, _) => false,
            (Modality::Affine, UsageCount::Zero) => true,
            (Modality::Affine, _) => false,
            (Modality::Relevant, _) => true,
            (Modality::Unrestricted, _) => true,
        }
    }
}

/// Linear typing context
///
/// Tracks all bindings with their modalities and usage counts.
/// Supports operations required for linear type checking:
/// - Split: For tensor introduction (disjoint union of contexts)
/// - Merge: For additive elimination (identical contexts)
/// - Check exhaustion: Verify all linear/relevant bindings are properly used
#[derive(Clone, Debug, Default)]
pub struct LinearContext {
    /// All bindings in scope
    bindings: Vec<LinearBinding>,
    /// Index by name for fast lookup
    index: HashMap<String, usize>,
}

impl LinearContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a binding with the given modality
    pub fn add(&mut self, name: impl Into<String>, typ: LinearType, modality: Modality) {
        let name = name.into();
        let idx = self.bindings.len();
        self.bindings
            .push(LinearBinding::new(name.clone(), typ, modality));
        self.index.insert(name, idx);
    }

    /// Add a linear binding
    pub fn add_linear(&mut self, name: impl Into<String>, typ: LinearType) {
        self.add(name, typ, Modality::Linear);
    }

    /// Add an affine binding
    pub fn add_affine(&mut self, name: impl Into<String>, typ: LinearType) {
        self.add(name, typ, Modality::Affine);
    }

    /// Add a relevant binding
    pub fn add_relevant(&mut self, name: impl Into<String>, typ: LinearType) {
        self.add(name, typ, Modality::Relevant);
    }

    /// Add an unrestricted binding
    pub fn add_unrestricted(&mut self, name: impl Into<String>, typ: LinearType) {
        self.add(name, typ, Modality::Unrestricted);
    }

    /// Look up a binding by name
    pub fn lookup(&self, name: &str) -> Option<&LinearBinding> {
        self.index.get(name).map(|&idx| &self.bindings[idx])
    }

    /// Look up a binding by name (mutable)
    pub fn lookup_mut(&mut self, name: &str) -> Option<&mut LinearBinding> {
        self.index.get(name).map(|&idx| &mut self.bindings[idx])
    }

    /// Use a variable (mark as used and return its type)
    pub fn use_var(&mut self, name: &str) -> Result<&LinearType, LinearError> {
        let idx = self
            .index
            .get(name)
            .ok_or_else(|| LinearError::UnboundVariable(name.to_string()))?;

        let binding = &mut self.bindings[*idx];

        // Check if we can still use this variable
        if !binding.can_use() {
            match binding.modality {
                Modality::Linear => return Err(LinearError::OverusedLinear(name.to_string())),
                Modality::Affine => return Err(LinearError::OverusedAffine(name.to_string())),
                _ => {}
            }
        }

        binding.mark_used();
        Ok(&self.bindings[*idx].typ)
    }

    /// Get all binding names
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.bindings.iter().map(|b| b.name.as_str())
    }

    /// Get all bindings
    pub fn bindings(&self) -> impl Iterator<Item = &LinearBinding> {
        self.bindings.iter()
    }

    /// Check that all bindings satisfy their modality constraints
    pub fn check_exhausted(&self) -> Result<(), LinearError> {
        for binding in &self.bindings {
            binding.check_usage()?;
        }
        Ok(())
    }

    /// Split the context for tensor introduction
    ///
    /// Linear/affine/relevant bindings go to exactly one side.
    /// Unrestricted bindings are copied to both sides.
    ///
    /// This returns a "split context" that allows specifying which bindings
    /// go to which side.
    pub fn split(self) -> ContextSplit {
        ContextSplit::new(self)
    }

    /// Merge two contexts for with/plus elimination
    ///
    /// The contexts must have the same bindings (for & elimination).
    /// Usage counts are combined.
    pub fn merge(left: LinearContext, right: LinearContext) -> Result<LinearContext, LinearError> {
        // For & elimination, contexts should be identical in structure
        if left.bindings.len() != right.bindings.len() {
            return Err(LinearError::ContextSplitFailed(
                "Context sizes don't match for merge".to_string(),
            ));
        }

        let mut result = LinearContext::new();

        for (l, r) in left.bindings.into_iter().zip(right.bindings.into_iter()) {
            if l.name != r.name || l.modality != r.modality {
                return Err(LinearError::ContextSplitFailed(
                    "Context structures don't match for merge".to_string(),
                ));
            }

            let mut merged = l;
            merged.usage = merged.usage.add(r.usage);
            let name = merged.name.clone();
            let idx = result.bindings.len();
            result.bindings.push(merged);
            result.index.insert(name, idx);
        }

        Ok(result)
    }

    /// Get only unrestricted bindings
    pub fn unrestricted_only(&self) -> LinearContext {
        let mut result = LinearContext::new();
        for binding in &self.bindings {
            if binding.modality == Modality::Unrestricted {
                result.add(
                    binding.name.clone(),
                    binding.typ.clone(),
                    Modality::Unrestricted,
                );
            }
        }
        result
    }

    /// Check if all bindings are unrestricted
    pub fn all_unrestricted(&self) -> bool {
        self.bindings
            .iter()
            .all(|b| b.modality == Modality::Unrestricted)
    }

    /// Get names of unused linear bindings
    pub fn unused_linear(&self) -> Vec<&str> {
        self.bindings
            .iter()
            .filter(|b| b.modality == Modality::Linear && b.usage == UsageCount::Zero)
            .map(|b| b.name.as_str())
            .collect()
    }

    /// Extend with bindings from another context
    pub fn extend(&mut self, other: LinearContext) {
        for binding in other.bindings {
            let name = binding.name.clone();
            let idx = self.bindings.len();
            self.bindings.push(binding);
            self.index.insert(name, idx);
        }
    }

    /// Create a child scope (for let bindings)
    pub fn child_scope(&self) -> LinearContext {
        self.clone()
    }

    /// Number of bindings
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

/// Helper for splitting a context
pub struct ContextSplit {
    /// Original context
    original: LinearContext,
    /// Which bindings go to the left context
    left_names: HashSet<String>,
    /// Which bindings go to the right context
    right_names: HashSet<String>,
}

impl ContextSplit {
    fn new(original: LinearContext) -> Self {
        Self {
            original,
            left_names: HashSet::new(),
            right_names: HashSet::new(),
        }
    }

    /// Assign a binding to the left context
    pub fn assign_left(&mut self, name: &str) -> &mut Self {
        self.left_names.insert(name.to_string());
        self.right_names.remove(name);
        self
    }

    /// Assign a binding to the right context
    pub fn assign_right(&mut self, name: &str) -> &mut Self {
        self.right_names.insert(name.to_string());
        self.left_names.remove(name);
        self
    }

    /// Complete the split
    ///
    /// Linear/affine/relevant bindings must be assigned to exactly one side.
    /// Unrestricted bindings are copied to both.
    pub fn complete(self) -> Result<(LinearContext, LinearContext), LinearError> {
        let mut left = LinearContext::new();
        let mut right = LinearContext::new();

        for binding in self.original.bindings {
            match binding.modality {
                Modality::Unrestricted => {
                    // Copy to both
                    left.add(
                        binding.name.clone(),
                        binding.typ.clone(),
                        Modality::Unrestricted,
                    );
                    right.add(binding.name, binding.typ, Modality::Unrestricted);
                }
                _ => {
                    // Must be assigned to exactly one
                    let in_left = self.left_names.contains(&binding.name);
                    let in_right = self.right_names.contains(&binding.name);

                    if in_left && in_right {
                        return Err(LinearError::ContextSplitFailed(format!(
                            "Binding '{}' assigned to both sides",
                            binding.name
                        )));
                    }

                    if in_left {
                        left.add(binding.name, binding.typ, binding.modality);
                    } else if in_right {
                        right.add(binding.name, binding.typ, binding.modality);
                    } else {
                        // Unassigned - error for linear, OK for affine
                        if binding.modality.must_use() {
                            return Err(LinearError::ContextSplitFailed(format!(
                                "Binding '{}' not assigned to any side",
                                binding.name
                            )));
                        }
                        // Affine can be discarded
                    }
                }
            }
        }

        Ok((left, right))
    }

    /// Auto-assign all unassigned bindings to the left
    pub fn auto_assign_left(mut self) -> Self {
        for binding in &self.original.bindings {
            if !self.left_names.contains(&binding.name) && !self.right_names.contains(&binding.name)
            {
                self.left_names.insert(binding.name.clone());
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_count_increment() {
        assert_eq!(UsageCount::Zero.increment(), UsageCount::One);
        assert_eq!(UsageCount::One.increment(), UsageCount::Many);
        assert_eq!(UsageCount::Many.increment(), UsageCount::Many);
    }

    #[test]
    fn test_binding_check_linear() {
        let mut binding = LinearBinding::new("x", LinearType::One, Modality::Linear);

        // Unused - error
        assert!(binding.check_usage().is_err());

        // Used once - ok
        binding.mark_used();
        assert!(binding.check_usage().is_ok());

        // Used twice - error
        binding.mark_used();
        assert!(binding.check_usage().is_err());
    }

    #[test]
    fn test_binding_check_affine() {
        let mut binding = LinearBinding::new("x", LinearType::One, Modality::Affine);

        // Unused - ok
        assert!(binding.check_usage().is_ok());

        // Used once - ok
        binding.mark_used();
        assert!(binding.check_usage().is_ok());

        // Used twice - error
        binding.mark_used();
        assert!(binding.check_usage().is_err());
    }

    #[test]
    fn test_binding_check_relevant() {
        let mut binding = LinearBinding::new("x", LinearType::One, Modality::Relevant);

        // Unused - error
        assert!(binding.check_usage().is_err());

        // Used once - ok
        binding.mark_used();
        assert!(binding.check_usage().is_ok());

        // Used twice - ok
        binding.mark_used();
        assert!(binding.check_usage().is_ok());
    }

    #[test]
    fn test_context_use_var() {
        let mut ctx = LinearContext::new();
        ctx.add_linear("x", LinearType::One);

        let typ = ctx.use_var("x").unwrap();
        assert!(matches!(typ, LinearType::One));

        // Using again should fail (linear)
        assert!(ctx.use_var("x").is_err());
    }

    #[test]
    fn test_context_split_unrestricted() {
        let mut ctx = LinearContext::new();
        ctx.add_unrestricted("x", LinearType::One);

        let (left, right) = ctx.split().complete().unwrap();

        // Both should have x
        assert!(left.lookup("x").is_some());
        assert!(right.lookup("x").is_some());
    }

    #[test]
    fn test_context_split_linear() {
        let mut ctx = LinearContext::new();
        ctx.add_linear("x", LinearType::One);
        ctx.add_linear("y", LinearType::One);

        let mut split = ctx.split();
        split.assign_left("x");
        split.assign_right("y");

        let (left, right) = split.complete().unwrap();

        assert!(left.lookup("x").is_some());
        assert!(left.lookup("y").is_none());
        assert!(right.lookup("x").is_none());
        assert!(right.lookup("y").is_some());
    }

    #[test]
    fn test_context_split_fails_unassigned_linear() {
        let mut ctx = LinearContext::new();
        ctx.add_linear("x", LinearType::One);

        // x is not assigned to either side
        let result = ctx.split().complete();
        assert!(result.is_err());
    }

    #[test]
    fn test_context_merge() {
        let mut left = LinearContext::new();
        left.add_linear("x", LinearType::One);

        let mut right = LinearContext::new();
        right.add_linear("x", LinearType::One);

        let merged = LinearContext::merge(left, right).unwrap();
        assert!(merged.lookup("x").is_some());
    }
}
