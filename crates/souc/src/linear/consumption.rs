//! Resource consumption tracking
//!
//! Tracks the consumption state of linear resources to ensure:
//! - Linear resources are used exactly once
//! - Affine resources are used at most once
//! - Relevant resources are used at least once
//! - Unrestricted resources can be used any number of times

use std::collections::HashMap;
use std::fmt;

use super::modality::Modality;

/// State of a resource
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceState {
    /// Resource is available (not yet consumed)
    Available,
    /// Resource has been consumed once
    Consumed,
    /// Resource has been consumed multiple times
    MultiplyConsumed,
    /// Resource was explicitly dropped (for affine)
    Dropped,
}

impl ResourceState {
    /// Transition to consumed state
    pub fn consume(self) -> Self {
        match self {
            ResourceState::Available => ResourceState::Consumed,
            ResourceState::Consumed | ResourceState::MultiplyConsumed => {
                ResourceState::MultiplyConsumed
            }
            ResourceState::Dropped => ResourceState::MultiplyConsumed,
        }
    }

    /// Transition to dropped state
    pub fn drop(self) -> Self {
        match self {
            ResourceState::Available => ResourceState::Dropped,
            other => other,
        }
    }

    /// Check if the resource can be consumed
    pub fn can_consume(self, modality: Modality) -> bool {
        match (self, modality) {
            (ResourceState::Available, _) => true,
            (ResourceState::Consumed, Modality::Relevant | Modality::Unrestricted) => true,
            (ResourceState::MultiplyConsumed, Modality::Unrestricted) => true,
            _ => false,
        }
    }

    /// Check if the resource can be dropped
    pub fn can_drop(self, modality: Modality) -> bool {
        match (self, modality) {
            (ResourceState::Available, Modality::Affine | Modality::Unrestricted) => true,
            _ => false,
        }
    }
}

impl fmt::Display for ResourceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceState::Available => write!(f, "available"),
            ResourceState::Consumed => write!(f, "consumed"),
            ResourceState::MultiplyConsumed => write!(f, "multiply-consumed"),
            ResourceState::Dropped => write!(f, "dropped"),
        }
    }
}

/// Error from consumption tracking
#[derive(Clone, Debug, thiserror::Error)]
pub enum ConsumptionError {
    #[error("Resource '{name}' has already been consumed")]
    AlreadyConsumed { name: String },

    #[error("Linear resource '{name}' was not consumed")]
    NotConsumed { name: String },

    #[error("Linear resource '{name}' was consumed multiple times")]
    MultipleConsumption { name: String },

    #[error("Cannot drop linear resource '{name}'")]
    CannotDrop { name: String },

    #[error("Relevant resource '{name}' was never used")]
    RelevantUnused { name: String },

    #[error("Resource '{name}' not found")]
    NotFound { name: String },

    #[error("Split error: {0}")]
    SplitError(String),
}

/// Tracks consumption of a single resource
#[derive(Clone, Debug)]
pub struct ResourceTracker {
    /// Name of the resource
    pub name: String,
    /// Modality of the resource
    pub modality: Modality,
    /// Current state
    pub state: ResourceState,
}

impl ResourceTracker {
    /// Create a new tracker
    pub fn new(name: impl Into<String>, modality: Modality) -> Self {
        Self {
            name: name.into(),
            modality,
            state: ResourceState::Available,
        }
    }

    /// Try to consume the resource
    pub fn consume(&mut self) -> Result<(), ConsumptionError> {
        if !self.state.can_consume(self.modality) {
            return Err(ConsumptionError::AlreadyConsumed {
                name: self.name.clone(),
            });
        }
        self.state = self.state.consume();
        Ok(())
    }

    /// Try to drop the resource
    pub fn drop_resource(&mut self) -> Result<(), ConsumptionError> {
        if !self.state.can_drop(self.modality) {
            return Err(ConsumptionError::CannotDrop {
                name: self.name.clone(),
            });
        }
        self.state = self.state.drop();
        Ok(())
    }

    /// Check if the resource is in a valid final state
    pub fn check_final(&self) -> Result<(), ConsumptionError> {
        match (self.modality, self.state) {
            // Linear: must be consumed exactly once
            (Modality::Linear, ResourceState::Consumed) => Ok(()),
            (Modality::Linear, ResourceState::Available) => Err(ConsumptionError::NotConsumed {
                name: self.name.clone(),
            }),
            (Modality::Linear, ResourceState::MultiplyConsumed) => {
                Err(ConsumptionError::MultipleConsumption {
                    name: self.name.clone(),
                })
            }
            (Modality::Linear, ResourceState::Dropped) => Err(ConsumptionError::CannotDrop {
                name: self.name.clone(),
            }),

            // Affine: consumed once or dropped
            (Modality::Affine, ResourceState::Consumed | ResourceState::Dropped) => Ok(()),
            (Modality::Affine, ResourceState::Available) => Ok(()), // Can leave unused
            (Modality::Affine, ResourceState::MultiplyConsumed) => {
                Err(ConsumptionError::MultipleConsumption {
                    name: self.name.clone(),
                })
            }

            // Relevant: must be consumed at least once
            (Modality::Relevant, ResourceState::Consumed | ResourceState::MultiplyConsumed) => {
                Ok(())
            }
            (Modality::Relevant, ResourceState::Available | ResourceState::Dropped) => {
                Err(ConsumptionError::RelevantUnused {
                    name: self.name.clone(),
                })
            }

            // Unrestricted: anything goes
            (Modality::Unrestricted, _) => Ok(()),
        }
    }

    /// Check if resource is still available
    pub fn is_available(&self) -> bool {
        self.state == ResourceState::Available
            || (self.modality.allows_contraction()
                && matches!(
                    self.state,
                    ResourceState::Consumed | ResourceState::MultiplyConsumed
                ))
    }
}

/// Tracks consumption of multiple resources
#[derive(Clone, Debug, Default)]
pub struct ConsumptionTracker {
    /// All tracked resources
    resources: HashMap<String, ResourceTracker>,
}

impl ConsumptionTracker {
    /// Create a new tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a resource to track
    pub fn add(&mut self, name: impl Into<String>, modality: Modality) {
        let name = name.into();
        self.resources
            .insert(name.clone(), ResourceTracker::new(name, modality));
    }

    /// Add a linear resource
    pub fn add_linear(&mut self, name: impl Into<String>) {
        self.add(name, Modality::Linear);
    }

    /// Add an affine resource
    pub fn add_affine(&mut self, name: impl Into<String>) {
        self.add(name, Modality::Affine);
    }

    /// Add a relevant resource
    pub fn add_relevant(&mut self, name: impl Into<String>) {
        self.add(name, Modality::Relevant);
    }

    /// Add an unrestricted resource
    pub fn add_unrestricted(&mut self, name: impl Into<String>) {
        self.add(name, Modality::Unrestricted);
    }

    /// Consume a resource
    pub fn consume(&mut self, name: &str) -> Result<(), ConsumptionError> {
        let tracker = self
            .resources
            .get_mut(name)
            .ok_or_else(|| ConsumptionError::NotFound {
                name: name.to_string(),
            })?;
        tracker.consume()
    }

    /// Drop a resource
    pub fn drop_resource(&mut self, name: &str) -> Result<(), ConsumptionError> {
        let tracker = self
            .resources
            .get_mut(name)
            .ok_or_else(|| ConsumptionError::NotFound {
                name: name.to_string(),
            })?;
        tracker.drop_resource()
    }

    /// Check all resources are in valid final states
    pub fn check_all_final(&self) -> Result<(), Vec<ConsumptionError>> {
        let errors: Vec<_> = self
            .resources
            .values()
            .filter_map(|r| r.check_final().err())
            .collect();

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get the state of a resource
    pub fn state(&self, name: &str) -> Option<ResourceState> {
        self.resources.get(name).map(|r| r.state)
    }

    /// Check if a resource is available
    pub fn is_available(&self, name: &str) -> bool {
        self.resources.get(name).is_some_and(|r| r.is_available())
    }

    /// Get all available resources
    pub fn available_resources(&self) -> Vec<&str> {
        self.resources
            .iter()
            .filter(|(_, r)| r.is_available())
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get all resources that must still be consumed
    pub fn must_consume(&self) -> Vec<&str> {
        self.resources
            .iter()
            .filter(|(_, r)| {
                r.state == ResourceState::Available
                    && matches!(r.modality, Modality::Linear | Modality::Relevant)
            })
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Split tracker for tensor introduction
    ///
    /// Linear/affine/relevant resources go to exactly one side.
    /// Unrestricted resources are copied to both.
    pub fn split(self, left_names: &[&str]) -> Result<(Self, Self), ConsumptionError> {
        let mut left = ConsumptionTracker::new();
        let mut right = ConsumptionTracker::new();

        let left_set: std::collections::HashSet<_> = left_names.iter().copied().collect();

        for (name, tracker) in self.resources {
            match tracker.modality {
                Modality::Unrestricted => {
                    // Copy to both
                    left.resources.insert(name.clone(), tracker.clone());
                    right.resources.insert(name, tracker);
                }
                _ => {
                    if left_set.contains(name.as_str()) {
                        left.resources.insert(name, tracker);
                    } else {
                        right.resources.insert(name, tracker);
                    }
                }
            }
        }

        Ok((left, right))
    }

    /// Merge trackers after additive elimination
    pub fn merge(left: Self, right: Self) -> Result<Self, ConsumptionError> {
        let mut result = ConsumptionTracker::new();

        // Combine usage from both branches
        for (name, left_tracker) in left.resources {
            if let Some(right_tracker) = right.resources.get(&name) {
                // Resource exists in both - combine states
                let combined_state = match (left_tracker.state, right_tracker.state) {
                    (ResourceState::Consumed, ResourceState::Consumed) => {
                        ResourceState::MultiplyConsumed
                    }
                    (ResourceState::Consumed, ResourceState::Available)
                    | (ResourceState::Available, ResourceState::Consumed) => {
                        ResourceState::Consumed
                    }
                    (ResourceState::Available, ResourceState::Available) => {
                        ResourceState::Available
                    }
                    _ => ResourceState::MultiplyConsumed,
                };

                result.resources.insert(
                    name.clone(),
                    ResourceTracker {
                        name,
                        modality: left_tracker.modality,
                        state: combined_state,
                    },
                );
            } else {
                result.resources.insert(name, left_tracker);
            }
        }

        // Add any resources only in right
        for (name, right_tracker) in right.resources {
            result.resources.entry(name).or_insert(right_tracker);
        }

        Ok(result)
    }

    /// Number of tracked resources
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_consumption() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_linear("x");

        // Can consume once
        assert!(tracker.consume("x").is_ok());

        // Cannot consume again
        assert!(tracker.consume("x").is_err());

        // Final state is valid
        assert!(tracker.check_all_final().is_ok());
    }

    #[test]
    fn test_linear_not_consumed() {
        let tracker = ConsumptionTracker::new();
        let mut tracker = tracker;
        tracker.add_linear("x");

        // Not consuming is an error
        let result = tracker.check_all_final();
        assert!(result.is_err());
    }

    #[test]
    fn test_affine_can_drop() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_affine("x");

        // Can leave unused
        assert!(tracker.check_all_final().is_ok());
    }

    #[test]
    fn test_affine_cannot_reuse() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_affine("x");

        tracker.consume("x").unwrap();

        // Cannot consume again
        assert!(tracker.consume("x").is_err());
    }

    #[test]
    fn test_relevant_must_use() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_relevant("x");

        // Not using is an error
        assert!(tracker.check_all_final().is_err());

        // Using makes it valid
        tracker.consume("x").unwrap();
        assert!(tracker.check_all_final().is_ok());
    }

    #[test]
    fn test_relevant_can_reuse() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_relevant("x");

        tracker.consume("x").unwrap();
        tracker.consume("x").unwrap();

        // Multiple use is fine for relevant
        assert!(tracker.check_all_final().is_ok());
    }

    #[test]
    fn test_unrestricted_any_usage() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_unrestricted("x");

        // Can use any number of times
        tracker.consume("x").unwrap();
        tracker.consume("x").unwrap();
        tracker.consume("x").unwrap();

        assert!(tracker.check_all_final().is_ok());

        // Can also not use at all
        let mut tracker2 = ConsumptionTracker::new();
        tracker2.add_unrestricted("y");
        assert!(tracker2.check_all_final().is_ok());
    }

    #[test]
    fn test_split_unrestricted() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_unrestricted("x");

        let (left, right) = tracker.split(&[]).unwrap();

        // Both have x
        assert!(left.is_available("x"));
        assert!(right.is_available("x"));
    }

    #[test]
    fn test_split_linear() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_linear("x");
        tracker.add_linear("y");

        let (left, right) = tracker.split(&["x"]).unwrap();

        // x goes to left, y goes to right
        assert!(left.is_available("x"));
        assert!(!left.is_available("y"));
        assert!(!right.is_available("x"));
        assert!(right.is_available("y"));
    }

    #[test]
    fn test_must_consume() {
        let mut tracker = ConsumptionTracker::new();
        tracker.add_linear("x");
        tracker.add_affine("y");
        tracker.add_relevant("z");

        let must = tracker.must_consume();
        assert!(must.contains(&"x"));
        assert!(!must.contains(&"y")); // Affine doesn't need to be consumed
        assert!(must.contains(&"z"));
    }

    #[test]
    fn test_resource_state_transitions() {
        let state = ResourceState::Available;

        // Consume once
        let state = state.consume();
        assert_eq!(state, ResourceState::Consumed);

        // Consume again
        let state = state.consume();
        assert_eq!(state, ResourceState::MultiplyConsumed);
    }
}
