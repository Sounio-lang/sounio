//! Versioning System for Knowledge
//!
//! Implements semantic versioning for epistemic knowledge, allowing tracking
//! of how knowledge evolves through different versions.
//!
//! # Semantic Versioning
//!
//! ```text
//! MAJOR.MINOR.PATCH
//!
//! MAJOR: Paradigm shift or contradiction with previous version
//!        - Guideline changes fundamental recommendation
//!        - Model invalidates previous predictions
//!
//! MINOR: Addition without contradiction
//!        - New indication added
//!        - Extra parameter in model
//!
//! PATCH: Minor correction or clarification
//!        - Typo fixed
//!        - Confidence interval refined
//! ```
//!
//! # Version Relations
//!
//! - `supersedes`: v2.0 makes v1.x obsolete (ε → 0)
//! - `extends`: v1.1 adds to v1.0 without contradiction
//! - `refines`: v1.0.1 improves precision of v1.0.0

use super::knowledge::TemporalKnowledge;
use super::types::{Temporal, Version, VersionInfo, VersionRelation};
use crate::epistemic::composition::{ConfidenceValue, EpistemicValue};
use chrono::{DateTime, Utc};

/// Versioned knowledge with full history
#[derive(Clone, Debug)]
pub struct VersionedKnowledge<T> {
    /// Current version
    pub current: TemporalKnowledge<T>,

    /// Version information
    pub version: VersionInfo,

    /// History of all previous versions
    pub history: Vec<VersionedKnowledge<T>>,
}

impl<T: Clone> VersionedKnowledge<T> {
    /// Create initial version (1.0.0)
    pub fn initial(
        knowledge: EpistemicValue<T>,
        author: impl Into<String>,
        changelog: impl Into<String>,
    ) -> Self {
        let version = Version::initial();

        VersionedKnowledge {
            current: TemporalKnowledge {
                core: knowledge,
                temporal: Temporal::Versioned {
                    version: version.clone(),
                    created: Utc::now(),
                    superseded_by: None,
                },
                history: None,
            },
            version: VersionInfo::new(version, author, changelog, Some(VersionRelation::Initial)),
            history: vec![],
        }
    }

    /// Release new major version (supersedes previous)
    ///
    /// Major versions indicate breaking changes:
    /// - Previous versions are marked as superseded
    /// - Superseded versions have ε = 0
    pub fn major_release(
        &self,
        new_knowledge: EpistemicValue<T>,
        author: impl Into<String>,
        changelog: impl Into<String>,
        supersede_reason: impl Into<String>,
    ) -> Self {
        let new_version = self.version.version.bump_major();

        // Mark current as superseded
        let mut superseded_current = self.current.clone();
        if let Temporal::Versioned {
            ref mut superseded_by,
            ..
        } = superseded_current.temporal
        {
            *superseded_by = Some(new_version.clone());
        }

        // Build new history, marking all prior versions as superseded
        let mut new_history: Vec<VersionedKnowledge<T>> = self
            .history
            .iter()
            .map(|h| {
                let mut superseded = h.clone();
                if let Temporal::Versioned {
                    ref mut superseded_by,
                    ..
                } = superseded.current.temporal
                    && superseded_by.is_none()
                {
                    *superseded_by = Some(new_version.clone());
                }
                superseded
            })
            .collect();
        new_history.push(VersionedKnowledge {
            current: superseded_current,
            version: self.version.clone(),
            history: vec![],
        });

        VersionedKnowledge {
            current: TemporalKnowledge {
                core: new_knowledge,
                temporal: Temporal::Versioned {
                    version: new_version.clone(),
                    created: Utc::now(),
                    superseded_by: None,
                },
                history: None,
            },
            version: VersionInfo::new(
                new_version.clone(),
                author,
                changelog,
                Some(VersionRelation::Supersedes {
                    previous: self.version.version.clone(),
                    reason: supersede_reason.into(),
                }),
            ),
            history: new_history,
        }
    }

    /// Release minor version (extends previous)
    ///
    /// Minor versions add new knowledge without contradicting existing:
    /// - Previous versions remain valid
    /// - New version has additional information
    pub fn minor_release(
        &self,
        new_knowledge: EpistemicValue<T>,
        author: impl Into<String>,
        changelog: impl Into<String>,
        additions: Vec<String>,
    ) -> Self {
        let new_version = self.version.version.bump_minor();

        // Keep history without superseding
        let mut new_history = self.history.clone();
        new_history.push(VersionedKnowledge {
            current: self.current.clone(),
            version: self.version.clone(),
            history: vec![],
        });

        VersionedKnowledge {
            current: TemporalKnowledge {
                core: new_knowledge,
                temporal: Temporal::Versioned {
                    version: new_version.clone(),
                    created: Utc::now(),
                    superseded_by: None,
                },
                history: None,
            },
            version: VersionInfo::new(
                new_version.clone(),
                author,
                changelog,
                Some(VersionRelation::Extends {
                    previous: self.version.version.clone(),
                    additions,
                }),
            ),
            history: new_history,
        }
    }

    /// Release patch version (refines previous)
    ///
    /// Patch versions improve precision without changing semantics:
    /// - Value remains conceptually the same
    /// - Confidence interval may be tighter
    pub fn patch_release(
        &self,
        new_knowledge: EpistemicValue<T>,
        author: impl Into<String>,
        changelog: impl Into<String>,
        improvements: Vec<String>,
    ) -> Self {
        let new_version = self.version.version.bump_patch();

        let mut new_history = self.history.clone();
        new_history.push(VersionedKnowledge {
            current: self.current.clone(),
            version: self.version.clone(),
            history: vec![],
        });

        VersionedKnowledge {
            current: TemporalKnowledge {
                core: new_knowledge,
                temporal: Temporal::Versioned {
                    version: new_version.clone(),
                    created: Utc::now(),
                    superseded_by: None,
                },
                history: None,
            },
            version: VersionInfo::new(
                new_version.clone(),
                author,
                changelog,
                Some(VersionRelation::Refines {
                    previous: self.version.version.clone(),
                    improvements,
                }),
            ),
            history: new_history,
        }
    }

    /// Get the current version number
    pub fn version(&self) -> &Version {
        &self.version.version
    }

    /// Get the current value
    pub fn value(&self) -> &T {
        self.current.value()
    }

    /// Get current confidence
    pub fn confidence(&self) -> ConfidenceValue {
        self.current.current_confidence()
    }

    /// Check if this version is still valid (not superseded)
    pub fn is_valid(&self) -> bool {
        match &self.current.temporal {
            Temporal::Versioned { superseded_by, .. } => superseded_by.is_none(),
            _ => true,
        }
    }

    /// Check if this version is superseded
    pub fn is_superseded(&self) -> bool {
        !self.is_valid()
    }

    /// Get knowledge at a specific version
    pub fn at_version(&self, target: &Version) -> Option<&TemporalKnowledge<T>> {
        if &self.version.version == target {
            Some(&self.current)
        } else {
            self.history
                .iter()
                .find(|h| &h.version.version == target)
                .map(|h| &h.current)
        }
    }

    /// Get all versions
    pub fn all_versions(&self) -> Vec<&Version> {
        let mut versions = vec![&self.version.version];
        for h in &self.history {
            versions.push(&h.version.version);
        }
        versions.sort();
        versions
    }

    /// Get the most recent valid version
    pub fn latest_valid(&self) -> Option<&VersionedKnowledge<T>> {
        if self.is_valid() {
            Some(self)
        } else {
            // Search history for latest valid
            self.history.iter().rev().find(|h| h.is_valid())
        }
    }

    /// Get version history as a timeline
    pub fn timeline(&self) -> Vec<VersionTimeline> {
        let mut timeline = Vec::new();

        // Add current
        timeline.push(VersionTimeline {
            version: self.version.version.clone(),
            created: self.version.created,
            is_current: true,
            is_valid: self.is_valid(),
            relation: self.version.relation_to_previous.clone(),
        });

        // Add history
        for h in self.history.iter().rev() {
            timeline.push(VersionTimeline {
                version: h.version.version.clone(),
                created: h.version.created,
                is_current: false,
                is_valid: h.is_valid(),
                relation: h.version.relation_to_previous.clone(),
            });
        }

        timeline
    }

    /// Compare two versions
    pub fn compare_versions(&self, v1: &Version, v2: &Version) -> Option<VersionComparison<T>> {
        let k1 = self.at_version(v1)?;
        let k2 = self.at_version(v2)?;

        Some(VersionComparison {
            version1: v1.clone(),
            version2: v2.clone(),
            confidence_change: k2.core.confidence().value() - k1.core.confidence().value(),
            time_between: k2.temporal.creation_time().and_then(|t2| {
                k1.temporal
                    .creation_time()
                    .map(|t1| t2.signed_duration_since(t1))
            }),
            _marker: std::marker::PhantomData,
        })
    }
}

/// Timeline entry for a version
#[derive(Clone, Debug)]
pub struct VersionTimeline {
    pub version: Version,
    pub created: DateTime<Utc>,
    pub is_current: bool,
    pub is_valid: bool,
    pub relation: Option<VersionRelation>,
}

/// Comparison between two versions
#[derive(Clone, Debug)]
pub struct VersionComparison<T> {
    pub version1: Version,
    pub version2: Version,
    pub confidence_change: f64,
    pub time_between: Option<chrono::Duration>,
    #[allow(dead_code)]
    _marker: std::marker::PhantomData<T>,
}

impl<T> VersionComparison<T> {
    /// Check if confidence increased
    pub fn confidence_improved(&self) -> bool {
        self.confidence_change > 0.0
    }
}

/// Builder for creating versioned knowledge
pub struct VersionedKnowledgeBuilder<T> {
    value: T,
    confidence: f64,
    author: String,
    changelog: String,
}

impl<T: Clone> VersionedKnowledgeBuilder<T> {
    /// Create a new builder
    pub fn new(value: T) -> Self {
        VersionedKnowledgeBuilder {
            value,
            confidence: 0.9,
            author: "unknown".to_string(),
            changelog: "Initial release".to_string(),
        }
    }

    /// Set confidence
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set author
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.author = author.into();
        self
    }

    /// Set changelog
    pub fn changelog(mut self, changelog: impl Into<String>) -> Self {
        self.changelog = changelog.into();
        self
    }

    /// Build the versioned knowledge
    pub fn build(self) -> VersionedKnowledge<T> {
        VersionedKnowledge::initial(
            EpistemicValue::with_confidence(self.value, self.confidence),
            self.author,
            self.changelog,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_version() {
        let vk: VersionedKnowledge<f64> = VersionedKnowledge::initial(
            EpistemicValue::with_confidence(5.0, 0.90),
            "FDA",
            "Initial dose recommendation",
        );

        assert_eq!(vk.version(), &Version::new(1, 0, 0));
        assert!(vk.is_valid());
        assert_eq!(vk.history.len(), 0);
    }

    #[test]
    fn test_major_release() {
        let v1: VersionedKnowledge<f64> = VersionedKnowledge::initial(
            EpistemicValue::with_confidence(100.0, 0.90),
            "FDA",
            "Initial dose",
        );

        let v2 = v1.major_release(
            EpistemicValue::with_confidence(150.0, 0.95),
            "FDA",
            "Updated based on trial",
            "New trial data",
        );

        assert_eq!(v2.version(), &Version::new(2, 0, 0));
        assert!(v2.is_valid());
        assert_eq!(v2.history.len(), 1);

        // v1 should be superseded
        let v1_in_history = v2.at_version(&Version::new(1, 0, 0)).unwrap();
        assert!(matches!(
            v1_in_history.temporal,
            Temporal::Versioned {
                superseded_by: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn test_minor_release() {
        let v1: VersionedKnowledge<f64> = VersionedKnowledge::initial(
            EpistemicValue::with_confidence(100.0, 0.90),
            "FDA",
            "Initial",
        );

        let v1_1 = v1.minor_release(
            EpistemicValue::with_confidence(100.0, 0.92),
            "FDA",
            "Added renal adjustment",
            vec!["renal_adjustment".to_string()],
        );

        assert_eq!(v1_1.version(), &Version::new(1, 1, 0));
        assert!(v1_1.is_valid());
    }

    #[test]
    fn test_patch_release() {
        let v1: VersionedKnowledge<f64> = VersionedKnowledge::initial(
            EpistemicValue::with_confidence(100.0, 0.90),
            "FDA",
            "Initial",
        );

        let v1_0_1 = v1.patch_release(
            EpistemicValue::with_confidence(100.0, 0.91),
            "FDA",
            "Fixed typo",
            vec!["typo_fix".to_string()],
        );

        assert_eq!(v1_0_1.version(), &Version::new(1, 0, 1));
    }

    #[test]
    fn test_all_versions() {
        let v1: VersionedKnowledge<f64> =
            VersionedKnowledge::initial(EpistemicValue::with_confidence(100.0, 0.90), "FDA", "v1");

        let v2 = v1.major_release(
            EpistemicValue::with_confidence(150.0, 0.95),
            "FDA",
            "v2",
            "Update",
        );

        let v3 = v2.major_release(
            EpistemicValue::with_confidence(175.0, 0.97),
            "FDA",
            "v3",
            "Another update",
        );

        let versions = v3.all_versions();
        assert_eq!(versions.len(), 3);
    }

    #[test]
    fn test_timeline() {
        let v1: VersionedKnowledge<f64> = VersionedKnowledge::initial(
            EpistemicValue::with_confidence(100.0, 0.90),
            "FDA",
            "Initial",
        );

        let v2 = v1.major_release(
            EpistemicValue::with_confidence(150.0, 0.95),
            "FDA",
            "Update",
            "Reason",
        );

        let timeline = v2.timeline();
        assert_eq!(timeline.len(), 2);
        assert!(timeline[0].is_current);
        assert!(!timeline[1].is_current);
    }

    #[test]
    fn test_builder() {
        let vk: VersionedKnowledge<f64> = VersionedKnowledgeBuilder::new(100.0)
            .confidence(0.95)
            .author("Test Author")
            .changelog("Test changelog")
            .build();

        assert_eq!(*vk.value(), 100.0);
        assert!((vk.confidence().value() - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_at_version() {
        let v1: VersionedKnowledge<f64> =
            VersionedKnowledge::initial(EpistemicValue::with_confidence(100.0, 0.90), "FDA", "v1");

        let v2 = v1.major_release(
            EpistemicValue::with_confidence(150.0, 0.95),
            "FDA",
            "v2",
            "Update",
        );

        // Can retrieve v1
        let v1_retrieved = v2.at_version(&Version::new(1, 0, 0));
        assert!(v1_retrieved.is_some());
        assert_eq!(*v1_retrieved.unwrap().value(), 100.0);

        // Can retrieve v2 (current)
        let v2_retrieved = v2.at_version(&Version::new(2, 0, 0));
        assert!(v2_retrieved.is_some());
        assert_eq!(*v2_retrieved.unwrap().value(), 150.0);

        // Cannot retrieve non-existent version
        let v3_retrieved = v2.at_version(&Version::new(3, 0, 0));
        assert!(v3_retrieved.is_none());
    }
}
