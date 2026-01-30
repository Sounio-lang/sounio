//! Epistemic Firewalls - Confidence Boundary Isolation
//!
//! Epistemic firewalls provide mechanisms to control and isolate
//! uncertainty propagation in computations. They allow:
//!
//! 1. **Confidence Reset** - Reset confidence to a specified value at boundaries
//! 2. **Confidence Clamping** - Enforce minimum/maximum confidence thresholds
//! 3. **Model Switching** - Change uncertainty model at boundaries
//! 4. **Audit Points** - Log provenance at critical junctures
//! 5. **Validation Gates** - Block values that don't meet criteria
//!
//! # Motivation
//!
//! In scientific computing, especially pharmacokinetics and regulatory contexts,
//! it's critical to:
//! - Prevent cumulative uncertainty from degrading results below usable thresholds
//! - Isolate uncertainty from different experimental sources
//! - Create audit points for regulatory compliance (21 CFR Part 11)
//! - Apply different uncertainty models to different computational phases
//!
//! # Example
//!
//! ```sounio
//! // Firewall that resets confidence for calibrated values
//! #[firewall(mode = "reset", confidence = 0.95)]
//! fn calibrate(raw: Knowledge[f64]) -> Knowledge[f64] {
//!     // After calibration, we're confident in the result
//!     apply_calibration_curve(raw)
//! }
//!
//! // Firewall that blocks low-confidence values
//! #[firewall(mode = "gate", min_confidence = 0.8)]
//! fn critical_calculation(x: Knowledge[f64]) -> Knowledge[f64] {
//!     // Only runs if x.confidence >= 0.8
//!     x * dosing_factor
//! }
//!
//! // Firewall with audit logging
//! #[firewall(mode = "audit", log_provenance = true)]
//! fn regulatory_checkpoint(data: Knowledge[f64]) -> Knowledge[f64] {
//!     // Logs full provenance to audit trail
//!     data
//! }
//! ```

use super::models::UncertaintyModel;
use super::provenance::Provenance;
use std::fmt;

/// Epistemic firewall configuration
#[derive(Debug, Clone)]
pub struct FirewallConfig {
    /// Operating mode of the firewall
    pub mode: FirewallMode,

    /// Name/identifier for this firewall (for logging/debugging)
    pub name: Option<String>,

    /// Whether to log provenance at this boundary
    pub log_provenance: bool,

    /// Custom validation predicate (optional)
    pub validation: Option<ValidationRule>,

    /// Model to switch to after firewall (optional)
    pub switch_model: Option<UncertaintyModel>,
}

impl Default for FirewallConfig {
    fn default() -> Self {
        Self {
            mode: FirewallMode::PassThrough,
            name: None,
            log_provenance: false,
            validation: None,
            switch_model: None,
        }
    }
}

impl FirewallConfig {
    /// Create a firewall that resets confidence to a specific value
    pub fn reset(confidence: f64) -> Self {
        Self {
            mode: FirewallMode::Reset { confidence },
            ..Default::default()
        }
    }

    /// Create a firewall that clamps confidence to a range
    pub fn clamp(min: f64, max: f64) -> Self {
        Self {
            mode: FirewallMode::Clamp { min, max },
            ..Default::default()
        }
    }

    /// Create a firewall that blocks values below a threshold
    pub fn gate(min_confidence: f64) -> Self {
        Self {
            mode: FirewallMode::Gate { min_confidence },
            ..Default::default()
        }
    }

    /// Create an audit-only firewall
    pub fn audit(name: &str) -> Self {
        Self {
            mode: FirewallMode::Audit,
            name: Some(name.to_string()),
            log_provenance: true,
            ..Default::default()
        }
    }

    /// Create a model-switching firewall
    pub fn switch_model(model: UncertaintyModel) -> Self {
        Self {
            mode: FirewallMode::SwitchModel,
            switch_model: Some(model),
            ..Default::default()
        }
    }

    /// Add a name to this firewall
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Enable provenance logging
    pub fn with_provenance_logging(mut self) -> Self {
        self.log_provenance = true;
        self
    }

    /// Add custom validation rule
    pub fn with_validation(mut self, rule: ValidationRule) -> Self {
        self.validation = Some(rule);
        self
    }
}

/// Firewall operating mode
#[derive(Debug, Clone, PartialEq)]
pub enum FirewallMode {
    /// Pass through without modification (useful with validation only)
    PassThrough,

    /// Reset confidence to a fixed value
    Reset { confidence: f64 },

    /// Clamp confidence to a range
    Clamp { min: f64, max: f64 },

    /// Block values below minimum confidence (returns error)
    Gate { min_confidence: f64 },

    /// Decay confidence by a factor
    Decay { factor: f64 },

    /// Boost confidence by a factor (capped at 1.0)
    Boost { factor: f64 },

    /// Switch uncertainty model (no confidence change)
    SwitchModel,

    /// Audit-only mode (log but don't modify)
    Audit,

    /// Isolate: reset provenance chain (fresh start)
    Isolate,
}

impl FirewallMode {
    /// Apply this mode to a confidence value
    pub fn apply(&self, confidence: f64) -> Result<f64, FirewallViolation> {
        match self {
            FirewallMode::PassThrough => Ok(confidence),

            FirewallMode::Reset {
                confidence: new_conf,
            } => Ok(*new_conf),

            FirewallMode::Clamp { min, max } => Ok(confidence.clamp(*min, *max)),

            FirewallMode::Gate { min_confidence } => {
                if confidence >= *min_confidence {
                    Ok(confidence)
                } else {
                    Err(FirewallViolation::BelowThreshold {
                        actual: confidence,
                        required: *min_confidence,
                    })
                }
            }

            FirewallMode::Decay { factor } => Ok((confidence * factor).clamp(0.0, 1.0)),

            FirewallMode::Boost { factor } => Ok((confidence * factor).min(1.0)),

            FirewallMode::SwitchModel => Ok(confidence),

            FirewallMode::Audit => Ok(confidence),

            FirewallMode::Isolate => Ok(confidence), // Provenance handled separately
        }
    }
}

/// Validation rules for firewall
#[derive(Debug, Clone)]
pub enum ValidationRule {
    /// Confidence must be above threshold
    MinConfidence(f64),

    /// Confidence must be below threshold
    MaxConfidence(f64),

    /// Confidence must be in range
    ConfidenceRange { min: f64, max: f64 },

    /// Provenance must include specific transformation
    RequireTransformation(String),

    /// Provenance must originate from specific source
    RequireSource(String),

    /// Provenance depth must not exceed limit
    MaxProvenanceDepth(usize),

    /// Custom validation (compile-time expression)
    Custom(String),

    /// All rules must pass
    All(Vec<ValidationRule>),

    /// Any rule must pass
    Any(Vec<ValidationRule>),
}

impl ValidationRule {
    /// Validate a value against this rule
    pub fn validate(
        &self,
        confidence: f64,
        provenance: &Provenance,
    ) -> Result<(), FirewallViolation> {
        match self {
            ValidationRule::MinConfidence(min) => {
                if confidence >= *min {
                    Ok(())
                } else {
                    Err(FirewallViolation::ValidationFailed {
                        rule: format!("confidence >= {}", min),
                        actual: format!("confidence = {}", confidence),
                    })
                }
            }

            ValidationRule::MaxConfidence(max) => {
                if confidence <= *max {
                    Ok(())
                } else {
                    Err(FirewallViolation::ValidationFailed {
                        rule: format!("confidence <= {}", max),
                        actual: format!("confidence = {}", confidence),
                    })
                }
            }

            ValidationRule::ConfidenceRange { min, max } => {
                if confidence >= *min && confidence <= *max {
                    Ok(())
                } else {
                    Err(FirewallViolation::ValidationFailed {
                        rule: format!("confidence in [{}, {}]", min, max),
                        actual: format!("confidence = {}", confidence),
                    })
                }
            }

            ValidationRule::RequireTransformation(name) => {
                let has_transform = provenance.trace.steps.iter().any(|t| t.name.contains(name));
                if has_transform {
                    Ok(())
                } else {
                    Err(FirewallViolation::ValidationFailed {
                        rule: format!("requires transformation: {}", name),
                        actual: provenance.path_string(),
                    })
                }
            }

            ValidationRule::RequireSource(source) => {
                let origin_str = format!("{:?}", provenance.origin);
                if origin_str.contains(source) {
                    Ok(())
                } else {
                    Err(FirewallViolation::ValidationFailed {
                        rule: format!("requires source: {}", source),
                        actual: origin_str,
                    })
                }
            }

            ValidationRule::MaxProvenanceDepth(max) => {
                if provenance.depth() <= *max {
                    Ok(())
                } else {
                    Err(FirewallViolation::ValidationFailed {
                        rule: format!("provenance depth <= {}", max),
                        actual: format!("depth = {}", provenance.depth()),
                    })
                }
            }

            ValidationRule::Custom(expr) => {
                // Would be evaluated by the compiler
                // For now, always pass (placeholder)
                let _ = expr;
                Ok(())
            }

            ValidationRule::All(rules) => {
                for rule in rules {
                    rule.validate(confidence, provenance)?;
                }
                Ok(())
            }

            ValidationRule::Any(rules) => {
                let mut last_error = None;
                for rule in rules {
                    match rule.validate(confidence, provenance) {
                        Ok(()) => return Ok(()),
                        Err(e) => last_error = Some(e),
                    }
                }
                Err(last_error.unwrap_or(FirewallViolation::ValidationFailed {
                    rule: "any".to_string(),
                    actual: "no rules provided".to_string(),
                }))
            }
        }
    }
}

/// Firewall violation error
#[derive(Debug, Clone)]
pub enum FirewallViolation {
    /// Confidence below required threshold
    BelowThreshold { actual: f64, required: f64 },

    /// Validation rule failed
    ValidationFailed { rule: String, actual: String },

    /// Provenance verification failed
    ProvenanceInvalid { reason: String },

    /// Model incompatibility
    ModelIncompatible {
        from: String,
        to: String,
        reason: String,
    },
}

impl fmt::Display for FirewallViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FirewallViolation::BelowThreshold { actual, required } => {
                write!(
                    f,
                    "confidence {} below required threshold {}",
                    actual, required
                )
            }
            FirewallViolation::ValidationFailed { rule, actual } => {
                write!(f, "validation failed: {} (got: {})", rule, actual)
            }
            FirewallViolation::ProvenanceInvalid { reason } => {
                write!(f, "provenance invalid: {}", reason)
            }
            FirewallViolation::ModelIncompatible { from, to, reason } => {
                write!(f, "cannot switch model from {} to {}: {}", from, to, reason)
            }
        }
    }
}

impl std::error::Error for FirewallViolation {}

/// Runtime firewall that can be applied to epistemic values
#[derive(Debug, Clone)]
pub struct EpistemicFirewall {
    config: FirewallConfig,
    /// Audit log entries
    audit_log: Vec<AuditEntry>,
}

impl EpistemicFirewall {
    /// Create a new firewall with the given configuration
    pub fn new(config: FirewallConfig) -> Self {
        Self {
            config,
            audit_log: Vec::new(),
        }
    }

    /// Create a pass-through firewall
    pub fn pass_through() -> Self {
        Self::new(FirewallConfig::default())
    }

    /// Create a reset firewall
    pub fn reset(confidence: f64) -> Self {
        Self::new(FirewallConfig::reset(confidence))
    }

    /// Create a gate firewall
    pub fn gate(min_confidence: f64) -> Self {
        Self::new(FirewallConfig::gate(min_confidence))
    }

    /// Apply this firewall to a confidence value and provenance
    pub fn apply(
        &mut self,
        confidence: f64,
        provenance: &Provenance,
    ) -> Result<FirewallResult, FirewallViolation> {
        // First, run validation if configured
        if let Some(ref rule) = self.config.validation {
            rule.validate(confidence, provenance)?;
        }

        // Apply the mode transformation
        let new_confidence = self.config.mode.apply(confidence)?;

        // Handle provenance based on mode
        let new_provenance = match self.config.mode {
            FirewallMode::Isolate => Provenance::computed("firewall_isolate"),
            _ => provenance.clone(),
        };

        // Log if configured
        if self.config.log_provenance {
            self.audit_log.push(AuditEntry {
                timestamp: std::time::SystemTime::now(),
                firewall_name: self.config.name.clone(),
                input_confidence: confidence,
                output_confidence: new_confidence,
                provenance_hash: provenance.integrity_hash.clone(),
                mode: format!("{:?}", self.config.mode),
            });
        }

        Ok(FirewallResult {
            confidence: new_confidence,
            provenance: new_provenance,
            model_switch: self.config.switch_model.clone(),
        })
    }

    /// Get the audit log
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }

    /// Clear the audit log
    pub fn clear_audit_log(&mut self) {
        self.audit_log.clear();
    }

    /// Get the firewall configuration
    pub fn config(&self) -> &FirewallConfig {
        &self.config
    }
}

/// Result of applying a firewall
#[derive(Debug, Clone)]
pub struct FirewallResult {
    /// New confidence value
    pub confidence: f64,
    /// New provenance (may be reset in Isolate mode)
    pub provenance: Provenance,
    /// Model to switch to (if any)
    pub model_switch: Option<UncertaintyModel>,
}

/// Audit log entry
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// When the firewall was applied
    pub timestamp: std::time::SystemTime,
    /// Name of the firewall
    pub firewall_name: Option<String>,
    /// Input confidence
    pub input_confidence: f64,
    /// Output confidence
    pub output_confidence: f64,
    /// Hash of provenance at this point
    pub provenance_hash: Option<String>,
    /// Mode that was applied
    pub mode: String,
}

impl AuditEntry {
    /// Format for regulatory compliance logging
    pub fn to_regulatory_format(&self) -> String {
        let timestamp = self
            .timestamp
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        format!(
            "AUDIT|{}|{}|{:.6}|{:.6}|{}|{}",
            timestamp,
            self.firewall_name.as_deref().unwrap_or("unnamed"),
            self.input_confidence,
            self.output_confidence,
            self.provenance_hash.as_deref().unwrap_or("none"),
            self.mode
        )
    }
}

// =============================================================================
// Compile-time Firewall Attributes
// =============================================================================

/// Parsed firewall attribute from source code
#[derive(Debug, Clone)]
pub struct FirewallAttribute {
    /// Attribute span in source
    pub span: Option<crate::common::Span>,
    /// Parsed configuration
    pub config: FirewallConfig,
}

impl FirewallAttribute {
    /// Parse from attribute arguments
    ///
    /// Expected formats:
    /// - `#[firewall(mode = "reset", confidence = 0.95)]`
    /// - `#[firewall(mode = "gate", min_confidence = 0.8)]`
    /// - `#[firewall(mode = "audit", log_provenance = true)]`
    pub fn parse(args: &[(String, String)]) -> Result<Self, String> {
        let mut config = FirewallConfig::default();

        let mut mode_str: Option<&str> = None;
        let mut confidence: Option<f64> = None;
        let mut min_confidence: Option<f64> = None;
        let mut max_confidence: Option<f64> = None;
        let mut factor: Option<f64> = None;

        for (key, value) in args {
            match key.as_str() {
                "mode" => mode_str = Some(value.as_str()),
                "confidence" => {
                    confidence = Some(
                        value
                            .parse()
                            .map_err(|_| format!("invalid confidence: {}", value))?,
                    )
                }
                "min_confidence" => {
                    min_confidence = Some(
                        value
                            .parse()
                            .map_err(|_| format!("invalid min_confidence: {}", value))?,
                    )
                }
                "max_confidence" => {
                    max_confidence = Some(
                        value
                            .parse()
                            .map_err(|_| format!("invalid max_confidence: {}", value))?,
                    )
                }
                "factor" => {
                    factor = Some(
                        value
                            .parse()
                            .map_err(|_| format!("invalid factor: {}", value))?,
                    )
                }
                "name" => config.name = Some(value.clone()),
                "log_provenance" => config.log_provenance = value == "true" || value == "1",
                _ => return Err(format!("unknown firewall attribute: {}", key)),
            }
        }

        // Build mode from parsed values
        config.mode = match mode_str {
            Some("reset") => FirewallMode::Reset {
                confidence: confidence.unwrap_or(1.0),
            },
            Some("clamp") => FirewallMode::Clamp {
                min: min_confidence.unwrap_or(0.0),
                max: max_confidence.unwrap_or(1.0),
            },
            Some("gate") => FirewallMode::Gate {
                min_confidence: min_confidence.unwrap_or(0.5),
            },
            Some("decay") => FirewallMode::Decay {
                factor: factor.unwrap_or(0.9),
            },
            Some("boost") => FirewallMode::Boost {
                factor: factor.unwrap_or(1.1),
            },
            Some("audit") => FirewallMode::Audit,
            Some("isolate") => FirewallMode::Isolate,
            Some("pass") | Some("passthrough") | None => FirewallMode::PassThrough,
            Some(other) => return Err(format!("unknown firewall mode: {}", other)),
        };

        Ok(Self { span: None, config })
    }
}

// =============================================================================
// Type-level Firewall Constraints
// =============================================================================

/// Compile-time firewall constraint for type checking
#[derive(Debug, Clone, PartialEq)]
pub enum FirewallConstraint {
    /// Confidence must be at least this value after firewall
    MinConfidencePost(f64),

    /// Confidence must be at most this value after firewall
    MaxConfidencePost(f64),

    /// Function must have firewall with specific mode
    RequireMode(String),

    /// Function must not have firewall (for composition)
    NoFirewall,
}

impl FirewallConstraint {
    /// Check if a firewall configuration satisfies this constraint
    pub fn satisfied_by(&self, config: &FirewallConfig) -> bool {
        match self {
            FirewallConstraint::MinConfidencePost(min) => match &config.mode {
                FirewallMode::Reset { confidence } => *confidence >= *min,
                FirewallMode::Clamp { min: clamp_min, .. } => *clamp_min >= *min,
                FirewallMode::Gate { min_confidence } => *min_confidence >= *min,
                FirewallMode::Boost { .. } => true, // Can't guarantee without input
                _ => false,
            },

            FirewallConstraint::MaxConfidencePost(max) => match &config.mode {
                FirewallMode::Reset { confidence } => *confidence <= *max,
                FirewallMode::Clamp { max: clamp_max, .. } => *clamp_max <= *max,
                FirewallMode::Decay { .. } => true, // Always decreases
                _ => false,
            },

            FirewallConstraint::RequireMode(mode_name) => format!("{:?}", config.mode)
                .to_lowercase()
                .contains(&mode_name.to_lowercase()),

            FirewallConstraint::NoFirewall => {
                matches!(config.mode, FirewallMode::PassThrough)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reset_firewall() {
        let mut firewall = EpistemicFirewall::reset(0.95);
        let provenance = Provenance::literal();

        let result = firewall.apply(0.5, &provenance).unwrap();
        assert!((result.confidence - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_gate_firewall_pass() {
        let mut firewall = EpistemicFirewall::gate(0.8);
        let provenance = Provenance::literal();

        let result = firewall.apply(0.9, &provenance);
        assert!(result.is_ok());
        assert!((result.unwrap().confidence - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_gate_firewall_block() {
        let mut firewall = EpistemicFirewall::gate(0.8);
        let provenance = Provenance::literal();

        let result = firewall.apply(0.7, &provenance);
        assert!(result.is_err());

        if let Err(FirewallViolation::BelowThreshold { actual, required }) = result {
            assert!((actual - 0.7).abs() < 1e-10);
            assert!((required - 0.8).abs() < 1e-10);
        } else {
            panic!("Expected BelowThreshold error");
        }
    }

    #[test]
    fn test_clamp_firewall() {
        let config = FirewallConfig::clamp(0.3, 0.9);
        let mut firewall = EpistemicFirewall::new(config);
        let provenance = Provenance::literal();

        // Below min
        let result = firewall.apply(0.1, &provenance).unwrap();
        assert!((result.confidence - 0.3).abs() < 1e-10);

        // Above max
        let result = firewall.apply(0.95, &provenance).unwrap();
        assert!((result.confidence - 0.9).abs() < 1e-10);

        // In range
        let result = firewall.apply(0.5, &provenance).unwrap();
        assert!((result.confidence - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_audit_logging() {
        let config = FirewallConfig::audit("test_checkpoint");
        let mut firewall = EpistemicFirewall::new(config);
        let provenance = Provenance::literal();

        firewall.apply(0.8, &provenance).unwrap();
        firewall.apply(0.9, &provenance).unwrap();

        assert_eq!(firewall.audit_log().len(), 2);
        assert_eq!(
            firewall.audit_log()[0].firewall_name,
            Some("test_checkpoint".to_string())
        );
    }

    #[test]
    fn test_validation_rule_min_confidence() {
        let rule = ValidationRule::MinConfidence(0.7);
        let provenance = Provenance::literal();

        assert!(rule.validate(0.8, &provenance).is_ok());
        assert!(rule.validate(0.6, &provenance).is_err());
    }

    #[test]
    fn test_validation_rule_all() {
        let rule = ValidationRule::All(vec![
            ValidationRule::MinConfidence(0.5),
            ValidationRule::MaxConfidence(0.9),
        ]);
        let provenance = Provenance::literal();

        assert!(rule.validate(0.7, &provenance).is_ok());
        assert!(rule.validate(0.4, &provenance).is_err());
        assert!(rule.validate(0.95, &provenance).is_err());
    }

    #[test]
    fn test_parse_firewall_attribute() {
        let args = vec![
            ("mode".to_string(), "reset".to_string()),
            ("confidence".to_string(), "0.95".to_string()),
            ("name".to_string(), "calibration".to_string()),
        ];

        let attr = FirewallAttribute::parse(&args).unwrap();

        assert_eq!(attr.config.name, Some("calibration".to_string()));
        if let FirewallMode::Reset { confidence } = attr.config.mode {
            assert!((confidence - 0.95).abs() < 1e-10);
        } else {
            panic!("Expected Reset mode");
        }
    }

    #[test]
    fn test_firewall_constraint() {
        let config = FirewallConfig::reset(0.9);

        assert!(FirewallConstraint::MinConfidencePost(0.8).satisfied_by(&config));
        assert!(!FirewallConstraint::MinConfidencePost(0.95).satisfied_by(&config));
        assert!(FirewallConstraint::MaxConfidencePost(0.95).satisfied_by(&config));
    }
}
