//! Lean 4 Interoperability for Sounio
//!
//! Native integration with Lean 4 proof assistant enabling:
//! - Export Sounio refinements/geometry to Lean theorems
//! - Import Lean proof certificates as epistemic axioms (confidence 1.0)
//! - Lean-inspired tactics for symbolic deduction
//! - Effect-based verification handler
//!
//! # Architecture
//!
//! ```text
//! Sounio                      Lean 4
//! ┌─────────────┐             ┌─────────────┐
//! │ Refinement  │ ──export──► │ Theorem     │
//! │ Constraint  │             │ Definition  │
//! └─────────────┘             └─────────────┘
//!                                   │
//!                              prove (tactics)
//!                                   │
//!                                   ▼
//! ┌─────────────┐             ┌─────────────┐
//! │ Knowledge   │ ◄──import── │ Proof       │
//! │ <Theorem>   │             │ Certificate │
//! │ conf = 1.0  │             └─────────────┘
//! └─────────────┘
//! ```
//!
//! # Example
//!
//! ```sounio
//! refine geometry_theorem Pythagoras {
//!     premise: RightTriangle(A, B, C);
//!     conclusion: sq(AB) + sq(BC) = sq(AC);
//! } export to lean "Mathlib.Geometry.Euclidean";
//!
//! import lean "Mathlib.Algebra.BigOperators" theorem sum_range { confidence = 1.0 };
//! ```

use std::collections::HashMap;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use crate::epistemic::bayesian::BetaConfidence;

// =============================================================================
// Core Types
// =============================================================================

/// Configuration for Lean interop
#[derive(Debug, Clone)]
pub struct LeanConfig {
    /// Path to Lean executable (default: "lean")
    pub lean_path: String,
    /// Path to mathlib (if available)
    pub mathlib_path: Option<String>,
    /// Timeout for proof attempts
    pub timeout: Duration,
    /// Whether to use lake for project builds
    pub use_lake: bool,
    /// Additional Lean options
    pub options: HashMap<String, String>,
}

impl Default for LeanConfig {
    fn default() -> Self {
        Self {
            lean_path: "lean".to_string(),
            mathlib_path: None,
            timeout: Duration::from_secs(60),
            use_lake: true,
            options: HashMap::new(),
        }
    }
}

/// Lean 4 server connection (LSP or subprocess)
pub struct LeanServer {
    process: Option<Child>,
    config: LeanConfig,
    initialized: bool,
}

impl LeanServer {
    /// Create new Lean server connection
    pub fn new(config: LeanConfig) -> Result<Self, LeanConnectionError> {
        Ok(Self {
            process: None,
            config,
            initialized: false,
        })
    }

    /// Check if Lean is available
    pub fn is_available(&self) -> bool {
        Command::new(&self.config.lean_path)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Get Lean version
    pub fn version(&self) -> Option<String> {
        Command::new(&self.config.lean_path)
            .arg("--version")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
    }

    /// Run Lean code and get result
    pub fn run_code(&self, code: &str) -> Result<LeanOutput, LeanExecutionError> {
        // Create temp file with Lean code
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("sounio_lean_proof.lean");

        std::fs::write(&temp_file, code).map_err(|e| LeanExecutionError::IoError(e.to_string()))?;

        // Run Lean on the file
        let start = Instant::now();
        let output = Command::new(&self.config.lean_path)
            .arg(&temp_file)
            .output()
            .map_err(|e| LeanExecutionError::ProcessError(e.to_string()))?;

        let elapsed = start.elapsed();

        // Clean up
        let _ = std::fs::remove_file(&temp_file);

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        Ok(LeanOutput {
            success: output.status.success(),
            stdout,
            stderr,
            elapsed,
        })
    }
}

/// Output from Lean execution
#[derive(Debug, Clone)]
pub struct LeanOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub elapsed: Duration,
}

/// Main Lean interoperability interface
pub struct LeanInterop {
    server: LeanServer,
    theorem_cache: HashMap<String, ProofCertificate>,
}

impl LeanInterop {
    /// Create new Lean interop with default config
    pub fn new() -> Result<Self, LeanConnectionError> {
        Self::with_config(LeanConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: LeanConfig) -> Result<Self, LeanConnectionError> {
        let server = LeanServer::new(config)?;
        Ok(Self {
            server,
            theorem_cache: HashMap::new(),
        })
    }

    /// Check if Lean is available on this system
    pub fn is_available(&self) -> bool {
        self.server.is_available()
    }

    /// Export a Sounio refinement constraint to Lean code
    pub fn export_refinement(&self, refinement: &RefinementConstraint) -> String {
        let mut code = String::new();

        // Header with imports
        code.push_str("-- Auto-generated from Sounio refinement\n");
        code.push_str("import Mathlib.Data.Real.Basic\n");
        code.push_str("import Mathlib.Algebra.Order.Ring.Lemmas\n\n");

        // Namespace
        code.push_str("namespace Sounio\n\n");

        // Translate refinement to Lean theorem
        let theorem_name = refinement.name.as_deref().unwrap_or("dem_refinement");
        let lean_type = self.translate_refinement_type(refinement);
        let lean_proof = self.generate_proof_skeleton(refinement);

        code.push_str(&format!("theorem {} : {} := by\n", theorem_name, lean_type));
        code.push_str(&format!("  {}\n", lean_proof));
        code.push_str("\nend Sounio\n");

        code
    }

    /// Export a geometry predicate to Lean
    pub fn export_geometry_predicate(&self, predicate: &GeometryPredicate) -> String {
        let mut code = String::new();

        code.push_str("-- Auto-generated from Sounio geometry\n");
        code.push_str("import Mathlib.Geometry.Euclidean.Basic\n");
        code.push_str("import Mathlib.Geometry.Euclidean.Angle.Unoriented.Basic\n\n");

        code.push_str("namespace Sounio.Geometry\n\n");

        // Point declarations
        code.push_str(
            "variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace \u{211d} V]\n",
        );
        code.push_str("variable {P : Type*} [MetricSpace P] [NormedAddTorsor V P]\n\n");

        // Predicate as theorem
        let theorem = self.translate_geometry_predicate(predicate);
        code.push_str(&theorem);

        code.push_str("\nend Sounio.Geometry\n");

        code
    }

    /// Export a causal model constraint to Lean
    pub fn export_causal_constraint(&self, constraint: &CausalConstraint) -> String {
        let mut code = String::new();

        code.push_str("-- Auto-generated from Sounio causal model\n");
        code.push_str("import Mathlib.Probability.Independence.Basic\n");
        code.push_str("import Mathlib.MeasureTheory.Measure.MeasureSpace\n\n");

        code.push_str("namespace Sounio.Causal\n\n");

        // Translate causal independence to measure-theoretic statement
        let theorem = self.translate_causal_constraint(constraint);
        code.push_str(&theorem);

        code.push_str("\nend Sounio.Causal\n");

        code
    }

    /// Attempt to prove a theorem in Lean
    pub fn prove(&mut self, lean_code: &str) -> Result<ProofCertificate, LeanProofError> {
        // Check cache first
        let code_hash = self.hash_code(lean_code);
        if let Some(cert) = self.theorem_cache.get(&code_hash) {
            return Ok(cert.clone());
        }

        // Run Lean
        let output = self
            .server
            .run_code(lean_code)
            .map_err(|e| LeanProofError::ExecutionError(e.to_string()))?;

        if output.success {
            // Parse the proof tree from output
            let cert = ProofCertificate {
                theorem_name: self.extract_theorem_name(lean_code),
                lean_code: lean_code.to_string(),
                proof_tree: self.parse_proof_tree(&output.stdout),
                verified: true,
                elapsed: output.elapsed,
                mathlib_deps: self.extract_imports(lean_code),
            };

            // Cache successful proofs
            self.theorem_cache.insert(code_hash, cert.clone());

            Ok(cert)
        } else {
            let goals = self.parse_remaining_goals(&output.stderr);
            Err(LeanProofError::ProofFailed {
                error: output.stderr,
                goals_remaining: goals,
            })
        }
    }

    /// Import a Lean theorem as epistemic knowledge
    pub fn import_theorem(&self, cert: &ProofCertificate) -> LeanTheorem {
        LeanTheorem {
            name: cert.theorem_name.clone(),
            statement: self.extract_statement(&cert.lean_code),
            confidence: BetaConfidence::from_confidence(1.0, 10000.0), // Lean proof = axiomatic
            provenance: ProofProvenance::Lean {
                proof_tree: cert.proof_tree.clone(),
                mathlib_deps: cert.mathlib_deps.clone(),
                verified_at: std::time::SystemTime::now(),
            },
        }
    }

    /// Import theorem directly from mathlib by name
    pub fn import_mathlib_theorem(&mut self, path: &str) -> Result<LeanTheorem, LeanImportError> {
        // Generate Lean code to check theorem exists
        let check_code = format!(
            "import {}\n#check @{}\n",
            path.rsplit('.')
                .skip(1)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("."),
            path.rsplit('.').next().unwrap_or(path)
        );

        let output = self
            .server
            .run_code(&check_code)
            .map_err(|e| LeanImportError::ExecutionError(e.to_string()))?;

        if output.success {
            Ok(LeanTheorem {
                name: path.to_string(),
                statement: output.stdout.lines().next().unwrap_or("").to_string(),
                confidence: BetaConfidence::from_confidence(1.0, 10000.0),
                provenance: ProofProvenance::Mathlib {
                    path: path.to_string(),
                    verified: true,
                },
            })
        } else {
            Err(LeanImportError::TheoremNotFound(path.to_string()))
        }
    }

    // =========================================================================
    // Translation Helpers
    // =========================================================================

    fn translate_refinement_type(&self, refinement: &RefinementConstraint) -> String {
        match &refinement.kind {
            RefinementKind::Positive => format!("0 < {}", refinement.variable),
            RefinementKind::NonNegative => format!("0 \u{2264} {}", refinement.variable),
            RefinementKind::Range { min, max } => {
                format!(
                    "{} \u{2264} {} \u{2227} {} \u{2264} {}",
                    min, refinement.variable, refinement.variable, max
                )
            }
            RefinementKind::Predicate(pred) => self.translate_predicate_to_lean(pred),
            RefinementKind::Custom(s) => s.clone(),
        }
    }

    fn translate_predicate_to_lean(&self, pred: &str) -> String {
        // Basic translation of Sounio predicates to Lean
        pred.replace("&&", "\u{2227}")
            .replace("||", "\u{2228}")
            .replace("!", "\u{00ac}")
            .replace(">=", "\u{2265}")
            .replace("<=", "\u{2264}")
            .replace("!=", "\u{2260}")
            .replace("==", "=")
    }

    fn generate_proof_skeleton(&self, refinement: &RefinementConstraint) -> String {
        match &refinement.kind {
            RefinementKind::Positive | RefinementKind::NonNegative => "linarith".to_string(),
            RefinementKind::Range { .. } => "constructor <;> linarith".to_string(),
            RefinementKind::Predicate(_) => "decide".to_string(),
            RefinementKind::Custom(_) => "sorry -- requires manual proof".to_string(),
        }
    }

    fn translate_geometry_predicate(&self, pred: &GeometryPredicate) -> String {
        match pred {
            GeometryPredicate::Collinear(a, b, c) => {
                format!(
                    "theorem collinear_{}_{}_{} : Collinear \u{211d} ({{{}, {}, {}}}) := by\n  sorry\n",
                    a, b, c, a, b, c
                )
            }
            GeometryPredicate::Perpendicular(l1, l2) => {
                format!(
                    "theorem perp_{}_{} : {}.toDirection \u{22a5} {}.toDirection := by\n  sorry\n",
                    l1, l2, l1, l2
                )
            }
            GeometryPredicate::Parallel(l1, l2) => {
                format!(
                    "theorem parallel_{}_{} : {}.toDirection = {}.toDirection \u{2228} {}.toDirection = -{}.toDirection := by\n  sorry\n",
                    l1, l2, l1, l2, l1, l2
                )
            }
            GeometryPredicate::Congruent(s1, s2) => {
                format!(
                    "theorem cong_{}_{} : dist {} = dist {} := by\n  sorry\n",
                    s1, s2, s1, s2
                )
            }
            GeometryPredicate::Cyclic(points) => {
                format!(
                    "theorem cyclic_{} : \u{2203} (c : P) (r : \u{211d}), \u{2200} p \u{2208} {{{}}}, dist c p = r := by\n  sorry\n",
                    points.join("_"),
                    points.join(", ")
                )
            }
            GeometryPredicate::Custom(s) => format!("-- Custom: {}\n", s),
        }
    }

    fn translate_causal_constraint(&self, constraint: &CausalConstraint) -> String {
        match constraint {
            CausalConstraint::Independence { x, y, given } => {
                if given.is_empty() {
                    format!(
                        "theorem indep_{}_{} : IndepFun {} {} \u{03bc} := by\n  sorry\n",
                        x, y, x, y
                    )
                } else {
                    format!(
                        "theorem condIndep_{}_{}_{} : CondIndepFun {} {} {} \u{03bc} := by\n  sorry\n",
                        x,
                        y,
                        given.join("_"),
                        x,
                        y,
                        given.join(" ")
                    )
                }
            }
            CausalConstraint::DoIntervention { target, value } => {
                format!(
                    "-- do({} := {}) intervention\naxiom intervention_{} : {} = {}\n",
                    target, value, target, target, value
                )
            }
            CausalConstraint::Counterfactual { condition, outcome } => {
                format!(
                    "-- Counterfactual: {} => {}\ntheorem cf_{} : {} \u{2192} {} := by\n  sorry\n",
                    condition,
                    outcome,
                    condition.replace(" ", "_"),
                    condition,
                    outcome
                )
            }
        }
    }

    fn hash_code(&self, code: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        code.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn extract_theorem_name(&self, code: &str) -> String {
        for line in code.lines() {
            if line.trim().starts_with("theorem ") || line.trim().starts_with("lemma ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    return parts[1].trim_end_matches(':').to_string();
                }
            }
        }
        "unknown".to_string()
    }

    fn extract_statement(&self, code: &str) -> String {
        for line in code.lines() {
            if line.contains(":=")
                && let Some(idx) = line.find(':')
            {
                let stmt = &line[idx + 1..];
                if let Some(eq_idx) = stmt.find(":=") {
                    return stmt[..eq_idx].trim().to_string();
                }
            }
        }
        "".to_string()
    }

    fn extract_imports(&self, code: &str) -> Vec<String> {
        code.lines()
            .filter(|l| l.trim().starts_with("import "))
            .map(|l| l.trim().strip_prefix("import ").unwrap_or("").to_string())
            .collect()
    }

    fn parse_proof_tree(&self, _output: &str) -> ProofTree {
        // Simplified proof tree - full implementation would parse Lean's proof terms
        ProofTree {
            root: ProofNode::Tactic("auto".to_string()),
            children: vec![],
        }
    }

    fn parse_remaining_goals(&self, stderr: &str) -> Vec<String> {
        let mut goals = vec![];
        let mut in_goals = false;
        for line in stderr.lines() {
            if line.contains("unsolved goals") {
                in_goals = true;
            } else if in_goals && !line.trim().is_empty() {
                goals.push(line.trim().to_string());
            }
        }
        goals
    }
}

impl Default for LeanInterop {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            server: LeanServer {
                process: None,
                config: LeanConfig::default(),
                initialized: false,
            },
            theorem_cache: HashMap::new(),
        })
    }
}

// =============================================================================
// Supporting Types
// =============================================================================

/// A refinement constraint from Sounio
#[derive(Debug, Clone)]
pub struct RefinementConstraint {
    pub name: Option<String>,
    pub variable: String,
    pub kind: RefinementKind,
}

/// Kind of refinement
#[derive(Debug, Clone)]
pub enum RefinementKind {
    Positive,
    NonNegative,
    Range { min: String, max: String },
    Predicate(String),
    Custom(String),
}

/// Geometry predicate types
#[derive(Debug, Clone)]
pub enum GeometryPredicate {
    Collinear(String, String, String),
    Perpendicular(String, String),
    Parallel(String, String),
    Congruent(String, String),
    Cyclic(Vec<String>),
    Custom(String),
}

/// Causal constraint types
#[derive(Debug, Clone)]
pub enum CausalConstraint {
    Independence {
        x: String,
        y: String,
        given: Vec<String>,
    },
    DoIntervention {
        target: String,
        value: String,
    },
    Counterfactual {
        condition: String,
        outcome: String,
    },
}

/// Proof certificate from Lean
#[derive(Debug, Clone)]
pub struct ProofCertificate {
    pub theorem_name: String,
    pub lean_code: String,
    pub proof_tree: ProofTree,
    pub verified: bool,
    pub elapsed: Duration,
    pub mathlib_deps: Vec<String>,
}

/// Simplified proof tree representation
#[derive(Debug, Clone)]
pub struct ProofTree {
    pub root: ProofNode,
    pub children: Vec<ProofTree>,
}

/// A node in the proof tree
#[derive(Debug, Clone)]
pub enum ProofNode {
    Tactic(String),
    Term(String),
    Hypothesis(String),
    Goal(String),
}

/// An imported Lean theorem with epistemic metadata
#[derive(Debug, Clone)]
pub struct LeanTheorem {
    pub name: String,
    pub statement: String,
    pub confidence: BetaConfidence,
    pub provenance: ProofProvenance,
}

/// Provenance tracking for imported theorems
#[derive(Debug, Clone)]
pub enum ProofProvenance {
    Lean {
        proof_tree: ProofTree,
        mathlib_deps: Vec<String>,
        verified_at: std::time::SystemTime,
    },
    Mathlib {
        path: String,
        verified: bool,
    },
    Axiom {
        source: String,
    },
}

// =============================================================================
// Errors
// =============================================================================

/// Error connecting to Lean
#[derive(Debug, Clone)]
pub enum LeanConnectionError {
    NotInstalled,
    VersionMismatch { expected: String, found: String },
    ConfigError(String),
}

impl std::fmt::Display for LeanConnectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInstalled => write!(f, "Lean 4 is not installed"),
            Self::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "Lean version mismatch: expected {}, found {}",
                    expected, found
                )
            }
            Self::ConfigError(e) => write!(f, "Lean config error: {}", e),
        }
    }
}

impl std::error::Error for LeanConnectionError {}

/// Error during Lean execution
#[derive(Debug, Clone)]
pub enum LeanExecutionError {
    IoError(String),
    ProcessError(String),
    Timeout,
    ParseError(String),
}

impl std::fmt::Display for LeanExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::ProcessError(e) => write!(f, "Process error: {}", e),
            Self::Timeout => write!(f, "Lean execution timed out"),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for LeanExecutionError {}

/// Error during proof attempt
#[derive(Debug, Clone)]
pub enum LeanProofError {
    ExecutionError(String),
    ProofFailed {
        error: String,
        goals_remaining: Vec<String>,
    },
    Timeout,
    InvalidTheorem(String),
}

impl std::fmt::Display for LeanProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExecutionError(e) => write!(f, "Execution error: {}", e),
            Self::ProofFailed {
                error,
                goals_remaining,
            } => {
                write!(
                    f,
                    "Proof failed: {}. Remaining goals: {:?}",
                    error, goals_remaining
                )
            }
            Self::Timeout => write!(f, "Proof attempt timed out"),
            Self::InvalidTheorem(e) => write!(f, "Invalid theorem: {}", e),
        }
    }
}

impl std::error::Error for LeanProofError {}

/// Error importing Lean theorem
#[derive(Debug, Clone)]
pub enum LeanImportError {
    TheoremNotFound(String),
    ExecutionError(String),
    ParseError(String),
}

impl std::fmt::Display for LeanImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TheoremNotFound(name) => write!(f, "Theorem not found: {}", name),
            Self::ExecutionError(e) => write!(f, "Execution error: {}", e),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for LeanImportError {}

/// Error exporting to Lean
#[derive(Debug, Clone)]
pub enum LeanExportError {
    UnsupportedConstruct(String),
    TranslationError(String),
}

impl std::fmt::Display for LeanExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedConstruct(c) => write!(f, "Unsupported construct: {}", c),
            Self::TranslationError(e) => write!(f, "Translation error: {}", e),
        }
    }
}

impl std::error::Error for LeanExportError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refinement_export_positive() {
        let interop = LeanInterop::default();
        let refinement = RefinementConstraint {
            name: Some("positive_dose".to_string()),
            variable: "dose".to_string(),
            kind: RefinementKind::Positive,
        };

        let code = interop.export_refinement(&refinement);
        assert!(code.contains("theorem positive_dose"));
        assert!(code.contains("0 < dose"));
        assert!(code.contains("linarith"));
    }

    #[test]
    fn test_refinement_export_range() {
        let interop = LeanInterop::default();
        let refinement = RefinementConstraint {
            name: Some("bounded_conc".to_string()),
            variable: "conc".to_string(),
            kind: RefinementKind::Range {
                min: "0".to_string(),
                max: "100".to_string(),
            },
        };

        let code = interop.export_refinement(&refinement);
        assert!(code.contains("theorem bounded_conc"));
        assert!(code.contains("0 \u{2264} conc"));
        assert!(code.contains("conc \u{2264} 100"));
    }

    #[test]
    fn test_geometry_predicate_collinear() {
        let interop = LeanInterop::default();
        let pred = GeometryPredicate::Collinear("A".to_string(), "B".to_string(), "C".to_string());

        let code = interop.export_geometry_predicate(&pred);
        assert!(code.contains("Collinear"));
        assert!(code.contains("A, B, C"));
    }

    #[test]
    fn test_geometry_predicate_cyclic() {
        let interop = LeanInterop::default();
        let pred = GeometryPredicate::Cyclic(vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ]);

        let code = interop.export_geometry_predicate(&pred);
        assert!(code.contains("cyclic_A_B_C_D"));
        assert!(code.contains("A, B, C, D"));
    }

    #[test]
    fn test_causal_independence() {
        let interop = LeanInterop::default();
        let constraint = CausalConstraint::Independence {
            x: "X".to_string(),
            y: "Y".to_string(),
            given: vec![],
        };

        let code = interop.export_causal_constraint(&constraint);
        assert!(code.contains("IndepFun"));
        assert!(code.contains("X"));
        assert!(code.contains("Y"));
    }

    #[test]
    fn test_causal_conditional_independence() {
        let interop = LeanInterop::default();
        let constraint = CausalConstraint::Independence {
            x: "X".to_string(),
            y: "Y".to_string(),
            given: vec!["Z".to_string()],
        };

        let code = interop.export_causal_constraint(&constraint);
        assert!(code.contains("CondIndepFun"));
        assert!(code.contains("Z"));
    }

    #[test]
    fn test_proof_certificate_creation() {
        let cert = ProofCertificate {
            theorem_name: "test_thm".to_string(),
            lean_code: "theorem test_thm : True := trivial".to_string(),
            proof_tree: ProofTree {
                root: ProofNode::Tactic("trivial".to_string()),
                children: vec![],
            },
            verified: true,
            elapsed: Duration::from_millis(100),
            mathlib_deps: vec![],
        };

        let interop = LeanInterop::default();
        let theorem = interop.import_theorem(&cert);

        assert_eq!(theorem.name, "test_thm");
        // Beta distribution clamps to [0.001, 0.999] for numerical stability
        assert!((theorem.confidence.mean() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lean_theorem_has_axiomatic_confidence() {
        let theorem = LeanTheorem {
            name: "pythagoras".to_string(),
            statement: "a^2 + b^2 = c^2".to_string(),
            confidence: BetaConfidence::from_confidence(1.0, 10000.0),
            provenance: ProofProvenance::Lean {
                proof_tree: ProofTree {
                    root: ProofNode::Tactic("ring".to_string()),
                    children: vec![],
                },
                mathlib_deps: vec!["Mathlib.Geometry.Euclidean".to_string()],
                verified_at: std::time::SystemTime::now(),
            },
        };

        // Lean proofs have confidence ~1.0 (axiomatic)
        // Beta distribution clamps to [0.001, 0.999] for numerical stability
        assert!((theorem.confidence.mean() - 1.0).abs() < 0.01);
        assert!(theorem.confidence.variance() < 0.001);
    }

    #[test]
    fn test_predicate_translation() {
        let interop = LeanInterop::default();
        let result = interop.translate_predicate_to_lean("x >= 0 && y <= 100");
        assert!(result.contains("\u{2265}"));
        assert!(result.contains("\u{2264}"));
        assert!(result.contains("\u{2227}"));
    }

    #[test]
    fn test_extract_theorem_name() {
        let interop = LeanInterop::default();
        let code = "theorem my_theorem : Nat := 42";
        assert_eq!(interop.extract_theorem_name(code), "my_theorem");

        let code2 = "lemma my_lemma : True := trivial";
        assert_eq!(interop.extract_theorem_name(code2), "my_lemma");
    }

    #[test]
    fn test_extract_imports() {
        let interop = LeanInterop::default();
        let code = "import Mathlib.Data.Real.Basic\nimport Mathlib.Algebra.Order\n\ntheorem x : True := trivial";
        let imports = interop.extract_imports(code);
        assert_eq!(imports.len(), 2);
        assert!(imports.contains(&"Mathlib.Data.Real.Basic".to_string()));
    }
}
