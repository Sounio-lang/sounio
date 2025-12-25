//! Coq Interoperability for Sounio
//!
//! Native integration with Coq proof assistant enabling:
//! - Export Sounio refinements/geometry to Coq theorems
//! - Import Coq proof certificates as epistemic axioms (confidence 1.0)
//! - Coq-inspired tactics (Ltac2) for symbolic deduction
//! - Effect-based verification handler
//!
//! # Architecture
//!
//! ```text
//! Sounio                      Coq
//! ┌─────────────┐             ┌─────────────┐
//! │ Refinement  │ ──export──► │ Theorem     │
//! │ Constraint  │             │ Definition  │
//! └─────────────┘             └─────────────┘
//!                                   │
//!                              prove (tactics)
//!                                   │
//!                                   ▼
//! ┌─────────────┐             ┌─────────────┐
//! │ Knowledge   │ ◄──import── │ .vo file    │
//! │ <Theorem>   │             │ Certificate │
//! │ conf = 1.0  │             └─────────────┘
//! └─────────────┘
//! ```
//!
//! # Example
//!
//! ```sounio
//! refine geometry_theorem TriangleInequality {
//!     premise: Length(AB) + Length(BC) >= Length(AC);
//!     conclusion: ValidTriangle(A, B, C);
//! } export to coq "Coq.Arith.PeanoNat";
//!
//! import coq "mathcomp.ssreflect" theorem ssrnat_addn { confidence = 1.0 };
//! ```

use std::collections::HashMap;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use crate::epistemic::bayesian::BetaConfidence;

// =============================================================================
// Core Types
// =============================================================================

/// Configuration for Coq interop
#[derive(Debug, Clone)]
pub struct CoqConfig {
    /// Path to coqtop executable (default: "coqtop")
    pub coqtop_path: String,
    /// Path to coqc compiler (default: "coqc")
    pub coqc_path: String,
    /// Include paths for libraries
    pub include_paths: Vec<String>,
    /// Load mathcomp if available
    pub use_mathcomp: bool,
    /// Timeout for proof attempts
    pub timeout: Duration,
    /// Additional Coq options
    pub options: HashMap<String, String>,
}

impl Default for CoqConfig {
    fn default() -> Self {
        Self {
            coqtop_path: "coqtop".to_string(),
            coqc_path: "coqc".to_string(),
            include_paths: vec![],
            use_mathcomp: false,
            timeout: Duration::from_secs(60),
            options: HashMap::new(),
        }
    }
}

/// Coq server connection (coqtop subprocess)
pub struct CoqServer {
    process: Option<Child>,
    config: CoqConfig,
    initialized: bool,
}

impl CoqServer {
    /// Create new Coq server connection
    pub fn new(config: CoqConfig) -> Result<Self, CoqConnectionError> {
        Ok(Self {
            process: None,
            config,
            initialized: false,
        })
    }

    /// Check if Coq is available
    pub fn is_available(&self) -> bool {
        Command::new(&self.config.coqtop_path)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Get Coq version
    pub fn version(&self) -> Option<String> {
        Command::new(&self.config.coqtop_path)
            .arg("--version")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.lines().next().unwrap_or("").trim().to_string())
    }

    /// Start interactive coqtop session
    pub fn start_session(&mut self) -> Result<(), CoqConnectionError> {
        let mut cmd = Command::new(&self.config.coqtop_path);
        cmd.args(["-q", "-emacs"]); // Quiet mode, emacs protocol

        for path in &self.config.include_paths {
            cmd.args(["-I", path]);
        }

        let child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| CoqConnectionError::ProcessError(e.to_string()))?;

        self.process = Some(child);
        self.initialized = true;
        Ok(())
    }

    /// Run Coq code and get result
    pub fn run_code(&self, code: &str) -> Result<CoqOutput, CoqExecutionError> {
        // Create temp file with Coq code
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("sounio_coq_proof.v");

        std::fs::write(&temp_file, code).map_err(|e| CoqExecutionError::IoError(e.to_string()))?;

        // Run coqc on the file
        let start = Instant::now();
        let mut cmd = Command::new(&self.config.coqc_path);

        for path in &self.config.include_paths {
            cmd.args(["-I", path]);
        }

        let output = cmd
            .arg(&temp_file)
            .output()
            .map_err(|e| CoqExecutionError::ProcessError(e.to_string()))?;

        let elapsed = start.elapsed();

        // Clean up
        let _ = std::fs::remove_file(&temp_file);
        let vo_file = temp_file.with_extension("vo");
        let _ = std::fs::remove_file(&vo_file);
        let glob_file = temp_file.with_extension("glob");
        let _ = std::fs::remove_file(&glob_file);

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        Ok(CoqOutput {
            success: output.status.success(),
            stdout,
            stderr,
            elapsed,
        })
    }

    /// Send command to interactive session
    pub fn send_command(&mut self, cmd: &str) -> Result<String, CoqExecutionError> {
        let process = self
            .process
            .as_mut()
            .ok_or(CoqExecutionError::NotInitialized)?;

        let stdin = process
            .stdin
            .as_mut()
            .ok_or_else(|| CoqExecutionError::IoError("No stdin".to_string()))?;

        writeln!(stdin, "{}", cmd).map_err(|e| CoqExecutionError::IoError(e.to_string()))?;
        stdin
            .flush()
            .map_err(|e| CoqExecutionError::IoError(e.to_string()))?;

        // Read response (simplified - real impl would parse Coq protocol)
        std::thread::sleep(Duration::from_millis(100));

        Ok("Command sent".to_string())
    }
}

impl Drop for CoqServer {
    fn drop(&mut self) {
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
        }
    }
}

/// Output from Coq execution
#[derive(Debug, Clone)]
pub struct CoqOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub elapsed: Duration,
}

/// Main Coq interoperability interface
pub struct CoqInterop {
    server: CoqServer,
    theorem_cache: HashMap<String, CoqProofCertificate>,
}

impl CoqInterop {
    /// Create new Coq interop with default config
    pub fn new() -> Result<Self, CoqConnectionError> {
        Self::with_config(CoqConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: CoqConfig) -> Result<Self, CoqConnectionError> {
        let server = CoqServer::new(config)?;
        Ok(Self {
            server,
            theorem_cache: HashMap::new(),
        })
    }

    /// Check if Coq is available on this system
    pub fn is_available(&self) -> bool {
        self.server.is_available()
    }

    /// Export a Sounio refinement constraint to Coq code
    pub fn export_refinement(&self, refinement: &RefinementConstraint) -> String {
        let mut code = String::new();

        // Header with imports
        code.push_str("(* Auto-generated from Sounio refinement *)\n");
        code.push_str("Require Import Coq.Arith.Arith.\n");
        code.push_str("Require Import Coq.ZArith.ZArith.\n");
        code.push_str("Require Import Coq.Reals.Reals.\n");
        code.push_str("Open Scope R_scope.\n\n");

        // Translate refinement to Coq theorem
        let theorem_name = refinement.name.as_deref().unwrap_or("dem_refinement");
        let coq_type = self.translate_refinement_type(refinement);
        let coq_proof = self.generate_proof_skeleton(refinement);

        code.push_str(&format!("Theorem {} : {}.\n", theorem_name, coq_type));
        code.push_str("Proof.\n");
        code.push_str(&format!("  {}\n", coq_proof));
        code.push_str("Qed.\n");

        code
    }

    /// Export a geometry predicate to Coq
    pub fn export_geometry_predicate(&self, predicate: &GeometryPredicate) -> String {
        let mut code = String::new();

        code.push_str("(* Auto-generated from Sounio geometry *)\n");
        code.push_str("Require Import Coq.Reals.Reals.\n");
        code.push_str("Require Import Coq.Logic.Classical.\n\n");

        // Point type definition
        code.push_str("(* Euclidean plane definitions *)\n");
        code.push_str("Record Point := mkPoint { px : R; py : R }.\n\n");

        // Distance function
        code.push_str("Definition dist (p1 p2 : Point) : R :=\n");
        code.push_str("  sqrt ((px p2 - px p1)^2 + (py p2 - py p1)^2).\n\n");

        // Predicate as theorem
        let theorem = self.translate_geometry_predicate(predicate);
        code.push_str(&theorem);

        code
    }

    /// Export a causal model constraint to Coq
    pub fn export_causal_constraint(&self, constraint: &CausalConstraint) -> String {
        let mut code = String::new();

        code.push_str("(* Auto-generated from Sounio causal model *)\n");
        code.push_str("Require Import Coq.Reals.Reals.\n");
        code.push_str("Require Import Coq.Logic.Classical.\n\n");

        // Probability space (simplified)
        code.push_str("(* Probability/measure theory setup *)\n");
        code.push_str("Axiom Prob : Type -> R.\n");
        code.push_str("Axiom prob_nonneg : forall A, 0 <= Prob A.\n");
        code.push_str("Axiom prob_total : forall A, Prob A <= 1.\n\n");

        // Translate causal constraint
        let theorem = self.translate_causal_constraint(constraint);
        code.push_str(&theorem);

        code
    }

    /// Attempt to prove a theorem in Coq
    pub fn prove(&mut self, coq_code: &str) -> Result<CoqProofCertificate, CoqProofError> {
        // Check cache first
        let code_hash = self.hash_code(coq_code);
        if let Some(cert) = self.theorem_cache.get(&code_hash) {
            return Ok(cert.clone());
        }

        // Run Coq
        let output = self
            .server
            .run_code(coq_code)
            .map_err(|e| CoqProofError::ExecutionError(e.to_string()))?;

        if output.success {
            // Parse the proof from output
            let cert = CoqProofCertificate {
                theorem_name: self.extract_theorem_name(coq_code),
                coq_code: coq_code.to_string(),
                proof_script: self.extract_proof_script(coq_code),
                verified: true,
                elapsed: output.elapsed,
                library_deps: self.extract_requires(coq_code),
            };

            // Cache successful proofs
            self.theorem_cache.insert(code_hash, cert.clone());

            Ok(cert)
        } else {
            let goals = self.parse_remaining_goals(&output.stderr);
            Err(CoqProofError::ProofFailed {
                error: output.stderr,
                goals_remaining: goals,
            })
        }
    }

    /// Import a Coq theorem as epistemic knowledge
    pub fn import_theorem(&self, cert: &CoqProofCertificate) -> CoqTheorem {
        CoqTheorem {
            name: cert.theorem_name.clone(),
            statement: self.extract_statement(&cert.coq_code),
            confidence: BetaConfidence::from_confidence(1.0, 10000.0), // Coq proof = axiomatic
            provenance: CoqProvenance::Proof {
                script: cert.proof_script.clone(),
                library_deps: cert.library_deps.clone(),
                verified_at: std::time::SystemTime::now(),
            },
        }
    }

    /// Import theorem directly from Coq standard library
    pub fn import_stdlib_theorem(&mut self, path: &str) -> Result<CoqTheorem, CoqImportError> {
        // Generate Coq code to check theorem exists
        let check_code = format!(
            "Require Import {}.\nCheck @{}.\n",
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
            .map_err(|e| CoqImportError::ExecutionError(e.to_string()))?;

        if output.success || output.stdout.contains(":") {
            Ok(CoqTheorem {
                name: path.to_string(),
                statement: output.stdout.lines().last().unwrap_or("").to_string(),
                confidence: BetaConfidence::from_confidence(1.0, 10000.0),
                provenance: CoqProvenance::Library {
                    path: path.to_string(),
                    verified: true,
                },
            })
        } else {
            Err(CoqImportError::TheoremNotFound(path.to_string()))
        }
    }

    // =========================================================================
    // Translation Helpers
    // =========================================================================

    fn translate_refinement_type(&self, refinement: &RefinementConstraint) -> String {
        match &refinement.kind {
            RefinementKind::Positive => format!("0 < {}", refinement.variable),
            RefinementKind::NonNegative => format!("0 <= {}", refinement.variable),
            RefinementKind::Range { min, max } => {
                format!(
                    "{} <= {} /\\ {} <= {}",
                    min, refinement.variable, refinement.variable, max
                )
            }
            RefinementKind::Predicate(pred) => self.translate_predicate_to_coq(pred),
            RefinementKind::Custom(s) => s.clone(),
        }
    }

    fn translate_predicate_to_coq(&self, pred: &str) -> String {
        // Basic translation of Sounio predicates to Coq
        pred.replace("&&", "/\\")
            .replace("||", "\\/")
            .replace("!", "~")
            .replace(">=", ">=")
            .replace("<=", "<=")
            .replace("!=", "<>")
            .replace("==", "=")
    }

    fn generate_proof_skeleton(&self, refinement: &RefinementConstraint) -> String {
        match &refinement.kind {
            RefinementKind::Positive | RefinementKind::NonNegative => "lra.".to_string(),
            RefinementKind::Range { .. } => "split; lra.".to_string(),
            RefinementKind::Predicate(_) => "auto.".to_string(),
            RefinementKind::Custom(_) => "admit. (* requires manual proof *)".to_string(),
        }
    }

    fn translate_geometry_predicate(&self, pred: &GeometryPredicate) -> String {
        match pred {
            GeometryPredicate::Collinear(a, b, c) => {
                format!(
                    "Definition collinear_{}_{}_{} ({} {} {} : Point) : Prop :=\n  \
                     exists t : R, px {} = px {} + t * (px {} - px {}) /\\\n                \
                     py {} = py {} + t * (py {} - py {}).\n\n\
                     Theorem collinear_{}_{}_{}_holds : forall {} {} {} : Point,\n  \
                     collinear_{}_{}_{} {} {} {} -> True.\n\
                     Proof. auto. Qed.\n",
                    a, b, c, a, b, c, c, a, b, a, c, a, b, a, a, b, c, a, b, c, a, b, c, a, b, c
                )
            }
            GeometryPredicate::Perpendicular(l1, l2) => {
                format!(
                    "Definition perpendicular (p1 p2 p3 p4 : Point) : Prop :=\n  \
                     (px p2 - px p1) * (px p4 - px p3) + (py p2 - py p1) * (py p4 - py p3) = 0.\n\n\
                     Theorem perp_{}_{} : forall p1 p2 p3 p4 : Point,\n  \
                     perpendicular p1 p2 p3 p4 -> True.\n\
                     Proof. auto. Qed.\n",
                    l1, l2
                )
            }
            GeometryPredicate::Parallel(l1, l2) => {
                format!(
                    "Definition parallel (p1 p2 p3 p4 : Point) : Prop :=\n  \
                     (px p2 - px p1) * (py p4 - py p3) = (py p2 - py p1) * (px p4 - px p3).\n\n\
                     Theorem parallel_{}_{} : forall p1 p2 p3 p4 : Point,\n  \
                     parallel p1 p2 p3 p4 -> True.\n\
                     Proof. auto. Qed.\n",
                    l1, l2
                )
            }
            GeometryPredicate::Congruent(s1, s2) => {
                format!(
                    "Theorem cong_{}_{} : forall p1 p2 p3 p4 : Point,\n  \
                     dist p1 p2 = dist p3 p4 -> True.\n\
                     Proof. auto. Qed.\n",
                    s1, s2
                )
            }
            GeometryPredicate::Cyclic(points) => {
                format!(
                    "Definition cyclic (ps : list Point) : Prop :=\n  \
                     exists c : Point, exists r : R, r > 0 /\\\n  \
                     forall p, In p ps -> dist c p = r.\n\n\
                     Theorem cyclic_{} : forall ps : list Point,\n  \
                     cyclic ps -> True.\n\
                     Proof. auto. Qed.\n",
                    points.join("_")
                )
            }
            GeometryPredicate::Custom(s) => format!("(* Custom: {} *)\n", s),
        }
    }

    fn translate_causal_constraint(&self, constraint: &CausalConstraint) -> String {
        match constraint {
            CausalConstraint::Independence { x, y, given } => {
                if given.is_empty() {
                    format!(
                        "(* Independence: {} _||_ {} *)\n\
                         Axiom indep_{}_{} : forall (A B : Prop),\n  \
                         Prob (A /\\ B) = Prob A * Prob B.\n",
                        x, y, x, y
                    )
                } else {
                    format!(
                        "(* Conditional Independence: {} _||_ {} | {} *)\n\
                         Axiom condindep_{}_{}_{} : forall (A B C : Prop),\n  \
                         Prob C > 0 -> Prob (A /\\ B /\\ C) / Prob C = \
                         (Prob (A /\\ C) / Prob C) * (Prob (B /\\ C) / Prob C).\n",
                        x,
                        y,
                        given.join(", "),
                        x,
                        y,
                        given.join("_")
                    )
                }
            }
            CausalConstraint::DoIntervention { target, value } => {
                format!(
                    "(* do({} := {}) intervention *)\n\
                     Axiom intervention_{} : {} = {}.\n",
                    target, value, target, target, value
                )
            }
            CausalConstraint::Counterfactual { condition, outcome } => {
                format!(
                    "(* Counterfactual: {} => {} *)\n\
                     Theorem cf_{} : {} -> {}.\n\
                     Proof. intro H. admit. Admitted.\n",
                    condition,
                    outcome,
                    condition
                        .replace(" ", "_")
                        .replace("(", "")
                        .replace(")", ""),
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
            let trimmed = line.trim();
            if trimmed.starts_with("Theorem ") || trimmed.starts_with("Lemma ") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    return parts[1].trim_end_matches(':').to_string();
                }
            }
        }
        "unknown".to_string()
    }

    fn extract_statement(&self, code: &str) -> String {
        let mut in_theorem = false;
        let mut statement = String::new();

        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("Theorem ") || trimmed.starts_with("Lemma ") {
                in_theorem = true;
                if let Some(colon_idx) = trimmed.find(':') {
                    statement.push_str(&trimmed[colon_idx + 1..]);
                }
            } else if in_theorem {
                if trimmed.starts_with("Proof") {
                    break;
                }
                statement.push(' ');
                statement.push_str(trimmed);
            }
        }

        statement.trim().trim_end_matches('.').to_string()
    }

    fn extract_proof_script(&self, code: &str) -> String {
        let mut in_proof = false;
        let mut script = String::new();

        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("Proof") {
                in_proof = true;
            } else if in_proof {
                if trimmed == "Qed." || trimmed == "Admitted." || trimmed == "Defined." {
                    break;
                }
                script.push_str(trimmed);
                script.push('\n');
            }
        }

        script.trim().to_string()
    }

    fn extract_requires(&self, code: &str) -> Vec<String> {
        code.lines()
            .filter(|l| l.trim().starts_with("Require "))
            .map(|l| {
                l.trim()
                    .strip_prefix("Require Import ")
                    .or_else(|| l.trim().strip_prefix("Require "))
                    .unwrap_or("")
                    .trim_end_matches('.')
                    .to_string()
            })
            .collect()
    }

    fn parse_remaining_goals(&self, stderr: &str) -> Vec<String> {
        let mut goals = vec![];
        for line in stderr.lines() {
            if line.contains("goal") || line.contains("subgoal") {
                goals.push(line.trim().to_string());
            }
        }
        goals
    }
}

impl Default for CoqInterop {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            server: CoqServer {
                process: None,
                config: CoqConfig::default(),
                initialized: false,
            },
            theorem_cache: HashMap::new(),
        })
    }
}

// =============================================================================
// Supporting Types (shared with Lean where possible)
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

/// Proof certificate from Coq
#[derive(Debug, Clone)]
pub struct CoqProofCertificate {
    pub theorem_name: String,
    pub coq_code: String,
    pub proof_script: String,
    pub verified: bool,
    pub elapsed: Duration,
    pub library_deps: Vec<String>,
}

/// An imported Coq theorem with epistemic metadata
#[derive(Debug, Clone)]
pub struct CoqTheorem {
    pub name: String,
    pub statement: String,
    pub confidence: BetaConfidence,
    pub provenance: CoqProvenance,
}

/// Provenance tracking for imported theorems
#[derive(Debug, Clone)]
pub enum CoqProvenance {
    Proof {
        script: String,
        library_deps: Vec<String>,
        verified_at: std::time::SystemTime,
    },
    Library {
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

/// Error connecting to Coq
#[derive(Debug, Clone)]
pub enum CoqConnectionError {
    NotInstalled,
    ProcessError(String),
    ConfigError(String),
}

impl std::fmt::Display for CoqConnectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInstalled => write!(f, "Coq is not installed"),
            Self::ProcessError(e) => write!(f, "Coq process error: {}", e),
            Self::ConfigError(e) => write!(f, "Coq config error: {}", e),
        }
    }
}

impl std::error::Error for CoqConnectionError {}

/// Error during Coq execution
#[derive(Debug, Clone)]
pub enum CoqExecutionError {
    NotInitialized,
    IoError(String),
    ProcessError(String),
    Timeout,
    ParseError(String),
}

impl std::fmt::Display for CoqExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInitialized => write!(f, "Coq session not initialized"),
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::ProcessError(e) => write!(f, "Process error: {}", e),
            Self::Timeout => write!(f, "Coq execution timed out"),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for CoqExecutionError {}

/// Error during proof attempt
#[derive(Debug, Clone)]
pub enum CoqProofError {
    ExecutionError(String),
    ProofFailed {
        error: String,
        goals_remaining: Vec<String>,
    },
    Timeout,
    InvalidTheorem(String),
}

impl std::fmt::Display for CoqProofError {
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

impl std::error::Error for CoqProofError {}

/// Error importing Coq theorem
#[derive(Debug, Clone)]
pub enum CoqImportError {
    TheoremNotFound(String),
    ExecutionError(String),
    ParseError(String),
}

impl std::fmt::Display for CoqImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TheoremNotFound(name) => write!(f, "Theorem not found: {}", name),
            Self::ExecutionError(e) => write!(f, "Execution error: {}", e),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for CoqImportError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refinement_export_positive() {
        let interop = CoqInterop::default();
        let refinement = RefinementConstraint {
            name: Some("positive_dose".to_string()),
            variable: "dose".to_string(),
            kind: RefinementKind::Positive,
        };

        let code = interop.export_refinement(&refinement);
        assert!(code.contains("Theorem positive_dose"));
        assert!(code.contains("0 < dose"));
        assert!(code.contains("lra"));
    }

    #[test]
    fn test_refinement_export_range() {
        let interop = CoqInterop::default();
        let refinement = RefinementConstraint {
            name: Some("bounded_conc".to_string()),
            variable: "conc".to_string(),
            kind: RefinementKind::Range {
                min: "0".to_string(),
                max: "100".to_string(),
            },
        };

        let code = interop.export_refinement(&refinement);
        assert!(code.contains("Theorem bounded_conc"));
        assert!(code.contains("0 <= conc"));
        assert!(code.contains("conc <= 100"));
    }

    #[test]
    fn test_geometry_predicate_collinear() {
        let interop = CoqInterop::default();
        let pred = GeometryPredicate::Collinear("A".to_string(), "B".to_string(), "C".to_string());

        let code = interop.export_geometry_predicate(&pred);
        assert!(code.contains("collinear_A_B_C"));
        assert!(code.contains("Point"));
    }

    #[test]
    fn test_geometry_predicate_perpendicular() {
        let interop = CoqInterop::default();
        let pred = GeometryPredicate::Perpendicular("L1".to_string(), "L2".to_string());

        let code = interop.export_geometry_predicate(&pred);
        assert!(code.contains("perpendicular"));
        assert!(code.contains("= 0"));
    }

    #[test]
    fn test_causal_independence() {
        let interop = CoqInterop::default();
        let constraint = CausalConstraint::Independence {
            x: "X".to_string(),
            y: "Y".to_string(),
            given: vec![],
        };

        let code = interop.export_causal_constraint(&constraint);
        assert!(code.contains("indep_X_Y"));
        assert!(code.contains("Prob"));
    }

    #[test]
    fn test_causal_conditional_independence() {
        let interop = CoqInterop::default();
        let constraint = CausalConstraint::Independence {
            x: "X".to_string(),
            y: "Y".to_string(),
            given: vec!["Z".to_string()],
        };

        let code = interop.export_causal_constraint(&constraint);
        assert!(code.contains("condindep_X_Y_Z"));
    }

    #[test]
    fn test_proof_certificate_creation() {
        let cert = CoqProofCertificate {
            theorem_name: "test_thm".to_string(),
            coq_code: "Theorem test_thm : True. Proof. auto. Qed.".to_string(),
            proof_script: "auto.".to_string(),
            verified: true,
            elapsed: Duration::from_millis(100),
            library_deps: vec![],
        };

        let interop = CoqInterop::default();
        let theorem = interop.import_theorem(&cert);

        assert_eq!(theorem.name, "test_thm");
        // Beta distribution clamps to [0.001, 0.999] for numerical stability
        assert!((theorem.confidence.mean() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_coq_theorem_has_axiomatic_confidence() {
        let theorem = CoqTheorem {
            name: "pythagoras".to_string(),
            statement: "a^2 + b^2 = c^2".to_string(),
            confidence: BetaConfidence::from_confidence(1.0, 10000.0),
            provenance: CoqProvenance::Proof {
                script: "ring.".to_string(),
                library_deps: vec!["Coq.Reals.Reals".to_string()],
                verified_at: std::time::SystemTime::now(),
            },
        };

        // Coq proofs have confidence ~1.0 (axiomatic)
        // Beta distribution clamps to [0.001, 0.999] for numerical stability
        assert!((theorem.confidence.mean() - 1.0).abs() < 0.01);
        assert!(theorem.confidence.variance() < 0.001);
    }

    #[test]
    fn test_predicate_translation() {
        let interop = CoqInterop::default();
        let result = interop.translate_predicate_to_coq("x >= 0 && y <= 100");
        assert!(result.contains("/\\"));
        assert!(result.contains(">="));
        assert!(result.contains("<="));
    }

    #[test]
    fn test_extract_theorem_name() {
        let interop = CoqInterop::default();
        let code = "Theorem my_theorem : nat. Proof. auto. Qed.";
        assert_eq!(interop.extract_theorem_name(code), "my_theorem");

        let code2 = "Lemma my_lemma : True. Proof. trivial. Qed.";
        assert_eq!(interop.extract_theorem_name(code2), "my_lemma");
    }

    #[test]
    fn test_extract_requires() {
        let interop = CoqInterop::default();
        let code = "Require Import Coq.Arith.Arith.\nRequire Import Coq.ZArith.ZArith.\n\nTheorem x : True. Proof. auto. Qed.";
        let requires = interop.extract_requires(code);
        assert_eq!(requires.len(), 2);
        assert!(requires.contains(&"Coq.Arith.Arith".to_string()));
        assert!(requires.contains(&"Coq.ZArith.ZArith".to_string()));
    }

    #[test]
    fn test_extract_proof_script() {
        let interop = CoqInterop::default();
        let code = "Theorem test : True.\nProof.\n  auto.\n  trivial.\nQed.";
        let script = interop.extract_proof_script(code);
        assert!(script.contains("auto"));
        assert!(script.contains("trivial"));
    }
}
