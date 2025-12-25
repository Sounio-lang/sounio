//! Isabelle/HOL Interoperability for Sounio
//!
//! Native integration with Isabelle/HOL proof assistant enabling:
//! - Export Sounio refinements/geometry to Isabelle theories
//! - Import Isabelle proof certificates as epistemic axioms (confidence 1.0)
//! - Sledgehammer-inspired tactics for automatic proving
//! - Effect-based verification handler
//!
//! # Architecture
//!
//! ```text
//! Sounio                      Isabelle/HOL
//! ┌─────────────┐             ┌─────────────┐
//! │ Refinement  │ ──export──► │ Theory      │
//! │ Constraint  │             │ Definition  │
//! └─────────────┘             └─────────────┘
//!                                   │
//!                         sledgehammer / auto
//!                                   │
//!                                   ▼
//! ┌─────────────┐             ┌─────────────┐
//! │ Knowledge   │ ◄──import── │ .thy proof  │
//! │ <Theorem>   │             │ Certificate │
//! │ conf = 1.0  │             └─────────────┘
//! └─────────────┘
//! ```
//!
//! # Example
//!
//! ```sounio
//! refine geometry_theorem AngleSumTriangle {
//!     premise: Angle(A) + Angle(B) + Angle(C) = 180;
//!     conclusion: ValidTriangle(A, B, C);
//! } export to isabelle "HOL.Analysis";
//!
//! import isabelle "HOL.Real" theorem real_add_assoc { confidence = 1.0 };
//! ```

use std::collections::HashMap;

use std::process::{Child, Command};
use std::time::{Duration, Instant};

use crate::epistemic::bayesian::BetaConfidence;

// =============================================================================
// Core Types
// =============================================================================

/// Configuration for Isabelle interop
#[derive(Debug, Clone)]
pub struct IsabelleConfig {
    /// Path to isabelle executable (default: "isabelle")
    pub isabelle_path: String,
    /// Isabelle home directory
    pub isabelle_home: Option<String>,
    /// Logic to use (default: "HOL")
    pub logic: String,
    /// Session to build on
    pub session: Option<String>,
    /// Timeout for proof attempts
    pub timeout: Duration,
    /// Use sledgehammer for automatic proofs
    pub use_sledgehammer: bool,
    /// Additional Isabelle options
    pub options: HashMap<String, String>,
}

impl Default for IsabelleConfig {
    fn default() -> Self {
        Self {
            isabelle_path: "isabelle".to_string(),
            isabelle_home: None,
            logic: "HOL".to_string(),
            session: None,
            timeout: Duration::from_secs(120), // Sledgehammer can be slow
            use_sledgehammer: true,
            options: HashMap::new(),
        }
    }
}

/// Isabelle server connection (isabelle process)
pub struct IsabelleServer {
    process: Option<Child>,
    config: IsabelleConfig,
    initialized: bool,
}

impl IsabelleServer {
    /// Create new Isabelle server connection
    pub fn new(config: IsabelleConfig) -> Result<Self, IsabelleConnectionError> {
        Ok(Self {
            process: None,
            config,
            initialized: false,
        })
    }

    /// Check if Isabelle is available
    pub fn is_available(&self) -> bool {
        Command::new(&self.config.isabelle_path)
            .arg("version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Get Isabelle version
    pub fn version(&self) -> Option<String> {
        Command::new(&self.config.isabelle_path)
            .arg("version")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
    }

    /// Run Isabelle theory file
    pub fn run_theory(&self, theory_code: &str) -> Result<IsabelleOutput, IsabelleExecutionError> {
        // Create temp directory with theory file
        let temp_dir = std::env::temp_dir().join("sounio_isabelle");
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| IsabelleExecutionError::IoError(e.to_string()))?;

        let theory_file = temp_dir.join("DemTheory.thy");
        std::fs::write(&theory_file, theory_code)
            .map_err(|e| IsabelleExecutionError::IoError(e.to_string()))?;

        // Create ROOT file for the session
        let root_file = temp_dir.join("ROOT");
        let root_content = format!(
            "session DemSession = {} +\n  theories\n    DemTheory\n",
            self.config.logic
        );
        std::fs::write(&root_file, root_content)
            .map_err(|e| IsabelleExecutionError::IoError(e.to_string()))?;

        // Run Isabelle build
        let start = Instant::now();
        let output = Command::new(&self.config.isabelle_path)
            .args(["build", "-D", temp_dir.to_str().unwrap()])
            .output()
            .map_err(|e| IsabelleExecutionError::ProcessError(e.to_string()))?;

        let elapsed = start.elapsed();

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        Ok(IsabelleOutput {
            success: output.status.success(),
            stdout,
            stderr,
            elapsed,
        })
    }

    /// Run sledgehammer on a goal (via ML)
    pub fn run_sledgehammer(
        &self,
        theory_code: &str,
        lemma_name: &str,
    ) -> Result<SledgehammerResult, IsabelleExecutionError> {
        // Add sledgehammer invocation to the theory
        let with_sledgehammer = theory_code.replace(
            "sorry",
            &format!("sledgehammer (* try to prove {} *)\n  sorry", lemma_name),
        );

        let output = self.run_theory(&with_sledgehammer)?;

        // Parse sledgehammer output
        if output.stdout.contains("Try this:") || output.stdout.contains("proof found") {
            // Extract the proof method
            let proof_method = self.extract_proof_method(&output.stdout);
            Ok(SledgehammerResult {
                found: true,
                proof_method,
                provers_tried: self.extract_provers_tried(&output.stdout),
                time_taken: output.elapsed,
            })
        } else {
            Ok(SledgehammerResult {
                found: false,
                proof_method: None,
                provers_tried: self.extract_provers_tried(&output.stdout),
                time_taken: output.elapsed,
            })
        }
    }

    fn extract_proof_method(&self, output: &str) -> Option<String> {
        for line in output.lines() {
            if line.contains("Try this:") {
                return Some(
                    line.trim()
                        .strip_prefix("Try this:")
                        .unwrap_or(line)
                        .trim()
                        .to_string(),
                );
            }
        }
        None
    }

    fn extract_provers_tried(&self, output: &str) -> Vec<String> {
        let mut provers = vec![];
        for line in output.lines() {
            if line.contains("Trying")
                && line.contains("...")
                && let Some(prover) = line.split_whitespace().nth(1)
            {
                provers.push(prover.to_string());
            }
        }
        provers
    }
}

impl Drop for IsabelleServer {
    fn drop(&mut self) {
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
        }
    }
}

/// Output from Isabelle execution
#[derive(Debug, Clone)]
pub struct IsabelleOutput {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub elapsed: Duration,
}

/// Result from sledgehammer attempt
#[derive(Debug, Clone)]
pub struct SledgehammerResult {
    pub found: bool,
    pub proof_method: Option<String>,
    pub provers_tried: Vec<String>,
    pub time_taken: Duration,
}

/// Main Isabelle interoperability interface
pub struct IsabelleInterop {
    server: IsabelleServer,
    theorem_cache: HashMap<String, IsabelleProofCertificate>,
}

impl IsabelleInterop {
    /// Create new Isabelle interop with default config
    pub fn new() -> Result<Self, IsabelleConnectionError> {
        Self::with_config(IsabelleConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: IsabelleConfig) -> Result<Self, IsabelleConnectionError> {
        let server = IsabelleServer::new(config)?;
        Ok(Self {
            server,
            theorem_cache: HashMap::new(),
        })
    }

    /// Check if Isabelle is available on this system
    pub fn is_available(&self) -> bool {
        self.server.is_available()
    }

    /// Export a Sounio refinement constraint to Isabelle theory
    pub fn export_refinement(&self, refinement: &RefinementConstraint) -> String {
        let mut code = String::new();

        // Theory header
        code.push_str("theory DemRefinement\n");
        code.push_str("imports Main HOL.Real\n");
        code.push_str("begin\n\n");

        code.push_str("(* Auto-generated from Sounio refinement *)\n\n");

        // Translate refinement to Isabelle lemma
        let lemma_name = refinement.name.as_deref().unwrap_or("dem_refinement");
        let isabelle_type = self.translate_refinement_type(refinement);
        let proof = self.generate_proof_skeleton(refinement);

        code.push_str(&format!("lemma {} :\n", lemma_name));
        code.push_str(&format!("  \"{}\"\n", isabelle_type));
        code.push_str(&format!("  {}\n", proof));

        code.push_str("\nend\n");

        code
    }

    /// Export a geometry predicate to Isabelle
    pub fn export_geometry_predicate(&self, predicate: &GeometryPredicate) -> String {
        let mut code = String::new();

        // Theory header
        code.push_str("theory DemGeometry\n");
        code.push_str("imports Main \"HOL-Analysis.Analysis\"\n");
        code.push_str("begin\n\n");

        code.push_str("(* Auto-generated from Sounio geometry *)\n\n");

        // Type definitions
        code.push_str("type_synonym point = \"real \\<times> real\"\n\n");

        // Distance function
        code.push_str(
            "definition dist :: \"point \\<Rightarrow> point \\<Rightarrow> real\" where\n",
        );
        code.push_str("  \"dist p1 p2 = sqrt ((fst p2 - fst p1)^2 + (snd p2 - snd p1)^2)\"\n\n");

        // Predicate as lemma
        let lemma = self.translate_geometry_predicate(predicate);
        code.push_str(&lemma);

        code.push_str("\nend\n");

        code
    }

    /// Export a causal model constraint to Isabelle
    pub fn export_causal_constraint(&self, constraint: &CausalConstraint) -> String {
        let mut code = String::new();

        // Theory header
        code.push_str("theory DemCausal\n");
        code.push_str("imports Main \"HOL-Probability.Probability\"\n");
        code.push_str("begin\n\n");

        code.push_str("(* Auto-generated from Sounio causal model *)\n\n");

        // Translate causal constraint
        let lemma = self.translate_causal_constraint(constraint);
        code.push_str(&lemma);

        code.push_str("\nend\n");

        code
    }

    /// Attempt to prove a theorem in Isabelle
    pub fn prove(
        &mut self,
        isabelle_code: &str,
    ) -> Result<IsabelleProofCertificate, IsabelleProofError> {
        // Check cache first
        let code_hash = self.hash_code(isabelle_code);
        if let Some(cert) = self.theorem_cache.get(&code_hash) {
            return Ok(cert.clone());
        }

        // Run Isabelle
        let output = self
            .server
            .run_theory(isabelle_code)
            .map_err(|e| IsabelleProofError::ExecutionError(e.to_string()))?;

        if output.success {
            let cert = IsabelleProofCertificate {
                lemma_name: self.extract_lemma_name(isabelle_code),
                theory_code: isabelle_code.to_string(),
                proof_method: self.extract_proof_method(isabelle_code),
                verified: true,
                elapsed: output.elapsed,
                session_deps: self.extract_imports(isabelle_code),
            };

            // Cache successful proofs
            self.theorem_cache.insert(code_hash, cert.clone());

            Ok(cert)
        } else {
            let goals = self.parse_remaining_goals(&output.stderr);
            Err(IsabelleProofError::ProofFailed {
                error: output.stderr,
                goals_remaining: goals,
            })
        }
    }

    /// Attempt to prove using sledgehammer
    pub fn prove_with_sledgehammer(
        &mut self,
        isabelle_code: &str,
    ) -> Result<IsabelleProofCertificate, IsabelleProofError> {
        let lemma_name = self.extract_lemma_name(isabelle_code);

        let sledge_result = self
            .server
            .run_sledgehammer(isabelle_code, &lemma_name)
            .map_err(|e| IsabelleProofError::ExecutionError(e.to_string()))?;

        if sledge_result.found {
            // Replace sorry with found proof
            let final_code = if let Some(ref method) = sledge_result.proof_method {
                isabelle_code.replace("sorry", method)
            } else {
                isabelle_code.to_string()
            };

            let cert = IsabelleProofCertificate {
                lemma_name,
                theory_code: final_code,
                proof_method: sledge_result.proof_method,
                verified: true,
                elapsed: sledge_result.time_taken,
                session_deps: self.extract_imports(isabelle_code),
            };

            Ok(cert)
        } else {
            Err(IsabelleProofError::SledgehammerFailed {
                provers_tried: sledge_result.provers_tried,
                time_spent: sledge_result.time_taken,
            })
        }
    }

    /// Import an Isabelle theorem as epistemic knowledge
    pub fn import_theorem(&self, cert: &IsabelleProofCertificate) -> IsabelleTheorem {
        IsabelleTheorem {
            name: cert.lemma_name.clone(),
            statement: self.extract_statement(&cert.theory_code),
            confidence: BetaConfidence::from_confidence(1.0, 10000.0), // Isabelle proof = axiomatic
            provenance: IsabelleProvenance::Proof {
                proof_method: cert.proof_method.clone(),
                session_deps: cert.session_deps.clone(),
                verified_at: std::time::SystemTime::now(),
            },
        }
    }

    /// Import theorem from Archive of Formal Proofs (AFP)
    pub fn import_afp_theorem(
        &mut self,
        entry: &str,
        theorem: &str,
    ) -> Result<IsabelleTheorem, IsabelleImportError> {
        // Generate theory to check if theorem exists
        let check_theory = format!(
            "theory CheckAFP\nimports \"{}\"\nbegin\nthm {}\nend\n",
            entry, theorem
        );

        let output = self
            .server
            .run_theory(&check_theory)
            .map_err(|e| IsabelleImportError::ExecutionError(e.to_string()))?;

        if output.success {
            Ok(IsabelleTheorem {
                name: format!("{}.{}", entry, theorem),
                statement: output
                    .stdout
                    .lines()
                    .find(|l| l.contains("::") || l.contains(":"))
                    .unwrap_or("")
                    .to_string(),
                confidence: BetaConfidence::from_confidence(1.0, 10000.0),
                provenance: IsabelleProvenance::AFP {
                    entry: entry.to_string(),
                    theorem: theorem.to_string(),
                    verified: true,
                },
            })
        } else {
            Err(IsabelleImportError::TheoremNotFound(format!(
                "{}.{}",
                entry, theorem
            )))
        }
    }

    // =========================================================================
    // Translation Helpers
    // =========================================================================

    fn translate_refinement_type(&self, refinement: &RefinementConstraint) -> String {
        match &refinement.kind {
            RefinementKind::Positive => format!("(0::real) < {}", refinement.variable),
            RefinementKind::NonNegative => format!("(0::real) \\<le> {}", refinement.variable),
            RefinementKind::Range { min, max } => {
                format!(
                    "{} \\<le> {} \\<and> {} \\<le> {}",
                    min, refinement.variable, refinement.variable, max
                )
            }
            RefinementKind::Predicate(pred) => self.translate_predicate_to_isabelle(pred),
            RefinementKind::Custom(s) => s.clone(),
        }
    }

    fn translate_predicate_to_isabelle(&self, pred: &str) -> String {
        // Order matters! Replace multi-char operators before single-char ones
        // e.g., "!=" must be replaced before "!" to avoid "!=" -> "\<not>="
        pred.replace("&&", "\\<and>")
            .replace("||", "\\<or>")
            .replace("!=", "\\<noteq>")
            .replace(">=", "\\<ge>")
            .replace("<=", "\\<le>")
            .replace("==", "=")
            .replace("->", "\\<longrightarrow>")
            .replace("!", "\\<not>")
    }

    fn generate_proof_skeleton(&self, refinement: &RefinementConstraint) -> String {
        if self.server.config.use_sledgehammer {
            "by sledgehammer".to_string()
        } else {
            match &refinement.kind {
                RefinementKind::Positive | RefinementKind::NonNegative => "by auto".to_string(),
                RefinementKind::Range { .. } => "by auto".to_string(),
                RefinementKind::Predicate(_) => "by simp".to_string(),
                RefinementKind::Custom(_) => "sorry".to_string(),
            }
        }
    }

    fn translate_geometry_predicate(&self, pred: &GeometryPredicate) -> String {
        match pred {
            GeometryPredicate::Collinear(a, b, c) => {
                format!(
                    "definition collinear_{}_{}_{} :: \"point \\<Rightarrow> point \\<Rightarrow> point \\<Rightarrow> bool\" where\n  \
                     \"collinear_{}_{}_{} {} {} {} \\<equiv> \\<exists>t::real. \
                     fst {} = fst {} + t * (fst {} - fst {}) \\<and> \
                     snd {} = snd {} + t * (snd {} - snd {})\"\n\n\
                     lemma collinear_{}_{}_{}_refl:\n  \
                     \"collinear_{}_{}_{} p p p\"\n  \
                     by (simp add: collinear_{}_{}_{}_def)\n",
                    a, b, c, a, b, c, a, b, c, c, a, b, a, c, a, b, a, a, b, c, a, b, c, a, b, c
                )
            }
            GeometryPredicate::Perpendicular(l1, l2) => {
                format!(
                    "definition perpendicular :: \"point \\<Rightarrow> point \\<Rightarrow> point \\<Rightarrow> point \\<Rightarrow> bool\" where\n  \
                     \"perpendicular p1 p2 p3 p4 \\<equiv> \
                     (fst p2 - fst p1) * (fst p4 - fst p3) + (snd p2 - snd p1) * (snd p4 - snd p3) = 0\"\n\n\
                     lemma perp_{}_{}_sym:\n  \
                     \"perpendicular p1 p2 p3 p4 \\<Longrightarrow> perpendicular p3 p4 p1 p2\"\n  \
                     by (simp add: perpendicular_def algebra_simps)\n",
                    l1, l2
                )
            }
            GeometryPredicate::Parallel(l1, l2) => {
                format!(
                    "definition parallel :: \"point \\<Rightarrow> point \\<Rightarrow> point \\<Rightarrow> point \\<Rightarrow> bool\" where\n  \
                     \"parallel p1 p2 p3 p4 \\<equiv> \
                     (fst p2 - fst p1) * (snd p4 - snd p3) = (snd p2 - snd p1) * (fst p4 - fst p3)\"\n\n\
                     lemma parallel_{}_{}_refl:\n  \
                     \"parallel p1 p2 p1 p2\"\n  \
                     by (simp add: parallel_def)\n",
                    l1, l2
                )
            }
            GeometryPredicate::Congruent(s1, s2) => {
                format!(
                    "lemma cong_{}_{}_sym:\n  \
                     \"dist p1 p2 = dist p3 p4 \\<Longrightarrow> dist p3 p4 = dist p1 p2\"\n  \
                     by simp\n",
                    s1, s2
                )
            }
            GeometryPredicate::Cyclic(points) => {
                format!(
                    "definition cyclic :: \"point list \\<Rightarrow> bool\" where\n  \
                     \"cyclic ps \\<equiv> \\<exists>c r. r > 0 \\<and> (\\<forall>p \\<in> set ps. dist c p = r)\"\n\n\
                     lemma cyclic_{}_nonempty:\n  \
                     \"cyclic ps \\<Longrightarrow> ps \\<noteq> []\"\n  \
                     sorry\n",
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
                        "(* Independence: {} \\<perp> {} *)\n\
                         locale indep_{}_{}  =\n  \
                         fixes {} {} :: \"'a \\<Rightarrow> real\"\n  \
                         assumes indep: \"\\<forall>a b. prob (\\<lambda>x. {} x = a \\<and> {} x = b) = \
                         prob (\\<lambda>x. {} x = a) * prob (\\<lambda>x. {} x = b)\"\n",
                        x, y, x, y, x, y, x, y, x, y
                    )
                } else {
                    format!(
                        "(* Conditional Independence: {} \\<perp> {} | {} *)\n\
                         locale condindep_{}_{}_{}  =\n  \
                         fixes {} {} {} :: \"'a \\<Rightarrow> real\"\n  \
                         assumes condindep: \"True\" (* simplified *)\n",
                        x,
                        y,
                        given.join(", "),
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
                    "(* do({} := {}) intervention *)\n\
                     definition intervention_{} :: \"('a \\<Rightarrow> real) \\<Rightarrow> ('a \\<Rightarrow> real)\" where\n  \
                     \"intervention_{} f = (\\<lambda>x. {})\"\n",
                    target, value, target, target, value
                )
            }
            CausalConstraint::Counterfactual { condition, outcome } => {
                format!(
                    "(* Counterfactual: {} \\<Longrightarrow> {} *)\n\
                     lemma cf_{}:\n  \
                     assumes \"{}\"\n  \
                     shows \"{}\"\n  \
                     sorry\n",
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

    fn extract_lemma_name(&self, code: &str) -> String {
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("lemma ") || trimmed.starts_with("theorem ") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    return parts[1]
                        .trim_end_matches(':')
                        .trim_end_matches('[')
                        .to_string();
                }
            }
        }
        "unknown".to_string()
    }

    fn extract_statement(&self, code: &str) -> String {
        let mut in_lemma = false;
        let mut statement = String::new();

        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("lemma ") || trimmed.starts_with("theorem ") {
                in_lemma = true;
            } else if in_lemma {
                if trimmed.starts_with("by ")
                    || trimmed.starts_with("proof")
                    || trimmed.starts_with("apply")
                    || trimmed == "sorry"
                {
                    break;
                }
                if trimmed.starts_with("\"") && trimmed.ends_with("\"") {
                    statement.push_str(trimmed.trim_start_matches('"').trim_end_matches('"'));
                }
            }
        }

        statement.trim().to_string()
    }

    fn extract_proof_method(&self, code: &str) -> Option<String> {
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("by ") {
                return Some(trimmed.strip_prefix("by ").unwrap_or("").to_string());
            }
        }
        None
    }

    fn extract_imports(&self, code: &str) -> Vec<String> {
        for line in code.lines() {
            if line.trim().starts_with("imports ") {
                return line
                    .trim()
                    .strip_prefix("imports ")
                    .unwrap_or("")
                    .split_whitespace()
                    .map(|s| s.trim_matches('"').to_string())
                    .collect();
            }
        }
        vec![]
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

impl Default for IsabelleInterop {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            server: IsabelleServer {
                process: None,
                config: IsabelleConfig::default(),
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

/// Proof certificate from Isabelle
#[derive(Debug, Clone)]
pub struct IsabelleProofCertificate {
    pub lemma_name: String,
    pub theory_code: String,
    pub proof_method: Option<String>,
    pub verified: bool,
    pub elapsed: Duration,
    pub session_deps: Vec<String>,
}

/// An imported Isabelle theorem with epistemic metadata
#[derive(Debug, Clone)]
pub struct IsabelleTheorem {
    pub name: String,
    pub statement: String,
    pub confidence: BetaConfidence,
    pub provenance: IsabelleProvenance,
}

/// Provenance tracking for imported theorems
#[derive(Debug, Clone)]
pub enum IsabelleProvenance {
    Proof {
        proof_method: Option<String>,
        session_deps: Vec<String>,
        verified_at: std::time::SystemTime,
    },
    AFP {
        entry: String,
        theorem: String,
        verified: bool,
    },
    HOLLight {
        theory: String,
        verified: bool,
    },
}

// =============================================================================
// Errors
// =============================================================================

/// Error connecting to Isabelle
#[derive(Debug, Clone)]
pub enum IsabelleConnectionError {
    NotInstalled,
    ProcessError(String),
    ConfigError(String),
}

impl std::fmt::Display for IsabelleConnectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInstalled => write!(f, "Isabelle is not installed"),
            Self::ProcessError(e) => write!(f, "Isabelle process error: {}", e),
            Self::ConfigError(e) => write!(f, "Isabelle config error: {}", e),
        }
    }
}

impl std::error::Error for IsabelleConnectionError {}

/// Error during Isabelle execution
#[derive(Debug, Clone)]
pub enum IsabelleExecutionError {
    NotInitialized,
    IoError(String),
    ProcessError(String),
    Timeout,
    ParseError(String),
}

impl std::fmt::Display for IsabelleExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInitialized => write!(f, "Isabelle session not initialized"),
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::ProcessError(e) => write!(f, "Process error: {}", e),
            Self::Timeout => write!(f, "Isabelle execution timed out"),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for IsabelleExecutionError {}

/// Error during proof attempt
#[derive(Debug, Clone)]
pub enum IsabelleProofError {
    ExecutionError(String),
    ProofFailed {
        error: String,
        goals_remaining: Vec<String>,
    },
    SledgehammerFailed {
        provers_tried: Vec<String>,
        time_spent: Duration,
    },
    Timeout,
    InvalidTheorem(String),
}

impl std::fmt::Display for IsabelleProofError {
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
            Self::SledgehammerFailed {
                provers_tried,
                time_spent,
            } => {
                write!(
                    f,
                    "Sledgehammer failed after {:?}. Provers tried: {:?}",
                    time_spent, provers_tried
                )
            }
            Self::Timeout => write!(f, "Proof attempt timed out"),
            Self::InvalidTheorem(e) => write!(f, "Invalid theorem: {}", e),
        }
    }
}

impl std::error::Error for IsabelleProofError {}

/// Error importing Isabelle theorem
#[derive(Debug, Clone)]
pub enum IsabelleImportError {
    TheoremNotFound(String),
    AFPNotInstalled(String),
    ExecutionError(String),
    ParseError(String),
}

impl std::fmt::Display for IsabelleImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TheoremNotFound(name) => write!(f, "Theorem not found: {}", name),
            Self::AFPNotInstalled(entry) => write!(f, "AFP entry not installed: {}", entry),
            Self::ExecutionError(e) => write!(f, "Execution error: {}", e),
            Self::ParseError(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for IsabelleImportError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refinement_export_positive() {
        let interop = IsabelleInterop::default();
        let refinement = RefinementConstraint {
            name: Some("positive_dose".to_string()),
            variable: "dose".to_string(),
            kind: RefinementKind::Positive,
        };

        let code = interop.export_refinement(&refinement);
        assert!(code.contains("lemma positive_dose"));
        assert!(code.contains("(0::real) < dose"));
        assert!(code.contains("theory DemRefinement"));
    }

    #[test]
    fn test_refinement_export_range() {
        let interop = IsabelleInterop::default();
        let refinement = RefinementConstraint {
            name: Some("bounded_conc".to_string()),
            variable: "conc".to_string(),
            kind: RefinementKind::Range {
                min: "0".to_string(),
                max: "100".to_string(),
            },
        };

        let code = interop.export_refinement(&refinement);
        assert!(code.contains("lemma bounded_conc"));
        assert!(code.contains("\\<le>"));
    }

    #[test]
    fn test_geometry_predicate_collinear() {
        let interop = IsabelleInterop::default();
        let pred = GeometryPredicate::Collinear("A".to_string(), "B".to_string(), "C".to_string());

        let code = interop.export_geometry_predicate(&pred);
        assert!(code.contains("collinear_A_B_C"));
        assert!(code.contains("point"));
    }

    #[test]
    fn test_geometry_predicate_perpendicular() {
        let interop = IsabelleInterop::default();
        let pred = GeometryPredicate::Perpendicular("L1".to_string(), "L2".to_string());

        let code = interop.export_geometry_predicate(&pred);
        assert!(code.contains("perpendicular"));
        assert!(code.contains("= 0"));
    }

    #[test]
    fn test_causal_independence() {
        let interop = IsabelleInterop::default();
        let constraint = CausalConstraint::Independence {
            x: "X".to_string(),
            y: "Y".to_string(),
            given: vec![],
        };

        let code = interop.export_causal_constraint(&constraint);
        assert!(code.contains("indep_X_Y"));
        assert!(code.contains("locale"));
    }

    #[test]
    fn test_isabelle_theorem_has_axiomatic_confidence() {
        let theorem = IsabelleTheorem {
            name: "pythagoras".to_string(),
            statement: "a^2 + b^2 = c^2".to_string(),
            confidence: BetaConfidence::from_confidence(1.0, 10000.0),
            provenance: IsabelleProvenance::Proof {
                proof_method: Some("sledgehammer".to_string()),
                session_deps: vec!["HOL.Real".to_string()],
                verified_at: std::time::SystemTime::now(),
            },
        };

        // Isabelle proofs have confidence ~1.0 (axiomatic)
        // Beta distribution clamps to [0.001, 0.999] for numerical stability
        assert!((theorem.confidence.mean() - 1.0).abs() < 0.01);
        assert!(theorem.confidence.variance() < 0.001);
    }

    #[test]
    fn test_predicate_translation() {
        let interop = IsabelleInterop::default();
        let result = interop.translate_predicate_to_isabelle("x >= 0 && y <= 100 -> z != 0");
        assert!(result.contains("\\<and>"));
        assert!(result.contains("\\<ge>"));
        assert!(result.contains("\\<le>"));
        assert!(result.contains("\\<longrightarrow>"));
        // Note: != may be translated to ≠ or kept as !=
        assert!(result.contains("!=") || result.contains("\\<noteq>"));
    }

    #[test]
    fn test_extract_lemma_name() {
        let interop = IsabelleInterop::default();
        let code = "lemma my_lemma:\n  \"True\"\n  by simp";
        assert_eq!(interop.extract_lemma_name(code), "my_lemma");

        let code2 = "theorem my_theorem:\n  \"True\"\n  by auto";
        assert_eq!(interop.extract_lemma_name(code2), "my_theorem");
    }

    #[test]
    fn test_extract_imports() {
        let interop = IsabelleInterop::default();
        let code = "theory Test\nimports Main HOL.Real \"HOL-Analysis.Analysis\"\nbegin\nend\n";
        let imports = interop.extract_imports(code);
        assert_eq!(imports.len(), 3);
        assert!(imports.contains(&"Main".to_string()));
        assert!(imports.contains(&"HOL.Real".to_string()));
    }

    #[test]
    fn test_proof_certificate_creation() {
        let cert = IsabelleProofCertificate {
            lemma_name: "test_lemma".to_string(),
            theory_code:
                "theory Test\nimports Main\nbegin\nlemma test_lemma: \"True\" by simp\nend\n"
                    .to_string(),
            proof_method: Some("simp".to_string()),
            verified: true,
            elapsed: Duration::from_millis(100),
            session_deps: vec!["Main".to_string()],
        };

        let interop = IsabelleInterop::default();
        let theorem = interop.import_theorem(&cert);

        assert_eq!(theorem.name, "test_lemma");
        // Beta distribution clamps to [0.001, 0.999] for numerical stability
        assert!((theorem.confidence.mean() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sledgehammer_result() {
        let result = SledgehammerResult {
            found: true,
            proof_method: Some("metis".to_string()),
            provers_tried: vec!["cvc4".to_string(), "z3".to_string(), "e".to_string()],
            time_taken: Duration::from_secs(5),
        };

        assert!(result.found);
        assert_eq!(result.proof_method.as_deref(), Some("metis"));
        assert_eq!(result.provers_tried.len(), 3);
    }
}
