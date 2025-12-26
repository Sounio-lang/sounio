// tests/integration/common/mod.rs — Test infrastructure for end-to-end tests
//
// Provides:
// - TestHarness: Run compiler end-to-end with controlled environment
// - CompileResult: Parse and assert on compilation outcomes
// - Golden file comparison utilities
// - Macros for common test patterns

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use std::time::{Duration, Instant};

// ============================================================================
// Test Harness
// ============================================================================

/// Main test harness for running compiler end-to-end
pub struct TestHarness {
    /// Path to compiler binary
    compiler_path: PathBuf,
    /// Temporary directory for test outputs
    temp_dir: PathBuf,
    /// Environment variables to set
    env_vars: HashMap<String, String>,
    /// Additional compiler flags
    flags: Vec<String>,
    /// Ontology paths to include
    ontology_paths: Vec<PathBuf>,
    /// Whether to capture timing information
    capture_timing: bool,
}

impl TestHarness {
    /// Create a new test harness
    pub fn new() -> Self {
        // Look for compiler binary (souc / souc.exe on Windows)
        // Check release first, then fall back to debug
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        #[cfg(windows)]
        let binary_name = "souc.exe";
        #[cfg(not(windows))]
        let binary_name = "souc";
        let release_path = manifest_dir.join(format!("target/release/{}", binary_name));
        let debug_path = manifest_dir.join(format!("target/debug/{}", binary_name));
        let compiler_path = if release_path.exists() {
            release_path
        } else {
            debug_path
        };

        // Use both process id and a counter to make temp dirs unique across parallel tests
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique_id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let temp_dir = std::env::temp_dir().join(format!(
            "sounio_test_{}_{}",
            std::process::id(),
            unique_id
        ));
        fs::create_dir_all(&temp_dir).expect("Failed to create temp dir");

        Self {
            compiler_path,
            temp_dir,
            env_vars: HashMap::new(),
            flags: Vec::new(),
            ontology_paths: Vec::new(),
            capture_timing: false,
        }
    }

    /// Add an environment variable
    pub fn env(mut self, key: &str, value: &str) -> Self {
        self.env_vars.insert(key.to_string(), value.to_string());
        self
    }

    /// Add a compiler flag
    pub fn flag(mut self, flag: &str) -> Self {
        self.flags.push(flag.to_string());
        self
    }

    /// Add multiple compiler flags
    pub fn flags(mut self, flags: &[&str]) -> Self {
        self.flags.extend(flags.iter().map(|s| s.to_string()));
        self
    }

    /// Add an ontology path
    pub fn ontology(mut self, path: impl AsRef<Path>) -> Self {
        self.ontology_paths.push(path.as_ref().to_path_buf());
        self
    }

    /// Enable timing capture
    pub fn with_timing(mut self) -> Self {
        self.capture_timing = true;
        self
    }

    /// Set semantic distance threshold
    pub fn threshold(self, value: f64) -> Self {
        self.flag(&format!("--semantic-threshold={}", value))
    }

    /// Enable JSON diagnostic output
    pub fn json_diagnostics(self) -> Self {
        self.flag("--error-format=json")
    }

    /// Compile source code from a string
    pub fn compile_str(&self, name: &str, source: &str) -> CompileResult {
        let source_path = self.temp_dir.join(format!("{}.dm", name));
        fs::write(&source_path, source).expect("Failed to write source file");
        self.compile_file(&source_path)
    }

    /// Compile a source file
    pub fn compile_file(&self, path: &Path) -> CompileResult {
        let mut cmd = Command::new(&self.compiler_path);
        // Use 'check' subcommand for type checking
        cmd.arg("check").arg(path);

        // Add ontology paths
        for onto_path in &self.ontology_paths {
            cmd.arg("--ontology").arg(onto_path);
        }

        // Add flags
        for flag in &self.flags {
            cmd.arg(flag);
        }

        // Set environment
        for (key, value) in &self.env_vars {
            cmd.env(key, value);
        }

        let start = Instant::now();
        let output = cmd.output().expect("Failed to execute compiler");
        let duration = start.elapsed();

        CompileResult::new(output, duration, self.capture_timing)
    }

    /// Compile multiple files as a project
    pub fn compile_project(&self, files: &[(&str, &str)]) -> CompileResult {
        let project_dir = self.temp_dir.join("project");
        fs::create_dir_all(&project_dir).expect("Failed to create project dir");

        let mut paths = Vec::new();
        for (name, source) in files {
            let path = project_dir.join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).ok();
            }
            fs::write(&path, source).expect("Failed to write source file");
            paths.push(path);
        }

        // Compile main file (first one)
        if let Some(main_path) = paths.first() {
            self.compile_file(main_path)
        } else {
            panic!("No files provided to compile_project");
        }
    }

    /// Get path to test fixtures
    pub fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    /// Get path to golden files
    pub fn golden_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/golden")
            .join(name)
    }
}

impl Default for TestHarness {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TestHarness {
    fn drop(&mut self) {
        // Clean up temp directory
        let _ = fs::remove_dir_all(&self.temp_dir);
    }
}

// ============================================================================
// Compile Result
// ============================================================================

/// Result of a compilation
#[derive(Debug)]
pub struct CompileResult {
    /// Raw process output
    output: Output,
    /// Parsed diagnostics (if JSON output was enabled)
    diagnostics: Vec<Diagnostic>,
    /// Compilation duration
    duration: Option<Duration>,
    /// Whether compilation succeeded
    success: bool,
    /// Exit code
    exit_code: i32,
}

impl CompileResult {
    fn new(output: Output, duration: Duration, capture_timing: bool) -> Self {
        let success = output.status.success();
        let exit_code = output.status.code().unwrap_or(-1);

        // Try to parse JSON diagnostics from stderr
        let stderr = String::from_utf8_lossy(&output.stderr);
        let diagnostics = Self::parse_diagnostics(&stderr);

        Self {
            output,
            diagnostics,
            duration: if capture_timing { Some(duration) } else { None },
            success,
            exit_code,
        }
    }

    fn parse_diagnostics(stderr: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        for line in stderr.lines() {
            if line.starts_with('{')
                && let Ok(diag) = serde_json::from_str::<Diagnostic>(line)
            {
                diagnostics.push(diag);
            }
        }

        diagnostics
    }

    // === Success/Failure Assertions ===

    /// Assert compilation succeeded
    pub fn assert_success(&self) -> &Self {
        assert!(
            self.success,
            "Expected compilation to succeed, but it failed with:\n{}",
            String::from_utf8_lossy(&self.output.stderr)
        );
        self
    }

    /// Assert compilation failed
    pub fn assert_failure(&self) -> &Self {
        assert!(
            !self.success,
            "Expected compilation to fail, but it succeeded"
        );
        self
    }

    /// Assert specific exit code
    pub fn assert_exit_code(&self, expected: i32) -> &Self {
        assert_eq!(
            self.exit_code, expected,
            "Expected exit code {}, got {}",
            expected, self.exit_code
        );
        self
    }

    // === Diagnostic Assertions ===

    /// Assert that a specific error code was emitted
    pub fn assert_error(&self, code: &str) -> &Self {
        assert!(
            self.has_error(code),
            "Expected error '{}' but it was not emitted.\nActual errors: {:?}",
            code,
            self.error_codes()
        );
        self
    }

    /// Assert that a specific warning was emitted
    pub fn assert_warning(&self, code: &str) -> &Self {
        assert!(
            self.has_warning(code),
            "Expected warning '{}' but it was not emitted.\nActual warnings: {:?}",
            code,
            self.warning_codes()
        );
        self
    }

    /// Assert no errors were emitted
    pub fn assert_no_errors(&self) -> &Self {
        let errors: Vec<_> = self
            .diagnostics
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "Expected no errors, but found: {:?}",
            errors
        );
        self
    }

    /// Assert exact number of errors
    pub fn assert_error_count(&self, count: usize) -> &Self {
        let actual = self.error_count();
        assert_eq!(actual, count, "Expected {} errors, found {}", count, actual);
        self
    }

    /// Assert error message contains text
    pub fn assert_error_contains(&self, text: &str) -> &Self {
        let stderr = self.stderr();
        assert!(
            stderr.contains(text),
            "Expected error output to contain '{}'\nActual output:\n{}",
            text,
            stderr
        );
        self
    }

    /// Assert error at specific location
    pub fn assert_error_at(&self, code: &str, line: u32, column: u32) -> &Self {
        let matching = self
            .diagnostics
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .filter(|d| d.code.as_deref() == Some(code))
            .filter(|d| {
                d.location
                    .as_ref()
                    .is_some_and(|loc| loc.line == line && loc.column == column)
            })
            .count();

        assert!(
            matching > 0,
            "Expected error '{}' at line {}, column {}, but not found.\nActual diagnostics: {:?}",
            code,
            line,
            column,
            self.diagnostics
        );
        self
    }

    // === Semantic Distance Assertions ===

    /// Assert a semantic distance suggestion was provided
    pub fn assert_distance_suggestion(&self, from: &str, to: &str) -> &Self {
        let has_suggestion = self.diagnostics.iter().any(|d| {
            d.suggestions.iter().any(|s| {
                s.semantic_distance
                    .as_ref()
                    .is_some_and(|sd| sd.from_type.contains(from) && sd.to_type.contains(to))
            })
        });

        assert!(
            has_suggestion,
            "Expected distance suggestion from '{}' to '{}', but not found",
            from, to
        );
        self
    }

    /// Assert semantic distance is within bounds
    pub fn assert_distance_within(&self, max_distance: f64) -> &Self {
        for diag in &self.diagnostics {
            for sugg in &diag.suggestions {
                if let Some(sd) = &sugg.semantic_distance {
                    assert!(
                        sd.distance <= max_distance,
                        "Semantic distance {} exceeds maximum {}",
                        sd.distance,
                        max_distance
                    );
                }
            }
        }
        self
    }

    // === Performance Assertions ===

    /// Assert compilation completed within time limit
    pub fn assert_duration_under(&self, max: Duration) -> &Self {
        if let Some(duration) = self.duration {
            assert!(
                duration <= max,
                "Compilation took {:?}, exceeds limit of {:?}",
                duration,
                max
            );
        }
        self
    }

    /// Assert compilation took at least minimum time (for perf tests)
    pub fn assert_duration_over(&self, min: Duration) -> &Self {
        if let Some(duration) = self.duration {
            assert!(
                duration >= min,
                "Compilation took {:?}, under minimum of {:?}",
                duration,
                min
            );
        }
        self
    }

    // === Output Assertions ===

    /// Assert stdout contains text
    pub fn assert_stdout_contains(&self, text: &str) -> &Self {
        let stdout = self.stdout();
        assert!(
            stdout.contains(text),
            "Expected stdout to contain '{}'\nActual:\n{}",
            text,
            stdout
        );
        self
    }

    /// Assert stderr contains text
    pub fn assert_stderr_contains(&self, text: &str) -> &Self {
        let stderr = self.stderr();
        assert!(
            stderr.contains(text),
            "Expected stderr to contain '{}'\nActual:\n{}",
            text,
            stderr
        );
        self
    }

    // === Golden File Assertions ===

    /// Compare stderr against golden file
    pub fn assert_stderr_matches_golden(&self, golden_name: &str) -> &Self {
        let golden_path = TestHarness::golden_path(golden_name);
        self.compare_golden(&golden_path, &self.stderr())
    }

    /// Compare stdout against golden file
    pub fn assert_stdout_matches_golden(&self, golden_name: &str) -> &Self {
        let golden_path = TestHarness::golden_path(golden_name);
        self.compare_golden(&golden_path, &self.stdout())
    }

    fn compare_golden(&self, golden_path: &Path, actual: &str) -> &Self {
        // Normalize line endings and whitespace
        let actual_normalized = normalize_output(actual);

        if std::env::var("UPDATE_GOLDEN").is_ok() {
            // Update mode: write actual output as new golden
            if let Some(parent) = golden_path.parent() {
                fs::create_dir_all(parent).ok();
            }
            fs::write(golden_path, &actual_normalized).expect("Failed to write golden file");
            return self;
        }

        if !golden_path.exists() {
            panic!(
                "Golden file not found: {:?}\nSet UPDATE_GOLDEN=1 to create it.\nActual output:\n{}",
                golden_path, actual_normalized
            );
        }

        let expected = fs::read_to_string(golden_path).expect("Failed to read golden file");
        let expected_normalized = normalize_output(&expected);

        if actual_normalized != expected_normalized {
            // Generate diff
            let diff = generate_diff(&expected_normalized, &actual_normalized);
            panic!(
                "Output differs from golden file {:?}\n\nDiff:\n{}\n\nSet UPDATE_GOLDEN=1 to update.",
                golden_path, diff
            );
        }

        self
    }

    // === Accessors ===

    pub fn stdout(&self) -> String {
        String::from_utf8_lossy(&self.output.stdout).to_string()
    }

    pub fn stderr(&self) -> String {
        String::from_utf8_lossy(&self.output.stderr).to_string()
    }

    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn duration(&self) -> Option<Duration> {
        self.duration
    }

    pub fn success(&self) -> bool {
        self.success
    }

    pub fn exit_code(&self) -> i32 {
        self.exit_code
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Warning)
            .count()
    }

    fn has_error(&self, code: &str) -> bool {
        self.diagnostics
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .any(|d| d.code.as_deref() == Some(code))
    }

    fn has_warning(&self, code: &str) -> bool {
        self.diagnostics
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Warning)
            .any(|d| d.code.as_deref() == Some(code))
    }

    fn error_codes(&self) -> Vec<&str> {
        self.diagnostics
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Error)
            .filter_map(|d| d.code.as_deref())
            .collect()
    }

    fn warning_codes(&self) -> Vec<&str> {
        self.diagnostics
            .iter()
            .filter(|d| d.level == DiagnosticLevel::Warning)
            .filter_map(|d| d.code.as_deref())
            .collect()
    }
}

// ============================================================================
// Diagnostic Types
// ============================================================================

/// A parsed diagnostic from JSON output
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Diagnostic {
    /// Error code (e.g., "E0308", "CHEBI_MISMATCH")
    pub code: Option<String>,
    /// Diagnostic message
    pub message: String,
    /// Severity level
    pub level: DiagnosticLevel,
    /// Primary location
    pub location: Option<Location>,
    /// Related locations
    #[serde(default)]
    pub related: Vec<RelatedInfo>,
    /// Fix suggestions
    #[serde(default)]
    pub suggestions: Vec<Suggestion>,
    /// Additional notes
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Diagnostic severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Note,
    Help,
}

/// Source location
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Location {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub end_line: Option<u32>,
    pub end_column: Option<u32>,
}

/// Related diagnostic information
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RelatedInfo {
    pub message: String,
    pub location: Location,
}

/// A fix suggestion
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Suggestion {
    pub message: String,
    /// Code to insert/replace
    pub replacement: Option<String>,
    /// Semantic distance info (for type mismatches)
    pub semantic_distance: Option<SemanticDistance>,
}

/// Semantic distance information
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SemanticDistance {
    pub from_type: String,
    pub to_type: String,
    pub distance: f64,
    /// Component breakdown
    pub path_distance: Option<f64>,
    pub ic_distance: Option<f64>,
    pub embedding_distance: Option<f64>,
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Normalize output for comparison
fn normalize_output(s: &str) -> String {
    // Regex to match temp directory paths like /tmp/sounio_test_12345/ or /tmp/sounio_test_12345_67/
    let temp_path_re =
        regex::Regex::new(r"/tmp/sounio_test_\d+(_\d+)?/").expect("Invalid temp path regex");

    s.lines()
        .map(|line| line.trim_end())
        .map(|line| {
            temp_path_re
                .replace_all(line, "/tmp/sounio_test_XXXX/")
                .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Generate a simple diff between two strings
fn generate_diff(expected: &str, actual: &str) -> String {
    let mut diff = String::new();
    let expected_lines: Vec<_> = expected.lines().collect();
    let actual_lines: Vec<_> = actual.lines().collect();

    let max_lines = expected_lines.len().max(actual_lines.len());

    for i in 0..max_lines {
        let exp = expected_lines.get(i).copied().unwrap_or("");
        let act = actual_lines.get(i).copied().unwrap_or("");

        if exp != act {
            if !exp.is_empty() {
                diff.push_str(&format!("-{}: {}\n", i + 1, exp));
            }
            if !act.is_empty() {
                diff.push_str(&format!("+{}: {}\n", i + 1, act));
            }
        }
    }

    if diff.is_empty() {
        "(no visible differences - check whitespace)".to_string()
    } else {
        diff
    }
}

// ============================================================================
// Test Macros
// ============================================================================

/// Define a test case that compiles source and checks for success
#[macro_export]
macro_rules! compile_pass {
    ($name:ident, $source:expr) => {
        #[test]
        fn $name() {
            let harness = $crate::integration::common::TestHarness::new();
            harness.compile_str(stringify!($name), $source)
                .assert_success();
        }
    };
    ($name:ident, $source:expr, $($assertion:tt)*) => {
        #[test]
        fn $name() {
            let harness = $crate::integration::common::TestHarness::new();
            let result = harness.compile_str(stringify!($name), $source)
                .assert_success();
            $($assertion)*
        }
    };
}

/// Define a test case that compiles source and expects failure
#[macro_export]
macro_rules! compile_fail {
    ($name:ident, $source:expr) => {
        #[test]
        fn $name() {
            let harness = $crate::integration::common::TestHarness::new();
            harness.compile_str(stringify!($name), $source)
                .assert_failure();
        }
    };
    ($name:ident, $source:expr, $error:expr) => {
        #[test]
        fn $name() {
            let harness = $crate::integration::common::TestHarness::new();
            harness.json_diagnostics()
                .compile_str(stringify!($name), $source)
                .assert_failure()
                .assert_error($error);
        }
    };
    ($name:ident, $source:expr, $error:expr, $($assertion:tt)*) => {
        #[test]
        fn $name() {
            let harness = $crate::integration::common::TestHarness::new();
            let result = harness.json_diagnostics()
                .compile_str(stringify!($name), $source)
                .assert_failure()
                .assert_error($error);
            $($assertion)*
        }
    };
}

/// Define a test case with custom harness configuration
#[macro_export]
macro_rules! test_case {
    ($name:ident, |$harness:ident| $body:block) => {
        #[test]
        fn $name() {
            let $harness = $crate::integration::common::TestHarness::new();
            $body
        }
    };
}

/// Define a golden file test
#[macro_export]
macro_rules! golden_test {
    ($name:ident, $source:expr, $golden:expr) => {
        #[test]
        fn $name() {
            let harness = $crate::integration::common::TestHarness::new();
            harness
                .compile_str(stringify!($name), $source)
                .assert_stderr_matches_golden($golden);
        }
    };
}

// ============================================================================
// Test Fixtures
// ============================================================================

/// Standard ontology test data
pub mod fixtures {
    /// ChEBI test molecules
    pub const CHEBI_ASPIRIN: &str = "CHEBI:15365";
    pub const CHEBI_IBUPROFEN: &str = "CHEBI:5855";
    pub const CHEBI_GLUCOSE: &str = "CHEBI:17234";
    pub const CHEBI_ETHANOL: &str = "CHEBI:16236";
    pub const CHEBI_WATER: &str = "CHEBI:15377";

    /// GO test terms
    pub const GO_APOPTOSIS: &str = "GO:0006915";
    pub const GO_CELL_CYCLE: &str = "GO:0007049";
    pub const GO_METABOLISM: &str = "GO:0008152";

    /// HP test terms
    pub const HP_HEADACHE: &str = "HP:0002315";
    pub const HP_FEVER: &str = "HP:0001945";
    pub const HP_FATIGUE: &str = "HP:0012378";

    /// MONDO test terms
    pub const MONDO_DIABETES: &str = "MONDO:0005015";
    pub const MONDO_HYPERTENSION: &str = "MONDO:0001134";
    pub const MONDO_CANCER: &str = "MONDO:0004992";

    /// Sample Sounio source files
    pub fn simple_phenotype_check() -> &'static str {
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology hp from "https://purl.obolibrary.org/obo/hp.owl";

type Drug = chebi:entity;
type Phenotype = hp:entity;

fn check_indication(drug: Drug, symptom: Phenotype) -> bool {
    // Simple check - real impl would query a knowledge base
    true
}

fn main() {
    let aspirin: Drug = chebi:15365;
    let headache: Phenotype = hp:0002315;
    check_indication(aspirin, headache);
}
"#
    }

    pub fn type_mismatch_example() -> &'static str {
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology go from "https://purl.obolibrary.org/obo/go.owl";

type Drug = chebi:entity;
type Process = go:biological_process;

fn analyze_drug(d: Drug) -> Drug {
    d
}

fn main() {
    let process: Process = go:0006915;
    // Error: passing GO term where ChEBI expected
    analyze_drug(process);
}
"#
    }

    pub fn semantic_coercion_example() -> &'static str {
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

type ChEBIDrug = chebi:drug;
type DrugBankEntity = drugbank:drug;

#[compat(threshold = 0.3)]
fn process_drug(d: ChEBIDrug) {
    // Process the drug
}

fn main() {
    let db_drug: DrugBankEntity = drugbank:DB00945;
    // Should coerce if semantic distance < 0.3
    process_drug(db_drug);
}
"#
    }
}
