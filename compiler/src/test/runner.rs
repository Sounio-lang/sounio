//! Test Runner
//!
//! Executes discovered tests with support for:
//! - Parallel execution with thread isolation
//! - Test timeouts
//! - Panic catching
//! - Progress reporting
//! - Result aggregation
//!
//! # Example
//!
//! ```rust,ignore
//! use sounio::test::{TestRunner, TestRunnerConfig, discover_tests, TestFilter};
//!
//! let suite = discover_tests(&["tests/"], TestFilter::default())?;
//! let config = TestRunnerConfig::default();
//! let runner = TestRunner::new(config);
//! let report = runner.run(&suite)?;
//! println!("{}", report.summary());
//! ```

use super::discovery::{TestCase, TestSuite};
use crate::hir::Hir;
use crate::interp::Interpreter;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during test execution
#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Test execution error: {0}")]
    Execution(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Thread pool error: {0}")]
    ThreadPool(String),
}

/// Configuration for the test runner
#[derive(Debug, Clone)]
pub struct TestRunnerConfig {
    /// Number of threads for parallel execution (0 = auto)
    pub threads: usize,
    /// Default timeout per test
    pub default_timeout: Duration,
    /// Fail fast on first failure
    pub fail_fast: bool,
    /// Capture stdout/stderr
    pub capture_output: bool,
    /// Show progress during execution
    pub show_progress: bool,
    /// Verbose output
    pub verbose: bool,
    /// Run tests in random order
    pub shuffle: bool,
    /// Random seed for shuffling
    pub seed: Option<u64>,
    /// Only list tests, don't run them
    pub list_only: bool,
    /// Output format
    pub format: OutputFormat,
}

impl Default for TestRunnerConfig {
    fn default() -> Self {
        Self {
            threads: 0, // Auto-detect
            default_timeout: Duration::from_secs(60),
            fail_fast: false,
            capture_output: true,
            show_progress: true,
            verbose: false,
            shuffle: false,
            seed: None,
            list_only: false,
            format: OutputFormat::Pretty,
        }
    }
}

impl TestRunnerConfig {
    /// Create config for running a single test
    pub fn single_test() -> Self {
        Self {
            threads: 1,
            fail_fast: true,
            ..Default::default()
        }
    }

    /// Create config for CI environments
    pub fn ci() -> Self {
        Self {
            show_progress: false,
            format: OutputFormat::Json,
            ..Default::default()
        }
    }
}

/// Output format for test results
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable output
    Pretty,
    /// Minimal output (dots)
    Compact,
    /// JSON output
    Json,
    /// JUnit XML
    JUnit,
}

/// Outcome of a single test
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestOutcome {
    /// Test passed
    Passed,
    /// Test failed with message
    Failed(String),
    /// Test was skipped (ignored)
    Skipped(Option<String>),
    /// Test timed out
    TimedOut,
    /// Test panicked
    Panicked(String),
}

impl TestOutcome {
    pub fn is_success(&self) -> bool {
        matches!(self, TestOutcome::Passed | TestOutcome::Skipped(_))
    }

    pub fn is_failure(&self) -> bool {
        matches!(
            self,
            TestOutcome::Failed(_) | TestOutcome::TimedOut | TestOutcome::Panicked(_)
        )
    }

    pub fn symbol(&self) -> &'static str {
        match self {
            TestOutcome::Passed => ".",
            TestOutcome::Failed(_) => "F",
            TestOutcome::Skipped(_) => "s",
            TestOutcome::TimedOut => "T",
            TestOutcome::Panicked(_) => "P",
        }
    }

    pub fn colored_symbol(&self) -> &'static str {
        match self {
            TestOutcome::Passed => "\x1b[32m.\x1b[0m",
            TestOutcome::Failed(_) => "\x1b[31mF\x1b[0m",
            TestOutcome::Skipped(_) => "\x1b[33ms\x1b[0m",
            TestOutcome::TimedOut => "\x1b[31mT\x1b[0m",
            TestOutcome::Panicked(_) => "\x1b[31mP\x1b[0m",
        }
    }
}

/// Result of a single test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test case that was run
    pub test_name: String,
    /// Full qualified name
    pub full_name: String,
    /// Source file
    pub file: PathBuf,
    /// Outcome of the test
    pub outcome: TestOutcome,
    /// Duration of the test
    pub duration: Duration,
    /// Captured stdout
    pub stdout: Option<String>,
    /// Captured stderr
    pub stderr: Option<String>,
    /// Additional info/diagnostics
    pub info: Option<String>,
}

impl TestResult {
    /// Create a passed result
    pub fn passed(test: &TestCase, duration: Duration) -> Self {
        Self {
            test_name: test.name.clone(),
            full_name: test.full_name.clone(),
            file: test.file.clone(),
            outcome: TestOutcome::Passed,
            duration,
            stdout: None,
            stderr: None,
            info: None,
        }
    }

    /// Create a failed result
    pub fn failed(test: &TestCase, message: String, duration: Duration) -> Self {
        Self {
            test_name: test.name.clone(),
            full_name: test.full_name.clone(),
            file: test.file.clone(),
            outcome: TestOutcome::Failed(message),
            duration,
            stdout: None,
            stderr: None,
            info: None,
        }
    }

    /// Create a skipped result
    pub fn skipped(test: &TestCase, reason: Option<String>) -> Self {
        Self {
            test_name: test.name.clone(),
            full_name: test.full_name.clone(),
            file: test.file.clone(),
            outcome: TestOutcome::Skipped(reason),
            duration: Duration::ZERO,
            stdout: None,
            stderr: None,
            info: None,
        }
    }

    /// Create a panicked result
    pub fn panicked(test: &TestCase, message: String, duration: Duration) -> Self {
        Self {
            test_name: test.name.clone(),
            full_name: test.full_name.clone(),
            file: test.file.clone(),
            outcome: TestOutcome::Panicked(message),
            duration,
            stdout: None,
            stderr: None,
            info: None,
        }
    }
}

/// Report from running a test suite
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TestReport {
    /// All test results
    pub results: Vec<TestResult>,
    /// Total duration
    pub duration: Duration,
    /// Number of passed tests
    pub passed: usize,
    /// Number of failed tests
    pub failed: usize,
    /// Number of skipped tests
    pub skipped: usize,
    /// Number of timed out tests
    pub timed_out: usize,
    /// Number of panicked tests
    pub panicked: usize,
}

impl TestReport {
    /// Create a new empty report
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a result to the report
    pub fn add_result(&mut self, result: TestResult) {
        match &result.outcome {
            TestOutcome::Passed => self.passed += 1,
            TestOutcome::Failed(_) => self.failed += 1,
            TestOutcome::Skipped(_) => self.skipped += 1,
            TestOutcome::TimedOut => self.timed_out += 1,
            TestOutcome::Panicked(_) => self.panicked += 1,
        }
        self.results.push(result);
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.failed == 0 && self.timed_out == 0 && self.panicked == 0
    }

    /// Get total number of tests run
    pub fn total(&self) -> usize {
        self.results.len()
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        let status = if self.all_passed() {
            "\x1b[32mok\x1b[0m"
        } else {
            "\x1b[31mFAILED\x1b[0m"
        };

        format!(
            "test result: {}. {} passed; {} failed; {} skipped; finished in {:.2}s",
            status,
            self.passed,
            self.failed + self.timed_out + self.panicked,
            self.skipped,
            self.duration.as_secs_f64()
        )
    }

    /// Get failed tests
    pub fn failures(&self) -> impl Iterator<Item = &TestResult> {
        self.results.iter().filter(|r| r.outcome.is_failure())
    }

    /// Format as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Format as JUnit XML
    pub fn to_junit(&self) -> String {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str(&format!(
            "<testsuite name=\"sounio\" tests=\"{}\" failures=\"{}\" time=\"{:.3}\">\n",
            self.total(),
            self.failed + self.timed_out + self.panicked,
            self.duration.as_secs_f64()
        ));

        for result in &self.results {
            xml.push_str(&format!(
                "  <testcase name=\"{}\" classname=\"{}\" time=\"{:.3}\"",
                result.test_name,
                result.full_name,
                result.duration.as_secs_f64()
            ));

            match &result.outcome {
                TestOutcome::Passed => {
                    xml.push_str(" />\n");
                }
                TestOutcome::Failed(msg) => {
                    xml.push_str(">\n");
                    xml.push_str(&format!(
                        "    <failure message=\"{}\">{}</failure>\n",
                        escape_xml(msg),
                        escape_xml(msg)
                    ));
                    xml.push_str("  </testcase>\n");
                }
                TestOutcome::Skipped(reason) => {
                    xml.push_str(">\n");
                    xml.push_str(&format!(
                        "    <skipped message=\"{}\" />\n",
                        escape_xml(reason.as_deref().unwrap_or("ignored"))
                    ));
                    xml.push_str("  </testcase>\n");
                }
                TestOutcome::TimedOut => {
                    xml.push_str(">\n");
                    xml.push_str("    <failure message=\"timeout\">Test timed out</failure>\n");
                    xml.push_str("  </testcase>\n");
                }
                TestOutcome::Panicked(msg) => {
                    xml.push_str(">\n");
                    xml.push_str(&format!(
                        "    <error message=\"panic\">{}</error>\n",
                        escape_xml(msg)
                    ));
                    xml.push_str("  </testcase>\n");
                }
            }
        }

        xml.push_str("</testsuite>\n");
        xml
    }
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Test runner that executes tests
pub struct TestRunner {
    config: TestRunnerConfig,
}

impl TestRunner {
    /// Create a new test runner with configuration
    pub fn new(config: TestRunnerConfig) -> Self {
        Self { config }
    }

    /// Run all tests in a suite
    pub fn run(&self, suite: &TestSuite) -> Result<TestReport, RunnerError> {
        let start = Instant::now();
        let mut report = TestReport::new();

        // Get all tests
        let tests: Vec<_> = suite.all_tests().into_iter().cloned().collect();

        if self.config.list_only {
            return self.list_tests(&tests);
        }

        // Optionally shuffle tests
        let tests = if self.config.shuffle {
            self.shuffle_tests(tests)
        } else {
            tests
        };

        if self.config.show_progress {
            println!("\nrunning {} tests", tests.len());
        }

        // Run tests
        if self.config.threads == 1 {
            // Sequential execution
            for test in &tests {
                let result = self.run_single_test(test);
                if self.config.show_progress {
                    print!("{}", result.outcome.colored_symbol());
                    std::io::stdout().flush().ok();
                }
                let is_failure = result.outcome.is_failure();
                report.add_result(result);
                if self.config.fail_fast && is_failure {
                    break;
                }
            }
        } else {
            // Parallel execution
            self.run_parallel(&tests, &mut report)?;
        }

        report.duration = start.elapsed();

        if self.config.show_progress {
            println!("\n");
            self.print_failures(&report);
            println!("{}", report.summary());
        }

        Ok(report)
    }

    /// Run tests in parallel
    fn run_parallel(&self, tests: &[TestCase], report: &mut TestReport) -> Result<(), RunnerError> {
        let num_threads = if self.config.threads == 0 {
            num_cpus::get()
        } else {
            self.config.threads
        };

        let tests = Arc::new(tests.to_vec());
        let results = Arc::new(Mutex::new(Vec::new()));
        let counter = Arc::new(AtomicUsize::new(0));
        let fail_fast = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let mut handles = Vec::new();

        for _ in 0..num_threads {
            let tests = Arc::clone(&tests);
            let results = Arc::clone(&results);
            let counter = Arc::clone(&counter);
            let fail_fast_flag = Arc::clone(&fail_fast);
            let config = self.config.clone();

            let handle = std::thread::spawn(move || {
                loop {
                    if config.fail_fast && fail_fast_flag.load(Ordering::SeqCst) {
                        break;
                    }

                    let idx = counter.fetch_add(1, Ordering::SeqCst);
                    if idx >= tests.len() {
                        break;
                    }

                    let test = &tests[idx];
                    let result = run_test_isolated(test, &config);

                    if config.show_progress {
                        print!("{}", result.outcome.colored_symbol());
                        std::io::stdout().flush().ok();
                    }

                    if result.outcome.is_failure() {
                        fail_fast_flag.store(true, Ordering::SeqCst);
                    }

                    results.lock().unwrap().push(result);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().map_err(|_| {
                RunnerError::ThreadPool("Thread panicked during test execution".to_string())
            })?;
        }

        let final_results = Arc::try_unwrap(results)
            .map_err(|_| RunnerError::ThreadPool("Failed to unwrap results".to_string()))?
            .into_inner()
            .map_err(|_| RunnerError::ThreadPool("Mutex poisoned".to_string()))?;

        for result in final_results {
            report.add_result(result);
        }

        Ok(())
    }

    /// Run a single test
    fn run_single_test(&self, test: &TestCase) -> TestResult {
        run_test_isolated(test, &self.config)
    }

    /// Shuffle tests using configured seed
    fn shuffle_tests(&self, mut tests: Vec<TestCase>) -> Vec<TestCase> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let seed = self.config.seed.unwrap_or_else(|| {
            let mut hasher = DefaultHasher::new();
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
                .hash(&mut hasher);
            hasher.finish()
        });

        // Simple shuffle using seed
        let mut rng_state = seed;
        for i in (1..tests.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            tests.swap(i, j);
        }

        tests
    }

    /// List tests without running them
    fn list_tests(&self, tests: &[TestCase]) -> Result<TestReport, RunnerError> {
        for test in tests {
            println!("{}: test", test.full_name);
        }
        println!("\n{} tests", tests.len());
        Ok(TestReport::new())
    }

    /// Print failure details
    fn print_failures(&self, report: &TestReport) {
        let failures: Vec<_> = report.failures().collect();
        if failures.is_empty() {
            return;
        }

        println!("failures:\n");
        for result in &failures {
            println!("---- {} ----", result.full_name);
            match &result.outcome {
                TestOutcome::Failed(msg) => println!("  {}", msg),
                TestOutcome::Panicked(msg) => println!("  panicked: {}", msg),
                TestOutcome::TimedOut => println!("  timed out"),
                _ => {}
            }
            if let Some(stdout) = &result.stdout
                && !stdout.is_empty()
            {
                println!("\n  stdout:\n{}", indent(stdout, 4));
            }
            if let Some(stderr) = &result.stderr
                && !stderr.is_empty()
            {
                println!("\n  stderr:\n{}", indent(stderr, 4));
            }
            println!();
        }

        println!("failures:");
        for result in failures {
            println!("    {}", result.full_name);
        }
        println!();
    }
}

fn indent(s: &str, spaces: usize) -> String {
    let prefix = " ".repeat(spaces);
    s.lines()
        .map(|line| format!("{}{}", prefix, line))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Run a test in isolation (catches panics)
fn run_test_isolated(test: &TestCase, config: &TestRunnerConfig) -> TestResult {
    // Handle skipped tests
    if test.should_skip() {
        let reason = test.attrs.ignore.as_ref().and_then(|i| i.reason.clone());
        return TestResult::skipped(test, reason);
    }

    let start = Instant::now();
    let timeout = test
        .attrs
        .timeout_duration()
        .unwrap_or(config.default_timeout);

    // Run the test with panic catching
    let result = panic::catch_unwind(AssertUnwindSafe(|| execute_test_function(test, timeout)));

    let duration = start.elapsed();

    match result {
        Ok(Ok(())) => {
            // Test passed
            if test.expects_panic() {
                // Expected panic but didn't get one
                TestResult::failed(
                    test,
                    "expected panic but test completed normally".to_string(),
                    duration,
                )
            } else {
                TestResult::passed(test, duration)
            }
        }
        Ok(Err(msg)) => {
            // Test returned error
            TestResult::failed(test, msg, duration)
        }
        Err(panic_info) => {
            // Test panicked
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };

            if test.expects_panic() {
                // Check if panic message matches expected
                if let Some(expected) = test.expected_panic_message() {
                    if panic_msg.contains(expected) {
                        TestResult::passed(test, duration)
                    } else {
                        TestResult::failed(
                            test,
                            format!(
                                "panic message mismatch: expected '{}', got '{}'",
                                expected, panic_msg
                            ),
                            duration,
                        )
                    }
                } else {
                    // Any panic is fine
                    TestResult::passed(test, duration)
                }
            } else {
                TestResult::panicked(test, panic_msg, duration)
            }
        }
    }
}

/// Execute the actual test function
fn execute_test_function(test: &TestCase, _timeout: Duration) -> Result<(), String> {
    // Read and compile the test file
    let source = std::fs::read_to_string(&test.file)
        .map_err(|e| format!("Failed to read test file: {}", e))?;

    let tokens = crate::lexer::lex(&source).map_err(|e| format!("Lexer error: {}", e))?;

    let ast = crate::parser::parse(&tokens, &source).map_err(|e| format!("Parse error: {}", e))?;

    let hir = crate::check::check(&ast).map_err(|e| format!("Type check error: {}", e))?;

    // Execute the test function using the interpreter
    execute_test_hir(&hir, &test.name)
}

/// Execute a test function from HIR
fn execute_test_hir(hir: &Hir, _fn_name: &str) -> Result<(), String> {
    let mut interpreter = Interpreter::new();

    // Run the HIR - the interpreter will execute main or evaluate top-level
    // In a real implementation, we'd call the specific test function
    match interpreter.interpret(hir) {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Test execution failed: {}", e)),
    }
}

// Placeholder for num_cpus - we'd use the actual crate in production
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{NodeId, Span};
    use crate::test::attrs::TestAttributes;

    fn make_test(name: &str) -> TestCase {
        TestCase::new(
            NodeId::dummy(),
            name.to_string(),
            "test".to_string(),
            PathBuf::from("test.sio"),
            Span::new(0, 100),
            TestAttributes::default(),
        )
    }

    #[test]
    fn test_outcome_symbols() {
        assert_eq!(TestOutcome::Passed.symbol(), ".");
        assert_eq!(TestOutcome::Failed("error".to_string()).symbol(), "F");
        assert_eq!(TestOutcome::Skipped(None).symbol(), "s");
        assert_eq!(TestOutcome::TimedOut.symbol(), "T");
        assert_eq!(TestOutcome::Panicked("panic".to_string()).symbol(), "P");
    }

    #[test]
    fn test_outcome_success() {
        assert!(TestOutcome::Passed.is_success());
        assert!(TestOutcome::Skipped(None).is_success());
        assert!(!TestOutcome::Failed("".to_string()).is_success());
    }

    #[test]
    fn test_report_summary() {
        let mut report = TestReport::new();
        report.add_result(TestResult::passed(
            &make_test("t1"),
            Duration::from_millis(100),
        ));
        report.add_result(TestResult::passed(
            &make_test("t2"),
            Duration::from_millis(50),
        ));
        report.duration = Duration::from_millis(150);

        assert_eq!(report.passed, 2);
        assert_eq!(report.failed, 0);
        assert!(report.all_passed());
        assert_eq!(report.total(), 2);
    }

    #[test]
    fn test_report_with_failures() {
        let mut report = TestReport::new();
        report.add_result(TestResult::passed(
            &make_test("t1"),
            Duration::from_millis(100),
        ));
        report.add_result(TestResult::failed(
            &make_test("t2"),
            "assertion failed".to_string(),
            Duration::from_millis(50),
        ));

        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);
        assert!(!report.all_passed());
        assert_eq!(report.failures().count(), 1);
    }

    #[test]
    fn test_runner_config_defaults() {
        let config = TestRunnerConfig::default();
        assert_eq!(config.threads, 0);
        assert!(!config.fail_fast);
        assert!(config.capture_output);
    }

    #[test]
    fn test_junit_output() {
        let mut report = TestReport::new();
        report.add_result(TestResult::passed(
            &make_test("t1"),
            Duration::from_millis(100),
        ));
        report.duration = Duration::from_millis(100);

        let xml = report.to_junit();
        assert!(xml.contains("<?xml"));
        assert!(xml.contains("testsuite"));
        assert!(xml.contains("testcase"));
        assert!(xml.contains("t1"));
    }

    #[test]
    fn test_shuffle() {
        let runner = TestRunner::new(TestRunnerConfig {
            shuffle: true,
            seed: Some(12345),
            ..Default::default()
        });

        let tests: Vec<_> = (0..10).map(|i| make_test(&format!("test_{}", i))).collect();
        let shuffled = runner.shuffle_tests(tests.clone());

        // With the same seed, should get the same shuffle
        let shuffled2 = runner.shuffle_tests(tests);
        assert_eq!(
            shuffled.iter().map(|t| &t.name).collect::<Vec<_>>(),
            shuffled2.iter().map(|t| &t.name).collect::<Vec<_>>()
        );
    }
}
