//! Testing Framework for the Sounio Language
//!
//! This module provides comprehensive testing infrastructure including:
//! - Test discovery and attribute parsing
//! - Test runner with parallel execution
//! - Benchmark runner with statistical analysis
//! - Code coverage tracking
//!
//! # Test Attributes
//!
//! ```d
//! #[test]
//! fn test_basic() {
//!     assert_eq(1 + 1, 2)
//! }
//!
//! #[test]
//! #[ignore("reason")]
//! fn test_slow() { ... }
//!
//! #[bench]
//! fn bench_algorithm(b: &Bencher) { ... }
//! ```

pub mod attrs;
pub mod bench;
pub mod coverage;
pub mod discovery;
pub mod runner;

pub use attrs::{BenchAttr, TestAttr, TestAttributes};
pub use bench::{BenchResult, Bencher, BenchmarkRunner};
pub use coverage::{CoverageData, CoverageReport, CoverageTracker};
pub use discovery::{TestCase, TestFilter, TestSuite, discover_tests};
pub use runner::{TestOutcome, TestReport, TestResult, TestRunner, TestRunnerConfig};
