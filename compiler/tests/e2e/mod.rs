// tests/e2e/mod.rs — Main end-to-end test module
//
// This module organizes all end-to-end integration tests for the
// Sounio compiler. Tests are organized by category:
//
// - common: Test harness and utilities
// - pharmacology: Real pharmacology scenario tests
// - cross_ontology: Cross-ontology alignment tests
// - diagnostics: Error message quality tests
// - edge_cases: Boundary condition tests
// - performance: Scalability and timing tests
// - golden: Snapshot comparison tests
//
// NOTE: E2E tests are currently Linux-only due to path handling differences.
// TODO: Add cross-platform support for e2e tests.

pub mod common;

#[cfg(target_os = "linux")]
mod cross_ontology;
#[cfg(target_os = "linux")]
mod diagnostics;
#[cfg(target_os = "linux")]
mod edge_cases;
#[cfg(target_os = "linux")]
mod golden;
#[cfg(target_os = "linux")]
mod performance;
#[cfg(target_os = "linux")]
mod pharmacology;

// Re-export test harness for use in tests
