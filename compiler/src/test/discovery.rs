//! Test Discovery Module
//!
//! Discovers and collects tests from source files based on attributes.
//!
//! # Example
//!
//! ```rust,ignore
//! use sounio::test::discovery::{discover_tests, TestFilter};
//!
//! let suite = discover_tests(&["src/"], TestFilter::default())?;
//! println!("Found {} tests", suite.tests.len());
//! ```

use super::attrs::{
    AttributeArgs, AttributeValue, RawAttribute, TestAttributes, parse_test_attributes,
};
use crate::ast::{Ast, FnDef, Item};
use crate::common::{NodeId, Span};
use crate::lexer;
use crate::parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during test discovery
#[derive(Debug, Error)]
pub enum DiscoveryError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error in {file}: {message}")]
    Parse { file: PathBuf, message: String },

    #[error("Invalid test filter pattern: {0}")]
    InvalidFilter(String),
}

/// A discovered test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    /// Unique identifier for the test
    pub id: NodeId,
    /// Test name (function name)
    pub name: String,
    /// Full module path (e.g., "tests::unit::math")
    pub module_path: String,
    /// Full qualified name (module_path::name)
    pub full_name: String,
    /// Source file containing the test
    pub file: PathBuf,
    /// Span of the test function
    pub span: Span,
    /// Test attributes
    pub attrs: TestAttributes,
    /// Whether this is a benchmark
    pub is_bench: bool,
    /// Source code of the test (for display)
    pub source: Option<String>,
}

impl TestCase {
    /// Create a new test case
    pub fn new(
        id: NodeId,
        name: String,
        module_path: String,
        file: PathBuf,
        span: Span,
        attrs: TestAttributes,
    ) -> Self {
        let is_bench = attrs.bench.is_some();
        let full_name = if module_path.is_empty() {
            name.clone()
        } else {
            format!("{}::{}", module_path, name)
        };
        Self {
            id,
            name,
            module_path,
            full_name,
            file,
            span,
            attrs,
            is_bench,
            source: None,
        }
    }

    /// Check if this test should be skipped
    pub fn should_skip(&self) -> bool {
        self.attrs.should_skip()
    }

    /// Check if this test expects a panic
    pub fn expects_panic(&self) -> bool {
        self.attrs.should_panic.is_some()
    }

    /// Get the expected panic message (if any)
    pub fn expected_panic_message(&self) -> Option<&str> {
        self.attrs
            .should_panic
            .as_ref()
            .and_then(|sp| sp.expected.as_deref())
    }
}

/// A collection of test cases from a module or file
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TestSuite {
    /// Name of the test suite
    pub name: String,
    /// All discovered test cases
    pub tests: Vec<TestCase>,
    /// Benchmarks (separate from regular tests)
    pub benchmarks: Vec<TestCase>,
    /// Setup functions
    pub setup_fns: Vec<String>,
    /// Teardown functions
    pub teardown_fns: Vec<String>,
    /// Fixtures
    pub fixtures: HashMap<String, TestCase>,
    /// Source files included
    pub files: Vec<PathBuf>,
    /// Child suites (nested modules)
    pub children: HashMap<String, TestSuite>,
}

impl TestSuite {
    /// Create a new empty test suite
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Add a test case to this suite
    pub fn add_test(&mut self, test: TestCase) {
        if test.is_bench {
            self.benchmarks.push(test);
        } else {
            self.tests.push(test);
        }
    }

    /// Get all tests (including from child suites)
    pub fn all_tests(&self) -> Vec<&TestCase> {
        let mut result: Vec<&TestCase> = self.tests.iter().collect();
        for child in self.children.values() {
            result.extend(child.all_tests());
        }
        result
    }

    /// Get all benchmarks (including from child suites)
    pub fn all_benchmarks(&self) -> Vec<&TestCase> {
        let mut result: Vec<&TestCase> = self.benchmarks.iter().collect();
        for child in self.children.values() {
            result.extend(child.all_benchmarks());
        }
        result
    }

    /// Get total test count
    pub fn test_count(&self) -> usize {
        self.tests.len()
            + self
                .children
                .values()
                .map(|c| c.test_count())
                .sum::<usize>()
    }

    /// Get total benchmark count
    pub fn bench_count(&self) -> usize {
        self.benchmarks.len()
            + self
                .children
                .values()
                .map(|c| c.bench_count())
                .sum::<usize>()
    }

    /// Filter tests based on a filter
    pub fn filter(&self, filter: &TestFilter) -> TestSuite {
        let mut filtered = TestSuite::new(&self.name);
        filtered.files = self.files.clone();
        filtered.setup_fns = self.setup_fns.clone();
        filtered.teardown_fns = self.teardown_fns.clone();

        for test in &self.tests {
            if filter.matches(test) {
                filtered.tests.push(test.clone());
            }
        }

        for bench in &self.benchmarks {
            if filter.matches(bench) {
                filtered.benchmarks.push(bench.clone());
            }
        }

        for (name, child) in &self.children {
            let filtered_child = child.filter(filter);
            if filtered_child.test_count() > 0 || filtered_child.bench_count() > 0 {
                filtered.children.insert(name.clone(), filtered_child);
            }
        }

        filtered
    }
}

/// Filter for selecting which tests to run
#[derive(Debug, Clone, Default)]
pub struct TestFilter {
    /// Only run tests matching this pattern
    pub pattern: Option<String>,
    /// Only run tests with these tags
    pub tags: Vec<String>,
    /// Exclude tests with these tags
    pub exclude_tags: Vec<String>,
    /// Include ignored tests
    pub include_ignored: bool,
    /// Only run ignored tests
    pub only_ignored: bool,
    /// Exact match (not substring)
    pub exact: bool,
}

impl TestFilter {
    /// Create a new filter with a pattern
    pub fn with_pattern(pattern: impl Into<String>) -> Self {
        Self {
            pattern: Some(pattern.into()),
            ..Default::default()
        }
    }

    /// Add a required tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Exclude a tag
    pub fn exclude_tag(mut self, tag: impl Into<String>) -> Self {
        self.exclude_tags.push(tag.into());
        self
    }

    /// Include ignored tests
    pub fn include_ignored(mut self) -> Self {
        self.include_ignored = true;
        self
    }

    /// Check if a test matches this filter
    pub fn matches(&self, test: &TestCase) -> bool {
        // Check ignored status
        if test.should_skip() {
            if self.only_ignored {
                return true;
            }
            if !self.include_ignored {
                return false;
            }
        } else if self.only_ignored {
            return false;
        }

        // Check pattern
        if let Some(pattern) = &self.pattern {
            let matches = if self.exact {
                test.full_name == *pattern || test.name == *pattern
            } else {
                test.full_name.contains(pattern) || test.name.contains(pattern)
            };
            if !matches {
                return false;
            }
        }

        // Check required tags
        if !self.tags.is_empty() {
            let has_all_tags = self.tags.iter().all(|t| test.attrs.tags.contains(t));
            if !has_all_tags {
                return false;
            }
        }

        // Check excluded tags
        if self
            .exclude_tags
            .iter()
            .any(|t| test.attrs.tags.contains(t))
        {
            return false;
        }

        true
    }
}

/// Discover tests from given paths
pub fn discover_tests(
    paths: &[impl AsRef<Path>],
    filter: TestFilter,
) -> Result<TestSuite, DiscoveryError> {
    let mut suite = TestSuite::new("root");

    for path in paths {
        let path = path.as_ref();
        if path.is_dir() {
            discover_in_directory(path, &mut suite)?;
        } else if path.extension().is_some_and(|e| e == "d") {
            discover_in_file(path, &mut suite)?;
        }
    }

    Ok(suite.filter(&filter))
}

/// Discover tests in a directory recursively
fn discover_in_directory(dir: &Path, suite: &mut TestSuite) -> Result<(), DiscoveryError> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Skip hidden directories and target
            if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && (name.starts_with('.') || name == "target")
            {
                continue;
            }
            discover_in_directory(&path, suite)?;
        } else if path.extension().is_some_and(|e| e == "d") {
            discover_in_file(&path, suite)?;
        }
    }
    Ok(())
}

/// Discover tests in a single file
fn discover_in_file(file: &Path, suite: &mut TestSuite) -> Result<(), DiscoveryError> {
    let source = std::fs::read_to_string(file)?;

    // Parse the file
    let tokens = lexer::lex(&source).map_err(|e| DiscoveryError::Parse {
        file: file.to_path_buf(),
        message: e.to_string(),
    })?;

    let ast = parser::parse(&tokens, &source).map_err(|e| DiscoveryError::Parse {
        file: file.to_path_buf(),
        message: e.to_string(),
    })?;

    // Extract module path from AST or file path
    let module_path = ast
        .module_name
        .as_ref()
        .map(|p| p.to_string())
        .unwrap_or_else(|| {
            file.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

    // Discover tests from items
    discover_from_ast(&ast, &module_path, file, &source, suite);

    suite.files.push(file.to_path_buf());
    Ok(())
}

/// Extract test cases from parsed AST
fn discover_from_ast(
    ast: &Ast,
    module_path: &str,
    file: &Path,
    source: &str,
    suite: &mut TestSuite,
) {
    for item in &ast.items {
        if let Item::Function(fn_def) = item {
            // In the actual implementation, we'd parse attributes from the AST
            // For now, we'll use a simplified approach checking for test naming conventions
            let attrs = extract_test_attrs(fn_def, source);

            if attrs.is_test_item() {
                let test = TestCase::new(
                    fn_def.id,
                    fn_def.name.clone(),
                    module_path.to_string(),
                    file.to_path_buf(),
                    fn_def.span,
                    attrs.clone(),
                );
                suite.add_test(test);
            }

            if attrs.setup {
                suite.setup_fns.push(fn_def.name.clone());
            }

            if attrs.teardown {
                suite.teardown_fns.push(fn_def.name.clone());
            }

            if attrs.fixture.is_some() {
                let fixture = TestCase::new(
                    fn_def.id,
                    fn_def.name.clone(),
                    module_path.to_string(),
                    file.to_path_buf(),
                    fn_def.span,
                    attrs,
                );
                suite.fixtures.insert(fn_def.name.clone(), fixture);
            }
        }
    }
}

/// Extract test attributes from a function definition
fn extract_test_attrs(fn_def: &FnDef, _source: &str) -> TestAttributes {
    // Convert AST attributes to RawAttributes
    let raw_attrs: Vec<RawAttribute> = fn_def
        .attributes
        .iter()
        .map(|attr| {
            let args = match &attr.args {
                crate::ast::AttributeArgs::Empty => AttributeArgs::None,
                crate::ast::AttributeArgs::Value(v) => AttributeArgs::Single(convert_attr_value(v)),
                crate::ast::AttributeArgs::Named(pairs) => AttributeArgs::Named(
                    pairs
                        .iter()
                        .map(|(k, v)| (k.clone(), convert_attr_value(v)))
                        .collect(),
                ),
                crate::ast::AttributeArgs::List(values) => {
                    AttributeArgs::List(values.iter().map(convert_attr_value).collect())
                }
            };
            RawAttribute {
                id: attr.id,
                name: attr.name.clone(),
                args,
                span: attr.span,
            }
        })
        .collect();

    parse_test_attributes(&raw_attrs)
}

/// Convert AST AttributeValue to test AttributeValue
fn convert_attr_value(v: &crate::ast::AttributeValue) -> AttributeValue {
    match v {
        crate::ast::AttributeValue::String(s) => AttributeValue::String(s.clone()),
        crate::ast::AttributeValue::Int(i) => AttributeValue::Int(*i),
        crate::ast::AttributeValue::Float(f) => AttributeValue::Float(*f),
        crate::ast::AttributeValue::Bool(b) => AttributeValue::Bool(*b),
        crate::ast::AttributeValue::Path(p) => AttributeValue::Path(p.segments.clone()),
        crate::ast::AttributeValue::Nested(name, _) => {
            // For nested attributes, just use the name as a path
            AttributeValue::Path(vec![name.clone()])
        }
    }
}

/// Parse attributes from raw attribute list
pub fn parse_attrs_from_raw(raw_attrs: &[RawAttribute]) -> TestAttributes {
    parse_test_attributes(raw_attrs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_matches_pattern() {
        let test = TestCase::new(
            NodeId::dummy(),
            "test_addition".to_string(),
            "math::ops".to_string(),
            PathBuf::from("src/math/ops.sio"),
            Span::new(0, 100),
            TestAttributes::default(),
        );

        let filter = TestFilter::with_pattern("addition");
        assert!(filter.matches(&test));

        let filter = TestFilter::with_pattern("subtraction");
        assert!(!filter.matches(&test));
    }

    #[test]
    fn test_filter_exact_match() {
        let test = TestCase::new(
            NodeId::dummy(),
            "test_add".to_string(),
            "math".to_string(),
            PathBuf::from("src/math.sio"),
            Span::new(0, 100),
            TestAttributes::default(),
        );

        let mut filter = TestFilter::with_pattern("test_add");
        filter.exact = true;
        assert!(filter.matches(&test));

        let mut filter = TestFilter::with_pattern("test_ad");
        filter.exact = true;
        assert!(!filter.matches(&test));
    }

    #[test]
    fn test_filter_tags() {
        let mut attrs = TestAttributes::default();
        attrs.tags = vec!["unit".to_string(), "fast".to_string()];

        let test = TestCase::new(
            NodeId::dummy(),
            "test_something".to_string(),
            "module".to_string(),
            PathBuf::from("src/module.sio"),
            Span::new(0, 100),
            attrs,
        );

        let filter = TestFilter::default().with_tag("unit");
        assert!(filter.matches(&test));

        let filter = TestFilter::default().with_tag("slow");
        assert!(!filter.matches(&test));

        let filter = TestFilter::default().exclude_tag("fast");
        assert!(!filter.matches(&test));
    }

    #[test]
    fn test_filter_ignored() {
        use super::super::attrs::IgnoreAttr;

        let mut attrs = TestAttributes::default();
        attrs.ignore = Some(IgnoreAttr::new(
            NodeId::dummy(),
            Some("not ready".to_string()),
            Span::new(0, 0),
        ));

        let test = TestCase::new(
            NodeId::dummy(),
            "test_ignored".to_string(),
            "module".to_string(),
            PathBuf::from("src/module.sio"),
            Span::new(0, 100),
            attrs,
        );

        // Default filter excludes ignored tests
        let filter = TestFilter::default();
        assert!(!filter.matches(&test));

        // Include ignored
        let filter = TestFilter::default().include_ignored();
        assert!(filter.matches(&test));
    }

    #[test]
    fn test_suite_hierarchy() {
        let mut root = TestSuite::new("root");

        let test1 = TestCase::new(
            NodeId::dummy(),
            "test_one".to_string(),
            "a".to_string(),
            PathBuf::from("a.sio"),
            Span::new(0, 10),
            TestAttributes::default(),
        );
        root.add_test(test1);

        let mut child = TestSuite::new("child");
        let test2 = TestCase::new(
            NodeId::dummy(),
            "test_two".to_string(),
            "a::b".to_string(),
            PathBuf::from("a/b.sio"),
            Span::new(0, 10),
            TestAttributes::default(),
        );
        child.add_test(test2);

        root.children.insert("b".to_string(), child);

        assert_eq!(root.test_count(), 2);
        assert_eq!(root.all_tests().len(), 2);
    }

    #[test]
    fn test_case_full_name() {
        let test = TestCase::new(
            NodeId::dummy(),
            "test_func".to_string(),
            "my::module".to_string(),
            PathBuf::from("src/module.sio"),
            Span::new(0, 100),
            TestAttributes::default(),
        );

        assert_eq!(test.full_name, "my::module::test_func");
    }

    #[test]
    fn test_case_empty_module_path() {
        let test = TestCase::new(
            NodeId::dummy(),
            "test_func".to_string(),
            String::new(),
            PathBuf::from("src/main.sio"),
            Span::new(0, 100),
            TestAttributes::default(),
        );

        assert_eq!(test.full_name, "test_func");
    }
}
