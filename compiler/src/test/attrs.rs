//! Test Attribute Definitions
//!
//! Defines attributes for marking tests, benchmarks, and test configuration.
//!
//! # Supported Attributes
//!
//! - `#[test]` - Mark a function as a test
//! - `#[bench]` - Mark a function as a benchmark
//! - `#[ignore]` / `#[ignore("reason")]` - Skip this test
//! - `#[should_panic]` / `#[should_panic(expected = "msg")]` - Test expects panic
//! - `#[timeout(ms)]` - Set test timeout in milliseconds
//! - `#[setup]` - Run before each test in module
//! - `#[teardown]` - Run after each test in module
//! - `#[fixture]` - Define a test fixture

use crate::common::{NodeId, Span};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// All attributes that can be attached to an item
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TestAttributes {
    /// Test attribute if present
    pub test: Option<TestAttr>,
    /// Benchmark attribute if present
    pub bench: Option<BenchAttr>,
    /// Ignore attribute if present
    pub ignore: Option<IgnoreAttr>,
    /// Should panic attribute if present
    pub should_panic: Option<ShouldPanicAttr>,
    /// Timeout in milliseconds
    pub timeout: Option<TimeoutAttr>,
    /// Setup function marker
    pub setup: bool,
    /// Teardown function marker
    pub teardown: bool,
    /// Fixture marker
    pub fixture: Option<FixtureAttr>,
    /// Custom tags for filtering
    pub tags: Vec<String>,
}

impl TestAttributes {
    /// Check if this has any test-related attributes
    pub fn is_test_item(&self) -> bool {
        self.test.is_some() || self.bench.is_some()
    }

    /// Check if this test should be skipped
    pub fn should_skip(&self) -> bool {
        self.ignore.is_some()
    }

    /// Get the timeout duration
    pub fn timeout_duration(&self) -> Option<Duration> {
        self.timeout.as_ref().map(|t| Duration::from_millis(t.ms))
    }
}

/// Test attribute: `#[test]`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestAttr {
    pub id: NodeId,
    pub span: Span,
}

impl TestAttr {
    pub fn new(id: NodeId, span: Span) -> Self {
        Self { id, span }
    }
}

/// Benchmark attribute: `#[bench]`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchAttr {
    pub id: NodeId,
    pub span: Span,
}

impl BenchAttr {
    pub fn new(id: NodeId, span: Span) -> Self {
        Self { id, span }
    }
}

/// Ignore attribute: `#[ignore]` or `#[ignore("reason")]`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IgnoreAttr {
    pub id: NodeId,
    pub reason: Option<String>,
    pub span: Span,
}

impl IgnoreAttr {
    pub fn new(id: NodeId, reason: Option<String>, span: Span) -> Self {
        Self { id, reason, span }
    }
}

/// Should panic attribute: `#[should_panic]` or `#[should_panic(expected = "msg")]`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShouldPanicAttr {
    pub id: NodeId,
    pub expected: Option<String>,
    pub span: Span,
}

impl ShouldPanicAttr {
    pub fn new(id: NodeId, expected: Option<String>, span: Span) -> Self {
        Self { id, expected, span }
    }
}

/// Timeout attribute: `#[timeout(1000)]` (milliseconds)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutAttr {
    pub id: NodeId,
    pub ms: u64,
    pub span: Span,
}

impl TimeoutAttr {
    pub fn new(id: NodeId, ms: u64, span: Span) -> Self {
        Self { id, ms, span }
    }

    pub fn duration(&self) -> Duration {
        Duration::from_millis(self.ms)
    }
}

/// Fixture attribute: `#[fixture]` or `#[fixture(scope = "module")]`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureAttr {
    pub id: NodeId,
    pub scope: FixtureScope,
    pub span: Span,
}

impl FixtureAttr {
    pub fn new(id: NodeId, scope: FixtureScope, span: Span) -> Self {
        Self { id, scope, span }
    }
}

/// Fixture scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FixtureScope {
    /// Fresh fixture for each test (default)
    #[default]
    Test,
    /// Shared across all tests in module
    Module,
    /// Shared across entire test session
    Session,
}

/// Raw attribute as parsed from source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawAttribute {
    pub id: NodeId,
    pub name: String,
    pub args: AttributeArgs,
    pub span: Span,
}

/// Attribute arguments
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AttributeArgs {
    /// No arguments: `#[test]`
    #[default]
    None,
    /// Single value: `#[ignore("reason")]` or `#[timeout(1000)]`
    Single(AttributeValue),
    /// Named arguments: `#[should_panic(expected = "msg")]`
    Named(Vec<(String, AttributeValue)>),
    /// List of values: `#[tags("unit", "fast")]`
    List(Vec<AttributeValue>),
}

/// Attribute argument value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Path(Vec<String>),
}

impl AttributeValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            AttributeValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            AttributeValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            AttributeValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Parse raw attributes into TestAttributes
pub fn parse_test_attributes(attrs: &[RawAttribute]) -> TestAttributes {
    let mut result = TestAttributes::default();

    for attr in attrs {
        match attr.name.as_str() {
            "test" => {
                result.test = Some(TestAttr::new(attr.id, attr.span));
            }
            "bench" => {
                result.bench = Some(BenchAttr::new(attr.id, attr.span));
            }
            "ignore" => {
                let reason = match &attr.args {
                    AttributeArgs::Single(AttributeValue::String(s)) => Some(s.clone()),
                    _ => None,
                };
                result.ignore = Some(IgnoreAttr::new(attr.id, reason, attr.span));
            }
            "should_panic" => {
                let expected = match &attr.args {
                    AttributeArgs::Named(pairs) => pairs
                        .iter()
                        .find(|(k, _)| k == "expected")
                        .and_then(|(_, v)| v.as_string().map(String::from)),
                    AttributeArgs::Single(AttributeValue::String(s)) => Some(s.clone()),
                    _ => None,
                };
                result.should_panic = Some(ShouldPanicAttr::new(attr.id, expected, attr.span));
            }
            "timeout" => {
                let ms = match &attr.args {
                    AttributeArgs::Single(AttributeValue::Int(ms)) => *ms as u64,
                    _ => 5000, // Default 5 seconds
                };
                result.timeout = Some(TimeoutAttr::new(attr.id, ms, attr.span));
            }
            "setup" => {
                result.setup = true;
            }
            "teardown" => {
                result.teardown = true;
            }
            "fixture" => {
                let scope = match &attr.args {
                    AttributeArgs::Named(pairs) => pairs
                        .iter()
                        .find(|(k, _)| k == "scope")
                        .and_then(|(_, v)| v.as_string())
                        .map(|s| match s {
                            "module" => FixtureScope::Module,
                            "session" => FixtureScope::Session,
                            _ => FixtureScope::Test,
                        })
                        .unwrap_or_default(),
                    _ => FixtureScope::default(),
                };
                result.fixture = Some(FixtureAttr::new(attr.id, scope, attr.span));
            }
            "tags" => {
                if let AttributeArgs::List(values) = &attr.args {
                    for v in values {
                        if let AttributeValue::String(tag) = v {
                            result.tags.push(tag.clone());
                        }
                    }
                }
            }
            _ => {
                // Unknown attribute, ignore for test purposes
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::NodeId;

    fn make_attr(name: &str, args: AttributeArgs) -> RawAttribute {
        RawAttribute {
            id: NodeId::dummy(),
            name: name.to_string(),
            args,
            span: Span::new(0, 0),
        }
    }

    #[test]
    fn test_parse_test_attr() {
        let attrs = vec![make_attr("test", AttributeArgs::None)];
        let result = parse_test_attributes(&attrs);
        assert!(result.test.is_some());
        assert!(result.is_test_item());
    }

    #[test]
    fn test_parse_bench_attr() {
        let attrs = vec![make_attr("bench", AttributeArgs::None)];
        let result = parse_test_attributes(&attrs);
        assert!(result.bench.is_some());
        assert!(result.is_test_item());
    }

    #[test]
    fn test_parse_ignore_with_reason() {
        let attrs = vec![make_attr(
            "ignore",
            AttributeArgs::Single(AttributeValue::String("not implemented".to_string())),
        )];
        let result = parse_test_attributes(&attrs);
        assert!(result.ignore.is_some());
        assert_eq!(
            result.ignore.as_ref().unwrap().reason,
            Some("not implemented".to_string())
        );
        assert!(result.should_skip());
    }

    #[test]
    fn test_parse_should_panic_expected() {
        let attrs = vec![make_attr(
            "should_panic",
            AttributeArgs::Named(vec![(
                "expected".to_string(),
                AttributeValue::String("overflow".to_string()),
            )]),
        )];
        let result = parse_test_attributes(&attrs);
        assert!(result.should_panic.is_some());
        assert_eq!(
            result.should_panic.unwrap().expected,
            Some("overflow".to_string())
        );
    }

    #[test]
    fn test_parse_timeout() {
        let attrs = vec![make_attr(
            "timeout",
            AttributeArgs::Single(AttributeValue::Int(1000)),
        )];
        let result = parse_test_attributes(&attrs);
        assert!(result.timeout.is_some());
        assert_eq!(result.timeout.as_ref().unwrap().ms, 1000);
        assert_eq!(result.timeout_duration(), Some(Duration::from_millis(1000)));
    }

    #[test]
    fn test_parse_tags() {
        let attrs = vec![make_attr(
            "tags",
            AttributeArgs::List(vec![
                AttributeValue::String("unit".to_string()),
                AttributeValue::String("fast".to_string()),
            ]),
        )];
        let result = parse_test_attributes(&attrs);
        assert_eq!(result.tags, vec!["unit", "fast"]);
    }

    #[test]
    fn test_parse_fixture_with_scope() {
        let attrs = vec![make_attr(
            "fixture",
            AttributeArgs::Named(vec![(
                "scope".to_string(),
                AttributeValue::String("module".to_string()),
            )]),
        )];
        let result = parse_test_attributes(&attrs);
        assert!(result.fixture.is_some());
        assert_eq!(result.fixture.unwrap().scope, FixtureScope::Module);
    }

    #[test]
    fn test_combined_attributes() {
        let attrs = vec![
            make_attr("test", AttributeArgs::None),
            make_attr("timeout", AttributeArgs::Single(AttributeValue::Int(5000))),
            make_attr(
                "tags",
                AttributeArgs::List(vec![AttributeValue::String("slow".to_string())]),
            ),
        ];
        let result = parse_test_attributes(&attrs);
        assert!(result.test.is_some());
        assert!(result.timeout.is_some());
        assert_eq!(result.tags, vec!["slow"]);
    }
}
