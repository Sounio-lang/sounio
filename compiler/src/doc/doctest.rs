//! Doctest Runner
//!
//! Extracts and executes code examples from documentation comments.
//! Similar to Rust's `cargo test --doc` functionality.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::doc::model::{ConstantDoc, CrateDoc, FunctionDoc, ModuleDoc, TraitDoc, TypeDoc};
use crate::doc::parser::ExampleDoc;

/// A single doctest extracted from documentation
#[derive(Debug, Clone)]
pub struct Doctest {
    /// Name/path of the documented item
    pub item_path: String,
    /// The code to test
    pub code: String,
    /// Whether this test should pass
    pub should_pass: bool,
    /// Whether this test should panic
    pub should_panic: bool,
    /// Whether to skip this test
    pub ignore: bool,
    /// Whether to compile but not run
    pub no_run: bool,
    /// Line number in source where example appears
    pub line_number: Option<usize>,
    /// File path where the doctest originated
    pub source_file: Option<String>,
}

/// Result of running a single doctest
#[derive(Debug, Clone)]
pub struct DoctestResult {
    /// The doctest that was run
    pub doctest: Doctest,
    /// Whether the test passed
    pub passed: bool,
    /// Duration of the test
    pub duration: Duration,
    /// Compiler output (if compilation failed)
    pub compile_output: Option<String>,
    /// Runtime output (stdout + stderr)
    pub runtime_output: Option<String>,
    /// Error message if test failed
    pub error: Option<String>,
}

/// Summary of all doctest results
#[derive(Debug, Clone, Default)]
pub struct DoctestSummary {
    /// Total number of doctests
    pub total: usize,
    /// Number of tests that passed
    pub passed: usize,
    /// Number of tests that failed
    pub failed: usize,
    /// Number of tests that were ignored
    pub ignored: usize,
    /// Total duration
    pub duration: Duration,
    /// Individual results
    pub results: Vec<DoctestResult>,
}

impl DoctestSummary {
    /// Returns true if all tests passed (excluding ignored)
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }

    /// Returns the success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        let run = self.total - self.ignored;
        if run == 0 {
            100.0
        } else {
            (self.passed as f64 / run as f64) * 100.0
        }
    }
}

/// Documentation coverage statistics
#[derive(Debug, Clone, Default)]
pub struct DocCoverage {
    /// Total number of public items
    pub total_items: usize,
    /// Number of items with documentation
    pub documented_items: usize,
    /// Number of items with examples
    pub items_with_examples: usize,
    /// Breakdown by item kind
    pub by_kind: BTreeMap<String, (usize, usize)>, // (total, documented)
    /// Undocumented items
    pub undocumented: Vec<String>,
    /// Items without examples
    pub no_examples: Vec<String>,
}

impl DocCoverage {
    /// Returns the documentation coverage percentage
    pub fn coverage_percentage(&self) -> f64 {
        if self.total_items == 0 {
            100.0
        } else {
            (self.documented_items as f64 / self.total_items as f64) * 100.0
        }
    }

    /// Returns the example coverage percentage
    pub fn example_coverage(&self) -> f64 {
        if self.total_items == 0 {
            100.0
        } else {
            (self.items_with_examples as f64 / self.total_items as f64) * 100.0
        }
    }
}

/// Doctest runner configuration
#[derive(Debug, Clone)]
pub struct DoctestConfig {
    /// Compiler path (dc or cargo run --)
    pub compiler_path: PathBuf,
    /// Temporary directory for test files
    pub temp_dir: PathBuf,
    /// Whether to run tests in parallel
    pub parallel: bool,
    /// Maximum test duration
    pub timeout: Duration,
    /// Whether to show output from passing tests
    pub show_output: bool,
    /// Whether to continue on failure
    pub continue_on_failure: bool,
    /// Test filter pattern
    pub filter: Option<String>,
}

impl Default for DoctestConfig {
    fn default() -> Self {
        Self {
            compiler_path: PathBuf::from("dc"),
            temp_dir: std::env::temp_dir().join("sounio-doctests"),
            parallel: false,
            timeout: Duration::from_secs(60),
            show_output: false,
            continue_on_failure: true,
            filter: None,
        }
    }
}

/// Doctest runner
pub struct DoctestRunner {
    config: DoctestConfig,
}

impl DoctestRunner {
    /// Create a new doctest runner
    pub fn new(config: DoctestConfig) -> Self {
        Self { config }
    }

    /// Extract all doctests from crate documentation
    pub fn extract_doctests(&self, crate_doc: &CrateDoc) -> Vec<Doctest> {
        let mut doctests = Vec::new();
        self.extract_from_module(&crate_doc.root_module, "", &mut doctests);
        doctests
    }

    fn extract_from_module(&self, module: &ModuleDoc, prefix: &str, doctests: &mut Vec<Doctest>) {
        let module_path = if prefix.is_empty() {
            module.name.clone()
        } else {
            format!("{}::{}", prefix, module.name)
        };

        // Extract from module doc
        if let Some(ref doc) = module.doc {
            self.extract_from_doc_string(doc, &module_path, doctests);
        }

        // Extract from functions
        for func in &module.functions {
            self.extract_from_function(func, &module_path, doctests);
        }

        // Extract from types
        for ty in &module.types {
            self.extract_from_type(ty, &module_path, doctests);
        }

        // Extract from traits
        for tr in &module.traits {
            self.extract_from_trait(tr, &module_path, doctests);
        }

        // Extract from constants
        for c in &module.constants {
            self.extract_from_constant(c, &module_path, doctests);
        }

        // Recurse into submodules
        for submod in &module.modules {
            self.extract_from_module(submod, &module_path, doctests);
        }
    }

    fn extract_from_function(&self, func: &FunctionDoc, prefix: &str, doctests: &mut Vec<Doctest>) {
        let path = format!("{}::{}", prefix, func.name);
        if let Some(ref doc) = func.doc {
            self.extract_from_doc_string(doc, &path, doctests);
        }
        // Extract examples from doc_sections if available
        if let Some(ref sections) = func.doc_sections {
            for example in &sections.examples {
                self.extract_from_example(example, &path, doctests);
            }
        }
    }

    fn extract_from_type(&self, ty: &TypeDoc, prefix: &str, doctests: &mut Vec<Doctest>) {
        let path = format!("{}::{}", prefix, ty.name);
        if let Some(ref doc) = ty.doc {
            self.extract_from_doc_string(doc, &path, doctests);
        }
        // Extract examples from doc_sections if available
        if let Some(ref sections) = ty.doc_sections {
            for example in &sections.examples {
                self.extract_from_example(example, &path, doctests);
            }
        }

        // Extract from methods
        for method in &ty.methods {
            self.extract_from_function(method, &path, doctests);
        }
    }

    fn extract_from_trait(&self, tr: &TraitDoc, prefix: &str, doctests: &mut Vec<Doctest>) {
        let path = format!("{}::{}", prefix, tr.name);
        if let Some(ref doc) = tr.doc {
            self.extract_from_doc_string(doc, &path, doctests);
        }
        // Extract examples from doc_sections if available
        if let Some(ref sections) = tr.doc_sections {
            for example in &sections.examples {
                self.extract_from_example(example, &path, doctests);
            }
        }

        // Extract from required methods
        for method in &tr.required_methods {
            self.extract_from_function(method, &path, doctests);
        }

        // Extract from provided methods
        for method in &tr.provided_methods {
            self.extract_from_function(method, &path, doctests);
        }
    }

    fn extract_from_constant(&self, c: &ConstantDoc, prefix: &str, doctests: &mut Vec<Doctest>) {
        let path = format!("{}::{}", prefix, c.name);
        if let Some(ref doc) = c.doc {
            self.extract_from_doc_string(doc, &path, doctests);
        }
    }

    fn extract_from_doc_string(&self, doc: &str, path: &str, doctests: &mut Vec<Doctest>) {
        // Parse code blocks from markdown
        let code_blocks = extract_code_blocks(doc);
        for (code, attrs) in code_blocks {
            let should_test = attrs.get("d").is_some()
                || attrs.get("sounio").is_some()
                || (attrs.get("text").is_none()
                    && attrs.get("ignore").is_none()
                    && !attrs.contains_key("notest"));

            if !should_test {
                continue;
            }

            let doctest = Doctest {
                item_path: path.to_string(),
                code,
                should_pass: !attrs.contains_key("compile_fail"),
                should_panic: attrs.contains_key("should_panic"),
                ignore: attrs.contains_key("ignore"),
                no_run: attrs.contains_key("no_run"),
                line_number: None,
                source_file: None,
            };

            if self.matches_filter(&doctest) {
                doctests.push(doctest);
            }
        }
    }

    fn extract_from_example(&self, example: &ExampleDoc, path: &str, doctests: &mut Vec<Doctest>) {
        let doctest = Doctest {
            item_path: format!("{} (example)", path),
            code: example.code.clone(),
            should_pass: example.should_test,
            should_panic: example.should_panic,
            ignore: example.ignore,
            no_run: example.no_run,
            line_number: None,
            source_file: None,
        };

        if self.matches_filter(&doctest) {
            doctests.push(doctest);
        }
    }

    fn matches_filter(&self, doctest: &Doctest) -> bool {
        match &self.config.filter {
            Some(filter) => doctest.item_path.contains(filter),
            None => true,
        }
    }

    /// Run all extracted doctests
    pub fn run_doctests(&self, doctests: Vec<Doctest>) -> DoctestSummary {
        let start = Instant::now();
        let mut summary = DoctestSummary {
            total: doctests.len(),
            ..Default::default()
        };

        // Ensure temp directory exists
        let _ = std::fs::create_dir_all(&self.config.temp_dir);

        for (i, doctest) in doctests.into_iter().enumerate() {
            if doctest.ignore {
                summary.ignored += 1;
                summary.results.push(DoctestResult {
                    doctest,
                    passed: true,
                    duration: Duration::ZERO,
                    compile_output: None,
                    runtime_output: None,
                    error: None,
                });
                continue;
            }

            let result = self.run_single_doctest(&doctest, i);

            if result.passed {
                summary.passed += 1;
            } else {
                summary.failed += 1;
                if !self.config.continue_on_failure {
                    summary.results.push(result);
                    break;
                }
            }

            summary.results.push(result);
        }

        summary.duration = start.elapsed();
        summary
    }

    fn run_single_doctest(&self, doctest: &Doctest, index: usize) -> DoctestResult {
        let start = Instant::now();
        let test_file = self.config.temp_dir.join(format!("doctest_{}.sio", index));
        let output_file = self.config.temp_dir.join(format!("doctest_{}", index));

        // Wrap the code in a main function if needed
        let wrapped_code = self.wrap_code(&doctest.code);

        // Write test file
        if let Err(e) = std::fs::write(&test_file, &wrapped_code) {
            return DoctestResult {
                doctest: doctest.clone(),
                passed: false,
                duration: start.elapsed(),
                compile_output: None,
                runtime_output: None,
                error: Some(format!("Failed to write test file: {}", e)),
            };
        }

        // Compile
        let compile_result = Command::new(&self.config.compiler_path)
            .args([
                "build",
                test_file.to_str().unwrap(),
                "-o",
                output_file.to_str().unwrap(),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        let compile_output = match compile_result {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let combined = format!("{}{}", stdout, stderr);

                if !output.status.success() {
                    // Compilation failed
                    let passed = !doctest.should_pass; // compile_fail tests pass when compilation fails
                    return DoctestResult {
                        doctest: doctest.clone(),
                        passed,
                        duration: start.elapsed(),
                        compile_output: Some(combined),
                        runtime_output: None,
                        error: if !passed {
                            Some("Compilation failed".to_string())
                        } else {
                            None
                        },
                    };
                }
                Some(combined)
            }
            Err(e) => {
                return DoctestResult {
                    doctest: doctest.clone(),
                    passed: false,
                    duration: start.elapsed(),
                    compile_output: None,
                    runtime_output: None,
                    error: Some(format!("Failed to run compiler: {}", e)),
                };
            }
        };

        // If no_run, we're done
        if doctest.no_run {
            return DoctestResult {
                doctest: doctest.clone(),
                passed: true,
                duration: start.elapsed(),
                compile_output,
                runtime_output: None,
                error: None,
            };
        }

        // Run the compiled test
        let run_result = Command::new(&output_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        match run_result {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let runtime_output = format!("{}{}", stdout, stderr);

                let panicked = !output.status.success();
                let passed = if doctest.should_panic {
                    panicked
                } else {
                    !panicked
                };

                DoctestResult {
                    doctest: doctest.clone(),
                    passed,
                    duration: start.elapsed(),
                    compile_output,
                    runtime_output: Some(runtime_output),
                    error: if !passed {
                        Some(if doctest.should_panic {
                            "Expected panic but test passed".to_string()
                        } else {
                            "Test panicked/failed".to_string()
                        })
                    } else {
                        None
                    },
                }
            }
            Err(e) => DoctestResult {
                doctest: doctest.clone(),
                passed: false,
                duration: start.elapsed(),
                compile_output,
                runtime_output: None,
                error: Some(format!("Failed to run test: {}", e)),
            },
        }
    }

    fn wrap_code(&self, code: &str) -> String {
        // Check if code already has a main function
        if code.contains("fn main") {
            code.to_string()
        } else {
            // Wrap in main function
            let mut wrapped = String::new();
            wrapped.push_str("// Auto-generated doctest wrapper\n\n");

            // Extract any imports/use statements that should be at the top
            let mut imports = Vec::new();
            let mut body = Vec::new();

            for line in code.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("use ") || trimmed.starts_with("import ") {
                    imports.push(line);
                } else {
                    body.push(line);
                }
            }

            for import in imports {
                wrapped.push_str(import);
                wrapped.push('\n');
            }

            wrapped.push_str("\nfn main() {\n");
            for line in body {
                wrapped.push_str("    ");
                wrapped.push_str(line);
                wrapped.push('\n');
            }
            wrapped.push_str("}\n");

            wrapped
        }
    }

    /// Calculate documentation coverage for a crate
    pub fn calculate_coverage(&self, crate_doc: &CrateDoc) -> DocCoverage {
        let mut coverage = DocCoverage::default();
        self.coverage_from_module(&crate_doc.root_module, "", &mut coverage);
        coverage
    }

    fn coverage_from_module(&self, module: &ModuleDoc, prefix: &str, coverage: &mut DocCoverage) {
        let module_path = if prefix.is_empty() {
            module.name.clone()
        } else {
            format!("{}::{}", prefix, module.name)
        };

        // Count module
        coverage.total_items += 1;
        let kind = "module".to_string();
        let entry = coverage.by_kind.entry(kind).or_insert((0, 0));
        entry.0 += 1;

        if module.doc.is_some() {
            coverage.documented_items += 1;
            entry.1 += 1;
        } else {
            coverage.undocumented.push(module_path.clone());
        }

        // Count functions
        for func in &module.functions {
            self.count_function(func, &module_path, coverage);
        }

        // Count types
        for ty in &module.types {
            self.count_type(ty, &module_path, coverage);
        }

        // Count traits
        for tr in &module.traits {
            self.count_trait(tr, &module_path, coverage);
        }

        // Count constants
        for c in &module.constants {
            self.count_constant(c, &module_path, coverage);
        }

        // Recurse
        for submod in &module.modules {
            self.coverage_from_module(submod, &module_path, coverage);
        }
    }

    fn count_function(&self, func: &FunctionDoc, prefix: &str, coverage: &mut DocCoverage) {
        let path = format!("{}::{}", prefix, func.name);
        coverage.total_items += 1;

        let kind = "function".to_string();
        let entry = coverage.by_kind.entry(kind).or_insert((0, 0));
        entry.0 += 1;

        if func.doc.is_some() {
            coverage.documented_items += 1;
            entry.1 += 1;
        } else {
            coverage.undocumented.push(path.clone());
        }

        let has_examples = func
            .doc_sections
            .as_ref()
            .map(|s| !s.examples.is_empty())
            .unwrap_or(false);
        if has_examples {
            coverage.items_with_examples += 1;
        } else {
            coverage.no_examples.push(path);
        }
    }

    fn count_type(&self, ty: &TypeDoc, prefix: &str, coverage: &mut DocCoverage) {
        let path = format!("{}::{}", prefix, ty.name);
        coverage.total_items += 1;

        let kind = "type".to_string();
        let entry = coverage.by_kind.entry(kind).or_insert((0, 0));
        entry.0 += 1;

        if ty.doc.is_some() {
            coverage.documented_items += 1;
            entry.1 += 1;
        } else {
            coverage.undocumented.push(path.clone());
        }

        let has_examples = ty
            .doc_sections
            .as_ref()
            .map(|s| !s.examples.is_empty())
            .unwrap_or(false);
        if has_examples {
            coverage.items_with_examples += 1;
        } else {
            coverage.no_examples.push(path.clone());
        }

        // Count methods
        for method in &ty.methods {
            self.count_function(method, &path, coverage);
        }
    }

    fn count_trait(&self, tr: &TraitDoc, prefix: &str, coverage: &mut DocCoverage) {
        let path = format!("{}::{}", prefix, tr.name);
        coverage.total_items += 1;

        let kind = "trait".to_string();
        let entry = coverage.by_kind.entry(kind).or_insert((0, 0));
        entry.0 += 1;

        if tr.doc.is_some() {
            coverage.documented_items += 1;
            entry.1 += 1;
        } else {
            coverage.undocumented.push(path.clone());
        }

        let has_examples = tr
            .doc_sections
            .as_ref()
            .map(|s| !s.examples.is_empty())
            .unwrap_or(false);
        if has_examples {
            coverage.items_with_examples += 1;
        } else {
            coverage.no_examples.push(path.clone());
        }

        // Count methods
        for method in &tr.required_methods {
            self.count_function(method, &path, coverage);
        }
        for method in &tr.provided_methods {
            self.count_function(method, &path, coverage);
        }
    }

    fn count_constant(&self, c: &ConstantDoc, prefix: &str, coverage: &mut DocCoverage) {
        let path = format!("{}::{}", prefix, c.name);
        coverage.total_items += 1;

        let kind = "constant".to_string();
        let entry = coverage.by_kind.entry(kind).or_insert((0, 0));
        entry.0 += 1;

        if c.doc.is_some() {
            coverage.documented_items += 1;
            entry.1 += 1;
        } else {
            coverage.undocumented.push(path.clone());
        }

        // Constants don't typically have examples
        coverage.no_examples.push(path);
    }

    /// Print a summary of doctest results
    pub fn print_summary(&self, summary: &DoctestSummary) {
        println!("\n=== Doctest Results ===\n");

        for result in &summary.results {
            let status = if result.doctest.ignore {
                "IGNORED"
            } else if result.passed {
                "PASSED"
            } else {
                "FAILED"
            };

            let symbol = if result.doctest.ignore {
                "○"
            } else if result.passed {
                "✓"
            } else {
                "✗"
            };

            println!(
                "{} {} ... {} ({:?})",
                symbol, result.doctest.item_path, status, result.duration
            );

            if !result.passed && !result.doctest.ignore {
                if let Some(ref error) = result.error {
                    println!("    Error: {}", error);
                }
                if self.config.show_output {
                    if let Some(ref compile_output) = result.compile_output
                        && !compile_output.is_empty()
                    {
                        println!("    Compile output:\n{}", indent_lines(compile_output, 8));
                    }
                    if let Some(ref runtime_output) = result.runtime_output
                        && !runtime_output.is_empty()
                    {
                        println!("    Runtime output:\n{}", indent_lines(runtime_output, 8));
                    }
                }
            }
        }

        println!("\n-----------------------");
        println!("Total:   {}", summary.total);
        println!("Passed:  {}", summary.passed);
        println!("Failed:  {}", summary.failed);
        println!("Ignored: {}", summary.ignored);
        println!("Success: {:.1}%", summary.success_rate());
        println!("Time:    {:?}", summary.duration);
    }

    /// Print documentation coverage report
    pub fn print_coverage(&self, coverage: &DocCoverage) {
        println!("\n=== Documentation Coverage ===\n");

        println!(
            "Overall: {}/{} items documented ({:.1}%)",
            coverage.documented_items,
            coverage.total_items,
            coverage.coverage_percentage()
        );

        println!(
            "Examples: {}/{} items with examples ({:.1}%)",
            coverage.items_with_examples,
            coverage.total_items,
            coverage.example_coverage()
        );

        println!("\nBy kind:");
        for (kind, (total, documented)) in &coverage.by_kind {
            let pct = if *total == 0 {
                100.0
            } else {
                (*documented as f64 / *total as f64) * 100.0
            };
            println!("  {:12} {}/{} ({:.1}%)", kind, documented, total, pct);
        }

        if !coverage.undocumented.is_empty() {
            println!("\nUndocumented items ({}):", coverage.undocumented.len());
            for item in coverage.undocumented.iter().take(20) {
                println!("  - {}", item);
            }
            if coverage.undocumented.len() > 20 {
                println!("  ... and {} more", coverage.undocumented.len() - 20);
            }
        }
    }

    /// Clean up temporary files
    pub fn cleanup(&self) {
        let _ = std::fs::remove_dir_all(&self.config.temp_dir);
    }
}

/// Extract code blocks from markdown text
fn extract_code_blocks(markdown: &str) -> Vec<(String, BTreeMap<String, String>)> {
    let mut blocks = Vec::new();
    let mut in_code_block = false;
    let mut current_code = String::new();
    let mut current_attrs = BTreeMap::new();

    for line in markdown.lines() {
        if line.starts_with("```") {
            if in_code_block {
                // End of code block
                blocks.push((current_code.clone(), current_attrs.clone()));
                current_code.clear();
                current_attrs.clear();
                in_code_block = false;
            } else {
                // Start of code block - parse attributes
                let attrs_str = line.trim_start_matches('`');
                current_attrs = parse_code_block_attrs(attrs_str);
                in_code_block = true;
            }
        } else if in_code_block {
            if !current_code.is_empty() {
                current_code.push('\n');
            }
            current_code.push_str(line);
        }
    }

    blocks
}

/// Parse code block attributes like ```d,ignore,should_panic
fn parse_code_block_attrs(attrs: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();

    for part in attrs.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        if let Some((key, value)) = part.split_once('=') {
            map.insert(key.trim().to_string(), value.trim().to_string());
        } else {
            map.insert(part.to_string(), String::new());
        }
    }

    map
}

/// Indent all lines of text
fn indent_lines(text: &str, spaces: usize) -> String {
    let indent = " ".repeat(spaces);
    text.lines()
        .map(|line| format!("{}{}", indent, line))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code_blocks() {
        let markdown = r#"
Some text

```d
let x = 42;
```

More text

```d,should_panic
panic("oops");
```

```text
This is not a test
```
"#;

        let blocks = extract_code_blocks(markdown);
        assert_eq!(blocks.len(), 3);

        assert_eq!(blocks[0].0, "let x = 42;");
        assert!(blocks[0].1.contains_key("d"));

        assert_eq!(blocks[1].0, "panic(\"oops\");");
        assert!(blocks[1].1.contains_key("should_panic"));

        assert!(blocks[2].1.contains_key("text"));
    }

    #[test]
    fn test_parse_code_block_attrs() {
        let attrs = parse_code_block_attrs("d,ignore,name=test");
        assert!(attrs.contains_key("d"));
        assert!(attrs.contains_key("ignore"));
        assert_eq!(attrs.get("name"), Some(&"test".to_string()));
    }

    #[test]
    fn test_wrap_code() {
        let runner = DoctestRunner::new(DoctestConfig::default());

        let code = "let x = 42;";
        let wrapped = runner.wrap_code(code);
        assert!(wrapped.contains("fn main()"));
        assert!(wrapped.contains("let x = 42;"));

        let code_with_main = "fn main() { let x = 42; }";
        let wrapped = runner.wrap_code(code_with_main);
        assert_eq!(wrapped, code_with_main);
    }

    #[test]
    fn test_coverage_calculation() {
        let coverage = DocCoverage {
            total_items: 10,
            documented_items: 8,
            items_with_examples: 5,
            ..Default::default()
        };

        assert!((coverage.coverage_percentage() - 80.0).abs() < 0.1);
        assert!((coverage.example_coverage() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_summary_success_rate() {
        let summary = DoctestSummary {
            total: 10,
            passed: 7,
            failed: 2,
            ignored: 1,
            ..Default::default()
        };

        // 7 passed out of 9 run (10 - 1 ignored)
        let rate = summary.success_rate();
        assert!((rate - 77.78).abs() < 0.1);
    }
}
