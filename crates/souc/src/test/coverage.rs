//! Code Coverage Tracking
//!
//! Tracks and reports code coverage during test execution.
//!
//! # Features
//!
//! - Line coverage tracking
//! - Branch coverage tracking
//! - Function coverage tracking
//! - Multiple output formats (LCOV, HTML, JSON)
//! - Coverage threshold enforcement
//!
//! # Example
//!
//! ```rust
//! use sounio::test::coverage::{CoverageTracker, CoverageConfig};
//!
//! let mut tracker = CoverageTracker::new(CoverageConfig::default());
//! tracker.start_tracking();
//!
//! // Run tests...
//!
//! let report = tracker.generate_report();
//! println!("Coverage: {:.1}%", report.line_coverage_percent());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Configuration for coverage tracking
#[derive(Debug, Clone)]
pub struct CoverageConfig {
    /// Track line coverage
    pub track_lines: bool,
    /// Track branch coverage
    pub track_branches: bool,
    /// Track function coverage
    pub track_functions: bool,
    /// Files to include (glob patterns)
    pub include_patterns: Vec<String>,
    /// Files to exclude (glob patterns)
    pub exclude_patterns: Vec<String>,
    /// Minimum line coverage threshold (0.0 - 100.0)
    pub min_line_coverage: Option<f64>,
    /// Minimum branch coverage threshold (0.0 - 100.0)
    pub min_branch_coverage: Option<f64>,
    /// Minimum function coverage threshold (0.0 - 100.0)
    pub min_function_coverage: Option<f64>,
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            track_lines: true,
            track_branches: true,
            track_functions: true,
            include_patterns: vec!["**/*.sio".to_string()],
            exclude_patterns: vec!["**/test/**".to_string(), "**/tests/**".to_string()],
            min_line_coverage: None,
            min_branch_coverage: None,
            min_function_coverage: None,
        }
    }
}

/// Coverage data for a single source file
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileCoverage {
    /// Path to the source file
    pub path: PathBuf,
    /// Line numbers and their hit counts
    pub lines: HashMap<u32, u64>,
    /// Total executable lines
    pub total_lines: u32,
    /// Covered lines (hit at least once)
    pub covered_lines: u32,
    /// Branch coverage: (branch_id, (taken, not_taken))
    pub branches: HashMap<u32, (u64, u64)>,
    /// Total branches
    pub total_branches: u32,
    /// Covered branches (both taken and not_taken exercised)
    pub covered_branches: u32,
    /// Function coverage: (function_name, hit_count)
    pub functions: HashMap<String, u64>,
    /// Total functions
    pub total_functions: u32,
    /// Covered functions (called at least once)
    pub covered_functions: u32,
}

impl FileCoverage {
    /// Create new file coverage data
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            ..Default::default()
        }
    }

    /// Record a line hit
    pub fn hit_line(&mut self, line: u32) {
        *self.lines.entry(line).or_insert(0) += 1;
    }

    /// Record a branch hit
    pub fn hit_branch(&mut self, branch_id: u32, taken: bool) {
        let entry = self.branches.entry(branch_id).or_insert((0, 0));
        if taken {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }

    /// Record a function call
    pub fn hit_function(&mut self, name: &str) {
        *self.functions.entry(name.to_string()).or_insert(0) += 1;
    }

    /// Calculate line coverage percentage
    pub fn line_coverage_percent(&self) -> f64 {
        if self.total_lines == 0 {
            return 100.0;
        }
        (self.covered_lines as f64 / self.total_lines as f64) * 100.0
    }

    /// Calculate branch coverage percentage
    pub fn branch_coverage_percent(&self) -> f64 {
        if self.total_branches == 0 {
            return 100.0;
        }
        (self.covered_branches as f64 / self.total_branches as f64) * 100.0
    }

    /// Calculate function coverage percentage
    pub fn function_coverage_percent(&self) -> f64 {
        if self.total_functions == 0 {
            return 100.0;
        }
        (self.covered_functions as f64 / self.total_functions as f64) * 100.0
    }

    /// Finalize coverage calculations
    pub fn finalize(&mut self) {
        self.covered_lines = self.lines.values().filter(|&&v| v > 0).count() as u32;
        self.covered_branches = self
            .branches
            .values()
            .filter(|(t, n)| *t > 0 && *n > 0)
            .count() as u32;
        self.covered_functions = self.functions.values().filter(|&&v| v > 0).count() as u32;
    }
}

/// Overall coverage data across all files
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoverageData {
    /// Coverage per file
    pub files: HashMap<PathBuf, FileCoverage>,
    /// Test run timestamp
    pub timestamp: Option<String>,
}

impl CoverageData {
    /// Create new coverage data
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create file coverage
    pub fn get_file_mut(&mut self, path: &Path) -> &mut FileCoverage {
        let path_buf = path.to_path_buf();
        self.files
            .entry(path_buf.clone())
            .or_insert_with(|| FileCoverage::new(path_buf))
    }

    /// Merge another coverage data into this one
    pub fn merge(&mut self, other: &CoverageData) {
        for (path, other_file) in &other.files {
            let file = self.get_file_mut(path);

            // Merge line hits
            for (&line, &count) in &other_file.lines {
                *file.lines.entry(line).or_insert(0) += count;
            }

            // Merge branch hits
            for (&branch_id, &(taken, not_taken)) in &other_file.branches {
                let entry = file.branches.entry(branch_id).or_insert((0, 0));
                entry.0 += taken;
                entry.1 += not_taken;
            }

            // Merge function hits
            for (name, &count) in &other_file.functions {
                *file.functions.entry(name.clone()).or_insert(0) += count;
            }

            // Update totals (take max in case of discrepancy)
            file.total_lines = file.total_lines.max(other_file.total_lines);
            file.total_branches = file.total_branches.max(other_file.total_branches);
            file.total_functions = file.total_functions.max(other_file.total_functions);
        }
    }

    /// Finalize all file coverage data
    pub fn finalize(&mut self) {
        for file in self.files.values_mut() {
            file.finalize();
        }
    }
}

/// Coverage report with summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    /// Raw coverage data
    pub data: CoverageData,
    /// Total lines across all files
    pub total_lines: u32,
    /// Covered lines across all files
    pub covered_lines: u32,
    /// Total branches across all files
    pub total_branches: u32,
    /// Covered branches across all files
    pub covered_branches: u32,
    /// Total functions across all files
    pub total_functions: u32,
    /// Covered functions across all files
    pub covered_functions: u32,
    /// Files with coverage below threshold
    pub below_threshold: Vec<PathBuf>,
}

impl CoverageReport {
    /// Create a report from coverage data
    pub fn from_data(mut data: CoverageData, config: &CoverageConfig) -> Self {
        data.finalize();

        let mut total_lines = 0u32;
        let mut covered_lines = 0u32;
        let mut total_branches = 0u32;
        let mut covered_branches = 0u32;
        let mut total_functions = 0u32;
        let mut covered_functions = 0u32;
        let mut below_threshold = Vec::new();

        for (path, file) in &data.files {
            total_lines += file.total_lines;
            covered_lines += file.covered_lines;
            total_branches += file.total_branches;
            covered_branches += file.covered_branches;
            total_functions += file.total_functions;
            covered_functions += file.covered_functions;

            // Check thresholds
            if let Some(min) = config.min_line_coverage
                && file.line_coverage_percent() < min
            {
                below_threshold.push(path.clone());
            }
        }

        Self {
            data,
            total_lines,
            covered_lines,
            total_branches,
            covered_branches,
            total_functions,
            covered_functions,
            below_threshold,
        }
    }

    /// Get line coverage percentage
    pub fn line_coverage_percent(&self) -> f64 {
        if self.total_lines == 0 {
            return 100.0;
        }
        (self.covered_lines as f64 / self.total_lines as f64) * 100.0
    }

    /// Get branch coverage percentage
    pub fn branch_coverage_percent(&self) -> f64 {
        if self.total_branches == 0 {
            return 100.0;
        }
        (self.covered_branches as f64 / self.total_branches as f64) * 100.0
    }

    /// Get function coverage percentage
    pub fn function_coverage_percent(&self) -> f64 {
        if self.total_functions == 0 {
            return 100.0;
        }
        (self.covered_functions as f64 / self.total_functions as f64) * 100.0
    }

    /// Check if coverage meets thresholds
    pub fn meets_thresholds(&self, config: &CoverageConfig) -> bool {
        if let Some(min) = config.min_line_coverage
            && self.line_coverage_percent() < min
        {
            return false;
        }
        if let Some(min) = config.min_branch_coverage
            && self.branch_coverage_percent() < min
        {
            return false;
        }
        if let Some(min) = config.min_function_coverage
            && self.function_coverage_percent() < min
        {
            return false;
        }
        true
    }

    /// Format as summary string
    pub fn summary(&self) -> String {
        format!(
            "Coverage Summary:\n  Lines:     {}/{} ({:.1}%)\n  Branches:  {}/{} ({:.1}%)\n  Functions: {}/{} ({:.1}%)",
            self.covered_lines,
            self.total_lines,
            self.line_coverage_percent(),
            self.covered_branches,
            self.total_branches,
            self.branch_coverage_percent(),
            self.covered_functions,
            self.total_functions,
            self.function_coverage_percent(),
        )
    }

    /// Export as LCOV format
    pub fn to_lcov(&self) -> String {
        let mut output = String::new();

        for (path, file) in &self.data.files {
            output.push_str("TN:\n");
            output.push_str(&format!("SF:{}\n", path.display()));

            // Function coverage
            for (name, &count) in &file.functions {
                output.push_str(&format!("FN:0,{}\n", name));
                output.push_str(&format!("FNDA:{},{}\n", count, name));
            }
            output.push_str(&format!("FNF:{}\n", file.total_functions));
            output.push_str(&format!("FNH:{}\n", file.covered_functions));

            // Branch coverage
            let mut branch_idx = 0;
            for (&branch_id, &(taken, not_taken)) in &file.branches {
                output.push_str(&format!("BRDA:{},0,{},{}\n", branch_id, branch_idx, taken));
                output.push_str(&format!(
                    "BRDA:{},0,{},{}\n",
                    branch_id,
                    branch_idx + 1,
                    not_taken
                ));
                branch_idx += 2;
            }
            output.push_str(&format!("BRF:{}\n", file.total_branches * 2));
            output.push_str(&format!("BRH:{}\n", file.covered_branches * 2));

            // Line coverage
            for (&line, &count) in &file.lines {
                output.push_str(&format!("DA:{},{}\n", line, count));
            }
            output.push_str(&format!("LF:{}\n", file.total_lines));
            output.push_str(&format!("LH:{}\n", file.covered_lines));

            output.push_str("end_of_record\n");
        }

        output
    }

    /// Export as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Generate HTML report
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>Sounio Coverage Report</title>\n");
        html.push_str("<style>\n");
        html.push_str(
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }\n",
        );
        html.push_str("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n");
        html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        html.push_str("th { background-color: #4a90d9; color: white; }\n");
        html.push_str("tr:nth-child(even) { background-color: #f2f2f2; }\n");
        html.push_str(".high { color: #28a745; }\n");
        html.push_str(".medium { color: #ffc107; }\n");
        html.push_str(".low { color: #dc3545; }\n");
        html.push_str(
            ".progress { background-color: #e0e0e0; border-radius: 4px; height: 20px; }\n",
        );
        html.push_str(
            ".progress-bar { height: 100%; border-radius: 4px; transition: width 0.3s; }\n",
        );
        html.push_str("</style>\n</head>\n<body>\n");

        html.push_str("<h1>Sounio Coverage Report</h1>\n");

        // Summary
        html.push_str("<h2>Summary</h2>\n");
        html.push_str("<table>\n");
        html.push_str("<tr><th>Metric</th><th>Covered</th><th>Total</th><th>Percentage</th><th>Bar</th></tr>\n");

        let metrics = [
            (
                "Lines",
                self.covered_lines,
                self.total_lines,
                self.line_coverage_percent(),
            ),
            (
                "Branches",
                self.covered_branches,
                self.total_branches,
                self.branch_coverage_percent(),
            ),
            (
                "Functions",
                self.covered_functions,
                self.total_functions,
                self.function_coverage_percent(),
            ),
        ];

        for (name, covered, total, percent) in &metrics {
            let class = if *percent >= 80.0 {
                "high"
            } else if *percent >= 50.0 {
                "medium"
            } else {
                "low"
            };
            let color = if *percent >= 80.0 {
                "#28a745"
            } else if *percent >= 50.0 {
                "#ffc107"
            } else {
                "#dc3545"
            };

            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td class=\"{}\">{:.1}%</td><td><div class=\"progress\"><div class=\"progress-bar\" style=\"width: {:.1}%; background-color: {};\"></div></div></td></tr>\n",
                name, covered, total, class, percent, percent, color
            ));
        }

        html.push_str("</table>\n");

        // File details
        html.push_str("<h2>File Details</h2>\n");
        html.push_str("<table>\n");
        html.push_str("<tr><th>File</th><th>Lines</th><th>Branches</th><th>Functions</th></tr>\n");

        for (path, file) in &self.data.files {
            let line_pct = file.line_coverage_percent();
            let branch_pct = file.branch_coverage_percent();
            let fn_pct = file.function_coverage_percent();

            html.push_str(&format!(
                "<tr><td>{}</td><td class=\"{}\">{:.1}%</td><td class=\"{}\">{:.1}%</td><td class=\"{}\">{:.1}%</td></tr>\n",
                path.display(),
                coverage_class(line_pct), line_pct,
                coverage_class(branch_pct), branch_pct,
                coverage_class(fn_pct), fn_pct,
            ));
        }

        html.push_str("</table>\n");
        html.push_str("</body>\n</html>");

        html
    }
}

fn coverage_class(percent: f64) -> &'static str {
    if percent >= 80.0 {
        "high"
    } else if percent >= 50.0 {
        "medium"
    } else {
        "low"
    }
}

/// Coverage tracker that instruments and tracks code execution
pub struct CoverageTracker {
    config: CoverageConfig,
    data: CoverageData,
    /// Currently tracking
    active: bool,
    /// Executable lines per file (from static analysis)
    executable_lines: HashMap<PathBuf, HashSet<u32>>,
    /// Branches per file
    branch_points: HashMap<PathBuf, HashSet<u32>>,
    /// Functions per file
    function_defs: HashMap<PathBuf, HashSet<String>>,
}

impl CoverageTracker {
    /// Create a new coverage tracker
    pub fn new(config: CoverageConfig) -> Self {
        Self {
            config,
            data: CoverageData::new(),
            active: false,
            executable_lines: HashMap::new(),
            branch_points: HashMap::new(),
            function_defs: HashMap::new(),
        }
    }

    /// Start tracking coverage
    pub fn start_tracking(&mut self) {
        self.active = true;
        self.data = CoverageData::new();
        self.data.timestamp = Some(chrono_timestamp());
    }

    /// Stop tracking coverage
    pub fn stop_tracking(&mut self) {
        self.active = false;
    }

    /// Check if tracking is active
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Record a line execution
    pub fn record_line(&mut self, file: &Path, line: u32) {
        if !self.active {
            return;
        }
        self.data.get_file_mut(file).hit_line(line);
    }

    /// Record a branch execution
    pub fn record_branch(&mut self, file: &Path, branch_id: u32, taken: bool) {
        if !self.active {
            return;
        }
        self.data.get_file_mut(file).hit_branch(branch_id, taken);
    }

    /// Record a function call
    pub fn record_function(&mut self, file: &Path, function_name: &str) {
        if !self.active {
            return;
        }
        self.data.get_file_mut(file).hit_function(function_name);
    }

    /// Analyze source files to find executable lines, branches, and functions
    pub fn analyze_source(&mut self, file: &Path, source: &str) {
        let mut executable = HashSet::new();
        let mut branches = HashSet::new();
        let mut functions = HashSet::new();

        let mut branch_id = 0u32;

        for (line_num, line) in source.lines().enumerate() {
            let line_num = (line_num + 1) as u32;
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with("/*") {
                continue;
            }

            // Skip structural-only lines
            if trimmed == "{" || trimmed == "}" || trimmed == ";" {
                continue;
            }

            // Detect function definitions
            if (trimmed.starts_with("fn ") || trimmed.contains("fn "))
                && let Some(name) = extract_function_name(trimmed)
            {
                functions.insert(name);
            }

            // Detect branch points
            if trimmed.starts_with("if ")
                || trimmed.starts_with("else")
                || trimmed.starts_with("match ")
                || trimmed.starts_with("while ")
                || trimmed.starts_with("for ")
            {
                branches.insert(branch_id);
                branch_id += 1;
            }

            // Mark as executable
            executable.insert(line_num);
        }

        // Store analysis results
        self.executable_lines.insert(file.to_path_buf(), executable);
        self.branch_points.insert(file.to_path_buf(), branches);
        self.function_defs.insert(file.to_path_buf(), functions);

        // Initialize file coverage with totals
        let file_cov = self.data.get_file_mut(file);
        file_cov.total_lines = self
            .executable_lines
            .get(file)
            .map(|s| s.len() as u32)
            .unwrap_or(0);
        file_cov.total_branches = self
            .branch_points
            .get(file)
            .map(|s| s.len() as u32)
            .unwrap_or(0);
        file_cov.total_functions = self
            .function_defs
            .get(file)
            .map(|s| s.len() as u32)
            .unwrap_or(0);
    }

    /// Generate a coverage report
    pub fn generate_report(&self) -> CoverageReport {
        CoverageReport::from_data(self.data.clone(), &self.config)
    }

    /// Reset coverage data
    pub fn reset(&mut self) {
        self.data = CoverageData::new();
    }

    /// Save coverage data to file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.data)?;
        std::fs::write(path, json)
    }

    /// Load coverage data from file
    pub fn load(&mut self, path: &Path) -> std::io::Result<()> {
        let json = std::fs::read_to_string(path)?;
        self.data = serde_json::from_str(&json)?;
        Ok(())
    }

    /// Merge coverage data from another tracker
    pub fn merge(&mut self, other: &CoverageTracker) {
        self.data.merge(&other.data);
    }
}

impl Default for CoverageTracker {
    fn default() -> Self {
        Self::new(CoverageConfig::default())
    }
}

/// Extract function name from a line of code
fn extract_function_name(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if let Some(idx) = trimmed.find("fn ") {
        let rest = &trimmed[idx + 3..];
        let name_end = rest.find(|c: char| c == '(' || c == '<' || c.is_whitespace())?;
        Some(rest[..name_end].to_string())
    } else {
        None
    }
}

/// Get current timestamp in ISO format
fn chrono_timestamp() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{}", now)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_coverage_line() {
        let mut cov = FileCoverage::new(PathBuf::from("test.sio"));
        cov.total_lines = 10;

        cov.hit_line(1);
        cov.hit_line(1);
        cov.hit_line(2);

        cov.finalize();

        assert_eq!(cov.lines.get(&1), Some(&2));
        assert_eq!(cov.covered_lines, 2);
        assert_eq!(cov.line_coverage_percent(), 20.0);
    }

    #[test]
    fn test_file_coverage_branch() {
        let mut cov = FileCoverage::new(PathBuf::from("test.sio"));
        cov.total_branches = 2;

        cov.hit_branch(0, true);
        cov.hit_branch(0, false); // Branch 0 fully covered
        cov.hit_branch(1, true); // Branch 1 only true taken

        cov.finalize();

        assert_eq!(cov.covered_branches, 1); // Only branch 0 is fully covered
    }

    #[test]
    fn test_coverage_merge() {
        let mut data1 = CoverageData::new();
        data1.get_file_mut(Path::new("a.sio")).hit_line(1);

        let mut data2 = CoverageData::new();
        data2.get_file_mut(Path::new("a.sio")).hit_line(1);
        data2.get_file_mut(Path::new("a.sio")).hit_line(2);

        data1.merge(&data2);

        let file = data1.files.get(Path::new("a.sio")).unwrap();
        assert_eq!(file.lines.get(&1), Some(&2));
        assert_eq!(file.lines.get(&2), Some(&1));
    }

    #[test]
    fn test_tracker_basic() {
        let mut tracker = CoverageTracker::new(CoverageConfig::default());
        tracker.start_tracking();

        assert!(tracker.is_active());

        tracker.record_line(Path::new("test.sio"), 1);
        tracker.record_line(Path::new("test.sio"), 2);
        tracker.record_function(Path::new("test.sio"), "main");

        tracker.stop_tracking();
        assert!(!tracker.is_active());

        let report = tracker.generate_report();
        assert!(report.data.files.contains_key(Path::new("test.sio")));
    }

    #[test]
    fn test_lcov_output() {
        let mut data = CoverageData::new();
        let file = data.get_file_mut(Path::new("test.sio"));
        file.total_lines = 10;
        file.hit_line(1);
        file.hit_line(2);
        file.hit_function("main");
        file.total_functions = 1;

        let report = CoverageReport::from_data(data, &CoverageConfig::default());
        let lcov = report.to_lcov();

        assert!(lcov.contains("SF:test.sio"));
        assert!(lcov.contains("DA:1,1"));
        assert!(lcov.contains("FN:0,main"));
    }

    #[test]
    fn test_extract_function_name() {
        assert_eq!(
            extract_function_name("fn main() {"),
            Some("main".to_string())
        );
        assert_eq!(
            extract_function_name("pub fn foo(x: i32) -> i32 {"),
            Some("foo".to_string())
        );
        assert_eq!(
            extract_function_name("fn generic<T>(x: T) {"),
            Some("generic".to_string())
        );
        assert_eq!(extract_function_name("let x = 5"), None);
    }

    #[test]
    fn test_coverage_thresholds() {
        let mut data = CoverageData::new();
        let file = data.get_file_mut(Path::new("test.sio"));
        file.total_lines = 100;
        // Add line hits so finalize() calculates correct covered_lines
        for i in 1..=80 {
            file.hit_line(i);
        }

        let config = CoverageConfig {
            min_line_coverage: Some(75.0),
            ..Default::default()
        };

        let report = CoverageReport::from_data(data.clone(), &config);
        assert!(report.meets_thresholds(&config));

        let strict_config = CoverageConfig {
            min_line_coverage: Some(90.0),
            ..Default::default()
        };
        let report2 = CoverageReport::from_data(data, &strict_config);
        assert!(!report2.meets_thresholds(&strict_config));
    }

    #[test]
    fn test_html_output() {
        let mut data = CoverageData::new();
        let file = data.get_file_mut(Path::new("src/main.sio"));
        file.total_lines = 100;
        file.covered_lines = 85;
        file.total_functions = 10;
        file.covered_functions = 8;

        let report = CoverageReport::from_data(data, &CoverageConfig::default());
        let html = report.to_html();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Coverage Report"));
        assert!(html.contains("src/main.sio"));
    }
}
