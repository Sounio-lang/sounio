//! Diagnostic Emitters
//!
//! This module provides different output formats for diagnostics:
//! - Human-readable console output with colors
//! - JSON output for tooling integration
//! - SARIF output for static analysis tools

use super::*;
use std::io::{self, Write};
use std::sync::Mutex;

/// Trait for emitting diagnostics
pub trait DiagnosticEmitter: Send + Sync {
    /// Emit a single diagnostic
    fn emit(&self, diagnostic: &Diagnostic);

    /// Emit a diagnostic with source map for location info
    fn emit_with_source_map(&self, diagnostic: &Diagnostic, source_map: &SourceMap) {
        // Default implementation ignores source map
        self.emit(diagnostic);
    }

    /// Emit summary of errors and warnings
    fn emit_summary(&self, errors: usize, warnings: usize) {
        // Default: no summary
        let _ = (errors, warnings);
    }
}

/// Human-readable emitter for console output
pub struct HumanEmitter {
    /// Output writer (wrapped in Mutex for interior mutability)
    writer: Mutex<Box<dyn Write + Send>>,
    /// Whether to use ANSI colors
    use_colors: bool,
    /// Whether to show source code snippets
    show_snippets: bool,
    /// Number of context lines for snippets
    context_lines: usize,
}

impl HumanEmitter {
    /// Create a new human emitter writing to stderr
    pub fn stderr() -> Self {
        HumanEmitter {
            writer: Mutex::new(Box::new(io::stderr())),
            use_colors: is_terminal(),
            show_snippets: true,
            context_lines: 2,
        }
    }

    /// Create a new human emitter with custom writer
    pub fn new(writer: Box<dyn Write + Send>) -> Self {
        HumanEmitter {
            writer: Mutex::new(writer),
            use_colors: false,
            show_snippets: true,
            context_lines: 2,
        }
    }

    /// Enable or disable colors
    pub fn colors(mut self, enabled: bool) -> Self {
        self.use_colors = enabled;
        self
    }

    /// Enable or disable source snippets
    pub fn snippets(mut self, enabled: bool) -> Self {
        self.show_snippets = enabled;
        self
    }

    /// Format the header line
    fn format_header(&self, diagnostic: &Diagnostic) -> String {
        let level_str = if self.use_colors {
            format!(
                "{}{}\x1b[0m",
                diagnostic.level.color(),
                diagnostic.level.as_str()
            )
        } else {
            diagnostic.level.as_str().to_string()
        };

        let code_str = diagnostic
            .code
            .as_ref()
            .map(|c| {
                if self.use_colors {
                    format!("\x1b[1m[{}]\x1b[0m", c)
                } else {
                    format!("[{}]", c)
                }
            })
            .unwrap_or_default();

        let message = if self.use_colors {
            format!("\x1b[1m{}\x1b[0m", diagnostic.message)
        } else {
            diagnostic.message.clone()
        };

        format!("{}{}: {}", level_str, code_str, message)
    }

    /// Format a source snippet with highlighting
    fn format_snippet(&self, file: &SourceFile, label: &Label) -> String {
        let mut output = String::new();

        let (start_line, start_col) = file.line_col(label.span.start);
        let (end_line, end_col) = file.line_col(label.span.end);

        // Calculate line number width
        let max_line = end_line + self.context_lines;
        let line_width = format!("{}", max_line).len();

        // Location line
        let arrow = if self.use_colors {
            "\x1b[1;34m-->\x1b[0m"
        } else {
            "-->"
        };
        let location = format!(
            "{:>width$}{} {}:{}:{}",
            "",
            arrow,
            file.path.display(),
            start_line,
            start_col,
            width = line_width
        );
        output.push_str(&location);
        output.push('\n');

        // Separator
        let pipe = if self.use_colors {
            "\x1b[1;34m|\x1b[0m"
        } else {
            "|"
        };
        output.push_str(&format!("{:>width$} {}\n", "", pipe, width = line_width));

        // Source lines
        let first_line = start_line.saturating_sub(self.context_lines + 1);
        let last_line = (end_line + self.context_lines).min(file.line_count());

        for line_num in first_line..last_line {
            let line_num_1 = line_num + 1; // 1-indexed for display

            if let Some(line_content) = file.line(line_num) {
                // Line number and content
                let line_num_str = if self.use_colors {
                    format!(
                        "\x1b[1;34m{:>width$}\x1b[0m",
                        line_num_1,
                        width = line_width
                    )
                } else {
                    format!("{:>width$}", line_num_1, width = line_width)
                };

                output.push_str(&format!("{} {} {}\n", line_num_str, pipe, line_content));

                // Underline if this is a highlighted line
                if line_num_1 >= start_line && line_num_1 <= end_line {
                    let underline_start = if line_num_1 == start_line {
                        start_col - 1
                    } else {
                        0
                    };
                    let underline_end = if line_num_1 == end_line {
                        end_col - 1
                    } else {
                        line_content.len()
                    };

                    let spaces = " ".repeat(underline_start);
                    let underline_char = if label.primary { '^' } else { '-' };
                    let underline = underline_char
                        .to_string()
                        .repeat(underline_end.saturating_sub(underline_start).max(1));

                    let underline_colored = if self.use_colors {
                        let color = if label.primary {
                            "\x1b[1;31m"
                        } else {
                            "\x1b[1;34m"
                        };
                        format!("{}{}\x1b[0m", color, underline)
                    } else {
                        underline
                    };

                    output.push_str(&format!(
                        "{:>width$} {} {}{}",
                        "",
                        pipe,
                        spaces,
                        underline_colored,
                        width = line_width
                    ));

                    // Add label message on last line
                    if line_num_1 == end_line && !label.message.is_empty() {
                        output.push_str(&format!(" {}", label.message));
                    }
                    output.push('\n');
                }
            }
        }

        output
    }

    /// Write output to the writer
    fn write_output(&self, output: &str) {
        if let Ok(mut writer) = self.writer.lock() {
            let _ = write!(writer, "{}", output);
        }
    }
}

impl DiagnosticEmitter for HumanEmitter {
    fn emit(&self, diagnostic: &Diagnostic) {
        let mut output = self.format_header(diagnostic);
        output.push('\n');

        // Notes
        for note in &diagnostic.notes {
            let prefix = if self.use_colors {
                "  = \x1b[1;36mnote\x1b[0m:".to_string()
            } else {
                "  = note:".to_string()
            };
            output.push_str(&format!("{} {}\n", prefix, note));
        }

        // Help
        for help in &diagnostic.help {
            let prefix = if self.use_colors {
                "  = \x1b[1;32mhelp\x1b[0m:".to_string()
            } else {
                "  = help:".to_string()
            };
            output.push_str(&format!("{} {}\n", prefix, help));
        }

        // Suggestions
        for suggestion in &diagnostic.suggestions {
            let prefix = if self.use_colors {
                "  = \x1b[1;32msuggestion\x1b[0m:".to_string()
            } else {
                "  = suggestion:".to_string()
            };
            output.push_str(&format!(
                "{} {} (replace with `{}`)\n",
                prefix, suggestion.message, suggestion.replacement
            ));
        }

        output.push('\n');
        self.write_output(&output);
    }

    fn emit_with_source_map(&self, diagnostic: &Diagnostic, source_map: &SourceMap) {
        let mut output = self.format_header(diagnostic);
        output.push('\n');

        // Labels with source snippets
        if self.show_snippets {
            for label in &diagnostic.labels {
                if let Some(file) = source_map.get_file(label.span.file_id) {
                    output.push_str(&self.format_snippet(file, label));
                }
            }
        }

        // Notes
        for note in &diagnostic.notes {
            let prefix = if self.use_colors {
                "  = \x1b[1;36mnote\x1b[0m:".to_string()
            } else {
                "  = note:".to_string()
            };
            output.push_str(&format!("{} {}\n", prefix, note));
        }

        // Help
        for help in &diagnostic.help {
            let prefix = if self.use_colors {
                "  = \x1b[1;32mhelp\x1b[0m:".to_string()
            } else {
                "  = help:".to_string()
            };
            output.push_str(&format!("{} {}\n", prefix, help));
        }

        // Suggestions
        for suggestion in &diagnostic.suggestions {
            let prefix = if self.use_colors {
                "  = \x1b[1;32msuggestion\x1b[0m:".to_string()
            } else {
                "  = suggestion:".to_string()
            };
            output.push_str(&format!(
                "{} {} (replace with `{}`)\n",
                prefix, suggestion.message, suggestion.replacement
            ));
        }

        // Child diagnostics
        for child in &diagnostic.children {
            self.emit_with_source_map(child, source_map);
        }

        output.push('\n');
        self.write_output(&output);
    }

    fn emit_summary(&self, errors: usize, warnings: usize) {
        let mut summary = String::new();

        if errors > 0 {
            let err_str = if self.use_colors {
                "\x1b[1;31merror\x1b[0m".to_string()
            } else {
                "error".to_string()
            };

            summary.push_str(&format!(
                "{}: aborting due to {} previous error{}",
                err_str,
                errors,
                if errors == 1 { "" } else { "s" }
            ));

            if warnings > 0 {
                summary.push_str(&format!(
                    "; {} warning{} emitted",
                    warnings,
                    if warnings == 1 { "" } else { "s" }
                ));
            }
        } else if warnings > 0 {
            let warn_str = if self.use_colors {
                "\x1b[1;33mwarning\x1b[0m".to_string()
            } else {
                "warning".to_string()
            };

            summary.push_str(&format!(
                "{}: {} warning{} emitted",
                warn_str,
                warnings,
                if warnings == 1 { "" } else { "s" }
            ));
        }

        summary.push('\n');
        self.write_output(&summary);
    }
}

/// JSON emitter for machine-readable output
pub struct JsonEmitter {
    /// Output writer
    writer: Mutex<Box<dyn Write + Send>>,
    /// Whether to pretty-print
    pretty: bool,
}

impl JsonEmitter {
    /// Create a new JSON emitter writing to stdout
    pub fn stdout() -> Self {
        JsonEmitter {
            writer: Mutex::new(Box::new(io::stdout())),
            pretty: false,
        }
    }

    /// Create with custom writer
    pub fn new(writer: Box<dyn Write + Send>) -> Self {
        JsonEmitter {
            writer: Mutex::new(writer),
            pretty: false,
        }
    }

    /// Enable pretty-printing
    pub fn pretty(mut self, enabled: bool) -> Self {
        self.pretty = enabled;
        self
    }

    /// Convert diagnostic to JSON string
    fn to_json(&self, diagnostic: &Diagnostic, source_map: Option<&SourceMap>) -> String {
        let mut json = String::from("{");

        // Level
        json.push_str(&format!("\"level\":\"{}\",", diagnostic.level.as_str()));

        // Code
        if let Some(code) = &diagnostic.code {
            json.push_str(&format!("\"code\":\"{}\",", code));
        }

        // Message
        json.push_str(&format!(
            "\"message\":\"{}\",",
            escape_json(&diagnostic.message)
        ));

        // Labels
        json.push_str("\"labels\":[");
        for (i, label) in diagnostic.labels.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push('{');
            json.push_str(&format!("\"primary\":{},", label.primary));
            json.push_str(&format!("\"message\":\"{}\",", escape_json(&label.message)));
            json.push_str(&format!(
                "\"span\":{{\"start\":{},\"end\":{},\"file_id\":{}}}",
                label.span.start, label.span.end, label.span.file_id
            ));

            // Add resolved location if source map available
            if let Some(sm) = source_map
                && let Some(loc) = sm.lookup_span(label.span)
            {
                json.push_str(&format!(
                        ",\"location\":{{\"file\":\"{}\",\"start_line\":{},\"start_col\":{},\"end_line\":{},\"end_col\":{}}}",
                        escape_json(&loc.file_path.display().to_string()),
                        loc.start_line, loc.start_col, loc.end_line, loc.end_col
                    ));
            }

            json.push('}');
        }
        json.push_str("],");

        // Notes
        json.push_str("\"notes\":[");
        for (i, note) in diagnostic.notes.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!("\"{}\"", escape_json(note)));
        }
        json.push_str("],");

        // Help
        json.push_str("\"help\":[");
        for (i, help) in diagnostic.help.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!("\"{}\"", escape_json(help)));
        }
        json.push_str("],");

        // Suggestions
        json.push_str("\"suggestions\":[");
        for (i, suggestion) in diagnostic.suggestions.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&format!(
                "{{\"message\":\"{}\",\"replacement\":\"{}\",\"applicability\":\"{}\"}}",
                escape_json(&suggestion.message),
                escape_json(&suggestion.replacement),
                suggestion.applicability.as_str()
            ));
        }
        json.push_str("],");

        // Children
        json.push_str("\"children\":[");
        for (i, child) in diagnostic.children.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&self.to_json(child, source_map));
        }
        json.push(']');

        json.push('}');
        json
    }

    /// Write output to the writer
    fn write_output(&self, output: &str) {
        if let Ok(mut writer) = self.writer.lock() {
            let _ = writeln!(writer, "{}", output);
        }
    }
}

impl DiagnosticEmitter for JsonEmitter {
    fn emit(&self, diagnostic: &Diagnostic) {
        let json = self.to_json(diagnostic, None);
        let output = if self.pretty {
            // Simple pretty-print (add newlines after commas)
            json.replace(',', ",\n  ")
        } else {
            json
        };
        self.write_output(&output);
    }

    fn emit_with_source_map(&self, diagnostic: &Diagnostic, source_map: &SourceMap) {
        let json = self.to_json(diagnostic, Some(source_map));
        let output = if self.pretty {
            json.replace(',', ",\n  ")
        } else {
            json
        };
        self.write_output(&output);
    }
}

/// SARIF (Static Analysis Results Interchange Format) emitter
pub struct SarifEmitter {
    /// Output writer
    writer: Mutex<Box<dyn Write + Send>>,
    /// Tool name
    tool_name: String,
    /// Tool version
    tool_version: String,
    /// Collected results
    results: Mutex<Vec<String>>,
}

impl SarifEmitter {
    /// Create a new SARIF emitter
    pub fn new(writer: Box<dyn Write + Send>) -> Self {
        SarifEmitter {
            writer: Mutex::new(writer),
            tool_name: "dc".to_string(),
            tool_version: env!("CARGO_PKG_VERSION").to_string(),
            results: Mutex::new(Vec::new()),
        }
    }

    /// Write the complete SARIF document
    pub fn finish(&self) {
        let results = self
            .results
            .lock()
            .map(|r| r.join(",\n      "))
            .unwrap_or_default();
        let sarif = format!(
            r#"{{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [{{
    "tool": {{
      "driver": {{
        "name": "{}",
        "version": "{}"
      }}
    }},
    "results": [
      {}
    ]
  }}]
}}"#,
            self.tool_name, self.tool_version, results
        );

        if let Ok(mut writer) = self.writer.lock() {
            let _ = write!(writer, "{}", sarif);
        }
    }
}

impl DiagnosticEmitter for SarifEmitter {
    fn emit(&self, _diagnostic: &Diagnostic) {
        // SARIF needs the full document structure, so we collect results
        // and write them all at once in finish()
    }

    fn emit_with_source_map(&self, diagnostic: &Diagnostic, source_map: &SourceMap) {
        let level = match diagnostic.level {
            DiagnosticLevel::Error | DiagnosticLevel::Fatal | DiagnosticLevel::Bug => "error",
            DiagnosticLevel::Warning => "warning",
            _ => "note",
        };

        let mut locations = Vec::new();
        for label in &diagnostic.labels {
            if let Some(loc) = source_map.lookup_span(label.span) {
                locations.push(format!(
                    r#"{{
          "physicalLocation": {{
            "artifactLocation": {{
              "uri": "{}"
            }},
            "region": {{
              "startLine": {},
              "startColumn": {},
              "endLine": {},
              "endColumn": {}
            }}
          }}
        }}"#,
                    escape_json(&loc.file_path.display().to_string()),
                    loc.start_line,
                    loc.start_col,
                    loc.end_line,
                    loc.end_col
                ));
            }
        }

        let result = format!(
            r#"{{
      "ruleId": "{}",
      "level": "{}",
      "message": {{
        "text": "{}"
      }},
      "locations": [
        {}
      ]
    }}"#,
            diagnostic.code.as_deref().unwrap_or("unknown"),
            level,
            escape_json(&diagnostic.message),
            locations.join(",\n        ")
        );

        if let Ok(mut results) = self.results.lock() {
            results.push(result);
        }
    }
}

/// Escape special JSON characters
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Check if stderr is a terminal (simplified)
fn is_terminal() -> bool {
    // In a real implementation, use atty crate or std::io::IsTerminal
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct TestWriter {
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for TestWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if let Ok(mut buffer) = self.buffer.lock() {
                buffer.extend_from_slice(buf);
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_json_emitter() {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = TestWriter {
            buffer: buffer.clone(),
        };
        let emitter = JsonEmitter::new(Box::new(writer));

        let diagnostic = Diagnostic::error("Type mismatch")
            .with_code("T0001")
            .with_label(Span::new(10, 20, 1), "expected int");

        emitter.emit(&diagnostic);

        let output = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
        assert!(output.contains("\"level\":\"error\""));
        assert!(output.contains("\"code\":\"T0001\""));
        assert!(output.contains("\"message\":\"Type mismatch\""));
    }
}
