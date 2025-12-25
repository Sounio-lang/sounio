//! Diagnostics provider
//!
//! Converts compiler errors to LSP diagnostics.

use tower_lsp::lsp_types::*;

/// Provider for diagnostics
pub struct DiagnosticsProvider;

impl DiagnosticsProvider {
    /// Create a new diagnostics provider
    pub fn new() -> Self {
        Self
    }

    /// Convert a miette error to an LSP diagnostic
    pub fn miette_to_diagnostic(&self, err: &miette::Report, source: &str) -> Diagnostic {
        // Extract error message
        let message = err.to_string();

        // Try to extract span information from the error
        // For now, we'll create a diagnostic at the beginning of the file
        // In a real implementation, we'd parse the miette error for location info
        let range = self.extract_range_from_error(&message, source);

        // Determine severity
        let severity = Some(DiagnosticSeverity::ERROR);

        // Extract error code if available
        let code = self.extract_error_code(&message);

        Diagnostic {
            range,
            severity,
            code: code.map(NumberOrString::String),
            source: Some("sounio".to_string()),
            message: self.clean_message(&message),
            related_information: None,
            tags: None,
            code_description: None,
            data: None,
        }
    }

    /// Extract range information from error message
    fn extract_range_from_error(&self, message: &str, source: &str) -> Range {
        // Try to parse position from message
        // Common formats:
        // - "at position X"
        // - "line X, column Y"
        // - "X:Y"

        // Look for "at position X" pattern
        if let Some(pos) = message.find("at position ") {
            let start = pos + "at position ".len();
            let end = message[start..]
                .find(|c: char| !c.is_ascii_digit())
                .map(|e| start + e)
                .unwrap_or(message.len());

            if let Ok(offset) = message[start..end].parse::<usize>() {
                let (line, col) = offset_to_line_col(source, offset);
                return Range {
                    start: Position {
                        line: line as u32,
                        character: col as u32,
                    },
                    end: Position {
                        line: line as u32,
                        character: (col + 1) as u32,
                    },
                };
            }
        }

        // Default to start of file
        Range {
            start: Position {
                line: 0,
                character: 0,
            },
            end: Position {
                line: 0,
                character: 1,
            },
        }
    }

    /// Extract error code from message
    fn extract_error_code(&self, message: &str) -> Option<String> {
        // Look for patterns like "E0001" or "[E0001]"
        let msg = message.to_uppercase();

        for pattern in &[
            "[E", "E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9",
        ] {
            if let Some(pos) = msg.find(pattern) {
                let start = if msg[pos..].starts_with('[') {
                    pos + 1
                } else {
                    pos
                };
                let end = msg[start..]
                    .find(|c: char| !c.is_ascii_alphanumeric())
                    .map(|e| start + e)
                    .unwrap_or_else(|| (start + 5).min(msg.len()));

                let code = &message[start..end];
                if code.len() >= 2 && code.chars().next() == Some('E') {
                    return Some(code.to_string());
                }
            }
        }

        None
    }

    /// Clean up error message for display
    fn clean_message(&self, message: &str) -> String {
        // Remove ANSI escape codes if present
        let clean = strip_ansi_codes(message);

        // Remove redundant error prefixes
        let clean = clean
            .trim_start_matches("error: ")
            .trim_start_matches("Error: ")
            .trim_start_matches("error[E")
            .split(']')
            .last()
            .unwrap_or(&clean)
            .trim();

        clean.to_string()
    }

    /// Create a diagnostic from components
    pub fn create_diagnostic(
        &self,
        range: Range,
        message: String,
        severity: DiagnosticSeverity,
        code: Option<String>,
    ) -> Diagnostic {
        Diagnostic {
            range,
            severity: Some(severity),
            code: code.map(NumberOrString::String),
            source: Some("sounio".to_string()),
            message,
            related_information: None,
            tags: None,
            code_description: None,
            data: None,
        }
    }

    /// Create an error diagnostic
    pub fn error(&self, range: Range, message: impl Into<String>) -> Diagnostic {
        self.create_diagnostic(range, message.into(), DiagnosticSeverity::ERROR, None)
    }

    /// Create a warning diagnostic
    pub fn warning(&self, range: Range, message: impl Into<String>) -> Diagnostic {
        self.create_diagnostic(range, message.into(), DiagnosticSeverity::WARNING, None)
    }

    /// Create an info diagnostic
    pub fn info(&self, range: Range, message: impl Into<String>) -> Diagnostic {
        self.create_diagnostic(range, message.into(), DiagnosticSeverity::INFORMATION, None)
    }

    /// Create a hint diagnostic
    pub fn hint(&self, range: Range, message: impl Into<String>) -> Diagnostic {
        self.create_diagnostic(range, message.into(), DiagnosticSeverity::HINT, None)
    }

    /// Create a deprecated diagnostic
    pub fn deprecated(&self, range: Range, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::HINT),
            code: None,
            source: Some("sounio".to_string()),
            message: message.into(),
            related_information: None,
            tags: Some(vec![DiagnosticTag::DEPRECATED]),
            code_description: None,
            data: None,
        }
    }

    /// Create an unused diagnostic
    pub fn unused(&self, range: Range, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            range,
            severity: Some(DiagnosticSeverity::HINT),
            code: None,
            source: Some("sounio".to_string()),
            message: message.into(),
            related_information: None,
            tags: Some(vec![DiagnosticTag::UNNECESSARY]),
            code_description: None,
            data: None,
        }
    }
}

impl Default for DiagnosticsProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert byte offset to line/column
fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(source.len());
    let mut line = 0;
    let mut col = 0;

    for (i, c) in source.char_indices() {
        if i >= offset {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    (line, col)
}

/// Strip ANSI escape codes from a string
fn strip_ansi_codes(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip escape sequence
            if chars.peek() == Some(&'[') {
                chars.next();
                // Skip until 'm' (end of ANSI sequence)
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next == 'm' {
                        break;
                    }
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_ansi_codes() {
        let input = "\x1b[31mError\x1b[0m: something went wrong";
        let output = strip_ansi_codes(input);
        assert_eq!(output, "Error: something went wrong");
    }

    #[test]
    fn test_offset_to_line_col() {
        let source = "line 1\nline 2\nline 3";
        assert_eq!(offset_to_line_col(source, 0), (0, 0));
        assert_eq!(offset_to_line_col(source, 5), (0, 5));
        assert_eq!(offset_to_line_col(source, 7), (1, 0));
        assert_eq!(offset_to_line_col(source, 14), (2, 0));
    }
}
