//! Rich Diagnostic Renderer
//!
//! Renders diagnostics with:
//! - ANSI colors (when supported)
//! - Unicode box drawing (when supported)
//! - ASCII fallback
//! - Contextual source snippets
//! - Multi-line annotations

use std::io::{self, Write};

use super::{Diagnostic, DiagnosticLevel, Label, SourceMap};

/// Terminal capabilities
#[derive(Debug, Clone, Copy)]
pub struct TerminalCaps {
    /// Supports ANSI color codes
    pub colors: bool,
    /// Supports Unicode box drawing
    pub unicode: bool,
    /// Terminal width (for wrapping)
    pub width: usize,
    /// Supports hyperlinks (OSC 8)
    pub hyperlinks: bool,
}

impl TerminalCaps {
    /// Detect terminal capabilities
    pub fn detect() -> Self {
        let colors = std::env::var("NO_COLOR").is_err() && is_terminal_stderr();

        let unicode = std::env::var("DEMETRIOS_ASCII").is_err();

        let width = terminal_width().unwrap_or(80);

        let hyperlinks = std::env::var("TERM_PROGRAM")
            .map(|p| p.contains("iTerm") || p.contains("WezTerm") || p.contains("kitty"))
            .unwrap_or(false);

        Self {
            colors,
            unicode,
            width,
            hyperlinks,
        }
    }

    /// Force no colors (for testing, piping)
    pub fn plain() -> Self {
        Self {
            colors: false,
            unicode: false,
            width: 80,
            hyperlinks: false,
        }
    }

    /// Force colors on
    pub fn with_colors() -> Self {
        let mut caps = Self::detect();
        caps.colors = true;
        caps
    }
}

impl Default for TerminalCaps {
    fn default() -> Self {
        Self::detect()
    }
}

/// ANSI color codes
pub mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const ITALIC: &str = "\x1b[3m";
    pub const UNDERLINE: &str = "\x1b[4m";

    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const WHITE: &str = "\x1b[37m";

    pub const BRIGHT_RED: &str = "\x1b[91m";
    pub const BRIGHT_GREEN: &str = "\x1b[92m";
    pub const BRIGHT_YELLOW: &str = "\x1b[93m";
    pub const BRIGHT_BLUE: &str = "\x1b[94m";
    pub const BRIGHT_CYAN: &str = "\x1b[96m";
}

/// Unicode box drawing characters
pub mod box_chars {
    pub mod unicode {
        pub const VERTICAL: &str = "│";
        pub const HORIZONTAL: &str = "─";
        pub const TOP_LEFT: &str = "┌";
        pub const TOP_RIGHT: &str = "┐";
        pub const BOTTOM_LEFT: &str = "└";
        pub const BOTTOM_RIGHT: &str = "┘";
        pub const VERTICAL_RIGHT: &str = "├";
        pub const VERTICAL_LEFT: &str = "┤";
        pub const HORIZONTAL_DOWN: &str = "┬";
        pub const HORIZONTAL_UP: &str = "┴";
        pub const CROSS: &str = "┼";
        pub const ARROW_RIGHT: &str = "→";
        pub const ARROW_LEFT: &str = "←";
        pub const BULLET: &str = "•";
        pub const CARET: &str = "^";
        pub const UNDERLINE: &str = "─";
    }

    pub mod ascii {
        pub const VERTICAL: &str = "|";
        pub const HORIZONTAL: &str = "-";
        pub const TOP_LEFT: &str = "+";
        pub const TOP_RIGHT: &str = "+";
        pub const BOTTOM_LEFT: &str = "+";
        pub const BOTTOM_RIGHT: &str = "+";
        pub const VERTICAL_RIGHT: &str = "+";
        pub const VERTICAL_LEFT: &str = "+";
        pub const HORIZONTAL_DOWN: &str = "+";
        pub const HORIZONTAL_UP: &str = "+";
        pub const CROSS: &str = "+";
        pub const ARROW_RIGHT: &str = "->";
        pub const ARROW_LEFT: &str = "<-";
        pub const BULLET: &str = "*";
        pub const CARET: &str = "^";
        pub const UNDERLINE: &str = "~";
    }
}

/// Style for different diagnostic elements
#[derive(Debug, Clone)]
pub struct DiagnosticStyle {
    /// Level-specific color
    pub level_color: &'static str,
    /// Primary highlight color
    pub primary_color: &'static str,
    /// Secondary highlight color
    pub secondary_color: &'static str,
    /// Note color
    pub note_color: &'static str,
    /// Help color
    pub help_color: &'static str,
}

impl DiagnosticStyle {
    pub fn for_level(level: DiagnosticLevel) -> Self {
        match level {
            DiagnosticLevel::Bug | DiagnosticLevel::Fatal | DiagnosticLevel::Error => Self {
                level_color: colors::BRIGHT_RED,
                primary_color: colors::RED,
                secondary_color: colors::BLUE,
                note_color: colors::CYAN,
                help_color: colors::GREEN,
            },
            DiagnosticLevel::Warning => Self {
                level_color: colors::BRIGHT_YELLOW,
                primary_color: colors::YELLOW,
                secondary_color: colors::BLUE,
                note_color: colors::CYAN,
                help_color: colors::GREEN,
            },
            DiagnosticLevel::Note => Self {
                level_color: colors::BRIGHT_BLUE,
                primary_color: colors::BLUE,
                secondary_color: colors::CYAN,
                note_color: colors::CYAN,
                help_color: colors::GREEN,
            },
            DiagnosticLevel::Help => Self {
                level_color: colors::BRIGHT_CYAN,
                primary_color: colors::CYAN,
                secondary_color: colors::BLUE,
                note_color: colors::DIM,
                help_color: colors::GREEN,
            },
        }
    }
}

/// Box character provider trait
trait BoxChars: Sync {
    fn vertical(&self) -> &'static str;
    fn horizontal(&self) -> &'static str;
    fn top_left(&self) -> &'static str;
    fn bottom_left(&self) -> &'static str;
    fn vertical_right(&self) -> &'static str;
    fn arrow_right(&self) -> &'static str;
    fn bullet(&self) -> &'static str;
    fn caret(&self) -> &'static str;
    fn underline(&self) -> &'static str;
}

struct UnicodeBoxChars;
struct AsciiBoxChars;

impl BoxChars for UnicodeBoxChars {
    fn vertical(&self) -> &'static str {
        box_chars::unicode::VERTICAL
    }
    fn horizontal(&self) -> &'static str {
        box_chars::unicode::HORIZONTAL
    }
    fn top_left(&self) -> &'static str {
        box_chars::unicode::TOP_LEFT
    }
    fn bottom_left(&self) -> &'static str {
        box_chars::unicode::BOTTOM_LEFT
    }
    fn vertical_right(&self) -> &'static str {
        box_chars::unicode::VERTICAL_RIGHT
    }
    fn arrow_right(&self) -> &'static str {
        box_chars::unicode::ARROW_RIGHT
    }
    fn bullet(&self) -> &'static str {
        box_chars::unicode::BULLET
    }
    fn caret(&self) -> &'static str {
        box_chars::unicode::CARET
    }
    fn underline(&self) -> &'static str {
        box_chars::unicode::UNDERLINE
    }
}

impl BoxChars for AsciiBoxChars {
    fn vertical(&self) -> &'static str {
        box_chars::ascii::VERTICAL
    }
    fn horizontal(&self) -> &'static str {
        box_chars::ascii::HORIZONTAL
    }
    fn top_left(&self) -> &'static str {
        box_chars::ascii::TOP_LEFT
    }
    fn bottom_left(&self) -> &'static str {
        box_chars::ascii::BOTTOM_LEFT
    }
    fn vertical_right(&self) -> &'static str {
        box_chars::ascii::VERTICAL_RIGHT
    }
    fn arrow_right(&self) -> &'static str {
        box_chars::ascii::ARROW_RIGHT
    }
    fn bullet(&self) -> &'static str {
        box_chars::ascii::BULLET
    }
    fn caret(&self) -> &'static str {
        box_chars::ascii::CARET
    }
    fn underline(&self) -> &'static str {
        box_chars::ascii::UNDERLINE
    }
}

static UNICODE_BOX: UnicodeBoxChars = UnicodeBoxChars;
static ASCII_BOX: AsciiBoxChars = AsciiBoxChars;

/// Rich diagnostic renderer
pub struct RichRenderer<'a, W: Write> {
    /// Output writer
    writer: &'a mut W,
    /// Terminal capabilities
    caps: TerminalCaps,
    /// Source map for looking up code
    source_map: &'a SourceMap,
    /// Box drawing characters to use
    box_chars: &'static dyn BoxChars,
    /// Context lines around snippets
    context_lines: usize,
}

impl<'a, W: Write> RichRenderer<'a, W> {
    pub fn new(writer: &'a mut W, caps: TerminalCaps, source_map: &'a SourceMap) -> Self {
        let box_chars: &'static dyn BoxChars = if caps.unicode {
            &UNICODE_BOX
        } else {
            &ASCII_BOX
        };

        Self {
            writer,
            caps,
            source_map,
            box_chars,
            context_lines: 2,
        }
    }

    /// Set number of context lines
    pub fn with_context_lines(mut self, lines: usize) -> Self {
        self.context_lines = lines;
        self
    }

    /// Render a single diagnostic
    pub fn render(&mut self, diag: &Diagnostic) -> io::Result<()> {
        let style = DiagnosticStyle::for_level(diag.level);

        // Header: error[E0312]: message
        self.render_header(diag, &style)?;

        // Source snippets with labels
        self.render_source_snippets(diag, &style)?;

        // Notes
        for note in &diag.notes {
            self.render_note(note, &style)?;
        }

        // Help
        for help in &diag.help {
            self.render_help(help, &style)?;
        }

        // Suggestions
        self.render_suggestions(diag, &style)?;

        // Children
        for child in &diag.children {
            self.render(child)?;
        }

        writeln!(self.writer)?;
        Ok(())
    }

    fn render_header(&mut self, diag: &Diagnostic, style: &DiagnosticStyle) -> io::Result<()> {
        if self.caps.colors {
            write!(self.writer, "{}{}", colors::BOLD, style.level_color)?;
        }

        write!(self.writer, "{}", diag.level)?;

        // Error code
        if let Some(ref code) = diag.code {
            write!(self.writer, "[{}]", code)?;
        }

        if self.caps.colors {
            write!(self.writer, "{}", colors::RESET)?;
        }

        write!(self.writer, ": ")?;

        if self.caps.colors {
            write!(self.writer, "{}", colors::BOLD)?;
        }

        writeln!(self.writer, "{}", diag.message)?;

        if self.caps.colors {
            write!(self.writer, "{}", colors::RESET)?;
        }

        Ok(())
    }

    fn render_source_snippets(
        &mut self,
        diag: &Diagnostic,
        style: &DiagnosticStyle,
    ) -> io::Result<()> {
        for label in &diag.labels {
            self.render_label_snippet(label, style)?;
        }
        Ok(())
    }

    fn render_label_snippet(&mut self, label: &Label, style: &DiagnosticStyle) -> io::Result<()> {
        let Some(file) = self.source_map.get_file(label.span.file_id) else {
            return Ok(());
        };

        let (start_line, start_col) = file.line_col(label.span.start);
        let (end_line, end_col) = file.line_col(label.span.end);

        // Calculate line number width for alignment
        let max_line = (end_line + self.context_lines).min(file.line_count());
        let line_width = format!("{}", max_line).len();

        // Location line: --> file:line:col
        if self.caps.colors {
            write!(self.writer, "{}", colors::BLUE)?;
        }
        write!(
            self.writer,
            "{:>width$} {} ",
            "",
            self.box_chars.arrow_right(),
            width = line_width
        )?;
        if self.caps.colors {
            write!(self.writer, "{}", colors::RESET)?;
        }
        writeln!(
            self.writer,
            "{}:{}:{}",
            file.path.display(),
            start_line,
            start_col
        )?;

        // Empty gutter line
        self.render_gutter(None, line_width)?;
        writeln!(self.writer)?;

        // Source lines
        let first_line = start_line.saturating_sub(self.context_lines + 1);
        let last_line = (end_line + self.context_lines).min(file.line_count());

        for line_num in first_line..last_line {
            let line_num_1 = line_num + 1; // 1-indexed

            if let Some(line_content) = file.line(line_num) {
                // Line number and source
                self.render_gutter(Some(line_num_1), line_width)?;
                writeln!(self.writer, " {}", line_content)?;

                // Underline if this line contains the span
                if line_num_1 >= start_line && line_num_1 <= end_line {
                    self.render_underline(
                        line_num_1,
                        start_line,
                        start_col,
                        end_line,
                        end_col,
                        line_content.len(),
                        &label.message,
                        label.primary,
                        line_width,
                        style,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn render_gutter(&mut self, line_num: Option<usize>, width: usize) -> io::Result<()> {
        if self.caps.colors {
            write!(self.writer, "{}", colors::BLUE)?;
        }

        if let Some(num) = line_num {
            write!(
                self.writer,
                "{:>width$} {}",
                num,
                self.box_chars.vertical(),
                width = width
            )?;
        } else {
            write!(
                self.writer,
                "{:>width$} {}",
                "",
                self.box_chars.vertical(),
                width = width
            )?;
        }

        if self.caps.colors {
            write!(self.writer, "{}", colors::RESET)?;
        }

        Ok(())
    }

    fn render_underline(
        &mut self,
        line_num: usize,
        start_line: usize,
        start_col: usize,
        end_line: usize,
        end_col: usize,
        line_len: usize,
        message: &str,
        is_primary: bool,
        gutter_width: usize,
        style: &DiagnosticStyle,
    ) -> io::Result<()> {
        // Gutter
        self.render_gutter(None, gutter_width)?;

        // Calculate underline range for this line
        let underline_start = if line_num == start_line {
            start_col.saturating_sub(1)
        } else {
            0
        };

        let underline_end = if line_num == end_line {
            end_col.saturating_sub(1)
        } else {
            line_len
        };

        // Padding to start of underline
        write!(self.writer, " {}", " ".repeat(underline_start))?;

        // Underline
        if self.caps.colors {
            let color = if is_primary {
                style.primary_color
            } else {
                style.secondary_color
            };
            write!(self.writer, "{}", color)?;
        }

        let underline_len = underline_end.saturating_sub(underline_start).max(1);
        let underline_char = if is_primary {
            self.box_chars.caret()
        } else {
            "-"
        };
        for _ in 0..underline_len {
            write!(self.writer, "{}", underline_char)?;
        }

        // Message on the last line of the span
        if line_num == end_line && !message.is_empty() {
            write!(self.writer, " {}", message)?;
        }

        if self.caps.colors {
            write!(self.writer, "{}", colors::RESET)?;
        }

        writeln!(self.writer)?;

        Ok(())
    }

    fn render_note(&mut self, note: &str, style: &DiagnosticStyle) -> io::Result<()> {
        write!(self.writer, "   {} = ", self.box_chars.vertical())?;

        if self.caps.colors {
            write!(self.writer, "{}note{}: ", style.note_color, colors::RESET)?;
        } else {
            write!(self.writer, "note: ")?;
        }

        writeln!(self.writer, "{}", note)?;
        Ok(())
    }

    fn render_help(&mut self, help: &str, style: &DiagnosticStyle) -> io::Result<()> {
        write!(self.writer, "   {} = ", self.box_chars.vertical())?;

        if self.caps.colors {
            write!(self.writer, "{}help{}: ", style.help_color, colors::RESET)?;
        } else {
            write!(self.writer, "help: ")?;
        }

        writeln!(self.writer, "{}", help)?;
        Ok(())
    }

    fn render_suggestions(&mut self, diag: &Diagnostic, style: &DiagnosticStyle) -> io::Result<()> {
        if diag.suggestions.is_empty() {
            return Ok(());
        }

        for suggestion in &diag.suggestions {
            write!(self.writer, "   {} = ", self.box_chars.vertical())?;

            if self.caps.colors {
                write!(
                    self.writer,
                    "{}suggestion{}: ",
                    style.help_color,
                    colors::RESET
                )?;
            } else {
                write!(self.writer, "suggestion: ")?;
            }

            writeln!(self.writer, "{}", suggestion.message)?;

            // Show replacement
            write!(self.writer, "   {}   ", self.box_chars.vertical())?;
            if self.caps.colors {
                write!(
                    self.writer,
                    "{}replace with{}: ",
                    colors::DIM,
                    colors::RESET
                )?;
            } else {
                write!(self.writer, "replace with: ")?;
            }

            if self.caps.colors {
                writeln!(
                    self.writer,
                    "{}`{}`{}",
                    colors::GREEN,
                    suggestion.replacement,
                    colors::RESET
                )?;
            } else {
                writeln!(self.writer, "`{}`", suggestion.replacement)?;
            }
        }

        Ok(())
    }
}

/// Semantic distance suggestion for "did you mean?"
#[derive(Debug, Clone)]
pub struct DistanceSuggestion {
    /// The suggested text
    pub text: String,
    /// Semantic distance (0.0-1.0)
    pub distance: f32,
    /// Brief description
    pub description: String,
}

/// Render semantic distance suggestions
pub fn render_distance_suggestions<W: Write>(
    writer: &mut W,
    suggestions: &[DistanceSuggestion],
    caps: &TerminalCaps,
) -> io::Result<()> {
    if suggestions.is_empty() {
        return Ok(());
    }

    let box_chars: &dyn BoxChars = if caps.unicode {
        &UNICODE_BOX
    } else {
        &ASCII_BOX
    };

    writeln!(writer, "   {}", box_chars.vertical())?;

    if caps.colors {
        write!(
            writer,
            "   {} = {}help{}: did you mean one of these?",
            box_chars.vertical(),
            colors::GREEN,
            colors::RESET
        )?;
    } else {
        write!(
            writer,
            "   {} = help: did you mean one of these?",
            box_chars.vertical()
        )?;
    }
    writeln!(writer)?;
    writeln!(writer, "   {}", box_chars.vertical())?;

    for suggestion in suggestions {
        write!(writer, "   {}        ", box_chars.vertical())?;

        if caps.colors {
            write!(writer, "{}", colors::GREEN)?;
        }

        write!(writer, "{:<24}", suggestion.text)?;

        if caps.colors {
            write!(writer, "{}", colors::RESET)?;
        }

        write!(writer, " (distance: {:.2})", suggestion.distance)?;

        if caps.colors {
            write!(
                writer,
                "{}  {}  {}",
                colors::DIM,
                box_chars.horizontal().repeat(2),
                colors::RESET
            )?;
        } else {
            write!(writer, "  {}  ", box_chars.horizontal().repeat(2))?;
        }

        writeln!(writer, "{}", suggestion.description)?;
    }

    Ok(())
}

/// Check if stderr is a terminal
fn is_terminal_stderr() -> bool {
    // Use std::io::IsTerminal when available (Rust 1.70+)
    // For now, use environment variable heuristics
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }
    if std::env::var("TERM").is_err() && std::env::var("WT_SESSION").is_err() {
        return false;
    }
    // Assume terminal if CI is not set and TERM is set
    std::env::var("CI").is_err() && std::env::var("TERM").is_ok()
}

/// Get terminal width
fn terminal_width() -> Option<usize> {
    // Try COLUMNS environment variable first
    if let Ok(cols) = std::env::var("COLUMNS")
        && let Ok(width) = cols.parse::<usize>()
    {
        return Some(width);
    }
    // Default fallback
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_caps_plain() {
        let caps = TerminalCaps::plain();
        assert!(!caps.colors);
        assert!(!caps.unicode);
        assert_eq!(caps.width, 80);
    }

    #[test]
    fn test_diagnostic_style() {
        let style = DiagnosticStyle::for_level(DiagnosticLevel::Error);
        assert_eq!(style.level_color, colors::BRIGHT_RED);

        let style = DiagnosticStyle::for_level(DiagnosticLevel::Warning);
        assert_eq!(style.level_color, colors::BRIGHT_YELLOW);
    }

    #[test]
    fn test_box_chars_unicode() {
        assert_eq!(UNICODE_BOX.vertical(), "│");
        assert_eq!(UNICODE_BOX.caret(), "^");
    }

    #[test]
    fn test_box_chars_ascii() {
        assert_eq!(ASCII_BOX.vertical(), "|");
        assert_eq!(ASCII_BOX.caret(), "^");
    }
}
