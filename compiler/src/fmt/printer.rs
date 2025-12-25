//! Pretty printer - converts Doc IR to string
//!
//! Implements the Wadler-Lindig algorithm with extensions for
//! practical code formatting.

use super::config::{EndOfLine, FormatConfig};
use super::ir::Doc;

/// Mode for printing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Try to fit on one line
    Flat,

    /// Multi-line mode
    Break,
}

/// Print command
#[derive(Debug, Clone)]
struct PrintCmd {
    indent: usize,
    mode: Mode,
    doc: Doc,
}

/// Pretty printer
pub struct Printer {
    config: FormatConfig,
}

impl Printer {
    /// Create new printer with config
    pub fn new(config: FormatConfig) -> Self {
        Printer { config }
    }

    /// Print a document to string
    pub fn print(&self, doc: &Doc) -> String {
        let mut output = String::new();
        let mut pos = 0; // Current column position
        let mut line_suffixes: Vec<Doc> = Vec::new();
        let mut cmds = vec![PrintCmd {
            indent: 0,
            mode: Mode::Break,
            doc: doc.clone(),
        }];

        while let Some(cmd) = cmds.pop() {
            match &cmd.doc {
                Doc::Empty => {}

                Doc::Text(s) => {
                    output.push_str(s);
                    pos += s.len();
                }

                Doc::Hardline => {
                    // Flush line suffixes
                    for suffix in line_suffixes.drain(..) {
                        cmds.push(PrintCmd {
                            indent: cmd.indent,
                            mode: Mode::Flat,
                            doc: suffix,
                        });
                    }

                    output.push('\n');
                    output.push_str(&self.make_indent(cmd.indent));
                    pos = cmd.indent;
                }

                Doc::Softline => {
                    if cmd.mode == Mode::Break {
                        // Flush line suffixes
                        for suffix in line_suffixes.drain(..) {
                            cmds.push(PrintCmd {
                                indent: cmd.indent,
                                mode: Mode::Flat,
                                doc: suffix,
                            });
                        }

                        output.push('\n');
                        output.push_str(&self.make_indent(cmd.indent));
                        pos = cmd.indent;
                    } else {
                        output.push(' ');
                        pos += 1;
                    }
                }

                Doc::Line => {
                    if cmd.mode == Mode::Break {
                        // Flush line suffixes
                        for suffix in line_suffixes.drain(..) {
                            cmds.push(PrintCmd {
                                indent: cmd.indent,
                                mode: Mode::Flat,
                                doc: suffix,
                            });
                        }

                        output.push('\n');
                        output.push_str(&self.make_indent(cmd.indent));
                        pos = cmd.indent;
                    }
                    // In flat mode, Line produces nothing
                }

                Doc::Concat(docs) => {
                    // Push in reverse order so they get processed in correct order
                    for d in docs.iter().rev() {
                        cmds.push(PrintCmd {
                            indent: cmd.indent,
                            mode: cmd.mode,
                            doc: d.clone(),
                        });
                    }
                }

                Doc::Indent(inner) => {
                    cmds.push(PrintCmd {
                        indent: cmd.indent + self.config.indent_width as usize,
                        mode: cmd.mode,
                        doc: (**inner).clone(),
                    });
                }

                Doc::Dedent(inner) => {
                    cmds.push(PrintCmd {
                        indent: cmd.indent.saturating_sub(self.config.indent_width as usize),
                        mode: cmd.mode,
                        doc: (**inner).clone(),
                    });
                }

                Doc::Align(inner) => {
                    cmds.push(PrintCmd {
                        indent: pos,
                        mode: cmd.mode,
                        doc: (**inner).clone(),
                    });
                }

                Doc::Group(inner) => {
                    // Try to fit in flat mode
                    let flat_width =
                        self.measure(inner, Mode::Flat, self.config.max_width as usize - pos);

                    let mode = if flat_width
                        .map(|w| pos + w <= self.config.max_width as usize)
                        .unwrap_or(false)
                    {
                        Mode::Flat
                    } else {
                        Mode::Break
                    };

                    cmds.push(PrintCmd {
                        indent: cmd.indent,
                        mode,
                        doc: (**inner).clone(),
                    });
                }

                Doc::IfBreak {
                    break_doc,
                    flat_doc,
                } => {
                    let doc = if cmd.mode == Mode::Break {
                        break_doc
                    } else {
                        flat_doc
                    };
                    cmds.push(PrintCmd {
                        indent: cmd.indent,
                        mode: cmd.mode,
                        doc: (**doc).clone(),
                    });
                }

                Doc::Fill(parts) => {
                    // Word wrapping: try to fit as many parts as possible on each line
                    self.print_fill(parts, cmd.indent, &mut output, &mut pos);
                }

                Doc::LineSuffix(inner) => {
                    // Defer until end of line
                    line_suffixes.push((**inner).clone());
                }

                Doc::BreakParent => {
                    // This is handled during measure phase
                }

                Doc::Trim => {
                    // Trim trailing whitespace from output
                    while output.ends_with(' ') || output.ends_with('\t') {
                        output.pop();
                    }
                }
            }
        }

        // Normalize line endings
        match self.config.end_of_line {
            EndOfLine::Lf => {
                output = output.replace("\r\n", "\n");
            }
            EndOfLine::Crlf => {
                // First normalize to LF, then convert to CRLF
                output = output.replace("\r\n", "\n").replace("\n", "\r\n");
            }
            EndOfLine::Cr => {
                output = output.replace("\r\n", "\r").replace("\n", "\r");
            }
        }

        // Ensure final newline
        if self.config.insert_final_newline && !output.ends_with('\n') && !output.ends_with('\r') {
            match self.config.end_of_line {
                EndOfLine::Lf => output.push('\n'),
                EndOfLine::Crlf => output.push_str("\r\n"),
                EndOfLine::Cr => output.push('\r'),
            }
        }

        // Remove trailing whitespace from lines
        let lines: Vec<&str> = output.lines().collect();
        output = lines
            .iter()
            .map(|line| line.trim_end())
            .collect::<Vec<_>>()
            .join(match self.config.end_of_line {
                EndOfLine::Lf => "\n",
                EndOfLine::Crlf => "\r\n",
                EndOfLine::Cr => "\r",
            });

        // Re-add final newline if needed
        if self.config.insert_final_newline && !output.ends_with('\n') && !output.ends_with('\r') {
            match self.config.end_of_line {
                EndOfLine::Lf => output.push('\n'),
                EndOfLine::Crlf => output.push_str("\r\n"),
                EndOfLine::Cr => output.push('\r'),
            }
        }

        output
    }

    /// Measure width of doc in flat mode
    fn measure(&self, doc: &Doc, mode: Mode, remaining: usize) -> Option<usize> {
        let mut width = 0;
        let mut stack = vec![(mode, doc)];

        while let Some((mode, doc)) = stack.pop() {
            if width > remaining {
                return None;
            }

            match doc {
                Doc::Empty => {}
                Doc::Text(s) => width += s.len(),
                Doc::Hardline => return None, // Can't fit
                Doc::Softline => {
                    if mode == Mode::Flat {
                        width += 1;
                    } else {
                        return None;
                    }
                }
                Doc::Line => {
                    if mode == Mode::Break {
                        return None;
                    }
                }
                Doc::Concat(docs) => {
                    for d in docs.iter().rev() {
                        stack.push((mode, d));
                    }
                }
                Doc::Indent(inner) | Doc::Dedent(inner) | Doc::Align(inner) => {
                    stack.push((mode, inner));
                }
                Doc::Group(inner) => {
                    stack.push((Mode::Flat, inner));
                }
                Doc::IfBreak { flat_doc, .. } => {
                    stack.push((mode, flat_doc));
                }
                Doc::Fill(parts) => {
                    for part in parts.iter().rev() {
                        stack.push((mode, part));
                    }
                }
                Doc::LineSuffix(_) => {}
                Doc::BreakParent => return None,
                Doc::Trim => {}
            }
        }

        Some(width)
    }

    /// Print fill document (word wrapping)
    fn print_fill(&self, parts: &[Doc], indent: usize, output: &mut String, pos: &mut usize) {
        for (i, part) in parts.iter().enumerate() {
            let part_width = part.flat_width().unwrap_or(0);

            // Check if we need to break before this part
            if i > 0 && *pos + part_width > self.config.max_width as usize {
                output.push('\n');
                output.push_str(&self.make_indent(indent));
                *pos = indent;
            } else if i > 0 {
                output.push(' ');
                *pos += 1;
            }

            // Print the part
            let printed = self.print_flat(part);
            output.push_str(&printed);
            *pos += printed.len();
        }
    }

    /// Print document in flat mode
    fn print_flat(&self, doc: &Doc) -> String {
        let mut output = String::new();

        fn print_flat_inner(doc: &Doc, output: &mut String) {
            match doc {
                Doc::Empty => {}
                Doc::Text(s) => output.push_str(s),
                Doc::Hardline | Doc::Line => {}
                Doc::Softline => output.push(' '),
                Doc::Concat(docs) => {
                    for d in docs {
                        print_flat_inner(d, output);
                    }
                }
                Doc::Indent(inner) | Doc::Dedent(inner) | Doc::Align(inner) | Doc::Group(inner) => {
                    print_flat_inner(inner, output);
                }
                Doc::IfBreak { flat_doc, .. } => {
                    print_flat_inner(flat_doc, output);
                }
                Doc::Fill(parts) => {
                    for (i, part) in parts.iter().enumerate() {
                        if i > 0 {
                            output.push(' ');
                        }
                        print_flat_inner(part, output);
                    }
                }
                Doc::LineSuffix(_) | Doc::BreakParent | Doc::Trim => {}
            }
        }

        print_flat_inner(doc, &mut output);
        output
    }

    /// Create indent string
    fn make_indent(&self, level: usize) -> String {
        if self.config.use_tabs {
            let tabs = level / self.config.tab_width as usize;
            let spaces = level % self.config.tab_width as usize;
            "\t".repeat(tabs) + &" ".repeat(spaces)
        } else {
            " ".repeat(level)
        }
    }
}

impl Default for Printer {
    fn default() -> Self {
        Printer::new(FormatConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_print() {
        let printer = Printer::default();
        let doc = Doc::Concat(vec![
            Doc::Text("fn ".to_string()),
            Doc::Text("foo".to_string()),
            Doc::Text("()".to_string()),
            Doc::Text(" {}".to_string()),
        ]);

        let result = printer.print(&doc);
        assert!(result.contains("fn foo() {}"));
    }

    #[test]
    fn test_indent() {
        let config = FormatConfig {
            indent_width: 4,
            ..Default::default()
        };
        let printer = Printer::new(config);

        let doc = Doc::Concat(vec![
            Doc::Text("{".to_string()),
            Doc::Indent(Box::new(Doc::Concat(vec![
                Doc::Hardline,
                Doc::Text("x".to_string()),
            ]))),
            Doc::Hardline,
            Doc::Text("}".to_string()),
        ]);

        let result = printer.print(&doc);
        assert!(result.contains("    x"));
    }

    #[test]
    fn test_group_fits() {
        let config = FormatConfig {
            max_width: 80,
            ..Default::default()
        };
        let printer = Printer::new(config);

        // Short content should stay on one line
        let doc = Doc::Group(Box::new(Doc::Concat(vec![
            Doc::Text("(".to_string()),
            Doc::Softline,
            Doc::Text("x".to_string()),
            Doc::Softline,
            Doc::Text(")".to_string()),
        ])));

        let result = printer.print(&doc);
        assert!(result.contains("( x )"));
    }

    #[test]
    fn test_group_breaks() {
        let config = FormatConfig {
            max_width: 10,
            indent_width: 2,
            ..Default::default()
        };
        let printer = Printer::new(config);

        // Long content should break
        let doc = Doc::Group(Box::new(Doc::Concat(vec![
            Doc::Text("(".to_string()),
            Doc::Indent(Box::new(Doc::Concat(vec![
                Doc::Softline,
                Doc::Text("this_is_a_long_argument".to_string()),
            ]))),
            Doc::Softline,
            Doc::Text(")".to_string()),
        ])));

        let result = printer.print(&doc);
        assert!(result.contains('\n'));
    }

    #[test]
    fn test_trailing_whitespace_removed() {
        let printer = Printer::default();
        let doc = Doc::Concat(vec![
            Doc::Text("line1   ".to_string()),
            Doc::Hardline,
            Doc::Text("line2".to_string()),
        ]);

        let result = printer.print(&doc);
        assert!(!result.contains("   \n"));
    }
}
