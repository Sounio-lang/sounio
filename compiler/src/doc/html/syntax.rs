//! Syntax highlighting for D code

use std::collections::HashSet;

/// Syntax highlighter for D
pub struct SyntaxHighlighter {
    keywords: HashSet<&'static str>,
    types: HashSet<&'static str>,
    effects: HashSet<&'static str>,
}

impl SyntaxHighlighter {
    /// Create a new syntax highlighter
    pub fn new() -> Self {
        let keywords: HashSet<&'static str> = [
            "fn",
            "let",
            "var",
            "const",
            "mut",
            "pub",
            "use",
            "mod",
            "struct",
            "enum",
            "trait",
            "impl",
            "type",
            "where",
            "if",
            "else",
            "match",
            "for",
            "while",
            "loop",
            "return",
            "break",
            "continue",
            "in",
            "as",
            "with",
            "unsafe",
            "async",
            "await",
            "move",
            "own",
            "ref",
            "linear",
            "affine",
            "kernel",
            "true",
            "false",
            "self",
            "Self",
            "super",
            "crate",
            "effect",
            "handler",
            "handle",
            "perform",
            "resume",
            "sample",
            "observe",
            "infer",
            "proof",
            "invariant",
            "requires",
            "ensures",
            "assert",
            "assume",
            "extern",
            "import",
            "export",
            "module",
            "device",
            "shared",
            "gpu",
            "spawn",
            "copy",
            "drop",
        ]
        .into_iter()
        .collect();

        let types: HashSet<&'static str> = [
            "int", "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "f32",
            "f64", "bool", "char", "unit", "String", "str", "Vec", "Option", "Result", "Box", "Rc",
            "Arc", "Cell", "RefCell", "HashMap", "HashSet", "BTreeMap", "BTreeSet", "Mutex",
            "RwLock", "Channel", "Sender", "Receiver", "Deque", "Iterator",
        ]
        .into_iter()
        .collect();

        let effects: HashSet<&'static str> =
            ["IO", "Mut", "Alloc", "Panic", "Async", "GPU", "Prob", "Div"]
                .into_iter()
                .collect();

        Self {
            keywords,
            types,
            effects,
        }
    }

    /// Highlight D source code
    pub fn highlight(&self, source: &str) -> String {
        let mut result = String::new();
        let mut chars = source.chars().peekable();

        while let Some(c) = chars.next() {
            match c {
                // Comments
                '/' => {
                    if chars.peek() == Some(&'/') {
                        chars.next();
                        result.push_str("<span class=\"comment\">//");

                        // Check for doc comment
                        if chars.peek() == Some(&'/') || chars.peek() == Some(&'!') {
                            result.clear();
                            let doc_char = chars.next().unwrap();
                            result.push_str(&format!("<span class=\"doc-comment\">//{}", doc_char));
                        }

                        // Read until end of line
                        while let Some(&next) = chars.peek() {
                            if next == '\n' {
                                break;
                            }
                            result.push(self.escape_html_char(chars.next().unwrap()));
                        }
                        result.push_str("</span>");
                    } else if chars.peek() == Some(&'*') {
                        chars.next();
                        let is_doc = chars.peek() == Some(&'*') || chars.peek() == Some(&'!');

                        if is_doc {
                            result.push_str("<span class=\"doc-comment\">/*");
                        } else {
                            result.push_str("<span class=\"comment\">/*");
                        }
                        if let Some(&next) = chars.peek() {
                            result.push(self.escape_html_char(chars.next().unwrap()));
                        }

                        // Read until */
                        let mut prev = ' ';
                        for next in chars.by_ref() {
                            result.push(self.escape_html_char(next));
                            if prev == '*' && next == '/' {
                                break;
                            }
                            prev = next;
                        }
                        result.push_str("</span>");
                    } else {
                        result.push(c);
                    }
                }

                // Strings
                '"' => {
                    result.push_str("<span class=\"string\">\"");
                    while let Some(next) = chars.next() {
                        result.push(self.escape_html_char(next));
                        if next == '"' {
                            break;
                        }
                        if next == '\\'
                            && let Some(escaped) = chars.next()
                        {
                            result.push(self.escape_html_char(escaped));
                        }
                    }
                    result.push_str("</span>");
                }

                // Characters
                '\'' => {
                    result.push_str("<span class=\"char\">'");
                    while let Some(next) = chars.next() {
                        result.push(self.escape_html_char(next));
                        if next == '\'' {
                            break;
                        }
                        if next == '\\'
                            && let Some(escaped) = chars.next()
                        {
                            result.push(self.escape_html_char(escaped));
                        }
                    }
                    result.push_str("</span>");
                }

                // Numbers
                c if c.is_ascii_digit() => {
                    result.push_str("<span class=\"number\">");
                    result.push(c);
                    while let Some(&next) = chars.peek() {
                        if next.is_ascii_alphanumeric() || next == '.' || next == '_' {
                            result.push(chars.next().unwrap());
                        } else {
                            break;
                        }
                    }
                    result.push_str("</span>");
                }

                // Identifiers
                c if c.is_ascii_alphabetic() || c == '_' => {
                    let mut ident = String::new();
                    ident.push(c);
                    while let Some(&next) = chars.peek() {
                        if next.is_ascii_alphanumeric() || next == '_' {
                            ident.push(chars.next().unwrap());
                        } else {
                            break;
                        }
                    }

                    // Classify identifier
                    if self.keywords.contains(ident.as_str()) {
                        result.push_str(&format!("<span class=\"keyword\">{}</span>", ident));
                    } else if self.types.contains(ident.as_str()) {
                        result.push_str(&format!("<span class=\"type\">{}</span>", ident));
                    } else if self.effects.contains(ident.as_str()) {
                        result.push_str(&format!("<span class=\"effect\">{}</span>", ident));
                    } else if ident.chars().next().unwrap().is_uppercase() {
                        result.push_str(&format!("<span class=\"type\">{}</span>", ident));
                    } else {
                        result.push_str(&ident);
                    }
                }

                // HTML special chars
                '<' => result.push_str("&lt;"),
                '>' => result.push_str("&gt;"),
                '&' => result.push_str("&amp;"),

                // Everything else
                _ => result.push(c),
            }
        }

        result
    }

    /// Highlight a single line (for inline code)
    pub fn highlight_inline(&self, code: &str) -> String {
        self.highlight(code)
    }

    /// Escape HTML character
    fn escape_html_char(&self, c: char) -> char {
        // For individual chars in strings, just return as-is
        // HTML escaping is handled in the string building
        c
    }
}

impl Default for SyntaxHighlighter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlight_keyword() {
        let highlighter = SyntaxHighlighter::new();
        let result = highlighter.highlight("let x = 1");
        assert!(result.contains("<span class=\"keyword\">let</span>"));
    }

    #[test]
    fn test_highlight_string() {
        let highlighter = SyntaxHighlighter::new();
        let result = highlighter.highlight("\"hello\"");
        assert!(result.contains("<span class=\"string\">"));
    }

    #[test]
    fn test_highlight_number() {
        let highlighter = SyntaxHighlighter::new();
        let result = highlighter.highlight("42");
        assert!(result.contains("<span class=\"number\">42</span>"));
    }

    #[test]
    fn test_highlight_type() {
        let highlighter = SyntaxHighlighter::new();
        let result = highlighter.highlight("Vec<int>");
        assert!(result.contains("<span class=\"type\">Vec</span>"));
    }

    #[test]
    fn test_highlight_comment() {
        let highlighter = SyntaxHighlighter::new();
        let result = highlighter.highlight("// comment");
        assert!(result.contains("<span class=\"comment\">"));
    }

    #[test]
    fn test_highlight_doc_comment() {
        let highlighter = SyntaxHighlighter::new();
        let result = highlighter.highlight("/// doc");
        assert!(result.contains("<span class=\"doc-comment\">"));
    }
}
