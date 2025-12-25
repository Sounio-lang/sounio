//! Documentation comment parser
//!
//! Parses markdown doc comments into structured documentation sections.

/// Parsed documentation sections
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DocSections {
    /// Brief summary (first paragraph)
    pub summary: Option<String>,

    /// Extended description
    pub description: Option<String>,

    /// Parameter documentation
    pub params: Vec<ParamDoc>,

    /// Return value documentation
    pub returns: Option<String>,

    /// Type parameter documentation
    pub type_params: Vec<TypeParamDoc>,

    /// Effect documentation
    pub effects: Vec<EffectDoc>,

    /// Example code blocks
    pub examples: Vec<ExampleDoc>,

    /// Panic conditions
    pub panics: Option<String>,

    /// Safety requirements (for unsafe)
    pub safety: Option<String>,

    /// Errors that can be returned
    pub errors: Option<String>,

    /// See also references
    pub see_also: Vec<CrossRef>,

    /// Version information
    pub since: Option<String>,

    /// Deprecation notice
    pub deprecated: Option<DeprecationInfo>,
}

/// Parameter documentation
#[derive(Debug, Clone, PartialEq)]
pub struct ParamDoc {
    pub name: String,
    pub description: String,
}

/// Type parameter documentation
#[derive(Debug, Clone, PartialEq)]
pub struct TypeParamDoc {
    pub name: String,
    pub description: String,
}

/// Effect documentation
#[derive(Debug, Clone, PartialEq)]
pub struct EffectDoc {
    pub effect: String,
    pub description: String,
}

/// Example code block
#[derive(Debug, Clone, PartialEq)]
pub struct ExampleDoc {
    /// Example title/description
    pub title: Option<String>,

    /// Code content
    pub code: String,

    /// Whether this should be run as a test
    pub should_test: bool,

    /// Whether test should pass or fail
    pub should_panic: bool,

    /// Whether to ignore this test
    pub ignore: bool,

    /// Whether to compile but not run
    pub no_run: bool,
}

/// Cross-reference to another item
#[derive(Debug, Clone, PartialEq)]
pub struct CrossRef {
    /// Path to referenced item
    pub path: String,

    /// Display text (if different from path)
    pub text: Option<String>,
}

/// Deprecation information
#[derive(Debug, Clone, PartialEq)]
pub struct DeprecationInfo {
    /// Version when deprecated
    pub since: Option<String>,

    /// Reason for deprecation
    pub reason: Option<String>,

    /// Suggested replacement
    pub replacement: Option<String>,
}

/// Parse a doc comment string into structured documentation
pub fn parse_doc_comment(content: &str, is_inner: bool) -> super::Documentation {
    let sections = parse_sections(content);

    super::Documentation {
        content: content.to_string(),
        sections,
        is_inner,
    }
}

/// Parse doc comment into sections
pub fn parse_sections(content: &str) -> DocSections {
    let mut sections = DocSections::default();

    // Handle empty content
    if content.trim().is_empty() {
        return sections;
    }

    // Split into paragraphs (separated by blank lines)
    let paragraphs: Vec<&str> = split_paragraphs(content);

    if paragraphs.is_empty() {
        return sections;
    }

    // First paragraph is summary
    sections.summary = Some(paragraphs[0].trim().to_string());

    // Track current section while parsing
    let mut current_section: Option<SectionType> = None;
    let mut current_content = String::new();
    let mut in_code_block = false;
    let mut code_block_lang = String::new();
    let mut code_block_attrs = Vec::new();
    let mut description_parts: Vec<String> = Vec::new();
    let mut past_summary = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Track code block boundaries
        if trimmed.starts_with("```") {
            if in_code_block {
                // End code block
                if current_section == Some(SectionType::Examples)
                    || (current_section.is_none() && code_block_lang == "d")
                {
                    let example =
                        parse_example(&code_block_lang, &code_block_attrs, &current_content);
                    sections.examples.push(example);
                }
                current_content.clear();
                in_code_block = false;
                code_block_lang.clear();
                code_block_attrs.clear();
            } else {
                // Start code block
                in_code_block = true;
                let rest = trimmed.trim_start_matches("```");
                let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();
                code_block_lang = parts.first().unwrap_or(&"").to_string();
                code_block_attrs = parts.iter().skip(1).map(|s| s.to_string()).collect();
                current_content.clear();
            }
            continue;
        }

        if in_code_block {
            if !current_content.is_empty() {
                current_content.push('\n');
            }
            current_content.push_str(line);
            continue;
        }

        // Check for section headers (# Header)
        if trimmed.starts_with("# ") {
            // Save previous section
            save_section(&mut sections, current_section, &current_content);

            // Parse new section header
            let header = trimmed[2..].trim().to_lowercase();
            current_section = match header.as_str() {
                "arguments" | "parameters" | "params" => Some(SectionType::Params),
                "returns" | "return" => Some(SectionType::Returns),
                "type parameters" | "generics" => Some(SectionType::TypeParams),
                "effects" => Some(SectionType::Effects),
                "examples" | "example" => Some(SectionType::Examples),
                "panics" | "panic" => Some(SectionType::Panics),
                "safety" => Some(SectionType::Safety),
                "errors" => Some(SectionType::Errors),
                "see also" => Some(SectionType::SeeAlso),
                _ => None,
            };
            current_content.clear();
            past_summary = true;
            continue;
        }

        // Check for attribute-style documentation (@param, @returns, etc.)
        if trimmed.starts_with('@') {
            parse_attribute(&mut sections, trimmed);
            past_summary = true;
            continue;
        }

        // Check for empty line (paragraph separator)
        if trimmed.is_empty() {
            if !current_content.is_empty() && current_section.is_none() {
                description_parts.push(current_content.trim().to_string());
                current_content.clear();
            }
            past_summary = true;
            continue;
        }

        // Accumulate content
        if !current_content.is_empty() {
            current_content.push('\n');
        }
        current_content.push_str(trimmed);

        if past_summary && current_section.is_none() && !trimmed.is_empty() {
            // This is part of the description
        }
    }

    // Save final section
    save_section(&mut sections, current_section, &current_content);

    // Handle remaining description content
    if !current_content.is_empty() && current_section.is_none() {
        description_parts.push(current_content.trim().to_string());
    }

    // Build description from non-first paragraphs
    if !description_parts.is_empty() {
        // Skip the summary (first paragraph)
        let desc = description_parts
            .into_iter()
            .filter(|s| !s.is_empty() && Some(s.as_str()) != sections.summary.as_deref())
            .collect::<Vec<_>>()
            .join("\n\n");
        if !desc.is_empty() {
            sections.description = Some(desc);
        }
    }

    sections
}

/// Section types for parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SectionType {
    Params,
    Returns,
    TypeParams,
    Effects,
    Examples,
    Panics,
    Safety,
    Errors,
    SeeAlso,
}

/// Split content into paragraphs
fn split_paragraphs(content: &str) -> Vec<&str> {
    // Simple approach: split on double newlines
    // This handles most common cases correctly
    content
        .split("\n\n")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parse an attribute-style doc tag
fn parse_attribute(sections: &mut DocSections, line: &str) {
    let line = line.trim();

    // @param name description
    if line.starts_with("@param") {
        let rest = line.strip_prefix("@param").unwrap_or("").trim();
        if let Some((name, desc)) = rest.split_once(char::is_whitespace) {
            sections.params.push(ParamDoc {
                name: name.trim().to_string(),
                description: desc.trim().to_string(),
            });
        }
        return;
    }

    // @returns description
    if line.starts_with("@return") {
        let rest = line
            .strip_prefix("@returns")
            .or_else(|| line.strip_prefix("@return"))
            .unwrap_or("")
            .trim();
        if !rest.is_empty() {
            sections.returns = Some(rest.to_string());
        }
        return;
    }

    // @typeparam name description
    if line.starts_with("@typeparam") {
        let rest = line.strip_prefix("@typeparam").unwrap_or("").trim();
        if let Some((name, desc)) = rest.split_once(char::is_whitespace) {
            sections.type_params.push(TypeParamDoc {
                name: name.trim().to_string(),
                description: desc.trim().to_string(),
            });
        }
        return;
    }

    // @effect name description
    if line.starts_with("@effect") {
        let rest = line.strip_prefix("@effect").unwrap_or("").trim();
        if let Some((effect, desc)) = rest.split_once(char::is_whitespace) {
            sections.effects.push(EffectDoc {
                effect: effect.trim().to_string(),
                description: desc.trim().to_string(),
            });
        }
        return;
    }

    // @since version
    if line.starts_with("@since") {
        let rest = line.strip_prefix("@since").unwrap_or("").trim();
        if !rest.is_empty() {
            sections.since = Some(rest.to_string());
        }
        return;
    }

    // @deprecated [reason]
    if line.starts_with("@deprecated") {
        let rest = line.strip_prefix("@deprecated").unwrap_or("").trim();
        sections.deprecated = Some(DeprecationInfo {
            since: None,
            reason: if rest.is_empty() {
                None
            } else {
                Some(rest.to_string())
            },
            replacement: None,
        });
        return;
    }

    // @see path
    if line.starts_with("@see") {
        let rest = line.strip_prefix("@see").unwrap_or("").trim();
        if !rest.is_empty() {
            sections.see_also.push(CrossRef {
                path: rest.to_string(),
                text: None,
            });
        }
        return;
    }

    // @example title
    if line.starts_with("@example") {
        // Just marks the beginning of an example section
    }
}

/// Parse an example code block
fn parse_example(lang: &str, attrs: &[String], code: &str) -> ExampleDoc {
    let attrs_set: std::collections::HashSet<&str> = attrs.iter().map(|s| s.as_str()).collect();

    ExampleDoc {
        title: None,
        code: code.trim().to_string(),
        should_test: lang == "d" && !attrs_set.contains("no_run") && !attrs_set.contains("ignore"),
        should_panic: attrs_set.contains("should_panic"),
        ignore: attrs_set.contains("ignore"),
        no_run: attrs_set.contains("no_run"),
    }
}

/// Save parsed content to appropriate section
fn save_section(sections: &mut DocSections, section: Option<SectionType>, content: &str) {
    let content = content.trim();
    if content.is_empty() {
        return;
    }

    match section {
        Some(SectionType::Params) => parse_list_items(content, |name, desc| {
            sections.params.push(ParamDoc {
                name,
                description: desc,
            });
        }),
        Some(SectionType::Returns) => {
            sections.returns = Some(content.to_string());
        }
        Some(SectionType::TypeParams) => parse_list_items(content, |name, desc| {
            sections.type_params.push(TypeParamDoc {
                name,
                description: desc,
            });
        }),
        Some(SectionType::Effects) => parse_list_items(content, |effect, desc| {
            sections.effects.push(EffectDoc {
                effect,
                description: desc,
            });
        }),
        Some(SectionType::Examples) => {
            // Examples are handled via code blocks
        }
        Some(SectionType::Panics) => {
            sections.panics = Some(content.to_string());
        }
        Some(SectionType::Safety) => {
            sections.safety = Some(content.to_string());
        }
        Some(SectionType::Errors) => {
            sections.errors = Some(content.to_string());
        }
        Some(SectionType::SeeAlso) => {
            parse_see_also(sections, content);
        }
        None => {
            // Content goes to description
        }
    }
}

/// Parse list items (- `name`: description)
fn parse_list_items<F>(content: &str, mut add_item: F)
where
    F: FnMut(String, String),
{
    for line in content.lines() {
        let trimmed = line.trim();

        // Handle list items: - `name`: description or * `name`: description
        if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
            let rest = &trimmed[2..];

            // Try to extract name and description
            if let Some((name_part, desc)) = rest.split_once(':') {
                let name = name_part.trim().trim_matches('`').to_string();
                let description = desc.trim().to_string();
                add_item(name, description);
            } else {
                // No colon, treat the whole thing as name
                let name = rest.trim().trim_matches('`').to_string();
                add_item(name, String::new());
            }
        }
    }
}

/// Parse see also references
fn parse_see_also(sections: &mut DocSections, content: &str) {
    // Match patterns like [text](path) or [`path`]
    for line in content.lines() {
        let trimmed = line.trim();

        // List item format
        let item = if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
            &trimmed[2..]
        } else {
            trimmed
        };

        // [text](path) format
        if item.starts_with('[') {
            if let Some(close_bracket) = item.find(']') {
                let text = &item[1..close_bracket];

                if item[close_bracket..].starts_with("](")
                    && let Some(close_paren) = item[close_bracket..].find(')')
                {
                    let path = &item[close_bracket + 2..close_bracket + close_paren];
                    sections.see_also.push(CrossRef {
                        path: path.to_string(),
                        text: Some(text.to_string()),
                    });
                    continue;
                }

                // [`path`] format - path is also the text
                let path = text.trim_matches('`');
                sections.see_also.push(CrossRef {
                    path: path.to_string(),
                    text: Some(text.to_string()),
                });
            }
        } else if !item.is_empty() {
            // Plain text reference
            sections.see_also.push(CrossRef {
                path: item.to_string(),
                text: None,
            });
        }
    }
}

/// Resolve cross-references in markdown content
pub fn resolve_cross_refs<F>(content: &str, resolver: F) -> String
where
    F: Fn(&str) -> Option<String>,
{
    let mut result = String::with_capacity(content.len());
    let mut chars = content.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '[' && chars.peek() == Some(&'`') {
            // Potential [`path`] reference
            chars.next(); // consume `
            let mut path = String::new();

            while let Some(&next) = chars.peek() {
                if next == '`' {
                    break;
                }
                path.push(chars.next().unwrap());
            }

            if chars.next() == Some('`') && chars.next() == Some(']') {
                // Check if already has a link
                if chars.peek() != Some(&'(') {
                    if let Some(url) = resolver(&path) {
                        result.push_str(&format!("[`{}`]({})", path, url));
                    } else {
                        result.push_str(&format!("`{}`", path));
                    }
                    continue;
                } else {
                    // Already has link, keep as-is
                    result.push_str(&format!("[`{}`]", path));
                }
            } else {
                // Not a valid reference, output as-is
                result.push('[');
                result.push('`');
                result.push_str(&path);
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Parse block doc comment (/** ... */)
pub fn parse_block_doc(content: &str) -> String {
    // Remove /** and */
    let content = content.trim();
    let content = content.strip_prefix("/**").unwrap_or(content);
    let content = content.strip_suffix("*/").unwrap_or(content);

    // Process lines, removing leading * if present
    content
        .lines()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with('*') && !trimmed.starts_with("*/") {
                trimmed[1..].trim()
            } else {
                trimmed
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Parse inner block doc comment (/*! ... */)
pub fn parse_block_doc_inner(content: &str) -> String {
    let content = content.trim();
    let content = content.strip_prefix("/*!").unwrap_or(content);
    let content = content.strip_suffix("*/").unwrap_or(content);

    content
        .lines()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with('*') && !trimmed.starts_with("*/") {
                trimmed[1..].trim()
            } else {
                trimmed
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_doc() {
        let content = "This is a summary.";
        let sections = parse_sections(content);
        assert_eq!(sections.summary, Some("This is a summary.".to_string()));
    }

    #[test]
    fn test_parse_with_description() {
        let content = "Summary line.\n\nThis is the description.\nIt spans multiple lines.";
        let sections = parse_sections(content);
        assert_eq!(sections.summary, Some("Summary line.".to_string()));
        assert!(sections.description.is_some());
    }

    #[test]
    fn test_parse_params_section() {
        let content =
            "Summary.\n\n# Parameters\n\n- `x`: The x coordinate\n- `y`: The y coordinate";
        let sections = parse_sections(content);
        assert_eq!(sections.params.len(), 2);
        assert_eq!(sections.params[0].name, "x");
        assert_eq!(sections.params[0].description, "The x coordinate");
    }

    #[test]
    fn test_parse_attribute_style() {
        let content = "Summary.\n\n@param x The x value\n@param y The y value\n@returns The sum";
        let sections = parse_sections(content);
        assert_eq!(sections.params.len(), 2);
        assert_eq!(sections.returns, Some("The sum".to_string()));
    }

    #[test]
    fn test_parse_example() {
        let content = "Summary.\n\n# Examples\n\n```d\nlet x = 1\n```";
        let sections = parse_sections(content);
        assert_eq!(sections.examples.len(), 1);
        assert_eq!(sections.examples[0].code, "let x = 1");
        assert!(sections.examples[0].should_test);
    }

    #[test]
    fn test_parse_example_no_run() {
        let content = "Summary.\n\n```d,no_run\nlet x = 1\n```";
        let sections = parse_sections(content);
        assert_eq!(sections.examples.len(), 1);
        assert!(!sections.examples[0].should_test);
        assert!(sections.examples[0].no_run);
    }

    #[test]
    fn test_parse_block_doc() {
        let content = "/**\n * Line 1\n * Line 2\n */";
        let result = parse_block_doc(content);
        assert_eq!(result, "Line 1\nLine 2");
    }

    #[test]
    fn test_parse_deprecated() {
        let content = "Summary.\n\n@deprecated Use new_func instead";
        let sections = parse_sections(content);
        assert!(sections.deprecated.is_some());
        assert_eq!(
            sections.deprecated.unwrap().reason,
            Some("Use new_func instead".to_string())
        );
    }

    #[test]
    fn test_parse_see_also() {
        let content = "Summary.\n\n# See Also\n\n- [`Vec`]\n- [HashMap](std::collections::HashMap)";
        let sections = parse_sections(content);
        assert_eq!(sections.see_also.len(), 2);
    }
}
