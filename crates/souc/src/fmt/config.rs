//! Formatter configuration
//!
//! Supports loading from .dfmt.toml, .dfmt.json, or d.toml [format] section.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatConfig {
    /// Maximum line width
    #[serde(default = "default_max_width")]
    pub max_width: u32,

    /// Indent width (spaces)
    #[serde(default = "default_indent_width")]
    pub indent_width: u32,

    /// Use tabs instead of spaces
    #[serde(default)]
    pub use_tabs: bool,

    /// Tab width (for display)
    #[serde(default = "default_tab_width")]
    pub tab_width: u32,

    /// End of line style
    #[serde(default)]
    pub end_of_line: EndOfLine,

    /// Insert final newline
    #[serde(default = "default_true")]
    pub insert_final_newline: bool,

    /// Trailing commas
    #[serde(default)]
    pub trailing_comma: TrailingComma,

    /// Brace style
    #[serde(default)]
    pub brace_style: BraceStyle,

    /// Single expression function bodies on same line
    #[serde(default = "default_true")]
    pub single_line_fn: bool,

    /// Group imports by category
    #[serde(default = "default_true")]
    pub group_imports: bool,

    /// Sort imports alphabetically
    #[serde(default = "default_true")]
    pub sort_imports: bool,

    /// Blank lines between items
    #[serde(default = "default_blank_lines")]
    pub blank_lines_between_items: u32,

    /// Format string literals
    #[serde(default)]
    pub format_strings: bool,

    /// Format comments
    #[serde(default = "default_true")]
    pub format_comments: bool,

    /// Wrap comments at max_width
    #[serde(default)]
    pub wrap_comments: bool,

    /// Normalize doc comments (/// style)
    #[serde(default = "default_true")]
    pub normalize_doc_comments: bool,

    /// Space after colon in struct fields
    #[serde(default = "default_true")]
    pub space_after_colon: bool,

    /// Space before colon in struct fields
    #[serde(default)]
    pub space_before_colon: bool,

    /// Spaces inside brackets
    #[serde(default)]
    pub spaces_inside_brackets: bool,

    /// Spaces inside parens
    #[serde(default)]
    pub spaces_inside_parens: bool,

    /// Spaces around operators
    #[serde(default = "default_true")]
    pub spaces_around_operators: bool,

    /// Maximum number of blank lines to preserve
    #[serde(default = "default_max_blank_lines")]
    pub max_blank_lines: u32,

    /// Chain method calls on new lines
    #[serde(default)]
    pub chain_method_break: ChainBreak,

    /// Array element layout
    #[serde(default)]
    pub array_layout: ArrayLayout,
}

fn default_max_width() -> u32 {
    100
}
fn default_indent_width() -> u32 {
    4
}
fn default_tab_width() -> u32 {
    4
}
fn default_true() -> bool {
    true
}
fn default_blank_lines() -> u32 {
    1
}
fn default_max_blank_lines() -> u32 {
    2
}

/// End of line style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EndOfLine {
    /// Unix style (LF)
    #[default]
    Lf,
    /// Windows style (CRLF)
    Crlf,
    /// Old Mac style (CR)
    Cr,
}

/// Trailing comma style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TrailingComma {
    /// Never use trailing commas
    Never,

    /// Use trailing commas in multi-line contexts
    #[default]
    Multiline,

    /// Always use trailing commas
    Always,
}

/// Brace placement style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BraceStyle {
    /// Opening brace on same line (K&R style)
    #[default]
    SameLine,

    /// Opening brace on new line (Allman style)
    NewLine,

    /// Same line for single expressions, new line otherwise
    PreferSameLine,
}

/// Method chain breaking style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChainBreak {
    /// Never break chains
    Never,

    /// Break chains when they exceed line width
    #[default]
    Auto,

    /// Always break chains
    Always,
}

/// Array element layout style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ArrayLayout {
    /// Auto-detect based on content
    #[default]
    Auto,

    /// Always single line
    SingleLine,

    /// Always multi-line
    MultiLine,

    /// One element per line
    OnePerLine,
}

impl Default for FormatConfig {
    fn default() -> Self {
        FormatConfig {
            max_width: 100,
            indent_width: 4,
            use_tabs: false,
            tab_width: 4,
            end_of_line: EndOfLine::Lf,
            insert_final_newline: true,
            trailing_comma: TrailingComma::Multiline,
            brace_style: BraceStyle::SameLine,
            single_line_fn: true,
            group_imports: true,
            sort_imports: true,
            blank_lines_between_items: 1,
            format_strings: false,
            format_comments: true,
            wrap_comments: false,
            normalize_doc_comments: true,
            space_after_colon: true,
            space_before_colon: false,
            spaces_inside_brackets: false,
            spaces_inside_parens: false,
            spaces_around_operators: true,
            max_blank_lines: 2,
            chain_method_break: ChainBreak::Auto,
            array_layout: ArrayLayout::Auto,
        }
    }
}

/// Configuration error
#[derive(Debug)]
pub enum ConfigError {
    Io(String),
    Parse(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(s) => write!(f, "IO error: {}", s),
            ConfigError::Parse(s) => write!(f, "Parse error: {}", s),
        }
    }
}

impl std::error::Error for ConfigError {}

impl FormatConfig {
    /// Load from file
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| ConfigError::Io(e.to_string()))?;

        if path.extension().map(|e| e == "toml").unwrap_or(false) {
            toml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))
        } else if path.extension().map(|e| e == "json").unwrap_or(false) {
            serde_json::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))
        } else {
            Err(ConfigError::Parse("Unknown config file format".to_string()))
        }
    }

    /// Find config file in directory hierarchy
    pub fn find_config(start: &Path) -> Option<Self> {
        let mut dir = if start.is_file() {
            start.parent()?.to_path_buf()
        } else {
            start.to_path_buf()
        };

        loop {
            // Check for sounio.toml [format] section
            let sounio_toml = dir.join("sounio.toml");
            if sounio_toml.exists()
                && let Ok(config) = Self::from_sounio_toml(&sounio_toml)
            {
                return Some(config);
            }

            // Check for d.toml [format] section (legacy)
            let d_toml = dir.join("d.toml");
            if d_toml.exists()
                && let Ok(config) = Self::from_d_toml(&d_toml)
            {
                return Some(config);
            }

            // Check for .souniofmt.toml
            let souniofmt = dir.join(".souniofmt.toml");
            if souniofmt.exists()
                && let Ok(config) = Self::from_file(&souniofmt)
            {
                return Some(config);
            }

            // Check for souniofmt.toml (without dot)
            let souniofmt_nodot = dir.join("souniofmt.toml");
            if souniofmt_nodot.exists()
                && let Ok(config) = Self::from_file(&souniofmt_nodot)
            {
                return Some(config);
            }

            // Check for .dfmt.toml (legacy)
            let dfmt = dir.join(".dfmt.toml");
            if dfmt.exists()
                && let Ok(config) = Self::from_file(&dfmt)
            {
                return Some(config);
            }

            // Check for dfmt.toml (without dot, legacy)
            let dfmt_nodot = dir.join("dfmt.toml");
            if dfmt_nodot.exists()
                && let Ok(config) = Self::from_file(&dfmt_nodot)
            {
                return Some(config);
            }

            // Check for .souniofmt.json
            let souniofmt_json = dir.join(".souniofmt.json");
            if souniofmt_json.exists()
                && let Ok(config) = Self::from_file(&souniofmt_json)
            {
                return Some(config);
            }

            // Check for .dfmt.json (legacy)
            let dfmt_json = dir.join(".dfmt.json");
            if dfmt_json.exists()
                && let Ok(config) = Self::from_file(&dfmt_json)
            {
                return Some(config);
            }

            if !dir.pop() {
                break;
            }
        }

        None
    }

    /// Load from sounio.toml [format] section
    fn from_sounio_toml(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| ConfigError::Io(e.to_string()))?;

        let manifest: toml::Value =
            toml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))?;

        if let Some(format) = manifest.get("format") {
            let config: FormatConfig = format
                .clone()
                .try_into()
                .map_err(|e: toml::de::Error| ConfigError::Parse(e.to_string()))?;
            Ok(config)
        } else {
            Ok(FormatConfig::default())
        }
    }

    /// Load from d.toml [format] section (legacy)
    fn from_d_toml(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| ConfigError::Io(e.to_string()))?;

        let manifest: toml::Value =
            toml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))?;

        if let Some(format) = manifest.get("format") {
            let config: FormatConfig = format
                .clone()
                .try_into()
                .map_err(|e: toml::de::Error| ConfigError::Parse(e.to_string()))?;
            Ok(config)
        } else {
            Ok(FormatConfig::default())
        }
    }

    /// Save to file
    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        let content = if path.extension().map(|e| e == "json").unwrap_or(false) {
            serde_json::to_string_pretty(self).map_err(|e| ConfigError::Parse(e.to_string()))?
        } else {
            toml::to_string_pretty(self).map_err(|e| ConfigError::Parse(e.to_string()))?
        };

        std::fs::write(path, content).map_err(|e| ConfigError::Io(e.to_string()))
    }

    /// Merge with another config (other takes precedence)
    pub fn merge(&mut self, other: &FormatConfig) {
        // For now, just replace
        *self = other.clone();
    }

    /// Create config for checking (strict mode)
    pub fn strict() -> Self {
        FormatConfig {
            max_width: 100,
            indent_width: 4,
            use_tabs: false,
            trailing_comma: TrailingComma::Always,
            ..Default::default()
        }
    }

    /// Create config for minimal changes
    pub fn minimal() -> Self {
        FormatConfig {
            format_comments: false,
            normalize_doc_comments: false,
            sort_imports: false,
            group_imports: false,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FormatConfig::default();
        assert_eq!(config.max_width, 100);
        assert_eq!(config.indent_width, 4);
        assert!(!config.use_tabs);
        assert!(config.insert_final_newline);
    }

    #[test]
    fn test_serialize_config() {
        let config = FormatConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        assert!(toml_str.contains("max_width"));
        assert!(toml_str.contains("indent_width"));
    }

    #[test]
    fn test_deserialize_config() {
        let toml_str = r#"
max_width = 80
indent_width = 2
use_tabs = true
trailing_comma = "always"
"#;
        let config: FormatConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.max_width, 80);
        assert_eq!(config.indent_width, 2);
        assert!(config.use_tabs);
        assert_eq!(config.trailing_comma, TrailingComma::Always);
    }
}
