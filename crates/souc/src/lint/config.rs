//! Lint configuration
//!
//! Supports loading from d.toml [lint] section or .dlint.toml

use super::LintLevel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Lint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintConfig {
    /// Level overrides by lint ID
    #[serde(default)]
    pub levels: HashMap<String, LintLevel>,

    /// Enabled lint groups
    #[serde(default)]
    pub groups: Vec<String>,

    /// Disabled lint groups
    #[serde(default)]
    pub disabled_groups: Vec<String>,

    /// Maximum cyclomatic complexity threshold
    #[serde(default = "default_max_complexity")]
    pub max_complexity: usize,

    /// Maximum function length (lines)
    #[serde(default = "default_max_function_length")]
    pub max_function_length: usize,

    /// Maximum nesting depth
    #[serde(default = "default_max_nesting")]
    pub max_nesting: usize,

    /// Maximum number of parameters
    #[serde(default = "default_max_params")]
    pub max_params: usize,
}

fn default_max_complexity() -> usize {
    15
}
fn default_max_function_length() -> usize {
    100
}
fn default_max_nesting() -> usize {
    5
}
fn default_max_params() -> usize {
    7
}

impl Default for LintConfig {
    fn default() -> Self {
        LintConfig {
            levels: HashMap::new(),
            groups: Vec::new(),
            disabled_groups: Vec::new(),
            max_complexity: default_max_complexity(),
            max_function_length: default_max_function_length(),
            max_nesting: default_max_nesting(),
            max_params: default_max_params(),
        }
    }
}

impl LintConfig {
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
            // Check for d.toml [lint] section
            let d_toml = dir.join("d.toml");
            if d_toml.exists()
                && let Ok(config) = Self::from_d_toml(&d_toml)
            {
                return Some(config);
            }

            // Check for .dlint.toml
            let dlint = dir.join(".dlint.toml");
            if dlint.exists()
                && let Ok(config) = Self::from_file(&dlint)
            {
                return Some(config);
            }

            // Check for dlint.toml (without dot)
            let dlint_nodot = dir.join("dlint.toml");
            if dlint_nodot.exists()
                && let Ok(config) = Self::from_file(&dlint_nodot)
            {
                return Some(config);
            }

            if !dir.pop() {
                break;
            }
        }

        None
    }

    /// Load from d.toml [lint] section
    fn from_d_toml(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(|e| ConfigError::Io(e.to_string()))?;

        let manifest: toml::Value =
            toml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))?;

        if let Some(lint) = manifest.get("lint") {
            let config: LintConfig = lint
                .clone()
                .try_into()
                .map_err(|e: toml::de::Error| ConfigError::Parse(e.to_string()))?;
            Ok(config)
        } else {
            Ok(LintConfig::default())
        }
    }

    /// Set lint level
    pub fn set_level(&mut self, lint_id: &str, level: LintLevel) {
        self.levels.insert(lint_id.to_string(), level);
    }

    /// Get lint level (returns None if not overridden)
    pub fn get_level(&self, lint_id: &str) -> Option<LintLevel> {
        self.levels.get(lint_id).copied()
    }

    /// Enable a lint group
    pub fn enable_group(&mut self, group: &str) {
        if !self.groups.contains(&group.to_string()) {
            self.groups.push(group.to_string());
        }
        self.disabled_groups.retain(|g| g != group);
    }

    /// Disable a lint group
    pub fn disable_group(&mut self, group: &str) {
        if !self.disabled_groups.contains(&group.to_string()) {
            self.disabled_groups.push(group.to_string());
        }
        self.groups.retain(|g| g != group);
    }

    /// Check if a group is enabled
    pub fn is_group_enabled(&self, group: &str) -> bool {
        !self.disabled_groups.contains(&group.to_string())
            && (self.groups.is_empty() || self.groups.contains(&group.to_string()))
    }

    /// Merge with another config (other takes precedence)
    pub fn merge(&mut self, other: &LintConfig) {
        for (id, level) in &other.levels {
            self.levels.insert(id.clone(), *level);
        }

        for group in &other.groups {
            self.enable_group(group);
        }

        for group in &other.disabled_groups {
            self.disable_group(group);
        }

        self.max_complexity = other.max_complexity;
        self.max_function_length = other.max_function_length;
        self.max_nesting = other.max_nesting;
        self.max_params = other.max_params;
    }

    /// Create strict config
    pub fn strict() -> Self {
        let mut config = LintConfig::default();
        config.set_level("missing_docs", LintLevel::Deny);
        config.set_level("naming_convention", LintLevel::Deny);
        config.max_complexity = 10;
        config.max_function_length = 50;
        config.max_nesting = 4;
        config.max_params = 5;
        config
    }

    /// Create relaxed config
    pub fn relaxed() -> Self {
        let mut config = LintConfig::default();
        config.set_level("unused_variable", LintLevel::Allow);
        config.set_level("naming_convention", LintLevel::Allow);
        config.max_complexity = 25;
        config.max_function_length = 200;
        config.max_nesting = 8;
        config.max_params = 10;
        config
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

// Implement serde for LintLevel
impl Serialize for LintLevel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(match self {
            LintLevel::Allow => "allow",
            LintLevel::Warn => "warn",
            LintLevel::Deny => "deny",
            LintLevel::Forbid => "forbid",
        })
    }
}

impl<'de> Deserialize<'de> for LintLevel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "allow" => Ok(LintLevel::Allow),
            "warn" => Ok(LintLevel::Warn),
            "deny" => Ok(LintLevel::Deny),
            "forbid" => Ok(LintLevel::Forbid),
            _ => Err(serde::de::Error::unknown_variant(
                &s,
                &["allow", "warn", "deny", "forbid"],
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LintConfig::default();
        assert!(config.levels.is_empty());
        assert!(config.groups.is_empty());
        assert_eq!(config.max_complexity, 15);
    }

    #[test]
    fn test_set_level() {
        let mut config = LintConfig::default();
        config.set_level("unused_variable", LintLevel::Deny);
        assert_eq!(config.get_level("unused_variable"), Some(LintLevel::Deny));
    }

    #[test]
    fn test_group_management() {
        let mut config = LintConfig::default();

        config.enable_group("correctness");
        assert!(config.is_group_enabled("correctness"));

        config.disable_group("correctness");
        assert!(!config.is_group_enabled("correctness"));
    }

    #[test]
    fn test_serialize_config() {
        let mut config = LintConfig::default();
        config.set_level("unused_variable", LintLevel::Warn);

        let toml_str = toml::to_string(&config).unwrap();
        assert!(toml_str.contains("unused_variable"));
    }
}
