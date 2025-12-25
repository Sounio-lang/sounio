//! Conditional Compilation (cfg) System
//!
//! This module provides a platform abstraction and conditional compilation system
//! similar to Rust's #[cfg] attributes. It allows code to be conditionally compiled
//! based on target platform, features, and custom predicates.

use super::spec::{Architecture, Environment, OperatingSystem, TargetSpec};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use thiserror::Error;

/// Errors that can occur during cfg evaluation.
#[derive(Debug, Error)]
pub enum CfgError {
    #[error("Invalid cfg predicate: {0}")]
    InvalidPredicate(String),

    #[error("Unknown cfg key: {0}")]
    UnknownKey(String),

    #[error("Syntax error in cfg expression: {0}")]
    SyntaxError(String),

    #[error("Undefined cfg variable: {0}")]
    UndefinedVariable(String),
}

/// Result type for cfg operations.
pub type CfgResult<T> = Result<T, CfgError>;

/// A cfg predicate that can be evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CfgPredicate {
    /// A simple key (e.g., `unix`, `windows`)
    Key(String),

    /// A key-value pair (e.g., `target_os = "linux"`)
    KeyValue { key: String, value: String },

    /// Negation: `not(predicate)`
    Not(Box<CfgPredicate>),

    /// Conjunction: `all(pred1, pred2, ...)`
    All(Vec<CfgPredicate>),

    /// Disjunction: `any(pred1, pred2, ...)`
    Any(Vec<CfgPredicate>),

    /// Always true
    True,

    /// Always false
    False,
}

impl CfgPredicate {
    /// Parse a cfg predicate from a string.
    pub fn parse(s: &str) -> CfgResult<Self> {
        let s = s.trim();

        if s.is_empty() {
            return Err(CfgError::SyntaxError("Empty predicate".to_string()));
        }

        // Handle not(...)
        if s.starts_with("not(") && s.ends_with(')') {
            let inner = &s[4..s.len() - 1];
            return Ok(Self::Not(Box::new(Self::parse(inner)?)));
        }

        // Handle all(...)
        if s.starts_with("all(") && s.ends_with(')') {
            let inner = &s[4..s.len() - 1];
            let predicates = Self::parse_list(inner)?;
            return Ok(Self::All(predicates));
        }

        // Handle any(...)
        if s.starts_with("any(") && s.ends_with(')') {
            let inner = &s[4..s.len() - 1];
            let predicates = Self::parse_list(inner)?;
            return Ok(Self::Any(predicates));
        }

        // Handle key = "value"
        if let Some(eq_pos) = s.find('=') {
            let key = s[..eq_pos].trim().to_string();
            let value = s[eq_pos + 1..].trim();

            // Remove quotes from value
            let value = if (value.starts_with('"') && value.ends_with('"'))
                || (value.starts_with('\'') && value.ends_with('\''))
            {
                value[1..value.len() - 1].to_string()
            } else {
                value.to_string()
            };

            return Ok(Self::KeyValue { key, value });
        }

        // Handle boolean literals
        if s == "true" {
            return Ok(Self::True);
        }
        if s == "false" {
            return Ok(Self::False);
        }

        // Simple key
        Ok(Self::Key(s.to_string()))
    }

    /// Parse a comma-separated list of predicates.
    fn parse_list(s: &str) -> CfgResult<Vec<CfgPredicate>> {
        let mut predicates = Vec::new();
        let mut current = String::new();
        let mut depth = 0;

        for c in s.chars() {
            match c {
                '(' => {
                    depth += 1;
                    current.push(c);
                }
                ')' => {
                    depth -= 1;
                    current.push(c);
                }
                ',' if depth == 0 => {
                    let pred = current.trim().to_string();
                    if !pred.is_empty() {
                        predicates.push(Self::parse(&pred)?);
                    }
                    current.clear();
                }
                _ => current.push(c),
            }
        }

        // Don't forget the last one
        let pred = current.trim().to_string();
        if !pred.is_empty() {
            predicates.push(Self::parse(&pred)?);
        }

        Ok(predicates)
    }

    /// Evaluate the predicate against a cfg context.
    pub fn evaluate(&self, ctx: &CfgContext) -> bool {
        match self {
            Self::Key(key) => ctx.is_set(key),
            Self::KeyValue { key, value } => ctx.get(key).map(|v| v == value).unwrap_or(false),
            Self::Not(inner) => !inner.evaluate(ctx),
            Self::All(preds) => preds.iter().all(|p| p.evaluate(ctx)),
            Self::Any(preds) => preds.iter().any(|p| p.evaluate(ctx)),
            Self::True => true,
            Self::False => false,
        }
    }

    /// Check if this predicate is satisfiable (can be true for some context).
    pub fn is_satisfiable(&self) -> bool {
        match self {
            Self::False => false,
            Self::Not(inner) => match inner.as_ref() {
                Self::True => false,
                _ => true,
            },
            Self::All(preds) => preds.iter().all(|p| p.is_satisfiable()),
            _ => true,
        }
    }

    /// Simplify the predicate.
    pub fn simplify(&self) -> Self {
        match self {
            Self::Not(inner) => match inner.as_ref() {
                Self::True => Self::False,
                Self::False => Self::True,
                Self::Not(inner2) => inner2.simplify(),
                _ => Self::Not(Box::new(inner.simplify())),
            },
            Self::All(preds) => {
                let simplified: Vec<_> = preds
                    .iter()
                    .map(|p| p.simplify())
                    .filter(|p| *p != Self::True)
                    .collect();

                if simplified.contains(&Self::False) {
                    Self::False
                } else if simplified.is_empty() {
                    Self::True
                } else if simplified.len() == 1 {
                    simplified.into_iter().next().unwrap()
                } else {
                    Self::All(simplified)
                }
            }
            Self::Any(preds) => {
                let simplified: Vec<_> = preds
                    .iter()
                    .map(|p| p.simplify())
                    .filter(|p| *p != Self::False)
                    .collect();

                if simplified.contains(&Self::True) {
                    Self::True
                } else if simplified.is_empty() {
                    Self::False
                } else if simplified.len() == 1 {
                    simplified.into_iter().next().unwrap()
                } else {
                    Self::Any(simplified)
                }
            }
            _ => self.clone(),
        }
    }
}

impl fmt::Display for CfgPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Key(key) => write!(f, "{}", key),
            Self::KeyValue { key, value } => write!(f, "{} = \"{}\"", key, value),
            Self::Not(inner) => write!(f, "not({})", inner),
            Self::All(preds) => {
                let strs: Vec<_> = preds.iter().map(|p| p.to_string()).collect();
                write!(f, "all({})", strs.join(", "))
            }
            Self::Any(preds) => {
                let strs: Vec<_> = preds.iter().map(|p| p.to_string()).collect();
                write!(f, "any({})", strs.join(", "))
            }
            Self::True => write!(f, "true"),
            Self::False => write!(f, "false"),
        }
    }
}

/// Context for evaluating cfg predicates.
#[derive(Debug, Clone, Default)]
pub struct CfgContext {
    /// Key-value pairs
    values: HashMap<String, String>,
    /// Boolean flags (keys without values)
    flags: HashSet<String>,
}

impl CfgContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a context from a target specification.
    pub fn from_target(spec: &TargetSpec) -> Self {
        let mut ctx = Self::new();

        // Target OS
        ctx.set("target_os", spec.os.os.name());
        match spec.os.os {
            OperatingSystem::Linux => ctx.set_flag("linux"),
            OperatingSystem::Windows => ctx.set_flag("windows"),
            OperatingSystem::MacOs => {
                ctx.set_flag("macos");
                ctx.set_flag("unix");
            }
            OperatingSystem::FreeBsd
            | OperatingSystem::NetBsd
            | OperatingSystem::OpenBsd
            | OperatingSystem::DragonFly => {
                ctx.set_flag("bsd");
                ctx.set_flag("unix");
            }
            OperatingSystem::Wasi => ctx.set_flag("wasi"),
            OperatingSystem::None => ctx.set_flag("bare_metal"),
            _ => {}
        }

        // Unix-like flag
        if spec.os.os.is_unix_like() {
            ctx.set_flag("unix");
        }

        // Target arch
        ctx.set("target_arch", spec.arch.arch.name());
        match spec.arch.arch {
            Architecture::X86_64 => ctx.set_flag("x86_64"),
            Architecture::X86 => ctx.set_flag("x86"),
            Architecture::Aarch64 => ctx.set_flag("aarch64"),
            Architecture::Arm => ctx.set_flag("arm"),
            Architecture::Riscv64 => ctx.set_flag("riscv64"),
            Architecture::Riscv32 => ctx.set_flag("riscv32"),
            Architecture::Wasm32 => ctx.set_flag("wasm32"),
            Architecture::Wasm64 => ctx.set_flag("wasm64"),
            _ => {}
        }

        // Pointer width
        let pointer_width = spec.arch.arch.pointer_width();
        ctx.set("target_pointer_width", &pointer_width.to_string());
        if pointer_width == 64 {
            ctx.set_flag("target_64bit");
        } else if pointer_width == 32 {
            ctx.set_flag("target_32bit");
        }

        // Environment
        ctx.set("target_env", spec.env.env.name());
        match spec.env.env {
            Environment::Gnu => ctx.set_flag("gnu"),
            Environment::Musl => ctx.set_flag("musl"),
            Environment::Msvc => ctx.set_flag("msvc"),
            _ => {}
        }

        // Target family
        if spec.triple.is_windows() {
            ctx.set("target_family", "windows");
        } else if spec.os.os.is_unix_like() {
            ctx.set("target_family", "unix");
        } else if spec.triple.is_wasm() {
            ctx.set("target_family", "wasm");
        }

        // Target triple
        ctx.set("target", &spec.triple.to_string());
        ctx.set("target_triple", &spec.triple.to_string());

        // Endianness
        ctx.set("target_endian", "little"); // Most targets are little-endian
        match spec.arch.arch {
            Architecture::Powerpc | Architecture::Mips | Architecture::Sparc64 => {
                ctx.set("target_endian", "big");
                ctx.set_flag("big_endian");
            }
            _ => {
                ctx.set_flag("little_endian");
            }
        }

        // Vendor
        ctx.set("target_vendor", &spec.triple.vendor);

        // CPU features
        for feature in spec.features.effective_features() {
            ctx.set_flag(&format!("target_feature_{}", feature));
            ctx.set("target_feature", &feature);
        }

        // Has std (not for bare metal or wasm without wasi)
        let has_std = !spec.triple.is_bare_metal()
            && !(spec.triple.is_wasm() && spec.os.os != OperatingSystem::Wasi);
        if has_std {
            ctx.set_flag("feature_std");
        }

        // Atomics support
        if let Some(width) = spec.options.max_atomic_width {
            ctx.set("target_has_atomic", &width.to_string());
            ctx.set_flag(&format!("target_has_atomic_{}", width));
        }

        // Debug/release
        #[cfg(debug_assertions)]
        ctx.set_flag("debug_assertions");

        ctx
    }

    /// Set a key-value pair.
    pub fn set(&mut self, key: &str, value: &str) {
        self.values.insert(key.to_string(), value.to_string());
    }

    /// Set a boolean flag.
    pub fn set_flag(&mut self, key: &str) {
        self.flags.insert(key.to_string());
    }

    /// Unset a flag.
    pub fn unset_flag(&mut self, key: &str) {
        self.flags.remove(key);
    }

    /// Get a value.
    pub fn get(&self, key: &str) -> Option<&String> {
        self.values.get(key)
    }

    /// Check if a flag is set.
    pub fn is_set(&self, key: &str) -> bool {
        self.flags.contains(key) || self.values.contains_key(key)
    }

    /// Get all set flags.
    pub fn flags(&self) -> &HashSet<String> {
        &self.flags
    }

    /// Get all values.
    pub fn values(&self) -> &HashMap<String, String> {
        &self.values
    }

    /// Merge another context into this one.
    pub fn merge(&mut self, other: &CfgContext) {
        for (k, v) in &other.values {
            self.values.insert(k.clone(), v.clone());
        }
        for flag in &other.flags {
            self.flags.insert(flag.clone());
        }
    }

    /// Create a child context that inherits from this one.
    pub fn child(&self) -> Self {
        self.clone()
    }

    /// Evaluate a predicate in this context.
    pub fn evaluate(&self, pred: &CfgPredicate) -> bool {
        pred.evaluate(self)
    }

    /// Evaluate a predicate string.
    pub fn evaluate_str(&self, s: &str) -> CfgResult<bool> {
        let pred = CfgPredicate::parse(s)?;
        Ok(self.evaluate(&pred))
    }
}

/// Built-in cfg options and their documentation.
#[derive(Debug)]
pub struct CfgOption {
    /// Option name
    pub name: &'static str,
    /// Description
    pub description: &'static str,
    /// Possible values (if key-value)
    pub values: Option<&'static [&'static str]>,
    /// Whether this is a flag (no value)
    pub is_flag: bool,
}

/// List of all built-in cfg options.
pub static BUILTIN_CFG_OPTIONS: &[CfgOption] = &[
    CfgOption {
        name: "target_os",
        description: "Target operating system",
        values: Some(&[
            "linux",
            "windows",
            "macos",
            "ios",
            "android",
            "freebsd",
            "netbsd",
            "openbsd",
            "dragonfly",
            "solaris",
            "illumos",
            "fuchsia",
            "redox",
            "wasi",
            "none",
        ]),
        is_flag: false,
    },
    CfgOption {
        name: "target_arch",
        description: "Target CPU architecture",
        values: Some(&[
            "x86_64",
            "x86",
            "aarch64",
            "arm",
            "riscv64",
            "riscv32",
            "wasm32",
            "wasm64",
            "powerpc64",
            "powerpc",
            "mips64",
            "mips",
            "sparc64",
            "s390x",
        ]),
        is_flag: false,
    },
    CfgOption {
        name: "target_env",
        description: "Target environment/ABI",
        values: Some(&[
            "gnu",
            "gnueabi",
            "gnueabihf",
            "musl",
            "musleabi",
            "musleabihf",
            "msvc",
            "android",
            "sgx",
            "uefi",
        ]),
        is_flag: false,
    },
    CfgOption {
        name: "target_family",
        description: "Target OS family",
        values: Some(&["unix", "windows", "wasm"]),
        is_flag: false,
    },
    CfgOption {
        name: "target_pointer_width",
        description: "Pointer width in bits",
        values: Some(&["16", "32", "64"]),
        is_flag: false,
    },
    CfgOption {
        name: "target_endian",
        description: "Target endianness",
        values: Some(&["little", "big"]),
        is_flag: false,
    },
    CfgOption {
        name: "target_vendor",
        description: "Target vendor",
        values: Some(&["unknown", "pc", "apple", "nvidia", "arm"]),
        is_flag: false,
    },
    CfgOption {
        name: "unix",
        description: "Unix-like target",
        values: None,
        is_flag: true,
    },
    CfgOption {
        name: "windows",
        description: "Windows target",
        values: None,
        is_flag: true,
    },
    CfgOption {
        name: "linux",
        description: "Linux target",
        values: None,
        is_flag: true,
    },
    CfgOption {
        name: "macos",
        description: "macOS target",
        values: None,
        is_flag: true,
    },
    CfgOption {
        name: "debug_assertions",
        description: "Debug assertions enabled",
        values: None,
        is_flag: true,
    },
    CfgOption {
        name: "feature_std",
        description: "Standard library available",
        values: None,
        is_flag: true,
    },
];

/// Conditional compilation directive for D code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfgDirective {
    /// Include code if predicate is true: `@cfg(predicate)`
    If(CfgPredicate),

    /// Include code if predicate is false: `@cfg_not(predicate)`
    IfNot(CfgPredicate),

    /// Set a cfg flag: `@cfg_set(name)`
    Set(String),

    /// Set a cfg value: `@cfg_set(name = "value")`
    SetValue { name: String, value: String },

    /// Conditional attribute: `@cfg_attr(predicate, attr)`
    Attr {
        predicate: CfgPredicate,
        attribute: String,
    },
}

impl CfgDirective {
    /// Parse a cfg directive from a string.
    pub fn parse(s: &str) -> CfgResult<Self> {
        let s = s.trim();

        if s.starts_with("@cfg(") && s.ends_with(')') {
            let inner = &s[5..s.len() - 1];
            return Ok(Self::If(CfgPredicate::parse(inner)?));
        }

        if s.starts_with("@cfg_not(") && s.ends_with(')') {
            let inner = &s[9..s.len() - 1];
            return Ok(Self::IfNot(CfgPredicate::parse(inner)?));
        }

        if s.starts_with("@cfg_set(") && s.ends_with(')') {
            let inner = &s[9..s.len() - 1].trim();
            if let Some(eq_pos) = inner.find('=') {
                let name = inner[..eq_pos].trim().to_string();
                let value = inner[eq_pos + 1..].trim();
                let value = if value.starts_with('"') && value.ends_with('"') {
                    value[1..value.len() - 1].to_string()
                } else {
                    value.to_string()
                };
                return Ok(Self::SetValue { name, value });
            } else {
                return Ok(Self::Set(inner.to_string()));
            }
        }

        if s.starts_with("@cfg_attr(") && s.ends_with(')') {
            let inner = &s[10..s.len() - 1];
            // Find the comma separating predicate and attribute
            let mut depth = 0;
            let mut comma_pos = None;
            for (i, c) in inner.char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    ',' if depth == 0 => {
                        comma_pos = Some(i);
                        break;
                    }
                    _ => {}
                }
            }

            if let Some(pos) = comma_pos {
                let pred_str = &inner[..pos].trim();
                let attr = inner[pos + 1..].trim().to_string();
                return Ok(Self::Attr {
                    predicate: CfgPredicate::parse(pred_str)?,
                    attribute: attr,
                });
            }
        }

        Err(CfgError::SyntaxError(format!("Invalid directive: {}", s)))
    }

    /// Check if this directive should include the annotated code.
    pub fn should_include(&self, ctx: &CfgContext) -> bool {
        match self {
            Self::If(pred) => pred.evaluate(ctx),
            Self::IfNot(pred) => !pred.evaluate(ctx),
            Self::Set(_) | Self::SetValue { .. } => true, // Always included
            Self::Attr { predicate, .. } => predicate.evaluate(ctx),
        }
    }
}

/// Platform detection utilities.
pub mod platform {
    use super::*;

    /// Get the host platform cfg context.
    pub fn host_context() -> CfgContext {
        let mut ctx = CfgContext::new();

        // OS
        #[cfg(target_os = "linux")]
        {
            ctx.set("target_os", "linux");
            ctx.set_flag("linux");
            ctx.set_flag("unix");
            ctx.set("target_family", "unix");
        }

        #[cfg(target_os = "windows")]
        {
            ctx.set("target_os", "windows");
            ctx.set_flag("windows");
            ctx.set("target_family", "windows");
        }

        #[cfg(target_os = "macos")]
        {
            ctx.set("target_os", "macos");
            ctx.set_flag("macos");
            ctx.set_flag("unix");
            ctx.set("target_family", "unix");
        }

        // Arch
        #[cfg(target_arch = "x86_64")]
        {
            ctx.set("target_arch", "x86_64");
            ctx.set_flag("x86_64");
            ctx.set("target_pointer_width", "64");
            ctx.set_flag("target_64bit");
        }

        #[cfg(target_arch = "aarch64")]
        {
            ctx.set("target_arch", "aarch64");
            ctx.set_flag("aarch64");
            ctx.set("target_pointer_width", "64");
            ctx.set_flag("target_64bit");
        }

        #[cfg(target_arch = "x86")]
        {
            ctx.set("target_arch", "x86");
            ctx.set_flag("x86");
            ctx.set("target_pointer_width", "32");
            ctx.set_flag("target_32bit");
        }

        // Endian
        #[cfg(target_endian = "little")]
        {
            ctx.set("target_endian", "little");
            ctx.set_flag("little_endian");
        }

        #[cfg(target_endian = "big")]
        {
            ctx.set("target_endian", "big");
            ctx.set_flag("big_endian");
        }

        // Environment
        #[cfg(target_env = "gnu")]
        ctx.set("target_env", "gnu");

        #[cfg(target_env = "musl")]
        ctx.set("target_env", "musl");

        #[cfg(target_env = "msvc")]
        ctx.set("target_env", "msvc");

        // Debug assertions
        #[cfg(debug_assertions)]
        ctx.set_flag("debug_assertions");

        ctx.set_flag("feature_std");

        ctx
    }

    /// Check if running on Unix-like platform.
    pub fn is_unix() -> bool {
        cfg!(unix)
    }

    /// Check if running on Windows.
    pub fn is_windows() -> bool {
        cfg!(windows)
    }

    /// Check if running on macOS.
    pub fn is_macos() -> bool {
        cfg!(target_os = "macos")
    }

    /// Check if running on Linux.
    pub fn is_linux() -> bool {
        cfg!(target_os = "linux")
    }

    /// Get the host architecture name.
    pub fn host_arch() -> &'static str {
        #[cfg(target_arch = "x86_64")]
        return "x86_64";

        #[cfg(target_arch = "aarch64")]
        return "aarch64";

        #[cfg(target_arch = "x86")]
        return "x86";

        #[cfg(target_arch = "arm")]
        return "arm";

        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "x86",
            target_arch = "arm"
        )))]
        return "unknown";
    }

    /// Get the host OS name.
    pub fn host_os() -> &'static str {
        #[cfg(target_os = "linux")]
        return "linux";

        #[cfg(target_os = "windows")]
        return "windows";

        #[cfg(target_os = "macos")]
        return "macos";

        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        return "unknown";
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_parse_key() {
        let pred = CfgPredicate::parse("unix").unwrap();
        assert_eq!(pred, CfgPredicate::Key("unix".to_string()));
    }

    #[test]
    fn test_predicate_parse_key_value() {
        let pred = CfgPredicate::parse("target_os = \"linux\"").unwrap();
        assert_eq!(
            pred,
            CfgPredicate::KeyValue {
                key: "target_os".to_string(),
                value: "linux".to_string()
            }
        );
    }

    #[test]
    fn test_predicate_parse_not() {
        let pred = CfgPredicate::parse("not(windows)").unwrap();
        assert_eq!(
            pred,
            CfgPredicate::Not(Box::new(CfgPredicate::Key("windows".to_string())))
        );
    }

    #[test]
    fn test_predicate_parse_all() {
        let pred = CfgPredicate::parse("all(unix, target_arch = \"x86_64\")").unwrap();
        match pred {
            CfgPredicate::All(preds) => {
                assert_eq!(preds.len(), 2);
                assert_eq!(preds[0], CfgPredicate::Key("unix".to_string()));
            }
            _ => panic!("Expected All"),
        }
    }

    #[test]
    fn test_predicate_parse_any() {
        let pred = CfgPredicate::parse("any(linux, macos)").unwrap();
        match pred {
            CfgPredicate::Any(preds) => {
                assert_eq!(preds.len(), 2);
            }
            _ => panic!("Expected Any"),
        }
    }

    #[test]
    fn test_predicate_evaluate() {
        let mut ctx = CfgContext::new();
        ctx.set_flag("unix");
        ctx.set("target_os", "linux");

        assert!(CfgPredicate::Key("unix".to_string()).evaluate(&ctx));
        assert!(!CfgPredicate::Key("windows".to_string()).evaluate(&ctx));

        assert!(
            CfgPredicate::KeyValue {
                key: "target_os".to_string(),
                value: "linux".to_string()
            }
            .evaluate(&ctx)
        );

        assert!(
            !CfgPredicate::KeyValue {
                key: "target_os".to_string(),
                value: "windows".to_string()
            }
            .evaluate(&ctx)
        );
    }

    #[test]
    fn test_predicate_not() {
        let mut ctx = CfgContext::new();
        ctx.set_flag("unix");

        let pred = CfgPredicate::Not(Box::new(CfgPredicate::Key("windows".to_string())));
        assert!(pred.evaluate(&ctx));

        let pred = CfgPredicate::Not(Box::new(CfgPredicate::Key("unix".to_string())));
        assert!(!pred.evaluate(&ctx));
    }

    #[test]
    fn test_predicate_all_any() {
        let mut ctx = CfgContext::new();
        ctx.set_flag("unix");
        ctx.set_flag("linux");

        let all_pred = CfgPredicate::All(vec![
            CfgPredicate::Key("unix".to_string()),
            CfgPredicate::Key("linux".to_string()),
        ]);
        assert!(all_pred.evaluate(&ctx));

        let all_pred = CfgPredicate::All(vec![
            CfgPredicate::Key("unix".to_string()),
            CfgPredicate::Key("windows".to_string()),
        ]);
        assert!(!all_pred.evaluate(&ctx));

        let any_pred = CfgPredicate::Any(vec![
            CfgPredicate::Key("linux".to_string()),
            CfgPredicate::Key("windows".to_string()),
        ]);
        assert!(any_pred.evaluate(&ctx));
    }

    #[test]
    fn test_predicate_simplify() {
        let pred = CfgPredicate::Not(Box::new(CfgPredicate::Not(Box::new(CfgPredicate::Key(
            "unix".to_string(),
        )))));
        let simplified = pred.simplify();
        assert_eq!(simplified, CfgPredicate::Key("unix".to_string()));

        let pred = CfgPredicate::All(vec![
            CfgPredicate::True,
            CfgPredicate::Key("unix".to_string()),
        ]);
        let simplified = pred.simplify();
        assert_eq!(simplified, CfgPredicate::Key("unix".to_string()));
    }

    #[test]
    fn test_context_from_target() {
        let spec = TargetSpec::from_triple("x86_64-unknown-linux-gnu").unwrap();
        let ctx = CfgContext::from_target(&spec);

        assert!(ctx.is_set("linux"));
        assert!(ctx.is_set("unix"));
        assert!(ctx.is_set("x86_64"));
        assert_eq!(ctx.get("target_os"), Some(&"linux".to_string()));
        assert_eq!(ctx.get("target_arch"), Some(&"x86_64".to_string()));
    }

    #[test]
    fn test_cfg_directive_parse() {
        let dir = CfgDirective::parse("@cfg(unix)").unwrap();
        match dir {
            CfgDirective::If(pred) => {
                assert_eq!(pred, CfgPredicate::Key("unix".to_string()));
            }
            _ => panic!("Expected If"),
        }

        let dir = CfgDirective::parse("@cfg_not(windows)").unwrap();
        match dir {
            CfgDirective::IfNot(pred) => {
                assert_eq!(pred, CfgPredicate::Key("windows".to_string()));
            }
            _ => panic!("Expected IfNot"),
        }
    }

    #[test]
    fn test_host_context() {
        let ctx = platform::host_context();

        // Should have at least these basic settings
        assert!(ctx.get("target_os").is_some());
        assert!(ctx.get("target_arch").is_some());
    }

    #[test]
    fn test_predicate_display() {
        let pred = CfgPredicate::All(vec![
            CfgPredicate::Key("unix".to_string()),
            CfgPredicate::Not(Box::new(CfgPredicate::Key("windows".to_string()))),
        ]);
        assert_eq!(pred.to_string(), "all(unix, not(windows))");
    }

    #[test]
    fn test_context_merge() {
        let mut ctx1 = CfgContext::new();
        ctx1.set_flag("unix");
        ctx1.set("target_os", "linux");

        let mut ctx2 = CfgContext::new();
        ctx2.set_flag("feature_x");
        ctx2.set("version", "1.0");

        ctx1.merge(&ctx2);

        assert!(ctx1.is_set("unix"));
        assert!(ctx1.is_set("feature_x"));
        assert_eq!(ctx1.get("target_os"), Some(&"linux".to_string()));
        assert_eq!(ctx1.get("version"), Some(&"1.0".to_string()));
    }
}
