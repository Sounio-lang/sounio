//! Package manifest parsing and validation
//!
//! Parses d.toml files that define D packages.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Semantic version
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub pre: Option<String>,
    pub build: Option<String>,
}

impl Version {
    pub fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
            pre: None,
            build: None,
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        let (version_part, build) = if let Some(idx) = s.find('+') {
            (&s[..idx], Some(s[idx + 1..].to_string()))
        } else {
            (s, None)
        };

        let (version_part, pre) = if let Some(idx) = version_part.find('-') {
            (
                &version_part[..idx],
                Some(version_part[idx + 1..].to_string()),
            )
        } else {
            (version_part, None)
        };

        let parts: Vec<&str> = version_part.split('.').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(format!("Invalid version format: {}", s));
        }

        let major = parts[0]
            .parse()
            .map_err(|_| format!("Invalid major version: {}", parts[0]))?;
        let minor = parts[1]
            .parse()
            .map_err(|_| format!("Invalid minor version: {}", parts[1]))?;
        let patch = if parts.len() > 2 {
            parts[2]
                .parse()
                .map_err(|_| format!("Invalid patch version: {}", parts[2]))?
        } else {
            0
        };

        Ok(Self {
            major,
            minor,
            patch,
            pre,
            build,
        })
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(ref pre) = self.pre {
            write!(f, "-{}", pre)?;
        }
        if let Some(ref build) = self.build {
            write!(f, "+{}", build)?;
        }
        Ok(())
    }
}

impl TryFrom<String> for Version {
    type Error = String;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        Version::parse(&s)
    }
}

impl From<Version> for String {
    fn from(v: Version) -> Self {
        v.to_string()
    }
}

impl Default for Version {
    fn default() -> Self {
        Self::new(0, 1, 0)
    }
}

/// Version requirement
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct VersionReq {
    pub comparators: Vec<Comparator>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comparator {
    pub op: Op,
    pub major: u64,
    pub minor: Option<u64>,
    pub patch: Option<u64>,
    pub pre: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Exact,     // =
    Greater,   // >
    GreaterEq, // >=
    Less,      // <
    LessEq,    // <=
    Tilde,     // ~
    Caret,     // ^
    Wildcard,  // *
}

impl VersionReq {
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.is_empty() || s == "*" {
            return Ok(Self {
                comparators: vec![Comparator {
                    op: Op::Wildcard,
                    major: 0,
                    minor: None,
                    patch: None,
                    pre: None,
                }],
            });
        }

        let mut comparators = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            comparators.push(Self::parse_comparator(part)?);
        }

        Ok(Self { comparators })
    }

    fn parse_comparator(s: &str) -> Result<Comparator, String> {
        let s = s.trim();

        let (op, version_str) = if s.starts_with(">=") {
            (Op::GreaterEq, &s[2..])
        } else if s.starts_with("<=") {
            (Op::LessEq, &s[2..])
        } else if s.starts_with('>') {
            (Op::Greater, &s[1..])
        } else if s.starts_with('<') {
            (Op::Less, &s[1..])
        } else if s.starts_with('=') {
            (Op::Exact, &s[1..])
        } else if s.starts_with('~') {
            (Op::Tilde, &s[1..])
        } else if s.starts_with('^') {
            (Op::Caret, &s[1..])
        } else if s == "*" {
            return Ok(Comparator {
                op: Op::Wildcard,
                major: 0,
                minor: None,
                patch: None,
                pre: None,
            });
        } else {
            // Default to caret
            (Op::Caret, s)
        };

        let version_str = version_str.trim();
        let parts: Vec<&str> = version_str.split('.').collect();
        if parts.is_empty() {
            return Err("Empty version".to_string());
        }

        let major = parts[0]
            .parse()
            .map_err(|_| format!("Invalid major version: {}", parts[0]))?;
        let minor = if parts.len() > 1 {
            Some(
                parts[1]
                    .parse()
                    .map_err(|_| format!("Invalid minor version: {}", parts[1]))?,
            )
        } else {
            None
        };
        let patch = if parts.len() > 2 {
            Some(
                parts[2]
                    .split('-')
                    .next()
                    .unwrap()
                    .parse()
                    .map_err(|_| format!("Invalid patch version: {}", parts[2]))?,
            )
        } else {
            None
        };

        Ok(Comparator {
            op,
            major,
            minor,
            patch,
            pre: None,
        })
    }

    pub fn matches(&self, version: &Version) -> bool {
        self.comparators.iter().all(|c| c.matches(version))
    }
}

impl Comparator {
    pub fn matches(&self, version: &Version) -> bool {
        match self.op {
            Op::Wildcard => true,
            Op::Exact => {
                version.major == self.major
                    && self.minor.is_none_or(|m| version.minor == m)
                    && self.patch.is_none_or(|p| version.patch == p)
            }
            Op::Greater => {
                if version.major != self.major {
                    return version.major > self.major;
                }
                if let Some(minor) = self.minor {
                    if version.minor != minor {
                        return version.minor > minor;
                    }
                    if let Some(patch) = self.patch {
                        return version.patch > patch;
                    }
                }
                false
            }
            Op::GreaterEq => {
                if version.major != self.major {
                    return version.major > self.major;
                }
                if let Some(minor) = self.minor {
                    if version.minor != minor {
                        return version.minor > minor;
                    }
                    if let Some(patch) = self.patch {
                        return version.patch >= patch;
                    }
                }
                true
            }
            Op::Less => {
                if version.major != self.major {
                    return version.major < self.major;
                }
                if let Some(minor) = self.minor {
                    if version.minor != minor {
                        return version.minor < minor;
                    }
                    if let Some(patch) = self.patch {
                        return version.patch < patch;
                    }
                }
                false
            }
            Op::LessEq => {
                if version.major != self.major {
                    return version.major < self.major;
                }
                if let Some(minor) = self.minor {
                    if version.minor != minor {
                        return version.minor < minor;
                    }
                    if let Some(patch) = self.patch {
                        return version.patch <= patch;
                    }
                }
                true
            }
            Op::Tilde => {
                // ~1.2.3 := >=1.2.3, <1.3.0
                if version.major != self.major {
                    return false;
                }
                if let Some(minor) = self.minor {
                    if version.minor != minor {
                        return false;
                    }
                    if let Some(patch) = self.patch {
                        return version.patch >= patch;
                    }
                }
                true
            }
            Op::Caret => {
                // ^1.2.3 := >=1.2.3, <2.0.0
                if version.major != self.major {
                    return false;
                }
                if self.major == 0 {
                    // ^0.x.y is more restrictive
                    if let Some(minor) = self.minor {
                        if version.minor != minor {
                            return false;
                        }
                        if minor == 0
                            && let Some(patch) = self.patch
                        {
                            return version.patch >= patch;
                        }
                    }
                }
                true
            }
        }
    }
}

impl Default for VersionReq {
    fn default() -> Self {
        Self {
            comparators: vec![Comparator {
                op: Op::Wildcard,
                major: 0,
                minor: None,
                patch: None,
                pre: None,
            }],
        }
    }
}

impl std::fmt::Display for VersionReq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts: Vec<String> = self.comparators.iter().map(|c| c.to_string()).collect();
        write!(f, "{}", parts.join(", "))
    }
}

impl std::fmt::Display for Comparator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op = match self.op {
            Op::Exact => "=",
            Op::Greater => ">",
            Op::GreaterEq => ">=",
            Op::Less => "<",
            Op::LessEq => "<=",
            Op::Tilde => "~",
            Op::Caret => "^",
            Op::Wildcard => return write!(f, "*"),
        };
        write!(f, "{}{}", op, self.major)?;
        if let Some(minor) = self.minor {
            write!(f, ".{}", minor)?;
            if let Some(patch) = self.patch {
                write!(f, ".{}", patch)?;
            }
        }
        Ok(())
    }
}

impl TryFrom<String> for VersionReq {
    type Error = String;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        VersionReq::parse(&s)
    }
}

impl From<VersionReq> for String {
    fn from(v: VersionReq) -> Self {
        v.to_string()
    }
}

/// Package manifest (d.toml)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Package information
    pub package: Package,

    /// Dependencies
    #[serde(default)]
    pub dependencies: BTreeMap<String, Dependency>,

    /// Dev dependencies (tests, benchmarks)
    #[serde(default, rename = "dev-dependencies")]
    pub dev_dependencies: BTreeMap<String, Dependency>,

    /// Build dependencies (build scripts)
    #[serde(default, rename = "build-dependencies")]
    pub build_dependencies: BTreeMap<String, Dependency>,

    /// Feature flags
    #[serde(default)]
    pub features: BTreeMap<String, Vec<String>>,

    /// Workspace configuration
    #[serde(default)]
    pub workspace: Option<Workspace>,

    /// Build configuration
    #[serde(default)]
    pub build: BuildConfig,

    /// Profile configurations
    #[serde(default)]
    pub profile: BTreeMap<String, Profile>,

    /// Binary targets
    #[serde(default, rename = "bin")]
    pub binaries: Vec<BinaryTarget>,

    /// Library target
    #[serde(default)]
    pub lib: Option<LibraryTarget>,

    /// Example targets
    #[serde(default)]
    pub example: Vec<ExampleTarget>,

    /// Test targets
    #[serde(default)]
    pub test: Vec<TestTarget>,

    /// Benchmark targets
    #[serde(default)]
    pub bench: Vec<BenchTarget>,
}

/// Package metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Package {
    /// Package name
    pub name: String,

    /// Package version (semver)
    pub version: Version,

    /// Package authors
    #[serde(default)]
    pub authors: Vec<String>,

    /// Package description
    #[serde(default)]
    pub description: Option<String>,

    /// Package license
    #[serde(default)]
    pub license: Option<String>,

    /// License file path
    #[serde(default, rename = "license-file")]
    pub license_file: Option<PathBuf>,

    /// Repository URL
    #[serde(default)]
    pub repository: Option<String>,

    /// Homepage URL
    #[serde(default)]
    pub homepage: Option<String>,

    /// Documentation URL
    #[serde(default)]
    pub documentation: Option<String>,

    /// README file path
    #[serde(default)]
    pub readme: Option<PathBuf>,

    /// Keywords for search
    #[serde(default)]
    pub keywords: Vec<String>,

    /// Categories
    #[serde(default)]
    pub categories: Vec<String>,

    /// Minimum D compiler version
    #[serde(default, rename = "d-version")]
    pub d_version: Option<VersionReq>,

    /// Edition (language version)
    #[serde(default)]
    pub edition: Option<String>,

    /// Exclude files from packaging
    #[serde(default)]
    pub exclude: Vec<String>,

    /// Include only these files
    #[serde(default)]
    pub include: Vec<String>,

    /// Publish to registry
    #[serde(default = "default_true")]
    pub publish: bool,

    /// Default features
    #[serde(default, rename = "default-features")]
    pub default_features: Vec<String>,
}

fn default_true() -> bool {
    true
}

/// Dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dependency {
    /// Simple version string
    Simple(String),

    /// Detailed specification
    Detailed(DependencyDetail),
}

/// Detailed dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyDetail {
    /// Version requirement
    #[serde(default)]
    pub version: Option<String>,

    /// Git repository URL
    #[serde(default)]
    pub git: Option<String>,

    /// Git branch
    #[serde(default)]
    pub branch: Option<String>,

    /// Git tag
    #[serde(default)]
    pub tag: Option<String>,

    /// Git revision
    #[serde(default)]
    pub rev: Option<String>,

    /// Local path
    #[serde(default)]
    pub path: Option<PathBuf>,

    /// Registry name
    #[serde(default)]
    pub registry: Option<String>,

    /// Features to enable
    #[serde(default)]
    pub features: Vec<String>,

    /// Use default features
    #[serde(default = "default_true", rename = "default-features")]
    pub default_features: bool,

    /// Optional dependency
    #[serde(default)]
    pub optional: bool,

    /// Package name (if different from key)
    #[serde(default)]
    pub package: Option<String>,
}

/// Workspace configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workspace {
    /// Member packages
    pub members: Vec<String>,

    /// Excluded packages
    #[serde(default)]
    pub exclude: Vec<String>,

    /// Default members for commands
    #[serde(default, rename = "default-members")]
    pub default_members: Vec<String>,

    /// Shared dependencies
    #[serde(default)]
    pub dependencies: BTreeMap<String, Dependency>,
}

/// Build configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Build script path
    #[serde(default)]
    pub script: Option<PathBuf>,

    /// Target directory
    #[serde(default, rename = "target-dir")]
    pub target_dir: Option<PathBuf>,

    /// Number of parallel jobs
    #[serde(default)]
    pub jobs: Option<u32>,

    /// Incremental compilation
    #[serde(default = "default_true")]
    pub incremental: bool,
}

/// Build profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    /// Optimization level (0-3)
    #[serde(default, rename = "opt-level")]
    pub opt_level: Option<u8>,

    /// Debug info (0-2)
    #[serde(default)]
    pub debug: Option<u8>,

    /// Enable debug assertions
    #[serde(default, rename = "debug-assertions")]
    pub debug_assertions: Option<bool>,

    /// Enable overflow checks
    #[serde(default, rename = "overflow-checks")]
    pub overflow_checks: Option<bool>,

    /// Link-time optimization
    #[serde(default)]
    pub lto: Option<LtoConfig>,

    /// Panic strategy
    #[serde(default)]
    pub panic: Option<PanicStrategy>,

    /// Code generation units
    #[serde(default, rename = "codegen-units")]
    pub codegen_units: Option<u32>,

    /// Runtime library
    #[serde(default)]
    pub rpath: Option<bool>,
}

/// LTO configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LtoConfig {
    Bool(bool),
    String(String), // "thin", "fat"
}

/// Panic strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PanicStrategy {
    Unwind,
    Abort,
}

/// Binary target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryTarget {
    pub name: String,
    pub path: Option<PathBuf>,
    #[serde(default = "default_true")]
    pub doc: bool,
    #[serde(default)]
    pub test: bool,
    #[serde(default)]
    pub bench: bool,
    #[serde(default, rename = "required-features")]
    pub required_features: Vec<String>,
}

/// Library target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryTarget {
    pub name: Option<String>,
    pub path: Option<PathBuf>,
    #[serde(default = "default_true")]
    pub doc: bool,
    #[serde(default = "default_true")]
    pub test: bool,
    #[serde(default = "default_true")]
    pub bench: bool,
    #[serde(default, rename = "crate-type")]
    pub crate_type: Vec<String>, // "lib", "dylib", "staticlib"
}

/// Example target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleTarget {
    pub name: String,
    pub path: Option<PathBuf>,
    #[serde(default, rename = "required-features")]
    pub required_features: Vec<String>,
}

/// Test target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestTarget {
    pub name: String,
    pub path: Option<PathBuf>,
    #[serde(default, rename = "required-features")]
    pub required_features: Vec<String>,
}

/// Benchmark target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchTarget {
    pub name: String,
    pub path: Option<PathBuf>,
    #[serde(default, rename = "required-features")]
    pub required_features: Vec<String>,
}

impl Manifest {
    /// Load manifest from file
    pub fn from_path(path: &Path) -> Result<Self, ManifestError> {
        let content = std::fs::read_to_string(path).map_err(ManifestError::Io)?;

        Self::from_str(&content)
    }

    /// Parse manifest from string
    pub fn from_str(content: &str) -> Result<Self, ManifestError> {
        toml::from_str(content).map_err(|e| ManifestError::Parse(e.to_string()))
    }

    /// Write manifest to file
    pub fn to_path(&self, path: &Path) -> Result<(), ManifestError> {
        let content =
            toml::to_string_pretty(self).map_err(|e| ManifestError::Serialize(e.to_string()))?;

        std::fs::write(path, content).map_err(ManifestError::Io)
    }

    /// Get package ID
    pub fn package_id(&self) -> PackageId {
        PackageId {
            name: self.package.name.clone(),
            version: self.package.version.clone(),
        }
    }

    /// Resolve dependency version requirement
    pub fn resolve_dep(&self, name: &str) -> Option<VersionReq> {
        self.dependencies.get(name).map(|dep| match dep {
            Dependency::Simple(v) => VersionReq::parse(v).unwrap_or_default(),
            Dependency::Detailed(d) => d
                .version
                .as_ref()
                .and_then(|v| VersionReq::parse(v).ok())
                .unwrap_or_default(),
        })
    }

    /// Get all dependencies including transitive
    pub fn all_dependencies(&self) -> Vec<(&String, &Dependency)> {
        self.dependencies.iter().collect()
    }

    /// Validate manifest
    pub fn validate(&self) -> Result<(), Vec<ManifestError>> {
        let mut errors = Vec::new();

        // Validate package name
        if !is_valid_package_name(&self.package.name) {
            errors.push(ManifestError::InvalidPackageName(self.package.name.clone()));
        }

        // Validate dependencies
        for (name, dep) in &self.dependencies {
            if let Err(e) = validate_dependency(name, dep) {
                errors.push(e);
            }
        }

        // Validate features
        for (feature, deps) in &self.features {
            for dep in deps {
                if !self.dependencies.contains_key(dep)
                    && !self.features.contains_key(dep)
                    && !dep.contains('/')
                {
                    errors.push(ManifestError::InvalidFeature(feature.clone(), dep.clone()));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Package identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PackageId {
    pub name: String,
    pub version: Version,
}

impl std::fmt::Display for PackageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.name, self.version)
    }
}

/// Manifest errors
#[derive(Debug)]
pub enum ManifestError {
    Io(std::io::Error),
    Parse(String),
    Serialize(String),
    InvalidPackageName(String),
    InvalidDependency(String, String),
    InvalidFeature(String, String),
    MissingField(String),
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestError::Io(e) => write!(f, "IO error: {}", e),
            ManifestError::Parse(e) => write!(f, "Parse error: {}", e),
            ManifestError::Serialize(e) => write!(f, "Serialize error: {}", e),
            ManifestError::InvalidPackageName(n) => write!(f, "Invalid package name: {}", n),
            ManifestError::InvalidDependency(n, e) => {
                write!(f, "Invalid dependency '{}': {}", n, e)
            }
            ManifestError::InvalidFeature(feat, d) => {
                write!(f, "Invalid feature '{}': unknown dependency '{}'", feat, d)
            }
            ManifestError::MissingField(field) => write!(f, "Missing required field: {}", field),
        }
    }
}

impl std::error::Error for ManifestError {}

/// Validate package name
fn is_valid_package_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 64 {
        return false;
    }

    let first = name.chars().next().unwrap();
    if !first.is_ascii_lowercase() && first != '_' {
        return false;
    }

    name.chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

/// Validate dependency
fn validate_dependency(name: &str, dep: &Dependency) -> Result<(), ManifestError> {
    match dep {
        Dependency::Simple(version) => {
            VersionReq::parse(version).map_err(|_| {
                ManifestError::InvalidDependency(
                    name.to_string(),
                    format!("invalid version requirement: {}", version),
                )
            })?;
        }
        Dependency::Detailed(d) => {
            // Must have at least one source
            if d.version.is_none() && d.git.is_none() && d.path.is_none() {
                return Err(ManifestError::InvalidDependency(
                    name.to_string(),
                    "must specify version, git, or path".to_string(),
                ));
            }

            // Validate version if present
            if let Some(ref v) = d.version {
                VersionReq::parse(v).map_err(|_| {
                    ManifestError::InvalidDependency(
                        name.to_string(),
                        format!("invalid version requirement: {}", v),
                    )
                })?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parse() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);

        let v = Version::parse("0.1.0-alpha").unwrap();
        assert_eq!(v.pre, Some("alpha".to_string()));

        let v = Version::parse("1.0.0+build.123").unwrap();
        assert_eq!(v.build, Some("build.123".to_string()));
    }

    #[test]
    fn test_version_req_caret() {
        let req = VersionReq::parse("^1.2.3").unwrap();
        assert!(req.matches(&Version::parse("1.2.3").unwrap()));
        assert!(req.matches(&Version::parse("1.2.4").unwrap()));
        assert!(req.matches(&Version::parse("1.9.0").unwrap()));
        assert!(!req.matches(&Version::parse("2.0.0").unwrap()));
    }

    #[test]
    fn test_version_req_tilde() {
        let req = VersionReq::parse("~1.2.3").unwrap();
        assert!(req.matches(&Version::parse("1.2.3").unwrap()));
        assert!(req.matches(&Version::parse("1.2.9").unwrap()));
        assert!(!req.matches(&Version::parse("1.3.0").unwrap()));
    }

    #[test]
    fn test_manifest_parse() {
        let toml = r#"
[package]
name = "my-package"
version = "0.1.0"

[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }
"#;

        let manifest = Manifest::from_str(toml).unwrap();
        assert_eq!(manifest.package.name, "my-package");
        assert_eq!(manifest.dependencies.len(), 2);
    }

    #[test]
    fn test_valid_package_names() {
        assert!(is_valid_package_name("hello"));
        assert!(is_valid_package_name("hello-world"));
        assert!(is_valid_package_name("hello_world"));
        assert!(is_valid_package_name("hello123"));
        assert!(!is_valid_package_name("Hello"));
        assert!(!is_valid_package_name("123hello"));
        assert!(!is_valid_package_name(""));
    }
}
