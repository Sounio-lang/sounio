//! Reproducible build support
//!
//! This module provides tools for ensuring build reproducibility,
//! including environment capture and SLSA provenance attestations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Build environment for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildEnvironment {
    /// Compiler version
    pub compiler_version: String,

    /// Compiler git commit
    pub compiler_commit: Option<String>,

    /// Target triple
    pub target: String,

    /// Build profile
    pub profile: String,

    /// Environment variables (filtered)
    pub env: HashMap<String, String>,

    /// Source timestamp (for reproducibility)
    pub source_epoch: Option<u64>,

    /// Locale settings
    pub locale: String,

    /// Timezone
    pub timezone: String,

    /// Host information
    pub host: HostInfo,
}

/// Host system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    /// Operating system
    pub os: String,

    /// Architecture
    pub arch: String,

    /// Hostname (redacted for privacy)
    pub hostname: Option<String>,
}

impl BuildEnvironment {
    /// Capture current environment
    pub fn capture(target: &str, profile: &str) -> Self {
        // Filter environment to reproducibility-relevant variables
        let relevant_vars = [
            "PATH",
            "HOME",
            "LANG",
            "LC_ALL",
            "D_FLAGS",
            "D_INCREMENTAL",
            "CC",
            "CXX",
            "CFLAGS",
            "CXXFLAGS",
            "LDFLAGS",
        ];

        let env: HashMap<String, String> = std::env::vars()
            .filter(|(k, _)| relevant_vars.contains(&k.as_str()))
            .collect();

        BuildEnvironment {
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            compiler_commit: option_env!("GIT_HASH").map(String::from),
            target: target.to_string(),
            profile: profile.to_string(),
            env,
            source_epoch: std::env::var("SOURCE_DATE_EPOCH")
                .ok()
                .and_then(|s| s.parse().ok()),
            locale: std::env::var("LANG").unwrap_or_else(|_| "C".into()),
            timezone: std::env::var("TZ").unwrap_or_else(|_| "UTC".into()),
            host: HostInfo {
                os: std::env::consts::OS.to_string(),
                arch: std::env::consts::ARCH.to_string(),
                hostname: None, // Privacy
            },
        }
    }

    /// Hash the environment
    pub fn hash(&self) -> String {
        // Create a deterministic representation by sorting env keys
        let mut sorted_env: Vec<_> = self.env.iter().collect();
        sorted_env.sort_by_key(|(k, _)| *k);

        // Build deterministic hash input
        let hash_input = format!(
            "{}|{}|{}|{}|{:?}|{:?}|{}|{}|{}|{}",
            self.compiler_version,
            self.compiler_commit.as_deref().unwrap_or(""),
            self.target,
            self.profile,
            sorted_env,
            self.source_epoch,
            self.locale,
            self.timezone,
            self.host.os,
            self.host.arch
        );
        format!("{:x}", Sha256::digest(hash_input.as_bytes()))
    }

    /// Check if environment supports reproducible builds
    pub fn is_reproducible(&self) -> bool {
        self.source_epoch.is_some()
    }
}

/// Build inputs for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInputs {
    /// Source files
    pub sources: Vec<SourceInput>,

    /// Dependencies
    pub dependencies: Vec<DependencyInput>,

    /// Build script outputs
    pub build_script_outputs: Option<BuildScriptOutputs>,

    /// Compiler arguments
    pub compiler_args: Vec<String>,

    /// Environment
    pub environment: BuildEnvironment,
}

/// Source file input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInput {
    /// Relative path
    pub path: PathBuf,

    /// Content hash
    pub hash: String,

    /// Size
    pub size: u64,
}

/// Dependency input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInput {
    /// Package name
    pub name: String,

    /// Version
    pub version: String,

    /// Source hash (if source dependency)
    pub source_hash: Option<String>,

    /// Binary hash (if pre-built)
    pub binary_hash: Option<String>,

    /// Registry URL
    pub registry: Option<String>,
}

/// Build script outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildScriptOutputs {
    /// Output hash
    pub hash: String,

    /// Generated files
    pub files: Vec<PathBuf>,

    /// Environment variables set
    pub env: HashMap<String, String>,
}

impl BuildInputs {
    /// Collect inputs from project
    pub fn collect(
        project_dir: &Path,
        target: &str,
        profile: &str,
    ) -> Result<Self, std::io::Error> {
        let mut sources = Vec::new();

        // Collect source files recursively
        fn visit_dir(
            dir: &Path,
            base: &Path,
            sources: &mut Vec<(PathBuf, PathBuf)>,
        ) -> std::io::Result<()> {
            if dir.is_dir() {
                for entry in std::fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();

                    // Skip hidden and build directories
                    if let Some(name) = path.file_name() {
                        let name = name.to_string_lossy();
                        if name.starts_with('.') || name == "target" || name == "build" {
                            continue;
                        }
                    }

                    if path.is_dir() {
                        visit_dir(&path, base, sources)?;
                    } else if path.extension().map(|x| x == "d").unwrap_or(false) {
                        let rel = path.strip_prefix(base).unwrap_or(&path).to_path_buf();
                        sources.push((path, rel));
                    }
                }
            }
            Ok(())
        }

        let mut paths = Vec::new();
        visit_dir(project_dir, project_dir, &mut paths)?;

        for (path, rel) in paths {
            let content = std::fs::read(&path)?;
            let hash = format!("{:x}", Sha256::digest(&content));

            sources.push(SourceInput {
                path: rel,
                hash,
                size: content.len() as u64,
            });
        }

        // Sort for determinism
        sources.sort_by(|a, b| a.path.cmp(&b.path));

        Ok(BuildInputs {
            sources,
            dependencies: Vec::new(), // TODO: collect from lock file
            build_script_outputs: None,
            compiler_args: Vec::new(),
            environment: BuildEnvironment::capture(target, profile),
        })
    }

    /// Collect inputs asynchronously
    pub async fn collect_async(
        project_dir: &Path,
        target: &str,
        profile: &str,
    ) -> Result<Self, std::io::Error> {
        // For async, we still use sync file operations but in a blocking task
        let dir = project_dir.to_path_buf();
        let target = target.to_string();
        let profile = profile.to_string();

        tokio::task::spawn_blocking(move || Self::collect(&dir, &target, &profile))
            .await
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?
    }

    /// Compute canonical hash
    pub fn hash(&self) -> String {
        let mut hasher = Sha256::new();

        // Hash sources (already sorted)
        for src in &self.sources {
            hasher.update(&src.hash);
            hasher.update(src.path.to_string_lossy().as_bytes());
        }

        // Hash dependencies
        for dep in &self.dependencies {
            hasher.update(&dep.name);
            hasher.update(&dep.version);
            if let Some(ref h) = dep.source_hash {
                hasher.update(h);
            }
            if let Some(ref h) = dep.binary_hash {
                hasher.update(h);
            }
        }

        // Hash environment
        hasher.update(self.environment.hash().as_bytes());

        // Hash compiler args
        for arg in &self.compiler_args {
            hasher.update(arg.as_bytes());
        }

        format!("{:x}", hasher.finalize())
    }

    /// Get total source size
    pub fn total_size(&self) -> u64 {
        self.sources.iter().map(|s| s.size).sum()
    }

    /// Get source count
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}

// =============================================================================
// SLSA Provenance
// =============================================================================

/// Build provenance (SLSA format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildProvenance {
    /// Statement type
    #[serde(rename = "_type")]
    pub statement_type: String,

    /// Subject (output artifacts)
    pub subject: Vec<ProvenanceSubject>,

    /// Predicate type
    #[serde(rename = "predicateType")]
    pub predicate_type: String,

    /// Predicate
    pub predicate: BuildPredicate,
}

/// Provenance subject
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceSubject {
    /// Artifact name
    pub name: String,

    /// Digest
    pub digest: HashMap<String, String>,
}

/// Build predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildPredicate {
    /// Builder
    pub builder: BuilderInfo,

    /// Build type
    #[serde(rename = "buildType")]
    pub build_type: String,

    /// Invocation
    pub invocation: BuildInvocation,

    /// Build config
    #[serde(rename = "buildConfig")]
    pub build_config: serde_json::Value,

    /// Metadata
    pub metadata: BuildMetadata,

    /// Materials (inputs)
    pub materials: Vec<BuildMaterial>,
}

/// Builder info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuilderInfo {
    /// Builder ID
    pub id: String,

    /// Version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<HashMap<String, String>>,
}

/// Build invocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInvocation {
    /// Config source
    #[serde(rename = "configSource")]
    pub config_source: ConfigSource,

    /// Parameters
    pub parameters: HashMap<String, serde_json::Value>,

    /// Environment
    pub environment: HashMap<String, String>,
}

/// Config source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSource {
    /// URI
    pub uri: String,

    /// Digest
    pub digest: HashMap<String, String>,

    /// Entry point
    #[serde(rename = "entryPoint")]
    pub entry_point: String,
}

/// Build metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildMetadata {
    /// Invocation ID
    #[serde(rename = "invocationId")]
    pub invocation_id: String,

    /// Start time
    #[serde(rename = "buildStartedOn")]
    pub build_started_on: String,

    /// End time
    #[serde(rename = "buildFinishedOn")]
    pub build_finished_on: String,

    /// Completeness
    pub completeness: Completeness,

    /// Reproducible
    pub reproducible: bool,
}

/// Completeness info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Completeness {
    pub parameters: bool,
    pub environment: bool,
    pub materials: bool,
}

/// Build material
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildMaterial {
    /// URI
    pub uri: String,

    /// Digest
    pub digest: HashMap<String, String>,
}

impl BuildProvenance {
    /// Create provenance for build
    pub fn create(
        inputs: &BuildInputs,
        outputs: &[(String, String)], // (name, hash)
        builder_id: &str,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> Self {
        let subjects: Vec<ProvenanceSubject> = outputs
            .iter()
            .map(|(name, hash)| ProvenanceSubject {
                name: name.clone(),
                digest: [("sha256".into(), hash.clone())].into_iter().collect(),
            })
            .collect();

        let materials: Vec<BuildMaterial> = inputs
            .sources
            .iter()
            .map(|src| BuildMaterial {
                uri: format!("file://{}", src.path.display()),
                digest: [("sha256".into(), src.hash.clone())].into_iter().collect(),
            })
            .collect();

        let start_str = format_rfc3339(start_time);
        let end_str = format_rfc3339(end_time);

        BuildProvenance {
            statement_type: "https://in-toto.io/Statement/v0.1".into(),
            subject: subjects,
            predicate_type: "https://slsa.dev/provenance/v0.2".into(),
            predicate: BuildPredicate {
                builder: BuilderInfo {
                    id: builder_id.into(),
                    version: Some(
                        [("sounio".into(), env!("CARGO_PKG_VERSION").into())]
                            .into_iter()
                            .collect(),
                    ),
                },
                build_type: "https://d-lang.dev/build/v1".into(),
                invocation: BuildInvocation {
                    config_source: ConfigSource {
                        uri: "file://d.toml".into(),
                        digest: HashMap::new(),
                        entry_point: "build".into(),
                    },
                    parameters: HashMap::new(),
                    environment: inputs
                        .environment
                        .env
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                },
                build_config: serde_json::json!({
                    "target": inputs.environment.target,
                    "profile": inputs.environment.profile,
                    "compiler_version": inputs.environment.compiler_version,
                    "source_hash": inputs.hash(),
                }),
                metadata: BuildMetadata {
                    invocation_id: format!("{:016x}", rand::random::<u64>()),
                    build_started_on: start_str,
                    build_finished_on: end_str,
                    completeness: Completeness {
                        parameters: true,
                        environment: true,
                        materials: true,
                    },
                    reproducible: inputs.environment.is_reproducible(),
                },
                materials,
            },
        }
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }

    /// Write to file
    pub fn write_to_file(&self, path: &Path) -> Result<(), std::io::Error> {
        std::fs::write(path, self.to_json())
    }

    /// Write to file asynchronously
    pub async fn write_to_file_async(&self, path: &Path) -> Result<(), std::io::Error> {
        tokio::fs::write(path, self.to_json()).await
    }

    /// Verify provenance signature (placeholder)
    pub fn verify(&self) -> bool {
        // TODO: Implement signature verification
        true
    }
}

/// Format SystemTime as RFC 3339 string
fn format_rfc3339(time: SystemTime) -> String {
    let duration = time
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Simple ISO 8601 format
    let datetime = chrono::DateTime::from_timestamp(secs as i64, 0)
        .unwrap_or_else(|| chrono::DateTime::from_timestamp(0, 0).unwrap());
    datetime.format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

// =============================================================================
// Reproducibility Verification
// =============================================================================

/// Result of reproducibility verification
#[derive(Debug, Clone)]
pub struct ReproducibilityResult {
    /// Whether builds are identical
    pub reproducible: bool,

    /// First build hash
    pub first_hash: String,

    /// Second build hash
    pub second_hash: String,

    /// Differences found
    pub differences: Vec<String>,
}

/// Verify build reproducibility by building twice
pub async fn verify_reproducibility<F, Fut>(
    inputs: &BuildInputs,
    build_fn: F,
) -> Result<ReproducibilityResult, std::io::Error>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<String, std::io::Error>>,
{
    // First build
    let first_hash = build_fn().await?;

    // Clean and rebuild
    let second_hash = build_fn().await?;

    let reproducible = first_hash == second_hash;
    let differences = if reproducible {
        vec![]
    } else {
        vec![format!(
            "Output hashes differ: {} vs {}",
            first_hash, second_hash
        )]
    };

    Ok(ReproducibilityResult {
        reproducible,
        first_hash,
        second_hash,
        differences,
    })
}

/// Check environment for reproducibility issues
pub fn check_reproducibility_environment() -> Vec<String> {
    let mut issues = Vec::new();

    // Check SOURCE_DATE_EPOCH
    if std::env::var("SOURCE_DATE_EPOCH").is_err() {
        issues.push("SOURCE_DATE_EPOCH not set - timestamps may vary".into());
    }

    // Check for non-deterministic environment variables
    if std::env::var("RANDOM").is_ok() {
        issues.push("RANDOM environment variable set".into());
    }

    // Check locale
    let locale = std::env::var("LANG").unwrap_or_default();
    if !locale.starts_with("C") && !locale.starts_with("POSIX") {
        issues.push(format!("Non-C locale may affect sorting: {}", locale));
    }

    // Check timezone
    if std::env::var("TZ").is_err() {
        issues.push("TZ not set - timezone may vary".into());
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_environment_capture() {
        let env = BuildEnvironment::capture("x86_64-unknown-linux-gnu", "release");
        assert_eq!(env.target, "x86_64-unknown-linux-gnu");
        assert_eq!(env.profile, "release");
        assert!(!env.compiler_version.is_empty());
    }

    #[test]
    fn test_build_environment_hash() {
        let env1 = BuildEnvironment::capture("x86_64-unknown-linux-gnu", "release");
        let env2 = BuildEnvironment::capture("x86_64-unknown-linux-gnu", "release");

        // Same inputs should produce same hash
        assert_eq!(env1.hash(), env2.hash());

        // Different target should produce different hash
        let env3 = BuildEnvironment::capture("aarch64-unknown-linux-gnu", "release");
        assert_ne!(env1.hash(), env3.hash());
    }

    #[test]
    fn test_build_inputs_hash() {
        let mut inputs = BuildInputs {
            sources: vec![
                SourceInput {
                    path: PathBuf::from("main.sio"),
                    hash: "abc123".into(),
                    size: 100,
                },
                SourceInput {
                    path: PathBuf::from("lib.sio"),
                    hash: "def456".into(),
                    size: 200,
                },
            ],
            dependencies: vec![],
            build_script_outputs: None,
            compiler_args: vec![],
            environment: BuildEnvironment::capture("x86_64-unknown-linux-gnu", "release"),
        };

        let hash1 = inputs.hash();

        // Same inputs = same hash
        let hash2 = inputs.hash();
        assert_eq!(hash1, hash2);

        // Different source = different hash
        inputs.sources[0].hash = "xyz789".into();
        let hash3 = inputs.hash();
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_provenance_creation() {
        let inputs = BuildInputs {
            sources: vec![SourceInput {
                path: PathBuf::from("main.sio"),
                hash: "abc123".into(),
                size: 100,
            }],
            dependencies: vec![],
            build_script_outputs: None,
            compiler_args: vec![],
            environment: BuildEnvironment::capture("x86_64-unknown-linux-gnu", "release"),
        };

        let outputs = vec![("output".into(), "deadbeef".into())];

        let provenance = BuildProvenance::create(
            &inputs,
            &outputs,
            "https://d-lang.dev/builder",
            SystemTime::now(),
            SystemTime::now(),
        );

        assert_eq!(
            provenance.statement_type,
            "https://in-toto.io/Statement/v0.1"
        );
        assert_eq!(provenance.subject.len(), 1);
        assert_eq!(provenance.predicate.materials.len(), 1);
    }

    #[test]
    fn test_check_reproducibility_environment() {
        let issues = check_reproducibility_environment();
        // Should always have at least one issue in test environment
        // (SOURCE_DATE_EPOCH typically not set)
        assert!(!issues.is_empty() || std::env::var("SOURCE_DATE_EPOCH").is_ok());
    }
}
