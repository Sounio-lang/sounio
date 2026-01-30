//! Dependency resolution using PubGrub-inspired algorithm
//!
//! Resolves a set of compatible dependency versions.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::manifest::{Dependency, DependencyDetail, Manifest, PackageId, Version, VersionReq};
use super::registry::Registry;

/// Dependency resolver
pub struct Resolver<'a> {
    /// Package registry
    registry: &'a dyn Registry,

    /// Root package
    root: &'a Manifest,

    /// Resolution cache
    cache: HashMap<String, Vec<Version>>,

    /// Feature activations
    features: HashMap<PackageId, HashSet<String>>,
}

/// Resolution result
#[derive(Debug, Clone)]
pub struct Resolution {
    /// Resolved packages with versions
    pub packages: BTreeMap<String, ResolvedPackage>,

    /// Feature activations
    pub features: HashMap<PackageId, HashSet<String>>,
}

/// A resolved package
#[derive(Debug, Clone)]
pub struct ResolvedPackage {
    /// Package ID
    pub id: PackageId,

    /// Source location
    pub source: PackageSource,

    /// Resolved dependencies
    pub dependencies: Vec<PackageId>,

    /// Activated features
    pub features: HashSet<String>,
}

/// Package source
#[derive(Debug, Clone)]
pub enum PackageSource {
    /// Registry package
    Registry { registry: String, checksum: String },

    /// Git repository
    Git {
        url: String,
        reference: GitReference,
        resolved: String, // commit hash
    },

    /// Local path
    Path { path: std::path::PathBuf },
}

/// Git reference
#[derive(Debug, Clone)]
pub enum GitReference {
    Branch(String),
    Tag(String),
    Rev(String),
    DefaultBranch,
}

/// Resolution error
#[derive(Debug)]
pub enum ResolveError {
    /// No version satisfies requirements
    NoSolution {
        package: String,
        requirements: Vec<(String, VersionReq)>,
    },

    /// Circular dependency detected
    Cycle(Vec<String>),

    /// Package not found
    NotFound(String),

    /// Registry error
    Registry(String),

    /// Feature not found
    FeatureNotFound { package: String, feature: String },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::NoSolution {
                package,
                requirements,
            } => {
                writeln!(f, "No version of '{}' satisfies all requirements:", package)?;
                for (from, req) in requirements {
                    writeln!(f, "  {} requires {}", from, req)?;
                }
                Ok(())
            }
            ResolveError::Cycle(path) => {
                write!(f, "Circular dependency detected: {}", path.join(" -> "))
            }
            ResolveError::NotFound(pkg) => write!(f, "Package not found: {}", pkg),
            ResolveError::Registry(e) => write!(f, "Registry error: {}", e),
            ResolveError::FeatureNotFound { package, feature } => {
                write!(
                    f,
                    "Feature '{}' not found in package '{}'",
                    feature, package
                )
            }
        }
    }
}

impl std::error::Error for ResolveError {}

impl<'a> Resolver<'a> {
    pub fn new(registry: &'a dyn Registry, root: &'a Manifest) -> Self {
        Self {
            registry,
            root,
            cache: HashMap::new(),
            features: HashMap::new(),
        }
    }

    /// Resolve all dependencies
    pub fn resolve(&mut self) -> Result<Resolution, ResolveError> {
        let mut resolved = BTreeMap::new();
        let mut pending = Vec::new();
        let mut visited = HashSet::new();
        let mut path = Vec::new();

        // Add root dependencies
        for (name, dep) in &self.root.dependencies {
            pending.push((name.clone(), dep.clone(), self.root.package.name.clone()));
        }

        // Activate default features
        self.activate_default_features(&self.root.package_id());

        // Process pending dependencies
        while let Some((name, dep, from)) = pending.pop() {
            // Check for cycles
            if path.contains(&name) {
                path.push(name);
                return Err(ResolveError::Cycle(path));
            }

            // Skip if already resolved
            if visited.contains(&name) {
                continue;
            }

            path.push(name.clone());

            // Resolve version
            let version = self.resolve_version(&name, &dep, &from)?;

            // Get package manifest
            let manifest = self
                .registry
                .get_manifest(&name, &version)
                .map_err(|e| ResolveError::Registry(e.to_string()))?;

            let pkg_id = manifest.package_id();

            // Activate features
            self.activate_features(&pkg_id, &dep);

            // Add transitive dependencies
            for (trans_name, trans_dep) in &manifest.dependencies {
                if !visited.contains(trans_name) {
                    pending.push((trans_name.clone(), trans_dep.clone(), name.clone()));
                }
            }

            // Record resolved package
            let source = self.get_source(&dep, &version)?;
            let deps = manifest
                .dependencies
                .keys()
                .map(|n| PackageId {
                    name: n.clone(),
                    version: Version::new(0, 0, 0), // Placeholder - will be updated
                })
                .collect();

            resolved.insert(
                name.clone(),
                ResolvedPackage {
                    id: pkg_id.clone(),
                    source,
                    dependencies: deps,
                    features: self.features.get(&pkg_id).cloned().unwrap_or_default(),
                },
            );

            visited.insert(name);
            path.pop();
        }

        Ok(Resolution {
            packages: resolved,
            features: self.features.clone(),
        })
    }

    /// Resolve a specific version for a dependency
    fn resolve_version(
        &mut self,
        name: &str,
        dep: &Dependency,
        from: &str,
    ) -> Result<Version, ResolveError> {
        let req = match dep {
            Dependency::Simple(v) => VersionReq::parse(v).unwrap_or_default(),
            Dependency::Detailed(d) => {
                // Handle path/git dependencies
                if d.path.is_some() || d.git.is_some() {
                    return self.resolve_non_registry(name, d);
                }

                d.version
                    .as_ref()
                    .and_then(|v| VersionReq::parse(v).ok())
                    .unwrap_or_default()
            }
        };

        // Get available versions from cache or registry
        let versions = self
            .cache
            .entry(name.to_string())
            .or_insert_with(|| self.registry.get_versions(name).unwrap_or_default());

        // Find best matching version (highest that satisfies requirement)
        versions
            .iter()
            .rev()
            .find(|v| req.matches(v))
            .cloned()
            .ok_or_else(|| ResolveError::NoSolution {
                package: name.to_string(),
                requirements: vec![(from.to_string(), req)],
            })
    }

    /// Resolve non-registry dependency
    fn resolve_non_registry(
        &self,
        name: &str,
        dep: &DependencyDetail,
    ) -> Result<Version, ResolveError> {
        if let Some(ref path) = dep.path {
            // Load manifest from path
            let manifest_path = path.join("d.toml");
            let manifest = Manifest::from_path(&manifest_path)
                .map_err(|_| ResolveError::NotFound(name.to_string()))?;
            return Ok(manifest.package.version);
        }

        if dep.git.is_some() {
            // Would fetch from git and read manifest
            // For now, return placeholder
            return Ok(Version::new(0, 0, 0));
        }

        Err(ResolveError::NotFound(name.to_string()))
    }

    /// Get package source
    fn get_source(
        &self,
        dep: &Dependency,
        _version: &Version,
    ) -> Result<PackageSource, ResolveError> {
        match dep {
            Dependency::Simple(_) => Ok(PackageSource::Registry {
                registry: "https://registry.sounio-lang.org".to_string(),
                checksum: String::new(),
            }),
            Dependency::Detailed(d) => {
                if let Some(ref path) = d.path {
                    return Ok(PackageSource::Path { path: path.clone() });
                }

                if let Some(ref git) = d.git {
                    let reference = if let Some(ref branch) = d.branch {
                        GitReference::Branch(branch.clone())
                    } else if let Some(ref tag) = d.tag {
                        GitReference::Tag(tag.clone())
                    } else if let Some(ref rev) = d.rev {
                        GitReference::Rev(rev.clone())
                    } else {
                        GitReference::DefaultBranch
                    };

                    return Ok(PackageSource::Git {
                        url: git.clone(),
                        reference,
                        resolved: String::new(),
                    });
                }

                Ok(PackageSource::Registry {
                    registry: d
                        .registry
                        .clone()
                        .unwrap_or_else(|| "https://registry.sounio-lang.org".to_string()),
                    checksum: String::new(),
                })
            }
        }
    }

    /// Activate features for a package
    fn activate_features(&mut self, pkg_id: &PackageId, dep: &Dependency) {
        let features = self.features.entry(pkg_id.clone()).or_default();

        if let Dependency::Detailed(d) = dep {
            // Add explicitly requested features
            for feature in &d.features {
                features.insert(feature.clone());
            }

            // Add default features if not disabled
            if d.default_features {
                self.activate_default_features(pkg_id);
            }
        } else {
            // Simple dependency gets default features
            self.activate_default_features(pkg_id);
        }
    }

    /// Activate default features
    fn activate_default_features(&mut self, pkg_id: &PackageId) {
        let features = self.features.entry(pkg_id.clone()).or_default();
        features.insert("default".to_string());
    }
}

/// Lockfile for reproducible builds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lockfile {
    /// Lockfile version
    pub version: u32,

    /// Resolved packages
    pub package: Vec<LockedPackage>,

    /// Metadata
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

/// A locked package entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockedPackage {
    pub name: String,
    pub version: String,
    pub source: Option<String>,
    pub checksum: Option<String>,
    #[serde(default)]
    pub dependencies: Vec<String>,
}

impl Lockfile {
    pub const FILENAME: &'static str = "d.lock";
    pub const VERSION: u32 = 1;

    /// Create new lockfile from resolution
    pub fn from_resolution(resolution: &Resolution) -> Self {
        let packages = resolution
            .packages
            .values()
            .map(|pkg| LockedPackage {
                name: pkg.id.name.clone(),
                version: pkg.id.version.to_string(),
                source: Some(format!("{:?}", pkg.source)),
                checksum: match &pkg.source {
                    PackageSource::Registry { checksum, .. } => Some(checksum.clone()),
                    _ => None,
                },
                dependencies: pkg
                    .dependencies
                    .iter()
                    .map(|d| format!("{} {}", d.name, d.version))
                    .collect(),
            })
            .collect();

        Self {
            version: Self::VERSION,
            package: packages,
            metadata: BTreeMap::new(),
        }
    }

    /// Load lockfile
    pub fn load(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Save lockfile
    pub fn save(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lockfile_roundtrip() {
        let lockfile = Lockfile {
            version: 1,
            package: vec![LockedPackage {
                name: "test-pkg".to_string(),
                version: "1.0.0".to_string(),
                source: Some("registry".to_string()),
                checksum: Some("abc123".to_string()),
                dependencies: vec!["dep1 1.0.0".to_string()],
            }],
            metadata: BTreeMap::new(),
        };

        let toml_str = toml::to_string(&lockfile).unwrap();
        let parsed: Lockfile = toml::from_str(&toml_str).unwrap();

        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.package.len(), 1);
        assert_eq!(parsed.package[0].name, "test-pkg");
    }
}
