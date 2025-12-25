//! Package registry interface
//!
//! Provides abstraction over package registries for dependency resolution.

use std::collections::HashMap;
use std::path::PathBuf;

use super::manifest::{Manifest, Version};

/// Package registry trait
pub trait Registry {
    /// Get all available versions of a package
    fn get_versions(&self, name: &str) -> Result<Vec<Version>, RegistryError>;

    /// Get the manifest for a specific package version
    fn get_manifest(&self, name: &str, version: &Version) -> Result<Manifest, RegistryError>;

    /// Search for packages matching a query
    fn search(&self, query: &str, limit: usize) -> Result<Vec<PackageSummary>, RegistryError>;

    /// Download a package
    fn download(&self, name: &str, version: &Version) -> Result<PathBuf, RegistryError>;

    /// Publish a package
    fn publish(&self, manifest: &Manifest, tarball: &[u8]) -> Result<(), RegistryError>;
}

/// Package summary for search results
#[derive(Debug, Clone)]
pub struct PackageSummary {
    pub name: String,
    pub version: Version,
    pub description: Option<String>,
    pub downloads: u64,
}

/// Registry error
#[derive(Debug)]
pub enum RegistryError {
    /// Network error
    Network(String),

    /// Package not found
    NotFound(String),

    /// Authentication error
    Auth(String),

    /// Rate limited
    RateLimited,

    /// Invalid package
    Invalid(String),

    /// IO error
    Io(std::io::Error),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::Network(e) => write!(f, "Network error: {}", e),
            RegistryError::NotFound(pkg) => write!(f, "Package not found: {}", pkg),
            RegistryError::Auth(e) => write!(f, "Authentication error: {}", e),
            RegistryError::RateLimited => write!(f, "Rate limited, please try again later"),
            RegistryError::Invalid(e) => write!(f, "Invalid package: {}", e),
            RegistryError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Default registry implementation (mock for now)
pub struct DefaultRegistry {
    /// Cache of package metadata
    cache: HashMap<String, Vec<(Version, Manifest)>>,

    /// Registry URL
    url: String,

    /// Authentication token
    token: Option<String>,
}

impl DefaultRegistry {
    pub const DEFAULT_URL: &'static str = "https://registry.sounio-lang.org";

    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            url: Self::DEFAULT_URL.to_string(),
            token: None,
        }
    }

    pub fn with_url(url: &str) -> Self {
        Self {
            cache: HashMap::new(),
            url: url.to_string(),
            token: None,
        }
    }

    pub fn with_token(mut self, token: String) -> Self {
        self.token = Some(token);
        self
    }

    /// Add a package to the local cache (for testing)
    pub fn add_package(&mut self, name: &str, manifest: Manifest) {
        let version = manifest.package.version.clone();
        self.cache
            .entry(name.to_string())
            .or_default()
            .push((version, manifest));
    }
}

impl Default for DefaultRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry for DefaultRegistry {
    fn get_versions(&self, name: &str) -> Result<Vec<Version>, RegistryError> {
        // Check cache first
        if let Some(versions) = self.cache.get(name) {
            let mut result: Vec<Version> = versions.iter().map(|(v, _)| v.clone()).collect();
            result.sort();
            return Ok(result);
        }

        // In a real implementation, this would fetch from the registry
        // For now, return empty list (package not in cache)
        Ok(Vec::new())
    }

    fn get_manifest(&self, name: &str, version: &Version) -> Result<Manifest, RegistryError> {
        // Check cache first
        if let Some(versions) = self.cache.get(name) {
            for (v, manifest) in versions {
                if v == version {
                    return Ok(manifest.clone());
                }
            }
        }

        // In a real implementation, this would fetch from the registry
        Err(RegistryError::NotFound(format!("{}@{}", name, version)))
    }

    fn search(&self, query: &str, limit: usize) -> Result<Vec<PackageSummary>, RegistryError> {
        // Search in cache
        let query_lower = query.to_lowercase();
        let results: Vec<PackageSummary> = self
            .cache
            .iter()
            .filter(|(name, _)| name.to_lowercase().contains(&query_lower))
            .flat_map(|(name, versions)| {
                versions
                    .iter()
                    .map(move |(version, manifest)| PackageSummary {
                        name: name.clone(),
                        version: version.clone(),
                        description: manifest.package.description.clone(),
                        downloads: 0,
                    })
            })
            .take(limit)
            .collect();

        Ok(results)
    }

    fn download(&self, name: &str, version: &Version) -> Result<PathBuf, RegistryError> {
        // In a real implementation, this would download from the registry
        // For now, return a placeholder path
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("sounio")
            .join("registry")
            .join(name)
            .join(version.to_string());

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&cache_dir).map_err(RegistryError::Io)?;

        Ok(cache_dir)
    }

    fn publish(&self, _manifest: &Manifest, _tarball: &[u8]) -> Result<(), RegistryError> {
        // In a real implementation, this would publish to the registry
        // For now, just validate
        if self.token.is_none() {
            return Err(RegistryError::Auth("Not logged in".to_string()));
        }

        // Would upload the package
        Ok(())
    }
}

/// Local file system registry (for local development)
pub struct LocalRegistry {
    /// Root directory containing packages
    root: PathBuf,
}

impl LocalRegistry {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl Registry for LocalRegistry {
    fn get_versions(&self, name: &str) -> Result<Vec<Version>, RegistryError> {
        let pkg_dir = self.root.join(name);
        if !pkg_dir.exists() {
            return Ok(Vec::new());
        }

        let mut versions = Vec::new();
        for entry in std::fs::read_dir(&pkg_dir).map_err(RegistryError::Io)? {
            let entry = entry.map_err(RegistryError::Io)?;
            let path = entry.path();
            if path.is_dir()
                && let Some(version_str) = path.file_name().and_then(|n| n.to_str())
                && let Ok(version) = Version::parse(version_str)
            {
                versions.push(version);
            }
        }

        versions.sort();
        Ok(versions)
    }

    fn get_manifest(&self, name: &str, version: &Version) -> Result<Manifest, RegistryError> {
        let manifest_path = self
            .root
            .join(name)
            .join(version.to_string())
            .join("d.toml");

        if !manifest_path.exists() {
            return Err(RegistryError::NotFound(format!("{}@{}", name, version)));
        }

        Manifest::from_path(&manifest_path)
            .map_err(|e| RegistryError::Invalid(format!("Invalid manifest: {}", e)))
    }

    fn search(&self, query: &str, limit: usize) -> Result<Vec<PackageSummary>, RegistryError> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();

        if self.root.exists() {
            for entry in std::fs::read_dir(&self.root).map_err(RegistryError::Io)? {
                let entry = entry.map_err(RegistryError::Io)?;
                let path = entry.path();
                if path.is_dir()
                    && let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && name.to_lowercase().contains(&query_lower)
                {
                    let versions = self.get_versions(name)?;
                    if let Some(latest) = versions.last()
                        && let Ok(manifest) = self.get_manifest(name, latest)
                    {
                        results.push(PackageSummary {
                            name: name.to_string(),
                            version: latest.clone(),
                            description: manifest.package.description,
                            downloads: 0,
                        });
                    }
                }

                if results.len() >= limit {
                    break;
                }
            }
        }

        Ok(results)
    }

    fn download(&self, name: &str, version: &Version) -> Result<PathBuf, RegistryError> {
        let pkg_dir = self.root.join(name).join(version.to_string());
        if !pkg_dir.exists() {
            return Err(RegistryError::NotFound(format!("{}@{}", name, version)));
        }
        Ok(pkg_dir)
    }

    fn publish(&self, manifest: &Manifest, _tarball: &[u8]) -> Result<(), RegistryError> {
        let pkg_dir = self
            .root
            .join(&manifest.package.name)
            .join(manifest.package.version.to_string());

        std::fs::create_dir_all(&pkg_dir).map_err(RegistryError::Io)?;

        // Write manifest
        manifest
            .to_path(&pkg_dir.join("d.toml"))
            .map_err(|e| RegistryError::Invalid(format!("Failed to write manifest: {}", e)))?;

        Ok(())
    }
}

/// Get user's home directory for cache/config
mod dirs {
    use std::path::PathBuf;

    pub fn cache_dir() -> Option<PathBuf> {
        #[cfg(target_os = "windows")]
        {
            std::env::var("LOCALAPPDATA").ok().map(PathBuf::from)
        }

        #[cfg(not(target_os = "windows"))]
        {
            std::env::var("XDG_CACHE_HOME")
                .ok()
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var("HOME")
                        .ok()
                        .map(|h| PathBuf::from(h).join(".cache"))
                })
        }
    }

    pub fn config_dir() -> Option<PathBuf> {
        #[cfg(target_os = "windows")]
        {
            std::env::var("APPDATA").ok().map(PathBuf::from)
        }

        #[cfg(not(target_os = "windows"))]
        {
            std::env::var("XDG_CONFIG_HOME")
                .ok()
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var("HOME")
                        .ok()
                        .map(|h| PathBuf::from(h).join(".config"))
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pkg::manifest::Package;

    #[test]
    fn test_default_registry() {
        let mut registry = DefaultRegistry::new();

        // Add a test package
        let manifest = Manifest {
            package: Package {
                name: "test-pkg".to_string(),
                version: Version::new(1, 0, 0),
                authors: Vec::new(),
                description: Some("A test package".to_string()),
                license: None,
                license_file: None,
                repository: None,
                homepage: None,
                documentation: None,
                readme: None,
                keywords: Vec::new(),
                categories: Vec::new(),
                d_version: None,
                edition: None,
                exclude: Vec::new(),
                include: Vec::new(),
                publish: true,
                default_features: Vec::new(),
            },
            dependencies: Default::default(),
            dev_dependencies: Default::default(),
            build_dependencies: Default::default(),
            features: Default::default(),
            workspace: None,
            build: Default::default(),
            profile: Default::default(),
            binaries: Vec::new(),
            lib: None,
            example: Vec::new(),
            test: Vec::new(),
            bench: Vec::new(),
        };

        registry.add_package("test-pkg", manifest);

        // Test get_versions
        let versions = registry.get_versions("test-pkg").unwrap();
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0], Version::new(1, 0, 0));

        // Test get_manifest
        let manifest = registry
            .get_manifest("test-pkg", &Version::new(1, 0, 0))
            .unwrap();
        assert_eq!(manifest.package.name, "test-pkg");

        // Test search
        let results = registry.search("test", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "test-pkg");
    }
}
