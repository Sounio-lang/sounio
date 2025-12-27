//! Package cache management for Sounio
//!
//! Handles local caching of downloaded packages for faster builds.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::manifest::Version;
use super::registry::{RegistryError, cache_dir, home_dir};

#[cfg(feature = "pkg")]
use flate2::read::GzDecoder;
#[cfg(feature = "pkg")]
use tar::Archive;

/// Package cache manager
pub struct PackageCache {
    /// Root cache directory
    root: PathBuf,

    /// Cache index
    index: CacheIndex,
}

/// Cache index tracking all cached packages
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheIndex {
    /// Cached packages: name -> version -> entry
    pub packages: HashMap<String, HashMap<String, CacheEntry>>,

    /// Last cleanup time
    pub last_cleanup: Option<String>,

    /// Cache version for migrations
    pub version: u32,
}

/// Entry for a cached package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Package name
    pub name: String,

    /// Package version
    pub version: String,

    /// SHA256 checksum
    pub checksum: String,

    /// Path relative to cache root
    pub path: String,

    /// When this was cached
    pub cached_at: String,

    /// Last access time
    pub last_accessed: String,

    /// Size in bytes
    pub size: u64,

    /// Registry URL this came from
    pub registry: String,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of packages
    pub package_count: usize,

    /// Total number of versions
    pub version_count: usize,

    /// Total size in bytes
    pub total_size: u64,

    /// Oldest entry timestamp
    pub oldest_entry: Option<String>,

    /// Newest entry timestamp
    pub newest_entry: Option<String>,
}

impl PackageCache {
    /// Default cache directory (~/.sounio/registry/cache)
    pub fn default_path() -> Option<PathBuf> {
        home_dir().map(|h| h.join(".sounio").join("registry").join("cache"))
    }

    /// Alternate XDG-compliant cache path
    pub fn xdg_path() -> Option<PathBuf> {
        cache_dir().map(|c| c.join("sounio").join("registry"))
    }

    /// Create a new package cache at the default location
    pub fn new() -> Result<Self, RegistryError> {
        let root = Self::default_path().ok_or_else(|| {
            RegistryError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine cache directory",
            ))
        })?;

        Self::with_root(root)
    }

    /// Create a cache at a specific location
    pub fn with_root(root: PathBuf) -> Result<Self, RegistryError> {
        std::fs::create_dir_all(&root)?;

        let index_path = root.join("index.toml");
        let index = if index_path.exists() {
            let content = std::fs::read_to_string(&index_path)?;
            toml::from_str(&content)
                .map_err(|e| RegistryError::Invalid(format!("Invalid cache index: {}", e)))?
        } else {
            CacheIndex {
                version: 1,
                ..Default::default()
            }
        };

        Ok(Self { root, index })
    }

    /// Get the cache root directory
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Check if a package version is cached
    pub fn is_cached(&self, name: &str, version: &Version) -> bool {
        let version_str = version.to_string();
        if let Some(versions) = self.index.packages.get(name) {
            if let Some(entry) = versions.get(&version_str) {
                // Verify the files still exist
                let pkg_path = self.root.join(&entry.path);
                return pkg_path.exists();
            }
        }
        false
    }

    /// Get path to a cached package
    pub fn get_path(&self, name: &str, version: &Version) -> Option<PathBuf> {
        let version_str = version.to_string();
        self.index
            .packages
            .get(name)
            .and_then(|versions| versions.get(&version_str))
            .map(|entry| self.root.join(&entry.path))
    }

    /// Get a cache entry
    pub fn get_entry(&self, name: &str, version: &Version) -> Option<&CacheEntry> {
        let version_str = version.to_string();
        self.index
            .packages
            .get(name)
            .and_then(|versions| versions.get(&version_str))
    }

    /// Add a package to the cache
    pub fn add(
        &mut self,
        name: &str,
        version: &Version,
        checksum: &str,
        registry: &str,
        data: &[u8],
    ) -> Result<PathBuf, RegistryError> {
        let version_str = version.to_string();
        let rel_path = format!("{}/{}", name, version_str);
        let pkg_dir = self.root.join(&rel_path);

        // Create package directory
        std::fs::create_dir_all(&pkg_dir)?;

        // Write tarball
        let tarball_path = pkg_dir.join(format!("{}-{}.tar.gz", name, version_str));
        std::fs::write(&tarball_path, data)?;

        // Extract tarball
        self.extract_tarball(data, &pkg_dir)?;

        let now = chrono::Utc::now().to_rfc3339();

        // Add to index
        let entry = CacheEntry {
            name: name.to_string(),
            version: version_str.clone(),
            checksum: checksum.to_string(),
            path: rel_path,
            cached_at: now.clone(),
            last_accessed: now,
            size: data.len() as u64,
            registry: registry.to_string(),
        };

        self.index
            .packages
            .entry(name.to_string())
            .or_default()
            .insert(version_str, entry);

        self.save_index()?;

        Ok(pkg_dir)
    }

    /// Remove a package from the cache
    pub fn remove(&mut self, name: &str, version: &Version) -> Result<bool, RegistryError> {
        let version_str = version.to_string();

        let removed = if let Some(versions) = self.index.packages.get_mut(name) {
            if let Some(entry) = versions.remove(&version_str) {
                // Remove files
                let pkg_path = self.root.join(&entry.path);
                if pkg_path.exists() {
                    std::fs::remove_dir_all(&pkg_path)?;
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        // Clean up empty package entries
        self.index
            .packages
            .retain(|_, versions| !versions.is_empty());

        if removed {
            self.save_index()?;
        }

        Ok(removed)
    }

    /// Remove all versions of a package
    pub fn remove_package(&mut self, name: &str) -> Result<usize, RegistryError> {
        let mut removed = 0;

        if let Some(versions) = self.index.packages.remove(name) {
            for entry in versions.values() {
                let pkg_path = self.root.join(&entry.path);
                if pkg_path.exists() {
                    std::fs::remove_dir_all(&pkg_path)?;
                    removed += 1;
                }
            }
        }

        if removed > 0 {
            self.save_index()?;
        }

        Ok(removed)
    }

    /// Clear the entire cache
    pub fn clear(&mut self) -> Result<usize, RegistryError> {
        let count = self.index.packages.values().map(|v| v.len()).sum();

        // Remove all package directories but keep the cache root
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() && path.file_name() != Some("index.toml".as_ref()) {
                std::fs::remove_dir_all(&path)?;
            } else if path.is_file() && path.file_name() != Some("index.toml".as_ref()) {
                std::fs::remove_file(&path)?;
            }
        }

        self.index.packages.clear();
        self.save_index()?;

        Ok(count)
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let mut total_size = 0u64;
        let mut oldest: Option<&str> = None;
        let mut newest: Option<&str> = None;
        let mut version_count = 0;

        for versions in self.index.packages.values() {
            version_count += versions.len();
            for entry in versions.values() {
                total_size += entry.size;

                let cached_at = entry.cached_at.as_str();
                if oldest.is_none() || cached_at < oldest.unwrap() {
                    oldest = Some(cached_at);
                }
                if newest.is_none() || cached_at > newest.unwrap() {
                    newest = Some(cached_at);
                }
            }
        }

        CacheStats {
            package_count: self.index.packages.len(),
            version_count,
            total_size,
            oldest_entry: oldest.map(|s| s.to_string()),
            newest_entry: newest.map(|s| s.to_string()),
        }
    }

    /// List all cached packages
    pub fn list_packages(&self) -> Vec<(&str, Vec<&str>)> {
        self.index
            .packages
            .iter()
            .map(|(name, versions)| {
                let version_list: Vec<&str> = versions.keys().map(|s| s.as_str()).collect();
                (name.as_str(), version_list)
            })
            .collect()
    }

    /// Clean up old entries (older than max_age_days)
    pub fn cleanup(&mut self, max_age_days: u32) -> Result<usize, RegistryError> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(max_age_days as i64);
        let cutoff_str = cutoff.to_rfc3339();
        let mut removed = 0;

        let mut to_remove: Vec<(String, String)> = Vec::new();

        for (name, versions) in &self.index.packages {
            for (version, entry) in versions {
                if entry.last_accessed < cutoff_str {
                    to_remove.push((name.clone(), version.clone()));
                }
            }
        }

        for (name, version) in to_remove {
            if let Some(versions) = self.index.packages.get_mut(&name) {
                if let Some(entry) = versions.remove(&version) {
                    let pkg_path = self.root.join(&entry.path);
                    if pkg_path.exists() {
                        let _ = std::fs::remove_dir_all(&pkg_path);
                    }
                    removed += 1;
                }
            }
        }

        // Clean up empty entries
        self.index
            .packages
            .retain(|_, versions| !versions.is_empty());

        if removed > 0 {
            self.index.last_cleanup = Some(chrono::Utc::now().to_rfc3339());
            self.save_index()?;
        }

        Ok(removed)
    }

    /// Update last access time for a package
    pub fn touch(&mut self, name: &str, version: &Version) -> Result<(), RegistryError> {
        let version_str = version.to_string();
        let now = chrono::Utc::now().to_rfc3339();

        if let Some(versions) = self.index.packages.get_mut(name) {
            if let Some(entry) = versions.get_mut(&version_str) {
                entry.last_accessed = now;
                self.save_index()?;
            }
        }

        Ok(())
    }

    /// Verify checksums of cached packages
    pub fn verify(&self) -> Vec<(String, String, String)> {
        use sha2::{Digest, Sha256};

        let mut mismatches = Vec::new();

        for (name, versions) in &self.index.packages {
            for (version, entry) in versions {
                let tarball_path = self
                    .root
                    .join(&entry.path)
                    .join(format!("{}-{}.tar.gz", name, version));

                if tarball_path.exists() {
                    if let Ok(data) = std::fs::read(&tarball_path) {
                        let mut hasher = Sha256::new();
                        hasher.update(&data);
                        let actual = hex::encode(hasher.finalize());

                        if actual != entry.checksum {
                            mismatches.push((
                                name.clone(),
                                version.clone(),
                                format!("expected {}, got {}", entry.checksum, actual),
                            ));
                        }
                    }
                } else {
                    mismatches.push((name.clone(), version.clone(), "tarball missing".to_string()));
                }
            }
        }

        mismatches
    }

    /// Save the cache index
    fn save_index(&self) -> Result<(), RegistryError> {
        let index_path = self.root.join("index.toml");
        let content = toml::to_string_pretty(&self.index)
            .map_err(|e| RegistryError::Invalid(format!("Failed to serialize index: {}", e)))?;
        std::fs::write(index_path, content)?;
        Ok(())
    }

    /// Extract gzipped tarball to directory
    #[cfg(feature = "pkg")]
    fn extract_tarball(&self, data: &[u8], dest: &Path) -> Result<(), RegistryError> {
        let decoder = GzDecoder::new(data);
        let mut archive = Archive::new(decoder);

        for entry in archive
            .entries()
            .map_err(|e| RegistryError::Invalid(format!("Invalid tarball: {}", e)))?
        {
            let mut entry = entry
                .map_err(|e| RegistryError::Invalid(format!("Invalid tarball entry: {}", e)))?;

            let path = entry
                .path()
                .map_err(|e| RegistryError::Invalid(format!("Invalid path: {}", e)))?;

            // Security: skip entries that try to escape
            let path_str = path.to_string_lossy();
            if path_str.contains("..") {
                continue;
            }

            let dest_path = dest.join(&*path);

            if entry.header().entry_type().is_dir() {
                std::fs::create_dir_all(&dest_path)?;
            } else {
                if let Some(parent) = dest_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let mut file = std::fs::File::create(&dest_path)?;
                std::io::copy(&mut entry, &mut file)?;
            }
        }

        Ok(())
    }

    /// Extract gzipped tarball to directory (stub without pkg feature)
    #[cfg(not(feature = "pkg"))]
    fn extract_tarball(&self, _data: &[u8], _dest: &Path) -> Result<(), RegistryError> {
        Err(RegistryError::Invalid(
            "Package extraction requires 'pkg' feature".to_string(),
        ))
    }
}

/// Create a tarball from a directory
#[cfg(feature = "pkg")]
pub fn create_tarball(
    source_dir: &Path,
    exclude_patterns: &[&str],
) -> Result<Vec<u8>, RegistryError> {
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;
    use tar::Builder;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());

    {
        let mut builder = Builder::new(&mut encoder);

        // Walk the directory
        add_directory_to_tar(&mut builder, source_dir, source_dir, exclude_patterns)?;

        builder
            .finish()
            .map_err(|e| RegistryError::Invalid(format!("Failed to finish tarball: {}", e)))?;
    }

    encoder
        .finish()
        .map_err(|e| RegistryError::Invalid(format!("Failed to compress tarball: {}", e)))
}

#[cfg(feature = "pkg")]
fn add_directory_to_tar<W: std::io::Write>(
    builder: &mut tar::Builder<W>,
    base: &Path,
    dir: &Path,
    exclude_patterns: &[&str],
) -> Result<(), RegistryError> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let rel_path = path.strip_prefix(base).unwrap();

        // Check exclusion patterns
        let rel_str = rel_path.to_string_lossy();
        let should_exclude = exclude_patterns.iter().any(|pattern| {
            // Simple glob matching
            if pattern.starts_with('*') {
                rel_str.ends_with(&pattern[1..])
            } else if pattern.ends_with('*') {
                rel_str.starts_with(&pattern[..pattern.len() - 1])
            } else {
                rel_str == *pattern || rel_str.starts_with(&format!("{}/", pattern))
            }
        });

        if should_exclude {
            continue;
        }

        if path.is_dir() {
            add_directory_to_tar(builder, base, &path, exclude_patterns)?;
        } else {
            builder
                .append_path_with_name(&path, rel_path)
                .map_err(|e| {
                    RegistryError::Invalid(format!("Failed to add file to tarball: {}", e))
                })?;
        }
    }

    Ok(())
}

/// Get total cache size on disk
pub fn get_cache_size(path: &Path) -> Result<u64, std::io::Error> {
    let mut total = 0u64;

    if path.is_file() {
        return Ok(std::fs::metadata(path)?.len());
    }

    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            total += get_cache_size(&path)?;
        } else {
            total += std::fs::metadata(&path)?.len();
        }
    }

    Ok(total)
}

/// Format size for human display
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_operations() {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = PackageCache::with_root(temp_dir.path().to_path_buf()).unwrap();

        let version = Version::new(1, 0, 0);
        let checksum = "abc123";
        let data = b"fake tarball data";

        // Initially not cached
        assert!(!cache.is_cached("test-pkg", &version));

        // This will fail because data isn't a valid tarball, but for testing index logic:
        // We can at least test the index management
        let result = cache.add(
            "test-pkg",
            &version,
            checksum,
            "https://registry.example.com",
            data,
        );
        // The extraction will fail, but let's test what we can

        // Test stats
        let stats = cache.stats();
        assert_eq!(stats.package_count, 0); // Failed to add

        // Test list
        let packages = cache.list_packages();
        assert_eq!(packages.len(), 0);
    }

    #[test]
    fn test_cache_stats() {
        let temp_dir = TempDir::new().unwrap();
        let cache = PackageCache::with_root(temp_dir.path().to_path_buf()).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.package_count, 0);
        assert_eq!(stats.version_count, 0);
        assert_eq!(stats.total_size, 0);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512 bytes");
        assert_eq!(format_size(1536), "1.50 KB");
        assert_eq!(format_size(1_500_000), "1.43 MB");
        assert_eq!(format_size(1_500_000_000), "1.40 GB");
    }

    #[test]
    fn test_cache_index_serialization() {
        let mut index = CacheIndex::default();
        index.version = 1;

        let entry = CacheEntry {
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            checksum: "abc123".to_string(),
            path: "test/1.0.0".to_string(),
            cached_at: "2024-01-01T00:00:00Z".to_string(),
            last_accessed: "2024-01-01T00:00:00Z".to_string(),
            size: 1024,
            registry: "https://registry.example.com".to_string(),
        };

        index
            .packages
            .entry("test".to_string())
            .or_default()
            .insert("1.0.0".to_string(), entry);

        let toml_str = toml::to_string(&index).unwrap();
        let parsed: CacheIndex = toml::from_str(&toml_str).unwrap();

        assert_eq!(parsed.version, 1);
        assert!(parsed.packages.contains_key("test"));
    }
}
