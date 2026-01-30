//! Build artifact caching with content-addressed storage.
//!
//! This module provides a caching system for compiled artifacts, allowing
//! incremental builds to reuse previously compiled results.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use super::graph::ContentHash;

/// Cache key for artifacts (content-addressed)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Source file content hash
    pub source_hash: ContentHash,

    /// Compiler version
    pub compiler_version: String,

    /// Build configuration hash
    pub config_hash: String,
}

impl CacheKey {
    /// Create a new cache key
    pub fn new(source_hash: ContentHash, compiler_version: String, config_hash: String) -> Self {
        CacheKey {
            source_hash,
            compiler_version,
            config_hash,
        }
    }

    /// Convert to a file-safe string
    pub fn to_string(&self) -> String {
        format!(
            "{}-{}-{}",
            self.source_hash.to_hex(),
            sanitize_version(&self.compiler_version),
            &self.config_hash
        )
    }
}

fn sanitize_version(version: &str) -> String {
    version.replace(['.', '-'], "_")
}

/// Cached artifact entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Cache key
    pub key: CacheKey,

    /// Path to cached artifact
    pub artifact_path: PathBuf,

    /// Size in bytes
    pub size: u64,

    /// When this entry was created
    pub created_at: SystemTime,

    /// When this entry was last accessed
    pub last_accessed: SystemTime,

    /// Access count
    pub access_count: u64,

    /// Metadata
    pub metadata: CacheMetadata,
}

impl CacheEntry {
    /// Create a new cache entry
    pub fn new(key: CacheKey, artifact_path: PathBuf, size: u64) -> Self {
        let now = SystemTime::now();
        CacheEntry {
            key,
            artifact_path,
            size,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            metadata: CacheMetadata::default(),
        }
    }

    /// Mark as accessed
    pub fn touch(&mut self) {
        self.last_accessed = SystemTime::now();
        self.access_count += 1;
    }

    /// Age of entry in seconds
    pub fn age_secs(&self) -> u64 {
        SystemTime::now()
            .duration_since(self.created_at)
            .unwrap_or_default()
            .as_secs()
    }
}

/// Metadata stored with cache entries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Compilation time in milliseconds
    pub compile_time_ms: u64,

    /// Number of warnings
    pub warnings: usize,

    /// Whether verification succeeded
    pub verified: bool,
}

/// Build artifact cache
pub struct ArtifactCache {
    /// Cache directory
    cache_dir: PathBuf,

    /// Cache entries indexed by key
    entries: HashMap<CacheKey, CacheEntry>,

    /// Cache configuration
    config: CacheConfig,

    /// Statistics
    stats: CacheStats,
}

impl ArtifactCache {
    /// Create a new artifact cache
    pub fn new(cache_dir: PathBuf) -> Self {
        ArtifactCache {
            cache_dir,
            entries: HashMap::new(),
            config: CacheConfig::default(),
            stats: CacheStats::default(),
        }
    }

    /// Set cache configuration
    pub fn with_config(mut self, config: CacheConfig) -> Self {
        self.config = config;
        self
    }

    /// Initialize cache (create directory, load index)
    pub fn init(&mut self) -> Result<(), CacheError> {
        std::fs::create_dir_all(&self.cache_dir)?;

        let index_path = self.cache_dir.join("index.bin");
        if index_path.exists() {
            self.load_index(&index_path)?;
        }

        Ok(())
    }

    /// Check if an artifact is cached
    pub fn contains(&self, key: &CacheKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Get a cached artifact
    pub fn get(&mut self, key: &CacheKey) -> Option<Vec<u8>> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.touch();
            self.stats.hits += 1;

            match std::fs::read(&entry.artifact_path) {
                Ok(data) => Some(data),
                Err(_) => {
                    // Entry exists but file is missing
                    self.stats.misses += 1;
                    None
                }
            }
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Store an artifact in the cache
    pub fn put(
        &mut self,
        key: CacheKey,
        data: &[u8],
        metadata: CacheMetadata,
    ) -> Result<(), CacheError> {
        // Check size limits
        if data.len() as u64 > self.config.max_entry_size {
            return Err(CacheError::EntrySizeExceeded {
                size: data.len() as u64,
                limit: self.config.max_entry_size,
            });
        }

        // Create artifact path
        let artifact_name = format!("{}.bin", key.to_string());
        let artifact_path = self.cache_dir.join(&artifact_name);

        // Write artifact
        std::fs::write(&artifact_path, data)?;

        // Create entry
        let mut entry = CacheEntry::new(key.clone(), artifact_path, data.len() as u64);
        entry.metadata = metadata;

        // Add to index
        self.entries.insert(key, entry);

        // Check if eviction is needed
        if self.should_evict() {
            self.evict()?;
        }

        Ok(())
    }

    /// Remove an entry from the cache
    pub fn remove(&mut self, key: &CacheKey) -> Result<(), CacheError> {
        if let Some(entry) = self.entries.remove(key) {
            std::fs::remove_file(&entry.artifact_path)?;
        }
        Ok(())
    }

    /// Check if eviction should run
    fn should_evict(&self) -> bool {
        let total_size: u64 = self.entries.values().map(|e| e.size).sum();
        total_size > self.config.max_cache_size
    }

    /// Evict entries based on policy
    fn evict(&mut self) -> Result<(), CacheError> {
        let target_size = (self.config.max_cache_size as f64 * 0.8) as u64;
        let current_size: u64 = self.entries.values().map(|e| e.size).sum();

        if current_size <= target_size {
            return Ok(());
        }

        let mut entries: Vec<_> = self.entries.values().collect();

        // Sort by eviction policy
        match self.config.eviction_policy {
            EvictionPolicy::Lru => {
                entries.sort_by_key(|e| e.last_accessed);
            }
            EvictionPolicy::Lfu => {
                entries.sort_by_key(|e| e.access_count);
            }
            EvictionPolicy::Fifo => {
                entries.sort_by_key(|e| e.created_at);
            }
            EvictionPolicy::Size => {
                entries.sort_by_key(|e| std::cmp::Reverse(e.size));
            }
        }

        let mut freed = 0u64;
        let mut to_remove = Vec::new();

        for entry in entries {
            if current_size - freed <= target_size {
                break;
            }

            to_remove.push(entry.key.clone());
            freed += entry.size;
        }

        for key in to_remove {
            self.remove(&key)?;
            self.stats.evictions += 1;
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.hits + self.stats.misses;
        if total == 0 {
            0.0
        } else {
            self.stats.hits as f64 / total as f64
        }
    }

    /// Get total cache size in bytes
    pub fn size(&self) -> u64 {
        self.entries.values().map(|e| e.size).sum()
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries
    pub fn clear(&mut self) -> Result<(), CacheError> {
        for entry in self.entries.values() {
            let _ = std::fs::remove_file(&entry.artifact_path);
        }
        self.entries.clear();
        self.stats = CacheStats::default();
        Ok(())
    }

    /// Prune invalid entries (missing files)
    pub fn prune(&mut self) -> Result<usize, CacheError> {
        let mut removed = 0;
        let mut to_remove = Vec::new();

        for (key, entry) in &self.entries {
            if !entry.artifact_path.exists() {
                to_remove.push(key.clone());
            }
        }

        for key in to_remove {
            self.entries.remove(&key);
            removed += 1;
        }

        Ok(removed)
    }

    /// Save cache index to disk
    pub fn save(&self) -> Result<(), CacheError> {
        let index_path = self.cache_dir.join("index.bin");

        let snapshot = CacheSnapshot {
            entries: self.entries.values().cloned().collect(),
            stats: self.stats.clone(),
        };

        let data = bincode::serialize(&snapshot)
            .map_err(|e| CacheError::SerializationError(e.to_string()))?;

        std::fs::write(index_path, data)?;
        Ok(())
    }

    /// Load cache index from disk
    fn load_index(&mut self, path: &Path) -> Result<(), CacheError> {
        let data = std::fs::read(path)?;

        let snapshot: CacheSnapshot = bincode::deserialize(&data)
            .map_err(|e| CacheError::SerializationError(e.to_string()))?;

        self.entries.clear();
        for entry in snapshot.entries {
            self.entries.insert(entry.key.clone(), entry);
        }

        self.stats = snapshot.stats;

        Ok(())
    }
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size in bytes (default: 1GB)
    pub max_cache_size: u64,

    /// Maximum single entry size (default: 100MB)
    pub max_entry_size: u64,

    /// Eviction policy
    pub eviction_policy: EvictionPolicy,

    /// Enable remote cache
    pub enable_remote: bool,

    /// Remote cache URL
    pub remote_url: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        CacheConfig {
            max_cache_size: 1024 * 1024 * 1024, // 1GB
            max_entry_size: 100 * 1024 * 1024,  // 100MB
            eviction_policy: EvictionPolicy::Lru,
            enable_remote: false,
            remote_url: None,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// First In First Out
    Fifo,
    /// Evict largest entries first
    Size,
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,

    /// Number of cache misses
    pub misses: u64,

    /// Number of evictions
    pub evictions: u64,
}

/// Serializable cache snapshot
#[derive(Serialize, Deserialize)]
struct CacheSnapshot {
    entries: Vec<CacheEntry>,
    stats: CacheStats,
}

/// Remote cache client for distributed builds
pub struct RemoteCache {
    /// Base URL for remote cache
    base_url: String,

    /// HTTP client
    #[cfg(feature = "remote-cache")]
    client: reqwest::blocking::Client,
}

impl RemoteCache {
    /// Create a new remote cache client
    pub fn new(base_url: String) -> Self {
        RemoteCache {
            base_url,
            #[cfg(feature = "remote-cache")]
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Check if an artifact exists in remote cache
    #[cfg(feature = "remote-cache")]
    pub fn contains(&self, key: &CacheKey) -> Result<bool, CacheError> {
        let url = format!("{}/artifacts/{}", self.base_url, key.to_string());

        let response = self
            .client
            .head(&url)
            .send()
            .map_err(|e| CacheError::RemoteError(e.to_string()))?;

        Ok(response.status().is_success())
    }

    #[cfg(not(feature = "remote-cache"))]
    pub fn contains(&self, _key: &CacheKey) -> Result<bool, CacheError> {
        Ok(false)
    }

    /// Get an artifact from remote cache
    #[cfg(feature = "remote-cache")]
    pub fn get(&self, key: &CacheKey) -> Result<Option<Vec<u8>>, CacheError> {
        let url = format!("{}/artifacts/{}", self.base_url, key.to_string());

        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|e| CacheError::RemoteError(e.to_string()))?;

        if response.status().is_success() {
            let data = response
                .bytes()
                .map_err(|e| CacheError::RemoteError(e.to_string()))?
                .to_vec();
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }

    #[cfg(not(feature = "remote-cache"))]
    pub fn get(&self, _key: &CacheKey) -> Result<Option<Vec<u8>>, CacheError> {
        Ok(None)
    }

    /// Put an artifact to remote cache
    #[cfg(feature = "remote-cache")]
    pub fn put(&self, key: &CacheKey, data: &[u8]) -> Result<(), CacheError> {
        let url = format!("{}/artifacts/{}", self.base_url, key.to_string());

        let response = self
            .client
            .put(&url)
            .body(data.to_vec())
            .send()
            .map_err(|e| CacheError::RemoteError(e.to_string()))?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(CacheError::RemoteError(format!(
                "Upload failed: {}",
                response.status()
            )))
        }
    }

    #[cfg(not(feature = "remote-cache"))]
    pub fn put(&self, _key: &CacheKey, _data: &[u8]) -> Result<(), CacheError> {
        Err(CacheError::RemoteError(
            "Remote cache not enabled".to_string(),
        ))
    }
}

/// Cache errors
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("Entry size {size} exceeds limit {limit}")]
    EntrySizeExceeded { size: u64, limit: u64 },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Remote cache error: {0}")]
    RemoteError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_key() {
        let hash = ContentHash::from_bytes(b"test");
        let key = CacheKey::new(hash, "1.0.0".to_string(), "debug".to_string());

        let key_str = key.to_string();
        assert!(key_str.contains("1_0_0"));
        assert!(key_str.contains("debug"));
    }

    #[test]
    fn test_artifact_cache() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache");

        let mut cache = ArtifactCache::new(cache_dir);
        cache.init().unwrap();

        let hash = ContentHash::from_bytes(b"source");
        let key = CacheKey::new(hash, "1.0.0".to_string(), "debug".to_string());

        // Store artifact
        let data = b"compiled artifact";
        cache
            .put(key.clone(), data, CacheMetadata::default())
            .unwrap();

        assert!(cache.contains(&key));
        assert_eq!(cache.len(), 1);

        // Retrieve artifact
        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved, data);

        // Check stats
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn test_cache_eviction() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache");

        let config = CacheConfig {
            max_cache_size: 1000, // 1KB
            max_entry_size: 500,
            eviction_policy: EvictionPolicy::Lru,
            enable_remote: false,
            remote_url: None,
        };

        let mut cache = ArtifactCache::new(cache_dir).with_config(config);
        cache.init().unwrap();

        // Add entries that exceed cache size
        for i in 0..5 {
            let hash = ContentHash::from_bytes(&format!("source{}", i).as_bytes());
            let key = CacheKey::new(hash, "1.0.0".to_string(), "debug".to_string());
            let data = vec![0u8; 300]; // 300 bytes each
            cache.put(key, &data, CacheMetadata::default()).unwrap();
        }

        // Should have triggered eviction
        assert!(cache.len() < 5);
        assert!(cache.stats().evictions > 0);
    }

    #[test]
    fn test_cache_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let cache_dir = temp_dir.path().join("cache");

        let hash = ContentHash::from_bytes(b"source");
        let key = CacheKey::new(hash, "1.0.0".to_string(), "debug".to_string());

        // Create cache and add entry
        {
            let mut cache = ArtifactCache::new(cache_dir.clone());
            cache.init().unwrap();
            cache
                .put(key.clone(), b"data", CacheMetadata::default())
                .unwrap();
            cache.save().unwrap();
        }

        // Load cache
        {
            let mut cache = ArtifactCache::new(cache_dir);
            cache.init().unwrap();
            assert!(cache.contains(&key));
        }
    }
}
