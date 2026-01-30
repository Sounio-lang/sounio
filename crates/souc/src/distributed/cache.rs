//! Remote build cache
//!
//! This module implements a distributed cache for build artifacts,
//! supporting both client and server components.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Server URL
    pub url: String,

    /// Authentication token
    pub token: Option<String>,

    /// Read-only mode
    pub read_only: bool,

    /// Timeout in seconds
    pub timeout_secs: u64,

    /// Maximum cache entry size
    pub max_entry_size: usize,

    /// Enable compression
    pub compression: bool,

    /// Local fallback directory
    pub local_fallback: Option<PathBuf>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        CacheConfig {
            url: "http://localhost:9877".into(),
            token: None,
            read_only: false,
            timeout_secs: 30,
            max_entry_size: 100 * 1024 * 1024, // 100MB
            compression: true,
            local_fallback: None,
        }
    }
}

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Entry key
    pub key: String,

    /// Content hash
    pub hash: String,

    /// Size in bytes
    pub size: usize,

    /// Creation time (Unix timestamp)
    pub created_at: u64,

    /// Last access time (Unix timestamp)
    pub accessed_at: u64,

    /// Access count
    pub access_count: u64,

    /// Compiler version
    pub compiler_version: String,

    /// Target
    pub target: String,

    /// Entry type
    pub entry_type: CacheEntryType,

    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl CacheMetadata {
    /// Create new metadata
    pub fn new(
        key: &str,
        hash: &str,
        size: usize,
        target: &str,
        entry_type: CacheEntryType,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        CacheMetadata {
            key: key.to_string(),
            hash: hash.to_string(),
            size,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            target: target.to_string(),
            entry_type,
            metadata: HashMap::new(),
        }
    }
}

/// Cache entry type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheEntryType {
    Object,
    Executable,
    Library,
    StaticLib,
    DynamicLib,
    Metadata,
    TestResult,
    DocBundle,
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub stores: u64,
    pub store_failures: u64,
    pub bytes_downloaded: u64,
    pub bytes_uploaded: u64,
}

/// Cache client for remote cache access
pub struct CacheClient {
    /// Configuration
    config: CacheConfig,

    /// Statistics
    stats: Mutex<CacheStats>,

    /// Local cache (fallback)
    local_cache: Option<LocalCache>,
}

impl CacheClient {
    /// Create new cache client
    pub fn new(config: CacheConfig) -> Self {
        let local_cache = config.local_fallback.as_ref().map(|path| {
            LocalCache::new(path.clone(), config.max_entry_size * 100) // 10GB default
        });

        CacheClient {
            config,
            stats: Mutex::new(CacheStats::default()),
            local_cache,
        }
    }

    /// Check if entry exists
    pub async fn contains(&self, key: &str) -> Result<bool, CacheError> {
        // Check local first
        if let Some(ref local) = self.local_cache {
            if local.contains(key) {
                return Ok(true);
            }
        }

        // Check remote
        let url = format!("{}/cache/{}", self.config.url, key);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build()
            .map_err(CacheError::Http)?;

        let mut request = client.head(&url);
        if let Some(ref token) = self.config.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await.map_err(CacheError::Http)?;

        Ok(response.status().is_success())
    }

    /// Get entry metadata
    pub async fn metadata(&self, key: &str) -> Result<Option<CacheMetadata>, CacheError> {
        let url = format!("{}/cache/{}/metadata", self.config.url, key);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build()
            .map_err(CacheError::Http)?;

        let mut request = client.get(&url);
        if let Some(ref token) = self.config.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await.map_err(CacheError::Http)?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }

        if !response.status().is_success() {
            return Err(CacheError::Server(response.status().to_string()));
        }

        let metadata: CacheMetadata = response.json().await.map_err(CacheError::Http)?;
        Ok(Some(metadata))
    }

    /// Get entry data
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, CacheError> {
        // Check local first
        if let Some(ref local) = self.local_cache {
            if let Some(data) = local.get(key)? {
                self.stats.lock().unwrap().hits += 1;
                return Ok(Some(data));
            }
        }

        // Fetch from remote
        let url = format!("{}/cache/{}", self.config.url, key);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build()
            .map_err(CacheError::Http)?;

        let mut request = client.get(&url);
        if let Some(ref token) = self.config.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }
        if self.config.compression {
            request = request.header("Accept-Encoding", "gzip, zstd");
        }

        let response = request.send().await.map_err(CacheError::Http)?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            self.stats.lock().unwrap().misses += 1;
            return Ok(None);
        }

        if !response.status().is_success() {
            return Err(CacheError::Server(response.status().to_string()));
        }

        let bytes = response.bytes().await.map_err(CacheError::Http)?;
        let data = bytes.to_vec();

        // Store locally for next time
        if let Some(ref local) = self.local_cache {
            let _ = local.store(key, &data);
        }

        let mut stats = self.stats.lock().unwrap();
        stats.hits += 1;
        stats.bytes_downloaded += data.len() as u64;

        Ok(Some(data))
    }

    /// Get entry to file
    pub async fn get_to_file(&self, key: &str, path: &Path) -> Result<bool, CacheError> {
        if let Some(data) = self.get(key).await? {
            tokio::fs::write(path, data).await.map_err(CacheError::Io)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Store entry
    pub async fn store(
        &self,
        key: &str,
        data: &[u8],
        metadata: CacheMetadata,
    ) -> Result<(), CacheError> {
        if self.config.read_only {
            return Ok(());
        }

        if data.len() > self.config.max_entry_size {
            return Err(CacheError::TooLarge(data.len()));
        }

        // Store locally
        if let Some(ref local) = self.local_cache {
            local.store(key, data)?;
        }

        // Store remotely
        let url = format!("{}/cache/{}", self.config.url, key);

        // Compress if enabled
        let body = if self.config.compression {
            use flate2::Compression;
            use flate2::write::GzEncoder;
            use std::io::Write;

            let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(data).map_err(CacheError::Io)?;
            encoder.finish().map_err(CacheError::Io)?
        } else {
            data.to_vec()
        };

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build()
            .map_err(CacheError::Http)?;

        let mut request = client
            .put(&url)
            .body(body.clone())
            .header("Content-Type", "application/octet-stream")
            .header(
                "X-Cache-Metadata",
                serde_json::to_string(&metadata).map_err(CacheError::Json)?,
            );

        if let Some(ref token) = self.config.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }
        if self.config.compression {
            request = request.header("Content-Encoding", "gzip");
        }

        let response = request.send().await.map_err(CacheError::Http)?;

        if !response.status().is_success() {
            self.stats.lock().unwrap().store_failures += 1;
            return Err(CacheError::Server(response.status().to_string()));
        }

        let mut stats = self.stats.lock().unwrap();
        stats.stores += 1;
        stats.bytes_uploaded += body.len() as u64;

        Ok(())
    }

    /// Store file
    pub async fn store_file(
        &self,
        key: &str,
        path: &Path,
        metadata: CacheMetadata,
    ) -> Result<(), CacheError> {
        let data = tokio::fs::read(path).await.map_err(CacheError::Io)?;
        self.store(key, &data, metadata).await
    }

    /// Delete entry
    pub async fn delete(&self, key: &str) -> Result<bool, CacheError> {
        if self.config.read_only {
            return Ok(false);
        }

        // Delete locally
        if let Some(ref local) = self.local_cache {
            local.delete(key)?;
        }

        // Delete remotely
        let url = format!("{}/cache/{}", self.config.url, key);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build()
            .map_err(CacheError::Http)?;

        let mut request = client.delete(&url);
        if let Some(ref token) = self.config.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await.map_err(CacheError::Http)?;

        Ok(response.status().is_success())
    }

    /// List entries by prefix
    pub async fn list(&self, prefix: &str) -> Result<Vec<CacheMetadata>, CacheError> {
        let url = format!("{}/cache?prefix={}", self.config.url, prefix);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build()
            .map_err(CacheError::Http)?;

        let mut request = client.get(&url);
        if let Some(ref token) = self.config.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request.send().await.map_err(CacheError::Http)?;

        if !response.status().is_success() {
            return Err(CacheError::Server(response.status().to_string()));
        }

        let entries: Vec<CacheMetadata> = response.json().await.map_err(CacheError::Http)?;
        Ok(entries)
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.lock().unwrap();
        let total = stats.hits + stats.misses;
        if total == 0 {
            0.0
        } else {
            stats.hits as f64 / total as f64
        }
    }
}

/// Local file-based cache
pub struct LocalCache {
    /// Storage directory
    storage_dir: PathBuf,

    /// Maximum cache size
    max_size: usize,

    /// Current size
    current_size: AtomicUsize,

    /// Index
    index: Mutex<HashMap<String, LocalCacheEntry>>,
}

/// Local cache entry
struct LocalCacheEntry {
    path: PathBuf,
    size: usize,
    accessed_at: u64,
}

impl LocalCache {
    /// Create new local cache
    pub fn new(storage_dir: PathBuf, max_size: usize) -> Self {
        let _ = std::fs::create_dir_all(&storage_dir);

        LocalCache {
            storage_dir,
            max_size,
            current_size: AtomicUsize::new(0),
            index: Mutex::new(HashMap::new()),
        }
    }

    /// Check if entry exists
    pub fn contains(&self, key: &str) -> bool {
        self.index.lock().unwrap().contains_key(key)
    }

    /// Get entry
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>, CacheError> {
        let mut index = self.index.lock().unwrap();

        if let Some(entry) = index.get_mut(key) {
            entry.accessed_at = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let data = std::fs::read(&entry.path).map_err(CacheError::Io)?;
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }

    /// Store entry
    pub fn store(&self, key: &str, data: &[u8]) -> Result<(), CacheError> {
        // Check size
        let current = self.current_size.load(Ordering::Relaxed);
        if current + data.len() > self.max_size {
            self.evict(data.len())?;
        }

        // Write file
        let hash = format!("{:x}", Sha256::digest(key.as_bytes()));
        let path = self.storage_dir.join(&hash[..2]).join(&hash);

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(CacheError::Io)?;
        }

        std::fs::write(&path, data).map_err(CacheError::Io)?;

        // Update index
        let mut index = self.index.lock().unwrap();
        index.insert(
            key.to_string(),
            LocalCacheEntry {
                path,
                size: data.len(),
                accessed_at: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        );

        self.current_size.fetch_add(data.len(), Ordering::Relaxed);

        Ok(())
    }

    /// Delete entry
    pub fn delete(&self, key: &str) -> Result<bool, CacheError> {
        let mut index = self.index.lock().unwrap();

        if let Some(entry) = index.remove(key) {
            let _ = std::fs::remove_file(&entry.path);
            self.current_size.fetch_sub(entry.size, Ordering::Relaxed);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Evict entries to free space (LRU)
    fn evict(&self, needed: usize) -> Result<(), CacheError> {
        let mut index = self.index.lock().unwrap();

        // Sort by access time
        let mut entries: Vec<_> = index
            .iter()
            .map(|(k, v)| (k.clone(), v.accessed_at, v.size))
            .collect();
        entries.sort_by_key(|(_, t, _)| *t);

        let mut freed = 0;
        for (key, _, size) in entries {
            if freed >= needed {
                break;
            }

            if let Some(entry) = index.remove(&key) {
                let _ = std::fs::remove_file(&entry.path);
                self.current_size.fetch_sub(entry.size, Ordering::Relaxed);
                freed += size;
            }
        }

        Ok(())
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.current_size.load(Ordering::Relaxed)
    }

    /// Get entry count
    pub fn len(&self) -> usize {
        self.index.lock().unwrap().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// =============================================================================
// Cache Server
// =============================================================================

/// Cache server state
pub struct CacheServer {
    /// Storage directory
    storage_dir: PathBuf,

    /// Maximum cache size
    max_size: usize,

    /// Current size
    current_size: AtomicUsize,

    /// Index
    index: tokio::sync::RwLock<HashMap<String, CacheMetadata>>,

    /// Statistics
    stats: CacheServerStats,
}

/// Server statistics
struct CacheServerStats {
    requests: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    bytes_in: AtomicU64,
    bytes_out: AtomicU64,
}

impl Default for CacheServerStats {
    fn default() -> Self {
        CacheServerStats {
            requests: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            bytes_in: AtomicU64::new(0),
            bytes_out: AtomicU64::new(0),
        }
    }
}

impl CacheServer {
    /// Create new cache server
    pub fn new(storage_dir: PathBuf, max_size: usize) -> std::sync::Arc<Self> {
        let _ = std::fs::create_dir_all(&storage_dir);

        std::sync::Arc::new(CacheServer {
            storage_dir,
            max_size,
            current_size: AtomicUsize::new(0),
            index: tokio::sync::RwLock::new(HashMap::new()),
            stats: CacheServerStats::default(),
        })
    }

    /// Start the cache server
    pub async fn start(self: std::sync::Arc<Self>, addr: &str) -> Result<(), std::io::Error> {
        use axum::{
            Json, Router,
            body::Bytes,
            extract::{Path as AxumPath, Query, State},
            http::{HeaderMap, StatusCode},
            response::IntoResponse,
            routing::{delete, get, head, put},
        };

        let router = Router::new()
            .route("/cache/:key", get(Self::handle_get))
            .route("/cache/:key", put(Self::handle_put))
            .route("/cache/:key", delete(Self::handle_delete))
            .route("/cache/:key", head(Self::handle_head))
            .route("/cache/:key/metadata", get(Self::handle_get_metadata))
            .route("/cache", get(Self::handle_list))
            .route("/stats", get(Self::handle_stats))
            .with_state(self);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        println!("Cache server listening on {}", addr);
        axum::serve(listener, router).await?;
        Ok(())
    }

    async fn handle_get(
        axum::extract::State(server): axum::extract::State<std::sync::Arc<CacheServer>>,
        axum::extract::Path(key): axum::extract::Path<String>,
    ) -> Result<Vec<u8>, axum::http::StatusCode> {
        server.stats.requests.fetch_add(1, Ordering::Relaxed);

        let path = server.key_to_path(&key);

        if !path.exists() {
            server.stats.misses.fetch_add(1, Ordering::Relaxed);
            return Err(axum::http::StatusCode::NOT_FOUND);
        }

        match tokio::fs::read(&path).await {
            Ok(data) => {
                server.stats.hits.fetch_add(1, Ordering::Relaxed);
                server
                    .stats
                    .bytes_out
                    .fetch_add(data.len() as u64, Ordering::Relaxed);

                // Update access time
                if let Some(meta) = server.index.write().await.get_mut(&key) {
                    meta.accessed_at = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    meta.access_count += 1;
                }

                Ok(data)
            }
            Err(_) => Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    async fn handle_put(
        axum::extract::State(server): axum::extract::State<std::sync::Arc<CacheServer>>,
        axum::extract::Path(key): axum::extract::Path<String>,
        headers: axum::http::HeaderMap,
        body: axum::body::Bytes,
    ) -> Result<axum::http::StatusCode, axum::http::StatusCode> {
        server.stats.requests.fetch_add(1, Ordering::Relaxed);
        server
            .stats
            .bytes_in
            .fetch_add(body.len() as u64, Ordering::Relaxed);

        // Parse metadata from header
        let metadata: CacheMetadata = if let Some(meta_header) = headers.get("X-Cache-Metadata") {
            match serde_json::from_slice(meta_header.as_bytes()) {
                Ok(m) => m,
                Err(_) => {
                    return Err(axum::http::StatusCode::BAD_REQUEST);
                }
            }
        } else {
            CacheMetadata::new(
                &key,
                &format!("{:x}", Sha256::digest(&body)),
                body.len(),
                "unknown",
                CacheEntryType::Object,
            )
        };

        // Check size
        let current = server.current_size.load(Ordering::Relaxed);
        if current + body.len() > server.max_size {
            server.evict(body.len()).await;
        }

        // Write file
        let path = server.key_to_path(&key);
        if let Some(parent) = path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }

        match tokio::fs::write(&path, &body).await {
            Ok(_) => {
                server.current_size.fetch_add(body.len(), Ordering::Relaxed);
                server.index.write().await.insert(key, metadata);
                Ok(axum::http::StatusCode::CREATED)
            }
            Err(_) => Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    async fn handle_delete(
        axum::extract::State(server): axum::extract::State<std::sync::Arc<CacheServer>>,
        axum::extract::Path(key): axum::extract::Path<String>,
    ) -> axum::http::StatusCode {
        server.stats.requests.fetch_add(1, Ordering::Relaxed);

        if let Some(meta) = server.index.write().await.remove(&key) {
            server.current_size.fetch_sub(meta.size, Ordering::Relaxed);
        }

        let path = server.key_to_path(&key);
        match tokio::fs::remove_file(&path).await {
            Ok(_) => axum::http::StatusCode::OK,
            Err(_) => axum::http::StatusCode::NOT_FOUND,
        }
    }

    async fn handle_head(
        axum::extract::State(server): axum::extract::State<std::sync::Arc<CacheServer>>,
        axum::extract::Path(key): axum::extract::Path<String>,
    ) -> axum::http::StatusCode {
        server.stats.requests.fetch_add(1, Ordering::Relaxed);

        let index = server.index.read().await;
        if index.contains_key(&key) {
            axum::http::StatusCode::OK
        } else {
            axum::http::StatusCode::NOT_FOUND
        }
    }

    async fn handle_get_metadata(
        axum::extract::State(server): axum::extract::State<std::sync::Arc<CacheServer>>,
        axum::extract::Path(key): axum::extract::Path<String>,
    ) -> Result<axum::Json<CacheMetadata>, axum::http::StatusCode> {
        server.stats.requests.fetch_add(1, Ordering::Relaxed);

        let index = server.index.read().await;
        match index.get(&key) {
            Some(meta) => Ok(axum::Json(meta.clone())),
            None => Err(axum::http::StatusCode::NOT_FOUND),
        }
    }

    async fn handle_list(
        axum::extract::State(server): axum::extract::State<std::sync::Arc<CacheServer>>,
        axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
    ) -> axum::Json<Vec<CacheMetadata>> {
        server.stats.requests.fetch_add(1, Ordering::Relaxed);

        let index = server.index.read().await;
        let prefix = params.get("prefix").map(|s| s.as_str()).unwrap_or("");

        let entries: Vec<CacheMetadata> = index
            .values()
            .filter(|m| m.key.starts_with(prefix))
            .cloned()
            .collect();

        axum::Json(entries)
    }

    async fn handle_stats(
        axum::extract::State(server): axum::extract::State<std::sync::Arc<CacheServer>>,
    ) -> axum::Json<serde_json::Value> {
        let index = server.index.read().await;
        let current_size = server.current_size.load(Ordering::Relaxed);

        axum::Json(serde_json::json!({
            "entries": index.len(),
            "size": current_size,
            "max_size": server.max_size,
            "utilization": current_size as f64 / server.max_size as f64,
            "requests": server.stats.requests.load(Ordering::Relaxed),
            "hits": server.stats.hits.load(Ordering::Relaxed),
            "misses": server.stats.misses.load(Ordering::Relaxed),
            "bytes_in": server.stats.bytes_in.load(Ordering::Relaxed),
            "bytes_out": server.stats.bytes_out.load(Ordering::Relaxed),
        }))
    }

    /// Convert key to file path
    fn key_to_path(&self, key: &str) -> PathBuf {
        let hash = format!("{:x}", Sha256::digest(key.as_bytes()));
        self.storage_dir.join(&hash[..2]).join(&hash)
    }

    /// Evict entries to free space
    async fn evict(&self, needed: usize) {
        let mut index = self.index.write().await;

        // Sort by access time (LRU)
        let mut entries: Vec<_> = index.iter().map(|(k, m)| (k.clone(), m.clone())).collect();
        entries.sort_by_key(|(_, m)| m.accessed_at);

        let mut freed = 0;
        let mut to_remove = Vec::new();

        for (key, meta) in entries {
            if freed >= needed {
                break;
            }
            to_remove.push(key.clone());
            freed += meta.size;
        }

        for key in to_remove {
            if let Some(meta) = index.remove(&key) {
                let path = self.key_to_path(&key);
                let _ = tokio::fs::remove_file(&path).await;
                self.current_size.fetch_sub(meta.size, Ordering::Relaxed);
            }
        }
    }
}

/// Cache error
#[derive(Debug)]
pub enum CacheError {
    Http(reqwest::Error),
    Io(std::io::Error),
    Json(serde_json::Error),
    Server(String),
    TooLarge(usize),
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::Http(e) => write!(f, "HTTP error: {}", e),
            CacheError::Io(e) => write!(f, "IO error: {}", e),
            CacheError::Json(e) => write!(f, "JSON error: {}", e),
            CacheError::Server(msg) => write!(f, "Server error: {}", msg),
            CacheError::TooLarge(size) => write!(f, "Entry too large: {} bytes", size),
        }
    }
}

impl std::error::Error for CacheError {}

impl From<reqwest::Error> for CacheError {
    fn from(e: reqwest::Error) -> Self {
        CacheError::Http(e)
    }
}

impl From<std::io::Error> for CacheError {
    fn from(e: std::io::Error) -> Self {
        CacheError::Io(e)
    }
}

impl From<serde_json::Error> for CacheError {
    fn from(e: serde_json::Error) -> Self {
        CacheError::Json(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.url, "http://localhost:9877");
        assert!(!config.read_only);
        assert!(config.compression);
    }

    #[test]
    fn test_cache_metadata_new() {
        let meta = CacheMetadata::new(
            "test-key",
            "abc123",
            100,
            "x86_64-linux",
            CacheEntryType::Object,
        );
        assert_eq!(meta.key, "test-key");
        assert_eq!(meta.hash, "abc123");
        assert_eq!(meta.size, 100);
        assert_eq!(meta.entry_type, CacheEntryType::Object);
    }

    #[test]
    fn test_local_cache() {
        let temp_dir = std::env::temp_dir().join("d-cache-test");
        let cache = LocalCache::new(temp_dir.clone(), 1024 * 1024);

        // Store
        cache.store("key1", b"hello world").unwrap();
        assert!(cache.contains("key1"));

        // Get
        let data = cache.get("key1").unwrap().unwrap();
        assert_eq!(data, b"hello world");

        // Delete
        cache.delete("key1").unwrap();
        assert!(!cache.contains("key1"));

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
