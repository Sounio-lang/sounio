//! Package registry interface
//!
//! Provides abstraction over package registries for dependency resolution.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageSummary {
    pub name: String,
    pub version: Version,
    pub description: Option<String>,
    pub downloads: u64,
}

/// Package metadata from registry API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageMetadata {
    pub name: String,
    pub versions: Vec<VersionInfo>,
    pub description: Option<String>,
    pub repository: Option<String>,
    pub homepage: Option<String>,
    pub documentation: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    pub downloads: u64,
}

/// Version info from registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub version: Version,
    pub checksum: String,
    pub yanked: bool,
    pub published_at: Option<String>,
    pub downloads: u64,
    pub size: u64,
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

    /// Checksum mismatch
    ChecksumMismatch { expected: String, actual: String },

    /// Package already exists
    AlreadyExists(String),

    /// Server error
    Server(String),
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
            RegistryError::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "Checksum mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            RegistryError::AlreadyExists(pkg) => write!(f, "Package already exists: {}", pkg),
            RegistryError::Server(e) => write!(f, "Server error: {}", e),
        }
    }
}

impl std::error::Error for RegistryError {}

impl From<std::io::Error> for RegistryError {
    fn from(e: std::io::Error) -> Self {
        RegistryError::Io(e)
    }
}

/// Registry API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

/// Search results from registry API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    pub packages: Vec<PackageSummary>,
    pub total: usize,
    pub page: usize,
    pub per_page: usize,
}

/// Publish response from registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishResponse {
    pub name: String,
    pub version: String,
    pub checksum: String,
}

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

    /// Get the registry URL
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Check if authenticated
    pub fn is_authenticated(&self) -> bool {
        self.token.is_some()
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

// Async registry client (requires 'pkg' feature)
#[cfg(feature = "pkg")]
pub mod async_client {
    use super::*;
    use sha2::{Digest, Sha256};

    /// Async HTTP registry client
    pub struct HttpRegistry {
        /// HTTP client
        client: reqwest::Client,

        /// Registry URL
        url: String,

        /// Authentication token
        token: Option<String>,
    }

    impl HttpRegistry {
        /// Create new HTTP registry client
        pub fn new(url: &str) -> Self {
            Self {
                client: reqwest::Client::builder()
                    .user_agent(concat!("souc/", env!("CARGO_PKG_VERSION")))
                    .build()
                    .expect("Failed to create HTTP client"),
                url: url.to_string(),
                token: None,
            }
        }

        /// Create with default registry URL
        pub fn default_registry() -> Self {
            Self::new(DefaultRegistry::DEFAULT_URL)
        }

        /// Set authentication token
        pub fn with_token(mut self, token: String) -> Self {
            self.token = Some(token);
            self
        }

        /// Set token reference
        pub fn set_token(&mut self, token: Option<String>) {
            self.token = token;
        }

        /// Get API URL for path
        fn api_url(&self, path: &str) -> String {
            format!("{}/api/v1{}", self.url, path)
        }

        /// Add authentication header if token is set
        fn auth_header(
            &self,
            request: reqwest::RequestBuilder,
        ) -> reqwest::RequestBuilder {
            if let Some(ref token) = self.token {
                request.header("Authorization", format!("Bearer {}", token))
            } else {
                request
            }
        }

        /// Fetch package metadata
        pub async fn fetch_package(
            &self,
            name: &str,
        ) -> Result<PackageMetadata, RegistryError> {
            let url = self.api_url(&format!("/packages/{}", name));

            let response = self
                .client
                .get(&url)
                .send()
                .await
                .map_err(|e| RegistryError::Network(e.to_string()))?;

            match response.status().as_u16() {
                200 => {
                    let api_response: ApiResponse<PackageMetadata> = response
                        .json()
                        .await
                        .map_err(|e| RegistryError::Invalid(e.to_string()))?;

                    api_response
                        .data
                        .ok_or_else(|| RegistryError::NotFound(name.to_string()))
                }
                404 => Err(RegistryError::NotFound(name.to_string())),
                401 | 403 => Err(RegistryError::Auth("Unauthorized".to_string())),
                429 => Err(RegistryError::RateLimited),
                status => Err(RegistryError::Server(format!(
                    "Unexpected status: {}",
                    status
                ))),
            }
        }

        /// Search for packages
        pub async fn search_packages(
            &self,
            query: &str,
            page: usize,
            per_page: usize,
        ) -> Result<SearchResults, RegistryError> {
            let url = self.api_url("/search");

            let response = self
                .client
                .get(&url)
                .query(&[
                    ("q", query),
                    ("page", &page.to_string()),
                    ("per_page", &per_page.to_string()),
                ])
                .send()
                .await
                .map_err(|e| RegistryError::Network(e.to_string()))?;

            match response.status().as_u16() {
                200 => {
                    let api_response: ApiResponse<SearchResults> = response
                        .json()
                        .await
                        .map_err(|e| RegistryError::Invalid(e.to_string()))?;

                    api_response
                        .data
                        .ok_or_else(|| RegistryError::Invalid("Empty response".to_string()))
                }
                429 => Err(RegistryError::RateLimited),
                status => Err(RegistryError::Server(format!(
                    "Unexpected status: {}",
                    status
                ))),
            }
        }

        /// Fetch available versions for a package
        pub async fn fetch_versions(
            &self,
            name: &str,
        ) -> Result<Vec<VersionInfo>, RegistryError> {
            let metadata = self.fetch_package(name).await?;
            Ok(metadata.versions)
        }

        /// Get latest non-yanked version
        pub async fn get_latest_version(
            &self,
            name: &str,
        ) -> Result<Version, RegistryError> {
            let versions = self.fetch_versions(name).await?;

            versions
                .into_iter()
                .filter(|v| !v.yanked)
                .max_by(|a, b| a.version.cmp(&b.version))
                .map(|v| v.version)
                .ok_or_else(|| {
                    RegistryError::NotFound(format!("No available versions for {}", name))
                })
        }

        /// Download package tarball
        pub async fn download_package(
            &self,
            name: &str,
            version: &Version,
            dest_dir: &std::path::Path,
        ) -> Result<PathBuf, RegistryError> {
            // Get version info for checksum
            let versions = self.fetch_versions(name).await?;
            let version_info = versions
                .iter()
                .find(|v| &v.version == version)
                .ok_or_else(|| {
                    RegistryError::NotFound(format!("{}@{}", name, version))
                })?;

            let expected_checksum = &version_info.checksum;

            // Download tarball
            let url = self.api_url(&format!(
                "/packages/{}/{}/download",
                name, version
            ));

            let response = self
                .auth_header(self.client.get(&url))
                .send()
                .await
                .map_err(|e| RegistryError::Network(e.to_string()))?;

            match response.status().as_u16() {
                200 => {
                    let bytes = response
                        .bytes()
                        .await
                        .map_err(|e| RegistryError::Network(e.to_string()))?;

                    // Verify checksum
                    let actual_checksum = compute_sha256(&bytes);
                    if &actual_checksum != expected_checksum {
                        return Err(RegistryError::ChecksumMismatch {
                            expected: expected_checksum.clone(),
                            actual: actual_checksum,
                        });
                    }

                    // Create destination directory
                    let pkg_dir = dest_dir.join(name).join(version.to_string());
                    std::fs::create_dir_all(&pkg_dir)?;

                    // Extract tarball
                    extract_tarball(&bytes, &pkg_dir)?;

                    Ok(pkg_dir)
                }
                404 => Err(RegistryError::NotFound(format!(
                    "{}@{}",
                    name, version
                ))),
                401 | 403 => {
                    Err(RegistryError::Auth("Unauthorized".to_string()))
                }
                429 => Err(RegistryError::RateLimited),
                status => Err(RegistryError::Server(format!(
                    "Unexpected status: {}",
                    status
                ))),
            }
        }

        /// Publish a package
        pub async fn publish_package(
            &self,
            manifest: &Manifest,
            tarball: Vec<u8>,
        ) -> Result<PublishResponse, RegistryError> {
            if self.token.is_none() {
                return Err(RegistryError::Auth(
                    "Not logged in. Run `souc login` first.".to_string(),
                ));
            }

            // Compute checksum
            let checksum = compute_sha256(&tarball);

            // Create multipart form
            let manifest_json = serde_json::to_string(manifest)
                .map_err(|e| RegistryError::Invalid(e.to_string()))?;

            let form = reqwest::multipart::Form::new()
                .text("manifest", manifest_json)
                .text("checksum", checksum.clone())
                .part(
                    "tarball",
                    reqwest::multipart::Part::bytes(tarball)
                        .file_name(format!(
                            "{}-{}.tar.gz",
                            manifest.package.name, manifest.package.version
                        ))
                        .mime_str("application/gzip")
                        .map_err(|e| RegistryError::Invalid(e.to_string()))?,
                );

            let url = self.api_url("/packages/publish");

            let response = self
                .auth_header(self.client.post(&url))
                .multipart(form)
                .send()
                .await
                .map_err(|e| RegistryError::Network(e.to_string()))?;

            match response.status().as_u16() {
                200 | 201 => {
                    let api_response: ApiResponse<PublishResponse> = response
                        .json()
                        .await
                        .map_err(|e| RegistryError::Invalid(e.to_string()))?;

                    api_response.data.ok_or_else(|| {
                        RegistryError::Invalid("Empty response".to_string())
                    })
                }
                409 => Err(RegistryError::AlreadyExists(format!(
                    "{}@{}",
                    manifest.package.name, manifest.package.version
                ))),
                401 | 403 => {
                    Err(RegistryError::Auth("Unauthorized".to_string()))
                }
                400 => {
                    let text = response.text().await.unwrap_or_default();
                    Err(RegistryError::Invalid(text))
                }
                429 => Err(RegistryError::RateLimited),
                status => Err(RegistryError::Server(format!(
                    "Unexpected status: {}",
                    status
                ))),
            }
        }

        /// Yank a package version
        pub async fn yank_version(
            &self,
            name: &str,
            version: &Version,
        ) -> Result<(), RegistryError> {
            if self.token.is_none() {
                return Err(RegistryError::Auth(
                    "Not logged in".to_string(),
                ));
            }

            let url = self.api_url(&format!(
                "/packages/{}/{}/yank",
                name, version
            ));

            let response = self
                .auth_header(self.client.delete(&url))
                .send()
                .await
                .map_err(|e| RegistryError::Network(e.to_string()))?;

            match response.status().as_u16() {
                200 | 204 => Ok(()),
                404 => Err(RegistryError::NotFound(format!(
                    "{}@{}",
                    name, version
                ))),
                401 | 403 => {
                    Err(RegistryError::Auth("Unauthorized".to_string()))
                }
                status => Err(RegistryError::Server(format!(
                    "Unexpected status: {}",
                    status
                ))),
            }
        }

        /// Unyank a package version
        pub async fn unyank_version(
            &self,
            name: &str,
            version: &Version,
        ) -> Result<(), RegistryError> {
            if self.token.is_none() {
                return Err(RegistryError::Auth(
                    "Not logged in".to_string(),
                ));
            }

            let url = self.api_url(&format!(
                "/packages/{}/{}/unyank",
                name, version
            ));

            let response = self
                .auth_header(self.client.put(&url))
                .send()
                .await
                .map_err(|e| RegistryError::Network(e.to_string()))?;

            match response.status().as_u16() {
                200 | 204 => Ok(()),
                404 => Err(RegistryError::NotFound(format!(
                    "{}@{}",
                    name, version
                ))),
                401 | 403 => {
                    Err(RegistryError::Auth("Unauthorized".to_string()))
                }
                status => Err(RegistryError::Server(format!(
                    "Unexpected status: {}",
                    status
                ))),
            }
        }

        /// Get package owners
        pub async fn get_owners(
            &self,
            name: &str,
        ) -> Result<Vec<String>, RegistryError> {
            let url = self.api_url(&format!("/packages/{}/owners", name));

            let response = self
                .auth_header(self.client.get(&url))
                .send()
                .await
                .map_err(|e| RegistryError::Network(e.to_string()))?;

            match response.status().as_u16() {
                200 => {
                    let api_response: ApiResponse<Vec<String>> = response
                        .json()
                        .await
                        .map_err(|e| RegistryError::Invalid(e.to_string()))?;

                    api_response
                        .data
                        .ok_or_else(|| RegistryError::Invalid("Empty response".to_string()))
                }
                404 => Err(RegistryError::NotFound(name.to_string())),
                status => Err(RegistryError::Server(format!(
                    "Unexpected status: {}",
                    status
                ))),
            }
        }

        /// Add package owner
        pub async fn add_owner(
            &self,
            name: &str,
            owner: &str,
        ) -> Result<(), RegistryError> {
            if self.token.is_none() {
                return Err(RegistryError::Auth(
                    "Not logged in".to_string(),
                ));
            }

            let url = self.api_url(&format!("/packages/{}/owners", name));

            let response = self
                .auth_header(self.client.put(&url))
                .json(&serde_json::json!({ "owner": owner }))
                .send()
                .await
                .map_err(|e| RegistryError::Network(e.to_string()))?;

            match response.status().as_u16() {
                200 | 201 => Ok(()),
                404 => Err(RegistryError::NotFound(name.to_string())),
                401 | 403 => {
                    Err(RegistryError::Auth("Unauthorized".to_string()))
                }
                status => Err(RegistryError::Server(format!(
                    "Unexpected status: {}",
                    status
                ))),
            }
        }

        /// Remove package owner
        pub async fn remove_owner(
            &self,
            name: &str,
            owner: &str,
        ) -> Result<(), RegistryError> {
            if self.token.is_none() {
                return Err(RegistryError::Auth(
                    "Not logged in".to_string(),
                ));
            }

            let url = self.api_url(&format!(
                "/packages/{}/owners/{}",
                name, owner
            ));

            let response = self
                .auth_header(self.client.delete(&url))
                .send()
                .await
                .map_err(|e| RegistryError::Network(e.to_string()))?;

            match response.status().as_u16() {
                200 | 204 => Ok(()),
                404 => Err(RegistryError::NotFound(name.to_string())),
                401 | 403 => {
                    Err(RegistryError::Auth("Unauthorized".to_string()))
                }
                status => Err(RegistryError::Server(format!(
                    "Unexpected status: {}",
                    status
                ))),
            }
        }
    }

    /// Compute SHA256 checksum of bytes
    pub fn compute_sha256(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Verify checksum
    pub fn verify_checksum(data: &[u8], expected: &str) -> bool {
        compute_sha256(data) == expected
    }

    /// Extract gzipped tarball
    fn extract_tarball(
        data: &[u8],
        dest: &std::path::Path,
    ) -> Result<(), RegistryError> {
        use flate2::read::GzDecoder;
        use std::io::Read;
        use tar::Archive;

        let decoder = GzDecoder::new(data);
        let mut archive = Archive::new(decoder);

        for entry in archive.entries().map_err(|e| {
            RegistryError::Invalid(format!("Invalid tarball: {}", e))
        })? {
            let mut entry = entry.map_err(|e| {
                RegistryError::Invalid(format!("Invalid tarball entry: {}", e))
            })?;

            let path = entry.path().map_err(|e| {
                RegistryError::Invalid(format!("Invalid path: {}", e))
            })?;

            // Skip entries that try to escape the destination
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

    pub fn home_dir() -> Option<PathBuf> {
        #[cfg(target_os = "windows")]
        {
            std::env::var("USERPROFILE").ok().map(PathBuf::from)
        }

        #[cfg(not(target_os = "windows"))]
        {
            std::env::var("HOME").ok().map(PathBuf::from)
        }
    }
}

pub use self::dirs::{cache_dir, config_dir, home_dir};

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

    #[cfg(feature = "pkg")]
    #[test]
    fn test_checksum() {
        let data = b"Hello, World!";
        let checksum = async_client::compute_sha256(data);
        assert!(async_client::verify_checksum(data, &checksum));
        assert!(!async_client::verify_checksum(b"Wrong data", &checksum));
    }
}
