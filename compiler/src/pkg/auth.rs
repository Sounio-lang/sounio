//! Credentials management for Sounio package registry
//!
//! Stores and retrieves authentication tokens for package registry access.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::registry::{RegistryError, home_dir};

/// Credentials store for registry authentication
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CredentialsStore {
    /// API tokens indexed by registry URL
    #[serde(default)]
    pub registries: HashMap<String, RegistryCredentials>,

    /// Default registry URL
    #[serde(default)]
    pub default_registry: Option<String>,
}

/// Credentials for a specific registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryCredentials {
    /// API token
    pub token: String,

    /// Optional username (for display purposes)
    #[serde(default)]
    pub username: Option<String>,

    /// When the token was created/updated
    #[serde(default)]
    pub created_at: Option<String>,

    /// Token expiration time
    #[serde(default)]
    pub expires_at: Option<String>,
}

/// Result type for auth operations
pub type AuthResult<T> = Result<T, AuthError>;

/// Authentication error
#[derive(Debug)]
pub enum AuthError {
    /// IO error reading/writing credentials
    Io(std::io::Error),

    /// Parse error in credentials file
    Parse(String),

    /// No credentials found for registry
    NotFound(String),

    /// Token expired
    Expired(String),

    /// Invalid token format
    InvalidToken(String),
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::Io(e) => write!(f, "IO error: {}", e),
            AuthError::Parse(e) => write!(f, "Parse error: {}", e),
            AuthError::NotFound(registry) => {
                write!(f, "No credentials found for registry: {}", registry)
            }
            AuthError::Expired(registry) => {
                write!(f, "Token expired for registry: {}", registry)
            }
            AuthError::InvalidToken(msg) => write!(f, "Invalid token: {}", msg),
        }
    }
}

impl std::error::Error for AuthError {}

impl From<std::io::Error> for AuthError {
    fn from(e: std::io::Error) -> Self {
        AuthError::Io(e)
    }
}

impl CredentialsStore {
    /// Default credentials file path (~/.sounio/credentials.toml)
    pub fn default_path() -> Option<PathBuf> {
        home_dir().map(|h| h.join(".sounio").join("credentials.toml"))
    }

    /// Load credentials from the default path
    pub fn load_default() -> AuthResult<Self> {
        match Self::default_path() {
            Some(path) if path.exists() => Self::load(&path),
            Some(_) => Ok(Self::default()),
            None => Err(AuthError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine home directory",
            ))),
        }
    }

    /// Load credentials from a file
    pub fn load(path: &Path) -> AuthResult<Self> {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content).map_err(|e| AuthError::Parse(e.to_string()))
    }

    /// Save credentials to the default path
    pub fn save_default(&self) -> AuthResult<()> {
        match Self::default_path() {
            Some(path) => self.save(&path),
            None => Err(AuthError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine home directory",
            ))),
        }
    }

    /// Save credentials to a file
    pub fn save(&self, path: &Path) -> AuthResult<()> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self).map_err(|e| AuthError::Parse(e.to_string()))?;

        // Write with restricted permissions on Unix
        #[cfg(unix)]
        {
            use std::fs::OpenOptions;
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;

            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600) // rw-------
                .open(path)?;
            file.write_all(content.as_bytes())?;
        }

        #[cfg(not(unix))]
        {
            std::fs::write(path, content)?;
        }

        Ok(())
    }

    /// Get token for a registry
    pub fn get_token(&self, registry_url: &str) -> Option<&str> {
        self.registries
            .get(registry_url)
            .map(|creds| creds.token.as_str())
    }

    /// Get token for default registry
    pub fn get_default_token(&self) -> Option<&str> {
        self.default_registry
            .as_ref()
            .and_then(|url| self.get_token(url))
    }

    /// Set token for a registry
    pub fn set_token(
        &mut self,
        registry_url: &str,
        token: String,
        username: Option<String>,
    ) -> AuthResult<()> {
        // Validate token format (basic check)
        if token.is_empty() {
            return Err(AuthError::InvalidToken("Token cannot be empty".to_string()));
        }

        let now = chrono::Utc::now().to_rfc3339();

        self.registries.insert(
            registry_url.to_string(),
            RegistryCredentials {
                token,
                username,
                created_at: Some(now),
                expires_at: None,
            },
        );

        // Set as default if no default exists
        if self.default_registry.is_none() {
            self.default_registry = Some(registry_url.to_string());
        }

        Ok(())
    }

    /// Remove token for a registry
    pub fn remove_token(&mut self, registry_url: &str) -> bool {
        let removed = self.registries.remove(registry_url).is_some();

        // Clear default if it was the removed registry
        if self.default_registry.as_deref() == Some(registry_url) {
            self.default_registry = self.registries.keys().next().cloned();
        }

        removed
    }

    /// Set default registry
    pub fn set_default_registry(&mut self, registry_url: &str) {
        self.default_registry = Some(registry_url.to_string());
    }

    /// Check if token exists for registry
    pub fn has_token(&self, registry_url: &str) -> bool {
        self.registries.contains_key(registry_url)
    }

    /// Get all configured registries
    pub fn list_registries(&self) -> Vec<&str> {
        self.registries.keys().map(|s| s.as_str()).collect()
    }

    /// Get credentials for a registry
    pub fn get_credentials(&self, registry_url: &str) -> Option<&RegistryCredentials> {
        self.registries.get(registry_url)
    }

    /// Check if token is expired
    pub fn is_token_expired(&self, registry_url: &str) -> bool {
        if let Some(creds) = self.registries.get(registry_url) {
            if let Some(ref expires_at) = creds.expires_at {
                if let Ok(expiry) = chrono::DateTime::parse_from_rfc3339(expires_at) {
                    return expiry < chrono::Utc::now();
                }
            }
        }
        false
    }
}

/// Token manager for handling registry authentication
pub struct TokenManager {
    store: CredentialsStore,
    store_path: Option<PathBuf>,
}

impl TokenManager {
    /// Create a new token manager with default credentials
    pub fn new() -> AuthResult<Self> {
        let store = CredentialsStore::load_default()?;
        Ok(Self {
            store,
            store_path: CredentialsStore::default_path(),
        })
    }

    /// Create with a specific credentials file
    pub fn with_path(path: PathBuf) -> AuthResult<Self> {
        let store = if path.exists() {
            CredentialsStore::load(&path)?
        } else {
            CredentialsStore::default()
        };
        Ok(Self {
            store,
            store_path: Some(path),
        })
    }

    /// Get token for a registry URL
    pub fn get_token(&self, registry_url: &str) -> Option<&str> {
        self.store.get_token(registry_url)
    }

    /// Get token for default registry
    pub fn get_default_token(&self) -> Option<&str> {
        self.store.get_default_token()
    }

    /// Login to a registry with a token
    pub fn login(
        &mut self,
        registry_url: &str,
        token: String,
        username: Option<String>,
    ) -> AuthResult<()> {
        self.store.set_token(registry_url, token, username)?;
        self.save()?;
        Ok(())
    }

    /// Logout from a registry
    pub fn logout(&mut self, registry_url: &str) -> AuthResult<bool> {
        let removed = self.store.remove_token(registry_url);
        if removed {
            self.save()?;
        }
        Ok(removed)
    }

    /// Save credentials to file
    pub fn save(&self) -> AuthResult<()> {
        if let Some(ref path) = self.store_path {
            self.store.save(path)?;
        }
        Ok(())
    }

    /// Get default registry URL
    pub fn default_registry(&self) -> Option<&str> {
        self.store.default_registry.as_deref()
    }

    /// Set default registry
    pub fn set_default_registry(&mut self, registry_url: &str) -> AuthResult<()> {
        self.store.set_default_registry(registry_url);
        self.save()
    }

    /// Check if logged in to a registry
    pub fn is_logged_in(&self, registry_url: &str) -> bool {
        self.store.has_token(registry_url) && !self.store.is_token_expired(registry_url)
    }

    /// List all logged-in registries
    pub fn list_logged_in(&self) -> Vec<(&str, Option<&str>)> {
        self.store
            .registries
            .iter()
            .filter(|(url, _)| !self.store.is_token_expired(url))
            .map(|(url, creds)| (url.as_str(), creds.username.as_deref()))
            .collect()
    }
}

impl Default for TokenManager {
    fn default() -> Self {
        Self {
            store: CredentialsStore::default(),
            store_path: CredentialsStore::default_path(),
        }
    }
}

/// Helper to read token from stdin interactively
pub fn read_token_from_stdin() -> std::io::Result<String> {
    use std::io::{BufRead, Write};

    print!("Please paste your API token: ");
    std::io::stdout().flush()?;

    let stdin = std::io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line)?;

    Ok(line.trim().to_string())
}

/// Validate token format
pub fn validate_token(token: &str) -> Result<(), AuthError> {
    if token.is_empty() {
        return Err(AuthError::InvalidToken("Token cannot be empty".to_string()));
    }

    if token.len() < 32 {
        return Err(AuthError::InvalidToken(
            "Token appears too short".to_string(),
        ));
    }

    // Check for valid base64 or hex characters
    if !token
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '=')
    {
        return Err(AuthError::InvalidToken(
            "Token contains invalid characters".to_string(),
        ));
    }

    Ok(())
}

/// Convert auth error to registry error
impl From<AuthError> for RegistryError {
    fn from(e: AuthError) -> Self {
        match e {
            AuthError::Io(io) => RegistryError::Io(io),
            AuthError::Parse(msg) => RegistryError::Invalid(msg),
            AuthError::NotFound(registry) => {
                RegistryError::Auth(format!("Not logged in to {}", registry))
            }
            AuthError::Expired(registry) => {
                RegistryError::Auth(format!("Token expired for {}", registry))
            }
            AuthError::InvalidToken(msg) => RegistryError::Auth(msg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_credentials_store() {
        let mut store = CredentialsStore::default();

        // Add a token
        store
            .set_token(
                "https://registry.example.com",
                "test-token".to_string(),
                None,
            )
            .unwrap();

        assert!(store.has_token("https://registry.example.com"));
        assert_eq!(
            store.get_token("https://registry.example.com"),
            Some("test-token")
        );

        // Check default was set
        assert_eq!(
            store.default_registry,
            Some("https://registry.example.com".to_string())
        );
    }

    #[test]
    fn test_credentials_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let creds_path = temp_dir.path().join("credentials.toml");

        let mut store = CredentialsStore::default();
        store
            .set_token(
                "https://registry.example.com",
                "my-secret-token".to_string(),
                Some("testuser".to_string()),
            )
            .unwrap();

        // Save
        store.save(&creds_path).unwrap();
        assert!(creds_path.exists());

        // Load
        let loaded = CredentialsStore::load(&creds_path).unwrap();
        assert_eq!(
            loaded.get_token("https://registry.example.com"),
            Some("my-secret-token")
        );

        let creds = loaded
            .get_credentials("https://registry.example.com")
            .unwrap();
        assert_eq!(creds.username, Some("testuser".to_string()));
    }

    #[test]
    fn test_remove_token() {
        let mut store = CredentialsStore::default();

        store
            .set_token("https://registry1.example.com", "token1".to_string(), None)
            .unwrap();
        store
            .set_token("https://registry2.example.com", "token2".to_string(), None)
            .unwrap();

        assert!(store.remove_token("https://registry1.example.com"));
        assert!(!store.has_token("https://registry1.example.com"));
        assert!(store.has_token("https://registry2.example.com"));

        // Default should have been updated
        assert_eq!(
            store.default_registry,
            Some("https://registry2.example.com".to_string())
        );
    }

    #[test]
    fn test_validate_token() {
        assert!(validate_token("").is_err());
        assert!(validate_token("short").is_err());
        assert!(validate_token("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").is_ok());
        // Token must be 32+ chars - use a valid long token with special allowed chars
        assert!(validate_token("abcd1234-ABCD-5678_efgh=01234567").is_ok());
        assert!(validate_token("invalid!token@with#special$chars").is_err());
    }

    #[test]
    fn test_token_manager() {
        let temp_dir = TempDir::new().unwrap();
        let creds_path = temp_dir.path().join("credentials.toml");

        let mut manager = TokenManager::with_path(creds_path.clone()).unwrap();

        manager
            .login(
                "https://registry.example.com",
                "test-token-12345678901234567890".to_string(),
                Some("user".to_string()),
            )
            .unwrap();

        assert!(manager.is_logged_in("https://registry.example.com"));
        assert_eq!(
            manager.get_token("https://registry.example.com"),
            Some("test-token-12345678901234567890")
        );

        // Reload and verify persistence
        let manager2 = TokenManager::with_path(creds_path).unwrap();
        assert!(manager2.is_logged_in("https://registry.example.com"));
    }

    #[test]
    fn test_list_registries() {
        let mut store = CredentialsStore::default();

        store
            .set_token("https://registry1.example.com", "token1".to_string(), None)
            .unwrap();
        store
            .set_token("https://registry2.example.com", "token2".to_string(), None)
            .unwrap();

        let registries = store.list_registries();
        assert_eq!(registries.len(), 2);
        assert!(registries.contains(&"https://registry1.example.com"));
        assert!(registries.contains(&"https://registry2.example.com"));
    }
}
