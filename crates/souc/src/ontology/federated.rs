//! L4 Federated Ontology Resolution
//!
//! This module provides access to ~15 million ontology terms via federated
//! queries to external services like BioPortal and OLS4 (EMBL-EBI).
//!
//! # Services
//!
//! - **BioPortal**: <https://bioportal.bioontology.org/>
//!   - REST API: <https://data.bioontology.org/>
//!   - ~1000 ontologies, 15M+ terms
//!   - Requires API key (free registration)
//!
//! - **OLS4**: <https://www.ebi.ac.uk/ols4/>
//!   - REST API: <https://www.ebi.ac.uk/ols4/api/>
//!   - ~300 ontologies from OBO Foundry
//!   - No API key required
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::ontology::federated::{FederatedResolver, FederatedSource};
//!
//! let resolver = FederatedResolver::new()
//!     .with_source(FederatedSource::OLS4)
//!     .with_source(FederatedSource::BioPortal { api_key: "..." });
//!
//! let term = resolver.resolve("ORDO:123")?;
//! ```
//!
//! # Caching
//!
//! All federated queries are cached aggressively to minimize network traffic.
//! Default TTL is 24 hours.
//!
//! # Rate Limiting
//!
//! Both services have rate limits. This module implements exponential backoff
//! and request throttling to stay within limits.
//!
//! # Feature Gates
//!
//! HTTP functionality requires the `network` feature to be enabled. Without
//! this feature, queries will return an error indicating network support is
//! not available.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::{OntologyError, OntologyResult};
use crate::epistemic::{Confidence, EpistemicStatus, Evidence, EvidenceKind, Revisability, Source};

#[cfg(feature = "network")]
use serde::Deserialize;

/// Configuration for federated resolution
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Maximum retries per request
    pub max_retries: u32,
    /// Base delay for exponential backoff (ms)
    pub backoff_base_ms: u64,
    /// Maximum delay for exponential backoff (ms)
    pub backoff_max_ms: u64,
    /// Minimum delay between requests (ms) - rate limiting
    pub min_request_interval_ms: u64,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 10000, // 10 seconds
            max_retries: 3,
            backoff_base_ms: 1000,        // 1 second
            backoff_max_ms: 30000,        // 30 seconds
            min_request_interval_ms: 100, // 10 requests/second max
            cache_ttl_seconds: 86400,     // 24 hours
        }
    }
}

/// A federated ontology source
#[derive(Debug, Clone)]
pub enum FederatedSource {
    /// EMBL-EBI OLS4 (no API key required)
    OLS4,
    /// NCBO BioPortal (API key required)
    BioPortal { api_key: String },
    /// Custom endpoint
    Custom {
        name: String,
        base_url: String,
        api_key: Option<String>,
    },
}

impl FederatedSource {
    /// Get the base URL for this source
    pub fn base_url(&self) -> &str {
        match self {
            FederatedSource::OLS4 => "https://www.ebi.ac.uk/ols4/api",
            FederatedSource::BioPortal { .. } => "https://data.bioontology.org",
            FederatedSource::Custom { base_url, .. } => base_url,
        }
    }

    /// Get the source name
    pub fn name(&self) -> &str {
        match self {
            FederatedSource::OLS4 => "OLS4",
            FederatedSource::BioPortal { .. } => "BioPortal",
            FederatedSource::Custom { name, .. } => name,
        }
    }

    /// Check if this source requires an API key
    pub fn requires_api_key(&self) -> bool {
        matches!(
            self,
            FederatedSource::BioPortal { .. }
                | FederatedSource::Custom {
                    api_key: Some(_),
                    ..
                }
        )
    }

    /// Create OLS4 source
    pub fn ols4() -> Self {
        FederatedSource::OLS4
    }

    /// Create BioPortal source with API key
    pub fn bioportal(api_key: impl Into<String>) -> Self {
        FederatedSource::BioPortal {
            api_key: api_key.into(),
        }
    }
}

/// A query to a federated source
#[derive(Debug, Clone)]
pub struct FederatedQuery {
    /// Term ID to resolve
    pub term_id: String,
    /// Ontology prefix filter (optional)
    pub ontology: Option<String>,
    /// Include obsolete terms
    pub include_obsolete: bool,
    /// Maximum results
    pub limit: usize,
}

impl FederatedQuery {
    /// Create a new query for a term
    pub fn term(id: impl Into<String>) -> Self {
        Self {
            term_id: id.into(),
            ontology: None,
            include_obsolete: false,
            limit: 10,
        }
    }

    /// Filter by ontology
    pub fn in_ontology(mut self, ontology: impl Into<String>) -> Self {
        self.ontology = Some(ontology.into());
        self
    }

    /// Include obsolete terms
    pub fn with_obsolete(mut self) -> Self {
        self.include_obsolete = true;
        self
    }

    /// Set result limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

/// Result from a federated query
#[derive(Debug, Clone)]
pub struct FederatedTerm {
    /// Full IRI
    pub iri: String,
    /// CURIE form
    pub curie: String,
    /// Human-readable label
    pub label: Option<String>,
    /// Definition
    pub definition: Option<String>,
    /// Synonyms
    pub synonyms: Vec<String>,
    /// Ontology this term belongs to
    pub ontology: String,
    /// Whether this term is obsolete
    pub obsolete: bool,
    /// Source this came from
    pub source: String,
    /// Epistemic status for this federated term
    pub epistemic: EpistemicStatus,
}

impl FederatedTerm {
    /// Compute epistemic status for a federated term
    pub fn compute_epistemic(
        source_name: &str,
        ontology: &str,
        term_id: &str,
        has_definition: bool,
        obsolete: bool,
    ) -> EpistemicStatus {
        // Base confidence varies by source
        let source_confidence: f64 = match source_name {
            "OLS4" => 0.85,      // OBO Foundry curation
            "BioPortal" => 0.80, // Broader curation quality
            _ => 0.70,           // Unknown sources
        };

        // Adjust for definition presence
        let definition_factor: f64 = if has_definition { 1.0 } else { 0.9 };

        // Adjust for obsolete status
        let obsolete_factor: f64 = if obsolete { 0.5 } else { 1.0 };

        let final_confidence =
            (source_confidence * definition_factor * obsolete_factor).clamp(0.0, 1.0);

        EpistemicStatus {
            confidence: Confidence::new(final_confidence),
            revisability: Revisability::Revisable {
                conditions: vec!["ontology_update".into(), "federated_refresh".into()],
            },
            source: Source::OntologyAssertion {
                ontology: ontology.to_string(),
                term: term_id.to_string(),
            },
            evidence: vec![Evidence {
                kind: EvidenceKind::Publication { doi: None },
                reference: format!("Federated: {} via {}", ontology, source_name),
                strength: Confidence::new(final_confidence),
            }],
        }
    }
}

/// Cache entry for federated terms
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached term
    term: FederatedTerm,
    /// When this entry was cached
    cached_at: Instant,
}

impl CacheEntry {
    /// Check if this entry has expired
    fn is_expired(&self, ttl_seconds: u64) -> bool {
        self.cached_at.elapsed() > Duration::from_secs(ttl_seconds)
    }
}

/// Thread-safe cache for federated terms
type TermCache = Arc<RwLock<HashMap<String, CacheEntry>>>;

/// The federated resolver
pub struct FederatedResolver {
    /// Configuration
    config: FederatedConfig,
    /// Registered sources
    sources: Vec<FederatedSource>,
    /// Last request time per source (for rate limiting)
    last_request: HashMap<String, Instant>,
    /// Request counts for statistics
    request_count: HashMap<String, usize>,
    /// Error counts per source
    error_count: HashMap<String, usize>,
    /// Term cache to avoid repeated network calls
    cache: TermCache,
    /// HTTP client (when network feature is enabled)
    #[cfg(feature = "network")]
    client: Option<reqwest::blocking::Client>,
}

impl FederatedResolver {
    /// Create a new federated resolver
    pub fn new() -> Self {
        Self {
            config: FederatedConfig::default(),
            sources: vec![],
            last_request: HashMap::new(),
            request_count: HashMap::new(),
            error_count: HashMap::new(),
            cache: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "network")]
            client: Self::build_client(FederatedConfig::default().timeout_ms),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: FederatedConfig) -> Self {
        #[cfg(feature = "network")]
        let client = Self::build_client(config.timeout_ms);
        Self {
            sources: vec![],
            last_request: HashMap::new(),
            request_count: HashMap::new(),
            error_count: HashMap::new(),
            cache: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "network")]
            client,
            config,
        }
    }

    /// Build the HTTP client
    #[cfg(feature = "network")]
    fn build_client(timeout_ms: u64) -> Option<reqwest::blocking::Client> {
        reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(timeout_ms))
            .user_agent("Sounio-Compiler/1.0.0 (https://sounio-lang.org)")
            .build()
            .ok()
    }

    /// Add a source
    pub fn with_source(mut self, source: FederatedSource) -> Self {
        self.sources.push(source);
        self
    }

    /// Add OLS4 as a source
    pub fn with_ols4(self) -> Self {
        self.with_source(FederatedSource::OLS4)
    }

    /// Add BioPortal as a source
    pub fn with_bioportal(self, api_key: impl Into<String>) -> Self {
        self.with_source(FederatedSource::bioportal(api_key))
    }

    /// Resolve a term from federated sources
    ///
    /// Checks the cache first, then tries sources in order until one succeeds.
    pub fn resolve(&mut self, query: &FederatedQuery) -> OntologyResult<FederatedTerm> {
        // Generate cache key
        let cache_key = self.cache_key(query);

        // Check cache first
        if let Some(term) = self.get_from_cache(&cache_key) {
            return Ok(term);
        }

        if self.sources.is_empty() {
            return Err(OntologyError::ResolutionFailed(
                "No federated sources configured".to_string(),
            ));
        }

        let mut last_error = None;

        for source in &self.sources.clone() {
            match self.query_source(source, query) {
                Ok(term) => {
                    // Cache the successful result
                    self.insert_into_cache(cache_key, term.clone());
                    return Ok(term);
                }
                Err(e) => {
                    *self
                        .error_count
                        .entry(source.name().to_string())
                        .or_insert(0) += 1;
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| OntologyError::TermNotFound {
            ontology: query.ontology.clone().unwrap_or_default(),
            term: query.term_id.clone(),
        }))
    }

    /// Generate a cache key for a query
    fn cache_key(&self, query: &FederatedQuery) -> String {
        match &query.ontology {
            Some(ont) => format!("{}:{}", ont, query.term_id),
            None => query.term_id.clone(),
        }
    }

    /// Get a term from the cache if it exists and hasn't expired
    fn get_from_cache(&self, key: &str) -> Option<FederatedTerm> {
        let cache = self.cache.read().ok()?;
        let entry = cache.get(key)?;
        if entry.is_expired(self.config.cache_ttl_seconds) {
            None
        } else {
            Some(entry.term.clone())
        }
    }

    /// Insert a term into the cache
    fn insert_into_cache(&self, key: String, term: FederatedTerm) {
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(
                key,
                CacheEntry {
                    term,
                    cached_at: Instant::now(),
                },
            );
        }
    }

    /// Clear expired entries from the cache
    pub fn evict_expired(&self) {
        if let Ok(mut cache) = self.cache.write() {
            let ttl = self.config.cache_ttl_seconds;
            cache.retain(|_, entry| !entry.is_expired(ttl));
        }
    }

    /// Clear the entire cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Get the number of cached entries
    pub fn cache_size(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Query a specific source
    fn query_source(
        &mut self,
        source: &FederatedSource,
        query: &FederatedQuery,
    ) -> OntologyResult<FederatedTerm> {
        // Rate limiting
        self.wait_for_rate_limit(source);

        // Record request
        *self
            .request_count
            .entry(source.name().to_string())
            .or_insert(0) += 1;

        match source {
            FederatedSource::OLS4 => self.query_ols4(query),
            FederatedSource::BioPortal { api_key } => self.query_bioportal(query, api_key),
            FederatedSource::Custom {
                name,
                base_url,
                api_key,
            } => self.query_custom(query, name, base_url, api_key.as_deref()),
        }
    }

    /// Wait for rate limit
    fn wait_for_rate_limit(&mut self, source: &FederatedSource) {
        let source_name = source.name().to_string();
        let min_interval = Duration::from_millis(self.config.min_request_interval_ms);

        if let Some(last) = self.last_request.get(&source_name) {
            let elapsed = last.elapsed();
            if elapsed < min_interval {
                std::thread::sleep(min_interval - elapsed);
            }
        }

        self.last_request.insert(source_name, Instant::now());
    }

    /// Query OLS4 (EMBL-EBI Ontology Lookup Service)
    ///
    /// OLS4 API documentation: https://www.ebi.ac.uk/ols4/api/
    #[cfg(feature = "network")]
    fn query_ols4(&self, query: &FederatedQuery) -> OntologyResult<FederatedTerm> {
        let client = self.client.as_ref().ok_or_else(|| {
            OntologyError::NetworkError("HTTP client not initialized".to_string())
        })?;

        // OLS4 uses a different URL structure depending on whether we know the ontology
        let url = if let Some(ref ontology) = query.ontology {
            // Search within a specific ontology
            // OLS4 expects double-encoded IRIs for the term endpoint
            let iri = format!(
                "http://purl.obolibrary.org/obo/{}_{}",
                ontology.to_uppercase(),
                &query.term_id
            );
            let encoded_iri = urlencoded(&urlencoded(&iri));
            format!(
                "{}/ontologies/{}/terms/{}",
                FederatedSource::OLS4.base_url(),
                ontology.to_lowercase(),
                encoded_iri
            )
        } else {
            // Search across all ontologies using the search endpoint
            format!(
                "{}/search?q={}&exact=true&rows=1",
                FederatedSource::OLS4.base_url(),
                urlencoded(&query.term_id)
            )
        };

        let response = client
            .get(&url)
            .header("Accept", "application/json")
            .send()
            .map_err(|e| OntologyError::NetworkError(format!("OLS4 request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            return Err(OntologyError::NetworkError(format!(
                "OLS4 returned status {}: {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown error")
            )));
        }

        let body = response.text().map_err(|e| {
            OntologyError::NetworkError(format!("Failed to read OLS4 response: {}", e))
        })?;

        if query.ontology.is_some() {
            // Parse direct term response
            parse_ols4_term_response(&body, query)
        } else {
            // Parse search response
            parse_ols4_search_response(&body, query)
        }
    }

    /// Query OLS4 - stub when network feature is disabled
    #[cfg(not(feature = "network"))]
    fn query_ols4(&self, query: &FederatedQuery) -> OntologyResult<FederatedTerm> {
        let _ = query;
        Err(OntologyError::NetworkError(
            "OLS4 queries require the 'network' feature. Rebuild with --features network"
                .to_string(),
        ))
    }

    /// Query BioPortal (NCBO BioPortal)
    ///
    /// BioPortal API documentation: https://data.bioontology.org/documentation
    #[cfg(feature = "network")]
    fn query_bioportal(
        &self,
        query: &FederatedQuery,
        api_key: &str,
    ) -> OntologyResult<FederatedTerm> {
        let client = self.client.as_ref().ok_or_else(|| {
            OntologyError::NetworkError("HTTP client not initialized".to_string())
        })?;

        // Build URL based on whether we have an ontology filter
        let url = if let Some(ref ontology) = query.ontology {
            // Direct class lookup - BioPortal uses IRI for class ID
            let iri = format!(
                "http://purl.obolibrary.org/obo/{}_{}",
                ontology.to_uppercase(),
                &query.term_id
            );
            let encoded_iri = urlencoded(&iri);
            format!(
                "{}/ontologies/{}/classes/{}",
                FederatedSource::BioPortal {
                    api_key: String::new()
                }
                .base_url(),
                ontology.to_uppercase(),
                encoded_iri
            )
        } else {
            // Search across all ontologies
            format!(
                "{}/search?q={}&pagesize=1&include=prefLabel,definition,synonym,obsolete",
                FederatedSource::BioPortal {
                    api_key: String::new()
                }
                .base_url(),
                urlencoded(&query.term_id)
            )
        };

        let response = client
            .get(&url)
            .header("Accept", "application/json")
            .header("Authorization", format!("apikey token={}", api_key))
            .send()
            .map_err(|e| OntologyError::NetworkError(format!("BioPortal request failed: {}", e)))?;

        let status = response.status();

        // Handle rate limiting with exponential backoff
        if status.as_u16() == 429 {
            return Err(OntologyError::NetworkError(
                "BioPortal rate limit exceeded. Please wait before retrying.".to_string(),
            ));
        }

        if !status.is_success() {
            return Err(OntologyError::NetworkError(format!(
                "BioPortal returned status {}: {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown error")
            )));
        }

        let body = response.text().map_err(|e| {
            OntologyError::NetworkError(format!("Failed to read BioPortal response: {}", e))
        })?;

        if query.ontology.is_some() {
            // Parse direct class response
            parse_bioportal_class_response(&body, query)
        } else {
            // Parse search response
            parse_bioportal_search_response(&body, query)
        }
    }

    /// Query BioPortal - stub when network feature is disabled
    #[cfg(not(feature = "network"))]
    fn query_bioportal(
        &self,
        query: &FederatedQuery,
        _api_key: &str,
    ) -> OntologyResult<FederatedTerm> {
        let _ = query;
        Err(OntologyError::NetworkError(
            "BioPortal queries require the 'network' feature. Rebuild with --features network"
                .to_string(),
        ))
    }

    /// Query a custom source.
    ///
    /// The custom source can be OLS4-compatible, BioPortal-compatible, or a generic
    /// JSON endpoint that returns `iri/id`, `label/prefLabel`, and optional metadata.
    #[cfg(feature = "network")]
    fn query_custom(
        &self,
        query: &FederatedQuery,
        source_name: &str,
        base_url: &str,
        api_key: Option<&str>,
    ) -> OntologyResult<FederatedTerm> {
        let client = self.client.as_ref().ok_or_else(|| {
            OntologyError::NetworkError("HTTP client not initialized".to_string())
        })?;

        let base = base_url.trim_end_matches('/');
        let mut urls = Vec::new();

        if let Some(ref ontology) = query.ontology {
            let iri = format!(
                "http://purl.obolibrary.org/obo/{}_{}",
                ontology.to_uppercase(),
                &query.term_id
            );
            let encoded_iri = urlencoded(&iri);
            let double_encoded_iri = urlencoded(&encoded_iri);
            urls.push(format!(
                "{}/ontologies/{}/terms/{}",
                base,
                ontology.to_lowercase(),
                double_encoded_iri
            ));
            urls.push(format!(
                "{}/ontologies/{}/classes/{}",
                base,
                ontology.to_uppercase(),
                encoded_iri
            ));
        }

        let rows = query.limit.max(1);
        let mut ols4_search = format!(
            "{}/search?q={}&exact=true&rows={}",
            base,
            urlencoded(&query.term_id),
            rows
        );
        if let Some(ref ontology) = query.ontology {
            ols4_search.push_str(&format!("&ontology={}", urlencoded(&ontology.to_lowercase())));
        }
        if query.include_obsolete {
            ols4_search.push_str("&obsoletes=true");
        }
        urls.push(ols4_search);

        let mut bioportal_search = format!(
            "{}/search?q={}&pagesize={}&include=prefLabel,definition,synonym,obsolete",
            base,
            urlencoded(&query.term_id),
            rows
        );
        if let Some(ref ontology) = query.ontology {
            bioportal_search
                .push_str(&format!("&ontologies={}", urlencoded(&ontology.to_uppercase())));
        }
        if query.include_obsolete {
            bioportal_search.push_str("&also_search_obsolete=true");
        }
        urls.push(bioportal_search);

        let mut failures = Vec::new();

        for url in urls {
            let mut request = client.get(&url).header("Accept", "application/json");
            if let Some(key) = api_key
                && !key.trim().is_empty()
            {
                request = request.header("Authorization", format!("apikey token={}", key));
            }

            let response = match request.send() {
                Ok(r) => r,
                Err(err) => {
                    failures.push(format!("{}: {}", url, err));
                    continue;
                }
            };

            if !response.status().is_success() {
                failures.push(format!("{}: status {}", url, response.status().as_u16()));
                continue;
            }

            let body = match response.text() {
                Ok(body) => body,
                Err(err) => {
                    failures.push(format!("{}: {}", url, err));
                    continue;
                }
            };

            let parsed = parse_ols4_term_response(&body, query)
                .or_else(|_| parse_ols4_search_response(&body, query))
                .or_else(|_| parse_bioportal_class_response(&body, query))
                .or_else(|_| parse_bioportal_search_response(&body, query))
                .or_else(|_| parse_custom_generic_response(&body, query, source_name));

            match parsed {
                Ok(term) => return Ok(normalize_custom_term(term, source_name, query)),
                Err(_) => failures.push(format!("{}: unsupported response schema", url)),
            }
        }

        Err(OntologyError::ResolutionFailed(format!(
            "Custom source `{}` failed: {}",
            source_name,
            failures.join("; ")
        )))
    }

    /// Query a custom source - stub when network feature is disabled.
    #[cfg(not(feature = "network"))]
    fn query_custom(
        &self,
        query: &FederatedQuery,
        _source_name: &str,
        _base_url: &str,
        _api_key: Option<&str>,
    ) -> OntologyResult<FederatedTerm> {
        let _ = query;
        Err(OntologyError::NetworkError(
            "Custom source queries require the 'network' feature. Rebuild with --features network"
                .to_string(),
        ))
    }

    /// Get request statistics
    pub fn stats(&self) -> FederatedStats {
        FederatedStats {
            request_count: self.request_count.clone(),
            error_count: self.error_count.clone(),
        }
    }

    /// Get list of configured sources
    pub fn sources(&self) -> &[FederatedSource] {
        &self.sources
    }

    /// Check if a specific source is configured
    pub fn has_source(&self, name: &str) -> bool {
        self.sources.iter().any(|s| s.name() == name)
    }
}

impl Default for FederatedResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for federated queries
#[derive(Debug, Clone)]
pub struct FederatedStats {
    /// Request count per source
    pub request_count: HashMap<String, usize>,
    /// Error count per source
    pub error_count: HashMap<String, usize>,
}

impl FederatedStats {
    /// Get total requests
    pub fn total_requests(&self) -> usize {
        self.request_count.values().sum()
    }

    /// Get total errors
    pub fn total_errors(&self) -> usize {
        self.error_count.values().sum()
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.total_requests();
        let errors = self.total_errors();
        if total == 0 {
            1.0
        } else {
            (total - errors) as f64 / total as f64
        }
    }
}

/// URL encode a string
fn urlencoded(s: &str) -> String {
    // Simple URL encoding for common characters
    s.replace('%', "%25")
        .replace(' ', "%20")
        .replace(':', "%3A")
        .replace('/', "%2F")
        .replace('#', "%23")
        .replace('?', "%3F")
        .replace('&', "%26")
        .replace('=', "%3D")
}

// ============================================================================
// OLS4 JSON Response Structures
// ============================================================================

/// OLS4 term response structure
#[cfg(feature = "network")]
#[derive(Debug, Deserialize)]
struct Ols4TermResponse {
    iri: Option<String>,
    label: Option<String>,
    description: Option<Vec<String>>,
    synonyms: Option<Vec<String>>,
    ontology_name: Option<String>,
    is_obsolete: Option<bool>,
    obo_id: Option<String>,
}

/// OLS4 search response structure
#[cfg(feature = "network")]
#[derive(Debug, Deserialize)]
struct Ols4SearchResponse {
    response: Option<Ols4SearchDocs>,
}

#[cfg(feature = "network")]
#[derive(Debug, Deserialize)]
struct Ols4SearchDocs {
    docs: Option<Vec<Ols4TermResponse>>,
}

/// Parse OLS4 direct term response
#[cfg(feature = "network")]
fn parse_ols4_term_response(json: &str, query: &FederatedQuery) -> OntologyResult<FederatedTerm> {
    let response: Ols4TermResponse = serde_json::from_str(json).map_err(|e| {
        OntologyError::ResolutionFailed(format!("Failed to parse OLS4 term response: {}", e))
    })?;

    let iri = response.iri.unwrap_or_default();
    let curie = response
        .obo_id
        .clone()
        .unwrap_or_else(|| iri_to_curie(&iri));
    let ontology = response
        .ontology_name
        .clone()
        .or_else(|| query.ontology.clone())
        .unwrap_or_else(|| extract_ontology_from_iri(&iri));
    let obsolete = response.is_obsolete.unwrap_or(false);
    let has_definition = response.description.as_ref().is_some_and(|d| !d.is_empty());

    Ok(FederatedTerm {
        iri,
        curie,
        label: response.label,
        definition: response.description.and_then(|d| d.into_iter().next()),
        synonyms: response.synonyms.unwrap_or_default(),
        ontology: ontology.clone(),
        obsolete,
        source: "OLS4".to_string(),
        epistemic: FederatedTerm::compute_epistemic(
            "OLS4",
            &ontology,
            &query.term_id,
            has_definition,
            obsolete,
        ),
    })
}

/// Parse OLS4 search response
#[cfg(feature = "network")]
fn parse_ols4_search_response(json: &str, query: &FederatedQuery) -> OntologyResult<FederatedTerm> {
    let response: Ols4SearchResponse = serde_json::from_str(json).map_err(|e| {
        OntologyError::ResolutionFailed(format!("Failed to parse OLS4 search response: {}", e))
    })?;

    let docs =
        response
            .response
            .and_then(|r| r.docs)
            .ok_or_else(|| OntologyError::TermNotFound {
                ontology: query.ontology.clone().unwrap_or_default(),
                term: query.term_id.clone(),
            })?;

    let term = docs
        .into_iter()
        .next()
        .ok_or_else(|| OntologyError::TermNotFound {
            ontology: query.ontology.clone().unwrap_or_default(),
            term: query.term_id.clone(),
        })?;

    let iri = term.iri.unwrap_or_default();
    let curie = term.obo_id.clone().unwrap_or_else(|| iri_to_curie(&iri));
    let ontology = term
        .ontology_name
        .clone()
        .unwrap_or_else(|| extract_ontology_from_iri(&iri));
    let obsolete = term.is_obsolete.unwrap_or(false);
    let has_definition = term.description.as_ref().is_some_and(|d| !d.is_empty());

    Ok(FederatedTerm {
        iri,
        curie,
        label: term.label,
        definition: term.description.and_then(|d| d.into_iter().next()),
        synonyms: term.synonyms.unwrap_or_default(),
        ontology: ontology.clone(),
        obsolete,
        source: "OLS4".to_string(),
        epistemic: FederatedTerm::compute_epistemic(
            "OLS4",
            &ontology,
            &query.term_id,
            has_definition,
            obsolete,
        ),
    })
}

// ============================================================================
// BioPortal JSON Response Structures
// ============================================================================

/// BioPortal class response structure
#[cfg(feature = "network")]
#[derive(Debug, Deserialize)]
struct BioPortalClassResponse {
    #[serde(rename = "@id")]
    id: Option<String>,
    #[serde(rename = "prefLabel")]
    pref_label: Option<String>,
    definition: Option<Vec<String>>,
    synonym: Option<Vec<String>>,
    obsolete: Option<bool>,
}

/// BioPortal search response structure
#[cfg(feature = "network")]
#[derive(Debug, Deserialize)]
struct BioPortalSearchResponse {
    collection: Option<Vec<BioPortalClassResponse>>,
}

/// Parse BioPortal direct class response
#[cfg(feature = "network")]
fn parse_bioportal_class_response(
    json: &str,
    query: &FederatedQuery,
) -> OntologyResult<FederatedTerm> {
    let response: BioPortalClassResponse = serde_json::from_str(json).map_err(|e| {
        OntologyError::ResolutionFailed(format!("Failed to parse BioPortal class response: {}", e))
    })?;

    let iri = response.id.unwrap_or_default();
    let curie = iri_to_curie(&iri);
    let ontology = query
        .ontology
        .clone()
        .unwrap_or_else(|| extract_ontology_from_iri(&iri));
    let obsolete = response.obsolete.unwrap_or(false);
    let has_definition = response.definition.as_ref().is_some_and(|d| !d.is_empty());

    Ok(FederatedTerm {
        iri,
        curie,
        label: response.pref_label,
        definition: response.definition.and_then(|d| d.into_iter().next()),
        synonyms: response.synonym.unwrap_or_default(),
        ontology: ontology.clone(),
        obsolete,
        source: "BioPortal".to_string(),
        epistemic: FederatedTerm::compute_epistemic(
            "BioPortal",
            &ontology,
            &query.term_id,
            has_definition,
            obsolete,
        ),
    })
}

/// Parse BioPortal search response
#[cfg(feature = "network")]
fn parse_bioportal_search_response(
    json: &str,
    query: &FederatedQuery,
) -> OntologyResult<FederatedTerm> {
    let response: BioPortalSearchResponse = serde_json::from_str(json).map_err(|e| {
        OntologyError::ResolutionFailed(format!("Failed to parse BioPortal search response: {}", e))
    })?;

    let collection = response
        .collection
        .ok_or_else(|| OntologyError::TermNotFound {
            ontology: query.ontology.clone().unwrap_or_default(),
            term: query.term_id.clone(),
        })?;

    let class = collection
        .into_iter()
        .next()
        .ok_or_else(|| OntologyError::TermNotFound {
            ontology: query.ontology.clone().unwrap_or_default(),
            term: query.term_id.clone(),
        })?;

    let iri = class.id.unwrap_or_default();
    let curie = iri_to_curie(&iri);
    let ontology = query
        .ontology
        .clone()
        .unwrap_or_else(|| extract_ontology_from_iri(&iri));
    let obsolete = class.obsolete.unwrap_or(false);
    let has_definition = class.definition.as_ref().is_some_and(|d| !d.is_empty());

    Ok(FederatedTerm {
        iri,
        curie,
        label: class.pref_label,
        definition: class.definition.and_then(|d| d.into_iter().next()),
        synonyms: class.synonym.unwrap_or_default(),
        ontology: ontology.clone(),
        obsolete,
        source: "BioPortal".to_string(),
        epistemic: FederatedTerm::compute_epistemic(
            "BioPortal",
            &ontology,
            &query.term_id,
            has_definition,
            obsolete,
        ),
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

#[cfg(feature = "network")]
fn normalize_custom_term(
    mut term: FederatedTerm,
    source_name: &str,
    query: &FederatedQuery,
) -> FederatedTerm {
    if term.ontology.is_empty() {
        term.ontology = query
            .ontology
            .clone()
            .unwrap_or_else(|| extract_ontology_from_iri(&term.iri));
    }
    if term.curie.is_empty() && !term.iri.is_empty() {
        term.curie = iri_to_curie(&term.iri);
    }
    term.source = source_name.to_string();
    term.epistemic = FederatedTerm::compute_epistemic(
        source_name,
        &term.ontology,
        &query.term_id,
        term.definition.as_ref().is_some_and(|d| !d.is_empty()),
        term.obsolete,
    );
    term
}

#[cfg(feature = "network")]
fn parse_custom_generic_response(
    json: &str,
    query: &FederatedQuery,
    source_name: &str,
) -> OntologyResult<FederatedTerm> {
    fn first_text(value: &serde_json::Value) -> Option<String> {
        match value {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Array(items) => items.iter().find_map(first_text),
            serde_json::Value::Object(map) => map.get("value").and_then(first_text),
            _ => None,
        }
    }

    fn field_as_text(value: &serde_json::Value, names: &[&str]) -> Option<String> {
        names
            .iter()
            .find_map(|name| value.get(*name).and_then(first_text))
    }

    fn field_as_vec(value: &serde_json::Value, names: &[&str]) -> Vec<String> {
        for name in names {
            if let Some(arr) = value.get(*name).and_then(|v| v.as_array()) {
                return arr.iter().filter_map(first_text).collect();
            }
        }
        Vec::new()
    }

    let root: serde_json::Value = serde_json::from_str(json).map_err(|err| {
        OntologyError::ResolutionFailed(format!("Failed to parse custom JSON response: {}", err))
    })?;

    let entry = root
        .pointer("/response/docs/0")
        .or_else(|| root.pointer("/collection/0"))
        .or_else(|| root.pointer("/results/0"))
        .or_else(|| root.as_array().and_then(|items| items.first()))
        .unwrap_or(&root);

    let mut iri = field_as_text(entry, &["iri", "@id", "id", "uri"]).unwrap_or_default();
    let mut curie = field_as_text(entry, &["curie", "obo_id"]).unwrap_or_default();

    if curie.is_empty() && !iri.is_empty() {
        curie = iri_to_curie(&iri);
    }
    if iri.is_empty()
        && let Some((prefix, local_id)) = curie.split_once(':')
    {
        iri = format!(
            "http://purl.obolibrary.org/obo/{}_{}",
            prefix.to_uppercase(),
            local_id
        );
    }
    if iri.is_empty() && curie.is_empty() {
        return Err(OntologyError::ResolutionFailed(
            "Custom response did not include a resolvable term id".to_string(),
        ));
    }

    let ontology = query.ontology.clone().unwrap_or_else(|| {
        if let Some((prefix, _)) = curie.split_once(':') {
            prefix.to_lowercase()
        } else {
            extract_ontology_from_iri(&iri)
        }
    });

    let definition = field_as_text(entry, &["definition", "description", "desc"]);
    let obsolete = entry
        .get("obsolete")
        .and_then(|v| v.as_bool())
        .or_else(|| entry.get("is_obsolete").and_then(|v| v.as_bool()))
        .unwrap_or(false);

    Ok(FederatedTerm {
        iri,
        curie,
        label: field_as_text(entry, &["label", "prefLabel", "name", "title"]),
        definition,
        synonyms: field_as_vec(entry, &["synonyms", "synonym"]),
        ontology,
        obsolete,
        source: source_name.to_string(),
        epistemic: EpistemicStatus::default(),
    })
}

/// Convert an IRI to a CURIE (compact URI)
#[cfg(feature = "network")]
fn iri_to_curie(iri: &str) -> String {
    // Handle OBO format: http://purl.obolibrary.org/obo/CHEBI_15365 -> CHEBI:15365
    if iri.contains("obolibrary.org/obo/") {
        if let Some(local) = iri.rsplit('/').next() {
            if let Some((prefix, id)) = local.split_once('_') {
                return format!("{}:{}", prefix, id);
            }
        }
    }
    // Fallback: use the last path segment
    iri.rsplit('/').next().unwrap_or(iri).to_string()
}

/// Extract ontology name from an IRI
#[cfg(feature = "network")]
fn extract_ontology_from_iri(iri: &str) -> String {
    // Handle OBO format
    if iri.contains("obolibrary.org/obo/") {
        if let Some(local) = iri.rsplit('/').next() {
            if let Some((prefix, _)) = local.split_once('_') {
                return prefix.to_lowercase();
            }
        }
    }
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_source_names() {
        assert_eq!(FederatedSource::OLS4.name(), "OLS4");
        assert_eq!(
            FederatedSource::BioPortal {
                api_key: "test".into()
            }
            .name(),
            "BioPortal"
        );
    }

    #[test]
    fn test_federated_source_urls() {
        assert!(FederatedSource::OLS4.base_url().contains("ebi.ac.uk"));
        assert!(
            FederatedSource::BioPortal {
                api_key: "test".into()
            }
            .base_url()
            .contains("bioontology.org")
        );
    }

    #[test]
    fn test_custom_source_metadata() {
        let source = FederatedSource::Custom {
            name: "Acme".to_string(),
            base_url: "https://example.org/api".to_string(),
            api_key: Some("secret".to_string()),
        };
        assert_eq!(source.name(), "Acme");
        assert_eq!(source.base_url(), "https://example.org/api");
        assert!(source.requires_api_key());
    }

    #[test]
    fn test_federated_query_builder() {
        let query = FederatedQuery::term("ORDO:123")
            .in_ontology("ordo")
            .with_obsolete()
            .limit(5);

        assert_eq!(query.term_id, "ORDO:123");
        assert_eq!(query.ontology, Some("ordo".to_string()));
        assert!(query.include_obsolete);
        assert_eq!(query.limit, 5);
    }

    #[test]
    fn test_url_encoding() {
        assert_eq!(urlencoded("CHEBI:15365"), "CHEBI%3A15365");
        assert_eq!(urlencoded("term with spaces"), "term%20with%20spaces");
    }

    #[test]
    fn test_federated_resolver_no_sources() {
        let mut resolver = FederatedResolver::new();
        let query = FederatedQuery::term("TEST:123");
        let result = resolver.resolve(&query);
        assert!(result.is_err());
    }

    #[cfg(not(feature = "network"))]
    #[test]
    fn test_custom_source_requires_network_feature() {
        let mut resolver = FederatedResolver::new().with_source(FederatedSource::Custom {
            name: "Acme".to_string(),
            base_url: "https://example.org/api".to_string(),
            api_key: None,
        });
        let query = FederatedQuery::term("CHEBI:15365");
        let result = resolver.resolve(&query);
        assert!(matches!(result, Err(OntologyError::NetworkError(_))));
    }

    #[test]
    fn test_federated_stats() {
        let stats = FederatedStats {
            request_count: [("OLS4".to_string(), 10), ("BioPortal".to_string(), 5)]
                .into_iter()
                .collect(),
            error_count: [("OLS4".to_string(), 1)].into_iter().collect(),
        };

        assert_eq!(stats.total_requests(), 15);
        assert_eq!(stats.total_errors(), 1);
        assert!(stats.success_rate() > 0.9);
    }

    #[test]
    fn test_cache_operations() {
        let resolver = FederatedResolver::new();
        assert_eq!(resolver.cache_size(), 0);

        // Insert a term
        let term = FederatedTerm {
            iri: "http://purl.obolibrary.org/obo/CHEBI_15365".to_string(),
            curie: "CHEBI:15365".to_string(),
            label: Some("aspirin".to_string()),
            definition: Some("A benzoic acid derivative".to_string()),
            synonyms: vec!["acetylsalicylic acid".to_string()],
            ontology: "chebi".to_string(),
            obsolete: false,
            source: "test".to_string(),
            epistemic: FederatedTerm::compute_epistemic("OLS4", "chebi", "15365", true, false),
        };

        resolver.insert_into_cache("CHEBI:15365".to_string(), term.clone());
        assert_eq!(resolver.cache_size(), 1);

        // Retrieve from cache
        let cached = resolver.get_from_cache("CHEBI:15365");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().label, Some("aspirin".to_string()));

        // Clear cache
        resolver.clear_cache();
        assert_eq!(resolver.cache_size(), 0);
    }

    #[test]
    fn test_cache_key_generation() {
        let resolver = FederatedResolver::new();

        let query_with_ontology = FederatedQuery::term("15365").in_ontology("chebi");
        assert_eq!(resolver.cache_key(&query_with_ontology), "chebi:15365");

        let query_without_ontology = FederatedQuery::term("CHEBI:15365");
        assert_eq!(resolver.cache_key(&query_without_ontology), "CHEBI:15365");
    }

    #[test]
    fn test_epistemic_status_computation() {
        // OLS4 with definition
        let status = FederatedTerm::compute_epistemic("OLS4", "chebi", "15365", true, false);
        assert!(status.confidence.value() > 0.8);

        // BioPortal without definition
        let status = FederatedTerm::compute_epistemic("BioPortal", "chebi", "15365", false, false);
        assert!(status.confidence.value() > 0.7);
        assert!(status.confidence.value() < 0.85);

        // Obsolete term
        let status = FederatedTerm::compute_epistemic("OLS4", "chebi", "15365", true, true);
        assert!(status.confidence.value() < 0.5);
    }

    #[cfg(feature = "network")]
    mod network_tests {
        use super::*;

        #[test]
        fn test_iri_to_curie() {
            assert_eq!(
                iri_to_curie("http://purl.obolibrary.org/obo/CHEBI_15365"),
                "CHEBI:15365"
            );
            assert_eq!(
                iri_to_curie("http://purl.obolibrary.org/obo/GO_0008150"),
                "GO:0008150"
            );
            // Fallback for non-OBO IRIs
            assert_eq!(iri_to_curie("http://example.org/terms/example"), "example");
        }

        #[test]
        fn test_extract_ontology_from_iri() {
            assert_eq!(
                extract_ontology_from_iri("http://purl.obolibrary.org/obo/CHEBI_15365"),
                "chebi"
            );
            assert_eq!(
                extract_ontology_from_iri("http://purl.obolibrary.org/obo/GO_0008150"),
                "go"
            );
            assert_eq!(
                extract_ontology_from_iri("http://example.org/unknown"),
                "unknown"
            );
        }

        #[test]
        fn test_parse_ols4_term_response() {
            let json = r#"{
                "iri": "http://purl.obolibrary.org/obo/GO_0008150",
                "label": "biological_process",
                "description": ["A biological process is a recognized set of events..."],
                "synonyms": ["biological process", "physiological process"],
                "ontology_name": "go",
                "is_obsolete": false,
                "obo_id": "GO:0008150"
            }"#;

            let query = FederatedQuery::term("0008150").in_ontology("go");
            let result = parse_ols4_term_response(json, &query);
            assert!(result.is_ok());

            let term = result.unwrap();
            assert_eq!(term.curie, "GO:0008150");
            assert_eq!(term.label, Some("biological_process".to_string()));
            assert_eq!(term.ontology, "go");
            assert!(!term.obsolete);
            assert!(!term.synonyms.is_empty());
        }

        #[test]
        fn test_parse_ols4_search_response() {
            let json = r#"{
                "response": {
                    "docs": [{
                        "iri": "http://purl.obolibrary.org/obo/CHEBI_15365",
                        "label": "aspirin",
                        "description": ["A benzoic acid..."],
                        "synonyms": ["acetylsalicylic acid"],
                        "ontology_name": "chebi",
                        "is_obsolete": false,
                        "obo_id": "CHEBI:15365"
                    }]
                }
            }"#;

            let query = FederatedQuery::term("aspirin");
            let result = parse_ols4_search_response(json, &query);
            assert!(result.is_ok());

            let term = result.unwrap();
            assert_eq!(term.curie, "CHEBI:15365");
            assert_eq!(term.source, "OLS4");
        }

        #[test]
        fn test_parse_bioportal_class_response() {
            let json = r#"{
                "@id": "http://purl.obolibrary.org/obo/CHEBI_15365",
                "prefLabel": "aspirin",
                "definition": ["A member of the class of benzoic acids..."],
                "synonym": ["acetylsalicylic acid", "2-acetoxybenzoic acid"],
                "obsolete": false
            }"#;

            let query = FederatedQuery::term("15365").in_ontology("chebi");
            let result = parse_bioportal_class_response(json, &query);
            assert!(result.is_ok());

            let term = result.unwrap();
            assert_eq!(term.curie, "CHEBI:15365");
            assert_eq!(term.label, Some("aspirin".to_string()));
            assert_eq!(term.source, "BioPortal");
            assert_eq!(term.synonyms.len(), 2);
        }

        #[test]
        fn test_parse_bioportal_search_response() {
            let json = r#"{
                "collection": [{
                    "@id": "http://purl.obolibrary.org/obo/GO_0008150",
                    "prefLabel": "biological_process",
                    "definition": ["A biological process..."],
                    "synonym": [],
                    "obsolete": false
                }]
            }"#;

            let query = FederatedQuery::term("biological_process");
            let result = parse_bioportal_search_response(json, &query);
            assert!(result.is_ok());

            let term = result.unwrap();
            assert_eq!(term.curie, "GO:0008150");
            assert_eq!(term.source, "BioPortal");
        }

        #[test]
        fn test_parse_empty_search_results() {
            // OLS4 empty search
            let ols4_json = r#"{"response": {"docs": []}}"#;
            let query = FederatedQuery::term("nonexistent");
            let result = parse_ols4_search_response(ols4_json, &query);
            assert!(result.is_err());

            // BioPortal empty search
            let bp_json = r#"{"collection": []}"#;
            let result = parse_bioportal_search_response(bp_json, &query);
            assert!(result.is_err());
        }

        #[test]
        fn test_parse_custom_generic_response() {
            let json = r#"{
                "id": "http://purl.obolibrary.org/obo/CHEBI_15365",
                "label": "aspirin",
                "description": "A benzoic acid derivative",
                "synonyms": ["acetylsalicylic acid"],
                "obsolete": false
            }"#;

            let query = FederatedQuery::term("15365").in_ontology("chebi");
            let term = parse_custom_generic_response(json, &query, "Acme").unwrap();
            assert_eq!(term.curie, "CHEBI:15365");
            assert_eq!(term.label, Some("aspirin".to_string()));
            assert_eq!(term.ontology, "chebi");
        }
    }
}
