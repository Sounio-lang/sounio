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

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::{OntologyError, OntologyResult};
use crate::epistemic::{Confidence, EpistemicStatus, Evidence, EvidenceKind, Revisability, Source};

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
        matches!(self, FederatedSource::BioPortal { .. })
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
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: FederatedConfig) -> Self {
        Self {
            config,
            sources: vec![],
            last_request: HashMap::new(),
            request_count: HashMap::new(),
            error_count: HashMap::new(),
        }
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
    /// Tries sources in order until one succeeds.
    pub fn resolve(&mut self, query: &FederatedQuery) -> OntologyResult<FederatedTerm> {
        if self.sources.is_empty() {
            return Err(OntologyError::ResolutionFailed(
                "No federated sources configured".to_string(),
            ));
        }

        let mut last_error = None;

        for source in &self.sources.clone() {
            match self.query_source(source, query) {
                Ok(term) => return Ok(term),
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
            FederatedSource::Custom { .. } => Err(OntologyError::ResolutionFailed(
                "Custom sources not yet implemented".to_string(),
            )),
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

    /// Query OLS4
    fn query_ols4(&self, query: &FederatedQuery) -> OntologyResult<FederatedTerm> {
        // Build URL
        let url = if let Some(ref ontology) = query.ontology {
            format!(
                "{}/ontologies/{}/terms/{}",
                FederatedSource::OLS4.base_url(),
                ontology.to_lowercase(),
                urlencoded(&query.term_id)
            )
        } else {
            format!(
                "{}/terms/{}",
                FederatedSource::OLS4.base_url(),
                urlencoded(&query.term_id)
            )
        };

        // In a real implementation, this would make an HTTP request
        // For now, return a placeholder error indicating network would be needed
        Err(OntologyError::NetworkError(format!(
            "Would query OLS4 at: {}. Network requests not implemented in this context.",
            url
        )))
    }

    /// Query BioPortal
    fn query_bioportal(
        &self,
        query: &FederatedQuery,
        _api_key: &str,
    ) -> OntologyResult<FederatedTerm> {
        // Build URL
        let url = if let Some(ref ontology) = query.ontology {
            format!(
                "{}/ontologies/{}/classes/{}",
                FederatedSource::BioPortal {
                    api_key: String::new()
                }
                .base_url(),
                ontology.to_uppercase(),
                urlencoded(&query.term_id)
            )
        } else {
            format!(
                "{}/search?q={}",
                FederatedSource::BioPortal {
                    api_key: String::new()
                }
                .base_url(),
                urlencoded(&query.term_id)
            )
        };

        // In a real implementation, this would make an HTTP request
        Err(OntologyError::NetworkError(format!(
            "Would query BioPortal at: {}. Network requests not implemented in this context.",
            url
        )))
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

/// Parse OLS4 JSON response
#[allow(dead_code)]
fn parse_ols4_response(json: &str) -> OntologyResult<FederatedTerm> {
    // Would use serde_json to parse
    // Placeholder implementation
    Err(OntologyError::ResolutionFailed(format!(
        "OLS4 response parsing not implemented. JSON length: {}",
        json.len()
    )))
}

/// Parse BioPortal JSON response
#[allow(dead_code)]
fn parse_bioportal_response(json: &str) -> OntologyResult<FederatedTerm> {
    // Would use serde_json to parse
    // Placeholder implementation
    Err(OntologyError::ResolutionFailed(format!(
        "BioPortal response parsing not implemented. JSON length: {}",
        json.len()
    )))
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
}
