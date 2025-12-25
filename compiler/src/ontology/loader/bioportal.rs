//! BioPortal API Integration
//!
//! Federated resolution of 15+ million terms via NCBO BioPortal REST API.
//!
//! # Rate Limiting
//!
//! BioPortal has rate limits. We implement:
//! - Local caching to minimize API calls
//! - Exponential backoff on 429 responses
//! - Batch queries where possible
//!
//! # API Key
//!
//! Required for production use. Get one at https://bioportal.bioontology.org/account

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::{IRI, LoadedTerm, OntologyId, ResolutionError, Synonym, SynonymScope};

const BIOPORTAL_BASE: &str = "https://data.bioontology.org";

/// Simple URL encoding (percent encoding for URIs)
fn url_encode(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 3);
    for c in s.chars() {
        match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                result.push(c);
            }
            _ => {
                for byte in c.to_string().as_bytes() {
                    result.push_str(&format!("%{:02X}", byte));
                }
            }
        }
    }
    result
}

/// Rate limiter for API calls
struct RateLimiter {
    /// Requests per window
    max_requests: u32,
    /// Window duration
    window: Duration,
    /// Request timestamps
    timestamps: Mutex<Vec<Instant>>,
    /// Backoff multiplier (increases on 429)
    backoff_multiplier: AtomicU64,
}

impl RateLimiter {
    fn new(max_requests: u32, window: Duration) -> Self {
        Self {
            max_requests,
            window,
            timestamps: Mutex::new(Vec::new()),
            backoff_multiplier: AtomicU64::new(100), // 1.0 as fixed point * 100
        }
    }

    fn wait(&self) {
        let Ok(mut timestamps) = self.timestamps.lock() else {
            return;
        };
        let now = Instant::now();

        // Remove old timestamps
        timestamps.retain(|t| now.duration_since(*t) < self.window);

        // If at limit, wait
        if timestamps.len() >= self.max_requests as usize {
            let oldest = timestamps[0];
            let wait_time = self.window - now.duration_since(oldest);
            let multiplier = self.backoff_multiplier.load(Ordering::Relaxed) as f64 / 100.0;
            let actual_wait = Duration::from_secs_f64(wait_time.as_secs_f64() * multiplier);

            drop(timestamps); // Release lock before sleeping
            std::thread::sleep(actual_wait);

            let Ok(mut timestamps) = self.timestamps.lock() else {
                return;
            };
            timestamps.retain(|t| Instant::now().duration_since(*t) < self.window);
            timestamps.push(Instant::now());
            return;
        }

        timestamps.push(Instant::now());
    }

    fn backoff(&self) {
        // Increase backoff by 50%
        self.backoff_multiplier
            .fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |v| Some((v * 150 / 100).min(1000)), // Max 10x backoff
            )
            .ok();
    }

    fn reset_backoff(&self) {
        self.backoff_multiplier.store(100, Ordering::Relaxed);
    }
}

/// BioPortal federated resolver
pub struct BioPortalClient {
    api_key: Option<String>,
    rate_limiter: RateLimiter,
    timeout: Duration,
}

impl BioPortalClient {
    pub fn new(api_key: Option<String>, timeout_secs: u64) -> Result<Self, BioportalError> {
        Ok(Self {
            api_key,
            rate_limiter: RateLimiter::new(15, Duration::from_secs(1)), // 15 req/sec
            timeout: Duration::from_secs(timeout_secs),
        })
    }

    /// Resolve a single term by IRI
    pub fn resolve_term(&self, iri: &IRI) -> Result<LoadedTerm, ResolutionError> {
        self.rate_limiter.wait();

        // Determine which ontology this IRI belongs to
        let ontology_acronym = self.iri_to_ontology_acronym(iri)?;

        // Encode the IRI for URL
        let encoded_iri = url_encode(&iri.to_string());

        let url = format!(
            "{}/ontologies/{}/classes/{}",
            BIOPORTAL_BASE, ontology_acronym, encoded_iri
        );

        let response = self.make_request(&url)?;

        match response.status {
            200 => {
                self.rate_limiter.reset_backoff();
                let bp_class: BioPortalClass = serde_json::from_str(&response.body)
                    .map_err(|e| ResolutionError::ParseError(e.to_string()))?;
                Ok(bp_class.into_loaded_term())
            }
            404 => Err(ResolutionError::NotFound(iri.clone())),
            429 => {
                self.rate_limiter.backoff();
                // Retry after backoff
                self.resolve_term(iri)
            }
            status => Err(ResolutionError::ApiError {
                status,
                message: response.body,
            }),
        }
    }

    /// Search for terms matching a query
    pub fn search(
        &self,
        query: &str,
        ontologies: Option<&[OntologyId]>,
        page_size: usize,
    ) -> Result<Vec<SearchResult>, BioportalError> {
        self.rate_limiter.wait();

        let mut url = format!(
            "{}/search?q={}&pagesize={}",
            BIOPORTAL_BASE,
            url_encode(query),
            page_size
        );

        if let Some(onts) = ontologies {
            let ont_str = onts
                .iter()
                .map(|o| o.bioportal_acronym())
                .collect::<Vec<_>>()
                .join(",");
            url.push_str(&format!("&ontologies={}", ont_str));
        }

        let response = self.make_request(&url)?;

        if response.status != 200 {
            return Err(BioportalError::ApiError {
                status: response.status,
                message: response.body,
            });
        }

        let results: BioPortalSearchResults = serde_json::from_str(&response.body)
            .map_err(|e| BioportalError::ParseError(e.to_string()))?;

        Ok(results.collection.into_iter().map(|r| r.into()).collect())
    }

    /// Get all classes in an ontology (paginated)
    pub fn get_all_classes(
        &self,
        ontology: OntologyId,
        page: usize,
        page_size: usize,
    ) -> Result<PagedResults<LoadedTerm>, BioportalError> {
        self.rate_limiter.wait();

        let url = format!(
            "{}/ontologies/{}/classes?page={}&pagesize={}",
            BIOPORTAL_BASE,
            ontology.bioportal_acronym(),
            page,
            page_size
        );

        let response = self.make_request(&url)?;

        if response.status != 200 {
            return Err(BioportalError::ApiError {
                status: response.status,
                message: response.body,
            });
        }

        let paged: BioPortalPagedClasses = serde_json::from_str(&response.body)
            .map_err(|e| BioportalError::ParseError(e.to_string()))?;

        Ok(PagedResults {
            items: paged
                .collection
                .into_iter()
                .map(|c| c.into_loaded_term())
                .collect(),
            page: paged.page,
            page_count: paged.page_count,
            total_count: paged.total_count.unwrap_or(0),
        })
    }

    /// Get ontology metadata
    pub fn get_ontology_metadata(
        &self,
        ontology: OntologyId,
    ) -> Result<OntologyInfo, BioportalError> {
        self.rate_limiter.wait();

        let url = format!(
            "{}/ontologies/{}",
            BIOPORTAL_BASE,
            ontology.bioportal_acronym()
        );

        let response = self.make_request(&url)?;

        if response.status != 200 {
            return Err(BioportalError::ApiError {
                status: response.status,
                message: response.body,
            });
        }

        let info: BioPortalOntology = serde_json::from_str(&response.body)
            .map_err(|e| BioportalError::ParseError(e.to_string()))?;

        Ok(OntologyInfo {
            acronym: info.acronym,
            name: info.name,
            description: info.description,
            version: info.version,
            status: info.status,
            class_count: info.class_count,
        })
    }

    /// Get ancestors of a term
    pub fn get_ancestors(&self, iri: &IRI) -> Result<Vec<LoadedTerm>, BioportalError> {
        self.rate_limiter.wait();

        let ontology_acronym = self
            .iri_to_ontology_acronym(iri)
            .map_err(|e| BioportalError::ResolutionError(e.to_string()))?;

        let encoded_iri = url_encode(&iri.to_string());

        let url = format!(
            "{}/ontologies/{}/classes/{}/ancestors",
            BIOPORTAL_BASE, ontology_acronym, encoded_iri
        );

        let response = self.make_request(&url)?;

        if response.status != 200 {
            return Err(BioportalError::ApiError {
                status: response.status,
                message: response.body,
            });
        }

        let classes: Vec<BioPortalClass> = serde_json::from_str(&response.body)
            .map_err(|e| BioportalError::ParseError(e.to_string()))?;

        Ok(classes.into_iter().map(|c| c.into_loaded_term()).collect())
    }

    /// Get descendants of a term
    pub fn get_descendants(&self, iri: &IRI) -> Result<Vec<LoadedTerm>, BioportalError> {
        self.rate_limiter.wait();

        let ontology_acronym = self
            .iri_to_ontology_acronym(iri)
            .map_err(|e| BioportalError::ResolutionError(e.to_string()))?;

        let encoded_iri = url_encode(&iri.to_string());

        let url = format!(
            "{}/ontologies/{}/classes/{}/descendants",
            BIOPORTAL_BASE, ontology_acronym, encoded_iri
        );

        let response = self.make_request(&url)?;

        if response.status != 200 {
            return Err(BioportalError::ApiError {
                status: response.status,
                message: response.body,
            });
        }

        let classes: Vec<BioPortalClass> = serde_json::from_str(&response.body)
            .map_err(|e| BioportalError::ParseError(e.to_string()))?;

        Ok(classes.into_iter().map(|c| c.into_loaded_term()).collect())
    }

    fn iri_to_ontology_acronym(&self, iri: &IRI) -> Result<String, ResolutionError> {
        let ontology = iri.ontology();
        if ontology == OntologyId::Unknown {
            // Try to extract from IRI pattern
            if iri.0.contains("obolibrary.org")
                && let Some((prefix, _)) = iri.to_curie()
            {
                return Ok(prefix);
            }
            return Err(ResolutionError::OntologyNotAvailable(OntologyId::Unknown));
        }
        Ok(ontology.bioportal_acronym().to_string())
    }

    #[cfg(feature = "network")]
    fn make_request(&self, url: &str) -> Result<HttpResponse, BioportalError> {
        // Use reqwest for HTTP requests when network feature is enabled
        let client = reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .user_agent("Sounio-Compiler/0.47.0")
            .build()
            .map_err(|e| BioportalError::NetworkError(e.to_string()))?;

        let mut request = client.get(url).header("Accept", "application/json");

        if let Some(ref key) = self.api_key {
            request = request.header("Authorization", format!("apikey token={}", key));
        }

        match request.send() {
            Ok(response) => {
                let status = response.status().as_u16();
                let body = response
                    .text()
                    .map_err(|e| BioportalError::NetworkError(e.to_string()))?;
                Ok(HttpResponse { status, body })
            }
            Err(e) => Err(BioportalError::NetworkError(e.to_string())),
        }
    }

    #[cfg(not(feature = "network"))]
    fn make_request(&self, _url: &str) -> Result<HttpResponse, BioportalError> {
        // Network feature not enabled - return stub error
        Err(BioportalError::NetworkError(
            "Network feature not enabled. Enable 'network' feature to use BioPortal API."
                .to_string(),
        ))
    }
}

struct HttpResponse {
    status: u16,
    body: String,
}

/// BioPortal API response structures
#[derive(Debug, Deserialize)]
struct BioPortalClass {
    #[serde(rename = "@id")]
    id: String,

    #[serde(rename = "prefLabel")]
    pref_label: Option<String>,

    definition: Option<Vec<String>>,

    synonym: Option<Vec<String>>,

    #[serde(rename = "subClassOf")]
    sub_class_of: Option<Vec<String>>,

    #[serde(rename = "hasChildren")]
    has_children: Option<bool>,

    obsolete: Option<bool>,

    #[serde(rename = "@type")]
    rdf_type: Option<String>,
}

impl BioPortalClass {
    fn into_loaded_term(self) -> LoadedTerm {
        let iri = IRI::from_string(&self.id);
        let label = self.pref_label.unwrap_or_else(|| {
            iri.to_curie()
                .map(|(_, local)| local)
                .unwrap_or_else(|| self.id.clone())
        });
        let ontology = iri.ontology();

        let mut term = LoadedTerm::new(iri, label, ontology);
        term.definition = self.definition.and_then(|d| d.into_iter().next());
        term.synonyms = self
            .synonym
            .unwrap_or_default()
            .into_iter()
            .map(|s| Synonym {
                text: s,
                scope: SynonymScope::Exact,
            })
            .collect();
        term.superclasses = self
            .sub_class_of
            .unwrap_or_default()
            .into_iter()
            .filter(|s| !s.starts_with("_:")) // Filter blank nodes
            .map(|s| IRI::from_string(&s))
            .collect();
        term.is_obsolete = self.obsolete.unwrap_or(false);

        term
    }
}

#[derive(Debug, Deserialize)]
struct BioPortalSearchResults {
    collection: Vec<BioPortalSearchResult>,
    page: usize,
    #[serde(rename = "pageCount")]
    page_count: usize,
    #[serde(rename = "totalCount")]
    total_count: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct BioPortalSearchResult {
    #[serde(rename = "@id")]
    id: String,
    #[serde(rename = "prefLabel")]
    pref_label: Option<String>,
    definition: Option<Vec<String>>,
    ontology: Option<String>,
    #[serde(rename = "matchType")]
    match_type: Option<String>,
}

impl From<BioPortalSearchResult> for SearchResult {
    fn from(r: BioPortalSearchResult) -> Self {
        SearchResult {
            iri: IRI::from_string(&r.id),
            label: r.pref_label.unwrap_or_default(),
            definition: r.definition.and_then(|d| d.into_iter().next()),
            ontology: r
                .ontology
                .map(|o| OntologyId::from_prefix(&o))
                .unwrap_or(OntologyId::Unknown),
            match_type: r.match_type.unwrap_or_else(|| "unknown".to_string()),
        }
    }
}

#[derive(Debug, Deserialize)]
struct BioPortalPagedClasses {
    collection: Vec<BioPortalClass>,
    page: usize,
    #[serde(rename = "pageCount")]
    page_count: usize,
    #[serde(rename = "totalCount")]
    total_count: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct BioPortalOntology {
    acronym: String,
    name: String,
    description: Option<String>,
    version: Option<String>,
    status: Option<String>,
    #[serde(rename = "classCount")]
    class_count: Option<usize>,
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub iri: IRI,
    pub label: String,
    pub definition: Option<String>,
    pub ontology: OntologyId,
    pub match_type: String,
}

/// Paged results
#[derive(Debug, Clone)]
pub struct PagedResults<T> {
    pub items: Vec<T>,
    pub page: usize,
    pub page_count: usize,
    pub total_count: usize,
}

/// Ontology information
#[derive(Debug, Clone)]
pub struct OntologyInfo {
    pub acronym: String,
    pub name: String,
    pub description: Option<String>,
    pub version: Option<String>,
    pub status: Option<String>,
    pub class_count: Option<usize>,
}

/// BioPortal-specific errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum BioportalError {
    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("API error: status {status}, message: {message}")]
    ApiError { status: u16, message: String },

    #[error("Resolution error: {0}")]
    ResolutionError(String),

    #[error("Rate limited")]
    RateLimited,
}

impl From<BioportalError> for ResolutionError {
    fn from(e: BioportalError) -> Self {
        match e {
            BioportalError::NetworkError(msg) => ResolutionError::NetworkError(msg),
            BioportalError::ParseError(msg) => ResolutionError::ParseError(msg),
            BioportalError::ApiError { status, message } => {
                ResolutionError::ApiError { status, message }
            }
            BioportalError::ResolutionError(msg) => ResolutionError::ParseError(msg),
            BioportalError::RateLimited => ResolutionError::RateLimited {
                retry_after_secs: 60,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bioportal_class_parsing() {
        let json = r#"{
            "@id": "http://purl.obolibrary.org/obo/GO_0008150",
            "prefLabel": "biological_process",
            "definition": ["A biological process represents a specific objective..."],
            "synonym": ["biological process", "physiological process"],
            "subClassOf": ["http://purl.obolibrary.org/obo/BFO_0000015"],
            "hasChildren": true
        }"#;

        let class: BioPortalClass = serde_json::from_str(json).unwrap();
        assert_eq!(class.pref_label, Some("biological_process".to_string()));

        let term = class.into_loaded_term();
        assert_eq!(term.label, "biological_process");
        assert!(!term.superclasses.is_empty());
    }

    #[test]
    fn test_search_result_parsing() {
        let json = r#"{
            "@id": "http://purl.obolibrary.org/obo/CHEBI_15365",
            "prefLabel": "aspirin",
            "definition": ["A member of the class of benzoic acids..."],
            "ontology": "CHEBI",
            "matchType": "prefLabel"
        }"#;

        let result: BioPortalSearchResult = serde_json::from_str(json).unwrap();
        let search_result: SearchResult = result.into();

        assert_eq!(search_result.label, "aspirin");
        assert_eq!(search_result.ontology, OntologyId::ChEBI);
    }
}
