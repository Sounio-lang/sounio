//! Ontology Downloader
//!
//! Downloads ontologies from OBO Foundry, Schema.org, and FHIR,
//! then converts them to the compact .dontology format.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::ontology::{OntologyError, OntologyResult};

/// Configuration for ontology downloads
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Output directory for .dontology files
    pub output_dir: PathBuf,
    /// Whether to force re-download even if files exist
    pub force: bool,
    /// Whether to verify checksums
    pub verify: bool,
    /// Timeout in seconds for HTTP requests
    pub timeout_secs: u64,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from(".sounio/ontology"),
            force: false,
            verify: true,
            timeout_secs: 60,
        }
    }
}

/// Progress callback for downloads
pub type DownloadProgress = Box<dyn Fn(&str, usize, usize) + Send>;

/// Known ontology sources
#[derive(Debug, Clone)]
pub struct OntologySource {
    /// Ontology ID (e.g., "chebi", "go")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Download URL
    pub url: String,
    /// Format (owl, obo, json)
    pub format: OntologyFormat,
    /// Approximate size in MB
    pub size_mb: f64,
    /// Whether this is a core ontology
    pub core: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OntologyFormat {
    Owl,
    Obo,
    JsonLd,
    Fhir,
}

/// List of core ontologies to download
pub fn core_ontologies() -> Vec<OntologySource> {
    vec![
        // L1: Primitive (always needed)
        OntologySource {
            id: "bfo".to_string(),
            name: "Basic Formal Ontology".to_string(),
            url: "http://purl.obolibrary.org/obo/bfo.owl".to_string(),
            format: OntologyFormat::Owl,
            size_mb: 0.1,
            core: true,
        },
        OntologySource {
            id: "ro".to_string(),
            name: "Relation Ontology".to_string(),
            url: "http://purl.obolibrary.org/obo/ro.owl".to_string(),
            format: OntologyFormat::Owl,
            size_mb: 2.0,
            core: true,
        },
        OntologySource {
            id: "cob".to_string(),
            name: "Core Ontology for Biology".to_string(),
            url: "http://purl.obolibrary.org/obo/cob.owl".to_string(),
            format: OntologyFormat::Owl,
            size_mb: 0.5,
            core: true,
        },
        // L2: Foundation
        OntologySource {
            id: "pato".to_string(),
            name: "Phenotype And Trait Ontology".to_string(),
            url: "http://purl.obolibrary.org/obo/pato.owl".to_string(),
            format: OntologyFormat::Owl,
            size_mb: 5.0,
            core: true,
        },
        OntologySource {
            id: "uo".to_string(),
            name: "Units of Measurement Ontology".to_string(),
            url: "http://purl.obolibrary.org/obo/uo.owl".to_string(),
            format: OntologyFormat::Owl,
            size_mb: 1.0,
            core: true,
        },
        OntologySource {
            id: "iao".to_string(),
            name: "Information Artifact Ontology".to_string(),
            url: "http://purl.obolibrary.org/obo/iao.owl".to_string(),
            format: OntologyFormat::Owl,
            size_mb: 2.0,
            core: true,
        },
        // L3: Domain (commonly used)
        OntologySource {
            id: "chebi".to_string(),
            name: "Chemical Entities of Biological Interest".to_string(),
            url: "http://purl.obolibrary.org/obo/chebi.owl".to_string(),
            format: OntologyFormat::Owl,
            size_mb: 500.0,
            core: false,
        },
        OntologySource {
            id: "go".to_string(),
            name: "Gene Ontology".to_string(),
            url: "http://purl.obolibrary.org/obo/go.owl".to_string(),
            format: OntologyFormat::Owl,
            size_mb: 200.0,
            core: false,
        },
        OntologySource {
            id: "doid".to_string(),
            name: "Human Disease Ontology".to_string(),
            url: "http://purl.obolibrary.org/obo/doid.owl".to_string(),
            format: OntologyFormat::Owl,
            size_mb: 50.0,
            core: false,
        },
        OntologySource {
            id: "uberon".to_string(),
            name: "Uber-anatomy Ontology".to_string(),
            url: "http://purl.obolibrary.org/obo/uberon.owl".to_string(),
            format: OntologyFormat::Owl,
            size_mb: 100.0,
            core: false,
        },
    ]
}

/// Ontology downloader
#[derive(Debug)]
pub struct OntologyDownloader {
    config: DownloadConfig,
}

impl OntologyDownloader {
    /// Create a new downloader with the given config
    pub fn new(config: DownloadConfig) -> Self {
        Self { config }
    }

    /// Download all core ontologies
    pub fn download_core(
        &self,
        progress: Option<DownloadProgress>,
    ) -> OntologyResult<Vec<PathBuf>> {
        let sources: Vec<_> = core_ontologies().into_iter().filter(|s| s.core).collect();
        self.download_many(&sources, progress)
    }

    /// Download specific ontologies by ID
    pub fn download_by_ids(
        &self,
        ids: &[&str],
        progress: Option<DownloadProgress>,
    ) -> OntologyResult<Vec<PathBuf>> {
        let all = core_ontologies();
        let sources: Vec<_> = all
            .into_iter()
            .filter(|s| ids.contains(&s.id.as_str()))
            .collect();

        if sources.is_empty() {
            return Err(OntologyError::OntologyNotAvailable(format!(
                "Unknown ontology IDs: {:?}",
                ids
            )));
        }

        self.download_many(&sources, progress)
    }

    /// Download multiple ontologies
    fn download_many(
        &self,
        sources: &[OntologySource],
        progress: Option<DownloadProgress>,
    ) -> OntologyResult<Vec<PathBuf>> {
        // Ensure output directory exists
        fs::create_dir_all(&self.config.output_dir).map_err(|e| {
            OntologyError::DatabaseError(format!("Failed to create output dir: {}", e))
        })?;

        let mut results = Vec::new();

        for (i, source) in sources.iter().enumerate() {
            if let Some(ref cb) = progress {
                cb(&source.name, i, sources.len());
            }

            let output_path = self
                .config
                .output_dir
                .join(format!("{}.dontology", source.id));

            // Skip if exists and not forcing
            if output_path.exists() && !self.config.force {
                results.push(output_path);
                continue;
            }

            // Download the raw file
            let raw_path = self.download_raw(source)?;

            // Parse and convert to .dontology format
            let concepts = self.parse_ontology(&raw_path, source)?;

            // Write .dontology file
            self.write_dontology(&output_path, source, &concepts)?;

            // Clean up raw file
            let _ = fs::remove_file(&raw_path);

            results.push(output_path);
        }

        Ok(results)
    }

    /// Download the raw ontology file
    fn download_raw(&self, source: &OntologySource) -> OntologyResult<PathBuf> {
        let raw_path = self.config.output_dir.join(format!("{}.raw", source.id));

        // Use reqwest or ureq for HTTP - for now, use a simple curl fallback
        // In production, we'd use reqwest with proper async handling

        #[cfg(feature = "ontology")]
        {
            use std::process::Command;

            let status = Command::new("curl")
                .args([
                    "-L", // Follow redirects
                    "-o",
                    raw_path.to_str().unwrap(),
                    "--max-time",
                    &self.config.timeout_secs.to_string(),
                    "-s", // Silent
                    &source.url,
                ])
                .status()
                .map_err(|e| OntologyError::NetworkError(format!("curl failed: {}", e)))?;

            if !status.success() {
                return Err(OntologyError::NetworkError(format!(
                    "Failed to download {}: curl exit code {:?}",
                    source.url,
                    status.code()
                )));
            }
        }

        #[cfg(not(feature = "ontology"))]
        return Err(OntologyError::OntologyNotAvailable(
            "Ontology download requires --features ontology".to_string(),
        ));

        #[cfg(feature = "ontology")]
        Ok(raw_path)
    }

    /// Parse an ontology file into concept entries
    fn parse_ontology(
        &self,
        path: &Path,
        source: &OntologySource,
    ) -> OntologyResult<Vec<ParsedConcept>> {
        match source.format {
            OntologyFormat::Owl => self.parse_owl(path, &source.id),
            OntologyFormat::Obo => self.parse_obo(path, &source.id),
            OntologyFormat::JsonLd => self.parse_jsonld(path, &source.id),
            OntologyFormat::Fhir => self.parse_fhir(path, &source.id),
        }
    }

    /// Parse OWL/RDF format (simplified string-based extraction)
    fn parse_owl(&self, path: &Path, prefix: &str) -> OntologyResult<Vec<ParsedConcept>> {
        let file = File::open(path)
            .map_err(|e| OntologyError::DatabaseError(format!("Failed to open file: {}", e)))?;
        let reader = BufReader::new(file);

        let mut concepts = Vec::new();
        let mut current: Option<ParsedConcept> = None;

        for line in reader.lines() {
            let line = line.map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

            // Check for class declaration: <owl:Class rdf:about="...">
            if let Some(iri) = extract_attribute(&line, "owl:Class", "rdf:about") {
                // Save previous concept
                if let Some(c) = current.take() {
                    concepts.push(c);
                }

                let curie = iri_to_curie(&iri, prefix);
                current = Some(ParsedConcept {
                    curie,
                    iri,
                    label: None,
                    definition: None,
                    parent: None,
                });
            }

            if let Some(ref mut c) = current {
                // Extract label: <rdfs:label>...</rdfs:label>
                if let Some(label) = extract_element_content(&line, "rdfs:label") {
                    c.label = Some(label);
                }

                // Extract parent: <rdfs:subClassOf rdf:resource="..."/>
                if let Some(parent_iri) =
                    extract_attribute(&line, "rdfs:subClassOf", "rdf:resource")
                {
                    c.parent = Some(iri_to_curie(&parent_iri, prefix));
                }

                // Extract definition: <obo:IAO_0000115>...</obo:IAO_0000115>
                if let Some(def) = extract_element_content(&line, "obo:IAO_0000115") {
                    c.definition = Some(def);
                }
            }

            // End of class
            if line.contains("</owl:Class>")
                && let Some(c) = current.take()
            {
                concepts.push(c);
            }
        }

        // Don't forget last concept
        if let Some(c) = current {
            concepts.push(c);
        }

        Ok(concepts)
    }

    /// Parse OBO format
    fn parse_obo(&self, path: &Path, prefix: &str) -> OntologyResult<Vec<ParsedConcept>> {
        let file = File::open(path)
            .map_err(|e| OntologyError::DatabaseError(format!("Failed to open file: {}", e)))?;
        let reader = BufReader::new(file);

        let mut concepts = Vec::new();
        let mut current: Option<ParsedConcept> = None;
        let mut in_term = false;

        for line in reader.lines() {
            let line = line.map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
            let line = line.trim();

            if line == "[Term]" {
                if let Some(c) = current.take() {
                    concepts.push(c);
                }
                in_term = true;
                current = Some(ParsedConcept::default());
                continue;
            }

            if line.starts_with('[') {
                // Another stanza type
                if let Some(c) = current.take() {
                    concepts.push(c);
                }
                in_term = false;
                continue;
            }

            if !in_term {
                continue;
            }

            if let Some(ref mut c) = current {
                if let Some(id) = line.strip_prefix("id: ") {
                    c.curie = id.to_string();
                    c.iri = curie_to_iri(id);
                } else if let Some(name) = line.strip_prefix("name: ") {
                    c.label = Some(name.to_string());
                } else if let Some(def) = line.strip_prefix("def: ") {
                    // Definition is in quotes
                    if let Some(start) = def.find('"')
                        && let Some(end) = def[start + 1..].find('"')
                    {
                        c.definition = Some(def[start + 1..start + 1 + end].to_string());
                    }
                } else if let Some(parent) = line.strip_prefix("is_a: ") {
                    // Format: "CHEBI:12345 ! Parent Name"
                    let parent_id = parent.split_whitespace().next().unwrap_or(parent);
                    c.parent = Some(parent_id.to_string());
                }
            }
        }

        if let Some(c) = current {
            concepts.push(c);
        }

        Ok(concepts)
    }

    /// Parse JSON-LD format (Schema.org)
    fn parse_jsonld(&self, _path: &Path, _prefix: &str) -> OntologyResult<Vec<ParsedConcept>> {
        // Schema.org parsing would go here
        // For now, return empty - Schema.org is embedded in foundation module
        Ok(Vec::new())
    }

    /// Parse FHIR format
    fn parse_fhir(&self, _path: &Path, _prefix: &str) -> OntologyResult<Vec<ParsedConcept>> {
        // FHIR parsing would go here
        // For now, return empty - FHIR is embedded in foundation module
        Ok(Vec::new())
    }

    /// Write concepts to .dontology format
    fn write_dontology(
        &self,
        path: &Path,
        source: &OntologySource,
        concepts: &[ParsedConcept],
    ) -> OntologyResult<()> {
        use super::storage::{ConceptEntry, StringTable};
        use super::{DONTOLOGY_MAGIC, DONTOLOGY_VERSION};

        let mut file = File::create(path)
            .map_err(|e| OntologyError::DatabaseError(format!("Failed to create file: {}", e)))?;

        // Build string table
        let mut string_table = StringTable::new();
        let mut concept_entries = Vec::new();

        for c in concepts {
            let label_idx = c.label.as_ref().map(|l| string_table.intern(l));
            let def_idx = c.definition.as_ref().map(|d| string_table.intern(d));
            let parent_idx = c.parent.as_ref().map(|p| string_table.intern(p));
            let curie_idx = string_table.intern(&c.curie);

            concept_entries.push(ConceptEntry {
                curie_idx,
                label_idx,
                definition_idx: def_idx,
                parent_idx,
                flags: 0,
            });
        }

        // Write header
        file.write_all(DONTOLOGY_MAGIC)
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
        file.write_all(&DONTOLOGY_VERSION.to_le_bytes())
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        // Write ontology ID (length-prefixed)
        let id_bytes = source.id.as_bytes();
        file.write_all(&(id_bytes.len() as u32).to_le_bytes())
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
        file.write_all(id_bytes)
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        // Write version (length-prefixed)
        let version = "1.0.0";
        let version_bytes = version.as_bytes();
        file.write_all(&(version_bytes.len() as u32).to_le_bytes())
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
        file.write_all(version_bytes)
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        // Write concept count
        file.write_all(&(concept_entries.len() as u32).to_le_bytes())
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        // Write string table
        string_table
            .write_to(&mut file)
            .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;

        // Write concept entries
        for entry in &concept_entries {
            entry
                .write_to(&mut file)
                .map_err(|e| OntologyError::DatabaseError(e.to_string()))?;
        }

        Ok(())
    }
}

/// A parsed concept before storage
#[derive(Debug, Clone, Default)]
struct ParsedConcept {
    curie: String,
    iri: String,
    label: Option<String>,
    definition: Option<String>,
    parent: Option<String>,
}

/// Extract an attribute value from an XML element
/// e.g., extract_attribute("<owl:Class rdf:about=\"http://...\">", "owl:Class", "rdf:about")
fn extract_attribute(line: &str, element: &str, attr: &str) -> Option<String> {
    // Check if line contains the element
    let element_start = format!("<{}", element);
    if !line.contains(&element_start) {
        return None;
    }

    // Look for attr="value"
    let attr_pattern = format!("{}=\"", attr);
    let attr_start = line.find(&attr_pattern)?;
    let value_start = attr_start + attr_pattern.len();
    let rest = &line[value_start..];
    let value_end = rest.find('"')?;
    Some(rest[..value_end].to_string())
}

/// Extract text content from an XML element
/// e.g., extract_element_content("<rdfs:label>Aspirin</rdfs:label>", "rdfs:label")
fn extract_element_content(line: &str, element: &str) -> Option<String> {
    let open_tag_simple = format!("<{}>", element);
    let open_tag_attrs = format!("<{} ", element);
    let close_tag = format!("</{}>", element);

    // Find opening tag (with or without attributes)
    let content_start = if let Some(pos) = line.find(&open_tag_simple) {
        pos + open_tag_simple.len()
    } else if let Some(pos) = line.find(&open_tag_attrs) {
        // Has attributes, find the >
        let rest = &line[pos..];
        let tag_end = rest.find('>')?;
        pos + tag_end + 1
    } else {
        return None;
    };

    // Find closing tag
    let content_end = line.find(&close_tag)?;

    if content_start <= content_end {
        Some(line[content_start..content_end].to_string())
    } else {
        None
    }
}

/// Convert an IRI to a CURIE
fn iri_to_curie(iri: &str, default_prefix: &str) -> String {
    // OBO format: http://purl.obolibrary.org/obo/CHEBI_15365
    if iri.contains("obolibrary.org/obo/")
        && let Some(term) = iri.rsplit('/').next()
        && let Some((prefix, local)) = term.split_once('_')
    {
        return format!("{}:{}", prefix.to_uppercase(), local);
    }

    // Fallback: use default prefix with hash fragment or last path component
    if let Some(fragment) = iri.split('#').next_back()
        && fragment != iri
    {
        return format!("{}:{}", default_prefix.to_uppercase(), fragment);
    }

    if let Some(last) = iri.rsplit('/').next() {
        return format!("{}:{}", default_prefix.to_uppercase(), last);
    }

    iri.to_string()
}

/// Convert a CURIE to an OBO IRI
fn curie_to_iri(curie: &str) -> String {
    if let Some((prefix, local)) = curie.split_once(':') {
        format!(
            "http://purl.obolibrary.org/obo/{}_{}",
            prefix.to_uppercase(),
            local
        )
    } else {
        curie.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iri_to_curie() {
        assert_eq!(
            iri_to_curie("http://purl.obolibrary.org/obo/CHEBI_15365", "chebi"),
            "CHEBI:15365"
        );
        assert_eq!(
            iri_to_curie("http://purl.obolibrary.org/obo/GO_0008150", "go"),
            "GO:0008150"
        );
    }

    #[test]
    fn test_curie_to_iri() {
        assert_eq!(
            curie_to_iri("CHEBI:15365"),
            "http://purl.obolibrary.org/obo/CHEBI_15365"
        );
    }

    #[test]
    fn test_core_ontologies() {
        let cores = core_ontologies();
        assert!(cores.iter().any(|o| o.id == "bfo"));
        assert!(cores.iter().any(|o| o.id == "chebi"));
    }
}
