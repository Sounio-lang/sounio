//! Merkle Provenance Trees for Verifiable Audit Trails
//!
//! This module extends the provenance system with cryptographic verification
//! using Merkle DAG structures. Key features:
//!
//! 1. **Content-addressed storage** - Each provenance node is identified by its hash
//! 2. **DAG structure** - Multiple inputs can contribute to a single output
//! 3. **Incremental hashing** - Efficient updates using BLAKE3
//! 4. **Multi-signature support** - Multiple parties can sign provenance
//! 5. **Time-lock verification** - Optional timestamp authorities
//!
//! # Regulatory Compliance
//!
//! This module is designed to support 21 CFR Part 11 compliance for pharmaceutical
//! applications, providing:
//! - Complete audit trails
//! - Tamper-evident records
//! - Electronic signatures
//! - Timestamp verification
//!
//! # Example
//!
//! ```sounio
//! let measurement = Knowledge::from_measurement(98.6, "thermometer_001");
//!
//! // Provenance is automatically tracked with Merkle hashes
//! let celsius = measurement.map(|f| (f - 32.0) * 5.0 / 9.0);
//!
//! // Verify the provenance chain
//! assert!(celsius.provenance().verify_integrity());
//!
//! // Get audit trail for regulatory submission
//! let audit = celsius.provenance().to_audit_trail();
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// We use a simple hash for now - in production, use blake3 crate
// This is a placeholder that could be replaced with:
// use blake3::Hasher;

/// 256-bit hash value (BLAKE3 output)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hash256([u8; 32]);

impl Hash256 {
    /// Create a zero hash
    pub fn zero() -> Self {
        Hash256([0u8; 32])
    }

    /// Create from bytes
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Hash256(bytes)
    }

    /// Get as bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Parse from hex string
    pub fn from_hex(hex: &str) -> Result<Self, HashError> {
        if hex.len() != 64 {
            return Err(HashError::InvalidLength(hex.len()));
        }

        let mut bytes = [0u8; 32];
        for i in 0..32 {
            bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .map_err(|_| HashError::InvalidHex)?;
        }
        Ok(Hash256(bytes))
    }
}

impl fmt::Debug for Hash256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hash256({}...)", &self.to_hex()[..8])
    }
}

impl fmt::Display for Hash256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Hash computation errors
#[derive(Debug, Clone)]
pub enum HashError {
    InvalidLength(usize),
    InvalidHex,
    VerificationFailed,
}

impl fmt::Display for HashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HashError::InvalidLength(len) => write!(f, "invalid hash length: {}", len),
            HashError::InvalidHex => write!(f, "invalid hex encoding"),
            HashError::VerificationFailed => write!(f, "hash verification failed"),
        }
    }
}

impl std::error::Error for HashError {}

/// Simple BLAKE3-like hasher (placeholder implementation)
/// In production, replace with actual blake3 crate
pub struct Hasher {
    state: [u64; 4],
    buffer: Vec<u8>,
}

impl Hasher {
    pub fn new() -> Self {
        Self {
            state: [
                0x6a09e667f3bcc908,
                0xbb67ae8584caa73b,
                0x3c6ef372fe94f82b,
                0xa54ff53a5f1d36f1,
            ],
            buffer: Vec::new(),
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        // Simple mixing (not cryptographically secure - placeholder only)
        for chunk in self.buffer.chunks(8) {
            let mut val = 0u64;
            for (i, &byte) in chunk.iter().enumerate() {
                val |= (byte as u64) << (i * 8);
            }
            self.state[0] = self.state[0].wrapping_add(val);
            self.state[1] = self.state[1].wrapping_mul(self.state[0].wrapping_add(1));
            self.state[2] ^= self.state[1].rotate_left(13);
            self.state[3] = self.state[3].wrapping_add(self.state[2]);
        }
    }

    pub fn finalize(&self) -> Hash256 {
        let mut result = [0u8; 32];
        for (i, &s) in self.state.iter().enumerate() {
            let bytes = s.to_le_bytes();
            result[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
        }
        Hash256(result)
    }
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute hash of arbitrary data
pub fn hash(data: &[u8]) -> Hash256 {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

/// Compute hash of multiple items
pub fn hash_combine(items: &[&[u8]]) -> Hash256 {
    let mut hasher = Hasher::new();
    for item in items {
        hasher.update(item);
    }
    hasher.finalize()
}

// =============================================================================
// Merkle Provenance Node
// =============================================================================

/// A node in the Merkle provenance DAG
#[derive(Debug, Clone)]
pub struct MerkleProvenanceNode {
    /// Unique identifier (content hash)
    pub id: Hash256,

    /// Parent node hashes (inputs to this computation)
    pub parents: Vec<Hash256>,

    /// Operation/transformation that produced this node
    pub operation: ProvenanceOperation,

    /// Timestamp (Unix epoch seconds)
    pub timestamp: u64,

    /// Optional signatures from authorities
    pub signatures: Vec<ProvenanceSignature>,

    /// Metadata for regulatory compliance
    pub metadata: ProvenanceMetadata,
}

impl MerkleProvenanceNode {
    /// Create a new root node (no parents)
    pub fn root(operation: ProvenanceOperation) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut node = Self {
            id: Hash256::zero(),
            parents: Vec::new(),
            operation,
            timestamp,
            signatures: Vec::new(),
            metadata: ProvenanceMetadata::default(),
        };
        node.id = node.compute_hash();
        node
    }

    /// Create a derived node from parents
    pub fn derived(parents: Vec<Hash256>, operation: ProvenanceOperation) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut node = Self {
            id: Hash256::zero(),
            parents,
            operation,
            timestamp,
            signatures: Vec::new(),
            metadata: ProvenanceMetadata::default(),
        };
        node.id = node.compute_hash();
        node
    }

    /// Compute the hash of this node
    fn compute_hash(&self) -> Hash256 {
        let mut hasher = Hasher::new();

        // Hash parents
        for parent in &self.parents {
            hasher.update(parent.as_bytes());
        }

        // Hash operation
        hasher.update(self.operation.name().as_bytes());
        hasher.update(&self.operation.kind().to_bytes());

        // Hash timestamp
        hasher.update(&self.timestamp.to_le_bytes());

        // Hash metadata
        hasher.update(self.metadata.user.as_deref().unwrap_or("").as_bytes());
        hasher.update(self.metadata.system.as_deref().unwrap_or("").as_bytes());

        hasher.finalize()
    }

    /// Verify the integrity of this node
    pub fn verify(&self) -> Result<(), HashError> {
        let computed = self.compute_hash();
        if computed == self.id {
            Ok(())
        } else {
            Err(HashError::VerificationFailed)
        }
    }

    /// Add a signature to this node
    pub fn sign(&mut self, signature: ProvenanceSignature) {
        self.signatures.push(signature);
    }

    /// Check if this node has a valid signature from a specific authority
    pub fn has_signature_from(&self, authority: &str) -> bool {
        self.signatures.iter().any(|s| s.authority == authority)
    }

    /// Get the total number of ancestors (recursively)
    pub fn depth(&self) -> usize {
        if self.parents.is_empty() {
            0
        } else {
            1 // Would need DAG to compute recursively
        }
    }

    /// Extend this node with a new transformation
    pub fn extend(&self, transformation_name: &str) -> Self {
        Self::derived(
            vec![self.id],
            ProvenanceOperation::transformation(transformation_name, 1.0),
        )
    }

    /// Merge two provenance nodes with a combining operation
    pub fn merge(&self, other: &Self, operation_name: &str) -> Self {
        Self::derived(
            vec![self.id, other.id],
            ProvenanceOperation::new(operation_name, OperationKind::Aggregation),
        )
    }
}

/// Operation that produced a provenance node
#[derive(Debug, Clone)]
pub struct ProvenanceOperation {
    /// Human-readable name
    name: String,
    /// Operation kind
    kind: OperationKind,
    /// Confidence factor applied
    confidence_factor: f64,
    /// Source location (file:line:col)
    location: Option<String>,
}

impl ProvenanceOperation {
    pub fn new(name: &str, kind: OperationKind) -> Self {
        Self {
            name: name.to_string(),
            kind,
            confidence_factor: 1.0,
            location: None,
        }
    }

    pub fn literal(value_repr: &str) -> Self {
        Self::new(&format!("literal({})", value_repr), OperationKind::Literal)
    }

    pub fn measurement(instrument: &str) -> Self {
        Self::new(
            &format!("measurement({})", instrument),
            OperationKind::Measurement,
        )
    }

    pub fn computation(function: &str) -> Self {
        Self::new(function, OperationKind::Computation)
    }

    pub fn transformation(name: &str, factor: f64) -> Self {
        Self {
            name: name.to_string(),
            kind: OperationKind::Transformation,
            confidence_factor: factor,
            location: None,
        }
    }

    pub fn external(uri: &str) -> Self {
        Self::new(&format!("external({})", uri), OperationKind::External)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> OperationKind {
        self.kind
    }

    pub fn confidence_factor(&self) -> f64 {
        self.confidence_factor
    }

    pub fn with_location(mut self, location: &str) -> Self {
        self.location = Some(location.to_string());
        self
    }
}

/// Kind of provenance operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationKind {
    /// Literal value in source code
    Literal,
    /// Measurement from instrument
    Measurement,
    /// Computational derivation
    Computation,
    /// Data transformation
    Transformation,
    /// External data source
    External,
    /// User input
    UserInput,
    /// Database query
    Database,
    /// Ontology assertion
    OntologyAssertion,
    /// Model prediction
    ModelPrediction,
    /// Aggregation/merge
    Aggregation,
}

impl OperationKind {
    pub fn to_bytes(&self) -> [u8; 1] {
        [*self as u8]
    }
}

/// Cryptographic signature on provenance
#[derive(Debug, Clone)]
pub struct ProvenanceSignature {
    /// Authority that signed
    pub authority: String,
    /// Signature bytes (placeholder - would be actual crypto signature)
    pub signature: Vec<u8>,
    /// Timestamp of signature
    pub timestamp: u64,
    /// Algorithm used
    pub algorithm: SignatureAlgorithm,
}

impl ProvenanceSignature {
    /// Create a new signature (placeholder implementation)
    pub fn new(authority: &str, private_key: &[u8], node_hash: &Hash256) -> Self {
        // In production, use actual cryptographic signing
        let mut sig = Vec::with_capacity(64);
        sig.extend_from_slice(&private_key[..32.min(private_key.len())]);
        sig.extend_from_slice(&node_hash.as_bytes()[..32]);

        Self {
            authority: authority.to_string(),
            signature: sig,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            algorithm: SignatureAlgorithm::Ed25519,
        }
    }

    /// Verify this signature (placeholder)
    pub fn verify(&self, _public_key: &[u8], _node_hash: &Hash256) -> bool {
        // In production, use actual cryptographic verification
        !self.signature.is_empty()
    }
}

/// Signature algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureAlgorithm {
    Ed25519,
    EcdsaP256,
    RsaPss2048,
}

/// Metadata for regulatory compliance
#[derive(Debug, Clone, Default)]
pub struct ProvenanceMetadata {
    /// User who performed the operation
    pub user: Option<String>,
    /// System identifier
    pub system: Option<String>,
    /// Software version
    pub version: Option<String>,
    /// Regulatory context (e.g., "21 CFR Part 11")
    pub regulatory_context: Option<String>,
    /// Custom key-value pairs
    pub custom: HashMap<String, String>,
}

impl ProvenanceMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_user(mut self, user: &str) -> Self {
        self.user = Some(user.to_string());
        self
    }

    pub fn with_system(mut self, system: &str) -> Self {
        self.system = Some(system.to_string());
        self
    }

    pub fn with_regulatory_context(mut self, context: &str) -> Self {
        self.regulatory_context = Some(context.to_string());
        self
    }

    pub fn with_custom(mut self, key: &str, value: &str) -> Self {
        self.custom.insert(key.to_string(), value.to_string());
        self
    }
}

// =============================================================================
// Merkle Provenance DAG
// =============================================================================

/// A Merkle DAG of provenance nodes
#[derive(Debug, Clone)]
pub struct MerkleProvenanceDAG {
    /// All nodes indexed by hash
    pub(crate) nodes: HashMap<Hash256, MerkleProvenanceNode>,
    /// Current head nodes (leaves of the DAG)
    pub(crate) heads: HashSet<Hash256>,
    /// Root nodes (nodes with no parents)
    pub(crate) roots: HashSet<Hash256>,
}

impl MerkleProvenanceDAG {
    /// Create an empty DAG
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            heads: HashSet::new(),
            roots: HashSet::new(),
        }
    }

    /// Create a DAG with a single root node
    pub fn with_root(operation: ProvenanceOperation) -> Self {
        let mut dag = Self::new();
        dag.add_root(operation);
        dag
    }

    /// Add a root node (no parents)
    pub fn add_root(&mut self, operation: ProvenanceOperation) -> Hash256 {
        let node = MerkleProvenanceNode::root(operation);
        let id = node.id;
        self.nodes.insert(id, node);
        self.roots.insert(id);
        self.heads.insert(id);
        id
    }

    /// Add a derived node
    pub fn add_derived(
        &mut self,
        parents: Vec<Hash256>,
        operation: ProvenanceOperation,
    ) -> Hash256 {
        let node = MerkleProvenanceNode::derived(parents.clone(), operation);
        let id = node.id;

        // Update heads: remove parents from heads, add new node
        for parent in &parents {
            self.heads.remove(parent);
        }
        self.heads.insert(id);

        self.nodes.insert(id, node);
        id
    }

    /// Get a node by its hash
    pub fn get(&self, id: &Hash256) -> Option<&MerkleProvenanceNode> {
        self.nodes.get(id)
    }

    /// Get a mutable node by its hash
    pub fn get_mut(&mut self, id: &Hash256) -> Option<&mut MerkleProvenanceNode> {
        self.nodes.get_mut(id)
    }

    /// Get all head nodes
    pub fn heads(&self) -> impl Iterator<Item = &MerkleProvenanceNode> {
        self.heads.iter().filter_map(|id| self.nodes.get(id))
    }

    /// Get all root nodes
    pub fn roots(&self) -> impl Iterator<Item = &MerkleProvenanceNode> {
        self.roots.iter().filter_map(|id| self.nodes.get(id))
    }

    /// Verify integrity of entire DAG
    pub fn verify_all(&self) -> Result<(), (Hash256, HashError)> {
        for (id, node) in &self.nodes {
            node.verify().map_err(|e| (*id, e))?;

            // Verify all parents exist
            for parent in &node.parents {
                if !self.nodes.contains_key(parent) {
                    return Err((*id, HashError::VerificationFailed));
                }
            }
        }
        Ok(())
    }

    /// Get the full path from roots to a specific node
    pub fn path_to(&self, target: &Hash256) -> Vec<Hash256> {
        let mut path = Vec::new();
        let mut visited = HashSet::new();
        self.collect_ancestors(target, &mut path, &mut visited);
        path.reverse();
        path
    }

    fn collect_ancestors(
        &self,
        id: &Hash256,
        path: &mut Vec<Hash256>,
        visited: &mut HashSet<Hash256>,
    ) {
        if visited.contains(id) {
            return;
        }
        visited.insert(*id);

        if let Some(node) = self.nodes.get(id) {
            path.push(*id);
            for parent in &node.parents {
                self.collect_ancestors(parent, path, visited);
            }
        }
    }

    /// Compute the combined confidence factor from roots to heads
    pub fn total_confidence_factor(&self) -> f64 {
        let mut min_factor: f64 = 1.0;
        for head in self.heads() {
            let factor = self.confidence_factor_to(head);
            min_factor = min_factor.min(factor);
        }
        min_factor
    }

    fn confidence_factor_to(&self, node: &MerkleProvenanceNode) -> f64 {
        let mut factor = node.operation.confidence_factor();

        for parent_id in &node.parents {
            if let Some(parent) = self.nodes.get(parent_id) {
                factor *= self.confidence_factor_to(parent);
            }
        }

        factor
    }

    /// Generate audit trail for regulatory submission
    pub fn to_audit_trail(&self) -> AuditTrail {
        let mut entries = Vec::new();

        // Topological sort
        let sorted = self.topological_sort();

        for id in sorted {
            if let Some(node) = self.nodes.get(&id) {
                entries.push(AuditEntry {
                    hash: id.to_hex(),
                    operation: node.operation.name().to_string(),
                    operation_kind: format!("{:?}", node.operation.kind()),
                    timestamp: node.timestamp,
                    parents: node.parents.iter().map(|p| p.to_hex()).collect(),
                    user: node.metadata.user.clone(),
                    system: node.metadata.system.clone(),
                    signatures: node
                        .signatures
                        .iter()
                        .map(|s| s.authority.clone())
                        .collect(),
                });
            }
        }

        AuditTrail {
            entries,
            generated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            dag_hash: self.compute_dag_hash(),
        }
    }

    pub(crate) fn topological_sort(&self) -> Vec<Hash256> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();

        for root in &self.roots {
            self.topo_visit(root, &mut visited, &mut result);
        }

        result
    }

    fn topo_visit(&self, id: &Hash256, visited: &mut HashSet<Hash256>, result: &mut Vec<Hash256>) {
        if visited.contains(id) {
            return;
        }
        visited.insert(*id);
        result.push(*id);

        // Find children (nodes that have this as parent)
        for (child_id, node) in &self.nodes {
            if node.parents.contains(id) {
                self.topo_visit(child_id, visited, result);
            }
        }
    }

    /// Compute a hash of the entire DAG
    pub fn compute_dag_hash(&self) -> Hash256 {
        let mut hasher = Hasher::new();

        // Hash all head hashes (sorted for determinism)
        let mut heads: Vec<_> = self.heads.iter().collect();
        heads.sort_by_key(|h| h.to_hex());

        for head in heads {
            hasher.update(head.as_bytes());
        }

        hasher.finalize()
    }

    /// Number of nodes in the DAG
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if DAG is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for MerkleProvenanceDAG {
    fn default() -> Self {
        Self::new()
    }
}

/// Audit trail for regulatory submission
#[derive(Debug, Clone)]
pub struct AuditTrail {
    /// Ordered list of entries
    pub entries: Vec<AuditEntry>,
    /// When the audit trail was generated
    pub generated_at: u64,
    /// Hash of the entire DAG at generation time
    pub dag_hash: Hash256,
}

impl AuditTrail {
    /// Convert to JSON format
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\n");
        json.push_str(&format!("  \"generated_at\": {},\n", self.generated_at));
        json.push_str(&format!("  \"dag_hash\": \"{}\",\n", self.dag_hash));
        json.push_str("  \"entries\": [\n");

        for (i, entry) in self.entries.iter().enumerate() {
            json.push_str("    {\n");
            json.push_str(&format!("      \"hash\": \"{}\",\n", entry.hash));
            json.push_str(&format!("      \"operation\": \"{}\",\n", entry.operation));
            json.push_str(&format!("      \"kind\": \"{}\",\n", entry.operation_kind));
            json.push_str(&format!("      \"timestamp\": {},\n", entry.timestamp));
            json.push_str(&format!("      \"parents\": {:?},\n", entry.parents));
            json.push_str(&format!(
                "      \"user\": {},\n",
                entry
                    .user
                    .as_ref()
                    .map(|u| format!("\"{}\"", u))
                    .unwrap_or_else(|| "null".to_string())
            ));
            json.push_str(&format!("      \"signatures\": {:?}\n", entry.signatures));
            json.push_str("    }");
            if i < self.entries.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }

        json.push_str("  ]\n}");
        json
    }

    /// Convert to CSV format (for regulatory systems)
    pub fn to_csv(&self) -> String {
        let mut csv =
            String::from("hash,operation,kind,timestamp,parents,user,system,signatures\n");

        for entry in &self.entries {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                entry.hash,
                entry.operation,
                entry.operation_kind,
                entry.timestamp,
                entry.parents.join(";"),
                entry.user.as_deref().unwrap_or(""),
                entry.system.as_deref().unwrap_or(""),
                entry.signatures.join(";")
            ));
        }

        csv
    }
}

/// Single entry in an audit trail
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub hash: String,
    pub operation: String,
    pub operation_kind: String,
    pub timestamp: u64,
    pub parents: Vec<String>,
    pub user: Option<String>,
    pub system: Option<String>,
    pub signatures: Vec<String>,
}

// =============================================================================
// Integration with existing Provenance
// =============================================================================

use super::provenance::{FunctorTrace, Origin, Provenance, Transformation, TransformationKind};

impl From<&Provenance> for MerkleProvenanceDAG {
    fn from(prov: &Provenance) -> Self {
        let mut dag = MerkleProvenanceDAG::new();

        // Create root from origin
        let root_op = match &prov.origin {
            Origin::Literal => ProvenanceOperation::literal("value"),
            Origin::External { uri } => ProvenanceOperation::external(uri),
            Origin::Computed { function } => ProvenanceOperation::computation(function),
            Origin::UserInput { context } => {
                ProvenanceOperation::new(context, OperationKind::UserInput)
            }
            Origin::Database { query, .. } => {
                ProvenanceOperation::new(query, OperationKind::Database)
            }
            Origin::OntologyAssertion { ontology, term } => ProvenanceOperation::new(
                &format!("{}:{}", ontology, term),
                OperationKind::OntologyAssertion,
            ),
        };

        let mut current = dag.add_root(root_op);

        // Add each transformation
        for transform in &prov.trace.steps {
            let op =
                ProvenanceOperation::transformation(&transform.name, transform.confidence_factor);
            current = dag.add_derived(vec![current], op);
        }

        dag
    }
}

impl From<&MerkleProvenanceDAG> for Provenance {
    fn from(dag: &MerkleProvenanceDAG) -> Self {
        // Get the first root
        let root = dag.roots().next();

        let origin = root
            .map(|r| match r.operation.kind() {
                OperationKind::Literal => Origin::Literal,
                OperationKind::External => Origin::External {
                    uri: r.operation.name().to_string(),
                },
                OperationKind::Computation => Origin::Computed {
                    function: r.operation.name().to_string(),
                },
                _ => Origin::Literal,
            })
            .unwrap_or(Origin::Literal);

        // Collect transformations from the path
        let mut trace = FunctorTrace::empty();

        // Simple linear traversal for now (DAG to linear)
        for head in dag.heads() {
            let path = dag.path_to(&head.id);
            for id in path {
                if let Some(node) = dag.get(&id)
                    && node.operation.kind() == OperationKind::Transformation
                {
                    let transform =
                        Transformation::new(node.operation.name(), TransformationKind::Function)
                            .with_confidence(node.operation.confidence_factor());
                    trace = trace.append(transform);
                }
            }
        }

        Provenance {
            trace,
            origin,
            integrity_hash: Some(dag.compute_dag_hash().to_hex()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_computation() {
        let data = b"test data";
        let h1 = hash(data);
        let h2 = hash(data);
        assert_eq!(h1, h2);

        let h3 = hash(b"different data");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hash_hex_roundtrip() {
        let h = hash(b"test");
        let hex = h.to_hex();
        let h2 = Hash256::from_hex(&hex).unwrap();
        assert_eq!(h, h2);
    }

    #[test]
    fn test_merkle_node_creation() {
        let op = ProvenanceOperation::literal("42");
        let node = MerkleProvenanceNode::root(op);

        assert!(node.parents.is_empty());
        assert!(node.verify().is_ok());
    }

    #[test]
    fn test_merkle_dag_basic() {
        let mut dag = MerkleProvenanceDAG::new();

        let root1 = dag.add_root(ProvenanceOperation::measurement("sensor_a"));
        let root2 = dag.add_root(ProvenanceOperation::measurement("sensor_b"));

        let derived = dag.add_derived(
            vec![root1, root2],
            ProvenanceOperation::computation("average"),
        );

        assert_eq!(dag.len(), 3);
        assert_eq!(dag.heads().count(), 1);
        assert_eq!(dag.roots().count(), 2);

        let node = dag.get(&derived).unwrap();
        assert_eq!(node.parents.len(), 2);
    }

    #[test]
    fn test_dag_verification() {
        let mut dag = MerkleProvenanceDAG::new();
        dag.add_root(ProvenanceOperation::literal("test"));
        dag.add_derived(
            dag.roots().map(|n| n.id).collect(),
            ProvenanceOperation::computation("transform"),
        );

        assert!(dag.verify_all().is_ok());
    }

    #[test]
    fn test_confidence_factor() {
        let mut dag = MerkleProvenanceDAG::new();
        let root = dag.add_root(ProvenanceOperation::measurement("sensor"));
        dag.add_derived(
            vec![root],
            ProvenanceOperation::transformation("calibrate", 0.95),
        );
        dag.add_derived(
            dag.heads().map(|n| n.id).collect(),
            ProvenanceOperation::transformation("convert", 0.99),
        );

        let total = dag.total_confidence_factor();
        assert!((total - 0.95 * 0.99).abs() < 0.01);
    }

    #[test]
    fn test_audit_trail_generation() {
        let mut dag = MerkleProvenanceDAG::new();
        dag.add_root(ProvenanceOperation::measurement("thermometer"));
        dag.add_derived(
            dag.heads().map(|n| n.id).collect(),
            ProvenanceOperation::transformation("f_to_c", 1.0),
        );

        let trail = dag.to_audit_trail();
        assert_eq!(trail.entries.len(), 2);

        let json = trail.to_json();
        assert!(json.contains("thermometer"));
        assert!(json.contains("f_to_c"));
    }

    #[test]
    fn test_provenance_conversion() {
        // Create a classic Provenance
        let prov = Provenance::computed("test_fn")
            .extend(Transformation::function("step1").with_confidence(0.95))
            .extend(Transformation::function("step2").with_confidence(0.99));

        // Convert to MerkleProvenanceDAG
        let dag = MerkleProvenanceDAG::from(&prov);

        assert!(dag.len() >= 2); // At least root + transformations

        // Convert back
        let prov2 = Provenance::from(&dag);
        assert!(prov2.integrity_hash.is_some());
    }
}
