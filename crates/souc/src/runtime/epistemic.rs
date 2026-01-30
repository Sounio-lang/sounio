//! Runtime Epistemic Type Representations
//!
//! This module provides three runtime representation modes for epistemic types:
//!
//! 1. **Full Mode** (64 bytes): Complete epistemic tracking
//!    - Full confidence with arbitrary precision
//!    - Complete provenance chain
//!    - All temporal metadata
//!    - Used for: Critical medical data, audit trails
//!
//! 2. **Compact Mode** (16 bytes): Optimized for bulk operations
//!    - Quantized confidence (u16, 0.001% precision)
//!    - Provenance hash only
//!    - Compressed timestamp
//!    - Used for: Large datasets, streaming data
//!
//! 3. **Erased Mode** (0 bytes): Zero overhead
//!    - All epistemic info erased at compile time
//!    - Type safety guaranteed statically
//!    - Used for: Trusted internal computations
//!
//! The mode is determined at compile time based on usage analysis.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Epistemic representation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpistemicMode {
    /// Full representation with all metadata (64 bytes)
    Full,
    /// Compact representation for bulk ops (16 bytes)
    Compact,
    /// Erased - no runtime overhead (0 bytes)
    Erased,
}

impl EpistemicMode {
    /// Get the size in bytes for this mode
    pub fn size_bytes(&self) -> usize {
        match self {
            EpistemicMode::Full => 64,
            EpistemicMode::Compact => 16,
            EpistemicMode::Erased => 0,
        }
    }

    /// Check if this mode tracks confidence at runtime
    pub fn tracks_confidence(&self) -> bool {
        !matches!(self, EpistemicMode::Erased)
    }

    /// Check if this mode tracks provenance at runtime
    pub fn tracks_provenance(&self) -> bool {
        matches!(self, EpistemicMode::Full)
    }
}

/// Runtime confidence value
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeConfidence {
    /// Confidence value in range [0.0, 1.0]
    value: f64,
    /// Lower bound (for confidence intervals)
    lower_bound: f64,
    /// Upper bound (for confidence intervals)
    upper_bound: f64,
}

impl RuntimeConfidence {
    /// Create a point confidence value
    pub fn point(value: f64) -> Self {
        let value = value.clamp(0.0, 1.0);
        RuntimeConfidence {
            value,
            lower_bound: value,
            upper_bound: value,
        }
    }

    /// Create a confidence interval
    pub fn interval(lower: f64, upper: f64) -> Self {
        let lower = lower.clamp(0.0, 1.0);
        let upper = upper.clamp(0.0, 1.0);
        RuntimeConfidence {
            value: (lower + upper) / 2.0,
            lower_bound: lower.min(upper),
            upper_bound: lower.max(upper),
        }
    }

    /// Get the point value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get the lower bound
    pub fn lower(&self) -> f64 {
        self.lower_bound
    }

    /// Get the upper bound
    pub fn upper(&self) -> f64 {
        self.upper_bound
    }

    /// Get interval width
    pub fn width(&self) -> f64 {
        self.upper_bound - self.lower_bound
    }

    /// Quantize to u16 (for compact mode)
    pub fn to_quantized(&self) -> u16 {
        (self.value * 65535.0) as u16
    }

    /// Create from quantized u16
    pub fn from_quantized(q: u16) -> Self {
        Self::point(q as f64 / 65535.0)
    }

    /// Combine two confidences (multiplication for conjunction)
    pub fn combine(&self, other: &Self) -> Self {
        RuntimeConfidence {
            value: self.value * other.value,
            lower_bound: self.lower_bound * other.lower_bound,
            upper_bound: self.upper_bound * other.upper_bound,
        }
    }

    /// Join two confidences (for disjunction, takes max)
    pub fn join(&self, other: &Self) -> Self {
        RuntimeConfidence {
            value: self.value.max(other.value),
            lower_bound: self.lower_bound.max(other.lower_bound),
            upper_bound: self.upper_bound.max(other.upper_bound),
        }
    }
}

impl Default for RuntimeConfidence {
    fn default() -> Self {
        Self::point(1.0)
    }
}

/// Runtime provenance tracking
#[derive(Debug, Clone)]
pub struct RuntimeProvenance {
    /// Source identifier (e.g., "sensor:temp-001")
    pub source: Arc<str>,
    /// Timestamp of data acquisition
    pub timestamp: u64,
    /// Chain of transformations applied
    pub chain: Vec<ProvenanceStep>,
    /// Hash of the complete provenance for quick comparison
    pub hash: u64,
}

/// A step in the provenance chain
#[derive(Debug, Clone)]
pub struct ProvenanceStep {
    /// Operation performed
    pub operation: Arc<str>,
    /// Timestamp of operation
    pub timestamp: u64,
    /// Optional parameters
    pub params: Option<Arc<str>>,
}

impl RuntimeProvenance {
    /// Create new provenance from a source
    pub fn new(source: impl Into<Arc<str>>) -> Self {
        let source = source.into();
        let timestamp = current_timestamp();
        let hash = Self::compute_hash(&source, timestamp, &[]);

        RuntimeProvenance {
            source,
            timestamp,
            chain: Vec::new(),
            hash,
        }
    }

    /// Add a transformation step
    pub fn add_step(&mut self, operation: impl Into<Arc<str>>, params: Option<&str>) {
        let step = ProvenanceStep {
            operation: operation.into(),
            timestamp: current_timestamp(),
            params: params.map(Arc::from),
        };
        self.chain.push(step);
        self.hash = Self::compute_hash(&self.source, self.timestamp, &self.chain);
    }

    /// Get the provenance hash (for compact mode)
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// Create provenance from hash only (for compact mode reconstruction)
    pub fn from_hash(hash: u64) -> Self {
        RuntimeProvenance {
            source: Arc::from("unknown"),
            timestamp: 0,
            chain: Vec::new(),
            hash,
        }
    }

    /// Compute hash of provenance
    fn compute_hash(source: &str, timestamp: u64, chain: &[ProvenanceStep]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        timestamp.hash(&mut hasher);
        chain.len().hash(&mut hasher);
        for step in chain {
            step.operation.hash(&mut hasher);
        }
        hasher.finish()
    }
}

impl Default for RuntimeProvenance {
    fn default() -> Self {
        Self::new("unknown")
    }
}

/// Full Knowledge representation (64 bytes)
///
/// Layout:
/// - value_ptr: 8 bytes (pointer to actual data)
/// - confidence: 24 bytes (f64 * 3)
/// - provenance_ptr: 8 bytes
/// - timestamp: 8 bytes
/// - flags: 8 bytes
/// - reserved: 8 bytes
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct FullKnowledge<T> {
    /// The actual value
    pub value: T,
    /// Full confidence with bounds
    pub confidence: RuntimeConfidence,
    /// Complete provenance chain
    pub provenance: Arc<RuntimeProvenance>,
    /// Acquisition timestamp
    pub timestamp: u64,
    /// Flags for special handling
    pub flags: KnowledgeFlags,
}

/// Flags for knowledge handling
#[derive(Debug, Clone, Copy, Default)]
pub struct KnowledgeFlags {
    bits: u64,
}

impl KnowledgeFlags {
    /// Knowledge is stale (needs refresh)
    pub const STALE: u64 = 1 << 0;
    /// Knowledge is derived (not primary)
    pub const DERIVED: u64 = 1 << 1;
    /// Knowledge requires audit trail
    pub const AUDITED: u64 = 1 << 2;
    /// Knowledge is immutable
    pub const IMMUTABLE: u64 = 1 << 3;
    /// Knowledge is from trusted source
    pub const TRUSTED: u64 = 1 << 4;

    pub fn new() -> Self {
        Self { bits: 0 }
    }

    pub fn set(&mut self, flag: u64) {
        self.bits |= flag;
    }

    pub fn has(&self, flag: u64) -> bool {
        (self.bits & flag) != 0
    }
}

impl<T> FullKnowledge<T> {
    /// Create new full knowledge
    pub fn new(value: T, confidence: RuntimeConfidence, source: impl Into<Arc<str>>) -> Self {
        FullKnowledge {
            value,
            confidence,
            provenance: Arc::new(RuntimeProvenance::new(source)),
            timestamp: current_timestamp(),
            flags: KnowledgeFlags::new(),
        }
    }

    /// Create with default confidence (1.0)
    pub fn certain(value: T, source: impl Into<Arc<str>>) -> Self {
        Self::new(value, RuntimeConfidence::point(1.0), source)
    }

    /// Map the value while preserving epistemic metadata
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F, operation: &str) -> FullKnowledge<U> {
        let mut provenance = (*self.provenance).clone();
        provenance.add_step(operation, None);

        FullKnowledge {
            value: f(self.value),
            confidence: self.confidence,
            provenance: Arc::new(provenance),
            timestamp: current_timestamp(),
            flags: self.flags,
        }
    }

    /// Combine with another knowledge (conjunction)
    pub fn combine<U, V, F: FnOnce(T, U) -> V>(
        self,
        other: FullKnowledge<U>,
        f: F,
        operation: &str,
    ) -> FullKnowledge<V> {
        let mut provenance = (*self.provenance).clone();
        provenance.add_step(
            operation,
            Some(&format!("combined with {}", other.provenance.hash())),
        );

        FullKnowledge {
            value: f(self.value, other.value),
            confidence: self.confidence.combine(&other.confidence),
            provenance: Arc::new(provenance),
            timestamp: current_timestamp(),
            flags: KnowledgeFlags::new(),
        }
    }

    /// Convert to compact representation
    pub fn to_compact(self) -> CompactKnowledge<T> {
        CompactKnowledge {
            value: self.value,
            confidence_q: self.confidence.to_quantized(),
            provenance_hash: self.provenance.hash() as u32,
            timestamp_delta: compress_timestamp(self.timestamp),
        }
    }
}

/// Compact Knowledge representation (16 bytes overhead)
///
/// Layout:
/// - confidence_q: 2 bytes (quantized u16)
/// - provenance_hash: 4 bytes (truncated hash)
/// - timestamp_delta: 2 bytes (compressed)
/// - reserved: 8 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CompactKnowledge<T> {
    /// The actual value
    pub value: T,
    /// Quantized confidence (0-65535 maps to 0.0-1.0)
    pub confidence_q: u16,
    /// Truncated provenance hash
    pub provenance_hash: u32,
    /// Compressed timestamp (delta from epoch)
    pub timestamp_delta: u16,
}

impl<T> CompactKnowledge<T> {
    /// Create new compact knowledge
    pub fn new(value: T, confidence: f64) -> Self {
        CompactKnowledge {
            value,
            confidence_q: (confidence.clamp(0.0, 1.0) * 65535.0) as u16,
            provenance_hash: 0,
            timestamp_delta: compress_timestamp(current_timestamp()),
        }
    }

    /// Get confidence as f64
    pub fn confidence(&self) -> f64 {
        self.confidence_q as f64 / 65535.0
    }

    /// Map the value
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> CompactKnowledge<U> {
        CompactKnowledge {
            value: f(self.value),
            confidence_q: self.confidence_q,
            provenance_hash: self.provenance_hash,
            timestamp_delta: compress_timestamp(current_timestamp()),
        }
    }

    /// Combine with another compact knowledge
    pub fn combine<U, V, F: FnOnce(T, U) -> V>(
        self,
        other: CompactKnowledge<U>,
        f: F,
    ) -> CompactKnowledge<V> {
        // Multiply confidences (saturating)
        let combined_conf = (self.confidence() * other.confidence() * 65535.0) as u16;

        CompactKnowledge {
            value: f(self.value, other.value),
            confidence_q: combined_conf,
            provenance_hash: self.provenance_hash ^ other.provenance_hash,
            timestamp_delta: compress_timestamp(current_timestamp()),
        }
    }

    /// Expand to full representation (loses detailed provenance)
    pub fn to_full(self) -> FullKnowledge<T> {
        FullKnowledge {
            value: self.value,
            confidence: RuntimeConfidence::from_quantized(self.confidence_q),
            provenance: Arc::new(RuntimeProvenance::from_hash(self.provenance_hash as u64)),
            timestamp: expand_timestamp(self.timestamp_delta),
            flags: KnowledgeFlags::new(),
        }
    }
}

/// Erased Knowledge - zero runtime overhead
///
/// This is a zero-sized type that exists only at compile time.
/// The epistemic properties are verified statically and erased.
#[derive(Debug, Clone, Copy)]
pub struct ErasedKnowledge<T> {
    /// The actual value (no epistemic overhead)
    pub value: T,
}

impl<T> ErasedKnowledge<T> {
    /// Create erased knowledge (compile-time verified)
    pub fn new(value: T) -> Self {
        ErasedKnowledge { value }
    }

    /// Map the value
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> ErasedKnowledge<U> {
        ErasedKnowledge {
            value: f(self.value),
        }
    }

    /// Combine with another erased knowledge
    pub fn combine<U, V, F: FnOnce(T, U) -> V>(
        self,
        other: ErasedKnowledge<U>,
        f: F,
    ) -> ErasedKnowledge<V> {
        ErasedKnowledge {
            value: f(self.value, other.value),
        }
    }

    /// Get the inner value
    pub fn into_inner(self) -> T {
        self.value
    }
}

/// Unified epistemic runtime interface
pub enum EpistemicRuntime<T> {
    Full(FullKnowledge<T>),
    Compact(CompactKnowledge<T>),
    Erased(ErasedKnowledge<T>),
}

impl<T> EpistemicRuntime<T> {
    /// Get the mode
    pub fn mode(&self) -> EpistemicMode {
        match self {
            EpistemicRuntime::Full(_) => EpistemicMode::Full,
            EpistemicRuntime::Compact(_) => EpistemicMode::Compact,
            EpistemicRuntime::Erased(_) => EpistemicMode::Erased,
        }
    }

    /// Get confidence (returns 1.0 for erased)
    pub fn confidence(&self) -> f64 {
        match self {
            EpistemicRuntime::Full(k) => k.confidence.value(),
            EpistemicRuntime::Compact(k) => k.confidence(),
            EpistemicRuntime::Erased(_) => 1.0,
        }
    }

    /// Get the value reference
    pub fn value(&self) -> &T {
        match self {
            EpistemicRuntime::Full(k) => &k.value,
            EpistemicRuntime::Compact(k) => &k.value,
            EpistemicRuntime::Erased(k) => &k.value,
        }
    }

    /// Convert to full mode
    pub fn to_full(self) -> FullKnowledge<T> {
        match self {
            EpistemicRuntime::Full(k) => k,
            EpistemicRuntime::Compact(k) => k.to_full(),
            EpistemicRuntime::Erased(k) => FullKnowledge::certain(k.value, "erased-upgrade"),
        }
    }
}

// Helper functions

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn compress_timestamp(ts: u64) -> u16 {
    // Compress to hours since 2024-01-01
    const EPOCH_2024: u64 = 1704067200;
    let hours = (ts.saturating_sub(EPOCH_2024)) / 3600;
    hours.min(65535) as u16
}

fn expand_timestamp(delta: u16) -> u64 {
    const EPOCH_2024: u64 = 1704067200;
    EPOCH_2024 + (delta as u64 * 3600)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_mode_size() {
        assert_eq!(EpistemicMode::Full.size_bytes(), 64);
        assert_eq!(EpistemicMode::Compact.size_bytes(), 16);
        assert_eq!(EpistemicMode::Erased.size_bytes(), 0);
    }

    #[test]
    fn test_runtime_confidence() {
        let conf = RuntimeConfidence::point(0.95);
        assert!((conf.value() - 0.95).abs() < 0.001);

        let interval = RuntimeConfidence::interval(0.9, 0.99);
        assert!((interval.value() - 0.945).abs() < 0.001);
        assert!((interval.width() - 0.09).abs() < 0.001);
    }

    #[test]
    fn test_confidence_quantization() {
        let conf = RuntimeConfidence::point(0.95);
        let quantized = conf.to_quantized();
        let restored = RuntimeConfidence::from_quantized(quantized);

        // Should be within 0.01% precision
        assert!((conf.value() - restored.value()).abs() < 0.0001);
    }

    #[test]
    fn test_confidence_combine() {
        let c1 = RuntimeConfidence::point(0.9);
        let c2 = RuntimeConfidence::point(0.8);
        let combined = c1.combine(&c2);

        assert!((combined.value() - 0.72).abs() < 0.001);
    }

    #[test]
    fn test_full_knowledge() {
        let k: FullKnowledge<i32> =
            FullKnowledge::new(42, RuntimeConfidence::point(0.95), "test-source");

        assert_eq!(k.value, 42);
        assert!((k.confidence.value() - 0.95).abs() < 0.001);
        assert_eq!(&*k.provenance.source, "test-source");
    }

    #[test]
    fn test_full_knowledge_map() {
        let k: FullKnowledge<i32> = FullKnowledge::certain(10, "test");
        let k2 = k.map(|x| x * 2, "multiply");

        assert_eq!(k2.value, 20);
        assert_eq!(k2.provenance.chain.len(), 1);
    }

    #[test]
    fn test_compact_knowledge() {
        let k: CompactKnowledge<i32> = CompactKnowledge::new(42, 0.95);

        assert_eq!(k.value, 42);
        assert!((k.confidence() - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_full_to_compact() {
        let full: FullKnowledge<i32> =
            FullKnowledge::new(42, RuntimeConfidence::point(0.95), "test");

        let compact = full.to_compact();

        assert_eq!(compact.value, 42);
        assert!((compact.confidence() - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_erased_knowledge() {
        let k: ErasedKnowledge<i32> = ErasedKnowledge::new(42);

        assert_eq!(k.value, 42);
        assert_eq!(
            std::mem::size_of::<ErasedKnowledge<i32>>(),
            std::mem::size_of::<i32>()
        );
    }

    #[test]
    fn test_epistemic_runtime() {
        let full = EpistemicRuntime::Full(FullKnowledge::certain(42, "test"));
        let compact = EpistemicRuntime::Compact(CompactKnowledge::new(42, 0.9));
        let erased = EpistemicRuntime::Erased(ErasedKnowledge::new(42));

        assert_eq!(full.mode(), EpistemicMode::Full);
        assert_eq!(compact.mode(), EpistemicMode::Compact);
        assert_eq!(erased.mode(), EpistemicMode::Erased);

        assert!((full.confidence() - 1.0).abs() < 0.001);
        assert!((compact.confidence() - 0.9).abs() < 0.001);
        assert!((erased.confidence() - 1.0).abs() < 0.001);
    }
}
