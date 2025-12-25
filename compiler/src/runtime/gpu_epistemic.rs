//! GPU Memory Layouts for Epistemic Types
//!
//! This module provides memory layouts optimized for GPU processing of
//! epistemic data. Two layouts are supported:
//!
//! 1. **Structure of Arrays (SoA)**: Optimal for coalesced memory access
//!    - Values stored contiguously
//!    - Confidences stored contiguously
//!    - Provenance hashes stored contiguously
//!    - Best for: SIMD operations, GPU kernels
//!
//! 2. **Array of Structures (AoS)**: Traditional layout
//!    - Each element contains all its fields
//!    - Better cache locality for single-element access
//!    - Best for: CPU sequential access
//!
//! The compiler automatically selects the optimal layout based on usage patterns.

/// GPU memory layout strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMemoryLayout {
    /// Structure of Arrays - optimal for GPU coalescing
    SoA,
    /// Array of Structures - traditional layout
    AoS,
    /// Hybrid - SoA for hot paths, AoS for cold
    Hybrid,
}

impl GpuMemoryLayout {
    /// Get recommended layout based on array size and access pattern
    pub fn recommend(size: usize, is_sequential: bool) -> Self {
        if size < 64 {
            // Small arrays: AoS is fine
            GpuMemoryLayout::AoS
        } else if is_sequential {
            // Large sequential access: SoA for coalescing
            GpuMemoryLayout::SoA
        } else {
            // Large random access: Hybrid
            GpuMemoryLayout::Hybrid
        }
    }
}

/// Structure of Arrays layout for epistemic data
///
/// This layout stores each field in a separate contiguous array,
/// enabling coalesced memory access on GPUs and efficient SIMD.
///
/// Memory layout:
/// ```text
/// values:      [v0, v1, v2, v3, ...]
/// confidences: [c0, c1, c2, c3, ...]
/// prov_hashes: [p0, p1, p2, p3, ...]
/// timestamps:  [t0, t1, t2, t3, ...]
/// ```
#[derive(Debug, Clone)]
pub struct SoAKnowledge<T> {
    /// Contiguous array of values
    pub values: Vec<T>,
    /// Contiguous array of quantized confidences (u16)
    pub confidences: Vec<u16>,
    /// Contiguous array of provenance hashes
    pub provenance_hashes: Vec<u32>,
    /// Contiguous array of timestamps
    pub timestamps: Vec<u16>,
    /// Capacity for pre-allocation
    capacity: usize,
}

impl<T> SoAKnowledge<T> {
    /// Create a new SoA with given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        SoAKnowledge {
            values: Vec::with_capacity(capacity),
            confidences: Vec::with_capacity(capacity),
            provenance_hashes: Vec::with_capacity(capacity),
            timestamps: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Create empty SoA
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Push a new element
    pub fn push(&mut self, value: T, confidence: f64, prov_hash: u32, timestamp: u16) {
        self.values.push(value);
        self.confidences
            .push((confidence.clamp(0.0, 1.0) * 65535.0) as u16);
        self.provenance_hashes.push(prov_hash);
        self.timestamps.push(timestamp);
    }

    /// Get value at index
    pub fn get_value(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    /// Get confidence at index
    pub fn get_confidence(&self, index: usize) -> Option<f64> {
        self.confidences.get(index).map(|&c| c as f64 / 65535.0)
    }

    /// Get provenance hash at index
    pub fn get_provenance(&self, index: usize) -> Option<u32> {
        self.provenance_hashes.get(index).copied()
    }

    /// Get all confidences as f64 slice (for SIMD operations)
    pub fn confidences_f64(&self) -> Vec<f64> {
        self.confidences
            .iter()
            .map(|&c| c as f64 / 65535.0)
            .collect()
    }

    /// Apply a transformation to all values
    pub fn map_values<U, F: Fn(&T) -> U>(&self, f: F) -> SoAKnowledge<U> {
        SoAKnowledge {
            values: self.values.iter().map(f).collect(),
            confidences: self.confidences.clone(),
            provenance_hashes: self.provenance_hashes.clone(),
            timestamps: self.timestamps.clone(),
            capacity: self.capacity,
        }
    }

    /// Filter elements by confidence threshold
    pub fn filter_by_confidence(&self, min_confidence: f64) -> SoAKnowledge<T>
    where
        T: Clone,
    {
        let threshold = (min_confidence.clamp(0.0, 1.0) * 65535.0) as u16;

        let mut result = SoAKnowledge::with_capacity(self.len());

        for i in 0..self.len() {
            if self.confidences[i] >= threshold {
                result.values.push(self.values[i].clone());
                result.confidences.push(self.confidences[i]);
                result.provenance_hashes.push(self.provenance_hashes[i]);
                result.timestamps.push(self.timestamps[i]);
            }
        }

        result
    }

    /// Get memory layout info
    pub fn memory_info(&self) -> SoAMemoryInfo {
        let value_size = std::mem::size_of::<T>();
        SoAMemoryInfo {
            element_count: self.len(),
            values_bytes: self.len() * value_size,
            confidences_bytes: self.len() * 2,
            provenance_bytes: self.len() * 4,
            timestamps_bytes: self.len() * 2,
            total_bytes: self.len() * (value_size + 8),
            layout: GpuMemoryLayout::SoA,
        }
    }
}

impl<T: Default> SoAKnowledge<T> {
    /// Create with default values
    pub fn with_defaults(count: usize) -> Self {
        let mut soa = Self::with_capacity(count);
        for _ in 0..count {
            soa.push(T::default(), 1.0, 0, 0);
        }
        soa
    }
}

impl<T> Default for SoAKnowledge<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory info for SoA layout
#[derive(Debug, Clone)]
pub struct SoAMemoryInfo {
    pub element_count: usize,
    pub values_bytes: usize,
    pub confidences_bytes: usize,
    pub provenance_bytes: usize,
    pub timestamps_bytes: usize,
    pub total_bytes: usize,
    pub layout: GpuMemoryLayout,
}

/// Array of Structures layout for epistemic data
///
/// Traditional layout where each element contains all its fields.
/// Better for cache locality when accessing all fields of one element.
#[derive(Debug, Clone)]
pub struct AoSKnowledge<T> {
    /// Array of complete elements
    elements: Vec<AoSElement<T>>,
}

/// Single element in AoS layout
#[derive(Debug, Clone)]
pub struct AoSElement<T> {
    pub value: T,
    pub confidence: u16,
    pub provenance_hash: u32,
    pub timestamp: u16,
}

impl<T> AoSKnowledge<T> {
    /// Create new empty AoS
    pub fn new() -> Self {
        AoSKnowledge {
            elements: Vec::new(),
        }
    }

    /// Create with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        AoSKnowledge {
            elements: Vec::with_capacity(capacity),
        }
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Push element
    pub fn push(&mut self, value: T, confidence: f64, prov_hash: u32, timestamp: u16) {
        self.elements.push(AoSElement {
            value,
            confidence: (confidence.clamp(0.0, 1.0) * 65535.0) as u16,
            provenance_hash: prov_hash,
            timestamp,
        });
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Option<&AoSElement<T>> {
        self.elements.get(index)
    }

    /// Get mutable element at index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut AoSElement<T>> {
        self.elements.get_mut(index)
    }

    /// Convert to SoA layout
    pub fn to_soa(&self) -> SoAKnowledge<T>
    where
        T: Clone,
    {
        let mut soa = SoAKnowledge::with_capacity(self.len());
        for elem in &self.elements {
            soa.values.push(elem.value.clone());
            soa.confidences.push(elem.confidence);
            soa.provenance_hashes.push(elem.provenance_hash);
            soa.timestamps.push(elem.timestamp);
        }
        soa
    }

    /// Iterate over elements
    pub fn iter(&self) -> impl Iterator<Item = &AoSElement<T>> {
        self.elements.iter()
    }
}

impl<T> Default for AoSKnowledge<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> SoAKnowledge<T> {
    /// Convert to AoS layout
    pub fn to_aos(&self) -> AoSKnowledge<T> {
        let mut aos = AoSKnowledge::with_capacity(self.len());
        for i in 0..self.len() {
            aos.elements.push(AoSElement {
                value: self.values[i].clone(),
                confidence: self.confidences[i],
                provenance_hash: self.provenance_hashes[i],
                timestamp: self.timestamps[i],
            });
        }
        aos
    }
}

/// GPU-optimized epistemic array with automatic layout selection
pub struct GpuEpistemicArray<T> {
    /// The data in chosen layout
    data: GpuEpistemicData<T>,
    /// Selected layout
    layout: GpuMemoryLayout,
    /// Array size
    size: usize,
}

enum GpuEpistemicData<T> {
    SoA(SoAKnowledge<T>),
    AoS(AoSKnowledge<T>),
}

impl<T> GpuEpistemicArray<T> {
    /// Create with automatic layout selection
    pub fn new(size: usize, sequential_access: bool) -> Self
    where
        T: Default,
    {
        let layout = GpuMemoryLayout::recommend(size, sequential_access);

        let data = match layout {
            GpuMemoryLayout::SoA | GpuMemoryLayout::Hybrid => {
                GpuEpistemicData::SoA(SoAKnowledge::with_defaults(size))
            }
            GpuMemoryLayout::AoS => {
                let mut aos = AoSKnowledge::with_capacity(size);
                for _ in 0..size {
                    aos.push(T::default(), 1.0, 0, 0);
                }
                GpuEpistemicData::AoS(aos)
            }
        };

        GpuEpistemicArray { data, layout, size }
    }

    /// Create from SoA data
    pub fn from_soa(soa: SoAKnowledge<T>) -> Self {
        let size = soa.len();
        GpuEpistemicArray {
            data: GpuEpistemicData::SoA(soa),
            layout: GpuMemoryLayout::SoA,
            size,
        }
    }

    /// Create from AoS data
    pub fn from_aos(aos: AoSKnowledge<T>) -> Self {
        let size = aos.len();
        GpuEpistemicArray {
            data: GpuEpistemicData::AoS(aos),
            layout: GpuMemoryLayout::AoS,
            size,
        }
    }

    /// Get the layout
    pub fn layout(&self) -> GpuMemoryLayout {
        self.layout
    }

    /// Get size
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get value at index
    pub fn get_value(&self, index: usize) -> Option<&T> {
        match &self.data {
            GpuEpistemicData::SoA(soa) => soa.get_value(index),
            GpuEpistemicData::AoS(aos) => aos.get(index).map(|e| &e.value),
        }
    }

    /// Get confidence at index
    pub fn get_confidence(&self, index: usize) -> Option<f64> {
        match &self.data {
            GpuEpistemicData::SoA(soa) => soa.get_confidence(index),
            GpuEpistemicData::AoS(aos) => aos.get(index).map(|e| e.confidence as f64 / 65535.0),
        }
    }

    /// Get raw confidence array (for SIMD/GPU operations)
    /// Returns None if not in SoA layout
    pub fn raw_confidences(&self) -> Option<&[u16]> {
        match &self.data {
            GpuEpistemicData::SoA(soa) => Some(&soa.confidences),
            GpuEpistemicData::AoS(_) => None,
        }
    }

    /// Get raw values array (for SIMD/GPU operations)
    /// Returns None if not in SoA layout
    pub fn raw_values(&self) -> Option<&[T]> {
        match &self.data {
            GpuEpistemicData::SoA(soa) => Some(&soa.values),
            GpuEpistemicData::AoS(_) => None,
        }
    }

    /// Convert to SoA layout (if not already)
    pub fn ensure_soa(&mut self)
    where
        T: Clone,
    {
        if let GpuEpistemicData::AoS(aos) = &self.data {
            self.data = GpuEpistemicData::SoA(aos.to_soa());
            self.layout = GpuMemoryLayout::SoA;
        }
    }

    /// Convert to AoS layout (if not already)
    pub fn ensure_aos(&mut self)
    where
        T: Clone,
    {
        if let GpuEpistemicData::SoA(soa) = &self.data {
            self.data = GpuEpistemicData::AoS(soa.to_aos());
            self.layout = GpuMemoryLayout::AoS;
        }
    }
}

/// SIMD-friendly confidence operations
pub mod simd_ops {
    /// Multiply all confidences by a factor (vectorizable)
    pub fn scale_confidences(confidences: &mut [u16], factor: f64) {
        let factor_q = (factor.clamp(0.0, 1.0) * 65535.0) as u32;
        for c in confidences.iter_mut() {
            *c = ((*c as u32 * factor_q) / 65535).min(65535) as u16;
        }
    }

    /// Find minimum confidence (vectorizable)
    pub fn min_confidence(confidences: &[u16]) -> Option<f64> {
        confidences.iter().min().map(|&c| c as f64 / 65535.0)
    }

    /// Find maximum confidence (vectorizable)
    pub fn max_confidence(confidences: &[u16]) -> Option<f64> {
        confidences.iter().max().map(|&c| c as f64 / 65535.0)
    }

    /// Count elements above threshold (vectorizable)
    pub fn count_above_threshold(confidences: &[u16], threshold: f64) -> usize {
        let threshold_q = (threshold.clamp(0.0, 1.0) * 65535.0) as u16;
        confidences.iter().filter(|&&c| c >= threshold_q).count()
    }

    /// Compute mean confidence
    pub fn mean_confidence(confidences: &[u16]) -> f64 {
        if confidences.is_empty() {
            return 0.0;
        }
        let sum: u64 = confidences.iter().map(|&c| c as u64).sum();
        (sum as f64 / confidences.len() as f64) / 65535.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_basic() {
        let mut soa: SoAKnowledge<i32> = SoAKnowledge::with_capacity(10);

        soa.push(1, 0.9, 100, 0);
        soa.push(2, 0.8, 200, 1);
        soa.push(3, 0.7, 300, 2);

        assert_eq!(soa.len(), 3);
        assert_eq!(soa.get_value(0), Some(&1));
        assert!((soa.get_confidence(1).unwrap() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_soa_filter() {
        let mut soa: SoAKnowledge<i32> = SoAKnowledge::new();

        soa.push(1, 0.9, 0, 0);
        soa.push(2, 0.5, 0, 0);
        soa.push(3, 0.95, 0, 0);

        let filtered = soa.filter_by_confidence(0.8);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered.get_value(0), Some(&1));
        assert_eq!(filtered.get_value(1), Some(&3));
    }

    #[test]
    fn test_aos_to_soa() {
        let mut aos: AoSKnowledge<i32> = AoSKnowledge::new();

        aos.push(1, 0.9, 100, 0);
        aos.push(2, 0.8, 200, 1);

        let soa = aos.to_soa();

        assert_eq!(soa.len(), 2);
        assert_eq!(soa.values, vec![1, 2]);
    }

    #[test]
    fn test_soa_to_aos() {
        let mut soa: SoAKnowledge<i32> = SoAKnowledge::new();

        soa.push(1, 0.9, 100, 0);
        soa.push(2, 0.8, 200, 1);

        let aos = soa.to_aos();

        assert_eq!(aos.len(), 2);
        assert_eq!(aos.get(0).unwrap().value, 1);
    }

    #[test]
    fn test_gpu_array_layout_selection() {
        // Small array should get AoS
        let small: GpuEpistemicArray<i32> = GpuEpistemicArray::new(10, false);
        assert_eq!(small.layout(), GpuMemoryLayout::AoS);

        // Large sequential should get SoA
        let large: GpuEpistemicArray<i32> = GpuEpistemicArray::new(1000, true);
        assert_eq!(large.layout(), GpuMemoryLayout::SoA);
    }

    #[test]
    fn test_simd_ops() {
        let mut confidences = vec![65535, 32768, 49152]; // 1.0, 0.5, 0.75

        simd_ops::scale_confidences(&mut confidences, 0.5);

        // All should be halved
        assert!((confidences[0] as f64 / 65535.0 - 0.5).abs() < 0.01);
        assert!((confidences[1] as f64 / 65535.0 - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_simd_mean() {
        let confidences = vec![65535, 32768, 0]; // 1.0, 0.5, 0.0
        let mean = simd_ops::mean_confidence(&confidences);

        assert!((mean - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_memory_layout_recommend() {
        assert_eq!(GpuMemoryLayout::recommend(10, false), GpuMemoryLayout::AoS);
        assert_eq!(GpuMemoryLayout::recommend(1000, true), GpuMemoryLayout::SoA);
        assert_eq!(
            GpuMemoryLayout::recommend(1000, false),
            GpuMemoryLayout::Hybrid
        );
    }
}
