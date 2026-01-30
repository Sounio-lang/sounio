//! Peer-to-Peer (P2P) GPU Memory Transfers
//!
//! Provides GPUDirect-style P2P memory transfers between GPUs:
//! - Direct device-to-device memory copies
//! - P2P copy with reduction (for ring allreduce)
//! - Async transfer management
//!
//! # Architecture
//!
//! ```text
//! GPU 0 Memory ──────► GPU 1 Memory
//!      │                    │
//!      │    P2P Direct      │
//!      │   (NVLink/PCIe)    │
//!      │                    │
//!      └────────────────────┘
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use super::multi_gpu::{DeviceId, MultiGpuError, MultiGpuRuntime, P2PCapability};

// ============================================================================
// Transfer Types
// ============================================================================

/// Direction of a P2P transfer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Device to device
    DeviceToDevice,
    /// Device to host (not strictly P2P but included for completeness)
    DeviceToHost,
    /// Host to device
    HostToDevice,
}

/// A P2P transfer descriptor
#[derive(Debug, Clone)]
pub struct TransferDescriptor {
    /// Source device
    pub src_device: DeviceId,
    /// Source offset in bytes
    pub src_offset: usize,
    /// Destination device
    pub dst_device: DeviceId,
    /// Destination offset in bytes
    pub dst_offset: usize,
    /// Transfer size in bytes
    pub size: usize,
    /// Transfer direction
    pub direction: TransferDirection,
}

impl TransferDescriptor {
    /// Create a device-to-device transfer
    pub fn d2d(
        src_device: DeviceId,
        src_offset: usize,
        dst_device: DeviceId,
        dst_offset: usize,
        size: usize,
    ) -> Self {
        Self {
            src_device,
            src_offset,
            dst_device,
            dst_offset,
            size,
            direction: TransferDirection::DeviceToDevice,
        }
    }

    /// Create a simple transfer (offset 0 on both sides)
    pub fn simple(src_device: DeviceId, dst_device: DeviceId, size: usize) -> Self {
        Self::d2d(src_device, 0, dst_device, 0, size)
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Reduction operation for P2P copy with reduce
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum of values
    Sum,
    /// Product of values
    Product,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Average (sum / count)
    Average,
    /// Copy (no reduction, just overwrite)
    Copy,
}

impl ReduceOp {
    /// Get the identity element for this operation (f32)
    pub fn identity_f32(&self) -> f32 {
        match self {
            ReduceOp::Sum | ReduceOp::Average => 0.0,
            ReduceOp::Product => 1.0,
            ReduceOp::Min => f32::INFINITY,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Copy => 0.0,
        }
    }

    /// Get the identity element for this operation (f64)
    pub fn identity_f64(&self) -> f64 {
        match self {
            ReduceOp::Sum | ReduceOp::Average => 0.0,
            ReduceOp::Product => 1.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
            ReduceOp::Copy => 0.0,
        }
    }

    /// Apply reduction to two f32 values
    pub fn apply_f32(&self, a: f32, b: f32) -> f32 {
        match self {
            ReduceOp::Sum | ReduceOp::Average => a + b,
            ReduceOp::Product => a * b,
            ReduceOp::Min => a.min(b),
            ReduceOp::Max => a.max(b),
            ReduceOp::Copy => b,
        }
    }

    /// Apply reduction to two f64 values
    pub fn apply_f64(&self, a: f64, b: f64) -> f64 {
        match self {
            ReduceOp::Sum | ReduceOp::Average => a + b,
            ReduceOp::Product => a * b,
            ReduceOp::Min => a.min(b),
            ReduceOp::Max => a.max(b),
            ReduceOp::Copy => b,
        }
    }
}

// ============================================================================
// P2P Manager
// ============================================================================

/// Manager for P2P transfers between GPUs
pub struct P2PManager {
    /// Reference to multi-GPU runtime
    runtime: Arc<MultiGpuRuntime>,
    /// Enabled P2P pairs (bidirectional)
    enabled_pairs: HashSet<(DeviceId, DeviceId)>,
    /// Transfer statistics
    stats: P2PStats,
}

/// Statistics for P2P transfers
#[derive(Debug, Clone, Default)]
pub struct P2PStats {
    /// Total bytes transferred
    pub total_bytes: u64,
    /// Number of transfers
    pub transfer_count: u64,
    /// Number of failed transfers
    pub failed_count: u64,
}

impl P2PManager {
    /// Create a new P2P manager
    pub fn new(runtime: Arc<MultiGpuRuntime>) -> Self {
        // Copy enabled pairs from runtime
        let mut enabled_pairs = HashSet::new();
        for src in runtime.device_ids() {
            for dst in runtime.device_ids() {
                if runtime.is_p2p_enabled(src, dst) {
                    enabled_pairs.insert((src, dst));
                }
            }
        }

        Self {
            runtime,
            enabled_pairs,
            stats: P2PStats::default(),
        }
    }

    /// Enable P2P access between two devices
    pub fn enable_peer_access(
        &mut self,
        src: DeviceId,
        dst: DeviceId,
    ) -> Result<(), MultiGpuError> {
        if src == dst {
            return Ok(()); // Same device
        }

        // Check if P2P is supported
        let cap = self
            .runtime
            .p2p_capability(src, dst)
            .ok_or(MultiGpuError::DeviceNotFound(src))?;

        if !cap.peer_access {
            return Err(MultiGpuError::P2PNotAvailable { src, dst });
        }

        self.enabled_pairs.insert((src, dst));
        Ok(())
    }

    /// Disable P2P access between two devices
    pub fn disable_peer_access(&mut self, src: DeviceId, dst: DeviceId) {
        self.enabled_pairs.remove(&(src, dst));
    }

    /// Check if P2P is enabled between two devices
    pub fn is_enabled(&self, src: DeviceId, dst: DeviceId) -> bool {
        src == dst || self.enabled_pairs.contains(&(src, dst))
    }

    /// Get P2P capability between two devices
    pub fn capability(&self, src: DeviceId, dst: DeviceId) -> Option<&P2PCapability> {
        self.runtime.p2p_capability(src, dst)
    }

    /// Get transfer statistics
    pub fn stats(&self) -> &P2PStats {
        &self.stats
    }

    /// Reset transfer statistics
    pub fn reset_stats(&mut self) {
        self.stats = P2PStats::default();
    }

    /// Perform a P2P memory copy (simulated)
    ///
    /// In a real implementation, this would use:
    /// - cuMemcpyPeerAsync for CUDA
    /// - Similar APIs for other backends
    pub fn copy(&mut self, desc: &TransferDescriptor) -> Result<(), MultiGpuError> {
        // Validate P2P is enabled
        if !self.is_enabled(desc.src_device, desc.dst_device) {
            return Err(MultiGpuError::P2PNotAvailable {
                src: desc.src_device,
                dst: desc.dst_device,
            });
        }

        // Simulated: just update stats
        self.stats.total_bytes += desc.size as u64;
        self.stats.transfer_count += 1;

        Ok(())
    }

    /// Perform a P2P memory copy with reduction
    ///
    /// The destination buffer is reduced with the source:
    /// dst[i] = op(dst[i], src[i])
    pub fn copy_reduce(
        &mut self,
        desc: &TransferDescriptor,
        _op: ReduceOp,
    ) -> Result<(), MultiGpuError> {
        // Validate P2P is enabled
        if !self.is_enabled(desc.src_device, desc.dst_device) {
            return Err(MultiGpuError::P2PNotAvailable {
                src: desc.src_device,
                dst: desc.dst_device,
            });
        }

        // Simulated: just update stats
        self.stats.total_bytes += desc.size as u64;
        self.stats.transfer_count += 1;

        Ok(())
    }

    /// Perform multiple P2P copies in parallel
    pub fn copy_batch(&mut self, descriptors: &[TransferDescriptor]) -> Result<(), MultiGpuError> {
        for desc in descriptors {
            self.copy(desc)?;
        }
        Ok(())
    }

    /// Get the estimated transfer time in microseconds
    pub fn estimate_transfer_time_us(&self, src: DeviceId, dst: DeviceId, bytes: usize) -> f64 {
        if src == dst {
            return 0.0;
        }

        if let Some(cap) = self.capability(src, dst) {
            let bandwidth = cap.bandwidth_gbps();
            if bandwidth > 0.0 {
                // Time = bytes / bandwidth + latency
                let transfer_time = (bytes as f64) / (bandwidth * 1e9) * 1e6; // Convert to us
                let latency = cap.interconnect.latency_us();
                transfer_time + latency
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        }
    }
}

// ============================================================================
// Async Transfer Handle
// ============================================================================

/// Handle for an async P2P transfer
#[derive(Debug)]
pub struct AsyncTransfer {
    /// Transfer descriptor
    desc: TransferDescriptor,
    /// Transfer ID
    id: u64,
    /// Whether the transfer is complete
    complete: bool,
}

impl AsyncTransfer {
    /// Create a new async transfer
    pub fn new(desc: TransferDescriptor, id: u64) -> Self {
        Self {
            desc,
            id,
            complete: false,
        }
    }

    /// Get the transfer descriptor
    pub fn descriptor(&self) -> &TransferDescriptor {
        &self.desc
    }

    /// Get the transfer ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Check if the transfer is complete
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Wait for the transfer to complete (simulated)
    pub fn wait(&mut self) -> Result<(), MultiGpuError> {
        self.complete = true;
        Ok(())
    }
}

// ============================================================================
// Chunk-based Transfers
// ============================================================================

/// Chunk descriptor for pipelined transfers
#[derive(Debug, Clone)]
pub struct TransferChunk {
    /// Chunk index
    pub index: usize,
    /// Offset in source buffer
    pub src_offset: usize,
    /// Offset in destination buffer
    pub dst_offset: usize,
    /// Size of this chunk
    pub size: usize,
}

/// Split a transfer into chunks for pipelining
pub fn split_into_chunks(total_size: usize, chunk_size: usize) -> Vec<TransferChunk> {
    let num_chunks = total_size.div_ceil(chunk_size);
    let mut chunks = Vec::with_capacity(num_chunks);

    let mut offset = 0;
    for i in 0..num_chunks {
        let size = (total_size - offset).min(chunk_size);
        chunks.push(TransferChunk {
            index: i,
            src_offset: offset,
            dst_offset: offset,
            size,
        });
        offset += size;
    }

    chunks
}

// ============================================================================
// Ring Transfer Pattern
// ============================================================================

/// A single step in a ring transfer pattern
#[derive(Debug, Clone)]
pub struct RingTransferStep {
    /// Source device
    pub src: DeviceId,
    /// Destination device (next in ring)
    pub dst: DeviceId,
    /// Chunk index being transferred
    pub chunk_index: usize,
    /// Offset in buffer
    pub offset: usize,
    /// Size of chunk
    pub size: usize,
}

/// Generate ring transfer steps for reduce-scatter phase
pub fn generate_reduce_scatter_steps(
    ring: &[DeviceId],
    chunk_size: usize,
) -> Vec<Vec<RingTransferStep>> {
    let n = ring.len();
    if n < 2 {
        return Vec::new();
    }

    let mut all_steps = Vec::with_capacity(n - 1);

    for step in 0..(n - 1) {
        let mut step_transfers = Vec::with_capacity(n);

        for (i, &src) in ring.iter().enumerate() {
            let dst = ring[(i + 1) % n];
            // In step k, device i sends chunk (i - k - 1) mod n
            let chunk_index = (i + n - step - 1) % n;

            step_transfers.push(RingTransferStep {
                src,
                dst,
                chunk_index,
                offset: chunk_index * chunk_size,
                size: chunk_size,
            });
        }

        all_steps.push(step_transfers);
    }

    all_steps
}

/// Generate ring transfer steps for allgather phase
pub fn generate_allgather_steps(
    ring: &[DeviceId],
    chunk_size: usize,
) -> Vec<Vec<RingTransferStep>> {
    let n = ring.len();
    if n < 2 {
        return Vec::new();
    }

    let mut all_steps = Vec::with_capacity(n - 1);

    for step in 0..(n - 1) {
        let mut step_transfers = Vec::with_capacity(n);

        for (i, &src) in ring.iter().enumerate() {
            let dst = ring[(i + 1) % n];
            // In step k, device i sends chunk (i - k) mod n
            let chunk_index = (i + n - step) % n;

            step_transfers.push(RingTransferStep {
                src,
                dst,
                chunk_index,
                offset: chunk_index * chunk_size,
                size: chunk_size,
            });
        }

        all_steps.push(step_transfers);
    }

    all_steps
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_descriptor() {
        let desc = TransferDescriptor::simple(DeviceId(0), DeviceId(1), 1024);
        assert_eq!(desc.src_device, DeviceId(0));
        assert_eq!(desc.dst_device, DeviceId(1));
        assert_eq!(desc.size, 1024);
        assert_eq!(desc.direction, TransferDirection::DeviceToDevice);
    }

    #[test]
    fn test_reduce_op() {
        assert_eq!(ReduceOp::Sum.apply_f32(1.0, 2.0), 3.0);
        assert_eq!(ReduceOp::Product.apply_f32(2.0, 3.0), 6.0);
        assert_eq!(ReduceOp::Min.apply_f32(1.0, 2.0), 1.0);
        assert_eq!(ReduceOp::Max.apply_f32(1.0, 2.0), 2.0);
        assert_eq!(ReduceOp::Copy.apply_f32(1.0, 2.0), 2.0);
    }

    #[test]
    fn test_p2p_manager() {
        let runtime = Arc::new(MultiGpuRuntime::simulated(4));
        let mut manager = P2PManager::new(runtime);

        assert!(manager.is_enabled(DeviceId(0), DeviceId(1)));
        assert!(manager.is_enabled(DeviceId(0), DeviceId(0))); // Same device

        let desc = TransferDescriptor::simple(DeviceId(0), DeviceId(1), 1024);
        manager.copy(&desc).unwrap();

        assert_eq!(manager.stats().transfer_count, 1);
        assert_eq!(manager.stats().total_bytes, 1024);
    }

    #[test]
    fn test_split_into_chunks() {
        let chunks = split_into_chunks(1000, 300);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].size, 300);
        assert_eq!(chunks[3].size, 100); // Last chunk is smaller
    }

    #[test]
    fn test_ring_transfer_steps() {
        let ring = vec![DeviceId(0), DeviceId(1), DeviceId(2), DeviceId(3)];
        let steps = generate_reduce_scatter_steps(&ring, 256);

        // For 4 devices, we need 3 steps
        assert_eq!(steps.len(), 3);

        // Each step has 4 transfers (one per device)
        for step in &steps {
            assert_eq!(step.len(), 4);
        }
    }

    #[test]
    fn test_estimate_transfer_time() {
        let runtime = Arc::new(MultiGpuRuntime::simulated(4));
        let manager = P2PManager::new(runtime);

        // Same device should be instant
        let time = manager.estimate_transfer_time_us(DeviceId(0), DeviceId(0), 1024);
        assert_eq!(time, 0.0);

        // Different devices should have some transfer time
        let time = manager.estimate_transfer_time_us(DeviceId(0), DeviceId(1), 1024 * 1024);
        assert!(time > 0.0);
    }
}
