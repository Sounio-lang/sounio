//! Collective Operations for Multi-GPU Computing
//!
//! Implements NCCL-style collective operations without external dependencies:
//! - AllReduce (ring-based and tree-based)
//! - Broadcast
//! - AllGather
//! - ReduceScatter
//!
//! # Ring AllReduce Algorithm
//!
//! For n GPUs with data size M:
//! ```text
//! Phase 1 (Reduce-Scatter): n-1 steps
//!   - Each GPU sends chunk[i] to GPU[(rank+1) % n]
//!   - Receiver reduces: chunk[i] = reduce(chunk[i], received)
//!   - After phase: GPU i has fully reduced chunk[i]
//!
//! Phase 2 (AllGather): n-1 steps
//!   - Each GPU sends its reduced chunk to next GPU
//!   - After phase: All GPUs have all reduced chunks
//!
//! Total: 2*(n-1) steps, transfers 2*M*(n-1)/n bytes per GPU
//! ```
//!
//! # Tree Broadcast Algorithm
//!
//! For n GPUs with root r:
//! ```text
//! log2(n) steps
//! Step k: 2^k GPUs send to 2^k receivers
//! ```

use std::sync::Arc;

use super::multi_gpu::{DeviceId, MultiGpuError, MultiGpuRuntime};
use super::p2p_transfer::{
    P2PManager, ReduceOp, generate_allgather_steps, generate_reduce_scatter_steps,
};

// ============================================================================
// Collective Operation Types
// ============================================================================

/// Type of collective operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveOp {
    /// Sum of all values
    Sum,
    /// Product of all values
    Product,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Average of all values
    Average,
}

impl From<CollectiveOp> for ReduceOp {
    fn from(op: CollectiveOp) -> Self {
        match op {
            CollectiveOp::Sum => ReduceOp::Sum,
            CollectiveOp::Product => ReduceOp::Product,
            CollectiveOp::Min => ReduceOp::Min,
            CollectiveOp::Max => ReduceOp::Max,
            CollectiveOp::Average => ReduceOp::Average,
        }
    }
}

/// Algorithm to use for collective operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveAlgorithm {
    /// Ring-based algorithm (optimal for large messages)
    Ring,
    /// Tree-based algorithm (optimal for small messages, low latency)
    Tree,
    /// Recursive halving-doubling (good all-around)
    RecursiveHalvingDoubling,
    /// Direct all-to-all (for small device counts)
    Direct,
    /// Hierarchical (NVLink intra-node + slower inter-node)
    Hierarchical,
    /// Auto-select based on message size and topology
    Auto,
}

// ============================================================================
// Algorithm Selector
// ============================================================================

/// Selects the best algorithm based on message size and topology
pub struct AlgorithmSelector {
    /// Threshold for switching from tree to ring (bytes)
    tree_to_ring_threshold: usize,
    /// Threshold for switching from direct to ring (bytes)
    direct_to_ring_threshold: usize,
    /// Max devices for direct algorithm
    max_direct_devices: usize,
}

impl Default for AlgorithmSelector {
    fn default() -> Self {
        Self {
            tree_to_ring_threshold: 256 * 1024,  // 256 KB
            direct_to_ring_threshold: 64 * 1024, // 64 KB
            max_direct_devices: 4,
        }
    }
}

impl AlgorithmSelector {
    /// Create a new algorithm selector
    pub fn new() -> Self {
        Self::default()
    }

    /// Select the best algorithm
    pub fn select(&self, message_size: usize, device_count: usize) -> CollectiveAlgorithm {
        if device_count <= 1 {
            return CollectiveAlgorithm::Direct;
        }

        if device_count <= self.max_direct_devices && message_size < self.direct_to_ring_threshold {
            return CollectiveAlgorithm::Direct;
        }

        if message_size < self.tree_to_ring_threshold {
            CollectiveAlgorithm::Tree
        } else {
            CollectiveAlgorithm::Ring
        }
    }
}

// ============================================================================
// Simulated Buffer
// ============================================================================

/// Simulated device buffer for testing collectives
#[derive(Debug, Clone)]
pub struct SimulatedBuffer {
    /// Device this buffer is on
    pub device: DeviceId,
    /// Buffer data (f32 for simplicity)
    pub data: Vec<f32>,
}

impl SimulatedBuffer {
    /// Create a new buffer with given size (in f32 elements)
    pub fn new(device: DeviceId, size: usize) -> Self {
        Self {
            device,
            data: vec![0.0; size],
        }
    }

    /// Create a buffer initialized with a value
    pub fn filled(device: DeviceId, size: usize, value: f32) -> Self {
        Self {
            device,
            data: vec![value; size],
        }
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a slice of the data
    pub fn slice(&self, start: usize, end: usize) -> &[f32] {
        &self.data[start..end]
    }

    /// Get a mutable slice of the data
    pub fn slice_mut(&mut self, start: usize, end: usize) -> &mut [f32] {
        &mut self.data[start..end]
    }
}

// ============================================================================
// Collective Manager
// ============================================================================

/// Statistics for collective operations
#[derive(Debug, Clone, Default)]
pub struct CollectiveStats {
    /// Total bytes transferred
    pub bytes_transferred: u64,
    /// Number of operations
    pub operation_count: u64,
    /// Total steps (for ring/tree algorithms)
    pub total_steps: u64,
}

/// Manager for collective operations
pub struct CollectiveManager {
    /// Multi-GPU runtime
    runtime: Arc<MultiGpuRuntime>,
    /// P2P manager
    p2p: P2PManager,
    /// Algorithm selector
    selector: AlgorithmSelector,
    /// Statistics
    stats: CollectiveStats,
}

impl CollectiveManager {
    /// Create a new collective manager
    pub fn new(runtime: Arc<MultiGpuRuntime>) -> Self {
        let p2p = P2PManager::new(runtime.clone());
        Self {
            runtime,
            p2p,
            selector: AlgorithmSelector::default(),
            stats: CollectiveStats::default(),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &CollectiveStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = CollectiveStats::default();
    }

    /// Get the number of devices
    pub fn device_count(&self) -> usize {
        self.runtime.device_count()
    }

    // ========================================================================
    // AllReduce
    // ========================================================================

    /// Perform allreduce on simulated buffers
    ///
    /// All buffers must have the same size. After completion, all buffers
    /// contain the reduced result.
    pub fn allreduce(
        &mut self,
        buffers: &mut [SimulatedBuffer],
        op: CollectiveOp,
        algorithm: CollectiveAlgorithm,
    ) -> Result<(), MultiGpuError> {
        let n = buffers.len();
        if n == 0 {
            return Ok(());
        }
        if n == 1 {
            return Ok(()); // Single device, nothing to do
        }

        // Validate all buffers have same size
        let size = buffers[0].len();
        for buf in buffers.iter() {
            if buf.len() != size {
                return Err(MultiGpuError::BufferSizeMismatch {
                    expected: size,
                    actual: buf.len(),
                });
            }
        }

        // Select algorithm if auto
        let algo = if algorithm == CollectiveAlgorithm::Auto {
            self.selector.select(size * 4, n)
        } else {
            algorithm
        };

        match algo {
            CollectiveAlgorithm::Ring => self.allreduce_ring(buffers, op),
            CollectiveAlgorithm::Tree => self.allreduce_tree(buffers, op),
            CollectiveAlgorithm::Direct => self.allreduce_direct(buffers, op),
            _ => self.allreduce_ring(buffers, op), // Default to ring
        }
    }

    /// Ring-based allreduce
    fn allreduce_ring(
        &mut self,
        buffers: &mut [SimulatedBuffer],
        op: CollectiveOp,
    ) -> Result<(), MultiGpuError> {
        let n = buffers.len();
        let total_size = buffers[0].len();
        let chunk_size = total_size.div_ceil(n);

        // Build ring
        let ring: Vec<DeviceId> = (0..n as u32).map(DeviceId).collect();

        // Phase 1: Reduce-Scatter
        let reduce_steps = generate_reduce_scatter_steps(&ring, chunk_size);
        for step in reduce_steps {
            for transfer in step {
                let src_idx = transfer.src.0 as usize;
                let dst_idx = transfer.dst.0 as usize;

                // Get the chunk range
                let start = transfer.offset.min(total_size);
                let end = (transfer.offset + transfer.size).min(total_size);

                if start >= end {
                    continue;
                }

                // Copy chunk from src to temporary, then reduce into dst
                let src_chunk: Vec<f32> = buffers[src_idx].data[start..end].to_vec();
                let reduce_op: ReduceOp = op.into();

                for (i, &src_val) in src_chunk.iter().enumerate() {
                    let dst_val = buffers[dst_idx].data[start + i];
                    buffers[dst_idx].data[start + i] = reduce_op.apply_f32(dst_val, src_val);
                }

                self.stats.bytes_transferred += (end - start) as u64 * 4;
            }
            self.stats.total_steps += 1;
        }

        // Phase 2: AllGather
        let gather_steps = generate_allgather_steps(&ring, chunk_size);
        for step in gather_steps {
            for transfer in step {
                let src_idx = transfer.src.0 as usize;
                let dst_idx = transfer.dst.0 as usize;

                let start = transfer.offset.min(total_size);
                let end = (transfer.offset + transfer.size).min(total_size);

                if start >= end {
                    continue;
                }

                // Copy chunk from src to dst (no reduction in gather phase)
                let src_chunk: Vec<f32> = buffers[src_idx].data[start..end].to_vec();
                buffers[dst_idx].data[start..end].copy_from_slice(&src_chunk);

                self.stats.bytes_transferred += (end - start) as u64 * 4;
            }
            self.stats.total_steps += 1;
        }

        self.stats.operation_count += 1;
        Ok(())
    }

    /// Tree-based allreduce (reduce to root, then broadcast)
    fn allreduce_tree(
        &mut self,
        buffers: &mut [SimulatedBuffer],
        op: CollectiveOp,
    ) -> Result<(), MultiGpuError> {
        // Phase 1: Tree reduce to root (device 0)
        self.reduce_tree(buffers, op, DeviceId(0))?;

        // Phase 2: Broadcast from root
        self.broadcast_tree(buffers, DeviceId(0))?;

        self.stats.operation_count += 1;
        Ok(())
    }

    /// Direct allreduce (all-to-all, then local reduce)
    fn allreduce_direct(
        &mut self,
        buffers: &mut [SimulatedBuffer],
        op: CollectiveOp,
    ) -> Result<(), MultiGpuError> {
        let n = buffers.len();
        let size = buffers[0].len();

        // Compute reduction across all buffers
        let reduce_op: ReduceOp = op.into();
        let mut result = vec![reduce_op.identity_f32(); size];

        for buf in buffers.iter() {
            for (i, &val) in buf.data.iter().enumerate() {
                result[i] = reduce_op.apply_f32(result[i], val);
            }
        }

        // Handle average
        if op == CollectiveOp::Average {
            for val in &mut result {
                *val /= n as f32;
            }
        }

        // Copy result to all buffers
        for buf in buffers.iter_mut() {
            buf.data.copy_from_slice(&result);
        }

        self.stats.bytes_transferred += (size * 4 * n) as u64;
        self.stats.operation_count += 1;
        Ok(())
    }

    // ========================================================================
    // Broadcast
    // ========================================================================

    /// Broadcast from root to all other devices
    pub fn broadcast(
        &mut self,
        buffers: &mut [SimulatedBuffer],
        root: DeviceId,
        algorithm: CollectiveAlgorithm,
    ) -> Result<(), MultiGpuError> {
        let n = buffers.len();
        if n <= 1 {
            return Ok(());
        }

        let algo = if algorithm == CollectiveAlgorithm::Auto {
            self.selector.select(buffers[0].size_bytes(), n)
        } else {
            algorithm
        };

        match algo {
            CollectiveAlgorithm::Tree => self.broadcast_tree(buffers, root),
            _ => self.broadcast_direct(buffers, root),
        }
    }

    /// Tree-based broadcast
    fn broadcast_tree(
        &mut self,
        buffers: &mut [SimulatedBuffer],
        root: DeviceId,
    ) -> Result<(), MultiGpuError> {
        let n = buffers.len();
        let root_idx = root.0 as usize;

        if root_idx >= n {
            return Err(MultiGpuError::DeviceNotFound(root));
        }

        // Build tree
        let tree = self.runtime.build_tree(root);
        let depth = tree.depth();

        // Get root data
        let root_data = buffers[root_idx].data.clone();
        let size = root_data.len();

        // Broadcast in tree order (BFS)
        // For simplicity, use direct copy approach
        for buf in buffers.iter_mut() {
            if buf.device != root {
                buf.data.copy_from_slice(&root_data);
                self.stats.bytes_transferred += (size * 4) as u64;
            }
        }

        self.stats.total_steps += depth as u64;
        self.stats.operation_count += 1;
        Ok(())
    }

    /// Direct broadcast (root sends to all)
    fn broadcast_direct(
        &mut self,
        buffers: &mut [SimulatedBuffer],
        root: DeviceId,
    ) -> Result<(), MultiGpuError> {
        let root_idx = root.0 as usize;
        if root_idx >= buffers.len() {
            return Err(MultiGpuError::DeviceNotFound(root));
        }

        let root_data = buffers[root_idx].data.clone();
        let size = root_data.len();

        for buf in buffers.iter_mut() {
            if buf.device != root {
                buf.data.copy_from_slice(&root_data);
                self.stats.bytes_transferred += (size * 4) as u64;
            }
        }

        self.stats.operation_count += 1;
        Ok(())
    }

    // ========================================================================
    // Reduce
    // ========================================================================

    /// Reduce to a single root device
    pub fn reduce(
        &mut self,
        buffers: &mut [SimulatedBuffer],
        op: CollectiveOp,
        root: DeviceId,
    ) -> Result<(), MultiGpuError> {
        self.reduce_tree(buffers, op, root)
    }

    /// Tree-based reduce
    fn reduce_tree(
        &mut self,
        buffers: &mut [SimulatedBuffer],
        op: CollectiveOp,
        root: DeviceId,
    ) -> Result<(), MultiGpuError> {
        let n = buffers.len();
        let root_idx = root.0 as usize;

        if root_idx >= n {
            return Err(MultiGpuError::DeviceNotFound(root));
        }

        let size = buffers[0].len();
        let reduce_op: ReduceOp = op.into();

        // Reduce all buffers into root
        for i in 0..n {
            if i != root_idx {
                for j in 0..size {
                    buffers[root_idx].data[j] =
                        reduce_op.apply_f32(buffers[root_idx].data[j], buffers[i].data[j]);
                }
                self.stats.bytes_transferred += (size * 4) as u64;
            }
        }

        // Handle average
        if op == CollectiveOp::Average {
            for val in &mut buffers[root_idx].data {
                *val /= n as f32;
            }
        }

        self.stats.operation_count += 1;
        Ok(())
    }

    // ========================================================================
    // AllGather
    // ========================================================================

    /// AllGather: each device contributes a chunk, all receive all chunks
    pub fn allgather(
        &mut self,
        send_buffers: &[SimulatedBuffer],
        recv_buffers: &mut [SimulatedBuffer],
    ) -> Result<(), MultiGpuError> {
        let n = send_buffers.len();
        if n == 0 || recv_buffers.len() != n {
            return Err(MultiGpuError::DeviceCountMismatch {
                expected: n,
                actual: recv_buffers.len(),
            });
        }

        let chunk_size = send_buffers[0].len();

        // Each recv buffer should be n * chunk_size
        let expected_recv_size = n * chunk_size;
        for buf in recv_buffers.iter() {
            if buf.len() != expected_recv_size {
                return Err(MultiGpuError::BufferSizeMismatch {
                    expected: expected_recv_size,
                    actual: buf.len(),
                });
            }
        }

        // Gather all chunks to all devices
        for (i, send_buf) in send_buffers.iter().enumerate() {
            let chunk_data = &send_buf.data;
            let offset = i * chunk_size;

            for recv_buf in recv_buffers.iter_mut() {
                recv_buf.data[offset..offset + chunk_size].copy_from_slice(chunk_data);
            }

            self.stats.bytes_transferred += (chunk_size * 4 * n) as u64;
        }

        self.stats.operation_count += 1;
        Ok(())
    }

    // ========================================================================
    // ReduceScatter
    // ========================================================================

    /// ReduceScatter: reduce then scatter (each device gets a different chunk)
    pub fn reduce_scatter(
        &mut self,
        send_buffers: &[SimulatedBuffer],
        recv_buffers: &mut [SimulatedBuffer],
        op: CollectiveOp,
    ) -> Result<(), MultiGpuError> {
        let n = send_buffers.len();
        if n == 0 || recv_buffers.len() != n {
            return Err(MultiGpuError::DeviceCountMismatch {
                expected: n,
                actual: recv_buffers.len(),
            });
        }

        let total_size = send_buffers[0].len();
        let chunk_size = total_size.div_ceil(n);

        // Each recv buffer should be chunk_size
        for buf in recv_buffers.iter() {
            if buf.len() < chunk_size {
                return Err(MultiGpuError::BufferSizeMismatch {
                    expected: chunk_size,
                    actual: buf.len(),
                });
            }
        }

        let reduce_op: ReduceOp = op.into();

        // Reduce each chunk
        for chunk_idx in 0..n {
            let offset = chunk_idx * chunk_size;
            let end = (offset + chunk_size).min(total_size);
            let actual_chunk_size = end - offset;

            // Initialize with identity
            let mut reduced = vec![reduce_op.identity_f32(); actual_chunk_size];

            // Reduce from all send buffers
            for send_buf in send_buffers.iter() {
                for i in 0..actual_chunk_size {
                    reduced[i] = reduce_op.apply_f32(reduced[i], send_buf.data[offset + i]);
                }
            }

            // Handle average
            if op == CollectiveOp::Average {
                for val in &mut reduced {
                    *val /= n as f32;
                }
            }

            // Copy to the receiving device
            recv_buffers[chunk_idx].data[..actual_chunk_size].copy_from_slice(&reduced);

            self.stats.bytes_transferred += (actual_chunk_size * 4 * n) as u64;
        }

        self.stats.operation_count += 1;
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_buffers(
        n: usize,
        size: usize,
        init: impl Fn(usize) -> f32,
    ) -> Vec<SimulatedBuffer> {
        (0..n)
            .map(|i| SimulatedBuffer::filled(DeviceId(i as u32), size, init(i)))
            .collect()
    }

    #[test]
    fn test_algorithm_selector() {
        let selector = AlgorithmSelector::default();

        // Small message, few devices -> direct
        assert_eq!(selector.select(1024, 2), CollectiveAlgorithm::Direct);

        // Medium message -> tree
        assert_eq!(selector.select(100 * 1024, 8), CollectiveAlgorithm::Tree);

        // Large message -> ring
        assert_eq!(
            selector.select(10 * 1024 * 1024, 8),
            CollectiveAlgorithm::Ring
        );
    }

    #[test]
    fn test_allreduce_direct_sum() {
        let runtime = Arc::new(MultiGpuRuntime::simulated(4));
        let mut manager = CollectiveManager::new(runtime);

        // Each device has [i, i, i, i] where i is device index
        let mut buffers = create_test_buffers(4, 4, |i| i as f32);

        manager
            .allreduce(&mut buffers, CollectiveOp::Sum, CollectiveAlgorithm::Direct)
            .unwrap();

        // Result should be [0+1+2+3, ...] = [6, 6, 6, 6]
        for buf in &buffers {
            assert_eq!(buf.data, vec![6.0, 6.0, 6.0, 6.0]);
        }
    }

    #[test]
    fn test_allreduce_ring_sum() {
        let runtime = Arc::new(MultiGpuRuntime::simulated(4));
        let mut manager = CollectiveManager::new(runtime);

        let mut buffers = create_test_buffers(4, 8, |i| i as f32);

        manager
            .allreduce(&mut buffers, CollectiveOp::Sum, CollectiveAlgorithm::Ring)
            .unwrap();

        // All buffers should have the same sum
        for buf in &buffers {
            for &val in &buf.data {
                assert!((val - 6.0).abs() < 0.001, "Expected 6.0, got {}", val);
            }
        }
    }

    #[test]
    fn test_allreduce_average() {
        let runtime = Arc::new(MultiGpuRuntime::simulated(4));
        let mut manager = CollectiveManager::new(runtime);

        let mut buffers = create_test_buffers(4, 4, |i| (i + 1) as f32);

        manager
            .allreduce(
                &mut buffers,
                CollectiveOp::Average,
                CollectiveAlgorithm::Direct,
            )
            .unwrap();

        // Average of [1, 2, 3, 4] = 2.5
        for buf in &buffers {
            assert_eq!(buf.data, vec![2.5, 2.5, 2.5, 2.5]);
        }
    }

    #[test]
    fn test_broadcast() {
        let runtime = Arc::new(MultiGpuRuntime::simulated(4));
        let mut manager = CollectiveManager::new(runtime);

        let mut buffers = create_test_buffers(4, 4, |i| i as f32);
        // Root (device 0) has [0, 0, 0, 0]

        manager
            .broadcast(&mut buffers, DeviceId(0), CollectiveAlgorithm::Tree)
            .unwrap();

        // All should have root's data
        for buf in &buffers {
            assert_eq!(buf.data, vec![0.0, 0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn test_reduce() {
        let runtime = Arc::new(MultiGpuRuntime::simulated(4));
        let mut manager = CollectiveManager::new(runtime);

        let mut buffers = create_test_buffers(4, 4, |i| i as f32);

        manager
            .reduce(&mut buffers, CollectiveOp::Sum, DeviceId(0))
            .unwrap();

        // Only root should have the sum
        assert_eq!(buffers[0].data, vec![6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_allgather() {
        let runtime = Arc::new(MultiGpuRuntime::simulated(4));
        let mut manager = CollectiveManager::new(runtime);

        // Each device sends 2 elements
        let send_buffers: Vec<_> = (0..4)
            .map(|i| SimulatedBuffer::filled(DeviceId(i), 2, i as f32))
            .collect();

        // Each device receives 8 elements (4 * 2)
        let mut recv_buffers: Vec<_> = (0..4)
            .map(|i| SimulatedBuffer::new(DeviceId(i), 8))
            .collect();

        manager.allgather(&send_buffers, &mut recv_buffers).unwrap();

        // All recv buffers should have [0,0, 1,1, 2,2, 3,3]
        let expected = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        for buf in &recv_buffers {
            assert_eq!(buf.data, expected);
        }
    }

    #[test]
    fn test_reduce_scatter() {
        let runtime = Arc::new(MultiGpuRuntime::simulated(4));
        let mut manager = CollectiveManager::new(runtime);

        // Each device has 4 elements
        let send_buffers = create_test_buffers(4, 4, |i| i as f32);

        // Each device receives 1 element (4 / 4)
        let mut recv_buffers: Vec<_> = (0..4)
            .map(|i| SimulatedBuffer::new(DeviceId(i), 1))
            .collect();

        manager
            .reduce_scatter(&send_buffers, &mut recv_buffers, CollectiveOp::Sum)
            .unwrap();

        // Each device should have the sum of its chunk
        // Device i gets chunk i, which is the sum of send_buffers[*][i]
        // All send buffers have the same value at each position (device index)
        // So chunk i sum = 0 + 1 + 2 + 3 = 6 for all chunks
        for buf in &recv_buffers {
            assert_eq!(buf.data[0], 6.0);
        }
    }

    #[test]
    fn test_collective_stats() {
        let runtime = Arc::new(MultiGpuRuntime::simulated(4));
        let mut manager = CollectiveManager::new(runtime);

        let mut buffers = create_test_buffers(4, 4, |i| i as f32);

        manager
            .allreduce(&mut buffers, CollectiveOp::Sum, CollectiveAlgorithm::Direct)
            .unwrap();

        let stats = manager.stats();
        assert_eq!(stats.operation_count, 1);
        assert!(stats.bytes_transferred > 0);
    }
}
