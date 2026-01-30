//! Integration tests for Multi-GPU / Distributed Computing (Phase 10)
//!
//! Tests the complete multi-GPU infrastructure:
//! - Device topology discovery
//! - P2P transfer management
//! - Collective operations (allreduce, broadcast, etc.)
//! - Algorithm selection and performance characteristics

use std::sync::Arc;

use sounio::codegen::gpu::{
    // Collective operations
    AlgorithmSelector,
    CollectiveAlgorithm,
    CollectiveManager,
    CollectiveOp,
    // Multi-GPU core
    DeviceId,
    MultiGpuRuntime,
    // P2P transfers
    P2PManager,
    ReduceOp,
    SimulatedBuffer,
    TransferDescriptor,
    generate_allgather_steps,
    generate_reduce_scatter_steps,
    split_into_chunks,
};

// ============================================================================
// Topology Integration Tests
// ============================================================================

#[test]
fn test_simulated_8gpu_topology() {
    // Create a simulated 8-GPU DGX-style topology
    let runtime = MultiGpuRuntime::simulated(8);

    // Should have 8 devices
    assert_eq!(runtime.device_count(), 8);

    // All devices should be the same architecture in simulation
    for id in runtime.device_ids() {
        let info = runtime.device_info(id).expect("Device should exist");
        assert!(info.total_memory > 0);
    }

    // P2P should be available between all pairs
    for src in runtime.device_ids() {
        for dst in runtime.device_ids() {
            if src != dst {
                assert!(
                    runtime.p2p_capability(src, dst).is_some(),
                    "P2P capability should exist between {:?} and {:?}",
                    src,
                    dst
                );
            }
        }
    }
}

#[test]
fn test_topology_ring_and_tree() {
    let runtime = MultiGpuRuntime::simulated(4);

    // Build a ring for collective operations
    let ring = runtime.topology().build_ring();
    assert_eq!(ring.len(), 4);
    // Ring should visit each device exactly once
    let mut seen = std::collections::HashSet::new();
    for &id in &ring {
        assert!(seen.insert(id), "Ring should not repeat devices");
    }

    // Build a tree for broadcast
    let tree = runtime.topology().build_tree(DeviceId(0));
    // Root should be device 0 (field access)
    assert_eq!(tree.root, DeviceId(0));
}

#[test]
fn test_device_groups() {
    let runtime = MultiGpuRuntime::simulated(8);

    // Should have device groups
    let groups = runtime.groups();
    assert!(!groups.is_empty());

    // Each group should have devices
    for group in groups {
        assert!(!group.devices.is_empty());
    }
}

// ============================================================================
// P2P Transfer Integration Tests
// ============================================================================

#[test]
fn test_p2p_manager_lifecycle() {
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));
    let mut manager = P2PManager::new(runtime);

    // Initial stats should be zero
    assert_eq!(manager.stats().total_bytes, 0);
    assert_eq!(manager.stats().transfer_count, 0);

    // Perform some transfers
    let desc = TransferDescriptor::simple(DeviceId(0), DeviceId(1), 1024);
    manager.copy(&desc).unwrap();

    let desc2 = TransferDescriptor::d2d(DeviceId(1), 512, DeviceId(2), 0, 2048);
    manager.copy(&desc2).unwrap();

    // Stats should reflect transfers
    assert_eq!(manager.stats().total_bytes, 1024 + 2048);
    assert_eq!(manager.stats().transfer_count, 2);

    // Reset stats
    manager.reset_stats();
    assert_eq!(manager.stats().total_bytes, 0);
}

#[test]
fn test_p2p_batch_transfers() {
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));
    let mut manager = P2PManager::new(runtime);

    // Create batch of transfers
    let descriptors = vec![
        TransferDescriptor::simple(DeviceId(0), DeviceId(1), 1024),
        TransferDescriptor::simple(DeviceId(1), DeviceId(2), 1024),
        TransferDescriptor::simple(DeviceId(2), DeviceId(3), 1024),
        TransferDescriptor::simple(DeviceId(3), DeviceId(0), 1024),
    ];

    // Execute batch
    manager.copy_batch(&descriptors).unwrap();

    // Should have 4 transfers of 1KB each
    assert_eq!(manager.stats().transfer_count, 4);
    assert_eq!(manager.stats().total_bytes, 4096);
}

#[test]
fn test_p2p_copy_reduce() {
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));
    let mut manager = P2PManager::new(runtime);

    let desc = TransferDescriptor::simple(DeviceId(0), DeviceId(1), 1024);

    // Test different reduce operations
    manager.copy_reduce(&desc, ReduceOp::Sum).unwrap();
    manager.copy_reduce(&desc, ReduceOp::Max).unwrap();
    manager.copy_reduce(&desc, ReduceOp::Min).unwrap();

    assert_eq!(manager.stats().transfer_count, 3);
}

#[test]
fn test_ring_step_generation() {
    let ring = vec![DeviceId(0), DeviceId(1), DeviceId(2), DeviceId(3)];
    let chunk_size = 1024;

    // Generate reduce-scatter steps
    let rs_steps = generate_reduce_scatter_steps(&ring, chunk_size);
    assert_eq!(rs_steps.len(), 3); // n-1 steps for 4 GPUs

    // Each step should have 4 transfers (one per GPU)
    for step in &rs_steps {
        assert_eq!(step.len(), 4);
    }

    // Generate allgather steps
    let ag_steps = generate_allgather_steps(&ring, chunk_size);
    assert_eq!(ag_steps.len(), 3); // n-1 steps for 4 GPUs
}

#[test]
fn test_chunk_splitting() {
    // Test splitting a buffer into chunks
    let total_size = 10000;
    let chunk_size = 3000;

    let chunks = split_into_chunks(total_size, chunk_size);

    // Should have 4 chunks: 3000 + 3000 + 3000 + 1000
    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks[0].size, 3000);
    assert_eq!(chunks[1].size, 3000);
    assert_eq!(chunks[2].size, 3000);
    assert_eq!(chunks[3].size, 1000);

    // Offsets should be sequential
    assert_eq!(chunks[0].src_offset, 0);
    assert_eq!(chunks[1].src_offset, 3000);
    assert_eq!(chunks[2].src_offset, 6000);
    assert_eq!(chunks[3].src_offset, 9000);
}

// ============================================================================
// Collective Operations Integration Tests
// ============================================================================

#[test]
fn test_collective_manager_creation() {
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));
    let manager = CollectiveManager::new(runtime);

    // Initial stats should be zero
    let stats = manager.stats();
    assert_eq!(stats.operation_count, 0);
    assert_eq!(stats.bytes_transferred, 0);
}

#[test]
fn test_allreduce_sum_integration() {
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));
    let mut manager = CollectiveManager::new(runtime);

    // Create buffers with test data using filled() helper
    // Each device has [device_id + 1.0] replicated
    let mut buffers: Vec<SimulatedBuffer> = (0..4)
        .map(|i| SimulatedBuffer::filled(DeviceId(i as u32), 10, (i + 1) as f32))
        .collect();

    // AllReduce Sum should give us [1+2+3+4] = [10.0] on all devices
    manager
        .allreduce(&mut buffers, CollectiveOp::Sum, CollectiveAlgorithm::Ring)
        .unwrap();

    // All buffers should now have the same sum
    for buffer in &buffers {
        for &val in &buffer.data {
            assert!((val - 10.0).abs() < 0.001, "Expected 10.0, got {}", val);
        }
    }

    // Stats should reflect operations
    assert!(manager.stats().operation_count > 0);
}

#[test]
fn test_allreduce_max_integration() {
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));
    let mut manager = CollectiveManager::new(runtime);

    // Each device has different max values
    let mut buffers: Vec<SimulatedBuffer> = (0..4)
        .map(|i| SimulatedBuffer::filled(DeviceId(i as u32), 10, (i + 1) as f32 * 10.0))
        .collect();

    // AllReduce Max should give us [40.0] on all devices (max of 10, 20, 30, 40)
    manager
        .allreduce(&mut buffers, CollectiveOp::Max, CollectiveAlgorithm::Direct)
        .unwrap();

    for buffer in &buffers {
        for &val in &buffer.data {
            assert!((val - 40.0).abs() < 0.001, "Expected 40.0, got {}", val);
        }
    }
}

#[test]
fn test_broadcast_from_root() {
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));
    let mut manager = CollectiveManager::new(runtime);

    // Only root (device 0) has meaningful data
    let mut buffers: Vec<SimulatedBuffer> = (0..4)
        .map(|i| {
            let value = if i == 0 { 42.0 } else { 0.0 };
            SimulatedBuffer::filled(DeviceId(i as u32), 10, value)
        })
        .collect();

    // Broadcast from device 0
    manager
        .broadcast(&mut buffers, DeviceId(0), CollectiveAlgorithm::Tree)
        .unwrap();

    // All devices should now have the root's data
    for buffer in &buffers {
        for &val in &buffer.data {
            assert!((val - 42.0).abs() < 0.001, "Expected 42.0, got {}", val);
        }
    }
}

#[test]
fn test_reduce_to_root() {
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));
    let mut manager = CollectiveManager::new(runtime);

    // Each device has [device_id + 1.0]
    let mut buffers: Vec<SimulatedBuffer> = (0..4)
        .map(|i| SimulatedBuffer::filled(DeviceId(i as u32), 10, (i + 1) as f32))
        .collect();

    // Reduce to device 0 with Sum
    manager
        .reduce(&mut buffers, CollectiveOp::Sum, DeviceId(0))
        .unwrap();

    // Only root should have the reduced value
    for &val in &buffers[0].data {
        assert!(
            (val - 10.0).abs() < 0.001,
            "Root expected 10.0, got {}",
            val
        );
    }
}

#[test]
fn test_allgather_integration() {
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));
    let mut manager = CollectiveManager::new(runtime);

    // Each device has a unique chunk value
    let send_buffers: Vec<SimulatedBuffer> = (0..4)
        .map(|i| SimulatedBuffer::filled(DeviceId(i as u32), 2, (i + 1) as f32))
        .collect();

    // Receive buffers should be 4x larger to hold all chunks
    let mut recv_buffers: Vec<SimulatedBuffer> = (0..4)
        .map(|i| SimulatedBuffer::new(DeviceId(i as u32), 8))
        .collect();

    manager.allgather(&send_buffers, &mut recv_buffers).unwrap();

    // Each receive buffer should have all chunks: [1,1,2,2,3,3,4,4]
    for recv in &recv_buffers {
        assert_eq!(recv.data.len(), 8);
        // Check chunks
        assert!((recv.data[0] - 1.0).abs() < 0.001);
        assert!((recv.data[2] - 2.0).abs() < 0.001);
        assert!((recv.data[4] - 3.0).abs() < 0.001);
        assert!((recv.data[6] - 4.0).abs() < 0.001);
    }
}

#[test]
fn test_reduce_scatter_integration() {
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));
    let mut manager = CollectiveManager::new(runtime);

    // Each device has full data
    let send_buffers: Vec<SimulatedBuffer> = (0..4)
        .map(|i| SimulatedBuffer::filled(DeviceId(i as u32), 8, 1.0))
        .collect();

    // Receive buffers are 1/4 the size (each gets their chunk)
    let mut recv_buffers: Vec<SimulatedBuffer> = (0..4)
        .map(|i| SimulatedBuffer::new(DeviceId(i as u32), 2))
        .collect();

    manager
        .reduce_scatter(&send_buffers, &mut recv_buffers, CollectiveOp::Sum)
        .unwrap();

    // Each device should have reduced chunk: sum of 4 ones = 4.0
    for recv in &recv_buffers {
        for &val in &recv.data {
            assert!((val - 4.0).abs() < 0.001, "Expected 4.0, got {}", val);
        }
    }
}

// ============================================================================
// Algorithm Selection Tests
// ============================================================================

#[test]
fn test_algorithm_auto_selection() {
    let selector = AlgorithmSelector::default();

    // Small messages with few devices should use direct
    let algo = selector.select(1024, 4);
    assert_eq!(algo, CollectiveAlgorithm::Direct);

    // Large messages should use ring
    let algo = selector.select(100_000_000, 4);
    assert_eq!(algo, CollectiveAlgorithm::Ring);

    // Medium messages with many devices should use tree
    let algo = selector.select(100_000, 8);
    assert_eq!(algo, CollectiveAlgorithm::Tree);
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

#[test]
fn test_complete_allreduce_workflow() {
    // This test simulates a complete distributed training allreduce workflow

    // 1. Create topology
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));

    // 2. Create collective manager
    let mut manager = CollectiveManager::new(runtime.clone());

    // 3. Simulate gradient data on each device
    let mut gradients: Vec<SimulatedBuffer> = (0..4)
        .map(|i| {
            // Simulate gradients: predictable values
            let mut buf = SimulatedBuffer::new(DeviceId(i as u32), 1000);
            for j in 0..1000 {
                buf.data[j] = ((i * 1000 + j) % 100) as f32 / 100.0;
            }
            buf
        })
        .collect();

    // 4. AllReduce gradients (sum across all workers)
    manager
        .allreduce(&mut gradients, CollectiveOp::Sum, CollectiveAlgorithm::Auto)
        .unwrap();

    // 5. Verify all devices have the same result
    let reference = gradients[0].data.clone();
    for (i, buffer) in gradients.iter().enumerate().skip(1) {
        for (j, &val) in buffer.data.iter().enumerate() {
            assert!(
                (val - reference[j]).abs() < 0.001,
                "Device {} differs at index {}: {} vs {}",
                i,
                j,
                val,
                reference[j]
            );
        }
    }

    // 6. Check stats
    let stats = manager.stats();
    assert!(stats.operation_count > 0);
}

#[test]
fn test_hierarchical_broadcast_workflow() {
    // Simulate model parameter broadcast from root to all workers

    let runtime = Arc::new(MultiGpuRuntime::simulated(8));
    let mut manager = CollectiveManager::new(runtime);

    // Root device has model parameters
    let mut buffers: Vec<SimulatedBuffer> = (0..8)
        .map(|i| {
            let mut buf = if i == 0 {
                SimulatedBuffer::new(DeviceId(i as u32), 1000)
            } else {
                SimulatedBuffer::new(DeviceId(i as u32), 1000)
            };
            if i == 0 {
                for j in 0..1000 {
                    buf.data[j] = j as f32 * 0.001;
                }
            }
            buf
        })
        .collect();

    // Broadcast parameters
    manager
        .broadcast(&mut buffers, DeviceId(0), CollectiveAlgorithm::Tree)
        .unwrap();

    // Verify all devices have root's data
    let reference = buffers[0].data.clone();
    for (i, buffer) in buffers.iter().enumerate().skip(1) {
        for (j, &val) in buffer.data.iter().enumerate() {
            assert!(
                (val - reference[j]).abs() < 0.001,
                "Device {} differs at index {}",
                i,
                j
            );
        }
    }
}

#[test]
fn test_p2p_transfer_time_estimation() {
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));
    let manager = P2PManager::new(runtime);

    // Same device should be instant
    let time = manager.estimate_transfer_time_us(DeviceId(0), DeviceId(0), 1024 * 1024);
    assert_eq!(time, 0.0);

    // Cross-device transfer should take some time
    let time = manager.estimate_transfer_time_us(DeviceId(0), DeviceId(1), 1024 * 1024 * 1024);
    assert!(time > 0.0, "Transfer time should be positive");

    // Larger transfers should take longer
    let time_small = manager.estimate_transfer_time_us(DeviceId(0), DeviceId(1), 1024 * 1024);
    let time_large =
        manager.estimate_transfer_time_us(DeviceId(0), DeviceId(1), 1024 * 1024 * 1024);
    assert!(
        time_large > time_small,
        "Larger transfers should take longer"
    );
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_p2p_disabled_error() {
    let runtime = Arc::new(MultiGpuRuntime::simulated(4));
    let mut manager = P2PManager::new(runtime);

    // Disable P2P between devices 0 and 1
    manager.disable_peer_access(DeviceId(0), DeviceId(1));

    // Transfer should fail
    let desc = TransferDescriptor::simple(DeviceId(0), DeviceId(1), 1024);
    let result = manager.copy(&desc);

    assert!(result.is_err());
}
