//! Multi-GPU Runtime and Topology Management
//!
//! Provides infrastructure for multi-GPU computing:
//! - Device enumeration and management
//! - Topology discovery (NVLink, PCIe, NVSwitch)
//! - Device groups for hierarchical operations
//! - Simulated backend for testing
//!
//! # Architecture
//!
//! ```text
//! MultiGpuRuntime
//!       │
//!       ├── GpuTopology (device graph + interconnects)
//!       │
//!       ├── P2P Capability Matrix
//!       │
//!       └── Device Groups (for hierarchical collectives)
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::runtime::GpuBackend;

// ============================================================================
// Device Identification
// ============================================================================

/// Unique identifier for a GPU device within the multi-GPU context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId(pub u32);

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU{}", self.0)
    }
}

impl From<u32> for DeviceId {
    fn from(id: u32) -> Self {
        DeviceId(id)
    }
}

impl From<DeviceId> for u32 {
    fn from(id: DeviceId) -> Self {
        id.0
    }
}

// ============================================================================
// Interconnect Types
// ============================================================================

/// Type of interconnect between two GPUs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterconnectType {
    /// Same physical device (intra-GPU)
    SameDevice,
    /// NVIDIA NVLink direct connection
    NVLink {
        /// NVLink version (1, 2, 3, 4)
        version: u8,
        /// Number of links
        links: u8,
        /// Bandwidth in GB/s per direction
        bandwidth_gbps: f64,
    },
    /// NVSwitch fabric (fully connected)
    NVSwitch {
        /// NVSwitch generation
        generation: u8,
        /// Bandwidth in GB/s
        bandwidth_gbps: f64,
    },
    /// PCIe connection
    PCIe {
        /// PCIe generation (3, 4, 5)
        pcie_gen: u8,
        /// Number of lanes (x8, x16)
        lanes: u8,
        /// Whether through CPU (vs direct P2P)
        through_cpu: bool,
    },
    /// Network connection (multi-node)
    Network {
        /// Bandwidth in Gbps
        bandwidth_gbps: f64,
        /// Latency in microseconds
        latency_us: f64,
    },
    /// No direct connection available
    None,
}

impl InterconnectType {
    /// Get the bandwidth in GB/s
    pub fn bandwidth_gbps(&self) -> f64 {
        match self {
            InterconnectType::SameDevice => f64::INFINITY,
            InterconnectType::NVLink { bandwidth_gbps, .. } => *bandwidth_gbps,
            InterconnectType::NVSwitch { bandwidth_gbps, .. } => *bandwidth_gbps,
            InterconnectType::PCIe {
                pcie_gen, lanes, ..
            } => {
                // PCIe bandwidth calculation (GB/s, bidirectional)
                let lane_speed = match pcie_gen {
                    3 => 0.985, // ~1 GB/s per lane
                    4 => 1.969, // ~2 GB/s per lane
                    5 => 3.938, // ~4 GB/s per lane
                    _ => 0.5,
                };
                lane_speed * (*lanes as f64)
            }
            InterconnectType::Network { bandwidth_gbps, .. } => *bandwidth_gbps / 8.0,
            InterconnectType::None => 0.0,
        }
    }

    /// Get approximate latency in microseconds
    pub fn latency_us(&self) -> f64 {
        match self {
            InterconnectType::SameDevice => 0.0,
            InterconnectType::NVLink { .. } => 1.0,   // ~1us
            InterconnectType::NVSwitch { .. } => 2.0, // ~2us
            InterconnectType::PCIe { through_cpu, .. } => {
                if *through_cpu {
                    5.0
                } else {
                    2.0
                }
            }
            InterconnectType::Network { latency_us, .. } => *latency_us,
            InterconnectType::None => f64::INFINITY,
        }
    }

    /// Check if this is a high-bandwidth interconnect (NVLink/NVSwitch)
    pub fn is_high_bandwidth(&self) -> bool {
        matches!(
            self,
            InterconnectType::NVLink { .. } | InterconnectType::NVSwitch { .. }
        )
    }
}

impl fmt::Display for InterconnectType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterconnectType::SameDevice => write!(f, "SameDevice"),
            InterconnectType::NVLink {
                version,
                links,
                bandwidth_gbps,
            } => {
                write!(
                    f,
                    "NVLink{} x{} ({:.0} GB/s)",
                    version, links, bandwidth_gbps
                )
            }
            InterconnectType::NVSwitch {
                generation,
                bandwidth_gbps,
            } => {
                write!(f, "NVSwitch Gen{} ({:.0} GB/s)", generation, bandwidth_gbps)
            }
            InterconnectType::PCIe {
                pcie_gen,
                lanes,
                through_cpu,
            } => {
                let via = if *through_cpu { " via CPU" } else { "" };
                write!(f, "PCIe Gen{} x{}{}", pcie_gen, lanes, via)
            }
            InterconnectType::Network { bandwidth_gbps, .. } => {
                write!(f, "Network ({:.0} Gbps)", bandwidth_gbps)
            }
            InterconnectType::None => write!(f, "None"),
        }
    }
}

// ============================================================================
// Device Information
// ============================================================================

/// GPU architecture family
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuArchitecture {
    /// NVIDIA Turing (RTX 20xx, sm_75)
    Turing,
    /// NVIDIA Ampere (RTX 30xx, A100, sm_80-86)
    Ampere,
    /// NVIDIA Ada Lovelace (RTX 40xx, sm_89)
    Ada,
    /// NVIDIA Hopper (H100, sm_90)
    Hopper,
    /// NVIDIA Blackwell (B100, sm_100+)
    Blackwell,
    /// Apple Silicon (M1/M2/M3)
    AppleSilicon,
    /// AMD RDNA
    AmdRdna,
    /// Intel Arc
    IntelArc,
    /// Simulated/Unknown
    Simulated,
}

impl GpuArchitecture {
    /// Get compute capability for NVIDIA GPUs
    pub fn compute_capability(&self) -> Option<(u8, u8)> {
        match self {
            GpuArchitecture::Turing => Some((7, 5)),
            GpuArchitecture::Ampere => Some((8, 0)),
            GpuArchitecture::Ada => Some((8, 9)),
            GpuArchitecture::Hopper => Some((9, 0)),
            GpuArchitecture::Blackwell => Some((10, 0)),
            _ => None,
        }
    }
}

/// Information about a single GPU device
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device ID
    pub id: DeviceId,
    /// Device name
    pub name: String,
    /// Architecture
    pub architecture: GpuArchitecture,
    /// Total memory in bytes
    pub total_memory: u64,
    /// Number of SMs/CUs
    pub compute_units: u32,
    /// Max threads per block
    pub max_threads_per_block: u32,
    /// Max shared memory per block
    pub max_shared_memory: u32,
    /// Supports unified memory
    pub unified_memory: bool,
    /// Supports peer access
    pub peer_access_supported: bool,
}

impl DeviceInfo {
    /// Create a simulated device for testing
    pub fn simulated(id: u32) -> Self {
        Self {
            id: DeviceId(id),
            name: format!("Simulated GPU {}", id),
            architecture: GpuArchitecture::Simulated,
            total_memory: 16 * 1024 * 1024 * 1024, // 16 GB
            compute_units: 108,
            max_threads_per_block: 1024,
            max_shared_memory: 48 * 1024,
            unified_memory: true,
            peer_access_supported: true,
        }
    }
}

// ============================================================================
// P2P Capability
// ============================================================================

/// Peer-to-peer capability between two devices
#[derive(Debug, Clone, Copy)]
pub struct P2PCapability {
    /// Source device
    pub src: DeviceId,
    /// Destination device
    pub dst: DeviceId,
    /// Can enable peer access
    pub peer_access: bool,
    /// Supports atomic operations across devices
    pub peer_atomics: bool,
    /// Supports native memory ordering
    pub native_ordering: bool,
    /// Interconnect type
    pub interconnect: InterconnectType,
}

impl P2PCapability {
    /// Create capability for same device
    pub fn same_device(id: DeviceId) -> Self {
        Self {
            src: id,
            dst: id,
            peer_access: true,
            peer_atomics: true,
            native_ordering: true,
            interconnect: InterconnectType::SameDevice,
        }
    }

    /// Create simulated P2P capability
    pub fn simulated(src: DeviceId, dst: DeviceId) -> Self {
        if src == dst {
            Self::same_device(src)
        } else {
            Self {
                src,
                dst,
                peer_access: true,
                peer_atomics: true,
                native_ordering: true,
                interconnect: InterconnectType::NVLink {
                    version: 4,
                    links: 12,
                    bandwidth_gbps: 450.0,
                },
            }
        }
    }

    /// Get bandwidth in GB/s
    pub fn bandwidth_gbps(&self) -> f64 {
        self.interconnect.bandwidth_gbps()
    }
}

// ============================================================================
// Device Groups
// ============================================================================

/// A group of devices for hierarchical operations
#[derive(Debug, Clone)]
pub struct DeviceGroup {
    /// Group name
    pub name: String,
    /// Devices in this group
    pub devices: Vec<DeviceId>,
    /// Internal interconnect type (within group)
    pub internal_interconnect: InterconnectType,
    /// Leader device for inter-group communication
    pub leader: DeviceId,
}

impl DeviceGroup {
    /// Create a new device group
    pub fn new(
        name: impl Into<String>,
        devices: Vec<DeviceId>,
        interconnect: InterconnectType,
    ) -> Self {
        let leader = devices.first().copied().unwrap_or(DeviceId(0));
        Self {
            name: name.into(),
            devices,
            internal_interconnect: interconnect,
            leader,
        }
    }

    /// Check if a device is in this group
    pub fn contains(&self, device: DeviceId) -> bool {
        self.devices.contains(&device)
    }

    /// Get the number of devices
    pub fn len(&self) -> usize {
        self.devices.len()
    }

    /// Check if the group is empty
    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }
}

// ============================================================================
// GPU Topology
// ============================================================================

/// GPU topology information
#[derive(Debug, Clone)]
pub struct GpuTopology {
    /// Device information by ID
    devices: HashMap<DeviceId, DeviceInfo>,
    /// Interconnect between device pairs
    interconnects: HashMap<(DeviceId, DeviceId), InterconnectType>,
    /// Device groups for hierarchical operations
    groups: Vec<DeviceGroup>,
}

impl GpuTopology {
    /// Create an empty topology
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
            interconnects: HashMap::new(),
            groups: Vec::new(),
        }
    }

    /// Create a simulated topology with n devices
    pub fn simulated(n: u32) -> Self {
        let mut topology = Self::new();

        // Add devices
        for i in 0..n {
            let info = DeviceInfo::simulated(i);
            topology.devices.insert(DeviceId(i), info);
        }

        // Add interconnects (fully connected NVLink)
        for i in 0..n {
            for j in 0..n {
                let interconnect = if i == j {
                    InterconnectType::SameDevice
                } else {
                    InterconnectType::NVLink {
                        version: 4,
                        links: 12,
                        bandwidth_gbps: 450.0,
                    }
                };
                topology
                    .interconnects
                    .insert((DeviceId(i), DeviceId(j)), interconnect);
            }
        }

        // Create a single group with all devices
        let all_devices: Vec<_> = (0..n).map(DeviceId).collect();
        topology.groups.push(DeviceGroup::new(
            "all",
            all_devices,
            InterconnectType::NVLink {
                version: 4,
                links: 12,
                bandwidth_gbps: 450.0,
            },
        ));

        topology
    }

    /// Add a device to the topology
    pub fn add_device(&mut self, info: DeviceInfo) {
        let id = info.id;
        self.devices.insert(id, info);
        // Add self-connection
        self.interconnects
            .insert((id, id), InterconnectType::SameDevice);
    }

    /// Set interconnect between two devices
    pub fn set_interconnect(
        &mut self,
        src: DeviceId,
        dst: DeviceId,
        interconnect: InterconnectType,
    ) {
        self.interconnects.insert((src, dst), interconnect);
    }

    /// Add a device group
    pub fn add_group(&mut self, group: DeviceGroup) {
        self.groups.push(group);
    }

    /// Get device info
    pub fn device(&self, id: DeviceId) -> Option<&DeviceInfo> {
        self.devices.get(&id)
    }

    /// Get all device IDs
    pub fn device_ids(&self) -> impl Iterator<Item = DeviceId> + '_ {
        self.devices.keys().copied()
    }

    /// Get number of devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get interconnect between two devices
    pub fn interconnect(&self, src: DeviceId, dst: DeviceId) -> InterconnectType {
        self.interconnects
            .get(&(src, dst))
            .copied()
            .unwrap_or(InterconnectType::None)
    }

    /// Get all device groups
    pub fn groups(&self) -> &[DeviceGroup] {
        &self.groups
    }

    /// Find the group containing a device
    pub fn find_group(&self, device: DeviceId) -> Option<&DeviceGroup> {
        self.groups.iter().find(|g| g.contains(device))
    }

    /// Build a ring order for collective operations
    pub fn build_ring(&self) -> Vec<DeviceId> {
        // Simple ring: just use device IDs in order
        let mut devices: Vec<_> = self.device_ids().collect();
        devices.sort();
        devices
    }

    /// Build an optimal binary tree for broadcast/reduce
    pub fn build_tree(&self, root: DeviceId) -> BroadcastTree {
        let mut devices: Vec<_> = self.device_ids().collect();
        devices.sort();

        // Remove root and place it first
        devices.retain(|&d| d != root);

        BroadcastTree {
            root,
            children: devices,
        }
    }
}

impl Default for GpuTopology {
    fn default() -> Self {
        Self::new()
    }
}

/// Binary tree structure for broadcast/reduce
#[derive(Debug, Clone)]
pub struct BroadcastTree {
    /// Root device
    pub root: DeviceId,
    /// Child devices (in tree order)
    pub children: Vec<DeviceId>,
}

impl BroadcastTree {
    /// Get the parent of a device in the tree
    pub fn parent(&self, device: DeviceId) -> Option<DeviceId> {
        if device == self.root {
            return None;
        }

        // Find position in children
        if let Some(pos) = self.children.iter().position(|&d| d == device) {
            if pos == 0 {
                Some(self.root)
            } else {
                // Parent is at (pos-1)/2
                let parent_pos = (pos - 1) / 2;
                if parent_pos == 0 {
                    Some(self.root)
                } else {
                    self.children.get(parent_pos - 1).copied()
                }
            }
        } else {
            None
        }
    }

    /// Get the depth of the tree
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            0
        } else {
            // Binary tree depth: log2(n+1)
            ((self.children.len() + 1) as f64).log2().ceil() as usize
        }
    }
}

// ============================================================================
// Multi-GPU Runtime
// ============================================================================

/// Error type for multi-GPU operations
#[derive(Debug, Clone)]
pub enum MultiGpuError {
    /// Device not found
    DeviceNotFound(DeviceId),
    /// P2P access not available
    P2PNotAvailable { src: DeviceId, dst: DeviceId },
    /// Buffer size mismatch
    BufferSizeMismatch { expected: usize, actual: usize },
    /// Device count mismatch
    DeviceCountMismatch { expected: usize, actual: usize },
    /// Invalid device group
    InvalidGroup(String),
    /// Backend error
    BackendError(String),
    /// Operation not supported
    NotSupported(String),
}

impl fmt::Display for MultiGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiGpuError::DeviceNotFound(id) => write!(f, "Device {} not found", id),
            MultiGpuError::P2PNotAvailable { src, dst } => {
                write!(f, "P2P access not available: {} -> {}", src, dst)
            }
            MultiGpuError::BufferSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Buffer size mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            MultiGpuError::DeviceCountMismatch { expected, actual } => {
                write!(
                    f,
                    "Device count mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            MultiGpuError::InvalidGroup(name) => write!(f, "Invalid device group: {}", name),
            MultiGpuError::BackendError(msg) => write!(f, "Backend error: {}", msg),
            MultiGpuError::NotSupported(msg) => write!(f, "Not supported: {}", msg),
        }
    }
}

impl std::error::Error for MultiGpuError {}

/// Configuration for multi-GPU runtime
#[derive(Debug, Clone)]
pub struct MultiGpuConfig {
    /// Backend type
    pub backend: GpuBackend,
    /// Number of devices (for simulated)
    pub device_count: u32,
    /// Enable P2P by default
    pub enable_p2p: bool,
    /// Enable unified memory
    pub enable_unified_memory: bool,
}

impl Default for MultiGpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::Simulated,
            device_count: 4,
            enable_p2p: true,
            enable_unified_memory: true,
        }
    }
}

/// Multi-GPU runtime context
pub struct MultiGpuRuntime {
    /// Configuration
    config: MultiGpuConfig,
    /// Topology information
    topology: GpuTopology,
    /// P2P capability matrix
    p2p_capabilities: HashMap<(DeviceId, DeviceId), P2PCapability>,
    /// Enabled P2P pairs
    p2p_enabled: HashSet<(DeviceId, DeviceId)>,
}

impl MultiGpuRuntime {
    /// Create a new multi-GPU runtime with default configuration
    pub fn new() -> Self {
        Self::with_config(MultiGpuConfig::default())
    }

    /// Create a new multi-GPU runtime with specific configuration
    pub fn with_config(config: MultiGpuConfig) -> Self {
        let topology = match config.backend {
            GpuBackend::Simulated => GpuTopology::simulated(config.device_count),
            _ => GpuTopology::simulated(config.device_count), // TODO: real discovery
        };

        // Build P2P capability matrix
        let mut p2p_capabilities = HashMap::new();
        for src in topology.device_ids() {
            for dst in topology.device_ids() {
                let cap = P2PCapability::simulated(src, dst);
                p2p_capabilities.insert((src, dst), cap);
            }
        }

        let mut runtime = Self {
            config,
            topology,
            p2p_capabilities,
            p2p_enabled: HashSet::new(),
        };

        // Enable P2P if configured
        if runtime.config.enable_p2p {
            runtime.enable_all_p2p();
        }

        runtime
    }

    /// Create a runtime with simulated devices
    pub fn simulated(device_count: u32) -> Self {
        Self::with_config(MultiGpuConfig {
            backend: GpuBackend::Simulated,
            device_count,
            ..MultiGpuConfig::default()
        })
    }

    /// Get the number of devices
    pub fn device_count(&self) -> usize {
        self.topology.device_count()
    }

    /// Get device IDs
    pub fn device_ids(&self) -> impl Iterator<Item = DeviceId> + '_ {
        self.topology.device_ids()
    }

    /// Get device info
    pub fn device_info(&self, id: DeviceId) -> Result<&DeviceInfo, MultiGpuError> {
        self.topology
            .device(id)
            .ok_or(MultiGpuError::DeviceNotFound(id))
    }

    /// Get the topology
    pub fn topology(&self) -> &GpuTopology {
        &self.topology
    }

    /// Get P2P capability between two devices
    pub fn p2p_capability(&self, src: DeviceId, dst: DeviceId) -> Option<&P2PCapability> {
        self.p2p_capabilities.get(&(src, dst))
    }

    /// Check if P2P is enabled between two devices
    pub fn is_p2p_enabled(&self, src: DeviceId, dst: DeviceId) -> bool {
        src == dst || self.p2p_enabled.contains(&(src, dst))
    }

    /// Enable P2P access between two devices
    pub fn enable_p2p(&mut self, src: DeviceId, dst: DeviceId) -> Result<(), MultiGpuError> {
        if src == dst {
            return Ok(()); // Same device, always enabled
        }

        let cap = self
            .p2p_capabilities
            .get(&(src, dst))
            .ok_or(MultiGpuError::DeviceNotFound(src))?;

        if !cap.peer_access {
            return Err(MultiGpuError::P2PNotAvailable { src, dst });
        }

        self.p2p_enabled.insert((src, dst));
        Ok(())
    }

    /// Enable P2P between all device pairs
    pub fn enable_all_p2p(&mut self) {
        // Collect device IDs first to avoid borrow conflict
        let device_ids: Vec<_> = self.topology.device_ids().collect();
        for &src in &device_ids {
            for &dst in &device_ids {
                if src != dst {
                    let _ = self.enable_p2p(src, dst);
                }
            }
        }
    }

    /// Disable P2P access between two devices
    pub fn disable_p2p(&mut self, src: DeviceId, dst: DeviceId) {
        self.p2p_enabled.remove(&(src, dst));
    }

    /// Get the backend type
    pub fn backend(&self) -> GpuBackend {
        self.config.backend
    }

    /// Get device groups
    pub fn groups(&self) -> &[DeviceGroup] {
        self.topology.groups()
    }

    /// Build a ring for collective operations
    pub fn build_ring(&self) -> Vec<DeviceId> {
        self.topology.build_ring()
    }

    /// Build a tree for broadcast/reduce
    pub fn build_tree(&self, root: DeviceId) -> BroadcastTree {
        self.topology.build_tree(root)
    }
}

impl Default for MultiGpuRuntime {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Multi-GPU Synchronization
// ============================================================================

/// Multi-GPU barrier for synchronization
#[derive(Debug)]
pub struct MultiGpuBarrier {
    /// Devices participating in the barrier
    devices: Vec<DeviceId>,
    /// Barrier count (for reuse)
    count: u64,
}

impl MultiGpuBarrier {
    /// Create a new barrier for the given devices
    pub fn new(devices: Vec<DeviceId>) -> Self {
        Self { devices, count: 0 }
    }

    /// Create a barrier for all devices in a runtime
    pub fn all(runtime: &MultiGpuRuntime) -> Self {
        Self::new(runtime.device_ids().collect())
    }

    /// Get the devices in this barrier
    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// Synchronize all devices (simulated)
    pub fn wait(&mut self) -> Result<(), MultiGpuError> {
        self.count += 1;
        // In real implementation, this would use CUDA events
        Ok(())
    }
}

/// Multi-GPU event for fine-grained synchronization
#[derive(Debug)]
pub struct MultiGpuEvent {
    /// Device this event is on
    device: DeviceId,
    /// Event ID
    id: u64,
    /// Whether the event has been recorded
    recorded: bool,
}

impl MultiGpuEvent {
    /// Create a new event on a device
    pub fn new(device: DeviceId, id: u64) -> Self {
        Self {
            device,
            id,
            recorded: false,
        }
    }

    /// Record the event (simulated)
    pub fn record(&mut self) -> Result<(), MultiGpuError> {
        self.recorded = true;
        Ok(())
    }

    /// Wait for the event (simulated)
    pub fn wait(&self) -> Result<(), MultiGpuError> {
        if !self.recorded {
            return Err(MultiGpuError::NotSupported(
                "Event not recorded".to_string(),
            ));
        }
        Ok(())
    }

    /// Get the device
    pub fn device(&self) -> DeviceId {
        self.device
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_id() {
        let id = DeviceId(3);
        assert_eq!(format!("{}", id), "GPU3");
        assert_eq!(u32::from(id), 3);
    }

    #[test]
    fn test_interconnect_bandwidth() {
        let nvlink = InterconnectType::NVLink {
            version: 4,
            links: 12,
            bandwidth_gbps: 450.0,
        };
        assert_eq!(nvlink.bandwidth_gbps(), 450.0);
        assert!(nvlink.is_high_bandwidth());

        let pcie = InterconnectType::PCIe {
            pcie_gen: 4,
            lanes: 16,
            through_cpu: false,
        };
        assert!((pcie.bandwidth_gbps() - 31.5).abs() < 1.0); // ~32 GB/s
        assert!(!pcie.is_high_bandwidth());
    }

    #[test]
    fn test_simulated_topology() {
        let topology = GpuTopology::simulated(4);
        assert_eq!(topology.device_count(), 4);

        // Check interconnects
        let conn = topology.interconnect(DeviceId(0), DeviceId(1));
        assert!(matches!(conn, InterconnectType::NVLink { .. }));

        let self_conn = topology.interconnect(DeviceId(0), DeviceId(0));
        assert!(matches!(self_conn, InterconnectType::SameDevice));
    }

    #[test]
    fn test_multi_gpu_runtime() {
        let runtime = MultiGpuRuntime::simulated(4);
        assert_eq!(runtime.device_count(), 4);

        // Check P2P
        assert!(runtime.is_p2p_enabled(DeviceId(0), DeviceId(1)));
        assert!(runtime.is_p2p_enabled(DeviceId(0), DeviceId(0))); // Same device
    }

    #[test]
    fn test_build_ring() {
        let runtime = MultiGpuRuntime::simulated(4);
        let ring = runtime.build_ring();
        assert_eq!(ring.len(), 4);
        assert_eq!(
            ring,
            vec![DeviceId(0), DeviceId(1), DeviceId(2), DeviceId(3)]
        );
    }

    #[test]
    fn test_build_tree() {
        let runtime = MultiGpuRuntime::simulated(4);
        let tree = runtime.build_tree(DeviceId(0));
        assert_eq!(tree.root, DeviceId(0));
        assert_eq!(tree.children.len(), 3);
    }

    #[test]
    fn test_device_group() {
        let group = DeviceGroup::new(
            "node0",
            vec![DeviceId(0), DeviceId(1)],
            InterconnectType::NVLink {
                version: 4,
                links: 12,
                bandwidth_gbps: 450.0,
            },
        );
        assert!(group.contains(DeviceId(0)));
        assert!(!group.contains(DeviceId(2)));
        assert_eq!(group.len(), 2);
    }

    #[test]
    fn test_barrier() {
        let runtime = MultiGpuRuntime::simulated(4);
        let mut barrier = MultiGpuBarrier::all(&runtime);
        assert_eq!(barrier.devices().len(), 4);
        barrier.wait().unwrap();
    }

    #[test]
    fn test_event() {
        let mut event = MultiGpuEvent::new(DeviceId(0), 1);
        assert!(!event.recorded);
        event.record().unwrap();
        assert!(event.recorded);
        event.wait().unwrap();
    }
}
