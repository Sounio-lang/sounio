//! CUDA Graphs with Dynamic Control Flow
//!
//! Implements CUDA Graph API abstractions for Sounio, enabling:
//! - Kernel launch graphs with dependencies
//! - Memory transfer nodes (H2D, D2H, D2D)
//! - Dynamic control flow (if/switch/while)
//! - Graph instantiation and execution
//!
//! This enables efficient multi-kernel workflows with reduced launch overhead.
//!
//! # Example
//!
//! ```ignore
//! let mut graph = GpuGraph::new("inference");
//! let input_node = graph.add_memcpy_h2d("input", host_buf, device_buf);
//! let layer1_node = graph.add_kernel("layer1", kernel1, &[input_node]);
//! let output_node = graph.add_memcpy_d2h("output", device_buf, host_buf, &[layer1_node]);
//! let executable = graph.instantiate();
//! executable.launch(stream);
//! ```

use super::ir::{GpuModule, GpuType, ValueId};
use rustc_hash::FxHashMap;

/// Unique identifier for a graph node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphNodeId(pub u32);

/// Unique identifier for a buffer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

/// Unique identifier for a stream
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(pub u32);

/// GPU execution graph
#[derive(Debug, Clone)]
pub struct GpuGraph {
    /// Graph name
    pub name: String,

    /// Graph nodes
    pub nodes: Vec<GraphNode>,

    /// Edges (dependencies): (from, to)
    pub edges: Vec<(GraphNodeId, GraphNodeId)>,

    /// Entry nodes (no dependencies)
    pub entry_nodes: Vec<GraphNodeId>,

    /// Exit nodes (no dependents)
    pub exit_nodes: Vec<GraphNodeId>,

    /// Node counter
    next_node_id: u32,

    /// Buffer registry
    pub buffers: FxHashMap<BufferId, BufferInfo>,

    /// Next buffer ID
    next_buffer_id: u32,
}

/// Information about a buffer
#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub name: String,
    pub elem_type: GpuType,
    pub size: usize,
    pub location: BufferLocation,
}

/// Buffer location
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferLocation {
    Host,
    Device,
    Managed, // Unified memory
}

/// A node in the GPU graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Node ID
    pub id: GraphNodeId,

    /// Node name (for debugging)
    pub name: String,

    /// Node type
    pub node_type: GraphNodeType,

    /// Dependencies (must complete before this node)
    pub dependencies: Vec<GraphNodeId>,
}

/// Types of graph nodes
#[derive(Debug, Clone)]
pub enum GraphNodeType {
    /// Kernel launch
    Kernel(KernelNode),

    /// Host-to-device memory copy
    MemcpyH2D(MemcpyNode),

    /// Device-to-host memory copy
    MemcpyD2H(MemcpyNode),

    /// Device-to-device memory copy
    MemcpyD2D(MemcpyNode),

    /// Memory set (fill with value)
    Memset(MemsetNode),

    /// Host function callback
    HostCallback(HostCallbackNode),

    /// Conditional branch (CUDA 12.0+)
    Conditional(ConditionalNode),

    /// Loop (CUDA 12.0+)
    Loop(LoopNode),

    /// Empty node (synchronization point)
    Empty,

    /// Event record
    EventRecord(EventNode),

    /// Event wait
    EventWait(EventNode),

    /// Child graph (nested graph)
    ChildGraph(Box<GpuGraph>),
}

/// Kernel launch node
#[derive(Debug, Clone)]
pub struct KernelNode {
    /// Kernel to launch
    pub kernel_name: String,

    /// Grid dimensions
    pub grid: (u32, u32, u32),

    /// Block dimensions
    pub block: (u32, u32, u32),

    /// Shared memory size (bytes)
    pub shared_mem: u32,

    /// Kernel arguments (buffer IDs or immediate values)
    pub args: Vec<GraphKernelArg>,
}

/// Graph kernel argument
#[derive(Debug, Clone)]
pub enum GraphKernelArg {
    Buffer(BufferId),
    ImmediateI32(i32),
    ImmediateI64(i64),
    ImmediateF32(f32),
    ImmediateF64(f64),
    ImmediatePtr(u64),
}

/// Memory copy node
#[derive(Debug, Clone)]
pub struct MemcpyNode {
    /// Source buffer
    pub src: BufferId,

    /// Destination buffer
    pub dst: BufferId,

    /// Size in bytes (None = entire buffer)
    pub size: Option<usize>,

    /// Source offset
    pub src_offset: usize,

    /// Destination offset
    pub dst_offset: usize,
}

/// Memory set node
#[derive(Debug, Clone)]
pub struct MemsetNode {
    /// Target buffer
    pub buffer: BufferId,

    /// Value to set (as u8 pattern)
    pub value: u8,

    /// Size in bytes
    pub size: usize,

    /// Offset
    pub offset: usize,
}

/// Host callback node
#[derive(Debug, Clone)]
pub struct HostCallbackNode {
    /// Callback function name
    pub callback_name: String,

    /// User data pointer
    pub user_data: u64,
}

/// Conditional node (if/else in graph)
#[derive(Debug, Clone)]
pub struct ConditionalNode {
    /// Condition type
    pub condition_type: ConditionType,

    /// Condition handle (device-side flag)
    pub condition_handle: ValueId,

    /// Graph to execute if condition is true
    pub then_graph: Box<GpuGraph>,

    /// Graph to execute if condition is false (optional)
    pub else_graph: Option<Box<GpuGraph>>,
}

/// Condition type for conditional nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionType {
    /// Condition is a boolean flag
    If,
    /// Condition is an integer for switch
    Switch,
    /// Condition controls loop iteration
    While,
}

/// Loop node
#[derive(Debug, Clone)]
pub struct LoopNode {
    /// Loop body graph
    pub body: Box<GpuGraph>,

    /// Loop count (None = condition-based)
    pub count: Option<u32>,

    /// Condition handle for while loops
    pub condition_handle: Option<ValueId>,
}

/// Event node
#[derive(Debug, Clone)]
pub struct EventNode {
    /// Event identifier
    pub event_id: u32,
}

impl GpuGraph {
    /// Create a new empty graph
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            entry_nodes: Vec::new(),
            exit_nodes: Vec::new(),
            next_node_id: 0,
            buffers: FxHashMap::default(),
            next_buffer_id: 0,
        }
    }

    /// Allocate a new node ID
    fn alloc_node_id(&mut self) -> GraphNodeId {
        let id = GraphNodeId(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    /// Allocate a new buffer ID
    pub fn alloc_buffer(
        &mut self,
        name: &str,
        elem_type: GpuType,
        size: usize,
        location: BufferLocation,
    ) -> BufferId {
        let id = BufferId(self.next_buffer_id);
        self.next_buffer_id += 1;
        self.buffers.insert(
            id,
            BufferInfo {
                name: name.into(),
                elem_type,
                size,
                location,
            },
        );
        id
    }

    /// Add a kernel launch node
    pub fn add_kernel(
        &mut self,
        name: &str,
        kernel_name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: Vec<GraphKernelArg>,
        deps: &[GraphNodeId],
    ) -> GraphNodeId {
        let id = self.alloc_node_id();
        let node = GraphNode {
            id,
            name: name.into(),
            node_type: GraphNodeType::Kernel(KernelNode {
                kernel_name: kernel_name.into(),
                grid,
                block,
                shared_mem: 0,
                args,
            }),
            dependencies: deps.to_vec(),
        };
        self.nodes.push(node);
        self.add_dependencies(id, deps);
        id
    }

    /// Add a host-to-device memory copy
    pub fn add_memcpy_h2d(
        &mut self,
        name: &str,
        src: BufferId,
        dst: BufferId,
        deps: &[GraphNodeId],
    ) -> GraphNodeId {
        let id = self.alloc_node_id();
        let node = GraphNode {
            id,
            name: name.into(),
            node_type: GraphNodeType::MemcpyH2D(MemcpyNode {
                src,
                dst,
                size: None,
                src_offset: 0,
                dst_offset: 0,
            }),
            dependencies: deps.to_vec(),
        };
        self.nodes.push(node);
        self.add_dependencies(id, deps);
        id
    }

    /// Add a device-to-host memory copy
    pub fn add_memcpy_d2h(
        &mut self,
        name: &str,
        src: BufferId,
        dst: BufferId,
        deps: &[GraphNodeId],
    ) -> GraphNodeId {
        let id = self.alloc_node_id();
        let node = GraphNode {
            id,
            name: name.into(),
            node_type: GraphNodeType::MemcpyD2H(MemcpyNode {
                src,
                dst,
                size: None,
                src_offset: 0,
                dst_offset: 0,
            }),
            dependencies: deps.to_vec(),
        };
        self.nodes.push(node);
        self.add_dependencies(id, deps);
        id
    }

    /// Add a conditional node (if/else)
    pub fn add_conditional(
        &mut self,
        name: &str,
        condition_handle: ValueId,
        then_graph: GpuGraph,
        else_graph: Option<GpuGraph>,
        deps: &[GraphNodeId],
    ) -> GraphNodeId {
        let id = self.alloc_node_id();
        let node = GraphNode {
            id,
            name: name.into(),
            node_type: GraphNodeType::Conditional(ConditionalNode {
                condition_type: ConditionType::If,
                condition_handle,
                then_graph: Box::new(then_graph),
                else_graph: else_graph.map(Box::new),
            }),
            dependencies: deps.to_vec(),
        };
        self.nodes.push(node);
        self.add_dependencies(id, deps);
        id
    }

    /// Add a loop node
    pub fn add_loop(
        &mut self,
        name: &str,
        body: GpuGraph,
        count: Option<u32>,
        deps: &[GraphNodeId],
    ) -> GraphNodeId {
        let id = self.alloc_node_id();
        let node = GraphNode {
            id,
            name: name.into(),
            node_type: GraphNodeType::Loop(LoopNode {
                body: Box::new(body),
                count,
                condition_handle: None,
            }),
            dependencies: deps.to_vec(),
        };
        self.nodes.push(node);
        self.add_dependencies(id, deps);
        id
    }

    /// Add a child graph node
    pub fn add_child_graph(
        &mut self,
        name: &str,
        child: GpuGraph,
        deps: &[GraphNodeId],
    ) -> GraphNodeId {
        let id = self.alloc_node_id();
        let node = GraphNode {
            id,
            name: name.into(),
            node_type: GraphNodeType::ChildGraph(Box::new(child)),
            dependencies: deps.to_vec(),
        };
        self.nodes.push(node);
        self.add_dependencies(id, deps);
        id
    }

    /// Add an empty synchronization node
    pub fn add_sync_point(&mut self, name: &str, deps: &[GraphNodeId]) -> GraphNodeId {
        let id = self.alloc_node_id();
        let node = GraphNode {
            id,
            name: name.into(),
            node_type: GraphNodeType::Empty,
            dependencies: deps.to_vec(),
        };
        self.nodes.push(node);
        self.add_dependencies(id, deps);
        id
    }

    /// Add dependency edges
    fn add_dependencies(&mut self, to: GraphNodeId, from: &[GraphNodeId]) {
        for &dep in from {
            self.edges.push((dep, to));
        }
    }

    /// Compute entry and exit nodes
    pub fn finalize(&mut self) {
        let mut has_incoming: FxHashMap<GraphNodeId, bool> = FxHashMap::default();
        let mut has_outgoing: FxHashMap<GraphNodeId, bool> = FxHashMap::default();

        for node in &self.nodes {
            has_incoming.insert(node.id, false);
            has_outgoing.insert(node.id, false);
        }

        for (from, to) in &self.edges {
            has_incoming.insert(*to, true);
            has_outgoing.insert(*from, true);
        }

        self.entry_nodes = self
            .nodes
            .iter()
            .filter(|n| !has_incoming.get(&n.id).copied().unwrap_or(false))
            .map(|n| n.id)
            .collect();

        self.exit_nodes = self
            .nodes
            .iter()
            .filter(|n| !has_outgoing.get(&n.id).copied().unwrap_or(false))
            .map(|n| n.id)
            .collect();
    }

    /// Get topologically sorted node order
    pub fn topological_sort(&self) -> Vec<GraphNodeId> {
        let mut in_degree: FxHashMap<GraphNodeId, usize> = FxHashMap::default();
        let mut adj: FxHashMap<GraphNodeId, Vec<GraphNodeId>> = FxHashMap::default();

        for node in &self.nodes {
            in_degree.insert(node.id, 0);
            adj.insert(node.id, Vec::new());
        }

        for (from, to) in &self.edges {
            *in_degree.get_mut(to).unwrap() += 1;
            adj.get_mut(from).unwrap().push(*to);
        }

        let mut queue: Vec<GraphNodeId> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut result = Vec::new();

        while let Some(node) = queue.pop() {
            result.push(node);
            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    let deg = in_degree.get_mut(&neighbor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(neighbor);
                    }
                }
            }
        }

        result
    }

    /// Get node by ID
    pub fn get_node(&self, id: GraphNodeId) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Check if graph has cycles (should be DAG)
    pub fn is_acyclic(&self) -> bool {
        self.topological_sort().len() == self.nodes.len()
    }
}

// ============================================================================
// Module to Graph Conversion (Phase 6 Integration)
// ============================================================================

/// Build a dependency graph from a GPU module
///
/// Analyzes kernels in the module and creates a graph representing
/// potential execution dependencies. This is used for kernel fusion
/// analysis.
///
/// Note: Without runtime information, this creates a simple sequential
/// graph. For more accurate dependency analysis, use runtime profiling
/// or explicit dependency annotations.
pub fn build_graph_from_module(module: &GpuModule) -> GpuGraph {
    let mut graph = GpuGraph::new(&module.name);

    // Create a node for each kernel
    let mut kernel_nodes: FxHashMap<String, GraphNodeId> = FxHashMap::default();

    for name in module.kernels.keys() {
        let node_id = graph.alloc_node_id();
        let node = GraphNode {
            id: node_id,
            name: name.clone(),
            node_type: GraphNodeType::Kernel(KernelNode {
                kernel_name: name.clone(),
                grid: (1, 1, 1),    // Placeholder
                block: (256, 1, 1), // Default block size
                shared_mem: 0,
                args: Vec::new(),
            }),
            dependencies: Vec::new(),
        };
        graph.nodes.push(node);
        kernel_nodes.insert(name.clone(), node_id);
    }

    // Without explicit dependency information, assume kernels can be
    // executed independently (no edges). The fusion analysis will
    // determine actual fusion opportunities based on kernel properties.
    //
    // Future: Add heuristic dependency analysis based on:
    // - Shared memory buffer names
    // - Parameter types and sizes
    // - Kernel naming conventions

    // Mark all nodes as entry nodes (no dependencies)
    graph.entry_nodes = kernel_nodes.values().copied().collect();

    // Mark all nodes as exit nodes (no dependents)
    graph.exit_nodes = kernel_nodes.values().copied().collect();

    graph
}

/// Graph execution configuration
#[derive(Debug, Clone)]
pub struct GraphExecConfig {
    /// Stream to execute on
    pub stream: StreamId,

    /// Enable graph capture
    pub capture_mode: bool,

    /// Upload graph updates
    pub update_mode: bool,
}

impl Default for GraphExecConfig {
    fn default() -> Self {
        Self {
            stream: StreamId(0),
            capture_mode: false,
            update_mode: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = GpuGraph::new("test_graph");
        assert_eq!(graph.name, "test_graph");
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_buffer_allocation() {
        let mut graph = GpuGraph::new("test");
        let buf1 = graph.alloc_buffer("input", GpuType::F32, 1024, BufferLocation::Host);
        let buf2 = graph.alloc_buffer("output", GpuType::F32, 1024, BufferLocation::Device);

        assert_eq!(buf1.0, 0);
        assert_eq!(buf2.0, 1);
        assert_eq!(graph.buffers.len(), 2);
    }

    #[test]
    fn test_kernel_node() {
        let mut graph = GpuGraph::new("test");
        let buf = graph.alloc_buffer("data", GpuType::F32, 1024, BufferLocation::Device);

        let node = graph.add_kernel(
            "compute",
            "my_kernel",
            (256, 1, 1),
            (256, 1, 1),
            vec![GraphKernelArg::Buffer(buf)],
            &[],
        );

        assert_eq!(node.0, 0);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_memcpy_nodes() {
        let mut graph = GpuGraph::new("test");
        let host_buf = graph.alloc_buffer("host", GpuType::F32, 1024, BufferLocation::Host);
        let dev_buf = graph.alloc_buffer("device", GpuType::F32, 1024, BufferLocation::Device);

        let h2d = graph.add_memcpy_h2d("upload", host_buf, dev_buf, &[]);
        let d2h = graph.add_memcpy_d2h("download", dev_buf, host_buf, &[h2d]);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0], (h2d, d2h));
    }

    #[test]
    fn test_dependency_chain() {
        let mut graph = GpuGraph::new("pipeline");
        let buf = graph.alloc_buffer("data", GpuType::F32, 1024, BufferLocation::Device);

        let n1 = graph.add_kernel(
            "step1",
            "kernel1",
            (1, 1, 1),
            (1, 1, 1),
            vec![GraphKernelArg::Buffer(buf)],
            &[],
        );
        let n2 = graph.add_kernel(
            "step2",
            "kernel2",
            (1, 1, 1),
            (1, 1, 1),
            vec![GraphKernelArg::Buffer(buf)],
            &[n1],
        );
        let n3 = graph.add_kernel(
            "step3",
            "kernel3",
            (1, 1, 1),
            (1, 1, 1),
            vec![GraphKernelArg::Buffer(buf)],
            &[n2],
        );

        assert_eq!(graph.edges.len(), 2);

        let sorted = graph.topological_sort();
        assert_eq!(sorted, vec![n1, n2, n3]);
    }

    #[test]
    fn test_parallel_nodes() {
        let mut graph = GpuGraph::new("parallel");
        let buf = graph.alloc_buffer("data", GpuType::F32, 1024, BufferLocation::Device);

        // Two parallel kernels, then sync
        let n1 = graph.add_kernel(
            "branch1",
            "kernel1",
            (1, 1, 1),
            (1, 1, 1),
            vec![GraphKernelArg::Buffer(buf)],
            &[],
        );
        let n2 = graph.add_kernel(
            "branch2",
            "kernel2",
            (1, 1, 1),
            (1, 1, 1),
            vec![GraphKernelArg::Buffer(buf)],
            &[],
        );
        let _n3 = graph.add_sync_point("join", &[n1, n2]);

        graph.finalize();

        assert_eq!(graph.entry_nodes.len(), 2);
        assert_eq!(graph.exit_nodes.len(), 1);
        assert!(graph.is_acyclic());
    }

    #[test]
    fn test_conditional_node() {
        let mut graph = GpuGraph::new("conditional");

        let then_graph = GpuGraph::new("then_branch");
        let else_graph = GpuGraph::new("else_branch");

        let _cond =
            graph.add_conditional("if_check", ValueId(0), then_graph, Some(else_graph), &[]);

        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_loop_node() {
        let mut graph = GpuGraph::new("loop_test");
        let body = GpuGraph::new("loop_body");

        let _loop_node = graph.add_loop("iteration", body, Some(10), &[]);

        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_child_graph() {
        let mut parent = GpuGraph::new("parent");
        let child = GpuGraph::new("child");

        let _child_node = parent.add_child_graph("nested", child, &[]);

        assert_eq!(parent.node_count(), 1);
    }

    #[test]
    fn test_inference_pipeline() {
        // Realistic inference pipeline
        let mut graph = GpuGraph::new("inference");

        // Buffers
        let host_input = graph.alloc_buffer("host_input", GpuType::F32, 4096, BufferLocation::Host);
        let dev_input = graph.alloc_buffer("dev_input", GpuType::F32, 4096, BufferLocation::Device);
        let dev_hidden =
            graph.alloc_buffer("dev_hidden", GpuType::F32, 2048, BufferLocation::Device);
        let dev_output =
            graph.alloc_buffer("dev_output", GpuType::F32, 1024, BufferLocation::Device);
        let host_output =
            graph.alloc_buffer("host_output", GpuType::F32, 1024, BufferLocation::Host);

        // Pipeline
        let upload = graph.add_memcpy_h2d("upload_input", host_input, dev_input, &[]);
        let layer1 = graph.add_kernel(
            "layer1",
            "dense_layer",
            (16, 1, 1),
            (256, 1, 1),
            vec![
                GraphKernelArg::Buffer(dev_input),
                GraphKernelArg::Buffer(dev_hidden),
            ],
            &[upload],
        );
        let layer2 = graph.add_kernel(
            "layer2",
            "dense_layer",
            (8, 1, 1),
            (256, 1, 1),
            vec![
                GraphKernelArg::Buffer(dev_hidden),
                GraphKernelArg::Buffer(dev_output),
            ],
            &[layer1],
        );
        let download = graph.add_memcpy_d2h("download_output", dev_output, host_output, &[layer2]);

        graph.finalize();

        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.entry_nodes, vec![GraphNodeId(0)]);
        assert_eq!(graph.exit_nodes, vec![download]);

        let order = graph.topological_sort();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], upload);
        assert_eq!(order[3], download);
    }
}
