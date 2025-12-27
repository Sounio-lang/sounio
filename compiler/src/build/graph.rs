//! Build dependency graph and incremental compilation tracking.
//!
//! This module provides the core dependency graph for the build system,
//! tracking compilation units, their dependencies, and managing invalidation
//! when source files change.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Unique identifier for a compilation unit
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnitId(pub u64);

impl UnitId {
    /// Create a new unit ID from a path
    pub fn from_path(path: &Path) -> Self {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        path.hash(&mut hasher);
        UnitId(hasher.finish())
    }
}

/// Content hash for tracking file changes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Compute hash from file contents
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        ContentHash(hasher.finalize().into())
    }

    /// Compute hash from file
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        Ok(Self::from_bytes(&data))
    }

    /// Get hash as hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Parse from hex string
    pub fn from_hex(s: &str) -> Result<Self, hex::FromHexError> {
        let bytes = hex::decode(s)?;
        if bytes.len() != 32 {
            return Err(hex::FromHexError::InvalidStringLength);
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(ContentHash(arr))
    }
}

/// A single compilation unit (typically a source file)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationUnit {
    /// Unique identifier
    pub id: UnitId,

    /// Source file path
    pub path: PathBuf,

    /// Content hash for change detection
    pub content_hash: ContentHash,

    /// Last modification time
    pub modified_time: SystemTime,

    /// Direct dependencies (imports)
    pub dependencies: Vec<UnitId>,

    /// Reverse dependencies (who depends on this unit)
    pub dependents: Vec<UnitId>,

    /// Whether this unit is dirty and needs recompilation
    pub dirty: bool,

    /// Metadata about the compiled output
    pub metadata: UnitMetadata,
}

impl CompilationUnit {
    /// Create a new compilation unit
    pub fn new(path: PathBuf, content_hash: ContentHash) -> Self {
        let id = UnitId::from_path(&path);
        let modified_time = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        CompilationUnit {
            id,
            path,
            content_hash,
            modified_time,
            dependencies: Vec::new(),
            dependents: Vec::new(),
            dirty: true,
            metadata: UnitMetadata::default(),
        }
    }

    /// Check if this unit has been modified
    pub fn is_modified(&self) -> std::io::Result<bool> {
        let current_hash = ContentHash::from_file(&self.path)?;
        Ok(current_hash != self.content_hash)
    }

    /// Update content hash and modification time
    pub fn update_hash(&mut self) -> std::io::Result<()> {
        self.content_hash = ContentHash::from_file(&self.path)?;
        self.modified_time = std::fs::metadata(&self.path)?.modified()?;
        Ok(())
    }
}

/// Metadata about compiled unit
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnitMetadata {
    /// Compilation time in milliseconds
    pub compile_time_ms: u64,

    /// Number of lines of code
    pub lines_of_code: usize,

    /// Number of exported symbols
    pub exported_symbols: usize,

    /// Warnings generated during compilation
    pub warnings: Vec<String>,
}

/// Build dependency graph
pub struct BuildGraph {
    /// All compilation units indexed by ID
    units: HashMap<UnitId, CompilationUnit>,

    /// Path to unit ID mapping
    path_to_id: HashMap<PathBuf, UnitId>,

    /// Root units (entry points)
    roots: HashSet<UnitId>,

    /// Graph version for incremental updates
    version: u64,
}

impl BuildGraph {
    /// Create a new empty build graph
    pub fn new() -> Self {
        BuildGraph {
            units: HashMap::new(),
            path_to_id: HashMap::new(),
            roots: HashSet::new(),
            version: 0,
        }
    }

    /// Add a compilation unit to the graph
    pub fn add_unit(&mut self, unit: CompilationUnit) {
        let id = unit.id;
        let path = unit.path.clone();
        self.units.insert(id, unit);
        self.path_to_id.insert(path, id);
        self.version += 1;
    }

    /// Get a unit by ID
    pub fn get_unit(&self, id: UnitId) -> Option<&CompilationUnit> {
        self.units.get(&id)
    }

    /// Get a mutable unit by ID
    pub fn get_unit_mut(&mut self, id: UnitId) -> Option<&mut CompilationUnit> {
        self.units.get_mut(&id)
    }

    /// Get unit ID by path
    pub fn get_unit_id(&self, path: &Path) -> Option<UnitId> {
        self.path_to_id.get(path).copied()
    }

    /// Add a dependency edge
    pub fn add_dependency(&mut self, from: UnitId, to: UnitId) -> Result<(), BuildGraphError> {
        // Check for cycle before adding
        if self.would_create_cycle(from, to) {
            return Err(BuildGraphError::CyclicDependency {
                from: self.units.get(&from).map(|u| u.path.clone()),
                to: self.units.get(&to).map(|u| u.path.clone()),
            });
        }

        // Add forward edge
        if let Some(unit) = self.units.get_mut(&from)
            && !unit.dependencies.contains(&to)
        {
            unit.dependencies.push(to);
        }

        // Add reverse edge
        if let Some(unit) = self.units.get_mut(&to)
            && !unit.dependents.contains(&from)
        {
            unit.dependents.push(from);
        }

        self.version += 1;
        Ok(())
    }

    /// Mark a unit as root (entry point)
    pub fn add_root(&mut self, id: UnitId) {
        self.roots.insert(id);
    }

    /// Get all root units
    pub fn roots(&self) -> impl Iterator<Item = &CompilationUnit> {
        self.roots.iter().filter_map(|id| self.units.get(id))
    }

    /// Check if adding an edge would create a cycle
    fn would_create_cycle(&self, from: UnitId, to: UnitId) -> bool {
        if from == to {
            return true;
        }

        // BFS from 'to' to see if we can reach 'from'
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(to);

        while let Some(current) = queue.pop_front() {
            if current == from {
                return true;
            }

            if visited.insert(current)
                && let Some(unit) = self.units.get(&current)
            {
                for &dep in &unit.dependencies {
                    queue.push_back(dep);
                }
            }
        }

        false
    }

    /// Detect cycles in the dependency graph
    pub fn detect_cycles(&self) -> Vec<Vec<UnitId>> {
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut path = Vec::new();

        for &id in self.units.keys() {
            if !visited.contains(&id) {
                self.find_cycles_dfs(id, &mut visited, &mut rec_stack, &mut path, &mut cycles);
            }
        }

        cycles
    }

    /// DFS helper for cycle detection
    fn find_cycles_dfs(
        &self,
        node: UnitId,
        visited: &mut HashSet<UnitId>,
        rec_stack: &mut HashSet<UnitId>,
        path: &mut Vec<UnitId>,
        cycles: &mut Vec<Vec<UnitId>>,
    ) {
        visited.insert(node);
        rec_stack.insert(node);
        path.push(node);

        if let Some(unit) = self.units.get(&node) {
            for &dep in &unit.dependencies {
                if !visited.contains(&dep) {
                    self.find_cycles_dfs(dep, visited, rec_stack, path, cycles);
                } else if rec_stack.contains(&dep) {
                    // Found a cycle
                    if let Some(cycle_start) = path.iter().position(|&id| id == dep) {
                        cycles.push(path[cycle_start..].to_vec());
                    }
                }
            }
        }

        path.pop();
        rec_stack.remove(&node);
    }

    /// Topological sort of the dependency graph
    /// Returns units in build order (dependencies before dependents)
    pub fn topological_sort(&self) -> Result<Vec<UnitId>, BuildGraphError> {
        let mut in_degree: HashMap<UnitId, usize> = HashMap::new();

        // Calculate in-degrees (number of dependencies each node has)
        for (id, unit) in &self.units {
            // Each node's in-degree is its dependency count
            in_degree.insert(*id, unit.dependencies.len());
        }

        // Queue nodes with in-degree 0 (no dependencies - can be built first)
        let mut queue: VecDeque<UnitId> = in_degree
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut sorted = Vec::new();

        while let Some(node) = queue.pop_front() {
            sorted.push(node);

            // When we "build" this node, reduce in-degree of its dependents
            if let Some(unit) = self.units.get(&node) {
                for &dependent in &unit.dependents {
                    if let Some(degree) = in_degree.get_mut(&dependent) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(dependent);
                        }
                    }
                }
            }
        }

        // Check if all nodes were processed
        if sorted.len() != self.units.len() {
            Err(BuildGraphError::CyclicDependency {
                from: None,
                to: None,
            })
        } else {
            Ok(sorted)
        }
    }

    /// Mark a unit and its dependents as dirty
    pub fn invalidate(&mut self, id: UnitId) {
        let mut to_invalidate = VecDeque::new();
        to_invalidate.push_back(id);

        while let Some(current) = to_invalidate.pop_front() {
            if let Some(unit) = self.units.get_mut(&current)
                && !unit.dirty
            {
                unit.dirty = true;
                for &dependent in &unit.dependents {
                    to_invalidate.push_back(dependent);
                }
            }
        }

        self.version += 1;
    }

    /// Get all dirty units
    pub fn dirty_units(&self) -> Vec<UnitId> {
        self.units
            .iter()
            .filter(|(_, unit)| unit.dirty)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Mark a unit as clean (compiled)
    pub fn mark_clean(&mut self, id: UnitId) {
        if let Some(unit) = self.units.get_mut(&id) {
            unit.dirty = false;
            self.version += 1;
        }
    }

    /// Get all units
    pub fn units(&self) -> impl Iterator<Item = &CompilationUnit> {
        self.units.values()
    }

    /// Get the number of units
    pub fn len(&self) -> usize {
        self.units.len()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    /// Get graph version (incremented on changes)
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Prune units that no longer exist on disk
    pub fn prune_missing(&mut self) {
        let mut to_remove = Vec::new();

        for (id, unit) in &self.units {
            if !unit.path.exists() {
                to_remove.push(*id);
            }
        }

        for id in to_remove {
            self.remove_unit(id);
        }
    }

    /// Remove a unit from the graph
    pub fn remove_unit(&mut self, id: UnitId) {
        if let Some(unit) = self.units.remove(&id) {
            self.path_to_id.remove(&unit.path);
            self.roots.remove(&id);

            // Remove from dependencies of other units
            for other_unit in self.units.values_mut() {
                other_unit.dependencies.retain(|&dep_id| dep_id != id);
                other_unit.dependents.retain(|&dep_id| dep_id != id);
            }

            self.version += 1;
        }
    }

    /// Get compilation order (respecting dependencies)
    pub fn compilation_order(&self) -> Result<Vec<UnitId>, BuildGraphError> {
        // Only include dirty units
        let dirty: HashSet<_> = self.dirty_units().into_iter().collect();

        if dirty.is_empty() {
            return Ok(Vec::new());
        }

        // Get full topological order
        let full_order = self.topological_sort()?;

        // Filter to only dirty units, but maintain order
        Ok(full_order
            .into_iter()
            .filter(|id| dirty.contains(id))
            .collect())
    }

    /// Save graph to disk
    pub fn save(&self, path: &Path) -> Result<(), BuildGraphError> {
        let data = bincode::serialize(&GraphSnapshot {
            units: self.units.values().cloned().collect(),
            roots: self.roots.iter().copied().collect(),
            version: self.version,
        })
        .map_err(|e| BuildGraphError::SerializationError(e.to_string()))?;

        std::fs::write(path, data).map_err(BuildGraphError::IoError)?;

        Ok(())
    }

    /// Load graph from disk
    pub fn load(path: &Path) -> Result<Self, BuildGraphError> {
        let data = std::fs::read(path).map_err(BuildGraphError::IoError)?;

        let snapshot: GraphSnapshot = bincode::deserialize(&data)
            .map_err(|e| BuildGraphError::SerializationError(e.to_string()))?;

        let mut graph = BuildGraph::new();

        for unit in snapshot.units {
            let id = unit.id;
            let path = unit.path.clone();
            graph.units.insert(id, unit);
            graph.path_to_id.insert(path, id);
        }

        graph.roots = snapshot.roots.into_iter().collect();
        graph.version = snapshot.version;

        Ok(graph)
    }
}

impl Default for BuildGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable snapshot of build graph
#[derive(Serialize, Deserialize)]
struct GraphSnapshot {
    units: Vec<CompilationUnit>,
    roots: Vec<UnitId>,
    version: u64,
}

/// Build graph errors
#[derive(Debug, thiserror::Error)]
pub enum BuildGraphError {
    #[error("Cyclic dependency detected")]
    CyclicDependency {
        from: Option<PathBuf>,
        to: Option<PathBuf>,
    },

    #[error("Unit not found: {0:?}")]
    UnitNotFound(UnitId),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_hash() {
        let data = b"hello world";
        let hash1 = ContentHash::from_bytes(data);
        let hash2 = ContentHash::from_bytes(data);
        assert_eq!(hash1, hash2);

        let hex = hash1.to_hex();
        let hash3 = ContentHash::from_hex(&hex).unwrap();
        assert_eq!(hash1, hash3);
    }

    #[test]
    fn test_build_graph_basic() {
        let mut graph = BuildGraph::new();

        let path1 = PathBuf::from("test1.sio");
        let hash1 = ContentHash::from_bytes(b"module test1");
        let unit1 = CompilationUnit::new(path1, hash1);
        let id1 = unit1.id;

        graph.add_unit(unit1);
        assert_eq!(graph.len(), 1);
        assert!(graph.get_unit(id1).is_some());
    }

    #[test]
    fn test_dependency_tracking() {
        let mut graph = BuildGraph::new();

        let path1 = PathBuf::from("main.sio");
        let path2 = PathBuf::from("lib.sio");

        let unit1 = CompilationUnit::new(path1, ContentHash::from_bytes(b"main"));
        let unit2 = CompilationUnit::new(path2, ContentHash::from_bytes(b"lib"));

        let id1 = unit1.id;
        let id2 = unit2.id;

        graph.add_unit(unit1);
        graph.add_unit(unit2);

        // main depends on lib
        graph.add_dependency(id1, id2).unwrap();

        let unit1 = graph.get_unit(id1).unwrap();
        assert!(unit1.dependencies.contains(&id2));

        let unit2 = graph.get_unit(id2).unwrap();
        assert!(unit2.dependents.contains(&id1));
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = BuildGraph::new();

        let unit1 = CompilationUnit::new(PathBuf::from("a.sio"), ContentHash::from_bytes(b"a"));
        let unit2 = CompilationUnit::new(PathBuf::from("b.sio"), ContentHash::from_bytes(b"b"));

        let id1 = unit1.id;
        let id2 = unit2.id;

        graph.add_unit(unit1);
        graph.add_unit(unit2);

        // a -> b
        graph.add_dependency(id1, id2).unwrap();

        // b -> a would create a cycle
        assert!(graph.add_dependency(id2, id1).is_err());
    }

    #[test]
    fn test_invalidation() {
        let mut graph = BuildGraph::new();

        let unit1 = CompilationUnit::new(PathBuf::from("a.sio"), ContentHash::from_bytes(b"a"));
        let unit2 = CompilationUnit::new(PathBuf::from("b.sio"), ContentHash::from_bytes(b"b"));
        let unit3 = CompilationUnit::new(PathBuf::from("c.sio"), ContentHash::from_bytes(b"c"));

        let id1 = unit1.id;
        let id2 = unit2.id;
        let id3 = unit3.id;

        graph.add_unit(unit1);
        graph.add_unit(unit2);
        graph.add_unit(unit3);

        // c -> b -> a
        graph.add_dependency(id3, id2).unwrap();
        graph.add_dependency(id2, id1).unwrap();

        // Mark all as clean
        graph.mark_clean(id1);
        graph.mark_clean(id2);
        graph.mark_clean(id3);

        assert_eq!(graph.dirty_units().len(), 0);

        // Invalidate a
        graph.invalidate(id1);

        // Both b and c should be dirty
        let dirty = graph.dirty_units();
        assert!(dirty.contains(&id1));
        assert!(dirty.contains(&id2));
        assert!(dirty.contains(&id3));
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = BuildGraph::new();

        let unit1 = CompilationUnit::new(PathBuf::from("a.sio"), ContentHash::from_bytes(b"a"));
        let unit2 = CompilationUnit::new(PathBuf::from("b.sio"), ContentHash::from_bytes(b"b"));
        let unit3 = CompilationUnit::new(PathBuf::from("c.sio"), ContentHash::from_bytes(b"c"));

        let id1 = unit1.id;
        let id2 = unit2.id;
        let id3 = unit3.id;

        graph.add_unit(unit1);
        graph.add_unit(unit2);
        graph.add_unit(unit3);

        // c -> b, c -> a, b -> a
        graph.add_dependency(id3, id2).unwrap();
        graph.add_dependency(id3, id1).unwrap();
        graph.add_dependency(id2, id1).unwrap();

        let sorted = graph.topological_sort().unwrap();

        // a should come before b and c
        let pos_a = sorted.iter().position(|&id| id == id1).unwrap();
        let pos_b = sorted.iter().position(|&id| id == id2).unwrap();
        let pos_c = sorted.iter().position(|&id| id == id3).unwrap();

        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_c);
    }
}
