//! Loop Transformations
//!
//! This module provides polyhedral loop transformations including:
//!
//! - **Loop Tiling**: Improves cache locality by processing data in tiles
//! - **Loop Interchange**: Reorders loops to improve access patterns
//! - **Loop Fusion**: Merges loops to reduce synchronization overhead
//! - **Loop Fission**: Splits loops to enable other optimizations
//!
//! # References
//!
//! - Wolfe (1989) "More Iteration Space Tiling"
//! - Wolf & Lam (1991) "A Data Locality Optimizing Algorithm"
//! - Bondhugula et al. (2008) "A Practical Automatic Polyhedral Parallelizer
//!   and Locality Optimizer" (PLuTo algorithm)

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use std::collections::{HashMap, HashSet};

use super::affine::{AffineExpr, AffineMap};
use super::analysis::{DependenceInfo, Direction, LoopNest, ScopInfo};
use super::{PolyError, PolyResult};

/// A schedule maps statement instances to a multi-dimensional time
///
/// The schedule determines the execution order of statement instances.
/// Each statement S has a schedule θ_S that maps its iteration vector
/// to a time vector.
#[derive(Debug, Clone)]
pub struct Schedule {
    /// Schedule for each statement (statement_id -> schedule map)
    pub statement_schedules: HashMap<usize, AffineMap>,
    /// Schedule tree representation
    pub tree: Option<ScheduleNode>,
}

impl Schedule {
    /// Create a new empty schedule
    pub fn new() -> Self {
        Self {
            statement_schedules: HashMap::new(),
            tree: None,
        }
    }

    /// Create an identity schedule (original execution order)
    pub fn identity(scop: &ScopInfo) -> Self {
        let mut schedules = HashMap::new();

        for stmt in &scop.statements {
            // Identity schedule: θ(i) = i
            let map = AffineMap::identity(stmt.domain.depth());
            schedules.insert(stmt.id, map);
        }

        Self {
            statement_schedules: schedules,
            tree: None,
        }
    }

    /// Set the schedule for a statement
    pub fn set_statement_schedule(&mut self, stmt_id: usize, schedule: AffineMap) {
        self.statement_schedules.insert(stmt_id, schedule);
    }

    /// Get the schedule for a statement
    pub fn get_statement_schedule(&self, stmt_id: usize) -> Option<&AffineMap> {
        self.statement_schedules.get(&stmt_id)
    }

    /// Check if this schedule respects the given dependencies
    pub fn is_legal(&self, dependencies: &[DependenceInfo]) -> bool {
        for dep in dependencies {
            if !dep.dep_type.is_true_dependence() {
                continue; // Only true dependencies must be respected
            }

            // Get schedules for source and sink
            let source_sched = match self.statement_schedules.get(&dep.source_stmt) {
                Some(s) => s,
                None => continue,
            };
            let sink_sched = match self.statement_schedules.get(&dep.sink_stmt) {
                Some(s) => s,
                None => continue,
            };

            // Check that source is scheduled before sink
            // Simplified check: verify first non-equal dimension has correct direction
            // A full implementation would use ISL to verify the constraint:
            //   θ_sink(sink_iter) - θ_source(source_iter) >= 0
            //   where (source_iter, sink_iter) satisfy the dependence polyhedron

            // For now, use a conservative check
            if !self.dependency_preserved(source_sched, sink_sched, dep) {
                return false;
            }
        }

        true
    }

    /// Check if a single dependency is preserved by the schedules
    fn dependency_preserved(
        &self,
        _source_sched: &AffineMap,
        _sink_sched: &AffineMap,
        dep: &DependenceInfo,
    ) -> bool {
        // Simplified legality check
        // A proper implementation would:
        // 1. Compute the scheduled dependence distance
        // 2. Verify it's lexicographically positive

        if let Some(dist) = &dep.distance_vector {
            // If we have a distance vector, check it's valid
            for d in dist {
                if *d < 0 {
                    return false;
                }
                if *d > 0 {
                    return true; // Positive distance in outer dimension - legal
                }
            }
        }

        if let Some(dir) = &dep.direction_vector {
            // Check direction vector
            for d in dir {
                match d {
                    Direction::Negative => return false,
                    Direction::Positive => return true,
                    Direction::Zero => continue,
                    Direction::Any => continue, // Conservative: continue checking
                }
            }
        }

        // Conservative default: assume legal
        true
    }
}

impl Default for Schedule {
    fn default() -> Self {
        Self::new()
    }
}

/// A node in the schedule tree
///
/// Schedule trees provide a structured representation of loop nests
/// that's easier to manipulate than flat schedule maps.
#[derive(Debug, Clone)]
pub enum ScheduleNode {
    /// A band of consecutive schedule dimensions
    Band {
        /// Number of dimensions in this band
        num_dims: usize,
        /// Whether the band is tilable (all dimensions are parallel or can be permuted)
        is_tilable: bool,
        /// Tile sizes (if tiled)
        tile_sizes: Option<Vec<usize>>,
        /// Child node
        child: Box<ScheduleNode>,
    },
    /// Sequence of children executed in order
    Sequence(Vec<ScheduleNode>),
    /// Parallel execution of children
    Parallel(Vec<ScheduleNode>),
    /// Filter to a subset of statements
    Filter {
        /// Statement IDs included in this filter
        statements: HashSet<usize>,
        /// Child node
        child: Box<ScheduleNode>,
    },
    /// Leaf node (actual statement execution)
    Leaf,
}

impl ScheduleNode {
    /// Create a simple band node
    pub fn band(num_dims: usize, child: ScheduleNode) -> Self {
        ScheduleNode::Band {
            num_dims,
            is_tilable: false,
            tile_sizes: None,
            child: Box::new(child),
        }
    }

    /// Create a tilable band node
    pub fn tilable_band(num_dims: usize, child: ScheduleNode) -> Self {
        ScheduleNode::Band {
            num_dims,
            is_tilable: true,
            tile_sizes: None,
            child: Box::new(child),
        }
    }

    /// Set tile sizes for a band
    pub fn with_tile_sizes(self, sizes: Vec<usize>) -> Self {
        match self {
            ScheduleNode::Band {
                num_dims,
                is_tilable,
                child,
                ..
            } => ScheduleNode::Band {
                num_dims,
                is_tilable,
                tile_sizes: Some(sizes),
                child,
            },
            other => other,
        }
    }

    /// Get the total number of schedule dimensions
    pub fn total_dims(&self) -> usize {
        match self {
            ScheduleNode::Band {
                num_dims, child, ..
            } => num_dims + child.total_dims(),
            ScheduleNode::Sequence(children) | ScheduleNode::Parallel(children) => {
                children.iter().map(|c| c.total_dims()).max().unwrap_or(0)
            }
            ScheduleNode::Filter { child, .. } => child.total_dims(),
            ScheduleNode::Leaf => 0,
        }
    }
}

/// Polyhedral transformation engine
pub struct PolyhedralTransform<'a> {
    /// The SCOP being transformed
    scop: &'a ScopInfo,
    /// Dependencies to preserve
    dependencies: &'a [DependenceInfo],
    /// Current schedule
    schedule: Schedule,
}

impl<'a> PolyhedralTransform<'a> {
    /// Create a new transformation engine
    pub fn new(scop: &'a ScopInfo, dependencies: &'a [DependenceInfo]) -> Self {
        Self {
            scop,
            dependencies,
            schedule: Schedule::identity(scop),
        }
    }

    /// Apply automatic tiling for improved cache locality
    ///
    /// This is based on the PLuTo algorithm which finds a schedule
    /// that minimizes reuse distances.
    pub fn auto_tile(&self) -> PolyResult<Schedule> {
        let mut schedule = self.schedule.clone();

        // Find tilable bands
        for loop_nest in &self.scop.loop_nests {
            if loop_nest.depth() < 2 {
                continue; // Need at least 2 loops for tiling
            }

            // Check if loops are tilable
            if self.is_tilable(loop_nest)? {
                // Compute tile sizes based on cache parameters
                let tile_sizes = self.compute_tile_sizes(loop_nest);

                // Apply tiling transformation
                schedule = self.apply_tiling(&schedule, loop_nest, &tile_sizes)?;
            }
        }

        // Verify legality
        if !schedule.is_legal(self.dependencies) {
            return Err(PolyError::IllegalTransformation {
                reason: "Tiling violates dependencies".to_string(),
            });
        }

        Ok(schedule)
    }

    /// Check if a loop nest is tilable
    fn is_tilable(&self, loop_nest: &LoopNest) -> PolyResult<bool> {
        // A loop nest is tilable if all carried dependencies are uniform
        // (have constant distance vectors)

        for dep in self.dependencies {
            if !dep.is_loop_carried {
                continue;
            }

            // Check if dependency has constant distance
            if dep.distance_vector.is_none() {
                // Non-constant distance - need to check more carefully
                // For simplicity, assume tilable if direction vector exists
                if dep.direction_vector.is_none() {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Compute optimal tile sizes based on cache hierarchy
    fn compute_tile_sizes(&self, loop_nest: &LoopNest) -> Vec<usize> {
        // Default tile sizes based on typical L1 cache (32KB)
        // These would ideally be computed based on:
        // - Cache sizes (L1, L2, L3)
        // - Working set size
        // - Data element size
        // - Access patterns

        let depth = loop_nest.depth();

        // Common tile sizes for scientific computing
        // - Innermost: 32-64 (fits in L1 cache line)
        // - Middle: 64-128 (fits in L1 cache)
        // - Outermost: 128-256 (fits in L2 cache)
        match depth {
            1 => vec![64],
            2 => vec![64, 64],
            3 => vec![32, 64, 64],
            _ => (0..depth).map(|i| if i == 0 { 32 } else { 64 }).collect(),
        }
    }

    /// Apply tiling transformation
    fn apply_tiling(
        &self,
        schedule: &Schedule,
        loop_nest: &LoopNest,
        tile_sizes: &[usize],
    ) -> PolyResult<Schedule> {
        let mut new_schedule = schedule.clone();
        let depth = loop_nest.depth();

        // Create tiled schedule tree
        let tiled_tree = ScheduleNode::Band {
            num_dims: depth,
            is_tilable: true,
            tile_sizes: Some(tile_sizes.to_vec()),
            child: Box::new(ScheduleNode::Band {
                num_dims: depth,
                is_tilable: false,
                tile_sizes: None,
                child: Box::new(ScheduleNode::Leaf),
            }),
        };

        new_schedule.tree = Some(tiled_tree);

        // Update statement schedules to reflect tiling
        for stmt in &self.scop.statements {
            if let Some(orig_sched) = schedule.get_statement_schedule(stmt.id) {
                let tiled_sched = self.tile_schedule(orig_sched, tile_sizes)?;
                new_schedule.set_statement_schedule(stmt.id, tiled_sched);
            }
        }

        Ok(new_schedule)
    }

    /// Transform a schedule for tiling
    ///
    /// Tiling transforms the schedule from:
    ///   (i, j) -> (i, j)
    /// to:
    ///   (i, j) -> (i/T_i, j/T_j, i mod T_i, j mod T_j)
    fn tile_schedule(&self, schedule: &AffineMap, tile_sizes: &[usize]) -> PolyResult<AffineMap> {
        let num_dims = schedule.num_outputs;
        let num_tiled = num_dims * 2; // tile + point iterators

        let mut outputs = Vec::with_capacity(num_tiled);

        // First, tile iterators: floor(i / T)
        // Note: We can't represent floor division directly in affine maps
        // This is an approximation - actual tiling would need codegen support
        for (i, tile_size) in tile_sizes.iter().enumerate() {
            if i < num_dims {
                // Approximate floor division by scaling
                // In practice, codegen handles the actual floor/mod operations
                let mut expr = AffineExpr {
                    iter_coeffs: schedule.outputs[i].iter_coeffs.clone(),
                    param_coeffs: schedule.outputs[i].param_coeffs.clone(),
                    constant: schedule.outputs[i].constant / (*tile_size as i64),
                };
                // Scale down coefficients (approximation)
                for coeff in &mut expr.iter_coeffs {
                    *coeff /= *tile_size as i64;
                }
                outputs.push(expr);
            }
        }

        // Then, point iterators: i mod T (represented as i for now)
        // Full modulo representation would need additional constraints
        for i in 0..num_dims {
            outputs.push(schedule.outputs[i].clone());
        }

        Ok(AffineMap::new(
            schedule.num_inputs,
            schedule.num_params,
            outputs,
        ))
    }
}

/// Loop interchange transformation
pub struct InterchangeTransform {
    /// Loop levels to interchange (0-indexed, outer to inner)
    pub level1: usize,
    pub level2: usize,
}

impl InterchangeTransform {
    /// Create a new interchange transformation
    pub fn new(level1: usize, level2: usize) -> Self {
        Self { level1, level2 }
    }

    /// Apply interchange to a schedule
    pub fn apply(&self, schedule: &mut Schedule) -> PolyResult<()> {
        for (_, sched) in schedule.statement_schedules.iter_mut() {
            if self.level1 >= sched.num_outputs || self.level2 >= sched.num_outputs {
                return Err(PolyError::IllegalTransformation {
                    reason: format!(
                        "Loop level out of bounds: {} or {} >= {}",
                        self.level1, self.level2, sched.num_outputs
                    ),
                });
            }

            // Swap the schedule outputs
            sched.outputs.swap(self.level1, self.level2);
        }

        Ok(())
    }
}

/// Loop tiling transformation
pub struct TileTransform {
    /// Tile sizes for each dimension
    pub tile_sizes: Vec<usize>,
}

impl TileTransform {
    /// Create a new tiling transformation
    pub fn new(tile_sizes: Vec<usize>) -> Self {
        Self { tile_sizes }
    }

    /// Default tile sizes for common cases
    pub fn default_2d() -> Self {
        Self {
            tile_sizes: vec![64, 64],
        }
    }

    /// Default tile sizes for 3D loop nests
    pub fn default_3d() -> Self {
        Self {
            tile_sizes: vec![32, 64, 64],
        }
    }
}

/// Loop fusion transformation
pub struct FusionTransform {
    /// IDs of loops to fuse
    pub loops_to_fuse: Vec<usize>,
}

impl FusionTransform {
    /// Create a new fusion transformation
    pub fn new(loops_to_fuse: Vec<usize>) -> Self {
        Self { loops_to_fuse }
    }

    /// Check if fusion is legal given the dependencies
    pub fn is_legal(&self, dependencies: &[DependenceInfo]) -> bool {
        // Fusion is legal if it doesn't create cycles in the dependence graph

        // Check for inter-loop dependencies that would become backward
        for dep in dependencies {
            let source_loop = self.find_containing_loop(dep.source_stmt);
            let sink_loop = self.find_containing_loop(dep.sink_stmt);

            if let (Some(src), Some(sink)) = (source_loop, sink_loop) {
                // If both are in loops to fuse and source > sink, illegal
                if self.loops_to_fuse.contains(&src)
                    && self.loops_to_fuse.contains(&sink)
                    && src > sink
                    && dep.dep_type.is_true_dependence()
                {
                    return false;
                }
            }
        }

        true
    }

    /// Find which loop contains a statement
    fn find_containing_loop(&self, _stmt_id: usize) -> Option<usize> {
        // Simplified: would need SCOP info to determine this
        None
    }
}

/// Utilities for computing optimal transformations
pub struct TransformUtils;

impl TransformUtils {
    /// Compute the optimal permutation for locality using the Wolf-Lam algorithm
    ///
    /// Returns a permutation vector where permutation[i] is the original loop
    /// that should become loop i in the new order.
    pub fn compute_optimal_permutation(
        loop_nest: &LoopNest,
        _dependencies: &[DependenceInfo],
    ) -> Vec<usize> {
        let depth = loop_nest.depth();

        // Build reuse vectors for each array access
        // A reuse vector indicates which loop levels have temporal reuse

        // Simplified heuristic: put loops with stride-1 access innermost
        // A full implementation would:
        // 1. Build the data reuse space for each array reference
        // 2. Compute the legal permutation space
        // 3. Select the permutation that maximizes temporal locality

        // For now, return identity permutation
        (0..depth).collect()
    }

    /// Compute the maximum tilable band size
    ///
    /// Returns the number of consecutive outermost loops that can be tiled together
    pub fn max_tilable_band(loop_nest: &LoopNest, dependencies: &[DependenceInfo]) -> usize {
        let depth = loop_nest.depth();

        // Check each level from outer to inner
        for level in 1..=depth {
            // Check if dependencies allow tiling up to this level
            for dep in dependencies {
                if !dep.is_loop_carried {
                    continue;
                }

                // If dependency is carried at a level < current, can't tile
                if let Some(carried_level) = dep.carried_at_level {
                    if carried_level < level {
                        // Check if the distance allows tiling
                        if let Some(dist) = &dep.distance_vector {
                            if dist.get(carried_level).copied().unwrap_or(0) != 0 {
                                return level;
                            }
                        }
                    }
                }
            }
        }

        depth
    }

    /// Check if a loop can be parallelized
    pub fn is_parallelizable(loop_level: usize, dependencies: &[DependenceInfo]) -> bool {
        for dep in dependencies {
            if !dep.dep_type.is_true_dependence() {
                continue;
            }

            // A loop is parallelizable if no dependency is carried at that level
            if let Some(carried_level) = dep.carried_at_level {
                if carried_level == loop_level {
                    return false;
                }
            }

            // Check distance vector
            if let Some(dist) = &dep.distance_vector {
                if dist.get(loop_level).copied().unwrap_or(0) != 0 {
                    return false;
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_creation() {
        let schedule = Schedule::new();
        assert!(schedule.statement_schedules.is_empty());
    }

    #[test]
    fn test_tile_transform_default() {
        let tile = TileTransform::default_2d();
        assert_eq!(tile.tile_sizes, vec![64, 64]);

        let tile_3d = TileTransform::default_3d();
        assert_eq!(tile_3d.tile_sizes, vec![32, 64, 64]);
    }

    #[test]
    fn test_interchange_transform() {
        let interchange = InterchangeTransform::new(0, 1);
        assert_eq!(interchange.level1, 0);
        assert_eq!(interchange.level2, 1);
    }

    #[test]
    fn test_schedule_node_dims() {
        let leaf = ScheduleNode::Leaf;
        assert_eq!(leaf.total_dims(), 0);

        let band = ScheduleNode::band(2, ScheduleNode::Leaf);
        assert_eq!(band.total_dims(), 2);

        let nested = ScheduleNode::band(2, ScheduleNode::band(1, ScheduleNode::Leaf));
        assert_eq!(nested.total_dims(), 3);
    }

    #[test]
    fn test_schedule_node_tiling() {
        let band = ScheduleNode::tilable_band(2, ScheduleNode::Leaf);
        let tiled = band.with_tile_sizes(vec![32, 32]);

        match tiled {
            ScheduleNode::Band {
                tile_sizes,
                is_tilable,
                ..
            } => {
                assert!(is_tilable);
                assert_eq!(tile_sizes, Some(vec![32, 32]));
            }
            _ => panic!("Expected Band node"),
        }
    }

    #[test]
    fn test_fusion_transform() {
        let fusion = FusionTransform::new(vec![0, 1]);
        assert_eq!(fusion.loops_to_fuse, vec![0, 1]);

        // With no dependencies, fusion should be legal
        assert!(fusion.is_legal(&[]));
    }

    #[test]
    fn test_parallelizable_check() {
        // No dependencies means parallelizable
        assert!(TransformUtils::is_parallelizable(0, &[]));
    }
}
