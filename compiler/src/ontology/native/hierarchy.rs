//! Hierarchy Index with O(1) LCA Queries
//!
//! Implements Bender & Farach-Colton (2000) algorithm for
//! constant-time Lowest Common Ancestor queries.
//!
//! # Algorithm Overview
//!
//! 1. **Euler Tour**: DFS traversal recording each node visit
//! 2. **Depth Array**: Depth at each Euler tour position
//! 3. **First Occurrence**: First position of each node in tour
//! 4. **Sparse Table**: RMQ (Range Minimum Query) for O(1) LCA
//!
//! The key insight: LCA(u, v) is the shallowest node between
//! first[u] and first[v] in the Euler tour.

use std::collections::HashMap;

use super::storage::NativeStore;
use crate::ontology::OntologyResult;

/// Query interface for LCA operations
pub trait LcaQuery {
    /// Check if `ancestor` is an ancestor of `descendant`
    fn is_ancestor(&self, descendant: &str, ancestor: &str) -> bool;

    /// Find the Lowest Common Ancestor of two nodes
    fn lca(&self, u: &str, v: &str) -> Option<&str>;

    /// Get all ancestors of a node (transitive closure)
    fn get_ancestors(&self, node: &str) -> Vec<&str>;
}

/// Hierarchy index using Euler tour + sparse table for O(1) LCA
#[derive(Debug)]
pub struct HierarchyIndex {
    /// Euler tour: sequence of node indices
    euler_tour: Vec<usize>,
    /// Depth at each position in Euler tour
    depths: Vec<usize>,
    /// First occurrence of each node in Euler tour
    first: HashMap<String, usize>,
    /// Sparse table for RMQ
    sparse_table: SparseTable,
    /// Reference to concepts for CURIE lookup
    curie_to_idx: HashMap<String, usize>,
    /// Index to CURIE (for reverse lookup in LCA result)
    idx_to_curie: Vec<String>,
    /// Parent indices for ancestor traversal
    parents: Vec<Option<usize>>,
}

impl HierarchyIndex {
    /// Build the hierarchy index from a native store
    pub fn build(store: &NativeStore) -> OntologyResult<Self> {
        let n = store.concepts.len();
        if n == 0 {
            return Ok(Self::empty());
        }

        // Build curie mappings
        let mut curie_to_idx = HashMap::new();
        let mut idx_to_curie = Vec::with_capacity(n);
        let mut parents = Vec::with_capacity(n);
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut roots = Vec::new();

        for i in 0..n {
            if let Some(curie) = store.curie_at(i) {
                curie_to_idx.insert(curie.to_string(), i);
                idx_to_curie.push(curie.to_string());
            } else {
                idx_to_curie.push(String::new());
            }

            if let Some(parent_idx) = store.parent_idx(i) {
                parents.push(Some(parent_idx));
                children[parent_idx].push(i);
            } else {
                parents.push(None);
                roots.push(i);
            }
        }

        // Build Euler tour via DFS
        let mut euler_tour = Vec::with_capacity(2 * n);
        let mut depths = Vec::with_capacity(2 * n);
        let mut first = HashMap::new();

        fn dfs(
            node: usize,
            depth: usize,
            children: &[Vec<usize>],
            euler_tour: &mut Vec<usize>,
            depths: &mut Vec<usize>,
            first: &mut HashMap<String, usize>,
            idx_to_curie: &[String],
        ) {
            let pos = euler_tour.len();
            euler_tour.push(node);
            depths.push(depth);

            // Record first occurrence
            let curie = &idx_to_curie[node];
            if !curie.is_empty() && !first.contains_key(curie) {
                first.insert(curie.clone(), pos);
            }

            // Visit children
            for &child in &children[node] {
                dfs(
                    child,
                    depth + 1,
                    children,
                    euler_tour,
                    depths,
                    first,
                    idx_to_curie,
                );
                // Back to parent
                euler_tour.push(node);
                depths.push(depth);
            }
        }

        // Start DFS from all roots
        for root in roots {
            dfs(
                root,
                0,
                &children,
                &mut euler_tour,
                &mut depths,
                &mut first,
                &idx_to_curie,
            );
        }

        // Build sparse table for RMQ
        let sparse_table = SparseTable::build(&depths);

        Ok(Self {
            euler_tour,
            depths,
            first,
            sparse_table,
            curie_to_idx,
            idx_to_curie,
            parents,
        })
    }

    /// Create an empty index
    pub fn empty() -> Self {
        Self {
            euler_tour: Vec::new(),
            depths: Vec::new(),
            first: HashMap::new(),
            sparse_table: SparseTable::empty(),
            curie_to_idx: HashMap::new(),
            idx_to_curie: Vec::new(),
            parents: Vec::new(),
        }
    }

    /// Check if descendant is under ancestor in the hierarchy
    ///
    /// This is O(1) using the Euler tour property:
    /// u is an ancestor of v iff first[u] <= first[v] <= last[u]
    pub fn is_ancestor(&self, descendant: &str, ancestor: &str) -> bool {
        if descendant == ancestor {
            return true;
        }

        // Find LCA and check if it equals ancestor
        if let Some(lca) = self.lca(descendant, ancestor) {
            lca == ancestor
        } else {
            false
        }
    }

    /// Find LCA of two nodes in O(1)
    pub fn lca(&self, u: &str, v: &str) -> Option<&str> {
        let pos_u = *self.first.get(u)?;
        let pos_v = *self.first.get(v)?;

        let (left, right) = if pos_u <= pos_v {
            (pos_u, pos_v)
        } else {
            (pos_v, pos_u)
        };

        // RMQ to find minimum depth position
        let min_pos = self.sparse_table.rmq(left, right, &self.depths)?;

        // Get node at that position
        let node_idx = *self.euler_tour.get(min_pos)?;
        self.idx_to_curie.get(node_idx).map(|s| s.as_str())
    }

    /// Get all ancestors of a node (walks up the tree)
    pub fn get_ancestors(&self, node: &str) -> Vec<&str> {
        let mut ancestors = Vec::new();

        let mut current = self.curie_to_idx.get(node).copied();
        while let Some(idx) = current {
            if let Some(parent_idx) = self.parents.get(idx).and_then(|p| *p) {
                if let Some(curie) = self.idx_to_curie.get(parent_idx) {
                    ancestors.push(curie.as_str());
                }
                current = Some(parent_idx);
            } else {
                break;
            }
        }

        ancestors
    }
}

impl LcaQuery for HierarchyIndex {
    fn is_ancestor(&self, descendant: &str, ancestor: &str) -> bool {
        self.is_ancestor(descendant, ancestor)
    }

    fn lca(&self, u: &str, v: &str) -> Option<&str> {
        self.lca(u, v)
    }

    fn get_ancestors(&self, node: &str) -> Vec<&str> {
        self.get_ancestors(node)
    }
}

/// Sparse table for Range Minimum Query (RMQ)
///
/// Preprocessing: O(n log n)
/// Query: O(1)
#[derive(Debug)]
struct SparseTable {
    /// table[k][i] = index of minimum in range [i, i + 2^k)
    table: Vec<Vec<usize>>,
    /// log2 lookup table
    log2: Vec<usize>,
}

impl SparseTable {
    /// Build sparse table from depth array
    fn build(depths: &[usize]) -> Self {
        let n = depths.len();
        if n == 0 {
            return Self::empty();
        }

        // Compute log2 table
        let mut log2 = vec![0; n + 1];
        for i in 2..=n {
            log2[i] = log2[i / 2] + 1;
        }

        let k = log2[n] + 1;
        let mut table = vec![vec![0; n]; k];

        // Base case: table[0][i] = i (each element is its own minimum)
        for i in 0..n {
            table[0][i] = i;
        }

        // Build table
        for j in 1..k {
            let len = 1 << j;
            for i in 0..=n.saturating_sub(len) {
                let left = table[j - 1][i];
                let right_start = i + (1 << (j - 1));
                if right_start < n {
                    let right = table[j - 1][right_start];
                    table[j][i] = if depths[left] <= depths[right] {
                        left
                    } else {
                        right
                    };
                } else {
                    table[j][i] = left;
                }
            }
        }

        Self { table, log2 }
    }

    /// Create empty sparse table
    fn empty() -> Self {
        Self {
            table: Vec::new(),
            log2: Vec::new(),
        }
    }

    /// Query minimum in range [left, right] (inclusive)
    fn rmq(&self, left: usize, right: usize, depths: &[usize]) -> Option<usize> {
        if left > right || right >= depths.len() {
            return None;
        }

        let len = right - left + 1;
        if len == 0 || self.log2.len() <= len {
            return None;
        }

        let k = self.log2[len];
        if k >= self.table.len() {
            return None;
        }

        let left_min = self.table[k].get(left).copied()?;
        let right_start = right + 1 - (1 << k);
        let right_min = self.table[k].get(right_start).copied()?;

        Some(if depths[left_min] <= depths[right_min] {
            left_min
        } else {
            right_min
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_table_basic() {
        let depths = vec![0, 1, 2, 1, 2, 1, 0];
        let table = SparseTable::build(&depths);

        // RMQ should find minimum
        let min = table.rmq(0, 6, &depths).unwrap();
        assert!(depths[min] == 0); // Root has depth 0

        let min = table.rmq(1, 4, &depths).unwrap();
        assert!(depths[min] == 1); // Minimum in range is 1
    }

    #[test]
    fn test_sparse_table_single() {
        let depths = vec![5];
        let table = SparseTable::build(&depths);
        let min = table.rmq(0, 0, &depths).unwrap();
        assert_eq!(min, 0);
    }

    #[test]
    fn test_ancestor_check() {
        // Simple tree: A -> B -> C
        // Euler tour: A B C B A
        // Depths:     0 1 2 1 0
        // First: A=0, B=1, C=2

        let euler_tour = vec![0, 1, 2, 1, 0];
        let depths = vec![0, 1, 2, 1, 0];
        let mut first = HashMap::new();
        first.insert("A".to_string(), 0);
        first.insert("B".to_string(), 1);
        first.insert("C".to_string(), 2);

        let mut curie_to_idx = HashMap::new();
        curie_to_idx.insert("A".to_string(), 0);
        curie_to_idx.insert("B".to_string(), 1);
        curie_to_idx.insert("C".to_string(), 2);

        let idx_to_curie = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let parents = vec![None, Some(0), Some(1)];

        let sparse_table = SparseTable::build(&depths);

        let index = HierarchyIndex {
            euler_tour,
            depths,
            first,
            sparse_table,
            curie_to_idx,
            idx_to_curie,
            parents,
        };

        // A is ancestor of B
        assert!(index.is_ancestor("B", "A"));
        // A is ancestor of C
        assert!(index.is_ancestor("C", "A"));
        // B is ancestor of C
        assert!(index.is_ancestor("C", "B"));
        // C is not ancestor of A
        assert!(!index.is_ancestor("A", "C"));
        // C is not ancestor of B
        assert!(!index.is_ancestor("B", "C"));

        // LCA tests
        assert_eq!(index.lca("B", "C"), Some("B"));
        assert_eq!(index.lca("A", "C"), Some("A"));
    }

    #[test]
    fn test_get_ancestors() {
        let mut curie_to_idx = HashMap::new();
        curie_to_idx.insert("A".to_string(), 0);
        curie_to_idx.insert("B".to_string(), 1);
        curie_to_idx.insert("C".to_string(), 2);

        let idx_to_curie = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let parents = vec![None, Some(0), Some(1)];

        let index = HierarchyIndex {
            euler_tour: vec![0, 1, 2, 1, 0],
            depths: vec![0, 1, 2, 1, 0],
            first: HashMap::new(),
            sparse_table: SparseTable::empty(),
            curie_to_idx,
            idx_to_curie,
            parents,
        };

        let ancestors = index.get_ancestors("C");
        assert_eq!(ancestors, vec!["B", "A"]);
    }
}
