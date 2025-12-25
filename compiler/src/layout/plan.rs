//! Layout Plan Generation
//!
//! Converts clustering results into concrete memory layout recommendations.

use std::collections::HashMap;

use super::cluster::ClusteringResult;

/// Memory region classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryRegion {
    /// Hot data: frequently accessed, should be in L1/L2 cache
    Hot,
    /// Warm data: moderately accessed, L2/L3 cache
    Warm,
    /// Cold data: rarely accessed, main memory
    Cold,
}

impl MemoryRegion {
    /// Get the priority (lower = hotter)
    pub fn priority(&self) -> u8 {
        match self {
            MemoryRegion::Hot => 0,
            MemoryRegion::Warm => 1,
            MemoryRegion::Cold => 2,
        }
    }
}

/// Configuration for layout generation
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// Maximum number of clusters
    pub max_clusters: usize,
    /// Cache size for simulation (in number of concepts)
    pub cache_size: usize,
    /// Threshold for hot classification (percentile of access frequency)
    pub hot_threshold: f32,
    /// Threshold for warm classification
    pub warm_threshold: f32,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            max_clusters: 4,
            cache_size: 16,
            hot_threshold: 0.7,  // Top 30% of accesses = hot
            warm_threshold: 0.3, // Middle 40% = warm, bottom 30% = cold
        }
    }
}

/// A layout recommendation for a concept
#[derive(Debug, Clone)]
pub struct ConceptLayout {
    /// The concept CURIE
    pub concept: String,
    /// Recommended memory region
    pub region: MemoryRegion,
    /// Cluster ID this concept belongs to
    pub cluster_id: usize,
    /// Layout order within cluster (for cache line packing)
    pub order: usize,
}

/// Complete layout plan
#[derive(Debug, Clone)]
pub struct LayoutPlan {
    /// Layout for each concept
    pub layouts: Vec<ConceptLayout>,
    /// Concept -> layout lookup
    pub by_concept: HashMap<String, ConceptLayout>,
    /// Concepts grouped by region
    pub by_region: HashMap<MemoryRegion, Vec<String>>,
    /// Total concepts
    pub total_concepts: usize,
}

impl LayoutPlan {
    /// Create an empty layout plan
    pub fn empty() -> Self {
        Self {
            layouts: Vec::new(),
            by_concept: HashMap::new(),
            by_region: HashMap::new(),
            total_concepts: 0,
        }
    }

    /// Get layout for a specific concept
    pub fn get(&self, concept: &str) -> Option<&ConceptLayout> {
        self.by_concept.get(concept)
    }

    /// Get all concepts in a region
    pub fn in_region(&self, region: MemoryRegion) -> &[String] {
        self.by_region
            .get(&region)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get concepts sorted by layout order (for memory allocation)
    pub fn allocation_order(&self) -> Vec<&str> {
        let mut sorted: Vec<_> = self.layouts.iter().collect();
        sorted.sort_by(|a, b| {
            // Sort by region (hot first), then by cluster, then by order
            a.region
                .priority()
                .cmp(&b.region.priority())
                .then(a.cluster_id.cmp(&b.cluster_id))
                .then(a.order.cmp(&b.order))
        });
        sorted.iter().map(|l| l.concept.as_str()).collect()
    }

    /// Summary statistics
    pub fn summary(&self) -> LayoutSummary {
        LayoutSummary {
            total: self.total_concepts,
            hot: self.in_region(MemoryRegion::Hot).len(),
            warm: self.in_region(MemoryRegion::Warm).len(),
            cold: self.in_region(MemoryRegion::Cold).len(),
        }
    }
}

/// Summary of layout distribution
#[derive(Debug, Clone)]
pub struct LayoutSummary {
    pub total: usize,
    pub hot: usize,
    pub warm: usize,
    pub cold: usize,
}

/// Generate a layout plan from clustering results
pub fn generate_layout(clustering: ClusteringResult, config: LayoutConfig) -> LayoutPlan {
    if clustering.clusters.is_empty() {
        return LayoutPlan::empty();
    }

    // Sort clusters by hotness (total accesses)
    let sorted_clusters = clustering.sorted_by_hotness();

    // Calculate access thresholds
    let total_accesses: u32 = sorted_clusters.iter().map(|c| c.total_accesses).sum();
    let hot_cutoff = (total_accesses as f32 * config.hot_threshold) as u32;
    let warm_cutoff = (total_accesses as f32 * config.warm_threshold) as u32;

    let mut layouts = Vec::new();
    let mut by_concept = HashMap::new();
    let mut by_region: HashMap<MemoryRegion, Vec<String>> = HashMap::new();

    let mut cumulative_accesses = 0u32;

    for cluster in sorted_clusters {
        // Determine region based on cumulative access frequency
        let region = if cumulative_accesses < hot_cutoff {
            MemoryRegion::Hot
        } else if cumulative_accesses < hot_cutoff + warm_cutoff {
            MemoryRegion::Warm
        } else {
            MemoryRegion::Cold
        };

        // Create layouts for each concept in the cluster
        for (order, concept) in cluster.concepts.iter().enumerate() {
            let layout = ConceptLayout {
                concept: concept.clone(),
                region,
                cluster_id: cluster.id,
                order,
            };

            layouts.push(layout.clone());
            by_concept.insert(concept.clone(), layout);
            by_region.entry(region).or_default().push(concept.clone());
        }

        cumulative_accesses = cumulative_accesses.saturating_add(cluster.total_accesses);
    }

    let total_concepts = layouts.len();

    LayoutPlan {
        layouts,
        by_concept,
        by_region,
        total_concepts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_region_priority() {
        assert!(MemoryRegion::Hot.priority() < MemoryRegion::Warm.priority());
        assert!(MemoryRegion::Warm.priority() < MemoryRegion::Cold.priority());
    }

    #[test]
    fn test_layout_config_default() {
        let config = LayoutConfig::default();
        assert_eq!(config.max_clusters, 4);
        assert_eq!(config.cache_size, 16);
    }

    #[test]
    fn test_empty_layout_plan() {
        let plan = LayoutPlan::empty();
        assert_eq!(plan.total_concepts, 0);
        assert!(plan.layouts.is_empty());
    }

    #[test]
    fn test_generate_layout_empty() {
        let clustering = ClusteringResult::empty();
        let config = LayoutConfig::default();
        let plan = generate_layout(clustering, config);
        assert_eq!(plan.total_concepts, 0);
    }

    #[test]
    fn test_generate_layout_single_cluster() {
        use super::super::cluster::Cluster;

        let cluster = Cluster {
            id: 0,
            concepts: vec!["A:001".to_string(), "A:002".to_string()],
            avg_distance: 1.0,
            total_accesses: 100,
        };

        let clustering = ClusteringResult {
            clusters: vec![cluster],
            dendrogram: Vec::new(),
        };

        let config = LayoutConfig::default();
        let plan = generate_layout(clustering, config);

        assert_eq!(plan.total_concepts, 2);
        assert!(plan.get("A:001").is_some());
        assert!(plan.get("A:002").is_some());
    }

    #[test]
    fn test_layout_allocation_order() {
        use super::super::cluster::Cluster;

        let hot = Cluster {
            id: 0,
            concepts: vec!["hot:001".to_string()],
            avg_distance: 0.0,
            total_accesses: 1000,
        };
        let cold = Cluster {
            id: 1,
            concepts: vec!["cold:001".to_string()],
            avg_distance: 0.0,
            total_accesses: 1,
        };

        let clustering = ClusteringResult {
            clusters: vec![cold, hot], // Out of order
            dendrogram: Vec::new(),
        };

        let config = LayoutConfig::default();
        let plan = generate_layout(clustering, config);

        let order = plan.allocation_order();
        // Hot should come before cold
        let hot_pos = order.iter().position(|&c| c == "hot:001").unwrap();
        let cold_pos = order.iter().position(|&c| c == "cold:001").unwrap();
        assert!(hot_pos < cold_pos, "Hot concepts should be allocated first");
    }

    #[test]
    fn test_layout_summary() {
        use super::super::cluster::Cluster;

        let hot = Cluster {
            id: 0,
            concepts: vec!["h1".to_string(), "h2".to_string()],
            avg_distance: 0.0,
            total_accesses: 100,
        };
        let warm = Cluster {
            id: 1,
            concepts: vec!["w1".to_string()],
            avg_distance: 0.0,
            total_accesses: 20,
        };
        let cold = Cluster {
            id: 2,
            concepts: vec!["c1".to_string(), "c2".to_string(), "c3".to_string()],
            avg_distance: 0.0,
            total_accesses: 1,
        };

        let clustering = ClusteringResult {
            clusters: vec![hot, warm, cold],
            dendrogram: Vec::new(),
        };

        let config = LayoutConfig::default();
        let plan = generate_layout(clustering, config);
        let summary = plan.summary();

        assert_eq!(summary.total, 6);
        // Distribution depends on thresholds
        assert!(summary.hot > 0);
    }
}
