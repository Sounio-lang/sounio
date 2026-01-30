//! Hierarchical Clustering
//!
//! Clusters concepts by semantic proximity using hierarchical agglomerative clustering.
//! We use a simple implementation rather than external crates for control.

use super::distance::DistanceMatrix;
use super::extract::ConceptUsage;

/// A cluster of concepts
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Cluster ID
    pub id: usize,
    /// Concepts in this cluster
    pub concepts: Vec<String>,
    /// Average intra-cluster distance
    pub avg_distance: f32,
    /// Total access count for concepts in this cluster
    pub total_accesses: u32,
}

impl Cluster {
    /// Create a singleton cluster
    pub fn singleton(id: usize, concept: String, accesses: u32) -> Self {
        Self {
            id,
            concepts: vec![concept],
            avg_distance: 0.0,
            total_accesses: accesses,
        }
    }

    /// Merge two clusters
    pub fn merge(id: usize, a: Cluster, b: Cluster, distance: f32) -> Self {
        let mut concepts = a.concepts;
        concepts.extend(b.concepts);
        let total_accesses = a.total_accesses + b.total_accesses;

        Self {
            id,
            concepts,
            avg_distance: distance,
            total_accesses,
        }
    }
}

/// Result of clustering
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Final clusters
    pub clusters: Vec<Cluster>,
    /// Dendrogram: (cluster_a, cluster_b, distance, new_cluster_size)
    pub dendrogram: Vec<(usize, usize, f32, usize)>,
}

impl ClusteringResult {
    /// Create an empty result
    pub fn empty() -> Self {
        Self {
            clusters: Vec::new(),
            dendrogram: Vec::new(),
        }
    }

    /// Get clusters sorted by total accesses (hottest first)
    pub fn sorted_by_hotness(&self) -> Vec<&Cluster> {
        let mut sorted: Vec<_> = self.clusters.iter().collect();
        sorted.sort_by(|a, b| b.total_accesses.cmp(&a.total_accesses));
        sorted
    }
}

/// Cluster concepts using hierarchical agglomerative clustering
pub fn cluster_concepts(
    usage: &ConceptUsage,
    distances: &DistanceMatrix,
    max_clusters: usize,
) -> ClusteringResult {
    if distances.is_empty() {
        return ClusteringResult::empty();
    }

    let n = distances.len();
    if n == 1 {
        let concept = distances.concepts[0].clone();
        let accesses = *usage.access_counts.get(&concept).unwrap_or(&0);
        return ClusteringResult {
            clusters: vec![Cluster::singleton(0, concept, accesses)],
            dendrogram: Vec::new(),
        };
    }

    // Initialize singleton clusters
    let mut clusters: Vec<Option<Cluster>> = distances
        .concepts
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let accesses = *usage.access_counts.get(c).unwrap_or(&0);
            Some(Cluster::singleton(i, c.clone(), accesses))
        })
        .collect();

    // Build weighted distance matrix (incorporating co-occurrence)
    let max_co = usage.max_co_occurrence();
    let mut dist_matrix = build_weighted_matrix(distances, usage, max_co);

    let mut dendrogram = Vec::new();
    let mut next_id = n;

    // Agglomerative clustering (complete linkage)
    while count_active(&clusters) > max_clusters.max(1) {
        // Find closest pair
        let (i, j, dist) = find_closest_pair(&clusters, &dist_matrix);
        if i == j {
            break; // No valid pairs left
        }

        // Merge clusters
        let cluster_i = clusters[i].take().unwrap();
        let cluster_j = clusters[j].take().unwrap();
        let new_size = cluster_i.concepts.len() + cluster_j.concepts.len();

        dendrogram.push((i, j, dist, new_size));

        let merged = Cluster::merge(next_id, cluster_i, cluster_j, dist);

        // Update distance matrix (complete linkage: max of distances)
        update_distances(&mut dist_matrix, i, j, &clusters);

        clusters[i] = Some(merged);
        next_id += 1;
    }

    // Collect final clusters
    let final_clusters: Vec<Cluster> = clusters.into_iter().flatten().collect();

    ClusteringResult {
        clusters: final_clusters,
        dendrogram,
    }
}

/// Count active (non-None) clusters
fn count_active(clusters: &[Option<Cluster>]) -> usize {
    clusters.iter().filter(|c| c.is_some()).count()
}

/// Build weighted distance matrix incorporating co-occurrence
fn build_weighted_matrix(
    distances: &DistanceMatrix,
    usage: &ConceptUsage,
    max_co: u32,
) -> Vec<Vec<f32>> {
    let n = distances.len();
    let mut matrix = vec![vec![f32::MAX; n]; n];

    for i in 0..n {
        matrix[i][i] = 0.0;
        for j in (i + 1)..n {
            let base_dist = distances.get(i, j);

            // Get co-occurrence
            let a = &distances.concepts[i];
            let b = &distances.concepts[j];
            let key = if a < b {
                (a.clone(), b.clone())
            } else {
                (b.clone(), a.clone())
            };
            let co_occ = *usage.co_occurrences.get(&key).unwrap_or(&0);

            // Weight: higher co-occurrence reduces effective distance
            let weight = if max_co > 0 {
                1.0 - (co_occ as f32 / max_co as f32) * 0.5
            } else {
                1.0
            };

            let weighted = base_dist as f32 * weight;
            matrix[i][j] = weighted;
            matrix[j][i] = weighted;
        }
    }

    matrix
}

/// Find the closest pair of active clusters
fn find_closest_pair(
    clusters: &[Option<Cluster>],
    dist_matrix: &[Vec<f32>],
) -> (usize, usize, f32) {
    let mut best_i = 0;
    let mut best_j = 0;
    let mut best_dist = f32::MAX;

    for i in 0..clusters.len() {
        if clusters[i].is_none() {
            continue;
        }
        for j in (i + 1)..clusters.len() {
            if clusters[j].is_none() {
                continue;
            }
            if dist_matrix[i][j] < best_dist {
                best_dist = dist_matrix[i][j];
                best_i = i;
                best_j = j;
            }
        }
    }

    (best_i, best_j, best_dist)
}

/// Update distances after merging clusters i and j into i
fn update_distances(
    dist_matrix: &mut [Vec<f32>],
    merged_into: usize,
    merged_from: usize,
    clusters: &[Option<Cluster>],
) {
    let n = dist_matrix.len();

    for k in 0..n {
        if k == merged_into || k == merged_from || clusters[k].is_none() {
            continue;
        }

        // Complete linkage: use max distance
        let dist = dist_matrix[merged_into][k].max(dist_matrix[merged_from][k]);
        dist_matrix[merged_into][k] = dist;
        dist_matrix[k][merged_into] = dist;
    }

    // Mark merged_from as inactive
    for k in 0..n {
        dist_matrix[merged_from][k] = f32::MAX;
        dist_matrix[k][merged_from] = f32::MAX;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::native::NativeOntology;

    #[test]
    fn test_singleton_cluster() {
        let cluster = Cluster::singleton(0, "CHEBI:15365".to_string(), 5);
        assert_eq!(cluster.concepts.len(), 1);
        assert_eq!(cluster.total_accesses, 5);
        assert_eq!(cluster.avg_distance, 0.0);
    }

    #[test]
    fn test_cluster_merge() {
        let a = Cluster::singleton(0, "A:001".to_string(), 5);
        let b = Cluster::singleton(1, "B:001".to_string(), 3);
        let merged = Cluster::merge(2, a, b, 2.5);

        assert_eq!(merged.concepts.len(), 2);
        assert_eq!(merged.total_accesses, 8);
        assert_eq!(merged.avg_distance, 2.5);
    }

    #[test]
    fn test_clustering_empty() {
        let usage = ConceptUsage::new();
        let ontology = NativeOntology::empty("test");
        let distances = DistanceMatrix::build(&[], &ontology);

        let result = cluster_concepts(&usage, &distances, 3);
        assert!(result.clusters.is_empty());
    }

    #[test]
    fn test_clustering_single() {
        let mut usage = ConceptUsage::new();
        usage.record_access("A:001");

        let ontology = NativeOntology::empty("test");
        let concepts = vec!["A:001".to_string()];
        let distances = DistanceMatrix::build(&concepts, &ontology);

        let result = cluster_concepts(&usage, &distances, 3);
        assert_eq!(result.clusters.len(), 1);
    }

    #[test]
    fn test_clustering_multiple() {
        let mut usage = ConceptUsage::new();
        usage.record_scope(&["A:001", "A:002", "B:001"]);

        let ontology = NativeOntology::empty("test");
        let concepts = vec![
            "A:001".to_string(),
            "A:002".to_string(),
            "B:001".to_string(),
        ];
        let distances = DistanceMatrix::build(&concepts, &ontology);

        let result = cluster_concepts(&usage, &distances, 2);
        // Should cluster down to 2
        assert!(result.clusters.len() <= 2);
    }

    #[test]
    fn test_sorted_by_hotness() {
        let clusters = vec![
            Cluster::singleton(0, "cold".to_string(), 1),
            Cluster::singleton(1, "hot".to_string(), 100),
            Cluster::singleton(2, "warm".to_string(), 10),
        ];
        let result = ClusteringResult {
            clusters,
            dendrogram: Vec::new(),
        };

        let sorted = result.sorted_by_hotness();
        assert_eq!(sorted[0].concepts[0], "hot");
        assert_eq!(sorted[1].concepts[0], "warm");
        assert_eq!(sorted[2].concepts[0], "cold");
    }
}
