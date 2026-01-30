//! Report Generation
//!
//! Generates human-readable reports for layout analysis results.

use super::cluster::ClusteringResult;
use super::instrument::{CacheStats, LayoutComparison};
use super::plan::{LayoutPlan, LayoutSummary, MemoryRegion};

/// Generate a complete layout analysis report
pub fn generate_report(
    plan: &LayoutPlan,
    clustering: &ClusteringResult,
    comparison: Option<&LayoutComparison>,
) -> String {
    let mut report = String::new();

    // Header
    report.push_str("# Layout Synthesis Report\n\n");

    // Summary
    report.push_str("## Summary\n\n");
    let summary = plan.summary();
    report.push_str(&format_summary(&summary));
    report.push('\n');

    // Clustering info
    report.push_str("## Clusters\n\n");
    report.push_str(&format_clusters(clustering));
    report.push('\n');

    // Region breakdown
    report.push_str("## Memory Regions\n\n");
    report.push_str(&format_regions(plan));
    report.push('\n');

    // Cache performance (if available)
    if let Some(comp) = comparison {
        report.push_str("## Cache Performance\n\n");
        report.push_str(&format_cache_comparison(comp));
        report.push('\n');

        // Hypothesis validation
        report.push_str("## Hypothesis Validation\n\n");
        report.push_str(&format_hypothesis(comp));
        report.push('\n');
    }

    // Allocation order
    report.push_str("## Recommended Allocation Order\n\n");
    report.push_str(&format_allocation_order(plan));

    report
}

fn format_summary(summary: &LayoutSummary) -> String {
    format!(
        "- Total concepts: {}\n- Hot: {} ({:.1}%)\n- Warm: {} ({:.1}%)\n- Cold: {} ({:.1}%)\n",
        summary.total,
        summary.hot,
        if summary.total > 0 {
            summary.hot as f64 / summary.total as f64 * 100.0
        } else {
            0.0
        },
        summary.warm,
        if summary.total > 0 {
            summary.warm as f64 / summary.total as f64 * 100.0
        } else {
            0.0
        },
        summary.cold,
        if summary.total > 0 {
            summary.cold as f64 / summary.total as f64 * 100.0
        } else {
            0.0
        },
    )
}

fn format_clusters(clustering: &ClusteringResult) -> String {
    let mut s = String::new();

    for (i, cluster) in clustering.clusters.iter().enumerate() {
        s.push_str(&format!(
            "### Cluster {} (accesses: {}, avg_dist: {:.2})\n",
            i, cluster.total_accesses, cluster.avg_distance
        ));

        for concept in &cluster.concepts {
            s.push_str(&format!("  - {}\n", concept));
        }
        s.push('\n');
    }

    if clustering.clusters.is_empty() {
        s.push_str("_No clusters generated._\n");
    }

    s
}

fn format_regions(plan: &LayoutPlan) -> String {
    let mut s = String::new();

    for region in [MemoryRegion::Hot, MemoryRegion::Warm, MemoryRegion::Cold] {
        let concepts = plan.in_region(region);
        let name = match region {
            MemoryRegion::Hot => "Hot (Stack/L1-L2)",
            MemoryRegion::Warm => "Warm (Arena/L2-L3)",
            MemoryRegion::Cold => "Cold (Heap/RAM)",
        };

        s.push_str(&format!("### {}\n", name));

        if concepts.is_empty() {
            s.push_str("  _empty_\n");
        } else {
            for concept in concepts {
                s.push_str(&format!("  - {}\n", concept));
            }
        }
        s.push('\n');
    }

    s
}

fn format_cache_stats(stats: &CacheStats, label: &str) -> String {
    format!(
        "**{}**:\n  - Accesses: {}\n  - Hits: {} ({:.1}%)\n  - Misses: {} ({:.1}%)\n",
        label,
        stats.accesses,
        stats.hits,
        stats.hit_rate(),
        stats.misses,
        stats.miss_rate(),
    )
}

fn format_cache_comparison(comparison: &LayoutComparison) -> String {
    let mut s = String::new();

    s.push_str(&format_cache_stats(
        &comparison.baseline,
        "Baseline (alphabetical)",
    ));
    s.push('\n');
    s.push_str(&format_cache_stats(
        &comparison.optimized,
        "Optimized (semantic)",
    ));
    s.push('\n');

    s.push_str(&format!(
        "**Improvement**: {:.1} percentage points ({:.1}% relative)\n",
        comparison.improvement,
        comparison.relative_improvement()
    ));

    s
}

fn format_hypothesis(comparison: &LayoutComparison) -> String {
    let mut s = String::new();

    s.push_str("> **Hypothesis**: Semantic clustering improves cache performance.\n\n");

    if comparison.is_improvement() {
        s.push_str("**Result: SUPPORTED**\n\n");
        s.push_str(&format!(
            "The semantic layout achieved a {:.1}% hit rate compared to {:.1}% for the baseline.\n",
            comparison.optimized.hit_rate(),
            comparison.baseline.hit_rate()
        ));
        s.push_str(&format!(
            "This represents a {:.1} percentage point improvement.\n",
            comparison.improvement
        ));

        if comparison.improvement >= 10.0 {
            s.push_str("\n*Significant improvement observed.*\n");
        } else if comparison.improvement >= 5.0 {
            s.push_str("\n*Moderate improvement observed.*\n");
        } else {
            s.push_str("\n*Marginal improvement observed.*\n");
        }
    } else if comparison.improvement < 0.0 {
        s.push_str("**Result: NOT SUPPORTED**\n\n");
        s.push_str(&format!(
            "The semantic layout achieved a {:.1}% hit rate compared to {:.1}% for the baseline.\n",
            comparison.optimized.hit_rate(),
            comparison.baseline.hit_rate()
        ));
        s.push_str(
            "The baseline performed better than the semantic layout for this access pattern.\n",
        );
        s.push_str("\n*Consider reviewing the clustering parameters or access patterns.*\n");
    } else {
        s.push_str("**Result: INCONCLUSIVE**\n\n");
        s.push_str("No significant difference observed between layouts.\n");
        s.push_str("\n*May need more data or different access patterns to validate.*\n");
    }

    s
}

fn format_allocation_order(plan: &LayoutPlan) -> String {
    let mut s = String::new();

    let order = plan.allocation_order();

    if order.is_empty() {
        s.push_str("_No concepts to allocate._\n");
        return s;
    }

    s.push_str("Concepts should be allocated in this order for optimal cache behavior:\n\n");
    s.push_str("```\n");

    for (i, concept) in order.iter().enumerate() {
        let region = plan
            .get(concept)
            .map(|l| match l.region {
                MemoryRegion::Hot => "HOT ",
                MemoryRegion::Warm => "WARM",
                MemoryRegion::Cold => "COLD",
            })
            .unwrap_or("????");

        s.push_str(&format!("{:3}. [{}] {}\n", i + 1, region, concept));
    }

    s.push_str("```\n");

    s
}

/// Generate a brief one-line summary
pub fn brief_summary(plan: &LayoutPlan, comparison: Option<&LayoutComparison>) -> String {
    let summary = plan.summary();

    let perf = if let Some(comp) = comparison {
        if comp.is_improvement() {
            format!(", +{:.1}% cache hit rate", comp.improvement)
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    format!(
        "{} concepts: {} hot, {} warm, {} cold{}",
        summary.total, summary.hot, summary.warm, summary.cold, perf
    )
}

#[cfg(test)]
mod tests {
    use super::super::cluster::Cluster;
    use super::super::cluster::ClusteringResult;
    use super::super::plan::LayoutConfig;
    use super::super::plan::generate_layout;
    use super::*;

    #[test]
    fn test_generate_report_empty() {
        let plan = LayoutPlan::empty();
        let clustering = ClusteringResult::empty();

        let report = generate_report(&plan, &clustering, None);

        assert!(report.contains("Layout Synthesis Report"));
        assert!(report.contains("Total concepts: 0"));
    }

    #[test]
    fn test_generate_report_with_data() {
        let cluster = Cluster {
            id: 0,
            concepts: vec!["CHEBI:15365".to_string(), "GO:0008150".to_string()],
            avg_distance: 2.5,
            total_accesses: 100,
        };

        let clustering = ClusteringResult {
            clusters: vec![cluster],
            dendrogram: Vec::new(),
        };

        let config = LayoutConfig::default();
        let plan = generate_layout(clustering.clone(), config);

        let report = generate_report(&plan, &clustering, None);

        assert!(report.contains("CHEBI:15365"));
        assert!(report.contains("GO:0008150"));
        assert!(report.contains("Cluster 0"));
    }

    #[test]
    fn test_format_hypothesis_supported() {
        let comparison = LayoutComparison {
            baseline: CacheStats {
                accesses: 100,
                hits: 50,
                misses: 50,
            },
            optimized: CacheStats {
                accesses: 100,
                hits: 70,
                misses: 30,
            },
            improvement: 20.0,
        };

        let hypothesis = format_hypothesis(&comparison);

        assert!(hypothesis.contains("SUPPORTED"));
        assert!(hypothesis.contains("70.0% hit rate"));
    }

    #[test]
    fn test_format_hypothesis_not_supported() {
        let comparison = LayoutComparison {
            baseline: CacheStats {
                accesses: 100,
                hits: 70,
                misses: 30,
            },
            optimized: CacheStats {
                accesses: 100,
                hits: 50,
                misses: 50,
            },
            improvement: -20.0,
        };

        let hypothesis = format_hypothesis(&comparison);

        assert!(hypothesis.contains("NOT SUPPORTED"));
    }

    #[test]
    fn test_brief_summary() {
        let cluster = Cluster {
            id: 0,
            concepts: vec!["A".to_string()],
            avg_distance: 0.0,
            total_accesses: 100,
        };

        let clustering = ClusteringResult {
            clusters: vec![cluster],
            dendrogram: Vec::new(),
        };

        let config = LayoutConfig::default();
        let plan = generate_layout(clustering, config);

        let brief = brief_summary(&plan, None);

        assert!(brief.contains("1 concepts"));
    }
}
