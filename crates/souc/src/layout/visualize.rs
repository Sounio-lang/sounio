//! Layout Visualization - Day 39
//!
//! Generates visual representations of layout plans:
//! - ASCII diagrams for terminal output
//! - Mermaid diagrams for documentation

use super::plan::{LayoutPlan, MemoryRegion};

/// Generate a Mermaid flowchart diagram of the layout
pub fn generate_mermaid(layout: &LayoutPlan) -> String {
    let mut mermaid = String::new();

    mermaid.push_str("```mermaid\nflowchart TB\n");

    // Group clusters by region
    let hot_clusters: Vec<_> = layout
        .layouts
        .iter()
        .filter(|l| l.region == MemoryRegion::Hot)
        .map(|l| l.cluster_id)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let warm_clusters: Vec<_> = layout
        .layouts
        .iter()
        .filter(|l| l.region == MemoryRegion::Warm)
        .map(|l| l.cluster_id)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let cold_clusters: Vec<_> = layout
        .layouts
        .iter()
        .filter(|l| l.region == MemoryRegion::Cold)
        .map(|l| l.cluster_id)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Hot region subgraph
    if !hot_clusters.is_empty() {
        mermaid.push_str("    subgraph Hot_L1_L2 [\"🔥 Hot (L1/L2)\"]\n");
        for cluster_id in &hot_clusters {
            write_cluster_subgraph(&mut mermaid, layout, *cluster_id, "hot");
        }
        mermaid.push_str("    end\n");
        mermaid.push_str("    style Hot_L1_L2 fill:#ff6b6b,stroke:#c92a2a,color:#000\n");
    }

    // Warm region subgraph
    if !warm_clusters.is_empty() {
        mermaid.push_str("    subgraph Warm_L2_L3 [\"🌡️ Warm (L2/L3)\"]\n");
        for cluster_id in &warm_clusters {
            write_cluster_subgraph(&mut mermaid, layout, *cluster_id, "warm");
        }
        mermaid.push_str("    end\n");
        mermaid.push_str("    style Warm_L2_L3 fill:#ffd43b,stroke:#fab005,color:#000\n");
    }

    // Cold region subgraph
    if !cold_clusters.is_empty() {
        mermaid.push_str("    subgraph Cold_RAM [\"❄️ Cold (RAM)\"]\n");
        for cluster_id in &cold_clusters {
            write_cluster_subgraph(&mut mermaid, layout, *cluster_id, "cold");
        }
        mermaid.push_str("    end\n");
        mermaid.push_str("    style Cold_RAM fill:#69db7c,stroke:#2f9e44,color:#000\n");
    }

    mermaid.push_str("```\n");
    mermaid
}

fn write_cluster_subgraph(
    mermaid: &mut String,
    layout: &LayoutPlan,
    cluster_id: usize,
    prefix: &str,
) {
    let concepts: Vec<_> = layout
        .layouts
        .iter()
        .filter(|l| l.cluster_id == cluster_id)
        .collect();

    if concepts.is_empty() {
        return;
    }

    let cluster_name = format!("{}_{}", prefix, cluster_id);
    mermaid.push_str(&format!(
        "        subgraph {} [\"Cluster {}\"]\n",
        cluster_name, cluster_id
    ));

    // Show up to 5 concepts
    for concept_layout in concepts.iter().take(5) {
        let node_id = sanitize_node_id(&concept_layout.concept);
        let label = truncate_label(&concept_layout.concept, 20);
        mermaid.push_str(&format!("            {}[\"{}\"]\n", node_id, label));
    }

    if concepts.len() > 5 {
        mermaid.push_str(&format!(
            "            {}_more[\"... +{}\"]\n",
            cluster_name,
            concepts.len() - 5
        ));
    }

    mermaid.push_str("        end\n");
}

/// Generate an ASCII visualization for terminal output
pub fn generate_ascii(layout: &LayoutPlan) -> String {
    let mut output = String::new();

    output.push_str("┌─────────────────────────────────────────────────────────────┐\n");
    output.push_str("│                    LAYOUT VISUALIZATION                     │\n");
    output.push_str("└─────────────────────────────────────────────────────────────┘\n\n");

    for region in [MemoryRegion::Hot, MemoryRegion::Warm, MemoryRegion::Cold] {
        let (header, marker) = match region {
            MemoryRegion::Hot => ("HOT (L1/L2 Cache)", "█"),
            MemoryRegion::Warm => ("WARM (L2/L3 Cache)", "▓"),
            MemoryRegion::Cold => ("COLD (RAM)", "░"),
        };

        output.push_str(&format!("{} {}\n", marker.repeat(3), header));
        output.push_str("─────────────────────────────────────────────────────────────\n");

        // Get concepts in this region
        let region_concepts = layout.in_region(region);

        if region_concepts.is_empty() {
            output.push_str("  (empty)\n");
        } else {
            // Group by cluster
            let mut clusters: std::collections::HashMap<usize, Vec<&str>> =
                std::collections::HashMap::new();

            for concept in region_concepts {
                if let Some(layout_info) = layout.get(concept) {
                    clusters
                        .entry(layout_info.cluster_id)
                        .or_default()
                        .push(concept);
                }
            }

            for (cluster_id, concepts) in clusters {
                let concepts_str: Vec<&str> = concepts.iter().take(4).copied().collect();
                let suffix = if concepts.len() > 4 {
                    format!(" (+{})", concepts.len() - 4)
                } else {
                    String::new()
                };

                output.push_str(&format!(
                    "{} Cluster {}: [{}]{}\n",
                    marker,
                    cluster_id,
                    concepts_str.join(", "),
                    suffix
                ));
            }
        }

        output.push('\n');
    }

    // Summary
    let summary = layout.summary();
    output.push_str("─────────────────────────────────────────────────────────────\n");
    output.push_str(&format!(
        "Total: {} concepts | Hot: {} | Warm: {} | Cold: {}\n",
        summary.total, summary.hot, summary.warm, summary.cold
    ));

    output
}

/// Generate a compact one-line summary
pub fn generate_summary(layout: &LayoutPlan) -> String {
    let summary = layout.summary();
    format!(
        "Layout: {} concepts → Hot: {} ({:.0}%), Warm: {} ({:.0}%), Cold: {} ({:.0}%)",
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

/// Generate detailed cluster information
pub fn generate_cluster_details(layout: &LayoutPlan) -> String {
    let mut output = String::new();

    // Collect unique clusters
    let mut cluster_info: std::collections::HashMap<usize, (MemoryRegion, Vec<&str>)> =
        std::collections::HashMap::new();

    for concept_layout in &layout.layouts {
        cluster_info
            .entry(concept_layout.cluster_id)
            .or_insert((concept_layout.region, Vec::new()))
            .1
            .push(&concept_layout.concept);
    }

    let mut sorted_clusters: Vec<_> = cluster_info.into_iter().collect();
    sorted_clusters.sort_by_key(|(id, _)| *id);

    for (cluster_id, (region, concepts)) in sorted_clusters {
        let region_icon = match region {
            MemoryRegion::Hot => "🔥",
            MemoryRegion::Warm => "🌡️",
            MemoryRegion::Cold => "❄️",
        };

        output.push_str(&format!(
            "\nCluster {} {} ({:?})\n",
            cluster_id, region_icon, region
        ));
        output.push_str(&format!("  Concepts: {}\n", concepts.len()));

        for (i, concept) in concepts.iter().enumerate().take(10) {
            output.push_str(&format!("    {}. {}\n", i + 1, concept));
        }

        if concepts.len() > 10 {
            output.push_str(&format!("    ... and {} more\n", concepts.len() - 10));
        }
    }

    output
}

/// Generate a table view of the layout
pub fn generate_table(layout: &LayoutPlan) -> String {
    let mut output = String::new();

    output.push_str("┌────────────────────────────────┬──────────┬─────────┬───────┐\n");
    output.push_str("│ Concept                        │ Region   │ Cluster │ Order │\n");
    output.push_str("├────────────────────────────────┼──────────┼─────────┼───────┤\n");

    let mut sorted_layouts: Vec<_> = layout.layouts.iter().collect();
    sorted_layouts.sort_by(|a, b| {
        a.region
            .priority()
            .cmp(&b.region.priority())
            .then(a.cluster_id.cmp(&b.cluster_id))
            .then(a.order.cmp(&b.order))
    });

    for concept_layout in sorted_layouts.iter().take(50) {
        let region_str = match concept_layout.region {
            MemoryRegion::Hot => "Hot",
            MemoryRegion::Warm => "Warm",
            MemoryRegion::Cold => "Cold",
        };

        let concept_display = if concept_layout.concept.len() > 30 {
            format!("{}...", &concept_layout.concept[..27])
        } else {
            concept_layout.concept.clone()
        };

        output.push_str(&format!(
            "│ {:<30} │ {:<8} │ {:>7} │ {:>5} │\n",
            concept_display, region_str, concept_layout.cluster_id, concept_layout.order
        ));
    }

    if layout.layouts.len() > 50 {
        output.push_str(&format!(
            "│ ... and {} more                │          │         │       │\n",
            layout.layouts.len() - 50
        ));
    }

    output.push_str("└────────────────────────────────┴──────────┴─────────┴───────┘\n");

    output
}

/// Sanitize a string to be a valid Mermaid node ID
fn sanitize_node_id(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Truncate a label for display
fn truncate_label(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::cluster::{Cluster, ClusteringResult};
    use crate::layout::plan::generate_layout;

    fn test_layout() -> LayoutPlan {
        let clustering = ClusteringResult {
            clusters: vec![
                Cluster {
                    id: 0,
                    concepts: vec!["hot_a".to_string(), "hot_b".to_string()],
                    avg_distance: 1.0,
                    total_accesses: 100,
                },
                Cluster {
                    id: 1,
                    concepts: vec!["warm_a".to_string()],
                    avg_distance: 1.5,
                    total_accesses: 30,
                },
                Cluster {
                    id: 2,
                    concepts: vec![
                        "cold_a".to_string(),
                        "cold_b".to_string(),
                        "cold_c".to_string(),
                    ],
                    avg_distance: 2.0,
                    total_accesses: 5,
                },
            ],
            dendrogram: Vec::new(),
        };

        generate_layout(clustering, super::super::plan::LayoutConfig::default())
    }

    #[test]
    fn test_mermaid_generation() {
        let layout = test_layout();
        let mermaid = generate_mermaid(&layout);

        assert!(mermaid.contains("```mermaid"));
        assert!(mermaid.contains("flowchart TB"));
        assert!(mermaid.contains("```"));
    }

    #[test]
    fn test_ascii_generation() {
        let layout = test_layout();
        let ascii = generate_ascii(&layout);

        assert!(ascii.contains("LAYOUT VISUALIZATION"));
        assert!(ascii.contains("HOT"));
        assert!(ascii.contains("WARM"));
        assert!(ascii.contains("COLD"));
    }

    #[test]
    fn test_summary_generation() {
        let layout = test_layout();
        let summary = generate_summary(&layout);

        assert!(summary.contains("Layout:"));
        assert!(summary.contains("concepts"));
        assert!(summary.contains("Hot:"));
    }

    #[test]
    fn test_table_generation() {
        let layout = test_layout();
        let table = generate_table(&layout);

        assert!(table.contains("Concept"));
        assert!(table.contains("Region"));
        assert!(table.contains("Cluster"));
    }

    #[test]
    fn test_sanitize_node_id() {
        assert_eq!(sanitize_node_id("CHEBI:15365"), "CHEBI_15365");
        assert_eq!(sanitize_node_id("simple"), "simple");
        assert_eq!(sanitize_node_id("with spaces"), "with_spaces");
    }
}
