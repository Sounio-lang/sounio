//! A/B Comparison: Classic vs Attention Register Allocation
//!
//! Run with: cargo run --example ab_comparison
//!
//! This demonstrates the difference between allocation policies by
//! simulating spill decisions with varying epistemic metadata.

use sounio::sir::{
    ABComparisonResult, AllocationMetrics, AttentionConfig,
    EpistemicMetadata, compute_attention_score,
};

fn main() {
    println!("{}", "=".repeat(70));
    println!("A/B COMPARISON: Classic vs Attention Register Allocation");
    println!("{}", "=".repeat(70));
    println!();

    // Create test scenarios with different epistemic profiles
    let scenarios = vec![
        ("Uniform confidence (0.5)", create_scenario(100, 0.5, 0.5, false)),
        ("High confidence (0.9)", create_scenario(100, 0.9, 0.1, false)),
        ("Low confidence (0.1)", create_scenario(100, 0.1, 0.9, false)),
        ("Mixed confidence", create_mixed_scenario(100)),
        ("Provenance-heavy", create_provenance_scenario(100)),
    ];

    for (name, values) in scenarios {
        println!("{}", "-".repeat(70));
        println!("Scenario: {}", name);
        println!("{}", "-".repeat(70));

        // Simulate Classic allocation (spills furthest values)
        let classic_metrics = simulate_classic(&values);

        // Simulate Attention allocation (spills low-confidence values)
        let attention_metrics = simulate_attention(&values, &AttentionConfig::default());

        // Simulate Calibrated allocation (spills with BO-optimized weights)
        let calibrated_metrics = simulate_attention(&values, &AttentionConfig::calibrated());

        // Compute comparison
        let comparison = ABComparisonResult::compute(
            classic_metrics,
            attention_metrics,
            Some(calibrated_metrics),
        );

        println!("{}", comparison.summary());
        println!();
    }

    println!("{}", "=".repeat(70));
    println!("A/B Comparison Complete");
    println!("{}", "=".repeat(70));
}

/// Simulated value with epistemic metadata
#[derive(Clone)]
struct SimulatedValue {
    #[allow(dead_code)]
    id: u32,
    confidence: f64,
    uncertainty: f64,
    has_provenance: bool,
    use_density: f64,
    lifetime: usize,
}

/// Simulate Classic allocation (furthest next use)
fn simulate_classic(values: &[SimulatedValue]) -> AllocationMetrics {
    let mut metrics = AllocationMetrics::default();
    metrics.policy = "Classic".to_string();

    // Sort by lifetime (furthest = spill first)
    let mut sorted: Vec<_> = values.iter().cloned().collect();
    sorted.sort_by(|a, b| b.lifetime.cmp(&a.lifetime));

    // Simulate spilling 30% of values (register pressure)
    let spill_count = values.len() * 30 / 100;
    for v in sorted.iter().take(spill_count) {
        metrics.total_spills += 1;
        if v.confidence >= 0.7 {
            metrics.high_confidence_spills += 1;
        } else if v.confidence <= 0.3 {
            metrics.low_confidence_spills += 1;
        }
        if v.has_provenance {
            metrics.provenance_values_spilled += 1;
        }
    }

    metrics.provenance_values_total = values.iter().filter(|v| v.has_provenance).count();
    metrics
}

/// Simulate Attention allocation (epistemic-aware)
fn simulate_attention(values: &[SimulatedValue], config: &AttentionConfig) -> AllocationMetrics {
    let mut metrics = AllocationMetrics::default();
    metrics.policy = "Attention".to_string();

    // Compute attention scores
    let mut scored: Vec<_> = values.iter().map(|v| {
        let epistemic = EpistemicMetadata {
            confidence: Some(v.confidence),
            uncertainty: Some(v.uncertainty),
            has_provenance: v.has_provenance,
            provenance_depth: if v.has_provenance { 1 } else { 0 },
            source_op: None,
        };
        let score = compute_attention_score(
            v.use_density,
            false, // crosses_call
            v.lifetime as f64 / 100.0,
            &epistemic,
            config,
        );
        (v.clone(), score)
    }).collect();

    // Sort by attention score (LOWEST score = spill first)
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Record score statistics
    let scores: Vec<f64> = scored.iter().map(|(_, s)| *s).collect();
    metrics.min_attention_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    metrics.max_attention_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    metrics.avg_attention_score = scores.iter().sum::<f64>() / scores.len() as f64;

    // Simulate spilling 30% of values (register pressure)
    let spill_count = values.len() * 30 / 100;
    let spilled_scores: Vec<f64> = scored.iter().take(spill_count).map(|(_, s)| *s).collect();
    let kept_scores: Vec<f64> = scored.iter().skip(spill_count).map(|(_, s)| *s).collect();

    metrics.spilled_avg_score = if !spilled_scores.is_empty() {
        spilled_scores.iter().sum::<f64>() / spilled_scores.len() as f64
    } else { 0.0 };
    metrics.kept_avg_score = if !kept_scores.is_empty() {
        kept_scores.iter().sum::<f64>() / kept_scores.len() as f64
    } else { 0.0 };

    for (v, _) in scored.iter().take(spill_count) {
        metrics.total_spills += 1;
        if v.confidence >= 0.7 {
            metrics.high_confidence_spills += 1;
        } else if v.confidence <= 0.3 {
            metrics.low_confidence_spills += 1;
        }
        if v.has_provenance {
            metrics.provenance_values_spilled += 1;
        }
    }

    metrics.provenance_values_total = values.iter().filter(|v| v.has_provenance).count();
    metrics
}

/// Create scenario with uniform values
fn create_scenario(count: usize, confidence: f64, uncertainty: f64, has_prov: bool) -> Vec<SimulatedValue> {
    (0..count).map(|i| SimulatedValue {
        id: i as u32,
        confidence,
        uncertainty,
        has_provenance: has_prov,
        use_density: 0.5,
        lifetime: 10 + (i % 20),
    }).collect()
}

/// Create scenario with mixed confidence levels
fn create_mixed_scenario(count: usize) -> Vec<SimulatedValue> {
    (0..count).map(|i| {
        let (conf, unc) = match i % 3 {
            0 => (0.9, 0.1),  // High
            1 => (0.5, 0.5),  // Medium
            _ => (0.1, 0.9),  // Low
        };
        SimulatedValue {
            id: i as u32,
            confidence: conf,
            uncertainty: unc,
            has_provenance: i % 5 == 0,
            use_density: 0.3 + (i % 10) as f64 * 0.05,
            lifetime: 10 + (i % 30),
        }
    }).collect()
}

/// Create scenario with many provenance-tracked values
fn create_provenance_scenario(count: usize) -> Vec<SimulatedValue> {
    (0..count).map(|i| {
        let has_prov = i % 2 == 0;
        SimulatedValue {
            id: i as u32,
            confidence: if has_prov { 0.8 } else { 0.4 },
            uncertainty: if has_prov { 0.2 } else { 0.6 },
            has_provenance: has_prov,
            use_density: 0.4,
            lifetime: 15 + (i % 20),
        }
    }).collect()
}
