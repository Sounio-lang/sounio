# Brain Connectivity Analysis in Sounio

This guide covers functional connectivity analysis with full epistemic uncertainty tracking. Sounio provides correlation-based connectivity, phase synchronization measures, and comprehensive graph-theoretic network metrics.

## Overview

Brain connectivity analysis in Sounio includes:

- **Correlation-based connectivity**: Pearson correlation with Fisher-z confidence intervals
- **Phase synchronization**: PLV, PLI, wPLI for oscillatory coupling
- **Network metrics**: Centrality, clustering, modularity, small-world properties
- **Epistemic tracking**: Every metric carries uncertainty bounds

## Correlation-Based Functional Connectivity

### Basic Pearson Correlation

```sio
use fmri::connectivity::{pearson_corr, fisher_z, fisher_z_inv, fisher_z_se}

/// Compute Pearson correlation between two ROI timeseries
fn compute_correlation(roi1: [f64; 100], roi2: [f64; 100], n: i64) {
    let r = pearson_corr(roi1, roi2, n)
    print("Pearson r = ", r, "\n")
}
```

### Fisher-z Transformation

The Fisher-z transformation stabilizes variance for correlation coefficients:

```sio
use fmri::connectivity::{fisher_z, fisher_z_inv, fisher_z_se}

fn fisher_z_example(r: f64, n: i64) {
    // Transform correlation to z-score
    // z = 0.5 * ln((1+r)/(1-r)) = arctanh(r)
    let z = fisher_z(r)

    // Standard error of z depends only on sample size
    // SE(z) = 1 / sqrt(n - 3)
    let se = fisher_z_se(n)

    // 95% confidence interval in z-space
    let z_crit = 1.96
    let z_lower = z - z_crit * se
    let z_upper = z + z_crit * se

    // Transform back to correlation space
    let r_lower = fisher_z_inv(z_lower)
    let r_upper = fisher_z_inv(z_upper)

    print("r = ", r, " [", r_lower, ", ", r_upper, "]\n")
}
```

### Connectivity with Full Uncertainty

```sio
use fmri::connectivity::{FCResult, compute_fc}

/// Connectivity result with confidence interval
struct FCResult {
    r: f64,             // Pearson correlation
    z: f64,             // Fisher-z transformed
    se: f64,            // Standard error of z
    ci_lower: f64,      // 95% CI lower bound (r)
    ci_upper: f64,      // 95% CI upper bound (r)
}

fn full_connectivity_analysis(roi1: [f64; 100], roi2: [f64; 100], n: i64) {
    let fc = compute_fc(roi1, roi2, n)

    print("Functional connectivity results:\n")
    print("  Correlation: ", fc.r, "\n")
    print("  Fisher z: ", fc.z, "\n")
    print("  SE(z): ", fc.se, "\n")
    print("  95% CI: [", fc.ci_lower, ", ", fc.ci_upper, "]\n")

    // Width of CI indicates precision
    let ci_width = fc.ci_upper - fc.ci_lower
    if ci_width > 0.3 {
        print("  Warning: Wide confidence interval, consider more data\n")
    }
}
```

## Epistemic-Aware Connectivity

The `connectivity_epistemic` module provides enhanced tracking:

```sio
use fmri::connectivity_epistemic::{
    ConnectivityEdge, ConnectivityMatrix, EpistemicStatus,
    compute_fc_with_ci, status_from_uncertainty,
    inflate_uncertainty_for_motion
}

/// Epistemic status classification
enum EpistemicStatus {
    Verified,       // High confidence (relative uncertainty < 10%)
    Provisional,    // Moderate confidence (10-30% relative uncertainty)
    Uncertain,      // Low confidence (> 30% relative uncertainty)
}

fn epistemic_connectivity(roi1: [f64; 200], roi2: [f64; 200], n: i64) {
    let edge = compute_fc_with_ci(roi1, roi2, n)

    // Classify epistemic status based on relative uncertainty
    let status = status_from_uncertainty(edge.r, edge.uncertainty)

    match status {
        EpistemicStatus::Verified => {
            print("High confidence result - suitable for inference\n")
        },
        EpistemicStatus::Provisional => {
            print("Moderate confidence - interpret with caution\n")
        },
        EpistemicStatus::Uncertain => {
            print("Low confidence - consider additional data\n")
        },
    }
}
```

### Motion-Adjusted Uncertainty

Data quality affects uncertainty estimates:

```sio
use fmri::connectivity_epistemic::{inflate_uncertainty_for_motion}

fn adjust_for_motion(
    base_uncertainty: f64,
    mean_fd: f64,           // Mean framewise displacement (mm)
    scrub_fraction: f64     // Fraction of volumes scrubbed
) -> f64 {
    // Uncertainty inflation formula:
    // adjusted = base * (1 + 2*mean_fd) * (1 + 0.5*scrub_fraction)
    let adjusted = inflate_uncertainty_for_motion(
        base_uncertainty,
        mean_fd,
        scrub_fraction
    )

    print("Base uncertainty: ", base_uncertainty, "\n")
    print("Motion-adjusted: ", adjusted, "\n")

    adjusted
}
```

## Connectivity Matrix

```sio
use fmri::connectivity_epistemic::{ConnectivityMatrix, connectivity_matrix_new}

/// Full connectivity matrix with uncertainty
struct ConnectivityMatrix {
    matrix: [[f64; 50]; 50],        // Correlation values
    uncertainty: [[f64; 50]; 50],   // Uncertainty per edge
    ci_lower: [[f64; 50]; 50],      // Lower CI bounds
    ci_upper: [[f64; 50]; 50],      // Upper CI bounds
    n_rois: i64,
    n_volumes_used: i64,
    mean_fd: f64,
}

fn build_connectivity_matrix(
    timeseries: [[f64; 200]; 50],   // 50 ROIs x 200 timepoints
    n_rois: i64,
    n_timepoints: i64
) -> ConnectivityMatrix {
    var conn = connectivity_matrix_new()
    conn.n_rois = n_rois
    conn.n_volumes_used = n_timepoints

    // Compute pairwise connectivity
    var i: i64 = 0
    while i < n_rois {
        var j: i64 = i + 1
        while j < n_rois {
            // Extract timeseries for this pair
            var ts_i: [f64; 200] = timeseries[i as usize]
            var ts_j: [f64; 200] = timeseries[j as usize]

            let edge = compute_fc_with_ci(ts_i, ts_j, n_timepoints)

            // Fill symmetric matrix
            conn.matrix[i as usize][j as usize] = edge.r
            conn.matrix[j as usize][i as usize] = edge.r

            conn.uncertainty[i as usize][j as usize] = edge.uncertainty
            conn.uncertainty[j as usize][i as usize] = edge.uncertainty

            conn.ci_lower[i as usize][j as usize] = edge.ci_lower
            conn.ci_lower[j as usize][i as usize] = edge.ci_lower

            conn.ci_upper[i as usize][j as usize] = edge.ci_upper
            conn.ci_upper[j as usize][i as usize] = edge.ci_upper

            j = j + 1
        }

        // Diagonal = 1 (self-correlation)
        conn.matrix[i as usize][i as usize] = 1.0
        conn.uncertainty[i as usize][i as usize] = 0.0

        i = i + 1
    }

    conn
}
```

## Phase Synchronization

Phase-based connectivity measures oscillatory coupling between brain regions.

### Phase-Locking Value (PLV)

```sio
use connectivity::phase::{
    phase_locking_value, plv_from_phases, instantaneous_phase
}

/// PLV measures consistency of phase difference
/// PLV = |mean(exp(j*(phi1 - phi2)))|
/// Range: 0 (no synchrony) to 1 (perfect synchrony)

fn compute_plv(signal1: [f64; 2048], signal2: [f64; 2048], n: i64) {
    let plv = phase_locking_value(signal1, signal2, n)

    print("Phase-locking value: ", plv, "\n")

    // Interpretation
    if plv > 0.8 {
        print("Strong phase synchronization\n")
    } else if plv > 0.4 {
        print("Moderate phase synchronization\n")
    } else {
        print("Weak or no synchronization\n")
    }
}

// Efficient computation from pre-extracted phases
fn plv_efficient(
    signals: [[f64; 2048]; 16],  // 16 channels
    n: i64
) -> [[f64; 16]; 16] {
    // Extract all phases first
    var phases: [[f64; 2048]; 16] = [[0.0; 2048]; 16]
    var ch: i64 = 0
    while ch < 16 {
        phases[ch as usize] = instantaneous_phase(signals[ch as usize], n)
        ch = ch + 1
    }

    // Compute pairwise PLV
    var plv_matrix: [[f64; 16]; 16] = [[0.0; 16]; 16]
    var i: i64 = 0
    while i < 16 {
        var j: i64 = i + 1
        while j < 16 {
            let plv = plv_from_phases(phases[i as usize], phases[j as usize], n)
            plv_matrix[i as usize][j as usize] = plv
            plv_matrix[j as usize][i as usize] = plv
            j = j + 1
        }
        plv_matrix[i as usize][i as usize] = 1.0
        i = i + 1
    }

    plv_matrix
}
```

### Phase Lag Index (PLI)

PLI is robust to volume conduction artifacts:

```sio
use connectivity::phase::{phase_lag_index, pli_from_phases}

/// PLI measures asymmetry of phase differences
/// PLI = |mean(sign(dphi))|
/// Robust to zero-lag (volume conduction) effects

fn compute_pli(signal1: [f64; 2048], signal2: [f64; 2048], n: i64) {
    let pli = phase_lag_index(signal1, signal2, n)

    print("Phase lag index: ", pli, "\n")

    // PLI = 0 indicates symmetric phase distribution (no consistent lag)
    // PLI = 1 indicates all phase differences have same sign
}
```

### Weighted Phase Lag Index (wPLI)

wPLI weights contributions by imaginary component magnitude:

```sio
use connectivity::phase::{weighted_phase_lag_index, debiased_wpli}

fn compute_wpli(signal1: [f64; 2048], signal2: [f64; 2048], n: i64) {
    // Standard wPLI
    let wpli = weighted_phase_lag_index(signal1, signal2, n)
    print("Weighted PLI: ", wpli, "\n")

    // Debiased wPLI - reduces positive bias for small samples
    let dwpli = debiased_wpli(signal1, signal2, n)
    print("Debiased wPLI: ", dwpli, "\n")
}
```

### Phase Connectivity with Significance

```sio
use fmri::connectivity_epistemic::{compute_plv_with_pvalue}

fn plv_with_stats(phase1: [f64; 200], phase2: [f64; 200], n: i64) {
    let edge = compute_plv_with_pvalue(phase1, phase2, n)

    print("PLV: ", edge.r, "\n")
    print("P-value (Rayleigh test): ", edge.p_value, "\n")
    print("95% CI: [", edge.ci_lower, ", ", edge.ci_upper, "]\n")

    if edge.p_value < 0.05 {
        print("Significant phase synchronization\n")
    }
}
```

## Network Metrics

The `connectivity::network_metrics` module provides graph-theoretic analysis with uncertainty propagation.

### Metric Value Structure

```sio
use connectivity::network_metrics::{MetricValue, metric_value_new}

/// Every metric includes uncertainty
struct MetricValue {
    value: f64,
    uncertainty: f64,
    ci_lower: f64,
    ci_upper: f64,
    p_value: f64,
    is_significant: bool,
}
```

### Centrality Measures

#### Degree Centrality

```sio
use connectivity::network_metrics::{
    degree_binary, strength_weighted, degree_with_uncertainty
}

fn compute_centrality(
    weighted: &[[f64; 500]; 500],
    uncertainty: &[[f64; 500]; 500],
    n: i64,
    threshold: f64
) {
    // Binary degree: count of suprathreshold connections
    var degrees: [i64; 500] = [0; 500]
    let binary = threshold_matrix(weighted, threshold, n)
    degree_binary(&binary, n, &!degrees)

    // Weighted degree (strength): sum of connection weights
    var strengths: [f64; 500] = [0.0; 500]
    strength_weighted(weighted, threshold, n, &!strengths)

    // Degree with uncertainty propagation
    let node = 0
    let deg_unc = degree_with_uncertainty(
        weighted, uncertainty, threshold, n, node
    )

    print("Node 0 degree: ", deg_unc.value, "\n")
    print("Uncertainty: ", deg_unc.uncertainty, "\n")
}
```

### Clustering Coefficient

```sio
use connectivity::network_metrics::{
    clustering_coefficient_node, clustering_coefficient_weighted, transitivity
}

fn compute_clustering(
    weighted: &[[f64; 500]; 500],
    binary: &[[bool; 500]; 500],
    n: i64,
    threshold: f64
) {
    // Local clustering (binary)
    let cc_binary = clustering_coefficient_node(binary, n, 0)

    // Local clustering (weighted) - uses geometric mean of triangle weights
    let cc_weighted = clustering_coefficient_weighted(weighted, threshold, n, 0)

    // Global transitivity (ratio of triangles to connected triples)
    let trans = transitivity(binary, n)

    print("Node 0 clustering coefficient: ", cc_weighted, "\n")
    print("Global transitivity: ", trans, "\n")
}
```

### Global and Local Efficiency

```sio
use connectivity::network_metrics::{
    floyd_warshall, characteristic_path_length, global_efficiency, local_efficiency_node
}

fn compute_efficiency(weighted: &[[f64; 500]; 500], n: i64, threshold: f64) {
    // Compute shortest paths (connection length = 1/weight)
    let distances = floyd_warshall(weighted, n)

    // Characteristic path length (average shortest path)
    let cpl = characteristic_path_length(&distances, n)

    // Global efficiency (average inverse path length)
    let ge = global_efficiency(&distances, n)

    // Local efficiency (efficiency of node's neighborhood)
    let le = local_efficiency_node(weighted, n, 0, threshold)

    print("Characteristic path length: ", cpl, "\n")
    print("Global efficiency: ", ge, "\n")
    print("Node 0 local efficiency: ", le, "\n")
}
```

### Modularity

```sio
use connectivity::network_metrics::{
    modularity_q, louvain_modularity
}

fn compute_modularity(weighted: &[[f64; 500]; 500], n: i64) {
    // Detect communities using Louvain algorithm
    var modules: [i32; 500] = [0; 500]
    let q = louvain_modularity(weighted, n, &!modules)

    print("Modularity Q: ", q, "\n")

    // Count modules
    var max_module: i32 = 0
    var i: i64 = 0
    while i < n {
        if modules[i as usize] > max_module {
            max_module = modules[i as usize]
        }
        i = i + 1
    }
    print("Number of modules: ", max_module + 1, "\n")

    // List nodes in each module
    var m: i32 = 0
    while m <= max_module {
        print("Module ", m, ": ")
        i = 0
        while i < n {
            if modules[i as usize] == m {
                print(i, " ")
            }
            i = i + 1
        }
        print("\n")
        m = m + 1
    }
}
```

### Small-World Properties

```sio
use connectivity::network_metrics::{
    small_world_sigma, small_world_omega
}

fn analyze_small_world(
    avg_clustering: f64,
    char_path_length: f64,
    n_nodes: i64,
    n_edges: i64
) {
    // Sigma (Humphries & Gurney 2008)
    // sigma = (C/C_rand) / (L/L_rand)
    // sigma > 1 indicates small-world organization
    let sigma = small_world_sigma(avg_clustering, char_path_length, n_nodes, n_edges)

    // Omega (Telesford et al. 2011)
    // omega = L_rand/L - C/C_lattice
    // omega near 0 indicates small-world
    // omega < 0 indicates more lattice-like
    // omega > 0 indicates more random
    let omega = small_world_omega(avg_clustering, char_path_length, n_nodes, n_edges)

    print("Small-world sigma: ", sigma, "\n")
    print("Small-world omega: ", omega, "\n")

    if sigma > 1.0 && omega > -0.5 && omega < 0.5 {
        print("Network exhibits small-world properties\n")
    }
}
```

### Rich-Club Organization

```sio
use connectivity::network_metrics::{rich_club_coefficient}

fn analyze_rich_club(
    degrees: &[i64; 500],
    weights: &[[f64; 500]; 500],
    n: i64
) {
    print("Rich-club analysis:\n")

    // Compute rich-club coefficient for different degree thresholds
    var k: i64 = 5
    while k < 30 {
        let rc = rich_club_coefficient(degrees, weights, n, k)
        print("  k=", k, ": phi=", rc, "\n")
        k = k + 5
    }

    // Rich-club coefficient > expected from random network
    // indicates rich-club organization
}
```

### Hub Classification

```sio
use connectivity::network_metrics::{
    NodeMetrics, participation_coefficient
}

fn classify_hubs(
    weighted: &[[f64; 500]; 500],
    modules: &[i32; 500],
    n: i64,
    threshold: f64
) {
    // Participation coefficient: between-module connectivity
    // P = 0: all connections within module
    // P = 1: connections evenly distributed across modules

    var i: i64 = 0
    while i < n {
        let pc = participation_coefficient(weighted, modules, n, i)

        // Hub classification (Guimera & Amaral 2005)
        // Provincial hubs: high degree, low participation (P < 0.3)
        // Connector hubs: high degree, high participation (P >= 0.3)

        if pc > 0.3 {
            print("Node ", i, ": Connector hub (P=", pc, ")\n")
        }

        i = i + 1
    }
}
```

### Complete Network Analysis

```sio
use connectivity::network_metrics::{
    GlobalMetrics, NodeMetrics, compute_global_metrics, compute_node_metrics
}

/// Global network metrics
struct GlobalMetrics {
    n_nodes: i64,
    n_edges: i64,
    density: MetricValue,
    char_path_length: MetricValue,
    global_efficiency: MetricValue,
    transitivity: MetricValue,
    avg_clustering: MetricValue,
    modularity: MetricValue,
    n_modules: i64,
    small_world_sigma: MetricValue,
    small_world_omega: MetricValue,
    assortativity: MetricValue,
}

fn full_network_analysis(
    weighted: &[[f64; 500]; 500],
    uncertainty: &[[f64; 500]; 500],
    n: i64,
    threshold: f64
) {
    // Compute all global metrics
    let global = compute_global_metrics(weighted, n, threshold)

    print("=== Global Network Metrics ===\n")
    print("Nodes: ", global.n_nodes, "\n")
    print("Edges: ", global.n_edges, "\n")
    print("Density: ", global.density.value, "\n")
    print("Clustering: ", global.avg_clustering.value, "\n")
    print("Path length: ", global.char_path_length.value, "\n")
    print("Efficiency: ", global.global_efficiency.value, "\n")
    print("Modularity: ", global.modularity.value, " (", global.n_modules, " modules)\n")
    print("Small-world sigma: ", global.small_world_sigma.value, "\n")

    // Compute node-level metrics
    var node_metrics: [NodeMetrics; 500] = [node_metrics_new(); 500]
    compute_node_metrics(weighted, uncertainty, n, threshold, &!node_metrics)

    print("\n=== Hub Nodes ===\n")
    var i: i64 = 0
    while i < n {
        if node_metrics[i as usize].is_hub {
            let nm = &node_metrics[i as usize]
            print("Node ", i, ":\n")
            print("  Degree: ", nm.degree.value, "\n")
            print("  Clustering: ", nm.clustering_coef.value, "\n")
            print("  Participation: ", nm.participation_coef.value, "\n")

            if nm.is_connector_hub {
                print("  Type: Connector Hub\n")
            } else if nm.is_provincial_hub {
                print("  Type: Provincial Hub\n")
            }
        }
        i = i + 1
    }
}
```

## Epistemic Considerations in Connectivity

### Sample Size and Uncertainty

Correlation reliability depends on sample size:

| Timepoints | Approx. 95% CI Width (for r=0.5) |
|------------|----------------------------------|
| 50 | +/- 0.24 |
| 100 | +/- 0.17 |
| 200 | +/- 0.12 |
| 500 | +/- 0.07 |

### Threshold Selection

Network thresholding affects metrics:

```sio
fn threshold_sensitivity(
    weighted: &[[f64; 500]; 500],
    n: i64
) {
    // Analyze metrics across threshold range
    var threshold = 0.1
    while threshold <= 0.5 {
        let global = compute_global_metrics(weighted, n, threshold)

        print("Threshold ", threshold, ":\n")
        print("  Density: ", global.density.value, "\n")
        print("  Clustering: ", global.avg_clustering.value, "\n")
        print("  Modularity: ", global.modularity.value, "\n")

        threshold = threshold + 0.1
    }
}
```

## References

1. Rubinov M, Sporns O. (2010). "Complex network measures of brain connectivity: Uses and interpretations." *NeuroImage* 52(3):1059-69.

2. Lachaux JP, et al. (1999). "Measuring phase synchrony in brain signals." *Human Brain Mapping* 8(4):194-208.

3. Stam CJ, et al. (2007). "Phase lag index: Assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources." *Human Brain Mapping* 28(11):1178-93.

4. Humphries MD, Gurney K. (2008). "Network 'small-world-ness': A quantitative method for determining canonical network equivalence." *PLoS ONE* 3(4):e0002051.

5. Guimera R, Amaral LAN. (2005). "Functional cartography of complex metabolic networks." *Nature* 433(7028):895-900.
