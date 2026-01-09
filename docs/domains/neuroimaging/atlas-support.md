# Brain Atlas Support in Sounio

Sounio provides native support for standard brain parcellation atlases, enabling type-safe ROI-based analysis with automatic coordinate lookups and network assignments.

## Overview

The `fmri::atlas` module provides:

- Built-in atlas definitions (AAL, Schaefer, Harvard-Oxford, Glasser)
- MNI coordinate lookups for region centroids
- Network membership (Yeo 7/17 networks)
- Hemisphere and lobe classification
- Distance calculations between regions
- Network-based submatrix extraction

## Supported Atlases

### Atlas Types

```sio
use fmri::atlas::{AtlasType}

/// Standard atlas types
enum AtlasType {
    AAL,                // Automated Anatomical Labeling (116 regions)
    AAL3,               // AAL version 3 (166 regions)
    Schaefer100,        // Schaefer 100 parcels (7 networks)
    Schaefer200,        // Schaefer 200 parcels (7 networks)
    Schaefer400,        // Schaefer 400 parcels (7 networks)
    Schaefer1000,       // Schaefer 1000 parcels (17 networks)
    HarvardOxford,      // Harvard-Oxford cortical+subcortical (69 regions)
    DesikanKilliany,    // FreeSurfer DK atlas (68 regions)
    Destrieux,          // FreeSurfer Destrieux (148 regions)
    Gordon,             // Gordon 333 parcels
    Glasser,            // HCP-MMP1.0 (360 parcels)
    Brodmann,           // Brodmann areas (47 regions)
    Yeo7,               // Yeo 7 networks
    Yeo17,              // Yeo 17 networks
    Custom,             // User-defined atlas
}
```

### Atlas Comparison

| Atlas | Regions | Coverage | Networks | Best For |
|-------|---------|----------|----------|----------|
| **AAL** | 116 | Whole-brain | None | Clinical studies, lesion mapping |
| **Schaefer 100** | 100 | Cortical | Yeo 7 | Quick network analysis |
| **Schaefer 200** | 200 | Cortical | Yeo 7 | Balanced resolution |
| **Schaefer 400** | 400 | Cortical | Yeo 7 | High-resolution analysis |
| **Harvard-Oxford** | 69 | Whole-brain | None | Anatomical reference |
| **Glasser** | 360 | Cortical | Multimodal | HCP-compatible studies |
| **Gordon** | 333 | Cortical | Custom | Resting-state FC |

## Loading Atlases

### Factory Function

```sio
use fmri::atlas::{Atlas, AtlasType, create_atlas}

fn load_atlas_example() {
    // Create atlas by type
    let aal = create_atlas(AtlasType::AAL)
    let schaefer = create_atlas(AtlasType::Schaefer100)
    let ho = create_atlas(AtlasType::HarvardOxford)

    print("AAL regions: ", aal.n_regions, "\n")          // 116
    print("Schaefer regions: ", schaefer.n_regions, "\n") // 100
    print("H-O regions: ", ho.n_regions, "\n")           // 69
}
```

### Specialized Constructors

```sio
use fmri::atlas::{atlas_aal, atlas_schaefer100, atlas_schaefer400, atlas_harvard_oxford}

fn specialized_loaders() {
    let aal = atlas_aal()
    let schaefer100 = atlas_schaefer100()
    let schaefer400 = atlas_schaefer400()
    let harvard_oxford = atlas_harvard_oxford()
}
```

## Atlas Structure

### Region Definition

```sio
use fmri::atlas::{AtlasRegion, Hemisphere, Lobe, Network7, Network17}

/// Single region/parcel definition
struct AtlasRegion {
    // Identity
    index: i32,                 // 1-based region index
    label: [i8; 64],            // Region name (e.g., "Precentral_L")
    abbreviation: [i8; 16],     // Short name (e.g., "PreCG.L")

    // Spatial properties
    hemisphere: Hemisphere,     // Left, Right, Bilateral, Subcortical
    lobe: Lobe,                 // Frontal, Parietal, Temporal, etc.

    // MNI coordinates (centroid)
    mni_x: f64,
    mni_y: f64,
    mni_z: f64,

    // Volume
    volume_mm3: f64,
    n_voxels: i64,

    // Network membership
    network_7: Network7,        // Yeo 7-network assignment
    network_17: Network17,      // Yeo 17-network assignment

    // Connectivity profile (optional)
    mean_connectivity: f64,
    hub_score: f64,
}
```

### Hemisphere Classification

```sio
use fmri::atlas::{Hemisphere}

enum Hemisphere {
    Left,
    Right,
    Bilateral,      // Midline structures
    Subcortical,    // Deep brain structures
}
```

### Lobe Classification

```sio
use fmri::atlas::{Lobe}

enum Lobe {
    Frontal,
    Parietal,
    Temporal,
    Occipital,
    Limbic,
    Insular,
    Subcortical,
    Cerebellar,
    Brainstem,
    Unknown,
}
```

### Network Assignments

```sio
use fmri::atlas::{Network7, Network17}

/// Yeo 7 Networks
enum Network7 {
    Visual,
    Somatomotor,
    DorsalAttention,
    VentralAttention,
    Limbic,
    Frontoparietal,     // Control network
    Default,            // DMN
    Subcortical,
    Unknown,
}

/// Yeo 17 Networks
enum Network17 {
    VisualA, VisualB,
    SomatomotorA, SomatomotorB,
    DorsalAttentionA, DorsalAttentionB,
    VentralAttentionA, VentralAttentionB,
    LimbicA, LimbicB,
    FrontoparietalA, FrontoparietalB, FrontoparietalC,
    DefaultA, DefaultB, DefaultC,
    TemporalParietal,
    Subcortical,
    Unknown,
}
```

## Working with Regions

### Accessing Regions

```sio
use fmri::atlas::{Atlas, atlas_aal, atlas_get_region, atlas_find_region}

fn access_regions() {
    let atlas = atlas_aal()

    // Get region by index (1-based)
    let region = atlas_get_region(&atlas, 1)
    print("Region 1: ", region.label, "\n")  // "Precentral_L"
    print("MNI: (", region.mni_x, ", ", region.mni_y, ", ", region.mni_z, ")\n")
    print("Lobe: ", region.lobe, "\n")

    // Find region by label (returns 0 if not found)
    var search_label: [i8; 64] = [0; 64]
    // Copy "Hippocampus_L" to search_label
    let idx = atlas_find_region(&atlas, &search_label)
    if idx > 0 {
        print("Found Hippocampus_L at index ", idx, "\n")
    }
}
```

### Filtering by Hemisphere

```sio
use fmri::atlas::{Atlas, Hemisphere, atlas_get_hemisphere_regions}

fn get_hemisphere_regions(atlas: &Atlas) {
    var left_indices: [i32; 500] = [0; 500]
    var n_left: i64 = 0

    atlas_get_hemisphere_regions(atlas, Hemisphere::Left, &!left_indices, &!n_left)

    print("Left hemisphere regions: ", n_left, "\n")

    // List first 10
    var i: i64 = 0
    while i < 10 && i < n_left {
        let region = atlas_get_region(atlas, left_indices[i as usize])
        print("  ", left_indices[i as usize], ": MNI x=", region.mni_x, "\n")
        i = i + 1
    }
}
```

### Filtering by Lobe

```sio
use fmri::atlas::{Atlas, Lobe, atlas_get_lobe_regions}

fn get_frontal_regions(atlas: &Atlas) {
    var frontal_indices: [i32; 200] = [0; 200]
    var n_frontal: i64 = 0

    atlas_get_lobe_regions(atlas, Lobe::Frontal, &!frontal_indices, &!n_frontal)

    print("Frontal lobe regions: ", n_frontal, "\n")
}
```

### Filtering by Network

```sio
use fmri::atlas::{Atlas, Network7, atlas_schaefer100, atlas_get_network_regions}

fn get_dmn_regions() {
    let atlas = atlas_schaefer100()

    var dmn_indices: [i32; 200] = [0; 200]
    var n_dmn: i64 = 0

    atlas_get_network_regions(&atlas, Network7::Default, &!dmn_indices, &!n_dmn)

    print("Default Mode Network regions: ", n_dmn, "\n")

    // Print region details
    var i: i64 = 0
    while i < n_dmn {
        let region = atlas_get_region(&atlas, dmn_indices[i as usize])
        print("  ", dmn_indices[i as usize], ": ")
        print("MNI (", region.mni_x, ", ", region.mni_y, ", ", region.mni_z, ")\n")
        i = i + 1
    }
}
```

## Spatial Operations

### Distance Between Regions

```sio
use fmri::atlas::{Atlas, atlas_region_distance}

fn compute_distances(atlas: &Atlas) {
    // Euclidean distance between region centroids (mm)
    let dist = atlas_region_distance(atlas, 1, 2)  // Precentral_L to Precentral_R

    print("Inter-hemispheric distance: ", dist, " mm\n")

    // Build distance matrix
    var i: i64 = 1
    while i <= 10 {
        var j: i64 = i + 1
        while j <= 10 {
            let d = atlas_region_distance(atlas, i as i32, j as i32)
            print("Distance(", i, ",", j, "): ", d, " mm\n")
            j = j + 1
        }
        i = i + 1
    }
}
```

### Find Nearest Region to Coordinate

```sio
use fmri::atlas::{Atlas, atlas_nearest_region, atlas_get_region}

fn coordinate_lookup(atlas: &Atlas) {
    // Find nearest region to MNI coordinate
    let mni_x = -6.0
    let mni_y = -52.0
    let mni_z = 32.0

    let nearest_idx = atlas_nearest_region(atlas, mni_x, mni_y, mni_z)
    let region = atlas_get_region(atlas, nearest_idx)

    print("Nearest region to (", mni_x, ", ", mni_y, ", ", mni_z, "):\n")
    print("  Index: ", nearest_idx, "\n")
    print("  Label: ", region.label, "\n")
    print("  Centroid: (", region.mni_x, ", ", region.mni_y, ", ", region.mni_z, ")\n")
}
```

## Network-Based Analysis

### Extract Network Submatrix

```sio
use fmri::atlas::{
    Atlas, Network7, NetworkSubmatrix, network_submatrix_new,
    extract_network_connectivity
}

/// Network connectivity submatrix
struct NetworkSubmatrix {
    network: Network7,
    indices: [i32; 200],        // Region indices in this network
    n_regions: i64,
    matrix: [[f64; 200]; 200],  // Connectivity submatrix
    mean_within: f64,           // Mean within-network connectivity
    mean_between: f64,          // Mean between-network connectivity
}

fn analyze_dmn_connectivity(
    atlas: &Atlas,
    full_matrix: &[[f64; 500]; 500]
) {
    let dmn_sub = extract_network_connectivity(atlas, full_matrix, Network7::Default)

    print("DMN analysis:\n")
    print("  Regions: ", dmn_sub.n_regions, "\n")
    print("  Mean within-network FC: ", dmn_sub.mean_within, "\n")

    // Access submatrix elements
    var i: i64 = 0
    while i < dmn_sub.n_regions && i < 5 {
        var j: i64 = i + 1
        while j < dmn_sub.n_regions && j < 5 {
            print("  FC(", dmn_sub.indices[i as usize], ",",
                  dmn_sub.indices[j as usize], "): ",
                  dmn_sub.matrix[i as usize][j as usize], "\n")
            j = j + 1
        }
        i = i + 1
    }
}
```

### Between-Network Connectivity

```sio
use fmri::atlas::{Atlas, Network7, between_network_connectivity}

fn analyze_network_interactions(atlas: &Atlas, full_matrix: &[[f64; 500]; 500]) {
    // Compute mean connectivity between networks
    let dmn_fpn = between_network_connectivity(
        atlas, full_matrix, Network7::Default, Network7::Frontoparietal
    )
    let dmn_sal = between_network_connectivity(
        atlas, full_matrix, Network7::Default, Network7::VentralAttention
    )

    print("DMN-Frontoparietal connectivity: ", dmn_fpn, "\n")
    print("DMN-Salience connectivity: ", dmn_sal, "\n")

    // Build network-level matrix
    print("\nNetwork connectivity matrix:\n")
    print("        VIS   SOM   DAN   VAN   LIM   FPN   DMN\n")

    let networks = [
        Network7::Visual,
        Network7::Somatomotor,
        Network7::DorsalAttention,
        Network7::VentralAttention,
        Network7::Limbic,
        Network7::Frontoparietal,
        Network7::Default
    ]

    var i: i64 = 0
    while i < 7 {
        var j: i64 = 0
        while j < 7 {
            let fc = between_network_connectivity(
                atlas, full_matrix, networks[i as usize], networks[j as usize]
            )
            print("  ", fc)
            j = j + 1
        }
        print("\n")
        i = i + 1
    }
}
```

## Example: Complete Atlas-Based Analysis

```sio
use fmri::atlas::{
    Atlas, AtlasType, create_atlas, Network7,
    atlas_get_region, atlas_get_network_regions,
    atlas_nearest_region, atlas_region_distance,
    extract_network_connectivity, between_network_connectivity
}
use fmri::connectivity::{compute_fc}

fn complete_atlas_analysis(
    timeseries: [[f64; 200]; 100],  // 100 ROIs x 200 timepoints
    n_rois: i64,
    n_timepoints: i64
) {
    // 1. Load atlas
    let atlas = create_atlas(AtlasType::Schaefer100)
    print("Loaded ", atlas.n_regions, " region atlas\n")

    // 2. Build connectivity matrix
    var conn_matrix: [[f64; 500]; 500] = [[0.0; 500]; 500]

    var i: i64 = 0
    while i < n_rois {
        var j: i64 = i + 1
        while j < n_rois {
            var ts_i: [f64; 100] = [0.0; 100]
            var ts_j: [f64; 100] = [0.0; 100]

            // Copy first 100 timepoints
            var t: i64 = 0
            while t < 100 && t < n_timepoints {
                ts_i[t as usize] = timeseries[i as usize][t as usize]
                ts_j[t as usize] = timeseries[j as usize][t as usize]
                t = t + 1
            }

            let fc = compute_fc(ts_i, ts_j, 100)
            conn_matrix[i as usize][j as usize] = fc.r
            conn_matrix[j as usize][i as usize] = fc.r

            j = j + 1
        }
        conn_matrix[i as usize][i as usize] = 1.0
        i = i + 1
    }

    // 3. Analyze network-level connectivity
    print("\n=== Network Analysis ===\n")

    // Within-network connectivity for each network
    let networks = [
        (Network7::Visual, "Visual"),
        (Network7::Somatomotor, "Somatomotor"),
        (Network7::DorsalAttention, "Dorsal Attention"),
        (Network7::VentralAttention, "Ventral Attention"),
        (Network7::Limbic, "Limbic"),
        (Network7::Frontoparietal, "Frontoparietal"),
        (Network7::Default, "Default Mode")
    ]

    var n: i64 = 0
    while n < 7 {
        let net = networks[n as usize].0
        let name = networks[n as usize].1

        let sub = extract_network_connectivity(&atlas, &conn_matrix, net)
        print(name, " (", sub.n_regions, " regions): ")
        print("within-FC = ", sub.mean_within, "\n")

        n = n + 1
    }

    // 4. Key between-network interactions
    print("\n=== Key Network Interactions ===\n")

    // DMN-Frontoparietal anticorrelation (often negative in healthy controls)
    let dmn_fpn = between_network_connectivity(
        &atlas, &conn_matrix, Network7::Default, Network7::Frontoparietal
    )
    print("DMN-Frontoparietal: ", dmn_fpn, "\n")

    // DMN-Salience (ventral attention)
    let dmn_sal = between_network_connectivity(
        &atlas, &conn_matrix, Network7::Default, Network7::VentralAttention
    )
    print("DMN-Salience: ", dmn_sal, "\n")

    // Sensorimotor systems
    let vis_som = between_network_connectivity(
        &atlas, &conn_matrix, Network7::Visual, Network7::Somatomotor
    )
    print("Visual-Somatomotor: ", vis_som, "\n")

    // 5. Identify hub regions
    print("\n=== Hub Regions ===\n")

    // Calculate mean connectivity for each region
    var mean_fc: [f64; 100] = [0.0; 100]
    i = 0
    while i < n_rois {
        var sum: f64 = 0.0
        var j: i64 = 0
        while j < n_rois {
            if i != j {
                sum = sum + conn_matrix[i as usize][j as usize]
            }
            j = j + 1
        }
        mean_fc[i as usize] = sum / (n_rois - 1) as f64
        i = i + 1
    }

    // Find top 5 hubs
    var top_hubs: [i64; 5] = [0; 5]
    var top_fc: [f64; 5] = [0.0; 5]

    i = 0
    while i < n_rois {
        // Check if this region has higher FC than current top 5
        var pos: i64 = 4
        while pos >= 0 && mean_fc[i as usize] > top_fc[pos as usize] {
            pos = pos - 1
        }
        pos = pos + 1

        if pos < 5 {
            // Insert at position pos
            var k: i64 = 4
            while k > pos {
                top_hubs[k as usize] = top_hubs[(k-1) as usize]
                top_fc[k as usize] = top_fc[(k-1) as usize]
                k = k - 1
            }
            top_hubs[pos as usize] = i
            top_fc[pos as usize] = mean_fc[i as usize]
        }

        i = i + 1
    }

    // Print hubs
    i = 0
    while i < 5 {
        let region = atlas_get_region(&atlas, (top_hubs[i as usize] + 1) as i32)
        print("Hub ", i+1, ": Region ", top_hubs[i as usize]+1)
        print(" (Network: ", region.network_7, ")")
        print(" Mean FC = ", top_fc[i as usize], "\n")
        i = i + 1
    }
}
```

## AAL Atlas Details

The Automated Anatomical Labeling (AAL) atlas provides 116 anatomical regions:

| Index | Region | Lobe | Notes |
|-------|--------|------|-------|
| 1-2 | Precentral | Frontal | Primary motor cortex |
| 3-4 | Frontal_Sup | Frontal | Superior frontal gyrus |
| 5-6 | Frontal_Sup_Orb | Frontal | Orbital part |
| 7-8 | Frontal_Mid | Frontal | Middle frontal gyrus |
| ... | ... | ... | ... |
| 37-38 | Hippocampus | Limbic | Memory processing |
| 39-40 | Amygdala | Limbic | Emotion processing |
| 41-42 | Caudate | Subcortical | Basal ganglia |
| 43-44 | Putamen | Subcortical | Basal ganglia |
| ... | ... | ... | ... |

## Schaefer Atlas Details

The Schaefer parcellation provides functionally-defined parcels:

| Version | Parcels | Networks | Resolution |
|---------|---------|----------|------------|
| Schaefer100 | 100 | Yeo 7 | Coarse |
| Schaefer200 | 200 | Yeo 7 | Medium |
| Schaefer400 | 400 | Yeo 7 | Fine |
| Schaefer1000 | 1000 | Yeo 17 | Ultra-fine |

Network distribution (Schaefer100):

| Network | Regions | % Total |
|---------|---------|---------|
| Visual | 14 | 14% |
| Somatomotor | 14 | 14% |
| Dorsal Attention | 14 | 14% |
| Ventral Attention | 12 | 12% |
| Limbic | 10 | 10% |
| Frontoparietal | 14 | 14% |
| Default | 22 | 22% |

## References

1. Tzourio-Mazoyer N, et al. (2002). "Automated anatomical labeling of activations in SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject brain." *NeuroImage* 15(1):273-89.

2. Schaefer A, et al. (2018). "Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI." *Cereb Cortex* 28(9):3095-3114.

3. Yeo BTT, et al. (2011). "The organization of the human cerebral cortex estimated by intrinsic functional connectivity." *J Neurophysiol* 106(3):1125-65.

4. Glasser MF, et al. (2016). "A multi-modal parcellation of human cerebral cortex." *Nature* 536(7615):171-178.

5. Desikan RS, et al. (2006). "An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest." *NeuroImage* 31(3):968-80.
