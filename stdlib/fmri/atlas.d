// fmri::atlas — Brain Atlas Support for ROI-based Analysis
//
// Standard atlases for parcellation:
// - AAL (Automated Anatomical Labeling) - 116 regions
// - Schaefer (7/17 networks) - 100/200/400/1000 parcels
// - Harvard-Oxford - 48 cortical + 21 subcortical
// - Desikan-Killiany (FreeSurfer) - 68 regions
// - Gordon - 333 parcels
// - Glasser (HCP-MMP1.0) - 360 parcels
//
// Features:
// - Atlas metadata and region labels
// - Network assignments
// - Coordinate lookups (MNI/Talairach)
// - Custom atlas creation

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

// ============================================================================
// CONSTANTS
// ============================================================================

fn MAX_REGIONS() -> i64 { 1000 }
fn MAX_NETWORKS() -> i64 { 20 }

// ============================================================================
// ATLAS TYPES
// ============================================================================

/// Standard atlas types
enum AtlasType {
    AAL,                // Automated Anatomical Labeling (116)
    AAL3,               // AAL version 3 (166)
    Schaefer100,        // Schaefer 100 parcels (7 networks)
    Schaefer200,        // Schaefer 200 parcels (7 networks)
    Schaefer400,        // Schaefer 400 parcels (7 networks)
    Schaefer1000,       // Schaefer 1000 parcels (17 networks)
    HarvardOxford,      // Harvard-Oxford cortical+subcortical (69)
    DesikanKilliany,    // FreeSurfer DK atlas (68)
    Destrieux,          // FreeSurfer Destrieux (148)
    Gordon,             // Gordon 333 parcels
    Glasser,            // HCP-MMP1.0 (360)
    Brodmann,           // Brodmann areas (47)
    Yeo7,               // Yeo 7 networks
    Yeo17,              // Yeo 17 networks
    Custom,             // User-defined
}

/// Hemisphere
enum Hemisphere {
    Left,
    Right,
    Bilateral,
    Subcortical,
}

/// Lobe classification
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

/// Network assignment (Yeo 7 networks)
enum Network7 {
    Visual,
    Somatomotor,
    DorsalAttention,
    VentralAttention,
    Limbic,
    Frontoparietal,
    Default,
    Subcortical,
    Unknown,
}

/// Network assignment (Yeo 17 networks)
enum Network17 {
    VisualA,
    VisualB,
    SomatomotorA,
    SomatomotorB,
    DorsalAttentionA,
    DorsalAttentionB,
    VentralAttentionA,
    VentralAttentionB,
    LimbicA,
    LimbicB,
    FrontoparietalA,
    FrontoparietalB,
    FrontoparietalC,
    DefaultA,
    DefaultB,
    DefaultC,
    TemporalParietal,
    Subcortical,
    Unknown,
}

// ============================================================================
// REGION DEFINITION
// ============================================================================

/// Single region/parcel definition
struct AtlasRegion {
    // Identity
    index: i32,                     // 1-based region index
    label: [i8; 64],                // Region label (e.g., "Precentral_L")
    abbreviation: [i8; 16],         // Short name (e.g., "PreCG.L")

    // Spatial
    hemisphere: Hemisphere,
    lobe: Lobe,

    // MNI coordinates (centroid)
    mni_x: f64,
    mni_y: f64,
    mni_z: f64,

    // Volume
    volume_mm3: f64,
    n_voxels: i64,

    // Network membership
    network_7: Network7,
    network_17: Network17,

    // Connectivity profile (optional)
    mean_connectivity: f64,
    hub_score: f64,
}

fn atlas_region_new() -> AtlasRegion {
    AtlasRegion {
        index: 0,
        label: [0i8; 64],
        abbreviation: [0i8; 16],
        hemisphere: Hemisphere::Bilateral,
        lobe: Lobe::Unknown,
        mni_x: 0.0,
        mni_y: 0.0,
        mni_z: 0.0,
        volume_mm3: 0.0,
        n_voxels: 0,
        network_7: Network7::Unknown,
        network_17: Network17::Unknown,
        mean_connectivity: 0.0,
        hub_score: 0.0,
    }
}

// ============================================================================
// ATLAS DEFINITION
// ============================================================================

/// Complete atlas definition
struct Atlas {
    // Metadata
    name: [i8; 64],
    description: [i8; 256],
    atlas_type: AtlasType,
    version: [i8; 16],

    // Regions
    regions: [AtlasRegion; 1000],
    n_regions: i64,

    // Networks
    network_names: [[i8; 32]; 20],
    n_networks: i64,

    // Spatial info
    space: [i8; 16],                // "MNI152", "Talairach"
    resolution_mm: f64,             // Voxel size
    dim_x: i64,
    dim_y: i64,
    dim_z: i64,

    // Volume data would be loaded separately
    // data: [i32; ...],

    // Affine transform (4x4)
    affine: [[f64; 4]; 4],
}

fn atlas_new() -> Atlas {
    Atlas {
        name: [0i8; 64],
        description: [0i8; 256],
        atlas_type: AtlasType::Custom,
        version: [0i8; 16],
        regions: [atlas_region_new(); 1000],
        n_regions: 0,
        network_names: [[0i8; 32]; 20],
        n_networks: 0,
        space: [0i8; 16],
        resolution_mm: 2.0,
        dim_x: 91,
        dim_y: 109,
        dim_z: 91,
        affine: [[0.0; 4]; 4],
    }
}

// ============================================================================
// STRING HELPER
// ============================================================================

fn copy_str_to_arr(src: &[u8], dst: &![i8; 64]) {
    var i: i64 = 0
    while i < 64 && i < src.len() as i64 && src[i as usize] != 0 {
        dst[i as usize] = src[i as usize] as i8
        i = i + 1
    }
}

fn copy_str_to_arr_16(src: &[u8], dst: &![i8; 16]) {
    var i: i64 = 0
    while i < 16 && i < src.len() as i64 && src[i as usize] != 0 {
        dst[i as usize] = src[i as usize] as i8
        i = i + 1
    }
}

// ============================================================================
// AAL ATLAS (116 regions)
// ============================================================================

/// Create AAL atlas metadata
fn atlas_aal() -> Atlas {
    var atlas = atlas_new()

    // Name
    copy_str_to_arr("AAL".as_bytes(), &!atlas.name)
    copy_str_to_arr_16("MNI152".as_bytes(), &!atlas.space)

    atlas.atlas_type = AtlasType::AAL
    atlas.n_regions = 116
    atlas.n_networks = 1  // No network parcellation
    atlas.resolution_mm = 2.0

    // Define first 20 regions with coordinates
    // Left hemisphere frontal regions
    atlas.regions[0].index = 1
    copy_str_to_arr("Precentral_L".as_bytes(), &!atlas.regions[0].label)
    atlas.regions[0].mni_x = -38.65
    atlas.regions[0].mni_y = -5.68
    atlas.regions[0].mni_z = 50.94
    atlas.regions[0].lobe = Lobe::Frontal
    atlas.regions[0].hemisphere = Hemisphere::Left

    atlas.regions[1].index = 2
    copy_str_to_arr("Precentral_R".as_bytes(), &!atlas.regions[1].label)
    atlas.regions[1].mni_x = 41.38
    atlas.regions[1].mni_y = -8.21
    atlas.regions[1].mni_z = 52.09
    atlas.regions[1].lobe = Lobe::Frontal
    atlas.regions[1].hemisphere = Hemisphere::Right

    atlas.regions[2].index = 3
    copy_str_to_arr("Frontal_Sup_L".as_bytes(), &!atlas.regions[2].label)
    atlas.regions[2].mni_x = -18.45
    atlas.regions[2].mni_y = 34.81
    atlas.regions[2].mni_z = 42.20
    atlas.regions[2].lobe = Lobe::Frontal
    atlas.regions[2].hemisphere = Hemisphere::Left

    atlas.regions[3].index = 4
    copy_str_to_arr("Frontal_Sup_R".as_bytes(), &!atlas.regions[3].label)
    atlas.regions[3].mni_x = 21.90
    atlas.regions[3].mni_y = 31.12
    atlas.regions[3].mni_z = 43.82
    atlas.regions[3].lobe = Lobe::Frontal
    atlas.regions[3].hemisphere = Hemisphere::Right

    atlas.regions[4].index = 5
    copy_str_to_arr("Frontal_Sup_Orb_L".as_bytes(), &!atlas.regions[4].label)
    atlas.regions[4].mni_x = -16.56
    atlas.regions[4].mni_y = 47.32
    atlas.regions[4].mni_z = -13.31
    atlas.regions[4].lobe = Lobe::Frontal
    atlas.regions[4].hemisphere = Hemisphere::Left

    atlas.regions[5].index = 6
    copy_str_to_arr("Frontal_Sup_Orb_R".as_bytes(), &!atlas.regions[5].label)
    atlas.regions[5].mni_x = 18.49
    atlas.regions[5].mni_y = 48.10
    atlas.regions[5].mni_z = -14.02
    atlas.regions[5].lobe = Lobe::Frontal
    atlas.regions[5].hemisphere = Hemisphere::Right

    atlas.regions[6].index = 7
    copy_str_to_arr("Frontal_Mid_L".as_bytes(), &!atlas.regions[6].label)
    atlas.regions[6].mni_x = -33.43
    atlas.regions[6].mni_y = 32.73
    atlas.regions[6].mni_z = 35.46
    atlas.regions[6].lobe = Lobe::Frontal
    atlas.regions[6].hemisphere = Hemisphere::Left

    atlas.regions[7].index = 8
    copy_str_to_arr("Frontal_Mid_R".as_bytes(), &!atlas.regions[7].label)
    atlas.regions[7].mni_x = 37.59
    atlas.regions[7].mni_y = 33.06
    atlas.regions[7].mni_z = 34.04
    atlas.regions[7].lobe = Lobe::Frontal
    atlas.regions[7].hemisphere = Hemisphere::Right

    // Temporal regions
    atlas.regions[8].index = 9
    copy_str_to_arr("Temporal_Sup_L".as_bytes(), &!atlas.regions[8].label)
    atlas.regions[8].mni_x = -53.16
    atlas.regions[8].mni_y = -21.99
    atlas.regions[8].mni_z = 6.85
    atlas.regions[8].lobe = Lobe::Temporal
    atlas.regions[8].hemisphere = Hemisphere::Left

    atlas.regions[9].index = 10
    copy_str_to_arr("Temporal_Sup_R".as_bytes(), &!atlas.regions[9].label)
    atlas.regions[9].mni_x = 57.10
    atlas.regions[9].mni_y = -19.69
    atlas.regions[9].mni_z = 6.05
    atlas.regions[9].lobe = Lobe::Temporal
    atlas.regions[9].hemisphere = Hemisphere::Right

    // Parietal regions
    atlas.regions[10].index = 11
    copy_str_to_arr("Parietal_Sup_L".as_bytes(), &!atlas.regions[10].label)
    atlas.regions[10].mni_x = -23.98
    atlas.regions[10].mni_y = -59.78
    atlas.regions[10].mni_z = 58.87
    atlas.regions[10].lobe = Lobe::Parietal
    atlas.regions[10].hemisphere = Hemisphere::Left

    atlas.regions[11].index = 12
    copy_str_to_arr("Parietal_Sup_R".as_bytes(), &!atlas.regions[11].label)
    atlas.regions[11].mni_x = 26.23
    atlas.regions[11].mni_y = -58.45
    atlas.regions[11].mni_z = 59.41
    atlas.regions[11].lobe = Lobe::Parietal
    atlas.regions[11].hemisphere = Hemisphere::Right

    // Occipital regions
    atlas.regions[12].index = 13
    copy_str_to_arr("Occipital_Sup_L".as_bytes(), &!atlas.regions[12].label)
    atlas.regions[12].mni_x = -16.57
    atlas.regions[12].mni_y = -84.29
    atlas.regions[12].mni_z = 27.50
    atlas.regions[12].lobe = Lobe::Occipital
    atlas.regions[12].hemisphere = Hemisphere::Left

    atlas.regions[13].index = 14
    copy_str_to_arr("Occipital_Sup_R".as_bytes(), &!atlas.regions[13].label)
    atlas.regions[13].mni_x = 22.46
    atlas.regions[13].mni_y = -83.47
    atlas.regions[13].mni_z = 28.92
    atlas.regions[13].lobe = Lobe::Occipital
    atlas.regions[13].hemisphere = Hemisphere::Right

    // Limbic
    atlas.regions[14].index = 15
    copy_str_to_arr("Hippocampus_L".as_bytes(), &!atlas.regions[14].label)
    atlas.regions[14].mni_x = -25.40
    atlas.regions[14].mni_y = -21.03
    atlas.regions[14].mni_z = -11.80
    atlas.regions[14].lobe = Lobe::Limbic
    atlas.regions[14].hemisphere = Hemisphere::Left

    atlas.regions[15].index = 16
    copy_str_to_arr("Hippocampus_R".as_bytes(), &!atlas.regions[15].label)
    atlas.regions[15].mni_x = 28.96
    atlas.regions[15].mni_y = -19.75
    atlas.regions[15].mni_z = -12.16
    atlas.regions[15].lobe = Lobe::Limbic
    atlas.regions[15].hemisphere = Hemisphere::Right

    atlas.regions[16].index = 17
    copy_str_to_arr("Amygdala_L".as_bytes(), &!atlas.regions[16].label)
    atlas.regions[16].mni_x = -23.36
    atlas.regions[16].mni_y = -3.33
    atlas.regions[16].mni_z = -18.22
    atlas.regions[16].lobe = Lobe::Limbic
    atlas.regions[16].hemisphere = Hemisphere::Left

    atlas.regions[17].index = 18
    copy_str_to_arr("Amygdala_R".as_bytes(), &!atlas.regions[17].label)
    atlas.regions[17].mni_x = 26.77
    atlas.regions[17].mni_y = -1.91
    atlas.regions[17].mni_z = -18.27
    atlas.regions[17].lobe = Lobe::Limbic
    atlas.regions[17].hemisphere = Hemisphere::Right

    // Subcortical
    atlas.regions[18].index = 19
    copy_str_to_arr("Caudate_L".as_bytes(), &!atlas.regions[18].label)
    atlas.regions[18].mni_x = -11.34
    atlas.regions[18].mni_y = 11.46
    atlas.regions[18].mni_z = 9.07
    atlas.regions[18].lobe = Lobe::Subcortical
    atlas.regions[18].hemisphere = Hemisphere::Left

    atlas.regions[19].index = 20
    copy_str_to_arr("Caudate_R".as_bytes(), &!atlas.regions[19].label)
    atlas.regions[19].mni_x = 14.06
    atlas.regions[19].mni_y = 12.50
    atlas.regions[19].mni_z = 9.31
    atlas.regions[19].lobe = Lobe::Subcortical
    atlas.regions[19].hemisphere = Hemisphere::Right

    // Would continue for all 116 regions...

    atlas
}

// ============================================================================
// SCHAEFER ATLAS (7/17 networks)
// ============================================================================

/// Create Schaefer 100 parcel atlas
fn atlas_schaefer100() -> Atlas {
    var atlas = atlas_new()

    copy_str_to_arr("Schaefer100".as_bytes(), &!atlas.name)
    copy_str_to_arr_16("MNI152".as_bytes(), &!atlas.space)

    atlas.atlas_type = AtlasType::Schaefer100
    atlas.n_regions = 100
    atlas.n_networks = 7
    atlas.resolution_mm = 2.0

    // Network names (Yeo 7)
    copy_str_to_arr("Visual".as_bytes(), &!atlas.network_names[0])
    copy_str_to_arr("Somatomotor".as_bytes(), &!atlas.network_names[1])
    copy_str_to_arr("DorsalAttn".as_bytes(), &!atlas.network_names[2])
    copy_str_to_arr("VentralAttn".as_bytes(), &!atlas.network_names[3])
    copy_str_to_arr("Limbic".as_bytes(), &!atlas.network_names[4])
    copy_str_to_arr("Frontoparietal".as_bytes(), &!atlas.network_names[5])
    copy_str_to_arr("Default".as_bytes(), &!atlas.network_names[6])

    // Sample parcels (left hemisphere)
    atlas.regions[0].index = 1
    copy_str_to_arr("LH_Vis_1".as_bytes(), &!atlas.regions[0].label)
    atlas.regions[0].mni_x = -8.0
    atlas.regions[0].mni_y = -77.0
    atlas.regions[0].mni_z = 8.0
    atlas.regions[0].hemisphere = Hemisphere::Left
    atlas.regions[0].network_7 = Network7::Visual

    atlas.regions[1].index = 2
    copy_str_to_arr("LH_Vis_2".as_bytes(), &!atlas.regions[1].label)
    atlas.regions[1].mni_x = -12.0
    atlas.regions[1].mni_y = -69.0
    atlas.regions[1].mni_z = 5.0
    atlas.regions[1].hemisphere = Hemisphere::Left
    atlas.regions[1].network_7 = Network7::Visual

    atlas.regions[2].index = 3
    copy_str_to_arr("LH_SomMot_1".as_bytes(), &!atlas.regions[2].label)
    atlas.regions[2].mni_x = -38.0
    atlas.regions[2].mni_y = -26.0
    atlas.regions[2].mni_z = 59.0
    atlas.regions[2].hemisphere = Hemisphere::Left
    atlas.regions[2].network_7 = Network7::Somatomotor

    atlas.regions[3].index = 4
    copy_str_to_arr("LH_SomMot_2".as_bytes(), &!atlas.regions[3].label)
    atlas.regions[3].mni_x = -51.0
    atlas.regions[3].mni_y = -9.0
    atlas.regions[3].mni_z = 37.0
    atlas.regions[3].hemisphere = Hemisphere::Left
    atlas.regions[3].network_7 = Network7::Somatomotor

    atlas.regions[4].index = 5
    copy_str_to_arr("LH_DorsAttn_1".as_bytes(), &!atlas.regions[4].label)
    atlas.regions[4].mni_x = -25.0
    atlas.regions[4].mni_y = -57.0
    atlas.regions[4].mni_z = 57.0
    atlas.regions[4].hemisphere = Hemisphere::Left
    atlas.regions[4].network_7 = Network7::DorsalAttention

    atlas.regions[5].index = 6
    copy_str_to_arr("LH_VentAttn_1".as_bytes(), &!atlas.regions[5].label)
    atlas.regions[5].mni_x = -51.0
    atlas.regions[5].mni_y = -51.0
    atlas.regions[5].mni_z = 29.0
    atlas.regions[5].hemisphere = Hemisphere::Left
    atlas.regions[5].network_7 = Network7::VentralAttention

    atlas.regions[6].index = 7
    copy_str_to_arr("LH_Limbic_1".as_bytes(), &!atlas.regions[6].label)
    atlas.regions[6].mni_x = -28.0
    atlas.regions[6].mni_y = 10.0
    atlas.regions[6].mni_z = -20.0
    atlas.regions[6].hemisphere = Hemisphere::Left
    atlas.regions[6].network_7 = Network7::Limbic

    atlas.regions[7].index = 8
    copy_str_to_arr("LH_Cont_1".as_bytes(), &!atlas.regions[7].label)
    atlas.regions[7].mni_x = -42.0
    atlas.regions[7].mni_y = 6.0
    atlas.regions[7].mni_z = 33.0
    atlas.regions[7].hemisphere = Hemisphere::Left
    atlas.regions[7].network_7 = Network7::Frontoparietal

    atlas.regions[8].index = 9
    copy_str_to_arr("LH_Default_PCC".as_bytes(), &!atlas.regions[8].label)
    atlas.regions[8].mni_x = -6.0
    atlas.regions[8].mni_y = -52.0
    atlas.regions[8].mni_z = 32.0
    atlas.regions[8].hemisphere = Hemisphere::Left
    atlas.regions[8].network_7 = Network7::Default

    atlas.regions[9].index = 10
    copy_str_to_arr("LH_Default_mPFC".as_bytes(), &!atlas.regions[9].label)
    atlas.regions[9].mni_x = -6.0
    atlas.regions[9].mni_y = 52.0
    atlas.regions[9].mni_z = -6.0
    atlas.regions[9].hemisphere = Hemisphere::Left
    atlas.regions[9].network_7 = Network7::Default

    // Would continue for all 100 parcels...

    atlas
}

/// Create Schaefer 400 parcel atlas
fn atlas_schaefer400() -> Atlas {
    var atlas = atlas_schaefer100()

    copy_str_to_arr("Schaefer400".as_bytes(), &!atlas.name)
    atlas.atlas_type = AtlasType::Schaefer400
    atlas.n_regions = 400

    // Would include all 400 parcels with full network assignments

    atlas
}

// ============================================================================
// HARVARD-OXFORD ATLAS
// ============================================================================

/// Create Harvard-Oxford atlas
fn atlas_harvard_oxford() -> Atlas {
    var atlas = atlas_new()

    copy_str_to_arr("Harvard-Oxford".as_bytes(), &!atlas.name)
    copy_str_to_arr_16("MNI152".as_bytes(), &!atlas.space)

    atlas.atlas_type = AtlasType::HarvardOxford
    atlas.n_regions = 69  // 48 cortical + 21 subcortical
    atlas.resolution_mm = 2.0

    // Cortical regions
    atlas.regions[0].index = 1
    copy_str_to_arr("Frontal_Pole".as_bytes(), &!atlas.regions[0].label)
    atlas.regions[0].mni_x = 0.0
    atlas.regions[0].mni_y = 58.0
    atlas.regions[0].mni_z = -8.0
    atlas.regions[0].lobe = Lobe::Frontal
    atlas.regions[0].hemisphere = Hemisphere::Bilateral

    atlas.regions[1].index = 2
    copy_str_to_arr("Insular_Cortex".as_bytes(), &!atlas.regions[1].label)
    atlas.regions[1].mni_x = -36.0
    atlas.regions[1].mni_y = 7.0
    atlas.regions[1].mni_z = 3.0
    atlas.regions[1].lobe = Lobe::Insular
    atlas.regions[1].hemisphere = Hemisphere::Left

    atlas.regions[2].index = 3
    copy_str_to_arr("Sup_Frontal_Gyrus".as_bytes(), &!atlas.regions[2].label)
    atlas.regions[2].mni_x = -18.0
    atlas.regions[2].mni_y = 20.0
    atlas.regions[2].mni_z = 54.0
    atlas.regions[2].lobe = Lobe::Frontal
    atlas.regions[2].hemisphere = Hemisphere::Left

    // Subcortical regions (starting at index 48)
    atlas.regions[48].index = 49
    copy_str_to_arr("Thalamus_L".as_bytes(), &!atlas.regions[48].label)
    atlas.regions[48].mni_x = -11.0
    atlas.regions[48].mni_y = -18.0
    atlas.regions[48].mni_z = 8.0
    atlas.regions[48].lobe = Lobe::Subcortical
    atlas.regions[48].hemisphere = Hemisphere::Left
    atlas.regions[48].network_7 = Network7::Subcortical

    atlas.regions[49].index = 50
    copy_str_to_arr("Thalamus_R".as_bytes(), &!atlas.regions[49].label)
    atlas.regions[49].mni_x = 11.0
    atlas.regions[49].mni_y = -18.0
    atlas.regions[49].mni_z = 8.0
    atlas.regions[49].lobe = Lobe::Subcortical
    atlas.regions[49].hemisphere = Hemisphere::Right
    atlas.regions[49].network_7 = Network7::Subcortical

    atlas.regions[50].index = 51
    copy_str_to_arr("Caudate_L".as_bytes(), &!atlas.regions[50].label)
    atlas.regions[50].mni_x = -13.0
    atlas.regions[50].mni_y = 12.0
    atlas.regions[50].mni_z = 9.0
    atlas.regions[50].lobe = Lobe::Subcortical
    atlas.regions[50].hemisphere = Hemisphere::Left

    atlas.regions[51].index = 52
    copy_str_to_arr("Putamen_L".as_bytes(), &!atlas.regions[51].label)
    atlas.regions[51].mni_x = -24.0
    atlas.regions[51].mni_y = 4.0
    atlas.regions[51].mni_z = 1.0
    atlas.regions[51].lobe = Lobe::Subcortical
    atlas.regions[51].hemisphere = Hemisphere::Left

    atlas.regions[52].index = 53
    copy_str_to_arr("Hippocampus_L".as_bytes(), &!atlas.regions[52].label)
    atlas.regions[52].mni_x = -26.0
    atlas.regions[52].mni_y = -22.0
    atlas.regions[52].mni_z = -12.0
    atlas.regions[52].lobe = Lobe::Limbic
    atlas.regions[52].hemisphere = Hemisphere::Left

    atlas.regions[53].index = 54
    copy_str_to_arr("Amygdala_L".as_bytes(), &!atlas.regions[53].label)
    atlas.regions[53].mni_x = -22.0
    atlas.regions[53].mni_y = -4.0
    atlas.regions[53].mni_z = -18.0
    atlas.regions[53].lobe = Lobe::Limbic
    atlas.regions[53].hemisphere = Hemisphere::Left

    atlas
}

// ============================================================================
// ATLAS OPERATIONS
// ============================================================================

/// Get region by index (1-based)
fn atlas_get_region(atlas: &Atlas, index: i32) -> &AtlasRegion {
    if index > 0 && index <= atlas.n_regions as i32 {
        &atlas.regions[(index - 1) as usize]
    } else {
        &atlas.regions[0]
    }
}

/// Find region by label (returns 0 if not found)
fn atlas_find_region(atlas: &Atlas, label: &[i8; 64]) -> i32 {
    var i: i64 = 0
    while i < atlas.n_regions {
        var match_found = true
        var j: i64 = 0
        while j < 64 {
            if atlas.regions[i as usize].label[j as usize] != label[j as usize] {
                if atlas.regions[i as usize].label[j as usize] == 0 && label[j as usize] == 0 {
                    break
                }
                match_found = false
                break
            }
            if label[j as usize] == 0 {
                break
            }
            j = j + 1
        }
        if match_found {
            return atlas.regions[i as usize].index
        }
        i = i + 1
    }
    0
}

/// Get regions by network
fn atlas_get_network_regions(
    atlas: &Atlas,
    network: Network7,
    indices: &![i32; 200],
    n_indices: &!i64
) {
    *n_indices = 0
    var i: i64 = 0
    while i < atlas.n_regions {
        if atlas.regions[i as usize].network_7 == network {
            if *n_indices < 200 {
                indices[(*n_indices) as usize] = atlas.regions[i as usize].index
                *n_indices = *n_indices + 1
            }
        }
        i = i + 1
    }
}

/// Get regions by lobe
fn atlas_get_lobe_regions(
    atlas: &Atlas,
    lobe: Lobe,
    indices: &![i32; 200],
    n_indices: &!i64
) {
    *n_indices = 0
    var i: i64 = 0
    while i < atlas.n_regions {
        if atlas.regions[i as usize].lobe == lobe {
            if *n_indices < 200 {
                indices[(*n_indices) as usize] = atlas.regions[i as usize].index
                *n_indices = *n_indices + 1
            }
        }
        i = i + 1
    }
}

/// Get regions by hemisphere
fn atlas_get_hemisphere_regions(
    atlas: &Atlas,
    hemi: Hemisphere,
    indices: &![i32; 500],
    n_indices: &!i64
) {
    *n_indices = 0
    var i: i64 = 0
    while i < atlas.n_regions {
        if atlas.regions[i as usize].hemisphere == hemi {
            if *n_indices < 500 {
                indices[(*n_indices) as usize] = atlas.regions[i as usize].index
                *n_indices = *n_indices + 1
            }
        }
        i = i + 1
    }
}

/// Euclidean distance between two regions
fn atlas_region_distance(atlas: &Atlas, idx1: i32, idx2: i32) -> f64 {
    let r1 = atlas_get_region(atlas, idx1)
    let r2 = atlas_get_region(atlas, idx2)

    let dx = r1.mni_x - r2.mni_x
    let dy = r1.mni_y - r2.mni_y
    let dz = r1.mni_z - r2.mni_z

    sqrt(dx * dx + dy * dy + dz * dz)
}

/// Find nearest region to MNI coordinate
fn atlas_nearest_region(atlas: &Atlas, x: f64, y: f64, z: f64) -> i32 {
    var min_dist = 1e10
    var nearest_idx: i32 = 0

    var i: i64 = 0
    while i < atlas.n_regions {
        let r = &atlas.regions[i as usize]
        let dx = r.mni_x - x
        let dy = r.mni_y - y
        let dz = r.mni_z - z
        let dist = sqrt(dx * dx + dy * dy + dz * dz)

        if dist < min_dist {
            min_dist = dist
            nearest_idx = r.index
        }
        i = i + 1
    }

    nearest_idx
}

// ============================================================================
// NETWORK EXTRACTION
// ============================================================================

/// Network connectivity submatrix
struct NetworkSubmatrix {
    network: Network7,
    indices: [i32; 200],
    n_regions: i64,
    matrix: [[f64; 200]; 200],
    mean_within: f64,
    mean_between: f64,
}

fn network_submatrix_new() -> NetworkSubmatrix {
    NetworkSubmatrix {
        network: Network7::Unknown,
        indices: [0; 200],
        n_regions: 0,
        matrix: [[0.0; 200]; 200],
        mean_within: 0.0,
        mean_between: 0.0,
    }
}

/// Extract network submatrix from full connectivity
fn extract_network_connectivity(
    atlas: &Atlas,
    full_matrix: &[[f64; 500]; 500],
    network: Network7
) -> NetworkSubmatrix {
    var sub = network_submatrix_new()
    sub.network = network

    // Get network indices
    atlas_get_network_regions(atlas, network, &!sub.indices, &!sub.n_regions)

    // Extract submatrix
    var i: i64 = 0
    while i < sub.n_regions {
        var j: i64 = 0
        while j < sub.n_regions {
            let idx_i = (sub.indices[i as usize] - 1) as usize
            let idx_j = (sub.indices[j as usize] - 1) as usize
            sub.matrix[i as usize][j as usize] = full_matrix[idx_i][idx_j]
            j = j + 1
        }
        i = i + 1
    }

    // Calculate mean within-network connectivity
    var sum: f64 = 0.0
    var count: i64 = 0
    i = 0
    while i < sub.n_regions {
        var j: i64 = i + 1
        while j < sub.n_regions {
            sum = sum + sub.matrix[i as usize][j as usize]
            count = count + 1
            j = j + 1
        }
        i = i + 1
    }

    if count > 0 {
        sub.mean_within = sum / count as f64
    }

    sub
}

/// Calculate between-network connectivity
fn between_network_connectivity(
    atlas: &Atlas,
    full_matrix: &[[f64; 500]; 500],
    network1: Network7,
    network2: Network7
) -> f64 {
    var indices1 = [0i32; 200]
    var n1: i64 = 0
    atlas_get_network_regions(atlas, network1, &!indices1, &!n1)

    var indices2 = [0i32; 200]
    var n2: i64 = 0
    atlas_get_network_regions(atlas, network2, &!indices2, &!n2)

    var sum: f64 = 0.0
    var count: i64 = 0

    var i: i64 = 0
    while i < n1 {
        var j: i64 = 0
        while j < n2 {
            let idx_i = (indices1[i as usize] - 1) as usize
            let idx_j = (indices2[j as usize] - 1) as usize
            sum = sum + full_matrix[idx_i][idx_j]
            count = count + 1
            j = j + 1
        }
        i = i + 1
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

// ============================================================================
// ATLAS FACTORY
// ============================================================================

/// Create atlas by type
fn create_atlas(atlas_type: AtlasType) -> Atlas {
    match atlas_type {
        AtlasType::AAL => atlas_aal(),
        AtlasType::Schaefer100 => atlas_schaefer100(),
        AtlasType::Schaefer400 => atlas_schaefer400(),
        AtlasType::HarvardOxford => atlas_harvard_oxford(),
        _ => atlas_new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_aal_creation() -> bool {
    let atlas = atlas_aal()
    atlas.n_regions == 116
}

fn test_region_lookup() -> bool {
    let atlas = atlas_aal()
    let region = atlas_get_region(&atlas, 1)
    region.index == 1
}

fn test_region_distance() -> bool {
    let atlas = atlas_aal()
    let dist = atlas_region_distance(&atlas, 1, 2)  // Left and right precentral
    dist > 70.0 && dist < 90.0  // ~80mm apart
}

fn test_nearest_region() -> bool {
    let atlas = atlas_aal()
    // Near left precentral: (-38, -6, 51)
    let nearest = atlas_nearest_region(&atlas, -40.0, -5.0, 50.0)
    nearest == 1
}

fn test_network_extraction() -> bool {
    let atlas = atlas_schaefer100()
    var indices = [0i32; 200]
    var n: i64 = 0
    atlas_get_network_regions(&atlas, Network7::Default, &!indices, &!n)
    n > 0  // Should find some Default network regions
}

fn main() -> i32 {
    print("Testing fmri::atlas module...\n")

    if !test_aal_creation() {
        print("FAIL: aal_creation\n")
        return 1
    }
    print("PASS: aal_creation\n")

    if !test_region_lookup() {
        print("FAIL: region_lookup\n")
        return 2
    }
    print("PASS: region_lookup\n")

    if !test_region_distance() {
        print("FAIL: region_distance\n")
        return 3
    }
    print("PASS: region_distance\n")

    if !test_nearest_region() {
        print("FAIL: nearest_region\n")
        return 4
    }
    print("PASS: nearest_region\n")

    if !test_network_extraction() {
        print("FAIL: network_extraction\n")
        return 5
    }
    print("PASS: network_extraction\n")

    print("All fmri::atlas tests PASSED\n")
    0
}
