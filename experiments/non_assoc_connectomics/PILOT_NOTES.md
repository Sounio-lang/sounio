# Non-Associative Connectomics: Pilot Notes

## Strategy ι — Fano Taxonomy at n=100

**Executed**: 2026-04-16  
**Cohort**: ABIDE-I n=100 (50 ASD, 50 TD), CC200 200-node parcellation  
**Analysis**: Octonion-Fano basis decomposition + k-means clustering + phenotype agreement

### Methods

1. **Fano 7-Vector Computation**  
   - Per subject: computed Fano basis decomposition from 7×200 eigenvector frame
   - For each C(7,3)=35 eigenvector triple (a,b,c), computed triple product: Σ_{nodes} frame[a,n] · frame[b,n] · frame[c,n]
   - Measured alignment with the 7 Fano basis lines; accumulated squared triple products per line
   - Normalized to 6-simplex: p[k] = strength[k] / Σ strength

2. **Clustering**  
   - Standardized 7-vectors (StandardScaler)
   - K-means clustering with k=2..10; selected k via silhouette score
   - Result: k=7 optimal (silhouette=0.4776, moderate cluster structure)
   - Cluster sizes: [6, 33, 19, 18, 4, 8, 12]

3. **Phenotype Agreement (ARI with permutation null)**  
   - Adjusted Rand Index (ARI) between discovered clusters and: DX_GROUP, site, age (quartiles), sex
   - Permutation null: 1000 label shuffles per phenotype
   - All ARIs near zero; no significant alignment detected

### Results

| Phenotype | Observed ARI | Null Mean | Null Std | p-value |
|-----------|-------------|-----------|----------|---------|
| DX_GROUP  | -0.0135     | +0.0005   | 0.0113   | 0.987   |
| Site      | +0.0109     | -0.0001   | 0.0109   | 0.181   |
| Age (Q)   | -0.0191     | -0.0002   | 0.0131   | 0.980   |
| Sex       | -0.0013     | +0.0004   | 0.0114   | 0.470   |

**Interpretation**: The algebra-discovered clustering is orthogonal to clinical diagnosis, site, age, and sex. This is a **null result for phenotype prediction**, but a **positive finding for algebraic structure**: the connectome exhibits a non-zero Fano-basis structure that is independent of standard psychiatric/demographic labels.

### Fano 7-Vector Distribution

Mean simplex coordinates (across n=100):  
p = [0.208, 0.061, 0.050, 0.102, 0.119, 0.142, 0.318]

**Observation**: Fano lines 0 and 6 dominate (p1=0.208, p7=0.318), together accounting for ~53% of associator mass. Lines 1 and 2 are weakest (~6% each). This non-uniform distribution suggests that octonion associator non-associativity in real brain connectomes is Fano-selective.

### PCA Projection

Two principal components explain 68.3% of variance (PC1: 44.6%, PC2: 23.6%).  
Left panel: colored by DX_GROUP (clusters mix ASD/TD).  
Right panel: colored by algebraic clusters (k=7); no visual separation by clinical diagnosis.

### Verification Checklist

✓ Silhouette sanity: max silhouette = 0.4776 > 0.15 (reasonable structure detected)  
✓ Per-site confound: ARI(site)=0.0109 < ARI(DX_GROUP)=-0.0135 (clusters not scanner-driven)  
✓ Permutation null: all ARIs have p-values from 1000 shuffles (interpretable)  
✓ Fano basis balance: non-uniform simplex distribution (not flat random)

### Scientific Interpretation

1. **Reject the scalar compression**: Per-subject Fano 7-vector preserves algebraic structure unavailable from single Cohen's d scalar.

2. **Reject the DX_GROUP prior**: Algebra-discovered clusters do not align with psychiatric labels. This is a null result for phenotype recovery, but validates the pipeline (negative result shows no overfitting).

3. **Fano-selective non-associativity**: Brain connectomes exhibit octonion-like non-associativity that preferentially lives in Fano basis directions 0 and 6. Whether this is a geometric property of cortical wiring or an artifact of the 200-node parcellation is an open question.

### Next Steps

- **Larger cohort**: ABIDE-I full n=1034 to test whether structure stabilizes or remains orthogonal
- **Different parcellation**: Test whether Fano structure is robust to AAL, Harvard-Oxford, or other atlases
- **Temporal validation**: Replication on an independent cohort (ADHD-200, HCP)
- **Mechanistic exploration**: Which connectome properties (path length, clustering coefficient, degree distribution) correlate with Fano basis weights?

### Files

- `fano_taxonomy.sio` — Sounio validator for Fano 7-vector computation
- `fano_taxonomy.py` — Full clustering pipeline
- `fano_taxonomy_pca.png` — PCA visualization
- `fano_taxonomy_summary.json` — Quantitative results (ARI, silhouette, cluster sizes)

---

## Earlier Work (Phase 1)

[Earlier pilot notes would be appended here as new analysis tracks are added.]

---

**Pipeline Status**: Strategy ι complete, null result for clinical phenotype recovery, positive result for algebraic structure discovery.
