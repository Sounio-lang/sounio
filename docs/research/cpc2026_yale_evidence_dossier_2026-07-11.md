<!-- docs:meta
topic_id: repo.docs.research.cpc2026-yale-evidence-dossier-2026-07-11
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cpc2026-yale-evidence-dossier-2026-07-11
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CPC 2026 Yale evidence dossier

## Use this document for the poster

This is the claim-control handoff for the CPC 2026 Yale poster:

> Entropic Curvature in Hyperbolic Semantic Manifolds Indexes
> Psychopathology-Like Transitions

The poster combines two separate substrates. They answer separate questions and
must not be presented as cross-validation of one another.

1. **Study A, observed:** density-matched graph-level curvature in derived
   depression-text co-occurrence networks.
2. **Study B, exploratory:** deterministic no-training octonion state-space
   dynamics on the independent SWOW-EN association network.

No patient-level prediction, diagnosis, causal clinical inference, treatment
claim, or biomarker claim is established.

## Decision summary

| Surface | Status | Poster-safe interpretation |
|---|---|---|
| Study A density-matched curvature | Observed | Minimum-severity bucket was most negatively curved in 8/8 resampling seeds. Report as a robustness frequency. |
| Markov/Langevin abstract numbers | Superseded | The accepted abstract projections are not the observed pipeline values. Do not quote `d=1.84`, `47%`, or uniformly lower Hurst. |
| Frozen O-SSM paper-scale simulation | Exploratory, reproducible Python | `10,000 x 500` deterministic no-training simulation. Quote `d=11.65` and `d=-2.78` only with the model-construction caveat below. |
| Python same-subset audit | Recomputed in this audit | First 1,000 trajectories x 500 steps recover `d=11.6023` and `d=-2.7346`. |
| Legacy native Sounio n=1000 JSON | Historical, excluded from parity | Directionally similar, but component metrics differ by up to 21.1%; it is not numerical parity. |
| Current Sounio O-SSM source | Bounded native execution | Rebuilt Madaros compiles and runs `2 x 8`, producing JSON with `n=2` per regime. This is runtime evidence, not numerical parity. |
| omega parity receipt | Previously reproduced | `2.03e-10` absolute delta was recorded with omega 1.0.0-beta.4 after a parser correction. The omega binary was not available for re-verification on 2026-07-11. |
| Sounio epistemic receipts | Reverified | GUM `0.640000`, MC `0.643979`, exact order spread `2.044226`, all under `lean_single`. |

## Study A: observed density-controlled result

Source:

`hyperbolic-semantic-networks/results/unified/stratified_density_match_depression.json`

Parameters: exact-OT Ollivier-Ricci curvature with `alpha=0.5`, 1,000-node
subgraphs, 10 endpoint-degree bins, and 8 resampling seeds.

| Severity-text bucket | Mean matched ORC | Across-seed SD |
|---|---:|---:|
| Minimum | -0.174289 | 0.010564 |
| Mild | -0.090210 | 0.021840 |
| Moderate | -0.096295 | 0.030493 |
| Severe | -0.102075 | 0.028292 |

The minimum bucket is most hyperbolic in `8/8` seeds. Raw ordering is preserved
in `5/8` seeds.

Do **not** quote the edge-level Mann-Whitney p-values from the source artifact.
Network edges are dependent, so treating edges as independent replicates would
be pseudo-replication. The poster correctly reports the 8/8 directional
robustness frequency instead.

Also do not repeat the source JSON's phrase "conference biomarker." This is a
derived text-graph observation, not a clinical biomarker.

## Markov/Langevin correction ledger

Source:

`hyperbolic-semantic-networks/results/cpc2026/statistical_summary.json`

| Accepted-abstract projection | Observed pipeline result | Required correction |
|---|---|---|
| Cohen's `d=1.84` for C-ent variance | `d=-0.2659` | Normative variance is higher than anxious variance in the implemented model. |
| About 47% longer anxious residence | `+31.08%` | Direction survives; magnitude is smaller. |
| Pathology-like regimes have lower Hurst | Normative `0.7499`, anxious `0.8622`, ruminative `0.4803`, psychotic `0.8314` | Only the ruminative condition is lower. |
| SWOW-EN near eta about 2.5 | SWOW-EN eta about `0.0195`; reference critical point about `2.94` | SWOW-EN is deeply subcritical, not near the generic transition. |

The Markov results are real simulation outputs, but the accepted abstract
numbers are projections and must not be presented as confirmed findings.

## Study B: frozen O-SSM result

Sources:

- `hyperbolic-semantic-networks/code/cpc2026/ossm_reference_simulator.py`
- `hyperbolic-semantic-networks/results/cpc2026/ossm_simulation_summary.json`
- `hyperbolic-semantic-networks/results/cpc2026/ossm_statistical_summary.json`

The frozen simulation uses four octonion hidden units, 10,000 trajectories per
regime, 500 steps, seed `20260409`, fixed structured A/B/C/D parameters, and no
training.

### Poster-facing means

| Simulated regime | Hidden entropy-production rate | Mean associator norm |
|---|---:|---:|
| Normative | 0.00167601 | 0.27271810 |
| Anxious-like | 0.01476461 | 0.05174229 |
| Ruminative-like | 0.00053422 | 0.22752427 |
| Psychotic-like | 0.02121881 | 0.00620538 |

Python defines Cohen's d as candidate minus baseline. For anxious minus
normative:

- Hidden entropy-production rate: `d=11.650868`.
- Mean associator norm: `d=-2.781366`.

### Construction caveat that must remain visible

The four regimes share A/B/C/D. Regime identity is instantiated through
temperature, valence gain, and initialization. The anxious and ruminative
initial states are intentionally concentrated in lower-dimensional component
patterns; normative uses all eight octonion components. Therefore lower anxious
associator norm is partly a consequence of the constructed regime hypothesis,
not a discovery from unconstrained training or patient data.

Poster-safe wording:

> In a deterministic no-training octonion dynamical system whose synthetic
> regimes are instantiated by initialization, temperature, and valence gain,
> the anxious-like condition shows substantially higher hidden-state entropy
> production and lower associator norm than the normative condition.

Do not write:

> We discovered that anxious cognition naturally confines itself to a
> quaternionic subspace.

## Same-subset recomputation

Command:

```bash
uv run --with numpy python scripts/research/cpc2026_ossm_subset_audit.py \
  --scientific-repo /workspace/hyperbolic-semantic-networks \
  --native-json examples/cognitive_ossm/results/ossm_sounio_native_n1000.json
```

The independent implementation reads the compact node-feature and trajectory
CSVs directly. For the first 1,000 trajectories x 500 steps it obtains:

- Hidden entropy-production `d=11.602309`.
- Associator norm `d=-2.734612`.
- Normative mean entropy production `0.001676300`.
- Anxious mean entropy production `0.014814602`.
- Normative mean associator norm `0.274760979`.
- Anxious mean associator norm `0.051670581`.

This supports the frozen Python effect direction and scale. It does not rescue
the legacy native JSON as parity: relative discrepancies reach 11.6% for the
normative associator, 21.1% for the anxious associator, and 20.1% for anxious
C-ent variance.

## Sounio execution matrix

| Receipt | Engine | Current result |
|---|---|---|
| `order_spread_exact_n4.sio` | lean_single | `2.044226`, pass |
| `octonion_associator_gum_validation.sio` | lean_single | compiler variance `0.640000`, analytical `0.640000`, pass |
| `associator_variance_mc.sio` | lean_single | MC `0.643979` vs analytical `0.640000`, relative error `0.006218`, pass |
| `run_ossm_native_reference.sio` check | default Madaros | pass |
| `run_ossm_native_reference.sio` compile/run | rebuilt Madaros native-v2 | pass at `2 x 8`; associator output is zero and is not parity evidence |
| Bounded corrected-parser receipt | omega 1.0.0-beta.4 | previously reproduced `2.03e-10`; not reverified this session |

Run the evidence gate:

```bash
CPC2026_SCIENTIFIC_REPO=/workspace/hyperbolic-semantic-networks \
CPC2026_MADAROS_RAW_BIN=/tmp/rebuilt-madaros \
  bash scripts/ci/cpc2026_yale_evidence_gate.sh
```

The gate fails closed if the frozen O-SSM numbers drift, if the legacy JSON
loses its exclusion label, if rebuilt Madaros cannot compile and execute the
bounded source, if the bounded JSON is structurally invalid, or if a
lean_single receipt fails.

## Poster QA

The workspace-only poster was visually inspected at 1600 x 1200. The two-study
logic, evidence ledger, charts, and limitations are readable and internally
consistent. Before final print:

1. Replace the compiler label `Madares v0.80.0` with canonical `Madaros v0.80.0`.
2. Label the omega receipt as "previously reproduced" unless omega beta.4 is
   made available and rerun.
3. Keep "no patient-level or clinical prediction" in the first viewport.
4. Keep the legacy native n=1000 artifact excluded from parity.
5. Do not promote the open od256 frontier branch to a released capability.

The current QR decodes to:

`https://github.com/agourakis82/hyperbolic-semantic-networks#cpc-2026-extension`

## Minimal poster copy

### Study A headline

> Density-matched graph curvature separation retained its direction in 8/8
> resampling seeds.

### Study B headline

> Constructed no-training octonion dynamics separate synthetic regimes in
> hidden entropy production (`d=11.65`) and associator norm (`d=-2.78`).

### Required footer

> Separate substrates. Synthetic regimes are model parameterizations, not
> diagnosed cohorts. No patient-level validation, diagnosis, causality, or
> treatment inference.

## Confidence assessment

- **Ready to share:** Study A graph-level directional robustness with the
  pseudo-replication caveat; the frozen O-SSM simulation as an exploratory
  constructed-model result; the three reverified lean_single receipts.
- **Share only with explicit provenance:** the prior omega parity receipt.
- **Do not use as parity evidence:** the historical native n=1000 JSON.
- **Ready to claim narrowly:** rebuilt Madaros compiles and executes the O-SSM
  runner at `2 x 8`, producing structurally valid JSON.
- **Not ready to claim:** Madaros/Python numerical parity, native associator
  parity, clinical translation, natural emergence of quaternionic confinement,
  or released od256 hardware support.

## Resolved execution blocker

```text
Blocker-ID: BLK-20260711-CPC-OSSM-NATIVE-V2
Status: resolved
Severity: B1
Class: compiler-semantics
Owner: Codex
Lane: cpc2026-ossm-native-promotion
Worktree: /tmp/sounio-cpc-ossm-native-v2
Branch: fix/cpc2026-ossm-native-v2
Repro: CPC2026_MADAROS_RAW_BIN=/tmp/madaros-cpc-final6 bash scripts/ci/cpc2026_yale_evidence_gate.sh
Observed: rebuilt Madaros emits a runnable ELF; the bounded run exits zero and writes JSON with n=2 per regime
Expected: compile emits a runnable ELF with structurally valid bounded output
Acceptance-Gate: scripts/ci/cpc2026_yale_evidence_gate.sh requires compile plus bounded runtime success and explicitly sets parity_claim=false
Evidence-Level: E3
Evidence: CPC2026_SOUNIO_NATIVE_OK status=bounded_runtime trajectories=2 steps=8 parity_claim=false
Fallback-Path: none; Python owns paper-scale results and is not a silent Sounio fallback
Legacy-Kept: yes; historical n=100/n=1000 JSON retained with exclusion metadata
LLM-Offload: xAI and Z.AI math-review completed; publication-facing wording remains subject to M3 fan-out
Next-Action: treat zero native associator output as a separate numerical-parity investigation before making any native parity claim
```

The source-local scientific-notation parser defect found during external review
was fixed in this PR (`1.2e-3` is now parsed with an explicit base-10 exponent).
That repair removes an input-corruption bug but does not discharge the compiler
blocker above.
