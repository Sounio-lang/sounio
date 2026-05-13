# Type Mapping: Knowledge Across the Triple Sounio Ecosystem

This document defines the three core representations of epistemic knowledge in the Sounio ecosystem and how to convert between them.

## Overview

The Sounio ecosystem handles epistemic knowledge (values with uncertainty) in three different contexts, each with slightly different semantic emphasis:

| Context | Language | Type | Focus |
|---------|----------|------|-------|
| **Python Integration** | Python | `Knowledge(value, epsilon, provenance)` | GUM standard uncertainty |
| **Shared Types** | Sounio | `Knowledge[T]` with `ε`, `prov` | Confidence-centric (0-1) |
| **Drug Discovery** | Sounio | `DrugKnowledge` with value/epsilon/prov_id | Prov indexing for PK/PD |

---

## 1. Python Representation

### Definition
```python
class Knowledge:
    value: float          # Central estimate of the quantity
    epsilon: float        # Standard uncertainty (k=1, one sigma)
    provenance: str       # Free-form label describing the measurement source
```

### Semantics
- **epsilon** = GUM standard uncertainty (BIPM/JCGM-100 definition)
- Represents measurement uncertainty in the same units as `value`
- `provenance` tracks source/operation chain for audit trails
- Implements full GUM propagation rules for arithmetic

### Example
```python
k = Knowledge(500.0, 2.5, "calibration_batch_2026_03")
# 500 ± 2.5 mg, relative uncertainty = 2.5/500 = 0.5%
```

### Key Properties
- `relative_uncertainty` = epsilon / |value|
- `confidence` = 1 - relative_uncertainty (clamped to [0,1])
- `is_reliable(threshold=0.05)` = True if rel_uncertainty < 5%

---

## 2. Sounio Shared Types Representation

### Definition
```sio
struct Knowledge[T] {
    value: T,
    ε: f64,                     // Confidence in [0.0, 1.0]
    prov: string,               // Provenance string
    metadata: KnowledgeMetadata,
}
```

### Semantics
- **ε** (epsilon) = confidence score in [0.0, 1.0], NOT standard uncertainty
  - ε = 1.0 → fully confident (certain)
  - ε = 0.5 → 50% confident
  - ε = 0.0 → no confidence
- `prov` mirrors Python provenance
- `metadata` adds timestamps, source IDs, validation status, units
- Generic type parameter `[T]` allows `Knowledge[f64]`, `Knowledge[i64]`, etc.

### Example
```sio
let dose: Knowledge[f64] = Knowledge[f64] {
    value: 500.0,
    ε: 0.95,                // 95% confident
    prov: "calibration_batch_2026_03",
    metadata: { ... }
}
```

### Type Aliases (Common Drug Discovery Cases)
```sio
type DrugConcentration = Knowledge[f64]  // mg/L
type Time = Knowledge[f64]               // hours
type Dose = Knowledge[f64]               // mg
type Probability = Knowledge[f64]        // 0.0-1.0
type EfficacyScore = Knowledge[f64]      // Higher is better
type ToxicityScore = Knowledge[f64]      // Lower is better
type TherapeuticIndex = Knowledge[f64]   // Ratio TD50/EC50
type BloodPressure = Knowledge[f64]      // mmHg
type LabValue = Knowledge[f64]           // Various units
```

---

## 3. Drug Discovery Sounio Representation

### Definition
```sio
struct DrugKnowledge {
    value: f64,
    epsilon: f64,
    prov_id: i64
}
```

### Semantics
- **epsilon** = multiplicative confidence factor (not standard uncertainty like Python)
  - Semantics: ε ≈ confidence, but used in multiplicative chains (e.g., dk_mul multiplies epsilons)
  - NOT a direct mapping to GUM standard uncertainty
- **prov_id** = integer index into a global provenance table (not a string)
  - Allows compact binary serialization and deduplication
  - Mapped to string provenance via external lookup table
- Optimized for tight PK/PD loops and vectorized operations
- Arithmetic functions: `dk_add`, `dk_sub`, `dk_mul`, `dk_div`, `dk_scale`

### Example
```sio
let dose = dk_new(500.0, 0.95, 30)
// 500.0 mg, confidence 0.95, provenance ID 30 (maps to "dosing_protocol_2026")
```

### Arithmetic Rules
```sio
fn dk_add(a: DrugKnowledge, b: DrugKnowledge) -> DrugKnowledge {
    DrugKnowledge { value: a.value + b.value, epsilon: a.epsilon * b.epsilon, prov_id: 100 }
}

fn dk_mul(a: DrugKnowledge, b: DrugKnowledge) -> DrugKnowledge {
    DrugKnowledge { value: a.value * b.value, epsilon: a.epsilon * b.epsilon, prov_id: 100 }
}
```

---

## 4. Semantic Differences: epsilon Interpretation

The **critical difference** is how `epsilon` is interpreted:

| Aspect | Python Knowledge | Sounio Knowledge[T] | DrugKnowledge |
|--------|------------------|-------------------|-----------------|
| **epsilon semantics** | GUM standard uncertainty (units: same as value) | Confidence score ∈ [0,1] | Multiplicative confidence factor |
| **Interpretation** | σ (one sigma) | Probability / assurance | ε₁ * ε₂ * ... chain |
| **Range** | [0, ∞) | [0, 1] (typically) | [0, 1] (typically) |
| **Propagation** | Quadrature: sqrt(ε₁² + ε₂²) | Geometric mean or product | Product: ε₁ * ε₂ |
| **Provenance** | String ("sensor_123") | String with metadata | Integer ID to lookup table |

**CRITICAL**: Do NOT confuse Python epsilon (standard deviation) with Sounio epsilon (confidence).

---

## 5. Canonical Output Format

All three representations serialize to a unified **canonical output format** for display and logging:

```
Knowledge { value: X epsilon: Y prov: "Z" }
```

### Rules
1. `X` (value): 6 significant digits (%.6g format)
2. `Y` (epsilon): 6 significant digits (%.6g format)
3. `Z` (provenance): double-quoted string, must match literal

### Examples
```
Knowledge { value: 500 epsilon: 2.5 prov: "calibration_batch_2026_03" }
Knowledge { value: 1.23456e+08 epsilon: 5e+06 prov: "simulation_run_42" }
Knowledge { value: 0.95 epsilon: 0.001 prov: "high_confidence" }
```

### Python Serialization
```python
def to_sounio_format(self) -> str:
    """Serialize to canonical Sounio output format."""
    return f'Knowledge {{ value: {self.value:.6g} epsilon: {self.epsilon:.6g} prov: "{self.provenance}" }}'

@classmethod
def from_sounio_output(cls, text: str) -> 'Knowledge':
    """Parse a single Knowledge value from Sounio output format."""
    import re
    pattern = r'Knowledge \{ value: ([\d.e+-]+) epsilon: ([\d.e+-]+) prov: "([^"]+)" \}'
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Cannot parse Knowledge from: {text}")
    return cls(float(match.group(1)), float(match.group(2)), match.group(3))
```

---

## 6. Conversion Rules

### Python → Sounio Knowledge[f64]

**Conversion**:
- Python `value` → Sounio `value`
- Python `epsilon` → Sounio `ε` (MUST CONVERT SEMANTICS)
  - If Python epsilon ≈ GUM σ, and you need a confidence:
    - `ε_sounio = 1.0 - (python_epsilon / |value|)` (relative uncertainty → confidence)
    - Or preserve raw epsilon if semantic mismatch is acceptable
- Python `provenance` → Sounio `prov`
- Create dummy/empty `metadata`

**Example**:
```python
# Python
k_py = Knowledge(500.0, 2.5, "calibration_2026")  # rel_unc = 0.5%

# Conversion: Python epsilon is GUM σ
rel_unc = 2.5 / 500.0  # 0.005
epsilon_sounio = 1.0 - rel_unc  # 0.995 confidence

# Sounio equivalent
let k_sio = Knowledge[f64] {
    value: 500.0,
    ε: 0.995,  // 99.5% confident
    prov: "calibration_2026",
    metadata: { ... }
}
```

### Sounio Knowledge[T] → Python

**Conversion**:
- Sounio `value` → Python `value`
- Sounio `ε` → Python `epsilon` (MUST CONVERT SEMANTICS)
  - `python_epsilon = (1.0 - ε_sounio) * |value|` (confidence → relative uncertainty)
  - Or keep raw if semantics differ
- Sounio `prov` → Python `provenance`

**Example**:
```sio
// Sounio
let k = Knowledge[f64] {
    value: 500.0,
    ε: 0.95,  // 95% confident
    prov: "measurement_xyz",
    metadata: { ... }
}

// Conversion to Python
epsilon_python = (1.0 - 0.95) * 500.0  // 0.05 * 500 = 25.0 mg
k_py = Knowledge(500.0, 25.0, "measurement_xyz")
```

### DrugKnowledge → Python

**Conversion**:
- `value` → `value`
- `epsilon` → use as-is (confidence factor)
  - Python epsilon = drug_epsilon * |value| if you want GUM σ
  - Or preserve raw if keeping multiplicative semantics
- `prov_id` → lookup in table to get provenance string

**Example**:
```sio
// Sounio drug discovery
let dose = dk_new(500.0, 0.95, 30)

// Conversion (assuming prov_id 30 = "dosing_protocol_2026")
k_py = Knowledge(500.0, 0.95 * 500.0, "dosing_protocol_2026")
// OR preserve semantics:
k_py = Knowledge(500.0, 0.95, "dosing_protocol_2026")  // keep confidence as epsilon
```

### DrugKnowledge → Sounio Knowledge[f64]

**Conversion**:
- `value` → `value`
- `epsilon` → `ε` directly (both are confidence-like)
- `prov_id` → lookup string, store in `prov`
- Create `metadata`

**Example**:
```sio
fn drug_knowledge_to_shared(dk: DrugKnowledge, prov_table: [string]) -> Knowledge[f64] {
    let prov_str = prov_table[dk.prov_id]
    Knowledge[f64] {
        value: dk.value,
        ε: dk.epsilon,
        prov: prov_str,
        metadata: empty_metadata()
    }
}
```

---

## 7. Validation Checklist for Conversions

Before converting between representations, verify:

- [ ] **Source semantics understood**: Is epsilon a standard deviation, confidence, or multiplicative factor?
- [ ] **Target semantics clear**: What does the target representation expect epsilon to mean?
- [ ] **Provenance chain preserved**: Original source information flows through conversion
- [ ] **Units/metadata tracked**: Are units, timestamps, validator info preserved?
- [ ] **Rounding/precision acceptable**: Does %.6g precision match requirements?
- [ ] **Arithmetic consistency**: Will subsequent operations use correct propagation rules?

---

## 8. References

- **GUM**: BIPM/JCGM-100 "Evaluation of measurement data — Guide to the expression of uncertainty in measurement" (2008)
- **Sounio shared types**: `/ecosystem/shared/epistemic_types.sio`
- **Drug discovery types**: `/ecosystem/drug-discovery/src/types.sio`
- **Python Knowledge**: `/ecosystem/sounio-py/python/sounio/knowledge.py`

---

## 9. FAQ

### Q: Why three different representations?

**A**: Each serves a different optimization objective:
1. **Python** — Full GUM traceability, audit trails, human-readable provenance
2. **Sounio shared** — Extensible metadata, type-safe, portable across PK/PD/clinical
3. **Drug discovery** — Compact integer indices, tight loops, multiplicative confidence chains

### Q: Can I use Python epsilon as Sounio ε directly?

**A**: NO. Python epsilon is GUM σ (standard deviation); Sounio ε is confidence ∈ [0,1].
- Convert: `ε_sounio = 1 - (python_epsilon / |value|)`

### Q: Does DrugKnowledge epsilon mean the same as Sounio ε?

**A**: Semantically similar (both confidence-like), but DrugKnowledge uses multiplicative propagation while Sounio may use geometric mean. Check the arithmetic functions.

### Q: Where do I store units and metadata in DrugKnowledge?

**A**: DrugKnowledge is compact; units/metadata are **external** (e.g., in the containing struct like `MoleculeDesc` or `PKParams`). Use Sounio `Knowledge[T]` with metadata if you need structured units.

### Q: What if my provenance string contains double quotes?

**A**: Escape them: `\"`. The canonical parser handles `\"` in provenance.

---

## Version History

- **v1.0.0** (2026-03-18): Initial mapping for Python ↔ Sounio ↔ DrugKnowledge
