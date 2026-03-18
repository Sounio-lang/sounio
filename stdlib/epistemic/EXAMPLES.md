# Epistemic Examples

## 1. GUMUncertainty Builder Pattern

```sio
use epistemic::gum::GUMUncertainty;

pub fn main() {
    // Type A: from statistical data (n observations)
    let u_a = GUMUncertainty::type_a(0.1, 10)  // std_dev=0.1, n=10
        .with_sensitivity(2.0);
    
    // Type B: from a priori knowledge
    let u_b = GUMUncertainty::type_b(0.05)
        .with_sensitivity(1.0);
    
    // Type B from uniform distribution: u = a/sqrt(3)
    let u_uniform = GUMUncertainty::type_b_uniform(0.1);
    
    // Type B from triangular distribution: u = a/sqrt(6)
    let u_tri = GUMUncertainty::type_b_triangular(0.1);
}
```

## 2. Uncertainty Propagation with Knowledge

```sio
use epistemic::knowledge::Knowledge;

pub fn main() with Div {
    // Measured values with uncertainty
    let dose = Knowledge::measured(500.0, 25.0, "scale_A");
    let volume = Knowledge::measured(10.0, 0.1, "pipette_B");
    
    // Arithmetic propagates variance automatically
    let concentration = dose / volume;
    
    // concentration.value ≈ 50.0
    // concentration.variance propagated via GUM
    assert(concentration.value > 49.0 && concentration.value < 51.0);
}
```

## 3. Confidence Degradation

```sio
use epistemic::knowledge::{Knowledge, BetaConfidence};

pub fn main() {
    // High confidence measurement
    let high_conf = BetaConfidence::certain();  // alpha=1, beta=0
    
    // Degrade by 5%
    let degraded = high_conf.degrade(0.05);
    
    // Mean confidence decreases
    assert(degraded.mean() < high_conf.mean());
}
```

## 4. Provenance Tracking

```sio
use epistemic::knowledge::Knowledge;

pub fn main() with Div {
    let a = Knowledge::measured(10.0, 0.1, "sensor_A");
    let b = Knowledge::measured(2.0, 0.05, "sensor_B");
    
    let result = a / b;
    
    // Provenance tracks computation history
    assert(result.provenance.source != "");
}
```

## 5. GUM Result with Coverage Factor

```sio
use epistemic::gum::{GUMUncertainty, GUMResult};

pub fn main() {
    let u1 = GUMUncertainty::type_a(0.1, 10);
    let u2 = GUMUncertainty::type_b(0.05);
    
    // Combine uncertainties
    let combined = u1.combine(&u2);
    
    // Get effective degrees of freedom
    let v_eff = combined.degrees_of_freedom;
    
    // Expanded uncertainty with k=2 (≈95% CI)
    let expanded = combined.std_uncertainty * 2.0;
}
```

---

**Links to tests:** [`tests/stdlib/epistemic/`](../../tests/stdlib/epistemic/)
