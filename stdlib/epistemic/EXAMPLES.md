# Epistemic Examples

## 1. GUM Uncertainty Components

```sio
use epistemic::gum::{
    gum_type_a, gum_type_b, gum_type_b_uniform, gum_type_b_triangular,
    gum_with_sensitivity, gum_combine2, gum_std_u, gum_dof, gum_u95,
}

pub fn main() with Mut, Div, Panic {
    // Type A: from n observations (u = std_dev / sqrt(n), nu = n - 1)
    let u_a = gum_with_sensitivity(gum_type_a(0.1, 10), 2.0)

    // Type B: from a priori knowledge
    let u_b = gum_with_sensitivity(gum_type_b(0.05), 1.0)

    // Type B from a uniform distribution: u = half_width / sqrt(3)
    let u_uniform = gum_type_b_uniform(0.1)

    // Type B from a triangular distribution: u = half_width / sqrt(6)
    let u_tri = gum_type_b_triangular(0.1)

    // Combine and read the budget
    let combined = gum_combine2(1.0, u_a, u_b)
    let uc = gum_std_u(combined)    // combined standard uncertainty
    let v_eff = gum_dof(combined)   // effective degrees of freedom
    let expanded = gum_u95(combined) // expanded uncertainty at 95%
}
```

Anchor: `tests/stdlib/epistemic/test_gum_stdlib.sio`.

## 2. Uncertainty Propagation with Epistemic

```sio
use epistemic::knowledge::{ep_measured, ep_div, ep_val}

pub fn main() with Div, Panic {
    // Measured values: (val, std_dev). Variance is stored as std_dev^2.
    let dose = ep_measured(500.0, 25.0)
    let volume = ep_measured(10.0, 0.1)

    // Division propagates variance by the GUM delta method.
    let concentration = ep_div(&dose, &volume)

    let v = ep_val(&concentration)
    assert(v > 49.0 && v < 51.0)
}
```

Anchor: `tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio`.

## 3. Confidence Degradation

```sio
use epistemic::knowledge::{ep_measured, ep_certain, ep_add, ep_confidence}

pub fn main() with Panic {
    // ep_certain stores confidence 1000; ep_measured stores 900.
    let exact = ep_certain(10.0)
    let noisy = ep_measured(2.0, 0.05)

    // Combination applies operation-specific decay to the lower input
    // confidence: ep_add/ep_sub multiply by 99/100, ep_mul by 98/100,
    // ep_div by 97/100. It never increases.
    let result = ep_add(&exact, &noisy)
    assert(ep_confidence(&result) <= ep_confidence(&exact))
    assert(ep_confidence(&result) == 891)
}
```

`BetaConfidence::certain()` and `.degrade()` are not on the checked surface. The beta-posterior type is the separate `EpistemicBeta` in `stdlib/epistemic/beta_confidence.sio` (`eb_new`, `eb_mean`), anchored by `tests/run-pass/beta_confidence_rule.sio`.

## 4. Reading a Result

```sio
use epistemic::knowledge::{ep_measured, ep_div, ep_val, ep_std}

pub fn main() with Mut, Div, Panic {
    let a = ep_measured(10.0, 0.1)
    let b = ep_measured(2.0, 0.05)

    let result = ep_div(&a, &b)
    let v = ep_val(&result)
    let s = ep_std(&result)
}
```

`Epistemic` has no `provenance` field. Shared-source covariance tracking lives in `stdlib/epistemic/affine.sio` (anchor: `tests/run-pass/affine_shared_source_add.sio`). Provenance bookkeeping is in `stdlib/epistemic/prov.sio`.

---

**Links to tests:** [`tests/stdlib/epistemic/`](../../tests/stdlib/epistemic/)
