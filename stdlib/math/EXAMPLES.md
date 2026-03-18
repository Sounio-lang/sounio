# Math Examples

## 1. Quaternion Rotation

```sio
use math::ga::quaternion::{quat_new, quat_mul, quat_conjugate, quat_rotate_vec3, quat_identity};

pub fn main() with Mut, Div, Panic {
    // Identity quaternion (no rotation)
    let id = quat_identity();
    let w = quat_new(1.0, 0.0, 0.0, 0.0);
    
    // Quaternion multiplication
    let q1 = quat_new(1.0, 0.0, 0.0, 0.0);
    let q2 = quat_new(0.0, 1.0, 0.0, 0.0);
    let prod = quat_mul(q1, q2);
    
    // Conjugate
    let conj = quat_conjugate(q1);
}
```

## 2. Epistemic Geometric Algebra

```sio
use math::ga::epistemic::EpistemicMultivector;
use epistemic::knowledge::Knowledge;

pub fn main() with Mut, Div, Panic {
    // Multivector with uncertainty
    let mv = EpistemicMultivector::new();
    
    // Operations propagate uncertainty through GA product
}
```

## 3. FFT with Uncertainty

```sio
use math::fft::FFT;

pub fn main() with Mut, Div, Panic {
    // FFT on epistemic input
    // (Implementation depends on fft.sio interface)
}
```

## 4. Lie Group Exponential

```sio
use math::lie::LieAlgebra;

pub fn main() with Mut, Div, Panic {
    // Exponential map from Lie algebra to Lie group
    // exp: so(3) -> SO(3)
    // (Implementation depends on lie.sio interface)
}
```

## 5. Cayley-Dickson Construction

```sio
use math::cayley_dickson::Complex;
use math::octonion::Octonion;

pub fn main() with Mut, Div, Panic {
    // Build hypercomplex numbers via Cayley-Dickson
    // R -> C -> H -> O -> S (quaternions, octonions, sedenions)
}
```

---

**Links to tests:** [`tests/stdlib/math/`](../../tests/stdlib/math/)
