# Math Examples

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
