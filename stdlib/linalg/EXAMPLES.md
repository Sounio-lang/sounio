# Linalg Examples

## 1. Epistemic Matrix Builder Pattern

```sio
use linalg::epistemic_matrix::EpistemicMatrix;

pub fn main() with Mut, Div, Panic {
    // Create identity matrix with uncertainty
    let m = EpistemicMatrix::identity(2)
        .uncertainty(0.01)
        .confidence(0.95);
    
    // Access element
    let val = m.get(0, 0);
    assert(val == 1.0);
}
```

## 2. Uncertainty Propagation in Matrix Multiplication

```sio
use linalg::epistemic_matrix::EpistemicMatrix;

pub fn main() with Mut, Div, Panic {
    // Matrix A with uncertainty 0.1
    let a = EpistemicMatrix::zeros(2, 2)
        .uncertainty(0.1)
        .confidence(0.90)
        .set(0, 0, 1.0)
        .set(0, 1, 2.0)
        .set(1, 0, 3.0)
        .set(1, 1, 4.0);
    
    // Matrix B with uncertainty 0.05
    let b = EpistemicMatrix::identity(2)
        .uncertainty(0.05)
        .confidence(0.95);
    
    // C = A * B: uncertainty propagates via GUM
    let c = a.matmul(&b);
    
    // Check propagated uncertainty
    let unc = c.uncertainty_at(0, 0);
    assert(unc > 0.0);
}
```

## 3. Confidence Degradation

```sio
use linalg::epistemic_matrix::EpistemicMatrix;

pub fn main() with Mut, Div, Panic {
    // High confidence matrix
    let a = EpistemicMatrix::identity(2)
        .uncertainty(0.01)
        .confidence(0.99);
    
    // Lower confidence matrix
    let b = EpistemicMatrix::identity(2)
        .uncertainty(0.1)
        .confidence(0.70);
    
    // Result confidence = min(a.conf, b.conf) = 0.70
    let c = a.matmul(&b);
    
    let conf = c.confidence_at(0, 0);
    assert(conf == 0.70);
}
```

## 4. Standard Matrix Operations

```sio
use linalg::matrix::Matrix;

pub fn main() with Mut, Div, Panic {
    let m = Matrix::zeros(3, 3);
    // Standard operations without epistemic tracking
}
```

## 5. Vector Operations

```sio
use linalg::vector::Vector;

pub fn main() with Mut, Div, Panic {
    let v = Vector::new([1.0, 2.0, 3.0]);
    let norm = v.norm();
    assert(norm > 3.0);
}
```

---

**Links to tests:** [`tests/stdlib/linalg/`](../../tests/stdlib/linalg/)
