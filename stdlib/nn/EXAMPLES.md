# NN Examples

## 1. Dense Layer Forward Pass

```sio
use nn::dense::Dense;

pub fn main() with Mut, Panic {
    // Create 2-input, 1-output layer
    let layer = Dense::new(2, 1);
    
    // Forward pass
    let input = [1.0, 0.5];
    let output = layer.forward(input);
    
    // Output is sigmoid(w*x + b)
    assert(output[0] > 0.0 && output[0] < 1.0);
}
```

## 2. Epistemic Layer with Uncertainty

```sio
use nn::epistemic_layer::EpistemicLayer;
use epistemic::knowledge::Knowledge;

pub fn main() with Mut, Div, Panic {
    // Layer that propagates uncertainty
    let layer = EpistemicLayer::new(2, 1);
    
    // Input with uncertainty
    let input = [
        Knowledge::measured(1.0, 0.1, "sensor_A"),
        Knowledge::measured(0.5, 0.05, "sensor_B"),
    ];
    
    // Forward pass propagates uncertainty
    let output = layer.forward(input);
    
    // Output has propagated uncertainty
    assert(output[0].uncertainty > 0.0);
}
```

## 3. Autograd Tape for Gradient Computation

```sio
use nn::autograd::Tape;

pub fn main() with Mut, Panic {
    let mut tape = Tape::new();
    
    // Push variables
    let x = tape.push_var(2.0);
    let w = tape.push_var(3.0);
    
    // Compute: y = w * x
    let y = tape.push_mul(w, x);
    
    // Backward pass
    tape.backward();
    
    // Gradients: dy/dw = x = 2, dy/dx = w = 3
    let dw = tape.grad(w);
    let dx = tape.grad(x);
    
    assert(dw == 2.0);
    assert(dx == 3.0);
}
```

## 4. Activation Functions

```sio
use nn::activation::{sigmoid, relu, tanh};

pub fn main() with Div, Panic {
    // Sigmoid: σ(x) = 1 / (1 + e^(-x))
    let sig = sigmoid(0.0);
    assert(sig > 0.49 && sig < 0.51);
    
    // ReLU: max(0, x)
    let r = relu(-1.0);
    assert(r == 0.0);
    
    // Tanh
    let t = tanh(0.0);
    assert(t == 0.0);
}
```

## 5. Quaternion Layer for Geometric DL

```sio
use nn::dense_quaternion::DenseQuaternion;
use math::octonion::Quaternion;

pub fn main() with Mut, Div, Panic {
    // Quaternion-valued layer for rotation-equivariant learning
    let layer = DenseQuaternion::new(4, 2);
    
    // Forward with quaternion weights
    // (Implementation depends on quaternion representation)
}
```

---

**Links to tests:** [`tests/stdlib/nn/`](../../tests/stdlib/nn/)
