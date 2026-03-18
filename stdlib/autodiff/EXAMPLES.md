# Autodiff Examples

## 1. Tape-Based Reverse-Mode AD

```sio
use autodiff::tape::Tape;

pub fn main() with Mut, Div, Panic {
    let mut tape = Tape::new();
    
    // Push variable: x = 2
    let x = tape.push_var(2.0);
    
    // Compute: y = x² + 3x
    let x2 = tape.push_mul(x, x);
    let x3 = tape.push_mul(x, tape.push_const(3.0));
    let y = tape.push_add(x2, x3);
    
    // Backward pass
    tape.backward();
    
    // Gradient: dy/dx = 2x + 3 = 7
    let dx = tape.grad(x);
    assert(dx > 6.9 && dx < 7.1);
}
```

## 2. Epistemic Dual Numbers

```sio
use autodiff::epistemic_dual::EpistemicDual;
use epistemic::knowledge::Knowledge;

pub fn main() with Mut, Div, Panic {
    // Dual number with uncertainty: (value, derivative, knowledge)
    let x = EpistemicDual::new(2.0, 1.0, Knowledge::measured(2.0, 0.1, "source"));
    
    // Square: f(x) = x²
    let y = x * x;
    
    // y.value = 4.0
    // y.derivative = 4.0 (dy/dx = 2x = 4)
    // y.uncertainty propagated
    assert(y.value > 3.9 && y.value < 4.1);
}
```

## 3. Gradient of Function

```sio
use autodiff::grad::grad;

pub fn main() with Mut, Div, Panic {
    // Compute gradient of f(x) = x² at x = 3
    let g = grad(|x| x * x, 3.0);
    
    // f'(3) = 2*3 = 6
    assert(g > 5.9 && g < 6.1);
}
```

## 4. Forward-Mode with Dual Numbers

```sio
use autodiff::dual::Dual;

pub fn main() with Mut, Div, Panic {
    // Dual number: (value, derivative)
    let x = Dual::new(3.0, 1.0);  // x = 3, dx = 1
    
    // f(x) = x² + 2x + 1
    let y = x * x + Dual::new(2.0, 0.0) * x + Dual::new(1.0, 0.0);
    
    // y.value = 9 + 6 + 1 = 16
    // y.derivative = 2*3 + 2 = 8
    assert(y.value > 15.9 && y.value < 16.1);
    assert(y.derivative > 7.9 && y.derivative < 8.1);
}
```

## 5. Multiple Operations

```sio
use autodiff::tape::Tape;

pub fn main() with Mut, Div, Panic {
    let mut tape = Tape::new();
    
    let x = tape.push_var(1.0);
    
    // f(x) = sin(x²)
    let x2 = tape.push_mul(x, x);
    let y = tape.push_sin(x2);
    
    tape.backward();
    
    // dy/dx = cos(x²) * 2x = cos(1) * 2 ≈ 1.08
    let dx = tape.grad(x);
    assert(dx > 1.0 && dx < 1.2);
}
```

---

**Links to tests:** [`tests/stdlib/autodiff/`](../../tests/stdlib/autodiff/)
