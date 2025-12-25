// dense.d - Simple Dense (Fully Connected) Neural Network Layer
//
// Implements a 1-input, 1-output dense layer: y = sigmoid(w*x + b)
// Uses the autograd tape for automatic gradient computation.
//
// Usage:
//   1. Create layer with initial weights
//   2. Forward pass to compute output
//   3. Backward pass to compute gradients
//   4. Update weights using gradients (gradient descent)

// ============================================================================
// MATH HELPERS (copied from autograd.d for standalone use)
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn exp_f64(x: f64) -> f64 {
    if x > 20.0 { return exp_f64(x / 2.0) * exp_f64(x / 2.0) }
    if x < 0.0 - 20.0 { return 1.0 / exp_f64(0.0 - x) }
    let mut sum = 1.0
    let mut term = 1.0
    let mut i = 1
    while i <= 20 { term = term * x / i; sum = sum + term; i = i + 1 }
    return sum
}

fn cos_f64(x: f64) -> f64 {
    let pi = 3.141592653589793
    let mut y = x
    while y > pi { y = y - 2.0 * pi }
    while y < 0.0 - pi { y = y + 2.0 * pi }
    let y2 = y * y
    let mut sum = 1.0
    let mut term = 1.0
    term = term * (0.0 - y2) / 2.0; sum = sum + term
    term = term * (0.0 - y2) / 12.0; sum = sum + term
    term = term * (0.0 - y2) / 30.0; sum = sum + term
    term = term * (0.0 - y2) / 56.0; sum = sum + term
    term = term * (0.0 - y2) / 90.0; sum = sum + term
    return sum
}

// ============================================================================
// OPERATION CODES
// ============================================================================

fn OP_VAR() -> i64 { return 1 }
fn OP_ADD() -> i64 { return 2 }
fn OP_MUL() -> i64 { return 4 }
fn OP_SIGMOID() -> i64 { return 15 }

// ============================================================================
// TAPE STRUCTURE - 6 slots
// Layout for dense layer: [w, b, x, w*x, w*x+b, sigmoid]
// ============================================================================

struct Tape {
    op0: i64, a10: i64, a20: i64, v0: f64, g0: f64,
    op1: i64, a11: i64, a21: i64, v1: f64, g1: f64,
    op2: i64, a12: i64, a22: i64, v2: f64, g2: f64,
    op3: i64, a13: i64, a23: i64, v3: f64, g3: f64,
    op4: i64, a14: i64, a24: i64, v4: f64, g4: f64,
    op5: i64, a15: i64, a25: i64, v5: f64, g5: f64,
    len: i64
}

fn tape_new() -> Tape {
    return Tape {
        op0: 0, a10: 0, a20: 0, v0: 0.0, g0: 0.0,
        op1: 0, a11: 0, a21: 0, v1: 0.0, g1: 0.0,
        op2: 0, a12: 0, a22: 0, v2: 0.0, g2: 0.0,
        op3: 0, a13: 0, a23: 0, v3: 0.0, g3: 0.0,
        op4: 0, a14: 0, a24: 0, v4: 0.0, g4: 0.0,
        op5: 0, a15: 0, a25: 0, v5: 0.0, g5: 0.0,
        len: 0
    }
}

// ============================================================================
// GETTERS
// ============================================================================

fn get_v(t: Tape, i: i64) -> f64 {
    if i == 0 { return t.v0 }
    if i == 1 { return t.v1 }
    if i == 2 { return t.v2 }
    if i == 3 { return t.v3 }
    if i == 4 { return t.v4 }
    if i == 5 { return t.v5 }
    return 0.0
}

fn get_g(t: Tape, i: i64) -> f64 {
    if i == 0 { return t.g0 }
    if i == 1 { return t.g1 }
    if i == 2 { return t.g2 }
    if i == 3 { return t.g3 }
    if i == 4 { return t.g4 }
    if i == 5 { return t.g5 }
    return 0.0
}

fn get_op(t: Tape, i: i64) -> i64 {
    if i == 0 { return t.op0 }
    if i == 1 { return t.op1 }
    if i == 2 { return t.op2 }
    if i == 3 { return t.op3 }
    if i == 4 { return t.op4 }
    if i == 5 { return t.op5 }
    return 0
}

fn get_a1(t: Tape, i: i64) -> i64 {
    if i == 0 { return t.a10 }
    if i == 1 { return t.a11 }
    if i == 2 { return t.a12 }
    if i == 3 { return t.a13 }
    if i == 4 { return t.a14 }
    if i == 5 { return t.a15 }
    return 0
}

fn get_a2(t: Tape, i: i64) -> i64 {
    if i == 0 { return t.a20 }
    if i == 1 { return t.a21 }
    if i == 2 { return t.a22 }
    if i == 3 { return t.a23 }
    if i == 4 { return t.a24 }
    if i == 5 { return t.a25 }
    return 0
}

// ============================================================================
// SETTERS
// ============================================================================

fn set_g0(t: Tape, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: v,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: t.len }
}

fn set_g1(t: Tape, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: v,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: t.len }
}

fn set_g2(t: Tape, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: v,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: t.len }
}

fn set_g3(t: Tape, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: v,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: t.len }
}

fn set_g4(t: Tape, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: v,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: t.len }
}

fn set_g5(t: Tape, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: v, len: t.len }
}

fn set_g(t: Tape, i: i64, v: f64) -> Tape {
    if i == 0 { return set_g0(t, v) }
    if i == 1 { return set_g1(t, v) }
    if i == 2 { return set_g2(t, v) }
    if i == 3 { return set_g3(t, v) }
    if i == 4 { return set_g4(t, v) }
    if i == 5 { return set_g5(t, v) }
    return t
}

// ============================================================================
// PUSH OPERATIONS
// ============================================================================

fn push0(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape { op0: op, a10: a1, a20: a2, v0: v, g0: 0.0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: 1 }
}

fn push1(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: op, a11: a1, a21: a2, v1: v, g1: 0.0,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: 2 }
}

fn push2(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: op, a12: a1, a22: a2, v2: v, g2: 0.0,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: 3 }
}

fn push3(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: op, a13: a1, a23: a2, v3: v, g3: 0.0,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: 4 }
}

fn push4(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: op, a14: a1, a24: a2, v4: v, g4: 0.0,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: 5 }
}

fn push5(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: op, a15: a1, a25: a2, v5: v, g5: 0.0, len: 6 }
}

fn push(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    let i = t.len
    if i == 0 { return push0(t, op, a1, a2, v) }
    if i == 1 { return push1(t, op, a1, a2, v) }
    if i == 2 { return push2(t, op, a1, a2, v) }
    if i == 3 { return push3(t, op, a1, a2, v) }
    if i == 4 { return push4(t, op, a1, a2, v) }
    if i == 5 { return push5(t, op, a1, a2, v) }
    return t
}

// ============================================================================
// TAPE OPERATIONS
// ============================================================================

fn tvar(t: Tape, v: f64) -> Tape { return push(t, OP_VAR(), 0 - 1, 0 - 1, v) }

fn tadd(t: Tape, a: i64, b: i64) -> Tape {
    return push(t, OP_ADD(), a, b, get_v(t, a) + get_v(t, b))
}

fn tmul(t: Tape, a: i64, b: i64) -> Tape {
    return push(t, OP_MUL(), a, b, get_v(t, a) * get_v(t, b))
}

fn tsigmoid(t: Tape, a: i64) -> Tape {
    let av = get_v(t, a)
    return push(t, OP_SIGMOID(), a, 0 - 1, 1.0 / (1.0 + exp_f64(0.0 - av)))
}

// ============================================================================
// BACKWARD PASS (simplified for 6 slots)
// ============================================================================

fn backward_step(t: Tape, i: i64) -> Tape {
    let op = get_op(t, i)
    let a1 = get_a1(t, i)
    let a2 = get_a2(t, i)
    let v = get_v(t, i)
    let dout = get_g(t, i)

    if abs_f64(dout) < 0.0000000001 {
        return t
    }

    let cur_g0 = t.g0
    let cur_g1 = t.g1
    let cur_g2 = t.g2
    let cur_g3 = t.g3
    let cur_g4 = t.g4
    let cur_g5 = t.g5

    let mut new_g0 = cur_g0
    let mut new_g1 = cur_g1
    let mut new_g2 = cur_g2
    let mut new_g3 = cur_g3
    let mut new_g4 = cur_g4
    let mut new_g5 = cur_g5

    if op == OP_ADD() {
        if a1 == 0 { new_g0 = new_g0 + dout }
        if a1 == 1 { new_g1 = new_g1 + dout }
        if a1 == 2 { new_g2 = new_g2 + dout }
        if a1 == 3 { new_g3 = new_g3 + dout }
        if a1 == 4 { new_g4 = new_g4 + dout }
        if a1 == 5 { new_g5 = new_g5 + dout }
        if a2 == 0 { new_g0 = new_g0 + dout }
        if a2 == 1 { new_g1 = new_g1 + dout }
        if a2 == 2 { new_g2 = new_g2 + dout }
        if a2 == 3 { new_g3 = new_g3 + dout }
        if a2 == 4 { new_g4 = new_g4 + dout }
        if a2 == 5 { new_g5 = new_g5 + dout }
    }
    if op == OP_MUL() {
        let av = get_v(t, a1)
        let bv = get_v(t, a2)
        let ga = dout * bv
        let gb = dout * av
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
        if a2 == 0 { new_g0 = new_g0 + gb }
        if a2 == 1 { new_g1 = new_g1 + gb }
        if a2 == 2 { new_g2 = new_g2 + gb }
        if a2 == 3 { new_g3 = new_g3 + gb }
        if a2 == 4 { new_g4 = new_g4 + gb }
        if a2 == 5 { new_g5 = new_g5 + gb }
    }
    if op == OP_SIGMOID() {
        let ga = dout * v * (1.0 - v)
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
    }

    return Tape {
        op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: new_g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: new_g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: new_g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: new_g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: new_g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: new_g5,
        len: t.len
    }
}

fn backward(tape: Tape, out: i64) -> Tape {
    // Inline the seed gradient to avoid struct parameter issues
    // Set g[out] = 1.0 by creating new tape with that gradient
    let t0 = Tape {
        op0: tape.op0, a10: tape.a10, a20: tape.a20, v0: tape.v0,
        g0: if out == 0 { 1.0 } else { tape.g0 },
        op1: tape.op1, a11: tape.a11, a21: tape.a21, v1: tape.v1,
        g1: if out == 1 { 1.0 } else { tape.g1 },
        op2: tape.op2, a12: tape.a12, a22: tape.a22, v2: tape.v2,
        g2: if out == 2 { 1.0 } else { tape.g2 },
        op3: tape.op3, a13: tape.a13, a23: tape.a23, v3: tape.v3,
        g3: if out == 3 { 1.0 } else { tape.g3 },
        op4: tape.op4, a14: tape.a14, a24: tape.a24, v4: tape.v4,
        g4: if out == 4 { 1.0 } else { tape.g4 },
        op5: tape.op5, a15: tape.a15, a25: tape.a25, v5: tape.v5,
        g5: if out == 5 { 1.0 } else { tape.g5 },
        len: tape.len
    }

    // Process each slot from output to input
    let t5 = if t0.len > 5 { backward_step(t0, 5) } else { t0 }
    let t4 = if t5.len > 4 { backward_step(t5, 4) } else { t5 }
    let t3 = if t4.len > 3 { backward_step(t4, 3) } else { t4 }
    let t2 = if t3.len > 2 { backward_step(t3, 2) } else { t3 }
    let t1 = if t2.len > 1 { backward_step(t2, 1) } else { t2 }
    let t_final = if t1.len > 0 { backward_step(t1, 0) } else { t1 }

    return t_final
}

// ============================================================================
// DENSE LAYER
// ============================================================================

// Dense layer: y = sigmoid(w * x + b)
// Tape layout: [0]=w, [1]=b, [2]=x, [3]=w*x, [4]=w*x+b, [5]=sigmoid
struct DenseLayer {
    weight: f64,
    bias: f64
}

fn dense_new(w: f64, b: f64) -> DenseLayer {
    return DenseLayer { weight: w, bias: b }
}

// Forward pass: builds tape and computes output
fn dense_forward(layer: DenseLayer, x: f64) -> Tape {
    let mut t = tape_new()
    t = tvar(t, layer.weight)  // [0] = w
    t = tvar(t, layer.bias)    // [1] = b
    t = tvar(t, x)             // [2] = x
    t = tmul(t, 0, 2)          // [3] = w * x
    t = tadd(t, 3, 1)          // [4] = w*x + b
    t = tsigmoid(t, 4)         // [5] = sigmoid(w*x + b)
    return t
}

// Get output value from tape
fn dense_output(t: Tape) -> f64 {
    return get_v(t, 5)
}

// Get gradients after backward pass
fn dense_grad_w(t: Tape) -> f64 { return get_g(t, 0) }
fn dense_grad_b(t: Tape) -> f64 { return get_g(t, 1) }
fn dense_grad_x(t: Tape) -> f64 { return get_g(t, 2) }

// Update layer weights using gradient descent
fn dense_update(layer: DenseLayer, grad_w: f64, grad_b: f64, lr: f64) -> DenseLayer {
    return DenseLayer {
        weight: layer.weight - lr * grad_w,
        bias: layer.bias - lr * grad_b
    }
}

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

// Binary cross-entropy loss: -[y*log(p) + (1-y)*log(1-p)]
// For simplicity, we compute MSE loss instead: (y - p)^2
fn mse_loss(predicted: f64, target: f64) -> f64 {
    let diff = predicted - target
    return diff * diff
}

// Gradient of MSE loss w.r.t. predicted: 2*(p - y)
fn mse_loss_grad(predicted: f64, target: f64) -> f64 {
    return 2.0 * (predicted - target)
}

// ============================================================================
// TESTS
// ============================================================================

fn main() -> i32 {
    println("=== Dense Layer Tests ===")
    println("")

    let tol = 0.01
    let mut ok = true

    // Test 1: Forward pass
    println("Test 1: Forward pass")
    let layer1 = dense_new(1.0, 0.0)  // w=1, b=0
    let t1 = dense_forward(layer1, 0.0)  // x=0
    let y1 = dense_output(t1)
    // sigmoid(1*0 + 0) = sigmoid(0) = 0.5
    println("  y = sigmoid(1*0 + 0) = ")
    println(y1)
    if abs_f64(y1 - 0.5) > tol { ok = false; println("  FAIL") }
    println("")

    // Test 2: Forward with different input
    println("Test 2: Forward with x=2")
    let layer2 = dense_new(0.5, 0.5)  // w=0.5, b=0.5
    let t2 = dense_forward(layer2, 2.0)  // x=2
    let y2 = dense_output(t2)
    // sigmoid(0.5*2 + 0.5) = sigmoid(1.5) ≈ 0.818
    let expected2 = 1.0 / (1.0 + exp_f64(0.0 - 1.5))
    println("  y = sigmoid(0.5*2 + 0.5) = ")
    println(y2)
    println("  expected = ")
    println(expected2)
    if abs_f64(y2 - expected2) > tol { ok = false; println("  FAIL") }
    println("")

    // Test 3: Backward pass - gradient check
    println("Test 3: Backward pass gradients")
    let layer3 = dense_new(2.0, 1.0)  // w=2, b=1
    let t3_fwd = dense_forward(layer3, 0.5)  // x=0.5
    let t3_bwd = backward(t3_fwd, 5)
    let y3 = dense_output(t3_bwd)
    let dw3 = dense_grad_w(t3_bwd)
    let db3 = dense_grad_b(t3_bwd)
    let dx3 = dense_grad_x(t3_bwd)
    // y = sigmoid(2*0.5 + 1) = sigmoid(2) ≈ 0.881
    // dy/dw = dy/d(wx+b) * d(wx+b)/dw = y*(1-y) * x = 0.881*0.119*0.5 ≈ 0.0524
    // dy/db = y*(1-y) * 1 ≈ 0.105
    // dy/dx = y*(1-y) * w = 0.881*0.119*2 ≈ 0.210
    let sig3 = 1.0 / (1.0 + exp_f64(0.0 - 2.0))
    let dsig3 = sig3 * (1.0 - sig3)
    let exp_dw3 = dsig3 * 0.5
    let exp_db3 = dsig3 * 1.0
    let exp_dx3 = dsig3 * 2.0
    println("  y = ")
    println(y3)
    println("  dL/dw = ")
    println(dw3)
    println("  expected dL/dw = ")
    println(exp_dw3)
    println("  dL/db = ")
    println(db3)
    println("  expected dL/db = ")
    println(exp_db3)
    println("  dL/dx = ")
    println(dx3)
    println("  expected dL/dx = ")
    println(exp_dx3)
    if abs_f64(dw3 - exp_dw3) > tol { ok = false; println("  FAIL: dw") }
    if abs_f64(db3 - exp_db3) > tol { ok = false; println("  FAIL: db") }
    if abs_f64(dx3 - exp_dx3) > tol { ok = false; println("  FAIL: dx") }
    println("")

    // Test 4: Training loop (learning XOR-like pattern)
    println("Test 4: Training on simple pattern")
    println("  Goal: Learn y=1 when x>0, y=0 when x<0")
    let mut layer4 = dense_new(0.1, 0.0)  // Start with small weights
    let lr = 1.0  // Learning rate

    // Training data: (x, target)
    // x=1 -> target=1
    // x=-1 -> target=0

    let mut epoch = 0
    let mut loss_sum = 0.0
    while epoch < 100 {
        // Forward on x=1, target=1
        let t_pos = dense_forward(layer4, 1.0)
        let y_pos = dense_output(t_pos)
        let loss_pos = mse_loss(y_pos, 1.0)
        let t_pos_bwd = backward(t_pos, 5)
        let dL_dy_pos = mse_loss_grad(y_pos, 1.0)
        let dw_pos = dense_grad_w(t_pos_bwd) * dL_dy_pos
        let db_pos = dense_grad_b(t_pos_bwd) * dL_dy_pos

        // Forward on x=-1, target=0
        let t_neg = dense_forward(layer4, 0.0 - 1.0)
        let y_neg = dense_output(t_neg)
        let loss_neg = mse_loss(y_neg, 0.0)
        let t_neg_bwd = backward(t_neg, 5)
        let dL_dy_neg = mse_loss_grad(y_neg, 0.0)
        let dw_neg = dense_grad_w(t_neg_bwd) * dL_dy_neg
        let db_neg = dense_grad_b(t_neg_bwd) * dL_dy_neg

        // Average gradients and update
        let avg_dw = (dw_pos + dw_neg) / 2.0
        let avg_db = (db_pos + db_neg) / 2.0
        layer4 = dense_update(layer4, avg_dw, avg_db, lr)

        loss_sum = (loss_pos + loss_neg) / 2.0
        epoch = epoch + 1
    }

    println("  After 100 epochs:")
    println("  Final weight = ")
    println(layer4.weight)
    println("  Final bias = ")
    println(layer4.bias)
    println("  Final loss = ")
    println(loss_sum)

    // Test predictions
    let final_pos = dense_output(dense_forward(layer4, 1.0))
    let final_neg = dense_output(dense_forward(layer4, 0.0 - 1.0))
    println("  Prediction for x=1: ")
    println(final_pos)
    println("  Prediction for x=-1: ")
    println(final_neg)

    // Should have learned: x=1 -> ~1, x=-1 -> ~0
    if final_pos < 0.7 { ok = false; println("  FAIL: x=1 should predict >0.7") }
    if final_neg > 0.3 { ok = false; println("  FAIL: x=-1 should predict <0.3") }
    println("")

    if ok {
        println("ALL TESTS PASSED")
        return 0
    } else {
        println("SOME TESTS FAILED")
        return 1
    }
}
