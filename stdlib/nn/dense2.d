// dense2.d - 2-Input Dense Neural Network Layer
//
// Implements a 2-input, 1-output dense layer: y = sigmoid(w1*x1 + w2*x2 + b)
// Uses a 10-slot autograd tape for gradient computation.
//
// Tape layout:
//   [0] w1      [1] w2      [2] b       [3] x1      [4] x2
//   [5] w1*x1   [6] w2*x2   [7] sum     [8] sum+b   [9] sigmoid

// ============================================================================
// MATH HELPERS
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

// ============================================================================
// OPERATION CODES
// ============================================================================

fn OP_VAR() -> i64 { return 1 }
fn OP_ADD() -> i64 { return 2 }
fn OP_MUL() -> i64 { return 4 }
fn OP_SIGMOID() -> i64 { return 15 }

// ============================================================================
// 10-SLOT TAPE STRUCTURE
// ============================================================================

struct Tape {
    // Slot 0-4
    op0: i64, a10: i64, a20: i64, v0: f64, g0: f64,
    op1: i64, a11: i64, a21: i64, v1: f64, g1: f64,
    op2: i64, a12: i64, a22: i64, v2: f64, g2: f64,
    op3: i64, a13: i64, a23: i64, v3: f64, g3: f64,
    op4: i64, a14: i64, a24: i64, v4: f64, g4: f64,
    // Slot 5-9
    op5: i64, a15: i64, a25: i64, v5: f64, g5: f64,
    op6: i64, a16: i64, a26: i64, v6: f64, g6: f64,
    op7: i64, a17: i64, a27: i64, v7: f64, g7: f64,
    op8: i64, a18: i64, a28: i64, v8: f64, g8: f64,
    op9: i64, a19: i64, a29: i64, v9: f64, g9: f64,
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
        op6: 0, a16: 0, a26: 0, v6: 0.0, g6: 0.0,
        op7: 0, a17: 0, a27: 0, v7: 0.0, g7: 0.0,
        op8: 0, a18: 0, a28: 0, v8: 0.0, g8: 0.0,
        op9: 0, a19: 0, a29: 0, v9: 0.0, g9: 0.0,
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
    if i == 6 { return t.v6 }
    if i == 7 { return t.v7 }
    if i == 8 { return t.v8 }
    if i == 9 { return t.v9 }
    return 0.0
}

fn get_g(t: Tape, i: i64) -> f64 {
    if i == 0 { return t.g0 }
    if i == 1 { return t.g1 }
    if i == 2 { return t.g2 }
    if i == 3 { return t.g3 }
    if i == 4 { return t.g4 }
    if i == 5 { return t.g5 }
    if i == 6 { return t.g6 }
    if i == 7 { return t.g7 }
    if i == 8 { return t.g8 }
    if i == 9 { return t.g9 }
    return 0.0
}

fn get_op(t: Tape, i: i64) -> i64 {
    if i == 0 { return t.op0 }
    if i == 1 { return t.op1 }
    if i == 2 { return t.op2 }
    if i == 3 { return t.op3 }
    if i == 4 { return t.op4 }
    if i == 5 { return t.op5 }
    if i == 6 { return t.op6 }
    if i == 7 { return t.op7 }
    if i == 8 { return t.op8 }
    if i == 9 { return t.op9 }
    return 0
}

fn get_a1(t: Tape, i: i64) -> i64 {
    if i == 0 { return t.a10 }
    if i == 1 { return t.a11 }
    if i == 2 { return t.a12 }
    if i == 3 { return t.a13 }
    if i == 4 { return t.a14 }
    if i == 5 { return t.a15 }
    if i == 6 { return t.a16 }
    if i == 7 { return t.a17 }
    if i == 8 { return t.a18 }
    if i == 9 { return t.a19 }
    return 0
}

fn get_a2(t: Tape, i: i64) -> i64 {
    if i == 0 { return t.a20 }
    if i == 1 { return t.a21 }
    if i == 2 { return t.a22 }
    if i == 3 { return t.a23 }
    if i == 4 { return t.a24 }
    if i == 5 { return t.a25 }
    if i == 6 { return t.a26 }
    if i == 7 { return t.a27 }
    if i == 8 { return t.a28 }
    if i == 9 { return t.a29 }
    return 0
}

// ============================================================================
// PUSH OPERATIONS (create new tape with slot filled)
// ============================================================================

fn push0(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape {
        op0: op, a10: a1, a20: a2, v0: v, g0: 0.0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5,
        op6: t.op6, a16: t.a16, a26: t.a26, v6: t.v6, g6: t.g6,
        op7: t.op7, a17: t.a17, a27: t.a27, v7: t.v7, g7: t.g7,
        op8: t.op8, a18: t.a18, a28: t.a28, v8: t.v8, g8: t.g8,
        op9: t.op9, a19: t.a19, a29: t.a29, v9: t.v9, g9: t.g9,
        len: 1
    }
}

fn push1(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape {
        op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: op, a11: a1, a21: a2, v1: v, g1: 0.0,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5,
        op6: t.op6, a16: t.a16, a26: t.a26, v6: t.v6, g6: t.g6,
        op7: t.op7, a17: t.a17, a27: t.a27, v7: t.v7, g7: t.g7,
        op8: t.op8, a18: t.a18, a28: t.a28, v8: t.v8, g8: t.g8,
        op9: t.op9, a19: t.a19, a29: t.a29, v9: t.v9, g9: t.g9,
        len: 2
    }
}

fn push2(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape {
        op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: op, a12: a1, a22: a2, v2: v, g2: 0.0,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5,
        op6: t.op6, a16: t.a16, a26: t.a26, v6: t.v6, g6: t.g6,
        op7: t.op7, a17: t.a17, a27: t.a27, v7: t.v7, g7: t.g7,
        op8: t.op8, a18: t.a18, a28: t.a28, v8: t.v8, g8: t.g8,
        op9: t.op9, a19: t.a19, a29: t.a29, v9: t.v9, g9: t.g9,
        len: 3
    }
}

fn push3(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape {
        op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: op, a13: a1, a23: a2, v3: v, g3: 0.0,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5,
        op6: t.op6, a16: t.a16, a26: t.a26, v6: t.v6, g6: t.g6,
        op7: t.op7, a17: t.a17, a27: t.a27, v7: t.v7, g7: t.g7,
        op8: t.op8, a18: t.a18, a28: t.a28, v8: t.v8, g8: t.g8,
        op9: t.op9, a19: t.a19, a29: t.a29, v9: t.v9, g9: t.g9,
        len: 4
    }
}

fn push4(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape {
        op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: op, a14: a1, a24: a2, v4: v, g4: 0.0,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5,
        op6: t.op6, a16: t.a16, a26: t.a26, v6: t.v6, g6: t.g6,
        op7: t.op7, a17: t.a17, a27: t.a27, v7: t.v7, g7: t.g7,
        op8: t.op8, a18: t.a18, a28: t.a28, v8: t.v8, g8: t.g8,
        op9: t.op9, a19: t.a19, a29: t.a29, v9: t.v9, g9: t.g9,
        len: 5
    }
}

fn push5(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape {
        op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: op, a15: a1, a25: a2, v5: v, g5: 0.0,
        op6: t.op6, a16: t.a16, a26: t.a26, v6: t.v6, g6: t.g6,
        op7: t.op7, a17: t.a17, a27: t.a27, v7: t.v7, g7: t.g7,
        op8: t.op8, a18: t.a18, a28: t.a28, v8: t.v8, g8: t.g8,
        op9: t.op9, a19: t.a19, a29: t.a29, v9: t.v9, g9: t.g9,
        len: 6
    }
}

fn push6(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape {
        op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5,
        op6: op, a16: a1, a26: a2, v6: v, g6: 0.0,
        op7: t.op7, a17: t.a17, a27: t.a27, v7: t.v7, g7: t.g7,
        op8: t.op8, a18: t.a18, a28: t.a28, v8: t.v8, g8: t.g8,
        op9: t.op9, a19: t.a19, a29: t.a29, v9: t.v9, g9: t.g9,
        len: 7
    }
}

fn push7(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape {
        op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5,
        op6: t.op6, a16: t.a16, a26: t.a26, v6: t.v6, g6: t.g6,
        op7: op, a17: a1, a27: a2, v7: v, g7: 0.0,
        op8: t.op8, a18: t.a18, a28: t.a28, v8: t.v8, g8: t.g8,
        op9: t.op9, a19: t.a19, a29: t.a29, v9: t.v9, g9: t.g9,
        len: 8
    }
}

fn push8(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape {
        op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5,
        op6: t.op6, a16: t.a16, a26: t.a26, v6: t.v6, g6: t.g6,
        op7: t.op7, a17: t.a17, a27: t.a27, v7: t.v7, g7: t.g7,
        op8: op, a18: a1, a28: a2, v8: v, g8: 0.0,
        op9: t.op9, a19: t.a19, a29: t.a29, v9: t.v9, g9: t.g9,
        len: 9
    }
}

fn push9(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape {
        op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5,
        op6: t.op6, a16: t.a16, a26: t.a26, v6: t.v6, g6: t.g6,
        op7: t.op7, a17: t.a17, a27: t.a27, v7: t.v7, g7: t.g7,
        op8: t.op8, a18: t.a18, a28: t.a28, v8: t.v8, g8: t.g8,
        op9: op, a19: a1, a29: a2, v9: v, g9: 0.0,
        len: 10
    }
}

fn push(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    let i = t.len
    if i == 0 { return push0(t, op, a1, a2, v) }
    if i == 1 { return push1(t, op, a1, a2, v) }
    if i == 2 { return push2(t, op, a1, a2, v) }
    if i == 3 { return push3(t, op, a1, a2, v) }
    if i == 4 { return push4(t, op, a1, a2, v) }
    if i == 5 { return push5(t, op, a1, a2, v) }
    if i == 6 { return push6(t, op, a1, a2, v) }
    if i == 7 { return push7(t, op, a1, a2, v) }
    if i == 8 { return push8(t, op, a1, a2, v) }
    if i == 9 { return push9(t, op, a1, a2, v) }
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
// BACKWARD PASS
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

    // Read current gradients
    let cur_g0 = t.g0
    let cur_g1 = t.g1
    let cur_g2 = t.g2
    let cur_g3 = t.g3
    let cur_g4 = t.g4
    let cur_g5 = t.g5
    let cur_g6 = t.g6
    let cur_g7 = t.g7
    let cur_g8 = t.g8
    let cur_g9 = t.g9

    let mut new_g0 = cur_g0
    let mut new_g1 = cur_g1
    let mut new_g2 = cur_g2
    let mut new_g3 = cur_g3
    let mut new_g4 = cur_g4
    let mut new_g5 = cur_g5
    let mut new_g6 = cur_g6
    let mut new_g7 = cur_g7
    let mut new_g8 = cur_g8
    let mut new_g9 = cur_g9

    if op == OP_ADD() {
        if a1 == 0 { new_g0 = new_g0 + dout }
        if a1 == 1 { new_g1 = new_g1 + dout }
        if a1 == 2 { new_g2 = new_g2 + dout }
        if a1 == 3 { new_g3 = new_g3 + dout }
        if a1 == 4 { new_g4 = new_g4 + dout }
        if a1 == 5 { new_g5 = new_g5 + dout }
        if a1 == 6 { new_g6 = new_g6 + dout }
        if a1 == 7 { new_g7 = new_g7 + dout }
        if a1 == 8 { new_g8 = new_g8 + dout }
        if a1 == 9 { new_g9 = new_g9 + dout }
        if a2 == 0 { new_g0 = new_g0 + dout }
        if a2 == 1 { new_g1 = new_g1 + dout }
        if a2 == 2 { new_g2 = new_g2 + dout }
        if a2 == 3 { new_g3 = new_g3 + dout }
        if a2 == 4 { new_g4 = new_g4 + dout }
        if a2 == 5 { new_g5 = new_g5 + dout }
        if a2 == 6 { new_g6 = new_g6 + dout }
        if a2 == 7 { new_g7 = new_g7 + dout }
        if a2 == 8 { new_g8 = new_g8 + dout }
        if a2 == 9 { new_g9 = new_g9 + dout }
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
        if a1 == 6 { new_g6 = new_g6 + ga }
        if a1 == 7 { new_g7 = new_g7 + ga }
        if a1 == 8 { new_g8 = new_g8 + ga }
        if a1 == 9 { new_g9 = new_g9 + ga }
        if a2 == 0 { new_g0 = new_g0 + gb }
        if a2 == 1 { new_g1 = new_g1 + gb }
        if a2 == 2 { new_g2 = new_g2 + gb }
        if a2 == 3 { new_g3 = new_g3 + gb }
        if a2 == 4 { new_g4 = new_g4 + gb }
        if a2 == 5 { new_g5 = new_g5 + gb }
        if a2 == 6 { new_g6 = new_g6 + gb }
        if a2 == 7 { new_g7 = new_g7 + gb }
        if a2 == 8 { new_g8 = new_g8 + gb }
        if a2 == 9 { new_g9 = new_g9 + gb }
    }
    if op == OP_SIGMOID() {
        let ga = dout * v * (1.0 - v)
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
        if a1 == 6 { new_g6 = new_g6 + ga }
        if a1 == 7 { new_g7 = new_g7 + ga }
        if a1 == 8 { new_g8 = new_g8 + ga }
        if a1 == 9 { new_g9 = new_g9 + ga }
    }

    return Tape {
        op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: new_g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: new_g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: new_g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: new_g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: new_g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: new_g5,
        op6: t.op6, a16: t.a16, a26: t.a26, v6: t.v6, g6: new_g6,
        op7: t.op7, a17: t.a17, a27: t.a27, v7: t.v7, g7: new_g7,
        op8: t.op8, a18: t.a18, a28: t.a28, v8: t.v8, g8: new_g8,
        op9: t.op9, a19: t.a19, a29: t.a29, v9: t.v9, g9: new_g9,
        len: t.len
    }
}

fn backward(tape: Tape, out: i64) -> Tape {
    // Seed output gradient
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
        op6: tape.op6, a16: tape.a16, a26: tape.a26, v6: tape.v6,
        g6: if out == 6 { 1.0 } else { tape.g6 },
        op7: tape.op7, a17: tape.a17, a27: tape.a27, v7: tape.v7,
        g7: if out == 7 { 1.0 } else { tape.g7 },
        op8: tape.op8, a18: tape.a18, a28: tape.a28, v8: tape.v8,
        g8: if out == 8 { 1.0 } else { tape.g8 },
        op9: tape.op9, a19: tape.a19, a29: tape.a29, v9: tape.v9,
        g9: if out == 9 { 1.0 } else { tape.g9 },
        len: tape.len
    }

    // Process each slot from output to input
    let t9 = if t0.len > 9 { backward_step(t0, 9) } else { t0 }
    let t8 = if t9.len > 8 { backward_step(t9, 8) } else { t9 }
    let t7 = if t8.len > 7 { backward_step(t8, 7) } else { t8 }
    let t6 = if t7.len > 6 { backward_step(t7, 6) } else { t7 }
    let t5 = if t6.len > 5 { backward_step(t6, 5) } else { t6 }
    let t4 = if t5.len > 4 { backward_step(t5, 4) } else { t5 }
    let t3 = if t4.len > 3 { backward_step(t4, 3) } else { t4 }
    let t2 = if t3.len > 2 { backward_step(t3, 2) } else { t3 }
    let t1 = if t2.len > 1 { backward_step(t2, 1) } else { t2 }
    let t_final = if t1.len > 0 { backward_step(t1, 0) } else { t1 }

    return t_final
}

// ============================================================================
// 2-INPUT DENSE LAYER
// ============================================================================

// Dense layer: y = sigmoid(w1*x1 + w2*x2 + b)
// Tape layout: [0]=w1, [1]=w2, [2]=b, [3]=x1, [4]=x2,
//              [5]=w1*x1, [6]=w2*x2, [7]=sum, [8]=sum+b, [9]=sigmoid
struct Dense2Layer {
    w1: f64,
    w2: f64,
    bias: f64
}

fn dense2_new(w1: f64, w2: f64, b: f64) -> Dense2Layer {
    return Dense2Layer { w1: w1, w2: w2, bias: b }
}

// Forward pass: builds tape and computes output
fn dense2_forward(layer: Dense2Layer, x1: f64, x2: f64) -> Tape {
    let mut t = tape_new()
    t = tvar(t, layer.w1)      // [0] = w1
    t = tvar(t, layer.w2)      // [1] = w2
    t = tvar(t, layer.bias)    // [2] = b
    t = tvar(t, x1)            // [3] = x1
    t = tvar(t, x2)            // [4] = x2
    t = tmul(t, 0, 3)          // [5] = w1 * x1
    t = tmul(t, 1, 4)          // [6] = w2 * x2
    t = tadd(t, 5, 6)          // [7] = w1*x1 + w2*x2
    t = tadd(t, 7, 2)          // [8] = sum + b
    t = tsigmoid(t, 8)         // [9] = sigmoid(sum + b)
    return t
}

// Get output value
fn dense2_output(t: Tape) -> f64 {
    return get_v(t, 9)
}

// Get gradients after backward pass
fn dense2_grad_w1(t: Tape) -> f64 { return get_g(t, 0) }
fn dense2_grad_w2(t: Tape) -> f64 { return get_g(t, 1) }
fn dense2_grad_b(t: Tape) -> f64 { return get_g(t, 2) }
fn dense2_grad_x1(t: Tape) -> f64 { return get_g(t, 3) }
fn dense2_grad_x2(t: Tape) -> f64 { return get_g(t, 4) }

// Update layer weights
fn dense2_update(layer: Dense2Layer, dw1: f64, dw2: f64, db: f64, lr: f64) -> Dense2Layer {
    return Dense2Layer {
        w1: layer.w1 - lr * dw1,
        w2: layer.w2 - lr * dw2,
        bias: layer.bias - lr * db
    }
}

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

fn mse_loss(predicted: f64, target: f64) -> f64 {
    let diff = predicted - target
    return diff * diff
}

fn mse_loss_grad(predicted: f64, target: f64) -> f64 {
    return 2.0 * (predicted - target)
}

// ============================================================================
// TESTS
// ============================================================================

fn main() -> i32 {
    println("=== 2-Input Dense Layer Tests ===")
    println("")

    let tol = 0.01
    let mut ok = true

    // Test 1: Forward pass with zero inputs
    println("Test 1: Forward pass with x1=0, x2=0")
    let layer1 = dense2_new(1.0, 1.0, 0.0)
    let t1 = dense2_forward(layer1, 0.0, 0.0)
    let y1 = dense2_output(t1)
    // sigmoid(1*0 + 1*0 + 0) = sigmoid(0) = 0.5
    println("  y = ")
    println(y1)
    println("  expected = 0.5")
    if abs_f64(y1 - 0.5) > tol { ok = false; println("  FAIL") }
    println("")

    // Test 2: Forward pass with non-zero inputs
    println("Test 2: Forward pass with x1=1, x2=2")
    let layer2 = dense2_new(0.5, 0.3, 0.1)  // w1=0.5, w2=0.3, b=0.1
    let t2 = dense2_forward(layer2, 1.0, 2.0)
    let y2 = dense2_output(t2)
    // sigmoid(0.5*1 + 0.3*2 + 0.1) = sigmoid(1.2) ≈ 0.769
    let expected2 = 1.0 / (1.0 + exp_f64(0.0 - 1.2))
    println("  y = sigmoid(0.5*1 + 0.3*2 + 0.1) = ")
    println(y2)
    println("  expected = ")
    println(expected2)
    if abs_f64(y2 - expected2) > tol { ok = false; println("  FAIL") }
    println("")

    // Test 3: Backward pass gradients
    println("Test 3: Backward pass gradients")
    let layer3 = dense2_new(1.0, 2.0, 0.5)  // w1=1, w2=2, b=0.5
    let t3_fwd = dense2_forward(layer3, 0.5, 0.25)  // x1=0.5, x2=0.25
    let t3_bwd = backward(t3_fwd, 9)
    let y3 = dense2_output(t3_bwd)
    // y = sigmoid(1*0.5 + 2*0.25 + 0.5) = sigmoid(1.5) ≈ 0.818
    let sig3 = 1.0 / (1.0 + exp_f64(0.0 - 1.5))
    let dsig3 = sig3 * (1.0 - sig3)  // ≈ 0.149
    // dy/dw1 = dsig * x1 = 0.149 * 0.5 = 0.0745
    // dy/dw2 = dsig * x2 = 0.149 * 0.25 = 0.0373
    // dy/db = dsig * 1 = 0.149
    // dy/dx1 = dsig * w1 = 0.149 * 1 = 0.149
    // dy/dx2 = dsig * w2 = 0.149 * 2 = 0.298
    let dw1 = dense2_grad_w1(t3_bwd)
    let dw2 = dense2_grad_w2(t3_bwd)
    let db = dense2_grad_b(t3_bwd)
    let dx1 = dense2_grad_x1(t3_bwd)
    let dx2 = dense2_grad_x2(t3_bwd)
    println("  y = ")
    println(y3)
    println("  dL/dw1 = ")
    println(dw1)
    println("  expected = ")
    println(dsig3 * 0.5)
    println("  dL/dw2 = ")
    println(dw2)
    println("  expected = ")
    println(dsig3 * 0.25)
    println("  dL/db = ")
    println(db)
    println("  expected = ")
    println(dsig3)
    println("  dL/dx1 = ")
    println(dx1)
    println("  expected = ")
    println(dsig3 * 1.0)
    println("  dL/dx2 = ")
    println(dx2)
    println("  expected = ")
    println(dsig3 * 2.0)
    if abs_f64(dw1 - dsig3 * 0.5) > tol { ok = false; println("  FAIL: dw1") }
    if abs_f64(dw2 - dsig3 * 0.25) > tol { ok = false; println("  FAIL: dw2") }
    if abs_f64(db - dsig3) > tol { ok = false; println("  FAIL: db") }
    if abs_f64(dx1 - dsig3 * 1.0) > tol { ok = false; println("  FAIL: dx1") }
    if abs_f64(dx2 - dsig3 * 2.0) > tol { ok = false; println("  FAIL: dx2") }
    println("")

    // Test 4: Learn AND gate
    println("Test 4: Learn AND gate")
    println("  (0,0)->0, (0,1)->0, (1,0)->0, (1,1)->1")
    let mut layer4 = dense2_new(0.1, 0.1, 0.0)
    let lr = 5.0

    let mut epoch = 0
    while epoch < 10 {
        // (0, 0) -> 0
        let t_00 = dense2_forward(layer4, 0.0, 0.0)
        let y_00 = dense2_output(t_00)
        let t_00_bwd = backward(t_00, 9)
        let dl_00 = mse_loss_grad(y_00, 0.0)
        let dw1_00 = dense2_grad_w1(t_00_bwd) * dl_00
        let dw2_00 = dense2_grad_w2(t_00_bwd) * dl_00
        let db_00 = dense2_grad_b(t_00_bwd) * dl_00

        // (0, 1) -> 0
        let t_01 = dense2_forward(layer4, 0.0, 1.0)
        let y_01 = dense2_output(t_01)
        let t_01_bwd = backward(t_01, 9)
        let dl_01 = mse_loss_grad(y_01, 0.0)
        let dw1_01 = dense2_grad_w1(t_01_bwd) * dl_01
        let dw2_01 = dense2_grad_w2(t_01_bwd) * dl_01
        let db_01 = dense2_grad_b(t_01_bwd) * dl_01

        // (1, 0) -> 0
        let t_10 = dense2_forward(layer4, 1.0, 0.0)
        let y_10 = dense2_output(t_10)
        let t_10_bwd = backward(t_10, 9)
        let dl_10 = mse_loss_grad(y_10, 0.0)
        let dw1_10 = dense2_grad_w1(t_10_bwd) * dl_10
        let dw2_10 = dense2_grad_w2(t_10_bwd) * dl_10
        let db_10 = dense2_grad_b(t_10_bwd) * dl_10

        // (1, 1) -> 1
        let t_11 = dense2_forward(layer4, 1.0, 1.0)
        let y_11 = dense2_output(t_11)
        let t_11_bwd = backward(t_11, 9)
        let dl_11 = mse_loss_grad(y_11, 1.0)
        let dw1_11 = dense2_grad_w1(t_11_bwd) * dl_11
        let dw2_11 = dense2_grad_w2(t_11_bwd) * dl_11
        let db_11 = dense2_grad_b(t_11_bwd) * dl_11

        // Average gradients
        let avg_dw1 = (dw1_00 + dw1_01 + dw1_10 + dw1_11) / 4.0
        let avg_dw2 = (dw2_00 + dw2_01 + dw2_10 + dw2_11) / 4.0
        let avg_db = (db_00 + db_01 + db_10 + db_11) / 4.0

        layer4 = dense2_update(layer4, avg_dw1, avg_dw2, avg_db, lr)
        epoch = epoch + 1
    }

    println("  After 10 epochs:")
    println("  w1 = ")
    println(layer4.w1)
    println("  w2 = ")
    println(layer4.w2)
    println("  bias = ")
    println(layer4.bias)

    let pred_00 = dense2_output(dense2_forward(layer4, 0.0, 0.0))
    let pred_01 = dense2_output(dense2_forward(layer4, 0.0, 1.0))
    let pred_10 = dense2_output(dense2_forward(layer4, 1.0, 0.0))
    let pred_11 = dense2_output(dense2_forward(layer4, 1.0, 1.0))

    println("  Predictions:")
    println("  (0,0) -> ")
    println(pred_00)
    println("  (0,1) -> ")
    println(pred_01)
    println("  (1,0) -> ")
    println(pred_10)
    println("  (1,1) -> ")
    println(pred_11)

    // AND gate: (0,0), (0,1), (1,0) should be <0.5, (1,1) should be >0.5
    if pred_00 > 0.45 { ok = false; println("  FAIL: (0,0)") }
    if pred_01 > 0.45 { ok = false; println("  FAIL: (0,1)") }
    if pred_10 > 0.45 { ok = false; println("  FAIL: (1,0)") }
    if pred_11 < 0.55 { ok = false; println("  FAIL: (1,1)") }
    println("")

    // Test 5: Learn OR gate
    println("Test 5: Learn OR gate")
    println("  (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1")
    let mut layer5 = dense2_new(0.1, 0.1, 0.0)

    let mut epoch5 = 0
    while epoch5 < 10 {
        // (0, 0) -> 0
        let t_00 = dense2_forward(layer5, 0.0, 0.0)
        let y_00 = dense2_output(t_00)
        let t_00_bwd = backward(t_00, 9)
        let dl_00 = mse_loss_grad(y_00, 0.0)
        let dw1_00 = dense2_grad_w1(t_00_bwd) * dl_00
        let dw2_00 = dense2_grad_w2(t_00_bwd) * dl_00
        let db_00 = dense2_grad_b(t_00_bwd) * dl_00

        // (0, 1) -> 1
        let t_01 = dense2_forward(layer5, 0.0, 1.0)
        let y_01 = dense2_output(t_01)
        let t_01_bwd = backward(t_01, 9)
        let dl_01 = mse_loss_grad(y_01, 1.0)
        let dw1_01 = dense2_grad_w1(t_01_bwd) * dl_01
        let dw2_01 = dense2_grad_w2(t_01_bwd) * dl_01
        let db_01 = dense2_grad_b(t_01_bwd) * dl_01

        // (1, 0) -> 1
        let t_10 = dense2_forward(layer5, 1.0, 0.0)
        let y_10 = dense2_output(t_10)
        let t_10_bwd = backward(t_10, 9)
        let dl_10 = mse_loss_grad(y_10, 1.0)
        let dw1_10 = dense2_grad_w1(t_10_bwd) * dl_10
        let dw2_10 = dense2_grad_w2(t_10_bwd) * dl_10
        let db_10 = dense2_grad_b(t_10_bwd) * dl_10

        // (1, 1) -> 1
        let t_11 = dense2_forward(layer5, 1.0, 1.0)
        let y_11 = dense2_output(t_11)
        let t_11_bwd = backward(t_11, 9)
        let dl_11 = mse_loss_grad(y_11, 1.0)
        let dw1_11 = dense2_grad_w1(t_11_bwd) * dl_11
        let dw2_11 = dense2_grad_w2(t_11_bwd) * dl_11
        let db_11 = dense2_grad_b(t_11_bwd) * dl_11

        let avg_dw1 = (dw1_00 + dw1_01 + dw1_10 + dw1_11) / 4.0
        let avg_dw2 = (dw2_00 + dw2_01 + dw2_10 + dw2_11) / 4.0
        let avg_db = (db_00 + db_01 + db_10 + db_11) / 4.0

        layer5 = dense2_update(layer5, avg_dw1, avg_dw2, avg_db, lr)
        epoch5 = epoch5 + 1
    }

    println("  After 10 epochs:")
    let pred5_00 = dense2_output(dense2_forward(layer5, 0.0, 0.0))
    let pred5_01 = dense2_output(dense2_forward(layer5, 0.0, 1.0))
    let pred5_10 = dense2_output(dense2_forward(layer5, 1.0, 0.0))
    let pred5_11 = dense2_output(dense2_forward(layer5, 1.0, 1.0))

    println("  Predictions:")
    println("  (0,0) -> ")
    println(pred5_00)
    println("  (0,1) -> ")
    println(pred5_01)
    println("  (1,0) -> ")
    println(pred5_10)
    println("  (1,1) -> ")
    println(pred5_11)

    if pred5_00 > 0.45 { ok = false; println("  FAIL: (0,0)") }
    if pred5_01 < 0.55 { ok = false; println("  FAIL: (0,1)") }
    if pred5_10 < 0.55 { ok = false; println("  FAIL: (1,0)") }
    if pred5_11 < 0.55 { ok = false; println("  FAIL: (1,1)") }
    println("")

    if ok {
        println("ALL TESTS PASSED")
        return 0
    } else {
        println("SOME TESTS FAILED")
        return 1
    }
}
