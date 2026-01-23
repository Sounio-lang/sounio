// pbpk_test_simple.d - Ultra-simple test with minimal loop iterations
//
// Test the 2-compartment PBPK with very few steps to debug hanging issue

struct PBPKState {
    gut: f64,
    central: f64
}

struct PBPKDeriv {
    d_gut: f64,
    d_central: f64
}

fn pbpk_ode(st: PBPKState, ka: f64, ke: f64) -> PBPKDeriv {
    return PBPKDeriv {
        d_gut: 0.0 - ka * st.gut,
        d_central: ka * st.gut - ke * st.central
    }
}

fn rk4_step(st: PBPKState, dt: f64, ka: f64, ke: f64) -> PBPKState {
    // Stage 1
    let k1 = pbpk_ode(st, ka, ke)

    // Stage 2
    let st2_gut = st.gut + 0.5 * dt * k1.d_gut
    let st2_central = st.central + 0.5 * dt * k1.d_central
    let st2 = PBPKState { gut: st2_gut, central: st2_central }
    let k2 = pbpk_ode(st2, ka, ke)

    // Stage 3
    let st3_gut = st.gut + 0.5 * dt * k2.d_gut
    let st3_central = st.central + 0.5 * dt * k2.d_central
    let st3 = PBPKState { gut: st3_gut, central: st3_central }
    let k3 = pbpk_ode(st3, ka, ke)

    // Stage 4
    let st4_gut = st.gut + dt * k3.d_gut
    let st4_central = st.central + dt * k3.d_central
    let st4 = PBPKState { gut: st4_gut, central: st4_central }
    let k4 = pbpk_ode(st4, ka, ke)

    // Combine
    let new_gut = st.gut + (dt / 6.0) * (k1.d_gut + 2.0*k2.d_gut + 2.0*k3.d_gut + k4.d_gut)
    let new_central = st.central + (dt / 6.0) * (k1.d_central + 2.0*k2.d_central + 2.0*k3.d_central + k4.d_central)

    return PBPKState { gut: new_gut, central: new_central }
}

fn main() -> i32 {
    println("=== Simple PBPK Test ===")
    println("")

    let ka = 1.0
    let ke = 0.1
    let dt = 0.1

    let mut st = PBPKState { gut: 500.0, central: 0.0 }

    println("Initial state:")
    println("  Gut:")
    println(st.gut)
    println("  Central:")
    println(st.central)
    println("")

    // Take just 10 steps
    let max_steps = 10
    let mut i = 0

    while i < max_steps {
        st = rk4_step(st, dt, ka, ke)
        i = i + 1
    }

    println("After 10 steps (t=1h):")
    println("  Gut:")
    println(st.gut)
    println("  Central:")
    println(st.central)
    println("")

    // Expected: gut should decrease, central should increase
    let gut_decreased = st.gut < 500.0
    let central_increased = st.central > 0.0

    if gut_decreased && central_increased {
        println("TEST PASSED")
        return 0
    } else {
        println("TEST FAILED")
        return 1
    }
}
