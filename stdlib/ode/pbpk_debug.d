// pbpk_debug.d - Debug why RK4 step isn't working

struct PBPKState {
    gut: f64,
    central: f64
}

struct PBPKDeriv {
    d_gut: f64,
    d_central: f64
}

fn pbpk_ode(st: PBPKState, ka: f64, ke: f64) -> PBPKDeriv {
    let dg = 0.0 - ka * st.gut
    let dc = ka * st.gut - ke * st.central
    return PBPKDeriv {
        d_gut: dg,
        d_central: dc
    }
}

fn main() -> i32 {
    println("=== PBPK Debug ===")
    println("")

    let ka = 1.0
    let ke = 0.1
    let dt = 0.1

    let st = PBPKState { gut: 500.0, central: 0.0 }

    println("Initial state:")
    println("  Gut: 500, Central: 0")
    println("")

    // Test the ODE function
    println("Testing ODE function:")
    let deriv = pbpk_ode(st, ka, ke)
    println("  d_gut:")
    println(deriv.d_gut)
    println("  d_central:")
    println(deriv.d_central)
    println("")

    // Expected: d_gut = -500, d_central = 500
    println("Expected: d_gut = -500, d_central = 500")
    println("")

    // Test Euler step manually
    let gut_new = st.gut + dt * deriv.d_gut
    let central_new = st.central + dt * deriv.d_central

    println("After Euler step (dt=0.1):")
    println("  Gut:")
    println(gut_new)
    println("  Central:")
    println(central_new)
    println("")

    println("Expected: Gut = 450, Central = 50")
    println("")

    // Validate
    let gut_ok = gut_new > 400.0 && gut_new < 460.0
    let central_ok = central_new > 40.0 && central_new < 60.0

    if gut_ok && central_ok {
        println("TEST PASSED - ODE function works correctly")
        return 0
    } else {
        println("TEST FAILED - ODE function not working")
        return 1
    }
}
