// pbpk_tiny.d - Minimal test with only 10 RK4 steps

struct PBPKState {
    gut: f64,
    central: f64
}

struct PBPKDeriv {
    d_gut: f64,
    d_central: f64
}

struct PBPKStepResult {
    state_new: PBPKState
}

fn pbpk_ode(st: PBPKState, ka: f64, ke: f64) -> PBPKDeriv {
    return PBPKDeriv {
        d_gut: 0.0 - ka * st.gut,
        d_central: ka * st.gut - ke * st.central
    }
}

fn rk4_step(st: PBPKState, dt: f64, ka: f64, ke: f64) -> PBPKStepResult {
    let k1 = pbpk_ode(st, ka, ke)

    let st2 = PBPKState {
        gut: st.gut + 0.5 * dt * k1.d_gut,
        central: st.central + 0.5 * dt * k1.d_central
    }
    let k2 = pbpk_ode(st2, ka, ke)

    let st3 = PBPKState {
        gut: st.gut + 0.5 * dt * k2.d_gut,
        central: st.central + 0.5 * dt * k2.d_central
    }
    let k3 = pbpk_ode(st3, ka, ke)

    let st4 = PBPKState {
        gut: st.gut + dt * k3.d_gut,
        central: st.central + dt * k3.d_central
    }
    let k4 = pbpk_ode(st4, ka, ke)

    return PBPKStepResult {
        state_new: PBPKState {
            gut: st.gut + (dt / 6.0) * (k1.d_gut + 2.0*k2.d_gut + 2.0*k3.d_gut + k4.d_gut),
            central: st.central + (dt / 6.0) * (k1.d_central + 2.0*k2.d_central + 2.0*k3.d_central + k4.d_central)
        }
    }
}

fn main() -> i32 {
    println("=== PBPK Tiny Test (10 steps) ===")
    println("")

    let ka = 1.0
    let ke = 0.1
    let dt = 0.1

    let mut st = PBPKState { gut: 500.0, central: 0.0 }

    println("Initial: Gut=500, Central=0")
    println("")

    // Take exactly 10 RK4 steps
    let mut i = 0
    while i < 10 {
        let result = rk4_step(st, dt, ka, ke)
        st = result.state_new
        i = i + 1

        // Print every 2 steps to monitor progress
        if i == 2 || i == 5 || i == 10 {
            println("After step")
            println(i)
            println("  Gut:")
            println(st.gut)
            println("  Central:")
            println(st.central)
            println("")
        }
    }

    println("Final (t=1h): Gut<500, Central>0")

    let gut_decreased = st.gut < 500.0 && st.gut > 0.0
    let central_increased = st.central > 0.0

    if gut_decreased && central_increased {
        println("TEST PASSED")
        return 0
    } else {
        println("TEST FAILED")
        return 1
    }
}
