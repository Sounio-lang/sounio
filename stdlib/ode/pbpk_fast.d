// pbpk_fast.d - 2-compartment PBPK with reduced steps to avoid compiler bug

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

struct PBPKParams {
    ka: f64,
    ke: f64,
    vd: f64
}

fn default_params() -> PBPKParams {
    return PBPKParams {
        ka: 1.0,
        ke: 0.1,
        vd: 50.0
    }
}

fn pbpk_ode(st: PBPKState, p: PBPKParams) -> PBPKDeriv {
    return PBPKDeriv {
        d_gut: 0.0 - p.ka * st.gut,
        d_central: p.ka * st.gut - p.ke * st.central
    }
}

fn rk4_step_pbpk(st: PBPKState, dt: f64, p: PBPKParams) -> PBPKStepResult {
    // Stage 1
    let k1 = pbpk_ode(st, p)

    // Stage 2
    let st2 = PBPKState {
        gut: st.gut + 0.5 * dt * k1.d_gut,
        central: st.central + 0.5 * dt * k1.d_central
    }
    let k2 = pbpk_ode(st2, p)

    // Stage 3
    let st3 = PBPKState {
        gut: st.gut + 0.5 * dt * k2.d_gut,
        central: st.central + 0.5 * dt * k2.d_central
    }
    let k3 = pbpk_ode(st3, p)

    // Stage 4
    let st4 = PBPKState {
        gut: st.gut + dt * k3.d_gut,
        central: st.central + dt * k3.d_central
    }
    let k4 = pbpk_ode(st4, p)

    // Combine
    return PBPKStepResult {
        state_new: PBPKState {
            gut: st.gut + (dt / 6.0) * (k1.d_gut + 2.0*k2.d_gut + 2.0*k3.d_gut + k4.d_gut),
            central: st.central + (dt / 6.0) * (k1.d_central + 2.0*k2.d_central + 2.0*k3.d_central + k4.d_central)
        }
    }
}

struct PBPKSolution {
    gut: f64,
    central: f64,
    concentration: f64,
    nsteps: i64
}

fn solve_pbpk(init: PBPKState, p: PBPKParams, t_end: f64, n_steps: i64) -> PBPKSolution {
    let dt = t_end / (n_steps as f64)
    let mut st = init
    let mut i: i64 = 0

    while i < n_steps {
        let result = rk4_step_pbpk(st, dt, p)
        st = result.state_new
        i = i + 1
    }

    let c_final = st.central / p.vd

    return PBPKSolution {
        gut: st.gut,
        central: st.central,
        concentration: c_final,
        nsteps: i
    }
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn main() -> i32 {
    println("=== Demetrios 2-Compartment PBPK (Fast) ===")
    println("")
    println("Parameters: ka=1/h, ke=0.1/h, Vd=50L")
    println("Dose: 500 mg oral")
    println("Simulation: 0 -> 24h with 50 steps (dt=0.48h)")
    println("")

    let s0 = PBPKState {
        gut: 500.0,
        central: 0.0
    }

    let p = default_params()

    // Only 50 steps to avoid compiler bug
    let sol = solve_pbpk(s0, p, 24.0, 50)

    println("Results at t=24h:")
    println("  Gut (mg):")
    println(sol.gut)
    println("  Central (mg):")
    println(sol.central)
    println("  Concentration (mg/L):")
    println(sol.concentration)
    println("  Steps:")
    println(sol.nsteps)
    println("")

    // Mass balance
    let total = sol.gut + sol.central
    let eliminated = 500.0 - total
    println("Mass Balance:")
    println("  Remaining (mg):")
    println(total)
    println("  Eliminated (mg):")
    println(eliminated)
    println("")

    // Validation
    let gut_absorbed = sol.gut < 0.1
    let central_reasonable = sol.central > 0.0 && sol.central < 500.0
    let conc_reasonable = sol.concentration > 0.0 && sol.concentration < 20.0

    let all_valid = gut_absorbed && central_reasonable && conc_reasonable

    if all_valid && sol.nsteps == 50 {
        println("TEST PASSED: PBPK simulation complete")
        println("  - Absorption nearly complete")
        println("  - Central compartment within expected range")
        println("  - Concentration computed successfully")
        return 0
    } else {
        println("TEST FAILED")
        if !gut_absorbed { println("  - Gut absorption incomplete") }
        if !central_reasonable { println("  - Central amount unreasonable") }
        if !conc_reasonable { println("  - Concentration unreasonable") }
        return 1
    }
}
