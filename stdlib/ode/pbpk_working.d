// pbpk_working.d - Working 2-compartment PBPK using proven RK4 pattern

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
    cmax: f64,
    tmax: f64,
    auc: f64,
    nsteps: i64
}

fn solve_pbpk(init: PBPKState, p: PBPKParams, t_end: f64, n_steps: i64) -> PBPKSolution {
    let dt = t_end / (n_steps as f64)
    let mut st = init
    let mut i: i64 = 0
    let mut auc = 0.0
    let mut cmax = 0.0
    let mut tmax = 0.0

    while i < n_steps {
        // Current concentration
        let c_curr = st.central / p.vd

        // Update Cmax tracking
        if c_curr > cmax {
            cmax = c_curr
            tmax = (i as f64) * dt
        }

        // Take step
        let result = rk4_step_pbpk(st, dt, p)
        st = result.state_new

        // Update AUC using trapezoidal rule
        let c_next = st.central / p.vd
        auc = auc + 0.5 * dt * (c_curr + c_next)

        i = i + 1
    }

    let c_final = st.central / p.vd

    return PBPKSolution {
        gut: st.gut,
        central: st.central,
        concentration: c_final,
        cmax: cmax,
        tmax: tmax,
        auc: auc,
        nsteps: i
    }
}

// Helper functions
fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn exp_approx(x: f64) -> f64 {
    let x2 = x * x
    let x3 = x2 * x
    let x4 = x3 * x
    let x5 = x4 * x
    let x6 = x5 * x
    let x7 = x6 * x
    let x8 = x7 * x
    return 1.0 + x + x2/2.0 + x3/6.0 + x4/24.0 + x5/120.0 + x6/720.0 + x7/5040.0 + x8/40320.0
}

fn one_comp_analytical(dose: f64, ka: f64, ke: f64, vd: f64, t: f64) -> f64 {
    let factor = dose * ka / (vd * (ka - ke))
    let exp_ke = exp_approx(0.0 - ke * t)
    let exp_ka = exp_approx(0.0 - ka * t)
    return factor * (exp_ke - exp_ka)
}

fn main() -> i32 {
    println("=== Demetrios 2-Compartment PBPK Model ===")
    println("")
    println("Model: Gut (absorption) -> Central (elimination)")
    println("  Parameters: ka=1/h, ke=0.1/h, Vd=50L")
    println("  Dose: 500 mg oral")
    println("  Simulation: 0 -> 24h with 240 steps (dt=0.1h)")
    println("")

    let s0 = PBPKState {
        gut: 500.0,
        central: 0.0
    }

    let p = default_params()

    // Solve with RK4
    let sol = solve_pbpk(s0, p, 24.0, 240)

    println("Results at t=24h:")
    println("  Gut (mg):")
    println(sol.gut)
    println("  Central (mg):")
    println(sol.central)
    println("  Concentration (mg/L):")
    println(sol.concentration)
    println("")

    println("PK Metrics:")
    println("  Cmax (mg/L):")
    println(sol.cmax)
    println("  Tmax (h):")
    println(sol.tmax)
    println("  AUC_0-24h (mg*h/L):")
    println(sol.auc)
    println("  Steps:")
    println(sol.nsteps)
    println("")

    // Validate against analytical solution
    let c_analytical = one_comp_analytical(500.0, p.ka, p.ke, p.vd, 24.0)
    let err = abs_f64(sol.concentration - c_analytical)
    let rel_err = err / c_analytical

    println("Validation:")
    println("  Analytical C(24h):")
    println(c_analytical)
    println("  Relative error:")
    println(rel_err)
    println("")

    // Mass balance check
    let total = sol.gut + sol.central
    let eliminated = 500.0 - total
    println("Mass Balance:")
    println("  Remaining (mg):")
    println(total)
    println("  Eliminated (mg):")
    println(eliminated)
    println("  Fraction eliminated:")
    println(eliminated / 500.0)
    println("")

    // Validation criteria
    let gut_absorbed = sol.gut < 0.01
    let accurate = rel_err < 0.001
    let cmax_reasonable = sol.cmax > 5.0 && sol.cmax < 15.0
    let tmax_reasonable = sol.tmax > 0.5 && sol.tmax < 5.0
    let auc_positive = sol.auc > 0.0

    let all_valid = gut_absorbed && accurate && cmax_reasonable && tmax_reasonable && auc_positive

    if all_valid && sol.nsteps == 240 {
        println("TEST PASSED: 2-compartment PBPK working correctly")
        println("  - Absorption complete")
        println("  - Matches analytical solution (< 0.1% error)")
        println("  - PK metrics computed correctly")
        return 0
    } else {
        println("TEST FAILED")
        if !gut_absorbed { println("  - Gut absorption incomplete") }
        if !accurate { println("  - Accuracy insufficient") }
        if !cmax_reasonable { println("  - Cmax unreasonable") }
        if !tmax_reasonable { println("  - Tmax unreasonable") }
        if !auc_positive { println("  - AUC not computed") }
        return 1
    }
}
