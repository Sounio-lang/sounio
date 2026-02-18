# Benchmark: RK4 ODE Solver — Julia equivalent
# Hand-rolled RK4 (no DifferentialEquations.jl) for fair comparison.

function rk4_step_decay(u, t, dt)
    f(u) = -0.1 * u
    k1 = f(u)
    k2 = f(u + 0.5 * dt * k1)
    k3 = f(u + 0.5 * dt * k2)
    k4 = f(u + dt * k3)
    return u + dt / 6.0 * (k1 + 2k2 + 2k3 + k4), t + dt
end

function bench_exp_decay(n_steps)
    dt = 10.0 / n_steps
    u, t = 1.0, 0.0
    for _ in 1:n_steps
        u, t = rk4_step_decay(u, t, dt)
    end
    return u
end

function rk4_step_harmonic(x, v, t, dt)
    fx(x, v) = v
    fv(x, v) = -x

    k1x = fx(x, v)
    k1v = fv(x, v)
    k2x = fx(x + 0.5dt*k1x, v + 0.5dt*k1v)
    k2v = fv(x + 0.5dt*k1x, v + 0.5dt*k1v)
    k3x = fx(x + 0.5dt*k2x, v + 0.5dt*k2v)
    k3v = fv(x + 0.5dt*k2x, v + 0.5dt*k2v)
    k4x = fx(x + dt*k3x, v + dt*k3v)
    k4v = fv(x + dt*k3x, v + dt*k3v)

    xn = x + dt/6 * (k1x + 2k2x + 2k3x + k4x)
    vn = v + dt/6 * (k1v + 2k2v + 2k3v + k4v)
    return xn, vn, t + dt
end

function bench_harmonic(n_steps)
    dt = 2π / n_steps
    x, v, t = 1.0, 0.0, 0.0
    for _ in 1:n_steps
        x, v, t = rk4_step_harmonic(x, v, t, dt)
    end
    return x
end

function main()
    n = 100_000

    # Warmup
    bench_exp_decay(10)
    bench_harmonic(10)

    # Benchmark 1: Exponential decay
    t1 = @elapsed result_decay = bench_exp_decay(n)
    exact_decay = exp(-1.0)
    err_decay = abs(result_decay - exact_decay)
    println("RK4 exp-decay  ($n steps): error = $err_decay, time = $(t1*1000) ms")

    # Benchmark 2: Harmonic oscillator
    t2 = @elapsed result_harm = bench_harmonic(n)
    err_harm = abs(result_harm - 1.0)
    println("RK4 harmonic   ($n steps): error = $err_harm, time = $(t2*1000) ms")

    println("\nTotal: $((t1+t2)*1000) ms")
end

main()
