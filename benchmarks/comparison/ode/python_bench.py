"""Benchmark: RK4 ODE Solver — Python equivalent.

Hand-rolled RK4 (no SciPy) for fair comparison.
"""
import time
import math


def bench_exp_decay(n_steps: int) -> float:
    dt = 10.0 / n_steps
    u, t = 1.0, 0.0
    for _ in range(n_steps):
        k1 = -0.1 * u
        k2 = -0.1 * (u + 0.5 * dt * k1)
        k3 = -0.1 * (u + 0.5 * dt * k2)
        k4 = -0.1 * (u + dt * k3)
        u += dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
    return u


def bench_harmonic(n_steps: int) -> float:
    dt = 2.0 * math.pi / n_steps
    x, v, t = 1.0, 0.0, 0.0
    for _ in range(n_steps):
        k1x, k1v = v, -x
        k2x = v + 0.5*dt*k1v
        k2v = -(x + 0.5*dt*k1x)
        k3x = v + 0.5*dt*k2v
        k3v = -(x + 0.5*dt*k2x)
        k4x = v + dt*k3v
        k4v = -(x + dt*k3x)
        x += dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
        v += dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
        t += dt
    return x


def main():
    n = 100_000

    # Benchmark 1: Exponential decay
    t0 = time.perf_counter()
    result_decay = bench_exp_decay(n)
    t1 = time.perf_counter()
    exact_decay = math.exp(-1.0)
    err_decay = abs(result_decay - exact_decay)
    ms_decay = (t1 - t0) * 1000
    print(f"RK4 exp-decay  ({n} steps): error = {err_decay:.2e}, time = {ms_decay:.2f} ms")

    # Benchmark 2: Harmonic oscillator
    t0 = time.perf_counter()
    result_harm = bench_harmonic(n)
    t1 = time.perf_counter()
    err_harm = abs(result_harm - 1.0)
    ms_harm = (t1 - t0) * 1000
    print(f"RK4 harmonic   ({n} steps): error = {err_harm:.2e}, time = {ms_harm:.2f} ms")

    print(f"\nTotal: {ms_decay + ms_harm:.2f} ms")


if __name__ == "__main__":
    main()
