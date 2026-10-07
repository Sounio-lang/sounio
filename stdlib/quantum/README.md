# stdlib/quantum

Exact quantum simulation on a classical statevector. No noise model, no shot
noise and no hardware backend.

- `linear.sio`: a linear `Qubit` type. A qubit value must be used exactly once,
  which is the type-system analogue of no-cloning and no-deleting. Cap
  `QUANTUM_MAX_QUBITS = 10` (1024 amplitudes); the 11th `qubit_alloc` panics.
  Gates: H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CZ, SWAP, Toffoli. Circuits: Bell,
  GHZ3, teleportation.
  Measurement is `measure_qubit` (not `measure`: that name is the builtin
  Knowledge constructor).
- `vqe.sio`: Pauli-sum Hamiltonians, gate-list circuits, exact expectation
  values, an exact ground-state oracle (Jacobi) and a Rotosolve VQE driver.
  Cap: 4 qubits.
- `epistemic_vqe.sio`: the two-qubit H2 VQE (O'Malley et al., PRX 6, 031007
  (2016)) with a second-order GUM band on the gate angles, validated against
  Monte Carlo. Cap: 4 qubits.

Rx/Ry/Rz also exist on `Statevector` in `epistemic_vqe.sio`. The two simulators
are not unified.

`epistemic_linear.sio` runs parameter-shift GUM on linear `ry`. The angle
is an `Epistemic`: its variance is a struct field and survives a call.
Builtin `Knowledge<f64>` drops its uncertainty when passed as an argument;
deposit the band into `Epistemic` in the function that built it.
At θ = π the first-order band is zero. Teleportation leaves both the
probability and its derivative unchanged.

`born_channel.sio` keeps those two zeros apart. A filled channel has a numeric
read. An empty channel does not: `band_variance` panics, and the unread payload
is a poison, not a variance. `born_prepared` fills the angle channel and leaves
the shot channel empty. `born_one_shot` does the opposite, after one
`measure_qubit`. One `Epistemic` on two wires fills the covariance with
∂a ∂b σ². Two `Epistemic` values with the same numbers leave that covariance
empty, and `pair_var_sum` does not treat the hole as zero.

`born_shot_law.sio` separates the two variances a shot can carry. For any
mixture of Bernoullis the variance of one bit is μ(1−μ) and the width Var(P)
cancels. That width is the covariance of two shots that shared P.
`ry_shot_law` computes μ and Var(P) by 16-point Gauss–Hermite quadrature of
linear Ry. At θ = π/2 the one-shot variance stays 1/4. At θ = π it moves with
the bias of the mean, and Var(P) stays in the shared covariance.

`ry_product_cov` reads that covariance off two wires that spend the same angle and are not entangled. `ry_cnot_cov` spends the same angle and then a CNOT: at pi/2 the control marginal stays 1/2, the bit covariance is the computed zero, and the target marginal shifts by twice the width.
The same quadrature splits that covariance: `given_angle` is the mean of the conditional covariance, `across_angles` is the covariance of the two conditional means. At pi/2 with a CNOT both read the computed zero. At pi with sigma 0.1 the conditional piece is negative, about -0.00244, and the across-angle piece stays near -0.000025.
`target_read` is the variance a control measurement removes from the target. For a two-bit joint it equals cov squared over Var(control). At pi/2 that removal is about 2.5e-9. `ry_cz_cov` matches the product probabilities: the phase on |11> does not move the removal. CNOT at the pole removes about 0.00244.
At the pole, mu(1-mu)/Var(P) equals 2/(1-exp(-sigma^2)), which is 201.00 at sigma 0.1, not the leading 200. At pi/2 the third central moment of P is the computed zero, so three shared shots add nothing. A two-point angle matched to the same Var(P) keeps that zero and the pair, and splits on the fourth moment: Gaussian 1.87e-9 against two-point 6.25e-10.
That fourth-moment gap is Var(P(1-P)). For a symmetric angle it cannot move the law of 1, 2, or 3 shared shots. The four-shot count laws differ by delta (s-1)^4, and the total variation is 8|delta|. At pi/2 with sigma 0.01 the two-point conditional variance is 0 and the Gaussian record sits 9.996e-9 away.
That distance is half the gap in four-shot parity. E[(1-2P)^4] equals E[(-1)^K] and equals 16 E[Z^4]. Three-shot parity is the computed zero. Six-shot parity is E[sin^6], about 1.5e-11, and the degree-4 instrument does not read it.
For a symmetric shared angle, U=Z^2 stays in [0, 1/4], so the four-shot parity stays in [16 v^2, 4v]. At sigma 0.01 the Gaussian uses a fraction 0.000200 of that interval. Four product Ry(pi/2) wires read parity 0. GHZ on four wires has the same marginal 1/2 and parity 1, outside the interval.
That Z parity is also what a copied computational bit does. ry_chsh_exit turns both wires: the copy then reads 0 in X while the Bell pair reads 1 in Z and in X. CHSH on the pair is 2*sqrt(2). On each computational-basis state it is plus or minus sqrt(2).
With one offset per party the whole CHSH number is 2*sqrt(2)*cos(delta_a-delta_b) at every realization. A shared offset leaves it at 2*sqrt(2). Independent Gaussian offsets of variance sigma^2 have mean 2*sqrt(2)*exp(-sigma^2), which the circuit reads as 2 at sigma = sqrt(ln 2 / 2).
Under that same party offset the copied bit is exactly half the pair, and the anticorrelated basis state is minus half. A split between one party's two settings leaves the copy at sqrt(2) and moves only the pair, as sqrt(2)*(1+cos epsilon).
With both moves at once the pair reads sqrt(2)*(cos(eta)+cos(eta+epsilon)). The factor 2 is epsilon = 0. At Bob's offset pi/4 and Alice's split pi/2 both read 2, and past that point the copy is larger while the pair is below 2.
Ry(phi) then CNOT keeps ZZ at 1 and sets XX to sin phi. The usual angles read sqrt(2)*(1+sin phi) and stay below 2 until sin phi = sqrt(2)-1. Bob angles with tan beta = sin phi read 2*sqrt(1+sin(phi)^2) and pass 2 for every phi other than 0.
qasm_tape records a circuit that has not run. tape_replay checks it on the linear statevector. tape_qasm emits OpenQASM 3.0, including stdgates.inc, for a submitter outside the language.
tape_counts draws N shots from those replay probabilities.

Demo, independent C++ cross-check and the full list of limits:
[`demos/quantum/`](../../demos/quantum/README.md).
