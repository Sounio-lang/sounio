# ODE factory and PBPK14 stability repair

The original ode/test_tsit5_e2e.sio and ode/test_pbpk14_rk4_e2e.sio remain
byte-identical. Their original inputs, numerical assertions, tolerances and
requested integration intervals are retained.

Slurm18476 isolated two failures on the frozen original source. The default
adaptive scalar solver completed, but high_accuracy_options produced a
10-million-attempt limit that the solver's existing one-million validation
boundary rejected immediately (status3). The factory now uses that accepted
one-million budget. Its tighter relative/absolute tolerances remain unchanged;
no validation boundary was relaxed.

The PBPK14 original test reached its 24-hour/100-interval elimination assertion
and failed. The unmodified coarse RK4 model is numerically unstable at this
step size. The independent C++23 replay also exhibits nonphysical negative or
enormous compartment amounts; its coarse partial-compartment sum alone is not
a valid physical correctness criterion. The first reference attempt18482
incorrectly assumed that this partial sum had to exceed80 to demonstrate the
instability. It failed that reference assertion and is preserved. The corrected
reference tests actual nonnegative/finite/mass constraints instead. No original
Sounio assertion or expected value changed.

The solver now retains the requested outer intervals and subdivides each one
using four times the largest removal diagonal of the unchanged linear PBPK
model. Every accepted substep therefore has h*max_removal at most one quarter.
All original flows, volumes, partition coefficients, absorption and elimination
equations are retained. The raw rk4_step_pbpk14 routine remains a single RK4
step; the bounded stability policy belongs to solve_pbpk14_rk4.

PBPK14Solution now exposes all14 compartment amounts, status, t_final, the
completed outer interval count nsteps, actual internal_steps and nfeval.
An invalid request does not report successful integration. Status0 means
complete,1 internal budget,2 no time progress,3 invalid input,4 arithmetic.
At most one million outer intervals and ten million internal steps are allowed.
The existing arterial filtration formulation must retain a nonnegative
arterial-to-kidney coefficient; unsupported parameter regimes are explicitly
refused. No negative amount is clipped into a successful result. A failure
during integration returns the last valid state and completed work counters.

The independent C++23 reference constructs the linear operator directly from
the model and evaluates its exponential action using positive uniformization
with small Poisson chunks. It does not call the Sounio solver or use its RK4
implementation for the native concordance oracle. Its selftest additionally
compares a separate C++ RK4 implementation against this matrix-exponential
reference and rejects the original coarse physical state. Both implementations
use the same stated model parameters; this establishes numerical concordance
for the exercised model, rather than clinical calibration of that model.

Slurm18484 completed the two unchanged original tests, the generated13-case
PBPK14 fixture and the scalar solver diagnostic, each check/compile/native0.
All14 compartments agreed with the independent matrix-exponential calculation
at the fixed1e-8 absolute threshold; positivity, total-mass bounds, status and
step accounting controls passed. The original scalar solver diagnostic agreed
with C++23 std::exp for both default and high-accuracy options. Source, compiler,
runtime, generated fixture and executable custody guards passed.

Slurm18488 regenerated the checked PBPK14 fixture and the three-factory scalar
fixture from C++23. The reusable pbpk14_native_gate.sh additionally requires
byte-identical regeneration, all three option factories, the unchanged
one-million budget refusal and rejection of a deliberately wrong output.
Its first launch18492 failed before any test phase because job submission raced
the source archive transport. That setup witness and its erratum are preserved.
The full reusable-gate result is recorded separately after qualification.


Slurm18494 executed every planned reusable-gate phase successfully, including
both unchanged originals, the13 PBPK cases, all3 scalar option factories,
independent numerical concordance and the expected wrong-value refusal.
All source/tool/ELF/derived/artifact guards passed and gate STATUS was0.
The outer job nevertheless ended FAILED2:0 because its subsequent grep tried
to write a verification match into its own Slurm stdout file. The reusable
gate script itself does not perform that wrapper operation. Slurm18498 then
completed0:0, verified the exact complete phase table and gate marker, and
rechecked every preserved hash. Its scope is explicitly a custody supplement,
rather than a new numerical run. The failed wrapper receipt remains intact.
