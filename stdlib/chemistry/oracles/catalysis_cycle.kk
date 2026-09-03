/* catalysis_cycle.kk
   ============================================================================
   Independent Koka oracle for the classic single-substrate enzyme mechanism

       E + S <=> ES -> E + P          (rate constants kf, kr, kcat)

   This program is a from-scratch, from-the-ODE reimplementation of the
   mass-action kinetics used in Sounio's `stdlib/chemistry/kinetics.sio`
   (`enzyme_rhs` / `simulate_enzyme_crn` / `test_enzyme_crn`). It does not
   read, call, or translate any Sounio source -- it exists to be diffed
   against Sounio's numeric output for matching inputs, as an independent
   numerical check written in a different language/paradigm (Koka, with
   explicit algebraic effects), per the "founder order" that every
   validation oracle for this suite must be C++23, Koka, Lean4, F-star, or
   F# -- never Python or Rust.

   Mass-action ODEs (E, S, ES, P are concentrations):

       fwd = kf * E * S
       rev = kr * ES
       cat = kcat * ES
       dE/dt  = -fwd + rev + cat
       dES/dt =  fwd - rev - cat
       dS/dt  = -fwd + rev
       dP/dt  =  cat

   Integrated with a hand-written, fixed-step classical RK4 (4 explicit
   stages, no adaptive/library ODE solver -- fully auditable), matching the
   numerical method Sounio's own oracle also hand-rolls rather than calling
   an opaque solver.

   Effect discipline (the specific thing Koka is good at):
     - `simulate` (the RK4 integration of the pure ODE system) is a totally
       pure function: its type carries no effect row at all. There is
       nothing in the mass-action RHS or the RK4 stage arithmetic that can
       divide by zero, so this stays effect-free by construction.
     - Turning `product_formed` into TON = product/E0 and TOF = TON/elapsed
       *does* involve division, and both denominators (E0, elapsed) can be
       (near-)zero for degenerate inputs. That danger is captured as a
       user-defined algebraic effect `divby0`, so "this computation might
       fail to divide" is visible directly in the function's type signature
       (`... : <divby0> float64`), not just guarded by a runtime `if`. A
       caller cannot even invoke `metrics` without either handling
       `divby0` or propagating it in their own effect row -- the compiler
       enforces it.
     - Printing is a separate, explicit `:console` effect, layered on top
       via `with` handlers -- never mixed silently into the pure numerics.
   ============================================================================
*/
module catalysis_cycle

import std/num/float64

// ----------------------------------------------------------------------------
// Reaction state: (E, S, ES, P) concentrations, in that fixed order.
// ----------------------------------------------------------------------------
alias rxstate = (float64, float64, float64, float64)

// ----------------------------------------------------------------------------
// Pure mass-action right-hand side for E + S <=> ES -> E + P.
// No effects: this is ordinary total arithmetic (multiply/add/subtract).
// ----------------------------------------------------------------------------
fun rhs(y : rxstate, kf : float64, kr : float64, kcat : float64) : rxstate
  val (e, s, es, _p) = y
  val fwd = kf * e * s
  val rev = kr * es
  val cat = kcat * es
  ( 0.0 - fwd + rev + cat,   // dE/dt
    0.0 - fwd + rev,         // dS/dt
    fwd - rev - cat,         // dES/dt
    cat )                    // dP/dt

// y + scale * d, componentwise over the 4-tuple state.
fun axpy(y : rxstate, d : rxstate, scale : float64) : rxstate
  val (y1, y2, y3, y4) = y
  val (d1, d2, d3, d4) = d
  (y1 + scale * d1, y2 + scale * d2, y3 + scale * d3, y4 + scale * d4)

// ----------------------------------------------------------------------------
// One classical 4th-order Runge-Kutta step, all four stages written out
// explicitly (no adaptive/library ODE solver) so the method is fully
// transparent and auditable:
//
//   k1 = f(y)
//   k2 = f(y + dt/2 * k1)
//   k3 = f(y + dt/2 * k2)
//   k4 = f(y + dt   * k3)
//   y' = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
// ----------------------------------------------------------------------------
fun rk4-step(y : rxstate, dt : float64, kf : float64, kr : float64, kcat : float64) : rxstate
  val k1 = rhs(y, kf, kr, kcat)
  val k2 = rhs(axpy(y, k1, dt / 2.0), kf, kr, kcat)
  val k3 = rhs(axpy(y, k2, dt / 2.0), kf, kr, kcat)
  val k4 = rhs(axpy(y, k3, dt), kf, kr, kcat)
  // weighted-sum = k1 + 2*k2 + 2*k3 + k4
  val wsum = axpy(axpy(axpy(k1, k2, 2.0), k3, 2.0), k4, 1.0)
  axpy(y, wsum, dt / 6.0)

// ----------------------------------------------------------------------------
// PURE numeric core: fixed-step RK4 integration of the enzyme mechanism
// from (E0, S0, 0, 0) for `steps` steps of size `dt`.
//
// Note the type signature: the *only* effect is `:div` (Koka's built-in
// totality-tracking effect for functions whose termination the compiler
// cannot prove statically, such as this bounded `while` loop) -- there is
// no `:exn`, no `:console`, no `:divby0`, nothing I/O-shaped at all. That
// is exactly the property we want of the numerical core that is being
// cross-checked against Sounio's simulation: the arithmetic itself can
// never throw or perform I/O, it can only (provably, for `steps >= 0`)
// eventually return.
// ----------------------------------------------------------------------------
fun simulate(kf : float64, kr : float64, kcat : float64,
             e0 : float64, s0 : float64,
             dt : float64, steps : int) : div rxstate
  var y := (e0, s0, 0.0, 0.0)
  var i := 0
  while { i < steps }
    y := rk4-step(y, dt, kf, kr, kcat)
    i := i + 1
  y

// ----------------------------------------------------------------------------
// Explicit, typed division-by-near-zero effect.
//
// `divby0` is declared as a control operation returning a fully generic
// result type `a` (à la the `raise` idiom in Koka's own handler samples),
// so `safe-div` below can be reused for both the TON and TOF divisions.
// Any function that calls `safe-div` transitively picks up `divby0` in
// its own effect row -- e.g. `metrics`'s signature below reads
// `: <divby0> (float64, float64)`, which *is* the guarantee "this may
// fail to divide, and you must handle that or propagate it" showing up
// directly in the type, rather than being a fact only discoverable by
// reading the function body or by observing a NaN/Inf at runtime.
// ----------------------------------------------------------------------------
effect ctl divby0(msg : string) : a

fun fabs(x : float64) : float64
  if x < 0.0 then (0.0 - x) else x

fun safe-div(num : float64, den : float64, msg : string) : <divby0> float64
  if fabs(den) < 1.0e-12
    then divby0(msg)
    else num / den

// ----------------------------------------------------------------------------
// Derive TON (turnover number) and TOF (turnover frequency) from a finished
// simulation. Both divisions go through `safe-div`, so this function's own
// type is forced to carry `<divby0>` -- Koka will not let you write this
// function (or call it) while pretending the division might not fail.
// ----------------------------------------------------------------------------
fun metrics(product-formed : float64, e0 : float64, elapsed : float64) : <divby0> (float64, float64)
  val ton = safe-div(product-formed, e0, "E0 is zero (or numerically negligible): turnover number (product/E0) is undefined")
  val tof = safe-div(ton, elapsed, "elapsed time is zero (or numerically negligible): turnover frequency (TON/elapsed) is undefined")
  (ton, tof)

// ----------------------------------------------------------------------------
// Conservation self-check: total enzyme E + ES must stay equal to E0
// (catalyst is neither created nor destroyed by this mechanism), up to
// floating-point tolerance. Pure predicate, no effects.
// ----------------------------------------------------------------------------
fun conserved(final-state : rxstate, e0 : float64, tol : float64) : bool
  val (e, _s, es, _p) = final-state
  fabs((e + es) - e0) < tol

// ----------------------------------------------------------------------------
// Handle `divby0` by printing a diagnostic and falling back to (0.0, 0.0).
// This is where the pure `<divby0>` effect gets discharged into the
// `:console` I/O effect -- explicitly, at a single well-defined site,
// rather than being silently absorbed into NaN/Inf arithmetic.
// ----------------------------------------------------------------------------
fun run-metrics(product-formed : float64, e0 : float64, elapsed : float64) : console (float64, float64)
  with final ctl divby0(msg)
    println("  [divby0 EFFECT CAUGHT] " ++ msg)
    (0.0, 0.0)
  metrics(product-formed, e0, elapsed)

// ----------------------------------------------------------------------------
// Main: I/O-effectful driver. Runs the hand-checkable reference case
//   kf=1.2, kr=0.3, kcat=0.8, E0=0.05, S0=2.0, dt=0.02, steps=300
// (matching kinetics.sio's `test_enzyme_crn`), prints final concentrations,
// TON, TOF, and the conservation PASS/FAIL verdict at high precision.
// Then, separately, demonstrates the `divby0` effect actually firing on a
// deliberately degenerate input (E0 = 0), proving the typed effect is real
// and not just decorative.
// ----------------------------------------------------------------------------
fun report(kf : float64, kr : float64, kcat : float64,
           e0 : float64, s0 : float64, dt : float64, steps : int) : <div,console> ()
  val elapsed = dt * steps.float64
  val final-state = simulate(kf, kr, kcat, e0, s0, dt, steps)
  val (e, s, es, p) = final-state
  val product-formed = p          // P(0) = 0, so product_formed = P(elapsed) - 0 = P(elapsed)

  println("kf=" ++ kf.show ++ "  kr=" ++ kr.show ++ "  kcat=" ++ kcat.show)
  println("E0=" ++ e0.show ++ "  S0=" ++ s0.show ++ "  dt=" ++ dt.show ++ "  steps=" ++ steps.show ++ "  elapsed=" ++ elapsed.show)
  println("")
  println("Final concentrations (t = " ++ elapsed.show ++ "):")
  println("  E  = " ++ e.show-fixed(12))
  println("  S  = " ++ s.show-fixed(12))
  println("  ES = " ++ es.show-fixed(12))
  println("  P  = " ++ p.show-fixed(12))
  println("")
  println("product_formed = P(elapsed) - P(0) = " ++ product-formed.show-fixed(12))

  val (ton, tof) = run-metrics(product-formed, e0, elapsed)
  println("TON (turnover number, product/E0)     = " ++ ton.show-fixed(12))
  println("TOF (turnover frequency, TON/elapsed)  = " ++ tof.show-fixed(12))
  println("")

  val tol = 1.0e-9
  val ok = conserved(final-state, e0, tol)
  val (e2, _s2, es2, _p2) = final-state
  println("Conservation check: E + ES = " ++ (e2 + es2).show-fixed(12) ++ "   E0 = " ++ e0.show-fixed(12) ++ "   |diff| = " ++ fabs((e2+es2) - e0).show-fixed(15) ++ "   tol = " ++ tol.show)
  if ok
    then println("Conservation: PASS")
    else println("Conservation: FAIL")

pub fun main() : <div,console> ()
  println("==================================================================")
  println(" catalysis_cycle.kk -- independent Koka oracle for E+S<=>ES->E+P")
  println(" (hand-rolled fixed-step RK4, cross-checked against kinetics.sio)")
  println("==================================================================")
  println("")
  println("-- Reference case (matches kinetics.sio test_enzyme_crn) --")
  report(1.2, 0.3, 0.8, 0.05, 2.0, 0.02, 300)

  println("")
  println("-- Effect-typing demonstration: degenerate input E0 = 0.0 --")
  println("   (TON = product/E0 is undefined; `divby0` must fire and be")
  println("    handled explicitly -- it cannot silently become NaN/Inf)")
  report(1.2, 0.3, 0.8, 0.0, 2.0, 0.02, 300)
