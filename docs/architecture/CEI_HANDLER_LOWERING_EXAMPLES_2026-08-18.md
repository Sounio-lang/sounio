<!-- docs:meta
topic_id: repo.docs.architecture.cei-handler-lowering-examples-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-27
validated_by: backlog rescue from PR #1926
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.cei-handler-lowering-examples-2026-08-18
-->

# CEI handler lowering — the two effect-uncertainty examples, preserved as source

**These do not compile on `main`.** They are kept here as a record, not as runnable
examples, and that is deliberate.

They were written against the algebraic-effect handler lowering proposed in
[#1926](https://github.com/Sounio-lang/sounio/pull/1926) (`self-hosted/check/check.sio`,
`self-hosted/ir/lower.sio`, `self-hosted/native/codegen_x86_linux.sio`). That lowering is
an RFC its own author marked "not for direct merge", and it has not landed. Without it,
both files fail `souc check` on `main` with `E137 use of undeclared variable` and
`E011 no method named for this type` — measured 2026-08-27, not assumed.

Putting them under `examples/` would therefore add two files that do not build. Deleting
them with #1926 would lose the only record of what the lowering was *for*. This file is
the third option: the source survives, and nothing claims it works yet.

If the lowering lands, move these back to `examples/` and delete this file.

## `effect_uncertainty_smoke.sio`

```sounio
//@ run-pass
//@ expect-stdout: SMOKE 5
// CEI WS-A smoke: handle<Epistemic>{ Epistemic.add(2,3) } dispatched to the clause.
// No result annotation needed (P2: ExprHandle results carry scalar-kind).
fn main() -> i32 with IO {
    let r = handle<Epistemic> {
        Epistemic.add(2, 3)
    } with {
        let add = |a: i64, b: i64| { a + b }
    }
    print("SMOKE ")
    println(r)
    0
}
```

## `effect_uncertainty_gum_vs_mc.sio`

```sounio
//@ run-pass
// CEI WS-A P2/P3 — handler-selectable uncertainty semantics, ALL measurements
// threaded SOURCE -> HANDLER. The source performs Epistemic.width(xv,xu,yv,yu)
// with the full (value, uncertainty) pair for each of x~N(1.0,0.3), y~N(1.0,0.3);
// each with{} handler chooses the calculus for the 95% half-width of z = x*y.
// First-order GUM drops the sigma_x^2*sigma_y^2 cross term a product carries;
// Monte-Carlo captures it -> DIFFERENT bounds from an identical 4-arg source.
// (Enabled by the IrCallIndirect argc-cap fix — PR #1877 — which lifts the
// 2-arg limit on indirect/closure calls, so a handler clause can take 4 f64 args.)
fn main() -> i32 with IO {
    println("z = x*y,  x~N(1.0,0.3), y~N(1.0,0.3)  -- 95% half-width, by handler:")

    let _g = handle<Epistemic> {
        Epistemic.width(1.0, 0.3, 1.0, 0.3)
    } with {
        let width = |xv: f64, xu: f64, yv: f64, yu: f64| {
            let var_z: f64 = yv * yv * xu * xu + xv * xv * yu * yu
            print("  GUM (delta):    ")
            println(1.96 * sqrt(var_z))
            0
        }
    }

    let _m = handle<Epistemic> {
        Epistemic.width(1.0, 0.3, 1.0, 0.3)
    } with {
        let width = |xv: f64, xu: f64, yv: f64, yu: f64| {
            var seed: i64 = 987654321
            var sum: f64 = 0.0
            var sumsq: f64 = 0.0
            let n: i64 = 40000
            var i = 0
            while i < n {
                var ax: f64 = 0.0
                var kx = 0
                while kx < 12 {
                    seed = (seed * 1103515245 + 12345) % 2147483648
                    ax = ax + (seed as f64) / 2147483648.0
                    kx = kx + 1
                }
                var ay: f64 = 0.0
                var ky = 0
                while ky < 12 {
                    seed = (seed * 1103515245 + 12345) % 2147483648
                    ay = ay + (seed as f64) / 2147483648.0
                    ky = ky + 1
                }
                let x: f64 = xv + xu * (ax - 6.0)
                let y: f64 = yv + yu * (ay - 6.0)
                let z: f64 = x * y
                sum = sum + z
                sumsq = sumsq + z * z
                i = i + 1
            }
            let nf: f64 = n as f64
            let mean: f64 = sum / nf
            let var_z: f64 = sumsq / nf - mean * mean
            print("  Monte-Carlo:    ")
            println(1.96 * sqrt(var_z))
            0
        }
    }
    println("Same 4-arg source `Epistemic.width(1.0,0.3,1.0,0.3)`; only the with{} handler changed.")
    0
}
```
