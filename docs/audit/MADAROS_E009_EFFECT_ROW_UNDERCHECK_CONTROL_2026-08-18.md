# Madaros E009 Effect-Row Under-Check Control

Date: 2026-08-18

## Purpose

This is a standalone compiler control for the E009 investigation. It is not
a PBPK scientific test and it is not a function-name collision test. The
program contains one pure callback, one effectful callback, and two callback
slots. The decisive invalid case passes an effectful callback to a pure slot.

The source shape was:

```sounio
fn pure_model(x: [f64; 10]) -> f64 {
    x[0]
}

fn effect_model(x: [f64; 10]) -> f64 with Mut, Div, Panic {
    var y = x[0]
    y = y / 2.0
    y
}

fn accept_pure(model: fn([f64; 10]) -> f64) -> f64 {
    model([1.0; 10])
}

fn accept_effect(model: fn([f64; 10]) -> f64 with Mut, Div, Panic)
    -> f64 with Mut, Div, Panic {
    model([1.0; 10])
}

fn main() -> i32 with IO, Mut, Div, Panic {
    let _p = accept_pure(pure_model)
    let _e = accept_effect(effect_model)
    let _bad = accept_pure(effect_model)
    0
}
```

The invalid call was removed for the corrected control run. The fixture was
`/tmp/e009-fn-controls.sio`; the exact source is retained here so the witness
does not depend on that temporary path.

## Results

With the default Madaros launcher and the invalid call present:

```text
error[E009] ... argument type does not match parameter
  = expected fn#2
  = found fn#1
DEFAULT_MADAROS_INVALID_RC=1
```

The same invalid source under the legacy engine produced:

```text
LEAN_SINGLE_INVALID_RC=0
```

With the invalid call removed, both engines accepted the corrected control:

```text
DEFAULT_MADAROS_CORRECTED_RC=0
LEAN_SINGLE_CORRECTED_RC=0
```

## Interpretation

This is a motor differential, not a cosmetic check. Default Madaros rejects
the effect-row violation at the function argument boundary, while
`lean_single` accepts the identical invalid program. The latter is a measured
under-checking gap in the legacy engine and should remain visible as its own
finding rather than being hidden in the PBPK report.

The E009 in `pbpk28_sobol_pce.sio` is therefore explained by the public Sobol
API declaring a pure callback slot while invoking callbacks with
`Mut, Div, Panic`. The numeric function IDs in the diagnostic are local
`FnSigTable` indices and are not evidence that the resolver selected the
wrong function.

The source-built rerun of the repaired PBPK28 case used the published branch
`f1c94bd1a0526cd2c7c9345abcbb23df5346e206`; it built a `100084269` byte
Madaros ELF and emitted no E009. It returned one unrelated E035 for the
missing `Epistemic` effect on `ep28_selftest_main`.

No compiler, IR, codegen, or name-pool source was changed for this control.
