`souc build --gpu-launch-report` prints, per `kernel fn`, whether the parameter
plan classifies it for a device launch or which refusal keeps it on the CPU
lane, together with the roles it derived:

```
launch: vec_mul_launch device [extent,slice_in,slice_in,slice_out]
launch: read_only cpu:no_output_slice [extent,slice_in]
```

The verdict is computed before codegen, in the native compile path, which has no
HLIR of its own: the pass exists so that a launch has a plan at all. The host
launch emitter will read the same table, so the decision and the ABI cannot
drift apart.

This is a report, not yet a launch. `perform GPU.launch` still lowers to a CPU
call and the program still runs on the CPU lane.

The pass lowers the module to HLIR, and `HlirModule` is returned by value
(~12MB of frame), so `bin/madaros` raises the stack for this flag the way the
canonical compiler gate already does.
