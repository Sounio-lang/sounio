# Replace the identity square-root stub in geometry distances

point2d_distance and point3d_distance now use the native square-root intrinsic with scaled component norms. Their public Div-only signatures are retained. Scaling prevents avoidable overflow and underflow from squaring finite components; infinity dominates NaN, other NaNs propagate, and zero is positive zero.

Slurm19251 completed successfully: the original geometry Run-proof executes with exit0, existing geo_core checks with exit0, and an independent C++23 reference accepts30 adversarial distances with maximum observed error1ULP against a4ULP sample bound. The wrong-result control is rejected with exit1. Compiler and ELF hash guards pass. This finite sample is not a uniform error proof.

The three-dimensional reference composes binary std::hypot operations. A first attempt exposed the nonfinite-value underspecification of the three-argument overload discussed in LWG3172. A subsequent prototype imported a hidden effectful sqrt and was rejected; the qualified source uses the native intrinsic without that import.

A fresh integrated whole-stdlib run remains required.
