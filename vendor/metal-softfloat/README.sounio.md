# Vendored MSL binary64 implementation

Upstream: https://github.com/guyfischman/metal-softfloat
Pinned commit: `8b6c592e2e383040fe2778bed8dda7904df284b1`.

Only the redistributable, generated MSL header and both required licenses are
vendored. No Rust implementation, build, numerical oracle or test harness is
used. Sounio verification uses C++23 and actual Metal execution. Upstream's
conformance/performance claims are not substituted for local measurements.

Use the ordinary packed `__softfloat64_*` entry points with rounding mode zero;
the normal-only `_unp_*` fast path is not valid for arbitrary language f64.
NaN results use upstream's canonical quiet NaN; payload preservation is not
claimed. Keep both licenses with source and binary distributions.
