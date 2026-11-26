Interval results from `verify::interval` and `epistemic::interval_ieee`
are wider than before: every computed endpoint is now pushed strictly
outward with the Rump predecessor/successor push shared with
`epistemic::sound_enclosure` (lie corpus item 1d), so the true real result
is contained. Previously the round-to-nearest endpoint was returned as-is
(e.g. `[0.1,0.1]+[0.2,0.2]` was the point `0.30000000000000004`, which
excludes the true sum 0.3). Arithmetic results now carry the `IV_ROUNDED`
provenance tag, and `epistemic::interval_ieee`'s entire interval is really
`[-inf, +inf]` (via `se_entire`) instead of `+-1e308`. Pinned by
`tests/run-pass/lie01d_iv_add_contains_true_sum.sio`,
`tests/run-pass/lie01d_ieee_outward_rounding.sio` and
`tests/run-pass/lie01d_probrange_fields.sio`.
