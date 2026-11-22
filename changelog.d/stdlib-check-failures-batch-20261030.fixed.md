Eight stdlib e2e tests that failed `souc check` now compile and pass. Each test's fix:
- `csv`: `csv_write_file`, `df_write_csv` and `df_read_csv` now declare the `IO` effect that `write_file` and `read_file` require.
- `nn::epistemic_backward`: the in-module MLP test called the nonexistent `epistemic_mlp_4x4`; it now calls `epistemic_mlp_4_4_4`.
- `particle_physics`: `measurement_new` built an 8-element literal for the 4-element `syst_errs` field.
- `pbpk::rapamycin_units_bridge`: `units` now exports `dim_volume_rate` (m³·s⁻¹), which the bridge test asserts against.
- `stats` (`shapiro`, `ols_diag`, `ols_madaros`): the tests passed `[f64; 256]` buffers to `shapiro_wilk`, whose API is `[f64; 100]` (Royston, n ≤ 100).

Separately, `physics::classical` imports the public `units::dim_force` instead of rebuilding it with the private `dim_add_exp`, and `gb_sort_vals` declares `Mut`, which repairs the `dataframe_groupby_workflow` example.
