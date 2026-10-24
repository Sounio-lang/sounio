# Genomic scaling and ideal-gas entropy: independent demonstrations

`viz_genomics_entropy.sio` is a real synthetic visualization program using the
public checked APIs in `genomics::normalization` and `physics::entropy`.
The two panels demonstrate independent scientific operations. Their juxtaposition
does not imply a biological/thermodynamic relationship.

## Inputs and interpretation

The left panel plots centered log size factors for three synthetic samples.
Counts use physical row stride 8, with rows `[100,120,80]`, `[1000,4000,250]`,
`[500,700,400]`, and `[0,200,300]`. The zero-containing row is excluded by the
normalization contract. The three eligible genes exercise both nearby and distant
log-ratio paths. Sample indices denote categorical order; the connecting line
only joins the displayed samples. The API does not assess composition/reference
assumptions, and this example makes no biological inference.

The right panel plots ideal-gas isothermal entropy change for n=1 mol, initial
volume 1 in a common volume unit, and seven final/initial volume ratios:
0.5, 0.75, 0.999999999999, 1, 1.000000000001, 1.5 and 2. These include contraction,
equality, expansion and nearby volumes. Amount standard uncertainty is 0.001 mol;
both volume standard uncertainties are 0.001 in the common volume unit.
The report prints propagated first-order standard uncertainties from both the
independent model and the correlated-volume model with supplied rho=0.5.
Correlation is declared, not estimated. Standard uncertainty is not a confidence
interval. Only nominal entropy is plotted; uncertainty values are in the report.

## Execution and resource boundary

The program exits 1 if a required API or scene construction is unavailable; it
returns 0 after emitting the full SVG/HTML. It contains no assertions or expected
numeric reference. It reuses two1024-element plotting buffers and one64-byte label
buffer; SVG-only output allocates no Canvas. Scene-owned storage is separate from
these input buffers. This does not make the entire program allocation-free.

[Execution receipt](../docs/plans/DOMAIN_LOG1P_EXECUTION_2026_10_21.md) records the
frozen source, exact compiler/ELF identities, selected typechecks, native execution,
figure inspection and whole-program resource observations when completed.
Independent numerical concordance, global precision, full corpus, CI and merge
are separate acceptance boundaries.

The ordinary report formatter and the full-scale viewport may not distinguish
near-unity ratios or their tiny entropy changes. These samples exercise the
consumer path; they are not a tiny-contrast accuracy demonstration.
