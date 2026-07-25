# FAERS Fano order-asymmetry — expansion v2

## Discovery

`data/faers_drugbank.csv` column `fano` **mismatches** the Fano geometry in
`stdlib/medical/cyp450_fano.sio` on **14 of 35** CYP triples. Any analysis that
trusted the CSV label is invalid.

## v2 protocol

- Artefact: `data/faers_drugbank.csv` (sha256 `511d6111…`)
- Fano membership: geometry (`FANO_LINES`), not CSV
- Inclusion: `temporal > 0`, `asymmetry >= 0` → n_F=7, n_N=28
- Contrast: `mean(non-Fano) − mean(Fano)`, bootstrap B=1000 seed=20260725

```bash
./bin/souc run experiments/faers_fano_order_asymmetry/faers_fano_order_asymmetry_v2.sio
```

## Result (2026-07-25)

| Metric | Value |
|---|---|
| mean Fano / non-Fano | 0.147 / 0.171 |
| diff (unweighted) | **+0.024** |
| weighted diff | **+0.041** |
| 95% CI | **[-0.066, 0.112]** |
| **Verdict** | **INCONCLUSIVE** |

Point estimate is in the H1 direction after correct labeling; CI still crosses zero.
Not a clinical claim.
