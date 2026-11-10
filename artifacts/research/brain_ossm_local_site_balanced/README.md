`parsed_fixed/`, `parsed_pre_fix/`, and `brain_ossm_abide_site_balanced_fixed.out` were copies of the files kept below. The TSV tables and the `.out` log were byte-identical. `overall_metrics.json` differed only in path strings (an absolute manifest path versus a relative one, and `source_output` naming either `brain_ossm_abide_site_balanced.out` or the `_fixed.out` copy of the same bytes). SHA-256 of each kept file:

- `brain_ossm_abide_site_balanced.out` `ec28980d6b32a940da4ff608a0db8c1271d0bd16d1cf595bce4db93f0690232e`
- `parsed/overall_metrics.json` `b38a74d2a4b2bd5da6d9480ca67041af03c8b54b24d0f6094e6935f1805c4fe0`
- `parsed/overall_metrics.tsv` `785fb2de46c8199966595f9ac04a8aa5ff1e5aed54601c43134eafc7f030764f`
- `parsed/per_seed_metrics.tsv` `7ddbc251bc9be461adc78c6ca400294bc5c562ddd4ca93f89445e045dd024271`
- `parsed/per_site_metrics.tsv` `849cd21699c5aad905ec1c52cec0e48bf381ce5b1e0862a8f0aa9c797b54c8b1`
- `parsed/per_site_seed_metrics.tsv` `98c5f14ffaa9f6b1409a76ce599cfb20cfd4f6990e48e49479d72326f916a0ca`
- `parsed/prediction_rows.tsv` `5587775537b5272f0af12757427a94ae1bf721fdf2e0cd492e8c793b5e51d4bb`

Removed `overall_metrics.json` copies, path strings only: `parsed_fixed` `3ea6a92a528ab6bc9106e36f3e032cb304aa21d36fb4d367d6ab455a17e6a773`, `parsed_pre_fix` `b497c18b424cffcae9fe4089ef808c523a90975ec743653ccce3f262f12623e1`.
