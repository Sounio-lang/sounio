# Sedation weaning moved

The Proof-Carrying Weaning / opioid–α2 occupancy study no longer lives in
this stdlib. Canonical source:

**Sibling repository:** https://github.com/Sounio-lang/sounio-pbpk-sedation-weaning  
Local checkout: `../sounio-pbpk-sedation-weaning`

## What left this tree

| Former path | New path (sibling) |
|---|---|
| `stdlib/darwin_pbpk/proof_carrying_weaning.sio` | `src/proof_carrying_weaning.sio` |
| `stdlib/darwin_pbpk/pd/opioid_alpha2_occupancy.sio` | `src/pd/opioid_alpha2_occupancy.sio` |
| `stdlib/darwin_pbpk/drugs/fentanyl.sio` | `src/drugs/fentanyl.sio` |
| `stdlib/darwin_pbpk/drugs/morphine.sio` | `src/drugs/morphine.sio` |
| `stdlib/darwin_pbpk/drugs/clonidine.sio` | `src/drugs/clonidine.sio` |
| `stdlib/darwin_pbpk/test_proof_carrying_weaning.sio` | `tests/test_proof_carrying_weaning.sio` |
| `stdlib/darwin_pbpk/test_occupancy_validation.sio` | `tests/test_occupancy_validation.sio` |
| `tests/stdlib/darwin_pbpk/test_morphine_validation.sio` | `tests/test_morphine_validation.sio` |
| `tests/stdlib/darwin_pbpk/test_clonidine_validation.sio` | `tests/test_clonidine_validation.sio` |
| `formal/lean4/SounioOpioidWeaningSafety.lean` | `formal/lean4/SounioOpioidWeaningSafety.lean` |

Drug modules now export `pbpk28_params_*` for the canonical
`darwin_pbpk::tsit5_pbpk28::pbpk28_full_cn_step` runtime (CN, not Tsit5).
`PBPKParams14` remains provenance (fu/rb); conversion is
`core/pbpk28_bridge.sio` in the sibling. The Tsit5-14 solver stays unused
by the weaning package.

## What stays here

- `drugs/midazolam.sio` and CYP3A DDI vertical
- Tsit5 / PBPK28 / GUM / `pd/d2_occupancy.sio` / rapamycin–GLP-1 TMDD
- Madaros runtime limitation items 1–7 in `RUNTIME_NOTES.md`

## Origin SHAs (kernel history)

- `d83a085aa`
- `cd27d1707`
- `42728e6b7`

Branch at extraction: `darwin-pbpk/proof-carrying-weaning-p3`.
