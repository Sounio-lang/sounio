/-
Small kernel-checked facts about the two seed loaders over the model
Bootstrap/X86Min.lean, cheap enough for every CI run. The full self-reproduction
and cross-check theorems are generated and checked by
bootstrap/model/seed_proofs.sh (about 2.5 hours of kernel time in all).

Bytes: '0' 48, '1' 49, '3' 51, '4' 52, '7' 55, '8' 56, 'a' 97, 'f' 102,
'g' 103, ' ' 32, '\n' 10, '\t' 9, '\r' 13, '#' 35, ';' 59.
-/
import Bootstrap.X86Min
import Bootstrap.SeedData
open X86Min X86Min.Data

namespace X86Min.Seeds

/-- seed-a: hex pairs, separators and a `#` comment. -/
theorem a_pairs : exec 1000 seedA [48, 49, 32, 97, 102, 10, 35, 103, 10, 102, 102] = .exit 0 [1, 175, 255] := by
  decide +kernel

/-- seed-a refuses a character that is not a hex digit, after the bytes before it. -/
theorem a_refuses_g : exec 1000 seedA [48, 49, 103] = .exit 1 [1] := by decide +kernel

/-- seed-a refuses an odd number of digits at the end of input. -/
theorem a_refuses_odd : exec 1000 seedA [48, 49, 48] = .exit 1 [1] := by decide +kernel

/-- seed-a refuses carriage return. -/
theorem a_refuses_cr : exec 1000 seedA [13] = .exit 1 [] := by decide +kernel

/-- seed-a: empty input is a valid listing of no bytes. -/
theorem a_empty : exec 1000 seedA [] = .exit 0 [] := by decide +kernel

/-- seed-b: octal groups, a tab, a `;` comment. -/
theorem b_groups : exec 1000 seedB [48, 48, 49, 9, 51, 55, 55, 59, 55, 10, 49, 52, 55] = .exit 0 [1, 255, 103] := by
  decide +kernel

/-- seed-b refuses the digit 8. -/
theorem b_refuses_8 : exec 1000 seedB [48, 48, 56] = .exit 1 [] := by decide +kernel

/-- seed-b refuses a first digit above 3 (a group above 377). -/
theorem b_refuses_400 : exec 1000 seedB [52, 48, 48] = .exit 1 [] := by decide +kernel

/-- seed-b refuses a separator inside a group. -/
theorem b_refuses_split : exec 1000 seedB [48, 32, 48, 49] = .exit 1 [] := by decide +kernel

/-- seed-b refuses an incomplete group at the end of input. -/
theorem b_refuses_short : exec 1000 seedB [48, 48, 49, 48, 48] = .exit 1 [1] := by decide +kernel

/-- seed-b: empty input is a valid listing of no bytes. -/
theorem b_empty : exec 1000 seedB [] = .exit 0 [] := by decide +kernel

end X86Min.Seeds
