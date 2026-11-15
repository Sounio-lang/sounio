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

/-! Model regressions found by the CPU differential (bootstrap/model/x86diff_gate.sh). -/

/-- One step of `bytes` at 0x400000 from registers `regs`, all flags defined. -/
def step1 (bytes : List Nat) (regs : List Nat) : Sum St Outcome :=
  step (Img.mk' bytes 0x400000 bytes.length)
    { regs := regs.map some, rip := 0x400000, fl := ⟨some true, some false, some true, some false⟩,
      mem := List.replicate winSize (some 0), lo := 0x10800 - winSize / 2, inp := [], out := [] }

/-- `shl r10d, 0xc0`: the count masks to 0, so the flags stay, but the 32-bit
destination is still written and bits 32..63 are cleared. Measured on a Xeon 6730P,
2026-11-15 (case `41c1e2c0`); the first version of the model left r10 unchanged. -/
theorem shl32_count0_zero_extends :
    (match step1 [0x41, 0xc1, 0xe2, 0xc0]
        ((List.replicate 16 0).set 4 0x10800 |>.set 10 0x75e0dd3226a6fe92) with
      | .inl t => t.reg 10 == some 0x26a6fe92 && t.fl == ⟨some true, some false, some true, some false⟩
      | .inr _ => false) = true := by decide +kernel

end X86Min.Seeds
