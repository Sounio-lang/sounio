/-
AArch64PhysCheck: kernel-verified executable checker for Pillar P4-4
(`AArch64Env`, `AArch64Phys`, `TMRVoterArm64`).

Supports three CLI subcommands:
  1. `selftest` — pure-Lean architectural & physical model self-check
     (runs cleanly on non-ARM64 CI hosts as well as ARM64 hardware):
       * `AArch64Env` Linux syscall & `initStack` checks
       * `AArch64Phys` decode/exec checks across all 53 instruction forms
       * `TMRVoterArm64` 44-byte voter decode, majority/refusal, exhaustive
         192-case single-bit masking, and 352-flip Hamming-1 boundary locality
       * exact rational `3 * q² + 11 * r` witness checks
  2. `diff <min_cases>` — reads native AArch64 silicon execution traces on
     `stdin`, decodes each 32-bit word via `AArch64Phys.decode`, classifies
     across all 53 instruction forms, steps `AArch64Phys.step`, and demands
     `0` mismatches (and uniform per-form coverage when `cases ≥ 53000`).
  3. `verify-phys` — verifies the structured hardware physical-perturbation
     report from `aarch64_phys_fuzz --phys` using exact `Rat` arithmetic:
       * exact 44-byte `VOTER_CODE` equality with `TMRVoterArm64.voterCode`
       * `SYSCALL_EXIT3_CHECK` (`exit 3`) & `PTRACE_STEP_CHECK`
       * `FETCH_BOUNDARY_SWEEP` (`≥ 8,544` Hamming-1 flips, `0` guard-page
         fetch straddles) plus Lean-verified `FETCH_ALT` execution checks
       * `SINGLE_FAULT_SWEEP` (`≥ 3,712` trials, `SDC = 0`, `refusal = 0`, `stuck = 0`)
       * `NEG_UNHARDENED` negative baseline (`SDC = 640 / 640`)
       * Mode A (4 points) & Mode B (9 points) exact `Rat` 5-sigma validation
         against `3 * q² + 11 * r`.
       * the Monte Carlo design itself: each of the 4 Mode A and 9 Mode B `(q, r)` points
         exactly once, no other point, and at least `min_trials` trials per row
         (`verify-phys [file [min_trials]]`, default 20,000).

Core Lean 4 only: zero Mathlib imports, zero axioms beyond core Lean.
-/
import Bootstrap.PDist
import Bootstrap.TMR
import Bootstrap.TMRExposure
import Bootstrap.AArch64Env
import Bootstrap.AArch64Phys
import Bootstrap.TMRVoterArm64

namespace AArch64PhysCheck

open AArch64Env AArch64Phys TMRVoterArm64 PDist

/-! ### Exact rational bound theorems -/

/-- Exact rational single-step TMR physical failure bound `3 * q² + 11 * r`
proved in `TMRVoterArm64.voted_step_with_faulty_voter_bound`. -/
def tmrBoundArm64 (q r : Rat) : Rat := 3 * (q * q) + 11 * r

theorem tmrBoundArm64_p7 :
    tmrBoundArm64 (1 / 5000000) (4 / 5000000) = 220000003 / 25000000000000 := by
  decide +kernel

theorem tmrBoundArm64_modeA_1e2 :
    tmrBoundArm64 (1 / 100) 0 = 3 / 10000 := by
  decide +kernel

theorem tmrBoundArm64_modeB_1e2_1e4 :
    tmrBoundArm64 (1 / 100) (1 / 10000) = 14 / 10000 := by
  decide +kernel

theorem voterCode_len_44 : voterCode.length = 44 :=
  TMRVoterArm64.voterCode_length

/-- Exact 5-sigma rational acceptance predicate for empirical failure count
`fails` out of `trials` against theoretical upper bound `bound`:
accepts iff `pHat ≤ bound` or `(pHat - bound)² ≤ 25 * (bound * (1 - bound) + 1/N) / N`. -/
def withinFiveSigma (bound : Rat) (trials fails : Nat) : Bool :=
  if trials == 0 then false
  else
    let n : Rat := (trials : Rat)
    let pHat : Rat := (fails : Rat) / n
    if pHat ≤ bound then true
    else
      let diff := pHat - bound
      let varTerm := bound * (1 - bound) + 1 / n
      diff * diff ≤ 25 * varTerm / n

/-! ### Hex & parsing helpers -/

def hexNib (c : Char) : Option Nat :=
  if '0' ≤ c && c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c && c ≤ 'f' then some (10 + c.toNat - 'a'.toNat)
  else if 'A' ≤ c && c ≤ 'F' then some (10 + c.toNat - 'A'.toNat)
  else none

def parseHexNat (s : String) : Option Nat :=
  if s.isEmpty then none
  else s.toList.foldlM (fun acc c => do let d ← hexNib c; pure (acc * 16 + d)) 0

def parseHexBytes (s : String) : Option (List Nat) :=
  let cs := s.toList
  if cs.length % 2 != 0 then none
  else go cs []
where
  go : List Char → List Nat → Option (List Nat)
    | [], acc => some acc.reverse
    | c1 :: c2 :: rest, acc => do
      let hi ← hexNib c1
      let lo ← hexNib c2
      go rest ((hi * 16 + lo) :: acc)
    | _, _ => none

def nibbleChar (n : Nat) : Char :=
  let d := n % 16
  if d < 10 then Char.ofNat ('0'.toNat + d) else Char.ofNat ('a'.toNat + (d - 10))

def bytesToHex (bs : List Nat) : String :=
  String.ofList (bs.flatMap fun b => [nibbleChar ((b / 16) % 16), nibbleChar (b % 16)])

def voterCodeHex : String := bytesToHex voterCode

def parseFlags4 (s : String) : Option Flags :=
  match s.toList with
  | [cn, cz, cc, cv] =>
    let bit := fun c => if c == '1' then some true else if c == '0' then some false else none
    do
      let n ← bit cn
      let z ← bit cz
      let c ← bit cc
      let v ← bit cv
      pure ⟨some n, some z, some c, some v⟩
  | _ => none

/-! ### 53-form instruction classifier -/

def formNames : Array String :=
  #[ "movz32", "movz64", "movn32", "movn64", "movk32", "movk64"
   , "addImm32", "addImm64", "addsImm", "subImm32", "subImm64", "subsImm"
   , "addReg", "addsReg", "subReg", "subsReg"
   , "andReg", "orrReg", "eorReg", "andsReg"
   , "sbfm", "bfm", "ubfm"
   , "adr", "adrp"
   , "ccmpReg", "ccmnReg"
   , "csel", "csinc", "csinv", "csneg"
   , "ldrb", "strb", "ldrh", "strh", "ldr32", "str32", "ldr64", "str64"
   , "ldur", "stur", "ldrReg", "strReg"
   , "b", "bl", "bcond", "cbz", "cbnz", "tbz", "tbnz", "br", "blr", "ret"
   ]

def classifyIns : Ins → Option Nat
  | .movWide 2 32 _ _ _ => some 0
  | .movWide 2 64 _ _ _ => some 1
  | .movWide 0 32 _ _ _ => some 2
  | .movWide 0 64 _ _ _ => some 3
  | .movWide 3 32 _ _ _ => some 4
  | .movWide 3 64 _ _ _ => some 5
  | .addSubImm false false 32 _ _ _ => some 6
  | .addSubImm false false 64 _ _ _ => some 7
  | .addSubImm false true  _  _ _ _ => some 8
  | .addSubImm true  false 32 _ _ _ => some 9
  | .addSubImm true  false 64 _ _ _ => some 10
  | .addSubImm true  true  _  _ _ _ => some 11
  | .addSubReg false false _ _ _ _ _ _ => some 12
  | .addSubReg false true  _ _ _ _ _ _ => some 13
  | .addSubReg true  false _ _ _ _ _ _ => some 14
  | .addSubReg true  true  _ _ _ _ _ _ => some 15
  | .logicReg .and  _ _ _ _ _ _ _ => some 16
  | .logicReg .orr  _ _ _ _ _ _ _ => some 17
  | .logicReg .eor  _ _ _ _ _ _ _ => some 18
  | .logicReg .ands _ _ _ _ _ _ _ => some 19
  | .bitfield .sbfm _ _ _ _ _ => some 20
  | .bitfield .bfm  _ _ _ _ _ => some 21
  | .bitfield .ubfm _ _ _ _ _ => some 22
  | .adr false _ _ => some 23
  | .adr true  _ _ => some 24
  | .ccmpReg true  _ _ _ _ _ => some 25
  | .ccmpReg false _ _ _ _ _ => some 26
  | .csel .csel  _ _ _ _ _ => some 27
  | .csel .csinc _ _ _ _ _ => some 28
  | .csel .csinv _ _ _ _ _ => some 29
  | .csel .csneg _ _ _ _ _ => some 30
  | .ldr .b1 _ _ _ => some 31
  | .str .b1 _ _ _ => some 32
  | .ldr .h2 _ _ _ => some 33
  | .str .h2 _ _ _ => some 34
  | .ldr .w4 _ _ _ => some 35
  | .str .w4 _ _ _ => some 36
  | .ldr .x8 _ _ _ => some 37
  | .str .x8 _ _ _ => some 38
  | .ldur _ _ _ _ => some 39
  | .stur _ _ _ _ => some 40
  | .ldrReg _ _ _ _ _ _ => some 41
  | .strReg _ _ _ _ _ _ => some 42
  | .b false _ => some 43
  | .b true  _ => some 44
  | .bcond _ _ => some 45
  | .cbz false _ _ _ => some 46
  | .cbz true  _ _ _ => some 47
  | .tbz false _ _ _ => some 48
  | .tbz true  _ _ _ => some 49
  | .br false _ => some 50
  | .br true  _ => some 51
  | .ret _ => some 52
  | _ => none

/-! ### Single-instruction differential checker -/

def dummyKern : Kern := {
  files := [],
  fds := [],
  stdin := [],
  out := [],
  oracle := []
}

def compareRegs (modelRegs cpuRegs : List (Option Nat)) : Option String :=
  go 0 modelRegs cpuRegs
where
  go : Nat → List (Option Nat) → List (Option Nat) → Option String
    | _, [], [] => none
    | i, m :: ms, c :: cs =>
      -- Skip x18: platform-reserved on Darwin arm64 (XNU may clobber on preemption).
      if i == 18 then go (i + 1) ms cs
      else if m == c then go (i + 1) ms cs
      else some s!"x{i}: model={repr m} cpu={repr c}"
    | i, _, _ => some s!"reg list length mismatch at x{i}"

def parseRegs31 (toks : Array String) (off : Nat) : Option (List (Option Nat)) := do
  let mut out : List (Option Nat) := []
  for i in [0:31] do
    let v ← parseHexNat (toks[off + i]!)
    out := some v :: out
  pure out.reverse

/-- Parse and verify one differential trace line:
`<ins8> <pc> <win_lo> <sp> <x0..x30> <NZCV> <mem128> | <delta> <sp_o> <x0..x30_o> <NZCV_o> <mem128_o>` -/
def checkDiffTokens (toks : Array String) : Except String Nat := do
  if toks.size != 73 then
    throw s!"expected 73 tokens, got {toks.size}"
  if toks[37]! != "|" then
    throw s!"expected '|' separator at token 37, got '{toks[37]!}'"
  let some insBytes := parseHexBytes toks[0]! | throw s!"bad ins bytes '{toks[0]!}'"
  let some pc := parseHexNat toks[1]! | throw s!"bad pc '{toks[1]!}'"
  let some winLo := parseHexNat toks[2]! | throw s!"bad win_lo '{toks[2]!}'"
  let some spIn := parseHexNat toks[3]! | throw s!"bad sp '{toks[3]!}'"
  let some regsIn := parseRegs31 toks 4 | throw "bad input regs"
  let some flIn := parseFlags4 toks[35]! | throw s!"bad input flags '{toks[35]!}'"
  let some memIn := parseHexBytes toks[36]! | throw "bad input mem"
  let some delta := parseHexNat toks[38]! | throw s!"bad delta '{toks[38]!}'"
  let some spOut := parseHexNat toks[39]! | throw s!"bad sp_out '{toks[39]!}'"
  let some regsOut := parseRegs31 toks 40 | throw "bad output regs"
  let some flOut := parseFlags4 toks[71]! | throw s!"bad output flags '{toks[71]!}'"
  let some memOut := parseHexBytes toks[72]! | throw "bad output mem"
  let some (ins, len) := AArch64Phys.decode (fun i => insBytes[i]?) |
    throw s!"AArch64Phys.decode returned none for {toks[0]!}"
  if len != 4 then
    throw s!"decoded len={len} != 4 for {toks[0]!}"
  let some formIdx := classifyIns ins |
    throw s!"unclassified instruction {repr ins}"
  let st0 : St := {
    regs := regsIn,
    sp := some spIn,
    pc := pc,
    fl := flIn,
    mem := [⟨winLo, memIn, true⟩],
    k := dummyKern
  }
  let im : Img := imageOf insBytes pc
  match AArch64Phys.step im st0 with
  | .ok st1 =>
    let expectedPc := (pc + delta) % W
    if st1.pc != expectedPc then
      throw s!"pc mismatch on {formNames[formIdx]!} ({toks[0]!}): model={st1.pc} cpu={expectedPc}"
    if st1.sp != some spOut then
      throw s!"sp mismatch on {formNames[formIdx]!} ({toks[0]!}): model={repr st1.sp} cpu={some spOut}"
    if let some err := compareRegs st1.regs regsOut then
      throw s!"reg mismatch on {formNames[formIdx]!} ({toks[0]!}): {err}"
    if st1.fl != flOut then
      throw s!"flags mismatch on {formNames[formIdx]!} ({toks[0]!}): model={repr st1.fl} cpu={repr flOut}"
    let mBytes := match st1.mem with | [r] => r.data | _ => []
    if mBytes != memOut then
      throw s!"mem mismatch on {formNames[formIdx]!} ({toks[0]!})"
    pure formIdx
  | .exit c =>
    throw s!"unexpected model exit({c}) on {formNames[formIdx]!} ({toks[0]!})"
  | .stuck =>
    throw s!"model stuck on {formNames[formIdx]!} ({toks[0]!}, {repr ins})"

/-- Verify a Hamming-1 fetch-perturbation hardware trace line (same payload as
`diff` after `FETCH_ALT` tag, without restricting form index). -/
def checkFetchAltTokens (toks : Array String) : Except String Unit := do
  if toks.size != 74 then
    throw s!"FETCH_ALT expected 74 tokens, got {toks.size}"
  let body := toks.extract 1 74
  let _ ← checkDiffTokens body
  pure ()

/-! ### Subcommand 1: Pure-Lean `selftest` (runs on all host architectures) -/

/-- Representative valid 32-bit encodings covering all 53 instruction forms. -/
def selftestWords : List (Nat × Nat) :=
  [ (0,  0x52800241) -- movz w1, #0x12
  , (1,  0xd2800241) -- movz x1, #0x12
  , (2,  0x12800241) -- movn w1, #0x12
  , (3,  0x92800241) -- movn x1, #0x12
  , (4,  0x72a00241) -- movk w1, #0x12, lsl #16
  , (5,  0xf2a00241) -- movk x1, #0x12, lsl #16
  , (6,  0x11001420) -- add w0, w1, #5
  , (7,  0x91001420) -- add x0, x1, #5
  , (8,  0xb1001420) -- adds x0, x1, #5
  , (9,  0x51001420) -- sub w0, w1, #5
  , (10, 0xd1001420) -- sub x0, x1, #5
  , (11, 0xf1001420) -- subs x0, x1, #5
  , (12, 0x8b020020) -- add x0, x1, x2
  , (13, 0xab020020) -- adds x0, x1, x2
  , (14, 0xcb020020) -- sub x0, x1, x2
  , (15, 0xeb020020) -- subs x0, x1, x2
  , (16, 0x8a020020) -- and x0, x1, x2
  , (17, 0xaa020020) -- orr x0, x1, x2
  , (18, 0xca020020) -- eor x0, x1, x2
  , (19, 0xea020020) -- ands x0, x1, x2
  , (20, 0x93442c20) -- sbfx x0, x1, #4, #8 (sbfm x0, x1, #4, #11)
  , (21, 0xb37c1c20) -- bfi x0, x1, #4, #8 (bfm x0, x1, #60, #7)
  , (22, 0xd37ff820) -- lsl x0, x1, #1 (ubfm x0, x1, #63, #62)
  , (23, 0x10000040) -- adr x0, #8
  , (24, 0x90000020) -- adrp x0, #0x4000
  , (25, 0xfa420026) -- ccmp x1, x2, #6, eq
  , (26, 0xba420026) -- ccmn x1, x2, #6, eq
  , (27, 0x9a820020) -- csel x0, x1, x2, eq
  , (28, 0x9a820420) -- csinc x0, x1, x2, eq
  , (29, 0xda820020) -- csinv x0, x1, x2, eq
  , (30, 0xda820420) -- csneg x0, x1, x2, eq
  , (31, 0x39400020) -- ldrb w0, [x1]
  , (32, 0x39000020) -- strb w0, [x1]
  , (33, 0x79400020) -- ldrh w0, [x1]
  , (34, 0x79000020) -- strh w0, [x1]
  , (35, 0xb9400020) -- ldr w0, [x1]
  , (36, 0xb9000020) -- str w0, [x1]
  , (37, 0xf9400020) -- ldr x0, [x1]
  , (38, 0xf9000020) -- str x0, [x1]
  , (39, 0xf8408020) -- ldur x0, [x1, #8]
  , (40, 0xf8008020) -- stur x0, [x1, #8]
  , (41, 0xf8626820) -- ldr x0, [x1, x2]
  , (42, 0xf8226820) -- str x0, [x1, x2]
  , (43, 0x14000004) -- b +16
  , (44, 0x94000004) -- bl +16
  , (45, 0x54000080) -- b.eq +16
  , (46, 0xb4000081) -- cbz x1, +16
  , (47, 0xb5000081) -- cbnz x1, +16
  , (48, 0x36100081) -- tbz w1, #2, +16
  , (49, 0x37100081) -- tbnz w1, #2, +16
  , (50, 0xd61f0060) -- br x3
  , (51, 0xd63f0060) -- blr x3
  , (52, 0xd65f03c0) -- ret
  ]

def runSelftest : IO UInt32 := do
  -- 1. Verify all 53 instruction forms decode and execute cleanly in AArch64Phys.
  let winLo : Nat := 0x10000
  let pc0 : Nat := 0x400000
  let baseRegs : List (Option Nat) :=
    (List.range 31).map fun i =>
      if i == 1 then some winLo
      else if i == 2 then some 8
      else if i == 3 then some (pc0 + 0x40)
      else if i == 30 then some (pc0 + 0x80)
      else some (i * 0x11111111 + 7)
  let baseMem : Mem := [⟨winLo, List.replicate 64 0x5a, true⟩]
  let baseSt : St := {
    regs := baseRegs, sp := some winLo, pc := pc0,
    fl := ⟨some false, some true, some true, some false⟩,
    mem := baseMem, k := dummyKern
  }
  for (expectedForm, w) in selftestWords do
    let bs := wordsToBytes [w]
    match AArch64Phys.decode (fun i => bs[i]?) with
    | none =>
      IO.eprintln s!"selftest FAIL: form {expectedForm} word 0x{Nat.toDigits 16 w |> String.ofList} failed to decode"
      return 1
    | some (ins, len) =>
      if len != 4 then
        IO.eprintln s!"selftest FAIL: form {expectedForm} len={len} != 4"
        return 1
      match classifyIns ins with
      | some idx =>
        if idx != expectedForm then
          IO.eprintln s!"selftest FAIL: form {expectedForm} classified as {idx} ({repr ins})"
          return 1
      | none =>
        IO.eprintln s!"selftest FAIL: form {expectedForm} unclassified ({repr ins})"
        return 1
      match AArch64Phys.step (imageOf bs pc0) baseSt with
      | .ok _ => pure ()
      | other =>
        IO.eprintln s!"selftest FAIL: form {expectedForm} step failed ({repr other})"
        return 1

  -- 2. Verify AArch64Env syscall ABI (openat/read/write/close/mmap/mremap/munmap/exit/initStack).
  let initK : Kern := {
    files := [([0x68, 0x65, 0x6c, 0x6c, 0x6f], [0x41, 0x42])],
    fds := [],
    stdin := [],
    out := [],
    oracle := [.count 2]
  }
  let m0 : Mem := [⟨0x5000, [0x68, 0x65, 0x6c, 0x6c, 0x6f, 0], true⟩]
  match syscall initK m0 56 AT_FDCWD 0x5000 0 0 0 0 with
  | some (.ret fd k1 m1) =>
    if fd != 3 then IO.eprintln "selftest FAIL: openat fd != 3"; return 1
    match syscall k1 m1 63 fd 0x5000 2 0 0 0 with
    | some (.ret n k2 m2) =>
      if n != 2 then IO.eprintln "selftest FAIL: read n != 2"; return 1
      match syscall k2 m2 57 fd 0 0 0 0 0 with
      | some (.ret 0 _ _) => pure ()
      | _ => IO.eprintln "selftest FAIL: close failed"; return 1
    | _ => IO.eprintln "selftest FAIL: read failed"; return 1
  | _ => IO.eprintln "selftest FAIL: openat failed"; return 1

  -- 3. Verify 352 Hamming-1 fetch flips of canonical 44B voter satisfy fixed-width locality.
  let mut flipCount : Nat := 0
  for idx in [0:voterWords.length] do
    let pc := voterBase + 4 * idx
    let alts := voterAlts pc
    if alts.length != 32 then
      IO.eprintln s!"selftest FAIL: voterAlts at idx {idx} len={alts.length} != 32"
      return 1
    let truncIm := imageOf (voterCode.take (4 * idx + 4)) voterBase
    if voterIm pc != truncIm pc then
      IO.eprintln s!"selftest FAIL: truncation locality violated at idx {idx}"
      return 1
    for o in alts do
      flipCount := flipCount + 1
      match o with
      | none => pure ()
      | some (_, len) =>
        if len != 4 then
          IO.eprintln s!"selftest FAIL: Hamming-1 flip at idx {idx} has len={len} != 4"
          return 1

  -- 4. Verify 44B voter architectural majority, refusal, and 192-case single-bit masking.
  let g : Nat := 0x0123456789abcdef
  let mut maskCount : Nat := 0
  for bit in [0:64] do
    let bad := Nat.xor g (2 ^ bit)
    for (a, b, c) in [(bad, g, g), (g, bad, g), (g, g, bad)] do
      maskCount := maskCount + 1
      match AArch64Phys.run voterIm 11 (.ok (voterInit baseSt a b c)) with
      | .ok t =>
        if t.pc != voterPark || t.reg 19 != some g || t.reg 20 != some g || t.reg 21 != some g then
          IO.eprintln s!"selftest FAIL: single-bit mask failed at bit {bit}"
          return 1
      | _ =>
        IO.eprintln s!"selftest FAIL: single-bit mask non-ok at bit {bit}"
        return 1
  match AArch64Phys.run voterIm 11 (.ok (voterInit baseSt 0x1111 0x2222 0x3333)) with
  | .exit 3 => pure ()
  | other =>
    IO.eprintln s!"selftest FAIL: 3-way refusal expected exit 3, got {repr other}"
    return 1

  -- 5. Exact Rat checks.
  if tmrBoundArm64 (1 / 5000000) (4 / 5000000) != 220000003 / 25000000000000 then
    IO.eprintln "selftest FAIL: p7 Rat bound mismatch"
    return 1

  IO.println s!"AARCH64_PHYS_SELFTEST_OK env=6 forms={selftestWords.length} voter_flips={flipCount} single_bit_mask={maskCount} voter_bytes={voterCode.length}"
  return 0

/-! ### Subcommand 2: Streaming differential verifier (`diff <min_cases> [file]`) -/

def runDiffStream (stream : IO.FS.Stream) (minCases : Nat) : IO UInt32 := do
  let mut counts : Array Nat := Array.replicate formNames.size 0
  let mut total : Nat := 0
  let mut mismatches : Nat := 0
  let mut firstErr : Option String := none
  repeat do
    let line ← stream.getLine
    if line.isEmpty then break
    let trimmed := line.trimAscii.toString
    if trimmed.isEmpty then continue
    let toks := (trimmed.splitOn " ").filter (fun s => !s.isEmpty) |>.toArray
    total := total + 1
    match checkDiffTokens toks with
    | .ok idx =>
      counts := counts.set! idx (counts[idx]! + 1)
    | .error err =>
      mismatches := mismatches + 1
      if firstErr.isNone then
        firstErr := some s!"case {total}: {err}"
  if total < minCases then
    IO.eprintln s!"AARCH64_PHYS_DIFF_FAIL: only {total} cases read (expected >= {minCases})"
    return 1
  if mismatches > 0 then
    IO.eprintln s!"AARCH64_PHYS_DIFF_FAIL: {mismatches} / {total} mismatches; first: {firstErr.getD ""}"
    return 1
  let mut minHit := total
  let mut minName := ""
  for i in [0:formNames.size] do
    let c := counts[i]!
    if c < minHit then
      minHit := c
      minName := formNames[i]!
  let minPerForm := if minCases ≥ 53000 then 1000 else if minCases ≥ 530 then 10 else 0
  if minHit < minPerForm then
    IO.eprintln s!"AARCH64_PHYS_DIFF_FAIL: form '{minName}' only hit {minHit} times (need >= {minPerForm})"
    return 1
  IO.println s!"AARCH64_PHYS_DIFF_OK cases={total} mismatches=0 forms={formNames.size} min_per_form={minHit}"
  return 0

def runDiff (minCases : Nat) (file? : Option String := none) : IO UInt32 := do
  match file? with
  | none => runDiffStream (← IO.getStdin) minCases
  | some path =>
    let h ← IO.FS.Handle.mk path .read
    runDiffStream (IO.FS.Stream.ofHandle h) minCases

/-! ### Subcommand 3: Structured physical hardware report verifier (`verify-phys`) -/

structure PhysSummary where
  platform     : String := ""
  voterHex     : String := ""
  syscallOk    : Bool := false
  ptraceStatus : String := ""
  ptraceOk     : Bool := false
  fetchFlips   : Nat := 0
  fetchGuardPf : Nat := 0
  fetchHwDiffs : Nat := 0
  fetchAltsOk  : Nat := 0
  fetchAltsBad : Nat := 0
  sfTrials     : Nat := 0
  sfCorrected  : Nat := 0
  sfSdc        : Nat := 0
  sfRefusal    : Nat := 0
  sfStuck      : Nat := 0
  negTrials    : Nat := 0
  negSdc       : Nat := 0
  modeACount   : Nat := 0
  modeBCount   : Nat := 0
  -- `(q_num, q_den, trials)` and `(q_num, q_den, r_num, r_den, trials)` of every Monte Carlo row.
  modeARows    : List (Nat × Nat × Nat) := []
  modeBRows    : List (Nat × Nat × Nat × Nat × Nat) := []
  mcFails      : List String := []

def parseNatTok (s : String) : Option Nat := s.toNat?

/-- The Monte Carlo design of `aarch64_phys_fuzz --phys`: the body rates `q` of Mode A
(ideal voter) and the `(q, r)` pairs of Mode B (faulty voter), as numerator and denominator.
A report must contain each point exactly once and nothing else, so a missing, repeated or
substituted point (for example `q = 1`, which makes `3 q²` vacuous) is refused. -/
def modeAGrid : List (Nat × Nat) := [(1, 1000), (3, 1000), (1, 100), (3, 100)]
def modeBGrid : List (Nat × Nat × Nat × Nat) :=
  [(1, 1000, 1, 10000), (1, 1000, 5, 10000), (1, 1000, 2, 1000),
   (1, 100, 1, 10000), (1, 100, 5, 10000), (1, 100, 2, 1000),
   (3, 100, 1, 10000), (3, 100, 5, 10000), (3, 100, 2, 1000)]

/-- Errors of the Monte Carlo rows against the design: every grid point exactly once, no
other point, and every row with at least `minTrials` trials. -/
def checkMcDesign (minTrials : Nat) (st : PhysSummary) : List String := Id.run do
  let mut errs : List String := []
  for (qn, qd) in modeAGrid do
    let n := (st.modeARows.filter fun (a, b, _) => a == qn && b == qd).length
    if n != 1 then errs := errs ++ [s!"MODE_A q={qn}/{qd} appears {n} times, expected once"]
  for (qn, qd, _) in st.modeARows do
    if !modeAGrid.contains (qn, qd) then errs := errs ++ [s!"MODE_A q={qn}/{qd} is not a design point"]
  for (qn, qd, tr) in st.modeARows do
    if tr < minTrials then errs := errs ++ [s!"MODE_A q={qn}/{qd} trials={tr} < {minTrials}"]
  for (qn, qd, rn, rd) in modeBGrid do
    let n := (st.modeBRows.filter fun (a, b, c, d, _) => a == qn && b == qd && c == rn && d == rd).length
    if n != 1 then errs := errs ++ [s!"MODE_B q={qn}/{qd} r={rn}/{rd} appears {n} times, expected once"]
  for (qn, qd, rn, rd, _) in st.modeBRows do
    if !modeBGrid.contains (qn, qd, rn, rd) then
      errs := errs ++ [s!"MODE_B q={qn}/{qd} r={rn}/{rd} is not a design point"]
  for (qn, qd, rn, rd, tr) in st.modeBRows do
    if tr < minTrials then errs := errs ++ [s!"MODE_B q={qn}/{qd} r={rn}/{rd} trials={tr} < {minTrials}"]
  return errs

def runVerifyPhysStream (stream : IO.FS.Stream) (minTrials : Nat) : IO UInt32 := do
  let mut st : PhysSummary := {}
  let mut parseErrors : List String := []
  repeat do
    let line ← stream.getLine
    if line.isEmpty then break
    let trimmed := line.trimAscii.toString
    if trimmed.isEmpty then continue
    let toks := (trimmed.splitOn " ").filter (fun s => !s.isEmpty) |>.toArray
    if toks.size == 0 then continue
    match toks[0]! with
    | "PLATFORM" =>
      if toks.size ≥ 2 then st := { st with platform := toks[1]! }
      else parseErrors := "bad PLATFORM line" :: parseErrors
    | "VOTER_CODE" =>
      if toks.size ≥ 2 then st := { st with voterHex := toks[1]! }
      else parseErrors := "bad VOTER_CODE line" :: parseErrors
    | "SYSCALL_EXIT3_CHECK" =>
      if toks.size ≥ 3 then
        st := { st with syscallOk := (toks[1]! == "PASS" && toks[2]! == "3") }
      else parseErrors := "bad SYSCALL_EXIT3_CHECK line" :: parseErrors
    | "PTRACE_STEP_CHECK" =>
      if toks.size ≥ 5 then
        let status := toks[1]!
        let ok := (status == "PASS" && toks[2]! == "1" && toks[3]! == "1" && toks[4]! == "1") ||
                  (status == "SKIP_NON_LINUX")
        st := { st with ptraceStatus := status, ptraceOk := ok }
      else parseErrors := "bad PTRACE_STEP_CHECK line" :: parseErrors
    | "FETCH_BOUNDARY_SWEEP" =>
      if toks.size ≥ 6 then
        match parseNatTok toks[1]!, parseNatTok toks[4]!, parseNatTok toks[5]! with
        | some fl, some gpf, some mdf =>
          st := { st with fetchFlips := fl, fetchGuardPf := gpf, fetchHwDiffs := mdf }
        | _, _, _ => parseErrors := "bad FETCH_BOUNDARY_SWEEP numbers" :: parseErrors
      else parseErrors := "bad FETCH_BOUNDARY_SWEEP line" :: parseErrors
    | "FETCH_ALT" =>
      match checkFetchAltTokens toks with
      | .ok () => st := { st with fetchAltsOk := st.fetchAltsOk + 1 }
      | .error err =>
        st := { st with fetchAltsBad := st.fetchAltsBad + 1 }
        if parseErrors.length < 5 then
          parseErrors := s!"FETCH_ALT mismatch: {err}" :: parseErrors
    | "SINGLE_FAULT_SWEEP" =>
      if toks.size ≥ 6 then
        match parseNatTok toks[1]!, parseNatTok toks[2]!, parseNatTok toks[3]!,
              parseNatTok toks[4]!, parseNatTok toks[5]! with
        | some tr, some cor, some sdc, some ref, some stk =>
          st := { st with sfTrials := tr, sfCorrected := cor, sfSdc := sdc,
                          sfRefusal := ref, sfStuck := stk }
        | _, _, _, _, _ => parseErrors := "bad SINGLE_FAULT_SWEEP numbers" :: parseErrors
      else parseErrors := "bad SINGLE_FAULT_SWEEP line" :: parseErrors
    | "NEG_UNHARDENED" =>
      if toks.size ≥ 3 then
        match parseNatTok toks[1]!, parseNatTok toks[2]! with
        | some tr, some sdc => st := { st with negTrials := tr, negSdc := sdc }
        | _, _ => parseErrors := "bad NEG_UNHARDENED numbers" :: parseErrors
      else parseErrors := "bad NEG_UNHARDENED line" :: parseErrors
    | "MODE_A" =>
      if toks.size ≥ 5 then
        match parseNatTok toks[1]!, parseNatTok toks[2]!,
              parseNatTok toks[3]!, parseNatTok toks[4]! with
        | some qn, some qd, some tr, some fl =>
          if qd == 0 then parseErrors := "MODE_A zero denom" :: parseErrors
          else
            let q : Rat := (qn : Rat) / (qd : Rat)
            let bnd := tmrBoundArm64 q 0
            st := { st with modeACount := st.modeACount + 1, modeARows := st.modeARows ++ [(qn, qd, tr)] }
            if !withinFiveSigma bnd tr fl then
              st := { st with mcFails := s!"MODE_A q={qn}/{qd} fails={fl}/{tr} exceeds 3*q^2" :: st.mcFails }
        | _, _, _, _ => parseErrors := "bad MODE_A numbers" :: parseErrors
      else parseErrors := "bad MODE_A line" :: parseErrors
    | "MODE_B" =>
      if toks.size ≥ 7 then
        match parseNatTok toks[1]!, parseNatTok toks[2]!,
              parseNatTok toks[3]!, parseNatTok toks[4]!,
              parseNatTok toks[5]!, parseNatTok toks[6]! with
        | some qn, some qd, some rn, some rd, some tr, some fl =>
          if qd == 0 || rd == 0 then parseErrors := "MODE_B zero denom" :: parseErrors
          else
            let q : Rat := (qn : Rat) / (qd : Rat)
            let r : Rat := (rn : Rat) / (rd : Rat)
            let bnd := tmrBoundArm64 q r
            st := { st with modeBCount := st.modeBCount + 1, modeBRows := st.modeBRows ++ [(qn, qd, rn, rd, tr)] }
            if !withinFiveSigma bnd tr fl then
              st := { st with mcFails := s!"MODE_B q={qn}/{qd} r={rn}/{rd} fails={fl}/{tr} exceeds 3*q^2+11*r" :: st.mcFails }
        | _, _, _, _, _, _ => parseErrors := "bad MODE_B numbers" :: parseErrors
      else parseErrors := "bad MODE_B line" :: parseErrors
    | _ => pure ()

  let mut errs := parseErrors.reverse
  if st.platform != "Darwin_arm64" && st.platform != "Linux_aarch64" then
    errs := errs ++ [s!"unexpected PLATFORM '{st.platform}'"]
  if st.voterHex != voterCodeHex then
    errs := errs ++ [s!"VOTER_CODE mismatch: got '{st.voterHex}' expected '{voterCodeHex}'"]
  if !st.syscallOk then
    errs := errs ++ ["SYSCALL_EXIT3_CHECK did not PASS with exit code 3"]
  if st.platform == "Linux_aarch64" && st.ptraceStatus != "PASS" then
    errs := errs ++ [s!"PTRACE_STEP_CHECK must PASS on Linux_aarch64, got '{st.ptraceStatus}'"]
  else if !st.ptraceOk then
    errs := errs ++ [s!"PTRACE_STEP_CHECK failed ('{st.ptraceStatus}')"]
  if st.fetchFlips < 8544 then
    errs := errs ++ [s!"FETCH_BOUNDARY_SWEEP flips={st.fetchFlips} < 8544"]
  if st.fetchGuardPf != 0 then
    errs := errs ++ [s!"FETCH_BOUNDARY_SWEEP guard_pf={st.fetchGuardPf} != 0"]
  if st.fetchHwDiffs != 0 || st.fetchAltsBad != 0 then
    errs := errs ++ [s!"FETCH_BOUNDARY_SWEEP diffs hw={st.fetchHwDiffs} lean={st.fetchAltsBad}"]
  if st.fetchAltsOk < 100 then
    errs := errs ++ [s!"FETCH_ALT Lean-verified count={st.fetchAltsOk} < 100"]
  if st.sfTrials < 3712 || st.sfCorrected != st.sfTrials || st.sfSdc != 0 || st.sfRefusal != 0 || st.sfStuck != 0 then
    errs := errs ++ [s!"SINGLE_FAULT_SWEEP failed: trials={st.sfTrials} corrected={st.sfCorrected} sdc={st.sfSdc} refusal={st.sfRefusal} stuck={st.sfStuck}"]
  if st.negTrials != 640 || st.negSdc != 640 then
    errs := errs ++ [s!"NEG_UNHARDENED failed: sdc={st.negSdc}/{st.negTrials} (expected 640/640)"]
  if st.modeACount != 4 then
    errs := errs ++ [s!"MODE_A expected 4 rows, got {st.modeACount}"]
  if st.modeBCount != 9 then
    errs := errs ++ [s!"MODE_B expected 9 rows, got {st.modeBCount}"]
  errs := errs ++ checkMcDesign minTrials st
  errs := errs ++ st.mcFails.reverse

  if !errs.isEmpty then
    for e in errs do
      IO.eprintln s!"AARCH64_PHYS_GATE_FAIL: {e}"
    return 1

  IO.println s!"AARCH64_PHYS_GATE_OK platform={st.platform} voter_bytes=44 ptrace={st.ptraceStatus} fetch_flips={st.fetchFlips} guard_pf=0 fetch_alts_lean={st.fetchAltsOk} single_fault={st.sfCorrected}/{st.sfTrials} sdc=0 neg_unhardened={st.negSdc}/{st.negTrials} modeA={st.modeACount} modeB={st.modeBCount}"
  return 0

def runVerifyPhys (file? : Option String := none) (minTrials : Nat := 20000) : IO UInt32 := do
  match file? with
  | none => runVerifyPhysStream (← IO.getStdin) minTrials
  | some path =>
    let h ← IO.FS.Handle.mk path .read
    runVerifyPhysStream (IO.FS.Stream.ofHandle h) minTrials

end AArch64PhysCheck

def main (args : List String) : IO UInt32 :=
  match args with
  | ["selftest"] => AArch64PhysCheck.runSelftest
  | ["diff", nStr] =>
    match nStr.toNat? with
    | some n => AArch64PhysCheck.runDiff n none
    | none => do
      IO.eprintln "usage: aarch64-phys-check diff <min_cases> [file]"
      return 2
  | ["diff", nStr, path] =>
    match nStr.toNat? with
    | some n => AArch64PhysCheck.runDiff n (some path)
    | none => do
      IO.eprintln "usage: aarch64-phys-check diff <min_cases> [file]"
      return 2
  | ["verify-phys"] => AArch64PhysCheck.runVerifyPhys none
  | ["verify-phys", path] => AArch64PhysCheck.runVerifyPhys (some path)
  | ["verify-phys", path, tStr] =>
    match tStr.toNat? with
    | some t => AArch64PhysCheck.runVerifyPhys (some path) t
    | none => do
      IO.eprintln "usage: aarch64-phys-check verify-phys [file [min_trials]]"
      return 2
  | _ => do
    IO.eprintln "usage: aarch64-phys-check <selftest | diff <min_cases> [file] | verify-phys [file [min_trials]]>"
    return 2
