import Bootstrap.X86Uni

/-!
Differential model runner for `X86Uni` against real x86-64 hardware traces generated
by `bootstrap/model/x86uni_diff_cpu.cpp`.

Each input line has the form:
  <tag> <instrHex> <r0> ... <r15> <cszo> <mem128hex> | <afterRip> <r0'> ... <r15'> <cszo'> <mem128hex'>

For every case, this runner:
1. Decodes `<instrHex>` via `X86Uni.decode` at `INS_ADDR = 0x20180` and asserts `len == bytes.length`.
2. Executes `X86Uni.exec` on the exact pre-state `(regs, INS_ADDR, fl, [⟨WIN_LO, preMem, true⟩])`.
3. Asserts exact agreement on post-RIP, all 16 64-bit GPRs, all defined architectural flags
   (`CF`, `ZF`, `SF`, `OF`, ignoring `none`), and all 64 bytes of stack window `[0x117e0, 0x11820)`.
4. Classifies each case by decoded `X86Uni.Ins` form and enforces total cases `>= 50000`,
   every narrow form bucket `>= 1000` hits, and full coverage across all `X86Dec` forms.
-/

namespace X86UniDiffModel

open X86Env X86Uni

def WIN_LO : Nat := 0x117e0
def WIN_SIZE : Nat := 64
def INS_ADDR : Nat := 0x20180
def W64 : Nat := 2 ^ 64

private def hexVal? (c : Char) : Option Nat :=
  if '0' ≤ c && c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 'a'.toNat + 10)
  else if 'A' ≤ c && c ≤ 'F' then some (c.toNat - 'A'.toNat + 10)
  else none

private def parseHexNat? (s : String) : Option Nat :=
  if s.isEmpty then none
  else
    s.toList.foldlM (fun acc c => do
      let d ← hexVal? c
      pure (acc * 16 + d)) 0

private def parseHexBytes? (s : String) : Option (List Nat) :=
  let cs := s.toList
  if cs.length % 2 ≠ 0 || cs.isEmpty then none
  else
    let rec go : List Char → Option (List Nat)
      | [] => some []
      | a :: b :: rest => do
        let hi ← hexVal? a
        let lo ← hexVal? b
        let tl ← go rest
        pure ((hi * 16 + lo) :: tl)
      | _ => none
    go cs

private def parseFlags? (s : String) : Option Flags :=
  match s.toList with
  | [c, z, sf, o] =>
    let bit? : Char → Option Bool
      | '0' => some false
      | '1' => some true
      | _ => none
    do
      let cf ← bit? c
      let zf ← bit? z
      let sfb ← bit? sf
      let ofb ← bit? o
      pure ⟨some cf, some zf, some sfb, some ofb⟩
  | _ => none

private def flagMatch (m : Option Bool) (cpu : Bool) : Bool :=
  match m with
  | none => true
  | some b => b == cpu

private def flagsAgree (m cpu : Flags) : Bool :=
  match cpu.cf, cpu.zf, cpu.sf, cpu.of with
  | some c, some z, some s, some o =>
    flagMatch m.cf c && flagMatch m.zf z && flagMatch m.sf s && flagMatch m.of o
  | _, _, _, _ => false

private def extractMemWindow (mem : Mem) : Option (List Nat) :=
  (List.range WIN_SIZE).mapM fun off => mem.rd (WIN_LO + off)

private def extractRegs (s : St) : Option (List Nat) :=
  (List.range 16).mapM fun r => s.reg r

private def hex16 (n : Nat) : String :=
  let digits := (List.range 16).reverse.map fun i =>
    let d := (n / 16 ^ i) % 16
    if d < 10 then Char.ofNat ('0'.toNat + d) else Char.ofNat ('a'.toNat + d - 10)
  String.ofList digits

private def classifyForm (bytes : List Nat) (ins : Ins) : String :=
  match ins with
  | .alImm8 .add _ => "narrow_alImm8_add"
  | .alImm8 .sub _ => "narrow_alImm8_sub"
  | .alImm8 .cmp _ => "narrow_alImm8_cmp"
  | .rr32 .xor _ _ => "narrow_rr32_xor"
  | .rr32 .or _ _ => "narrow_rr32_or"
  | .rr32 .test _ _ => "narrow_rr32_test"
  | .rr32 .mov _ _ => "narrow_rr32_mov"
  | .cmpI32 _ _ =>
      let b0 := bytes.headD 0
      let opc := if 0x40 ≤ b0 && b0 ≤ 0x4f then bytes[1]?.getD 0 else b0
      if opc == 0x81 then "narrow_cmpI32_81" else "narrow_cmpI32_83"
  | .shl32 _ c =>
      let cnt := c % 32
      if cnt == 0 then "narrow_shl32_cnt0"
      else if cnt == 1 then "narrow_shl32_cnt1"
      else "narrow_shl32_cntgt1"
  | .inc 32 _ => "narrow_inc32"
  | .inc 64 _ => "narrow_inc64"
  | .inc _ _ => "narrow_inc_other"
  | .ofInt i =>
      let s := String.ofList ((reprStr i).toList.drop 11) -- strip "X86Int.Ins."
      let parts := s.splitOn " "
      match parts with
      | [] => "int_unknown"
      | ctor :: rest =>
        if ctor == "alu" || ctor == "aluI" then
          let op := (rest.headD "").replace "(" "" |>.replace ")" ""
          let opShort := (op.splitOn ".").getLastD op
          s!"int_{ctor}_{opShort}"
        else if ctor == "movI" then
          if bytes.length <= 6 then "int_movI32" else "int_movI64"
        else
          s!"int_{ctor}"

private def requiredNarrowForms : List String :=
  [ "narrow_alImm8_add"
  , "narrow_alImm8_sub"
  , "narrow_alImm8_cmp"
  , "narrow_rr32_xor"
  , "narrow_rr32_or"
  , "narrow_rr32_test"
  , "narrow_cmpI32_81"
  , "narrow_cmpI32_83"
  , "narrow_shl32_cnt0"
  , "narrow_shl32_cnt1"
  , "narrow_shl32_cntgt1"
  , "narrow_inc32"
  , "narrow_inc64"
  ]

private def requiredIntForms : List String :=
  [ "int_movI32"
  , "int_movI64"
  , "int_alu_add"
  , "int_alu_sub"
  , "int_alu_and"
  , "int_alu_or"
  , "int_alu_xor"
  , "int_mov32"
  , "int_mov"
  , "int_ld"
  , "int_st"
  , "int_ld32"
  , "int_test"
  , "int_pushI"
  , "int_cmp"
  , "int_aluI_add"
  , "int_aluI_or"
  , "int_aluI_and"
  , "int_aluI_sub"
  , "int_aluI_xor"
  , "int_cmpI"
  , "int_shlI"
  , "int_shrI"
  , "int_lea"
  , "int_st8"
  , "int_ld8"
  , "int_jcc"
  , "int_jmp"
  , "int_call"
  , "int_ret"
  , "int_push"
  , "int_pop"
  , "int_jtab"
  ]

structure Counts where
  items : Array (String × Nat) := #[]

private def Counts.inc (c : Counts) (k : String) : Counts :=
  Id.run do
    for i in [:c.items.size] do
      let (name, cnt) := c.items[i]!
      if name == k then
        return { items := c.items.set! i (name, cnt + 1) }
    return { items := c.items.push (k, 1) }

private def Counts.get (c : Counts) (k : String) : Nat :=
  Id.run do
    for (name, cnt) in c.items do
      if name == k then
        return cnt
    return 0

private def optToExcept {α : Type} (o : Option α) (msg : String) : Except String α :=
  match o with
  | some a => .ok a
  | none => .error msg

private def checkLine (idx : Nat) (line : String) : Except String String := do
  let toks := line.splitOn " " |>.filter (fun t => !t.isEmpty)
  -- Expected: tag, instr, 16 preRegs, preFlags, preMem, "|", postRip, 16 postRegs, postFlags, postMem
  -- Total tokens = 1 + 1 + 16 + 1 + 1 + 1 + 1 + 16 + 1 + 1 = 40
  if toks.length ≠ 40 then
    throw s!"line {idx}: expected 40 tokens, got {toks.length}"
  let tag := toks[0]!
  let instrHex := toks[1]!
  let preRegStrs := (toks.drop 2).take 16
  let preFlStr := toks[18]!
  let preMemStr := toks[19]!
  let pipeStr := toks[20]!
  let postRipStr := toks[21]!
  let postRegStrs := (toks.drop 22).take 16
  let postFlStr := toks[38]!
  let postMemStr := toks[39]!
  if pipeStr ≠ "|" then
    throw s!"line {idx}: expected '|', got '{pipeStr}'"
  let bytes ← optToExcept (parseHexBytes? instrHex) s!"line {idx}: bad instr hex '{instrHex}'"
  let preRegs ← optToExcept (preRegStrs.mapM parseHexNat?) s!"line {idx}: bad pre-regs"
  let preFl ← optToExcept (parseFlags? preFlStr) s!"line {idx}: bad pre-flags '{preFlStr}'"
  let preMem ← optToExcept (parseHexBytes? preMemStr) s!"line {idx}: bad pre-mem"
  if preMem.length ≠ WIN_SIZE then
    throw s!"line {idx}: pre-mem len {preMem.length} != {WIN_SIZE}"
  let cpuRip ← optToExcept (parseHexNat? postRipStr) s!"line {idx}: bad post-rip '{postRipStr}'"
  let cpuRegs ← optToExcept (postRegStrs.mapM parseHexNat?) s!"line {idx}: bad post-regs"
  let cpuFl ← optToExcept (parseFlags? postFlStr) s!"line {idx}: bad post-flags '{postFlStr}'"
  let cpuMem ← optToExcept (parseHexBytes? postMemStr) s!"line {idx}: bad post-mem"
  if cpuMem.length ≠ WIN_SIZE then
    throw s!"line {idx}: post-mem len {cpuMem.length} != {WIN_SIZE}"
  let bFn : Nat → Option Nat := fun i => bytes[i]?
  match X86Uni.decode bFn INS_ADDR with
  | none => throw s!"line {idx} ({tag}): X86Uni.decode returned none for {instrHex}"
  | some (ins, len) =>
    if len ≠ bytes.length then
      throw s!"line {idx} ({tag}): decoded len {len} != bytes.length {bytes.length} for {instrHex}"
    let formKey := classifyForm bytes ins
    if !tag.isEmpty && tag ≠ formKey then
      -- allow tag matching prefix or exact match
      if tag ≠ formKey then
        throw s!"line {idx}: tag '{tag}' != decoded form '{formKey}' for {instrHex}"
    let s0 : St :=
      { regs := preRegs.map some
        rip := INS_ADDR
        fl := preFl
        mem := [⟨WIN_LO, preMem, true⟩]
        k := ⟨[], [], [], [], []⟩ }
    match X86Uni.exec ins len s0 with
    | .stuck => throw s!"line {idx} ({formKey}): X86Uni.exec got stuck for {instrHex}"
    | .exit c => throw s!"line {idx} ({formKey}): X86Uni.exec exited {c} unexpectedly"
    | .ok s1 =>
      if s1.rip % W64 ≠ cpuRip % W64 then
        throw s!"line {idx} ({formKey}): RIP mismatch model=0x{hex16 s1.rip} cpu=0x{hex16 cpuRip} ({instrHex})"
      match extractRegs s1 with
      | none => throw s!"line {idx} ({formKey}): undefined GPR in post-state"
      | some mRegs =>
        if mRegs ≠ cpuRegs then
          let diffs := (List.range 16).filterMap fun r =>
            let mv := mRegs[r]!
            let cv := cpuRegs[r]!
            if mv ≠ cv then some s!"r{r}:model=0x{hex16 mv},cpu=0x{hex16 cv}" else none
          throw s!"line {idx} ({formKey}): GPR mismatch {String.intercalate " " diffs} ({instrHex})"
        if !flagsAgree s1.fl cpuFl then
          throw s!"line {idx} ({formKey}): flag mismatch model={reprStr s1.fl} cpu={postFlStr} ({instrHex})"
        match extractMemWindow s1.mem with
        | none => throw s!"line {idx} ({formKey}): undefined mem window"
        | some mMem =>
          if mMem ≠ cpuMem then
            throw s!"line {idx} ({formKey}): stack memory mismatch ({instrHex})"
          pure formKey

def runDiffFile (path : System.FilePath) (minTotal minNarrow : Nat) : IO UInt32 := do
  let raw ← IO.FS.readFile path
  let lines := raw.splitOn "\n" |>.map (fun l => l.trimAscii.toString) |>.filter (fun l => !l.isEmpty)
  let mut total : Nat := 0
  let mut mismatches : Nat := 0
  let mut firstErrs : Array String := #[]
  let mut counts : Counts := {}
  for line in lines do
    total := total + 1
    match checkLine total line with
    | .ok key =>
      counts := counts.inc key
    | .error err =>
      mismatches := mismatches + 1
      if firstErrs.size < 10 then
        firstErrs := firstErrs.push err
  IO.println s!"X86UniDiffModel: checked {total} cases, mismatches={mismatches}"
  for (k, cnt) in counts.items do
    IO.println s!"  {k}: {cnt}"
  if mismatches > 0 then
    IO.eprintln "FAIL: differential mismatches detected:"
    for e in firstErrs do
      IO.eprintln s!"  {e}"
    return 1
  if total < minTotal then
    IO.eprintln s!"FAIL: total cases {total} < required {minTotal}"
    return 1
  for nf in requiredNarrowForms do
    let c := counts.get nf
    if c < minNarrow then
      IO.eprintln s!"FAIL: narrow form {nf} has {c} hits < required {minNarrow}"
      return 1
  for inf in requiredIntForms do
    let c := counts.get inf
    if c == 0 then
      IO.eprintln s!"FAIL: X86Dec form {inf} has 0 hits"
      return 1
  IO.println "PASS: 0 disagreements across all X86Uni narrow + X86Dec forms"
  return 0

end X86UniDiffModel

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] => X86UniDiffModel.runDiffFile path 50000 1000
  | [path, minTot] =>
    match minTot.toNat? with
    | some n => X86UniDiffModel.runDiffFile path n 1000
    | none => IO.eprintln "usage: X86UniDiffModel <cases.txt> [min_total]"; pure 2
  | _ =>
    IO.eprintln "usage: X86UniDiffModel <cases.txt> [min_total]"
    pure 2
