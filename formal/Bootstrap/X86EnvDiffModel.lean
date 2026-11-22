/-
The model side of the process-environment differential test for `X86Env`,
`X86Int` (`syscall`), and `X86Dec` (`0f 05`).

  lake build x86env-diff && .lake/build/bin/x86env-diff < cpu.txt > model.txt
  (or: lean --run Bootstrap/X86EnvDiffModel.lean < cpu.txt > model.txt)

Reads the multi-step syscall scenario (`SCEN`) and `execve` initial-stack
(`STACK`) lines emitted by `bootstrap/model/x86env_diff_cpu`, evaluates every
step through `X86Dec.decode` and `X86Int.step` (threading `Kern` and `Mem`
across all steps of each scenario and checking `Mem.wf` at every step), and
verifies `X86Env.initStack` on every `STACK` line. Writes `-` for `r11` after
`syscall`, which the architecture leaves undefined. Any refused, stuck, or
invariant-violating step fails the harness.
-/
import Bootstrap.X86Dec

open X86Env X86Int X86Dec

def mem0Base : Nat := 0x100000
def probeLen : Nat := 16

def hexVal (c : Char) : Option Nat :=
  if '0' ≤ c && c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 'a'.toNat + 10) else none

def parseHex (s : String) : Option Nat :=
  if s.isEmpty then none else
  s.foldl (fun acc c => do let a ← acc; let v ← hexVal c; pure (a * 16 + v)) (some 0)

def bytesOf (s : String) : Option (List Nat) :=
  let rec go : List Char → Option (List Nat)
    | [] => some []
    | a :: b :: rest => do
      let x ← hexVal a; let y ← hexVal b; let r ← go rest; pure ((x * 16 + y) :: r)
    | _ => none
  go s.toList

def hexDigit (n : Nat) : Char :=
  match n with
  | 0 => '0' | 1 => '1' | 2 => '2' | 3 => '3'
  | 4 => '4' | 5 => '5' | 6 => '6' | 7 => '7'
  | 8 => '8' | 9 => '9' | 10 => 'a' | 11 => 'b'
  | 12 => 'c' | 13 => 'd' | 14 => 'e' | _ => 'f'

def hex2 (v : Nat) : String :=
  String.ofList [hexDigit ((v / 16) % 16), hexDigit (v % 16)]

def hex4 (v : Nat) : String :=
  String.ofList [
    hexDigit ((v / 4096) % 16),
    hexDigit ((v / 256) % 16),
    hexDigit ((v / 16) % 16),
    hexDigit (v % 16)
  ]

def hex16 (v : Nat) : String :=
  let u := UInt64.ofNat v
  let rec go (i : Nat) (acc : List Char) : List Char :=
    match i with
    | 0 => acc
    | i' + 1 =>
      let nib := ((u >>> UInt64.ofNat (4 * (15 - i'))) &&& 15).toNat
      go i' (hexDigit nib :: acc)
  String.ofList (go 16 [])

def bytesHex (bs : List Nat) : String :=
  String.join (bs.map hex2)

def flagCh : Option Bool → Char
  | none => '-'
  | some true => '1'
  | some false => '0'

def stateStr (regs : List (Option Nat)) (fl : Flags) (probe : List Nat) (outBytes : List Nat) : String :=
  let r := " ".intercalate (regs.map fun o => match o with | some v => hex16 v | none => "----------------")
  let f := String.ofList [flagCh fl.cf, flagCh fl.sf, flagCh fl.zf, flagCh fl.of]
  let m := bytesHex probe
  let o := if outBytes.isEmpty then "-" else bytesHex outBytes
  s!"{r} {f} {m} {o}"

/-- Computable pairwise disjointness check on `Mem`. -/
def pairwiseDisjoint : Mem → Bool
  | [] => true
  | r :: rest => (rest.all fun s => decide (r.hi ≤ s.base ∨ s.hi ≤ r.base)) && pairwiseDisjoint rest

/-- Computable check for `X86Env.Mem.wf`. -/
def memWfBool (m : Mem) : Bool :=
  (m.all fun r => r.base % page == 0) && pairwiseDisjoint m

theorem pairwiseDisjoint_spec : ∀ (m : Mem), pairwiseDisjoint m = true →
    m.Pairwise (fun r s => r.hi ≤ s.base ∨ s.hi ≤ r.base)
  | [], _ => List.Pairwise.nil
  | r :: rest, h => by
    simp only [pairwiseDisjoint, Bool.and_eq_true] at h
    refine List.Pairwise.cons ?_ (pairwiseDisjoint_spec rest h.2)
    intro s hs
    have hall := List.all_eq_true.1 h.1 s hs
    exact of_decide_eq_true hall

theorem memWfBool_spec {m : Mem} (h : memWfBool m = true) : m.wf := by
  simp only [memWfBool, Bool.and_eq_true] at h
  refine ⟨?_, pairwiseDisjoint_spec m h.2⟩
  intro r hr
  have hall := List.all_eq_true.1 h.1 r hr
  exact beq_iff_eq.1 hall

def catNames : Array String := #[
  "open_ok", "open_enoent",
  "read_full", "read_short", "read_eof", "read_ebadf",
  "write_ok", "write_ebadf",
  "close_ok", "close_ebadf",
  "mmap_ok", "mmap_einval", "mmap_enomem",
  "mremap_inplace", "mremap_moved",
  "munmap_ok",
  "exit",
  "init_stack"
]

def catIdxOf (name : String) : Option Nat :=
  let rec go (i : Nat) : Option Nat :=
    if h : i < catNames.size then
      if catNames[i] == name then some i else go (i + 1)
    else none
  go 0

def parseFiles (s : String) : Option Files :=
  if s == "-" then some [] else
  (s.splitOn ",").mapM fun item =>
    match item.splitOn ":" with
    | [nh, dh] => do
      let nb ← bytesOf nh
      let db ← bytesOf dh
      pure (nb, db)
    | _ => none

def parseOptBytes (s : String) : Option (List Nat) :=
  if s == "-" then some [] else bytesOf s

def parseChoice (s : String) : Option Choice :=
  if s == "nomem" then some .nomem
  else match s.splitOn ":" with
  | ["at", ah] => do let a ← parseHex ah; pure (.at a)
  | ["cnt", nh] => do let n ← parseHex nh; pure (.count n)
  | _ => none

def parseOracle (s : String) : Option (List Choice) :=
  if s == "-" then some [] else (s.splitOn ",").mapM parseChoice

def availReadBytes (k : Kern) (fdNum : Nat) : Option Nat :=
  if fdNum == 0 then some k.stdin.length
  else if 3 ≤ fdNum then
    match k.fds[fdNum - 3]? with
    | some (some fd) => some (fd.data.drop fd.pos).length
    | _ => none
  else none

def checkStepCat (cat : String) (kPre : Kern) (mPre : Mem)
    (nr a1 a2 a3 : Nat) (rax : Nat) : Bool :=
  match cat with
  | "open_ok" => nr == 2 && 3 ≤ rax && rax < W - 4096
  | "open_enoent" => nr == 2 && rax == neg ENOENT
  | "read_full" =>
    match availReadBytes kPre a1 with
    | some avail => nr == 0 && 0 < rax && rax == min a3 avail
    | none => false
  | "read_short" =>
    match availReadBytes kPre a1 with
    | some avail => nr == 0 && 0 < rax && rax < min a3 avail
    | none => false
  | "read_eof" =>
    match availReadBytes kPre a1 with
    | some avail => nr == 0 && 0 < a3 && avail == 0 && rax == 0
    | none => false
  | "read_ebadf" => nr == 0 && rax == neg EBADF
  | "write_ok" => nr == 1 && a1 == 1 && 0 < rax && rax ≤ a3
  | "write_ebadf" => nr == 1 && a1 != 1 && rax == neg EBADF
  | "close_ok" => nr == 3 && rax == 0
  | "close_ebadf" => nr == 3 && rax == neg EBADF
  | "mmap_ok" => nr == 9 && 0 < a2 && rax % page == 0 && 0 < rax && rax + pages a2 ≤ 2 ^ 47
  | "mmap_einval" => nr == 9 && a2 == 0 && rax == neg EINVAL
  | "mmap_enomem" => nr == 9 && 0 < a2 && rax == neg ENOMEM
  | "mremap_inplace" =>
    nr == 25 && rax == a1 && pages a2 < pages a3 && (mPre.find a1 0).isSome
  | "mremap_moved" =>
    nr == 25 && rax != a1 && rax % page == 0 && 0 < rax && rax + pages a3 ≤ 2 ^ 47 &&
      pages a2 < pages a3
  | "munmap_ok" => nr == 11 && rax == 0
  | _ => false

def evalSteps : List String → Kern → Mem → List String → List Nat → (List String × List Nat × Bool)
  | [], k, _, accOut, accCats =>
    if k.oracle.isEmpty then (accOut.reverse, accCats.reverse, false)
    else ((s!"ORACLE_LEFT_{k.oracle.length}" :: accOut).reverse, accCats.reverse, true)
  | stepStr :: rest, k, m, accOut, accCats =>
    let fs := stepStr.splitOn ":"
    if fs.length != 21 then
      ((s!"STEP_UNPARSED" :: accOut).reverse, accCats.reverse, true)
    else
      let cat := fs[0]!
      match catIdxOf cat, bytesOf fs[1]!, parseHex fs[2]!,
            (fs.drop 3 |>.take 16).mapM parseHex, parseHex fs[20]! with
      | some catIdx, some ins, some rip, some regs, some probeAddr =>
        let flChars := fs[19]!.toList
        if flChars.length != 4 then
          ((s!"BAD_FLAGS" :: accOut).reverse, accCats.reverse, true)
        else
          match decode (fun i => ins[i]?) rip with
          | some (.syscall, 2) =>
            let b (c : Char) : Option Bool := some (c == '1')
            let flags : Flags := ⟨b flChars[0]!, b flChars[2]!, b flChars[1]!, b flChars[3]!⟩
            let s : St := {
              regs := regs.map some,
              rip := rip,
              fl := flags,
              mem := m,
              k := k
            }
            let nr := regs[0]!
            let a1 := regs[7]!
            let a2 := regs[6]!
            let a3 := regs[2]!
            let im := imageOf ins rip
            match step im s with
            | .ok t =>
              if !memWfBool t.mem then
                ((s!"MEM_WF_FAIL" :: accOut).reverse, (catIdx :: accCats).reverse, true)
              else
                let rax := (t.reg 0).getD 0
                if !checkStepCat cat k m nr a1 a2 a3 rax then
                  ((s!"CAT_CHECK_FAIL_{cat}" :: accOut).reverse, (catIdx :: accCats).reverse, true)
                else
                  match t.mem.rdN probeAddr probeLen with
                  | some probeBytes =>
                    let delta := (t.rip + W - rip) % W
                    let outS := s!"{hex16 delta} {stateStr t.regs t.fl probeBytes t.k.out}"
                    evalSteps rest t.k t.mem (outS :: accOut) (catIdx :: accCats)
                  | none =>
                    ((s!"PROBE_READ_FAIL" :: accOut).reverse, (catIdx :: accCats).reverse, true)
            | .exit code =>
              if cat == "exit" && nr == 60 && code == a1 % 256 && rest.isEmpty && k.oracle.isEmpty then
                let outHex := if k.out.isEmpty then "-" else bytesHex k.out
                let outS := s!"EXIT_{hex2 code} {outHex}"
                ((outS :: accOut).reverse, (catIdx :: accCats).reverse, false)
              else
                ((s!"UNEXPECTED_EXIT_{code}" :: accOut).reverse, (catIdx :: accCats).reverse, true)
            | .stuck =>
              ((s!"STUCK_{cat}" :: accOut).reverse, (catIdx :: accCats).reverse, true)
          | _ =>
            ((s!"DECODE_FAIL" :: accOut).reverse, (catIdx :: accCats).reverse, true)
      | _, _, _, _, _ =>
        ((s!"STEP_PARSE_FAIL" :: accOut).reverse, accCats.reverse, true)

def evalScen (caseStr : String) : String × List Nat × Bool :=
  let fs := caseStr.splitOn " "
  if fs.length != 6 then
    (s!"{caseStr} | SCEN_UNPARSED", [], true)
  else
    match parseFiles fs[1]!, parseOptBytes fs[2]!, bytesOf fs[3]!, parseOracle fs[4]! with
    | some files, some stdinBytes, some m0Prefix, some oracle =>
      if m0Prefix.length != 64 then
        (s!"{caseStr} | BAD_M0_PREFIX", [], true)
      else
        let m0Data := m0Prefix ++ List.replicate (page - 64) 0
        let m0 : Mem := [⟨mem0Base, m0Data, true⟩]
        if !memWfBool m0 then
          (s!"{caseStr} | INIT_MEM_WF_FAIL", [], true)
        else
          let k0 : Kern := ⟨files, [], stdinBytes, [], oracle⟩
          let stepStrs := fs[5]!.splitOn ";"
          let (stepOuts, cats, failed) := evalSteps stepStrs k0 m0 [] []
          let joined := ";".intercalate stepOuts
          (s!"{caseStr} | {joined}", cats, failed)
    | _, _, _, _ =>
      (s!"{caseStr} | SCEN_PARSE_FAIL", [], true)

def evalStack (caseStr : String) : String × List Nat × Bool :=
  let fs := caseStr.splitOn " "
  if fs.length != 3 then
    (s!"{caseStr} | STACK_UNPARSED", [], true)
  else
    match parseHex fs[1]!, (fs[2]!.splitOn ",").mapM bytesOf, catIdxOf "init_stack" with
    | some top, some argv, some stackCatIdx =>
      if top % page != 0 || argv.isEmpty then
        (s!"{caseStr} | BAD_STACK_ARGS", [stackCatIdx], true)
      else
        let (r, sp) := initStack top argv
        let m : Mem := [r]
        let wfOk := memWfBool m && r.base + r.data.length == top && sp == r.base + stackRoom
        let spMod16 := sp % 16
        let words := 1 + argv.length + 1 + 1 + 2
        let expectedStrAt := sp + words * 8
        let roomOk :=
          (Mem.wrN m (sp - 8) (le64 1)).isSome &&
          (Mem.wrN m (sp - stackRoom) (le64 1)).isSome &&
          (Mem.wrN m (sp - stackRoom - 8) (le64 1)).isNone
        match rd64 m sp,
              rd64 m (sp + 8 * (argv.length + 1)),
              rd64 m (sp + 8 * (argv.length + 2)),
              rd64 m (sp + 8 * (argv.length + 3)),
              rd64 m (sp + 8 * (argv.length + 4)),
              (List.range argv.length).mapM (fun i => rd64 m (sp + 8 * (i + 1))) with
        | some argcVal, some argvNull, some envpNull, some auxK, some auxV, some ptrs =>
          let ptr0 := ptrs.headD 0
          if !wfOk || spMod16 != 0 || argcVal != argv.length || ptr0 != expectedStrAt || !roomOk then
            (s!"{caseStr} | STACK_INV_FAIL", [stackCatIdx], true)
          else
            let cstrsOpt := ptrs.mapM fun p => do
              let cs ← Mem.cstr m p 4096
              pure (p - ptr0, cs)
            match cstrsOpt with
            | some relAndStrs =>
              if relAndStrs.map Prod.snd != argv then
                (s!"{caseStr} | STACK_ARGV_MISMATCH", [stackCatIdx], true)
              else
                let wfCh := if wfOk && spMod16 == 0 then "1" else "0"
                let roomCh := if roomOk then "1" else "0"
                let auxNull := Nat.lor auxK auxV
                let relParts := relAndStrs.map fun (rel, cs) => s!"{hex4 rel}:{bytesHex cs}"
                let relJoined := ",".intercalate relParts
                let res := s!"{wfCh} {hex2 spMod16} {hex16 argcVal} {hex16 argvNull} {hex16 envpNull} {hex16 auxNull} {roomCh} {relJoined}"
                (s!"{caseStr} | {res}", [stackCatIdx], false)
            | none =>
              (s!"{caseStr} | STACK_CSTR_FAIL", [stackCatIdx], true)
        | _, _, _, _, _, _ =>
          (s!"{caseStr} | STACK_RD64_FAIL", [stackCatIdx], true)
    | _, _, _ =>
      (s!"{caseStr} | STACK_PARSE_FAIL", [], true)

def evalLine (line : String) : String × List Nat × Bool :=
  let caseStr := (line.splitOn " | ").head!
  if caseStr.startsWith "SCEN " then evalScen caseStr
  else if caseStr.startsWith "STACK " then evalStack caseStr
  else (s!"{line} | UNKNOWN_LINE", [], true)

partial def loop (h : IO.FS.Stream) (o : IO.FS.Stream) (lines steps bad : Nat) (counts : Array Nat) :
    IO (Nat × Nat × Nat × Array Nat) := do
  let line ← h.getLine
  if line.isEmpty then return (lines, steps, bad, counts)
  let line := line.trimAsciiEnd.toString
  let (outLine, cats, failed) := evalLine line
  o.putStrLn outLine
  let counts' := cats.foldl (fun acc idx => acc.modify idx (· + 1)) counts
  loop h o (lines + 1) (steps + cats.length) (if failed then bad + 1 else bad) counts'

def main : IO UInt32 := do
  let initCounts := Array.replicate catNames.size 0
  let (lines, steps, bad, counts) ← loop (← IO.getStdin) (← IO.getStdout) 0 0 0 initCounts
  IO.eprintln s!"x86env_diff model: {lines} lines, {steps} steps, {bad} failed"
  let mut minCount := if counts.isEmpty then 0 else counts[0]!
  for i in [:catNames.size] do
    let c := counts[i]!
    if c < minCount then minCount := c
    IO.eprintln s!"  category {catNames[i]!}: {c}"
  IO.eprintln s!"min_category_count={minCount}"
  return (if bad == 0 then 0 else 1)
