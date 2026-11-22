/-
TMRBoundCheck: exact rational (`Rat`) verifier for the hardware fault-injection
campaign (`bootstrap/model/tmr_fault_inject.cpp`) of the canonical 38-byte
machine-code TMR voter (`formal/Bootstrap/TMRVoter.lean`).

Reads the summary lines emitted by `tmr_fault_inject` on stdin and verifies:
  1. The 38-byte machine-code voter image matches `voterCode` byte-for-byte.
  2. The real `0f 05` (`syscall` `exit(3)`) fork check passed.
  3. The exhaustive deterministic single-bit fault sweep across all 3 replicas
     (`rax`, `rbx`, `rcx`) and the 13 other GPRs has `SDC = 0`, `refusal = 0`,
     `stuck = 0`, and `corrected = trials`.
  4. Every Mode A (`r_total = 0`) and Mode B (`r_total > 0`) Monte Carlo sweep
     point satisfies the exact rational upper bound `3 · q_body² + 9 · r_total`
     (plus documented finite-sample binomial tolerance `tol`).
-/
import Bootstrap.PDist
import Bootstrap.TMR
import Bootstrap.TMRExposure
import Bootstrap.X86Env
import Bootstrap.X86Int
import Bootstrap.X86Dec

open X86Env X86Int X86Dec PDist TMR

def voterBase : Nat := 0x400000

/-- Canonical 38-byte machine-code voter, identical to `TMRVoter.voterCode`. -/
def voterCode : List Nat :=
  [ 0x48, 0x39, 0xd8              -- 0x00: cmp rax, rbx
  , 0x74, 0x0d                    -- 0x03: je  0x400012
  , 0x48, 0x39, 0xc8              -- 0x05: cmp rax, rcx
  , 0x74, 0x08                    -- 0x08: je  0x400012
  , 0x48, 0x39, 0xcb              -- 0x0a: cmp rbx, rcx
  , 0x75, 0x0b                    -- 0x0d: jne 0x40001a
  , 0x48, 0x89, 0xd8              -- 0x0f: mov rax, rbx
  , 0x48, 0x89, 0xc3              -- 0x12: mov rbx, rax
  , 0x48, 0x89, 0xc1              -- 0x15: mov rcx, rax
  , 0xeb, 0xfe                    -- 0x18: jmp 0x400018
  , 0xbf, 0x03, 0x00, 0x00, 0x00  -- 0x1a: mov edi, 3
  , 0xb8, 0x3c, 0x00, 0x00, 0x00  -- 0x1f: mov eax, 60
  , 0x0f, 0x05                    -- 0x24: syscall
  ]

theorem voterCode_length : voterCode.length = 38 := by decide +kernel

def voterIm : Nat → Option (Ins × Nat) :=
  X86Dec.imageOf voterCode voterBase

theorem voterIm_decode :
    voterIm (voterBase + 0x00) = some (.cmp 0 3, 3) ∧
    voterIm (voterBase + 0x03) = some (.jcc 4 (voterBase + 0x12), 2) ∧
    voterIm (voterBase + 0x05) = some (.cmp 0 1, 3) ∧
    voterIm (voterBase + 0x08) = some (.jcc 4 (voterBase + 0x12), 2) ∧
    voterIm (voterBase + 0x0a) = some (.cmp 3 1, 3) ∧
    voterIm (voterBase + 0x0d) = some (.jcc 5 (voterBase + 0x1a), 2) ∧
    voterIm (voterBase + 0x0f) = some (.mov 0 3, 3) ∧
    voterIm (voterBase + 0x12) = some (.mov 3 0, 3) ∧
    voterIm (voterBase + 0x15) = some (.mov 1 0, 3) ∧
    voterIm (voterBase + 0x18) = some (.jmp (voterBase + 0x18), 2) ∧
    voterIm (voterBase + 0x1a) = some (.movI 7 3, 5) ∧
    voterIm (voterBase + 0x1f) = some (.movI 0 60, 5) ∧
    voterIm (voterBase + 0x24) = some (.syscall, 2) := by
  decide +kernel

/-- The formal upper bound proved in `TMRVoter`: `3 · q_body² + 9 · r_total`. -/
def tmrBound (qBody rTotal : Rat) : Rat :=
  3 * (qBody * qBody) + 9 * rTotal

theorem tmrBound_modeA_100 : tmrBound (1 / 100) 0 = 3 / 10000 := by decide +kernel
theorem tmrBound_modeA_50  : tmrBound (1 / 50) 0  = 3 / 2500  := by decide +kernel
theorem tmrBound_modeA_20  : tmrBound (1 / 20) 0  = 3 / 400   := by decide +kernel
theorem tmrBound_modeA_10  : tmrBound (1 / 10) 0  = 3 / 100   := by decide +kernel

theorem tmrBound_modeB_50_500 : tmrBound (1 / 50) (1 / 500) = 12 / 625  := by decide +kernel
theorem tmrBound_modeB_20_200 : tmrBound (1 / 20) (1 / 200) = 21 / 400  := by decide +kernel
theorem tmrBound_modeB_10_100 : tmrBound (1 / 10) (1 / 100) = 3 / 25    := by decide +kernel

def hexDigit (n : Nat) : Char :=
  match n with
  | 0 => '0' | 1 => '1' | 2 => '2' | 3 => '3'
  | 4 => '4' | 5 => '5' | 6 => '6' | 7 => '7'
  | 8 => '8' | 9 => '9' | 10 => 'a' | 11 => 'b'
  | 12 => 'c' | 13 => 'd' | 14 => 'e' | _ => 'f'

def hex2 (v : Nat) : String :=
  String.ofList [hexDigit ((v / 16) % 16), hexDigit (v % 16)]

def expectedVoterHex : String :=
  String.join (voterCode.map hex2)

def parseNat (s : String) : Option Nat :=
  if s.isEmpty then none
  else s.foldl (fun acc c => do
    let a ← acc
    if '0' ≤ c && c ≤ '9' then some (a * 10 + (c.toNat - '0'.toNat)) else none) (some 0)

def parseRat (s : String) : Option Rat :=
  match s.splitOn "/" with
  | [nStr] => do
    let n ← parseNat nStr
    pure (n : Rat)
  | [nStr, dStr] => do
    let n ← parseNat nStr
    let d ← parseNat dStr
    if d == 0 then none else pure ((n : Rat) / (d : Rat))
  | _ => none

def getKV (fields : List String) (key : String) : Option String :=
  let pfx := key ++ "="
  fields.findSome? fun f =>
    if f.startsWith pfx then some ((f.drop pfx.length).toString) else none

def padLeft (s : String) (w : Nat) : String :=
  if s.length >= w then s
  else String.ofList (List.replicate (w - s.length) ' ') ++ s

def padRight (s : String) (w : Nat) : String :=
  if s.length >= w then s
  else s ++ String.ofList (List.replicate (w - s.length) ' ')

/-- Format a `Rat` as fixed-point `0.ddddd` using exact integer arithmetic. -/
def ratFixed5 (r : Rat) : String :=
  let sign := if r < 0 then "-" else ""
  let n : Nat := (r.num.natAbs * 100000) / r.den
  let whole := n / 100000
  let frac := n % 100000
  let fracStr := toString frac
  let zeros := String.ofList (List.replicate (5 - min 5 fracStr.length) '0')
  s!"{sign}{whole}.{zeros}{fracStr}"

structure CheckState where
  sawVoterCode : Bool := false
  sawSyscall : Bool := false
  sawSingleSweep : Bool := false
  modeACount : Nat := 0
  modeBCount : Nat := 0
  failures : List String := []

def checkSweepRow (modeLabel : String) (q r tol : Rat) (fs : List String) (st : CheckState) :
    IO CheckState := do
  match getKV fs "trials" >>= parseNat,
        getKV fs "clean" >>= parseNat,
        getKV fs "corrected" >>= parseNat,
        getKV fs "refusal" >>= parseNat,
        getKV fs "stuck" >>= parseNat,
        getKV fs "sdc" >>= parseNat,
        getKV fs "err" >>= parseNat with
  | some trials, some clean, some corr, some ref, some stuck, some sdc, some err =>
    if trials == 0 then
      return { st with failures := st.failures ++ [s!"{modeLabel}: trials=0"] }
    let sumOk := clean + corr + ref + stuck + sdc == trials && err == ref + stuck + sdc
    let empRate : Rat := (err : Rat) / (trials : Rat)
    let bound : Rat := tmrBound q r
    let boundOk : Bool := decide (empRate ≤ bound + tol)
    let sdcRate : Rat := (sdc : Rat) / (trials : Rat)
    let sdcOk : Bool := decide (sdcRate ≤ bound + tol)
    let ok := sumOk && boundOk && sdcOk
    let status := if ok then "PASS" else "FAIL"
    IO.println s!"  {padRight modeLabel 22} | {padLeft (toString trials) 6} | {padLeft (toString clean) 6} | {padLeft (toString corr) 6} | {padLeft (toString ref) 5} | {padLeft (toString stuck) 5} | {padLeft (toString sdc) 4} | {padLeft (ratFixed5 empRate) 8} | {padLeft (ratFixed5 bound) 8} | {padLeft (ratFixed5 (bound + tol)) 9} | {status}"
    if ok then
      return st
    else
      return { st with failures := st.failures ++ [s!"{modeLabel}: bound check failed (emp={empRate}, bound={bound}, tol={tol}, sdc={sdc})"] }
  | _, _, _, _, _, _, _ =>
    return { st with failures := st.failures ++ [s!"{modeLabel}: malformed counts"] }

partial def processLines (h : IO.FS.Stream) (st : CheckState) : IO CheckState := do
  let line ← h.getLine
  if line.isEmpty then return st
  let line := line.trimAsciiEnd.toString
  if line.isEmpty then
    processLines h st
  else
    let fs := line.splitOn " "
    match fs with
    | ["VOTER_CODE", hexStr] =>
      if hexStr == expectedVoterHex then
        IO.println s!"[OK] voterCode (38B): {hexStr}"
        processLines h { st with sawVoterCode := true }
      else
        IO.println s!"[FAIL] voterCode mismatch:\n  got:      {hexStr}\n  expected: {expectedVoterHex}"
        processLines h { st with sawVoterCode := true, failures := st.failures ++ ["voterCode 38-byte hex mismatch"] }
    | ["SYSCALL_EXIT3_CHECK", res] =>
      if res == "PASS" then
        IO.println "[OK] fork() child syscall 0f 05 exit(3) on 3-way disagreement: PASS"
        processLines h { st with sawSyscall := true }
      else
        IO.println s!"[FAIL] fork() child syscall exit(3) check: {res}"
        processLines h { st with sawSyscall := true, failures := st.failures ++ ["syscall exit(3) check failed"] }
    | "SINGLE_FAULT_SWEEP" :: rest =>
      match getKV rest "trials" >>= parseNat,
            getKV rest "clean" >>= parseNat,
            getKV rest "corrected" >>= parseNat,
            getKV rest "refusal" >>= parseNat,
            getKV rest "stuck" >>= parseNat,
            getKV rest "sdc" >>= parseNat with
      | some trials, some clean, some corr, some ref, some stuck, some sdc =>
        let ok := trials >= 192 && sdc == 0 && ref == 0 && stuck == 0 && clean == 0 && corr == trials
        let tag := if ok then "OK" else "FAIL"
        IO.println s!"[{tag}] Exhaustive single-fault sweep: trials={trials} corrected={corr} refusal={ref} stuck={stuck} SDC={sdc}"
        IO.println ""
        IO.println "  Mode / Rates           | Trials |  Clean |   Corr | Refus | Stuck |  SDC |  EmpRate | 3q^2+9r  | Bound+Tol | Status"
        IO.println "  -----------------------+--------+--------+--------+-------+-------+------+----------+----------+-----------+-------"
        let st' := { st with sawSingleSweep := true }
        if ok then
          processLines h st'
        else
          processLines h { st' with failures := st'.failures ++ [s!"single-fault sweep failed: corrected={corr}/{trials}, refusal={ref}, stuck={stuck}, SDC={sdc}"] }
      | _, _, _, _, _, _ =>
        processLines h { st with failures := st.failures ++ ["malformed SINGLE_FAULT_SWEEP line"] }
    | "MODE_A" :: rest =>
      match getKV rest "q" >>= parseRat, getKV rest "tol" >>= parseRat with
      | some q, some tol =>
        let qStr := (getKV rest "q").getD "?"
        let st' ← checkSweepRow s!"MODE_A q={qStr}" q 0 tol rest st
        processLines h { st' with modeACount := st'.modeACount + 1 }
      | _, _ =>
        processLines h { st with failures := st.failures ++ [s!"malformed MODE_A line: {line}"] }
    | "MODE_B" :: rest =>
      match getKV rest "q" >>= parseRat, getKV rest "r" >>= parseRat, getKV rest "tol" >>= parseRat with
      | some q, some r, some tol =>
        let qStr := (getKV rest "q").getD "?"
        let rStr := (getKV rest "r").getD "?"
        let st' ← checkSweepRow s!"MODE_B q={qStr} r={rStr}" q r tol rest st
        processLines h { st' with modeBCount := st'.modeBCount + 1 }
      | _, _, _ =>
        processLines h { st with failures := st.failures ++ [s!"malformed MODE_B line: {line}"] }
    | _ =>
      processLines h { st with failures := st.failures ++ [s!"unrecognized line: {line}"] }

def main : IO UInt32 := do
  let st ← processLines (← IO.getStdin) {}
  let mut errs := st.failures
  if !st.sawVoterCode then errs := errs ++ ["missing VOTER_CODE line"]
  if !st.sawSyscall then errs := errs ++ ["missing SYSCALL_EXIT3_CHECK line"]
  if !st.sawSingleSweep then errs := errs ++ ["missing SINGLE_FAULT_SWEEP line"]
  if st.modeACount != 4 then errs := errs ++ [s!"expected 4 MODE_A rows, got {st.modeACount}"]
  if st.modeBCount != 9 then errs := errs ++ [s!"expected 9 MODE_B rows, got {st.modeBCount}"]
  IO.println ""
  if errs.isEmpty then
    IO.println s!"TMR_FAULT_GATE_OK: single-fault SDC=0, all {st.modeACount} Mode A and {st.modeBCount} Mode B points satisfy 3*q^2 + 9*r_total"
    return 0
  else
    for e in errs do
      IO.eprintln s!"TMR_FAULT_GATE_FAIL: {e}"
    return 1
