/-
TMRBurstCheck: Lean model generator and exact rational (`Rat`) hardware gate
verifier for Pillar #4 (`formal/Bootstrap/TMRBurst.lean` and
`bootstrap/model/tmr_burst_straddle_cpu.cpp`):

  1. Page-Straddle `#PF` vs `#UD` vs Retirement Differential (`>= 50,000` cases
     + exhaustive `k = 1..15` boundary sweep at `page0End - k` right before an
     unmapped `PROT_NONE` guard page at `0x100000000`):
     - Compares `X86Dec.decode` on the truncated byte stream
       `bTrunc k bs i = if i < k then bs[i]? else none` against the real Xeon
       CPU (`SIGSEGV` fetch `#PF` with `si_addr == page0End`, `SIGILL` `#UD`, or
       `SIGTRAP` clean retirement matching `X86Int.step` across all 16 GPRs,
       `CF/SF/ZF/OF` flags, `rip` delta, and 64-byte memory window).
     - Includes negative control verification (`STRADDLE_NEG_CTRL`).

  2. 64B Adversarial Cache-Line Burst & Decision-50 Arena Injection on the
     canonical 415-byte `memExposureKernelCode` across 3 page-separated 4096B
     arenas (`0x100000`, `0x200000`, `0x300000`) surrounded by `PROT_NONE`
     guard pages:
     - Verifies the 415-byte `memExposureKernelCode` image byte-for-byte and
       checks `memExposureIm_decode` by `decide +kernel`.
     - Verifies `fork()` child `0f 05` `syscall` `exit(3)` on 3-way split and
       `exit(2)` on 64-bit overflow, plus `PROT_NONE` guard-page isolation and
       unmodified `RFLAGS.TF = 1` hardware single-stepping.
     - Verifies `SDC = 0` and `SDC_cert = 0` across exhaustive single 64-byte
       cache-line corruptions (`all-0x00`, `all-0xff`, all 512 bit-flips of the
       64B cache line in all 3 arenas, targeted certificate-underestimation
       bursts, and random 512-bit adversarial payloads across `val`,
       `last_step`, `step`, `acc`).
     - Verifies multi-burst Monte Carlo sweeps against the exact rational bound
       `3 · q_burst²` (`TMRBurst.shared_mem_three_burst_draws_err_le` /
       `TMRBurst.burst_tri_err`).
     - Verifies the packed-arena (`< 64B`) negative control (`PACKED_NEG_CTRL`),
       where packing two replicas 32 bytes apart inside one 64B cache line
       causes single-burst `SDC > 0` and `SDC_cert > 0`.
-/
import Bootstrap.PDist
import Bootstrap.TMR
import Bootstrap.TMRExposure
import Bootstrap.X86Env
import Bootstrap.X86Int
import Bootstrap.X86Dec
import Bootstrap.TMRVoter
import Bootstrap.TMRBurst

open X86Env X86Int X86Dec PDist TMR TMRBurst

def page0End : Nat := 0x100000000
def spInt : Nat := 0x11800
def winLo : Nat := 0x117e0
def winSize : Nat := 64
def poolSize : Nat := 32

/-! ### Canonical 415-byte Decision-50 `memExposureKernelCode` -/

def memExposureBase : Nat := 0x400000
def memExposurePark : Nat := memExposureBase + 0x185
def memExposureOverflow : Nat := memExposureBase + 0x187
def memExposureRefuse : Nat := memExposureBase + 0x193

def memExposureKernelCode : List Nat :=
  -- Stage 0: slot bounds check and arena/slot pointer setup (0x000..0x038)
  [ 0x48, 0x81, 0xff, 0xff, 0x00, 0x00, 0x00  -- 0x000: cmp rdi, 255
  , 0x0f, 0x83, 0x7a, 0x01, 0x00, 0x00        -- 0x007: jae 0x400187 (overflow/bad slot)
  , 0x41, 0xb8, 0x00, 0x00, 0x10, 0x00        -- 0x00d: mov r8d, 0x100000
  , 0x41, 0xb9, 0x00, 0x00, 0x20, 0x00        -- 0x013: mov r9d, 0x200000
  , 0x41, 0xba, 0x00, 0x00, 0x30, 0x00        -- 0x019: mov r10d, 0x300000
  , 0x49, 0x89, 0xfb                          -- 0x01f: mov r11, rdi
  , 0x49, 0xc1, 0xe3, 0x04                    -- 0x022: shl r11, 4
  , 0x4d, 0x89, 0xc6                          -- 0x026: mov r14, r8
  , 0x4d, 0x01, 0xde                          -- 0x029: add r14, r11
  , 0x4d, 0x89, 0xcf                          -- 0x02c: mov r15, r9
  , 0x4d, 0x01, 0xdf                          -- 0x02f: add r15, r11
  , 0x4c, 0x89, 0xd2                          -- 0x032: mov rdx, r10
  , 0x4c, 0x01, 0xda                          -- 0x035: add rdx, r11

  -- Stage 1: load, majority-vote, and repair val[w] (0x038..0x067)
  , 0x49, 0x8b, 0x46, 0x10                    -- 0x038: mov rax, [r14 + 16]
  , 0x49, 0x8b, 0x5f, 0x10                    -- 0x03c: mov rbx, [r15 + 16]
  , 0x48, 0x8b, 0x4a, 0x10                    -- 0x040: mov rcx, [rdx + 16]
  , 0x48, 0x39, 0xd8                          -- 0x044: cmp rax, rbx
  , 0x74, 0x12                                -- 0x047: je  0x40005b
  , 0x48, 0x39, 0xc8                          -- 0x049: cmp rax, rcx
  , 0x74, 0x0d                                -- 0x04c: je  0x40005b
  , 0x48, 0x39, 0xcb                          -- 0x04e: cmp rbx, rcx
  , 0x74, 0x05                                -- 0x051: je  0x400058
  , 0xe9, 0x3b, 0x01, 0x00, 0x00              -- 0x053: jmp 0x400193 (refuse)
  , 0x48, 0x89, 0xd8                          -- 0x058: mov rax, rbx
  , 0x49, 0x89, 0x46, 0x10                    -- 0x05b: mov [r14 + 16], rax
  , 0x49, 0x89, 0x47, 0x10                    -- 0x05f: mov [r15 + 16], rax
  , 0x48, 0x89, 0x42, 0x10                    -- 0x063: mov [rdx + 16], rax

  -- Stage 2: load, majority-vote, and repair last_step[w] into r11 (0x067..0x099)
  , 0x49, 0x8b, 0x46, 0x18                    -- 0x067: mov rax, [r14 + 24]
  , 0x49, 0x8b, 0x5f, 0x18                    -- 0x06b: mov rbx, [r15 + 24]
  , 0x48, 0x8b, 0x4a, 0x18                    -- 0x06f: mov rcx, [rdx + 24]
  , 0x48, 0x39, 0xd8                          -- 0x073: cmp rax, rbx
  , 0x74, 0x12                                -- 0x076: je  0x40008a
  , 0x48, 0x39, 0xc8                          -- 0x078: cmp rax, rcx
  , 0x74, 0x0d                                -- 0x07b: je  0x40008a
  , 0x48, 0x39, 0xcb                          -- 0x07d: cmp rbx, rcx
  , 0x74, 0x05                                -- 0x080: je  0x400087
  , 0xe9, 0x0c, 0x01, 0x00, 0x00              -- 0x082: jmp 0x400193 (refuse)
  , 0x48, 0x89, 0xd8                          -- 0x087: mov rax, rbx
  , 0x49, 0x89, 0x46, 0x18                    -- 0x08a: mov [r14 + 24], rax
  , 0x49, 0x89, 0x47, 0x18                    -- 0x08e: mov [r15 + 24], rax
  , 0x48, 0x89, 0x42, 0x18                    -- 0x092: mov [rdx + 24], rax
  , 0x49, 0x89, 0xc3                          -- 0x096: mov r11, rax

  -- Stage 3: load, majority-vote, and repair step into r12 (0x099..0x0c5)
  , 0x49, 0x8b, 0x00                          -- 0x099: mov rax, [r8]
  , 0x49, 0x8b, 0x19                          -- 0x09c: mov rbx, [r9]
  , 0x49, 0x8b, 0x0a                          -- 0x09f: mov rcx, [r10]
  , 0x48, 0x39, 0xd8                          -- 0x0a2: cmp rax, rbx
  , 0x74, 0x12                                -- 0x0a5: je  0x4000b9
  , 0x48, 0x39, 0xc8                          -- 0x0a7: cmp rax, rcx
  , 0x74, 0x0d                                -- 0x0aa: je  0x4000b9
  , 0x48, 0x39, 0xcb                          -- 0x0ac: cmp rbx, rcx
  , 0x74, 0x05                                -- 0x0af: je  0x4000b6
  , 0xe9, 0xdd, 0x00, 0x00, 0x00              -- 0x0b1: jmp 0x400193 (refuse)
  , 0x48, 0x89, 0xd8                          -- 0x0b6: mov rax, rbx
  , 0x49, 0x89, 0x00                          -- 0x0b9: mov [r8], rax
  , 0x49, 0x89, 0x01                          -- 0x0bc: mov [r9], rax
  , 0x49, 0x89, 0x02                          -- 0x0bf: mov [r10], rax
  , 0x49, 0x89, 0xc4                          -- 0x0c2: mov r12, rax

  -- Stage 4: load, majority-vote, and repair acc into r13 (0x0c5..0x0f7)
  , 0x49, 0x8b, 0x40, 0x08                    -- 0x0c5: mov rax, [r8 + 8]
  , 0x49, 0x8b, 0x59, 0x08                    -- 0x0c9: mov rbx, [r9 + 8]
  , 0x49, 0x8b, 0x4a, 0x08                    -- 0x0cd: mov rcx, [r10 + 8]
  , 0x48, 0x39, 0xd8                          -- 0x0d1: cmp rax, rbx
  , 0x74, 0x12                                -- 0x0d4: je  0x4000e8
  , 0x48, 0x39, 0xc8                          -- 0x0d6: cmp rax, rcx
  , 0x74, 0x0d                                -- 0x0d9: je  0x4000e8
  , 0x48, 0x39, 0xcb                          -- 0x0db: cmp rbx, rcx
  , 0x74, 0x05                                -- 0x0de: je  0x4000e5
  , 0xe9, 0xae, 0x00, 0x00, 0x00              -- 0x0e0: jmp 0x400193 (refuse)
  , 0x48, 0x89, 0xd8                          -- 0x0e5: mov rax, rbx
  , 0x49, 0x89, 0x40, 0x08                    -- 0x0e8: mov [r8 + 8], rax
  , 0x49, 0x89, 0x41, 0x08                    -- 0x0ec: mov [r9 + 8], rax
  , 0x49, 0x89, 0x42, 0x08                    -- 0x0f0: mov [r10 + 8], rax
  , 0x49, 0x89, 0xc5                          -- 0x0f4: mov r13, rax

  -- Stage 5: delta := step - last[w] + 1, step_new := step + 1, overflow checks (0x0f7..0x117)
  , 0x4c, 0x89, 0xe0                          -- 0x0f7: mov rax, r12
  , 0x4c, 0x29, 0xd8                          -- 0x0fa: sub rax, r11
  , 0x0f, 0x82, 0x84, 0x00, 0x00, 0x00        -- 0x0fd: jc  0x400187
  , 0x48, 0x83, 0xc0, 0x01                    -- 0x103: add rax, 1
  , 0x0f, 0x82, 0x7a, 0x00, 0x00, 0x00        -- 0x107: jc  0x400187
  , 0x49, 0x83, 0xc4, 0x01                    -- 0x10d: add r12, 1
  , 0x0f, 0x82, 0x70, 0x00, 0x00, 0x00        -- 0x111: jc  0x400187

  -- Stage 6: shift-and-add loop computing rbx := delta^2 (0x117..0x143)
  , 0x48, 0x31, 0xdb                          -- 0x117: xor rbx, rbx
  , 0x48, 0x89, 0xc1                          -- 0x11a: mov rcx, rax
  , 0x48, 0x89, 0xc5                          -- 0x11d: mov rbp, rax   [sq_loop]
  , 0x48, 0x83, 0xe5, 0x01                    -- 0x120: and rbp, 1
  , 0x74, 0x09                                -- 0x124: jz  0x40012f
  , 0x48, 0x01, 0xcb                          -- 0x126: add rbx, rcx
  , 0x0f, 0x82, 0x58, 0x00, 0x00, 0x00        -- 0x129: jc  0x400187
  , 0x48, 0xc1, 0xe8, 0x01                    -- 0x12f: shr rax, 1     [sq_skip_add]
  , 0x48, 0x85, 0xc0                          -- 0x133: test rax, rax
  , 0x74, 0x0b                                -- 0x136: jz  0x400143
  , 0x48, 0x01, 0xc9                          -- 0x138: add rcx, rcx
  , 0x0f, 0x82, 0x46, 0x00, 0x00, 0x00        -- 0x13b: jc  0x400187
  , 0xeb, 0xda                                -- 0x141: jmp 0x40011d

  -- Stage 7: acc += delta^2, commit to all 3 arenas, normalize scratch, park (0x143..0x187)
  , 0x49, 0x01, 0xdd                          -- 0x143: add r13, rbx   [sq_done]
  , 0x0f, 0x82, 0x3b, 0x00, 0x00, 0x00        -- 0x146: jc  0x400187
  , 0x4d, 0x89, 0x20                          -- 0x14c: mov [r8], r12
  , 0x4d, 0x89, 0x68, 0x08                    -- 0x14f: mov [r8 + 8], r13
  , 0x49, 0x89, 0x76, 0x10                    -- 0x153: mov [r14 + 16], rsi
  , 0x4d, 0x89, 0x66, 0x18                    -- 0x157: mov [r14 + 24], r12
  , 0x4d, 0x89, 0x21                          -- 0x15b: mov [r9], r12
  , 0x4d, 0x89, 0x69, 0x08                    -- 0x15e: mov [r9 + 8], r13
  , 0x49, 0x89, 0x77, 0x10                    -- 0x162: mov [r15 + 16], rsi
  , 0x4d, 0x89, 0x67, 0x18                    -- 0x166: mov [r15 + 24], r12
  , 0x4d, 0x89, 0x22                          -- 0x16a: mov [r10], r12
  , 0x4d, 0x89, 0x6a, 0x08                    -- 0x16d: mov [r10 + 8], r13
  , 0x48, 0x89, 0x72, 0x10                    -- 0x171: mov [rdx + 16], rsi
  , 0x4c, 0x89, 0x62, 0x18                    -- 0x175: mov [rdx + 24], r12
  , 0x4c, 0x89, 0xe8                          -- 0x179: mov rax, r13
  , 0x48, 0x31, 0xdb                          -- 0x17c: xor rbx, rbx
  , 0x48, 0x31, 0xc9                          -- 0x17f: xor rcx, rcx
  , 0x48, 0x31, 0xed                          -- 0x182: xor rbp, rbp
  , 0xeb, 0xfe                                -- 0x185: jmp 0x400185   [park]

  -- Overflow / invalid slot exit(2) (0x187..0x193)
  , 0xbf, 0x02, 0x00, 0x00, 0x00              -- 0x187: mov edi, 2
  , 0xb8, 0x3c, 0x00, 0x00, 0x00              -- 0x18c: mov eax, 60
  , 0x0f, 0x05                                -- 0x191: syscall

  -- 3-way split exit(3) (0x193..0x19f)
  , 0xbf, 0x03, 0x00, 0x00, 0x00              -- 0x193: mov edi, 3
  , 0xb8, 0x3c, 0x00, 0x00, 0x00              -- 0x198: mov eax, 60
  , 0x0f, 0x05                                -- 0x19d: syscall
  ]

theorem memExposureKernelCode_length : memExposureKernelCode.length = 415 := by
  decide +kernel

def memExposureIm : Nat → Option (Ins × Nat) :=
  X86Dec.imageOf memExposureKernelCode memExposureBase

/-- Compile-time verification that `memExposureKernelCode` decodes at entry,
all 4 voter stages, loop entry, park (`0x400185`), overflow syscall (`0x400191`),
and refuse syscall (`0x40019d`). -/
theorem memExposureIm_check :
    memExposureIm (memExposureBase + 0x000) = some (.cmpI 7 255, 7) ∧
    memExposureIm (memExposureBase + 0x038) = some (.ld 0 14 16, 4) ∧
    memExposureIm (memExposureBase + 0x067) = some (.ld 0 14 24, 4) ∧
    memExposureIm (memExposureBase + 0x099) = some (.ld 0 8 0, 3) ∧
    memExposureIm (memExposureBase + 0x0c5) = some (.ld 0 8 8, 4) ∧
    memExposureIm (memExposureBase + 0x11d) = some (.mov 5 0, 3) ∧
    memExposureIm memExposurePark = some (.jmp memExposurePark, 2) ∧
    memExposureIm (memExposureOverflow + 10) = some (.syscall, 2) ∧
    memExposureIm (memExposureRefuse + 10) = some (.syscall, 2) := by
  decide +kernel

/-- Exact rational bound `3 · q_burst²` for 3-arena cache-line burst faults. -/
def tmrBurstBound (qBurst : Rat) : Rat :=
  3 * (qBurst * qBurst)

theorem tmrBurstBound_100 : tmrBurstBound (1 / 100) = 3 / 10000 := by decide +kernel
theorem tmrBurstBound_50  : tmrBurstBound (1 / 50)  = 3 / 2500  := by decide +kernel
theorem tmrBurstBound_20  : tmrBurstBound (1 / 20)  = 3 / 400   := by decide +kernel
theorem tmrBurstBound_10  : tmrBurstBound (1 / 10)  = 3 / 100   := by decide +kernel

/-! ### Formatting and Parsing Helpers -/

def hexDigit (n : Nat) : Char :=
  match n with
  | 0 => '0' | 1 => '1' | 2 => '2' | 3 => '3'
  | 4 => '4' | 5 => '5' | 6 => '6' | 7 => '7'
  | 8 => '8' | 9 => '9' | 10 => 'a' | 11 => 'b'
  | 12 => 'c' | 13 => 'd' | 14 => 'e' | _ => 'f'

def hex2Tbl : Array String :=
  Array.ofFn fun (i : Fin 256) =>
    String.ofList [hexDigit (i.1 / 16), hexDigit (i.1 % 16)]

def hex2 (v : Nat) : String :=
  hex2Tbl[v % 256]!

def hex16 (v : Nat) : String :=
  let u := UInt64.ofNat v
  let b7 := hex2Tbl[((u >>> 56) &&& 0xff).toNat]!
  let b6 := hex2Tbl[((u >>> 48) &&& 0xff).toNat]!
  let b5 := hex2Tbl[((u >>> 40) &&& 0xff).toNat]!
  let b4 := hex2Tbl[((u >>> 32) &&& 0xff).toNat]!
  let b3 := hex2Tbl[((u >>> 24) &&& 0xff).toNat]!
  let b2 := hex2Tbl[((u >>> 16) &&& 0xff).toNat]!
  let b1 := hex2Tbl[((u >>> 8) &&& 0xff).toNat]!
  let b0 := hex2Tbl[(u &&& 0xff).toNat]!
  b7 ++ b6 ++ b5 ++ b4 ++ b3 ++ b2 ++ b1 ++ b0

def expectedKernelHex : String :=
  String.join (memExposureKernelCode.map hex2)

def flagCh : Option Bool → Char
  | none => '-'
  | some true => '1'
  | some false => '0'

def xsStep (x : UInt64) : UInt64 :=
  let x1 := x ^^^ (x <<< 13)
  let x2 := x1 ^^^ (x1 >>> 7)
  x2 ^^^ (x2 <<< 17)

structure BaseState where
  regs : Array Nat
  cf : Bool
  sf : Bool
  zf : Bool
  of : Bool
  mem : List Nat
deriving Inhabited

/-- Construct deterministic base state `idx ∈ 0..31` from `seed` identically to C++. -/
def mkBaseState (seed : UInt64) (idx : Nat) : BaseState := Id.run do
  let mut st := xsStep (0x9e3779b97f4a7c15 ^^^ seed ^^^ UInt64.ofNat (idx * 0x10001 + 1))
  let mut regs : Array Nat := Array.mkEmpty 16
  for r in [:16] do
    st := xsStep st
    let v0 := st
    let m1 : UInt64 := 0xffffffffffff00ff
    let m2 : UInt64 := 0xffffffffffffff00
    let v : UInt64 :=
      if r == 4 then UInt64.ofNat spInt
      else if r < 4 then
        (v0 &&& m1) ||| (UInt64.ofNat (0xa1 + r * 0x11) <<< 8)
      else if r < 8 then
        (v0 &&& m2) ||| UInt64.ofNat (0x10 + r)
      else v0
    regs := regs.push v.toNat
  st := xsStep st
  let cf := (st &&& 1) != 0
  let sf := ((st >>> 1) &&& 1) != 0
  let zf := ((st >>> 2) &&& 1) != 0
  let of := ((st >>> 3) &&& 1) != 0
  let mut memRev : List Nat := []
  for _ in [:winSize] do
    st := xsStep st
    memRev := (st &&& 0xff).toNat :: memRev
  return { regs := regs, cf := cf, sf := sf, zf := zf, of := of, mem := memRev.reverse }

def setLe64 (m : List Nat) (off v : Nat) : List Nat :=
  let bs := bytes8 v
  (List.range 8).foldl (fun acc i => acc.set (off + i) (bs[i]?.getD 0)) m

def validCcs : Array Nat := #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15]

/-- Emit a valid memory operand `[base + disp]`. -/
def emitValidMem (st0 : UInt64) (regField base : Nat) : UInt64 × List Nat := Id.run do
  let mut st := xsStep st0
  let rMd := (st % 10).toNat
  let mut md := if rMd < 6 then 2 else if rMd < 8 then 1 else 0
  if md == 0 && (base % 8) == 5 then md := 2
  st := xsStep st
  let useSib := (base % 8) == 4 || (st % 3 == 0)
  let mByte := md * 64 + (regField % 8) * 8 + (if useSib then 4 else base % 8)
  let mut out : List Nat := [mByte]
  if useSib then
    st := xsStep st
    let scale := (st % 4).toNat
    out := out ++ [scale * 64 + 4 * 8 + (base % 8)]
  if md == 1 then
    st := xsStep st
    out := out ++ [(st &&& 0xff).toNat]
  else if md == 2 then
    for _ in [:4] do
      st := xsStep st
      out := out ++ [(st &&& 0xff).toNat]
  return (st, out)

/-- Generate one valid stage-1 non-syscall instruction of form `formIdx ∈ 0..44`. -/
def genValidIns (st0 : UInt64) (formIdx : Nat) : UInt64 × List Nat := Id.run do
  let mut st := xsStep st0
  let r := (st % 16).toNat
  st := xsStep st
  let b := (st % 16).toNat
  let mkRex (w : Bool) (rr bb : Nat) (force : Bool) : UInt64 × List Nat := Id.run do
    let x := 0x40 + (if w then 8 else 0) + (if rr >= 8 then 4 else 0) + (if bb >= 8 then 1 else 0)
    let st' := xsStep st
    if x != 0x40 || force || (st' % 3 == 0) then return (st', [x]) else return (st', [])
  let mkImm (n : Nat) (sIn : UInt64) : UInt64 × List Nat := Id.run do
    let mut s := sIn
    let mut acc : List Nat := []
    for _ in [:n] do
      s := xsStep s
      acc := acc ++ [(s &&& 0xff).toNat]
    return (s, acc)
  match formIdx % 45 with
  | 0 =>
    let (s1, rx) := mkRex false 0 b false
    let (s2, im) := mkImm 4 s1
    return (s2, rx ++ [0xb8 + (b % 8)] ++ im)
  | 1 =>
    let (s1, rx) := mkRex true 0 b true
    let (s2, im) := mkImm 8 s1
    return (s2, rx ++ [0xb8 + (b % 8)] ++ im)
  | 2 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x01, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 3 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x29, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 4 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x21, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 5 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x09, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 6 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x31, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 7 =>
    let (s1, rx) := mkRex false r r false
    return (s1, rx ++ [0x31, 0xc0 + (r % 8) * 8 + (r % 8)])
  | 8 =>
    let (s1, rx) := mkRex false r b false
    return (s1, rx ++ [0x89, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 9 =>
    let (s1, rx) := mkRex false r b false
    return (s1, rx ++ [0x8b, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 10 =>
    let (s1, rx) := mkRex false r b false
    let (s2, mem) := emitValidMem s1 r b
    return (s2, rx ++ [0x8b] ++ mem)
  | 11 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x89, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 12 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x8b, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 13 =>
    let (s1, rx) := mkRex true r b true
    let (s2, mem) := emitValidMem s1 r b
    return (s2, rx ++ [0x89] ++ mem)
  | 14 =>
    let (s1, rx) := mkRex true r b true
    let (s2, mem) := emitValidMem s1 r b
    return (s2, rx ++ [0x8b] ++ mem)
  | 15 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x85, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 16 =>
    let (s1, im) := mkImm 1 st
    return (s1, [0x6a] ++ im)
  | 17 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x39, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 18 =>
    let (s1, rx) := mkRex true r b true
    return (s1, rx ++ [0x3b, 0xc0 + (r % 8) * 8 + (b % 8)])
  | 19 | 20 | 21 | 22 | 23 | 24 =>
    let extTbl : Array Nat := #[0, 1, 4, 5, 6, 7]
    let ext := extTbl[(formIdx % 45) - 19]!
    let (s1, rx) := mkRex true 0 b true
    let (s2, im) := mkImm 1 s1
    return (s2, rx ++ [0x83, 0xc0 + ext * 8 + (b % 8)] ++ im)
  | 25 | 26 | 27 | 28 | 29 | 30 =>
    let extTbl : Array Nat := #[0, 1, 4, 5, 6, 7]
    let ext := extTbl[(formIdx % 45) - 25]!
    let (s1, rx) := mkRex true 0 b true
    let (s2, im) := mkImm 4 s1
    return (s2, rx ++ [0x81, 0xc0 + ext * 8 + (b % 8)] ++ im)
  | 31 =>
    let (s1, rx) := mkRex true 0 b true
    let (s2, im) := mkImm 1 s1
    return (s2, rx ++ [0xc1, 0xe0 + (b % 8)] ++ im)
  | 32 =>
    let (s1, rx) := mkRex true 0 b true
    let (s2, im) := mkImm 1 s1
    return (s2, rx ++ [0xc1, 0xe8 + (b % 8)] ++ im)
  | 33 =>
    let (s1, rx) := mkRex true r b true
    let (s2, mem) := emitValidMem s1 r b
    return (s2, rx ++ [0x8d] ++ mem)
  | 34 =>
    let (s1, rx) := mkRex (r % 2 == 0) r b (r >= 4)
    let (s2, mem) := emitValidMem s1 r b
    return (s2, rx ++ [0x88] ++ mem)
  | 35 =>
    let (s1, rx) := mkRex (r % 2 == 0) r b false
    let (s2, mem) := emitValidMem s1 r b
    return (s2, rx ++ [0x0f, 0xb6] ++ mem)
  | 36 =>
    let cc := validCcs[(r % 14)]!
    let (s1, im) := mkImm 4 st
    return (s1, [0x0f, 0x80 + cc] ++ im)
  | 37 =>
    let cc := validCcs[(r % 14)]!
    let (s1, im) := mkImm 1 st
    return (s1, [0x70 + cc] ++ im)
  | 38 =>
    let (s1, im) := mkImm 1 st
    return (s1, [0xeb] ++ im)
  | 39 =>
    let (s1, im) := mkImm 4 st
    return (s1, [0xe9] ++ im)
  | 40 =>
    let (s1, im) := mkImm 4 st
    return (s1, [0xe8] ++ im)
  | 41 =>
    return (st, [0xc3])
  | 42 =>
    let (s1, rx) := mkRex false 0 b false
    return (s1, rx ++ [0x50 + (b % 8)])
  | 43 =>
    let (s1, rx) := mkRex false 0 b false
    return (s1, rx ++ [0x58 + (b % 8)])
  | _ =>
    let idx := if r == 4 then 5 else r
    let mut md := if (b % 3) == 0 then 0 else if (b % 3) == 1 then 1 else 2
    if md == 0 && (b % 8) == 5 then md := 2
    let rx := (if idx >= 8 then 2 else 0) + (if b >= 8 then 1 else 0)
    let pfx := if rx != 0 || (idx % 2 == 0) then [0x40 + rx] else []
    let mByte := md * 64 + 4 * 8 + 4
    let sByte := 3 * 64 + (idx % 8) * 8 + (b % 8)
    let (s1, dispBs) := if md == 0 then (st, []) else if md == 1 then mkImm 1 st else mkImm 4 st
    return (s1, pfx ++ [0xff, mByte, sByte] ++ dispBs)

/-- Pad a byte sequence to 15 bytes using `0x06` (`#UD` in 64-bit mode). -/
def padWith06 (bs : List Nat) : Array Nat := Id.run do
  let mut arr := bs.toArray
  while arr.size < 15 do
    arr := arr.push 0x06
  return arr

/-- Truncated byte stream at `insAddr = page0End - k`: only bytes `i < k` lie
inside the executable page; bytes `i >= k` lie in the `PROT_NONE` guard page
and return `none`. -/
def bTrunc (k : Nat) (bs : Array Nat) (i : Nat) : Option Nat :=
  if i < k then bs[i]? else none

structure StraddleGenStats where
  totalCases : Nat := 0
  okCases : Nat := 0
  pfCases : Nat := 0
  udCases : Nat := 0
  sweeps15 : Nat := 0
  decodeInvFailures : Nat := 0

/-- Emit one `k = 1..15` boundary sweep for a valid `X86Dec` instruction `origBs`
padded with `0x06` to 15 bytes:
  - for `k < origBs.length`, `X86Dec.decode (bTrunc k bs15) (page0End - k)` MUST
    be `none`, and the CPU must raise fetch `SIGSEGV` (`#PF`, `kind = 1`) at
    `si_addr == page0End`;
  - for `k >= origBs.length`, `X86Dec.decode (bTrunc k bs15) (page0End - k)` MUST
    be `some (ins, origBs.length)`, and the CPU must retire cleanly (`SIGTRAP`,
    `kind = 0`) matching `X86Int.step`. -/
def emitValidSweep15 (pool : Array BaseState) (casesOut : IO.FS.Stream)
    (origBs : List Nat) (st0 : StraddleGenStats) : IO StraddleGenStats := do
  let origLen := origBs.length
  let bs15 := padWith06 origBs
  let mut st := { st0 with sweeps15 := st0.sweeps15 + 1 }
  for k in [1:16] do
    let insAddr := page0End - k
    let candK := (bs15.extract 0 k).toList
    let candHex := String.join (candK.map hex2)
    let decTrunc := X86Dec.decode (bTrunc k bs15) insAddr
    let decFull := X86Dec.decode (fun i => bs15[i]?) insAddr
    if k < origLen then
      -- Instruction straddles page0End into PROT_NONE guard page -> fetch #PF
      if decTrunc.isSome || decFull.isNone then
        st := { st with decodeInvFailures := st.decodeInvFailures + 1 }
      let poolIdx := st.totalCases % poolSize
      casesOut.putStrLn s!"{k} {candHex} 1 {origLen} {poolIdx} 16 0000000000000000 64 0000000000000000 0 | 0000000000000000 16 0000000000000000 16 0000000000000000 ---- 64 0 0000000000000000"
      st := { st with totalCases := st.totalCases + 1, pfCases := st.pfCases + 1 }
    else
      -- k >= origLen: instruction fits inside [page0End - k, page0End) -> retires cleanly
      match decTrunc with
      | none =>
        st := { st with decodeInvFailures := st.decodeInvFailures + 1 }
      | some (ins, len) =>
        if len != origLen || len > k then
          st := { st with decodeInvFailures := st.decodeInvFailures + 1 }
        let caseNum := st.totalCases
        let poolIdx := caseNum % poolSize
        let base := pool[poolIdx]!
        let (fixRegIdx, fixRegVal, fixMemOff, fixMemVal, isBranch) : Nat × Nat × Nat × Nat × Nat :=
          match ins with
          | .ld _ b disp | .st b disp _ =>
            let off := (caseNum * 7) % 57
            (b, (winLo + off + W - (disp % W)) % W, 64, 0, 0)
          | .ld32 _ b disp =>
            let off := (caseNum * 7) % 61
            (b, (winLo + off + W - (disp % W)) % W, 64, 0, 0)
          | .ld8 _ b disp | .st8 b disp _ =>
            let off := (caseNum * 7) % 64
            (b, (winLo + off + W - (disp % W)) % W, 64, 0, 0)
          | .jtab b i disp =>
            let off := (caseNum % 7) * 8
            let ea := winLo + off
            let regIVal := base.regs[i]!
            let rVal :=
              if b == i then
                ((UInt64.ofNat ea - UInt64.ofNat (disp % W)) * 0x8e38e38e38e38e39).toNat
              else
                (UInt64.ofNat ea - 8 * UInt64.ofNat regIVal - UInt64.ofNat (disp % W)).toNat
            let tgt := 0x80000000 + ((caseNum * 0x12345 + 0x1000) % 0x70000000)
            (b, rVal, off, tgt, 1)
          | .ret =>
            let tgt := 0x80000000 + ((caseNum * 0x12345 + 0x1000) % 0x70000000)
            (16, 0, 32, tgt, 1)
          | .jmp _ | .call _ | .jcc _ _ =>
            (16, 0, 64, 0, 1)
          | _ =>
            (16, 0, 64, 0, 0)
        let regsIn : Array Nat :=
          if fixRegIdx < 16 then base.regs.set! fixRegIdx fixRegVal else base.regs
        let memIn : List Nat :=
          if fixMemOff <= 56 then setLe64 base.mem fixMemOff fixMemVal else base.mem
        let flags : Flags := ⟨some base.cf, some base.zf, some base.sf, some base.of⟩
        let s : St := {
          regs := regsIn.toList.map some,
          rip := insAddr,
          fl := flags,
          mem := [⟨winLo, memIn, true⟩],
          k := ⟨[], [], [], [], []⟩
        }
        match step (imageOf candK insAddr) s with
        | .ok t =>
          let delta := (t.rip + W - insAddr) % W
          let mut mod1Idx : Nat := 16
          let mut mod1Val : Nat := 0
          let mut mod2Idx : Nat := 16
          let mut mod2Val : Nat := 0
          let tRegs := t.regs.toArray
          for r in [:16] do
            let vAfter := (tRegs[r]?.getD none).getD 0
            let vBefore := regsIn[r]!
            if vAfter != vBefore then
              if mod1Idx == 16 then
                mod1Idx := r
                mod1Val := vAfter
              else
                mod2Idx := r
                mod2Val := vAfter
          let (modMemOff, modMemLen, modMemVal) : Nat × Nat × Nat :=
            match ins with
            | .st b disp r =>
              let ea := (regsIn[b]! + disp) % W
              (ea - winLo, 8, regsIn[r]!)
            | .st8 b disp r =>
              let ea := (regsIn[b]! + disp) % W
              (ea - winLo, 1, regsIn[r]! % 256)
            | .pushI imm =>
              (24, 8, imm % W)
            | .push r =>
              (24, 8, regsIn[r]!)
            | .call _ =>
              (24, 8, insAddr + len)
            | _ =>
              (64, 0, 0)
          let flStr := String.ofList [flagCh t.fl.cf, flagCh t.fl.sf, flagCh t.fl.zf, flagCh t.fl.of]
          casesOut.putStrLn s!"{k} {candHex} 0 {len} {poolIdx} {fixRegIdx} {hex16 fixRegVal} {fixMemOff} {hex16 fixMemVal} {isBranch} | {hex16 delta} {mod1Idx} {hex16 mod1Val} {mod2Idx} {hex16 mod2Val} {flStr} {modMemOff} {modMemLen} {hex16 modMemVal}"
          st := { st with totalCases := st.totalCases + 1, okCases := st.okCases + 1 }
        | _ =>
          st := { st with decodeInvFailures := st.decodeInvFailures + 1 }
  return st

/-- Emit one `k = 1..15` boundary sweep for a `#UD` sequence `udBs` of length
`L_ud = udBs.length` (where bytes `0 .. L_ud - 2` are valid prefixes `0x40..0x4f`
or `0x0f`, and byte `L_ud - 1` completes an unconditional 64-bit `#UD` opcode):
  - for `k < L_ud`, only the prefix bytes lie in `[page0End - k, page0End)` and
    the `#UD` opcode byte is at or past `page0End` (`PROT_NONE`), so the CPU must
    raise fetch `SIGSEGV` (`#PF`, `kind = 1`) at `si_addr == page0End`;
  - for `k >= L_ud`, the entire `#UD` opcode lies inside `[page0End - k, page0End)`,
    so the CPU must raise `SIGILL` (`#UD`, `kind = 2`). -/
def emitUdSweep15 (casesOut : IO.FS.Stream) (udBs : List Nat)
    (st0 : StraddleGenStats) : IO StraddleGenStats := do
  let udLen := udBs.length
  let bs15 := padWith06 udBs
  let mut st := { st0 with sweeps15 := st0.sweeps15 + 1 }
  for k in [1:16] do
    let insAddr := page0End - k
    let candK := (bs15.extract 0 k).toList
    let candHex := String.join (candK.map hex2)
    let decTrunc := X86Dec.decode (bTrunc k bs15) insAddr
    if decTrunc.isSome then
      st := { st with decodeInvFailures := st.decodeInvFailures + 1 }
    let poolIdx := st.totalCases % poolSize
    if k < udLen then
      casesOut.putStrLn s!"{k} {candHex} 1 {udLen} {poolIdx} 16 0000000000000000 64 0000000000000000 0 | 0000000000000000 16 0000000000000000 16 0000000000000000 ---- 64 0 0000000000000000"
      st := { st with totalCases := st.totalCases + 1, pfCases := st.pfCases + 1 }
    else
      casesOut.putStrLn s!"{k} {candHex} 2 {udLen} {poolIdx} 16 0000000000000000 64 0000000000000000 0 | 0000000000000000 16 0000000000000000 16 0000000000000000 ---- 64 0 0000000000000000"
      st := { st with totalCases := st.totalCases + 1, udCases := st.udCases + 1 }
  return st

def ud1Opcodes : List Nat :=
  [0x06, 0x07, 0x0e, 0x16, 0x17, 0x1e, 0x1f, 0x27, 0x2f, 0x37, 0x3f, 0x60, 0x61, 0xce, 0xd6]

def runGen (seedGroups : Nat) (seed : UInt64) (casesPath statsPath : String) : IO UInt32 := do
  let pool : Array BaseState := Array.ofFn fun (i : Fin poolSize) => mkBaseState seed i.1
  let casesFile ← IO.FS.Handle.mk casesPath IO.FS.Mode.write
  let casesOut := IO.FS.Stream.ofHandle casesFile
  let mut rng := xsStep (0x9e3779b97f4a7c15 ^^^ seed)
  let mut st : StraddleGenStats := {}

  -- Part 1: `seedGroups` (default 3,400) valid stage-1 `X86Dec` instructions
  -- swept across all `k = 1..15` (3,400 * 15 = 51,000 cases).
  for i in [:seedGroups] do
    let (r1, origBs) := genValidIns rng i
    rng := r1
    st ← emitValidSweep15 pool casesOut origBs st

  -- Part 2: Exhaustive `k = 1..15` boundary sweep for 1-byte, 2-byte (`REX + #UD`
  -- and `0x0f 0x0b` `ud2`), and 3-byte (`REX + 0x0f 0x0b`) `#UD` sequences
  -- (272 sequences * 15 = 4,080 cases).
  for op in ud1Opcodes do
    st ← emitUdSweep15 casesOut [op] st
  for rex in [0x40:0x50] do
    for op in ud1Opcodes do
      st ← emitUdSweep15 casesOut [rex, op] st
  st ← emitUdSweep15 casesOut [0x0f, 0x0b] st
  for rex in [0x40:0x50] do
    st ← emitUdSweep15 casesOut [rex, 0x0f, 0x0b] st

  casesOut.flush
  let statsContent :=
    s!"straddle_total_cases={st.totalCases}\n" ++
    s!"straddle_ok_cases={st.okCases}\n" ++
    s!"straddle_pf_cases={st.pfCases}\n" ++
    s!"straddle_ud_cases={st.udCases}\n" ++
    s!"straddle_k1_to_15_sweeps={st.sweeps15}\n" ++
    s!"straddle_decode_invariant_failures={st.decodeInvFailures}\n"
  IO.FS.writeFile statsPath statsContent
  if st.decodeInvFailures != 0 then
    IO.eprintln s!"TMR_BURST_STRADDLE_GEN_FAIL: decode_invariant_failures={st.decodeInvFailures}"
    return 1
  return 0

/-! ### Report Verification (`verify`) -/

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

def lookupStat (lines : List String) (k : String) : Option Nat :=
  let pfx := k ++ "="
  lines.findSome? fun l =>
    if l.startsWith pfx then parseNat ((l.drop pfx.length).toString) else none

def padLeft (s : String) (w : Nat) : String :=
  if s.length >= w then s
  else String.ofList (List.replicate (w - s.length) ' ') ++ s

def padRight (s : String) (w : Nat) : String :=
  if s.length >= w then s
  else s ++ String.ofList (List.replicate (w - s.length) ' ')

def ratFixed5 (r : Rat) : String :=
  let sign := if r < 0 then "-" else ""
  let n : Nat := (r.num.natAbs * 100000) / r.den
  let whole := n / 100000
  let frac := n % 100000
  let fracStr := toString frac
  let zeros := String.ofList (List.replicate (5 - min 5 fracStr.length) '0')
  s!"{sign}{whole}.{zeros}{fracStr}"

structure VerifyState where
  straddleExec : Nat := 0
  straddleOk : Nat := 0
  straddlePf : Nat := 0
  straddleUd : Nat := 0
  straddleDiffs : Nat := 0
  sawStraddleSummary : Bool := false
  sawStraddleNegCtrl : Bool := false
  sawKernelCode : Bool := false
  sawArenaGuard : Bool := false
  sawSyscallExit3 : Bool := false
  sawSyscallExit2 : Bool := false
  sawTfSweep : Bool := false
  sawSingleBurst : Bool := false
  sawPackedNegCtrl : Bool := false
  mcCount : Nat := 0
  failures : List String := []

def checkMcRow (qStr : String) (q tol : Rat) (fs : List String) (st : VerifyState) :
    IO VerifyState := do
  match getKV fs "trials" >>= parseNat,
        getKV fs "clean" >>= parseNat,
        getKV fs "corrected" >>= parseNat,
        getKV fs "refusal" >>= parseNat,
        getKV fs "stuck" >>= parseNat,
        getKV fs "sdc" >>= parseNat,
        getKV fs "sdc_cert" >>= parseNat,
        getKV fs "err" >>= parseNat with
  | some trials, some clean, some corr, some ref, some stuck, some sdc, some sdcCert, some err =>
    if trials == 0 then
      return { st with failures := st.failures ++ [s!"BURST_MC q={qStr}: trials=0"] }
    let sumOk := clean + corr + ref + stuck + sdc == trials && err == ref + stuck + sdc && sdcCert <= sdc
    let empRate : Rat := (err : Rat) / (trials : Rat)
    let sdcRate : Rat := (sdc : Rat) / (trials : Rat)
    let certRate : Rat := (sdcCert : Rat) / (trials : Rat)
    let bound : Rat := tmrBurstBound q
    let errOk : Bool := decide (empRate ≤ bound + tol)
    let sdcOk : Bool := decide (sdcRate ≤ bound + tol)
    let certOk : Bool := decide (certRate ≤ bound + tol)
    let ok := sumOk && errOk && sdcOk && certOk
    let status := if ok then "PASS" else "FAIL"
    IO.println s!"  {padRight (s!"BURST_MC q={qStr}") 18} | {padLeft (toString trials) 6} | {padLeft (toString clean) 6} | {padLeft (toString corr) 6} | {padLeft (toString ref) 5} | {padLeft (toString sdc) 4} | {padLeft (toString sdcCert) 7} | {padLeft (ratFixed5 empRate) 8} | {padLeft (ratFixed5 bound) 8} | {padLeft (ratFixed5 (bound + tol)) 9} | {status}"
    if ok then
      return st
    else
      return { st with failures := st.failures ++ [s!"BURST_MC q={qStr}: bound check failed (emp={empRate}, sdc={sdc}, sdc_cert={sdcCert}, bound={bound}, tol={tol})"] }
  | _, _, _, _, _, _, _, _ =>
    return { st with failures := st.failures ++ [s!"BURST_MC q={qStr}: malformed counts"] }

def runVerify (cpuReportPath statsPath : String) : IO UInt32 := do
  let statsRaw ← IO.FS.readFile statsPath
  let cpuRaw ← IO.FS.readFile cpuReportPath
  let sLines := (statsRaw.splitOn "\n").map (·.trimAsciiEnd.toString)
  let cLines := (cpuRaw.splitOn "\n").map (·.trimAsciiEnd.toString)

  let expTot := (lookupStat sLines "straddle_total_cases").getD 0
  let expOk := (lookupStat sLines "straddle_ok_cases").getD 0
  let expPf := (lookupStat sLines "straddle_pf_cases").getD 0
  let expUd := (lookupStat sLines "straddle_ud_cases").getD 0
  let expSweeps := (lookupStat sLines "straddle_k1_to_15_sweeps").getD 0
  let expInvBad := (lookupStat sLines "straddle_decode_invariant_failures").getD 1

  let mut st : VerifyState := {}
  if expInvBad != 0 || expTot < 50000 then
    st := { st with failures := st.failures ++ [s!"straddle gen stats invalid: total={expTot} (<50000) invBad={expInvBad}"] }

  for line in cLines do
    if !line.isEmpty then
      if line.startsWith "DIFF " || line.startsWith "  " then
        IO.eprintln line
      else
        let fs := line.splitOn " "
        match fs with
        | "STRADDLE_SUMMARY" :: rest =>
          match getKV rest "executed" >>= parseNat,
                getKV rest "ok" >>= parseNat,
                getKV rest "pf" >>= parseNat,
                getKV rest "ud" >>= parseNat,
                getKV rest "disagreements" >>= parseNat with
          | some exec, some okCnt, some pfCnt, some udCnt, some diffs =>
            let ok := exec == expTot && okCnt == expOk && pfCnt == expPf && udCnt == expUd && diffs == 0 && exec >= 50000
            let tag := if ok then "OK" else "FAIL"
            IO.println s!"[{tag}] Page-Straddle #PF vs #UD vs Retirement Differential (k=1..15):"
            IO.println s!"     total_cases={exec} (sweeps_k1_15={expSweeps}) retire_ok={okCnt} fetch_pf={pfCnt} trap_ud={udCnt} disagreements={diffs}"
            st := { st with
              sawStraddleSummary := true,
              straddleExec := exec,
              straddleOk := okCnt,
              straddlePf := pfCnt,
              straddleUd := udCnt,
              straddleDiffs := diffs,
              failures := if ok then st.failures else st.failures ++ [s!"straddle summary mismatch: exec={exec}/{expTot} ok={okCnt}/{expOk} pf={pfCnt}/{expPf} ud={udCnt}/{expUd} diffs={diffs}"] }
          | _, _, _, _, _ =>
            st := { st with failures := st.failures ++ ["malformed STRADDLE_SUMMARY line"] }
        | "STRADDLE_NEG_CTRL" :: res :: _ =>
          if res == "PASS" then
            IO.println "[OK] Page-Straddle negative control (untruncated straddle / #PF vs #UD swap detected): PASS"
            st := { st with sawStraddleNegCtrl := true }
          else
            IO.println s!"[FAIL] Page-Straddle negative control: {line}"
            st := { st with sawStraddleNegCtrl := true, failures := st.failures ++ ["STRADDLE_NEG_CTRL failed"] }
        | ["MEM_EXPOSURE_CODE", hexStr] =>
          if hexStr == expectedKernelHex then
            IO.println s!"[OK] memExposureKernelCode (415B at 0x400000): verified byte-for-byte against X86Dec.imageOf"
            st := { st with sawKernelCode := true }
          else
            IO.println s!"[FAIL] memExposureKernelCode 415B hex mismatch"
            st := { st with sawKernelCode := true, failures := st.failures ++ ["memExposureKernelCode 415B hex mismatch"] }
        | "ARENA_GUARD_CHECK" :: res :: _ =>
          if res == "PASS" then
            IO.println "[OK] 3 page-separated 4096B arenas (0x100000, 0x200000, 0x300000) + 6 PROT_NONE guard pages: PASS"
            st := { st with sawArenaGuard := true }
          else
            IO.println s!"[FAIL] Arena guard-page check: {line}"
            st := { st with sawArenaGuard := true, failures := st.failures ++ ["ARENA_GUARD_CHECK failed"] }
        | ["SYSCALL_EXIT3_SPLIT_CHECK", res] =>
          if res == "PASS" then
            IO.println "[OK] fork() child 0f 05 syscall exit(3) on 3-way split (val, last_step, step, acc): PASS"
            st := { st with sawSyscallExit3 := true }
          else
            IO.println s!"[FAIL] fork() child syscall exit(3) check: {res}"
            st := { st with sawSyscallExit3 := true, failures := st.failures ++ ["SYSCALL_EXIT3_SPLIT_CHECK failed"] }
        | ["SYSCALL_EXIT2_OVERFLOW_CHECK", res] =>
          if res == "PASS" then
            IO.println "[OK] fork() child 0f 05 syscall exit(2) on 64-bit overflow / invalid slot: PASS"
            st := { st with sawSyscallExit2 := true }
          else
            IO.println s!"[FAIL] fork() child syscall exit(2) check: {res}"
            st := { st with sawSyscallExit2 := true, failures := st.failures ++ ["SYSCALL_EXIT2_OVERFLOW_CHECK failed"] }
        | "UNMODIFIED_TF_SWEEP" :: rest =>
          match getKV rest "trials" >>= parseNat,
                getKV rest "corrected" >>= parseNat,
                getKV rest "refusal" >>= parseNat,
                getKV rest "stuck" >>= parseNat,
                getKV rest "sdc" >>= parseNat,
                getKV rest "sdc_cert" >>= parseNat with
          | some trials, some corr, some ref, some stuck, some sdc, some sdcCert =>
            let ok := trials >= 200 && corr == trials && ref == 0 && stuck == 0 && sdc == 0 && sdcCert == 0
            let tag := if ok then "OK" else "FAIL"
            IO.println s!"[{tag}] Unmodified 415B RFLAGS.TF=1 hardware single-stepped burst sweep: trials={trials} corrected={corr} SDC={sdc} SDC_cert={sdcCert}"
            st := { st with
              sawTfSweep := true,
              failures := if ok then st.failures else st.failures ++ [s!"UNMODIFIED_TF_SWEEP failed: corr={corr}/{trials} sdc={sdc} sdc_cert={sdcCert}"] }
          | _, _, _, _, _, _ =>
            st := { st with failures := st.failures ++ ["malformed UNMODIFIED_TF_SWEEP line"] }
        | "SINGLE_BURST64_SWEEP" :: rest =>
          match getKV rest "trials" >>= parseNat,
                getKV rest "clean" >>= parseNat,
                getKV rest "corrected" >>= parseNat,
                getKV rest "refusal" >>= parseNat,
                getKV rest "stuck" >>= parseNat,
                getKV rest "sdc" >>= parseNat,
                getKV rest "sdc_cert" >>= parseNat with
          | some trials, some clean, some corr, some ref, some stuck, some sdc, some sdcCert =>
            let ok := trials >= 3000 && clean == 0 && corr == trials && ref == 0 && stuck == 0 && sdc == 0 && sdcCert == 0
            let tag := if ok then "OK" else "FAIL"
            IO.println s!"[{tag}] Exhaustive single 64B cache-line burst sweep (0x00, 0xff, 512b flips, cert-attack, random 512b):"
            IO.println s!"     trials={trials} corrected={corr} refusal={ref} stuck={stuck} SDC={sdc} SDC_cert={sdcCert}"
            IO.println ""
            IO.println "  Sweep / Burst Rate | Trials |  Clean |   Corr | Refus |  SDC | SDCCert |  EmpRate |   3*q^2  | Bound+Tol | Status"
            IO.println "  -------------------+--------+--------+--------+-------+------+---------+----------+----------+-----------+-------"
            st := { st with
              sawSingleBurst := true,
              failures := if ok then st.failures else st.failures ++ [s!"SINGLE_BURST64_SWEEP failed: corr={corr}/{trials} ref={ref} stuck={stuck} SDC={sdc} SDC_cert={sdcCert}"] }
          | _, _, _, _, _, _, _ =>
            st := { st with failures := st.failures ++ ["malformed SINGLE_BURST64_SWEEP line"] }
        | "BURST_MC" :: rest =>
          match getKV rest "q" >>= parseRat, getKV rest "tol" >>= parseRat with
          | some q, some tol =>
            let qStr := (getKV rest "q").getD "?"
            st ← checkMcRow qStr q tol rest st
            st := { st with mcCount := st.mcCount + 1 }
          | _, _ =>
            st := { st with failures := st.failures ++ [s!"malformed BURST_MC line: {line}"] }
        | "PACKED_NEG_CTRL" :: res :: rest =>
          match getKV rest "trials" >>= parseNat,
                getKV rest "sdc" >>= parseNat,
                getKV rest "sdc_cert" >>= parseNat with
          | some trials, some sdc, some sdcCert =>
            let ok := res == "PASS" && trials > 0 && sdc > 0 && sdcCert > 0
            let tag := if ok then "OK" else "FAIL"
            IO.println ""
            IO.println s!"[{tag}] Packed-arena (<64B separation, A0=0x100000, A1=0x100020) negative control: trials={trials} single-burst SDC={sdc} SDC_cert={sdcCert}"
            st := { st with
              sawPackedNegCtrl := true,
              failures := if ok then st.failures else st.failures ++ [s!"PACKED_NEG_CTRL failed: res={res} trials={trials} sdc={sdc} sdc_cert={sdcCert}"] }
          | _, _, _ =>
            st := { st with failures := st.failures ++ [s!"malformed PACKED_NEG_CTRL line: {line}"] }
        | _ =>
          st := { st with failures := st.failures ++ [s!"unrecognized report line: {line}"] }

  let mut errs := st.failures
  if !st.sawStraddleSummary then errs := errs ++ ["missing STRADDLE_SUMMARY line"]
  if !st.sawStraddleNegCtrl then errs := errs ++ ["missing STRADDLE_NEG_CTRL line"]
  if !st.sawKernelCode then errs := errs ++ ["missing MEM_EXPOSURE_CODE line"]
  if !st.sawArenaGuard then errs := errs ++ ["missing ARENA_GUARD_CHECK line"]
  if !st.sawSyscallExit3 then errs := errs ++ ["missing SYSCALL_EXIT3_SPLIT_CHECK line"]
  if !st.sawSyscallExit2 then errs := errs ++ ["missing SYSCALL_EXIT2_OVERFLOW_CHECK line"]
  if !st.sawTfSweep then errs := errs ++ ["missing UNMODIFIED_TF_SWEEP line"]
  if !st.sawSingleBurst then errs := errs ++ ["missing SINGLE_BURST64_SWEEP line"]
  if st.mcCount != 4 then errs := errs ++ [s!"expected 4 BURST_MC rows, got {st.mcCount}"]
  if !st.sawPackedNegCtrl then errs := errs ++ ["missing PACKED_NEG_CTRL line"]

  IO.println ""
  if errs.isEmpty then
    IO.println s!"TMR_BURST_STRADDLE_GATE_OK: {st.straddleExec} page-straddle cases (0 disagreements), single-64B-burst SDC=0 & SDC_cert=0, all {st.mcCount} MC points <= 3*q_burst^2, negative controls PASS"
    return 0
  else
    for e in errs do
      IO.eprintln s!"TMR_BURST_STRADDLE_GATE_FAIL: {e}"
    return 1

def main (args : List String) : IO UInt32 := do
  match args with
  | ["gen", gStr, sStr, casesPath, statsPath] =>
    let g := (parseNat gStr).getD 3400
    let s := UInt64.ofNat ((parseNat sStr).getD 1)
    runGen g s casesPath statsPath
  | ["verify", cpuReportPath, statsPath] =>
    runVerify cpuReportPath statsPath
  | _ =>
    IO.eprintln "usage: TMRBurstCheck gen <seed_groups> <seed> <straddle_cases_file> <straddle_stats_file>"
    IO.eprintln "   or: TMRBurstCheck verify <cpu_report_file> <straddle_stats_file>"
    return 2
