/-
X86Dec: the byte decoder from x86-64 machine code to `X86Int.Ins`, the bridge
between a hex image and the semantics the TMR theorem is about (decisions 37
to 42 of the root-A stage-1 design).

`decode b a` reads the instruction at address `a`, where `b i` is the byte at
`a + i`, and returns the `X86Int.Ins` it means together with its length.
Targets of `jmp`, `jcc` and `call` come out resolved to absolute addresses, as
`X86Int` expects. The encodings are those of the Intel manual (vol. 2); each
accepted form is deliberately the narrowest one that has the 64-bit semantics
`X86Int.exec` gives it:

  - 64-bit operand size (`REX.W`) for register and memory forms, so no implicit
    32-bit zero-extension has to be modelled; the three exceptions
    (`mov r32, imm32`, `xor r32, r32` on one register, `movzx`) zero-extend to
    exactly the 64-bit value `exec` computes;
  - memory operands are `[base + disp]`; `[rip + disp]`, absolute and
    scaled-index forms are refused except the one `jtab` needs;
  - anything else decodes to `none`, which `X86Int` reads as `stuck`: outside
    the model, never a guess.

`altsAt` is the supplier of Hamming-distance-1 neighbours that decision 42 asks
of the byte decoder: one entry per flipped bit of the instruction's bytes.

`6A` is `push imm8`, `85` is `test r64, r64` and a 32-bit register move is
`mov32`. An indirect `jmp r64` (FF /4) is not decoded yet.
-/
import Bootstrap.X86Int

namespace X86Dec

open X86Env X86Int

/-- Sign-extend a byte / a 32-bit word to a 64-bit word. -/
def sx8 (v : Nat) : Nat := if v < 128 then v else v + W - 256
def sx32 (v : Nat) : Nat := if v < 2 ^ 31 then v else v + W - 2 ^ 32

/-- `n` bytes at offset `off`, little-endian. -/
def leB (b : Nat → Option Nat) (off : Nat) : Nat → Option Nat
  | 0 => some 0
  | n + 1 => do
    let x ← b off
    let r ← leB b (off + 1) n
    pure (x + 256 * r)

/-- The memory operand `[base + disp]` of ModRM byte `m` (`bb` the REX.B bit as
8, `x` the REX.X bit); `off` is the offset of the byte after ModRM. Returns the
base register, the displacement as a 64-bit word, and the offset after the
operand. A SIB byte is accepted only without an index. -/
def memOp (b : Nat → Option Nat) (off m bb : Nat) (x : Bool) : Option (Nat × Nat × Nat) := do
  let md := m / 64
  let rm := m % 8
  if md == 3 then none else
  let (base, off1) ←
    if rm == 4 then do
      let s ← b off
      if s / 8 % 8 != 4 || x then none else
      if md == 0 && s % 8 == 5 then none else
      pure (s % 8 + bb, off + 1)
    else if md == 0 && rm == 5 then none
    else pure (rm + bb, off)
  match md with
  | 0 => pure (base, 0, off1)
  | 1 => do let d ← b off1; pure (base, sx8 d, off1 + 1)
  | _ => do let d ← leB b off1 4; pure (base, sx32 d, off1 + 4)

def aluOfRR (o : Nat) : Option Alu :=
  if o == 0x01 then some .add else if o == 0x29 then some .sub
  else if o == 0x21 then some .and else if o == 0x09 then some .or
  else if o == 0x31 then some .xor else none

def aluOfExt (e : Nat) : Option Alu :=
  if e == 0 then some .add else if e == 1 then some .or
  else if e == 4 then some .and else if e == 5 then some .sub
  else if e == 6 then some .xor else none

/-- Decode one instruction at address `a`; `b i` is the byte at `a + i`. -/
def decode (b : Nat → Option Nat) (a : Nat) : Option (Ins × Nat) := do
  let b0 ← b 0
  let hasRex := 0x40 ≤ b0 && b0 ≤ 0x4f
  let rex := if hasRex then b0 - 0x40 else 0
  let p := if hasRex then 1 else 0
  let w := rex / 8 % 2 == 1
  let r := (rex / 4 % 2) * 8
  let x := rex / 2 % 2 == 1
  let bb := (rex % 2) * 8
  let o ← b p
  let tgt (len rel : Nat) : Nat := (a + len + rel) % W
  if 0xb8 ≤ o && o ≤ 0xbf then
    if r != 0 || x then none else
    if w then
      let imm ← leB b (p + 1) 8
      pure (.movI (o - 0xb8 + bb) imm, p + 9)
    else
      let imm ← leB b (p + 1) 4
      pure (.movI (o - 0xb8 + bb) imm, p + 5)
  else if o == 0x01 || o == 0x29 || o == 0x21 || o == 0x09 || o == 0x31 then
    let m ← b (p + 1)
    if m / 64 != 3 || x then none else
    let d := m % 8 + bb
    let s := m / 8 % 8 + r
    let op ← aluOfRR o
    if w then pure (.alu op d s, p + 2)
    else if o == 0x31 && d == s then pure (.alu .xor d d, p + 2)
    else none
  else if o == 0x89 || o == 0x8b then
    if x then none else
    let m ← b (p + 1)
    if !w then
      -- only the 32-bit register move: the low half, zero-extended
      if m / 64 != 3 then none else
      let rm := m % 8 + bb
      let rg := m / 8 % 8 + r
      if o == 0x89 then pure (.mov32 rm rg, p + 2) else pure (.mov32 rg rm, p + 2)
    else if m / 64 == 3 then
      let rm := m % 8 + bb
      let rg := m / 8 % 8 + r
      if o == 0x89 then pure (.mov rm rg, p + 2) else pure (.mov rg rm, p + 2)
    else
      let (base, disp, off') ← memOp b (p + 2) m bb x
      let rg := m / 8 % 8 + r
      if o == 0x89 then pure (.st base disp rg, off') else pure (.ld rg base disp, off')
  else if o == 0x85 then
    if !w || x then none else
    let m ← b (p + 1)
    if m / 64 != 3 then none else
    pure (.test (m % 8 + bb) (m / 8 % 8 + r), p + 2)
  else if o == 0x6a then
    if hasRex then none else
    let i ← b 1
    pure (.pushI (sx8 i), 2)
  else if o == 0x39 || o == 0x3b then
    if !w || x then none else
    let m ← b (p + 1)
    if m / 64 != 3 then none else
    let rm := m % 8 + bb
    let rg := m / 8 % 8 + r
    if o == 0x39 then pure (.cmp rm rg, p + 2) else pure (.cmp rg rm, p + 2)
  else if o == 0x83 || o == 0x81 then
    if !w || r != 0 || x then none else
    let m ← b (p + 1)
    if m / 64 != 3 then none else
    let d := m % 8 + bb
    let e := m / 8 % 8
    let (imm, len) ←
      if o == 0x83 then do let i ← b (p + 2); pure (sx8 i, p + 3)
      else do let i ← leB b (p + 2) 4; pure (sx32 i, p + 6)
    if e == 7 then pure (.cmpI d imm, len) else
    let op ← aluOfExt e
    pure (.aluI op d imm, len)
  else if o == 0xc1 then
    if !w || r != 0 || x then none else
    let m ← b (p + 1)
    let c ← b (p + 2)
    if m / 64 != 3 then none else
    let d := m % 8 + bb
    if m / 8 % 8 == 4 then pure (.shlI d c, p + 3)
    else if m / 8 % 8 == 5 then pure (.shrI d c, p + 3)
    else none
  else if o == 0x8d then
    if !w || x then none else
    let m ← b (p + 1)
    let (base, disp, off') ← memOp b (p + 2) m bb x
    pure (.lea (m / 8 % 8 + r) base disp, off')
  else if o == 0x88 then
    if x then none else
    let m ← b (p + 1)
    let (base, disp, off') ← memOp b (p + 2) m bb x
    let reg := m / 8 % 8
    if !hasRex && reg ≥ 4 then none else
    pure (.st8 base disp (reg + r), off')
  else if o == 0x0f then
    let o2 ← b (p + 1)
    if o2 == 0x05 then
      if hasRex then none else pure (.syscall, 2)
    else if o2 == 0xb6 then
      if x then none else
      let m ← b (p + 2)
      let (base, disp, off') ← memOp b (p + 3) m bb x
      pure (.ld8 (m / 8 % 8 + r) base disp, off')
    else if 0x80 ≤ o2 && o2 ≤ 0x8f then
      if hasRex then none else
      let d ← leB b 2 4
      pure (.jcc (o2 - 0x80) (tgt 6 (sx32 d)), 6)
    else none
  else if 0x70 ≤ o && o ≤ 0x7f then
    if hasRex then none else
    let d ← b 1
    pure (.jcc (o - 0x70) (tgt 2 (sx8 d)), 2)
  else if o == 0xeb then
    if hasRex then none else
    let d ← b 1
    pure (.jmp (tgt 2 (sx8 d)), 2)
  else if o == 0xe9 then
    if hasRex then none else
    let d ← leB b 1 4
    pure (.jmp (tgt 5 (sx32 d)), 5)
  else if o == 0xe8 then
    if hasRex then none else
    let d ← leB b 1 4
    pure (.call (tgt 5 (sx32 d)), 5)
  else if o == 0xc3 then
    if hasRex then none else pure (.ret, 1)
  else if 0x50 ≤ o && o ≤ 0x57 then
    if w || r != 0 || x then none else pure (.push (o - 0x50 + bb), p + 1)
  else if 0x58 ≤ o && o ≤ 0x5f then
    if w || r != 0 || x then none else pure (.pop (o - 0x58 + bb), p + 1)
  else if o == 0xff then
    -- jmp qword [base + 8·idx + disp]: ModRM /4 with a SIB of scale 8
    if w || r != 0 then none else
    let m ← b (p + 1)
    if m / 8 % 8 != 4 || m / 64 == 3 || m % 8 != 4 then none else
    let s ← b (p + 2)
    if s / 64 != 3 then none else
    let idx := s / 8 % 8 + (if x then 8 else 0)
    if idx == 4 then none else
    let md := m / 64
    if md == 0 && s % 8 == 5 then none else
    let base := s % 8 + bb
    match md with
    | 0 => pure (.jtab base idx 0, p + 3)
    | 1 => do let d ← b (p + 3); pure (.jtab base idx (sx8 d), p + 4)
    | _ => do let d ← leB b (p + 3) 4; pure (.jtab base idx (sx32 d), p + 7)
  else none

/-- `b` with bit `k` of the byte string flipped (byte `k / 8`, bit `k % 8`). -/
def flipBit (b : Nat → Option Nat) (k : Nat) : Nat → Option Nat :=
  fun i => if i == k / 8 then (b i).map fun v => Nat.xor v (2 ^ (k % 8)) else b i

/-- What a corrupted fetch may decode to at address `a`, for an instruction of
length `len`: one entry per encoding at Hamming distance 1 (decision 42). -/
def altsAt (b : Nat → Option Nat) (a len : Nat) : List (Option (Ins × Nat)) :=
  (List.range (8 * len)).map fun k => decode (flipBit b k) a

/-- The image seen by `X86Int.step`: the code bytes `code` loaded at `base`. -/
def imageOf (code : List Nat) (base : Nat) : Nat → Option (Ins × Nat) :=
  fun a => decode (fun i => code[a + i - base]?) a

/-! ### Encodings checked against the Intel manual

Each line is derived by hand from the opcode map, not produced by the decoder:
the byte string, then the instruction it must mean. -/

def at0 (bs : List Nat) (a : Nat := 0) : Option (Ins × Nat) := decode (fun i => bs[i]?) a

-- 49 89 d4: REX.WB, 89 /r, ModRM 11 010 100 → mov r12, rdx
example : at0 [0x49, 0x89, 0xd4] = some (.mov 12 2, 3) := by decide +kernel
-- 48 89 e5: mov rbp, rsp
example : at0 [0x48, 0x89, 0xe5] = some (.mov 5 4, 3) := by decide +kernel
-- 48 8b 45 f8: mov rax, [rbp - 8]
example : at0 [0x48, 0x8b, 0x45, 0xf8] = some (.ld 0 5 (W - 8), 4) := by decide +kernel
-- 48 89 44 24 10: mov [rsp + 16], rax (SIB 0x24 = no index, base rsp)
example : at0 [0x48, 0x89, 0x44, 0x24, 0x10] = some (.st 4 16 0, 5) := by decide +kernel
-- 41 8b 04 24: mov eax,[r12] is 32-bit and refused; 49 8b 04 24: mov rax, [r12]
example : at0 [0x41, 0x8b, 0x04, 0x24] = none := by decide +kernel
example : at0 [0x49, 0x8b, 0x04, 0x24] = some (.ld 0 12 0, 4) := by decide +kernel
-- 48 83 f8 40: cmp rax, 0x40
example : at0 [0x48, 0x83, 0xf8, 0x40] = some (.cmpI 0 0x40, 4) := by decide +kernel
-- 48 83 c0 ff: add rax, -1
example : at0 [0x48, 0x83, 0xc0, 0xff] = some (.aluI .add 0 (W - 1), 4) := by decide +kernel
-- 48 29 c3: sub rbx, rax
example : at0 [0x48, 0x29, 0xc3] = some (.alu .sub 3 0, 3) := by decide +kernel
-- 31 ff: xor edi, edi
example : at0 [0x31, 0xff] = some (.alu .xor 7 7, 2) := by decide +kernel
-- 48 c1 e0 04 / 48 c1 e8 04: shl rax, 4 / shr rax, 4
example : at0 [0x48, 0xc1, 0xe0, 0x04] = some (.shlI 0 4, 4) := by decide +kernel
example : at0 [0x48, 0xc1, 0xe8, 0x04] = some (.shrI 0 4, 4) := by decide +kernel
-- 0f b6 04 24: movzx eax, byte [rsp];  88 07: mov [rdi], al
example : at0 [0x0f, 0xb6, 0x04, 0x24] = some (.ld8 0 4 0, 4) := by decide +kernel
example : at0 [0x88, 0x07] = some (.st8 7 0 0, 2) := by decide +kernel
-- 40 88 37: mov [rdi], sil (REX present); without REX reg 6 is dh and refused
example : at0 [0x40, 0x88, 0x37] = some (.st8 7 0 6, 3) := by decide +kernel
example : at0 [0x88, 0x37] = none := by decide +kernel
-- ff 24 c3: jmp qword [rbx + rax*8]
example : at0 [0xff, 0x24, 0xc3] = some (.jtab 3 0 0, 3) := by decide +kernel
-- eb fe at 0x400000: jmp to itself
example : at0 [0xeb, 0xfe] 0x400000 = some (.jmp 0x400000, 2) := by decide +kernel
-- 75 f0 at 0x100: jne 0x100 + 2 - 16
example : at0 [0x75, 0xf0] 0x100 = some (.jcc 5 0xf2, 2) := by decide +kernel
-- 0f 84 00 01 00 00 at 0: je +0x100 from the end (6)
example : at0 [0x0f, 0x84, 0x00, 0x01, 0x00, 0x00] = some (.jcc 4 0x106, 6) := by decide +kernel
-- e8 05 00 00 00 at 0: call 10;  c3: ret;  0f 05: syscall;  57 / 5f: push rdi / pop rdi
example : at0 [0xe8, 0x05, 0x00, 0x00, 0x00] = some (.call 10, 5) := by decide +kernel
example : at0 [0xc3] = some (.ret, 1) := by decide +kernel
example : at0 [0x0f, 0x05] = some (.syscall, 2) := by decide +kernel
example : at0 [0x57] = some (.push 7, 1) := by decide +kernel
example : at0 [0x41, 0x5c] = some (.pop 12, 2) := by decide +kernel
-- b8 3c 00 00 00: mov eax, 60;  48 b8 + imm64: mov rax, 0x0102030405060708
example : at0 [0xb8, 0x3c, 0x00, 0x00, 0x00] = some (.movI 0 60, 5) := by decide +kernel
example : at0 [0x48, 0xb8, 8, 7, 6, 5, 4, 3, 2, 1] = some (.movI 0 0x0102030405060708, 10) := by
  decide +kernel
-- 48 8d 7c 24 08: lea rdi, [rsp + 8]
example : at0 [0x48, 0x8d, 0x7c, 0x24, 0x08] = some (.lea 7 4 8, 5) := by decide +kernel
-- 6a 3c: push 60;  6a ff: push -1;  48 85 c0: test rax, rax;  44 89 f7: mov edi, r14d
example : at0 [0x6a, 0x3c] = some (.pushI 60, 2) := by decide +kernel
example : at0 [0x6a, 0xff] = some (.pushI (W - 1), 2) := by decide +kernel
example : at0 [0x48, 0x85, 0xc0] = some (.test 0 0, 3) := by decide +kernel
example : at0 [0x44, 0x89, 0xf7] = some (.mov32 7 14, 3) := by decide +kernel
-- 7c 49 at 0x7d: jl 0x7d + 2 + 0x49;  78 34: js
example : at0 [0x7c, 0x49] 0x7d = some (.jcc 12 0xc8, 2) := by decide +kernel
-- refused: rip-relative, truncated input
example : at0 [0x48, 0x8b, 0x05, 0, 0, 0, 0] = none := by decide +kernel
example : at0 [0x48, 0x89] = none := by decide +kernel

end X86Dec
