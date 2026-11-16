/-
X86Int: the x86-64 instructions the Sporos interpreter is written in, over the
process memory and kernel of X86Env, with the silicon's faults in the PDist
monad (decisions 37 to 42 of the root-A stage-1 design).

Two semantics, one machine:

  - `step`: the architectural step, deterministic. It is what the Intel manual
    describes and what the differential harness tests against the CPU.
  - `pstep`: the physical step. The silicon may fail in four places
    (decision 42): the fetch may decode a neighbour of the instruction (`Alts`,
    one entry per encoding at Hamming distance 1, supplied by the byte
    decoder), and once the instruction retires, a bit of the register file,
    of the mapped memory or of the flags may flip. Each place has its own rate
    (`Rates`). Outcomes carry exact weights and a tag saying whether a fault
    occurred.

`faultFree_prun`: for every fault model, every program and every number of
steps, the zero-fault branch of the physical run is exactly the architectural
run. The deterministic semantics is not an idealisation chosen beside the
physics: it is read off the physics. `prun_ok`: the physical run is a
distribution.

`baseline`: for rates anywhere in their calibration intervals, a run of n steps
faults with probability at most n times the sum of the upper ends. This is the
unhardened bound that the triple-modular-redundancy theorem (decision 43) is
measured against.

Dispatch is a jump table (decision 38): `jtab b i d` is `jmp qword [b + 8·i + d]`;
`step_jtab` says where it goes. Instructions are given decoded, with their
lengths, by the image `im`.

Registers hold `Option Nat`: `none` is a value the architecture leaves undefined
(r11 after `syscall`), and reading it sticks the machine.
-/
import Bootstrap.X86Env
import Bootstrap.PDist

namespace X86Int

open X86Env PDist

/-- Register numbers: 0 rax 1 rcx 2 rdx 3 rbx 4 rsp 5 rbp 6 rsi 7 rdi 8..15 r8..r15. -/
abbrev Reg := Nat

inductive Alu where
  | add | sub | and | or | xor
deriving DecidableEq, Repr

inductive Ins where
  /-- `mov r64, imm64` -/
  | movI (d imm : Nat)
  /-- `mov r64, r64` -/
  | mov (d s : Reg)
  /-- `op r64, r64`: `d := d op s`, flags set -/
  | alu (op : Alu) (d s : Reg)
  /-- `op r64, imm` (imm already sign-extended to 64 bits) -/
  | aluI (op : Alu) (d imm : Nat)
  /-- `cmp r64, r64` -/
  | cmp (a b : Reg)
  /-- `cmp r64, imm` -/
  | cmpI (a imm : Nat)
  /-- `shl r64, imm8`, `shr r64, imm8`: flags left undefined -/
  | shlI (d c : Nat)
  | shrI (d c : Nat)
  /-- `mov r64, [b + disp]` and `mov [b + disp], r64` -/
  | ld (d b disp : Nat)
  | st (b disp s : Nat)
  /-- `movzx r64, byte [b + disp]` and `mov byte [b + disp], r8` -/
  | ld8 (d b disp : Nat)
  | st8 (b disp s : Nat)
  /-- `lea r64, [b + disp]` -/
  | lea (d b disp : Nat)
  /-- `jmp`, `jcc`, `call` with the target already resolved to an address -/
  | jmp (t : Nat)
  | jcc (cc t : Nat)
  /-- `jmp qword [b + 8·i + disp]`: dispatch through a table -/
  | jtab (b i disp : Nat)
  | call (t : Nat)
  | ret
  | push (s : Reg)
  | pop (d : Reg)
  | syscall
deriving DecidableEq, Repr

structure Flags where
  cf : Option Bool
  zf : Option Bool
deriving DecidableEq, Repr

structure St where
  regs : List (Option Nat)
  rip : Nat
  fl : Flags
  mem : Mem
  k : Kern
deriving Repr

inductive Out where
  | ok (s : St)
  | exit (status : Nat)
  /-- outside the model: an undefined value read, an unmapped address, a call
  the environment does not cover, a missing instruction -/
  | stuck
deriving Repr

def St.reg (s : St) (i : Reg) : Option Nat := (s.regs[i]?).getD none
def St.set (s : St) (i : Reg) (v : Option Nat) : St := { s with regs := s.regs.set i v }

/-- Little-endian bytes to a number. -/
def leVal : List Nat → Nat
  | [] => 0
  | b :: bs => b + 256 * leVal bs

def bytes8 (v : Nat) : List Nat := (List.range 8).map fun i => v / 256 ^ i % 256

def rd64 (m : Mem) (a : Nat) : Option Nat := (m.rdN a 8).map leVal
def wr64 (m : Mem) (a v : Nat) : Option Mem := m.wrN a (bytes8 v)

def aluF (op : Alu) (a b : Nat) : Nat × Flags :=
  match op with
  | .add => let r := (a + b) % W; (r, ⟨some (W ≤ a + b), some (r == 0)⟩)
  | .sub => let r := (a + W - b) % W; (r, ⟨some (a < b), some (r == 0)⟩)
  | .and => let r := Nat.land a b; (r, ⟨some false, some (r == 0)⟩)
  | .or => let r := Nat.lor a b; (r, ⟨some false, some (r == 0)⟩)
  | .xor => let r := Nat.xor a b; (r, ⟨some false, some (r == 0)⟩)

/-- Unsigned conditions: 2 b, 3 ae, 4 e, 5 ne, 6 be, 7 a. -/
def cond (f : Flags) (cc : Nat) : Option Bool := do
  match cc with
  | 2 => f.cf
  | 3 => (! ·) <$> f.cf
  | 4 => f.zf
  | 5 => (! ·) <$> f.zf
  | 6 => do let c ← f.cf; let z ← f.zf; pure (c || z)
  | 7 => do let c ← f.cf; let z ← f.zf; pure (!(c || z))
  | _ => none

def ea (b d : Nat) : Nat := (b + d) % W

/-- Lift a partial state into an outcome. -/
def lift : Option St → Out
  | some s => .ok s
  | none => .stuck

/-- Execute instruction `ins` of length `len` at `s.rip`. -/
def exec (ins : Ins) (len : Nat) (s : St) : Out :=
    let nx := s.rip + len
    let go (t : St) : St := { t with rip := nx }
    match ins with
    | .movI d imm => .ok (go (s.set d (some (imm % W))))
    | .mov d r => lift do let v ← s.reg r; pure (go (s.set d (some v)))
    | .alu op d r => lift do
        let a ← s.reg d; let b ← s.reg r
        let (v, f) := aluF op a b
        pure (go { s.set d (some v) with fl := f })
    | .aluI op d imm => lift do
        let a ← s.reg d
        let (v, f) := aluF op a (imm % W)
        pure (go { s.set d (some v) with fl := f })
    | .cmp a b => lift do
        let x ← s.reg a; let y ← s.reg b
        pure (go { s with fl := (aluF .sub x y).2 })
    | .cmpI a imm => lift do
        let x ← s.reg a
        pure (go { s with fl := (aluF .sub x (imm % W)).2 })
    | .shlI d c => lift do
        let a ← s.reg d
        pure (go { s.set d (some (a * 2 ^ (c % 64) % W)) with fl := ⟨none, none⟩ })
    | .shrI d c => lift do
        let a ← s.reg d
        pure (go { s.set d (some (a / 2 ^ (c % 64))) with fl := ⟨none, none⟩ })
    | .ld d b disp => lift do
        let a ← s.reg b; let v ← rd64 s.mem (ea a disp)
        pure (go (s.set d (some v)))
    | .st b disp r => lift do
        let a ← s.reg b; let v ← s.reg r; let m ← wr64 s.mem (ea a disp) v
        pure (go { s with mem := m })
    | .ld8 d b disp => lift do
        let a ← s.reg b; let v ← s.mem.rd (ea a disp)
        pure (go (s.set d (some v)))
    | .st8 b disp r => lift do
        let a ← s.reg b; let v ← s.reg r; let m ← s.mem.wr (ea a disp) (v % 256)
        pure (go { s with mem := m })
    | .lea d b disp => lift do
        let a ← s.reg b
        pure (go (s.set d (some (ea a disp))))
    | .jmp t => .ok { s with rip := t }
    | .jcc cc t => lift do
        let c ← cond s.fl cc
        pure { s with rip := if c then t else nx }
    | .jtab b i disp => lift do
        let x ← s.reg b; let y ← s.reg i
        let t ← rd64 s.mem ((x + 8 * y + disp) % W)
        pure { s with rip := t }
    | .call t => lift do
        let sp ← s.reg 4
        let sp' := (sp + W - 8) % W
        let m ← wr64 s.mem sp' nx
        pure { s.set 4 (some sp') with mem := m, rip := t }
    | .ret => lift do
        let sp ← s.reg 4
        let t ← rd64 s.mem sp
        pure { s.set 4 (some ((sp + 8) % W)) with rip := t }
    | .push r => lift do
        let v ← s.reg r; let sp ← s.reg 4
        let sp' := (sp + W - 8) % W
        let m ← wr64 s.mem sp' v
        pure (go { s.set 4 (some sp') with mem := m })
    | .pop d => lift do
        let sp ← s.reg 4
        let v ← rd64 s.mem sp
        pure (go ((s.set 4 (some ((sp + 8) % W))).set d (some v)))
    | .syscall =>
      match s.reg 0, s.reg 7, s.reg 6, s.reg 2, s.reg 10, s.reg 8, s.reg 9 with
      | some nr, some a1, some a2, some a3, some a4, some a5, some a6 =>
        match syscall s.k s.mem nr a1 a2 a3 a4 a5 a6 with
        | some (.ret rax k m) =>
          .ok (go { ((s.set 0 (some rax)).set 1 (some nx)).set 11 none with k := k, mem := m })
        | some (.exit c) => .exit c
        | none => .stuck
      | _, _, _, _, _, _, _ => .stuck

/-- The architectural step. `im a` is the instruction at `a` and its length. -/
def step (im : Nat → Option (Ins × Nat)) (s : St) : Out :=
  match im s.rip with
  | none => .stuck
  | some (ins, len) => exec ins len s

/-- The architectural run: at most `n` steps. -/
def run (im : Nat → Option (Ins × Nat)) : Nat → Out → Out
  | 0, o => o
  | n + 1, .ok s => run im n (step im s)
  | _ + 1, o => o

/-! ### Jump-table dispatch -/

theorem step_jtab {im : Nat → Option (Ins × Nat)} {s : St} {b i disp len x y t : Nat}
    (hi : im s.rip = some (.jtab b i disp, len)) (hx : s.reg b = some x) (hy : s.reg i = some y)
    (ht : rd64 s.mem ((x + 8 * y + disp) % W) = some t) :
    step im s = .ok { s with rip := t } := by
  simp only [step, exec, hi, hx, hy, ht, lift, bind, Option.bind, pure]

/-! ### Uniform faults -/

/-- With probability `1 - p` the clean outcome `c`; otherwise one of `xs`,
uniformly. With no alternatives the clean outcome is certain. -/
def unif {β : Type} (p : Rat) (c : β) (xs : List β) : PDF β :=
  match xs with
  | [] => pureF c
  | _ :: _ => ((c, false), 1 - p) :: xs.map fun x => ((x, true), p / xs.length)

theorem faultFree_unif {β : Type} (p : Rat) (c : β) (xs : List β) : faultFree (unif p c xs) = [c] := by
  cases xs with
  | nil => rfl
  | cons x r =>
    simp only [unif, faultFree, Bool.false_eq_true, if_false, List.cons.injEq, true_and]
    generalize (p / (r.length + 1 : Nat)) = w
    generalize (x :: r) = l
    induction l with
    | nil => rfl
    | cons y l ih => simpa [faultFree] using ih

theorem mass_const {β : Type} (l : List β) (w : Rat) (t : Bool) :
    mass (l.map fun x => ((x, t), w)) = l.length * w := by
  induction l with
  | nil => show (0 : Rat) = ((0 : Nat) : Rat) * w; rw [show ((0 : Nat) : Rat) = 0 from rfl, Rat.zero_mul]
  | cons x r ih =>
    simp only [List.map_cons, mass, ih, List.length_cons, Rat.natCast_add]
    rw [Rat.add_mul, show ((1 : Nat) : Rat) = 1 from rfl, Rat.one_mul, Rat.add_comm]

theorem fm_const {β : Type} (l : List β) (w : Rat) : fm (l.map fun x => ((x, true), w)) = l.length * w := by
  induction l with
  | nil => show (0 : Rat) = ((0 : Nat) : Rat) * w; rw [show ((0 : Nat) : Rat) = 0 from rfl, Rat.zero_mul]
  | cons x r ih =>
    simp only [List.map_cons, fm, if_true, ih, List.length_cons, Rat.natCast_add]
    rw [Rat.add_mul, show ((1 : Nat) : Rat) = 1 from rfl, Rat.one_mul, Rat.add_comm]

theorem share_sum (n : Nat) (h : n ≠ 0) (p : Rat) : (n : Rat) * (p / n) = p := by
  rw [Rat.div_def, Rat.mul_comm p, ← Rat.mul_assoc, Rat.mul_inv_cancel _ (by exact_mod_cast h), Rat.one_mul]

theorem share_nonneg (n : Nat) (h : n ≠ 0) {p : Rat} (hp : 0 ≤ p) : 0 ≤ p / n := by
  rw [Rat.div_def]
  exact Rat.mul_nonneg hp (Rat.le_of_lt (Rat.inv_pos.2 (Rat.natCast_pos.2 (Nat.pos_of_ne_zero h))))

theorem unif_ok {β : Type} {p : Rat} (hp : Rate p) (c : β) (xs : List β) : (unif p c xs).ok := by
  cases xs with
  | nil => exact pureF_ok c
  | cons x r =>
    have hn : (x :: r).length ≠ 0 := by simp
    refine ⟨?_, ?_⟩
    · intro y hy
      simp only [unif, List.mem_cons, List.mem_map] at hy
      rcases hy with rfl | ⟨z, _, rfl⟩
      · exact (Rat.le_iff_sub_nonneg _ _).1 hp.2
      · exact share_nonneg _ hn hp.1
    · show (1 - p) + mass ((x :: r).map fun y => ((y, true), p / (x :: r).length)) = 1
      rw [mass_const, share_sum _ hn, Rat.sub_add_cancel]

theorem fm_unif_le {β : Type} {p : Rat} (hp : Rate p) (c : β) (xs : List β) : fm (unif p c xs) ≤ p := by
  cases xs with
  | nil => rw [unif, fm_pureF]; exact hp.1
  | cons x r =>
    have hn : (x :: r).length ≠ 0 := by simp
    show (if false = true then 1 - p else 0) + fm ((x :: r).map fun y => ((y, true), p / (x :: r).length)) ≤ p
    rw [fm_const, share_sum _ hn]
    simp only [Bool.false_eq_true, if_false]
    rw [Rat.zero_add]; exact Rat.le_refl

theorem nonneg_add {a b : Rat} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a + b := by
  have := add_le_add' ha hb; rwa [Rat.add_zero] at this

/-! ### Relabelling -/

def relabel {α β : Type} (g : α → β) (d : PDF α) : PDF β := d.map fun x => ((g x.1.1, x.1.2), x.2)

theorem faultFree_relabel {α β : Type} (g : α → β) (d : PDF α) :
    faultFree (relabel g d) = (faultFree d).map g := by
  induction d with
  | nil => rfl
  | cons x r ih =>
    obtain ⟨⟨a, t⟩, w⟩ := x
    cases t <;> simp_all [relabel, faultFree]

theorem mass_relabel {α β : Type} (g : α → β) (d : PDF α) : mass (relabel g d) = mass d := by
  induction d with
  | nil => rfl
  | cons x r ih => simp_all [relabel, mass]

theorem fm_relabel {α β : Type} (g : α → β) (d : PDF α) : fm (relabel g d) = fm d := by
  induction d with
  | nil => rfl
  | cons x r ih => obtain ⟨⟨a, t⟩, w⟩ := x; simp_all [relabel, fm]

theorem relabel_ok {α β : Type} (g : α → β) {d : PDF α} (h : d.ok) : (relabel g d).ok := by
  refine ⟨?_, by rw [mass_relabel]; exact h.2⟩
  intro y hy
  simp only [relabel, List.mem_map] at hy
  obtain ⟨x, hx, rfl⟩ := hy
  exact h.1 x hx

/-! ### Fault models -/

/-- A fault model on retired states: what the silicon may do to the state once
an instruction retires. Without a fault the state is untouched (`clean`), the
outcomes form a distribution (`ok`), and a fault occurs with probability at most
`q` (`le`). -/
structure Fault where
  hit : St → PDF St
  q : Rat
  q0 : 0 ≤ q
  clean : ∀ s, faultFree (hit s) = [s]
  ok : ∀ s, (hit s).ok
  le : ∀ s, fm (hit s) ≤ q

/-- No fault. -/
def Fault.none : Fault := ⟨pureF, 0, Rat.le_refl, fun _ => rfl, pureF_ok, fun s => by rw [fm_pureF]; exact Rat.le_refl⟩

/-- One fault model after another: rates add. -/
def Fault.seq (F G : Fault) : Fault where
  hit s := bindF (F.hit s) G.hit
  q := F.q + G.q
  q0 := nonneg_add F.q0 G.q0
  clean s := by rw [faultFree_bindF, F.clean]; simp [G.clean]
  ok s := bindF_ok (F.ok s) G.ok
  le s := Rat.le_trans (fm_bindF_le' (F.ok s) G.q0 G.ok G.le) (add_le_add' (F.le s) Rat.le_refl)

/-- Invert bit `b` of register `i`; an undefined register stays undefined. -/
def St.flipReg (s : St) (i b : Nat) : St := s.set i ((s.reg i).map fun v => Nat.xor v (2 ^ b))

/-- Every single-bit upset of the 16 × 64-bit register file. -/
def regFlips (s : St) : List St := (List.range (16 * 64)).map fun j => s.flipReg (j / 64) (j % 64)

/-- Invert bit `b` of byte `j` of mapping `i`. -/
def St.flipMem (s : St) (i j b : Nat) : St :=
  { s with mem := s.mem.modify i fun r => { r with data := r.data.modify j fun v => Nat.xor v (2 ^ b) } }

/-- Every single-bit upset of the mapped bytes. -/
def memFlips (s : St) : List St :=
  (List.range s.mem.length).flatMap fun i =>
    (List.range ((s.mem[i]?.map fun r => r.data.length).getD 0)).flatMap fun j =>
      (List.range 8).map fun b => s.flipMem i j b

/-- Every single-bit upset of the defined flags. -/
def flagFlips (s : St) : List St :=
  (s.fl.cf.toList.map fun c => { s with fl := { s.fl with cf := some (!c) } }) ++
  (s.fl.zf.toList.map fun z => { s with fl := { s.fl with zf := some (!z) } })

/-- Upsets uniform over the alternatives `alt s`, at rate `p`. -/
def Fault.uniform (alt : St → List St) (p : Rat) (hp : Rate p) : Fault where
  hit s := unif p s (alt s)
  q := p
  q0 := hp.1
  clean s := faultFree_unif p s (alt s)
  ok s := unif_ok hp s (alt s)
  le s := fm_unif_le hp s (alt s)

/-! ### The physical step -/

/-- What a corrupted fetch may decode to, for the instruction `ins` of length
`len`: one entry per encoding at Hamming distance 1, `none` when the corrupted
bytes do not decode. Supplied by the byte decoder. -/
abbrev Alts := Ins → Nat → List (Option (Ins × Nat))

def execO (s : St) : Option (Ins × Nat) → Out
  | some (j, l) => exec j l s
  | none => .stuck

/-- The rates of the four places the silicon can fail (decision 42). -/
structure Rates where
  fetch : Rat
  reg : Rat
  mem : Rat
  flag : Rat

def Rates.ok (r : Rates) : Prop := Rate r.fetch ∧ Rate r.reg ∧ Rate r.mem ∧ Rate r.flag

def Rates.total (r : Rates) : Rat := r.fetch + (r.reg + (r.mem + r.flag))

/-- The fault model on retired states: register file, then memory, then flags. -/
def Rates.state (r : Rates) (h : r.ok) : Fault :=
  (Fault.uniform regFlips r.reg h.2.1).seq ((Fault.uniform memFlips r.mem h.2.2.1).seq
    (Fault.uniform flagFlips r.flag h.2.2.2))

/-- The physical step: a fetch that may decode to a neighbour of the instruction,
its execution, then the state faults on the state it retires to. A machine
that halts or sticks has no state to corrupt. -/
def pstep (alts : Alts) (F : Fault) (pf : Rat) (im : Nat → Option (Ins × Nat)) (s : St) : PDF Out :=
  let fetched : PDF Out := match im s.rip with
    | none => pureF .stuck
    | some (ins, len) => unif pf (exec ins len s) ((alts ins len).map (execO s))
  bindF fetched fun o => match o with
    | .ok s' => relabel Out.ok (F.hit s')
    | o => pureF o

theorem retire_ok (F : Fault) (o : Out) :
    (match o with | .ok s' => relabel Out.ok (F.hit s') | o => pureF o).ok := by
  cases o with
  | ok s' => exact relabel_ok _ (F.ok s')
  | exit c => exact pureF_ok _
  | stuck => exact pureF_ok _

theorem retire_le (F : Fault) (o : Out) :
    fm (match o with | .ok s' => relabel Out.ok (F.hit s') | o => pureF o) ≤ F.q := by
  cases o with
  | ok s' => rw [fm_relabel]; exact F.le s'
  | exit c => show fm (pureF (Out.exit c)) ≤ F.q; rw [fm_pureF]; exact F.q0
  | stuck => show fm (pureF Out.stuck) ≤ F.q; rw [fm_pureF]; exact F.q0

theorem retire_clean (F : Fault) (o : Out) :
    faultFree (match o with | .ok s' => relabel Out.ok (F.hit s') | o => pureF o) = [o] := by
  cases o with
  | ok s' => rw [faultFree_relabel, F.clean]; rfl
  | exit c => rfl
  | stuck => rfl

theorem faultFree_pstep (alts : Alts) (F : Fault) (pf : Rat) (im : Nat → Option (Ins × Nat)) (s : St) :
    faultFree (pstep alts F pf im s) = [step im s] := by
  unfold pstep step
  rw [faultFree_bindF]
  cases im s.rip with
  | none => simp [faultFree_pureF, retire_clean]
  | some p =>
    obtain ⟨ins, len⟩ := p
    simp [faultFree_unif, retire_clean]

theorem fetched_ok {pf : Rat} (hpf : Rate pf) (alts : Alts) (im : Nat → Option (Ins × Nat)) (s : St) :
    (match im s.rip with
      | none => pureF Out.stuck
      | some (ins, len) => unif pf (exec ins len s) ((alts ins len).map (execO s))).ok := by
  cases im s.rip with
  | none => exact pureF_ok _
  | some p => obtain ⟨ins, len⟩ := p; exact unif_ok hpf _ _

theorem fetched_le {pf : Rat} (hpf : Rate pf) (alts : Alts) (im : Nat → Option (Ins × Nat)) (s : St) :
    fm (match im s.rip with
      | none => pureF Out.stuck
      | some (ins, len) => unif pf (exec ins len s) ((alts ins len).map (execO s))) ≤ pf := by
  cases im s.rip with
  | none => rw [fm_pureF]; exact hpf.1
  | some p => obtain ⟨ins, len⟩ := p; exact fm_unif_le hpf _ _

theorem pstep_ok (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf) (im : Nat → Option (Ins × Nat))
    (s : St) : (pstep alts F pf im s).ok :=
  bindF_ok (fetched_ok hpf alts im s) (retire_ok F)

theorem fm_pstep (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf) (im : Nat → Option (Ins × Nat))
    (s : St) : fm (pstep alts F pf im s) ≤ pf + F.q :=
  Rat.le_trans (fm_bindF_le' (fetched_ok hpf alts im s) F.q0 (retire_ok F) (retire_le F))
    (add_le_add' (fetched_le hpf alts im s) Rat.le_refl)

/-- The physical run: at most `n` steps, each a physical step. -/
def prun (alts : Alts) (F : Fault) (pf : Rat) (im : Nat → Option (Ins × Nat)) : Nat → Out → PDF Out
  | 0, o => pureF o
  | n + 1, .ok s => bindF (pstep alts F pf im s) (prun alts F pf im n)
  | _ + 1, o => pureF o

/-- The zero-fault branch of the physical run is the architectural run, for
every fault model, program, start and length. -/
theorem faultFree_prun (alts : Alts) (F : Fault) (pf : Rat) (im : Nat → Option (Ins × Nat)) :
    ∀ n o, faultFree (prun alts F pf im n o) = [run im n o] := by
  intro n
  induction n with
  | zero => intro o; rfl
  | succ n ih =>
    intro o
    cases o with
    | ok s =>
      simp only [prun, run]
      rw [faultFree_bindF, faultFree_pstep]
      simp [ih]
    | exit c => rfl
    | stuck => rfl

theorem prun_ok (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf) (im : Nat → Option (Ins × Nat)) :
    ∀ n o, (prun alts F pf im n o).ok := by
  intro n
  induction n with
  | zero => intro o; exact pureF_ok _
  | succ n ih =>
    intro o
    cases o with
    | ok s => exact bindF_ok (pstep_ok alts F hpf im s) ih
    | exit c => exact pureF_ok _
    | stuck => exact pureF_ok _

/-- The baseline bound: a run of `n` steps faults with probability at most `n`
times the per-step rate. -/
theorem fm_prun (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf) (im : Nat → Option (Ins × Nat)) :
    ∀ n o, fm (prun alts F pf im n o) ≤ n * (pf + F.q) := by
  have h0 : 0 ≤ pf + F.q := nonneg_add hpf.1 F.q0
  intro n
  induction n with
  | zero =>
    intro o; rw [prun, fm_pureF]
    show (0 : Rat) ≤ ((0 : Nat) : Rat) * _
    rw [show ((0 : Nat) : Rat) = 0 from rfl, Rat.zero_mul]; exact Rat.le_refl
  | succ n ih =>
    intro o
    have hn : ((n + 1 : Nat) : Rat) * (pf + F.q) = (pf + F.q) + n * (pf + F.q) := by
      rw [Rat.natCast_add, Rat.add_mul, show ((1 : Nat) : Rat) = 1 from rfl, Rat.one_mul, Rat.add_comm]
    have hnn : 0 ≤ (n : Rat) * (pf + F.q) := Rat.mul_nonneg Rat.natCast_nonneg h0
    cases o with
    | ok s =>
      simp only [prun]
      rw [hn]
      exact Rat.le_trans (fm_bindF_le' (pstep_ok alts F hpf im s) hnn (prun_ok alts F hpf im n) ih)
        (add_le_add' (fm_pstep alts F hpf im s) Rat.le_refl)
    | exit c =>
      show fm (pureF (Out.exit c)) ≤ _; rw [fm_pureF, hn]
      exact Rat.le_trans hnn (by
        have := add_le_add' h0 (Rat.le_refl (a := (n : Rat) * (pf + F.q)))
        rwa [Rat.zero_add] at this)
    | stuck =>
      show fm (pureF Out.stuck) ≤ _; rw [fm_pureF, hn]
      exact Rat.le_trans hnn (by
        have := add_le_add' h0 (Rat.le_refl (a := (n : Rat) * (pf + F.q)))
        rwa [Rat.zero_add] at this)

/-- The baseline with every failure place: for rates anywhere in their
calibrations, a run of `n` steps faults with probability at most `n` times the
sum of the upper ends. Outside the faulty outcomes, the run is the architectural
one (`faultFree_prun`). -/
theorem baseline (alts : Alts) (im : Nat → Option (Ins × Nat)) (cf cr cm cl : Iv)
    (r : Rates) (hr : r.ok) (hf : cf.mem r.fetch) (hg : cr.mem r.reg) (hm : cm.mem r.mem)
    (hl : cl.mem r.flag) (n : Nat) (o : Out) :
    fm (prun alts (r.state hr) r.fetch im n o) ≤ n * (cf.hi + (cr.hi + (cm.hi + cl.hi))) := by
  refine Rat.le_trans (fm_prun alts (r.state hr) hr.1 im n o) ?_
  refine Rat.mul_le_mul_of_nonneg_left ?_ Rat.natCast_nonneg
  show r.fetch + (r.reg + (r.mem + r.flag)) ≤ _
  exact add_le_add' hf.2 (add_le_add' hg.2 (add_le_add' hm.2 hl.2))

/-! ### Witness -/

/-- A dispatch: `rbx` points at a two-entry table at 0x1000, `rcx` = 1 selects
the second entry, 0x40, where the program exits with status 7. -/
def tbl : Region := ⟨0x1000, bytes8 0x20 ++ bytes8 0x40, false⟩

def prog : Nat → Option (Ins × Nat)
  | 0x00 => some (.movI 3 0x1000, 10)
  | 0x0a => some (.movI 1 1, 10)
  | 0x14 => some (.jtab 3 1 0, 3)
  | 0x20 => some (.movI 7 1, 10)
  | 0x40 => some (.movI 7 7, 10)
  | 0x4a => some (.movI 0 60, 10)
  | 0x54 => some (.syscall, 2)
  | _ => none

def k0 : Kern := ⟨[], [], [], [], []⟩

def s0 : St := ⟨List.replicate 16 (some 0), 0, ⟨none, none⟩, [tbl], k0⟩

def Out.code : Out → Option Nat
  | .exit c => some c
  | _ => none

theorem witness : (run prog 6 (.ok s0)).code = some 7 := by
  decide +kernel

/-- Under every rate in [5·10⁻⁸, 2·10⁻⁷] at each of the four failure places, the
six-step run is a distribution, it faults with probability at most 6 · 8·10⁻⁷,
and its zero-fault branch is the exit with status 7. -/
theorem witness_phys (alts : Alts) (r : Rates) (hr : r.ok) (hf : p7.mem r.fetch) (hg : p7.mem r.reg)
    (hm : p7.mem r.mem) (hl : p7.mem r.flag) :
    (prun alts (r.state hr) r.fetch prog 6 (.ok s0)).ok ∧
    fm (prun alts (r.state hr) r.fetch prog 6 (.ok s0)) ≤ 6 * (4 / 5000000) ∧
    faultFree (prun alts (r.state hr) r.fetch prog 6 (.ok s0)) = [run prog 6 (.ok s0)] := by
  refine ⟨prun_ok _ _ hr.1 _ _ _, ?_, faultFree_prun _ _ _ _ _ _⟩
  have := baseline alts prog p7 p7 p7 p7 r hr hf hg hm hl 6 (.ok s0)
  refine Rat.le_trans this ?_
  decide +kernel

end X86Int
