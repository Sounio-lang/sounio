/-
X86Env: the Linux process environment the Sporos interpreter runs in
(decisions 25, 33 and 35 of the root-A stage-1 design).

X86Min models what the two seeds need: a 32-byte stack window, one-byte reads
from stdin, one-byte writes to stdout. The interpreter needs more:

  - memory that the kernel maps on request (`mmap`), grows (`mremap`) and frees
    (`munmap`); bytes of an anonymous mapping start at 0, as Linux guarantees;
  - a file named on the command line (`argv[1]`), opened read-only (`open`);
  - reads and writes of many bytes, which the kernel may cut short;
  - the initial process stack: `argc`, the `argv` pointers and strings.

What the kernel is free to choose, it chooses through an oracle, a list of
`Choice`s consumed one per decision: where a mapping lands, whether memory is
refused, how many bytes a read or write moves. A theorem about a program holds
for every oracle that is well-formed (`Choice.wf`), so it holds whatever the
real kernel picks. Nothing here touches X86Min, whose seed proofs stay as they
are.

Error returns are what Linux returns: `-errno` in two's complement.

`syscall_wf`: every call, under every oracle, keeps the mappings page-aligned
and pairwise disjoint.
-/
namespace X86Env

def W : Nat := 2 ^ 64
def page : Nat := 4096

/-- `-errno` as a 64-bit word. -/
def neg (e : Nat) : Nat := W - e

def ENOENT : Nat := 2
def EBADF : Nat := 9
def ENOMEM : Nat := 12
def EFAULT : Nat := 14
def EINVAL : Nat := 22

/-- Round up to whole pages. -/
def pages (n : Nat) : Nat := (n + page - 1) / page * page

/-- One mapping: its first address, its bytes, and whether it is writable. -/
structure Region where
  base : Nat
  data : List Nat
  rw : Bool
deriving DecidableEq, Repr

def Region.hi (r : Region) : Nat := r.base + r.data.length

abbrev Mem := List Region

/-- The region holding address `a`, with its index. -/
def Mem.find : Mem → Nat → Nat → Option (Nat × Region)
  | [], _, _ => none
  | r :: rest, a, i => if r.base ≤ a ∧ a < r.hi then some (i, r) else Mem.find rest a (i + 1)

def Mem.rd (m : Mem) (a : Nat) : Option Nat := do
  let (_, r) ← Mem.find m a 0
  r.data[a - r.base]?

def Mem.wr (m : Mem) (a v : Nat) : Option Mem := do
  let (i, r) ← Mem.find m a 0
  if r.rw then pure (m.set i { r with data := r.data.set (a - r.base) (v % 256) }) else none

/-- Read `n` bytes from `a`. -/
def Mem.rdN (m : Mem) : Nat → Nat → Option (List Nat)
  | _, 0 => some []
  | a, n + 1 => do
    let b ← m.rd a
    let r ← Mem.rdN m (a + 1) n
    pure (b :: r)

/-- Write the bytes `bs` from `a`. -/
def Mem.wrN (m : Mem) : Nat → List Nat → Option Mem
  | _, [] => some m
  | a, b :: bs => do
    let m ← m.wr a b
    Mem.wrN m (a + 1) bs

/-- `[a, a + n)` meets no mapping. -/
def Mem.free (m : Mem) (a n : Nat) : Bool := m.all fun r => a + n ≤ r.base || r.hi ≤ a

/-- No two mappings overlap, every mapping is page-aligned. -/
def Mem.wf (m : Mem) : Prop :=
  (∀ r ∈ m, r.base % page = 0) ∧ m.Pairwise (fun r s => r.hi ≤ s.base ∨ s.hi ≤ r.base)

/-- What the kernel decides when it is free to. -/
inductive Choice where
  /-- place a new or moved mapping at this address -/
  | at (a : Nat)
  /-- refuse memory: `-ENOMEM` -/
  | nomem
  /-- move this many bytes in a read or write -/
  | count (n : Nat)
deriving DecidableEq, Repr

/-- A file system: names and contents. -/
abbrev Files := List (List Nat × List Nat)

/-- An open file: its contents and the position of the next read. -/
structure Fd where
  data : List Nat
  pos : Nat
deriving DecidableEq, Repr

structure Kern where
  files : Files
  /-- descriptor `3 + i` is `fds[i]` -/
  fds : List (Option Fd)
  stdin : List Nat
  /-- bytes written to stdout, in order -/
  out : List Nat
  /-- the oracle, consumed from the front -/
  oracle : List Choice
deriving DecidableEq, Repr

/-- The next choice, if any. -/
def Kern.next (k : Kern) : Option Choice × Kern :=
  match k.oracle with
  | [] => (none, k)
  | c :: rest => (some c, { k with oracle := rest })

/-- How many of `want` bytes (with `have_` available) a read moves: the
oracle's count when it is a legal short count, else all it can. A count of 0 on
a nonempty request would mean end of file, so it is not a legal short count. -/
def Kern.count (k : Kern) (want avail : Nat) : Nat × Kern :=
  let full := min want avail
  match k.next with
  | (some (.count n), k') => if 0 < n ∧ n ≤ full then (n, k') else (full, k')
  | (_, _) => (full, k)

/-- A NUL-terminated string of at most `n` bytes from `a`. -/
def Mem.cstr (m : Mem) : Nat → Nat → Option (List Nat)
  | _, 0 => none
  | a, n + 1 => do
    let b ← m.rd a
    if b = 0 then pure [] else do
      let r ← Mem.cstr m (a + 1) n
      pure (b :: r)

def lookup (fs : Files) (name : List Nat) : Option (List Nat) :=
  match fs.find? (fun f => f.1 == name) with
  | some f => some f.2
  | none => none

/-- The lowest free descriptor slot. -/
def freeSlot : List (Option Fd) → Nat
  | [] => 0
  | none :: _ => 0
  | some _ :: r => freeSlot r + 1

def setSlot (fds : List (Option Fd)) (i : Nat) (v : Option Fd) : List (Option Fd) :=
  if i < fds.length then fds.set i v else fds ++ [v]

/-- The result of a system call: the value in rax and the new kernel and memory,
or the end of the process. `none` from a call means the model does not cover it
(the machine is stuck). -/
inductive SysRes where
  | ret (rax : Nat) (k : Kern) (m : Mem)
  | exit (status : Nat)
deriving Repr

def MAP_PRIVATE_ANON : Nat := 0x22
def PROT_RW : Nat := 3
def MREMAP_MAYMOVE : Nat := 1

/-- A place for a mapping of `n` bytes, chosen by the oracle; `none` when the
oracle gives no legal address (the model does not decide for the kernel). -/
def place (m : Mem) (c : Option Choice) (n : Nat) : Option (Option Nat) :=
  match c with
  | some .nomem => some none
  | some (.at a) =>
    if a % page = 0 ∧ 0 < a ∧ a + n ≤ 2 ^ 47 ∧ m.free a n then some (some a) else none
  | _ => none

/-- Linux x86-64 system calls the interpreter uses, by number in rax and
arguments in rdi, rsi, rdx, r10, r8, r9. -/
def syscall (k : Kern) (m : Mem) (nr a1 a2 a3 a4 a5 a6 : Nat) : Option SysRes :=
  match nr with
  | 0 => -- read(fd, buf, count)
    if a1 = 0 then
      let (n, k1) := k.count a3 k.stdin.length
      do let m' ← m.wrN a2 (k.stdin.take n)
         pure (.ret n { k1 with stdin := k.stdin.drop n } m')
    else if 3 ≤ a1 then
      match k.fds[a1 - 3]? with
      | some (some fd) =>
        let rest := fd.data.drop fd.pos
        let (n, k1) := k.count a3 rest.length
        do let m' ← m.wrN a2 (rest.take n)
           pure (.ret n { k1 with fds := k.fds.set (a1 - 3) (some { fd with pos := fd.pos + n }) } m')
      | _ => pure (.ret (neg EBADF) k m)
    else pure (.ret (neg EBADF) k m)
  | 1 => -- write(fd, buf, count): stdout only
    if a1 = 1 then
      let (n, k1) := k.count a3 a3
      do let bs ← m.rdN a2 n
         pure (.ret n { k1 with out := k.out ++ bs } m)
    else pure (.ret (neg EBADF) k m)
  | 2 => -- open(path, O_RDONLY)
    if a2 ≠ 0 then none else do
      let name ← m.cstr a1 4096
      match lookup k.files name with
      | some data =>
        let i := freeSlot k.fds
        pure (.ret (3 + i) { k with fds := setSlot k.fds i (some ⟨data, 0⟩) } m)
      | none => pure (.ret (neg ENOENT) k m)
  | 3 => -- close(fd)
    if 3 ≤ a1 ∧ (k.fds[a1 - 3]?).join.isSome then
      pure (.ret 0 { k with fds := k.fds.set (a1 - 3) none } m)
    else pure (.ret (neg EBADF) k m)
  | 9 => -- mmap(0, len, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
    if a1 ≠ 0 ∨ a3 ≠ PROT_RW ∨ a4 ≠ MAP_PRIVATE_ANON ∨ a5 ≠ W - 1 ∨ a6 ≠ 0 then none else
    if a2 = 0 then pure (.ret (neg EINVAL) k m) else
    let n := pages a2
    let (c, k1) := k.next
    do match ← place m c n with
       | none => pure (.ret (neg ENOMEM) k1 m)
       | some a => pure (.ret a k1 (m ++ [⟨a, List.replicate n 0, true⟩]))
  | 11 => -- munmap(addr, len): a whole mapping, exactly
    match m.find a1 0 with
    | some (i, r) =>
      if r.base = a1 ∧ r.data.length = pages a2 then pure (.ret 0 k (m.eraseIdx i))
      else none
    | none => none
  | 25 => -- mremap(old, oldlen, newlen, MREMAP_MAYMOVE)
    if a4 ≠ MREMAP_MAYMOVE then none else
    match m.find a1 0 with
    | some (i, r) =>
      if r.base ≠ a1 ∨ r.data.length ≠ pages a2 ∨ ¬ r.rw then none else
      let n := pages a3
      if n < r.data.length then none else
      let rest := m.eraseIdx i
      let (c, k1) := k.next
      do match ← place rest c n with
         | none => pure (.ret (neg ENOMEM) k1 m)
         | some a => pure (.ret a k1 (rest ++ [⟨a, r.data ++ List.replicate (n - r.data.length) 0, true⟩]))
    | none => none
  | 60 => pure (.exit (a1 % 256))
  | _ => none

/-! ### The initial process stack -/

/-- 64-bit little-endian bytes. -/
def le64 (v : Nat) : List Nat := (List.range 8).map fun i => v / 256 ^ i % 256

/-- The strings of `argv`, each NUL-terminated, laid out from `a`; returns the
bytes and the address of each string. -/
def layStrings : Nat → List (List Nat) → List Nat × List Nat
  | _, [] => ([], [])
  | a, s :: rest =>
    let (bs, ptrs) := layStrings (a + s.length + 1) rest
    (s ++ [0] ++ bs, a :: ptrs)

/-- The stack at entry, as the kernel builds it for a static executable: at
`sp`, `argc`, then the `argv` pointers and a 0, an empty environment (0), an
empty auxiliary vector (two 0 words), then the strings. The region is writable,
ends at `top`, and is whole pages: the frame, and `stackRoom` bytes below `sp`
for the program's own pushes and calls. Linux gives the stack far more room
(8 MiB by default); a program that needs more than `stackRoom` gets stuck in the
model where it would run on the machine, never the reverse. -/
def stackRoom : Nat := page

def initStack (top : Nat) (argv : List (List Nat)) : Region × Nat :=
  let strLen := (argv.map (·.length + 1)).foldl (· + ·) 0
  let words := 1 + argv.length + 1 + 1 + 2
  let need := words * 8 + strLen
  let size := stackRoom + pages (need + 1)
  let base := top - size
  -- 16-byte aligned sp, `stackRoom` above the bottom of the region
  let sp := base + stackRoom
  let strAt := sp + words * 8
  let (strs, ptrs) := layStrings strAt argv
  let head := le64 argv.length ++ (ptrs.map le64).flatten ++ le64 0 ++ le64 0 ++ le64 0 ++ le64 0
  let bytes := head ++ strs
  (⟨base, List.replicate stackRoom 0 ++ bytes ++
    List.replicate (size - stackRoom - bytes.length) 0, true⟩, sp)

/-! ### Witnesses -/

/-- A kernel with one file `/p` holding `[7, 8, 9]` and an oracle that places
the first mapping at 0x10000 and then cuts the next read to 2 bytes. -/
def kDemo : Kern :=
  ⟨[([47, 112], [7, 8, 9])], [], [], [], [.at 0x10000, .count 2]⟩

def chkMmap : Bool :=
  match syscall kDemo [] 9 0 10 3 0x22 (W - 1) 0 with
  | some (.ret a _ m) => a == 0x10000 && m.length == 1 && m.rd 0x10fff == some 0 && m.rd 0x11000 == none
  | _ => false

/-- `mmap` of 10 bytes gives one zeroed page at the oracle's address. -/
theorem demo_mmap : chkMmap = true := by decide +kernel

def chkNomem : Bool :=
  match syscall { kDemo with oracle := [.nomem] } [] 9 0 10 3 0x22 (W - 1) 0 with
  | some (.ret a _ m) => a == neg ENOMEM && m == []
  | _ => false

/-- The oracle may also refuse: `-ENOMEM`, memory unchanged. -/
theorem demo_nomem : chkNomem = true := by decide +kernel

def chkOpenRead : Bool :=
  let m0 : Mem := [⟨0x10000, [47, 112, 0] ++ List.replicate 13 0, true⟩]
  let k0 : Kern := { kDemo with oracle := [.count 2] }
  match syscall k0 m0 2 0x10000 0 0 0 0 0 with
  | some (.ret fd k1 m1) =>
    fd == 3 && match syscall k1 m1 0 fd 0x10004 3 0 0 0 with
      | some (.ret n _ m2) => n == 2 && Mem.rdN m2 0x10004 3 == some [7, 8, 0]
      | _ => false
  | _ => false

/-- After mapping a page holding the path, `open` finds the file, and a read of
3 bytes cut to 2 by the oracle returns 2 and stores `[7, 8]`. -/
theorem demo_open_read : chkOpenRead = true := by decide +kernel

def chkStack : Bool :=
  let (r, sp) := initStack 0x7ffffffff000 [[115], [47, 112]]
  Mem.rdN [r] sp 8 == some (le64 2) && r.base % page == 0 && sp % 16 == 0 &&
    Mem.rdN [r] (sp + 8) 8 == some (le64 (sp + 56)) &&
    (Mem.wrN [r] (sp - 8) (le64 1)).isSome && (Mem.wrN [r] (sp - stackRoom) (le64 1)).isSome &&
    (Mem.wrN [r] (sp - stackRoom - 8) (le64 1)).isNone

/-- The initial stack holds `argc` and, after it, a pointer to the first string;
there are `stackRoom` writable bytes below `sp`, and none beyond them. -/
theorem demo_stack : chkStack = true := by decide +kernel

/-! ### Memory stays well-formed -/


def shape (m : Mem) : List (Nat × Nat) := m.map fun r => (r.base, r.hi)

theorem find_spec : ∀ (m : Mem) (a i : Nat) {j : Nat} {r : Region}, Mem.find m a i = some (j, r) →
    i ≤ j ∧ m[j - i]? = some r
  | [], _, _, _, _, h => by simp [Mem.find] at h
  | s :: rest, a, i, j, r, h => by
    simp only [Mem.find] at h
    split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h; simp
    · obtain ⟨h1, h2⟩ := find_spec rest a (i + 1) h
      refine ⟨by omega, ?_⟩
      have : j - i = (j - (i + 1)) + 1 := by omega
      rw [this]; simpa using h2

theorem wr_shape {m m' : Mem} {a v : Nat} (h : m.wr a v = some m') : shape m' = shape m := by
  simp only [Mem.wr, bind, Option.bind] at h
  split at h
  · contradiction
  · rename_i p hp
    obtain ⟨i, r⟩ := p
    have hr := (find_spec m a 0 hp).2
    simp only [Nat.sub_zero] at hr
    dsimp only at h
    split at h
    · simp only [Option.some.injEq, pure] at h
      subst h
      simp only [shape, List.map_set, Region.hi, List.length_set]
      apply List.ext_getElem?
      intro j
      rw [List.getElem?_set, List.getElem?_map]
      by_cases hj : i = j
      · subst hj
        obtain ⟨hi, he⟩ := List.getElem?_eq_some_iff.1 hr
        simp [hi, he]
      · simp [hj]
    · contradiction

theorem wrN_shape : ∀ (m : Mem) (a : Nat) (bs : List Nat) {m' : Mem}, Mem.wrN m a bs = some m' → shape m' = shape m
  | m, _, [], m', h => by simp only [Mem.wrN, Option.some.injEq] at h; rw [h]
  | m, a, b :: bs, m', h => by
    simp only [Mem.wrN, bind, Option.bind] at h
    split at h
    · contradiction
    · rename_i m1 h1
      rw [wrN_shape m1 (a + 1) bs h, wr_shape h1]

def wfS (s : List (Nat × Nat)) : Prop :=
  (∀ p ∈ s, p.1 % page = 0) ∧ s.Pairwise (fun p q => p.2 ≤ q.1 ∨ q.2 ≤ p.1)

theorem wf_iff (m : Mem) : m.wf ↔ wfS (shape m) := by
  simp only [Mem.wf, wfS, shape, List.pairwise_map, List.forall_mem_map, Region.hi]

theorem wf_of_shape {m m' : Mem} (h : shape m' = shape m) (hw : m.wf) : m'.wf := by
  rw [wf_iff, h]; exact (wf_iff m).1 hw

theorem free_spec {m : Mem} {a n : Nat} (h : m.free a n = true) : ∀ r ∈ m, a + n ≤ r.base ∨ r.hi ≤ a := by
  intro r hr
  have := List.all_eq_true.1 h r hr
  simpa using this

theorem wf_append {m : Mem} (hw : m.wf) {a : Nat} {d : List Nat} {b : Bool} (ha : a % page = 0)
    (hf : m.free a d.length = true) : Mem.wf (m ++ [(⟨a, d, b⟩ : Region)]) := by
  obtain ⟨h1, h2⟩ := hw
  refine ⟨?_, ?_⟩
  · intro r hr
    rcases List.mem_append.1 hr with hr | hr
    · exact h1 r hr
    · simp at hr; subst hr; exact ha
  · rw [List.pairwise_append]
    refine ⟨h2, by simp, ?_⟩
    intro r hr s hs
    simp at hs; subst hs
    rcases free_spec hf r hr with h | h
    · exact Or.inr (by simpa [Region.hi] using h)
    · exact Or.inl h

theorem wf_erase {m : Mem} (hw : m.wf) (i : Nat) : Mem.wf (m.eraseIdx i) :=
  ⟨fun r hr => hw.1 r (List.mem_of_mem_eraseIdx hr), hw.2.sublist (List.eraseIdx_sublist m i)⟩

theorem place_spec {m : Mem} {c : Option Choice} {n a : Nat} (h : place m c n = some (some a)) :
    a % page = 0 ∧ m.free a n = true := by
  unfold place at h
  split at h
  · simp at h
  · split at h
    · rename_i hc; simp only [Option.some.injEq] at h; subst h; exact ⟨hc.1, hc.2.2.2⟩
    · contradiction
  · contradiction

/-- Every system call, under every oracle, keeps memory well-formed: mappings
stay page-aligned and disjoint. -/
theorem syscall_wf {k : Kern} {m : Mem} {nr a1 a2 a3 a4 a5 a6 : Nat} {r : Nat} {k' : Kern} {m' : Mem}
    (hw : m.wf) (h : syscall k m nr a1 a2 a3 a4 a5 a6 = some (.ret r k' m')) : m'.wf := by
  unfold syscall at h
  split at h
  all_goals (repeat' split at h)
  all_goals (try simp only [bind, Option.bind, pure, Option.some.injEq, SysRes.ret.injEq, reduceCtorEq] at h)
  all_goals (repeat' split at h)
  all_goals (try simp only [Option.some.injEq, SysRes.ret.injEq, reduceCtorEq] at h)
  all_goals (repeat' split at h)
  all_goals (try simp only [Option.some.injEq, SysRes.ret.injEq] at h)
  all_goals first
    | (obtain ⟨-, -, rfl⟩ := h
       first
         | exact hw
         | exact wf_of_shape (wrN_shape _ _ _ (by assumption)) hw
         | exact wf_erase hw _
         | (rename_i hp
            obtain ⟨ha, hf⟩ := place_spec hp
            first
              | exact wf_append hw ha (by simpa using hf)
              | (refine wf_append (wf_erase hw _) ha ?_
                 rw [List.length_append, List.length_replicate]
                 rw [show ∀ x y : Nat, ¬ y < x → x + (y - x) = y from fun x y h => by omega] <;>
                   first | exact hf | assumption)))

end X86Env
