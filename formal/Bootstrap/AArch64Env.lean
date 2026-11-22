/-
AArch64Env: the Linux AArch64 (`aarch64-linux-gnu`) process environment over
which `AArch64Phys` and stage-1 AArch64 binaries (such as `spr1_check_arm64`
and `TMRVoterArm64`) execute.

Mirrors `X86Env` with the Linux `asm-generic` AArch64 syscall ABI:
  - `openat = 56` (`dirfd = AT_FDCWD = -100`, `O_RDONLY = 0`)
  - `close  = 57`
  - `read   = 63` (stdin and open read-only file descriptors, oracle short counts)
  - `write  = 64` (stdout, oracle short counts)
  - `exit   = 93` / `exit_group = 94`
  - `munmap = 215`
  - `mremap = 216` (`MREMAP_MAYMOVE = 1`)
  - `mmap   = 222` (`PROT_READ | PROT_WRITE`, `MAP_PRIVATE | MAP_ANONYMOUS`)

Kernel nondeterminism (mapping placement, `-ENOMEM`, short read/write counts)
is driven by an explicit oracle `List Choice`, with well-formedness predicate
`Choice.wf`. `syscall_wf` proves that every system call under every oracle
preserves page-aligned, pairwise-disjoint memory mappings.

Core Lean 4 only: zero Mathlib imports, zero axioms beyond core Lean.
-/
namespace AArch64Env

def W : Nat := 2 ^ 64
def page : Nat := 4096
def userMax : Nat := 2 ^ 48

/-- `-errno` as a 64-bit two's-complement word. -/
def neg (e : Nat) : Nat := W - e

def ENOENT : Nat := 2
def EBADF  : Nat := 9
def ENOMEM : Nat := 12
def EFAULT : Nat := 14
def EINVAL : Nat := 22

/-- `AT_FDCWD = -100` in 64-bit sign-extended and 32-bit unsigned representations. -/
def AT_FDCWD   : Nat := W - 100
def AT_FDCWD32 : Nat := 2 ^ 32 - 100

/-- Linux `asm-generic` / `aarch64` syscall numbers (passed in `x8`). -/
def NR_openat     : Nat := 56
def NR_close      : Nat := 57
def NR_read       : Nat := 63
def NR_write      : Nat := 64
def NR_exit       : Nat := 93
def NR_exit_group : Nat := 94
def NR_munmap     : Nat := 215
def NR_mremap     : Nat := 216
def NR_mmap       : Nat := 222

def MAP_PRIVATE_ANON : Nat := 0x22
def PROT_RW          : Nat := 3
def MREMAP_MAYMOVE   : Nat := 1

/-- Round up to whole 4 KiB pages. -/
def pages (n : Nat) : Nat := (n + page - 1) / page * page

/-- One virtual memory mapping: its page-aligned base, byte contents, and writability. -/
structure Region where
  base : Nat
  data : List Nat
  rw   : Bool
deriving DecidableEq, Repr

def Region.hi (r : Region) : Nat := r.base + r.data.length

abbrev Mem := List Region

/-- The region containing address `a`, together with its list index. -/
def Mem.find : Mem → Nat → Nat → Option (Nat × Region)
  | [], _, _ => none
  | r :: rest, a, i => if r.base ≤ a ∧ a < r.hi then some (i, r) else Mem.find rest a (i + 1)

def Mem.rd (m : Mem) (a : Nat) : Option Nat := do
  let (_, r) ← Mem.find m a 0
  r.data[a - r.base]?

def Mem.wr (m : Mem) (a v : Nat) : Option Mem := do
  let (i, r) ← Mem.find m a 0
  if r.rw then pure (m.set i { r with data := r.data.set (a - r.base) (v % 256) }) else none

/-- Read `n` bytes starting at address `a`. -/
def Mem.rdN (m : Mem) : Nat → Nat → Option (List Nat)
  | _, 0 => some []
  | a, n + 1 => do
    let b ← m.rd a
    let r ← Mem.rdN m (a + 1) n
    pure (b :: r)

/-- Write bytes `bs` starting at address `a`. -/
def Mem.wrN (m : Mem) : Nat → List Nat → Option Mem
  | _, [] => some m
  | a, b :: bs => do
    let m ← m.wr a b
    Mem.wrN m (a + 1) bs

/-- `[a, a + n)` does not intersect any existing mapping. -/
def Mem.free (m : Mem) (a n : Nat) : Bool :=
  m.all fun r => a + n ≤ r.base || r.hi ≤ a

/-- No two mappings overlap and every mapping is page-aligned. -/
def Mem.wf (m : Mem) : Prop :=
  (∀ r ∈ m, r.base % page = 0) ∧ m.Pairwise (fun r s => r.hi ≤ s.base ∨ s.hi ≤ r.base)

/-- Kernel nondeterministic oracle choice. -/
inductive Choice where
  /-- Place a new or relocated mapping at virtual address `a`. -/
  | at (a : Nat)
  /-- Refuse allocation with `-ENOMEM`. -/
  | nomem
  /-- Transfer `n` bytes in a `read` or `write` syscall. -/
  | count (n : Nat)
deriving DecidableEq, Repr

/-- Well-formedness of a single kernel oracle choice:
mapping addresses are nonzero, page-aligned, and within the 48-bit user VA space;
explicit short counts are strictly positive; `-ENOMEM` is always well-formed. -/
def Choice.wf : Choice → Prop
  | .at a => a % page = 0 ∧ 0 < a ∧ a < userMax
  | .nomem => True
  | .count n => 0 < n

/-- Every choice in an oracle list is well-formed. -/
def OracleWf (o : List Choice) : Prop := ∀ c ∈ o, c.wf

/-- A read-only file table: `(pathBytes, contentBytes)`. -/
abbrev Files := List (List Nat × List Nat)

/-- An open file descriptor: its contents and current byte offset. -/
structure Fd where
  data : List Nat
  pos  : Nat
deriving DecidableEq, Repr

structure Kern where
  files  : Files
  /-- Descriptor `3 + i` is `fds[i]`. -/
  fds    : List (Option Fd)
  stdin  : List Nat
  /-- Bytes written to stdout, in order. -/
  out    : List Nat
  /-- Oracle choices, consumed from the front. -/
  oracle : List Choice
deriving DecidableEq, Repr

/-- Pop the next oracle choice, if any. -/
def Kern.next (k : Kern) : Option Choice × Kern :=
  match k.oracle with
  | [] => (none, k)
  | c :: rest => (some c, { k with oracle := rest })

/-- Decide how many of `want` bytes (with `avail` available) a read or write
moves: uses the oracle's short count when `0 < n ≤ min want avail`, else full. -/
def Kern.count (k : Kern) (want avail : Nat) : Nat × Kern :=
  let full := min want avail
  match k.next with
  | (some (.count n), k') => if 0 < n ∧ n ≤ full then (n, k') else (full, k')
  | (_, _) => (full, k)

/-- Read a NUL-terminated string of at most `n` bytes from `a`. -/
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

/-- Lowest free descriptor slot (offset from fd 3). -/
def freeSlot : List (Option Fd) → Nat
  | [] => 0
  | none :: _ => 0
  | some _ :: r => freeSlot r + 1

def setSlot (fds : List (Option Fd)) (i : Nat) (v : Option Fd) : List (Option Fd) :=
  if i < fds.length then fds.set i v else fds ++ [v]

/-- Result of a Linux AArch64 system call: return value in `x0` with updated
kernel and memory, or process exit. `none` means outside the modelled ABI. -/
inductive SysRes where
  | ret (x0 : Nat) (k : Kern) (m : Mem)
  | exit (status : Nat)
deriving DecidableEq, Repr

/-- Choose a placement for an `n`-byte mapping from oracle choice `c`. -/
def place (m : Mem) (c : Option Choice) (n : Nat) : Option (Option Nat) :=
  match c with
  | some .nomem => some none
  | some (.at a) =>
    if a % page = 0 ∧ 0 < a ∧ a + n ≤ userMax ∧ m.free a n then some (some a) else none
  | _ => none

/-- Linux AArch64 system calls by number (`x8`) and arguments (`x0..x5`). -/
def syscall (k : Kern) (m : Mem) (nr a1 a2 a3 a4 a5 a6 : Nat) : Option SysRes :=
  match nr with
  | 56 => -- openat(dirfd, pathname, flags, mode): AT_FDCWD + O_RDONLY only
    if (a1 ≠ AT_FDCWD ∧ a1 ≠ AT_FDCWD32) ∨ a3 ≠ 0 then none else do
      let name ← m.cstr a2 4096
      match lookup k.files name with
      | some data =>
        let i := freeSlot k.fds
        pure (.ret (3 + i) { k with fds := setSlot k.fds i (some ⟨data, 0⟩) } m)
      | none => pure (.ret (neg ENOENT) k m)
  | 57 => -- close(fd)
    if 3 ≤ a1 ∧ (k.fds[a1 - 3]?).join.isSome then
      pure (.ret 0 { k with fds := k.fds.set (a1 - 3) none } m)
    else pure (.ret (neg EBADF) k m)
  | 63 => -- read(fd, buf, count)
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
  | 64 => -- write(fd, buf, count): stdout only
    if a1 = 1 then
      let (n, k1) := k.count a3 a3
      do let bs ← m.rdN a2 n
         pure (.ret n { k1 with out := k.out ++ bs } m)
    else pure (.ret (neg EBADF) k m)
  | 93 | 94 => -- exit(status) / exit_group(status)
    pure (.exit (a1 % 256))
  | 215 => -- munmap(addr, len): a whole mapping, exact match
    match m.find a1 0 with
    | some (i, r) =>
      if r.base = a1 ∧ r.data.length = pages a2 then pure (.ret 0 k (m.eraseIdx i))
      else none
    | none => none
  | 216 => -- mremap(old, oldlen, newlen, MREMAP_MAYMOVE)
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
  | 222 => -- mmap(0, len, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
    if a1 ≠ 0 ∨ a3 ≠ PROT_RW ∨ a4 ≠ MAP_PRIVATE_ANON ∨ (a5 ≠ W - 1 ∧ a5 ≠ 2 ^ 32 - 1) ∨ a6 ≠ 0 then
      none
    else if a2 = 0 then pure (.ret (neg EINVAL) k m)
    else
      let n := pages a2
      let (c, k1) := k.next
      do match ← place m c n with
         | none => pure (.ret (neg ENOMEM) k1 m)
         | some a => pure (.ret a k1 (m ++ [⟨a, List.replicate n 0, true⟩]))
  | _ => none

/-! ### The initial Linux AArch64 process stack -/

/-- 64-bit little-endian byte encoding. -/
def le64 (v : Nat) : List Nat := (List.range 8).map fun i => v / 256 ^ i % 256

/-- Lay out NUL-terminated `argv` strings from address `a`, returning `(bytes, ptrs)`. -/
def layStrings : Nat → List (List Nat) → List Nat × List Nat
  | _, [] => ([], [])
  | a, s :: rest =>
    let (bs, ptrs) := layStrings (a + s.length + 1) rest
    (s ++ [0] ++ bs, a :: ptrs)

def stackRoom : Nat := page

/-- Build the initial static-ELF process stack ending at `top`:
16-byte aligned `sp` with `stackRoom` writable bytes below `sp`, and
`argc`, `argv` pointers, `0`, empty `envp` (`0`), empty `auxv` (`0, 0`),
and NUL-terminated argument strings at and above `sp`. -/
def initStack (top : Nat) (argv : List (List Nat)) : Region × Nat :=
  let strLen := (argv.map (·.length + 1)).foldl (· + ·) 0
  let words := 1 + argv.length + 1 + 1 + 2
  let need := words * 8 + strLen
  let size := stackRoom + pages (need + 1)
  let base := top - size
  let sp := base + stackRoom
  let strAt := sp + words * 8
  let (strs, ptrs) := layStrings strAt argv
  let head := le64 argv.length ++ (ptrs.map le64).flatten ++ le64 0 ++ le64 0 ++ le64 0 ++ le64 0
  let bytes := head ++ strs
  (⟨base, List.replicate stackRoom 0 ++ bytes ++
    List.replicate (size - stackRoom - bytes.length) 0, true⟩, sp)

/-! ### Memory and oracle well-formedness preservation -/

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

theorem wrN_shape : ∀ (m : Mem) (a : Nat) (bs : List Nat) {m' : Mem},
    Mem.wrN m a bs = some m' → shape m' = shape m
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

theorem free_spec {m : Mem} {a n : Nat} (h : m.free a n = true) :
    ∀ r ∈ m, a + n ≤ r.base ∨ r.hi ≤ a := by
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
    a % page = 0 ∧ 0 < a ∧ a + n ≤ userMax ∧ m.free a n = true := by
  unfold place at h
  split at h
  · simp at h
  · split at h
    · rename_i hc; simp only [Option.some.injEq] at h; subst h; exact hc
    · contradiction
  · contradiction

/-- Any nonempty placement chosen by `place` satisfies `Choice.wf`. -/
theorem place_choice_wf {m : Mem} {c : Choice} {n a : Nat} (hn : 0 < n)
    (h : place m (some c) n = some (some a)) : c.wf := by
  cases c with
  | «at» x =>
    simp only [place] at h
    split at h
    · rename_i hc
      exact ⟨hc.1, hc.2.1, by omega⟩
    · contradiction
  | nomem => simp [place] at h
  | count _ => simp [place] at h

theorem pages_pos {n : Nat} (hn : 0 < n) : 0 < pages n := by
  unfold pages page
  omega

theorem pages_mod (n : Nat) : pages n % page = 0 := by
  unfold pages
  exact Nat.mul_mod_left _ _

theorem kern_next_oracle_wf {k k' : Kern} {c : Option Choice}
    (ho : OracleWf k.oracle) (hn : k.next = (c, k')) : OracleWf k'.oracle := by
  unfold Kern.next at hn
  cases h : k.oracle with
  | nil =>
    rw [h] at hn
    simp only [Prod.mk.injEq] at hn
    obtain ⟨_, rfl⟩ := hn
    exact ho
  | cons x xs =>
    rw [h] at hn
    simp only [Prod.mk.injEq] at hn
    obtain ⟨_, rfl⟩ := hn
    intro y hy
    exact ho y (by rw [h]; exact List.mem_cons_of_mem x hy)

theorem kern_count_oracle_wf {k k' : Kern} {want avail n : Nat}
    (ho : OracleWf k.oracle) (hc : k.count want avail = (n, k')) : OracleWf k'.oracle := by
  unfold Kern.count at hc
  dsimp only at hc
  cases hn : k.next with
  | mk oc k1 =>
    rw [hn] at hc
    cases oc with
    | none =>
      simp only [Prod.mk.injEq] at hc
      obtain ⟨_, rfl⟩ := hc
      exact ho
    | some c =>
      cases c with
      | count m =>
        dsimp only at hc
        split at hc
        · simp only [Prod.mk.injEq] at hc
          obtain ⟨_, rfl⟩ := hc
          exact kern_next_oracle_wf ho hn
        · simp only [Prod.mk.injEq] at hc
          obtain ⟨_, rfl⟩ := hc
          exact kern_next_oracle_wf ho hn
      | «at» _ =>
        simp only [Prod.mk.injEq] at hc
        obtain ⟨_, rfl⟩ := hc
        exact ho
      | nomem =>
        simp only [Prod.mk.injEq] at hc
        obtain ⟨_, rfl⟩ := hc
        exact ho

/-- Every Linux AArch64 system call, under every oracle, preserves page-aligned
pairwise-disjoint memory mappings. -/
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
            obtain ⟨ha, -, -, hf⟩ := place_spec hp
            first
              | exact wf_append hw ha (by simpa using hf)
              | (refine wf_append (wf_erase hw _) ha ?_
                 rw [List.length_append, List.length_replicate]
                 rw [show ∀ x y : Nat, ¬ y < x → x + (y - x) = y from fun x y h => by omega] <;>
                   first | exact hf | assumption)))

/-- The initial process stack is page-aligned, 16-byte SP-aligned, and well-formed
whenever `top` is page-aligned. -/
theorem initStack_wf {top : Nat} {argv : List (List Nat)}
    (htop : top % page = 0) :
    let (r, sp) := initStack top argv
    r.base % page = 0 ∧ sp % 16 = 0 ∧ Mem.wf [r] := by
  dsimp only [initStack]
  let need := (1 + argv.length + 1 + 1 + 2) * 8 + (argv.map (·.length + 1)).foldl (· + ·) 0
  let sz := stackRoom + pages (need + 1)
  have hsz_mod : sz % page = 0 := by
    show (stackRoom + pages (need + 1)) % page = 0
    rw [Nat.add_mod, pages_mod, Nat.add_zero]
    decide
  have hbase : (top - sz) % page = 0 := by
    have hq1 : top = page * (top / page) := by
      have hd := Nat.div_add_mod top page
      rw [htop, Nat.add_zero] at hd
      exact hd.symm
    have hq2 : sz = page * (sz / page) := by
      have hd := Nat.div_add_mod sz page
      rw [hsz_mod, Nat.add_zero] at hd
      exact hd.symm
    have hsub : top - sz = page * (top / page - sz / page) := by
      rw [Nat.mul_sub_left_distrib, ← hq1, ← hq2]
    rw [hsub, Nat.mul_mod_right]
  have hsp : (top - sz + stackRoom) % 16 = 0 := by
    have hpage : (top - sz) % 4096 = 0 := hbase
    have hsr : stackRoom = 4096 := rfl
    omega
  refine ⟨hbase, hsp, ?_, List.pairwise_singleton _ _⟩
  intro r hr
  simp only [List.mem_singleton] at hr
  subst hr
  exact hbase

/-! ### High-address fixture and kernel-checked syscall witnesses -/

/-- Mapping addresses above 4 GiB (4 GiB apart) for oracle fixtures. -/
def highAddrs : List Nat := (List.range 8).map fun i => 0x7f0000000000 + 0x100000000 * i

theorem highAddrs_above_4gib : highAddrs.all (2 ^ 32 ≤ ·) = true := by decide +kernel

theorem highAddrs_choice_wf : ∀ a ∈ highAddrs, Choice.wf (.at a) := by
  intro a ha
  have h : highAddrs.all (fun x => x % page == 0 && 0 < x && x < userMax) = true := by decide +kernel
  have hx := List.all_eq_true.1 h a ha
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hx
  exact ⟨hx.1.1, hx.1.2, hx.2⟩

def kDemo : Kern :=
  ⟨[([47, 112], [7, 8, 9])], [], [], [], [.at 0x10000, .count 2]⟩

def chkMmap : Bool :=
  match syscall kDemo [] NR_mmap 0 10 PROT_RW MAP_PRIVATE_ANON (W - 1) 0 with
  | some (.ret a _ m) => a == 0x10000 && m.length == 1 && m.rd 0x10fff == some 0 && m.rd 0x11000 == none
  | _ => false

theorem demo_mmap : chkMmap = true := by decide +kernel

def chkNomem : Bool :=
  match syscall { kDemo with oracle := [.nomem] } [] NR_mmap 0 10 PROT_RW MAP_PRIVATE_ANON (W - 1) 0 with
  | some (.ret a _ m) => a == neg ENOMEM && m == []
  | _ => false

theorem demo_nomem : chkNomem = true := by decide +kernel

def chkOpenatReadClose : Bool :=
  let m0 : Mem := [⟨0x10000, [47, 112, 0] ++ List.replicate 13 0, true⟩]
  let k0 : Kern := { kDemo with oracle := [.count 2] }
  match syscall k0 m0 NR_openat AT_FDCWD 0x10000 0 0 0 0 with
  | some (.ret fd k1 m1) =>
    fd == 3 && match syscall k1 m1 NR_read fd 0x10004 3 0 0 0 with
      | some (.ret n k2 m2) =>
        n == 2 && Mem.rdN m2 0x10004 3 == some [7, 8, 0] &&
          match syscall k2 m2 NR_close fd 0 0 0 0 0 with
          | some (.ret 0 _ _) => true
          | _ => false
      | _ => false
  | _ => false

theorem demo_openat_read_close : chkOpenatReadClose = true := by decide +kernel

def chkMremapMunmap : Bool :=
  let k0 : Kern := { kDemo with oracle := [.at 0x10000, .at 0x20000] }
  match syscall k0 [] NR_mmap 0 4096 PROT_RW MAP_PRIVATE_ANON (W - 1) 0 with
  | some (.ret a1 k1 m1) =>
    a1 == 0x10000 && match syscall k1 m1 NR_mremap a1 4096 8192 MREMAP_MAYMOVE 0 0 with
      | some (.ret a2 k2 m2) =>
        a2 == 0x20000 && m2.rd 0x21fff == some 0 && m2.rd 0x10000 == none &&
          match syscall k2 m2 NR_munmap a2 8192 0 0 0 0 with
          | some (.ret 0 _ m3) => m3 == []
          | _ => false
      | _ => false
  | _ => false

theorem demo_mremap_munmap : chkMremapMunmap = true := by decide +kernel

def chkExitGroup : Bool :=
  syscall kDemo [] NR_exit 3 0 0 0 0 0 == some (.exit 3) &&
  syscall kDemo [] NR_exit_group 259 0 0 0 0 0 == some (.exit 3)

theorem demo_exit_group : chkExitGroup = true := by decide +kernel

def chkStack : Bool :=
  let (r, sp) := initStack 0x7ffffffff000 [[115], [47, 112]]
  Mem.rdN [r] sp 8 == some (le64 2) && r.base % page == 0 && sp % 16 == 0 &&
    Mem.rdN [r] (sp + 8) 8 == some (le64 (sp + 56)) &&
    (Mem.wrN [r] (sp - 8) (le64 1)).isSome && (Mem.wrN [r] (sp - stackRoom) (le64 1)).isSome &&
    (Mem.wrN [r] (sp - stackRoom - 8) (le64 1)).isNone

theorem demo_stack : chkStack = true := by decide +kernel

end AArch64Env
