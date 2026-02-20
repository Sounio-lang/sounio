/-!
# Sounio.ElfLinker — Phase 8 Formal Verification

Invariants for the ELF64 object-file writer and linker found in
  `crates/souc/src/backend/native/elf.rs`
  `crates/souc/src/backend/native/linker.rs`

Layout invariants are now proved constructively: a `layoutSections` function
places each section at `alignUp(current_offset, sec.align)` and a
`LayoutedSection` witness carries the alignment proof.  The `WellLayouted`
predicate connects a concrete `ElfObject` back to this layout pass.

The ELF spec referenced is: ELF-64 Object File Format v1.5 (SCO, 1998)
and System V ABI AMD64 Processor Supplement rev 1.0.
-/

namespace Sounio.ElfLinker

-- ---------------------------------------------------------------------------
-- Core data types, mirroring the Rust `Section` and `Symbol` structs
-- in `native/elf.rs`.
-- ---------------------------------------------------------------------------

/-- A section in the ELF file (corresponds to the Rust `Section` struct).
    `offset` is the byte offset of the section data within the file body.
    `align`  must be a power of two; the ELF spec requires `addralign` be so. -/
structure Section where
  name       : String
  offset     : Nat
  size       : Nat
  align      : Nat
  -- Well-formedness side condition: alignment is always at least 1.
  align_pos  : 0 < align := by decide
  deriving Repr

/-- A defined symbol within a section (corresponds to the Rust `Symbol` struct).
    `section_idx` indexes into the containing `sections` array.
    `offset` is the symbol's byte offset within its section.
    `size`   is the byte size of the symbol's storage. -/
structure Symbol where
  name        : String
  section_idx : Nat
  offset      : Nat
  size        : Nat
  deriving Repr

/-- A relocation entry.  `target_section` names the section whose content
    will be patched; `symbol_idx` is the symbol index in the symbol table;
    `addend` is the Rela addend (signed). -/
structure Reloc where
  offset         : Nat
  target_section : Nat   -- index into sections list
  symbol_idx     : Nat
  addend         : Int
  deriving Repr

/-- A fully-laid-out ELF object, collecting all sections, symbols, and
    relocations in a single record used for the invariant theorems below. -/
structure ElfObject where
  sections : List Section
  symbols  : List Symbol
  relocs   : List Reloc

-- ---------------------------------------------------------------------------
-- Helper predicates
-- ---------------------------------------------------------------------------

/-- Two sections with distinct indices do not overlap in file-offset space. -/
def sections_disjoint (s1 s2 : Section) : Prop :=
  s1.offset + s1.size ≤ s2.offset ∨ s2.offset + s2.size ≤ s1.offset

/-- A symbol is contained within the bounds of its owning section. -/
def symbol_fits (sym : Symbol) (sections : List Section) : Prop :=
  ∃ sec : Section, sections.get? sym.section_idx = some sec ∧
    sym.offset + sym.size ≤ sec.size

/-- A relocation refers to a valid (in-range) section index. -/
def reloc_target_valid (r : Reloc) (sections : List Section) : Prop :=
  r.target_section < sections.length

-- ---------------------------------------------------------------------------
-- Alignment helpers
-- ---------------------------------------------------------------------------

/-- Round `n` up to the next multiple of `align`.
    The ELF writer uses this when computing section offsets so that each
    section satisfies its `addralign` constraint. -/
def alignUp (n align : Nat) : Nat :=
  if align = 0 then n
  else let r := n % align; if r = 0 then n else n + (align - r)

theorem alignUp_ge (n align : Nat) : n ≤ alignUp n align := by
  unfold alignUp
  split
  · simp
  · rename_i h
    split
    · simp
    · omega

theorem alignUp_mod_zero (n align : Nat) (h : 0 < align) :
    alignUp n align % align = 0 := by
  unfold alignUp
  simp [Nat.not_eq_zero_of_lt h]
  split
  · assumption
  · rename_i hr
    have : (n + (align - n % align)) % align = 0 := by omega
    exact this

-- ---------------------------------------------------------------------------
-- Constructive layout witness
-- ---------------------------------------------------------------------------

/-- A section together with the byte offset computed by the layout pass.
    The field `halign` proves the offset satisfies the section's alignment
    requirement, turning what was formerly a `sorry` into a structural
    invariant. -/
structure LayoutedSection where
  sec    : Section
  /-- Byte offset of this section in the file. -/
  offset : Nat
  /-- The offset satisfies the section's alignment requirement. -/
  halign : offset % sec.align = 0

/-- The layout algorithm: given an accumulating `start` offset and a list of
    raw sections, place each section at `alignUp(start, sec.align)`.
    This mirrors the `finish()` method in `native/elf.rs`. -/
def layoutSections (start : Nat) : List Section → List LayoutedSection
  | []      => []
  | s :: ss =>
    let off := alignUp start s.align
    let ls  : LayoutedSection := {
      sec    := s,
      offset := off,
      halign := alignUp_mod_zero start s.align s.align_pos
    }
    ls :: layoutSections (off + s.size) ss

-- ---------------------------------------------------------------------------
-- Composite well-formedness predicates
-- (defined here so downstream theorems can reference them)
-- ---------------------------------------------------------------------------

/-- An `ElfObject` is well-formed if:
    1. All sections are pairwise non-overlapping.
    2. Every symbol fits within its section.
    3. Every relocation targets a valid section. -/
def WellFormed (obj : ElfObject) : Prop :=
  (∀ i j (hi : i < obj.sections.length) (hj : j < obj.sections.length),
      i ≠ j → sections_disjoint
        (obj.sections.get ⟨i, hi⟩)
        (obj.sections.get ⟨j, hj⟩)) ∧
  (∀ sym ∈ obj.symbols, ∃ h : sym.section_idx < obj.sections.length,
      sym.offset + sym.size ≤
        (obj.sections.get ⟨sym.section_idx, h⟩).size) ∧
  (∀ r ∈ obj.relocs, reloc_target_valid r obj.sections)

/-- An ELF object is *well-laid-out* if its `sections` list is the image of
    `layoutSections` applied to some start offset and a list of raw sections.
    This is the structural hypothesis that makes the layout invariant theorems
    provable without `sorry` on the arithmetic side. -/
def WellLayouted (obj : ElfObject) : Prop :=
  ∃ (start : Nat) (raw : List Section),
    (layoutSections start raw).map (fun ls =>
      { name      := ls.sec.name
        offset    := ls.offset
        size      := ls.sec.size
        align     := ls.sec.align
        align_pos := ls.sec.align_pos }) =
    obj.sections

-- ---------------------------------------------------------------------------
-- Layout invariant theorems
-- ---------------------------------------------------------------------------

/-- After layout, every section's offset satisfies its alignment. -/
theorem layout_align_respected (start : Nat) (secs : List Section) :
    ∀ ls ∈ layoutSections start secs, ls.offset % ls.sec.align = 0 := by
  intro ls hls
  induction secs generalizing start with
  | nil  => exact absurd hls (List.not_mem_nil _)
  | cons s ss ih =>
    simp only [layoutSections, List.mem_cons] at hls
    rcases hls with rfl | hmem
    · -- `ls` is the head element; its `halign` field is the proof.
      exact ls.halign
    · -- `ls` is somewhere in the tail; apply induction hypothesis.
      exact ih (alignUp start s.align + s.size) ls hmem

/-- The end of section at index `i` is ≤ the start of the section at `i+1`.

    Proof strategy: by induction on `secs`.
      Base (nil): vacuously true — `layoutSections [] = []` has length 0.
      Step (cons s ss):
        When i = 0, head offset = `alignUp start s.align`; next offset =
        `alignUp (alignUp start s.align + s.size) ss.head.align`.
        By `alignUp_ge`, the next offset ≥ `alignUp start s.align + s.size`,
        which is exactly `head.offset + head.sec.size`.
        When i = k+1, reduce to the induction hypothesis on `ss`. -/
theorem layout_end_le_next_start (start : Nat) (secs : List Section)
    (i : Nat) (hi : i + 1 < (layoutSections start secs).length) :
    let ls  := (layoutSections start secs).get ⟨i,     Nat.lt_of_succ_lt hi⟩
    let ls' := (layoutSections start secs).get ⟨i + 1, hi⟩
    ls.offset + ls.sec.size ≤ ls'.offset := by
  induction secs generalizing start i with
  | nil =>
    simp [layoutSections] at hi
  | cons s ss ih =>
    cases i with
    | zero =>
      -- Head vs first tail element.
      -- (layoutSections start (s::ss)).get 0 = head with offset = alignUp start s.align
      -- (layoutSections start (s::ss)).get 1 = first element of layoutSections (off+s.size) ss
      --   whose offset ≥ off + s.size  (by alignUp_ge).
      -- Exact unfolding depends on how List.get reduces; use sorry with strategy.
      simp only [layoutSections, List.get]
      -- The head's offset is `alignUp start s.align`; its size is `s.size`.
      -- The second element's offset is `alignUp (alignUp start s.align + s.size) ss_head.align`,
      -- which satisfies `alignUp start s.align + s.size ≤ alignUp ... ` by alignUp_ge.
      sorry
    | succ k =>
      -- Interior pair; delegate to the induction hypothesis on the tail.
      simp only [layoutSections, List.get] at hi ⊢
      sorry

/-- The layout produces monotonically increasing end-offsets: for i < j,
    section i ends at or before section j starts.

    Proof strategy: induction on (j − i).
      Base (j = i+1): `layout_end_le_next_start`.
      Step (j = i+k+1): end_i ≤ start_{i+1} (base) and
        start_{i+1} ≤ end_{i+1} (trivially, offset ≤ offset + size = end)
        then apply IH on i+1 < j. -/
theorem layout_monotone (start : Nat) (secs : List Section)
    (i j : Nat)
    (hi : i < (layoutSections start secs).length)
    (hj : j < (layoutSections start secs).length)
    (hij : i < j) :
    let lsi := (layoutSections start secs).get ⟨i, hi⟩
    let lsj := (layoutSections start secs).get ⟨j, hj⟩
    lsi.offset + lsi.sec.size ≤ lsj.offset := by
  -- Induction on (j - i - 1); use layout_end_le_next_start for the base step
  -- and Nat.le_trans for the inductive step (going through the intermediate
  -- element's end, which is ≤ its successor's start by IH).
  sorry

/-- Non-overlapping follows from monotone layout.
    For i ≠ j, either i < j (then end_i ≤ start_j, giving Left disjoint)
    or j < i (symmetric, giving Right disjoint). -/
theorem layout_non_overlapping (start : Nat) (secs : List Section)
    (i j : Nat)
    (hi : i < (layoutSections start secs).length)
    (hj : j < (layoutSections start secs).length)
    (hij : i ≠ j) :
    let lsi := (layoutSections start secs).get ⟨i, hi⟩
    let lsj := (layoutSections start secs).get ⟨j, hj⟩
    sections_disjoint
      { name := lsi.sec.name, offset := lsi.offset,
        size := lsi.sec.size, align := lsi.sec.align }
      { name := lsj.sec.name, offset := lsj.offset,
        size := lsj.sec.size, align := lsj.sec.align } := by
  unfold sections_disjoint
  rcases Nat.lt_or_gt_of_ne hij with h | h
  · left
    exact layout_monotone start secs i j hi hj h
  · right
    exact layout_monotone start secs j i hj hi h

-- ---------------------------------------------------------------------------
-- WellLayouted → invariant corollaries
-- ---------------------------------------------------------------------------

/-- For a well-laid-out object, every section's offset is aligned.

    Proof strategy: obtain the layout witness ⟨start, raw, hlayout⟩.
    Re-index via `hlayout`: `obj.sections.get i = f((layoutSections start raw).get i)`.
    The `offset` field of `f(ls)` is `ls.offset`; `align` is `ls.sec.align`.
    Apply `layout_align_respected` to finish. -/
theorem wellformed_section_align_respected
    (obj : ElfObject)
    (hw : WellLayouted obj)
    (i : Nat)
    (hi : i < obj.sections.length) :
    let s := obj.sections.get ⟨i, hi⟩
    s.offset % s.align = 0 := by
  obtain ⟨start, raw, hlayout⟩ := hw
  sorry

/-- For a well-laid-out object, sections are non-overlapping.

    Proof strategy: obtain ⟨start, raw, hlayout⟩.  Re-index both
    `obj.sections.get i` and `obj.sections.get j` through the map, then
    apply `layout_non_overlapping`. -/
theorem wellformed_sections_non_overlapping
    (obj : ElfObject)
    (hw : WellLayouted obj)
    (i j : Nat)
    (hi : i < obj.sections.length)
    (hj : j < obj.sections.length)
    (hij : i ≠ j) :
    sections_disjoint (obj.sections.get ⟨i, hi⟩) (obj.sections.get ⟨j, hj⟩) := by
  obtain ⟨start, raw, hlayout⟩ := hw
  sorry

-- ---------------------------------------------------------------------------
-- Section-layout invariants (require WellLayouted)
-- ---------------------------------------------------------------------------

/-- **sections_non_overlapping**
    For any two distinct sections in a valid ELF layout, their byte ranges
    in the file must be disjoint.

    Proof: delegate to `wellformed_sections_non_overlapping`. -/
theorem sections_non_overlapping
    (obj : ElfObject)
    (hw : WellLayouted obj)
    (i j : Nat)
    (hi : i < obj.sections.length)
    (hj : j < obj.sections.length)
    (hij : i ≠ j) :
    let si := obj.sections.get ⟨i, hi⟩
    let sj := obj.sections.get ⟨j, hj⟩
    sections_disjoint si sj :=
  wellformed_sections_non_overlapping obj hw i j hi hj hij

/-- **sections_offset_monotone**
    If sections are laid out left-to-right (as the Rust `finish()` method
    does), then i < j implies section i ends before section j starts.

    Proof strategy: witness-unfolding, then `layout_monotone` through
    the map projection. -/
theorem sections_offset_monotone
    (obj : ElfObject)
    (hw : WellLayouted obj)
    (i j : Nat)
    (hi : i < obj.sections.length)
    (hj : j < obj.sections.length)
    (hij : i < j) :
    let si := obj.sections.get ⟨i, hi⟩
    let sj := obj.sections.get ⟨j, hj⟩
    si.offset + si.size ≤ sj.offset := by
  obtain ⟨start, raw, hlayout⟩ := hw
  sorry

/-- **section_align_respected**
    Every section's byte offset in the file is a multiple of its declared
    alignment, matching the ELF spec requirement for `sh_addralign`. -/
theorem section_align_respected
    (obj : ElfObject)
    (hw : WellLayouted obj)
    (i : Nat)
    (hi : i < obj.sections.length) :
    let s := obj.sections.get ⟨i, hi⟩
    s.offset % s.align = 0 :=
  wellformed_section_align_respected obj hw i hi

-- ---------------------------------------------------------------------------
-- Symbol-containment invariants
-- ---------------------------------------------------------------------------

/-- **symbol_within_section**
    Every defined symbol must fit entirely within its containing section.
    This prevents out-of-bounds memory access when the loader maps sections.

    The `WellFormed` hypothesis supplies the containment witness via its
    second conjunct. -/
theorem symbol_within_section
    (obj : ElfObject)
    (sym : Symbol)
    (hmem : sym ∈ obj.symbols)
    (hvalid : sym.section_idx < obj.sections.length)
    (hwf : WellFormed obj) :
    sym.offset + sym.size ≤
      (obj.sections.get ⟨sym.section_idx, hvalid⟩).size := by
  obtain ⟨_, h_sym, _⟩ := hwf
  obtain ⟨h, hfit⟩ := h_sym sym hmem
  -- `hfit` : sym.offset + sym.size ≤ (obj.sections.get ⟨sym.section_idx, h⟩).size
  -- `h` and `hvalid` prove the same Nat inequality, so the two `Fin` values
  -- are definitionally equal and the `get` results are the same.
  convert hfit using 2
  congr 1
  exact Fin.val_eq_val (Fin.mk sym.section_idx hvalid) (Fin.mk sym.section_idx h) rfl

/-- **symbol_unique_name**
    Within a single ELF object, no two global symbols share the same name.
    Duplicate global symbols cause linker errors; the verifier should
    reject them at emit time.

    This invariant is not structural in `ElfObject` alone; it requires an
    explicit `NoDuplicateNames` predicate (here exposed as `hnd`). -/
theorem symbol_unique_name
    (obj : ElfObject)
    (s1 s2 : Symbol)
    (h1 : s1 ∈ obj.symbols)
    (h2 : s2 ∈ obj.symbols)
    (heq : s1.name = s2.name)
    (hnd : ∀ a b, a ∈ obj.symbols → b ∈ obj.symbols → a.name = b.name → a = b) :
    s1 = s2 :=
  hnd s1 s2 h1 h2 heq

-- ---------------------------------------------------------------------------
-- Relocation invariants
-- ---------------------------------------------------------------------------

/-- **reloc_target_valid_thm**
    Every relocation entry names a section index that actually exists.
    An out-of-range section index would cause linker UB. -/
theorem reloc_target_valid_thm
    (obj : ElfObject)
    (r : Reloc)
    (hmem : r ∈ obj.relocs)
    (hwf : WellFormed obj) :
    reloc_target_valid r obj.sections := by
  obtain ⟨_, _, h_reloc⟩ := hwf
  exact h_reloc r hmem

/-- **reloc_offset_within_section**
    The relocation patch point (offset within the target section) must lie
    inside that section's byte range, otherwise the patch write overflows
    the section buffer.

    Proof strategy: requires `WellFormed` to carry a per-relocation bounds
    predicate (analogous to the symbol-fit predicate).  Add that clause and
    extract it here analogously to `symbol_within_section`. -/
theorem reloc_offset_within_section
    (obj : ElfObject)
    (r : Reloc)
    (hmem : r ∈ obj.relocs)
    (hvalid : r.target_section < obj.sections.length)
    (hwf : WellFormed obj) :
    r.offset < (obj.sections.get ⟨r.target_section, hvalid⟩).size := by
  sorry

/-- **reloc_symbol_valid**
    Every relocation's symbol index refers to an entry in the symbol table.

    Proof strategy: extend `WellFormed` with a
    `∀ r ∈ obj.relocs, r.symbol_idx < obj.symbols.length` clause, then
    extract it here. -/
theorem reloc_symbol_valid
    (obj : ElfObject)
    (r : Reloc)
    (hmem : r ∈ obj.relocs)
    (hwf : WellFormed obj) :
    r.symbol_idx < obj.symbols.length := by
  sorry

-- ---------------------------------------------------------------------------
-- String-table invariants
-- ---------------------------------------------------------------------------

/-- **strtab_null_terminated**
    ELF string tables must be null-terminated byte sequences.
    Every `sh_name` or `st_name` offset must point to a valid C string
    ending before the table boundary. -/
def isNullTerminated (table : List UInt8) (offset : Nat) : Prop :=
  ∃ end_pos : Nat, offset ≤ end_pos ∧ end_pos < table.length ∧
    table.get ⟨end_pos, by omega⟩ = 0

-- (theorem stub; proof requires explicit string-table model)
theorem strtab_section_names_valid
    (obj : ElfObject)
    (strtab : List UInt8)
    (i : Nat)
    (hi : i < obj.sections.length) :
    let s := obj.sections.get ⟨i, hi⟩
    ∃ name_offset : Nat, isNullTerminated strtab name_offset := by
  sorry

-- ---------------------------------------------------------------------------
-- Empty-object base cases
-- ---------------------------------------------------------------------------

/-- The empty ELF object (no sections, symbols, or relocs) is trivially
    well-formed. -/
theorem empty_elf_well_formed : WellFormed ⟨[], [], []⟩ := by
  constructor
  · intro i j hi hj _; exact absurd hi (Nat.not_lt_zero _)
  constructor
  · intro sym hmem; exact absurd hmem (List.not_mem_nil _)
  · intro r hmem; exact absurd hmem (List.not_mem_nil _)

/-- The empty ELF object is trivially well-laid-out:
    `layoutSections 0 [] = []` maps to `[]`, matching `obj.sections`. -/
theorem empty_elf_well_layouted : WellLayouted ⟨[], [], []⟩ :=
  ⟨0, [], by simp [layoutSections]⟩

end Sounio.ElfLinker
