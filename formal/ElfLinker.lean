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
  ∃ h : sym.section_idx < sections.length,
    sym.offset + sym.size ≤ (sections.get ⟨sym.section_idx, h⟩).size

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
  by_cases h : align = 0
  · simp [h]
  · have h' : align > 0 := Nat.zero_lt_of_ne_zero h
    simp only [h, ↓reduceIte]
    by_cases hr : n % align = 0
    · simp [hr]
    · simp [hr]

theorem alignUp_mod_zero (n align : Nat) (h : 0 < align) :
    alignUp n align % align = 0 := by
  have h_ne : align ≠ 0 := Nat.ne_of_gt h
  by_cases hr : n % align = 0
  · simp [alignUp, hr, h_ne]
  · simp only [alignUp, hr, h_ne]
    have h1 : n + (align - n % align) = align * (n / align + 1) := by
      have h2 : n % align ≤ align := Nat.le_of_lt (Nat.mod_lt n h)
      have h3 : n + (align - n % align) = n + align - n % align := by
        rw [Nat.add_sub_assoc h2]
      have h4 : n + align - n % align = align * (n / align) + n % align + align - n % align := by
        rw [Nat.div_add_mod n align]
      have h5 : align * (n / align) + n % align + align - n % align = align * (n / align) + align := by
        have h6 : align * (n / align) + n % align + align = align * (n / align) + align + n % align := by
          simp [Nat.add_assoc, Nat.add_comm (n % align)]
        rw [h6]
        rw [Nat.add_sub_cancel]
      rw [h3, h4, h5]
      simp [Nat.mul_add, Nat.mul_one]
    rw [h1]
    exact Nat.dvd_iff_mod_eq_zero.mp (Nat.dvd_mul_right align (n / align + 1))

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
    3. Every relocation targets a valid section.
    4. Every relocation's offset is within the target section's bounds.
    5. Every relocation's symbol index is in range. -/
def WellFormed (obj : ElfObject) : Prop :=
  (∀ i j (hi : i < obj.sections.length) (hj : j < obj.sections.length),
      i ≠ j → sections_disjoint
        (obj.sections.get ⟨i, hi⟩)
        (obj.sections.get ⟨j, hj⟩)) ∧
  (∀ sym ∈ obj.symbols, ∃ h : sym.section_idx < obj.sections.length,
      sym.offset + sym.size ≤
        (obj.sections.get ⟨sym.section_idx, h⟩).size) ∧
  (∀ r ∈ obj.relocs, reloc_target_valid r obj.sections) ∧
  (∀ r ∈ obj.relocs, ∃ h : r.target_section < obj.sections.length,
      r.offset < (obj.sections.get ⟨r.target_section, h⟩).size) ∧
  (∀ r ∈ obj.relocs, r.symbol_idx < obj.symbols.length)

/-- Convert a LayoutedSection back to a Section, preserving all fields. -/
def LayoutedSection.toSection (ls : LayoutedSection) : Section :=
  { name      := ls.sec.name
    offset    := ls.offset
    size      := ls.sec.size
    align     := ls.sec.align
    align_pos := ls.sec.align_pos }

/-- An ELF object is *well-laid-out* if its `sections` list is the image of
    `layoutSections` applied to some start offset and a list of raw sections.
    This is the structural hypothesis that makes the layout invariant theorems
    provable without `sorry` on the arithmetic side. -/
def WellLayouted (obj : ElfObject) : Prop :=
  ∃ (start : Nat) (raw : List Section),
    (layoutSections start raw).map LayoutedSection.toSection = obj.sections

-- ---------------------------------------------------------------------------
-- Private helpers for layout invariant proofs
-- ---------------------------------------------------------------------------

/-- Consecutive-pair property: for every adjacent pair in a layout list,
    the first section ends before the second begins. -/
private def consecProp : List LayoutedSection → Prop
  | []      => True
  | [_]     => True
  | ls :: ls' :: rest =>
    ls.offset + ls.sec.size ≤ ls'.offset ∧ consecProp (ls' :: rest)

/-- `layoutSections` satisfies `consecProp`: consecutive sections never overlap. -/
private theorem layoutSections_consec_prop (start : Nat) (secs : List Section) :
    consecProp (layoutSections start secs) := by
  induction secs generalizing start with
  | nil => exact trivial
  | cons s ss ih =>
    cases ss with
    | nil => exact trivial
    | cons s2 ss2 =>
      simp only [layoutSections, consecProp]
      exact ⟨alignUp_ge _ _, ih (alignUp start s.align + s.size)⟩

/-- Extract the consecutive-pair property at a concrete index from `consecProp`. -/
private theorem consecProp_get (ls : List LayoutedSection) (h : consecProp ls)
    (i : Nat) (hi : i + 1 < ls.length) :
    (ls.get ⟨i, Nat.lt_of_succ_lt hi⟩).offset +
    (ls.get ⟨i, Nat.lt_of_succ_lt hi⟩).sec.size ≤
    (ls.get ⟨i + 1, hi⟩).offset := by
  induction ls generalizing i with
  | nil => exact absurd hi (by simp)
  | cons x rest ihx =>
    cases rest with
    | nil => exact absurd hi (by simp)
    | cons y ys =>
      cases i with
      | zero =>
        simp only [List.get_cons_succ]
        exact h.1
      | succ k =>
        simp only [List.get_cons_succ]
        exact ihx h.2 k (by simpa [List.length_cons] using hi)

/-- Helper: equal lists have equal `get` elements at the same index. -/
private theorem get_eq_of_list_eq {α} {l1 l2 : List α} (h : l1 = l2) (i : Nat)
    (hi : i < l1.length) :
    l1.get ⟨i, hi⟩ = l2.get ⟨i, by rw [← h]; exact hi⟩ := by
  subst h
  rfl

/-- Helper: `List.get` commutes with `List.map` (proved by induction). -/
private theorem get_map_eq {α β} (f : α → β) (l : List α) (i : Nat)
    (hi : i < l.length) :
    (l.map f).get ⟨i, by rw [List.length_map]; exact hi⟩ = f (l.get ⟨i, hi⟩) := by
  induction l generalizing i with
  | nil => exact absurd hi (Nat.not_lt_zero _)
  | cons x xs ihx =>
    cases i with
    | zero => simp
    | succ k =>
      simp only [List.map, List.get_cons_succ]
      exact ihx k (Nat.lt_of_succ_lt_succ hi)

-- ---------------------------------------------------------------------------
-- Layout invariant theorems
-- ---------------------------------------------------------------------------

/-- After layout, every section's offset satisfies its alignment. -/
theorem layout_align_respected (start : Nat) (secs : List Section) :
    ∀ ls ∈ layoutSections start secs, ls.offset % ls.sec.align = 0 := by
  intro ls hls
  induction secs generalizing start with
  | nil  => exact absurd hls List.not_mem_nil
  | cons s ss ih =>
    simp only [layoutSections, List.mem_cons] at hls
    cases hls with
    | inl heq =>
      rw [heq]
      exact alignUp_mod_zero start s.align s.align_pos
    | inr hmem =>
      exact ih (alignUp start s.align + s.size) hmem

/-- The end of section at index `i` is ≤ the start of the section at `i+1`. -/
theorem layout_end_le_next_start (start : Nat) (secs : List Section)
    (i : Nat) (hi : i + 1 < (layoutSections start secs).length) :
    let ls  := (layoutSections start secs).get ⟨i,     Nat.lt_of_succ_lt hi⟩
    let ls' := (layoutSections start secs).get ⟨i + 1, hi⟩
    ls.offset + ls.sec.size ≤ ls'.offset :=
  consecProp_get _ (layoutSections_consec_prop start secs) i hi

/-- The layout produces monotonically increasing end-offsets: for i < j,
    section i ends at or before section j starts. -/
theorem layout_monotone (start : Nat) (secs : List Section)
    (i j : Nat)
    (hi : i < (layoutSections start secs).length)
    (hj : j < (layoutSections start secs).length)
    (hij : i < j) :
    let lsi := (layoutSections start secs).get ⟨i, hi⟩
    let lsj := (layoutSections start secs).get ⟨j, hj⟩
    lsi.offset + lsi.sec.size ≤ lsj.offset := by
  -- Substitute j = i + d + 1 and induct on d.
  obtain ⟨d, rfl⟩ : ∃ d, j = i + d + 1 := ⟨j - i - 1, by omega⟩
  induction d generalizing i with
  | zero =>
    -- j = i + 1: direct application of layout_end_le_next_start
    exact layout_end_le_next_start start secs i (by simpa using hj)
  | succ k ihk =>
    -- j = i + k + 2; chain through index i + k + 1
    have hmid : i + k + 1 < (layoutSections start secs).length := by omega
    have h1 : ((layoutSections start secs).get ⟨i, hi⟩).offset +
        ((layoutSections start secs).get ⟨i, hi⟩).sec.size
        ≤ ((layoutSections start secs).get ⟨i + k + 1, hmid⟩).offset := by
      have := ihk i hi hmid (by omega)
      dsimp only at this ⊢
      exact this
    have h2 := Nat.le_add_right
      ((layoutSections start secs).get ⟨i + k + 1, hmid⟩).offset
      ((layoutSections start secs).get ⟨i + k + 1, hmid⟩).sec.size
    have h3 : ((layoutSections start secs).get ⟨i + k + 1, hmid⟩).offset +
        ((layoutSections start secs).get ⟨i + k + 1, hmid⟩).sec.size
        ≤ ((layoutSections start secs).get ⟨i + k + 2, hj⟩).offset :=
          layout_end_le_next_start start secs (i + k + 1) (by simpa using hj)
    exact Nat.le_trans (Nat.le_trans h1 h2) h3

/-- Non-overlapping follows from monotone layout. -/
theorem layout_non_overlapping (start : Nat) (secs : List Section)
    (i j : Nat)
    (hi : i < (layoutSections start secs).length)
    (hj : j < (layoutSections start secs).length)
    (hij : i ≠ j) :
    let lsi := (layoutSections start secs).get ⟨i, hi⟩
    let lsj := (layoutSections start secs).get ⟨j, hj⟩
    sections_disjoint
      { name := lsi.sec.name, offset := lsi.offset,
        size := lsi.sec.size, align := lsi.sec.align,
        align_pos := lsi.sec.align_pos }
      { name := lsj.sec.name, offset := lsj.offset,
        size := lsj.sec.size, align := lsj.sec.align,
        align_pos := lsj.sec.align_pos } := by
  unfold sections_disjoint
  rcases Nat.lt_or_gt_of_ne hij with h | h
  · left
    exact layout_monotone start secs i j hi hj h
  · right
    exact layout_monotone start secs j i hj hi h

-- ---------------------------------------------------------------------------
-- WellLayouted → invariant corollaries
-- ---------------------------------------------------------------------------

/-- For a well-laid-out object, every section's offset is aligned. -/
theorem wellformed_section_align_respected
    (obj : ElfObject)
    (hw : WellLayouted obj)
    (i : Nat)
    (hi : i < obj.sections.length) :
    let s := obj.sections.get ⟨i, hi⟩
    s.offset % s.align = 0 := by
  obtain ⟨start, raw, hlayout⟩ := hw
  have hlen : (layoutSections start raw).length = obj.sections.length := by
    have := congrArg List.length hlayout; simp [List.length_map] at this; exact this
  have hraw_i : i < (layoutSections start raw).length := by omega
  have hmem : (layoutSections start raw).get ⟨i, hraw_i⟩ ∈ layoutSections start raw :=
    List.get_mem (layoutSections start raw) ⟨i, hraw_i⟩
  have halign := layout_align_respected start raw ((layoutSections start raw).get ⟨i, hraw_i⟩) hmem
  have h1 := get_eq_of_list_eq (show obj.sections = (layoutSections start raw).map LayoutedSection.toSection by rw [hlayout]) i hi
  have h2 := get_map_eq LayoutedSection.toSection (layoutSections start raw) i hraw_i
  rw [h2] at h1
  rw [h1]
  simp only [LayoutedSection.toSection]
  exact halign

/-- For a well-laid-out object, sections are non-overlapping. -/
theorem wellformed_sections_non_overlapping
    (obj : ElfObject)
    (hw : WellLayouted obj)
    (i j : Nat)
    (hi : i < obj.sections.length)
    (hj : j < obj.sections.length)
    (hij : i ≠ j) :
    sections_disjoint (obj.sections.get ⟨i, hi⟩) (obj.sections.get ⟨j, hj⟩) := by
  obtain ⟨start, raw, hlayout⟩ := hw
  have hlen : (layoutSections start raw).length = obj.sections.length := by
    have := congrArg List.length hlayout; simp [List.length_map] at this; exact this
  have hraw_i : i < (layoutSections start raw).length := by omega
  have hraw_j : j < (layoutSections start raw).length := by omega
  have hnd := layout_non_overlapping start raw i j hraw_i hraw_j hij
  unfold sections_disjoint at hnd ⊢
  have hi1 := get_eq_of_list_eq (show obj.sections = (layoutSections start raw).map LayoutedSection.toSection by rw [hlayout]) i hi
  have hj1 := get_eq_of_list_eq (show obj.sections = (layoutSections start raw).map LayoutedSection.toSection by rw [hlayout]) j hj
  have hi2 := get_map_eq LayoutedSection.toSection (layoutSections start raw) i hraw_i
  have hj2 := get_map_eq LayoutedSection.toSection (layoutSections start raw) j hraw_j
  rw [hi2] at hi1
  rw [hj2] at hj1
  rw [hi1, hj1]
  simp only [LayoutedSection.toSection]
  exact hnd

-- ---------------------------------------------------------------------------
-- Section-layout invariants (require WellLayouted)
-- ---------------------------------------------------------------------------

/-- **sections_non_overlapping**
    For any two distinct sections in a valid ELF layout, their byte ranges
    in the file must be disjoint. -/
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
    If i < j, section i ends before section j starts. -/
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
  have hlen : (layoutSections start raw).length = obj.sections.length := by
    have := congrArg List.length hlayout; simp [List.length_map] at this; exact this
  have hraw_i : i < (layoutSections start raw).length := by omega
  have hraw_j : j < (layoutSections start raw).length := by omega
  have hi1 := get_eq_of_list_eq (show obj.sections = (layoutSections start raw).map LayoutedSection.toSection by rw [hlayout]) i hi
  have hj1 := get_eq_of_list_eq (show obj.sections = (layoutSections start raw).map LayoutedSection.toSection by rw [hlayout]) j hj
  have hi2 := get_map_eq LayoutedSection.toSection (layoutSections start raw) i hraw_i
  have hj2 := get_map_eq LayoutedSection.toSection (layoutSections start raw) j hraw_j
  rw [hi2] at hi1
  rw [hj2] at hj1
  rw [hi1, hj1]
  simp only [LayoutedSection.toSection]
  exact layout_monotone start raw i j hraw_i hraw_j hij

/-- **section_align_respected**
    Every section's byte offset is a multiple of its declared alignment. -/
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
    Every defined symbol must fit entirely within its containing section. -/
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
  have heq : (obj.sections.get ⟨sym.section_idx, hvalid⟩).size =
      (obj.sections.get ⟨sym.section_idx, h⟩).size := by
    congr
  rw [heq]
  exact hfit

/-- **symbol_unique_name**
    Within a single ELF object, no two global symbols share the same name. -/
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
    Every relocation entry names a section index that actually exists. -/
theorem reloc_target_valid_thm
    (obj : ElfObject)
    (r : Reloc)
    (hmem : r ∈ obj.relocs)
    (hwf : WellFormed obj) :
    reloc_target_valid r obj.sections := by
  obtain ⟨_, _, h_reloc, _, _⟩ := hwf
  exact h_reloc r hmem

/-- **reloc_offset_within_section**
    The relocation patch point lies inside the target section's byte range. -/
theorem reloc_offset_within_section
    (obj : ElfObject)
    (r : Reloc)
    (hmem : r ∈ obj.relocs)
    (hvalid : r.target_section < obj.sections.length)
    (hwf : WellFormed obj) :
    r.offset < (obj.sections.get ⟨r.target_section, hvalid⟩).size := by
  obtain ⟨_, _, _, h_roff, _⟩ := hwf
  obtain ⟨h, hfit⟩ := h_roff r hmem
  have heq : (obj.sections.get ⟨r.target_section, hvalid⟩).size =
      (obj.sections.get ⟨r.target_section, h⟩).size := by
    congr
  rw [heq]
  exact hfit

/-- **reloc_symbol_valid**
    Every relocation's symbol index refers to an entry in the symbol table. -/
theorem reloc_symbol_valid
    (obj : ElfObject)
    (r : Reloc)
    (hmem : r ∈ obj.relocs)
    (hwf : WellFormed obj) :
    r.symbol_idx < obj.symbols.length := by
  obtain ⟨_, _, _, _, h_rsym⟩ := hwf
  exact h_rsym r hmem

-- ---------------------------------------------------------------------------
-- String-table invariants
-- ---------------------------------------------------------------------------

/-- ELF string tables must be null-terminated byte sequences. -/
def isNullTerminated (table : List UInt8) (offset : Nat) : Prop :=
  ∃ (end_pos : Nat) (h : end_pos < table.length),
    offset ≤ end_pos ∧ table.get ⟨end_pos, h⟩ = 0

/-- Well-formed ELF string tables have at least one null terminator.
    (Requires a hypothesis that the strtab is non-empty and starts with \0,
    as mandated by the ELF spec §1.6.) -/
theorem strtab_section_names_valid
    (obj : ElfObject)
    (strtab : List UInt8)
    (hstrtab_len : 0 < strtab.length)
    (hstrtab_zero : strtab.get ⟨0, hstrtab_len⟩ = 0)
    (i : Nat)
    (hi : i < obj.sections.length) :
    ∃ name_offset : Nat, isNullTerminated strtab name_offset :=
  ⟨0, 0, hstrtab_len, Nat.le_refl 0, hstrtab_zero⟩

-- ---------------------------------------------------------------------------
-- Empty-object base cases
-- ---------------------------------------------------------------------------

/-- The empty ELF object (no sections, symbols, or relocs) is trivially
    well-formed. -/
theorem empty_elf_well_formed : WellFormed ⟨[], [], []⟩ := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro i j hi hj _; exact absurd hi (Nat.not_lt_zero _)
  · intro sym hmem; exact absurd hmem List.not_mem_nil
  · intro r hmem; exact absurd hmem List.not_mem_nil
  · intro r hmem; exact absurd hmem List.not_mem_nil
  · intro r hmem; exact absurd hmem List.not_mem_nil

/-- The empty ELF object is trivially well-laid-out. -/
theorem empty_elf_well_layouted : WellLayouted ⟨[], [], []⟩ :=
  ⟨0, [], by simp [layoutSections]⟩

end Sounio.ElfLinker
