/-!
# Sounio.ElfLinker — Phase 8 Formal Verification

Invariants for the ELF64 object-file writer and linker found in
  `crates/souc/src/backend/native/elf.rs`
  `crates/souc/src/backend/native/linker.rs`

All theorems are currently admitted with `sorry`; the intent is to
discharge them with a combination of omega/simp (arithmetic) and
constructive witnesses from the Rust implementation once a verified
extraction path exists.

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
-- Section-layout invariants
-- ---------------------------------------------------------------------------

/-- **sections_non_overlapping**
    For any two distinct sections in a valid ELF layout, their byte ranges
    in the file must be disjoint.

    Proof strategy (TODO): induction on the section list; the layout pass
    places each section at `alignUp(prev_end, sec.align)`, which guarantees
    monotonically increasing, non-overlapping offsets. -/
theorem sections_non_overlapping
    (obj : ElfObject)
    (i j : Nat)
    (hi : i < obj.sections.length)
    (hj : j < obj.sections.length)
    (hij : i ≠ j) :
    let si := obj.sections.get ⟨i, hi⟩
    let sj := obj.sections.get ⟨j, hj⟩
    sections_disjoint si sj := by
  sorry

/-- **sections_offset_monotone**
    If sections are laid out left-to-right (as the Rust `finish()` method
    does), then i < j implies section i starts before section j. -/
theorem sections_offset_monotone
    (obj : ElfObject)
    (i j : Nat)
    (hi : i < obj.sections.length)
    (hj : j < obj.sections.length)
    (hij : i < j) :
    let si := obj.sections.get ⟨i, hi⟩
    let sj := obj.sections.get ⟨j, hj⟩
    si.offset + si.size ≤ sj.offset := by
  sorry

/-- **section_align_respected**
    Every section's byte offset in the file is a multiple of its declared
    alignment, matching the ELF spec requirement for `sh_addralign`. -/
theorem section_align_respected
    (obj : ElfObject)
    (i : Nat)
    (hi : i < obj.sections.length) :
    let s := obj.sections.get ⟨i, hi⟩
    s.offset % s.align = 0 := by
  sorry

-- ---------------------------------------------------------------------------
-- Symbol-containment invariants
-- ---------------------------------------------------------------------------

/-- **symbol_within_section**
    Every defined symbol must fit entirely within its containing section.
    This prevents out-of-bounds memory access when the loader maps sections. -/
theorem symbol_within_section
    (obj : ElfObject)
    (sym : Symbol)
    (hmem : sym ∈ obj.symbols)
    (hvalid : sym.section_idx < obj.sections.length) :
    sym.offset + sym.size ≤
      (obj.sections.get ⟨sym.section_idx, hvalid⟩).size := by
  sorry

/-- **symbol_unique_name**
    Within a single ELF object, no two global symbols share the same name.
    Duplicate global symbols cause linker errors; the verifier should
    reject them at emit time. -/
theorem symbol_unique_name
    (obj : ElfObject)
    (s1 s2 : Symbol)
    (h1 : s1 ∈ obj.symbols)
    (h2 : s2 ∈ obj.symbols)
    (heq : s1.name = s2.name) :
    s1 = s2 := by
  sorry

-- ---------------------------------------------------------------------------
-- Relocation invariants
-- ---------------------------------------------------------------------------

/-- **reloc_target_valid**
    Every relocation entry names a section index that actually exists.
    An out-of-range section index would cause linker UB. -/
theorem reloc_target_valid_thm
    (obj : ElfObject)
    (r : Reloc)
    (hmem : r ∈ obj.relocs) :
    reloc_target_valid r obj.sections := by
  sorry

/-- **reloc_offset_within_section**
    The relocation patch point (offset within the target section) must lie
    inside that section's byte range, otherwise the patch write overflows
    the section buffer. -/
theorem reloc_offset_within_section
    (obj : ElfObject)
    (r : Reloc)
    (hmem : r ∈ obj.relocs)
    (hvalid : r.target_section < obj.sections.length) :
    r.offset < (obj.sections.get ⟨r.target_section, hvalid⟩).size := by
  sorry

/-- **reloc_symbol_valid**
    Every relocation's symbol index refers to an entry in the symbol table. -/
theorem reloc_symbol_valid
    (obj : ElfObject)
    (r : Reloc)
    (hmem : r ∈ obj.relocs) :
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
-- Composite well-formedness predicate
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

/-- The empty ELF object (no sections, symbols, or relocs) is trivially
    well-formed. -/
theorem empty_elf_well_formed : WellFormed ⟨[], [], []⟩ := by
  constructor
  · intro i j hi hj _; exact absurd hi (Nat.not_lt_zero _)
  constructor
  · intro sym hmem; exact absurd hmem (List.not_mem_nil _)
  · intro r hmem; exact absurd hmem (List.not_mem_nil _)

end Sounio.ElfLinker
