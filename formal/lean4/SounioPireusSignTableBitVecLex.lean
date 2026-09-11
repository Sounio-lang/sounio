/-
  Exact FORMAL_PARITY representation of a frozen Pireus 16 x 16 sign table
  as a 256-bit word.

  Cells are enumerated in row-major order, left lane first.  The first cell
  is stored as the most-significant bit, so the unsigned BitVec order is the
  frozen lexicographic order on cells 0, 1, ..., 255.  Both conversion
  directions are proved inverse before this representation is used by the
  concrete quotient action and its finite minimum.
-/
import SounioPireusGL4AnalyticActionCensus

namespace SounioPireusSignTableBitVecLex

open SounioPireusGaugeCoboundaryAction

def cellIndex (cell : Cell) : Fin 256 :=
  ⟨cell.1.val * 16 + cell.2.val, by omega⟩

def cellOfIndex (index : Fin 256) : Cell :=
  (⟨index.val / 16, by omega⟩,
    ⟨index.val % 16, Nat.mod_lt _ (by decide)⟩)

theorem cell_index_of_index (index : Fin 256) :
    cellIndex (cellOfIndex index) = index := by
  apply Fin.ext
  exact Nat.div_add_mod' index.val 16

theorem cell_of_index_index (cell : Cell) :
    cellOfIndex (cellIndex cell) = cell := by
  apply Prod.ext
  · apply Fin.ext
    change (cell.1.val * 16 + cell.2.val) / 16 = cell.1.val
    rw [Nat.mul_comm cell.1.val 16]
    rw [Nat.mul_add_div (by decide)]
    simp [Nat.div_eq_of_lt cell.2.isLt]
  · apply Fin.ext
    change (cell.1.val * 16 + cell.2.val) % 16 = cell.2.val
    exact Nat.mul_add_mod_of_lt cell.2.isLt

def tableCellList (table : SignTable) : List Bool :=
  List.ofFn fun index : Fin 256 => table (cellOfIndex index)

theorem table_cell_list_length (table : SignTable) :
    (tableCellList table).length = 256 := by
  unfold tableCellList
  exact List.length_ofFn

def packTable (table : SignTable) : BitVec 256 :=
  (BitVec.ofBoolListBE (tableCellList table)).cast
    (table_cell_list_length table)

def unpackTable (bits : BitVec 256) : SignTable :=
  fun cell => bits.getMsbD (cellIndex cell).val

theorem pack_table_get_msb
    (table : SignTable) (index : Nat) (indexLt : index < 256) :
    (packTable table).getMsbD index =
      table (cellOfIndex ⟨index, indexLt⟩) := by
  unfold packTable
  rw [BitVec.getMsbD_cast, BitVec.getMsbD_ofBoolListBE]
  unfold tableCellList
  rw [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
  simp [indexLt]

theorem unpack_pack_table (table : SignTable) :
    unpackTable (packTable table) = table := by
  funext cell
  rw [unpackTable,
    pack_table_get_msb table (cellIndex cell).val (cellIndex cell).isLt]
  exact congrArg table (cell_of_index_index cell)

theorem pack_unpack_table (bits : BitVec 256) :
    packTable (unpackTable bits) = bits := by
  apply BitVec.eq_of_getMsbD_eq
  intro index indexLt
  rw [pack_table_get_msb (unpackTable bits) index indexLt]
  simp [unpackTable, cell_index_of_index]

theorem pack_table_injective : Function.Injective packTable := by
  intro left right packedEqual
  have unpackedEqual := congrArg unpackTable packedEqual
  simpa [unpack_pack_table] using unpackedEqual

theorem unpack_table_injective : Function.Injective unpackTable := by
  intro left right unpackedEqual
  have packedEqual := congrArg packTable unpackedEqual
  simpa [pack_unpack_table] using packedEqual

end SounioPireusSignTableBitVecLex
