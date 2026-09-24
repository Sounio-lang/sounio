# Sounio Formal Verification — Axiom Inventory

Generated according to **Fase 1 — Inventario e reproducao da inconsistencia em `formal/`**.

## Classificacao Epistemica:
- **Core Standard**: `propext`, `Quot.sound`, `Classical.choice` (axiomas padrao do Lean 4 Core = **OK**)
- **Axiomas Locais**: Postulados sem prova formal via `axiom` = **NAO ESTABELECIDO** (Aviso de Fronteira de Confianca)

| Arquivo | Teorema / Axioma | Linha | Enunciado / Declaracao | Classificacao |
|---|---|---|---|---|
| `formal/Epistemic.lean` | `float_add_nonneg` | 265 | `axiom float_add_nonneg (a b : Float) : 0.0 ≤ a → 0.0 ≤ b → 0.0 ≤ a + b` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_mul_nonneg` | 268 | `axiom float_mul_nonneg (a b : Float) : 0.0 ≤ a → 0.0 ≤ b → 0.0 ≤ a * b` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_abs_nonneg` | 271 | `axiom float_abs_nonneg (a : Float) : 0.0 ≤ Float.abs a` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_le_antisymm` | 274 | `axiom float_le_antisymm (a b : Float) : a ≤ b → b ≤ a → a = b` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_add_le_add` | 277 | `axiom float_add_le_add (a b c d : Float) : a ≤ b → c ≤ d → a + c ≤ b + d` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_mul_le_mul_left` | 280 | `axiom float_mul_le_mul_left (a b c : Float) : b ≤ c → 0.0 ≤ a → a * b ≤ a * c` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_le_refl` | 283 | `axiom float_le_refl (a : Float) : a ≤ a` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_le_trans` | 286 | `axiom float_le_trans (a b c : Float) : a ≤ b → b ≤ c → a ≤ c` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_add_comm` | 289 | `axiom float_add_comm (a b : Float) : a + b = b + a` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_mul_comm` | 294 | `axiom float_mul_comm (a b : Float) : a * b = b * a` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_add_zero` | 299 | `axiom float_add_zero (a : Float) : a + 0.0 = a` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_zero_add` | 302 | `axiom float_zero_add (a : Float) : 0.0 + a = a` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_mul_one` | 305 | `axiom float_mul_one (a : Float) : a * 1.0 = a` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_one_mul` | 308 | `axiom float_one_mul (a : Float) : 1.0 * a = a` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_mul_zero` | 311 | `axiom float_mul_zero (a : Float) : a * 0.0 = 0.0` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_zero_mul` | 314 | `axiom float_zero_mul (a : Float) : 0.0 * a = 0.0` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_abs_one` | 319 | `axiom float_abs_one : Float.abs 1.0 = 1.0` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_zero_nonneg` | 322 | `axiom float_zero_nonneg : (0.0 : Float) ≤ 0.0` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_sub_self` | 325 | `axiom float_sub_self (a : Float) : a - a = 0.0` | **NAO ESTABELECIDO** |
| `formal/Epistemic.lean` | `float_div_one` | 328 | `axiom float_div_one (a : Float) : a / 1.0 = a` | **NAO ESTABELECIDO** |
| `formal/HessianAD.lean` | `float_zero_mul_ax` | 543 | `axiom float_zero_mul_ax (a : Float) : 0.0 * a = 0.0` | **NAO ESTABELECIDO** |
| `formal/HessianAD.lean` | `float_mul_zero_ax` | 546 | `axiom float_mul_zero_ax (a : Float) : a * 0.0 = 0.0` | **NAO ESTABELECIDO** |
| `formal/HessianAD.lean` | `float_zero_add_ax` | 549 | `axiom float_zero_add_ax (a : Float) : 0.0 + a = a` | **NAO ESTABELECIDO** |
| `formal/HessianAD.lean` | `float_add_zero_ax` | 552 | `axiom float_add_zero_ax (a : Float) : a + 0.0 = a` | **NAO ESTABELECIDO** |
| `formal/HessianAD.lean` | `float_add_comm_ax` | 555 | `axiom float_add_comm_ax (a b : Float) : a + b = b + a` | **NAO ESTABELECIDO** |
| `formal/HessianAD.lean` | `float_mul_comm_ax` | 558 | `axiom float_mul_comm_ax (a b : Float) : a * b = b * a` | **NAO ESTABELECIDO** |
| `formal/NonAssocHessian.lean` | `decidable_fano` | 259 | `axiom decidable_fano :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_mul_add_left` | 170 | `axiom oct_mul_add_left (x y z : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_mul_add_right` | 174 | `axiom oct_mul_add_right (x y z : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_left_alternative` | 215 | `axiom oct_left_alternative (x y : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_right_alternative` | 219 | `axiom oct_right_alternative (x y : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_flexibility` | 228 | `axiom oct_flexibility (x y : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_moufang_left` | 238 | `axiom oct_moufang_left (x y z : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_moufang_right` | 242 | `axiom oct_moufang_right (x y z : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_moufang_middle` | 246 | `axiom oct_moufang_middle (x y z : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_scalar_comm` | 256 | `axiom oct_scalar_comm (n : Int) (x y : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_scalar_comm_right` | 260 | `axiom oct_scalar_comm_right (n : Int) (x y : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_conj_antimultiplicative` | 269 | `axiom oct_conj_antimultiplicative (x y : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_mul_conj` | 283 | `axiom oct_mul_conj (x : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_conj_mul` | 287 | `axiom oct_conj_mul (x : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_norm_multiplicative` | 303 | `axiom oct_norm_multiplicative (x y : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OctonionAlgebra.lean` | `oct_sq_comm_left` | 312 | `axiom oct_sq_comm_left (x : Oct) :` | **NAO ESTABELECIDO** |
| `formal/OntologyELPlus.lean` | `and` | 571 | `axiom and the role inclusion contribute nothing to the atomic shadow. -/` | **NAO ESTABELECIDO** |
| `formal/OntologyELPlusNormalization.lean` | `collapses` | 36 | `axiom collapses to a reflexive pair (`normAxioms_collapse_eq`). The` | **NAO ESTABELECIDO** |
| `formal/OntologyEvolutionRepair.lean` | `addition` | 10 | `axiom addition is modelled*. Here the edit language gains **removal** —` | **NAO ESTABELECIDO** |
| `formal/SecondOrderGUM.lean` | `float_le_refl` | 153 | `axiom float_le_refl (a : Float) : a ≤ a` | **NAO ESTABELECIDO** |
| `formal/SecondOrderGUM.lean` | `float_add_le_add` | 156 | `axiom float_add_le_add (a b c d : Float) : a ≤ b → c ≤ d → a + c ≤ b + d` | **NAO ESTABELECIDO** |
| `formal/SecondOrderGUM.lean` | `float_mul_nonneg` | 160 | `axiom float_mul_nonneg (a b : Float) : 0.0 ≤ a → 0.0 ≤ b → 0.0 ≤ a * b` | **NAO ESTABELECIDO** |
| `formal/SecondOrderGUM.lean` | `float_trace_term_nonneg` | 165 | `axiom float_trace_term_nonneg` | **NAO ESTABELECIDO** |
| `formal/TypeCheckerSoundness.lean` | `float_le_refl` | 47 | `axiom float_le_refl (a : Float) : a ≤ a` | **NAO ESTABELECIDO** |
| `formal/TypeCheckerSoundness.lean` | `float_le_trans` | 50 | `axiom float_le_trans (a b c : Float) : a ≤ b → b ≤ c → a ≤ c` | **NAO ESTABELECIDO** |
| `formal/TypeCheckerSoundness.lean` | `float_add_le_add` | 53 | `axiom float_add_le_add (a b c d : Float) : a ≤ b → c ≤ d → a + c ≤ b + d` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioErdos90UnitSpectrum.lean` | `planar_udg_K23_free` | 278 | `axiom planar_udg_K23_free :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioFloatInstance.lean` | `Float.mul_bounded_error` | 94 | `axiom Float.mul_bounded_error :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioFloatInstance.lean` | `Float.le_trans` | 113 | `axiom Float.le_trans :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioFloatInstance.lean` | `Float.zero_le_zero` | 116 | `axiom Float.zero_le_zero :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioFloatInstance.lean` | `Float.mul_le_mul_of_nonneg_right_bounded` | 119 | `axiom Float.mul_le_mul_of_nonneg_right_bounded :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioFloatInstance.lean` | `Float.add_le_add_right_bounded` | 124 | `axiom Float.add_le_add_right_bounded :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioFloatInstance.lean` | `Float.le_trans` | 209 | `axiom Float.le_trans :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioFloatInstance.lean` | `Float.zero_le_zero` | 216 | `axiom Float.zero_le_zero : (0.0 : Float) ≤ 0.0` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioFloatInstance.lean` | `Float.mul_le_mul_of_nonneg_right_bounded` | 230 | `axiom Float.mul_le_mul_of_nonneg_right_bounded :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioFloatInstance.lean` | `Float.add_le_add_right_bounded` | 241 | `axiom Float.add_le_add_right_bounded :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioIEEE754Spec.lean` | `Float.toRat` | 105 | `axiom Float.toRat : Float → Rat` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioIEEE754Spec.lean` | `Float.IsFiniteNormal` | 122 | `axiom Float.IsFiniteNormal : Float → Prop` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioIEEE754Spec.lean` | `Float.toRat_le_iff_finite` | 137 | `axiom Float.toRat_le_iff_finite :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioIEEE754Spec.lean` | `Float.mul_rne_bound` | 183 | `axiom Float.mul_rne_bound :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioIEEE754Spec.lean` | `Float.add_rne_bound` | 209 | `axiom Float.add_rne_bound :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioIEEE754Spec.lean` | `Float.div_rne_bound` | 242 | `axiom Float.div_rne_bound :` | **NAO ESTABELECIDO** |
| `formal/lean4/SounioImpossibilityChain.lean` | `hurwitz_nda_iff_lt4` | 43 | `axiom hurwitz_nda_iff_lt4 : ∀ n : CDLevel, is_nda n ↔ n < 4` | **NAO ESTABELECIDO** |
