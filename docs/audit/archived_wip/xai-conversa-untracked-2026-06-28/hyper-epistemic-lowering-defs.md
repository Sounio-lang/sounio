# Definições de Lowering para Primitivo Hipercomplexo Epistêmico

**Primitivo proposto (novo opcode no IR):**  
`IrHyperEpistemicMul` (ou `IrPsychHyperOp` para o ângulo de psiquiatria computacional)

### Definição no IR (self-hosted/ir/ir.sio style)

```sio
IrHyperEpistemicMul,  // Hypercomplex (Octonion/Sedenion) mul com propagação GUM epistêmica
// src1, src2: Knowledge<Hyper> (octonion ou sedenion)
// imm_i64: algebra_kind (0=Octonion, 1=Sedenion, ...)
// imm_flags: bits para Fano hint, epistemic_mode (propagate variance?)
// label_id: ctrl_base para constantes Fano + máscara epistêmica
// dst: Knowledge<Hyper> com ε atualizado
// Semântica: (a * b) com correção de variância baseada em associator + GUM
// + provenance da interação (para double-bind / eigenform)
```

**Metadata que viaja:**
- value (o octonion/sedenion)
- ε / σ² (do Knowledge)
- provenance (string ou id da medição/interação)
- context (ex: nível de double-bind ou observador)

O lowering é **arquitetura-específico** e pode decidir:
- Caminho Fano-barato (168 theorem) vs completo
- Injetar variância só onde necessário
- Usar features específicas do hardware (EVEX, SVE, tensor cores, etc.)

---

## 1. x86 (AVX512 / EVEX - principal hoje)

**Arquivo:** `self-hosted/native/lower_ir.sio` (extensão de `lower_hyper_mul_o` + `lower_assoc_variance`)

```sio
pub fn lower_hyper_epistemic_mul(nc: &! NativeCompiler, instr: IrInstr) with Mut, Panic, Div {
    let algebra = instr.imm_i64;
    let epistemic_mode = (instr.imm_flags >> 2) & 1;  // 0 = fast, 1 = full GUM
    let is_fano = compute_fano_hint(instr.src1, instr.src2, instr.imm_i64);

    if is_fano && epistemic_mode == 0 {
        // Caminho barato (168 theorem): 1 VXORPD + mul simples
        emit_fano_inline(nc, instr.src1, instr.src2, instr.dst, ...);
        // Injeta ε só no final (barato)
        emit_variance_inject_light(nc, instr.dst, instr.imm_f64 /*σ²*/);
    } else {
        // Caminho completo + GUM
        let t = allocate_temp_zmms(12);
        emit_fano_full(nc, instr.src1, instr.src2, t, ctrl_base);
        emit_assoc_variance(nc, t, instr.dst, σ²_from_knowledge);
        // Preserva provenance no metadata (não em instrução, mas em side table ou rodata)
    }
}
```

**Emissão concreta:**
- Reusa `emit_evex_pd_rr_full` + `emit_fano_inline`.
- Para epistemic: adiciona VMULPD para variância no final.
- Registros: dst usa ZMMs; temps em dst+2..dst+13.
- Total instr: ~31 (Fano) ou ~128 (full) + custo da variância.

**Vantagem única:** o compilador decide o caminho baseado no teorema + ε. Nenhuma outra linguagem faz isso no lowering.

---

## 2. AArch64 (ARM64 - SVE/NEON)

**Arquivo:** `self-hosted/native/aarch64.sio` + extensão em lower_ir (novo)

```sio
pub fn lower_hyper_epistemic_mul_aarch64(nc: &! NativeCompiler, instr: IrInstr) {
    let use_sve = has_sve_support();
    if use_sve {
        // SVE: scalable vectors, bom para octonions (128-bit lanes)
        sve_fano_mul(nc, src1, src2, dst);
        if epistemic_mode {
            sve_fma_variance(nc, dst, σ²);  // FMA + predication
        }
    } else {
        // NEON fallback (4x128-bit)
        neon_octonion_mul(nc, src1, src2, dst);
        neon_scale_add_variance(nc, dst, σ²);
    }
}
```

**Emissão:**
- Usa `fmla` / `fmls` para mul + add.
- Para SVE: predication masks para os casos Fano (zero associator).
- Registros: V0-V31 ou Z0-Z31 (SVE).
- Não tem equivalente exato do Fano EVEX, mas podemos carregar constantes Fano em tabelas e usar selects.

**Observação:** Menos otimizado que x86 EVEX hoje (fewer lanes), mas o lowering pode emitir código que explora SVE2 para melhor throughput em hypercomplex.

---

## 3. ARM (32-bit - legacy NEON)

**Arquivo:** `self-hosted/native/` (arm32 support via apple_arm64_preview + general)

```sio
pub fn lower_hyper_epistemic_mul_arm32(nc: &! NativeCompiler, instr: IrInstr) {
    // NEON 128-bit (Q registers)
    neon_q_mul_octonion(nc, src1, src2, dst);
    if epistemic {
        neon_q_fma_variance(nc, dst, σ²);
    }
}
```

**Emissão:** Similar ao NEON AArch64 mas com restrições de registradores (Q0-Q15).
- Menos lanes = mais instruções.
- Usado principalmente para compatibilidade ou embedded.

**Nota:** Em 2026, ARM 32-bit está morrendo para compute científico, mas o lowering existe para completude.

---

## 4. CUBIN (NVIDIA - via Kretikos)

**Arquivos:** `self-hosted/gpu/kretikos_emit_cubin.sio`, `kretikos_kaxi_to_ptx.sio`, `hlir_to_gpu.sio`, `epistemic_tensor_core.sio`

```sio
pub fn lower_hyper_epistemic_mul_cubin(kernel: GpuKernel, instr: GpuInstr) {
    let sm = target_sm();  // SM80+
    if has_tensor_cores() && epistemic_mode {
        // Tensor core path (WMMA / mma.sync)
        emit_tensor_hyper_mul(nc, src1, src2, dst, fano_table_ptr);
        emit_gum_tensor_correction(nc, dst, σ²);  // usando atomic ou shared mem
    } else {
        // PTX/CUBIN scalar/vector
        emit_ptx_fma_octonion(nc, src1, src2, dst);
        if epistemic {
            emit_ptx_variance_inject(nc, dst, σ²);
        }
    }
    // Escreve bytes reais de CUBIN
}
```

**Emissão:**
- Reusa `gpu_bare_sm80_*` chunks + custom.
- Para epistemic: dual-lane (valor + variância) em registradores.
- Otimização Fano: branch no PTX ou predication.
- Resultado: CUBIN byte-exato que roda com semântica hipercomplexa + GUM preservada.

**Único no Sounio:** o lowering decide tensor-core vs scalar baseado em ε e algebra.

---

## 5. METAL (Apple - MSL)

**Arquivos:** `self-hosted/gpu/kretikos_emit_metal.sio`, `metal.sio`, `hlir_to_gpu.sio`

```sio
pub fn lower_hyper_epistemic_mul_metal(kernel: MetalKernel, instr: GpuInstr) {
    // MSL (Metal Shading Language)
    if has_apple_silicon_tensor() {
        emit_msl_simdgroup_hyper_mul(nc, src1, src2, dst);
    } else {
        emit_msl_threadgroup_fma(nc, src1, src2, dst);
    }
    if epistemic {
        emit_msl_atomic_variance(nc, dst, σ²);  // usando metal::atomic
    }
}
```

**Emissão:**
- MSL com `simdgroup` ou `threadgroup`.
- Metal não tem tensor cores expostos como NVIDIA, mas usa Apple AMX-like via intrinsics em chips M-series.
- Fano table carregada em constant buffer.
- Resultado: .metallib que pode rodar no GPU Apple com mesma semântica.

---

## 6. GPU AMD (ROCm / HIP)

**Status atual:** Em desenvolvimento (ramos recentes adicionam HIP backend).

**Esboço de lowering (em `kretikos_emit_hip.sio` ou equivalente):**

```sio
pub fn lower_hyper_epistemic_mul_hip(kernel: HipKernel, instr: GpuInstr) {
    let gfx = target_gfx();  // gfx1100+ para RDNA3 etc.
    if has_wmma() {  // Wave Matrix Multiply Accumulate
        emit_hip_wmma_hyper_mul(nc, src1, src2, dst, fano_table);
        emit_hip_gum_accumulate(nc, dst, σ²);
    } else {
        emit_hip_vector_fma(nc, src1, src2, dst);
        emit_hip_variance_fma(nc, dst, σ²);
    }
}
```

**Emissão:**
- HIP kernels (C++-like para AMD).
- Usa `wmma` intrinsics quando disponível (similar a tensor cores).
- Fano tables em __constant__.
- Atomic adds para variância em memória global/shared.
- Resultado: .hsaco ou fatbin que roda em GPUs AMD.

**Vantagem Sounio:** mesma semântica epistêmica + hipercomplexa em NVIDIA, Apple e AMD — lowering escolhe o melhor path por arquitetura.

---

## Resumo: o que torna isso "além do assembly"

- Não é só emitir ADD/MUL. É um **novo primitivo semântico** (`IrHyperEpistemicMul`) cujo lowering é:
  - Teorema-dirigido (168)
  - Epistemic-aware (σ² via Knowledge)
  - Arquitetura-específico (EVEX vs SVE vs WMMA vs MSL)
- O programador usa como se fosse uma instrução de alto nível.
- O hardware executa código que **só existe** porque o compilador Sounio decidiu o caminho.
- Na psiquiatria computacional: permite simular interações não-associativas com confiança quantificada de forma que roda nativamente no silício de várias plataformas.

Isso não é replicável em C++/CUDA + libs porque o otimizador não vê a álgebra + epistemic no lowering.

Quer que eu transforme isso em código Sounio mais próximo (pseudocódigo de IR + lowering functions) ou foque em como isso aparece no paper (seção de "novel architecture-specific lowering")? 

Pode ajustar o nome do op ou adicionar mais flags.