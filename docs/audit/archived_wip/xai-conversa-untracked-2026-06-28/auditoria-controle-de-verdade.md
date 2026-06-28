# Auditoria Focada: "Controle de Verdade" até Machine Code

**Contexto**: Papo solto nível xAI. Foco estrito em onde o Sounio já dá **controle real** (novo opcode IR + lowering custom que emite instruções x86 concretas) versus stubs/copies/placeholders.

Data da auditoria: 2026-06-27 (baseado em estado atual do workspace + worktrees).

## 1. O que conta como "Controle de Verdade"

- Definição de novo `IrOpcode` em `self-hosted/ir/ir.sio`
- Implementação de lowering custom em `self-hosted/native/lower_ir.sio` (não apenas `lower_copy`)
- Emissão de instruções reais (EVEX, Fano sequences, VMULPD/VSUBPD/VXORPD, relocs, rodata, etc.)
- Uso de invariantes científicas no próprio codegen (ex: 168 theorem como branch de otimização)

## 2. Exemplos Reais de Controle (o que já funciona de verdade)

### 2.1 Hypercomplex / Não-Associativo (o mais avançado)

- `lower_hyper_mul_o_fano`
- `lower_hyper_mul_o`
- `lower_hyper_mul_s`
- `lower_associator`
- `lower_assoc_variance` ← **o mais impressionante**

Em `lower_assoc_variance`:

```sio
// Comentário no código:
"First compiler to correctly propagate GUM uncertainty through non-associative arithmetic."

// 168 theorem como gate de otimização no codegen:
Fano triples (168/343) → 1 VXORPD (correção = 0)
Non-Fano (175/343) → ~128 instruções EVEX para ||[a,b,c]||² × σ²

// Emissão real:
- emit_fano_inline (4 chamadas)
- emit_evex_pd_rr_full para VXORPD, VSUBPD, VMULPD
- MOVSD + VBROADCASTSD de σ² (da imm_f64 vinda do checker)
- relocs + data section para constantes
```

Isso é **controle de verdade** no nível de fronteira científica. Você está emitindo sequências de instruções que dependem de teoremas matemáticos (168) e de epistemic metadata (σ² do Knowledge).

Outros:
- `lower_j3o_jordan_mul`, `lower_j3o_trace`, `lower_j3o_det` (Albert algebra J3(O)) — em desenvolvimento, com comentários de "first compiler".

### 2.2 Outros custom lowerings não-trivial

- Vector ops com EVEX: `lower_vec_add`, `lower_vec_mul`, `lower_vec_fma`, `lower_vec_minmax`, `lower_vec_addsub`, `lower_vec_permil*`, `lower_mask_*`
- `lower_hyper_conjugate_o`
- `lower_sed_zd_check` (parcial: reconhece que o trabalho real é no checker, lowering é copy)

## 3. O que ainda é Stub / Copy / Placeholder (pouco controle)

Muitos ops epistêmicos "novos" ainda caem em:

```sio
lower_copy(nc, instr)   // só MOV bits
```

Exemplos:
- `IrLiftKnowledge` → "Knowledge<T> → Contest<T> bridge" → lower_copy
- `IrMeasure` → "raw observation → Knowledge<T>" → lower_copy
- `IrTropicalAdd`, `IrTropicalMul` (definidos no IR, mas lowering não aparece como custom dedicado nos greps principais)
- `IrFreeConvolve`, `IrBooleanConvolve`, `IrWassersteinDist`, `IrORC` (comentários descrevem o que deveriam fazer, mas lowering atual parece simples ou delegado)

`lower_sed_zd_check` explicitamente:
```sio
lower_copy(nc, instr)  // "The real work happens in the type-checker"
```

Muitos epistemic "Doors" (Ω, Σ, Δ, Φ, β) estão no IR como conceitos avançados, mas a descida para instruções concretas ainda é superficial.

## 4. Knowledge Runtime Guards (emergente, promissor)

Existe estrutura de planejamento:
- `knowledge_runtime_guard_lowering_plan`
- `knowledge_runtime_guard_native_emission_plan`
- Probes: `k2_knowledge_runtime_guard_native_emission_step_probe`

Isso mostra intenção de levar obrigações epistêmicas (do checker) até emissão nativa (possivelmente traps/guards no código gerado).

Ainda em fase de probes e planos — não é emissão massiva de instruções custom ainda, mas é um dos melhores ganchos para "controle científico" futuro (ex: runtime checks de epsilon, unidades, etc. no machine code).

## 5. Onde Está o Verdadeiro Poder Hoje

**Maior alavanca atual**:
- Adicionar novo `IrOpcode` (ou reutilizar os hyper/epistemic)
- Escrever `lower_xxx` dedicado que faz emissão real de instruções (usando `emit_evex_*`, `emit_fano_inline`, relocs, rodata, etc.)
- Usar metadata do checker (imm_f64 para σ², label_id para masks, etc.) para condicionar o código gerado em teoremas/propriedades científicas.

Exemplo concreto já funcionando: o caminho do associator variance usa o 168 theorem para decidir entre 1 instrução ou ~128. Isso é o tipo de coisa que permite "quebrar fronteiras" — o compilador gera código diferente baseado em matemática de fronteira + incerteza.

**O que ainda falta para controle pleno**:
- A maioria das operações em `Knowledge<T>` ainda é "wrap + copy bits" no lowering. A propagação GUM real ainda não está fundida nas instruções de forma geral.
- Muitos ops científicos avançados definidos no IR, mas lowering ainda não emite a semântica completa.

## 6. Implicações para Fronteiras Científicas

Com o controle que já existe (mesmo que parcial):

- Você pode prototipar operadores que **não existem** em nenhuma outra linguagem/compilador (ex: convolve livre com R-transform + GUM no nível de registradores ZMM).
- Pode fazer codegen que respeita não-associatividade de forma otimizada (já demonstrado).
- Pode injetar guards/traps baseados em constraints epistêmicas diretamente no ELF.
- O fato de tudo ser escrito em Sounio + lowering em Sounio significa que você pode fazer o compilador participar da ciência (passes que usam Knowledge, lowering que usa teoremas).

**Recomendação prática**:
Comece por estender os caminhos que já têm lowering custom real:
1. Olhe `lower_assoc_variance` + `emit_fano_inline` como template.
2. Adicione ops novos no IR para o que você quer explorar.
3. Implemente lowering que emite sequências específicas (não copy).
4. Use imm_*/label_id para passar metadata científica do checker.

Os stubs atuais (Lift/Measure, tropicals, convolves) são exatamente os lugares onde falta "controle de verdade" — são as maiores oportunidades de impacto.

---

Este arquivo vive aqui porque é papo solto nível xAI. Pode ser editado, expandido, ou usado como base pra brainstorm de arquiteturas de fronteira.

Quer que eu aprofunde em algum lowering específico (ex: o Fano + 168 completo) ou compare com algum branch/worktree que está mexendo nisso agora?