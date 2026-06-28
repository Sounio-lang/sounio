# Auditoria 2+3: Lanes + Relatório Estruturado
**Foco**: Leitura de diffs/arquivos chave de lanes relevantes + relatório sobre onde há "controle de verdade" (IR ops + lowering custom que emite instruções reais) vs stubs, com ênfase em Madaros (substituindo lean_single) e caminho GPU até CUBIN.

Data: 2026-06-27
Contexto: papo solto no xai-conversa. Tema central = usar controle total do compilador (fonte até machine code / CUBIN) para quebrar fronteiras científicas (epistemic, incerteza GUM, álgebras não-associativas, etc.).

## 1. Lanes / Worktrees / Branches inspecionados (dos resultados da auditoria anterior)

Usamos dados de `git worktree list`, branches recentes e o TSV do audit.

**Codex-heavy (muito ativos em Madaros + lowering + native/GPU):**
- `/tmp/sounio-madaros-close-20260627` → `codex/madaros-close-20260627`
- `/tmp/sounio-madaros-retire-lean-single-20260627` → `codex/madaros-retire-lean-single-20260627` (dirty em module_frontend, scripts Madaros)
- `/tmp/sounio-sota-plus-20260627` → `codex/sota-plus-runtime-abi-20260627` (dirty em ir/lower + module_frontend)
- `/tmp/sounio-tuple-signature-type-lowering-20260627` → lowering de tuplas
- `/workspace/sounio-madaros-*` vários (check-segv, main-proof, source-elf-consolidated, project-spine)
- `/workspace/sounio-no-caveats` → codex/no-caveats-warning-zero
- Vários `codex/sota-gpu-*`, `codex/native-closure-*`, `codex/gum-*`, `codex/lowerer-error-propagation`

**Claude lanes (mais em IR, codegen, effects, checker):**
- `/workspace/sounio-codegen` → `claude/codegen-largestruct-fix` (mexeu em codegen)
- `/workspace/sounio-ir` → `claude/ir-heap-indirect`
- `/workspace/sounio-checker`, `/workspace/sounio-effects`, `/workspace/sounio-parser-*` etc.

**Pesquisa / Epistemic / Non-assoc (muito relevantes pro tema):**
- `.claude/worktrees/madaros-default` → `fix/assoc-variance-273-consolidated` (trabalho pesado no 168 theorem + GUM em lowering)
- `/workspace/sounio-affine` → `feat/affine-nonassoc-uncertainty`
- `/workspace/sounio-affine-pg` → `feat/affine-octonion-correlation`
- `/workspace/sounio-epitensor` → dissertation/epistemic-tensor
- `/workspace/sounio-solver-sota` → research/solver-sota-class
- `/workspace/sounio-zd-surgery` → research/sedenion-zd-chromatic
- GPU lanes: codex/sota-gpu-hlir-bridge, claude/solver-gpu-native-path, gpu/epistemic-tensor-core-gum-sm75

**Agentes em .claude/worktrees (persistentes):**
- `agent-adc1cd8b9d52ba53b` → tocou diretamente `lower_ir.sio`, `codegen_x86_linux.sio`, `native/codegen.sio`, `suite.sio` (apareceu como critical dirty na auditoria)
- `agent-a203887be9ace9526`

**Outros notáveis:**
- `/workspace/sounio-merge` → integration/native-v2-honest
- `/workspace/sounio-nv2-consolidate` → native-v2 paths
- Vários /tmp/sounio-* de 27/06 com lowering, GPU, Madaros.

**Nosso canto:**
- `.claude/worktrees/xai-conversa` → `xai/conversa` (isolado, limpo)

A auditoria oficial mostrou 80 worktrees, 56 dirty, 5 critical unallowed (vários tocando exatamente lowering/native/Madaros).

## 2. Leitura de arquivos chave + highlights de "controle de verdade"

Foco em onde há lowering custom que **emite instruções reais** (EVEX, Fano sequences, bytes de CUBIN, etc.) vs lower_copy / stubs.

### 2.1 Caminho Madaros + Lowering Nativo (o coração do controle até machine code)

**Arquivos inspecionados:**
- `self-hosted/native/lower_ir.sio` (o principal)
- `self-hosted/native/codegen_x86_linux.sio`
- `self-hosted/ir/ir.sio` (definição de opcodes)
- `self-hosted/compiler/module_native_driver.sio` (integração Madaros)
- `self-hosted/compiler/native_compile_driver.sio`

**O que é controle de verdade (custom emission):**

- `lower_hyper_mul_o_fano`, `lower_hyper_mul_o`, `lower_hyper_mul_s`, `lower_associator`
- **`lower_assoc_variance`** (o destaque do madaros-default / assoc-variance branch):

  Usa o **168 theorem** como otimização de codegen:
  - Fano triples (168/343): 1 instrução VXORPD (correção = 0)
  - Non-Fano: sequência de ~128 EVEX (Fano multiplies + VSUBPD + VMULPD + broadcast de σ² vinda do checker via imm_f64)

  Emissão real:
  - `emit_fano_inline` (reutiliza infra)
  - `emit_evex_pd_rr_full` para VXORPD (0x57), VSUBPD (0x5C), VMULPD (0x59)
  - MOVSD + VBROADCASTSD de constante de rodata + relocs
  - Máscaras, zb, ctrl_base vindos de label_id/imm_flags

  Comentário no código: "First compiler to correctly propagate GUM uncertainty through non-associative arithmetic."

  Isso é **exatamente** o tipo de alavanca que você quer: o compilador gera código de máquina diferente baseado em teorema matemático + incerteza epistêmica.

- Outros custom:
  - Vector ops EVEX (add, mul, fma, minmax, addsub, permil, mask_*)
  - Várias roots e cartan muls para álgebras E6/E7/E8/F4 (J3O Jordan, trace, det)
  - `IrProfCounter` → emite `INC QWORD PTR [RIP+disp32]` + reloc real
  - `IrLoadFnRef` → LEA rax, [RIP + disp32] + call reloc

**O que ainda é stub/copy (pouco controle):**

- Quase todos os epistemic "wrappers":
  - `IrLiftKnowledge`, `IrMeasure`, `IrContest`, `IrProveRobust`, `IrValidated`, `IrAdmitAction`, `IrDeferAction`, `IrPlan*`, `IrCommitAlternative` → todos `lower_copy` (só MOV)
  - `IrSedZDCheck` → explicitamente lower_copy (trabalho real fica no checker)
  - Muitos ops epistêmicos novos (TropicalAdd/Mul, FreeConvolve, BooleanConvolve, WassersteinDist, ORC) têm definição no IR mas lowering atual é trivial ou não custom.

No Madaros path (module_native_driver + native drivers):
- O driver já despacha GPU (gpu_binary_format = cubin etc.).
- Mas muito do lowering profundo ainda herda do caminho antigo ou está em transição (vários ramos codex/madaros estão fechando exatamente esses gaps).

### 2.2 Caminho GPU até CUBIN (o "2+3" que você mencionou)

**Arquivos chave:**
- `self-hosted/gpu/kretikos_emit_cubin.sio`
- `self-hosted/gpu/kretikos_emit_ptx.sio`, `kretikos_kaxi_to_ptx.sio`
- `self-hosted/gpu/hlir_to_gpu.sio`, `lower_to_ptx.sio`, `ptx_emitter.sio`
- `self-hosted/gpu/kretikos_cubin_validate.sio`
- `self-hosted/gpu/epistemic_*` (tensor_core, kernels, autodiff, etc.)
- Integração: `module_native_driver.sio` (parâmetros gpu_target, gpu_binary_format, gpu_strict_parity)

**O que é controle de verdade:**

- `kretikos_emit_cubin.sio` gera **bytes reais de CUBIN** (SM80, vec_add_f32, fma, vec_add_f64 via nvidia_bare chunks, etc.).
- Emite decimal byte a byte ou hex chunks.
- Suporta epistemic_dual_f32 (menção explícita).
- Há emissão de SASS nativo em alguns tracks (byte-exact).
- Kretikos tem múltiplos backends (PTX, CUBIN, SPIR-V, Metal, HIP/ROCm em ramos recentes).
- Muita coisa epistêmica já existe no GPU: epistemic_tensor_core, second_order_gum no otimizador, epistemic fusion, tiled covariance com incerteza.

**O que ainda é limitado:**
- GPU CLI entry point ainda é stub em alguns lugares (gpu/lib.sio).
- Muitos kernels epistêmicos ainda em PTX reference ou emulação; emissão full CUBIN com GUM profundo ainda em desenvolvimento.
- Transição Madaros → GPU emission está acontecendo (vários ramos codex/sota-gpu e claude/solver-gpu-native-path).

Commits recentes mostram trabalho ativo:
- bridge kretikos gpu runtime to modular compiler
- native SASS encoder
- real AMD ROCm HIP backend
- epistemic path-bearing knowledge surfaces + GPU Phase Y GUM

## 3. Onde está o verdadeiro poder (e as maiores oportunidades)

**Controle de verdade (já existe e pode ser estendido agora):**
- Adicionar novo `IrOpcode` (ou estender os hyper/epistemic) + `lower_xxx` dedicado que emite instruções específicas (EVEX sequences, Fano custom, bytes CUBIN, thread intrinsics, etc.).
- Usar metadata do checker (imm_f64 = σ², label_id = masks, imm_flags) para condicionar a emissão em teoremas científicos.
- Exemplo já rodando: assoc_variance com 168 theorem como branch de otimização no codegen.
- GPU: emissão de CUBIN com suporte a dual-lane epistemic.

**Stubs / onde falta controle:**
- A maioria das operações em `Knowledge<T>` ainda vira "wrap + copy bits" no lowering.
- Muitos ops científicos avançados (convolves, tropical, Wasserstein) estão no IR mas o lowering não desce a semântica completa.
- Guards epistêmicos (knowledge_runtime_guard_*) têm planejamento bom, mas a emissão nativa/GPU ainda está em probes.

**Implicações para fronteiras científicas (no contexto Madaros + GPU/CUBIN):**
- Madaros está reorganizando exatamente o substrate que permite estender isso de forma limpa (module drivers + lowering separado).
- O caminho até CUBIN + os epistemic_ no GPU é uma das maiores alavancas: você pode gerar kernels que propagam incerteza de forma que compiladores CUDA normais + libs não conseguem (ou só de forma frágil).
- Non-assoc + GUM no nível de instruções/GPU = algo que quase ninguém tem.
- Self-hosting + Madaros significa que você pode fazer o próprio compilador participar da ciência (passes com Knowledge, lowering que usa teoremas).

## 4. Resumo rápido + recomendações

- **Quem está puxando as alavancas de controle real**:
  - Codex: maioria dos ramos Madaros + lowering + GPU SOTA (muitos /tmp + /workspace/sounio-madaros-*)
  - Agente adc1cd8b9d52ba53b (em .claude/worktrees): mexeu diretamente em lower_ir + codegen
  - madaros-default (assoc-variance): o trabalho mais avançado em non-assoc + incerteza no codegen
  - GPU/epistemic lanes: codex/sota-gpu + claude/solver-gpu

- **Onde focar para quebrar fronteira**:
  1. Estender os custom lowerings que já emitem instruções (use lower_assoc_variance + emit_fano + emit_evex como template).
  2. Fazer ops epistêmicos descerem de verdade para CUBIN (não só PTX reference).
  3. Usar os guards + Madaros modular para constraints científicas virarem código nativo/GPU.
  4. Aproveitar o 168 theorem style: detecção de casos especiais no lowering para emissão radicalmente diferente.

Se quiser, posso agora:
- Aprofundar um lowering específico (ex: o Fano completo ou kretikos_cubin)
- Comparar estado atual vs algum branch específico
- Atualizar este relatório com mais detalhes de diffs recentes

O arquivo está salvo em `.claude/worktrees/xai-conversa/auditoria-2-3-lanes-e-controle.md`

Quer continuar por qual ângulo? (Madaros vs GPU, um lowering específico, ou quem mexeu em quê nos últimos dias?)