# Framework de computação heterogênea em Sounio — design

Status: Fase 0 (design + planner core). Data: 2026-10-23. Branch de referência: `main`.
Escopo: framework público para executar grafos ML/DL e inferência de LLMs sobre
pools de devices quaisquer — 1 device, N devices iguais, ou N devices desiguais
(NVIDIA multi-geração, AMD, CPU; Metal futuro). O que não existir, constrói-se
em Sounio; dependência externa só atrás de interface (ver "Definição de nativo").

## 1. Problema

Pools reais são heterogêneos: gerações de GPU diferentes, vendors diferentes,
CPU ao lado, interconexão que varia de NVLink a rede de 100 Gb/s com 1 GPU por
nó. Frameworks atuais assumem homogeneidade: shard 1/N igual, all-reduce
síncrono — o device mais lento vira o ritmo de todos. Ao mesmo tempo, o usuário
comum pode ter exatamente o caso homogêneo (8x a mesma GPU) ou uma GPU só, e
nesses casos o framework não pode cobrar pedágio da maquinaria heterogênea.

Tese: heterogeneidade é o caso geral; homogeneidade é o caso degenerado que o
mesmo planner resolve com o algoritmo clássico. A decisão de repartir um grafo,
repartir por camadas, ou não repartir é tomada por custo **medido** na máquina
do usuário, nunca por ficha técnica nem por constante de laboratório.

## 2. Componentes

### 2.1 DeviceCaps — capacidade medida

Por device: `backend` (CUDA-PTX, SPIR-V, CPU, Metal), classe de arquitetura
(ex.: SM89, SM121, gfx1201), VRAM (bytes), memória unificada (sim/não),
throughput medido por (op, dtype) em unidades declaradas, e links par-a-par
(banda GB/s, latência us). Gerado por probe automático na primeira execução;
o usuário nunca escreve essa tabela. Valores sem medição são inválidos:
`measured = false` impede o planner de usar o device para split (só colocação).

Implementação Fase 0: `stdlib/hetero/types.sio` (tipos) e registro em
`stdlib/hetero/caps.sio`.

### 2.2 Backend — interface única por vendor

Alocar, copiar, lançar kernel, sincronizar, reportar DeviceCaps.
Implementações planejadas: PTX multi-geração (flags por SM: WMMA muda de
Turing a GB10), SPIR-V para AMD (gate go/no-go na Fase 2), AVX-512 para CPU
(referência numérica permanente e fallback), Metal como stub de interface.
Nenhum chamador acima de Backend conhece vendor.

### 2.3 GraphIR — grafo único inferência/treino

Ops com shape/dtype por aresta; cada op declara kernels por backend, custo
estimado e dimensões fatiáveis. O backward do treino é o mesmo IR percorrido
ao contrário (Fase 6), não um segundo sistema de autograd.

### 2.4 CostModel — custo medido, plano inspecionável

Tempo estimado de uma op num device = trabalho / throughput medido (op,dtype);
transferência = bytes / banda do link + latência. O planner simula o plano
antes de executar e o plano é um artefato legível: usuário pode ver por que
um grafo foi repartido, colocado, ou degenerou para o clássico.

### 2.5 Planner — três modos sobre o mesmo grafo

1. **Identidade** (1 device): sem partição, sem overhead.
2. **Clássico degenerado** (capacidades medidas iguais dentro de tolerância):
   split igual, coletivos padrão (ring all-reduce / TP / PP conforme topologia).
3. **Heterogêneo**: split com fatias proporcionais ao throughput medido por
   op, limitado por VRAM por fatia; ou colocação de subgrafos inteiros quando
   o custo de comunicação supera o ganho de repartir (caso típico: decode de
   LLM com 1 GPU por nó e rede entre elas). "Não repartir" é um resultado
   válido e esperado do planner, não uma falha.

Fatia proporcional (modo 3), para trabalho total W e throughputs t_i:
fatia_i = W * t_i / sum(t), corrigida por teto de VRAM com redistribuição
iterativa do excedente entre os devices com folga. Implementação Fase 0:
`stdlib/hetero/planner.sio`, com testes em `examples/hetero_planner_demo.sio`.

### 2.6 Coletivos

All-reduce/all-gather em anel ou árvore sobre o elo mais lento, redução no
device mais rápido (ou CPU), buckets por tamanho. No modo clássico, o anel
padrão. Ordem de redução fixada pela regra numérica (seção 3).

## 3. Regra numérica canônica

Devices diferentes arredondam diferente; sem regra, "mesmo resultado" entre
partições não é alegação testável. Regra v1:

- Acumulação de GEMM e de reduções sempre em f32 (entradas bf16/f16 inclusive).
- Ordem de redução determinística por plano (a árvore é parte do plano).
- Paridade entre backends/partições é declarada por tolerância por dtype e
  medida contra a referência CPU; modo determinístico estrito é opção do
  usuário, com custo declarado.

## 4. Definição de "nativamente"

- Caminho de execução: binário ELF compilado pelo `souc` + kernels PTX/SPIR-V
  emitidos pelo próprio compilador. Zero Python, PyTorch ou CUDA C++.
- Externo permitido apenas atrás de `Backend`/FFI (`stdlib/ffi`): driver
  `libcuda`, runtime AMD, transporte de rede. cuBLAS/NCCL como bootstrap são
  decisão registrada neste documento com data para sair; nunca dependência
  silenciosa. Decisão inicial (2026-10-23): GEMM próprio via WMMA
  generalizado; cuBLAS-FFI apenas como oráculo de paridade na Fase 1/2.

## 5. Consolidação (auditoria Fase 0, medida em 2026-10-23)

Medido em `main` @ 3ef3dbaa6e: `stdlib/tensor` 1.775 LOC (f64, rank<=8),
`stdlib/nn` 16.246, `stdlib/ml` 3.056, `stdlib/autodiff` 4.035,
`stdlib/qnn` 2.219, `stdlib/snn` 5.325, `stdlib/ssm` 932, `stdlib/learn` 211,
`stdlib/distributed` 69 (só registry + stubs), `stdlib/gpu` 2.037.
Duplicações confirmadas: `pub struct Tensor` em `tensor/types.sio`,
`tensor/tape.sio`, `linalg/epistemic_tensor.sio`, `linalg/tensor16.sio`;
tapes/backward concorrentes em `tensor/tape.sio`, `autodiff/tape.sio`,
`ml/autodiff.sio`, `nn/autograd.sio`, `nn/autograd_v2.sio`.

Decisão: o canônico a partir deste design é **Tensor v2** (Fase 1, evolução de
`stdlib/tensor`) + **GraphIR** como único substrato de autograd (Fase 6).
`nn/ml/autodiff/qnn/snn/ssm/learn` ficam como legado consumível até a Fase 6
migrar o subconjunto mantido; nenhum código novo nasce neles. `distributed`
será substituído pelo runtime da Fase 4 (a casca de registry pode ser reusada).

## 6. Gates públicos (paridade antes de diferencial)

- Homogêneo: throughput/latência dentro de margem declarada antes de medir
  contra vLLM (inferência) e PyTorch (treino) em bancada homogênea; proposta
  inicial >= 90% do baseline para anunciar suporte.
- Heterogêneo: eficiência = throughput do pool / soma dos individuais medidos;
  publica-se o número qualquer que seja.
- Diferencial Sounio (tipos epistêmicos `Knowledge<T>`/GUM sobre o grafo,
  kernels com evidência) entra sobre a paridade, nunca abaixo dela.
- Toda alegação de performance exige harness reproduzível por terceiros
  (scripts, seeds, baselines versionados).

## 7. Fases

0. Este design + DeviceCaps/planner core com testes (esta entrega).
1. Tensor v2 (dtype f32/f16/bf16, device, refcount) + Backend CPU AVX-512.
2. Backends GPU: PTX multi-geração; spike go/no-go AMD SPIR-V; microbenchmarks
   alimentando o CostModel (o probe medido de verdade nasce aqui, via FFI).
3. GraphIR + CostModel + planner completo, validado previsto-vs-executado.
4. Runtime distribuído (workers, transferências, coletivos, replano).
5. LLM sobre o framework (safetensors, BPE, DecoderBlock, KV, serve).
6. Treino heterogêneo (autodiff no GraphIR, AdamW, LoRA, DDP proporcional).
7. Produto externo (pacote semver, quickstart, matriz de compatibilidade).

## 8. Registro de medição — Fase 2 (2026-10-23, OBSERVED)

Substrato de execução provado a partir de binário 100% Sounio, sem link-time
CUDA: o backend nativo não tem linker dinâmico arbitrário, mas seu tier
dynlink conhece `dlopen`/`dlsym` e resultados de `dlsym` são chamáveis por
cast de fn-pointer (`tests/run-pass/kl14d_dlopen.sio`). Sobre isso:

- `stdlib/hetero/probe_cuda.sio`: enumeração medida. Job Slurm 16083 (nó
  r770, partição bench): `NVIDIA L4, sm=89, vram 22 GiB` — primeiro
  DeviceCaps com números da máquina, não de ficha.
- `stdlib/hetero/cuda_exec.sio`: contexto primário, memória, transferências,
  `cuModuleLoadData`, `cuLaunchKernel`, eventos CUDA.
- Kernel GEMM executando: `stdlib/hetero/kernels/gemm_naive_f32.ptx`
  (artefato PTX próprio interino) lançado por `examples/hetero_kernel_demo.sio`,
  job 16088: GEMM 64x64 com **erro máximo 0** vs CPU; GEMM naive 1024
  a **1.681 GFLOP/s** medidos por eventos CUDA na L4. Número de kernel
  naive sem tiling — é o piso honesto do caminho, não o teto.
- cuBLAS: **ausente em todos os nós worker** (`ldconfig` só tem libcuda).
  O oráculo registrado na seção 4 não existe no parque; o caminho nativo
  acima é, portanto, o caminho principal, não o substituto futuro.
  `stdlib/hetero/cublas_bench.sio` permanece para máquinas que o tenham.
- Limitação registrada: o emissor PTX público (`souc build --backend gpu`)
  ainda emite kernels vazios; load/store a partir de `kernel fn` é o
  contrato que falta (Kretikos). Até lá, kernels são artefatos PTX no repo
  (precedente: `docs/gpu/*.ptx`), lançados pelo substrato acima.
- Bug encontrado e corrigido: `dlclose` de libcuda após `cuInit` pendura o
  processo na saída (job 16072: output completo, TIMEOUT, nó drenado por
  "Kill task failed"; nó retomado via `scontrol update ... state=resume`).
  `cuda_probe_close` não descarrega mais a lib, e demos chamam `exit(0)`.

Spike AMD (dl380, R9700): ambiente ESTABLISHED — `/dev/kfd`, HSA runtime,
`libamdhip64`, OpenCL presentes no worker. Execução adiada sem preempção:
a única R9700 está alocada ao serving (job 12798, gres/gpu=1). Caminho
escolhido quando houver janela: HIP via dlopen (mesma técnica do CUDA)
para enumeração; SPIR-V via OpenCL program-with-IL como candidata a kernel.

### Fechamento das pendências da Fase 2 (2026-10-23, OBSERVED)

- **Kernel emitido pelo compilador, executando**: o pipeline Kretikos
  (`self-hosted/gpu/kretikos_emit_ptx.sio`, padrões sobre KernelIR->PTX)
  emite PTX real com `ld.global`/`st.global`. O `vec_add` gerado por ele
  executou na L4 via `examples/hetero_vecadd_demo.sio` (job 16129):
  **erro máximo 0** em 1024 elementos. A limitação que resta é estreita e
  nomeada: a superfície pública `kernel fn` do `souc build --backend gpu`
  ainda não baixa corpos com load/store (emite `ret`); quem emite kernel
  real hoje é o pipeline Kretikos programático. Unir os dois é trabalho de
  compilador, fora deste framework. Artefato gerado versionado em
  `stdlib/hetero/kernels/vec_add.kretikos.ptx`.
- **GEMM ladrilhado**: `stdlib/hetero/kernels/gemm_tiled_f32.ptx`
  (tiles 16x16 em shared memory) verificado (erro 0) e medido no job
  16135: **2.394 GFLOP/s** no GEMM 1024 na L4, +42% sobre o naive
  (1.681). Ainda longe do pico FP32 da L4; próximos passos seriam tiles
  maiores, vetorização f32x4 e WMMA bf16 — nesta ordem, medindo cada um.
  Diagnóstico de PTX agora usa `cuModuleLoadDataEx` com JIT error log
  (`ce_module_log`), que apontou os dois erros reais desta sessão
  (cvta 32/64-bit e endereçamento shared) sem chute.
- **AMD/HIP**: probe pronto e compilado (`stdlib/hetero/hip_probe.sio`,
  `examples/hetero_hip_probe_demo.sio`, degrada sem ROCm). Execução segue
  bloqueada pelo serving na R9700 (job 12798, >4h no ar); submeter o demo
  quando a GPU liberar — não preemptar serving por probe.
- **GB10/DGX Spark**: NO-GO com evidência. `--target aarch64-linux` é
  aceito pelo front-end, mas o backend nativo falha (rc=11) até num
  hello-world, no build local do madaros medido em 2026-10-23. Sem ELF
  aarch64 Linux não há binário Sounio no Spark; os jobs x86_64 seguem
  válidos nos nós x86. Reavaliar quando o backend aarch64 Linux pousar
  (há `codegen_arm64_macos`, que é Mach-O, não serve).

## 9. Fase 3 — GraphIR + CostModel (2026-10-23)

Entregue: `stdlib/hetero/graph.sio` (GraphIR com capacidade fixa e guards
ADR-004; nós declaram op/dtype/FLOPs/bytes e dimensão fatiável),
`stdlib/hetero/cost.sio` (CostModel sobre capacidades medidas + planner de
cadeia por DP exata sobre (nó, device), com custo de transferência por link
medido; colocação pura é candidata dentro da DP — "não repartir" emerge dos
números). Testes puros em `examples/hetero_cost_demo.sio` (ALL PASS):
especialistas heterogêneos repartem [A,B,A]; link lento (0,01 GB/s) força
colocação; link não medido nunca é atravessado em silêncio; cadeia homogênea
não se parte.

Validação previsto-vs-executado (gate da fase): CostModel calibrado apenas
no GEMM tiled medido em n=1024 (2.394 GFLOP/s, job 16135) prevê, via
`examples/hetero_predict_demo.sio`:

| n | Previsto (ms/iter) | Executado (ms/iter) | Erro |
|---|---|---|---|
| 512 | 0,112 | 0,121 (2.204 GFLOP/s) | +8,0% |
| 1024 | 0,897 | 0,895 (2.397 GFLOP/s) | -0,2% (calibração) |
| 2048 | 7,176 | 7,318 (2.347 GFLOP/s) | +2,0% |

Executado no job 16171 (L4, kernel tiled, eventos CUDA). **Gate ±25%
aprovado com folga** (pior erro fora da calibração: +8%): neste intervalo
de shapes o throughput do kernel tiled é quase constante, e o CostModel
prevê — não apenas repete a calibração. O mesmo job registrou o bench CPU
do Tensor2 estourando o limite de tempo; medido à parte (n=256, 1 iter,
workspace): 0,034 GFLOP em 21,8 s de CPU ≈ **0,0015 GFLOP/s**. Esse número
mede o caminho de acesso elemento-a-elemento com conversão de dtype do
`ops.sio` (o oráculo de referência), **não** um kernel GEMM CPU — kernel
CPU com loops tipados diretos é trabalho de backend (Fase 4) e terá seu
próprio número medido. Registrar o acessor como "throughput de CPU" no
DeviceCaps seria um erro de calibração de três ordens de grandeza.

## 10. Riscos declarados

- AMD via SPIR-V emitido: caminho de execução não provado (spike Fase 2).
- GB10/SM121 e memória unificada podem exigir alocador e kernels próprios.
- Com 1 GPU por nó e rede entre devices, partição fina perde para
  single-device; o planner precisa poder concluir isso (seção 2.5).
- Paridade homogênea com vLLM/PyTorch é o gate mais duro; sem ela, o
  diferencial público fica restrito ao heterogêneo.

## 11. Fase 4 — Runtime distribuído v1 (2026-10-23, OBSERVED)

Entregue: executor multi-device single-host (`stdlib/hetero/runtime.sio`),
kernel CPU de loops tipados (`stdlib/tensor2/cpu_kernel.sio`) e transporte
multi-nó (`stdlib/hetero/net_worker.sio`, sockets libc via dlopen/dlsym,
protocolo v1: header i64 `[77, rows, n, k]` + shards f64). Demos:
`examples/hetero_runtime_demo.sio`, `examples/hetero_worker_demo.sio`.

Medido (job SLURM 16336, L4 + CPU do nó r770; job 16343 cross-node):

- Split GPU+CPU correto: n=128 forçado 64/64, `max_diff = 0` vs referência
  CPU f64; split forçado 768/256 em n=1024 também exato vs GPU-only.
- Kernel CPU (streaming i-k-j, unroll 4): **0,16 GFLOP/s**; variante
  register-blocked i-j-k mediu **0,11** (job 16336) e foi revertida com o
  motivo registrado no código. O gargalo é o intrínseco por acesso, não o
  algoritmo — número honesto de CPU escalar single-thread neste host.
- GPU via runtime (staging f64→f32 + transferências incluídos): 202
  GFLOP/s em n=512, 459 GFLOP/s em n=1024 (kernel puro: 2.394). A
  conversão de staging domina em n grande; é custo do runtime v1,
  registrado como tal.
- **Planner recusou repartir**: com CPU a 0,16 vs GPU a ~200+ GFLOP/s, a
  fatia proporcional da CPU é < 1 linha → veredito `PLACEMENT gpu-only`.
  Split forçado (768/256): makespan 5,08 s vs 4,68 ms GPU-only;
  eficiência heterogênea = 0,42/(459+0,16) ≈ **0,0009**. Publicado como
  medido: neste par GPU+CPU, repartir é 1000× pior e o planner conclui
  isso sozinho — o comportamento projetado na seção 2.5, agora executado,
  não só planejado. O ganho heterogêneo real do parque é GPU↔GPU entre
  nós/gerações, não GPU↔CPU.
- **Cross-node provado** (job 16343): worker em `cpuops-t560-proxmox`
  (10.0.5.252), coordenador em `dl380-dl380-proxmox`, shard remoto exato
  (`max_diff = 0`). Loopback idem. Degradação graciosa v1: worker
  inalcançável → coordenador computa o shard localmente e o resultado
  continua verificado (`WORKER PASS (fallback)`).

Aberto (não declarar fechado): (i) fiação GraphIR→executor (o `plan_chain`
da Fase 3 ainda não dirige o runtime; v1 executa decisões de shard para
nós GEMM); (ii) replano ao vivo com sessão sobrevivendo à queda de um
worker (v1 tem fallback local, não replano); (iii) gate de paridade
homogênea ≥90% — pertence à Fase 5 (serving), onde há baseline vLLM;
(iv) staging f32 sem conversão elemento-a-elemento (packing vetorizado)
para recuperar os 2.394 GFLOP/s do kernel dentro do runtime.

## 12. Fase 4 — fechamento do runtime (2026-10-23, OBSERVED)

Os quatro itens abertos da seção 11 foram atacados; três fechados com
medição, um reclassificado com número:

- **Fiação GraphIR→executor** (`rt_exec_chain` em `runtime.sio`): o
  `plan_chain` (DP da Fase 3) dirige a execução nó a nó, com tabela de
  binding pool→executor (0=GPU local, 1=CPU local, 2=worker remoto),
  ativações em ping-pong nos masters do host e caminho **residente no
  device** quando todos os nós caem na GPU (sobe x+pesos uma vez, GEMMs
  device→device, desce só a ativação final; `ChainResult.resident`
  registra qual caminho rodou). Job 16392 (L4): cadeia 3×GEMM n=256
  planejada toda na GPU, executada residente (`resident=1`, kernels
  0,060/0,041/0,038 ms), `max_diff = 0` vs referência f64. Previsto (só
  computação) 0,042 ms vs executado 0,477 ms: o restante é transferência
  de borda pinned (2 MB f64 para dentro, 0,5 MB para fora) — o modelo de
  custo não cobra borda host↔device e isso fica registrado como
  limitação conhecida do CostModel v1.
- **Replano ao vivo**: falha de device no meio da cadeia risca o device
  do pool (caps zerados = custo infinito), replana a sub-cadeia restante
  com a mesma DP e continua; `replans`/`failed_device`/`ran_on` tornam o
  replano auditável. Provado em loopback e **cross-node** (job 16376:
  worker no t560 morre após o nó 0; coordenador no dl380 replana e
  termina, `ran_on = 0 1 1`, `max_diff = 0`). Bug real encontrado e
  corrigido no caminho: `send` sem `MSG_NOSIGNAL` matava o coordenador
  com SIGPIPE ao escrever para o worker morto — o cliente morria em
  silêncio; agora a morte do peer é um `-1` replanejável.
- **Staging**: a conversão host f64→f32 (mesmo empacotada 2×f32/word)
  media **272 ms** num GEMM n=1024 — 98% do end-to-end. Correção:
  kernels próprios de conversão no device (`kernels/cvt_f64_f32.ptx`,
  `cvt.rn.f32.f64`/`cvt.f64.f32`) e masters em **memória pinned**
  (`cuMemHostAlloc`, `rt_master`). End-to-end n=1024 (job 16392):
  269,9 ms (conversão no host) → 8,9 ms (cvt no device, masters pageable)
  → **2,32 ms** (masters pinned), com a janela de kernel+transferências
  em 1,99 ms contra 0,895 ms do kernel puro com buffers já residentes:
  overhead do caminho homogêneo do runtime ≈ **2,6×**, medido e
  publicado — não é paridade, e a composição é conhecida (bytes f64 em
  dobro na borda + lançamentos cvt). Paridade ≥90% contra vLLM continua
  sendo gate da Fase 5, onde existe baseline de serving.
- Regressão: `hetero_runtime_demo`, `hetero_worker_demo`,
  `hetero_cost_demo`, `hetero_planner_demo`, `tensor2_demo` — todos PASS
  com o runtime final.

Com isso o runtime está fechado no escopo da Fase 4: planner decide,
executor executa (1 host, N devices, workers remotos), falha vira
replano, e cada número acima é reproduzível pelos demos citados.

## 13. Fase 5 — LLM nativo (2026-10-24, OBSERVED)

Entregue em `stdlib/llm/`: `safetensors.sio` (leitura binária via libc,
scanner de header próprio — o `json.sio` da stdlib é só emissor),
`llama.sio` (forward DecoderBlock: RMSNorm, RoPE rotate-half, atenção
GQA com KV cache, SwiGLU, embeddings amarrados opcionais, biases QKV
Qwen2 detectados pela presença do tensor) e `bpe.sio` (tokenizer
byte-level GPT-2 a partir do `tokenizer.json`: mapa byte↔unicode,
merges por rank, pré-tokenizador em scanner ASCII fiel). Pesos viram
masters f64 com as conversões IEEE reais da Fase 1; projeções são
transpostas uma vez na carga e o caminho quente é `ck_gemm`.

Provado, com números:

- Loader: tiny-random-Llama (F32) e SmolLM2-135M (BF16), shapes/dtypes/
  somas conferidos; tensor ausente reporta `found=false`, nunca quebra.
- Forward: KV token-a-token ≡ prefill de uma vez com `max_diff = 0`
  (bit-exato, mesma ordem de acumulação) e determinismo bit-exato em
  três arquiteturas/configs: tiny-random (MHA), SmolLM2-135M (GQA 9/3,
  tied), Qwen2.5-0.5B-Instruct (GQA, biases QKV, vocab 151.936).
- Tokenizer: `decode(encode(x)) == x` e ids plausíveis
  ("The capital of France is" → `[504 3575 282 4649 314]`).
- **Geração real**: SmolLM2-135M greedy a partir desse prompt produziu
  "Paris. It is the largest city in France and the capital of the
  department of department of ..." — coerente até o loop de repetição
  típico de greedy em modelo de 135M. É a prova end-to-end (tokenizer +
  loader + forward + cache) mais forte que um diff sintético.

Aberto, declarado (não chamar de gate externo ainda):

- **Gate de logits contra vLLM**: não executado. O parque serve
  Qwen2.5-Coder-3B + LoRA GRPO (adapter não público) e modelos de
  27–80B; o gate limpo exige servir o modelo-base (HF, safetensors) num
  job vLLM temporário e comparar top-k logprobs token a token. O
  pré-requisito de código (biases Qwen2) fechou hoje; falta o job.
- O forward v1 é o caminho CPU de referência (f64, 1 thread): ordem de
  2–4 s/token no 135M medido em parede de job. As projeções ainda não
  passam pelo runtime heterogêneo (GPU/workers) — é troca de backend
  atrás das mesmas funções. Kernel GEMV dedicado (pesos `[out][in]`
  para m=1) e carga paralela são os itens de performance conhecidos;
  a carga do 0.5B leva minutos (decode elemento-a-elemento).
- `llm serve` (HTTP OpenAI-compatível), amostragem (temperatura/top-p)
  e KV paginado ficam para a fatia de produto; o BPE não separa tokens
  especiais no encode e aproxima classes Unicode de letras por byte
  ≥ 0x80 no pré-tokenizador.

## 14. Fase 5 — projeções no runtime + gate externo vLLM (2026-10-24, OBSERVED)

Dois cortes sobre o núcleo da §13:

**Corte 1 — projeções roteadas pelo runtime heterogêneo.**
`rt_gemm_rect` (runtime.sio) generaliza o split GPU+CPU para GEMM
retangular (o kernel tiled já era retangular-seguro); `llama_forward_rt`
(llama.sio) roda as projeções por `ll_proj` — linhas na GPU via runtime,
restante na CPU dentro da mesma chamada; falha de GPU recomputa na CPU
e reporta status. Gate (`examples/llama_gpu_demo.sio`, job SLURM 16764,
L4): sonda anti-fallback explícita (`status=0 gpu_rows=16 cpu_rows=4`,
`diff=8e-6`), logits GPU-vs-CPU `max_diff = 16e-6`, argmax idêntico e
10 tokens greedy idênticos nos dois caminhos. Regressão pós-mudança:
`hetero_runtime_demo`, `llama_tiny_demo`, `llama_load_demo` PASS.

**Corte 2 — gate de logits contra vLLM (o item "aberto" da §13).**
Job 16784 serviu os MESMOS pesos SmolLM2-135M em vLLM 0.32.0
(`--dtype bfloat16 --enforce-eager`) e gravou top-10 logprobs +
continuação greedy para o prompt de ids `[0,504,3575,282,4649,314]`;
o comparador é Sounio puro (`examples/llama_gate_demo.sio`: nosso
forward f64 + log-softmax + parser JSON próprio). Resultado OBSERVED:

- Top-10: **conjunto idêntico (10/10 ids)**; ranks 2–9 na mesma ordem
  nos dois lados.
- Diffs de logprob por token: **0,014–0,059** (máx 0,0586).
- **Top-1 invertido**: vLLM `" the"` (−1,5085) > `" Paris"` (−1,5710);
  nós `" Paris"` (−1,5283) > `" the"` (−1,5559). Os dois candidatos
  estão a 0,03–0,06 um do outro e o deslocamento entre os forwards
  (~0,045) troca o argmax. Consequência: `greedy_text_equal=0` já no
  primeiro token gerado (" the city of Paris..." vs nosso
  " Paris. It is the largest city..."), convergindo depois para texto
  de mesmo conteúdo.
- Veredito do gate como codificado: `top1_match=0` — NÃO passa nos
  critérios estritos (top-1 + greedy). A concordância distribucional
  (top-10 completo, diffs ≤ 0,06) está medida; a divergência é do
  tamanho que separa precisão bf16 de erro sistemático pequeno.

**Veredito do teste discriminante (job 16822, `--dtype float32`,
OBSERVED)**: com o mesmo comparador, `GATE PASS` —
`top1_match=1`, `greedy_text_equal=1` (16/16 tokens), diffs de
logprob por token **0,0007–0,0066** (máx 0,0066, uma ordem de
magnitude abaixo do bf16) e ranks 0–6 do top-10 na mesma ordem
(7/8 trocam entre si com gap de 0,004). Conclusão medida: o nosso
forward f64 concorda com a referência independente (vLLM) dentro da
precisão de um serving fp32; o deslocamento de ~0,045 do bf16 era
ruído de precisão do servidor, **não** erro sistemático do forward —
os suspeitos listados acima (escala da atenção, RoPE, RMSNorm) ficam
descartados por evidência, não por inspeção. O gate externo da Fase 5
fecha assim: concordância distribucional e de argmax contra vLLM com
os mesmos pesos e os mesmos token ids.

Aberto depois disto (inalterado da §13): `llm serve` OpenAI-compatível,
amostragem temperatura/top-p, KV paginado/distribuído real, kernel
GEMV m=1, carga mais rápida (0.5B em minutos), backends AMD/GB10.

## 15. Fase 5 — `llm serve` OpenAI-compatível (2026-10-24, OBSERVED)

`stdlib/llm/serve.sio` (+ `examples/llm_serve.sio`) serve o forward
nativo por HTTP sobre os sockets de `hetero::net_worker`: processo
carrega o modelo uma vez e responde `GET /health`, `GET /v1/models`
e `POST /v1/completions`. Parser JSON próprio (o mesmo padrão do gate
da §14); `prompt` aceita string (BPE aqui, BOS prependido como nos
demos) ou array de token ids (verbatim); resposta no formato OpenAI
com `usage` contado. Sequencial, uma request por conexão, KV fresco
por request.

Gate end-to-end (curl real contra o processo compilado,
SmolLM2-135M na porta 8090 do workspace, OBSERVED):

- prompt string `"The capital of France is"`, 16 tokens greedy →
  `" Paris. It is the largest city in France and the capital of the
  department of"` — idêntico ao nosso forward de referência e ao
  greedy do vLLM fp32 da §14.
- prompt por ids `[0,504,3575,282,4649,314]`, 8 tokens → mesmo texto,
  `usage` 6/8/14.
- `temperature: 0.7` → 400 declarado ("only greedy decoding
  (temperature=0) is implemented in llm serve v1") — sampling é o
  corte separado; o servidor não finge suportar o que não faz.
- rota fora do escopo (`/v1/chat/completions`) → 404 com o mapa de
  rotas v1. Chat template não é reivindicado para modelo-base.

Limites medidos/declarados: decode é o caminho CPU f64 de referência
(~2:28 para prefill 6 + 16 tokens — lento, é o preço do oráculo f64;
o caminho `llama_forward_rt` e o kernel GEMV m=1 são os cortes de
performance seguintes); buffers de KV por request não são liberados
(sem free na v1 — processo de longa duração acumula); a porta só é
alcançável dentro do workspace — exposição via tailnet depende do
mecanismo que hoje publica o :8080 e fica fora deste corte.
Servidor de pé em tmux `llm-serve` no workspace
(`tmux kill-session -t llm-serve` para parar).

Aberto: sampling temperatura/top-p, GEMV m=1, decode via runtime
GPU no serve, carga rápida, streaming SSE.

## 16. GEMV m=1 para o decode — e o teto medido da codegen (2026-10-24, OBSERVED)

`ck_gemv` (tensor2/cpu_kernel) ataca a forma do decode (m=1) sobre o
layout transposto das projeções (`y[j] = sum_i x[i] * w[i*n+j]`):
blocos de 8 colunas acumulam em locais sobre toda a contração e `y` é
escrito uma vez — o `ck_gemm` m=1 fazia read-modify-write de `y` a
cada `i` (3 acessos/MAC contra ~1). Roteado em `ll_proj` quando
`rows == 1` e na projeção `lm_head`; formas largas seguem no GEMM de
referência. Benchmarks: `examples/llama_decode_bench.sio` (decode
greedy de contagem fixa, tempo externo) e
`examples/tensor2_gemv_bench.sio` (shapes isolados).

Provado, com números:

- **Correção**: `ck_gemv` vs `ck_gemm` bit-exato (`max_diff = 0`) nos
  shapes 576×49152 (lm_head), 576×1536 (gate) e 576×576 (qproj) —
  mesma ordem de acumulação por coluna. Sequência greedy de 24 tokens
  no modelo real idêntica antes/depois; `llama_smol_demo` PASS com
  kv-vs-prefill `max_diff = 0`.
- **Velocidade**: decode de 24 tokens (SmolLM2-135M, mesma máquina e
  binário de bench): 336 s → 269/274 s de parede total; descontada a
  carga (~68 s, idêntica), ≈11,2 → ≈8,5 s/token, **~1,3×**. Ganho
  real, mas muito abaixo do que a redução de acessos sugeria — e a
  causa foi medida, não chutada:
- **O teto é da codegen, não do algoritmo.** Micro-bench isolado:
  soma sequencial pura de 850M de `read_f64` levou 27,7 s de CPU
  (~30M iterações/s); o padrão RMW do GEMM roda a ~24M MAC/s. Um loop
  escalar f64 gerado pelo Madaros atual custa ~30 ns por iteração —
  duas ordens de magnitude acima do custo de memória. Enquanto
  `read_f64`/`write_f64` forem o único acesso (indexação direta de
  ponteiro não typecheca) e a codegen não vetorizar, **nenhum kernel
  CPU em Sounio passa de ~0,1 GFLOP/s**, qualquer que seja o shape do
  loop. O próximo salto de decode é GPU (`llama_forward_rt` no serve)
  ou codegen — registrado como insumo para as sessões do compilador,
  não como defeito deste kernel.

Aberto (inalterado): sampling, decode GPU no serve, carga rápida,
SSE, KV paginado. O `llm serve` segue greedy sobre o caminho CPU —
agora ~1,3× menos lento, com a lentidão restante atribuída e medida.

## 17. Sampling

Seeded sampling in `stdlib/llm/sample.sio`: xorshift64* PRNG
(`rng_new`/`rng_f64`), temperature via soft-max over the f64 logits, and
top-p (nucleus) by repeated max-extraction (cap 8,192 nucleus members,
renormalised over the covered mass). `temperature = 0` (or absent) stays
exactly greedy (`llama_argmax`). Not cryptographic; determinism by seed
is the feature: same seed, same completion.

`llm serve` accepts `temperature`, `top_p` (default 1.0) and `seed`
(default 42). Out-of-range values are rejected honestly: `temperature < 0`
and `top_p` outside (0, 1] return 400.

Gates (`examples/llama_sample_demo.sio`; synthetic 8-token distribution
with logits = ln p, p = [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.015,
0.005], N = 20,000 draws; OBSERVED):

- temperature 1.0: empirical [0.3996, 0.2479, 0.1467, 0.1022, 0.0503,
  0.0324, 0.0151, 0.0060], max deviation 0.0033 (threshold 0.015).
- top_p = 0.5 (nucleus {0, 1}): zero draws outside the nucleus,
  f0 = 0.6192 (expected 0.6154).
- temperature 2.0: f0/f1 = 1.264 (theory sqrt(1.6) = 1.2649).
- temperature 0: sampler output equals greedy on 8/8 steps.
- Real model, seed 777 (temp 0.8, top_p 0.95): same seed reproduces
  16/16 tokens; seed 778 differs in 15/16.

Serve end-to-end (OBSERVED): two identical seeded requests return
byte-identical text; temperature 0 and absent temperature both return
the canonical greedy continuation; temperature -0.5 and top_p 1.5
return 400; top_p 0.9 returns a distinct sampled continuation.

Limits: nucleus construction is O(vocab · members) per token — fine at
135M scale, not at large vocabularies; the PRNG is deterministic by
design and unsuitable for anything security-adjacent.

## 18. Streaming (SSE)

`"stream": true` on `POST /v1/completions` switches the reply to
Server-Sent Events: an SSE head (`Content-Type: text/event-stream`, no
Content-Length), one OpenAI-shaped chunk per generated token
(`choices[0].text` = the token's decoded bytes, JSON-escaped,
`finish_reason: null`), a final chunk with empty text and the real
`finish_reason`, then `data: [DONE]`. Validation errors stay plain
JSON: the SSE head is only sent after the request fully validates.
Without `stream` (or with `false`) the reply is byte-identical to
before.

Gates (OBSERVED, curl against the workspace server):

- Streamed greedy: 8 delta chunks arrive incrementally, spaced by the
  per-token decode time (~3.4 s at 135M on the workspace CPU), then
  the final chunk (`finish_reason: "length"`) and `[DONE]`.
  Concatenating the deltas reproduces the non-streamed canonical text
  byte for byte.
- Streamed sampling (temperature 0.7, seed 42): concatenated deltas
  equal the non-streamed completion with the same seed.
- Regression: non-streamed greedy, explicit `stream: false`, /health
  and the 400 validation paths unchanged.

Limits: client disconnect mid-stream is not detected — sends fail
silently (MSG_NOSIGNAL) and generation runs to completion; usage
statistics are only reported in the non-streamed reply.

## 19. Paged KV cache

`KvCache` no longer holds one dense buffer per layer sized to `cap`.
Positions live in blocks of 16, addressed by a block table (K and V,
one entry per layer per block, 0 = not allocated). A block is
allocated on the first write of a position inside it, so a sequence
of length `len` touches `ceil(len/16)` blocks per layer instead of
the whole cap. `kv_free` returns every block and both tables;
`kv_blocks_used` counts what is live. `ck_free` is the matching
release for a `CpuBuf`.

The serve calls `kv_free` (and frees the prompt-id buffer) at the end
of every completion, streamed or not.

Gates (OBSERVED):

- Tiny-random and SmolLM2-135M: token-by-token through paged blocks
  versus prefill-at-once, max abs diff 0. On SmolLM2 (30 layers,
  len 40, cap 512) both paths allocate 180 blocks, which is
  `2 * layers * ceil(40/16)`. Bytes touched: 4,423,680 against
  47,185,920 for the dense cap-sized cache. After `kv_free`, the
  count is 0 and both tables are null.
- `llama_smol_demo` on the paged cache: kv-versus-prefill max diff 0,
  argmax 7042, greedy text the canonical continuation. PASS.
- Serve, five identical greedy requests (`max_tokens` 8) on the paged
  build: each returns ` Paris. It is the largest city in`.

RSS of the workspace server, VmRSS, steady per-request growth
(OBSERVED, same prompt):

- Dense cache, nothing freed: +39,232 KiB/request (two identical steps).
- Paged cache, `kv_free` per request: +38,028 KiB/request (four identical steps).
- Returned to the heap: 1,204 KiB/request.

The residual ~37 MiB/request is unfreed forward temporaries and the
per-request HTTP buffers. Paging does not claim to remove that; it
removes the KV accumulation, and it stops allocating the unused tail
of `cap` (the 10× gap above is that tail, at cap 512).

## 20. Forward temporaries

`ll_forward_impl` releases every buffer at its last use (`ck_free`):
the two RMSNorms, Q/K/V, the attention output, the MLP activations,
the score row, the hidden state and the last-position copy. The
returned logits stay alive — the caller owns them. `llm serve` frees
the previous logits before each decode step, the one-token id buffer,
and the final logits at the end of the request, together with the
paged KV.

Gates (OBSERVED, SmolLM2-135M):

- `llama_smol_demo`: kv-versus-prefill max abs diff 0, argmax 7042,
  canonical greedy text. PASS. The paged-KV demo on tiny-random is
  unchanged (max diff 0, 12 blocks, free returns to 0).
- Serve RSS, five greedy requests, `max_tokens` 8, same prompt as §19.
  Steady growth is +212 KiB/request (four steps of 212 KiB, one of
  208). The §19 build, which freed only the KV, grew +38,028 KiB/request.
  Text of all five replies: ` Paris. It is the largest city in`.
  Wall time for the eight tokens stayed in the same band (~49–51 s).

What remains is the request-scoped HTTP buffers (`sv_handle` does not
release its read buffer, the JSON body or the decoded text). That is
the +212 KiB, not the forward.

## 21. Decode on the GPU

The tiled GEMM rounds its GPU rows down to a multiple of 16, so an
m=1 projection — every decode step — never left the CPU. `rt_load_gemv`
loads `kernels/gemv_f32.ptx` (one thread per output column) and a
resident-weight table keyed by the host pointer: each matrix is
uploaded and converted to f32 once, and a decode step ships the
activation only. `rt_gemm_rect` takes that path when `m == 1` and the
kernel is loaded. A device failure recomputes the row on the CPU.
Shapes that are wider than one row and shorter than a tile are
unchanged: they stay on the CPU.

Bench `examples/llama_gpu_decode_bench.sio`, job 17039, partition
`bench`, node `gpuorangefs-r770-proxmox`, NVIDIA L4. SmolLM2-135M,
the canonical prompt, greedy. OBSERVED:

- Probe, one m=1 projection: `gpu_rows=1`, status 0, max abs diff
  3e-6 against the f64 GEMM.
- Prefill of 6 rows (below the tile, so both paths are CPU): max abs
  diff 0.
- CPU decode, 8 steps, 14.097 s, 1.762 s/token (steps 1.713–1.824 s).
- GPU decode after one warm step, 7 steps, 3.472 s, 0.496 s/token
  (steps 0.487–0.503 s). 3.55× against the CPU on the same node.
  The warm step, which pays the weight upload, was 0.666 s.
- Token sequence identical, text ` Paris. It is the largest city in`.

The whole step is timed, attention and norms included; they stay on
the CPU. The process did not return after printing the result (CUDA
teardown). Job 17039 was cancelled once the log was complete, so the
L4 was not left occupied.

## 22. Load width and GPU serve

bf16 tensors are unpacked four values per aligned 64-bit word inside
`st_load_f64`. A fused convert-and-transpose was measured and was
slower (51.9 s against 44.2 s), so the projection transpose stays a
second pass.

`llama_load` of SmolLM2-135M on the workspace, `examples/llama_load_bench.sio`,
OBSERVED: 44.218 s before, 37.706 s after. The prefill argmax on the
canonical prompt is 7042 either way.

`llm_serve_rt` uses `llama_forward_rt` when the runtime is ready, so
decode projections take the resident GEMV. Without PTX paths,
`llm_serve` is the CPU server. Job 17080 on the L4
(`gpuorangefs-r770-proxmox`): the process printed `gpu=1`, and
`POST /v1/completions` with `max_tokens` 4 returned
` Paris. It is`. The request, prefill included, took 8.748 s.
The per-token decode figure remains the one in §21.

## 23. Small training

`stdlib/learn/adam_f64.sio` is Adam over a flat f64 buffer. The
16-wide surgical Adam stays the kernel-preserving variant.
`batch_shares` gives integer batch sizes proportional to measured
throughput, largest remainder, summing to the total. That is the
batch side of heterogeneous data parallel; it does not move
gradients between nodes.

`examples/tiny_train_demo.sio` on the workspace, OBSERVED, one run:

- Linear map, squared error, 40 Adam steps from zero: loss 0.002304,
  and the second run prints the same value.
- Rank-1 LoRA on a frozen zero matrix: loss 1.020100 → 0.010557.
- Bigram table, vocab 4, cross-entropy, 30 passes from zero:
  1.386294 → 0.004871. The first number is ln 4, the uniform start.
- Throughputs 2 and 1, total batch 9: shares 6 and 3.

`TRAIN PASS`.

## 24. A one-block GPT

`examples/tiny_gpt_demo.sio` is one decoder block: embedding, causal
attention, residual, ReLU FFN, residual, logits. Vocabulary 16, hidden
8, sequence 8, one head, FFN width 16. The loss is the mean
cross-entropy of the next token at the first seven positions of
`0,1,2,3,4,5,6,7`. The gradient is a central finite difference. Adam
is `learn::adam_f64`. The seed is the initial table, not a clock.

OBSERVED on the workspace, two calls in one process, 125 s wall:

- `nll_first` 2.772595, against ln 16 = 2.772589.
- `nll_last` 0.000611, and the second call 0.000611.
- `GPT PASS`.

The smoke script type-checks this file and does not run it: the
two-call process is over a minute.

## 25. LoRA on the frozen head

`llama_forward_ln` writes the final RMSNorm vector into a caller
buffer and returns the frozen head's logits. `llama_forward` and
`llama_forward_rt` are unchanged.

`examples/llama_lora_demo.sio` puts a rank-1 adapter on that vector:
`z[j] + b[j] * dot(a, ln)`. `a` (length 576) is a deterministic
table; `b` (length 49152) starts at zero, so the first loss is the
frozen SmolLM2. The body is not written. Two Adam steps. Calling `ln`
directly from the long loss function segfaulted; the same call in a
small helper does not.

OBSERVED, workspace, canonical prompt, five next-token terms:

- nll_before 4.057791.
- nll_after 2.794695, and the second call 2.794695.
- Embedding prefix sum unchanged (−3.099 at ×1e3).
- `LORA PASS`.

A third step raised the loss (6.92). The run stops at two. The smoke
script type-checks the file and does not run it.


## 26. On-device forward (2026-10-25, OBSERVED)

The path that uploaded an activation, ran one projection and copied the vector back measured 0.496 s/token on job 17039. The forward in this section keeps the weights, the KV cache and the activations on the L4. The host receives the logits of the last position and nothing else in between.

A decode step, and a prefill of at most eight tokens, runs as follows.

- RMSNorm, one block per row. The reciprocal square root is `rsqrt.approx` plus two Newton steps, in f32.
- GEMM. For M of 1 to 16, one skinny launch: each column reads its element of B once and reuses it across the rows. For M of 16 or more, when M, N and K are multiples of the tensor-core tile, the kernel is `mma.sync.aligned.m16n8k16` with bf16 operands and an f32 accumulator. Any other shape uses the skinny kernel in chunks of 16 rows. The bf16 file is converted to resident f32 once, at upload.
- RoPE angles are computed on the host with `math::pure`, so they match the CPU forward, and then uploaded. The rotation is a device kernel.
- Causal attention is one launch per head: the score, the softmax and the AV product. The score matrix stays in shared memory.
- Residual, SwiGLU, the output projection, and the same contract on all 30 layers. The LM head runs once, on the last row.

`llama_forward` is unchanged. `llm serve` calls `llama_forward_dev` once `dev_open` has succeeded. Each request calls `dev_reset`, so the device KV cache does not cross requests.

Gate. Job 17500, partition `bench`, node `gpuorangefs-r770-proxmox`, SmolLM2-135M, token ids 0, 504, 3575, 282, 4649, 314. The clock is `date +%s%N` on the job script, one stamp per line.

- Prefill argmax, CPU and device: 7042.
- Greedy continuation, eight tokens, both paths: 7042, 30, 657, 314, 260, 3995, 2240, 281.
- Maximum absolute logit difference on that prefill: 3.37e-5 (the program printed 33715 at a scale of 1e9).
- After one warm step, seven forwards span 515629394 ns, which is 0.0737 s/token. The threshold is 0.100 s. An earlier binary on the same prompt, with attention on a single thread per head and a fresh driver lookup on every launch, measured 0.1067 s/token (job 17499, 746740015 ns over the same seven forwards). That binary did not meet the threshold. Job 17500 is the cut that did.

The MMA module loaded (`mma=1`). The measured steps have M of 1 and of 6, so they used the skinny kernel. The tensor-core kernel is the dispatch for a tile-aligned M of 16 or more; this gate does not exercise it.

The device prefill of the six prompt tokens, outside the per-token window, took 74747954 ns.


## 27. Decode captured as a CUDA graph (2026-10-25, OBSERVED)

Job 17500 left a correct on-device decode at 0.0737 s/token, and the seven timed steps were flat. The next cut captures that M=1 body once and replays it.

The legacy NULL stream cannot be captured, so launches that belong to the graph go to a non-blocking stream (`cuStreamCreate` with flag 1). `cuStreamBeginCapture` / `cuStreamEndCapture` / `cuGraphInstantiateWithFlags` / `cuGraphLaunch` sit next to `ce_launch_raw`. Each captured launch keeps its own argument cells, so a later launch does not overwrite an earlier node's parameters.

What still changes on every token stays outside the graph. The host writes the RoPE angles for the current position with `math::pure` and uploads that small table. The position itself lives in the device symbol `g_pos`. The eager prefill passes the position as a kernel argument. The captured decode passes −1, and the kernel then loads `g_pos`, which is updated before each replay. The prefill (M=6) is not part of the graph. The first decode step captures; the timed steps only replay. `llm serve` takes the same split: a prompt longer than one token stays eager, and each generated token replays.

M=1 GEMM is one thread per output column. A tile of A is staged in shared memory and reused across the columns. M above 1 keeps the previous skinny kernel.

Gate. Job 17506, partition `bench`, node `gpuorangefs-r770-proxmox`, SmolLM2-135M, ids 0, 504, 3575, 282, 4649, 314. The clock is `date +%s%N` on the job script.

- Prefill argmax, CPU and device: 7042.
- Greedy continuation, eight tokens, both paths: 7042, 30, 657, 314, 260, 3995, 2240, 281.
- Maximum absolute logit difference on that prefill: 3.37e-5 (33715 at a scale of 1e9).
- The timed lines print `GRAPH=1`.
- After one warm step (the capture), seven replays span 122755231 ns, which is 0.0175 s/token. The threshold for this cut is 0.010 s. The measured step is under the 0.0737 s of job 17500 and above 0.010 s. The threshold is not moved. This cut does not pass it.

The node returned to idle.


## 28. One kernel for the whole forward (2026-10-25, OBSERVED)

Job 17506 replayed a graph of about 500 nodes at 0.0175 s/token. The two M=1 GEMMs tried on that cut, one reloading A and one staging a tile of A in shared memory, landed on the same number, so the floor was the graph, not the inner loop. SmolLM2-135M resident as f32 is about 540 MB. At the L4's 300 GB/s that traffic is 1.8 ms if each weight is read once. The threshold for this cut is 0.005 s/token, about three times that ceiling and three times under the graph.

One launch does the prefill and the decode, for M from 1 to 8. The host passes the token ids and, at the end, reads the f64 logits of the last position. There is no device-to-host copy in between. Each thread owns a stripe of four columns and walks K, so each element of B is loaded once and reused across the M rows. The K loop is unrolled by four. N and K of this model are multiples of four. The grid is 58 blocks of 256 threads, one block per SM of the L4, which is what lets a software grid barrier return. A sense-reversing atomic barrier separates the phases that need the whole vector: after the embedding, and inside each layer around the RMSNorm, the QKV projection, the attention, the output projection, the post-norm, the gate and up projections, the SwiGLU, the down projection and the residual. Block 0 owns the RMSNorm, the RoPE, the KV write, the attention, the residual and the SwiGLU. Every block owns the GEMV. A CTA barrier sits after the RoPE and before the KV copy, and again after the copy and before the attention, so one warp does not copy a K element another warp has not rotated yet. The RoPE table for 512 positions is built once at upload with `math::pure` and stays resident. The kernel indexes `(position + row)`. The host does not recompute a sine or a cosine on the measured step. The logits are converted to f64 in the kernel. `llama_forward` on the CPU is unchanged and is the reference. If this module does not load, `dev_open` returns 0 and the measured step fails. The graph of job 17506 is not a fallback.

Gate. Job 17553, node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. Partition `bench` has no nodes after the reboot of this node at 2026-10-25T13:21:10Z (the node came back in `gpu-orangefs` only), so the job was pinned on `gpu-orangefs` with the same resources as job 17506. SmolLM2-135M, ids 0, 504, 3575, 282, 4649, 314. The clock is `date +%s%N` on the job script.

- Prefill argmax, CPU and device: 7042.
- Greedy continuation, eight tokens, both paths: 7042, 30, 657, 314, 260, 3995, 2240, 281.
- Maximum absolute logit difference on that prefill: 3.68e-5 (the program printed 36789 at a scale of 1e9).
- The timed lines print `RESIDENT=1`.
- The device prefill of the six prompt tokens took 82773670 ns. That span has no threshold.
- After one warm step, seven forwards span 406812284 ns, which is 0.0581 s/token. The threshold is 0.005 s. The measured step is above the 0.0175 s of job 17506 and above 0.005 s. Effective traffic is about 9 GB/s against a 300 GB/s ceiling of 1.8 ms. The threshold is not moved. This cut does not pass it.

The node returned to idle.


## 29. Streaming the resident GEMV (2026-10-25, OBSERVED)

Job 17553 read each weight once and matched the CPU tokens, at 0.0581 s/token, about 9 GB/s. The ceiling for 540 MB at 300 GB/s is 1.8 ms. The threshold stays 0.005 s. The miss was the inner loop: four columns per thread, 32 live accumulators, and a chain of addresses, so the next load waited on the previous one. Eight warps per SM do not fill the L4.

The same single launch still does the prefill and the decode, M from 1 to 8. Each thread owns one column. Sixteen addresses along K are computed before any of those loads, then issued together. The B value is reused across the M rows, so each weight element is still read once. At most eight accumulators are live, one per row. The entry carries `.maxnreg 64`, placed between the parameter list and the body, which is the form this driver's assembler accepts. The grid is 116 blocks of 128 threads: two blocks on each of the 58 SMs, which is what fits under that cap with margin. A larger grid is not launched. The sense-reversing grid barrier is unchanged. The block still waits after RoPE before copying K, and again before attention. `llama_forward` on the CPU is unchanged. If the module does not load, the measured step fails. The graph is not a fallback.

Gate. Job 17557, node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. Partition `bench` still has no nodes, so the job was pinned on `gpu-orangefs`, as in job 17553. SmolLM2-135M, ids 0, 504, 3575, 282, 4649, 314. The clock is `date +%s%N` on the job script.

- Prefill argmax, CPU and device: 7042.
- Greedy continuation, eight tokens, both paths: 7042, 30, 657, 314, 260, 3995, 2240, 281.
- Maximum absolute logit difference on that prefill: 3.69e-5 (the program printed 36926 at a scale of 1e9).
- The timed lines print `RESIDENT=1`.
- The device prefill of the six prompt tokens took 45636532 ns. That span has no threshold.
- After one warm step, seven forwards span 300957314 ns, which is 0.0430 s/token. The threshold is 0.005 s. The measured step is under the 0.0581 s of job 17553 and above both the 0.0175 s of job 17506 and the 0.005 s threshold. Effective traffic is about 13 GB/s. The threshold is not moved. This cut does not pass it.

The node returned to idle.


## 30. One lean kernel per layer (2026-10-25, OBSERVED)

Job 17557 kept every weight read inside one launch and matched the CPU tokens at 0.0430 s/token. `.maxnreg 64` on that fat entry was the spill. The grid barrier also forbade launching more blocks than fit at once, and the GEMV itself does not need a cross-block reduction.

The decode is no longer one launch of all 30 layers. Each layer is one launch and the head is another: 31 launches on one stream, one context sync at the end, no device-to-host copy in between. The prefill of six tokens is six passes of that same kernel, with the KV cache growing. The GEMV is M=1: four coalesced columns per thread (`ld.global.cs.v4.f32`), four accumulators, and eight K addresses computed before those loads. There is no `.maxnreg`. RoPE, attention, the residual and SwiGLU stay in the layer launch. The grid barrier runs only before a phase that needs the whole vector. The block still syncs after RoPE and before attention. After the module loads, the host calls `cuOccupancyMaxActiveBlocksPerMultiprocessor` and launches `occupancy × 58` blocks of 128 threads. That is the resident grid, and not more. If the occupancy is 0, or the module does not load, the measured step fails. The graph and the 0.0430 s kernel are not fallbacks.

Gate. Job 17583, node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. Partition `bench` still has no nodes, so the job was pinned on `gpu-orangefs`. SmolLM2-135M, ids 0, 504, 3575, 282, 4649, 314. The clock is `date +%s%N` on the job script.

- The driver reported `occ=12` for the layer and the head. The launch was 696 blocks of 128 threads.
- Prefill argmax, CPU and device: 7042.
- Greedy continuation, eight tokens, both paths: 7042, 30, 657, 314, 260, 3995, 2240, 281.
- Maximum absolute logit difference on that prefill: 3.69e-5 (the program printed 36926 at a scale of 1e9).
- The timed lines print `RESIDENT=1`.
- The device prefill of the six prompt tokens took 224820637 ns. That span has no threshold.
- After one warm step, seven forwards span 424803544 ns, which is 0.0607 s/token. The threshold is 0.005 s. The measured step is above the 0.0430 s of job 17557 and above the 0.0175 s of job 17506. Occupancy 12 is real, and most of those blocks own no column: a projection of width 576 has 144 working threads. The extra blocks only wait on the barrier. The threshold is not moved. This cut does not pass it.

The node returned to idle.


## 31. K split across the 58 SMs (2026-10-26, OBSERVED)

Job 17583 asked the driver for occupancy and launched `12 × 58 = 696` blocks. A projection of width 576 has 144 threads with a column, and the rest waited on the barrier. That step was 0.0607 s/token. The ceiling for 540 MB at 300 GB/s is still 1.8 ms. The threshold stays 0.005 s.

This cut does not ask occupancy. The grid is 58 blocks of 256 threads, one block per SM. Each GEMV gives block `b` the rows of K in `[b * span, min(K, (b + 1) * span))`, with `span = ceil(K / 58)`. A thread owns four coalesced columns and four accumulators, and writes `partial[b, col]`. A grid barrier follows. Every block then sums the 58 partials of its own column stripe into the destination. A slice that starts past K writes zeros. There is no `atomicAdd` and no `.maxnreg`. The partial buffer is `58 * 49152` f32, about 11 MB, allocated once.

One launch is one token, M=1. The prefill of six tokens is six of those launches, and the KV cache grows. The thirty layers and the head run inside the launch. The host syncs once after the token, and does not copy a vector back in between. The block still syncs after RoPE and before the KV copy, and again before attention. If the module does not load, or the barrier does not return, the measured step fails. The graph of job 17506 and the 0.0430 s kernel of job 17557 are not fallbacks.

Gate. Job 17616, node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. Partition `bench` still has no nodes, so the job was pinned on `gpu-orangefs`. Jobs 17613 and 17614 held the node; this job waited and did not cancel them. SmolLM2-135M, ids 0, 504, 3575, 282, 4649, 314. The clock is `date +%s%N` on the job script.

- The timed lines print `RESIDENT=1 grid=58`.
- Prefill argmax, CPU and device: 7042.
- Greedy continuation, eight tokens, both paths: 7042, 30, 657, 314, 260, 3995, 2240, 281.
- Maximum absolute logit difference on that prefill: 1.60e-5 (the program printed 16026 at a scale of 1e9). That span has no threshold.
- The device prefill of the six prompt tokens took 96452947 ns. That span has no threshold.
- After one warm step, seven forwards span 296776154 ns, which is 0.0424 s/token. The gaps between the seven `RESIDENT=1 grid=58` lines and the end of the window run from 31.5 ms to 51.5 ms as the cache grows, so the mean is not one slow step. The threshold is 0.005 s. The measured step is under the 0.0607 s of job 17583, next to the 0.0430 s of job 17557, and above the 0.0175 s of job 17506. Effective traffic is 12.7 GB/s against the 1.8 ms ceiling. The threshold is not moved. This cut does not pass it.

The node returned to idle.


## 32. Attention on one warp per head (2026-10-26, OBSERVED)

Job 17616 split K across 58 SMs and matched the CPU tokens at 0.0424 s/token. The seven timed gaps rose from 31.5 ms to 51.5 ms, about 3.3 ms for each extra token of context. The only work that grows with the position is attention. It ran on block 0, and only on the nine threads with `tid < heads`. Each of those threads walked `d = 0..63` with a dependent global load, three times, for every past position, on all 30 layers. Nine chains cannot hide L2 latency. The other 57 blocks waited on the grid barrier for that walk. The GEMV was not the slope.

This cut keeps one launch per token and the grid of 58 blocks of 256 threads. It does not ask occupancy and it does not set `.maxnreg`. Attention on block 0 is one warp per head. Warps 0–7 take heads 0–7, and warp 0 takes head 8 on the next pass. Each lane holds two elements of Q (`dh = 64`), reads coalesced K, and a `shfl.sync.down` sums the dot. The max, the exponential and the sum recompute that dot. The V pass writes the head. Grouped-query attention is unchanged: head `h` reads KV head `h / 3`. The block barrier after RoPE and before the KV copy stays, and another block barrier sits after attention so warp 0 finishes head 8 before the grid barrier. Q, K and V write three distinct partial regions, `58 * 49152` f32 each, so the three splits share one grid barrier and the three reductions share the next. Gate and up do the same. The K loop of the split computes four addresses before the four vector loads. If the module does not load, or the barrier does not return, the measured step fails. The graph and the 0.0424 s kernel are not fallbacks.

Gate. Job 17633, node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. Partition `bench` still has no nodes, so the job was pinned on `gpu-orangefs`. Job 17632 held the node; this job waited and did not cancel it. SmolLM2-135M, ids 0, 504, 3575, 282, 4649, 314. The clock is `date +%s%N` on the job script.

- The timed lines print `RESIDENT=1 grid=58`.
- Prefill argmax, CPU and device: 7042.
- Greedy continuation, eight tokens, both paths: 7042, 30, 657, 314, 260, 3995, 2240, 281.
- Maximum absolute logit difference on that prefill: 1.69e-5 (the program printed 16936 at a scale of 1e9). That span has no threshold.
- The device prefill of the six prompt tokens took 23428724 ns. That span has no threshold.
- After one warm step, seven forwards span 32483054 ns, which is 0.00464 s/token. The seven gaps sit between 3.3 ms and 4.9 ms. They do not climb with the cache. The threshold is 0.005 s. The measured step is under the 0.0424 s of job 17616, under the 0.0175 s of job 17506, and under 0.005 s. Effective traffic is 116 GB/s. The 1.8 ms ceiling is still below this step. The threshold is not moved. This cut passes it.

The node returned to idle.


## 33. A grid barrier only where the SMs cross (2026-10-26, OBSERVED)

Job 17633 passed 0.005 s/token at 0.00464 s, 116 GB/s. The ceiling for 540 MB at 300 GB/s is 1.8 ms. That is the threshold of this cut.

The kernel had 14 grid barriers inside the 30-layer loop and three more in the head, about 420 per token. Each one opened with `membar.gl` on every thread. Q, K, V, O, gate, up and down have N at most 1536. The reduction strides columns by `58 * 256 * 4`, so only block 0 writes those vectors. The grid barrier after those reductions published nothing to the other SMs.

The barrier that remains starts with the block `bar.sync`. Thread 0 then does one `membar.gl` and the atomic arrival. The last arriver writes the sense. The other threads wait on the volatile load. There is no per-thread `membar.gl`. Four grid barriers in the layer become a block `bar.sync` on block 0 only: after the QKV reductions, after the O reduction, after the gate and up reductions, and after the down reduction. The other blocks skip to the next grid barrier and do not touch the partial buffer in between. The grid barrier after the f64 head reduction is gone. The host `ce_sync` waits for the kernel. `ATTS`, the RMS publications, the split barriers and the head split stay. The block barriers after RoPE and after the warp attention stay. If the module does not load, or a barrier does not return, the measured step fails. The 0.00464 s kernel is not a fallback.

Gate. Job 17669, node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. Partition `bench` still has no nodes, so the job was pinned on `gpu-orangefs`. The node was idle at submit. SmolLM2-135M, ids 0, 504, 3575, 282, 4649, 314. The clock is `date +%s%N` on the job script.

- The timed lines print `RESIDENT=1 grid=58`.
- Prefill argmax, CPU and device: 7042.
- Greedy continuation, eight tokens, both paths: 7042, 30, 657, 314, 260, 3995, 2240, 281.
- Maximum absolute logit difference on that prefill: 1.69e-5 (the program printed 16936 at a scale of 1e9). That span has no threshold. It is the same difference job 17633 printed.
- The device prefill of the six prompt tokens took 21926800 ns. That span has no threshold.
- After one warm step, seven forwards span 31181338 ns, which is 0.00445 s/token. The seven gaps sit between 3.0 ms and 5.2 ms. The threshold is 0.0018 s. The measured step is under the 0.00464 s of job 17633 and above 0.0018 s. Effective traffic is 121 GB/s. The per-thread fence and the five block-local barriers were not the 4.6 ms. The threshold is not moved. This cut does not pass it.

Job 17669 completed in 47 s and released the allocation. Job 17670, a different job, then took the node.


## 34. Eight B rows in flight (2026-10-26, OBSERVED)

Job 17669 removed the per-thread fence and four grid barriers and moved from 0.00464 s to 0.00445 s/token. The logits did not change. The barrier was not the 4.5 ms. In `gemv_split` the four `ld.global.cs.v4.f32` were consumed by sixteen `fma`s before the next group's addresses existed. On the head each thread owns 48 columns and a K slice of 10 rows, so that load-use chain repeats. The ceiling for 540 MB at 300 GB/s stays 1.8 ms. That is the threshold of this cut.

`gemv_split` now copies each B row with `cp.async.cg.shared.global` of 16 bytes into a CTA buffer of `8 * 256 * 16` bytes. Eight groups are committed before the first `fma`. `cp.async.wait_group 4` releases the first four rows, `wait_group 0` releases the next four, and each ready row is read with `ld.shared.v4.f32`. A stays a cached `ld.global.ca.f32`, four values at a time. The copy is not issued for `k >= k1`. A slice shorter than eight rows, and the tail of a longer slice, still uses the one-row load. The rest of the launch is the kernel of job 17669: one token, grid 58 by 256, warp attention, three partial regions, the thread-0 fence. There is no `.maxnreg`. If the module does not load, the measured step fails. The 0.00445 s kernel is not a fallback.

Gate. Job 17691, node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. Partition `bench` still has no nodes, so the job was pinned on `gpu-orangefs`. Job 17688 held the node; this job waited and did not cancel it. SmolLM2-135M, ids 0, 504, 3575, 282, 4649, 314. The clock is `date +%s%N` on the job script.

- The timed lines print `RESIDENT=1 grid=58`.
- Prefill argmax, CPU and device: 7042.
- Greedy continuation, eight tokens, both paths: 7042, 30, 657, 314, 260, 3995, 2240, 281.
- Maximum absolute logit difference on that prefill: 1.69e-5 (the program printed 16936 at a scale of 1e9). That span has no threshold. It is the same difference jobs 17633 and 17669 printed.
- The device prefill of the six prompt tokens took 23218168 ns. That span has no threshold.
- After one warm step, seven forwards span 33124293 ns, which is 0.00473 s/token. The seven gaps sit between 3.3 ms and 5.2 ms. The threshold is 0.0018 s. The measured step is above the 0.00445 s of job 17669 and above 0.0018 s. Effective traffic is 114 GB/s. The trip through shared memory did not fill the HBM. The threshold is not moved. This cut does not pass it.

Job 17691 completed and released the allocation. A later job then held the node.
