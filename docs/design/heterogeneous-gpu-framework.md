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


## 34. An sm_89 image written in Sounio (2026-10-26, OBSERVED)

Job 17669 is the fastest resident forward that matches the tokens, at 0.00445 s/token. That kernel is still PTX text. The driver compiles it. This cut does not replace that kernel and does not move the 0.0018 s ceiling.

The generator is a Sounio program. It writes an ELF image for sm_89 from bytes measured once. ptxas 12.8.61 and nvdisasm 12.8.55 produced three cubins of the same stamp kernel, differing by the immediate 0x89abcdef, 0x00000001 and 0x11223344. Each file is 2600 bytes. The only bytes that changed were file offsets 1428 through 1431, and those bytes were the immediate, little-endian. ELF e_machine is 190. e_flags is 5834073. `.text.stamp` is 256 bytes at offset 1408. The second instruction is `IMAD.MOV.U32 R5, RZ, RZ, imm`, and the immediate is byte 4 of that instruction.

`sm89_stamp_image` copies the measured container and writes the immediate at offset 1428. The value in the probe is 0x01020304 (16909060). It was not one of the three specimens. The RDNA encoder and the AGX encoder return an empty image. No field has been measured on those machines. `ce_module_image` checks the ELF magic and passes the pointer to `cuModuleLoadData`. The probe does not call ptxas.

Gate. Job 17714, node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. Partition `bench` still has no nodes, so the job was pinned on `gpu-orangefs`. The node was idle at submit. Binary md5 `25c74fc128324e5aff396c77b8045544`.

- `SM89 nbytes=2600 ok=1`. `RDNA` and `AGX` report `ok=0`.
- `BYTES=16909060`.
- `LOAD=1`, `FUNC=1`, `LAUNCH=0`.
- `STAMP=16909060`.
- The program printed `SILICON PASS` and the script printed `EXIT:0`.

Job 17713 printed the same stamp and then did not return. The script killed it at the timeout (`EXIT:124`). Job 17714 is the same image with `exit(0)` after the pass. The node is idle after 17714.

This cut does not run SmolLM2. The load, the fma, the store, the barrier and the shuffle do not yet have an isolated field. The 0.0018 s threshold is not moved. This cut does not pass it.

A Metal specimen on the M5 Max compiled to AIR (`air64_v29`). That is not the GPU encoding. The AGX table stays empty.



## 35. Three sm_89 fields, measured by difference (2026-10-26, OBSERVED)

Section 34 patched one mov immediate. This cut asks which other bytes are a field. The specimen assembler is ptxas 12.8.61, target sm_89. The probe does not call it.

An FFMA whose addend is 3.0, against the same FFMA with addend 4.0, produces two cubins of 2728 bytes. One byte differs, inside the u32 at file offset 1508. That u32 is the f32 encoding itself: 0x40400000 and 0x40800000. A uniform shuffle was deleted by the assembler: the two cubins were identical. Keeping the lane id, and storing at `out[tid]`, delta 1 against delta 4 differs by one byte at file offset 1478, 0x20 against 0x80. Delta 2 on the non-indexed form wrote 0x40 in that same byte of its SHFL instruction. The byte is `delta << 5`. `BAR.SYNC` ids 0, 1, 2 and 3 move two bytes. File offset 1430 is 0x00, 0x40, 0x80, 0xc0, which is `id << 6`. File offset 2506, outside `.text`, is 0x10, 0x20, 0x30, 0x40, which is `(id + 1) << 4`. Swapping which register is stored moved the mov immediates and left the store register field where it was. A global load at offset 0 against the same load at offset 4 differs in 9 bytes, and the register assignment moved. Those two are not encoded.

The probe writes an FFMA addend of 5.0, bits 0x40A00000 (1084227584). That value was not a specimen. The instruction is `2.0 * 1.0 + addend`, so the stored word must be 7.0, bits 0x40E00000 (1088421888). The shuffle image is the delta-1 container with byte 1478 set to 0x80. Thirty-two lanes run, and lane `i` must store `i xor 4`. The barrier image patches both measured bytes to the id-1 specimen.

Job 17739 launched the barrier with only the text byte patched. The module loaded, the launch returned 0, and `ce_sync` returned 715. The next `cuModuleLoadData` also returned 715. That context does not survive the instruction. Job 17740 does not launch that image.

Gate. Job 17740, node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. Partition `bench` still has no nodes, so the job was pinned on `gpu-orangefs`. The node was idle at submit. Binary md5 `bb0474f66878ca6c0385190724b6799f`.

- `BYTES=16909060` and `STAMP=16909060`.
- `FMA_BITS=1084227584`, `FMA_LOAD=1`, `FMA=1088421888`.
- `SHFL_BYTE=128`, `SHFL_LOAD=1`, `SHFL=4`.
- `BAR_BYTE=64`, `BAR_META=32`, `BAR_META_LOAD=1`, `BAR_META_LAUNCH=0`, `BAR1=16909060`.
- `RDNA ok=0`, `AGX ok=0`.
- The program printed `SILICON PASS` and the script printed `EXIT:0`.

The node is idle.

Two AIR files from the M5 Max, metal 32023.921, are 3312 bytes and differ in 7. The four-byte group at file offset 2862 is `c8080901` against `d81c2929`, which is not the source immediate. The AGX table stays empty. The R9700 was not used. Job 12798 still holds `dl380`.

This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.



## 36. The 128-bit load and the cache byte (2026-10-26, OBSERVED)

Section 35 patched fields that already sat inside an instruction. This cut asks where a vector load keeps its address, and whether the cache mode is a byte.

ptxas 12.8.61, `--opt-level 0`, sm_89. A `ld.global.cs.v4.f32` at `+0`, `+16` and `+32` does not put the displacement in the load. The `+0` kernel has no add. `+16` against `+32` differs by one byte, file offset 1652, and that byte is the displacement: `0x10` and `0x20`. The instruction is `IADD3 R2, P0, R4, imm, RZ`. The load that follows is `LDG.E.EF.128`, with no displacement of its own. The four cache modifiers on a scalar load, same registers `R0` and `[R2]`, stay on one instruction at the same program counter. `cs` disassembles as `LDG.E.EF` and the byte at file offset 1754 is `0x0e`. `lu` is `LDG.E.LU` and that byte is `0x3e`. `ca` (`LDG.E.STRONG.SM`) and `cg` (`LDG.E.STRONG.GPU`) also change the previous byte, `0x19` to `0xb9` and `0xf9`. There is no formula past those four rows.

The probe copies the `+16` container and writes displacement 48 (`0x30`). That value was not a specimen. The input words are `100 + i`. A 16-byte load from offset 48 must return 112, 113, 114, 115. Offset 16 would have returned 104 through 107, and offset 32 would have returned 108 through 111. The cache image is the `cs` container with byte 1754 set to `0x3e`. It loads the word 42. The same word coming back shows that the encoding is accepted. It does not show that the cache policy changed.

Gate. Job 17757, node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The node was idle at submit. Binary md5 `16cfdc7c2e952fc59016b86e39cd94cf`.

- `LDG_IMM=48`, `LDG_LOAD=1`, `LDG=112 113 114 115`.
- `CACHE_BYTE=62`, `CACHE_LOAD=1`, `CACHE_LAUNCH=0`, `CACHE_SYNC=0`, `CACHE=42`.
- The mov, the FFMA 7.0, the shuffle and the barrier still match. `SILICON PASS`, `EXIT:0`.

The node is idle.

A Metal binary archive on the M5 Max, dumped through the Metal framework, is 12768 bytes for each of two immediates. 595 bytes differ, in 49 groups, including hash text. The four-byte group is again `c8080901` against `d81c2929`, the same non-identity word the AIR file held. The AGX table stays empty. Job 12798 still holds `dl380`, so the R9700 was not used.

This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.



## 37. The IADD3 register byte (2026-10-26, OBSERVED)

Constants fold, and the allocator renames. A store of one immediate against a store of another, at opt-level 0, changed the immediate and left the store on R0. Naming the add's destination r3 or r5 produced identical cubins. The register number is not the PTX name.

Memory operands do not fold. sub.u32 of two loaded words, default optimisation, is IADD3 R7, R0, -R7, RZ against IADD3 R7, -R0, R7, RZ. Two bytes differ. At opt-level 0 the same subtract is IADD3 R0, R0, -R2, RZ against IADD3 R0, R2, -R0, RZ, and the two bytes are 00 02 against 02 00. Byte 2 of the instruction is the destination: 0 for R0, 7 for R7. Byte 3 is source 0 and byte 4 is source 1: 0 for R0, 2 for R2, 7 for R7. The negate is not that byte. On the tight kernel, moving the negate from source 1 onto source 0 changes byte 7 from 0x80 to 0x00 and byte 9 from 0xe0 to 0xe1.

The probe loads 10 and 3. The unpatched image must store 7. The two negate bytes, written as measured, must store 3 - 10 as a u32, which is 4294967289. A third image sets only the destination byte from 7 to 0. The store still names R7. R7 then still holds the second load, so the stored word must be 3.

Gate. Job 17767, node gpuorangefs-r770-proxmox, --gres=gpu:1, --mem=16G, --cpus-per-task=4. The node was idle at submit. Binary md5 7ec42d72df5b8b6351c0daed478a7b73.

- SUB_DST=7, SUB_LOAD=1, SUB=7.
- SUB_SWAP=4294967289.
- SUB_DEST_LOAD=1, SUB_DEST_LAUNCH=0, SUB_DEST_SYNC=0, SUB_DEST=3.
- The earlier stamp, FFMA, shuffle, barrier, 128-bit load and cache byte still match. SILICON PASS, EXIT:0.

The node is idle.

metal-objdump does not read the M5 binary archive: it is not a Mach-O. The two archives still contain no raw immediate. The AGX table stays empty. Job 12798 still holds dl380.

This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.


## 38. Source, store and the small load offset (2026-10-26, OBSERVED)

Section 37 showed that byte 2 of IADD3 is the destination. The same subtract kernel names the other operands.

IADD3 R7, R0, -R7, RZ has byte 4 equal to 7, the second source. The two loads are LDG.E R0, [R2] and LDG.E R7, [R2+4]. Byte 2 is 0 and 7. Byte 3 is 2 on both, the address register. Byte 5 is 0 and 4, the displacement. One other byte, at position 13, also differs between those two loads. The store is STG.E [R4], R7. Byte 4 is 7. An earlier store, STG.E [R2], R5, matches this word except byte 3 (2 against 4) and byte 4 (5 against 7).

Three images, each changing one byte of the kernel that stores 7 for inputs 10 and 3.

- Source 1 set from 7 to 0. The subtract is R0 minus R0. The stored word must be 0.
- The store's data byte set from 7 to 0. The subtract still writes R7. The store reads R0, which still holds the first load. The stored word must be 10.
- The second load's displacement byte set from 4 to 0, and byte 13 left alone. Both loads read the first word. The stored word must be 0.

Gate. Job 17790, node gpuorangefs-r770-proxmox, --gres=gpu:1, --mem=16G, --cpus-per-task=4. The node was idle at submit. Binary md5 93a5c8b4e4bdddf6ead76cc5a68df98c.

- SRC_BYTE=0, SRC_LOAD=1, SRC_SYNC=0, SRC=0.
- STG_BYTE=0, STG_LOAD=1, STG_SYNC=0, STG=10.
- OFF_BYTE=0, OFF_LOAD=1, OFF_SYNC=0, OFF=0.
- SUB=7, SUB_SWAP=4294967289 and SUB_DEST=3 still hold. The earlier stamp, FFMA, shuffle, barrier, 128-bit load and cache byte still match. SILICON PASS, EXIT:0.

The node is idle.

This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.


## 39. Store address and the FFMA destination (2026-10-26, OBSERVED)

The subtract kernel keeps the input pointer in R2 and the output pointer in R4. Its store is STG.E [R4], R7. Byte 3 of that instruction is 4. The FFMA that stores 2.0 times 1.0 plus the addend is FFMA R5, R5, R0, imm. Byte 2 is 5, the same slot as the IADD3 destination.

Two images. The store's address byte is set from 4 to 2. The output buffer is cleared first. The input words are 10 and 3. R2 still holds the input pointer, so the result 7 must land on the first input word, the second input word must stay 3, and the output must stay 0. The FFMA image keeps the addend at 5.0 and sets byte 2 from 5 to 0. The multiply-add writes R0. The store still reads R5, which the mov before the FFMA set to 2.0. The stored bits must be 1073741824.

Gate. Job 17797, node gpuorangefs-r770-proxmox, --gres=gpu:1, --mem=16G, --cpus-per-task=4. Job 17793 held the CPUs on that node and did not hold the GPU. This job shared the node and did not cancel 17793. Binary md5 b0ff6cd3f5bbf107914527eae89bdbdd.

- ADDR_LOAD=1, ADDR_SYNC=0, ADDR_IN=7, ADDR_IN1=3, ADDR_OUT=0.
- FDEST_LOAD=1, FDEST_SYNC=0, FDEST=1073741824.
- The earlier stamp, FFMA 7.0, shuffle, barrier, 128-bit load, cache byte, subtract, negate swap, destination 3, source 0, store data 10 and load offset 0 still match. SILICON PASS, EXIT:0.

17793 was still running after this job ended. The node was not idle.

This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.


## 40. An sm_89 atlas, and one assembled add (2026-10-26, OBSERVED)

The corpus is 1328 instructions from 43 cubins, read with nvdisasm 12.8.55. Thirty-one families. Bytes 0 and 1 stay fixed inside a family. Bytes 13 and 14 move inside that family and are the schedule. The assembler writes the opcode and the fields, and leaves the schedule bytes of the slot it replaces.

| family | bytes 0, 1 | fields |
| --- | --- | --- |
| MOV rr | 02 72 | dst at 2, src at 4, RZ is 255 |
| MOV imm | 02 78 | dst at 2, u32 at 4 |
| IMAD.MOV imm | 24 74 | dst at 2, u32 at 4, byte 3 is 255 |
| IMAD.MOV c[] | 24 76 | dst at 2, constant offset/4 at 5 |
| LDG.E | 81 79 | dst at 2, address at 3, displacement at 5, byte 4 is 4 |
| STG.E | 86 79 | address at 3, data at 4, displacement at 5 |
| IADD3 rr | 10 72 | dst at 2, src0 at 3, src1 at 4, negate-src1 is 128 at byte 7 |
| IADD3 imm | 10 78 | dst at 2, src at 3, immediate at 4 |
| SHFL.BFLY | 89 7f | dst at 2, src at 3, delta shifted 5 at byte 6 |
| BAR.SYNC | 1d 7b | id shifted 6 at byte 6 |
| S2R tid | 19 79 | dst at 2 |
| EXIT | 4d 79 | schedule only |

`atlas_iadd3_lo(7, 0, 0, 0)` builds `10 72 07 00 00 00 00 00`. That word is not one of the specimens. The specimens with no negate are `R0, R0, R2` and `R7, R0, R7`. This one is `R7, R0, R0`. It replaces the subtract in the kernel whose loads are 10 and 3. The schedule half of that slot stays. The sum is 20.

Gate. Job 17813, node gpuorangefs-r770-proxmox, --gres=gpu:1, --mem=16G, --cpus-per-task=4. The node was idle at submit. Binary md5 576acd012f4080061fe9aa9c04b4e72b.

- ASM_DST=7, ASM_LOAD=1, ASM_SYNC=0, ASM=20.
- The earlier field patches still match. SILICON PASS, EXIT:0.

The node is idle.

The negate bit is bit 63 and this assembler does not form it. AMD and AGX are not in the atlas. This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.


## 41. The whole IADD3 word, negate included (2026-10-26, OBSERVED)

Section 40 built only the low 8 bytes of an add and left the schedule half in the slot. The negate bit was not formed.

`atlas_iadd3_lo` now sets bit 63 when negate is requested. That bit is byte 7 equal to 128. `atlas_iadd3_hi` builds the other 8 bytes. Two schedule rows exist in the corpus, byte 13 equal to 0xca and to 0xc6. `atlas_stg_lo` builds the low half of STG.E from the address register and the data register.

Job 17817 wrote the add R7, R0, R7 with the 0xc6 schedule into the subtract slot. The loads were 10 and 3. The stored word was 3, the second load, still sitting in R7. The add did not commit. 0xc6 is not a substitute for 0xca in this slot. The threshold of that experiment is not a sum.

The same words with byte 13 equal to 0xca, which is the schedule already measured on this slot:

- R7, R0, R7, no negate. 10 + 3 = 13.
- R7, R0, R0, negate on source 1. 10 - 10 = 0. The negate bit was formed by the assembler.
- STG.E [R4], R0 built by `atlas_stg_lo(4, 0)`. The subtract still writes R7. The store reads R0, which holds 10.

Gate. Job 17819, node gpuorangefs-r770-proxmox, --gres=gpu:1, --mem=16G, --cpus-per-task=4. The node was idle at submit. Binary md5 9702f778de0dc1c7431aa11d2c58436f.

- ADD_BYTE=202, ADD_LOAD=1, ADD_SYNC=0, ADD=13.
- NEG_BYTE=128, NEG_LOAD=1, NEG_SYNC=0, NEG=0.
- STG_ASM=0, STGASM_LOAD=1, STGASM_SYNC=0, STGASM=10.
- ASM=20 and the earlier field patches still match. SILICON PASS, EXIT:0.

The node is idle.

AMD and AGX are not in the atlas. This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.


## 42. The optimized immediate move is IMAD.MOV (2026-10-26, OBSERVED)

An optimized `mov.u32` is not opcode `02 78`. ptxas 12.8.61 at default opt emits `IMAD.MOV.U32 Rd, RZ, RZ, imm`, opcode bytes `24 74`, destination at byte 2, byte 3 equal to 255, the immediate at bytes 4–7. Opcode `02 78` is the opt-level 0 form. Its own schedule row is `00 0f 00 00 00 de 3f 00`.

`atlas_mov_imm_lo` matches that opt-level 0 low word. Splicing it over the subtract slot does not deliver the immediate. Job 17831 used the `e2 0f` schedule row and stored 0. Job 17833 kept the low word and used the subtract slot's `0xca` schedule; the store returned 3, the second load, still sitting in R7. The move did not commit. Binary md5 of that run: 19ebdaae4a5a44ba2ce45170618fb443. The node was idle afterwards.

`atlas_imad_mov_lo(5, 42)` builds the whole low word `24 74 05 ff 2a 00 00 00`. The specimen in the container was immediate 7, destination R5. 42 was not in that cubin. The schedule half of the slot is left as the cubin had it, `ff 00 8e 07 00 e2 0f 00`, which is the half the following `STG.E [R2.64], R5` was compiled against.

Gate. Job 17846, node gpuorangefs-r770-proxmox, --gres=gpu:1, --mem=16G, --cpus-per-task=4. The node was idle at submit. Binary md5 e177a37bc1bc1865344378fac8d50adf.

- IMAD_IMM=42, IMAD_LOAD=1, IMAD_SYNC=0, IMAD=42.
- MOV_SLOT=3, the opt-level 0 opcode in the subtract slot, recorded and not treated as a pass.
- ADD=13, NEG=0, STGASM=10, ASM=20 and the earlier field patches still match. SILICON PASS, EXIT:0.

The node is idle.

AMD and AGX are not in the atlas. This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.


## 43. Four instructions from one schedule encoder (2026-10-26, OBSERVED)

`atlas_sched` places the control field in the high 8 bytes: stall at bits 41..44, yield at bit 45, the read barrier at 46..48, the write barrier at 49..51, the wait mask at 52..57. A barrier value of 7 is the unset slot. The measured pair 0xca / 0xc6 is stall 5 versus stall 3, with the other control fields held. Stall 5 commits in the subtract slot. Stall 3 does not.

The operand residue under that field stays the measured leftover. The published fixed-bit pattern for IADD3 does not match this cubin, so it is not what the encoder writes.

Job 17876 replaced, in one image, both `LDG.E` words, the `IADD3` word and the `STG.E` word. The loads are the measured pair: R0 at displacement 0, stall 4; R7 at displacement 4, stall 1. The add is R7 = R0 + R7, which is not the specimen subtract. The store reads R7. The inputs were 10 and 3.

- PIPE_BYTE=202, PIPE_LOAD=1, PIPE_SYNC=0, PIPE=13.

The same four words, with the store's data register set to R0. The specimen store reads R7, not R0. R0 still holds the first load.

- ST0_LOAD=1, ST0_SYNC=0, ST0=10.

`FFMA R5, R5, R0, 5` keeps its second factor in byte 8. The specimen factor is R0, so that byte is 0 and the field was invisible. `atlas_ffma_hi(5)` writes R5 there. R5 holds 2.0. 2 * 2 + 5 = 9.

- FSRC_LOAD=1, FSRC_SYNC=0, FSRC=1091567616, the bits of 9.0.

ADD=13, NEG=0, STGASM=10, ASM=20, IMAD=42 and MOV_SLOT=3 still match.

Gate. Job 17876, node gpuorangefs-r770-proxmox, --gres=gpu:1, --mem=16G, --cpus-per-task=4. The node was idle at submit. Binary md5 d9a1081e75609478987dab17e0a1367c. SILICON PASS, EXIT:0. The node is idle.

AMD and AGX are not in the atlas. The ELF container is still the one ptxas wrote. This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.


## 44. The kernel body, assembled (2026-10-26, OBSERVED)

The family of cubins, tight and `--opt-level 0`, keeps one residue per instruction form. Changing the register does not change it. The register number and the constant offset live in the low 8 bytes.

Residues measured that way: IMAD.MOV c[] 126746879, MOV 3840, ULDC.64 2560, LDG.E 203299072, STG.E 202381572, IADD3 with RZ as the fourth source 134209791, the same add with the negate on source 0 134210047, EXIT 58720256. Negate-on-source-0 is bit 8 of the high half. Bit 63 stays clear. That is the opposite of negate-on-source-1.

Job 17904 replaced every instruction of the subtract kernel, from the constant moves through EXIT, with those formulas. The loads, the store and the prologue match the specimen bytes. The add does not: it is R7 = −R0 + R7. The inputs were 10 and 3. −10 + 3 is −7.

- BODY_NEG=225, the negate byte.
- BODY_LOAD=1, BODY_SYNC=0, BODY=4294967289.

PIPE=13, ST0=10, FSRC=1091567616, IMAD=42 and the earlier checks still match.

The node was not idle. Job 17884, a foreign CPU-only trace with OverSubscribe=OK and no GPU, held gpuorangefs-r770-proxmox. This probe was moved to partition `all` and shared the free GPU. 17884 was not cancelled.

Gate. --gres=gpu:1, --mem=16G, --cpus-per-task=4. Binary md5 aaeec55cac927f692d4a25dc0d73fbc8. SILICON PASS, EXIT:0.

The ELF container is still the one ptxas wrote. AMD and AGX are not in the atlas. This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.


## 45. The register count is part of the image (2026-10-26, OBSERVED)

Two bytes carry `SHI_REGISTERS`. One is the low byte of `.nv.info` at file 0x330. The other is bits 24..31 of `sh_info` on `.text.stamp`, file 0x9ef in this 2728-byte image. The low byte of that `sh_info` stays 6. Across the family the count is 8, 10, 12 or 14, and those two bytes move together. `atlas_shi_info` builds the `sh_info` word.

The specimen count is 10. The highest register in that kernel is R7. Job 17906 keeps the assembled body and writes the negate into R8, which the specimen does not use. The store reads R8. Both count bytes become 11.

- SHI=11 on the host, and the specimen image still has 10 at both bytes.
- REG8_LOAD=1, REG8_SYNC=0, REG8=4294967289. That is −10+3, now through R8.

BODY=4294967289, PIPE=13, ST0=10, FSRC=1091567616 and IMAD=42 still match.

Gate. Job 17906, node gpuorangefs-r770-proxmox, idle at submit, --gres=gpu:1, --mem=16G, --cpus-per-task=4. Binary md5 46a80905dd61e7030f1b42d5524d52f0. SILICON PASS, EXIT:0. The node is idle.

The rest of the ELF container is still the one ptxas wrote. AMD and AGX are not in the atlas. This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.


## 46. The exit-offset attribute is not read by this launch (2026-10-27, OBSERVED)

`cuobjdump` 12.8.55 names the records in `.nv.info.stamp`. `EIATTR_REGCOUNT` is the register count already measured. `EIATTR_CBANK_PARAM_SIZE` is a half-value at file 0x362, 16 for the two pointers. `EIATTR_PARAM_CBANK` carries the same size in the byte at file 0x35e and the constant offset 0x160. `EIATTR_MAXREG_COUNT` is 255. `EIATTR_EXIT_INSTR_OFFSETS` is the u32 at file 0x38c. In the specimen that u32 is 160, the PC of `EXIT`. `EIATTR_KPARAM_INFO` has one record per pointer. A `R_CUDA_CONST_FIELD22_37` relocation sits on the constant field. Those names are the tool's. Only the exit offset was varied.

Job 18063 keeps the assembled body, the negate on source 0, and writes 176 into that u32. 176 is the `BRA` after `EXIT`, not an exit. The inputs were 10 and 3.

- The specimen byte at file 908 is 160. The varied image byte is 176.
- XOFF_LOAD=1, XOFF_SYNC=0, XOFF=4294967289.

The stored word is still −10+3. This launch does not consult `EIATTR_EXIT_INSTR_OFFSETS`. The parameter-size byte was left at 16 and was not varied.

REG8=4294967289, BODY=4294967289, PIPE=13, ST0=10, FSRC=1091567616 and IMAD=42 still match.

Gate. Job 18063, node gpuorangefs-r770-proxmox, idle at submit, --gres=gpu:1, --mem=16G, --cpus-per-task=4. Binary md5 e1b3aa2d44bcd3c0e0048a4adc0ee03a. SILICON PASS, EXIT:0. The node is idle.

The relocation and the parameter records are still the ones ptxas wrote. AMD and AGX are not in the atlas. This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.


## 47. The constant offset chooses the pointer (2026-10-27, OBSERVED)

`EIATTR_PARAM_CBANK` in this image is bank 2 and the value 0x100160. The low byte of that value, file 0x35c, is 96, which is the base 0x160. The parameter block is 16 bytes. The first pointer sits at `c[0][0x160]`. The second sits at `c[0][0x168]`. The constant section ends at 0x170, so the block fills the tail.

Job 18077 moved that base to 0x140 and retargeted the loads and the store with it. The module loaded. `cuModuleGetFunction` returned 0. Job 18079 repeated that image and got the same `CBANK_FUN=0`. The driver does not accept a parameter base other than 0x160 on this launch.

The same job then left the base at 0x160 and assembled the store pointer from `c[0][0x160]`, the input slot. The specimen store pointer is `c[0][0x168]`. The loads still read the input. The stored word is −10+3. A fresh output buffer was zeroed before the launch.

- Specimen byte 1509 is 90. The assembled store offset byte is 88.
- SLOT_LOAD=1, SLOT_FUN=1, SLOT_SYNC=0.
- SLOT_IN0=4294967289. SLOT_IN1=3. SLOT_OUT=0.

The first word of the input is the result. The second word of the input is untouched. The output buffer is untouched.

XOFF=4294967289, REG8=4294967289, BODY=4294967289, PIPE=13, ST0=10, FSRC=1091567616 and IMAD=42 still match.

Gate. Job 18079, node gpuorangefs-r770-proxmox, idle at submit, --gres=gpu:1, --mem=16G, --cpus-per-task=4. Binary md5 762118edb8bc2920fa870a024ae6bf1f. SILICON PASS, EXIT:0. The node is idle.

The `R_CUDA_CONST_FIELD22_37` record in `.nv.rel.action` was not varied. AMD and AGX are not in the atlas. This cut does not run SmolLM2. The 0.0018 s threshold is not moved. This cut does not pass it.


## 48. Ampere encoding on the L4, Volta refused (2026-10-27, OBSERVED)

There is no V100 on this cluster. The A5000 (sm_86) sits in job 17498, `gpu-guard-r740`, which asked for the Quadro and was allocated both GPUs and 72 CPUs, `OverSubscribe=NO`. That job was not cancelled. The L4 (sm_89) was idle. A cubin for compute capability 8.6 is accepted by 8.9. A cubin for 7.0 is not.

ptxas 12.8.61 emitted the same `mov.u32` kernel for sm_70, sm_80, sm_86 and sm_89. The immediate move `IMAD.MOV.U32 R5, RZ, RZ, 0x7` is the word `24 74 05 ff 07 00 00 00 | ff 00 8e 07 00 e2 0f 00` on sm_80, sm_86 and sm_89. On sm_70 the same word is present, one slot later, and the kernel also contains `@!PT SHFL.IDX` and `STG.E.SYS` (opcode `86 73`, not `86 79`).

`atlas_imad_mov_lo(5, 42)` replaced that low word in the sm_86 image. Job 18084, on the L4:

- AMP_LOAD=1, AMP86=42.

The f64 kernel at `--opt-level 0` has `DMUL R2, R2, R4`. The word `28 72 02 02 04 00 00 00` is the same in the sm_70, sm_80, sm_86 and sm_89 specimens. The high half of the second factor was the specimen 3.0 (`0x40080000`). The atlas wrote 5.0 (`0x40140000`). 2.0 times 5.0 is 10.0.

- DMUL_LOAD=1, DMUL=4621819117588971520, the bits of 10.0.

The sm_70 image was then loaded and not launched.

- image rc=209, VOLTA_LOAD=0.

209 is `CUDA_ERROR_NO_BINARY_FOR_GPU`. The context stayed alive: the line after it is SILICON PASS.

This is not an A5000 run and not a V100 run. The f64 rate of the L4 is not the rate of a V100. The DMUL word in the Volta specimen matches the word the L4 executed, and the Volta container was refused.

Gate. Job 18084, node gpuorangefs-r770-proxmox, idle at submit, --gres=gpu:1, --mem=16G, --cpus-per-task=4. Binary md5 a46e89b064041643bfde8eb7cc5cc63c. SILICON PASS, EXIT:0. The node is idle.

The 0.0018 s threshold is not moved. This cut does not pass it.


## 49. Both halves of the f64 multiply (2026-10-27, OBSERVED)

Job 18084 stored the bits of 10.0. That product is `0x4024000000000000`. The low 32 bits are 0, so a store of only the high half would have looked the same.

`atlas_mov_imm_lo` now forms an immediate whose top 16 bits are at least 32768. The sign of that half is written as `(hi - 65536) * 2^48`, the same adjustment the IMAD immediate already used. The four moves in front of `DMUL` were replaced:

- R2, R3 hold 1.1, bits `0x3ff199999999999a`. The low half `0x9999999a` is the case the old formula could not form.
- R4, R5 hold 2.3, bits `0x4002666666666666`.

The IEEE product of those two values is `0x40043d70a3d70a3d`. The low 32 bits are `0xa3d70a3d`.

Job 18090, on the L4, the sm_86 image:

- AMP86=42 still.
- DMUL_LOAD=1, DMUL=4612879472328641085, DMUL_LO=2748779069.
- image rc=209, VOLTA_LOAD=0.

The A5000 is still inside job 17498. There is still no V100. This is the sm_86 `DMUL` word, executed on the L4, with both halves of the product nonzero in the sense that the low half of the product is not zero. The low half of each factor was also nonzero.

Gate. Node gpuorangefs-r770-proxmox, idle at submit, --gres=gpu:1, --mem=16G, --cpus-per-task=4. Binary md5 b61ff06bd9e6ad3c99b4819de663cc5d. SILICON PASS, EXIT:0. The node is idle.

The 0.0018 s threshold is not moved. This cut does not pass it.


## 50. The DMUL source register (2026-10-27, OBSERVED)

Job 18090 multiplied 1.1 by 2.3. The second factor sat in R4. The low word of that `DMUL` was `28 72 02 02 04 00 00 00`. Byte 4 was 4.

`atlas_dmul_lo(2, 2, 2)` writes `28 72 02 02 02 00 00 00`. The factors are unchanged: R2 and R3 are still 1.1, R4 and R5 are still 2.3. The instruction now names R2 as both factors, so it squares 1.1. The IEEE square is `0x3ff35c28f5c28f5d`. The low 32 bits are `0xf5c28f5d`.

Job 18091, on the L4, the sm_86 image:

- Host byte 1524 is 2. The specimen byte at that position is 4.
- DMUL_LOAD=1, DMUL=4608128174721765213, DMUL_LO=4123168605.
- AMP86=42. image rc=209, VOLTA_LOAD=0.

If byte 4 were not the source, the product would still have been 1.1 times 2.3, which is 4612879472328641085. It was not.

The A5000 is still inside job 17498. There is still no V100.

Gate. Node gpuorangefs-r770-proxmox, idle at submit, --gres=gpu:1, --mem=16G, --cpus-per-task=4. Binary md5 64193081a24cfda9ad078a114700d90a. SILICON PASS, EXIT:0. The node is idle.

The 0.0018 s threshold is not moved. This cut does not pass it.


## 51. The other DMUL source (2026-10-27, OBSERVED)

Job 18090 multiplied 1.1 by 2.3. The low word was `28 72 02 02 04`: destination R2, first source R2, second source R4. Job 18091 changed only the second source to R2 and stored 1.1 squared. Byte 4 is that source.

`atlas_dmul_lo(2, 4, 4)` writes `28 72 02 02 04` with byte 3 also equal to 4. The factors are the same ones: R2 is 1.1, R4 is 2.3. Both sources now name R4, so the operation squares 2.3. The IEEE square is `0x401528f5c28f5c28`. The low 32 bits are `0xc28f5c28`.

Job 18097, on the L4, the sm_86 image:

- Host bytes 1523 and 1524 are 4. In job 18090 byte 3 was 2 and byte 4 was 4.
- DMUL_LOAD=1, DMUL=4617642028934585384, DMUL_LO=3264175144.
- AMP86=42. image rc=209, VOLTA_LOAD=0.

The product is not 1.1 times 2.3 and not 1.1 squared. It is 2.3 squared. Byte 3 selects the first source the same way byte 4 selects the second.

The A5000 is still inside job 17498. There is still no V100.

Gate. Node gpuorangefs-r770-proxmox, idle at submit, --gres=gpu:1, --mem=16G, --cpus-per-task=4. Binary md5 f624ce4b88439b1f353aea651519cf85. SILICON PASS, EXIT:0. The node is idle.

The 0.0018 s threshold is not moved. This cut does not pass it.


## 52. The DMUL destination (2026-10-27, OBSERVED)

Job 18097 squared 2.3 into R2. The low word was `28 72 02 04 04`. Byte 2 was the destination, 2.

`atlas_dmul_lo(4, 4, 4)` writes `28 72 04 04 04`. The sources are still both R4, so the arithmetic is still 2.3 squared. The destination is R4. The store after it still reads R2. R2 was loaded with 1.1 and nothing else writes it.

Job 18201, on the L4, the sm_86 image:

- Host byte 1522 is 4. In job 18097 that byte was 2.
- DMUL_LOAD=1, DMUL=4607632778762754458, DMUL_LO=2576980378.
- Those are the bits of 1.1, `0x3ff199999999999a`. The low half is `0x9999999a`.
- AMP86=42. image rc=209, VOLTA_LOAD=0.

The square did not reach the store. It was written to R4. Byte 2 is the destination.

The A5000 is still inside job 17498. There is still no V100.

Gate. Node gpuorangefs-r770-proxmox, idle at submit, --gres=gpu:1, --mem=16G, --cpus-per-task=4. Binary md5 d40c2ca0734fcbaf5c68a9e3aefc710c. SILICON PASS, EXIT:0. The node is idle.

The 0.0018 s threshold is not moved. This cut does not pass it.


## 53. An sm_89 image that reads no cubin (2026-10-27, OBSERVED)

Sounio allocates 2600 bytes and writes the ELF. The probe does not open a cubin. `e_machine` is 190. `e_flags` is `0x590559`. The entry is `stamp`. One parameter. The register count is 8, written by `atlas_shi_info` at file `0x330` and at file `0x96f`.

The first body loaded the pointer with `IMAD.MOV` from `c[0][0x160]`, moved 42 into R5 afterwards, and padded the section with `EXIT`. Job 18279. md5 `038cb4b304550246538ed0e7be1446d6`. `OWN_LOAD=1`. The output word was not cleared. `OWN=1073741824`, the bits of 2.0.

Job 18297 used that same image and cleared the word to 0 before launch. md5 `f13571c4e3b087233a6ce772914e413b`. `OWN=1073741824` again. The kernel wrote 2.0. The cleared 0 did not remain.

The text was then put in the order whose atlas words match a kernel that has stored 42: `IMAD.MOV` R1 from `c[0][0x28]`, `IMAD.MOV` R5 of 42, `MOV` R2 from `c[0][0x160]`, `ULDC.64` UR4 from `c[0][0x118]` at stall 1 yield 1, `MOV` R3 from `c[0][0x164]`, `STG.E` of R5, `EXIT`, then the `BRA` and `NOP` bytes that fill the 256-byte section. The word was cleared first.

Job 18304. The node was idle. md5 `0e31e400e6c96c1508a22ddce69a04cf`. `OWN_LOAD=1`, `OWN=42`, `EXIT:0`. The cleared word became 42.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.

## 54. The mnemonic list of the resident forward (2026-10-27, OBSERVED as a list)

`nvdisasm` 12.8.55 on the cubin from `ptxas` 12.8.61 of `stdlib/hetero/kernels/fwd_resident.ptx` at `a50f6285db`, target `sm_89`, entry `token_m1`. Eighty-one mnemonics. This list is the atlas scope. A name with no device number is not an encoder.

The opcode column is the low 16 bits of the first instruction of that mnemonic in this cubin, printed as the disassembler's word. It is the specimen, not a proof.

Job 18304's 42 is the only number this ELF has stored. It shows that `IMAD.MOV` of the immediate 42 reached `STG.E` through the pointer this body loaded with `MOV` from `c[0][0x160]` and `c[0][0x164]`. The other names are not proven here.

- 242 `7c10` IADD3
- 184 `7a02` MOV
- 167 `7624` IMAD.MOV.U32
- 141 `8947` BRA
- 135 `7825` IMAD.WIDE.U32
- 123 `2223` FFMA
- 122 `7981` LDG.E
- 104 `8221` FADD
- 89 `7c82` UMOV
- 77 `7a0c` ISETP.GE.U32.AND
- 57 `720c` ISETP.NE.U32.AND
- 52 `7a24` IMAD
- 48 `7820` FMUL
- 46 `7945` BSSY
- 46 `7941` BSYNC
- 42 `7944` CALL.REL.NOINC
- 41 `7b1d` BAR.SYNC.DEFER
- 39 `7986` STG.E
- 35 `7812` LOP3.LUT
- 24 `7984` LDS
- 24 `2824` IMAD.IADD
- 23 `7ab9` ULDC.64
- 23 `7ab9` ULDC
- 22 `8388` STS
- 22 `781c` PLOP3.LUT
- 20 `7919` S2R
- 20 `7948` WARPSYNC
- 17 `7981` LDG.E.64
- 16 `7f89` SHFL.DOWN
- 15 `7946` YIELD
- 14 `7890` UIADD3
- 12 `7992` MEMBAR.SC.GPU
- 12 `79ab` ERRBAR
- 12 `798f` CCTL.IVALL
- 12 `79a8` ATOMG.E.ADD.STRONG.GPU
- 12 `7981` LDG.E.STRONG.SYS
- 12 `780c` ISETP.LE.U32.OR
- 12 `7981` LDG.E.EF.128
- 12 `7981` LDG.E.STRONG.SM
- 11 `7308` MUFU.RCP
- 11 `2824` IMAD.SHL.U32
- 10 `7224` IMAD.MOV
- 10 `7227` IMAD.HI.U32
- 10 `7219` SHF.L.U32
- 9 `7950` RET.REL.NODEC
- 8 `720c` ISETP.GT.U32.AND
- 8 `7918` NOP
- 7 `7819` SHF.R.U32.HI
- 7 `7805` CS2R
- 6 `780b` FSETP.GEU.AND
- 6 `7207` SEL
- 6 `7589` SHFL.IDX
- 6 `7811` LEA
- 6 `7e24` IMAD.U32
- 5 `7302` FCHK
- 5 `7b06` I2F.U32.RP
- 5 `7305` F2I.FTZ.U32.TRUNC.NTZ
- 5 `7947` BRA.DIV
- 4 `7308` MUFU.RSQ
- 4 `780b` FSETP.NEU.FTZ.AND
- 4 `720c` ISETP.GE.AND
- 4 `7310` F2F.F64.F32
- 4 `7986` STG.E.64
- 3 `7a0c` ISETP.LT.U32.AND
- 3 `7899` USHF.R.U32.HI
- 3 `7a45` I2FP.F32.S32
- 3 `7208` FSEL
- 3 `7308` MUFU.EX2
- 3 `720c` ISETP.NE.AND
- 2 `780b` FSETP.GTU.FTZ.AND
- 2 `7221` FADD.FTZ
- 2 `7986` STG.E.128
- 1 `7311` F2I.F64.TRUNC
- 1 `8209` FMNMX
- 1 `794d` EXIT
- 1 `780c` ISETP.GT.U32.OR
- 1 `780c` ISETP.GT.AND
- 1 `7223` FFMA.RZ
- 1 `7223` FFMA.RP
- 1 `7223` FFMA.RM
- 1 `7217` IMNMX.U32

The 0.0018 s threshold is not moved. This list does not pass it.


## 55. IADD3 inside the owned image (2026-10-27, OBSERVED)

The image of job 18304 stayed the container. Four instructions were inserted before `EXIT`, and the register count became 11. The cleared buffer is two words.

Job 18313 wrote the exit-offset word as `7172 + 160 * 2^32`. That low half is `0x1c04`, not the attribute header `0x00041c04`. `cuModuleLoadData` returned `image rc=200`. md5 `419a515c26c7d313300eafb441f7ffee`. The image was not launched.

Job 18315 repaired that word to `269316 + 160 * 2^32`, which is `04 1c 04 00 a0 00 00 00`. `IMAD.MOV` of 10 and of 3 used the ordinary immediate schedule, stall 1. `IADD3` R8, R6, R7 then `STG.E` of R8 at displacement 4. `OWN=42`, `OWN_ADD=0`. md5 `61729dcbd05f1b4135f4a2bd54726c56`. The second word stayed the cleared 0.

Job 18319 kept the displacement and stored R5, which holds 42, at displacement 4. `OWN=42`, `OWN_ADD=42`. md5 `1ed58045a0083ccaa667a6d3eed0cb3a`. Byte 5 of `STG.E` is the displacement. The 0 in job 18315 was the value in R8, not a missed address.

Job 18325 gave those two immediate moves stall 5 and yield 1, and stored R8 again. `OWN=42`, `OWN_ADD=13`, `EXIT:0`. md5 `8cd5efbcff8d2f5ff8ee01a44b343ef5`. Ten plus three is 13. `IADD3` opcode `10 72` committed inside this ELF.

The resident kernel's first `IADD3` in the list of section 54 is opcode `10 7c`. That form is not this word. The row stays open. Seventy-nine of the eighty-one names still have no number from this image.

The 0.0018 s threshold is not moved. This cut does not pass it.


## 56. A load from the owned image (2026-10-27, OBSERVED)

Three instructions were inserted before `EXIT` in the image that stored 13. Slot 10 is `atlas_ldg_lo(4, 2, 8)` and `atlas_ldg_hi(4)`. The low word is `81 79 04 02 04 08 00 00`. The high word is `00 19 1e 0c 00 a8 0e 00`. The residue `203299072` is `0x0c1e1900`, so byte 1 of that half is `0x19`, the 32-bit `LDG.E` form. Stall is 4, yield is set, the read barrier is 2, and the write barrier is 7. Slot 11 is `IADD3` R0, R4, R7. R7 already holds 3. Slot 12 stores R0 at displacement 16. `EXIT` moved from pc `0xa0` to pc `0xd0`. The exit-offset word kept the low half `0x00041c04`. The high half is 208.

The host cleared 24 bytes and wrote 17 at offset 8. Offset 12 was left at 0.

The L4 was `IDLE+DRAIN` because the kill of job 18427 had failed at 17:08, and that job was already gone. The node was resumed. Pending job 18691, which had asked for the same node, started and was not cancelled. After it released the GPU, job 18697 ran. md5 `51967e8086db1300f49e1f3a1ed0f652`.

`OWN=42`, `OWN_ADD=13`, `OWN_SRC=17`, `OWN_WIDE=0`, `OWN_SUM=20`, `EXIT:0`.

The word at offset 16, which had been cleared, became 20. That is 17 plus 3. The 17 at offset 8 was still there. Offset 12 stayed 0. Read barrier 2 did not hang this launch. Stall 4 was enough for this load to be visible to the add.

Byte 4 of this load is 4. The common `LDG.E` in `token_m1` uses 6 at displacement 0. That byte was not changed in this job.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 57. The add's wait mask (2026-10-27, OBSERVED)

The load of job 18697 was left as it was. `atlas_iadd3_hi(0)` on the sum is residue `134209791` plus `atlas_sched(5, 0, 7, 7, 4)`. That 4 is the wait mask, bit 2, barrier 2, the barrier the load registers. This job uses `atlas_sched(5, 0, 7, 7, 0)`. Stall 5 stays. The mask bit is off. Host byte 1598 is 15 (`0x0f`), where the previous image had 79 (`0x4f`).

The host still writes 17 at offset 8. The probe still requires 20 at offset 16.

The L4 was `IDLE+DRAIN` because the kill of job 18758 failed at 23:14, and that job was already gone. The node was resumed so 18761 could run. Revert with `scontrol update NodeName=gpuorangefs-r770-proxmox State=DRAIN Reason="Kill task failed (JobId=18758 StepId=0)"`.

Job 18761. md5 `718355dd94e09b5ca75d37475d2f20fa`.

`OWN=42`, `OWN_ADD=13`, `OWN_SRC=17`, `OWN_WIDE=0`, `OWN_SUM=3`, `EXIT:1`.

The cleared word at offset 16 came back 3. That is the 3 already in R7, added to an R4 that was still 0. The load had not delivered 17 when the add issued. With the mask on, job 18697 stored 20 in that word. The mask bit is the wait. The required 20 was not changed to 3.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 58. Load byte 4 (2026-10-27, OBSERVED)

Job 18697 stored 20 with load byte 4 equal to 4 and the add's wait mask equal to 4. Job 18761 cleared that mask and stored 3. This job puts the mask back, `atlas_iadd3_hi(0)`, and changes only byte 4 of the load from 4 to 6. The displacement stays 8. The low word is `81 79 04 02 06 08 00 00`. Host byte 1572 is 6. Host byte 1598 is 79 (`0x4f`), the mask bit back on.

The host still writes 17 at offset 8. The probe still requires 20.

The common `LDG.E` in `token_m1` uses byte 4 equal to 6 at displacement 0. This cubin has no load with byte 4 equal to 6 and displacement 8. That is the combination launched here.

Job 18808. The node was idle, not drained. md5 `c8287b87723ad0dc25ebd00da6e34604`.

`OWN=42`, `OWN_ADD=13`, `OWN_SRC=17`, `OWN_WIDE=0`, `OWN_SUM=20`, `EXIT:0`.

The word at offset 16 is 20 again. Byte 4 equal to 6, with displacement 8 and the wait mask on, delivered the same 17. The 42 and the 13 stayed. Offset 12 stayed 0. The 17 at offset 8 was still there.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 59. Displacement 0 reads the stored 42 (2026-10-28, OBSERVED)

The load of job 18808 had byte 4 equal to 6 and displacement 8, and stored 20. This job changes byte 5 from 8 to 0. Byte 4 stays 6. The add's wait mask stays on. The low word is `81 79 04 02 06 00 00 00`. That is the shape of `LDG.E R15, [R2.64]` at pc `0x1e40` in `token_m1`, with R4 and R2 in place of R15 and the same R2. Host byte 1573 is 0.

When the load runs, slot 10, offset 0 already holds the 42 stored at slot 5. Offset 4 holds 13. Offset 8 holds 17. Offset 12 holds 0. The sum of a load from offset 0 with the 3 in R7 is 45.

`token_m1` loads through that R2 before it stores. `LDG.E R15, [R2.64]` at `0x1e40` registers read barrier 3. `FFMA R27, R10, R15.reuse, -R27` at `0x1eb0` waits on mask 8. `STG.E [R2.64], R27` at `0x1ed0` waits on nobody and registers write barrier 0. R2 is not written between the load and the store. There is no load of an address this kernel just stored. A search of twenty instructions after every `STG.E` found none whose address register was still the one just stored.

Job 19265. The node was idle. md5 `d6a9d4a5dd77d332c06735452ab7554e`.

`OWN=42`, `OWN_ADD=13`, `OWN_SRC=17`, `OWN_WIDE=0`, `OWN_SUM=45`, `EXIT:0`.

The word at offset 16, cleared before launch, came back 45. The load at displacement 0 saw the 42 written at slot 5. No extra barrier was added. The 42 and the 13 stayed. The 17 at offset 8 was still there. Offset 12 stayed 0.

Across the 104 `.reuse` tags in `token_m1`, the last byte of the instruction carries one bit per source, counting from the destination. The first source is bit 2 (value 4): 24 `IADD3` and 22 `FFMA`. The second source is bit 3 (value 8): 13 `FFMA`. The third source is bit 4 (value 16): 3 `IMAD.WIDE` and 2 `IMAD`. The sum that stored 20 reads R4 once and leaves that byte at 0. This job does not set those bits.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 60. Four loads, one sum (2026-10-28, OBSERVED)

The image no longer stores 42 and 13. It loads four words and adds them. The host writes 17, 19, 23 and 29 at offsets 0, 4, 8 and 12. The sum is 88, stored at offset 16.

Each load is `LDG.E` with byte 4 equal to 6. Displacements are 0, 4, 8 and 12. The read barriers are 2, 3, 4 and 5. The first add waits on mask 12, barriers 2 and 3, and adds R4 and R6. The second waits on mask 16, barrier 4, and adds R8. The third waits on mask 32, barrier 5, and adds R9. Register count is 12. `EXIT` is at pc `0xc0`.

Job 19267, a CPU validation with `OverSubscribe=OK` and no GPU request, held the node. It was not cancelled. Job 19268 was moved to partition `all` so it could use the free GPU.

Job 19268. md5 `c0ab579ade99a0db7a9376a43bfaa0b5`.

`OWN=17`, `OWN_ADD=19`, `OWN_SRC=23`, `OWN_WIDE=29`, `OWN_SUM=88`, `EXIT:0`.

The four inputs were still there. The cleared word at offset 16 came back 88. The four loads were in flight under four barrier ids, and each add waited on the barrier of the load it consumed.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 61. Two FFMAs (2026-10-28, OBSERVED)

The four loads stay. The adds are now `FFMA`, opcode `23 72`, the unpredicated form in `token_m1`. The low word is `35 + 114 * 256`, destination, first source, second source. The addend is byte 0 of the high half. The first addend is 255, RZ, so the product starts from zero. The second addend is R0, the running sum. Stall is 5. The first `FFMA` waits on mask 12, barriers 2 and 3. The second waits on mask 48, barriers 4 and 5.

The host writes the bits of 2.0, 3.0, 4.0 and 5.0. In f32, 2 times 3 plus 4 times 5 is 26.0. The bits of 26.0 are `0x41d00000`, which is 1104150528. An earlier draft of the expected word used 1101529088. That was a bad conversion of the same hex. It was corrected before the launch.

Job 19267 still held the node with CPUs only and no GPU. It was not cancelled. Job 19271 moved to partition `all`.

Job 19271. md5 `7f28a8a5e5b520e63752fee0f11b5d80`.

`OWN=1073741824`, `OWN_ADD=1077936128`, `OWN_SRC=1082130432`, `OWN_WIDE=1084227584`, `OWN_SUM=1104150528`, `EXIT:0`.

Those four words are 2.0, 3.0, 4.0 and 5.0. The stored word is 26.0. The inputs were not overwritten.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 62. Three products (2026-10-28, OBSERVED)

The two-product kernel left the slots after the store empty. Those slots now hold two more loads, a third `FFMA`, the store, and `EXIT`. The text stays 256 bytes. A fourth product does not fit. Eight loads, four `FFMA`s, the four setup instructions, the store and `EXIT` are eighteen instructions, and `.text` sits against the section headers.

The third pair reuses R4 and R6. The first `FFMA` has already waited on those registers. The new loads register barriers 2 and 3 again. The third `FFMA` waits on mask 12, the same mask the first one used. The second `FFMA` still waits on mask 48. `EXIT` is at pc 224.

The host writes the bits of 2.0, 3.0, 4.0, 5.0, 6.0 and 7.0. In f32, 2 times 3 plus 4 times 5 plus 6 times 7 is 68.0. The bits of 68.0 are `0x42880000`, which is 1116209152. The node was idle. No drain was cleared and no foreign job was moved.

Job 19296. md5 `98989b7019c0f14cd8e97373d7c2cd09`.

`OWN=1073741824`, `OWN_ADD=1077936128`, `OWN_SRC=1082130432`, `OWN_WIDE=1084227584`, `OWN_EXT=1086324736`, `OWN_EXT2=1088421888`, `OWN_SUM=1116209152`, `EXIT:0`.

Those six words are 2.0, 3.0, 4.0, 5.0, 6.0 and 7.0. The stored word is 68.0. The inputs were not overwritten. The third product was added to the running sum.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 63. Four products, text grown by 32 bytes (2026-10-28, OBSERVED)

Three products filled the 256-byte text. A fourth product needs two more loads and one more `FFMA`. The section headers sat against the end of `.text`, so the image grew. The first 1664 bytes stay. The section-header table and the program headers move down 32 bytes. `e_shoff` goes from `0x680` to `0x6a0`. `e_phoff` goes from `0x980` to `0x9a0`. The `.text` size and the `stamp` symbol size go from 256 to 288. The load of the constant bank plus the text grows its file size and memory size from `0x2c0` to `0x2e0`. The two program headers that point at the program-header table move from `0x980` to `0x9a0`. `.debug_frame` still carries the old length 256. The driver loaded the image anyway.

The fourth pair reuses R8 and R9. The second `FFMA` has already waited on barriers 4 and 5. The new loads register those barriers again. The fourth `FFMA` waits on mask 48. The third product still waits on mask 12. The store displacement is 32. `EXIT` is at pc 272. The register count stays 12.

The host writes the bits of 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 and 9.0. In f32, 2 times 3 plus 4 times 5 plus 6 times 7 plus 8 times 9 is 140.0. The bits of 140.0 are `0x430c0000`, which is 1124859904. The node was idle. No drain was cleared.

Job 19381. md5 `bc2dd72a3581e3d3d8b2fe65f41c1413`.

`OWN=1073741824`, `OWN_ADD=1077936128`, `OWN_SRC=1082130432`, `OWN_WIDE=1084227584`, `OWN_EXT=1086324736`, `OWN_EXT2=1088421888`, `OWN_E8=1090519040`, `OWN_E9=1091567616`, `OWN_SUM=1124859904`, `EXIT:0`.

Those eight words are 2.0 through 9.0. The stored word is 140.0. The inputs were not overwritten.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 64. Five products, text grown to 336 bytes (2026-10-28, OBSERVED)

The fourth product left the text at 288 bytes. The fifth needs two more loads and one more `FFMA`. The same move applies, counted from the original 2600-byte container: the tail after the old text shifts by 80 bytes, not 32. The file is 2680 bytes. `e_shoff` is `0x6d0`. `e_phoff` is `0x9d0`. The `.text` size and the `stamp` symbol size are 336. The load of the constant bank plus the text has file size and memory size `0x310`. `.debug_frame` still carries the length 256. The driver loaded the image.

The fifth pair reuses R4 and R6. Barriers 2 and 3 were already reused once, for 6.0 and 7.0. The third `FFMA` waits on them, and the new loads register them again. The fifth `FFMA` waits on mask 12. The fourth still waits on mask 48. The store displacement is 40. `EXIT` is at pc 320. The register count stays 12.

The host writes the bits of 2.0 through 11.0. In f32, 2 times 3 plus 4 times 5 plus 6 times 7 plus 8 times 9 plus 10 times 11 is 250.0. The bits of 250.0 are `0x437a0000`, which is 1132068864. The node was idle. No drain was cleared.

Job 19385. md5 `169164c5b241574cff84769cde6324af`.

`OWN=1073741824`, `OWN_ADD=1077936128`, `OWN_SRC=1082130432`, `OWN_WIDE=1084227584`, `OWN_EXT=1086324736`, `OWN_EXT2=1088421888`, `OWN_E8=1090519040`, `OWN_E9=1091567616`, `OWN_E10=1092616192`, `OWN_E11=1093664768`, `OWN_SUM=1132068864`, `EXIT:0`.

Those ten words are 2.0 through 11.0. The stored word is 250.0. The inputs were not overwritten. Barriers 2 and 3 survived a second reuse.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 65. Six products, text at the 128-byte alignment (2026-10-28, OBSERVED)

The fifth product left the text at 336 bytes. The sixth needs two more loads and one more `FFMA`. Counted from the original 2600-byte container, the tail after the old text shifts by 128 bytes. The file is 2728 bytes. `.text` is 384 bytes, `0x180`, a multiple of the section alignment `0x80`. `e_shoff` is `0x700`. `e_phoff` is `0xa00`. The `stamp` symbol size is 384. The load of the constant bank plus the text has file size and memory size `0x340`. `.debug_frame` still carries the length 256. The driver loaded the image.

The sixth pair reuses R8 and R9. Barriers 4 and 5 were already reused once, for 8.0 and 9.0. The fourth `FFMA` waits on them, and the new loads register them again. The sixth `FFMA` waits on mask 48. The fifth still waits on mask 12. The store displacement is 48. `EXIT` is at pc 368. The register count stays 12.

The host writes the bits of 2.0 through 13.0. In f32, 2 times 3 plus 4 times 5 plus 6 times 7 plus 8 times 9 plus 10 times 11 plus 12 times 13 is 406.0. The bits of 406.0 are `0x43cb0000`, which is 1137377280. Job 19443 held the node with CPUs only and no GPU. It was not cancelled. Job 19444 moved to partition `all`.

Job 19444. md5 `a308fd258c6191e051cfa1fea7a2fa87`.

`OWN=1073741824`, `OWN_ADD=1077936128`, `OWN_SRC=1082130432`, `OWN_WIDE=1084227584`, `OWN_EXT=1086324736`, `OWN_EXT2=1088421888`, `OWN_E8=1090519040`, `OWN_E9=1091567616`, `OWN_E10=1092616192`, `OWN_E11=1093664768`, `OWN_E12=1094713344`, `OWN_E13=1095761920`, `OWN_SUM=1137377280`, `EXIT:0`.

Those twelve words are 2.0 through 13.0. The stored word is 406.0. The inputs were not overwritten. Barriers 4 and 5 survived a second reuse.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 66. Seven products (2026-10-28, OBSERVED)

The sixth product left the text at 384 bytes. The seventh needs two more loads and one more `FFMA`. Counted from the original 2600-byte container, the tail after the old text shifts by 176 bytes. The file is 2776 bytes. `.text` is 432 bytes, `0x1b0`. `e_shoff` is `0x730`. `e_phoff` is `0xa30`. The `stamp` symbol size is 432. The load of the constant bank plus the text has file size and memory size `0x370`. `.debug_frame` still carries the length 256. The driver loaded the image.

The seventh pair reuses R4 and R6. Barriers 2 and 3 have now been registered three times. The fifth `FFMA` waits on the second registration, and the new loads register them again. The seventh `FFMA` waits on mask 12. The sixth still waits on mask 48. The store displacement is 56. `EXIT` is at pc 416. The register count stays 12.

The host writes the bits of 2.0 through 15.0. In f32, the seven products from 2 times 3 through 14 times 15 sum to 616.0. The bits of 616.0 are `0x441a0000`, which is 1142554624. Job 19443 still held the node with CPUs only and no GPU. It was not cancelled. Job 19446 moved to partition `all`.

Job 19446. md5 `1ffa61df65c4f1af08ff3d7ce911dfeb`.

`OWN=1073741824`, `OWN_ADD=1077936128`, `OWN_SRC=1082130432`, `OWN_WIDE=1084227584`, `OWN_EXT=1086324736`, `OWN_EXT2=1088421888`, `OWN_E8=1090519040`, `OWN_E9=1091567616`, `OWN_E10=1092616192`, `OWN_E11=1093664768`, `OWN_E12=1094713344`, `OWN_E13=1095761920`, `OWN_E14=1096810496`, `OWN_E15=1097859072`, `OWN_SUM=1142554624`, `EXIT:0`.

Those fourteen words are 2.0 through 15.0. The stored word is 616.0. The inputs were not overwritten. Barriers 2 and 3 survived a third registration.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 67. Eight products (2026-10-28, OBSERVED)

The seventh product left the text at 432 bytes. The eighth needs two more loads and one more `FFMA`. Counted from the original 2600-byte container, the tail after the old text shifts by 224 bytes. The file is 2824 bytes. `.text` is 480 bytes, `0x1e0`. `e_shoff` is `0x760`. `e_phoff` is `0xa60`. The `stamp` symbol size is 480. The load of the constant bank plus the text has file size and memory size `0x3a0`. `.debug_frame` still carries the length 256. The driver loaded the image.

Section 66 said barriers 2 and 3 were registered three times in the seven-product image. The instruction list registers them four times: the pairs 2/3, 6/7, 10/11 and 14/15. Barriers 4 and 5 were registered three times there. The stored 616.0 is unchanged. This image registers barriers 4 and 5 a fourth time, for 16.0 and 17.0, after the sixth `FFMA` has waited on the third registration. The eighth `FFMA` waits on mask 48. The seventh still waits on mask 12. The store displacement is 64. `EXIT` is at pc 464. The register count stays 12.

The host writes the bits of 2.0 through 17.0. In f32, the eight products from 2 times 3 through 16 times 17 sum to 888.0. The bits of 888.0 are `0x445e0000`, which is 1147011072. The node was idle. No drain was cleared.

Job 19629. md5 `fdce2620e78c5173f4762e20b0468432`.

`OWN=1073741824`, `OWN_ADD=1077936128`, `OWN_SRC=1082130432`, `OWN_WIDE=1084227584`, `OWN_EXT=1086324736`, `OWN_EXT2=1088421888`, `OWN_E8=1090519040`, `OWN_E9=1091567616`, `OWN_E10=1092616192`, `OWN_E11=1093664768`, `OWN_E12=1094713344`, `OWN_E13=1095761920`, `OWN_E14=1096810496`, `OWN_E15=1097859072`, `OWN_E16=1098907648`, `OWN_E17=1099431936`, `OWN_SUM=1147011072`, `EXIT:0`.

Those sixteen words are 2.0 through 17.0. The stored word is 888.0. The inputs were not overwritten. Barriers 4 and 5 survived a fourth registration.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 68. Nine products (2026-10-29, OBSERVED)

The eighth product left the text at 480 bytes. The ninth needs two more loads and one more `FFMA`. Counted from the original 2600-byte container, the tail after the old text shifts by 272 bytes. The file is 2872 bytes. `.text` is 528 bytes, `0x210`. `e_shoff` is `0x790`. `e_phoff` is `0xa90`. The `stamp` symbol size is 528. The load of the constant bank plus the text has file size and memory size `0x3d0`. `.debug_frame` still carries the length 256. The driver loaded the image.

The ninth pair reuses R4 and R6. Barriers 2 and 3 are registered a fifth time, for 18.0 and 19.0, after the seventh `FFMA` has waited on the fourth registration. The ninth `FFMA` waits on mask 12. The eighth still waits on mask 48. The store displacement is 72. `EXIT` is at pc 512, so the exit-offset attribute now uses two bytes: the low byte is 0 and the next byte is 2. The register count stays 12.

The host writes the bits of 2.0 through 19.0. In f32, the nine products from 2 times 3 through 18 times 19 sum to 1230.0. The bits of 1230.0 are `0x4499c000`, which is 1150926848. The node was idle. No drain was cleared.

Job 19653. md5 `0c6e3ec49cad29fd5dabdca33785d7c5`.

`OWN=1073741824`, `OWN_ADD=1077936128`, `OWN_SRC=1082130432`, `OWN_WIDE=1084227584`, `OWN_EXT=1086324736`, `OWN_EXT2=1088421888`, `OWN_E8=1090519040`, `OWN_E9=1091567616`, `OWN_E10=1092616192`, `OWN_E11=1093664768`, `OWN_E12=1094713344`, `OWN_E13=1095761920`, `OWN_E14=1096810496`, `OWN_E15=1097859072`, `OWN_E16=1098907648`, `OWN_E17=1099431936`, `OWN_E18=1099956224`, `OWN_E19=1100480512`, `OWN_SUM=1150926848`, `EXIT:0`.

Those eighteen words are 2.0 through 19.0. The stored word is 1230.0. The inputs were not overwritten. Barriers 2 and 3 survived a fifth registration.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 69. Ten products (2026-10-29, OBSERVED)

The ninth product left the text at 528 bytes. The tenth needs two more loads and one more `FFMA`. Counted from the original 2600-byte container, the tail after the old text shifts by 320 bytes. The file is 2920 bytes. `.text` is 576 bytes, `0x240`. `e_shoff` is `0x7c0`. `e_phoff` is `0xac0`. The `stamp` symbol size is 576. The load of the constant bank plus the text has file size and memory size `0x400`. `.debug_frame` still carries the length 256. The driver loaded the image.

The tenth pair reuses R8 and R9. Barriers 4 and 5 are registered a fifth time, for 20.0 and 21.0, after the eighth `FFMA` has waited on the fourth registration. The tenth `FFMA` waits on mask 48. The ninth still waits on mask 12. The store displacement is 80. `EXIT` is at pc 560. The register count stays 12.

The host writes the bits of 2.0 through 21.0. In f32, the ten products from 2 times 3 through 20 times 21 sum to 1650.0. The bits of 1650.0 are `0x44ce4000`, which is 1154367488. The bits of 20.0 are 1101004800. The bits of 21.0 are 1101529088. The node was idle. No drain was cleared.

Job 19671. md5 `e9fe9f328f5b652e61c1be01b454e719`.

`OWN=1073741824`, `OWN_ADD=1077936128`, `OWN_SRC=1082130432`, `OWN_WIDE=1084227584`, `OWN_EXT=1086324736`, `OWN_EXT2=1088421888`, `OWN_E8=1090519040`, `OWN_E9=1091567616`, `OWN_E10=1092616192`, `OWN_E11=1093664768`, `OWN_E12=1094713344`, `OWN_E13=1095761920`, `OWN_E14=1096810496`, `OWN_E15=1097859072`, `OWN_E16=1098907648`, `OWN_E17=1099431936`, `OWN_E18=1099956224`, `OWN_E19=1100480512`, `OWN_E20=1101004800`, `OWN_E21=1101529088`, `OWN_SUM=1154367488`, `EXIT:0`.

Those twenty words are 2.0 through 21.0. The stored word is 1650.0. The inputs were not overwritten. Barriers 4 and 5 survived a fifth registration.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 70. Ten products with the scoreboard off (2026-10-29, OBSERVED)

The ten-product program of section 69 is unchanged in its arithmetic. What changed is the scoreboard. Every load registers no barrier: the barrier field is 7. Every `FFMA` waits on mask 0. Stall 5 on the `FFMA` and stall 1 on the load stay. The required stored word stays the bits of 1650.0, which is 1154367488. That word was not rewritten after the run.

The node was idle. No drain was cleared.

Job 19674. md5 `a39125b5322aee6e4fb9f00c16273610`.

`OWN=1073741824`, `OWN_ADD=1077936128`, `OWN_SRC=1082130432`, `OWN_WIDE=1084227584`, `OWN_EXT=1086324736`, `OWN_EXT2=1088421888`, `OWN_E8=1090519040`, `OWN_E9=1091567616`, `OWN_E10=1092616192`, `OWN_E11=1093664768`, `OWN_E12=1094713344`, `OWN_E13=1095761920`, `OWN_E14=1096810496`, `OWN_E15=1097859072`, `OWN_E16=1098907648`, `OWN_E17=1099431936`, `OWN_E18=1099956224`, `OWN_E19=1100480512`, `OWN_E20=1101004800`, `OWN_E21=1101529088`, `OWN_SUM=0`. The probe exits 1.

Those twenty words are 2.0 through 21.0. They were not overwritten. The stored sum is 0. With the scoreboard removed, none of the ten products landed in the accumulator. The same program with the waits on, job 19671, stored 1650.0.

This cut does not separate "the load registered no barrier" from "the consumer waited on no mask." Both are off. Job 18761 already stored 3, rather than 20, when one consumer's mask was cleared and the load still registered its barrier.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it. The required word 1154367488 is not moved either.


## 71. Sixteen stall-15 moves, scoreboard still off (2026-10-29, OBSERVED)

The scoreboard stays off, as in section 70. The shape of the program changes. Twenty loads write R4 through R23, displacements 0 through 76. Sixteen moves of `c[0][0x28]` into R1 follow, each with stall 15. Then the ten products run, still with mask 0. The register count is 26. `.text` is 832 bytes. The file is 3176 bytes. `e_shoff` is `0x8c0`. `EXIT` is at pc 816. The required stored word stays 1154367488, the bits of 1650.0. It was not rewritten after the run.

The node was idle. No drain was cleared.

Job 19725. md5 `c3eeabe6938b572ca29ff52798cf8d6a`.

The twenty input words are 2.0 through 21.0, unchanged. `OWN_SUM=2985189878`. That is `0xb1ee61f6`. Read as f32, the pattern is about -6.94e-9. It is not the bits of 1650, nor of 0, nor of any partial sum of the ten products. The probe exits 1.

The same inputs with the scoreboard on, job 19671, stored 1650.0. With the scoreboard off and no extra gap, job 19674 stored 0. The sixteen stall-15 moves did not turn this program into the 1650 result.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. The required word 1154367488 is not moved either.


## 72. The container emitter replays job 19671 (2026-10-29, OBSERVED)

`emit_container` takes the instruction count, the register count, and the EXIT slot. The growth in words is `ninsns * 2 - 32`. From that one number it moves the section headers and the program headers, and it writes the section size, the `stamp` symbol size, the LOAD sizes, the two header offsets, the register-count bytes, and the EXIT pc. The measured 2600-byte container in `own_word` stays the source of the bytes that do not move.

The first caller is the ten-product program of job 19671. Thirty-six instructions. Register count 12. EXIT at slot 35. The scoreboard masks are on. The file the function produces is 2920 bytes. `e_shoff` is 1984. The EXIT pc is 560.

The required stored word is 1154367488, the bits of 1650.0. It was not rewritten before the run.

The node was idle. No drain was cleared.

Job 19732. md5 `0ad3ed7be39a5116285313f68264254d`.

The twenty input words are 2.0 through 21.0, unchanged. `OWN_SUM=1154367488`, `EXIT:0`.

That is the same word job 19671 stored, from a program whose container arithmetic is now a function of the instruction count.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. This cut does not pass it.


## 73. Four taken branches store 4 (2026-10-29, OBSERVED)

`emit_container` replayed a straight-line product. This cut names the `BRA` displacement, then launches one program whose store was fixed at 4 before the run.

ptxas 12.8.61, `--opt-level 0`, sm_89. nvdisasm 12.8.55 with `-hex`. Two unconditional backward branches:

| target, from the next instruction | low word | high word | bytes 4..7 |
| --- | --- | --- | --- |
| −16 | `0xfffffff000007947` | `0x003fde000383ffff` | `f0 ff ff ff` |
| −48 | `0xffffffd000007947` | `0x003fde000383ffff` | `d0 ff ff ff` |

Bytes 4..7 are a signed displacement in bytes. The high word is the same on both. A branch whose target is the next instruction, displacement 0, has low word `0x0000000000007947` and high word `0x003fde0003800000`. Those two high words differ by `0x3ffff`, the low 18 bits, which carry the sign of the field. The opcode bytes stay `47 79`.

The program uses displacement +16. Bytes 4..7 are `10 00 00 00`. The high word is the displacement-0 specimen, because bit 31 of +16 is clear. The two negative branches are the pair that located the field. The zero branch supplies the high word used when the sign bits are clear.

Twenty instructions. `emit_container(buf, 20, 7, 19)`. The file is 2664 bytes. `.text` is 320 bytes. `e_shoff` is 1728. The EXIT pc is 304. The register count is 7.

`IMAD.MOV` writes 0 into R0 and 1 into R4, both at stall 5 and yield 1, the schedule on which job 18325's add saw its sources. Then, four times: `BRA` over one `IMAD.MOV` of 99 into R0, then `IADD3 R0, R0, R4, RZ` at stall 5. `STG.E [R2+80], R0` and `EXIT` follow.

Four taken branches skip every 99. The four adds leave 4, and 4 is stored. Four fall-throughs each write 99 and the following add leaves 100, so the store is 100. The required word is 4. It was not rewritten after the run.

The node was idle when the job ran. No drain was cleared. Foreign job 19867 had held the same GPU and had already ended `FAILED` `2:0`. It was not cancelled.

Job 19868. md5 `672f685e54f46c5aa4a63a955001676e`.

The twenty input words are 2.0 through 21.0, unchanged. `OWN_SUM=4`, `EXIT:0`.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. The loop body is still an integer add. The text of this program is four branches, and it does not yet stay the same size when K grows.


## 74. One loop body stores the ten products (2026-10-29, OBSERVED)

The four branches of job 19868 each skip one instruction. The branch does not return, and the text grows with the trip count. This cut puts the ten products in one body. The trip count is the compare immediate 10. The required stored word is 1154367488, the bits of 1650.0. It was written before either launch and was not rewritten after the miss.

ptxas 12.8.61, `--opt-level 0`, sm_89. `ISETP.LT.U32.AND P0, PT, R0, imm`. Immediate 4 is low word `0x000000040000780c`. Immediate 8 is `0x000000080000780c`. Byte 4 is the immediate. The high word stays `0x003fde0003f01070`. A second pair compares R0 and R2. Byte 3 moves with the register: 0 for R0, 2 for R2. `PLOP3.LUT` between the compare and the branch is the same word in every specimen, low `0x000000000000781c`, high `0x003fde000070f070`. The `@P0 BRA` uses byte 1 `0x09` and the displacement field already measured. Displacement −128 is bytes `80 ff ff ff` with the negative high word `0x003fde000383ffff`.

The body loads one pair through the walking pointer R6:R7, multiplies with `FFMA` into R0 under mask 12, adds 8 to the low half of that pointer, and adds 1 to R5. R0 starts at the bit pattern 0. R6:R7 starts as a copy of R2:R3. The store uses the original R2 at displacement 80. Twenty instructions. Register count 13. The file is 2664 bytes. `.text` is 320 bytes. `e_shoff` is 1728. The EXIT pc is 304.

The node was idle. No drain was cleared.

Job 19921. md5 `682f8d1281dd3a4a998aa6d08df47429`.

The compare in that image names R0, the accumulator. After one product R0 holds the bits of 6.0, and that unsigned value is not less than 10. The branch falls through. The twenty inputs, 2.0 through 21.0, are unchanged. `OWN_SUM=1086324736`, the bits of 6.0. `EXIT:1`. The required word stayed 1154367488.

Byte 3 of `ISETP` was then set to 5, so the compare reads the counter. The required word stayed 1154367488.

Job 19925. md5 `e4620888be507dc9c8bdebf1a1fd2970`.

The same inputs. `OWN_SUM=1154367488`, `EXIT:0`. Ten products, one body. The text does not contain a copy of the product per trip.

The pointer step is a 32-bit add on the low half. This allocation returned 1650.0, so that add was enough for these eighty bytes.

Gate. Node `gpuorangefs-r770-proxmox`, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved.


## 75. One block of 576 columns (2026-10-29, OBSERVED)

The loop of job 19925 is one thread. A projection of width 576 has one column per thread in this cut. The block size is 576. K is 576. Even columns hold 1.0 and odd columns hold 2.0, and the vector is 1.0, so an even column is 576.0 and an odd column is 1152.0. Those bit patterns were fixed before the launch: 1141899264 and 1150287872. The sum of the thread indices 0 through 575 is 165600. That sum was not rewritten after the run.

`S2R` was named from ptxas 12.8.61 at opt-level 0 before the launch. `SR_LANEID` and `SR_TID.X`, same destination, differ in byte 9: `0x00` and `0x21`. `SR_CTAID.X` puts `0x25` in that byte. The high word of `SR_TID.X` is `0x00321e0000002100`. Byte 2 is the destination.

The node was idle. No drain was cleared.

Job 19962. md5 `c76fa23a73d7b0b7f5433b56c8c0692f`.

`DOT0=1141899264`, `DOT1=1150287872`, `DOT575=1150287872`. `TID0=0`, `TID1=1`, `TID575=575`. `TID_SUM=155392`. The probe exits 1. The required sum stays 165600.

Jobs 19963 and 19967 repeat the same device program and print more of the index vector. md5 of 19967 is `6c58b5bfeb35151422dd5e5445f6e40f`. `T32=32`, `T256=256`, `T512=512`. `T128=0`, `T191=0`, `D128=0`. The low half of the index sum is 77568 and the high half is 77824. Together they are 155392 again. The gap below 165600 is 10208, which is the sum of the integers 128 through 191. The two samples read inside that span are 0, and the dot at column 128 is 0.

The columns that were read outside that span match the words written down before the launch. The index sum does not.

Gate. Node `gpuorangefs-r770-proxmox`, one block of 576, `--gres=gpu:1`, `--mem=16G`, `--cpus-per-task=4`. The 0.0018 s threshold is not moved. The forward still runs the PTX kernel of job 17669.
