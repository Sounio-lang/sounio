# inbox/: propostas para o Pireus

Contrato do canal entre quem propõe e quem valida. A pasta só guarda propostas.

- **Quem escreve:** Babá do Pireus (o Grok Bot do Demetrios), por commit neste branch (`cursor/pireus-r740-arch-pilot-b43d`, PR #3646). Os nomes seguem `<YYYYMMDD>-b<lote>-<NN>.proposal.json`.
- **Quem valida:** a sessão Claude Code do Demetrios. Ela passa cada `*.proposal.json` pela admissão do Sounio (`admission.sio`, o motor que o `cycle.py validate` chama), faz o dedup por `plan_id` e pelo material (PTX), e grava os recibos ADMIT/REFUSE em `tools/pireus/continuity/receipts/`. Quem escreve aqui não cria nem edita `receipts/`.
- **Status das propostas:** toda proposta é **entrada não confiável**. Ela é dado, não ordem, e não prova nada: a admissão recalcula tudo, e o ganho só se mede no hardware, pela regra congelada em `r740/arch_gain_rule.lock.json` (mediana ≥ 50000 ppm e ci95_low > 0 contra os dois controles, direct e shuffle).
- **Execução:** nada daqui roda no r740 (nem srun, nem pausa de serviço, nem timing) sem a autorização do Demetrios dada por ele mesmo no chat do Infra Sounio. Um ADMIT não autoriza um run.

## Contexto

As propostas usam o contexto do piloto r740:

- arquivo: `tools/pireus/continuity/r740/phase-a/context.json`
- sha256 dos bytes no branch: `c49450650ce3fbfff081fb9bf6da128302fba639b3b1a011f22455a0d3bcbfdb`

É o mesmo `context` dos oito `phase-a/proposals/*.proposal.json`. Se a validação usar outro context.json, a admissão recusa com `SNAPSHOT`.

## Lote b001 (2026-11-20)

O lote tem 8 propostas kind=1, no formato exato do `proposal_template` do `cycle.py`, serializadas como o `encoded()` do `cycle.py`: chaves ordenadas, JSON compacto e `\n` no fim.

Racional. Na Phase A medida (run `20261119T134949Z-287035-12861da4a7d4763d`), os 16 pares deram NO_GAIN. O candidato mais perto do controle direct foi o sm75 `004` (lane_stride 9, lane_offset 0, load 0, layout 0, unroll 16), com medianas de −641 ppm contra direct e +24088 ppm contra shuffle. Na sm75 o padrão foi claro: load=0 com layout=0 (`000` e `004`) ficou no ruído contra direct, layout=1 custou cerca de 1% e load=1 custou de 1% a 2%. Por isso o lote fixa load=0 e layout=0 e varia lane_offset, lane_stride e unroll em torno do `004`.

Os 32 planos da Phase B em `r740/proposals.py` já cobrem essa vizinhança. Então o lote não inventa planos: ele é a fatia da Phase B com load=0 e layout=0 (9 planos), menos `(round 2, index 0)` (stride 1, offset 2, unroll 1), que é o mais longe do `004`. Nenhum dos 8 repete um plano da Phase A, e os `plan_id` são todos distintos.

| arquivo | origem Phase B | stride | offset | load | layout | unroll | plan_id | por quê |
|---|---|---|---|---|---|---|---|---|
| b001-01 | r1 i4 | 9 | 1 | 0 | 0 | 16 | 18576 | `004` com offset +1 |
| b001-02 | r2 i4 | 9 | 2 | 0 | 0 | 16 | 18704 | `004` com offset +2 |
| b001-03 | r0 i12 | 9 | 1 | 0 | 0 | 4 | 18564 | stride 9 com unroll menor |
| b001-04 | r0 i8 | 1 | 1 | 0 | 0 | 8 | 2184 | stride 1 (como o `000`), unroll alto |
| b001-05 | r1 i8 | 1 | 2 | 0 | 0 | 8 | 2312 | idem, offset 2 |
| b001-06 | r2 i8 | 1 | 3 | 0 | 0 | 8 | 2440 | idem, offset 3 |
| b001-07 | r0 i16 | 1 | 2 | 0 | 0 | 2 | 2306 | stride 1, unroll 2 |
| b001-08 | r1 i0 | 1 | 1 | 0 | 0 | 1 | 2177 | o vizinho mais próximo do direct (offset 1) |

Pré-checagem do proponente (DECLARADO, não é recibo): compilei `admission.sio` deste branch (head `19551eb5`) com `./bin/souc` na máquina do Grok Bot (elf sha256 `db3fe1ff63bab1876ee3133008a06c9df9f950c54efdcf1d1219996dffb15be9`). Esse binário reproduz byte a byte o PTX vendorizado de `000`, `004`, direct e shuffle em sm86 e sm75. Com ele, os 8 deram ADMIT, e os PTX sm86 e sm75 dos 8 são distintos entre si e dos PTX da Phase A. O recibo que vale é o que a sessão de validação gravar em `receipts/`.

Expectativa honesta: a regra pede +50000 ppm (5%) contra direct. Nenhum plano load=0/layout=0 da Phase A passou do ruído contra direct, então a chance a priori de um GAIN neste lote é baixa. O lote serve para fechar a vizinhança do `004` antes de abrir o espaço de busca (kind=2 ou uma gramática nova).


## Lote b002 (2026-11-20), kind=2

O lote tem 16 propostas kind=2, ou seja, operadores novos e não lowerings. Elas usam o mesmo `context` e a mesma serialização `encoded()` do `cycle.py`. Cada uma acrescenta só a chave `phase`.

**O que é um operador por `phase`.** A gramática está em `admission.sio` e no `OPERATOR_PROMPT` do `cycle.py`: `e_i * e_j = cd_sigma(i,j) * (-1)^(i^T B j) e_(i XOR j)`, onde B é uma matriz 4x4 sobre F2 e `phase` é o inteiro de 16 bits em que o bit `4*i+j` é `B[i][j]`. A admissão reconstrói o tensor inteiro e aceita qualquer `phase` de 0 a 65535. A novidade fica com `NOT_CLASSIFIED`, porque quem decide isso é o atlas `eisa_h.pireus_operator_atlas.v1`. Duas fases que diferem por uma matriz alternada (simétrica, com diagonal zero) dão o mesmo operador a menos de um reescalonamento de sinais da base. Por isso só importam `diag(B)` e `B+B^T`, que somam 1024 classes. O representante canônico usado aqui é o triangular superior.

**Como o lote cobre o espaço.** Sem o atlas, separei as 1024 classes por três invariantes que dá para recalcular a partir da tabela de sinais: quantos `e_i` com i > 0 têm quadrado negativo, quantos pares (i,j) não comutam e quantos triplos não associam. O terceiro deu 1848 em todas as classes: a torção bilinear não mexe na associatividade. Os dois primeiros separam 7 famílias. O sedênion (`phase=0`) é (15, 210). Os códigos que já apareceram no repositório (1128, 2, 8, 74, 198, 1129, 4, 16) caem só em (5, 90), (9, 90) e (11, 114). Ninguém tinha usado ainda (3, 114), (7, 114) e (7, 210). O lote põe 13 das 16 propostas nessas três famílias novas e só uma em cada família antiga, sempre com um representante diferente dos códigos já usados. Dentro de cada família, os representantes foram escolhidos por máxima distância de Hamming entre os códigos, começando pelo de maior peso.

Os planos de lane também foram para os extremos e para combinações que nunca rodaram. Cada `lane_stride` ímpar de 1 a 15 aparece duas vezes. `lane_offset` é uma permutação de 0 a 15, e os offsets de 4 a 15 nunca tinham sido medidos. Cada par (load, layout) aparece 4 vezes, incluindo shuffle+SoA. Unroll 16 aparece 4 vezes, e 1, 2, 4 e 8 aparecem 3 vezes cada.

| arquivo | phase | família (quadrados negativos, pares que não comutam) | stride | offset | load | layout | unroll |
|---|---|---|---|---|---|---|---|
| b002-01 | 36077 | C3 (3, 114) | 1 | 15 | 0 | 0 | 16 |
| b002-02 | 35 | C3 (3, 114) | 15 | 0 | 1 | 1 | 1 |
| b002-03 | 35840 | C3 (3, 114) | 3 | 8 | 0 | 1 | 8 |
| b002-04 | 1029 | C3 (3, 114) | 13 | 7 | 1 | 0 | 2 |
| b002-05 | 1120 | C3 (3, 114) | 5 | 12 | 0 | 0 | 4 |
| b002-06 | 36076 | C7a (7, 114) | 11 | 3 | 1 | 1 | 16 |
| b002-07 | 67 | C7a (7, 114) | 7 | 10 | 0 | 1 | 1 |
| b002-08 | 36 | C7a (7, 114) | 9 | 5 | 1 | 0 | 8 |
| b002-09 | 1160 | C7a (7, 114) | 15 | 14 | 0 | 0 | 2 |
| b002-10 | 2223 | C7a (7, 114) | 1 | 1 | 1 | 1 | 4 |
| b002-11 | 33825 | C7b (7, 210) | 13 | 9 | 0 | 1 | 16 |
| b002-12 | 1 | C7b (7, 210) | 3 | 6 | 1 | 0 | 1 |
| b002-13 | 32 | C7b (7, 210) | 11 | 13 | 0 | 0 | 8 |
| b002-14 | 36079 | C5 (5, 90) | 5 | 2 | 1 | 1 | 2 |
| b002-15 | 36046 | C9 (9, 90) | 9 | 11 | 0 | 1 | 4 |
| b002-16 | 35015 | C11 (11, 114) | 7 | 4 | 1 | 0 | 16 |

**Paridade.** Um operador kind=2 não calcula o produto de sedênions, então o `reference.hex` da sonda, que vem das saídas Cayley-Dickson das fixtures, não é a referência dele. A referência bit a bit de cada candidato sai de `r740/operator_reference.py` com a fase do candidato, sobre as mesmas 320 entradas. A aritmética é a de `numeric_fixtures.sio` e do PTX: produto, sinal, soma ascendente, sem FMA. O script se recusa a gerar qualquer coisa se a fase 0 não reproduzir as `output_bits` da Sounio palavra por palavra. A decisão de paridade continua com `material_parity.sio`.

**Expectativa honesta.** O PTX de um kind=2 é instrução por instrução o mesmo de um kind=1 com o mesmo plano de lane. Só a tabela constante `pireus_signs[16]` muda. A fase escolhe qual álgebra se calcula, não quanto ela custa. Então qualquer diferença de tempo vem do plano de lane, e os extremos de plano já perderam ou empataram nos lotes kind=1. Este lote mede, pela primeira vez no r740, operadores de famílias nunca alcançadas, com paridade bit a bit de cada um. Um ganho de 5% sobre o direct seria surpresa.

## Lote b003 (2026-11-20), kind=2 com sign_mode=1

Este é o primeiro lote da extensão 1 da gramática, `sign_mode`. As 16 propostas são kind=2 com `unroll=16` e `sign_mode=1`. Fora isso, usam o mesmo `context` e a mesma serialização do b002.

**O que `sign_mode` muda.** A chave é opcional (bit 17 da máscara de `admission.sio`, valor 131072). Ausente ou 0, nada muda: o PTX e o recibo de toda proposta anterior saem byte a byte iguais. O teste `test_sign_mode_absent_or_zero_keeps_the_b001_b002_bytes` confere isso contra `r740/frozen_inbox_outputs.json`, e `phase_a.check_engine` confere a Phase A.

Com `sign_mode=1`, o PTX não carrega `pireus_signs[j]` nem multiplica o produto por ±1 em cada termo. Ele lê uma vez, por lane, a palavra `pireus_lane_signs[k]`, em que o bit j vale 1 quando sigma(k^j, j) < 0. Em cada termo, faz `sub.rn.f64` ou `add.rn.f64` predicado por esse bit. Por termo, isso tira um `ld.const`, um `selp` e um `mul.rn`.

O resultado é bit a bit o mesmo em IEEE binary64 com arredondamento ao mais próximo: `p*(+1)` é `p` exato, e `a + (-p)` é a mesma operação que `a - p`, inclusive para zeros com sinal. Só a carga de NaN poderia diferir, e o contrato de paridade (finite-bits-nan-class-v1) não a compara.

A admissão recusa `sign_mode>1` com `SIGN_MODE` e `sign_mode=1` sem `unroll=16` com `SIGN_MODE_UNROLL`, porque o bit j precisa ser conhecido na compilação. O recibo ganha `"sign_mode":1`, e o `plan_id` soma 65536. O tensor é o mesmo, porque o operador não muda e só muda a forma de calculá-lo.

**Paridade antes da GPU.** `r740/ptx_sim.py` interpreta exatamente o texto PTX emitido: 16 threads em lockstep, incluindo `shfl.sync`, e recusa qualquer instrução que não conheça. O teste `test_sign_mode_ptx_computes_the_operator_reference_bit_for_bit` roda o PTX nos dois modos, kind=1 e três fases kind=2, sobre as 320 fixtures contra `operator_reference.py`, e exige zero divergências. Um controle que troca um bit da tabela precisa divergir. Isso prova o texto, não o hardware: a paridade material continua sendo a da sonda nas GPUs, decidida por `material_parity.sio`.

**O lote.**
- b003-01, -02 e -03 são pares exatos de b002-01, -06 e -11: mesmo operador e mesmo plano de lane, e só o `sign_mode` muda. Isso isola o efeito da extensão.
- As outras 13 são representantes canônicos novos das três famílias que o b002 abriu: 5 em (3, 114), 4 em (7, 114) e 4 em (7, 210). Foram sorteados com a semente 20261120003 entre os que ainda não tinham sido usados.
- Os planos de lane alternam (load, layout) em ciclo, e nenhum `lane_stride` aparece mais de duas vezes entre as novas.

| arquivo | phase | família | stride | offset | load | layout | origem |
|---|---|---|---|---|---|---|---|
| b003-01 | 36077 | C3 (3, 114) | 1 | 15 | 0 | 0 | par de b002-01 |
| b003-02 | 36076 | C7a (7, 114) | 11 | 3 | 1 | 1 | par de b002-06 |
| b003-03 | 33825 | C7b (7, 210) | 13 | 9 | 0 | 1 | par de b002-11 |
| b003-04 | 1123 | C3 (3, 114) | 11 | 2 | 0 | 0 | nova |
| b003-05 | 32939 | C3 (3, 114) | 9 | 2 | 1 | 0 | nova |
| b003-06 | 35849 | C3 (3, 114) | 3 | 3 | 0 | 1 | nova |
| b003-07 | 35947 | C3 (3, 114) | 7 | 14 | 1 | 1 | nova |
| b003-08 | 34029 | C3 (3, 114) | 15 | 9 | 0 | 0 | nova |
| b003-09 | 35983 | C7a (7, 114) | 5 | 7 | 1 | 0 | nova |
| b003-10 | 2081 | C7a (7, 114) | 13 | 4 | 0 | 1 | nova |
| b003-11 | 33930 | C7a (7, 114) | 1 | 10 | 1 | 1 | nova |
| b003-12 | 32774 | C7a (7, 114) | 3 | 5 | 0 | 0 | nova |
| b003-13 | 1056 | C7b (7, 210) | 5 | 10 | 1 | 0 | nova |
| b003-14 | 33824 | C7b (7, 210) | 9 | 2 | 0 | 1 | nova |
| b003-15 | 33792 | C7b (7, 210) | 7 | 2 | 1 | 1 | nova |
| b003-16 | 33 | C7b (7, 210) | 15 | 2 | 0 | 0 | nova |

**Expectativa honesta.** Depois das cargas, cada termo tinha 11 instruções (mul, quatro para achar e ler a palavra da tabela, shr, and, setp, selp, mul por ±1 e add). Agora tem 5 (mul, and, setp e as duas predicadas, das quais só uma executa por lane), sem acesso à memória constante dentro do laço. As cargas de `a` e de `b` (ou o shuffle) continuam iguais, e com 320 vetores o kernel provavelmente é dominado por latência e lançamento, não por aritmética. Os pares b003-01..03 contra b002-01, -06 e -11 dizem se a extensão move o tempo. Um ganho de 5% sobre o direct, nas duas GPUs e com ci95 positivo, continua sendo improvável.

## Lote b004 (2026-11-20), kind=2 com pack=1

Este é o primeiro lote da extensão 2 da gramática, `pack`. As 16 propostas são kind=2 com `pack=1`. Seis delas também usam `sign_mode=1`, sempre com `unroll=16`.

**Que forma de pack este kernel admite.** O lançamento não é do PTX. Quem lança é o `gpu_probe.py`, com grade igual ao número de vetores e blocos de 16 threads (`cuLaunchKernel(f, n,1,1, 16,1,1, ...)`). Esse arquivo é da Spark e tem o hash travado em `UNCHANGED`. Um bloco de 32 threads com máscara de shuffle de 32 bits exigiria mudar a geometria do lançamento, então isso ficou fora deste lote.

A forma mínima correta dentro do kernel é esta: o bloco c calcula os vetores 2c (A) e min(2c+1, n-1) (B), e os blocos com 2c >= n saem na hora.
- Cada lane k faz os termos dos dois vetores intercalados, em j crescente. Cada vetor tem seu próprio acumulador e a mesma ordem de operações do kernel sem pack.
- O sinal de (k, j) não depende do vetor. Por isso a busca na tabela (ou o predicado do `sign_mode`) é feita uma vez por par de termos.
- Com `load=1`, cada lane guarda o `b` dos dois vetores e o shuffle dobra, sempre dentro dos mesmos 16 lanes (máscara 65535).
- Com n ímpar, o último bloco calcula o mesmo vetor duas vezes e grava bits iguais.
- `.maxntid 16,1,1` não muda.

**Admissão.** A chave é opcional, no bit 18 da máscara (262144). Ausente ou 0, nada muda: os PTX e recibos da Phase A, b001, b002 e b003 saem idênticos. `frozen_inbox_outputs.json` agora inclui o b003, e o teste também confere as variantes explícitas `pack=0` e `sign_mode=0`.

`pack>1` é recusado com `PACK`. Não existe outra combinação inválida: `pack` funciona com qualquer `unroll`, `load` e `layout`. O `sign_mode=1` continua exigindo `unroll=16` (`SIGN_MODE_UNROLL`). O recibo ganha `"pack":1`, e o `plan_id` soma 131072. O tensor não muda.

**Paridade antes da GPU.** `test_pack_ptx_computes_the_operator_reference_bit_for_bit` roda o PTX com pack pelo `ptx_sim.py` sobre as 320 fixtures. Os casos cobrem kind=1 e quatro fases, os dois loads, os dois layouts, unroll de 2 a 16 e o `sign_mode`. Dois casos usam 319 vetores, para exercitar a cauda ímpar. O resultado é zero divergências contra `operator_reference.py`.

Há dois controles, e os dois precisam divergir: um bit trocado na tabela de sinais e a remoção do store do vetor B. O simulador agora começa a saída com um sentinela finito, para que uma saída nunca gravada não passe por resultado. Todos os PTX do lote também passaram pelo `ptxas` para sm_86 e sm_75.

**O lote.**
- b004-01..04 repetem b003-01, -02, -03 e -14 com `pack=1`: `sign_mode=1` nos dois.
- b004-05..07 repetem b002-01, -06 e -11 com `pack=1`: `sign_mode` ausente nos dois.
- Com o b002 e o b003, os três operadores b002-01, -06 e -11 ficam num fatorial 2x2 (`sign_mode` × `pack`), com o mesmo plano de lane.
- b004-08 e -09 repetem os melhores do b002 contra direct (b002-07 no sm86 e b002-09 no sm75), com `unroll` 1 e 2.
- b004-10..16 são sete representantes canônicos novos: 2 em (3, 114), 3 em (7, 114) e 2 em (7, 210). Foram sorteados com a semente 20261120004, e os pares (load, layout) alternam.

| arquivo | phase | família | stride | offset | load | layout | unroll | sign_mode | origem |
|---|---|---|---|---|---|---|---|---|---|
| b004-01 | 36077 | C3 (3, 114) | 1 | 15 | 0 | 0 | 16 | 1 | par de b003-01 |
| b004-02 | 36076 | C7a (7, 114) | 11 | 3 | 1 | 1 | 16 | 1 | par de b003-02 |
| b004-03 | 33825 | C7b (7, 210) | 13 | 9 | 0 | 1 | 16 | 1 | par de b003-03 |
| b004-04 | 33824 | C7b (7, 210) | 9 | 2 | 0 | 1 | 16 | 1 | par de b003-14 |
| b004-05 | 36077 | C3 (3, 114) | 1 | 15 | 0 | 0 | 16 | 0 | par de b002-01 |
| b004-06 | 36076 | C7a (7, 114) | 11 | 3 | 1 | 1 | 16 | 0 | par de b002-06 |
| b004-07 | 33825 | C7b (7, 210) | 13 | 9 | 0 | 1 | 16 | 0 | par de b002-11 |
| b004-08 | 67 | C7a (7, 114) | 7 | 10 | 0 | 1 | 1 | 0 | par de b002-07 |
| b004-09 | 1160 | C7a (7, 114) | 15 | 14 | 0 | 0 | 2 | 0 | par de b002-09 |
| b004-10 | 32931 | C3 (3, 114) | 3 | 0 | 0 | 0 | 16 | 1 | nova |
| b004-11 | 36075 | C3 (3, 114) | 15 | 6 | 1 | 0 | 1 | 0 | nova |
| b004-12 | 2190 | C7a (7, 114) | 7 | 6 | 0 | 1 | 16 | 1 | nova |
| b004-13 | 2113 | C7a (7, 114) | 3 | 1 | 1 | 1 | 4 | 0 | nova |
| b004-14 | 34018 | C7a (7, 114) | 9 | 1 | 0 | 0 | 16 | 1 | nova |
| b004-15 | 32768 | C7b (7, 210) | 11 | 12 | 1 | 0 | 8 | 0 | nova |
| b004-16 | 1057 | C7b (7, 210) | 15 | 5 | 0 | 1 | 2 | 0 | nova |

**Expectativa honesta.** Com pack, metade dos blocos sai sem trabalhar, e a outra metade tem duas cadeias independentes de 16 termos por lane, que o `ptxas` pode intercalar para esconder a latência das cargas. O número de warps residentes cai à metade, e o sinal por termo é dividido entre os dois vetores. Se o kernel for limitado por latência e não por ocupação, pack pode ajudar. Se for limitado pelo número de warps em voo, deve perder. Um ganho de 5% sobre o direct, nas duas GPUs e com ci95 positivo, ainda seria surpresa.
