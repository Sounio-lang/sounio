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
