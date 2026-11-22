<!-- docs:meta
topic_id: repo.docs.plans.artifact-archive-expanded-2026-10-20
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.artifact-archive-expanded-2026-10-20
-->

> **Atualização executada:** o usuário autorizou `execute`; os 107 snapshots da ampliação foram retirados com journal A/B verificado. A quota de 3 TiB foi mantida, a escrita foi restabelecida e 108 ponteiros de restauração foram gravados, incluindo o snapshot do passo anterior. Consulte [o recibo de execução](ARTIFACT_OFFLOAD_EXECUTION_2026_10_20.md). O texto abaixo preserva as propostas e os checkpoints anteriores; condições de autorização pendente foram superadas pela execução registrada.

# Ampliação proposta de arquivamento — 20/10/2026

## Estado e motivo

A primeira retirada autorizada preservou22.408.313bytes em custódia externa, mas a gravação de um pequeno recibo quente falhou EDQUOT. A limpeza foi interrompida. O Ceph contabilizou3.299.673.408.283bytes de uso para quota3.298.534.883.328bytes: excedente1.138.524.955bytes. Esses números são uma observação, não reserva de capacidade.

O C++23 de inventário identificou107runs terminais próprios do sprint stats-viz, sem nomes contendo scatter ou bar. Os source.tar.gz somam1.240.532.571bytes; pastas completas somam1.287.891.575bytes lógicos. A lista exata está em [STATS_TERMINAL_107.tsv](STATS_TERMINAL_107.tsv). Quatro entradas com estado desconhecido foram excluídas.

Remover apenas esses107snapshots, sem outros escritores, deixaria102.007.616bytes de margem. É uma folga pequena para retomar gates focados; não resolve retenção geral do volume. Nenhuma remoção deste lote foi autorizada por esta proposta ou executada durante sua preparação.

## Proposta concreta

Após aprovação específica do lote ampliado e confirmação de duas cópias verificadas mais custódia GitHub criptografada, revalidar por arquivo tamanho, SHA256, único hardlink, estado terminal e consumidores conhecidos. Retirar SOMENTE source.tar.gz dos107nomes listados. Manter todos os logs, resultados, receipts, manifests, ELFs e as pastas de execução. Não remover por glob ou idade; não tocar em outras frentes ou alterar quota.

Custódia A: `/home/devsounio/sounio-artifact-archive/2026-10-20/stats-terminal-expanded/`.
Custódia B: `/workspace/sounio-artifact-archive/2026-10-20/stats-terminal-expanded/`.

Cada arquivo completo recebe manifesto recursivo, teste zstd, restauração isolada, comparação de hashes/tamanhos/tipos/modos e nova leitura dos originais. Cópia e verificação são preparação reversível. Preservar bytes não reexecuta a ciência.

O bundle GitHub é criptografado porque snapshots históricos incluem material além do suplemento. A chave permanece em `github-custody/RECOVERY_KEY.txt` nas custódias A/B; não vai ao Git ou à release. O asset deve ser baixado e conferido antes de qualquer nova retirada. GitHub sem chave não permite recuperação.

## Interrupção e restauração

Se hashes ou consumidores divergirem, parar. Se uma retirada não permitir escrita do recibo, registrar na custódia externa, parar e não ampliar novamente a seleção. Medir o atributo de uso do Ceph e o espaço disponível; o df satura em zero quando já há excesso.

Restaurar exige espaço para o snapshot selecionado. Usar a cópia restaurada, confirmar o hash registrado em `pilot.tsv` e repor o caminho exato sem sobrescrever conteúdo existente. Manter os manifests históricos com seus caminhos originais.

## Validação do sprint

A última aceitação nativa é111/111 do commit7cbe54d415 na base anterior. O novo código9975c1d017 está publicado e tem revisão estática sem bloqueador concreto;83casos do verificador C++23 passaram. Baseline111 sobre main atual, signed-before e corpus114 ainda estão pendentes de armazenamento. Não há nova aceitação nativa, CI verde ou merge alegados.

## Preparação concluída e protocolo corrigido

Os107arquivos completos foram criados:1.245.420.775bytes de tar.zst. Todos passaram zstd, extração, hashes recursivos, tamanhos, modos, tipos, caminhos e destinos de symlinks. Proprietário não foi comparado após `--no-same-owner`. As107origens foram rechecadas. A cópia B recebeu todos os arquivos e manifests por stream direto do cluster e passou a conferência completa.

Manifests:

- `COHORT_RUNS.txt`: fa0534d0bae56fb4f09d877d67ff73a79a3e5fdea13fb9d341ece05747235165
- `ARCHIVES.sha256`: 805f43b7a65118865269b94ed46d97b03445bc7c3714f28f37c568d149739b2e
- `RECEIPTS.sha256`: 6db6f8f745c525ef8d571d32ee75998a0b62bf822d1df23ec89186482345da70

Como o uso começa acima da quota, a ampliação precisa registrar o índice/journal nas custódias externas A/B ANTES de qualquer retirada; tentar escrever o índice quente após a primeira retirada repetiria a falha conhecida. O journal externo deve guardar nome exato e hashes antes/depois de cada retirada. Criar ponteiros quentes somente quando o Ceph voltar a admitir escrita. Esta descrição substitui a condição anterior de interromper apenas porque um ponteiro quente continua bloqueado durante o excedente conhecido. Divergência de hash/consumidor ou falha de custódia/journal exige interrupção.

Isso amplia o lote inicialmente revisado de seis arquivos para107nomes explícitos. Nenhum arquivo desse novo lote foi retirado. A decisão pendente é autorizar essa ampliação mantendo logs/resultados no Ceph e a quota inalterada. A margem estimada102.007.616bytes depende do uso real no momento da execução.

## GitHub: publicação e readback concluídos

Release de custódia: [107 terminal runs encrypted](https://github.com/Sounio-lang/sounio/releases/tag/archive-stats-viz-terminal107-20261020).

- Asset `stats-terminal-107.tar.gpg`:1.245.552.735bytes.
- SHA256 criptografado: `13987e2fc7bf30dbf42766d3daac25e3b60847fb9cd535162f33f2c2ac5bac81`.
- Descriptografia reproduziu o hash do tar original.
- Asset baixado integralmente do GitHub e conferido: `GH_107_CUSTODY_READBACK_VERIFIED`.
- Chaves nas custódias A/B têm hash idêntico e não foram publicadas.
- A primeira release de seis runs é imutável; este lote usa uma segunda release, preservando a primeira.

A ampliação de remoção foi submetida para revisão do usuário. Até esta atualização, nenhum source.tar.gz do lote107 foi removido; quota e disponibilidade de escrita continuam sem mudança.
