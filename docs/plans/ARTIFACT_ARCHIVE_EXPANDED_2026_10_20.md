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
