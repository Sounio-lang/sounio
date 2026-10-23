<!-- docs:meta
topic_id: repo.docs.plans.artifact-offload-execution-2026-10-20
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.artifact-offload-execution-2026-10-20
-->

# Retirada autorizada dos snapshots — 20/10/2026

## Resultado executado

Após a autorização explícita `execute`, foram retirados somente os 107 arquivos `source.tar.gz` da lista revisada. Os demais arquivos dos runs, logs, resultados, manifests e ELFs foram preservados. A quota continua em 3 TiB.

Auditoria C++23 concluída:

| Verificação | Resultado |
|---|---:|
| Origens selecionadas ausentes | 107 |
| Registros de intenção | 107 |
| Registros de remoção | 107 |
| Journals A/B idênticos byte a byte | sim |
| Bytes lógicos retirados neste lote | 1.240.532.571 |
| Snapshot anterior retirado separadamente | 22.408.313 |
| Total lógico retirado nos dois passos | 1.262.940.884 |
| Espaço disponível após retirada, antes dos gates | 96.468.992 |

O espaço disponível é uma observação `statvfs`, não uma reserva. O atributo lógico do Ceph nesse checkpoint deixou 93.125.676 bytes sob a quota; o arredondamento/contabilidade de `statvfs` não precisa coincidir com esse valor. Após novas submissões foi observado espaço disponível de 67.108.864 bytes.

## Custódia e reversibilidade

Os 107 arquivos completos passaram por teste zstd, restauração e comparação recursiva de hashes, tamanhos, tipos, modos, caminhos e destinos de symlinks. A segunda cópia persistente foi lida e conferida. Propriedade dos arquivos restaurados não foi comparada por uso de `--no-same-owner`.

- A: `/home/devsounio/sounio-artifact-archive/2026-10-20/stats-terminal-expanded/`.
- B: `/workspace/sounio-artifact-archive/2026-10-20/stats-terminal-expanded/`.
- [Custódia criptografada GitHub dos 107 runs](https://github.com/Sounio-lang/sounio/releases/tag/archive-stats-viz-terminal107-20261020).
- [Custódia criptografada GitHub dos seis runs anteriores](https://github.com/Sounio-lang/sounio/releases/tag/archive-stats-viz-tails-20261020).

Os bundles foram descriptografados e comparados ao tar original, depois baixados integralmente do GitHub e conferidos por SHA256. As chaves ficam separadas, em `github-custody/RECOVERY_KEY.txt` nas respectivas custódias A/B. Não foram publicadas. Os dois lotes históricos se sobrepõem; seus números não representam 113 runs distintos.

Cada intenção foi registrada e espelhada ANTES de retirar o arquivo. Depois da retirada, o resultado também foi espelhado. Uma falha transitória de DNS interrompeu o processo antes da retirada seguinte. O endereço atual da API foi confirmado e o restante foi executado com TLS e nome de servidor preservados, sem alterar kubeconfig global ou quota. O journal permitiu retomar sem repetir remoções ou intenções.

Com escrita novamente disponível, foram criados 108 ponteiros `ARCHIVED_SOURCE.json`: os 107 deste lote mais o primeiro snapshot retirado no passo anterior. A restauração exige espaço e conferência dos hashes, sem sobrescrever arquivos existentes.

Recibos publicados em [artifacts/offload-20261020](artifacts/offload-20261020/), incluindo o plano exato, journal completo e auditoria final. Arquivos grandes e chaves não entram no Git.

## Gates retomados e problema identificado

Foram submetidos os jobs 15706 (barras), 15707 (before), 15708 (controle) e 15709 (baseline main). Os pacotes incluíam `viz_html.sio` truncado no meio de `vh_emit_axis`, terminando em `let ver`. A truncagem surgiu ao resolver o merge; estava no fonte preparado e nos snapshots, não no compilador nem no processo de arquivamento.

Os logs de parsing preservam essa falha. BEFORE e controle não são evidência discriminante válida, pois não chegaram à execução nativa. O resultado não estende a aceitação anterior 111/111 do commit7cbe54d415 à nova main ou às barras. A restauração do fonte completo e novos gates são necessários.

## Recuperação e validação concluídas

O fonte completo foi restaurado em b9cfcc7d98 e a margem do exemplo foi corrigida em f17bb2c67b. A nova baseline15712 passou111/111, e o final15718 passou114/114, com controles discriminantes válidos e SVG nativo conferido independentemente. Consulte [o recibo final de barras](BARS_EXECUTION_2026_10_20.md). A folga após os snapshots de validação ficou em4194304bytes; não é margem suficiente para novas baterias massivas sem retenção adicional revisada.
