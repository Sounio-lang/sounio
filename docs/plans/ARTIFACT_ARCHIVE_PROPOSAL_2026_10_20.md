<!-- docs:meta
topic_id: repo.docs.plans.artifact-archive-proposal-2026-10-20
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.artifact-archive-proposal-2026-10-20
-->

> **Atualização executada:** o usuário autorizou `execute`; os 107 snapshots da ampliação foram retirados com journal A/B verificado. A quota de 3 TiB foi mantida, a escrita foi restabelecida e 108 ponteiros de restauração foram gravados, incluindo o snapshot do passo anterior. Consulte [o recibo de execução](ARTIFACT_OFFLOAD_EXECUTION_2026_10_20.md). O texto abaixo preserva as propostas e os checkpoints anteriores; condições de autorização pendente foram superadas pela execução registrada.

# Estado executado e custódia GitHub — 20/10/2026

Esta seção substitui o estado de preparação da proposta histórica preservada abaixo.

- O usuário autorizou publicar o necessário no GitHub e liberar espaço mantendo a quota.
- Código preparado publicado: `9975c1d017c9eb7164e5264a80db07c3974f076c`, branch `sprint/bars-main-d7077-20261020`.
- Custódia histórica completa criptografada: [GitHub archive release](https://github.com/Sounio-lang/sounio/releases/tag/archive-stats-viz-tails-20261020). Não é release de software nem aceitação dos novos gates.
- O bundle contém os seis arquivos completos e manifests. OpenPGP AES256 protege material histórico não específico deste sprint. A chave NÃO foi publicada: está em `github-custody/RECOVERY_KEY.txt` nas duas custódias persistentes A/B. Sem a chave o GitHub não basta para recuperar o conteúdo.
- O bundle foi descriptografado e comparado ao SHA256 do tar original; o asset foi baixado do GitHub e comparado ao hash criptografado. A chave nas cópias A/B tem hash idêntico. Os seis tar.zst completos continuam disponíveis nas duas cópias.
- Slurm está saudável pelo transporte Kubernetes usado pelo Forge. Os seis jobs são terminais; nenhum dos 14 jobs atuais referenciou os seis caminhos em WorkDir ou scripts submetidos. Isso não exclui consumidores indiretos.

## Liberação parcial e interrupção

Somente `stats-viz-tails-20261016-before/source.tar.gz` foi retirado da área quente, após conferir tamanho, hash, hardlink único, arquivos completos e ausência de abertura local detectável. São **22408313 bytes lógicos** preservados nas duas custódias e no bundle criptografado GitHub.

A tentativa de gravar o pequeno `ARCHIVED_SOURCE.json` falhou com **Disk quota exceeded**. O arquivo vazio criado pela tentativa foi retirado. O recibo completo está em `FIRST_SOURCE_ARCHIVED.json`, `APPROVAL_SCOPE.executed.json` e `OFFLOAD_RECEIPT.tsv` na custódia A; deve ser espelhado na B. Os logs/resultados históricos permaneceram no local.

A limpeza PAROU nesse ponto. Os outros cinco source.tar.gz seguem intactos. O `df` continua reportando **zero bytes disponíveis**. O atributo `ceph.dir.rbytes` observado após a primeira retirada foi **3299673408283**, superior ao limite **3298534883328**; esse excedente não é exibido pelo df limitado à quota. O pequeno lote inicialmente proposto é insuficiente para restabelecer a escrita. Não houve expansão da quota ou remoção de outras frentes.

Os gates atualizados baseline111, signed-before e barras114 continuam pendentes de armazenamento. Os scripts e pacotes imutáveis estão preparados; nenhum novo resultado nativo é alegado. A última aceitação permanece o corpus111 do commit7cbe54d415 sobre a base anterior.

---

# Proposta histórica anterior à autorização e tentativa de liberação

O texto abaixo registra o escopo revisado e os testes de restauração anteriores. Afirmações de "não autorizado", "intacto" e "nada removido" descrevem aquele momento e são substituídas pelo estado acima.

# Proposta de arquivamento para revisão — 20/10/2026

## Decisão solicitada

Manter a quota atual de **3 TiB**. Autorizar, em uma etapa posterior, a remoção
**somente das seis cópias quentes de `source.tar.gz`** listadas abaixo, após uma
nova conferência dos hashes. Os diretórios, recibos, logs, resultados, JUnit,
controles e manifests continuarão no Ceph. Nada foi removido nesta preparação.

As cópias completas e a restauração de teste já estão prontas. A retirada dos
originais da área quente ainda NÃO foi autorizada nem executada.

## Lote exato

Prefixo no host t560:
`/var/lib/cephfs-slurm/training/sounio-foundry/runs/`

| Diretório / arquivo selecionado | Job | Estado terminal | Bytes do source.tar.gz |
|---|---:|---|---:|
| stats-viz-tails-20261016-before/source.tar.gz | 14672 | FAILED 1:0 | 22408313 |
| stats-viz-tails-20261016-spearman-before/source.tar.gz | 14678 | FAILED 1:0 | 22413268 |
| stats-viz-tails-20261016-after2/source.tar.gz | 14676 | COMPLETED 0:0 | 22414742 |
| stats-viz-tails-20261016-final/source.tar.gz | 14679 | COMPLETED 0:0 | 22413128 |
| stats-viz-tails-20261016-after1/source.tar.gz | 14675 | FAILED 1:0 | 22414752 |
| stats-viz-tails-20261016-before2/source.tar.gz | 14674 | FAILED 1:0 | 22409005 |

Todos encerraram em 16/10/2026; proprietário devsounio:devsounio. Cada snapshot
é arquivo regular com um único hardlink. Falhas e sucessos foram preservados
juntos: um job FAILED não é tratado como evidência descartável.

Somatório executado em C++23: **134473208 bytes lógicos = 128,244 MiB** nos seis
snapshots. Este é o limite bruto proposto para liberação de quota, não uma
promessa de espaço físico nem de disponibilidade sob escritores concorrentes.
O índice de arquivamento ocupará uma pequena parte desse limite.

## Cópias já preparadas e verificadas

Uma pasta de execução COMPLETA foi preservada em cada arquivo POSIX pax tar.zst,
incluindo o source.tar.gz original como bytes opacos. Nenhum ELF foi reconstruído,
relinkado ou alterado. A compressão usou `zstd -T2 -3`.

**Cópia A, persistente no t560, fora da quota de training:**
`/home/devsounio/sounio-artifact-archive/2026-10-20/tails-six/`

Filesystem verificado: `/dev/mapper/pve-root`, ext4; aproximadamente83 GiB livres
na observação. Uma escrita/leitura de teste também passou.

**Cópia B, persistente no workspace, em outro filesystem/quota:**
`/workspace/sounio-artifact-archive/2026-10-20/tails-six/`

Filesystem verificado: `/dev/rbd9`; aproximadamente93 GiB livres na observação.
Os seis hashes de arquivo foram conferidos após a transferência, lendo a cópia B.
Duas cópias não constituem garantia de independência contra toda falha de cluster.

Os seis arquivos completos somam **134377624 bytes = 128,152 MiB** por cópia.
Como os snapshots já eram gzip, recompressão trouxe pouco ganho. O benefício
proposto decorre de transferir a custódia para fora da quota, não de prometer uma
redução grande do conteúdo científico.

O caminho `/home/devsounio/projects` foi rejeitado como destino: uma criação de
diretório ficou bloqueada em `thin_defer_cell`. A tentativa foi interrompida;
nenhum snapshot foi copiado para esse destino. A seleção final usa o filesystem
raiz, com escrita/leitura efetivamente verificadas.

## Verificações concluídas

Para cada uma das seis pastas:

1. Manifesto recursivo SHA256 de TODOS os arquivos, incluindo snapshots e arquivos
   aninhados; lista de tamanhos dos arquivos e metadados de tipos/permissões/paths.
2. Criação do arquivo completo em destino externo à quota, sem mudar a origem.
3. `zstd -t` do arquivo.
4. Extração para diretório isolado de teste, sem substituir originais.
5. SHA256 de cada arquivo restaurado e comparação exata de tamanhos, caminhos,
   tipos e permissões com a origem. Não havia symlinks no lote selecionado.
6. Nova conferência SHA256 da origem após a cópia.
7. Segunda cópia persistente e nova conferência dos seis hashes de arquivo.

Os seis testes de restauração passaram. Não houve reexecução científica dos
binários; preservação de bytes não é uma nova validação nativa do Sounio.

A origem continua intacta e a quota continua em3298534883328 bytes. Nenhum
espaço foi liberado desta quota durante o piloto.

## Evidência e hashes

Na pasta de cada cópia:

- `pilot.tsv`: arquivo exato, tamanho, SHA256 do snapshot original e do tar.zst.
- `ARCHIVES.sha256`: hashes dos seis arquivos completos.
- `RUN.receipt/FILES.sha256`: manifesto recursivo original/restaurado.
- `RUN.receipt/FILE_BYTES.tsv` e `RESTORED_FILE_BYTES.tsv`: tamanhos comparados.
- `RUN.receipt/METADATA.tsv` e `RESTORED_METADATA.tsv`: metadados comparados.
- `slurm-states.txt`: estados terminais e horários reais.
- `SUMMARY.txt` e `summarize.cpp`: soma reproduzível em C++23.

Cuidado: `artifact_hashes.tsv` do Foundry exclui `source.tar.gz` e arquivos com
profundidade maior que dois componentes. Por isso ele foi preservado, mas não
usado sozinho como prova de completude. Os manifests recursivos complementam a
custódia. Referências históricas `/orangefs/...` permanecem inalteradas nos recibos.

## Ação proposta após aprovação

- Revalidar estado, hash, tamanho, hardlink único e ambas as cópias para os seis
  nomes exatos; interromper se qualquer arquivo tiver mudado ou estiver em uso.
- Retirar da área quente APENAS os seis source.tar.gz acima e registrar um índice
  explícito `ARCHIVED_SOURCE.json` por execução, com localização, hash e restore.
- Conferir a variação real de quota. Se outros escritores consumirem o espaço ou
  a medição divergir da hipótese, parar e relatar; não ampliar o lote.
- Usar o espaço para os gates preparados: baseline111 após a incorporação de main,
  regressão signed-before, corpus de barras114 e controle negativo de clipping.
  Estes gates ainda NÃO foram executados nem aceitos.

Os checkpoints recentes cluster108 e scatter111 ficam quentes, assim como toda
execução fora desta lista. Não há limpeza por idade, glob abrangente, remoção de
ELFs de outras frentes, mudança de quota ou limpeza automática posterior.

## Impacto e reversão

O Forge status, resumos, logs e resultados dos seis jobs continuam disponíveis.
Uma reprodução que requeira o source.tar.gz histórico exigirá restore explícito;
não se promete relocação transparente dos caminhos antigos.

Restore de um snapshot, usando sua pasta restaurada já verificada no t560:

```sh
run=stats-viz-tails-20261016-final
archive_root=/home/devsounio/sounio-artifact-archive/2026-10-20/tails-six
source_root=/var/lib/cephfs-slurm/training/sounio-foundry/runs
# Hash específico deste run, já registrado em pilot.tsv:
expected=978996685e0bcd8e4cbbf5c63e5172366ef6a827d313016c97dadd8725766ab3
target="$source_root/$run/source.tar.gz"
# Conferir antes o espaço exigido pelo tamanho exato em pilot.tsv.
test ! -e "$target" && test ! -L "$target"
cp --no-clobber "$archive_root/restore-check/$run/$run/source.tar.gz" "$target"
test -f "$target" && test ! -L "$target"
printf '%s  %s\n' "$expected" "$target" | sha256sum --check --strict
```

A cópia B pode repor a cópia A se necessário. Restore precisa de espaço suficiente;
o teto bruto deste lote é pequeno frente ao volume de3 TiB e não resolve a
política geral de retenção. Aprovação deste lote não autoriza remover outros
arquivos para acomodar uma restauração.

## Continuidade do sprint

Main atual incorporada: d7077ef1ed21df355eee57ef3479981d554d8ee4.
Merge local: b53cb77d9ee6ce3e752eb439366ebb332156235d.
Correção do empacotamento:1143c5f2ab, removendo referências a dois arquivos NB ausentes.
Worktree: `/workspace/sounio-sprint-stats-viz-main5-bars-20261020`.

Barras compartilhadas, viewport, três fixtures e verificadores estão preparados.
O C++23 independente passou83 casos (61retângulos,22larguras); root recompilou
sem avisos e reproduziu a fixture byte a byte. Isso NÃO substitui os gates
Sounio, bloqueados antes da submissão pelo limite do armazenamento.
A última entrega nativamente aceita/publicada permanece7cbe54d415,111/111 sobre
sua base anterior. Não se estende essa aceitação à nova main nem às novas barras.
