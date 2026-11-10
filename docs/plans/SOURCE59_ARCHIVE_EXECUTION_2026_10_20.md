<!-- docs:meta
topic_id: repo.docs.plans.source59-archive-execution-2026-10-20
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.source59-archive-execution-2026-10-20
-->

# SOURCE59: execução autorizada do arquivamento, 2026-10-20

## Autorização e escopo

O usuário aprovou explicitamente: **“Autorizar a retirada desses 59 snapshots”**
após receber a proposta publicada em `0191b73a27f27d53c8102f8b0a292498c1f1a9f7`.
O escopo é somente os 59 caminhos `RUN/source.tar.gz` do manifesto
[SOURCE59_RUNS_2026_10_20.txt](SOURCE59_RUNS_2026_10_20.txt), totalizando
10.847.851.056 bytes lógicos. Os demais arquivos dos runs não são alvos.
A quota de 3 TiB foi mantida.

## Custódia externa e recuperação

[Release criptografada publicada](https://github.com/Sounio-lang/sounio/releases/tag/archive-sounio-source59-20261020):
59 snapshots, um pacote de metadados/recibos e SHA256SUMS, exatamente 61 assets.
Todos os 60 ciphertexts foram baixados integralmente e comparados por SHA256;
o próprio SHA256SUMS público também foi baixado e comparado byte a byte.
A descriptografia local foi comparada aos hashes originais antes do upload.
A chave não foi publicada: permanece nas custódias internas e em cópia protegida
fora do cluster no Mac, com hash conferido.

Duas cópias persistentes conservam os snapshots opacos e 295 recibos pequenos:

- A: `/home/devsounio/sounio-artifact-archive/2026-10-20/source-expanded-59`.
- B: `/workspace/sounio-artifact-archive/2026-10-20/source-expanded-59`.

É custódia dos snapshots de fonte, não arquivo completo dos runs. A reprodução
histórica exige restauração explícita, conforme a proposta. Nunca sobrescrever
um destino existente e sempre conferir o hash original após restaurar.

## Gates antes da retirada

- 59/59 jobs históricos terminais.
- 14/14 scripts dos jobs atuais em fila/execução legíveis; nenhuma referência
  exata aos nomes do lote nos scripts ou diretórios de trabalho.
- `lsof` local sem matches. Visibilidade incompleta: usuário não privilegiado
  no t560, outros clientes Ceph e referências dinâmicas não comprovados ausentes.
- Nova conferência integral dos manifests e dos bytes A/B antes da retirada.
- Originais regulares, sem symlink, um hardlink, tamanho/modo/mtime/UID/GID
  preservados e hash original conferido antes de cada remoção.
- Lista e manifests fixados por SHA256 no procedimento.
- INTENT registrado nas duas custódias antes de cada remoção; conclusão
  registrada após a criação de `ARCHIVED_SOURCE.json` no run original.

## Resultado medido

**Concluído: 59 originais retirados e 59 índices criados; auditoria PASS.**

| Medição | Bytes |
|---|---:|
| Snapshots retirados, tamanho lógico | 10.847.851.056 |
| Espaço disponível antes | 4.194.304 |
| Espaço disponível depois | 10.733.223.936 |
| Aumento observado de espaço disponível | 10.729.029.632 |

Consultar os recibos `artifacts/source59-20261020/REMOVAL_AUDIT.tsv`,
`QUOTA_BEFORE.txt`, `QUOTA_AFTER.txt`, `REMOVAL_JOURNAL.tsv` e `REMOVAL_COMPLETE`.
A auditoria final C++23 exige os 59 originais ausentes, 59 índices existentes,
59 intents e 59 conclusões únicos, e tamanhos preservados na custódia A.
A diferença de espaço disponível é observada sob escritores concorrentes;
não confundir bytes lógicos retirados com bytes físicos exclusivos.

## Limites científicos

Esta operação não executa novos gates científicos e não altera a implementação
da stdlib. A validação anterior de barras permanece 114/114 no job 15718,
com recibos específicos em BARS_EXECUTION_2026_10_20.md. A main avançada não foi
rebaseada nem validada por este arquivamento.

## Interrupção recuperada

A primeira tentativa parou antes do primeiro unlink porque `runuser` não estava
no PATH do t560. A identidade da sessão já era UID/GID 1000 (devsounio), igual
à proprietária dos arquivos. A retomada exigiu essa identidade, preservou o
primeiro INTENT e o diário original, e usou rm/tee sob a própria sessão.
O INTENT existente foi reutilizado sem duplicar a entrada; cada arquivo voltou
a passar por hash e metadados antes do unlink. A auditoria C++23 final confere
a unicidade dos 59 INTENTs e das 59 conclusões.
