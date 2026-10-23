<!-- docs:meta
topic_id: repo.docs.plans.source59-archive-proposal-2026-10-20
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.source59-archive-proposal-2026-10-20
-->

# Proposta exata SOURCE59 — revisão do escopo ampliado, 20/10/2026

## Estado e decisão

**Proposta posteriormente aprovada e executada.** Após a publicação desta lista,
o usuário autorizou explicitamente a retirada dos 59 snapshots. A execução
concluiu a remoção dos 59 originais e criou seus índices de recuperação.
Consultar [o recibo final](SOURCE59_ARCHIVE_EXECUTION_2026_10_20.md) e a auditoria
C++23. O conteúdo abaixo conserva o escopo e o protocolo apresentados à revisão.

O lote contém exatamente 59 snapshots históricos opacos, totalizando
**10847851056 bytes lógicos de source.tar.gz**. A soma é do inventário C++23
autoritativo; não é medição de espaço físico livre ou liberação já realizada.
Há também 295 recibos pequenos copiados, 611161 bytes, de cinco tipos JSON/TSV.
Esses recibos e os demais arquivos dos diretórios originais não são alvos de
retirada. O campo full_run_bytes do cohort NÃO representa bytes propostos para
remoção.

A/B foram verificadas. A release criptografada GitHub está publicada com
**61 assets exatos e únicos: 59 snapshots, 1 pacote de metadados e SHA256SUMS**.
O download integral dos 60 assets criptografados passou na comparação SHA256;
o próprio SHA256SUMS público também foi baixado e comparado byte a byte. A cópia protegida da chave
no Mac foi verificada por SHA256 contra a custódia A; permissões restringem acesso ao proprietário.
Os recibos finais estão em artifacts/source59-20261020/. A retirada posterior está registrada no recibo final de execução.

## Lista fechada e identidade

Os nomes exatos estão nos arquivos copiados por stream, sem reconstrução manual:

- [SOURCE59_COHORT_2026_10_20.tsv](SOURCE59_COHORT_2026_10_20.tsv):
  nomes, jobs, estados, tamanhos e IDs de proprietário;
  SHA256 a8282fcbf62f2ea7ab2a692feb1019354918344dca914ea0d55eca5978646ed7.
- [SOURCE59_RUNS_2026_10_20.txt](SOURCE59_RUNS_2026_10_20.txt):
  lista exata dos 59 nomes, um por linha.
- [SOURCE59_SOURCE_2026_10_20.sha256](SOURCE59_SOURCE_2026_10_20.sha256):
  hashes originais e caminhos relativos payload/RUN/source.tar.gz;
  SHA256 7c392f4c3366235b19e020c3ca28a7fecfdad695de7e8657516de24bff2955d0.

As cópias são byte a byte dos arquivos COHORT.tsv, COHORT_RUNS.txt e SOURCE.sha256
da custódia A. O cohort preserva o formato original, incluindo separadores
literais quando presentes; não assumir uma conversão de formato na cópia.
Os 59 jobs foram classificados terminais no TERMINAL_AUDIT.tsv original.

Raiz quente original no t560:
`/var/lib/cephfs-slurm/training/sounio-foundry/runs/RUN/source.tar.gz`.

## Custódia e alcance

**A:** `/home/devsounio/sounio-artifact-archive/2026-10-20/source-expanded-59`.

**B:** `/workspace/sounio-artifact-archive/2026-10-20/source-expanded-59`.

Os snapshots estão em payload/RUN/source.tar.gz. São cópias de arquivos opacos,
com modo/mtime preservados, não hardlinks dos originais. Os recibos pequenos são:
run_packet.json, summary.json, results.tsv, runtime-receipt.json e submission.json.
O gate READY_FOR_GH foi produzido depois da verificação da segunda cópia.
Nenhuma promessa de independência absoluta entre falhas do cluster é inferida.

Manifests originais adicionais na custódia:

- FILES.sha256:3964be916049b0491574b75b98acf94d4314073944a24ce110baa65443915479.
- PAYLOAD_METADATA.tsv:c6a5c87fb14c81ed80f163b749e9fb6cfa3da905b13bf81595b0978c296bbc85.
- ORIGINAL_METADATA.tsv, TERMINAL_STATES.tsv, TERMINAL_AUDIT.tsv e CONSUMER_SCAN.txt.

É custódia **SOURCE-ONLY**, não arquivo completo das execuções. ELFs, logs e
outros resultados externos ao snapshot permanecem quentes e não foram incluídos
nem propostos para remoção. O conteúdo interno de cada source.tar.gz permanece
opaco e inalterado; não confundir isso com uma seleção de arquivos internos.
Runs recentes, pacotes atuais, gates aceitos e projetos alheios ficam excluídos.
Nenhuma seleção por glob, idade futura ou limpeza automática é autorizada.

A inspeção de consumidores identifica scripts legíveis e estado de fila no
momento registrado. Não prova ausência universal de consumidores futuros.
Retirar uma cópia quente exigiria restore explícito para reproduções históricas;
não haverá promessa de transparência dos caminhos antigos.

## GitHub criptografado

Release publicada e conferida:
[archive-sounio-source59-20261020](https://github.com/Sounio-lang/sounio/releases/tag/archive-sounio-source59-20261020).

O publicador cria 59 assets individuais .source.tar.gz.gpg e um asset
SOURCE59-METADATA.tar.gpg, mais SHA256SUMS. AES256 OpenPGP usa chave privada
guardada separadamente. A chave NÃO é asset público, NÃO entra no tar de
metadados e NÃO deve entrar neste repositório/documento/logs.

Cada asset foi descriptografado localmente e comparado por SHA256 ao
snapshot original. Os gates de publicação, lista de assets e readback passaram. A publicação só será considerada custódia externa verificada
após baixar integralmente os 60 assets criptografados e conferir seus hashes.
A existência de uma release/tag ou o sucesso do upload não substitui readback.
O próprio SHA256SUMS publicado também deve ser conferido, e a lista final
de assets deve corresponder ao cohort sem faltas/duplicatas.

A publicação preserva snapshots históricos com material de projeto interno ao
source.tar.gz, sem tentar apresentá-los como release de software, execução
científica nova ou novo resultado de compilador.

## Restore verificável, sem sobrescrever

Preferir uma cópia A/B já verificada. Para recuperação GitHub, baixar o asset
específico e o manifesto criptografado; verificar o ciphertext com SHA256SUMS,
descriptografar em diretório isolado e comparar o SHA256 original com
SOURCE59_SOURCE_2026_10_20.sha256. Nunca imprimir a chave ou fornecê-la como
argumento literal na linha de comando.

Exemplo de checagem de um snapshot em cópia interna, SEM copiar para a área quente:

```sh
# Definir RUN como um nome exato de SOURCE59_RUNS_2026_10_20.txt.
# Definir ARCHIVE_ROOT como a custódia A ou B verificada.
test -f "$ARCHIVE_ROOT/payload/$RUN/source.tar.gz"
test ! -L "$ARCHIVE_ROOT/payload/$RUN/source.tar.gz"
(
  cd "$ARCHIVE_ROOT"
  # O manifesto completo valida também todos os demais snapshots preservados.
  sha256sum --check SOURCE.sha256
)
```

Para restore quente, primeiro confirmar espaço suficiente, tipo regular,
proprietário/permissões e hash esperado. Se o destino existir, parar.
Copiar para nome temporário no destino autorizado, verificar SHA256 contra o
manifesto confiável e instalar sem sobrescrita. Verificar novamente o arquivo
final. Registrar origem, destino, hash, bytes e responsável. Bytes restaurados
não constituem uma nova aceitação científica/nativa; essa execução é separada.

## Se houver aprovação posterior da retirada

Revalidar todos os 59 hashes/tamanhos/tipos/hardlinks/estados e consumidores atuais,
as duas cópias, readback GitHub e custódia recuperável da chave no Mac.
Qualquer mudança interrompe a operação; não ampliar automaticamente o cohort.

O alvo máximo proposto é SOMENTE os 59 caminhos RUN/source.tar.gz da lista.
Pastas, manifests originais, recibos, controles, logs e ELFs permanecem.
Criar índice explícito de arquivo em cada execução e registrar a variação real
de quota após a operação. A projeção bruta de 10847851056 bytes não equivale a
espaço físico recuperável nem garante espaço livre sob escritores concorrentes.
Quota mantida: nenhum aumento é proposto.

## Limites da entrega científica atual

A validação nativa anterior de barras passou 114/114 no job 15718; os recibos
específicos estão em BARS_EXECUTION_2026_10_20.md. Este arquivamento não executa
novos testes nem amplia aquela aceitação. Base histórica aceita: d7077ef1ed21df355eee57ef3479981d554d8ee4;
a main observada durante esta operação avançou para b10c046e7b003c4346d5f2d97f47f380081cf06a.
Não houve rebase ou validação nativa dessa nova ponta nesta operação.
