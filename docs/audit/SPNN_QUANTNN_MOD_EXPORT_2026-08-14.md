<!-- docs:meta
topic_id: repo.docs.audit.spnn-quantnn-mod-export-2026-08-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.spnn-quantnn-mod-export-2026-08-14
-->

# spnn / quantnn — FECHADAS, e a lição é o confundidor (2026-08-14)

Status: **fechadas** (check rc=0 e run rc=0 nas duas, com HYPER_METRIC correto).

## CORREÇÃO de uma versão anterior deste documento

Uma versão anterior afirmava: *"o pool de exports não registra `pub` para módulos-raiz de
pacote (`mod.sio`)"*, com repro de 6 linhas. **Isso estava ERRADO.** Era artefato de medição.

O pod exporta `SOUNIO_STDLIB_PATH=/workspace/sounio/stdlib` globalmente. Um `./bin/souc check`
rodado de QUALQUER worktree lê a stdlib do **checkout compartilhado**, não a do worktree sob
teste. Eu editava `stdlib/spnn/mod.sio` aqui e checava contra outra árvore -- por isso o `pub`
parecia não surtir efeito, e por isso inventei um defeito de resolver que não existe.

Com o caminho correto:

    SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc check /tmp/probe_pub.sio  -> check: OK

`pub` em `mod.sio` funciona normalmente.

## O que era de fato, e foi corrigido

Causa 1 -- forma de import. Os testes usavam `use spnn::*` / `use quantnn::*`. Medido:

    use spnn::*        -> AST closure incomplete, unresolved=1
    use spnn::mod::*   -> resolve

`docs/compiler/PACKAGE_IMPORT_RESOLUTION.md` documenta a forma nua apenas para
`packages/<nome>/src/lib.sio`, não para diretórios de stdlib com `mod.sio`. Corrigido.

Causa 2 -- exports faltando. 7 símbolos sem `pub` (4 em spnn, 3 em quantnn), exatamente os
que os testes chamam. Sem blanket-pub. Corrigido.

Resultado das duas lanes:

    tests/stdlib/spnn/test_spiking_e2e.sio      check rc=0  run rc=0
    tests/stdlib/quantnn/test_quantum_e2e.sio   check rc=0  run rc=0

## O achado que vale mais que o fix

O confundidor não é meu erro isolado -- é sistêmico e atinge o gate.
`scripts/stdlib/stdlib_hyper_execution_gate.sh:174`:

    export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

O `:-` cede à variável já exportada. Logo o gate, rodado de um worktree, **mede a stdlib do
checkout compartilhado**, não a da árvore sob teste. Qualquer correção feita num worktree lê
como ausente para o gate; qualquer regressão do compartilhado contamina o resultado.

Consequência prática: um agente que conserte stdlib num worktree e rode o gate verá o gate
ignorar o conserto -- e pode concluir, como eu conclui, que o conserto não funciona.

Correção sugerida (não aplicada -- script de CI, provavelmente de outra lane):
tornar a linha um export incondicional `"$ROOT_DIR/stdlib"`.

## Sugestão barata e independente (mantida)

O diagnóstico `AST closure incomplete` não imprime `unresolved_imports[0]`, embora o array
exista na struct `ModuleClosure` (`self-hosted/compiler/main.sio:2120`). Uma linha, e esta
investigação teria sido uma leitura. É a mesma lacuna que o comentário logo acima diz ter
corrigido para `parse_failed`.
