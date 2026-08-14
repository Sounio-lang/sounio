<!-- docs:meta
topic_id: repo.docs.audit.spnn-quantnn-mod-export-2026-08-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.spnn-quantnn-mod-export-2026-08-14
-->

# spnn / quantnn — E175 em mod.sio (lane de closure/AST, 2026-08-14)

Status: **NÃO fechado**. Reduzido de falha opaca a defeito preciso com repro de 6 linhas.

## O que era

Os dois testes falhavam com `run_check_mode: AST closure incomplete nodes=1 unresolved=1`,
sem nomear o import não resolvido. O diagnóstico tem os arrays `unresolved_imports` e
`unresolved_callers` na struct `ModuleClosure` (`self-hosted/compiler/main.sio:2120`) mas
NÃO os imprime -- exatamente a lacuna de observabilidade que o comentário logo acima diz
ter corrigido para `parse_failed`.

## Causa 1: forma de import não suportada

Os testes usavam `use spnn::*` / `use quantnn::*`. Medido:

    use spnn::*        -> AST closure incomplete, unresolved=1
    use spnn::mod::*   -> check chega ao typecheck (2 modules)

`docs/compiler/PACKAGE_IMPORT_RESOLUTION.md` documenta a forma nua apenas para
`packages/<nome>/src/lib.sio`, não para diretórios de stdlib com `mod.sio`. Só 20 arquivos
no repo usam a forma nua, vários com `;` estilo Rust (provável resíduo de doc).

Corrigido nos dois testes para `use <pkg>::mod::*`.

## Causa 2 (a real): `pub` não é honrado em `mod.sio`

Com o import resolvendo, aparece E175 em 7 símbolos. Marquei os 7 como `pub fn`
(4 em `stdlib/spnn/mod.sio`, 3 em `stdlib/quantnn/mod.sio`) e **o E175 permanece**.

Contraste medido -- `pub fn` em módulo COMUM funciona:

    use math::sedenion::*  + sed_zero()  -> nenhum E175 (erros são outros, internos)
    use spnn::mod::*       + spiking_fired() -> E175 "function is private"

Repro mínimo (6 linhas), com `spiking_fired` já marcado `pub fn`:

    use spnn::mod::*
    fn main() -> i64 {
        let n = spiking_neuron_new(1.0, 0.5)
        if spiking_fired(n) { 1 } else { 0 }
    }

Conclusão: o pool de exports não registra `pub` para módulos-raiz de pacote (`mod.sio`).
Isso provavelmente explica por que os testes foram escritos com a forma nua -- a intenção
era import de raiz de pacote, e nenhum dos dois caminhos jamais funcionou.

NÃO fui adiante: o conserto é no pool de exports do resolver, área que o PR #1697 acabou
de mexer, e é decisão de API se `use pkg::*` deve resolver para `pkg/mod.sio`.

## Sugestão barata e independente

Fazer o diagnóstico imprimir `unresolved_imports[0]`. Custa uma linha e teria transformado
esta investigação inteira em uma leitura.
