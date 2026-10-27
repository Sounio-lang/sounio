<!-- docs:meta
topic_id: repo.docs.decisions.pending
authority: repo_only
audience: users
last_validated: 2026-10-13
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.pending
-->

# Decisões pendentes do mantenedor

Cada seção abaixo é uma decisão que não cabe a quem corrige bugs: semântica da linguagem, API pública, escolha entre alternativas legítimas, ou pergunta científica sem defeito de código. A issue de origem foi fechada com o link para a seção; a decisão mora aqui, com tudo o que é preciso para tomá-la sem reinvestigar.

Formato de cada seção: **Origem** (issue, PRs relacionados) · **Pergunta** (uma frase) · **Opções** · **Recomendação** · **Teste discriminante** (o teste que passa a existir/ser verdadeiro conforme a opção escolhida) · **Causa localizada** (arquivo e função, medida em `main` `c2e3df23c`).

Os ADRs de [`README.md`](README.md) registram decisões já tomadas; este arquivo registra as que ainda não foram. Quando uma for tomada, ela vira ADR e a seção sai daqui. D-15, D-16, D-20, D-22, D-24, D-25, D-26 e D-27 foram decididas em 2026-10-25: [ADR-020](adr-020-rulings-2026-10-25.md).

| § | decisão | origem |
|---|---|---|
| [D-01](#d-01) | Modelo de `Float` na prova em Lean | #2761 |
| [D-02](#d-02) | Capacidades de autoridade opacas além de structs privadas | #1386 |
| [D-03](#d-03) | Certificado de suporte de base `Q` para admitir produtos de sedenions (E251) | #2380 |
| [D-04](#d-04) | `LibraryId` como entidade de FFI de primeira classe | #2535 |
| [D-05](#d-05) | Semântica de interposição de símbolos de `LibraryId` | #2536 |
| [D-06](#d-06) | Repositório canônico de produção (R3): corte e remoção de fontes | #1122 |
| [D-07](#d-07) | Linha de pesquisa epistêmica "source-fresh aliasing / associador": reaterrissar ou aposentar | #1423 |
| [D-08](#d-08) | Hessiana do cérebro sedenion: reenquadrar a hipótese nula | #1484 |
| [D-09](#d-09) | EISA-H: onde mora o detector de ZD e o repositório de hardware | #839 |
| [D-10](#d-10) | Cadeia de bootstrap `stage0 → boot4` | #725 |
| [D-11](#d-11) | Suíte `compile-fail` neutra de engine | #2241 |
| [D-12](#d-12) | Gate de regressão de corpus: baseline e fiação | #2306 |
| [D-13](#d-13) | Os dois pipelines `-O` | #2368 |
| [D-14](#d-14) | Censo de alcançabilidade (~314 módulos sem importador) | #2371 |
| [D-17](#d-17) | Variantes de enum com payload | #645 |
| [D-18](#d-18) | Aliasing de retornos de tupla-array (não reproduzível) | #1582 |
| [D-19](#d-19) | `Knowledge<T where {...}>`: o predicado é imposto? | #2753 |
| [D-21](#d-21) | SOIR: colisão magic+versão 6 | #1162 |
| [D-23](#d-23) | `Knowledge ⊗ Knowledge` no Madaros (E245) e NaN além do teto de recursão | #1706 |
| [D-28](#d-28) | Operadores infixos `!` e `~` (aplicado provisoriamente no PR #2921) | #2249 |
| [D-29](#d-29) | Seed macOS ARM64 com recibo de proveniência | #821 |
| [D-30](#d-30) | Raízes precisas para o GC de handles | #1491 |
| [D-31](#d-31) | Reborrow por parâmetro `&!` pode aliasar? | #1581 (resíduo) |
| [D-32](#d-32) | `spawn {}` / `.await` / `async fn` no Madaros | #2604 (resíduo) |
| [D-34](#d-34) | Primitiva intervalar certificada: as cinco decisões da proposta | [proposta](../proposals/CERTIFIED_INTERVAL_PRIMITIVE_2026-10-22.md) |

---

<a id="d-01"></a>
## D-01 — Modelo de `Float` na prova em Lean

- **Origem:** #2761.
- **Pergunta:** os axiomas `float_le_refl`, `float_le_trans` e `float_add_le_add` falam de `Float` do Lean como se fosse uma ordem total sem NaN; qual modelo de `Float` as provas devem assumir?
- **Opções:** (A) manter os axiomas e documentar a restrição a valores finitos em `AXIOM_INVENTORY.md`; (B) acrescentar hipóteses `isFinite` aos axiomas e propagá-las pelos teoremas; (C) trocar `Float` por um modelo `Real`/racional e provar a ponte para IEEE 754 separadamente.
- **Recomendação:** (B) agora, com a ressalva em `AXIOM_INVENTORY.md` no mesmo commit; (C) como trabalho de pesquisa posterior.
- **Teste discriminante:** `#print axioms` nos teoremas de cabeçalho deve listar só axiomas com hipótese de finitude, e a testemunha `native_decide` com NaN (`(0.0/0.0) ≤ (0.0/0.0)` é falsa) não pode derivar `False`.
- **Causa localizada:** `formal/TypeCheckerSoundness.lean:47,50,53`; os mesmos axiomas em `formal/Epistemic.lean:277,283,286` e `formal/SecondOrderGUM.lean:153,156`. Não há toolchain Lean no ambiente da campanha, então a testemunha `native_decide` não foi rodada de novo.

<a id="d-02"></a>
## D-02 — Capacidades de autoridade opacas além de structs privadas

- **Origem:** #1386. O PR #2927 (`Refs`) leva a privacidade de campo entre módulos (E259) e `pub linear struct` ao `lean_single`; a linearidade entre módulos e as perguntas de projeto seguem abertas.
- **Pergunta:** o contrato de capacidade (atenuação, frescor/revogação, impossibilidade de forjar por cast ou serialização) deve ser garantido pelo compilador nos dois engines, ou só no Madaros?
- **Opções:** (A) imposição nos dois engines (E259 para campo privado, E039 para uso linear entre módulos também no `lean_single`); (B) declarar que as garantias valem só no Madaros e documentar; (C) estudar tipos de capacidade dedicados (atenuação/revogação) antes de decidir A ou B.
- **Recomendação:** (B) imediatamente (documentar), (A) em seguida, (C) como pesquisa.
- **Teste discriminante:** `forge.sio` (`Token { scope: 99 }` fora do módulo definidor) deve dar erro nos dois engines (hoje: Madaros E259; `lean_single` imprime `99`); `lin3.sio` (resgatar duas vezes) deve dar E039 nos dois (hoje `lean_single` roda e imprime `1`, `1`).
- **Causa localizada:** Madaros: E259, commit `c592f2c61` (KL-4, #2488). `lean_single`: não há verificação de privacidade de campo nem de linearidade entre módulos em `self-hosted/compiler/lean_single.sio` (o `check` devolve rc=0 em todas as sondas). `docs/research/external_authority_capability_contract_2026-07-21.md` ainda não está em `main`.

<a id="d-03"></a>
## D-03 — Certificado de suporte de base `Q` (E251)

- **Origem:** #2380 (decisão de 2026-09-01: opção (b); não implementada).
- **Pergunta:** implementar agora o refinamento de `Knowledge` por suporte de base (∪ para `+`, fecho XOR para `*`) para que E251 deixe de recusar incondicionalmente produtos de sedenions certificados?
- **Opções:** (a) manter E251 incondicional; (b) suporte refinado no tipo, com descarga de E251; (c) verificação em tempo de execução do suporte.
- **Recomendação:** (b), tamanho L, em PR próprio com as 7 testemunhas do comentário da decisão.
- **Teste discriminante:** um produto de dois sedenions com suportes disjuntos e XOR-fechados compila; um com suporte não certificado continua recusado com E251.
- **Causa localizada:** `self-hosted/check/check.sio:8307-8309` (`kk_both && kk_hyper && e.bin_op == BinaryOp::OpMul && algebra_kind >= 4` → `checker_report_error_at_inplace(..., 251, ...)`); `grep -rn 'affine_basis_support\|Support<'` não encontra nada.

<a id="d-04"></a>
## D-04 — `LibraryId` como entidade de FFI de primeira classe

- **Origem:** #2535; PR #2663 deduplica a tabela mas a mantém.
- **Pergunta:** `LibraryId` continua uma tabela fixa no codegen ou vira `ExternSymbol` declarável pelo usuário?
- **Opções:** (A) promover a entidade de primeira classe, começando pela recusa fail-closed de IDs desconhecidos; (B) manter a tabela e documentar; (C) só a recusa fail-closed.
- **Recomendação:** (A), com (C) como primeiro passo.
- **Teste discriminante:** `extern "C" from "libm.so.6" { fn cbrt }` deve produzir `NEEDED libm.so.6` no ELF; um nome desconhecido deve ser recusado em vez de cair em `libkl14b_probe.so`.
- **Causa localizada:** `self-hosted/native/codegen_x86_linux.sio`: `native_v2_dynlink_lib_id_for_name` (~1904–1931, tabela fixa), `native_v2_dynlink_soname_for_lib_id` (~1934–1949, cai silenciosamente em `libkl14b_probe.so`), `native_v2_dynlink_soname_for_bytes` (~1952, sem chamador).

<a id="d-05"></a>
## D-05 — Interposição de símbolos de `LibraryId`

- **Origem:** #2536.
- **Pergunta:** qual a semântica quando duas bibliotecas exportam o mesmo símbolo?
- **Opções:** (A) só proveniência (escopo global ELF, primeira correspondência vence), documentado; (B) isolamento por `dlopen(RTLD_LOCAL)` como opt-in posterior.
- **Recomendação:** (A) agora, documentado; (B) depois.
- **Teste discriminante:** duas bibliotecas de teste exportando `f` com valores distintos; o programa deve imprimir o valor da primeira em `DT_NEEDED` sob (A).
- **Causa localizada:** o escritor da seção dinâmica em `codegen_x86_linux.sio` (~14430–14560) emite `DT_NEEDED` simples e dynsyms indefinidos `STB_GLOBAL|STT_FUNC`, sem versionamento nem `DT_FLAGS`.

<a id="d-06"></a>
## D-06 — Repositório canônico de produção (R3)

- **Origem:** #1122.
- **Pergunta:** o mapeamento está respondido (comentários de 2026-07-20); o corte e a remoção dos pacotes de `main` serão feitos?
- **Opções:** (A) fechar como respondido e abrir decisão separada de corte; (B) cortar agora; (C) abandonar R3.
- **Recomendação:** (A).
- **Teste discriminante:** `packages/{epistemic-core,sounio-formats,sounio-io-primitives,sounio-units}` ausentes de `main` e os cinco repositórios-alvo nos mesmos heads (`3e7d49fb`, `4d6987ce`, `b991a4ea`, `229d310f`, `a22f66e0`).
- **Causa localizada:** os quatro pacotes seguem em `main`, sem alteração desde `d834ab4e5`.

<a id="d-07"></a>
## D-07 — Linha de pesquisa epistêmica "source-fresh aliasing / associador"

- **Origem:** #1423.
- **Pergunta:** reaterrissar a camada de recibos/aliasing ou aposentar a linha e manter o protocolo em `docs/research`?
- **Opções:** (1) cherry-pick dos três commits (`1d96a35d8`, `0a8131abf`, `1573ca2bf`, buscados por SHA no GitHub); (2) aposentar.
- **Recomendação:** se a linha não for retomada, **primeiro** preservar os três SHAs como tag `archive/` (hoje só alcançáveis no GitHub, sem branch nem tag: podem sofrer GC); depois fechar como aposentada.
- **Teste discriminante:** `stdlib/epistemic/state_aliasing_receipts.sio`, `parenthesization_receipts.sio` e `scripts/ci/epistemic_receipt_source_fresh_gate.sh` existem em `main` (opção 1) ou a tag `archive/` resolve os três SHAs (opção 2).
- **Causa localizada:** os três branches `codex/psychiatric-inference-*` retornam 404; os arquivos acima estão ausentes de `main`. O bloqueio técnico original (testemunha com `1 != 0`) está resolvido: imprime `OPNE_OK` nos dois engines.

<a id="d-08"></a>
## D-08 — Hessiana do cérebro sedenion: resultado nulo

- **Origem:** #1484 (parte de código duplicada por #1489, em PR #2906).
- **Pergunta científica:** a hipótese "ASD tem mais direções nulas que ADHD" é reenquadrada ou descartada, dado que o falso nulo foi corrigido e a Hessiana não é identicamente zero?
- **Opções:** (A) registrar o resultado negativo corroborado entre engines e descartar a hipótese; (B) reenquadrar a hipótese com outra métrica (traço/anisotropia); (C) repetir com outra topologia.
- **Recomendação:** (A) com (B) como seguimento.
- **Teste discriminante:** `Null dims = 0/16` em ASD e ADHD nos dois engines; traço ASD `24.594517`, ADHD `−9.941127`, anisotropia `9.156645` (Madaros) / `9.156641` (`lean_single`), concordando em ≥ 6 algarismos significativos.
- **Causa localizada:** sem defeito de código restante; o falso nulo vinha de efeitos (corrigido em `c63315965`, #1745) e de aliasing (#1480).

<a id="d-09"></a>
## D-09 — EISA-H: repositório de hardware e detector exato de ZD

- **Origem:** #839.
- **Pergunta:** os 3 certificados de partição pendentes e a revisão passam para uma issue de `sounio-hardware`, mantendo aqui só o gate de conformidade entre repositórios?
- **Opções:** (A) mover e manter só o gate cruzado aqui; (B) manter tudo aqui.
- **Recomendação:** (A).
- **Teste discriminante:** um gate em `scripts/ci` que lê o recibo de `sounio-hardware` (hoje `grep -rln sounio-hardware scripts/ci` não encontra nada).
- **Causa localizada:** lado Sounio: `scripts/ci/eisa_h_zd_reference_gate.sh` e `tests/stdlib/eisa/test_eisa_h_zd.sio` existem; o head de `sounio-hardware` é `23fa553d…`, sem mudança.

<a id="d-10"></a>
## D-10 — Cadeia de bootstrap `stage0 → boot4`

- **Origem:** #725.
- **Pergunta:** a cadeia C `stage0 → boot4` deve voltar a reproduzir o ponto fixo atual, ou a documentação passa a nomear a seed commitada como raiz de proveniência?
- **Opções:** (A, S) reescrever `bootstrap/README.md:14,35,48` para nomear a seed e `bin/souc-lean-single-x86_64.SeedReceipt.json` / `verify_lean_seed.sh SOUNIO_SEED_DDC=1` como raiz, declarando que boot4 não analisa o `lean_single.sio` atual; (B, L) elevar os tetos do boot4 e refazer a cadeia.
- **Recomendação:** (A) agora; (B) só se a cadeia C for requisito de confiança.
- **Teste discriminante:** `timeout 180 artifacts/bootstrap/boot4.elf self-hosted/compiler/lean_single.sio g1.elf` deve terminar com rc=0 e gerar o ELF (opção B). Hoje: rc=124 após 181 s, `BAD NODE node=262143` em laço (log de 1,33 GB); `bootstrap/boot4_a1.elf` dá rc=139.
- **Causa localizada:** `nd_cap=262143` do boot4 contra `tokens=362541` do `lean_single.sio` atual (2.396.418 B); `Makefile:32-33` ainda usa `./bin/souc-linux-x86_64`; `bootstrap/README.md:35` ainda afirma que boot4 compila `lean_single.sio`.

<a id="d-11"></a>
## D-11 — Suíte `compile-fail` neutra de engine

- **Origem:** #2241; PR #2660 limita o job Madaros aos testes `requires: madaros`.
- **Pergunta:** as asserções de `tests/compile-fail` passam a usar códigos E neutros ou continuam escopadas por `//@ requires:`?
- **Opções:** (A) padrões neutros por código E; (B) escopo por `//@ requires:`; (C) status quo.
- **Recomendação:** (A) para o que for comum aos dois engines, (B) para o resto.
- **Teste discriminante:** `SOUNIO_TEST_SOUC_BIN=bin/madaros-linux-x86_64` sobre `tests/compile-fail` com a contagem de padrões não satisfeitos (tamanho de suíte; não rodado na campanha).
- **Causa localizada:** a suíte completa roda só `lean_single` (`ci.yml:1514`, `:1571`, `SOUNIO_TEST_SOUC_BIN: /tmp/souc-stage2`); `tests/compile-fail/relevant_must_use.sio:2-3` tem `error-pattern` + `//@ ignore`; `tests/vacuous_expect_baseline.txt:152-153`.

<a id="d-12"></a>
## D-12 — Gate de regressão de corpus

- **Origem:** #2306 (parcial: #2353, #2356).
- **Pergunta:** quando e com que baseline o `madaros_corpus_regression_gate.sh` entra no CI?
- **Opções:** (A) rodar uma vez em máquina ociosa, regenerar o baseline e ligar o gate; (B) manter desligado.
- **Recomendação:** (A), mantendo os crashers de #2354/#2355 como entradas do baseline.
- **Teste discriminante:** `bash scripts/ci/madaros_corpus_regression_gate.sh` verde contra `tests/madaros_corpus_baseline.txt` (280 linhas hoje).
- **Causa localizada:** `ci.yml:1409` ("deliberately NOT wired in yet"); o gate é do tamanho do corpus e não foi rodado na campanha.

<a id="d-13"></a>
## D-13 — Os dois pipelines `-O`

- **Origem:** #2368.
- **Pergunta:** o caminho multimódulo deve rodar inline/LICM/TCO, e as cópias sombreadas de `module_loader` são removidas?
- **Opções:** (A) remover as cópias sombreadas e fazer o multimódulo usar o pipeline completo; (B) remover só as cópias; (C) documentar a diferença.
- **Recomendação:** (B) primeiro (puro, sem mudança de comportamento), (A) depois, refazendo a varredura `-O`.
- **Teste discriminante:** um único símbolo `compile_multimodule_native` no repositório e a varredura `-O` repetida com os mesmos resultados nos dois caminhos.
- **Causa localizada:** definidas duas vezes: `compile_multimodule_native` (`module_loader.sio:4082` e `module_native_driver.sio:1325`), `_advanced` (`:3273` e `:1239`), `_with_ir` (`:3922` e `:1216`); `main.sio:52` importa o conjunto de `module_native_driver`; as chamadas inline/LICM/TCO estão só em `module_loader.sio:3061-3067` e `main.sio:5571-5600`.

<a id="d-14"></a>
## D-14 — Censo de alcançabilidade

- **Origem:** #2371.
- **Pergunta:** quais dos ~314 módulos sem importador são removidos?
- **Opções:** (A) corrigir a lista de raízes (inclui drivers compilados por gate, p.ex. `compiler/native_compile_driver.sio`, `ci.yml:1077`), recontar e decidir módulo a módulo; (B) manter.
- **Recomendação:** (A).
- **Teste discriminante:** o censo refeito com as raízes completas reporta a nova contagem de módulos inalcançáveis (`ir/locality.sio` 3500 linhas, `effects/handlers.sio` 2991, `ir/incremental.sio` 2936, `check/type_erasure.sio` 3179 têm 0 importadores).
- **Causa localizada:** a lista de 4 raízes do censo estava incompleta (`scripts/ci/epistemic_witness_gate.sh:35`, `epistemic_monotonicity_gate.sh:45` compilam o driver diretamente).

<a id="d-17"></a>
## D-17 — Variantes de enum com payload

- **Origem:** #645.
- **Pergunta:** a issue vira "superada pela recusa fail-closed KL-20" e o recurso passa a ser rastreado em uma única issue, ou #645 permanece como a issue do recurso?
- **Opções:** (a) fechar #645 e rastrear o recurso (tag + campos em heap, parser/checker/lowerer e `lean_single`); (b) manter #645 como issue do recurso.
- **Recomendação:** (a).
- **Teste discriminante:** `enum V { P, M { eps: f64 } }` com `V::M { eps: e }` compila e preserva o payload nos dois engines (hoje Madaros E264 deliberado e `lean_single` E200/E218).
- **Causa localizada:** Madaros recusa com E264 desde `974c29864` (testes `enum_payload_*_refused.sio`, KL-20); o lowering representa o enum como discriminante nu.

<a id="d-18"></a>
## D-18 — Aliasing de retornos de tupla-array

- **Origem:** #1582.
- **Pergunta:** fechar como não reproduzível, com a sonda como teste de regressão?
- **Opções:** (a) fechar e acrescentar `i1582b.sio` como `run-pass`; (b) manter aberta até aparecer a assinatura original.
- **Recomendação:** (a).
- **Teste discriminante:** dois retornos `(f64,[f64;4])` e `([f64;4],[f64;4])` vivos no mesmo laço, mais k1/k2/k3 `(f64,[f64;10])` de estilo RK: `bad=0` nos dois engines (hoje já é).
- **Causa localizada:** não identificada; a única descrição é a prosa de `048f4d2a0` e as sondas originais foram apagadas.

<a id="d-19"></a>
## D-19 — `Knowledge<T where {...}>`

- **Origem:** #2753. O PR #2923 (`Fixes #2753`) aplica o passo fail-closed: os dois engines recusam `where` em `Knowledge<T>` com o novo E269. Esta seção registra só o que resta decidir.
- **Pergunta:** o predicado `where` de `Knowledge<T>` passa a ser imposto pelo compilador (guarda de execução), ou a recusa E269 é o estado final?
- **Opções:** (a) guarda de execução imposta pelo compilador; (b) manter a recusa E269 (já em #2923).
- **Recomendação:** (b) agora (já aplicada), (a) como item de design.
- **Teste discriminante:** sob (a), com `Knowledge<Patient where { age >= 18 }>`: `age = 13` sai com código ≠ 0 e `age = 21` imprime `age=21` com rc=0, nos dois engines. Sob (b) ambos são erro E269 (`tests/compile-fail/knowledge_where_refinement_refused_invalid_2753.sio`). Em `main` hoje: Madaros `age=13`, rc=0; `lean_single` P0003 (fronteira ε, não o predicado) também para 21.
- **Causa localizada:** check/lower do Madaros não carregam predicado para refinamentos de `Knowledge<T>` (a expansão vive em `scripts/ontology/expand_knowledge_runtime_guards.sh`); o P0003 do `lean_single` vem da checagem de confiança.

<a id="d-21"></a>
## D-21 — SOIR: colisão de magic + versão 6

- **Origem:** #1162.
- **Pergunta:** como separar os dois formatos que hoje escrevem magic `SOIR` e versão 6?
- **Opções:** (A) dar ao legado `ir/serialize.sio` magic distinto (ou versão ≥ 0x80); (B) estender o legado no lugar; (C) congelar os dois.
- **Recomendação:** (A), urgente: (B) é o que a decisão fundadora da issue proíbe.
- **Teste discriminante:** bytes de `serialize_ir_module(empty)` entregues a `soir_v6_reader` e um contêiner seccionado v6 entregue a `deserialize_ir_module` são ambos recusados com status estável. Esse teste não existe em `main`.
- **Causa localizada:** `self-hosted/ir/serialize.sio:56` (`SOIR_VERSION: i8 = 6`, aceita também v5 em `:4781`); `ir/soir_v6_writer.sio:16` (`SOIR_V6_WRITER_VERSION: i8 = 6`), `soir_v6_reader.sio:12,199-212`; o v5 sombra de arena em `ir/soir_writer.sio:13` também coexiste.

<a id="d-23"></a>
## D-23 — `Knowledge ⊗ Knowledge` no Madaros (E245) e NaN além do teto de recursão

- **Origem:** #1706 (PRs #2933 e #2945, `Refs`).
- **Pergunta 1:** E245 ("arithmetic between two Knowledge<T> values is not supported") é retirado, passando o Madaros a baixar `kA * kB` com variância GUM, com que regra para confiança e proveniência?
- **Pergunta 2:** recursão mais profunda que o teto de desenrolamento (8 níveis, 96 inlines por ponto de chamada externo) lê NaN; é isso, ou teto maior, recusa em compilação, ou a soma conservadora das variâncias dos argumentos?
- **Recomendação:** 1: retirar E245 só junto com a regra explícita de confiança/proveniência (o `lean_single` já passa o teste com #2941); 2: recusa em compilação quando o teto é atingido (NaN nunca é um valor honesto de "não sei").
- **Teste discriminante:** 1: `tests/known_failures/madaros_gum_fo_knowledge_ops` passa a rodar e dá a variância analítica; 2: `pw(a, 8)` com valor analítico 2621,44 dá o valor ou é recusada em compilação.
- **Causa localizada:** 1: emissão de E245 em `self-hosted/check/check.sio` (exceção E230/E245 em ~8098–8130, #2140); 2: o desenrolamento por re-lowering em `self-hosted/ir/lower.sio` (PR #2945).

<a id="d-28"></a>
## D-28 — Operadores infixos `!` e `~`

- **Origem:** #2249; PR #2921 aplica a opção recomendada.
- **Pergunta:** `a ! b` e `a ~ b` têm significado?
- **Opções:** (A) não: recusa fail-closed nos dois engines; (B) dar-lhes uma semântica.
- **Recomendação:** (A), já implementada em #2921 (a confirmar).
- **Teste discriminante:** `x ~ y`, `p ! q` e `x! (y)` são erro de compilação; o prefixo (`!t`, `~x`), `^` e `!=` continuam funcionando (`tests/compile-fail/infix_*_refused_issue_2249.sio`, `tests/run-pass/prefix_bang_tilde_line_start_2249.sio`).
- **Causa localizada:** `self-hosted/lexer/token.sio` `tk_precedence` (`Bang`/`Tilde` = 13) e `self-hosted/parser/exprs.sio` `token_to_binop` (padrão `_ => BinaryOp::OpAdd`).

<a id="d-29"></a>
## D-29 — Seed macOS ARM64 com recibo de proveniência

- **Origem:** #821.
- **Pergunta:** quem promove um seed arm64 construído por cross-build, e com que recibo?
- **Opções:** (a) promover num host Apple M-series com um `SeedReceipt` JSON espelhando o x86; (b) manter os artefatos de cross-build.
- **Recomendação:** (a); o bloqueio antigo (#2431) já foi resolvido.
- **Teste discriminante:** `artifacts/self-hosted/souc-self-hosted-arm64-macos` com recibo sidecar verificado por um gate; hoje não há recibo (`git ls-files artifacts/self-hosted`).
- **Causa localizada:** nenhum commit toca o binário arm64 desde `d834ab4e5`; só a metade do `flock` foi tratada (#822, `scripts/dev/souc-build-lock.sh:45-51`). Precisa de host Apple, que a campanha não tem.

<a id="d-30"></a>
## D-30 — Raízes precisas para o GC de handles

- **Origem:** #1491 (itens 1–2; o item 3, `precise_stack_maps` honesto, está corrigido em `603fcf600`, e a reclamação por região em `2f6f0472b`).
- **Pergunta:** implementar mapas de raízes precisos e ligar o reset de frame no backend ativo, ou aceitar que grafos longevos que escapam não são recuperados?
- **Opções:** (A) raízes precisas + `native_v2_emit_gc_empty_frame_reset` no backend vivo; (B) manter a contagem conservadora com a reclamação por região.
- **Recomendação:** (A) se houver carga real com > 2^22 handles vivos; senão (B), documentado.
- **Teste discriminante:** um programa que mantém > 2^22 handles vivos em retornos que escapam termina sem "handles full" (rc=182). A sonda `docs/handoff/repros/handle_table_182_per_construction_madaros.sio` já passa; esta ainda precisa ser escrita.
- **Causa localizada:** `self-hosted/native/codegen_x86_linux.sio:247` (`precise_stack_maps: false`), `:3559` (`native_v2_emit_gc_empty_frame_reset`, só chamada no legado `native/codegen.sio:3993,4013`); `native/gc.sio:39-41,63`.

<a id="d-31"></a>
## D-31 — Reborrow por parâmetro `&!` pode aliasar?

- **Origem:** #1581 (a divergência de engines da issue está corrigida; este é o resíduo que as duas engines compartilham).
- **Pergunta:** `sub(&(*y), &!(*y))` com `y: &![f64; 4]` (um empréstimo compartilhado e um exclusivo do mesmo lugar, via reborrow, na mesma chamada) deve ser recusado como `sub(&y, &!y)` é?
- **Opções:** (a) sim: o reborrow `&(*p)`/`&!(*p)` conta como empréstimo do lugar subjacente nos dois engines (E037/E038); (b) não: reborrows são livres e a responsabilidade é do chamador.
- **Recomendação:** (a): (b) deixa um alias mutável aceito por `check`, o mesmo tipo de furo de #888.
- **Teste discriminante:** `fn step(y: &![f64; 4]) with Mut { sub(&(*y), &!(*y)) }` (e `sub(&s.y, &!s.y)` com `s: &!St`) recusado com E037/E038 nos dois engines; hoje Madaros e `lean_single` aceitam e imprimem `4.000000`.
- **Causa localizada:** a composição de empréstimos por campo/lugar de `self-hosted/check/borrow.sio` (commit `4f2a392ca`) não normaliza `&(*p)` para o lugar de `p`; o `lean_single` tem a mesma lacuna em `lean_single.sio`.

<a id="d-32"></a>
## D-32 — `spawn {}`, `.await` e `async fn` no Madaros

- **Origem:** #2604 (a metade "Madaros não faz type-check do próprio driver" está corrigida em `15e8184ef`; este é o resíduo).
- **Pergunta:** o Madaros deve implementar `spawn { ... }`, o postfix `.await` e `async fn`, ou o recurso fica só no `lean_single` até haver decisão sobre o runtime assíncrono?
- **Opções:** (a) implementar no Madaros (AST + parser + checker de efeito `Async` + lowering); (b) manter recusa fail-closed, com mensagem própria no lugar de E012, e escopar `tests/run-pass/async_spawn.sio` com `//@ requires: lean_single`; (c) retirar o recurso da linguagem.
- **Recomendação:** (b) agora (a recusa já é fail-closed, mas o E012 "this type has no field named" engana: aponta um campo inexistente em vez do recurso ausente), (a) quando o modelo de execução de `Async` for decidido.
- **Teste discriminante:** `tests/run-pass/async_spawn.sio` imprime `spawn_basic: PASS` e `spawn_multiple: PASS` no Madaros (opção a), ou o Madaros o recusa com uma mensagem que nomeia `spawn`/`.await` (opção b). Hoje: Madaros `error[E012] … this type has no field named` (×8), `lean_single` `PASS`/`PASS`.
- **Causa localizada:** `self-hosted/parser/exprs.sio` não tem nó de AST para `await` nem para `spawn` (`grep -n 'await' self-hosted/parser/exprs.sio` não encontra nada), então `.await` é lido como acesso a campo e o checker de `self-hosted/check/` levanta E012.

<a id="d-34"></a>
## D-34 — Primitiva intervalar certificada: as cinco decisões

- **Origem:** [proposta](../proposals/CERTIFIED_INTERVAL_PRIMITIVE_2026-10-22.md) (2026-10-22). Vira ADR quando estas cinco forem tomadas.
- **Pergunta:** como a incerteza estatística `u` e o envelope rigoroso `e` convivem no `Knowledge`, e quais alvos entram no Marco 0?
- **Causa localizada:** nenhuma no código: hoje a propagação do `Knowledge` não carrega termo de arredondamento (`self-hosted/check/epistemic.sio`, `self-hosted/ir/lower.sio`; medido em `main` `27afa340`), então nenhuma opção muda o comportamento atual.

| # | decisão | opções | recomendação | teste discriminante |
|---|---|---|---|---|
| D2 | `u` e `e` separados no `Knowledge` | separar, com conversão `e → u` só explícita e registrada na proveniência; ou somar em quadratura | separar (a soma implícita proposta em 2026-09-23 sai antes de entrar) | dobrar `e` em `u` sem `e_to_typeB(…)` é erro de compilação |
| D3 | propagar `u` sob `e` não degenerado | (a) Jacobiano intervalar; (b) recusar; (c) Jacobiano pontual só via `assume_point` | (a) por padrão, (c) como saída explícita | `u` de `f(x)` com `e` largo não usa o Jacobiano no ponto médio sem `assume_point` |
| D6 | FMA intervalar explícita | permitir `ifma.rd/.ru`; ou proibir | permitir; a contração implícita continua proibida | a mesma expressão com e sem `ifma` explícita: só a explícita emite FMA |
| D10 | `if` sobre condição só "possivelmente" verdadeira | (a) erro de compilação, com `hull_branches` explícito; (b) alargar em silêncio | (a) | `if [0,2] > [1,1] { … }` recusado sem `hull_branches` |
| D11 | alvos do Marco 0 | x86 EVEX, RISC-V F/D e RVV (Spike), PTX; AMD no Marco 0b; Metal, AMX e U250 fora | como na tabela da proposta | identidade bit a bit dos extremos entre os alvos do Marco 0 no corpus de casos-limite |

