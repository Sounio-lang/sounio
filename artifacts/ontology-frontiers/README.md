# Ontology Frontiers — relatório final da meta autônoma

**Data:** 2026-08-02
**Lane:** `kimi--ontology-frontiers-20260802` (released)
**Método:** mineração de literatura via scite MCP (ontologias biomédicas /
validação semântica) → 3 fronteiras profundas, cada uma com documento de
evidência, protótipo executável e formalização Lean 4 verificada.

## Fronteiras atacadas

### 1. `epistemic-alignment-repair/`
Reparo de alinhamentos de ontologias com confiança epistêmica tipada.
Âncoras: Jiménez-Ruiz et al. 2011 (`10.1186/2041-1480-2-s1-s2`),
Solimando et al. 2016 (`10.1007/s10115-016-0983-3`), Rovai 2026
(`10.48550/arxiv.2605.09184`).
- `alignment_repair.sio` — `souc check` OK, `souc run` → `ALL PASS`.
- `formal/OntologyAlignmentRepair.lean` — soundness (`mem_repair_nil`),
  correção (`pairwise_repair_nil`), testemunha de maximalidade
  (`repair_witness_nil`); instância `Fin 5` por `native_decide`.

### 2. `epistemic-claim-status/`
Status epistêmico verificável de claims em knowledge graphs.
Âncoras: arXiv:2602.15353, arXiv:2601.21116, arXiv:2604.11759,
arXiv:2603.28444.
- `claim_status.sio` — `souc check` OK, `souc run` → `ALL PASS`.
- `formal/OntologyClaimStatus.lean` — cadeia weakest-link
  (`chainConf_le_acc`, `chainConf_le_mem`, `chainConf_ge`) e fusão DS
  (`dsNum_ge_max`), confianças em por-mil (Nat, aritmética exata).

### 3. `consistent-ontology-evolution/`
Evolução de ontologias com consistência a priori verificada.
Âncoras: Bayoudhi et al. 2018 (`10.1111/exsy.12355`), Jiménez-Ruiz et al.
2011 (consistency principle).
- `version_chain.sio` — `souc check` OK, `souc run` → `ALL PASS`.
- `formal/OntologyEvolution.lean` — transição com guarda
  (`consistent_applyEdit`), invariante a priori
  (`mem_versions_consistent`), preservação em rejeição
  (`applyEdit_reject`); importa `OntologyAlignmentRepair`.

## Verificação

- `bin/souc check` nos 3 protótipos: `check: OK`.
- `bin/souc run` nos 3 protótipos: `ALL PASS`.
- `cd formal && lake build`: **Build completed successfully** (lib inteira,
  incluindo as 3 novas roots; zero `sorry`, zero novos axioms).
- Revisão matemática obrigatória (política do repo):
  `bin/llm-offload -t math-review -p xai` → **PASS** (todos os teoremas
  [OK]; única nota estilística [TIGHTENABLE] em `conflictsAny_*`).
  Log: `.claude/llm_offload_log.md` (2026-08-02); saída completa em
  `LEAN_MATH_REVIEW_XAI.md`.

## Limitações do compilador encontradas (documentadas nos artefatos)

1. Refinamentos `where result.confidence >= ...` (usados em exemplos como
   `examples/epistemic_dempster_shafer.sio`) **não são aceitos pelo parser
   Madaros atual** — os contratos foram enforcement de runtime.
2. Arrays de structs e arrays sem inicialização splat (`var a: [f64; N]`)
   **segfaultam em runtime**; protótipos usam arrays paralelos primitivos
   com splat (`= [0.0; N]`).
3. A equivalência entre o greedy "drop-weaker" par a par (protótipo F1) e
   o fold por prioridade (Lean) está argumentada, não mecanizada — lacuna
   registrada no FRONTIER.md da fronteira 1.

## Lacunas por fronteira

Ver seção "Lacunas e riscos" de cada `FRONTIER.md`: oráculos de conflito
abstratos (não ligados a reasoner OWL/EL++), escala real (SNOMED CT 300k+
entidades) fora de escopo, álgebra de propagação escolhida (min/DS) é uma
entre várias.

## Rodada 2 — swarm de fechamento de lacunas (2026-08-02)

Quatro agentes em paralelo fecharam as lacunas documentadas:

1. **Equivalência mecanizada (fronteira 1)** —
   `formal/OntologyRepairEquivalence.lean` (~630 linhas): teorema
   `repair_iff_greedy` prova que o greedy par-a-par do protótipo e o fold
   por prioridade computam o mesmo conjunto retido, sob confianças
   distintas e conflitos em **grafo de clusters** (união disjunta de
   cliques). Descoberta: a afirmação original "confianças distintas
   bastam" é **falsa em geral** — contraexemplo mecanizado
   (`cx_equivalence_fails`): no caminho de conflitos 0—1—2 com confianças
   0<1<2, o greedy retém {2} e o fold retém {0,2}. O comentário em
   `OntologyAlignmentRepair.lean` foi corrigido.
2. **Remoção + repair-then-retry (fronteira 3)** —
   `formal/OntologyEvolutionRepair.lean`: edits `add | remove`; sublista
   de versão consistente é consistente; invariante de cadeia generalizado;
   `repair_retry` (após remover o *único* parceiro conflitante, o axioma
   rejeitado é aceito e a versão segue consistente). Protótipo
   `version_chain_removal.sio`: add 1,2,3 → add 4 rejeitado → remove 2 →
   re-add 4 aceito → {1,3,4}; `ALL PASS`.
3. **Confianças intervalares (fronteira 2)** — `formal/ClaimStatusInterval.lean`:
   `IConf {lo, hi}` por-mil; preservação de validade, contenção do
   resultado pontual, preservação de limiar no lado `lo`, e
   `ds_lo ≥ max` das fontes. Protótipo `interval_claims.sio`: `ALL PASS`.
4. **Repros de compilador** — `compiler-repros/`: 4 arquivos mínimos +
   `REPORT.md` com saídas verificadas (parse error em `where`; SIGSEGV em
   arrays de structs e arrays sem splat; controle com splat OK).
   Referência cruzada com `docs/compiler/KNOWN_LIMITATIONS.md`. Nota: o
   wrapper atual sai com código 1 (não 0) no parse error — saída de erro
   continua só em stdout.

Todas as novas formalizações passaram por math-review xai (PASS, log em
`.claude/llm_offload_log.md`). Verificação central: `lake build` verde com
as 6 roots de fronteira; `souc check`/`run` re-executados nos novos
protótipos.

Lacunas restantes (declaradas nos arquivos): equivalência greedy≡fold só
sob hipótese de cluster; repair-then-retry exige parceiro conflitante
único (remoção minimal entre vários parceiros = trabalho futuro, conecta
com a fronteira 1); p-box/GUM de segunda ordem completo segue aberto;
oráculos de conflito continuam abstratos.

## Rodada 3 — remoção minimal + gate CI (2026-08-02)

- `formal/OntologyMinimalRepair.lean`: decisão admitir-vs-rejeitar para
  candidato conflitando com **múltiplos** parceiros — o conjunto de remoção
  é forçado (todos os parceiros; unicidade/minimalidade nas duas
  direções), a decisão por massa epistêmica retida é ótima entre as duas
  opções, e ambos os ramos preservam consistência. Protótipo
  `minimal_repair_demo.sio`: `ALL PASS`. Math-review xai: PASS.
- `scripts/ci/ontology_frontiers_gate.sh` criado (ver "Gate CI").

## Rodada 4 — oráculo fundamentado em semântica EL (2026-08-02)

- `formal/OntologyELReasoner.lean`: mini lógica de descrição (axiomas
  `sub`/`disj`), semântica de Tarski, fecho transitivo de subsunção como
  sistema dedutivo indutivo, e o teorema central **`incoherent_empty`**:
  toda classe marcada incoerente pelo fecho é vazia em todo modelo da
  TBox. `oracle_sound` conecta isso ao oráculo de conflitos das fronteiras
  anteriores: pares sinalizados não podem valer simultaneamente em nenhum
  modelo. Instância `Fin 8` biomédica com 11 checks `decide`/`native_decide`.
  Math-review xai: PASS. Limitação honesta (registrada no arquivo): o
  fecho booleano computacional foi validado contra o sistema indutivo só
  na instância concreta — uma verificação geral do fecho fica para a
  próxima rodada.
- `el-grounding/` (nova fronteira): `el_conflict_demo.sio` **deriva** os
  conflitos de uma TBox em miniatura (fecho de subsunção + disjunção) em
  vez de hardcodá-los, e confirma que o oráculo derivado coincide com o
  hardcoded da fronteira 1 na instância compartilhada (mesmos sobreviventes
  do reparo).

## Rodada 5 — fecho verificado + empates determinísticos (2026-08-02)

- `formal/OntologyELClosureVerified.lean` (~700 linhas): a ponte
  computacional↔dedutiva fechada em **generalidade total** —
  `subB_iff_subDer` (o fecho booleano coincide com o sistema indutivo,
  soundness via invariante de iteração e completeness via linearização de
  derivações em walks + corte de laços com cota n+1 iterações) e
  `conflictB_iff` (o oráculo booleano É o `DerivedConflict` semântico, nas
  duas direções). A limitação honesta da rodada 4 está fechada.
  Math-review xai: PASS.
- `formal/OntologyRepairTies.lean`: a equivalência greedy≡fold estendida
  de confianças distintas para **arbitrárias**, via prioridade
  lexicográfica (conf desc, id asc) codificada injetivamente em Nat —
  `repair_iff_greedy` da rodada 2 aplicado verbatim com `conf := prio`;
  `greedyStep_prio_eq_sio` prova que o passo do greedy com prioridade é
  definicionalmente o tie-break do protótipo `.sio`; determinismo do
  greedy provado; instância `Fin 6` com empate real (m0/m1 a 0.50)
  computada pelos dois algoritmos com resultado igual. Protótipo
  `tie_repair_demo.sio`: `ALL PASS` (determinismo em duas execuções,
  sobreviventes livres de conflito, testemunhas). Math-review xai: PASS.

## Rodada 6 — dados reais + empacotamento (2026-08-02)

- **`real-data/` — validação no OAEI 2016 Anatomy track real**
  (human.owl 3.304 classes / mouse.owl 2.744, reference.rdf oficial):
  1.961 classes (cap fechado por ancestrais), 2.266 axiomas sub, 17
  disjunções, 6.638 mappings candidatos (matcher lexical Jaccard;
  P=0.187/R=0.817 contra a referência oficial), **368 conflitos derivados
  (0 entre mappings de referência)**, reparo descarta 246 (conf média
  0.41 vs 0.55 dos retidos) e **apenas 3 dos 1.238 mappings de
  referência** — o reparo epistêmico remove preferencialmente
  não-referências. Driver `real_repair_driver.sio` com espelho python
  cruzado: `ALL PASS` (~17s). Reprodução completa (URLs, sha256,
  comandos) em `real-data/REAL_RESULTS.md`.
- **`stdlib/ontology/` — empacotamento**: `closure.sio`, `repair.sio`,
  `evolve.sio` (módulos reutilizáveis, convenção `stdlib/algo`) +
  `examples/ontology_pipeline_demo.sio` end-to-end (`ALL PASS`).
- **`TECHNICAL_NOTE.md`** — nota técnica externa (5 rodadas, teoremas,
  limitações); revisão externa `llm-offload --raw`: deepseek PASS 9/10,
  xai PASS (gemini indisponível — fallback de 2 providers, conforme
  política).
- **Novos pitfalls de compilador encontrados** (a propagar para
  `compiler-repros/` numa próxima rodada): >682 statements por função
  silenciosamente descartados; arrays splat em nível de módulo com
  elementos iniciais sujos; atribuição a elemento de array f64 fora de
  `main` é no-op silencioso; thin-link multimódulo falha além de ~24k
  assignments; **forma qualificada de import (`mod::f`) miscompila**
  (mutações `&!` perdidas) — a forma nomeada funciona.

## Rodada 7 — EL+, escala sem cap, miscompile (2026-08-02)

- **`formal/OntologyELPlus.lean`** (~530 linhas): o fragmento EL⁺ que a
  SNOMED CT realmente usa — conceitos `atom | ⊤ | ⊓ | ∃r.C`, axiomas
  `sub | disj | roleSub`, semântica de Tarski com interpretação de papéis,
  sistema dedutivo de 9 regras (**nenhuma descartada**) com `der_sound`
  por indução, `incoherentP_empty`, ponte de oráculo `oracle_sound_P`, e
  projeção atômica reutilizando o fecho verificado (`subBP_sound`,
  `conflictBP_sound`). Instância `Fin 8 × Fin 2` estilo SNOMED
  (Pneumonia ⊑ ∃RoleGroup.(Lung ⊓ Inflammation) ⊑ ∃RoleGroup.Organ).
  Composição de papéis `r∘s⊑t` declarada como próxima fronteira.
  Math-review xai: PASS.
- **Escala sem cap** (`real-data/scale/`): a TBox Anatomy **completa**
  (3.304 classes, sem cap) roda até `ALL PASS` — 21.859 arestas de fecho,
  368 conflitos, kept 6.392 / dropped 246, **byte-idêntico** à rodada 6
  (confirmando que o cap ancestral era lossless). Estratégia esparsa
  (BFS por classe, sem matriz N²): **nenhum teto encontrado** (estrela de
  10M classes em 3.5s; cadeia de 30k classes com 450M arestas em 8.9s).
  Tetos reais medidos: muro de ~24k statements por compilação (single e
  multimódulo); denso N² OK até N=50.000 (7.5GB); N=100k → handoff Slurm
  documentado (não executado, conforme regra do repo).
- **Miscompile caçado** (`compiler-repros/` + `docs/audit/`): os 5
  pitfalls reproduzidos com limiares refinados (P1 é uma família
  dependente de forma — 256 a 682; P3 = P1 com f64 RMW; P4 = 10.2k/10.4k
  statements). **Causa-raiz do P5 (import qualificado)**:
  `self-hosted/ir/lower.sio:15698-15717` mangles `m::f` → `m_f`; funções
  importadas registradas como `f`, então o linker fabrica um stub sem
  corpo e a chamada cai nele silenciosamente — confirmado empiricamente
  no compilador não modificado. Patch candidato (não aplicado, dry-run
  OK) em `compiler-repros/qualified_import_fix_candidate.diff`; auditoria
  completa em `docs/audit/QUALIFIED_IMPORT_MISCOMPILE_2026-08-02.md`.

## Demo executável — fecho EL⁺ completo (rodada 8)

- **`examples/ontology_elplus_closure_demo.sio`**: espelho executável do
  motor de saturação role-aware de `formal/OntologyELPlusClosureComplete.lean`
  (o fecho `closeSatF` completo, com `subBPlusC_iff` / `conflictBPlusC_iff`),
  instanciado na TBox SNOMED de `formal/OntologyELPlus.lean` (`Fin 8 × Fin 3`).
  Conceitos internados em tabela (átomos 0..7, ⊤, `Lung ⊓ Inflammation` e as
  restrições existenciais necessárias), relações `S` (subsunção), `R`
  (arestas de papel com filler base) e `rclos` (fecho da hierarquia de
  papéis) como arrays primitivos paralelos; a rodada de fixpoint reproduz
  o `crStep` (transitividade, ⊓-elim/intro, stoR/RtoS, Rmono, roleSub via
  `rclos`, composição `DirectSite ∘ PartOf ⊑ RoleGroup`) e itera até
  estabilidade (3 rodadas). Checks: `Pneumonia ⊑ ∃RoleGroup.Organ` = true,
  `conflict(Pneumonia, Drug)` = true, `conflict(DrugInducedDisorder, ele
  mesmo)` = true (testemunha de incoerência), `Organ ⊑ Disorder` = false.
  `souc check` OK; `souc run` → `ALL PASS` (marcadores `//@ run-pass` /
  `//@ expect-stdout: ALL PASS`).

## Rodada 9 — fecho EL⁺ role-aware no pipeline de dados reais (2026-08-04)

Integração do fecho booleano EL⁺ com papéis (o motor verificado de
`formal/OntologyELPlusClosureComplete.lean`) no pipeline OAEI Anatomy:

- **`real-data/extract_tbox.py`** não descarta mais as restrições
  anônimas: `owl:Restriction`/`someValuesFrom` viram linhas
  `exsub <ont> <child> <role> <filler>` (1.637 mouse / 1.662 human),
  `owl:ObjectProperty` + `rdfs:subPropertyOf` viram `roleSub`,
  `owl:propertyChainAxiom` vira `roleComp`, e uma nova `roles.tsv`
  tabela os papéis. Classes/sub/disj extraídos são byte-idênticos às
  rodadas anteriores.
- **`real-data/scale/gen_elplus_data.py`** — espelho python do fixpoint
  de 8 regras (transitividade, ⊓-elim/intro, stoR/RtoS, Rmono, roleSub,
  roleComp) sobre o universo internado completo (átomos + ⊤ + ∃r.f para
  todo papel r e filler base; U = 9.915 conceitos), com cross-check
  packed (máscaras de bits de ancestrais + fórmulas reduzidas) — o script
  aborta se as duas representações divergirem.
- **`stdlib/ontology/elplus.sio`** — módulo reutilizável: variante densa
  (fixpoint de 8 regras, ≤64 conceitos / 8 papéis / 8 cadeias) e
  variante esparsa (BFS por classe + expansão in-place de ancestrais,
  ≤4.096 classes). As matrizes de trabalho da variante esparsa são
  globais do próprio módulo: **arrays em nível de módulo passados por
  `&!` miscompilam** (mutações caem num stub do linker; spreads grandes
  de índice → SIGSEGV) — novo pitfall encontrado nesta rodada, variante
  da família P5 documentada em `compiler-repros/`.
- **`real-data/scale/elplus_scale_driver.sio`** — Parte A: instância
  sintética estilo SNOMED (40 conceitos) via variante densa, exercitando
  ⊓/roleSub/roleComp (as 6 queries da demo da rodada 8 + contagens do
  espelho: 201 células S, 140 arestas de papel). Parte B: TBox Anatomy
  humana completa com papéis via variante esparsa — 21.859 arestas de
  fecho atômico (= rodada 7), **21.761 arestas de papel com fonte
  atômica / 72.089 totais**, 103.863 células S no universo internado,
  **736 conflitos derivados — byte-idêntico à rodada 7**: os papéis
  estendem a subsunção (alvos existenciais) sem alterar a disjunção
  atômica, logo o reparo da rodada 7 carrega-se sem alteração (m_keep /
  m_conf não re-emitidos). `ALL PASS`.
- **Limitação honesta** (afirmada no driver): a track Anatomy tem UM
  papel ativo (`part_of`; a segunda propriedade declarada,
  `ObsoleteProperty`, nunca aparece em restrições) e ZERO axiomas
  roleSub/roleComp — essas regras são exercidas apenas na instância
  sintética da Parte A.

## Rodada 10 — EL+ role-aware nos drivers de repair (2026-08-04)

Integração do `stdlib/ontology/elplus.sio` nos três drivers de repair,
que passam a computar conflitos com o fecho role-aware (o motor
verificado de `formal/OntologyELPlusClosureComplete.lean`) em vez do
fecho atômico/oráculo hardcoded:

- **`stdlib/ontology/elplus.sio`** — 3 novos exports de integração (API
  existente intacta): `elplus_derive_conflicts` (deriva a relação de
  conflito entre mappings a partir da matriz EL+ densa fechada; saída no
  mesmo stride 256 de `ontology::closure::derive_conflicts`, então
  `ontology::repair` pluga sem mudança), `elplus_subsumes_sparse` e
  `elplus_edge_sparse` (acessores O(1) de leitura sobre as matrizes de
  trabalho esparsas, para drivers consultarem o fecho role-aware sem
  cruzar fronteira `&!` com arrays de módulo).
- **`real-data/real_repair_driver.sio`** — o fixpoint atômico próprio
  foi substituído pela variante esparsa do elplus (BFS por classe +
  seeding dos fillers + expansão de ancestrais); todas as consultas de
  fecho passam pelo motor EL+. `gen_sounio_data.py` agora carrega as
  linhas `exsub` do `tbox.txt` (extração da rodada 9): **862 de 1.662**
  restrições existenciais `C ⊑ ∃part_of.F` sobrevivem ao cap (ambos os
  endpoints mantidos; papel ativo único afirmado), emitidas como
  `ex_c`/`ex_f` (e `h_sub` paddado a 4096 para casar com as assinaturas
  do stdlib). Novos valores de espelho: `expected_exsub()=862`,
  `expected_closure_edges()=12669`, `expected_role_edges_atom()=10801`.
  **Saída byte-idêntica às rodadas 6-7/9, documentada e intencional**:
  o perfil Anatomy não tem conjunções, tem um único papel ativo, zero
  roleSub/roleComp e endpoints de disjunção todos atômicos, e nenhuma
  regra EL+ adiciona alvos atômicos de subsunção além do fecho atômico
  (stoR/RtoS/Rmono só alcançam alvos existenciais) — logo os conflitos
  derivados role-aware são EXATAMENTE os atômicos: 736 pares ordenados,
  kept 6.392 / dropped 246, mesmo top-5; as duas linhas da camada de
  papéis (exsub, role edges) são o único acréscimo à saída. O espelho
  python aborta se as premissas do perfil forem violadas (endpoint
  não-atômico de disjunção ou segundo papel ativo), então uma extração
  futura que introduzisse conflitos role-derivados dispara o gerador em
  vez de mudar o repair silenciosamente.
- **`epistemic-alignment-repair/alignment_repair.sio`** — o oráculo
  hardcoded `fn conflicts` foi substituído pela variante densa do elplus
  (`elplus_fixpoint` + `elplus_derive_conflicts`). A mini TBox ganha uma
  camada de papel (`heart ⊑ ∃part_of.Organ`, `∃part_of.Organ ⊥
  DrugClass`): o conflito CONCEITO `conflict(heart, drugclass)` é
  genuinamente role-derivado (invisível ao fecho atômico), enquanto os
  conflitos de MAPPING permanecem exatamente `{m0-m1, m2-m3}` — a
  instância compartilhada de 5 mappings e os sobreviventes
  `{m0, m2, m4}` não mudam.
- **`examples/ontology_pipeline_demo.sio`** — as fases de fecho +
  derivação de conflitos migraram de `ontology::closure` para
  `ontology::elplus` (variante densa, mesma TBox estendida); as fases de
  repair (`ontology::repair`) e evolução (`ontology::evolve`) não mudam.
- Gate `scripts/ci/ontology_frontiers_gate.sh`: **12/12 OK**.

## Rodada 11 — fecho EL+ role-aware em ontologia real ROLE-RICA (GO/RO) (2026-08-04)

A limitação honesta da rodada 9 (Anatomy: UM papel ativo, ZERO axiomas
roleSub/roleComp) é fechada com dados reais role-ricos: **GO
(`go-plus.owl`, 237 MB) + RO (`ro.owl`)**, baixados de
`purl.obolibrary.org` para `real-data/downloads/`. O fallback sintético
(Track A) não foi necessário.

- **`real-data/extract_tbox.py --go`** — novo modo GO/RO:
  `owl:TransitiveProperty` (elemento ou `rdf:type`) vira a cadeia
  `r ∘ r ⊑ r`; `rdfs:subPropertyOf` vira `roleSub`;
  `owl:propertyChainAxiom` (2 membros; RO usa `rdf:Description
  rdf:about=`, não `owl:ObjectProperty rdf:resource=`) vira `roleComp`.
  Slice **ancestor-closed** de go-plus com raiz `GO:0051301` ("cell
  division", cone de 50 descendentes): a política é fillers/pais/parceiros
  de disjunção **somente GO** (sem isso o fecho ancestral explode:
  2.263 classes para GO:0006915; medido) + parceiros de disjunção. O
  conjunto de papéis é **RO-fechado** (superpropriedades e alvos de
  cadeias cujos membros são usados), o que adiciona `overlaps`
  (RO:0002131) — papel que só recebe arestas derivadas. Restrições no
  lado superclasse (`∃r.F ⊑ C`) foram sondadas: **0 ocorrências** em
  go-plus. Resultado: **H=204 classes, NR=8 papéis, 253 sub, 93 exsub,
  43 disj, 2 roleSub (part_of/has_part ⊑ overlaps), 9 roleComp**
  (transitividade de part_of/has_part/overlaps/regulates + cadeias
  cruzadas como `has_part ∘ part_of ⊑ overlaps`), universo internado
  U = 1.845 conceitos (cap U ≤ 2.048 para viabilidade densa).
- **`real-data/scale/gen_elplus_data.py --go`** — o mesmo espelho python
  do fixpoint geral de 8 regras da rodada 9, agora sobre o slice GO,
  mais: projeção atômica (bitmask de ancestrais só do sub estatuído),
  **ablações** (fixpoint sem roleComp / sem roleSub) e a **asserção do
  teorema de perfil**: sem conjunções e sem restrições superclasse,
  papéis não adicionam subsunções/conflitos ATÔMICOS (o script aborta se
  violado). Emite `go_elplus_data.sio` (400 assignments; packing
  `child*1024+parent`, `(child*32+role)*1024+filler`) e
  `go_elplus_driver.sio` com os valores do espelho em `go_expected_*()`.
- **`real-data/scale/go_elplus_driver.sio`** — driver auto-contido: a
  capacidade densa do stdlib (64 conceitos) não comporta U=1.845 e
  matrizes de módulo não cruzam fronteira `&!`, então as matrizes de
  trabalho são globais do próprio driver (sb/sx separadas + cubo de
  papéis; stor/rtos como helpers guardados). Três fixpoints: completo,
  sem roleComp, sem roleSub — mais a projeção atômica própria.
  **`ALL PASS` em ~6 s** (compilação + run), todos os 16 números iguais
  ao espelho.
- **Resultados**: 24.524 células S / 21.628 arestas de papel (3.380 com
  fonte atômica = 3.380 alvos existenciais revelados por papéis);
  1.051 arestas atômicas role-aware **= projeção atômica**; 8.436
  conflitos ordenados **= projeção atômica** (teorema de perfil verificado
  computacionalmente nos dois lados). Ablações: sem roleComp → 18.006
  células S / 15.110 arestas (**roleComp contribui 6.518 arestas — a
  família dominante neste dado**); sem roleSub → 21.834 / 18.938
  (roleSub contribui 2.690). Comparação com a rodada 9: em Anatomy
  roleSub/roleComp eram ZERO; aqui respondem por 42% das arestas de papel.
- **Limitações honestas**: slice de 204 classes (cap por viabilidade
  densa; GO completo tem ~52k classes GO / 85k declaradas); fillers
  externos (CHEBI/CL/UBERON) excluídos; como go-plus não tem restrições
  superclasse nem conjunções extraídas, os conflitos atômicos não mudam
  com papéis — **medido**, não assumido.
- Gate `scripts/ci/ontology_frontiers_gate.sh`: **13/13 OK** (driver GO
  adicionado).
- Revisão matemática obrigatória (política do repo):
  `bin/llm-offload -t math-review -p xai` sobre as 4 claims (teorema de
  perfil, bijeção stoR/RtoS, equivalência do fixpoint, framing das
  ablações) → **PASS**, todos [OK]. Log em
  `agent_logs/go_elplus_offload_2026-08-04.md` (o log canônico
  `.claude/llm_offload_log.md` estava sob claim ativa de outra lane).


## Arquivos criados

- `artifacts/ontology-frontiers/{epistemic-alignment-repair,epistemic-claim-status,consistent-ontology-evolution}/FRONTIER.md`
- `artifacts/ontology-frontiers/epistemic-alignment-repair/alignment_repair.sio`
- `artifacts/ontology-frontiers/epistemic-claim-status/{claim_status.sio,interval_claims.sio}`
- `artifacts/ontology-frontiers/consistent-ontology-evolution/{version_chain.sio,version_chain_removal.sio}`
- `artifacts/ontology-frontiers/compiler-repros/` (4 repros + REPORT.md)
- `artifacts/ontology-frontiers/LEAN_MATH_REVIEW_XAI.md`
- `formal/OntologyAlignmentRepair.lean`, `formal/OntologyClaimStatus.lean`,
  `formal/OntologyEvolution.lean`, `formal/OntologyRepairEquivalence.lean`,
  `formal/OntologyEvolutionRepair.lean`, `formal/ClaimStatusInterval.lean`
- `scripts/ci/ontology_frontiers_gate.sh` — gate CI standalone (rodada 3,
  lane `frontier-gate`; ver seção "Gate CI").
- `artifacts/ontology-frontiers/consistent-ontology-evolution/minimal_repair_demo.sio` (rodada 3)
- `artifacts/ontology-frontiers/el-grounding/{FRONTIER.md,el_conflict_demo.sio}` (rodada 4)
- `formal/OntologyMinimalRepair.lean` (rodada 3),
  `formal/OntologyELReasoner.lean` (rodada 4)
- `formal/OntologyELClosureVerified.lean`,
  `formal/OntologyRepairTies.lean` (rodada 5)
- `artifacts/ontology-frontiers/epistemic-alignment-repair/tie_repair_demo.sio` (rodada 5)
- `examples/ontology_elplus_closure_demo.sio` (demo executável do fecho EL⁺
  completo — ver seção acima)
- `stdlib/ontology/elplus.sio` (rodada 9 — fecho EL⁺ role-aware: variante
  densa + variante esparsa)
- `artifacts/ontology-frontiers/real-data/scale/gen_elplus_data.py`
  (rodada 9 — espelho python do fixpoint de 8 regras + cross-check packed)
- `artifacts/ontology-frontiers/real-data/scale/{elplus_data.sio,elplus_synth_data.sio,elplus_scale_driver.sio}`
  (rodada 9, gerados)
- `artifacts/ontology-frontiers/real-data/roles.tsv` (rodada 9, gerado)
- `artifacts/ontology-frontiers/real-data/downloads/{go-plus.owl,ro.owl}`
  (rodada 11 — GO + RO, 237 MB + 1,2 MB)
- `artifacts/ontology-frontiers/real-data/{go_elplus_tbox.txt,go_roles.tsv,go_classes.tsv}`
  (rodada 11, gerados pelo modo `--go` do extract_tbox.py)
- `artifacts/ontology-frontiers/real-data/scale/{go_elplus_data.sio,go_elplus_driver.sio}`
  (rodada 11, gerados pelo modo `--go` do gen_elplus_data.py)

## Gate CI

Rodada 3 adicionou um gate standalone que re-verifica os protótipos sem
editar nenhum arquivo existente:

```bash
bash scripts/ci/ontology_frontiers_gate.sh   # funciona a partir de qualquer cwd
```

O que ele checa, para cada um dos 13 protótipos (`alignment_repair.sio`,
`claim_status.sio`, `interval_claims.sio`, `version_chain.sio`,
`version_chain_removal.sio`, `minimal_repair_demo.sio`,
`el_conflict_demo.sio`, `tie_repair_demo.sio`, `real_repair_driver.sio`,
`full_scale_driver.sio`, `elplus_scale_driver.sio` — rodada 9,
`go_elplus_driver.sio` — rodada 11 — e
`examples/ontology_pipeline_demo.sio`):

1. `./bin/souc check <file>` — exige `check: OK` na saída e ausência de
   `parse error`;
2. `./bin/souc run <file>` — exige uma linha exata `ALL PASS` na saída.

O exit code do `souc` não é confiável, então todos os vereditos vêm do
stdout capturado. O gate imprime uma linha OK/FAIL por arquivo e sai com
código 1 se qualquer protótipo falhar. Cada `souc run` leva ~30–60s; há um
timeout por arquivo (`ONTOLOGY_FRONTIERS_RUN_TIMEOUT`, default 180s). O
wrapper pode ser trocado via `SOUC_BIN`. Os repros de compilador em
`compiler-repros/` são propositalmente excluídos (eles demonstram falhas).

## Arquivos editados

- `formal/lakefile.lean` — apenas adição das 8 novas roots (permitido pela
  meta).
- `formal/OntologyAlignmentRepair.lean` — apenas o comentário de header
  (correção da lacuna de equivalência após o contraexemplo mecanizado).
- `scripts/ci/ontology_frontiers_gate.sh` — lista de protótipos (rodadas
  3-4, 9).
- `.claude/llm_offload_log.md` — linhas de log das revisões (política do
  repo).
- Rodada 10 (integração EL+ nos drivers de repair):
  - `stdlib/ontology/elplus.sio` — 3 novos exports
    (`elplus_derive_conflicts`, `elplus_subsumes_sparse`,
    `elplus_edge_sparse`); API existente intacta.
  - `artifacts/ontology-frontiers/epistemic-alignment-repair/alignment_repair.sio`
    — oráculo hardcoded → variante densa do elplus (+ camada de papel).
  - `artifacts/ontology-frontiers/real-data/real_repair_driver.sio` —
    fixpoint atômico próprio → variante esparsa do elplus.
  - `artifacts/ontology-frontiers/real-data/gen_sounio_data.py` — carrega
    `exsub`, espelha a camada de papéis, emite `ex_c`/`ex_f` +
    `expected_exsub/closure_edges/role_edges_atom`.
  - `artifacts/ontology-frontiers/real-data/tbox_data.sio` — regenerado
    (números das rodadas 6-7 preservados; role layer adicionada).
  - `examples/ontology_pipeline_demo.sio` — fases de fecho/conflito via
    elplus denso.
  - `artifacts/ontology-frontiers/real-data/REAL_RESULTS.md` — adendo da
    rodada 10 (§10) + saída atualizada (§5).
- Rodada 11 (GO/RO role-rich):
  - `artifacts/ontology-frontiers/real-data/extract_tbox.py` — modo
    `--go/--ro/--go-root`: slice ancestor-closed de GO (GO-only) +
    axiomas de papel de RO (TransitiveProperty → cadeia, subPropertyOf,
    propertyChainAxiom com membros `rdf:Description`), RO-fecho do
    conjunto de papéis, caps de papel/universo.
  - `artifacts/ontology-frontiers/real-data/scale/gen_elplus_data.py` —
    modo `--go`: espelho do fixpoint de 8 regras sobre o slice GO,
    projeção atômica, ablações roleComp/roleSub, asserção do teorema de
    perfil; emite `go_elplus_data.sio` + `go_elplus_driver.sio`.
  - `scripts/ci/ontology_frontiers_gate.sh` — 13 protótipos (driver GO).

Commits na branch: `54cef93d7` (rodadas 1-2), `156858916` (rodada 3);
rodada 4 aguardando autorização de commit.
