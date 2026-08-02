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

## Gate CI

Rodada 3 adicionou um gate standalone que re-verifica os protótipos sem
editar nenhum arquivo existente:

```bash
bash scripts/ci/ontology_frontiers_gate.sh   # funciona a partir de qualquer cwd
```

O que ele checa, para cada um dos 5 protótipos (`alignment_repair.sio`,
`claim_status.sio`, `interval_claims.sio`, `version_chain.sio`,
`version_chain_removal.sio`):

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

- `formal/lakefile.lean` — apenas adição das 6 novas roots (permitido pela
  meta).
- `formal/OntologyAlignmentRepair.lean` — apenas o comentário de header
  (correção da lacuna de equivalência após o contraexemplo mecanizado).
- `.claude/llm_offload_log.md` — linhas de log das revisões (política do
  repo).

Nada foi commitado; tudo está no worktree.
