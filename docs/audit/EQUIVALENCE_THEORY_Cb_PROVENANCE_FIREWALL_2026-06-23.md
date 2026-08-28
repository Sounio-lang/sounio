<!-- docs:meta
topic_id: repo.docs.audit.equivalence-theory-cb-provenance-firewall-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.equivalence-theory-cb-provenance-firewall-2026-06-23
-->

# Equivalence Theory — Feature C-b: causal-set provenance quasi-metric (FIREWALLED HEURISTIC)
## Teoria da Equivalência — Recurso C-b: quase-métrica de proveniência por conjunto causal (HEURÍSTICA COM FIREWALL)

*Date / Data:* 2026-06-23 · *Branch:* `feat/equivalence-theory-exactness-gate`
*Author of record:* Demetrios C. Agourakis · *Standing co-author:* Dionisio Chiuratto Agourakis

---

## EN — Summary

The last, deliberately isolated feature. A provenance DAG is read as a **causal set**: order + a counting measure → a quasi-metric "evidential distance" (Bombelli–Lee–Meyer–Sorkin, *order + number = geometry*). A conclusion **timelike-separated** from its evidence reads as "properly derived"; **spacelike** as an "unjustified leap".

**This is generative analogy, NOT isomorphism**, and it is firewalled accordingly. The load-bearing deliverable here is not the math — it is the **build-level firewall** that guarantees this speculation can never reach a correctness path.

### What was built

- **`stdlib/experimental/heuristic/provenance.sio`** — `ProvDag` (≤64-node provenance DAG; edge `i→j` = "i is evidence for j", `i ≺ j`) with:
  - `prov_longest_chain(a,b)` — the **counting measure**: the longest derivation chain (causal-set discrete proper time); `0` if `a==b`, positive if `b` is derivable from `a`, `-1` if `b` is not in `a`'s causal future.
  - `prov_evidential_distance(a,b)` — the **asymmetric quasi-metric** (forward = chain length, backward = spacelike sentinel `-1`).
  - `prov_is_timelike` / `prov_is_spacelike` — "properly derived" vs "unjustified leap".
  - Every public symbol is annotated **SPECULATIVE**; the module opens with a metaphor-firewall banner restating that causal-set → metric recovery needs extra axioms (manifoldlikeness, embeddability) that a real provenance DAG has no a priori reason to satisfy.
- **`scripts/ci/heuristic_firewall_gate.sh`** — the firewall. It **fails the build** if any *correctness-critical* module imports the `experimental::heuristic` namespace. Correctness-critical = `stdlib/**` (except `stdlib/experimental/**`), `self-hosted/**`, `bootstrap/**`. Allowed = `tests/**`, `examples/**`, and the namespace itself.

### Verification

| Check | Result |
|---|---|
| Firewall gate on the repo | **PASS** — no correctness-critical module imports `experimental::heuristic` |
| Firewall self-demo (`--demo`) | **PASS** — a planted `self-hosted/check/violator.sio` import is caught; a `tests/` import and the namespace's own self-import are allowed |
| Provenance math (`tests/run-pass/heuristic_provenance.sio`) | **exit 0** (10 assertions): longest-chain = 3 over the direct shortcut; causal order asymmetric; quasi-metric forward/backward; timelike vs spacelike classification — compiled by the seed `bin/souc-lean-single-x86_64` |
| `provenance.sio` standalone `souc check` | **OK** |

### Honest accounting — "no claims X, delivers Y"

- **Heuristic only; never a metric guarantee, never load-bearing.** The quasi-metric is an evidence-RANKING signal. The firewall gate enforces §6.2 mechanically and is wired into CI — it runs in the `contracts` job of `.github/workflows/ci.yml` (the `--demo` self-test then the real scan), so the isolation is enforced on every PR and push.
- **Analogy, explicitly.** The module documents that imposing a Lorentzian/causal-set structure on a provenance graph is the *same trap* as imposing Lorentz invariance on a psychiatric state space (the prompt's own §6.1 / §8 flag). No manifoldlikeness or embeddability is claimed or checked.
- **Verified under the seed** (the run-pass test passes there; the prebuilt madaros has the pre-existing by-value-struct codegen bug). `ProvDag` carries a 4096-bool adjacency array; queries take it by shared reference to avoid copies.

### Files

- `stdlib/experimental/heuristic/provenance.sio`, `scripts/ci/heuristic_firewall_gate.sh`, `tests/run-pass/heuristic_provenance.sio`.

---

## PT — Resumo

O último recurso, deliberadamente isolado. Um DAG de proveniência é lido como **conjunto causal**: ordem + uma medida de contagem → uma quase-métrica de "distância evidencial" (Bombelli–Lee–Meyer–Sorkin, *ordem + número = geometria*). Uma conclusão **separada por intervalo tipo-tempo** de sua evidência lê-se como "devidamente derivada"; **tipo-espaço** como "salto injustificado".

**Isto é analogia geradora, NÃO isomorfismo**, e está com firewall conforme. A entrega central aqui não é a matemática — é o **firewall em nível de build** que garante que esta especulação jamais alcance um caminho de correção.

### O que foi construído

- **`stdlib/experimental/heuristic/provenance.sio`** — `ProvDag` (DAG de ≤64 nós; aresta `i→j` = "i é evidência de j", `i ≺ j`): `prov_longest_chain` (medida de contagem = cadeia de derivação mais longa, o "tempo próprio" discreto), `prov_evidential_distance` (quase-métrica assimétrica), `prov_is_timelike`/`prov_is_spacelike`. Todo símbolo público anotado **SPECULATIVE**; banner de firewall de metáfora no cabeçalho.
- **`scripts/ci/heuristic_firewall_gate.sh`** — o firewall. **Falha o build** se qualquer módulo *crítico para correção* (`stdlib/**` exceto `stdlib/experimental/**`, `self-hosted/**`, `bootstrap/**`) importar o namespace `experimental::heuristic`. Permitidos: `tests/**`, `examples/**`, o próprio namespace.

### Verificação

Gate no repositório: **PASS** (nenhum módulo crítico importa). Autodemo (`--demo`): **PASS** — pega um import plantado em `self-hosted/`, permite import em `tests/` e o auto-import do namespace. Matemática (`tests/run-pass/heuristic_provenance.sio`): **exit 0** (10 asserções) compilado pelo seed. `souc check` isolado: **OK**.

### Prestação de contas honesta — "não prometer X e entregar Y"

- **Apenas heurística; nunca garantia métrica, nunca carregadora.** Sinal de RANKING de evidência. O gate impõe §6.2 mecanicamente e está conectado à CI — roda no job `contracts` de `.github/workflows/ci.yml` (o autoteste `--demo` e depois a varredura real), de modo que o isolamento é imposto em todo PR e push.
- **Analogia, explicitamente.** O módulo documenta que impor estrutura lorentziana/de conjunto causal a um grafo de proveniência é a *mesma armadilha* de impor invariância de Lorentz a um espaço de estados psiquiátrico. Nenhuma manifoldlikeness/embeddability é alegada ou verificada.
- **Verificado sob o seed** (o teste passa lá; o madaros pré-compilado tem o bug pré-existente de struct por valor).

### Arquivos

- `stdlib/experimental/heuristic/provenance.sio`, `scripts/ci/heuristic_firewall_gate.sh`, `tests/run-pass/heuristic_provenance.sio`.

---

## AI disclosure / Divulgação de IA (GAIDeT-style, ICMJE 2025)

Developed with AI assistance (Anthropic Claude, "Opus 4.8", Claude Code agent harness) under human direction; the module, firewall gate, test, and this bilingual note were AI-drafted and human-reviewed. The firewall's behaviour is backed by a re-runnable self-demo (catches a planted violation) and the math by a seed-compiled test (exit 0). / Desenvolvido com assistência de IA sob direção humana; o comportamento do firewall tem respaldo em autodemo reexecutável e a matemática em teste compilado pelo seed (saída 0).
