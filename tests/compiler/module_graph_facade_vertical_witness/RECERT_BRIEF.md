# RECERT_BRIEF — current-main recertification of the ModuleGraph semantic fidelity witness

> **Status:** ready for dispatch to a recertifying agent (e.g. GLM-5.2 in long-context mode).
> **Canonical source worktree (DO NOT MODIFY):** `/tmp/sounio-modgraph-witness-20260719` on branch `witness/modulegraph-facade-vertical-20260719`, pinned at commit `b0a8af6e96ec9e29dbe4565af0712017de0b05f4`.
> **Canonical receipt (DO NOT EDIT):** `artifacts/witnesses/module_graph_facade_vertical_20260719T132620Z.json` inside that worktree.
> **Date issued:** 2026-07-19.

---

## TAREFA

Recertificação current-main do ModuleGraph semantic fidelity witness.

Crie um worktree isolado a partir do `origin/main` mais recente. Não altere nem faça rebase no worktree canonical existente.

Execute o witness duas vezes consecutivas contra um Madaros source-fresh construído do mesmo SHA de `origin/main`.

Se já existir um raw source-fresh comprovadamente construído do mesmo Git SHA, ele pode ser reutilizado somente após registrar e verificar seu SHA-256. Caso contrário, construa via Slurm / Compiler Foundry (`/home/devsounio/projects/sounio/sounio-forge submit full-compiler --source wip --gpu auto` se o host-control-plane estiver acessível; senão, documente a indisponibilidade).

---

## Transporte permitido (somente estes artefatos)

A partir do worktree canonical `/tmp/sounio-modgraph-witness-20260719`, copiar para o novo worktree:

```bash
# Em VAR_CANONICAL=/tmp/sounio-modgraph-witness-20260719
# Em VAR_NEW=<novo worktree a ser criado>

cp "$VAR_CANONICAL/scripts/dev/module_graph_facade_vertical_witness.sh" \
   "$VAR_NEW/scripts/dev/module_graph_facade_vertical_witness.sh"

mkdir -p "$VAR_NEW/tests/compiler/module_graph_facade_vertical_witness"
cp "$VAR_CANONICAL/tests/compiler/module_graph_facade_vertical_witness/"{leaf,facade,main}.sio \
   "$VAR_NEW/tests/compiler/module_graph_facade_vertical_witness/"

# O schema do receipt canonical é referência, não transportado como arquivo:
cat "$VAR_CANONICAL/artifacts/witnesses/module_graph_facade_vertical_20260719T132620Z.json" \
  > /tmp/recert_canonical_schema_reference.json
```

O script e os fixtures são os únicos artefatos autorizados a cruzar a fronteira do worktree.

---

## Entregáveis obrigatórios

1. **Git SHA exato do `origin/main` testado** — `git rev-parse HEAD` no novo worktree, antes e depois do run, para confirmar que não avançou durante o teste.
2. **SHA-256 e tamanho do Madaros source-fresh** — `sha256sum` e `wc -c` do raw ELF usado. Deve estar construído do mesmo SHA do entregável #1; registrar comando de build + log de build.
3. **SHA-256 das fixtures** — `sha256sum` de `leaf.sio`, `facade.sio`, `main.sio` no novo worktree. Devem ser idênticos aos do worktree canonical.
4. **Por mutação (8 casos: CONTROL_A, CONTROL_B, PROBE_A, PROBE_B, × 2 runs):**
   - `compile_rc`
   - `exec_rc`  
   - `stdout` (última linha, valor literal)
   - `stderr` (SHA-256 + últimas 5 linhas; o receipt canonical não serializa stderr — capturar externamente via `tee` ou redirecionamento)
   - `ELF SHA-256`
5. **Dois receipts consecutivos com a mesma classificação** — comparar `verdict`, `result_control.class`, `result_probe.class`, e todos os SHAs entre os dois runs.
6. **Delta factual contra o canonical `20260719T132620Z.json`** — tabela explicitando o que mudou (compile_rc, stdout, ELF SHA, classification) e o que permaneceu idêntico.
7. **Classificação final:** exatamente uma de:
   - `PASS` — toda mutação do corpo importado produz o stdout correspondente (`999` para CONTROL_B, `7` para PROBE_B)
   - `BLOCKED` — mesma classe observada no canonical (`closure_not_consumed_runtime`)
   - `INFRA` — não foi possível testar (Madaros não construído, launcher ausente, etc.)
   - `NON_REPRODUCIBLE` — os dois runs consecutivos produziram classificações diferentes
8. **Comandos exatos executados** — transcript literal do shell, incluindo `git fetch`, `git worktree add`, build Madaros, e as duas invocações do witness.

---

## Boundary (rigorosa)

- **Zero alterações** em `self-hosted/compiler/`, `self-hosted/parser/`, `self-hosted/check/`, `self-hosted/ir/`, `self-hosted/native/`.
- **Zero alterações** em binários (`bin/souc`, `bin/madaros`), resolvers (`scripts/lib/resolve_*.sh`), `Makefile`, CI (`.github/workflows/`, `scripts/ci/`), ou governança (`docs/governance/`, `.claude/`).
- **Não afirmar que o merge é a causa.** O receipt canonical usa `primary suspect ... pending differential IR hashes`. Manter essa framing.
- **Separar observação runtime de suspeita arquitetural** em todos os campos de boundary.
- **Não substituir ou editar os quatro receipts históricos** (`20260719T124544Z`, `T131623Z`, `T131802Z`, `T132620Z`) no worktree canonical.
- **Não abrir roadmap novo.**
- **Não promover para CI** — o `promotion_path` no receipt canonical é direção futura, não ação presente.

---

## Critério de PASS

`PASS` somente se **cada** mutação do corpo importado produzir o stdout correspondente:
- CONTROL_A: `greet=42` → stdout termina em `42`
- CONTROL_B: `greet=999` → stdout termina em `999`
- PROBE_A: `leaf=42` → stdout termina em `42`
- PROBE_B: `leaf=7` → stdout termina em `7`

ELF SHA diferente entre mutações é evidência auxiliar, **não substitui** a observação semântica do stdout.

Se ELF SHA mudar mas stdout permanecer `42` em todos os casos, a classificação permanece `BLOCKED` com `class=closure_not_consumed_runtime` — exatamente como no canonical.

---

## Porque esta recertificação importa

O canonical `132620Z.json` pinou o commit `b0a8af6e9`. Desde então, `origin/main` avançou pelo menos 5 commits (incluindo #1176, #1193, #1196, #1197). É possível que algum desses commits tenha acidentalmente corrigido a perda semântica, ou que tenha mudado a classe da falha. Esta recertificação responde uma pergunta objetiva: **o `main` de hoje ainda produz semântica silenciosamente errada?**

A pergunta é independente do trabalho de differential IR hashes que o agente principal fará em paralelo no worktree canonical. Não há colisão: esta tarefa é read-only em relação ao compiler source e ao worktree canonical.

---

## Após a recertificação

Uma segunda tarefa adequada seria um **mapa read-only** dos pontos exatos onde capturar `H_pre`, `H_post` e `H_driver` no código do Madaros — sem editar o merge, porque isso colidiria com a frente compiler-owned mais importante que o agente principal conduzirá.

Esta segunda tarefa só deve ser despachada depois que a recertificação retornar um veredicto estável.

---

## Auto-check antes de declarar pronto

- [ ] Worktree novo foi criado de `origin/main` (não do canonical, não de branch local)
- [ ] Madaros source-fresh construído ou reusado COM registro SHA-256 vs Git SHA
- [ ] Script + 3 fixtures copiados (nada mais)
- [ ] Quatro receipts históricos no canonical intocados (`git -C /tmp/sounio-modgraph-witness-20260719 status` limpo)
- [ ] Duas invocações consecutivas do witness executadas
- [ ] Receipts produzidos no NOVO worktree (sob `artifacts/witnesses/`)
- [ ] Delta factual contra canonical documentado
- [ ] Nenhuma afirmação de "merge é a causa"
- [ ] Nenhuma alteração em `self-hosted/*`, `bin/`, `Makefile`, `scripts/ci/`, `scripts/lib/`, governança
