# Sounio — Regras de Operação para Agentes

> Toda sessão de agente (Claude, Codex, Cursor, Kimi, …) começa por aqui.
> Fonte de verdade em runtime: `cd /workspace/sounio && ./sounio-whereami --quick`.
> Claude Code onboarding link, para agentes em outras sessões/máquinas:
> <https://claude.ai/claude-code/onboard/Zi0vHDGtC038>

## 0. Primeiro comando, sempre

```bash
cd /workspace/sounio && ./sounio-whereami --quick
```

Ele reporta: pod, usuário, repo, **branch atual**, nó K8s, compilador selecionado,
GPU owner, estado do Slurm e OrangeFS. Não adivinhe o mapa — leia o whereami.

Antes de interpretar, contestar ou normalizar a direção científica do projeto,
leia também `FOUNDER_INTENT.md`. Ele preserva a intenção entre threads e
modelos; não substitui evidência executável nem as regras operacionais deste
arquivo.

Antes de abrir uma lane que altere o significado de tipos, efeitos, campos de
IR ou claims científicas, leia também
`docs/internal/concepts/SEMANTIC_LANE_CONTRACT.md` e
`docs/internal/concepts/registry.tsv`. O estado vivo pode ser inspecionado com
`bash scripts/dev/sounio_semantic_status.sh`.

## 1. Dois mundos — não confunda

| Mundo | Caminho | whereami |
|---|---|---|
| Workspace / pod K8s | `/workspace/sounio` | `cd /workspace/sounio && ./sounio-whereami --quick` |
| Host / control plane | `/home/devsounio/projects/sounio` | `/home/devsounio/projects/sounio/sounio-whereami --quick` |

- Dentro do pod, **`/home/devsounio` não existe**. Não assuma que existe.
- Caminhos VM-era (`/home/demetrios/RustroverProjects/sounio`) são históricos e mortos.

## 2. Branch é a verdade operacional

- A **branch atualmente em checkout** manda nesta sessão.
- **Não** troque para `main` ou `integration/sounio-dev-ready-base` só porque docs antigos mencionam.
- `BEAGLE_WORKSPACE_BRANCH` ≠ branch git atual. Sempre confira `git branch --show-current`.
- Imprima a branch depois de qualquer checkout, antes de assumir que pegou.

## 3. Superfícies — onde rodar o quê

- `/workspace/sounio` = superfície interativa: **edição + checks leves OK**.
- **Validação pesada / stress** → Sounio Compiler Foundry / Slurm, **nunca** no workspace.
- Output de runs → raízes de artefato da foundry, **não** na árvore viva.
- OrangeFS (`/orangefs/training`) está ausente local; presente só no login Slurm.
- Para pedir validação pesada, use `docs/ops/foundry_slurm_handoff.md`.

## 4. Compilador

- Wrapper canônico: `bin/souc` (resolve `bin/souc-linux-x86_64`).
- `$SOUC check <arquivo>.sio` / `$SOUC compile <src> -o <out>`.

## 5. Cluster

- GPU owner deste pod: reportado pelo whereami (lane GPU fica no habitat-0).
- `sinfo` direto pode dar timeout no workspace → whereami cai pro login pod.
- Submissão de jobs k8s/Slurm: via BeagleCockpit MCP. **Nunca** escreva YAML à mão.

## 6. Hard No

- Não `reset`, `clean`, `rebase` nem troca de branch sem o usuário pedir explicitamente.
- Não rode stress pesado em `/workspace/sounio`.
- Não use `/workspace/sounio` como scratch de batch.
- Não assuma `/home/devsounio` dentro do pod.
- Não `git add -A` em branch compartilhada (vaza WIP de outros agentes).

## 7. Coordenação multi-agente

- Múltiplas sessões Opus/Codex rodam concorrentes neste repo.
- **Coordene antes de editar arquivos compartilhados.** Re-cheque `git status` antes de stage.
- Edits podem aparecer no commit de outro agente sob mensagens não-relacionadas. Não brigue com a história.

## 8. Precedência quando docs divergem

1. Arquivos reais do repo + scripts executáveis
2. Docs de governança commitados
3. Contratos de prompt históricos
4. Suposições (último recurso)

---

Contratos por-CLI: `CLAUDE.md` (Claude Code) · `AGENTS.md` (Codex) · `CLAUDE_HANDOFF.md` (estado de handoff).
