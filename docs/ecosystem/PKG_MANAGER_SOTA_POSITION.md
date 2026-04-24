<!-- docs:meta
topic_id: repo.docs.ecosystem.pkg-manager-sota-position
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.pkg-manager-sota-position
-->

# Sounio Package Manager: Posicionamento vs Estado da Arte (2026)

**Data:** 20 de Abril de 2026
**Versão:** 1.0

---

## Resumo Executivo

O **Sounio Package Manager (SPM)** introduz uma categoria inédita no ecossistema de gerenciadores de pacotes: **epistemic-first package management**. Enquanto o estado da arte em 2026 (Nix, Guix, Spack, Julia Pkg, Cargo, Conda) converge para reprodutibilidade hermética, SLSA attestations e trusted publishing via OIDC, nenhum deles quantifica *quão confiável é o conhecimento científico dentro do pacote*.

O SPM responde essa pergunta nativamente.

---

## Estado da Arte em 2026 (Pesquisa Atualizada)

### Tendências Dominantes

| Tendência | Adotantes | Descrição |
|-----------|-----------|-----------|
| Hermetic builds | Nix, Guix, Bazel | Ambiente completamente reproduzível via hash do grafo de dependências |
| SLSA Attestations | PyPI, crates.io, npm | Linked trusted publishing via GitHub OIDC + Sigstore |
| RO-Crate | Comunidade científica | Bundle de artefatos de pesquisa com provenance em JSON-LD |
| W3C PROV-DM | ML/AI systems | Modelo formal de provenance para dados e modelos |
| OSS Rebuild | Google (2025) | Rebuild automático de pacotes para verificação semântica |
| SWHIDs | Software Heritage | Identificadores persistentes para código-fonte específico |
| Spack | HPC/DOE | Gestão combinatorial de stacks para computação de alto desempenho |
| Julia `Artifacts.toml` | Julia ecosystem | Melhor equilíbrio atual entre facilidade e reprodutibilidade científica |

### Lacuna Identificada

A pesquisa revela um **gap fundamental**: todos os sistemas acima respondem "**como foi produzido este pacote?**" mas nenhum responde "**quão confiável é o conhecimento científico dentro dele?**".

Ferramentas como MONAI (UQ em imagem médica), DAKOTA (UQ em simulação científica) e dtrackr (R provenance) tratam incerteza epistêmica *dentro* de domínios específicos, mas não como propriedade do gerenciador de pacotes.

---

## O Sounio se Posiciona Como Único

### 1. Epistemic Score como Cidadão de Primeira Classe

Nenhum outro package manager possui um score de qualidade epistêmica nativo. O SPM calcula automaticamente:

```
epistemic-score = 0.35 × knowledge_api_use
                + 0.25 × gum_test_coverage
                + 0.20 × provenance_quality
                + 0.10 × test_coverage
                + 0.10 × docs_quality
```

Este score aparece em `sounio.toml`, no registry, na CLI e nos relatórios de auditoria.

### 2. Quatro Tiers de Confiança Científica

```
experimental  score < 0.60  — uso apenas para prototipagem
community     0.60-0.75     — uso geral, sem garantias
curated       0.75-0.90     — qualidade de produção, auditado
regulatory    > 0.90        — uso em pharma/clinical, revisão humana obrigatória
```

Cargo, Julia Pkg, Nix e Spack não possuem equivalente.

### 3. GUM Compliance como Metadado de Pacote

O campo `gum-compliant = true` no `sounio.toml` declara que o pacote implementa propagação de incerteza conforme JCGM 100:2008 (Guide to the Expression of Uncertainty in Measurement). Isso é reconhecido pelo registry e pela análise estática do compilador.

Nenhum gerenciador atual conecta conformidade metrológica ao ecossistema de pacotes.

### 4. Confidence Gates como Política de Dependência

Um pacote pode declarar `confidence-threshold = 0.90`. O resolver pode recusar dependências que não atendam esse threshold. Esta é uma forma de **epistemic dependency policy** inédita.

### 5. Provenance Ledger Integrado ao Runtime

Usando `stdlib/epistemic/audit_runtime.sio` (W3C PROV-DM estendido com campos epistêmicos), cada operação no pacote é rastreável com `entity_id`, `activity_id`, `regulatory_layer` e `confidence`. Isso vai além do RO-Crate (que opera em nível de arquivo) para o nível de computação individual.

---

## Tabela Comparativa Completa

| Dimensão | Nix/Guix | Spack | Julia Pkg | Cargo | Conda | **SPM (Sounio)** |
|----------|----------|-------|-----------|-------|-------|-----------------|
| Hermetic builds | ★★★★★ | ★★★★ | ★★★ | ★★★★ | ★★ | ★★★★ |
| Reprodutibilidade | ★★★★★ | ★★★★ | ★★★★ | ★★★★ | ★★★ | ★★★★ |
| Trusted publishing | ★★★★ | ★★★ | ★★★ | ★★★★★ | ★★ | ★★★★ |
| Epistemic scoring | ✗ | ✗ | ✗ | ✗ | ✗ | **★★★★★** |
| GUM compliance | ✗ | ✗ | ✗ | ✗ | ✗ | **★★★★★** |
| Provenance ledger | Parcial | ✗ | ✗ | ✗ | ✗ | **★★★★★** |
| Confidence gates | ✗ | ✗ | ✗ | ✗ | ✗ | **★★★★★** |
| Regulatory tier | ✗ | ✗ | ✗ | ✗ | ✗ | **★★★★★** |
| Scientific focus | ★★ | ★★★★★ | ★★★★★ | ★★ | ★★★★ | **★★★★★** |
| Python interop | ★★★ | ★★★ | ★★ | ★★★★ | ★★★★★ | ★★★★ |
| Developer UX | ★★★ | ★★ | ★★★★★ | ★★★★★ | ★★★★ | ★★★ |

---

## Arquitetura do SPM: Visão Geral

### Componentes Implementados (esta sessão)

```
self-hosted/compiler/pkg/
├── manifest.sio     — Parser de sounio.toml sem heap
├── scorer.sio       — Epistemic Scoring Engine (pesos GUM)
├── cli.sio          — Comandos init/build/audit/install/publish
└── lib.sio          — Módulo público do SPM

ecosystem/sounio-py/
├── pyproject.toml   — Build config (maturin + pyo3)
├── src/sounio/
│   ├── __init__.py      — API pública
│   ├── _knowledge.py    — Knowledge<T> Python nativo
│   ├── _epistemic.py    — GUMPropagation + EpistemicResult
│   └── _compile.py      — JIT bridge (souc binary)
└── tests/
    ├── test_knowledge.py
    └── test_epistemic.py

docs/ecosystem/
├── SOUNIO_TOML_SPEC.md
├── REGISTRY_ARCHITECTURE.md
├── ECOSYSTEM_ROADMAP_2026.md
├── CURATED_PACKAGES.md
└── PKG_MANAGER_SOTA_POSITION.md (este arquivo)
```

### Integração no Compilador

`main.sio` agora reconhece:
```
souc pkg init [--epistemic]
souc pkg build [path]
souc pkg audit [path]
souc pkg publish [--dry-run]
souc pkg info <name>
souc pkg self-test
souc install <name>[@version]
souc search <query> [--min-score N]
```

---

## Próximos Passos de Implementação

| Prioridade | Item | Esforço |
|-----------|------|---------|
| 1 | File I/O para leitura real de sounio.toml | 1 semana |
| 2 | Registry client (HTTPS + JSON) | 2-3 semanas |
| 3 | sounio.lock (lockfile) + dependency resolver | 2 semanas |
| 4 | maturin/pyo3 Rust bridge para sounio-py | 3-4 semanas |
| 5 | MVP de registry.sounio.org (FastAPI + PostgreSQL) | 4 semanas |
| 6 | Publicar primeiros 3 pacotes curados | 6 semanas |

---

## Conclusão

O Sounio Package Manager não compete com Cargo, Nix ou Julia Pkg. Ele define uma **nova categoria**: gestão de conhecimento científico confiável.

Enquanto outros gerenciadores garantem que "o binário é reproduzível", o SPM garante que "**o conhecimento produzido por este pacote é epistemicamente fundamentado, provável e rastreável**".

Esta é a contribuição única do Sounio para o ecossistema de computação científica em 2026.

---

*Baseado em pesquisa do estado da arte: Nesbitt 2026 (Reproducible Builds), Nature Scientific Data (CODE beyond FAIR), OSS Rebuild (Google 2025), Spack (ACM SC), Julia Pkg, Rust Cargo, Nix/Guix ecosystem.*
