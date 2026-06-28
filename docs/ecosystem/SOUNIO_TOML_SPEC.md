<!-- docs:meta
topic_id: repo.docs.ecosystem.sounio-toml-spec
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.sounio-toml-spec
-->

# Sounio Package Specification (sounio.toml)

**Versão:** 0.1.0 (Draft)
**Data:** 2026-04-20
**Autor:** Análise de Ecossistema Sounio
Status: Draft/local package manifest contract; public registry publishing is not launched.

## 1. Objetivo

Definir um formato de manifesto de pacote simples, legível e rico em metadados epistêmicos para o ecossistema Sounio.

## 2. Estrutura do Arquivo `sounio.toml`

```toml
[package]
name = "epistemic-pbpk"
version = "0.4.2"
edition = "2026"
authors = ["Laboratório de Computação Epistêmica", "Equipe Darwin PBPK"]
license = "Apache-2.0"
description = "Modelo PBPK de 14 compartimentos com propagação epistêmica de incerteza via GUM"
repository = "https://github.com/sounio/pbpk"
documentation = "https://docs.sounio.org/pbpk"
keywords = ["pbpk", "pharmacokinetics", "epistemic", "gum", "regulatory"]
categories = ["science", "pharma", "epistemic"]

# Metadados epistêmicos (obrigatórios para pacotes curados)
[epistemic]
score = 0.94                    # 0.0 a 1.0 — qualidade da modelagem epistêmica
confidence-threshold = 0.90
provenance-level = "strong"     # weak | medium | strong | regulatory
gum-compliant = true
regulatory-ready = true
validation-coverage = 0.87

[dependencies]
epistemic-core = "0.5"
knowledge = { version = "1.2", features = ["beta", "provenance"] }
ode-solver = "0.3"

[lib]
name = "pbpk"
path = "src/lib.sio"
crate-type = ["lib", "cdylib"]   # para bindings Python

[[example]]
name = "brain_plasma_tac"
path = "examples/brain_plasma_tac.sio"

[[test]]
name = "epistemic_pbpk_e2e"
path = "tests/test_pbpk_epistemic.sio"
```

## 3. Campos Obrigatórios

- `package.name`
- `package.version` (seguir SemVer)
- `package.description`
- `epistemic.score`
- `epistemic.provenance-level`

## 4. Níveis de Proveniência

- `weak`: apenas `Knowledge<T>` básico
- `medium`: propagação GUM + provenance básico
- `strong`: confidence gates + ledger completo
- `regulatory`: strong + testes de validação regulatória + paper trail

## 5. CLI Integration

Os comandos abaixo descrevem a direção de design. A superfície suportada hoje é
o wrapper local `tools/sounio-pkg/sounio-pkg` para `new`, `build`, `check` e
`test`, junto com os imports locais gateados em `packages/*`.

```bash
tools/sounio-pkg/sounio-pkg new my-package
tools/sounio-pkg/sounio-pkg build
tools/sounio-pkg/sounio-pkg check
tools/sounio-pkg/sounio-pkg test
```

## 6. Registry Metadata

Um registry público futuro armazenaria:
- Hash do pacote
- Epistemic score (calculado + revisado por humanos)
- Lista de dependências resolvidas
- Artefatos: `.sio-pkg`, `.whl` (para Python), `.tar.gz` (source)

## 7. Próximos Passos

1. Implementar parser de `sounio.toml` em `self-hosted/compiler/pkg/`
2. Criar comando `souc pkg init`
3. Desenvolver `sounio-py` bindings
4. Lançar registry MVP em `registry.sounio.org`

---

**Esta especificação é o fundamento do ecossistema Sounio.**
Ela combina simplicidade (como Cargo.toml) com metadados epistêmicos únicos.

**Status:** Draft — aberto a refinamento pela comunidade; sem registry pública lançada.
