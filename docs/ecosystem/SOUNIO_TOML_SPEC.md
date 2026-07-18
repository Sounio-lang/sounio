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

Definir um formato de manifesto de pacote simples, legível e com declarações
explícitas de fronteira científica para o ecossistema Sounio.

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

# Declaração de fronteira científica
[science]
schema = "sounio.science-manifest.v1"
ring = "scientific-package"
evidence-status = "passes-gate"
context-of-use = "PBPK research software for a declared model version"
visibility = "public"
allowed-claim-classes = ["compile", "runtime", "validated_research"]
evidence-refs = ["gate:pbpk-package-gate", "review:model-version-0.4.2"]
next-gate = "package-boundary-receipt"
review-state = "draft"

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
maturity = "implemented"
context-of-use = "PBPK research software for a declared model version"
evidence-refs = ["source:examples/brain_plasma_tac.sio", "gate:pbpk-package-gate"]

[[test]]
name = "epistemic_pbpk_e2e"
path = "tests/test_pbpk_epistemic.sio"
```

## 3. Campos Obrigatórios

- `package.name`
- `package.version` (seguir SemVer)
- `package.description`
- `science.schema`, quando houver declaração científica
- `science.ring`
- `science.evidence-status`
- `science.context-of-use`
- `science.visibility`
- `science.allowed-claim-classes`
- `science.evidence-refs`

## 4. Rings e evidência

Os rings conclusivos são `pl-core`, `scientific-package` e `research`.
`scientific-package-candidate`, `mixed-unresolved` e `unclassified` são
auditáveis, mas não produzem `OK`.

`evidence-status` descreve o testemunho mais forte realmente alcançado. Não é
um score e não autoriza classes de claim. Claims dependem de um
`sounio.claim-contract.v1` separado e de contexto de uso compatível.

O parser legado de `[epistemic]` permanece apenas para leitura compatível e
emite `W-SRB-LEGACY-001`. `score`, `regulatory-ready`, `provenance-level`,
`gum-compliant` e `validation-coverage` não influenciam rings, claims,
promoção ou release, e não são traduzidos automaticamente.

## 5. CLI Integration

O wrapper local `tools/sounio-pkg/sounio-pkg` continua suportando `new`,
`build`, `check` e `test`, junto com os imports locais gateados em
`packages/*`. R2.5 acrescenta ao launcher público um release local opt-in:

```bash
tools/sounio-pkg/sounio-pkg new my-package
tools/sounio-pkg/sounio-pkg build
tools/sounio-pkg/sounio-pkg check
tools/sounio-pkg/sounio-pkg test
bin/souc pkg build . --science-boundary strict --claim-contract claim.toml
bin/souc pkg verify target/release/<name>-<version>.sio-release --root .
```

`pkg build` em modo `strict` exige um claim contract local ao package root. Ele
cria por padrão `<name>-<version>.sio-release` sob `target/release/`; uma saída
alternativa pode ser escolhida com `--release-bundle`. O diretório final só é
promovido depois de verdict `OK`, closure raw-AST completa e revalidação dos
hashes de fonte, policy, claim, compilador e ELF. Falha, `REJECT`, `UNKNOWN` ou
tamper deixam o bundle final ausente. O formato é
`sounio.package-release-bundle.v1` e permanece `identity-only`.

## 6. Registry Metadata

Um registry público futuro armazenaria:
- Hash do pacote
- Lista de dependências resolvidas
- Declaração de ring e receipt de fronteira, quando aplicável
- Artefatos locais verificáveis `.sio-release`
- Attestations R2.6 `unsigned-local-policy-evaluation` para decisões locais de
  catálogo; publicação, issuer identity e assinatura remota permanecem fora
  deste contrato

## 7. Próximos Passos

1. Completar o inventário de rings do `stdlib`
2. Manter o gate R2.5 de receipts opt-in para package/release
3. Manter o gate R2.6 de registry attestation local sem habilitar publicação
4. Desenvolver `sounio-py` bindings sem ampliar autoridade científica
5. Manter o inventário e o materializador local R3; destinos reais e remoção da origem exigem aprovações separadas

---

**Esta especificação é o contrato local do ecossistema Sounio.**
Ela combina um manifesto simples com fronteiras e claims explicitamente
separados de scores escalares.

**Status:** Draft — aberto a refinamento pela comunidade; sem registry pública lançada.
