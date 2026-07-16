<!-- docs:meta
topic_id: repo.docs.ecosystem.pkg-manager-sota-position
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.pkg-manager-sota-position
-->

# Sounio Package Manager: Fronteira Cientifica R0-R2

**Data:** 16 de Julho de 2026
**Versao:** 2.0

## Posicionamento

O Sounio Package Manager separa tres perguntas que antes estavam misturadas:

```text
qual codigo foi construido?
qual fronteira de software foi aplicada?
qual claim cientifica foi explicitamente solicitada e evidenciada?
```

R0-R2 responde de forma executavel apenas as duas primeiras e valida a forma
do contrato da terceira. Nao atribui uma probabilidade de verdade ao pacote.
Scores escalares permanecem legiveis por compatibilidade, mas nao controlam
resolucao, promocao, publicacao, claims ou releases.

## Relacao Com O Estado Da Arte

Gerenciadores como Cargo, Nix, Guix, Spack, Conda e Julia Pkg tratam diferentes
aspectos de resolucao, reproducibilidade e distribuicao. SLSA, W3C PROV,
RO-Crate e Software Heritage tratam identidade, provenance ou preservacao sob
contratos proprios. A fronteira Sounio nao substitui esses sistemas e nao
infere assurance cientifica a partir de provenance.

O diferencial implementado e mais estreito: um build pode declarar rings,
contexto de uso, visibilidade e classes de claim permitidas; o compilador usa
sua closure AST para aceitar, rejeitar ou registrar autoridade incompleta. O
resultado e um receipt deterministico `identity-only` com limitacoes explicitas.

## Superficies Implementadas

```text
sounio.toml [science]                  declaracao local
science-rings.tsv                     inventario de repositorio
sounio.claim-contract.v1              claim explicita e evidencia tipada
sounio.package-boundary-receipt.v1    identidade do grafo, policy e artefato
```

Os rings conclusivos sao `pl-core`, `scientific-package` e `research`. Rings
candidatos ou nao classificados produzem `UNKNOWN`; strict mode recusa antes
do lowering. Claims empiricas e clinicas exigem evidencia propria e nunca sao
autorizadas apenas por compilacao, execucao, nome de diretorio ou metadados.

## Superficies Legadas

O parser de `[epistemic]` continua somente para leitura compativel e emite
`W-SRB-LEGACY-001`. O relatorio escalar interno e um diagnostico historico sem
autoridade. Declaracoes booleanas de conformidade GUM nao substituem metodo e
witness nomeados. Provenance e assurance de receipt permanecem categorias
separadas. Thresholds usados por programas em runtime pertencem a configuracao
operacional desses programas, nao a qualidade do pacote.

## Registry E Publicacao

Nao existe registry publico autorizado por este marco. O servidor local de
desenvolvimento e um catalogo read-only; `POST /api/v1/packages` e recusado.
Trusted publishing, assinatura remota, attested execution, independent replay,
`ClinicalAuthority` e `ClinicalRelease` exigem projetos e gates posteriores.

## Limites Da Claim

Um receipt `OK` significa que a closure declarada respeitou a matriz de rings,
visibilidade e contratos explicitamente fornecidos, com identidades de arquivo
verificadas. Ele nao significa que dados sao genuinos, que um modelo e correto,
que uma revisao e independente ou que um resultado e clinicamente valido.

O posicionamento publico suportado por R0-R2 e, no maximo,
`validated_research` vinculado a um gate nominal e a um contexto de uso. Essa
classe nao equivale a validacao clinica ou regulatoria.

## Proximos Gates

1. Fechar o inventario ring-by-ring do `stdlib`.
2. Ativar strict de forma opt-in em fixtures e releases claim-bearing.
3. Especificar registry attestation e assinatura separadamente.
4. Exigir replay independente antes de qualquer ampliacao de assurance.
