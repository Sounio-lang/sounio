<!-- docs:meta
topic_id: repo.docs.ecosystem.curated-packages
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.curated-packages
-->

# Pacotes Curados do Ecossistema Sounio (v1.0)

Status: roadmap/design list; these are not published public-registry packages.

Este documento define uma lista inicial de pacotes candidatos que devem ser desenvolvidos e mantidos com alto padrão de qualidade epistêmica antes de qualquer lançamento público.

## Critérios de Pacote Curado

Um pacote candidato a curadoria deve atender **todos** os critérios:

1. Ring, contexto de uso, visibilidade e classes de claim explicitamente declarados
2. Cobertura de testes reportada como métrica de cobertura, sem inferência de validação
3. Documentação completa com exemplos executáveis e maturidade declarada
4. Evidências nomeadas e vinculadas por digest aos gates aplicáveis
5. Manutenção e revisão com responsáveis identificados
6. Receipt `package-boundary-receipt` verificável para o release avaliado

---

## Pacotes da Fase 1 (Q2-Q3 2026)

### 1. `epistemic-core` (Fundação)

**Descrição:** Tipos básicos, propagação GUM, `Knowledge<T>`, ledger, confidence gates.

**Componentes chave:**
- `Knowledge<T>` implementation
- Propagação automática (adição, multiplicação, divisão)
- `measure()`, `combine()`, `observe()`
- `Ledger` e provenance tracking
- `confidence_gate` macro

**Dependências:** Nenhuma
**Próximo gate:** `package-boundary-receipt`; qualquer claim GUM requer método e witness próprios.

---

### 2. `epistemic-stats`

**Descrição:** Estatística inferencial e descritiva com tratamento epistêmico.

**Funcionalidades:**
- Distribuições (Normal, Beta, Gamma, LogNormal) com priors epistêmicos
- Testes de hipótese com p-values epistêmicos
- Intervalos de confiança com provenance
- ARIMA, correlação e regressão epistêmica
- MCMC diagnostics (R-hat, ESS, Geweke) com uncertainty

**Próximo gate:** inventário científico e validação por contexto de uso.

---

### 3. `darwin-pbpk`

**Descrição:** Modelo PBPK de 14 compartimentos com propagação epistêmica completa.

**Funcionalidades:**
- `DarwinPBPK14` struct
- Solvers Tsit5 e RK4 epistêmicos
- Simulação multi-dose com uncertainty propagation
- Brain-plasma TAC reference tables
- Integração com dados experimentais (CHB-MIT, ABIDE, etc.)

**Próximo gate:** qualificação PBPK específica à finalidade e versão do modelo.

---

### 4. `snn-fractal`

**Descrição:** Spiking Neural Networks com arquiteturas fractais e sedenions.

**Funcionalidades:**
- Camadas densas epistêmicas (`dense_layer.sio`)
- Treinamento com BPTT epistêmico
- Análise de landscape de perda com fractal dimension
- Sedenion algebra integration
- Interpretação de incerteza em embeddings

**Próximo gate:** evidência experimental reproduzível no contexto declarado.

---

### 5. `regulatory-tools`

**Descrição:** Ferramentas para qualificação regulatória (FDA, EMA, ANVISA).

**Funcionalidades:**
- Geração automática de relatórios de incerteza
- Templates de qualificação de modelo (VVUQ)
- Rastreabilidade completa de provenance
- Comparação com padrões GUM, ICH Q2, etc.
- Exportação para formatos regulatórios (PEtab, NONMEM)

**Próximo gate:** especificação separada; o pacote não concede autoridade regulatória.

---

## Pacotes da Fase 2 (Q4 2026)

6. `causal-epistemic` — Do-calculus com uncertainty
7. `theorem-prover` — Interface amigável para o kernel de provas
8. `medlang` — DSL médica madura com bindings Python
9. `bayesian-optimization` — Optmização epistêmica (Nelder-Mead + Bayesian)
10. `fmri-epistemic` — Análise de conectividade cerebral com uncertainty

---

## Critérios de Aceitação para Publicação

Todo pacote curado deve ter:

- `sounio.toml` completo
- `examples/` com pelo menos 3 exemplos reais
- `tests/` com testes E2E epistêmicos
- Documentação gerada via `souniodoc`
- Receipt de fronteira com verdict, hashes, engine e limitações
- Testes passando no `stdlib_hyper_execution_gate.sh`

---

**Este documento completa o TODO "curated-packages".**

**Recomendação:** Começar o desenvolvimento por `epistemic-core` seguido de `darwin-pbpk`, pois são os de maior valor científico e regulatório imediato.

**Próximo:** Arquitetura do registry público futuro.
