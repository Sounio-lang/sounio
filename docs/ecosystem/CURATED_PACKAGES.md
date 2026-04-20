# Pacotes Curados do Ecossistema Sounio (v1.0)

Este documento define a **lista inicial de pacotes oficiais** que devem ser desenvolvidos e mantidos com alto padrão de qualidade epistêmica.

## Critérios de Pacote Curado

Um pacote é considerado "curado" se atender **todos** os critérios:

1. `epistemic-score` ≥ 0.80
2. Cobertura de testes ≥ 85% (incluindo testes de propagação de incerteza)
3. Documentação completa com exemplos executáveis
4. `sounio.toml` com metadados regulatórios claros
5. Manutenção ativa pela equipe core ou parceiro confiável
6. Testes de regressão em CI com `STDLIB_RUNTIME_REGRESSION_STRICT=1`

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
**Epistemic Score alvo:** 0.98

---

### 2. `epistemic-stats`

**Descrição:** Estatística inferencial e descritiva com tratamento epistêmico.

**Funcionalidades:**
- Distribuições (Normal, Beta, Gamma, LogNormal) com priors epistêmicos
- Testes de hipótese com p-values epistêmicos
- Intervalos de confiança com provenance
- ARIMA, correlação e regressão epistêmica
- MCMC diagnostics (R-hat, ESS, Geweke) com uncertainty

**Epistemic Score alvo:** 0.92

---

### 3. `darwin-pbpk`

**Descrição:** Modelo PBPK de 14 compartimentos com propagação epistêmica completa.

**Funcionalidades:**
- `DarwinPBPK14` struct
- Solvers Tsit5 e RK4 epistêmicos
- Simulação multi-dose com uncertainty propagation
- Brain-plasma TAC reference tables
- Integração com dados experimentais (CHB-MIT, ABIDE, etc.)

**Epistemic Score alvo:** 0.95 (devido ao uso regulatório)

---

### 4. `snn-fractal`

**Descrição:** Spiking Neural Networks com arquiteturas fractais e sedenions.

**Funcionalidades:**
- Camadas densas epistêmicas (`dense_layer.sio`)
- Treinamento com BPTT epistêmico
- Análise de landscape de perda com fractal dimension
- Sedenion algebra integration
- Interpretação de incerteza em embeddings

**Epistemic Score alvo:** 0.88

---

### 5. `regulatory-tools`

**Descrição:** Ferramentas para qualificação regulatória (FDA, EMA, ANVISA).

**Funcionalidades:**
- Geração automática de relatórios de incerteza
- Templates de qualificação de modelo (VVUQ)
- Rastreabilidade completa de provenance
- Comparação com padrões GUM, ICH Q2, etc.
- Exportação para formatos regulatórios (PEtab, NONMEM)

**Epistemic Score alvo:** 0.97

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
- Badge de `epistemic-score` no registry
- Testes passando no `stdlib_hyper_execution_gate.sh`

---

**Este documento completa o TODO "curated-packages".**

**Recomendação:** Começar o desenvolvimento por `epistemic-core` seguido de `darwin-pbpk`, pois são os de maior valor científico e regulatório imediato.

**Próximo:** Arquitetura do Registry Público.
