<!-- docs:meta
topic_id: repo.docs.dissertation.visao-geral
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.visao-geral
-->

# Visão Geral da Dissertação de Mestrado

**Título**: *GUM-Native Pharmacokinetic Simulation via Epistemic Gradual Compilation (Rapamycin PBPK)*  
**Instituição**: PUC-SP  
**Prazo Crítico para Defesa**: 22 de Setembro de 2026  
**Público-Alvo**: Orientador (Farmacologia), Membros da Banca e Desenvolvedores do Sounio  

---

## 1. Introdução e Contexto Unificado

Esta dissertação apresenta a unificação entre a computação epistêmica e a modelagem farmacocinética de base fisiológica (PBPK). O Sounio serve como plataforma científica para propagar de forma rigorosa as incertezas de parâmetros cinéticos através de equações diferenciais ordinárias (EDOs) de transporte e eliminação de fármacos.

### O Conceito de Compilação Epistêmica Gradual (Epistemic Gradual Compilation)

O Sounio introduz o conceito de **Compilação Epistêmica Gradual**, adaptando o formalismo técnico de gradualidade (Siek & Taha 2006) para o domínio da incerteza metrológica e da evidência científica:
1. **Fase Estática**: Quando todas as premissas científicas (distribuições de parâmetros, dimensões e asserções ontológicas) são conhecidas em tempo de compilação, o compilador as resolve e descarrega estaticamente via sistema de tipos e SKS (Semantic Knowledge Spine), gerando zero de custo de tempo de execução.
2. **Fase Dinâmica**: Quando o código consome dados do mundo real ou fluxos externos (não determináveis estaticamente), o compilador relaxa gradualmente as exigências de tipos puros e injeta de forma automatizada **guardas de runtime** e asserções de segurança (`assert`).
3. **Gradualidade**: Isto garante um espectro contínuo onde programas com graus parciais de prova matemática ainda compilam com segurança, tendo seus limites de evidência estritamente garantidos por testes e armadilhas (traps) de execução.

---

## 2. As Três Grandes Contribuições Científicas

### Contribuição 1: GUM-through-ODE (Propagação de Incerteza no Resolvedor) — [x] **Implementado**
A incerteza de cada parâmetro físico (como depuração $CL$, volume de distribuição $V_d$ e biodisponibilidade $F$) é propagada passo a passo no resolvedor numérico de EDOs (utilizando métodos como Bogacki-Shampine 3(2) ou Tsitouras 5(4)).
* **Mecanismo**: buffers de sombra de variância (`BSS_VAR`) no compilador associam dinamicamente a variância acumulada às variáveis físicas em mutação contínua no laço de integração.
* **Eficiência e Complexidade**: 
  - Enquanto simulações tradicionais de Monte Carlo requerem de $10^4$ a $10^6$ execuções completas de trajetórias de EDO para aproximar envelopes de incerteza, o Sounio resolve as equações diferenciais de sensibilidade analiticamente em uma única execução do integrador.
  - A complexidade para propagação de primeira ordem (sensibilidade linear) é de $O(N_p \cdot N_s)$, onde $N_p$ é o número de parâmetros incertos e $N_s$ é o número de estados do modelo. Isto representa um ganho de velocidade (speedup) de 4 a 6 ordens de grandeza em relação ao Monte Carlo.
  - Para sistemas altamente não-lineares, a propagação de segunda ordem (Hessiana) possui complexidade $O(N_p^2 \cdot N_s)$, mantendo-se imensamente mais eficiente que simulações estocásticas.

### Contribuição 2: Compile-Time Confidence Gates (Portas de Confiança Estáticas) — [🔄] **Em Integração**
As simulações científicas carregam restrições sobre o espaço de parâmetros em que são clinicamente válidas. No Sounio, essas restrições são codificadas estaticamente através de contratos de efeito algébrico (`with Epistemic` e `with Hypothesis`).
* **Mecanismo**: O compilador avalia se os intervalos ou as distribuições dos parâmetros ultrapassam os limites de evidência clínica estabelecidos na literatura.
* **Resultado**: Se uma simulação for parametrizada para um cenário extrapolado (ex: dosagem fora do perfil de segurança clínica), o Sounio rejeita a compilação do código (`compile-time reject`), impedindo a execução de simulações inválidas antes mesmo que elas rodem.

### Contribuição 3: ISO Uncertainty Budgets (Decomposição de Orçamento de Incerteza) — [x] **Implementado**
Permite ao cientista identificar a contribuição percentual exata de cada fonte de incerteza (parâmetro) sobre o resultado final da concentração de fármaco em cada órgão.
* **Mecanismo**: Uso de derivadas parciais (Jacobianas) de primeira ordem e matrizes Hessianas de segunda ordem para capturar interações complexas e não-lineares de múltiplos parâmetros simultaneamente, de acordo com o formalismo do **GUM Suplemento 2 (JCGM 102:2011)**.
* **Resultado**: O compilador extrai a assinatura de incerteza, isolando a variação Tipo A (estatística/experimental) da variação Tipo B (estimativa baseada em literatura).

---

## 3. Os Dois Modelos PBPK Coexistentes no Repositório

O repositório do Sounio abriga duas arquiteturas distintas de modelos farmacocinéticos baseados na fisiologia humana (ambas compartilhando a mesma infraestrutura matemática da biblioteca padrão):

```
       [ Modelo PBPK14 (Well-Stirred) ]          [ Modelo PBPK28 (Permeability-Limited) ]
    ────────────────────────────────────────    ──────────────────────────────────────────
    • Instantaneous vascular-tissue equilibrium • Transport rate-limiting barrier
    • 14 scalar states (1 per organ)            • 28 coupled states (14 vascular C_v, 
    • Used for general epistemic-comp cases       14 interstitial C_t coupled by PS)
    • Models: Rapamycin, Tacrolimus,            • Used for high-fidelity clinical chapters
      Haloperidol, Vancomycin, Olanzapine.      • Models: Rapamycin (DES), Semaglutide.
```

---

## 4. Portfólio de Drogas e Validações Clínicas

O Sounio valida seu compilador através de implementações de farmacocinética do mundo real:

| Fármaco (Drug Arm) | Modelo de EDO | Arquitetura | Status no Repositório | Gateway de Validação |
|---|---|---|---|---|
| **Rapamycin** | 3-Comp & PBPK14/PBPK28 | Permeability-Limited & TMDD | [x] **Implementado** | `dissertation_pbpk_suite_gate.sh` |
| **Semaglutide** | PBPK28 | SC Depot + TMDD GLP-1R | [x] **Implementado** | `dissertation_pbpk28_parity_ref_semaglutide.sio` |
| **Tirzepatide** | PBPK14 | SC Dual GIP/GLP-1R GUM | [x] **Implementado** | `dissertation_pbpk_suite_gate.sh` |
| **Vancomycin** | PBPK14 | ICU TDM (Monitoramento Clínico) | [x] **Implementado** | `test_vancomycin_pbpk_v2.sio` |
| **Tacrolimus** | PBPK14 | DDI + CYP3A4 Inibição | [x] **Implementado** | `test_aminoglycoside_correlation_sensitivity.sio` |
| **Haloperidol** | PBPK14 | CYP2D6 Pgx + Colisão mToR | [x] **Implementado** | `dissertation_pbpk_suite_gate.sh` |

*Nota Metodológica: As validações clínicas servem de prova de conceito para demonstrar que o motor matemático do Sounio é capaz de replicar com precisão trajetórias descritas em literatura (como os dados de Ferron 1997 para Rapamicina e Overgaard 2019 para Semaglutide), não devendo ser interpretadas como ferramentas validadas para uso diagnóstico ou prescrição individual direta.*

---

## 5. Teoremas e Verificação Formal (Lean 4)

A consistência lógica do Sounio é ancorada por provas matemáticas formais no diretório `formal/lean4/` (que não usam `sorry` e rodam via `native_decide`):

1. **`SounioTacrolimusDosingSafety.lean`**:
   * *Teorema Principal*: Monotonicidade da concentração mínima nas 24h ($C_{24h}$) em relação à biodisponibilidade oral ($F_{oral}$), volume do compartimento central ($V_c$) e depuração ($CL$).
   * *Enquadramento Científico*: Este teorema não visa descobrir um novo fenômeno farmacológico (uma vez que a redução da exposição pelo clearance é um comportamento linear esperado), mas serve como uma **verificação automatizada de consistência interna do compilador**. Provar a monotonicidade garante que o resolvedor numérico e as equações diferenciais de sensibilidade do Sounio preservam a monotonicidade física e estão livres de erros de inversão de sinal nas derivadas (o que causaria falha catastrófica no cálculo do GUM).
   * *Base para Segurança*: Funciona como base para futuras extensões de provas que garantam limites absolutos de segurança clínica (ex: $\forall \text{ dose } \le X \implies C_{max} \le \text{limite de toxicidade}$).
2. **`SounioTacrolimusDDI.lean`**:
   * *Teorema Principal*: Enclausuramento de Fréchet para a interação medicamentosa competitiva (DDI) entre tacrolimus e sirolimus. Prova a robustez e estabilidade matemática das taxas de depuração sob inibição mútua competitiva no fígado.

---

## 6. Cronograma de Desenvolvimento (Rumo à Defesa)

Como a infraestrutura do compilador e os modelos PBPK já se encontram desenvolvidos e validados, o cronograma final foca na consolidação de dados complexos, análises de sensibilidade e escrita da dissertação física:

```
 [Maio 2026] ──────► [Junho 2026] ───────► [Julho 2026] ────────► [Agosto 2026] ───────► [Setembro 2026]
  Integração β⁵        Análise Sobol/        Confidence Gates       Revisão Final &         DEFESA DO
  Interprocedural      Cut-HDMR (§4.10)      státicos +             Capítulos 4-5 do        MESTRADO (PUC-SP)
  no compilador.       e Sensibilidade ODE.  Capítulos 1-3.         Texto Escrito.          Até 22/Set.
```
