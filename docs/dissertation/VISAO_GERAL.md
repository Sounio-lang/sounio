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

Diferente de abordagens tradicionais que dependem puramente de simulações de Monte Carlo (lentas e computacionalmente exaustivas), o Sounio implementa o formalismo analítico do **Guia ISO/JCGM 100:2008 (GUM)** diretamente no compilador e runtime, alcançando precisão matemática com custo de computação quase nulo (zero-cost abstraction).

---

## 2. As Três Grandes Contribuições Científicas

### Contribuição 1: GUM-through-ODE (Propagação de Incerteza no Resolvedor)
A incerteza de cada parâmetro físico (como depuração $CL$, volume de distribuição $V_d$ e biodisponibilidade $F$) é propagada passo a passo no resolvedor numérico de EDOs (utilizando métodos como Bogacki-Shampine 3(2) ou Tsitouras 5(4)).
* **Mecanismo**: buffers de sombra de variância (`BSS_VAR`) no compilador associam dinamicamente a variância acumulada às variáveis físicas em mutação contínua no laço de integração.
* **Resultado**: O resolvedor adaptativo ajusta o tamanho do passo ($dt$) não apenas pelo erro de discretização numérica, mas também pela taxa de crescimento de incerteza metrológica (controle *lookbehind* de variância).

### Contribuição 2: Compile-Time Confidence Gates (Portas de Confiança Estáticas)
As simulações científicas carregam restrições sobre o espaço de parâmetros em que são clinicamente válidas. No Sounio, essas restrições são codificadas estaticamente através de contratos de efeito algébrico (`with Epistemic` e `with Hypothesis`).
* **Mecanismo**: O compilador avalia se os intervalos ou as distribuições dos parâmetros ultrapassam os limites de evidência clínica estabelecidos na literatura.
* **Resultado**: Se uma simulação for parametrizada para um cenário extrapolado (ex: dosagem fora do perfil de segurança clínica), o Sounio rejeita a compilação do código (`compile-time reject`), impedindo a execução de simulações inválidas antes mesmo que elas rodem.

### Contribuição 3: ISO Uncertainty Budgets (Decomposição de Orçamento de Incerteza)
Permite ao cientista identificar a contribuição percentual exata de cada fonte de incerteza (parâmetro) sobre o resultado final da concentração de fármaco em cada órgão.
* **Mecanismo**: Uso de derivadas parciais e matrizes Hessianas de segunda ordem para capturar interações complexas e não-lineares de múltiplos parâmetros simultaneamente.
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
| **Rapamycin** | 3-Comp & PBPK14/PBPK28 | Permeability-Limited & TMDD | Totalmente Implementado | `dissertation_pbpk_suite_gate.sh` |
| **Semaglutide** | PBPK28 | SC Depot + TMDD GLP-1R | Implementado (Via de Paridade) | `dissertation_pbpk28_parity_ref_semaglutide.sio` |
| **Tirzepatide** | PBPK14 | SC Dual GIP/GLP-1R GUM | Validado com propagação | `dissertation_pbpk_suite_gate.sh` |
| **Vancomycin** | PBPK14 | ICU TDM (Monitoramento Clínico) | Validado e auditado | `test_vancomycin_pbpk_v2.sio` |
| **Tacrolimus** | PBPK14 | DDI + CYP3A4 Inibição | Validado com provas formais | `test_aminoglycoside_correlation_sensitivity.sio` |
| **Haloperidol** | PBPK14 | CYP2D6 Pgx + Colisão mToR | Polifarmácia e Clones Pop. | `dissertation_pbpk_suite_gate.sh` |

---

## 5. Teoremas e Verificação Formal (Lean 4)

A consistência de segurança de dosagem do Sounio é ancorada por provas matemáticas formais no diretório `formal/lean4/` (que não usam `sorry` e rodam via `native_decide`):

1. **`SounioTacrolimusDosingSafety.lean`**:
   * *Teorema Principal*: Monotonicidade da concentração mínima nas 24h ($C_{24h}$) em relação à biodisponibilidade oral ($F_{oral}$), volume do compartimento central ($V_c$) e depuração ($CL$).
   * *Significado*: Prova matematicamente que o aumento de $CL$ reduz monotonamente a exposição ao fármaco, garantindo que o compilador não tenha inversões de sinal nas equações de sensibilidade.
2. **`SounioTacrolimusDDI.lean`**:
   * *Teorema Principal*: Enclausuramento de Fréchet para a interação medicamentosa competitiva entre tacrolimus e sirolimus. Prova a estabilidade do clearance sob inibição competitiva.

---

## 6. Cronograma de Desenvolvimento (Rumo à Defesa)

```
 [Maio 2026] ──────► [Junho 2026] ───────► [Julho 2026] ────────► [Agosto 2026] ───────► [Setembro 2026]
  Variância ß⁵         Propagação GUM        Confidence Gates       Revisão Final &         DEFESA DO
  Interprocedural      completa + Sens-      státicos +             Capítulos 4-5 do        MESTRADO (PUC-SP)
  no 3-Comp Stub.      itividade de ODE.     Capítulos 1-3.         Texto Escrito.          Até 22/Set.
```
