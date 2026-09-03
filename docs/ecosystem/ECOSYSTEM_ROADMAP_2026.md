<!-- docs:meta
topic_id: repo.docs.ecosystem.ecosystem-roadmap-2026
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.ecosystem-roadmap-2026
-->

# Roadmap do Ecossistema Sounio (2026-2027)
**Foco:** Package Manager, Registry Público e Python Interoperability

**Data:** 20 de Abril de 2026
**Versão:** 1.0

## Visão Estratégica

Transformar o Sounio em uma plataforma de computação científica na qual
fronteiras de software, contexto de uso e evidência permaneçam explícitos, sem
converter maturidade de implementação em autoridade científica ou regulatória.

**Métrica de sucesso em 12 meses:**
- 25 pacotes no registry público
- `pip install sounio` com > 500 downloads/mês
- Pelo menos 3 publicações científicas usando Sounio + Python
- receipts de fronteira verificáveis para todos os releases claim-bearing

---

## Roadmap por Trimestre

### **T1 2026 (Abr-Jun): Fundação (3 meses)**

**Prioridade Alta**

1. **sounio.toml Specification** — *Concluído*
   - Especificação formal + parser inicial

2. **CLI `souc pkg` (MVP)**
   - `souc pkg init`
   - `souc pkg build`
   - `souc pkg validate`
   - Estimativa: 4 semanas

3. **sounio-py v0.1 (Core Bindings)**
   - Classe `Knowledge`
   - Operações aritméticas com propagação GUM
   - Integração básica com numpy
   - Estimativa: 5 semanas

4. **Registry Local Avançado**
   - Cache, lockfiles, resolução de dependências
   - Estimativa: 3 semanas

**Milestone T1:** `souc install epistemic-core` funciona localmente + binding Python básico.

---

### **T2 2026 (Jul-Set): Registry Público + Python Maturidade**

**Prioridade Crítica**

1. **Registry Público (registry.sounio.org)**
   - Backend (Rust ou Sounio self-hosted)
   - API de publish/search
   - Web UI simples
   - Indexação de rings, contexts of use e receipts verificáveis
   - Estimativa: 6-7 semanas

2. **sounio-py v0.2**
   - JIT compilation (`sounio.compile()`)
   - PBPK wrapper completo
   - Integração com pandas/xarray
   - Estimativa: 6 semanas

3. **Pacotes Curados Iniciais (5 pacotes)**
   - `epistemic-core`
   - `epistemic-stats`
   - `darwin-pbpk`
   - `snn-fractal`
   - `regulatory-tools`
   - Estimativa: 4 semanas

**Milestone T2:** Primeiro paper usando Sounio via Python publicado.

---

### **T3 2026 (Out-Dez): Ecossistema e Qualidade**

1. **Ferramentas de Qualidade**
   - `souc pkg audit` com métricas nomeadas e não autoritativas
   - Cobertura de testes identificada como cobertura, não validação
   - Verificação de provenance separada de assurance

2. **sounio-py v0.3**
   - Epistemic Neural Networks
   - Suporte a JAX (custom vjp)
   - Ferramentas de visualização de uncertainty

3. **Documentação e Comunidade**
   - Site docs.sounio.org com exemplos interativos
   - Template de projeto (`souc new epistemic-model`)
   - Discord/Forum ativo

**Milestone T3:** 15 pacotes no registry, `sounio` trending em repositórios científicos.

---

### **T4 2026 / T1 2027: Evidência e Qualificação Específica**

- Pesquisa de ferramentas para contextos regulatórios específicos
- Integração com PyMC, Stan e NONMEM
- Gates de qualificação definidos por finalidade antes de qualquer claim clínica
- Suporte a computação em nuvem epistêmica

---

## Estimativa de Esforço (Homem-mês)

| Área                        | Esforço | Prioridade | Dependências              |
|----------------------------|--------|----------|--------------------------|
| sounio.toml + CLI pkg      | 2.0    | Alta     | -                        |
| Registry Público           | 3.5    | Crítica  | CLI pkg                  |
| sounio-py Core             | 3.0    | Crítica  | Knowledge runtime        |
| Pacotes Curados (5)        | 4.0    | Alta     | Registry + Python bindings |
| Documentação + Exemplos    | 2.5    | Média    | Pacotes curados          |
| Ferramentas de Qualidade   | 2.0    | Alta     | Registry                 |
| **Total**                  | **17.0** | -      | -                        |

**Equipe mínima recomendada:** 3-4 pessoas (1 compiler, 1 Python bindings, 1 scientific content, 1 DevRel)

---

## Riscos e Mitigações

**Risco Alto:** Complexidade do binding Python com `Knowledge<T>`
**Mitigação:** Começar com FFI limpo + pyo3 antes de JIT avançado.

**Risco Alto:** Baixa adoção inicial
**Mitigação:** Focar em nichos de alto valor (PBPK regulatório, epistemic ML em saúde) com cases de sucesso reais.

**Risco Médio:** Manutenção do registry
**Mitigação:** Começar com modelo simples (S3 + SQLite) antes de escalar.

---

**Próximos Passos Imediatos (Próximas 2 Semanas):**

1. Fechar o inventário ring-by-ring do `stdlib`
2. Manter a integração R2.5 de `package-boundary-receipt` nos releases opt-in
3. Criar `sounio-py` com binding mínimo de `Knowledge`
4. Manter o gate R2.6 de registry attestation local com publicação desabilitada
5. Manter inventário, materialização, autorização, execução local, aprovação
   Git/rehearsal, execução Git R3, assessment de gaps de produção e o
   processador não-autorizante da futura seleção de mapeamento; a execução
   canônica está provada apenas em fixtures, a issue #1122 ainda não recebeu a
   escolha humana dos cinco targets, e a origem oficial exige política
   `canonical-production`, destinos reais e decisão humana explícita antes de
   qualquer remoção

---

**Este documento completa o item A4 do plano.**

**Status atual do plano geral:** 4/6 todos concluídos.
