# Inventário de Arquivos .disabled na Standard Library

**Data:** 2026-03-17
**Total inicial:** 106 arquivos .disabled
**Após limpeza:** 95 arquivos .disabled
**Análise realizada conforme [`CONVENTIONS.md`](CONVENTIONS.md)**

---

## Resumo Executivo

| Categoria | Quantidade | Ação | Status |
|-----------|------------|------|--------|
| Duplicados (versão ativa existe) | 11 | Deletar .disabled | ✅ EXECUTADO |
| STUB+DISABLED (stub placeholder) | 89 | Documentar - promissor mas pausado | 📝 Documentado |
| Órfãos (sem .sio correspondente) | 6 | Documentar - avaliar futuramente | 📝 Documentado |

---

## 1. Arquivos DUPLICADOS - DELETAR (11)

Estes arquivos .disabled devem ser **DELETADOS** pois existe uma versão ativa (.sio) com conteúdo real (>100 bytes) no mesmo caminho.

| Arquivo .disabled | Versão ativa (.sio) | Tamanho ativo | Decisão |
|-------------------|---------------------|---------------|---------|
| `ffi/callback.sio.disabled` | `ffi/callback.sio` | 1110 bytes | DELETAR |
| `ffi/ctypes.sio.disabled` | `ffi/ctypes.sio` | 608 bytes | DELETAR |
| `ffi/library.sio.disabled` | `ffi/library.sio` | 979 bytes | DELETAR |
| `geometry/types.sio.disabled` | `geometry/types.sio` | 3719 bytes | DELETAR |
| `nn/dense_quaternion.sio.disabled` | `nn/dense_quaternion.sio` | 2915 bytes | DELETAR |
| `nn/g2_equivariant.sio.disabled` | `nn/g2_equivariant.sio` | 1285 bytes | DELETAR |
| `nn/octonion.sio.disabled` | `nn/octonion.sio` | 1249 bytes | DELETAR |
| `nn/optimizers_quaternion.sio.disabled` | `nn/optimizers_quaternion.sio` | 3260 bytes | DELETAR |
| `nn/quaternion.sio.disabled` | `nn/quaternion.sio` | 2788 bytes | DELETAR |
| `onn/g2_activation.sio.disabled` | `onn/g2_activation.sio` | 2188 bytes | DELETAR |
| `prob/distributions.sio.disabled` | `prob/distributions.sio` | 4919 bytes | DELETAR |

**Ação executada:** Ver seção "Ações Realizadas" abaixo.

---

## 2. Arquivos ÓRFÃOS - AVALIAR (6)

Estes arquivos .disabled **não possuem** um arquivo .sio correspondente. São candidatos a:
- Renomear para .sio se prontos (requer teste)
- Criar stub .sio se ainda em desenvolvimento

| Arquivo .disabled | Status | Linhas | Descrição | Decisão |
|-------------------|--------|--------|-----------|---------|
| `test/helpers.sio.disabled` | Experimental | 185 | Helpers para testes numéricos (check_near, check_approx) | **DOCUMENTAR** - útil, mas requer validação |
| `heliobiology/effects.sio.disabled` | Experimental | 410 | Modelos de efeitos biológicos de clima espacial | **DOCUMENTAR** - domínio específico, sem testes |
| `heliobiology/solar.sio.disabled` | Experimental | - | Dados solares | **DOCUMENTAR** - dependência circular com indices |
| `heliobiology/indices.sio.disabled` | Experimental | - | Índices de clima espacial (Kp, Dst, Ap) | **DOCUMENTAR** - requer integração |
| `genomics/io/fasta.sio.disabled` | Experimental | 174 | Parser FASTA para genômica | **DOCUMENTAR** - promissor, sem testes |
| `genomics/gpu/gf4.sio.disabled` | Experimental | - | Operações GF(4) em GPU | **DOCUMENTAR** - requer backend GPU |

---

## 3. Arquivos STUB+DISABLED - PAUSADOS (92)

Estes arquivos têm um **stub** .sio (<100 bytes, apenas comentário "disabled") e uma versão .disabled com implementação. Representam módulos promissores mas pausados.

### 3.1 Encoding (2 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `encoding/base64.sio.disabled` | Experimental | DOCUMENTAR - RFC 4648 completo, aguarda FFI estável |
| `encoding/hex.sio.disabled` | Experimental | DOCUMENTAR - encoding hex, aguarda FFI estável |

### 3.2 Crypto (3 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `crypto/hash.sio.disabled` | Experimental | DOCUMENTAR - SHA-256/512, MD5, SHA-1; aguarda auditoria |
| `crypto/hmac.sio.disabled` | Experimental | DOCUMENTAR - HMAC, aguarda hash.sio ativo |
| `crypto/random.sio.disabled` | Experimental | DOCUMENTAR - CSPRNG, aguarda integração com OS |

### 3.3 Compress (2 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `compress/gzip.sio.disabled` | Experimental | DOCUMENTAR - gzip via zlib FFI |
| `compress/zstd.sio.disabled` | Experimental | DOCUMENTAR - zstd compression |

### 3.4 Geometry (3 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `geometry/engine.sio.disabled` | Experimental | DOCUMENTAR - motor de dedução geométrica (AlphaGeometry) |
| `geometry/predicates.sio.disabled` | Experimental | DOCUMENTAR - predicados geométricos |
| `geometry/symbolic_engine.sio.disabled` | Experimental | DOCUMENTAR - motor simbólico |

### 3.5 Heliobiology (4 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `heliobiology/effects.sio.disabled` | Experimental | DOCUMENTAR - ver seção Órfãos |
| `heliobiology/indices.sio.disabled` | Experimental | DOCUMENTAR - ver seção Órfãos |
| `heliobiology/solar.sio.disabled` | Experimental | DOCUMENTAR - ver seção Órfãos |
| `heliobiology/units.sio.disabled` | Experimental | DOCUMENTAR - unidades específicas |

### 3.6 MedLang (14 arquivos)

Sistema DSL para modelagem farmacocinética/populacional.

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `medlang/ast.sio.disabled` | Experimental | DOCUMENTAR - AST para MedLang DSL |
| `medlang/codegen.sio.disabled` | Experimental | DOCUMENTAR - geração de código |
| `medlang/lexer.sio.disabled` | Experimental | DOCUMENTAR - lexer MedLang |
| `medlang/parser.sio.disabled` | Experimental | DOCUMENTAR - parser MedLang |
| `medlang/integrate.sio.disabled` | Experimental | DOCUMENTAR - integração com ODE |
| `medlang/dose/mod.sio.disabled` | Experimental | DOCUMENTAR - módulo de dosagem |
| `medlang/pk/mod.sio.disabled` | Experimental | DOCUMENTAR - farmacocinética |
| `medlang/pk/one_compartment.sio.disabled` | Experimental | DOCUMENTAR - modelo 1 compartimento |
| `medlang/pk/two_compartment.sio.disabled` | Experimental | DOCUMENTAR - modelo 2 compartimentos |
| `medlang/pk/three_compartment.sio.disabled` | Experimental | DOCUMENTAR - modelo 3 compartimentos |
| `medlang/pk/multi_compartment.sio.disabled` | Experimental | DOCUMENTAR - modelo multi-compartment |
| `medlang/policy/mod.sio.disabled` | Experimental | DOCUMENTAR - políticas |
| `medlang/population/mod.sio.disabled` | Experimental | DOCUMENTAR - análise populacional |
| `medlang/population/model.sio.disabled` | Experimental | DOCUMENTAR - modelo populacional |
| `medlang/population/estimation.sio.disabled` | Experimental | DOCUMENTAR - estimação de parâmetros |
| `medlang/population/simulation.sio.disabled` | Experimental | DOCUMENTAR - simulação |
| `medlang/population/variability.sio.disabled` | Experimental | DOCUMENTAR - variabilidade inter-individual |

### 3.7 ODE/PBPK (10 arquivos)

Várias versões experimentais de integradores ODE para PBPK.

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `ode/pbpk14.sio.disabled` | Obsoleto | DOCUMENTAR - versão superseded |
| `ode/pbpk14_rk4.sio.disabled` | Obsoleto | DOCUMENTAR - versão superseded |
| `ode/pbpk3_stable.sio.disabled` | Experimental | DOCUMENTAR - versão estável alternativa |
| `ode/pbpk_debug.sio.disabled` | Debug | DOCUMENTAR - versão de debug |
| `ode/pbpk_fast.sio.disabled` | Experimental | DOCUMENTAR - versão otimizada |
| `ode/pbpk_minimal.sio.disabled` | Experimental | DOCUMENTAR - versão minimal |
| `ode/pbpk_tiny.sio.disabled` | Experimental | DOCUMENTAR - versão compacta |
| `ode/pbpk_unrolled.sio.disabled` | Experimental | DOCUMENTAR - versão unrolled |
| `ode/pbpk_working.sio.disabled` | Experimental | DOCUMENTAR - versão funcional de referência |
| `ode/tsit5_multicomp.sio.disabled` | Experimental | DOCUMENTAR - TSIT5 multi-compartment |

### 3.8 ONN (Octonion Neural Network) (8 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `onn/activation.sio.disabled` | Experimental | DOCUMENTAR - funções de ativação octoniónicas |
| `onn/attention.sio.disabled` | Experimental | DOCUMENTAR - mecanismo de atenção |
| `onn/conv.sio.disabled` | Experimental | DOCUMENTAR - convolução |
| `onn/linear.sio.disabled` | Experimental | DOCUMENTAR - camada linear |
| `onn/loss.sio.disabled` | Experimental | DOCUMENTAR - funções de perda |
| `onn/normalization.sio.disabled` | Experimental | DOCUMENTAR - normalização |
| `onn/optimizer.sio.disabled` | Experimental | DOCUMENTAR - otimizadores |
| `onn/training.sio.disabled` | Experimental | DOCUMENTAR - loop de treinamento |

### 3.9 NN (Neural Networks) (5 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `nn/beam_search.sio.disabled` | Experimental | DOCUMENTAR - beam search |
| `nn/mlp_classifier.sio.disabled` | Experimental | DOCUMENTAR - classificador MLP |
| `nn/mlp_xor.sio.disabled` | Experimental | DOCUMENTAR - XOR demo |

### 3.10 Ontology (10 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `ontology/biomedical/go.sio.disabled` | Experimental | DOCUMENTAR - Gene Ontology |
| `ontology/biomedical/hpo.sio.disabled` | Experimental | DOCUMENTAR - Human Phenotype Ontology |
| `ontology/biomedical/loinc.sio.disabled` | Experimental | DOCUMENTAR - LOINC codes |
| `ontology/biomedical/mod.sio.disabled` | Experimental | DOCUMENTAR - módulo biomédico |
| `ontology/biomedical/snomed.sio.disabled` | Experimental | DOCUMENTAR - SNOMED CT |
| `ontology/cache.sio.disabled` | Experimental | DOCUMENTAR - cache de ontologias |
| `ontology/model.sio.disabled` | Experimental | DOCUMENTAR - modelo de ontologia |
| `ontology/namespaces.sio.disabled` | Experimental | DOCUMENTAR - namespaces |
| `ontology/query.sio.disabled` | Experimental | DOCUMENTAR - queries |
| `ontology/reasoner.sio.disabled` | Experimental | DOCUMENTAR - reasoner |

### 3.11 Prob (6 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `prob/beta.sio.disabled` | Experimental | DOCUMENTAR - distribuição Beta |
| `prob/inference.sio.disabled` | Experimental | DOCUMENTAR - inferência estatística |
| `prob/mcmc.sio.disabled` | Experimental | DOCUMENTAR - MCMC |
| `prob/normal.sio.disabled` | Experimental | DOCUMENTAR - distribuição Normal |
| `prob/random.sio.disabled` | Experimental | DOCUMENTAR - geração de aleatórios |

### 3.12 Test (3 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `test/assert_advanced.sio.disabled` | Experimental | DOCUMENTAR - asserções avançadas |
| `test/helpers.sio.disabled` | Experimental | DOCUMENTAR - ver seção Órfãos |
| `test/mock.sio.disabled` | Experimental | DOCUMENTAR - mocking |

### 3.13 Text (3 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `text/case.sio.disabled` | Experimental | DOCUMENTAR - conversão de caso |
| `text/unicode.sio.disabled` | Experimental | DOCUMENTAR - operações Unicode |
| `text/wrap.sio.disabled` | Experimental | DOCUMENTAR - quebra de texto |

### 3.14 Time (3 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `time/datetime.sio.disabled` | Experimental | DOCUMENTAR - data/hora |
| `time/duration.sio.disabled` | Experimental | DOCUMENTAR - duração |
| `time/instant.sio.disabled` | Experimental | DOCUMENTAR - instante temporal |

### 3.15 GPU (3 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `gpu/fft.sio.disabled` | Experimental | DOCUMENTAR - FFT em GPU |
| `gpu/smooth.sio.disabled` | Experimental | DOCUMENTAR - smoothing em GPU |
| `gpu/stats.sio.disabled` | Experimental | DOCUMENTAR - estatísticas em GPU |

### 3.16 Outros (16 arquivos)

| Arquivo | Status | Decisão |
|---------|--------|---------|
| `bayes/prior.sio.disabled` | Experimental | DOCUMENTAR - priors Bayesianos |
| `fractal/curvature.sio.disabled` | Experimental | DOCUMENTAR - curvatura fractal |
| `fractal/kec.sio.disabled` | Experimental | DOCUMENTAR - KEC |
| `fractal/multifractal.sio.disabled` | Experimental | DOCUMENTAR - análise multifractal |
| `genomics/io/fasta.sio.disabled` | Experimental | DOCUMENTAR - ver seção Órfãos |
| `genomics/gpu/gf4.sio.disabled` | Experimental | DOCUMENTAR - ver seção Órfãos |
| `medical/sedenion_pbpk.sio.disabled` | Experimental | DOCUMENTAR - PBPK com sedenions |
| `optimize/uncertainty.sio.disabled` | Experimental | DOCUMENTAR - otimização com incerteza |
| `os/env.sio.disabled` | Experimental | DOCUMENTAR - variáveis de ambiente |
| `os/process_new.sio.disabled` | Experimental | DOCUMENTAR - gerenciamento de processos |
| `pbpk/regulatory.sio.disabled` | Experimental | DOCUMENTAR - modelos regulatórios |
| `profile/async_profile.sio.disabled` | Experimental | DOCUMENTAR - profiling async |
| `qnn/optimizer_advanced.sio.disabled` | Experimental | DOCUMENTAR - otimizador quântico avançado |
| `qnn/training.sio.disabled` | Experimental | DOCUMENTAR - treinamento QNN |
| `quantum/vqe.sio.disabled` | Experimental | DOCUMENTAR - VQE |
| `signal/fractal.sio.disabled` | Experimental | DOCUMENTAR - análise fractal de sinais |
| `types/refinement.sio.disabled` | Experimental | DOCUMENTAR - refinement types |

---

## Ações Realizadas

### ✅ Deletados (11 arquivos .disabled duplicados)

**Executado em:** 2026-03-17

```
removed 'stdlib/ffi/callback.sio.disabled'
removed 'stdlib/ffi/ctypes.sio.disabled'
removed 'stdlib/ffi/library.sio.disabled'
removed 'stdlib/geometry/types.sio.disabled'
removed 'stdlib/nn/dense_quaternion.sio.disabled'
removed 'stdlib/nn/g2_equivariant.sio.disabled'
removed 'stdlib/nn/octonion.sio.disabled'
removed 'stdlib/nn/optimizers_quaternion.sio.disabled'
removed 'stdlib/nn/quaternion.sio.disabled'
removed 'stdlib/onn/g2_activation.sio.disabled'
removed 'stdlib/prob/distributions.sio.disabled'
```

### Renomeados (0 arquivos)

Nenhum arquivo foi renomeado de .disabled para .sio pois:
1. Os candidatos órfãos requerem criação de testes primeiro
2. Os STUB+DISABLED requerem validação de funcionalidade

---

## Recomendações Futuras

1. **Prioridade Alta - Criar testes para:**
   - `test/helpers.sio.disabled` - útil para todo o ecossistema
   - `encoding/base64.sio.disabled` - dependência comum
   - `encoding/hex.sio.disabled` - dependência comum

2. **Prioridade Média - Validar implementação:**
   - `crypto/hash.sio.disabled` - requer auditoria de segurança
   - `bayes/prior.sio.disabled` - usado por epistemic

3. **Prioridade Baixa - Domínios específicos:**
   - `heliobiology/*` - nicho de pesquisa
   - `medlang/*` - DSL complexa, requer design review
   - `ontology/*` - requer infraestrutura de dados

---

## Conformidade com CONVENTIONS.md

Este inventário segue as diretrizes de [`CONVENTIONS.md`](CONVENTIONS.md):
- Documentação completa de cada módulo
- Status claro (experimental/bugado/obsoleto)
- Decisões justificadas
- Ações rastreáveis
