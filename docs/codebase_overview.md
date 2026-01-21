# PCS Meta-Repo — Visão Arquitetural Atualizada

Este guia descreve os componentes ativos da plataforma Darwin RAG++ (v5) e mostra onde localizar cada responsabilidade no repositório.

## 1. Camada de Aplicação Darwin
- **`main.py`** instancia a aplicação FastAPI, carrega as configurações validadas, anexa middlewares, expõe todos os routers (health, RAG, memória, tree-search, contratos, contexto, RAG++ e testes) e publica endpoints de status que integram cache e backends de RAG.【F:main.py†L1-L200】
- **Configuração centralizada** em `services/settings.py` define parâmetros de GCP, RAG tradicional, RAG++, cache de contexto, limites de requisição e autenticação via Pydantic Settings, garantindo que variáveis de ambiente controlem comportamentos críticos.【F:services/settings.py†L1-L120】
- **Segurança** é aplicada em `api/security.py`, que implementa autenticação por API key, limitador de taxa com token bucket e cabeçalhos padrão utilizados pelos routers HTTP.【F:api/security.py†L1-L200】

## 2. Superfície HTTP (Routers)
Cada router aplica `require_api_key` + `rate_limit` e fala com serviços especializados:
- **/health** verifica backends, sistema e readiness do RAG Vertex.【F:api/routers/health.py†L1-L120】
- **/rag** expõe respostas com citações e recuperação pura sobre os motores Vertex (Engine e Vector).【F:api/routers/rag.py†L1-L134】
- **/rag-plus** entrega o agente Darwin RAG++ completo: consultas simples, iteração ReAct, discovery científico contínuo e controles de monitoramento.【F:api/routers/rag_plus.py†L1-L180】
- **/memory** registra sessões em JSONL + SQLite e permite pesquisa textual sobre o histórico de interações.【F:api/routers/memory.py†L1-L200】
- **/context-cache** analisa prompts, sugere prefixos estáveis, mantém estatísticas e permite limpeza/configuração do cache de contexto Gemini.【F:api/routers/context_cache.py†L1-L200】
- **/tree-search** executa buscas PUCT, retorna a árvore explorada e oferece modo rápido para prototipação.【F:api/routers/tree_search.py†L1-L200】
- **/contracts** expõe a sandbox de score contracts (delta_kec, zuco_reading, editorial) com execução unitária ou em lote.【F:api/routers/score_contracts.py†L1-L200】
- **/test-gemini** e **/test-bigquery** validam integrações externas essenciais para Vertex/Gemini.【F:api/routers/gemini_test.py†L1-L30】【F:api/routers/bigquery_test.py†L1-L38】

## 3. Serviços de Domínio
- **RAG Vertex** (`services/rag_vertex.py`) encapsula backends de Engine e Vector Search, com cache interno e checagem de saúde para respostas com citações.【F:services/rag_vertex.py†L1-L120】
- **RAG++ nativo** (`services/rag_plus.py`) inicializa clientes Vertex/BigQuery, gera o esquema da base de conhecimento, monitora discovery e coordena jobs contínuos.【F:services/rag_plus.py†L1-L160】
- **Cache de contexto** (`services/context_cache.py`) identifica prefixos estáveis, mantém métricas e prepara prompts otimizados para Gemini.【F:services/context_cache.py†L1-L140】
- **Memória de sessões** (`api/routers/memory.py`) combina armazenamento JSONL/SQLite com busca textual, reutilizando a mesma infraestrutura usada pelos endpoints.【F:api/routers/memory.py†L57-L200】
- **Tree Search** (`services/tree_search.py`) implementa PUCT/MCTS com widening progressivo, métricas e serialização dos nós para uso via API.【F:services/tree_search.py†L1-L160】
- **Score contracts** (`services/score_contracts.py`) define contratos, sandbox de execução assíncrona e normalização de métricas para KEC, ZuCo e análises editoriais.【F:services/score_contracts.py†L1-L188】

## 4. Ferramentas RAG++ e Discovery
- **`rag_plus_agent.py`** implementa o agente de pesquisa longo-horizonte (BigQuery + Vertex, ReAct, ingestão contínua e citações).【F:rag_plus_agent.py†L1-L200】
- **`rag_plus_main.py`** orquestra execuções CLI, carrega YAML, inicia agente + radar e oferece modos demo/monitoramento.【F:rag_plus_main.py†L1-L120】
- **`scientific_discovery_radar.py`** monitora fontes científicas (RSS/API), calcula novidade por embeddings e alimenta a base de conhecimento.【F:scientific_discovery_radar.py†L1-L176】

## 5. Pesquisa, Dados e Utilidades
- **`src/datasets/`** contém loaders robustos (ex.: `ZuCoRealLoader`) para EEG/eye-tracking com validações e extração de features reais.【F:src/datasets/zuco_real_loader.py†L1-L108】
- **`src/`** também abriga utilitários numéricos, estatísticos e de qualidade (`pcs_toolbox`, `pcs_math`, `pcs_opt`, `pcs_qc`, `pcs_graph`, `swow`) utilizados nos notebooks e pipelines descritos no README.【F:README.md†L48-L70】
- **`notebooks/`, `data/` e `outputs/`** participam do pipeline reprodutível v5, com datasets reais e notebooks didáticos ligados aos módulos acima.【F:README.md†L12-L27】

## 6. Automação Operacional e Documentação
- **`ops/`** fornece pipelines shell completos para coleta de telemetria, alimentação do servidor e validações (ex.: `feed-server-pipeline.sh` documentado em `IMPLEMENTATION_COMPLETE.md`).【F:ops/IMPLEMENTATION_COMPLETE.md†L1-L67】
- **Scripts utilitários** como `scripts/generate_docs.py` automatizam a geração dos guias API/G1 e reforçam o fluxo de documentação contínua.【F:scripts/generate_docs.py†L1-L160】
- **Guardrails automatizados** (`agents.md`) definem pre-commit, lint de Markdown, lint de YAML e verificações de reprodutibilidade executadas pelo pipeline de qualidade.【F:agents.md†L1-L104】

## 7. Testes e Qualidade
- A suíte `tests/test_darwin_platform.py` cobre configurações, RAG Vertex, segurança, contratos, cache e tree-search, servindo como regressão central da API Darwin.【F:tests/test_darwin_platform.py†L1-L120】
- O README documenta rotinas de pytest, cobertura e validação F1 para quem precisa executar as checagens manualmente.【F:README.md†L124-L160】

Use esta visão como mapa inicial: cada seção aponta para módulos ativos e mantidos, evitando confundir com scripts legados ou deprecados. Ajustes de configuração ou extensões devem respeitar os serviços e routers descritos acima para preservar a arquitetura da Darwin RAG++ Platform.
