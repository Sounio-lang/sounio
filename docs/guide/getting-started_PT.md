<!-- docs:meta
topic_id: repo.docs.guide.getting-started-pt
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.getting-started-pt
-->

# Introdução ao Sounio

> **Outros guias**: [Início Rápido para Cientistas](../QUICK_START_GUIDE.md) | [Início Rápido para LLMs](SOUNIO_QUICK_START.md) | [Contrato Conservativo](MINIMUM_VIABLE_SOUNIO.md)

Bem-vindo ao **Sounio**, uma linguagem de programação e plataforma de pesquisa para código científico que requer incerteza explícita, proveniência e validação baseada em portas (gates) de qualidade.

Este guia é intencionalmente conservador e reflete o estado do repositório validado em 22 de abril de 2026.

## 1. Utilizando um Artefato de Compilador Real

Para este checkout, o caminho mais fácil é usar o inicializador do compilador self-hosted em `bin/souc`. Ele seleciona automaticamente o artefato correspondente para Linux `x86_64`, macOS `arm64` ou macOS `x86_64`:

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio

export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" --version
"$SOUC_BIN" info
"$SOUC_BIN" check examples/hello.sio
"$SOUC_BIN" compile examples/hello.sio -o /tmp/souc-madaros
"$SOUC_BIN" compile examples/hello.sio -o /tmp/hello-macos --target aarch64-macos
```

Neste snapshot do repositório, `bin/souc` é o padrão conservador para trabalho local e roteia para **Madaros**.
Se você precisar da engine legado de bootstrap para checagens de compatibilidade, use `SOUNIO_SOUC_ENGINE=lean_single` na invocação.
Ele resolve o artefato do host automaticamente, expõe comandos de compatibilidade para `check`/`run`/`compile`/`build` e ainda suporta a interface bruta do compilador self-hosted quando você deseja uma invocação explícita de `<origem> <destino>`.

Há também um artefato separado de GPU/JIT para Linux `x86_64` específico para fluxos de trabalho com GPU:

```bash
export SOUC_GPU_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
"$SOUC_GPU_BIN" info
"$SOUC_GPU_BIN" check examples/gpu.sio
"$SOUC_GPU_BIN" build examples/kernel_matmul.sio --backend gpu -o /tmp/kernel_matmul.ptx
```

Se você precisar que o repositório resolva um caminho de binário fixado para você:

```bash
scripts/omega/omega_resolve_souc_bin.sh --print-path --allow-local-fallback
```

## 2. Começando com Verificações de Fumaça (Smokes) Conservadoras

A maneira mais confiável de validar o artefato self-hosted é usar os comandos de compatibilidade enquanto se prova que o compilador consegue reconstruir a si mesmo.

```bash
"$SOUC_BIN" check examples/hello.sio
"$SOUC_BIN" compile examples/hello.sio -o /tmp/souc-madaros
"$SOUC_BIN" run self-hosted/compiler/native_print_f64_smoke.sio
```

Comportamento esperado:

- Em hosts Linux, as saídas compiladas são binários ELF nativos.
- Em hosts macOS, as saídas compiladas são binários Mach-O nativos para o destino selecionado.
- Saídas de compilação cruzada (cross-target) devem ser executadas no sistema operacional/arquitetura de destino correspondente.

## 3. Seu Primeiro Programa

Crie um arquivo chamado `hello.sio`:

```sounio
fn main() with IO {
    println("Hello, Sounio!")
}
```

Compile-o:

```bash
"$SOUC_BIN" compile hello.sio -o /tmp/hello.out
```

## 4. O Que Está Realmente Verificado Hoje

O resumo público suportado por portas de qualidade (gates) neste repositório é:

- `artifacts/stdlib/stdlib_reliability_status.v1.json`: `251 pass / 0 fail / 0 skip / 251 total`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`: `pass` para os pipelines de `fmri` e `darwin_pbpk`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`: `pass` para as 7 vias (lanes) hyper requeridas
- `artifacts/omega/gpu_runtime_attest_gate.v1.json`: `pass` para o conjunto de testes de fumaça do runtime de GPU na via verificada
- Provas de regressão de runtime científico local ainda são registradas em modo "soft", a menos que a aplicação estrita de CI esteja habilitada.

Para o contrato conservador completo, leia o [Mínimo Viável do Sounio](MINIMUM_VIABLE_SOUNIO.md).

## 5. Conceitos Chave

### 1. Tipos Epistêmicos

A funcionalidade assinatura do Sounio é o tipo `Knowledge<T>`:

```sounio
let risky = Knowledge { value: 15.0, epsilon: 0.4 }
let safe = Knowledge { value: 15.0, epsilon: 0.9 }
```

### 2. Variáveis

```sounio
let x = 5       // Imutável
var y = 10      // Mutável

y = y + 1
```

### 3. Referências

O Sounio usa `&!` para referências exclusivas e mutáveis:

```sounio
fn increment(x: &!i32) {
    *x = *x + 1
}
```

### 4. Unidades Físicas

As unidades físicas agora são verificadas estaticamente, incluindo assinaturas `f64<UnitExpr>`:

```sounio
let distance: f64<m> = 100.0 m
let time: f64<s> = 9.58 s
let speed = distance / time
```

### 5. Demos Interativas Rápidas (PPM / Unidades / GUM / Portas)

Para experimentar os recursos únicos de ciência e metrologia do Sounio diretamente neste checkout, execute as seguintes demonstrações interativas nativas:

```bash
# Demo 1: Verificação dimensional estática (compilação aceita)
./bin/souc run demo_unidades.sio

# Demo 2: Propagação analítica de incertezas em tempo real (ISO GUM)
./bin/souc run demo_incerteza.sio

# Demo 3: Portas de confiança dinâmicas (Sucesso - guarda aceita)
bash scripts/ontology/expand_knowledge_runtime_guards.sh demo_portas_sucesso.sio /tmp/demo3_sucesso.sio && ./bin/souc run /tmp/demo3_sucesso.sio

# Demo 4: Portas de confiança dinâmicas (Rejeição - asserção falha no runtime)
bash scripts/ontology/expand_knowledge_runtime_guards.sh demo_portas_rejeicao.sio /tmp/demo3_rejeicao.sio && ./bin/souc run /tmp/demo3_rejeicao.sio
```

### 6. Efeitos Algébricos

Todas as funções que causam efeitos colaterais devem declará-los explicitamente usando a cláusula `with`:

```sounio
fn read_file(path: &str) -> String with IO {
    "demo"
}
```

## 6. Referência de Comandos

```bash
souc check file.sio
souc run file.sio
souc build file.sio -o output
souc check file.sio --show-ast
souc check file.sio --show-types
```
