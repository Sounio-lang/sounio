---
title: "Manifesto da Computação Epistêmica"
description: "Computação na fronteira entre o que sabemos e o que não sabemos"
layout: "manifesto"
subtitle: "Cinco Princípios para Computação Científica"
---

> *"A medida da inteligência é a capacidade de mudar."*
> — Albert Einstein

Por décadas, as linguagens de programação trataram os números como perfeitos: `3.14159` é exatamente isso, nem mais, nem menos. Mas a ciência não funciona assim. Toda medição tem erro. Todo modelo tem incerteza. Toda previsão tem limites de confiança.

**Sounio** é construído sobre uma premissa radical: **a incerteza não é um bug — é um recurso**.

---

## Os Cinco Princípios da Computação Epistêmica

### 1. Todo Conhecimento é Incerto

No mundo físico, não existe algo como uma medição perfeita. O princípio da incerteza de Heisenberg não é uma limitação de nossos instrumentos — é uma propriedade fundamental da realidade. Mesmo medições macroscópicas carregam ruído, erro de calibração e precisão finita.

```sio
// Errado: fingindo que sabemos exatamente
let concentration = 5.23  // mg/L... mas realmente?

// Certo: reconhecendo a incerteza
let concentration = Knowledge::new(5.23 mg/L, uncertainty: 0.15 mg/L)
```

Sounio torna isso explícito. Quando você declara um valor, deve considerar: *quão bem eu realmente conheço isso?*

### 2. A Proveniência é Inegociável

Dados sem origem são dados sem confiança. Quando uma agência regulatória pergunta "de onde veio esse número?", você deve ter uma resposta que rastreie de volta às fontes primárias.

```sio
let clearance = Knowledge::new(
    value: 10.5 L/h,
    uncertainty: 1.2 L/h,
    source: Source {
        origin: "Phase III Trial NCT04123456",
        timestamp: 2025-03-15,
        method: "Population PK analysis",
        confidence: 0.95
    }
)
```

Todo `Knowledge<T>` carrega sua proveniência. A linhagem de seus dados é tão importante quanto os dados em si.

### 3. A Incerteza Propaga Automaticamente

A propagação manual de incerteza é tediosa e propensa a erros. O GUM (Guide to the Expression of Uncertainty in Measurement) define como as incertezas se combinam por meio de operações matemáticas. Sounio implementa isso automaticamente.

```sio
let mass = Knowledge::new(100.0 g, uncertainty: 0.5 g)
let volume = Knowledge::new(50.0 mL, uncertainty: 0.2 mL)

// Cálculo de densidade com propagação automática
let density = mass / volume
// density.uncertainty é computado via GUM:
// δρ/ρ = sqrt((δm/m)² + (δV/V)²)
```

Você escreve a física. O compilador cuida da estatística.

### 4. Portões de Confiança Controlam a Execução

Nem todas as computações devem prosseguir cegamente. Quando a confiança cai abaixo de um limiar, a execução deve pausar, avisar ou tomar caminhos alternativos.

```sio
fn critical_decision(data: Knowledge<f64>) -> Action {
    if data.confidence < 0.90 {
        return Action::RequestMoreData
    }

    if data.confidence < 0.95 {
        return Action::ProceedWithCaution(data)
    }

    Action::Proceed(data)
}
```

Isso não é programação defensiva — é *programação epistêmica*. O sistema sabe o que não sabe.

### 5. Conformidade com Padrões por Design

A ciência tem padrões por uma razão. Sounio é construído para cumprir:

- **GUM** — ISO Guide to the Expression of Uncertainty in Measurement
- **ISO 17025** — Competence of testing and calibration laboratories
- **21 CFR Part 11** — Electronic records and signatures (FDA)
- **FAIR Principles** — Findable, Accessible, Interoperable, Reusable data

Esses não são pensamentos posteriores — são fundações arquiteturais.

---

## O Problema que Estamos Resolvendo

### A Crise de Reprodutibilidade

Entre 2011 e 2021, estima-se que US$ 28 bilhões foram desperdiçados em pesquisas pré-clínicas irreprodutíveis apenas nos Estados Unidos. As causas são muitas, mas uma se destaca: **perda de informação de incerteza**.

Quando uma medição de `5.23 mg/L` é passada entre sistemas, armazenada em bancos de dados e usada em cálculos — o `±0.15` frequentemente desaparece. Análises downstream a tratam como exata. Conclusões são tiradas que a incerteza original teria impedido.

### A Solução

Sounio torna a incerteza *contagiosa*. Você não pode perdê-la acidentalmente. O sistema de tipos não permite converter `Knowledge<T>` para um `T` nu sem reconhecimento explícito.

```sio
let safe_value = measurement.value  // Erro do compilador!

let safe_value = measurement.unwrap_certain()  // Requer confidence > 0.99

let safe_value = measurement.acknowledge_uncertainty()  // Opt-out explícito, registrado
```

---

## Por Que "Sounio"?

Cape Sounion, na ponta da Ática, é onde antigos marinheiros gregos observavam o horizonte. O Templo de Poseidon lá era tanto um marco quanto uma oração — um ponto fixo de onde navegar pelo mar incerto.

Sounio, a linguagem, serve ao mesmo propósito: uma fundação estável para navegar dados incertos. As colunas são seu sistema de tipos. O mar é seu domínio científico. O horizonte é onde a certeza termina e a exploração começa.

Lord Byron visitou em 1810 e gravou seu nome no mármore (por favor, não faça isso). Ele escreveu:

> *"Place me on Sunium's marbled steep,"*
> *"Where nothing, save the waves and I,"*
> *"May hear our mutual murmurs sweep;"*
> *"There, swan-like, let me sing and die."*

(Coloque-me no pico de mármore de Sunium,
Onde nada, exceto as ondas e eu,
Possa ouvir nossos murmúrios mútuos varrerem;
Lá, como um cisne, deixe-me cantar e morrer.)

Não somos tão dramáticos. Mas estamos construindo algo que, como aquelas colunas, pode durar.

---

## O Caminho Adiante

Sounio não está terminado. Pode nunca estar. Mas os princípios estão definidos:

1. **A incerteza é de primeira classe** — Não uma biblioteca, não uma anotação, mas um tipo fundamental.

2. **A propagação é correta** — Compatível com GUM, testada, verificada.

3. **A proveniência é preservada** — Da fonte ao resultado, a cadeia é inquebrantável.

4. **A confiança é acionável** — O sistema responde ao que sabe e ao que não sabe.

5. **Os padrões são integrados** — A conformidade não é opcional.

Se você acredita que a ciência merece ferramentas melhores — que a incerteza deve ser computada, não ignorada — que a reprodutibilidade é um recurso, não um acidente — então Sounio é para você.

---

*Junte-se a nós no horizonte.*

**🏛️ SOUNIO 🌊**
