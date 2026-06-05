# NAMING.md — Ecossistema Sounio / The Sounio Ecosystem

> Convenção de nomes do compilador Sounio e suas peças.
> Naming convention for the Sounio compiler and its components.
>
> **Eixo / Axis:** geografia de Sfakiá, Creta — raiz familiar dos autores.
> Sfakia, Crete — the authors' ancestral ground.
>
> Cada nome *argumenta*: não é toponímia decorativa, é função.
> Each name *argues*: not decorative toponymy, but function.

---

## Tabela mestre / Master table

| Peça / Component | Nome / Name | Binário / Binary | Lugar–mito / Place–myth |
|---|---|---|---|
| Compilador / Compiler | **Madáres** (Μαδάρες) | `madares` | Altiplano nu das Lefká Óri / The bare highland of the White Mountains |
| Verificador formal Lean 4 / Lean 4 formal verifier | **Rhadamanthys** (Ῥαδάμανθυς) | `rada` | Juiz incorruptível e legislador de Creta / Incorruptible judge and lawgiver of Crete |
| Runtime | **Loutró** (Λουτρό) | `loutro` | Porto acessível só por mar ou a pé / Harbour reachable only by sea or on foot |
| LSP server | **Arádena** (Αράδαινα) | `aradena` | Garganta com a ponte suspensa / Gorge with the suspended bridge |
| Gestor de pacotes / Package manager | **Sfakiá** (Σφακιά) | `sfakia` | A região, o território comum / The region, the common ground |
| Cluster de agentes / Agent cluster | **Drosoulites** (Δροσουλίτες) | `drosoulites` | Exército espectral de Frangokastello / The spectral army of Frangokastello |
| Formatter | **Lefká** (Λευκά) | `lefka` | "As brancas" — as Montanhas Brancas / "The white ones" — the White Mountains |

---

## Etimologia e justificação / Etymology and rationale

### Madáres — o compilador / the compiler
**PT:** O nome sfakiano local do altiplano de pedra e vento das Lefká Óri — não o "Montanhas Brancas" turístico, mas o termo dos pastores da região. Superfície dura que não negocia com código mal-formado. O sistema de efeitos `NonUnitary`, o type-checking e as provas Lean zero-sorries são a recusa do altiplano em deixar passar o que não se sustenta.
**EN:** The local Sfakian name for the stone-and-wind highland of the White Mountains — the shepherds' word, not the tourist label. The hard surface that does not negotiate with ill-formed code. The `NonUnitary` effect system, type-checking and zero-sorry Lean proofs are the highland's refusal to let through what cannot stand.

### Rhadamanthys — o verificador formal / the formal verifier
**PT:** Irmão de Minos, juiz incorruptível dos mortos e legislador mítico de Creta. O type-checker como magistrado que sentencia correção sem apelo.
**EN:** Brother of Minos, incorruptible judge of the dead and mythical lawgiver of Crete. The type-checker as a magistrate that rules on correctness without appeal.

### Loutró — o runtime
**PT:** Porto sfakiano a que só se chega por mar ou a pé. O destino onde o código compilado aporta e ganha vida — alcançado apenas após a travessia.
**EN:** Sfakian harbour reachable only by sea or on foot. The destination where compiled code lands and comes alive — reached only after the crossing.

> *Nota / Note:* primeira escolha **Talos** (autómato de bronze guardião de Creta) descartada — ocupada nos três registries. / first choice **Talos** (Crete's bronze guardian automaton) dropped — taken on all three registries.

### Arádena — o LSP server
**PT:** Garganta sfakiana cuja ponte suspensa cruza o abismo. A ponte literal entre editor e compilador, vencendo o vão em tempo real. Já auto-hospedado em Sounio (38 métodos).
**EN:** Sfakian gorge whose suspended bridge spans the chasm. The literal bridge between editor and compiler, crossing the gap in real time. Already self-hosted in Sounio (38 methods).

### Sfakiá — o gestor de pacotes / the package manager
**PT:** A região inteira: o território comum onde os módulos vivem. `sfakia add`, `sfakia publish`.
**EN:** The whole region: the common ground where modules live. `sfakia add`, `sfakia publish`.

### Drosoulites — o cluster de agentes / the agent cluster
**PT:** O exército espectral que se materializa ao amanhecer em Frangokastello. O daemon que conjura agentes paralelos (6× Claude Code + Codex) para o sprint — a hoste fantasma que aparece, executa o trabalho noturno e some.
**EN:** The spectral army that materialises at dawn over Frangokastello. The daemon that conjures parallel agents (6× Claude Code + Codex) for the sprint — the ghost host that appears, runs the night's work, and vanishes.

### Lefká — o formatter
**PT:** "As brancas" — as Montanhas Brancas. Branco = limpo: o formatter que devolve o código ao estado imaculado. `lefka fmt`.
**EN:** "The white ones" — the White Mountains. White = clean: the formatter that returns code to its immaculate state. `lefka fmt`.

---

## Grafia / Spelling

- **Pacotes e binários / Packages & binaries:** sempre ASCII minúsculo — `madares`, `rada`, `loutro`, `aradena`, `sfakia`, `drosoulites`, `lefka`. Sem acentos (quebram em paths/URLs). / always lowercase ASCII; no accents (they break in paths/URLs).
- **Logo, README, prosa / Logo, README, prose:** grego original ou forma acentuada — Μαδάρες / Madáres.

---

## Disponibilidade verificada / Verified availability

Verificado em / verified on 2026-06-05 — crates.io, PyPI, npm. Todos livres / all free:

| Nome / Name | crates.io | PyPI | npm |
|---|---|---|---|
| madares | ✅ | ✅ | ✅ |
| rhadamanthys | ✅ | ✅ | ✅ |
| loutro | ✅ | ✅ | ✅ |
| aradena | ✅ | ✅ | ✅ |
| sfakia | ✅ | ✅ | ✅ |
| drosoulites | ✅ | ✅ | ✅ |
| lefka | ✅ | ✅ | ✅ |

> GitHub: repositórios sob a org `agourakis82` (o handle de org nu é irrelevante).
> GitHub: repos under the `agourakis82` org (the bare org handle is irrelevant).

---

## Reservas / Reserves

Livres nos três registries, para expansão futura / free on all three registries, for future growth:

`pachnes` (pico mais alto, 2.453 m — kernel/otimizador / highest peak — kernel/optimiser) ·
`samaria` (a garganta célebre — debugger / the famous gorge — debugger) ·
`askifou` (o planalto — REPL/playground / the plateau) ·
`volakias`, `gigilos` (picos — sentinelas / peaks — sentinels) ·
`imbros`, `tripiti`, `komitades`, `selouda`, `anopoli`, `frangokastello`

---

*Autoria / Authorship:* Demetrios Chiuratto Agourakis; Dionisio Chiuratto Agourakis.

*Disclosure (GAIDeT/ICMJE 2025):* nomenclatura desenvolvida com assistência de Claude (Anthropic, Opus 4.8), incluindo verificação de disponibilidade em registries. / naming developed with assistance from Claude (Anthropic, Opus 4.8), including registry availability checks.
