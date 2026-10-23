<!-- docs:meta
topic_id: repo.docs.llm-guide.translation-and-repair
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm-guide.translation-and-repair
-->

# Translate intent, then repair with evidence

Use this with the [task curriculum](task-playbook.md), [full manual](../guide/LLM_PROGRAMMING_GUIDE.md),
[Portuguese manual](../guide/LLM_PROGRAMMING_GUIDE_PT.md) and
[route-aware error catalog](error-catalog.md). This is a practical method,
not a promise that every foreign feature has a Sounio equivalent. Examples
of foreign habits below are descriptions, not invented runnable Sounio.

## 1. Extract the contract before translating spelling

Record the original algorithm's inputs, output representation, failure
behavior, ownership, mutation, effects, precision, dependence assumptions,
units and observable scientific channels. Identify which properties belong
to the user requirement and which merely reflect the source language.

For example, “a growable sequence of real measurements with explicit source
identity” has three distinct obligations: collection behavior, numeric
representation and provenance. A fixed array of plain numbers may resemble
the source loop while violating two of those obligations. Do not call that
translation complete.

For non-associative or floating operations, preserve parenthesization and
evaluation dependencies. For uncertain inputs, preserve source identity and
dependence assumptions. Do not collapse absent information into zero, an
unknown channel into a confident estimate, or a tagged sample into an actual
sensor read. Consult [founder intent](../../FOUNDER_INTENT.md) before changing
the meaning of a research primitive.

## 2. Pick a witnessed representation

Find the closest complete task pack, read the real exported declarations and
make a representation decision explicitly. A familiar name is not an API.

| Foreign habit | Sounio evidence to retrieve | Unsafe shortcut to avoid |
|---|---|---|
| Python list or C++ growable vector | `bounded-vector`, concrete export and capacity behavior | Assuming a generic, unlimited `Vec<T>` API from the collection name. |
| Python dynamic object or Rust record | `record-construction`, `record-parameter`, actual field declarations | Adding undeclared fields or depending on runtime attribute lookup. |
| Python tuple unpacking or Rust destructuring | `tuple-access`, `tuple-signatures`, exact selected-route witness | Recommending the quarantined destructuring fixture as passed. |
| Rust mutable reference | `exclusive-array-update`, reference spelling, enclosing effect declaration | Blind replacement without checking aliasing and the write location. |
| C++ const reference or Rust shared borrow | `shared-reference`, owner and reborrow path | Strengthening permission or copying the scientific value to hide a conflict. |
| Lambda, closure capture or anonymous callback | `named-callback`, signature and named-function route | Calling named-function success proof of closure capture support. |
| Rust generic call | `generic-function`, complete declaration and passed call syntax | Mixing spellings from failed bare-angle cases. |
| Exception or implicit error propagation | `explicit-result`, actual error payload and branches | Inventing exceptions, automatic conversion or foreign propagation operators. |
| Optional/nullable return | `optional-control-flow`, actual enum and matched consumer | Equating one shorthand check with every optional runtime representation. |
| Module import | Both public-module tasks and provider source | Moving just the consumer while assuming its dependency remains discoverable. |
| Print/debug formatting | `first-program`, actual printing signature | Importing macro syntax or arbitrary variadic format arguments. |
| Implicit IO or mutation | `pure-versus-io`, `effect-superset`, relevant negative control | Adding all effects to every signature or declaring unsupported effect names. |
| Plain numeric library result | `math-library-import`, actual type, domain and effects | Narrowing required precision or changing tolerances to get a pass. |
| Measurement wrapper | `input-provenance`, `measured-knowledge`, `epistemic-library` | Treating Knowledge, Epistemic and EISA as interchangeable wrappers. |
| Parallel loop, async or GPU request | Target chapter, operation-specific lowering and real device witness | Treating a parsed declaration, simulator output or request enum as execution support. |

The task pack includes full reference/control programs; this table is a
retrieval index, not substitute source code. All catalog exercises are
adaptations of existing sources, not newly executed application examples.

## 3. Translate one semantic unit

First translate data representation, then one pure operation, then its call
boundary, then required observation/effects. Only after these are understood
should an agent combine modules or add a second abstraction. Keep the original
program and initial complete Sounio reference available for comparison.

When the requested capability has no evidenced route, provide a precise gap:
the required operation, inspected signature/source, selected artifact,
earliest failing stage, and the invariant that must not be weakened. An
explicit limitation is more useful than a different program presented as
success. A proposed alternative needs the user's approval when it changes
scientific meaning, representation obligations or authorized infrastructure.

If a simpler representation really preserves the contract, state the
equivalence and its limits; establish it with appropriate tests or proof.
Do not imply that familiar syntax, a successful parse or an LLM's confidence
establishes that equivalence.

## 4. Diagnose the stage, not the last line

| Observation | First discriminating question | Preserve before repair |
|---|---|---|
| Missing executable / permission / wrong format | Did the compiler launch on a compatible authorized host? | Actual selected raw path, digest, architecture and launcher stderr. |
| Parse refusal | What is the earliest relevant source span and exact construct? | Full source identity, original diagnostic, selected engine. |
| Unknown symbol / module | Is the declaration exported and is its provider visible on this route? | Import path, actual signature, library root and dependency files. |
| Type or generic mismatch | What representation and specialization does the real callee accept? | Both declaration and call; do not erase types to silence the message. |
| Borrow conflict | Which value is owned, borrowed, written or still live? | Last uses, exclusive/shared intent and the negative control's reason. |
| Effect mismatch | Which operation/callee generates the obligation on this route? | Caller/callee effect lists and exact emitter/message. |
| Lowering/emission failure after check | Was the construct accepted but unsupported or inconsistent later? | Check, compile and emission logs separately; identify the artifact if any. |
| Runtime crash or timeout | Did execution start, and did the operation terminate within the declared budget? | Fresh artifact identity, actual status/signal and output; never count as intended refusal. |
| Exit mismatch | Was the intended result zero, a nonzero application value or a refusal? | Per-case contract; do not normalize all nonzero results into the same category. |
| Missing output marker | Did the program implement the task, or only print compiler progress? | Full stdout/stderr and the source-level observation path. |
| Correct central value, missing scientific channel | Were uncertainty, units, source identity, precision or ordering lost? | All required channels and dependency assumptions, not only the scalar result. |
| CLI and MCP disagree | Were they the same artifact, library root and effective environment? | Server-side identity; current MCP normalization alone does not supply it. |

Names such as `PREFLIGHT STOP` and quarantine are guide/reporting concepts,
not newly invented compiler diagnostic codes or runtime thresholds. Resolve
actual diagnostic identities using the source-backed error catalog. Many
historical per-code pages remain explicitly not refreshed; do not let direct
retrieval erase their warning.

## 5. Use a bounded repair loop

```text
1. Preserve the failing candidate and the exact first relevant diagnostic.
2. Classify host/launcher, parse, resolve, type/effect/borrow, lowering,
   emission, execution, assertion, or scientific-channel failure.
3. Retrieve one declaration and one closest complete witness.
4. State a specific cause and a minimal change that could distinguish it.
5. Change only the authorized candidate; do not rewrite reference expectations.
6. Repeat the failed stage. If it passes, validate downstream stages required
   by the task. Preserve a nearby control that should still pass or refuse.
7. If the same cause repeats without new evidence, stop the blind rewrite
   loop, retain artifacts, and report the missing capability or decision.
```

The user/environment determines the time and repair budget. This procedure
does not grant infinite retries, permission to install tools, or authority
to execute on an unapproved host. A failing control must not be removed
merely to make the ledger green.

### Worked repair scenario: generic syntax

The user requests a small generic helper. The model has a failing bare-angle
fixture but the same profile has a passed `generics` example. Retrieve
`generic-function`, compare the declaration and exact call spelling, and
change that syntax only in the authorized candidate. Do not simultaneously
replace generic values with concrete numbers, alter the algorithm, or
declare that Sounio has no generics.

The old failed fixture remains quarantined. A new passing candidate, if
actually observed, supports only its own source/profile/contract. It does
not retroactively promote a different failed source.

### Worked repair scenario: imports

The model copied the public consumer into another directory and now cannot
resolve its provider. Retrieve `public-module-consumer` and its included
provider. Check declaration visibility, relative/conventional discovery and
the working directory against real resolver behavior. Preserve both files
and check them under one identified library environment.

Do not invent a package-manager command, change all import separators, or
copy an unrelated module with a similar name. The original pair has check-only
evidence: a request for runtime behavior still needs actual compilation and
execution of the intended application entry point.

### Worked repair scenario: a false negative-test pass

A unit-bearing program was intended to fail for a dimensional mismatch but
instead failed to parse an unrelated token. Nonzero alone is insufficient.
Preserve the source, status and exact message; classify it as failure to
reach the intended refusal, not a dimensional-safety witness. Establish a
positive route for the syntax before retrying the semantic negative control.

Do not erase the units, loosen the diagnostic contract to “any error”, or
promote a runtime crash to an expected refusal. A missing scientific
capability is an open requirement, not permission to replace the question.

### Worked repair scenario: a missing effect

The model added a store through an exclusive reference. Retrieve
`exclusive-array-update`, its real signature and the intended missing-`Mut`
control. Identify the operation generating the obligation and the relevant
caller boundary. Add or propagate only the required declaration where
appropriate, then repeat the actual changed source and discriminating control.

Do not copy the missing-effect negative program as the working recipe. Do
not mechanically add mutation effects to unrelated local loops. A successful
repair on this engine is not evidence about every alternate checking route.

## 6. Ask for the right kind of answer

### Chat-only generation

```text
Use the supplied complete task pack as reference data. Produce a complete
candidate for my stated task, preserving the requested semantics and exact
API spelling. List required files/imports and unresolved assumptions.
Do not claim to have executed tools. Give the authorized verification steps
as instructions, clearly separated from observations. Do not fabricate logs.
```

### Agent with tools

```text
Within the stated write set, reproduce the unchanged reference on the
authorized identified profile, then make one candidate change. Check and,
where required, compile/run that changed file. Preserve commands, source
hashes and actual outputs. Do not switch engines, weaken scientific channels,
modify reference contracts or publish anything without authorization.
```

### Diagnostic repair

```text
Here are the complete changed source, profile and actual diagnostic streams.
Identify the earliest relevant cause, retrieve the real declaration and one
closest witness, and propose the smallest discriminating repair. Preserve
the failing observation and controls. Do not claim another stage passed
unless the supplied evidence or an authorized new execution establishes it.
```

### Scientific translation

```text
First enumerate representation, precision, units, dependence, uncertainty,
confidence, provenance and ordering obligations from my original task.
Translate one operation without silently replacing these obligations.
Separate source inspection, execution, ADR009-authorized numerical checking,
Lean proof and empirical evidence. State any capability gap precisely.
```

## 7. Completion checklist

- The candidate is complete or all required source files are supplied.
- Every called API exists with the stated visibility, types and effects.
- Ownership and explicit/exclusive borrowing preserve the task's intent.
- Inputs, success/error paths and observable behavior match the requirement.
- Current execution evidence belongs to this candidate and named profile.
- Historical or check-only evidence is not reported as a fresh runtime pass.
- Required numerical/scientific channels are neither narrowed nor discarded.
- Negative controls fail for the intended reason, not a launch error or crash.
- The delivery lists what changed, actual commands/results and pending stages.
- No commit, push, PR, upload or third-party message is implied by generation.

## Em português: traduzir intenção e reparar sem apagar significado

Antes de converter grafia, registre entradas, representação, saídas, erros,
posse, mutação, efeitos, precisão, unidades, dependência e proveniência.
Escolha uma representação com fonte e testemunho reais. Listas de Python,
vetores de C++, tuplas, closures e Result de Rust não autorizam inventar APIs
Sounio semelhantes. O catálogo oferece referências completas para cada
família; preserve a grafia e as assinaturas reais antes de adaptar.

Traduza uma unidade semântica por vez: dados, operação, chamada, observação.
Se a capacidade necessária não estiver demonstrada, descreva operação,
artefato, etapa que falhou e invariante que não pode ser enfraquecido. Não
troque uma pergunta científica por outra mais fácil e a apresente como sucesso.

No reparo, preserve fonte e diagnóstico originais; classifique lançamento,
parse, resolução, tipos/efeitos/empréstimos, lowering, emissão, execução ou
asserção. Leia uma declaração real e uma referência próxima; faça a menor
mudança discriminante; repita a etapa relevante e as etapas seguintes
exigidas. Um controle que falha pelo motivo errado continua falhando.

Os quatro cenários detalhados deste guia tratam sintaxe genérica, imports,
falso passe de teste negativo e efeito ausente. Nenhum permite retirar
unidades, reduzir precisão, apagar proveniência, inventar helpers, promover
crashes a recusas esperadas ou modificar a expectativa do teste para ficar verde.
Os nomes de preflight e quarentena são conceitos do guia, não novos códigos
de diagnóstico nem limiares secretos do compilador.

Modelo sem ferramentas entrega candidato e verificação pendente, nunca logs
inventados. Agente com ferramentas identifica ambiente/perfil e respeita
arquivos e operações autorizados. O pacote em português sai de `show` com
`--lang pt`; tipos, efeitos e identificadores do código não são traduzidos.
Ao concluir, separe esperado de observado, checagem de execução, cálculo de
prova e implementação de evidência empírica. Essa disciplina é parte da
facilidade de uso: evita ensinar ao próximo modelo uma falsa linguagem.
