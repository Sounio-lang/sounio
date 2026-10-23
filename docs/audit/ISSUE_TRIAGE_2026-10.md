<!-- docs:meta
topic_id: repo.docs.audit.issue-triage-2026-10
authority: repo_only
audience: users
last_validated: 2026-10-13
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.issue-triage-2026-10
-->

# Triagem das issues abertas — outubro de 2026

Inventário das **117** issues abertas em `Sounio-lang/sounio` em 2026-10-07 (API REST `/issues?state=open&per_page=100&page=N`, páginas 1–2; os pull requests que a mesma rota devolve foram descartados), mais as três que a própria campanha encontrou (#2874, #2881, #2896). Medição inicial em `main` `d834ab4e5`; refeita em `deede5c87` (ver abaixo) e conferida de novo em `c2e3df23c` para os estados desta tabela.

Regras da campanha: nenhuma issue nova; uma issue só fecha com prova (correção própria com teste que falha antes e passa depois, já corrigida em `main` com reprodução rodada e commit citado, duplicata com a canônica, ou obsoleta com o caminho ausente), nunca por idade, pelo rótulo `stale` ou por "provavelmente resolvida"; decisões do mantenedor vão para [`docs/decisions/PENDING.md`](../decisions/PENDING.md) e a issue fecha com o link da seção. **Nenhum PR foi mesclado.**

## Legenda

| disp. | significado |
|---|---|
| a | corrigida por commit desta campanha, com teste que falha antes e passa depois; fecha quando o PR for mesclado (`Fixes #N`) |
| b | já corrigida em `main`; reprodução rodada em `main` e commit da correção citados |
| c | duplicata; conteúdo útil movido para a issue canônica |
| d | obsoleta: o código, arquivo ou alvo não existe mais |
| decisão | decisão do mantenedor ou pergunta científica; registrada em `PENDING.md` (seção D-nn) |
| aberta | trabalho ainda não feito ou bloqueado; ver a coluna "pendência" |

**Estado**: `fechada` (com o link do comentário de prova), `fechada → decisão D-nn` (fechada com o link da seção), `PR aberto` (fecha no merge), `aberta`. **CI** é o resultado dos check-runs no head exato do PR no momento da conferência (2026-10-13).

**Dono**: letra da sessão paralela que pegou a issue (A, C, E, H, L1, L2, M, N, S, T: uma branch e um worktree por PR), para que duas sessões não trabalhem na mesma.

## Miscompilação / resposta errada silenciosa (25)

| # | título | aberta em | disp. | dono | estado | prova / PR e CI no head |
|---|---|---|---|---|---|---|
| [#846](https://github.com/Sounio-lang/sounio/issues/846) | Systemic: ~40 hand-rolled exp helpers share the #841 catastrophic-cancellation bug (no shared mat… | 2026-07-13 | a | S | PR aberto | [#2926](https://github.com/Sounio-lang/sounio/pull/2926) `67f6b4744` (fecha) CI verde |
| [#878](https://github.com/Sounio-lang/sounio/issues/878) | SOIR decoder: unknown opcode wire tags silently deserialize as IrNop | 2026-07-14 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/878#issuecomment-6134122376) |
| [#1161](https://github.com/Sounio-lang/sounio/issues/1161) | SOIR v4: writer silently drops current IR semantics and callers accept refusal as success | 2026-07-18 | a | C | PR aberto | [#2942](https://github.com/Sounio-lang/sounio/pull/2942) `82123f052` (fecha) CI verde<br>[#2936](https://github.com/Sounio-lang/sounio/pull/2936) `3557dd07c` (parcial) CI verde |
| [#1542](https://github.com/Sounio-lang/sounio/issues/1542) | [compiler] closure_arity_2 regressed from LEAN-ONLY to compiles-but-prints-nothing after #1541 | 2026-07-28 | a | M | PR aberto | [#2864](https://github.com/Sounio-lang/sounio/pull/2864) `ae2f4d9fe` (fecha) CI verde |
| [#1568](https://github.com/Sounio-lang/sounio/issues/1568) | Madaros: fully-qualified call without a use declaration compiles but silently misbehaves (no-op, … | 2026-07-30 | a | M | PR aberto | [#2844](https://github.com/Sounio-lang/sounio/pull/2844) `dbcd9ce4f` (fecha) CI verde |
| [#1577](https://github.com/Sounio-lang/sounio/issues/1577) | Madaros: three miscompile/limit patterns blocking long 0-D chemistry integrations (rc=182 wall, R… | 2026-07-31 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/1577#issuecomment-6134117498) |
| [#1610](https://github.com/Sounio-lang/sounio/issues/1610) | lean_single accepts f128/f256 and silently computes in f64 — Madaros refuses the same program wit… | 2026-08-03 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/1610#issuecomment-6134118269) |
| [#1702](https://github.com/Sounio-lang/sounio/issues/1702) | Unlanded from the July SOIR stack: explicit-ref field projection fix (#889) and heap-bridge gate … | 2026-08-10 | b | M | fechada (b parte 1, d parte 2) | [comentário](https://github.com/Sounio-lang/sounio/issues/1702#issuecomment-6151862575) |
| [#2218](https://github.com/Sounio-lang/sounio/issues/2218) | Madaros: a tuple-returned array's elements lose their f64 typing; the tuple's scalar element does… | 2026-08-28 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2218#issuecomment-6134117902) |
| [#2249](https://github.com/Sounio-lang/sounio/issues/2249) | Madaros silently computes x ! y and x ~ y as x + y; a run-pass witness added in #2247 pins it as … | 2026-08-28 | a | M | PR aberto | [#2921](https://github.com/Sounio-lang/sounio/pull/2921) `37fa15852` (fecha) CI verde |
| [#2355](https://github.com/Sounio-lang/sounio/issues/2355) | Madaros runtime: mutable global struct field write/read miscompiles (SEGV), print not involved | 2026-09-01 | a | M | PR aberto | [#2856](https://github.com/Sounio-lang/sounio/pull/2856) `05e4870e4` (fecha) CI verde |
| [#2377](https://github.com/Sounio-lang/sounio/issues/2377) | match on module-level const patterns is broken differently in each engine (Madaros binds, lean_si… | 2026-09-01 | a | L1 | PR aberto | [#2863](https://github.com/Sounio-lang/sounio/pull/2863) `fcd2f3e0f` (fecha) CI verde |
| [#2617](https://github.com/Sounio-lang/sounio/issues/2617) | Miscompilation: expr_mul(dl, r) evaluates wrong when dl/r come from recursive calls over a global… | 2026-09-22 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2617#issuecomment-6134114472) |
| [#2619](https://github.com/Sounio-lang/sounio/issues/2619) | Miscompilation: dead if-branch corrupts later use of a var (site_screening_selftest.sio / ss_exp … | 2026-09-22 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2619#issuecomment-6134114907) |
| [#2620](https://github.com/Sounio-lang/sounio/issues/2620) | Miscompilation: bf_exp(x, sc) returns 2^k too-large result when x/sc live in a called function (t… | 2026-09-22 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2620#issuecomment-6134115389) |
| [#2621](https://github.com/Sounio-lang/sounio/issues/2621) | Miscompilation: whole-struct assignment to a global var loses some array-field elements (linalg_f… | 2026-09-22 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2621#issuecomment-6134115934) |
| [#2658](https://github.com/Sounio-lang/sounio/issues/2658) | Miscompilation: &t.0 on a one-field tuple projection emitted a null pointer (lean_single tuple_bo… | 2026-09-23 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2658#issuecomment-6134116512) |
| [#2718](https://github.com/Sounio-lang/sounio/issues/2718) | Venlafaxine parity reference and website JS engine still fabricate ODV mass; the PBPK28 parity ga… | 2026-09-26 | a | S | PR aberto | [#2914](https://github.com/Sounio-lang/sounio/pull/2914) `fc5bbccf6` (fecha) CI verde |
| [#2766](https://github.com/Sounio-lang/sounio/issues/2766) | Madaros: destructuring or matching a linear struct gives false E039/E040 (correct single use reje… | 2026-10-05 | a | M | PR aberto | [#2855](https://github.com/Sounio-lang/sounio/pull/2855) `779d0d6b8` (fecha) CI verde |
| [#2769](https://github.com/Sounio-lang/sounio/issues/2769) | Madaros reads subnormal f64 literals as 0.0 (lean_single keeps them) | 2026-10-06 | — | M | fechada (pelo mantenedor ou por merge) |  |
| [#2773](https://github.com/Sounio-lang/sounio/issues/2773) | Silent wrong answer on both engines: a module's call to its own function resolves to a same-named… | 2026-10-06 | a | L1 | PR aberto | [#2784](https://github.com/Sounio-lang/sounio/pull/2784) `52c19d0ad` (fecha) CI verde |
| [#2816](https://github.com/Sounio-lang/sounio/issues/2816) | Dissertation steady-state demos label whole-blood AUC as plasma AUC (rb_ratio not applied) | 2026-10-06 | a | S | PR aberto | [#2916](https://github.com/Sounio-lang/sounio/pull/2916) `f57a53daa` (fecha) CI verde |
| [#2828](https://github.com/Sounio-lang/sounio/issues/2828) | Madaros: struct de usuário chamado Box sombreia o builtin em silêncio e o ELF morre com SIGSEGV | 2026-10-06 | a | C2 | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2828#issuecomment-6134119996) |
| [#2830](https://github.com/Sounio-lang/sounio/issues/2830) | u64: `>>` sobre parâmetro de função é aritmético e `>` é com sinal (Madaros e lean_single; regres… | 2026-10-06 | a | M | PR aberto | [#2897](https://github.com/Sounio-lang/sounio/pull/2897) `28aa69a28` (fecha) CI verde |
| [#2832](https://github.com/Sounio-lang/sounio/issues/2832) | lean_single: read_file trunca em silêncio em 16 MiB | 2026-10-06 | a | L2 | PR aberto | [#2899](https://github.com/Sounio-lang/sounio/pull/2899) `2257ab845` (fecha) CI verde |

## Divergência entre engines (20)

| # | título | aberta em | disp. | dono | estado | prova / PR e CI no head |
|---|---|---|---|---|---|---|
| [#645](https://github.com/Sounio-lang/sounio/issues/645) | souc: REGRESSION — data-carrying enum variants rejected (E200 'undefined variable' on field name) | 2026-07-05 | decisão | B | fechada → decisão D-17 | [comentário](https://github.com/Sounio-lang/sounio/issues/645#issuecomment-6151738643) |
| [#830](https://github.com/Sounio-lang/sounio/issues/830) | Madaros: honor contextual f32 typing for decimal literals | 2026-07-12 | a | M | PR aberto | [#2843](https://github.com/Sounio-lang/sounio/pull/2843) `e745e5288` (fecha) CI verde |
| [#842](https://github.com/Sounio-lang/sounio/issues/842) | pub use re-export does not forward symbols (mod.sio facade unusable under lean_single) | 2026-07-13 | a | L1 | PR aberto | [#2848](https://github.com/Sounio-lang/sounio/pull/2848) `a582f6638` (fecha) CI verde |
| [#1472](https://github.com/Sounio-lang/sounio/issues/1472) | [parser] two divergent keyword tables: lexer/tables.sio vs an inlined keyword_lookup in parser/pa… | 2026-07-26 | a | D | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/1472#issuecomment-6134118776) |
| [#1494](https://github.com/Sounio-lang/sounio/issues/1494) | [compiler] typecheck errors in imported modules are non-fatal and still emit code | 2026-07-26 | decisão | L1 | fechada → decisão D-20 | [comentário](https://github.com/Sounio-lang/sounio/issues/1494#issuecomment-6151739991) |
| [#1495](https://github.com/Sounio-lang/sounio/issues/1495) | [seed] two lean_single defects: assignment place with two field hops before an index, and named c… | 2026-07-26 | a | L1 | PR aberto | [#2851](https://github.com/Sounio-lang/sounio/pull/2851) `c440736e5` (fecha) CI verde |
| [#1581](https://github.com/Sounio-lang/sounio/issues/1581) | Madaros: E038 rejects &y + &!y same-call aliasing that lean_single accepts (engine divergence on … | 2026-07-31 | b | D | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/1581#issuecomment-6151816383) |
| [#1582](https://github.com/Sounio-lang/sounio/issues/1582) | lean_single: two simultaneously-held tuple-array returns alias each other | 2026-07-31 | decisão | — | fechada → decisão D-18 | [comentário](https://github.com/Sounio-lang/sounio/issues/1582#issuecomment-6151739326) |
| [#1644](https://github.com/Sounio-lang/sounio/issues/1644) | lean_single carries its own pre-#1626 decimal->binary64 conversion | 2026-08-04 | a | L2 | PR aberto | [#2904](https://github.com/Sounio-lang/sounio/pull/2904) `c17165a53` (fecha) CI vermelho |
| [#2354](https://github.com/Sounio-lang/sounio/issues/2354) | Madaros print dispatch: scalar kind lost through inferred let of indexed/field/cast expressions (… | 2026-09-01 | a | C | PR aberto | [#2938](https://github.com/Sounio-lang/sounio/pull/2938) `f3a0b831a` (fecha) CI verde<br>[#2908](https://github.com/Sounio-lang/sounio/pull/2908) `e320d97ac` (parcial) CI verde |
| [#2502](https://github.com/Sounio-lang/sounio/issues/2502) | Madaros: user-declared fn named like an intrinsic (measure) is shadowed by the intrinsic (E008) | 2026-09-14 | b | C2 | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2502#issuecomment-6151752319) |
| [#2604](https://github.com/Sounio-lang/sounio/issues/2604) | Madaros cannot type-check its own compiler source (E035/E012) and lacks .await (async_spawn.sio) | 2026-09-22 | b | M | fechada (b) + decisão D-32 | [comentário](https://github.com/Sounio-lang/sounio/issues/2604#issuecomment-6151847524) |
| [#2787](https://github.com/Sounio-lang/sounio/issues/2787) | Madaros: match on scalar Option mis-dispatches (leading None arm matches everything; Some(0) is t… | 2026-10-06 | a | C1 | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2787#issuecomment-6134116996) |
| [#2789](https://github.com/Sounio-lang/sounio/issues/2789) | Madaros computes wrong values for the sedenion SSM seizure examples (lean_single passes) | 2026-10-06 | a | S | PR aberto | [#2932](https://github.com/Sounio-lang/sounio/pull/2932) `b6fd1cfdb` (fecha) CI verde |
| [#2790](https://github.com/Sounio-lang/sounio/issues/2790) | E-KAN regression benchmarks print different numbers on Madaros and lean_single from epoch 0 | 2026-10-06 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2790#issuecomment-6134121264) |
| [#2798](https://github.com/Sounio-lang/sounio/issues/2798) | lean_single mistypes the middle element of an imported (f64, i64, i64) tuple return as f64 | 2026-10-06 | a | L2 | PR aberto | [#2911](https://github.com/Sounio-lang/sounio/pull/2911) `c27b895c4` (fecha) CI verde |
| [#2825](https://github.com/Sounio-lang/sounio/issues/2825) | lean_single: xorshift64* em u64 imprime valor errado (Madaros correto) | 2026-10-06 | c | A | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2830#issuecomment-6134138964) |
| [#2826](https://github.com/Sounio-lang/sounio/issues/2826) | lean_single: o literal 1.0e-300 é arredondado 13 ulps fora (Madaros correto) | 2026-10-06 | a | L2 | PR aberto | [#2904](https://github.com/Sounio-lang/sounio/pull/2904) `c17165a53` (fecha) CI vermelho |
| [#2831](https://github.com/Sounio-lang/sounio/issues/2831) | lean_single resolve `use` relativo ao CWD; Madaros, ao arquivo de entrada | 2026-10-06 | a | L2 | PR aberto | [#2922](https://github.com/Sounio-lang/sounio/pull/2922) `70559bb09` (fecha) CI verde |
| [#2833](https://github.com/Sounio-lang/sounio/issues/2833) | Madaros aceita a palavra-chave `var` como nome de campo de struct; lean_single rejeita | 2026-10-06 | decisão | — | fechada → decisão D-16 | [comentário](https://github.com/Sounio-lang/sounio/issues/2833#issuecomment-6151737825) |

## Epistêmico / GUM / variância (6)

| # | título | aberta em | disp. | dono | estado | prova / PR e CI no head |
|---|---|---|---|---|---|---|
| [#1535](https://github.com/Sounio-lang/sounio/issues/1535) | [epistemic] lean_single divides correlated variance by the number of uses; both engines lose it i… | 2026-07-27 | c | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2776#issuecomment-6134139531) |
| [#1706](https://github.com/Sounio-lang/sounio/issues/1706) | 14 madaros_gum_fo_* tests assert a PASS marker neither engine produces (one segfaults) | 2026-08-10 | decisão | E | aberta (decisão D-23, a fechar com o link) | [#2945](https://github.com/Sounio-lang/sounio/pull/2945) `00bbfae62` (parcial) CI verde<br>[#2933](https://github.com/Sounio-lang/sounio/pull/2933) `f2a9b6cad` (parcial) CI verde |
| [#2504](https://github.com/Sounio-lang/sounio/issues/2504) | Madaros: Knowledge<i64> from measure(42, uncertainty: ...) has .value != 42 with no diagnostic (f… | 2026-09-14 | a | M | PR aberto | [#2860](https://github.com/Sounio-lang/sounio/pull/2860) `c4d52b99d` (fecha) CI verde |
| [#2776](https://github.com/Sounio-lang/sounio/issues/2776) | lean_single loses first-order variance through struct fields, constructor lets, field-returning c… | 2026-10-06 | a | E | PR aberto | [#2941](https://github.com/Sounio-lang/sounio/pull/2941) `0b579dc48` (fecha) CI verde<br>[#2917](https://github.com/Sounio-lang/sounio/pull/2917) `340a7f902` (parcial) CI verde |
| [#2778](https://github.com/Sounio-lang/sounio/issues/2778) | madaros_wide_struct_variance_field: variance through a wide-struct field is 0 on both shipped eng… | 2026-10-06 | c | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2776#issuecomment-6134139531) |
| [#2827](https://github.com/Sounio-lang/sounio/issues/2827) | lean_single: variância GUM dentro de UMA função sai 0; Madaros propaga (e o inverso atravessando … | 2026-10-06 | a | E | PR aberto | [#2941](https://github.com/Sounio-lang/sounio/pull/2941) `0b579dc48` (parcial) CI verde<br>[#2940](https://github.com/Sounio-lang/sounio/pull/2940) `89b291244` (fecha) CI verde<br>[#2909](https://github.com/Sounio-lang/sounio/pull/2909) `04b0a4a26` (parcial) CI verde |

## Units (3)

| # | título | aberta em | disp. | dono | estado | prova / PR e CI no head |
|---|---|---|---|---|---|---|
| [#2388](https://github.com/Sounio-lang/sounio/issues/2388) | Dispatch: derived unit types do not parse, and units are lost through inference and at call bound… | 2026-09-02 | a | M | PR aberto | [#2907](https://github.com/Sounio-lang/sounio/pull/2907) `ad40615b9` (fecha) CI verde |
| [#2751](https://github.com/Sounio-lang/sounio/issues/2751) | Units: m·s accepted into a let annotated m/s (lean_single soundness hole) | 2026-10-04 | a | B | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2751#issuecomment-6134120450) |
| [#2752](https://github.com/Sounio-lang/sounio/issues/2752) | Madaros: unit-annotated let from a float literal fails with E001 | 2026-10-04 | a | D | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2752#issuecomment-6134120800) |

## Efeitos e checker (5)

| # | título | aberta em | disp. | dono | estado | prova / PR e CI no head |
|---|---|---|---|---|---|---|
| [#888](https://github.com/Sounio-lang/sounio/issues/888) | checker: reject field stores through shared references | 2026-07-14 | a | L1 | PR aberto | [#2854](https://github.com/Sounio-lang/sounio/pull/2854) `340c985bd` (fecha) CI verde |
| [#1157](https://github.com/Sounio-lang/sounio/issues/1157) | parser AST helpers omit Mut and block lean_single strict composite checking | 2026-07-18 | c | L1 | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2834#issuecomment-6134140682) |
| [#1263](https://github.com/Sounio-lang/sounio/issues/1263) | [compiler] Enforce affine at-most-once semantics in Madaros | 2026-07-20 | a | M | PR aberto | [#2852](https://github.com/Sounio-lang/sounio/pull/2852) `e76a81fc5` (fecha) CI verde |
| [#2760](https://github.com/Sounio-lang/sounio/issues/2760) | Madaros: calling builtin println without 'with IO' passes souc check (lean_single rejects, E035) | 2026-10-05 | a | M | PR aberto | [#2762](https://github.com/Sounio-lang/sounio/pull/2762) `ee15f04a5` (fecha) rascunho CI verde |
| [#2834](https://github.com/Sounio-lang/sounio/issues/2834) | Madaros não impõe os efeitos IO (read_file, print) nem Mut (var reatribuída em helper); lean_sing… | 2026-10-06 | aberta | M | aberta (PR parcial) | [#2845](https://github.com/Sounio-lang/sounio/pull/2845) `f0169de11` (parcial) CI verde<br>[#2762](https://github.com/Sounio-lang/sounio/pull/2762) `ee15f04a5` (parcial) rascunho CI verde |

## Native / codegen / reloc (13)

| # | título | aberta em | disp. | dono | estado | prova / PR e CI no head |
|---|---|---|---|---|---|---|
| [#852](https://github.com/Sounio-lang/sounio/issues/852) | native codegen: silent crash when a function makes a nested call with several [256] arrays live (… | 2026-07-13 | d | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/852#issuecomment-6134125629) |
| [#874](https://github.com/Sounio-lang/sounio/issues/874) | native-v2: evolve heap ABI jointly for 16-byte alignment and heap_realloc | 2026-07-14 | a | N | PR aberto | [#2920](https://github.com/Sounio-lang/sounio/pull/2920) `62bd13ae3` (fecha) CI verde |
| [#1156](https://github.com/Sounio-lang/sounio/issues/1156) | Madaros: large single-module parser composite SIGSEGVs at lower_array: seed_begin | 2026-07-18 | aberta | N | aberta |  |
| [#1491](https://github.com/Sounio-lang/sounio/issues/1491) | [runtime] handles are never reclaimed; the machinery exists but stack maps carry counts, not roots | 2026-07-26 | decisão | (decision) | fechada → decisão D-30 | [comentário](https://github.com/Sounio-lang/sounio/issues/1491#issuecomment-6151742363) |
| [#1646](https://github.com/Sounio-lang/sounio/issues/1646) | Madaros: three more silent capacity walls in the native backend (driver labels, return_jumps, flo… | 2026-08-04 | a | N | PR aberto | [#2928](https://github.com/Sounio-lang/sounio/pull/2928) `6329d0bfd` (fecha) CI verde |
| [#1693](https://github.com/Sounio-lang/sounio/issues/1693) | Layout-sensitive segfault in opt_cleanup: adding dead code fixes it, fixing the seed does not | 2026-08-09 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/1693#issuecomment-6134126050) |
| [#2368](https://github.com/Sounio-lang/sounio/issues/2368) | There are TWO -O pipelines, and every optimiser measurement so far exercised only one of them | 2026-09-01 | decisão | — | fechada → decisão D-13 | [comentário](https://github.com/Sounio-lang/sounio/issues/2368#issuecomment-6151735396) |
| [#2371](https://github.com/Sounio-lang/sounio/issues/2371) | Reachability census: ~314 non-test modules are not reachable from any entry point, including 3 IR… | 2026-09-01 | decisão | — | fechada → decisão D-14 | [comentário](https://github.com/Sounio-lang/sounio/issues/2371#issuecomment-6151736248) |
| [#2421](https://github.com/Sounio-lang/sounio/issues/2421) | [Bug]: lean_single SIGSEGVs on both targets at 1025 globals instead of refusing | 2026-09-06 | a | L2 | PR aberto — reaberta: o bot de stale a fechou em 2026-10-13 12:06 UTC por inatividade | [#2913](https://github.com/Sounio-lang/sounio/pull/2913) `489884004` (fecha) CI verde |
| [#2661](https://github.com/Sounio-lang/sounio/issues/2661) | parser: give the impl-method exclusion its own attribution knob (SOUNIO_DISABLE_GLOBAL_INIT_IMPL_… | 2026-09-23 | a | C | PR aberto | [#2943](https://github.com/Sounio-lang/sounio/pull/2943) `abe40df70` (fecha) CI verde<br>[#2934](https://github.com/Sounio-lang/sounio/pull/2934) `1cce450b7` (parcial) CI verde |
| [#2684](https://github.com/Sounio-lang/sounio/issues/2684) | ir_arena_store segfaults on Box-d arg list from standalone program | 2026-09-25 | a | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2684#issuecomment-6134121826) |
| [#2813](https://github.com/Sounio-lang/sounio/issues/2813) | Madaros: programs that construct the native compiler state segfault in compiler_new() | 2026-10-06 | a | N | PR aberto | [#2918](https://github.com/Sounio-lang/sounio/pull/2918) `c92e629c8` (fecha) CI verde |
| [#2814](https://github.com/Sounio-lang/sounio/issues/2814) | Madaros: ossm_hessian_newton and ossm_hessian_lm_mandelbrot die with 'handles full' (rc=182); fix… | 2026-10-06 | a | N | PR aberto | [#2895](https://github.com/Sounio-lang/sounio/pull/2895) `4b4a91ef0` (fecha) CI verde |

## Bootstrap / fixed point (6)

| # | título | aberta em | disp. | dono | estado | prova / PR e CI no head |
|---|---|---|---|---|---|---|
| [#725](https://github.com/Sounio-lang/sounio/issues/725) | Bootstrap integrity: stage0→boot4 C chain no longer reproduces the current fixed point (seed is a… | 2026-07-10 | decisão | — | fechada → decisão D-10 | [comentário](https://github.com/Sounio-lang/sounio/issues/725#issuecomment-6151732796) |
| [#821](https://github.com/Sounio-lang/sounio/issues/821) | macOS ARM64: refresh and promote a source-bootstrap seed with provenance receipt | 2026-07-12 | decisão | — | fechada → decisão D-29 | [comentário](https://github.com/Sounio-lang/sounio/issues/821#issuecomment-6151741499) |
| [#1655](https://github.com/Sounio-lang/sounio/issues/1655) | lean_single: storing a struct into a GLOBAL array is a silent no-op (blocks the #1649 arena) | 2026-08-05 | a | L1 | PR aberto | [#2847](https://github.com/Sounio-lang/sounio/pull/2847) `764c4e369` (fecha) CI verde |
| [#2190](https://github.com/Sounio-lang/sounio/issues/2190) | lean_single: [StructLit; N] repeat-init of a struct-element array segfaults on access (no borrow)… | 2026-08-26 | a | L1 | PR aberto | [#2847](https://github.com/Sounio-lang/sounio/pull/2847) `764c4e369` (fecha) CI verde |
| [#2363](https://github.com/Sounio-lang/sounio/issues/2363) | lean_single.sio: the two duplicate definitions differ by a PRIVACY CHECK, and the copy that wins … | 2026-09-01 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2363#issuecomment-6134119623) |
| [#2836](https://github.com/Sounio-lang/sounio/issues/2836) | Madaros self-compile gate red on main | 2026-10-07 | — | H | fechada (pelo mantenedor ou por merge) |  |

## CI / gates / timeouts (17)

| # | título | aberta em | disp. | dono | estado | prova / PR e CI no head |
|---|---|---|---|---|---|---|
| [#1240](https://github.com/Sounio-lang/sounio/issues/1240) | [docs] repair stale routes and validation date in governance metadata generator | 2026-07-19 | a | S | PR aberto | [#2902](https://github.com/Sounio-lang/sounio/pull/2902) `18a17c99e` (fecha) CI verde |
| [#1680](https://github.com/Sounio-lang/sounio/issues/1680) | madaros --self-test segfaults at test 24 of 1162 — 97.7% of the compiler's own suite has never ru… | 2026-08-06 | a | #2937) | PR aberto | [#2953](https://github.com/Sounio-lang/sounio/pull/2953) `a54fc3e9f` (fecha) CI verde<br>[#2952](https://github.com/Sounio-lang/sounio/pull/2952) `d88e843e8` (parcial) CI verde<br>[#2951](https://github.com/Sounio-lang/sounio/pull/2951) `da6c0f826` (parcial) CI verde<br>[#2950](https://github.com/Sounio-lang/sounio/pull/2950) `587f15ca2` (parcial) CI verde<br>[#2949](https://github.com/Sounio-lang/sounio/pull/2949) `3297e4fcb` (fecha) CI verde<br>[#2948](https://github.com/Sounio-lang/sounio/pull/2948) `e373c7122` (parcial) CI verde<br>[#2937](https://github.com/Sounio-lang/sounio/pull/2937) `42f78d84c` (parcial) CI verde |
| [#2219](https://github.com/Sounio-lang/sounio/issues/2219) | scripts/ci/souc-native-wrapper.sh: raw-mode-detection typo + missing stack ulimit break Full Test… | 2026-08-28 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2219#issuecomment-6134123492) |
| [#2240](https://github.com/Sounio-lang/sounio/issues/2240) | Madaros Current-Source f64 Lowering job's 35-minute timeout is too tight for large merges (800+ c… | 2026-08-28 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2240#issuecomment-6134124020) |
| [#2241](https://github.com/Sounio-lang/sounio/issues/2241) | The re-keyed compile-fail assertions are lean_single-specific; pointing the suite at Madaros woul… | 2026-08-28 | decisão | — | fechada → decisão D-11 | [comentário](https://github.com/Sounio-lang/sounio/issues/2241#issuecomment-6151733573) |
| [#2306](https://github.com/Sounio-lang/sounio/issues/2306) | The corpus regression gate reports ~380 failures against every compiler, and 364 of them are real… | 2026-08-30 | decisão | — | fechada → decisão D-12 | [comentário](https://github.com/Sounio-lang/sounio/issues/2306#issuecomment-6151734421) |
| [#2375](https://github.com/Sounio-lang/sounio/issues/2375) | PR #1301's headline benchmark credits an engine that never ran | 2026-09-01 | a | S | PR aberto | [#2901](https://github.com/Sounio-lang/sounio/pull/2901) `47f1dafa2` (fecha) CI verde |
| [#2422](https://github.com/Sounio-lang/sounio/issues/2422) | [Bug]: aarch64-linux compile does not terminate where x86 refuses with E200 | 2026-09-06 | a | a64) | PR aberto — reaberta: o bot de stale a fechou em 2026-10-13 12:06 UTC por inatividade | [#2930](https://github.com/Sounio-lang/sounio/pull/2930) `715132aef` (fecha) CI verde |
| [#2424](https://github.com/Sounio-lang/sounio/issues/2424) | [Bug]: four compile-fail fixtures are refused with no diagnostic, so they assert nothing | 2026-09-06 | a | S | PR aberto — reaberta: o bot de stale a fechou em 2026-10-13 12:06 UTC por inatividade | [#2900](https://github.com/Sounio-lang/sounio/pull/2900) `e65de3850` (fecha) CI verde |
| [#2425](https://github.com/Sounio-lang/sounio/issues/2425) | [Bug]: 16 of 18 refusal gates still read rc != 0 as a semantic refusal | 2026-09-06 | aberta | S | aberta (PR parcial) | [#2925](https://github.com/Sounio-lang/sounio/pull/2925) `7be0de9cf` (parcial) CI verde |
| [#2428](https://github.com/Sounio-lang/sounio/issues/2428) | [Bug]: f64 imported-capacity gate is timing-marginal against a hardcoded 300s compile cap | 2026-09-06 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2428#issuecomment-6134124439) |
| [#2429](https://github.com/Sounio-lang/sounio/issues/2429) | [Bug]: aarch64 never reaches the E067 confounding check; E035 masks it | 2026-09-06 | a | a64) | PR aberto | [#2930](https://github.com/Sounio-lang/sounio/pull/2930) `715132aef` (fecha) CI verde |
| [#2431](https://github.com/Sounio-lang/sounio/issues/2431) | [Bug]: Release Gate red for 8 runs — lean.sio cannot cross-build for aarch64 (x86/_a64 mirrors di… | 2026-09-06 | — | — | fechada (pelo mantenedor ou por merge) |  |
| [#2443](https://github.com/Sounio-lang/sounio/issues/2443) | [Bug]: the Contracts job runs 109 steps with no timeout-minutes, so any one of them can stall for… | 2026-09-07 | a | F | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2443#issuecomment-6134124816) |
| [#2475](https://github.com/Sounio-lang/sounio/issues/2475) | [CI] f64 Lowering job reclaimed by runner at ~55min (gen2 self-compile ~56min) — systemic flake f… | 2026-09-11 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2475#issuecomment-6134125199) |
| [#2507](https://github.com/Sounio-lang/sounio/issues/2507) | epistemic_hessian_transcendentals witness refused with E221 since #2172; Hessian chain rule for t… | 2026-09-14 | a | E | PR aberto | [#2919](https://github.com/Sounio-lang/sounio/pull/2919) `f5614ce5e` (parcial) CI verde<br>[#2608](https://github.com/Sounio-lang/sounio/pull/2608) `cfa5d2fbf` (fecha) CI verde |
| [#2592](https://github.com/Sounio-lang/sounio/issues/2592) | Full Test Suite red on main since 2026-09-19: 18 tests fail (12 timeouts, 6 non-zero incl. a segf… | 2026-09-21 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/2592#issuecomment-6134122910) |

## FFI (6)

| # | título | aberta em | disp. | dono | estado | prova / PR e CI no head |
|---|---|---|---|---|---|---|
| [#2533](https://github.com/Sounio-lang/sounio/issues/2533) | [ffi] native_v2_dynlink_unique_lib_count saturates at 4; writer overflow check is dead code | 2026-09-16 | a | N | PR aberto | [#2910](https://github.com/Sounio-lang/sounio/pull/2910) `1d9f4597f` (fecha) CI em execução |
| [#2534](https://github.com/Sounio-lang/sounio/issues/2534) | [ffi] dynstr fixed at 256 bytes — needs bounds check or dynamic sizing before KL-14d2 | 2026-09-16 | a | N | PR aberto | [#2910](https://github.com/Sounio-lang/sounio/pull/2910) `1d9f4597f` (fecha) CI em execução |
| [#2535](https://github.com/Sounio-lang/sounio/issues/2535) | [design] Promote LibraryId from hard-coded table to first-class FFI entity (ExternSymbol) | 2026-09-16 | decisão | — | fechada → decisão D-04 | [comentário](https://github.com/Sounio-lang/sounio/issues/2535#issuecomment-6151727723) |
| [#2536](https://github.com/Sounio-lang/sounio/issues/2536) | [design] Decide symbol-interposition semantics for LibraryId (provenance vs binding ownership) | 2026-09-16 | decisão | — | fechada → decisão D-05 | [comentário](https://github.com/Sounio-lang/sounio/issues/2536#issuecomment-6151728432) |
| [#2537](https://github.com/Sounio-lang/sounio/issues/2537) | [test] Missing witness for SysV callee alignment in dynlink thunk (post-41fae115e) | 2026-09-16 | — | N | fechada (pelo mantenedor ou por merge) |  |
| [#2829](https://github.com/Sounio-lang/sounio/issues/2829) | lean_single: depois de print_int(floor(7.9) as i64) o programa para em silêncio | 2026-10-06 | a | L2 | PR aberto | [#2898](https://github.com/Sounio-lang/sounio/pull/2898) `98b498b9a` (fecha) CI verde |

## Stdlib e exemplos (4)

| # | título | aberta em | disp. | dono | estado | prova / PR e CI no head |
|---|---|---|---|---|---|---|
| [#1489](https://github.com/Sounio-lang/sounio/issues/1489) | [examples] sedenion Hessian demo hangs printing its own 1.0e30 sentinel when all eigenvalues fall… | 2026-07-26 | a | S | PR aberto | [#2906](https://github.com/Sounio-lang/sounio/pull/2906) `e44d3d016` (fecha) CI verde |
| [#1584](https://github.com/Sounio-lang/sounio/issues/1584) | Madaros: visibility preflight fails on any chemistry::kinetics import (blocks all chemistry demos… | 2026-07-31 | b | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/1584#issuecomment-6134119276) |
| [#2779](https://github.com/Sounio-lang/sounio/issues/2779) | examples/showcase/effect_test_harness.sio cannot run: Madaros parse failure (E-SRB-000), lean_sin… | 2026-10-06 | aberta | S | aberta (PR parcial) | [#2849](https://github.com/Sounio-lang/sounio/pull/2849) `b29ece3f6` (parcial) CI verde<br>[#2846](https://github.com/Sounio-lang/sounio/pull/2846) `6ad2eea2e` (parcial) CI verde |
| [#2820](https://github.com/Sounio-lang/sounio/issues/2820) | Pireus examples/tests: remaining Madaros runtime failures (arena full, miscompile, crashes) and e… | 2026-10-06 | decisão | C | aberta (decisão D-26, a fechar com o link) | [#2947](https://github.com/Sounio-lang/sounio/pull/2947) `a51c3f295` (parcial) CI verde<br>[#2946](https://github.com/Sounio-lang/sounio/pull/2946) `f5282ed30` (parcial) CI verde<br>[#2944](https://github.com/Sounio-lang/sounio/pull/2944) `f905daa37` (parcial) CI verde<br>[#2935](https://github.com/Sounio-lang/sounio/pull/2935) `1e13b735c` (parcial) CI verde |

## Research / design (12)

| # | título | aberta em | disp. | dono | estado | prova / PR e CI no head |
|---|---|---|---|---|---|---|
| [#839](https://github.com/Sounio-lang/sounio/issues/839) | EISA-H: establish hardware repository and exact sedenion ZD detector | 2026-07-12 | decisão | — | fechada → decisão D-09 | [comentário](https://github.com/Sounio-lang/sounio/issues/839#issuecomment-6151731876) |
| [#1122](https://github.com/Sounio-lang/sounio/issues/1122) | R3: decide canonical production target repository mapping | 2026-07-17 | decisão | — | fechada → decisão D-06 | [comentário](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-6151729240) |
| [#1162](https://github.com/Sounio-lang/sounio/issues/1162) | SOIR v6: sectioned transactional container for compiler-owned scientific semantics | 2026-07-18 | decisão | — | fechada → decisão D-21 | [comentário](https://github.com/Sounio-lang/sounio/issues/1162#issuecomment-6151740796) |
| [#1386](https://github.com/Sounio-lang/sounio/issues/1386) | feat(language): study opaque authority capabilities beyond private structs | 2026-07-21 | decisão | L2 | fechada → decisão D-02 | [comentário](https://github.com/Sounio-lang/sounio/issues/1386#issuecomment-6151725973) |
| [#1423](https://github.com/Sounio-lang/sounio/issues/1423) | research(epistemic): source-fresh state aliasing and numerical associator contest | 2026-07-24 | decisão | — | fechada → decisão D-07 | [comentário](https://github.com/Sounio-lang/sounio/issues/1423#issuecomment-6151730114) |
| [#1484](https://github.com/Sounio-lang/sounio/issues/1484) | [research] sedenion brain Hessian was an identically-zero false null; still unvalidated after the… | 2026-07-26 | decisão | — | fechada → decisão D-08 | [comentário](https://github.com/Sounio-lang/sounio/issues/1484#issuecomment-6151730963) |
| [#1712](https://github.com/Sounio-lang/sounio/issues/1712) | Foundry/Slurm: run CS6 event-normal TM2R carrier transport | 2026-08-11 | decisão | — | fechada | [prova](https://github.com/Sounio-lang/sounio/issues/1712#issuecomment-6134126430) |
| [#2293](https://github.com/Sounio-lang/sounio/issues/2293) | No spelling for the empty effect set: 'with Pure' is E246, so a pure combinator cannot satisfy th… | 2026-08-30 | decisão | — | fechada → decisão D-15 | [comentário](https://github.com/Sounio-lang/sounio/issues/2293#issuecomment-6151736995) |
| [#2380](https://github.com/Sounio-lang/sounio/issues/2380) | design: type-level basis-support certificate Q so E251 can admit certified sedenion products (cod… | 2026-09-01 | decisão | — | fechada → decisão D-03 | [comentário](https://github.com/Sounio-lang/sounio/issues/2380#issuecomment-6151726827) |
| [#2753](https://github.com/Sounio-lang/sounio/issues/2753) | Knowledge<T where {...}> refinement is silently unenforced by souc run (both engines) | 2026-10-04 | a | M | PR aberto | [#2923](https://github.com/Sounio-lang/sounio/pull/2923) `d76d0060e` (fecha) CI verde |
| [#2761](https://github.com/Sounio-lang/sounio/issues/2761) | formal: TypeCheckerSoundness axioms float_le_refl and float_add_le_add are false for IEEE 754; mo… | 2026-10-05 | decisão | — | fechada → decisão D-01 | [comentário](https://github.com/Sounio-lang/sounio/issues/2761#issuecomment-6151725167) |
| [#2835](https://github.com/Sounio-lang/sounio/issues/2835) | stdlib/heliobiology: coeficientes de effects.sio sem fonte, incerteza fixa de 15% e constants.sio… | 2026-10-06 | a | S | PR aberto | [#2912](https://github.com/Sounio-lang/sounio/pull/2912) `5df8e880b` (fecha) CI verde |

## Encontradas durante a campanha (fora das 117)

| # | título | estado | PR e CI no head |
|---|---|---|---|
| [#2874](https://github.com/Sounio-lang/sounio/issues/2874) | [FIX IMEDIATO] Madaros distribuído: parâmetros com unidades derivadas | PR aberto | [#2939](https://github.com/Sounio-lang/sounio/pull/2939) `09c5e60c6` (fecha) CI verde<br>[#2924](https://github.com/Sounio-lang/sounio/pull/2924) `764ca3929` (parcial) CI verde<br>[#2905](https://github.com/Sounio-lang/sounio/pull/2905) `7b44e27de` (parcial) CI verde |
| [#2881](https://github.com/Sounio-lang/sounio/issues/2881) | Madaros self-compile gate red on main | fechada pelo mantenedor (duplicada por #2896) | — |
| [#2896](https://github.com/Sounio-lang/sounio/issues/2896) | Madaros self-compile gate red on main | PR aberto | [#2954](https://github.com/Sounio-lang/sounio/pull/2954) `a983a97fa` (fecha) CI verde; o rascunho #2886 traz a mesma correção |

## Resumo

- Das 117 issues, **61 estão fechadas** (todas com prova ou com o link da seção de `PENDING.md`; 4 foram fechadas pelo mantenedor/merge) e **56 seguem abertas**; mais 2 encontradas na campanha (#2874, #2896).
- Abertas cujo PR as fecha no merge (`Fixes`): **50**. Abertas com PR só parcial (`Refs`): **5** (#1706, #2425, #2779, #2820, #2834). Abertas sem PR: **1** (#1156).
- **A contagem zero só é alcançável com o merge dos PRs** (a campanha não mescla nada). Hoje as issues restantes dependem de PRs não mesclados ou de trabalho que exige host/memória que a campanha não tinha (ver "Não concluído").
- O bot de stale (`actions/stale`, `days-before-close`) **continua fechando issues por inatividade** em `main` até o PR [#2892](https://github.com/Sounio-lang/sounio/pull/2892) ser mesclado: em 2026-10-13 12:06 UTC fechou #2421, #2422 e #2424, que foram reabertas.

## PRs abertos da campanha e CI no head exato

| PR | head | rascunho | issues | CI | observação |
|---|---|---|---|---|---|
| [#2608](https://github.com/Sounio-lang/sounio/pull/2608) | `cfa5d2fbf` | não | Fixes #2507 | verde (neutral=1,skipped=6,success=24) |  |
| [#2762](https://github.com/Sounio-lang/sounio/pull/2762) | `ee15f04a5` | sim | Fixes #2760, Refs #2834 | verde (skipped=5,success=23) |  |
| [#2784](https://github.com/Sounio-lang/sounio/pull/2784) | `52c19d0ad` | não | Fixes #2773 | verde (neutral=1,skipped=8,success=27) |  |
| [#2843](https://github.com/Sounio-lang/sounio/pull/2843) | `e745e5288` | não | Fixes #830 | verde (neutral=1,skipped=11,success=23) |  |
| [#2844](https://github.com/Sounio-lang/sounio/pull/2844) | `dbcd9ce4f` | não | Fixes #1568 | verde (neutral=1,skipped=9,success=26) |  |
| [#2845](https://github.com/Sounio-lang/sounio/pull/2845) | `f0169de11` | não | Refs #2834 | verde (neutral=1,skipped=8,success=26) |  |
| [#2846](https://github.com/Sounio-lang/sounio/pull/2846) | `6ad2eea2e` | não | Refs #2779 | verde (neutral=1,skipped=9,success=22) |  |
| [#2847](https://github.com/Sounio-lang/sounio/pull/2847) | `764c4e369` | não | Fixes #1655, Fixes #2190 | verde (neutral=1,skipped=8,success=26) |  |
| [#2848](https://github.com/Sounio-lang/sounio/pull/2848) | `a582f6638` | não | Fixes #842 | verde (neutral=1,skipped=6,success=24) |  |
| [#2849](https://github.com/Sounio-lang/sounio/pull/2849) | `b29ece3f6` | não | Refs #2779 | verde (neutral=1,skipped=8,success=26) |  |
| [#2851](https://github.com/Sounio-lang/sounio/pull/2851) | `c440736e5` | não | Fixes #1495 | verde (neutral=1,skipped=8,success=26) |  |
| [#2852](https://github.com/Sounio-lang/sounio/pull/2852) | `e76a81fc5` | não | Fixes #1263 | verde (neutral=1,skipped=9,success=26) |  |
| [#2854](https://github.com/Sounio-lang/sounio/pull/2854) | `340c985bd` | não | Fixes #888 | verde (neutral=1,skipped=8,success=27) |  |
| [#2855](https://github.com/Sounio-lang/sounio/pull/2855) | `779d0d6b8` | não | Fixes #2766 | verde (neutral=1,skipped=11,success=23) |  |
| [#2856](https://github.com/Sounio-lang/sounio/pull/2856) | `05e4870e4` | não | Fixes #2355, Refs #2617 | verde (neutral=1,skipped=12,success=24) |  |
| [#2860](https://github.com/Sounio-lang/sounio/pull/2860) | `c4d52b99d` | não | Fixes #2504 | verde (neutral=1,skipped=11,success=23) |  |
| [#2863](https://github.com/Sounio-lang/sounio/pull/2863) | `fcd2f3e0f` | não | Fixes #2377 | verde (neutral=1,skipped=8,success=26) |  |
| [#2864](https://github.com/Sounio-lang/sounio/pull/2864) | `ae2f4d9fe` | não | Fixes #1542 | verde (neutral=1,skipped=12,success=24) |  |
| [#2886](https://github.com/Sounio-lang/sounio/pull/2886) | `ea39efec8` | sim | Refs #2881 | verde (skipped=6,success=25) |  |
| [#2892](https://github.com/Sounio-lang/sounio/pull/2892) | `52271ac6b` | não |  | verde (neutral=1,skipped=5,success=22) | head 52271ac6b corrige o GATE_VACUITY_GATE_FAIL do head anterior (42e4f553d). Enquanto não for mesclado, o bot de stale continua fechando issues |
| [#2894](https://github.com/Sounio-lang/sounio/pull/2894) | `f872280ba` | não | Refs #2218 | verde (neutral=1,skipped=5,success=22) | testemunhas de defeitos já corrigidos em `main` (#2218, #2658, #2684, #1577) |
| [#2895](https://github.com/Sounio-lang/sounio/pull/2895) | `4b4a91ef0` | não | Fixes #2814 | verde (neutral=1,skipped=5,success=22) |  |
| [#2897](https://github.com/Sounio-lang/sounio/pull/2897) | `28aa69a28` | não | Fixes #2830 | verde (neutral=1,skipped=8,success=25) |  |
| [#2898](https://github.com/Sounio-lang/sounio/pull/2898) | `98b498b9a` | não | Fixes #2829 | verde (neutral=1,skipped=10,success=26) |  |
| [#2899](https://github.com/Sounio-lang/sounio/pull/2899) | `2257ab845` | não | Fixes #2832 | verde (neutral=1,skipped=10,success=26) |  |
| [#2900](https://github.com/Sounio-lang/sounio/pull/2900) | `e65de3850` | não | Fixes #2424 | verde (neutral=1,skipped=5,success=22) |  |
| [#2901](https://github.com/Sounio-lang/sounio/pull/2901) | `47f1dafa2` | não | Fixes #2375 | verde (neutral=1,skipped=5,success=22) |  |
| [#2902](https://github.com/Sounio-lang/sounio/pull/2902) | `18a17c99e` | não | Fixes #1240 | verde (neutral=1,skipped=7,success=25) |  |
| [#2904](https://github.com/Sounio-lang/sounio/pull/2904) | `c17165a53` | não | Fixes #1644, Fixes #2826 | vermelho (failure=1,neutral=1,skipped=8,success=24) | vermelho só em "Chemistry probe golden": os números do paper mudam com o arredondamento correto; regenerar é decisão do autor (PENDING D-27) |
| [#2905](https://github.com/Sounio-lang/sounio/pull/2905) | `7b44e27de` | não | Refs #2874 | verde (neutral=1,skipped=9,success=21) |  |
| [#2906](https://github.com/Sounio-lang/sounio/pull/2906) | `e44d3d016` | não | Fixes #1489 | verde (neutral=1,skipped=5,success=22) |  |
| [#2907](https://github.com/Sounio-lang/sounio/pull/2907) | `ad40615b9` | não | Fixes #2388 | verde (neutral=1,skipped=9,success=21) |  |
| [#2908](https://github.com/Sounio-lang/sounio/pull/2908) | `e320d97ac` | não | Refs #2354 | verde (neutral=1,skipped=7,success=24) |  |
| [#2909](https://github.com/Sounio-lang/sounio/pull/2909) | `04b0a4a26` | não | Refs #2827 | verde (neutral=1,skipped=11,success=22) |  |
| [#2910](https://github.com/Sounio-lang/sounio/pull/2910) | `1d9f4597f` | não | Fixes #2533, Fixes #2534 | em execução (in_progress=1,neutral=1,skipped=5,success=20) |  |
| [#2911](https://github.com/Sounio-lang/sounio/pull/2911) | `c27b895c4` | não | Fixes #2798 | verde (neutral=1,skipped=10,success=26) |  |
| [#2912](https://github.com/Sounio-lang/sounio/pull/2912) | `5df8e880b` | não | Fixes #2835 | verde (neutral=1,skipped=9,success=18) |  |
| [#2913](https://github.com/Sounio-lang/sounio/pull/2913) | `489884004` | não | Fixes #2421 | verde (neutral=1,skipped=10,success=26) |  |
| [#2914](https://github.com/Sounio-lang/sounio/pull/2914) | `fc5bbccf6` | não | Fixes #2718 | verde (neutral=1,skipped=5,success=22) |  |
| [#2916](https://github.com/Sounio-lang/sounio/pull/2916) | `f57a53daa` | não | Fixes #2816 | verde (neutral=1,skipped=9,success=19) |  |
| [#2917](https://github.com/Sounio-lang/sounio/pull/2917) | `340a7f902` | não | Refs #2776 | verde (neutral=1,skipped=8,success=25) |  |
| [#2918](https://github.com/Sounio-lang/sounio/pull/2918) | `c92e629c8` | não | Fixes #2813 | verde (neutral=1,skipped=5,success=22) |  |
| [#2919](https://github.com/Sounio-lang/sounio/pull/2919) | `f5614ce5e` | não | Refs #2507 | verde (neutral=1,skipped=9,success=18) |  |
| [#2920](https://github.com/Sounio-lang/sounio/pull/2920) | `62bd13ae3` | não | Fixes #874 | verde (neutral=1,skipped=6,success=24) |  |
| [#2921](https://github.com/Sounio-lang/sounio/pull/2921) | `37fa15852` | não | Fixes #2249 | verde (neutral=1,skipped=8,success=25) |  |
| [#2922](https://github.com/Sounio-lang/sounio/pull/2922) | `70559bb09` | não | Fixes #2831 | verde (neutral=1,skipped=8,success=25) |  |
| [#2923](https://github.com/Sounio-lang/sounio/pull/2923) | `d76d0060e` | não | Fixes #2753 | verde (neutral=1,skipped=6,success=25) |  |
| [#2924](https://github.com/Sounio-lang/sounio/pull/2924) | `764ca3929` | não | Refs #2874 | verde (neutral=1,skipped=8,success=25) |  |
| [#2925](https://github.com/Sounio-lang/sounio/pull/2925) | `7be0de9cf` | não | Refs #2425 | verde (neutral=1,skipped=9,success=25) |  |
| [#2926](https://github.com/Sounio-lang/sounio/pull/2926) | `67f6b4744` | não | Fixes #846 | verde (neutral=1,skipped=5,success=23) |  |
| [#2927](https://github.com/Sounio-lang/sounio/pull/2927) | `49a0280af` | não | Refs #1386 | verde (neutral=1,skipped=8,success=25) |  |
| [#2928](https://github.com/Sounio-lang/sounio/pull/2928) | `6329d0bfd` | não | Fixes #1646 | verde (neutral=1,skipped=7,success=23) |  |
| [#2930](https://github.com/Sounio-lang/sounio/pull/2930) | `715132aef` | não | Fixes #2422, Fixes #2429 | verde (neutral=1,skipped=6,success=24) |  |
| [#2932](https://github.com/Sounio-lang/sounio/pull/2932) | `b6fd1cfdb` | não | Fixes #2789 | verde (neutral=1,skipped=5,success=22) |  |
| [#2933](https://github.com/Sounio-lang/sounio/pull/2933) | `f2a9b6cad` | não | Refs #1706 | verde (neutral=1,skipped=7,success=21) | head f2a9b6cad corrige o XPAS de `fo_call_boundary_neg` e encolhe o baseline de known-failure (os dois vermelhos do head anterior ddc3538f5) |
| [#2934](https://github.com/Sounio-lang/sounio/pull/2934) | `1cce450b7` | não | Refs #2661 | verde (neutral=1,skipped=8,success=25) |  |
| [#2935](https://github.com/Sounio-lang/sounio/pull/2935) | `1e13b735c` | não | Refs #2820 | verde (neutral=1,skipped=9,success=18) |  |
| [#2936](https://github.com/Sounio-lang/sounio/pull/2936) | `3557dd07c` | não | Refs #1161 | verde (neutral=1,skipped=5,success=22) |  |
| [#2937](https://github.com/Sounio-lang/sounio/pull/2937) | `42f78d84c` | não | Refs #1680 | verde (neutral=1,skipped=8,success=25) | head 42f78d84c carrega o commit de #2847 (`1e2c9ece1`) e um refresh de seed, para ficar verde sozinho; quem mesclar depois refaz o seed |
| [#2938](https://github.com/Sounio-lang/sounio/pull/2938) | `f3a0b831a` | não | Fixes #2354 | verde (neutral=1,skipped=5,success=23) | empilhado sobre #2908 (E270) |
| [#2939](https://github.com/Sounio-lang/sounio/pull/2939) | `09c5e60c6` | não | Fixes #2874 | verde (neutral=1,skipped=6,success=24) |  |
| [#2940](https://github.com/Sounio-lang/sounio/pull/2940) | `89b291244` | não | Fixes #2827 | verde (neutral=1,skipped=7,success=20) |  |
| [#2941](https://github.com/Sounio-lang/sounio/pull/2941) | `0b579dc48` | não | Fixes #2776, Refs #2827 | verde (neutral=1,skipped=8,success=26) |  |
| [#2942](https://github.com/Sounio-lang/sounio/pull/2942) | `82123f052` | não | Fixes #1161 | verde (neutral=1,skipped=5,success=22) |  |
| [#2943](https://github.com/Sounio-lang/sounio/pull/2943) | `abe40df70` | não | Fixes #2661 | verde (neutral=1,skipped=6,success=25) |  |
| [#2944](https://github.com/Sounio-lang/sounio/pull/2944) | `f905daa37` | não | Refs #2820 | verde (neutral=1,skipped=5,success=22) |  |
| [#2945](https://github.com/Sounio-lang/sounio/pull/2945) | `00bbfae62` | não | Refs #1706 | verde (neutral=1,skipped=9,success=22) | empilhado sobre uma cópia rebaseada de #2933; mesclar #2933 primeiro ou este no lugar |
| [#2946](https://github.com/Sounio-lang/sounio/pull/2946) | `f5282ed30` | não | Refs #2820 | verde (neutral=1,skipped=7,success=23) | relata a divergência de empréstimo implícito (PENDING D-26) |
| [#2947](https://github.com/Sounio-lang/sounio/pull/2947) | `a51c3f295` | não | Refs #2820 | verde (neutral=1,skipped=7,success=23) |  |
| [#2948](https://github.com/Sounio-lang/sounio/pull/2948) | `e373c7122` | não | Refs #1680 | verde (neutral=1,skipped=6,success=24) |  |
| [#2949](https://github.com/Sounio-lang/sounio/pull/2949) | `3297e4fcb` | não | Fixes #1680, Refs #1680 | verde (neutral=1,skipped=8,success=25) |  |
| [#2950](https://github.com/Sounio-lang/sounio/pull/2950) | `587f15ca2` | não | Refs #1680 | verde (neutral=1,skipped=6,success=24) |  |
| [#2951](https://github.com/Sounio-lang/sounio/pull/2951) | `da6c0f826` | não | Refs #1680 | verde (neutral=1,skipped=6,success=24) |  |
| [#2952](https://github.com/Sounio-lang/sounio/pull/2952) | `d88e843e8` | não | Refs #1680 | verde (neutral=1,skipped=6,success=24) |  |
| [#2953](https://github.com/Sounio-lang/sounio/pull/2953) | `a54fc3e9f` | não | Fixes #1680 | verde (neutral=1,skipped=8,success=25) |  |
| [#2954](https://github.com/Sounio-lang/sounio/pull/2954) | `a983a97fa` | não | Fixes #2896 | verde (neutral=1,skipped=6,success=23) | fecha #2896; o rascunho #2886 traz a mesma correção (mais um workflow e um check de opcodes) |

## Não concluído

- **Merge.** Nenhum PR foi mesclado (regra da campanha). As 50 issues com `Fixes` e as 5 com PR parcial só fecham quando o mantenedor mesclar; a contagem final de issues abertas é, por isso, a da seção Resumo e não zero.
- **#1156** precisa de um nó com ≥16 GB: o composto grande do parser passa de 4,5 GB de RSS só compilando, e o host da campanha tem 7 GB. Comando: `souc run G4/i1156_driver.sio` com o Madaros de `main`; critério: passar de `lower_array: seed_begin`. O PR #2931 (mesclado, `Refs`) removeu o frame de 671 MB por valor e o SIGSEGV de execução da API de tokens.
- **#1706** continua aberta por causa de `madaros_gum_fo_knowledge_ops`, bloqueado por E245 (PENDING D-23); 13 dos 14 testes passam com os PRs #2933 e #2945.
- **#2820** (guarda-chuva): ficam quatro testes Pireus ainda falhando no Madaros (`quotient`, `lowering`, `orbit`, `novelty_feedback` com arena cheia), `material_engine_admission` que precisa do prebuilt atualizado, exemplos com mais de 1200 s, `pdf_text_projection` com segfault do compilador e `inflate` com segfault no `lean_single`. Os PRs #2944, #2946 e #2947 tratam as classes de miscompile e de material de vendor.
- **#1680**: o crash do `--self-test` está corrigido no PR #2937 (que agora carrega o commit de lean_single de #2847); com ele o `--self-test` roda até o fim (613/1162 testes passam; os demais são defeitos separados que os PRs #2948–#2953 tratam em parte). A issue só fecha com o merge desses PRs.
- **#2425, #2779, #2834** têm PR só parcial: #2925 (16 dos 18 gates de refusal), #2849/#2846 (as duas metades de `effect_test_harness`), #2845 (a metade `Mut`; a metade `IO` duplica #2760, cujo PR #2762 é rascunho).
- **Gate de CI vermelho por motivo estranho à campanha**: PRs antigos do mantenedor (por exemplo #2775, #2724, #2704, #2703, #2686, #2681, #2663, #2660, #2655) já estavam vermelhos antes e não são da campanha.

