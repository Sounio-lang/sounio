<!-- docs:meta
topic_id: repo.docs.audit.divida-zero-v3-bloco0-inventory-2026-09-22
authority: repo_only
audience: users
last_validated: 2026-09-22
validated_by: claude-divida-zero-v3-bloco0
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.divida-zero-v3-bloco0-inventory-2026-09-22
-->

# Dívida Zero v3 — Bloco 0: instrumentos e inventário (2026-09-22)

Natureza: primeira entrega do plano `SOUNIO_DIVIDA_ZERO_PLANO_v3_2026-09-22` (§5, Bloco 0 /
DZ-00, DZ-01). Este documento é instrumento e inventário — **não** é uma correção de
defeito, não fecha M1/M2/M3, e não afirma que a suíte completa foi executada nesta
sessão. Base: commit `69b7fe7546e8` na branch
`claude/sounio-divida-zero-v3-g7svhy`.

**Proveniência do plano citado (apontado em revisão,
`Sounio-lang/sounio#2637` `discussion_r4071109861`):** nem
`SOUNIO_DIVIDA_ZERO_PLANO_v3_2026-09-22` nem seu antecedente
`SOUNIO_DIVIDA_ZERO_PLANO_v2_2026-09-20` estão versionados neste repositório —
confirmado por `find . -iname '*PLANO*' -o -iname '*DIVIDA*'` nesta árvore, que só
retorna este próprio documento. O plano v3 foi fornecido inteiramente como texto na
sessão que gerou este Bloco 0 (não é um arquivo, não tem commit, não tem URL externa
estável). As referências `§N` deste documento apontam para a numeração desse texto de
sessão, não para um artefato reproduzível de dentro do repositório; um leitor sem acesso
a essa sessão não consegue verificar as citações de seção por conta própria. Este achado
**não é corrigido nesta entrega** — commitar o plano inteiro é uma decisão de outra lane
(é um documento de campanha, não um artefato de auditoria técnica, e sua autoria/dono não
foi determinado aqui); a proveniência é apenas tornada explícita para que a limitação de
verificabilidade fique registrada.

## 1. Recalibração de escala (§1 do CLAUDE.md)

`bash scripts/dev/measure_repo_scale.sh` em `69b7fe7546e8`:

| Métrica | Valor medido nesta sessão | Valor no CLAUDE.md (2026-07-11) | Delta |
|---|---:|---:|---:|
| `.sio` rastreados (arquivos) | 8,584 | 6,130 | +2,454 |
| `.sio` rastreados (linhas, soma `git ls-files`) | 3,345,790 | 2,208,306 | +1,137,484 |
| `self-hosted/` (linhas) | 631,508 | 554,892 | +76,616 |
| `stdlib/` (linhas) | 640,015 | 478,355 | +161,660 |
| `tests/` (linhas) | 305,415 | 236,693 | +68,722 |
| `examples/` (linhas) | 158,544 | 130,370 | +28,174 |
| CI gate scripts (`scripts/ci/`) | 596 | — | — |

**Achado:** a tabela de calibração do `CLAUDE.md` §1 está desatualizada desde
2026-07-11. Precisão sobre o delta, para não repetir aqui o mesmo pecado que o `CLAUDE.md`
§1 cobra de sessões anteriores ("meça antes de afirmar"): a contagem de linhas cresceu
1,137,484 linhas, isto é **+51.5%** sobre a base antiga de 2,208,306
(`(3,345,790 − 2,208,306) / 2,208,306`); equivalentemente, a base antiga é **34.0%** menor
que o total atual. Um "~40%" solto, sem dizer qual denominador, mistura as duas leituras —
corrigido aqui após revisão (`Sounio-lang/sounio#2637`, comentário
`discussion_r4071109706`). Isso não bloqueia este Bloco 0, mas é uma obrigação separada e
barata (atualizar §1 do `CLAUDE.md` com a saída deste comando) — registrada aqui, não
corrigida nesta entrega, para não misturar escopo de documentação com escopo de
inventário.

## 2. Hazards de instrumento (§5.4 — controle antes de confiar no instrumento)

Achados sobre as próprias ferramentas de medição, antes de usá-las para julgar código:

### 2.1 `scripts/dev/doctor_workspace.sh` assume `/workspace`

O script abre com `cd /workspace` incondicional (linha 4). Neste checkout
(`/home/user/sounio`, sem `/workspace/sounio`) ele falha imediatamente com
`No such file or directory` antes de emitir qualquer diagnóstico de saúde do
workspace. **Não foi corrigido nesta entrega** — é preservação de comportamento
observado, não um bug de compilador; corrigi-lo é uma mudança de uma linha fora do
escopo declarado deste Bloco (instrumentos de medição de corpus/inventário), e alterar
scripts de outra lane sem coordenação prévia viola §11 do plano v3. Registrado como
achado; publicado no bus de coordenação (`bin/sounio-coord send`, `msg-1790076290-751-10405`).

### 2.2 Seletor "slow": existe (via wrapper), mas a CI de #2622 não o escopa

Esta seção já foi corrigida uma vez (histórico abaixo) e a correção anterior **também
estava errada** — registrado aqui integralmente, porque um instrumento que erra duas
vezes sobre o mesmo fato precisa mostrar o rastro, não só o resultado final:

- **Erro original:** "não há marcador `slow` localizável" — vinha de grepar
  `scripts/run_sio_test_suite.sh` pelas strings erradas (`@slow`), sem notar que esse
  arquivo é um wrapper de 5 linhas (`exec bash .../run_sio_test_suite.sh "$@"`), não o
  harness real.
- **Primeira correção (d245c5f7), também errada:** ao investigar o wrapper eu li seu
  alvo nominal (`scripts/dev/run_sio_test_suite.sh`) como se fosse outro arquivo de
  texto e apliquei `grep` nele diretamente — mas **é um symlink** (`120000` no
  `git ls-tree`, confirmado tanto em `main` quanto em `pr-2622-recheck`), apontando para
  `run_sio_test_suite_v2.sh`. Eu nunca segui o link; concluí que o job `slow-lane`
  chamava "o v1, sem qualquer noção de `requires: slow`" — falso. (Apontado em revisão,
  `discussion_r4071219101`'s review body, "Correct wrapper call chain...".)
- **Estado real, verificado agora seguindo a cadeia completa** (`git ls-tree`, `readlink`,
  `git show`, em `main` `69b7fe7546e8` e em `pr-2622-recheck` = #2622 `35513cf80f22…`):
  `scripts/run_sio_test_suite.sh` → `exec` → `scripts/dev/run_sio_test_suite.sh`
  (symlink) → `run_sio_test_suite_v2.sh`. Ou seja, o job `slow-lane` do `ci.yml` de
  #2622, ao rodar `bash scripts/run_sio_test_suite.sh --format junit --jobs 4`, **chega
  em v2**, não em algum "v1" sem seletor. Em `main`, esse mesmo v2 já suporta
  `--test-list` mas **não** tem nenhuma ocorrência de `slow` (`grep -n slow
  scripts/dev/run_sio_test_suite_v2.sh` → vazio) — o seletor `requires: slow`
  (linha ~461, gated por `SOUNIO_SLOW_TESTS_AVAILABLE`) é adicionado pelo próprio
  #2622. Logo: o job **consome** `SOUNIO_SLOW_TESTS_AVAILABLE` normalmente (a variável
  não é lida por um script morto); o único defeito real é a ausência de `--test-list`
  na chamada do `ci.yml`, fazendo o job rodar N e S juntos sob o orçamento pensado só
  para S — exatamente o que a revisão original (`discussion_r4071109755`) disse antes
  de eu "corrigi-la" incorretamente.
- Isto **permanece não corrigido nesta entrega** — é dívida de #2622/CI (`ci.yml`, trocar
  a chamada para incluir `--test-list <arquivo-S>`), fora do write-set deste Bloco 0
  (instrumentos de inventário, branch `claude/sounio-divida-zero-v3-g7svhy`); alterar o
  workflow de outra PR sem coordenação viola §11 do plano v3. Fica como pré-requisito
  explícito para §5.2/§6.2 antes de qualquer partição N/S ser declarada confiável.
  **Escopo exato do que falta** (corrigido após revisão, `discussion_r4071317039`, que
  apontou generalização indevida aqui): `SOUNIO_SLOW_TESTS_AVAILABLE` só é setada no job
  `slow-lane` do `ci.yml` de #2622 (confirmado: única ocorrência da variável no arquivo);
  o job normal (`Full Test Suite`, N apenas) roda com a variável ausente, então o ramo
  `requires: slow` recém-adicionado já pula S corretamente ali, e a alegação "zero
  falhas" desse job N-only **não** é afetada por este defeito. O que falta é
  especificamente evidência de uma execução isolada de **S** (só os testes lentos, via
  `--test-list`) — não uma dúvida sobre o resultado N-only já relatado. A alegação da
  #2622 no plano v3 §3 é "zero falhas na seleção que exclui casos lentos" (isto é, sobre
  N), que este achado não contesta; o que continua sem lastro é qualquer alegação futura
  sobre S isolado.

## 3. Amostra do universo do corpus (§5.2) — contagens, não partição qualificada

Contagens brutas em `69b7fe7546e8`, sem execução da suíte (nenhum teste foi rodado
nesta sessão):

| Conjunto amostrado | Contagem | Método |
|---|---:|---|
| `tests/known_failures/` — arquivos `.sio` | 30 | `find tests/known_failures -maxdepth 1 -name '*.sio' \| wc -l` |
| `tests/known_failures/` — arquivos totais (inclui `hardened_diagnostics_full_suite.txt`) | 31 | `ls tests/known_failures \| wc -l` |
| Arquivos com `known_failure` no nome fora de `tests/known_failures/` | 4 | `find tests -iname '*known_failure*'`, excluindo o diretório e as 2 entradas já dentro dele |
| Arquivos citando `known_failure`/`expected_failure`/`XPASS`/`xfail` no corpo | 18 | `grep -rl -e known_failure -e expected_failure -e XPASS -e xfail tests/ --include='*.sio'` |
| Arquivos anotados `requires: madaros` (gramática do executor real, `run_sio_test_suite_v2.sh`: substring `//@ requires` + `requires:[[:space:]]*(.+)` + match exato contra `madaros`) | 1,171 | `grep -rlE '//@ requires:[[:space:]]*madaros' tests/ --include='*.sio'` |
| Arquivos `.sio` totais em `tests/` (informativo, do measure_repo_scale) | 4,530 | `scripts/dev/measure_repo_scale.sh` |

Correção de quatro problemas desta tabela (apontado em revisão,
`discussion_r4071264975`: o cabeçalho dizia "três" com quatro itens já listados):

1. (Apontado em revisão, `Sounio-lang/sounio#2637` `discussion_r4071109816`.) A linha
   original rotulava os 31 resultados de `ls` como ".sio", mas 30 são `.sio` e 1 é
   `hardened_diagnostics_full_suite.txt` — separados acima. A linha "fora do diretório"
   original (7) somava, sem dizer, o próprio diretório `tests/known_failures` mais as 2
   entradas já contadas na primeira linha; o `find` bruto retorna 7 caminhos, mas os
   arquivos genuinamente fora do diretório e ainda não contados são 4
   (`tests/run-pass/{ontology_multiple_disjoint,wide_i128_fn_abi,wide_i128_signed_div,wide_i256_divfull_scratch}_known_failure.sio`).
2. (Apontado em revisão, `discussion_r4071171510`.) O comando original das duas últimas
   linhas usava o glob `tests/**/*.sio`, que só expande recursivamente com `shopt -s
   globstar` habilitado — ausente por padrão num shell não-interativo. Sem `globstar`,
   `tests/**/*.sio` expande como `tests/*/*.sio` (um nível), reduzindo a contagem
   literal de `requires: madaros` de 1,165 para 1,154 (verificado: `bash -c 'shopt -u
   globstar; grep -rl "requires: madaros" tests/**/*.sio | wc -l'` → 1154). Os comandos
   acima passam a recursar via `grep -r`/`--include` sobre o diretório, eliminando a
   dependência de `globstar`.
3. (Achado adicional desta correção, não reportado em revisão: `grep -rl "requires:
   madaros" tests/` sem `--include='*.sio'` retornava 1,166 — captura também
   `tests/vacuous_expect_baseline.txt`, um `.txt` fora do escopo ".sio" desta linha.)
4. (Apontado em revisão, `discussion_r4071219101`, e refinado após uma segunda rodada,
   `discussion_r4071317039`'s review body, "Align regex grammar with the actual v2
   harness source".) A contagem 1,165 (substring literal `"requires: madaros"`, com
   espaço, sem âncora de prefixo) não é a gramática de nenhum consumidor real desta
   anotação. Há três gramáticas distintas na árvore, e a linha original desta correção
   já tinha atribuído a errada como "a do harness":
   - `scripts/ci/madaros_changed_tests_gate.sh` (linha 46): literal exato
     `//@ requires: madaros` (um espaço obrigatório, gate de seleção de testes mudados,
     não o executor).
   - `scripts/ci/known_failure_madaros_recheck.sh` (linha 50): regex Python
     `//@\s*requires:\s*madaros\b`, aceitando zero ou mais espaços em qualquer um dos
     dois pontos — inclusive **zero espaços entre `//@` e `requires`**, o que nenhum
     outro consumidor aceita. Este é um script de recheck secundário, não o executor
     principal da suíte.
   - `scripts/dev/run_sio_test_suite_v2.sh` (linhas 363–364, 419, 426) — **este sim é o
     executor real**: um `case` em shell casa a substring literal `//@ requires`
     (espaço único obrigatório entre `//@` e `requires`, sem flexibilidade — é texto
     literal num padrão glob, não regex), extrai o valor com
     `[[ "$line" =~ requires:[[:space:]]*(.+) ]]` (espaço opcional só depois dos
     dois-pontos), e então compara esse valor com `case "$requires" in madaros) ...`
     — **igualdade exata** contra a string `madaros`.

   O comando `grep -rlE '//@ requires:[[:space:]]*madaros' tests/ --include='*.sio'`
   (linha acima) já reproduz corretamente a gramática do executor real (espaço único
   fixo entre `//@` e `requires`, espaço opcional só após os dois-pontos) — produz
   **1,171** arquivos: 8 usam `//@ requires:madaros` sem espaço após os dois-pontos
   (fora da contagem 1,165, que exigia esse espaço) e a antiga contagem literal também
   incluía 2 arquivos com apenas `// requires: madaros` (sem `@`, que nenhum dos três
   consumidores reconhece) que a forma âncorada em `//@` corretamente exclui. A
   divergência entre os três consumidores do próprio repositório sobre espaço
   obrigatório vs. opcional (e sobre qual delimitador é literal vs. regex) é, ela
   mesma, um achado de instrumento — não resolvido nesta entrega, fora do write-set
   deste Bloco 0.

Isto **não é** a partição `U = N ⊎ S ⊎ X` exigida por §5.2 — é uma amostra de
instrumentação para dimensionar o problema antes de construir a partição real. A
partição qualificada requer: (a) resolver o achado §2.2 acima, (b) rodar o gerador de
census/datasets mencionado em §6.2, (c) declarar `X` (exclusões contratuais) por
engine/target explicitamente. Nenhuma dessas três etapas foi executada nesta sessão.

## 4. Esqueleto do inventário por identidade (§5.1)

Campos do schema §5.1 aplicados às referências já citadas no plano v3 §3/§14. Os campos
de evidência de execução (`compiler_digest`, `witness_id`, `evidence_receipt`) ficam
`PENDING` — não foram gerados nesta sessão, que não compilou nem executou testes.

| contract_id (local) | capability | referência | base_commit | state | evidence_receipt |
|---|---|---|---|---|---|
| DZ-BLOCO1-2622 | BSS sizing, primal preservation, tuple-borrow local | PR #2622 `35513cf80f22…` | 69b7fe75 | PENDING (não requalificada nesta sessão) | — |
| DZ-BLOCO1-2557 | CI em camadas / artifact compartilhado | PR #2557 `c5248c46436c…` | 69b7fe75 | PENDING | — |
| DZ-BLOCO2-2598 | identidade de função homônima cross-module | PR #2598 `f23be0816d17…` | 69b7fe75 | PENDING | — |
| DZ-BLOCO2/5-2501 | inventário de deltas estabilizados (não integrar monolítico) | PR #2501 `8bfab7e6939a…` | 69b7fe75 | PENDING (revisão em blocos, §7.2 do plano) | — |
| DZ-BLOCO3-2511 | ODE/covariância/modelo | PR #2511 `0d389e07dd98…` | 69b7fe75 | PENDING | — |
| DZ-BLOCO3-2612 | EL+/SNOMED fail-closed | PR #2612 `bf7172c98b66…` | 69b7fe75 | PENDING | — |
| DZ-BLOCO2/3-2615 | array-reference lowering + wide-float coverage | PR #2615 `a80ca04ac4ec…` (ver polaridade anti-f64 apontada em revisão) | 69b7fe75 | PENDING | — |

`owner`/`write_set`/`dependencies` não foram atribuídos — §11 do plano v3 é explícito
que "responsáveis/lane assignments são propostas, não leases efetivamente adquiridos";
este documento não cria leases além do já registrado em `bin/sounio-coord scope` para
esta sessão (`claude`, lane `session-fc7d9b6f-e027-5e0f-89b0--66f0de5791`, intenção
"Bloco 0: inventário e instrumentos, sem build pesado").

## 5. O que este documento não faz

- Não executa `full-test-suite`, `make build`, `make build-madaros`, Contracts, nem
  qualquer build pesado (respeitando a disciplina de concorrência do `CLAUDE.md`,
  seção "Concurrency discipline").
- Não requalifica nenhuma PR citada; não muda estado de merge de nenhuma delas.
- Não corrige `doctor_workspace.sh` nem o seletor slow — apenas os cataloga.
- Não declara M1, M2 ou M3. Não fecha DZ-00 nem DZ-01; abre o instrumento para que
  Bloco 1 (§6 do plano v3) possa correr sobre uma base medida, não presumida.

## 6. Próximo passo declarado

Bloco 1 (§6.1–§6.3 do plano v3): localizar o mecanismo real de seleção `slow`
(resolvendo o achado §2.2 deste documento) antes de qualquer tentativa de declarar a
partição N/S para a candidata baseada na #2622. Esse é um trabalho de leitura/repro
mínima (§6.1 do `CLAUDE.md`, "um blocker sem reprodução mínima não está diagnosticado"),
não um build completo, e deve ser o próximo item desta lane.
