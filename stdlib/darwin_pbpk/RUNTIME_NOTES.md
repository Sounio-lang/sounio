# Runtime notes — Madaros v0.80 (cluster `sounio-workspace-control`)

Limitações de runtime do compilador Madaros v0.80 encontradas durante o
desenvolvimento PBPK (incluindo a suíte UTIped). Cada item foi isolado
com probe mínimo em `tests/run-pass/probe*.sio` no cluster. Estas
limitações são do **build Madaros v0.80 atual**, não da semântica da
linguagem.

O estudo **Proof-Carrying Weaning** (ocupância μ/α2, fentanil/morfina/
clonidina, Lean F5c) foi extraído para o repositório irmão
`../sounio-pbpk-sedation-weaning` — ver `WEANING_MOVED.md`. Notas
específicas do estudo e drivers Euler/CN vivem lá.

## Limitações e workarounds

1. **`tsit5_step_pbpk` / `pbpk_ode` (`darwin_pbpk/tsit5_pbpk14.sio`) — segfault
   em runtime.** Compila, mas o binário segfaulta na primeira chamada.
   *Workaround:* drivers consomem os **construtores de parâmetros** dos
   módulos canônicos (que funcionam) e integram com kernel Euler/CN de passo
   fixo próprio (`State28/Params28`, convenção de `core/pbpk28_params.sio`).

2. **Thin-link exige import explícito da dependência transitiva no arquivo
   raiz.** Importar um módulo de fármaco que depende de `PBPKParams14` falha
   com `multimodule native thin-link compilation failed` se a raiz não
   importar também `darwin_pbpk::tsit5_pbpk14::{PBPKParams14}`.
   *Workaround:* sempre importar a transitiva no root.

3. ~~Structs locais + `use` no mesmo arquivo quebram o thin-link (rc=12),
   assim como funções de módulo que retornam arrays.~~ **CORRIGIDO
   2026-08-13 (review PR #1714): não era um bug de thin-link.** Era
   `with Mut` faltando nos helpers matemáticos locais do módulo — o
   compilador stage2/Madaros aceita a chamada sem o efeito declarado mas
   rejeita (ou, dependendo da versão, compila incorretamente) em vez de
   propagar o erro de forma óbvia. Com os efeitos corrigidos, struct local
   + `use` no mesmo arquivo + `pub fn` retornando `[f64; 14]` cruzando
   módulo funcionam normalmente.

4. `print`/`println` fora de `main` → segfault (helpers devem ser puros).
5. Tupla como retorno de função → segfault; usar struct + `return` explícito.
6. `let` com divisão de `i64` → segfault; var i64 inicializada de outra var
   i64 → segfault (desenrolar com literais).
7. >~100–150k passos cumulativos de simulação por processo → corrupção
   silenciosa de estado. *Workaround:* 1 fatia por processo (workers),
   agregação externa.

## Gotcha histórico (resolvido)

- `exp_f64` por Taylor-20 só converge |x|≲4 — corrigido com squaring
  recursivo (`x>20 → exp(x/2)²`); revalidado com drift 0.06%.

## Referências

- Weaning / opioid–α2: `../sounio-pbpk-sedation-weaning` (`WEANING_MOVED.md`).
- Midazolam CYP3A DDI e core Tsit5/PBPK28/GUM: este stdlib.
- Madaros native **run** of imported `pbpk28_full_cn_step`: **stdlib-mitigated**
  2026-08-16 (mut + `pbpk28_cn_apply_schur` split; smoke cv1 parity). Compiler
  SRET / same-frame local-array bugs remain OPEN — see
  `docs/audit/MADAROS_IMPORTED_PBPK28_CN_SIGSEGV_2026-08-16.md`. Sibling may still
  use `SOUNIO_SOUC_ENGINE=lean_single` until broader multimodule trust expands.
