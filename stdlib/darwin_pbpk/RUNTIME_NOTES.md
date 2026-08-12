# Runtime notes — Madaros v0.80 (cluster `sounio-workspace-control`)

Notas de runtime encontradas durante a Fase 0–6 do estudo UTIped PBPK
(Agourakis DC, 2026). Cada item foi isolado com probe mínimo em
`tests/run-pass/probe*.sio` no cluster. Estas limitações são do **build
Madaros v0.80 atual**, não da semântica da linguagem; drivers da suíte
UTIped contornam cada uma de forma documentada.

## Limitações e workarounds

1. **`tsit5_step_pbpk` / `pbpk_ode` (`darwin_pbpk/tsit5_pbpk14.sio`) — segfault
   em runtime.** Compila, mas o binário segfaulta na primeira chamada.
   *Workaround:* drivers consomem os **construtores de parâmetros** dos
   módulos canônicos (que funcionam) e integram com kernel Euler/CN de passo
   fixo próprio (`State28/Params28`, convenção de `core/pbpk28_params.sio`).

2. **Thin-link exige import explícito da dependência transitiva no arquivo
   raiz.** `use darwin_pbpk::drugs::morphine::{...}` falha com
   `multimodule native thin-link compilation failed` se a raiz não importar
   também `darwin_pbpk::tsit5_pbpk14::{PBPKParams14}` (dependência do módulo).
   *Workaround:* sempre importar a transitiva no root.

3. **Structs locais + `use` no mesmo arquivo quebram o thin-link** (rc=12),
   assim como funções de módulo que retornam arrays (`[f64; 14]`).
   *Workaround:* drivers self-contained com literais extraídos do módulo
   canônico via programa de dump (ex.: `tests/run-pass/morph_dump.sio`);
   o módulo permanece a fonte única de verdade.

4. `print`/`println` fora de `main` → segfault (helpers devem ser puros).
5. Tupla como retorno de função → segfault; usar struct + `return` explícito.
6. `let` com divisão de `i64` → segfault; var i64 inicializada de outra var
   i64 → segfault (desenrolar com literais).
7. >~100–150k passos cumulativos de simulação por processo → corrupção
   silenciosa de estado. *Workaround:* 1 fatia por processo (workers),
   agregação externa (ver `fentanyl_fase3_gum_worker.sio`).

## Gotcha histórico (resolvido na suíte)

- `exp_f64` por Taylor-20 só converge |x|≲4 — corrigido com squaring
  recursivo (`x>20 → exp(x/2)²`); revalidado com drift 0.06%.

## Referências

- Estudo: "Uncertainty-quantified PBPK modeling of opioid weaning in
  critically ill children" (Fases 0–6: fentanil, midazolam, metadona,
  morfina; coorte HMMG 2025, n=61 elegíveis).
- Módulos: `drugs/fentanyl.sio`, `drugs/morphine.sio` (este último com
  ontogenia UGT2B7×OCT1 e anotações ontológicas).
- Teste: `tests/stdlib/darwin_pbpk/test_morphine_validation.sio`.
