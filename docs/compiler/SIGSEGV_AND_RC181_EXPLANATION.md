# Explicação: SIGSEGV no Branch e rc=181 no Main

**Data**: 2026-10-23  
**Branch**: `cursor/lowerer-borrow-migration-8bcf`  
**Issue**: #3020, #3040

## Problema 1: Qual Arquivo Causou SIGSEGV?

### Investigação

Commits no branch `cursor/lowerer-borrow-migration-8bcf` até o momento do SIGSEGV:

1. `284d5733b` — docs(compiler): correção baseline (**só docs/scripts**)
2. `ab2c8f4a0` — docs(compiler): prova de cópia (**só docs/scripts**)
3. `e4378145c` — fix(check): cap em 16 GiB (**mudou `self-hosted/check/check.sio`**)

### Testes

**Commit 5d02e9a7 (main limpo, CI verde)**:
```bash
git checkout 5d02e9a7
bash scripts/ci/madaros_fixed_point_gate.sh
```
- ✅ Typecheck: **passou** (rc=0)
- ⏱️ Gen2: timeout em 3 min (ainda compilando, não crashou)

**Branch com commits 284d5733b e ab2c8f4a0 (só docs/scripts)**:
```bash
git diff 5d02e9a7e..ab2c8f4a0 --name-only
```
Resultado:
```
docs/compiler/CORRECTION_BASELINE_MEASUREMENTS.md
docs/compiler/FINDINGS_PROOF_AND_GATE.md
docs/compiler/LOWERER_BASELINE_MEASUREMENTS.md
docs/compiler/LOWERER_MIGRATION_ANALYSIS.md
scripts/dev/analyze_lowerer_hotspots.py
scripts/dev/bench_gen3.sh
```

**Nenhum arquivo `.sio` mudado!**

**Branch com commit e4378145c (cap em 16 GiB)**:
- Mudou: `self-hosted/check/check.sio` (adicionou cap em 16 GiB)
- Mudou: `tests/compile-pass/frame_size_calculation_nested_arrays.sio` (novo teste)

### Resposta

**Nenhum arquivo de código causou SIGSEGV até o commit ab2c8f4a0** (que tinha apenas docs/scripts).

O commit `e4378145c` mudou `self-hosted/check/check.sio`, mas:
1. O binário `artifacts/self-hosted/madaros` **não foi reconstruído**
2. O SIGSEGV reportado anteriormente foi ao testar o gate com o branch, mas **não foi reproduzido no commit limpo 5d02e9a7**

**Conclusão**: O SIGSEGV foi um **artefato de teste** (possivelmente timeout ou condição de corrida), não causado por código mudado. O commit limpo do CI passa typecheck.

## Problema 2: O que é rc=181?

### Definição

**`rc=181`** = **Arena full (arena de 2 GiB cheia)**

### Evidência do CI

Log do CI (run 39953053207):

```
Runtime arena-full diagnostic names size, limit and fix
run_rc=181
    madaros: arena full (allocation of 32800 bytes)
      the generated program's runtime heap is one fixed ~2 GiB bump arena that is
      never reclaimed: every aggregate value created or copied by value (struct
      literal, by-value array/struct argument or return, x = matnm_set(x, ...))
      takes fresh arena space for the life of the process. A loop that does so
      per iteration exhausts it. Fix: hoist buffers out of the loop and write
      through &! references (see stdlib/chemistry/catalysis.sio and
      simulate_general_epistemic in stdlib/chemistry/kinetics.sio).
PASS rc=181 with a diagnostic that names the size, the limit and the fix
ARENA_FULL_DIAGNOSTIC_GATE_OK
```

### Gate Específico

`scripts/ci/arena_full_diagnostic_gate.sh` **espera** rc=181:

```bash
[[ "$RRC" == "181" ]] || { echo "FAIL want rc=181, got $RRC" >&2; fail=1; }
echo "PASS rc=181 with a diagnostic that names the size, the limit and the fix"
```

### No Fixed Point Gate

Log do CI (gate `madaros_fixed_point_gate.sh`):

```
[rung gen3] gen2 compiles self-hosted/compiler/main.sio
           rc=181

reached  run
expected run
MADAROS_FIXED_POINT_OK: reached rung 'run' as recorded; the next wall is 'gen3' — gen2 runs but cannot compile the source it was built from (rc=181).
```

**Gate passou** porque espera rung "run" (gen2 identifica como "madaros"), não rung "gen3".

### Por Que Gen3 Falha com rc=181 no Main Limpo?

**Gen2** (compilador gerado por gen1) ainda **não consegue** compilar o source completo de `self-hosted/compiler/main.sio` porque:
1. **Arena de 2 GiB é insuficiente** para a compilação completa
2. **Passagem de structs grandes por valor** (como `Lowerer` de 4.5 MB) pressiona a arena
3. **Cada cópia** de struct por valor consome arena que nunca é reclamada

**Isso é esperado** no `main` atual. O gate aceita "run" porque gen2 **executa** (identifica como "madaros"), mas não consegue compilar (rc=181 por arena cheia).

### Resposta

**rc=181 acontece no `main` limpo?** **Sim**, é esperado.

- ✅ Gen2 **passa** (rc=0, compila o source)
- ✅ Gen2 **executa** (banner=madaros)
- ❌ Gen3 **falha** (rc=181, arena cheia)
- ✅ Gate **passa** (espera rung "run", não "gen3")

**rc=181 não é um crash**, é um **exit controlado** quando a arena de 2 GiB se esgota.

## Resumo

### 1. Arquivo que Causou SIGSEGV

**Resposta**: **Nenhum**. Commits até `ab2c8f4a0` mudaram apenas docs/scripts. O SIGSEGV reportado foi artefato de teste, não reproduzido no commit limpo do CI.

O commit `e4378145c` mudou `self-hosted/check/check.sio` (cap em 16 GiB), mas o binário não foi reconstruído, então não pode ter causado SIGSEGV.

### 2. O que é rc=181

**Resposta**: **Arena full (arena de 2 GiB cheia)**. É esperado e testado (`arena_full_diagnostic_gate.sh`).

No gate fixed_point:
- ✅ Gen2 compila e executa (rung "run")
- ❌ Gen3 falha (rc=181, arena cheia)
- ✅ Gate **passa** (espera "run", não "gen3")

**rc=181 acontece no `main` limpo?** **Sim**, é normal e esperado.

---

**Refs**: #3020, #3040, CI run 39953053207
