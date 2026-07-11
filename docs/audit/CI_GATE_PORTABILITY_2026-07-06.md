<!-- docs:meta
topic_id: repo.docs.audit.ci-gate-portability-2026-07-06
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ci-gate-portability-2026-07-06
-->

# Auditoria de portabilidade — gates EISA/Sounio para Slurm (2026-07-06)

**Escopo:** `scripts/ci/eisa_bridge_conformance_gate.sh`, `slurm-jobs/eisa/submit-eisa-battery.sh`, e `bin/souc` (invocado indiretamente via `./bin/souc run`).

**Contexto do incidente:** primeira bateria Slurm falhou na lane *anti-vacuity* porque o nó `gpuorangefs-5860-proxmox` não possui `strings(1)` (pacote `binutils`). A falha foi de tooling, não semântica.

**Nota sobre `scripts/lib/`:** `submit-eisa-battery.sh` empacota `scripts/lib` no tarball, mas **nenhum** dos dois scripts faz `source` de `scripts/lib/*.sh` em runtime. A cadeia EISA depende apenas de `bin/souc` + gate + utilitários POSIX.

---

## 1. Sonda mandatória no cluster

Comando (executado a partir do pod, 2026-07-06T02:47:18Z):

```bash
srun --partition=all --time=00:02:00 --mem=256M bash -c \
  'for t in strings cmp diff sha256sum awk sed tr tar mktemp grep od python3 gcc; do
     command -v $t >/dev/null && echo "$t OK" || echo "$t MISSING"
   done'
```

Avisos esperados em stderr (`couldn't chdir to /workspace/sounio-eisa`, `TMPDIR → /tmp`) — nós não montam `/workspace`.

| Binário | Estado no nó | Usado na cadeia EISA? |
|---------|--------------|----------------------|
| `strings` | **MISSING** | **Sim** — lane anti-vacuity (`eisa_bridge_conformance_gate.sh`) |
| `cmp` | OK | Não (sonda de referência) |
| `diff` | OK | Sim — conformidade + tamper-sensitivity |
| `sha256sum` | OK | Não |
| `awk` | OK | Não |
| `sed` | OK | Sim — extração de digit runs |
| `tr` | OK | Não (fallback candidato para `strings`) |
| `tar` | OK | Sim — empacotar/desempacotar bateria |
| `mktemp` | OK | Sim — via `bin/souc` (`souc run` → temp ELF em `/tmp`) |
| `grep` | OK | Sim — anti-vacuity, contagem PASS, tail de logs |
| `od` | OK | Não (sonda de diagnóstico) |
| `python3` | OK | Não |
| `gcc` | **MISSING** | **Não** — bateria usa ELFs pré-compilados (`bin/souc-lean-single-x86_64`) |

Nó da sonda mandatória: hostname não impresso nesta srun (stderr apenas); nós da frota `gpuorangefs-*` compartilham a mesma imagem mínima.

---

## 2. Sonda complementar (binários adicionais do inventário)

```bash
srun --partition=all --time=00:02:00 --mem=256M bash -c \
  'for t in head chmod du cut date hostname sort wc bash rm mkdir cat sbatch srun sync tee; do
     command -v $t >/dev/null && echo "$t OK" || echo "$t MISSING"
   done'
```

**Resultado:** todos OK (2026-07-06T02:47:26Z).

---

## 3. Inventário completo de binários externos invocados

### 3.1 `scripts/ci/eisa_bridge_conformance_gate.sh`

| Binário | Invocação | Lane |
|---------|-----------|------|
| `bash` | shebang + builtins (`mapfile`, `[[`, loops) | todas |
| `./bin/souc` | `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run …` | emit + EVM ref |
| `cat` | concatena stdout dos ELFs | conformidade |
| `diff` | `diff -u`, `diff -q` | conformidade, tamper |
| `head` | `head -n 1` | tamper |
| `strings` | `strings -a "$elf" \| grep …` | **anti-vacuity (BLOCKER)** |
| `grep` | `-o`, `-q`, pipeline com `sed`/`sort` | anti-vacuity |
| `sed` | `sed 's/^m//'` | anti-vacuity |
| `sort` | `sort -u` | anti-vacuity |

Artefactos gerados em paths **relativos** ao checkout desempacotado: `artifacts/eisa/`, `artifacts/eisa/.gate_tmp/`.

### 3.2 `slurm-jobs/eisa/submit-eisa-battery.sh`

| Binário | Invocação | Fase |
|---------|-----------|------|
| `bash` | shebang, heredoc sbatch, subshells | todas |
| `date` | `-u +%Y%m%dT%H%M%S`, `-u +%FT%TZ` | run-id, SUMMARY |
| `tar` | `-czf` (pod), `-xzf` (nó) | staging |
| `du` / `cut` | tamanho do tarball | staging (pod) |
| `srun` | pipe stdin → OrangeFS; fetch | staging + monitor |
| `rm` | limpeza `/tmp` e workdir | staging + pós-job |
| `cat` | heredoc, fetch, pipe staging | staging |
| `sbatch` | submissão | submit |
| `mkdir` | `-p` work/out/share | nó |
| `grep` | `-v receipt`, `-c '^PASS'` | test loop, gate |
| `tail` | `-1`, `-40` | test loop, fetch |
| `hostname` | SUMMARY | pós-job |
| `cp` | `/tmp/gate.log` → `${OUT}/gate.log` | **artefacto corrupto (ver §5)** |

Mencionado mas **não invocado** pelo script: `sacct` (apenas eco de monitorização).

### 3.3 `bin/souc` (cadeia `souc run` com `SOUNIO_SOUC_ENGINE=lean_single`)

| Binário | Invocação |
|---------|-----------|
| `head` | `-c2` — detecção ELF |
| `mktemp` | `/tmp/souc-lean-run-XXXXXX.elf`, `/tmp/souc-lean-check-XXXXXX.elf` |
| `chmod` | `+x` no temp ELF |
| `rm` | limpeza temp check |
| `exec` / `env` | delegação ao lean_single |

---

## 4. Pressupostos de path absoluto

| Path | Onde | Risco no nó Slurm |
|------|------|-------------------|
| `/tmp/${RUN_ID}.*` | `submit-eisa-battery.sh` (tarball, sbatch, work, logs) | **OK** — `/tmp` local ao nó existe |
| `/tmp/souc-lean-run-*.elf` | `bin/souc` | **OK** |
| `/tmp/one.log`, `/tmp/gate.log` | heredoc sbatch | **OK** para escrita; **gate.log → OrangeFS via `cp` corrompe** (§5) |
| `/orangefs/training/eisa` | `SHARE` fixo | **OK** — superfície partilhada uid-1000 |
| `/workspace/sounio-eisa` | *não referenciado* nos scripts | Slurm avisa ao arrancar srun (pod cwd inexistente no nó); inofensivo |
| `/dev/null` | `bin/souc compile` fallback | OK |
| `ROOT_DIR` relativo | resolvido via `BASH_SOURCE` | OK após `tar -C "${WORK}" -xzf` |

**Não há** hardcode de `/workspace/...` dentro dos scripts EISA; o checkout vive em `${WORK}` no nó.

---

## 5. Sonda OrangeFS: `cp` vs redirecionamento

Executado em `gpuorangefs-multi-r740-proxmox` (2026-07-06T02:47:26Z):

| Método | Destino | Tamanho | Bytes NUL (`grep -a -c $'\x00'`) | Conteúdo |
|--------|---------|---------|-----------------------------------|----------|
| `printf … > /tmp/…-redir.txt` | `/tmp` | 62 | 3 (newlines) | texto legível |
| `cp /tmp/…-src.txt` | `/orangefs/training/eisa/…/via-cp.txt` | 62 | **1** (ficheiro quase todo NUL) | **corrupto** |
| `cat /tmp/…-redir.txt > …/via-cat-redir.txt` | OrangeFS | 62 | 3 | **idêntico ao original** |

```
cmp redir vs via-cp: DIFFER
cmp redir vs via-cat-redir: IDENTICAL
```

**Conclusão factual:** `cp` de `/tmp` → OrangeFS produz ficheiro NUL-padded/corrupto neste cluster; redirecionamento (`>`) ou `cat … >` preserva bytes. Isto explica `gate.log` ilegível observado em 2026-07-06 quando a bateria usou `cp /tmp/gate.log "${OUT}/gate.log"`.

---

## 6. Fallbacks propostos (patches para agente pai — **não aplicados**)

### 6.1 `strings` → `grep -a` (mínimo, sem binutils)

A lane anti-vacuity só precisa de saber se uma substring aparece nos bytes do ELF. `grep -a -q` no ficheiro binário substitui `strings -a … | grep -q` sem exigir runs imprimíveis completos.

```diff
--- a/scripts/ci/eisa_bridge_conformance_gate.sh
+++ b/scripts/ci/eisa_bridge_conformance_gate.sh
@@ -106,13 +106,13 @@ for name in "${programs[@]}"; do
   case "$name" in
     v1-*|v1e-*) expected_prefix="v=2 prog=" ;;
   esac
-  if ! strings -a "$elf" | grep -q "$expected_prefix"; then
+  if ! grep -a -q "$expected_prefix" "$elf"; then
     echo "FAIL anti-vacuity ${name}: label prefix not found in ELF (strings check broken)"
     exit 1
   fi
   # Mantissa digit runs (>= 8 digits) from val/roundoff/u fields must not
   # appear as literal bytes in the ELF.
   mapfile -t digit_runs < <(grep -o 'm[0-9]\{8,\}' "$per_prog_out" | sed 's/^m//' | sort -u)
   for run in "${digit_runs[@]}"; do
-    if strings -a "$elf" | grep -q "$run"; then
+    if grep -a -q "$run" "$elf"; then
       echo "FAIL anti-vacuity ${name}: receipt digits '${run}' baked into ELF bytes"
       exit 1
     fi
   done
 done
```

**Alternativa mais fiel a `strings(1)`** (se algum prefixo deixar de ser contíguo nos bytes):

```bash
# _eisa_strings FILE — extrai runs imprimíveis ≥4 chars sem binutils
tr -c '[:print:]' '\n' < "$1" | awk 'length >= 4'
```

Preferir `grep -a` primeiro: mais simples e adequado aos padrões actuais (`v=1 prog=`, digit runs contíguos).

### 6.2 `cp` → escrita directa em OrangeFS (gate.log)

Eliminar round-trip `/tmp` → `cp` → OrangeFS.

```diff
--- a/slurm-jobs/eisa/submit-eisa-battery.sh
+++ b/slurm-jobs/eisa/submit-eisa-battery.sh
@@ -88,9 +88,8 @@ for t in tests/stdlib/eisa/*.sio tests/stdlib/math/test_qd128_core.sio tests/st
   echo "\${st} rc=\${rc} \${t} :: \${last}" >> "\${OUT}/battery.log"
 done
 
-bash scripts/ci/eisa_bridge_conformance_gate.sh > /tmp/gate.log 2>&1
+bash scripts/ci/eisa_bridge_conformance_gate.sh > "\${OUT}/gate.log" 2>&1
 grc=\$?
-lanes=\$(grep -c '^PASS' /tmp/gate.log)
-cp /tmp/gate.log "\${OUT}/gate.log"
+lanes=\$(grep -c '^PASS' "\${OUT}/gate.log")
 if [[ \$grc -eq 0 ]]; then gst=PASS; else gst=FAIL; fail=\$((fail+1)); fi
```

**Fallback conservador** (se quiser manter `/tmp/gate.log` para debug local ao nó):

```bash
bash scripts/ci/eisa_bridge_conformance_gate.sh 2>&1 | tee /tmp/gate.log > "${OUT}/gate.log"
```

### 6.3 Preflight de tooling no job Slurm (fail-fast)

```diff
--- a/slurm-jobs/eisa/submit-eisa-battery.sh
+++ b/slurm-jobs/eisa/submit-eisa-battery.sh
@@ -74,6 +74,12 @@ cd "\${WORK}"
 export SOUNIO_STDLIB_PATH="\${WORK}/stdlib"
 export SOUNIO_SOUC_ENGINE=lean_single
 export TMPDIR="\${WORK}/tmpdir"
 mkdir -p "\${TMPDIR}"
+
+for _need in bash tar diff grep sed sort head mktemp chmod; do
+  command -v "\${_need}" >/dev/null || { echo "FAIL preflight: missing \${_need}"; exit 127; }
+done
+# anti-vacuity needs grep -a (not strings); verified on gpuorangefs-* 2026-07-06
 
 pass=0; fail=0
```

---

## 7. Top-3 patches recomendados (prioridade)

| # | Patch | Motivo | Evidência |
|---|-------|--------|-----------|
| **1** | `strings` → `grep -a` no gate (`§6.1`) | **Bloqueador** — gate FAIL na lane anti-vacuity | `strings MISSING` na sonda mandatória; incidente 2026-07-06 |
| **2** | Escrita directa de `gate.log` para OrangeFS (`§6.2`) | **Artefactos corruptos** — impossível auditar falhas remotamente | Sonda `cp` vs `>` confirma NUL-corruption (`cmp DIFFER`) |
| **3** | Preflight de tooling no heredoc sbatch (`§6.3`) | Fail-fast com mensagem clara; evita horas de compute antes de tooling gap | Complementa #1; `gcc MISSING` é inofensivo hoje mas preflight documenta dependências reais |

**Não prioritário:** instalar `binutils`/`gcc` nos nós — a bateria EISA não compila C; `grep -a` + ELFs pré-buildados bastam.

---

## 8. Comandos de reprodução

```bash
# Sonda mandatória
srun --partition=all --time=00:02:00 --mem=256M bash -c \
  'for t in strings cmp diff sha256sum awk sed tr tar mktemp grep od python3 gcc; do
     command -v $t >/dev/null && echo "$t OK" || echo "$t MISSING"; done'

# Sonda OrangeFS cp vs redirecionamento (limpar PROBE_ID após teste)
srun --partition=all --time=00:02:00 --mem=256M bash -c '
  OUT=/orangefs/training/eisa/portability-probe-manual
  mkdir -p "$OUT" && printf "line1\nline2\n" > /tmp/p.src
  cp /tmp/p.src "$OUT/via-cp.txt"
  cat /tmp/p.src > "$OUT/via-cat.txt"
  cmp /tmp/p.src "$OUT/via-cp.txt" && echo cp_ok || echo cp_corrupt
  cmp /tmp/p.src "$OUT/via-cat.txt" && echo cat_ok || echo cat_bad
  rm -rf "$OUT" /tmp/p.src'
```

---

*Auditoria read-only. Nenhum script foi editado. Nenhum commit criado.*
