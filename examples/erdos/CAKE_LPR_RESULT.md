# G_529 LRAT verification with cake_lpr

Formally verified check that Heule's 529-vertex de Grey graph `G_529` is **not**
4-colourable, using the CakeML/HOL4 machine-code LRAT checker `cake_lpr`.

Reproduce end-to-end:

```bash
chmod +x examples/erdos/verify_lrat_cake.sh
examples/erdos/verify_lrat_cake.sh
```

## Environment

- Date: 2026-05-29
- Host: Linux x86-64
- Graph: `examples/erdos/data/degrey_529.edge` (`p edge 529 2670`)
- Solver: `examples/erdos/souc_sat.sio` compiled with
  `artifacts/self-hosted/souc-self-hosted-x86_64` (via `SOUNIO_SOUC_BIN`)

## Step 1–2: generate DRAT certificate

```bash
export SOUNIO_SOUC_BIN="$(pwd)/artifacts/self-hosted/souc-self-hosted-x86_64"
"$SOUNIO_SOUC_BIN" examples/erdos/souc_sat.sio /tmp/souc_sat.elf
chmod +x /tmp/souc_sat.elf
mkdir -p /tmp/cake && cd /tmp/cake
/tmp/souc_sat.elf 0 4 1 1 /workspace/sounio/examples/erdos/data/degrey_529.edge
```

Solver output (stdout):

```
[worker seed=0 UNSAT conflicts=327208 restarts=2123 lemmas=638767 -> souc_sat_worker.cnf/.drat (streamed)]
```

Artifacts:

| File | Size |
|------|-----:|
| `souc_sat_worker.cnf` | 147 000 bytes (2116 vars, 11212 clauses) |
| `souc_sat_worker.drat` | 71 762 225 bytes |

## Step 3–4: drat-trim DRAT verify + LRAT generation

```bash
cd /tmp && git clone --depth 1 https://github.com/marijnheule/drat-trim.git dt
gcc -O2 dt/drat-trim.c -o /tmp/dtrim
cd /tmp/cake
/tmp/dtrim souc_sat_worker.cnf souc_sat_worker.drat -L g529.lrat 2>&1 | tr '\r' '\n' | tee dtrim.log
```

**drat-trim verdict:** `s VERIFIED` (DRAT proof valid; verification time ~5.1 s)

LRAT output: `g529.lrat` — **36 207 727 bytes** (~34.5 MiB)

## Step 5: cake_lpr source

No prebuilt x86-64 download URL was found; the official release path is the
**pre-extracted assembly** repository (not a from-scratch CakeML/HOL4 bootstrap):

```bash
git clone --depth 1 https://github.com/tanyongkiam/cake_lpr.git /tmp/cake_lpr_repo
make -C /tmp/cake_lpr_repo cake_lpr
```

| Field | Value |
|-------|-------|
| Repository | https://github.com/tanyongkiam/cake_lpr |
| Commit | `a4323b203cc9ecd584ba7da9e3fff08135a09d5f` (2026-03-16) |
| Build | `gcc -O2 basis_ffi.c cake_lpr.S -o cake_lpr -std=c99` |
| Extracted from CakeML | `fb377b4bb704497c921cde68ccc8da3b4f0e9132` (per repo README) |
| HOL4 proof revision | `0ae7030322cdf2b0d46dc9d5503e2d5eae2fa726` |
| Binary size | 571 776 bytes |

Formal proofs live in CakeML:
https://github.com/CakeML/cakeml/tree/master/examples/lpr_checker

## Step 6: cake_lpr verified LRAT check

```bash
export CML_HEAP_SIZE=65536
export CML_STACK_SIZE=16384
/tmp/cake_lpr_repo/cake_lpr /tmp/cake/souc_sat_worker.cnf /tmp/cake/g529.lrat
```

**cake_lpr verdict:** `s VERIFIED UNSAT` (wall time ~1.9 s)

Full stdout:

```
s VERIFIED UNSAT
```

## Summary

| Stage | Tool | Result |
|-------|------|--------|
| SAT solve | `souc_sat.sio` | UNSAT |
| DRAT check | drat-trim (unverified C) | `s VERIFIED` |
| LRAT emit | drat-trim `-L` | 36 207 727 bytes |
| **Verified check** | **cake_lpr** (CakeML extraction) | **`s VERIFIED UNSAT`** |

This closes the SAT leg of the χ(R²) ≥ 5 certificate for G_529 with a
machine-checked proof checker (trusted computing base: HOL4/CakeML extraction
pipeline, not drat-trim).

## Blockers

None in this environment. If `cake_lpr` fails with `CakeML heap space exhausted`,
raise `CML_HEAP_SIZE` / `CML_STACK_SIZE` (see `verify_lrat_cake.sh` defaults).
