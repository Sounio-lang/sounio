<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave13-tip-green-2026-07-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave13-tip-green-2026-07-21
-->

# Madaros Wave13 — tip-green lock (cd_exact required)

**Date:** 2026-07-21  
**Role:** Wave13 Agent C — tip-green promotion after #1392  
**Branch:** `test/madaros-wave13-showcase`  
**Engine:** default `bin/souc` → Madaros v0.80.0  

## What changed vs Wave12

Wave12 tip-green locked nine science/compiler gates and listed `cd_exact_generic_i64_elf` under **claims_not_made**.

Wave13 tip-green **adds** a tenth **required** gate:

| # | Gate | Script | Sentinel |
|---|------|--------|----------|
| 10 | `cd_exact` | `scripts/dev/madaros_cd_exact_generic_i64_gate.sh` | `MADAROS_CD_EXACT_GENERIC_I64_GATE_OK` |

Science tokens required inside that gate: `ZD PROVED`, `SQ PASS`, `NONZERO PASS`, 16× `COMP i 0`.

Anchor: [PR #1392](https://github.com/Sounio-lang/sounio/pull/1392).

## One-command lock

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE
ulimit -s unlimited 2>/dev/null || true

bash scripts/dev/madaros_wave13_tip_green_gate.sh
# MADAROS_WAVE13_TIP_GREEN_GATE_OK
# receipt: artifacts/compiler/madaros_wave13_tip_green_receipt.v1.json
```

## Prebuilt lag

If stock `bin/madaros-linux-x86_64` predates #1392:

```bash
scripts/dev/souc-build-lock.sh make build-madaros
MADAROS_RAW_BIN=artifacts/self-hosted/madaros bash scripts/dev/madaros_wave13_tip_green_gate.sh
```

## Claims when green

Includes Wave12 tip claims **plus**:

- `cd_exact_generic_i64_elf`
- `cd_exact_zd_proved_pr1392`

Still **not** claimed: full residual census closed, all dual pairs, language-level `Knowledge<T>` import, full linalg parity.

## Related

- Public showcase: `docs/audit/MADAROS_WAVE13_SHOWCASE_2026-07-21.md`
- Wave12 tip-green: `docs/audit/MADAROS_WAVE12_TIP_GREEN_2026-07-21.md`
