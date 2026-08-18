<!-- docs:meta
topic_id: repo.tests.vectors.f128-f256-v0c.gate-consumption
authority: repo_only
audience: internal+codex
last_validated: 2026-08-17
validated_by: grok-cli3
source_of_truth: scripts/ci/madaros_f128_f256_v0c_wire_gate.sh
-->

# V0-C gate consumption of grok-cli1 wire corpus

**Gate:** `bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0c`  
**Oracle corpus:** `tests/vectors/f128_f256_v0c/wire_f{128,256}.jsonl` (IEEE-754-2008 structural encodings; generator `gen/wire_encoding_gen.c`)

## Engine (named — V0-A taught this matters)

| Path | Engine |
|---|---|
| Scaffold probes (descriptor / payload / wire) | **lean_single seed ELF** (`bin/souc-lean-single-x86_64`), same as existing `madaros_f128_f256_numeric_*_gate.sh` |
| Corpus green / V0-C complete | Requires a **codec consumer** that maps the external JSONL through limb/wire encode–decode — not claimed as lean_single-only and not silent Madaros E218 |

This gate does **not** assert “E218 on Madaros” (that is V0-B). V0-C fails today because the **external corpus is unconsumed**, while scaffolds still pass on hard-coded cases.

## Consumed today

| Asset | How |
|---|---|
| `wire_f128.jsonl` (31) | md5 + structural oracle (`ws_g_v0c_wire_corpus_oracle.py`) |
| `wire_f256.jsonl` (24) | same |
| accept=33 / reject=22 | count check vs README |
| Scaffold probes | positive control — must fire exact PASS receipts |

## Not consumed (why gate is red)

No Sounio/Python runner yet maps each corpus row through `IrWideNumericPayload` / wire codec. Required consumer markers (any one):

- `self-hosted/compiler/f128_f256_v0c_wire_corpus_probe.sio`
- `tests/run-pass/f128_v0c_wire_corpus_smoke.sio`
- `scripts/dev/ws_g_v0c_codec_corpus_runner.py`

Until one exists and the oracle prints `PASS v0c_codec_consumer_present`, the gate **FAIL**s with:

`diagnosis=V0-C_scaffold_alive_but_external_wire_corpus_unconsumed`

## Green receipt (ladder)

```
PASS f128_f256_v0c_wire limbs=8 order=lsw-first payloads=4 wire_bytes=272 roundtrip=exact decode_negative=24 encode_negative=4 checksum=adler32 ir_emit=green soir_bss=green corpus_f128=31 corpus_f256=24 accept=33 reject=22
PASS madaros_f128_f256_ladder_gate stage=v0c
```
