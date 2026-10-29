# MoE mega-kernel + GPU combine + GGUF Q5/Q8 Implementation Plan (Plan H)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **NO COMMIT without human "bora".** Propose commit messages only. Assisted-by line when human commits.

**Goal:** Ship mega-kernel B (gate+up SiLU + down + weighted combine on device) for Q4_K and Q5_K, a Q8_0 mega or documented GPU-combine mixed path, wire `moe_ffn_*` to prefer mega, and extend GGUF bind to Q5_K + Q8_0 with L4 prefill demos.

**Architecture:** One CUDA mega entry per quant family writes `dst[token, hidden]` from packs + `x` + `ids` + `weights`. Hot path has no host `moe_silu_mul` / `moe_weighted_sum`. Fallback = Plan G GUS (Q4) or two-mmid + host SiLU, then mmid down + host combine. GGUF reuses Plan G reader; bind Q5/Q8 packs into the same stride contract.

**Tech Stack:** Sounio `.sio`, CUDA/PTX sm_89, hetero runtime, L4 `srun --gres=gpu:1` on gpuorangefs-r770-proxmox, branch `cursor/moe-f32-gguf-silu`.

**Spec:** `docs/superpowers/specs/2026-10-29-moe-f32-mega-combine-design.md`

## Global Constraints

- No llama.cpp runtime linkage or headers.
- Approach B mega-kernel required for Q4_K and Q5_K.
- Q8_0: prefer mega; if blocked, mixed path (act on device + mmid Q8 down + GPU elementwise weighted combine) with explicit plan/receipt note. No host combine on the claimed hot path.
- Fallback when mega PTX missing: Plan G path for Q4 (GUS + mmid down + host combine); Q5/Q8 keep two-mmid + host SiLU + host combine until mega/mixed ready.
- Lab NMSE(dst) <= 5e-4 vs host/CPU oracle; amax=1e5.
- Microbench: report mega us vs Plan G pipeline (GUS+down+host for Q4); meta mega <= pipeline.
- Prefill demos `ntok=1` and `ntok=4`; finite, no NaN; print DONE then exit 0.
- L4 with `srun --gres=gpu:1`.
- ASCII only (no emdash, unicode arrows).
- Commits human-gated. Coord lane `moe-f32-gguf-silu`. Never merge `main`.
- Notion prefix `SIO-moe-f32`.

## File map

| File | Role |
|------|------|
| `stdlib/hetero/kernels/moe_ffn_mega_q4k_f32.cu` | Mega Q4_K + lab host |
| `stdlib/hetero/kernels/moe_ffn_mega_q4k_f32.ptx` | Built PTX |
| `stdlib/hetero/kernels/moe_ffn_mega_q5k_f32.cu/.ptx` | Mega Q5_K twin |
| `stdlib/hetero/kernels/moe_ffn_mega_q8_0_f32.cu/.ptx` **or** `moe_weighted_sum_f32.cu` + reuse mmid Q8 | Q8 path |
| `scripts/gpu/build_moe_ffn_mega_q*_f32.sh` | nvcc PTX (+ lab) |
| `scripts/gpu/run_moe_ffn_mega_q*_l4_lab.sh` | L4 NMSE + us |
| `stdlib/hetero/runtime.sio` | `mega_q4k_func` / `mega_q5k_func` / `mega_q8_0_func` (or combine_func); load + launch |
| `stdlib/hetero/moe.sio` | `moe_ffn_q4k` / `q5k` / `q8_0` prefer mega |
| `examples/hetero_moe_ffn_mega_demo.sio` | FFN smoke Q4 (+ optional Q5/Q8 argv) |
| `stdlib/llm/gguf.sio` | Type tags for Q5_K / Q8_0 if needed |
| `stdlib/llm/llama.sio` | `llama_load_gguf` bind Q5/Q8; forward dispatch |
| `scripts/llm/gen_tiny_mixtral_gguf.py` | Multi-quant fixture flags |
| `examples/llama_gguf_forward_demo.sio` | Prefill Q4/Q5/Q8 fixtures |
| `docs/design/moe-f32-mmid.md` | Plan H receipts |
| `changelog.d/moe-f32-mega-combine.added.md` | Fragment |

---

### Task 1: Mega Q4_K CUDA + L4 lab

**Files:**
- Create: `stdlib/hetero/kernels/moe_ffn_mega_q4k_f32.cu`
- Create: `scripts/gpu/build_moe_ffn_mega_q4k_f32.sh`
- Create: `scripts/gpu/run_moe_ffn_mega_q4k_l4_lab.sh`
- Test: L4 lab; expect `LAB_EXIT:0` and NMSE lines

**Interfaces:**
- Consumes: Q4_K `block_q4_K` + dequant helpers from `moe_gate_up_silu_q4k_f32.cu` / `mmid_q4k_dp4a_f32.cu`
- Produces: entry `moe_ffn_mega_q4k_f32` writing `dst` f32; lab under `-DMOE_MEGA_LAB_HOST`

- [ ] **Step 1: Skeleton CU**

```cuda
extern "C" __global__ void moe_ffn_mega_q4k_f32(
    const char * gate_w, const char * up_w, const char * down_w,
    const float * x, const int * ids, const float * weights,
    float * dst,
    int hidden, int inter, int experts, int slots, int ntokens,
    size_t expert_stride_g, size_t row_stride_g,
    size_t expert_stride_u, size_t row_stride_u,
    size_t expert_stride_d, size_t row_stride_d,
    size_t x_nb1, size_t ids_nb1, size_t w_nb1, size_t dst_nb1);
```

Per token: accumulate into `dst[token, :]`. For each slot: expert = ids[token,slot]; build act[inter] via silu(dot_q4k(gate))*dot_q4k(up); for each out row: dst[o] += weights[token,slot] * dot_q4k(down row, act). SiLU = `g/(1.f+expf(-g))`. Start with small lab shapes (hidden=256, inter=256, experts=4, slots=2, ntokens=1 and 4). Conservatively tile; shared mem optional day-1.

- [ ] **Step 2: Lab host oracle + NMSE + baseline us**

CPU oracle: same math as host `moe_silu_mul` + `moe_weighted_sum` with CPU Q4 dequant+dot (reuse GUS lab oracle style). Time a pipeline baseline labeled clearly (GUS + mmid down + host combine, or CPU pipeline). Print:

```text
n=<ntokens> NMSE=<val> mega_us=<val> pipeline_us=<val>
LAB_EXIT:0
```

Gate: all NMSE <= 5e-4, amax=1e5. Fail lab if any NaN.

- [ ] **Step 3: Build + run scripts**

Mirror `scripts/gpu/build_moe_gate_up_silu_q4k_f32.sh` and `run_moe_gate_up_silu_q4k_l4_lab.sh` with `srun --gres=gpu:1`.

- [ ] **Step 4: Run L4 lab**

```bash
bash scripts/gpu/run_moe_ffn_mega_q4k_l4_lab.sh
```

Expected: `LAB_EXIT:0`, NMSE <= 5e-4, mega_us and pipeline_us printed.

- [ ] **Step 5: Propose commit only**

```
hetero: moe_ffn_mega_q4k mega-kernel + L4 lab

Assisted-by: Composer
```

---

### Task 2: Runtime load + launch mega Q4

**Files:**
- Modify: `stdlib/hetero/runtime.sio`
- Test: `bin/souc check examples/hetero_moe_gate_up_silu_demo.sio`

**Interfaces:**
- Produces: `Runtime.mega_q4k_func: i64`; placeholders `mega_q5k_func`, `mega_q8_0_func`; `rt_load_moe_ffn_mega(rt, ptx) -> bool`; `rt_moe_ffn_mega_q4k(...) -> MmidResult`

- [ ] **Step 1: Extend Runtime**

Add `mega_q4k_func: i64`, `mega_q5k_func: i64`, `mega_q8_0_func: i64` (init 0). Update every `Runtime { ... }` that lists `gus_q4k_func` (search `gus_q4k_func`).

- [ ] **Step 2: Loader**

```sio
pub fn rt_load_moe_ffn_mega(rt: &!Runtime, ptx_path: string) -> bool with Alloc, Mut, Panic, Div, IO {
    if rt.mega_q4k_func != 0 { return true }
    // read_file; ce_module_load; ce_function(..., "moe_ffn_mega_q4k_f32"); set mega_q4k_func
}
```

- [ ] **Step 3: Launcher**

```sio
pub fn rt_moe_ffn_mega_q4k(
    rt: &Runtime,
    gate_w: *mut u8, gate_bytes: i64,
    up_w: *mut u8, up_bytes: i64,
    down_w: *mut u8, down_bytes: i64,
    x: *mut u8, x_bytes: i64,
    ids: *mut u8, ids_bytes: i64,
    weights: *mut u8, weights_bytes: i64,
    dst: *mut u8, dst_bytes: i64,
    hidden: i64, inter: i64, experts: i64, slots: i64, n_tokens: i64,
    expert_stride_g: i64, row_stride_g: i64,
    expert_stride_u: i64, row_stride_u: i64,
    expert_stride_d: i64, row_stride_d: i64,
    x_nb1: i64, ids_nb1: i64, w_nb1: i64, dst_nb1: i64
) -> MmidResult with Alloc, Mut, Panic, Div
```

H2D packs/x/ids/weights, device dst, launch, sync, D2H. If `mega_q4k_func==0` or `!rt.exec.ok`, return `ok=false`, status -1.

- [ ] **Step 4: Typecheck**

```bash
bin/souc check examples/hetero_moe_gate_up_silu_demo.sio
```

Expected: `check: OK`.

- [ ] **Step 5: Propose commit only**

```
hetero: runtime load/launch moe_ffn_mega_q4k

Assisted-by: Composer
```

---

### Task 3: Wire `moe_ffn_q4k` prefer-mega + hetero demo

**Files:**
- Modify: `stdlib/hetero/moe.sio`
- Create: `examples/hetero_moe_ffn_mega_demo.sio`
- Test: check + L4 demo

**Interfaces:**
- Consumes: `rt_moe_ffn_mega_q4k`, existing GUS fallback path
- Produces: `moe_ffn_q4k` writes `dst` via mega when available

- [ ] **Step 1: Prefer mega in moe_ffn_q4k**

```sio
use hetero::runtime::{..., rt_moe_ffn_mega_q4k, rt_moe_gate_up_silu_q4k}

// at start of moe_ffn_q4k, before GUS:
let mega = rt_moe_ffn_mega_q4k(
    rt, gate_w, gate_bytes, up_w, up_bytes, down_w, down_bytes,
    x, n_tokens * hidden * 4, ids, n_tokens * slots * 4,
    route_w, n_tokens * slots * 4,
    dst, n_tokens * hidden * 4,
    hidden, inter, experts, slots, n_tokens,
    expert_stride_g, row_stride_g, expert_stride_u, row_stride_u,
    expert_stride_d, row_stride_d,
    hidden * 4, slots * 4, slots * 4, hidden * 4
)
if mega.ok {
    res.ms_gate = mega.ms
    res.ms_up = 0.0
    res.ms_down = 0.0
    res.ok = true
    res.status = 0
    return res
}
// else existing GUS / two-mmid + mmid down + moe_weighted_sum
```

- [ ] **Step 2: Demo**

Args: `[gemm-ptx] [mmid-ptx] [gus-ptx] [mega-ptx]`. Load mmid + gus + mega. Call `moe_ffn_q4k`. Require ok + finite dst. Print `MOE MEGA DONE`.

- [ ] **Step 3: Verify**

```bash
bin/souc check examples/hetero_moe_ffn_mega_demo.sio examples/hetero_moe_gate_up_silu_demo.sio
# L4: tar sources + srun --gres=gpu:1 run demo
```

Expected: check OK; L4 `MOE MEGA DONE` RC=0.

- [ ] **Step 4: Propose commit only**

```
hetero: moe_ffn_q4k prefers mega-kernel

Assisted-by: Composer
```

---

### Task 4: Mega Q5_K twin + wire `moe_ffn_q5k`

**Files:**
- Create: `stdlib/hetero/kernels/moe_ffn_mega_q5k_f32.cu` (+ build/run scripts)
- Modify: `stdlib/hetero/runtime.sio` (`mega_q5k_func`, `rt_load_moe_ffn_mega_q5k`, `rt_moe_ffn_mega_q5k`)
- Modify: `stdlib/hetero/moe.sio` (`moe_ffn_q5k` prefer mega)
- Test: L4 lab Q5 + check + optional demo argv

**Interfaces:**
- Same launcher shape as Q4 mega; Q5_K block layout from `mmid_q5k_dp4a_f32.cu`

- [ ] **Step 1: CU + lab**

Copy-adapt Task 1 with Q5 dequant. Same print contract; NMSE <= 5e-4.

```bash
bash scripts/gpu/run_moe_ffn_mega_q5k_l4_lab.sh
```

Expected: `LAB_EXIT:0`.

- [ ] **Step 2: Runtime**

Prefer separate load if PTX files stay separate (Plan G style):

```sio
pub fn rt_load_moe_ffn_mega_q5k(rt: &!Runtime, ptx_path: string) -> bool ...
pub fn rt_moe_ffn_mega_q5k(/* same args as q4k */) -> MmidResult ...
```

- [ ] **Step 3: Wire moe_ffn_q5k**

Prefer `rt_moe_ffn_mega_q5k`; fallback existing two `rt_mul_mat_id_q5k` + `moe_silu_mul` + down + `moe_weighted_sum`.

- [ ] **Step 4: Typecheck + L4 smoke**

```bash
bin/souc check examples/hetero_moe_q5k_demo.sio examples/hetero_moe_ffn_mega_demo.sio
```

Expected: check OK; L4 DONE when mega-q5 PTX passed.

- [ ] **Step 5: Propose commit only**

```
hetero: moe_ffn_mega_q5k twin + moe_ffn_q5k wire

Assisted-by: Composer
```

---

### Task 5: Q8_0 mega or mixed GPU-combine path

**Files:**
- Prefer create: `stdlib/hetero/kernels/moe_ffn_mega_q8_0_f32.cu` (+ scripts)
- Or create: `stdlib/hetero/kernels/moe_weighted_sum_f32.cu` (GPU combine) and keep mmid Q8 for down
- Modify: `runtime.sio`, `moe.sio` (`moe_ffn_q8_0`)
- Test: L4 lab; document choice in receipts

**Decision rule (implementer must pick one and write it in `docs/design/moe-f32-mmid.md`):**
- If Q8 mega lab greens within same shape family as Q4: ship mega.
- Else: mixed path only - act via two `rt_mul_mat_id_q8_0` (+ SiLU into act), mmid Q8 down on device, then `moe_weighted_sum_f32` GPU kernel into dst. Host `moe_weighted_sum` must not run on the claimed hot path when combine PTX loaded.

- [ ] **Step 1: Implement chosen path + lab**

NMSE(dst) <= 5e-4. Print path tag `PATH=mega` or `PATH=mixed`.

- [ ] **Step 2: Runtime + moe_ffn_q8_0 prefer**

Load symbols; early-return on success like Q4 mega wire.

- [ ] **Step 3: Verify L4**

```bash
bash scripts/gpu/run_moe_ffn_mega_q8_0_l4_lab.sh
# or run_moe_weighted_sum_l4_lab.sh + q8 ffn smoke
bin/souc check examples/hetero_moe_q8_0_demo.sio
```

Expected: lab LAB_EXIT:0; check OK.

- [ ] **Step 4: Propose commit only**

```
hetero: moe_ffn_q8_0 mega or GPU-combine mixed path

Assisted-by: Composer
```

---

### Task 6: GGUF Q5_K + Q8_0 bind + multi-quant fixture

**Files:**
- Modify: `stdlib/llm/gguf.sio` (type constants / checks as needed)
- Modify: `stdlib/llm/llama.sio` (`llama_load_gguf` bind; `moe_quant` values)
- Modify: `scripts/llm/gen_tiny_mixtral_gguf.py`
- Test: generate fixtures; `bin/souc check examples/llama_gguf_load_smoke.sio`

**Interfaces:**
- Align with existing `moe_quant` enum in `llama.sio` (Plan E/F: typically 0=Q4_K, 1=Q5_K; assign Q8_0 to next free id if unused). Do not renumber silently.
- On type mismatch: clear error; no partial bind. Reject mixed gate/up/down types in one layer.

- [ ] **Step 1: Read current moe_quant mapping**

```bash
rg -n "moe_quant" stdlib/llm/llama.sio docs/design/moe-f32-mmid.md
```

- [ ] **Step 2: Fixture generator**

Extend `gen_tiny_mixtral_gguf.py` with `--quant q4k|q5k|q8_0` (default q4k). Write e.g. `tiny-mixtral-moe-q5k.gguf`, `tiny-mixtral-moe-q8_0.gguf`. Keep hidden/inter divisible by 256 for QK.

- [ ] **Step 3: llama_load_gguf bind**

When MoE tensor ggml type is Q5_K / Q8_0, set packs + `cfg.moe_quant` accordingly.

- [ ] **Step 4: Smoke**

```bash
python3 scripts/llm/gen_tiny_mixtral_gguf.py --quant q5k --out /tmp/tiny-mixtral-moe-q5k.gguf
python3 scripts/llm/gen_tiny_mixtral_gguf.py --quant q8_0 --out /tmp/tiny-mixtral-moe-q8_0.gguf
bin/souc check examples/llama_gguf_load_smoke.sio
```

Expected: files written; check OK.

- [ ] **Step 5: Propose commit only**

```
llm: GGUF MoE bind Q5_K and Q8_0 + multi-quant fixtures

Assisted-by: Composer
```

---

### Task 7: Llama GGUF forward demos (prefill)

**Files:**
- Modify: `examples/llama_gguf_forward_demo.sio` (and/or sibling demos)
- Modify: `stdlib/llm/llama.sio` if forward must load mega PTX / dispatch q8
- Test: L4 ntok=1 and ntok=4 for Q4, Q5, Q8 fixtures

- [ ] **Step 1: Ensure forward loads mega PTX when present**

Same pattern as GUS load in Plan G demos. If mega load fails, fallback allowed; print which path ran.

- [ ] **Step 2: Dispatch**

Add q8 branch calling `moe_ffn_q8_0` if missing from current `moe_quant` switch.

- [ ] **Step 3: L4 runs**

For each fixture quant, run demo with `ntok=1` and `ntok=4` via `srun --gres=gpu:1`. Expected: DONE, RC=0, finite logits, `last_moe_layers_ok > 0` when packs+GPU ready. Real GGUF optional: missing = skip; fixture required green.

- [ ] **Step 4: Propose commit only**

```
llm: llama_gguf forward demos Q5/Q8 prefill on mega path

Assisted-by: Composer
```

---

### Task 8: Docs + changelog + Notion

**Files:**
- Modify: `docs/design/moe-f32-mmid.md` (Plan H section)
- Create: `changelog.d/moe-f32-mega-combine.added.md`
- Notion card under `SIO-moe-f32`

- [ ] **Step 1: Receipts**

Append Plan H: approach B, lab NMSE tables (Q4/Q5/Q8), microbench mega vs pipeline, Q8 path choice, GGUF fixture commands, demo receipts.

- [ ] **Step 2: Changelog fragment**

```markdown
### Added
- MoE FFN mega-kernel (gate+up SiLU+down+combine) Q4_K/Q5_K and Q8 path; GGUF Q5/Q8 MoE bind
```

- [ ] **Step 3: Notion**

Create/update card linking spec + plan + receipts. Title sketch: `SIO-moe-f32 Plan H mega-combine`.

- [ ] **Step 4: Propose commit only**

```
docs: Plan H mega-combine receipts + changelog

Assisted-by: Composer
```

---

## Self-review (plan author)

- Spec coverage: mega Q4/Q5, Q8 mega-or-mixed, runtime, moe_ffn wire, GGUF Q5/Q8, demos, labs, docs/Notion/changelog - Tasks 1-8.
- No TBD/TODO placeholders in steps.
- Signatures: `rt_moe_ffn_mega_q4k` / `_q5k` / Q8 variant aligned; `Runtime.mega_*_func`; SiLU formula matches Plan G.
- Human-gated commits throughout.

## Execution handoff

After human approval of this plan file, choose:

1. **Subagent-Driven (recommended)** - fresh subagent per task + review between tasks (`superpowers:subagent-driven-development`)
2. **Inline Execution** - this session with `superpowers:executing-plans` checkpoints
