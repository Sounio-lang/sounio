# MoE GGUF + fused gate/up SiLU Implementation Plan (Plan G)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Q4_K fused gate+up+SiLU super-kernel into `moe_ffn_*`, plus hybrid GGUF load (fixture required, real optional) with prefill demos on L4.

**Architecture:** Approach B: one CUDA kernel dots gate and up Q4_K rows against `x` and writes `act = silu(g)*u` on device. `moe_ffn_q4k` calls that instead of two `mmid` + host `moe_silu_mul`, then keeps `mmid` down + host combine. Fallback to the old path if PTX missing. GGUF reader fills the same pack/stride contract as `llama_load_moe`.

**Tech Stack:** Sounio `.sio`, CUDA/PTX sm_89, hetero runtime, L4 `srun --gres=gpu:1`, branch `cursor/moe-f32-gguf-silu`.

**Spec:** `docs/superpowers/specs/2026-10-29-moe-f32-gguf-silu-design.md`

## Global Constraints

- No llama.cpp runtime linkage or headers.
- Super-kernel is Q4_K first; Q5_K twin only if copy-adapt from Q4 stays green, else defer with written note (do not block GGUF on Q5).
- Fallback two-mmid + host `moe_silu_mul` when super-kernel unavailable; never silent wrong path.
- GGUF MoE packs required quant: Q4_K; other quants = skip/blocker text.
- Fixture GGUF must pass; real GGUF optional with honest skip.
- Lab NMSE(act) <= 5e-4 vs CPU oracle; amax=1e5.
- L4 with `srun` and `gpu:1`; demos print DONE then `exit(0)`.
- ASCII only in code/docs (no emdash, unicode arrows).
- Commits human-gated: propose message only unless human says commit. `Assisted-by:`.
- Coord lane `moe-f32-gguf-silu`; never merge `main`.
- Notion prefix `SIO-moe-f32` (controller for card).

## File map

| File | Role |
|------|------|
| `stdlib/hetero/kernels/moe_gate_up_silu_q4k_f32.cu` | Super-kernel + optional lab host |
| `stdlib/hetero/kernels/moe_gate_up_silu_q4k_f32.ptx` | Built PTX |
| `scripts/gpu/build_moe_gate_up_silu_q4k_f32.sh` | nvcc PTX (+ lab host) |
| `scripts/gpu/run_moe_gate_up_silu_q4k_l4_lab.sh` | L4 lab runner |
| `stdlib/hetero/runtime.sio` | `gus_q4k_func`, load + launch |
| `stdlib/hetero/moe.sio` | `moe_ffn_q4k` uses super-kernel + fallback |
| `examples/hetero_moe_gate_up_silu_demo.sio` | FFN smoke |
| `stdlib/llm/gguf.sio` | GGUF reader |
| `stdlib/llm/llama.sio` | `llama_load_gguf` |
| `stdlib/llm/mod.sio` | exports |
| `scripts/llm/gen_tiny_mixtral_gguf.py` | Fixture GGUF |
| `examples/llama_gguf_forward_demo.sio` | Load + ntok 1/4 |
| `docs/design/moe-f32-mmid.md` | Plan G receipts |
| `changelog.d/moe-f32-gguf-silu.added.md` | Fragment |

---

### Task 1: Super-kernel CUDA + L4 lab (Q4_K)

**Files:**
- Create: `stdlib/hetero/kernels/moe_gate_up_silu_q4k_f32.cu`
- Create: `scripts/gpu/build_moe_gate_up_silu_q4k_f32.sh`
- Create: `scripts/gpu/run_moe_gate_up_silu_q4k_l4_lab.sh`
- Test: L4 lab; expect `LAB_EXIT:0` and NMSE lines

**Interfaces:**
- Consumes: Q4_K `block_q4_K` layout and dequant helpers from `mmid_q4k_dp4a_f32.cu`
- Produces: entry `moe_gate_up_silu_q4k_f32` writing `act` f32; lab oracle under `-DMOE_GUS_LAB_HOST`

- [x] **Step 1: Skeleton CU with shared Q4_K helpers**

Copy `block_q4_K`, `get_scale_min_k4`, and a scalar row-dot helper from `mmid_q4k_dp4a_f32.cu`. Do not include llama.cpp.

```cuda
extern "C" __global__ void moe_gate_up_silu_q4k_f32(
    const char * gate_w, const char * up_w, const float * x, const int * ids,
    float * act, int hidden, int inter, int experts, int slots, int ntokens,
    size_t expert_stride_g, size_t row_stride_g,
    size_t expert_stride_u, size_t row_stride_u,
    size_t x_nb1, size_t ids_nb1, size_t act_nb1, size_t act_nb2);
```

For each `(token, slot, row)`: `act = silu(dot_q4k(gate)) * dot_q4k(up)` with `silu(g)=g/(1+expf(-g))`.

- [x] **Step 2: Lab host oracle + NMSE**

CPU oracle: dequant+dot gate, dequant+dot up, SiLU*mul. Gate NMSE <= 5e-4, amax=1e5. Include `ntokens>1`. Print:

```text
n=<ntokens> NMSE=<val> us/run=<val> baseline_us=<val>
LAB_EXIT:0
```

- [x] **Step 3: Build + run scripts**

Mirror `scripts/gpu/build_mmid_q4k_dp4a_f32.sh` and `run_mmid_q4k_l4_lab.sh` with `srun --gres=gpu:1`.

- [x] **Step 4: Run L4 lab**

```bash
bash scripts/gpu/run_moe_gate_up_silu_q4k_l4_lab.sh
```

Expected: `LAB_EXIT:0`, all NMSE <= 5e-4.

- [ ] **Step 5: Propose commit only**

```
hetero: moe_gate_up_silu_q4k super-kernel + L4 lab

Assisted-by: Composer
```

---

### Task 2: Runtime load + launch

**Files:**
- Modify: `stdlib/hetero/runtime.sio`
- Test: `bin/souc check examples/hetero_moe_layer_demo.sio`

**Interfaces:**
- Produces: `Runtime.gus_q4k_func: i64`; `rt_load_moe_gate_up_silu(rt, ptx) -> bool`; `rt_moe_gate_up_silu_q4k(...) -> MmidResult`

- [x] **Step 1: Extend Runtime**

Add `gus_q4k_func: i64`. Update every `Runtime { ... }` literal that lists mmid fields (search `mmid_q8_0_vec_func`).

- [x] **Step 2: Loader**

```sio
pub fn rt_load_moe_gate_up_silu(rt: &!Runtime, ptx_path: string) -> bool with Alloc, Mut, Panic, Div, IO {
    // read_file; ce_module_log; ce_function(..., "moe_gate_up_silu_q4k_f32"); set gus_q4k_func
}
```

- [x] **Step 3: Launcher**

H2D gate/up/x/ids, device act, launch, sync, D2H. If `gus_q4k_func==0` or `!rt.exec.ok`, return `ok=false` status -1.

- [x] **Step 4: Typecheck**

```bash
bin/souc check examples/hetero_moe_layer_demo.sio
```

Expected: `check: OK`.

- [ ] **Step 5: Propose commit only**

```
hetero: runtime load/launch moe_gate_up_silu_q4k

Assisted-by: Composer
```

---

### Task 3: Wire `moe_ffn_q4k` + hetero demo

**Files:**
- Modify: `stdlib/hetero/moe.sio`
- Create: `examples/hetero_moe_gate_up_silu_demo.sio`
- Test: check + L4 demo

**Interfaces:**
- `moe_ffn_q4k` prefers super-kernel; fallback two mmid + `moe_silu_mul`; down + weighted_sum unchanged

- [x] **Step 1: Prefer super-kernel**

```sio
let gus = rt_moe_gate_up_silu_q4k(/* packs, x, ids, act, dims, strides */)
if gus.ok {
    res.ms_gate = gus.ms
    res.ms_up = 0.0
} else {
    // existing mmid gate + mmid up + moe_silu_mul into act
}
// mmid down(act) + moe_weighted_sum
```

- [x] **Step 2: Demo**

Args `[gemm-ptx] [mmid-ptx] [gus-ptx]`. Load mmid (down) + gus. Call `moe_ffn_q4k`. Require ok + finite. Print `MOE GUS DONE`.

- [x] **Step 3: Verify**

```bash
bin/souc check examples/hetero_moe_gate_up_silu_demo.sio examples/llama_moe_ffn_demo.sio
# L4 tar+srun --gres=gpu:1
```

Expected: check OK; L4 DONE RC=0.

- [ ] **Step 4: Propose commit only**

```
hetero: moe_ffn_q4k uses gate/up SiLU super-kernel

Assisted-by: Composer
```

---

### Task 4: Q5_K twin or defer note

**Files:**
- Optional: Q5 kernel/runtime/`moe_ffn_q5k` wire
- Or docs defer paragraph in `docs/design/moe-f32-mmid.md`

- [x] **Step 1: Decide by cost**

If Q4 green and Q5 copy is mechanical, ship twin. Else write: "Q5_K super-kernel deferred; `moe_ffn_q5k` keeps two-mmid + host SiLU".

- [x] **Step 2: Verify**

Q5 lab green or defer note present; `bin/souc check examples/llama_moe_ffn_demo.sio`.

- [ ] **Step 3: Propose commit only**

```
hetero: moe_gate_up_silu Q5 twin
```
or
```
docs: defer Q5 gate/up SiLU super-kernel
```

---

### Task 5: GGUF reader module

**Files:**
- Create: `stdlib/llm/gguf.sio`
- Modify: `stdlib/llm/mod.sio`
- Test: probe demo or check after export

**Interfaces:**
- `GgufFile`, `gguf_open(path)`, `gguf_find(f, name) -> GgufTensor`, raw byte slice for Q4_K packs
- Support GGUF v3; dtypes F32, F16, Q4_K minimum

- [x] **Step 1: Parse header + tensor index**

Magic `GGUF`, version, tensor_count, kv_count, skip KV, read tensor infos, data offsets.

- [x] **Step 2: Export from mod.sio**

- [x] **Step 3: Typecheck**

```bash
bin/souc check examples/llama_load_demo.sio
```

(or tiny `examples/gguf_probe_demo.sio` opening a missing path with `ok=false`)

- [ ] **Step 4: Propose commit only**

```
llm: GGUF reader (header + tensor index)

Assisted-by: Composer
```

---

### Task 6: Fixture GGUF + `llama_load_gguf`

**Files:**
- Create: `scripts/llm/gen_tiny_mixtral_gguf.py`
- Modify: `stdlib/llm/llama.sio`, `stdlib/llm/mod.sio`
- Test: generate + load smoke

**Interfaces:**
- `llama_load_gguf(path) -> Llama` with Plan F MoE pack fields; fixture hidden=256, moe_inter=512, layers=2, experts=8, slots=2, vocab=64

- [x] **Step 1: Generator**

```bash
python3 scripts/llm/gen_tiny_mixtral_gguf.py --out /workspace/llm-models/tiny-mixtral-moe.gguf
```

- [x] **Step 2: llama_load_gguf**

Fill cfg from KV written by the fixture; bind Q4_K expert tensors; decode attn/embed; set strides like `llama_load_moe`.

- [x] **Step 3: Smoke**

Print `loaded=1 n_experts=8` with packs present.

- [ ] **Step 4: Propose commit only**

```
llm: llama_load_gguf + tiny Mixtral GGUF fixture gen

Assisted-by: Composer
```

---

### Task 7: `llama_gguf_forward_demo` (ntok 1 and 4)

**Files:**
- Create: `examples/llama_gguf_forward_demo.sio`
- Test: check + L4

- [x] **Step 1: Demo**

Default `/workspace/llm-models/tiny-mixtral-moe.gguf`. Missing fixture: `GGUF FIXTURE MISSING` + DONE exit 0. Load gus+mmid; forward ntok=1 and 4; assert finite and `last_moe_layers_ok == layers` when GPU+packs. Optional real GGUF argv with `GGUF REAL SKIP` if absent.

- [x] **Step 2: Verify**

```bash
python3 scripts/llm/gen_tiny_mixtral_gguf.py --out /workspace/llm-models/tiny-mixtral-moe.gguf
bin/souc check examples/llama_gguf_forward_demo.sio
# L4 srun gpu:1
```

Expected: `last_moe_layers_ok=2` both modes, DONE, RC=0.

- [ ] **Step 3: Propose commit only**

```
examples: llama GGUF forward demo (ntok 1 and 4)

Assisted-by: Composer
```

---

### Task 8: Docs + changelog + Notion

**Files:**
- Modify: `docs/design/moe-f32-mmid.md`
- Create: `changelog.d/moe-f32-gguf-silu.added.md`
- Modify: this plan checkboxes
- Notion: controller `SIO-moe-f32-gguf-silu`

- [x] **Step 1: Plan G receipts** (NMSE/us, baseline_us, fixture cmd, demo lines, Q5 twin/defer)

- [x] **Step 2: Changelog fragment**

- [ ] **Step 3: Notion IDs for controller** (controller-owned: `SIO-moe-f32-gguf-silu`; skip in agent lane)

- [ ] **Step 4: Propose docs commit only**

```
docs: Plan G GGUF + fused SiLU receipts

Assisted-by: Composer
```

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| Super-kernel B gate+up+SiLU Q4_K | 1 |
| Runtime load/launch | 2 |
| moe_ffn_q4k wire + fallback | 3 |
| Lab NMSE + microbench | 1, 8 |
| Q5 twin or defer | 4 |
| GGUF reader | 5 |
| llama_load_gguf + fixture | 6 |
| Prefill ntok 1 and 4 demo | 7 |
| Hybrid real GGUF optional | 7 |
| Docs + changelog | 8 |
| No llama.cpp | Global |
