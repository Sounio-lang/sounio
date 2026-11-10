# M5: device paged-attn wired into llm_serve behind explicit flag

## Change
- `llama_forward_batch_runs_rt` takes `pagedev: bool`; retired the
  `LL_PAGED_ATTN_GPU` const-gate (sole use removed with the const).
- Threaded through `svb_pump_fwd`, `svb_pump`, `sv_pump_turn`, `llm_serve`.
  Legacy `sv_turn`/`sv_turn_batch` (non-paged batch engine) untouched.
- `examples/llm_serve`: 5th CLI arg `paged-dev-0/1`, default 0 (host).
- Tests dual: `serve_pump`/`batch_cont` run gpu leg twice
  (pagedev false + true); `paged_runs` rt-leg passes true (PTX-load
  fallback keeps CPU-lane behavior identical to before).

## Receipts
- CPU login pod: SERVEPUMP PASS, BATCHCONT PASS, PAGEDRUNS PASS (rc=0).
- GPU RTX4000 Ada 5860 (`slurm-pilot-worker-gpuorangefs-9kfb5` relay):
  all three PASS; CUDA context proven live during run (CUDA_HIT=1).
- Live `llm_serve` CPU smoke (SmolLM2-135M, port 8093): /health ok,
  /v1/models ok, completions pagedev=0 vs 1 byte-identical text+tokens.

## Notes
- sbatch gpu-orangefs dispatch still broken for fresh jobs; GPU lane
  used srun (r770 probe) + kubectl cp/exec relay to the 5860 worker.
- `sv_turn_batch` stays on the legacy non-paged engine; flag covers
  the pump path (SV_PUMP=1, the live path).
