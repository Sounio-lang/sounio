#!/usr/bin/env python3
"""Real pinned input-embedding GPU reference versus lossless file-backed gather."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from safetensors import safe_open
import offline_generate as offline
import qualify_offline_preparation as preparation
import offload_embedding

def main():
    assert os.environ.get("PIREUS_EMBEDDING_OFFLOAD_PROBE") == "1"
    rank = int(os.environ["PIREUS_RANK"])
    import torch._inductor.config as inductor_config
    if inductor_config.compile_threads != 1:
        raise ValueError("bounded offline compiler worker profile")
    print(json.dumps(dict(stage="INDUCTOR_COMPILE_PROFILE", job=os.environ["SLURM_JOB_ID"],
                         rank=str(rank), compile_threads=inductor_config.compile_threads)), flush=True)
    args = offline.prepare_server_args(sys.argv[1:] + ["--random-seed", "20260907"])
    bench = offline.bench
    bench._set_envs_and_config(args)
    bench.initialize_moe_config(args)
    bench.initialize_fp8_gemm_config(args)
    bench.initialize_fp4_gemm_config(args)
    offline.initialize_bf16_gemm_config(args)
    offline.initialize_mamba_selective_state_update_backend(args)
    preparation.loader.ModelOptModelLoader.load_model = preparation.meta_load
    preparation.loader.DefaultModelLoader.load_model = preparation.meta_load
    runner, cache = offline.load_worker_and_cache(args, rank)
    model = runner.torch_runner
    embed = model.model.llm.embed_tokens
    old = embed.weight
    real = torch.nn.Parameter(torch.empty(old.shape, dtype=old.dtype, device="cuda"), requires_grad=False)
    real.__dict__.update(old.__dict__)
    embed.weight = real
    del old, real
    root = Path(args.model_path)
    key = "model.llm.embed.weight"
    index = json.loads((root/"model.safetensors.index.json").read_bytes())["weight_map"]
    with safe_open(root/index[key], framework="pt", device="cpu") as f:
        source = f.get_tensor(key)
        embed.weight_loader(embed.weight, source)
        del source
    comm = model.tp_group.pynccl_comm
    assert comm is not None and comm.available
    vocab = embed.num_embeddings
    def run(ids):
        with torch.no_grad(), comm.change_state(enable=True):
            output = embed(ids.to("cuda")).cpu()
        return output
    def rawhash(value):
        return hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest()
    # Every vocabulary row is compared to the real fused GPU embedding path.
    references = []
    for start in range(0, vocab, 512):
        references.append(rawhash(run(torch.arange(start, min(vocab, start+512)))))
    generator = torch.Generator().manual_seed(20260907)
    probes = [torch.randint(vocab, (344,), generator=generator),
              torch.tensor([vocab-1]), torch.tensor([0, vocab//2, vocab-1, 0])]
    expected = [run(ids) for ids in probes]
    receipt = offload_embedding.offload(embed, "/scratch/pireus/cache/embedding-offload")
    for block, start in enumerate(range(0, vocab, 512)):
        if rawhash(run(torch.arange(start, min(vocab, start+512)))) != references[block]:
            raise ValueError("full vocabulary GPU/file byte disagreement")
    for ids, ref in zip(probes, expected):
        if not torch.equal(run(ids).view(torch.uint8), ref.view(torch.uint8)):
            raise ValueError("344/1/boundary input byte disagreement")
    # Private COW corruption must be detected; gather drops it after copying.
    embed.weight.view(torch.uint8)[0,0] ^= 1
    corrupted = run(probes[-1])
    if torch.equal(corrupted.view(torch.uint8), expected[-1].view(torch.uint8)):
        raise ValueError("corruption control was insensitive")
    if not torch.equal(run(probes[-1]).view(torch.uint8), expected[-1].view(torch.uint8)):
        raise ValueError("private COW corruption persisted")
    receipt.update(stage="EMBEDDING_OFFLOAD_CONTROL_PASS", checkpoint_tensor=key,
        vocabulary_rows=vocab, output_components=vocab*embed.embedding_dim,
        full_vocabulary_byte_exact=True, token_shapes=[344,1,4],
        corruption_detected=True, private_corruption_reverted=True,
        transformer_layers_executed=False, inference_accepted=False, inductor_compile_threads=inductor_config.compile_threads,
        helper_sha256=offload_embedding.file_digest(Path(offload_embedding.__file__)))
    offline.write(Path("/scratch/pireus/receipts")/("embedding-offload-control-"+receipt["job"]+"-"+str(rank)+".json"), receipt)
    print(json.dumps(receipt), flush=True)
    dist.barrier(group=model.tp_group.cpu_group)

if __name__ == "__main__":
    main()
