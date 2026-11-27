<!-- docs:meta
topic_id: repo.docs.llm.granite-step0
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm.granite-step0
-->

# Granite-3.1-8B vs Qwen3-8B: forward-pass differences (Step 0, 2026-11-26)
Source: config.json of ibm-granite/granite-3.1-8b-instruct and Qwen/Qwen3-8B (Hub, 2026-11-26).
Modeling code verified against [Transformers v4.48.3 Granite](https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/granite/modeling_granite.py).
f32 bits computed in C++23 (strtof and strtod->float agree).

| Item | Qwen3-8B | Granite-3.1-8B | Primitive / note |
|---|---|---|---|
| layers | 36 | 40 | |
| hidden/heads/kv/head_dim | 4096/32/8/128 | 4096/32/8/128 (implied) | same GQA |
| intermediate | 12288 | 12800 (=100*128) | multiple of 128 and 256 |
| QK-norm | yes | no | |
| attention scale | 1/sqrt(128) | attention_multiplier=0.0078125=2^-7 (0x3C000000) | one mul after QK dot, before masking and softmax |
| embedding | plain | x embedding_multiplier=12.0 (0x41400000) | one mul on input embedding activation only |
| residual | x+f(x) | x+0.22*f(x) at both residual merges | 0x3E6147AE (=0.219999998808), NOT exact: load bits |
| logits | plain | / logits_scaling=16.0 | exact (power of 2): token order unchanged |
| tied embeddings | no | yes | lm_head = embedding^T, no embedding multiplier on output |
| vocab | 151936 (mod 128=0) | 49155 (mod 32=3) | NEW 3-element tail in argmax/lm_head |
| rope theta | 1e6 | 1e7, no scaling | table must be host-independent |
| rms_norm_eps | 1e-6 | 1e-5 | |
| bos/eos/pad | 151643/151645/- | 0/0/0 | |

Named risks: (a) residual multiplier must be the exact bits; (b) argmax tie-break lowest index on
equal bits everywhere; (c) RoPE table generator host libm vs contract ops; (d) vocab tail of 3.

Source trace: `GraniteAttention` supplies `config.attention_multiplier` to
attention; `eager_attention_forward` scales the dot product before masking
and softmax. `GraniteDecoderLayer.forward` multiplies both the attention
output (after output projection) and the MLP output (after down projection)
separately before their residual additions. `GraniteModel.forward` scales
input embeddings, not the weights. `GraniteForCausalLM.forward` applies the
unscaled tied linear weight then divides the resulting logits by 16.
These are placements for the Option A f32 engine, not a claim of bit identity
with Transformers' native BF16 activation arithmetic.

RoPE source inspection: `qwen3_fill_rope_table` imports `pow`, `sin`, and `cos`
from `stdlib/math/pure.sio`, not host libm. `pow` evaluates `exp(n*ln(x))`
using the in-tree arithmetic implementation; sine and cosine are in-tree
range reduction and Taylor series. No host libm substitution is required
at the source level. Cross-host execution evidence is still required to
establish identical table bits for the existing binary64 generator.
