# Sounio AI Structure Hybrid Dataset

`structure_hybrid.v1.jsonl` is the v13 compiler-in-the-loop training rung for
the small Sounio repair model. It combines executable Sounio programs with
compact octonion/O-SSM descriptors.

Scope:

- rung: `repair_v13_octo_ossm_hybrid`
- model size: continue the small adapter path, do not switch to a larger model
- excluded causal factor: Mandelbrot second-derivative prompts are reserved for
  v14
- output contract: every `output` is a complete Sounio source file that passed
  `souc check` when generated

Record schema:

- `instruction`, `input`, `output`: compatible with `scripts/dev/lora_finetune.py`
- `category`: `ossm_control`, `octonion_invariant`, `knowledge_octonion`,
  `science_spine`, or `compiler_repair`
- `rule`: prompt transformation family
- `source_path`: repo source used as compiler-backed oracle output
- `oracle_kind`: `souc_check` or `expect_stdout`
- `oct_descriptor`: eight-channel/Fano/invariant descriptor metadata

Regenerate:

```bash
python3 scripts/dev/build_sounio_ai_structure_hybrid_dataset.py
```
