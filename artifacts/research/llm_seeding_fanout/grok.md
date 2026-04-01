### Highest ROI Strategy: **Targeted LoRA Fine-Tuning on Code-Specific Open Models + Synthetic Data Amplification**

Fine-tuning is the **clear highest ROI** because:
- Your ~1000 verified examples + docs are already a goldmine (most new langs start with <100).
- LoRA is cheap (~$10-50 on Colab/A100 rental, 1-2 hours), preserves base model, and creates "Sounio-aware" adapters usable by anyone.
- It embeds syntax/effects/types into weights, fixing hallucinations permanently (no prompt fragility).
- Distributable on HF → viral seeding (users fine-tune further, merge into custom models).
- In-context/RAG/docs fail on zero-shot (LLMs default to Rust); ecosystem takes months; structural specs help but don't override training priors.

**Traps to avoid:**
- Full SFT/RLHF: Overkill, data too small (stick to LoRA).
- Synthetic data without verification: Hallucinations poison dataset.
- GitHub-only: Low visibility (needs active shilling).
- Long prompts: >10k tokens → context collapse.

**What I'd do first (Day 1 actionable plan): Build & train a LoRA dataset/adapter in 1 week.**

#### Step 1: Curate Dataset (2-3 days, ~5k high-quality pairs)
Use existing assets → **10k-20k instruction-code pairs** (input: task/prompt/errors; output: verified Sounio code).
- **Format:** Alpaca-style JSONL: `{"instruction": "...", "input": "...", "output": "full .sio code"}`.
- **Sources & Specific Augmentations (beyond what's there):**
  | Source | Pairs Generated | How |
  |--------|-----------------|-----|
  | `examples/*.sio` (~250) | 750 | For each: (1) "Write [file docstring/description]" → code. (2) "Fix this broken Rust-like code: [inject top-5 errors: semicolons, `let mut x`, `&mut`, `assert!`, unary `-`]" → corrected. |
  | `tests/run-pass/*.sio` (~200) | 800 | Same + "Generate unit test for [func]: expect [assert/print_i64 output]" → code. Verify compiles/runs. |
  | `docs/llm-guide/cookbook.md` (12 patterns) | 120 | "Implement pattern #X ([desc]) for [variation: e.g., IO+Mut for vec push]" → idiomatic code. 10 vars/pattern. |
  | `docs/llm-guide/error-catalog.md` (15 errors) | 300 | "Diagnose/fix: [error msg + broken code with injected hallucination]" → fix + explanation. |
  | `docs/guide/LLM_PROGRAMMING_GUIDE.md` + `llms.txt` | 500 | Extract rules → pairs like "Translate Rust `fn foo() -> Result<T,E> { ... }` to Sounio (use IO+Panic)" → code. Prefix all outputs with `//@ run-pass`. |
  | `stdlib-index.md` | 400 | "Use [stdlib fn] with effects: [task, e.g., `vec::push` needs Mut]" → snippet. |
  | **New: Synthetic bootstrapping** | 5k+ | Use base CodeLlama-7B-Instruct: Prompt with your `llms.txt` + 50 examples → generate 100 tasks/file. **Verify all:** Pipe to compiler (`sio check --effects`), filter passes (expect 60-80% yield). Repeat 5x. |

- **Tools:** Script in Python:
  ```python
  import glob, json, subprocess
  dataset = []
  for f in glob.glob("examples/*.sio"):
      code = open(f).read()
      task = f"Write a Sounio program to {extract_docstring(code)}."  # Parse //@ comments
      dataset.append({"instruction": task, "output": code})
      broken = inject_errors(code)  # e.g., add ";" randomly
      dataset.append({"instruction": f"Fix: {broken}", "output": code})
      # Verify: subprocess.run(["sio", "check", f]) == 0
  ```
- **Size goal:** 15k pairs (80/20 train/val). Balance: 40% full progs, 30% fixes, 20% snippets, 10% effects/types.
- **System prompt injection:** Every pair prepends `llms.txt` content to "instruction".

#### Step 2: Train & Host LoRA (1-2 days)
- **Model:** CodeLlama-7B-Instruct or DeepSeek-Coder-6.7B (best code priors, cheap).
- **LoRA config:** `peft` + `trl` (SFTTrainer). r=16, alpha=32, target `q_proj,v_proj` (syntax-heavy).
  ```bash
  pip install peft trl transformers datasets accelerate
  # train.py: load dataset, trainer.train(epochs=3, batch=4, lr=1e-4, max_seq=2048)
  ```
- **Cost:** Free on Colab Pro+ (T4 x2), or $20 RunPod A100.
- **Eval:** Post-train: 100 held-out tasks → compile rate >95%, no top-5 errors (manual spot-check + `sio check`).
- **Host:** Push to HF: `yourorg/sounio-lora-codellama7b`. Include Space demo: "Chat → Generate Sounio code".

#### Step 3: Immediate Seeding (Parallel, Day 4+)
- **HF + Viral:** Upload dataset too (`sounio/sounio-finetune-dataset`). Tag "code-generation sounio". Announce on r/MachineLearning, r/LocalLLaMA ("Sounio LoRA: Fix LLMs writing fake Rust").
- **In-context booster:** Distill LoRA → ultra-concise **"Sounio Anti-Hallucination Primer"** (1k tokens, 50 few-shots): Top-5 errors + fixes + 10 core snippets (effects, `&!`, `Knowledge<T>`, etc.). Host as `docs/llm-guide/primer.md` for copy-paste into any LLM.
- **RAG integration:** JSON spec file: `sounio-syntax.json` (BNF grammar + effects table + stdlib sigs). E.g.,
  ```json
  {"effects": {"IO": "require `use effect IO;`, no inference"}, "refs": {"exclusive": "&!T", "not": "&mut"}}
  ```
  Use in LangChain/Vectara for custom Sounio assistants.

#### Step 4: Amplify (Week 2+, compounding)
- **Self-play synthetics:** Use LoRA to generate 50k+ pairs → re-fine-tune → repeat (yield doubles knowledge).
- **Ecosystem push:** 
  - Blog: "Teaching LLMs Sounio in 1 Hour" (dev.to/HN) with before/after LLM outputs.
  - GH: Add `llms/` dir with LoRA usage, primer. Star-hunt: Cross-post to Rust/Systems langs Discords.
  - Colab notebook: "Try Sounio LoRA now" (load adapter, chat=code).
- **Advanced:** Submit LoRA merge PRs to CodeLlama/Dolphin forks on HF.

**Timeline/ROI Projection:**
| Week | Action | Impact |
|------|--------|--------|
| 1 | Dataset + LoRA | You + early users generate perfect Sounio (95% compile rate). |
| 2-4 | Synthetics + host | 10k downloads → forks → base model trainers notice. |
| 3m+ | Ecosystem | Usage → GitHub corpus → next LLM trainings (2025+). |

This beats others: In-context brittle; ecosystem slow; structural auxiliary. **Start with dataset script today**—it's 80% of value. If GitHub real, I'd fork & PR this exact plan.
