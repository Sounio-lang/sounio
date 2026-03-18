# MultiPL-E Sounio Translator

[MultiPL-E](https://github.com/nuprl/MultiPL-E) translator and evaluator for
the Sounio programming language. Converts OpenAI HumanEval Python problems to
idiomatic `.sio` files and evaluates pass@k on generated completions.

## Overview

MultiPL-E is the standard benchmark for evaluating code-generation LLMs across
programming languages. This directory provides:

- **`humaneval_to_sounio.py`** -- Translator from Python HumanEval prompts to
  Sounio function stubs with test harnesses.
- **`eval_sounio.py`** -- Evaluation script that runs `.sio` files through
  `souc` and computes pass@k metrics.

## Translation Rules

| Python | Sounio |
|--------|--------|
| `int` | `i64` |
| `float` | `f64` |
| `bool` | `bool` |
| `str` | `[i8; 256]` |
| `List[int]` | `[i64; 256]` (fixed-size, pass by `&`) |
| `Optional[int]` | `(i64, bool)` -- value + is_some flag |
| `assert x == y` | `assert(x == y)` |
| `-42` | `(0 - 42)` |
| `True` / `False` | `true` / `false` |
| `None` | `0` |
| semicolons | removed |
| `let mut` | `var` |
| `&mut` | `&!` |

All generated functions include `with Mut, Panic, Div` effects by default.
The Sounio type-checker will report if effects are unnecessary.

## Quick Start

```bash
# 1. Download HumanEval
wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
gunzip HumanEval.jsonl.gz

# 2. Translate all problems to .sio stubs
python humaneval_to_sounio.py --input HumanEval.jsonl --outdir generated/

# 3. (Have an LLM fill in the function bodies)

# 4. Evaluate
python eval_sounio.py --dir generated/ --k 1,10,100 --verbose
```

## Translator Usage

```bash
# Translate a single problem
python humaneval_to_sounio.py --input HumanEval.jsonl --problem HumanEval/0

# Read one problem from stdin
echo '{"task_id":"test/0","prompt":"def add(a:int,b:int)->int:\n    ...","test":"assert add(1,2)==3","entry_point":"add"}' \
  | python humaneval_to_sounio.py --stdin

# Show type mapping
python humaneval_to_sounio.py --show-types

# Use custom effects
python humaneval_to_sounio.py --input HumanEval.jsonl --effects "IO, Mut, Panic"
```

## Evaluator Usage

```bash
# Evaluate a single file
python eval_sounio.py --file generated/humaneval_0.sio

# Evaluate a directory with JSONL output
python eval_sounio.py --dir generated/ --output results.jsonl --verbose

# Type-check only (no execution)
python eval_sounio.py --dir generated/ --check-only

# Custom souc path and timeout
python eval_sounio.py --dir generated/ --souc /path/to/souc --timeout 60
```

## pass@k Computation

Uses the unbiased estimator from Chen et al. (2021) "Evaluating Large Language
Models Trained on Code":

    pass@k = 1 - C(n-c, k) / C(n, k)

where n = total samples per problem, c = correct samples.

## Integration with MultiPL-E

To add Sounio to a MultiPL-E evaluation run:

1. Place `humaneval_to_sounio.py` in the MultiPL-E `translators/` directory.
2. Register Sounio in the MultiPL-E configuration with extension `.sio`.
3. Use `eval_sounio.py` as the execution backend.

The translator follows the MultiPL-E convention of a class with a
`translate_problem()` method that takes a HumanEval problem dict and returns
target-language source code.

## Sounio-Specific Notes

- **No semicolons** -- all Sounio expressions are newline-terminated.
- **Effects** -- functions declare side effects: `with IO, Mut, Panic, Div`.
- **Fixed-size arrays** -- Python lists become `[T; 256]` with explicit length
  tracking.
- **No unary minus** -- `-x` becomes `0 - x`.
- **No closure literals** -- use named function references instead.
- **Struct wrappers** -- mutable array parameters often need a struct wrapper
  due to the `&![T; N]` JIT propagation bug.

See `docs/LLM_PROGRAMMING_GUIDE.md` for the full Sounio syntax reference.
