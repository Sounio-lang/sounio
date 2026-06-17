#!/usr/bin/env python3
"""Aggregate Sounio novelty datasets into a single Alpaca-format JSONL for Axolotl."""
import json
from pathlib import Path

DATASETS = [
    ('datasets/sounio-ai-epistemic-gum/epistemic_gum_pairs.jsonl', 'epistemic'),
    ('datasets/sounio-ai-lean-pairs/lean_pairs.jsonl', 'formal'),
    ('datasets/sounio-ai-compiler-patches/compiler_patches.jsonl', 'compiler'),
    ('datasets/sounio-ai-refinements/refinement_types.jsonl', 'refinement'),
]

OUT_PATH = 'datasets/sounio-novelty-alpaca.jsonl'

def main():
    repo_root = Path(__file__).parent.parent.parent
    out_file = repo_root / OUT_PATH
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    
    for rel_path, domain in DATASETS:
        p = repo_root / rel_path
        if not p.exists():
            print(f"Skipping {rel_path} (not found)")
            continue
            
        count = 0
        with p.open('r') as f:
            for line in f:
                data = json.loads(line)
                
                # Convert to Alpaca format
                # Most of my datasets already have instruction/input/output
                # Refinement dataset has a different structure, need to adapt it
                
                if domain == 'refinement':
                    # name, base_type, var, predicate_str, usages
                    name = data.get('name')
                    pred = data.get('predicate_str')
                    usages = data.get('usages', [])
                    
                    # Record 1: Definition
                    records.append({
                        "instruction": f"Explain the Sounio refinement type `{name}`.",
                        "input": "",
                        "output": f"The refinement type `{name}` is defined on base type `{data.get('base_type')}` with the predicate: `{pred}`."
                    })
                    
                    # Record 2: Usage example
                    if usages:
                        records.append({
                            "instruction": f"Show an example usage of the Sounio refinement type `{name}`.",
                            "input": "",
                            "output": f"Here is an example of `{name}` being used in a function signature:\n\n{usages[0]}"
                        })
                else:
                    # epistemic, formal, compiler
                    instr = data.get('instruction', f"Perform the {data.get('task', domain)} task for Sounio.")
                    inp = data.get('input', '')
                    out = data.get('output', '')
                    
                    records.append({
                        "instruction": instr,
                        "input": inp,
                        "output": out
                    })
                count += 1
        print(f"Loaded {count} records from {rel_path}")

    with out_file.open('w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
            
    print(f"\nWrote {len(records)} Alpaca-style records to {OUT_PATH}")

if __name__ == "__main__":
    main()
