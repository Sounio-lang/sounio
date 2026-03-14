#!/usr/bin/env python3
"""
Analyze syntactic constructions in self-hosted Sounio source files.
"""
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

def count_constructs(root_dir: str):
    root = Path(root_dir)
    sio_files = list(root.rglob("*.sio"))
    print(f"Found {len(sio_files)} .sio files")
    
    # Keywords to search for (whole word, case-sensitive)
    keywords = [
        "match", "use", "enum", "struct", "fn", "let", "var",
        "if", "while", "for", "return", "break", "continue",
        "type", "impl", "trait", "mod", "import", "extern",
        "knowledge", "units", "effect", "handle", "spawn",
        "async", "await", "try", "catch", "throw",
    ]
    
    # Regex pattern for each keyword (word boundary)
    patterns = {k: re.compile(rf'\b{k}\b') for k in keywords}
    counts = {k: 0 for k in keywords}
    per_module = defaultdict(Counter)
    
    for fpath in sio_files:
        try:
            content = fpath.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            print(f"Warning: cannot read {fpath}: {e}", file=sys.stderr)
            continue
        
        # Determine module (relative path from root)
        rel = fpath.relative_to(root)
        module = str(rel.parent) if rel.parent != Path('.') else '.'
        
        for kw, pat in patterns.items():
            matches = len(pat.findall(content))
            if matches:
                counts[kw] += matches
                per_module[module][kw] += matches
    
    # Sort modules by total occurrences
    sorted_modules = sorted(
        per_module.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )
    
    return {
        'total_files': len(sio_files),
        'counts': counts,
        'per_module': dict(sorted_modules[:20]),  # top 20
    }

def main():
    root = "./"
    if len(sys.argv) > 1:
        root = sys.argv[1]
    data = count_constructs(root)
    
    print("\n=== OCCURRENCES BY CONSTRUCTION ===")
    for kw, cnt in sorted(data['counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"{kw:12} {cnt:6}")
    
    print("\n=== TOP MODULES BY TOTAL OCCURRENCES ===")
    for mod, ctr in list(data['per_module'].items())[:10]:
        total = sum(ctr.values())
        print(f"{mod:40} {total:6}")
        # optionally show top keywords in this module
        top = ctr.most_common(3)
        for kw, cnt in top:
            print(f"    {kw:10} {cnt:6}")
    
    # Write machine-readable summary for inventory.md
    with open("inventory_summary.txt", "w") as f:
        f.write("TOTAL_FILES {}\n".format(data['total_files']))
        for kw, cnt in data['counts'].items():
            f.write("KEYWORD {} {}\n".format(kw, cnt))

if __name__ == "__main__":
    main()