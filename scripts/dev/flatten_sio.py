#!/usr/bin/env python3
"""flatten_sio.py -- recursively resolve `use` imports and produce a flat .sio file.

Usage: python3 scripts/flatten_sio.py <main.sio> [repo_root]

The repo_root defaults to the directory of main.sio, and imports are resolved
relative to <repo_root>/self-hosted/.
"""
import sys, os, re

def path_from_module(mod_path, repo_root):
    """Convert 'lexer::mod' -> repo_root/self-hosted/lexer/mod.sio"""
    parts = mod_path.replace('::', '/')
    return os.path.join(repo_root, 'self-hosted', parts + '.sio')

def flatten(main_file, repo_root=None, included=None, out_lines=None):
    if repo_root is None:
        repo_root = os.path.dirname(os.path.abspath(main_file))
        # Walk up to find the repo root (contains 'self-hosted/')
        d = repo_root
        while d != '/':
            if os.path.isdir(os.path.join(d, 'self-hosted')):
                repo_root = d
                break
            d = os.path.dirname(d)
    if included is None:
        included = set()
    if out_lines is None:
        out_lines = []

    abs_path = os.path.abspath(main_file)
    if abs_path in included:
        return out_lines
    included.add(abs_path)

    try:
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except FileNotFoundError:
        out_lines.append(f'// [flatten: file not found: {abs_path}]\n')
        return out_lines

    for line in lines:
        # Match: use module::path::maybe::star or use module::path::{...}
        m = re.match(r'^use\s+([\w:]+?)(?:::\*|::\{[^}]*\}|;|\s*$)', line.rstrip())
        if m:
            mod_path = m.group(1)
            imp_file = path_from_module(mod_path, repo_root)
            out_lines.append(f'// [import: {mod_path} from {imp_file}]\n')
            flatten(imp_file, repo_root, included, out_lines)
        else:
            out_lines.append(line)

    return out_lines

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: flatten_sio.py <main.sio> [repo_root]', file=sys.stderr)
        sys.exit(1)
    main_file = sys.argv[1]
    repo_root = sys.argv[2] if len(sys.argv) > 2 else None
    lines = flatten(main_file, repo_root)
    sys.stdout.write(''.join(lines))
