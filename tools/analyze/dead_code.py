#!/usr/bin/env python3
"""
Dead Code Analyzer for Sounio
Outputs detailed dead code findings with line/column information
Integrates with LSP for inline diagnostics
"""

import sys
import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class DeadCodeKind(Enum):
    UNUSED_FUNCTION = "unused_function"
    UNUSED_VARIABLE = "unused_variable"
    UNUSED_IMPORT = "unused_import"
    UNUSED_TYPE = "unused_type"
    UNREACHABLE_CODE = "unreachable_code"
    UNUSED_PARAMETER = "unused_parameter"

@dataclass
class DeadCodeItem:
    kind: DeadCodeKind
    name: str
    file: str
    line: int
    column: int
    end_line: int
    end_column: int
    message: str
    suggested_fix: Optional[str] = None
    
    def to_lsp_diagnostic(self) -> Dict:
        """Convert to LSP Diagnostic format"""
        severity_map = {
            DeadCodeKind.UNUSED_FUNCTION: 4,  # Hint
            DeadCodeKind.UNUSED_VARIABLE: 4,  # Hint
            DeadCodeKind.UNUSED_IMPORT: 4,    # Hint
            DeadCodeKind.UNUSED_TYPE: 4,      # Hint
            DeadCodeKind.UNREACHABLE_CODE: 2, # Warning
            DeadCodeKind.UNUSED_PARAMETER: 4, # Hint
        }
        
        return {
            "range": {
                "start": {"line": self.line - 1, "character": self.column - 1},
                "end": {"line": self.end_line - 1, "character": self.end_column - 1}
            },
            "severity": severity_map.get(self.kind, 4),
            "code": self.kind.value,
            "source": "sounio-dead-code",
            "message": self.message,
            "relatedInformation": [],
            "tags": [1]  # 1 = Unnecessary, 2 = Deprecated
        }
    
    def to_dict(self) -> Dict:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "location": {
                "file": self.file,
                "line": self.line,
                "column": self.column,
                "endLine": self.end_line,
                "endColumn": self.end_column
            },
            "message": self.message,
            "suggestedFix": self.suggested_fix
        }

class DeadCodeAnalyzer:
    def __init__(self, file_path: str, content: Optional[str] = None):
        self.file_path = file_path
        self.content = content or Path(file_path).read_text()
        self.lines = self.content.split('\n')
        self.items: List[DeadCodeItem] = []
        
        # Track definitions and uses
        self.definitions: Dict[str, Tuple[int, int, int, int]] = {}  # name -> (line, col, end_line, end_col)
        self.uses: set = set()
        
    def analyze(self) -> List[DeadCodeItem]:
        """Run all dead code analysis passes"""
        self._collect_definitions()
        self._collect_uses()
        self._find_unused()
        self._find_unreachable_code()
        return self.items
    
    def _collect_definitions(self):
        """Find all function, variable, type definitions"""
        
        # Functions: fn name(...) { ... } with full body
        func_pattern = re.compile(r'^(\s*)(?:pub\s+)?fn\s+(\w+)(?:<[^>]+>)?\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*\{', re.MULTILINE)
        for match in func_pattern.finditer(self.content):
            indent = len(match.group(1))
            name = match.group(2)
            start_line = self.content[:match.start()].count('\n') + 1
            line_start = self.content.rfind('\n', 0, match.start()) + 1
            start_col = match.start() - line_start + 1
            
            # Find the end of the function (matching closing brace)
            brace_start = self.content.find('{', match.end() - 1)
            brace_count = 1
            pos = brace_start + 1
            while pos < len(self.content) and brace_count > 0:
                if self.content[pos] == '{':
                    brace_count += 1
                elif self.content[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            end_line = self.content[:pos].count('\n') + 1
            end_line_content = self.lines[end_line - 1] if end_line <= len(self.lines) else ""
            end_col = len(end_line_content)
            
            self.definitions[name] = (start_line, start_col, end_line, end_col)
        
        # Let bindings: let name = ...
        for match in re.finditer(r'\blet\s+(?:mut\s+)?(\w+)', self.content):
            name = match.group(1)
            # Skip if inside a comment
            line = self.content[:match.start()].count('\n') + 1
            line_start = self.content.rfind('\n', 0, match.start()) + 1
            col = match.start() - line_start + 1
            end_col = col + len(name)
            
            # Only track local variables (not top-level constants)
            line_content = self.lines[line - 1]
            if not line_content.strip().startswith('const'):
                var_name = f"var:{name}:{line}"  # Make unique per line
                self.definitions[var_name] = (line, col, line, end_col)
        
        # Imports: use module::name or import name
        for match in re.finditer(r'\b(?:use|import)\s+([\w:]+)', self.content):
            name = match.group(1).split('::')[-1]  # Get last segment
            line = self.content[:match.start()].count('\n') + 1
            line_start = self.content.rfind('\n', 0, match.start()) + 1
            col = match.start() - line_start + 1
            end_col = col + len(match.group(1))
            self.definitions[f"import:{name}"] = (line, col, line, end_col)
        
        # Types: struct Name, enum Name, type Name
        for match in re.finditer(r'\b(?:struct|enum|type|trait)\s+(\w+)', self.content):
            name = match.group(1)
            line = self.content[:match.start()].count('\n') + 1
            line_start = self.content.rfind('\n', 0, match.start()) + 1
            col = match.start() - line_start + 1
            end_col = col + len(name)
            self.definitions[f"type:{name}"] = (line, col, line, end_col)
    
    def _collect_uses(self):
        """Find all identifier uses (not definitions)"""
        
        # Find all identifier references
        for match in re.finditer(r'\b([a-zA-Z_]\w*)\b', self.content):
            name = match.group(1)
            line = self.content[:match.start()].count('\n') + 1
            line_content = self.lines[line - 1]
            
            # Skip keywords
            keywords = {'fn', 'let', 'const', 'mut', 'if', 'else', 'match', 
                       'for', 'while', 'return', 'struct', 'enum', 'type',
                       'trait', 'impl', 'pub', 'with', 'as', 'in', 'use',
                       'import', 'true', 'false', 'self', 'Self'}
            if name in keywords:
                continue
            
            # Skip definitions (simple heuristic: check if preceded by certain patterns)
            before = self.content[max(0, match.start()-20):match.start()]
            if re.search(r'\b(?:fn|let\s+(?:mut\s+)?|const|use|import|struct|enum|type|trait)\s+$', before):
                continue
            
            # Skip type annotations (after :)
            if re.search(r':\s*$', before):
                continue
            
            self.uses.add(name)
    
    def _find_unused(self):
        """Identify unused definitions"""
        
        for full_name, (line, col, end_line, end_col) in self.definitions.items():
            kind_prefix, name = full_name.split(':', 1) if ':' in full_name else ('', full_name)
            
            # For variables with line numbers in key, extract just the name
            if kind_prefix == 'var':
                base_name = name.rsplit(':', 1)[0]  # Remove line number suffix
            else:
                base_name = name
            
            # Check if used
            if base_name not in self.uses:
                if kind_prefix == 'var':
                    # Check if variable is actually used in scope
                    if not self._is_variable_used(base_name, line):
                        self.items.append(DeadCodeItem(
                            kind=DeadCodeKind.UNUSED_VARIABLE,
                            name=name,
                            file=self.file_path,
                            line=line,
                            column=col,
                            end_line=end_line,
                            end_column=end_col,
                            message=f"Variable '{name}' is never used",
                            suggested_fix=f"Remove: let {name} = ..."
                        ))
                elif kind_prefix == 'import':
                    self.items.append(DeadCodeItem(
                        kind=DeadCodeKind.UNUSED_IMPORT,
                        name=name,
                        file=self.file_path,
                        line=line,
                        column=col,
                        end_line=end_line,
                        end_column=end_col,
                        message=f"Import '{name}' is never used",
                        suggested_fix=f"Remove: use ...::{name}"
                    ))
                elif kind_prefix == 'type':
                    self.items.append(DeadCodeItem(
                        kind=DeadCodeKind.UNUSED_TYPE,
                        name=name,
                        file=self.file_path,
                        line=line,
                        column=col,
                        end_line=end_line,
                        end_column=end_col,
                        message=f"Type '{name}' is never used"
                    ))
                else:
                    # Function (default)
                    # Skip main functions
                    if name not in ['main', 'main2'] and not name.startswith('_'):
                        self.items.append(DeadCodeItem(
                            kind=DeadCodeKind.UNUSED_FUNCTION,
                            name=name,
                            file=self.file_path,
                            line=line,
                            column=col,
                            end_line=end_line,
                            end_column=end_col,
                            message=f"Function '{name}' is never called",
                            suggested_fix=f"Remove function '{name}' or add '_' prefix"
                        ))
    
    def _is_variable_used(self, var_name: str, defined_line: int) -> bool:
        """Check if a variable is used in its scope"""
        defined_line_idx = defined_line - 1
        
        # Get the definition line to understand scope
        def_line_content = self.lines[defined_line_idx]
        defined_indent = len(def_line_content) - len(def_line_content.lstrip())
        
        # Find scope end by looking for lines with same or less indentation
        scope_end = len(self.lines)
        brace_count = 0
        
        for i in range(defined_line_idx + 1, len(self.lines)):
            line = self.lines[i]
            stripped = line.strip()
            
            # Track braces for scope
            brace_count += stripped.count('{') - stripped.count('}')
            
            # If we're back at same or lower indentation and brace_count is 0 or less
            if stripped:
                line_indent = len(line) - len(line.lstrip())
                if brace_count <= 0 and line_indent <= defined_indent:
                    # Check if it's the end of a block
                    if stripped.startswith('}') or stripped.startswith('else'):
                        scope_end = i
                        break
        
        # Search for variable use in scope (after definition)
        for i in range(defined_line_idx + 1, scope_end):
            line = self.lines[i]
            
            # Skip comment lines
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('#'):
                continue
            
            # Check for the variable name as a complete word
            # Make sure it's not inside a string literal
            if re.search(rf'\b{re.escape(var_name)}\b', line):
                # Additional check: ensure it's not another definition with same name
                before_match = line[:line.find(var_name)]
                if not re.search(r'\b(let|const|var|fn)\s+', before_match):
                    return True
        
        return False
    
    def _find_unreachable_code(self):
        """Find unreachable code (after return, break, etc.)"""
        
        in_function = False
        function_start = 0
        
        for i, line in enumerate(self.lines):
            stripped = line.strip()
            
            # Track function boundaries
            if re.match(r'^\s*(?:pub\s+)?fn\s+', line):
                in_function = True
                function_start = i
            
            if in_function:
                # Check for return/break/continue
                if re.match(r'^\s*(return|break|continue)\b', line):
                    # Check if there's code after on the same indentation level
                    current_indent = len(line) - len(line.lstrip())
                    
                    for j in range(i + 1, len(self.lines)):
                        next_line = self.lines[j]
                        if not next_line.strip():
                            continue
                        
                        next_indent = len(next_line) - len(next_line.lstrip())
                        
                        # If same indentation and not a closing brace, it's unreachable
                        if next_indent == current_indent and not next_line.strip().startswith('}'):
                            self.items.append(DeadCodeItem(
                                kind=DeadCodeKind.UNREACHABLE_CODE,
                                name="",
                                file=self.file_path,
                                line=j + 1,
                                column=next_indent + 1,
                                end_line=j + 1,
                                end_column=len(next_line) + 1,
                                message="Unreachable code after return/break/continue"
                            ))
                            break
                        elif next_indent < current_indent:
                            break
                
                # Check for infinite loops
                if 'loop {' in line:
                    # Code after infinite loop is unreachable
                    current_indent = len(line) - len(line.lstrip())
                    
                    for j in range(i + 1, len(self.lines)):
                        next_line = self.lines[j]
                        if not next_line.strip():
                            continue
                        
                        next_indent = len(next_line) - len(next_line.lstrip())
                        
                        if next_indent == current_indent and not next_line.strip().startswith('}'):
                            # Check if loop has break
                            loop_content = '\n'.join(self.lines[i:j])
                            if 'break' not in loop_content:
                                self.items.append(DeadCodeItem(
                                    kind=DeadCodeKind.UNREACHABLE_CODE,
                                    name="",
                                    file=self.file_path,
                                    line=j + 1,
                                    column=next_indent + 1,
                                    end_line=j + 1,
                                    end_column=len(next_line) + 1,
                                    message="Unreachable code after infinite loop"
                                ))
                            break
                        elif next_indent < current_indent:
                            break


def analyze_file(file_path: str, output_format: str = "text") -> str:
    """Analyze a single file"""
    analyzer = DeadCodeAnalyzer(file_path)
    items = analyzer.analyze()
    
    if output_format == "json":
        return json.dumps({
            "file": file_path,
            "dead_code": [item.to_dict() for item in items],
            "summary": {
                "total": len(items),
                "unused_functions": len([i for i in items if i.kind == DeadCodeKind.UNUSED_FUNCTION]),
                "unused_variables": len([i for i in items if i.kind == DeadCodeKind.UNUSED_VARIABLE]),
                "unused_imports": len([i for i in items if i.kind == DeadCodeKind.UNUSED_IMPORT]),
                "unreachable_code": len([i for i in items if i.kind == DeadCodeKind.UNREACHABLE_CODE])
            }
        }, indent=2)
    
    elif output_format == "lsp":
        return json.dumps({
            "jsonrpc": "2.0",
            "result": {
                "kind": "full",
                "items": [item.to_lsp_diagnostic() for item in items]
            }
        })
    
    else:  # text format
        if not items:
            return f"✅ No dead code found in {file_path}"
        
        lines = [f"🔍 Dead Code Analysis: {file_path}\n"]
        
        for item in items:
            icon = {
                DeadCodeKind.UNUSED_FUNCTION: "📦",
                DeadCodeKind.UNUSED_VARIABLE: "🔧",
                DeadCodeKind.UNUSED_IMPORT: "📥",
                DeadCodeKind.UNUSED_TYPE: "📋",
                DeadCodeKind.UNREACHABLE_CODE: "⚠️",
                DeadCodeKind.UNUSED_PARAMETER: "📎"
            }.get(item.kind, "•")
            
            lines.append(f"{icon} {item.message}")
            lines.append(f"   at {item.file}:{item.line}:{item.column}")
            if item.suggested_fix:
                lines.append(f"   💡 {item.suggested_fix}")
            lines.append("")
        
        return '\n'.join(lines)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Dead Code Analyzer for Sounio")
    parser.add_argument("path", help="File or directory to analyze")
    parser.add_argument("--format", choices=["text", "json", "lsp"], default="text",
                       help="Output format")
    parser.add_argument("--fix", action="store_true", help="Fix issues automatically")
    parser.add_argument("--exclude", nargs="+", default=[], help="Patterns to exclude")
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        result = analyze_file(str(path), args.format)
        print(result)
    
    elif path.is_dir():
        files = list(path.rglob("*.sio"))
        all_items = []
        
        for file in files:
            # Check exclusions
            if any(ex in str(file) for ex in args.exclude):
                continue
            
            analyzer = DeadCodeAnalyzer(str(file))
            items = analyzer.analyze()
            all_items.extend(items)
        
        if args.format == "json":
            print(json.dumps({
                "total_files": len(files),
                "total_issues": len(all_items),
                "items": [item.to_dict() for item in all_items]
            }, indent=2))
        else:
            for item in all_items:
                print(f"{item.file}:{item.line}:{item.column}: {item.message}")
            print(f"\nTotal: {len(all_items)} dead code items in {len(files)} files")
    
    else:
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
