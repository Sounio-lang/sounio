#!/usr/bin/env python3
"""
Workspace Cross-File Analyzer for Sounio
Detects unused exports, orphaned modules, circular dependencies
Uses swarm parallel processing for performance
"""

import sys
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from enum import Enum
import time


class DependencyKind(Enum):
    IMPORT = "import"
    EXPORT = "export"
    CALL = "call"
    TYPE_USE = "type_use"


@dataclass
class Symbol:
    """Represents a symbol (function, type, etc.) in a module"""
    name: str
    kind: str  # "function", "type", "variable", "trait"
    file: str
    line: int
    column: int
    is_exported: bool = False
    is_used: bool = False
    imported_by: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "location": {"file": self.file, "line": self.line, "column": self.column},
            "isExported": self.is_exported,
            "isUsed": self.is_used,
            "importedBy": list(self.imported_by)
        }


@dataclass
class Module:
    """Represents a module (.sio file)"""
    path: str
    imports: Set[str] = field(default_factory=set)  # Other modules this imports
    exports: Dict[str, Symbol] = field(default_factory=dict)  # Exported symbols
    imports_from: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))  # module -> symbols
    is_entry_point: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "imports": list(self.imports),
            "exports": {k: v.to_dict() for k, v in self.exports.items()},
            "importsFrom": {k: list(v) for k, v in self.imports_from.items()},
            "isEntryPoint": self.is_entry_point
        }


@dataclass
class AnalysisResult:
    """Complete workspace analysis result"""
    modules: Dict[str, Module]
    unused_exports: List[Symbol]
    orphaned_modules: List[str]
    circular_dependencies: List[List[str]]
    entry_points: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "summary": {
                "totalModules": len(self.modules),
                "unusedExports": len(self.unused_exports),
                "orphanedModules": len(self.orphaned_modules),
                "circularDependencies": len(self.circular_dependencies),
                "entryPoints": len(self.entry_points)
            },
            "modules": {k: v.to_dict() for k, v in self.modules.items()},
            "unusedExports": [s.to_dict() for s in self.unused_exports],
            "orphanedModules": self.orphaned_modules,
            "circularDependencies": self.circular_dependencies,
            "entryPoints": self.entry_points
        }


class ModuleAnalyzer:
    """Analyzes a single module for imports and exports"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.content = Path(file_path).read_text()
        self.lines = self.content.split('\n')
    
    def analyze(self) -> Module:
        """Extract imports and exports from module"""
        module = Module(path=self.file_path)
        
        # Find imports
        self._extract_imports(module)
        
        # Find exports
        self._extract_exports(module)
        
        # Check if entry point
        module.is_entry_point = self._is_entry_point()
        
        return module
    
    def _extract_imports(self, module: Module):
        """Extract import statements"""
        # use module::name
        # use module::{name1, name2}
        # import module
        
        patterns = [
            # use std::io::println
            (r'\buse\s+([\w:]+)::(\w+)\s*;', "symbol"),
            # use std::io::{println, print}
            (r'\buse\s+([\w:]+)::\{([^}]+)\}\s*;', "multi"),
            # import std::io
            (r'\bimport\s+([\w:]+)\s*;', "module"),
        ]
        
        for pattern, kind in patterns:
            for match in re.finditer(pattern, self.content):
                if kind == "symbol":
                    mod_path = match.group(1)
                    symbol_name = match.group(2)
                    module.imports.add(mod_path)
                    module.imports_from[mod_path].add(symbol_name)
                
                elif kind == "multi":
                    mod_path = match.group(1)
                    symbols = [s.strip() for s in match.group(2).split(',')]
                    module.imports.add(mod_path)
                    for sym in symbols:
                        module.imports_from[mod_path].add(sym)
                
                elif kind == "module":
                    mod_path = match.group(1)
                    module.imports.add(mod_path)
    
    def _extract_exports(self, module: Module):
        """Extract exported symbols"""
        
        # Exported functions: pub fn name
        for match in re.finditer(r'^\s*pub\s+fn\s+(\w+)', self.content, re.MULTILINE):
            name = match.group(1)
            line = self.content[:match.start()].count('\n') + 1
            line_start = self.content.rfind('\n', 0, match.start()) + 1
            col = match.start() - line_start + 1
            
            symbol = Symbol(
                name=name,
                kind="function",
                file=self.file_path,
                line=line,
                column=col,
                is_exported=True
            )
            module.exports[name] = symbol
        
        # Exported types: pub struct, pub enum, pub trait, pub type
        type_patterns = [
            (r'^\s*pub\s+struct\s+(\w+)', "struct"),
            (r'^\s*pub\s+enum\s+(\w+)', "enum"),
            (r'^\s*pub\s+trait\s+(\w+)', "trait"),
            (r'^\s*pub\s+type\s+(\w+)', "type"),
        ]
        
        for pattern, kind in type_patterns:
            for match in re.finditer(pattern, self.content, re.MULTILINE):
                name = match.group(1)
                line = self.content[:match.start()].count('\n') + 1
                line_start = self.content.rfind('\n', 0, match.start()) + 1
                col = match.start() - line_start + 1
                
                symbol = Symbol(
                    name=name,
                    kind=kind,
                    file=self.file_path,
                    line=line,
                    column=col,
                    is_exported=True
                )
                module.exports[name] = symbol
        
        # Exported constants: pub const
        for match in re.finditer(r'^\s*pub\s+const\s+(\w+)', self.content, re.MULTILINE):
            name = match.group(1)
            line = self.content[:match.start()].count('\n') + 1
            line_start = self.content.rfind('\n', 0, match.start()) + 1
            col = match.start() - line_start + 1
            
            symbol = Symbol(
                name=name,
                kind="constant",
                file=self.file_path,
                line=line,
                column=col,
                is_exported=True
            )
            module.exports[name] = symbol
    
    def _is_entry_point(self) -> bool:
        """Check if this is an entry point (has main function)"""
        # Has fn main() at top level
        return bool(re.search(r'^\s*(?:pub\s+)?fn\s+main\s*\(', self.content, re.MULTILINE))


class SwarmAnalyzer:
    """
    Swarm Parallel Workspace Analyzer
    Uses process pool for CPU-bound parsing, thread pool for I/O
    """
    
    def __init__(self, workspace_path: str, max_workers: Optional[int] = None):
        self.workspace_path = Path(workspace_path)
        self.max_workers = max_workers or mp.cpu_count()
        self.modules: Dict[str, Module] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
    def find_all_files(self) -> List[str]:
        """Find all .sio files in workspace"""
        files = []
        for pattern in ['**/*.sio', '**/*.d']:
            files.extend(self.workspace_path.glob(pattern))
        return [str(f) for f in files if f.is_file()]
    
    def analyze_single_file(self, file_path: str) -> Tuple[str, Optional[Module]]:
        """Analyze a single file (for parallel execution)"""
        try:
            analyzer = ModuleAnalyzer(file_path)
            module = analyzer.analyze()
            return (file_path, module)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}", file=sys.stderr)
            return (file_path, None)
    
    def analyze_parallel(self, files: List[str], progress_callback: Optional[Callable] = None) -> Dict[str, Module]:
        """Analyze all files in parallel using process pool"""
        modules = {}
        completed = 0
        total = len(files)
        
        print(f"🚀 Swarm Analysis: Processing {total} files with {self.max_workers} workers...")
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.analyze_single_file, f): f 
                for f in files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path, module = future.result()
                completed += 1
                
                if module:
                    modules[file_path] = module
                
                if progress_callback:
                    progress_callback(completed, total)
                elif completed % 10 == 0 or completed == total:
                    print(f"  Progress: {completed}/{total} files ({100*completed//total}%)")
        
        elapsed = time.time() - start_time
        print(f"✅ Parallel analysis complete: {len(modules)} modules in {elapsed:.2f}s")
        print(f"   Throughput: {len(modules)/elapsed:.1f} modules/sec")
        
        return modules
    
    def build_dependency_graph(self):
        """Build graph of module dependencies"""
        for path, module in self.modules.items():
            self.dependency_graph[path] = set()
            
            for imported_mod in module.imports:
                # Resolve import to file path
                imported_path = self._resolve_import(imported_mod, path)
                if imported_path and imported_path in self.modules:
                    self.dependency_graph[path].add(imported_path)
    
    def _resolve_import(self, import_path: str, from_file: str) -> Optional[str]:
        """Resolve import path to actual file path"""
        # Simple resolution: convert :: to / and try .sio extension
        relative_path = import_path.replace('::', '/') + '.sio'
        
        # Try relative to current file
        base_dir = Path(from_file).parent
        candidate = base_dir / relative_path
        if candidate.exists():
            return str(candidate)
        
        # Try relative to workspace root
        candidate = self.workspace_path / relative_path
        if candidate.exists():
            return str(candidate)
        
        # Try stdlib
        stdlib_path = self.workspace_path / 'stdlib' / relative_path
        if stdlib_path.exists():
            return str(stdlib_path)
        
        return None
    
    def find_unused_exports(self) -> List[Symbol]:
        """Find symbols that are exported but never imported"""
        unused = []
        
        # Build reverse dependency map: symbol -> modules that import it
        symbol_imports: Dict[str, Set[str]] = defaultdict(set)
        
        for path, module in self.modules.items():
            for imported_mod_path in module.imports_from:
                for symbol_name in module.imports_from[imported_mod_path]:
                    key = f"{imported_mod_path}::{symbol_name}"
                    symbol_imports[key].add(path)
        
        # Check each export
        for path, module in self.modules.items():
            for symbol_name, symbol in module.exports.items():
                # Check if this symbol is imported anywhere
                is_used = False
                
                # Direct import
                for other_path, other_module in self.modules.items():
                    if other_path == path:
                        continue
                    
                    # Check if other module imports this symbol
                    if path in other_module.imports_from:
                        if symbol_name in other_module.imports_from[path]:
                            is_used = True
                            symbol.imported_by.add(other_path)
                            break
                
                if not is_used and not module.is_entry_point:
                    symbol.is_used = False
                    unused.append(symbol)
                else:
                    symbol.is_used = True
        
        return unused
    
    def find_orphaned_modules(self) -> List[str]:
        """Find modules that are not imported by any other module"""
        orphaned = []
        
        # Build set of all imported modules
        all_imported = set()
        for module in self.modules.values():
            for imported_path in self.dependency_graph.get(module.path, set()):
                all_imported.add(imported_path)
        
        # Find modules never imported (except entry points)
        for path, module in self.modules.items():
            if path not in all_imported and not module.is_entry_point:
                # Check if it's actually the main module
                if 'main' not in Path(path).name.lower():
                    orphaned.append(path)
        
        return orphaned
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies using DFS"""
        circles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    circles.append(cycle)
            
            path.pop()
            rec_stack.remove(node)
        
        for node in self.dependency_graph:
            if node not in visited:
                dfs(node)
        
        return circles
    
    def analyze(self) -> AnalysisResult:
        """Run complete workspace analysis"""
        # Find all files
        files = self.find_all_files()
        
        if not files:
            return AnalysisResult(
                modules={},
                unused_exports=[],
                orphaned_modules=[],
                circular_dependencies=[],
                entry_points=[]
            )
        
        # Parallel analysis
        self.modules = self.analyze_parallel(files)
        
        # Build dependency graph
        print("🔗 Building dependency graph...")
        self.build_dependency_graph()
        
        # Find issues
        print("🔍 Finding unused exports...")
        unused_exports = self.find_unused_exports()
        
        print("🔍 Finding orphaned modules...")
        orphaned_modules = self.find_orphaned_modules()
        
        print("🔍 Finding circular dependencies...")
        circular_deps = self.find_circular_dependencies()
        
        entry_points = [p for p, m in self.modules.items() if m.is_entry_point]
        
        return AnalysisResult(
            modules=self.modules,
            unused_exports=unused_exports,
            orphaned_modules=orphaned_modules,
            circular_dependencies=circular_deps,
            entry_points=entry_points
        )


def format_results(result: AnalysisResult, format_type: str = "text") -> str:
    """Format analysis results for output"""
    
    if format_type == "json":
        return json.dumps(result.to_dict(), indent=2)
    
    lines = []
    lines.append("=" * 60)
    lines.append("📊 WORKSPACE CROSS-FILE ANALYSIS")
    lines.append("=" * 60)
    
    # Summary
    summary = result.to_dict()["summary"]
    lines.append(f"\n📈 Summary:")
    lines.append(f"   Total modules: {summary['totalModules']}")
    lines.append(f"   Entry points: {summary['entryPoints']}")
    lines.append(f"   Unused exports: {summary['unusedExports']}")
    lines.append(f"   Orphaned modules: {summary['orphanedModules']}")
    lines.append(f"   Circular dependencies: {summary['circularDependencies']}")
    
    # Entry points
    if result.entry_points:
        lines.append(f"\n🎯 Entry Points:")
        for ep in result.entry_points[:5]:  # Show first 5
            lines.append(f"   • {Path(ep).name}")
        if len(result.entry_points) > 5:
            lines.append(f"   ... and {len(result.entry_points) - 5} more")
    
    # Unused exports
    if result.unused_exports:
        lines.append(f"\n📦 Unused Exports ({len(result.unused_exports)}):")
        for sym in result.unused_exports[:10]:  # Show first 10
            lines.append(f"   • [{sym.kind}] {sym.name}")
            lines.append(f"     at {Path(sym.file).name}:{sym.line}")
        if len(result.unused_exports) > 10:
            lines.append(f"   ... and {len(result.unused_exports) - 10} more")
    
    # Orphaned modules
    if result.orphaned_modules:
        lines.append(f"\n🗑️ Orphaned Modules ({len(result.orphaned_modules)}):")
        for mod in result.orphaned_modules[:10]:
            lines.append(f"   • {Path(mod).name}")
        if len(result.orphaned_modules) > 10:
            lines.append(f"   ... and {len(result.orphaned_modules) - 10} more")
    
    # Circular dependencies
    if result.circular_dependencies:
        lines.append(f"\n🔄 Circular Dependencies ({len(result.circular_dependencies)}):")
        for i, cycle in enumerate(result.circular_dependencies[:5], 1):
            cycle_str = " → ".join([Path(p).name for p in cycle[:5]])
            if len(cycle) > 5:
                cycle_str += f" → ... ({len(cycle)-5} more)"
            lines.append(f"   Cycle {i}: {cycle_str}")
    
    # Dependency graph stats
    if result.modules:
        total_deps = sum(len(m.imports) for m in result.modules.values())
        avg_deps = total_deps / len(result.modules)
        lines.append(f"\n📊 Dependency Stats:")
        lines.append(f"   Total dependencies: {total_deps}")
        lines.append(f"   Average per module: {avg_deps:.1f}")
    
    lines.append("")
    return '\n'.join(lines)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cross-File Workspace Analyzer for Sounio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./src                           # Analyze workspace
  %(prog)s ./src --format json             # Output JSON
  %(prog)s ./src --workers 8               # Use 8 parallel workers
  %(prog)s ./src --exclude test_           # Exclude test files
        """
    )
    
    parser.add_argument("path", help="Workspace directory to analyze")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                       help="Output format")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--exclude", nargs="+", default=[],
                       help="Patterns to exclude (e.g., test_ vendor/)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SwarmAnalyzer(args.path, max_workers=args.workers)
    
    # Run analysis
    result = analyzer.analyze()
    
    # Output results
    output = format_results(result, args.format)
    print(output)
    
    # Exit with error code if issues found
    issues = len(result.unused_exports) + len(result.orphaned_modules) + len(result.circular_dependencies)
    if issues > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
