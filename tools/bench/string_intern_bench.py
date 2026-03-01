#!/usr/bin/env python3
"""
String Interning Benchmark for Sounio
Measures memory savings from string deduplication
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
import re


@dataclass
class StringStats:
    """Statistics about string usage"""
    total_strings: int = 0
    total_bytes: int = 0
    unique_strings: int = 0
    unique_bytes: int = 0
    duplicates: int = 0
    duplicate_bytes: int = 0
    
    def savings_pct(self) -> float:
        """Calculate memory savings percentage"""
        if self.total_bytes == 0:
            return 0.0
        return (self.duplicate_bytes / self.total_bytes) * 100
    
    def to_dict(self) -> Dict:
        return {
            "totalStrings": self.total_strings,
            "totalBytes": self.total_bytes,
            "uniqueStrings": self.unique_strings,
            "uniqueBytes": self.unique_bytes,
            "duplicates": self.duplicates,
            "duplicateBytes": self.duplicate_bytes,
            "savingsPercent": round(self.savings_pct(), 2),
            "deduplicationRatio": round(self.total_strings / max(self.unique_strings, 1), 2)
        }


class StringInternSimulator:
    """Simulates string interning to measure potential savings"""
    
    def __init__(self):
        self.interned: Dict[str, int] = {}
        self.strings: list = []
        self.stats = StringStats()
    
    def intern(self, s: str) -> int:
        """Intern a string, returning its ID"""
        self.stats.total_strings += 1
        self.stats.total_bytes += len(s.encode('utf-8'))
        
        if s in self.interned:
            self.stats.duplicates += 1
            self.stats.duplicate_bytes += len(s.encode('utf-8'))
            return self.interned[s]
        
        id = len(self.strings)
        self.interned[s] = id
        self.strings.append(s)
        self.stats.unique_strings += 1
        self.stats.unique_bytes += len(s.encode('utf-8'))
        return id
    
    def get_stats(self) -> StringStats:
        return self.stats


def analyze_file(file_path: str, interner: StringInternSimulator):
    """Analyze a single file for string duplication"""
    content = Path(file_path).read_text()
    
    # Function names: fn name(...)
    for match in re.finditer(r'\bfn\s+(\w+)', content):
        interner.intern(match.group(1))
    
    # Variable names: let name = ...
    for match in re.finditer(r'\blet\s+(?:mut\s+)?(\w+)', content):
        interner.intern(match.group(1))
    
    # Type names
    for match in re.finditer(r'\b(?:struct|enum|trait|type)\s+(\w+)', content):
        interner.intern(match.group(1))
    
    # String literals
    for match in re.finditer(r'"([^"]*)"', content):
        interner.intern(match.group(1))
    
    # Import paths and components
    for match in re.finditer(r'\buse\s+([\w:]+)', content):
        path = match.group(1)
        interner.intern(path)
        for part in path.split('::'):
            interner.intern(part)
    
    # Common keywords
    for kw in ['fn', 'let', 'mut', 'if', 'else', 'match', 'return', 'with', 'IO', 
               'i32', 'i64', 'bool', 'string', 'true', 'false', 'self', 'Self']:
        interner.intern(kw)


def benchmark_workspace(workspace_path: str) -> Dict:
    """Benchmark string interning across entire workspace"""
    
    print(f"🔬 String Interning Benchmark: {workspace_path}")
    print("=" * 60)
    
    interner = StringInternSimulator()
    files = list(Path(workspace_path).rglob("*.sio"))
    
    print(f"📁 Found {len(files)} files\n")
    
    start_time = time.time()
    
    for i, file_path in enumerate(files):
        try:
            analyze_file(str(file_path), interner)
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(files)} files...")
        except Exception as e:
            print(f"  Warning: Error in {file_path}: {e}")
    
    elapsed = time.time() - start_time
    
    stats = interner.get_stats()
    
    print(f"\n✅ Complete in {elapsed:.2f}s")
    print(f"   Throughput: {len(files)/elapsed:.1f} files/sec")
    
    return {
        "summary": stats.to_dict(),
        "filesAnalyzed": len(files),
        "timeSeconds": elapsed
    }


def format_results(results: Dict) -> str:
    """Format benchmark results for display"""
    
    summary = results["summary"]
    
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("📊 STRING INTERNING BENCHMARK RESULTS")
    lines.append("=" * 60)
    
    lines.append(f"\n📈 String Statistics:")
    lines.append(f"   Total strings: {summary['totalStrings']:,}")
    lines.append(f"   Total bytes: {summary['totalBytes']:,}")
    lines.append(f"   Unique strings: {summary['uniqueStrings']:,}")
    lines.append(f"   Unique bytes: {summary['uniqueBytes']:,}")
    
    lines.append(f"\n💾 Memory Impact:")
    lines.append(f"   Duplicates: {summary['duplicates']:,}")
    lines.append(f"   Duplicate bytes: {summary['duplicateBytes']:,}")
    lines.append(f"   🎯 SAVINGS: {summary['savingsPercent']}%")
    lines.append(f"   Deduplication ratio: {summary['deduplicationRatio']}x")
    
    # Memory visualization
    total_mb = summary['totalBytes'] / (1024 * 1024)
    unique_mb = summary['uniqueBytes'] / (1024 * 1024)
    
    lines.append(f"\n📦 Memory Usage:")
    lines.append(f"   Without interning: {total_mb:.2f} MB")
    lines.append(f"   With interning: {unique_mb:.2f} MB")
    lines.append(f"   Saved: {total_mb - unique_mb:.2f} MB")
    
    # Visual bar
    bar_width = 40
    if summary['totalBytes'] > 0:
        unique_pct = summary['uniqueBytes'] / summary['totalBytes']
        unique_width = int(bar_width * unique_pct)
        saved_width = bar_width - unique_width
        lines.append(f"\n   [{'█' * unique_width}{'░' * saved_width}] {summary['savingsPercent']}% saved")
    
    lines.append("")
    return '\n'.join(lines)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="String Interning Benchmark")
    parser.add_argument("path", help="Workspace directory")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    args = parser.parse_args()
    
    results = benchmark_workspace(args.path)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_results(results))


if __name__ == "__main__":
    main()
