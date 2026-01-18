# Analyze code for metrics and issues

Analyze Sounio code for metrics, complexity, dead code, and potential issues.

## Arguments
- `<path>` - File or directory to analyze (default: current directory)
- `--analysis <type>` - Analysis type: metrics, dead-code, complexity, all (default: all)
- `--format <text|json>` - Output format (default: text)
- `--verbose` - Show detailed output

## Examples
- `/sounio-analyze examples/` - Full analysis of examples
- `/sounio-analyze src/module.sio --analysis metrics` - Code metrics only
- `/sounio-analyze --analysis dead-code` - Find dead code
- `/sounio-analyze --analysis complexity --verbose` - Detailed complexity

$ARGUMENTS

Execute from the `compiler/` directory:

1. Determine target path (default to current directory if not specified)

2. Run the analyze command:
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo run -- analyze <path> [--analysis <type>]
   ```

3. Analysis types:
   - `metrics` - Lines of code, function count, type definitions, effect usage
   - `dead-code` - Unused functions, unreachable code, unused imports
   - `complexity` - Cyclomatic complexity, nesting depth, function size
   - `all` - Run all analyses

4. Metrics to report:
   - Total lines / code lines / comment lines
   - Function count and average size
   - Type definitions (structs, enums, aliases)
   - Effect annotations usage
   - Dependency graph complexity

5. For dead code analysis:
   - List unused items with file:line locations
   - Suggest removals

6. For complexity analysis:
   - Flag functions exceeding thresholds
   - Suggest refactoring opportunities
