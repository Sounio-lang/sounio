"""Jupyter magic commands for Sounio kernel."""

import importlib
import json
import os
import re
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class SounioMagics:
    """IPython-style magic commands for Sounio kernel."""

    def __init__(self, kernel: Any) -> None:
        """Initialize magics with kernel reference."""
        self.kernel = kernel
        self.executor = kernel.executor

    def handle_magic(self, code: str) -> Tuple[bool, str]:
        """
        Handle a magic command.

        Args:
            code: The magic command line

        Returns:
            Tuple of (handled, result_message)
        """
        parts = code.strip().split(None, 1)
        magic_name = parts[0]
        magic_args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "%time": self.magic_time,
            "%timeit": self.magic_timeit,
            "%writefile": self.magic_writefile,
            "%sounio": self.magic_sounio,
            "%check": self.magic_check,
            "%reset": self.magic_reset,
            "%ast": self.magic_ast,
            "%types": self.magic_types,
            "%ontology_search": self.magic_ontology_search,
            "%ontology_resolve": self.magic_ontology_resolve,
            "%clinical_normalize": self.magic_clinical_normalize,
            "%drug_pipeline": self.magic_drug_pipeline,
            "%python": self.magic_python,
        }

        if magic_name not in handlers:
            return False, ""

        handler = handlers[magic_name]
        try:
            result = handler(magic_args)
            return True, result
        except Exception as e:
            return True, f"Error executing {magic_name}: {str(e)}"

    def magic_time(self, line: str) -> str:
        """
        Time a single statement.

        Usage:
            %time <code>
        """
        if not line.strip():
            return "Usage: %time <sounio code>"

        code = line.strip()
        start = time.time()
        stdout, stderr, exitcode = self.executor.run_cell(code)
        elapsed = time.time() - start

        result = f"CPU time: {elapsed:.4f}s\n"
        if stdout:
            result += f"Output:\n{stdout}"
        if stderr and exitcode != 0:
            result += f"Error:\n{stderr}"

        return result

    def magic_timeit(self, line: str) -> str:
        """
        Time statement multiple times.

        Usage:
            %timeit <code>
            %timeit -n 100 <code>
        """
        # Parse number of iterations
        match = re.match(r"-n\s+(\d+)\s+(.*)", line.strip())
        if match:
            number = int(match.group(1))
            code = match.group(2)
        else:
            number = 3
            code = line.strip()

        if not code:
            return "Usage: %timeit [-n <number>] <sounio code>"

        times = []
        for _ in range(number):
            start = time.time()
            stdout, stderr, exitcode = self.executor.run_cell(code)
            times.append(time.time() - start)

        min_time = min(times)
        max_time = max(times)
        avg_time = sum(times) / len(times)

        result = f"Timeit results ({number} runs):\n"
        result += f"  Min: {min_time:.4f}s\n"
        result += f"  Max: {max_time:.4f}s\n"
        result += f"  Avg: {avg_time:.4f}s\n"

        return result

    def magic_reset(self, _line: str) -> str:
        """Clear the accumulated session declarations.

        Usage:
            %reset
        """
        self.executor.reset_session()
        return "Session reset."

    def magic_writefile(self, line: str, cell: Optional[str] = None) -> str:
        """
        Write cell content to a file.

        Usage:
            %%writefile filename.sio
            <code>
        """
        parts = line.strip().split()
        if not parts:
            return "Usage: %%writefile <filename>"

        filename = parts[0]

        # Get cell content from parent kernel
        # Note: This is called from the kernel's cell magic handler
        # which provides the cell content separately
        if cell:
            try:
                with open(filename, "w") as f:
                    f.write(cell)
                return f"Wrote {len(cell)} chars to {filename}"
            except Exception as e:
                return f"Error writing file: {str(e)}"

        return "Usage: %%writefile <filename> (use as cell magic)"

    def magic_sounio(self, line: str) -> str:
        """
        Sounio kernel information and settings.

        Usage:
            %sounio info       - Show kernel version and config
            %sounio stdlib     - Show stdlib path
            %sounio souc       - Show souc binary path
            %sounio session    - Show accumulated session source
        """
        subcmd = line.strip().lower() if line.strip() else "info"

        if subcmd == "info":
            return f"""Sounio Kernel v0.2.0
Language: Sounio 1.0.0-beta.4
Stdlib: {self.executor.stdlib_path or 'not found'}
Souc binary: {self.executor.souc_binary or 'not found'}
Session declarations: {len(getattr(self.executor, '_declarations', []))}
Features: Epistemic programming, uncertainty quantification, provenance tracking, session persistence"""

        elif subcmd == "stdlib":
            if self.executor.stdlib_path:
                return f"Stdlib path: {self.executor.stdlib_path}"
            else:
                return "Stdlib path not found. Set SOUNIO_STDLIB_PATH environment variable."

        elif subcmd == "souc":
            if self.executor.souc_binary:
                return f"Souc binary: {self.executor.souc_binary}"
            else:
                return "Souc binary not found. Set SOUC environment variable."

        elif subcmd == "session":
            src = self.executor.get_session_source()
            decls = getattr(self.executor, '_declarations', [])
            if src:
                return f"Current session ({len(decls)} declarations):\n{src}"
            return "Session is empty."

        else:
            return """Available %sounio subcommands:
  %sounio info     - Show kernel version and config
  %sounio stdlib   - Show stdlib path
  %sounio souc     - Show souc binary path
  %sounio session  - Show accumulated session source"""

    def _ontology_module(self):
        """Import ontology helpers from sounio-py, adding the repo path when needed."""
        try:
            return importlib.import_module("sounio.ontology")
        except ImportError:
            repo_root = Path(__file__).resolve().parents[3]
            py_pkg = repo_root / "ecosystem" / "sounio-py" / "python"
            if str(py_pkg) not in sys.path:
                sys.path.insert(0, str(py_pkg))
            return importlib.import_module("sounio.ontology")

    def magic_ontology_search(self, line: str) -> str:
        """Search ontology bundles by text.

        Usage:
            %ontology_search glucose
            %ontology_search diabetes
        """
        query = line.strip()
        if not query:
            return "Usage: %ontology_search <text>"
        ontology = self._ontology_module()
        results = ontology.search(query)
        if not results:
            return f"No ontology hits for: {query}"
        return json.dumps([item.to_dict() for item in results], indent=2)

    def magic_ontology_resolve(self, line: str) -> str:
        """Resolve a CURIE to the canonical ontology term object.

        Usage:
            %ontology_resolve SNOMED:44054006
        """
        curie = line.strip()
        if not curie:
            return "Usage: %ontology_resolve <CURIE>"
        ontology = self._ontology_module()
        term = ontology.resolve(curie)
        if term is None:
            return f"Unresolved ontology term: {curie}"
        return json.dumps(term.to_dict(), indent=2)

    def magic_clinical_normalize(self, line: str) -> str:
        """Normalize a small clinical payload using the compiler-backed ontology bundle model.

        Usage:
            %clinical_normalize {"patient_id":"pt-1","diagnoses":["SNOMED:44054006"],"labs":["LOINC:4548-4"],"phenotypes":["HPO:0003074"]}
            %clinical_normalize fixtures/patient.json
        """
        raw = line.strip()
        if not raw:
            return "Usage: %clinical_normalize <json-payload-or-path>"

        payload: Dict[str, Any]
        if raw.startswith("{"):
            payload = json.loads(raw)
        else:
            path = Path(raw)
            if not path.exists():
                return f"Input not found: {raw}"
            payload = json.loads(path.read_text())

        ontology = self._ontology_module()
        normalized = ontology.clinical_normalize(payload)
        return json.dumps(normalized, indent=2)

    def magic_check(self, line: str) -> str:
        """
        Type-check code without executing.

        Usage:
            %check <code>
        """
        if not line.strip():
            return "Usage: %check <sounio code>"

        code = line.strip()

        # Build source with session context
        session = self.executor.get_session_source()
        if not code.startswith("fn "):
            code = f"fn main() with IO {{\n    {code}\n}}"

        src = f"{session}\n{code}\n"

        ok, err = self.executor._compile_source_str(src)
        if ok:
            return "✓ Type check passed"
        return f"✗ Type check failed:\n{err}"

    def magic_ast(self, line: str) -> str:
        """
        Show AST for code.

        Usage:
            %ast <code>
        """
        if not line.strip():
            return "Usage: %ast <sounio code>"

        code = line.strip()

        # Wrap in main if needed
        if not code.startswith("fn "):
            code = f"fn main() with IO {{\n    {code}\n}}"

        # Write to temp file and show AST
        with tempfile.NamedTemporaryFile(suffix=".sio", mode="w", delete=False) as f:
            f.write(code)
            f.flush()
            temp_file = f.name

        try:
            import subprocess

            env = os.environ.copy()
            if self.executor.stdlib_path:
                env["SOUNIO_STDLIB_PATH"] = self.executor.stdlib_path

            result = subprocess.run(
                [self.executor.souc_binary, "check", "--show-ast", temp_file],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )

            return result.stdout + result.stderr
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def magic_types(self, line: str) -> str:
        """
        Show inferred types.

        Usage:
            %types <code>
        """
        if not line.strip():
            return "Usage: %types <sounio code>"

        code = line.strip()

        # Wrap in main if needed
        if not code.startswith("fn "):
            code = f"fn main() with IO {{\n    {code}\n}}"

        # Write to temp file and show types
        with tempfile.NamedTemporaryFile(suffix=".sio", mode="w", delete=False) as f:
            f.write(code)
            f.flush()
            temp_file = f.name

        try:
            import subprocess

            env = os.environ.copy()
            if self.executor.stdlib_path:
                env["SOUNIO_STDLIB_PATH"] = self.executor.stdlib_path

            result = subprocess.run(
                [self.executor.souc_binary, "check", "--show-types", temp_file],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )

            return result.stdout + result.stderr
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def magic_drug_pipeline(self, line: str) -> str:
        """
        Execute the drug discovery pipeline.

        Uses sounio.run_file() to execute the pipeline and returns rich HTML output
        with epistemic uncertainty visualization.

        Usage:
            %drug_pipeline                     # Use default pipeline
            %drug_pipeline path/to/pipeline.sio  # Use custom pipeline file
        """
        # Get pipeline path from args, or use default
        pipeline_path = line.strip() if line.strip() else "drug-discovery/examples/full_pipeline.sio"

        # Try to use sounio-py if available
        try:
            from sounio import run_file
        except ImportError:
            return """Error: sounio-py not installed.
Install it with: pip install sounio
Or use the souc binary directly."""

        try:
            # Check if file exists
            if not os.path.exists(pipeline_path):
                return f"Error: Pipeline file not found: {pipeline_path}"

            # Execute the pipeline via sounio.run_file()
            result = run_file(pipeline_path)

            if not result.ok:
                return f"Pipeline execution failed:\n{result.stderr}"

            # Format output with epistemic values
            output_html = self._format_drug_pipeline_output(
                result.stdout,
                result.knowledge_values,
            )

            return output_html

        except Exception as e:
            return f"Error executing drug pipeline: {str(e)}"

    def magic_python(self, line: str) -> str:
        """Execute Python code with Sounio Knowledge values injected into scope.

        Knowledge values from the last Sounio execution are available as
        ``k0``, ``k1``, ... and as the list ``knowledge_values``.

        Usage
        -----
            %python import math; print(math.pi)
            %python print(k0.value, k0.provenance)  # k0 is first Knowledge value
            %python print([k.confidence for k in knowledge_values])

        For multi-line Python, use %%python (cell magic) — write the magic on
        the first line and the code in the rest of the cell.
        """
        code = line.strip()
        if not code:
            return "Usage: %python <python code>  (or %%python for multi-line)"

        preamble = self._build_python_preamble()
        full_code = preamble + "\n" + code

        import subprocess

        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=30,
        )

        out = result.stdout
        if result.returncode != 0 and result.stderr:
            out += ("\n" if out else "") + "[stderr]\n" + result.stderr
        return out if out.strip() else "(no output)"

    def _build_python_preamble(self) -> str:
        """Build a Python preamble injecting last Sounio Knowledge values."""
        last_knowledge = getattr(self.kernel, "_last_knowledge_values", [])
        lines = [
            "try:",
            "    from sounio.knowledge import Knowledge as _K",
            "except ImportError:",
            "    class _K:",
            "        def __init__(self, v, e, p=''):",
            "            self.value=v; self.epsilon=e; self.provenance=p",
            "            self.relative_uncertainty = abs(e/v) if v else 0",
            "            self.confidence = max(0, min(1, 1-self.relative_uncertainty))",
            "        def __repr__(self):",
            "            return f'Knowledge({self.value} \\u00b1 {self.epsilon})'",
        ]
        if last_knowledge:
            for i, kv in enumerate(last_knowledge):
                lines.append(
                    f"k{i} = _K({kv.value!r}, {kv.epsilon!r}, {kv.provenance!r})"
                )
            lines.append(
                f"knowledge_values = [{', '.join(f'k{i}' for i in range(len(last_knowledge)))}]"
            )
        else:
            lines.append("knowledge_values = []")
        return "\n".join(lines) + "\n"

    def _format_drug_pipeline_output(
        self,
        stdout: str,
        knowledge_values: Optional[list] = None,
    ) -> str:
        """
        Format drug pipeline output with epistemic value visualization.

        Args:
            stdout: Raw stdout from the pipeline
            knowledge_values: List of Knowledge values parsed from output

        Returns:
            HTML string for Jupyter display
        """
        # Import display utilities
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "display",
            os.path.join(os.path.dirname(__file__), "display.py")
        )
        display_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(display_module)
        format_knowledge_html = display_module.format_knowledge_html

        # Build HTML header
        html_parts = [
            '<div class="drug-pipeline-output" style="'
            '    font-family: monospace; '
            '    background-color: #f8f9fa; '
            '    border: 1px solid #ddd; '
            '    border-radius: 4px; '
            '    padding: 12px; '
            '    margin: 8px 0; '
            '">'
        ]

        # Add raw output section
        if stdout:
            html_parts.append("<h4>Pipeline Output:</h4>")
            html_parts.append(
                f"<pre style='background-color: white; "
                f"padding: 8px; border-radius: 3px; overflow-x: auto;'>"
                f"{stdout}</pre>"
            )

        # Add epistemic values section if present
        if knowledge_values:
            html_parts.append("<h4>Epistemic Values:</h4>")
            for kv in knowledge_values:
                # Use display format function for each knowledge value
                epistemic_html = format_knowledge_html(kv.value, kv.epsilon, kv.prov)
                html_parts.append(epistemic_html)

        html_parts.append("</div>")

        return "\n".join(html_parts)
