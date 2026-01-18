# Show epistemic information

Display epistemic computing information: Knowledge types, confidence levels, provenance chains, and uncertainty.

## Arguments
- `<file>` - Sounio source file (required)
- `--show-confidence` - Display confidence levels for Knowledge values
- `--show-provenance` - Display provenance chains (data origin tracking)
- `--show-uncertainty` - Display uncertainty bounds
- `--show-temporal` - Display temporal indices
- `--all` - Show all epistemic information

## Examples
- `/sounio-epistemic examples/epistemic_basic.sio` - Basic epistemic info
- `/sounio-epistemic examples/measurement.sio --show-confidence` - Confidence levels
- `/sounio-epistemic examples/data_pipeline.sio --show-provenance` - Data lineage
- `/sounio-epistemic examples/sensor.sio --all` - All epistemic metadata

$ARGUMENTS

Execute from the `compiler/` directory:

1. Validate that a file path is provided

2. Parse and analyze the file for epistemic constructs:
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo run -- check <file> --show-types
   ```

3. Extract and display epistemic information:

   **Knowledge<T> types:**
   - Base type T
   - Confidence level (0.0 to 1.0)
   - Provenance chain
   - Temporal index

4. Confidence display (`--show-confidence`):
   - Show confidence values for all Knowledge expressions
   - Highlight low-confidence values (< 0.5)
   - Show confidence propagation through operations

5. Provenance display (`--show-provenance`):
   - Show origin of data (sensor, computation, user input)
   - Display transformation chain
   - Track data lineage through the program

6. Uncertainty display (`--show-uncertainty`):
   - Show uncertainty bounds (e.g., 95% confidence intervals)
   - Display measurement error propagation

7. For interactive exploration, suggest using `/sounio-repl` with epistemic commands:
   - `:confidence <expr>` - Get confidence of expression
   - `:provenance <expr>` - Get provenance chain
   - `:uncertainty <expr>` - Get uncertainty bounds
