import { useState, useEffect, useRef, useCallback } from 'react';

interface PlaygroundEditorProps {
  initialCode?: string;
  theme?: 'light' | 'dark';
}

const EXAMPLES = [
  {
    name: 'Hello World',
    code: `// Hello World in Sounio
fn main() with IO {
    print("Hello, World!")
}`,
  },
  {
    name: 'Uncertainty',
    code: `// Epistemic computing with native uncertainty
let mass: Knowledge<kg> = measure(
    value: 75.3,
    uncertainty: 0.5,
    source: "clinical_scale_001"
)

let height: Knowledge<m> = measure(
    value: 1.82,
    uncertainty: 0.01,
    source: "stadiometer_001"
)

// Uncertainty propagates automatically (GUM-compliant)
let bmi = mass / (height * height)

// Confidence gates control execution
if bmi.confidence > 0.95 {
    report_bmi(bmi)
} else {
    request_remeasurement()
}`,
  },
  {
    name: 'Effects',
    code: `// Algebraic effects for controlled side effects
fn read_sensor() -> f64 with IO, Fail {
    let raw = perform IO::read("/dev/sensor0")?
    let value = parse_float(raw)?
    value
}

// Effect handlers provide the implementation
fn main() with IO {
    handle read_sensor() {
        IO::read(path) => resume(fs::read_to_string(path)),
        Fail::error(msg) => {
            log_error(msg)
            resume(0.0)
        }
    }
}`,
  },
  {
    name: 'GPU Kernels',
    code: `// Native GPU kernel syntax
@kernel
fn matrix_multiply(
    a: &Tensor<f32, [M, K]>,
    b: &Tensor<f32, [K, N]>,
    c: &!Tensor<f32, [M, N]>
) with GPU {
    let row = gpu::thread_idx_y()
    let col = gpu::thread_idx_x()

    var sum: f32 = 0.0
    for k in 0..K {
        sum += a[row, k] * b[k, col]
    }
    c[row, col] = sum
}

// Launch with automatic grid/block sizing
let result = matrix_multiply<<<grid, block>>>(a, b)`,
  },
  {
    name: 'Units',
    code: `// Type-safe units of measure
let dose: mg = 500.0
let volume: mL = 250.0
let concentration: mg/mL = dose / volume

// Compile-time dimension checking
let mass: kg = 1.5
// let error = mass + volume  // Compile error!

// Unit conversions
let mass_mg: mg = mass.to::<mg>()  // 1500.0 mg

// Scientific constants with units
let c: m/s = SPEED_OF_LIGHT
let energy: J = mass * c * c`,
  },
];

export default function PlaygroundEditor({ initialCode, theme = 'dark' }: PlaygroundEditorProps) {
  const [code, setCode] = useState(initialCode || EXAMPLES[0].code);
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [selectedExample, setSelectedExample] = useState(EXAMPLES[0].name);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Load code from URL on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const urlCode = params.get('code');
    if (urlCode) {
      try {
        const decoded = atob(urlCode);
        setCode(decoded);
      } catch {
        // Invalid base64, ignore
      }
    }
  }, []);

  const handleRun = useCallback(async () => {
    setIsRunning(true);
    setOutput('Compiling...\n');

    // Simulate compilation delay
    await new Promise(resolve => setTimeout(resolve, 500));

    // Demo output based on code content
    let result = '✓ Compiled successfully in 42ms\n\nRunning...\n';

    if (code.includes('Knowledge<')) {
      result += 'BMI: 22.73 ± 0.31 kg/m² (confidence: 0.95)\n\nSource provenance:\n  └─ mass: clinical_scale_001\n  └─ height: stadiometer_001';
    } else if (code.includes('@kernel')) {
      result += 'GPU kernel compiled to PTX\nLaunching on device 0 (NVIDIA GeForce RTX 4090)\nMatrix multiplication: 1024x1024 @ 12.4 TFLOPS\n✓ Completed in 0.82ms';
    } else if (code.includes('perform IO')) {
      result += 'Sensor reading: 23.7°C\nEffect handled: IO::read\n✓ Completed';
    } else if (code.includes('mg =')) {
      result += 'Concentration: 2.0 mg/mL\nMass conversion: 1500.0 mg\nEnergy: 1.35e17 J';
    } else {
      result += 'Hello, World!';
    }

    setOutput(result);
    setIsRunning(false);
  }, [code]);

  const handleShare = useCallback(() => {
    const encoded = btoa(code);
    const url = new URL(window.location.href);
    url.searchParams.set('code', encoded);
    navigator.clipboard.writeText(url.toString());
    setOutput('✓ Link copied to clipboard!');
  }, [code]);

  const handleExampleChange = useCallback((name: string) => {
    const example = EXAMPLES.find(e => e.name === name);
    if (example) {
      setCode(example.code);
      setSelectedExample(name);
      setOutput('');
    }
  }, []);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleRun();
    }
  }, [handleRun]);

  return (
    <div className={`flex flex-col h-full ${theme === 'dark' ? 'bg-[#1e1e1e]' : 'bg-white'}`}>
      {/* Toolbar */}
      <div className="flex items-center justify-between p-3 bg-[var(--color-navy-900)] border-b border-white/10">
        <div className="flex items-center gap-3">
          <button
            onClick={handleRun}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 disabled:bg-green-500/50 text-white rounded-lg font-medium transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            </svg>
            {isRunning ? 'Running...' : 'Run'}
          </button>

          <button
            onClick={handleShare}
            className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg font-medium transition-colors"
          >
            Share
          </button>

          <select
            value={selectedExample}
            onChange={(e) => handleExampleChange(e.target.value)}
            className="px-3 py-2 bg-white/10 text-white rounded-lg font-medium border-0 cursor-pointer"
          >
            {EXAMPLES.map(ex => (
              <option key={ex.name} value={ex.name} className="bg-[var(--color-navy-900)]">
                {ex.name}
              </option>
            ))}
          </select>
        </div>

        <span className="text-white/50 text-sm">Ctrl+Enter to run</span>
      </div>

      {/* Editor and Output */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 min-h-0">
        {/* Code Editor */}
        <div className="border-r border-white/10 flex flex-col">
          <div className="p-2 text-white/60 text-xs font-mono border-b border-white/10">
            main.sio
          </div>
          <textarea
            ref={textareaRef}
            value={code}
            onChange={(e) => setCode(e.target.value)}
            onKeyDown={handleKeyDown}
            className={`flex-1 p-4 font-mono text-sm leading-relaxed resize-none focus:outline-none ${
              theme === 'dark'
                ? 'bg-[#1e1e1e] text-[#d4d4d4]'
                : 'bg-white text-gray-800'
            }`}
            spellCheck={false}
          />
        </div>

        {/* Output Panel */}
        <div className="flex flex-col">
          <div className="p-2 text-white/60 text-xs font-mono border-b border-white/10">
            Output
          </div>
          <pre className={`flex-1 p-4 font-mono text-sm overflow-auto ${
            theme === 'dark'
              ? 'bg-[#1e1e1e] text-green-400'
              : 'bg-gray-50 text-gray-800'
          }`}>
            {output || 'Click "Run" to execute your code...'}
          </pre>
        </div>
      </div>

      {/* Status Bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-[var(--color-navy-900)] text-white/60 text-xs font-mono border-t border-white/10">
        <span>Sounio v0.99.0</span>
        <span>UTF-8</span>
      </div>
    </div>
  );
}
