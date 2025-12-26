/**
 * Sounio Playground - Main Entry Point
 *
 * This file initializes the Monaco editor with Sounio syntax highlighting,
 * loads the WASM compiler module, and handles user interactions.
 */

import { SounioCompiler, Diagnostic } from './compiler';
import { generateShareUrl, loadFromUrl } from './share';

declare const require: any;
declare const LZString: any;

// Monaco editor instance
let editor: any = null;

// Compiler instance
let compiler: SounioCompiler | null = null;

// Example programs
const EXAMPLES: Record<string, string> = {
    hello: `// Hello World in Sounio
module hello

fn main() -> i32 {
    let message = "Hello, Sounio!"
    print(message)
    0
}
`,
    units: `// Units of Measure - Dimensional Analysis
module units_demo

// Sounio tracks units at compile time
let dose: mg = 500.0
let volume: mL = 100.0

// Unit conversion is automatic and type-safe
let concentration: mg/mL = dose / volume

// This would be a compile error:
// let wrong: kg = dose  // Error: cannot assign mg to kg

fn calculate_infusion(
    target_conc: mg/L,
    clearance: L/h,
    bioavailability: f64
) -> mg/h {
    let rate = target_conc * clearance / bioavailability
    rate
}

fn main() -> i32 {
    let rate = calculate_infusion(10.0 mg/L, 5.0 L/h, 0.8)
    print("Infusion rate: ", rate)
    0
}
`,
    effects: `// Algebraic Effects - Safe Side Effects
module effects_demo

// Effects are declared in function signatures
fn read_config(path: string) -> Config with IO {
    let contents = read_file(path)
    parse_config(contents)
}

// Multiple effects can be combined
fn process_data(input: &[f64]) -> Vec<f64> with IO, Alloc {
    var result = Vec::new()
    for x in input {
        result.push(x * 2.0)
    }
    log("Processed {} elements", result.len())
    result
}

// Effect handlers control how effects are interpreted
fn main() -> i32 with IO {
    let config = read_config("config.json")
    print("Config loaded: ", config.name)
    0
}
`,
    linear: `// Linear Types - Resource Safety
module linear_demo

// Linear types must be used exactly once
linear struct FileHandle {
    fd: i32
}

fn open_file(path: string) -> FileHandle with IO {
    let fd = syscall_open(path)
    FileHandle { fd }
}

fn close_file(handle: FileHandle) with IO {
    syscall_close(handle.fd)
    // handle is consumed here
}

fn main() -> i32 with IO {
    let file = open_file("data.txt")
    // If we forget to close, compiler error!
    close_file(file)
    0
}
`,
    refinement: `// Refinement Types - Compile-time Constraints
module refinement_demo

// Define refined types with constraints
type Positive = { x: i32 | x > 0 }
type Percentage = { p: f64 | p >= 0.0 && p <= 100.0 }
type NonEmpty<T> = { arr: [T] | arr.len() > 0 }

// Functions can require refined arguments
fn divide(a: i32, b: { x: i32 | x != 0 }) -> i32 {
    a / b
}

fn calculate_mean(data: NonEmpty<f64>) -> f64 {
    let sum = data.iter().sum()
    sum / (data.len() as f64)
}

fn main() -> i32 {
    // This is valid
    let result = divide(10, 2)

    // This would be caught at compile time:
    // let bad = divide(10, 0)  // Error: refinement violated

    print("Result: ", result)
    0
}
`,
    epistemic: `// Epistemic State - Uncertainty Tracking
module epistemic_demo

// Sounio tracks confidence and provenance
fn measure_biomarker(sample_id: string) -> f64 @ {
    confidence: 0.95,
    source: "lab_assay_v2",
    uncertainty: Normal(0.0, 0.05)
} with IO {
    let raw_value = read_sensor(sample_id)
    raw_value
}

// Uncertainty propagates through calculations
fn analyze_patient(id: string) -> Diagnosis @ epistemic with IO, Prob {
    let biomarker = measure_biomarker(id)
    let history = load_patient_history(id)

    // Confidence degrades appropriately
    let risk_score = calculate_risk(biomarker, history)

    // Epistemic state is tracked
    Diagnosis {
        risk: risk_score,
        recommendation: if risk_score > 0.7 { "follow_up" } else { "monitor" }
    }
}

fn main() -> i32 with IO, Prob {
    let diagnosis = analyze_patient("P001")
    print("Risk score: ", diagnosis.risk)
    print("Confidence: ", diagnosis@confidence)
    0
}
`
};

/**
 * Register Sounio language with Monaco
 */
function registerSounioLanguage(monaco: any) {
    // Register the language
    monaco.languages.register({ id: 'sounio' });

    // Define tokens
    monaco.languages.setMonarchTokensProvider('sounio', {
        keywords: [
            'fn', 'let', 'var', 'if', 'else', 'match', 'for', 'while', 'loop',
            'return', 'break', 'continue', 'struct', 'enum', 'trait', 'impl',
            'type', 'module', 'use', 'pub', 'with', 'where', 'as', 'in',
            'linear', 'affine', 'kernel', 'async', 'await', 'spawn', 'yield',
            'true', 'false', 'self', 'Self', 'super', 'crate'
        ],
        typeKeywords: [
            'i8', 'i16', 'i32', 'i64', 'i128',
            'u8', 'u16', 'u32', 'u64', 'u128',
            'f32', 'f64', 'bool', 'char', 'string',
            'Vec', 'Option', 'Result', 'Box', 'Rc', 'Arc'
        ],
        effects: [
            'IO', 'Mut', 'Alloc', 'Panic', 'Async', 'GPU', 'Prob', 'Div'
        ],
        units: [
            'mg', 'g', 'kg', 'mL', 'L', 'h', 's', 'min', 'mol', 'mmol'
        ],
        operators: [
            '=', '>', '<', '!', '~', '?', ':', '==', '<=', '>=', '!=',
            '&&', '||', '++', '--', '+', '-', '*', '/', '&', '|', '^', '%',
            '<<', '>>', '>>>', '+=', '-=', '*=', '/=', '&=', '|=', '^=',
            '%=', '<<=', '>>=', '>>>=', '->', '=>', '@', '&!'
        ],
        symbols: /[=><!~?:&|+\-*\/\^%@]+/,
        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

        tokenizer: {
            root: [
                // Identifiers and keywords
                [/[a-z_$][\w$]*/, {
                    cases: {
                        '@keywords': 'keyword',
                        '@typeKeywords': 'type',
                        '@effects': 'type.effect',
                        '@units': 'constant.unit',
                        '@default': 'identifier'
                    }
                }],
                [/[A-Z][\w$]*/, 'type.identifier'],

                // Whitespace
                { include: '@whitespace' },

                // Delimiters and operators
                [/[{}()\[\]]/, '@brackets'],
                [/[<>](?!@symbols)/, '@brackets'],
                [/@symbols/, {
                    cases: {
                        '@operators': 'operator',
                        '@default': ''
                    }
                }],

                // Numbers
                [/\d*\.\d+([eE][\-+]?\d+)?/, 'number.float'],
                [/0[xX][0-9a-fA-F]+/, 'number.hex'],
                [/0[bB][01]+/, 'number.binary'],
                [/\d+/, 'number'],

                // Delimiter
                [/[;,.]/, 'delimiter'],

                // Strings
                [/"([^"\\]|\\.)*$/, 'string.invalid'],
                [/"/, { token: 'string.quote', bracket: '@open', next: '@string' }],

                // Characters
                [/'[^\\']'/, 'string'],
                [/(')(@escapes)(')/, ['string', 'string.escape', 'string']],
                [/'/, 'string.invalid']
            ],

            comment: [
                [/[^\/*]+/, 'comment'],
                [/\/\*/, 'comment', '@push'],
                ["\\*/", 'comment', '@pop'],
                [/[\/*]/, 'comment']
            ],

            string: [
                [/[^\\"]+/, 'string'],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/"/, { token: 'string.quote', bracket: '@close', next: '@pop' }]
            ],

            whitespace: [
                [/[ \t\r\n]+/, 'white'],
                [/\/\*/, 'comment', '@comment'],
                [/\/\/.*$/, 'comment'],
            ],
        },
    });

    // Define theme colors for Sounio
    monaco.editor.defineTheme('sounio-dark', {
        base: 'vs-dark',
        inherit: true,
        rules: [
            { token: 'type.effect', foreground: '4EC9B0', fontStyle: 'italic' },
            { token: 'constant.unit', foreground: 'CE9178' },
            { token: 'keyword', foreground: '569CD6' },
            { token: 'type', foreground: '4EC9B0' },
            { token: 'type.identifier', foreground: '4EC9B0' },
            { token: 'number', foreground: 'B5CEA8' },
            { token: 'number.float', foreground: 'B5CEA8' },
            { token: 'string', foreground: 'CE9178' },
            { token: 'comment', foreground: '6A9955' },
            { token: 'operator', foreground: 'D4D4D4' },
        ],
        colors: {
            'editor.background': '#1e1e1e',
        }
    });

    monaco.editor.defineTheme('sounio-light', {
        base: 'vs',
        inherit: true,
        rules: [
            { token: 'type.effect', foreground: '267F99', fontStyle: 'italic' },
            { token: 'constant.unit', foreground: 'A31515' },
            { token: 'keyword', foreground: '0000FF' },
            { token: 'type', foreground: '267F99' },
            { token: 'type.identifier', foreground: '267F99' },
            { token: 'number', foreground: '098658' },
            { token: 'number.float', foreground: '098658' },
            { token: 'string', foreground: 'A31515' },
            { token: 'comment', foreground: '008000' },
            { token: 'operator', foreground: '000000' },
        ],
        colors: {
            'editor.background': '#ffffff',
        }
    });

    // Auto-completion
    monaco.languages.registerCompletionItemProvider('sounio', {
        provideCompletionItems: (model: any, position: any) => {
            const suggestions = [
                // Keywords
                ...['fn', 'let', 'var', 'if', 'else', 'match', 'for', 'while', 'loop',
                    'return', 'struct', 'enum', 'trait', 'impl', 'type', 'module', 'use',
                    'pub', 'with', 'linear', 'affine', 'kernel', 'async', 'await'].map(kw => ({
                    label: kw,
                    kind: monaco.languages.CompletionItemKind.Keyword,
                    insertText: kw,
                })),
                // Types
                ...['i32', 'i64', 'f32', 'f64', 'bool', 'string', 'Vec', 'Option', 'Result'].map(t => ({
                    label: t,
                    kind: monaco.languages.CompletionItemKind.Class,
                    insertText: t,
                })),
                // Effects
                ...['IO', 'Mut', 'Alloc', 'Panic', 'Async', 'GPU', 'Prob', 'Div'].map(e => ({
                    label: e,
                    kind: monaco.languages.CompletionItemKind.Interface,
                    insertText: e,
                })),
                // Snippets
                {
                    label: 'fn',
                    kind: monaco.languages.CompletionItemKind.Snippet,
                    insertText: 'fn ${1:name}(${2:params}) -> ${3:ReturnType} {\n\t$0\n}',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    documentation: 'Function definition'
                },
                {
                    label: 'struct',
                    kind: monaco.languages.CompletionItemKind.Snippet,
                    insertText: 'struct ${1:Name} {\n\t${2:field}: ${3:Type}\n}',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    documentation: 'Struct definition'
                },
                {
                    label: 'linear struct',
                    kind: monaco.languages.CompletionItemKind.Snippet,
                    insertText: 'linear struct ${1:Name} {\n\t${2:field}: ${3:Type}\n}',
                    insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                    documentation: 'Linear struct definition'
                },
            ];
            return { suggestions };
        }
    });
}

/**
 * Initialize the Monaco editor
 */
async function initEditor(): Promise<void> {
    return new Promise((resolve) => {
        require.config({
            paths: {
                'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs'
            }
        });

        require(['vs/editor/editor.main'], (monaco: any) => {
            registerSounioLanguage(monaco);

            const container = document.getElementById('editor-container');
            if (!container) {
                console.error('Editor container not found');
                return;
            }

            const isDark = !document.documentElement.classList.contains('light');

            editor = monaco.editor.create(container, {
                value: EXAMPLES.hello,
                language: 'sounio',
                theme: isDark ? 'sounio-dark' : 'sounio-light',
                automaticLayout: true,
                minimap: { enabled: false },
                fontSize: 14,
                fontFamily: "'Fira Code', 'Consolas', 'Monaco', monospace",
                fontLigatures: true,
                lineNumbers: 'on',
                renderLineHighlight: 'line',
                scrollBeyondLastLine: false,
                padding: { top: 12, bottom: 12 },
                tabSize: 4,
                insertSpaces: true,
                wordWrap: 'off',
                folding: true,
                bracketPairColorization: { enabled: true },
            });

            // Update cursor position display
            editor.onDidChangeCursorPosition((e: any) => {
                const pos = e.position;
                const posDisplay = document.getElementById('cursor-position');
                if (posDisplay) {
                    posDisplay.textContent = `Ln ${pos.lineNumber}, Col ${pos.column}`;
                }
            });

            // Keyboard shortcuts
            editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
                runCode();
            });

            editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
                shareCode();
            });

            editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyMod.Shift | monaco.KeyCode.KeyF, () => {
                formatCode();
            });

            resolve();
        });
    });
}

/**
 * Show a toast notification
 */
function showToast(message: string, duration: number = 3000): void {
    const toast = document.getElementById('toast');
    if (!toast) return;

    toast.textContent = message;
    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
    }, duration);
}

/**
 * Update the output panel
 */
function setOutput(content: string, className: string = ''): void {
    const output = document.getElementById('output');
    if (!output) return;

    output.textContent = content;
    output.className = className;
}

/**
 * Append to the output panel
 */
function appendOutput(content: string, className: string = ''): void {
    const output = document.getElementById('output');
    if (!output) return;

    if (className) {
        const span = document.createElement('span');
        span.className = className;
        span.textContent = content;
        output.appendChild(span);
    } else {
        output.textContent += content;
    }
}

/**
 * Format diagnostics for display
 */
function formatDiagnostics(diagnostics: Diagnostic[]): string {
    return diagnostics.map(d => {
        const level = d.severity === 'error' ? 'error' : d.severity === 'warning' ? 'warning' : 'note';
        const location = d.line ? `${d.line}:${d.column}` : '';
        return `${level}${location ? `[${location}]` : ''}: ${d.message}`;
    }).join('\n');
}

/**
 * Run the code in the editor
 */
async function runCode(): Promise<void> {
    if (!editor) return;

    const source = editor.getValue();
    const statusText = document.getElementById('status-text');
    const execTime = document.getElementById('exec-time');
    const runBtn = document.getElementById('run-btn') as HTMLButtonElement;

    if (statusText) statusText.textContent = 'Running...';
    if (runBtn) runBtn.disabled = true;

    setOutput('');

    const startTime = performance.now();

    try {
        if (compiler) {
            const result = await compiler.run(source);
            const endTime = performance.now();
            const elapsed = (endTime - startTime).toFixed(2);

            if (execTime) execTime.textContent = `${elapsed}ms`;

            if (result.diagnostics.length > 0) {
                const hasErrors = result.diagnostics.some(d => d.severity === 'error');
                appendOutput(formatDiagnostics(result.diagnostics), hasErrors ? 'output-error' : 'output-warning');
                appendOutput('\n\n');
            }

            if (result.output) {
                appendOutput(result.output, result.success ? 'output-success' : 'output-error');
            }

            if (statusText) statusText.textContent = result.success ? 'Success' : 'Failed';
        } else {
            // Fallback: simulate execution for demo purposes
            const endTime = performance.now();
            const elapsed = (endTime - startTime).toFixed(2);

            if (execTime) execTime.textContent = `${elapsed}ms`;

            setOutput(`[Demo Mode - WASM not loaded]\n\nYour code:\n${source}\n\nTo run real code, build the WASM module:\n  cd compiler && cargo build --target wasm32-unknown-unknown --features wasm`, 'output-warning');

            if (statusText) statusText.textContent = 'Demo Mode';
        }
    } catch (error) {
        const endTime = performance.now();
        const elapsed = (endTime - startTime).toFixed(2);

        if (execTime) execTime.textContent = `${elapsed}ms`;
        setOutput(`Error: ${error}`, 'output-error');
        if (statusText) statusText.textContent = 'Error';
    } finally {
        if (runBtn) runBtn.disabled = false;
    }
}

/**
 * Share the current code
 */
async function shareCode(): Promise<void> {
    if (!editor) return;

    const source = editor.getValue();
    const url = generateShareUrl(source);

    try {
        await navigator.clipboard.writeText(url);
        showToast('Link copied to clipboard!');
    } catch {
        // Fallback for browsers without clipboard API
        prompt('Copy this link:', url);
    }
}

/**
 * Format the code (placeholder - would need formatter WASM)
 */
function formatCode(): void {
    showToast('Formatting not yet implemented');
}

/**
 * Load an example
 */
function loadExample(name: string): void {
    if (!editor) return;

    const code = EXAMPLES[name];
    if (code) {
        editor.setValue(code);
        setOutput(`Loaded example: ${name}`);
    }
}

/**
 * Toggle theme
 */
function toggleTheme(): void {
    const html = document.documentElement;
    const isDark = html.classList.toggle('light');

    const darkIcon = document.getElementById('theme-icon-dark');
    const lightIcon = document.getElementById('theme-icon-light');

    if (darkIcon && lightIcon) {
        darkIcon.style.display = isDark ? 'none' : 'block';
        lightIcon.style.display = isDark ? 'block' : 'none';
    }

    if (editor) {
        require(['vs/editor/editor.main'], (monaco: any) => {
            monaco.editor.setTheme(isDark ? 'sounio-light' : 'sounio-dark');
        });
    }

    localStorage.setItem('sounio-theme', isDark ? 'light' : 'dark');
}

/**
 * Initialize resizer
 */
function initResizer(): void {
    const resizer = document.getElementById('resizer');
    const editorPanel = document.querySelector('.editor-panel') as HTMLElement;
    const outputPanel = document.querySelector('.output-panel') as HTMLElement;

    if (!resizer || !editorPanel || !outputPanel) return;

    let isResizing = false;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        const container = document.querySelector('.main') as HTMLElement;
        if (!container) return;

        const containerRect = container.getBoundingClientRect();
        const percentage = ((e.clientX - containerRect.left) / containerRect.width) * 100;

        if (percentage > 20 && percentage < 80) {
            editorPanel.style.flex = `0 0 ${percentage}%`;
            outputPanel.style.width = `${100 - percentage}%`;
        }
    });

    document.addEventListener('mouseup', () => {
        isResizing = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    });
}

/**
 * Main initialization
 */
async function main(): Promise<void> {
    // Restore theme preference
    const savedTheme = localStorage.getItem('sounio-theme');
    if (savedTheme === 'light') {
        document.documentElement.classList.add('light');
        const darkIcon = document.getElementById('theme-icon-dark');
        const lightIcon = document.getElementById('theme-icon-light');
        if (darkIcon && lightIcon) {
            darkIcon.style.display = 'none';
            lightIcon.style.display = 'block';
        }
    }

    // Initialize editor
    await initEditor();

    // Initialize compiler
    try {
        compiler = new SounioCompiler();
        await compiler.init();
        const wasmStatus = document.getElementById('wasm-status');
        if (wasmStatus) wasmStatus.textContent = 'WASM: Ready';
    } catch (error) {
        console.warn('WASM compiler not available, running in demo mode:', error);
        const wasmStatus = document.getElementById('wasm-status');
        if (wasmStatus) wasmStatus.textContent = 'WASM: Demo Mode';
    }

    // Initialize resizer
    initResizer();

    // Load code from URL if present
    const urlCode = loadFromUrl();
    if (urlCode && editor) {
        editor.setValue(urlCode);
    }

    // Event listeners
    document.getElementById('run-btn')?.addEventListener('click', runCode);
    document.getElementById('share-btn')?.addEventListener('click', shareCode);
    document.getElementById('format-btn')?.addEventListener('click', formatCode);
    document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);
    document.getElementById('clear-output')?.addEventListener('click', () => setOutput(''));

    // Example buttons
    document.querySelectorAll('[data-example]').forEach((btn) => {
        btn.addEventListener('click', () => {
            const name = (btn as HTMLElement).dataset.example;
            if (name) loadExample(name);
        });
    });

    // Status
    const statusText = document.getElementById('status-text');
    if (statusText) statusText.textContent = 'Ready';
}

// Start the application
main().catch(console.error);

export { editor, runCode, shareCode, loadExample };
