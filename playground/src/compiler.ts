/**
 * Sounio Compiler - WASM Interface
 *
 * This module handles loading and initializing the Sounio compiler WASM module,
 * and provides TypeScript interfaces for compilation and execution.
 */

/**
 * Diagnostic message from the compiler
 */
export interface Diagnostic {
    severity: 'error' | 'warning' | 'note';
    message: string;
    line?: number;
    column?: number;
    span?: {
        start: number;
        end: number;
    };
    code?: string;
    hint?: string;
}

/**
 * Result of compiling source code
 */
export interface CompileResult {
    success: boolean;
    wasm?: Uint8Array;
    diagnostics: Diagnostic[];
}

/**
 * Result of running source code
 */
export interface RunResult {
    success: boolean;
    output: string;
    diagnostics: Diagnostic[];
    returnValue?: any;
}

/**
 * WASM module exports interface
 */
interface SounioWasmExports {
    compile: (source: string) => string;
    run: (source: string) => string;
    format: (source: string) => string;
    version: () => string;
}

/**
 * Sounio Compiler class
 *
 * Handles loading the WASM module and provides methods for compilation and execution.
 */
export class SounioCompiler {
    private wasmModule: SounioWasmExports | null = null;
    private initialized: boolean = false;
    private initPromise: Promise<void> | null = null;

    /**
     * WASM module URL - can be overridden for different deployments
     */
    private wasmUrl: string = './sounio_compiler.wasm';

    constructor(wasmUrl?: string) {
        if (wasmUrl) {
            this.wasmUrl = wasmUrl;
        }
    }

    /**
     * Initialize the WASM module
     *
     * This method is idempotent - calling it multiple times will only
     * initialize once and return the same promise.
     */
    async init(): Promise<void> {
        if (this.initialized) {
            return;
        }

        if (this.initPromise) {
            return this.initPromise;
        }

        this.initPromise = this._doInit();
        return this.initPromise;
    }

    private async _doInit(): Promise<void> {
        try {
            // Try to load the wasm-bindgen generated module
            const wasmModule = await this.loadWasmBindgenModule();
            if (wasmModule) {
                this.wasmModule = wasmModule;
                this.initialized = true;
                return;
            }
        } catch (error) {
            console.warn('wasm-bindgen module not found, trying raw WASM:', error);
        }

        try {
            // Fall back to raw WASM loading
            await this.loadRawWasm();
            this.initialized = true;
        } catch (error) {
            console.error('Failed to load WASM module:', error);
            throw new Error(`Failed to initialize Sounio compiler: ${error}`);
        }
    }

    /**
     * Load the wasm-bindgen generated JavaScript module
     */
    private async loadWasmBindgenModule(): Promise<SounioWasmExports | null> {
        try {
            // Dynamic import of the generated JS glue code
            const module = await import('./sounio_compiler.js');
            await module.default();
            return {
                compile: module.compile,
                run: module.run,
                format: module.format || ((s: string) => s),
                version: module.version || (() => 'unknown'),
            };
        } catch {
            return null;
        }
    }

    /**
     * Load raw WASM module (without wasm-bindgen)
     */
    private async loadRawWasm(): Promise<void> {
        const response = await fetch(this.wasmUrl);
        if (!response.ok) {
            throw new Error(`Failed to fetch WASM: ${response.status} ${response.statusText}`);
        }

        const wasmBytes = await response.arrayBuffer();
        const wasmModule = await WebAssembly.instantiate(wasmBytes, {
            env: {
                // Provide any required imports here
                print: (ptr: number, len: number) => {
                    // Handle print calls from WASM
                    console.log('WASM print:', ptr, len);
                },
            },
        });

        // Extract exports
        const exports = wasmModule.instance.exports as any;

        this.wasmModule = {
            compile: (source: string) => {
                // Call the raw WASM compile function
                // This would need memory management for strings
                return JSON.stringify({ success: false, diagnostics: [] });
            },
            run: (source: string) => {
                return JSON.stringify({ success: false, output: '', diagnostics: [] });
            },
            format: (source: string) => source,
            version: () => exports.version?.() || 'unknown',
        };
    }

    /**
     * Check if the compiler is initialized
     */
    isReady(): boolean {
        return this.initialized && this.wasmModule !== null;
    }

    /**
     * Get the compiler version
     */
    version(): string {
        if (!this.wasmModule) {
            return 'not initialized';
        }
        return this.wasmModule.version();
    }

    /**
     * Compile source code to WASM
     */
    async compile(source: string): Promise<CompileResult> {
        await this.init();

        if (!this.wasmModule) {
            return {
                success: false,
                diagnostics: [{
                    severity: 'error',
                    message: 'Compiler not initialized',
                }],
            };
        }

        try {
            const resultJson = this.wasmModule.compile(source);
            const result = JSON.parse(resultJson);
            return {
                success: result.success ?? false,
                wasm: result.wasm ? new Uint8Array(result.wasm) : undefined,
                diagnostics: result.diagnostics ?? [],
            };
        } catch (error) {
            return {
                success: false,
                diagnostics: [{
                    severity: 'error',
                    message: `Compilation failed: ${error}`,
                }],
            };
        }
    }

    /**
     * Run source code and return the output
     */
    async run(source: string): Promise<RunResult> {
        await this.init();

        if (!this.wasmModule) {
            return {
                success: false,
                output: '',
                diagnostics: [{
                    severity: 'error',
                    message: 'Compiler not initialized',
                }],
            };
        }

        try {
            const resultJson = this.wasmModule.run(source);
            const result = JSON.parse(resultJson);
            return {
                success: result.success ?? false,
                output: result.output ?? '',
                diagnostics: result.diagnostics ?? [],
                returnValue: result.returnValue,
            };
        } catch (error) {
            return {
                success: false,
                output: '',
                diagnostics: [{
                    severity: 'error',
                    message: `Execution failed: ${error}`,
                }],
            };
        }
    }

    /**
     * Format source code
     */
    async format(source: string): Promise<string> {
        await this.init();

        if (!this.wasmModule) {
            return source;
        }

        try {
            return this.wasmModule.format(source);
        } catch {
            return source;
        }
    }
}

/**
 * Create a mock compiler for demo mode when WASM is not available
 */
export function createMockCompiler(): SounioCompiler {
    const mock = new SounioCompiler();

    // Override init to always succeed
    mock.init = async () => {};

    // Override run to provide demo output
    (mock as any).run = async (source: string): Promise<RunResult> => {
        // Simple pattern matching for demo output
        const lines = source.split('\n');
        const output: string[] = [];
        const diagnostics: Diagnostic[] = [];

        // Check for basic syntax issues
        let braceCount = 0;
        let parenCount = 0;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            braceCount += (line.match(/{/g) || []).length;
            braceCount -= (line.match(/}/g) || []).length;
            parenCount += (line.match(/\(/g) || []).length;
            parenCount -= (line.match(/\)/g) || []).length;

            // Check for print statements
            const printMatch = line.match(/print\s*\(\s*"([^"]*)"/);
            if (printMatch) {
                output.push(printMatch[1]);
            }

            // Check for print with variable
            const printVarMatch = line.match(/print\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)/);
            if (printVarMatch) {
                output.push(`<${printVarMatch[1]}>`);
            }
        }

        if (braceCount !== 0) {
            diagnostics.push({
                severity: 'error',
                message: braceCount > 0 ? 'Unclosed brace' : 'Unexpected closing brace',
            });
        }

        if (parenCount !== 0) {
            diagnostics.push({
                severity: 'error',
                message: parenCount > 0 ? 'Unclosed parenthesis' : 'Unexpected closing parenthesis',
            });
        }

        const hasErrors = diagnostics.some(d => d.severity === 'error');

        if (!hasErrors) {
            output.push('\n[Program executed successfully in demo mode]');
        }

        return {
            success: !hasErrors,
            output: output.join('\n'),
            diagnostics,
        };
    };

    return mock;
}

export default SounioCompiler;
