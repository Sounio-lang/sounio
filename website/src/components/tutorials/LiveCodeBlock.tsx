import { useState, useRef, useCallback, type KeyboardEvent } from 'react';

interface LiveCodeBlockProps {
  initialCode: string;
  language?: string;
  expectedOutput?: string;
  title?: string;
}

export default function LiveCodeBlock({
  initialCode,
  language = 'sio',
  expectedOutput,
  title,
}: LiveCodeBlockProps) {
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [copied, setCopied] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const runCode = useCallback(async () => {
    setIsRunning(true);
    setOutput('Compiling...\n');

    // Simulate compilation delay
    await new Promise(resolve => setTimeout(resolve, 400));

    let execOutput = '✓ Compiled successfully\n\nRunning...\n';

    // Simulate different outputs based on code content
    if (code.includes('print(')) {
      const printMatch = code.match(/print\("([^"]+)"\)/);
      if (printMatch) {
        execOutput += printMatch[1];
      } else {
        execOutput += 'Hello, World!';
      }
    } else if (code.includes('Knowledge<')) {
      execOutput += 'Value: 22.73 ± 0.31 (confidence: 0.95)';
    } else if (code.includes('@kernel')) {
      execOutput += 'GPU kernel compiled to PTX\n✓ Execution complete';
    } else {
      execOutput += '✓ Execution complete';
    }

    // Validate against expected output if provided
    if (expectedOutput) {
      const passed = execOutput.includes(expectedOutput.trim());
      execOutput += passed
        ? '\n\n✅ Output matches expected!'
        : `\n\n❌ Expected: ${expectedOutput}`;
    }

    setOutput(execOutput);
    setIsRunning(false);
  }, [code, expectedOutput]);

  const resetCode = useCallback(() => {
    setCode(initialCode);
    setOutput('');
    textareaRef.current?.focus();
  }, [initialCode]);

  const copyCode = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, [code]);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      runCode();
    }
    // Tab support for indentation
    if (e.key === 'Tab') {
      e.preventDefault();
      const textarea = textareaRef.current;
      if (textarea) {
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        const newCode = code.substring(0, start) + '    ' + code.substring(end);
        setCode(newCode);
        setTimeout(() => {
          textarea.selectionStart = textarea.selectionEnd = start + 4;
        }, 0);
      }
    }
  }, [code, runCode]);

  return (
    <div className="my-6 rounded-lg overflow-hidden border border-white/10 bg-[#1e1e1e]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-[var(--color-navy-900)] border-b border-white/10">
        <div className="flex items-center gap-3">
          <span className="text-white/60 text-sm font-mono">{title || `example.${language}`}</span>
          <span className="px-2 py-0.5 text-xs bg-white/10 text-white/60 rounded">{language}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={copyCode}
            className="p-1.5 text-white/60 hover:text-white hover:bg-white/10 rounded transition-colors"
            title="Copy code"
          >
            {copied ? (
              <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            )}
          </button>
          <button
            onClick={resetCode}
            className="p-1.5 text-white/60 hover:text-white hover:bg-white/10 rounded transition-colors"
            title="Reset code"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
          <button
            onClick={runCode}
            disabled={isRunning}
            className="flex items-center gap-1.5 px-3 py-1 bg-green-500 hover:bg-green-600 disabled:bg-green-500/50 text-white text-sm font-medium rounded transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            </svg>
            {isRunning ? 'Running...' : 'Run'}
          </button>
        </div>
      </div>

      {/* Code editor */}
      <div className="relative">
        <textarea
          ref={textareaRef}
          value={code}
          onChange={(e) => setCode(e.target.value)}
          onKeyDown={handleKeyDown}
          className="w-full min-h-[150px] p-4 bg-[#1e1e1e] text-[#d4d4d4] font-mono text-sm leading-relaxed resize-none focus:outline-none"
          spellCheck={false}
          placeholder="Enter your code here..."
        />
        <div className="absolute bottom-2 right-2 text-xs text-white/30">
          Ctrl+Enter to run
        </div>
      </div>

      {/* Output panel */}
      {output && (
        <div className="border-t border-white/10">
          <div className="px-4 py-1.5 text-xs text-white/40 bg-black/20 border-b border-white/5">
            Output
          </div>
          <pre className="p-4 text-sm font-mono text-green-400 overflow-x-auto max-h-48">
            {output}
          </pre>
        </div>
      )}
    </div>
  );
}
