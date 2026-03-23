import * as vscode from 'vscode';

interface RustismPattern {
    regex: RegExp;
    message: string;
    fix?: { search: RegExp; replace: string };
}

const RUSTISM_PATTERNS: RustismPattern[] = [
    {
        regex: /;\s*$/,
        message: 'Sounio does not use semicolons. Remove the trailing semicolon.',
        fix: { search: /;\s*$/, replace: '' }
    },
    {
        regex: /&mut\b/,
        message: 'Use `&!` for exclusive (mutable) references. `&mut` is Rust syntax.',
        fix: { search: /&mut\b/, replace: '&!' }
    },
    {
        regex: /\blet\s+mut\b/,
        message: 'Use `var` for mutable bindings. `let mut` is Rust syntax.',
        fix: { search: /\blet\s+mut\b/, replace: 'var' }
    },
    {
        regex: /\bassert!\s*\(/,
        message: 'Use `assert()` — Sounio has no macros. Remove the `!`.',
        fix: { search: /\bassert!\s*\(/, replace: 'assert(' }
    },
    {
        regex: /\bprintln!\s*\(/,
        message: 'Use `println()` — Sounio has no macros. Remove the `!`.',
        fix: { search: /\bprintln!\s*\(/, replace: 'println(' }
    },
    {
        regex: /\bprint!\s*\(/,
        message: 'Use `print()` — Sounio has no macros. Remove the `!`.',
        fix: { search: /\bprint!\s*\(/, replace: 'print(' }
    },
    {
        regex: /\bvec!\s*\[/,
        message: 'Use array literals `[...]` or `Vec<T>`. `vec!` is a Rust macro.',
    },
    {
        regex: /^#\[/,
        message: 'Sounio has no attributes. Remove `#[...]`.',
    },
    {
        regex: /->\s*impl\b/,
        message: 'No `impl Trait` in return position. Use concrete types or function references.',
    },
    {
        regex: /\bformat!\s*\(/,
        message: 'Sounio has no macros. Use `print()` / `println()` for output.',
    },
    {
        regex: /\beprintln!\s*\(/,
        message: 'Sounio has no macros. Use effect-based error reporting.',
    },
    {
        regex: /\bpanic!\s*\(/,
        message: 'Sounio has no macros. Use `assert()` or return error codes.',
    },
];

let rustismDiagnostics: vscode.DiagnosticCollection;
let debounceTimer: ReturnType<typeof setTimeout> | undefined;
let codeActionProvider: vscode.Disposable | undefined;

// Store fixes for quick-fix code actions
const pendingFixes = new Map<string, Map<string, { search: RegExp; replace: string }>>();

export function createRustismDetector(context: vscode.ExtensionContext): vscode.DiagnosticCollection {
    rustismDiagnostics = vscode.languages.createDiagnosticCollection('sounio-rustism');
    context.subscriptions.push(rustismDiagnostics);

    // Detect on change (debounced)
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument((e) => {
            if (e.document.languageId === 'sounio') {
                scheduleDetection(e.document);
            }
        })
    );

    // Detect on open
    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument((doc) => {
            if (doc.languageId === 'sounio') {
                detectRustisms(doc);
            }
        })
    );

    // Clear on close
    context.subscriptions.push(
        vscode.workspace.onDidCloseTextDocument((doc) => {
            rustismDiagnostics.delete(doc.uri);
            pendingFixes.delete(doc.uri.toString());
        })
    );

    // Register code action provider for quick fixes
    codeActionProvider = vscode.languages.registerCodeActionsProvider('sounio', {
        provideCodeActions(document, range, context): vscode.CodeAction[] {
            const actions: vscode.CodeAction[] = [];
            const uriStr = document.uri.toString();
            const fixes = pendingFixes.get(uriStr);
            if (!fixes) { return actions; }

            for (const diag of context.diagnostics) {
                if (diag.source !== 'sounio-rustism') { continue; }
                const key = `${diag.range.start.line}:${diag.range.start.character}`;
                const fix = fixes.get(key);
                if (!fix) { continue; }

                const lineText = document.lineAt(diag.range.start.line).text;
                const fixed = lineText.replace(fix.search, fix.replace);

                const action = new vscode.CodeAction(
                    `Fix: ${diag.message.split('.')[0]}`,
                    vscode.CodeActionKind.QuickFix
                );
                action.edit = new vscode.WorkspaceEdit();
                const fullLineRange = document.lineAt(diag.range.start.line).range;
                action.edit.replace(document.uri, fullLineRange, fixed);
                action.diagnostics = [diag];
                action.isPreferred = true;
                actions.push(action);
            }
            return actions;
        }
    }, { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] });
    context.subscriptions.push(codeActionProvider);

    // Detect in already-open files
    vscode.workspace.textDocuments.forEach((doc) => {
        if (doc.languageId === 'sounio') {
            detectRustisms(doc);
        }
    });

    return rustismDiagnostics;
}

function scheduleDetection(doc: vscode.TextDocument): void {
    if (debounceTimer) { clearTimeout(debounceTimer); }
    debounceTimer = setTimeout(() => detectRustisms(doc), 500);
}

function detectRustisms(doc: vscode.TextDocument): void {
    const cfg = vscode.workspace.getConfiguration('sounio');
    if (!cfg.get<boolean>('rustismDetector.enabled', true)) {
        rustismDiagnostics.delete(doc.uri);
        return;
    }

    const diagnostics: vscode.Diagnostic[] = [];
    const fixes = new Map<string, { search: RegExp; replace: string }>();
    const text = doc.getText();
    const lines = text.split('\n');

    for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
        const line = lines[lineIdx];

        // Skip comment lines
        const trimmed = line.trimStart();
        if (trimmed.startsWith('//')) { continue; }

        // Skip lines inside strings (simple heuristic: odd number of unescaped quotes before match)
        for (const pattern of RUSTISM_PATTERNS) {
            const match = pattern.regex.exec(line);
            if (!match) { continue; }

            // Don't flag semicolons inside string literals
            if (pattern.regex.source === ';\\s*$') {
                const beforeSemicolon = line.substring(0, match.index);
                const quoteCount = (beforeSemicolon.match(/(?<!\\)"/g) || []).length;
                if (quoteCount % 2 !== 0) { continue; }
            }

            const col = match.index;
            const range = new vscode.Range(lineIdx, col, lineIdx, col + match[0].length);
            const diag = new vscode.Diagnostic(range, pattern.message, vscode.DiagnosticSeverity.Warning);
            diag.source = 'sounio-rustism';
            diagnostics.push(diag);

            if (pattern.fix) {
                fixes.set(`${lineIdx}:${col}`, pattern.fix);
            }
        }
    }

    rustismDiagnostics.set(doc.uri, diagnostics);
    pendingFixes.set(doc.uri.toString(), fixes);
}
