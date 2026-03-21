import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

// ============================================================================
// Diagnostic collection — keyed per document URI
// ============================================================================

let diagnosticCollection: vscode.DiagnosticCollection;

// ============================================================================
// Activation
// ============================================================================

export function activate(context: vscode.ExtensionContext): void {
    diagnosticCollection = vscode.languages.createDiagnosticCollection('sounio');
    context.subscriptions.push(diagnosticCollection);

    // Check on save
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument((doc) => {
            if (doc.languageId === 'sounio') {
                const cfg = vscode.workspace.getConfiguration('sounio');
                if (cfg.get<boolean>('checkOnSave', true)) {
                    checkDocument(doc);
                }
            }
        })
    );

    // Check on open
    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument((doc) => {
            if (doc.languageId === 'sounio') {
                checkDocument(doc);
            }
        })
    );

    // Clear diagnostics when document closed
    context.subscriptions.push(
        vscode.workspace.onDidCloseTextDocument((doc) => {
            diagnosticCollection.delete(doc.uri);
        })
    );

    // Check already-open .sio files at startup
    vscode.workspace.textDocuments.forEach((doc) => {
        if (doc.languageId === 'sounio') {
            checkDocument(doc);
        }
    });

    // Register manual check command
    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.check', () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                checkDocument(editor.document);
            }
        })
    );
}

// ============================================================================
// Deactivation
// ============================================================================

export function deactivate(): void {
    if (diagnosticCollection) {
        diagnosticCollection.dispose();
    }
}

// ============================================================================
// souc binary resolution
// ============================================================================

function resolveSoucPath(): string | null {
    const cfg = vscode.workspace.getConfiguration('sounio');
    const configured = cfg.get<string>('soucPath', '').trim();
    if (configured) {
        return configured;
    }

    // Try workspace-relative common locations
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders) {
        for (const folder of workspaceFolders) {
            const candidates = [
                path.join(folder.uri.fsPath, 'artifacts', 'omega', 'souc-bin', 'souc-linux-x86_64-jit'),
                path.join(folder.uri.fsPath, 'souc'),
                path.join(folder.uri.fsPath, 'bin', 'souc'),
            ];
            for (const candidate of candidates) {
                if (fs.existsSync(candidate)) {
                    return candidate;
                }
            }
        }
    }

    // Fall back to PATH
    return 'souc';
}

// ============================================================================
// Run souc check and parse diagnostics
// ============================================================================

function checkDocument(doc: vscode.TextDocument): void {
    const soucPath = resolveSoucPath();
    if (!soucPath) {
        return;
    }

    const filePath = doc.uri.fsPath;
    const cfg = vscode.workspace.getConfiguration('sounio');
    const stdlibPath = cfg.get<string>('stdlibPath', '').trim();

    const env: NodeJS.ProcessEnv = { ...process.env };
    if (stdlibPath) {
        env['SOUNIO_STDLIB_PATH'] = stdlibPath;
    }

    const args = ['check', filePath];

    cp.execFile(
        soucPath,
        args,
        { env, maxBuffer: 2 * 1024 * 1024 },
        (error, _stdout, stderr) => {
            // souc exits non-zero on errors; that is expected
            const diagnostics = parseSoucOutput(stderr, doc);
            diagnosticCollection.set(doc.uri, diagnostics);
        }
    );
}

// ============================================================================
// Parse souc stderr into VSCode Diagnostics
//
// souc error format (observed from compiler output):
//   error[E001]: message
//     --> file.sio:LINE:COL
//
//   error: message at LINE:COL
//
//   file.sio:LINE:COL: error: message
//
// We handle all three patterns.
// ============================================================================

interface RawDiagnostic {
    line: number;      // 0-based
    col: number;       // 0-based
    message: string;
    severity: vscode.DiagnosticSeverity;
}

function parseSoucOutput(
    stderr: string,
    doc: vscode.TextDocument
): vscode.Diagnostic[] {
    const diagnostics: vscode.Diagnostic[] = [];
    const lines = stderr.split('\n');

    // Pattern 1: error[Exxx]: message  followed by   --> file:line:col
    // Pattern 2: file.sio:line:col: error: message
    // Pattern 3: error: message at line:col

    const pattern1Header = /^(error|warning|note)\[?[A-Z0-9]*\]?:\s+(.+)$/;
    const pattern1Location = /^\s+-->\s+.+:(\d+):(\d+)/;
    const pattern2 = /^.+:(\d+):(\d+):\s+(error|warning|note):\s+(.+)$/;
    const pattern3 = /^(error|warning):\s+(.+)\s+at\s+(\d+):(\d+)$/;

    let pendingMessage: string | null = null;
    let pendingSeverity = vscode.DiagnosticSeverity.Error;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        // Pattern 1 header
        const h = pattern1Header.exec(line);
        if (h) {
            pendingMessage = h[2].trim();
            pendingSeverity = severityFromWord(h[1]);
            continue;
        }

        // Pattern 1 location (follows header)
        if (pendingMessage !== null) {
            const loc = pattern1Location.exec(line);
            if (loc) {
                const lineNum = Math.max(0, parseInt(loc[1], 10) - 1);
                const colNum = Math.max(0, parseInt(loc[2], 10) - 1);
                diagnostics.push(makeDiagnostic(doc, lineNum, colNum, pendingMessage, pendingSeverity));
                pendingMessage = null;
                continue;
            }
            // If next line is not a location, discard pending
            if (!line.startsWith(' ') && !line.startsWith('\t')) {
                pendingMessage = null;
            }
        }

        // Pattern 2: file:line:col: severity: message
        const m2 = pattern2.exec(line);
        if (m2) {
            const lineNum = Math.max(0, parseInt(m2[1], 10) - 1);
            const colNum = Math.max(0, parseInt(m2[2], 10) - 1);
            const sev = severityFromWord(m2[3]);
            diagnostics.push(makeDiagnostic(doc, lineNum, colNum, m2[4].trim(), sev));
            continue;
        }

        // Pattern 3: error: message at line:col
        const m3 = pattern3.exec(line);
        if (m3) {
            const sev = severityFromWord(m3[1]);
            const msg = m3[2].trim();
            const lineNum = Math.max(0, parseInt(m3[3], 10) - 1);
            const colNum = Math.max(0, parseInt(m3[4], 10) - 1);
            diagnostics.push(makeDiagnostic(doc, lineNum, colNum, msg, sev));
            continue;
        }
    }

    return diagnostics;
}

function severityFromWord(word: string): vscode.DiagnosticSeverity {
    switch (word.toLowerCase()) {
        case 'warning': return vscode.DiagnosticSeverity.Warning;
        case 'note':    return vscode.DiagnosticSeverity.Information;
        default:        return vscode.DiagnosticSeverity.Error;
    }
}

function makeDiagnostic(
    doc: vscode.TextDocument,
    lineNum: number,
    colNum: number,
    message: string,
    severity: vscode.DiagnosticSeverity
): vscode.Diagnostic {
    const safeLineNum = Math.min(lineNum, doc.lineCount - 1);
    const docLine = doc.lineAt(safeLineNum);
    const safeColNum = Math.min(colNum, docLine.text.length);

    const range = new vscode.Range(
        new vscode.Position(safeLineNum, safeColNum),
        new vscode.Position(safeLineNum, docLine.text.length)
    );

    const diag = new vscode.Diagnostic(range, message, severity);
    diag.source = 'souc';
    return diag;
}
