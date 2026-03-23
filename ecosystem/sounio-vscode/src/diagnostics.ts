import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

let diagnosticCollection: vscode.DiagnosticCollection;

export function createDiagnostics(context: vscode.ExtensionContext): vscode.DiagnosticCollection {
    diagnosticCollection = vscode.languages.createDiagnosticCollection('sounio');
    context.subscriptions.push(diagnosticCollection);

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

    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument((doc) => {
            if (doc.languageId === 'sounio') {
                checkDocument(doc);
            }
        })
    );

    context.subscriptions.push(
        vscode.workspace.onDidCloseTextDocument((doc) => {
            diagnosticCollection.delete(doc.uri);
        })
    );

    // Check already-open .sio files
    vscode.workspace.textDocuments.forEach((doc) => {
        if (doc.languageId === 'sounio') {
            checkDocument(doc);
        }
    });

    return diagnosticCollection;
}

export function checkDocument(doc: vscode.TextDocument): void {
    const soucPath = resolveSoucPath();
    if (!soucPath) { return; }

    const filePath = doc.uri.fsPath;
    const cfg = vscode.workspace.getConfiguration('sounio');
    const stdlibPath = cfg.get<string>('stdlibPath', '').trim();

    const env: NodeJS.ProcessEnv = { ...process.env };
    if (stdlibPath) {
        env['SOUNIO_STDLIB_PATH'] = stdlibPath;
    }

    cp.execFile(
        soucPath,
        ['check', filePath],
        { env, maxBuffer: 2 * 1024 * 1024 },
        (_error, _stdout, stderr) => {
            const diagnostics = parseSoucOutput(stderr, doc);
            diagnosticCollection.set(doc.uri, diagnostics);
        }
    );
}

export function resolveSoucPath(): string | null {
    const cfg = vscode.workspace.getConfiguration('sounio');
    const configured = cfg.get<string>('soucPath', '').trim();
    if (configured) { return configured; }

    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders) {
        for (const folder of workspaceFolders) {
            const candidates = [
                path.join(folder.uri.fsPath, 'artifacts', 'omega', 'souc-bin', 'souc-linux-x86_64-jit'),
                path.join(folder.uri.fsPath, 'souc'),
                path.join(folder.uri.fsPath, 'bin', 'souc'),
            ];
            for (const c of candidates) {
                if (fs.existsSync(c)) { return c; }
            }
        }
    }

    return 'souc';
}

function parseSoucOutput(stderr: string, doc: vscode.TextDocument): vscode.Diagnostic[] {
    const diagnostics: vscode.Diagnostic[] = [];
    const lines = stderr.split('\n');

    const pattern1Header = /^(error|warning|note)\[?[A-Z0-9]*\]?:\s+(.+)$/;
    const pattern1Location = /^\s+-->\s+.+:(\d+):(\d+)/;
    const pattern2 = /^.+:(\d+):(\d+):\s+(error|warning|note):\s+(.+)$/;
    const pattern3 = /^(error|warning):\s+(.+)\s+at\s+(\d+):(\d+)$/;

    let pendingMessage: string | null = null;
    let pendingSeverity = vscode.DiagnosticSeverity.Error;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        const h = pattern1Header.exec(line);
        if (h) {
            pendingMessage = h[2].trim();
            pendingSeverity = severityFromWord(h[1]);
            continue;
        }

        if (pendingMessage !== null) {
            const loc = pattern1Location.exec(line);
            if (loc) {
                diagnostics.push(makeDiag(doc,
                    parseInt(loc[1], 10) - 1,
                    parseInt(loc[2], 10) - 1,
                    pendingMessage, pendingSeverity));
                pendingMessage = null;
                continue;
            }
            if (!line.startsWith(' ') && !line.startsWith('\t')) {
                pendingMessage = null;
            }
        }

        const m2 = pattern2.exec(line);
        if (m2) {
            diagnostics.push(makeDiag(doc,
                parseInt(m2[1], 10) - 1,
                parseInt(m2[2], 10) - 1,
                m2[4].trim(), severityFromWord(m2[3])));
            continue;
        }

        const m3 = pattern3.exec(line);
        if (m3) {
            diagnostics.push(makeDiag(doc,
                parseInt(m3[3], 10) - 1,
                parseInt(m3[4], 10) - 1,
                m3[2].trim(), severityFromWord(m3[1])));
        }
    }

    return diagnostics;
}

function severityFromWord(word: string): vscode.DiagnosticSeverity {
    switch (word.toLowerCase()) {
        case 'warning': return vscode.DiagnosticSeverity.Warning;
        case 'note': return vscode.DiagnosticSeverity.Information;
        default: return vscode.DiagnosticSeverity.Error;
    }
}

function makeDiag(
    doc: vscode.TextDocument,
    lineNum: number, colNum: number,
    message: string, severity: vscode.DiagnosticSeverity
): vscode.Diagnostic {
    const safeLine = Math.max(0, Math.min(lineNum, doc.lineCount - 1));
    const docLine = doc.lineAt(safeLine);
    const safeCol = Math.max(0, Math.min(colNum, docLine.text.length));
    const range = new vscode.Range(safeLine, safeCol, safeLine, docLine.text.length);
    const diag = new vscode.Diagnostic(range, message, severity);
    diag.source = 'souc';
    return diag;
}
