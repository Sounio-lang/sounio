import * as vscode from 'vscode';
import { LanguageClient } from 'vscode-languageclient/node';

let statusBarItem: vscode.StatusBarItem;

const highConfidenceDecoration = vscode.window.createTextEditorDecorationType({
    after: { contentText: ' \u{1f7e2}', margin: '0 0 0 4px' }
});
const mediumConfidenceDecoration = vscode.window.createTextEditorDecorationType({
    after: { contentText: ' \u{1f7e1}', margin: '0 0 0 4px' }
});
const lowConfidenceDecoration = vscode.window.createTextEditorDecorationType({
    after: { contentText: ' \u{1f534}', margin: '0 0 0 4px' }
});

export function createEpistemicUI(context: vscode.ExtensionContext, client: LanguageClient | null): void {
    // Status bar
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'sounio.toggleEpistemic';
    updateStatusBar();
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Toggle command
    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.toggleEpistemic', async () => {
            const config = vscode.workspace.getConfiguration('sounio');
            const current = config.get<boolean>('epistemic.enabled', true);
            await config.update('epistemic.enabled', !current, vscode.ConfigurationTarget.Workspace);
            updateStatusBar();
            vscode.window.showInformationMessage(`Epistemic mode: ${!current ? 'ON' : 'OFF'}`);
        })
    );

    // Confidence command
    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showConfidence', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'sounio') { return; }
            if (client) {
                try {
                    const pos = editor.selection.active;
                    const result = await client.sendRequest<any>('sounio/confidence', {
                        textDocument: { uri: editor.document.uri.toString() },
                        position: { line: pos.line, character: pos.character }
                    });
                    if (result) { showConfidencePanel(result); }
                } catch { /* LSP may not support this */ }
            } else {
                vscode.window.showInformationMessage('Confidence info requires the Sounio language server.');
            }
        })
    );

    // Provenance command
    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showProvenance', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'sounio') { return; }
            if (client) {
                try {
                    const pos = editor.selection.active;
                    const result = await client.sendRequest<any>('sounio/provenance', {
                        textDocument: { uri: editor.document.uri.toString() },
                        position: { line: pos.line, character: pos.character }
                    });
                    if (result) { showProvenancePanel(result); }
                } catch { /* LSP may not support this */ }
            } else {
                vscode.window.showInformationMessage('Provenance info requires the Sounio language server.');
            }
        })
    );

    // Uncertainty command
    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showUncertainty', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'sounio') { return; }
            if (client) {
                try {
                    const pos = editor.selection.active;
                    const result = await client.sendRequest<any>('sounio/uncertainty', {
                        textDocument: { uri: editor.document.uri.toString() },
                        position: { line: pos.line, character: pos.character }
                    });
                    if (result) { showUncertaintyPanel(result); }
                } catch { /* LSP may not support this */ }
            } else {
                vscode.window.showInformationMessage('Uncertainty info requires the Sounio language server.');
            }
        })
    );

    // Inlay hints
    const inlayProvider = vscode.languages.registerInlayHintsProvider('sounio', {
        provideInlayHints(document: vscode.TextDocument, range: vscode.Range): vscode.InlayHint[] {
            const hints: vscode.InlayHint[] = [];
            const config = vscode.workspace.getConfiguration('sounio');
            if (!config.get('inlayHints.enabled', true)) { return hints; }

            const text = document.getText(range);
            const lines = text.split('\n');

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const lineNum = range.start.line + i;

                if (config.get('inlayHints.typeHints', true)) {
                    const letMatch = line.match(/\b(let|var)\s+(\w+)\s*=(?!.*:)/);
                    if (letMatch) {
                        const col = line.indexOf(letMatch[0]) + letMatch[0].length;
                        const hint = new vscode.InlayHint(
                            new vscode.Position(lineNum, col),
                            ': /* inferred */',
                            vscode.InlayHintKind.Type
                        );
                        hint.tooltip = 'Type inferred by souc (use --show-types to see)';
                        hints.push(hint);
                    }
                }

                if (config.get('inlayHints.confidenceHints', true)) {
                    const uncMatch = line.match(/\bmeasure\s*\([^)]+\)/);
                    if (uncMatch) {
                        const col = line.indexOf(uncMatch[0]) + uncMatch[0].length;
                        const hint = new vscode.InlayHint(
                            new vscode.Position(lineNum, col),
                            ' /* \u03C3 = ? */',
                            vscode.InlayHintKind.Parameter
                        );
                        hint.tooltip = 'Uncertainty of this epistemic measurement';
                        hints.push(hint);
                    }
                }
            }
            return hints;
        }
    });
    context.subscriptions.push(inlayProvider);

    // Epistemic decorations (from LSP)
    if (client) {
        vscode.workspace.onDidChangeTextDocument(event => {
            const editor = vscode.window.activeTextEditor;
            if (editor && event.document === editor.document) {
                updateDecorations(editor, client);
            }
        });
        vscode.window.onDidChangeActiveTextEditor(editor => {
            if (editor) { updateDecorations(editor, client); }
        });
    }
}

function updateStatusBar(): void {
    if (!statusBarItem) { return; }
    const config = vscode.workspace.getConfiguration('sounio');
    const enabled = config.get<boolean>('epistemic.enabled', true);
    statusBarItem.text = enabled ? '$(telescope) Epistemic: ON' : '$(telescope) Epistemic: OFF';
    statusBarItem.tooltip = 'Toggle Sounio Epistemic Mode';
    statusBarItem.backgroundColor = enabled
        ? new vscode.ThemeColor('statusBarItem.prominentBackground')
        : undefined;
}

async function updateDecorations(editor: vscode.TextEditor, client: LanguageClient): Promise<void> {
    const config = vscode.workspace.getConfiguration('sounio');
    if (!config.get<boolean>('epistemic.enabled', true)) { return; }
    if (editor.document.languageId !== 'sounio') { return; }

    try {
        const result = await client.sendRequest<any>('sounio/epistemicAnnotations', {
            textDocument: { uri: editor.document.uri.toString() }
        });
        if (result && result.annotations) {
            const high: vscode.DecorationOptions[] = [];
            const medium: vscode.DecorationOptions[] = [];
            const low: vscode.DecorationOptions[] = [];

            for (const ann of result.annotations) {
                const range = new vscode.Range(
                    ann.range.start.line, ann.range.start.character,
                    ann.range.end.line, ann.range.end.character
                );
                const decoration = { range, hoverMessage: ann.message };
                if (ann.confidence >= 0.8) { high.push(decoration); }
                else if (ann.confidence >= 0.5) { medium.push(decoration); }
                else { low.push(decoration); }
            }

            editor.setDecorations(highConfidenceDecoration, high);
            editor.setDecorations(mediumConfidenceDecoration, medium);
            editor.setDecorations(lowConfidenceDecoration, low);
        }
    } catch { /* LSP may not support epistemic annotations yet */ }
}

function showConfidencePanel(result: any): void {
    const panel = vscode.window.createWebviewPanel('sounioConfidence', 'Confidence Info', vscode.ViewColumn.Beside, {});
    const c = result.confidence || 0;
    const badge = c >= 0.95 ? '\u{1f7e2}' : c >= 0.80 ? '\u{1f7e1}' : c >= 0.60 ? '\u{1f7e0}' : c >= 0.30 ? '\u{1f534}' : '\u26ab';
    const color = c >= 0.8 ? '#4CAF50' : c >= 0.5 ? '#FFC107' : '#F44336';
    panel.webview.html = `<!DOCTYPE html><html><head><style>
        body{font-family:var(--vscode-font-family);padding:20px}
        .badge{font-size:48px;text-align:center}.conf{font-size:24px;text-align:center;margin:20px 0}
        .bar{height:20px;background:#333;border-radius:10px;overflow:hidden}
        .fill{height:100%;background:${color}}
        h3{color:var(--vscode-foreground)}
    </style></head><body>
        <div class="badge">${badge}</div>
        <div class="conf">${(c * 100).toFixed(1)}%</div>
        <div class="bar"><div class="fill" style="width:${c * 100}%"></div></div>
        <h3>Source</h3><p>${result.source || 'Unknown'}</p>
        <h3>Revisability</h3><p>${result.revisability || 'Non-revisable'}</p>
    </body></html>`;
}

function showProvenancePanel(result: any): void {
    const panel = vscode.window.createWebviewPanel('sounioProvenance', 'Provenance Chain', vscode.ViewColumn.Beside, {});
    const chain = result.chain || [];
    const html = chain.map((s: any) => `<div class="step"><span>\u2192</span> <b>${s.name}</b> <i>${s.type}</i></div>`).join('');
    panel.webview.html = `<!DOCTYPE html><html><head><style>
        body{font-family:var(--vscode-font-family);padding:20px}
        h2{color:var(--vscode-foreground)}
        .step{padding:8px;margin:4px 0;background:var(--vscode-editor-background);border-radius:4px}
    </style></head><body>
        <h2>Provenance Chain</h2>${html || '<p>No provenance info available</p>'}
    </body></html>`;
}

function showUncertaintyPanel(result: any): void {
    const panel = vscode.window.createWebviewPanel('sounioUncertainty', 'Uncertainty Info', vscode.ViewColumn.Beside, {});
    const content = result.mean !== undefined ? `
        <div><span class="label">Mean</span><div class="val">${result.mean.toFixed(6)}</div></div>
        <div><span class="label">Std Dev</span><div class="val">\u00B1 ${result.std.toFixed(6)}</div></div>
        <div><span class="label">95% CI</span><div class="val">[${(result.mean - 1.96 * result.std).toFixed(6)}, ${(result.mean + 1.96 * result.std).toFixed(6)}]</div></div>
    ` : '<p>Deterministic value (no uncertainty)</p>';
    panel.webview.html = `<!DOCTYPE html><html><head><style>
        body{font-family:var(--vscode-font-family);padding:20px}
        h2{color:var(--vscode-foreground)}
        .label{color:var(--vscode-descriptionForeground)}
        .val{font-size:20px;font-weight:bold;margin:8px 0 16px}
    </style></head><body>
        <h2>Uncertainty Analysis</h2>${content}
    </body></html>`;
}
