import * as path from 'path';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;
let statusBarItem: vscode.StatusBarItem | undefined;

export function activate(context: vscode.ExtensionContext) {
    // Get server path from configuration
    const config = vscode.workspace.getConfiguration('sounio');
    const serverPath = config.get<string>('serverPath', 'souc');

    // Server options - run LSP via 'souc lsp'
    const serverOptions: ServerOptions = {
        command: serverPath,
        args: ['lsp', '--stdio'],
        transport: TransportKind.stdio
    };

    // Client options
    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'sounio' }
        ],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.{d,sio}')
        },
        outputChannelName: 'Sounio Language Server',
        traceOutputChannel: vscode.window.createOutputChannel('Sounio LSP Trace'),
        initializationOptions: {
            epistemicMode: config.get<boolean>('epistemic.enabled', true),
            confidenceThreshold: config.get<number>('epistemic.confidenceThreshold', 0.8)
        }
    };

    // Create and start the client
    client = new LanguageClient(
        'sounio',
        'Sounio Language Server',
        serverOptions,
        clientOptions
    );

    // Create status bar item for epistemic status
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'sounio.toggleEpistemic';
    updateStatusBar();
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // =========================================================================
    // GENERAL COMMANDS
    // =========================================================================

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.restartServer', async () => {
            if (client) {
                await client.stop();
                await client.start();
                vscode.window.showInformationMessage('Sounio language server restarted');
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.runFile', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                const filePath = editor.document.fileName;
                const terminal = getOrCreateTerminal('Sounio');
                terminal.show();
                terminal.sendText(`souc run "${filePath}"`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.runFileJit', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                const filePath = editor.document.fileName;
                const terminal = getOrCreateTerminal('Sounio JIT');
                terminal.show();
                terminal.sendText(`souc run --jit "${filePath}"`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.checkFile', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                const filePath = editor.document.fileName;
                const terminal = getOrCreateTerminal('Sounio Check');
                terminal.show();
                terminal.sendText(`souc check "${filePath}"`);
            }
        })
    );

    // =========================================================================
    // DEBUG COMMANDS
    // =========================================================================

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showHir', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                const filePath = editor.document.fileName;
                const terminal = getOrCreateTerminal('Sounio HIR');
                terminal.show();
                terminal.sendText(`souc check "${filePath}" --show-hir`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showHlir', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                const filePath = editor.document.fileName;
                const terminal = getOrCreateTerminal('Sounio HLIR');
                terminal.show();
                terminal.sendText(`souc check "${filePath}" --show-hlir`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showAst', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                const filePath = editor.document.fileName;
                const terminal = getOrCreateTerminal('Sounio AST');
                terminal.show();
                terminal.sendText(`souc check "${filePath}" --show-ast`);
            }
        })
    );

    // =========================================================================
    // EPISTEMIC COMMANDS
    // =========================================================================

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.toggleEpistemic', async () => {
            const config = vscode.workspace.getConfiguration('sounio');
            const current = config.get<boolean>('epistemic.enabled', true);
            await config.update('epistemic.enabled', !current, vscode.ConfigurationTarget.Workspace);
            updateStatusBar();
            vscode.window.showInformationMessage(
                `Epistemic mode: ${!current ? 'ON' : 'OFF'}`
            );
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showConfidence', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                // Request confidence info from LSP
                if (client) {
                    const position = editor.selection.active;
                    const result = await client.sendRequest('sounio/confidence', {
                        textDocument: { uri: editor.document.uri.toString() },
                        position: { line: position.line, character: position.character }
                    });
                    if (result) {
                        showConfidencePanel(result);
                    }
                }
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showProvenance', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                if (client) {
                    const position = editor.selection.active;
                    const result = await client.sendRequest('sounio/provenance', {
                        textDocument: { uri: editor.document.uri.toString() },
                        position: { line: position.line, character: position.character }
                    });
                    if (result) {
                        showProvenancePanel(result);
                    }
                }
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showUncertainty', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                if (client) {
                    const position = editor.selection.active;
                    const result = await client.sendRequest('sounio/uncertainty', {
                        textDocument: { uri: editor.document.uri.toString() },
                        position: { line: position.line, character: position.character }
                    });
                    if (result) {
                        showUncertaintyPanel(result);
                    }
                }
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.startRepl', async () => {
            const terminal = vscode.window.createTerminal({
                name: 'Sounio REPL',
                shellPath: 'souc',
                shellArgs: ['repl']
            });
            terminal.show();
        })
    );

    // =========================================================================
    // DECORATIONS FOR EPISTEMIC VISUALIZATION
    // =========================================================================

    const highConfidenceDecoration = vscode.window.createTextEditorDecorationType({
        after: {
            contentText: ' 🟢',
            margin: '0 0 0 4px'
        }
    });

    const mediumConfidenceDecoration = vscode.window.createTextEditorDecorationType({
        after: {
            contentText: ' 🟡',
            margin: '0 0 0 4px'
        }
    });

    const lowConfidenceDecoration = vscode.window.createTextEditorDecorationType({
        after: {
            contentText: ' 🔴',
            margin: '0 0 0 4px'
        }
    });

    // Start the client
    client.start();

    // Update decorations when document changes
    vscode.workspace.onDidChangeTextDocument(event => {
        const editor = vscode.window.activeTextEditor;
        if (editor && event.document === editor.document) {
            updateEpistemicDecorations(editor);
        }
    });

    vscode.window.onDidChangeActiveTextEditor(editor => {
        if (editor) {
            updateEpistemicDecorations(editor);
        }
    });

    // Helper function to update decorations
    async function updateEpistemicDecorations(editor: vscode.TextEditor) {
        const config = vscode.workspace.getConfiguration('sounio');
        if (!config.get<boolean>('epistemic.enabled', true)) {
            return;
        }

        // Request epistemic info from LSP
        if (client && editor.document.languageId === 'sounio') {
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
                            ann.range.start.line,
                            ann.range.start.character,
                            ann.range.end.line,
                            ann.range.end.character
                        );
                        const decoration = { range, hoverMessage: ann.message };

                        if (ann.confidence >= 0.8) {
                            high.push(decoration);
                        } else if (ann.confidence >= 0.5) {
                            medium.push(decoration);
                        } else {
                            low.push(decoration);
                        }
                    }

                    editor.setDecorations(highConfidenceDecoration, high);
                    editor.setDecorations(mediumConfidenceDecoration, medium);
                    editor.setDecorations(lowConfidenceDecoration, low);
                }
            } catch (e) {
                // LSP might not support this request yet
            }
        }
    }
}

// Get or create a named terminal
function getOrCreateTerminal(name: string): vscode.Terminal {
    const existing = vscode.window.terminals.find(t => t.name === name);
    if (existing) {
        return existing;
    }
    return vscode.window.createTerminal(name);
}

// Update status bar item
function updateStatusBar() {
    if (statusBarItem) {
        const config = vscode.workspace.getConfiguration('sounio');
        const enabled = config.get<boolean>('epistemic.enabled', true);
        statusBarItem.text = enabled ? '$(telescope) Epistemic: ON' : '$(telescope) Epistemic: OFF';
        statusBarItem.tooltip = 'Toggle Sounio Epistemic Mode';
        statusBarItem.backgroundColor = enabled
            ? new vscode.ThemeColor('statusBarItem.prominentBackground')
            : undefined;
    }
}

// Show confidence info in a panel
function showConfidencePanel(result: any) {
    const panel = vscode.window.createWebviewPanel(
        'sounioConfidence',
        'Confidence Info',
        vscode.ViewColumn.Beside,
        {}
    );

    const confidence = result.confidence || 0;
    const badge = confidence >= 0.95 ? '🟢' :
                  confidence >= 0.80 ? '🟡' :
                  confidence >= 0.60 ? '🟠' :
                  confidence >= 0.30 ? '🔴' : '⚫';

    panel.webview.html = `
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: var(--vscode-font-family); padding: 20px; }
                .badge { font-size: 48px; text-align: center; }
                .confidence { font-size: 24px; text-align: center; margin: 20px 0; }
                .bar { height: 20px; background: #333; border-radius: 10px; overflow: hidden; }
                .fill { height: 100%; background: ${confidence >= 0.8 ? '#4CAF50' : confidence >= 0.5 ? '#FFC107' : '#F44336'}; }
                .details { margin-top: 20px; }
                h3 { color: var(--vscode-foreground); }
            </style>
        </head>
        <body>
            <div class="badge">${badge}</div>
            <div class="confidence">${(confidence * 100).toFixed(1)}%</div>
            <div class="bar"><div class="fill" style="width: ${confidence * 100}%"></div></div>
            <div class="details">
                <h3>Source</h3>
                <p>${result.source || 'Unknown'}</p>
                <h3>Revisability</h3>
                <p>${result.revisability || 'Non-revisable'}</p>
            </div>
        </body>
        </html>
    `;
}

// Show provenance chain in a panel
function showProvenancePanel(result: any) {
    const panel = vscode.window.createWebviewPanel(
        'sounioProvenance',
        'Provenance Chain',
        vscode.ViewColumn.Beside,
        {}
    );

    const chain = result.chain || [];
    const chainHtml = chain.map((step: any) => `
        <div class="step">
            <span class="icon">→</span>
            <span class="name">${step.name}</span>
            <span class="type">${step.type}</span>
        </div>
    `).join('');

    panel.webview.html = `
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: var(--vscode-font-family); padding: 20px; }
                h2 { color: var(--vscode-foreground); }
                .chain { display: flex; flex-direction: column; gap: 10px; }
                .step { display: flex; align-items: center; gap: 10px; padding: 10px; background: var(--vscode-editor-background); border-radius: 5px; }
                .icon { font-size: 20px; }
                .name { font-weight: bold; }
                .type { color: var(--vscode-descriptionForeground); }
            </style>
        </head>
        <body>
            <h2>Provenance Chain</h2>
            <div class="chain">
                ${chainHtml || '<p>No provenance information available</p>'}
            </div>
        </body>
        </html>
    `;
}

// Show uncertainty info in a panel
function showUncertaintyPanel(result: any) {
    const panel = vscode.window.createWebviewPanel(
        'sounioUncertainty',
        'Uncertainty Info',
        vscode.ViewColumn.Beside,
        {}
    );

    panel.webview.html = `
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: var(--vscode-font-family); padding: 20px; }
                h2 { color: var(--vscode-foreground); }
                .metric { margin: 20px 0; }
                .label { color: var(--vscode-descriptionForeground); }
                .value { font-size: 24px; font-weight: bold; }
            </style>
        </head>
        <body>
            <h2>Uncertainty Analysis</h2>
            ${result.mean !== undefined ? `
                <div class="metric">
                    <div class="label">Mean</div>
                    <div class="value">${result.mean.toFixed(6)}</div>
                </div>
                <div class="metric">
                    <div class="label">Standard Deviation</div>
                    <div class="value">± ${result.std.toFixed(6)}</div>
                </div>
                <div class="metric">
                    <div class="label">95% Confidence Interval</div>
                    <div class="value">[${(result.mean - 1.96 * result.std).toFixed(6)}, ${(result.mean + 1.96 * result.std).toFixed(6)}]</div>
                </div>
            ` : '<p>Deterministic value (no uncertainty)</p>'}
        </body>
        </html>
    `;
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
