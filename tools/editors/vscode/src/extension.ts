import * as fs from 'fs';
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

// Resolve the souc launcher. Priority:
//   1. `sounio.serverPath` setting (absolute path or PATH-name).
//   2. `bin/souc` relative to the first workspace folder (in-tree dev).
//   3. Bare `souc` (relies on $PATH).
function resolveSoucPath(configured: string): string {
    if (configured && configured !== 'souc' && fs.existsSync(configured)) {
        return configured;
    }
    const folders = vscode.workspace.workspaceFolders;
    if (folders && folders.length > 0) {
        const local = path.join(folders[0].uri.fsPath, 'bin', 'souc');
        if (fs.existsSync(local)) { return local; }
    }
    return configured || 'souc';
}

export function activate(context: vscode.ExtensionContext) {
    // Get server path from configuration
    const config = vscode.workspace.getConfiguration('sounio');
    const configuredPath = config.get<string>('serverPath', 'souc');
    const serverPath = resolveSoucPath(configuredPath);

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
    // FORMATTING COMMANDS
    // =========================================================================

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.formatDocument', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                await vscode.commands.executeCommand('editor.action.formatDocument');
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.organizeImports', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                if (client) {
                    await client.sendRequest('workspace/executeCommand', {
                        command: 'sounio.organizeImports',
                        arguments: [editor.document.uri.toString()]
                    });
                    vscode.window.showInformationMessage('Imports organized');
                }
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.addImport', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                const moduleName = await vscode.window.showInputBox({
                    prompt: 'Enter module name to import',
                    placeHolder: 'e.g., std.math'
                });
                if (moduleName) {
                    const edit = new vscode.WorkspaceEdit();
                    const position = new vscode.Position(0, 0);
                    edit.insert(editor.document.uri, position, `use ${moduleName};\n`);
                    await vscode.workspace.applyEdit(edit);
                }
            }
        })
    );

    // =========================================================================
    // NAVIGATION COMMANDS
    // =========================================================================

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showOutline', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                await vscode.commands.executeCommand('workbench.action.gotoSymbol');
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.findAllReferences', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                await vscode.commands.executeCommand('editor.action.referenceSearch.trigger');
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.renameSymbol', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                await vscode.commands.executeCommand('editor.action.rename');
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showSignature', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                await vscode.commands.executeCommand('editor.action.triggerParameterHints');
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
    // CODE LENS PROVIDER
    // =========================================================================

    const codeLensProvider = vscode.languages.registerCodeLensProvider('sounio', {
        provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
            const codeLenses: vscode.CodeLens[] = [];
            const text = document.getText();
            const functionRegex = /^(?:pub\s+)?(?:fn|func|function|def)\s+(\w+)/gm;
            let match;

            while ((match = functionRegex.exec(text)) !== null) {
                const line = document.positionAt(match.index).line;
                const range = new vscode.Range(line, 0, line, 0);
                
                // Run code lens
                codeLenses.push(new vscode.CodeLens(range, {
                    title: '$(play) Run',
                    command: 'sounio.runFile',
                    tooltip: 'Run this file'
                }));

                // Check code lens
                codeLenses.push(new vscode.CodeLens(range, {
                    title: '$(check) Check',
                    command: 'sounio.checkFile',
                    tooltip: 'Type-check this file'
                }));
            }

            return codeLenses;
        }
    });
    context.subscriptions.push(codeLensProvider);

    // =========================================================================
    // INLAY HINTS PROVIDER
    // =========================================================================

    const inlayHintsProvider = vscode.languages.registerInlayHintsProvider('sounio', {
        provideInlayHints(document: vscode.TextDocument, range: vscode.Range): vscode.InlayHint[] {
            const hints: vscode.InlayHint[] = [];
            const config = vscode.workspace.getConfiguration('sounio');
            
            if (!config.get('inlayHints.enabled', true)) {
                return hints;
            }

            const text = document.getText(range);
            const lines = text.split('\n');
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const lineNum = range.start.line + i;

                // Type hints for let bindings
                if (config.get('inlayHints.typeHints', true)) {
                    const letMatch = line.match(/let\s+(\w+)\s*=(?!.*:)/);
                    if (letMatch) {
                        const char = line.indexOf(letMatch[0]) + letMatch[0].length;
                        const hint = new vscode.InlayHint(
                            new vscode.Position(lineNum, char),
                            ': /* inferred */',
                            vscode.InlayHintKind.Type
                        );
                        hint.tooltip = 'Inferred type (use --show-types to see actual type)';
                        hints.push(hint);
                    }
                }

                // Confidence hints for epistemic values
                if (config.get('inlayHints.confidenceHints', true)) {
                    const uncertainMatch = line.match(/uncertain\s*\([^)]+\)/);
                    if (uncertainMatch) {
                        const char = line.indexOf(uncertainMatch[0]) + uncertainMatch[0].length;
                        const hint = new vscode.InlayHint(
                            new vscode.Position(lineNum, char),
                            ' /* σ = ? */',
                            vscode.InlayHintKind.Parameter
                        );
                        hint.tooltip = 'Standard deviation of this uncertain value';
                        hints.push(hint);
                    }
                }
            }

            return hints;
        }
    });
    context.subscriptions.push(inlayHintsProvider);

    // =========================================================================
    // HOVER PROVIDER FOR EPISTEMIC INFO
    // =========================================================================

    const hoverProvider = vscode.languages.registerHoverProvider('sounio', {
        provideHover(document: vscode.TextDocument, position: vscode.Position) {
            const config = vscode.workspace.getConfiguration('sounio');
            if (!config.get('epistemic.enabled', true)) {
                return null;
            }

            const wordRange = document.getWordRangeAtPosition(position);
            if (!wordRange) {
                return null;
            }

            const word = document.getText(wordRange);
            const markdown = new vscode.MarkdownString();

            // Add epistemic info if available
            if (word.includes('uncertain') || word.includes('confidence')) {
                markdown.appendCodeblock(`epistemic ${word}`, 'sounio');
                markdown.appendMarkdown(`\n**Confidence**: Calculated from source\n`);
                markdown.appendMarkdown(`**Provenance**: Tracked\n`);
            }

            return markdown.value ? new vscode.Hover(markdown, wordRange) : null;
        }
    });
    context.subscriptions.push(hoverProvider);

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
