import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

import { createDiagnostics, checkDocument, resolveSoucPath } from './diagnostics';
import { createRustismDetector } from './rustismDetector';
import { createEpistemicUI } from './epistemicUI';

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext): void {
    // 1. Try LSP, fall back to souc-check diagnostics
    tryStartLsp(context);

    // 2. Always enable souc-check diagnostics as baseline
    createDiagnostics(context);

    // 3. Rustism detector (always available, no LSP needed)
    createRustismDetector(context);

    // 4. Epistemic UI (gracefully handles missing LSP)
    createEpistemicUI(context, client || null);

    // 5. Register commands
    registerCommands(context);

    // 6. Code lens
    registerCodeLens(context);
}

function tryStartLsp(context: vscode.ExtensionContext): void {
    const config = vscode.workspace.getConfiguration('sounio');
    if (!config.get<boolean>('lsp.enabled', true)) { return; }

    const soucPath = resolveSoucPath();
    if (!soucPath) { return; }

    const serverOptions: ServerOptions = {
        command: soucPath,
        args: ['lsp', '--stdio'],
        transport: TransportKind.stdio
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'sounio' }],
        synchronize: { fileEvents: vscode.workspace.createFileSystemWatcher('**/*.sio') },
        outputChannelName: 'Sounio Language Server',
        initializationOptions: {
            epistemicMode: config.get<boolean>('epistemic.enabled', true),
            confidenceThreshold: config.get<number>('epistemic.confidenceThreshold', 0.8)
        }
    };

    client = new LanguageClient('sounio', 'Sounio Language Server', serverOptions, clientOptions);

    client.start().then(() => {
        // LSP connected successfully
    }).catch(() => {
        // LSP not available — souc-check diagnostics will handle it
        client = undefined;
    });
}

function registerCommands(context: vscode.ExtensionContext): void {
    // Run
    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.runFile', () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'sounio') { return; }
            const term = getOrCreateTerminal('Sounio');
            term.show();
            const souc = resolveSoucPath() || 'souc';
            term.sendText(`${souc} run "${editor.document.fileName}"`);
        })
    );

    // Check
    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.checkFile', () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'sounio') { return; }
            checkDocument(editor.document);
        })
    );

    // Debug views
    for (const view of ['Ast', 'Hir', 'Hlir']) {
        context.subscriptions.push(
            vscode.commands.registerCommand(`sounio.show${view}`, () => {
                const editor = vscode.window.activeTextEditor;
                if (!editor || editor.document.languageId !== 'sounio') { return; }
                const term = getOrCreateTerminal(`Sounio ${view.toUpperCase()}`);
                term.show();
                const souc = resolveSoucPath() || 'souc';
                term.sendText(`${souc} check "${editor.document.fileName}" --show-${view.toLowerCase()}`);
            })
        );
    }

    // REPL
    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.startRepl', () => {
            const souc = resolveSoucPath() || 'souc';
            const term = vscode.window.createTerminal({ name: 'Sounio REPL', shellPath: souc, shellArgs: ['repl'] });
            term.show();
        })
    );

    // Restart LSP
    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.restartServer', async () => {
            if (client) {
                await client.stop();
                await client.start();
                vscode.window.showInformationMessage('Sounio language server restarted');
            } else {
                vscode.window.showInformationMessage('LSP not available. Using souc check for diagnostics.');
            }
        })
    );

    // Detect Rustisms (manual trigger)
    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.detectRustisms', () => {
            vscode.window.showInformationMessage('Rustism detection is always active. Check the Problems panel for warnings.');
        })
    );

    // Add import (no semicolons!)
    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.addImport', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.languageId !== 'sounio') { return; }
            const moduleName = await vscode.window.showInputBox({
                prompt: 'Enter module and item to import',
                placeHolder: 'e.g., epistemic::{Knowledge}'
            });
            if (moduleName) {
                const edit = new vscode.WorkspaceEdit();
                edit.insert(editor.document.uri, new vscode.Position(0, 0), `use ${moduleName}\n`);
                await vscode.workspace.applyEdit(edit);
            }
        })
    );
}

function registerCodeLens(context: vscode.ExtensionContext): void {
    const provider = vscode.languages.registerCodeLensProvider('sounio', {
        provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
            const lenses: vscode.CodeLens[] = [];
            const fnRegex = /^(?:pub\s+)?fn\s+(\w+)/gm;
            const text = document.getText();
            let match;

            while ((match = fnRegex.exec(text)) !== null) {
                const line = document.positionAt(match.index).line;
                const range = new vscode.Range(line, 0, line, 0);

                if (match[1] === 'main') {
                    lenses.push(new vscode.CodeLens(range, {
                        title: '$(play) Run',
                        command: 'sounio.runFile',
                        tooltip: 'Run this file'
                    }));
                }

                lenses.push(new vscode.CodeLens(range, {
                    title: '$(check) Check',
                    command: 'sounio.checkFile',
                    tooltip: 'Type-check this file'
                }));
            }

            return lenses;
        }
    });
    context.subscriptions.push(provider);
}

function getOrCreateTerminal(name: string): vscode.Terminal {
    const existing = vscode.window.terminals.find(t => t.name === name);
    return existing || vscode.window.createTerminal(name);
}

export function deactivate(): Thenable<void> | undefined {
    return client ? client.stop() : undefined;
}
