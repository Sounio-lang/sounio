import * as path from 'path';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext) {
    // Get server path from configuration
    const config = vscode.workspace.getConfiguration('sounio');
    const serverPath = config.get<string>('serverPath', 'sounio-lsp');

    // Server options
    const serverOptions: ServerOptions = {
        command: serverPath,
        args: ['--stdio'],
        transport: TransportKind.stdio
    };

    // Client options
    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'sounio' }
        ],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.{d,dem}')
        },
        outputChannelName: 'Sounio Language Server',
        traceOutputChannel: vscode.window.createOutputChannel('Sounio LSP Trace')
    };

    // Create and start the client
    client = new LanguageClient(
        'sounio',
        'Sounio Language Server',
        serverOptions,
        clientOptions
    );

    // Register commands
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
                const terminal = vscode.window.createTerminal('Sounio');
                terminal.show();
                terminal.sendText(`dc run "${filePath}"`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.runFileJit', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                const filePath = editor.document.fileName;
                const terminal = vscode.window.createTerminal('Sounio JIT');
                terminal.show();
                terminal.sendText(`dc run --jit "${filePath}"`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showHir', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                const filePath = editor.document.fileName;
                const terminal = vscode.window.createTerminal('Sounio HIR');
                terminal.show();
                terminal.sendText(`dc dump-hir "${filePath}"`);
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('sounio.showHlir', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'sounio') {
                const filePath = editor.document.fileName;
                const terminal = vscode.window.createTerminal('Sounio HLIR');
                terminal.show();
                terminal.sendText(`dc dump-hlir "${filePath}"`);
            }
        })
    );

    // Start the client
    client.start();
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
