interface TerminalOutputProps {
  title: string;
  html: string;
}

export default function TerminalOutput({ title, html }: TerminalOutputProps) {
  return (
    <div className="terminal-window" style={{ maxWidth: '100%', margin: '1.5rem 0' }}>
      <style>{`
        .terminal-window {
          background: #1a1a2e;
          border-radius: 8px;
          overflow: hidden;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
          font-family: 'JetBrains Mono', 'Fira Code', monospace;
        }
        .terminal-header {
          background: linear-gradient(180deg, #2d2d44, #252538);
          padding: 0.75rem 1rem;
          display: flex;
          align-items: center;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .terminal-buttons {
          display: flex;
          gap: 8px;
        }
        .terminal-button {
          width: 12px;
          height: 12px;
          border-radius: 50%;
        }
        .terminal-button.close { background: #ff5f57; }
        .terminal-button.minimize { background: #ffbd2e; }
        .terminal-button.maximize { background: #28ca42; }
        .terminal-title {
          flex: 1;
          text-align: center;
          color: rgba(255, 255, 255, 0.5);
          font-size: 0.75rem;
          margin-right: 52px;
        }
        .terminal-body {
          padding: 1rem;
          font-size: 0.75rem;
          line-height: 1.3;
          overflow-x: auto;
          background: #0d0d1a;
          color: #dfe6e9;
        }
        .terminal-body pre {
          margin: 0;
          white-space: pre;
          font-family: inherit;
        }
        .c-red { color: #ff6b6b; }
        .c-green { color: #4ecdc4; }
        .c-yellow { color: #ffd93d; }
        .c-blue { color: #6c5ce7; }
        .c-magenta { color: #fd79a8; }
        .c-cyan { color: #74b9ff; }
        .c-white { color: #ffffff; }
        .c-dim { opacity: 0.5; }
        .c-bold { font-weight: bold; }
      `}</style>
      <div className="terminal-header">
        <div className="terminal-buttons">
          <span className="terminal-button close"></span>
          <span className="terminal-button minimize"></span>
          <span className="terminal-button maximize"></span>
        </div>
        <span className="terminal-title">{title}</span>
      </div>
      <div className="terminal-body">
        <pre dangerouslySetInnerHTML={{ __html: html }} />
      </div>
    </div>
  );
}
