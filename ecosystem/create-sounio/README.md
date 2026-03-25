# create-sounio

Scaffold a new [Sounio](https://pages.souniolang.org) project instantly.

## Usage

```bash
npx create-sounio my-project
cd my-project
souc run main.sio
```

## What you get

```
my-project/
  main.sio     — Hello World in Sounio
  CLAUDE.md    — AI coding assistant context (works with Claude Code, Cursor, Copilot)
  .gitignore   — ignores compiled ELF binaries
```

## Install souc

Download the Sounio compiler from [GitHub Releases](https://github.com/sounio-lang/sounio/releases):

```bash
# Linux x86-64
curl -L https://github.com/sounio-lang/sounio/releases/latest/download/souc-linux-x86_64 -o souc
chmod +x souc
export SOUC=$(pwd)/souc
```

## About Sounio

Sounio is an L0 systems language for **epistemic computing** — programs that know their own uncertainty.

```sio
fn main() with IO {
    println("Hello from Sounio!")
}
```

Key features: effect system, `Knowledge<T>` for uncertainty propagation, linear types, units of measure, refinement types.

[Documentation](https://pages.souniolang.org) · [GitHub](https://github.com/sounio-lang/sounio)
