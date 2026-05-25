# Support Guidelines for Sounio

Welcome to the Sounio community! Sounio is a self-hosted systems and scientific programming language. Whether you are building clinical dosing models, modeling non-associative algebra, or writing GPU kernels, we are here to support your engineering journey.

Please use the appropriate channels below to ask questions, report bugs, or discuss language features.

---

## 1. Ask Questions & Get Help

If you are learning Sounio, running into compiler errors, or trying to design a custom algebraic effect:

- **GitHub Discussions**: 💬 [Sounio Discussions](https://github.com/Sounio-lang/sounio/discussions)  
  This is the best place for open-ended questions, architectural design help, and sharing what you've built.
- **Discord Community**: Join our developer Discord server to chat in real-time with the core language developers. *(Link available on our official website)*
- **WASM Playground**: 🎮 [Interactive Sounio Playground](https://www.souniolang.org/playground)  
  An interactive, zero-install environment in your browser where you can write, compile, and execute Sounio code instantly to experiment.

---

## 2. Document Bugs & Compiler Issues

If you believe you have found a bug in the Sounio compiler (`souc`), standard library, or runtime:

1. **Search Existing Issues**: Check if anyone else has already reported the same issue in the [Sounio Issue Tracker](https://github.com/Sounio-lang/sounio/issues).
2. **Create a Detailed Bug Report**: Use our structured GitHub Issue forms. Our issue triage automated system requires the following info:
   - Release scope (`Compiler`, `Stdlib`, `Docs`, `Website`, `CI`).
   - Sounio code snippet causing the compiler panic or execution error.
   - Core CLI commands executed (e.g. `./bin/souc run <file>.sio`).
   - Expected vs. actual compiler output.

---

## 3. Propose New Language Features (RFCs)

Sounio is an active research platform, and we welcome extensions to the type system, standard library, and runtime:

- To propose a new feature, open a **Feature Request** on GitHub.
- For complex changes touching the type checker, the effects engine, or the SSA optimization passes, please draft an **RFC (Request for Comments)** and submit it in the Sounio Discussions "RFC" category.

---

## 4. Documentation

Our comprehensive documentation guides are fully synchronized and authoritative:

- 📖 **Official Website Docs**: [Sounio Docs & Manuals](https://www.souniolang.org/docs/)
- 🐍 **LLM Programming Guide**: [Sounio LLM Guide](https://github.com/Sounio-lang/sounio/blob/main/docs/guide/LLM_PROGRAMMING_GUIDE.md) — The complete language specification written specifically for code tools and developers.

---

Thank you for being part of the Sounio ecosystem! 🏛️
