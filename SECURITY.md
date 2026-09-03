# Security Policy

Sounio is designed for safety-critical environments (including clinical dosing algorithms and aerospace sensor fusion). We treat security vulnerabilities with the highest priority and urgency.

## Supported Versions

Security patches are actively backported to all stable and pre-release milestones:

| Major Version Series | Supported |
| -------------------- | ------------------ |
| 2.x.x (Current Self-Hosted) | :white_check_mark: |
| 1.0.x-beta (WASM / Legacy) | :white_check_mark: |
| < 1.0.0 (Prerelease prototypes) | :x: |

---

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues, discussions, or PRs.**

If you discover a security vulnerability (e.g., compile-time bypass of epistemic confidence limits, buffer overflows in PTX emission, or privilege escalation in sandboxed compilation), please report it privately by emailing:

📧 **security@sounio.dev**

Please include the following details in your report to help our engineers evaluate and triage the issue quickly:

1. **Vulnerability Type**: (e.g., Out-of-bounds read, type-checker bypass, memory safety violation in runtime).
2. **Affected Code**: Full paths of files and line numbers in Sounio's standard library or self-hosted compiler.
3. **Environment**: Git commit SHA, branch, and compile/execution target (native x86_64, WASM playground, or GPU execution).
4. **Reproduction Steps**: Detailed instructions, commands, or a Sounio program (`.sio`) to reproduce the vulnerability.
5. **Impact**: An explanation of how this vulnerability could be exploited in a production clinical or scientific setting.

---

## Response and Disclosure Timeline

We follow a coordinated disclosure process to protect systems deploying Sounio:

- **Initial Response**: We will acknowledge and respond to your report within **24 hours**.
- **Assessment**: Within **5 business days**, we will send you our technical assessment and confirm the validity of the exploit.
- **Fix & Testing**: We target releasing a patched compiler/runtime version within **30 days** of validation.
- **Coordinated Disclosure**: Once patches are deployed, we will coordinate public disclosure (via CVE assignment and release notes) and credit you for your discovery (unless you prefer anonymity).

Thank you for helping keep Sounio secure! 🛡️
