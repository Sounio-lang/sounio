# Security Policy

## Supported Versions

Security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in Sounio, please report it privately by emailing:

**security@sounio.dev**

Please include the following information in your report:

- Type of vulnerability (e.g., buffer overflow, code injection, etc.)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Response Timeline

- **Initial Response**: Within 48 hours of your report
- **Status Update**: Within 7 days with our assessment
- **Resolution**: Depending on severity, typically within 30-90 days

## Disclosure Policy

- We will acknowledge receipt of your vulnerability report
- We will confirm the vulnerability and determine its impact
- We will release a fix as soon as possible, depending on complexity
- We will publicly disclose the vulnerability after the fix is released
- We will credit you for your discovery (unless you prefer to remain anonymous)

## Scope

This security policy applies to:

- The Sounio compiler (`souc`)
- The standard library (`stdlib/`)
- Official tooling (LSP server, package manager)

### Out of Scope

- Third-party packages or libraries
- Vulnerabilities in dependencies (report these upstream)
- Social engineering attacks
- Physical attacks

## Safe Harbor

We consider security research conducted in good faith to be authorized. We will not pursue civil or legal action against researchers who:

- Act in good faith to avoid privacy violations, data destruction, or service disruption
- Only interact with accounts they own or with explicit permission
- Report vulnerabilities promptly and do not exploit them maliciously
- Follow this responsible disclosure policy

Thank you for helping keep Sounio and its users safe.
