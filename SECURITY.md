# Security Policy

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in RCMES-MCP, please report it responsibly.

**Do not file a public GitHub issue for security vulnerabilities.**

Instead, please report vulnerabilities by emailing: **kyongsik.yun@jpl.nasa.gov**

Please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Best Practices

When deploying RCMES-MCP:

- Never commit `.env` files or API keys to version control
- Use environment variables or secret managers for credentials
- Keep dependencies up to date (`pip install --upgrade`)
- Run behind a reverse proxy (nginx) in production
- Restrict CORS origins to trusted domains
