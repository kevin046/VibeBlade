# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please
report it responsibly.

**Do NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report them by:

1. **Email**: Send a report to the maintainer through GitHub's
   [private vulnerability reporting](https://github.com/kevin046/VibeBlade/security)
   feature. This ensures the report is only visible to maintainers.

2. **What to include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fix (optional but appreciated)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 7 days
- **Fix timeline**: Depends on severity, but we aim for a patch release within 30 days

## Security Best Practices for Contributors

- Never commit API keys, tokens, passwords, or credentials
- Use environment variables for secrets (`os.environ.get("KEY")`)
- Review the output of `ruff check` before submitting PRs
- Keep dependencies up to date (`pip audit` or `uv pip compile`)
- Don't use `pickle` on untrusted data
- Validate all external inputs in inference paths

## Dependencies

VibeBlade has minimal dependencies:

- **Required**: numpy (compute), safetensors (model loading)
- **Optional**: onnxruntime (ONNX backend), torch (ONNX export), pybind11 (C++ extensions)
- **Dev**: pytest, ruff

We audit dependencies regularly and pin versions in release builds.
