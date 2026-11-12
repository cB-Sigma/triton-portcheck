# Security

## What the tool executes

`triton-portcheck` imports and compiles the kernel modules named in a manifest, and `capture` runs the command you
give it. That is arbitrary code execution by design: **only run it on code you trust**, exactly as you would
`pytest`. Child processes are isolated for robustness (timeouts, memory limits), not for security; they are not a
sandbox.

`env add` installs `triton==<version>` and any `--with` requirements from your configured package index.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/cb-Sigma/triton-portcheck/security/advisories/new) for this
repository. Include the version (`triton-portcheck --version`), what you observed and how to reproduce it. You can
expect an acknowledgement within a week.

## Supported versions

Only the latest release receives fixes.
