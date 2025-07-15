# Dev Container Toolchain

This repository uses VS Code Dev Containers to ensure consistent development environments across macOS (Metal/MPS) and Linux (CUDA).

## Installed Tools

All dev containers include the following tools installed in `/usr/local/bin`:

- **Poetry 1.8.2**: Python dependency management
- **Node.js 18**: JavaScript runtime (required for Claude CLI)
- **Claude Code CLI**: Anthropic's AI coding assistant (`@anthropic-ai/claude-code`)

## Rebuilding the Container

If you need to update tool versions or the container configuration:

1. Open Command Palette: `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Linux/Windows)
2. Run: **Dev Containers: Rebuild Container**

## Using Claude Code

Once inside the container, you can use Claude Code from any terminal:

```bash
claude chat
```

The CLI is globally installed via npm and available in all terminal sessions.

## Container Variants

- **metal**: Optimized for macOS with Metal/MPS support
- **cuda**: For Linux/WSL with NVIDIA GPU support (when available)

## Verification

The `postCreateCommand` automatically runs `./scripts/preflight.sh` to verify all tools are properly installed and accessible. This script checks:

- Poetry availability and version
- Node.js availability and version  
- Claude CLI availability and version
- GPU support (CUDA on Linux, Metal/MPS on macOS)