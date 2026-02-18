# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A FastMCP server exposing ProtTrans protein language model tools (embedding extraction, log-likelihood scoring, fitness model training/prediction) as MCP tools for Claude Code and other MCP clients.

## Setup & Run

```bash
# Quick setup (creates conda env with CUDA PyTorch)
bash quick_setup.sh

# Run the MCP server directly
env/bin/python src/server.py

# Docker (CPU-only PyTorch, for MCP serving)
docker build -t prottrans_mcp .
docker run prottrans_mcp
```

## Architecture

**MCP layer** (`src/`):
- `src/server.py` — Entry point. Creates root `FastMCP("prottrans_mcp")` and mounts sub-MCPs.
- `src/tools/prottrans_fitness_modeling.py` — Defines 4 MCP tools that wrap functions from `scripts/`. Uses `sys.path.insert` to import from `scripts/`.

**Core logic** (`scripts/`):
- `scripts/prottrans_embedding.py` — ProtT5-XL / ProtAlbert embedding extraction
- `scripts/prottrans_llh.py` — ProtBERT masked-LM log-likelihood scoring
- `scripts/prottrans_train_fitness.py` — PCA + regression (SVR, XGBoost, etc.) with 5-fold CV
- `scripts/prottrans_predict_fitness.py` — Inference with trained models
- `scripts/fitness_pred/` — Standalone training/embedding scripts (not used by MCP layer)

**Key pattern**: MCP tool functions in `src/tools/` are thin wrappers. Business logic lives in `scripts/`. The tool functions catch all exceptions and return `{"status": "error", ...}` dicts rather than raising.

## Dependencies

- `fastmcp` is installed separately with `--ignore-installed` (conflicts with other packages otherwise)
- `requirements.txt` has no `--index-url` — Docker uses CPU PyTorch (`--extra-index-url .../cpu`), conda env uses CUDA (`--index-url .../cu118`)
- `multiprocessing.set_start_method('spawn', force=True)` is set in `server.py` (required for PyTorch + MCP)

## CI/CD

- `.github/workflows/docker.yml` — Builds and pushes Docker image to GHCR on push to main or version tags
- `.github/workflows/build-env.yml` — Packs conda env with CUDA PyTorch, uploads as GitHub Release on `envs-v*` tags

## Directories in .gitignore

`repo/`, `env/`, `example/`, `tmp/`, `tests/`, `templates/` are all gitignored. `scripts/` and `src/` are tracked.
