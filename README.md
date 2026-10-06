# SplitSnap: preliminary experiments (CIS 5810)

Photograph a restaurant receipt, extract line items, subtotal, tax, tip and total, then split the bill. This repo holds the **traditional (classic CV + Tesseract, no LLM)** baseline and three measured iterations on top of it.

See `LOG.md` (what was run, what failed) and `DECISIONS.md` (every design decision and its tradeoffs).

## Setup
All tools live inside this folder.

```bash
# Python deps
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
source .venv/bin/activate

# Tesseract (only if you don't have it on PATH): project-local install
mkdir -p tools/mamba && curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xj -C tools/mamba bin/micromamba
MAMBA_ROOT_PREFIX=$PWD/tools/mamba tools/mamba/bin/micromamba create -y -p $PWD/tools/tess -c conda-forge tesseract
```
