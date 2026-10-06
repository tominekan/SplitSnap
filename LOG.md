# LOG

Chronological record of what was done, commands run, and what failed. Times are local (2026-10-06, overnight run).

## Task 1: Repo skeleton
- Found that the git repository enclosing this folder is `/Users/tominekan/Code` (a parent repo with no commits and many unrelated staged files). Committing there would have committed files outside this project, so I ran `git init -b main` **inside** `SplitSnap/` to create a separate, nested repo. All task commits go to this nested repo. The parent repo's index was not touched. (See DECISIONS.md D01.)
- **Tesseract was NOT installed on the system** (`which tesseract` found nothing; not in Homebrew either). Installing via Homebrew would write outside the project folder, which the constraints forbid. Instead I downloaded the standalone `micromamba` binary into `tools/mamba/` and created a conda-forge environment at `tools/tess/` containing Tesseract 5.5.3 (with `eng` traineddata). Everything lives inside the project; `tools/` is gitignored. (See DECISIONS.md D02.)
  - `curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xj -C tools/mamba bin/micromamba`
  - `MAMBA_ROOT_PREFIX=$PWD/tools/mamba tools/mamba/bin/micromamba create -y -p $PWD/tools/tess -c conda-forge tesseract`
- Python: created `.venv/` (Python 3.12) with `uv`, cache redirected to `.cache/uv` inside the project:
  - `UV_CACHE_DIR=$PWD/.cache/uv uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt`
  - Note: this pulled OpenCV 5.0.0 and scikit-image 0.26.0.
- Created folders `src/ data/ results/ figures/ notebooks/`, plus README.md, requirements.txt, .gitignore, LOG.md, DECISIONS.md.
