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

## Task 2: Data
- Downloaded SRD: `curl -L -o data/srd/Dataset.zip https://zenodo.org/records/13688441/files/Dataset.zip?download=1` (18,998,248 bytes), unzipped. Inspected contents by hand before writing the loader (see `data/README.md`). Surprises: (1) corner order in `homography.csv` is TL, TR, BL, BR, not clockwise; (2) corners are in the EXIF-rotated frame (5 images have an EXIF orientation tag); (3) 198 images but 195 label files, 5 images have no text and 2 labels have no image.
- Downloaded CORD v2 via `datasets` (HF cache redirected to `data/hf/`). Saved test + validation as PNG + JSON.
- `python src/data.py` -> `results/logs/data.txt`: CORD test 100 receipts (268 priced items), 37 with roi corners; SRD 198 images, 193 with text, 198 with corners, val 30 / test 168, reliable totals 171 of 193 (after widening the keyword list from 153).
