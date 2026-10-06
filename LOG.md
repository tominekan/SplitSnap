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

## Task 3: Baseline B0
- Wrote `src/ocr.py` (one `image_to_data` call), `src/parse_regex.py`, `src/pipeline.py` (switchable stages, OCR cache, 8 processes), `src/metrics.py`.
- Checked the regex parser on CORD **validation** (dev only): item P 0.474, R 0.116. Looked at 4 OCR outputs by hand: misses come from Tesseract psm 3 dropping or separating the price column, not from the regex.
- `python src/pipeline.py --config B0` -> `results/logs/pipeline_B0.txt` (298 receipts, 20 s). Predictions in `results/predictions/B0/`.
- Scored -> `results/logs/score_B0.txt`: CORD test item F1 0.213 (P 0.514, R 0.134), total acc 0.126; SRD test CER 0.393, total acc 0.240 (n=146).
- Note: on 6 SRD images Tesseract psm 3 returns no text at all on the raw photo (e.g. 1001, small text on a 645x1000 photo). Verified by calling Tesseract directly.

## Task 4: Rectification
- Wrote `src/rectify.py` (detect, warp, oracle warp, IoU via shapely, corner error / image diagonal, figures).
- First version (Canny then Otsu, first 4-gon): SRD validation mean IoU 0.798, detection 77%. Looked at failures (1107: receipt runs off the bottom edge; 1024: white tray and a credit card on the receipt).
- Compared 7 variants on the 30-image validation split only (`python src/dev/rectify_variants.py` -> `results/logs/rectify_variants_val.txt`). Chose "paper mask (bright + low saturation), then Otsu, then edges, convex hull": IoU 0.885, detection 90%.
- `python src/rectify.py` -> `results/logs/rectify.txt`, `results/rectify_summary.json`, `results/rectify_per_image.json`, 6 figures in `figures/rectify/` (2 best, 2 median, 2 worst by IoU among images where the receipt does not fill the frame).
  - Test (168): detection 0.929, mean IoU 0.832, IoU>=0.9 48%, corner error 0.050.
  - Fell back on 15 images: 1012 1013 1014 1033 1076 1135 1147 1162 1170 1175 1183 1184 1188 1191 1197 (several of them are receipts that fill the frame, where falling back is the right answer).
- Unsure: the SRD corner labels sometimes hug the printed text rather than the paper (1004), so IoU understates detection quality there.
