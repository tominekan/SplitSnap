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

If you already have Tesseract on PATH, `src/common.py` uses it when `tools/tess/` does not exist.

## Run each step
All commands run from the project root with the venv active.

| Step | Command | Output |
|---|---|---|
| 2. Data | `python src/data.py` | `data/cord/`, `data/srd/` (see `data/README.md`), counts in `results/logs/data.txt` |
| 3. Baseline B0 | `python src/pipeline.py --config B0` | `results/predictions/B0/` |
| 4. Rectification | `python src/rectify.py` | `results/rectify_summary.json`, `results/rectify_per_image.json`, `figures/rectify/` |
| 5. Preprocessing sweep | `python src/preprocess.py` (~12 min) | `results/preprocess_sweep.md/.csv`, `results/preprocess_best.json` |
| 6/7. Ablation table | `python src/evaluate.py` (~25 min cold, seconds with the OCR cache) | `results/summary.md`, `results/summary.csv`, `results/predictions/<config>/` |
| 9. Failure analysis | `python src/failures.py` | `results/failures.md`, `figures/rectify_failures/` |
| 10. Figures | `python src/figures.py` | `figures/*.png` |
| 11. Demo | `jupyter notebook notebooks/demo.ipynb` | |

Development scripts used to choose settings on validation splits only: `src/dev/rectify_variants.py`, `src/dev/rectify_margin.py` (logs in `results/logs/`).

`python src/evaluate.py --no-cache` wipes `data/cache/ocr/` and recomputes all OCR.

## SRD field labeling
SRD has transcriptions but no item or total labels. We label them by hand (rules: `labels/GUIDE.md`).

| Step | Command | Output |
|---|---|---|
| Drafts from the transcriptions | `python src/labeling/prefill.py` | `labels/srd_prefill/` |
| Label in the browser | `streamlit run src/labeling/app.py` | `labels/srd/NNNN.json` (committed) |
| Check progress and problems | `python src/labeling/check_labels.py` | printed report |
| Blind relabel for self-agreement | `streamlit run src/labeling/app.py -- --out labels/srd_relabel --no-prefill`, then `python src/labeling/check_labels.py --compare labels/srd_relabel` | printed agreement |

## Code map
- `src/common.py`: paths, Tesseract location, number and text normalization
- `src/data.py`: download and load CORD v2 and SRD, CORD label conversion, SRD split and derived totals
- `src/rectify.py`: receipt detection, warp, oracle warp, corner metrics (iteration 1)
- `src/preprocess.py`: preprocessing options and sweep (iteration 2)
- `src/ocr.py`: Tesseract wrapper (+ optional EasyOCR)
- `src/parse_regex.py`: baseline line-regex parser
- `src/parse_layout.py`: row grouping from word boxes (iteration 3)
- `src/pipeline.py`: switchable stages, configurations, checksum, OCR cache
- `src/metrics.py`, `src/evaluate.py`: scoring and the ablation table
- `src/failures.py`, `src/figures.py`: failure analysis and slide figures

## Datasets
- CORD v2, `naver-clova-ix/cord-v2` on Hugging Face (test split for reporting, validation split for development).
- SRD: labelled ExpressExpense restaurant receipts, Zenodo record 13688441. Auad, Alves, Kakizaki, Reis, Silva, "A Filtering and Image Preparation Approach to Enhance OCR for Fiscal Receipts", SIBGRAPI 2024.
