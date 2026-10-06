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

## Task 5: Preprocessing
- Wrote `src/preprocess.py` (5 options + sweep). First attempt estimated character height and skew from a global Otsu mask, which failed on unevenly lit receipts (1001: 5 px characters, 5 deg skew). Switched to an adaptive-threshold text mask; estimates became plausible (~20 px characters, < 1 deg skew on rectified receipts).
- `python src/preprocess.py` -> `results/logs/preprocess_sweep.txt`, `results/preprocess_sweep.md/.csv`, `results/preprocess_best.json`. Grid: 3 psm x 48 option combinations on the 30 SRD validation images after our rectification, ~11.5 min.
- First result (with the first rectification): psm 6 + denoise + upscale + CLAHE + Sauvola, CER 0.231. **Superseded**: after rectification was revised (see "Task 4, revised"), I cleared the cache and re-ran the sweep. Final: **psm 4 + denoise + upscale + CLAHE + deskew + Sauvola**, validation CER 0.208 (psm 3 with no preprocessing: 0.342; psm 6 with no preprocessing: 0.270). The winner changing between runs shows the exact combination is fragile; denoise + upscale + Sauvola is the stable part.
- While the sweep ran, I ran a parse-only check (no OCR) of the layout parser on cached CORD validation OCR, which may have slightly perturbed the sweep's runtime column (not the CER).

## Incident: disk full (during task 6 development)
- `OSError: [Errno 28] No space left on device` while running `src/dev/rectify_margin.py`; the volume had 130 MB free.
- Freed space **only inside the project**: deleted `data/hf/` (6.5 GB Hugging Face cache, mostly the unused CORD train split; test/validation are already exported to `data/cord/`) and `.cache/uv/` (uv download cache). Free space went to 6.7 GB. Nothing outside the folder was touched. To re-download CORD, delete `data/cord/*/index.json` and run `python src/data.py`.
- Free space on this machine is tight; the OCR cache in `data/cache/ocr/` is a few hundred MB and can be deleted at any time.

## Task 4, revised during task 6
- When I ran the full configurations on CORD **validation**, rectification made results worse (layout parser, psm 6: F1 0.349 without rectification, 0.186 with). Viewed before/after pairs: clipped price columns and sheared warps from a wrong corner (receipt running off the frame, shadows).
- Added a 6% outward margin, a plausibility check (corner angles 85-95 deg, opposite sides within 10%) and a bounding-box crop when the check fails. Compared 11 variants on CORD validation + SRD validation (`python src/dev/rectify_margin.py` -> `results/logs/rectify_margin_val.txt`). Chosen: margin 0.06, check (85, 0.9), else crop: CORD val F1 0.365, SRD val CER 0.273 (vs 0.349 / 0.312 with no rectification).
- Cleared the OCR cache, re-ran `python src/rectify.py` (figures regenerated; they now show the detected quad in red and the region used in orange) and re-ran the preprocessing sweep with the new rectification (the earlier sweep's numbers are superseded).

## Task 6: Layout-aware parsing
- Wrote `src/parse_layout.py` (row grouping by y-center with tolerance 0.5 x median word height, right-most price per row, leading quantity, name borrowed from the previous text row for name-less price rows).
- Development on CORD validation only: on B0 OCR, regex F1 0.186 vs layout 0.250 (row tolerance 0.3 / 0.5 / 0.7 -> 0.219 / 0.250 / 0.248).
- After the demo notebook (task 11) showed `TAX «$1.40` being missed, added stripping of stray symbols glued to price tokens (generic rule; applied before the final test evaluation).
- Test comparison on identical OCR (CORD test, from `results/summary.md`): `+rectify+preprocess` (regex) F1 0.339 vs `+rectify+preprocess+layout` F1 0.451; precision 0.496 -> 0.554, recall 0.257 -> 0.381. SRD total accuracy 0.479 -> 0.514 (CER is identical by construction: same OCR).

## Task 7: Evaluation
- Wrote `src/evaluate.py` (all configurations, CORD test + SRD test, writes `results/summary.csv`, `results/summary.md`, `results/cord_per_receipt_f1.csv`, log `results/logs/evaluate.txt`).
- Added an extra row beyond the task's list: `preprocess+layout (no rectify)`, the final pipeline with rectification switched off, to measure rectification's contribution directly.
- Cold run (empty OCR cache): about 8 minutes in total (22 s for B0, 121-164 s for the configurations with preprocessing; CORD images up to 4096x2304 make denoising slow). I then re-ran it from the cache after fixing decision-number references in the markdown text; `results/logs/evaluate.txt` is from that cached rerun. Its numbers are identical, and the runtimes are the ones stored by the cold run.
- Headline (CORD test item F1 / SRD test CER): B0 0.213 / 0.393; +rectify 0.233 / 0.379; +rectify+preprocess 0.339 / 0.245; +rectify+preprocess+layout **0.451** / 0.245; oracle-rectify 0.237 / 0.320; oracle-rectify+preprocess+layout 0.457 / **0.178**; preprocess+layout without rectify 0.437 / 0.265.
- Things to be careful about when presenting:
  - CORD oracle only exists for the 37/100 test receipts with a `roi`; the other 63 are unwarped, so on CORD the oracle row is mostly the no-rectify row.
  - On SRD, rectification helps the mean CER (0.265 -> 0.245) but not the median (0.192 -> 0.198): it rescues bad cases and slightly hurts some good ones.
  - Checksum pass rates are tiny (2-7%), and even ground truth passes only 54%, so the checksum is not yet a useful signal on CORD.

## Task 9: Failure analysis
- Wrote `src/failures.py`; `python src/failures.py` -> `results/failures.md`, `results/failures_cord.json`, `figures/rectify_failures/` (log `results/logs/failures.txt`).
- Best non-oracle configuration by CORD F1: `+rectify+preprocess+layout`. **53 of 100 CORD test receipts have item F1 = 0**, so "8 worst" is a tie; I broke ties by the number of true items.
- Automatic cause guesses (is the true price / name anywhere in the OCR text?) were then checked by looking at every image; I wrote a one-line note for each of the 12 cases (`MANUAL_NOTES`), which replaces the automatic guess. Main causes: digit misreads on faint/small/coloured print, two-line item formats, crumpled paper; for rectification, non-white paper, white-on-white backgrounds, and a plastic sleeve.

## Task 10: Figures
- Wrote `src/figures.py`; `python src/figures.py` -> `figures/ablation_cord_f1.png`, `figures/srd_cer.png`, `figures/pipeline.png`, `figures/preprocess_example.png` (SRD 1004), `figures/failure_1_test_059.png`, `figures/failure_2_test_014.png`, `figures/failure_3_test_061.png` (first 3 of the 8 CORD failures, with OCR word boxes coloured by grouped row and true vs predicted items). Rectification figures are in `figures/rectify/` (task 4) and `figures/rectify_failures/` (task 9).
- Looked at every figure; fixed one overlap (chart title vs legend note).
