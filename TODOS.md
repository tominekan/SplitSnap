# SplitSnap TODOs

Tracks the plan from the proposal (Oct 9, 2026). Tick a box when the work is merged **and** measured.
Owner in brackets. Targets for the final report: **CORD item F1 ≥ 0.65**, **SRD total accuracy ≥ 0.75**, and a checksum that reliably flags receipts for review.

Last updated: Oct 26, 2026

## Where we are

| Metric (test split) | B0 | Best classic now | Target | VLM reference |
|---|---|---|---|---|
| CORD item F1 | 0.213 | 0.512 | 0.65 | 0.934 |
| CORD total accuracy | 0.126 | 0.474 | | 0.989 |
| SRD CER (lower is better) | 0.393 | 0.250 | | 0.105 |
| SRD total accuracy | 0.240 | 0.610 | 0.75 | 1.000 |

Plots: `notebooks/results.ipynb` (also saved to `figures/results/`).

---

## Done

### Proposal: baseline and five iterations (Oct 9) [All]
- [x] Datasets: CORD v2 (validation for development, 100 test receipts) and SRD (30 validation, 168 test)
- [x] Metrics: item P/R/F1, field accuracy, checksum pass rate, SRD CER, SRD total accuracy, corner IoU and error
- [x] B0 baseline: Tesseract on the raw photo + regex parser
- [x] Iteration 1: rectification (mask + Otsu/Canny fallbacks, 6% margin, plausibility check, crop fallback)
- [x] Iteration 2: preprocessing sweep (144 combinations; denoise, upscale, CLAHE, deskew, Sauvola)
- [x] Iteration 3: layout-aware parser (word boxes grouped into rows)
- [x] Iteration 4: EasyOCR (single process, 1600 px cap, no preprocessing)
- [x] Iteration 5: parser v2, robust to OCR errors (`$` misreads, broken keywords, split prices, arithmetic tie-break)
- [x] Oracle-corner reference runs (headroom for a better corner detector)
- [x] Checksum stage (items = subtotal, subtotal + tax + tip = total, within 0.5%)
- [x] Bill-splitting demo notebook (`notebooks/demo.ipynb`)
- [x] Failure analysis and slide figures
- [x] LOG.md and DECISIONS.md

### Since the proposal
- [x] VLM reference baseline (Gemini 3.5 Flash) on all CORD and SRD test receipts, scored the same way (`src/vlm_baseline.py`, `results/vlm_summary_gemini-3.5-flash.md`)
- [x] Finding: some SRD transcriptions are garbled (e.g. receipt 1045), which inflates CER, so the text references aren't fully reliable
- [x] SRD labeling tooling: labeling guide (`labels/GUIDE.md`), drafts from the transcriptions, Streamlit labeling app, label checker
- [x] Results notebook with F1, CER and all assessments (`notebooks/results.ipynb`)

---

## In progress / overdue

### Hand-label SRD receipts (planned Oct 12–18) [Tomi]
- [ ] Label the 30 validation receipts first
- [ ] Label the test receipts, ~100 total for the proposal goal (all 198 eventually)
- [ ] Blind relabel ~20 receipts and report self-agreement
- [ ] Add SRD item F1 and subtotal/tax accuracy to `src/evaluate.py` using the labels
- [ ] Re-score the classic pipeline and the VLM on the new labels

### Compare other OCR models (planned Oct 19–25) [Connor, Makayla]
- [ ] Run EasyOCR on CORD validation on a GPU (Colab)
- [ ] Try PaddleOCR
- [ ] Try MMOCR
- [ ] Re-tune rectification for the chosen engine
- [ ] Repeat validation choices on three different validation splits

---

## Up next

### Corner detection and layout (Oct 26 – Nov 8) [Makayla, Tomi]
- [ ] Hough-line quadrilateral candidates
- [ ] Score candidate quads instead of taking the largest
- [ ] Learned corner model trained on SRD corners
- [ ] Dewarping for curled paper
- [ ] Goal: close the oracle gap (SRD CER 0.245 → 0.178 with labeled corners)
- [ ] Parse two-row items (name on one line, price on the next)
- [ ] Train LayoutLMv3 on CORD's 800 training receipts
- [ ] Donut as a learned reference point

### Remaining digit errors
- [ ] Re-read each row's price box with a digits-only OCR pass
- [ ] Checksum-guided correction over alternative readings of low-confidence digits

### Parser and checksum fixes (from the demo)
- [ ] Payment-line filter in the layout parser, with fuzzy matching so misread labels (e.g. "CG") are caught (the regex parser has an exact keyword list only)
- [ ] Three-state checksum: pass / fail / could not check (now it returns nothing when the subtotal is missing)
- [ ] Derive a missing subtotal from the items, or from total − tax − tip

### Splitting interface (Nov 9–20) [Connor]
- [ ] Interface: shared items, editable fields
- [ ] User-set tip, replacing the fixed 18%: no tip by default when the receipt shows none, with any assumption shown on screen
- [ ] Split-error metric (how far each person's share is from the true split)
- [ ] Confidence flags in the interface (receipts that need a manual check)

---

## Deadlines
- [x] Oct 9: Proposal
- [ ] Oct 26: TA progress meeting (pipeline + VLM baseline, metrics and datasets, one change and why)
- [ ] Nov 20: Midterm report
- [ ] Nov 21 – Dec 6: Robustness fixes and live demo
- [ ] Dec 6: Final Project Day
- [ ] Dec 10: Final report, demo video and code release

## Housekeeping
- [ ] Commit the VLM baseline, labeling tools, results notebook and TODOS.md
- [ ] Add `google-genai` to `requirements.txt`
- [ ] Rotate the Gemini API key (it was printed in a terminal session)
- [ ] Update PRESENTATION.md, LOG.md and DECISIONS.md with the VLM baseline and labeling plan
