# Task: SplitSnap preliminary experiments (CIS 5810 final project)

## Context
SplitSnap is a computer vision project: a user photographs a restaurant receipt, we extract line items (name, quantity, price), subtotal, tax and total, and split the bill. Tomorrow at 3 pm we present preliminary experiments to our professor. I need a traditional (no LLM, no API) baseline, three measured classic-CV iterations on top of it, honest metrics, figures for slides, and a written record of every design decision and its tradeoffs. Everything runs on two public datasets. I will review everything you do, so keep it simple, readable and well logged.

I am asleep while you run. Do not stop to ask me questions: make the most reasonable choice, record it in DECISIONS.md, and keep going.

## Constraints
- NEVER delete, move, or overwrite any file or folder outside this project folder, under any circumstances. Do not run `rm`, `mv`, or any destructive command on a path outside this folder, and never use `sudo`, `~`, `/`, or `..` paths in a delete. Only create, edit, or delete files inside this project folder. If a task seems to need changes outside the folder, skip it and note it in LOG.md instead.
- No LLMs, VLMs or paid APIs anywhere in the pipeline. Classic CV (OpenCV, scikit-image) and Tesseract (pytesseract) only, except the optional task 8.
- Python 3.10+, plain scripts plus one demo notebook. Keep dependencies minimal and list them in requirements.txt. Tesseract is already installed on the system.
- Add data/ downloads to .gitignore.
- Never invent or estimate results. Every number in results/ must come from a script run that is logged.
- If something fails (download, dependency), stop that piece, write the error in LOG.md, and move on to the next task.
- Commit after each task with a clear message.
- Every time you make a design choice, add an entry to DECISIONS.md right then, not at the end.

## Datasets
- CORD v2 (Hugging Face: naver-clova-ix/cord-v2): photographed restaurant receipts with full line-item labels. Use the test split (100 receipts). Prices are Indonesian rupiah without decimals, so normalize number strings carefully.
- Labeled ExpressExpense restaurant receipts (Zenodo record 13688441, https://zenodo.org/records/13688441, file Dataset.zip, about 19 MB): 200 hand-photographed US restaurant receipts with manual text annotations and four-corner annotations of each receipt. Cite: Auad, Alves, Kakizaki, Reis, Silva, "A Filtering and Image Preparation Approach to Enhance OCR for Fiscal Receipts", SIBGRAPI 2024. We call this dataset SRD below.

## Pipeline design
Build the pipeline as separate stages that can be switched on and off, so each iteration is a measured ablation:

    image -> [rectify] -> [preprocess] -> OCR (Tesseract) -> parse (regex OR layout-aware) -> checksum -> JSON

Output schema, one JSON per receipt:

    {"items": [{"name": str, "qty": number, "price": number}], "subtotal": number|null, "tax": number|null, "tip": number|null, "total": number|null, "raw_text": str}

Prices are line totals.

## Tasks (in order)
1. Repo skeleton: README.md (how to run each step), requirements.txt, .gitignore, folders src/, data/, results/, figures/, notebooks/. Create LOG.md and DECISIONS.md.
2. Data (src/data.py):
   - Download both datasets. For SRD, inspect the files first and write data/README.md describing exactly what the text and corner annotations contain and their format, with one example.
   - Convert CORD labels to our schema.
   - Load SRD corner labels as 4 (x, y) points per image, and the text annotation per image.
   - If SRD fails to download or parse, log it and continue with CORD only.
3. Baseline (B0): Tesseract on the raw image, then regex parsing (a line ending in a price is an item; lines containing subtotal/tax/total keywords fill those fields). Run on all CORD test images and all SRD images.
4. Iteration 1, rectification (src/rectify.py): find the receipt with OpenCV (grayscale, blur, edges or adaptive threshold, largest 4-point contour), apply a perspective warp, and fall back to the original image if no good quadrilateral is found (log which images fell back).
   - Score it on SRD against the labeled corners: detection rate, IoU between predicted and labeled quadrilaterals, and mean corner error normalized by image diagonal.
   - Also warp SRD images using the labeled corners ("oracle rectification") as an upper bound.
   - Save 6 SRD before/after figures to figures/rectify/ with labeled and predicted corners drawn on the original.
5. Iteration 2, preprocessing (src/preprocess.py): try a small set of options (e.g. upscaling to a target text height, CLAHE contrast, adaptive or Sauvola binarization, denoising, deskew) and keep the best combination by validation CER on a held-out 30-image SRD split. Report each option's effect, not just the winner.
6. Iteration 3, layout-aware parsing (src/parse_layout.py): use Tesseract word boxes (image_to_data), cluster words into rows by vertical position, take the right-most number in a row as its price, parse a leading quantity, and use keyword rows for subtotal, tax and total. Compare against regex parsing on the same OCR output.
7. Evaluation (src/evaluate.py), for every cumulative configuration (B0, +rectify, +rectify+preprocess, +rectify+preprocess+layout, and oracle rectify in place of ours):
   - CORD: item-level precision, recall, F1 (an item matches if the normalized name has fuzzy similarity >= 0.8 and the normalized price matches exactly); exact-match accuracy for subtotal, tax, total; checksum pass rate (items sum to subtotal, subtotal + tax (+ tip) = total, within 0.5% of total).
   - SRD: character error rate of the OCR text against the annotated text; total-amount accuracy if the total can be reliably found in the annotations (explain how in DECISIONS.md); the rectification metrics from task 4.
   - Both: mean runtime per image.
   - Report on the held-out test portion only where a stage was tuned. Write results/summary.csv and results/summary.md as an ablation table (rows = configurations, columns = metrics).
8. Optional, only if everything above is done: swap Tesseract for one learned OCR engine (EasyOCR or PaddleOCR; these are OCR models, not LLMs) in the best configuration and add it as one more row. Skip it if installation is painful, and say so in LOG.md.
9. Failure analysis: write results/failures.md listing the 8 worst CORD receipts by item F1 for the best configuration, with predicted vs true items side by side and a one-line guess at the cause (e.g. merged lines, missed modifier, misread digit), plus the 4 worst SRD rectification failures with a guess at the cause.
10. Figures for slides in figures/: an ablation bar chart of CORD item F1 per configuration, CER on SRD per configuration including oracle, a pipeline diagram (simple boxes and arrows with matplotlib is fine), a preprocessing before/after example, and 3 annotated failure examples.
11. notebooks/demo.ipynb: load one SRD photo, show original, rectified, preprocessed, OCR boxes grouped into rows, the parsed JSON, and a toy bill split among 3 people with tax and tip split proportionally.
12. LOG.md: what you did per task, commands run, anything that failed or that you were unsure about. Also a short section "AI usage" describing which files you generated (our course requires disclosing this).
13. DECISIONS.md, the decision and tradeoff report. One entry per design decision, covering at least: dataset choices and splits, label schema and conversion, choosing a traditional pipeline over an LLM/VLM, Tesseract and its config (page segmentation mode, language), the regex rules, the rectification method and its fallback, each preprocessing option, the row-grouping method, the item matching rule and its thresholds, the checksum tolerance, how SRD totals were derived, the choice of metrics, and anything else you decided along the way. Each entry uses this format:
    - Decision: what we chose
    - Why: the reason, tied to our goal and the time we had
    - Tradeoffs: what we gave up (accuracy, speed, complexity, robustness, dev time, reproducibility)
    - Alternatives considered: 2-3 other options (including learned approaches like Donut, LayoutLMv3 or a VLM where relevant), what each would do better or worse, and when we would switch to it
    - How to test it: the experiment that would tell us if a different choice is better (this becomes our iteration plan)
    - Evidence: any result from this run that supports or questions the choice, with the file it came from

    End with a section "Top 5 iterations to try next", ranked by expected impact versus effort, each pointing back to the decision it revisits. Use the gap between our rectification and oracle rectification as one input.

## Definition of done
I can run `python src/evaluate.py` and get the ablation table, every figure exists, LOG.md honestly says what works and what does not, and DECISIONS.md has an entry for every decision above. When finished, print a short summary of results and the top 3 things I should check by hand.
