# DECISIONS

Decision and tradeoff report for the SplitSnap preliminary experiments. Entries were written at the moment each decision was made, and the **Evidence** lines were filled in once the corresponding run finished. Numbers quoted here come from files in `results/`.

---

## D01. Separate git repository inside the project folder
- Decision: run `git init` inside `SplitSnap/` instead of committing to the enclosing `/Users/tominekan/Code` repo.
- Why: the enclosing repo has no commits and hundreds of unrelated staged files. The task says to commit after each task, and also never to touch anything outside the project. A nested repo satisfies both.
- Tradeoffs: the parent repo will see `CV/SplitSnap` as an embedded repo if you later `git add` it there. You may want to push this repo on its own, or remove `SplitSnap/.git` and commit from the parent.
- Alternatives considered: (a) commit in the parent repo, which would sweep in unrelated staged files; (b) skip commits, which breaks the task instructions.
- How to test it: n/a (housekeeping).
- Evidence: `git -C .. status` was not modified; `git log` in this folder shows one commit per task.

## D02. Install Tesseract locally inside the project (micromamba, conda-forge)
- Decision: Tesseract 5.5.3 from conda-forge, installed into `tools/tess/` with a project-local micromamba. `src/common.py` points pytesseract at `tools/tess/bin/tesseract` when it exists, else falls back to a `tesseract` on PATH.
- Why: the task assumed Tesseract was already installed, but it was not. `brew install` writes to `/opt/homebrew`, outside the project folder, which the constraints forbid.
- Tradeoffs: ~250 MB in `tools/` (gitignored). Teammates who already have a system Tesseract can ignore it. Version may differ from a teammate's Homebrew Tesseract, which can shift CER slightly.
- Alternatives considered: (a) Homebrew (simplest, but writes outside the folder); (b) building from source (slow, same issue for leptonica); (c) skipping OCR (impossible).
- How to test it: run `python src/evaluate.py` on a machine with system Tesseract and compare `results/summary.csv`.
- Evidence: `tools/tess/bin/tesseract --version` prints `tesseract 5.5.3`.

## D03. Datasets and splits
- Decision: CORD v2 **test** (100) for reporting item-level metrics, CORD v2 **validation** (100) for developing parser rules by eye (never used for reported numbers); SRD split into a 30-image validation set (random, seed 0, among labelled images) for all tuning (preprocessing, Tesseract config) and a 168-image test set for reporting.
- Why: CORD is the only public set here with item-level labels; SRD is the only one with US restaurant photos and receipt corners. The task asked for a 30-image held-out SRD split. Using CORD validation for rule development keeps the CORD test numbers honest without spending any test images.
- Tradeoffs: 30 tuning images is small, so the chosen preprocessing could be noisy (a 1-2 point CER difference may not be significant). CORD's Indonesian receipts differ from US ones (currency format, language), so CORD item scores tell us more about the parser than about US restaurants.
- Alternatives considered: (a) k-fold cross-validation on SRD (more stable choice, 5x the OCR cost); (b) use the CORD train split (800) for tuning too, which only matters if we train something; (c) annotate a handful of our own US receipts with items, the best match to the real use case but no time tonight.
- How to test it: repeat the preprocessing selection with 3 different seeds for the SRD split and check whether the winner changes.
- Evidence: `results/logs/data.txt` (counts: CORD test 100 receipts / 268 priced items; SRD 198 images, 193 with text, val 30, test 168).

## D04. Image loading honours EXIF orientation
- Decision: load every image through `PIL.ImageOps.exif_transpose` before handing it to OpenCV.
- Why: SRD corner labels are in the EXIF-rotated frame (4 images otherwise have corners outside the image). Phones also store photos this way, so the app must do it anyway.
- Tradeoffs: none of note (a few ms per image).
- Alternatives considered: ignoring EXIF (OpenCV `imread` with `IMREAD_IGNORE_ORIENTATION`), which mislabels those images.
- How to test it: `python src/data.py` prints any corner outside its image; it prints none.
- Evidence: `results/logs/data.txt`.

## D05. CORD label conversion to our schema
- Decision: each `menu` entry with a `price` becomes an item `{name: nm, qty: first number in cnt (default 1), price: price}`; each priced `menu.sub` modifier (e.g. "COCONUT JELLY 4,000") becomes its own item; when `price` is missing we fall back to `itemsubtotal`; entries or modifiers with neither are dropped (19 of 287 menu+modifier entries in test, mostly unpriced modifiers such as "Less Ice", "+M", "=*LARGE*=="). `subtotal = sub_total.subtotal_price`, `tax = sub_total.tax_price`, `tip = sub_total.service_price` (Indonesian service charge plays the role of a tip in the bill split), `total = total.total_price`. Discounts, cash, change and card amounts are not in our schema and are ignored.
- Why: matches the output schema the app needs (line totals per item) and what is printed as a priced line, which is what an OCR-based parser can see.
- Tradeoffs: treating modifiers as separate items means a parser that merges a modifier into its parent line is penalized twice (one miss, one wrong price). Ignoring discounts means some ground-truth receipts fail our own checksum, so we also report the ground-truth checksum pass rate as a ceiling.
- Alternatives considered: (a) fold modifier prices into the parent item (closer to how a person splits a bill, but no longer a 1:1 match to printed lines); (b) keep CORD's full 30-class schema (more informative, but more than the app needs).
- How to test it: re-score with modifiers folded into parents and see whether parser rankings change.
- Evidence: `results/logs/data.txt`; ground-truth checksum pass rate in `results/summary.md`.

## D06. Number normalization depends on the currency, which is given per dataset
- Decision: `parse_amount(s, currency)`. For USD, the last `.` or `,` followed by exactly two digits is the decimal mark. For IDR, `.` and `,` are thousands separators and a trailing 2-digit decimal part is dropped (`"24,000.00"` -> 24000). The pipeline is told the currency (`IDR` for CORD, `USD` for SRD).
- Why: CORD writes 60 thousand rupiah as `60.000`; a US-style parser would read 60.0. In the app the locale is known from the phone, so passing it is realistic.
- Tradeoffs: a receipt in an unexpected format (European `12,50`) would be misread. Price regexes also differ per currency (USD requires 2 decimals, IDR requires 3-digit groups or 4+ digits), which makes the parser stricter, so it misses OCR outputs that dropped a separator.
- Alternatives considered: (a) auto-detecting the format per receipt from the majority pattern; (b) a learned field extractor (LayoutLMv3/Donut) which learns formats from data.
- How to test it: run the regex parser with auto-detection instead of the given currency on both datasets.
- Evidence: unit checks in `src/common.py` docstring; item price matches in `results/summary.md`.

## D07. Deriving SRD ground-truth totals from the transcription
- Decision: candidate lines contain `total`, `amount due`, `amt due` or `balance due` and end in a `d.dd` amount; lines with `sub total`, `tax`, `item`, `qty`, `discount`, `savings`, `tip amount` or `change` are excluded. The total is the largest candidate. It is marked unreliable (and the receipt skipped for total accuracy) when no candidate exists or the total is below the largest annotated subtotal.
- Why: SRD has no field labels. Taking the largest "total-like" amount handles receipts that print "Total" twice or print a pre-tip and post-tip total. The subtotal check rejects odd cases such as discounts or split checks.
- Tradeoffs: 171 of 193 labelled receipts get a total (22 skipped: mostly receipts with no printed total amount on the same line, or totals below subtotal). On receipts with a printed "total with tip", we would take that larger number as truth.
- Alternatives considered: (a) hand-labelling totals for all 193 (most accurate, about an hour of work); (b) last total-like line instead of largest (fails when tip suggestions or "total with tip" follow).
- How to test it: hand-check 20 random derived totals against the images; I spot-checked about 20 while writing the rule (e.g. 1001 -> 69.25, 1072 -> 21.69, 1150 -> 80.45, 1102 correctly rejected: subtotal 29.85 > total 16.24).
- Evidence: count printed by `python -c "from data import ..."` during development; the evaluation reports how many SRD test receipts had a usable total.

## D08. Traditional pipeline (classic CV + Tesseract) instead of an LLM/VLM
- Decision: every stage is hand-written classic CV or rules on top of Tesseract. No learned layout model, no LLM, no VLM, no paid API.
- Why: the course asks for a traditional baseline we can measure and explain, and every stage can be switched off as an ablation. It runs offline on a laptop (~0.2-1 s per image), costs nothing, and gives a floor that later learned methods must beat.
- Tradeoffs: much lower accuracy than modern document models on messy photos; rules are brittle across receipt formats (CORD vs US); lots of hand-tuned thresholds.
- Alternatives considered: (a) **Donut** (OCR-free encoder-decoder fine-tuned on CORD; reported ~90 tree-edit accuracy on CORD) would beat us by a wide margin on CORD but needs a GPU to fine-tune and generalizes poorly to US receipts without new labels; (b) **LayoutLMv3** on top of OCR boxes (token classification into menu.nm / price / total) would replace our hand rules in task 6 and needs labelled boxes (CORD has them); (c) a **VLM** (GPT-4o, Claude, Qwen2-VL) prompted for JSON would likely be the most accurate on US receipts with zero training but costs per call, needs network, and is hard to ablate. We would switch once the classic baseline is documented, starting with LayoutLMv3 as a drop-in for the parser.
- How to test it: run Donut (CORD fine-tuned checkpoint) and a VLM on the same CORD test / SRD test splits with our metrics.
- Evidence: B0 scores in `results/logs/score_B0.txt` (CORD item F1 0.213, SRD CER 0.393) show how far a naive traditional pipeline is from usable.

## D09. Tesseract configuration
- Decision: Tesseract 5.5.3, LSTM engine (default `--oem`), language `eng` for both datasets, one `image_to_data` call per image (word boxes + text rebuilt from Tesseract's block/paragraph/line ids). B0 uses the default page segmentation `--psm 3` (fully automatic). The psm is then treated as a tunable option in iteration 2 (D13).
- Why: psm 3 is what a naive user gets, which is the right baseline. One `image_to_data` call gives both the text (for regex/CER) and the boxes (for the layout parser) without running OCR twice.
- Tradeoffs: `eng` lacks Indonesian words, so some CORD item names are misread. psm 3 often splits a receipt into a name column and a price column as separate blocks, which destroys "name ... price" lines (visible on CORD validation: `Subtota]:` followed by no number).
- Alternatives considered: (a) `eng+ind` for CORD (better names, slower, and not the US use case); (b) `--psm 4` (single column of variable size text) or `--psm 6` (one uniform block), usually better for receipts; (c) a learned OCR (EasyOCR/PaddleOCR, task 8).
- How to test it: psm sweep on the SRD validation split (done in task 5, see D13); `eng` vs `eng+ind` on CORD validation.
- Evidence: `results/logs/score_B0.txt`; psm sweep in `results/preprocess_sweep.md`.

## D10. Regex parsing rules (baseline parser)
- Decision: per text line: (1) find a price at the end of the line (USD `d.dd` with optional `$` and thousands commas; IDR thousands-grouped or 4+ digit numbers), allowing a 1-2 letter tax flag after it; (2) if the line matches a keyword, fill that field, first occurrence wins, checked in the order subtotal (`sub total`, `subttl`, tolerant of `0` for `o`), tax (`tax`, `pb1`, `ppn`, `vat`, `pajak`), tip (`tip`, `gratuity`, `service`, `svc`), total (`total`, `jumlah`, `amount due`, `balance due`, `grand`); (3) skip payment/noise lines (`cash`, `change`, `card`, `visa`, `item`, `qty`, `%`, ...); (4) otherwise it is an item: optional leading 1-2 digit quantity (`2`, `2x`), the rest is the name, which must contain at least 2 letters. Rules were written up front and checked on CORD validation only.
- Why: this is the simplest rule set that implements the task's B0 definition, and it is easy to explain on a slide.
- Tradeoffs: depends entirely on Tesseract's line grouping; a name and its price split onto different lines is lost. "First occurrence wins" fails when a receipt prints a running subtotal twice (SRD 1001 prints "Sub Total" twice). Tip-suggestion lines are skipped via `%`, which also drops lines like "PB1 10%" unless they matched tax first (they do, since keywords are checked first).
- Alternatives considered: (a) layout-aware rows from word boxes (task 6); (b) a CRF/LayoutLMv3 token classifier trained on CORD train (learns field types instead of keywords).
- How to test it: compare regex vs layout parsing on identical OCR output (done in task 6/7).
- Evidence: CORD validation B0: item P 0.474 / R 0.116 (development only). Test numbers in `results/summary.md`.

## D11. Pipeline engineering: OCR cache and parallelism
- Decision: run images in 8 worker processes; cache each image's OCR output (text, boxes, stage timings) in `data/cache/ocr/` keyed by (dataset, image, rectify mode, preprocessing options, psm). Runtime per image = rectify + preprocess + OCR + parse time measured when the OCR stage actually ran (stored in the cache entry), so cached reruns report the original timings.
- Why: the ablation re-runs the same OCR many times (regex vs layout share OCR); caching makes `python src/evaluate.py` cheap to rerun. Timings stay honest because they are stored, not re-estimated.
- Tradeoffs: timings are measured with 8 processes in parallel, so per-image wall time is somewhat higher than single-process. Stale cache is possible if you change rectify/preprocess code without changing options: use `python src/evaluate.py --no-cache` (wipes `data/cache/ocr`).
- Alternatives considered: no cache (simplest, ~4x slower to rerun); caching only parser inputs in memory.
- How to test it: run `--no-cache` and compare the table.
- Evidence: `results/logs/pipeline_B0.txt` (298 receipts in 20 s).
