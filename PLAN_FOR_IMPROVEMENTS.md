# SplitSnap: plan for improvements

This plan builds on the preliminary experiments (`results/summary.md`, `results/failures.md`, `DECISIONS.md`). Every step says what to change, why the evidence points there, how to measure it, and when to stop.

## Where we are

| | CORD item F1 | SRD CER | SRD total acc |
|---|---|---|---|
| Best pipeline (rectify + preprocess + layout) | 0.451 | 0.245 | 0.514 |
| Same with oracle corners | 0.457 | 0.178 | 0.527 |

Where the errors come from, in rough order of size:
1. **OCR misreads digits.** Faint, small or coloured thermal print (`38,000` -> `3,000`, `47,500` -> `47540`). One wrong digit turns a correct item into a miss *and* a false positive. This is the main reason 53/100 CORD receipts score F1 = 0.
2. **Rectification is not trusted.** Our corners are often wrong enough that a perspective warp shears the text, so 73% of SRD images only get a crop. Oracle corners cut SRD CER by 27% (0.245 -> 0.178).
3. **Parser grammar gaps.** Items that span two rows (name, then `qty x unit price   line price`), discount rows, and payment lines mistaken for items ("CG 4.97" is change given).
4. **Field selection.** "First occurrence wins" picks the wrong subtotal/total when they are printed twice; totals are right only ~30% (CORD) to ~51% (SRD) of the time.

The goal is a receipt the user only has to *confirm*, not retype. A realistic next target is **CORD item F1 >= 0.65, SRD total accuracy >= 0.75**, and a checksum that reliably flags receipts needing a manual look.

---

## Phase 0: make measurements trustworthy

Do this first, because several conclusions so far rest on 30 validation images.

- **Stable validation.** Re-run the preprocessing selection with 3 split seeds (or 5-fold CV on SRD) and keep only choices that win consistently. The winner already changed once between two runs (DECISIONS D13).
- **Error taxonomy.** Hand-label the cause for 30 random F1 = 0 CORD receipts and 30 SRD receipts (OCR digit error / missed line / two-row item / wrong field / rectification). Count them. Re-rank the phases below by these counts, not by the 12 anecdotes in `failures.md`.
- **Diagnostic metrics.** Add name-only and price-only item matching, and an order-free bag-of-words F1 next to CER. These tell us whether a change fixes recognition or reading order.
- **Hand-check SRD totals.** Verify the derived totals (D07) on ~20 images, since total accuracy is our main SRD field metric.

## Phase 1: cheap rule fixes (1-2 days, expected +0.05 to +0.10 CORD F1)

All in `src/parse_layout.py`, `src/parse_regex.py` and `src/common.py`; develop on CORD validation only.

1. **Price-token repair.** Join thousands groups split by a space (`64 200`), and inside *price-shaped* tokens map common confusions (`O`/`o` -> 0, `S` -> 5, `l`/`I` -> 1, `B` -> 8, `?` -> 7). Restrict this to the right-hand column so names are untouched.
2. **Two-row item grammar.** Recognize "name row + `N x unit_price  line_price` row", attach discount rows to the preceding item, and use `qty x unit_price = line_price` as a consistency check when both are present.
3. **Better field selection.** Replace "first occurrence wins" with rules that use position and arithmetic: total = the largest total-like amount below the items, subtotal must be at most total, and prefer a candidate set where subtotal + tax (+ service) = total.
4. **Stricter non-item filter.** Treat anything below the total line (cash, change, card, abbreviations like `CG`) as payment, not items.

Measure: CORD validation F1 and field accuracy, then a single test run at the end of the phase.

## Phase 2: fix the OCR bottleneck (3-5 days, expected the largest single gain)

1. **Re-OCR the price column (classic, cheap).** After row grouping, crop each row's right-most token box (with padding), upscale it, and re-run Tesseract with `--psm 7` and a digit whitelist (`0123456789.,$Rp`). Keep the re-read when it parses as a valid price. This attacks failure #1 directly with no new dependency.
2. **Adaptive preprocessing fallback.** If the binarized image yields very little text or no price tokens (test_075, test_020), re-run OCR on the grayscale (non-binarized) image and keep the output with more valid prices. Generalize to "try 2-3 preprocessing variants and keep the one whose parse passes the checksum".
3. **Learned OCR, measured properly.** EasyOCR is already integrated: run `SPLITSNAP_EASYOCR=1 python src/evaluate.py --workers 1` (one model in memory, ~1 GB). Also try PaddleOCR (PP-OCR is usually stronger on receipts). Compare with and without binarization, since learned models expect natural images. Switch engines only if CORD F1 *and* SRD CER both improve at an acceptable runtime (target < 3 s/image on a laptop).
4. **Hybrid.** Use the learned detector for text boxes and keep whichever recognizer is better on digits, if the two disagree systematically.

## Phase 3: rectification we can trust (3-5 days, up to -0.067 SRD CER available)

The oracle gap is the measured ceiling for this phase.

1. **Hough-line quadrilaterals.** Fit the 4 dominant paper edges and intersect them. This handles receipts running off the frame, which produce most of the bad corners today.
2. **Score candidates instead of taking the first.** Generate quads from all masks and methods, score each one by edge strength along its sides, rectangularity and text alignment after warping, and keep the best.
3. **Learned corner model.** A small CNN corner regressor or segmentation net (a U-Net or DocAligner-style model) trained on SRD corners (use the 168 test images only for evaluation, train with augmentation on the rest, plus CORD `roi` for 37+ images and CORD train if it has roi). This is the step most likely to close the oracle gap.
4. **Curl and fold.** For crumpled receipts, dewarp text lines (fit a smooth baseline per row and remap), or simply group rows with a skew-tolerant method so a little curl stops breaking rows.
5. **Colour-agnostic paper mask.** The pink receipt (SRD 1169) fails because the mask assumes white paper; use local contrast or texture instead of low saturation.

Success: raw detector IoU >= 0.9 on 75% of SRD test images, warps applied to most images, and SRD CER within 0.02 of the oracle.

## Phase 4: learned parsing (1-2 weeks)

When the rules in Phase 1 stop paying off:

1. **LayoutLMv3 (or LiLT) token classification** on our OCR boxes, trained on CORD train (800 receipts with word-level labels: menu name, count, price, subtotal, tax, total). It replaces hand-written keyword and row rules, and handles two-row items and modifiers naturally. Evaluate on identical OCR against the layout parser.
2. **Donut** (OCR-free, CORD-fine-tuned checkpoint) as a reference point. It is strong on CORD, but we should check how it transfers to US receipts.
3. **A VLM upper bound** (one-off, not in the product pipeline): run a vision-language model on SRD test to know how far the classic and learned pipelines are from the state of the art, which helps decide where to stop.

## Phase 5: product-level reliability (ongoing)

1. **US receipts with item labels.** CORD is Indonesian; SRD has no item labels. Hand-label items/subtotal/tax/tip/total for ~100 SRD receipts (or our own photos). Without this, we cannot measure item F1 on the actual use case.
2. **Confidence and user-in-the-loop.** Use the checksum plus OCR confidences to flag items and fields that need confirmation, and design the UI so a correction takes one tap. Measure "receipts needing at most one correction".
3. **Checksum-guided correction.** Given the arithmetic identities, search over alternative readings for low-confidence digits and pick the combination that makes the receipt add up.
4. **Speed.** Preprocessing on large CORD images costs ~5 s/image; downscale before denoising, and profile before moving to a phone.

---
