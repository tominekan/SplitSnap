# Data

Everything in `data/` except this file is gitignored. Recreate with `python src/data.py`.

## CORD v2 (`data/cord/`)
Source: Hugging Face `naver-clova-ix/cord-v2`. We save the `test` (100) and `validation` (100) splits as
`data/cord/<split>/<split>_NNN.png` + `<split>_NNN.json`. Each JSON holds `gt` (converted to our schema,
see DECISIONS.md D05), the original `gt_parse`, and `roi` (4 receipt corners TL, TR, BR, BL; present for
37 of 100 test images). Prices are Indonesian rupiah, written with `.` or `,` as thousands separators
(`"60.000"`, `"24,000"`, `"17500"`).

## SRD: labeled ExpressExpense restaurant receipts (`data/srd/`)
Source: Zenodo record 13688441, `Dataset.zip` (18,998,248 bytes). Cite: Auad, Alves, Kakizaki, Reis, Silva,
"A Filtering and Image Preparation Approach to Enhance OCR for Fiscal Receipts", SIBGRAPI 2024.

What the zip actually contains (inspected 2026-10-06):

| Content | Count | Format |
|---|---|---|
| `NNNN-receipt.jpg` (NNNN = 1000..1199) | 198 images | JPEG, mostly 750x1000, some small (250x250 to 562x1000) |
| `Labels/NNNN-receipt.txt` | 195 files | plain ASCII transcription |
| `homography.csv` | 199 rows (198 images + header) | `;`-separated corners |

- 193 images have both a text label and corners. Images without a text label: 1025, 1034, 1041, 1109, 1192.
  Text labels without an image: 1063, 1158. All 198 images have corners.
- **Text annotation**: a manual line-by-line transcription of the receipt, top to bottom, one printed line per
  text line, words separated by single spaces. Blank lines separate blocks; some lines contain only spaces
  (apparently placeholders for unreadable or redacted lines). There is no per-word box and no field-level
  (item/total) labelling, so totals have to be derived from the text (DECISIONS.md D07).
- **Corner annotation**: `homography.csv`, header `Image;P1;P2;P3;P4`, one row per image id. Each `Pk` is
  `x,y` in pixels. Order is **P1 = top-left, P2 = top-right, P3 = bottom-left, P4 = bottom-right** (note:
  not clockwise). Coordinates are in the **EXIF-rotated** image frame: 5 JPEGs carry an EXIF orientation
  tag (6 or 8), and for 4 of them the raw-pixel frame puts the corners outside the image. After
  `PIL.ImageOps.exif_transpose`, every corner lies inside its image. `src/data.py` reorders corners to
  TL, TR, BR, BL.

Example (`1001-receipt.jpg`, 645x1000 after EXIF):

```
homography.csv:  1001;83,77;545,90;72,824;576,788
                 -> TL (83,77), TR (545,90), BL (72,824), BR (576,788)

Labels/1001-receipt.txt (excerpt):
GRAND LUX CAFE
Roosevelt Field
...
Coffee 2.95
Chicken Parmesan 17.95
Prime Top Sirloin 25.95
The Bacon-Cheese burger 13.95
...
Sub Total: 63.75
tax: 5.50
Sub Total: 69.25
04/25 17:42 TOTAL: 69.25
```

Split: 30 randomly chosen labelled images (seed 0) form the **validation** split used for all tuning; the
other 168 images (163 with text) form the **test** split used for reporting.
