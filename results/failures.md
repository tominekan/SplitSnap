# Failure analysis

Best non-oracle configuration by CORD item F1: **+rectify+preprocess+layout** (from `results/summary.csv`).
Causes: automatic checks against the OCR text (`src/failures.py`), then reviewed by hand by looking at each image; the hand-written notes (MANUAL_NOTES in the script) replace the automatic guess.

**Patterns from the review** (8 CORD + 4 SRD cases): (1) OCR digit errors on faint, small or coloured thermal print are the main CORD failure; one wrong digit makes an otherwise perfect item count as wrong. (2) Two-line item formats (name row, then a qty x unit-price row, sometimes with discount rows) are not handled by the row parser. (3) Crumpled or curved paper makes Tesseract drop whole columns. (4) Rectification fails when the paper is not white (pink receipt), when the background is also white (napkin), or when a sleeve or fold splits the receipt; the bounding-box crop can then clip text.

## 8 worst CORD test receipts by item F1

53 of 100 CORD test receipts have item F1 = 0; ties are broken by the number of true items (more items lost ranks worse).

### test_059 (item F1 0.00)

Likely cause: faint low-contrast thermal print: lines are found but digits are misread (78,000 -> 78006, 47,500 -> 47540)

| true item | true price | predicted item | predicted price |
|---|---|---|---|
| Tahu Isi | 78,000 | f Tahu Tsi yp | 78,006 |
| Tea | 30,000 | i Keniting 7-8.4ons Cnlily | 7,470 |
| Kepiting 7-8.4ons - Chilly 0.7 1 | 297,490 | Udang Telur Asin Reqular | 47,540 |
| Udang Telur Asin - Regular | 47,500 | LT hailen. kasg | 45,000 |
| Kailan 2 Rasa | 45,000 | Sweet | 16,000 |
| Mantau - 2 pcs | 25,000 |  |  |
| Sweet Tea | 16,000 |  |  |
| Nasi Putih | 42,000 |  |  |
| Mantau - 3 pcs | 37,500 |  |  |

True subtotal / tax / total: 618490.0 / 61849.0 / 680500.0; predicted: None / None / None

### test_014 (item F1 0.00)

Likely cause: small purple print: leading price digits dropped or misread (38,000 -> 3,000; 65,000 -> 43,000)

| true item | true price | predicted item | predicted price |
|---|---|---|---|
| RedVelvet Cheese Slc | 38,000 | RedVelvet Cheese | 3,000 |
| Chicken Baked Rice | 55,000 | Fish nm Chips | 43,000 |
| Fish n Chips | 65,000 | fushroom Soup | 33,000 |
| Mushroom Soup | 35,000 | Iced Tea | 9,000 |
| Chamomile | 25,000 | Coupon | 100,000 |
| Iced Tea | 20,000 |  |  |

True subtotal / tax / total: 238000.0 / 25585.0 / 281435.0; predicted: None / None / None

### test_061 (item F1 0.00)

Likely cause: two-line item format (name row, then '1x 6500' price row) with discount rows in between; prices also misread ('6 S00')

| true item | true price | predicted item | predicted price |
|---|---|---|---|
| TALAM UNGU | 6,500 | PLASTIK KECIL v | 5,000 |
| AREM - AREM | 8,000 |  |  |
| LEMPER | 18,000 |  |  |
| MIKA SEDANG | 0 |  |  |
| PLASTIK KECIL | 0 |  |  |
| TEH KOTAK | 5,000 |  |  |

True subtotal / tax / total: 24500.0 / None / 24500.0; predicted: None / None / None

### test_075 (item F1 0.00)

Likely cause: very faded print: after Sauvola binarization almost no text survives, OCR returns ~nothing

| true item | true price | predicted item | predicted price |
|---|---|---|---|
| CHOCO PUFF | 29,091 |  |  |
| CREAMY BEEF CLS FTC | 42,727 |  |  |
| NEW ORIENTAL CHK RICE | 34,545 |  |  |
| LIPTON PITCHER | 54,545 |  |  |
| SC/P SUPER SUPREME | 47,273 |  |  |
| CB/P BLACK PEPP BEEF | 48,182 |  |  |

True subtotal / tax / total: 256363.0 / 25637.0 / 282000.0; predicted: None / None / None

### test_020 (item F1 0.00)

Likely cause: faded print held in hand: prices split by a space or misread ('64 200', '4? 500'), so no price token matches

| true item | true price | predicted item | predicted price |
|---|---|---|---|
| cashew nuts chkn | 64,500 |  |  |
| garlic pepper beef | 79,500 |  |  |
| red curry beef | 69,500 |  |  |
| phad thai | 64,500 |  |  |
| steamed rice | 47,600 |  |  |

True subtotal / tax / total: 325600.0 / 34351.0 / 377859.0; predicted: None / None / None

### test_045 (item F1 0.00)

Likely cause: OCR reads the item names but drops the whole right-hand price column (crumpled paper, curved lines)

| true item | true price | predicted item | predicted price |
|---|---|---|---|
| Lemon Tea (L). | 25,000 |  |  |
| Mike & Ike Original | 28,000 |  |  |
| Mike & Berry | 28,000 |  |  |
| Roti Keju Manis. | 7,000 |  |  |

True subtotal / tax / total: None / None / 88000.0; predicted: None / None / None

### test_004 (item F1 0.00)

Likely cause: heavily crumpled paper with faint print: OCR output is mostly garbage

| true item | true price | predicted item | predicted price |
|---|---|---|---|
| ICE BLACKCOFFE | 82,000 |  |  |
| AVOCADO COFFEE | 61,000 |  |  |
| CHIKEN KATSU FF | 51,000 |  |  |

True subtotal / tax / total: 194000.0 / None / 174600.0; predicted: None / None / None

### test_024 (item F1 0.00)

Likely cause: two-line item format with 'Rp.' prices on the row below the name, and 'Rp.56.000' misread as 'Ro esée000'

| true item | true price | predicted item | predicted price |
|---|---|---|---|
| 5001-Corn Flakes Cookies | 56,000 |  |  |
| 5003-Chokoreto Cookies | 62,000 |  |  |
| 6002-Plastic Bag Medium | 0 |  |  |

True subtotal / tax / total: None / None / 118000.0; predicted: None / None / None

## 4 worst SRD test rectifications by IoU

| image | IoU | corner err | fell back | likely cause | figure |
|---|---|---|---|---|---|
| 1169 | 0.05 | 0.278 | False | pink receipt is not low-saturation 'paper', so the paper mask misses it; the edge mask then locks onto the checkered tablecloth beside it | `figures/rectify_failures/srd_1169.png` |
| 1104 | 0.28 | 0.307 | False | a glossy plastic sleeve and fold split the receipt; the paper mask finds only the header band above it | `figures/rectify_failures/srd_1104.png` |
| 1176 | 0.32 | 0.207 | False | white receipt on a white napkin: no contrast to the background, the detected quad spans the napkin and is implausible, so we crop a region much larger than the receipt | `figures/rectify_failures/srd_1176.png` |
| 1001 | 0.44 | 0.163 | False | crumpled receipt on a light grey table: the quad follows a table edge on the right, and the bounding-box crop clips the left side of the text | `figures/rectify_failures/srd_1001.png` |
