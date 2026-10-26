# SRD field labeling guide (version 1)

Labels for the SRD receipts (`data/srd/`), one JSON file per receipt in `labels/srd/NNNN.json`.
They use the same schema as our CORD labels, so `src/metrics.py` scores both datasets the same way.
Label from the **photo**. The transcription and the pre-filled draft are only shortcuts. If they disagree
with the photo, the photo wins. Do not look at system predictions while labeling.

Run the app from the project root: `streamlit run src/labeling/app.py`

## Fields

| Field | Meaning |
|---|---|
| `items` | Purchased lines, each `{name, qty, price}`, in receipt order |
| `subtotal` | Printed pre-tax subtotal |
| `tax` | Printed tax (sum of all tax lines) |
| `tip` | Gratuity or service charge **included in the bill** |
| `total` | **Printed** amount due, before any handwritten tip |
| `tip_handwritten` | Tip written by hand, if any |
| `status` | `done`, or `flagged` when something prevents a complete label |
| `flags`, `notes` | Why a receipt is flagged, anything unusual |

All amounts are US dollars as plain numbers (`35.52`). Leave a field empty (`null`) if the receipt does not
print it. **Never guess or compute a value that is not printed.**

## Rules

**Items**
1. Every line with its own price is an item. `price` is the **line total** printed on that line:
   `2 Lunch 45.90` gives `qty 2, price 45.90`. `qty` is 1 if no quantity is shown.
2. Priced modifiers or add-ons (`+ Bacon 2.00`) are **separate items**. Unpriced modifiers
   (`no onions`, `medium rare`) are ignored.
3. Discounts and coupons are items with a **negative** price (`Happy Hour -3.00`).
4. Voided or cancelled items are left out.
5. Keep the printed name, including abbreviations (`CHKN PARM`). Fix obvious transcription typos only
   if the photo clearly shows the right spelling.

**Totals**
6. `subtotal` is the pre-tax subtotal. If a receipt prints two "Sub Total" lines (before and after tax,
   e.g. 1001), use the first, pre-tax one.
7. Several tax lines (state + city, `Tax 1` + `Tax 2`): add them into one `tax` value.
8. `total` is the printed total or amount due, before any handwritten tip. A handwritten tip goes in
   `tip_handwritten`. A handwritten final total is not labeled.
9. `tip` is only a gratuity or service charge that is part of the bill (`Gratuity 18% 12.00` added
   before the total). **Suggested-tip tables** (`18% = $6.30, 20% = $7.00`) are ignored.
10. Payment lines (cash, card, change, balance after payment) are never items and never totals.

**When something is wrong**
11. Part of the receipt is cut off, blurred or unreadable: fill what you can read, leave the rest empty,
    set `status: flagged` and pick a flag (`cropped`, `illegible`, ...).
12. Several receipts in one photo, or a photo that is not a restaurant receipt: flag it and leave the
    fields empty.
13. Items that do not add up to the subtotal (or subtotal + tax + tip not equal to total) are fine
    **if that is what is printed**. The app's arithmetic check is a typo detector, not a rule.

## Changing the rules
If a new case needs a new rule, add it here and bump the version at the top. New labels record the
`guide_version` they were made under, so older labels affected by the change can be found and rechecked.
