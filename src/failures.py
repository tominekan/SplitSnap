"""Failure analysis (run after src/evaluate.py).

    python src/failures.py   -> results/failures.md, results/failures_cord.json, figures/rectify_failures/

Cause guesses are produced by simple checks against the OCR text (is the true price / name present
anywhere in the OCR output?) and then reviewed by hand; manual notes in MANUAL_NOTES override them.
"""
import json
import re

import pandas as pd
from rapidfuzz import fuzz

from common import FIGURES, RESULTS, normalize_name
from data import load_cord, load_srd, srd_split
from metrics import match_items

N_CORD, N_SRD = 8, 4
# Hand-written notes after looking at the images (id -> cause); they override the automatic guess.
MANUAL_NOTES = {
    "test_059": "faint low-contrast thermal print: lines are found but digits are misread (78,000 -> 78006, 47,500 -> 47540)",
    "test_014": "small purple print: leading price digits dropped or misread (38,000 -> 3,000; 65,000 -> 43,000)",
    "test_061": "two-line item format (name row, then '1x 6500' price row) with discount rows in between; prices also misread ('6 S00')",
    "test_075": "very faded print: after Sauvola binarization almost no text survives, OCR returns ~nothing",
    "test_020": "faded print held in hand: prices split by a space or misread ('64 200', '4? 500'), so no price token matches",
    "test_045": "OCR reads the item names but drops the whole right-hand price column (crumpled paper, curved lines)",
    "test_004": "heavily crumpled paper with faint print: OCR output is mostly garbage",
    "test_024": "two-line item format with 'Rp.' prices on the row below the name, and 'Rp.56.000' misread as 'Ro esée000'",
    "srd_1169": "pink receipt is not low-saturation 'paper', so the paper mask misses it; the edge mask then locks onto the checkered tablecloth beside it",
    "srd_1104": "a glossy plastic sleeve and fold split the receipt; the paper mask finds only the header band above it",
    "srd_1176": "white receipt on a white napkin: no contrast to the background, the detected quad spans the napkin and is implausible, so we crop a region much larger than the receipt",
    "srd_1001": "crumpled receipt on a light grey table: the quad follows a table edge on the right, and the bounding-box crop clips the left side of the text",
}


def price_in_text(price, text):
    digits = re.sub(r"[^\d\n ]", "", text)
    return str(int(price)) in digits.replace(" ", "") or str(int(price)) in re.sub(r"\D", "", text)


def guess_cord_cause(gt, pred):
    text = pred["raw_text"] or ""
    if len(text.strip()) < 30:
        return "OCR returned almost no text"
    matched = {g for g, _, _ in match_items(pred["items"], gt["items"])}
    missed = [it for k, it in enumerate(gt["items"]) if k not in matched]
    if not missed:
        return "extra predicted items (non-item rows parsed as items)"
    causes = []
    for it in missed:
        p_ok = price_in_text(it["price"], text)
        n_ok = fuzz.partial_ratio(normalize_name(it["name"]), normalize_name(text)) >= 80
        if p_ok and n_ok:
            near = [p for p in pred["items"] if abs(p["price"] - it["price"]) < 0.5]
            causes.append("name misread (price found)" if near else "name and price not joined into one item row")
        elif n_ok:
            causes.append("price misread or missing in OCR")
        elif p_ok:
            causes.append("item name misread by OCR")
        else:
            causes.append("whole line misread or missed by OCR")
    top = max(set(causes), key=causes.count)
    return f"{top} ({causes.count(top)}/{len(missed)} missed items)"


def guess_rect_cause(row, gt_area_frac, pred_area_frac):
    if not row["detected"]:
        return "no 4-sided contour found, fell back to the full frame"
    if pred_area_frac > gt_area_frac * 1.3:
        return "detected region includes background (bright table, tray or plate merged with the receipt)"
    if pred_area_frac < gt_area_frac * 0.7:
        return "picked a sub-region of the receipt (fold, shadow edge or partly covered receipt)"
    return "quadrilateral of the right size but misplaced corners (curled or torn edge)"


def main():
    df = pd.read_csv(RESULTS / "summary.csv")
    ours = df[~df.config.str.startswith(("oracle", "EasyOCR"))]  # failure analysis is for the Tesseract pipeline
    best = ours.loc[ours.CORD_item_F1.idxmax(), "config"]
    per = pd.read_csv(RESULTS / "cord_per_receipt_f1.csv", index_col="id")[best].dropna()
    cord = {r["id"]: r for r in load_cord("test")}
    # ties (many receipts have F1 = 0) are broken by the number of true items: more items lost ranks worse
    n_items = {i: len(cord[i]["gt"]["items"]) for i in per.index}
    worst = sorted(per.index, key=lambda i: (per[i], -n_items[i], i))[:N_CORD]
    L = ["# Failure analysis", "",
         f"Best non-oracle configuration by CORD item F1: **{best}** (from `results/summary.csv`).",
         "Causes: automatic checks against the OCR text (`src/failures.py`), then reviewed by hand by looking at "
         "each image; the hand-written notes (MANUAL_NOTES in the script) replace the automatic guess.", "",
         "**Patterns from the review** (8 CORD + 4 SRD cases): (1) OCR digit errors on faint, small or coloured "
         "thermal print are the main CORD failure; one wrong digit makes an otherwise perfect item count as "
         "wrong. (2) Two-line item formats (name row, then a qty x unit-price row, sometimes with discount rows) "
         "are not handled by the row parser. (3) Crumpled or curved paper makes Tesseract drop whole columns. "
         "(4) Rectification fails when the paper is not white (pink receipt), when the background is also white "
         "(napkin), or when a sleeve or fold splits the receipt; the bounding-box crop can then clip text.", "",
         f"## {N_CORD} worst CORD test receipts by item F1", "",
         f"{int((per == 0).sum())} of {len(per)} CORD test receipts have item F1 = 0; ties are broken by the number "
         "of true items (more items lost ranks worse).", ""]
    out = []
    for rid in worst:
        gt = cord[rid]["gt"]
        pred = json.loads((RESULTS / "predictions" / best / "cord" / f"{rid}.json").read_text())
        cause = MANUAL_NOTES.get(rid, guess_cord_cause(gt, pred))
        out.append({"id": rid, "config": best, "f1": float(per[rid]), "cause": cause})
        L += [f"### {rid} (item F1 {per[rid]:.2f})", "", f"Likely cause: {cause}", "",
              "| true item | true price | predicted item | predicted price |", "|---|---|---|---|"]
        n = max(len(gt["items"]), len(pred["items"]))
        for k in range(n):
            g = gt["items"][k] if k < len(gt["items"]) else None
            p = pred["items"][k] if k < len(pred["items"]) else None
            L.append(f"| {g['name'] if g else ''} | {g['price']:,.0f} |" if g else "|  |  |")
            L[-1] += (f" {p['name']} | {p['price']:,.0f} |" if p else "  |  |")
        L += ["", f"True subtotal / tax / total: {gt['subtotal']} / {gt['tax']} / {gt['total']}; "
              f"predicted: {pred['subtotal']} / {pred['tax']} / {pred['total']}", ""]
    (RESULTS / "failures_cord.json").write_text(json.dumps(out, indent=1))

    # SRD rectification failures (test split)
    from rectify import draw_figure, full_quad, order_corners
    from data import load_image
    from shapely.geometry import Polygon
    srd = {r["id"]: r for r in load_srd()}
    test = {r["id"] for r in srd_split(list(srd.values()))[1]}
    rows = [r for r in json.loads((RESULTS / "rectify_per_image.json").read_text()) if r["id"] in test]
    rows = sorted(rows, key=lambda r: r["iou"])[:N_SRD]
    figdir = FIGURES / "rectify_failures"
    figdir.mkdir(parents=True, exist_ok=True)
    L += [f"## {N_SRD} worst SRD test rectifications by IoU", "",
          "| image | IoU | corner err | fell back | likely cause | figure |", "|---|---|---|---|---|---|"]
    from rectify import find_receipt
    for r in rows:
        rec = srd[r["id"]]
        img = load_image(rec["path"])
        area = img.shape[0] * img.shape[1]
        gt_frac = Polygon(order_corners(rec["corners"])).area / area
        q, _ = find_receipt(img)
        pred_frac = Polygon(q if q is not None else full_quad(img)).area / area
        cause = MANUAL_NOTES.get("srd_" + r["id"], guess_rect_cause(r, gt_frac, pred_frac))
        fig = figdir / f"srd_{r['id']}.png"
        draw_figure(rec, r, fig)
        L.append(f"| {r['id']} | {r['iou']:.2f} | {r['corner_err']:.3f} | {not r['detected']} | {cause} | "
                 f"`figures/rectify_failures/{fig.name}` |")
    (RESULTS / "failures.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
