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
# Hand-written notes after looking at the images (id -> cause). Filled in during review.
MANUAL_NOTES = {}


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
    ours = df[~df.config.str.startswith("oracle")]
    best = ours.loc[ours.CORD_item_F1.idxmax(), "config"]
    per = pd.read_csv(RESULTS / "cord_per_receipt_f1.csv", index_col="id")[best].dropna()
    cord = {r["id"]: r for r in load_cord("test")}
    worst = per.sort_values(kind="stable").index[:N_CORD]
    L = ["# Failure analysis", "",
         f"Best non-oracle configuration by CORD item F1: **{best}** (from `results/summary.csv`).",
         "Cause guesses come from automatic checks against the OCR text (`src/failures.py`), reviewed by hand.", "",
         f"## {N_CORD} worst CORD test receipts by item F1", ""]
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
    from rectify import detect_quad
    for r in rows:
        rec = srd[r["id"]]
        img = load_image(rec["path"])
        area = img.shape[0] * img.shape[1]
        gt_frac = Polygon(order_corners(rec["corners"])).area / area
        q, _ = detect_quad(img)
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
