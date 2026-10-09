"""Development: score the layout parser on the VALIDATION splits only, for Tesseract and EasyOCR output.

    python src/dev/parser_dev.py                 # Tesseract only (fast)
    SPLITSNAP_EASYOCR=1 python src/dev/parser_dev.py --easyocr --workers 1

OCR comes from the shared cache (data/cache/ocr), so after the first run only parsing is recomputed.
Prints CORD validation item F1 / field accuracy and SRD validation total accuracy.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_cord, load_srd, srd_split, srd_total_from_text  # noqa: E402
from metrics import same_amount, score_cord  # noqa: E402
from parse_layout import parse_layout  # noqa: E402
from pipeline import best_preprocess, run_config  # noqa: E402


def val_records(srd_only=False):
    recs = [] if srd_only else [{"dataset": "cordval", "id": r["id"], "path": r["path"], "currency": "IDR",
             "quad": r["roi"].tolist() if r["roi"] is not None else None} for r in load_cord("validation")]
    val, _ = srd_split(load_srd())
    recs += [{"dataset": "srd", "id": r["id"], "path": r["path"], "currency": "USD",
              "quad": r["corners"].tolist() if r["corners"] is not None else None} for r in val]
    return recs


def score(name, cfg, recs, workers):
    res = run_config(name, cfg, recs, workers=workers, save=False)
    from pipeline import _ocr_stage
    cord = {r["id"]: r["gt"] for r in load_cord("validation")}
    val, _ = srd_split(load_srd())
    totals = {r["id"]: srd_total_from_text(r["text"]) for r in val}
    # re-parse from cached OCR so parser edits are picked up without re-running OCR
    preds = {}
    for r in recs:
        o = _ocr_stage(r, cfg["rectify"], cfg["pre"], cfg["psm"], cfg.get("engine", "tesseract"))
        preds[(r["dataset"], r["id"])] = parse_layout(o["words"], r["currency"], raw_text=o["text"])
    cord = {i: g for i, g in cord.items() if ("cordval", i) in preds}
    c, _ = score_cord({i: preds[("cordval", i)] for i in cord}, cord) if cord else ({k: float("nan") for k in
                      ["item_F1", "item_P", "item_R", "subtotal_acc", "tax_acc", "total_acc"]}, None)
    t = [same_amount(preds[("srd", i)]["total"], v) for i, v in totals.items() if v is not None]
    print(f"{name:34s} CORD val F1 {c['item_F1']:.3f} (P {c['item_P']:.3f} R {c['item_R']:.3f}) "
          f"sub {c['subtotal_acc']:.3f} tax {c['tax_acc']:.3f} total {c['total_acc']:.3f} | "
          f"SRD val total {sum(t) / len(t):.3f} ({sum(t)}/{len(t)})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--easyocr", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--srd-only", action="store_true", help="skip CORD validation (large photos, more memory)")
    a = ap.parse_args()
    pre, psm = best_preprocess()
    recs = val_records(a.srd_only)
    score("tesseract +rectify+preprocess+layout", dict(rectify="ours", pre=pre, psm=psm, parser="layout"),
          recs, a.workers)
    if a.easyocr:
        score("easyocr +rectify+layout", dict(rectify="ours", pre=None, psm=psm, parser="layout",
                                              engine="easyocr"), recs, 1)
