"""Run every cumulative configuration and write the ablation table.

    python src/evaluate.py              # uses the OCR cache in data/cache/ocr when present
    python src/evaluate.py --no-cache   # recompute all OCR from scratch

Outputs: results/summary.csv, results/summary.md, results/cord_per_receipt_f1.csv,
results/predictions/<config>/..., and a copy of stdout in results/logs/evaluate.txt.

Reporting splits: CORD test (100, nothing was tuned on it); SRD test (168 images, 163 with text).
The 30 SRD validation images were used to choose rectification variant and preprocessing, so they
are excluded from every SRD number here.
"""
import argparse
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd

from common import CACHE, RESULTS
from data import load_cord, load_srd, srd_split, srd_total_from_text
from metrics import score_cord, score_srd
from pipeline import checksum, dataset_records, get_configs, run_config
from rectify import summarize


class Tee:
    def __init__(self, path):
        self.f, self.out = open(path, "w"), sys.stdout

    def write(self, s):
        self.f.write(s), self.out.write(s)

    def flush(self):
        self.f.flush(), self.out.flush()


def rectify_columns(cfg, srd_test):
    """Detection rate / IoU / corner error of the rectification used by this config on SRD test."""
    if cfg["rectify"] == "none":
        return {}
    if cfg["rectify"] == "oracle":
        return {"SRD_rect_detect": 1.0, "SRD_rect_IoU": 1.0, "SRD_rect_corner_err": 0.0}
    rows = [r for r in json.loads((RESULTS / "rectify_per_image.json").read_text()) if r["id"] in srd_test]
    s = summarize(rows)
    return {"SRD_rect_detect": s["detection_rate"], "SRD_rect_IoU": s["iou_mean_all"],
            "SRD_rect_corner_err": s["corner_err_mean_all"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--configs", nargs="*", help="subset of configuration names")
    a = ap.parse_args()
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(RESULTS / "logs" / "evaluate.txt")
    if a.no_cache and (CACHE / "ocr").exists():
        shutil.rmtree(CACHE / "ocr")
    if not (RESULTS / "rectify_per_image.json").exists():
        import rectify
        rectify.main()

    cord = load_cord("test")
    cord_gt = {r["id"]: r["gt"] for r in cord}
    srd = load_srd()
    _, srd_test_recs = srd_split(srd)
    srd_test = {r["id"] for r in srd_test_recs}
    texts = {r["id"]: r["text"] for r in srd_test_recs if r["text"]}
    totals = {i: srd_total_from_text(t) for i, t in texts.items()}
    recs = [r for r in dataset_records() if r["dataset"] == "cord" or r["id"] in srd_test]
    configs = get_configs()
    if a.configs:
        configs = {k: v for k, v in configs.items() if k in a.configs}
    print(f"Evaluating {len(configs)} configurations on CORD test ({len(cord)}) and SRD test "
          f"({len(srd_test)} images, {len(texts)} with text, {sum(v is not None for v in totals.values())} with a usable total)")
    print("Preprocessing / psm from results/preprocess_best.json:", json.dumps(get_configs()["+rectify+preprocess"]))

    rows, per_receipt = [], {}
    for name, cfg in configs.items():
        t = time.time()
        res = run_config(name, cfg, recs, workers=a.workers)
        cp = {i: res[("cord", i)] for i in cord_gt}
        sp = {i: res[("srd", i)] for i in srd_test}
        c, per = score_cord(cp, cord_gt)
        s = score_srd(sp, texts, totals)
        per_receipt[name] = per
        row = {"config": name, **{f"CORD_{k}": v for k, v in c.items()},
               "CORD_rect_fallback": float(np.mean([p["rectify_info"]["fallback"] for p in cp.values()]))
               if cfg["rectify"] != "none" else float("nan"),
               "CORD_runtime_s": float(np.mean([p["runtime_s"] for p in cp.values()])),
               **{f"SRD_{k}": v for k, v in s.items()},
               **rectify_columns(cfg, srd_test),
               "SRD_runtime_s": float(np.mean([p["runtime_s"] for p in sp.values()]))}
        rows.append(row)
        print(f"{name:34s} CORD F1 {c['item_F1']:.3f}  total {c['total_acc']:.3f}  | SRD CER {s['CER']:.3f} "
              f"total {s['total_acc']:.3f}  ({time.time() - t:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "summary.csv", index=False)
    pd.DataFrame(per_receipt).rename_axis("id").to_csv(RESULTS / "cord_per_receipt_f1.csv")
    gt_check = np.mean([checksum(g) is True for g in cord_gt.values()])
    write_markdown(df, gt_check, len(texts), sum(v is not None for v in totals.values()))
    print(f"Ground-truth checksum pass rate on CORD test (ceiling): {gt_check:.3f}")
    print("Wrote results/summary.csv and results/summary.md")


def write_markdown(df, gt_check, n_text, n_total):
    f = lambda v, d=3: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"
    L = ["# Ablation table", "",
         "Generated by `python src/evaluate.py` (log: `results/logs/evaluate.txt`). "
         "CORD = test split, 100 receipts, 268 priced items (nothing tuned on it). "
         f"SRD = test split, 168 images ({n_text} with text for CER, {n_total} with a usable derived total). "
         "Cumulative configurations; *oracle* replaces our rectification with the labelled corners "
         "(SRD corner labels; CORD `roi` where present, 37/100, else no warp).", "",
         "## CORD (item extraction)", "",
         "| config | item P | item R | item F1 | subtotal acc | tax acc | total acc | checksum pass | rect fallback | s/img |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.config} | {f(r.CORD_item_P)} | {f(r.CORD_item_R)} | **{f(r.CORD_item_F1)}** | "
                 f"{f(r.CORD_subtotal_acc)} | {f(r.CORD_tax_acc)} | {f(r.CORD_total_acc)} | {f(r.CORD_checksum_pass)} | "
                 f"{f(r.CORD_rect_fallback, 2)} | {f(r.CORD_runtime_s, 2)} |")
    r0 = df.iloc[0]
    L += ["", f"Field accuracy denominators: subtotal n={int(r0.CORD_subtotal_n)}, tax n={int(r0.CORD_tax_n)}, "
          f"total n={int(r0.CORD_total_n)} (receipts where the field exists). "
          f"Ground-truth labels themselves pass our checksum on {gt_check:.0%} of CORD test receipts "
          "(discounts, rounding and unlabelled charges), so that is the ceiling for the checksum column.", "",
          "## SRD (US restaurant photos)", "",
          "| config | CER (mean) | CER (median) | total acc | rect detection | rect IoU | rect corner err | s/img |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r.config} | **{f(r.SRD_CER)}** | {f(r.SRD_CER_median)} | {f(r.SRD_total_acc)} | "
                 f"{f(r.get('SRD_rect_detect'))} | {f(r.get('SRD_rect_IoU'))} | {f(r.get('SRD_rect_corner_err'))} | "
                 f"{f(r.SRD_runtime_s, 2)} |")
    L += ["", "CER is computed on lowercased text with whitespace collapsed and empty lines dropped "
          "(DECISIONS.md D16). Rectification corner error is normalized by the image diagonal. "
          "Runtimes are wall-clock per image with 8 images processed in parallel on an 8-core laptop."]
    (RESULTS / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
