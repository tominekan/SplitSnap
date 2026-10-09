"""Slide figures (run after src/evaluate.py and src/failures.py).

    python src/figures.py

figures/ablation_cord_f1.png    CORD item F1 per configuration
figures/srd_cer.png             SRD CER per configuration (incl. oracle rectification)
figures/pipeline.png            pipeline diagram
figures/preprocess_example.png  rectified vs preprocessed crop
figures/failure_*.png           3 annotated CORD failure examples (best configuration)
"""
import json

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402

BLUE, GRAY, ORANGE, INK, MUTED = "#2a78d6", "#8d8c87", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c8c7c2", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})


def best_config(df):
    ours = df[~df.config.str.startswith(("oracle", "EasyOCR"))]
    return ours.loc[ours.CORD_item_F1.idxmax(), "config"]


def bar(df, col, title, ylabel, path, lower_better=False):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    labels = [c.replace("EasyOCR: ", "EasyOCR").replace("+", "\n+").replace(" (", "\n(") for c in df.config]
    colors = [GRAY if c.startswith("oracle") else ORANGE if c.startswith("EasyOCR") else BLUE for c in df.config]
    x = np.arange(len(df))
    ax.bar(x, df[col], width=0.6, color=colors, edgecolor="white", linewidth=2)
    for xi, v in zip(x, df[col]):
        ax.text(xi, v + df[col].max() * 0.015, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, df[col].max() * 1.15)
    ax.yaxis.grid(True, color="#ecebe8"); ax.set_axisbelow(True)
    ax.set_title(title + ("  (lower is better)" if lower_better else "  (higher is better)"),
                 loc="left", color=INK, fontsize=12, pad=22)
    ax.text(0.0, 1.02, "blue = our pipeline (Tesseract), gray = oracle rectification (labelled corners), "
            "orange = EasyOCR in place of Tesseract",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=MUTED)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def pipeline_diagram(path):
    stages = [("photo", "input"), ("rectify", "iteration 1\ncontour + warp"),
              ("preprocess", "iteration 2\nupscale / CLAHE /\nbinarize ..."),
              ("OCR", "Tesseract 5\n(psm tuned)"), ("parse", "regex (B0) or\nlayout rows (it. 3)"),
              ("checksum", "items = subtotal\nsub + tax + tip = total"), ("JSON", "items, subtotal,\ntax, tip, total")]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.set_xlim(0, len(stages) * 1.9); ax.set_ylim(0, 2.2); ax.axis("off")
    for i, (name, sub) in enumerate(stages):
        x = i * 1.9 + 0.1
        optional = name in ("rectify", "preprocess")
        ax.add_patch(FancyBboxPatch((x, 0.9), 1.5, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#eaf2fc" if name not in ("photo", "JSON") else "#f2f1ee",
                                    ec=BLUE if name not in ("photo", "JSON") else GRAY,
                                    ls="--" if optional else "-", lw=1.6))
        ax.text(x + 0.75, 1.35, name, ha="center", va="center", fontsize=12, color=INK, weight="bold")
        ax.text(x + 0.75, 0.45, sub, ha="center", va="center", fontsize=8.5, color=MUTED)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 1.82, 1.35), xytext=(x + 1.52, 1.35),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(0.1, 2.05, "SplitSnap traditional pipeline (dashed = stage can be switched off for the ablation)",
            fontsize=10, color=MUTED)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def preprocess_example(path, srd_id="1004"):
    from data import load_image, load_srd
    from preprocess import preprocess, to_gray
    from rectify import rectify_image
    best = json.loads((RESULTS / "preprocess_best.json").read_text())
    rec = [r for r in load_srd() if r["id"] == srd_id][0]
    raw = load_image(rec["path"])
    rect, _ = rectify_image(raw)
    pre = preprocess(rect, best["options"])
    g = to_gray(rect)
    crop = lambda im: im[int(im.shape[0] * 0.30): int(im.shape[0] * 0.55)]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [0.7, 1, 1]})
    ax[0].imshow(raw[..., ::-1]); ax[0].set_title("raw photo", fontsize=10)
    ax[1].imshow(crop(g), cmap="gray"); ax[1].set_title("rectified (crop)", fontsize=10)
    ax[2].imshow(crop(pre), cmap="gray"); ax[2].set_title(f"preprocessed: {best['description']} (crop)", fontsize=10)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"SRD {srd_id}: preprocessing chosen on the validation split (psm {best['psm']})",
                 fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def failure_example(cfg_name, rid, path, cause):
    from data import load_cord, load_image
    from parse_layout import group_rows
    from pipeline import get_configs, _ocr_stage
    from rectify import warp, order_corners
    cfg = get_configs()[cfg_name]
    rec = [r for r in load_cord("test") if r["id"] == rid][0]
    o = _ocr_stage({"dataset": "cord", "id": rid, "path": rec["path"],
                    "quad": rec["roi"].tolist() if rec["roi"] is not None else None},
                   cfg["rectify"], cfg["pre"], cfg["psm"])
    img = load_image(rec["path"])
    q = o["rectify_info"]["quad"]
    if q is not None:
        img = warp(img, order_corners(q))
    if cfg["pre"]:
        from preprocess import preprocess
        img = preprocess(img, cfg["pre"])
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pred = json.loads((RESULTS / "predictions" / cfg_name / "cord" / f"{rid}.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1, 1.1]})
    ax[0].imshow(img[..., ::-1])
    for k, row in enumerate(group_rows(o["words"])):
        for w in row:
            ax[0].add_patch(Rectangle((w["left"], w["top"]), w["width"], w["height"], fill=False,
                                      ec=[BLUE, "#eb6834"][k % 2], lw=1))
    ax[0].set_title(f"CORD {rid}: OCR word boxes, colour alternates per grouped row", fontsize=9)
    ax[0].axis("off")
    gt = rec["gt"]["items"]
    fmt = lambda it: f"{it['qty']} x {it['name'][:28]:28s} {it['price']:>10,.0f}"
    txt = "TRUE ITEMS\n" + "\n".join(fmt(i) for i in gt) + \
          f"\nsubtotal {rec['gt']['subtotal']}  tax {rec['gt']['tax']}  total {rec['gt']['total']}" + \
          "\n\nPREDICTED ITEMS\n" + ("\n".join(fmt(i) for i in pred["items"]) or "(none)") + \
          f"\nsubtotal {pred['subtotal']}  tax {pred['tax']}  total {pred['total']}" + \
          f"\n\nLikely cause: {cause}"
    ax[1].text(0, 1, txt, va="top", ha="left", family="monospace", fontsize=8.5, color=INK, wrap=True)
    ax[1].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    df = pd.read_csv(RESULTS / "summary.csv")
    FIGURES.mkdir(exist_ok=True)
    bar(df, "CORD_item_F1", "CORD test: item F1 per configuration", "item F1", FIGURES / "ablation_cord_f1.png")
    bar(df, "SRD_CER", "SRD test: OCR character error rate per configuration", "mean CER",
        FIGURES / "srd_cer.png", lower_better=True)
    pipeline_diagram(FIGURES / "pipeline.png")
    preprocess_example(FIGURES / "preprocess_example.png")
    fails = json.loads((RESULTS / "failures_cord.json").read_text())
    for k, f in enumerate(fails[:3], 1):
        failure_example(f["config"], f["id"], FIGURES / f"failure_{k}_{f['id']}.png", f["cause"])
    print("figures:", sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
