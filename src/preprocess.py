"""Iteration 2: image preprocessing before OCR (DECISIONS.md D13).

    python src/preprocess.py   # sweep options on the 30-image SRD validation split (after our rectification)

Options (applied in this order when enabled):
    denoise   non-local means on grayscale (h=10)
    upscale   resize so the median character height is ~30 px (estimated from connected components)
    clahe     contrast-limited adaptive histogram equalization (clip 2.0, 8x8 tiles)
    deskew    rotate by the angle in [-5, 5] deg that maximizes the row-profile variance
    binarize  'none' | 'adaptive' (OpenCV Gaussian, block 31, C 15) | 'sauvola' (window 31, k 0.2)
plus the Tesseract page segmentation mode (psm 3 / 4 / 6), which is swept alongside.
Every combination is scored by mean CER; the best combination is written to results/preprocess_best.json.
"""
import itertools
import json
import time

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from common import RESULTS

TARGET_CHAR_H = 30
MAX_SIDE = 4000


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def text_mask(gray):
    """Dark-text mask from a local (adaptive) threshold; robust to uneven lighting."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)


def char_height(gray):
    """Median height of character-sized connected components of the dark-text mask."""
    inv = text_mask(gray)
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    H, W = gray.shape
    h, w = stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_WIDTH]
    keep = (h >= 5) & (h <= 0.08 * H) & (w <= 0.1 * W) & (w >= 1)
    return float(np.median(h[keep])) if keep.sum() >= 10 else None


def upscale(gray):
    ch = char_height(gray)
    s = 1.0 if ch is None else float(np.clip(TARGET_CHAR_H / ch, 1.0, 3.5))
    s = min(s, MAX_SIDE / max(gray.shape))
    if s <= 1.01:
        return gray
    return cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def deskew_angle(gray):
    inv = text_mask(gray)
    s = 600 / inv.shape[1]
    small = cv2.resize(inv, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best, best_a = -1, 0.0
    for a in np.arange(-5, 5.01, 0.25):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rot = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_NEAREST)
        score = np.var(rot.sum(axis=1, dtype=np.float64))
        if score > best:
            best, best_a = score, a
    return best_a


def deskew(gray):
    a = deskew_angle(gray)
    if abs(a) < 0.25:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def binarize(gray, method):
    if method == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    if method == "sauvola":
        return ((gray > threshold_sauvola(gray, window_size=31, k=0.2)) * 255).astype(np.uint8)
    return gray


def preprocess(img, opts):
    g = to_gray(img)
    if opts.get("denoise"):
        g = cv2.fastNlMeansDenoising(g, None, h=10, templateWindowSize=7, searchWindowSize=21)
    if opts.get("upscale"):
        g = upscale(g)
    if opts.get("clahe"):
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    if opts.get("deskew"):
        g = deskew(g)
    return binarize(g, opts.get("binarize", "none"))


# ------------------------------------------------------------------ sweep

OPTION_GRID = {
    "upscale": [False, True],
    "clahe": [False, True],
    "denoise": [False, True],
    "deskew": [False, True],
    "binarize": ["none", "adaptive", "sauvola"],
}
PSMS = [3, 4, 6]


def all_combos():
    keys = list(OPTION_GRID)
    for vals in itertools.product(*OPTION_GRID.values()):
        yield dict(zip(keys, vals))


def describe(opts):
    on = [k for k in ["denoise", "upscale", "clahe", "deskew"] if opts[k]]
    if opts["binarize"] != "none":
        on.append(opts["binarize"])
    return "+".join(on) or "none"


def main():
    import pandas as pd
    from data import load_srd, srd_split
    from metrics import cer
    from pipeline import dataset_records, run_config

    val, _ = srd_split(load_srd())
    texts = {r["id"]: r["text"] for r in val}
    recs = dataset_records(srd_ids=set(texts))
    recs = [r for r in recs if r["dataset"] == "srd"]
    rows = []
    t0 = time.time()
    # psm alone, no preprocessing (rectified input)
    for psm in PSMS:
        res = run_config("sweep", dict(rectify="ours", pre=None, psm=psm, parser="regex"), recs, save=False)
        c = [cer(res[("srd", i)]["raw_text"], t) for i, t in texts.items()]
        rt = np.mean([res[("srd", i)]["runtime_s"] for i in texts])
        rows.append({"psm": psm, "options": "no preprocessing", "CER": np.mean(c), "runtime_s": rt, "opts": None})
        print(f"psm {psm} no preprocessing: CER {np.mean(c):.4f}", flush=True)
    for psm in PSMS:
        for opts in all_combos():
            res = run_config("sweep", dict(rectify="ours", pre=opts, psm=psm, parser="regex"), recs, save=False)
            c = [cer(res[("srd", i)]["raw_text"], t) for i, t in texts.items()]
            rt = np.mean([res[("srd", i)]["runtime_s"] for i in texts])
            rows.append({"psm": psm, "options": describe(opts), "CER": np.mean(c), "runtime_s": rt,
                         "opts": json.dumps(opts), **opts})
        print(f"psm {psm} grid done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "preprocess_sweep.csv", index=False)
    grid = df[df.opts.notna()].copy()
    best = grid.loc[grid.CER.idxmin()]
    best_opts = json.loads(best.opts)
    (RESULTS / "preprocess_best.json").write_text(json.dumps(
        {"options": best_opts, "psm": int(best.psm), "val_CER": float(best.CER),
         "description": best.options, "chosen_on": "SRD validation split (30 images), after our rectification"},
        indent=1))

    # Report: psm alone, each option alone (at the best psm), marginal effect across the grid, top 10.
    bp = int(best.psm)
    g = grid[grid.psm == bp]
    base = g[g.options == "none"].CER.iloc[0]
    lines = ["# Preprocessing sweep (SRD validation split, 30 images, after our rectification)", "",
             "CER = mean per-image character error rate (lower is better). Generated by `python src/preprocess.py`.", "",
             "## Page segmentation mode, no preprocessing", "", "| psm | CER | runtime s/img |", "|---|---|---|"]
    for _, r in df[df.opts.isna()].iterrows():
        lines.append(f"| {r.psm} | {r.CER:.4f} | {r.runtime_s:.2f} |")
    lines += ["", f"## Each option alone (psm {bp}, grayscale input)", "",
              "| option | CER | change vs grayscale only | runtime s/img |", "|---|---|---|---|",
              f"| grayscale only | {base:.4f} | - | {g[g.options == 'none'].runtime_s.iloc[0]:.2f} |"]
    for name in ["denoise", "upscale", "clahe", "deskew", "adaptive", "sauvola"]:
        r = g[g.options == name].iloc[0]
        lines.append(f"| {name} | {r.CER:.4f} | {r.CER - base:+.4f} | {r.runtime_s:.2f} |")
    lines += ["", f"## Marginal effect of each option (psm {bp}): mean CER of all grid combinations with vs without it", "",
              "| option | mean CER with | mean CER without | difference |", "|---|---|---|---|"]
    for k in ["denoise", "upscale", "clahe", "deskew"]:
        w, wo = g[g[k] == True].CER.mean(), g[g[k] == False].CER.mean()  # noqa: E712
        lines.append(f"| {k} | {w:.4f} | {wo:.4f} | {w - wo:+.4f} |")
    for b in ["adaptive", "sauvola"]:
        w, wo = g[g.binarize == b].CER.mean(), g[g.binarize == "none"].CER.mean()
        lines.append(f"| binarize={b} (vs none) | {w:.4f} | {wo:.4f} | {w - wo:+.4f} |")
    lines += ["", "## Top 10 combinations (all psm)", "", "| psm | options | CER | runtime s/img |", "|---|---|---|---|"]
    for _, r in grid.nsmallest(10, "CER").iterrows():
        lines.append(f"| {r.psm} | {r.options} | {r.CER:.4f} | {r.runtime_s:.2f} |")
    lines += ["", f"**Chosen:** psm {bp}, `{best.options}`, validation CER {best.CER:.4f}."]
    (RESULTS / "preprocess_sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
