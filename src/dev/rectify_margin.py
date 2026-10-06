"""Development: does expanding the detected quad outward, and rejecting implausible (shearing) quads, help OCR? CORD validation (item F1, layout parser)
and SRD validation (CER), psm 6, no preprocessing, no cache.
    python src/dev/rectify_margin.py  -> results/logs/rectify_margin_val.txt"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from data import load_cord, load_image, load_srd, srd_split
from metrics import cer, score_cord
from ocr import run_tesseract
from parse_layout import parse_layout
from rectify import detect_quad, expand_quad, plausible_quad, warp


def run(args):
    path, margin, check = args[:3]
    crop = len(args) > 3 and args[3]
    img = load_image(path)
    if margin is not None:
        q, _ = detect_quad(img)
        if q is not None and (not check or plausible_quad(q, *check)):
            img = warp(img, expand_quad(q, margin, img.shape))
        elif q is not None and crop:  # implausible: axis-aligned crop to the expanded quad, no warp
            e = expand_quad(q, margin, img.shape)
            x0, y0 = e.min(0).astype(int)
            x1, y1 = e.max(0).astype(int)
            img = img[y0:y1 + 1, x0:x1 + 1]
    return run_tesseract(img, psm=6)


if __name__ == "__main__":
    cord = load_cord("validation")
    val, _ = srd_split(load_srd())
    with ProcessPoolExecutor(8) as ex:
        variants = [(None, None), (0.0, None), (0.03, None), (0.06, None), (0.10, None),
                    (0.0, (80, 0.9)), (0.06, (80, 0.9)), (0.06, (85, 0.9)), (0.06, (75, 0.8)),
                    (0.06, (85, 0.9), True), (0.06, (80, 0.9), True)]
        for v in variants:
            margin, check, crop = (v + (False,))[:3]
            o = list(ex.map(run, [(r["path"], margin, check, crop) for r in cord]))
            s, _ = score_cord({r["id"]: parse_layout(w, "IDR", t) for r, (t, w) in zip(cord, o)},
                              {r["id"]: r["gt"] for r in cord})
            o2 = list(ex.map(run, [(r["path"], margin, check, crop) for r in val]))
            c = np.mean([cer(t, r["text"]) for r, (t, _) in zip(val, o2)])
            name = "no rectify" if margin is None else f"rectify, margin {margin:.2f}, check {check}" + (", else crop" if crop else "")
            print(f"{name:52s} CORD val item F1 {s['item_F1']:.3f} (P {s['item_P']:.3f} R {s['item_R']:.3f})  SRD val CER {c:.4f}", flush=True)
