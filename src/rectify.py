"""Iteration 1: find the receipt and warp it flat (DECISIONS.md D12).

    python src/rectify.py   # scores detection on SRD vs labelled corners, saves figures/rectify/

Method: downscale -> blur -> three candidate masks, tried in order:
  1. "paper": bright (Otsu on HSV value) AND low saturation pixels, i.e. white paper,
  2. "otsu": Otsu threshold on grayscale (bright receipt on a darker background),
  3. "edges": Canny edges thickened with a dilation,
-> 5 largest contours -> convex hull -> polygon approximation -> first convex 4-point polygon
covering 15-97% of the image. The quad must then pass a plausibility check (every corner angle within
85-95 deg, opposite sides within 10% in length), otherwise a wrong corner would shear the receipt and
push prices onto the wrong text row; an implausible quad is replaced by its axis-aligned bounding box
(a plain crop, no shear). The quad is expanded outward by 6% first so no text is clipped.
If no quad is found at all we fall back to the original image.
Mask order and hull: src/dev/rectify_variants.py (SRD validation). Check and margin:
src/dev/rectify_margin.py (SRD validation CER and CORD validation item F1).
"""
import json

import cv2
import numpy as np
from shapely.geometry import Polygon

from common import FIGURES, RESULTS

WORK_H = 800          # detection runs on a downscaled copy
MIN_AREA, MAX_AREA = 0.15, 0.97
EPS = [0.02, 0.03, 0.05]
MIN_ANGLE, MIN_SIDE_RATIO = 85, 0.9   # plausibility check, see plausible_quad
MARGIN = 0.06                          # expand the detected quad outward before warping


def order_corners(pts):
    """Return 4x2 float32 in TL, TR, BR, BL order."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def _candidate_masks(small):
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
    _, bright = cv2.threshold(hsv[..., 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = cv2.bitwise_and(bright, (hsv[..., 1] < 60).astype(np.uint8) * 255)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    blur = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return [("paper", paper), ("otsu", otsu), ("edges", edges)]


def detect_quad(img):
    """Return (quad 4x2 in original pixel coords or None, method name)."""
    h, w = img.shape[:2]
    scale = WORK_H / h
    small = cv2.resize(img, (int(round(w * scale)), WORK_H), interpolation=cv2.INTER_AREA)
    area = small.shape[0] * small.shape[1]
    for name, mask in _candidate_masks(small):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            c = cv2.convexHull(c)
            peri = cv2.arcLength(c, True)
            for eps in EPS:
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    frac = cv2.contourArea(approx) / area
                    if MIN_AREA <= frac <= MAX_AREA:
                        return order_corners(approx) / scale, name
    return None, "fallback"


def plausible_quad(quad, min_angle=MIN_ANGLE, min_side_ratio=MIN_SIDE_RATIO):
    """Reject quads that would shear the receipt: every corner angle within [min_angle, 180 - min_angle]
    and each pair of opposite sides with length ratio >= min_side_ratio."""
    q = np.asarray(quad, dtype=np.float64)
    for i in range(4):
        a, b = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        ang = np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1)))
        if not (min_angle <= ang <= 180 - min_angle):
            return False
    side = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    return min(side[0], side[2]) / max(side[0], side[2]) >= min_side_ratio and \
        min(side[1], side[3]) / max(side[1], side[3]) >= min_side_ratio


def expand_quad(quad, margin, shape):
    """Push each corner away from the centroid by `margin` x (corner-centroid distance), clipped to the image.
    Clipping text off the receipt costs far more than keeping a little background (D12)."""
    c = quad.mean(axis=0)
    q = c + (quad - c) * (1 + margin)
    h, w = shape[:2]
    q[:, 0] = np.clip(q[:, 0], 0, w - 1)
    q[:, 1] = np.clip(q[:, 1], 0, h - 1)
    return q.astype(np.float32)


def warp(img, quad):
    """Perspective-warp the quadrilateral (TL, TR, BR, BL) to an upright rectangle."""
    tl, tr, br, bl = quad
    W = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    H = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def find_receipt(img):
    """Full detection used by the pipeline: detect -> expand by MARGIN -> plausibility check.
    Returns (quad or None, method). method is the mask name ('paper'/'otsu'/'edges') for a perspective
    warp, '<mask>+crop' when the quad was implausible and replaced by its bounding box, or 'fallback'."""
    quad, method = detect_quad(img)
    if quad is None:
        return None, method
    plausible = plausible_quad(quad)
    quad = expand_quad(quad, MARGIN, img.shape)
    if plausible:
        return quad, method
    (x0, y0), (x1, y1) = quad.min(0), quad.max(0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32), method + "+crop"


def rectify_image(img, mode="ours", oracle_quad=None):
    """mode 'ours' detects the receipt; 'oracle' uses the labelled corners (if available).
    Returns (image, info) where info records the quad and whether we fell back."""
    if mode == "oracle":
        quad = order_corners(oracle_quad) if oracle_quad is not None else None
        method = "oracle" if quad is not None else "fallback"
    else:
        quad, method = find_receipt(img)
    if quad is None:
        return img, {"fallback": True, "quad": None, "method": method}
    return warp(img, quad), {"fallback": False, "quad": quad.tolist(), "method": method}


# ------------------------------------------------------------------ scoring on SRD

def full_quad(img):
    h, w = img.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def quad_iou(a, b):
    pa, pb = Polygon(a), Polygon(b)
    if not pa.is_valid or not pb.is_valid:
        pa, pb = pa.buffer(0), pb.buffer(0)
    inter = pa.intersection(pb).area
    return inter / (pa.union(pb).area + 1e-9)


def corner_error(a, b, img):
    h, w = img.shape[:2]
    return float(np.linalg.norm(a - b, axis=1).mean() / np.hypot(w, h))


def score_rectification(recs):
    """Per image: detected?, IoU and normalized corner error vs the labelled corners, for the quad the
    pipeline actually uses (after check and margin) and for the raw detector output ('raw_' keys).
    When we fall back, the 'prediction' is the full image frame."""
    from data import load_image
    rows = []
    for r in recs:
        if r["corners"] is None:
            continue
        img = load_image(r["path"])
        gt = order_corners(r["corners"])
        raw, _ = detect_quad(img)
        raw_pred = raw if raw is not None else full_quad(img)
        quad, method = find_receipt(img)
        pred = quad if quad is not None else full_quad(img)
        rows.append({"id": r["id"], "detected": quad is not None, "method": method,
                     "iou": quad_iou(pred, gt), "corner_err": corner_error(pred, gt, img),
                     "raw_detected": raw is not None, "raw_iou": quad_iou(raw_pred, gt),
                     "raw_corner_err": corner_error(raw_pred, gt, img),
                     "iou_fullframe": quad_iou(full_quad(img), gt)})
    return rows


def summarize(rows):
    det = [r for r in rows if r["detected"]]
    return {
        "n": len(rows),
        "detection_rate": len(det) / len(rows),
        "iou_mean_all": float(np.mean([r["iou"] for r in rows])),
        "iou_mean_detected": float(np.mean([r["iou"] for r in det])) if det else float("nan"),
        "corner_err_mean_all": float(np.mean([r["corner_err"] for r in rows])),
        "corner_err_mean_detected": float(np.mean([r["corner_err"] for r in det])) if det else float("nan"),
        "good_rate_iou90": float(np.mean([r["iou"] >= 0.9 for r in rows])),
        "iou_mean_fullframe": float(np.mean([r["iou_fullframe"] for r in rows])),
        "raw_detection_rate": float(np.mean([r["raw_detected"] for r in rows])),
        "raw_iou_mean_all": float(np.mean([r["raw_iou"] for r in rows])),
        "raw_corner_err_mean_all": float(np.mean([r["raw_corner_err"] for r in rows])),
        "raw_good_rate_iou90": float(np.mean([r["raw_iou"] >= 0.9 for r in rows])),
        "implausible_cropped": int(sum(r["method"].endswith("+crop") for r in rows)),
    }


def draw_figure(r, row, path):
    """Original with labelled (green) and predicted (red) corners | our warp | oracle warp."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from data import load_image
    img = load_image(r["path"])
    gt = order_corners(r["corners"])
    quad, method = find_receipt(img)
    raw, _ = detect_quad(img)
    ours = warp(img, quad) if quad is not None else img
    oracle = warp(img, gt)
    fig, ax = plt.subplots(1, 3, figsize=(12, 5.5))
    ax[0].imshow(img[..., ::-1])
    ax[0].plot(*np.vstack([gt, gt[:1]]).T, "g-o", lw=2, ms=5, label="labelled")
    if raw is not None:
        ax[0].plot(*np.vstack([raw, raw[:1]]).T, "r--o", lw=2, ms=5, label="detected" +
                   (" (implausible)" if method.endswith("+crop") else ""))
    if quad is not None:
        ax[0].plot(*np.vstack([quad, quad[:1]]).T, ":", color="orange", lw=2,
                   label="used: " + ("bounding-box crop" if method.endswith("+crop") else "warp") + " (+6% margin)")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].set_title(f"SRD {r['id']}: IoU {row['iou']:.2f}, corner err {row['corner_err']:.3f}"
                    + ("" if quad is not None else " (fallback)"), fontsize=9)
    ax[1].imshow(ours[..., ::-1]); ax[1].set_title(f"our rectification ({method})", fontsize=9)
    ax[2].imshow(oracle[..., ::-1]); ax[2].set_title("oracle (labelled corners)", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    from data import load_srd, srd_split
    recs = load_srd()
    rows = score_rectification(recs)
    val_ids = {r["id"] for r in srd_split(recs)[0]}
    summ = {"all": summarize(rows),
            "val": summarize([r for r in rows if r["id"] in val_ids]),
            "test": summarize([r for r in rows if r["id"] not in val_ids])}
    (RESULTS / "rectify_per_image.json").write_text(json.dumps(rows, indent=1))
    (RESULTS / "rectify_summary.json").write_text(json.dumps(summ, indent=1))
    print("SRD rectification (val = used to choose the method, test = reported):", json.dumps(summ, indent=1))
    print("fell back, no quadrilateral found:", [r["id"] for r in rows if r["method"] == "fallback"])
    print("implausible quad, cropped to its bounding box instead of warping:",
          [r["id"] for r in rows if r["method"].endswith("+crop")])
    by_id = {r["id"]: r for r in recs}
    # 6 figures: 2 best, 2 median, 2 worst by IoU (among images where the receipt is not the full frame)
    ranked = sorted([r for r in rows if r["iou_fullframe"] < 0.9], key=lambda r: r["iou"])
    picks = ranked[-2:] + ranked[len(ranked) // 2 - 1: len(ranked) // 2 + 1] + ranked[:2]
    out = FIGURES / "rectify"
    out.mkdir(parents=True, exist_ok=True)
    for tag, row in zip(["best1", "best2", "median1", "median2", "worst1", "worst2"], picks):
        draw_figure(by_id[row["id"]], row, out / f"srd_{tag}_{row['id']}.png")
    print("figures:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
