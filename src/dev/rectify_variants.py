"""Development: compare receipt-detection variants on the SRD validation split (30 images).
    python src/dev/rectify_variants.py  -> results/logs/rectify_variants_val.txt"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import cv2, numpy as np
import rectify as R
from data import load_srd, srd_split, load_image

def masks(small, which):
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    out = []
    for w in which:
        if w == 'edges':
            e = cv2.dilate(cv2.Canny(blur, 50, 150), np.ones((5, 5), np.uint8), iterations=2); out.append(e)
        if w == 'otsu':
            _, o = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            out.append(cv2.morphologyEx(o, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)))
        if w == 'hsv':
            hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
            _, v = cv2.threshold(hsv[..., 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            m = cv2.bitwise_and(v, (hsv[..., 1] < 60).astype(np.uint8) * 255)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            out.append(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)))
    return out

def detect(img, which, select, maxa=0.97, hull=False):
    h, w = img.shape[:2]; s = 800 / h
    small = cv2.resize(img, (int(round(w * s)), 800), interpolation=cv2.INTER_AREA)
    area = small.shape[0] * small.shape[1]
    cands = []
    for m in masks(small, which):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            if hull: c = cv2.convexHull(c)
            peri = cv2.arcLength(c, True)
            for eps in R.EPS:
                a = cv2.approxPolyDP(c, eps * peri, True)
                if len(a) == 4 and cv2.isContourConvex(a):
                    frac = cv2.contourArea(a) / area
                    if R.MIN_AREA <= frac <= maxa:
                        fill = cv2.contourArea(c) / (cv2.contourArea(a) + 1e-9)
                        cands.append((frac, min(fill, 1 / max(fill, 1e-9)), R.order_corners(a) / s))
                        break
        if select == 'first' and cands: return cands[0][2]
    if not cands: return None
    if select == 'fill':
        good = [c for c in cands if c[1] > 0.9] or cands
        return max(good, key=lambda c: c[0])[2]
    return cands[0][2]

val, test = srd_split(load_srd())
imgs = [(load_image(r['path']), R.order_corners(r['corners'])) for r in val]
variants = {
 'A edges,otsu first': (['edges', 'otsu'], 'first', 0.97, False),
 'B otsu,edges first': (['otsu', 'edges'], 'first', 0.97, False),
 'C hsv,otsu,edges first': (['hsv', 'otsu', 'edges'], 'first', 0.97, False),
 'D all fill-select': (['edges', 'otsu', 'hsv'], 'fill', 0.97, False),
 'E hsv,otsu,edges first hull': (['hsv', 'otsu', 'edges'], 'first', 0.97, True),
 'F all fill-select hull': (['edges', 'otsu', 'hsv'], 'fill', 0.97, True),
 'G A maxarea .90': (['edges', 'otsu'], 'first', 0.90, False),
}
for name, (wh, sel, ma, hu) in variants.items():
    ious, det = [], 0
    for img, gt in imgs:
        q = detect(img, wh, sel, ma, hu)
        det += q is not None
        ious.append(R.quad_iou(q if q is not None else R.full_quad(img), gt))
    print(f"{name:32s} det {det/len(imgs):.2f}  meanIoU {np.mean(ious):.3f}  IoU>=.9 {np.mean(np.array(ious)>=.9):.2f}")
