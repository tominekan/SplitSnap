"""Download and load the two datasets.

    python src/data.py      # downloads CORD v2 + SRD into data/ and prints a summary

CORD v2 (naver-clova-ix/cord-v2): test split for reporting, validation split for developing parser rules.
SRD (Zenodo 13688441, Dataset.zip): 200 US restaurant receipts, text labels + 4 corner labels.
"""
import csv
import json
import random
import re
import sys
import urllib.request
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from common import DATA, parse_amount

CORD_DIR = DATA / "cord"
SRD_DIR = DATA / "srd"
SRD_URL = "https://zenodo.org/records/13688441/files/Dataset.zip?download=1"
SRD_VAL_SIZE = 30
SPLIT_SEED = 0


# ---------------------------------------------------------------- CORD

def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _as_str(x):
    return " ".join(map(str, x)) if isinstance(x, list) else (str(x) if x is not None else None)


def _qty(cnt):
    m = re.search(r"\d+(?:[.,]\d+)?", _as_str(cnt) or "")
    if not m:
        return 1
    q = float(m.group(0).replace(",", "."))
    return int(q) if q == int(q) else q


def cord_to_schema(gt_parse):
    """Convert one CORD gt_parse dict to our schema (see DECISIONS.md D05)."""
    items = []
    for entry in _as_list(gt_parse.get("menu")):
        rows = [entry] + _as_list(entry.get("sub"))  # priced modifiers become their own items
        for r in rows:
            price = parse_amount(_as_str(r.get("price") or r.get("itemsubtotal")), "IDR")
            if price is None or not r.get("nm"):
                continue
            items.append({"name": _as_str(r["nm"]), "qty": _qty(r.get("cnt")), "price": price})
    sub = gt_parse.get("sub_total") or {}
    tot = gt_parse.get("total") or {}
    if isinstance(sub, list):
        sub = sub[0]
    if isinstance(tot, list):
        tot = tot[0]
    return {
        "items": items,
        "subtotal": parse_amount(_as_str(sub.get("subtotal_price")), "IDR"),
        "tax": parse_amount(_as_str(sub.get("tax_price")), "IDR"),
        "tip": parse_amount(_as_str(sub.get("service_price")), "IDR"),
        "total": parse_amount(_as_str(tot.get("total_price")), "IDR"),
        "raw_text": None,
    }


def _roi_quad(roi):
    """CORD roi (x1..x4, y1..y4; TL, TR, BR, BL) -> 4x2 array, or None if absent."""
    if not roi or "x1" not in roi:
        return None
    return np.array([[roi[f"x{i}"], roi[f"y{i}"]] for i in range(1, 5)], dtype=np.float32)


def download_cord():
    from datasets import load_dataset
    import os
    os.environ.setdefault("HF_HOME", str(DATA / "hf"))
    for split in ["test", "validation"]:
        out = CORD_DIR / split
        if (out / "index.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        ds = load_dataset("naver-clova-ix/cord-v2", split=split, cache_dir=str(DATA / "hf"))
        index = []
        for i, row in enumerate(ds):
            g = json.loads(row["ground_truth"])
            rid = f"{split}_{i:03d}"
            row["image"].convert("RGB").save(out / f"{rid}.png")
            roi = _roi_quad(g.get("roi"))
            rec = {"id": rid, "gt": cord_to_schema(g["gt_parse"]),
                   "roi": roi.tolist() if roi is not None else None, "gt_parse": g["gt_parse"]}
            (out / f"{rid}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
            index.append(rid)
        (out / "index.json").write_text(json.dumps(index))


def load_cord(split="test"):
    """List of dicts: id, path, gt (our schema), roi (4x2 TL,TR,BR,BL or None)."""
    out = CORD_DIR / split
    recs = []
    for rid in json.loads((out / "index.json").read_text()):
        rec = json.loads((out / f"{rid}.json").read_text())
        rec["path"] = str(out / f"{rid}.png")
        rec["roi"] = np.array(rec["roi"], dtype=np.float32) if rec["roi"] else None
        recs.append(rec)
    return recs


# ---------------------------------------------------------------- SRD

def download_srd():
    SRD_DIR.mkdir(parents=True, exist_ok=True)
    z = SRD_DIR / "Dataset.zip"
    if not z.exists():
        urllib.request.urlretrieve(SRD_URL, z)
    if not (SRD_DIR / "homography.csv").exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(SRD_DIR)


def load_srd_corners():
    """homography.csv: Image;P1;P2;P3;P4 with P1=TL, P2=TR, P3=BL, P4=BR as 'x,y'.
    Returned as {id: 4x2 array in TL, TR, BR, BL order}."""
    corners = {}
    with open(SRD_DIR / "homography.csv") as f:
        for row in list(csv.reader(f, delimiter=";"))[1:]:
            p = [tuple(map(float, c.split(","))) for c in row[1:5]]
            corners[row[0]] = np.array([p[0], p[1], p[3], p[2]], dtype=np.float32)
    return corners


def load_image(path):
    """Read an image as BGR uint8, applying EXIF orientation (SRD corner labels assume it)."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def load_srd():
    """List of dicts: id, path, corners (4x2 or None), text (str or None)."""
    corners = load_srd_corners()
    recs = []
    for p in sorted(SRD_DIR.glob("*-receipt.jpg")):
        rid = p.name.split("-")[0]
        lab = SRD_DIR / "Labels" / f"{rid}-receipt.txt"
        recs.append({"id": rid, "path": str(p), "corners": corners.get(rid),
                     "text": lab.read_text(errors="replace") if lab.exists() else None})
    return recs


def srd_split(recs):
    """Held-out validation split (30 images with text labels) used for all tuning; rest is test."""
    labeled = sorted(r["id"] for r in recs if r["text"] is not None)
    val = set(random.Random(SPLIT_SEED).sample(labeled, SRD_VAL_SIZE))
    return [r for r in recs if r["id"] in val], [r for r in recs if r["id"] not in val]


_AMT = r"\$?\s?(\d{1,3}(?:,\d{3})*\.\d{2})\b"


def srd_total_from_text(text):
    """Derive the ground-truth total from an SRD annotation (DECISIONS.md D07).

    Candidate lines contain 'total', 'amount due', 'amt due' or 'balance due' and end in an amount, excluding sub-total, tax-total and
    item-count lines. We take the largest candidate amount. 'Reliable' requires that the
    total is >= the annotated subtotal when a subtotal line exists. Returns float or None."""
    totals, subs = [], []
    for ln in text.splitlines():
        low = ln.lower()
        amts = re.findall(_AMT, ln)
        if not amts:
            continue
        amt = parse_amount(amts[-1], "USD")
        if re.search(r"sub\s*-?\s*total", low):
            subs.append(amt)
        elif (re.search(r"total|amount due|amt due|balance due", low)
              and not re.search(r"tax|item|qty|discount|savings|tip amount|change", low)):
            totals.append(amt)
    if not totals:
        return None
    total = max(totals)
    if subs and total < max(subs):
        return None
    return total


def main():
    try:
        download_cord()
        test, val = load_cord("test"), load_cord("validation")
        n_items = sum(len(r["gt"]["items"]) for r in test)
        print(f"CORD: test={len(test)} receipts ({n_items} priced items), validation={len(val)}; "
              f"test with roi={sum(r['roi'] is not None for r in test)}")
    except Exception as e:
        print("CORD failed:", repr(e), file=sys.stderr)
    try:
        download_srd()
        srd = load_srd()
        val, test = srd_split(srd)
        n_tot = sum(srd_total_from_text(r["text"]) is not None for r in srd if r["text"])
        print(f"SRD: {len(srd)} images, {sum(r['text'] is not None for r in srd)} with text, "
              f"{sum(r['corners'] is not None for r in srd)} with corners, val={len(val)}, test={len(test)}, "
              f"reliable totals={n_tot}")
        for r in srd:  # sanity check: corners inside the (EXIF-rotated) image
            if r["corners"] is not None:
                h, w = load_image(r["path"]).shape[:2]
                c = r["corners"]
                if c[:, 0].max() > w + 2 or c[:, 1].max() > h + 2 or c.min() < -2:
                    print("  corner outside image:", r["id"], (w, h), c.tolist())
    except Exception as e:
        print("SRD failed:", repr(e), file=sys.stderr)


if __name__ == "__main__":
    main()
