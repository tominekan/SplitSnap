"""The SplitSnap pipeline with switchable stages:

    image -> [rectify] -> [preprocess] -> OCR (Tesseract) -> parse (regex | layout) -> checksum -> JSON

    python src/pipeline.py --config B0     # run one configuration on CORD test + all SRD, save JSONs

OCR outputs are cached in data/cache/ocr/ keyed by (image, rectify mode, preprocessing, psm), so configs
that share an OCR stage (e.g. regex vs layout parsing) reuse it. Timings are stored with the cache entry.
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor

from common import CACHE, RESULTS
from data import load_cord, load_image, load_srd
from ocr import run_tesseract
from parse_layout import parse_layout
from parse_regex import parse_regex

BEST_PRE_FILE = RESULTS / "preprocess_best.json"


def best_preprocess():
    """Preprocessing + psm chosen on the SRD validation split by src/preprocess.py."""
    if BEST_PRE_FILE.exists():
        b = json.loads(BEST_PRE_FILE.read_text())
        return b["options"], b["psm"]
    return None, 3


def get_configs():
    pre, psm = best_preprocess()
    return {
        "B0": dict(rectify="none", pre=None, psm=3, parser="regex"),
        "+rectify": dict(rectify="ours", pre=None, psm=3, parser="regex"),
        "+rectify+preprocess": dict(rectify="ours", pre=pre, psm=psm, parser="regex"),
        "+rectify+preprocess+layout": dict(rectify="ours", pre=pre, psm=psm, parser="layout"),
        "oracle-rectify": dict(rectify="oracle", pre=None, psm=3, parser="regex"),
        "oracle-rectify+preprocess+layout": dict(rectify="oracle", pre=pre, psm=psm, parser="layout"),
        # extra ablation: the final pipeline with rectification switched off
        "preprocess+layout (no rectify)": dict(rectify="none", pre=pre, psm=psm, parser="layout"),
        **easyocr_configs(pre, psm),
    }


def easyocr_configs(pre, psm):
    """Optional task 8: the best configuration with Tesseract swapped for EasyOCR. Opt-in with
    SPLITSNAP_EASYOCR=1 (and easyocr installed); run it with --workers 1, each worker loads its own model."""
    import os
    if os.environ.get("SPLITSNAP_EASYOCR") != "1":
        return {}
    try:
        import easyocr  # noqa: F401
    except ImportError:
        return {}
    return {
        "EasyOCR: +rectify+preprocess+layout": dict(rectify="ours", pre=pre, psm=psm, parser="layout", engine="easyocr"),
        "EasyOCR: +rectify+layout (no preprocess)": dict(rectify="ours", pre=None, psm=psm, parser="layout", engine="easyocr"),
    }


def checksum(r, rel_tol=0.005):
    """Items sum to subtotal, and subtotal + tax (+ tip) = total, within 0.5% of total (D16).
    Returns None when the fields needed are missing."""
    if r["total"] is None or r["subtotal"] is None or not r["items"]:
        return None
    tol = rel_tol * abs(r["total"])
    items_ok = abs(sum(i["price"] for i in r["items"]) - r["subtotal"]) <= tol
    total_ok = abs(r["subtotal"] + (r["tax"] or 0) + (r["tip"] or 0) - r["total"]) <= tol
    return bool(items_ok and total_ok)


def _ocr_stage(rec, rectify, pre, psm, engine="tesseract"):
    """rectify -> preprocess -> OCR for one image, cached on disk."""
    if engine == "easyocr":  # the downscale cap changes EasyOCR output, so it is part of the cache key
        from ocr import EASYOCR_MAX_SIDE
        engine = f"easyocr@{EASYOCR_MAX_SIDE}"
    parts = [rec["dataset"], rec["id"], rectify, pre, psm] + ([engine] if engine != "tesseract" else [])
    key = json.dumps(parts, sort_keys=True)
    path = CACHE / "ocr" / (hashlib.md5(key.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())
    img = load_image(rec["path"])
    t0 = time.perf_counter()
    info = {"fallback": None, "quad": None}
    if rectify != "none":
        from rectify import rectify_image
        img, info = rectify_image(img, mode=rectify, oracle_quad=rec.get("quad"))
    t1 = time.perf_counter()
    if pre:
        from preprocess import preprocess
        img = preprocess(img, pre)
    t2 = time.perf_counter()
    if engine.startswith("easyocr"):
        from ocr import run_easyocr
        text, words = run_easyocr(img)
    else:
        text, words = run_tesseract(img, psm=psm)
    t3 = time.perf_counter()
    out = {"text": text, "words": words, "rectify_info": info, "shape": list(img.shape[:2]),
           "t_rectify": t1 - t0, "t_pre": t2 - t1, "t_ocr": t3 - t2}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def _run_one(args):
    rec, cfg = args
    o = _ocr_stage(rec, cfg["rectify"], cfg["pre"], cfg["psm"], cfg.get("engine", "tesseract"))
    t0 = time.perf_counter()
    if cfg["parser"] == "regex":
        res = parse_regex(o["text"], rec["currency"])
    else:
        res = parse_layout(o["words"], rec["currency"], raw_text=o["text"])
    res["checksum"] = checksum(res)
    t_parse = time.perf_counter() - t0
    res["runtime_s"] = o["t_rectify"] + o["t_pre"] + o["t_ocr"] + t_parse
    res["rectify_info"] = o["rectify_info"]
    return rec["dataset"], rec["id"], res


def dataset_records(srd_ids=None):
    """CORD test + SRD records in a common format (currency, oracle quad)."""
    recs = [{"dataset": "cord", "id": r["id"], "path": r["path"], "currency": "IDR",
             "quad": r["roi"].tolist() if r["roi"] is not None else None} for r in load_cord("test")]
    for r in load_srd():
        if srd_ids is None or r["id"] in srd_ids:
            recs.append({"dataset": "srd", "id": r["id"], "path": r["path"], "currency": "USD",
                         "quad": r["corners"].tolist() if r["corners"] is not None else None})
    return recs


def run_config(name, cfg, recs, workers=8, save=True):
    """Run one configuration over records; returns {(dataset, id): result} and saves JSONs."""
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_run_one, [(r, cfg) for r in recs], chunksize=2))
    out = {}
    for ds, rid, res in results:
        out[(ds, rid)] = res
        if save:
            p = RESULTS / "predictions" / name / ds / f"{rid}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="B0")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = get_configs()[a.config]
    t = time.time()
    res = run_config(a.config, cfg, dataset_records(), workers=a.workers)
    print(f"{a.config}: {len(res)} receipts in {time.time() - t:.0f}s -> results/predictions/{a.config}/")
