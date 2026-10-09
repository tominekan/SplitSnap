"""Tesseract wrapper: one image_to_data call gives both word boxes and the line text."""
import cv2
import pytesseract

import common  # noqa: F401  (sets the tesseract path)


def run_tesseract(img, psm=3, lang="eng"):
    """Return (text, words). words = list of dicts with text, conf, left, top, width, height,
    block, par, line. Text is rebuilt from Tesseract's own line grouping."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
    d = pytesseract.image_to_data(rgb, lang=lang, config=f"--psm {psm}",
                                  output_type=pytesseract.Output.DICT)
    words, lines = [], {}
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t or float(d["conf"][i]) < 0:
            continue
        w = {k: int(d[k][i]) for k in ["left", "top", "width", "height", "block_num", "par_num", "line_num"]}
        w["text"], w["conf"] = t, float(d["conf"][i])
        words.append(w)
        lines.setdefault((w["block_num"], w["par_num"], w["line_num"]), []).append(t)
    text = "\n".join(" ".join(ws) for ws in lines.values())
    return text, words


_EASY = None
# Memory cap: EasyOCR's memory grows with image size (a 4032x3024 photo peaked at ~11 GB footprint), so
# images are downscaled to at most this many pixels on the long side before OCR, and boxes are mapped back.
# Measured on that photo with 2 threads: 1600 px -> 2.7 GB resident, 12.4 s; 1024 px -> 2.1 GB, 4.4 s.
EASYOCR_MAX_SIDE = 1600


def run_easyocr(img):
    """Optional learned OCR (task 8). Returns (text, words) in the same format as run_tesseract.
    EasyOCR returns text boxes that often hold several words; each box is split on spaces into words
    whose x-extent is proportional to their character count. Text lines are rebuilt by grouping
    words into rows, like the layout parser does."""
    global _EASY
    import easyocr
    if _EASY is None:
        from common import ROOT
        d = ROOT / "tools" / "easyocr"  # keep model downloads inside the project (default is ~/.EasyOCR)
        d.mkdir(parents=True, exist_ok=True)
        _EASY = easyocr.Reader(["en"], gpu=False, verbose=False, model_storage_directory=str(d / "model"),
                               user_network_directory=str(d / "user_network"))
    scale = min(1.0, EASYOCR_MAX_SIDE / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    words = []
    for box, t, conf in _EASY.readtext(small, canvas_size=EASYOCR_MAX_SIDE, batch_size=1):
        xs, ys = [p[0] / scale for p in box], [p[1] / scale for p in box]
        x0, width, n = min(xs), max(xs) - min(xs), max(len(t), 1)
        pos = 0
        for part in t.split(" "):
            if part:
                words.append({"text": part, "conf": float(conf) * 100, "left": int(x0 + width * pos / n),
                              "top": int(min(ys)), "width": max(1, int(width * len(part) / n)),
                              "height": int(max(ys) - min(ys)), "block_num": 0, "par_num": 0, "line_num": 0})
            pos += len(part) + 1
    from parse_layout import group_rows
    text = "\n".join(" ".join(w["text"] for w in row) for row in group_rows(words))
    return text, words
