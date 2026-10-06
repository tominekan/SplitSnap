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


def run_easyocr(img):
    """Optional learned OCR (task 8). Returns (text, words) in the same format as run_tesseract.
    EasyOCR returns text boxes that often hold several words; each box is split on spaces into words
    whose x-extent is proportional to their character count. Text lines are rebuilt by grouping
    words into rows, like the layout parser does."""
    global _EASY
    import easyocr
    if _EASY is None:
        _EASY = easyocr.Reader(["en"], gpu=False, verbose=False)
    words = []
    for box, t, conf in _EASY.readtext(img):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
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
