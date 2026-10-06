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
