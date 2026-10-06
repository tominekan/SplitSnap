"""Shared paths, Tesseract setup, and number/name normalization."""
import os
import re
import shutil
from pathlib import Path

import pytesseract

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = DATA / "cache"

# Project-local Tesseract (see DECISIONS.md D02); fall back to whatever is on PATH.
_LOCAL_TESS = ROOT / "tools" / "tess" / "bin" / "tesseract"
if _LOCAL_TESS.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_LOCAL_TESS)
    os.environ.setdefault("TESSDATA_PREFIX", str(ROOT / "tools" / "tess" / "share" / "tessdata"))
elif shutil.which("tesseract") is None:
    raise RuntimeError("Tesseract not found. See README.md for the project-local install.")

# Keep each worker single-threaded; we parallelize across images instead.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


def parse_amount(s, currency):
    """Turn a price string into a number, or None.

    USD: '$1,234.56' -> 1234.56 (the last [.,] followed by exactly 2 digits is the decimal mark).
    IDR: '60.000' / '24,000' / '17500' / '24,000.00' -> integer rupiah ('.' and ',' are
         thousands separators; a trailing 2-digit decimal part is dropped).
    """
    if s is None:
        return None
    s = str(s)
    neg = "-" in s
    s = re.sub(r"[^0-9.,]", "", s).strip(".,")
    if not s or not re.search(r"\d", s):
        return None
    if currency == "USD":
        m = re.fullmatch(r"([\d.,]*?)[.,](\d{2})", s)
        if m:
            whole = re.sub(r"\D", "", m.group(1)) or "0"
            val = float(f"{whole}.{m.group(2)}")
        else:
            val = float(re.sub(r"\D", "", s))
    else:  # IDR
        m = re.fullmatch(r"([\d.,]*\d)[.,]\d{2}", s)
        if m and len(re.sub(r"\D", "", m.group(1))) >= 3:
            s = m.group(1)
        val = float(re.sub(r"\D", "", s))
    return -val if neg else val


def normalize_name(s):
    """Lowercase, keep letters/digits/spaces, collapse whitespace."""
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def normalize_text(s):
    """Text normalization used for CER: lowercase, collapse spaces, drop empty lines."""
    lines = [re.sub(r"\s+", " ", ln).strip().lower() for ln in str(s).splitlines()]
    return "\n".join(ln for ln in lines if ln)
