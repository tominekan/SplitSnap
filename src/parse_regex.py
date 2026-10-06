"""Baseline parser: regular expressions over Tesseract's text lines (DECISIONS.md D10).

A line ending in a price is an item, unless it contains a keyword. Keyword lines fill
subtotal / tax / tip / total (first occurrence wins). Payment and noise lines are skipped.
"""
import re

from common import parse_amount

# Price tokens. USD needs 2 decimals; IDR needs thousands groups or 4+ digits.
PRICE = {
    "USD": r"-?\$?\s?\d{1,3}(?:,\d{3})*[.,]\d{2}",
    "IDR": r"-?(?:Rp\.?\s?)?(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?|\d{4,})",
}
# A line "ends in a price" if the price is followed by at most a short tax flag (e.g. 'T', 'F').
END_PRICE = {c: re.compile(rf"({p})\s*[A-Za-z]{{0,2}}\s*$") for c, p in PRICE.items()}

FIELD_PATTERNS = [  # checked in order: subtotal before total
    ("subtotal", re.compile(r"sub\s*-?\s*t[o0]?t[a]?l|subttl", re.I)),
    ("tax", re.compile(r"\btax|pb1|ppn|\bvat\b|pajak", re.I)),
    ("tip", re.compile(r"\btip\b|gratuity|service|\bsvc\b", re.I)),
    ("total", re.compile(r"t[o0]tal|jumlah|amount due|amt due|balance due|grand", re.I)),
]
SKIP = re.compile(r"cash|change|kembali|tunai|card|visa|master|debit|credit|tender|paid|payment|"
                  r"discount|disc\b|round|item|qty|tel\b|phone|%|x{3,}|\*{3,}|auth|approval|"
                  r"donation|bayar", re.I)
LEADING_QTY = re.compile(r"^\s*(\d{1,2})\s*[xX]?\s+(?=\D)")


def classify(text):
    """Return 'subtotal' / 'tax' / 'tip' / 'total' / 'skip' / None for a line of text."""
    for field, pat in FIELD_PATTERNS:
        if pat.search(text):
            if field == "total" and re.search(r"item|qty", text, re.I):
                return "skip"
            return field
    return "skip" if SKIP.search(text) else None


def split_qty_name(name):
    """'2 x NASI GORENG' -> (2, 'NASI GORENG')."""
    m = LEADING_QTY.match(name)
    if m:
        return int(m.group(1)), name[m.end():].strip()
    return 1, name.strip()


def clean_name(name):
    return re.sub(r"^[\W_]+|[\W_]+$", "", name).strip()


def empty_result(raw_text=""):
    return {"items": [], "subtotal": None, "tax": None, "tip": None, "total": None, "raw_text": raw_text}


def parse_regex(text, currency):
    out = empty_result(text)
    for line in text.splitlines():
        line = line.strip()
        m = END_PRICE[currency].search(line)
        if not m:
            continue
        amount = parse_amount(m.group(1), currency)
        kind = classify(line)
        if kind in ("subtotal", "tax", "tip", "total"):
            if out[kind] is None:
                out[kind] = amount
            continue
        if kind == "skip":
            continue
        qty, name = split_qty_name(line[:m.start()])
        name = clean_name(name)
        if len(re.findall(r"[A-Za-z]", name)) < 2:
            continue
        out["items"].append({"name": name, "qty": qty, "price": amount})
    return out
