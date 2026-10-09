"""Iteration 3: layout-aware parsing from Tesseract word boxes (DECISIONS.md D14).

1. Group words into rows by vertical center: a word joins the current row if its center is within
   ROW_TOL x (median word height) of the row's running mean center (words sorted by center y).
2. In each row (sorted left to right), the right-most token that is a price is the line price
   (a trailing 1-2 letter tax flag is allowed; two adjacent tokens are joined if that makes a price,
   e.g. '$ 9.39' or '60. 000').
3. Keyword rows (same keyword lists as the regex parser) fill subtotal/tax/tip/total; a keyword row
   with no price takes its price from the next row when that row is just a number.
4. Other priced rows are items: an optional leading quantity ('2', '2x'), the name is the remaining
   non-numeric tokens. A priced row with no name (e.g. '2 x 25,000   50,000') takes its name from
   the previous unpriced text row.
5. Robustness to OCR errors in the critical characters (parser v2, DECISIONS.md D21), developed on the
   validation splits for both Tesseract and EasyOCR output:
   - a '$' misread as 'S' is accepted (USD); a leading '5' or '8' that may be a misread '$' gives an
     alternative amount, and subtotal/total/tax pick the alternative that fits the receipt's arithmetic;
   - keywords are matched with OCR confusions ('Tota | Due', 'Ba Tance Due') via a '|' -> 'l' repair and
     fuzzy matching;
   - a price split into three tokens ('27 . 48') is joined; on subtotal/tax/tip/total rows a decimal
     point read as a space or junk ('$13 03', '$88 _ 62') is accepted;
   - 'UB TOTAL' (subtotal with the S dropped) is not taken as the total.
"""
import re

import numpy as np
from rapidfuzz import fuzz

from common import parse_amount
from parse_regex import PRICE, classify, clean_name, empty_result

ROW_TOL = 0.5
QTY_TOKEN = re.compile(r"^(\d{1,2})[xX]?$")
FLAG_TOKEN = re.compile(r"^[A-Za-z]{1,2}$")


def group_rows(words, tol=ROW_TOL):
    """Return a list of rows (each a list of words sorted left to right), top to bottom."""
    if not words:
        return []
    mh = np.median([w["height"] for w in words])
    rows = []
    for w in sorted(words, key=lambda w: w["top"] + w["height"] / 2):
        cy = w["top"] + w["height"] / 2
        if rows and abs(cy - rows[-1]["cy"]) <= tol * mh:
            r = rows[-1]
            r["words"].append(w)
            r["cy"] = np.mean([x["top"] + x["height"] / 2 for x in r["words"]])
        else:
            rows.append({"cy": cy, "words": [w]})
    return [sorted(r["words"], key=lambda w: w["left"]) for r in rows]


def _amounts(token, currency):
    """Candidate values for a price token: [primary] or [primary, alternative]. For USD, a leading
    '5' or '8' directly before a 2+ digit integer part may be a misread '$' (EasyOCR does this)."""
    if currency == "USD" and re.match(r"^[58]\d{3,}[.,]\d{2}$", token):
        return [parse_amount(token[1:], currency)]  # '8143.71': 4+ digits without a comma, so the 8 is a '$'
    vals = [parse_amount(token, currency)]
    if currency == "USD" and re.match(r"^[58]\d{1,2}[.,]\d{2}$", token):
        vals.append(parse_amount(token[1:], currency))  # '532.70' could be 532.70 or $32.70
    return vals


def find_price_cands(tokens, currency, field_row=False):
    """Return (candidate amounts, index of first price token) for the right-most price in the row,
    or ([], None). The first candidate is the literal reading. On subtotal/tax/tip/total rows
    (field_row=True), a USD amount whose decimal point was read as a space or junk ('$13 03',
    '$88 _ 62') is also accepted."""
    pat = re.compile(rf"^(?:{PRICE[currency]})$")
    # strip OCR junk glued to a price, e.g. '«$1.40' or '$9.39)'
    tokens = [re.sub(r"^[^\w$.,-]+|[^\w.,]+$", "", t) or t for t in tokens]
    if currency == "USD":  # '$' misread as 'S': 'S64.43' -> '$64.43'
        tokens = [re.sub(r"^[Ss](?=\d+[.,]\d{2}$)", "$", t) for t in tokens]
    end = len(tokens)
    if end and FLAG_TOKEN.match(tokens[-1]) and end > 1:
        end -= 1  # trailing tax flag like 'T' or 'F'
    misread = re.compile(r"^[58]\d{3,}[.,]\d{2}$") if currency == "USD" else None
    if field_row and currency == "USD" and end >= 2 and re.fullmatch(r"\d{2}", tokens[end - 1]):
        j = end - 2
        if j >= 1 and tokens[j] in ("_", "-", "~", "'"):
            j -= 1  # junk in place of the decimal point
        if re.fullmatch(r"[$Ss]?\d{1,3}(?:,\d{3})*", tokens[j]):
            return _amounts(re.sub(r"^[Ss]", "$", tokens[j]) + "." + tokens[end - 1], currency), j
    for i in range(end - 1, -1, -1):
        if misread is not None and misread.match(tokens[i]):
            return _amounts(tokens[i], currency), i
        if i > 1 and tokens[i - 1] in (".", ","):
            cand = "".join(tokens[i - 2:i + 1])  # price split in three: '27 . 48'
            if pat.match(cand):
                return _amounts(cand, currency), i - 2
        if i > 0 and (tokens[i - 1] in ("$", "Rp", "Rp.") or tokens[i - 1][-1] in ".,"
                      or tokens[i][0] in ".,"):
            cand = tokens[i - 1] + tokens[i]  # split price like '$ 9.39' or '60. 000'
            if pat.match(cand):
                return _amounts(cand, currency), i - 1
        if pat.match(tokens[i]):
            return _amounts(tokens[i], currency), i
        if i < end - 1:
            break  # only the right-most (or second right-most) token may be the price
    return [], None


def find_price(tokens, currency):
    """Return (amount, index of first price token) for the right-most price in the row, or (None, None)."""
    cands, i = find_price_cands(tokens, currency)
    return (cands[0], i) if cands else (None, None)


FUZZY_KEYWORDS = [  # letters-only keyword, field; checked in order (subtotal before total)
    ("subtotal", "subtotal"), ("balancedue", "total"), ("amountdue", "total"), ("amtdue", "total"),
    ("grandtotal", "total"), ("total", "total"),
]


def classify_row(text):
    """classify() from the regex parser, made robust to OCR errors in keywords: a '|' after a letter is
    read as 'l' ('Tota | Due' -> 'Total Due'), and if nothing matches, keywords are matched fuzzily on
    the letters-only row text (rapidfuzz partial_ratio >= 85)."""
    fixed = re.sub(r"(?<=[A-Za-z]) ?\| ?", "l ", text)
    kind = classify(fixed)
    letters = re.sub(r"[^a-z]", "", fixed.lower())
    if kind == "total" and re.search(r"(?:sub|ub|sb|stb)t[o0]?ta", letters):
        return "subtotal"  # 'UB TOTAL' (the S dropped) is a subtotal, not a total
    if kind is not None:
        return kind
    for kw, field in FUZZY_KEYWORDS:
        if len(letters) >= len(kw) - 1 and fuzz.partial_ratio(kw, letters) >= 85:
            return field
    return None


def resolve(cands, expected):
    """Pick the candidate closest to an expected value (from the receipt's arithmetic), else the literal one."""
    if len(cands) == 1 or expected is None:
        return cands[0]
    return min(cands, key=lambda v: abs(v - expected))


def name_tokens(tokens):
    """Drop quantity/number/symbol tokens; keep words that contain letters."""
    return [t for t in tokens if re.search(r"[A-Za-z]{2,}", t) or (re.search(r"[A-Za-z]", t) and len(t) <= 3)]


def parse_layout(words, currency, raw_text="", tol=ROW_TOL):
    out = empty_result(raw_text)
    rows = [[w["text"] for w in r] for r in group_rows(words, tol)]
    pending_name = None  # previous text-only row, waiting for a price row
    field_cands = {}
    for k, toks in enumerate(rows):
        text = " ".join(toks)
        kind = classify_row(text)
        field_row = kind in ("subtotal", "tax", "tip", "total")
        cands, pi = find_price_cands(toks, currency, field_row)
        amount = cands[0] if cands else None
        if field_row:
            if not cands and k + 1 < len(rows):
                nxt, nxt_i = find_price_cands(rows[k + 1], currency, True)
                if nxt and nxt_i == 0:
                    cands = nxt
            if cands and kind not in field_cands:
                field_cands[kind] = cands
            pending_name = None
            continue
        if kind == "skip":
            pending_name = None
            continue
        if amount is None:
            nm = name_tokens(toks)
            pending_name = (toks, nm) if len("".join(nm)) >= 2 else pending_name
            continue
        left = toks[:pi]
        qty = 1
        if left and QTY_TOKEN.match(left[0]):
            qty, left = int(QTY_TOKEN.match(left[0]).group(1)), left[1:]
        name = clean_name(" ".join(name_tokens(left)))
        if len(re.findall(r"[A-Za-z]", name)) < 2 and pending_name is not None:
            ptoks, pnm = pending_name
            if ptoks and QTY_TOKEN.match(ptoks[0]) and qty == 1:
                qty = int(QTY_TOKEN.match(ptoks[0]).group(1))
            name = clean_name(" ".join(pnm))
        pending_name = None
        if len(re.findall(r"[A-Za-z]", name)) < 2:
            continue
        out["items"].append({"name": name, "qty": qty, "price": amount})
    # resolve fields whose reading is ambiguous ('$' read as '5'/'8') using the receipt's arithmetic
    items_sum = sum(i["price"] for i in out["items"]) or None
    if "tip" in field_cands:
        out["tip"] = field_cands["tip"][0]
    if "subtotal" in field_cands:
        out["subtotal"] = resolve(field_cands["subtotal"], items_sum)
    if "total" in field_cands:
        base = out["subtotal"] if out["subtotal"] is not None else items_sum
        tax0 = field_cands["tax"][0] if "tax" in field_cands else 0
        out["total"] = resolve(field_cands["total"], None if base is None else base + tax0 + (out["tip"] or 0))
    if "tax" in field_cands:
        diff = None
        if out["total"] is not None and out["subtotal"] is not None:
            diff = out["total"] - out["subtotal"] - (out["tip"] or 0)
        out["tax"] = resolve(field_cands["tax"], diff)
    return out
