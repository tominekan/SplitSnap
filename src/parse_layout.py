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
"""
import re

import numpy as np

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


def find_price(tokens, currency):
    """Return (amount, index of first price token) for the right-most price in the row, or (None, None)."""
    pat = re.compile(rf"^(?:{PRICE[currency]})$")
    end = len(tokens)
    if end and FLAG_TOKEN.match(tokens[-1]) and end > 1:
        end -= 1  # trailing tax flag like 'T' or 'F'
    for i in range(end - 1, -1, -1):
        if i > 0 and (tokens[i - 1] in ("$", "Rp", "Rp.") or tokens[i - 1][-1] in ".,"
                      or tokens[i][0] in ".,"):
            cand = tokens[i - 1] + tokens[i]  # split price like '$ 9.39' or '60. 000'
            if pat.match(cand):
                return parse_amount(cand, currency), i - 1
        if pat.match(tokens[i]):
            return parse_amount(tokens[i], currency), i
        if i < end - 1:
            break  # only the right-most (or second right-most) token may be the price
    return None, None


def name_tokens(tokens):
    """Drop quantity/number/symbol tokens; keep words that contain letters."""
    return [t for t in tokens if re.search(r"[A-Za-z]{2,}", t) or (re.search(r"[A-Za-z]", t) and len(t) <= 3)]


def parse_layout(words, currency, raw_text="", tol=ROW_TOL):
    out = empty_result(raw_text)
    rows = [[w["text"] for w in r] for r in group_rows(words, tol)]
    pending_name = None  # previous text-only row, waiting for a price row
    for k, toks in enumerate(rows):
        text = " ".join(toks)
        amount, pi = find_price(toks, currency)
        kind = classify(text)
        if kind in ("subtotal", "tax", "tip", "total"):
            if amount is None and k + 1 < len(rows):
                nxt_amt, nxt_i = find_price(rows[k + 1], currency)
                if nxt_amt is not None and nxt_i == 0:
                    amount = nxt_amt
            if amount is not None and out[kind] is None:
                out[kind] = amount
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
    return out
