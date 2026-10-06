"""Scoring functions (DECISIONS.md D15 item matching, D17 metrics)."""
import jiwer
import numpy as np
from rapidfuzz import fuzz

from common import normalize_name, normalize_text

NAME_SIM = 0.8   # normalized-name similarity threshold for an item match
AMOUNT_EPS = 0.005


def same_amount(a, b):
    return a is not None and b is not None and abs(a - b) < AMOUNT_EPS


def match_items(pred, gt):
    """Greedy one-to-one matching. A pair matches if prices are equal and the normalized names
    have rapidfuzz ratio >= 0.8. Returns list of (gt_idx, pred_idx, sim)."""
    cands = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if same_amount(p["price"], g["price"]):
                sim = fuzz.ratio(normalize_name(p["name"]), normalize_name(g["name"])) / 100
                if sim >= NAME_SIM:
                    cands.append((sim, gi, pi))
    used_g, used_p, pairs = set(), set(), []
    for sim, gi, pi in sorted(cands, reverse=True):
        if gi not in used_g and pi not in used_p:
            used_g.add(gi), used_p.add(pi), pairs.append((gi, pi, sim))
    return pairs


def prf(tp, n_pred, n_gt):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score_cord(preds, gts):
    """preds, gts: dicts id -> schema dict. Micro-averaged item P/R/F1, field exact match,
    checksum pass rate. Also returns per-receipt item F1."""
    tp = n_pred = n_gt = 0
    per = {}
    field_hits = {k: [0, 0] for k in ["subtotal", "tax", "total"]}
    checks = []
    for rid, g in gts.items():
        p = preds[rid]
        m = len(match_items(p["items"], g["items"]))
        tp, n_pred, n_gt = tp + m, n_pred + len(p["items"]), n_gt + len(g["items"])
        per[rid] = prf(m, len(p["items"]), len(g["items"]))[2] if g["items"] else None
        for k in field_hits:
            if g[k] is not None:
                field_hits[k][1] += 1
                field_hits[k][0] += same_amount(p[k], g[k])
        checks.append(p.get("checksum") is True)
    P, R, F = prf(tp, n_pred, n_gt)
    out = {"item_P": P, "item_R": R, "item_F1": F,
           "checksum_pass": float(np.mean(checks))}
    for k, (h, n) in field_hits.items():
        out[f"{k}_acc"] = h / n if n else float("nan")
        out[f"{k}_n"] = n
    return out, per


def cer(pred_text, gt_text):
    g, p = normalize_text(gt_text), normalize_text(pred_text)
    if not g:
        return float("nan")
    return jiwer.cer(g, p) if p else 1.0


def score_srd(preds, texts, totals):
    """preds: id -> schema dict; texts: id -> annotation; totals: id -> derived total or None."""
    cers = [cer(preds[i]["raw_text"], t) for i, t in texts.items()]
    tot = [same_amount(preds[i]["total"], v) for i, v in totals.items() if v is not None]
    return {"CER": float(np.nanmean(cers)), "CER_median": float(np.nanmedian(cers)),
            "total_acc": float(np.mean(tot)) if tot else float("nan"), "total_n": len(tot)}
