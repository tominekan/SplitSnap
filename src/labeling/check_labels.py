"""Check the SRD labels: progress, schema problems, arithmetic, and agreement with another label set.

    python src/labeling/check_labels.py                                  # labels/srd vs the drafts
    python src/labeling/check_labels.py --compare labels/srd_relabel     # self-agreement on a blind relabel

Agreement uses the same matching as our metrics (src/metrics.py): an item agrees if its price is equal
and its normalized name is at least 80% similar; amount fields agree if they are equal (or both empty).
"""
import argparse
from pathlib import Path

from labels_io import FIELDS, GUIDE_VERSION, LABEL_DIR, PREFILL_DIR, ROOT, read_json, receipts  # noqa: F401
from metrics import match_items, prf, same_amount  # noqa: E402

EPS = 0.005


def problems(label):
    """Schema problems in one saved label (empty list if it is fine)."""
    out = []
    if label.get("status") not in ("done", "flagged"):
        out.append(f"status is {label.get('status')!r}")
    for i, it in enumerate(label.get("items", []), 1):
        if not str(it.get("name") or "").strip():
            out.append(f"item {i} has no name")
        if it.get("price") is None:
            out.append(f"item {i} has no price")
        elif it["price"] < 0 and not str(it.get("name")).strip():
            out.append(f"item {i} is a discount without a name")
    for f in FIELDS:
        v = label.get(f)
        if v is not None and (not isinstance(v, (int, float)) or v < 0):
            out.append(f"{f} = {v!r} is not a non-negative number")
    if label.get("status") == "flagged" and not label.get("flags"):
        out.append("flagged without a flag")
    if label.get("status") == "done" and not label.get("items") and label.get("total") is None:
        out.append("done but has no items and no total")
    if label.get("guide_version", 0) < GUIDE_VERSION:
        out.append(f"made under guide version {label.get('guide_version')} (current {GUIDE_VERSION})")
    return out


def arithmetic(label):
    """(items add up to subtotal, subtotal + tax + tip = total); None where a value is missing."""
    prices = [it["price"] for it in label.get("items", []) if it.get("price") is not None]
    sub, total = label.get("subtotal"), label.get("total")
    items_ok = None if sub is None or not prices else abs(sum(prices) - sub) < EPS
    base = sub if sub is not None else (sum(prices) if prices else None)
    total_ok = None if total is None or base is None else \
        abs(base + (label.get("tax") or 0) + (label.get("tip") or 0) - total) < EPS
    return items_ok, total_ok


def agreement(a, b, ids):
    """Field agreement and item P/R/F1 of label set b against label set a, on the given ids."""
    field_hits = {f: 0 for f in FIELDS}
    tp = n_a = n_b = 0
    for i in ids:
        for f in FIELDS:
            va, vb = a[i].get(f), b[i].get(f)
            field_hits[f] += (va is None and vb is None) or same_amount(va, vb)
        ia, ib = a[i].get("items", []), b[i].get("items", [])
        tp, n_a, n_b = tp + len(match_items(ib, ia)), n_a + len(ia), n_b + len(ib)
    p, r, f1 = prf(tp, n_b, n_a)
    return {f: h / len(ids) for f, h in field_hits.items()}, (p, r, f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(LABEL_DIR))
    ap.add_argument("--compare", help="second label folder (e.g. a blind relabel) to compare against")
    a = ap.parse_args()
    resolve = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    label_dir = resolve(a.labels)

    recs = receipts()
    labels = {r["id"]: read_json(label_dir / f"{r['id']}.json") for r in recs}
    saved = {i: lab for i, lab in labels.items() if lab}
    split = {r["id"]: r["split"] for r in recs}
    n_val = sum(s == "validation" for s in split.values())

    print(f"Labels in {label_dir.relative_to(ROOT) if label_dir.is_relative_to(ROOT) else label_dir}")
    print(f"  {len(saved)}/{len(recs)} labeled: {sum(l['status'] == 'done' for l in saved.values())} done, "
          f"{sum(l['status'] == 'flagged' for l in saved.values())} flagged; validation "
          f"{sum(split[i] == 'validation' for i in saved)}/{n_val}, test "
          f"{sum(split[i] == 'test' for i in saved)}/{len(recs) - n_val}")
    print(f"  {sum(len(l.get('items', [])) for l in saved.values())} items labeled")

    bad = {i: problems(lab) for i, lab in saved.items() if problems(lab)}
    print(f"\nSchema problems: {len(bad)}")
    for i, ps in sorted(bad.items()):
        print(f"  {i}: {'; '.join(ps)}")

    flagged = {i: lab for i, lab in saved.items() if lab["status"] == "flagged"}
    print(f"\nFlagged: {len(flagged)}")
    for i, lab in sorted(flagged.items()):
        print(f"  {i}: {', '.join(lab.get('flags', []))}  {lab.get('notes', '')}")

    checks = {i: arithmetic(lab) for i, lab in saved.items() if lab["status"] == "done"}
    items_bad = sorted(i for i, (x, _) in checks.items() if x is False)
    total_bad = sorted(i for i, (_, y) in checks.items() if y is False)
    print(f"\nArithmetic on {len(checks)} done labels (a mismatch is fine if printed that way; recheck for typos):")
    print(f"  items != subtotal: {len(items_bad)} {items_bad}")
    print(f"  subtotal + tax + tip != total: {len(total_bad)} {total_bad}")

    if a.compare:
        other_dir = resolve(a.compare)
        other = {i: read_json(other_dir / f"{i}.json") for i in saved}
        ids = sorted(i for i, lab in other.items() if lab)
        title, ref, cmp = f"Agreement with {a.compare}", saved, other
    else:
        drafts = {i: read_json(PREFILL_DIR / f"{i}.json") for i in saved}
        ids = sorted(i for i, d in drafts.items() if d and d.get("prefill_source") == "transcription")
        title, ref, cmp = "How much the drafts were corrected (draft vs label)", saved, drafts
    if ids:
        fields, (p, r, f1) = agreement(ref, cmp, ids)
        print(f"\n{title}, {len(ids)} receipts:")
        print("  fields equal: " + ", ".join(f"{f} {v:.0%}" for f, v in fields.items()))
        print(f"  items: precision {p:.3f}, recall {r:.3f}, F1 {f1:.3f}")


if __name__ == "__main__":
    main()
