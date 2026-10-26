"""Draft labels for every SRD receipt, parsed from its manual transcription.

    python src/labeling/prefill.py

Writes labels/srd_prefill/NNNN.json. The app opens a receipt with this draft until a label is saved,
so labeling becomes checking. Drafts never overwrite saved labels (those live in labels/srd/).
The draft uses the regex baseline parser on the transcription (clean text, so it gets most lines right)
and the rule-based SRD total (DECISIONS.md D07). It is a starting point only: it skips discounts,
and its tip may pick up suggested-tip lines, which the guide says to ignore.
"""
import json
import re

from labels_io import PREFILL_DIR, blank_label, receipts  # also puts src/ on the import path
from data import srd_total_from_text  # noqa: E402
from parse_regex import parse_regex  # noqa: E402

SUGGESTED_TIP = re.compile(r"suggest|tip\s*guide|\d{1,2}\s*%\s*(tip|=)", re.I)


def draft(rec):
    label = blank_label(rec["id"])
    if not rec["text"]:
        return label  # 5 images have no transcription: start blank
    parsed = parse_regex(rec["text"], "USD")
    for it in parsed["items"]:  # the parser trims a closing bracket: 'Bak Choy (L' -> 'Bak Choy (L)'
        if it["name"].count("(") > it["name"].count(")"):
            it["name"] += ")"
    label.update({
        "items": parsed["items"],
        "subtotal": parsed["subtotal"],
        "tax": parsed["tax"],
        # suggested-tip tables are not a tip (GUIDE rule 9); leave the tip for the labeler then
        "tip": None if SUGGESTED_TIP.search(rec["text"]) else parsed["tip"],
        "total": srd_total_from_text(rec["text"]) or parsed["total"],
        "prefill_source": "transcription",
    })
    return label


def main():
    PREFILL_DIR.mkdir(parents=True, exist_ok=True)
    recs = receipts()
    n_items = 0
    for rec in recs:
        label = draft(rec)
        n_items += len(label["items"])
        (PREFILL_DIR / f"{rec['id']}.json").write_text(json.dumps(label, indent=1, ensure_ascii=False) + "\n")
    with_text = sum(bool(r["text"]) for r in recs)
    print(f"Wrote {len(recs)} drafts to {PREFILL_DIR} ({with_text} from a transcription, "
          f"{len(recs) - with_text} blank), {n_items} draft items in total.")


if __name__ == "__main__":
    main()
