"""Shared helpers for SRD field labels: paths, receipt order, load and save.

Label files live in labels/srd/NNNN.json (committed to git); drafts from prefill.py live in
labels/srd_prefill/NNNN.json. Rules for what goes in each field: labels/GUIDE.md.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the project modules
from common import ROOT  # noqa: E402
from data import load_srd, srd_split  # noqa: E402

GUIDE_VERSION = 1
LABELS = ROOT / "labels"
LABEL_DIR = LABELS / "srd"
PREFILL_DIR = LABELS / "srd_prefill"
FIELDS = ["subtotal", "tax", "tip", "total", "tip_handwritten"]
FLAG_OPTIONS = ["cropped", "illegible", "multiple_receipts", "not_a_receipt", "handwritten_items", "other"]


def receipts():
    """All SRD receipts in labeling order: the 30 validation receipts first, then test, each by id.
    Each record: id, path, text (transcription or None), split ('validation' or 'test')."""
    val, test = srd_split(load_srd())
    return ([{**r, "split": "validation"} for r in sorted(val, key=lambda r: r["id"])]
            + [{**r, "split": "test"} for r in sorted(test, key=lambda r: r["id"])])


def blank_label(rid):
    return {"id": rid, "status": "todo", "items": [], **{k: None for k in FIELDS},
            "flags": [], "notes": "", "guide_version": GUIDE_VERSION, "prefill_source": "none"}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def load_label(rid, label_dir=LABEL_DIR, use_prefill=True):
    """Saved label if there is one, else the pre-filled draft, else a blank label."""
    saved = read_json(Path(label_dir) / f"{rid}.json")
    if saved:
        return saved
    draft = read_json(PREFILL_DIR / f"{rid}.json") if use_prefill else None
    return draft or blank_label(rid)


def save_label(label, label_dir=LABEL_DIR):
    """Write one label atomically (temporary file, then rename), so a crash never leaves half a file."""
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    label = {**label, "guide_version": GUIDE_VERSION,
             "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = label_dir / f"{label['id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(label, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return label


def saved_status(rid, label_dir=LABEL_DIR):
    saved = read_json(Path(label_dir) / f"{rid}.json")
    return saved["status"] if saved else "todo"
