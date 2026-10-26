"""Streamlit app for labeling SRD receipts with items, subtotal, tax, tip and total.

    streamlit run src/labeling/app.py
    streamlit run src/labeling/app.py -- --out labels/srd_relabel --no-prefill   # blind relabel for agreement

Rules for each field: labels/GUIDE.md. Each receipt opens with its saved label, else the draft from
prefill.py, else an empty form. Nothing is written until you press a Save button.
"""
import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels_io import (FIELDS, FLAG_OPTIONS, LABEL_DIR, PREFILL_DIR, ROOT, blank_label,  # noqa: E402
                       load_label, read_json, receipts, save_label, saved_status)
from data import load_image  # noqa: E402

FIELD_NAMES = {"subtotal": "Subtotal", "tax": "Tax (all tax lines)", "tip": "Tip / service in bill",
               "total": "Total (printed, before handwritten tip)", "tip_handwritten": "Handwritten tip"}
EPS = 0.005  # amounts are equal if they differ by less than half a cent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LABEL_DIR), help="folder for saved labels")
    ap.add_argument("--no-prefill", action="store_true", help="open unlabeled receipts with an empty form")
    a = ap.parse_args(sys.argv[1:])
    out = Path(a.out)
    return (out if out.is_absolute() else ROOT / out), not a.no_prefill


OUT, USE_PREFILL = parse_args()
st.set_page_config(page_title="SRD labeling", layout="wide")
ss = st.session_state


@st.cache_data
def all_receipts():
    return [{"id": r["id"], "path": r["path"], "text": r["text"], "split": r["split"]} for r in receipts()]


@st.cache_data
def rgb_image(path):
    return Image.fromarray(load_image(path)[:, :, ::-1].copy())  # EXIF-corrected, BGR -> RGB


def num(v):
    """Widget / table value -> float or None (empty cells come back as None or NaN)."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return None
    return round(float(v), 2)


# ------------------------------------------------------------------ receipt order and filters
RECS = all_receipts()
BY_ID = {r["id"]: r for r in RECS}
ORDER = [r["id"] for r in RECS]
STATUS = {rid: saved_status(rid, OUT) for rid in ORDER}
FILTERS = {
    "To do": lambda rid: STATUS[rid] == "todo",
    "All": lambda rid: True,
    "Flagged": lambda rid: STATUS[rid] == "flagged",
    "Done": lambda rid: STATUS[rid] == "done",
    "Validation split": lambda rid: BY_ID[rid]["split"] == "validation",
    "Test split": lambda rid: BY_ID[rid]["split"] == "test",
}

ss.setdefault("ver", 0)  # bumped after each save or reset, so the form is rebuilt from the file
ss.setdefault("filter", "To do")
ss.setdefault("current", next((r for r in ORDER if STATUS[r] == "todo"), ORDER[0]))


def in_filter(rid):
    return FILTERS[ss.filter](rid)


def neighbour(step):
    """Next (step=1) or previous (step=-1) receipt in the current filter, or None at the end."""
    i = ORDER.index(ss.current)
    seq = ORDER[i + 1:] if step > 0 else ORDER[:i][::-1]
    return next((rid for rid in seq if in_filter(rid)), None)


def key(name):
    return f"{ss.current}_{ss.ver}_{name}"


# ------------------------------------------------------------------ current label from the widgets
def edited_items(base_items):
    """Apply the data editor's edit log (edited / deleted / added rows) to the items it started from.
    Reading the log, not the editor's last return value, keeps an edit made just before a button click."""
    delta = ss.get(key("items"), {})
    rows = [dict(r) for r in base_items]
    for i, change in delta.get("edited_rows", {}).items():
        rows[int(i)].update(change)
    deleted = set(delta.get("deleted_rows", []))
    rows = [r for i, r in enumerate(rows) if i not in deleted] + [dict(r) for r in delta.get("added_rows", [])]
    items = []
    for r in rows:
        name, price = str(r.get("name") or "").strip(), num(r.get("price"))
        if not name and price is None:
            continue  # empty row
        qty = num(r.get("qty"))
        qty = 1 if qty is None else (int(qty) if qty == int(qty) else qty)
        items.append({"name": name, "qty": qty, "price": price})
    return items


def current_label(base):
    return {**base, "items": edited_items(base["items"]),
            **{f: num(ss.get(key(f), base.get(f))) for f in FIELDS},
            "flags": ss.get(key("flags"), base.get("flags", [])),
            "notes": ss.get(key("notes"), base.get("notes", ""))}


def content(label):
    return {k: label.get(k) for k in ["items", *FIELDS, "flags", "notes"]}


# ------------------------------------------------------------------ button actions (run before the rerun)
def save(status, advance):
    label = current_label(ss.base)
    label["status"] = status
    save_label(label, OUT)
    STATUS[ss.current] = status
    ss.toast = f"Saved {ss.current} as {status}"
    ss.ver += 1
    ss.pop("force_draft", None)
    nxt = neighbour(1) if advance else None
    if advance and nxt is None:
        ss.toast += " (last receipt in this filter)"
    if nxt:
        ss.current = nxt


def move(step):
    nxt = neighbour(step)
    if nxt:
        ss.current, ss.ver = nxt, ss.ver + 1
        ss.pop("force_draft", None)


def jump(widget_key):
    ss.current, ss.ver = ss[widget_key], ss.ver + 1
    ss.pop("force_draft", None)


def reset_to_draft():
    ss.force_draft, ss.ver = ss.current, ss.ver + 1


# ------------------------------------------------------------------ sidebar: progress, filter, overview
visible = [rid for rid in ORDER if in_filter(rid)]
if ss.current not in visible and visible:
    ss.current = visible[0]

counts = {s: sum(v == s for v in STATUS.values()) for s in ["done", "flagged", "todo"]}
val_ids = [r for r in ORDER if BY_ID[r]["split"] == "validation"]
with st.sidebar:
    st.subheader("Progress")
    st.progress((counts["done"] + counts["flagged"]) / len(ORDER),
                text=f"{counts['done']} done · {counts['flagged']} flagged · {counts['todo']} to do")
    st.caption(f"Validation: {sum(STATUS[r] != 'todo' for r in val_ids)}/{len(val_ids)} labeled")
    st.radio("Show", list(FILTERS), key="filter")
    if visible:
        jk = f"jump_{ss.current}_{ss.ver}"
        st.selectbox(f"Receipt ({len(visible)} in this filter)", visible, index=visible.index(ss.current),
                     key=jk, on_change=jump, args=(jk,),
                     format_func=lambda r: f"{r} · {BY_ID[r]['split']} · {STATUS[r]}")
    st.caption(f"Labels folder: `{OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT}`")

if "toast" in ss:
    st.toast(ss.pop("toast"))
if not visible:
    st.info("No receipts in this filter. Pick another filter in the sidebar.")
    st.stop()

# ------------------------------------------------------------------ load the receipt shown
rid = ss.current
rec = BY_ID[rid]
if ss.get("force_draft") == rid:
    base = (read_json(PREFILL_DIR / f"{rid}.json") if USE_PREFILL else None) or blank_label(rid)
else:
    base = load_label(rid, OUT, use_prefill=USE_PREFILL)
ss.base = base
img = rgb_image(rec["path"])

st.markdown(f"### Receipt {rid} &nbsp; <small>{rec['split']} · saved: **{STATUS[rid]}** · "
            f"opened from: {'saved label' if STATUS[rid] != 'todo' and ss.get('force_draft') != rid else base.get('prefill_source', 'none') + ' draft'}</small>",
            unsafe_allow_html=True)
left, right = st.columns([5, 6], gap="large")

# ------------------------------------------------------------------ left: zoomable photo
with left:
    z1, z2, z3 = st.columns(3)
    zoom = z1.select_slider("Zoom", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0], key="zoom")
    xp = z2.slider("Left ↔ right", 0, 100, 50, key="xpos", disabled=zoom == 1.0)
    yp = z3.slider("Top ↕ bottom", 0, 100, 0, key="ypos", disabled=zoom == 1.0)
    W, H = img.size
    cw, ch = W / zoom, H / zoom
    x0, y0 = (W - cw) * xp / 100, (H - ch) * yp / 100
    box = (int(x0), int(y0), int(x0 + cw), int(y0 + ch))
    st.image(img.crop(box), width="stretch")
    with st.sidebar:
        overview = img.copy()
        ImageDraw.Draw(overview).rectangle(box, outline=(230, 30, 30), width=max(3, W // 150))
        st.image(overview, caption="Whole receipt (red box = zoomed area)", width="stretch")

# ------------------------------------------------------------------ right: the form
with right:
    with st.expander("Transcription (reference only; the photo wins)"):
        st.code(rec["text"] or "(no transcription for this image)", language=None)

    st.markdown("**Items** · one row per priced line; discounts are negative; add rows at the bottom")
    items_df = pd.DataFrame(base["items"], columns=["name", "qty", "price"])
    st.data_editor(items_df, key=key("items"), num_rows="dynamic", hide_index=True, width="stretch",
                   column_config={
                       "name": st.column_config.TextColumn("Name", width="large"),
                       "qty": st.column_config.NumberColumn("Qty", min_value=0, step=1, width="small"),
                       "price": st.column_config.NumberColumn("Line price", step=0.01, format="%.2f",
                                                              width="small")})

    c1, c2 = st.columns(2)  # left column in receipt order; total and handwritten tip on the right
    for col, fields in [(c1, ["subtotal", "tax", "tip"]), (c2, ["total", "tip_handwritten"])]:
        for f in fields:
            col.number_input(FIELD_NAMES[f], value=base.get(f), step=0.01, format="%.2f",
                             key=key(f), placeholder="not printed")

    # live arithmetic check: a typo detector, not a rule (GUIDE rule 13)
    lab = current_label(base)
    prices = [it["price"] for it in lab["items"] if it["price"] is not None]
    items_sum = round(sum(prices), 2)
    sub, tax, tip, total = lab["subtotal"], lab["tax"], lab["tip"], lab["total"]
    a, b = st.columns(2)
    if sub is None:
        a.info(f"Items sum {items_sum:.2f} (no subtotal to compare)")
    elif abs(items_sum - sub) < EPS:
        a.success(f"Items sum {items_sum:.2f} = subtotal ✓")
    else:
        a.error(f"Items sum {items_sum:.2f} ≠ subtotal {sub:.2f} (off by {items_sum - sub:+.2f})")
    if total is None:
        b.info("No total to compare")
    else:
        expect = (sub if sub is not None else items_sum) + (tax or 0) + (tip or 0)
        if abs(expect - total) < EPS:
            b.success(f"{'Subtotal' if sub is not None else 'Items'} + tax + tip = total {total:.2f} ✓")
        else:
            b.error(f"{'Subtotal' if sub is not None else 'Items'} + tax + tip = {expect:.2f} ≠ total "
                    f"{total:.2f} (off by {expect - total:+.2f})")
    problems = [f"row {i + 1}: missing {'name' if not it['name'] else 'price'}"
                for i, it in enumerate(lab["items"]) if not it["name"] or it["price"] is None]
    if problems:
        st.warning("Incomplete items: " + "; ".join(problems))

    f1, f2 = st.columns([1, 1])
    f1.multiselect("Flags", FLAG_OPTIONS, default=base.get("flags", []), key=key("flags"))
    f2.text_input("Notes", value=base.get("notes", ""), key=key("notes"))

    if content(lab) != content(base):
        st.caption("✏️ Unsaved changes: moving to another receipt without saving discards them.")

    b1, b2, b3, b4, b5 = st.columns([2, 2, 1, 1, 1])
    b1.button("Save & next", type="primary", on_click=save, args=("done", True), width="stretch")
    b2.button("Save as flagged & next", on_click=save, args=("flagged", True), width="stretch",
              help="Something prevents a complete label (cut off, unreadable, ...). Pick a flag first.")
    b3.button("◀ Prev", on_click=move, args=(-1,), width="stretch", disabled=neighbour(-1) is None)
    b4.button("Next ▶", on_click=move, args=(1,), width="stretch", disabled=neighbour(1) is None)
    b5.button("Reset", on_click=reset_to_draft, width="stretch",
              help="Reload the draft into the form (not saved until you press Save)")
