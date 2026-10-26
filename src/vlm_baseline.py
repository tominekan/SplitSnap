"""VLM reference baseline: one Gemini call per receipt photo, scored with our metrics.

This is a reference point (how far is the classic pipeline from a modern vision-language model?),
not part of the SplitSnap pipeline, which stays LLM-free (DECISIONS.md D03).

    python src/vlm_baseline.py                       # SRD test, then CORD test
    python src/vlm_baseline.py --datasets srd --limit 25
    python src/vlm_baseline.py --score-only          # rescore cached answers, no API calls

The API key (GEMINI_API_KEY) is read from the environment or .env (gitignored). Raw answers are cached in
data/cache/vlm/<model>/<dataset>/<id>.json, so a run that hits a rate limit can be resumed.
Outputs: results/predictions/VLM <model>/..., results/vlm_summary_<model>.md, results/logs/vlm_<model>.txt.
"""
import argparse
import io
import json
import os
import re
import sys
import time

from PIL import Image

from common import CACHE, RESULTS, ROOT, parse_amount
from data import load_cord, load_image, load_srd, srd_split, srd_total_from_text
from metrics import score_cord, score_srd
from pipeline import checksum

MAX_SIDE = 2048  # downscale before upload; receipt text stays readable and requests stay small
BEST_CLASSIC = ["+rectify+preprocess+layout", "EasyOCR: +rectify+layout (no preprocess)"]

PROMPT = """You are reading a photo of a restaurant receipt. Return JSON with:
- "raw_text": the full receipt text, transcribed line by line top to bottom, exactly as printed.
- "items": every purchased line that has its own price, in order. Priced add-ons or modifiers are
  separate items. Skip lines without a price, and skip subtotal, tax, service, tip, total, payment,
  cash and change lines. For each item: "name" as printed, "qty" (1 if not shown), and "price" as the
  line total printed on that line.
- "subtotal", "tax", "tip" (tip, gratuity or service charge), "total": as printed, or null if absent.
Write every amount as a string exactly as printed (keep separators, drop currency symbols).
Do not compute or guess values that are not printed."""

SCHEMA = {
    "type": "object",
    "properties": {
        "raw_text": {"type": "string"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "qty": {"type": "number"}, "price": {"type": "string"}},
            "required": ["name", "price"]}},
        "subtotal": {"type": "string", "nullable": True},
        "tax": {"type": "string", "nullable": True},
        "tip": {"type": "string", "nullable": True},
        "total": {"type": "string", "nullable": True},
    },
    "required": ["raw_text", "items", "subtotal", "tax", "tip", "total"],
}


def api_key(name):
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for ln in env.read_text().splitlines():
            if ln.startswith(f"{name}="):
                return ln.split("=", 1)[1].strip().strip("'\"")
    sys.exit(f"No {name} in the environment or .env")


def jpeg_bytes(path):
    im = Image.fromarray(load_image(path)[:, :, ::-1])  # EXIF-corrected, BGR -> RGB
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


class RetryableError(Exception):
    def __init__(self, code, msg, wait=None):
        super().__init__(f"{code}: {msg}")
        self.code, self.wait = code, wait


def with_backoff(call, max_tries=10):
    """Run call() with backoff on rate limits / overload. Returns (answer text, seconds)."""
    for attempt in range(max_tries):
        t0 = time.perf_counter()
        try:
            return call(), time.perf_counter() - t0
        except RetryableError as e:
            if attempt == max_tries - 1:
                raise
            wait = e.wait or min(15 * 2 ** attempt, 300)
            print(f"  {e.code}, waiting {wait:.0f}s", flush=True)
            time.sleep(wait)


def gemini_asker(model):
    from google import genai
    from google.genai import errors, types
    client = genai.Client(api_key=api_key("GEMINI_API_KEY"), http_options=types.HttpOptions(timeout=180_000))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA,
                                      temperature=0)

    def call(path):
        part = types.Part.from_bytes(data=jpeg_bytes(path), mime_type="image/jpeg")
        try:
            return client.models.generate_content(model=model, contents=[part, PROMPT], config=cfg).text
        except errors.APIError as e:
            if e.code in (429, 500, 503, 504):
                m = re.search(r"retry in ([\d.]+)s", str(e))
                raise RetryableError(e.code, str(e)[:200], float(m.group(1)) + 1 if m else None)
            raise
    return call


def to_schema(ans, currency):
    amt = lambda s: parse_amount(s, currency) if s not in (None, "") else None
    items = []
    for it in ans.get("items") or []:
        p = amt(it.get("price"))
        if p is not None and it.get("name"):
            items.append({"name": it["name"], "qty": it.get("qty") or 1, "price": p})
    out = {"items": items, **{k: amt(ans.get(k)) for k in ["subtotal", "tax", "tip", "total"]},
           "raw_text": ans.get("raw_text") or ""}
    out["checksum"] = checksum(out)
    return out


def run(ask, dataset, recs, cache_dir, rpm):
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in recs if not (cache_dir / f"{r['id']}.json").exists()]
    print(f"{dataset}: {len(recs) - len(todo)} cached, {len(todo)} to request", flush=True)
    gap = 60 / rpm
    for n, r in enumerate(todo, 1):
        t_start = time.time()
        try:
            text, secs = with_backoff(lambda: ask(r["path"]))
        except Exception as e:  # daily quota or a hard error: stop, keep what we have
            print(f"  stopped at {r['id']}: {str(e)[:200]}", flush=True)
            return False
        (cache_dir / f"{r['id']}.json").write_text(json.dumps({"answer": text, "runtime_s": secs}))
        print(f"  [{n}/{len(todo)}] {r['id']} {secs:.1f}s", flush=True)
        time.sleep(max(0.0, gap - (time.time() - t_start)))
    return True


def load_answers(cache_dir, ids, currency):
    preds = {}
    for i in ids:
        f = cache_dir / f"{i}.json"
        if not f.exists():
            continue
        c = json.loads(f.read_text())
        try:
            ans = json.loads(c["answer"])
        except (json.JSONDecodeError, TypeError):
            ans = {}  # unparseable answer counts as an empty prediction
        preds[i] = {**to_schema(ans, currency), "runtime_s": c["runtime_s"]}
    return preds


def classic_preds(config, dataset, ids):
    d = RESULTS / "predictions" / config / dataset
    return {i: json.loads((d / f"{i}.json").read_text()) for i in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--datasets", nargs="+", default=["srd", "cord"], choices=["srd", "cord"])
    ap.add_argument("--limit", type=int, help="only the first N receipts of each dataset")
    ap.add_argument("--rpm", type=float, default=8, help="max requests per minute (free tier)")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()

    cord = load_cord("test")
    _, srd_test = srd_split(load_srd())
    recs = {"cord": [{"id": r["id"], "path": r["path"]} for r in cord],
            "srd": [{"id": r["id"], "path": r["path"]} for r in srd_test]}
    if a.limit:
        recs = {k: v[:a.limit] for k, v in recs.items()}
    safe = re.sub(r"[/:]", "_", a.model)  # model name safe for file paths
    cache = CACHE / "vlm" / safe

    if not a.score_only:
        ask = gemini_asker(a.model)
        for ds in a.datasets:
            if not run(ask, ds, recs[ds], cache / ds, a.rpm):
                break

    # Score on the receipts that have an answer, and score the best classic configs on the same set.
    name = f"VLM {safe}"
    out_dir = RESULTS / "predictions" / name
    L = [f"# VLM reference baseline ({a.model})", "",
         "Generated by `python src/vlm_baseline.py`. One API call per photo (EXIF-corrected, long side "
         f"<= {MAX_SIDE} px), temperature 0, JSON output, amounts parsed with the same rules as our parsers. "
         "Classic rows are re-scored on exactly the same receipts, so the comparison is like for like.", ""]
    cord_gt = {r["id"]: r["gt"] for r in cord}
    cp = load_answers(cache / "cord", [r["id"] for r in recs["cord"]], "IDR")
    if cp:
        L += [f"## CORD test ({len(cp)} of {len(cord)} receipts)", "",
              "| config | item P | item R | item F1 | subtotal acc | tax acc | total acc | checksum pass | s/img |",
              "|---|---|---|---|---|---|---|---|---|"]
        gts = {i: cord_gt[i] for i in cp}
        for cfg, preds in [(name, cp)] + [(c, classic_preds(c, "cord", cp)) for c in BEST_CLASSIC]:
            s, _ = score_cord(preds, gts)
            rt = sum(p["runtime_s"] for p in preds.values()) / len(preds)
            L.append(f"| {cfg} | {s['item_P']:.3f} | {s['item_R']:.3f} | **{s['item_F1']:.3f}** | "
                     f"{s['subtotal_acc']:.3f} | {s['tax_acc']:.3f} | {s['total_acc']:.3f} | "
                     f"{s['checksum_pass']:.3f} | {rt:.2f} |")
        L.append("")
    texts = {r["id"]: r["text"] for r in srd_test if r["text"]}
    sp = load_answers(cache / "srd", [r["id"] for r in recs["srd"]], "USD")
    if sp:
        t = {i: v for i, v in texts.items() if i in sp}
        tot = {i: srd_total_from_text(v) for i, v in t.items()}
        L += [f"## SRD test ({len(sp)} of {len(srd_test)} images; {len(t)} with text, "
              f"{sum(v is not None for v in tot.values())} with a usable total)", "",
              "| config | CER (mean) | CER (median) | total acc | s/img |", "|---|---|---|---|---|"]
        for cfg, preds in [(name, sp)] + [(c, classic_preds(c, "srd", sp)) for c in BEST_CLASSIC]:
            s = score_srd(preds, t, tot)
            rt = sum(p["runtime_s"] for p in preds.values()) / len(preds)
            L.append(f"| {cfg} | **{s['CER']:.3f}** | {s['CER_median']:.3f} | {s['total_acc']:.3f} | {rt:.2f} |")
        L.append("")
    L.append("VLM s/img is API round-trip time for one request at a time; classic s/img is local "
             "wall-clock with 8 images in parallel, so runtimes are not directly comparable.")

    for ds, preds in [("cord", cp), ("srd", sp)]:
        (out_dir / ds).mkdir(parents=True, exist_ok=True)
        for i, p in preds.items():
            (out_dir / ds / f"{i}.json").write_text(json.dumps(p, indent=1, ensure_ascii=False))
    md = "\n".join(L) + "\n"
    (RESULTS / f"vlm_summary_{safe}.md").write_text(md)
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    (RESULTS / "logs" / f"vlm_{safe}.txt").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
