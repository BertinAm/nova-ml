"""Bootstrap a CFA franc (XAF) currency image dataset from Wikimedia Commons.

Strategy: enumerate Commons *categories* for Central African CFA franc money
(precise, one API call per category, avoids search rate limits), classify
each file into a denomination class from its title, download, dedupe.

Classes (10): notes 500/1000/2000/5000/10000 FCFA + coins 25/50/100/200/500.
The 500 FCFA exists as BOTH note and coin — separate classes.

Output layout (raw, pre-augmentation):
    <out>/raw/<class_name>/*.jpg

    python scripts/scrape_cfa_images.py --out datasets/cfa_currency_scraped
"""
import argparse
import hashlib
import io
import re
import time
from pathlib import Path

import requests
from PIL import Image

UA = {"User-Agent": "NOVA-assistive-tech-dataset/1.0 (academic research, Univ. of Buea)"}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Candidate Commons categories (checked in order; missing ones are skipped).
CATEGORIES = [
    "Category:Banknotes of the Central African CFA franc",
    "Category:Coins of the Central African CFA franc",
    "Category:Central African CFA franc",
    "Category:Money of Cameroon",
    "Category:Coins of Cameroon",
    "Category:Banknotes of Cameroon",
]

# Reject files that are clearly the *West* African CFA (BCEAO/XOF) — same
# denominations, different currency zone.
EXCLUDE = re.compile(r"bceao|ouest|west\s*afric|xof|c[ôo]te d.ivoire|s[ée]n[ée]gal|"
                     r"benin|b[ée]nin|burkina|mali|niger|togo|guin[ée]e.bissau", re.I)

DENOMS = [10000, 5000, 2000, 1000, 500, 200, 100, 50, 25]
COIN_HINT = re.compile(r"coin|pi[èe]ce|munt|moneta|moneda", re.I)
NOTE_HINT = re.compile(r"banknote|billet|note|banconota|billete", re.I)

MIN_SIDE = 200


def api_get(params: dict, retries: int = 5) -> dict:
    """GET with exponential backoff on 429/5xx."""
    delay = 2.0
    for attempt in range(retries):
        r = requests.get(COMMONS_API, params=params, headers=UA, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            print(f"    HTTP {r.status_code}, backing off {delay:.0f}s...")
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
    raise RuntimeError(f"API failed after {retries} retries")


def category_files(category: str) -> list[str]:
    """All file titles in a category (follows pagination, 1 level deep)."""
    titles: list[str] = []
    params = {
        "action": "query", "format": "json", "list": "categorymembers",
        "cmtitle": category, "cmtype": "file", "cmlimit": 500,
    }
    while True:
        data = api_get(params)
        members = data.get("query", {}).get("categorymembers", [])
        titles += [m["title"] for m in members]
        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)
        time.sleep(1)
    return titles


def subcategories(category: str) -> list[str]:
    params = {
        "action": "query", "format": "json", "list": "categorymembers",
        "cmtitle": category, "cmtype": "subcat", "cmlimit": 500,
    }
    data = api_get(params)
    return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]


def image_urls(titles: list[str]) -> dict[str, str]:
    """title -> direct URL, batched 50 at a time."""
    urls: dict[str, str] = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        params = {
            "action": "query", "format": "json", "titles": "|".join(batch),
            "prop": "imageinfo", "iiprop": "url|size",
        }
        data = api_get(params)
        for page in data.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("url") and min(info.get("width", 0), info.get("height", 0)) >= MIN_SIDE:
                urls[page["title"]] = info["url"]
        time.sleep(1)
    return urls


def classify(title: str) -> str | None:
    """Map a file title to a class name, or None if ambiguous/excluded."""
    if EXCLUDE.search(title):
        return None
    # Find the denomination: largest matching number wins (so "10000" isn't
    # misread as "1000" or "100").
    text = title.replace(" ", "").replace(".", "").replace(",", "")
    denom = next((d for d in DENOMS if str(d) in text), None)
    if denom is None:
        return None
    is_coin = bool(COIN_HINT.search(title))
    is_note = bool(NOTE_HINT.search(title))
    if denom >= 1000:
        return f"fcfa_note_{denom}"          # only notes exist ≥ 1000
    if denom == 500:
        if is_coin:
            return "fcfa_coin_500"
        if is_note:
            return "fcfa_note_500"
        return None                          # ambiguous 500 — skip
    # 25/50/100/200 are coins only
    return f"fcfa_coin_{denom}"


def download_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        return img if min(img.size) >= MIN_SIDE else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="datasets/cfa_currency_scraped")
    args = parser.parse_args()

    out = Path(args.out) / "raw"

    # 1. Collect candidate file titles from categories (+1 level of subcats)
    all_titles: set[str] = set()
    for cat in CATEGORIES:
        try:
            files = category_files(cat)
            print(f"{cat}: {len(files)} files")
            all_titles.update(files)
            for sub in subcategories(cat):
                sub_files = category_files(sub)
                if sub_files:
                    print(f"  {sub}: {len(sub_files)} files")
                all_titles.update(sub_files)
                time.sleep(1)
        except Exception as exc:
            print(f"{cat}: skipped ({exc})")
        time.sleep(1)
    print(f"\nTotal candidate files: {len(all_titles)}")

    # 2. Classify by title
    classified: dict[str, list[str]] = {}
    for title in sorted(all_titles):
        cls = classify(title)
        if cls:
            classified.setdefault(cls, []).append(title)
    for cls, titles in sorted(classified.items()):
        print(f"  {cls:ter 18s} {len(titles)} candidates" if False else f"  {cls:18s} {len(titles)} candidates")

    # 3. Resolve URLs and download
    seen_hashes: set[str] = set()
    totals: dict[str, int] = {}
    for cls, titles in sorted(classified.items()):
        class_dir = out / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        urls = image_urls(titles)
        count = 0
        for title, url in urls.items():
            img = download_image(url)
            if img is None:
                continue
            digest = hashlib.sha256(img.tobytes()).hexdigest()[:16]
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            if max(img.size) > 1600:
                scale = 1600 / max(img.size)
                img = img.resize((int(img.width * scale), int(img.height * scale)))
            img.save(class_dir / f"{cls}_{digest}.jpg", quality=92)
            count += 1
            time.sleep(0.3)
        totals[cls] = count
        print(f"{cls}: downloaded {count}")

    print("\n=== Scrape summary ===")
    for k in sorted(totals):
        print(f"  {k:18s} {totals[k]}")
    print(f"Total: {sum(totals.values())} images -> {out}")


if __name__ == "__main__":
    main()
