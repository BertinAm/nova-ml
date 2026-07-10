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


# ── BEAC official (beac.int) — authoritative note scans + coin PDFs ──────
BEAC_NOTE_PAGES = [
    # (url, series tag)
    ("https://www.beac.int/billets-pieces/signes-monetaires/billets-de-gamme-2020/", "g2020"),
    ("https://www.beac.int/billets-pieces/signes-monetaires/billets-de-gamme-2002/", "g2002"),
]
BEAC_COIN_PDFS = [
    ("https://www.beac.int/wp-content/uploads/2016/11/gamme-de-pieces-2006.pdf", "g2006"),
    ("https://www.beac.int/wp-content/uploads/2025/04/NPM-BEAC-2024.pdf", "g2024"),
]
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
SIZE_SUFFIX = re.compile(r"-\d+x\d+(?=\.(?:jpg|jpeg|png|webp)$)", re.I)


def beac_note_denom(url: str) -> int | None:
    """Filename like '10-000-R-250x128.png' or '500-V.jpg' -> denomination."""
    stem = Path(url).stem
    stem = SIZE_SUFFIX.sub("", stem + Path(url).suffix).rsplit(".", 1)[0]
    digits = re.sub(r"\D", "", stem)
    return int(digits) if digits and int(digits) in DENOMS else None


def save_image(img: Image.Image, class_dir: Path, name: str,
               seen_hashes: set[str], totals: dict[str, int], cls: str) -> None:
    digest = hashlib.sha256(img.tobytes()).hexdigest()[:16]
    if digest in seen_hashes:
        return
    seen_hashes.add(digest)
    if max(img.size) > 1600:
        scale = 1600 / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    class_dir.mkdir(parents=True, exist_ok=True)
    img.save(class_dir / f"{name}_{digest}.jpg", quality=92)
    totals[cls] = totals.get(cls, 0) + 1


def scrape_beac(out: Path, seen_hashes: set[str], totals: dict[str, int]) -> None:
    # Notes: official recto/verso scans, thumbnail suffix stripped for full size
    for page_url, tag in BEAC_NOTE_PAGES:
        try:
            r = requests.get(page_url, headers=BROWSER_UA, timeout=30)
            r.raise_for_status()
        except Exception as exc:
            print(f"BEAC page failed {page_url}: {exc}")
            continue
        thumbs = set(re.findall(
            r'src="(https://www\.beac\.int/wp-content/uploads/[^"]+\.(?:jpg|jpeg|png|webp))"',
            r.text, re.I))
        for thumb in sorted(thumbs):
            if "drapeau" in thumb or "Logo" in thumb or "logo" in thumb:
                continue
            denom = beac_note_denom(thumb)
            if denom is None or denom < 500:
                continue
            full = SIZE_SUFFIX.sub("", thumb)
            img = download_image(full) or download_image(thumb)
            if img is None:
                continue
            cls = f"fcfa_note_{denom}"
            save_image(img, out / cls, f"{cls}_beac_{tag}", seen_hashes, totals, cls)
            time.sleep(0.5)
        print(f"BEAC notes {tag}: cumulative {sum(totals.values())} images")

    # Coins: extract embedded images from the official series PDFs.
    # PDFs aren't labelled per denomination -> save to a review folder;
    # a human sorts them into fcfa_coin_* (minutes of work, few images).
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not installed — skipping BEAC coin PDFs")
        return
    review = out.parent / "review_coins_from_beac_pdfs"
    review.mkdir(parents=True, exist_ok=True)
    for pdf_url, tag in BEAC_COIN_PDFS:
        try:
            r = requests.get(pdf_url, headers=BROWSER_UA, timeout=60)
            r.raise_for_status()
            doc = fitz.open(stream=r.content, filetype="pdf")
        except Exception as exc:
            print(f"BEAC PDF failed {pdf_url}: {exc}")
            continue
        n = 0
        for page_idx in range(len(doc)):
            for img_info in doc.get_page_images(page_idx):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if min(pix.width, pix.height) < MIN_SIDE:
                        continue
                    pix.save(review / f"beac_{tag}_p{page_idx}_{xref}.png")
                    n += 1
                except Exception:
                    continue
        print(f"BEAC coin PDF {tag}: extracted {n} images -> {review} (sort manually)")


# ── Numista (primary source: obverse/reverse photos per catalogue item) ──
NUMISTA_SEARCH = "https://en.numista.com/catalogue/index.php"
# Issuer section headers we accept (Central African zone only)
NUMISTA_ISSUER_OK = re.compile(r"central african|cameroon|cameroun|beac", re.I)


def numista_photos(denom: int) -> list[tuple[str, bool]]:
    """Search Numista for one denomination. Returns [(photo_url, is_note)].

    Parses the search results page: results are grouped under <h2> issuer
    headers; each result block's photos carry class ``paper_photo`` for
    banknotes. Thumbnail URLs are rewritten to the -original full size.
    """
    photos: list[tuple[str, bool]] = []
    for page in (1, 2):
        params = {"mode": "simplifie", "p": page, "q": 50,
                  "r": f"{denom} francs BEAC"}
        try:
            r = requests.get(NUMISTA_SEARCH, params=params, headers=BROWSER_UA, timeout=30)
            r.raise_for_status()
        except Exception as exc:
            print(f"  numista search p{page} failed: {exc}")
            break
        html = r.text
        # Walk the page keeping track of the current issuer header.
        tokens = re.split(r'(<h2[^>]*>.*?</h2>|<div class="resultat_recherche">)', html)
        issuer_ok = False
        for tok in tokens:
            if tok.startswith("<h2"):
                issuer_ok = bool(NUMISTA_ISSUER_OK.search(tok)) and not EXCLUDE.search(tok)
            elif tok.startswith('<div class="resultat_recherche"'):
                continue
            elif issuer_ok:
                for m in re.finditer(
                    r'<div class="photo_(?:avers|revers)([^"]*)">.*?'
                    r'src="(https://en\.numista\.com/catalogue/photos/[^"]+?)-\d+\.jpg"',
                    tok, re.S,
                ):
                    is_note = "paper_photo" in m.group(1)
                    photos.append((m.group(2) + "-original.jpg", is_note))
        if "resultat_recherche" not in html:
            break
        time.sleep(2)
    return photos


def scrape_numista(out: Path, seen_hashes: set[str], totals: dict[str, int]) -> None:
    for denom in DENOMS:
        pairs = numista_photos(denom)
        # A photo's class: notes ≥1000 always note; 25-200 always coin;
        # 500 decided by the paper_photo flag.
        for url, is_note in pairs:
            if denom >= 1000:
                cls = f"fcfa_note_{denom}"
            elif denom == 500:
                cls = "fcfa_note_500" if is_note else "fcfa_coin_500"
            else:
                if is_note:
                    continue  # no such thing as a 25-200 XAF note; misfiled
                cls = f"fcfa_coin_{denom}"
            img = download_image(url)
            if img is None:
                continue
            digest = hashlib.sha256(img.tobytes()).hexdigest()[:16]
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            class_dir = out / cls
            class_dir.mkdir(parents=True, exist_ok=True)
            if max(img.size) > 1600:
                scale = 1600 / max(img.size)
                img = img.resize((int(img.width * scale), int(img.height * scale)))
            img.save(class_dir / f"{cls}_numista_{digest}.jpg", quality=92)
            totals[cls] = totals.get(cls, 0) + 1
            time.sleep(0.5)
        print(f"numista {denom} francs: done "
              f"(cumulative: { {k: v for k, v in totals.items() if str(denom) in k} })")
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="datasets/cfa_currency_scraped")
    parser.add_argument("--skip-commons", action="store_true")
    args = parser.parse_args()

    out = Path(args.out) / "raw"
    seen_hashes: set[str] = set()
    totals: dict[str, int] = {}

    # Stage 1: BEAC official (authoritative note scans + coin PDFs)
    print("=== Stage 1: BEAC official ===")
    scrape_beac(out, seen_hashes, totals)

    # Stage 2: Numista catalogue (volume: many photos per type)
    print("\n=== Stage 2: Numista ===")
    scrape_numista(out, seen_hashes, totals)

    print("\n=== Scrape summary ===")
    for k in sorted(totals):
        print(f"  {k:18s} {totals[k]}")
    print(f"Total: {sum(totals.values())} images -> {out}")
    if args.skip_commons:
        return

    # Stage 3 (weak): Wikimedia Commons categories — yields very little for
    # XAF but free-licensed, so keep as a bonus pass.
    print("\n=== Stage 3: Wikimedia Commons ===")
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
        print(f"  {cls:18s} {len(titles)} candidates")

    # 3. Resolve URLs and download
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
