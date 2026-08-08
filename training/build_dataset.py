"""Synthesise a labelled dataset for the category classifier.

Only 4 of the 15 categories appear in the 23 real invoices, so there is no usable
training signal in the database alone. What the project *does* have is 153
keyword->category rules in category.py, written against real Turkish invoices, plus
15 category glosses. Those are templated into invoice-shaped phrases here.

Every example is `"<vendor>. <items>"` -- the exact shape classifier.py builds at
inference time. Training and serving must see the same string shape or the embedding
distance means something different in each.

    python training/build_dataset.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from malimusavir.category import CATEGORY_GLOSS, KEYWORD_RULES  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "categories.jsonl"

#: Deterministic output -- a dataset that changes between runs makes an accuracy
#: change impossible to attribute to the change you actually made.
SEED = 20260808

#: Realistic Turkish business names per category. These deliberately cover the *kinds*
#: of business in the held-out evaluation set (nalbur, kuafor, optik, pastane, oto
#: servis, fidanci) WITHOUT reusing the exact test strings -- the point is to teach the
#: category, not to memorise the test.
VENDORS: dict[str, tuple[str, ...]] = {
    "telekom": (
        "Turkcell İletişim Hizmetleri A.Ş.", "Vodafone Telekomünikasyon A.Ş.",
        "Türk Telekom A.Ş.", "TTNET A.Ş.", "Turkcell Superonline İletişim Hizmetleri A.Ş.",
        "Millenicom Telekomünikasyon", "Netspeed Fiber İnternet Ltd. Şti.",
    ),
    "enerji": (
        "Enerjisa Enerji A.Ş.", "BEDAŞ Boğaziçi Elektrik", "İGDAŞ Doğalgaz Dağıtım",
        "İSKİ Su ve Kanalizasyon İdaresi", "AYEDAŞ Anadolu Yakası Elektrik",
        "Aksa Doğalgaz Dağıtım A.Ş.", "ASKİ Ankara Su İdaresi",
    ),
    "abonelik": (
        "Netflix International B.V.", "Spotify AB", "Microsoft Ireland Operations",
        "Google Ireland Limited", "Adobe Systems Software Ireland",
        "JetBrains s.r.o.", "Amazon Web Services EMEA",
    ),
    "elektronik": (
        "Teknosa İç ve Dış Ticaret A.Ş.", "Vatan Bilgisayar San. Tic. A.Ş.",
        "MediaMarkt Turkey Ticaret Ltd. Şti.", "Kuzey Bilişim Teknolojileri A.Ş.",
        "Robotistan Elektronik Ticaret A.Ş.", "Direnç Elektronik San. Tic. Ltd. Şti.",
        "İNT-EL İnternational Elektronik Sanayi ve Ticaret Limited Şirketi",
    ),
    "ev": (
        "IKEA Mobilya Sanayi Ticaret A.Ş.", "Koçtaş Yapı Marketleri Tic. A.Ş.",
        "Bauhaus Yapı Market", "Doğtaş Mobilya San. Tic. A.Ş.",
        "Merkez Hırdavat ve Nalburiye Ltd. Şti.", "Yılmaz Nalburiye Tic. Ltd. Şti.",
        "Anadolu Hırdavat San. Tic.", "Bosch Ev Aletleri Servisi",
        "Arçelik Yetkili Bayii", "Beyaz Eşya Dünyası Ltd. Şti.",
    ),
    "market": (
        "Migros Ticaret A.Ş.", "CarrefourSA Carrefour Sabancı Ticaret Merkezi A.Ş.",
        "A101 Yeni Mağazacılık A.Ş.", "BİM Birleşik Mağazalar A.Ş.",
        "ŞOK Marketler Ticaret A.Ş.", "Getir Perakende Lojistik A.Ş.",
        "Ege Zeytincilik Ltd. Şti.", "Anadolu Gıda Sanayi A.Ş.",
    ),
    "yeme-içme": (
        "Kahve Dünyası Gıda San. Tic. A.Ş.", "Starbucks Coffee Turkey",
        "Yemeksepeti Elektronik Hizmetler A.Ş.", "Divan Pastaneleri A.Ş.",
        "Yıldız Pastacılık ve Unlu Mamüller", "Sabahattin Restoran İşletmeciliği",
        "Simit Sarayı Yatırım ve Ticaret A.Ş.", "Nazar Lokantası Ltd. Şti.",
    ),
    "giyim": (
        "LC Waikiki Mağazacılık Hizmetleri Tic. A.Ş.", "DeFacto Perakende Ticaret A.Ş.",
        "Koton Mağazacılık Tekstil San. Tic. A.Ş.", "Boyner Perakende ve Tekstil A.Ş.",
        "Flo Mağazacılık ve Pazarlama A.Ş.", "Yıldız Tekstil San. ve Tic. A.Ş.",
        "Mavi Giyim Sanayi ve Ticaret A.Ş.",
    ),
    "ulaşım": (
        "OPET Petrolcülük A.Ş.", "Shell & Turcas Petrol A.Ş.",
        "Petrol Ofisi A.Ş.", "Türk Hava Yolları A.O.",
        "Güven Oto Bakım ve Servis Hizmetleri", "Star Oto Servis San. Tic. Ltd. Şti.",
        "BiTaksi Teknoloji A.Ş.", "HGS Otoyol Geçiş Sistemi",
        "Mavi Kargo Lojistik A.Ş.",
    ),
    "sağlık": (
        "Selin Eczanesi", "Acıbadem Sağlık Hizmetleri A.Ş.",
        "Memorial Sağlık Grubu", "Deniz Optik Gözlük San. Tic.",
        "Vizyon Optik ve Gözlük Merkezi", "Medikal Destek Sağlık Ürünleri Ltd. Şti.",
        "Özel Anadolu Diş Kliniği", "Merkez Laboratuvar Hizmetleri",
    ),
    "sigorta": (
        "Anadolu Anonim Türk Sigorta Şirketi", "Allianz Sigorta A.Ş.",
        "AXA Sigorta A.Ş.", "Aksigorta A.Ş.", "Mapfre Sigorta A.Ş.",
        "Sompo Sigorta A.Ş.", "Groupama Sigorta A.Ş.",
    ),
    "kitap-medya": (
        "D&R Doğan Müzik Kitap Mağazacılık", "Remzi Kitabevi A.Ş.",
        "İdefix İnternet Ticaret A.Ş.", "Yapı Kredi Yayınları",
        "Özgür Yayıncılık Ltd. Şti.", "Can Sanat Yayınları A.Ş.",
        "Valve Corporation Steam", "Sony Interactive Entertainment",
    ),
    "ofis": (
        "Kırtasiye Dünyası Tic. Ltd. Şti.", "Ofix Bilişim ve Kırtasiye A.Ş.",
        "Faber-Castell Kırtasiye Ürünleri", "Bilgi Toner ve Kartuş Dolum Merkezi",
        "Ofis Market Büro Malzemeleri Ltd. Şti.", "Barış Ofis Mobilyaları",
    ),
    "hizmet": (
        "Yıldız Mühendislik ve Danışmanlık Limited Şirketi",
        "Beyaz Mali Danışmanlık Hizmetleri Ltd. Şti.",
        "Öz Hukuk Bürosu Avukatlık Ortaklığı",
        "Bahar Kuaför ve Güzellik Salonu", "Stil Erkek Kuaförü",
        "Mavi Bahçe Peyzaj ve Tasarım Ltd. Şti.", "Ege Fidancılık ve Bahçe Ürünleri",
        "Temiz İş Temizlik Hizmetleri A.Ş.", "Usta Tadilat ve Dekorasyon",
        "İZMİT Yazılım Robot Teknolojileri Ltd. Şti.",
    ),
    "diğer": (
        "Genel Ticaret Ltd. Şti.", "Çeşitli Hizmetler A.Ş.",
        "Karma Ürünler Pazarlama", "Muhtelif Tedarik Ltd. Şti.",
    ),
}

#: Item-line phrasings a keyword gets templated into. Real invoice item lines read like
#: these -- a bare keyword embeds differently from "X bedeli" on a real document.
ITEM_TEMPLATES = (
    "{kw}",
    "{kw} bedeli",
    "{kw} ücreti",
    "{kw} alışverişi",
    "{kw} hizmet bedeli",
    "1 Adet {kw}",
    "{kw} - aylık",
    "{kw} tutarı",
)


def real_invoice_examples() -> list[dict[str, str]]:
    """Ground truth from the actual database, if it exists."""
    from malimusavir import db
    from malimusavir.items import items_text

    path = Path(__file__).resolve().parent.parent / "faturalar.db"
    if not path.exists():
        return []

    conn = db.connect(path)
    rows = conn.execute(
        "SELECT vendor, category, raw_text FROM invoices "
        "WHERE category IS NOT NULL AND vendor IS NOT NULL"
    ).fetchall()
    conn.close()

    seen, out = set(), []
    for row in rows:
        items = items_text(row["raw_text"] or "", limit=200)
        text = f"{row['vendor']}. {items}".strip().rstrip(".")
        if text not in seen:
            seen.add(text)
            out.append({"text": text, "category": row["category"], "source": "real"})
    return out


def keyword_examples(rng: random.Random) -> list[dict[str, str]]:
    """Each keyword rule, templated into invoice-shaped item lines."""
    out = []
    for category, keywords in KEYWORD_RULES:
        vendors = VENDORS.get(category, VENDORS["diğer"])
        for keyword in keywords:
            kw = keyword.strip()
            for template in rng.sample(ITEM_TEMPLATES, k=4):
                item = template.format(kw=kw)
                vendor = rng.choice(vendors)
                out.append({
                    "text": f"{vendor}. {item}",
                    "category": category,
                    "source": "keyword",
                })
                # Also an item-only form: some invoices have an unhelpful vendor name
                # and all the signal lives in the line items.
                out.append({"text": item, "category": category, "source": "keyword"})
    return out


def vendor_examples(rng: random.Random) -> list[dict[str, str]]:
    """Vendor names paired with their category's own gloss terms."""
    gloss = dict(CATEGORY_GLOSS)
    out = []
    for category, vendors in VENDORS.items():
        terms = [t.strip() for t in gloss.get(category, "").replace("(", ",").replace(")", "")
                 .split(",") if t.strip()]
        for vendor in vendors:
            out.append({"text": vendor, "category": category, "source": "vendor"})
            for term in terms[:4]:
                out.append({
                    "text": f"{vendor}. {term}",
                    "category": category,
                    "source": "vendor",
                })
            out.append({
                "text": f"{vendor}. {rng.choice(ITEM_TEMPLATES).format(kw=terms[0] if terms else category)}",
                "category": category,
                "source": "vendor",
            })
    return out


def main() -> int:
    rng = random.Random(SEED)

    examples = real_invoice_examples() + keyword_examples(rng) + vendor_examples(rng)

    # Deduplicate on text; a duplicated string would otherwise weight one phrasing.
    seen, unique = set(), []
    for ex in examples:
        key = ex["text"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(ex)
    rng.shuffle(unique)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for ex in unique:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    counts = Counter(ex["category"] for ex in unique)
    sources = Counter(ex["source"] for ex in unique)
    print(f"{len(unique)} examples -> {OUT}")
    print(f"sources: {dict(sources)}\n")
    print(f"{'category':14} {'n':>5}")
    print("-" * 21)
    for category, _ in CATEGORY_GLOSS:
        n = counts.get(category, 0)
        flag = "  <- thin" if n < 30 else ""
        print(f"{category:14} {n:5}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
