"""Category inference -- the only field an LLM is trusted with.

Two reasons this is the sole LLM-extracted field:

* Category is genuinely *not on the document*. Every other field is printed text that
  regex can lift verbatim, so involving a model there only adds a chance to be wrong.
* The label set is closed. Asking a 4B model to pick one of eleven fixed strings is a
  classification task it handles reliably; asking it to invent a category produces a
  long tail of near-duplicates ("elektronik", "Elektronik ürün", "teknoloji") that
  makes `spend by category` meaningless.

A keyword pass runs first, so for recurring issuers the model is never called at all.
"""

from __future__ import annotations

import re

from . import classifier, foundry
from .items import items_text
from .normalize import fold_tr

#: Minimum classifier confidence before a prediction is used at all. Below this the
#: question falls through to the model (if enabled) or to "diğer" + review.
#:
#: Tuned against the held-out cases, where the classifier put "Deniz Optik" in sağlık
#: -- correctly -- at 0.358, and an earlier arbitrary 0.45 threw that away. With 14
#: classes, chance is 0.071, so 0.35 is still ~5x random rather than a coin flip. Note
#: what the floor does NOT do: the one genuinely wrong prediction scored 0.502 and
#: clears every threshold tested, so this guards against "no idea", not against
#: "confidently wrong".
CONFIDENCE_MIN = 0.35

#: Above this, a prediction is treated as settled. Between the two it is used but
#: flagged for review -- the same "commit the answer, admit the doubt" rule the
#: extractor follows when its arithmetic does not reconcile.
CONFIDENCE_CLEAR = 0.60

#: The closed label set, each with the gloss shown to the model. Glosses matter: with
#: bare labels a 4B model put an insurance policy under "abonelik" and a cafe receipt
#: under "diger". They also document what belongs where when reading the stats output.
CATEGORY_GLOSS: tuple[tuple[str, str], ...] = (
    ("telekom", "telefon, internet, GSM, fiber hattı faturaları"),
    ("enerji", "elektrik, su, doğalgaz faturaları"),
    ("abonelik", "dijital servis üyelikleri (Netflix, Spotify, yazılım lisansı)"),
    ("elektronik", "bilgisayar, telefon, kulaklık, kablo gibi cihaz ve donanım"),
    ("ev", "mobilya, beyaz eşya, ev gereçleri, dekorasyon"),
    ("market", "market ve gıda alışverişi"),
    ("yeme-içme", "restoran, kafe, yemek siparişi"),
    ("giyim", "kıyafet, ayakkabı, tekstil"),
    ("ulaşım", "akaryakıt, bilet, taksi, otoyol geçişi, araç bakımı"),
    ("sağlık", "eczane, hastane, muayene, ilaç, medikal"),
    ("sigorta", "sigorta poliçesi, kasko, trafik sigortası"),
    ("kitap-medya", "kitap, dergi, film, müzik, oyun"),
    ("ofis", "kırtasiye, toner, ofis malzemesi"),
    ("hizmet", "danışmanlık, muhasebe, hukuk, tamir, işçilik, nakliye"),
    ("diğer", "yukarıdakilerin hiçbirine girmiyorsa"),
)

CATEGORIES: tuple[str, ...] = tuple(label for label, _ in CATEGORY_GLOSS)

DEFAULT_CATEGORY = "diğer"

#: Folded label -> canonical label, so model output like "saglik" still resolves.
_CANONICAL = {fold_tr(c): c for c in CATEGORIES}

#: (category, folded keywords). Checked in order; first hit wins.
KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("telekom", ("turkcell", "superonline", "vodafone", "turk telekom", "ttnet",
                 "avea", "fiber internet", "mobil hat", "gsm", "aylik paket ucreti")),
    ("enerji", ("elektrik", "dogalgaz", "dogal gaz", "su faturasi", "aski", "iski",
                "izsu", "bedas", "ayedas", "enerjisa", "igdas", "cngaz")),
    ("sigorta", ("sigorta", "police", "kasko", "trafik sigortasi", "bes katki",
                 "anadolu sigorta", "allianz", "axa")),
    ("abonelik", ("netflix", "spotify", "youtube premium", "amazon prime", "abonelik",
                  "uyelik bedeli", "microsoft 365", "google one", "icloud",
                  "yazilim lisansi", "lisans bedeli")),
    ("elektronik", ("kulaklik", "bilgisayar", "laptop", "monitor", "sarj kablo",
                    "klavye", "mouse", "ssd", "harddisk", "usb", "bluetooth",
                    "televizyon", "tablet", "kamera", "powerbank")),
    ("ev", ("mobilya", "beyaz esya", "buzdolabi", "camasir makinesi", "bulasik makinesi",
            "koltuk", "yatak", "ikea", "supurge", "firin",
            # Hardware and hand tools.
            "vidalama", "bits ucu", "matkap", "tornavida", "hirdavat", "el aleti",
            "anahtar takimi", "nalbur", "vida", "civata")),
    ("market", ("migros", "carrefour", "a101", "bim ", "sok market", "getir",
                "market alisveris", "gida maddesi")),
    ("yeme-içme", ("restoran", "lokanta", "kafe", "cafe", "kahve", "yemeksepeti",
                   "trendyol yemek", "pastane", "sandvic")),
    ("giyim", ("giyim", "tekstil", "ayakkabi", "defacto", "lc waikiki", "koton",
               "mont", "tisort", "pantolon")),
    ("ulaşım", ("akaryakit", "benzin", "motorin", "opet", "shell", "petrol ofisi",
                "hgs", "ogs", "otoyol", "ucak bileti", "uber", "taksi", "arac bakim")),
    ("sağlık", ("eczane", "hastane", "ilac", "saglik", "medikal", "doktor",
                "laboratuvar", "dis klinigi", "muayene")),
    ("kitap-medya", ("kitap", "yayincilik", "yayinevi", "dergi", "roman", "steam",
                     "playstation", "bilet - sinema", "muzik")),
    ("ofis", ("kirtasiye", "toner", "kartus", "ofis malzeme", "yazici", "fotokopi")),
    ("hizmet", ("danismanlik", "muhasebe", "mali musavir", "hukuk", "avukat",
                "proje hizmeti", "bakim onarim", "montaj", "nakliye", "kargo",
                "peyzaj", "iscilik", "tadilat", "temizlik hizmeti")),
)

_PROMPT = """Bir faturayı harcama kategorisine ayırıyorsun.

Kategoriler:
{labels}

Satıcı: {vendor}

Fatura içeriği:
{excerpt}

Yukarıdaki listeden SADECE BİR kategori adı yaz. Açıklama yapma, cümle kurma.

Kategori:"""


def _match_keywords(haystack: str) -> str | None:
    for category, keywords in KEYWORD_RULES:
        if any(keyword in haystack for keyword in keywords):
            return category
    return None


def _coerce(raw: str) -> str | None:
    """Map free-form model output onto the closed label set."""
    folded = fold_tr(raw).strip()
    if folded in _CANONICAL:
        return _CANONICAL[folded]
    # The model often answers with a short phrase ("Kategori: telekom"), so look for a
    # known label inside it. Longest-first, and bounded so "ev" cannot match mid-word.
    for key in sorted(_CANONICAL, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", folded):
            return _CANONICAL[key]
    return None


def excerpt_for_classification(text: str, *, limit: int = 700) -> str:
    """The part of the invoice that says what was actually bought.

    Prefers the line-item table. Falling back to the top of the document was the
    original behaviour and it was wrong: the header is the seller's name and address,
    which carries no signal about *what* was purchased and is near-identical across
    every invoice from that issuer.
    """
    items = items_text(text, limit=limit)
    if items:
        return items

    # No recognisable item table -- keep the old scan, minus the parts known to be
    # noise, so classification still has something to work with.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        folded = fold_tr(stripped)
        if any(skip in folded for skip in (
            "odenecek tutar", "toplam vergi", "hesaplanan kdv", "vergiler dahil",
            "mal hizmet toplam", "vergi kimlik", "vergi dairesi", "[gizlendi]",
        )):
            continue
        lines.append(stripped)
        if sum(len(item) for item in lines) > limit:
            break
    return "\n".join(lines)[:limit]


def classify(vendor: str | None, text: str, *, use_llm: bool = True,
             use_classifier: bool = True) -> tuple[str, str]:
    """Infer a spend category.

    Returns ``(category, source)`` where source is ``keyword``, ``classifier``,
    ``classifier_low`` (used, but flagged for review), ``llm`` or ``default``.

    Strategies run cheapest-and-surest first:

    1. **Keyword rules** -- exact, instant, and already correct for the recurring
       issuers that dominate a real corpus.
    2. **Trained classifier** (classifier.py) -- a logistic head over frozen
       embeddings. Measured on 6 held-out vendors: embeddings 3/6 zero-shot against
       qwen3-4b's 1/6 and qwen3-8b's 2/6, at ~0.2s versus ~30-60s.
    3. **The generative model**, only when asked for. Still the least reliable and by
       far the slowest.

    Below CONFIDENCE_MIN the classifier's guess is discarded rather than stored: an
    uncertain category is flagged, never committed, which is the same rule the rest of
    the pipeline follows.
    """
    haystack = fold_tr(f"{vendor or ''}\n{text}")

    hit = _match_keywords(haystack)
    if hit:
        return hit, "keyword"

    if use_classifier:
        # Same "<vendor>. <items>" shape the head was trained on -- training and
        # serving must build the string identically or the geometry shifts.
        items = excerpt_for_classification(text, limit=400)
        probe = f"{vendor}. {items}".strip().rstrip(".") if vendor else items
        prediction = classifier.classify(probe)
        if prediction and prediction[1] >= CONFIDENCE_MIN:
            label, confidence = prediction
            return label, ("classifier" if confidence >= CONFIDENCE_CLEAR
                           else "classifier_low")

    if not use_llm:
        return DEFAULT_CATEGORY, "default"

    prompt = _PROMPT.format(
        vendor=vendor or "bilinmiyor",
        excerpt=excerpt_for_classification(text),
        labels="\n".join(f"- {label}: {gloss}" for label, gloss in CATEGORY_GLOSS),
    )

    for _ in range(2):  # one retry; the model occasionally answers with a sentence
        try:
            # 24 tokens, not 16: at 16 the model truncated a correct "yeme-içme"
            # mid-word and the answer was discarded as unparseable.
            answer = foundry.chat(prompt, temperature=0.0, max_tokens=24)
        except Exception:  # noqa: BLE001 - classification must never break ingest
            return DEFAULT_CATEGORY, "default"
        category = _coerce(answer)
        if category:
            return category, "llm"

    return DEFAULT_CATEGORY, "default"
