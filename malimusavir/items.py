"""Pull the goods/services description out of an invoice.

Retrieval lives or dies on this. An invoice's *header* -- company name, address,
"Merhaba" -- is near-identical across every document from the same issuer, so a summary
built from the top of the page cannot distinguish one invoice from another. What
actually distinguishes them is the line-item table, which sits in the middle.

The table is found structurally: e-Arsiv invoices print a recognisable column-header
row, and the totals block below it is an equally recognisable terminator.
"""

from __future__ import annotations

import re

from .normalize import fold_tr

#: Column-header rows that open a line-item table.
_TABLE_HEADERS = (
    r"mal\s*/?\s*hizmet\s*cinsi",
    r"malzeme\s*aciklama",
    r"hizmet\s*/?\s*urun\s*adi",
    r"hizmet\s*aciklamasi",
    r"aciklama.*tutar",
    r"fatura\s*ozeti",          # telecom bills call their summary block this
    r"urun\s*aciklamasi",
)

#: The totals block, which always follows the items.
_TABLE_ENDERS = (
    "mal hizmet toplam", "odenecek tutar", "toplam vergi", "genel toplam",
    "vergiler dahil", "yalniz", "toplam tutar", "net ara toplam", "ara toplam",
    "hesaplanan kdv", "toplam iskonto", "devlete iletilecek", "fatura toplami",
    "toplam iad", "iade bolumu", "banka bilgi", "mesajiniz var",
)

#: Noise inside the item region: codes, bare quantities, currency columns.
_NOISE = re.compile(
    r"\b\d+[.,]\d+\s*(?:TL|TRY)?\b"      # prices
    r"|\b%\s*\d+(?:[.,]\d+)?\b"          # VAT rates
    r"|\b\d+\s*(?:Adet|adet|ADET)\b"     # quantities
    r"|\b[A-Z]{2,}\d{6,}\b"              # stock codes
    r"|\b\d{6,}\b",                      # long reference numbers
)

_MAX_ITEM_LINES = 12

#: Column-header vocabulary. Multi-line headers leave fragments ("Tutar **Indirim
#: Toplam") inside the item region; a line made only of these words is not an item.
_HEADER_WORDS = frozenset(
    "tutar indirim toplam oran orani tutari vergiler vergi miktar birim fiyat adet "
    "aciklama cinsi no sira sno kdv isk diger b urun hizmet mal malzeme".split()
)


def line_items(text: str) -> list[str]:
    """Return the descriptive item lines of an invoice, in order."""
    lines = text.splitlines()
    folded = [fold_tr(line) for line in lines]

    start = None
    for index, line in enumerate(folded):
        if any(re.search(pattern, line) for pattern in _TABLE_HEADERS):
            start = index + 1
            break
    if start is None:
        return []

    items: list[str] = []
    for index in range(start, len(lines)):
        if any(token in folded[index] for token in _TABLE_ENDERS):
            break
        cleaned = _clean_item(lines[index])
        if cleaned:
            items.append(cleaned)
        if len(items) >= _MAX_ITEM_LINES:
            break
    return items


def _clean_item(line: str) -> str:
    """Strip prices, codes and quantities, keeping the words that describe the item."""
    stripped = _NOISE.sub(" ", line)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip(" .:-|")
    # Drop rows that were only numbers, and column-continuation fragments.
    if len(stripped) < 4 or not re.search(r"[^\W\d_]{3,}", stripped):
        return ""
    words = [w for w in re.findall(r"[^\W\d_]+", fold_tr(stripped)) if len(w) > 1]
    if words and all(word in _HEADER_WORDS for word in words):
        return ""
    return stripped


def items_text(text: str, *, limit: int = 400) -> str:
    """Line items as one string, for embedding or prompting."""
    joined = "; ".join(line_items(text))
    return joined[:limit]
