"""Turkish number, date and rate parsing.

Pure functions, no I/O and no LLM. This module is where extraction correctness is
actually pinned down -- if a total is wrong, it is almost always wrong here.

Turkish invoices use `.` for thousands and `,` for decimals, which is the exact
inverse of the English convention, so a naive float() silently produces numbers that
are wrong by a factor of 1000.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

# Whitespace variants that show up in PDF text extraction.
_SPACES = "       "

_CURRENCY_TOKENS = ("TRY", "TL", "₺", "TURK LIRASI")

# Turkish month names, ASCII-folded (see fold_tr) so both "Mayıs" and "MAYIS" hit.
_MONTHS = {
    "ocak": 1, "subat": 2, "mart": 3, "nisan": 4, "mayis": 5, "haziran": 6,
    "temmuz": 7, "agustos": 8, "eylul": 9, "ekim": 10, "kasim": 11, "aralik": 12,
}

_TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
})


def fold_tr(text: str) -> str:
    """Casefold Turkish text to ASCII for robust matching.

    Python's str.lower() mishandles Turkish: "MAYIS".lower() is "mayis" (dotted i)
    while the real word is "Mayıs" (dotless), so the two never compare equal. Folding
    every Turkish-specific letter to its ASCII base sidesteps the whole problem.
    """
    folded = text.translate(_TR_MAP).lower()
    # Strip any remaining combining marks (e.g. decomposed forms from some PDFs).
    return "".join(c for c in unicodedata.normalize("NFD", folded) if not unicodedata.combining(c))


def _strip_currency(text: str) -> str:
    cleaned = text
    for token in _CURRENCY_TOKENS:
        cleaned = re.sub(re.escape(token), " ", cleaned, flags=re.IGNORECASE)
    for space in _SPACES:
        cleaned = cleaned.replace(space, " ")
    return cleaned.strip()


def parse_tr_amount(text: str | float | int | None) -> float | None:
    """Parse a Turkish-formatted monetary amount into a float.

    >>> parse_tr_amount("1.748,80 TL")
    1748.8
    >>> parse_tr_amount("748,80TL")
    748.8
    >>> parse_tr_amount("1.234.567,89")
    1234567.89

    Returns None when the text holds no parseable number.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)

    cleaned = _strip_currency(str(text))
    match = re.search(r"-?\d[\d.,\s]*", cleaned)
    if not match:
        return None

    raw = re.sub(r"\s", "", match.group(0)).rstrip(".,")
    if not raw:
        return None

    negative = raw.startswith("-")
    raw = raw.lstrip("-")

    has_comma, has_dot = "," in raw, "." in raw

    if has_comma and has_dot:
        # Whichever separator appears last is the decimal point; the other groups.
        if raw.rindex(",") > raw.rindex("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif has_comma:
        # Multiple commas can only be English-style grouping ("1,234,567").
        if raw.count(",") > 1:
            raw = raw.replace(",", "")
        else:
            raw = raw.replace(",", ".")
    elif has_dot:
        tail = raw.rsplit(".", 1)[1]
        # A single dot followed by exactly 3 digits is Turkish grouping ("1.748"),
        # anything shorter is a decimal fraction ("748.80").
        if raw.count(".") > 1 or len(tail) == 3:
            raw = raw.replace(".", "")

    try:
        value = float(raw)
    except ValueError:
        return None
    return -value if negative else value


# A monetary amount, as opposed to any old number. Requires either two decimal digits
# or a trailing currency token, and refuses anything preceded by "%" so VAT rates
# ("%20,00") are never mistaken for tax amounts.
#
# This strictness is load-bearing: invoice PDFs are two-column, so the text to the
# right of "Odenecek Tutar" is frequently the recipient's address. Without it, the
# "No:119 D:3 Avcilar" in a delivery address parsed as a 119 TL total.
_MONEY_RE = re.compile(
    r"(?<![%\d.,])-?\d{1,3}(?:\.\d{3})+,\d{1,2}(?!\d)"
    r"|(?<![%\d.,])-?\d+,\d{1,2}(?!\d)"
    r"|(?<![%\d.,])-?\d{1,3}(?:\.\d{3})+(?=\s*(?:TL|TRY|₺))"
    r"|(?<![%\d.,])-?\d+(?=\s*(?:TL|TRY|₺))",
    re.IGNORECASE,
)


def parse_money(text: str | None, *, last: bool = False) -> float | None:
    """Parse a monetary amount, rejecting bare integers that are not money.

    >>> parse_money("Odenecek Tutar   1.536,00 TL")
    1536.0
    >>> parse_money("No:119 D:3 Avcilar Istanbul 34320") is None
    True
    >>> parse_money("KDV Orani %20,00") is None
    True

    Set ``last=True`` to take the rightmost amount on the line, which is what totals
    tables want when a row carries both a unit price and a line total.
    """
    if text is None:
        return None
    found = [m.group(0) for m in _MONEY_RE.finditer(str(text))]
    if not found:
        return None
    return parse_tr_amount(found[-1] if last else found[0])


def parse_tr_date(text: str | None) -> str | None:
    """Parse a Turkish invoice date into an ISO ``YYYY-MM-DD`` string.

    Accepts ``03/03/2026``, ``03 / 03 / 2026``, ``03.03.2026``, ``03-03-2026``,
    ``3 Mart 2026`` and dates already in ISO form. Returns None if unparseable.
    """
    if not text:
        return None
    raw = str(text)
    for space in _SPACES:
        raw = raw.replace(space, " ")
    raw = raw.strip()

    # Already ISO.
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", raw)
    if iso:
        return _safe_date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))

    # Numeric DD.MM.YYYY with optional spaces around the separators.
    numeric = re.search(r"\b(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{2,4})\b", raw)
    if numeric:
        day, month, year = (int(g) for g in numeric.groups())
        if year < 100:
            year += 2000
        return _safe_date(year, month, day)

    # "3 Mart 2026" / "03 MART 2026"
    named = re.search(r"\b(\d{1,2})\s+([^\W\d_]+)\s+(\d{4})\b", raw, flags=re.UNICODE)
    if named:
        month = _MONTHS.get(fold_tr(named.group(2)))
        if month:
            return _safe_date(int(named.group(3)), month, int(named.group(1)))

    return None


def _safe_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def parse_vat_rate(text: str | float | int | None) -> float | None:
    """Parse a KDV rate into a percentage float.

    >>> parse_vat_rate("%20")
    20.0
    >>> parse_vat_rate("20,00")
    20.0
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)

    cleaned = str(text).replace("%", " ")
    value = parse_tr_amount(cleaned)
    if value is None:
        return None
    # A rate expressed as a fraction (0.20) rather than a percentage.
    if 0 < value < 1:
        value *= 100
    return value if 0 <= value <= 100 else None


def parse_tax_id(text: str | None) -> str | None:
    """Extract a 10-digit VKN (or 11-digit TCKN used as a VKN) from a label value."""
    if not text:
        return None
    digits = re.sub(r"\D", "", str(text))
    if len(digits) in (10, 11):
        return digits
    match = re.search(r"\b(\d{10,11})\b", str(text))
    return match.group(1) if match else None


def format_tr_amount(value: float | None) -> str:
    """Format a number in Turkish convention: 1234.5 -> '1.234,50'."""
    if value is None:
        return "-"
    return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def clean_ws(text: str | None) -> str | None:
    """Collapse runs of whitespace; return None for anything empty."""
    if text is None:
        return None
    collapsed = re.sub(r"\s+", " ", str(text)).strip(" :–-")
    return collapsed or None
