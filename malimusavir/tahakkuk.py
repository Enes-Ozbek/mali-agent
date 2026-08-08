"""Extraction for tahakkuk fişi -- the tax office's accrual/assessment receipt.

A tahakkuk fiş says what a taxpayer was assessed for a period: which tax, the accrued
amount, anything offset against it (mahsup), what is actually payable, and the due date.
That makes it the document worth reading, because it is the figure the client actually
owes -- and it can be checked against the KDV the invoices imply.

Built label-anchored, for the same reason invoice extraction is: these are
government-issued forms with stable field names ("Vergilendirme Dönemi", "TAHAKKUK
EDEN", "TOPLAM"). A model asked to copy these digits would eventually copy them wrong,
and this is a number the client pays.

The line-item table is the load-bearing part:

    TÜRÜ   MATRAH   TAHAKKUK EDEN   MAHSUP EDİLEN   ÖDENECEK OLAN   VADESİ
    0015 KDV   0,00      0,00         11.861,90         0,00     28/06/2026
    1048 5035  0,00    791,00              0,00       791,00     28/06/2026
                                          TOPLAM       791,00
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .normalize import fold_tr, parse_money, parse_tr_date

#: Turkish tax codes seen on these receipts. The code is authoritative; the label beside
#: it varies in wording between offices.
TAX_CODES: dict[str, str] = {
    "0015": "kdv",            # Gerçek usulde katma değer vergisi
    "0003": "gelir_stopaj",   # Gelir vergisi stopajı
    "0001": "gelir",
    "0010": "kurumlar",
    "0032": "gecici",         # Geçici vergi
    "0040": "damga",          # Damga vergisi
    "1047": "damga",
    "1048": "damga",
    "9021": "mtv",
}

#: Words that identify the tax when no code maps cleanly.
_KIND_WORDS: tuple[tuple[str, str], ...] = (
    ("katma deger", "kdv"),
    ("muhtasar", "muhtasar"),
    ("gecici vergi", "gecici"),
    ("damga", "damga"),
    ("kurumlar", "kurumlar"),
    ("gelir vergisi", "gelir"),
)

#: One assessment line: a 4-digit tax code, four money columns, then a due date.
_ROW = re.compile(
    r"^\s*(\d{4})\s+(\S+)\s+"
    r"([\d.]+,\d{2})\s+([\d.]+,\d{2})\s+([\d.]+,\d{2})\s+([\d.]+,\d{2})\s+"
    r"(\d{2}/\d{2}/\d{4})",
    re.MULTILINE,
)

#: "05/2026-05/2026" -- a taxation period, given as a range even for a single month.
_PERIOD = re.compile(r"(\d{2})/(\d{4})\s*-\s*(\d{2})/(\d{4})")

#: The receipt's own serial, e.g. 2026062601L520000305.
_RECEIPT_NO = re.compile(r"\b(\d{10}[A-Z]\d{9,12})\b")

#: The taxpayer's VKN. Kept deliberately: it is a business identifier used to confirm the
#: receipt belongs to this client, and it is not a national ID -- those are redacted
#: upstream by pdf_text.redact before this module ever sees the text.
#:
#: The boundaries exclude letters, not just digits. The receipt serial
#: "2026062601L520000305" opens with ten digits followed by a letter, and a digits-only
#: lookahead happily read that prefix as the VKN.
_VKN = re.compile(r"(?<![\dA-Za-z])(\d{10})(?![\dA-Za-z])")


@dataclass
class AssessmentLine:
    """One row of the assessment table."""

    code: str
    kind: str | None
    matrah: float | None          # taxable base
    accrued: float | None         # tahakkuk eden
    offset: float | None          # mahsup edilen
    payable: float | None         # ödenecek olan
    due_date: str | None          # ISO


@dataclass
class Tahakkuk:
    """A parsed tahakkuk fişi."""

    receipt_no: str | None = None
    kind: str | None = None            # the primary tax, from Ana Vergi Kodu
    period: str | None = None          # "2026-05"
    period_end: str | None = None      # "2026-05" -- differs for quarterly taxes
    accepted_date: str | None = None   # Kabul Tarihi
    issue_date: str | None = None      # Düzenleme Tarihi
    due_date: str | None = None        # earliest Vadesi across the lines
    total_payable: float | None = None # TOPLAM
    taxpayer_tax_id: str | None = None
    lines: list[AssessmentLine] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)


def looks_like_tahakkuk(text: str) -> bool:
    return "tahakkuk" in fold_tr(text)


def _kind_from(code: str, label: str, text: str) -> str | None:
    if code in TAX_CODES:
        return TAX_CODES[code]
    folded = fold_tr(f"{label} {text}")
    for needle, kind in _KIND_WORDS:
        if needle in folded:
            return kind
    return None


def _iso_period(month: str, year: str) -> str:
    return f"{year}-{month}"


def parse(text: str) -> Tahakkuk:
    """Read a tahakkuk fişi. Never raises -- unreadable fields become review reasons."""
    out = Tahakkuk()

    if not looks_like_tahakkuk(text):
        out.review_reasons.append("tahakkuk:not_recognised")
        return out

    receipt = _RECEIPT_NO.search(text)
    if receipt:
        out.receipt_no = receipt.group(1)

    period = _PERIOD.search(text)
    if period:
        out.period = _iso_period(period.group(1), period.group(2))
        out.period_end = _iso_period(period.group(3), period.group(4))

    # "Ana Vergi Kodu 0015" names the tax the receipt is primarily for.
    main = re.search(r"ana\s*vergi\s*kodu\s*[:\-]?\s*(\d{4})", fold_tr(text))
    if main:
        out.kind = _kind_from(main.group(1), "", text)

    # Two dates sit on the same line as the period: Kabul Tarihi and Düzenleme Tarihi.
    for line in text.splitlines():
        if out.period and _PERIOD.search(line):
            dates = re.findall(r"\d{2}/\d{2}/\d{4}", line)
            if dates:
                out.accepted_date = parse_tr_date(dates[0])
            if len(dates) > 1:
                out.issue_date = parse_tr_date(dates[-1])
            break

    for match in _ROW.finditer(text):
        code, label, matrah, accrued, offset, payable, due = match.groups()
        out.lines.append(AssessmentLine(
            code=code,
            kind=_kind_from(code, label, ""),
            matrah=parse_money(matrah),
            accrued=parse_money(accrued),
            offset=parse_money(offset),
            payable=parse_money(payable),
            due_date=parse_tr_date(due),
        ))

    due_dates = sorted(d for d in (line.due_date for line in out.lines) if d)
    if due_dates:
        out.due_date = due_dates[0]

    # TOPLAM is the authoritative payable figure; the per-line sum is a cross-check.
    total = re.search(r"toplam\s+([\d.]+,\d{2})", fold_tr(text))
    if total:
        out.total_payable = parse_money(total.group(1))

    vkn = _VKN.search(text)
    if vkn:
        out.taxpayer_tax_id = vkn.group(1)

    _validate(out)
    return out


def _validate(out: Tahakkuk) -> None:
    """Flag anything a human should look at before the figure is trusted."""
    if not out.period:
        out.review_reasons.append("missing:period")
    if not out.lines:
        out.review_reasons.append("missing:assessment_lines")
    if out.total_payable is None:
        out.review_reasons.append("missing:total")

    # The stated total must equal the sum of the payable column. A mismatch means a row
    # was misread -- exactly the failure mode that would otherwise put a wrong amount
    # owed in front of the user.
    payables = [line.payable for line in out.lines if line.payable is not None]
    if payables and out.total_payable is not None:
        summed = round(sum(payables), 2)
        if abs(summed - out.total_payable) > 0.02:
            out.review_reasons.append(
                f"reconcile:satirlar {summed} != toplam {out.total_payable}")

    if out.kind is None and out.lines:
        out.kind = next((line.kind for line in out.lines if line.kind), None)
    if out.kind is None:
        out.review_reasons.append("missing:kind")
