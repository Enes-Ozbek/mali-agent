"""Profile framework for label-anchored invoice extraction.

Every field carries the profile that produced it, so `--review` can show exactly where
each number came from instead of asking the user to trust an opaque blob of JSON.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple

from ..normalize import (
    clean_ws,
    fold_tr,
    parse_money,
    parse_tax_id,
    parse_tr_date,
    parse_vat_rate,
)

# Amounts and totals must reconcile to within one kurus of rounding noise.
RECONCILE_TOLERANCE = 0.02

FIELDS = (
    "invoice_no",
    "date",
    "vendor",
    "vendor_tax_id",
    "total_amount",
    "tax_amount",
    "net_amount",
    "vat_rate",
    "currency",
    "payment_method",
    "category",
)

REQUIRED_FIELDS = ("invoice_no", "date", "total_amount")


@dataclass
class ExtractedInvoice:
    """One invoice's structured data plus the provenance of every field."""

    invoice_no: str | None = None
    date: str | None = None
    vendor: str | None = None
    vendor_tax_id: str | None = None
    total_amount: float | None = None
    tax_amount: float | None = None
    net_amount: float | None = None
    vat_rate: float | None = None
    currency: str | None = "TL"
    payment_method: str | None = None
    category: str | None = None

    profile: str = "unknown"
    source_path: str | None = None
    content_hash: str | None = None
    #: The redacted source text. Carried through so storage and embedding work from
    #: the same scrubbed copy the extractor saw, never from the original PDF.
    raw_text: str | None = None

    #: Which client's archive this came from. None for the pre-multi-client rows and for
    #: a plain --ingest with no --client, both of which stay valid.
    client_id: int | None = None
    #: The archive year folder this was filed under. The invoice's own `date` remains
    #: authoritative for every calculation; this only records where it was filed, so a
    #: mismatch can be flagged as a filing error.
    doc_year: int | None = None
    #: "alis" (purchase) or "satis" (sale), decided by whether the seller's tax id is the
    #: client's own. Drives output-vs-input VAT once a client's sales invoices are loaded.
    direction: str | None = None
    field_sources: dict[str, str] = field(default_factory=dict)
    review_reasons: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)

    def set(self, name: str, value: Any, source: str) -> None:
        """Record a field value together with where it came from."""
        setattr(self, name, value)
        self.field_sources[name] = source if value is not None else "missing"

    def validate(self) -> None:
        """Populate review_reasons with anything a human should look at."""
        for name in REQUIRED_FIELDS:
            if getattr(self, name) is None:
                self.review_reasons.append(f"missing:{name}")

        # Cross-check the arithmetic. This catches the failure mode that matters most:
        # a correctly-shaped number lifted from the wrong row of the totals table.
        if None not in (self.net_amount, self.tax_amount, self.total_amount):
            expected = round(self.net_amount + self.tax_amount, 2)
            if abs(expected - self.total_amount) > RECONCILE_TOLERANCE:
                self.review_reasons.append(
                    f"reconcile:{self.net_amount}+{self.tax_amount}!={self.total_amount}"
                )

        if self.total_amount is not None and self.total_amount <= 0:
            self.review_reasons.append(f"nonpositive_total:{self.total_amount}")

    def to_row(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in FIELDS}


# A tail separated from its label by more whitespace than this is the *next column*,
# not this label's value. e-Arsiv PDFs print the seller block and the buyer block side
# by side, so "Vergi No" in the left column is routinely followed, on the same text
# line, by the buyer's tax number from the right column.
COLUMN_GAP = 12

#: How many non-empty following lines to search when the label line holds no value.
LOOKAHEAD = 3


class LabelHit(NamedTuple):
    index: int          #: line number of the label
    column: int         #: column the label starts at, used to tell columns apart
    tail: str           #: same-line text after the label, leading spaces intact
    lines: list[str]    #: all document lines, for lookahead

    @property
    def own_column_tail(self) -> str:
        """The tail, but only if it plausibly belongs to this label's column."""
        gap = len(self.tail) - len(self.tail.lstrip())
        return "" if gap > COLUMN_GAP else self.tail

    def following(self) -> Iterator[str]:
        """Non-empty lines after the label, capped at LOOKAHEAD."""
        seen = 0
        for offset in range(1, LOOKAHEAD * 4):
            index = self.index + offset
            if index >= len(self.lines) or seen >= LOOKAHEAD:
                return
            candidate = self.lines[index]
            if candidate.strip():
                seen += 1
                yield candidate


def iter_label_hits(text: str, label: str, *, by_column: bool = False) -> list[LabelHit]:
    """Find every line matching ``label``, in document order.

    ``label`` is a regex fragment matched case-insensitively against ASCII-folded text,
    so "Ödenecek Tutar" is found by the pattern "odenecek\\s*tutar".

    ``by_column`` sorts leftmost-column-first instead. That is only right for fields
    that appear in both the seller and buyer blocks (the seller's is the left-hand
    one). For everything else document order wins -- sorting dates by column let a
    figure from page 2 outrank the issue date on page 1.
    """
    lines = text.splitlines()
    folded_lines = fold_tr(text).splitlines()
    pattern = re.compile(label + r"\s*[:\-]?\s*(.*)$", re.IGNORECASE)

    hits: list[LabelHit] = []
    for index, folded_line in enumerate(folded_lines):
        match = pattern.search(folded_line)
        if not match:
            continue
        # Re-slice from the original line so the value keeps its real characters.
        # fold_tr strips combining marks, which is not length-preserving, so fall back
        # to the folded capture whenever the offsets cannot be trusted.
        line = lines[index] if index < len(lines) else ""
        aligned = len(line) == len(folded_line)
        tail = line[match.start(1):] if aligned else match.group(1)
        hits.append(LabelHit(index, match.start() if aligned else 0, tail, lines))

    if by_column:
        hits.sort(key=lambda hit: (hit.column, hit.index))
    return hits


def _search(text: str, label: str, parse) -> object | None:
    """Shared lookup: try this label's own column, then the lines below it."""
    for hit in iter_label_hits(text, label):
        value = parse(hit.own_column_tail)
        if value is not None and value != "":
            return value
        for candidate in hit.following():
            value = parse(candidate)
            if value is not None and value != "":
                return value
    return None


def search_label(text: str, label: str) -> str | None:
    """Return the text following a label, on the same line or the next non-empty one."""
    return _search(text, label, clean_ws)  # type: ignore[return-value]


def search_label_money(text: str, label: str) -> float | None:
    """Find the monetary value belonging to a label."""
    return _search(text, label, parse_money)  # type: ignore[return-value]


def search_label_date(text: str, label: str) -> str | None:
    """Find the date belonging to a label.

    Some issuers stack two column headers ("Fatura Tarihi   Ödeme Tarihi") on one line
    and both values on the next, so the label line itself holds no date at all.
    """
    return _search(text, label, parse_tr_date)  # type: ignore[return-value]


class InvoiceProfile:
    """Base class for vendor-specific extraction profiles."""

    name: ClassVar[str] = "base"
    #: ASCII-folded substrings that identify this issuer in the document text.
    markers: ClassVar[tuple[str, ...]] = ()
    #: Fixed vendor display name, when the profile knows the issuer.
    vendor_name: ClassVar[str | None] = None

    def matches(self, doc_text: str) -> bool:
        folded = fold_tr(doc_text)
        return any(marker in folded for marker in self.markers)

    def extract(self, doc_text: str) -> ExtractedInvoice:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- helpers shared by every profile -------------------------------------

    def _text(self, text: str, *labels: str) -> str | None:
        for label in labels:
            value = search_label(text, label)
            if value:
                return value
        return None

    def _amount(self, text: str, *labels: str) -> float | None:
        for label in labels:
            value = search_label_money(text, label)
            if value is not None:
                return value
        return None

    def _date(self, text: str, *labels: str) -> str | None:
        for label in labels:
            value = search_label_date(text, label)
            if value:
                return value
        return None

    def _rate(self, text: str, *labels: str) -> float | None:
        for label in labels:
            value = parse_vat_rate(search_label(text, label))
            if value is not None:
                return value
        return None

    def _tax_id(self, text: str, *labels: str) -> str | None:
        for label in labels:
            value = parse_tax_id(search_label(text, label))
            if value:
                return value
        return None
