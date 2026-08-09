"""PDF text extraction with redaction applied at the parser boundary.

Redaction happens *here*, before text reaches any model, the database, or the
embedding store. Asking an LLM to "please ignore the national ID" is a request, not a
guarantee; removing the bytes before they are ever passed on is a guarantee.

Removed: buyer TC Kimlik No, delivery/invoice addresses, IBANs, card numbers, emails
and phone numbers. Kept: the seller's Vergi Kimlik No, which is needed to group spend
by legal entity.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from . import paths
from .normalize import fold_tr

REDACTED = "[GIZLENDI]"

# Labels whose value is a national ID we never want to retain.
_TCKN_LABEL = re.compile(
    r"(T\.?\s?C\.?\s*Kimlik\s*(?:No|Numaras[ıi])?|TCKN)\s*[:\-]?\s*(\d[\d\s]{9,15})",
    re.IGNORECASE,
)

# Address blocks: everything from the label up to a blank line or the next label.
_ADDRESS_LABEL = re.compile(
    r"^(.*?(?:Teslimat\s*Adresi|Fatura\s*Adresi|Adres(?:i)?|M[üu]şteri\s*Adresi)\s*[:\-]?)(.*)$",
    re.IGNORECASE | re.MULTILINE,
)

_IBAN = re.compile(r"\bTR\s?\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

# Turkish mobile numbers, with or without a 0/+90/90 prefix. Real invoices print all
# three forms ("0532 123 45 67", "5321234567", "905321234567").
_PHONE = re.compile(
    r"(?<!\d)(?:\+?90[\s.-]?|0)?\(?5\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)"
)

# Four groups of four; see _mask_card for why matching alone is not enough to redact.
_MASKED_CARD = re.compile(
    r"\b(?:\d{4}|[*x]{4})(?:[\s.-]*(?:\d{4}|[*x]{4})){3}\b", re.IGNORECASE
)
_MASK_GROUP = re.compile(r"[*x]{4}", re.IGNORECASE)
_LONG_DIGITS = re.compile(r"(?<!\d)\d{11}(?!\d)")

# Buyer identity: the recipient block is introduced by one of these and is never
# labelled "Adres", so the address regex above cannot see it.
_RECIPIENT_NAME = re.compile(
    r"^(\s*)(?:Sayın|SAYIN|Sn\.?|Alıcı)\s*[:.]?\s*(.+)$", re.IGNORECASE | re.MULTILINE
)

# Street-type words are strong enough on their own to mark a line as an address.
_ADDRESS_STRONG = re.compile(
    r"\b(?:Mah\.|Mah\b|Mh\.|Mahallesi|Sok\.|Sok\b|Sk\.|Sokak|Cad\.|Cd\.|Caddesi|"
    r"Bulvar\w*|Blok|Apt\.|Sitesi)",
    re.IGNORECASE,
)

# "No:" on its own is NOT an address marker -- it appears in "Vergi Kimlik No : ...",
# "Fatura No:", "Siparis No:". Only a building/flat *pair* ("No:119 D:3") is decisive.
_ADDRESS_UNIT = re.compile(r"\bNo\s*:\s*\d", re.IGNORECASE)
_ADDRESS_FLAT = re.compile(r"\b(?:D\s*:\s*\d|Daire\s*:?\s*\d|Kat\s*:\s*\d)", re.IGNORECASE)


def _address_cut(line: str) -> int | None:
    """Where the address portion of a line begins, or None if there is no address."""
    starts = [m.start() for m in (_ADDRESS_STRONG.search(line),) if m]
    unit, flat = _ADDRESS_UNIT.search(line), _ADDRESS_FLAT.search(line)
    if unit and flat:
        starts.append(min(unit.start(), flat.start()))
    return min(starts) if starts else None

# Tax ids explicitly attributed to the buyer.
_BUYER_TAX_ID = re.compile(
    r"(M[üu]şteri[^\n:]*(?:VKN|TCKN|V\.?\s?D\.?)[^\n:]*:)([^\n]*)", re.IGNORECASE
)

# Start of the recipient block. Any tax id printed within the next few lines is the
# buyer's: on these invoices the seller's number is always in the header above.
# Unanchored: the recipient block often sits in the right-hand column, so "Sayın"
# appears mid-line with the seller's address to its left.
_BUYER_BLOCK_START = re.compile(r"\b(?:Sayın|SAYIN|Sn\.?|Alıcı|ALICI)\b", re.IGNORECASE)
_BUYER_BLOCK_SPAN = 8
_TAX_ID_LINE = re.compile(r"((?:VKN|TCKN|Vergi\s*(?:Kimlik\s*)?No)\s*[:\-]?\s*)([\d\s]{10,20})",
                          re.IGNORECASE)

#: Official forms label the taxpayer's name outright. A tahakkuk fişi carries
#: "SOYADI (UNVANI)     YILMAZ"; the recipient-block heuristics never see it because the
#: line has a label and no address beneath it.
_NAMED_FIELD = re.compile(
    r"^(\s*(?:SOYADI|ADI SOYADI|ADI|UNVANI|AD[İI] VE SOYADI)\s*(?:\([^)]*\))?\s*[:\-]?\s*)(\S.*)$",
    re.IGNORECASE | re.MULTILINE,
)

#: A line holding just a personal name: a few words, letters only, no field labels.
_NAME_LINE = re.compile(r"^\s*[^\W\d_][^\d:]{2,45}\s*\.?\s*$", re.UNICODE)
_NAME_STOPWORDS = (
    "fatura", "vergi", "adres", "tel", "posta", "sayin", "musteri", "toplam", "tutar",
    "merhaba", "sirketi", "ltd", "a.s", "sti", "banka", "sube", "hizmet", "odeme",
    "turkcell", "superonline", "mesaj", "sayfa", "belge", "kodu", "no",
)

# A VKN label immediately before a number means "keep this one".
_VKN_CONTEXT = re.compile(r"(Vergi\s*(?:Kimlik|Dairesi)?\s*(?:No|Numaras[ıi])?|VKN)\s*[:\-]?\s*$",
                          re.IGNORECASE)


def is_valid_tckn(digits: str) -> bool:
    """Validate a Turkish national ID against its checksum.

    Used so redaction targets real TCKNs rather than every 11-digit run, which would
    otherwise eat invoice numbers and order references.
    """
    if len(digits) != 11 or not digits.isdigit() or digits[0] == "0":
        return False
    d = [int(c) for c in digits]
    odd, even = sum(d[0:9:2]), sum(d[1:8:2])
    if (odd * 7 - even) % 10 != d[9]:
        return False
    return sum(d[:10]) % 10 == d[10]


def _is_luhn_valid(digits: str) -> bool:
    """Luhn checksum, used to tell a real card number from a long reference number."""
    if not digits.isdigit():
        return False
    total, parity = 0, len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _mask_card(match: re.Match[str]) -> str:
    """Redact card-shaped runs, but only when they really are cards.

    A 16-digit Turkcell *invoice number* matches the same shape as an unseparated PAN,
    and an earlier version redacted it -- destroying the document's primary key. So a
    match is only treated as a card if it is visibly masked, or passes Luhn.
    """
    token = match.group(0)
    if _MASK_GROUP.search(token):
        return REDACTED
    digits = re.sub(r"\D", "", token)
    return REDACTED if len(digits) == 16 and _is_luhn_valid(digits) else token


def redact(text: str, *, extra_terms: Sequence[str] | None = None) -> str:
    """Strip personal data from extracted invoice text.

    ``extra_terms`` are literal strings the caller knows to be personal -- typically
    the account holder's own name and tax number, from configuration.
    """

    def _mask_labelled_tckn(match: re.Match[str]) -> str:
        return f"{match.group(1)}: {REDACTED}"

    text = _TCKN_LABEL.sub(_mask_labelled_tckn, text)

    def _mask_bare_tckn(match: re.Match[str]) -> str:
        """A checksum-valid TCKN is redacted whatever label precedes it.

        There used to be an exemption for numbers labelled "Vergi Kimlik No", on the
        theory that such a number is a company tax id rather than a national one. A real
        tahakkuk fişi disproved it: sole traders file under their TCKN, so the document
        reads "VERGİ KİMLİK NUMARASI <11 hane> ( T.C. Kimlik No )" and the exemption
        preserved a national ID verbatim. Since _LONG_DIGITS only matches 11-digit runs
        and company VKNs are 10 digits, that exemption could never protect a real VKN --
        it only ever shielded TCKNs. Client identity comes from the archive folder, so
        nothing here needs the number.
        """
        return REDACTED if is_valid_tckn(match.group(0)) else match.group(0)

    text = _LONG_DIGITS.sub(_mask_bare_tckn, text)
    text = _IBAN.sub(REDACTED, text)
    text = _MASKED_CARD.sub(_mask_card, text)
    text = _EMAIL.sub(REDACTED, text)
    text = _PHONE.sub(REDACTED, text)
    # The buyer's tax number is personal data and is never needed: vendor_tax_id is
    # the seller's. Redacting it here means a profile bug cannot leak it into the DB.
    text = _BUYER_TAX_ID.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
    text = _ADDRESS_LABEL.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
    learned = _learn_recipient_names(text)
    for match in _NAMED_FIELD.finditer(text):
        candidate = match.group(2).strip(" .:")
        if _looks_like_name(candidate):
            learned.add(candidate)
    text = _NAMED_FIELD.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    text = _redact_buyer_block_tax_ids(text)

    raw_lines = text.split("\n")
    # Record which lines held addresses before blanking them -- the name pass needs to
    # know where the addresses were, and by then the evidence is gone.
    address_rows = {i for i, line in enumerate(raw_lines) if _address_cut(line) is not None}
    lines = [_redact_address_line(line) for line in raw_lines]
    lines, above_address = _redact_names_above_addresses(lines, address_rows)
    learned |= above_address
    text = "\n".join(lines)

    # A name identified anywhere in the document is that person's name everywhere in
    # it. Consumer bills greet the recipient by name at the top ("Merhaba, AYSE
    # YILMAZ") with no label at all, so redacting only the addressee block leaves the
    # greeting behind.
    for name in learned:
        text = re.sub(re.escape(name), REDACTED, text, flags=re.IGNORECASE)
        # PDF extraction sometimes pads names with extra spaces ("AYSE   YILMAZ").
        spaced = r"\s+".join(re.escape(part) for part in name.split())
        text = re.sub(spaced, REDACTED, text, flags=re.IGNORECASE)

    # Anything the caller knows is personal (the account holder's own name and tax
    # number). Detection heuristics cannot recognise an arbitrary personal name, and
    # on these consumer bills the recipient's name is printed with no label at all.
    for term in extra_terms or ():
        if term and term.strip():
            text = re.sub(re.escape(term.strip()), REDACTED, text, flags=re.IGNORECASE)
    return text


def _column_segment(line: str, start: int) -> str:
    """The text of one layout column: from ``start`` to the next run of 2+ spaces.

    Necessary because a single text line spans both columns -- "AYSE YILMAZ" in the
    recipient column and "Senaryo EARSIVFATURA" in the next are the same line.
    """
    body = line[start:].lstrip()
    return re.split(r"\s{2,}", body, maxsplit=1)[0].strip() if body else ""


def _learn_recipient_names(text: str) -> set[str]:
    """Collect names introduced by "Sayın"/"Sn."/"Alıcı", wherever they sit.

    The marker is frequently mid-line (the recipient block is the right-hand column)
    and the name itself is usually on the line *below* it, in the same column -- so
    neither a line-anchored regex nor a same-line capture finds it.
    """
    lines = text.split("\n")
    found: set[str] = set()

    for index, line in enumerate(lines):
        for match in _BUYER_BLOCK_START.finditer(line):
            column = match.start()
            candidate = _column_segment(line, match.end()).lstrip(":. ")
            if _looks_like_name(candidate):
                found.add(candidate.strip(" .:"))
                continue
            seen = 0
            for offset in range(1, 5):
                if index + offset >= len(lines) or seen >= 2:
                    break
                below = lines[index + offset]
                if not below.strip():
                    continue
                seen += 1
                candidate = _column_segment(below, column)
                if _looks_like_name(candidate):
                    found.add(candidate.strip(" .:"))
                    break
    return found


def _redact_buyer_block_tax_ids(text: str) -> str:
    """Redact tax ids printed inside the recipient block.

    The seller's VKN is always in the header, above the recipient block, so a tax id
    appearing just after "SAYIN"/"Alıcı" belongs to the buyer. Without this, invoices
    that label the buyer's number plainly as "VKN:" leaked it.
    """
    lines = text.split("\n")
    countdown = 0
    for index, line in enumerate(lines):
        if _BUYER_BLOCK_START.search(line):
            countdown = _BUYER_BLOCK_SPAN
            continue
        if countdown > 0:
            countdown -= 1
            lines[index] = _TAX_ID_LINE.sub(lambda m: f"{m.group(1)}{REDACTED}", line)
    return "\n".join(lines)


def _looks_like_name(line: str) -> bool:
    stripped = line.strip()
    if not stripped or not _NAME_LINE.match(line) or len(stripped.split()) > 4:
        return False
    folded = fold_tr(stripped)
    return not any(word in folded for word in _NAME_STOPWORDS)


def _redact_names_above_addresses(
    lines: list[str], address_rows: set[int]
) -> tuple[list[str], set[str]]:
    """Redact the recipient's name where it sits directly above their address.

    Consumer telecom bills print the addressee block as a bare name followed by the
    street address, with no "Sayın" label to anchor on. The address is detectable, so
    the unlabelled line immediately above it is the name.

    Returns the updated lines and the names found, so callers can strip other
    occurrences of the same name elsewhere in the document.
    """
    out = list(lines)
    found: set[str] = set()
    for index in sorted(address_rows):
        for offset in (1, 2):
            above = index - offset
            if above < 0:
                break
            if not out[above].strip():
                continue
            if _looks_like_name(out[above]):
                found.add(out[above].strip(" .:"))
                indent = out[above][: len(out[above]) - len(out[above].lstrip())]
                out[above] = f"{indent}{REDACTED}"
            break
    return out, found


def _redact_address_line(line: str) -> str:
    """Redact the address portion of a line, keeping the rest of the line intact.

    These PDFs are two-column, so an unlabelled address in the right-hand column
    routinely shares a text line with a real field label on the left -- the recipient's
    street address sits directly beside "Odenecek Tutar". Blanking the whole line
    therefore destroys the label and the total becomes unreadable.

    So: cut from the column boundary preceding the address token to end of line, and
    keep everything to its left. Indentation is preserved because iter_label_hits uses
    column positions to tell the seller block from the buyer block.
    """
    start = _address_cut(line)
    if start is None:
        return line
    boundary = line.rfind("  ", 0, start)
    head = line[: boundary + 2] if boundary != -1 else line[: len(line) - len(line.lstrip())]
    return f"{head}{REDACTED}"


@dataclass(frozen=True)
class PdfDocument:
    """Redacted text of one invoice PDF."""

    path: Path
    text: str
    page_count: int
    content_hash: str
    is_scanned: bool

    @property
    def folded(self) -> str:
        """ASCII-folded lowercase text, for case-insensitive vendor matching."""
        return fold_tr(self.text)


def redact_terms() -> tuple[str, ...]:
    """Literal personal strings to strip, from MALIMUSAVIR_REDACT or redact.txt.

    The account holder's own name and tax number appear on every invoice they receive,
    usually with no label to anchor a pattern on. Naming them once is more reliable
    than any heuristic, so this is the supported way to guarantee they never persist.
    """
    terms: list[str] = []
    raw = os.environ.get("MALIMUSAVIR_REDACT", "")
    terms.extend(part.strip() for part in raw.split(",") if part.strip())

    config = paths.user_data("redact.txt")
    if config.exists():
        for line in config.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
    return tuple(terms)


def load_pdf(path: str | Path, *, extra_terms: Sequence[str] | None = None) -> PdfDocument:
    """Extract redacted, layout-preserved text from a digital PDF.

    ``layout=True`` keeps column alignment, which matters because invoice amounts sit
    in right-aligned table columns; without it labels and values interleave and the
    label-anchored regexes in extractors/ cannot tell which number belongs to which row.
    """
    path = Path(path)
    pages: list[str] = []

    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            try:
                content = page.extract_text(layout=True) or ""
            except Exception:  # noqa: BLE001 - a bad page should not kill the document
                content = page.extract_text() or ""
            pages.append(content)

    raw = "\n".join(pages)
    # No extractable text means a scan or photo; OCR is explicitly out of scope, so
    # this is surfaced as a flag rather than silently producing an empty invoice.
    is_scanned = len(raw.strip()) < 50

    text = redact(raw, extra_terms=extra_terms if extra_terms is not None else redact_terms())
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return PdfDocument(
        path=path,
        text=text,
        page_count=page_count,
        content_hash=content_hash,
        is_scanned=is_scanned,
    )
