"""Walk a client-organised archive.

    <root>/<client>/<year>/<month>/<doc type>/*.pdf
    <root>/<client>/<year>/<doc type>/*.pdf          (month omitted)

Client, year, month and document type all come from the path, so nothing has to be
inferred from document contents and no buyer identity needs storing.

The month level is optional because both layouts exist in real practices, but it is
never *invented*: a document filed straight under the year keeps ``month=None`` and the
UI shows it as "ay belirtilmemiş". Deriving the month from the document's own date
instead would quietly disagree with the folder it actually lives in, which defeats the
point of mirroring the disk.

The layout rules are deliberately strict. A folder that does not fit is *reported*, never
guessed at: silently inventing a client called "2026" or filing a stray PDF under the
wrong taxpayer is far worse than telling the operator their tree is malformed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .normalize import fold_tr

#: Words that select a handler, matched anywhere in the folded folder name. Substring
#: rather than prefix, because the standard layout numbers its folders for ordering:
#: "1_Gelir_Faturalari" has to reach the invoice pipeline the same as "Faturalar".
INVOICE_FOLDERS = ("fatura",)
#: The tax office's accrual receipt (tahakkuk fişi) states what is owed and is parsed by
#: tahakkuk.py. "beyanname" is the filed declaration itself -- a different document with
#: its own layout -- but both live in the declarations table, and the extractor
#: recognises the receipt by its own heading, so a beyanname is stored and flagged
#: rather than silently read as a receipt.
DECLARATION_FOLDERS = ("tahakkuk", "beyanname", "beyan")
#: Bank statements: stored and listed, never parsed. They arrive as Excel as often as
#: PDF, which is why _files() below cannot be PDF-only for document folders.
BANK_FOLDERS = ("banka", "ekstre", "hesap hareket")

#: Folder words that state the direction of the invoices inside them. The standard
#: layout separates sales from purchases by folder, which is a far better signal than
#: comparing tax ids: it is stated by whoever filed the document, works before any
#: VKN is on file, and is right even when the seller's tax id fails to extract.
SALE_FOLDERS = ("gelir", "satis", "kesilen")
PURCHASE_FOLDERS = ("gider", "alis", "masraf", "gelen")

#: "1_", "01-", "3. " -- ordering prefixes that exist to sort folders in Explorer and
#: say nothing about what is inside them.
_ORDER_PREFIX = re.compile(r"^\d+\s*[._\-)]+\s*")

_YEAR = re.compile(r"^(19|20)\d{2}$")

#: "45678912345 - Canan Aydın E-Ticaret", optionally wrapped in brackets. The client
#: folder carries the taxpayer's own VKN/TCKN, which is what makes sales/purchase
#: direction and the "is this receipt really theirs" cross-check work without anyone
#: typing the number into the UI.
_CLIENT_FOLDER = re.compile(
    r"^[\[\(]?\s*(\d{10,11})\s*[-–—_]+\s*(.+?)\s*[\]\)]?$"
)

#: Documents that are stored but never parsed may be spreadsheets; anything that goes
#: through text extraction must be a PDF.
PARSEABLE_SUFFIXES = (".pdf",)
STORED_SUFFIXES = (".pdf", ".xlsx", ".xls", ".xlsm", ".csv")

#: Turkish month folder names, folded. Index + 1 is the month number.
MONTH_NAMES = ("ocak", "subat", "mart", "nisan", "mayis", "haziran",
               "temmuz", "agustos", "eylul", "ekim", "kasim", "aralik")

#: A leading number in a month folder: "01", "1", "01-Ocak", "03_Mart", "12. Aralik".
#: (?!\d) rather than \b: underscore is a word character, so \b never fires after the
#: digits in "03_Mart" and that folder read as "not a month".
_MONTH_NUMBER = re.compile(r"^(\d{1,2})(?!\d)")

#: Anything older or newer than this is far likelier to be a mis-named folder than a real
#: filing year, and treating it as a year would create a nonsense partition.
YEAR_MIN, YEAR_MAX = 1990, 2100


class Kind:
    INVOICE = "invoice"
    DECLARATION = "declaration"
    DOCUMENT = "document"


@dataclass(frozen=True)
class ClientFolder:
    """What a client folder's name says about the taxpayer."""

    name: str                    #: the folder name, verbatim -- still the identity
    tax_id: str | None = None    #: VKN (10) or TCKN (11) parsed out of the name
    title: str | None = None     #: the ünvan after the dash


def parse_client_folder(name: str) -> ClientFolder:
    """Split "45678912345 - Canan Aydın E-Ticaret" into its parts.

    A folder that does not follow the convention is not an error -- it simply yields no
    tax id, and the client keeps working exactly as before.
    """
    match = _CLIENT_FOLDER.match(name.strip())
    if not match:
        return ClientFolder(name=name)
    return ClientFolder(name=name, tax_id=match.group(1), title=match.group(2).strip())


def folder_key(name: str) -> str:
    """A folder name reduced to the words that classify it.

    Strips the ordering prefix so "1_Gelir_Faturalari" and "Gelir Faturaları" are the
    same thing, and normalises separators so substring matching works on either.
    """
    folded = fold_tr(name)
    stripped = _ORDER_PREFIX.sub("", folded)
    return stripped.replace("_", " ").replace("-", " ").strip()


def pretty_folder(name: str) -> str:
    """The folder name as a human should read it: "1_Gelir_Faturalari" -> "Gelir
    Faturalari". Display only -- the verbatim name stays the identity."""
    cleaned = _ORDER_PREFIX.sub("", name.strip())
    return cleaned.replace("_", " ").strip() or name


def direction_for_folder(name: str) -> str | None:
    """"satis" / "alis" when the folder says so, else None."""
    from .clients import PURCHASE, SALE

    key = folder_key(name)
    if any(word in key for word in SALE_FOLDERS):
        return SALE
    if any(word in key for word in PURCHASE_FOLDERS):
        return PURCHASE
    return None


@dataclass(frozen=True)
class ArchiveItem:
    """One PDF, with everything its path says about it."""

    path: Path
    client: str
    year: int
    doc_type: str          #: the folder name, verbatim
    kind: str              #: Kind.*
    month: int | None = None       #: 1-12 from the folder, None when not filed by month
    month_folder: str | None = None  #: the folder name, verbatim
    #: "satis"/"alis" when the document-type folder says so ("1_Gelir_Faturalari"),
    #: else None and the tax-id comparison decides.
    direction: str | None = None


@dataclass
class ArchiveProblem:
    """Something the walker refused to guess at."""

    path: str
    reason: str


@dataclass
class WalkResult:
    items: list[ArchiveItem] = field(default_factory=list)
    problems: list[ArchiveProblem] = field(default_factory=list)

    @property
    def clients(self) -> list[str]:
        return sorted({item.client for item in self.items})


def classify_folder(name: str) -> str:
    """Which handler a document-type folder maps to."""
    key = folder_key(name)
    # Bank statements first: "5_Banka_Ekstreleri" contains no invoice word today, but
    # checking it ahead of the others keeps a folder like "Banka Fatura Ödemeleri" out
    # of the invoice pipeline, where every row would fail extraction.
    if any(word in key for word in BANK_FOLDERS):
        return Kind.DOCUMENT
    if any(word in key for word in INVOICE_FOLDERS):
        return Kind.INVOICE
    if any(word in key for word in DECLARATION_FOLDERS):
        return Kind.DECLARATION
    return Kind.DOCUMENT


def parse_year(name: str) -> int | None:
    if not _YEAR.match(name.strip()):
        return None
    year = int(name.strip())
    return year if YEAR_MIN <= year <= YEAR_MAX else None


def parse_month(name: str) -> int | None:
    """A month folder's number, or None if the folder is not a month.

    Accepts the forms practices actually use: "Ocak", "01", "1", "01-Ocak", "03_Mart".
    Returning None is meaningful -- it is how walk() tells a month folder apart from a
    document-type folder sitting directly under the year.
    """
    cleaned = name.strip()
    if not cleaned:
        return None

    folded = fold_tr(cleaned)
    for index, month in enumerate(MONTH_NAMES, start=1):
        # startswith, not equality: "01-Ocak" folds to "01-ocak" and is handled by the
        # numeric branch, but "Ocak 2026" and "Ocak Ayı" should still read as January.
        if folded.startswith(month):
            return index

    match = _MONTH_NUMBER.match(folded)
    if match:
        number = int(match.group(1))
        if 1 <= number <= 12:
            return number
    return None


def _files(folder: Path, suffixes: tuple[str, ...]) -> Iterator[Path]:
    """Matching files under a document-type folder, at any depth below it."""
    yield from sorted(
        (p for p in folder.rglob("*")
         if p.is_file() and p.suffix.lower() in suffixes),
        key=lambda p: str(p).lower(),
    )


def walk(root: str | Path, *, only_client: str | None = None) -> WalkResult:
    """Enumerate an archive, reporting anything that does not fit the layout."""
    root = Path(root)
    result = WalkResult()

    if not root.is_dir():
        result.problems.append(ArchiveProblem(str(root), "arşiv klasörü bulunamadı"))
        return result

    wanted = fold_tr(only_client) if only_client else None

    for client_dir in sorted((p for p in root.iterdir() if p.is_dir()),
                             key=lambda p: p.name.lower()):
        if wanted and fold_tr(client_dir.name) != wanted:
            continue

        # A PDF sitting directly in the client folder has no year and no type. Report it
        # rather than inventing either.
        for stray in client_dir.glob("*.pdf"):
            result.problems.append(
                ArchiveProblem(str(stray), "yıl klasörü içinde değil — atlandı"))

        year_dirs = [p for p in client_dir.iterdir() if p.is_dir()]
        if not year_dirs:
            result.problems.append(
                ArchiveProblem(str(client_dir), "yıl klasörü yok — atlandı"))
            continue

        for year_dir in sorted(year_dirs, key=lambda p: p.name):
            year = parse_year(year_dir.name)
            if year is None:
                result.problems.append(ArchiveProblem(
                    str(year_dir), f"yıl olarak okunamadı ({year_dir.name!r}) — atlandı"))
                continue

            for stray in year_dir.glob("*.pdf"):
                result.problems.append(ArchiveProblem(
                    str(stray), "belge türü klasörü içinde değil — atlandı"))

            for sub_dir in sorted((p for p in year_dir.iterdir() if p.is_dir()),
                                  key=lambda p: p.name.lower()):
                month = parse_month(sub_dir.name)
                if month is None:
                    # Not a month, so this is a document-type folder filed straight
                    # under the year. Kept working rather than rejected: plenty of
                    # archives have no month level, and month stays None instead of
                    # being back-filled from the document's own date.
                    _collect(result, sub_dir, client_dir.name, year, None, None)
                    continue

                for stray in sub_dir.glob("*.pdf"):
                    result.problems.append(ArchiveProblem(
                        str(stray), "belge türü klasörü içinde değil — atlandı"))

                type_dirs = [p for p in sub_dir.iterdir() if p.is_dir()]
                if not type_dirs:
                    result.problems.append(ArchiveProblem(
                        str(sub_dir), "ay klasöründe belge türü klasörü yok — atlandı"))
                    continue

                for type_dir in sorted(type_dirs, key=lambda p: p.name.lower()):
                    _collect(result, type_dir, client_dir.name, year,
                             month, sub_dir.name)

    return result


def _collect(result: WalkResult, type_dir: Path, client: str, year: int,
             month: int | None, month_folder: str | None) -> None:
    """Record every usable file under one document-type folder."""
    kind = classify_folder(type_dir.name)
    direction = direction_for_folder(type_dir.name)

    # Invoices and declarations go through text extraction, so they have to be PDFs.
    # Everything else is stored and listed, never read, so a bank statement may just as
    # well be the .xlsx the bank actually exports.
    parseable = kind in (Kind.INVOICE, Kind.DECLARATION)
    wanted = PARSEABLE_SUFFIXES if parseable else STORED_SUFFIXES

    for path in _files(type_dir, wanted):
        result.items.append(ArchiveItem(
            path=path, client=client, year=year, doc_type=type_dir.name, kind=kind,
            month=month, month_folder=month_folder, direction=direction,
        ))

    if parseable:
        # A spreadsheet in an invoice folder cannot be extracted. Report it rather than
        # letting pdfplumber raise halfway through the run.
        for other in _files(type_dir, tuple(s for s in STORED_SUFFIXES
                                            if s not in PARSEABLE_SUFFIXES)):
            result.problems.append(ArchiveProblem(
                str(other), "PDF değil — bu klasördeki belgeler okunamaz, atlandı"))
