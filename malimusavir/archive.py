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

#: Folder-name prefixes that select a handler. Matched on fold_tr output, so "Faturalar",
#: "faturalar" and "FATURALAR" are the same thing.
INVOICE_FOLDERS = ("fatura",)
#: Tahakkuk fişi -- the tax office's accrual receipt, and the document that actually
#: states what is owed. "beyanname" is kept as an alias because practices file the two
#: together and the extractor recognises the receipt by its own heading either way.
DECLARATION_FOLDERS = ("tahakkuk", "beyanname", "beyan")

_YEAR = re.compile(r"^(19|20)\d{2}$")

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
class ArchiveItem:
    """One PDF, with everything its path says about it."""

    path: Path
    client: str
    year: int
    doc_type: str          #: the folder name, verbatim
    kind: str              #: Kind.*
    month: int | None = None       #: 1-12 from the folder, None when not filed by month
    month_folder: str | None = None  #: the folder name, verbatim


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
    folded = fold_tr(name)
    if any(folded.startswith(p) for p in INVOICE_FOLDERS):
        return Kind.INVOICE
    if any(folded.startswith(p) for p in DECLARATION_FOLDERS):
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


def _pdfs(folder: Path) -> Iterator[Path]:
    """PDFs under a document-type folder, at any depth below it."""
    yield from sorted(
        (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"),
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
    """Record every PDF under one document-type folder."""
    kind = classify_folder(type_dir.name)
    for pdf in _pdfs(type_dir):
        result.items.append(ArchiveItem(
            path=pdf, client=client, year=year, doc_type=type_dir.name, kind=kind,
            month=month, month_folder=month_folder,
        ))
