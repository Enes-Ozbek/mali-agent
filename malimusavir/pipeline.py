"""PDF -> ExtractedInvoice, tying together text extraction, profiles and category."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from . import db
from .category import classify
from .extractors import extract_invoice
from .pdf_text import PdfDocument, load_pdf


def extract_from_text(
    text: str,
    *,
    source_path: str | None = None,
    content_hash: str | None = None,
    use_llm: bool = False,
):
    """Run the full extraction chain over already-loaded invoice text.

    ``use_llm`` defaults to False. Measured on this machine, qwen3-4b classified 1 of 6
    held-out vendors correctly (a hairdresser came back as "abonelik", a plant nursery
    as "enerji") across three prompt designs, at ~30s per call. The keyword table is
    both accurate and instant, so the model is opt-in and anything it produces is
    flagged for review rather than trusted into the category aggregates.
    """
    invoice = extract_invoice(text)
    invoice.source_path = source_path
    invoice.content_hash = content_hash
    invoice.raw_text = text

    category, source = classify(invoice.vendor, text, use_llm=use_llm)
    invoice.set("category", category, source)
    if source == "default":
        invoice.review_reasons.append("category:unresolved")
    elif source == "llm":
        invoice.review_reasons.append("category:llm_unverified")
    elif source == "classifier_low":
        # Confident enough to use, not confident enough to leave unchecked.
        invoice.review_reasons.append("category:low_confidence")
    return invoice


def extract_from_pdf(path: str | Path, *, use_llm: bool = False):
    """Load one PDF and extract a structured invoice from it."""
    doc: PdfDocument = load_pdf(path)

    if doc.is_scanned:
        # OCR is explicitly out of scope; surface the file rather than emit a blank row.
        invoice = extract_from_text(
            doc.text,
            source_path=str(doc.path),
            content_hash=doc.content_hash,
            use_llm=False,
        )
        invoice.review_reasons.append("scanned:no_extractable_text")
        return invoice

    return extract_from_text(
        doc.text,
        source_path=str(doc.path),
        content_hash=doc.content_hash,
        use_llm=use_llm,
    )


@dataclass
class IngestReport:
    """Outcome of ingesting a folder."""

    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.inserted + self.updated + self.skipped


def ingest_folder(
    conn,
    folder: str | Path,
    *,
    use_llm: bool = False,
    on_progress=None,
    client_name: str | None = None,
) -> IngestReport:
    """Extract every PDF under ``folder`` and store the results.

    Safe to re-run: unchanged invoices are skipped rather than duplicated.

    ``client_name`` files everything under one client. Without it the invoices are
    stored unassigned, which is how this worked before clients existed and remains
    valid -- a single-user archive has no clients to speak of.
    """
    from . import clients as clients_mod

    report = IngestReport()
    client = clients_mod.resolve(conn, client_name) if client_name else None

    for path in find_pdfs(folder):
        try:
            invoice = extract_from_pdf(path, use_llm=use_llm)
            if not invoice.invoice_no:
                report.failed.append((str(path), "no invoice number found"))
                continue
            if client is not None:
                invoice.client_id = client.id
                invoice.direction = clients_mod.direction_for(client, invoice.vendor_tax_id)
            result = db.upsert_invoice(conn, invoice)
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not abort the run
            report.failed.append((str(path), f"{type(exc).__name__}: {exc}"))
            continue

        setattr(report, result.value, getattr(report, result.value) + 1)
        if invoice.needs_review:
            report.flagged.append(f"{path.name}: {', '.join(invoice.review_reasons)}")
        if on_progress:
            on_progress(path, invoice, result)

    return report


@dataclass
class ArchiveReport:
    """Outcome of ingesting a whole client archive."""

    invoices: IngestReport = field(default_factory=IngestReport)
    declarations: int = 0
    documents: int = 0
    clients: list[str] = field(default_factory=list)
    problems: list[tuple[str, str]] = field(default_factory=list)
    misfiled: list[str] = field(default_factory=list)


def ingest_archive(
    conn,
    root: str | Path,
    *,
    only_client: str | None = None,
    use_llm: bool = False,
    on_progress=None,
) -> ArchiveReport:
    """Ingest `<root>/<client>/<year>/<type>/*.pdf`.

    Invoices go through the existing extraction pipeline. Declaration and other document
    folders are recorded so they can be listed and opened -- deliberately without being
    parsed, since guessing at unseen layouts is how confidently wrong data gets stored.
    """
    from . import archive, clients as clients_mod

    walked = archive.walk(root, only_client=only_client)
    report = ArchiveReport(
        clients=walked.clients,
        problems=[(p.path, p.reason) for p in walked.problems],
    )

    for item in walked.items:
        client = clients_mod.resolve_folder(conn, item.client)
        try:
            if item.kind == archive.Kind.INVOICE:
                _ingest_invoice(conn, item, client, report, use_llm=use_llm)
            elif item.kind == archive.Kind.DECLARATION:
                report.declarations += _record_declaration(conn, item, client)
            else:
                report.documents += _record_document(conn, item, client)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the archive
            report.problems.append((str(item.path), f"{type(exc).__name__}: {exc}"))
            continue
        if on_progress:
            on_progress(item, client)

    return report


def _ingest_invoice(conn, item, client, report: ArchiveReport, *, use_llm: bool) -> None:
    from . import clients as clients_mod

    invoice = extract_from_pdf(item.path, use_llm=use_llm)
    if not invoice.invoice_no:
        report.invoices.failed.append((str(item.path), "no invoice number found"))
        return

    invoice.client_id = client.id
    invoice.doc_year = item.year
    invoice.doc_month = item.month
    invoice.doc_type = item.doc_type
    # The folder wins when it states the direction. Whoever filed the document knew
    # which side of the ledger it belonged on; the tax-id comparison is a fallback that
    # silently calls everything a purchase whenever the client's own VKN is missing or
    # the seller's failed to extract.
    invoice.direction = item.direction or clients_mod.direction_for(
        client, invoice.vendor_tax_id)

    # The invoice's own date rules every calculation; the folder only says where it was
    # filed. A mismatch is a filing error worth surfacing, not something to silently
    # accept or to "correct" by overriding the document.
    if invoice.date and not invoice.date.startswith(str(item.year)):
        invoice.review_reasons.append(f"misfiled:{item.year}")
        report.misfiled.append(f"{item.path.name}: {invoice.date} -> {item.year}/")
    elif item.month and invoice.date and int(invoice.date[5:7]) != item.month:
        # Same rule one level down. Only checked when the year agrees, so a
        # wrongly-filed document is reported once with the most useful message.
        invoice.review_reasons.append(f"misfiled_month:{item.month}")
        report.misfiled.append(
            f"{item.path.name}: {invoice.date} -> {item.year}/{item.month_folder}/")

    result = db.upsert_invoice(conn, invoice)
    setattr(report.invoices, result.value, getattr(report.invoices, result.value) + 1)
    if invoice.needs_review:
        report.invoices.flagged.append(
            f"{item.path.name}: {', '.join(invoice.review_reasons)}")


def _record_declaration(conn, item, client) -> int:
    """Store a tahakkuk fişi, with its fields extracted.

    `payable` is money the client owes, so it is read label-anchored by tahakkuk.py and
    cross-checked against the receipt's own TOPLAM. A receipt whose rows do not sum to
    its stated total keeps needs_review set rather than presenting a figure that might
    be wrong.
    """
    from . import archive as archive_mod, tahakkuk as tahakkuk_mod

    doc = load_pdf(item.path)
    parsed = tahakkuk_mod.parse(doc.text)

    reasons = list(parsed.review_reasons)

    # 3_Beyannameler and 4_Tahakkuklar are two different documents of the same event.
    # Only the accrual receipt has an extractor; a beyanname is a GİB system output
    # with its own layout, and no real sample has been checked against yet. Reporting
    # it as a malformed receipt ("missing:total") would read as an extraction bug
    # rather than what it is -- a document type this tool does not parse. Stored and
    # listed so it can be opened, and flagged so nothing downstream treats it as data.
    if "tahakkuk" not in archive_mod.folder_key(item.doc_type) and parsed.needs_review:
        reasons = ["beyanname:not_parsed"]
    # The receipt names its taxpayer. If that VKN is on file for this client and does not
    # match, the document is filed under the wrong client -- worth surfacing loudly.
    if (parsed.taxpayer_tax_id and client.tax_id
            and parsed.taxpayer_tax_id != "".join(c for c in client.tax_id if c.isdigit())):
        reasons.append(f"wrong_client:{parsed.taxpayer_tax_id}")
    if parsed.period and not parsed.period.startswith(str(item.year)):
        reasons.append(f"misfiled:{item.year}")

    lines = json.dumps(
        [
            {"code": ln.code, "kind": ln.kind, "matrah": ln.matrah,
             "accrued": ln.accrued, "offset": ln.offset, "payable": ln.payable,
             "due_date": ln.due_date}
            for ln in parsed.lines
        ],
        ensure_ascii=False,
    )
    accrued = sum(ln.accrued for ln in parsed.lines if ln.accrued is not None) or None
    offset = sum(ln.offset for ln in parsed.lines if ln.offset is not None) or None

    return _insert_ignore(
        conn,
        "INSERT OR IGNORE INTO declarations "
        "(client_id, kind, period, accrued, offset_amount, payable, due_date, "
        " issue_date, receipt_no, taxpayer_tax_id, lines, doc_year, doc_month, doc_type, "
        " source_path, content_hash, raw_text, needs_review, review_reasons, ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (client.id, parsed.kind, parsed.period, accrued, offset, parsed.total_payable,
         parsed.due_date, parsed.issue_date, parsed.receipt_no, parsed.taxpayer_tax_id,
         lines, item.year, item.month, item.doc_type, str(item.path), doc.content_hash, doc.text,
         int(bool(reasons)), json.dumps(reasons, ensure_ascii=False), _now()),
    )


def _record_document(conn, item, client) -> int:
    """Store a document that is listed but never read.

    Bank statements arrive as .xlsx or .csv, so this cannot assume a PDF. Non-PDFs are
    hashed from their bytes and no text is extracted -- there is no parser for them and
    inventing one would be worse than leaving the file to be opened by hand.
    """
    if item.path.suffix.lower() == ".pdf":
        content_hash = load_pdf(item.path).content_hash
    else:
        content_hash = hashlib.sha256(item.path.read_bytes()).hexdigest()

    return _insert_ignore(
        conn,
        "INSERT OR IGNORE INTO documents "
        "(client_id, doc_type, doc_year, doc_month, filename, source_path, "
        " content_hash, ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (client.id, item.doc_type, item.year, item.month, item.path.name,
         str(item.path), content_hash, _now()),
    )


def _insert_ignore(conn, sql: str, params: tuple) -> int:
    cursor = conn.execute(sql, params)
    conn.commit()
    return cursor.rowcount or 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_pdfs(folder: str | Path) -> Iterator[Path]:
    """Yield every PDF under a folder, recursively, in a stable order."""
    root = Path(folder)
    if root.is_file():
        yield root
        return
    yield from sorted(
        (p for p in root.rglob("*") if p.suffix.lower() == ".pdf"),
        key=lambda p: str(p).lower(),
    )
