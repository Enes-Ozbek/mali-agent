"""SQLite storage: schema management and idempotent invoice insertion."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .extractors.base import FIELDS, ExtractedInvoice

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "faturalar.db"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"


class IngestResult(str, Enum):
    """What happened to one invoice on ingest."""

    INSERTED = "inserted"
    UPDATED = "updated"      # same invoice, changed content -- the PDF was re-issued
    SKIPPED = "skipped"      # already stored, byte-identical


def connect(path: str | Path | None = None, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open the database, creating the schema if needed.

    ``check_same_thread=False`` is for the web API only (see malimusavir.api.get_conn):
    FastAPI resolves a sync-generator dependency and then runs the route body as two
    separate calls into anyio's worker thread pool, which are not guaranteed to land
    on the same OS thread -- sqlite3 raises "objects created in a thread can only be
    used in that same thread" the moment they don't. Safe to disable here because each
    request gets its own connection, used strictly sequentially within that request
    and never shared across requests. The CLI never needs this; its default (True)
    keeps sqlite3's normal thread-safety check.
    """
    target = Path(path) if path else DEFAULT_DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    _migrate(conn)
    return conn


#: Columns added to `invoices` after the single-user era. schema.sql cannot express these
#: with CREATE TABLE IF NOT EXISTS -- that is a no-op on an existing table -- so they are
#: applied here instead, letting a database written by an earlier version upgrade in place
#: rather than needing a re-ingest.
_INVOICE_COLUMNS = (
    ("client_id", "INTEGER REFERENCES clients(id)"),
    ("doc_year", "INTEGER"),
    ("direction", "TEXT"),
    # The archive gained an optional <month>/ level between year and document type.
    ("doc_month", "INTEGER"),
    # Which category folder the invoice was filed in ("1_Gelir_Faturalari").
    ("doc_type", "TEXT"),
)

#: Added when tahakkuk extraction replaced "store the PDF unparsed".
_DECLARATION_COLUMNS = (
    ("offset_amount", "REAL"),
    ("payable", "REAL"),
    ("due_date", "TEXT"),
    ("issue_date", "TEXT"),
    ("receipt_no", "TEXT"),
    ("taxpayer_tax_id", "TEXT"),
    ("lines", "TEXT"),
    ("doc_month", "INTEGER"),
    ("doc_type", "TEXT"),
)

#: documents/ predates the month level too.
_DOCUMENT_COLUMNS = (
    ("doc_month", "INTEGER"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing database up to the current schema. Idempotent."""
    for table, columns in (("invoices", _INVOICE_COLUMNS),
                           ("declarations", _DECLARATION_COLUMNS),
                           ("documents", _DOCUMENT_COLUMNS)):
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue  # schema.sql just created it with every column already present
        for column, decl in columns:
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # The identity key gained client_id. An invoice number is unique per issuer, so the
    # extra column never changes the outcome in practice -- it exists so that filing the
    # same PDF under two clients cannot have one client's ingest silently overwrite the
    # other's row. Dropping the old index is safe: the new key is strictly more specific,
    # so anything unique under the old one stays unique under this one.
    conn.execute("DROP INDEX IF EXISTS idx_invoices_identity")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_invoices_identity_client "
        "ON invoices (invoice_no, COALESCE(vendor_tax_id, ''), COALESCE(client_id, -1))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_invoices_client ON invoices (client_id, doc_year)"
    )
    conn.commit()


def upsert_invoice(conn: sqlite3.Connection, invoice: ExtractedInvoice) -> IngestResult:
    """Store one invoice. Re-ingesting an unchanged folder is a no-op.

    Identity is (invoice_no, vendor_tax_id). If a row with that identity already exists
    and carries the same content hash, nothing is written -- so ingest can be re-run
    freely. A changed hash means the source document itself changed, and the row is
    refreshed rather than duplicated.
    """
    if not invoice.invoice_no:
        raise ValueError(f"cannot store an invoice with no number: {invoice.source_path}")

    existing = conn.execute(
        "SELECT id, content_hash FROM invoices "
        "WHERE invoice_no = ? AND COALESCE(vendor_tax_id, '') = COALESCE(?, '') "
        "AND COALESCE(client_id, -1) = COALESCE(?, -1)",
        (invoice.invoice_no, invoice.vendor_tax_id, invoice.client_id),
    ).fetchone()

    if existing and existing["content_hash"] == invoice.content_hash:
        return IngestResult.SKIPPED

    payload = _to_payload(invoice)

    if existing:
        assignments = ", ".join(f"{name} = :{name}" for name in payload)
        conn.execute(
            f"UPDATE invoices SET {assignments} WHERE id = :id",
            {**payload, "id": existing["id"]},
        )
        # The text changed, so any cached embedding is stale.
        conn.execute("DELETE FROM embeddings WHERE invoice_id = ?", (existing["id"],))
        conn.commit()
        return IngestResult.UPDATED

    columns = ", ".join(payload)
    placeholders = ", ".join(f":{name}" for name in payload)
    conn.execute(f"INSERT INTO invoices ({columns}) VALUES ({placeholders})", payload)
    conn.commit()
    return IngestResult.INSERTED


def _to_payload(invoice: ExtractedInvoice) -> dict[str, Any]:
    payload: dict[str, Any] = {name: getattr(invoice, name) for name in FIELDS}
    payload.update(
        client_id=invoice.client_id,
        doc_year=invoice.doc_year,
        doc_month=invoice.doc_month,
        doc_type=invoice.doc_type,
        direction=invoice.direction,
        raw_text=invoice.raw_text,
        source_path=invoice.source_path,
        content_hash=invoice.content_hash or "",
        extraction_profile=invoice.profile,
        field_sources=json.dumps(invoice.field_sources, ensure_ascii=False),
        review_reasons=json.dumps(invoice.review_reasons, ensure_ascii=False),
        needs_review=int(invoice.needs_review),
        ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return payload


def all_invoices(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM invoices ORDER BY date, invoice_no").fetchall()


def flagged_invoices(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM invoices WHERE needs_review = 1 ORDER BY date, invoice_no"
    ).fetchall()


def count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS n FROM invoices").fetchone()["n"])
