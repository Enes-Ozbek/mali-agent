"""Storage: schema, dedupe and idempotent re-ingest."""

from __future__ import annotations

import json

import pytest

from malimusavir import db
from malimusavir.extractors.base import ExtractedInvoice


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    yield connection
    connection.close()


def make_invoice(**overrides) -> ExtractedInvoice:
    invoice = ExtractedInvoice(
        invoice_no="TST2026000000001",
        date="2026-03-03",
        vendor="Ornek Sirketi",
        vendor_tax_id="4590874863",
        total_amount=1536.0,
        tax_amount=256.0,
        net_amount=1280.0,
        vat_rate=20.0,
        currency="TL",
        payment_method="KREDIKARTI",
        category="elektronik",
        profile="generic_earsiv",
        source_path=r"C:\invoices\a.pdf",
        content_hash="hash-a",
        raw_text="Fatura No : TST2026000000001",
    )
    for name, value in overrides.items():
        setattr(invoice, name, value)
    return invoice


def test_insert_then_reingest_is_a_noop(conn):
    assert db.upsert_invoice(conn, make_invoice()) is db.IngestResult.INSERTED
    assert db.upsert_invoice(conn, make_invoice()) is db.IngestResult.SKIPPED
    assert db.count(conn) == 1


def test_changed_content_updates_in_place(conn):
    db.upsert_invoice(conn, make_invoice())
    changed = make_invoice(content_hash="hash-b", total_amount=1600.0)
    assert db.upsert_invoice(conn, changed) is db.IngestResult.UPDATED

    assert db.count(conn) == 1
    assert db.all_invoices(conn)[0]["total_amount"] == 1600.0


def test_same_invoice_number_from_different_sellers_coexist(conn):
    """Invoice numbers are only unique per issuer -- "1" is not a global key."""
    db.upsert_invoice(conn, make_invoice(invoice_no="1", vendor_tax_id="1111111111"))
    db.upsert_invoice(conn, make_invoice(invoice_no="1", vendor_tax_id="2222222222",
                                         content_hash="hash-b"))
    assert db.count(conn) == 2


def test_missing_tax_id_still_dedupes(conn):
    """SQLite treats NULLs as distinct, so a plain UNIQUE index would duplicate here."""
    db.upsert_invoice(conn, make_invoice(vendor_tax_id=None))
    db.upsert_invoice(conn, make_invoice(vendor_tax_id=None))
    assert db.count(conn) == 1


def test_provenance_is_stored(conn):
    invoice = make_invoice()
    invoice.field_sources = {"total_amount": "regex:generic_earsiv"}
    invoice.review_reasons = ["category:unresolved"]
    db.upsert_invoice(conn, invoice)

    row = db.all_invoices(conn)[0]
    assert json.loads(row["field_sources"])["total_amount"] == "regex:generic_earsiv"
    assert json.loads(row["review_reasons"]) == ["category:unresolved"]
    assert row["needs_review"] == 1
    assert row["extraction_profile"] == "generic_earsiv"
    assert db.flagged_invoices(conn)[0]["invoice_no"] == "TST2026000000001"


def test_invoice_without_a_number_is_rejected(conn):
    with pytest.raises(ValueError):
        db.upsert_invoice(conn, make_invoice(invoice_no=None))


def test_updating_clears_the_stale_embedding(conn):
    db.upsert_invoice(conn, make_invoice())
    invoice_id = db.all_invoices(conn)[0]["id"]
    conn.execute(
        "INSERT INTO embeddings (invoice_id, content_hash, summary, dim, vector) "
        "VALUES (?, ?, ?, ?, ?)",
        (invoice_id, "hash-a", "ozet", 2, b"\x00\x00\x00\x00"),
    )
    conn.commit()

    db.upsert_invoice(conn, make_invoice(content_hash="hash-b"))
    assert conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"] == 0
