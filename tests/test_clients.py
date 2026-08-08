"""Clients, the archive walker, and — most importantly — scope isolation.

The isolation tests here are not style checks. In a practice, one client's figures or
documents surfacing under another is a confidentiality failure, so those cases assert on
real queries rather than on the presence of a filter argument.
"""

from __future__ import annotations

import pytest

from malimusavir import archive, clients, db, pipeline, rag, stats
from malimusavir.extractors.base import ExtractedInvoice


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "clients.db")
    yield connection
    connection.close()


def add_invoice(conn, client_id, no, total, *, date="2026-03-01", year=2026,
                vendor="Ornek Ltd.", tax_id="1112223334"):
    invoice = ExtractedInvoice(
        invoice_no=no, date=date, vendor=vendor, vendor_tax_id=tax_id,
        total_amount=total, tax_amount=round(total / 6, 2),
        net_amount=round(total * 5 / 6, 2), category="hizmet", currency="TL",
        content_hash=f"h-{no}", profile="test",
    )
    invoice.client_id = client_id
    invoice.doc_year = year
    db.upsert_invoice(conn, invoice)
    return invoice


# --- clients ----------------------------------------------------------------------


def test_resolve_creates_then_reuses(conn):
    first = clients.resolve(conn, "Mehmet")
    second = clients.resolve(conn, "Mehmet")
    assert first.id == second.id
    assert len(clients.all_clients(conn)) == 1


def test_resolve_is_case_and_accent_insensitive(conn):
    """Windows paths are case-insensitive; a re-cased folder must not fork the history."""
    a = clients.resolve(conn, "mehmet")
    b = clients.resolve(conn, "MEHMET")
    c = clients.resolve(conn, "Mehmet")
    assert a.id == b.id == c.id


def test_resolve_rejects_an_empty_name(conn):
    with pytest.raises(ValueError):
        clients.resolve(conn, "   ")


def test_label_prefers_the_display_name(conn):
    client = clients.resolve(conn, "mehmet")
    assert client.label == "mehmet"
    updated = clients.set_metadata(conn, client.id, display="Mehmet Yılmaz")
    assert updated.label == "Mehmet Yılmaz"


def test_set_metadata_ignores_unknown_fields(conn):
    client = clients.resolve(conn, "mehmet")
    updated = clients.set_metadata(conn, client.id, tax_id="1234567890", nonsense="x")
    assert updated.tax_id == "1234567890"


# --- sales vs purchases -------------------------------------------------------------


def test_direction_is_purchase_without_a_client_tax_id(conn):
    client = clients.resolve(conn, "mehmet")
    assert clients.direction_for(client, "9998887776") == clients.PURCHASE


def test_direction_is_sale_when_the_seller_is_the_client(conn):
    client = clients.resolve(conn, "mehmet")
    client = clients.set_metadata(conn, client.id, tax_id="1234567890")
    assert clients.direction_for(client, "1234567890") == clients.SALE
    assert clients.direction_for(client, "9998887776") == clients.PURCHASE


def test_direction_ignores_tax_id_formatting(conn):
    """VKNs are printed with spaces on real invoices ("265 017 9910")."""
    client = clients.resolve(conn, "mehmet")
    client = clients.set_metadata(conn, client.id, tax_id="123 456 7890")
    assert clients.direction_for(client, "1234567890") == clients.SALE


# --- migration and identity ---------------------------------------------------------


def test_unassigned_invoices_remain_valid(conn):
    """Rows from before clients existed keep working and are reachable on their own."""
    add_invoice(conn, None, "OLD1", 100.0)
    add_invoice(conn, None, "OLD2", 200.0)
    assert stats.totals(stats.load_frame(conn)).invoices == 2
    assert stats.totals(stats.load_frame(conn, client_id=stats.UNASSIGNED)).invoices == 2


def test_same_invoice_under_two_clients_is_two_rows(conn):
    """Misfiling one PDF under two clients must not let one overwrite the other."""
    a = clients.resolve(conn, "ahmet")
    m = clients.resolve(conn, "mehmet")
    add_invoice(conn, a.id, "SHARED1", 100.0)
    add_invoice(conn, m.id, "SHARED1", 100.0)
    assert db.count(conn) == 2


def test_reingesting_the_same_client_invoice_is_a_noop(conn):
    m = clients.resolve(conn, "mehmet")
    add_invoice(conn, m.id, "ONE", 100.0)
    add_invoice(conn, m.id, "ONE", 100.0)
    assert db.count(conn) == 1


# --- scoping ------------------------------------------------------------------------


def test_stats_scope_to_one_client(conn):
    a = clients.resolve(conn, "ahmet")
    m = clients.resolve(conn, "mehmet")
    add_invoice(conn, a.id, "A1", 1000.0)
    add_invoice(conn, m.id, "M1", 300.0)
    add_invoice(conn, m.id, "M2", 200.0)

    assert stats.totals(stats.load_frame(conn)).total == pytest.approx(1500.0)
    assert stats.totals(stats.load_frame(conn, client_id=a.id)).total == pytest.approx(1000.0)
    assert stats.totals(stats.load_frame(conn, client_id=m.id)).total == pytest.approx(500.0)


def test_stats_scope_by_year(conn):
    m = clients.resolve(conn, "mehmet")
    add_invoice(conn, m.id, "Y25", 100.0, date="2025-05-01", year=2025)
    add_invoice(conn, m.id, "Y26", 400.0, date="2026-05-01", year=2026)

    got = stats.totals(stats.load_frame(conn, client_id=m.id, year=2025))
    assert (got.invoices, got.total) == (1, pytest.approx(100.0))


def test_unassigned_is_distinct_from_all(conn):
    m = clients.resolve(conn, "mehmet")
    add_invoice(conn, m.id, "M1", 100.0)
    add_invoice(conn, None, "U1", 900.0)

    assert stats.totals(stats.load_frame(conn)).invoices == 2
    assert stats.totals(stats.load_frame(conn, client_id=stats.UNASSIGNED)).total == 900.0


def test_retrieval_cannot_cross_clients(conn, monkeypatch):
    """The confidentiality boundary: filtering happens in SQL, before any scoring, so
    an out-of-scope invoice can never enter the ranking regardless of similarity."""
    a = clients.resolve(conn, "ahmet")
    m = clients.resolve(conn, "mehmet")
    add_invoice(conn, a.id, "AHM1", 1000.0, vendor="Deniz Ambalaj Ltd.")
    add_invoice(conn, m.id, "MEH1", 200.0, vendor="Ornek Teknoloji Ltd.")

    # Deterministic stand-in for the embedding model.
    monkeypatch.setattr(rag.foundry, "embed",
                        lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts])
    rag.embed_pending(conn)

    for client, expected in ((a, "AHM1"), (m, "MEH1")):
        hits = rag.search(conn, "Deniz Ambalaj", client_id=client.id)
        assert [h["invoice_no"] for h in hits] == [expected]

    assert len(rag.search(conn, "Deniz Ambalaj")) == 2   # unscoped still sees both


def test_summaries_report_per_client_totals(conn):
    a = clients.resolve(conn, "ahmet")
    m = clients.resolve(conn, "mehmet")
    add_invoice(conn, a.id, "A1", 1000.0)
    add_invoice(conn, m.id, "M1", 500.0, year=2025, date="2025-01-01")
    add_invoice(conn, m.id, "M2", 500.0, year=2026)

    rows = {r["name"]: r for r in clients.summaries(conn)}
    assert rows["ahmet"]["invoices"] == 1
    assert rows["mehmet"]["total"] == pytest.approx(1000.0)
    assert rows["mehmet"]["years"] == [2026, 2025]


# --- archive walker -----------------------------------------------------------------


def build_tree(root):
    for rel in (
        "mehmet/2026/faturalar/a.pdf",
        "mehmet/2026/beyannameler/kdv.pdf",
        "mehmet/2026/sozlesmeler/kira.pdf",
        "mehmet/2025/faturalar/b.pdf",
        "ahmet/2026/faturalar/c.pdf",
        "ahmet/2026/loose.pdf",          # not in a type folder
        "ahmet/loose2.pdf",              # not in a year folder
        "ahmet/arsiv/faturalar/d.pdf",   # "arsiv" is not a year
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4\n")
    (root / "bos").mkdir(parents=True, exist_ok=True)   # client with no years


def test_walk_reads_client_year_and_type(tmp_path):
    build_tree(tmp_path)
    result = archive.walk(tmp_path)

    assert result.clients == ["ahmet", "mehmet"]
    kinds = {(i.client, i.year, i.doc_type): i.kind for i in result.items}
    assert kinds[("mehmet", 2026, "faturalar")] == archive.Kind.INVOICE
    assert kinds[("mehmet", 2026, "beyannameler")] == archive.Kind.DECLARATION
    assert kinds[("mehmet", 2026, "sozlesmeler")] == archive.Kind.DOCUMENT
    assert kinds[("mehmet", 2025, "faturalar")] == archive.Kind.INVOICE


def test_walk_reports_malformed_folders_instead_of_guessing(tmp_path):
    build_tree(tmp_path)
    reasons = " | ".join(p.reason for p in archive.walk(tmp_path).problems)

    assert "yıl klasörü içinde değil" in reasons     # PDF loose in the client folder
    assert "belge türü klasörü içinde değil" in reasons  # PDF loose in the year folder
    assert "yıl olarak okunamadı" in reasons         # "arsiv" not treated as a year
    assert "yıl klasörü yok" in reasons              # empty client
    # And none of those became invoices.
    assert all(i.doc_type != "arsiv" for i in archive.walk(tmp_path).items)


def test_walk_can_target_one_client(tmp_path):
    build_tree(tmp_path)
    assert archive.walk(tmp_path, only_client="MEHMET").clients == ["mehmet"]


@pytest.mark.parametrize(
    ("folder", "kind"),
    [("faturalar", archive.Kind.INVOICE), ("Faturalar", archive.Kind.INVOICE),
     ("FATURA", archive.Kind.INVOICE), ("beyannameler", archive.Kind.DECLARATION),
     ("Beyanname", archive.Kind.DECLARATION), ("sozlesmeler", archive.Kind.DOCUMENT),
     ("her ne ise", archive.Kind.DOCUMENT)],
)
def test_folder_classification(folder, kind):
    assert archive.classify_folder(folder) == kind


@pytest.mark.parametrize(
    ("name", "expected"),
    [("2026", 2026), ("1999", 1999), ("arsiv", None), ("20261", None),
     ("26", None), ("1800", None), ("2500", None)],
)
def test_year_parsing(name, expected):
    assert archive.parse_year(name) == expected


def test_missing_archive_root_is_reported(tmp_path):
    result = archive.walk(tmp_path / "yok")
    assert result.items == []
    assert result.problems and "bulunamadı" in result.problems[0].reason
