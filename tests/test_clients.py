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

# --- the optional <month>/ level ------------------------------------------------------
#
# The archive may file documents as Year/Month/Type or as Year/Type. Both are walked;
# the month is recorded when the folder says so and left None when it does not. It is
# never back-filled from the document's own date -- the field exists to say where the
# file physically sits, so inventing it would defeat the purpose.


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("Ocak", 1), ("ocak", 1), ("OCAK", 1), ("Aralık", 12), ("Şubat", 2),
        ("01", 1), ("1", 1), ("09", 9), ("12", 12),
        ("01-Ocak", 1), ("03_Mart", 3), ("Ocak 2026", 1),
        ("faturalar", None), ("beyannameler", None), ("tahakkuk", None),
        ("belgeler", None), ("13", None), ("0", None), ("", None),
    ],
)
def test_parse_month(folder, expected):
    assert archive.parse_month(folder) == expected


def test_walk_reads_the_month_folder(tmp_path):
    pdf = tmp_path / "mehmet" / "2026" / "Ocak" / "faturalar" / "a.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")

    result = archive.walk(tmp_path)
    assert len(result.items) == 1
    item = result.items[0]
    assert (item.year, item.month, item.month_folder) == (2026, 1, "Ocak")
    assert item.doc_type == "faturalar"
    assert item.kind == archive.Kind.INVOICE


def test_an_archive_without_months_still_walks(tmp_path):
    """The layout that existed before the month level must keep working."""
    pdf = tmp_path / "mehmet" / "2026" / "faturalar" / "a.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")

    result = archive.walk(tmp_path)
    assert len(result.items) == 1
    assert result.items[0].month is None
    assert not result.problems


def test_both_layouts_can_coexist_under_one_year(tmp_path):
    year = tmp_path / "mehmet" / "2026"
    (year / "Ocak" / "faturalar").mkdir(parents=True)
    (year / "Ocak" / "faturalar" / "a.pdf").write_bytes(b"%PDF-1.4")
    (year / "belgeler").mkdir(parents=True)
    (year / "belgeler" / "levha.pdf").write_bytes(b"%PDF-1.4")

    months = {i.path.name: i.month for i in archive.walk(tmp_path).items}
    assert months == {"a.pdf": 1, "levha.pdf": None}


def test_a_month_folder_with_no_type_folder_is_reported_not_guessed(tmp_path):
    """PDFs loose in a month folder have no document type; say so rather than
    inventing one."""
    loose = tmp_path / "mehmet" / "2026" / "Ocak" / "a.pdf"
    loose.parent.mkdir(parents=True)
    loose.write_bytes(b"%PDF-1.4")

    result = archive.walk(tmp_path)
    assert result.items == []
    assert any("belge türü" in p.reason for p in result.problems)


# --- the practice's standard folder convention ----------------------------------------
#
#   [VKN/TCKN - Ünvan]/<year>/<NN_Ay>/<N_Kategori>/
#
# Two things in that layout are load-bearing beyond navigation: the client folder
# carries the taxpayer's own tax number, and the category folder states whether the
# invoices inside are sales or purchases.


@pytest.mark.parametrize(
    ("folder", "kind"),
    [
        ("1_Gelir_Faturalari", archive.Kind.INVOICE),
        ("2_Gider_Faturalari", archive.Kind.INVOICE),
        ("3_Beyannameler", archive.Kind.DECLARATION),
        ("4_Tahakkuklar", archive.Kind.DECLARATION),
        ("5_Banka_Ekstreleri", archive.Kind.DOCUMENT),
        # the pre-numbering names must keep working
        ("faturalar", archive.Kind.INVOICE),
        ("tahakkuk", archive.Kind.DECLARATION),
        ("belgeler", archive.Kind.DOCUMENT),
    ],
)
def test_numbered_category_folders_classify(folder, kind):
    """The ordering prefix is for Explorer; it must not change the handler."""
    assert archive.classify_folder(folder) == kind


@pytest.mark.parametrize(
    ("folder", "direction"),
    [
        ("1_Gelir_Faturalari", clients.SALE),
        ("2_Gider_Faturalari", clients.PURCHASE),
        ("Satis Faturalari", clients.SALE),
        ("Alis Faturalari", clients.PURCHASE),
        ("faturalar", None),          # says nothing; fall back to the tax-id comparison
        ("5_Banka_Ekstreleri", None),
    ],
)
def test_folder_states_the_invoice_direction(folder, direction):
    assert archive.direction_for_folder(folder) == direction


@pytest.mark.parametrize(
    ("folder", "tax_id", "title"),
    [
        ("45678912345 - Canan Aydın E-Ticaret", "45678912345", "Canan Aydın E-Ticaret"),
        ("5556667778 - Kaya Yapı Ltd. Şti.", "5556667778", "Kaya Yapı Ltd. Şti."),
        ("[1020304050 - Zirve Lojistik]", "1020304050", "Zirve Lojistik"),
        ("1020304050_Zirve Lojistik", "1020304050", "Zirve Lojistik"),
        ("mehmet", None, None),        # no convention: still a valid client folder
        ("2026 - Bir Sey", None, None),  # 4 digits is not a tax number
    ],
)
def test_client_folder_carries_the_tax_number(folder, tax_id, title):
    parsed = archive.parse_client_folder(folder)
    assert (parsed.tax_id, parsed.title) == (tax_id, title)


def test_ingest_fills_the_tax_id_from_the_folder_name(tmp_path):
    """This is what makes Toplam Gelir work: without a tax_id every invoice would be
    classed as a purchase."""
    conn = db.connect(tmp_path / "conv.db")
    client = clients.resolve_folder(conn, "45678912345 - Canan Aydın E-Ticaret")
    assert client.tax_id == "45678912345"
    assert client.display == "Canan Aydın E-Ticaret"
    conn.close()


def test_a_renamed_folder_does_not_fork_the_client(tmp_path):
    """The ünvan gets corrected; the tax number does not. Matching on it keeps one
    client instead of silently creating a second."""
    conn = db.connect(tmp_path / "rename.db")
    first = clients.resolve_folder(conn, "45678912345 - Canan Aydin E-Ticaret")
    second = clients.resolve_folder(conn, "45678912345 - Canan Aydın E-Ticaret ve Danışmanlık")
    assert first.id == second.id
    assert len(clients.all_clients(conn)) == 1
    conn.close()


def test_an_operator_edit_survives_the_next_ingest(tmp_path):
    conn = db.connect(tmp_path / "edit.db")
    client = clients.resolve_folder(conn, "45678912345 - Eski Unvan")
    clients.set_metadata(conn, client.id, display="Elle Düzeltilmiş Ünvan")
    again = clients.resolve_folder(conn, "45678912345 - Eski Unvan")
    assert again.display == "Elle Düzeltilmiş Ünvan"
    conn.close()


def test_bank_statements_are_collected_even_though_they_are_not_pdfs(tmp_path):
    """Banks export .xlsx/.csv. PDF-only collection dropped them silently."""
    month = tmp_path / "111 - X" / "2026" / "01_Ocak"
    (month / "5_Banka_Ekstreleri").mkdir(parents=True)
    (month / "5_Banka_Ekstreleri" / "ocak.csv").write_text("a;b")
    (month / "5_Banka_Ekstreleri" / "ocak.xlsx").write_bytes(b"PK\x03\x04")

    found = {i.path.name for i in archive.walk(tmp_path).items}
    assert found == {"ocak.csv", "ocak.xlsx"}


def test_a_spreadsheet_in_an_invoice_folder_is_reported_not_parsed(tmp_path):
    """Invoices go through text extraction, so a non-PDF there is a filing mistake --
    say so rather than letting the PDF reader raise mid-run."""
    month = tmp_path / "111 - X" / "2026" / "01_Ocak"
    (month / "1_Gelir_Faturalari").mkdir(parents=True)
    (month / "1_Gelir_Faturalari" / "liste.xlsx").write_bytes(b"PK\x03\x04")

    result = archive.walk(tmp_path)
    assert result.items == []
    assert any("PDF değil" in p.reason for p in result.problems)


def test_walk_records_the_direction_off_the_category_folder(tmp_path):
    month = tmp_path / "45678912345 - X" / "2026" / "01_Ocak"
    for folder in ("1_Gelir_Faturalari", "2_Gider_Faturalari"):
        (month / folder).mkdir(parents=True)
        (month / folder / "a.pdf").write_bytes(b"%PDF-1.4")

    by_folder = {i.doc_type: i.direction for i in archive.walk(tmp_path).items}
    assert by_folder["1_Gelir_Faturalari"] == clients.SALE
    assert by_folder["2_Gider_Faturalari"] == clients.PURCHASE


def test_pretty_folder_strips_the_ordering_prefix():
    assert archive.pretty_folder("1_Gelir_Faturalari") == "Gelir Faturalari"
    assert archive.pretty_folder("5_Banka_Ekstreleri") == "Banka Ekstreleri"
    assert archive.pretty_folder("belgeler") == "belgeler"


# --- the archive is the source of truth, so the database has to follow it -------------


def _make_pdf(lines: list[str]) -> bytes:
    """Minimal one-page PDF, mirroring tests/test_pdf_text.py's builder."""
    text_ops = ["BT", "/F1 11 Tf", "50 740 Td", "14 TL"]
    for line in lines:
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        text_ops.append(f"({escaped}) Tj T*")
    text_ops.append("ET")
    stream = "\n".join(text_ops).encode("latin-1")

    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        b"<</Length %d>>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1, xref_at,
    )
    return bytes(out)


#: A real invoice PDF, because these tests run the extraction pipeline rather than the
#: walker alone -- a stub "%PDF-1.4" yields no invoice number and is never stored.
def _invoice_pdf_bytes(no="F1"):
    return _make_pdf([
        "ORNEK TEDARIK LIMITED SIRKETI", "e-ARSIV FATURA",
        f"FATURA NO: {no}", "FATURA TARIHI: 15.01.2026",
        "VERGI KIMLIK NO: 1234567890",
        "MAL/HIZMET TOPLAM TUTARI: 1.000,00 TL",
        "HESAPLANAN KDV(%20): 200,00 TL",
        "ODENECEK TUTAR: 1.200,00 TL", "PARA BIRIMI: TRY",
    ])


def _archive(tmp_path, client="111 - X", month="01_Ocak", name="a.pdf", no="F1"):
    folder = tmp_path / client / "2026" / month / "1_Gelir_Faturalari"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(_invoice_pdf_bytes(no))
    return folder / name


def test_moving_a_document_updates_its_month_and_path(tmp_path):
    """Identity is (client_id, content_hash), so INSERT OR IGNORE did nothing on a move:
    the row kept January and a path to a file that was no longer there."""
    src = tmp_path / "111 - X" / "2026" / "01_Ocak" / "4_Tahakkuklar" / "t.pdf"
    src.parent.mkdir(parents=True)
    src.write_bytes(_make_pdf(["TAHAKKUK FISI", "Ana Vergi Kodu 0015"]))

    conn = db.connect(tmp_path / "move.db")
    pipeline.ingest_archive(conn, tmp_path)
    assert conn.execute("SELECT doc_month FROM declarations").fetchone()["doc_month"] == 1

    dst = src.parent.parent.parent / "02_Şubat" / "4_Tahakkuklar" / "t.pdf"
    dst.parent.mkdir(parents=True)
    src.rename(dst)
    pipeline.ingest_archive(conn, tmp_path)

    row = conn.execute("SELECT doc_month, source_path FROM declarations").fetchone()
    assert row["doc_month"] == 2
    assert "02_" in row["source_path"]
    assert conn.execute("SELECT COUNT(*) n FROM declarations").fetchone()["n"] == 1
    conn.close()


def test_a_deleted_file_leaves_the_database(tmp_path):
    """Otherwise its amount keeps counting towards every total forever."""
    pdf = _archive(tmp_path)
    conn = db.connect(tmp_path / "prune.db")
    pipeline.ingest_archive(conn, tmp_path)
    assert conn.execute("SELECT COUNT(*) n FROM invoices").fetchone()["n"] == 1

    pdf.unlink()
    report = pipeline.ingest_archive(conn, tmp_path)
    assert conn.execute("SELECT COUNT(*) n FROM invoices").fetchone()["n"] == 0
    assert len(report.removed) == 1
    conn.close()


def test_pruning_never_touches_another_clients_rows(tmp_path):
    """--client mehmet must not delete everyone else. The scope guard is the whole
    reason this is safe to do automatically."""
    _archive(tmp_path, client="111 - Bir", no="A1")
    _archive(tmp_path, client="222 - Iki", no="B1")
    conn = db.connect(tmp_path / "scope.db")
    pipeline.ingest_archive(conn, tmp_path)
    assert conn.execute("SELECT COUNT(*) n FROM invoices").fetchone()["n"] == 2

    # Re-ingest only one client, with the other's files untouched on disk.
    report = pipeline.ingest_archive(conn, tmp_path, only_client="111 - Bir")
    assert conn.execute("SELECT COUNT(*) n FROM invoices").fetchone()["n"] == 2
    assert report.removed == []
    conn.close()


def test_pruning_is_confined_to_the_ingested_root(tmp_path):
    """Ingesting a different folder must not delete rows sourced from elsewhere."""
    _archive(tmp_path / "archive_a")
    conn = db.connect(tmp_path / "roots.db")
    pipeline.ingest_archive(conn, tmp_path / "archive_a")
    assert conn.execute("SELECT COUNT(*) n FROM invoices").fetchone()["n"] == 1

    other = tmp_path / "archive_b"
    _archive(other, client="333 - Uc", no="C1")
    report = pipeline.ingest_archive(conn, other)
    assert conn.execute("SELECT COUNT(*) n FROM invoices").fetchone()["n"] == 2
    assert report.removed == []
    conn.close()
