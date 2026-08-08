"""The web API: JSON safety, routing to the right answer source, upload isolation.

No live Foundry Local calls -- the RAG-fallback path is tested by monkeypatching
rag.answer to raise foundry.FoundryError, exactly like test_router.py avoids live
model calls for the router itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from malimusavir import api, db, foundry
from malimusavir.extractors.base import ExtractedInvoice


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "api.db"
    monkeypatch.setattr(api, "DB_PATH", db_path)

    conn = db.connect(db_path)
    rows = [
        ("T1", "2025-06-20", "Turkcell İletişim Hizmetleri A.Ş.", "11", 230.0, 53.08,
         176.92, "telekom", 0, []),
        ("T2", "2025-07-20", "Turkcell İletişim Hizmetleri A.Ş.", "11", 232.0, 53.54,
         178.46, "telekom", 0, []),
        ("T3", "2025-08-20", "Turkcell İletişim Hizmetleri A.Ş.", "11", 308.0, 71.08,
         236.92, "telekom", 0, []),
        ("S1", "2025-07-31", "Turkcell Superonline İletişim Hizmetleri A.Ş.", "22",
         500.0, 112.76, 387.24, "telekom", 0, []),
        ("D1", "2026-05-13", "D-MARKET Elektronik Hizmetler ve Ticaret A.Ş.", "33",
         652.82, 108.80, 544.02, "ev", 1, ["category:unresolved"]),
    ]
    for no, when, vendor, tax_id, total, tax, net, category, flagged, reasons in rows:
        invoice = ExtractedInvoice(
            invoice_no=no, date=when, vendor=vendor, vendor_tax_id=tax_id,
            total_amount=total, tax_amount=tax, net_amount=net, category=category,
            currency="TL", content_hash=f"h-{no}", profile="test",
        )
        invoice.review_reasons = reasons
        db.upsert_invoice(conn, invoice)
    conn.close()

    return TestClient(api.app)


# --- JSON safety: numpy/pandas leakage would raise inside FastAPI's own validation --


def test_summary_fields_are_plain_json_types(client):
    body = client.get("/api/summary").json()
    assert body["invoices"] == 5
    assert body["total"] == pytest.approx(1922.82)
    assert body["tax"] == pytest.approx(399.26)
    assert body["net"] == pytest.approx(1523.56)
    assert body["first_date"] == "2025-06-20"
    assert body["last_date"] == "2026-05-13"
    assert body["flagged"] == 1
    assert body["currencies"] == ["TL"]
    assert body["mixed_currency"] is False


def test_by_category_returns_plain_numbers(client):
    rows = client.get("/api/by-category").json()
    telekom = next(r for r in rows if r["category"] == "telekom")
    assert type(telekom["toplam"]) is float
    assert type(telekom["adet"]) is int
    assert telekom["toplam"] == pytest.approx(1270.0)
    assert telekom["adet"] == 4


def test_by_month_returns_plain_numbers(client):
    rows = client.get("/api/by-month").json()
    assert all(type(r["toplam"]) is float and type(r["adet"]) is int for r in rows)


def test_by_vendor_returns_plain_numbers(client):
    rows = client.get("/api/by-vendor").json()
    assert all(type(r["toplam"]) is float and type(r["adet"]) is int for r in rows)


def test_largest_date_is_a_string_not_a_timestamp(client):
    """stats.largest() keeps date as a pandas Timestamp; the API must convert it."""
    rows = client.get("/api/largest?n=1").json()
    assert rows[0]["date"] == "2026-05-13"
    assert isinstance(rows[0]["date"], str)


def test_recurring_months_are_plain_numbers(client):
    rows = client.get("/api/recurring").json()
    turkcell = next(r for r in rows if "Superonline" not in r["vendor"])
    assert turkcell["adet"] == 3
    assert all(type(m["toplam"]) is float for m in turkcell["months"])


# --- /api/invoices ----------------------------------------------------------------


def test_invoices_newest_first(client):
    rows = client.get("/api/invoices").json()
    assert [r["invoice_no"] for r in rows[:2]] == ["D1", "T3"]


def test_invoices_limit_is_respected(client):
    rows = client.get("/api/invoices?limit=2").json()
    assert len(rows) == 2


def test_review_reasons_round_trips_as_a_json_array(client):
    """review_reasons is a JSON *string* column in SQLite; the API must parse it."""
    rows = client.get("/api/invoices").json()
    d1 = next(r for r in rows if r["invoice_no"] == "D1")
    assert d1["review_reasons"] == ["category:unresolved"]
    assert d1["needs_review"] is True
    t1 = next(r for r in rows if r["invoice_no"] == "T1")
    assert t1["review_reasons"] == []
    assert t1["needs_review"] is False


# --- /api/ask ----------------------------------------------------------------------


def test_ask_router_path_needs_no_model(client):
    r = client.post("/api/ask", json={"question": "toplam ne kadar harcadim"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "router"
    assert body["intent"] == "total"
    assert "1.922,82" in body["text"]


def test_ask_router_path_rows_are_json_safe(client):
    """by_category's rows come straight from a DataFrame -- numpy scalars included."""
    r = client.post("/api/ask", json={"question": "kategorilere gore ne kadar harcadim"})
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert rows and all(isinstance(row["toplam"], float) for row in rows)


def test_ask_empty_question_is_rejected(client):
    r = client.post("/api/ask", json={"question": "   "})
    assert r.status_code == 400


def test_ask_semantic_question_without_foundry_returns_503(client, monkeypatch):
    """The router can't answer this; the RAG fallback needs Foundry Local, which
    isn't running in tests. A 503 with a clear message, not a raw 500."""
    def boom(*args, **kwargs):
        raise foundry.FoundryError("Cannot reach Foundry Local at http://127.0.0.1:5267.")

    monkeypatch.setattr("malimusavir.api.rag.embed_pending", boom)
    r = client.post("/api/ask", json={"question": "vidalama seti hangi faturada var"})
    assert r.status_code == 503
    assert "Foundry Local" in r.json()["detail"]


# --- /api/chat ---------------------------------------------------------------------


def test_chat_fast_mode_needs_no_model(client):
    r = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "toplam ne kadar harcadim"}],
        "use_llm": False,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "router"
    assert "1.922,82" in body["text"]


def test_chat_grounds_the_model_and_returns_the_facts(client, monkeypatch):
    monkeypatch.setattr(
        "malimusavir.agent.foundry.chat_turns",
        lambda messages, **kw: "Toplam 1.922,82 TL harcadınız.",
    )
    r = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "toplam ne kadar harcadim"}],
    })
    body = r.json()
    assert body["source"] == "router+llm"
    assert body["text"] == "Toplam 1.922,82 TL harcadınız."
    # The computed line comes back too, so the UI can show what SQL actually returned.
    assert "1.922,82" in body["facts"]


def test_chat_follow_up_keeps_the_previous_vendor(client):
    r = client.post("/api/chat", json={
        "messages": [
            {"role": "user", "content": "Superonline'a ne kadar odedim"},
            {"role": "assistant", "content": "500,00 TL."},
            {"role": "user", "content": "peki kac fatura vardi"},
        ],
        "use_llm": False,
    })
    assert "Superonline" in r.json()["text"]


def test_chat_degrades_to_computed_text_when_the_model_is_down(client, monkeypatch):
    """A router-answerable question must still answer without Foundry Local."""
    def boom(*a, **k):
        raise foundry.FoundryError("Cannot reach Foundry Local.")

    monkeypatch.setattr("malimusavir.agent.foundry.chat_turns", boom)
    r = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "toplam ne kadar harcadim"}],
    })
    assert r.status_code == 200
    assert r.json()["source"] == "router"


def test_chat_semantic_question_without_foundry_returns_503(client, monkeypatch):
    """Retrieval genuinely cannot work without the model -- that one must surface."""
    def boom(*a, **k):
        raise foundry.FoundryError("Cannot reach Foundry Local.")

    monkeypatch.setattr("malimusavir.agent.rag.embed_pending", boom)
    r = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "vidalama seti hangi faturada"}],
    })
    assert r.status_code == 503


def test_chat_rejects_an_empty_question(client):
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "  "}]})
    assert r.status_code == 400


def test_chat_rejects_a_conversation_with_no_user_turn(client):
    r = client.post("/api/chat", json={
        "messages": [{"role": "assistant", "content": "merhaba"}],
    })
    assert r.status_code == 400


# --- /api/ingest ---------------------------------------------------------------


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


def test_ingest_isolates_a_bad_file_from_a_good_one(client):
    good = _make_pdf([
        "e-ARSIV FATURA", "Ornek Sirketi", "Vergi Kimlik No : 9998887770",
        "Fatura No : APITEST0000001", "Fatura Tarihi : 01.02.2026",
        "Odenecek Tutar 100,00 TL",
    ])
    files = [
        ("files", ("good.pdf", good, "application/pdf")),
        ("files", ("bad.pdf", b"not a pdf at all", "application/pdf")),
    ]
    r = client.post("/api/ingest", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["inserted"] == 1
    assert len(body["failed"]) == 1
    assert body["failed"][0]["path"].endswith("bad.pdf")


def test_ingest_rejects_non_pdf_filenames(client):
    r = client.post("/api/ingest", files=[("files", ("notes.txt", b"hi", "text/plain"))])
    assert r.status_code == 400


def test_ingest_then_invoices_reflects_the_new_row(client):
    good = _make_pdf([
        "e-ARSIV FATURA", "Ornek Sirketi", "Vergi Kimlik No : 9998887770",
        "Fatura No : APITEST0000002", "Fatura Tarihi : 01.02.2026",
        "Odenecek Tutar 250,00 TL",
    ])
    r = client.post("/api/ingest", files=[("files", ("good.pdf", good, "application/pdf"))])
    assert r.json()["inserted"] == 1

    rows = client.get("/api/invoices").json()
    assert any(row["invoice_no"] == "APITEST0000002" for row in rows)


# --- client scoping -----------------------------------------------------------------


@pytest.fixture
def multi_client(tmp_path, monkeypatch):
    """Two clients with distinct invoices, plus one unassigned row."""
    from malimusavir import clients

    db_path = tmp_path / "multi.db"
    monkeypatch.setattr(api, "DB_PATH", db_path)
    conn = db.connect(db_path)

    ahmet = clients.resolve(conn, "ahmet")
    mehmet = clients.resolve(conn, "mehmet")

    def add(client_id, no, total, vendor, year=2026, date="2026-03-01"):
        invoice = ExtractedInvoice(
            invoice_no=no, date=date, vendor=vendor, vendor_tax_id="1112223334",
            total_amount=total, tax_amount=round(total / 6, 2),
            net_amount=round(total * 5 / 6, 2), category="hizmet", currency="TL",
            content_hash=f"h-{no}", profile="test",
        )
        invoice.client_id = client_id
        invoice.doc_year = year
        db.upsert_invoice(conn, invoice)

    add(ahmet.id, "AHM1", 1000.0, "Deniz Ambalaj Ltd.")
    add(mehmet.id, "MEH1", 300.0, "Ornek Teknoloji Ltd.")
    add(mehmet.id, "MEH2", 200.0, "Ornek Teknoloji Ltd.", year=2025, date="2025-06-01")
    add(None, "ORPHAN", 77.0, "Eski Kayit Ltd.")
    conn.close()

    return TestClient(api.app), ahmet.id, mehmet.id


def test_clients_endpoint_reports_per_client_totals(multi_client):
    client, ahmet_id, mehmet_id = multi_client
    rows = {r["id"]: r for r in client.get("/api/clients").json()}
    assert rows[ahmet_id]["invoices"] == 1
    assert rows[ahmet_id]["total"] == pytest.approx(1000.0)
    assert rows[mehmet_id]["invoices"] == 2
    assert rows[mehmet_id]["total"] == pytest.approx(500.0)


def test_unassigned_invoices_are_still_reachable(multi_client):
    """Rows predating clients must not vanish from a client-centric UI."""
    client, _, _ = multi_client
    rows = {r["id"]: r for r in client.get("/api/clients").json()}
    assert rows[-1]["label"] == "(atanmamış)"
    assert rows[-1]["invoices"] == 1


def test_summary_scopes_to_one_client(multi_client):
    client, ahmet_id, mehmet_id = multi_client
    assert client.get("/api/summary").json()["total"] == pytest.approx(1577.0)
    assert client.get(f"/api/summary?client={ahmet_id}").json()["total"] == pytest.approx(1000.0)
    assert client.get(f"/api/summary?client={mehmet_id}").json()["total"] == pytest.approx(500.0)
    assert client.get("/api/summary?client=none").json()["total"] == pytest.approx(77.0)


def test_year_filter_scopes_further(multi_client):
    client, _, mehmet_id = multi_client
    body = client.get(f"/api/summary?client={mehmet_id}&year=2025").json()
    assert (body["invoices"], body["total"]) == (1, pytest.approx(200.0))


def test_invoices_endpoint_scopes(multi_client):
    client, ahmet_id, mehmet_id = multi_client
    got = [r["invoice_no"] for r in client.get(f"/api/invoices?client={mehmet_id}").json()]
    assert set(got) == {"MEH1", "MEH2"}
    assert "AHM1" not in got


def test_an_invalid_client_is_rejected_not_widened(multi_client):
    """Falling back to "all clients" on a bad id would leak figures across clients."""
    client, _, _ = multi_client
    assert client.get("/api/summary?client=abc").status_code == 400


def test_chat_scopes_to_the_requested_client(multi_client):
    client, ahmet_id, mehmet_id = multi_client
    for cid, expected in ((ahmet_id, "1.000,00"), (mehmet_id, "500,00")):
        body = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "toplam ne kadar harcadim"}],
            "use_llm": False, "client_id": cid,
        }).json()
        assert expected in body["text"]


def test_chat_cannot_retrieve_another_clients_invoices(multi_client, monkeypatch):
    """The confidentiality boundary, end to end through HTTP.

    The question names Ahmet's vendor explicitly; asked inside Mehmet's scope it must
    return none of Ahmet's invoices, however well they match.
    """
    client, ahmet_id, mehmet_id = multi_client
    monkeypatch.setattr(api.rag.foundry, "embed",
                        lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts])
    monkeypatch.setattr(api.rag.foundry, "chat_turns", lambda *a, **k: "cevap")

    body = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "Deniz Ambalaj faturasi hangisi"}],
        "client_id": mehmet_id,
    }).json()
    assert all(s["invoice_no"] != "AHM1" for s in body["sources"])


def test_declarations_and_documents_scope_by_client(multi_client):
    client, ahmet_id, mehmet_id = multi_client
    assert client.get(f"/api/clients/{ahmet_id}/declarations").json() == []
    assert client.get(f"/api/clients/{mehmet_id}/documents").json() == []


def test_updating_client_metadata_round_trips(multi_client):
    client, ahmet_id, _ = multi_client
    updated = client.post(f"/api/clients/{ahmet_id}",
                          json={"display": "Ahmet Yılmaz", "tax_id": "1234567890"}).json()
    assert updated["label"] == "Ahmet Yılmaz"
    assert updated["tax_id"] == "1234567890"


def test_updating_an_unknown_client_is_404(multi_client):
    client, _, _ = multi_client
    assert client.post("/api/clients/9999", json={"display": "x"}).status_code == 404


# --- original-file streaming: the Dosyalar view opens the actual archived PDF -------


@pytest.fixture
def archived_files(tmp_path, monkeypatch):
    """One invoice, one declaration and one document, each backed by a real file on
    disk -- the streaming endpoints read source_path straight off it, so a fixture
    with no file behind it would not exercise the actual FileResponse path."""
    from malimusavir import clients

    db_path = tmp_path / "files.db"
    monkeypatch.setattr(api, "DB_PATH", db_path)
    conn = db.connect(db_path)
    ahmet = clients.resolve(conn, "ahmet")

    pdf_path = tmp_path / "fatura.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake invoice bytes")
    invoice = ExtractedInvoice(
        invoice_no="F1", date="2026-01-10", vendor="Test Ltd.", vendor_tax_id="123",
        total_amount=100.0, tax_amount=20.0, net_amount=80.0, category="hizmet",
        currency="TL", content_hash="h-f1", profile="test", source_path=str(pdf_path),
    )
    invoice.client_id = ahmet.id
    db.upsert_invoice(conn, invoice)
    invoice_id = conn.execute(
        "SELECT id FROM invoices WHERE invoice_no = 'F1'").fetchone()["id"]

    decl_path = tmp_path / "tahakkuk.pdf"
    decl_path.write_bytes(b"%PDF-1.4 fake tahakkuk bytes")
    conn.execute(
        "INSERT INTO declarations (client_id, kind, period, doc_year, source_path, "
        "content_hash, needs_review, ingested_at) VALUES (?, 'kdv', '2026-01', 2026, "
        "?, 'h-d1', 0, '2026-01-01T00:00:00+00:00')",
        (ahmet.id, str(decl_path)),
    )
    declaration_id = conn.execute("SELECT id FROM declarations").fetchone()["id"]

    doc_path = tmp_path / "belge.pdf"
    doc_path.write_bytes(b"%PDF-1.4 fake document bytes")
    conn.execute(
        "INSERT INTO documents (client_id, doc_type, doc_year, filename, source_path, "
        "content_hash, ingested_at) VALUES (?, 'sozlesmeler', 2026, 'belge.pdf', ?, "
        "'h-doc1', '2026-01-01T00:00:00+00:00')",
        (ahmet.id, str(doc_path)),
    )
    document_id = conn.execute("SELECT id FROM documents").fetchone()["id"]
    conn.commit()
    conn.close()

    return TestClient(api.app), invoice_id, declaration_id, document_id


def test_invoice_file_streams_the_archived_pdf(archived_files):
    client, invoice_id, _, _ = archived_files
    r = client.get(f"/api/invoices/{invoice_id}/file")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content == b"%PDF-1.4 fake invoice bytes"


def test_declaration_file_streams_the_archived_pdf(archived_files):
    client, _, declaration_id, _ = archived_files
    r = client.get(f"/api/declarations/{declaration_id}/file")
    assert r.status_code == 200
    assert r.content == b"%PDF-1.4 fake tahakkuk bytes"


def test_document_file_streams_the_archived_pdf(archived_files):
    client, _, _, document_id = archived_files
    r = client.get(f"/api/documents/{document_id}/file")
    assert r.status_code == 200
    assert r.content == b"%PDF-1.4 fake document bytes"


def test_unknown_invoice_file_is_404(archived_files):
    client, _, _, _ = archived_files
    assert client.get("/api/invoices/9999/file").status_code == 404


def test_invoice_file_missing_from_disk_is_404_not_a_crash(archived_files, tmp_path):
    """The DB row can outlive the file (moved archive, deleted PDF) -- that must
    surface as a clean 404, not an unhandled FileResponse error."""
    client, invoice_id, _, _ = archived_files
    conn = db.connect(api.DB_PATH)
    conn.execute("UPDATE invoices SET source_path = ? WHERE id = ?",
                (str(tmp_path / "gone.pdf"), invoice_id))
    conn.commit()
    conn.close()
    assert client.get(f"/api/invoices/{invoice_id}/file").status_code == 404


# --- opening files in the desktop's own viewer --------------------------------------
#
# _launch is monkeypatched throughout: these tests must never actually spawn a PDF
# reader or an Explorer window on the machine running them.


@pytest.fixture
def launches(monkeypatch):
    """Record what would have been opened, instead of opening it."""
    calls = []
    monkeypatch.setattr(api, "_launch",
                        lambda path, *, reveal: calls.append((str(path), reveal)))
    return calls


def test_opening_an_invoice_hands_the_archived_path_to_the_os(archived_files, launches):
    client, invoice_id, _, _ = archived_files
    r = client.post(f"/api/invoices/{invoice_id}/open")
    assert r.status_code == 200
    assert r.json()["revealed"] is False
    assert len(launches) == 1
    assert launches[0][0].endswith("fatura.pdf")
    assert launches[0][1] is False


def test_reveal_asks_the_file_manager_to_select_it(archived_files, launches):
    client, invoice_id, _, _ = archived_files
    r = client.post(f"/api/invoices/{invoice_id}/open?reveal=true")
    assert r.status_code == 200
    assert r.json()["revealed"] is True
    assert launches[0][1] is True


def test_declarations_and_documents_open_too(archived_files, launches):
    client, _, declaration_id, document_id = archived_files
    assert client.post(f"/api/declarations/{declaration_id}/open").status_code == 200
    assert client.post(f"/api/documents/{document_id}/open").status_code == 200
    assert [Path(p).name for p, _ in launches] == ["tahakkuk.pdf", "belge.pdf"]


def test_opening_an_unknown_row_is_404(archived_files, launches):
    client, _, _, _ = archived_files
    assert client.post("/api/invoices/9999/open").status_code == 404
    assert not launches


def test_opening_a_file_that_left_the_disk_is_404(archived_files, launches, tmp_path):
    client, invoice_id, _, _ = archived_files
    conn = db.connect(api.DB_PATH)
    conn.execute("UPDATE invoices SET source_path = ? WHERE id = ?",
                 (str(tmp_path / "gone.pdf"), invoice_id))
    conn.commit()
    conn.close()
    assert client.post(f"/api/invoices/{invoice_id}/open").status_code == 404
    assert not launches


def test_a_non_pdf_source_path_is_refused(archived_files, launches, tmp_path):
    """The guard that stops this endpoint becoming a program launcher.

    _launch hands the path to the shell's default handler, so opening a .bat/.exe would
    *run* it. Nothing in the walker stores such a row today; this asserts the endpoint
    would still refuse if one ever appeared.
    """
    client, invoice_id, _, _ = archived_files
    nasty = tmp_path / "payload.bat"
    nasty.write_text("echo pwned")
    conn = db.connect(api.DB_PATH)
    conn.execute("UPDATE invoices SET source_path = ? WHERE id = ?",
                 (str(nasty), invoice_id))
    conn.commit()
    conn.close()

    assert client.post(f"/api/invoices/{invoice_id}/open").status_code == 400
    assert not launches, "a non-PDF must never reach the shell"


def test_no_registered_pdf_handler_reports_cleanly(archived_files, monkeypatch):
    """A desktop with nothing bound to .pdf raises OSError -- that is a 500 with a
    reason, not a stack trace, because /file still works as a fallback."""
    client, invoice_id, _, _ = archived_files

    def no_handler(path, *, reveal):
        raise OSError("no application is associated with this file")

    monkeypatch.setattr(api, "_launch", no_handler)
    r = client.post(f"/api/invoices/{invoice_id}/open")
    assert r.status_code == 500
    assert "açılamadı" in r.json()["detail"]


def test_the_open_endpoint_takes_no_path_from_the_caller(archived_files, launches):
    """Paths come from the row id only. A path-shaped query parameter must be ignored,
    not honoured -- otherwise any page in the browser could open arbitrary files."""
    client, invoice_id, _, _ = archived_files
    client.post(f"/api/invoices/{invoice_id}/open?path=C:/Windows/System32/calc.exe")
    assert launches[0][0].endswith("fatura.pdf")


def test_streamed_files_are_shown_inline_not_downloaded(archived_files):
    """A local app should not litter ~/Downloads with copies of files already on disk."""
    client, invoice_id, _, _ = archived_files
    r = client.get(f"/api/invoices/{invoice_id}/file")
    assert "inline" in r.headers["content-disposition"]
    assert "attachment" not in r.headers["content-disposition"]


# --- static mount -------------------------------------------------------------------


def test_root_serves_the_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_runtime_and_design_system_assets_resolve(client):
    """Regression guard for the mount-order/relative-path setup: the dashboard's
    `./support.js` and `_ds/.../styles.css` references must resolve from `/`."""
    assert client.get("/support.js").status_code == 200
    assert client.get(
        "/_ds/classical-03ce8088-f55b-4974-ac53-6ee0c3c447d4/styles.css"
    ).status_code == 200


def test_api_routes_are_not_shadowed_by_the_static_mount(client):
    """If StaticFiles were mounted before the /api/* routes, this would 404 as a
    missing static file instead of hitting the real endpoint."""
    r = client.get("/api/summary")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")


# --- VAT summary, tree and file presence ---------------------------------------------


@pytest.fixture
def vat_client(tmp_path, monkeypatch):
    """One client with a sale and a purchase in January, plus a KDV tahakkuk."""
    from malimusavir import clients

    db_path = tmp_path / "vat.db"
    monkeypatch.setattr(api, "DB_PATH", db_path)
    conn = db.connect(db_path)
    acme = clients.resolve(conn, "acme")
    clients.set_metadata(conn, acme.id, tax_id="1112223334")

    pdf = tmp_path / "satis.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    def add(no, direction, net, tax, month, path=None):
        invoice = ExtractedInvoice(
            invoice_no=no, date=f"2026-{month:02d}-10", vendor="V", vendor_tax_id="9",
            total_amount=net + tax, tax_amount=tax, net_amount=net, category="hizmet",
            currency="TL", content_hash=f"h-{no}", profile="test",
            source_path=str(path) if path else None,
        )
        invoice.client_id = acme.id
        invoice.doc_year = 2026
        invoice.doc_month = month
        invoice.direction = direction
        db.upsert_invoice(conn, invoice)

    add("S1", "satis", 10000.0, 2000.0, 1, pdf)     # sale: output VAT 2000
    add("P1", "alis", 4000.0, 800.0, 1)             # purchase: input VAT 800
    add("S2", "satis", 5000.0, 1000.0, 2)           # February, must not leak into Jan

    conn.execute(
        "INSERT INTO declarations (client_id, kind, period, payable, doc_year, "
        "doc_month, source_path, content_hash, needs_review, ingested_at) "
        "VALUES (?, 'kdv', '2026-01', 1200.0, 2026, 1, ?, 'h', 0, '2026-02-01T00:00:00')",
        (acme.id, str(tmp_path / "tahakkuk.pdf")),
    )
    conn.commit()
    conn.close()
    return TestClient(api.app), acme.id


def test_vat_summary_splits_sales_from_purchases(vat_client):
    client, cid = vat_client
    body = client.get(f"/api/vat-summary?client={cid}&year=2026&month=1").json()
    assert body["income"] == pytest.approx(10000.0)
    assert body["expense"] == pytest.approx(4000.0)
    assert body["output_vat"] == pytest.approx(2000.0)
    assert body["input_vat"] == pytest.approx(800.0)
    assert body["payable"] == pytest.approx(1200.0)
    assert body["carried_forward"] == pytest.approx(0.0)
    assert body["no_sales_recorded"] is False


def test_vat_summary_scopes_to_the_selected_month(vat_client):
    """February's sale must not appear in January's figures."""
    client, cid = vat_client
    jan = client.get(f"/api/vat-summary?client={cid}&year=2026&month=1").json()
    feb = client.get(f"/api/vat-summary?client={cid}&year=2026&month=2").json()
    assert jan["income"] == pytest.approx(10000.0)
    assert feb["income"] == pytest.approx(5000.0)


def test_excess_input_vat_carries_forward_instead_of_going_negative(vat_client):
    """Devreden KDV: a negative balance is carried, never shown as money owed."""
    client, cid = vat_client
    conn = db.connect(api.DB_PATH)
    conn.execute("UPDATE invoices SET tax_amount = 9000 WHERE invoice_no = 'P1'")
    conn.commit()
    conn.close()
    body = client.get(f"/api/vat-summary?client={cid}&year=2026&month=1").json()
    assert body["payable"] == pytest.approx(0.0)
    assert body["carried_forward"] == pytest.approx(7000.0)
    assert body["vat_balance"] == pytest.approx(-7000.0)


def test_the_assessed_figure_is_reported_beside_the_computed_one(vat_client):
    """The tahakkuk says 1.200,00 and the invoices imply 1.200,00 here -- but they are
    returned separately so a disagreement stays visible instead of being reconciled."""
    client, cid = vat_client
    body = client.get(f"/api/vat-summary?client={cid}&year=2026&month=1").json()
    assert body["assessed_vat"] == pytest.approx(1200.0)
    assert body["assessed_receipts"] == 1
    assert body["payable"] == pytest.approx(1200.0)


def test_no_sales_recorded_is_flagged_rather_than_reported_as_zero_income(tmp_path,
                                                                          monkeypatch):
    """A client with no tax_id has every invoice classed as a purchase, so "0,00 gelir"
    would be a missing-data artefact presented as fact."""
    from malimusavir import clients

    db_path = tmp_path / "nosales.db"
    monkeypatch.setattr(api, "DB_PATH", db_path)
    conn = db.connect(db_path)
    who = clients.resolve(conn, "notaxid")
    invoice = ExtractedInvoice(
        invoice_no="X1", date="2026-01-05", vendor="V", vendor_tax_id="9",
        total_amount=120.0, tax_amount=20.0, net_amount=100.0, category="hizmet",
        currency="TL", content_hash="h-x1", profile="test",
    )
    invoice.client_id = who.id
    invoice.doc_year = 2026
    invoice.direction = "alis"
    db.upsert_invoice(conn, invoice)
    conn.close()

    body = TestClient(api.app).get(f"/api/vat-summary?client={who.id}").json()
    assert body["income"] == pytest.approx(0.0)
    assert body["no_sales_recorded"] is True


def test_tree_mirrors_the_folder_layout(vat_client):
    client, cid = vat_client
    tree = client.get(f"/api/clients/{cid}/tree").json()
    assert [y["year"] for y in tree] == [2026]
    months = {m["label"]: m["count"] for m in tree[0]["months"]}
    assert months["Ocak"] == 3        # S1 + P1 + the KDV tahakkuk
    assert months["Şubat"] == 1
    ocak = next(m for m in tree[0]["months"] if m["label"] == "Ocak")
    assert {c["kind"] for c in ocak["categories"]} == {"invoice", "declaration"}


def test_tree_puts_month_less_documents_in_their_own_bucket_last(vat_client):
    """Licences filed under the year, not a month, must not be invented into one."""
    client, cid = vat_client
    conn = db.connect(api.DB_PATH)
    conn.execute(
        "INSERT INTO documents (client_id, doc_type, doc_year, doc_month, filename, "
        "source_path, content_hash, ingested_at) VALUES (?, 'belgeler', 2026, NULL, "
        "'levha.pdf', 'C:/a/2026/belgeler/levha.pdf', 'h', '2026-01-01T00:00:00')",
        (cid,),
    )
    conn.commit()
    conn.close()

    months = client.get(f"/api/clients/{cid}/tree").json()[0]["months"]
    assert months[-1]["label"] == "Ay belirtilmemiş"
    assert months[-1]["month"] is None
    assert months[-1]["categories"][0]["doc_type"] == "belgeler"


def test_file_exists_reflects_the_disk(vat_client, tmp_path):
    client, cid = vat_client
    rows = {r["invoice_no"]: r for r in
            client.get(f"/api/invoices?client={cid}&year=2026&month=1").json()}
    assert rows["S1"]["file_exists"] is True      # fixture wrote satis.pdf
    assert rows["P1"]["file_exists"] is False     # no source_path at all


def test_file_exists_turns_false_when_the_file_is_removed(vat_client, tmp_path):
    """The archive is the source of truth and moves underneath the database."""
    client, cid = vat_client
    (tmp_path / "satis.pdf").unlink()
    rows = {r["invoice_no"]: r for r in
            client.get(f"/api/invoices?client={cid}&year=2026&month=1").json()}
    assert rows["S1"]["file_exists"] is False


# --- selecting a category folder narrows the dashboard --------------------------------


@pytest.fixture
def foldered(tmp_path, monkeypatch):
    """One month laid out the way the practice files it."""
    from malimusavir import clients

    db_path = tmp_path / "folders.db"
    monkeypatch.setattr(api, "DB_PATH", db_path)
    conn = db.connect(db_path)
    who = clients.resolve_folder(conn, "5556667778 - Kaya Yapı Ltd. Şti")

    def add(no, folder, direction, net, tax):
        invoice = ExtractedInvoice(
            invoice_no=no, date="2026-01-10", vendor="V", vendor_tax_id="9",
            total_amount=net + tax, tax_amount=tax, net_amount=net, category="hizmet",
            currency="TL", content_hash=f"h-{no}", profile="test",
        )
        invoice.client_id, invoice.doc_year, invoice.doc_month = who.id, 2026, 1
        invoice.doc_type, invoice.direction = folder, direction
        db.upsert_invoice(conn, invoice)

    add("SAT1", "1_Gelir_Faturalari", "satis", 10000.0, 2000.0)
    add("ALS1", "2_Gider_Faturalari", "alis", 4000.0, 800.0)
    conn.execute(
        "INSERT INTO documents (client_id, doc_type, doc_year, doc_month, filename, "
        "source_path, content_hash, ingested_at) VALUES (?, '5_Banka_Ekstreleri', 2026, "
        "1, 'ocak.csv', 'C:/a/ocak.csv', 'h', '2026-02-01T00:00:00')", (who.id,))
    conn.commit()
    conn.close()
    return TestClient(api.app), who.id


def test_selecting_a_category_narrows_the_invoice_list(foldered):
    client, cid = foldered
    base = f"/api/invoices?client={cid}&year=2026&month=1"
    assert len(client.get(base).json()) == 2
    gelir = client.get(base + "&doc_type=1_Gelir_Faturalari").json()
    assert [r["invoice_no"] for r in gelir] == ["SAT1"]
    gider = client.get(base + "&doc_type=2_Gider_Faturalari").json()
    assert [r["invoice_no"] for r in gider] == ["ALS1"]


def test_selecting_a_category_narrows_the_vat_summary(foldered):
    """The centre panel must follow the tree, not just the invoice table."""
    client, cid = foldered
    base = f"/api/vat-summary?client={cid}&year=2026&month=1"
    both = client.get(base).json()
    assert (both["income"], both["expense"]) == (10000.0, 4000.0)

    gelir = client.get(base + "&doc_type=1_Gelir_Faturalari").json()
    assert gelir["income"] == pytest.approx(10000.0)
    assert gelir["expense"] == pytest.approx(0.0)


def test_selecting_a_document_category_yields_no_invoices(foldered):
    """5_Banka_Ekstreleri holds no invoices; the ledger must empty rather than keep
    showing the month's rows underneath a bank-statement heading."""
    client, cid = foldered
    rows = client.get(
        f"/api/invoices?client={cid}&year=2026&month=1&doc_type=5_Banka_Ekstreleri").json()
    assert rows == []
    docs = client.get(
        f"/api/clients/{cid}/documents?year=2026&month=1&doc_type=5_Banka_Ekstreleri").json()
    assert [d["filename"] for d in docs] == ["ocak.csv"]


def test_no_sales_is_not_warned_about_when_the_vkn_is_known(foldered):
    """A client that only bought this month is normal. The "no sales" warning exists
    for a missing VKN, which makes every invoice look like a purchase -- firing it when
    the tax id is on file would train the user to ignore it."""
    client, cid = foldered
    body = client.get(f"/api/vat-summary?client={cid}&year=2026&month=1"
                      "&doc_type=2_Gider_Faturalari").json()
    assert body["no_sales_recorded"] is True
    assert body["tax_id_missing"] is False


def test_a_missing_vkn_is_still_reported(tmp_path, monkeypatch):
    from malimusavir import clients

    db_path = tmp_path / "novkn.db"
    monkeypatch.setattr(api, "DB_PATH", db_path)
    conn = db.connect(db_path)
    who = clients.resolve(conn, "mehmet")          # plain folder, no tax number
    invoice = ExtractedInvoice(
        invoice_no="X1", date="2026-01-05", vendor="V", vendor_tax_id="9",
        total_amount=120.0, tax_amount=20.0, net_amount=100.0, category="hizmet",
        currency="TL", content_hash="h", profile="test",
    )
    invoice.client_id, invoice.doc_year, invoice.direction = who.id, 2026, "alis"
    db.upsert_invoice(conn, invoice)
    conn.close()

    body = TestClient(api.app).get(f"/api/vat-summary?client={who.id}").json()
    assert body["no_sales_recorded"] is True
    assert body["tax_id_missing"] is True


def test_the_tree_labels_strip_the_ordering_prefix(foldered):
    client, cid = foldered
    cats = client.get(f"/api/clients/{cid}/tree").json()[0]["months"][0]["categories"]
    by_type = {c["doc_type"]: c["label"] for c in cats}
    assert by_type["1_Gelir_Faturalari"] == "Gelir Faturalari"
    assert by_type["5_Banka_Ekstreleri"] == "Banka Ekstreleri"
