"""The web API: JSON safety, routing to the right answer source, upload isolation.

No live Foundry Local calls -- the RAG-fallback path is tested by monkeypatching
rag.answer to raise foundry.FoundryError, exactly like test_router.py avoids live
model calls for the router itself.
"""

from __future__ import annotations

import json

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
