"""The dashboard, driven in a real browser.

Every UI regression this project has had was invisible to the Python suite and to
tag-balance checks, because none of them were about whether the markup parsed:

  * the document preview rendered inside the tree rail at 208x157px -- a third child of
    a two-column grid, so CSS wrapped it onto an implicit second row
  * the Hesap Planı panel could never appear -- nested inside the Dosyalar branch, so
    its condition and its parent's were never true together
  * the client search sat 929px down a 720px viewport once the Gündem board landed
    above it: present in the DOM, gone from the screen

All three parsed cleanly. What they broke was *geometry* and *reachability*, which is
what these assert: that things are on screen, the right size, and reachable by clicking.

The server is started once per session against a temporary database seeded here, so
these never touch the developer's own archive.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")
expect = playwright_api.expect

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _seed(db_path: Path) -> None:
    """A practice with everything the UI has a panel for."""
    sys.path.insert(0, str(ROOT))
    from malimusavir import clients, db
    from malimusavir.extractors.base import ExtractedInvoice

    conn = db.connect(db_path)
    zeynep = clients.resolve(conn, "56473829102 - Zeynep Çelik Yazılım")
    clients.set_metadata(conn, zeynep.id, display="Zeynep Çelik Yazılım",
                         tax_id="56473829102")
    kaya = clients.resolve(conn, "5556667778 - Kaya Yapı")
    clients.set_metadata(conn, kaya.id, display="Kaya Yapı Ltd.", tax_id="5556667778")

    pdf = db_path.parent / "fatura.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    def invoice(client_id, no, direction, net, tax, month, doc_type, path=None):
        row = ExtractedInvoice(
            invoice_no=no, date=f"2026-{month:02d}-10", vendor="Örnek Tedarik Ltd.",
            vendor_tax_id="1234567890", net_amount=net, tax_amount=tax,
            total_amount=round(net + tax, 2), category="hizmet", currency="TL",
            content_hash=f"h-{no}", profile="test",
            source_path=str(path) if path else None,
        )
        row.client_id, row.doc_year, row.doc_month = client_id, 2026, month
        row.doc_type, row.direction = doc_type, direction
        db.upsert_invoice(conn, row)

    invoice(zeynep.id, "S1", "satis", 45000.0, 9000.0, 1, "1_Gelir_Faturalari", pdf)
    invoice(zeynep.id, "S2", "satis", 30000.0, 6000.0, 2, "1_Gelir_Faturalari")
    invoice(kaya.id, "A1", "alis", 100000.0, 20000.0, 1, "2_Gider_Faturalari", pdf)

    conn.execute(
        "INSERT INTO declarations (client_id, kind, period, payable, due_date, doc_year,"
        " doc_month, doc_type, source_path, content_hash, needs_review, ingested_at) "
        "VALUES (?, 'kdv', '2026-01', 9000.0, '2026-02-28', 2026, 1, '4_Tahakkuklar',"
        " ?, 'h-t1', 0, '2026-02-01T00:00:00')", (zeynep.id, str(pdf)))
    conn.commit()
    conn.close()


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    """The real app, on its own port, against a throwaway database."""
    workdir = tmp_path_factory.mktemp("ui")
    db_path = workdir / "ui.db"
    _seed(db_path)

    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "main.py", "--serve", "--db", str(db_path),
         "--port", str(port), "--no-model"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"

    for _ in range(120):
        if process.poll() is not None:
            pytest.fail(f"server exited: {process.stdout.read().decode('utf-8', 'replace')}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.25)
    else:
        process.kill()
        pytest.fail("server did not start")

    yield base
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture
def page(server, browser):
    """A loaded dashboard, with console errors turned into test failures."""
    context = browser.new_context(viewport={"width": 1280, "height": 720})
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(server, wait_until="networkidle")
    page.wait_for_selector(".crow")
    yield page
    context.close()
    assert not errors, f"uncaught JS errors: {errors}"


def open_client(page, name: str) -> None:
    page.locator(".crow", has_text=name).last.click()
    page.wait_for_selector("text=← Müşteriler")


def on_screen(page, selector: str) -> bool:
    """Visible *and* inside the viewport -- the distinction the search bug turned on."""
    box = page.locator(selector).first.bounding_box()
    if box is None:
        return False
    height = page.viewport_size["height"]
    return 0 <= box["y"] < height


# --- the landing page -------------------------------------------------------------------


def test_the_agenda_leads_with_what_needs_attention(page):
    expect(page.locator("text=Gündem")).to_be_visible()
    expect(page.locator("text=Vadesi geçti").first).to_be_visible()
    expect(page.locator("text=Zeynep Çelik").first).to_be_visible()


def test_an_overdue_filing_shows_its_own_due_date(page):
    """The date comes off the tahakkuk, not a calendar this code holds."""
    expect(page.locator("text=28.02.2026").first).to_be_visible()


def test_the_board_says_a_passed_date_is_not_an_unpaid_bill(page):
    """Nothing records payment. The wording has to carry that."""
    expect(page.locator("text=ödenmediği anlamına gelmez")).to_be_visible()


def test_a_client_row_carries_a_status(page):
    expect(page.locator(".crow", has_text="Kaya Yapı").last.locator(".pill")).to_be_visible()


# --- search: the regression that started this file ----------------------------------------


def test_the_search_box_is_on_screen_without_scrolling(page):
    """It was 929px down a 720px viewport -- in the DOM, off the screen."""
    assert on_screen(page, "#clientsearch")


def test_searching_filters_the_client_list(page):
    page.fill("#clientsearch", "zeynep")
    expect(page.locator("text=1 / 2")).to_be_visible()
    expect(page.locator(".crow", has_text="Zeynep Çelik").last).to_be_visible()


def test_searching_by_tax_number_works(page):
    page.fill("#clientsearch", "5556")
    expect(page.locator(".crow", has_text="Kaya Yapı").last).to_be_visible()


def test_the_match_is_actually_visible_not_just_filtered(page):
    """Finding the row is half of it: with the agenda still on top the match sat below
    the fold and the search looked like it had done nothing."""
    page.fill("#clientsearch", "kaya")
    page.wait_for_timeout(300)
    assert on_screen(page, "#clientlist")


def test_the_agenda_returns_when_the_search_is_cleared(page):
    page.fill("#clientsearch", "kaya")
    expect(page.locator("text=Gündem")).to_have_count(0)
    page.fill("#clientsearch", "")
    expect(page.locator("text=Gündem")).to_be_visible()


def test_slash_focuses_the_search(page):
    page.locator("body").click()
    page.keyboard.press("/")
    assert page.evaluate("document.activeElement.id") == "clientsearch"


def test_searching_from_inside_a_client_returns_to_the_list(page):
    open_client(page, "Zeynep Çelik")
    page.fill("#clientsearch", "kaya")
    expect(page.locator("text=← Müşteriler")).to_have_count(0)
    expect(page.locator(".crow", has_text="Kaya Yapı").last).to_be_visible()


# --- the client workspace -----------------------------------------------------------------


def test_opening_a_client_scopes_the_figures_to_them(page):
    open_client(page, "Zeynep Çelik")
    expect(page.locator("text=Zeynep Çelik Yazılım").first).to_be_visible()
    expect(page.locator("text=90.000,00").first).to_be_visible()    # 54.000 + 36.000


def test_the_tree_mirrors_the_archive(page):
    open_client(page, "Zeynep Çelik")
    expect(page.locator(".tnode", has_text="2026")).to_be_visible()
    page.locator(".tnode", has_text="Ocak").click()
    expect(page.locator(".tnode", has_text="Gelir Faturalari")).to_be_visible()


def test_selecting_a_month_narrows_the_summary(page):
    open_client(page, "Zeynep Çelik")
    page.locator(".tnode", has_text="Ocak").click()
    expect(page.locator("text=Ocak 2026").first).to_be_visible()
    expect(page.locator("text=45.000,00").first).to_be_visible()    # January only


def test_the_vat_panel_shows_the_position_not_just_the_tax(page):
    """Matched as written in the DOM, not as rendered.

    The labels are uppercased by CSS, and Playwright's case-insensitive text= cannot
    bridge that in Turkish: "İNDİRİLECEK".lower() keeps a combining dot and never
    equals "indirilecek". Asserting on the source text avoids the whole question.
    """
    open_client(page, "Zeynep Çelik")
    expect(page.locator("text=Ödenecek KDV").first).to_be_visible()
    expect(page.locator("text=Hesaplanan KDV").first).to_be_visible()
    expect(page.locator("text=İndirilecek KDV").first).to_be_visible()


def test_all_three_tabs_render_their_own_panel(page):
    """Hesap Planı was nested inside the Dosyalar branch and could never appear."""
    open_client(page, "Zeynep Çelik")

    page.locator("button", has_text="Dosyalar").click()
    expect(page.locator("text=Dosyalar").last).to_be_visible()

    page.locator("button", has_text="Hesap Planı").click()
    expect(page.locator("text=Varsayılanlar")).to_be_visible()

    page.locator("button", has_text="Genel Bakış").click()
    expect(page.locator("text=Mali Özet")).to_be_visible()


def test_the_journal_download_carries_the_selected_scope(page):
    open_client(page, "Zeynep Çelik")
    page.locator(".tnode", has_text="Ocak").click()
    page.wait_for_timeout(400)
    href = page.locator("a", has_text="Yevmiye indir").get_attribute("href")
    assert "year=2026" in href and "month=1" in href


# --- the document preview -------------------------------------------------------------------


def test_the_preview_opens_large_enough_to_read(page):
    """It rendered at 208x157 inside the tree rail: a third child of a two-column grid,
    wrapped onto an implicit second row."""
    open_client(page, "Zeynep Çelik")
    page.locator(".krow.crow").first.click()
    page.wait_for_selector(".sheet")

    box = page.locator(".sheet").bounding_box()
    assert box["width"] > 900, f"preview only {box['width']}px wide"
    assert box["height"] > 500, f"preview only {box['height']}px tall"


def test_the_preview_closes_on_escape(page):
    open_client(page, "Zeynep Çelik")
    page.locator(".krow.crow").first.click()
    page.wait_for_selector(".sheet")
    page.keyboard.press("Escape")
    expect(page.locator(".sheet")).to_have_count(0)


def test_clicking_inside_the_preview_does_not_close_it(page):
    open_client(page, "Zeynep Çelik")
    page.locator(".krow.crow").first.click()
    page.wait_for_selector(".sheet")
    page.locator(".sheethead").click()
    expect(page.locator(".sheet")).to_have_count(1)


def test_clicking_the_backdrop_closes_the_preview(page):
    open_client(page, "Zeynep Çelik")
    page.locator(".krow.crow").first.click()
    page.wait_for_selector(".sheet")
    page.locator(".ovl").click(position={"x": 5, "y": 5})
    expect(page.locator(".sheet")).to_have_count(0)


# --- the assistant -----------------------------------------------------------------------------


def test_the_assistant_sits_in_the_same_place_in_both_views(page):
    """It used to live inside the client branch and vanished on the landing page."""
    before = page.locator("text=Yardımcı").first.bounding_box()
    open_client(page, "Zeynep Çelik")
    after = page.locator("text=Yardımcı").first.bounding_box()
    assert abs(before["x"] - after["x"]) < 2


def test_fast_mode_answers_without_a_model(page):
    """--no-model: nothing is running, so anything needing one would fail here."""
    open_client(page, "Zeynep Çelik")
    page.locator("button", has_text="Hızlı").click()
    page.locator("button", has_text="KDV durumu").click()
    expect(page.locator("text=KDV durumu").last).to_be_visible(timeout=15_000)


def test_the_chips_follow_the_selected_period(page):
    open_client(page, "Zeynep Çelik")
    page.locator(".tnode", has_text="Ocak").click()
    expect(page.locator("button", has_text="Ocak KDV özeti")).to_be_visible()


# --- navigation ---------------------------------------------------------------------------------


def test_a_deadline_row_opens_that_client_at_that_month(page):
    page.locator(".crow", has_text="28.02.2026").first.click()
    page.wait_for_selector("text=← Müşteriler")
    expect(page.locator("text=Ocak 2026").first).to_be_visible()


def test_the_back_link_returns_to_the_list(page):
    open_client(page, "Zeynep Çelik")
    page.locator("button", has_text="← Müşteriler").click()
    expect(page.locator("text=Gündem")).to_be_visible()
