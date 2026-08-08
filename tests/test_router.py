"""Question routing: arithmetic to SQL, meaning to embeddings.

These run fully offline -- the router never calls a model, which is the point of it.
"""

from __future__ import annotations

from datetime import date

import pytest

from malimusavir import db, router
from malimusavir.extractors.base import ExtractedInvoice
from malimusavir.router import Intent

VENDORS = [
    "Turkcell Superonline İletişim Hizmetleri A.Ş.",
    "Turkcell İletişim Hizmetleri A.Ş.",
    "D-MARKET Elektronik Hizmetler ve Ticaret A.Ş.",
]
CATEGORIES = ["telekom", "elektronik", "ev"]
TODAY = date(2026, 5, 20)


def parse(question: str) -> router.Question:
    return router.classify(question, vendors=VENDORS, categories=CATEGORIES, today=TODAY)


# --- intent classification ------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("en son ne zaman alisveris yaptim", Intent.LAST),
        ("son ne zaman fatura aldim", Intent.LAST),
        ("ilk ne zaman alisveris yaptim", Intent.FIRST),
        ("toplam ne kadar harcadim", Intent.TOTAL),
        ("ne kadar para odedim", Intent.TOTAL),
        ("en pahali faturam hangisi", Intent.LARGEST),
        ("en ucuz fatura", Intent.SMALLEST),
        ("kac faturam var", Intent.COUNT),
        ("ne kadar kdv odedim", Intent.TAX),
        ("kategorilere gore harcamam", Intent.BY_CATEGORY),
        ("hangi firmaya ne kadar odedim", Intent.BY_VENDOR),
        ("aylik harcamam nedir", Intent.BY_MONTH),
        ("duzenli odemelerim neler", Intent.RECURRING),
        ("aboneliklerim neler", Intent.RECURRING),
    ],
)
def test_aggregate_intents(question, expected):
    assert parse(question).intent is expected


@pytest.mark.parametrize(
    "question",
    [
        "vidalama seti hangi faturada var",
        "kulaklik aldigim fatura hangisi",
        "hangi faturada sensor var",
        "bosch urunu aldim mi",
    ],
)
def test_item_questions_fall_through_to_search(question):
    """The router must never capture a question it cannot answer from SQL."""
    parsed = parse(question)
    assert parsed.intent is Intent.SEMANTIC
    assert not parsed.is_aggregate


def test_specific_phrases_beat_general_ones():
    """"en son ne zaman" must not be swallowed by the generic amount patterns."""
    assert parse("en son ne zaman ne kadar harcadim").intent is Intent.LAST


# --- slot extraction ------------------------------------------------------------


def test_ambiguous_vendor_matches_every_candidate():
    """"Turkcell" names two legal entities; answering for one silently is wrong."""
    parsed = parse("Turkcell'e toplam ne kadar odedim")
    assert set(parsed.vendors) == {
        "Turkcell Superonline İletişim Hizmetleri A.Ş.",
        "Turkcell İletişim Hizmetleri A.Ş.",
    }


def test_more_specific_vendor_name_wins():
    parsed = parse("Superonline'a ne kadar odedim")
    assert parsed.vendors == ["Turkcell Superonline İletişim Hizmetleri A.Ş."]


def test_turkish_case_suffixes_do_not_block_matching():
    for question in ("Turkcell'den gelen faturalar toplam ne kadar",
                     "Turkcell'e toplam ne kadar odedim"):
        assert parse(question).vendors


def test_generic_company_words_do_not_match():
    """"hizmetleri"/"ticaret" appear in most company names and identify nobody."""
    assert parse("hizmetleri toplam ne kadar").vendors == []


def test_category_is_detected():
    assert parse("telekom kategorisinde ne kadar harcadim").category == "telekom"


@pytest.mark.parametrize(
    ("question", "since", "until"),
    [
        ("2025 yilinda ne kadar harcadim", "2025-01-01", "2025-12-31"),
        ("bu yil ne kadar harcadim", "2026-01-01", "2026-12-31"),
        ("gecen yil ne kadar harcadim", "2025-01-01", "2025-12-31"),
        ("bu ay ne kadar harcadim", "2026-05-01", "2026-05-31"),
        ("gecen ay ne kadar harcadim", "2026-04-01", "2026-04-30"),
        ("mart 2026 ne kadar harcadim", "2026-03-01", "2026-03-31"),
        ("son 3 ay ne kadar harcadim", "2026-02-01", "2026-05-20"),
    ],
)
def test_period_extraction(question, since, until):
    parsed = parse(question)
    assert (parsed.since, parsed.until) == (since, until)


def test_no_period_means_no_filter():
    parsed = parse("toplam ne kadar harcadim")
    assert parsed.since is None and parsed.until is None


def test_list_intent_requires_something_to_filter_on():
    """A bare "hangi faturalar" is a search; with a category it is a query."""
    assert parse("telekom faturalarini listele").intent is Intent.LIST
    assert parse("hangi faturalarda ne var listele").intent is Intent.SEMANTIC


# --- answering ------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "router.db")
    rows = [
        ("T1", "2025-06-20", "Turkcell İletişim Hizmetleri A.Ş.", "11", 230.0, 53.08,
         176.92, "telekom"),
        ("T2", "2025-07-20", "Turkcell İletişim Hizmetleri A.Ş.", "11", 232.0, 53.54,
         178.46, "telekom"),
        ("T3", "2025-08-20", "Turkcell İletişim Hizmetleri A.Ş.", "11", 308.0, 71.08,
         236.92, "telekom"),
        ("S1", "2025-07-31", "Turkcell Superonline İletişim Hizmetleri A.Ş.", "22",
         500.0, 112.76, 387.24, "telekom"),
        ("D1", "2026-05-13", "D-MARKET Elektronik Hizmetler ve Ticaret A.Ş.", "33",
         652.82, 108.80, 544.02, "ev"),
    ]
    for no, when, vendor, tax_id, total, tax, net, category in rows:
        invoice = ExtractedInvoice(
            invoice_no=no, date=when, vendor=vendor, vendor_tax_id=tax_id,
            total_amount=total, tax_amount=tax, net_amount=net, category=category,
            currency="TL", content_hash=f"h-{no}", profile="test",
        )
        db.upsert_invoice(connection, invoice)
    yield connection
    connection.close()


def test_last_purchase_is_computed_not_generated(conn):
    """The failure this router exists to fix: the model invented a date."""
    answer = router.route(conn, "en son ne zaman alisveris yaptim")
    assert answer is not None
    assert "2026-05-13" in answer.text
    assert "D-MARKET" in answer.text


def test_total_matches_the_sum(conn):
    answer = router.route(conn, "toplam ne kadar harcadim")
    assert "1.922,82" in answer.text          # 230+232+308+500+652.82
    assert "5 fatura" in answer.text


def test_vendor_scoped_total(conn):
    answer = router.route(conn, "Superonline'a ne kadar odedim")
    assert "500,00" in answer.text
    assert "1 fatura" in answer.text


def test_ambiguous_vendor_total_covers_both_and_says_so(conn):
    answer = router.route(conn, "Turkcell'e toplam ne kadar odedim")
    assert "1.270,00" in answer.text          # 230+232+308+500
    assert "2 satıcı" in answer.text


def test_largest_invoice(conn):
    assert "652,82" in router.route(conn, "en pahali faturam hangisi").text


def test_count(conn):
    assert "5 fatura" in router.route(conn, "kac faturam var").text


def test_tax_total(conn):
    assert "399,26" in router.route(conn, "ne kadar kdv odedim").text


def test_category_breakdown(conn):
    text = router.route(conn, "kategorilere gore ne kadar harcadim").text
    assert "telekom" in text and "ev" in text


def test_period_filter_applies(conn):
    answer = router.route(conn, "2026 yilinda ne kadar harcadim")
    assert "652,82" in answer.text
    assert "1 fatura" in answer.text


def test_empty_scope_reports_no_invoices(conn):
    answer = router.route(conn, "2019 yilinda ne kadar harcadim")
    assert "bulunamadı" in answer.text


def test_semantic_question_returns_none(conn):
    """None is the signal to fall through to embeddings."""
    assert router.route(conn, "vidalama seti hangi faturada var") is None


# --- naming a client in the question -------------------------------------------------
#
# The bug these cover, reported from the live app: asked "canan aydının kaç faturası
# var" on the practice-wide page, the router answered with the *global* count (13) and
# the model presented it as "Canan Aydın'ın 13 faturası vardır". The number was real and
# the subject was wrong, which is worse than no answer -- so these assert both the
# figure and the name attached to it.


@pytest.fixture
def practice(tmp_path):
    """Two clients with different invoice counts, plus a client-owned vendor name."""
    from malimusavir import clients as clients_mod

    connection = db.connect(tmp_path / "practice.db")
    canan = clients_mod.resolve(connection, "Canan Aydın")
    clients_mod.set_metadata(connection, canan.id,
                             display="Canan Aydın E-Ticaret ve Danışmanlık")
    kaya = clients_mod.resolve(connection, "Kaya Yapı Mimarlık İnşaat Ltd. Şti")

    rows = [
        (canan.id, "C1", "2026-01-15", "Canan Aydın E-Ticaret", 3000.0),
        (canan.id, "C2", "2026-02-11", "Canan Aydın E-Ticaret", 2160.0),
        (canan.id, "C3", "2026-03-08", "Bir Tedarikçi Ltd.", 3840.0),
        (kaya.id, "K1", "2026-01-10", "Çelik Hazır Beton A.Ş.", 120000.0),
        (kaya.id, "K2", "2026-02-05", "Demir Çelik Malzeme", 78000.0),
    ]
    for client_id, no, when, vendor, total in rows:
        invoice = ExtractedInvoice(
            invoice_no=no, date=when, vendor=vendor, vendor_tax_id="1",
            total_amount=total, tax_amount=round(total / 6, 2),
            net_amount=round(total * 5 / 6, 2), category="hizmet", currency="TL",
            content_hash=f"h-{no}", profile="test",
        )
        invoice.client_id = client_id
        db.upsert_invoice(connection, invoice)

    yield connection, canan.id, kaya.id
    connection.close()


def test_a_named_client_scopes_the_count(practice):
    """The reported bug: this answered 5 (every client) and called it Canan's."""
    connection, _, _ = practice
    text = router.route(connection, "canan aydının kaç faturası var").text
    assert "3 fatura" in text
    assert "Canan Aydın" in text, "the figure must carry the name it belongs to"


def test_a_named_client_scopes_the_total(practice):
    connection, _, _ = practice
    text = router.route(connection, "kaya yapının toplamı ne kadar").text
    assert "198.000,00" in text          # 120000 + 78000, not the practice total
    assert "Kaya" in text


def test_an_unscoped_figure_says_it_covers_everyone(practice):
    """Without the scope label a global number reads as one client's, which is exactly
    how the reported answer went wrong."""
    connection, _, _ = practice
    text = router.route(connection, "toplam ne kadar harcandı").text
    assert "207.000,00" in text          # 9.000 (Canan) + 198.000 (Kaya)
    assert "tüm müşteriler" in text


def test_a_client_page_ignores_no_client_named(practice):
    connection, canan_id, _ = practice
    text = router.route(connection, "kaç faturam var", canan_id).text
    assert "3 fatura" in text
    assert "Canan Aydın" in text


def test_asking_about_another_client_from_a_clients_page_is_refused(practice):
    """The page scope is the confidentiality boundary; answering with the page's client
    under the name the user typed would repeat the wrong-subject bug."""
    connection, _, kaya_id = practice
    text = router.route(connection, "canan aydının kaç faturası var", kaya_id).text
    assert "3 fatura" not in text and "2 fatura" not in text
    assert "Kaya" in text and "Canan" in text


def test_naming_a_client_does_not_also_filter_by_their_own_vendor_name(practice):
    """A client's name is also the seller on their sales invoices. Left as a vendor
    filter it would drop C3, the one thing Canan bought rather than sold."""
    parsed = router.classify(
        "canan aydının kaç faturası var",
        vendors=router.known_vendors(practice[0]),
        categories=router.known_categories(practice[0]),
        clients=router.known_clients(practice[0]),
    )
    assert parsed.vendors == []
    assert [c.label for c in parsed.clients] == ["Canan Aydın E-Ticaret ve Danışmanlık"]


def test_listing_invoices_needs_no_vendor_or_category(practice):
    """"listele o faturaları" used to fall through to semantic search, where the model
    produced repetitive nonsense instead of a list."""
    connection, _, _ = practice
    answer = router.route(connection, "faturaları listele")
    assert answer is not None
    assert answer.intent is Intent.LIST
    assert "5 fatura" in answer.text


def test_a_content_question_still_goes_to_search_despite_a_list_word(practice):
    """Enumerating dates and totals cannot answer "what is in them"."""
    connection, _, _ = practice
    assert router.route(connection, "hangi faturalarda ne var listele") is None


def test_a_missing_vendor_name_is_not_printed_as_none(practice):
    """Extraction fails to find a seller on some şahıs layouts; str(None) would show a
    seller literally called "None"."""
    connection, _, _ = practice
    connection.execute("UPDATE invoices SET vendor = NULL WHERE invoice_no = 'C1'")
    connection.commit()
    text = router.route(connection, "faturaları listele").text
    assert "None" not in text
    assert "okunamadı" in text
