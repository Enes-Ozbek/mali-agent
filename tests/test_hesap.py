"""Journal entries from invoices, against the Tekdüzen Hesap Planı.

The property that matters most here is that an unbalanced entry never leaves the
module. A ledger import that is out by a kuruş posts cleanly, looks right, and shows up
weeks later as a trial balance that will not close -- far worse than an export that
refused and said why.
"""

from __future__ import annotations

import pytest

from malimusavir import clients, db, hesap
from malimusavir.extractors.base import ExtractedInvoice


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "hesap.db")
    yield connection
    connection.close()


def add(conn, no, *, direction, net, tax, total=None, category="hizmet",
        client_id=None, year=2026, month=1):
    invoice = ExtractedInvoice(
        invoice_no=no, date=f"{year}-{month:02d}-10", vendor="Örnek Tedarik Ltd.",
        vendor_tax_id="1234567890", net_amount=net, tax_amount=tax,
        total_amount=total if total is not None else (
            None if net is None else round(net + (tax or 0), 2)),
        category=category, currency="TL", content_hash=f"h-{no}", profile="test",
    )
    invoice.client_id, invoice.doc_year, invoice.doc_month = client_id, year, month
    invoice.direction = direction
    db.upsert_invoice(conn, invoice)


# --- the standard double entry ---------------------------------------------------------


def test_a_sale_debits_the_customer_and_credits_revenue_and_output_vat(conn):
    add(conn, "S1", direction="satis", net=45000.0, tax=9000.0)
    entry = hesap.journal(conn).entries[0]

    posted = {(l.account, l.debit, l.credit) for l in entry.lines}
    assert posted == {
        ("120", 54000.0, 0.0),      # Alıcılar
        ("600", 0.0, 45000.0),      # Yurt İçi Satışlar
        ("391", 0.0, 9000.0),       # Hesaplanan KDV
    }
    assert entry.balanced


def test_a_purchase_debits_expense_and_input_vat_and_credits_the_supplier(conn):
    add(conn, "A1", direction="alis", net=100000.0, tax=20000.0)
    entry = hesap.journal(conn).entries[0]

    posted = {(l.account, l.debit, l.credit) for l in entry.lines}
    assert posted == {
        ("770", 100000.0, 0.0),     # Genel Yönetim Giderleri
        ("191", 20000.0, 0.0),      # İndirilecek KDV
        ("320", 0.0, 120000.0),     # Satıcılar
    }
    assert entry.balanced


def test_an_exempt_invoice_posts_without_a_vat_line(conn):
    """KDV'siz bir fatura still has to balance -- with two lines, not three."""
    add(conn, "E1", direction="alis", net=1000.0, tax=0.0)
    entry = hesap.journal(conn).entries[0]
    assert [l.account for l in entry.lines] == ["770", "320"]
    assert entry.balanced


def test_account_names_are_the_official_ones(conn):
    """A ledger import that matches on description needs the plan's own wording."""
    assert hesap.ACCOUNTS["191"] == "İndirilecek KDV"
    assert hesap.ACCOUNTS["391"] == "Hesaplanan KDV"
    assert hesap.ACCOUNTS["600"] == "Yurt İçi Satışlar"
    assert hesap.ACCOUNTS["770"] == "Genel Yönetim Giderleri"


# --- the refusal to post something wrong -------------------------------------------------


def test_an_invoice_whose_parts_do_not_sum_is_refused(conn):
    """Extraction can lift a figure from the wrong row of a totals table. Posting that
    produces a ledger that will not close."""
    add(conn, "B1", direction="alis", net=100.0, tax=20.0, total=999.0)
    report = hesap.journal(conn)
    assert report.entries == []
    assert report.rejected[0][0] == "B1"
    assert "tutmuyor" in report.rejected[0][1]


def test_an_invoice_with_no_total_is_refused(conn):
    add(conn, "B2", direction="alis", net=None, tax=None, total=None)
    report = hesap.journal(conn)
    assert report.entries == []
    assert "toplam" in report.rejected[0][1]


def test_one_bad_invoice_does_not_block_the_good_ones(conn):
    """An accountant exporting two hundred invoices must not be stopped by one."""
    add(conn, "OK1", direction="alis", net=100.0, tax=20.0)
    add(conn, "BAD", direction="alis", net=100.0, tax=20.0, total=999.0)
    add(conn, "OK2", direction="satis", net=200.0, tax=40.0)

    report = hesap.journal(conn)
    assert {e.invoice_no for e in report.entries} == {"OK1", "OK2"}
    assert [no for no, _ in report.rejected] == ["BAD"]


def test_every_emitted_entry_balances(conn):
    """The property the module exists to guarantee, over a mixed batch."""
    add(conn, "M1", direction="satis", net=1234.56, tax=246.91)
    add(conn, "M2", direction="alis", net=99.99, tax=20.0)
    add(conn, "M3", direction="alis", net=0.01, tax=0.0)

    report = hesap.journal(conn)
    assert len(report.entries) == 3
    assert all(e.balanced for e in report.entries)
    assert report.total_debit == pytest.approx(report.total_credit)


def test_a_missing_base_is_derived_rather_than_refused(conn):
    """Total and KDV are the two figures extraction reads most reliably; the base
    follows from them, so an invoice missing only that is still postable."""
    add(conn, "D1", direction="alis", net=None, tax=20.0, total=120.0)
    entry = hesap.journal(conn).entries[0]
    expense = next(l for l in entry.lines if l.account == "770")
    assert expense.debit == pytest.approx(100.0)
    assert entry.balanced


# --- the expense account is a decision, not a guess ---------------------------------------


def test_expenses_default_to_770(conn):
    """Where the great majority of a small taxpayer's costs belong."""
    assert hesap.expense_account("telekom")[0] == "770"
    assert hesap.expense_account(None)[0] == "770"
    assert hesap.expense_account("bilinmeyen-kategori")[0] == "770"


def test_a_large_equipment_purchase_is_capitalised_not_expensed(conn):
    """VUK md. 313: above the annual limit a fixed asset is depreciated, not written
    off in one year. Getting this wrong is an error a tax inspection finds."""
    account, note = hesap.expense_account("elektronik", net=40000.0)
    assert account == "255"
    assert "demirbaş" in note


def test_a_small_equipment_purchase_is_expensed_directly(conn):
    """Below the limit it may be written off, which is what most purchases are."""
    account, note = hesap.expense_account("elektronik", net=5000.0)
    assert (account, note) == ("770", None)


def test_the_limit_is_the_2026_figure_excluding_vat(conn):
    assert hesap.CAPITALISATION_LIMIT == 12_000.0
    assert hesap.expense_account("elektronik", net=12_000.0)[0] == "770"
    assert hesap.expense_account("elektronik", net=12_000.01)[0] == "255"


def test_the_limit_does_not_apply_outside_equipment_categories(conn):
    """A 50.000 TL consultancy invoice is still 770 -- it is not a fixed asset."""
    assert hesap.expense_account("hizmet", net=50000.0)[0] == "770"


def test_an_explicit_override_beats_capitalisation(conn):
    """The operator's stated intent wins over the default rule."""
    assert hesap.expense_account("elektronik", {"elektronik": "153"}, 40000.0)[0] == "153"


def test_a_capitalised_entry_carries_a_note_and_still_balances(conn):
    add(conn, "K1", direction="alis", net=40000.0, tax=8000.0, category="elektronik")
    report = hesap.journal(conn)
    entry = report.entries[0]
    assert [l.account for l in entry.lines] == ["255", "191", "320"]
    assert entry.balanced
    assert report.noted == [("K1", entry.note)]


def test_an_override_moves_a_category_to_another_account(conn):
    """153 vs 760 vs 770 depends on the business, not the invoice text -- so it is
    stated explicitly rather than inferred from a keyword."""
    add(conn, "T1", direction="alis", net=500.0, tax=100.0, category="market")
    report = hesap.journal(conn, overrides={"market": "153"})
    accounts = [l.account for l in report.entries[0].lines]
    assert "153" in accounts and "770" not in accounts


def test_an_override_to_an_unknown_account_is_ignored(conn):
    """A typo in configuration must not invent an account number."""
    assert hesap.expense_account("market", {"market": "999"})[0] == "770"


# --- scoping and export -------------------------------------------------------------------


def test_journal_scopes_to_client_year_and_month(conn):
    first = clients.resolve(conn, "111 - Bir").id
    second = clients.resolve(conn, "222 - Iki").id
    add(conn, "C1", direction="alis", net=10.0, tax=2.0, client_id=first, month=1)
    add(conn, "C2", direction="alis", net=20.0, tax=4.0, client_id=first, month=2)
    add(conn, "C3", direction="alis", net=30.0, tax=6.0, client_id=second, month=1)

    assert len(hesap.journal(conn).entries) == 3
    assert len(hesap.journal(conn, client_id=first).entries) == 2
    assert [e.invoice_no for e in
            hesap.journal(conn, client_id=first, year=2026, month=1).entries] == ["C1"]


def test_csv_uses_turkish_amounts_and_semicolons(conn):
    """Amounts carry commas as decimal separators, so the field separator cannot be one."""
    add(conn, "X1", direction="satis", net=1234.56, tax=246.91)
    text = hesap.to_csv(hesap.journal(conn))

    assert text.splitlines()[0].startswith("Tarih;Fiş No;Hesap Kodu")
    assert "1.481,47" in text          # 1234.56 + 246.91, Turkish formatted
    assert "1234.56" not in text


def test_csv_writes_one_row_per_line_not_per_invoice(conn):
    add(conn, "X1", direction="satis", net=100.0, tax=20.0)
    rows = hesap.to_csv(hesap.journal(conn)).splitlines()
    assert len(rows) == 4          # header + three postings


def test_csv_strips_semicolons_out_of_a_vendor_name(conn):
    """A stray separator inside a field would shift every column after it."""
    add(conn, "X1", direction="alis", net=100.0, tax=20.0)
    conn.execute("UPDATE invoices SET vendor = 'A ; B Ltd.' WHERE invoice_no = 'X1'")
    conn.commit()
    for row in hesap.to_csv(hesap.journal(conn)).splitlines()[1:]:
        assert len(row.split(";")) == len(hesap.CSV_COLUMNS)
