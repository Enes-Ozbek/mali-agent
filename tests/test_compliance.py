"""Deadlines and document gaps.

The rules under test are mostly about restraint: use the date the tax office actually
set rather than one this code computed, do not call a passed date an unpaid bill, and do
not flag a period as missing a declaration before it was ever due.
"""

from __future__ import annotations

from datetime import date

import pytest

from malimusavir import clients, compliance, db
from malimusavir.extractors.base import ExtractedInvoice


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "compliance.db")
    yield connection
    connection.close()


def add_client(conn, name="45678912345 - Acme", display="Acme Ltd."):
    client = clients.resolve(conn, name)
    clients.set_metadata(conn, client.id, display=display, tax_id="45678912345")
    return client.id


def add_invoice(conn, client_id, no, year, month, path=None):
    invoice = ExtractedInvoice(
        invoice_no=no, date=f"{year}-{month:02d}-10", vendor="V", vendor_tax_id="9",
        total_amount=120.0, tax_amount=20.0, net_amount=100.0, category="hizmet",
        currency="TL", content_hash=f"h-{no}", profile="test",
        source_path=str(path) if path else None,
    )
    invoice.client_id, invoice.doc_year, invoice.doc_month = client_id, year, month
    db.upsert_invoice(conn, invoice)


def add_declaration(conn, client_id, *, period, due_date, payable=1000.0,
                    kind="kdv", year=2026, month=1, needs_review=0, path=None):
    # source_path is NOT NULL: every declaration row came from a file on disk. Tests
    # that do not care about the file still have to name one.
    source = str(path) if path else f"C:/arsiv/{period}-{kind}-{needs_review}.pdf"
    conn.execute(
        "INSERT INTO declarations (client_id, kind, period, payable, due_date, "
        "doc_year, doc_month, source_path, content_hash, needs_review, ingested_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-01-01T00:00:00')",
        (client_id, kind, period, payable, due_date, year, month, source,
         f"h-{period}-{kind}-{needs_review}", needs_review),
    )
    conn.commit()


# --- the receipt's own vade wins -------------------------------------------------------


def test_the_due_date_comes_from_the_receipt_not_a_calendar(conn):
    """GİB extends deadlines -- May 2026 KDV moved to 3 June. A receipt stating an
    extended date must be reported as it stands, not "corrected" to the 28th."""
    cid = add_client(conn)
    add_declaration(conn, cid, period="2026-05", due_date="2026-06-03", month=5)

    found = compliance.deadlines(conn, today=date(2026, 6, 1))
    assert [d.due_date for d in found] == ["2026-06-03"]
    assert found[0].days_left == 2


def test_a_flagged_receipt_never_reaches_the_deadline_board(conn):
    """Its amount was not verified. A figure nobody checked must not appear on a board
    people act on."""
    cid = add_client(conn)
    add_declaration(conn, cid, period="2026-01", due_date="2026-02-28", needs_review=1)
    assert compliance.deadlines(conn, today=date(2026, 2, 1)) == []


def test_a_malformed_due_date_is_skipped_rather_than_crashing(conn):
    cid = add_client(conn)
    add_declaration(conn, cid, period="2026-01", due_date="not-a-date")
    assert compliance.deadlines(conn, today=date(2026, 2, 1)) == []


# --- buckets ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("today", "bucket", "days_left"),
    [
        (date(2026, 3, 1), compliance.OVERDUE, -1),
        (date(2026, 2, 28), compliance.THIS_WEEK, 0),
        (date(2026, 2, 25), compliance.THIS_WEEK, 3),
        (date(2026, 2, 10), compliance.THIS_MONTH, 18),
        (date(2026, 1, 5), compliance.LATER, 54),
    ],
)
def test_buckets_follow_the_date(conn, today, bucket, days_left):
    cid = add_client(conn)
    add_declaration(conn, cid, period="2026-01", due_date="2026-02-28")
    found = compliance.deadlines(conn, today=today)[0]
    assert (found.bucket, found.days_left) == (bucket, days_left)


def test_total_due_is_assessed_not_outstanding(conn):
    """Nothing records payment, so this can only ever mean "assessed and due". The
    property name and the docstring both say so; this pins the arithmetic."""
    cid = add_client(conn)
    add_declaration(conn, cid, period="2026-01", due_date="2026-02-28", payable=500.0)
    add_declaration(conn, cid, period="2026-02", due_date="2026-03-28", payable=250.0,
                    month=2)
    assert compliance.overview(conn, today=date(2026, 3, 1)).total_due == pytest.approx(750.0)


# --- missing declarations ---------------------------------------------------------------


def test_the_current_period_is_not_called_missing(conn):
    """KDV for January is filed in February. Flagging January on 2 February would fire
    for every client every month and train the user to ignore the panel."""
    cid = add_client(conn)
    add_invoice(conn, cid, "F1", 2026, 1)
    assert [g for g in compliance.gaps(conn, today=date(2026, 2, 2))
            if g.reason == "missing_declaration"] == []


def test_a_period_past_its_deadline_and_grace_is_flagged(conn):
    cid = add_client(conn)
    add_invoice(conn, cid, "F1", 2026, 1)
    found = [g for g in compliance.gaps(conn, today=date(2026, 3, 20))
             if g.reason == "missing_declaration"]
    assert len(found) == 1
    assert (found[0].doc_year, found[0].doc_month) == (2026, 1)
    assert "2026-02-28" in found[0].detail


def test_the_grace_period_holds_just_after_the_deadline(conn):
    """GİB extensions are announced late and are usually days. Flagging the morning
    after would be wrong more often than right."""
    cid = add_client(conn)
    add_invoice(conn, cid, "F1", 2026, 1)
    just_after = [g for g in compliance.gaps(conn, today=date(2026, 3, 2))
                  if g.reason == "missing_declaration"]
    assert just_after == []


def test_a_period_with_a_declaration_is_not_a_gap(conn):
    cid = add_client(conn)
    add_invoice(conn, cid, "F1", 2026, 1)
    add_declaration(conn, cid, period="2026-01", due_date="2026-02-28", month=1)
    assert [g for g in compliance.gaps(conn, today=date(2026, 6, 1))
            if g.reason == "missing_declaration"] == []


def test_december_rolls_into_the_next_year(conn):
    assert compliance.statutory_due(2026, 12) == date(2027, 1, 28)
    assert compliance.statutory_due(2026, 1) == date(2026, 2, 28)


def test_muhtasar_is_due_two_days_before_kdv(conn):
    assert compliance.statutory_due(2026, 1, "kdv") == date(2026, 2, 28)
    assert compliance.statutory_due(2026, 1, "muhtasar") == date(2026, 2, 26)


# --- the other two kinds of gap ----------------------------------------------------------


def test_unreadable_declarations_are_reported_separately(conn):
    """A file we hold but cannot read is a different job from one we never received."""
    cid = add_client(conn)
    add_declaration(conn, cid, period="2026-01", due_date=None, needs_review=1)
    found = [g for g in compliance.gaps(conn, today=date(2026, 6, 1))
             if g.reason == "unreadable"]
    assert len(found) == 1 and found[0].count == 1


def test_a_file_that_left_the_archive_is_reported(conn, tmp_path):
    cid = add_client(conn)
    present = tmp_path / "there.pdf"
    present.write_bytes(b"%PDF-1.4")
    add_invoice(conn, cid, "F1", 2026, 1, path=present)
    add_invoice(conn, cid, "F2", 2026, 1, path=tmp_path / "gone.pdf")

    found = [g for g in compliance.gaps(conn, today=date(2026, 6, 1))
             if g.reason == "missing_file"]
    assert len(found) == 1
    assert found[0].count == 1          # only the absent one


# --- scoping ------------------------------------------------------------------------------


def test_everything_scopes_to_one_client(conn):
    first = add_client(conn, "111 - Bir", "Bir Ltd.")
    second = add_client(conn, "222 - Iki", "Iki Ltd.")
    add_declaration(conn, first, period="2026-01", due_date="2026-02-28")
    add_declaration(conn, second, period="2026-01", due_date="2026-02-28")

    assert len(compliance.deadlines(conn, today=date(2026, 2, 1))) == 2
    scoped = compliance.deadlines(conn, today=date(2026, 2, 1), client_id=first)
    assert len(scoped) == 1
    assert scoped[0].client_id == first


def test_deadlines_are_sorted_soonest_first(conn):
    cid = add_client(conn)
    add_declaration(conn, cid, period="2026-03", due_date="2026-04-28", month=3)
    add_declaration(conn, cid, period="2026-01", due_date="2026-02-28", month=1)
    found = compliance.deadlines(conn, today=date(2026, 1, 1))
    assert [d.due_date for d in found] == ["2026-02-28", "2026-04-28"]


def test_the_client_label_prefers_the_display_name(conn):
    cid = add_client(conn, "45678912345 - Acme", "Acme Danışmanlık")
    add_declaration(conn, cid, period="2026-01", due_date="2026-02-28")
    assert compliance.deadlines(conn, today=date(2026, 2, 1))[0].client_label \
        == "Acme Danışmanlık"
