"""What needs attention: filing deadlines, and documents that should exist but don't.

An accounting practice does not start its day from a client list, it starts from what
is due. Everything this module reports is already in the database -- it is the one view
the archive could always have supported and never did.

Two rules shape the whole module, and both are about not stating more than is known:

*The receipt's own vade is authoritative.* A filed tahakkuk fişi carries the date the
tax office actually set, and that beats any calendar this code could hold. GİB extends
deadlines routinely -- May 2026's KDV and muhtasar were both pushed to 3 June -- so a
statutory rule is only ever used for a declaration that does not exist yet, and is
labelled an expectation rather than a fact.

*A passed due date is not an unpaid bill.* Nothing in this system records payment: the
tahakkuk states what was assessed, not whether it was settled. So an overdue item means
"vadesi geçti", never "ödenmedi". Bank statement reconciliation is what would close that
gap, and until it exists the honest word is the vaguer one.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

#: Statutory filing day of the month *following* the period, per tax kind. Used only to
#: work out whether a missing declaration is late enough to be worth flagging -- never
#: to display a date for a receipt that exists, which carries its own.
#: KDV is filed and paid by the 28th, muhtasar/MUHSGK by the 26th.
STATUTORY_FILING_DAY: dict[str, int] = {
    "kdv": 28,
    "muhtasar": 26,
    "gelir_stopaj": 26,
    "damga": 26,
}
DEFAULT_FILING_DAY = 28

#: Grace beyond the statutory day before a missing declaration is called a gap. GİB
#: extensions are usually days, not weeks, and flagging a client the morning after the
#: deadline -- when the extension may not even be announced yet -- trains the user to
#: ignore the panel.
GRACE_DAYS = 7

OVERDUE = "gecikmis"
THIS_WEEK = "bu_hafta"
THIS_MONTH = "bu_ay"
LATER = "ileri"


@dataclass
class Deadline:
    """One filing whose due date is known, from the receipt itself."""

    client_id: int
    client_label: str
    kind: str | None
    period: str | None
    payable: float | None
    due_date: str
    days_left: int          #: negative once the date has passed
    bucket: str
    declaration_id: int
    doc_year: int | None = None
    doc_month: int | None = None


@dataclass
class Gap:
    """Something an accountant would want chased."""

    client_id: int
    client_label: str
    reason: str             #: missing_declaration | unreadable | missing_file
    detail: str
    doc_year: int | None = None
    doc_month: int | None = None
    count: int = 1


@dataclass
class Overview:
    deadlines: list[Deadline] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)

    @property
    def overdue(self) -> list[Deadline]:
        return [d for d in self.deadlines if d.bucket == OVERDUE]

    @property
    def due_soon(self) -> list[Deadline]:
        return [d for d in self.deadlines if d.bucket in (THIS_WEEK, THIS_MONTH)]

    @property
    def total_due(self) -> float:
        """Assessed and not yet known to be paid. See the module docstring: this is
        what was assessed, not what is outstanding -- no payment record exists."""
        return round(sum(d.payable or 0.0 for d in self.deadlines
                         if d.bucket in (OVERDUE, THIS_WEEK, THIS_MONTH)), 2)


def _client_filter(client_id: int | str | None) -> tuple[str, list]:
    from . import stats

    if client_id == stats.UNASSIGNED:
        return "d.client_id IS NULL", []
    if client_id is None:
        return "1=1", []
    return "d.client_id = ?", [int(client_id)]


def _bucket(due: date, today: date) -> str:
    if due < today:
        return OVERDUE
    if due <= today + timedelta(days=7):
        return THIS_WEEK
    if due.year == today.year and due.month == today.month:
        return THIS_MONTH
    return LATER


def statutory_due(year: int, month: int, kind: str | None = None) -> date:
    """When a declaration for this period was due, by the ordinary rule.

    An expectation, not a fact: the tax office moves these. Used to decide whether a
    *missing* declaration is late, never to label one that exists.
    """
    day = STATUTORY_FILING_DAY.get(kind or "", DEFAULT_FILING_DAY)
    year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return date(year, month, day)


def deadlines(conn: sqlite3.Connection, *, today: date | None = None,
              client_id: int | str | None = None) -> list[Deadline]:
    """Filings with a known due date, soonest first.

    Only receipts that parsed cleanly are included. A flagged one may hold any figure,
    and putting an amount nobody has checked on a deadline board is how a wrong number
    ends up being acted on.
    """
    today = today or date.today()
    where, params = _client_filter(client_id)

    rows = conn.execute(
        "SELECT d.id, d.client_id, d.kind, d.period, d.payable, d.due_date, "
        "       d.doc_year, d.doc_month, "
        "       COALESCE(c.display, c.name) AS label "
        "FROM declarations d LEFT JOIN clients c ON c.id = d.client_id "
        f"WHERE {where} AND d.due_date IS NOT NULL AND d.needs_review = 0 "
        "ORDER BY d.due_date, d.id", params).fetchall()

    out: list[Deadline] = []
    for row in rows:
        try:
            due = date.fromisoformat(row["due_date"])
        except (TypeError, ValueError):
            continue          # a malformed date is a review problem, not a deadline
        out.append(Deadline(
            client_id=row["client_id"], client_label=row["label"] or "(atanmamış)",
            kind=row["kind"], period=row["period"], payable=row["payable"],
            due_date=row["due_date"], days_left=(due - today).days,
            bucket=_bucket(due, today), declaration_id=row["id"],
            doc_year=row["doc_year"], doc_month=row["doc_month"],
        ))
    return out


def gaps(conn: sqlite3.Connection, *, today: date | None = None,
         client_id: int | str | None = None) -> list[Gap]:
    """Documents that should be on file and are not.

    Three separate problems, deliberately not merged: a period with invoices but no
    declaration (chase the client), a declaration that could not be read (open it), and
    a row whose file has left the disk (the archive moved).
    """
    today = today or date.today()
    return (_missing_declarations(conn, today, client_id)
            + _unreadable_declarations(conn, client_id)
            + _missing_files(conn, client_id))


def _missing_declarations(conn, today: date, client_id) -> list[Gap]:
    where, params = _client_filter(client_id)
    where = where.replace("d.client_id", "i.client_id")

    rows = conn.execute(
        "SELECT i.client_id, i.doc_year, i.doc_month, COUNT(*) AS n, "
        "       COALESCE(c.display, c.name) AS label "
        "FROM invoices i LEFT JOIN clients c ON c.id = i.client_id "
        f"WHERE {where} AND i.doc_year IS NOT NULL AND i.doc_month IS NOT NULL "
        "  AND NOT EXISTS (SELECT 1 FROM declarations d "
        "                  WHERE d.client_id = i.client_id "
        "                    AND d.doc_year = i.doc_year "
        "                    AND d.doc_month = i.doc_month) "
        "GROUP BY i.client_id, i.doc_year, i.doc_month "
        "ORDER BY i.doc_year, i.doc_month", params).fetchall()

    out = []
    for row in rows:
        due = statutory_due(row["doc_year"], row["doc_month"])
        if today <= due + timedelta(days=GRACE_DAYS):
            # Not late yet. The current period always looks "missing" -- KDV is filed
            # the month after -- so flagging it would fire every month for every client.
            continue
        out.append(Gap(
            client_id=row["client_id"], client_label=row["label"] or "(atanmamış)",
            reason="missing_declaration",
            detail=f"{row['n']} fatura var, tahakkuk/beyanname yok "
                   f"(beklenen son gün {due.isoformat()})",
            doc_year=row["doc_year"], doc_month=row["doc_month"], count=row["n"],
        ))
    return out


def _unreadable_declarations(conn, client_id) -> list[Gap]:
    where, params = _client_filter(client_id)
    rows = conn.execute(
        "SELECT d.client_id, COUNT(*) AS n, COALESCE(c.display, c.name) AS label "
        "FROM declarations d LEFT JOIN clients c ON c.id = d.client_id "
        f"WHERE {where} AND d.needs_review = 1 "
        "GROUP BY d.client_id", params).fetchall()

    return [Gap(client_id=r["client_id"], client_label=r["label"] or "(atanmamış)",
                reason="unreadable",
                detail=f"{r['n']} belge okunamadı — tutar ve vade bilinmiyor",
                count=r["n"])
            for r in rows]


def _missing_files(conn, client_id) -> list[Gap]:
    """Rows whose PDF has left the archive.

    Checked against the disk rather than a stored flag: the archive is the source of
    truth and moves underneath the database between ingests.
    """
    where, params = _client_filter(client_id)
    missing: dict[tuple[int, str], int] = {}
    labels: dict[int, str] = {}

    for table in ("invoices", "declarations", "documents"):
        rows = conn.execute(
            f"SELECT d.client_id, d.source_path, COALESCE(c.display, c.name) AS label "
            f"FROM {table} d LEFT JOIN clients c ON c.id = d.client_id "
            f"WHERE {where} AND d.source_path IS NOT NULL", params).fetchall()
        for row in rows:
            try:
                if Path(row["source_path"]).is_file():
                    continue
            except OSError:
                pass
            key = (row["client_id"], table)
            missing[key] = missing.get(key, 0) + 1
            labels[row["client_id"]] = row["label"] or "(atanmamış)"

    return [Gap(client_id=cid, client_label=labels[cid], reason="missing_file",
                detail=f"{n} kaydın dosyası diskte bulunamadı ({table})", count=n)
            for (cid, table), n in sorted(missing.items())]


def overview(conn: sqlite3.Connection, *, today: date | None = None,
             client_id: int | str | None = None) -> Overview:
    today = today or date.today()
    return Overview(deadlines=deadlines(conn, today=today, client_id=client_id),
                    gaps=gaps(conn, today=today, client_id=client_id))
