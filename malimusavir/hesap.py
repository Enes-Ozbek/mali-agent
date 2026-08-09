"""Tekdüzen Hesap Planı: turning an extracted invoice into a journal entry.

This is the bridge between "we read your PDFs" and "you can post this". A Turkish
practice keeps its books in Luca, Zirve or Mikro; an archive tool that cannot hand them
a fiş is a tool that sits beside the workflow instead of inside it.

An invoice becomes the standard double entry:

    Satış (gelir faturası)          Alış (gider faturası)
      120 Alıcılar          B tutar   770 Genel Yönetim Gid.  B matrah
        600 Yurt İçi Satış  A matrah  191 İndirilecek KDV     B kdv
        391 Hesaplanan KDV  A kdv       320 Satıcılar         A tutar

The load-bearing rule in this module is that **an entry that does not balance is never
emitted**. A ledger import that is out by a kuruş is worse than no import: it posts,
looks fine, and surfaces weeks later as a trial balance that will not close. Every
entry is checked before it leaves here, and anything that fails is reported by invoice
number instead of being written.

The expense account is the one genuinely business-specific choice. Everything defaults
to 770 because that is where the great majority of a small taxpayer's costs belong;
goods bought for resale (153) and marketing spend (760) are the exceptions an operator
has to make deliberately, so they are configurable rather than guessed from a keyword.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from .clients import SALE

#: Tekdüzen Hesap Planı, the accounts an invoice can touch. Names are the official
#: ones -- a ledger import matching on description needs them spelled as the plan does.
ACCOUNTS: dict[str, str] = {
    "100": "Kasa",
    "102": "Bankalar",
    "120": "Alıcılar",
    "153": "Ticari Mallar",
    "191": "İndirilecek KDV",
    "320": "Satıcılar",
    "360": "Ödenecek Vergi ve Fonlar",
    "391": "Hesaplanan KDV",
    "600": "Yurt İçi Satışlar",
    "610": "Satıştan İadeler (-)",
    "621": "Satılan Ticari Mallar Maliyeti (-)",
    "740": "Hizmet Üretim Maliyeti",
    "760": "Pazarlama, Satış ve Dağıtım Giderleri",
    "770": "Genel Yönetim Giderleri",
}

RECEIVABLE = "120"      #: what a customer owes us
PAYABLE = "320"         #: what we owe a supplier
SALES = "600"
OUTPUT_VAT = "391"      #: hesaplanan -- charged on our sales
INPUT_VAT = "191"       #: indirilecek -- paid on our purchases
DEFAULT_EXPENSE = "770"

#: Expense accounts an operator may legitimately choose between. Deliberately short:
#: offering the whole plan invites a wrong pick, and these are the ones that actually
#: differ for a small taxpayer.
EXPENSE_CHOICES = ("770", "760", "740", "153")

#: A kuruş. Anything past this is a real imbalance, not float noise.
TOLERANCE = 0.01


@dataclass(frozen=True)
class Line:
    """One side of a journal entry."""

    account: str
    name: str
    debit: float = 0.0      #: borç
    credit: float = 0.0     #: alacak


@dataclass
class Entry:
    """A balanced journal entry (yevmiye fişi) for one invoice."""

    invoice_id: int
    invoice_no: str
    date: str | None
    vendor: str | None
    direction: str
    lines: list[Line] = field(default_factory=list)

    @property
    def debit_total(self) -> float:
        return round(sum(l.debit for l in self.lines), 2)

    @property
    def credit_total(self) -> float:
        return round(sum(l.credit for l in self.lines), 2)

    @property
    def balanced(self) -> bool:
        return abs(self.debit_total - self.credit_total) <= TOLERANCE


@dataclass
class JournalReport:
    entries: list[Entry] = field(default_factory=list)
    #: (invoice_no, reason) for everything deliberately not posted.
    rejected: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_debit(self) -> float:
        return round(sum(e.debit_total for e in self.entries), 2)

    @property
    def total_credit(self) -> float:
        return round(sum(e.credit_total for e in self.entries), 2)


def expense_account(category: str | None, overrides: dict[str, str] | None = None) -> str:
    """Which expense account a purchase belongs to.

    Defaults to 770 for everything. Whether a cost is really 153 or 760 depends on the
    business, not on the invoice text, so it comes from an explicit per-category
    override rather than a keyword guess -- guessing here misposts silently and the
    error only shows up at the year end.
    """
    if overrides and category and category in overrides:
        account = overrides[category]
        if account in ACCOUNTS:
            return account
    return DEFAULT_EXPENSE


def entry_for(invoice: sqlite3.Row | dict,
              overrides: dict[str, str] | None = None) -> Entry | str:
    """Build the journal entry for one invoice, or return why it cannot be built."""
    row = dict(invoice)
    net, tax, total = row.get("net_amount"), row.get("tax_amount"), row.get("total_amount")

    if total is None:
        return "toplam tutar okunamadı"
    # A missing KDV line is normal on an exempt invoice; a missing base is not, because
    # the entry's expense/revenue side is exactly that figure.
    tax = 0.0 if tax is None else float(tax)
    if net is None:
        net = round(float(total) - tax, 2)
    net, total = float(net), float(total)

    if abs((net + tax) - total) > TOLERANCE:
        return f"matrah+KDV toplamı tutmuyor ({net} + {tax} ≠ {total})"

    entry = Entry(
        invoice_id=row.get("id", 0), invoice_no=row.get("invoice_no") or "?",
        date=row.get("date"), vendor=row.get("vendor"),
        direction=row.get("direction") or "alis",
    )

    if entry.direction == SALE:
        entry.lines.append(Line(RECEIVABLE, ACCOUNTS[RECEIVABLE], debit=total))
        entry.lines.append(Line(SALES, ACCOUNTS[SALES], credit=net))
        if tax:
            entry.lines.append(Line(OUTPUT_VAT, ACCOUNTS[OUTPUT_VAT], credit=tax))
    else:
        account = expense_account(row.get("category"), overrides)
        entry.lines.append(Line(account, ACCOUNTS[account], debit=net))
        if tax:
            entry.lines.append(Line(INPUT_VAT, ACCOUNTS[INPUT_VAT], debit=tax))
        entry.lines.append(Line(PAYABLE, ACCOUNTS[PAYABLE], credit=total))

    if not entry.balanced:
        return (f"fiş denk değil (borç {entry.debit_total} ≠ "
                f"alacak {entry.credit_total})")
    return entry


def journal(conn: sqlite3.Connection, *, client_id: int | str | None = None,
            year: int | None = None, month: int | None = None,
            overrides: dict[str, str] | None = None) -> JournalReport:
    """Journal entries for a scope, plus everything that could not be posted.

    Rejections are returned rather than raised: one unreadable invoice must not stop an
    accountant exporting the other two hundred, but it must not be silently dropped
    either -- it is listed by invoice number so it can be fixed.
    """
    from . import stats

    where, params = [], []
    if client_id == stats.UNASSIGNED:
        where.append("client_id IS NULL")
    elif client_id is not None:
        where.append("client_id = ?")
        params.append(int(client_id))
    if year is not None:
        where.append("doc_year = ?")
        params.append(int(year))
    if month is not None:
        where.append("doc_month = ?")
        params.append(int(month))

    sql = ("SELECT id, invoice_no, date, vendor, category, direction, "
           "       net_amount, tax_amount, total_amount, needs_review "
           "FROM invoices")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY date, invoice_no"

    report = JournalReport()
    for row in conn.execute(sql, params):
        built = entry_for(row, overrides)
        if isinstance(built, str):
            report.rejected.append((row["invoice_no"] or "?", built))
            continue
        report.entries.append(built)
    return report


#: Header of the exported file. Column names are the ones a Turkish ledger import
#: expects to see, so the file can be mapped without renaming anything first.
CSV_COLUMNS = ("Tarih", "Fiş No", "Hesap Kodu", "Hesap Adı", "Açıklama",
               "Borç", "Alacak")


def _tr_amount(value: float) -> str:
    """1234.5 -> "1.234,50" -- the format Turkish ledger software parses."""
    whole, cents = f"{value:,.2f}".split(".")
    return f"{whole.replace(',', '.')},{cents}"


def to_csv(report: JournalReport) -> str:
    """The journal as a semicolon-separated file.

    Semicolons, not commas: the amounts contain commas as decimal separators, and
    Turkish Excel opens `;` files directly. UTF-8 BOM is added by the caller so Excel
    detects the encoding instead of mangling every ş and ğ.
    """
    rows = [";".join(CSV_COLUMNS)]
    for entry in report.entries:
        for line in entry.lines:
            note = (entry.vendor or "").replace(";", " ").strip()
            rows.append(";".join([
                entry.date or "",
                entry.invoice_no,
                line.account,
                line.name,
                note,
                _tr_amount(line.debit) if line.debit else "",
                _tr_amount(line.credit) if line.credit else "",
            ]))
    return "\n".join(rows)
