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
from .normalize import fold_tr

#: Tekdüzen Hesap Planı, the accounts an invoice can touch. Names are the official
#: ones -- a ledger import matching on description needs them spelled as the plan does.
ACCOUNTS: dict[str, str] = {
    "100": "Kasa",
    "102": "Bankalar",
    "120": "Alıcılar",
    "153": "Ticari Mallar",
    "191": "İndirilecek KDV",
    "255": "Demirbaşlar",
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
EXPENSE_CHOICES = ("770", "760", "740", "153", "255")

FIXED_ASSET = "255"

#: VUK md. 313: a fixed asset costing more than this, KDV hariç, cannot be written off
#: in the year it was bought -- it is capitalised to 255 Demirbaşlar and depreciated
#: over its useful life. 12.000 TL for 2026, and it is re-set annually, so it lives here
#: as one named constant rather than buried in a comparison.
CAPITALISATION_LIMIT = 12_000.0

#: Categories whose purchases can actually *be* fixed assets. The limit is meaningless
#: outside them: a 50.000 TL consultancy invoice is still 770, not equipment.
CAPITALISABLE = frozenset({"elektronik", "ofis"})

#: Where each category posts by default.
#:
#: Almost everything is 770, and that is the honest answer rather than a lazy one: for a
#: şahıs or small Ltd the overwhelming majority of costs really are genel yönetim
#: giderleri. The accounts that differ -- 153 for goods bought to resell, 760 for
#: marketing, 740 for the direct cost of services sold -- depend on what the business
#: does, not on what the invoice says, so they are left to an explicit override.
#:
#: The one place a default earns its keep is capitalisation, which is a rule rather than
#: a preference: see CAPITALISATION_LIMIT.
DEFAULT_CATEGORY_ACCOUNTS: dict[str, str] = {
    "telekom": "770", "enerji": "770", "abonelik": "770", "ofis": "770",
    "sigorta": "770", "sağlık": "770", "kitap-medya": "770", "ev": "770",
    "hizmet": "770", "market": "770", "yeme-içme": "770", "giyim": "770",
    "ulaşım": "770", "elektronik": "770", "diğer": "770",
}

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
    #: Set when the posting was not the plain default -- currently only capitalisation.
    #: Surfaced in the preview so an accountant sees the decision rather than
    #: discovering it in the ledger.
    note: str | None = None

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
    def noted(self) -> list[tuple[str, str]]:
        """(invoice_no, note) for entries posted somewhere other than the default."""
        return [(e.invoice_no, e.note) for e in self.entries if e.note]

    @property
    def total_debit(self) -> float:
        return round(sum(e.debit_total for e in self.entries), 2)

    @property
    def total_credit(self) -> float:
        return round(sum(e.credit_total for e in self.entries), 2)


def vendor_rules(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT tax_id, name_key, account, label FROM vendor_rules").fetchall()


def match_rule(rules, tax_id: str | None, vendor: str | None) -> str | None:
    """The account a supplier rule says to use, if one applies.

    Tax id first: it is the stable identifier, and a rule keyed on it keeps working
    when the seller's name arrives spelled differently on the next invoice.
    """
    digits = "".join(ch for ch in (tax_id or "") if ch.isdigit())
    if digits:
        for rule in rules:
            if rule["tax_id"] and rule["tax_id"] == digits:
                return rule["account"]

    key = fold_tr(vendor or "").strip()
    if key:
        for rule in rules:
            if rule["name_key"] and rule["name_key"] == key:
                return rule["account"]
    return None


def expense_account(category: str | None, overrides: dict[str, str] | None = None,
                    net: float | None = None,
                    rule_account: str | None = None) -> tuple[str, str | None]:
    """Which account a purchase belongs to, and a note when the choice is worth seeing.

    Most specific first:

    1. A supplier rule. A category is a guess about a kind of spend; a supplier is a
       fact about a counterparty, so it wins.
    2. An explicit category override -- whether a cost is really 153 or 760 depends on
       the business rather than the invoice text, so it is stated rather than inferred.
    3. Otherwise, a purchase in a category that *can* be equipment and costs more than
       the VUK limit is capitalised to 255. A rule, not a preference: writing a 40.000
       TL machine off in one year is an error a tax inspection finds.
    4. Otherwise the category default, which is 770 for everything.

    Where an operator has stated an account and the amount crosses the capitalisation
    limit anyway, their instruction stands and a note is raised instead. They are the
    professional; silently overriding them would be worse than flagging it. But letting
    a stale rule expense a fixed asset without a word would be worse still.
    """
    over_limit = net is not None and net > CAPITALISATION_LIMIT
    stated = rule_account if rule_account in ACCOUNTS else None
    if stated is None and overrides and category and category in overrides:
        stated = overrides[category] if overrides[category] in ACCOUNTS else None

    if stated is not None:
        if over_limit and stated != FIXED_ASSET and category in CAPITALISABLE:
            return stated, (
                f"{_tr_amount(net)} TL amortisman sınırının üzerinde — "
                f"demirbaş (255) olmalı mı, kontrol edin")
        return stated, None

    if category in CAPITALISABLE and over_limit:
        return FIXED_ASSET, (
            f"{_tr_amount(net)} TL > {_tr_amount(CAPITALISATION_LIMIT)} TL "
            f"amortisman sınırı — demirbaş olarak kaydedildi")

    return DEFAULT_CATEGORY_ACCOUNTS.get(category or "", DEFAULT_EXPENSE), None


def entry_for(invoice: sqlite3.Row | dict,
              overrides: dict[str, str] | None = None,
              rules=()) -> Entry | str:
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
        rule_account = match_rule(rules, row.get("vendor_tax_id"), row.get("vendor"))
        account, note = expense_account(row.get("category"), overrides, net,
                                        rule_account)
        entry.note = note
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

    sql = ("SELECT id, invoice_no, date, vendor, vendor_tax_id, category, direction, "
           "       net_amount, tax_amount, total_amount, needs_review "
           "FROM invoices")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY date, invoice_no"

    rules = vendor_rules(conn)
    report = JournalReport()
    for row in conn.execute(sql, params):
        built = entry_for(row, overrides, rules)
        if isinstance(built, str):
            report.rejected.append((row["invoice_no"] or "?", built))
            continue
        report.entries.append(built)
    return report


@dataclass
class RuleSuggestion:
    """A supplier seen often enough to be worth deciding about once."""

    tax_id: str | None
    name_key: str | None
    label: str
    invoices: int
    total: float
    current_account: str      #: where it posts today, without a rule


#: A supplier billing at least this often is a recurring relationship rather than a
#: one-off, and worth the operator spending ten seconds on. Below it the rule would
#: cost more attention than it saves.
SUGGEST_MIN_INVOICES = 2


def suggest_rules(conn: sqlite3.Connection, *, client_id: int | str | None = None,
                  limit: int = 20) -> list[RuleSuggestion]:
    """Suppliers that recur and have no rule yet, busiest first.

    Only purchases: a sale posts to 600 regardless of who the customer is, so a rule
    there would never change anything.
    """
    from . import stats

    clauses, params = ["COALESCE(direction, 'alis') != 'satis'"], []
    if client_id == stats.UNASSIGNED:
        clauses.append("client_id IS NULL")
    elif client_id is not None:
        clauses.append("client_id = ?")
        params.append(int(client_id))

    rows = conn.execute(
        "SELECT vendor_tax_id, vendor, category, COUNT(*) n, "
        "       COALESCE(SUM(total_amount), 0) total "
        "FROM invoices WHERE " + " AND ".join(clauses) +
        " GROUP BY COALESCE(vendor_tax_id, ''), COALESCE(vendor, '') "
        " ORDER BY n DESC, total DESC", params).fetchall()

    existing = vendor_rules(conn)
    out: list[RuleSuggestion] = []
    for row in rows:
        if row["n"] < SUGGEST_MIN_INVOICES:
            continue
        if match_rule(existing, row["vendor_tax_id"], row["vendor"]) is not None:
            continue          # already decided
        digits = "".join(ch for ch in (row["vendor_tax_id"] or "") if ch.isdigit())
        label = row["vendor"] or (digits and f"VKN {digits}") or "(satıcı adı okunamadı)"
        account, _ = expense_account(row["category"])
        out.append(RuleSuggestion(
            tax_id=digits or None,
            name_key=None if digits else (fold_tr(row["vendor"] or "").strip() or None),
            label=label, invoices=row["n"], total=float(row["total"]),
            current_account=account,
        ))
        if len(out) >= limit:
            break
    return [s for s in out if s.tax_id or s.name_key]


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
