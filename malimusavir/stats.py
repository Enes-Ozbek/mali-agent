"""Spend aggregates.

Deliberately computed with SQL and pandas, never routed through the LLM. "How much did
I spend" has one correct answer and arithmetic is not a language task -- a model that
is merely usually right about totals is worse than useless for accounting.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

#: A vendor billing this many times, at roughly monthly spacing, is a subscription.
RECURRING_MIN_INVOICES = 3
RECURRING_MIN_MEDIAN_DAYS = 20
RECURRING_MAX_MEDIAN_DAYS = 40


#: Sentinel for "the invoices belonging to no client" -- distinct from None, which means
#: "every client". Without it there is no way to ask for the unassigned bucket.
UNASSIGNED = "none"

#: Imported rather than redefined: clients.py owns these strings, and a drifting copy
#: here would silently split sales from purchases on the wrong value.
from .clients import PURCHASE, SALE  # noqa: E402


def load_frame(conn: sqlite3.Connection, client_id: int | str | None = None,
               year: int | None = None, month: int | None = None) -> pd.DataFrame:
    """All invoices as a DataFrame, with dates parsed and amounts numeric.

    ``client_id`` scopes to one client (``UNASSIGNED`` for the pre-client rows, None for
    all of them). Every aggregate in this module builds on this one query, so scoping
    here scopes totals, categories, months, vendors and recurring detection at once --
    and, more importantly, means a client's figures cannot accidentally include another's.
    """
    where, params = [], []
    if client_id == UNASSIGNED:
        where.append("client_id IS NULL")
    elif client_id is not None:
        where.append("client_id = ?")
        params.append(int(client_id))
    if year is not None:
        where.append("doc_year = ?")
        params.append(int(year))
    if month is not None:
        # The archive month folder, not the invoice's own date. Selecting "Ocak" in the
        # tree must show what is filed in that folder, including anything misfiled --
        # that is how a filing error becomes visible instead of silently moving.
        where.append("doc_month = ?")
        params.append(int(month))

    frame = pd.read_sql_query(
        "SELECT id, invoice_no, date, vendor, vendor_tax_id, total_amount, tax_amount, "
        "net_amount, category, currency, payment_method, needs_review, client_id, "
        "doc_year, doc_month, direction, source_path FROM invoices"
        + (" WHERE " + " AND ".join(where) if where else ""),
        conn,
        params=params,
    )
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")
    for column in ("total_amount", "tax_amount", "net_amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def date_range(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Filter to an inclusive ISO date window."""
    if frame.empty:
        return frame
    filtered = frame
    if start:
        filtered = filtered[filtered["date"] >= pd.Timestamp(start)]
    if end:
        filtered = filtered[filtered["date"] <= pd.Timestamp(end)]
    return filtered


@dataclass
class Totals:
    invoices: int
    total: float
    tax: float
    net: float
    first_date: str | None
    last_date: str | None
    flagged: int
    #: Currencies present. More than one means the totals above are meaningless --
    #: summing TL and EUR gives a number in no currency at all.
    currencies: tuple[str, ...] = ("TL",)

    @property
    def mixed_currency(self) -> bool:
        return len(self.currencies) > 1


def totals(frame: pd.DataFrame) -> Totals:
    if frame.empty:
        return Totals(0, 0.0, 0.0, 0.0, None, None, 0, ())
    dates = frame["date"].dropna()
    currencies = tuple(sorted(frame["currency"].dropna().unique())) or ("TL",)
    return Totals(
        invoices=len(frame),
        total=float(frame["total_amount"].sum()),
        tax=float(frame["tax_amount"].sum()),
        net=float(frame["net_amount"].sum()),
        first_date=dates.min().date().isoformat() if not dates.empty else None,
        last_date=dates.max().date().isoformat() if not dates.empty else None,
        flagged=int(frame["needs_review"].sum()),
        currencies=currencies,
    )


@dataclass
class VatSummary:
    """The KDV position for a period, as an accountant reads it.

    Derived from invoices only. It is NOT the same number as the tahakkuk fişi's
    `payable`, which is what the tax office actually assessed -- comparing the two is
    the point, so they are deliberately kept apart rather than merged into one figure.
    """

    income: float           #: Toplam Gelir -- sales, net of VAT
    expense: float          #: Toplam Gider -- purchases, net of VAT
    output_vat: float       #: Hesaplanan KDV -- VAT charged on sales
    input_vat: float        #: İndirilecek KDV -- VAT paid on purchases
    sales_count: int
    purchase_count: int
    #: True when no invoice is marked as a sale. `direction` defaults to "alis" unless
    #: the client's own tax_id is on file, so this is usually a missing-tax_id problem
    #: rather than a client who genuinely sold nothing -- and reporting 0,00 TL income
    #: as though it were a fact would be misleading.
    no_sales_recorded: bool = False

    @property
    def vat_balance(self) -> float:
        """Hesaplanan − İndirilecek. Positive is payable, negative carries forward."""
        return round(self.output_vat - self.input_vat, 2)

    @property
    def payable(self) -> float:
        """Ödenecek KDV -- zero when input VAT exceeds output."""
        return max(self.vat_balance, 0.0)

    @property
    def carried_forward(self) -> float:
        """Devreden KDV -- the excess input VAT carried to the next period."""
        return max(-self.vat_balance, 0.0)


def vat_summary(frame: pd.DataFrame) -> VatSummary:
    """Split a period into sales and purchases and compute the KDV position."""
    if frame.empty:
        return VatSummary(0.0, 0.0, 0.0, 0.0, 0, 0, no_sales_recorded=True)

    direction = frame["direction"].fillna(PURCHASE)
    sales = frame[direction == SALE]
    purchases = frame[direction != SALE]

    def _sum(part: pd.DataFrame, column: str) -> float:
        return round(float(part[column].fillna(0).sum()), 2)

    return VatSummary(
        income=_sum(sales, "net_amount"),
        expense=_sum(purchases, "net_amount"),
        output_vat=_sum(sales, "tax_amount"),
        input_vat=_sum(purchases, "tax_amount"),
        sales_count=len(sales),
        purchase_count=len(purchases),
        no_sales_recorded=sales.empty,
    )


def _grouped(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[column, "toplam", "adet", "ortalama"])
    grouped = (
        frame.groupby(frame[column].fillna("(bilinmiyor)"), dropna=False)["total_amount"]
        .agg(toplam="sum", adet="count", ortalama="mean")
        .reset_index()
        .sort_values("toplam", ascending=False)
    )
    return grouped


def by_category(frame: pd.DataFrame) -> pd.DataFrame:
    return _grouped(frame, "category")


def vendor_key(frame: pd.DataFrame) -> pd.Series:
    """Group vendors by tax id where known, falling back to name.

    Grouping on the name alone splits one seller into several rows whenever the name
    extracts slightly differently between invoices -- and company names *do* wrap
    across lines differently depending on the layout. The tax id is the stable
    identity; the name is only a label.
    """
    tax_id = frame["vendor_tax_id"].fillna("")
    name = frame["vendor"].fillna("(bilinmiyor)")
    return tax_id.where(tax_id != "", name)


def by_vendor(frame: pd.DataFrame) -> pd.DataFrame:
    """Spend per seller, keyed on tax id but labelled with the vendor name."""
    if frame.empty:
        return pd.DataFrame(columns=["vendor", "toplam", "adet", "ortalama"])
    working = frame.copy()
    working["_key"] = vendor_key(working)
    grouped = (
        working.groupby("_key")
        .agg(
            vendor=("vendor", lambda names: _best_label(names)),
            toplam=("total_amount", "sum"),
            adet=("total_amount", "count"),
            ortalama=("total_amount", "mean"),
        )
        .reset_index(drop=True)
        .sort_values("toplam", ascending=False)
    )
    return grouped


def _best_label(names: pd.Series) -> str:
    """The most complete name seen for a vendor -- longest wins over a truncation."""
    cleaned = [n for n in names.dropna().unique() if str(n).strip()]
    return max(cleaned, key=len) if cleaned else "(bilinmiyor)"


def by_month(frame: pd.DataFrame) -> pd.DataFrame:
    """Monthly spend, oldest first."""
    if frame.empty:
        return pd.DataFrame(columns=["ay", "toplam", "adet"])
    dated = frame.dropna(subset=["date"]).copy()
    dated["ay"] = dated["date"].dt.strftime("%Y-%m")
    return (
        dated.groupby("ay")["total_amount"]
        .agg(toplam="sum", adet="count")
        .reset_index()
        .sort_values("ay")
    )


def largest(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.nlargest(n, "total_amount")[
        ["date", "vendor", "category", "total_amount", "invoice_no"]
    ]


def _recurring_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """(vendor label, invoices) for every vendor billing on a roughly monthly cycle.

    Shared by recurring_vendors() and recurring_vendor_months() so both agree on
    exactly the same vendor set -- computing the recurring test twice, independently,
    would risk the two endpoints silently disagreeing on who's "recurring".
    """
    if frame.empty:
        return []
    dated = frame.dropna(subset=["date"]).copy()
    dated["_key"] = vendor_key(dated)
    groups = []
    for _, group in dated.groupby("_key"):
        if len(group) < RECURRING_MIN_INVOICES:
            continue
        gaps = group.sort_values("date")["date"].diff().dt.days.dropna()
        if gaps.empty:
            continue
        median_gap = float(gaps.median())
        if RECURRING_MIN_MEDIAN_DAYS <= median_gap <= RECURRING_MAX_MEDIAN_DAYS:
            groups.append((_best_label(group["vendor"]), group, median_gap))
    return groups


def recurring_vendors(frame: pd.DataFrame) -> pd.DataFrame:
    """Vendors that bill on a roughly monthly cycle.

    Subscriptions behave differently from one-off purchases -- they are committed
    future spend -- and this corpus is mostly telecom bills, so separating them is the
    question the data actually raises.
    """
    empty = pd.DataFrame(columns=["vendor", "adet", "toplam", "aylik_ortalama",
                                  "ortalama_gun"])
    rows = [
        {
            "vendor": label,
            "adet": len(group),
            "toplam": float(group["total_amount"].sum()),
            "aylik_ortalama": float(group["total_amount"].mean()),
            "ortalama_gun": round(median_gap, 1),
        }
        for label, group, median_gap in _recurring_groups(frame)
    ]
    if not rows:
        return empty
    return pd.DataFrame(rows).sort_values("toplam", ascending=False)


def recurring_vendor_months(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-vendor monthly totals for recurring vendors, for a sparkline.

    Same vendor set as recurring_vendors() (see _recurring_groups); one row per
    (vendor, month) that vendor actually billed in.
    """
    empty = pd.DataFrame(columns=["vendor", "ay", "toplam"])
    rows = []
    for label, group, _median_gap in _recurring_groups(frame):
        monthly = group.copy()
        monthly["ay"] = monthly["date"].dt.strftime("%Y-%m")
        for ay, month_group in monthly.groupby("ay"):
            rows.append({"vendor": label, "ay": ay, "toplam": float(month_group["total_amount"].sum())})
    if not rows:
        return empty
    return pd.DataFrame(rows).sort_values(["vendor", "ay"])


def one_off(frame: pd.DataFrame) -> pd.DataFrame:
    """Invoices from vendors that are not on a recurring cycle."""
    if frame.empty:
        return frame
    recurring = set(recurring_vendors(frame)["vendor"])
    labels = vendor_key(frame).map(
        lambda key: _best_label(frame.loc[vendor_key(frame) == key, "vendor"])
    )
    return frame[~labels.isin(recurring)]
