"""FastAPI app: the JSON API the web dashboard consumes, plus the dashboard itself.

One process, one origin. `python main.py --serve` runs uvicorn against `app` here,
which answers `/api/*` and serves `web/` at `/`. Same-origin end to end, so there is
no CORS to configure.

Route order matters: every `/api/*` route must be declared before the static mount at
the bottom of this file, or the mount shadows them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from sqlite3 import Connection

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agent, archive, clients, db, foundry, pipeline, rag, router, stats

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

#: Set by main.py's --serve wiring before the server starts. None means db.DEFAULT_DB_PATH.
DB_PATH: str | Path | None = None


def get_conn():
    """One connection per request, closed after.

    check_same_thread=False is required, not optional: FastAPI resolves this sync
    generator dependency and then runs the route body as two separate calls into
    anyio's worker thread pool, and those calls are not guaranteed to land on the same
    OS thread (confirmed live -- this passed under TestClient's in-process portal but
    failed immediately under real uvicorn). It's safe because each request's
    connection is used strictly sequentially by that one request and is never shared
    across requests.
    """
    conn = db.connect(DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


def _json_safe(value):
    """Recursively convert numpy/pandas scalars the plain json encoder can't handle.

    Every stats.py function returns a pandas DataFrame; .to_dict("records") leaves
    numpy.int64/float64 inside the dicts, and router.py's Answer.rows is built from
    exactly those records for the by_category/by_vendor/by_month/recurring intents.
    Declared Pydantic response models coerce this automatically; router.py's rows
    field is typed as a plain dict and does not, so it is sanitized here explicitly.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


app = FastAPI(title="Mali Müşavir")


# ---- response models ---------------------------------------------------------------


class SummaryOut(BaseModel):
    invoices: int
    total: float
    tax: float
    net: float
    first_date: str | None
    last_date: str | None
    flagged: int
    currencies: list[str]
    mixed_currency: bool


class CategoryRow(BaseModel):
    category: str
    toplam: float
    adet: int
    ortalama: float


class MonthRow(BaseModel):
    ay: str
    toplam: float
    adet: int


class VendorRow(BaseModel):
    vendor: str
    toplam: float
    adet: int
    ortalama: float


class RecurringMonth(BaseModel):
    ay: str
    toplam: float


class RecurringRow(BaseModel):
    vendor: str
    adet: int
    toplam: float
    aylik_ortalama: float
    ortalama_gun: float
    months: list[RecurringMonth]


class LargestRow(BaseModel):
    date: str | None
    vendor: str | None
    category: str | None
    total_amount: float | None
    invoice_no: str


class InvoiceOut(BaseModel):
    id: int
    invoice_no: str
    date: str | None
    vendor: str | None
    vendor_tax_id: str | None
    total_amount: float | None
    tax_amount: float | None
    net_amount: float | None
    vat_rate: float | None
    currency: str | None
    payment_method: str | None
    category: str | None
    needs_review: bool
    review_reasons: list[str]
    extraction_profile: str | None
    source_path: str | None
    doc_year: int | None = None
    doc_month: int | None = None
    file_exists: bool = False


class VatSummaryOut(BaseModel):
    """The KDV position for the selected period, from invoices."""

    income: float
    expense: float
    output_vat: float
    input_vat: float
    vat_balance: float
    payable: float
    carried_forward: float
    sales_count: int
    purchase_count: int
    #: Set when nothing is marked as a sale. The UI must warn rather than present
    #: "0,00 TL gelir" as a fact -- it usually means the client's tax_id is missing,
    #: which makes direction_for() classify every invoice as a purchase.
    no_sales_recorded: bool
    #: Whether the client's own VKN/TCKN is on file. Without it direction_for() calls
    #: every invoice a purchase, so "no sales" means "we cannot tell". With it, no
    #: sales is simply a fact about the period and must not be warned about.
    tax_id_missing: bool = False
    #: What the tax office actually assessed for the same period, from tahakkuk fişi.
    #: Independent of everything above; the two disagreeing is the finding, so they are
    #: reported side by side rather than reconciled into one number.
    assessed_vat: float | None = None
    assessed_receipts: int = 0


class TreeCategory(BaseModel):
    doc_type: str          #: the folder name, verbatim -- what filters the dashboard
    #: The same name without its ordering prefix, for display. "1_Gelir_Faturalari"
    #: sorts correctly in Explorer but reads badly in a sidebar.
    label: str
    kind: str              #: invoice | declaration | document
    count: int


class TreeMonth(BaseModel):
    month: int | None      #: None means filed directly under the year
    label: str
    count: int
    categories: list[TreeCategory]


class TreeYear(BaseModel):
    year: int
    count: int
    months: list[TreeMonth]


class AskRequest(BaseModel):
    question: str


class AskSource(BaseModel):
    invoice_no: str
    date: str | None
    vendor: str | None
    total_amount: float | None
    score: float


class AskResponse(BaseModel):
    text: str
    intent: str
    rows: list[dict]
    sources: list[AskSource] = []
    source: str  # "router" | "rag"


class ClientRow(BaseModel):
    id: int
    name: str
    label: str
    display: str | None = None
    tax_id: str | None = None
    form: str | None = None
    city: str | None = None
    invoices: int
    total: float
    flagged: int
    last_activity: str | None = None
    years: list[int] = []


class ClientUpdate(BaseModel):
    display: str | None = None
    tax_id: str | None = None
    form: str | None = None
    city: str | None = None


class DeclarationLine(BaseModel):
    code: str
    kind: str | None = None
    matrah: float | None = None
    accrued: float | None = None
    offset: float | None = None
    payable: float | None = None
    due_date: str | None = None


class DeclarationRow(BaseModel):
    id: int
    kind: str | None = None
    period: str | None = None
    accrued: float | None = None
    offset_amount: float | None = None
    #: What the client actually owes -- the figure the UI leads with.
    payable: float | None = None
    due_date: str | None = None
    issue_date: str | None = None
    receipt_no: str | None = None
    taxpayer_tax_id: str | None = None
    lines: list[DeclarationLine] = []
    doc_year: int
    doc_month: int | None = None
    filename: str
    #: Whether source_path still resolves to a real file. The archive is the source of
    #: truth and can move underneath the database, so the UI shows this per row rather
    #: than only discovering it when someone clicks.
    file_exists: bool = False
    #: Where the PDF actually sits. Shown as the row tooltip -- for an app whose job is
    #: finding a client's paperwork, "which folder is this in" is half the answer.
    source_path: str | None = None
    needs_review: bool
    review_reasons: list[str] = []


class DocumentRow(BaseModel):
    id: int
    doc_type: str
    doc_year: int
    doc_month: int | None = None
    filename: str
    file_exists: bool = False
    source_path: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    use_llm: bool = True
    #: Scope the assistant to one client. Omitted means every client, matching the
    #: single-user behaviour this API had before clients existed.
    client_id: int | str | None = None


class ChatResponse(BaseModel):
    text: str
    source: str            # "router+llm" | "rag" | "router"
    intent: str
    facts: str | None = None
    sources: list[AskSource] = []


class IngestFailure(BaseModel):
    path: str
    reason: str


class IngestReportOut(BaseModel):
    inserted: int
    updated: int
    skipped: int
    failed: list[IngestFailure]
    flagged: list[str]


# ---- aggregate routes ----------------------------------------------------------


def _scope(client: str | None) -> int | str | None:
    """Turn a `?client=` query value into a load_frame/search scope.

    Absent -> every client (the pre-clients behaviour). "none" -> the unassigned bucket.
    Anything else must be an integer id; a bad value is rejected rather than silently
    widened to "all clients", which would leak one client's figures into another's page.
    """
    if client is None or client == "":
        return None
    if client in (stats.UNASSIGNED, "-1"):
        return stats.UNASSIGNED
    try:
        return int(client)
    except (TypeError, ValueError):
        raise HTTPException(400, f"invalid client: {client!r}") from None


def _scope_id(client_id: int | str | None) -> int | str | None:
    """Same rules as _scope, for a body field rather than a query parameter.

    The synthetic id -1 the client list uses for the unassigned bucket maps back to the
    UNASSIGNED sentinel here, so the UI can pass a client id through uniformly.
    """
    if client_id is None or client_id == "":
        return None
    if client_id in (stats.UNASSIGNED, -1, "-1"):
        return stats.UNASSIGNED
    try:
        return int(client_id)
    except (TypeError, ValueError):
        raise HTTPException(400, f"invalid client_id: {client_id!r}") from None


def _frame(conn: Connection, since: str | None, until: str | None,
           client: str | None = None, year: int | None = None,
           month: int | None = None, doc_type: str | None = None) -> pd.DataFrame:
    frame = stats.load_frame(conn, client_id=_scope(client), year=year, month=month,
                             doc_type=doc_type)
    return stats.date_range(frame, since, until)


@app.get("/api/summary", response_model=SummaryOut)
def get_summary(since: str | None = None, until: str | None = None,
                client: str | None = None, year: int | None = None, month: int | None = None,
                doc_type: str | None = None,
                conn: Connection = Depends(get_conn)):
    t = stats.totals(_frame(conn, since, until, client, year, month, doc_type))
    return SummaryOut(
        invoices=t.invoices, total=t.total, tax=t.tax, net=t.net,
        first_date=t.first_date, last_date=t.last_date, flagged=t.flagged,
        currencies=list(t.currencies), mixed_currency=t.mixed_currency,
    )


@app.get("/api/by-category", response_model=list[CategoryRow])
def get_by_category(since: str | None = None, until: str | None = None,
                    client: str | None = None, year: int | None = None, month: int | None = None,
                doc_type: str | None = None,
                    conn: Connection = Depends(get_conn)):
    return stats.by_category(_frame(conn, since, until, client, year, month, doc_type)).to_dict("records")


@app.get("/api/by-month", response_model=list[MonthRow])
def get_by_month(since: str | None = None, until: str | None = None,
                 client: str | None = None, year: int | None = None, month: int | None = None,
                doc_type: str | None = None,
                 conn: Connection = Depends(get_conn)):
    return stats.by_month(_frame(conn, since, until, client, year, month, doc_type)).to_dict("records")


@app.get("/api/by-vendor", response_model=list[VendorRow])
def get_by_vendor(since: str | None = None, until: str | None = None,
                  client: str | None = None, year: int | None = None, month: int | None = None,
                doc_type: str | None = None,
                  conn: Connection = Depends(get_conn)):
    return stats.by_vendor(_frame(conn, since, until, client, year, month, doc_type)).to_dict("records")


@app.get("/api/recurring", response_model=list[RecurringRow])
def get_recurring(since: str | None = None, until: str | None = None,
                  client: str | None = None, year: int | None = None, month: int | None = None,
                doc_type: str | None = None,
                  conn: Connection = Depends(get_conn)):
    frame = _frame(conn, since, until, client, year, month, doc_type)
    summary = stats.recurring_vendors(frame)
    monthly = stats.recurring_vendor_months(frame)
    months_by_vendor: dict[str, list[dict]] = {
        vendor: group[["ay", "toplam"]].to_dict("records")
        for vendor, group in monthly.groupby("vendor")
    } if not monthly.empty else {}
    return [
        {**row, "months": months_by_vendor.get(row["vendor"], [])}
        for row in summary.to_dict("records")
    ]


@app.get("/api/largest", response_model=list[LargestRow])
def get_largest(n: int = 5, since: str | None = None, until: str | None = None,
                client: str | None = None, year: int | None = None, month: int | None = None,
                doc_type: str | None = None,
                conn: Connection = Depends(get_conn)):
    rows = stats.largest(_frame(conn, since, until, client, year, month, doc_type), n).to_dict("records")
    for row in rows:
        # stats.largest() keeps `date` as a pandas Timestamp (it's a raw column
        # slice, unlike by_category/by_month which never carry a date column) --
        # the response model wants an ISO string.
        if isinstance(row.get("date"), pd.Timestamp):
            row["date"] = row["date"].date().isoformat()
    return rows


@app.get("/api/invoices", response_model=list[InvoiceOut])
def get_invoices(limit: int = 200, client: str | None = None, year: int | None = None,
                 month: int | None = None, doc_type: str | None = None,
                 conn: Connection = Depends(get_conn)):
    scope = _scope(client)
    where, params = [], []
    if scope == stats.UNASSIGNED:
        where.append("client_id IS NULL")
    elif scope is not None:
        where.append("client_id = ?")
        params.append(int(scope))
    if year is not None:
        where.append("doc_year = ?")
        params.append(year)
    if month is not None:
        where.append("doc_month = ?")
        params.append(month)
    if doc_type is not None:
        where.append("doc_type = ?")
        params.append(doc_type)

    sql = "SELECT * FROM invoices"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY date DESC, invoice_no DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    out = []
    for row in rows:
        record = dict(row)
        record["needs_review"] = bool(record["needs_review"])
        record["review_reasons"] = json.loads(record["review_reasons"] or "[]")
        record["file_exists"] = _file_exists(record.get("source_path"))
        out.append(record)
    return out


@app.get("/api/vat-summary", response_model=VatSummaryOut)
def get_vat_summary(since: str | None = None, until: str | None = None,
                    client: str | None = None, year: int | None = None,
                    month: int | None = None, doc_type: str | None = None,
                    conn: Connection = Depends(get_conn)):
    """Gelir/Gider and the KDV position for the selected period.

    The invoice-derived figures and the tahakkuk-derived `assessed_vat` are computed
    independently and both returned. Reconciling them here would hide the one thing
    worth knowing -- that the receipts and the invoices disagree.
    """
    summary = stats.vat_summary(_frame(conn, since, until, client, year, month, doc_type))

    scope = _scope(client)
    tax_id_missing = True
    if isinstance(scope, int):
        known = clients.get(conn, scope)
        tax_id_missing = not (known and known.tax_id)
    where, params = [], []
    if scope == stats.UNASSIGNED:
        where.append("client_id IS NULL")
    elif scope is not None:
        where.append("client_id = ?")
        params.append(int(scope))
    if year is not None:
        where.append("doc_year = ?")
        params.append(year)
    if month is not None:
        where.append("doc_month = ?")
        params.append(month)
    # Only KDV receipts: a muhtasar or damga accrual is a real liability but not part
    # of the VAT position, and summing them together would produce a figure that
    # matches neither the invoices nor any single receipt.
    where.append("kind = 'kdv'")
    where.append("needs_review = 0")
    row = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(payable), 0) total FROM declarations WHERE "
        + " AND ".join(where), params).fetchone()

    return VatSummaryOut(
        income=summary.income, expense=summary.expense,
        output_vat=summary.output_vat, input_vat=summary.input_vat,
        vat_balance=summary.vat_balance, payable=summary.payable,
        carried_forward=summary.carried_forward,
        sales_count=summary.sales_count, purchase_count=summary.purchase_count,
        no_sales_recorded=summary.no_sales_recorded,
        tax_id_missing=tax_id_missing,
        assessed_vat=float(row["total"]) if row["n"] else None,
        assessed_receipts=row["n"],
    )


# ---- clients ----------------------------------------------------------------------


@app.get("/api/clients", response_model=list[ClientRow])
def get_clients(conn: Connection = Depends(get_conn)):
    """Every client plus the counts the landing page needs.

    Invoices with no client (ingested before clients existed, or via a plain --ingest)
    surface as a synthetic "(atanmamış)" row with id -1 so they stay reachable in the UI
    rather than silently disappearing from a client-centric view.
    """
    rows = clients.summaries(conn)
    unassigned = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(total_amount), 0) t, "
        "COALESCE(SUM(needs_review), 0) f, MAX(date) last "
        "FROM invoices WHERE client_id IS NULL"
    ).fetchone()
    if unassigned["n"]:
        rows.append({
            "id": -1, "name": stats.UNASSIGNED, "label": "(atanmamış)",
            "display": None, "tax_id": None, "form": None, "city": None,
            "invoices": unassigned["n"], "total": unassigned["t"],
            "flagged": unassigned["f"], "last_activity": unassigned["last"], "years": [],
        })
    return rows


@app.post("/api/clients/{client_id}", response_model=ClientRow)
def update_client(client_id: int, payload: ClientUpdate,
                  conn: Connection = Depends(get_conn)):
    if clients.get(conn, client_id) is None:
        raise HTTPException(404, f"client {client_id} not found")
    clients.set_metadata(conn, client_id,
                         **payload.model_dump(exclude_none=True))
    for row in clients.summaries(conn):
        if row["id"] == client_id:
            return row
    raise HTTPException(404, f"client {client_id} not found")


#: Month names for the tree. Index 0 is the "no month folder" bucket.
_MONTH_LABELS = ("Ay belirtilmemiş", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs",
                 "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")


@app.get("/api/clients/{client_id}/tree", response_model=list[TreeYear])
def get_tree(client_id: int, conn: Connection = Depends(get_conn)):
    """The client's archive as Year > Month > Document type.

    Categories are the *folder names off disk*, taken from source_path, not logical
    labels invented here. That is the whole point of the tree: it should show what the
    accountant would see in Explorer, so "Gider Faturaları" and "Beyannameler" appear
    exactly as they are named on their machine.
    """
    rows = conn.execute(
        "SELECT doc_year, doc_month, doc_type, source_path, 'invoice' AS kind "
        "  FROM invoices WHERE client_id = :cid AND doc_year IS NOT NULL "
        "UNION ALL "
        "SELECT doc_year, doc_month, doc_type, source_path, 'declaration' "
        "  FROM declarations WHERE client_id = :cid "
        "UNION ALL "
        "SELECT doc_year, doc_month, doc_type, source_path, 'document' "
        "  FROM documents WHERE client_id = :cid",
        {"cid": client_id},
    ).fetchall()

    # {year: {month_or_0: {(doc_type, kind): count}}}
    tree: dict[int, dict[int, dict[tuple[str, str], int]]] = {}
    for row in rows:
        year = row["doc_year"]
        month = row["doc_month"] or 0
        # The stored column, not the path: it is exactly what ?doc_type= filters on,
        # so a tree node can never point at a value the dashboard cannot match. Rows
        # ingested before the column existed fall back to the folder off the path.
        folder = row["doc_type"] or (
            Path(row["source_path"]).parent.name if row["source_path"] else "?")
        key = (folder, row["kind"])
        tree.setdefault(year, {}).setdefault(month, {})
        tree[year][month][key] = tree[year][month].get(key, 0) + 1

    out: list[TreeYear] = []
    for year in sorted(tree, reverse=True):
        months: list[TreeMonth] = []
        # 0 (no month folder) sorts last: a real month is the common case and should
        # not be pushed down the list by the leftovers bucket.
        for month in sorted(tree[year], key=lambda m: (m == 0, m)):
            categories = [
                TreeCategory(doc_type=doc_type, label=archive.pretty_folder(doc_type),
                             kind=kind, count=count)
                # Sorted on the raw name so the numeric prefixes keep the order the
                # practice chose: Gelir before Gider before Beyannameler.
                for (doc_type, kind), count in sorted(tree[year][month].items())
            ]
            months.append(TreeMonth(
                month=month or None,
                label=_MONTH_LABELS[month] if month <= 12 else str(month),
                count=sum(c.count for c in categories),
                categories=categories,
            ))
        out.append(TreeYear(year=year, count=sum(m.count for m in months), months=months))
    return out


@app.get("/api/clients/{client_id}/declarations", response_model=list[DeclarationRow])
def get_declarations(client_id: int, year: int | None = None, month: int | None = None,
                doc_type: str | None = None,
                     conn: Connection = Depends(get_conn)):
    """Ingested beyanname documents.

    These are real filed declarations read from the client's beyannameler/ folder. Their
    fields are NOT parsed yet -- needs_review stays 1 until extraction has been checked
    against real documents, so nothing downstream mistakes an unread PDF for known data.
    """
    sql = "SELECT * FROM declarations WHERE client_id = ?"
    params: list = [client_id]
    if year is not None:
        sql += " AND doc_year = ?"
        params.append(year)
    if month is not None:
        sql += " AND doc_month = ?"
        params.append(month)
    if doc_type is not None:
        sql += " AND doc_type = ?"
        params.append(doc_type)
    sql += " ORDER BY doc_year DESC, period DESC, id"

    out = []
    for row in conn.execute(sql, params):
        record = dict(row)
        out.append({
            "id": record["id"], "kind": record["kind"], "period": record["period"],
            "accrued": record["accrued"], "offset_amount": record["offset_amount"],
            "payable": record["payable"], "due_date": record["due_date"],
            "issue_date": record["issue_date"], "receipt_no": record["receipt_no"],
            "taxpayer_tax_id": record["taxpayer_tax_id"],
            "lines": json.loads(record["lines"] or "[]"),
            "doc_year": record["doc_year"],
            "doc_month": record["doc_month"],
            "filename": Path(record["source_path"]).name,
            "source_path": record["source_path"],
            "file_exists": _file_exists(record["source_path"]),
            "needs_review": bool(record["needs_review"]),
            "review_reasons": json.loads(record["review_reasons"] or "[]"),
        })
    return out


@app.get("/api/clients/{client_id}/documents", response_model=list[DocumentRow])
def get_documents(client_id: int, year: int | None = None, month: int | None = None,
                doc_type: str | None = None,
                  conn: Connection = Depends(get_conn)):
    sql = ("SELECT id, doc_type, doc_year, doc_month, filename, source_path "
           "FROM documents WHERE client_id = ?")
    params: list = [client_id]
    if year is not None:
        sql += " AND doc_year = ?"
        params.append(year)
    if month is not None:
        sql += " AND doc_month = ?"
        params.append(month)
    if doc_type is not None:
        sql += " AND doc_type = ?"
        params.append(doc_type)
    rows = conn.execute(sql + " ORDER BY doc_year DESC, doc_type, filename", params)
    return [{**dict(r), "file_exists": _file_exists(r["source_path"])} for r in rows]


# ---- original files ----------------------------------------------------------------
#
# The Dosyalar (files) view in the client workspace lets an operator reach the actual PDF
# behind a ledger row or declaration -- the archive folder is the source of truth, so the
# UI should be able to show it, not just the fields extracted from it.
#
# Two ways to reach a file, for two different situations:
#
#   /open  -- hands the path to the OS so the PDF opens in the user's own viewer, or
#             shows it selected in the file manager. This is the one the UI uses. The
#             app is local and the documents are already sitting on this machine, so
#             re-downloading a copy into the browser's Downloads folder would just
#             litter the disk with duplicates of files the user already has.
#   /file  -- streams the bytes over HTTP. Kept as a fallback for viewing inside the
#             browser, and it is what makes the file reachable at all if the desktop
#             has no PDF handler registered.
#
# Neither route ever accepts a filesystem path from the caller: the path is looked up
# from the row id, so the only files reachable are ones the ingest walker already put in
# the database.


class OpenResult(BaseModel):
    opened: str
    revealed: bool = False


def _file_exists(source_path: str | None) -> bool:
    """Whether a stored path still points at a real file.

    One stat() per row. That is fine for a local archive of a few hundred documents and
    is the only honest way to show the indicator: the database records where a file was
    at ingest time, and the accountant may have moved or deleted it since.
    """
    if not source_path:
        return False
    try:
        return Path(source_path).is_file()
    except OSError:
        # A path that is too long, or on a drive that is no longer mounted.
        return False


def _resolve_pdf(source_path: str | None) -> Path:
    if not source_path:
        raise HTTPException(404, "dosya bulunamadı")
    path = Path(source_path)
    if not path.is_file():
        # The row can outlive the file: the archive gets moved, or a PDF is deleted
        # after ingest. Say so plainly rather than failing deeper in the OS call.
        raise HTTPException(404, f"dosya diskte bulunamadı: {path}")
    if path.suffix.lower() != ".pdf":
        # _launch hands this path to the shell's default handler, so a non-PDF would be
        # *executed* by whatever program claims its extension. The walker only ever
        # ingests PDFs, which makes this unreachable today -- it stays because the cost
        # of being wrong here is arbitrary code execution, and the check is one line.
        raise HTTPException(400, "yalnızca PDF dosyaları açılabilir")
    return path


def _launch(path: Path, *, reveal: bool) -> None:
    """Open a PDF in the desktop's own viewer, or select it in the file manager."""
    if reveal:
        if sys.platform == "win32":
            # explorer parses its own command line and wants /select,<path> as a single
            # token, so the usual list form does not work here. It also exits 1 even on
            # success, hence check=False.
            subprocess.run(f'explorer /select,"{path}"', check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path.parent)], check=False)
        return

    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606 - path is DB-sourced and .pdf-checked
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def _open_row(conn: Connection, table: str, row_id: int, reveal: bool) -> OpenResult:
    row = conn.execute(
        f"SELECT source_path FROM {table} WHERE id = ?", (row_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"{table[:-1]} {row_id} not found")
    path = _resolve_pdf(row["source_path"])
    try:
        _launch(path, reveal=reveal)
    except OSError as exc:
        # No registered handler for .pdf, or the shell refused. The browser fallback
        # (/file) still works, so this must not read as "the file is gone".
        raise HTTPException(500, f"dosya açılamadı: {exc}") from exc
    return OpenResult(opened=str(path), revealed=reveal)


@app.post("/api/invoices/{invoice_id}/open", response_model=OpenResult)
def open_invoice(invoice_id: int, reveal: bool = False,
                 conn: Connection = Depends(get_conn)):
    return _open_row(conn, "invoices", invoice_id, reveal)


@app.post("/api/declarations/{declaration_id}/open", response_model=OpenResult)
def open_declaration(declaration_id: int, reveal: bool = False,
                     conn: Connection = Depends(get_conn)):
    return _open_row(conn, "declarations", declaration_id, reveal)


@app.post("/api/documents/{document_id}/open", response_model=OpenResult)
def open_document(document_id: int, reveal: bool = False,
                  conn: Connection = Depends(get_conn)):
    return _open_row(conn, "documents", document_id, reveal)


def _stream_pdf(source_path: str | None) -> FileResponse:
    path = _resolve_pdf(source_path)
    # inline, not attachment: the point is to view it, not to save a second copy.
    return FileResponse(path, media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{path.name}"'})


@app.get("/api/invoices/{invoice_id}/file")
def get_invoice_file(invoice_id: int, conn: Connection = Depends(get_conn)):
    row = conn.execute(
        "SELECT source_path FROM invoices WHERE id = ?", (invoice_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"invoice {invoice_id} not found")
    return _stream_pdf(row["source_path"])


@app.get("/api/declarations/{declaration_id}/file")
def get_declaration_file(declaration_id: int, conn: Connection = Depends(get_conn)):
    row = conn.execute(
        "SELECT source_path FROM declarations WHERE id = ?", (declaration_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"declaration {declaration_id} not found")
    return _stream_pdf(row["source_path"])


@app.get("/api/documents/{document_id}/file")
def get_document_file(document_id: int, conn: Connection = Depends(get_conn)):
    row = conn.execute(
        "SELECT source_path FROM documents WHERE id = ?", (document_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"document {document_id} not found")
    return _stream_pdf(row["source_path"])


# ---- ask ------------------------------------------------------------------------


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest, conn: Connection = Depends(get_conn)):
    question = payload.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    computed = router.route(conn, question)
    if computed is not None:
        return AskResponse(
            text=computed.text, intent=computed.intent.value,
            rows=_json_safe(computed.rows), sources=[], source="router",
        )

    try:
        rag.embed_pending(conn)
        text, hits = rag.answer(conn, question)
    except foundry.FoundryError as exc:
        raise HTTPException(503, str(exc)) from exc

    sources = [
        AskSource(invoice_no=h["invoice_no"], date=h["date"], vendor=h["vendor"],
                  total_amount=h["total_amount"], score=h["score"])
        for h in hits
    ]
    return AskResponse(text=text, intent="semantic", rows=[], sources=sources, source="rag")


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, conn: Connection = Depends(get_conn)):
    """Conversational answering. Numbers still come from SQL -- see agent.py."""
    messages = [{"role": m.role, "content": m.content} for m in payload.messages]
    if not any(m["role"] == "user" for m in messages):
        raise HTTPException(400, "no user message to answer")
    if not messages[-1]["content"].strip():
        raise HTTPException(400, "question must not be empty")

    try:
        reply = agent.converse(conn, messages, use_llm=payload.use_llm,
                               client_id=_scope_id(payload.client_id))
    except foundry.FoundryError as exc:
        # agent.answer() already degrades to the computed text when the model is
        # unreachable on a router-answerable question; reaching here means the
        # question needed retrieval, which cannot work without Foundry at all.
        raise HTTPException(503, str(exc)) from exc

    return ChatResponse(
        text=reply.text, source=reply.source, intent=reply.intent, facts=reply.facts,
        sources=[AskSource(**s) for s in reply.sources],
    )


# ---- ingest -----------------------------------------------------------------------


@app.post("/api/ingest", response_model=IngestReportOut)
def ingest(files: list[UploadFile] = File(...), conn: Connection = Depends(get_conn)):
    # Deliberately a sync `def`, not `async def`. FastAPI resolves a sync-generator
    # dependency like get_conn() in the threadpool; an async route body runs on the
    # event loop thread instead, and the two threads don't match -- sqlite3 then
    # raises "objects created in a thread can only be used in that same thread" on
    # first use. Every route in this module stays sync so dependency resolution and
    # the handler body always land on the same thread. UploadFile.file is the
    # underlying sync file object, so this needs no `await`.
    if not files:
        raise HTTPException(400, "no files provided")
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, f"not a PDF: {f.filename}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for f in files:
            # Path(...).name strips any directory components a hostile filename
            # might carry, so the upload can never write outside the temp dir.
            dest = tmp_path / Path(f.filename).name
            dest.write_bytes(f.file.read())
        report = pipeline.ingest_folder(conn, tmp_path, use_llm=False)

    try:
        # Newly-ingested invoices are not retrievable by --ask/semantic search until
        # embedded. This mirrors main.py's cmd_ingest -- failure here must not fail
        # the ingest itself, Foundry Local may simply not be running.
        rag.embed_pending(conn)
    except foundry.FoundryError:
        pass

    return IngestReportOut(
        inserted=report.inserted, updated=report.updated, skipped=report.skipped,
        failed=[{"path": p, "reason": r} for p, r in report.failed],
        flagged=report.flagged,
    )


# ---- static frontend, mounted last so it cannot shadow the routes above -----------

app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
