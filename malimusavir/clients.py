"""Clients: the accounting practice's taxpayers, one per archive folder.

Identity comes from the folder name, never from document contents. That is deliberate:
it means the buyer's TCKN, VKN and name can stay redacted by pdf_text.redact and never
reach the database, while the system still knows whose invoice it is looking at.

`tax_id` is optional and only becomes load-bearing when a client's own *sales* invoices
are ingested: an invoice whose seller tax id equals the client's is a sale, everything
else is a purchase. Without it every document is treated as a purchase, which is correct
for a client whose archive holds only what they bought.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from .normalize import fold_tr

#: What "this invoice is one of the client's own sales" vs "a purchase" is called in the
#: stored data. Kept as constants so the strings cannot drift between modules.
SALE = "satis"
PURCHASE = "alis"


@dataclass(frozen=True)
class Client:
    id: int
    name: str
    display: str | None = None
    tax_id: str | None = None
    form: str | None = None
    city: str | None = None

    @property
    def label(self) -> str:
        """What a human should see -- the edited display name, else the folder name."""
        return self.display or self.name


def _row_to_client(row: sqlite3.Row) -> Client:
    return Client(
        id=row["id"], name=row["name"], display=row["display"],
        tax_id=row["tax_id"], form=row["form"], city=row["city"],
    )


def resolve(conn: sqlite3.Connection, name: str) -> Client:
    """Get the client with this folder name, creating it on first sight.

    Matching is case- and accent-insensitive via fold_tr, so "Mehmet", "mehmet" and
    "MEHMET" are one client rather than three -- Windows paths are case-insensitive and
    an archive re-organised with different capitalisation must not fork the history.
    """
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("client name must not be empty")

    folded = fold_tr(cleaned)
    for row in conn.execute("SELECT * FROM clients"):
        if fold_tr(row["name"]) == folded:
            return _row_to_client(row)

    conn.execute(
        "INSERT INTO clients (name, created_at) VALUES (?, ?)",
        (cleaned, datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM clients WHERE name = ?", (cleaned,)).fetchone()
    return _row_to_client(row)


def resolve_folder(conn: sqlite3.Connection, folder_name: str) -> Client:
    """Get-or-create a client from an archive folder name.

    The standard layout names the folder "<VKN/TCKN> - <Ünvan>". When it does, the tax
    id is the identity: matching on it first means correcting a typo in the ünvan --
    or the practice simply renaming the folder -- does not fork one client into two,
    which plain name matching would do.

    The parsed tax id and title are filled in on first sight and never overwrite values
    already on file, so an operator's edit in the UI survives the next ingest.
    """
    from .archive import parse_client_folder

    parsed = parse_client_folder(folder_name)
    if not parsed.tax_id:
        return resolve(conn, folder_name)

    row = conn.execute("SELECT * FROM clients WHERE tax_id = ?", (parsed.tax_id,)).fetchone()
    if row is None:
        client = resolve(conn, folder_name)
        return set_metadata(conn, client.id, tax_id=parsed.tax_id,
                            display=parsed.title) or client

    client = _row_to_client(row)
    updates = {}
    if client.name != folder_name:
        # The folder was renamed; follow it, since the folder is what the operator sees.
        conn.execute("UPDATE clients SET name = ? WHERE id = ?", (folder_name, client.id))
        conn.commit()
    if not client.display and parsed.title:
        updates["display"] = parsed.title
    return set_metadata(conn, client.id, **updates) if updates else get(conn, client.id)


def get(conn: sqlite3.Connection, client_id: int) -> Client | None:
    row = conn.execute("SELECT * FROM clients WHERE id = ?", (client_id,)).fetchone()
    return _row_to_client(row) if row else None


def all_clients(conn: sqlite3.Connection) -> list[Client]:
    return [_row_to_client(r) for r in conn.execute("SELECT * FROM clients ORDER BY name")]


def set_metadata(conn: sqlite3.Connection, client_id: int, **fields) -> Client | None:
    """Update editable client fields. Unknown keys are ignored, not an error."""
    allowed = {"display", "tax_id", "form", "city"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if updates:
        assignments = ", ".join(f"{k} = :{k}" for k in updates)
        conn.execute(f"UPDATE clients SET {assignments} WHERE id = :id",
                     {**updates, "id": client_id})
        conn.commit()
    return get(conn, client_id)


def direction_for(client: Client | None, vendor_tax_id: str | None) -> str:
    """Whether an invoice is one of the client's sales or one of their purchases.

    An invoice the client *issued* carries their own tax id as the seller. Everything
    else was issued to them. With no client tax id on file there is nothing to compare
    against, so it is treated as a purchase -- the safe default, and the correct one for
    an archive of received invoices.
    """
    if client is None or not client.tax_id or not vendor_tax_id:
        return PURCHASE
    digits = "".join(ch for ch in vendor_tax_id if ch.isdigit())
    own = "".join(ch for ch in client.tax_id if ch.isdigit())
    return SALE if digits and digits == own else PURCHASE


def summaries(conn: sqlite3.Connection) -> list[dict]:
    """Every client with the counts the landing page needs, in one query."""
    rows = conn.execute(
        """
        SELECT c.id, c.name, c.display, c.tax_id, c.form, c.city,
               COUNT(i.id)                             AS invoices,
               COALESCE(SUM(i.total_amount), 0)        AS total,
               COALESCE(SUM(i.needs_review), 0)        AS flagged,
               MAX(i.date)                             AS last_activity
        FROM clients c
        LEFT JOIN invoices i ON i.client_id = c.id
        GROUP BY c.id
        ORDER BY c.name
        """
    ).fetchall()

    out = []
    for row in rows:
        record = dict(row)
        record["label"] = record["display"] or record["name"]
        record["years"] = [
            r["doc_year"] for r in conn.execute(
                "SELECT DISTINCT doc_year FROM invoices "
                "WHERE client_id = ? AND doc_year IS NOT NULL ORDER BY doc_year DESC",
                (row["id"],),
            )
        ]
        out.append(record)
    return out
