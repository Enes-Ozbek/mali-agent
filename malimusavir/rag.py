"""Semantic question answering over stored invoices.

Two design choices worth stating:

* **A synthesized Turkish summary is embedded, not the raw invoice text.** Invoice PDFs
  are ~90% boilerplate that is near-identical across documents from the same issuer, so
  embedding them puts every vector in almost the same place and retrieval stops
  discriminating. A one-sentence summary carrying date, vendor, category, amount and
  the item descriptions is what actually distinguishes one invoice from another.

* **Vectors live in SQLite as float32 blobs and are scored with numpy.** At this corpus
  size brute-force cosine over a few dozen vectors is instant, and it avoids a vector
  database dependency for no benefit.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from . import foundry
from .items import items_text

TOP_K = 5

#: Summaries per embedding request.
BATCH_SIZE = 32

# Qwen3 embedding models are trained for asymmetric retrieval: documents are embedded
# bare, queries with a task instruction. Measured on this corpus, adding it widened the
# top-to-bottom score spread and moved the right invoice to rank 1.
_QUERY_INSTRUCTION = (
    "Instruct: Bir faturayı arayan sorguya en uygun faturayı getir\nQuery: {question}"
)

# Two details here are load-bearing, both established by measurement:
#   * entries are numbered -- an unnumbered list left the model unable to refer to a
#     specific invoice, and it answered "Faturalarda bu bilgi yok" with the answer
#     sitting in front of it;
#   * the refusal instruction comes last and is qualified with "sadece ... yoksa".
#     Placing it near the top made it an easy escape hatch the model took by default.
_ANSWER_PROMPT = """Kullanıcının faturaları aşağıda listelenmiştir.

{context}

Bu faturalara bakarak soruyu Türkçe ve kısa cevapla. Cevabında ilgili faturanın
tarihini, satıcısını ve tutarını belirt. Sadece hiçbir faturada ilgili bilgi yoksa
"Faturalarda bu bilgi yok." yaz.

Soru: {question}
Cevap:"""


def build_summary(row: sqlite3.Row | dict) -> str:
    """One Turkish sentence describing an invoice, plus what was bought."""
    get = row.__getitem__ if isinstance(row, sqlite3.Row) else row.get

    date = get("date") or "tarihi bilinmeyen"
    vendor = get("vendor") or "bilinmeyen satıcı"
    category = get("category") or "diğer"
    total = get("total_amount")
    amount = f"{total:.2f} TL" if total is not None else "tutarı bilinmeyen"

    parts = [
        f"{date} tarihinde {vendor} firmasından {category} kategorisinde "
        f"{amount} tutarında fatura."
    ]
    if get("payment_method"):
        parts.append(f"Ödeme şekli: {get('payment_method')}.")
    if get("invoice_no"):
        parts.append(f"Fatura no: {get('invoice_no')}.")

    # The line items are what make "hangi faturada vidalama seti var" answerable: they
    # are the only part of the document that differs meaningfully between invoices
    # from the same issuer.
    #
    # Leading with the items instead was tried and measured worse on this corpus
    # (top-to-bottom spread 0.125 vs 0.142), so the metadata stays first.
    if get("raw_text"):
        items = items_text(get("raw_text"), limit=400)
        if items:
            parts.append(f"Alınan ürün/hizmetler: {items}")

    return " ".join(parts)


def _to_blob(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def embed_pending(conn: sqlite3.Connection, *, force: bool = False) -> int:
    """Embed invoices that have no current vector. Returns how many were embedded.

    An embedding is current when its stored content_hash still matches the invoice's,
    so re-running after a no-op ingest costs nothing.
    """
    query = """
        SELECT i.id, i.invoice_no, i.date, i.vendor, i.category, i.total_amount,
               i.payment_method, i.raw_text, i.content_hash
        FROM invoices i
        LEFT JOIN embeddings e ON e.invoice_id = i.id
        WHERE :force OR e.invoice_id IS NULL OR e.content_hash <> i.content_hash
    """
    rows = conn.execute(query, {"force": int(force)}).fetchall()
    if not rows:
        return 0

    summaries = [build_summary(row) for row in rows]
    # Batched: one request per BATCH_SIZE summaries. Sending the whole corpus in a
    # single call works at 23 invoices but grows into an ever-larger request that will
    # eventually time out or be rejected.
    vectors: list[list[float]] = []
    for start in range(0, len(summaries), BATCH_SIZE):
        vectors.extend(foundry.embed(summaries[start:start + BATCH_SIZE]))

    conn.executemany(
        "INSERT INTO embeddings (invoice_id, content_hash, summary, dim, vector) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(invoice_id) DO UPDATE SET "
        "content_hash = excluded.content_hash, summary = excluded.summary, "
        "dim = excluded.dim, vector = excluded.vector",
        [
            (row["id"], row["content_hash"], summary, len(vector), _to_blob(vector))
            for row, summary, vector in zip(rows, summaries, vectors)
        ],
    )
    conn.commit()
    return len(rows)


def search(conn: sqlite3.Connection, question: str, *, k: int = TOP_K,
           client_id: int | str | None = None) -> list[dict]:
    """Return the k invoices most similar to the question, best first.

    ``client_id`` restricts retrieval to one client. This filter is a confidentiality
    boundary, not an optimisation: in a practice, one client's assistant returning
    another client's invoices would be a real disclosure. It is applied in SQL, before
    any vector is scored, so an out-of-scope invoice can never reach the ranking.
    """
    where, params = [], []
    if client_id == "none":
        where.append("i.client_id IS NULL")
    elif client_id is not None:
        where.append("i.client_id = ?")
        params.append(int(client_id))

    rows = conn.execute(
        "SELECT e.invoice_id, e.summary, e.vector, i.* "
        "FROM embeddings e JOIN invoices i ON i.id = e.invoice_id"
        + (" WHERE " + " AND ".join(where) if where else ""),
        params,
    ).fetchall()
    if not rows:
        return []

    matrix = np.vstack([_from_blob(row["vector"]) for row in rows])
    embedded = foundry.embed([_QUERY_INSTRUCTION.format(question=question)])
    query = np.asarray(embedded[0], dtype=np.float32)

    # Cosine similarity: normalise both sides, then a single dot product.
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    query /= np.linalg.norm(query) + 1e-12
    scores = matrix @ query

    best = np.argsort(-scores)[:k]
    return [{**dict(rows[i]), "score": float(scores[i])} for i in best]


def answer(
    conn: sqlite3.Connection,
    question: str,
    *,
    k: int = TOP_K,
    history: list[dict[str, str]] | None = None,
    system: str | None = None,
    hits: list[dict] | None = None,
    client_id: int | str | None = None,
) -> tuple[str, list[dict]]:
    """Answer a Turkish question from the retrieved invoices.

    Returns the answer and the invoices it was grounded in, so the caller can show
    which documents the model actually saw. ``history`` and ``system`` are optional so
    a conversational caller (agent.py) can carry prior turns and a persona prompt
    through the same grounded-answer path the plain CLI uses without either. Pass
    ``hits`` when the caller already ran search() -- e.g. to check retrieval relevance
    before deciding to call this at all -- so the question is not re-embedded.
    """
    if hits is None:
        hits = search(conn, question, k=k, client_id=client_id)
    if not hits:
        return "Hiç fatura bulunamadı. Önce --ingest ile fatura ekleyin.", []

    context = "\n".join(f"{i}. {hit['summary']}" for i, hit in enumerate(hits, 1))
    turns = list(history or [])
    turns.append({"role": "user", "content": _ANSWER_PROMPT.format(context=context, question=question)})
    reply = foundry.chat_turns(turns, system=system, temperature=0.0, max_tokens=250)
    return reply, hits
