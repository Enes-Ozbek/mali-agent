-- One client per folder in the archive root. The folder name IS the identity -- nothing
-- is inferred from document contents, so buyer identity (TCKN, buyer VKN, recipient
-- name) stays redacted by pdf_text.redact and never reaches the database.
CREATE TABLE IF NOT EXISTS clients (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,   -- the folder name, verbatim
    display    TEXT,                   -- prettier label, editable in the UI
    tax_id     TEXT,                   -- VKN/TCKN; also separates sales from purchases
    form       TEXT,                   -- "Ltd. Şti." | "A.Ş." | "Şahıs"
    city       TEXT,
    created_at TEXT NOT NULL
);

-- One row per ingested tahakkuk fişi -- the tax office's accrual receipt, read from the
-- client's tahakkuk/ folder. These are REAL assessments, never something this tool
-- computed and assumed: `payable` is the figure the client actually owes, so it is
-- extracted label-anchored (see tahakkuk.py) rather than guessed at, and a receipt whose
-- lines do not sum to its stated TOPLAM is flagged instead of stored as fact.
CREATE TABLE IF NOT EXISTS declarations (
    id             INTEGER PRIMARY KEY,
    client_id      INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    kind           TEXT,               -- "kdv" | "damga" | "muhtasar" | "gecici" | ...
    period         TEXT,               -- taxation period, "2026-05"
    accrued        REAL,               -- tahakkuk eden
    offset_amount  REAL,               -- mahsup edilen
    payable        REAL,               -- odenecek olan; the figure actually owed
    due_date       TEXT,               -- vadesi, ISO
    issue_date     TEXT,               -- duzenleme tarihi, ISO
    receipt_no     TEXT,               -- the tahakkuk fisi serial
    taxpayer_tax_id TEXT,              -- VKN on the receipt; confirms it is this client
    lines          TEXT,               -- JSON: the per-tax assessment rows
    doc_year       INTEGER NOT NULL,   -- the folder it was filed under
    source_path    TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    raw_text       TEXT,               -- REDACTED text only
    needs_review   INTEGER NOT NULL DEFAULT 1,
    review_reasons TEXT,
    ingested_at    TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_declarations_identity
    ON declarations (client_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_declarations_client ON declarations (client_id, doc_year);

-- Everything in a client's other folders. Stored and listed, deliberately not parsed --
-- the archive can hold any document type and guessing at unknown layouts is how you get
-- confidently wrong data.
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY,
    client_id    INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    doc_type     TEXT NOT NULL,        -- the folder name, verbatim
    doc_year     INTEGER NOT NULL,
    filename     TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    ingested_at  TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_identity
    ON documents (client_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_client ON documents (client_id, doc_year);

-- One row per invoice. Line items are deliberately not stored: every aggregate this
-- tool reports operates at invoice level, and per-product questions are out of scope.

CREATE TABLE IF NOT EXISTS invoices (
    id                 INTEGER PRIMARY KEY,

    invoice_no         TEXT    NOT NULL,
    date               TEXT,               -- ISO YYYY-MM-DD
    vendor             TEXT,
    vendor_tax_id      TEXT,               -- the SELLER's VKN, never the buyer's
    total_amount       REAL,
    tax_amount         REAL,
    net_amount         REAL,
    vat_rate           REAL,
    currency           TEXT    DEFAULT 'TL',
    payment_method     TEXT,
    category           TEXT,

    -- provenance
    source_path        TEXT,
    content_hash       TEXT    NOT NULL,
    extraction_profile TEXT,
    field_sources      TEXT,               -- JSON: field -> where the value came from
    review_reasons     TEXT,               -- JSON array; empty means nothing to check
    needs_review       INTEGER NOT NULL DEFAULT 0,
    raw_text           TEXT,               -- REDACTED text only; see pdf_text.redact
    ingested_at        TEXT    NOT NULL
);

-- NOTE: the invoice identity index is created by db._migrate(), not here. It references
-- client_id, which this script cannot assume exists yet -- executescript() runs before
-- the ALTER TABLE that adds it to databases created by an earlier version.

CREATE INDEX IF NOT EXISTS idx_invoices_date     ON invoices (date);
CREATE INDEX IF NOT EXISTS idx_invoices_vendor   ON invoices (vendor_tax_id);
CREATE INDEX IF NOT EXISTS idx_invoices_category ON invoices (category);

-- Embedding vectors for semantic retrieval, keyed to the invoice they describe.
-- Stored as raw float32 bytes; at this corpus size a vector database is pure overhead.
CREATE TABLE IF NOT EXISTS embeddings (
    invoice_id   INTEGER PRIMARY KEY REFERENCES invoices(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,      -- lets us re-embed only what actually changed
    summary      TEXT NOT NULL,      -- the Turkish sentence that was embedded
    dim          INTEGER NOT NULL,
    vector       BLOB NOT NULL
);
