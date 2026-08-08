# Mali Müşavir

Local Turkish e-Arşiv invoice assistant. Ingests invoice PDFs, extracts structured
financial data, answers questions about them, and reports spend aggregates.

**Nothing leaves the machine.** All inference runs through [Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/);
there are no network calls to any model provider.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
foundry model download qwen3-4b
foundry model download qwen3-embedding-0.6b
```

Verify the environment:

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py
```

## Usage

```powershell
.\.venv\Scripts\python.exe main.py --ingest "C:\path\to\faturalar"
.\.venv\Scripts\python.exe main.py --stats
.\.venv\Scripts\python.exe main.py --ask "en son ne zaman alışveriş yaptım"
.\.venv\Scripts\python.exe main.py --review
```

`--ask` handles both kinds of question. Ones that are really database queries are
answered from SQL, instantly and exactly:

| Question | Answered by |
|---|---|
| "toplam ne kadar harcadım" | SQL |
| "Turkcell'e ne kadar ödedim" | SQL, scoped to that seller |
| "en son ne zaman alışveriş yaptım" | SQL (`MAX(date)`) |
| "2026 yılında ne kadar harcadım" | SQL, date-filtered |
| "kategorilere göre harcamam" | SQL |
| "düzenli ödemelerim neler" | SQL |
| "vidalama seti hangi faturada" | semantic search |

Useful flags: `--dry-run` (extract and print, write nothing), `--since` / `--until`
(date-bounded stats), `--explain` (show how a question was classified), `--semantic`
(bypass the router), `--llm-category` (see *Known limitations*), `--db PATH`.

Ingest is idempotent — re-running over the same folder is a no-op, so it is safe to
point at a growing directory.

## Web dashboard

```powershell
.\.venv\Scripts\python.exe main.py --serve
```

Opens `http://127.0.0.1:8000` — three columns: recurring payments, category totals and
PDF upload on the left; the monthly strip and full invoice ledger in the middle; the
**AI assistant on the right**. One local process serves both the page and its JSON API
(`malimusavir/api.py`) — same origin, no CORS, nothing leaves the machine. `--port`
picks a different port; `--db` picks a different database file, same as every other
command.

### How the assistant answers

`agent.py` splits the work so the model never touches arithmetic:

1. `router.py` answers the question from SQL — exact, instant, already tested.
2. Those computed facts are handed to the model as ground truth, and it is asked only
   to phrase them in natural Turkish and carry the conversation.

**The model controls the wording; the database controls the numbers.** The computed
line is shown beneath each answer, so you can always check the phrasing against what
SQL actually returned. This split exists because it was measured: asked to answer
"en son ne zaman alışveriş yaptım" from embeddings alone, `qwen3-4b` replied with a
date that appears nowhere in the corpus.

Follow-up questions inherit unstated filters. Ask "Superonline'a ne kadar ödedim" then
"peki kaç fatura vardı", and the second answer stays scoped to Superonline — the bare
router, being stateless, would have returned the global count. Anything a follow-up
names explicitly wins over what it inherited.

Questions about the assistant itself — "neler yapabilirsin", "merhaba", "yardım" — are
detected by rule and answered directly from `agent.capabilities()`, which is generated
from the router's own intent table plus a live summary of your data. No model call.

That branch is deterministic on purpose, and the reason is worth recording. It was
first built to let the model write the answer, and four attempts each failed
differently: it invented a "1.234,56 TL" largest invoice that does not exist, echoed
the prompt's own vocabulary back, and rendered the instructions as markdown headings —
at ~80s each. "What can you do" has one fixed correct answer, so there is nothing for a
model to add and a fabricated figure to lose. `test_agent.py` asserts every router
intent appears in that list *and* that every example question it offers actually routes
where it claims, so the help text cannot drift from what the tool really does.

Two modes, toggled in the chat header:

| Mode | Behaviour | Speed |
|---|---|---|
| **AI** | SQL computes, model phrases, conversation carries | ~25–45s on CPU |
| **Hızlı** | Raw computed answer, no model call | instant |

If Foundry Local isn't running, AI mode degrades to the computed answer rather than
failing — you lose the phrasing, not the number. Item lookups ("vidalama seti hangi
faturada") genuinely need the model for retrieval, and those do surface an error.

The page (`web/index.html`) is a static export from a Claude Design session, wired by
hand to real data — the layout came from Claude Design, the data underneath it is 100%
this project's existing `stats.py`/`router.py`/`rag.py`, unchanged. `web/support.js`
and `web/_ds/` are its runtime and design tokens; treat them as generated assets, not
something to hand-edit.

## Architecture

```
PDF ──► pdf_text.py ──► extractors/ ──► category.py ──► db.py ──► SQLite
        (text +          (label-anchored   (keyword         (dedupe on
         REDACTION)       regex profiles)   rules)           invoice_no+VKN)
                                                                  │
                                                     ┌────────────┴───────┐
                             --ask ──► router.py ────┤                    │
                                    (intent + slots) │                    │
                                            │        │                    │
                              aggregate ────┘        │              --stats
                                   │                 │                    │
                                   ▼                 ▼                    ▼
                              stats.py           rag.py               stats.py
                          (SQL + pandas,   (embed → cosine →      (SQL + pandas,
                             no LLM)        grounded answer)          no LLM)
```

| Module | Responsibility |
|---|---|
| `pdf_text.py` | pdfplumber extraction; **redaction happens here** |
| `normalize.py` | Turkish number/date/rate parsing |
| `extractors/` | Per-issuer profiles over a generic GİB fallback |
| `items.py` | Locates the line-item table |
| `category.py` | Keyword classifier, optional LLM fallback |
| `db.py` | SQLite schema and idempotent insert |
| `router.py` | Classifies a question; routes arithmetic to SQL |
| `rag.py` | Summary → embedding → cosine retrieval → answer |
| `stats.py` | Aggregates, computed with SQL/pandas |
| `agent.py` | Conversation: SQL supplies the facts, the model phrases them |
| `api.py` | JSON API for `--serve`; thin — calls the modules above, adds nothing |

## Design decisions

**Extraction is rule-based, not LLM-based.** e-Arşiv invoices are generated from a
standardized GİB template, so field labels are stable and label-anchored regex reads
them deterministically. A small local model asked to copy digits will eventually copy
them wrong, and silently. The LLM's only extraction job is `category`, which is the one
field genuinely not printed on the document.

**Redaction happens in the PDF parser, not in the prompt.** TC Kimlik No, addresses,
IBANs, card numbers, phone numbers, the recipient's name and the *buyer's* tax number
are removed before the text reaches any model, the database or the embedding store.
Asking a model to ignore a national ID is a request; deleting the bytes is a guarantee.
TCKN removal is checksum-validated so it strips real IDs without eating order
references. Add literal terms (your own name, your VKN) to `redact.txt` or
`MALIMUSAVIR_REDACT` — no heuristic reliably recognises an arbitrary personal name.

**`vendor_tax_id` is the seller's.** Buyer and courier tax numbers appear on the same
page and are explicitly excluded; storing the buyer's would be both wrong and a
privacy leak.

**Categories are a closed set.** Free-form category generation produces a long tail of
near-duplicates ("elektronik", "Elektronik ürün", "teknoloji") that makes spend-by-
category meaningless.

**Aggregates never go through the LLM.** "How much did I spend" has one correct answer,
and arithmetic is not a language task. `router.py` recognises those questions and
answers them from SQL, so every figure shown is computed rather than generated. The
router is rule-based for the same reason extraction is: the vocabulary is enumerable,
matching is exact, and a misrouted question yields a confidently wrong answer. Anything
it does not recognise falls through to search, so it can only add precision.

**Ambiguity is reported, not resolved silently.** "Turkcell'e ne kadar ödedim" names two
legal entities; the answer covers both and says so, rather than picking one.

**One row per invoice.** Line items are parsed for retrieval but not stored as rows;
every aggregate here operates at invoice level.

## Known limitations

These are measured on real invoices, not assumed.

**LLM category inference is unreliable on CPU.** Across three prompt designs, `qwen3-4b`
classified 1 of 6 held-out vendors correctly (a hairdresser came back as `abonelik`, a
plant nursery as `enerji`), at ~30s per call. The keyword table carries accuracy
instead; the model is opt-in via `--llm-category` and anything it produces is flagged
`category:llm_unverified`. Unmatched vendors land in `diğer` for review — honest rather
than confidently wrong. Widening `KEYWORD_RULES` in `category.py` is the better fix.

**Semantic search only handles item questions.** Retrieval works when the question names
something concrete: "vidalama seti aldığım fatura" returns the right invoice with a
clear margin. It cannot answer questions that are really database queries — those go
through the router instead. Phrasings the router does not recognise still reach search,
so an unusually worded aggregate question can still produce a wrong answer; `--explain`
shows which path a question took.

**Verify search answers against the sources printed beneath them.** `qwen3-4b` sometimes
mixes fields between retrieved invoices — naming one vendor with another's total. `--ask`
therefore always prints the retrieved invoices with their real dates and amounts, and
those figures come straight from the database. Expect 40–70s per search answer on CPU;
routed answers are instant.

**Telecom `net_amount` is derived**, as `total − tax`, because these bills carry no
net line. It is marked `:derived` in `field_sources`, and the reconcile cross-check
cannot validate those rows.

**Turkcell/Superonline PDFs are "bilgilendirme" documents**, not the legal e-Faturas —
they say so on the page. The figures are accurate; the binding documents live in the
GİB portal.

## Out of scope

Deliberate omissions, not oversights:

- **OCR.** Digital PDFs only. Scanned files are flagged `scanned:no_extractable_text`
  rather than silently producing an empty invoice. Adding Tesseract would introduce a
  second, much noisier accuracy problem on top of a solved one.
- **File reorganization.** Renaming or moving invoice files is destructive and needs a
  plan → confirm → execute flow to be safe.
- **Line-item storage.** Add it if per-product questions become a goal.

A GUI was originally out of scope for the same reason as the others — the CLI covered
the task and a UI adds surface area for no functional gain. That changed only because
a UI became something the user separately wanted to look at, not because the original
reasoning was wrong; `--serve` reuses every existing module as-is rather than growing a
second implementation of anything.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Fixtures are redacted invoice *text*, never PDFs, so nothing personal is committed.
`tests/test_real_layouts.py` holds one regression case per defect found against real
documents — each reproduces a layout quirk that broke extraction or leaked data.
