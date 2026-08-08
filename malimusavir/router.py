"""Question router: send arithmetic to SQL and meaning to the embeddings.

"En son ne zaman alışveriş yaptım" is `MAX(date)`. Embeddings cannot compute it, and
when asked to, the model answered with a date that was not in the corpus at all. The
router recognises questions of that shape and answers them from the database instead,
so the numbers a user sees are computed, never generated.

The classifier is rule-based for the same reason extraction is: it is a fixed,
enumerable vocabulary, matching is exact, and it costs nothing. The alternative --
asking qwen3-4b to classify intent -- was measured at 1/6 accuracy on an equivalent
task, and a misrouted question produces a confidently wrong answer.

Anything the rules do not recognise falls through to semantic search, so the router can
only ever add precision, never remove capability.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd

from . import stats
from .normalize import fold_tr, format_tr_amount

MONTHS = {
    "ocak": 1, "subat": 2, "mart": 3, "nisan": 4, "mayis": 5, "haziran": 6,
    "temmuz": 7, "agustos": 8, "eylul": 9, "ekim": 10, "kasim": 11, "aralik": 12,
}


class Intent(str, Enum):
    TOTAL = "total"
    LAST = "last"
    FIRST = "first"
    LARGEST = "largest"
    SMALLEST = "smallest"
    COUNT = "count"
    BY_CATEGORY = "by_category"
    BY_VENDOR = "by_vendor"
    BY_MONTH = "by_month"
    RECURRING = "recurring"
    TAX = "tax"
    LIST = "list"
    SEMANTIC = "semantic"      #: not an aggregate -- hand to RAG


#: Checked in order; the first match wins, so specific phrases precede general ones.
#: "en son ne zaman" must beat "ne kadar", and "en cok hangi kategori" must beat "en cok".
INTENT_PATTERNS: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (Intent.RECURRING, ("duzenli", "abonelik", "her ay odedi", "surekli odedi")),
    (Intent.BY_MONTH, ("aylik", "aya gore", "aylara gore", "ay bazinda", "her ay ne kadar",
                       "hangi ay")),
    (Intent.BY_CATEGORY, ("kategori", "neye harcadi", "nelere harcadi")),
    (Intent.BY_VENDOR, ("hangi firma", "hangi satici", "firma bazinda", "satici bazinda",
                        "firmalara gore", "kime ne kadar", "en cok nereye")),
    (Intent.LAST, ("en son ne zaman", "son ne zaman", "en son alisveris", "en son fatura",
                   "son faturam", "en yeni fatura")),
    (Intent.FIRST, ("ilk ne zaman", "en eski", "ilk faturam")),
    (Intent.LARGEST, ("en pahali", "en buyuk", "en yuksek", "en fazla tutar")),
    (Intent.SMALLEST, ("en ucuz", "en dusuk", "en kucuk")),
    (Intent.COUNT, ("kac fatura", "kac tane", "kac adet", "fatura sayisi", "kac faturam")),
    (Intent.TAX, ("kdv", "vergi")),
    (Intent.TOTAL, ("toplam ne kadar", "ne kadar harca", "ne kadar ode", "toplam harca",
                    "toplam tutar", "ne kadar para", "toplam gider", "ne kadar tuttu")),
)

#: Phrases that ask for a listing rather than a figure.
_LIST_MARKERS = ("hangi faturalar", "faturalari listele", "faturalarimi goster",
                 "hangilerinde", "listele", "goster")

_RELATIVE_PERIODS = ("bu ay", "gecen ay", "bu yil", "gecen yil", "son 3 ay", "son 6 ay",
                     "son 12 ay", "son bir yil")


@dataclass
class Question:
    """A parsed question: what is being asked, filtered to what."""

    intent: Intent
    raw: str
    vendors: list[str] = field(default_factory=list)
    category: str | None = None
    since: str | None = None
    until: str | None = None
    matched: str | None = None      #: the phrase that triggered the intent

    @property
    def is_aggregate(self) -> bool:
        return self.intent is not Intent.SEMANTIC

    def filters(self) -> list[str]:
        parts = []
        if len(self.vendors) == 1:
            parts.append(self.vendors[0])
        elif self.vendors:
            # Say so rather than silently answering for one of several matches.
            parts.append(f"{len(self.vendors)} satıcı: " + ", ".join(
                v[:28] for v in self.vendors))
        if self.category:
            parts.append(f"{self.category} kategorisi")
        if self.since or self.until:
            parts.append(f"{self.since or '...'} - {self.until or '...'}")
        return parts


@dataclass
class Answer:
    """A computed answer plus the rows it was computed from."""

    text: str
    rows: list[dict] = field(default_factory=list)
    intent: Intent = Intent.SEMANTIC


def known_vendors(conn: sqlite3.Connection) -> list[str]:
    return [
        row["vendor"]
        for row in conn.execute(
            "SELECT DISTINCT vendor FROM invoices WHERE vendor IS NOT NULL"
        ).fetchall()
    ]


def known_categories(conn: sqlite3.Connection) -> list[str]:
    return [
        row["category"]
        for row in conn.execute(
            "SELECT DISTINCT category FROM invoices WHERE category IS NOT NULL"
        ).fetchall()
    ]


#: Stems too generic to identify a seller -- they appear in most Turkish company names.
#: Stems rather than whole words because Turkish inflects: "Hizmetler" in a company
#: name is a substring of "hizmetleri" in a question, so an exact-word stoplist leaks.
_GENERIC_NAME_STEMS = (
    "iletisim", "hizmet", "ticaret", "limited", "sirket", "anonim", "elektronik",
    "sanayi", "perakende", "teknoloji", "yazilim", "magaza", "dagitim", "pazarlama",
    "grup", "holding", "sanayii",
)


def _is_generic(word: str) -> bool:
    return any(word.startswith(stem) for stem in _GENERIC_NAME_STEMS)


def _match_vendors(folded_question: str, vendors: list[str]) -> list[str]:
    """Every vendor named in the question.

    Matches distinctive words rather than the full legal name: nobody types "Turkcell
    Superonline Iletisim Hizmetleri A.S.", they type "Superonline". Turkish case
    suffixes ("Turkcell'e", "Turkcell'den") fall out of substring matching.

    Returns a *list* because the match is genuinely ambiguous: "Turkcell'e ne kadar
    odedim" names two legal entities here, Turkcell Iletisim and Turkcell Superonline.
    Picking one arbitrarily reported 4.806,80 TL when the honest answer spans both, so
    all matches are kept and the caller reports the scope it used.
    """
    matched: list[tuple[int, str]] = []
    for vendor in vendors:
        best_word = 0
        for word in fold_tr(vendor).replace(".", " ").split():
            if len(word) < 5 or _is_generic(word):
                continue
            if word in folded_question:
                best_word = max(best_word, len(word))
        if best_word:
            matched.append((best_word, vendor))
    if not matched:
        return []

    # A more specific name wins outright: "superonline" (11) beats a bare "turkcell"
    # (8), so asking about Superonline does not drag in the mobile account.
    strongest = max(score for score, _ in matched)
    return [vendor for score, vendor in matched if score == strongest]


def _match_category(folded_question: str, categories: list[str]) -> str | None:
    for category in categories:
        if fold_tr(category) in folded_question:
            return category
    return None


def _match_period(folded: str, today: date | None = None) -> tuple[str | None, str | None]:
    """Extract a date window from the question."""
    today = today or date.today()

    if "gecen yil" in folded:
        year = today.year - 1
        return f"{year}-01-01", f"{year}-12-31"
    if "bu yil" in folded:
        return f"{today.year}-01-01", f"{today.year}-12-31"
    if "gecen ay" in folded:
        month, year = (today.month - 1, today.year) if today.month > 1 else (12, today.year - 1)
        return _month_window(year, month)
    if "bu ay" in folded:
        return _month_window(today.year, today.month)

    months_back = re.search(r"son\s+(\d{1,2})\s+ay", folded)
    if months_back:
        count = int(months_back.group(1))
        start_month, start_year = today.month - count, today.year
        while start_month <= 0:
            start_month += 12
            start_year -= 1
        return f"{start_year}-{start_month:02d}-01", today.isoformat()
    if "son bir yil" in folded or "son 1 yil" in folded:
        return f"{today.year - 1}-{today.month:02d}-01", today.isoformat()

    # "Mayis 2026" or a bare "Mayis"; a bare month means the most recent one.
    for name, number in MONTHS.items():
        if re.search(rf"\b{name}\b", folded):
            year_match = re.search(r"\b(20\d{2})\b", folded)
            year = int(year_match.group(1)) if year_match else today.year
            return _month_window(year, number)

    year_match = re.search(r"\b(20\d{2})\b", folded)
    if year_match:
        year = int(year_match.group(1))
        return f"{year}-01-01", f"{year}-12-31"

    return None, None


def _month_window(year: int, month: int) -> tuple[str, str]:
    start = date(year, month, 1)
    end_year, end_month = (year + 1, 1) if month == 12 else (year, month + 1)
    end = date(end_year, end_month, 1) - pd.Timedelta(days=1)
    return start.isoformat(), end.date().isoformat() if hasattr(end, "date") else end.isoformat()


def classify(question: str, *, vendors: list[str] | None = None,
             categories: list[str] | None = None, today: date | None = None) -> Question:
    """Parse a Turkish question into an intent plus filters."""
    folded = fold_tr(question)

    matched_vendors = _match_vendors(folded, vendors or [])
    category = _match_category(folded, categories or [])
    since, until = _match_period(folded, today)

    for intent, phrases in INTENT_PATTERNS:
        for phrase in phrases:
            if phrase in folded:
                return Question(intent, question, matched_vendors, category,
                                since, until, phrase)

    # A listing request only counts as an aggregate when it names something to filter on;
    # "hangi faturada vidalama seti var" has no filter and belongs to semantic search.
    if (matched_vendors or category) and any(m in folded for m in _LIST_MARKERS):
        return Question(Intent.LIST, question, matched_vendors, category,
                        since, until, "liste")

    return Question(Intent.SEMANTIC, question, matched_vendors, category, since, until)


def _filtered(conn: sqlite3.Connection, parsed: Question,
              client_id: int | str | None = None) -> pd.DataFrame:
    frame = stats.load_frame(conn, client_id=client_id)
    frame = stats.date_range(frame, parsed.since, parsed.until)
    if frame.empty:
        return frame
    if parsed.vendors:
        frame = frame[frame["vendor"].isin(parsed.vendors)]
    if parsed.category:
        frame = frame[frame["category"] == parsed.category]
    return frame


def _scope(parsed: Question) -> str:
    filters = parsed.filters()
    return f" ({', '.join(filters)})" if filters else ""


def _rows(frame: pd.DataFrame) -> list[dict]:
    out = []
    for _, row in frame.iterrows():
        record = row.to_dict()
        if isinstance(record.get("date"), pd.Timestamp):
            record["date"] = record["date"].date().isoformat()
        out.append(record)
    return out


def answer(conn: sqlite3.Connection, parsed: Question,
           client_id: int | str | None = None) -> Answer | None:
    """Compute an answer, or return None to hand the question to semantic search.

    ``client_id`` scopes every figure to one client, so "toplam ne kadar" asked on a
    client's page means that client's total and nothing else.
    """
    if not parsed.is_aggregate:
        return None

    frame = _filtered(conn, parsed, client_id)
    scope = _scope(parsed)
    if frame.empty:
        return Answer(f"Bu kapsamda{scope} fatura bulunamadı.", [], parsed.intent)

    money = format_tr_amount
    summary = stats.totals(frame)
    intent = parsed.intent

    if intent is Intent.TOTAL:
        text = (f"Toplam{scope}: {money(summary.total)} TL "
                f"({summary.invoices} fatura, {summary.first_date} - {summary.last_date}).")
        if summary.mixed_currency:
            text += f"  UYARI: birden fazla para birimi ({', '.join(summary.currencies)})."
        return Answer(text, [], intent)

    if intent is Intent.TAX:
        return Answer(
            f"Toplam KDV/vergi{scope}: {money(summary.tax)} TL "
            f"(matrah {money(summary.net)} TL, genel toplam {money(summary.total)} TL).",
            [], intent,
        )

    if intent is Intent.COUNT:
        return Answer(f"{summary.invoices} fatura{scope} kayıtlı "
                      f"({summary.first_date} - {summary.last_date}).", [], intent)

    if intent in (Intent.LAST, Intent.FIRST):
        dated = frame.dropna(subset=["date"])
        if dated.empty:
            return Answer(f"Tarihli fatura bulunamadı{scope}.", [], intent)
        row = dated.loc[dated["date"].idxmax() if intent is Intent.LAST
                        else dated["date"].idxmin()]
        when = "En son" if intent is Intent.LAST else "İlk"
        return Answer(
            f"{when} fatura{scope}: {row['date'].date().isoformat()} tarihinde "
            f"{row['vendor']} - {money(row['total_amount'])} TL "
            f"({row['category']}, fatura no {row['invoice_no']}).",
            _rows(dated.loc[[row.name]]), intent,
        )

    if intent in (Intent.LARGEST, Intent.SMALLEST):
        valued = frame.dropna(subset=["total_amount"])
        if valued.empty:
            return Answer(f"Tutarı okunabilen fatura yok{scope}.", [], intent)
        row = valued.loc[valued["total_amount"].idxmax() if intent is Intent.LARGEST
                         else valued["total_amount"].idxmin()]
        label = "En yüksek" if intent is Intent.LARGEST else "En düşük"
        return Answer(
            f"{label} fatura{scope}: {money(row['total_amount'])} TL, "
            f"{row['date'].date().isoformat()} tarihinde {row['vendor']} "
            f"({row['category']}, fatura no {row['invoice_no']}).",
            _rows(valued.loc[[row.name]]), intent,
        )

    if intent is Intent.BY_CATEGORY:
        grouped = stats.by_category(frame)
        lines = [f"Kategoriye göre{scope} (toplam {money(summary.total)} TL):"]
        lines += [f"  {row['category']:<14} {money(row['toplam']):>12} TL  "
                  f"({int(row['adet'])} fatura)" for _, row in grouped.iterrows()]
        return Answer("\n".join(lines), grouped.to_dict("records"), intent)

    if intent is Intent.BY_VENDOR:
        grouped = stats.by_vendor(frame)
        lines = [f"Satıcıya göre{scope} (toplam {money(summary.total)} TL):"]
        lines += [f"  {str(row['vendor'])[:40]:<42} {money(row['toplam']):>12} TL  "
                  f"({int(row['adet'])} fatura)" for _, row in grouped.iterrows()]
        return Answer("\n".join(lines), grouped.to_dict("records"), intent)

    if intent is Intent.BY_MONTH:
        grouped = stats.by_month(frame)
        lines = [f"Aylık{scope}:"]
        lines += [f"  {row['ay']}   {money(row['toplam']):>12} TL  ({int(row['adet'])} fatura)"
                  for _, row in grouped.iterrows()]
        return Answer("\n".join(lines), grouped.to_dict("records"), intent)

    if intent is Intent.RECURRING:
        grouped = stats.recurring_vendors(frame)
        if grouped.empty:
            return Answer(f"Düzenli (aylık) ödeme tespit edilmedi{scope}.", [], intent)
        lines = [f"Düzenli ödemeler{scope}:"]
        lines += [f"  {str(row['vendor'])[:40]:<42} {money(row['toplam']):>12} TL  "
                  f"{int(row['adet'])} fatura, ~{row['ortalama_gun']:.0f} günde bir"
                  for _, row in grouped.iterrows()]
        lines.append(f"  Aylık düzenli gider: {money(float(grouped['aylik_ortalama'].sum()))} TL")
        return Answer("\n".join(lines), grouped.to_dict("records"), intent)

    if intent is Intent.LIST:
        listed = frame.sort_values("date", ascending=False)
        lines = [f"{len(listed)} fatura{scope}, toplam {money(summary.total)} TL:"]
        lines += [
            f"  {row['date'].date().isoformat() if pd.notna(row['date']) else '-'}  "
            f"{money(row['total_amount']):>12} TL  {str(row['vendor'])[:38]}"
            for _, row in listed.iterrows()
        ]
        return Answer("\n".join(lines), _rows(listed), intent)

    return None


def route(conn: sqlite3.Connection, question: str,
          client_id: int | str | None = None) -> Answer | None:
    """Classify and answer in one step. None means: use semantic search."""
    parsed = classify(
        question,
        vendors=known_vendors(conn),
        categories=known_categories(conn),
    )
    return answer(conn, parsed, client_id)
