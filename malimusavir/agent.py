"""Conversational agent: the LLM writes the answer, SQL supplies the facts.

The split matters, and it is the whole point of this module.

Asked to answer "en son ne zaman alışveriş yaptım" from embeddings alone, qwen3-4b
replied with a date that appears nowhere in the corpus. So the model is never asked to
compute. Instead:

  1. router.py answers the question from SQL -- exact, instant, already tested.
  2. Those computed facts are handed to the model as ground truth, and it is asked
     only to phrase them in natural Turkish and carry the conversation.

The model controls the wording; the database controls the numbers. A model that
paraphrases "9.744,74 TL" badly is a cosmetic problem. A model that *derives*
"9.744,74 TL" is a correctness problem, and this module never lets it try.

Questions the router cannot answer (item lookups -- "vidalama seti hangi faturada")
still go to rag.py, where retrieval grounds the model in real invoice summaries.
"""

from __future__ import annotations

import datetime
import sqlite3
from dataclasses import dataclass, field

from . import foundry, rag, router
from .normalize import fold_tr, format_tr_amount

_TR_WEEKDAYS = ("Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar")
_TR_MONTHS = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos",
             "Eylül", "Ekim", "Kasım", "Aralık")

#: How many prior turns to replay. Each turn costs prompt-processing time on CPU, and
#: invoice questions rarely depend on more than the last exchange or two.
HISTORY_TURNS = 6

#: Below this top-to-bottom retrieval-score gap, nothing in the corpus is actually
#: relevant to the question. Measured on this corpus: off-topic questions ("bugün
#: günlerden ne", "hava nasıl", "merhaba") spread 0.006-0.010; a genuine invoice lookup
#: ("vidalama seti hangi faturada") spreads 0.14-0.15 -- a 14x gap. Spread is used
#: rather than the absolute top score because absolute scores are not comparable across
#: questions: "merhaba" scored 0.534 on its own, higher than the 5th-place hit of a
#: genuinely good query (0.481). Set well below the measured "genuine" floor so a real
#: invoice question with no exact match (e.g. an item that was never bought) still gets
#: the safe "not found in your invoices" answer instead of being treated as off-topic.
RELEVANCE_SPREAD_MIN = 0.04

#: What each router intent means, in the user's words. The model is told about its own
#: abilities from THIS table rather than from a hand-written blurb, so it cannot
#: advertise a feature that does not exist. test_agent.py asserts every Intent appears
#: here, so adding an intent without describing it fails the suite.
#: Phrased as example *questions*, deliberately, not as noun-phrases. An earlier
#: version listed abilities as nouns ("en yüksek tutarlı fatura") and the model read
#: the list as a menu of statistics awaiting values -- then invented them, reporting a
#: fabricated "1.234,56 TL" maximum. A question cannot be filled in with a fake answer.
CAPABILITY_LABELS: dict[router.Intent, str] = {
    router.Intent.TOTAL: "Toplam ne kadar harcadım?",
    router.Intent.TAX: "Ne kadar KDV ödedim?",
    router.Intent.VAT_POSITION: "Ödenecek KDV ne kadar?",
    router.Intent.INCOME: "Toplam gelir ne kadar?",
    router.Intent.EXPENSE: "Toplam gider ne kadar?",
    router.Intent.DECLARATION: "Tahakkuk fişleri ne durumda?",
    router.Intent.DOCUMENT: "Vergi levhası hangi klasörde?",
    router.Intent.COUNT: "Kaç faturam var?",
    router.Intent.LAST: "En son ne zaman alışveriş yaptım?",
    router.Intent.FIRST: "İlk faturam ne zaman?",
    router.Intent.LARGEST: "En pahalı faturam hangisi?",
    router.Intent.SMALLEST: "En ucuz faturam hangisi?",
    router.Intent.BY_CATEGORY: "Kategorilere göre ne kadar harcadım?",
    router.Intent.BY_VENDOR: "Hangi firmaya ne kadar ödedim?",
    router.Intent.BY_MONTH: "Aylık harcamam nedir?",
    router.Intent.RECURRING: "Düzenli ödemelerim neler?",
    router.Intent.LIST: "Telekom faturalarını listele",
    # SEMANTIC is not asked for by name -- it is the fallback that searches line items,
    # described separately in capabilities() below.
    router.Intent.SEMANTIC: "",
}

_SYSTEM_BASE = """Sen bir Türk mali müşavir asistanısın. Kullanıcının kendi faturaları
üzerinde çalışıyorsun.

Kurallar:
- Sana "HESAPLANAN VERİ" olarak verilen sayılar veritabanından gelir ve DOĞRUDUR.
  Bu sayıları aynen kullan; asla değiştirme, yuvarlama veya kendi hesabını yapma.
- Sana verilmeyen bir sayıyı asla uydurma. Veri yoksa "bu bilgi faturalarda yok" de.
- Kısa ve net konuş. Türkçe yanıt ver. Para birimini "1.234,56 TL" biçiminde yaz.
- Madde işareti kullanma, düz cümle kur. Markdown başlık kullanma.
- HESAPLANAN VERİ'deki parantez içi KAPSAMI aynen koru. Veri "tüm müşteriler"
  kapsamındaysa rakamı tek bir müşteriye ATFETME; "tüm müşteriler toplamında" de.
  Veri bir müşteri adı taşıyorsa yanıtta o adı kullan, başka bir ad uydurma.
- Kullanıcı bir mali müşavirdir; faturalar onun müşterilerine aittir. Bir müşteri
  adı verildiyse üçüncü şahıs kullan ("Canan Aydın ... ödemiş"). Asla "ödedim"
  veya "harcadım" deme."""

_GROUNDED = """HESAPLANAN VERİ (veritabanından, doğrudur):
{facts}

Kullanıcının sorusu: {question}

Yukarıdaki veriyi kullanarak soruyu doğal bir Türkçe cümleyle yanıtla."""

_OFF_TOPIC = """Kullanıcının sorusu faturalarıyla ilgili değil: "{question}"

Bunu sıradan bir sohbet gibi, kendi genel bilgine dayanarak kısa ve samimi şekilde
yanıtla. BUGÜNÜN TARİHİ satırındaki bilgiyi tarih/gün sorularında kullanabilirsin.
Ama: gerçek zamanlı bilgin yok (hava durumu, güncel haberler, canlı veriler) —
böyle bir şey sorulursa bunu dürüstçe söyle, uydurma. Kullanıcının faturaları
hakkında hiçbir rakam veya bilgi verme; bu soru onlarla ilgili değil."""


def _now_tr() -> str:
    """Today, in Turkish, computed locally -- never something to ask the model.

    Training data has a fixed cutoff and "today" changes daily, so no amount of model
    quality fixes this: it is a fact that does not exist yet at training time. It has
    to be handed to the model as context, the same way computed invoice totals are.
    """
    now = datetime.datetime.now()
    return f"{now.day} {_TR_MONTHS[now.month - 1]} {now.year}, {_TR_WEEKDAYS[now.weekday()]}"


def _system_prompt() -> str:
    return f"{_SYSTEM_BASE}\n\nBUGÜNÜN TARİHİ: {_now_tr()}"



@dataclass
class AgentReply:
    """What the agent said, and what it was standing on when it said it."""

    text: str
    source: str                       #: "router+llm" | "rag" | "router"
    intent: str = "semantic"
    #: The exact figures SQL computed. Shown in the UI so the user can check the
    #: model's phrasing against the real numbers.
    facts: str | None = None
    sources: list[dict] = field(default_factory=list)


#: Questions about the assistant itself, or greetings -- not about invoice data. Cheap
#: exact matching, for the same reason router.py classifies with rules: the vocabulary
#: is small and closed, and a mistake here sends the question down the wrong path.
_META_PATTERNS = (
    "neler yapabilir", "ne yapabilir", "neler yapar", "ne yaparsin", "ne ise yararsin",
    "nasil kullan", "ne sorabilir", "hangi sorular", "ornek soru", "komutlar",
    "yardim",
    "kimsin", "nesin", "sen nesin", "ne yapiyorsun",
    "merhaba", "selam", "gunaydin", "iyi gunler", "iyi aksamlar", "nasilsin",
)


def is_meta_question(question: str) -> bool:
    """Whether a question is about the assistant rather than about the invoices."""
    folded = fold_tr(question).strip(" ?!.")
    return any(pattern in folded for pattern in _META_PATTERNS)


def capabilities(conn: sqlite3.Connection, client_id: int | str | None = None) -> str:
    """What this assistant can actually do, right now -- shown to the user verbatim.

    Every line is derived: the example questions from CAPABILITY_LABELS, the data
    summary from the database in front of it. Nothing here is a hand-maintained blurb
    that can quietly go stale, which is what makes it safe to return without review.
    """
    from . import stats  # local import: stats imports pandas, agent is used without it

    frame = stats.load_frame(conn, client_id=client_id)
    summary = stats.totals(frame)

    if summary.invoices:
        cats = sorted(c for c in frame["category"].dropna().unique())
        data = (
            f"Merhaba. {summary.invoices} faturanız yüklü "
            f"({summary.first_date} - {summary.last_date} arası), "
            f"{frame['vendor'].dropna().nunique()} farklı satıcıdan, "
            f"toplam {format_tr_amount(summary.total)} TL. "
            f"Kategoriler: {', '.join(cats)}."
        )
    else:
        data = "Merhaba. Henüz yüklenmiş fatura yok."

    examples = "\n".join(f"  • {label}" for label in CAPABILITY_LABELS.values() if label)
    return (
        f"{data}\n\n"
        f"Şunları sorabilirsiniz:\n{examples}\n"
        "  • \"Vidalama seti hangi faturada?\" gibi ürün aramaları\n\n"
        "Bir soruyu yanıtladığımda sayılar veritabanından hesaplanır — tahmin "
        "etmem. Yeni fatura eklemek için soldaki \"PDF bırak\" alanını kullanın."
    )


#: Intents whose computed answer is a table or a statement where every clause carries
#: information. The model is not asked to reword these.
#:
#: Measured with scripts/eval_agent.py against qwen3-4b. Given the full monthly
#: breakdown it replied "Aylık harcama, tüm müşterilerin aylık faturalarının
#: toplamıdır" -- a definition of the term with no data in it. Given "hesaplanan 0,00,
#: indirilecek 39.000,00, ödenecek yok; 39.000,00 devreden" it replied "Ödenecek KDV
#: yoktur", dropping the carried-forward figure, which is the part an accountant needs.
#:
#: Single-figure answers survive rewording intact and still go through the model, which
#: is where its phrasing is worth the wait. A table cannot be improved by it, only
#: shortened.
TABULAR_INTENTS = frozenset({
    router.Intent.BY_CATEGORY, router.Intent.BY_VENDOR, router.Intent.BY_MONTH,
    router.Intent.RECURRING, router.Intent.LIST, router.Intent.DECLARATION,
    router.Intent.DOCUMENT, router.Intent.VAT_POSITION,
})


def _is_tabular(computed: router.Answer) -> bool:
    return computed.intent in TABULAR_INTENTS or "\n" in computed.text


def _history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """The last few turns, excluding the current question."""
    return [m for m in messages if m.get("role") in ("user", "assistant")][-HISTORY_TURNS:]


def _route_in_context(conn: sqlite3.Connection, question: str, previous: str | None,
                      client_id: int | str | None = None,
                      ) -> tuple[router.Answer | None, router.Question]:
    """Route a question, inheriting unstated filters from the previous question.

    router.py is stateless by design -- it classifies each question on its own. That
    is right for one-shot CLI use and wrong for a conversation: asked "Superonline'a ne
    kadar ödedim" and then "peki kaç fatura vardı", the bare router answers the second
    with the *global* invoice count, having dropped the vendor. The chat would look
    conversational while quietly answering a different question than the one asked.

    So: if the follow-up names no vendor, category or period of its own, it inherits
    whatever the previous question established. Anything it *does* name wins outright,
    so "peki Turkcell?" correctly switches vendors rather than merging them.
    """
    vendors = router.known_vendors(conn)
    categories = router.known_categories(conn)
    clients = router.known_clients(conn)
    parsed = router.classify(question, vendors=vendors, categories=categories,
                             clients=clients)

    prior = None
    if previous:
        prior = router.classify(previous, vendors=vendors, categories=categories,
                                clients=clients)

    # "Canan Aydın'ın kaç faturası var" -> "listele onları". The follow-up names nothing,
    # so classify() cannot see it is about invoices and drops it into semantic search,
    # where retrieval scores it off-topic and the model answers with conversational
    # filler. Promoted here instead, because the previous turn supplies the subject the
    # question is missing. Only for a bare list command: a question asking what is
    # *inside* the invoices still belongs to line-item search.
    if (prior is not None and not parsed.is_aggregate
            and router.wants_listing(parsed.raw)
            and (prior.clients or prior.vendors or prior.category)):
        parsed.intent = router.Intent.LIST
        parsed.matched = "liste"

    if previous and parsed.is_aggregate:
        if not parsed.vendors and prior.vendors:
            parsed.vendors = prior.vendors
        if not parsed.category and prior.category:
            parsed.category = prior.category
        if not parsed.since and not parsed.until and (prior.since or prior.until):
            parsed.since, parsed.until = prior.since, prior.until
        # "Canan Aydın'ın kaç faturası var" then "listele onları" -- the follow-up names
        # nobody, so without this it would list every client's invoices under a question
        # the user clearly meant about Canan.
        if not parsed.clients and prior.clients:
            parsed.clients = prior.clients

    return router.answer(conn, parsed, client_id), parsed


def answer(
    conn: sqlite3.Connection,
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    use_llm: bool = True,
    client_id: int | str | None = None,
) -> AgentReply:
    """Answer one question, optionally in the context of a conversation.

    ``use_llm=False`` returns the router's raw computed text with no model call --
    instant, and the honest fallback when Foundry Local is unreachable.
    """
    turns = list(history or [])

    # "neler yapabilirsin" is about the assistant, not the invoices. Without this it
    # falls through to retrieval, which dutifully returns the five nearest invoices --
    # and the model then describes one of them as if that were the answer.
    # Answered directly, without the model, and deliberately so. "What can you do" has
    # one fixed correct answer -- there is no variable content for a model to add, and
    # four measured attempts at letting qwen3-4b write it each failed differently:
    # it invented a "1.234,56 TL" maximum invoice that does not exist, echoed the
    # prompt's own vocabulary back, and rendered the instructions as markdown headings,
    # at ~80s per answer. The generated text below is instant and cannot be wrong.
    if is_meta_question(question):
        return AgentReply(capabilities(conn, client_id=client_id), "agent", "help")

    prior_questions = [m["content"] for m in turns if m.get("role") == "user"]
    computed, parsed = _route_in_context(
        conn, question, prior_questions[-1] if prior_questions else None, client_id
    )

    # A question that names one client scopes retrieval to them as well, not just the
    # aggregates. Without this, "Canan Aydın'ın vidalama seti hangi faturada" asked on
    # the practice-wide page searches every client's invoices and answers about whoever
    # happens to match -- the same wrong-subject failure the aggregates just fixed.
    if len(parsed.clients) == 1 and (client_id is None or client_id == "none"):
        client_id = parsed.clients[0].id

    if computed is not None:
        # A scope refusal is an instruction to the user, not a figure to be phrased.
        # Measured: the model turned "this panel covers Zeynep, ask on Canan's page"
        # into "that information is not in the invoices", which is a different and
        # false claim. Returned unchanged, the same way capabilities() is.
        if computed.verbatim or not use_llm or _is_tabular(computed):
            return AgentReply(computed.text, "router", computed.intent.value,
                              facts=computed.text)
        try:
            phrased = foundry.chat_turns(
                turns + [{
                    "role": "user",
                    "content": _GROUNDED.format(facts=computed.text, question=question),
                }],
                system=_system_prompt(),
            )
        except foundry.FoundryError:
            # The numbers are already correct without the model; degrade to them
            # rather than failing the request outright.
            return AgentReply(computed.text, "router", computed.intent.value,
                              facts=computed.text)
        return AgentReply(phrased or computed.text, "router+llm",
                          computed.intent.value, facts=computed.text)

    # Not an aggregate. Retrieval decides which of two very different things this is:
    # a genuine invoice lookup ("vidalama seti hangi faturada"), or a question that
    # merely failed to match any router pattern despite not being about invoices at
    # all ("bugün günlerden ne"). Forcing the latter through the invoice-grounded
    # prompt is what previously made the model describe a random Turkcell bill as the
    # answer to "what day is it" -- it was told to answer *from these invoices* and
    # had no path to say "that question has nothing to do with them."
    rag.embed_pending(conn)
    hits = rag.search(conn, question, client_id=client_id)

    if not hits:
        return AgentReply(
            "Henüz yüklenmiş fatura yok. Soldaki \"PDF bırak\" alanından e-Arşiv "
            "faturalarınızı ekleyebilirsiniz.", "agent", "empty",
        )

    spread = (hits[0]["score"] - hits[-1]["score"]) if len(hits) > 1 else 1.0
    if spread < RELEVANCE_SPREAD_MIN:
        if not use_llm:
            return AgentReply(
                "Bu soruyu faturalarınızdan yanıtlayamam.", "agent", "off_topic",
            )
        try:
            text = foundry.chat_turns(
                turns + [{"role": "user", "content": _OFF_TOPIC.format(question=question)}],
                system=_system_prompt(), max_tokens=250,
            )
        except foundry.FoundryError:
            raise
        return AgentReply(text or "Şu an yanıt veremiyorum.", "agent+llm", "off_topic")

    text, hits = rag.answer(conn, question, history=turns, system=_system_prompt(),
                            hits=hits, client_id=client_id)
    sources = [
        {
            "invoice_no": h["invoice_no"], "date": h["date"], "vendor": h["vendor"],
            "total_amount": h["total_amount"], "score": h["score"],
        }
        for h in hits
    ]
    return AgentReply(text, "rag", "semantic", sources=sources)


def converse(
    conn: sqlite3.Connection,
    messages: list[dict[str, str]],
    *,
    use_llm: bool = True,
    client_id: int | str | None = None,
) -> AgentReply:
    """Answer the latest user message in a conversation."""
    users = [m for m in messages if m.get("role") == "user"]
    if not users:
        raise ValueError("no user message to answer")
    question = users[-1]["content"].strip()
    return answer(conn, question, history=_history(messages[:-1]), use_llm=use_llm,
                  client_id=client_id)
