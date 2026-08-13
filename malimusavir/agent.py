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
import re
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
#: Written in the third person, about a client's books. They used to read "En son ne
#: zaman alışveriş yaptım?" and "Kaç faturam var?" -- leftovers from when this tool
#: tracked one person's own spending. An accountant does not go shopping in their
#: clients' ledgers, and a suggestion list that talks that way tells the user the tool
#: was built for someone else.
#: An empty label means "answerable, but not worth suggesting". Three were dropped for
#: being questions no müşavir asks:
#:
#: TAX summed hesaplanan and indirilecek KDV into one figure. Output VAT is money owed
#: to the state and input VAT is money reclaimable from it; adding them produces a
#: number with no meaning in any ledger, and offering it invites someone to read it as
#: the KDV position, which is VAT_POSITION's job.
#: TOTAL adds sales to purchases the same way. SMALLEST and FIRST answer questions
#: ("the cheapest invoice", "the oldest one") that carry no consequence for a filing.
#: All four still answer if asked outright -- they are simply not advertised.
CAPABILITY_LABELS: dict[router.Intent, str] = {
    router.Intent.VAT_POSITION: "Ödenecek KDV ne kadar?",
    router.Intent.INCOME: "Dönem hasılatı ne kadar?",
    router.Intent.EXPENSE: "Dönem gider toplamı ne kadar?",
    router.Intent.COUNT: "Kaç belge kayıtlı?",
    router.Intent.DECLARATION: "Tahakkuk fişleri ne durumda?",
    router.Intent.DEADLINE: "Hangi müşterinin vadesi geçti?",
    router.Intent.GAP: "Hangi müşteride eksik belge var?",
    router.Intent.BY_CLIENT: "Müşteri bazında dağılım nedir?",
    router.Intent.DOCUMENT: "Vergi levhası hangi klasörde?",
    router.Intent.LIST: "Gider faturalarını listele",
    router.Intent.BY_CATEGORY: "Kategorilere göre dağılım nedir?",
    router.Intent.BY_VENDOR: "Hangi satıcıdan ne kadar alım yapıldı?",
    router.Intent.BY_MONTH: "Aylık dağılım nedir?",
    router.Intent.RECURRING: "Düzenli ödemeler neler?",
    router.Intent.LARGEST: "En yüksek tutarlı fatura hangisi?",
    router.Intent.LAST: "Son belge tarihi nedir?",
    router.Intent.TAX: "",
    router.Intent.TOTAL: "",
    router.Intent.SMALLEST: "",
    router.Intent.FIRST: "",
    # SEMANTIC is not asked for by name -- it is the fallback that searches line items,
    # described separately in capabilities() below.
    router.Intent.SEMANTIC: "",
}

#: The suggestions differ by scope, because the same question is not equally sensible in
#: both. "Dönem hasılatı ne kadar?" is the figure that goes on a client's beyanname; run
#: across the whole practice it adds up the turnover of unrelated businesses and means
#: nothing to anybody. Offering it on the landing page told the user this tool tracks
#: one set of books -- the reader's own -- which is the opposite of what it is for.
PRACTICE_QUESTIONS: tuple[router.Intent, ...] = (
    router.Intent.DEADLINE,
    router.Intent.GAP,
    router.Intent.DECLARATION,
    router.Intent.BY_CLIENT,
    router.Intent.DOCUMENT,
)

#: Working a single client's file: the numbers that go on a beyanname, the paperwork
#: behind them, and the checks worth making before filing.
CLIENT_QUESTIONS: tuple[router.Intent, ...] = (
    router.Intent.VAT_POSITION,
    router.Intent.INCOME,
    router.Intent.EXPENSE,
    router.Intent.DECLARATION,
    router.Intent.BY_VENDOR,
    router.Intent.LIST,
    router.Intent.BY_MONTH,
    router.Intent.BY_CATEGORY,
    router.Intent.RECURRING,
    router.Intent.LARGEST,
    router.Intent.LAST,
    router.Intent.COUNT,
    router.Intent.DOCUMENT,
)

#: Every rule here was earned by a real wrong answer, so none were dropped for being
#: verbose -- but several were doing a job something else now does better, and prompt
#: length is the whole latency bill. Measured: "Kaç belge kayıtlı?" spent 41.7s
#: producing *thirteen* tokens. Generation is free; processing 652 prompt tokens on CPU
#: is not, and this block was 394 of them, resent on every call.
#:
#: Retired: the ban on repeating "HESAPLANAN VERİ" headers and most of the markdown
#: rule, because plain_text() now strips both deterministically rather than asking the
#: model to refrain. Asking costs tokens on every request; stripping costs none.
#: Retired with them: the instructions about reproducing rows, which no longer reach
#: the model at all now that tables are returned verbatim.
_SYSTEM_BASE = """Sen bir Türk mali müşavir bürosunun asistanısın. Belgeler kullanıcının
MÜŞTERİLERİNE aittir, kendisine değil.

Kurallar:
- "HESAPLANAN VERİ" sayıları veritabanından gelir ve DOĞRUDUR. Aynen kullan; asla
  değiştirme, yuvarlama veya kendi hesabını yapma.
- Verilmeyen bir sayıyı asla uydurma. Veri yoksa "bu bilgi faturalarda yok" de.
- Kısa, düz Türkçe yaz. Para birimi "1.234,56 TL". Markdown kullanma.
- Verideki parantez içi kapsamı koru: "tüm müşteriler" kapsamındaki bir rakamı tek bir
  müşteriye atfetme. Bir müşteri adı geçiyorsa o adı kullan, başka ad uydurma.
- Üçüncü şahıs yaz ("Canan Aydın ödemiş"). Asla "ödedim" veya "faturanız" deme."""

_GROUNDED = """HESAPLANAN VERİ (doğrudur):
{facts}

Soru: {question}

Bu veriyle soruyu tek bir doğal Türkçe cümleyle yanıtla. Soruyu tekrar etme.
"Hangisi/hangi/kim/ne zaman" sorularında verideki satıcı adını ve tarihi de yaz --
tek başına tutar bu soruların cevabı değildir."""

#: One line, several figures -- the KDV position. Keeps the completeness rule that
#: stopped qwen3-4b dropping the devreden figure, and drops the layout rule that made
#: it bullet a sentence apart. Copying punctuation exactly is spelled out because the
#: model reliably turned "5 müşteri):" into "5 müşteri)):", and the ban on capitals
#: because it rendered the closing caveat as a shouted heading.
_GROUNDED_FIGURES = """HESAPLANAN VERİ (veritabanından, doğrudur):
{facts}

Kullanıcının ŞU ANKİ sorusu: {question}

Bu veriyi tek bir kısa paragraf olarak, akıcı Türkçe cümlelerle aktar. Kurallar:
- Verideki HER rakamı yaz. Hiçbirini atlama.
- Rakam içermeyen cümleleri de aktar. Özellikle uyarı cümlelerini ("uyuşmuyor",
  "kontrol edin", "okunamadı" gibi) ASLA atlama -- rakamlar kadar önemlidirler.
- Rakamları ve noktalama işaretlerini aynen kopyala; yeniden hesaplama veya yuvarlama
  yapma, parantezleri çoğaltma.
- Madde işareti, tire ile başlayan satır, liste ya da başlık KULLANMA. Düz cümle yaz.
- BÜYÜK HARFLE yazma; normal cümle düzeni kullan.
- Soruyu tekrar etme, yönergeleri tekrar etme, "CEVAP:" gibi bir etiket yazma."""

_OFF_TOPIC = """Kullanıcının sorusu faturalarıyla ilgili değil: "{question}"

Bunu sıradan bir sohbet gibi, kendi genel bilgine dayanarak kısa ve samimi şekilde
yanıtla. BUGÜNÜN TARİHİ satırındaki bilgiyi tarih/gün sorularında kullanabilirsin.
Ama: gerçek zamanlı bilgin yok (hava durumu, güncel haberler, canlı veriler) —
böyle bir şey sorulursa bunu dürüstçe söyle, uydurma. Kullanıcının faturaları
hakkında hiçbir rakam veya bilgi verme; bu soru onlarla ilgili değil."""


#: Markdown the model emits despite being told not to, and the prompt scaffolding it
#: sometimes copies into its answer.
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_RULE = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)
_STAR_BULLET = re.compile(r"^(\s*)\*\s+", re.MULTILINE)
#: Any line mentioning the prompt's own label, not just one starting with it -- the
#: model also produces trailing fragments like "Yukarıdaki HESAPLANAN VERİ".
_SCAFFOLD = re.compile(
    r"^.*HESAPLANAN VER[İI].*$", re.MULTILINE | re.IGNORECASE)
#: The parenthetical from that same header, which arrives *without* the label attached:
#: the model swaps in a noun of its own and emits "Aylık dağılım (veritabanından,
#: doğrudur)):". Stripping only lines containing "HESAPLANAN VERİ" never caught it.
_SCAFFOLD_ASIDE = re.compile(r"\s*\(\s*veritabanından,?\s*doğrudur\s*\)", re.IGNORECASE)
#: Labels the model invents to structure a reply it was asked to give as plain prose.
_ANSWER_LABEL = re.compile(r"^\s*(CEVAP|YANIT|SORU)\s*:\s*", re.MULTILINE | re.IGNORECASE)
#: A bracket left doubled by the substitutions above, or by the model copying one.
_DOUBLED_CLOSE = re.compile(r"\)\s*\)+")
_BLANK_RUN = re.compile(r"\n{3,}")


def plain_text(text: str, question: str | None = None) -> str:
    """Strip markdown and echoed prompt headers from a model reply.

    The panel renders replies as plain text, so "**19.000,00 TL**" reaches the user
    with the asterisks visible. qwen3-4b emits them anyway -- told twice, in the system
    prompt and again in the task prompt, it still produced bold headings and copied the
    "HESAPLANAN VERİ" label straight into its answer.

    Formatting is worth asking for and not worth depending on, so it is also removed
    here. Presentation only: this touches markers, never digits, so a figure cannot be
    changed by it.
    """
    if not text:
        return text
    # Markers come off before the scaffold check: the header arrives as
    # "**HESAPLANAN VERİ**", which does not start with the word itself.
    cleaned = _BOLD.sub(r"\1", text)
    cleaned = _HEADING.sub("", cleaned)
    cleaned = _SCAFFOLD.sub("", cleaned)
    # Collapse before stripping the aside, not after: "(veritabanından, doğrudur))"
    # loses one bracket to the aside, and a lone ")" left mid-sentence reads as damage
    # rather than the leak it came from.
    cleaned = _DOUBLED_CLOSE.sub(")", cleaned)
    cleaned = _SCAFFOLD_ASIDE.sub("", cleaned)
    cleaned = _ANSWER_LABEL.sub("", cleaned)
    cleaned = _RULE.sub("", cleaned)
    cleaned = _STAR_BULLET.sub(r"\1- ", cleaned)
    cleaned = cleaned.replace("**", "")
    cleaned = _BLANK_RUN.sub("\n\n", cleaned).strip()

    # The model often opens by restating the question ("Peki kaç fatura vardı?\n3
    # fatura."). Only an exact repeat is dropped, so a first line that merely starts
    # similarly -- or contains any of the answer -- is left alone.
    if question:
        head, _, rest = cleaned.partition("\n")
        if rest.strip() and _same_text(head, question):
            cleaned = rest.strip()
        # The echo is not always on a line of its own: "Toplam KDV ne kadar? Toplam
        # KDV/vergi (...): 101.692,00 TL" repeats the question inline, and splitting on
        # newlines never saw it. Only an exact prefix is removed, so an answer that
        # merely opens with similar words is left alone.
        elif fold_tr(cleaned).startswith(fold_tr(question.strip())):
            trimmed = cleaned[len(question.strip()):].lstrip(" \t?!.:-—")
            if trimmed:
                cleaned = trimmed
    return cleaned


def _same_text(left: str, right: str) -> bool:
    strip = str.maketrans("", "", " \t?!.,:;\"'()")
    return fold_tr(left).translate(strip) == fold_tr(right).translate(strip)


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

    # Says whose books are in scope. The greeting used to read "N faturanız yüklü" --
    # "your invoices" -- to a user whose invoices these are not.
    scoped = client_id is not None and client_id != stats.UNASSIGNED
    whose = "Bu müşteride" if scoped else "Büro genelinde"

    if summary.invoices:
        cats = sorted(c for c in frame["category"].dropna().unique())
        # A client whose invoices are all sales has no seller on any row, and the
        # greeting read "3 fatura kayıtlı, 0 farklı satıcıdan" -- a clause that says
        # nothing and looks like a bug. Named only when there is a number worth naming.
        vendors = frame["vendor"].dropna().nunique()
        origin = f"{vendors} farklı satıcıdan, " if vendors else ""
        data = (
            f"Merhaba. {whose} {summary.invoices} fatura kayıtlı "
            f"({summary.first_date} - {summary.last_date} arası), "
            f"{origin}"
            f"toplam {format_tr_amount(summary.total)} TL. "
            f"Kategoriler: {', '.join(cats)}."
        )
    else:
        data = (f"Merhaba. {whose} henüz kayıtlı fatura yok."
                if scoped else
                "Merhaba. Henüz kayıtlı fatura yok.")

    wanted = CLIENT_QUESTIONS if scoped else PRACTICE_QUESTIONS
    examples = "\n".join(f"  • {CAPABILITY_LABELS[intent]}" for intent in wanted)
    return (
        f"{data}\n\n"
        f"Şunları sorabilirsiniz:\n{examples}\n"
        "  • Ürün ya da kalem adı yazarak hangi faturada geçtiğini bulabilirsiniz\n\n"
        "Sayılar veritabanından hesaplanır — tahmin etmem. Belge eklemek için arşivi "
        "\"--ingest-archive\" ile yeniden tarayın."
    )


#: Intents whose computed answer is a table rather than a single figure. Not a bypass:
#: these still go through the model, but with an instruction that every row has to
#: survive, because measurement showed it dropping them.
#:
#: VAT_POSITION used to be in here and should never have been: its answer is one prose
#: sentence carrying several figures, not rows. Given the table prompt -- which ends
#: "put each row on its own line starting with -" -- the model had no rows to lay out
#: and improvised, every single run: it split the sentence into bullets, duplicated a
#: bracket into "5 müşteri))", SHOUTED the closing caveat, and sometimes echoed the
#: prompt scaffold ("Kullanıcının ŞU ANKİ sorusu: ... CEVAP:") into the reply. The
#: figures stayed correct throughout, which is why no numeric test caught it.
TABULAR_INTENTS = frozenset({
    router.Intent.BY_CATEGORY, router.Intent.BY_VENDOR, router.Intent.BY_MONTH,
    router.Intent.RECURRING, router.Intent.LIST, router.Intent.DECLARATION,
    router.Intent.DOCUMENT, router.Intent.DEADLINE, router.Intent.GAP,
    router.Intent.BY_CLIENT,
})

#: A Turkish amount: "1.234,56" or "60.508,00".
_AMOUNT = re.compile(r"\d[\d.]*,\d{2}")


def _is_tabular(computed: router.Answer) -> bool:
    """Does the computed answer actually have rows to lay out?

    Decided by the shape of the text, not by intent. An intent list is a claim about
    what an answer looks like, and it went stale the moment VAT_POSITION's wording
    changed from a list to a sentence -- nothing connected the two.
    """
    return computed.intent in TABULAR_INTENTS or "\n" in computed.text


def _is_multi_figure(computed: router.Answer) -> bool:
    """One line, several numbers -- the KDV position, and nothing else so far.

    Needs the table prompt's completeness rule, because with the plain template
    qwen3-4b reported the payable figure and dropped the carried-forward one. Must not
    get its layout rule, which is what produced the bulleted mess.
    """
    return not _is_tabular(computed) and len(_AMOUNT.findall(computed.text)) >= 2


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
        # A table of rows is data, not wording, and the model must not retype it.
        # Asked to reproduce nine overdue filings, qwen3-4b returned a KDV 2026-02
        # filing for Kaya Yapı -- a client who has 2026-01 and 2026-04 and nothing in
        # between -- while dropping Zirve's real 12.450,75 MUHTASAR and Canan's 360,00.
        # It invented a tax liability and hid two others in the same answer.
        #
        # Everything already tried to prevent that was a request the model could
        # decline: reproduce every row, keep the total, do not summarise. Rows now
        # bypass the model entirely, which makes the failure impossible rather than
        # unlikely. It also makes the slowest answers instant, since these are the ones
        # that carried a 1200-token budget.
        #
        # Prose answers still go through it. One figure in a sentence is checkable at a
        # glance and is where phrasing actually helps.
        if computed.verbatim or not use_llm or _is_tabular(computed):
            return AgentReply(computed.text, "router", computed.intent.value,
                              facts=computed.text)
        template = _GROUNDED_FIGURES if _is_multi_figure(computed) else _GROUNDED
        try:
            phrased = foundry.chat_turns(
                turns + [{
                    "role": "user",
                    "content": template.format(facts=computed.text, question=question),
                }],
                system=_system_prompt(),
                max_tokens=700,
            )
        except foundry.FoundryError:
            # The numbers are already correct without the model; degrade to them
            # rather than failing the request outright.
            return AgentReply(computed.text, "router", computed.intent.value,
                              facts=computed.text)
        return AgentReply(plain_text(phrased, question) or computed.text, "router+llm",
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
            # Told the user to drop PDFs into a panel that does not exist -- ingest is
            # a scan of the archive folder. And "faturalarınızı" addresses the wrong
            # person: the documents belong to the clients, not to the müşavir.
            "Kayıtlı fatura yok. Arşiv klasörünü \"--ingest-archive\" ile taratınca "
            "belgeler buraya gelir.", "agent", "empty",
        )

    spread = (hits[0]["score"] - hits[-1]["score"]) if len(hits) > 1 else 1.0
    if spread < RELEVANCE_SPREAD_MIN:
        if not use_llm:
            return AgentReply(
                "Bu soruyu kayıtlı belgelerden yanıtlayamam.", "agent", "off_topic",
            )
        try:
            text = foundry.chat_turns(
                turns + [{"role": "user", "content": _OFF_TOPIC.format(question=question)}],
                system=_system_prompt(), max_tokens=250,
            )
        except foundry.FoundryError:
            raise
        return AgentReply(plain_text(text, question) or "Şu an yanıt veremiyorum.",
                          "agent+llm", "off_topic")

    text, hits = rag.answer(conn, question, history=turns, system=_system_prompt(),
                            hits=hits, client_id=client_id)
    sources = [
        {
            "invoice_no": h["invoice_no"], "date": h["date"], "vendor": h["vendor"],
            "total_amount": h["total_amount"], "score": h["score"],
        }
        for h in hits
    ]
    return AgentReply(plain_text(text, question), "rag", "semantic", sources=sources)


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
