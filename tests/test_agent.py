"""The conversational agent.

The contract under test is the one that makes this safe: SQL computes the numbers,
the model only phrases them. So the model is stubbed throughout -- what matters is
*what it is handed* and *what happens when it misbehaves or is unavailable*, neither
of which needs a live Foundry Local.
"""

from __future__ import annotations

import re

import pytest

from malimusavir import agent, db, foundry, router
from malimusavir.extractors.base import ExtractedInvoice


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "agent.db")
    rows = [
        ("T1", "2025-06-20", "Turkcell İletişim Hizmetleri A.Ş.", "11", 230.0, "telekom"),
        ("T2", "2025-07-20", "Turkcell İletişim Hizmetleri A.Ş.", "11", 232.0, "telekom"),
        ("T3", "2025-08-20", "Turkcell İletişim Hizmetleri A.Ş.", "11", 308.0, "telekom"),
        ("S1", "2025-07-31", "Turkcell Superonline İletişim Hizmetleri A.Ş.", "22",
         500.0, "telekom"),
        ("S2", "2025-08-31", "Turkcell Superonline İletişim Hizmetleri A.Ş.", "22",
         500.0, "telekom"),
        ("D1", "2026-05-13", "D-MARKET Elektronik Hizmetler ve Ticaret A.Ş.", "33",
         652.82, "ev"),
    ]
    for no, when, vendor, tax_id, total, category in rows:
        db.upsert_invoice(connection, ExtractedInvoice(
            invoice_no=no, date=when, vendor=vendor, vendor_tax_id=tax_id,
            total_amount=total, tax_amount=round(total / 6, 2),
            net_amount=round(total * 5 / 6, 2), category=category,
            currency="TL", content_hash=f"h-{no}", profile="test",
        ))
    yield connection
    connection.close()


@pytest.fixture
def spy(monkeypatch):
    """Capture what the model was asked, and control what it replies."""
    calls = []

    def fake(messages, **kwargs):
        calls.append({"messages": messages, "kwargs": kwargs})
        return "Modelin cümlesi."

    monkeypatch.setattr(foundry, "chat_turns", fake)
    return calls


# --- the grounding contract ------------------------------------------------------


def test_computed_facts_are_handed_to_the_model(conn, spy):
    """The model must receive the SQL answer as ground truth, not the bare question."""
    reply = agent.answer(conn, "toplam ne kadar harcadim")

    assert reply.source == "router+llm"
    assert reply.text == "Modelin cümlesi."
    # The exact computed figure travelled to the model.
    prompt = spy[0]["messages"][-1]["content"]
    assert "2.422,82" in prompt
    assert "HESAPLANAN VERİ" in prompt


def test_facts_are_returned_alongside_the_phrasing(conn, spy):
    """The UI shows the raw computed line so a user can check the model's wording."""
    reply = agent.answer(conn, "toplam ne kadar harcadim")
    assert "2.422,82" in reply.facts


def test_the_model_is_told_to_phrase_not_to_compute(conn, spy):
    """It is asked to phrase, not to compute -- the difference this module exists for."""
    agent.answer(conn, "en pahali faturam hangisi")
    prompt = spy[0]["messages"][-1]["content"]
    system = spy[0]["kwargs"]["system"]

    # The answer is handed over as fact, and the ask is to word it.
    assert "652,82" in prompt
    assert "yanıtla" in prompt
    # And the standing instruction forbids inventing or recomputing figures.
    assert "kendi hesabını yapma" in system
    assert "asla uydurma" in system


def test_use_llm_false_skips_the_model_entirely(conn, spy):
    reply = agent.answer(conn, "toplam ne kadar harcadim", use_llm=False)
    assert reply.source == "router"
    assert "2.422,82" in reply.text
    assert spy == []


# --- degradation ------------------------------------------------------------------


def test_unreachable_model_falls_back_to_the_computed_answer(conn, monkeypatch):
    """Foundry being down must not lose an answer SQL already had."""
    def boom(*a, **k):
        raise foundry.FoundryError("Cannot reach Foundry Local.")

    monkeypatch.setattr(foundry, "chat_turns", boom)
    reply = agent.answer(conn, "toplam ne kadar harcadim")
    assert reply.source == "router"
    assert "2.422,82" in reply.text


def test_empty_model_output_falls_back_to_the_computed_answer(conn, monkeypatch):
    monkeypatch.setattr(foundry, "chat_turns", lambda *a, **k: "")
    reply = agent.answer(conn, "toplam ne kadar harcadim")
    assert "2.422,82" in reply.text


# --- conversational slot inheritance ---------------------------------------------


def test_follow_up_inherits_the_vendor_from_the_previous_question(conn):
    """Without this the router answers a *different* question than the user asked."""
    messages = [
        {"role": "user", "content": "Superonline'a ne kadar odedim"},
        {"role": "assistant", "content": "1.000,00 TL."},
        {"role": "user", "content": "peki kac fatura vardi"},
    ]
    reply = agent.converse(conn, messages, use_llm=False)
    assert "2 fatura" in reply.text
    assert "Superonline" in reply.text


def test_an_explicit_vendor_overrides_the_inherited_one(conn):
    messages = [
        {"role": "user", "content": "Superonline'a ne kadar odedim"},
        {"role": "assistant", "content": "1.000,00 TL."},
        {"role": "user", "content": "peki D-MARKET'e ne kadar odedim"},
    ]
    reply = agent.converse(conn, messages, use_llm=False)
    assert "652,82" in reply.text


def test_a_standalone_question_inherits_nothing(conn):
    reply = agent.answer(conn, "kac faturam var", use_llm=False)
    assert "6 fatura" in reply.text


def test_inherited_period_carries_forward(conn):
    messages = [
        {"role": "user", "content": "2025 yilinda ne kadar harcadim"},
        {"role": "assistant", "content": "1.770,00 TL."},
        {"role": "user", "content": "peki kac fatura"},
    ]
    reply = agent.converse(conn, messages, use_llm=False)
    assert "5 fatura" in reply.text        # the 2026 D-MARKET invoice is excluded


def test_history_is_replayed_to_the_model(conn, spy):
    messages = [
        {"role": "user", "content": "Superonline'a ne kadar odedim"},
        {"role": "assistant", "content": "1.000,00 TL."},
        {"role": "user", "content": "peki kac fatura vardi"},
    ]
    agent.converse(conn, messages)
    sent = spy[0]["messages"]
    assert sent[0]["content"] == "Superonline'a ne kadar odedim"
    assert sent[1]["content"] == "1.000,00 TL."


def test_converse_requires_a_user_message(conn):
    with pytest.raises(ValueError):
        agent.converse(conn, [{"role": "assistant", "content": "merhaba"}])


# --- meta-questions ----------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    ["neler yapabilirsin", "ne yapabilirsin", "merhaba", "selam", "nasilsin",
     "ne sorabilirim", "yardım", "kimsin", "günaydın"],
)
def test_meta_questions_are_detected(question):
    assert agent.is_meta_question(question)


@pytest.mark.parametrize(
    "question",
    ["toplam ne kadar harcadim", "en pahali faturam hangisi", "kac faturam var",
     "vidalama seti hangi faturada var", "Turkcell'e ne kadar odedim",
     "kategorilere gore harcamam", "duzenli odemelerim neler"],
)
def test_real_questions_are_not_mistaken_for_meta(question):
    """A false positive here would answer a data question with a help screen."""
    assert not agent.is_meta_question(question)


def test_meta_question_answers_without_the_model(conn, spy):
    """"What can you do" has one fixed answer -- no model call, so it cannot be wrong."""
    reply = agent.answer(conn, "neler yapabilirsin")
    assert reply.source == "agent"
    assert reply.intent == "help"
    assert spy == []


def test_meta_answer_describes_the_real_corpus(conn):
    text = agent.answer(conn, "neler yapabilirsin").text
    # "Büro genelinde", not "faturanız": these are the accountant's clients' books,
    # not their own. The greeting used to say "your invoices" to someone they do not
    # belong to.
    assert "Büro genelinde 6 fatura" in text        # the fixture's real count
    assert "2025-06-20 - 2026-05-13" in text        # its real date range
    assert "telekom" in text and "ev" in text       # its real categories


def test_meta_answer_invents_no_amounts(conn):
    """An earlier LLM-written version reported a "1.234,56 TL" maximum that did not
    exist. The only figure here is the corpus total, which is computed."""
    from malimusavir import stats

    text = agent.answer(conn, "neler yapabilirsin").text
    total = stats.totals(stats.load_frame(conn)).total
    amounts = re.findall(r"\d[\d.]*,\d{2}", text)
    assert amounts == [f"{total:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")]


def test_every_router_intent_is_described(conn):
    """Anti-drift: adding an intent to router.py without describing it fails here,
    so the assistant can never advertise a capability it does not have -- or omit one
    it does."""
    assert set(agent.CAPABILITY_LABELS) == set(router.Intent)


def test_described_capabilities_are_actually_routable(conn):
    """Every example question offered to the user must reach the intent it claims."""
    vendors = router.known_vendors(conn)
    categories = router.known_categories(conn)
    for intent, label in agent.CAPABILITY_LABELS.items():
        if not label:
            continue
        parsed = router.classify(label, vendors=vendors, categories=categories)
        assert parsed.intent is intent, f"{label!r} routed to {parsed.intent}, not {intent}"


# --- the panel renders plain text, so markdown has to come off ------------------------


def test_plain_text_strips_markdown_the_model_emits_anyway():
    """Told twice not to, qwen3-4b still returns bold headings; the panel shows the
    asterisks literally. Asking is worth it, depending on it is not."""
    raw = "**Ödenecek KDV:** 19.000,00 TL\n### Detay\n* satır bir\n---\n"
    cleaned = agent.plain_text(raw)
    assert "*" not in cleaned
    assert "#" not in cleaned
    assert "Ödenecek KDV: 19.000,00 TL" in cleaned
    assert "- satır bir" in cleaned


def test_plain_text_drops_the_echoed_prompt_header():
    """The model copies the "HESAPLANAN VERİ" label -- scaffolding meant for it, not
    for the user -- into its answer, wrapped in bold."""
    raw = "**HESAPLANAN VERİ (veritabanından)**\nToplam: 9.000,00 TL"
    cleaned = agent.plain_text(raw)
    assert "HESAPLANAN" not in cleaned
    assert "Toplam: 9.000,00 TL" in cleaned


def test_plain_text_never_touches_the_figures():
    """Presentation only. A cleaner that could alter a number would be worse than the
    markdown it removes."""
    raw = "**1.234,56 TL** ve *9.744,74 TL* ile 39.000,00 TL"
    cleaned = agent.plain_text(raw)
    for amount in ("1.234,56", "9.744,74", "39.000,00"):
        assert amount in cleaned


def test_plain_text_leaves_ordinary_prose_alone():
    text = "Canan Aydın'ın 3 faturası vardır, toplam 9.000,00 TL."
    assert agent.plain_text(text) == text


def test_plain_text_drops_an_echoed_question():
    """"Peki kaç fatura vardı?\n3 fatura." -- the restatement adds nothing."""
    cleaned = agent.plain_text("Peki kaç fatura vardı?\n3 fatura.",
                               question="Peki kaç fatura vardı?")
    assert cleaned == "3 fatura."


def test_plain_text_keeps_a_first_line_that_is_not_just_the_question():
    """Only an exact repeat goes; a first line carrying any answer stays."""
    reply = "Kaç faturası var? Üç tane.\nToplam 9.000,00 TL."
    assert agent.plain_text(reply, question="Kaç faturası var?") == reply


def test_plain_text_keeps_a_single_line_reply_even_if_it_echoes():
    """Dropping it would leave nothing at all."""
    assert agent.plain_text("Kaç faturası var?", question="Kaç faturası var?") \
        == "Kaç faturası var?"


def test_the_help_text_is_written_for_an_accountant_not_a_consumer(conn):
    """It used to offer "En son ne zaman alışveriş yaptım?" and "Kaç faturam var?" --
    first-person consumer questions left over from when this tracked one person's own
    spending. An accountant does not go shopping in a client's ledger, and a suggestion
    list that talks that way tells the user the tool was built for someone else.
    """
    text = agent.answer(conn, "neler yapabilirsin").text
    for consumer_phrase in ("alışveriş", "faturam", "harcadım", "ödedim",
                            "harcamam", "ödemelerim", "faturanız"):
        assert consumer_phrase not in text, f"consumer wording survives: {consumer_phrase}"


def test_the_help_text_does_not_point_at_a_ui_that_no_longer_exists(conn):
    """It told users to drop PDFs on a panel removed when the archive tree landed."""
    assert "PDF bırak" not in agent.answer(conn, "neler yapabilirsin").text


def test_every_suggested_question_routes_where_it_claims(conn):
    """The help text is generated from the router's own intent table, so a suggestion
    that no longer routes is the tool lying about itself."""
    from malimusavir import router

    for intent, label in agent.CAPABILITY_LABELS.items():
        if not label:
            continue
        parsed = router.classify(label, vendors=router.known_vendors(conn),
                                 categories=router.known_categories(conn),
                                 clients=router.known_clients(conn))
        assert parsed.intent is intent, (
            f"{label!r} is offered as {intent.value} but routes to {parsed.intent.value}")


def test_the_dashboard_chips_ask_questions_the_router_can_answer(conn):
    """The chat suggestion chips live in index.html, so no Python test saw them and
    they rotted independently of CAPABILITY_LABELS: the practice-wide view offered
    "Toplam ne kadar harcandı?", which across unrelated clients sums to a figure with
    no meaning. Any chip that falls through to semantic search is a button that
    visibly does nothing useful."""
    import re
    from pathlib import Path

    from malimusavir import router

    html = Path(__file__).resolve().parents[1] / "web" / "index.html"
    asks = re.findall(r"askNow\(['\"`]([^'\"`]+)['\"`]\)",
                      html.read_text(encoding="utf-8"))
    assert asks, "no chips found -- the extraction regex has drifted"

    vendors = router.known_vendors(conn)
    categories = router.known_categories(conn)
    clients = router.known_clients(conn)
    for ask in asks:
        # Template chips carry ${monthName}/${st.year}; substitute a real period so the
        # question is the shape a user actually sends.
        question = re.sub(r"\$\{st\.year\}", "2026", ask)
        question = re.sub(r"\$\{monthName\}", "Ocak", question)
        parsed = router.classify(question, vendors=vendors, categories=categories,
                                 clients=clients)
        assert parsed.intent is not router.Intent.SEMANTIC, (
            f"chip {ask!r} falls through to semantic search")
