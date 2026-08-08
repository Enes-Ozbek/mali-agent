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
    assert "6 faturanız" in text                    # the fixture's real count
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
