"""The trained category classifier.

Runs fully offline: the embedding call is stubbed, so these test the head's maths and
the pipeline's ordering/threshold logic rather than the encoder's quality. Model
quality is measured separately by training/evaluate.py against held-out vendors.
"""

from __future__ import annotations

import numpy as np
import pytest

from malimusavir import category, classifier


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    """A tiny 3-class head over 4-dim vectors, with a known correct answer."""
    labels = ["telekom", "market", "hizmet"]
    # One-hot-ish rows: dimension i votes for class i.
    coef = np.array([
        [4.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
    ], dtype=np.float32)
    model = classifier._Model(coef, np.zeros(3, dtype=np.float32), labels, 0.9)
    monkeypatch.setattr(classifier, "load_model", lambda: model)
    return model


def stub_embed(monkeypatch, vector):
    monkeypatch.setattr(classifier.foundry, "embed", lambda texts: [list(vector)])


# --- the head's maths -------------------------------------------------------------


def test_predicts_the_dominant_dimension(fake_model, monkeypatch):
    stub_embed(monkeypatch, [0.0, 1.0, 0.0, 0.0])
    label, confidence = classifier.classify("Migros. sut ekmek")
    assert label == "market"
    assert 0.0 < confidence <= 1.0


def test_confidence_is_a_probability_over_the_classes(fake_model, monkeypatch):
    stub_embed(monkeypatch, [1.0, 0.0, 0.0, 0.0])
    _, confident = classifier.classify("x")
    stub_embed(monkeypatch, [1.0, 1.0, 1.0, 0.0])   # perfectly ambiguous
    _, ambiguous = classifier.classify("x")
    assert confident > ambiguous
    assert ambiguous == pytest.approx(1 / 3, abs=0.05)


def test_input_is_normalised_before_scoring(fake_model, monkeypatch):
    """An unnormalised vector would scale every logit and inflate confidence."""
    stub_embed(monkeypatch, [0.0, 1.0, 0.0, 0.0])
    _, small = classifier.classify("x")
    stub_embed(monkeypatch, [0.0, 50.0, 0.0, 0.0])  # same direction, 50x magnitude
    _, large = classifier.classify("x")
    assert small == pytest.approx(large, abs=1e-5)


# --- degradation ------------------------------------------------------------------


def test_missing_artifact_returns_none(monkeypatch):
    monkeypatch.setattr(classifier, "load_model", lambda: None)
    assert classifier.classify("herhangi bir sey") is None


def test_unreachable_embedding_model_returns_none(fake_model, monkeypatch):
    def boom(texts):
        raise RuntimeError("Foundry down")

    monkeypatch.setattr(classifier.foundry, "embed", boom)
    assert classifier.classify("x") is None


def test_empty_text_returns_none(fake_model):
    assert classifier.classify("   ") is None


# --- ordering and threshold inside category.classify() -----------------------------


def test_keyword_rules_win_over_the_classifier(monkeypatch):
    """Keywords are exact and instant; the classifier must never override them."""
    called = []
    monkeypatch.setattr(category.classifier, "classify",
                        lambda text: called.append(text) or ("giyim", 0.99))
    label, source = category.classify("Turkcell İletişim Hizmetleri A.Ş.",
                                      "Aylık paket ücreti", use_llm=False)
    assert (label, source) == ("telekom", "keyword")
    assert called == [], "classifier should not have been consulted"


def test_classifier_handles_what_keywords_miss(monkeypatch):
    monkeypatch.setattr(category.classifier, "classify", lambda text: ("hizmet", 0.88))
    label, source = category.classify("Bahar Kuafor", "Sac kesimi", use_llm=False)
    assert (label, source) == ("hizmet", "classifier")


def test_below_the_floor_is_not_committed(monkeypatch):
    """No idea at all -- fall through rather than store a guess."""
    monkeypatch.setattr(category.classifier, "classify",
                        lambda text: ("sigorta", category.CONFIDENCE_MIN - 0.01))
    label, source = category.classify("Bilinmeyen Firma", "tanimsiz kalem", use_llm=False)
    assert (label, source) == (category.DEFAULT_CATEGORY, "default")


def test_middling_confidence_is_used_but_flagged(monkeypatch):
    """Between the floor and CONFIDENCE_CLEAR: commit the answer, admit the doubt.

    This band exists because a correct "sağlık" call scored 0.358 on the held-out set
    -- discarding it lost a right answer, and taking it silently would have hidden a
    real uncertainty.
    """
    midpoint = (category.CONFIDENCE_MIN + category.CONFIDENCE_CLEAR) / 2
    monkeypatch.setattr(category.classifier, "classify",
                        lambda text: ("sağlık", midpoint))
    assert category.classify("X", "y", use_llm=False) == ("sağlık", "classifier_low")


def test_confidence_exactly_at_the_floor_is_accepted(monkeypatch):
    monkeypatch.setattr(category.classifier, "classify",
                        lambda text: ("sigorta", category.CONFIDENCE_MIN))
    label, source = category.classify("X", "y", use_llm=False)
    assert (label, source) == ("sigorta", "classifier_low")


def test_low_confidence_reaches_the_review_queue(monkeypatch):
    """The flag must survive into review_reasons, or it is invisible to the user."""
    from malimusavir.pipeline import extract_from_text

    monkeypatch.setattr(category.classifier, "classify", lambda text: ("sağlık", 0.40))
    invoice = extract_from_text(
        "Fatura No : TST9\nFatura Tarihi : 01.02.2026\nOdenecek Tutar 100,00 TL\n"
        "Bilinmeyen bir kalem\n",
        use_llm=False,
    )
    assert invoice.category == "sağlık"
    assert "category:low_confidence" in invoice.review_reasons


def test_classifier_can_be_disabled(monkeypatch):
    monkeypatch.setattr(category.classifier, "classify",
                        lambda text: pytest.fail("should not be called"))
    label, source = category.classify("Bilinmeyen", "kalem", use_llm=False,
                                      use_classifier=False)
    assert (label, source) == (category.DEFAULT_CATEGORY, "default")


def test_unavailable_classifier_falls_through(monkeypatch):
    monkeypatch.setattr(category.classifier, "classify", lambda text: None)
    label, source = category.classify("Bilinmeyen", "kalem", use_llm=False)
    assert (label, source) == (category.DEFAULT_CATEGORY, "default")


def test_probe_matches_the_training_text_shape(monkeypatch):
    """Training used "<vendor>. <items>"; serving must build the same string or the
    embedding geometry the head learned no longer applies.

    The sample text is deliberately free of any keyword rule -- "vida"/"civata" are
    themselves rules, so a nalbur example never reaches the classifier at all.
    """
    seen = []
    monkeypatch.setattr(category.classifier, "classify",
                        lambda text: seen.append(text) or ("ev", 0.9))
    category.classify("Mavi Deniz Ltd. Şti.", "genel tedarik kalemi", use_llm=False)
    assert seen, "classifier was never consulted"
    assert seen[0].startswith("Mavi Deniz Ltd. Şti.")
