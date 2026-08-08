"""Category inference. The keyword path is tested offline; the LLM path is not
exercised here because these tests must run without Foundry Local."""

from __future__ import annotations

import pytest

from malimusavir.category import (
    CATEGORIES,
    DEFAULT_CATEGORY,
    _coerce,
    classify,
    excerpt_for_classification,
)


@pytest.mark.parametrize(
    ("vendor", "text", "expected"),
    [
        ("Turkcell İletişim Hizmetleri A.Ş.", "Aylık Paket Ücreti", "telekom"),
        ("Turkcell Superonline", "Fiber internet", "telekom"),
        (None, "Anker Bluetooth Kulaklık", "elektronik"),
        (None, "Netflix üyelik bedeli", "abonelik"),
        (None, "Migros market alışveriş", "market"),
        (None, "Opet motorin", "ulaşım"),
        (None, "Eczane ilaç bedeli", "sağlık"),
        (None, "Toner ve kartuş", "ofis"),
        (None, "Proje Danışmanlık Hizmeti", "hizmet"),
        (None, "LC Waikiki tişört", "giyim"),
        (None, "Elektrik tüketim bedeli", "enerji"),
        (None, "Zorunlu trafik sigortası poliçesi", "sigorta"),
        (None, "Bahçe peyzaj işçiliği", "hizmet"),
        (None, "Filtre kahve ve sandviç", "yeme-içme"),
        (None, "Buzdolabı beyaz eşya", "ev"),
    ],
)
def test_keyword_classification(vendor, text, expected):
    category, source = classify(vendor, text, use_llm=False)
    assert category == expected
    assert source == "keyword"


def test_unknown_falls_back_without_llm():
    category, source = classify(None, "tanimsiz bir kalem", use_llm=False)
    assert category == DEFAULT_CATEGORY
    assert source == "default"


def test_every_category_is_reachable_as_canonical():
    for category in CATEGORIES:
        assert _coerce(category) == category


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("elektronik", "elektronik"),
        ("Elektronik", "elektronik"),
        ("saglik", "sağlık"),  # ASCII-folded model output still resolves
        ("ulasim", "ulaşım"),
        ("diger", "diğer"),
        ("Kategori: telekom", "telekom"),
        ("yeme-icme", "yeme-içme"),  # hyphenated label must survive coercion
        ("yeme-içme", "yeme-içme"),
        ("ev eşyası", "ev"),
        ("bilinmeyen sey", None),
        ("", None),
    ],
)
def test_coerce_model_output(raw, expected):
    assert _coerce(raw) == expected


def test_coerce_does_not_match_labels_mid_word():
    """The two-letter 'ev' label must not fire inside an unrelated word."""
    assert _coerce("seviye") is None


def test_llm_sourced_category_is_flagged_for_review():
    """Model-inferred categories must never enter the aggregates unchallenged."""
    from malimusavir.pipeline import extract_from_text

    inv = extract_from_text(
        "Fatura No : TST1\nFatura Tarihi : 01.01.2026\nOdenecek Tutar 100,00 TL\n"
        "Tanimsiz bir kalem\n",
        use_llm=False,
    )
    assert inv.category == DEFAULT_CATEGORY
    assert "category:unresolved" in inv.review_reasons


def test_keyword_category_is_not_flagged():
    from malimusavir.pipeline import extract_from_text

    inv = extract_from_text(
        "Fatura No : TST2\nFatura Tarihi : 01.01.2026\nOdenecek Tutar 100,00 TL\n"
        "Turkcell aylik paket ucreti\n",
        use_llm=False,
    )
    assert inv.category == "telekom"
    assert not any(r.startswith("category:") for r in inv.review_reasons)


def test_excerpt_drops_totals_and_identifiers():
    text = (
        "Anker Soundcore Kulaklık\n"
        "Vergi Kimlik No : 4590874863\n"
        "Ödenecek Tutar 1.536,00 TL\n"
        "USB-C Şarj Kablosu\n"
    )
    excerpt = excerpt_for_classification(text)
    assert "Kulaklık" in excerpt
    assert "USB-C" in excerpt
    assert "4590874863" not in excerpt
    assert "Ödenecek" not in excerpt
