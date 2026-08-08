"""Golden-fixture tests for the extraction profiles.

Fixtures are redacted invoice *text* rather than PDFs, so they are safe to commit and
fast to run. Each asserts the full field set a profile is expected to produce.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from malimusavir.extractors import extract_invoice, select_profile

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("fixture", "expected_profile"),
    [
        ("amazon_earsiv.txt", "amazon"),
        ("turkcell_earsiv.txt", "turkcell"),
        ("generic_earsiv.txt", "generic_earsiv"),
    ],
)
def test_profile_dispatch(fixture, expected_profile):
    assert select_profile(load(fixture)).name == expected_profile


def test_amazon_extraction():
    inv = extract_invoice(load("amazon_earsiv.txt"))

    assert inv.invoice_no == "AMZ2026000045123"
    assert inv.date == "2026-03-03"
    assert inv.vendor_tax_id == "4590874863"
    assert inv.net_amount == 1280.00
    assert inv.tax_amount == 256.00
    assert inv.total_amount == 1536.00
    assert inv.vat_rate == 20.0
    assert inv.currency == "TL"
    assert inv.payment_method == "KREDIKARTI"
    assert inv.vendor == "Amazon Turkey Perakende Hizmetleri Limited Şirketi"
    assert not inv.needs_review, inv.review_reasons


def test_turkcell_extraction():
    inv = extract_invoice(load("turkcell_earsiv.txt"))

    assert inv.invoice_no == "TCL2026000998877"
    assert inv.date == "2026-04-21"
    assert inv.vendor_tax_id == "8710365349"
    assert inv.net_amount == 624.00
    assert inv.tax_amount == 124.80
    assert inv.total_amount == 748.80
    assert inv.vat_rate == 20.0
    assert not inv.needs_review, inv.review_reasons


def test_generic_extraction_with_turkish_month_name():
    inv = extract_invoice(load("generic_earsiv.txt"))

    assert inv.invoice_no == "YMD2025000001204"
    assert inv.date == "2025-05-15"
    assert inv.vendor_tax_id == "2930118475"
    assert inv.net_amount == 12500.00
    assert inv.tax_amount == 2500.00
    assert inv.total_amount == 15000.00
    assert inv.vat_rate == 20.0
    assert inv.vendor == "Yıldız Mühendislik ve Danışmanlık Limited Şirketi"
    assert not inv.needs_review, inv.review_reasons


def test_every_field_records_its_source():
    inv = extract_invoice(load("amazon_earsiv.txt"))
    assert inv.field_sources["total_amount"] == "regex:amazon"
    # category is the one inferred field and is filled in later, not by a profile.
    assert inv.field_sources["category"] == "missing"


def test_reconcile_mismatch_is_flagged():
    """A total that does not equal net + tax must not pass silently."""
    text = load("amazon_earsiv.txt").replace("Ödenecek Tutar                  1.536,00 TL",
                                             "Ödenecek Tutar                  9.999,00 TL")
    inv = extract_invoice(text)
    assert inv.needs_review
    assert any(r.startswith("reconcile:") for r in inv.review_reasons)


def test_missing_required_fields_are_flagged():
    inv = extract_invoice("Fatura\nhicbir sey yok\n")
    assert inv.needs_review
    assert "missing:invoice_no" in inv.review_reasons
    assert "missing:total_amount" in inv.review_reasons
