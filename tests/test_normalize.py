"""Table-driven tests for Turkish number/date/rate parsing."""

from __future__ import annotations

import pytest

from malimusavir.normalize import (
    clean_ws,
    fold_tr,
    parse_tax_id,
    parse_tr_amount,
    parse_tr_date,
    parse_vat_rate,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Standard Turkish formatting.
        ("1.748,80 TL", 1748.80),
        ("748,80TL", 748.80),
        ("1.234.567,89", 1234567.89),
        ("0,00", 0.0),
        ("1.000,00 ₺", 1000.0),
        ("249,99 TRY", 249.99),
        # Whitespace variants produced by PDF extraction.
        ("1 748,80 TL", 1748.80),
        (" 748,80 TL", 748.80),
        ("  1.500,50  ", 1500.50),
        # Bare integers and plain decimals.
        ("20", 20.0),
        ("748.80", 748.80),  # one dot, 2-digit tail -> decimal
        ("1.748", 1748.0),  # one dot, 3-digit tail -> thousands
        ("1.234.567", 1234567.0),
        # English-formatted input still parses correctly.
        ("1,234.56", 1234.56),
        ("1,234,567", 1234567.0),
        # Negatives (credit notes / iade).
        ("-748,80 TL", -748.80),
        # Junk.
        ("", None),
        ("   ", None),
        ("TL", None),
        (None, None),
        # Passthrough for already-numeric values.
        (1748.8, 1748.8),
        (20, 20.0),
    ],
)
def test_parse_tr_amount(raw, expected):
    assert parse_tr_amount(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("03/03/2026", "2026-03-03"),
        ("03 / 03 / 2026", "2026-03-03"),
        ("03.03.2026", "2026-03-03"),
        ("03-03-2026", "2026-03-03"),
        ("3/3/2026", "2026-03-03"),
        ("2026-03-03", "2026-03-03"),
        ("Fatura Tarihi: 22.05.2026", "2026-05-22"),
        # Turkish month names, both cases -- the dotted/dotless i trap.
        ("3 Mart 2026", "2026-03-03"),
        ("03 MART 2026", "2026-03-03"),
        ("15 Mayıs 2026", "2026-05-15"),
        ("15 MAYIS 2026", "2026-05-15"),
        ("1 Şubat 2025", "2025-02-01"),
        ("1 ŞUBAT 2025", "2025-02-01"),
        ("9 Ağustos 2025", "2025-08-09"),
        ("30 Eylül 2025", "2025-09-30"),
        ("31 Aralık 2025", "2025-12-31"),
        # Two-digit years.
        ("03.03.26", "2026-03-03"),
        # Invalid calendar dates must not raise.
        ("31/02/2026", None),
        ("00/00/0000", None),
        ("", None),
        (None, None),
        ("hicbir tarih yok", None),
    ],
)
def test_parse_tr_date(raw, expected):
    assert parse_tr_date(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("%20", 20.0),
        ("20%", 20.0),
        ("20", 20.0),
        ("20,00", 20.0),
        ("%18", 18.0),
        ("%1", 1.0),
        ("%0", 0.0),
        ("0,20", 20.0),  # fraction -> percentage
        ("KDV Oranı: %10", 10.0),
        ("", None),
        (None, None),
        ("%200", None),  # out of range
    ],
)
def test_parse_vat_rate(raw, expected):
    assert parse_vat_rate(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1234567890", "1234567890"),
        ("Vergi Kimlik No: 1234567890", "1234567890"),
        ("123 456 7890", "1234567890"),
        ("12345678901", "12345678901"),
        ("123", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_tax_id(raw, expected):
    assert parse_tax_id(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MAYIS", "mayis"),
        ("Mayıs", "mayis"),
        ("ŞUBAT", "subat"),
        ("Ağustos", "agustos"),
        ("İSTANBUL", "istanbul"),
        ("Türk Telekom", "turk telekom"),
    ],
)
def test_fold_tr(raw, expected):
    assert fold_tr(raw) == expected


def test_fold_tr_makes_turkish_case_variants_equal():
    """The dotted/dotless i problem str.lower() cannot solve."""
    assert fold_tr("MAYIS") == fold_tr("Mayıs")
    assert "MAYIS".lower() != "Mayıs".lower()  # documents why fold_tr exists


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Fatura   No  ", "Fatura No"),
        ("Toplam:", "Toplam"),
        ("a\nb", "a b"),
        ("", None),
        ("   ", None),
        (None, None),
    ],
)
def test_clean_ws(raw, expected):
    assert clean_ws(raw) == expected
