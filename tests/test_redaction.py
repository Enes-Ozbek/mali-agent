"""Redaction must remove personal data before anything downstream sees it."""

from __future__ import annotations

import pytest

from malimusavir.pdf_text import REDACTED, is_valid_tckn, redact

# Checksum-valid synthetic TCKNs (not real identities).
VALID_TCKN = "10000000146"
ANOTHER_VALID_TCKN = "29343241086"


@pytest.mark.parametrize(
    ("digits", "expected"),
    [
        (VALID_TCKN, True),
        (ANOTHER_VALID_TCKN, True),
        ("12345678901", False),  # fails checksum
        ("00000000000", False),  # leading zero
        ("1234567890", False),  # 10 digits is a VKN, not a TCKN
        ("abcdefghijk", False),
        ("", False),
    ],
)
def test_is_valid_tckn(digits, expected):
    assert is_valid_tckn(digits) is expected


def test_labelled_tckn_is_removed():
    text = f"T.C. Kimlik No : {VALID_TCKN}\n"
    out = redact(text)
    assert VALID_TCKN not in out
    assert REDACTED in out


def test_bare_valid_tckn_is_removed():
    out = redact(f"Alici {VALID_TCKN} kayitli\n")
    assert VALID_TCKN not in out


def test_vendor_tax_id_is_preserved():
    """The seller's VKN is needed for grouping and must survive redaction."""
    out = redact("Vergi Kimlik No : 4590874863\n")
    assert "4590874863" in out


def test_invoice_number_is_not_eaten():
    """11-digit runs that fail the TCKN checksum are ordinary references."""
    out = redact("Siparis No : 40312345678\n")
    assert "40312345678" in out


def test_iban_is_removed():
    out = redact("IBAN: TR33 0006 1005 1978 6457 8413 26\n")
    assert "6457" not in out
    assert REDACTED in out


def test_email_and_phone_are_removed():
    out = redact("Mail: kisi@ornek.com.tr  Tel: 0532 123 45 67\n")
    assert "kisi@ornek.com.tr" not in out
    assert "532" not in out


def test_address_block_is_removed():
    out = redact("Teslimat Adresi : Bagdat Cad. No:12 Kadikoy/Istanbul\n")
    assert "Bagdat" not in out
    assert REDACTED in out


def test_masked_card_is_removed():
    out = redact("Kart: 4548 **** **** 1234\n")
    assert "1234" not in out


def test_redaction_is_idempotent():
    once = redact(f"T.C. Kimlik No : {VALID_TCKN}\n")
    assert redact(once) == once
