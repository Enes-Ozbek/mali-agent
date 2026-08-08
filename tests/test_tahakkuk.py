"""Tahakkuk fişi extraction.

The fixture below reproduces the layout of a real receipt (an İstanbul/Avcılar KDV
accrual), with the taxpayer's identity replaced. `payable` is money the client owes, so
these tests assert on exact figures rather than on the parse merely succeeding.
"""

from __future__ import annotations

import pytest

from malimusavir import tahakkuk

REAL = """\
                                   TAHAKKUK FİŞİ
              T.C
          İSTANBUL
                                                            2026062601L520000305
              AVCILAR
                                                  MAKİNA NO
                                                  SIRA NO
    VERGİ KİMLİK NUMARASI [GIZLENDI] ( T.C. Kimlik No )
    SOYADI (UNVANI)     [GIZLENDI]
                                                  ADRES
    Ana Vergi Kodu      0015
    GERÇEK USULDE KATMA DEĞER VERGİSİ
                                         Düzenleme
       Kabul Tarihi  Vergilendirme Dönemi
                                          Tarihi
       26/06/2026 05/2026-05/2026        26/06/2026
                                 TAHAKKUK     MAHSUP      ÖDENECEK
                                                                      VADESİ
                                  EDEN        EDİLEN        OLAN
       TÜRÜ       MATRAH
    0015 KDV            0,00            0,00     11.861,90       0,00 28/06/2026
    1048 5035           0,00          791,00        0,00        791,00 28/06/2026
                                                TOPLAM          791,00
                                                              6670297133
    YALNIZ YEDİYÜZDOKSANBİR TL .dir
"""


@pytest.fixture
def parsed():
    return tahakkuk.parse(REAL)


def test_recognises_the_document(parsed):
    assert tahakkuk.looks_like_tahakkuk(REAL)
    assert not tahakkuk.looks_like_tahakkuk("e-ARŞİV FATURA\nÖdenecek Tutar 100,00 TL")


def test_receipt_serial(parsed):
    assert parsed.receipt_no == "2026062601L520000305"


def test_taxation_period(parsed):
    """"05/2026-05/2026" is a range even for a single month."""
    assert parsed.period == "2026-05"
    assert parsed.period_end == "2026-05"


def test_dates(parsed):
    assert parsed.accepted_date == "2026-06-26"
    assert parsed.issue_date == "2026-06-26"
    assert parsed.due_date == "2026-06-28"


def test_primary_tax_comes_from_ana_vergi_kodu(parsed):
    assert parsed.kind == "kdv"


def test_assessment_lines(parsed):
    assert len(parsed.lines) == 2

    kdv, damga = parsed.lines
    assert (kdv.code, kdv.kind) == ("0015", "kdv")
    assert kdv.offset == pytest.approx(11861.90)
    assert kdv.payable == pytest.approx(0.0)

    assert (damga.code, damga.kind) == ("1048", "damga")
    assert damga.accrued == pytest.approx(791.00)
    assert damga.payable == pytest.approx(791.00)


def test_total_payable_is_the_figure_owed(parsed):
    assert parsed.total_payable == pytest.approx(791.00)


def test_taxpayer_vkn_is_not_the_receipt_serial(parsed):
    """The serial opens with ten digits then a letter; a digits-only boundary read that
    prefix as the VKN."""
    assert parsed.taxpayer_tax_id == "6670297133"
    assert not parsed.receipt_no.startswith(parsed.taxpayer_tax_id)


def test_a_clean_receipt_needs_no_review(parsed):
    assert parsed.review_reasons == []
    assert parsed.needs_review is False


# --- the checks that stop a wrong figure being presented as fact --------------------


def test_rows_that_do_not_sum_to_the_total_are_flagged():
    """A misread row is exactly how a wrong "amount owed" would reach the user."""
    broken = REAL.replace("TOPLAM          791,00", "TOPLAM        9.999,00")
    result = tahakkuk.parse(broken)
    assert result.needs_review
    assert any(r.startswith("reconcile:") for r in result.review_reasons)


def test_a_document_without_the_table_is_flagged_not_zeroed():
    result = tahakkuk.parse("TAHAKKUK FİŞİ\nbaska hicbir sey yok\n")
    assert result.needs_review
    assert "missing:assessment_lines" in result.review_reasons
    assert "missing:total" in result.review_reasons
    assert result.total_payable is None      # never silently 0.00


def test_a_non_tahakkuk_document_is_rejected():
    result = tahakkuk.parse("e-ARŞİV FATURA\nÖdenecek Tutar 100,00 TL\n")
    assert "tahakkuk:not_recognised" in result.review_reasons
    assert result.lines == []


def test_parse_never_raises_on_junk():
    for junk in ("", "   ", "TAHAKKUK", "\x00\x01"):
        tahakkuk.parse(junk)      # must not raise


@pytest.mark.parametrize(
    ("code", "kind"),
    [("0015", "kdv"), ("1048", "damga"), ("0032", "gecici"), ("0010", "kurumlar")],
)
def test_known_tax_codes(code, kind):
    assert tahakkuk.TAX_CODES[code] == kind
