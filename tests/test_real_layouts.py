"""Regression tests for defects found only against real invoices.

Every case here reproduces a layout quirk that broke extraction or leaked personal
data on an actual document. The text is synthetic -- real names, addresses and tax
numbers are replaced -- but the *shape* is copied faithfully, because the shape is
what caused each bug.
"""

from __future__ import annotations

from malimusavir.extractors import extract_invoice, select_profile
from malimusavir.extractors.base import search_label_money
from malimusavir.normalize import parse_money
from malimusavir.pdf_text import REDACTED, redact

# --- strict money parsing -------------------------------------------------------


def test_money_parser_rejects_address_numbers():
    """A delivery address shares a layout line with "Odenecek Tutar"."""
    assert parse_money("No:119 D:3 Avcilar Istanbul 34320") is None


def test_money_parser_rejects_vat_rate():
    assert parse_money("%20,00") is None
    assert parse_money("KDV Orani %20") is None


def test_money_parser_accepts_real_amounts():
    assert parse_money("600,00  TL") == 600.0
    assert parse_money("1.536,00 TL") == 1536.0
    assert parse_money("GENEL TOPLAM 652,82 TL") == 652.82


# --- two-column layout ----------------------------------------------------------

TELECOM_TOTAL_BLOCK = """\
                                                   ADI SOYADI
                                                   Ornek Mah. Ornek Sk.
   Odenecek Tutar                                  No:119 D:3 Avcilar Istanbul 34320
                                                   ISTANBUL
   600,00  TL
   Son Odeme Tarihi : 11.03.2026
"""


def test_amount_found_below_label_when_tail_is_another_column():
    """The label line's tail is the recipient address; the figure is two lines down."""
    assert search_label_money(TELECOM_TOTAL_BLOCK, r"odenecek\s*tutar") == 600.0


# --- label boundaries -----------------------------------------------------------

FATURA_NOTU = """\
    Fatura No                        : RZ12026000003430
    Fatura Tarihi                    : 29.04.2026
    Odenecek Tutar                     442,80 TL
    Fatura Notu
    Yaziyla Toplam Tutar: DortYuzKirkIkiTurkLirasi
"""


def test_fatura_notu_does_not_masquerade_as_fatura_no():
    """Without a trailing \\b, "Fatura Notu" matched "Fatura No" and returned "tu"."""
    assert extract_invoice(FATURA_NOTU).invoice_no == "RZ12026000003430"


# --- date selection -------------------------------------------------------------

NEXT_INVOICE_DATE = """\
    Duzenleme Tarihi / Zamani: 20 Nisan 2026 / 00:00
    Fatura No : 0012026083217398
    Odenecek Tutar 376,90 TL
    Son Odeme Tarihi : 4 Mayis 2026
    Bir Sonraki Fatura Tarihi: 20 Mayis 2026
"""


def test_next_invoice_date_is_not_the_invoice_date():
    """"Bir Sonraki Fatura Tarihi" dated 20 of 23 real invoices a month ahead."""
    assert extract_invoice(NEXT_INVOICE_DATE).date == "2026-04-20"


STACKED_DATE_HEADERS = """\
    Fatura Numarasi                                  RZ12026000003430
    Mersis No                                Fatura Tarihi    Odeme Tarihi
                                             29.04.2026 - 13:06:58 29.04.2026
    Odenecek Tutar 442,80 TL
"""


def test_shared_date_header_row_is_still_accepted():
    """"Odeme Tarihi" in the same header row must not disqualify "Fatura Tarihi"."""
    assert extract_invoice(STACKED_DATE_HEADERS).date == "2026-04-29"


# --- seller vs. everyone else ---------------------------------------------------

MARKETPLACE_INVOICE = """\
            e-Arsiv Fatura
            ORNEK MARKET ELEKTRONIK HIZMETLER VE
            TICARET A.S.
            Bogazici Kurumlar V.D.: 265 017 9910
            Musteri V.D. - VKN/TCKN : AVCILAR VERGI DA - 6670297133
            Fatura No:7245788381
            Duzenleme Tarihi: 13.05.2026
            Odeme Sekli:KREDI KARTI/BANKA KARTI
            Tasiyici VKN:2650701090
            FATURA TOPLAMI 544,02 TL
            KDV(%20) (544,02) 108,80 TL
            GENEL TOPLAM 652,82 TL
"""


def test_seller_tax_id_not_buyer_or_carrier():
    inv = extract_invoice(MARKETPLACE_INVOICE)
    assert inv.vendor_tax_id == "2650179910"      # the seller
    assert inv.vendor_tax_id != "6670297133"      # the buyer -- a privacy leak
    assert inv.vendor_tax_id != "2650701090"      # the courier


def test_marketplace_totals():
    inv = extract_invoice(MARKETPLACE_INVOICE)
    assert (inv.net_amount, inv.tax_amount, inv.total_amount) == (544.02, 108.80, 652.82)
    assert inv.date == "2026-05-13"
    assert not inv.needs_review, inv.review_reasons


# --- multi-line company names ---------------------------------------------------

WRAPPED_VENDOR = """\
           ORN-EL INTERNATIONAL
           ELEKTRONIK SANAYI VE
           TICARET LIMITED SIRKETI
           VKN: 4780059180
           Fatura No DRN2026000025322
           Fatura Tarihi 30-04-2026
           Mal Hizmet Toplam Tutari  826,60 TL
           Hesaplanan KDV(%20.00)    165,32 TL
           Odenecek Tutar            991,92 TL
"""


def test_wrapped_company_name_is_assembled():
    """The legal-form token lands on the last line; alone it yields a fragment."""
    vendor = extract_invoice(WRAPPED_VENDOR).vendor
    assert vendor == "ORN-EL INTERNATIONAL ELEKTRONIK SANAYI VE TICARET LIMITED SIRKETI"


# --- redaction ------------------------------------------------------------------


def test_long_invoice_number_is_not_redacted_as_a_card():
    """A 16-digit invoice number has the same shape as an unseparated PAN."""
    out = redact("Fatura No : 0012026083217398\n")
    assert "0012026083217398" in out


def test_address_redaction_preserves_the_label_beside_it():
    """Blanking the whole line destroyed "Odenecek Tutar" and lost every total."""
    out = redact("   Odenecek Tutar          Ornek Mah. Karadut Sk. No:119 D:3\n")
    assert "Odenecek Tutar" in out
    assert "Karadut" not in out
    assert REDACTED in out


def test_recipient_name_removed_everywhere_including_the_greeting():
    text = (
        "Merhaba,\n"
        "ADI   SOYADI\n"
        "                    Sayin\n"
        "                    Adi Soyadi\n"
        "                    Ornek Mah. Ornek Sk. No:2 D:12\n"
    )
    out = redact(text)
    assert "SOYADI" not in out.upper()


def test_buyer_tax_id_in_recipient_block_is_removed():
    text = (
        "   Ornek Sirketi\n"
        "   VKN: 4780059180\n"
        "   SAYIN\n"
        "   ADI SOYADI\n"
        "   VKN: 6670297133\n"
    )
    out = redact(text)
    assert "6670297133" not in out
    assert "4780059180" in out      # the seller's must survive


def test_configured_terms_are_removed():
    out = redact("Merhaba, ORNEK KISI\n", extra_terms=["Ornek Kisi"])
    assert "ORNEK KISI" not in out.upper()


# --- profile dispatch -----------------------------------------------------------


def test_product_brand_does_not_select_a_profile():
    """A "Bosch" bit set on a marketplace invoice must not pick a Bosch profile."""
    text = MARKETPLACE_INVOICE.replace(
        "Fatura No:7245788381",
        "1 Bosch 42 Parca Hassas Vidalama - Bits Ucu Seti\n            Fatura No:7245788381",
    )
    assert select_profile(text).name in ("generic_earsiv", "dmarket")
