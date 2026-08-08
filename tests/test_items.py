"""Line-item extraction -- the input that makes semantic retrieval work."""

from __future__ import annotations

from malimusavir.items import items_text, line_items

MARKETPLACE = """\
            D-MARKET ELEKTRONIK HIZMETLER VE
            Bogazici Kurumlar V.D.: 265 017 9910
            Teslimat Adresi: [GIZLENDI]
            No  Malzeme Aciklama                  KDV   Adet  B. Fiyat Tutar
                        Bosch 42 Parca Hassas Vidalama - Bits Ucu Seti -
            1   HBCV000067JAM9                    20      1   627,35 627,35
            YALNIZ ALTIYUZELLIIKI TL SEKSENIKI KR
            GENEL TOPLAM 652,82 TL
"""

COMPONENTS = """\
    IZMIT YAZILIM ROBOT TEKNOLOJILERI
    No  Hizmet / Urun Adi       Miktar Birim    Birim Fiyat KDV Orani     Toplam
    1   MAX98357 Modul            1    Adet     126,00 TL   %20,00      126,00 TL
    2   HX711 Modul               1    Adet      40,50 TL   %20,00       40,50 TL
    3   Kargo Bedeli              1    Adet     112,50 TL   %20,00      112,50 TL
                                     Mal Hizmet Toplam Tutari 369,00 TL
                                     Odenecek Tutar 442,80 TL
"""

TELECOM = """\
   Odenecek Tutar                                  [GIZLENDI]
   376,90  TL
  FATURA OZETI
   Iletisim Ucretleriniz                      350,00 TL
   Dolu Ultra 10 GB (Nisan)                     350,00
   Yuvarlama Farki                               -0,08
   Toplam Tutar                              376,90 TL
"""


def test_marketplace_item_is_found():
    items = line_items(MARKETPLACE)
    assert any("Vidalama" in item for item in items)
    assert any("Bosch" in item for item in items)


def test_header_and_totals_are_excluded():
    text = items_text(MARKETPLACE)
    assert "Bogazici" not in text          # header
    assert "GENEL TOPLAM" not in text      # totals block
    assert "Teslimat" not in text          # address


def test_multiple_components_are_captured():
    items = line_items(COMPONENTS)
    joined = " ".join(items)
    assert "MAX98357" in joined
    assert "HX711" in joined
    assert "Kargo" in joined
    assert "369,00" not in joined          # the totals row must not leak in


def test_telecom_summary_block_is_used():
    joined = items_text(TELECOM)
    assert "Dolu Ultra 10 GB" in joined
    assert "376,90" not in joined


def test_prices_and_rates_are_stripped():
    """Prices add no retrieval signal and crowd out the words that do."""
    joined = items_text(COMPONENTS)
    assert "126,00" not in joined
    assert "%20,00" not in joined


def test_column_header_fragments_are_dropped():
    """Multi-line table headers leave rows of pure header vocabulary behind."""
    text = "Sira Mal/Hizmet Cinsi Miktar Birim Fiyat\nOran Tutari Vergiler\nBME280 Sensor\n"
    assert line_items(text) == ["BME280 Sensor"]


def test_no_table_yields_nothing():
    assert line_items("Merhaba, bir sey yok burada\n") == []
