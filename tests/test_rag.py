"""Summary construction and vector round-tripping.

Retrieval quality itself is not asserted here -- it depends on a running model and is
measured separately. What is pinned down is that summaries carry the fields retrieval
needs, and that vectors survive storage intact.
"""

from __future__ import annotations

import numpy as np

from malimusavir import rag


def make_row(**overrides) -> dict:
    row = {
        "invoice_no": "DRN2026000025322",
        "date": "2026-04-30",
        "vendor": "Ornek Elektronik Ltd.",
        "category": "elektronik",
        "total_amount": 991.92,
        "payment_method": "KREDIKARTI",
        "raw_text": (
            "Sira Mal/Hizmet Cinsi Miktar Birim Fiyat\n"
            "1 BME280 I2C Basinc Sicaklik ve Nem Sensoru 1 Adet 157,27 TL\n"
            "2 ESP32-S3 Super Mini WiFi Bluetooth Modulu 1 Adet 265,12 TL\n"
            "Mal Hizmet Toplam Tutari 826,60 TL\n"
        ),
    }
    row.update(overrides)
    return row


def test_summary_contains_the_retrieval_fields():
    summary = rag.build_summary(make_row())
    assert "2026-04-30" in summary
    assert "Ornek Elektronik Ltd." in summary
    assert "elektronik" in summary
    assert "991.92" in summary
    assert "DRN2026000025322" in summary


def test_summary_contains_the_line_items():
    """Without these, invoices from one issuer are indistinguishable."""
    summary = rag.build_summary(make_row())
    assert "BME280" in summary
    assert "ESP32-S3" in summary


def test_summary_excludes_the_totals_block():
    assert "826,60" not in rag.build_summary(make_row())


def test_summary_survives_missing_fields():
    summary = rag.build_summary(
        make_row(date=None, vendor=None, category=None, total_amount=None,
                 payment_method=None, raw_text=None)
    )
    assert "bilinmeyen" in summary


def test_vector_blob_round_trip():
    vector = [0.5, -0.25, 0.125, 1.0]
    restored = rag._from_blob(rag._to_blob(vector))
    assert np.allclose(restored, np.asarray(vector, dtype=np.float32))
    assert restored.dtype == np.float32
