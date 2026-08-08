"""Integration test for load_pdf().

Builds a minimal but valid PDF in-process rather than committing a binary fixture, so
the pdfplumber -> text -> redaction path is exercised without shipping sample data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from malimusavir.pdf_text import REDACTED, load_pdf

VALID_TCKN = "10000000146"


def _make_pdf(lines: list[str]) -> bytes:
    """Assemble a one-page PDF with Helvetica text at fixed line positions."""
    text_ops = ["BT", "/F1 11 Tf", "50 740 Td", "14 TL"]
    for line in lines:
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        text_ops.append(f"({escaped}) Tj T*")
    text_ops.append("ET")
    stream = "\n".join(text_ops).encode("latin-1")

    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        b"<</Length %d>>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_at,
    )
    return bytes(out)


@pytest.fixture
def invoice_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "invoice.pdf"
    path.write_bytes(
        _make_pdf(
            [
                "e-ARSIV FATURA",
                "Ornek Teknoloji Limited Sirketi",
                "Vergi Kimlik No : 4590874863",
                f"T.C. Kimlik No : {VALID_TCKN}",
                "Fatura No : TST2026000000001",
                "Fatura Tarihi : 03.03.2026",
                "Odeme Sekli : KREDIKARTI",
                "Mal Hizmet Toplam Tutari 1.280,00 TL",
                "Toplam Vergi Tutari 256,00 TL",
                "Odenecek Tutar 1.536,00 TL",
            ]
        )
    )
    return path


def test_load_pdf_extracts_text(invoice_pdf):
    doc = load_pdf(invoice_pdf)
    assert doc.page_count == 1
    assert not doc.is_scanned
    assert "TST2026000000001" in doc.text


def test_load_pdf_redacts_personal_data(invoice_pdf):
    doc = load_pdf(invoice_pdf)
    assert VALID_TCKN not in doc.text
    assert REDACTED in doc.text
    # The seller's VKN must survive.
    assert "4590874863" in doc.text


def test_content_hash_is_stable_and_content_derived(invoice_pdf, tmp_path):
    first = load_pdf(invoice_pdf)
    copy = tmp_path / "renamed.pdf"
    copy.write_bytes(invoice_pdf.read_bytes())
    assert load_pdf(copy).content_hash == first.content_hash


def test_end_to_end_extraction_from_pdf(invoice_pdf):
    from malimusavir.pipeline import extract_from_pdf

    inv = extract_from_pdf(invoice_pdf, use_llm=False)
    assert inv.invoice_no == "TST2026000000001"
    assert inv.date == "2026-03-03"
    assert inv.vendor_tax_id == "4590874863"
    assert inv.net_amount == 1280.00
    assert inv.tax_amount == 256.00
    assert inv.total_amount == 1536.00
    assert inv.source_path == str(invoice_pdf)


def test_empty_pdf_is_flagged_as_scanned(tmp_path):
    path = tmp_path / "blank.pdf"
    path.write_bytes(_make_pdf([" "]))
    assert load_pdf(path).is_scanned
