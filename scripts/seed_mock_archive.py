"""Build a synthetic multi-client archive from mock practice data, then ingest it.

The mock data (agency_clients) mirrors what a real accounting practice's records look
like: several clients of different legal forms, invoices in both directions, and filed
tax declarations. It has no PDFs -- it is JSON -- so this script renders each invoice and
declaration as a real PDF (via reportlab) using the same field labels the extractors look
for, writes them into `<root>/<client>/<year>/<faturalar|tahakkuk>/*.pdf`, and then runs
the existing archive ingest over the result.

This is throwaway test data for exercising the multi-client UI end-to-end -- not a
fixture the extractors are tuned against. Real documents remain the ones extraction is
checkpointed on.

Usage:
    .venv\\Scripts\\python.exe scripts\\seed_mock_archive.py [--root sample_archive] [--db faturalar_demo.db]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# reportlab's built-in Helvetica uses WinAnsi encoding, which mangles Turkish dotless-i
# and cedilla characters ("ı", "ğ", "ş" print as garbage). A real Unicode TTF is required
# so the rendered PDF text round-trips through pdfplumber correctly.
_FONT_NAME = "TRFont"
pdfmetrics.registerFont(TTFont(_FONT_NAME, r"C:\Windows\Fonts\arial.ttf"))

MONTHS_TR = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
             "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

MOCK_DATA = {
    "agency_clients": [
        {
            "client_id": "CUST-1003",
            "owner_name": "Canan Aydın",
            "commercial_title": "Canan Aydın E-Ticaret ve Danışmanlık",
            "company_type": "Şahıs Şirketi",
            "tax_office": "Şişli V.D.",
            "tax_number": "45678912345",
            "tax_number_type": "TCKN",
            "status": "Active",
            "financial_records": {
                "2026": {
                    "Ocak": {
                        "invoices": [
                            {
                                "invoice_number": "EAR2026000145892",
                                "issue_date": "2026-01-15T09:30:00Z",
                                "direction": "SATIŞ",
                                "counterparty": {"title": "Bireysel Müşteri", "vkn_tckn": "11111111111"},
                                "subtotal": 2500.00, "kdv_total": 500.00, "grand_total": 3000.00,
                                "kdv_rate": 20,
                            }
                        ],
                        "beyannameler": [
                            {
                                "type": "KDV1",
                                "filing_date": "2026-02-24T10:15:00Z",
                                "accrual_receipt_no": "2026022401L520003001",
                                "amount_due": 500.00,
                                "status": "Ödendi",
                            }
                        ],
                    },
                    "Şubat": {
                        "invoices": [
                            {
                                "invoice_number": "EAR2026000146011",
                                "issue_date": "2026-02-11T10:05:00Z",
                                "direction": "SATIŞ",
                                "counterparty": {"title": "Bireysel Müşteri", "vkn_tckn": "22222222222"},
                                "subtotal": 1800.00, "kdv_total": 360.00, "grand_total": 2160.00,
                                "kdv_rate": 20,
                            }
                        ],
                        "beyannameler": [
                            {
                                "type": "KDV1",
                                "filing_date": "2026-03-24T10:00:00Z",
                                "accrual_receipt_no": "2026032401L520003002",
                                "amount_due": 360.00,
                                "status": "Ödendi",
                            }
                        ],
                    },
                    "Mart": {
                        "invoices": [
                            {
                                "invoice_number": "EAR2026000147233",
                                "issue_date": "2026-03-08T13:40:00Z",
                                "direction": "SATIŞ",
                                "counterparty": {"title": "Bireysel Müşteri", "vkn_tckn": "33333333333"},
                                "subtotal": 3200.00, "kdv_total": 640.00, "grand_total": 3840.00,
                                "kdv_rate": 20,
                            }
                        ],
                    },
                }
            },
            "documents": [
                {"filename": "Vergi Levhası.pdf", "doc_type": "belgeler"},
                {"filename": "İşyeri Açma Ruhsatı.pdf", "doc_type": "belgeler"},
            ],
        },
        {
            "client_id": "CUST-1004",
            "owner_name": None,
            "commercial_title": "Kaya Yapı Mimarlık İnşaat Ltd. Şti.",
            "company_type": "Limited Şirketi",
            "tax_office": "Nilüfer V.D.",
            "tax_number": "5556667778",
            "tax_number_type": "VKN",
            "status": "Active",
            "financial_records": {
                "2026": {
                    "Ocak": {
                        "invoices": [
                            {
                                "invoice_number": "GIB2026000000789",
                                "issue_date": "2026-01-10T14:20:00Z",
                                "direction": "ALIŞ",
                                "counterparty": {"title": "Çelik Hazır Beton A.Ş.", "vkn_tckn": "3332221110"},
                                "subtotal": 100000.00, "kdv_total": 20000.00, "grand_total": 120000.00,
                                "kdv_rate": 20,
                            }
                        ],
                        "beyannameler": [
                            {
                                "type": "KDV1",
                                "filing_date": "2026-02-24T16:00:00Z",
                                "accrual_receipt_no": "2026022401L520004001",
                                "amount_due": 0.00,
                                "status": "Devreden KDV",
                            }
                        ],
                    },
                    "Şubat": {
                        "invoices": [
                            {
                                "invoice_number": "GIB2026000001355",
                                "issue_date": "2026-02-05T09:10:00Z",
                                "direction": "ALIŞ",
                                "counterparty": {"title": "Demir Çelik İnşaat Malzemeleri", "vkn_tckn": "4443332221"},
                                "subtotal": 65000.00, "kdv_total": 13000.00, "grand_total": 78000.00,
                                "kdv_rate": 20,
                            }
                        ],
                    },
                    "Nisan": {
                        "invoices": [
                            {
                                "invoice_number": "GIB2026000002980",
                                "issue_date": "2026-04-18T11:00:00Z",
                                "direction": "ALIŞ",
                                "counterparty": {"title": "Vinç ve İş Makinaları Kiralama", "vkn_tckn": "5554443332"},
                                "subtotal": 30000.00, "kdv_total": 6000.00, "grand_total": 36000.00,
                                "kdv_rate": 20,
                            }
                        ],
                        "beyannameler": [
                            {
                                "type": "KDV1",
                                "filing_date": "2026-05-24T16:00:00Z",
                                "accrual_receipt_no": "2026052401L520004002",
                                "amount_due": 6000.00,
                                "status": "Tahakkuk Kesildi",
                            }
                        ],
                    },
                }
            },
            "documents": [
                {"filename": "İş Yeri Kira Sözleşmesi.pdf", "doc_type": "belgeler"},
                {"filename": "İmza Sirküleri.pdf", "doc_type": "belgeler"},
            ],
        },
        {
            "client_id": "CUST-1005",
            "owner_name": "Mustafa Arslan",
            "commercial_title": "Mustafa Arslan Gıda Üretim",
            "company_type": "Şahıs Şirketi",
            "tax_office": "Konak V.D.",
            "tax_number": "98712365401",
            "tax_number_type": "TCKN",
            "status": "Passive",
            "financial_records": {
                "2026": {
                    "Ocak": {
                        "invoices": [
                            {
                                "invoice_number": "FKB2026000000101",
                                "issue_date": "2026-01-20T11:45:00Z",
                                "direction": "SATIŞ",
                                "counterparty": {"title": "Süpermarket Zinciri A.Ş.", "vkn_tckn": "7778889990"},
                                "subtotal": 5000.00, "kdv_total": 50.00, "grand_total": 5050.00,
                                "kdv_rate": 1,
                            }
                        ],
                        "beyannameler": [
                            {
                                "type": "KDV1",
                                "filing_date": "2026-02-24T09:00:00Z",
                                "accrual_receipt_no": "2026022401L520005001",
                                "amount_due": 50.00,
                                "status": "Tahakkuk Kesildi",
                            }
                        ],
                    },
                    "Mart": {
                        "invoices": [
                            {
                                "invoice_number": "FKB2026000000188",
                                "issue_date": "2026-03-14T10:20:00Z",
                                "direction": "SATIŞ",
                                "counterparty": {"title": "Yerel Zincir Market", "vkn_tckn": "8889990001"},
                                "subtotal": 4200.00, "kdv_total": 42.00, "grand_total": 4242.00,
                                "kdv_rate": 1,
                            }
                        ],
                    },
                }
            },
            "documents": [
                {"filename": "Gıda Üretim İzin Belgesi.pdf", "doc_type": "belgeler"},
            ],
        },
        {
            "client_id": "CUST-1006",
            "owner_name": None,
            "commercial_title": "Zirve Lojistik Hizmetleri A.Ş.",
            "company_type": "Anonim Şirketi",
            "tax_office": "Çankaya V.D.",
            "tax_number": "1020304050",
            "tax_number_type": "VKN",
            "status": "Active",
            "financial_records": {
                "2026": {
                    "Ocak": {
                        "invoices": [
                            {
                                "invoice_number": "GIB2026000455887",
                                "issue_date": "2026-01-22T07:50:00Z",
                                "direction": "ALIŞ",
                                "counterparty": {"title": "Lastik ve Bakım Servisi Ltd. Şti.", "vkn_tckn": "7776665552"},
                                "subtotal": 18000.00, "kdv_total": 3600.00, "grand_total": 21600.00,
                                "kdv_rate": 20,
                            }
                        ],
                        "beyannameler": [
                            {
                                "type": "MUHSGK",
                                "filing_date": "2026-02-26T13:00:00Z",
                                "accrual_receipt_no": "2026022601L520006001",
                                "amount_due": 9800.50,
                                "status": "Ödendi",
                            }
                        ],
                    },
                    "Şubat": {
                        "invoices": [
                            {
                                "invoice_number": "GIB2026000456123",
                                "issue_date": "2026-02-12T08:15:00Z",
                                "direction": "ALIŞ",
                                "counterparty": {"title": "Marmara Akaryakıt Dağıtım", "vkn_tckn": "6665554443"},
                                "subtotal": 42500.00, "kdv_total": 8500.00, "grand_total": 51000.00,
                                "kdv_rate": 20,
                            }
                        ],
                        "beyannameler": [
                            {
                                "type": "MUHSGK",
                                "filing_date": "2026-03-26T13:30:00Z",
                                "accrual_receipt_no": "2026032601L520006002",
                                "amount_due": 12450.75,
                                "status": "Ödendi",
                            }
                        ],
                    },
                }
            },
            "documents": [
                {"filename": "Taşıma Yetki Belgesi (K1).pdf", "doc_type": "belgeler"},
                {"filename": "Filo Sigorta Poliçesi.pdf", "doc_type": "belgeler"},
            ],
        },
        {
            "client_id": "CUST-1007",
            "owner_name": "Zeynep Çelik",
            "commercial_title": "Zeynep Çelik Yazılım Geliştirme",
            "company_type": "Şahıs Şirketi",
            "tax_office": "Kadıköy V.D.",
            "tax_number": "56473829102",
            "tax_number_type": "TCKN",
            "status": "Active",
            "financial_records": {
                "2026": {
                    "Ocak": {
                        "invoices": [
                            {
                                "invoice_number": "EAR2026000000055",
                                "issue_date": "2026-01-30T17:00:00Z",
                                "direction": "SATIŞ",
                                "counterparty": {"title": "Global Reklam Ajansı", "vkn_tckn": "1231231234"},
                                "subtotal": 45000.00, "kdv_total": 9000.00, "grand_total": 54000.00,
                                "kdv_rate": 20,
                            }
                        ],
                        "beyannameler": [
                            {
                                "type": "KDV1",
                                "filing_date": "2026-02-24T15:45:00Z",
                                "accrual_receipt_no": "2026022401L520007001",
                                "amount_due": 9000.00,
                                "status": "Tahakkuk Kesildi",
                            }
                        ],
                    },
                    "Şubat": {
                        "invoices": [
                            {
                                "invoice_number": "EAR2026000000091",
                                "issue_date": "2026-02-20T16:10:00Z",
                                "direction": "SATIŞ",
                                "counterparty": {"title": "Kuzey Yıldızı Perakende A.Ş.", "vkn_tckn": "2342342345"},
                                "subtotal": 30000.00, "kdv_total": 6000.00, "grand_total": 36000.00,
                                "kdv_rate": 20,
                            }
                        ],
                        "beyannameler": [
                            {
                                "type": "KDV1",
                                "filing_date": "2026-03-24T15:00:00Z",
                                "accrual_receipt_no": "2026032401L520007002",
                                "amount_due": 6000.00,
                                "status": "Tahakkuk Kesildi",
                            }
                        ],
                    },
                    "Mart": {
                        "invoices": [
                            {
                                "invoice_number": "EAR2026000000122",
                                "issue_date": "2026-03-25T09:30:00Z",
                                "direction": "SATIŞ",
                                "counterparty": {"title": "Global Reklam Ajansı", "vkn_tckn": "1231231234"},
                                "subtotal": 20000.00, "kdv_total": 4000.00, "grand_total": 24000.00,
                                "kdv_rate": 20,
                            }
                        ],
                    },
                }
            },
            "documents": [
                {"filename": "Yazılım Lisans Sözleşmesi.pdf", "doc_type": "belgeler"},
                {"filename": "Serbest Meslek Makbuzu Yetki Belgesi.pdf", "doc_type": "belgeler"},
            ],
        },
    ]
}

#: KDV1 uses the real "Gerçek Usulde KDV" code. MUHSGK has no single Ana Vergi Kodu on a
#: real receipt (it's a combined payroll/withholding form) -- "0027" is left unmapped in
#: TAX_CODES on purpose so tahakkuk.py falls back to matching "muhtasar" in the label text.
_DECL_CODE = {"KDV1": "0015", "MUHSGK": "0027"}

_INVALID_FS = re.compile(r'[\\/:*?"<>|]')


def _safe_folder(name: str) -> str:
    # Windows silently drops trailing dots/spaces when a directory is created, so a name
    # ending "Ltd. Şti." becomes "Ltd. Şti" on disk. Stripping it here up front keeps the
    # folder name (used for clients.resolve before ingest) identical to what archive.walk
    # will actually see after ingest -- otherwise the two runs create two client rows.
    return _INVALID_FS.sub("_", name).strip().rstrip(". ")


def _money(value: float) -> str:
    """1234.5 -> '1.234,50', matching normalize.parse_money's expectations."""
    whole, cents = f"{value:,.2f}".split(".")
    whole = whole.replace(",", ".")
    return f"{whole},{cents}"


def _iso_to_tr(iso_date: str) -> str:
    date_part = iso_date.split("T")[0]
    year, month, day = date_part.split("-")
    return f"{day}.{month}.{year}"


def _draw(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont(_FONT_NAME, 10)
    width, height = A4
    y = height - 50
    for line in lines:
        c.drawString(50, y, line)
        y -= 14
        if y < 50:
            c.showPage()
            c.setFont(_FONT_NAME, 10)
            y = height - 50
    c.save()


def _invoice_pdf(path: Path, *, client, invoice: dict) -> None:
    is_sale = invoice["direction"] == "SATIŞ"
    if is_sale:
        vendor_name = client["commercial_title"]
        vendor_tax_id = client["tax_number"]
        buyer_name = invoice["counterparty"]["title"]
    else:
        vendor_name = invoice["counterparty"]["title"]
        vendor_tax_id = invoice["counterparty"]["vkn_tckn"]
        buyer_name = client["commercial_title"]

    lines = [
        vendor_name.upper(),
        "e-ARŞİV FATURA",
        "",
        f"FATURA NO: {invoice['invoice_number']}",
        f"FATURA TARİHİ: {_iso_to_tr(invoice['issue_date'])}",
        f"VERGİ KİMLİK NO: {vendor_tax_id}",
        "",
        f"SAYIN: {buyer_name}",
        "",
        f"MAL/HİZMET TOPLAM TUTARI: {_money(invoice['subtotal'])} TL",
        f"HESAPLANAN KDV(%{invoice['kdv_rate']}): {_money(invoice['kdv_total'])} TL",
        f"ÖDENECEK TUTAR: {_money(invoice['grand_total'])} TL",
        "PARA BİRİMİ: TRY",
        "ÖDEME ŞEKLİ: Banka Havalesi",
    ]
    _draw(path, lines)


def _tahakkuk_pdf(path: Path, *, client, declaration: dict, period_year: int,
                   period_month: int) -> None:
    """`period_year`/`period_month` is the tax period being declared -- e.g. January's
    KDV -- which is NOT the same month as `filing_date` (KDV is filed by the 24th-26th
    of the *following* month). Stamping the receipt with the filing month instead of the
    period would file a January declaration under February in the archive."""
    code = _DECL_CODE.get(declaration["type"], "0000")
    label = "MUHTASAR VE PRİM HİZMET BEYANNAMESİ" if declaration["type"] == "MUHSGK" else \
        "GERÇEK USULDE KATMA DEĞER VERGİSİ"
    filing = declaration["filing_date"].split("T")[0]
    f_year, f_month, _ = filing.split("-")
    p_month = f"{period_month:02d}"
    period = f"{p_month}/{period_year}-{p_month}/{period_year}"
    due = f"28/{f_month}/{f_year}"
    amount = declaration["amount_due"]
    digits = "".join(ch for ch in client["tax_number"] if ch.isdigit())

    lines = [
        "TAHAKKUK FİŞİ",
        client["tax_office"].upper(),
        "",
        declaration["accrual_receipt_no"],
        "",
        f"VERGİ KİMLİK NUMARASI {digits}",
        f"SOYADI (UNVANI)  {client['commercial_title']}",
        "",
        f"Ana Vergi Kodu   {code}",
        label,
        "",
        f"Kabul Tarihi  Vergilendirme Dönemi        Düzenleme Tarihi",
        f"{_iso_to_tr(filing)}  {period}        {_iso_to_tr(filing)}",
        "",
        "TÜRÜ       MATRAH   TAHAKKUK EDEN   MAHSUP EDİLEN   ÖDENECEK OLAN   VADESİ",
        f"{code} {declaration['type']}   0,00      {_money(amount)}         0,00     "
        f"{_money(amount)}   {due}",
        "",
        f"                                    TOPLAM       {_money(amount)}",
    ]
    _draw(path, lines)


def _document_pdf(path: Path, *, client, filename: str) -> None:
    """A generic archived document -- unlike invoices and declarations, its content is
    never parsed (archive.py stores it as-is), so the text only needs to be plausible."""
    lines = [
        filename.rsplit(".", 1)[0].upper(),
        "",
        client["commercial_title"],
        f"Vergi No: {client['tax_number']}",
        f"{client['tax_office']}",
        "",
        "Bu belge örnek/mock veridir.",
    ]
    _draw(path, lines)


def build_archive(root: Path) -> None:
    for client in MOCK_DATA["agency_clients"]:
        folder = _safe_folder(client["owner_name"] or client["commercial_title"])
        for year_str, months in client["financial_records"].items():
            year = int(year_str)
            for month_name, records in months.items():
                month_num = MONTHS_TR.index(month_name) + 1
                for invoice in records.get("invoices", []):
                    dest = root / folder / year_str / "faturalar" / f"{invoice['invoice_number']}.pdf"
                    _invoice_pdf(dest, client=client, invoice=invoice)
                for decl in records.get("beyannameler", []):
                    dest = root / folder / year_str / "tahakkuk" / f"{decl['accrual_receipt_no']}.pdf"
                    _tahakkuk_pdf(dest, client=client, declaration=decl,
                                  period_year=year, period_month=month_num)

            # Misc documents have no month of their own in the mock data -- they're
            # filed once per year, same as a real "belgeler" folder holding licences
            # and contracts rather than dated transactions.
            for doc in client.get("documents", []):
                dest = root / folder / year_str / doc["doc_type"] / doc["filename"]
                _document_pdf(dest, client=client, filename=doc["filename"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="sample_archive", help="where to render the archive")
    parser.add_argument("--db", default="faturalar_demo.db", help="database to ingest into")
    parser.add_argument("--no-ingest", action="store_true", help="only render PDFs, skip ingest")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    build_archive(root)
    print(f"Wrote synthetic archive to {root}")

    if args.no_ingest:
        return

    from malimusavir import clients as clients_mod, db, pipeline

    conn = db.connect(args.db)

    # tax_id must be on file *before* ingest: direction_for() compares each invoice's
    # seller tax id against the client's own, and with no tax_id every invoice reads as
    # a purchase regardless of who actually issued it.
    for client in MOCK_DATA["agency_clients"]:
        folder = _safe_folder(client["owner_name"] or client["commercial_title"])
        resolved = clients_mod.resolve(conn, folder)
        clients_mod.set_metadata(
            conn, resolved.id,
            tax_id=client["tax_number"], form=client["company_type"],
            city=client["tax_office"], display=client["commercial_title"],
        )

    report = pipeline.ingest_archive(conn, root)
    conn.close()

    print(f"Clients: {', '.join(report.clients)}")
    print(f"Invoices: inserted={report.invoices.inserted} updated={report.invoices.updated} "
          f"failed={len(report.invoices.failed)}")
    for path, reason in report.invoices.failed:
        print(f"  FAILED {path}: {reason}")
    print(f"Declarations ingested: {report.declarations}")
    if report.problems:
        print("Problems:")
        for path, reason in report.problems:
            print(f"  {path}: {reason}")
    print(f"\nDatabase written to {args.db}")
    print(f'Serve it with: .venv\\Scripts\\python.exe main.py --serve --db "{args.db}"')


if __name__ == "__main__":
    main()
