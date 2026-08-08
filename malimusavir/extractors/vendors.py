"""Issuer-specific profiles.

Each subclass overrides only what its issuer actually does differently.

A caution learned from real documents: profile markers must identify the *issuer*, not
words that merely appear somewhere on the page. An earlier "bosch" profile matched a
Hepsiburada invoice because "Bosch" was the brand of the item purchased; the seller was
D-MARKET. Markers therefore key on legal entity names and issuer domains only.
"""

from __future__ import annotations

from .base import ExtractedInvoice
from .generic_earsiv import GenericEArsivProfile


class TelecomBillProfile(GenericEArsivProfile):
    """Shared behaviour for Turkcell-family monthly bills.

    These PDFs are the customer-facing "bilgilendirme" summaries -- they state plainly
    that they carry no legal validity, the real e-Fatura having gone to the GIB system.
    They use their own wording throughout, and print the payable amount two lines below
    its label with the recipient's address in between.
    """

    def _total(self, text: str) -> float | None:
        return self._amount(
            text,
            r"odenecek\s*tutar",
            r"toplam\s*tutar",
            r"fatura\s*tutari",
            r"toplam\s*borc",
        ) or super()._total(text)

    def extract(self, doc_text: str) -> ExtractedInvoice:
        inv = super().extract(doc_text)
        src = f"regex:{self.name}"

        # "Devlete iletilecek vergiler toplami: 135,26 TL'dir." is the authoritative
        # total tax and must win over the generic KDV labels, which on these bills
        # match a per-tax breakdown table on a later page and report the KDV component
        # alone (92,95 where the true total including OIV was 135,26).
        tax = self._amount(doc_text, r"devlete\s*iletilecek\s*vergiler\s*toplami")
        if tax is not None:
            inv.set("tax_amount", tax, src)

        # These bills carry no "Mal/Hizmet Toplam Tutari" line at all, so net is
        # derived. Trusting the generic net labels here pulled unrelated figures off
        # later pages -- one invoice came back with a net of -0,02, the rounding line.
        if None not in (inv.total_amount, inv.tax_amount):
            inv.set("net_amount", round(inv.total_amount - inv.tax_amount, 2), f"{src}:derived")
        return inv


class TurkcellProfile(TelecomBillProfile):
    name = "turkcell"
    markers = ("turkcell iletisim hizmetleri", "turkcell'li", "turkcell.com.tr")
    vendor_name = "Turkcell İletişim Hizmetleri A.Ş."


class SuperonlineProfile(TelecomBillProfile):
    name = "superonline"
    markers = ("superonline",)
    vendor_name = "Turkcell Superonline İletişim Hizmetleri A.Ş."


class DMarketProfile(GenericEArsivProfile):
    """Hepsiburada's invoicing entity."""

    name = "dmarket"
    markers = ("d-market elektronik", "hepsiburada", "d-market")
    vendor_name = "D-MARKET Elektronik Hizmetler ve Ticaret A.Ş."

    def _total(self, text: str) -> float | None:
        # "GENEL TOPLAM" is the payable figure. "TOPLAM" alone is the pre-discount
        # line total and "FATURA TOPLAMI" is net of VAT, so order matters here.
        return self._amount(text, r"genel\s*toplam", r"odenecek\s*tutar")


class AmazonProfile(GenericEArsivProfile):
    name = "amazon"
    markers = ("amazon turkey", "amazon.com.tr", "amazon turkiye")
    vendor_name = "Amazon Turkey Perakende Hizmetleri Limited Şirketi"
