"""Extraction profile registry and dispatch."""

from __future__ import annotations

from .base import ExtractedInvoice, InvoiceProfile, search_label
from .generic_earsiv import GenericEArsivProfile
from .vendors import (
    AmazonProfile,
    DMarketProfile,
    SuperonlineProfile,
    TelecomBillProfile,
    TurkcellProfile,
)

#: Order matters -- the first profile whose markers match wins, so the most specific
#: issuers come first. Superonline precedes Turkcell because Superonline invoices also
#: mention "Turkcell" in the legal entity name.
REGISTRY: tuple[InvoiceProfile, ...] = (
    SuperonlineProfile(),
    TurkcellProfile(),
    AmazonProfile(),
    DMarketProfile(),
)

FALLBACK = GenericEArsivProfile()


def select_profile(doc_text: str) -> InvoiceProfile:
    """Pick the extraction profile for a document."""
    for profile in REGISTRY:
        if profile.matches(doc_text):
            return profile
    return FALLBACK


def extract_invoice(doc_text: str) -> ExtractedInvoice:
    """Run the best-matching profile and validate the result."""
    invoice = select_profile(doc_text).extract(doc_text)
    invoice.validate()
    return invoice


__all__ = [
    "REGISTRY",
    "FALLBACK",
    "ExtractedInvoice",
    "InvoiceProfile",
    "GenericEArsivProfile",
    "AmazonProfile",
    "TelecomBillProfile",
    "TurkcellProfile",
    "SuperonlineProfile",
    "DMarketProfile",
    "select_profile",
    "extract_invoice",
    "search_label",
]
