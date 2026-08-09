"""Label-anchored extraction for the standard GIB e-Arsiv layout.

e-Arsiv invoices are rendered from a government-mandated template, so the field labels
("Fatura No", "Odenecek Tutar", "Vergi Kimlik No") are stable across issuers. That is
what makes deterministic regex the right tool here and an LLM the wrong one: these
strings do not vary, and a small local model asked to copy digits will eventually
copy them wrong.

Every other profile inherits from this one and overrides only what its issuer differs on.
"""

from __future__ import annotations

import re

from ..normalize import clean_ws, fold_tr, parse_tax_id, parse_tr_date
from .base import ExtractedInvoice, InvoiceProfile, iter_label_hits

# Legal-form tokens that mark a line as a company name.
_COMPANY_TOKENS = (
    "limited sirketi", "anonim sirketi", "a.s.", "a.s ", " ltd", "ltd.", "sti", "sirketi",
)

# Words marking a tax number as belonging to someone other than the seller: the buyer,
# or the courier ("Tasiyici VKN"), which is a real trap on marketplace invoices.
_THIRD_PARTY_TOKENS = ("musteri", "sayin", "alici", "aliciya", "tasiyici", "tasiyan")

_INVOICE_NO = re.compile(r"\b([A-Z]{2,4}\d{10,20}|[A-Z0-9]{3}\d{4}\d{6,12}|\d{10,16})\b")


# Dates that sit on the page but are not the issue date.
DATE_DISQUALIFIERS = (
    "sonraki", "onceki", "son odeme", "vade", "siparis", "gonderim", "teslim",
    "baslangic", "bitis", "odeme tarihi", "iade",
)


def _is_third_party_context(fragment: str) -> bool:
    folded = fold_tr(fragment)
    return any(token in folded for token in _THIRD_PARTY_TOKENS)


def _is_other_date(line: str) -> bool:
    folded = fold_tr(line)
    return any(token in folded for token in DATE_DISQUALIFIERS)


def _is_name_continuation(line: str) -> bool:
    """Whether a line looks like the earlier part of a wrapped company name."""
    stripped = clean_ws(line)
    if not stripped or len(stripped) > 60:
        return False
    # Field labels, values and addresses all disqualify; a name fragment is plain text.
    if re.search(r"[:\d]", stripped):
        return False
    folded = fold_tr(stripped)
    return not any(
        token in folded
        for token in ("fatura", "vergi", "adres", "tel", "e-posta", "sayin", "musteri")
    )


class GenericEArsivProfile(InvoiceProfile):
    """Fallback profile driven purely by the standard GIB labels."""

    name = "generic_earsiv"
    # Deliberately empty: this profile is reached as select_profile()'s explicit
    # fallback, never by marker match. Leaving it markerless also keeps subclasses
    # honest -- they match on their own issuer markers via InvoiceProfile.matches().
    markers = ()

    def extract(self, doc_text: str) -> ExtractedInvoice:
        inv = ExtractedInvoice(profile=self.name)
        src = f"regex:{self.name}"

        inv.set("invoice_no", self._invoice_no(doc_text), src)
        inv.set("date", self._invoice_date(doc_text), src)
        inv.set("vendor", self.vendor_name or self._vendor(doc_text), src)
        inv.set("vendor_tax_id", self._seller_tax_id(doc_text), src)
        inv.set("total_amount", self._total(doc_text), src)
        inv.set(
            "tax_amount",
            self._amount(
                doc_text,
                r"toplam\s*vergi\s*tutari",
                # "Hesaplanan KDV(%20)", "Hesaplanan KDV GERCEK (%20.0)", "KDV(%20)"
                r"hesaplanan\s*kdv[^\d\n]*",
                r"\bkdv\s*\(\s*%?[\d.,]+\s*\)(?:\s*\([\d.,]+\))?",
                r"kdv\s*tutari",
                r"toplam\s*kdv",
                # Telecom bills state tax only in prose.
                r"devlete\s*iletilecek\s*vergiler\s*toplami",
            ),
            src,
        )
        inv.set(
            "net_amount",
            self._amount(
                doc_text,
                r"mal\s*/?\s*hizmet\s*toplam\s*tutari",
                r"mal\s*/?\s*hizmet\s*tutari",
                r"net\s*ara\s*toplam",
                r"ara\s*toplam",
                r"fatura\s*toplami",
                r"matrah",
            ),
            src,
        )
        inv.set("vat_rate", self._vat_rate(doc_text), src)
        inv.set("currency", self._currency(doc_text), src)
        inv.set(
            "payment_method",
            clean_ws(self._text(doc_text, r"odeme\s*sekli", r"odeme\s*turu", r"odeme\s*araci")),
            src,
        )
        # category is deliberately left unset: it is the one inferred field and is
        # filled in later by category.py.
        inv.field_sources["category"] = "missing"
        return inv

    # -- field-specific logic -------------------------------------------------

    def _invoice_no(self, text: str) -> str | None:
        # The trailing \b is essential: without it "Fatura Notu" matches "Fatura No"
        # and the invoice number comes back as "tu".
        value = self._text(
            text,
            r"fatura\s*(?:no|numarasi)\b",
            r"belge\s*no\b",
            r"\bett[nu]\b",
        )
        if not value:
            return None
        # The label line often carries trailing columns; keep the first token that
        # looks like an invoice serial rather than the whole tail.
        match = _INVOICE_NO.search(value.upper())
        if match:
            return match.group(1)
        return clean_ws(value.split()[0]) if value.split() else None

    def _invoice_date(self, text: str) -> str | None:
        """The date the invoice was issued.

        Both "Duzenleme" and "Duzenlenme" are in circulation. Other dates on the page
        are excluded by DATE_DISQUALIFIERS rather than by ordering, because they are
        not reliably ordered: telecom bills print "Bir Sonraki Fatura Tarihi" (the
        *next* bill's date) below the real one, which dated 20 of 23 invoices a month
        into the future.
        """
        for label in (
            r"fatura\s*tarihi",
            r"duzenlen?me\s*tarihi(?:\s*/?\s*zamani)?",
            r"belge\s*tarihi",
        ):
            for hit in iter_label_hits(text, label):
                # Only the text *preceding* the label disqualifies it ("Bir Sonraki
                # Fatura Tarihi"). Checking the whole line would also reject a valid
                # "Fatura Tarihi" that merely shares a header row with "Odeme Tarihi".
                if _is_other_date(hit.lines[hit.index][: hit.column]):
                    continue
                value = parse_tr_date(hit.own_column_tail)
                if value:
                    return value
                for candidate in hit.following():
                    value = parse_tr_date(candidate)
                    if value:
                        return value
        return None

    def _seller_tax_id(self, text: str) -> str | None:
        """The *seller's* tax number.

        The buyer's VKN/TCKN appears on these documents too, and an earlier version
        stored it -- both wrong and a privacy leak. Buyer-labelled lines are skipped
        outright, and otherwise the first tax id wins because the seller block is
        always printed above the recipient block.
        """
        for label in (
            r"vergi\s*kimlik\s*(?:no|numarasi)?",
            r"\bvkn\b",
            r"(?:\w+\s+)?kurumlar\s*v\.?\s*d\.?",
            r"vergi\s*dairesi",
            r"vergi\s*no",
            r"\bv\.?\s*d\.?\s*(?:no)?",
        ):
            for hit in iter_label_hits(text, label, by_column=True):
                # Check the whole line: "Musteri V.D. - VKN/TCKN : ..." carries the
                # marker *before* the label, so inspecting the tail alone misses it.
                if _is_third_party_context(hit.lines[hit.index]):
                    continue
                value = parse_tax_id(hit.own_column_tail)
                if value:
                    return value
                for candidate in hit.following():
                    if _is_third_party_context(candidate):
                        break
                    value = parse_tax_id(candidate)
                    if value:
                        return value
        return None

    def _total(self, text: str) -> float | None:
        # "Odenecek Tutar" is the authoritative payable figure; the others are
        # fallbacks in decreasing order of trustworthiness.
        return self._amount(
            text,
            r"odenecek\s*tutar",
            r"vergiler\s*dahil\s*toplam\s*tutar",
            r"genel\s*toplam",
            r"toplam\s*tutar",
            r"fatura\s*tutari",
        )

    def _vat_rate(self, text: str) -> float | None:
        rate = self._rate(text, r"kdv\s*orani")
        if rate is not None:
            return rate
        # Rates are frequently embedded in the label itself: "Hesaplanan KDV(%20)".
        match = re.search(r"kdv\s*\(?\s*%\s*(\d{1,2})", fold_tr(text))
        return float(match.group(1)) if match else None

    def _currency(self, text: str) -> str | None:
        value = self._text(text, r"para\s*birimi")
        if value:
            token = value.split()[0].upper().strip(":")
            if token in ("TRY", "TL"):
                return "TL"
            if token.isalpha() and len(token) == 3:
                return token
        folded = fold_tr(text)
        if "tl" in folded or "₺" in text or "try" in folded:
            return "TL"
        return None

    def _vendor(self, text: str) -> str | None:
        """Pick the issuer's legal name from the document header.

        Company names routinely wrap across two or three lines ("INT-EL INTERNATIONAL"
        / "ELEKTRONIK SANAYI VE" / "TICARET LIMITED SIRKETI"), and the legal-form token
        lands on the *last* of them -- so matching a single line yields a fragment.
        Preceding lines are pulled in while they still look like part of the name.
        """
        lines = [ln for ln in text.splitlines() if clean_ws(ln)]
        # Search the header first; the seller block is always above the buyer block.
        for index, line in enumerate(lines[:40]):
            folded = fold_tr(line)
            if not any(token in folded for token in _COMPANY_TOKENS):
                continue
            # A buyer that is itself a Ltd./A.Ş. carries the same legal-form tokens as a
            # seller name. Without this check, a "Sayın: ... A.Ş." line wins over the
            # real seller whenever the recipient is a company -- a common case, not an
            # edge case, for B2B purchase invoices.
            if _is_third_party_context(line):
                continue

            parts = [clean_ws(line)]
            for previous in reversed(lines[max(0, index - 2):index]):
                if not _is_name_continuation(previous):
                    break
                parts.insert(0, clean_ws(previous))

            candidate = clean_ws(" ".join(p for p in parts if p))
            if candidate and 5 < len(candidate) < 160:
                return candidate

        return self._sole_trader_name(lines)

    def _sole_trader_name(self, lines: list[str]) -> str | None:
        """The header line of a şahıs şirketi invoice, which has no legal-form token.

        A sole trader bills under a person's name -- "CANAN AYDIN E-TICARET" -- so the
        loop above finds nothing and the vendor came back None, printing as "(satıcı
        adı okunamadı)" in the ledger and in the assistant's answers. Many clients of a
        Turkish practice are şahıs, so this is a common case, not an edge one.

        Deliberately narrow: only the first few lines, only above the "e-Arşiv Fatura"
        heading that every one of these documents carries, and nothing that looks like a
        label or a value. Outside that shape it still returns None rather than guessing
        -- naming the wrong party as the seller would be worse than naming none.
        """
        for index, line in enumerate(lines[:6]):
            if "fatura" not in fold_tr(line):
                continue
            for previous in reversed(lines[:index]):
                candidate = clean_ws(previous)
                if not candidate or not _is_name_continuation(previous):
                    continue
                if _is_third_party_context(previous):
                    break
                if 5 < len(candidate) < 160:
                    return candidate
            break
        return None
