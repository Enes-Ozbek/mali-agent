# Claude Design prompt — Mali Müşavir dashboard

Paste the sections below into claude.ai/design. The data schema matches the real
`invoices` table (`schema.sql`) and the real aggregate shapes (`stats.py`, `router.py`)
so a future API layer can be wired in without changing the frontend's data model.

---

## Prompt

Build a local-first dashboard for "Mali Müşavir" — a personal Turkish invoice
assistant. It ingests e-Arşiv PDF invoices, extracts structured data, and answers
questions about spend. Turkish locale throughout: dates as YYYY-MM-DD or DD.MM.YYYY,
amounts as `1.234,56 TL` (dot for thousands, comma for decimal). No login/auth — single
local user.

**Screens:**

1. **Dashboard** — summary cards (total spend, invoice count, date range, flagged
   count), a spend-by-category bar or donut chart, a spend-by-month line/bar chart, a
   "recurring payments" panel highlighting subscription-like vendors with their monthly
   cost, and a "largest invoices" list.
2. **Invoices** — a filterable/sortable table of all invoices (columns: date, vendor,
   category, total_amount, needs_review). Filters for category, vendor, date range, and
   a "needs review" toggle. Clicking a row opens invoice detail.
3. **Invoice detail** — all fields from the schema below, with `review_reasons` shown
   as warning chips if `needs_review` is true, and `extraction_profile` shown as a
   small provenance badge (e.g. "extracted via: turkcell").
4. **Ask** — a chat-style box where the user types a Turkish question and gets an
   answer. Show the answer text prominently; if the response includes `sources`, list
   them as small invoice cards below the answer (date, vendor, amount, relevance
   score). If it includes `rows`, render them as a small table matching the row shape.

**Visual tone:** clean, data-dense, financial-tool feel — not playful. Category badges
color-coded and reused consistently across all screens (same category = same color
everywhere).

Use the data shapes below as the mock data contract — every screen should render
correctly against arrays of these shapes with no additional transformation.

---

## Data schema

```typescript
interface Invoice {
  id: number;
  invoice_no: string;
  date: string | null;              // ISO YYYY-MM-DD
  vendor: string | null;
  vendor_tax_id: string | null;     // seller's tax id, 10-11 digits
  total_amount: number | null;      // TL
  tax_amount: number | null;
  net_amount: number | null;
  vat_rate: number | null;          // percentage, e.g. 20.0
  currency: string;                 // "TL" by default
  payment_method: string | null;
  category: Category;
  needs_review: boolean;
  review_reasons: string[];         // e.g. ["category:unresolved", "reconcile:..."]
  source_path: string | null;
  extraction_profile: string;       // "turkcell" | "superonline" | "dmarket" | "amazon" | "generic_earsiv"
}

type Category =
  | "telekom" | "enerji" | "abonelik" | "elektronik" | "ev" | "market"
  | "yeme-içme" | "giyim" | "ulaşım" | "sağlık" | "sigorta" | "kitap-medya"
  | "ofis" | "hizmet" | "diğer";

interface StatsSummary {
  invoices: number;
  total: number;
  tax: number;
  net: number;
  first_date: string | null;
  last_date: string | null;
  flagged: number;
  currencies: string[];
  mixed_currency: boolean;          // true means totals span >1 currency, show a warning
}

interface CategoryBreakdown {
  category: string;
  toplam: number;                   // total spend
  adet: number;                     // invoice count
  ortalama: number;                 // average per invoice
}

interface VendorBreakdown {
  vendor: string;
  toplam: number;
  adet: number;
  ortalama: number;
}

interface MonthBreakdown {
  ay: string;                       // "2026-04"
  toplam: number;
  adet: number;
}

interface RecurringVendor {
  vendor: string;
  adet: number;
  toplam: number;
  aylik_ortalama: number;           // average monthly cost
  ortalama_gun: number;             // median days between invoices, ~28-31 for monthly
}

interface AskResponse {
  text: string;                     // the answer, in Turkish
  intent: string;                   // "total" | "last" | "largest" | "by_category" | "semantic" | ...
  rows?: Record<string, unknown>[]; // supporting rows for aggregate answers; shape varies by intent
  sources?: {                       // present only for semantic (search) answers
    invoice_no: string;
    date: string;
    vendor: string;
    total_amount: number;
    score: number;                  // 0-1 relevance
  }[];
}
```

## Sample data

```json
{
  "summary": {
    "invoices": 23, "total": 9744.74, "tax": 2172.63, "net": 7572.11,
    "first_date": "2025-06-20", "last_date": "2026-05-13", "flagged": 0,
    "currencies": ["TL"], "mixed_currency": false
  },
  "invoices": [
    {
      "id": 1, "invoice_no": "01S2026000669264", "date": "2026-04-30",
      "vendor": "Turkcell Superonline İletişim Hizmetleri A.Ş.",
      "vendor_tax_id": "1750331214", "total_amount": 600.00, "tax_amount": 135.26,
      "net_amount": 464.74, "vat_rate": 20.0, "currency": "TL",
      "payment_method": null, "category": "telekom", "needs_review": false,
      "review_reasons": [], "source_path": null, "extraction_profile": "superonline"
    },
    {
      "id": 2, "invoice_no": "DRN2026000025322", "date": "2026-04-30",
      "vendor": "Örnek Elektronik Sanayi ve Ticaret Limited Şirketi",
      "vendor_tax_id": "4780059180", "total_amount": 991.92, "tax_amount": 165.32,
      "net_amount": 826.60, "vat_rate": 20.0, "currency": "TL",
      "payment_method": null, "category": "elektronik", "needs_review": false,
      "review_reasons": [], "source_path": null, "extraction_profile": "generic_earsiv"
    },
    {
      "id": 3, "invoice_no": "7245688201", "date": "2026-05-13",
      "vendor": "D-MARKET Elektronik Hizmetler ve Ticaret A.Ş.",
      "vendor_tax_id": "2650179910", "total_amount": 652.82, "tax_amount": 108.80,
      "net_amount": 544.02, "vat_rate": 20.0, "currency": "TL",
      "payment_method": "KREDI KARTI/BANKA KARTI", "category": "ev",
      "needs_review": false, "review_reasons": [], "source_path": null,
      "extraction_profile": "dmarket"
    }
  ],
  "byCategory": [
    { "category": "telekom", "toplam": 7657.20, "adet": 20, "ortalama": 382.86 },
    { "category": "elektronik", "toplam": 991.92, "adet": 1, "ortalama": 991.92 },
    { "category": "ev", "toplam": 652.82, "adet": 1, "ortalama": 652.82 },
    { "category": "hizmet", "toplam": 442.80, "adet": 1, "ortalama": 442.80 }
  ],
  "byMonth": [
    { "ay": "2026-03", "toplam": 869.50, "adet": 2 },
    { "ay": "2026-04", "toplam": 2411.62, "adet": 4 },
    { "ay": "2026-05", "toplam": 652.82, "adet": 1 }
  ],
  "recurring": [
    { "vendor": "Turkcell Superonline İletişim Hizmetleri A.Ş.", "adet": 9,
      "toplam": 4806.80, "aylik_ortalama": 534.09, "ortalama_gun": 31 },
    { "vendor": "Turkcell İletişim Hizmetleri A.Ş.", "adet": 11,
      "toplam": 2850.40, "aylik_ortalama": 259.13, "ortalama_gun": 31 }
  ],
  "askExample": {
    "text": "Toplam: 9.744,74 TL (23 fatura, 2025-06-20 - 2026-05-13).",
    "intent": "total",
    "rows": []
  }
}
```
