"""Aggregates. These run entirely offline -- no model is involved in arithmetic."""

from __future__ import annotations

import pandas as pd
import pytest

from malimusavir import stats


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = [
        # A monthly telecom subscription.
        *[
            {"invoice_no": f"T{i}", "date": f"2025-{month:02d}-20", "vendor": "Telekom A.S.",
             "vendor_tax_id": "1", "total_amount": 230.0, "tax_amount": 53.08,
             "net_amount": 176.92, "category": "telekom", "currency": "TL",
             "payment_method": None, "needs_review": 0}
            for i, month in enumerate(range(6, 12))
        ],
        # Two one-off purchases.
        {"invoice_no": "E1", "date": "2025-09-15", "vendor": "Elektronik Ltd.",
         "vendor_tax_id": "2", "total_amount": 1000.0, "tax_amount": 200.0,
         "net_amount": 800.0, "category": "elektronik", "currency": "TL",
         "payment_method": None, "needs_review": 0},
        {"invoice_no": "H1", "date": "2025-10-05", "vendor": "Danismanlik A.S.",
         "vendor_tax_id": "3", "total_amount": 500.0, "tax_amount": 100.0,
         "net_amount": 400.0, "category": "hizmet", "currency": "TL",
         "payment_method": None, "needs_review": 1},
    ]
    data = pd.DataFrame(rows)
    data["date"] = pd.to_datetime(data["date"])
    return data


def test_totals(frame):
    result = stats.totals(frame)
    assert result.invoices == 8
    assert result.total == pytest.approx(230.0 * 6 + 1000.0 + 500.0)
    assert result.first_date == "2025-06-20"
    assert result.last_date == "2025-11-20"
    assert result.flagged == 1


def test_category_totals_sum_to_the_grand_total(frame):
    by_cat = stats.by_category(frame)
    assert by_cat["toplam"].sum() == pytest.approx(stats.totals(frame).total)
    assert by_cat.iloc[0]["category"] == "telekom"


def test_by_vendor_is_ordered_by_spend(frame):
    by_vendor = stats.by_vendor(frame)
    assert by_vendor.iloc[0]["vendor"] == "Telekom A.S."
    assert by_vendor.iloc[0]["adet"] == 6


def test_vendors_group_by_tax_id_not_name(frame):
    """One seller whose name extracts differently must not split into two rows."""
    variant = frame.copy()
    variant.loc[variant["vendor"] == "Telekom A.S., ", "vendor"] = "Telekom A.S."
    variant.loc[variant.index[:2], "vendor"] = "TELEKOM ANONIM SIRKETI"  # same tax id
    grouped = stats.by_vendor(variant)
    telekom = grouped[grouped["adet"] == 6]
    assert len(telekom) == 1, "same tax id must collapse to one row"
    # The label shown is the most complete name seen.
    assert telekom.iloc[0]["vendor"] == "TELEKOM ANONIM SIRKETI"


def test_mixed_currency_is_flagged(frame):
    mixed = frame.copy()
    mixed.loc[mixed.index[0], "currency"] = "EUR"
    assert stats.totals(mixed).mixed_currency
    assert stats.totals(frame).mixed_currency is False


def test_by_month(frame):
    months = stats.by_month(frame)
    assert list(months["ay"]) == ["2025-06", "2025-07", "2025-08", "2025-09",
                                  "2025-10", "2025-11"]
    assert months[months["ay"] == "2025-09"]["toplam"].iloc[0] == pytest.approx(1230.0)


def test_largest(frame):
    assert stats.largest(frame, 1).iloc[0]["total_amount"] == 1000.0


def test_recurring_detects_the_monthly_vendor(frame):
    recurring = stats.recurring_vendors(frame)
    assert list(recurring["vendor"]) == ["Telekom A.S."]
    assert recurring.iloc[0]["adet"] == 6
    assert 28 <= recurring.iloc[0]["ortalama_gun"] <= 32


def test_recurring_vendor_months_matches_the_same_vendor_set(frame):
    """The sparkline data and the summary row must never disagree on who's recurring."""
    summary = stats.recurring_vendors(frame)
    monthly = stats.recurring_vendor_months(frame)
    assert set(monthly["vendor"]) == set(summary["vendor"])


def test_recurring_vendor_months_totals_roll_up_to_the_summary_row(frame):
    monthly = stats.recurring_vendor_months(frame)
    rolled = monthly.groupby("vendor")["toplam"].sum()
    summary = stats.recurring_vendors(frame).set_index("vendor")["toplam"]
    for vendor, total in rolled.items():
        assert total == pytest.approx(summary[vendor])


def test_recurring_vendor_months_empty_frame_is_handled():
    empty = pd.DataFrame(columns=["invoice_no", "date", "vendor", "vendor_tax_id",
                                  "total_amount", "tax_amount", "net_amount",
                                  "category", "currency", "payment_method",
                                  "needs_review"])
    assert stats.recurring_vendor_months(empty).empty


def test_one_off_excludes_recurring_vendors(frame):
    vendors = set(stats.one_off(frame)["vendor"])
    assert vendors == {"Elektronik Ltd.", "Danismanlik A.S."}


def test_date_range_filters_inclusively(frame):
    window = stats.date_range(frame, "2025-09-01", "2025-09-30")
    assert len(window) == 2
    assert stats.totals(window).total == pytest.approx(1230.0)


def test_empty_frame_is_handled_everywhere():
    empty = pd.DataFrame(columns=["invoice_no", "date", "vendor", "vendor_tax_id",
                                  "total_amount", "tax_amount", "net_amount",
                                  "category", "currency", "payment_method",
                                  "needs_review"])
    assert stats.totals(empty).invoices == 0
    assert stats.by_category(empty).empty
    assert stats.by_month(empty).empty
    assert stats.recurring_vendors(empty).empty
