"""Phase 1 checkpoint: show what was extracted from each PDF, for manual review.

Writes nothing. Run this against a folder of invoices and read the output next to the
source documents before trusting anything downstream.

    python scripts/review_extraction.py "C:\\path\\to\\invoices" [--no-llm] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from malimusavir.extractors.base import FIELDS  # noqa: E402
from malimusavir.pipeline import extract_from_pdf, find_pdfs  # noqa: E402

MONEY_FIELDS = {"total_amount", "tax_amount", "net_amount"}


def _fmt(name: str, value: object) -> str:
    if value is None:
        return "-"
    if name in MONEY_FIELDS and isinstance(value, float):
        return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Review invoice extraction results.")
    parser.add_argument("folder", help="Folder (or single PDF) to inspect")
    parser.add_argument("--llm-category", action="store_true",
                        help="Also ask the model to infer category when no keyword "
                             "rule matches (slow, and low accuracy on this hardware)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a report")
    args = parser.parse_args()

    pdfs = list(find_pdfs(args.folder))
    if not pdfs:
        print(f"No PDFs found under {args.folder}")
        return 1

    results = []
    flagged = 0
    started = time.perf_counter()

    for index, path in enumerate(pdfs, start=1):
        try:
            inv = extract_from_pdf(path, use_llm=args.llm_category)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the review
            print(f"\n[{index}/{len(pdfs)}] {path.name}\n  ERROR: {type(exc).__name__}: {exc}")
            flagged += 1
            continue

        results.append(inv)
        if inv.needs_review:
            flagged += 1

        if args.json:
            continue

        status = "REVIEW" if inv.needs_review else "ok"
        print(f"\n[{index}/{len(pdfs)}] {path.name}  ->  {inv.profile}  [{status}]")
        for name in FIELDS:
            source = inv.field_sources.get(name, "missing")
            print(f"    {name:<15} {_fmt(name, getattr(inv, name)):<52} ({source})")
        if inv.needs_review:
            print(f"    ! {', '.join(inv.review_reasons)}")

    if args.json:
        print(json.dumps(
            [{**inv.to_row(), "source_path": inv.source_path, "profile": inv.profile,
              "review_reasons": inv.review_reasons} for inv in results],
            ensure_ascii=False, indent=2,
        ))
        return 0

    elapsed = time.perf_counter() - started
    print(f"\n{'-' * 72}")
    print(f"{len(pdfs)} file(s) in {elapsed:.1f}s -- {flagged} need review, "
          f"{len(results) - flagged} clean")

    by_profile: dict[str, int] = {}
    for inv in results:
        by_profile[inv.profile] = by_profile.get(inv.profile, 0) + 1
    if by_profile:
        print("profiles: " + ", ".join(f"{k}={v}" for k, v in sorted(by_profile.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
